# This program is the pipeline for testing expressiveness.
# It includes 4 stages:
#   1. pre-calculation;
#   2. dataset construction;
#   3. model construction;
#   4. evaluation


import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric
import torch_geometric.loader
from loguru import logger
import time
from BRECDataset import BRECDataset
from tqdm import tqdm
import os
import math


from core.config import cfg, update_cfg
from core.train_helper import run
from core.model import GNNAsKernel
from core.transform import SubgraphsTransform

from core.data import PlanarSATPairsDataset, calculate_stats


torch_geometric.seed_everything(2022)
NUM_RELABEL = 10
P_NORM = 2
OUTPUT_DIM = 8
EPSILON_MATRIX = 1e-7
EPSILON_CMP = 1e-6
SAMPLE_NUM = 400
REPEAT_TEST_NUM = 10

# part_dict: {graph generation type, range}
part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "CFI": (160, 260),
    "Extension": (260, 360),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
    "Reliability": (400, 800),
}


# Stage 1: pre calculation
# Here is for some calculation without data. e.g. generating all the k-substructures
def pre_calculation(*args, **kwargs):
    time_start = time.process_time()

    # Do something

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"pre-calculation time cost: {time_cost}")


# Stage 2: dataset construction
# Here is for dataset construction, including data processing
def get_dataset(cfg):
    time_start = time.process_time()

    # Do something

    transform_eval = SubgraphsTransform(
        cfg.subgraph.hops,
        walk_length=cfg.subgraph.walk_length,
        p=cfg.subgraph.walk_p,
        q=cfg.subgraph.walk_q,
        repeat=cfg.subgraph.walk_repeat,
        sampling_mode=None,
        random_init=False,
    )
    dataset = BRECDataset(transform=transform_eval)

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"dataset construction time cost: {time_cost}")

    return dataset


# Stage 3: model construction
# Here is for model construction.
def get_model(cfg):
    time_start = time.process_time()

    # Do something
    model = GNNAsKernel(
        None,
        None,
        nhid=cfg.model.hidden_size,
        nout=OUTPUT_DIM,
        nlayer_outer=cfg.model.num_layers,
        nlayer_inner=cfg.model.mini_layers,
        gnn_types=[cfg.model.gnn_type],
        hop_dim=cfg.model.hops_dim,
        use_normal_gnn=cfg.model.use_normal_gnn,
        vn=cfg.model.virtual_node,
        pooling=cfg.model.pool,
        embs=cfg.model.embs,
        embs_combine_mode=cfg.model.embs_combine_mode,
        mlp_layers=cfg.model.mlp_layers,
        dropout=cfg.train.dropout,
        subsampling=True if cfg.sampling.mode is not None else False,
        online=cfg.subgraph.online,
    ).to(cfg.device)

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"model construction time cost: {time_cost}")
    return model


# Stage 4: evaluation
# Here is for evaluation.
@torch.no_grad()
def evaluation(dataset, model, path, device):
    time_start = time.process_time()

    # Do something
    loader = iter(torch_geometric.loader.DataLoader(dataset))

    cnt = 0
    correct_list = []
    S_epsilon = torch.diag(
        torch.full(size=(OUTPUT_DIM, 1), fill_value=EPSILON_MATRIX).reshape(-1)
    ).to(device)

    model.eval()
    T_square_threshold_list = []

    for id in tqdm(range(SAMPLE_NUM)):
        T_square_threshold = torch.tensor([0.0], dtype=torch.float).to(device)
        for i in range(REPEAT_TEST_NUM):
            pred_1 = []
            pred_2 = []
            for i in range(NUM_RELABEL):
                data = next(loader).to(device)
                pred = F.normalize(model(data))
                pred_1.append(pred)
            for i in range(NUM_RELABEL):
                data = next(loader).to(device)
                pred = F.normalize(model(data))
                pred_2.append(pred)

            X = torch.cat([x for x in pred_1], dim=0).T
            Y = torch.cat([x for x in pred_2], dim=0).T
            D = X - Y
            D_mean = torch.mean(D, dim=1).reshape(-1, 1)
            S = torch.cov(D)
            inv_S = torch.inverse(S + S_epsilon)
            T_square = torch.mm(torch.mm(D_mean.T, inv_S), D_mean)

            T_square_threshold = torch.max(T_square, T_square_threshold)

        T_square_threshold_list.append(T_square)

    for part_name, part_range in part_dict.items():
        logger.info(f"{part_name} part starting ---")

        cnt_part = 0
        correct_list_part = []
        start = time.process_time()

        for id in tqdm(range(part_range[0], part_range[1])):
            pred_1 = []
            pred_2 = []
            for i in range(NUM_RELABEL):
                data = next(loader).to(device)
                pred = F.normalize(model(data))
                pred_1.append(pred)
            for i in range(NUM_RELABEL):
                data = next(loader).to(device)
                pred = F.normalize(model(data))
                pred_2.append(pred)

            X = torch.cat([x for x in pred_1], dim=0).T
            Y = torch.cat([x for x in pred_2], dim=0).T
            D = X - Y
            D_mean = torch.mean(D, dim=1).reshape(-1, 1)
            S = torch.cov(D)
            inv_S = torch.inverse(S + S_epsilon)
            T_square = torch.mm(torch.mm(D_mean.T, inv_S), D_mean)

            T_square_threshold = T_square_threshold_list[id % 400]

            diff_check = False
            if T_square > T_square_threshold and not torch.isclose(
                T_square, T_square_threshold, atol=EPSILON_CMP
            ):
                diff_check = True

            if diff_check:
                if part_name != "Reliability":
                    cnt += 1
                    correct_list.append(id)
                else:
                    print(D)
                    print(T_square)
                    print(T_square_threshold)
                cnt_part += 1
                correct_list_part.append(id)

        end = time.process_time()
        time_cost_part = round(end - start, 2)

        logger.info(
            f"{part_name} part costs time {time_cost_part}; Correct in {cnt_part} / {part_range[1] - part_range[0]}"
        )
        np.save(os.path.join(path, "part_result", part_name), correct_list_part)

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"evaluation time cost: {time_cost}")

    Acc = round(cnt / 400, 2)
    logger.info(f"Correct in {cnt} / 400, Acc = {Acc}")
    np.save(os.path.join(path, "result"), correct_list)


if __name__ == "__main__":
    cfg.merge_from_file("train/configs/BREC.yaml")
    cfg = update_cfg(cfg)

    # Command Line Arguments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NAME = f"h={cfg.subgraph.hops}_layer={cfg.model.num_layers}_hidden={cfg.model.hidden_size}"
    path = os.path.join("result", NAME)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "part_result"), exist_ok=True)

    LOG_NAME = os.path.join(path, "log.txt")
    logger.add(LOG_NAME)

    logger.info(cfg)

    pre_calculation()
    dataset = get_dataset(cfg)
    model = get_model(cfg)
    evaluation(dataset, model, path, device)
