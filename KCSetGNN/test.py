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
from core.train import run
from core.model.setgnn import KCSetGNN
from core.model.gnn import GNN
from core.model.ppgn import PPGN
from torch_geometric.datasets import ZINC
from core.transform import KCSetWLSubgraphs


torch_geometric.seed_everything(2022)
NUM_RELABEL = 10
P_NORM = 2
OUTPUT_DIM = 8
EPSILON_MATRIX = 1e-7
EPSILON_CMP = 1e-5
SAMPLE_NUM = 270
REPEAT_TEST_NUM = 10

# part_dict: {graph generation type, range}
part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 110),
    "Optimal": (110, 170),
    "Extension": (170, 270),
    "Reliability": (270, 540),
    # "4-Vertex_Condition": (360, 380),
    # "Distance_Regular": (360, 380),
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
def get_dataset(cfg, name):
    time_start = time.process_time()

    # Do something

    transform_eval = KCSetWLSubgraphs(
        cfg.subgraph.kmax,
        cfg.subgraph.stack,
        cfg.subgraph.kmin,
        cfg.subgraph.num_components,
        zero_init=cfg.subgraph.zero_init,
    )
    dataset = BRECDataset(pre_transform=transform_eval, name=name)

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"dataset construction time cost: {time_cost}")

    return dataset


# Stage 3: model construction
# Here is for model construction.
def get_model(cfg):
    time_start = time.process_time()

    # Do something
    model = KCSetGNN(
        None,
        None,
        nhid=cfg.model.hidden_size,
        nout=OUTPUT_DIM,
        nlayer_intra=cfg.model.num_inners,
        nlayer_inter=cfg.model.num_layers,
        gnn_type=cfg.model.gnn_type,
        bgnn_type=cfg.model.bgnn_type,
        dropout=cfg.train.dropout,
        res=True,
        pools=cfg.model.pools,
        mlp_layers=2,
        num_bipartites=cfg.subgraph.kmax - 1 - cfg.subgraph.kmin
        if cfg.subgraph.stack is True
        else 1,
        half_step=cfg.model.half_step,
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

            T_square_threshold = T_square_threshold_list[id % SAMPLE_NUM]

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
    cfg.merge_from_file("train/config/BREC.yaml")
    cfg = update_cfg(cfg)

    # Command Line Arguments
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    NAME = f"k={cfg.subgraph.kmax}_c={cfg.subgraph.num_components}layer={cfg.model.num_layers}_inners={cfg.model.num_inners}_hidden={cfg.model.hidden_size}"
    DATASET_NAME = f'{cfg.subgraph.kmax}_{cfg.subgraph.num_components}'
    path = os.path.join("result", NAME)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "part_result"), exist_ok=True)

    LOG_NAME = os.path.join(path, "log.txt")
    logger.add(LOG_NAME)

    logger.info(cfg)

    pre_calculation()
    dataset = get_dataset(cfg, DATASET_NAME)
    model = get_model(cfg)
    evaluation(dataset, model, path, device)
