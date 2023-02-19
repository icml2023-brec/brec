# This program is the pipeline for testing expressiveness.
# It includes 4 stages:
#   1. pre-calculation;
#   2. dataset construction;
#   3. model construction;
#   4. evaluation


import numpy as np
import torch
import torch_geometric
import torch_geometric.loader
from loguru import logger
import time
from BRECDataset import BRECDataset
from tqdm import tqdm
import os
import math

import random
import torch.nn as nn
import train_utils
from json import dumps
import argparse
import torch_geometric.transforms as T
import torch.nn.functional as F
from torch_geometric.nn import DataParallel
from data_utils import (
    extract_multi_hop_neighbors,
    PyG_collate,
    resistance_distance,
    post_transform,
)
from layers.input_encoder import EmbeddingEncoder
from layers.layer_utils import make_gnn_layer
from models.model_utils import make_GNN
from models.GraphClassification import GraphClassification


NUM_RELABEL = 10
P_NORM = 2
OUTPUT_DIM = 8
EPSILON = 1e-6
EPSILON_MATRIX = 1e-7
EPSILON_CMP = 1e-7
SAMPLE_NUM = 400
REPEAT_TEST_NUM = 10

# part_dict: {graph generation type, range}
part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "CFI": (160, 260),
    "Special": (260, 360),
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
def get_dataset(args, dataset_name):
    time_start = time.process_time()

    # Do something
    def pre_transform(g):
        return extract_multi_hop_neighbors(
            g,
            args.K,
            args.max_pe_num,
            args.max_hop_num,
            args.max_edge_type,
            args.max_edge_count,
            args.max_distance_count,
            args.kernel,
        )

    transform = post_transform(args.wo_path_encoding, args.wo_edge_feature)
    dataset = BRECDataset(
        name=dataset_name, pre_transform=pre_transform, transform=transform
    )

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"dataset construction time cost: {time_cost}")

    return dataset


# Stage 3: model construction
# Here is for model construction.
def get_model(args):
    time_start = time.process_time()

    # Do something
    layer = make_gnn_layer(args)
    init_emb = EmbeddingEncoder(args.input_size, args.hidden_size)
    GNNModel = make_GNN(args)
    gnn = GNNModel(
        num_layer=args.num_layer,
        gnn_layer=layer,
        JK=args.JK,
        norm_type=args.norm_type,
        init_emb=init_emb,
        residual=args.residual,
        virtual_node=args.virtual_node,
        use_rd=args.use_rd,
        num_hop1_edge=args.num_hop1_edge,
        max_edge_count=args.max_edge_count,
        max_hop_num=args.max_hop_num,
        max_distance_count=args.max_distance_count,
        wo_peripheral_edge=args.wo_peripheral_edge,
        wo_peripheral_configuration=args.wo_peripheral_configuration,
        drop_prob=args.drop_prob,
    )

    model = GraphClassification(
        embedding_model=gnn,
        pooling_method=args.pooling_method,
        output_size=args.output_size,
    ).to(args.device)

    model.reset_parameters()
    if args.parallel:
        model = DataParallel(model, args.gpu_ids)

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


def main():
    parser = argparse.ArgumentParser("arguments for training and testing")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./save",
        help="Base directory for saving information.",
    )
    parser.add_argument(
        "--seed", type=int, default=2022, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--dataset_name", type=str, default="wl", help="name of dataset"
    )
    parser.add_argument(
        "--drop_prob",
        type=float,
        default=0.0,
        help="Probability of zeroing an activation in dropout layers.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size per GPU. Scales automatically when \
                            multiple GPUs are available.",
    )
    parser.add_argument("--num_workers", type=int, default=0, help="number of worker.")
    parser.add_argument(
        "--load_path",
        type=str,
        default=None,
        help="Path to load as a model checkpoint.",
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--l2_wd", type=float, default=3e-6, help="L2 weight decay.")
    parser.add_argument(
        "--kernel",
        type=str,
        default="spd",
        choices=("gd", "spd"),
        help="the kernel used for K-hop computation",
    )
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs.")
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=5.0,
        help="Maximum gradient norm for gradient clipping.",
    )
    parser.add_argument(
        "--hidden_size", type=int, default=32, help="hidden size of the model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="KPGIN",
        choices=("KPGCN", "KPGIN", "KPGraphSAGE", "KPGINPlus"),
        help="Base GNN model",
    )
    parser.add_argument("--K", type=int, default=4, help="number of hop to consider")
    parser.add_argument(
        "--max_pe_num",
        type=int,
        default=1000,
        help="Maximum number of path encoding. Must be equal to or greater than 1",
    )
    parser.add_argument(
        "--max_edge_type",
        type=int,
        default=1,
        help="Maximum number of type of edge to consider in peripheral edge information",
    )
    parser.add_argument(
        "--max_edge_count",
        type=int,
        default=1000,
        help="Maximum count per edge type in peripheral edge information",
    )
    parser.add_argument(
        "--max_hop_num",
        type=int,
        default=4,
        help="Maximum number of hop to consider in peripheral configuration information",
    )
    parser.add_argument(
        "--max_distance_count",
        type=int,
        default=1000,
        help="Maximum count per hop in peripheral configuration information",
    )
    parser.add_argument(
        "--wo_peripheral_edge",
        action="store_true",
        help="remove peripheral edge information from model",
    )
    parser.add_argument(
        "--wo_peripheral_configuration",
        action="store_true",
        help="remove peripheral node configuration from model",
    )
    parser.add_argument(
        "--wo_path_encoding",
        action="store_true",
        help="remove path encoding from model",
    )
    parser.add_argument(
        "--wo_edge_feature", action="store_true", help="remove edge feature from model"
    )
    parser.add_argument(
        "--num_hop1_edge", type=int, default=1, help="Number of edge type in hop 1"
    )
    parser.add_argument(
        "--num_layer", type=int, default=4, help="Number of layer for feature encoder"
    )
    parser.add_argument(
        "--JK",
        type=str,
        default="last",
        choices=("sum", "max", "mean", "attention", "last"),
        help="Jumping knowledge method",
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        help="Whether to use residual connection between each layer",
    )
    parser.add_argument(
        "--use_rd",
        action="store_true",
        help="Whether to add resistance distance feature to model",
    )
    parser.add_argument(
        "--virtual_node",
        action="store_true",
        help="Whether add virtual node information in each layer",
    )
    parser.add_argument(
        "--eps", type=float, default=0.0, help="Initital epsilon in GIN"
    )
    parser.add_argument(
        "--train_eps", action="store_true", help="Whether the epsilon is trainable"
    )
    parser.add_argument(
        "--combine",
        type=str,
        default="geometric",
        choices=("attention", "geometric"),
        help="Jumping knowledge method",
    )
    parser.add_argument(
        "--pooling_method",
        type=str,
        default="sum",
        choices=("mean", "sum", "attention"),
        help="pooling method in graph classification",
    )
    parser.add_argument(
        "--norm_type",
        type=str,
        default="Batch",
        choices=("Batch", "Layer", "Instance", "GraphSize", "Pair"),
        help="normalization method in model",
    )
    parser.add_argument(
        "--aggr",
        type=str,
        default="add",
        help="aggregation method in GNN layer, only works in GraphSAGE",
    )
    parser.add_argument(
        "--split", type=int, default=10, help="number of fold in cross validation"
    )
    parser.add_argument(
        "--factor",
        type=float,
        default=0.5,
        help="factor in the ReduceLROnPlateau learning rate scheduler",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="patience in the ReduceLROnPlateau learning rate scheduler",
    )

    args = parser.parse_args()
    if args.wo_path_encoding:
        args.num_hopk_edge = 1
    else:
        args.num_hopk_edge = args.max_pe_num
    torch_geometric.seed_everything(args.seed)
    

    args.name = (
        args.model_name
        + "_"
        + args.kernel
        + "_"
        + str(args.K)
        + "_"
        + str(args.wo_peripheral_edge)
        + "_"
        + str(args.wo_peripheral_configuration)
        + "_"
        + str(args.wo_path_encoding)
        + "_"
        + str(args.wo_edge_feature)
    )

    NAME = args.name
    PATH = os.path.join("result_v61", NAME)
    DATASET_NAME = str(args.K)
    os.makedirs(PATH, exist_ok=True)
    os.makedirs(os.path.join(PATH, "part_result"), exist_ok=True)
    LOG_NAME = os.path.join(PATH, "log.txt")
    logger.add(LOG_NAME)

    args.device, args.gpu_ids = train_utils.get_available_devices()
    args.parallel = False
    args.batch_size = 1
    args.input_size = 2
    args.output_size = OUTPUT_DIM

    logger.info(args)
    pre_calculation()
    dataset = get_dataset(args, dataset_name=DATASET_NAME)
    model = get_model(args)
    evaluation(dataset=dataset, model=model, device=args.device, path=PATH)


if __name__ == "__main__":
    main()
