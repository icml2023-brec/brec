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

import argparse
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.utils import degree
from torch_geometric.nn import GCNConv, GINConv, global_add_pool
import torch_geometric.transforms as T
from k_gnn import GraphConv, max_pool
from k_gnn import TwoMalkin, ConnectedThreeMalkin
from dataloader import DataLoader  # use a custom dataloader to handle subgraphs
from utils import create_subgraphs


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


class BasicGCN(torch.nn.Module):
    def __init__(self, num_layers, hidden):
        super(BasicGCN, self).__init__()
        self.conv1 = GCNConv(1, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = torch.nn.Linear(hidden, hidden)
        # self.lin2 = Linear(hidden, dataset.num_classes)
        self.lin2 = Linear(hidden, 1)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        # edge_index, batch = data.edge_index, data.batch
        # batch =
        edge_index = data.edge_index
        if "x" in data:
            x = data.x
        else:
            x = torch.ones([data.num_nodes, 1]).to(edge_index.device)
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_add_pool(x, batch=None)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
        # return F.log_softmax(x, dim=1)

    def __repr__(self):
        return self.__class__.__name__


class NestedGCN(torch.nn.Module):
    def __init__(self, num_layers, hidden):
        super(NestedGCN, self).__init__()
        # self.conv1 = GCNConv(dataset.num_features, hidden)
        self.conv1 = GCNConv(1, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = torch.nn.Linear(hidden, hidden)
        # self.lin2 = Linear(hidden, dataset.num_classes)
        self.lin2 = Linear(hidden, OUTPUT_DIM)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        # edge_index, batch = data.edge_index, data.batch
        edge_index = data.edge_index
        if "x" in data:
            x = data.x
        else:
            x = torch.ones([data.num_nodes, 1]).to(edge_index.device)
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)

        x = global_add_pool(x, data.node_to_subgraph)
        x = global_add_pool(x, data.subgraph_to_graph)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        # return x
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__


# NGNN model
# add one hot optiton for initial encoding
class NestedGIN(torch.nn.Module):
    def __init__(
        self, num_layers, hidden, one_hot=False, num_hop=1, with_training=False
    ):
        super(NestedGIN, self).__init__()
        self.one_hot = one_hot
        self.num_hop = num_hop + 1
        if self.one_hot:
            print("one_hot")
            initial_lin = Linear(self.num_hop, hidden)
        else:
            initial_lin = Linear(1, hidden)
        self.conv1 = GINConv(
            Sequential(
                initial_lin,
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
            ),
            train_eps=False,
        )
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                    ),
                    train_eps=False,
                )
            )
        self.lin1 = torch.nn.Linear(hidden, hidden)
        self.lin2 = Linear(hidden, OUTPUT_DIM)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        # edge_index, batch = data.edge_index, data.batch
        edge_index = data.edge_index
        if "x" in data:
            x = data.x
        elif "z" in data:
            if self.one_hot:
                x = F.one_hot(data.z, num_classes=self.num_hop).squeeze().float()
            else:
                x = data.z.float()
        else:
            x = torch.ones([data.num_nodes, 1]).to(edge_index.device)
        x = self.conv1(x, edge_index)

        for conv in self.convs:
            x = conv(x, edge_index)

        x = global_add_pool(x, data.node_to_subgraph)
        x = global_add_pool(x, data.subgraph_to_graph)

        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)


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
def get_dataset(name, pre_transform):
    time_start = time.process_time()

    # Do something
    dataset = BRECDataset(name=name, pre_transform=pre_transform)

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"dataset construction time cost: {time_cost}")

    return dataset


# Stage 3: model construction
# Here is for model construction.
def get_model(args, device):
    time_start = time.process_time()

    # Do something
    if args.model == "GIN":
        model = NestedGIN(
            args.layers,
            args.width,
            not (args.node_label == "no"),
            args.h,
        ).to(device)
    elif args.model == "GCN":
        model = NestedGCN(args.layers, args.width).to(device)
    elif args.model == "BasicGCN":
        model = BasicGCN(args.layers, args.width).to(device)
    else:
        raise NotImplementedError("model type not supported")
    model.reset_parameters()

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
    loader = iter(DataLoader(dataset))

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
    parser = argparse.ArgumentParser(description="Nested GNN for EXP/CEXP datasets")
    parser.add_argument("--model", type=str, default="GIN")  # Base GNN used, GIN or GCN
    parser.add_argument(
        "--h",
        type=int,
        default=3,
        help="largest height of rooted subgraphs to simulate",
    )
    parser.add_argument("--layers", type=int, default=8)  # Number of GNN layers
    parser.add_argument(
        "--width", type=int, default=32
    )  # Dimensionality of GNN embeddings
    parser.add_argument("--node_label", type=str, default="no")
    args = parser.parse_args()

    # Command Line Arguments
    LAYERS = args.layers
    WIDTH = args.width

    MODEL = f"Nested{args.model}-"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pre_transform = None
    if args.h is not None:

        def pre_transform(g):
            return create_subgraphs(
                g,
                args.h,
                node_label=args.node_label,
                use_rd=False,
                subgraph_pretransform=None,
            )

    NAME = f"{MODEL}h={args.h}_layer={LAYERS}_hidden={WIDTH}_{args.node_label}"
    DATASET_NAME = f"h={args.h}_{args.node_label}"
    path = os.path.join("result_v61", NAME)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "part_result"), exist_ok=True)

    LOG_NAME = os.path.join(path, "log.txt")
    logger.add(LOG_NAME)

    logger.info(args)

    pre_calculation()
    dataset = get_dataset(name=DATASET_NAME, pre_transform=pre_transform)
    model = get_model(args, device)
    evaluation(dataset, model, path, device)


if __name__ == "__main__":
    main()
