# This program is the pipeline for testing expressiveness.
# It includes 4 stages:
#   1. pre-calculation;
#   2. dataset construction;
#   3. model construction;
#   4. evaluation

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from loguru import logger
import time
from BRECDataset import BRECDataset
from tqdm import tqdm
import os
from utils import (
    get_model,
)
from data import (
    policy2transform,
)


torch.set_num_threads(1)


NUM_RELABEL = 10
P_NORM = 2
OUTPUT_DIM = 8
EPSILON_MATRIX = 1e-6
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


def cov(X):
    D = X.shape[-1]
    mean = torch.mean(X, dim=-1).unsqueeze(-1)
    X = X - mean
    return 1 / (D - 1) * X @ X.transpose(-1, -2)


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
def get_dataset(args):
    time_start = time.process_time()

    # Do something
    if args.policy in ["ego_nets", "ego_nets_plus", "nested"]:
        name = os.path.join(args.policy, str(args.num_hops))
    else:
        name = "no_param"

    dataset = BRECDataset(
        name=name,
        pre_transform=policy2transform(policy=args.policy, num_hops=args.num_hops),
    )

    one_loader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        follow_batch=["subgraph_idx"],
    )

    # return one_loader, (in_dim, out_dim)

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"dataset construction time cost: {time_cost}")

    return one_loader


# Stage 3: model construction
# Here is for model construction.
def get_model_test(args, device):
    time_start = time.process_time()

    # Do something
    model = get_model(args, in_dim=1, out_dim=OUTPUT_DIM, device=device)

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"model construction time cost: {time_cost}")
    return model


# Stage 4: evaluation
# Here is for evaluation.
@torch.no_grad()
def evaluation(dataset_loader, model, path, device):
    time_start = time.process_time()

    # Do something
    loader = iter(dataset_loader)

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
            S = cov(D)
            # S = torch.from_numpy(np.cov(D.cpu().numpy())).to(torch.float).to(device)
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
            S = cov(D)
            # S = torch.from_numpy(np.cov(D.cpu().numpy())).to(torch.float).to(device)
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


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(
        description="GNN baselines with Pytorch Geometrics"
    )
    parser.add_argument(
        "--device", type=int, default=2, help="which gpu to use if any (default: 2)"
    )
    parser.add_argument(
        "--gnn_type",
        type=str,
        help="Type of convolution {gin, originalgin, zincgin, graphconv}",
    )
    parser.add_argument(
        "--random_ratio",
        type=float,
        default=0.0,
        help="Number of random features, > 0 only for RNI",
    )
    parser.add_argument("--model", type=str, help="Type of model {deepsets, dss, gnn}")
    parser.add_argument(
        "--drop_ratio", type=float, default=0.5, help="dropout ratio (default: 0.5)"
    )
    parser.add_argument(
        "--num_layer",
        type=int,
        default=5,
        help="number of GNN message passing layers (default: 5)",
    )
    parser.add_argument(
        "--channels",
        type=str,
        default="64-64",
        help='String with dimension of each DS layer, separated by "-"'
        "(considered only if args.model is deepsets)",
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=32,
        help="dimensionality of hidden units in GNNs (default: 32)",
    )
    parser.add_argument(
        "--jk",
        type=str,
        default="last",
        help="JK strategy, either last or concat (default: last)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="input batch size for training (default: 1)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="learning rate for training (default: 0.01)",
    )
    parser.add_argument(
        "--decay_rate",
        type=float,
        default=0.5,
        help="decay rate for training (default: 0.5)",
    )
    parser.add_argument(
        "--decay_step",
        type=int,
        default=50,
        help="decay step for training (default: 50)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="number of workers (default: 0)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ogbg-molhiv",
        help="dataset name (default: ogbg-molhiv)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/",
        help="directory where to store the data (default: dataset/)",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="edge_deleted",
        help="Subgraph selection policy in {edge_deleted, node_deleted, ego_nets}"
        " (default: edge_deleted)",
    )
    parser.add_argument(
        "--num_hops",
        type=int,
        default=2,  # FIXME in configs
        help="Depth of the ego net if policy is ego_nets (default: 2)",
    )
    parser.add_argument(
        "--seed", type=int, default=2022, help="random seed (default: 2022)"
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of subsampled subgraphs (1.0 means full bag aka no sampling)",
    )
    parser.add_argument(
        "--patience", type=int, default=20, help="patience (default: 20)"
    )
    parser.add_argument(
        "--task_idx",
        type=int,
        default=-1,
        help="Task idx for Counting substracture task",
    )
    parser.add_argument(
        "--use_transpose",
        type=str2bool,
        default=False,
        help="Whether to use transpose in SUN",
    )
    parser.add_argument(
        "--use_residual",
        type=str2bool,
        default=False,
        help="Whether to use residual in SUN",
    )
    parser.add_argument(
        "--use_cosine",
        type=str2bool,
        default=False,
        help="Whether to use cosine in SGD",
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="Optimizer, default Adam"
    )
    parser.add_argument(
        "--asam_rho", type=float, default=0.5, help="Rho parameter for asam."
    )
    parser.add_argument("--test", action="store_true", help="quick test")

    parser.add_argument(
        "--filename", type=str, default="", help="filename to output result (default: )"
    )
    parser.add_argument(
        "--add_bn", type=str2bool, default=True, help="Whether to use batchnorm in SUN"
    )
    parser.add_argument(
        "--use_readout",
        type=str2bool,
        default=True,
        help="Whether to use subgraph readout in SUN",
    )
    parser.add_argument(
        "--use_mlp",
        type=str2bool,
        default=True,
        help="Whether to use mlps (instead of linears) in SUN",
    )
    parser.add_argument(
        "--subgraph_readout",
        type=str,
        default="sum",
        help="Subgraph readout, default sum",
    )

    args = parser.parse_args()
    # set seed
    torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    LOG_NAME = f"{args.model}-{args.num_hops}_{args.num_layer}_{args.emb_dim}_{args.gnn_type}_{args.policy}"
    # LOG_PATH = get_save_dir(LOG_NAME)

    args.channels = list(map(int, args.channels.split("-")))
    if args.channels[0] == 0:
        # Used to get NestedGNN from DS
        args.channels = []
    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    print(device)

    path = os.path.join("result", LOG_NAME)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "part_result"), exist_ok=True)

    LOG_NAME = os.path.join(path, "log.txt")
    logger.add(LOG_NAME)

    logger.info(args)

    pre_calculation()
    dataset_loader = get_dataset(args)
    model = get_model_test(args, device)
    evaluation(dataset_loader, model, path, device)


if __name__ == "__main__":
    main()
