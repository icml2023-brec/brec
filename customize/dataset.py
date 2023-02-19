import networkx as nx
import numpy as np
import random
import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils.convert import from_networkx, to_networkx

torch_geometric.seed_everything(2022)


part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "CFI": (160, 260),
    "Extension": (260, 360),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
}
PAIR_NUM = 400
NUM_RELABEL = 10
REPEAT_TEST_NUM = 10


def relabel(g6):
    pyg_graph = from_networkx(nx.from_graph6_bytes(g6))
    n = pyg_graph.num_nodes
    edge_index_relabel = pyg_graph.edge_index.clone().detach()
    index_mapping = dict(zip(list(range(n)), np.random.permutation(n)))
    for i in range(edge_index_relabel.shape[0]):
        for j in range(edge_index_relabel.shape[1]):
            edge_index_relabel[i, j] = index_mapping[edge_index_relabel[i, j].item()]
    edge_index_relabel = edge_index_relabel[
        :, torch.randperm(edge_index_relabel.shape[1])
    ]
    pyg_graph_relabel = torch_geometric.data.Data(
        edge_index=edge_index_relabel, num_nodes=n
    )
    g6_relabel = nx.to_graph6_bytes(
        to_networkx(pyg_graph_relabel, to_undirected=True), header=False
    ).strip()
    return g6_relabel


def generate_relabel(g6, num=0):
    if num == 0:
        num = NUM_RELABEL
    g6_list = [g6]
    g6_set = set(g6_list)
    for id in range(num - 1):
        g6_relabel = relabel(g6)
        g6_set.add(g6_relabel)
        while len(g6_set) == len(g6_list):
            g6_relabel = relabel(g6)
            g6_set.add(g6_relabel)
        g6_list.append(g6_relabel)
    return g6_list


class BRECDataset(InMemoryDataset):
    def __init__(
        self,
        root="Data",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.root = root
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [
            "basic.npy",
            "regular.npy",
            "str.npy",
            "cfi.npy",
            "extension.npy",
            "4vtx.npy",
            "dr.npy",
        ]

    @property
    def processed_file_names(self):
        return ["brec.npy"]

    def process(self):
        g6_list = []

        # Basic graphs: 0 - 60
        basic = np.load(self.raw_paths[0], allow_pickle=True)
        for g6 in basic:
            g6_relabel_list = generate_relabel(g6.encode(), REPEAT_TEST_NUM)
            for g6_relabel in g6_relabel_list:
                g6_list.extend(generate_relabel(g6_relabel))

        # Simple regular graphs: 60 - 110
        regular = np.load(self.raw_paths[1], allow_pickle=True)
        for g6_tuple in regular:
            g6_relabel_list = generate_relabel(g6_tuple[0], REPEAT_TEST_NUM)
            for g6_relabel in g6_relabel_list:
                g6_list.extend(generate_relabel(g6_relabel))

            g6_relabel_list = generate_relabel(g6_tuple[1], REPEAT_TEST_NUM)
            for g6_relabel in g6_relabel_list:
                g6_list.extend(generate_relabel(g6_relabel))

        # Strongly regular graphs: 110 - 160
        stronglyregular = np.load(self.raw_paths[2], allow_pickle=True)
        for g6 in stronglyregular:
            g6_relabel_list = generate_relabel(g6.encode(), REPEAT_TEST_NUM)
            for g6_relabel in g6_relabel_list:
                g6_list.extend(generate_relabel(g6_relabel))

        # CFI graphs: 160 - 260
        optimal = np.load(self.raw_paths[3], allow_pickle=True)
        for g6_tuple in optimal:
            g6_relabel_list = generate_relabel(g6_tuple[0], REPEAT_TEST_NUM)
            for g6_relabel in g6_relabel_list:
                g6_list.extend(generate_relabel(g6_relabel))

            g6_relabel_list = generate_relabel(g6_tuple[1], REPEAT_TEST_NUM)
            for g6_relabel in g6_relabel_list:
                g6_list.extend(generate_relabel(g6_relabel))

        # Extension graphs: 260 - 360
        special = np.load(self.raw_paths[4], allow_pickle=True)
        for g6_tuple in special:
            g6_relabel_list = generate_relabel(g6_tuple[0].encode(), REPEAT_TEST_NUM)
            for g6_relabel in g6_relabel_list:
                g6_list.extend(generate_relabel(g6_relabel))

            g6_relabel_list = generate_relabel(g6_tuple[1].encode(), REPEAT_TEST_NUM)
            for g6_relabel in g6_relabel_list:
                g6_list.extend(generate_relabel(g6_relabel))

        # 4-vertex condition graphs: 360 - 380
        vtx_4 = np.load(self.raw_paths[5], allow_pickle=True)
        for g6 in vtx_4:
            g6_relabel_list = generate_relabel(g6.encode(), REPEAT_TEST_NUM)
            for g6_relabel in g6_relabel_list:
                g6_list.extend(generate_relabel(g6_relabel))

        # Distance regular graphs: 380 - 400
        distance_regular = np.load(self.raw_paths[6], allow_pickle=True)
        for g6 in distance_regular:
            g6_relabel_list = generate_relabel(g6, REPEAT_TEST_NUM)
            for g6_relabel in g6_relabel_list:
                g6_list.extend(generate_relabel(g6_relabel))

        # print(len(g6_list))

        # basic graphs: 0 - 60
        for g6 in basic:
            g6_list.extend(generate_relabel(g6.encode()))

        # simple regular graphs: 60 - 110
        for g6_tuple in regular:
            g6_list.extend(generate_relabel(g6_tuple[0]))
            g6_list.extend(generate_relabel(g6_tuple[1]))

        # strongly regular graphs: 110 - 160
        for g6 in stronglyregular:
            g6_list.extend(generate_relabel(g6.encode()))

        # CFI graphs: 160 - 260
        for g6_tuple in optimal:
            g6_list.extend(generate_relabel(g6_tuple[0]))
            g6_list.extend(generate_relabel(g6_tuple[1]))

        # Extension graphs: 260 - 360
        for g6_tuple in special:
            g6_list.extend(generate_relabel(g6_tuple[0].encode()))
            g6_list.extend(generate_relabel(g6_tuple[1].encode()))

        # 4-vertex condition graphs: 360 - 380
        for g6 in vtx_4:
            g6_list.extend(generate_relabel(g6.encode()))

        # Distance regular graphs: 380 - 400
        for g6 in distance_regular:
            g6_list.extend(generate_relabel(g6))

        # Reliability check
        for i in range(0, PAIR_NUM * 2, 2):
            flag = random.randint(0, 1)
            # print((i + flag) * NUM_RELABEL * REPEAT_TEST_NUM)
            g6_relabel_list = random.sample(
                g6_list[(i + flag) * NUM_RELABEL * REPEAT_TEST_NUM : (i + 1 + flag) * NUM_RELABEL * REPEAT_TEST_NUM],
                NUM_RELABEL * 2,
            )
            g6_list.extend(g6_relabel_list)

        # print(len(g6_list))
        np.save(self.processed_paths[0], g6_list)


def main():
    dataset = BRECDataset()


if __name__ == "__main__":
    main()
