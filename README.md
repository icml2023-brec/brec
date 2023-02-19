# Towards Better Evaluation of GNN Expressiveness with BREC Dataset

## About

This repository is the official implementation of the following paper: https://openreview.net/forum?id=PgiDLs5ehI

**BREC** (Basic, Regular, Extension, CFI) is a new expressiveness dataset, including a total of 400 pairs of non-isomorphic graphs carefully selected from four major categories, with difficulty up to 5-WL. Since the graphs are organized pair-wised that different from conventional process, we propose a new evaluation method **RAPC** (Reliable and Adaptive Paired Comparisons) to measure pure expressiveness without training procedure.

## Usages

### Evaluate GNNs on BREC Dataset

We first introduce how to evaluate **your own GNNs** on BREC Dataset. Only the **base** directory is required.

#### Requirements

Tested combination: Python 3.8.13 + [PyTorch 1.12.0](https://pytorch.org/get-started/previous-versions/) + [PyTorch_Geometric 2.0.4](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

Other required python libraries include: numpy, networkx, loguru etc.

#### File Structure

```bash
├──	Data				# BREC Dataset file
	└──	raw
		├──	brec.npy	# unprocessed BREC Dataset in graph6 format
		└──	brec_pair_visualize.npy	# unprocessed 400 pairs of graphs for visualization
├── BRECDataset.py		# BREC Dataset construction
└── test.py				# Evaluation framework
```

#### Evaluation Step

To test your own GNNs, implement **test.py** with your model and run (${configs} represents corresponding config usage):

```bash
python test.py ${configs}
```

test.py is the pipeline for evaluation, including four stages:

```bash
1. pre-calculation;

2. dataset construction;

3. model construction;

4. evaluation
```

**Pre-calculation** aims to organize off-line operations on graphs.

**Dataset construction** aims to process the dataset with specific operations. BRECDataset is implemented based on InMemoryDataset. It is recommended to use tranform and pre_transform to transform the graphs.

**Model construction** aims to construct the GNN. Fine-tuning can be added in this stage.

**Evaluation** implements RAPC. With model and dataset, it will produce the final results.

Suppose your own experiment is done by running python main.py. In practical usage, you can easily implement test.py with main.py. You can drop the training part in main.py and split the rest to corresponding stages in test.py.

#### Results Demonstration

The 400 pairs of graphs are from four categories: Basic, Regular, Extension, CFI. We further  4-vertex condition and distance regular graphs from Regular as a separate category. The "category-id_range" dictionary is as follows:

```python
  "Basic": (0, 60),
  "Regular": (60, 160),
  "CFI": (160, 260),
  "Extension": (260, 360),
  "4-Vertex_Condition": (360, 380),
  "Distance_Regular": (380, 400),
  "Reliability": (400, 800),
```

Where Reliability represents the Reliability Check results in RAPC, means the distinguish results of essentially same graphs, only around 0 accuracy is accepted.

The correctly distinguished pair id is stored in **result/results.npy**, you can refer to **brec_pair_visualize.npy** to check specific graph situation.

### Reproduce Other Results

For baseline results reproduction, please refer to respective directories:

| Baseline          | Directory                           |
| ----------------- | ----------------------------------- |
| NGNN              | NestedGNN                           |
| DS-GNN            | SUN                                 |
| DSS-GNN           | SUN                                 |
| SUN               | SUN                                 |
| PPGN              | ProvablyPowerfulGraphNetworks_torch |
| GNN-AK            | GNNAsKernel                         |
| DE+NGNN           | NestedGNN                           |
| KP-GNN            | KP-GNN                              |
| KC-SetGNN         | KCSetGNN                            |
| Non-GNN Baselines | Non-GNN                             |

To reduce memory costs, we do not include data file **brec.npy** for each baseline. Please reduplicate **brec.npy** to corresponding directory where necessary.

### Customize BREC Dataset

Some graphs in BREC may be too difficult for some models, like strongly regular graphs can not be distinguished by 3-WL. You can discard some graphs from BREC to reduce test time. In addition, the parameter $P,Q$ in RAPC can also be adjusted when customizing. Only the **customize** directory is required.

#### Requirements

Tested combination: Python 3.8.13 + [PyTorch 1.12.0](https://pytorch.org/get-started/previous-versions/) + [PyTorch_Geometric 2.0.4](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)

Other required python libraries include: numpy, networkx, etc.

#### File Structure

```bash
├──	Data					# Original grpah file
	└──	raw
		├── basic.npy		# Basic graphs
        ├── regular.npy		# Simple regular graphs
        ├── str.npy			# Strongly regular graphs
        ├── cfi.npy			# CFI graphs
        ├── extension.npy	# Extension graphs
        ├── 4vtx.npy		# 4-vertex condition graphs
        └── dr.npy			# Distance regular graphs
└── dataset.py				# Customized BREC Dataset construction
```

#### Step

Suppose you want to discard distance regular graphs from BREC. You need to delete dr.npy related codes. The total pair number and the "category-id_range" dictionary should also be adjusted.

"REPEAT_TEST_NUM" and "NUM" represent $P,Q$ in RAPC, which can be adjusted for a different RAPC check.

