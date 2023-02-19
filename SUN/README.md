# DS-GNN & DSS-GNN & SUN Reproduction

## Requirements

Please refer to [SUN](https://github.com/beabevi/SUN)

## Usages

To reproduce best result on DS-GNN, run:

```bash
python test.py --policy=ego_nets_plus --num_hops=2 --model=deepsets --gnn_type=originalgin --dataset=BREC --emb_dim=16 --num_layer=4 --channels=16-16
```

To reproduce best result on DSS-GNN, run:

```bash
python test.py --policy=ego_nets_plus --num_hops=2 --model=dss --gnn_type=originalgin --dataset=BREC --emb_dim=16 --num_layer=4
```

To reproduce best result on SUN, run:

```bash
python test.py --policy=ego_nets_plus --num_hops=2 --model=sun --gnn_type=originalgin --dataset=BREC --emb_dim=16 --num_layer=4
```

