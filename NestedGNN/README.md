# NGNN & DE+NGNN Reproduction

## Requirements

Please refer to [Nested Graph Neural Networks](https://github.com/muhanzhang/NestedGNN)

## Usages

To reproduce best result on NGNN, run:

```bash
python test.py --h=1 --layers=6 --width=16 --node_label=no
```

To reproduce best result on DE+NGNN, run:

```bash
python test_v61.py --h=6 --layers=8 --width=64 --node_label=hop
```

