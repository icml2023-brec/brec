import os
from itertools import product
import argparse

kernels = ["spd", "gd"]
parser = argparse.ArgumentParser("arguments for training and testing")
parser.add_argument("--kernel", type=str, default="none")
parser.add_argument("--K", type=int, default=0)
parser.add_argument("--model_name", type=str, default="KPGIN")
parser.add_argument("--hidden_size", type=int, default=32)
parser.add_argument("--num_layer", type=int, default=4)


args = parser.parse_args()
if args.kernel != "none":
    kernels = [args.kernel]
ks = [1, 2, 3, 4]
if args.K != 0:
    ks = [args.K]
grid = product(kernels, ks)

for parameter in grid:
    kernel, k = parameter

    script_base = f"python test.py --K={k} --kernel={kernel} --model_name={args.model_name} --hidden_size={args.hidden_size} --num_layer={args.num_layer} "
    script = script_base
    print(f"running -- {script}")
    os.system(script)

    script = script_base + "--wo_peripheral_configuration"
    print(f"running -- {script}")
    os.system(script)

    script = script_base + "--wo_peripheral_edge"
    print(f"running -- {script}")
    os.system(script)

    script = script_base + "--wo_peripheral_configuration --wo_peripheral_edge"
    print(f"running -- {script}")
    os.system(script)
