from argparse import ArgumentParser
import glob
from itertools import product
import numpy as np
import os

import yaml

def main(num_boundaries, glob_str):
    file_list = glob.glob(f"./sepsis_experiments/sepsis_{glob_str}.yaml")
    file_list = [f for f in file_list if "alpha" not in f]
    alphas = np.linspace(0, 1, num_boundaries)
    for alpha, f in product(alphas, file_list):
        newf = f[:-5] + f"_alpha{alpha:.3f}.yaml"
        if os.path.isfile(newf):
            print("Skipping", newf)
            continue
        print("Processing file", f, "->", newf)
        with open(f, "r") as file:
            cfg = yaml.safe_load(file)
            cfg["name"] = cfg["name"] + f"_alpha{alpha:.3f}"
            test_fn0 = cfg["other_sim_kwargs"]["test_decision_fn0"]
            if "kwargs" not in test_fn0:
                test_fn0["kwargs"] = {}
            test_fn0["kwargs"]["weights"] = [alpha.item(), 1 - alpha.item()] 
            test_fn1 = cfg["other_sim_kwargs"]["test_decision_fn1"]
            if "kwargs" not in test_fn1:
                test_fn1["kwargs"] = {}
            test_fn1["kwargs"]["weights"] = [alpha.item(), 1 - alpha.item()]
        with open(newf, "w") as file:
            yaml.dump(cfg, file)

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--num_boundaries", required=True, type=int)
    psr.add_argument("--glob", default="*", type=str)
    args = psr.parse_args()
    main(args.num_boundaries, args.glob)
