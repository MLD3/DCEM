from argparse import ArgumentParser
import glob
from itertools import product
import os

import yaml

def main(translate, glob_str, directory="./final_experiment_settings"):
    file_list = glob.glob(os.path.join(directory, f"{glob_str}.yaml"))
    file_list = [f for f in file_list if "phase" not in f]
    print("File list:", file_list)
    
    for deg, f in product(translate, file_list):
        newf = f[:-5] + f"_phase{deg}.yaml"
        if os.path.isfile(newf):
            print("Skipping", newf)
            continue
        print("Processing file", f, "->", newf)
        with open(f, "r") as file:
            cfg = yaml.safe_load(file)
            cfg["name"] = cfg["name"] + f"_phase{deg}"
            cfg["other_sim_kwargs"]["label_decision_fn0"]["kwargs"]["translate"] = deg % 360
            cfg["other_sim_kwargs"]["label_decision_fn1"]["kwargs"]["translate"] = deg % 360
        with open(newf, "w") as file:
            yaml.dump(cfg, file)

if __name__ == '__main__':
    psr = ArgumentParser()
    psr.add_argument("--translate", required=True, type=int, nargs='+')
    psr.add_argument("--glob", type=str, default="full_*")
    psr.add_argument("--dir", type=str, default="./final_experiment_settings")
    args = psr.parse_args()
    main(args.translate, args.glob, args.dir)
