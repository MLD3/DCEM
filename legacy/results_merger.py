from argparse import ArgumentParser
from functools import partial
import glob
from multiprocessing import Pool
import os
import re

import pandas as pd
from tqdm.auto import tqdm

from path_utils import *

def get_args():
    psr = ArgumentParser()
    psr.add_argument("--dir", type=str, default=os.path.join(BASE_PATH, MASTER_DIR))
    psr.add_argument("--glob", type=str, default="**/*Model*/") #"_*_".join(EXPERIMENTAL_PARAMS) + "_*/*Model*/")
    psr.add_argument("--param-names", type=str, nargs='+', default=EXPERIMENTAL_PARAMS)
    psr.add_argument("--limit", type=int, default=None)
    psr.add_argument("--n-cpus", type=int, default=96)
    return psr.parse_args()

def find_csvs(search_dir, pattern):
    print("Searching in", search_dir, "using pattern:", pattern)
    return glob.glob(f"{search_dir}/{pattern}/results.csv"), glob.glob(f"{search_dir}/{pattern}/bootstrap_results.csv")

def load_results_csv(index_cols, fname):
    # order of keys: (*experimental_params, model, metric
    df = pd.read_csv(fname, index_col=index_cols)
    model_name = get_model_name_from_dir(os.path.dirname(fname))
    exp_tuple = re.findall("[\d\.]+", fname.split("/")[-3]) 
    return exp_tuple + [model_name], df

def load_all_results(all_files, bootstrap=False):
    index_cols = (0, 1) if bootstrap else 0
    with Pool(args.n_cpus) as p:
        results = list(tqdm(p.imap(partial(load_results_csv, index_cols), all_files), total=len(all_files)))

    all_keys, all_dfs = zip(*results)
    print("Merging all", "bootstrap" if bootstrap else "", "results...")
    keys = list(map(tuple, all_keys))
    names = tuple(args.param_names + ["model"])
    try:
        all_results = pd.concat(all_dfs, keys=keys, names=names)
    except ValueError as e:
        if len(keys[0]) != len(names):
            raise ValueError(f"Got keys and names with different lengths for pd.concat (e.g., {keys[0]} vs. {names}). Maybe pass args to --param_names based on file names found (e.g.,", all_files[0] + ")?")
        else:
            raise e
    return all_results


if __name__ == '__main__':
    args = get_args()
    print("Finding CSV files...")
    all_csvs, all_bootstrap_csvs = find_csvs(args.dir, args.glob)
    print("Found", len(all_csvs), "results")
    if len(all_csvs) != 0:
        print("Loading all results...")
        all_results = load_all_results(all_csvs)
        print("Loading all bootstrap results...")
        all_bootstrap_results = load_all_results(all_bootstrap_csvs, bootstrap=True)
        all_results.to_csv(os.path.join(args.dir, "results.csv"))
        all_bootstrap_results.to_csv(os.path.join(args.dir, "bootstrap_results.csv"))  

