from argparse import ArgumentParser
from collections import defaultdict
import glob
from multiprocessing import Pool
import os

from checksumdir import dirhash
from tqdm.auto import tqdm

from path_utils import *


"""
    This script is to help double-check that versions of results on different machines match.
"""

ORIGINAL_DIRS = ["sweep_20230620_v2_server-4", "sweep_20230620_v2_server-6", "sweep_20230620_v2_server-7"]

def get_args():
    psr = ArgumentParser()
    psr.add_argument("--search-for", type=str, default=os.path.join(BASE_PATH, MASTER_DIR))
    psr.add_argument("--search-in", type=str, nargs='+', default=[os.path.join(BASE_PATH, d) for d in ORIGINAL_DIRS])
    psr.add_argument("--glob", type=str, default="_*_".join(EXPERIMENTAL_PARAMS) + "_*")
    psr.add_argument("--limit", type=int, default=None)
    psr.add_argument("--n-cpus", type=int, default=96)
    return psr.parse_args()

def get_glob(search, pattern):
    return glob.glob("/".join([search, pattern, "*Model*"]))

def get_checksums_for_subdirectory(model_dir): 
    dir_hash = dirhash(model_dir)
    return dir_hash, model_dir

def collect_checksums(search_in, pattern, n_cpus=96, limit=None):
    dirs = []
    for search_dir in search_in:
        dirs += get_glob(search_dir, pattern)
    if limit is not None:
        dirs = dirs[:limit]
    
    with Pool(n_cpus) as p:
        checksum_tuples = list(tqdm(p.imap(get_checksums_for_subdirectory, dirs), total=len(dirs)))
    checksum_dict = dict(checksum_tuples)
    return checksum_dict

def get_matches_for_subdirectory(model_dir):
    dir_hash = dirhash(model_dir)
    if dir_hash in checksum_dict:
        sweep_name = get_sweep_name_from_dir(checksum_dict[dir_hash])
    else:
        sweep_name = None
    return get_model_name_from_dir(model_dir), sweep_name

def _pass_checksum_dict(checksum_dict_):
    global checksum_dict
    checksum_dict = checksum_dict_

def check_hashes(search_for, pattern, checksum_dict, n_cpus=96):
    dirs = get_glob(search_for, pattern)
    with Pool(n_cpus, initializer=_pass_checksum_dict, initargs=(checksum_dict,)) as p:
        matches = list(tqdm(p.imap(get_matches_for_subdirectory, dirs), total=len(dirs)))
    # matches is a list of [(model_name, dir)] tuples - filter out duplicates
    matches = list(set(matches))

    match_dict = {}
    for k, v in matches:
        match_dict.setdefault(k, []).append(v)
    return match_dict

if __name__ == "__main__":
    args = get_args()
    checksum_dict = collect_checksums(args.search_in, args.glob, limit=args.limit, n_cpus=args.n_cpus)
    check_models = check_hashes(args.search_for, args.glob, checksum_dict)
    print(check_models)
