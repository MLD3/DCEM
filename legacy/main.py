from argparse import ArgumentParser
import multiprocessing as mp
import os
import sys

import yaml

import experiments

LOGGING_DIR = "logs/"
DATASET_CHOICES = ["simulated", "sepsis"]

def get_args():
    psr = ArgumentParser()
    psr.add_argument("--config", type=str, required=True)
    psr.add_argument("--split-by-models", action='store_true')
    psr.add_argument("--overwrite", action="store_true")
    psr.add_argument("--dataset", type=str, choices=DATASET_CHOICES, default="simulated")
    return psr.parse_args()

def _init_logging(cfg, models, client_ids): # TODO: use logging package instead
    with client_ids.get_lock():
        process_seq = client_ids.value
        fname = os.path.join(LOGGING_DIR, "_".join([cfg["name"], models[process_seq].__class__.__name__,  str(os.getpid())]) + ".log")
        print("Logging for model", models[process_seq].__class__.__name__, "at", fname)
        sys.stdout = open(fname, "w")
        sys.stderr = open(fname.replace(".log", ".err"), "w")
        client_ids.value += 1

def run_experiment(cfg, models, dataset):
    if dataset == "simulated":
        experiment_mgr_class = experiments.MitigationExperimentManager
    elif dataset == "sepsis":
        experiment_mgr_class = experiments.PseudoSyntheticExperimentManager
    else:
        raise ValueError(f"Dataset must be one of {DATASET_CHOICES} but got {dataset}.")
    mgr = experiment_mgr_class(
            models,
            cfg["name"],
            cfg["experimental_variables"],
            cfg["other_sim_kwargs"],
            cfg["model_hparams"],
            bootstrap=cfg["bootstrap"],
            n_cpus=cfg["n_cpus"],
            n=cfg.get("n", None),
            overwrite=cfg["overwrite"],
        )()

def get_models(cfg):
    models = []
    for model_cls in cfg["models"]:
        hparams = cfg["model_hparams"]["shared"] | cfg["model_hparams"].get(model_cls, {})

        import torch # must import this after matplotlib (in experiments) to avoid GLIBCXX error
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = getattr(experiments, model_cls)(hparams, device=device)
        models.append(model)
        print("=== MODEL ADDED ===")
        print(model_cls + f" ({device})", "-" * 10, hparams, sep="\n")
    return models

def main(args):
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["overwrite"] = args.overwrite
    models = get_models(cfg)
    if args.split_by_models:
        # run each model 
        tasks = [(cfg, [m]) for m in models]
        ctx = mp.get_context('spawn')
        client_ids = ctx.Value('i', 0)
        with ctx.Pool(len(models), initializer=_init_logging, initargs=(cfg, models, client_ids)) as p:
            p.starmap(run_experiment, tasks)

    else:
        run_experiment(cfg, models, args.dataset)

if __name__ == '__main__':
    args = get_args()
    main(args)
