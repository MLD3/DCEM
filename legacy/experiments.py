from collections import Iterable
from functools import partial
import itertools
import os
import numbers
import pickle
import pprint
import socket
import subprocess
import sys
import time
import traceback
from typing import Any, Dict, List, Optional
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.svm import SVC
from sklearn.metrics import brier_score_loss
import torch
import torch.nn as nn
try:
    from torch.utils.tensorboard import SummaryWriter # when recovering past results and importing experiments.py, this causes issues
except ImportError as e:
    print(e)
from tqdm.auto import tqdm

from disparate_censorship.simulation import (
        TestingBiasSimulator,
        MIMICIIISepsisDataAccessor,
        find_simulation_with_parameters,
        find_simulated_test_settings_with_parameters,
        validate_simulation
    )

import boundaries
from losses import lq_loss, js_scaled_loss, weighted_loss_wrapped
import metrics
from nn_modules import SimpleMLP, PeerLossMLP, GroupPeerLossMLP, PUEstimator, DisparateCensorshipEstimator, SELFEstimator, DivideMixEstimator, ITECorrectedMLP
from plotting import preview_sim_data, plot_model_predictions, plot_boundaries
from utils import retry_if_oom

# These keywords should be overriden in YAML files if needed.
DEFAULT_KWARGS = {
    "n_dims": 2,
    "test_threshold_type": "soft",
    "test_threshold_group0": 0.35,
    "test_threshold_group1": 0.55, 
    "label_threshold_group0": 1.0,
    "label_threshold_group1": 1.0,
    "label_threshold_type": "soft",
    "threshold_hardness_0": 30.,
    "threshold_hardness_1": 30.,
    "label_hardness_0": 30.,
    "label_hardness_1": 30.,
    "sigma": 0.03,     
    "mu0": 0.35,
    "mu1": 0.55, 
}


RESULTS_DICT = {
    "server-3": "/data4/username/disparate_censorship_mitigation",
    "server-4": "/data9/username/disparate_censorship_mitigation",
    "server-7": "/data2/username/disparate_censorship_mitigation",
    "server-6": "/data1/username/disparate_censorship_mitigation",
}
SEPSIS_FNAME = "sepsis3_mimic_ros_replication.csv"
SEPSIS_DICT = {
    "server-3": "/data4/mimiciii/",
    "server-7": "/data2/username/"

}

HOSTNAME = socket.gethostname()
if HOSTNAME in SEPSIS_DICT:
    SEPSIS_PATH = os.path.join(SEPSIS_DICT[HOSTNAME], SEPSIS_FNAME)
else:
    SEPSIS_PATH = None # or, if you're only running this on one machine, set this manually

if SEPSIS_PATH is None or not os.path.isfile(SEPSIS_PATH):
    warnings.warn(f"Note that the file specified by `SEPSIS_PATH` ({SEPSIS_PATH}) does not exist. Check that your machine hostname is in `SEPSIS_DICT` and associated with an existing directory, or set `SEPSIS_PATH` in `experiments.py` manually. This is OK if you are running synthetic-data experiments only, but will throw an error for the sepsis task.") 


if HOSTNAME in RESULTS_DICT:
    EXPERIMENT_RESULTS_DIR = os.path.join(RESULTS_DICT[HOSTNAME])
else:
    EXPERIMENT_RESULTS_DIR = None # or, if you're only running this on one machine, set this manually
if EXPERIMENT_RESULTS_DIR is None:
    warnings.warn("Experiment results directory specified by `EXPERIMENT_RESULTS_DIR` ({EXPERIMENT_RESULTS_DIR}) has not been specified in `RESULTS_DICT`. Check that your machine hostname is in `RESULTS_DICT` and associated with your desired results path, or set `EXPERIMENT_RESULTS_DIR` in `experiments.py` manually.")

PROJECT_NAME = "mitigating_disparate_censorship"

def get_dummy_experiment_manager():
    return ExperimentManager({}, "", {})

class ExperimentManager(object):
    def __init__(self, sim_kwargs: Dict[str, Any], setting_name: str, model_hparams: Dict[str, Any], n_cpus: Optional[int] = 0):
        self.sim_kwargs = sim_kwargs
        self.setting_name = setting_name
        self.n_cpus = n_cpus
        self.model_hparams = model_hparams

    def get_fixed_experiment_kwargs(self, starter_kwargs):
        experiment_kwargs = starter_kwargs.copy()
        for k, v in self.sim_kwargs.items(): # apply other overrides
            if "decision_fn" in k:
                fn_object = getattr(boundaries, v["name"])
                if isinstance(fn_object(), boundaries.DecisionBoundary):
                    experiment_kwargs[k] = fn_object(**v.get("kwargs", {}))
                elif callable(fn_object):
                    if "partial_args" in v:
                        experiment_kwargs[k] = partial(fn_object, *v["partial_args"])
                    else:
                        experiment_kwargs[k] = fn_object
                else:
                    raise ValueError(f"{v} passed as argument for key {k} but {v['name']} is not a valid decision boundary")
            else:
                try:
                    converted_val = float(v)
                except ValueError:
                    converted_val = v
                experiment_kwargs[k] = converted_val
        return experiment_kwargs


    def get_simulation(self, n_dims: Optional[int] = 2, **starter_kwargs):
        experiment_kwargs = self.get_fixed_experiment_kwargs(starter_kwargs)
        if isinstance(experiment_kwargs["mu0"], numbers.Number):
            experiment_kwargs["mu0"] = np.ones(n_dims) * experiment_kwargs["mu0"]

        if isinstance(experiment_kwargs["mu1"], numbers.Number):
            experiment_kwargs["mu1"] = np.ones(n_dims) * experiment_kwargs["mu1"]
            
        if isinstance(experiment_kwargs["sigma"], numbers.Number):
            experiment_kwargs["sigma"] = np.eye(n_dims) * experiment_kwargs["sigma"]
        sim = find_simulation_with_parameters(n_feats=n_dims, verbose=False, **experiment_kwargs)
        validate_simulation(sim)

        data = sim.get_dataset(sizes=self.n)
        return sim, data

    def set_seed(self, seed: int):
        np.random.seed(seed)

    def save_simulation_snapshot(self, sim, varname, fname):
        simulation_path = os.path.join(self.get_experiment_dirname(), fname)
        os.makedirs(simulation_path, exist_ok=True)
        sim.save(simulation_path)


class MitigationExperimentManager(ExperimentManager):
    def __init__(self, models: List, setting_name: str, experiment_settings: Dict[str, List], sim_kwargs: Dict[str, Any], model_hparams: Dict[str, Any], n: Optional[int] = None, bootstrap: Optional[int] = None, n_cpus: Optional[int] = 0, overwrite: Optional[bool] = False):
        super().__init__(sim_kwargs, setting_name, model_hparams, n_cpus)
        self.models = models
        self.model_names = [model.__class__.__name__ for model in models]
        self.sim_kwargs = sim_kwargs
        self.setting_name = setting_name
        self.experiment_settings = experiment_settings
        self.bootstrap = bootstrap
        self.n = n
        self.n_cpus = n_cpus 
        self.overwrite = overwrite # setting_name.startswith("check") or setting_name.endswith("sanity") or overwrite


    def get_experiment_dirname(self):
        return os.path.join(EXPERIMENT_RESULTS_DIR, self.setting_name) # f"{self.setting_name}_{varname}")

    def train_test_one_model_for_value(self, model, sim, data, logger, varnames, vals): 
        experiment_path = os.path.join(self.get_experiment_dirname(), "_".join(self.varname_val_tuple(varnames, vals)))
        model.setup(sim, data, logger)
        results, bootstrap_results = model.fit_and_evaluate(bootstrap=self.bootstrap, n_cpus=self.n_cpus, skip_bootstrap=(self.bootstrap < 1))
        model.save_model_data(experiment_path)
        return results, bootstrap_results

    def train_test_all_models_for_value(self, varnames, vals, pbar):
        """
            varname is a tuple of simulation parameter variables; e.g.,
            ("test_threshold_group0", "test_threshold_group1")

            val is a tuple of the values they will take on; e.g.,
            (0.45, 0.55)
        """

        def _results_are_complete(dirname):
            if not os.path.isdir(dirname): return False
            filelist = set(["bootstrap_results.csv", "results.csv", "decision_boundary.pdf", "model.pkl"])
            files = set(os.listdir(dirname))
            missing = filelist - files
            return len(missing) == 0
         
        exp_dict = dict(zip(varnames, vals))
        starter_kwargs = DEFAULT_KWARGS | exp_dict 
        varstrs = [f"{k}={v}" for k, v in exp_dict.items()]
        exp_name = " ".join(varstrs)

        experiment_path = os.path.join(self.get_experiment_dirname(), "_".join(self.varname_val_tuple(varnames, vals)))
        all_results_for_val, all_bootstraps_for_val = [], []
        data = None
        skipped_flag = False
        directories = [os.path.join(experiment_path, f"{m.__class__.__name__}_model_info") for m in self.models]
        if all(map(_results_are_complete, directories)) and not self.overwrite:
            print("All experiment records for", experiment_path, "exist for all models. Skipping.")
            skipped_flag = True
            return all_results_for_val, all_bootstraps_for_val, data, skipped_flag

        writer = SummaryWriter(os.path.join(EXPERIMENT_RESULTS_DIR, self.get_experiment_dirname(), "tensorboard_logs", exp_name)) # structure: LOG_DIR/experiment_dirname/varstrs
        sim, data = self.get_simulation(**starter_kwargs)
        n_exps = len(self.model_names)

        print("Evaluating setting:", exp_name)
        for i, (model, expname) in enumerate(zip(self.models, self.model_names)):
            model_path = os.path.join(experiment_path, f"{model.__class__.__name__}_model_info")
            if os.path.isdir(model_path) and not self.overwrite:
                print("Experiment record at", model_path, "already exists. Skipping.")
                skipped_flag = True
                continue
            print()
            pbar.set_description("Fitting " + model.__class__.__name__ +  f"... ({i+1}/{len(self.models)}) ")
            pbar.refresh()
            try:
                results, bootstrap_results = self.train_test_one_model_for_value(model, sim, data, writer, varnames, vals)      
                all_results_for_val.append(results)
                experiment_path = os.path.join(self.get_experiment_dirname(), "_".join(self.varname_val_tuple(varnames, vals)), f"{model.__class__.__name__}_model_info")
                results.to_csv(os.path.join(experiment_path, "results.csv"))
                if bootstrap_results is not None:
                    all_bootstraps_for_val.append(bootstrap_results)
                    bootstrap_results.to_csv(os.path.join(experiment_path, "bootstrap_results.csv"))
            except Exception as e:
                print(traceback.format_exc())

            pbar.update()
        self.save_simulation_snapshot(sim, varnames, "_".join(self.varname_val_tuple(varnames, vals)))
        return all_results_for_val, all_bootstraps_for_val, data, skipped_flag

    def varname_val_tuple(self, varnames, vals):
        return tuple([f"{name}_{single_val}" for name, single_val in zip(varnames, vals)])

    def __call__(self, preview_saved_files: Optional[bool] = True):

        # need to create a list of tuple-pairs (n0, ... nn), (var0, ..., varn)
        all_val_tuples = list(itertools.product(*list(self.experiment_settings.values())))
        varnames = tuple(self.experiment_settings.keys())
        # for varnames, values in {tuple(varnames): all_val_tuples}.items():
        all_results, all_bootstraps = [], []
        master_data_cache = {}
        
        # for each set of thresholds
        pbar = tqdm(all_val_tuples, total=len(self.models) * len(all_val_tuples))
        for val_tuple in pbar:
            try:
                all_results_for_val, all_bootstraps_for_val, data_for_val, skipped_flag = self.train_test_all_models_for_value(varnames, val_tuple, pbar)               
                all_results_for_val = pd.concat(all_results_for_val, keys=self.model_names, names=["model"]) # don't raise issue if one particular parameter set failed for all models

                all_results.append(all_results_for_val)

                if len (all_bootstraps_for_val) >= 1:
                    all_bootstraps_for_val = pd.concat(all_bootstraps_for_val, keys=self.model_names, names=["model"])
                    all_bootstraps.append(all_bootstraps_for_val)
            except ValueError as e:
                if "concatenate" in str(e) and skipped_flag:
                    warnings.warn("Caught concatenation error when combining results. If you are restarting an experiment, this is OK if all models for a single parameter setting were already saved.")
                else:
                    print("Exception caught while training all models for parameter setting:", *self.varname_val_tuple(varnames, val_tuple))
                    print(traceback.format_exc())
                continue

            exp_key = self.varname_val_tuple(varnames, val_tuple)
            master_data_cache[exp_key] = data_for_val

        if len(all_results): 
            all_results = pd.concat(all_results, keys=all_val_tuples, names=varnames)
        if len(all_bootstraps):
            all_bootstraps = pd.concat(all_bootstraps, keys=all_val_tuples, names=varnames)
        self.save_results(varnames, master_data_cache, all_results, all_bootstraps, preview=preview_saved_files)

    def save_results(self, varname, data_cache, results, bootstraps, preview=True):
        """
        A schematic of the general folder structure:

        |- master_experiment_folder
           |- varname_val
              |- model_name_data/
                 |- model.pkl
                 |- aux_results/
                    |- aux_model.pkl
                    |- aux_model_results.csv
              |- simulation.pkl
           |- ...
           |- data_dict
           |- result_df
           |- bootstrap_df
        """ 
        parent_dir = self.get_experiment_dirname()
        os.makedirs(parent_dir, exist_ok=True)

        # save data file
        data_path = os.path.join(parent_dir, "data_dict.pkl")
        with open(data_path, "wb") as f:
            pickle.dump(data_cache, f) 

class PseudoSyntheticExperimentManager(MitigationExperimentManager):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n is not None:
            warnings.warn(f"Pseudo-synethic experiment manager received value n={self.n} but dataset size is fixed since we are using real data.")


    def get_simulation(self, n_dims: Optional[int] = 2, **starter_kwargs):
        experiment_kwargs = self.get_fixed_experiment_kwargs(starter_kwargs)
        # possible keywords: test_threhsold_group0, test_threshold_group1, test_decision_fn0, test_decision_fn1, test_threshold_type
        # threshold_hardness_0, threshold_hardness_1.
        dummy_mimic_data = MIMICIIISepsisDataAccessor(SEPSIS_PATH, label_col="sepsis-3")
        group_balance = dummy_mimic_data.get_group_balance()
        prevalence = dummy_mimic_data.get_prevalence()
        sim = find_simulated_test_settings_with_parameters(group_balance, prevalence, **experiment_kwargs)
        validate_simulation(sim)
        data = sim.get_dataset()
        return sim, data

    


class Model(object):
    def __init__(self, hparams, use_enc=False, use_nn=True, device="cuda:7"):
        self.model = None
        self.sim = None
        self.data = None
        self.aux_models = None
        self.aux_model_results = None
        self.aux_data_dict = None
        self.aux_labels = None
        self.results = None
        self.bootstrap_results = None
        self.use_enc = use_enc
        self.data_key = "X_enc" if use_enc else "X"
        self.use_nn = use_nn
        self.hparams = hparams
        self.device = device
        self.logger = None

    def setup(self, sim, data, logger):
        self.sim = sim
        self.data = data
        self.logger = logger
        self.get_model()
        self.setup_aux()

    def setup_aux(self):
        self.get_aux_models()
        self.fit_aux_models()
        self.get_aux_labels()
        self.evaluate_aux_models()

    def fit_and_evaluate(self, bootstrap=None, skip_bootstrap=False, n_cpus=0, evaluate_on="test"):
        if self.sim is None:
            raise RuntimeError("Simulator not initialized -- run self.setup.")
        if self.model is None:
            raise RuntimeError("Model not initialized -- run self.setup.")
        for k in self.data:
            if len(self.data[k]) == 0:
                warnings.warn(f"Split {k} has no data!")
        retry_if_oom(self.fit_aux_models)
        print("Fitting models...", end="")
        retry_if_oom(self.fit_model)
        results = retry_if_oom(self.evaluate_model, split=evaluate_on)

        bootstrap_results = None
        if not skip_bootstrap:
            print("\r")
            bootstrap_results = retry_if_oom(self.bootstrap, bootstrap, n_cpus=n_cpus)

        return results, bootstrap_results

    def get_model(self):
        if self.use_nn:
            self.model = SimpleMLP(
                    self.data["train"]["X"].shape[-1],
                    device=self.device, 
                    parent=self.__class__.__name__,
                    **self.hparams
                )
        else:
            self.model = SVC(probability=True)
        return self.model

    def get_aux_models(self):
        pass # for methods like PSM or recensoring -- should set Y_aux attribute of data dict

    def evaluate_aux_models(self):
        pass

    def get_aux_labels(self):
        pass # for methods like recensoring

    def fit_model(self):
        pass # override in subclass

    def fit_aux_models(self):
        pass

    def bootstrap(self, n, split="test", n_cpus=0, metric_names=["AUC", "xAUC", "ROCGap"]):
        if n_cpus > 0:
            ctx = torch.multiprocessing.get_context("spawn")
            with ctx.Pool(n_cpus, initializer=_initializer, initargs=(self.data_key, self.data[split], self.model, metric_names)) as p:
                all_results = list(tqdm(p.imap(_bootstrap_eval_in_subprocess, range(n)), leave=False, total=n, desc=f"Bootstrapping ({n_cpus} procs.)"))
                p.close()
                p.join()
        else:
            all_results = []
            for i in tqdm(range(n), desc="Bootstrapping"): # parallelize
                results = self.evaluate_model(split=split, shuffle_seed=i, metric_names=metric_names) # todo: make data shuffle-able
                all_results.append(results) 
        all_results = pd.concat(all_results, keys=range(n))
        self.bootstrap_results = all_results
        return self.bootstrap_results

    def evaluate_model(self, split="test", metric_names=["AUC", "xAUC", "ROCGap"], use_enc=True, pred_probs=None, shuffle_seed=None, verbose=True): 
        X, Y, group = self.data[split][self.data_key], self.data[split]["Y"], self.data[split]["A"]
        if shuffle_seed is not None:
            np.random.seed(shuffle_seed)
            indices = np.random.choice(np.arange(len(X)), len(X), replace=True)
            X, Y, group = X[indices], Y[indices], group[indices]
        if pred_probs is None:
            if self.use_nn:
                pred_probs = self.model.predict_proba(X, group)
            else:
                pred_probs = self.model.predict_proba(X)
            self.data[split]["probs"] = pred_probs

        if torch.is_tensor(pred_probs):
            pred_probs = pred_probs.cpu()

        results = [getattr(metrics, m)()(pred_probs, Y, group) for m in metric_names]

        metric_df = pd.DataFrame(results).set_index('metric')
        if shuffle_seed is None: # log non-bootstrap final test value
            if self.use_nn:
                self.model.log_fairness_metrics_for_split(self.logger, pred_probs, Y, group, split) 
            if verbose:
                print("Results:")
                print(metric_df)
        
        return metric_df

    def plot_decision_boundary(self, save_path, preview_split="train", model_split="test"):
        fig, ax = preview_sim_data(
                self.data[preview_split][self.data_key],
                self.data[preview_split]["A"],
                self.data[preview_split]["Y"],
                self.data[preview_split]["Y_obs"],
                model_name=self.__class__.__name__
            )
        fig, ax = plot_model_predictions(self.data[model_split]["X"], self.model, title=self.__class__.__name__, fig=fig, ax=ax)
        self.logger.add_figure("/".join(["decision_boundary", self.__class__.__name__]), fig, global_step=self.model.global_step)
        fig.savefig(os.path.join(save_path, "decision_boundary.pdf"))
        
        fig, ax = plot_boundaries(self.sim, fig=fig, ax=ax)
        self.logger.add_figure("/".join(["all_boundaries", self.__class__.__name__]), fig, global_step=self.model.global_step)
        fig.savefig(os.path.join(save_path, "all_boundaries.pdf"))

        return fig, ax

    def save_model_data(self, parent_dir):
        """
            experiment_name_dir/
            |- model
            |- aux_models/
                |- ... <each model>
                |- aux_model_evaluation
        """

        save_path = os.path.join(parent_dir, f"{self.__class__.__name__}_model_info")
        def pickle_wrapper(fname, obj, base_dir=save_path):
            with open(os.path.join(base_dir, fname), "wb") as f:
                pickle.dump(obj, f)

        os.makedirs(save_path, exist_ok=True) # will (and should) raise error if path exists
        pickle_wrapper("model.pkl", self.model)
        if isinstance(self.model, nn.Module):
            torch.save(self.model.state_dict(), os.path.join(save_path, "model_state.pth"))
        if self.model.hparams.get("save_extra_info", False):
            pickle_wrapper("model_extra_info.pkl", self.model.extra_info)

        if self.data["train"][self.data_key].shape[-1] == 2:
            # don't try to plot an n-dimensional decision boundary -- the human mind was not meant to comprehend matplotlib in 10D
            fig, ax = self.plot_decision_boundary(save_path)
            pickle_wrapper("figax.pkl", {"fig": fig, "ax": ax})
            fig.clf()
            plt.close()

        if self.aux_models is not None:
            aux_model_path = os.path.join(save_path, "aux_models")
            os.makedirs(aux_model_path, exist_ok=True)
            if isinstance(self.aux_models, Iterable):
                for i, aux_model in enumerate(self.aux_models):
                    pickle_wrapper(f"aux_model_{i}.pkl", aux_model, base_dir=aux_model_path)
                    if isinstance(aux_model, nn.Module):
                        torch.save(aux_model.state_dict(), os.path.join(aux_model_path, f"aux_model_{i}_state.pth"))
            else:
                pickle_wrapper("aux_model.pkl", self.aux_models, base_dir=aux_model_path)
                if isinstance(self.aux_models, nn.Module):
                    torch.save(self.aux_models.state_dict(), os.path.join(aux_model_path, "aux_model_state.pth"))
            if self.aux_model_results is not None:
                self.aux_model_results.to_csv(os.path.join(aux_model_path, "aux_results.csv"))
            if self.aux_data_dict is not None:
                pickle_wrapper("aux_data.pkl", self.aux_data_dict, base_dir=aux_model_path)
        

class YModel(Model):

    def fit_model(self, split="train", cv_split="val"):
        if self.use_nn: # when using nn, we need to pass in val data during training to use for diagnostic purposes
            self.model.fit(
                    self.data[split][self.data_key],
                    self.data[split]["Y"],
                    X_val=self.data[cv_split][self.data_key],
                    y_val=self.data[cv_split]["Y"],
                    A=self.data[split]["A"],
                    A_val=self.data[cv_split]["A"],
                    T=self.data[split]["T"],
                    T_val=self.data[cv_split]["T"],
                    logger=self.logger,
                    )
        else:
            self.model.fit(self.data[split][self.data_key], self.data[split]["Y"])

    def plot_decision_boundary(self, save_path, preview_split="train", model_split="test"):
        fig, ax = preview_sim_data(
                self.data[preview_split][self.data_key],
                self.data[preview_split]["A"],
                self.data[preview_split]["Y"],
                self.data[preview_split]["Y"],
                model_name=self.__class__.__name__
            )
        fig, ax = plot_model_predictions(self.data[model_split]["X"], self.model, title=self.__class__.__name__, fig=fig, ax=ax)
        self.logger.add_figure("/".join(["decision_boundary", self.__class__.__name__]), fig, global_step=self.model.global_step)
        fig.savefig(os.path.join(save_path, "decision_boundary.pdf"))

        fig, ax = plot_boundaries(self.sim, fig=fig, ax=ax)
        self.logger.add_figure("/".join(["all_boundaries", self.__class__.__name__]), fig, global_step=self.model.global_step)
        fig.savefig(os.path.join(save_path, "all_boundaries.pdf"))
        return fig, ax


class YObsModel(Model):

    def fit_model(self, split="train", cv_split="val"):
        if self.use_nn:
            self.model.fit(
                    self.data[split][self.data_key],
                    self.data[split]["Y_obs"],
                    X_val=self.data[cv_split][self.data_key],
                    y_val=self.data[cv_split]["Y_obs"],
                    A=self.data[split]["A"],
                    A_val=self.data[cv_split]["A"],
                    T=self.data[split]["T"],
                    T_val=self.data[cv_split]["T"],
                    logger=self.logger,
                )
        else: 
            self.model.fit(self.data[split][self.data_key], self.data[split]["Y_obs"])

class TestedOnlyModel(Model):

    def fit_model(self, split="train", cv_split="val"):
        T = self.data[split]["T"]
        T_val = self.data[cv_split]["T"]
        if self.use_nn:
            self.model.fit(
                    self.data[split][self.data_key][T == 1],
                    self.data[split]["Y_obs"][T == 1],
                    X_val=self.data[cv_split][self.data_key][T_val == 1],
                    y_val=self.data[cv_split]["Y_obs"][T_val == 1],
                    A=self.data[split]["A"][T == 1],
                    A_val=self.data[cv_split]["A"][T_val ==1],
                    T=self.data[split]["T"][T == 1],
                    T_val=self.data[cv_split]["T"][T_val == 1],
                    logger=self.logger,
                )
        else:
            self.model.fit(self.data[split][self.data_key][T == 1], self.data[split]["Y_obs"][T == 1])

    def plot_decision_boundary(self, save_path, preview_split="train", model_split="test"):
        T = self.data[preview_split]["T"]
        fig, ax = preview_sim_data(
                self.data[preview_split][self.data_key][T == 1],
                self.data[preview_split]["A"][T == 1],
                self.data[preview_split]["Y"][T == 1],
                self.data[preview_split]["Y_obs"][T == 1],
                model_name=self.__class__.__name__
            )
        fig, ax = plot_model_predictions(self.data[model_split]["X"], self.model, title=self.__class__.__name__, fig=fig, ax=ax)
        self.logger.add_figure("/".join(["decision_boundary", self.__class__.__name__]), fig, global_step=self.model.global_step)
        fig.savefig(os.path.join(save_path, "decision_boundary.pdf"))
        
        fig, ax = plot_boundaries(self.sim, fig=fig, ax=ax)
        self.logger.add_figure("/".join(["all_boundaries", self.__class__.__name__]), fig, global_step=self.model.global_step)
        fig.savefig(os.path.join(save_path, "all_boundaries.pdf"))
        return fig, ax


class RecensoringModel(Model):

    def get_aux_models(self):
        aux_hparams = self.hparams.get("aux", self.hparams)
        if self.use_nn:
            self.aux_models = [SimpleMLP(self.data["train"][self.data_key].shape[-1], **aux_hparams), SimpleMLP(self.data["train"][self.data_key].shape[-1], **aux_hparams)]
        else:
            self.aux_models = [SVC(probability=True), SVC(probability=True)]  # for methods like PSM or recensoring -- should set Y_aux attribute of data dict

    def evaluate_aux_models(self):
        psm_0, psm_1 = self.aux_models
        results = []
        for split in self.data.keys():
            X = self.data[split][self.data_key]
            A = self.data[split]["A"]
            T = self.data[split]["T"]
            psm_0_probs = psm_0.predict_proba(X[A == 0])[:, 1].cpu()
            psm_1_probs = psm_1.predict_proba(X[A == 1])[:, 1].cpu()
            psm_0_preds = psm_0.predict(X[A == 0]).cpu()
            psm_1_preds = psm_1.predict(X[A == 1]).cpu()

            pred_probs = np.concatenate([psm_0_probs, psm_1_probs])
            preds = np.concatenate([psm_0_preds, psm_1_preds])
            group = np.concatenate([np.zeros((A == 0).sum()), np.ones((A == 1).sum())])
            result = [metrics.AUC()(pred_probs, T, group), metrics.Accuracy()(preds, T, group)]
            if self.logger is not None:
                for metric_dict, short_tag in zip(result, ["auc", "acc"]):
                    prefix = "/".join([short_tag, self.__class__.__name__ + "_Recensorer"])
                    self.logger.add_scalar(f"{prefix}/group0",  metric_dict["Group 0 value"]) 
                    self.logger.add_scalar(f"{prefix}/group1",  metric_dict["Group 1 value"])
                    self.logger.add_scalar(f"{prefix}/diff",  metric_dict["diff"])
                    self.logger.add_scalar(f"{prefix}/abs_diff",  np.abs(metric_dict["diff"]))
                    if metric_dict["overall"] is not None:
                        self.logger.add_scalar(prefix, metric_dict["overall"]) 

            df = pd.DataFrame(result).set_index('metric')
            results.append(df)
        self.aux_model_results = pd.concat(results, keys=self.data.keys())

    def get_aux_labels(self):
        psm_0, psm_1 = self.aux_models
        for split in self.data.keys():
            group0_mask, group1_mask = (self.data[split]["A"] == 0), (self.data[split]["A"] == 1)
            psm_0_censor = (psm_0.predict_proba(self.data[split][self.data_key])[:, 1] <= 0.5).cpu() # does this change if I do predict instead
            psm_1_censor = (psm_1.predict_proba(self.data[split][self.data_key])[:, 1] <= 0.5).cpu()
            Y_recensored = self.data[split]["Y_obs"].copy()
            Y_recensored[psm_1_censor & group0_mask] = 0
            Y_recensored[psm_0_censor & group1_mask] = 0
            self.data[split]["Y_aux"] = Y_recensored # we really only need this for train, but why not
        self.aux_labels = Y_recensored
        return Y_recensored

    def fit_aux_models(self, split="train"):
        psm_0, psm_1 = self.aux_models
        group0_mask, group1_mask = (self.data[split]["A"] == 0), (self.data[split]["A"] == 1)
        psm_0.fit(self.data[split][self.data_key][group0_mask], self.data[split]["T"][group0_mask], no_validate=True)
        psm_1.fit(self.data[split][self.data_key][group1_mask], self.data[split]["T"][group1_mask], no_validate=True)

    def fit_model(self, split="train", cv_split="val"):
        if self.use_nn:
            self.model.fit(
                    self.data[split][self.data_key],
                    self.data[split]["Y_aux"],
                    X_val=self.data[cv_split][self.data_key],
                    y_val=self.data[cv_split]["Y_aux"],
                    A=self.data[split]["A"],
                    A_val=self.data[cv_split]["A"],
                    T=self.data[split]["T"],
                    T_val=self.data[cv_split]["T"],
                    logger=self.logger,
                )
        else:
            self.model.fit(self.data[split][self.data_key], self.data[split]["Y_aux"])

    def plot_decision_boundary(self, save_path, preview_split="train", model_split="test"):
        fig, ax = preview_sim_data(
                self.data[preview_split][self.data_key],
                self.data[preview_split]["A"],
                self.data[preview_split]["Y"],
                self.data[preview_split]["Y_aux"],
                model_name=self.__class__.__name__
            )
        fig, ax = plot_model_predictions(self.data[model_split]["X"], self.model, title=self.__class__.__name__, fig=fig, ax=ax)
        self.logger.add_figure("/".join(["decision_boundary", self.__class__.__name__]), fig, global_step=self.model.global_step)
        fig.savefig(os.path.join(save_path, "decision_boundary.pdf"))
        
        fig, ax = plot_boundaries(self.sim, fig=fig, ax=ax)
        self.logger.add_figure("/".join(["all_boundaries", self.__class__.__name__]), fig, global_step=self.model.global_step)
        fig.savefig(os.path.join(save_path, "all_boundaries.pdf"))
        return fig, ax

class ReweightedModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_nn = True

    def get_model(self):
        if self.use_nn:
            self.model = SimpleMLP(
                    self.data["train"]["X"].shape[-1],
                    device=self.device, 
                    parent=self.__class__.__name__,
                    loss_fn=partial(weighted_loss_wrapped, nn.CrossEntropyLoss(reduction='none')),
                    **self.hparams
                )
        else:
            self.model = SVC(probability=True)
        return self.model


    def get_aux_models(self):
        aux_hparams = self.hparams.get("aux", self.hparams)
        self.aux_models = SimpleMLP(2 * self.data["train"][self.data_key].shape[-1] + 1, device=self.device,  parent=self.__class__.__name__ + "_propensity", **aux_hparams)
        return self.aux_models

    def fit_aux_models(self, split="train", cv_split="val"):

        print("Fitting aux models...")
        X = self.data[split][self.data_key]
        A = self.data[split]["A"]
        X_val = self.data[cv_split][self.data_key]
        A_val = self.data[cv_split]["A"]
        XA_train = np.concatenate([X, X * A.reshape(-1, 1), A.reshape(-1, 1)],axis=1)
        self.aux_models.fit(XA_train, self.data[cv_split]["T"], no_validate=True)


    def fit_model(self, split="train", cv_split="val"):
        X = self.data[split][self.data_key]
        A = self.data[split]["A"]
        X_val = self.data[cv_split][self.data_key]
        A_val = self.data[cv_split]["A"]

        XA_train = np.concatenate([X, X * A.reshape(-1, 1), A.reshape(-1, 1)],axis=1)
        XA_val = np.concatenate([X_val, X_val * A_val.reshape(-1, 1), A_val.reshape(-1, 1)], axis=1)
        aux_preds = self.aux_models.predict_proba(torch.from_numpy(XA_train).float().to(self.device))[:, 1]
        aux_preds_val = self.aux_models.predict_proba(torch.from_numpy(XA_val).float().to(self.device))[:, 1]
        weights = torch.where(torch.from_numpy(self.data[split]["T"] == 1).to(self.device), 1 / aux_preds, 1 / (1 - aux_preds))
        val_weights = torch.where(torch.from_numpy(self.data[cv_split]["T"] == 1).to(self.device), 1 / aux_preds_val, 1 / (1 - aux_preds_val))
        self.model.fit(
                    self.data[split][self.data_key],
                    self.data[split]["Y_obs"],
                    X_val=self.data[cv_split][self.data_key],
                    y_val=self.data[cv_split]["Y_obs"],
                    A=self.data[split]["A"],
                    A_val=self.data[cv_split]["A"],
                    T=self.data[split]["T"],
                    T_val=self.data[cv_split]["T"],
                    loss_weights=weights, 
                    val_loss_weights=val_weights,
                    logger=self.logger,
                )

    
class OneGroupBaselineModel(Model):
    def __init__(self, group_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group_id = group_id

    def fit_model(self, split="train", cv_split="val"):
        A = self.data[split]["A"]
        A_val = self.data[cv_split]["A"]
        X = self.data[split][self.data_key]
        X_val = self.data[cv_split][self.data_key]
        Y_obs = self.data[split]["Y_obs"]
        Y_obs_val = self.data[cv_split]["Y_obs"]
        T = self.data[split]["T"]
        T_val = self.data[cv_split]["T"]
        if self.use_nn:
            self.model.fit(
                X[A == self.group_id],
                Y_obs[A == self.group_id],
                X_val=X_val[A_val == self.group_id],
                y_val=Y_obs_val[A_val == self.group_id],
                A=A[A == self.group_id],
                A_val=A_val[A_val == self.group_id],
                T=T[A == self.group_id],
                T_val=T_val[A_val == self.group_id],
                logger=self.logger
            )
        else:
            self.model.fit(
                X[A == self.group_id],
                Y_obs[A == self.group_id],
                logger=self.logger
            )

class Group0BaselineModel(OneGroupBaselineModel):
    def __init__(self, *args, **kwargs):
        super().__init__(0, *args, **kwargs)

class Group1BaselineModel(OneGroupBaselineModel):
    def __init__(self, *args, **kwargs):
        super().__init__(1, *args, **kwargs)

class PeerLossModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.alpha = alpha
        self.use_nn = True # by default

    def get_optimal_params(self):
        Y_tr, Y_obs = self.data["train"]["Y"], self.data["train"]["Y_obs"]
        dy = 2 * Y_tr.mean() - 1 # == P(Y = 1) - P(Y = 0) = 2 * P(Y = 1) - 1
        dyobs = 2 * Y_obs.mean() - 1
        ratio = dy / dyobs
        yobs0_y1 = 1 - (1 - Y_obs[Y_tr == 1])
        self.alpha = 1 - (1 - yobs0_y1.mean()) * ratio
        if self.alpha < 0:
            print("Warning: calculated alpha was negative; setting to 0")
            self.alpha = 0

    def get_model(self):
        self.get_optimal_params()
        self.model = PeerLossMLP(
                self.data["train"][self.data_key].shape[-1],
                self.alpha,
                device=self.device,
                parent=self.__class__.__name__,
                **self.hparams
            ) 
        return self.model

    def fit_model(self, split="train", cv_split="val"):
        self.model.fit(
            self.data[split][self.data_key],
            self.data[split]["Y_obs"],
            X_val=self.data[cv_split][self.data_key],
            y_val=self.data[cv_split]["Y_obs"],
            A=self.data[split]["A"],
            A_val=self.data[cv_split]["A"],
            T=self.data[split]["T"],
            T_val=self.data[cv_split]["T"],
            logger=self.logger
        ) 

class GroupPeerLossModel(PeerLossModel):
    def get_optimal_params(self, eps=1e-8):
        Y_tr, Y_obs, A = self.data["train"]["Y"], self.data["train"]["Y_obs"], self.data["train"]["A"]
        dy = 2 * Y_tr.mean() - 1 # == P(Y = 1) - P(Y = 0) = 2 * P(Y = 1) - 1
        dyobs = 2 * Y_obs.mean() - 1
        ratio = dy / dyobs
        yobs1_y1 = Y_obs[Y_tr == 1] # P(Y_obs = 1 | Y = 1)
        self.alpha = 1 - (yobs1_y1.mean()) * ratio
        self.noise0 = np.clip(Y_obs[(Y_tr == 1) & (A == 0)].mean(), eps, 1-eps) # P(Y_obs = 1 | Y = 1, A = 0)
        self.noise1 = np.clip(Y_obs[(Y_tr == 1) & (A == 1)].mean(), eps, 1-eps)
        if self.alpha < 0:
            print(f"Warning: calculated alpha was {self.alpha}; setting to 0")
            self.alpha = 0
        print("Peer loss parameters:", self.alpha, self.noise0, self.noise1)

    def get_model(self):
        self.get_optimal_params()
        self.model = GroupPeerLossMLP(
                self.data["train"][self.data_key].shape[-1],
                self.alpha,
                self.noise0,
                self.noise1,
                device=self.device,
                parent=self.__class__.__name__,
                **self.hparams
            ) 
        return self.model


class SAREMModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_nn = True

    def get_model(self):
        self.model = PUEstimator(
                self.data["train"][self.data_key].shape[-1],
                device=self.device,
                parent=self.__class__.__name__,
                **self.hparams
            )         
        return self.model
    
    def fit_model(self, split="train", cv_split="val"):
        self.model.fit(
            self.data[split][self.data_key],
            self.data[split]["Y_obs"],
            self.data[split]["A"],
            self.data[cv_split][self.data_key],
            self.data[cv_split]["Y_obs"],
            self.data[cv_split]["A"],
            self.data[split]["Y"],
            self.data[cv_split]["Y"], # TODO: adapt for weight calc
            T=self.data[split]["T"],
            T_val=self.data[cv_split]["T"],
            logger=self.logger,
        ) 

class DCEMModel(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_nn = True

    def get_model(self):
        self.model = DisparateCensorshipEstimator(self.data["train"][self.data_key].shape[-1], device=self.device, parent=self.__class__.__name__, **self.hparams) 
        return self.model

    def fit_model(self, split="train", cv_split="val"):
        self.model.fit(
            self.data[split][self.data_key],
            self.data[split]["Y_obs"],
            self.data[split]["A"],
            self.data[split]["T"],
            self.data[cv_split][self.data_key],
            self.data[cv_split]["Y_obs"],
            self.data[cv_split]["A"],
            self.data[cv_split]["T"],
            self.data[split]["Y"],
            self.data[cv_split]["Y"],
            T_score=self.data[split]["T_prob"],
            T_score_val=self.data[cv_split]["T_prob"], 
            logger=self.logger,
            reg=self.hparams.get("reg", 1),
        )

    def plot_decision_boundary(self, save_path, preview_split="train", model_split="test"):
        fig, ax = preview_sim_data(
                self.data[preview_split][self.data_key],
                self.data[preview_split]["A"],
                self.data[preview_split]["Y"],
                self.data[preview_split]["Y_obs"],
                model_name=self.__class__.__name__
            )
        if self.model.hparams.get("tmle", False):
            print("Decision boundary not supported for TMLE")
            return fig, ax
        fig, ax = plot_model_predictions(self.data[model_split]["X"], self.model, title=self.__class__.__name__, fig=fig, ax=ax)
        self.logger.add_figure("/".join(["decision_boundary", self.__class__.__name__]), fig, global_step=self.model.global_step)
        fig.savefig(os.path.join(save_path, "decision_boundary.pdf"))
        
        fig, ax = plot_boundaries(self.sim, fig=fig, ax=ax)
        self.logger.add_figure("/".join(["all_boundaries", self.__class__.__name__]), fig, global_step=self.model.global_step)
        fig.savefig(os.path.join(save_path, "all_boundaries.pdf"))
        return fig, ax

class SELFModel(YObsModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_nn = True

    def get_model(self):
        self.model = SELFEstimator(self.data["train"][self.data_key].shape[-1], device=self.device, parent=self.__class__.__name__, **self.hparams)    
        return self.model
 

class TruncatedLQModel(YObsModel):
    def __init__(self, *args, q=0.7, k=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.q = q
        self.k = k

    def get_model(self, q=0.7, k=0.5): # from hpara search
        self.model = SimpleMLP(
                self.data["train"]["X"].shape[-1],
                loss_fn=partial(lq_loss, self.q, self.k),
                device=self.device,
                parent=self.__class__.__name__,
                **self.hparams
            )
        return self.model

class JSModel(YObsModel):
    def get_model(self):
        self.model = SimpleMLP(
                self.data["train"]["X"].shape[-1],
                loss_fn=js_scaled_loss,
                device=self.device,
                parent=self.__class__.__name__,
                **self.hparams
            )
        return self.model


class DivideMixBasedModel(YObsModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_nn = True

    def fit_model(self, split="train", cv_split="val"):
        self.model.fit(
                self.data[split][self.data_key],
                self.data[split]["Y_obs"],
                y_true=self.data[split]["Y"],
                X_val=self.data[cv_split][self.data_key],
                y_val=self.data[cv_split]["Y_obs"],
                A=self.data[split]["A"],
                A_val=self.data[cv_split]["A"],
                T=self.data[split]["T"],
                T_val=self.data[cv_split]["T"],
                logger=self.logger,
            )

    def get_model(self): 
        self.model = DivideMixEstimator(self.data["train"]["X"].shape[-1], device=self.device, parent=self.__class__.__name__, **self.hparams)
        return self.model

class ITECorrectedModel(YObsModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_nn = True

    def get_model(self):
        self.model = ITECorrectedMLP(self.data["train"]["X"].shape[-1], device=self.device, parent=self.__class__.__name__, **self.hparams)
        return self.model

    def plot_decision_boundary(self, save_path, preview_split="train", model_split="test"):
        for a in range(2):
            group_mask = (self.data[preview_split]["A"] == a)
            fig, ax = preview_sim_data(
                    self.data[preview_split][self.data_key][group_mask],
                    self.data[preview_split]["A"][group_mask],
                    self.data[preview_split]["Y"][group_mask],
                    self.data[preview_split]["Y_obs"][group_mask],
                    model_name=self.__class__.__name__
                )

            X = self.data[preview_split][self.data_key][group_mask]

            x0_min, x0_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
            x1_min, x1_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0
            xx0, xx1 = np.meshgrid(np.linspace(x0_min, x0_max, 100), np.linspace(x1_min, x1_max, 100))
            X_grid = np.c_[xx0.ravel(), xx1.ravel()]
            preds = self.model.predict_proba(X_grid, A=a * np.ones(len(X_grid)))

            fig, ax = plot_model_predictions(self.data[model_split]["X"], self.model, title=self.__class__.__name__, fig=fig, ax=ax, from_estimator=False, z=preds[:, 1], grid=(xx0, xx1))
            self.logger.add_figure("/".join([f"decision_boundary_{a}", self.__class__.__name__]), fig, global_step=self.model.global_step)
            fig.savefig(os.path.join(save_path, f"decision_boundary_{a}.pdf"))
            
            fig, ax = plot_boundaries(self.sim, fig=fig, ax=ax)
            self.logger.add_figure("/".join([f"all_boundaries_{a}", self.__class__.__name__]), fig, global_step=self.model.global_step)
            fig.savefig(os.path.join(save_path, f"all_boundaries_{a}.pdf"))

        return fig, ax

      
def _initializer(_data_key, _split, _model, _metric_names):
    global data_key
    data_key = _data_key
    global split
    split = _split
    global model
    model = _model
    global metric_names
    metric_names = _metric_names

def _bootstrap_eval_in_subprocess(shuffle_seed):
    np.random.seed(shuffle_seed)
    X, Y, group = split[data_key], split["Y"], split["A"]
    indices = np.random.choice(np.arange(len(X)), len(X), replace=True)
    X, Y, group = X[indices], Y[indices], group[indices]
    pred_probs = model.predict_proba(X)
    results = [getattr(metrics, m)()(pred_probs.cpu(), Y, group) for m in metric_names]
    return pd.DataFrame(results).set_index('metric')
