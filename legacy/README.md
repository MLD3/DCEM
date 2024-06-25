# From Biased Selective Labels to Pseudo-Labels: An Expectation-Maximization Framework for Learning from Biased Decisions

This is the official code release for the paper [From Biased Selective Labels to Pseudo-Labels: An Expectation-Maximization Framework for Learning from Biased Decisions](https://icml.cc/virtual/2024/poster/33393) (ICML 2024). 

We have released this repository as-is for reproducibility. Note that this repo may change without notice. We are working on releasing a streamlined implementation of DCEM for practical use.

## Quickstart

### Installation

Simply run `pip install -r requirements.txt`. Note that we used **Python 3.9** for all experiments.

### Main command 


All experiments are managed using `.yaml` files. Commands are of the form

```
    python main.py --config [YOUR_EXPERIMENT_CONFIG].yaml --dataset [DATASET] [--overwrite]
```
where the fully synthetic data is the default option. The other option for the `--dataset` option is `sepsis`. The `--overwrite` option is False by default, and can be used to forcibly overwrite existing experimental results. **We have replaced absolute paths on our machine with placeholders - please replace them with the relevant absolute paths on your machine following the "Setup" section.**

### Examples

**Train models on fully synthetic data**

```
    python main.py --config final_experiment_settings/full_dcem_smoothed.yaml
```
will train DCEM models for a bunch of simulation settings.

**Train models on the sepsis classification task**

For reference, we also include an example command for training on the sepsis task. 

```
    python main.py --config sepsis_experiments/sepsis_dcem.yaml --dataset sepsis 
```

### Results

Results will be saved at `{EXPERIMENT_RESULTS_DIR}/{NAME}/disparate_censorship_mitigation`. You can set the experimental results directory in `experiments.py`. `NAME` is the name of your experiment (as specified under `"name"` in the YAML config file).

For example, suppose you have:

**experiments.py**:
```
    EXPERIMENT_RESULTS_DIR = "/home/"
    ...
```
**my_experiment.yaml**:
```
    "name": "my_awesome_experiment"
    ...
```
Running `python main.py --config my_expeirment.yaml` will train a model on synthetic data, and save the results to `/home/my_awesome_experiment`.

This folder will have additional subdirectories of the form `/home/my_experiment/[SIMULATION_PARAMETERS]/[MODEL_NAME]`.


## Setup

### Data paths

* `RESULTS_DICT` is a dict mapping from the `hostname` of the machine to a file path. This is where we get `EXPERIMENT_RESULTS_DIR`. Files for the fully synthetic setting will be saved here.
* `SEPSIS_DICT` is the analogous variable for the pseudo-synthetic sepsis classification task.
* `SEPSIS_FNAME` is the location of the final sepsis cohort dataset.


### Additional details for obtaining Sepsis-3 MIMIC-III cohort

MIMIC-III is a publicly available but credentialed dataset. For more information about obtaining access to MIMIC-III, please visit the [PhysioNet](https://physionet.org/) and create an account there. To obtain our Sepsis-3 cohort, we follow the instructions from the [alistairewj/sepsis3-mimic](https://github.com/alistairewj/sepsis3-mimic) repository.

Runnning `sepsis-3-get-data.ipynb` from that repository should yield a CSV file with a list of individuals, along with whether Sepsis-3 criterion was met. Our notebook `Disparate Sepsis Prototyping.ipynb` starts by loading this CSV file and other MIMIC tables to create our final cohort.

## Reproducing our results

Here is a table describing all config files used to run our experiments. All experiments follow the general command template above.

**Synthetic data:**

All config paths are relative to `synthetic_data_experiments`. A glob `*` indicates an expansion of the form `_phase###` (*i.e.*, a different $s_Y$). We've provided links to baselines (papers and *official* code) that others proposed to deal with different noisy/biased label settings; please check out their awesome work as well!

|Model|Config file|
|--|--|
|`full_y_smoothed_*.yaml`|$y$-model|
|`full_yobs_smoothed_*.yaml`|$y$-obs model|
|`full_testedonly_smoothed_*.yaml`|tested-only model|
|`full_g0_smoothed_*.yaml`|Group 0 only|
|`full_g1_smoothed_*.yaml`|Group 1 only|
|`full_self_smoothed_*.yaml`|SELF [[paper]](https://arxiv.org/abs/1910.01842)|
|`full_dividemix_smoothed_*.yaml`|DivideMix [[paper]](https://arxiv.org/abs/2002.07394) [[code]](https://github.com/LiJunnan1992/DivideMix)|
|`full_js_smoothed_*.yaml`|Generalized Jensen-Shannon Loss [[paper]](https://arxiv.org/abs/2105.04522) [[code]](https://github.com/ErikEnglesson/GJS)|
|`full_lq_smoothed_*.yaml`|Truncated LQ Loss [[paper]](https://arxiv.org/pdf/1805.07836) |
|`full_ite_smoothed_*.yaml`|DragonNet correction [[paper]](https://arxiv.org/abs/1906.02120) [[code]](https://github.com/claudiashi57/dragonnet)|
|`full_peer_smoothed_*.yaml`|Peer Loss [[paper]](http://proceedings.mlr.press/v119/liu20e/liu20e.pdf) [[code]](https://github.com/gohsyi/PeerLoss)|
|`full_gpl_smoothed_*.yaml`|Group Peer Loss [[paper]](https://arxiv.org/abs/2011.00379) [[code]](https://github.com/Faldict/fair-classification-with-noisy-labels)|
|`full_sarem_smoothed_*.yaml`|SAR-EM [[paper]](https://arxiv.org/abs/1809.03207) [[code]](https://github.com/ML-KULeuven/SAR-PU)|
|`full_dcem_smoothed_*.yaml`|DCEM (ours) [[paper]](https://icml.cc/virtual/2024/poster/33393) [[code]]()|

**Sepsis classification:**

All config paths are relative to `sepsis_experiments`. A glob `*` indicates an expansion of the form `_alpha###` (*i.e.*, a different $s_T$).


|Model|Config file|
|--|--|
|`sepsis_y_*.yaml`|$y$-model|
|`sepsis_yobs_*.yaml`|$y$-obs model|
|`sepsis_testedonly_*.yaml`|tested-only model|
|`sepsis_g0_*.yaml`|Group 0 only|
|`sepsis_g1_*.yaml`|Group 1 only|
|`sepsis_self_*.yaml`|SELF [[paper]](https://arxiv.org/abs/1910.01842)|
|`sepsis_ite_*.yaml`|DragonNet correction [[paper]](https://arxiv.org/abs/1906.02120) [[code]](https://github.com/claudiashi57/dragonnet)|
|`sepsis_gpl_*.yaml`|Group Peer Loss [[paper]](https://arxiv.org/abs/2011.00379) [[code]](https://github.com/Faldict/fair-classification-with-noisy-labels)|
|`sepsis_sarem_*.yaml`|SAR-EM [[paper]](https://arxiv.org/abs/1809.03207) [[code]](https://github.com/ML-KULeuven/SAR-PU)|
|`sepsis_dcem_*.yaml`|DCEM (ours) [[paper]](https://icml.cc/virtual/2024/poster/33393) [[code]]()|

Config paths for re-running our sensivitity analyses are provided in the "Other details" section.

## Contact

Please reach out to `ctrenton` at `umich` dot `edu` if you have any questions, or file a Github issue.


## Other details
 
### Important Files

* `main.py`: Entry point for running all experiments 
* `experiments.py`: Experiment management classes; handles file/model/data saving and contains module wrappers.
* `nn_modules.py`: Raw PyTorch modules. 
* `losses.py`: Custom objective functions. 
* `metrics.py`: Evaluation metrics.
* `disparate_censorship/*.py`: code containing the disparate censorship data-generating process simulator.

### Experimental Settings

All experimental settings are managed using YAML files, and can be found in the `final_experiment_settings/` directory for each method, separated by model, for the fully synthetic setting, and the `sepsis_experiments/` directory for the pseudo-synthetic sepsis classification task. In general, some important keys for the YAML files include:

* `name`: determines where data will be stored.
* `model`: a list of models to test.
* `experimental_variables`: a nested dict of different simulation parameters to sweep over ($k, q_t, q_y$).
* `other_sim_kwargs`: simulation arguments. This is where we vary $s_T, s_Y$ (testing and true decision boundaries).
* `model_hparams`: a nested dict of hyperparameters, separated by model
* `n`: dataset size (fully synthetic data only)
* `data_path`: path to dataset (pseudo-synthetic sepsis classification task only)

Under the `other_sim_kwargs` key:
* `test_decision_fn{a}`: Testing decision boundary for group $a$, specified by the class name (in `boundaries.py`) and kwargs.
* `label_decision_fn{a}`: True class decision boundary for group $a$, specified by the class name (in `boundaries.py`) and kwargs. Equal across groups by assumption.

We provide an example config file for DCEM in `example_config.yaml`.


### Notebooks (figures and data processing)

* `notebooks/Figures.ipynb` contains all code for figure generation and saving. 
* `notebooks/Disparate Sepsis Prototyping` contains a version of the pre-processing code used to extract features from the MIMIC-III Sepsis-3 cohort, which is partially redacted to ensure compliance with the MIMIC-III terms of use.
* `notebooks/Demystifying sepsis testing boundary` contains Appendix Figure 9.

To plot the figures, make sure to run `results_merger.py` first with the relevant directory containing results for all models and all simulation parameters (the "name" field in the `.yaml` experiment files). The `notebooks/` directory should also contain the raw PDF figure files as they appear in the manuscript (including Appendix figures).

### Original results

As a courtesy, we include the full results of all models on all settings for both the fully and pseudo-synthetic tasks at `all_fully_synthetic_results.csv` and `all_pseudo_synthetic_sepsis_results.csv`, respectively. Results from these files were used to create our figures.

When running `main.py` directly, in addition to a CSV of results, we also save PDFs (for the fully synthetic data only) showing the decision boundaries learned by each model, with or without $s_T, s_Y$, and various `.pkl` files of the data used to train and test the model, the underlying simulation object, and all model objects (using `state_dict` where applicable). 

### Reproducing sensitivity analyses and ablation studies 

If you're interested in reproducing the sensitivity analyses and ablation studies of DCEM in our Appendix, check out the following config files. 

**Ablation study of DCEM components** (Appendix E.2)
|Ablation|Config file|
|--|--|
|Imputation-only/pseudo-labeling|`dcem_ablations/pseudolabel*.yaml`|
|No causal regularization|`dcem_ablations/full_dcem_noreg*.yaml`|

**Sensitivity analysis of other causal effect estimators** (Appendix E.3)

|Model|Config file|
|--|--|
|Tested-only + group|`causal_sensitivity_analysis/full_group_tested_*.yaml`|
|Reweighted (IPW)|`causal_sensitivity_analysis/reweighted_naive_*.yaml`|
|DR-Learner|`causal_sensitivity_analysis/xfitdr_*.yaml`|

**Robustness analysis of causal effect estimators with respect to overlap violations** (Appendix E.3)

|Overlap (as mult. of original)| Config file|
|--|--|
|1/4x|`causal_overlap_sensitivity/overlap7.5*.yaml`|
|1/2x|`causal_overlap_sensitivity/overlap15*.yaml`|
|2x|`causal_overlap_sensitivity/overlap60*.yaml`||
|4x|`causal_overlap_sensitivity/overlap120*.yaml`|

**Sensitivity analysis of propensity model softmax temperature** (Appendix E.4)
|Prop. Model Softmax Temp.| Config file|
|--|--|
|0.01 | `dcem_temp_sensitivity_analysis/full_dcem_smoothed_prior0.5_t0.01_*.yaml`|
|0.1 | `dcem_temp_sensitivity_analysis/full_dcem_smoothed_prior0.5_t0.1_*.yaml`|
|10.0 | `dcem_temp_sensitivity_analysis/full_dcem_smoothed_prior0.5_t10.0_*.yaml`|
|100.0 | `dcem_temp_sensitivity_analysis/full_dcem_smoothed_prior0.5_t100.0_*.yaml`|

*Note: $t=1$ is the default* 

**Initialization: tested-only vs. random** (Appendix E.5)

|Initialization| Config file|
|--|--|
|Random|`dcem_ablations/full_dcem_soothed*.yaml`|
|Tested-only|`synthetic_data_experiments/full_dcem_tested_only*.yaml`|




