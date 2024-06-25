# Disparate Censorship Expectation-Maximization (DCEM)

This is the official code release for "From Biased Selective Labels to Pseudo-Labels: An Expectation-Maximization Framework for Learning from Biased Decisions" (ICML '24).

Sometimes, when training a machine learning model, labels are missing/noisy. Sometimes, they're even missing/noisy in biased ways. We [proposed Disparate Censorship Expectation-Maximization (ICML '24)](https://icml.cc/virtual/2024/poster/33393) to help mitigate impacts on model performance and mitigate bias.

This algorithm was inspired by a real-world problem in machine learning for healthcare: often, we assume that individuals that didn't get a diagnostic test are negative. This has been documented in papers analyzing COVID-19 testing, sepsis diagnostic definitions, and more. 

If you're interested in reproducing our results from our paper, please check out the README in the `legacy` folder, which contains all of our experimental code. If you're interested in applying DCEM to your own problems/learning how to implement DCEM yourself, we recommend consulting this repo.

## Quickstart

Run `pip install -r requirements.txt` to get the required dependencies.

Here's how you can apply DCEM to your own data:
```
    from dcem import DCEM

    # setup (what you should initialize)
    train_data, test_data = ... # load this however you like
    X_tr, A_tr, T_tr, Y_obs_tr = train_data
    propensity_model = ... # must be a nn.Module or implement `.fit()`
    outcome_model = ... # must be a nn.Module
    model = DCEM(propensity_model, outcome_model)

    # training
    model.fit(X_tr, A_tr, T_tr, Y_obs_tr, Y=Y_tr) # optionally pass in y in synthetic data

    # inference
    X_tr, *_ = test_data
    preds = model.predict_proba(X_tr)[:, 1]
```

We also provide a full demo of DCEM with example synthetic data in `demo.py`. 

## How DCEM works (informally)

DCEM is designed for situations where labeling decisions are *noisy* and *potentially biased* (in a fairness/equity sense). In such situations, if we fit a model to simply predict the observed outcome, we'll probably also learn to replicate these labeling biases. That's often undesirable. 

Enter DCEM: our method leverages variables that we assume do not affect the outcome of interest (such as "protected attributes") to learn a model that "compensates" for labeling biases. For a comprehensive and formal treatment of DCEM, please see [our paper](https://icml.cc/virtual/2024/poster/33393).

## Contributing/reporting issues

**Contributions.** We absolutely welcome contributions. This is a fairly bare-bones implementation of DCEM, but we hope to grow the functionality. Please raise an issue to discuss potential extensions or features you'd like to see *before* submitting a pull request. 

**Issues/bugs.** All models are wrong; some are useful. Sadly, the same is not true of code. Please open an issue to discuss any potential bugs!

## Contact

Please reach out to `ctrenton` at `umich` dot `edu` or file a Github issue if you have any questions about our work. Thank you!
