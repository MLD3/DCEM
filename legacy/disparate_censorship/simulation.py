from functools import partial
import os
import pickle
from typing import Callable, Optional 
import warnings

import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from scipy.stats import zscore

from .utils import _identity, down_staircase, cell_prototype, get_rotation_matrix, get_py, get_pt

VARNAMES = ["X", "X_enc", "A", "T", "Y", "Y_obs", "T_prob", "Y_prob"]
DEFAULT_SPLITS = ["train", "val", "test"]
# ["Y_score", "T_score"] TODO: decide if I want to add these to the return


def sigmoid(z, noinf=True):
    if noinf:
        z = np.clip(z, -np.log(np.finfo(z.dtype).max), None) # for numerical stability
    return 1 / (1 + np.exp(-z))

def apply_threshold(S, threshold, threshold_type, hardness=None, smoothness=None, noinf=True):
    if threshold_type == "hard":
        probs = S > threshold + self.eps
        return probs, probs.astype(float)
    elif threshold_type == "hard_smooth":  # {0, 1} -> {eps, 1-eps}
        probs = (S > threshold + self.eps) * \
            (1 - 2 * smoothness) + smoothness
        return np.random.rand(len(S)) < probs, probs.astype(float)
    elif threshold_type == "soft":
        probs = sigmoid(hardness * (S - threshold), noinf=noinf)
        return np.random.rand(len(S)) < probs, probs.astype(float)
    else:
        raise ValueError(
            "Test threshold type must be 'hard' or 'soft' or 'hard_smooth'")

class Simulator(object):
    def __init__(
        self,
    ):
        pass

    def set_seed(self, seed: int):
        np.random.seed(seed)

    def save(self, parent_dir="."):
        with open(os.path.join(parent_dir, "sim.pkl"), "wb") as f:
            pickle.dump(self, f)

    def _check_result_shapes(self, X, X_enc, A, T, Y, Y_obs):
        assert X.shape[0] == X_enc.shape[0]
        assert X.shape[0] == A.shape[0]
        assert X.shape[0] == T.shape[0]
        assert X.shape[0] == Y.shape[0]
        assert X.shape[0] == Y_obs.shape[0]
        assert A.ndim == 1 or all([d == 1 for d in A.shape[1:]])
        assert T.ndim == 1 or all([d == 1 for d in T.shape[1:]])
        assert Y.ndim == 1 or all([d == 1 for d in Y.shape[1:]])
        assert Y_obs.ndim == 1 or all([d == 1 for d in Y_obs.shape[1:]])


class MIMICIIISepsisDataAccessor(Simulator): # should probably change the class name from simulation now, no?
    def __init__(self, data_path, label_col="sepsis_6h",
            test_threshold_group0=None, test_threshold_group1=None,
            test_decision_fn0=None, test_decision_fn1=None, test_threshold_type="hard",
            threshold_hardness_0=10., threshold_hardness_1=10., fillna=-9999, normalize=False,
            **kwargs):
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.feature_cols = ["max_lactic_acid", "first_shock_index_age", "last_shock_index_age",
                "max_wbc", "delta_lactic_acid", "max_neuts", "max_glucose", "max_bun",
                "max_rr", "last_albumin", "min_sbp", "max_creatinine", "max_temp_f"] # TODO: somehow make this config-able --- refactor to make compatible with any pandas
        self.label_col = label_col
        self.group_col = "ethnicity_id"
        self.X = self.df[self.feature_cols]
        self.normalize = normalize
        if self.normalize:
            self.X = self.X.apply(zscore, axis=0)
        self.X = self.X.fillna(fillna).to_numpy()
        self.Y = self.df[self.label_col].astype(int).to_numpy()
        self.A = self.df[self.group_col].astype(int).to_numpy()

        self.test_threshold_group0 = test_threshold_group0
        self.test_threshold_group1 = test_threshold_group1
        self.test_decision_fn0 = test_decision_fn0
        self.test_decision_fn1 = test_decision_fn1
        self.test_threshold_type = test_threshold_type
        self.threshold_hardness_0 = threshold_hardness_0
        self.threshold_hardness_1 = threshold_hardness_1
        self.fillna = fillna

        warnings.warn(f"Extra arguments ignored: {kwargs}")

    def _check_test_settings(self):
        assert self.test_threshold_group0 is not None
        assert self.test_threshold_group1 is not None
        assert self.test_decision_fn0 is not None
        assert self.test_decision_fn1 is not None
        assert self.test_threshold_type is not None
        assert self.threshold_hardness_0 is not None
        assert self.threshold_hardness_1 is not None


    def get_prevalence(self):
        return self.Y.mean()

    def get_group_balance(self):
        assert len(np.unique(self.A)) == 2
        return 1 - self.A.mean() # P(A = 0)

    def update_testing_thresholds(self, test_threshold_group0, test_threshold_group1):
        self.test_threshold_group0 = test_threshold_group0
        self.test_threshold_group1 = test_threshold_group1

    def get_dataset(self, splits=DEFAULT_SPLITS, split_sizes=[0.6, 0.2, 0.2], seed=42):

        def _process_data(indices):
            X = self.X[indices]
            Y = self.Y[indices]
            A = self.A[indices]

            TS = np.where(A == 0, self.test_decision_fn0(X), self.test_decision_fn1(X))
            T0, T_prob0 = apply_threshold(TS, self.test_threshold_group0, self.test_threshold_type,
                        hardness=self.threshold_hardness_0)
            T1, T_prob1 = apply_threshold(TS, self.test_threshold_group1, self.test_threshold_type,
                        hardness=self.threshold_hardness_1)

            T = np.where(A == 0, T0, T1)
            T_prob = np.where(A == 0, T_prob0, T_prob1)
            Y_obs = Y * T
            return X, A, T, Y, Y_obs, T_prob, Y

        assert sum(split_sizes) == 1
        self._check_test_settings()
        np.random.seed(seed)
        perm = np.random.permutation(len(self.df))
        bounds = np.append([0], (np.cumsum(split_sizes) * len(self.df)).astype(int))
        self.indices = [perm[bounds[i]:bounds[i+1]] for i in range(len(split_sizes))]
        all_data = [dict(zip(["X", "A", "T", "Y", "Y_obs", "T_prob", "Y_prob"], _process_data(indices))) for indices in self.indices]
        return dict(zip(splits, all_data))



class TestingBiasSimulator(Simulator):
    def __init__(
        self,
        n_feats: int,
        # defining the down-staircase function
        # mostliy used for discretization
        label_step_size: Optional[float] = 0.2,
        # for simulating disparate censorship
        test_threshold_group0: Optional[float] = 1.0,
        test_threshold_group1: Optional[float] = 0.8,
        threshold_hardness_0: Optional[float] = 1.,
        threshold_hardness_1: Optional[float] = 1.,
        threshold_smoothness_0: Optional[float] = 1.,
        threshold_smoothness_1: Optional[float] = 1.,
        test_decision_fn0: Optional[Callable] = partial(down_staircase, 0.2),
        label_decision_fn0: Optional[Callable] = partial(down_staircase, 0.2),
        test_decision_fn1: Optional[Callable] = partial(down_staircase, 0.2),
        label_decision_fn1: Optional[Callable] = partial(down_staircase, 0.2),
        # for simulating conditional shift in A P(Y|X) -> P(Y|X')
        p_group0: Optional[float] = 0.5,
        label_threshold_group0: Optional[float] = None,
        label_threshold_group1: Optional[float] = None,
        label_threshold_type: Optional[str] = "hard",
        label_hardness_0: Optional[float] = 1.,
        label_hardness_1: Optional[float] = 1.,
        label_smoothness_0: Optional[float] = 1.,
        label_smoothness_1: Optional[float] = 1.,
        rotation_group0: Optional[float] = 0.,
        rotation_group1: Optional[float] = 0.,
        test_threshold_type: Optional[str] = "hard",
        test_rotation_group0: Optional[float] = 0.,
        test_rotation_group1: Optional[float] = 0.,
        pivot0: Optional[np.ndarray] = 0.4 * np.ones(2),
        pivot1: Optional[np.ndarray] = 0.4 * np.ones(2),
        test_pivot0: Optional[int] = 0.4 * np.ones(2),
        test_pivot1: Optional[int] = 0.4 * np.ones(2),
        n_rot_dims0: Optional[int] = 2,
        n_rot_dims1: Optional[int] = 2,
        test_n_rot_dims0: Optional[int] = 2,
        test_n_rot_dims1: Optional[int] = 2,
        # clipping covariates
        min_value: Optional[float] = 0.0,
        max_value: Optional[float] = 1.0,
        # for simulating covariate shift in A P(X) -> P(X')
        mu0: Optional[int] = None,
        mu1: Optional[int] = None,
        sigma: Optional[np.ndarray] = 0.05 * np.eye(2),
        # numerical stability parameters
        discretize: Optional[bool] = False,
        eps: Optional[float] = 1e-8,
        seed: Optional[int] = 42,
        **kwargs,
    ):
        super(TestingBiasSimulator, self).__init__()
        self.n_feats = n_feats
        self.label_step_size = label_step_size

        self.min_value = min_value
        self.max_value = max_value
        self.mu0 = mu0
        self.mu1 = mu1

        self.p_group0 = p_group0

        self.sigma = sigma
        self.eps = eps

        self.test_th0 = test_threshold_group0
        self.test_th1 = test_threshold_group1
        self.test_threshold_type = test_threshold_type

        # the "slope" of the sigmoid for soft thresholding
        self.threshold_hardness_0 = threshold_hardness_0
        self.threshold_hardness_1 = threshold_hardness_1

        # for making the probailities not 0-1 but still sharp -- i.e., label smoothing
        self.threshold_smoothness_0 = threshold_smoothness_0
        self.threshold_smoothness_1 = threshold_smoothness_1

        self.test_decision_fn0 = test_decision_fn0
        self.label_decision_fn0 = label_decision_fn0
        self.test_decision_fn1 = test_decision_fn1
        self.label_decision_fn1 = label_decision_fn1

        self.label_th0 = label_threshold_group0
        self.label_th1 = self.label_th0 # by assumption

        self.label_threshold_type = label_threshold_type

        # the "slope" of the sigmoid for soft thresholding
        self.label_hardness_0 = label_hardness_0
        self.label_hardness_1 = label_hardness_1

        # for making the probailities not 0-1 but still sharp -- i.e., label smoothing
        self.label_smoothness_0 = label_smoothness_0
        self.label_smoothness_1 = label_smoothness_1

        self.rot_0 = rotation_group0
        self.rot_1 = rotation_group1
        self.n_rot_dims0 = n_rot_dims0
        self.n_rot_dims1 = n_rot_dims1

        mats0 = [get_rotation_matrix(rotation_group0)] * (n_rot_dims0 // 2) + [
            np.eye(2)] * ((n_feats - n_rot_dims0) // 2)
        self.rmat_0 = block_diag(*mats0)
        mats1 = [get_rotation_matrix(rotation_group1)] * (n_rot_dims1 // 2) + [
            np.eye(2)] * ((n_feats - n_rot_dims1) // 2)
        self.rmat_1 = block_diag(*mats1)

        self.test_rot_0 = test_rotation_group0
        self.test_rot_1 = test_rotation_group1
        self.test_n_rot_dims0 = test_n_rot_dims0
        self.test_n_rot_dims1 = test_n_rot_dims1
        test_mats0 = [get_rotation_matrix(test_rotation_group0)] * (
            test_n_rot_dims0 // 2) + [np.eye(2)] * ((n_feats - test_n_rot_dims0) // 2)
        self.test_rmat_0 = block_diag(*test_mats0)
        test_mats1 = [get_rotation_matrix(test_rotation_group1)] * (
            test_n_rot_dims1 // 2) + [np.eye(2)] * ((n_feats - test_n_rot_dims1) // 2)
        self.test_rmat_1 = block_diag(*test_mats1)

        self.pivot0 = pivot0  # "origin" of the rotation -- should be on the decision boundary
        self.pivot1 = pivot1
        self.test_pivot0 = test_pivot0
        self.test_pivot1 = test_pivot1

        # self.down_staircase = partial(down_staircase, self.label_step_size)
        self.discretize = discretize
        if self.discretize:
            self.prototype_fn = partial(cell_prototype, self.label_step_size)
        else:
            self.prototype_fn = _identity

        np.random.seed(seed)
        if len(kwargs):
            warnings.warn(f"Extra arguments ignored: {kwargs}")

    def get_dataset(self, splits=DEFAULT_SPLITS, sizes=[1000] * len(DEFAULT_SPLITS)):
        # val_n is deprecated and has no effect
        if isinstance(sizes, int):
            sizes = [sizes] * len(splits)
        all_data = [dict(zip(VARNAMES, self.simulate(size))) for size in sizes]
        return dict(zip(splits, all_data))


    def simulate(self, n: int, nbins: Optional[int] = 5):
        # Modeling rotational conditional shift
        # Operate on *prototypes* of points, not actual points -- to preserve perfect separation

        def protate(X, pivot, rot, rmat):
            if rot % 360 != 0.:
                return ((self.prototype_fn(X + self.eps) - pivot) @ rmat) + pivot
            else:
                return self.prototype_fn(X + self.eps)
 

        n0 = int(self.p_group0 * n * 2)
        n1 = 2 * n - n0
        sample0 = np.random.multivariate_normal(self.mu0, self.sigma, size=n0)
        sample1 = np.random.multivariate_normal(self.mu1, self.sigma, size=n1)
        A = np.concatenate([np.zeros(len(sample0)), np.ones(len(sample1))])
        X_A0 = np.clip(sample0, self.min_value + self.eps,
                       self.max_value - self.eps)
        X_A1 = np.clip(sample1, self.min_value + self.eps,
                       self.max_value - self.eps)
        X = np.concatenate([X_A0, X_A1])
        X_enc = self.bin_and_cat(X, nbins)

        R_A0 = protate(X_A0, self.pivot0, self.rot_0, self.rmat_0)
        R_A1 = protate(X_A1, self.pivot1, self.rot_1, self.rmat_1)
        TR_A0 = protate(X_A0, self.test_pivot0,
                        self.test_rot_0, self.test_rmat_0)
        TR_A1 = protate(X_A1, self.test_pivot1,
                        self.test_rot_1, self.test_rmat_1)

        # decision boundaries are calculated from the "rotated" prototypical scores (decision boundary transform)
        RS_A0 = self.test_decision_fn0(R_A0)
        RS_A1 = self.test_decision_fn1(R_A1)
        S_A0 = self.label_decision_fn0(TR_A0)
        S_A1 = self.label_decision_fn1(TR_A1)

        T0, PT0 = apply_threshold(RS_A0, self.test_th0, self.test_threshold_type,
                                  hardness=self.threshold_hardness_0, smoothness=self.threshold_smoothness_0)
        T1, PT1 = apply_threshold(RS_A1, self.test_th1, self.test_threshold_type,
                                  hardness=self.threshold_hardness_1, smoothness=self.threshold_smoothness_1)
        T = np.concatenate([T0, T1])
        PT = np.concatenate([PT0, PT1])

        Y0, PY0 = apply_threshold(S_A0, self.label_th0, self.label_threshold_type,
                                  hardness=self.label_hardness_0, smoothness=self.label_smoothness_0)
        Y1, PY1 = apply_threshold(S_A1, self.label_th1, self.label_threshold_type,
                                  hardness=self.label_hardness_1, smoothness=self.label_smoothness_1)
        Y = np.concatenate([Y0, Y1])
        PY = np.concatenate([PY0, PY1])
        Y_obs = T * Y

        self._check_result_shapes(X, X_enc, A, T, Y, Y_obs)
        return X, X_enc, A.astype(int), T.astype(int), Y.astype(int), Y_obs.astype(int), PT, PY

    def bin_and_cat(self, X: np.ndarray, nbins: int):
        dummy_onehots = np.eye(nbins)
        indices = np.digitize(X, np.linspace(
            X.min(), X.max() + 1e-6, nbins + 1), right=False) - 1
        X_enc = dummy_onehots[indices].reshape(X.shape[0], -1)
        return X_enc


def find_simulation_with_parameters(
        target_prevalence=0.25,
        k=2,
        testing_disparity=0.5,
        prevalence_disparity=1/3,
        group_balance=0.5,
        n_feats=2,
        verbose=False,
        **kwargs
    ):
        
        params = compute_target_parameters(target_prevalence, prevalence_disparity, testing_disparity, group_balance, k, verbose=True)
        final_kwargs = get_kwargs_with_target(*params, n_feats, kwargs, verbose=verbose) # pass the kwargs as a single dict for compactness
        return TestingBiasSimulator(n_feats, **final_kwargs)

def find_simulated_test_settings_with_parameters(
        group_balance,
        prevalence,
        k=2,
        testing_disparity=0.5,
        verbose=False,
        **kwargs,
    ):
        target_pt = k * prevalence
        if target_pt >= 1.:
            target_pt0 = 1.
            target_pt1 = 1.
        else:
            target_pt0, target_pt1 = compute_group_rates(target_pt, testing_disparity, group_balance)
        if verbose:
            print("Prevalence:", prevalence)
            print(f"Testing rate (K={k}):", target_pt)
            print("Group testing:", target_pt0, target_pt1)
        print("Finding target parameters for testing rate...")
        if target_pt >= 1.:
            test_threshold_group0 = float('-inf')
            test_threshold_group1 = float('-inf')
        else:
            test_threshold_group0, test_threshold_group1 = bisect_groupwise(target_pt0, "test_threshold_group0", target_pt1, "test_threshold_group1", get_pt, kwargs, min_guess=-10000, max_guess=10000, invert=True, verbose=verbose, simulation_only=False)
        final_kwargs = kwargs | {"test_threshold_group0": test_threshold_group0, "test_threshold_group1": test_threshold_group1}
        return MIMICIIISepsisDataAccessor(**final_kwargs)

def compute_target_parameters(target_prevalence, prevalence_disparity, testing_disparity, group_balance, k, verbose=False):
    target_py0, target_py1 = compute_group_rates(target_prevalence, prevalence_disparity, group_balance)
    target_pt = k * target_prevalence
    if target_prevalence * k >= 1.:
        target_pt0 = 1.
        target_pt1 = 1.
    else:
        target_pt0, target_pt1 = compute_group_rates(target_pt, testing_disparity, group_balance)
    if verbose:
        print("Prevalence:", target_prevalence)
        print(f"Testing rate (K={k}):", target_pt)
        print("Group prevalence:", target_py0, target_py1)
        print("Group testing:", target_pt0, target_pt1)
    return target_prevalence, target_pt, target_py0, target_py1, target_pt0, target_pt1

def compute_group_rates(total, ratio, pa, tol=1e-8):

    b = total / (pa * ratio + (1 - pa))
    a = b * ratio
    assert np.abs(pa * a + (1 - pa) * b - total) < tol, pa * a + (1 - pa) * b
    assert np.abs(a / b - ratio) < tol, a/b
    if a < 0 or a > 1 or b < 0 or b > 1:
        raise ValueError(f"Invalid group rates: a={a}, b={b} from total={total}, ratio={ratio}, and pa={pa}")
    return a, b

def bisect_groupwise(target_0, key_0, target_1, key_1, get_group_stat, default_kwargs, var_dims=2, n_feats=2, tol=1e-12, min_guess=-0.9, max_guess=1, invert=False, guess_tol=1e-12, bisection_sample_size=100000, verbose=False, report_every=1, max_iters=10000, simulation_only=True):
    """
        Uses the bisection algorithm to solve for parameter values that yield certain group-wise testing rates and condition prevalences.
        Condition prevalance is monotonic increasing in the group-wise means mu_0 and mu_1.
        Testing rate is monotonic **decreasing** in the testing threshold variables test_threshold_group0 and test_threshold_group1.

        target_0: target value for group 0
        key_0: parameter name (keyword argument for TestingBiasSimulator) to modify for group 0
        target_1: " for group 1
        key_1: " for group 1
        get_group_stat: function [data, group #] -> R to evaluate how close simulated data is to target_0, target_1
        var_dims: dimensionality of the variable(s) to be changed
        tol: tolerance parameter for residual
        min_guess: lower bound for the parameter values for TestingBiasSimulator
        max_guess: upper bound for the parameter values for TestingBiasSimulator
        invert: if True, a positive residual implies that the parameter value guess was too **low**; otherwise, a positive residual implies parameter value was too high 
        guess_tol: tolerance parameter for difference between the lower and upper guess at each iteration
        bisection_sample_size: simulation sample size to use for computing statistics via get_group_stat
    """

    # run bisection algorithm by group in parallel to find appropriate simulation parameters
    resid0 = float('inf')
    resid1 = float('inf')

    lb0, ub0 = min_guess, max_guess
    lb1, ub1 = min_guess, max_guess

    iters = 0
    if not simulation_only:
        kwargs = default_kwargs.copy()
        sim = MIMICIIISepsisDataAccessor(**kwargs)
    while (np.abs(resid0) > tol or np.abs(resid1) > tol) and not ((np.abs(ub0 - lb0) < guess_tol) or (np.abs(ub1 - lb1) < guess_tol)) and iters != max_iters:
        if iters == 0:
            # check that the guess bounds are good
            if simulation_only:
                kwargs = default_kwargs.copy()
                kwargs.update({key_0: np.ones(var_dims) * lb0,
                    key_1: np.ones(var_dims) * lb1}) # TODO: refactor this so we can just get residuals given simulation + variables to test
                sim = TestingBiasSimulator(n_feats, **kwargs)
                data_lb = sim.get_dataset(splits=["train"], sizes=[bisection_sample_size])
                p0_lb, p1_lb = get_group_stat(data_lb, group=0), get_group_stat(data_lb, group=1)

                kwargs.update({key_0: np.ones(var_dims) * ub0,
                    key_1: np.ones(var_dims) * ub1}) # TODO: refactor this so we can just get residuals given simulation + variables to test
                sim = TestingBiasSimulator(n_feats, **kwargs)
                data_ub = sim.get_dataset(splits=["train"], sizes=[bisection_sample_size])
                p0_ub, p1_ub = get_group_stat(data_ub, group=0), get_group_stat(data_ub, group=1)
            else:
                sim.update_testing_thresholds(lb0, lb1) 
                data_lb = sim.get_dataset(splits=["train"])
                p0_lb, p1_lb = get_group_stat(data_lb, group=0), get_group_stat(data_lb, group=1)
                
                sim.update_testing_thresholds(ub0, ub1)
                data_ub = sim.get_dataset(splits=["train"])
                p0_ub, p1_ub = get_group_stat(data_ub, group=0), get_group_stat(data_ub, group=1)
            
            resid_lb = p0_lb - target_0
            resid_ub = p0_ub - target_0
            if resid_lb * resid_ub > 0:
                warnings.warn(f"Guess for Group 0 is bad: residuals in interval [{lb0}, {ub0}] are [{resid_lb}, {resid_ub}].")


            resid_lb = p1_lb - target_1
            resid_ub = p1_ub - target_1
            if resid_lb * resid_ub > 0:
                warnings.warn(f"Guess for Group 1 is bad: residuals in interval [{lb1}, {ub1}] are [{resid_lb}, {resid_ub}].")



        guess0 = (lb0 + ub0) / 2
        guess1 = (lb1 + ub1) / 2
        if simulation_only:
            kwargs = default_kwargs.copy()
            kwargs.update({key_0: np.ones(var_dims) * guess0,
                          key_1: np.ones(var_dims) * guess1})
            sim = TestingBiasSimulator(n_feats, **kwargs)
            data = sim.get_dataset(splits=["train"], sizes=[bisection_sample_size])
        else:
            sim.update_testing_thresholds(guess0, guess1) # avoid expensive I/O every iteration
            data = sim.get_dataset(splits=["train"])

        

        p0, p1 = get_group_stat(data, group=0), get_group_stat(data, group=1)
        resid0 = p0 - target_0
        resid1 = p1 - target_1
        if resid0 > 0:
            # prevalance too high; adjust *up*ward (testing threshold)
            if invert:
                lb0 = guess0
            else:  # prevalance too high; adjust downward
                ub0 = guess0
        else:
            if invert: # prev too low; adjust downward (testing threhsold)
                ub0 = guess0
            else: # prev too low; adjust upward
                lb0 = guess0

        if resid1 > 0:
            if invert:
                lb1 = guess1
            else:
                ub1 = guess1
        else:
            if invert:
                ub1 = guess1
            else:
                lb1 = guess1
        iters += 1
        if verbose and iters % report_every == 0:
            print(f"Iter {iters}: guess {guess0:.3f}, {guess1:.3f} - resids {p0:.3f}-{target_0:.3f}={resid0:.3f}, {p1:.3f}-{target_1:.3f}={resid1:.3f} - search in ({lb0:.3f}, {ub0:.3f}), ({lb1:.3f}, {ub1:.3f})")
    if iters == max_iters: 
        warnings.warn("Bisection failed to converge in time; double-check parameters.")
    return guess0, guess1

def get_kwargs_with_target(target_py, target_pt, target_py0, target_py1, target_pt0, target_pt1, n_feats, default_kwargs, verbose=False, testing_only=False):
    kwargs = default_kwargs.copy()
    if not testing_only:
        print("Finding target parameters for prevalence...")
        mu0, mu1 = bisect_groupwise(target_py0, "mu0", target_py1, "mu1", get_py, kwargs, var_dims=n_feats, n_feats=n_feats, verbose=verbose)
        if target_py0 == target_py1:
            avg = (mu0 + mu1) / 2
            mu0 = avg
            mu1 = avg
        kwargs.update({"mu0": np.ones(n_feats) * mu0, "mu1": np.ones(n_feats) * mu1})

    print("Finding target parameters for testing rate...")
    test_threshold_group0, test_threshold_group1 = bisect_groupwise(target_pt0, "test_threshold_group0", target_pt1, "test_threshold_group1", get_pt, kwargs, var_dims=1, n_feats=n_feats, min_guess=-10, max_guess=1, invert=True, verbose=verbose)
    if target_pt0 >= 1:
        test_threshold_group0 = -9999
    if target_pt1 >= 1:
        test_threshold_group1 = -9999
    if target_pt0 == target_pt1 and target_py0 == target_py1 and mu0 == mu1:
        avg = (test_threshold_group0 + test_threshold_group1) / 2
        test_threshold_group0 = avg
        test_threshold_group1 = avg
    kwargs.update({"test_threshold_group0": test_threshold_group0, "test_threshold_group1": test_threshold_group1})
    return kwargs


def validate_simulation(sim, size=50000):
    print("Simulation validation results:")
    if isinstance(sim, TestingBiasSimulator):
        data = sim.get_dataset(sizes=[size])
    else:
        data = sim.get_dataset()

    actual_py = get_py(data)
    actual_pt = get_pt(data)
    actual_py0, actual_py1 = get_py(data, group=0), get_py(data, group=1)
    actual_pt0, actual_pt1 = get_pt(data, group=0), get_pt(data, group=1)

    print("Prevalence:", actual_py)
    print(f"Testing rate:", actual_pt)
    print("Group prevalence:", actual_py0, actual_py1)
    print("Group testing:", actual_pt0, actual_pt1)
