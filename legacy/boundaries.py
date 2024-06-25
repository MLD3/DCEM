import numbers

import numpy as np
from scipy.linalg import block_diag

from disparate_censorship.utils import get_rotation_matrix

class DecisionBoundary(object):
    """
        A class of parameterized functions f: R^n -> R to output a scalar-valued
        "score" for thresholding.
    """
    def __init__(self):
        pass

    def __call__(self):
        pass

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__dict__ == other.__dict__

class Linear(DecisionBoundary):
    def __init__(self, m=1., b=0.5, vert=False):
        super().__init__()
        if not isinstance(m, np.ndarray):
            self.m = np.array(m) # shape (n_dims - 1, 1)
        else:
            self.m = m
        self.b = b # shape (1)
        self.vert = vert
    
    def __call__(self, x):
        m = self.m
        if not len(m.shape):
            m = self.m.repeat(x.shape[1] - 1)
        xpartial = (x[..., :-1] @ m.reshape((-1, 1))).ravel() + self.b
        if self.vert:
            return xpartial
        return xpartial - x[..., -1]

class Biquadratic(DecisionBoundary):
    def __init__(self, a=2.5, b=0.5, c=0.5, d=0.4):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def __call__(self, x):
        return biquadratic(x, a=self.a, b=self.b, c=self.c, d=self.d)

class Quadratic(DecisionBoundary):
    def __init__(self, a=1., h=0.2, k=0.5):
        super().__init__()
        self.a = a
        self.h = h
        self.k = k

    def __call__(self, x):
        return quadratic_slice(x, a=self.a, h=self.h, k=self.k)

class TiltedQuadratic(DecisionBoundary):
    def __init__(self, a=10., h=0.2, k=0.5, deg=-30., dims=None, center=0.5):
        super().__init__()
        self.a = a
        self.h = h
        self.k = k
        self.dims = dims
        self.deg = deg
        self.center = center

    def __call__(self, x):
        assert x.shape[1] % 2 == 0
        if self.dims is None:
            self.dims = x.shape[1] # rotate all pairs of dimensions
        matlist = [get_rotation_matrix(self.deg)] * (self.dims // 2) + [np.eye(2)] * ((x.shape[1] - self.dims) // 2)
        rmat = block_diag(*matlist)
        xtrans = x @ rmat + np.ones(x.shape[1]) * self.center
        return quadratic_slice(xtrans, a=self.a, h=self.h, k=self.k)

class SpinnySinewave(DecisionBoundary):
    def __init__(self, a=0.25, translate=0., period=0.25, deg=-30., dims=None, center=0.5):
        super().__init__()
        self.a = a
        self.translate = translate
        self.period = period
        self.deg = deg
        self.dims = dims
        self.center = center

    def __call__(self, x):
        return spinny_sin(x, a=self.a, translate=self.translate, period=self.period, deg=self.deg, dims=self.dims, center=self.center)

class NaiveLinearSepsisBoundary(DecisionBoundary):
    def __init__(self, indices=[8, 10], norms=[9.807072379603103, 21.76634883880265], #, 44.21709377256948],
            thresholds=[22, 100], #, 50],
            invert=[False, True],
            weights=[0.5, 0.5]): #, False]):
        super().__init__()
        self.indices = np.array(indices)
        self.norms = np.array(norms)
        self.thresholds = np.array(thresholds)
        self.invert = np.array(invert).astype(int)
        self.weights = np.array(weights)

    def __call__(self, x):
        feature_scores = (x[:, self.indices] - self.thresholds) * (1 -2 * self.invert) / self.norms * self.weights
        return feature_scores.sum(axis=-1)

class NaiveQSofa(DecisionBoundary):
    def __init__(self, rr_idx=8, sbp_idx=10, rr_norm=12.665177604623324, sbp_norm=21.99300487850566):
        super().__init__()
        # very low tech testing boundary based on intersection of
        # covariates used in the Risk-of-Sepsis model and covariates
        # used in theqsofa score calc -- no GCS here
        self.rr_idx = rr_idx
        self.sbp_idx = sbp_idx
        self.rr_norm = rr_norm
        self.sbp_norm = sbp_norm

    def __call__(self, x):
        rr_threshold = 22 # qSOFA threshold
        sbp_threshold = 100 # qSOFA threshold
        excess_rr = (x[:, self.rr_idx] - 22) / self.rr_norm
        sbp_under = (100 - x[:, self.sbp_idx]) / self.sbp_norm
        return excess_rr + sbp_under

"""
    The following functions are deprecated and should be moved within the class functionality and yeeted
"""
def biquadratic(x, a=2.5, b=0.5, c=0.7, d=0.):
        return x[..., -1] - np.square(np.square(a * (x[..., :-1] - b).sum(axis=-1)) - c) - d

def last_dim_slice(i, x): # wrapped in partial for pickle-ability
    return x[..., i]

def quadratic_slice(x, a=1., h=0.2, k=0.5):
    x_vec = x[..., :-1] - h
    return x[..., -1] - a * np.square(x_vec).sum(axis=-1) - k

def spinny_sin(x, a=0.25, translate=0., period=0.25, deg=-30., dims=None, center=0.5):
    """
        This function first creates a rotated "view" of x (applying 
        rotations in orthogonal 2D subspaces by `deg` degree, then adding translation `center`), 
        and then applies a decision boundary based on a sine function with
        amplitude `a`, x-translation amount `y`, and period `period`.

        It's slightly hacky, yes -- but the main idea is that the boundary
        needs to be sufficiently irregular such that observing a portion
        of the boundary is insufficient to infer the structure of the rest
        of the boundary (i.e., high nonlinearity).
    """
    assert x.shape[1] % 2 == 0
    if dims is None:
        dims = x.shape[1] # rotate all pairs of dimensions
    matlist = [get_rotation_matrix(deg)] * (dims // 2) + [np.eye(2)] * ((x.shape[1] - dims) // 2)
    rmat = block_diag(*matlist)
    xtrans = x @ rmat + np.ones(x.shape[1]) * center
    return xtrans[..., 0] - a * np.sin(2 * np.pi / period * xtrans[..., 1:] + np.radians(translate)).sum(axis=-1)

