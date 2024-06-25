import numpy as np

def down_staircase(step, x): # R^d -> R
    return step * np.ceil(x / step).sum(axis=-1)

