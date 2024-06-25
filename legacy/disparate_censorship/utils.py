import numpy as np

def down_staircase(step, x): # R^d -> R
    return step * np.ceil(x / step).sum(axis=-1)

def cell_prototype(step, x): # R^d -> R^d
    return np.ceil(x / step) * step - step / 2

def get_rotation_matrix(deg): # 0 to 360
    # 2D rotation that "shifts" decision boundary by "deg" degrees counterclockwise (within some plane)
    rad = deg * np.pi / 180 
    return np.array([[np.cos(-rad), -np.sin(-rad)],
                     [np.sin(-rad), np.cos(-rad)]])

def _identity(x):
    return x

def get_group(data, split="train"):
    return data[split]["A"]

def get_py(data, group=None, split="train"):
    if group is not None:
        return data[split]["Y"][get_group(data) == group].mean()
    return data[split]["Y"].mean()

def get_pt(data, group=None, split="train"):
    if group is not None:
        return data[split]["T"][get_group(data) == group].mean()
    return data[split]["T"].mean()
