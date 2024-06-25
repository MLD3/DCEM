import torch
from torch.utils.data import Subset, TensorDataset

from typing import Optional, Union

TENSOR_INDICES = {
    "X": 0,
    "A": 1,
    "T": 2,
    "Y_obs": 3,
    "Y": 4,
    "prop_score": 5,
}

def _query_tensor_dataset(dataset: Union[TensorDataset, Subset], key: str) -> torch.Tensor:
    """Accessor for TensorDataset for the dataset format required by DCEM.

    Args:
        dataset (TensorDataset): A PyTorch TensorDataset object.
        key (str): The name of the variable type (see keys of `TENSOR_INDICES`).

    Returns:
        Tensor: a Tensor containing all values for the specified variable.
    """
    if isinstance(dataset, Subset):
        return torch.cat([dataset.dataset.tensors[TENSOR_INDICES[key]][idx] for idx in dataset.indices], dim=0)
    else:
        return dataset.tensors[TENSOR_INDICES[key]]

def _enforce_bounds(
        var,
        varname: str,
        _min: Optional[float] = float('-inf'),
        _max: Optional[float] = float('inf'),
    ) -> None:
    """Checks if an array-like is in-bounds.

    Args:
        var (Any): Array-like variable; e.g., a tensor.
        varname (str): Variable name (plaintext) to be used in the error message.
        _min (Optional[float], optional): Minimum allowed value. Defaults to float('-inf').
        _max (Optional[float], optional): Maximum allowed value. Defaults to float('inf').

    Raises:
        ValueError: Raises an error if the array is not completely within the specified bounds.
    """
    if var.min() < _min or var.max() > _max:
        raise ValueError(f"{varname} is not in [{_min}, {_max}]. Got min: {var.min()} and max {var.max()}")

def _enforce_scalar_bounds(
        var,
        varname: str,
        _min: Optional[float] = float('-inf'),
        _max: Optional[float] = float('inf'),
    ) -> None:
    """Same as _enforce_bounds, but for checking scalars.

    Args:
        var (Any): Scalar-like variable; e.g., a float.
        varname (str): Variable name (plaintext) to be used in the error message.
        _min (Optional[float], optional): Minimum allowed value. Defaults to float('-inf').
        _max (Optional[float], optional): Maximum allowed value. Defaults to float('inf').

    Raises:
        ValueError: Raises an error if the value is not completely within the specified bounds.
    """
    if var < _min or var > _max:
        raise ValueError(f"{varname} is not in [{_min}, {_max}]. Got: {var}")
