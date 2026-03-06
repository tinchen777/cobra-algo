# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import (Any, Optional, Set, List)

try:
    import torch
except ImportError:
    torch = None


def to_numpy(obj: Any, /, dtype: Optional[Any] = None, copy: bool = True):
    """
    Convert the given object to a `NumPy array`, i.e. :class:`np.ndarray` instance.

    Parameters
    ----------
        obj : Any
            The object to be converted to a `NumPy array`.
            - _torch.Tensor_(need :pkg:`torch`): Converted to a `NumPy array` after detaching and moving to CPU;
            - _scalar_: Converted to a 1-D `NumPy array` containing the scalar value;
            - _set_: Converted to a `NumPy array` containing the elements of the set;
            - _others_: Converted to a `NumPy array` directly.

        dtype : Optional[Any], default to `None`
            The data type of the resulting `NumPy array`.
            - `None`: Use the default data type of the object.

        copy : bool, default to `True`
            Control whether to create a copy of the object when converting to a `NumPy array`.
            - `True`: Create a copy of the object;
            - `False`: A copy will only be made if `__array__` returns a copy.

    Returns
    -------
        NDArray
            The converted `NumPy array` representation of the object.
    """
    if torch is not None and isinstance(obj, torch.Tensor):
        # as torch.Tensor
        obj = obj.detach().cpu()
    elif np.isscalar(obj):
        # as scalar
        obj = [obj]
    elif isinstance(obj, Set):
        # as set
        obj = list(obj)

    return np.array(obj, dtype=dtype, copy=copy)


class AlgoResult:
    """
    A lightweight container for algorithm outputs with unified format
    conversion utilities.

    The result behaves similarly to a NumPy array while providing
    convenient conversion methods such as `to_numpy()`, `to_tensor()`,
    and `to_list()`.
    """

    def __init__(self, values: NDArray[Any], /, class_name: str = "", **kwargs: Any):
        self._values = np.asarray(values)
        self._class_name = class_name
        self._attrs = kwargs

    def to_numpy(self, copy: bool = False) -> NDArray[Any]:
        """Return result as a NumPy array."""
        return self._values.copy() if copy else self._values

    def to_list(self) -> List[Any]:
        """Return result as a Python list."""
        return self._values.tolist()

    def to_tensor(self, copy: bool = False):
        """Convert result to a PyTorch tensor."""
        if torch is None:
            raise ImportError(f"`PyTorch` is required for `{self._class_name}.to_tensor()`.")
        if copy:
            return torch.tensor(self._values)
        return torch.from_numpy(self._values)

    def update_attr(self, **kwargs: Any):
        """Update additional attributes of the result."""
        self._attrs.update(kwargs)
        return self

    @property
    def values(self):
        """The NumPy array values."""
        return self._values

    def __array__(self):
        """Allow implicit NumPy conversion."""
        return self._values

    def __getitem__(self, item: Any):
        return self._values[item]

    def __len__(self):
        return len(self._values)

    def __getattr__(self, name):
        """Allow dynamic access to attributes stored in attrs."""
        if name in self._attrs:
            return self._attrs[name]

        raise AttributeError(f"{self._class_name} instance has no attribute {name!r}")

    def __repr__(self):
        shape = self._values.shape
        return f"AlgoResult(class={self._class_name}, shape={shape})"
