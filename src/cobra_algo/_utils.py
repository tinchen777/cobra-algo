# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

import numpy as np
from typing import (Any, Optional, Set)

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
