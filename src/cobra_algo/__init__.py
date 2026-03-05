# src/cobra_algo/__init__.py
"""
cobra-algo
===========

A collection of algorithms for data processing, machine learning, and deep learning.

Modules
-------
- :mod:`cobra_algo.text`: Text processing algorithms, including `TF-IDF` calculation and vectorization.

Examples
--------

```python
from cobra_algo.text import tf_idf

X = np.array([
    [2, 0, 1, 0],
    [0, 1, 1, 0],
    [1, 1, 0, 1]
], dtype=float)

r = tf_idf(X, X[[1, 2]])
```
"""

__author__ = "Zhen Tian"
__version__ = "0.1.0"

__all__ = [
]
