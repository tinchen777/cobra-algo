# src/cobra_algo/text/__init__.py
"""
Text processing algorithms for :pkg:`cobra-algo`.

Functions
---------
- :func:`tf_idf`: Calculate `TF-IDF` matrix for the given samples.
Classes
-------
- :class:`CountVectorizer`: Count Vectorizer for converting documents to a matrix of token counts.
- :class:`GeneralizedTFIDF`: A flexible `TF-IDF` transformer for generic feature matrices.
"""

from ._tf_idf import (tf_idf, CountVectorizer, GeneralizedTFIDF)

__all__ = [
    "tf_idf",
    "CountVectorizer",
    "GeneralizedTFIDF"
]
