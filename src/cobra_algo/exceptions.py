# src/cobra_algo/exceptions.py
"""
Exceptions for for :pkg:`cobra_algo` package.
"""


# === WARNING ===
class CobraAlgoWarning(Warning):
    r"""Base warning class for :pkg:`cobra_algo` package."""


# === ERROR ===
class CobraAlgoError(Exception):
    r"""Base error class for :pkg:`cobra_algo` package."""


class NotFittedError(CobraAlgoError):
    """Error raised when a method is called before fitting."""


class DimensionError(CobraAlgoError):
    """Error raised when there is a dimension mismatch."""


class IDFError(CobraAlgoError):
    """Error raised during IDF calculation."""


class TFIDFError(CobraAlgoError):
    """Error raised during TF-IDF calculation."""


class CountVectorizerError(CobraAlgoError):
    """Error raised during Count Vectorizer operations."""


class VocabularyError(CobraAlgoError):
    """Error raised when there is an issue with the vocabulary."""
