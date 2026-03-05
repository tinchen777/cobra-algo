# src/cobra_log/exceptions.py
"""
Exceptions for for :pkg:`cobra_log` package.
"""


# === WARNING ===
class CobraAlgoWarning(Warning):
    r"""Base warning class for :pkg:`cobra_log` package."""


# === ERROR ===
class CobraAlgoError(Exception):
    r"""Base error class for :pkg:`cobra_log` package."""


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
