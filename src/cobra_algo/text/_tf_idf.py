# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations
from collections import Counter
from cobra_array.compat import CompatArray
from typing import (Any, Dict, Literal, Union, Sequence, Optional, NamedTuple, TYPE_CHECKING)

from .._utils import CXP
from ..exceptions import (
    NotFittedError,
    DimensionError,
    IDFError,
    TFIDFError,
    VocabularyError,
    CountVectorizerError
)

if TYPE_CHECKING:
    from cobra_array.types import (ArrayLike, AnyDevice)

EPS = 1e-12


class CountResult(NamedTuple):
    result: CompatArray[type[float], Any]
    vocab: Union[Dict[Any, int], int]


class TFIDFResult(NamedTuple):
    result: CompatArray[type[float], Any]
    idf: Optional[CompatArray[type[float], Any]]


class AllTFIDFResult(NamedTuple):
    result: CompatArray[type[float], Any]
    idf: Optional[CompatArray[type[float], Any]]
    vocab: Union[Dict[Any, int], int]


def tf_idf(
    D: Any,
    /,
    d: Optional[Any] = None,
    input_mode: Literal["tokens", "matrix"] = "matrix",
    tf_mode: Literal["raw", "norm", "log"] = "norm",
    df_mode: Union[Literal["zero", "mean"], float] = "zero",
    idf_mode: Literal["log", "log_smooth"] = "log_smooth",
    norm: Optional[Literal["l1", "l2"]] = None
) -> AllTFIDFResult:
    """
    Calculate `TF-IDF` matrix for the given samples.

    Parameters
    ----------
        D : Sequence[Any]
            A list of samples to calculate `IDF`, shape as `[num_samples, num_features]`.
            Also use to transform to `TF-IDF` matrix (if :param:`d` is `None`)
            Must be `non-negative`.
            - _Sequence[Sequence[Any]]_: Each sample is a list of tokens when `input_mode` is `"tokens"`;
            - _MatrixLike_: All samples are represented as a matrix when `input_mode` is `"matrix"`.

        d : Optional[Sequence[Any]], default to `None`
            - _Sequence_: A list of specific samples to transform to `TF-IDF` matrix, shape as `[num_samples, num_features]`. Must be `non-negative` and have the same number of features as :param:`D`;
            - `None`: All samples in :param:`D` will be transformed to `TF-IDF` matrix.

        input_mode : Literal["tokens", "matrix"], default to `"matrix"`
            The input mode of samples.
            - `"tokens"`: Each sample is a list of tokens, and the vocabulary will be built from the unique tokens in the samples;
            - `"matrix"`: All samples are represented as a matrix when `input_mode` is `"matrix"`.

        tf_mode : Literal["raw", "norm", "log"], default to `"norm"`
            The calculation mode of Term Frequency (`TF`).
            - `"raw"`: Use raw term counts as `TF`;
            - `"norm"`: Normalize term counts by the total count of terms in the sample (L1 normalization);
            - `"log"`: Use logarithmically scaled term counts as `TF`, i.e., `TF = log(1 + count)`.

        df_mode : Union[Literal["zero", "mean"], float], default to `"zero"`
            The calculation mode of Document Frequency (`DF`).
            - `"zero"`: Count the number of samples where the term appears (count > 0) as `DF`;
            - `"mean"`: Count the number of samples where the term appears with a count greater than or equal to the mean count of that sample as `DF`;
            - _float_, value in [0, 1]: Count the number of samples where the term appears with a count greater than or equal to the specified percentile of counts for that term across all samples as `DF`.

        idf_mode : Literal["log", "log_smooth"], default to `"log_smooth"`
            The calculation mode of Inverse Document Frequency (`IDF`).
            - `"log"`: Use logarithmically scaled inverse document frequency, i.e., `IDF = log(N / DF)`, where `N` is the total number of samples;
            - `"log_smooth"`: Use smoothed logarithmically scaled inverse document frequency, i.e., `IDF = log((1 + N) / (1 + DF)) + 1`, which prevents division by zero and ensures that terms appearing in all samples do not get an `IDF` of zero.

        norm : Optional[Literal["l1", "l2"]], default to `None`
            The normalization mode for the output `TF-IDF` matrix.
            - `None`: No normalization is applied to the output `TF-IDF` matrix;
            - `"l1"`: Normalize each sample's `TF-IDF` vector to have an L1 norm of 1 (i.e., the sum of absolute values is 1);
            - `"l2"`: Normalize each sample's `TF-IDF` vector to have an L2 norm of 1 (i.e., the square root of the sum of squares is 1).

    Returns
    -------
        AllTFIDFResult
            - result(_CompatArray_): The TF-IDF matrix, shape as `[num_samples, num_features]`;
            - idf(_CompatArray_): The calculated IDF values used for transformation, shape as `[num_features]`;
            - vocab(_Union[Dict[Any, int], int]_): The vocabulary built from the samples.

    Examples
    --------
    >>> from cobra_algo.text import CountVectorizer, tf_idf
    ... X = np.array([
    ...     [0.1, 1.2, 0.3],
    ...     [0.0, 0.5, 1.0],
    ...     [0.5, 0.5, 0.0],
    ... ], dtype=float)
    >>> X = CountVectorizer("matrix").fit_transform(X).result
    >>> tfidf_result = tf_idf(
    ...     X,
    ...     tf_mode="raw",
    ...     idf_mode="log_smooth",
    ...     norm=None,
    ...     df_mode="zero"
    ... )
    >>> tfidf_result.result.to_numpy()
    [[0.12876821 1.2        0.38630462]
    [0.         0.5        1.28768207]
    [0.64384104 0.5        0.        ]]
    ...
    >>> tfidf_result.idf
    [1.28768207 1.         1.28768207]
    ...
    >>> tfidf_result.vocab
    3
    """
    pipe = CountVectorizer(input_mode)
    transformer = GeneralizedTFIDF(tf_mode, True, df_mode, idf_mode, norm)

    pipe_result = pipe.fit_transform(D)
    # transformer fit
    X = pipe_result.result
    transformer.fit(X)
    # transformer transform
    tfidf_result = transformer.transform(X if d is None else pipe.transform(d).result)

    return AllTFIDFResult(tfidf_result.result, tfidf_result.idf, pipe_result.vocab)


class CountVectorizer:
    """
    Count Vectorizer for converting documents to a matrix of token counts.

    Examples
    --------
    >>> from cobra_algo.text import CountVectorizer
    >>> docs = [
    ...     ["apple", "banana", "apple"],
    ...     ["banana", "orange"],
    ...     ["banana", "banana", "apple"]
    ... ]
    >>> vectorizer = CountVectorizer("tokens")
    >>> result = vectorizer.fit_transform(docs)
    ...
    >>> result.to_numpy()
    [[2. 1. 0.]
    [0. 1. 1.]
    [1. 2. 0.]]
    ...
    >>> result.vocab
    {'apple': 0, 'banana': 1, 'orange': 2}
    """
    def __init__(self, input_mode: Literal["tokens", "matrix"] = "matrix"):
        """
        Initialize `CountVectorizer`.

        Parameters
        ----------
            input_mode : Literal["tokens", "matrix"], default to `"matrix"`
                The input mode of documents.
                - `"tokens"`: Each document is a list of tokens, and the vocabulary will be built from the unique tokens in the documents;
                - `"matrix"`: Each document is represented as a vector, and the number of features (vocabulary size) will be determined by the number of columns in the input matrix.

        Raises
        ------
            ValueError
                If `input_mode` is not one of the allowed values.
        """
        if input_mode not in ("tokens", "matrix"):
            raise ValueError(f"`input_mode` parameter must be either 'tokens' or 'matrix', got {input_mode!r}.")
        self._input_mode = input_mode
        self._vocab: Union[Dict[Any, int], int] = 0

    def fit(self, docs: object, /) -> CountVectorizer:
        """
        Build vocabulary from documents.

        Parameters
        ----------
            docs : object
                A array of documents to build the vocabulary.
                - _Sequence[Sequence[Any]]_: Each document is a list of tokens when `input_mode` is `"tokens"`;
                - _MatrixLike_: All documents are represented as a matrix when `input_mode` is `"matrix"`.

        Returns
        -------
            CountVectorizer

        Raises
        ------
            VocabularyError
                If there is an error during building vocabulary.
        """
        try:
            if self._input_mode == "tokens":
                # input as tokens
                if not isinstance(docs, Sequence):
                    raise TypeError(f"`docs` parameter of `CountVectorizer.fit()` must be a sequence when `input_mode` is 'tokens', got {type(docs).__name__}")
                terms = sorted(set(term for doc in docs for term in doc))
                self._vocab = {term: i for i, term in enumerate(terms)}
            else:
                # input as matrix
                docs_arr = CompatArray(docs)
                if docs_arr.ndim != 2:
                    raise DimensionError(f"`docs` parameter of `CountVectorizer.fit()` must be a 2-D array when `input_mode` is 'matrix', got {docs_arr.ndim}-D")
                self._vocab = docs_arr.shape[1]
            return self

        except Exception as e:
            raise VocabularyError("Building vocabulary error.") from e

    def transform(self, docs: object, /) -> CountResult:
        """
        Transform documents to count matrix using the built vocabulary.

        Parameters
        ----------
            docs : object
                A array of documents to transform based on the built vocabulary.
                - _Sequence[Sequence[Any]]_: Each document is a list of tokens when `input_mode` is `"tokens"`;
                - _MatrixLike_: All documents are represented as a matrix when `input_mode` is `"matrix"`.

        Returns
        -------
            CountResult
                - result(_CompatArray_): The count matrix, shape as `[num_docs, num_vocabulary]`;
                - vocab(_Union[Dict[Any, int], int]_): The vocabulary built from the documents.

        Raises
        ------
            NotFittedError
                If the vocabulary has not been built. Call :meth:`CountVectorizer.fit` first.
            CountVectorizerError
                If there is an error during count vectorization.
        """
        if not self._vocab:
            raise NotFittedError("Vocabulary is empty. Call `CountVectorizer.fit()` first.")

        try:
            if isinstance(self._vocab, int):
                # input as matrix
                docs_arr = CompatArray(docs)
                if docs_arr.ndim != 2:
                    raise DimensionError(f"`docs` parameter of `CountVectorizer.transform()` must be a 2-D array when `input_mode` is 'matrix', got {docs_arr.ndim}-D")
                result = docs_arr.cxp.zeros(
                    (len(docs_arr), self._vocab),
                    dtype=float,
                    device=docs_arr.device
                )

                c = min(docs_arr.shape[1], self._vocab)
                result[:, :c] = docs_arr[:, :c]
            else:
                # input as tokens
                if not isinstance(docs, Sequence):
                    raise TypeError(f"`docs` parameter of `CountVectorizer.transform()` must be a sequence when `input_mode` is 'tokens', got {type(docs).__name__}")
                result = CXP.zeros((len(docs), len(self._vocab)), dtype=float)
                for i, doc in enumerate(docs):
                    term_counts = Counter(doc)
                    for term, count in term_counts.items():
                        if term in self._vocab:
                            result[i, self._vocab[term]] = count

            return CountResult(result, self._vocab)

        except Exception as e:
            raise CountVectorizerError("Count vectorization error.") from e

    def fit_transform(self, docs: object, /) -> CountResult:
        """
        Build vocabulary from documents, then transform documents to count matrix.

        Parameters
        ----------
            docs : object
                A array of documents to build the vocabulary and transform based on the built vocabulary.
                - _Sequence[Sequence[Any]]_: Each document is a list of tokens when `input_mode` is `"tokens"`;
                - _MatrixLike_: All documents are represented as a matrix when `input_mode` is `"matrix"`.

        Returns
        -------
            CountResult
                - result(_CompatArray_): The count matrix, shape as `[num_docs, num_vocabulary]`;
                - vocab(_Union[Dict[Any, int], int]_): The vocabulary built from the documents.
        """
        return self.fit(docs).transform(docs)

    @property
    def vocab(self):
        """
        The vocabulary built from the documents.
        - _Dict[Any, int]_: A mapping from term to index when `input_mode` is `"tokens"`.
        - _int_: The number of features (vocabulary size) when `input_mode` is `"matrix"`.
        """
        return self._vocab


class GeneralizedTFIDF:
    """
    A flexible `TF-IDF` transformer for generic feature matrices.

    This implementation generalizes `TF-IDF` beyond text count matrices and allows configurable strategies for `TF` normalization, `DF` estimation, `IDF` computation, and output normalization. It can be applied to arbitrary non-negative feature matrices where rows represent samples and columns represent features.

    Examples
    --------
    >>> from cobra_algo.text import CountVectorizer, GeneralizedTFIDF
    >>> X = np.array([
    >>>     [0.1, 1.2, 0.3],
    >>>     [0.0, 0.5, 1.0],
    >>>     [0.5, 0.5, 0.0],
    >>> ], dtype=float)
    ...
    >>> X = CountVectorizer("matrix").fit_transform(X).result
    ...
    >>> tfidf = GeneralizedTFIDF(
    >>>     tf_mode="raw",
    >>>     idf_mode="log_smooth",
    >>>     norm=None,
    >>>     df_mode="zero"
    >>> )
    >>> tfidf_result = tfidf.fit_transform(X)
    ...
    >>> tfidf_result.result.to_numpy()
    [[0.12876821 1.2        0.38630462]
    [0.         0.5        1.28768207]
    [0.64384104 0.5        0.        ]]
    ...
    >>> tfidf_result.idf
    [1.28768207 1.         1.28768207]
    ...
    >>> tfidf_result.vocab
    3
    """
    def __init__(
        self,
        tf_mode: Literal["raw", "norm", "log"] = "norm",
        use_idf: bool = True,
        df_mode: Union[Literal["zero", "mean"], float] = "zero",
        idf_mode: Literal["log", "log_smooth"] = "log_smooth",
        norm: Optional[Literal["l1", "l2"]] = None,
    ):
        """
        Initialize `GeneralizedTFIDF`.

        Notes
        -----
        - All parameters follow the usage conventions of :func:`tf_idf`.
        """
        # check parameters
        if df_mode not in ("zero", "mean") and not isinstance(df_mode, float):
            raise ValueError(f"`df_mode` parameter must be either 'zero', 'mean' or a float value in [0, 1], got {df_mode!r}.")
        self._df_mode = df_mode
        if tf_mode not in ("raw", "norm", "log"):
            raise ValueError(f"`tf_mode` parameter must be either 'raw', 'norm' or 'log', got {tf_mode!r}.")
        self._tf_mode = tf_mode
        self._use_idf = bool(use_idf)
        if idf_mode not in ("log", "log_smooth"):
            raise ValueError(f"`idf_mode` parameter must be either 'log' or 'log_smooth', got {idf_mode!r}.")
        self._idf_mode = idf_mode
        if norm not in ("l1", "l2", None):
            raise ValueError(f"`norm` parameter must be either 'l1', 'l2' or None, got {norm!r}.")
        self._norm = norm

        self._idf: Optional[CompatArray[type[float], AnyDevice]] = None

    def fit(self, X: ArrayLike[Any], /) -> GeneralizedTFIDF:
        """
        Calculate and cache `IDF` via the feature matrix from all samples.

        Parameters
        ----------
            X : ArrayLike[Any]
                The feature matrix from all samples to calculate `IDF`, shape as `[num_all_samples, num_features]`. Must be `non-negative`.

        Returns
        -------
            GeneralizedTFIDF

        Raises
        ------
            TypeError
                If `X` is not array-like.
            DimensionError
                If `X` is not a 2-D array.
            IDFError
                If there is an error during `IDF` calculation.
        """
        try:
            _X = CompatArray(X)
        except Exception:
            raise TypeError(f"`X` parameter of `GeneralizedTFIDF.fit()` must be array-like, got {type(X).__name__!r}.") from None

        if _X.ndim != 2:
            raise DimensionError(f"`X` parameter of `GeneralizedTFIDF.fit()` must be a 2-D array, got {_X.ndim}-D")

        try:
            df = self._compute_df(_X)
            self._idf = self._compute_idf(df, len(_X))
            return self
        except Exception as e:
            raise IDFError("IDF calculation error.") from e

    def transform(self, X: ArrayLike[Any], /) -> TFIDFResult:
        """
        Transform the feature matrix from specific documents to `TF-IDF` matrix using the calculated `IDF`.

        Parameters
        ----------
            X : ArrayLike[Any]
                The feature matrix from specific samples to transform, shape as `[num_samples, num_features]`. Must be `non-negative` and have the same number of features as the feature matrix used in :meth:`GeneralizedTFIDF.fit`.

        Returns
        -------
            TFIDFResult
                - values(_CompatArray_): The TF-IDF matrix, shape as `[num_samples, num_features]`;
                - idf(_CompatArray_): The calculated IDF values used for transformation, shape as `[num_features]`.

        Raises
        ------
            NotFittedError
                If `IDF` has not been calculated. Call :meth:`GeneralizedTFIDF.fit` first.
            TFIDFError
                If there is an error during `TF-IDF` calculation.
        """
        try:
            _X = CompatArray(X)
        except Exception:
            raise TypeError(f"`X` parameter of `GeneralizedTFIDF.transform()` must be array-like, got {type(X).__name__!r}.") from None

        if _X.ndim != 2:
            raise DimensionError(f"`X` parameter of `GeneralizedTFIDF.transform()` must be a 2-D array, got {_X.ndim}-D")

        if self._use_idf:
            if self._idf is None:
                raise NotFittedError("IDF is not calculated. Call `GeneralizedTFIDF.fit()` first.")
            if _X.shape[1] != len(self._idf):
                raise DimensionError(f"The number of features in `X` parameter must match the length of calculated `IDF`, got {_X.shape[1]} and {len(self._idf)}.")

        try:
            result = self._compute_tf(_X)
            if self._use_idf and self._idf is not None:
                result = result * self._idf

            return TFIDFResult(self._normalize(result), self._idf)

        except Exception as e:
            raise TFIDFError("TF-IDF calculation error.") from e

    def fit_transform(self, X: ArrayLike[Any], /) -> TFIDFResult:
        """
        Fit the feature matrix from all samples, then transform it to `TF-IDF` matrix.

        Parameters
        ----------
            X : ArrayLike[Any]
                The feature matrix from all samples to fit and transform to `TF-IDF` matrix, shape as `[num_all_samples, num_features]`. Must be `non-negative`.

        Returns
        -------
            TFIDFResult
                - values(_CompatArray_): The TF-IDF matrix, shape as `[num_all_samples, num_features]`;
                - idf(_CompatArray_): The calculated IDF values used for transformation, shape as `[num_features]`.
        """
        return self.fit(X).transform(X)

    def _compute_tf(self, X: CompatArray[type[float], AnyDevice]) -> CompatArray[type[float], AnyDevice]:
        """Calculate Term Frequency (TF) for each term in the sample set based on the feature matrix X.
        Result a float array shape as `[num_samples, num_features]`."""
        if self._tf_mode == "norm":
            # norm mode
            row_sum = X.sum(axis=1, keepdims=True)
            return X / row_sum.maximum(EPS)
        elif self._tf_mode == "log":
            # log mode
            return X.log1p()  # log(1 + X)
        else:
            # raw mode
            return X

    def _compute_df(self, X: CompatArray[type[float], AnyDevice]) -> CompatArray[type[int], AnyDevice]:
        """Calculate Document Frequency (DF) for each term in the sample set.
        Return a float array shape as `[num_features]`."""
        if isinstance(self._df_mode, float):
            # percentile mode
            raise NotImplementedError("Percentile-based DF calculation is not implemented yet.")
            threshold = np.quantile(X, self._df_mode, axis=0)
            return np.count_nonzero(X >= threshold, axis=0)
            (X >= threshold).count_nonzero(axis=0)
        elif self._df_mode == "zero":
            # zero mode
            return (X > 0).count_nonzero(axis=0)
        else:
            # mean mode
            row_mean = X.mean(axis=1, keepdims=True)
            return (X >= row_mean).count_nonzero(axis=0)

    def _compute_idf(self, df: CompatArray[type[int], AnyDevice], N: int) -> CompatArray[type[float], AnyDevice]:
        """Calculate Inverse Document Frequency (IDF) for each term based on the document frequency (DF) and total number of samples (N).
        Return a float array shape as `[num_features]`."""
        N_arr = df.cxp.asarray(N, dtype=float, device=df.device)
        if self._idf_mode == "log":
            # log mode
            return (N_arr / df.maximum(EPS)).log()
        else:
            # log_smooth mode
            return ((N_arr + 1) / (df + 1)).log() + 1

    def _normalize(self, tfidf: CompatArray[type[float], AnyDevice]) -> CompatArray[type[float], AnyDevice]:
        """Normalize the TF-IDF matrix using the specified norm (L1 or L2)."""
        if self._norm is None:
            return tfidf

        if self._norm == "l2":
            norms = tfidf.cxp.vector_norm(tfidf, axis=1, keepdims=True)
        else:
            norms = tfidf.abs().sum(axis=1, keepdims=True)
        return tfidf / norms.maximum(EPS)

    @property
    def idf(self) -> Optional[CompatArray[type[float], AnyDevice]]:
        """
        The calculated IDF values for the features. Shape as `[num_features]`.
        """
        return self._idf
