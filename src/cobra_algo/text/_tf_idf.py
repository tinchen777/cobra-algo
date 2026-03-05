# -*- coding: utf-8 -*-
# Python version: 3.9
# @TianZhen

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from collections import Counter
from typing import (Any, Dict, Literal, Union, Optional)

from .._utils import to_numpy
from ..exceptions import (
    NotFittedError,
    DimensionError,
    IDFError,
    TFIDFError,
    VocabularyError,
    CountVectorizerError
)

EPS = 1e-12


def tf_idf(
    D: Any,
    d: Optional[Any] = None,
    input_mode: Literal["tokens", "matrix"] = "matrix",
    tf_mode: Literal["raw", "norm", "log"] = "norm",
    df_mode: Union[Literal["zero", "mean"], float] = "zero",
    idf_mode: Literal["log", "log_smooth"] = "log_smooth",
    norm: Optional[Literal["l1", "l2"]] = None
):
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
        NDArray[np.float_]
            The calculated `TF-IDF` matrix, shape as `[num_samples, num_features]`.
    """
    pipe = CountVectorizer(input_mode)
    transformer = GeneralizedTFIDF(tf_mode, df_mode, idf_mode, norm)

    X = pipe.fit_transform(D)
    transformer.fit(X)
    if d is None:
        return transformer.transform(X)
    return transformer.transform(pipe.transform(d))


class CountVectorizer:
    """
    Count Vectorizer for converting documents to a matrix of token counts.
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

    def fit(self, docs: Any) -> CountVectorizer:
        """
        Build vocabulary from documents.

        Parameters
        ----------
            docs : Sequence[Any]
                A list of documents to build the vocabulary.
                - _Sequence[Sequence[Any]]_: Each document is a list of tokens when `input_mode` is `"tokens"`;
                - _MatrixLike_: All documents are represented as a matrix when `input_mode` is `"matrix"`.

        Returns
        -------
            CountVectorizer

        Raises
        ------
            DimensionError
                If the input documents have invalid dimensions for the specified `input_mode`.
            VocabularyError
                If there is an error during building vocabulary.
        """
        try:
            if self._input_mode == "tokens":
                # input as tokens
                terms = sorted(set(term for doc in docs for term in doc))
                self._vocab = {term: i for i, term in enumerate(terms)}
            else:
                # input as matrix
                docs_arr = to_numpy(docs, copy=False)
                if docs_arr.ndim != 2:
                    raise DimensionError(f"`docs` parameter of `CountVectorizer.fit()` must be a 2-D array when `input_mode` is 'matrix', got {docs_arr.ndim}-D")
                self._vocab = docs_arr.shape[1]
            return self

        except Exception as e:
            raise VocabularyError("Building vocabulary error.") from e

    def transform(self, docs: Any) -> NDArray[np.float_]:
        """
        Transform documents to count matrix using the built vocabulary.

        Parameters
        ----------
            docs : Sequence[Any]
                A list of documents to transform based on the built vocabulary.
                - _Sequence[Sequence[Any]]_: Each document is a list of tokens when `input_mode` is `"tokens"`;
                - _MatrixLike_: All documents are represented as a matrix when `input_mode` is `"matrix"`.

        Returns
        -------
            NDArray[np.float_]
                Count matrix, shape as `[num_docs, num_vocabulary]`.

        Raises
        ------
            NotFittedError
                If the vocabulary has not been built. Call :meth:`CountVectorizer.fit` first.
            DimensionError
                If the input documents have invalid dimensions for the specified `input_mode`.
            CountVectorizerError
                If there is an error during count vectorization.
        """
        if not self._vocab:
            raise NotFittedError("Vocabulary is empty. Call `CountVectorizer.fit()` first.")

        try:
            if isinstance(self._vocab, int):
                # input as matrix
                docs_arr = to_numpy(docs, copy=False)
                if docs_arr.ndim != 2:
                    raise DimensionError(f"`docs` parameter of `CountVectorizer.transform()` must be a 2-D array when `input_mode` is 'matrix', got {docs_arr.ndim}-D")
                out = np.zeros((len(docs_arr), self._vocab), dtype=float)

                c = min(docs_arr.shape[1], self._vocab)
                out[:, :c] = docs_arr[:, :c]
                return out
            # input as tokens
            X = np.zeros((len(docs), len(self._vocab)), dtype=float)
            for i, doc in enumerate(docs):
                term_counts = Counter(doc)
                for term, count in term_counts.items():
                    if term in self._vocab:
                        X[i, self._vocab[term]] = count
            return X

        except Exception as e:
            raise CountVectorizerError("Count vectorization error.") from e

    def fit_transform(self, docs: Any) -> NDArray[np.float_]:
        """
        Build vocabulary from documents, then transform documents to count matrix.

        Parameters
        ----------
            docs : Sequence[Any]
                A list of documents to build the vocabulary and transform based on the built vocabulary.
                - _Sequence[Sequence[Any]]_: Each document is a list of tokens when `input_mode` is `"tokens"`;
                - _MatrixLike_: All documents are represented as a matrix when `input_mode` is `"matrix"`.

        Returns
        -------
            NDArray[np.float_]
                Count matrix, shape as `[num_docs, num_vocabulary]`.
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
    """
    def __init__(
        self,
        tf_mode: Literal["raw", "norm", "log"] = "norm",
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
            raise ValueError(f"`df_mode` parameter must be either 'zero', 'mean' or a float value for percentile, got {df_mode!r}.")
        self._df_mode = df_mode
        if tf_mode not in ("raw", "norm", "log"):
            raise ValueError(f"`tf_mode` parameter must be either 'raw', 'norm' or 'log', got {tf_mode!r}.")
        self._tf_mode = tf_mode
        if idf_mode not in ("log", "log_smooth"):
            raise ValueError(f"`idf_mode` parameter must be either 'log' or 'log_smooth', got {idf_mode!r}.")
        self._idf_mode = idf_mode
        if norm not in ("l1", "l2", None):
            raise ValueError(f"`norm` parameter must be either 'l1', 'l2' or None, got {norm!r}.")
        self._norm = norm

        self._idf: Optional[NDArray[np.float_]] = None

    def fit(self, X: NDArray[np.float_]) -> GeneralizedTFIDF:
        """
        Calculate and cache `IDF` via the feature matrix from all samples.

        Parameters
        ----------
            X : NDArray[np.float_]
                The feature matrix from all samples to calculate `IDF`, shape as `[num_all_samples, num_features]`. Must be `non-negative`.

        Returns
        -------
            GeneralizedTFIDF

        Raises
        ------
            IDFError
                If there is an error during `IDF` calculation.
        """
        try:
            if X.ndim != 2:
                raise DimensionError(f"`X` parameter of `GeneralizedTFIDF.fit()` must be a 2-D array, got {X.ndim}-D")

            df = self._compute_df(X)
            self._idf = self._compute_idf(df, len(X))
            return self
        except Exception as e:
            raise IDFError("IDF calculation error.") from e

    def transform(self, X: NDArray[np.float_]) -> NDArray[np.float_]:
        """
        Transform the feature matrix from specific documents to `TF-IDF` matrix using the calculated `IDF`.

        Parameters
        ----------
            X : NDArray[np.float_]
                The feature matrix from specific samples to transform, shape as `[num_samples, num_features]`. Must be `non-negative` and have the same number of features as the feature matrix used in :meth:`GeneralizedTFIDF.fit`.

        Returns
        -------
            NDArray[np.float_]
                TF-IDF matrix, shape as `[num_samples, num_features]`.

        Raises
        ------
            NotFittedError
                If `IDF` has not been calculated. Call :meth:`GeneralizedTFIDF.fit` first.
            TFIDFError
                If there is an error during `TF-IDF` calculation.
        """
        if self._idf is None:
            raise NotFittedError("IDF is not calculated. Call `GeneralizedTFIDF.fit()` first.")
        try:
            if X.ndim != 2:
                raise DimensionError(f"`X` parameter of `GeneralizedTFIDF.transform()` must be a 2-D array, got {X.ndim}-D")
            if X.shape[1] != len(self._idf):
                raise DimensionError(f"The number of features in `X` parameter must match the length of calculated `IDF`, got {X.shape[1]} and {len(self._idf)}.")

            tf = self._compute_tf(X)
            tfidf = tf * self._idf

            return self._normalize(tfidf)
        except Exception as e:
            raise TFIDFError("TF-IDF calculation error.") from e

    def fit_transform(self, X: NDArray[np.float_]) -> NDArray[np.float_]:
        """
        Fit the feature matrix from all samples, then transform it to `TF-IDF` matrix.

        Parameters
        ----------
            X : NDArray[np.float_]
                The feature matrix from all samples to fit and transform to `TF-IDF` matrix, shape as `[num_all_samples, num_features]`. Must be `non-negative`.

        Returns
        -------
            NDArray[np.float_]
                TF-IDF matrix, shape as `[num_all_samples, num_features]`.
        """
        return self.fit(X).transform(X)

    def _compute_tf(self, X: NDArray[np.float_]) -> NDArray[np.float_]:
        """Calculate Term Frequency (TF) for each term in the sample set based on the feature matrix X.
        Result a float array shape as `[num_samples, num_features]`."""
        if self._tf_mode == "norm":
            # norm mode
            row_sum = X.sum(axis=1, keepdims=True)
            row_sum = np.maximum(row_sum, EPS)
            return X / row_sum
        elif self._tf_mode == "log":
            # log mode
            return np.log(1 + X)
        else:
            # raw mode
            return X

    def _compute_df(self, X: NDArray[np.float_]) -> NDArray[np.float_]:
        """Calculate Document Frequency (DF) for each term in the sample set.
        Return a float array shape as `[num_features]`."""
        if isinstance(self._df_mode, float):
            # percentile mode
            threshold = np.quantile(X, self._df_mode, axis=0)
            return np.count_nonzero(X >= threshold, axis=0)
        elif self._df_mode == "zero":
            # zero mode
            return np.count_nonzero(X > 0, axis=0)
        else:
            # mean mode
            row_mean = X.mean(axis=1, keepdims=True)
            return np.count_nonzero(X >= row_mean, axis=0)

    def _compute_idf(self, df: NDArray[np.float_], N: int) -> NDArray[np.float_]:
        """Calculate Inverse Document Frequency (IDF) for each term based on the document frequency (DF) and total number of samples (N).
        Return a float array shape as `[num_features]`."""
        if self._idf_mode == "log":
            # log mode
            df = np.maximum(df, EPS)
            return np.log(N / df)
        else:
            # log_smooth mode
            return np.log((1 + N) / (1 + df)) + 1  # type:ignore

    def _normalize(self, tfidf: NDArray[np.float_]) -> NDArray[np.float_]:
        """Normalize the TF-IDF matrix using the specified norm (L1 or L2)."""
        if self._norm is None:
            return tfidf

        if self._norm == "l2":
            norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        else:
            norms = np.sum(np.abs(tfidf), axis=1, keepdims=True)
        norms = np.maximum(norms, EPS)
        return tfidf / norms

    @property
    def idf(self) -> Optional[NDArray[np.float_]]:
        """
        The calculated IDF values for the features. Shape as `[num_features]`.
        """
        return self._idf
