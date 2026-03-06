# tests/test_tf_idf.py

import sys
sys.path.insert(0, "/data/tianzhen/my_packages/cobra-algo/src")

from cobra_algo.text import CountVectorizer, GeneralizedTFIDF, tf_idf
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer as SkCountVectorizer
import time


def test_compare_with_sklearn_1():
    print("=== Compare with sklearn's TfidfTransformer 1 ===")
    X = np.array([
        [1, 0, 2],
        [0, 1, 1],
        [1, 1, 0]
    ], dtype=float)
    start = time.time()
    my = GeneralizedTFIDF(tf_mode="raw", idf_mode="log_smooth", norm=None, use_idf=True)
    r1 = my.fit_transform(X).to_numpy()
    print(r1)
    print("Time taken:", time.time() - start, "seconds")
    
    start = time.time()
    sk = TfidfTransformer(norm=None, smooth_idf=True, use_idf=True)
    r2 = sk.fit_transform(X).toarray()
    print(r2)
    print("Time taken:", time.time() - start, "seconds")


def test_compare_with_sklearn_2():
    print("=== Compare with sklearn's TfidfTransformer 2 ===")
    X = np.array([
        [0.1, 1.2, 0.3],
        [0.0, 0.5, 1.0],
        [0.5, 0.5, 0.0],
    ], dtype=float)

    # X = np.array([
    #     [0.1, 1.2, 0.3],
    #     [0.2, 0.5, 1.0],
    #     [0.5, 0.5, 0.4],
    # ])

    X = CountVectorizer("matrix").fit_transform(X).values

    start = time.time()

    # r1 = tf_idf(X, tf_mode="raw", idf_mode="log_smooth", norm="l2", df_mode="mean")

    my = GeneralizedTFIDF(tf_mode="raw", idf_mode="log_smooth", norm=None, df_mode="zero", use_idf=True)
    r1 = my.fit_transform(X).to_numpy()
    print(r1)
    print("Time taken:", time.time() - start, "seconds")
    
    start = time.time()

    sk = TfidfTransformer(norm=None, smooth_idf=True, use_idf=True)
    r2 = sk.fit_transform(X).toarray()
    print(r2)
    print("Time taken:", time.time() - start, "seconds")


def test_zero_document():
    print("=== Test with Zero Document ===")
    X = np.array([
        [0, 0, 0],
        [1, 2, 3]
    ], dtype=float)

    tfidf = GeneralizedTFIDF(norm="l2", idf_mode="log_smooth")
    result = tfidf.fit_transform(X).to_numpy()
    assert np.all(np.isfinite(result))
    print("Result with zero document:\n", result)

    sk = TfidfTransformer(norm="l2", smooth_idf=True)
    r2 = sk.fit_transform(X).toarray()
    print("Result with sklearn:\n", r2)


def test_count():
    print("=== Test Count Vectorizer ===")
    docs = [
        ["apple", "banana", "apple"],
        ["banana", "orange"],
        ["banana", "banana", "apple"]
    ]
    start = time.time()
    pipe = CountVectorizer("tokens")
    a = pipe.fit_transform(docs)
    print("count matrix:\n", a.to_numpy())
    print(a.vocab)
    print("Time taken:", time.time() - start, "seconds")
    
    docs = [
        "apple banana apple",
        "banana orange",
        "banana banana apple"
    ]
    start = time.time()
    pipe = SkCountVectorizer()
    a = pipe.fit_transform(docs).toarray()
    print("sklearn count matrix:\n", a)
    print("Time taken:", time.time() - start, "seconds")


def test_tf_idf():
    print("=== Compare with sklearn's TfidfTransformer tf_idf ===")
    X = np.array([
        [0.1, 1.2, 0.3],
        [0.0, 0.5, 1.0],
        [0.5, 0.5, 0.0],
    ], dtype=float)

    # X = np.array([
    #     [0.1, 1.2, 0.3],
    #     [0.2, 0.5, 1.0],
    #     [0.5, 0.5, 0.4],
    # ])

    X = CountVectorizer("matrix").fit_transform(X).values

    start = time.time()

    r1 = tf_idf(X, tf_mode="raw", idf_mode="log_smooth", norm=None, df_mode="zero")

    # my = GeneralizedTFIDF(tf_mode="raw", idf_mode="log_smooth", norm=None, df_mode="zero", use_idf=True)
    # r1 = my.fit_transform(X).to_numpy()
    print(r1.to_numpy())
    print(r1.idf)
    print(r1.vocab)
    
    print("Time taken:", time.time() - start, "seconds")
    
    start = time.time()

    sk = TfidfTransformer(norm=None, smooth_idf=True, use_idf=True)
    r2 = sk.fit_transform(X).toarray()
    print(r2)
    print("Time taken:", time.time() - start, "seconds")


if __name__ == "__main__":
    test_compare_with_sklearn_1()
    test_compare_with_sklearn_2()
    test_zero_document()
    test_count()
    test_tf_idf()
