# tests/test_tf_idf.py

import sys
sys.path.insert(0, "/data/tianzhen/my_packages/cobra-algo/src")

from cobra_algo.text import CountVectorizer, GeneralizedTFIDF, tf_idf
import numpy as np


def test_count():
    docs = [
        ["apple", "banana", "apple"],
        ["banana", "orange"],
        ["banana", "banana", "apple"]
    ]
    pipe = CountVectorizer("tokens")
    a = pipe.fit_transform(docs)
    print("count matrix:\n", a)


def test_GeneralizedTFIDF():
    print("=== Test TF-IDF Calculation ===")

    tf_idf = GeneralizedTFIDF()
    # ts_arr = np.array(["term1", "term2", "term3"])
    X = np.array([
        [2, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 0, 1]
    ], dtype=float)

    r = tf_idf.fit_transform(X)
    print("TF-IDF:", r)


def test_tf_idf():
    print("=== Test TF-IDF Calculation ===")
    X = np.array([
        [2, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 1, 0, 1]
    ], dtype=float)

    r = tf_idf(X, X[[1, 2]])
    print("TF-IDF:", r)


if __name__ == "__main__":
    test_count()
    test_GeneralizedTFIDF()
    test_tf_idf()
