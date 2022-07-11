import numpy as np


def nme(pred, truth):
    a = np.sum(np.abs(pred - truth))
    b = np.sum(truth)
    return a/b


def mpe(pred, truth):
    pe = np.abs(pred - truth) / np.abs(truth)
    mpe = np.mean(pe)
    return mpe


def test_nme():
    assert nme(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0, "success"


def test_mpe():
    assert mpe(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0, "success"
