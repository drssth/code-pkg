import numpy as np


def nmb(pred, truth):
    a = np.sum(pred - truth)
    b = np.sum(truth)
    return a/b


def mpb(pred, truth):
    pb = (pred - truth) / truth
    mpb = np.mean(pb)
    return mpb



def test_nmb():
    assert nmb(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0, "success"


def test_mpb():
    assert mpb(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0, "success"
