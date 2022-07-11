import numpy as np




def d(pred, truth):
    a = (pred - truth) ** 2
    b = np.abs(pred - np.mean(truth))
    c = np.abs(truth - np.mean(truth))
    return 1 - (np.sum(a) / np.sum((b + c) ** 2))


def dr(pred, truth):
    a = np.sum(np.abs(pred - truth))
    b = 2 * np.sum(np.abs(truth - truth.mean()))

    if a <= b:
        return 1 - (a / b)
    return (b / a) - 1



def test_dr():
    assert dr(np.array([1, 2, 3]), np.array([1, 2, 3])) == 1, "success"
