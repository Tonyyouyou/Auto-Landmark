# daijiao
import numpy as np


def DTChk_NNR(A, oknans=False):
    if not oknans:
        # Check if A is a non-negative real numeric array
        okay = np.all(np.isreal(A)) and np.all(A >= 0)
    else:
        # Check if A contains NaN or real numeric values that are non-negative
        okay = np.all(np.isnan(A) | (np.isreal(A) & (A >= 0)))

    return okay