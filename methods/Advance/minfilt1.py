import numpy as np
from scipy.ndimage import maximum_filter1d
from maxfilt1 import maxfilt1

# result 与matlab完全一致
def minfilt1(x, n=3, blksz=None):
    if n == 1 or x.size == 0:
        return x

    if blksz is None:
        blksz = x.shape[0]  # Set block size to the number of rows

    switch_class = {
        bool: lambda x: ~maxfilt1(~x, n).astype(bool),
        (float, np.float32, np.float64): lambda x: -maxfilt1(-x, n),
    }

    x_class = x.dtype.type

    if x_class in switch_class:
        y = switch_class[x_class](x)
    else:
        mx = np.finfo(x_class).max
        if np.finfo(x_class).min == 0:
            y = mx - maxfilt1(mx - x, n)
        else:
            y = -1 - maxfilt1(-1 - x, n)
        if y.dtype != x.dtype:
            print('MINFILT1 unexpected class result.')
            y = y.astype(x.dtype)
            print('Continuing with correct class.')

    return y


# x1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]],dtype=np.float64)
# n1 = 2
# result1 = minfilt1(x1, n1)
# print(result1)
# x2 = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]],dtype=np.float64)
# n2 = 3
# result2 = minfilt1(x2, n2)
# print(result2)