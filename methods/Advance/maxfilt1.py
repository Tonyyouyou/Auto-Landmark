import numpy as np
from scipy.ndimage import maximum_filter1d

# 输出结果与matlab完全一致
def maxfilt1(x, n=3, blksz=None):

    x = x.T
    if n == 1 or x.size == 0:
        return x

    sz = x.shape
    x = np.swapaxes(x, 0, -1)
    nx = x.shape[0]

    if n == 0 or n >= nx:
        y = np.reshape(np.tile(np.max(x, axis=0), [nx, 1]), sz)
        return y

    nx2 = x.size // nx
    if blksz is None:
        blksz = nx
    elif blksz == 0:
        blksz = 1
    elif np.isinf(blksz):
        blksz = nx

    m = n // 2 if n % 2 == 0 else (n - 1) // 2

    try:
        if np.issubdtype(x.dtype, np.floating) and np.any(np.isnan(x)):
            mnval = np.float('-inf')
            X = np.reshape(np.where(np.isnan(x), mnval, x), x.shape)

            if np.issubdtype(x.dtype, bool):
                y = maximum_filter1d((X > 0).astype(np.int8), size=n, axis=0).astype(bool)
            else:
                y = maximum_filter1d(X, size=n, axis=0)

        else:
            if np.issubdtype(x.dtype, bool):
                y = maximum_filter1d(x.astype(np.int8), size=n, axis=0).astype(bool)
            else:
                y = maximum_filter1d(x, size=n, axis=0)

        y = np.reshape(y, sz)
        return y

    except:
        pass

    if np.max(np.unique(x).size) <= 2:
        xvals = np.unique(x)
        y = x - xvals[0]
        y = xvals[0] + np.convolve(y > 0, np.ones(n, dtype=int), mode='same')[::n]
        y = np.reshape(y, sz)
        return y

    mnval = np.float('-inf') if np.issubdtype(x.dtype, np.floating) else np.iinfo(x.dtype).min

    X = np.concatenate((np.tile(mnval, [1, m, nx2]), np.reshape(x, [1, nx, nx2]), np.tile(mnval, [1, m, nx2])), axis=1)

    if np.issubdtype(X.dtype, bool):
        X = X.astype(np.uint8)

    y = np.zeros((1, nx, nx2), dtype=X.dtype)

    indr = np.arange(n, dtype=np.uint32)
    indc = np.arange(1, nx + 1, dtype=np.uint32)

    for i in range(0, nx, blksz):
        nxi = min(i + blksz, nx)
        ind = np.add.outer(indc - 1, indr)[:n, :nxi - i]
        xx = np.reshape(X[0, ind, :], (n, nxi - i, nx2))
        y[0, i:nxi, :] = np.max(xx, axis=0)

    y = np.reshape(y, sz)

    if np.issubdtype(x.dtype, bool):
        y = y.astype(bool)

    return y

# x1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# n1 = 2
# result1 = maxfilt1(x1, n1)
# print(result1)
# x2 = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
# n2 = 3
# result2 = maxfilt1(x2, n2)
# print(result2)