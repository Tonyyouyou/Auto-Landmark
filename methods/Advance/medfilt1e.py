import numpy as np
from scipy.ndimage import generic_filter

# 输出与matlab不太一致
def medfilt1e(x, n=3, blksz=None):
    if isinstance(x, str) and x == '?':
        print('y = medfilt1e(x, <3|n>, <0|blksz|Inf>)')
        print('ycols = medfilt1e(xcols, <3|n>, <0|blksz|Inf>)')
        return

    if x.size == 0:
        return x

    if n is None:
        n = 3

    if n == 1:
        return x

    x = np.squeeze(x)
    nx = x.shape[0]
    nx2 = x.size // nx

    if blksz is None or blksz == 0:
        tb = x.itemsize
        memlimit = 2**64  # You can adjust this based on available memory
        blksz = max(1, np.ceil((memlimit - 2 * tb * x.size) / (2 * tb * nx2 + 5 * 4 * n) / 50))

    elif np.isinf(blksz):
        blksz = x.shape[0]

    if n % 2 != 1:
        m = n // 2
    else:
        m = (n - 1) // 2

    if nx >= n:
        try:
            y = of1_recurse(x, n)
        except:
            if x.dtype == bool:
                y = np.full((1, nx, nx2), False)
            else:
                y = np.full((1, nx, nx2), np.nan)

            X = np.concatenate([y[:, :m, :nx2], x.reshape((1, nx, nx2)), y[:, -m:, :nx2]], axis=1)

            if x.dtype == bool:
                y = np.full((1, nx, nx2), False)
            else:
                y = np.zeros((1, nx, nx2), dtype=x.dtype)

            indr = np.arange(n)
            indc = np.arange(1, nx + 1)
            for i in range(0, nx, blksz):
                nxi = min(i + blksz, nx)
                ind = (indr[:, np.newaxis] + indc[np.newaxis, i:nxi])
                xx = X[0, ind, :].reshape((n, nxi - i + 1, nx2))
                y[0, i:nxi, :] = np.median(xx, axis=0)

    else:
        y = x

    y = y.reshape(x.shape)

    for k in range(1, min(m, int(np.ceil(len(x) / 2))) + 1):
        y[k, :] = np.median(x[0:2 * k, :], axis=0)
        y[-k, :] = np.median(x[-(2 * k):, :], axis=0)

    return y

# 输出等于ordfilt2 in matlab
def of1_recurse(x, n):
    try:
        # Python中的这个函数，是padding从左边和上边加一排0，跟matlab的区别是，ordfilt2是右边和下边加一排0。其他内部计算方法是一样的，影响不大。
        y = generic_filter(x, lambda m: local_filter(m, int((n - 1) / 2)), footprint=np.ones((n, 1)), mode='constant')
    except MemoryError as err:
        if x.shape[0] > 2 * n:
            sx1 = x.shape[0]
            half_ndx = np.ceil(sx1 / 2).astype(int)

            y1 = of1_recurse(x[:half_ndx + np.ceil(n / 2).astype(int), :], n)
            y1 = y1[:half_ndx, :]

            y2 = of1_recurse(x[half_ndx - np.ceil(n / 2).astype(int):, :], n)
            y2 = y2[np.ceil(n / 2).astype(int):, :]

            y = np.vstack((y1, y2))
            del y1, y2
        else:
            raise err
    return y


def local_filter(x, order):
    x.sort()
    return x[order]

# # test
# # Specify the window size
# n = 3
#
# # Call the of1_recurse function
# I = np.array([[1,2,4,5],[5,3,5,1],[0,3,5,2],[2,1,7,7]])
# # result = generic_filter(I,np.nanmin,footprint=np.ones((2, 2)),mode='constant')
# result = generic_filter(I, lambda m: local_filter(m, int((n - 1) / 2)), footprint=np.ones((n, 1)), mode='constant')
# # Display the original input and the filtered result
# print("Original input:\n", I)
# print("\nFiltered result:\n", result)
