import numpy as np
# 有问题
def disk(J, K=None, L=None):
    if J == '?':
        print("Usage: mask = disk(J, K, <1|L>)")
        print("Usage: mask = disk([J, K, <1|L>])")
        return

    if J is None:
        return None

    if isinstance(J, (list, np.ndarray)):
        if len(J) == 3:
            J, K, L = J[0], J[1], J[2]
        elif len(J) == 2:
            J, K, L = J[0], J[1], 1
        elif len(J) == 1:
            K, L = J[0], 1
        else:
            raise ValueError("Invalid number of elements in the argument list.")

    if K is None:
        K = J

    J = int(J)
    K = int(K)
    L = int(L)

    xc = (J + 1) / 2
    yc = (K + 1) / 2
    zc = (L + 1) / 2
    xx = (np.arange(1, J + 1) - xc) / (J / 2)
    yy = (np.arange(1, K + 1) - yc) / (K / 2)
    zz = ((np.arange(1, L + 1) - zc) / (L / 2)).reshape(1, 1, L)

    if np.isnan(J) or np.isnan(K) or np.isnan(L):
        mask = np.zeros((J, K, L), dtype=bool)
    else:
        print(np.tile((yy[np.newaxis, np.newaxis,:] ** 2).reshape(-1, 1), (J, 1, L)))
        # mask = (np.tile((xx[:, np.newaxis, np.newaxis] ** 2).reshape(-1, 1), (1, K, L)) +
        #         np.tile((yy[np.newaxis, :, np.newaxis] ** 2).reshape(-1, 1), (J, 1, L)) +
        #         np.tile((zz[np.newaxis, np.newaxis, :] ** 2).reshape(-1, 1), (J, K, 1)) <= 1)
    #return mask

# Example usage
mask = disk([5, 6, 3])
#
# for i in range(mask.shape[2]):
#     print(f"ans(:,:,{i+1}) =\n")
#     print(mask[:, :, i].astype(int))
#     print()