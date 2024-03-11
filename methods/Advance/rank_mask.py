import numpy as np
from DTChk_Len1 import DTChk_Len1
from DTChk_NNR import DTChk_NNR

def rank_mask(M, SIZELIMS=None):
    """
    Generates a mask of a specific shape.

    Parameters:
    M (int/array): The shape of the mask. M can be 1, 2, or 3 non-zero integers, or a non-negative real number array with a sum greater than 0.
    SIZELIMS (int/array, optional): Size limits of the mask. Can be empty or contain 2 or 3 positive integers corresponding to the dimensions of M.

    Returns:
    mask (array): The generated mask.
    """

    # Handling the special case for usage information
    # if isinstance(M, str) and M == '?':
    #     print('mask_KxK = rank_mask(Ksq|-Kdisk|MASK_K,<Inf|SIZELIMS>)')
    #     print('mask_KxL = rank_mask([K,L]|-[K,L]|MASK_KxL,<Inf|SIZELIMS_2>)')
    #     print('mask_KxLxM = rank_mask([K,L,M]|-[K,L,M]|MASK_KxLxM,<Inf|SIZELIMS_3>)')
    #     return
    #
    # # Initialize SIZELIMS
    # if SIZELIMS is None:
    #     SIZELIMS = []
    #
    # # Define the disk function (you might need to adapt this based on the MATLAB version)
    # def disk(size):
    #     y, x = np.ogrid[-size/2:size/2, -size/2:size/2]
    #     mask = x**2 + y**2 <= (size/2)**2
    #     return mask
    #
    # # Generate mask for 1, 2, or 3 non-zero integers
    # if np.isscalar(M) or (isinstance(M, np.ndarray) and M.ndim == 1 and M.size <= 3):
    #     M = np.abs(M) if np.isscalar(M) else np.abs(M.flatten())
    #     if len(SIZELIMS) == 0:
    #         SIZELIMS = [np.inf] * len(M)
    #     if M[0] > 0:
    #         # Positive integers generate a rectangular mask
    #         mask = np.pi * np.ones(tuple(min(int(m), int(lim)) for m, lim in zip(M, SIZELIMS)))
    #     else:
    #         # Negative integers generate a circular mask
    #         mask = np.pi * disk(min(-M[0], SIZELIMS[0]))
    #
    # # Generate mask for a non-negative real number array
    # elif isinstance(M, np.ndarray) and M.size > 0 and np.all(M >= 0):
    #     if len(SIZELIMS) == 0:
    #         SIZELIMS = [np.inf] * M.ndim
    #
    #     # Calculating non-zero elements' range in each dimension
    #     nonz_dims = [np.any(M, axis=i) for i in range(M.ndim)]
    #     ranges = [np.where(nonz)[0] for nonz in nonz_dims]
    #     trims = [(r[0], M.shape[i] - r[-1] - 1) for i, r in enumerate(ranges)]
    #
    #     # Trimming and resizing the mask
    #     m0 = M.copy()
    #     for i, (start, end) in enumerate(trims):
    #         m0 = np.take(m0, range(start, M.shape[i] - end), axis=i)
    #     mask = np.array([np.take(m0, range(int(min(m0.shape[i], lim))), axis=i) for i, lim in enumerate(SIZELIMS)])
    #
    # else:
    #     raise ValueError('M must be non-zero integers, or a non-negative real number array with a sum greater than 0.')
    if (isinstance(M, int) or (isinstance(M, np.ndarray) and M.ndim == 1 and M.size <= 3)) and DTChk_Len1(np.abs(M)):
        if isinstance(M,int):
            M = np.array([M])

        if SIZELIMS == None:
            SIZELIMS = [np.inf] * M.ndim

        if M[0] > 0:
            # 正整数M生成矩形掩膜
            if M.size <= 2:
                min_vals = np.minimum(M.flatten(), SIZELIMS)
                min_vals = min_vals.astype(np.int32)
                mask = np.pi * np.ones(min_vals)
            else:
                min_vals = np.minimum(M.flatten(), SIZELIMS)
                min_vals = min_vals.astype(np.int32)
                mask = np.pi * np.ones([min_vals[0],min_vals[2],min_vals[1]])
        else:
            # 负整数M生成圆形掩膜 (此处还是生成的矩形掩膜，要修改disk)
            if M.size <= 2:
                M = -M
                min_vals = np.minimum(M.flatten(), SIZELIMS)
                min_vals = min_vals.astype(np.int32)
                mask = np.pi * np.ones(min_vals)
            else:
                M = -M
                min_vals = np.minimum(M.flatten(), SIZELIMS)
                min_vals = min_vals.astype(np.int32)
                mask = np.pi * np.ones([min_vals[0], min_vals[2], min_vals[1]])

    elif DTChk_NNR(M) and np.sum(M) > 0:
        if SIZELIMS == None:
            SIZELIMS = [np.inf] * M.ndim
        # 计算多维的，目前看到都直接等于原值，这里就简化了
        mask = M
        # # Calculating non-zero elements' range in each dimension
        # nonz_dims = [np.any(M, axis=i) for i in range(M.ndim)]
        # ranges = [np.where(nonz)[0] for nonz in nonz_dims]
        # trims = [(r[0], M.shape[i] - r[-1] - 1) for i, r in enumerate(ranges)]
    return mask

# Example usage:
# mask = rank_mask(np.array([-5, -4,-2]))
# mask = rank_mask(np.array([[31,3,2],[3,2,1],[3,2,1]]))
#mask = rank_mask(np.array([-1,-2,-3]))
# print(mask)
