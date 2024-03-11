import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import convolve

# 与matlab输出一致，这里rank_mask输出和MASK是一模一样的，所以直接去掉了
def binary_dilate(IM, MASK, RK=1):
    if IM is None:
        print('dilated_im = binary_dilate(IM, MASK, <1|RK>)')
        print('\tbased on:')
        print('rank_mask ?')
        return

    if IM is None:
        return IM

    if RK is None:
        RK = 1

    mask = MASK
    mvals = np.unique(mask)

    if RK == 1 and (len(mvals) == 1 and mvals[0] > 0):
        try:
            dilated = binary_dilation(np.array(IM, dtype=bool), structure=mask != 0)
            return dilated.astype(int)
        except:
            pass

    if RK == 1:
        result = convolve(IM, np.flipud(np.fliplr(mask)), mode='same')
    else:
        result = convolve(IM-RK,np.flipud(np.fliplr(mask)), mode='same')+RK * np.sum(mask)

    dilated = result >= (1 - RK + np.finfo(float).eps) / (1 + 2 * np.finfo(float).eps) * np.sum(mask)
    return dilated.astype(int)



# # Test the function
# IM = np.array([[0, 0, 0, 1, 0, 0, 0],
#                [0, 0, 0, 1, 0, 0, 0],
#                [0, 0, 0, 1, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0],
#                [0, 0, 0, 0, 0, 0, 0]])
# MASK = np.array([[0, 1, 0],
#                  [1, 1, 1],
#                  [0, 1, 0]])
# RK = 0.5
#
# dilated_im = binary_dilate(IM, MASK,RK)
# print(dilated_im)