import numpy as np
from rank_mask import rank_mask
from binary_dilate import binary_dilate
CHECK_TYPES = True

# 没检查输出
def binary_extendwhere(IM, MAP, NSTEPS, MASK=-3, RK=1):
    global CHECK_TYPES
    STD_MASK = -3
    if NSTEPS == '?' and isinstance(IM, str):
        print(f'extended = binary_extendwhere(IM, MAP, NSTEPS|Inf, <{STD_MASK}|MASK>, <1|RK>)')
        return

    if RK is None or np.isnan(RK):
        RK = 1

    if MASK is None or np.isnan(MASK):
        MASK = STD_MASK

    if CHECK_TYPES:
        if not isinstance(NSTEPS, int) or NSTEPS < 1:
            print('Number of iterations must be a Natural number (normally, >= 1).')
            NSTEPS = 1
            print(f'Continuing with {NSTEPS} iteration(s).')

        if IM.shape != MAP.shape:
            raise ValueError(f'Image and map arrays must have identical dimensions, size, and shape, not {IM.ndim} vs. {MAP.ndim} and [{IM.shape[0]}, {IM.shape[1]}] vs. [{MAP.shape[0]}, {MAP.shape[1]}].')

    mask = rank_mask(MASK, IM.shape)
    extended = MAP & IM
    old_CT = CHECK_TYPES

    try:
        for k in range(1, min(NSTEPS, np.prod(IM.shape)) + 1):
            new_extended = MAP & binary_dilate(extended, mask, RK)
            CHECK_TYPES = False
            if np.all(new_extended == extended):
                break
            extended = new_extended

        CHECK_TYPES = old_CT
    except Exception as e:
        CHECK_TYPES = old_CT
        raise e

    return extended