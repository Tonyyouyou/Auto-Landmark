# daijiao
import numpy as np

# 与matlab输出一致
def bandror(B_EN_SM, RATE=None, WDWSPACE=None, DIFF_TIME=None):
    DFLT_RATE = 16000
    DFLT_WDW = 16

    if RATE is None:
        RATE = DFLT_RATE
    if WDWSPACE is None:
        WDWSPACE = DFLT_WDW
    if DIFF_TIME is None:
        DIFF_TIME = 50 * (DFLT_RATE / RATE) / (DFLT_WDW / WDWSPACE)

    dt = max(1, min(int(np.ceil(B_EN_SM.shape[1] / 2) - 1), int(np.round(DIFF_TIME * (RATE / WDWSPACE) / 1000 / 2))))
    if dt < 1:
        bror = np.full_like(B_EN_SM, np.nan)
        return

    nbands = B_EN_SM.shape[0]
    starts = np.tile(B_EN_SM[:, 0], (dt - 1,1)).T
    ends = np.tile(B_EN_SM[:, -1], (dt - 1,1)).T

    diff1 = (B_EN_SM[:, 2 * dt:] - B_EN_SM[:, :-(2 * dt)]) / (2 * dt)
    diff2_start = (B_EN_SM[:, 1:dt] - starts) / np.tile(np.arange(1, dt), (nbands, 1))
    diff2_start = np.hstack((diff2_start,diff1))

    diff2_end = (ends - B_EN_SM[:, -dt:-1]) / np.tile(np.arange(dt - 1, 0, -1), (nbands, 1))
    diff2_end = np.hstack((diff2_start,diff2_end))

    bror = np.hstack((diff2_end[:,0].reshape(len(diff2_end),1),diff2_end))
    bror = np.hstack((bror,(diff2_end[:,-1].reshape(len(diff2_end),1))))
    return bror
