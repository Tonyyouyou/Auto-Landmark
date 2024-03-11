import numpy as np
from scipy.signal import convolve2d
from scipy.signal import convolve
from scipy.ndimage import maximum_filter1d, minimum_filter1d
from maxfilt1 import maxfilt1
from minfilt1 import minfilt1
from scipy.signal.windows import hann
from smooth import smooth
from binary_dilate import binary_dilate


# 输出peaks 与matlab顺位早1位，因为Python从0开始。 输出pvts与matlab不一样。
def band_pivots(B_EN_ROR_C, B_EN_ROR_F, THRESH=None, MAX_WDW=20):
    STD_MAX_WDW = 20
    LF_EARLY = 10
    LF_LATE = 8

    if THRESH is None:
        THRESH = np.max(np.abs(B_EN_ROR_C), axis=1) / 5
    if MAX_WDW is None:
        MAX_WDW = STD_MAX_WDW

    if np.any(np.isinf(THRESH)) or np.any(np.isnan(THRESH)):
        print("Inf or NaN threshold impossible to use.")
        print("Continuing by ignoring band(s) with unusable thresholds.")

    min_bands = 1 + np.floor(B_EN_ROR_C.shape[0] / 2)

    pksc, peaksc = find_pks(B_EN_ROR_C, THRESH)
    pksf, peaksf = find_pks(B_EN_ROR_F, THRESH)
    peaks = peaksf

    pkscp = np.where(pksc > 0, 1, 0)
    pkscn = np.where(pksc < 0, 1, 0)
    # binary_dilate有问题
    pkscpd = binary_dilate(pkscp, np.ones(2 * MAX_WDW).reshape(1, -1))
    pkscnd = binary_dilate(pkscn, np.ones(2 * MAX_WDW).reshape(1, -1))
    # pkscpd = pkscp
    # pkscnd = pkscn

    pf = np.zeros(pksf.shape)
    for b in range(B_EN_ROR_C.shape[0]):
        pf[b, :] = match_peaks(pkscpd[b, :], pksf[b, :])
        pf[b, :] = pf[b, :] - match_peaks(pkscnd[b, :], -pksf[b, :])

    smpksf = smooth(pksf, -1 - 2 * MAX_WDW) * MAX_WDW
    np.set_printoptions(threshold=np.inf)
    pvts = pf_to_pvts(pf, MAX_WDW, min_bands, smpksf)

    if B_EN_ROR_C.shape[0] < 4:
        return

    lowbands = np.arange(1, 1 + np.floor(B_EN_ROR_C.shape[0] / 5))
    highbands = np.arange(-1, 0) + pf.shape[0]
    fbands = np.array([lowbands, highbands]).astype(int).flatten()
    fpf = pf[fbands, :]

    fpf[np.array(lowbands).astype(int), :] = -fpf[np.array(lowbands).astype(int), :]
    fpf = atten_near_pvts(fpf, pvts, MAX_WDW, THRESH[fbands])

    fpvts = fpf_to_fpvts(fpf, MAX_WDW, min_bands, LF_EARLY, LF_LATE, lowbands, smpksf[fbands, :])

    if fpvts:
        smpksf = smpksf[fbands, :]
        fpvts[2, :] = smpksf[abs(fpvts[0, :].astype(int)), fpvts[1, :].astype(int)]
        fpvts[0, :] = 1j * np.sign(fpvts[0, :]) * fbands[abs(fpvts[0, :]).astype(int)]
        pvts = merge_pvts(pvts, fpvts)

    return pvts, peaks


# pf_to_pvts 输出与matlab完全一致
# 这里输入：PF是5*n array，MAX_WDW和MIN_BANDS是number，SMPKSF是5*n array
def pf_to_pvts(PF, MAX_WDW, MIN_BANDS, SMPKSF):
    pkt = np.sum(np.sign(PF), axis=0)
    # 这里pkt和np.hanning都是一维数组
    pkwdwp = 2 * convolve(np.maximum(0, pkt), np.hanning(2 * MAX_WDW + 1)[1:-1], mode='same')
    pkwdwn = 2 * convolve(np.maximum(0, -pkt), np.hanning(2 * MAX_WDW + 1)[1:-1], mode='same')

    pktxn = maxfilt1(pkt, MAX_WDW) + minimum_filter1d(pkt, MAX_WDW)

    condition1 = (pktxn > 0) & (pkwdwp >= 2 * MIN_BANDS - 1) & (np.diff(np.hstack([0, pkwdwp])) >= 0) & (
            np.diff(np.hstack([pkwdwp, 0])) <= 0)
    condition2 = (pktxn < 0) & (pkwdwn >= 2 * MIN_BANDS - 1) & (np.diff(np.hstack([0, pkwdwn])) >= 0) & (
            np.diff(np.hstack([pkwdwn, 0])) <= 0)
    ndxpvts = np.where(condition1 | condition2)[0]

    if ndxpvts is None:
        return

    # Convert indices to 1-based indexing (as in MATLAB)
    ndxpvts = ndxpvts + 1

    if len(ndxpvts) == 0:
        pvts = np.array([])
        return pvts

    # Calculate the first row of pvts
    sorted_indices = np.argsort(-np.abs(SMPKSF[:, ndxpvts]), axis=0)[int(MIN_BANDS):]
    pvts_row1 = np.sign(pktxn[ndxpvts - 1]) * sorted_indices[0]

    # Set the second row of pvts as ndxpvts
    pvts_row2 = ndxpvts

    # Combine the rows to form pvts
    pvts = np.vstack((pvts_row1, pvts_row2))

    # Add SMPKSF values based on pvts
    indices = (np.abs(pvts[0]).astype(int) - 1, pvts[1].astype(int) - 1)  # Convert to 0-based indexing
    pvts_row3 = SMPKSF[indices]
    pvts = np.vstack((pvts, pvts_row3))

    return pvts


# Provided conv2 function （部分matlab代码用得到这个，如果是1维数组的conv2， 直接用modified convolve函数）
def conv2(x, y, mode='same'):
    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)


# atten_near_pvts输出与matlab基本完全一致
# 这里的输入：pf是4*n 数组，pvts 是3*n 数组，MAX_WDW是number， THRESH是一维数组
def atten_near_pvts(PF, PVTS, MAX_WDW, THRESH):
    if PVTS.size != 0:
        ans = np.zeros((PF.shape[1],), dtype=int)
        ans[PVTS[1, :].astype(int) - 1] = 1
        ans = ans.reshape(1, -1)

        apf = PF * (1 - conv2(ans, hann(2 * MAX_WDW - 1).reshape(1, -1), mode='same'))
        THRESH_broadcasted = np.tile(THRESH, (PF.shape[1], 1)).T
        apf = apf * (np.abs(apf) > THRESH_broadcasted)
    else:
        apf = PF

    return apf


# fpf_to_fpvts 输出与matlab完全一致
# fpf 是4*n矩阵，中间四位是数字，loabands 是一维矩阵，smpksf是4*n 矩阵
def fpf_to_fpvts(FPF, MAX_WDW, MIN_BANDS, LF_EARLY, LF_LATE, LOWBANDS, SMPKSF):
    fpfp = np.maximum(0, FPF)

    fpfp[LOWBANDS.astype(int) - 1, :] = np.hstack(
        (np.zeros((len(LOWBANDS), LF_EARLY)), fpfp[LOWBANDS.astype(int) - 1, :-LF_EARLY]))
    fpfn = np.minimum(0, FPF)
    fpfn[LOWBANDS.astype(int) - 1, :] = np.hstack(
        (fpfn[LOWBANDS.astype(int) - 1, LF_LATE:], np.zeros((len(LOWBANDS), LF_LATE))))

    fpvts = pf_to_pvts(fpfp + fpfn, MAX_WDW, MIN_BANDS, SMPKSF)

    np.set_printoptions(threshold=np.inf)

    return fpvts


# 输出的pks与matlab完全一致， peaks序列号比matlab少1，因为python是从0开始，matlab是从1开始。
# B_EN_ROR是5*n矩阵，THRESH是一维矩阵
def find_pks(B_EN_ROR, THRESH):
    pks = np.zeros_like(B_EN_ROR)
    peaks = [None] * B_EN_ROR.shape[0]

    for bandno in range(B_EN_ROR.shape[0]):
        # Create masks for peak detection
        mask1 = B_EN_ROR[bandno, 1:-1] - B_EN_ROR[bandno, :-2]
        mask2 = B_EN_ROR[bandno, 1:-1] - B_EN_ROR[bandno, 2:]
        merge = ((mask1 * mask2) > 0)
        mask3 = (np.abs(B_EN_ROR[bandno, 1:-1]) > THRESH[bandno])
        # Combine masks using logical AND
        p = 1 + np.where(merge & mask3)[0]
        peaks[bandno] = p
        pks[bandno, p] = B_EN_ROR[bandno, p]

    return pks, peaks


# 与matlab输出完全一致
def match_peaks(PKSCD, PKSF):
    # Calculate the differences between adjacent elements in PKSCD
    diff_array = np.diff(np.concatenate(([0], PKSCD, [0])))

    # Find indices of rising edges (start of peaks)
    ndxon = np.where(diff_array > 0)[0]

    # Find indices of falling edges (end of peaks)
    ndxoff = np.where(diff_array < 0)[0]

    mpks = np.zeros(len(PKSF))
    for cpk_start, cpk_end in zip(ndxon, ndxoff):
        # Determine the range for searching the maximum in PKSF
        search_range = slice(max(0, cpk_start), min(len(PKSF), cpk_end))

        # Find the maximum value and its index within the specified range
        max_value = np.max(PKSF[search_range])
        max_index = np.argmax(PKSF[search_range])

        # Update mpks with the maximum value at the corresponding position
        mpks[search_range][max_index] = max_value

    return mpks


# 与matlab输出完全一致
def merge_pvts(PVTS1, PVTS2):
    combined_pvts = np.hstack((PVTS1, PVTS2))
    mpvts = combined_pvts[:, combined_pvts[1, :].argsort()]
    return mpvts
