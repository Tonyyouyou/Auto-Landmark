import numpy as np
from lm_bands_std import lm_bands_std
from smooth import smooth

# sgrm 是n*m array， RATE和WDWSPACE都是number，NDXRANGES是array
# 输出是n*m array，因为smooth算法不同，所以结果与matlab稍有不同，但是相差不大。
def smoothspecbands(SGRM, RATE=None, NDXRANGES=None, WDWSPACE=None, SMOOTHING_INTERVAL=50):
    STD_RATE = 16000
    STD_WSP = 8

    if SGRM is None:
        print(f"bands_NxL = smoothspecbands(SGRM_KxL, <{STD_RATE}|RATE>, <NDXRANGES_Nx2>, <{STD_WSP}|WDWSPACE>, <50|SMOOTHING_INTERVAL>)")
        print("based on:")
        # lm_bands_std ?

        return

    if RATE is None:
        RATE = STD_RATE

    if WDWSPACE is None:
        WDWSPACE = STD_WSP

    if NDXRANGES is None:
        TOP_FREQ = 8000
        AGE = 'CHILD'
        STD_SAMPLES = 128

        NDXRANGES = np.amin([RATE / STD_RATE, np.amin([TOP_FREQ, lm_bands_std(AGE)]) / TOP_FREQ]) * STD_SAMPLES / 2
        NDXRANGES[:, 0] = 1 + np.floor(NDXRANGES[:, 0])
        NDXRANGES[:, 1] = 1 + np.ceil(NDXRANGES[:, 1])
        NDXRANGES = np.minimum(NDXRANGES, np.array(SGRM.shape[0], SGRM.shape[0]))

    bands = np.zeros((NDXRANGES.shape[0], SGRM.shape[1]))

    for bandno in range(NDXRANGES.shape[0]):
        start_idx = int(NDXRANGES[bandno, 0])
        end_idx = int(NDXRANGES[bandno, 1])
        bands[bandno, :] = np.mean(SGRM[start_idx:end_idx, :], axis=0)

    len_bands = bands.shape[1]
    b0 = np.mean(bands[:, :2], axis=1)
    bend = np.mean(bands[:, -2:], axis=1)

    c = (b0 + bend)[:, np.newaxis] * np.ones((1, len_bands)) / 2 + (b0 - bend)[:, np.newaxis] * np.cos(np.pi * np.arange(1, len_bands + 1) / (len_bands + 1)) / 2

    smoothed_bands = smooth(bands-c, -round((RATE / WDWSPACE) * SMOOTHING_INTERVAL / 1000))+ c

    return smoothed_bands


