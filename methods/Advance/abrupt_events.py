#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Daijiao Liu

import numpy as np
from band_pivots import band_pivots
from band_energies import band_energies
from bandror import bandror
from justmore import justmore
from blend_rorthresh import blend_rorthresh
from lm_codes import lm_codes

def abrupt_events(SIGNAL, RATE, AGE="adult", PREV_RORTHR=np.nan, NOMED="med", FIGNO=None):
    """

    :param signal:
    :param rate:
    :param age:
    :param prev_rorthr:
    :param nomed:
    :param figno:

    :return:
    lms
    bandrate
    wdwlen
    rorthr
    bands
    bandsf
    """

    pvts, bandrate, wdwlen, rorthr, peaks, bands, bandsf = cons_pivots(SIGNAL, RATE, AGE, PREV_RORTHR, NOMED)

    if pvts.size > 0:
        valid_indices = np.logical_and(np.diff(np.sign(pvts[0, :])) == 0, np.diff(pvts[1, :]) == 1)
        if valid_indices.any() != False:
            pvts = pvts[:, valid_indices]

    # # rorthr有问题， lm_ndxs函数需要写。
    if pvts.size == 0:
        lms = np.zeros((3, 0))
    else:
        lms = np.vstack([
            pvts[1, :] / bandrate + 0.5 * wdwlen / RATE,
            (np.sign(pvts[0, :]) + 3) / 2,
            np.sign(pvts[2, :]) * (np.abs(pvts[2, :]) / rorthr[np.int32(np.abs(pvts[0, :]))])
        ])
        nonzero_indices = lms[1, :] != 0
        lms[1, nonzero_indices] = lm_ndxs(lms[1, nonzero_indices])
    return lms, bandrate, wdwlen, rorthr, bands, bandsf


# 少了三个pvts值，我觉得可能问题在medium滤波上面（印象中就smooth和medium不同）
def cons_pivots(SIGNAL, RATE, AGE, PREV_RORTHR, NOMED):
    FINE_TIMESCALE = 26

    bands, bandsf, bandrate, wdwlen = band_energies(SIGNAL, RATE, AGE, NOMED)

    bror = bandror(bands, bandrate, 1)
    brorf = bandror(bandsf, bandrate, 1, FINE_TIMESCALE)

    scalefac = (np.max(np.abs(bror), axis=1) / justmore(np.max(np.abs(brorf), axis=1)))[:, np.newaxis]
    scalefac = np.tile(scalefac,(1,len(brorf[0])))
    rorthr = blend_rorthresh(PREV_RORTHR, np.max(np.abs(bror), axis=1) / 5, len(SIGNAL) / RATE * 1000)
    pvts, peaks = band_pivots(bror[1:], brorf[1:] * scalefac[1:], rorthr[1:])

    if len(pvts) > 0:
        pvts[0, :] = np.sign(pvts[0, :]) * (1 + np.abs(pvts[0, :]))

    return pvts, bandrate, wdwlen, rorthr, peaks, bands, bandsf

# 与matlab一致
def lm_ndxs(LMCODES):
    # Define the LM codes for different cases
    MINUS_B = 1
    PLUS_B = 2
    PLUS_F = 1.5 + 0.5j
    MINUS_F = 1.5 - 0.5j

    # Create an array to store the LM codes for the given LMCODES
    lmndxs = np.full(LMCODES.shape, np.nan)

    # Define the LM codes for different cases
    LM_PB = lm_codes('PLUS_B')
    LM_MB = lm_codes('MINUS_B')
    LM_PF = lm_codes('PLUS_F')
    LM_MF = lm_codes('MINUS_F')

    # Map LMCODES to LM codes
    lmndxs[LMCODES == MINUS_B] = LM_MB
    lmndxs[LMCODES == PLUS_B] = LM_PB
    lmndxs[LMCODES == MINUS_F] = LM_MF
    lmndxs[LMCODES == PLUS_F] = LM_PF

    return lmndxs