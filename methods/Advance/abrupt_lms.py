#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Daijiao Liu

import numpy as np
from abrupt_events import abrupt_events
from maxfilt1 import maxfilt1
from minfilt1 import minfilt1
from lm_codes import lm_codes
from sortperm import sortperm
from binary_extendwhere import binary_extendwhere
#from binary_extendwhere import binary_extendwhere


def abrupt_lms(SIGNAL, RATE, AGE, PREV_RORTHR, NOMED, VOICING, VRATE):
    """
    Abrupt Consonantal landmarks of an acoustic speech-signal segment.

    Syntax:

    Parameters:
    :param signal: acoustic speech signal to be processed; (numpy array)
    :param rate: sampling rate of SIGNAL; (number)
    :param age:  "adult" [dflt.] or "child" (case-insens.) to cause the frequency-band
                  parameters to be set accordingly;  (string)
    :param prev_rorthr: threshhold, if any, [dB/msec; dflt. = NaN] used for six spectral bands' Rates Of Rise,
                        to be blended (by 'blend_rorthresh') with limits from current SIGNAL;
    :param nomed: "nomed" [case-insensitive] to suppress s/gram median filtering across times, or "" [the dflt.]
                   to perform this; the filtering is slow but suppresses many brief,
                   non-speech noises that can introduce (e.g.) spurious landmarks;
    :param voicing: (fuzzy) degree of voicing, as from 'deg_voiced'; exactly 0 where the source has been determined
                    to be unvoiced; if not provided, some pairs of types will be merged (e.g., "burst" & "syl.");
                    VOICING should be long enough that voicing can be determined for every detected landmark;
    :param vrate:  samping rate [Hz] of VOICING (often 125); note that VRATE must be supplied if VOICING is supplied,
                   and vice versa;

    Returns:
        lms  = 3xN array of landmarks (+g/-g, +b/-b, etc), as from 'lm_codes', for some N >= 0;
               each landmark is denoted by a time, type (code as returned by 'lm_codes'),
               and "degree"/"strength" (see Notes);
        bandrate = sampling rate of spectrograms exhibiting the landmarks
                   (generally a sub-multiple of RATE, often 1 kHz);
        wdwlen  = length of spectrogram window used to estimate energy; the first values reflect energy centered
                  in a window starting at the first sample, and of 'wdwlen' samples;
         rorthr6    = vector of min. thresholds [dB/msec] for "sufficiently" high rate of rise (or fall)
                      for each of 6 coarse-pass bands separately; coarse and fine ror's are scaled for
                      common thresholds; this is a combination (as from 'blend_rorthresh') of SIGNAL information
                      and PREV_RORTHR6.
    """

    if AGE is None:
        AGE = 'adult'
    if PREV_RORTHR is None:
        PREV_RORTHR = np.nan

    plms, bandrate, wdwlen, rorthr, _, _ = abrupt_events(SIGNAL=SIGNAL, RATE=RATE, AGE=AGE,
                                                                 PREV_RORTHR=PREV_RORTHR, NOMED=NOMED)

    # plms = np.loadtxt('plms.txt')
    plms[2, :] = np.sqrt(np.clip(np.abs(plms[2, :]), 0, 4)) * np.sign(plms[2, :])
    # plms = np.loadtxt('plms.txt')
    lmArray = None
    if VOICING is not None:
        if isinstance(VOICING, (list, np.ndarray)) and len(VOICING[0]) == 2:
            print('warning')
            VOICING = np.transpose(VOICING)
        vcg = VOICING[0, :]
        per = np.all(VOICING, axis=0)
        per = per.astype(int)
        vcg_extended = np.concatenate(
            (vcg, np.repeat(vcg[-1], np.ceil(max(plms[0, :]) * VRATE).astype(int) - len(vcg))))
        # if not np.all(per):
        #     binary_extendwhere(per, vcg_extended, np.inf)
        per = per.astype(float)
        lmArray = merged_plms_voicing(plms, vcg, per, VRATE)
    else:
        lmArray = plms

    return lmArray


# 测试了短的语音片段，输出一致（有些if语句没有被触发，不确定在长语音片段对不对）
def merged_plms_voicing(CLMS, VOICING, PER, VRATE):

    MINUS_G, PLUS_G, MINUS_B, PLUS_B, MINUS_S, PLUS_S, PLUS_F, MINUS_F, PLUS_P, MINUS_P = \
            lm_codes('MINUS_G', 'PLUS_G', 'MINUS_B', 'PLUS_B', 'MINUS_S', 'PLUS_S', 'PLUS_F', 'MINUS_F', 'PLUS_P', 'MINUS_P')

    lms = CLMS.copy()
    vlms, gmarks = contour_to_laryng(VOICING, PLUS_G, MINUS_G, VRATE)
    if PER is None or len(PER) == 0:
        plms = []
    else:
        # You need to define this function
        plms, pmarks = contour_to_laryng(PER, PLUS_P, MINUS_P, VRATE)


    lms, ndxspg1, ndxsmg1 = move_after_gs(lms, [PLUS_B, PLUS_F], gmarks, VRATE, MINUS_G, PLUS_G)
    lms[1, ndxspg1] = sv_from_bf_lms(lms[1, ndxspg1])

    if ndxspg1.size > 0:
        lms[2, ndxspg1] = np.sign(lms[2, ndxspg1]) * np.minimum(np.abs(lms[2, ndxspg1]), np.maximum(0.5, VOICING[ndxspg1])) * (lms[1, ndxspg1] != 0)

    lms, ndxspg2, ndxsmg2 = move_before_gs(lms, [MINUS_B, MINUS_F], gmarks, VRATE, MINUS_G, PLUS_G)
    lms[1, ndxsmg2] = sv_from_bf_lms(lms[1, ndxsmg2])

    if ndxsmg2.size > 0:
        lms[2, ndxsmg2] = np.sign(lms[2, ndxsmg2]) * np.minimum(np.abs(lms[2, ndxsmg2]), np.maximum(0.5, VOICING[ndxsmg2])) * (lms[1, ndxsmg2] != 0)

    unmoved = np.setdiff1d(np.arange(lms.shape[1]), np.concatenate((ndxsmg1, ndxspg2)))
    vndx0 = np.maximum(1, np.ceil(lms[0, unmoved] * VRATE).astype(int))

    vstr = VOICING[np.minimum(vndx0, len(VOICING) - 1)]

    unmoved = unmoved[vstr != 0]
    lms[1, unmoved] = sv_from_bf_lms(lms[1, unmoved])
    if unmoved.any():
        lms[2, unmoved] = (lms[1, unmoved] != 0) * np.sign(lms[2, unmoved]) * np.minimum(np.abs(lms[2, unmoved]),
                                                                                         np.maximum(0.5,
                                                                                                    vstr[vstr != 0]))

    sndxs = np.where(lms[1, :] == PLUS_S)[0]
    if sndxs.size > 0 and sndxs[-1] == lms.shape[1] - 1:
        sndxs = sndxs[:-1]
    if sndxs.size > 0:
        mask = (lms[1, sndxs + 1] == MINUS_S) & (lms[0, sndxs + 1] - lms[0, sndxs] < 0.032)
        lms = np.delete(lms, np.where(mask)[0], axis=1)

    mlms = np.hstack((lms, vlms, plms))
    mlms = mlms[:, np.argsort(mlms[0, :])]

    return mlms


# 输出与matlab一致
def contour_to_laryng(CONTOUR, ONCODE, OFFCODE, VRATE):
    KWID = 2
    marks = np.zeros(CONTOUR.shape)

    dc = np.diff(np.concatenate(([0], CONTOUR)))
    marks[dc == 1] = ONCODE
    marks[dc == -1] = OFFCODE

    if CONTOUR[0]:
        marks[0] = ONCODE

    lclmax = maxfilt1(CONTOUR, KWID + 1, 0)  # Assuming you have defined maxfilt1
    lclmin = minfilt1(CONTOUR, KWID + 1, 0)  # Assuming you have defined minfilt1

    stron = np.concatenate((np.zeros(KWID // 2), lclmax[:-KWID // 2])) - np.concatenate((lclmin[KWID // 2:], np.zeros(KWID // 2)))
    stroff = np.concatenate((lclmax[KWID // 2:], np.zeros(KWID // 2))) - np.concatenate((np.zeros(KWID // 2), lclmin[:-KWID // 2]))
    marked_indices = np.where(marks)

    lm = np.vstack([
        (marked_indices[0] + 1 - 0.5) / VRATE,
        marks[marked_indices],
        (marks[marked_indices] == ONCODE).astype(int) - (marks[marked_indices] == OFFCODE).astype(int)
    ])

    return lm,marks


# 函数输出都与matlab完全一致 （MINUS_G和PLUS_G在matlab中是全局变量，这里我直接参数输入了）
def move_after_gs(LMS, CLMTYPES, GMARKS, VRATE, MINUS_G, PLUS_G):
    """
    Shifts LMS values to after the given markers.
    """
    return move_around_gs(LMS, CLMTYPES, GMARKS, VRATE, +0.5 / VRATE, MINUS_G, PLUS_G)

# 函数输出都与matlab完全一致 （MINUS_G和PLUS_G在matlab中是全局变量，这里我直接参数输入了）
def move_before_gs(LMS, CLMTYPES, GMARKS, VRATE, MINUS_G, PLUS_G):
    """
    Shifts LMS values to before the given markers.
    """
    return move_around_gs(LMS, CLMTYPES, GMARKS, VRATE, -0.5 / VRATE, MINUS_G, PLUS_G)

# 函数输出都与matlab完全一致 （MINUS_G和PLUS_G在matlab中是全局变量，这里我直接参数输入了）
def move_around_gs(LMS, CLMTYPES, GMARKS, VRATE, TSHIFT, MINUS_G, PLUS_G):
    """
    General function to shift LMS values around given markers.
    """
    ndxspg, ndxsmg = find_near_gs(LMS, CLMTYPES, GMARKS, VRATE, MINUS_G, PLUS_G)
    shiftedlms = LMS.copy()
    # Assuming 'ans' is an index or indices obtained from somewhere; needs clarification
    ans = [ndxspg, ndxsmg]  # This needs to be defined based on the context of your MATLAB code
    shiftedlms[0, ans] = (LMS[0, ans] + TSHIFT) * (LMS[1, ans] != 0)
    return shiftedlms, ndxspg, ndxsmg

# 函数输出都与matlab完全一致 （MINUS_G和PLUS_G在matlab中是全局变量，这里我直接参数输入了）
def find_near_gs(LMS, CLMTYPES, GMARKS, VRATE, MINUS_G, PLUS_G):
    """
    Finds indices near certain markers.
    """

    if MINUS_G is None or PLUS_G is None:
        MINUS_G, PLUS_G = lm_codes('MINUS_G', 'PLUS_G')

    cndxs = np.where(np.isin(LMS[1, :], CLMTYPES))[0]
    gndxs = np.clip(np.ceil(LMS[0, cndxs] * VRATE).astype(int), 1, len(GMARKS))
    ndxspg = cndxs[GMARKS[np.minimum(gndxs, len(GMARKS) - 1)] == PLUS_G]
    ndxsmg = cndxs[GMARKS[np.minimum(gndxs, len(GMARKS) - 1)] == MINUS_G]
    return ndxspg, ndxsmg


def sv_from_bf_lms(BFLMS):
    """
    Converts BFLMS codes to corresponding SVLMS codes.
    """

    MINUS_B, PLUS_B, MINUS_S, PLUS_S, MINUS_F, PLUS_F, MINUS_V, PLUS_V = lm_codes(
            'MINUS_B', 'PLUS_B', 'MINUS_S', 'PLUS_S', 'MINUS_F', 'PLUS_F', 'MINUS_V', 'PLUS_V')

    # Creating a copy of the input array to modify
    svlms = BFLMS.copy()

    # Replacing the values according to the predefined codes
    svlms[BFLMS == MINUS_B] = MINUS_S
    svlms[BFLMS == MINUS_F] = MINUS_V
    svlms[BFLMS == PLUS_B] = PLUS_S
    svlms[BFLMS == PLUS_F] = PLUS_V

    return svlms