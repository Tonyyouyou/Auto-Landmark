#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Daijiao Liu

import numpy as np
from matplotlib import mlab
from minfilt1 import minfilt1
from medfilt1e import medfilt1e
from lm_bands_std import lm_bands_std
from abs2 import abs2
from smoothspecbands import smoothspecbands

# 与matlab输出不太一样
def demedfilt1up(sgm, mfknl):
    sgm_filtered = sgm - medfilt1e(sgm.T, 3*mfknl +1).T

    if sgm.shape[0] > 1:
        sgm_filtered = minfilt1(sgm_filtered, 3)

    sgm_filtered = np.maximum(0, sgm_filtered - 0.1)

    threshold = np.mean(sgm_filtered > 0, axis=0) > 1/10
    sgm_filtered = sgm_filtered * np.tile(threshold, (sgm_filtered.shape[0], 1))

    pad_size = 1 + 2 * round(sgm.shape[0] / 10)

    sgm_filtered = np.vstack((np.flipud(sgm_filtered[2:2 + pad_size, :]), sgm_filtered))

    sgm_filtered = np.minimum(sgm_filtered, medfilt1e(sgm_filtered, pad_size))

    impsg = sgm_filtered[pad_size:, :]

    return impsg


def band_energies(SIGNAL, RATE, AGE='CHILD', NOMED='nomed', FIGNO=None, NOISESPEC=None):
    STD_RATE = 16000
    STD_WSPACE = 16
    STD_SAMPLES = 128
    TOP_FREQ = 8000
    LOWRES_BR_FAC = 3

    if NOISESPEC is None:
        NOISESPEC = 0

    if FIGNO is None:
        FIGNO = 0

    if NOMED is None:
        NOMED = 'nomed'

    if AGE is None:
        AGE = 'CHILD'

    if RATE is None:
        RATE = STD_RATE

    suppress_mdn2d = NOMED.lower() == 'nomed'

    bandlims = lm_bands_std(AGE)
    bandlims[bandlims>TOP_FREQ] = TOP_FREQ
    bandlims = bandlims/TOP_FREQ
    bandlims[bandlims> (RATE/STD_RATE)] = RATE / STD_RATE
    NDXRANGES = bandlims * STD_SAMPLES / 2
    NDXRANGES[:,0] = np.floor(NDXRANGES[:,0]) + 1
    NDXRANGES[:,1] = np.ceil(NDXRANGES[:,1]) + 1

    bandrate = RATE / STD_WSPACE
    brfac = 1
    wdwlen = round(STD_SAMPLES * RATE / STD_RATE)
    nfft = wdwlen
    window = np.hanning(wdwlen)

    try:
        # sg0 值比matlab中的值小很多(原文中没有hanning，但是python中必须有）
        sg0,_,_ = mlab.specgram(SIGNAL, NFFT=nfft, Fs=RATE, window=window, noverlap=wdwlen - STD_WSPACE, mode='complex')
        sg0 = abs2(sg0)
        sg0 = sg0*10**4

    except:
        print('\tband_energies: Attempting 3x processing for {0}-sample signal.'.format(len(SIGNAL)))
        brfac = LOWRES_BR_FAC
        sg0,_,_ = mlab.specgram(SIGNAL, NFFT=nfft, Fs=RATE, window=window, noverlap=wdwlen - brfac * STD_WSPACE,mode='complex')
        sg0 = abs2(sg0)
        sg0 = sg0 * 10 ** 4

    tiny = 1e-300
    if sg0.dtype == np.float32:
        tiny = np.float32(1e-30)

    sg = (10 / np.log(10))*np.log(np.maximum(sg0,np.maximum(tiny,np.max(sg0)/(2**30))))

    mfknl = round(3 / 4 * wdwlen / (STD_WSPACE * brfac))

    sgm = medfilt1e(sg, 3, 0)

    if suppress_mdn2d:
        impsg = np.zeros(sgm.shape)
    elif mfknl > 0:
        if NOMED and not NOMED.lower() == 'med':
            print('Unrecognized NOMED keyword = "{0}". Using "".')
        impsg = demedfilt1up(sgm, mfknl)

    NDXRANGES = np.minimum(NDXRANGES, sg.shape[0])

    bands = smoothspecbands(sg - impsg.T, RATE, NDXRANGES, STD_WSPACE * brfac)
    bandsf = smoothspecbands(sg - impsg.T, RATE, NDXRANGES, STD_WSPACE * brfac, 26)

    # 还有两个状况外的条件，可能影响bands和bandsf取值

    return bands, bandsf, bandrate, wdwlen
