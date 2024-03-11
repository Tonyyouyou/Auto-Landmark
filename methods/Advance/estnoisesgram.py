#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Daijiao Liu

import numpy as np
import warnings
import scipy.special
from scipy.signal import medfilt
from abs2 import abs2

def estnoisesgram(sgm, nbdlen):
    STD_NBDLEN = 9

    if nbdlen is None:
        nbdlen = STD_NBDLEN

    # Check types and dimensions
    if type(nbdlen) != int and type(nbdlen) != float and type(nbdlen) != complex:
        raise ValueError('Neighborhood length (time slices) must be an integer greater than 0.')
    else:
        if nbdlen <= 0:
            raise ValueError('Neighborhood length (time slices) must be an integer greater than 0.')

    if sgm.shape[1] < 2 * nbdlen:
        raise ValueError(
            'Spectrogram width (time slices: {}) must be >= 2x neighborhood length (2*{}).'.format(sgm.shape[1],
                                                                                                   nbdlen))
    dfheur = np.sqrt(2)*nbdlen
    dfh_ends = dfheur / 2

    acfwid = (nbdlen*1.25 + 0.3) / 2
    acfw_ends = 1

    chisq2Scale = 0.7336

    if not np.isreal(sgm).all():
        warnings.warn('Non-Real spectrogram; proceeding with SQRT of estimate from absolute-square.')
        nspec = np.sqrt(estnoisesgram(abs2(sgm),nbdlen))
    elif np.any(sgm) < 0:
        raise ValueError(' Some array elements < 0; must be >= 0 for linear power (or else non-Real).')
    else:
        minMeanScale = dfheur / (2 * gammaincinv(dfheur/2, 1/(2**(acfwid/sgm.shape[1]-2*nbdlen))))
        minmnsc_ends = dfh_ends / (2 * gammaincinv(dfh_ends / 2, 1 / (2 ** (acfw_ends / (sgm.shape[1] - 2 * nbdlen)))))

        # medfilt1e()

        pass


def gammaincinv(a, p):
    # default use inverse of the regularized upper incomplete gamma function
    return scipy.special.gammaincinv(a, p)
