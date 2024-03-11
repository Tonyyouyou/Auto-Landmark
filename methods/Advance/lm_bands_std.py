#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Daijiao Liu

import numpy as np


def lm_bands_std(age):
    TOP_FREQ = 8000
    ALT_CHILD_LTRS = 'ic'
    ALT_ADULT_LTRS = 'mfne'

    if age.upper() == 'ADULT' or age.lower() in ALT_ADULT_LTRS:
        bandlims = np.array([[0, 400], [800, 1500], [1200, 2000], [2000, 3500], [3500, 5000], [5000, TOP_FREQ]],
                            np.float32)
    elif age.upper() == 'CHILD' or age.lower() in ALT_CHILD_LTRS:
        bandlims = np.array([[150, 600], [1200, 2500], [1800, 3000], [3000, 4000], [4000, 6000], [6000, TOP_FREQ]],
                            np.float32)
    else:
        raise TypeError(f'Unrecognized band-limits AGE = "{age}". Using default value.')

    return bandlims
