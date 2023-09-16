#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Cihan Xiao (Johns Hopkins University)

import numpy as np


def DTChk_Len1(A, oknans=False):
    if np.size(A) == 0:
        return True
    elif oknans:
        return np.all(np.logical_or(np.isnan(A), np.logical_and(np.imag(A) == 0, np.logical_and(A > 0, A == np.fix(A)))))
    else:
        return np.all(np.logical_and(np.imag(A) == 0, np.logical_and(A > 0, A == np.fix(A))))
