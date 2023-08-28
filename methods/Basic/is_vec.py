#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Cihan Xiao (Johns Hopkins University)

import numpy as np


def is_vec(x):
    """
    Helper function to mimic MATLAB's is_vec function.
    """
    # Return True if x is a scalar and it is Inf, or if x is an array with only one elemnt Inf, or if x is a vector (not a matrix)
    if np.isscalar(x) and np.isinf(x):
        return True
    if not np.isscalar(x) and np.sum(np.shape) == 1 and np.isinf(x[0]):
        return True
    if np.sum(np.array(x.shape) == 1) >= x.ndim - 1:
        return True
    return False
