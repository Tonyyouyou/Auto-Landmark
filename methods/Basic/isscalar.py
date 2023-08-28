#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Cihan Xiao (Johns Hopkins University)

import numpy as np


def isscalar(x):
    """
    Helper function to mimic MATLAB's isscalar function.
    """
    if (np.isscalar(x) or np.isinf(x).all()):
        return True
    return False
