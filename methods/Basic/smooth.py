#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Cihan Xiao (Johns Hopkins University)

import numpy as np
from scipy import signal
import argparse


def _hanning(n):
    """
    Helper function to mimic MATLAB's hanning function.
    """
    m = n/2
    w = .5 * (1 - np.cos(2 * np.pi * (np.arange(m) + 1) / (n + 1)))
    w = np.concatenate((w, w[::-1]))
    return w


def isscalar(x):
    """
    Helper function to mimic MATLAB's isscalar function.
    """
    if (np.isscalar(x) or np.isinf(x)).any():
        return True
    return False


def is_vec(x):
    """
    Helper function to mimic MATLAB's is_vec function.
    """
    if (np.isinf or np.sum(np.array(x.shape) == 1) >= x.ndim - 1):
        return True
    return False


def DTChk_Len1(A, oknans=False):
    if np.size(A) == 0:
        return True
    elif oknans:
        return np.all(np.logical_or(np.isnan(A), np.logical_and(np.imag(A) == 0, np.logical_and(A > 0, A == np.fix(A)))))
    else:
        return np.all(np.logical_and(np.imag(A) == 0, np.logical_and(A > 0, A == np.fix(A))))


def smooth2_noscalar(im, knl):
    if np.all(np.abs(knl) == 1) and len(knl) <= len(im.shape):
        return im
    elif np.array_equal(np.shape(knl), [1, 2]) and DTChk_Len1(np.abs(knl)):
        if np.all(knl != 1):
            out = smooth2_noscalar(smooth2_noscalar(
                im, [knl[0, 0], 1]), [1, knl[0, 1]])
            return out

        klen = knl[knl != 1]
        if klen > 0:
            kernel = np.ones((klen, 1))
        elif klen < 0:
            kernel = _hanning(-klen)
        else:
            raise ValueError(
                'Specifying kernel by 2 integers requires that both be non-zero.')

        kernel = kernel / np.sum(kernel)

        if knl[0] == 1:
            kernel = kernel.T
    else:
        kernel = (knl / np.sum(knl)).astype(im.dtype)

    if len(im.shape) == 1:
        return signal.convolve(im.reshape((im.shape[0], 1)), kernel, mode='same')
    return signal.convolve(im, kernel, mode='same')


def smooth2(im, knl, check_type=False):
    if check_type:
        if np.ndim(knl) != 2:
            raise ValueError(
                f'Smoothing kernel must have two dimensions, at most. Instead got {np.ndim(knl)}:\n{knl}')
        elif np.isscalar(knl) and not isinstance(knl, int):
            raise ValueError(
                'Kernel must be an Integer (or Inf), or a row-pair of same, or a non-scalar.')
        elif (not np.isscalar(knl) and knl.shape != (1, 2)) and (np.any(np.isnan(knl)) or np.any(np.isinf(knl)) or np.sum(knl) == 0):
            raise ValueError(
                'Non-scalar kernels must have non-zero sum, with no Inf or NaN elements.')

    if np.isscalar(knl):
        if knl == 0:
            out = im
        elif np.isinf(knl):
            out = np.full_like(im, np.mean(im))
        else:
            out = smooth2_noscalar(im, (knl, knl))
    else:
        out = smooth2_noscalar(im, knl)

    return out


def smooth(sig, knl, check_type=False):
    """Smooth a signal with a kernel.

    Parameters
    ----------
    sig : array_like
        Signal to be smoothed.
    knl : array_like
        Kernel to smooth the signal with.

    Returns
    -------
    array_like
        Smoothed signal.

    """
    if len(knl) == 1 and not np.isinf(knl[0]):
        knl = int(knl[0])
    elif np.isinf(knl[0]):
        knl = np.inf
    else:
        knl = np.asarray(knl)

    sig = np.asarray(sig)

    if check_type:
        # Type check, the kernel must be an integer or a vector of length > 1.
        if not is_vec(knl):
            raise ValueError(
                f'Kernel must be Integer or vector (length > 1). Instead got {knl}.')
        # Sanity check, a ector kernel must have non-zero sum, with no Inf or NaN elements
        if not isscalar(knl) and (np.isnan(knl).any() or np.isinf(knl).any() or sum(knl) == 0):
            raise ValueError(
                'Vector kernel must have non-zero sum, with no Inf or NaN elements.')

    if np.isscalar(knl):
        if not knl or abs(knl) == 1:
            return sig
        elif np.isinf(knl):
            out = np.ones(sig.shape) * np.mean(sig)
            if sig.shape[0] == 1:
                sig = np.repeat(sig, sig.shape[1], axis=0)
            return out
        elif knl > 0:
            knl = np.pi * np.ones(knl)
            knl = knl.reshape((knl.shape[0], 1))
        else:
            knl = _hanning(-knl)
    elif len(knl.shape) == 1:
        # This does not do anything in python, but kept to align with the MATLAB code
        knl = knl.reshape((knl.shape[0], 1))

    if is_vec(sig):
        out = smooth2(im=sig, knl=knl, check_type=check_type)
    else:
        try:
            out = smooth2(im=sig, knl=knl, check_type=check_type)
        except ValueError:
            out = np.full_like(sig, np.nan)

            ncols = sig.shape[1]
            try:
                for colno in range(0, ncols, round(np.sqrt(ncols))):
                    col_range = slice(colno, min(
                        ncols, colno + round(np.sqrt(ncols) - 1)))
                    out[:, col_range] = smooth2(sig[:, col_range], knl)
            except:
                for colno in range(ncols):
                    out[:, colno] = smooth2(sig[:, colno], knl)

    # Known Issue (Cihan): The convolution result for padded signal (and kernel) is not the same as MATLAB's
    # e.g. If the fully convolved signal is [0.5, 1.5, 5.5, 4.5], the MATLAB implementation returns [1.5, 5.5, 4.5],
    # whereas the python implementation returns [0.5, 1.5, 5.5].
    return out


def main():
    parser = argparse.ArgumentParser(
        description='Smooth a signal with a kernel.')
    parser.add_argument('--sig', '-s', type=float, nargs='+',
                        help='signal to be smoothed')
    parser.add_argument('--knl', '-k', type=float, nargs='+',
                        help='kernel to smooth the signal with')
    parser.add_argument('--check_type', '-c', action='store_true',
                        help='check if kernel is an integer or a vector of length > 1')
    args = parser.parse_args()

    print(smooth(sig=args.sig, knl=args.knl, check_type=args.check_type))


if __name__ == '__main__':
    main()
