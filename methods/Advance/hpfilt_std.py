import numpy as np
from desmooth import desmooth  # Import the desmooth function
from lm_hplim_std import lm_hplim_std  # Import the standard high-pass filter cutoff frequency function

def hpfilt_std(SIGNAL, RATE, AGE=''):
    """
    High-pass filter for processing signals using parameters suitable for speech.

    Syntax:
        hpsig = hpfilt_std(SIGNAL, RATE, <AGE>)

    Parameters:
        SIGNAL (numpy.array): Sampled speech signal.
        RATE (float): Sampling rate [Hz].
        <AGE> (str, optional): Age group label ("ADULT" or "CHILD") to select the frequency cutoff value suitable for that age group.
                              Default is an empty string, representing adults.

    Returns:
        hpsig (numpy.array): Signal after high-pass filtering using the frequency cutoff value obtained from the 'lm_hplim_std' function.
                             hpsig has the same dimensions as the input signal SIGNAL.
    """
    if np.isscalar(SIGNAL) and SIGNAL == '?':  # If SIGNAL parameter is the character '?', print information about the function syntax
        print('hpsig = hpfilt_std(SIGNAL, RATE, <AGE>)')
        print('\tbased on:')
        lm_hplim_std('?')  # Call the standard high-pass filter cutoff frequency function to print information about its syntax
        return

    if len(AGE) == 0:
        AGE = ''  # If no age group label is provided, set it to an empty string

    # Process each column separately and remove low-frequency components
    hpsig = desmooth(SIGNAL, -(1 + 2 * round(RATE / lm_hplim_std(AGE))))
    # SIZE(hpsig) = SIZE(SIGNAL)

    return hpsig