import numpy as np
from smooth import smooth  # Import the custom smooth function

def desmooth(SIG, KNL):
    if KNL == '?':  # Check if the value of KNL is '?'
        print("out = desmooth(SIGCOLS, 0|KNL|Inf|KNLLEN|-KNLLEN)")  # Print a help message
        return

    if np.isinf(KNL):  # Check if KNL is infinity
        KNL = 0  # If KNL is infinity, set it to 0

    if KNL == 0 and isinstance(SIG, np.ndarray) and len(SIG.shape) == 1:
        # If KNL is 0 and SIG is a 1D NumPy array
        # Compute the output by subtracting the mean from each element of SIG
        out = SIG - np.nanmean(SIG)
    elif KNL == 0:
        # If KNL is 0 and SIG is not 1D
        # Compute the output by subtracting the column-wise mean from SIG
        out = SIG - np.tile(np.nanmean(SIG, axis=1), (SIG.shape[1], 1))
    else:
        # For other values of KNL, subtract the result of the smooth function from SIG
        out = SIG - smooth(SIG, KNL)

    return out  # Return the resulting array 'out'