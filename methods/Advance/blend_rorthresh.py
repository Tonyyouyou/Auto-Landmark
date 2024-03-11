# daijiao
import numpy as np

# 输出和matlab一致
def blend_rorthresh(RORTHR1, RORTHR2, DURATION2):
    STD_DUR1 = 1000 * 1

    newthr = (STD_DUR1 * RORTHR1 + DURATION2 * RORTHR2) / (STD_DUR1 + DURATION2)
    nan_indices = np.isnan(newthr)
    # Replace NaN values in newthr with values from RORTHR2
    nan_indices = np.isnan(newthr)
    newthr[nan_indices] = RORTHR2[nan_indices]

    # # Replace NaN values in newthr with values from RORTHR1
    nan_indices = np.isnan(newthr)
    if not np.all(nan_indices == False):
        newthr[nan_indices] = RORTHR1[nan_indices]
    return newthr

# test
# RORTHR1 = np.array([1.0, 2.0, np.nan, np.nan, 5.0])
# RORTHR2 = np.array([6.0, np.nan, 8.0, 9.0, 10.0])
# DURATION2 = 2000  # DURATION2_msec in milliseconds
#
# newthr = blend_rorthresh(RORTHR1, RORTHR2, DURATION2)
# print(newthr)