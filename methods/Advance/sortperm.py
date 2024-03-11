import numpy as np

# 输入一致，输出结果比matlab少1（python序列从0开始）
def sortperm(ARR, DIM=None):
    """
    Sorts the array ARR along the specified dimension DIM and returns the permutation indices.
    If DIM is not provided, sorts along the last axis.
    """
    # Check for the special case where usage information is requested
    if isinstance(ARR, str) and ARR == '?' and not DIM:
        print('sp_KxL = sortperm(ARR_KxL, <DIM>)')
        return

    # Sorting and getting permutation indices
    if DIM is None:
        sp = np.argsort(ARR,axis=0)
    else:
        sp = np.argsort(ARR, axis=DIM-1)

    return sp

# arr2 = np.array([[3, 1], [4, 1], [5, 9]])
# arr1 = np.array([3, 1, 4, 1, 5])
# arr3 = np.array([9, 0, -7, 5, 3, 8, -10, 4, 2])
# sp1 = sortperm(arr2,2)
# print(sp1)