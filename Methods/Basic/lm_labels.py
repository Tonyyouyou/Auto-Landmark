import numpy as np

def lm_labels(LMNDXS):
    LABELS = ['-g', '+g',
              '-b', '+b',
              '-s', '+s',
              '-f', '+f',
              '-v', '+v',
              '-p', '+p',
              '-j', '+j']

    if len(LABELS) < 99:
        LABELS += list(np.tile('??', (99 - len(LABELS), 1)))
    LABELS[-2:] = ['+T', '-T']
    LABELS[40:42] = [' V', ' F']

    if LMNDXS == '?':
        print('lmchars_Nx2 = lm_labels(LMNDXS_N)')
        print('lmndxs_N = lm_labels(LMCHARS_Nx2)')
        return

    UNRECOG = '?'
    if isinstance(LMNDXS, str):
        lmchrs = lm_ndx_labels(LMNDXS, LABELS)
    else:
        lmchrs = lm_label_ndxs(LMNDXS, LABELS, UNRECOG)

    return lmchrs


def lm_label_ndxs(LMNDXS, LABELS, UNRECOG):
    CHECK_TYPES = True  # Set to False if type checking is not required

    if CHECK_TYPES:
        if not all(isinstance(x, int) and x > 0 for x in LMNDXS):
            raise ValueError('Landmark indices must be Integer > 0.')

    if LMNDXS.ndim > 1:
        LMNDXS = LMNDXS.flatten()
        print('Landmark codes should be a COLUMN VECTOR; converted.')

    lmchrs = np.zeros((len(LMNDXS), 2), dtype=str)
    invalid_indices = np.isnan(LMNDXS) | (LMNDXS > LABELS.shape[0])
    lmchrs[~invalid_indices, :] = LABELS[LMNDXS[~invalid_indices], :]
    if np.any(invalid_indices):
        lmchrs[invalid_indices, :] = UNRECOG

    return lmchrs


def lm_ndx_labels(LMCHRS, LABELS):
    CHECK_TYPES = True  # Set to False if type checking is not required

    if CHECK_TYPES:
        if LMCHRS.ndim > 2 or (LMCHRS.shape[1] != 2 and LMCHRS.shape[0] != 2):
            raise ValueError('Landmark labels must be Nx2 characters.')

    if LMCHRS.shape[1] != 2:
        LMCHRS = LMCHRS.T
        print('Landmark labels should be a width-2 COLUMN (Nx2); transposed.')

    lmndxs = np.zeros((LMCHRS.shape[0],), dtype=int)
    ndx = 0
    for lbl in LABELS:
        ndx += 1
        matches = np.where(np.all(LMCHRS == lbl, axis=1))[0]
        lmndxs[matches] = ndx

    invalid_labels = np.all(LABELS[lmndxs, :] == '?', axis=1)
    lmndxs[invalid_labels] = np.nan

    return lmndxs
