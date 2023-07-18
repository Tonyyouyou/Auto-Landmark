import numpy as np
from scipy.signal import medfilt1

def landmarks(SIGNAL, Fs, MAX_F0, DRAW, AGESZ, *varargin):
    VTHR = 0.8
    MAXDEGASYM = 0.9
    STD_Fs = 16000
    MIN_PG_MS = 0.020
    MIN_PG_MS_FRIC = 0.050
    MAX_PF_PG = 0.400
    MIN_DUR = 1 / 40
    PREV_RORTHR = np.nan
    MIN_DVMEDF = 3
    HWID_SUSTVF = 0.032
    ENV_SIL = 1.0
    NSIL_LCL_MINAMPL = 1 / 100
    NSIL_GL_MINAMPL = 1 / 1000

    MINUS_G, PLUS_G, PLUS_B, MINUS_S, PLUS_F, MINUS_V, PLUS_P = lm_codes('MINUS_G', 'PLUS_G', 'PLUS_B', 'MINUS_S',
                                                                         'PLUS_F', 'MINUS_V', 'PLUS_P')

    if SIGNAL == '?':
        print('[lms_3xN, pertimes_sec_K, pervals_msec_K, env_Lm1, envthr, voicg_K, vthr] = ...')
        print('\tlandmarks(SIGNAL_L|Fname, <%g|Fs_Hz>, <""|AGE_GENDER|MAX_F0_Hz|MAXMIN_F0_Hz>, ...' % STD_Fs)
        print('\t\t<False|DRAW|True|other>, <"adult"|AGESZ|"child">)')
        return

    if MAX_F0 is None:
        if AGESZ.lower() == 'adult':
            MAX_F0, min_f0 = maxf0_std('n')
        elif AGESZ.lower() == 'child':
            MAX_F0, min_f0 = maxf0_std('i')
        else:
            MAX_F0, min_f0 = maxf0_std('')
    elif isinstance(MAX_F0, str):
        MAX_F0, min_f0 = maxf0_std(MAX_F0)
    else:
        min_f0 = []

    if Fs is None:
        Fs = STD_Fs

    SIGNAL = waveform_freq(SIGNAL, Fs)

    if isinstance(MAX_F0, float):
        min_f0 = MAX_F0 / 5
    elif len(MAX_F0) == 2:
        if MAX_F0[0] < MAX_F0[1]:
            print('F0 limits should be in the order [maximum, minimum]. Reversing & continuing.')
            MAX_F0 = [MAX_F0[1], MAX_F0[0]]
        min_f0 = MAX_F0[1]
    else:
        warnerr('MAX_F0 (argument for F0 limits) must have 0, 1, or 2 elements, not %d.' % len(MAX_F0))
        MAX_F0 = [max(MAX_F0), min(MAX_F0)]
        print('\tContinuing with [max, min] of argument = [%g, %g].' % tuple(MAX_F0))

    if not np.isvector(SIGNAL):
        raise ValueError('SIGNAL is not a vector. Signal from multiple channels is not supported.')

    hpsig = hpfilt_std(SIGNAL, Fs, AGESZ)

    env0_std(np.diff(hpsig), Fs)
    ishpsingle = isinstance(ans, np.float32)

    degspe = 1

    degspe > 0 and (ans >= max(grey_dilate(ans, odd(ENV_SIL * Fs)) * NSIL_LCL_MINAMPL,
                               max(ans) * NSIL_GL_MINAMPL))
    binary_open(ans, round(Fs * MIN_DUR))

    hpsig[:-1] = hpsig[:-1] * smooth(ans.astype(float), -round(Fs * MIN_DUR))
    if ishpsingle:
        hpsig = hpsig.astype(np.float32)

    pcont, env, voicg, envthr, ppcont, times, vthr, p0, specslope, deg_v, deghv = pitch_utt(hpsig, Fs, [], MAX_F0)
    ptstep = np.diff(times[:2])

    times = times[0] + ptstep * np.round((times - times[0]) / ptstep)

    ndxoff = int(np.round(times[0] / ptstep)) - 1
    tms = np.arange(ptstep * 1, ptstep * (ndxoff + 1))
    tms = np.concatenate((tms, times))

    rem(times[0] / ptstep - 1 / 2, 1)
    if np.abs(ans) < 0.499:
        warnerr("lmadult 'tms' vs. 'times' quantization break: %g." % ans)

    deghnr = deg_voicedhnr_std(voicg, vthr, times, ppcont)
    deg_voiced(voicg, vthr, times, pcont, ppcont, 1000 / MAX_F0[0], p0, specslope)

    max(medfilt1(ans, MIN_DVMEDF),
        medfilt1((0 < pcont) & (pcont / 1000 < 1 / min_f0) * ans, odd(1 + 2 * HWID_SUSTVF / ptstep))) > VTHR

    np.vstack((ans, deghnr > VTHR))
    voicing = np.zeros_like(ans)
    voicing[1, ndxoff:] = ans

    times = np.concatenate((times, times[-1] + ptstep * np.arange(1, ndxoff + 1)))

    _, jndxs = jump_lms(times[:len(pcont)], pcont, deghnr)

    if len(jndxs) > 0:
        maxfilt1(deghnr, 3)
        deghnr[jndxs - 1] = ans[jndxs - 1]
        deghnr[jndxs] = ans[jndxs]
        voicing[1, ndxoff:] = (deghnr > VTHR)
        voicing[1, :ndxoff] = deghnr[0]

    lms0, bandrate, wdwlen, rorthr = abrupt_lms(hpsig, Fs, AGESZ, PREV_RORTHR, '', voicing, 1 / ptstep)

    jlms = jump_lms(times[:len(pcont)], pcont, min([voicing[:, ndxoff:], deghnr]))

    coinc_gp = lambda LMS: np.where((LMS[1, :-1] == PLUS_G) & (LMS[1, 1:] == PLUS_P) & (np.diff(LMS[0, :]) == 0))[0]
    coinc_pg = lambda LMS: np.where((LMS[1, :-1] == PLUS_P) & (LMS[1, 1:] == PLUS_G) & (np.diff(LMS[0, :]) == 0))[0]

    if np.abs(Fs - STD_Fs) > 0.05 * STD_Fs:
        pfs, qfs = rat(STD_Fs / Fs, 0.025)
        resample(SIGNAL, pfs, qfs)
    else:
        SIGNAL += 0

    Flms = fricative_lms(ans)

    lms = lm_wellformed_seq(np.concatenate((lms0, jlms, Flms), axis=1))

    ndxgp = coinc_gp(lms)
    ndxgp = np.concatenate(([ndxgp + 1], [ndxgp]))
    lms[:, np.sort(ndxgp)] = lms[:, np.array(ndxgp).reshape(-1, 1)]

    ndxgs = np.where((lms[1, :-1] == PLUS_G) & (lms[1, 1:] == MINUS_S))[0]

    if len(ndxgs) > 0:
        round((lms[0, :] * bandrate - 1 / 2) / (bandrate * ptstep)) * ptstep
        diff_times = np.tile(times, (1, len(ndxgs))) - np.tile(ans[ndxgs + 1], (len(tms), 1))

        ans, ndxmins = np.abs(ans), np.argmin(np.abs(ans), axis=0)
        pergs = pcont[np.minimum(len(pcont) - 1, ndxmins)]
        pergs[pergs == 0] = np.inf

        if lms[0, ndxgs[0] + 1] < times[0] - ptstep:
            lms[0, ndxgs + 1] < times[0]
            pergs = np.concatenate((np.tile(np.inf, (1, np.sum(ans))), pergs))

        ndxgs1 = ndxgs[lms[0, ndxgs + 1] - lms[0, ndxgs] <= np.minimum(3 * pergs / 1000, MIN_PG_MS)]
        lms[1, ndxgs1 + 1] = MINUS_V

        ndxb = ndxgs[np.where(ndxgs > 1)[0]] - 1
        ndxb = ndxb[np.isin(ndxb, coinc_pg(lms))]
        ndxb[ndxb] = np.maximum(1, ndxb[ndxb] - 1)

        if len(ndxb) == 0:
            ndxgs2 = []
        else:
            lms[1, ndxb] == PLUS_B or lms[1, ndxb] == PLUS_F
            ndxb = ndxb[np.where(ans)[0]]
            ndxgs2 = np.where((lms[0, ndxgs2 + 1] - lms[0, ndxgs2] <= MIN_PG_MS_FRIC) &
                             (lms[0, ndxgs2] - lms[0, ndxb] <= MAX_PF_PG))[0]

            lms[1, ndxgs2 + 1] = MINUS_V

    lms = lm_wellformed_seq(lms)

    tvals = tms
    if DRAW:
        try:
            bw = None
            if isinstance(DRAW, str):
                bw = DRAW

            smenvdur = 2 * MIN_DUR
            env = smooth(env, -odd(2 * Fs * smenvdur))

            lm_draw(np.diff(hpsig), Fs, lms, pcont, ans * (ans > envthr), times[:len(pcont)],
                    zcrh1_std(hpsig, -Fs), bw)
            drawnow()
        except Exception as e:
            warnerr('Unable to draw LMs figure. Error msg. was:\n\t"%s"' % str(e))

    return lms, tvals, pcont, env, envthr, voicg, vthr

def lm_wellformed_seq(LMS):
    MINUS_G, PLUS_G, PLUS_P, MINUS_P = lm_codes('MINUS_G', 'PLUS_G', 'PLUS_P', 'MINUS_P')

    if isinstance(LMS, dict):
        lm_wellformed_seq(lm_structarr(LMS))
        wf = lm_structarr(ans)
        return wf

    wf = np.sort(LMS, axis=1)

    coinc_pg = lambda LMS: np.where((LMS[1, :-1] == PLUS_P) & (LMS[1, 1:] == PLUS_G) & (np.diff(LMS[0, :]) == 0))[0]
    coinc_mgp = lambda LMS: np.where((LMS[1, :-1] == MINUS_G) & (LMS[1, 1:] == MINUS_P) & (np.diff(LMS[0, :]) == 0))[0]

    def swap_cols(LMS, NDXS):
        out = LMS.copy()
        out[:, [NDXS + 1, NDXS]] = LMS[:, [NDXS, NDXS + 1]]
        return out

    ndxpg = coinc_pg(wf)
    wf = swap_cols(wf, ndxpg)
    ndxmgp = coinc_mgp(wf)
    wf = swap_cols(wf, ndxmgp)

    return wf
