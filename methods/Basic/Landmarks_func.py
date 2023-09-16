##############################################
######### Author: Zhaocheng (David) Huang  #########
##############################################
# Modified by Xiangyu(Tony) Zhang

import numpy as np
import pandas as pd

from scipy import signal
from .thinkdsp import read_wave
# import thinkdsp as dsp
from scipy.signal import butter, lfilter, freqz
from scipy.fftpack import fft, rfft, dct, fftshift
from scipy.signal import find_peaks
from numpy.lib.stride_tricks import as_strided

# ==========================================================================
# hann_len   # Hanning window length (ms)
# hann_shift # Hanning window shift (ms)
# cp_sm      # coarse processing smoothing length (ms)
# fp_sm      # fine processing smoothing length (ms)
# cp_dt      # coarse processing derivative length (ms)
# fp_dt      # fine processing derivative length (ms)

class Landmarks_base:
    
    def __init__(self, wav_dir):
        self.wav_dir = wav_dir
    
    # Reading wav file
    def wav_read_info(self):
        audioFile = read_wave(self.wav_dir)
        Fs = audioFile.framerate    # Sampling rate
        Y = audioFile.ys            # Audio sample vector
        T = audioFile.ts            # Time vector (sec)
        Y_len = len(Y)
        eps = 1e-20
        ms_s = int(Fs//1e3)         # Number of samples in a ms
        return [Fs, Y, T, Y_len, ms_s]
    
    # Extracting each band energy
    def band_energy(self, hann_len=6, hann_shift=1, Nfft=512):  # Default: 6 ms Hanning window with 1 ms shift
        [Fs, Y, T, Y_len, ms_s] = self.wav_read_info()
        hann_window = np.hanning(hann_len*ms_s)           # Creating a Hanning window

        ### Modified by David from here
        frame_length = hann_len * ms_s
        hop_length = hann_shift * ms_s
        num_frames = int(1 + (len(Y) - frame_length) / hop_length)
        row_stride = Y.itemsize * hop_length
        col_stride = Y.itemsize

        # buffer like in matlab
        Y_buffer = as_strided(Y, shape=(num_frames, frame_length), strides=(row_stride, col_stride)) * hann_window

        # zero-padding before fft
        Y_zero_paded = np.concatenate([Y_buffer, np.zeros((Y_buffer.shape[0], Nfft - Y_buffer.shape[1]))], axis=1)


        def fft_row(a, Nfft):
            x = abs(fft(a, n=Nfft)[:Nfft//2])  # FFT with Nfft points
            return x
        Y_fft = np.apply_along_axis(fft_row, 1, Y_zero_paded, Nfft=Nfft)

        fRange = np.arange(Nfft)
        freq = fRange * Fs / Nfft  # Frequency vector
        freq = freq[:Nfft // 2]  # Selecting half of spectrum for FFT

        Y_fft_energy = Y_fft ** 2 # calculate the enery before summation within each band

        band_freqs = dict({1: [-1, 400],
                           2: [800, 1500],
                           3: [1200, 2000],
                           4: [2000, 3500],
                           5: [3500, 5000],
                           6: [5000, 8000]})

        energy_bands = np.zeros((Y_fft_energy.shape[0], len(band_freqs)))
        for i in range(1, len(band_freqs)+1, 1): # mean band energies in dB,
            energy_bands[:, i-1] = Y_fft_energy[:, (freq > band_freqs[i][0]) & (freq < band_freqs[i][1])].mean(axis=1)

        energy_dataframe = pd.DataFrame({'Frame_Sample': np.arange(num_frames) * hop_length,
                                          'band_1_E': energy_bands[:, 0],
                                          "band_2_E": energy_bands[:, 1],
                                          "band_3_E": energy_bands[:, 2],
                                          "band_4_E": energy_bands[:, 3],
                                          "band_5_E": energy_bands[:, 4],
                                          "band_6_E": energy_bands[:, 5],
                                          })

        return energy_dataframe
    
    # Smoothing (sm), Differentiaion (dt) and Peak selection in both Coarse (cp) and Fine (fp) Processing
    def band_peak_detect(self, thr, cp_sm=16, fp_sm=8, cp_dt=20, fp_dt=10):
        # cp_sm = int(20/1), fp_sm = int(10/1), cp_dt = int(50/1), fp_dt = int(26/1))
        energy_dataframe = self.band_energy()
        peak_dict = {}

        # Coarse Smoothing and Differentiaion, added centering to avoid delay introduced by either smoothing or differentiation
        log_band_E_sm = 10*np.log10(energy_dataframe.iloc[:, 1:].rolling(window=cp_sm, center=True).mean())
        log_band_E_sm_diff = log_band_E_sm.diff(periods=cp_dt).shift(periods=-int(cp_dt/2), axis=0).fillna(0) # 6-point ROR and centering to avoid any delay

        # Fine Smoothing and Differentiaion
        log_band_E_sm_f = 10 * np.log10(energy_dataframe.iloc[:, 1:].rolling(window=fp_sm, center=True).mean())
        log_band_E_sm_f_diff = log_band_E_sm_f.diff(periods=fp_dt).shift(periods=-int(fp_dt/2), axis=0).fillna(0)

        thr_g = 8 # for coarse processing
        thr_g_f = 5 # for fine processing
        peak_dist = 64 # threshods for peaks

        # Coarse Peak detection
        peaks_p = log_band_E_sm_diff.apply(lambda x: find_peaks(x, height=thr, distance=peak_dist), axis=0)
        peaks_n = log_band_E_sm_diff.apply(lambda x: find_peaks(-x, height=thr, distance=peak_dist), axis=0)

        # Fine Peak detection
        peaks_p_f = log_band_E_sm_f_diff.apply(lambda x: find_peaks(x, height=thr, distance=peak_dist), axis=0)
        peaks_n_f = log_band_E_sm_f_diff.apply(lambda x: find_peaks(-x, height=thr, distance=peak_dist), axis=0)

        peaks_p['band_1_E'] = find_peaks(log_band_E_sm_diff['band_1_E'], height=thr_g, distance=peak_dist)
        peaks_n['band_1_E'] = find_peaks(-log_band_E_sm_diff['band_1_E'], height=thr_g, distance=peak_dist)

        peaks_p_f['band_1_E'] = find_peaks(log_band_E_sm_diff['band_1_E'], height=thr_g_f, distance=peak_dist)
        peaks_n_f['band_1_E'] = find_peaks(-log_band_E_sm_diff['band_1_E'], height=thr_g_f, distance=peak_dist)

        for i in range(1, 7, 1):
            # Saving all the resulting peaks
            peak_dict["band_" + str(i)] = {"peak_p": peaks_p["band_" + str(i) + '_E'][0],
                                           "peak_n": peaks_n["band_" + str(i) + '_E'][0], \
                                           "peak_p_f": peaks_p_f["band_" + str(i) + '_E'][0], \
                                           "peak_n_f": peaks_n_f["band_" + str(i) + '_E'][0], \
                                           "x_peak_p": peaks_p["band_" + str(i) + '_E'][1]['peak_heights'], \
                                           "x_peak_n": peaks_n["band_" + str(i) + '_E'][1]['peak_heights'], \
                                           "x_peak_p_f": peaks_p_f["band_" + str(i) + '_E'][1]['peak_heights'], \
                                           "x_peak_n_f": peaks_n_f["band_" + str(i) + '_E'][1]['peak_heights']}

        return peak_dict, energy_dataframe.shape[0]
    
class Localized_peaks:
    
    def __init__(self, Landmarks_base_obj, thr):
        self.peak_dict, self.length = Landmarks_base_obj.band_peak_detect(thr=thr)

    def localPeak(self):
        L = self.length
        peak_dict = self.peak_dict
        landmarks = {}
        localized_peak_p_dict = {}
        localized_peak_n_dict = {}

        for i in range(1, 7, 1):
            # need to generalize across other bands
            peaks_cp = peak_dict['band_' +  str(i)]['peak_p']
            peaks_fp = peak_dict['band_' +  str(i)]['peak_p_f']
            index_fp = np.zeros((peaks_fp.shape[0]), dtype=bool)
            for i_fp, fp_step in enumerate(peaks_fp):
                if ((fp_step > peaks_cp - 15) & (fp_step < peaks_cp + 15)).sum() >= 1:
                    index_fp[i_fp] = True

            peaks_cn = peak_dict['band_' +  str(i)]['peak_n']
            peaks_fn = peak_dict['band_' +  str(i)]['peak_n_f']
            index_fn = np.zeros((peaks_fn.shape[0]), dtype=bool)
            for i_fn, fn_step in enumerate(peaks_fn):
                if ((fn_step > peaks_cn - 15) & (fn_step < peaks_cn + 15)).sum() >= 1:
                    index_fn[i_fn] = True

            localized_peak_p_dict['band_' +  str(i)] = peaks_fp[index_fp]
            localized_peak_n_dict['band_' + str(i)] = peaks_fn[index_fn]

        pp_arr = localized_peak_p_dict['band_1']
        pn_arr = localized_peak_n_dict['band_1']

        # g+ and g- landmarks
        landmarks['g+'] = pp_arr
        landmarks['g-'] = pn_arr

        peak_p = pd.DataFrame({'time': pp_arr,
                               'lmk': 'g+',
                               'direction': 1})
        peak_n = pd.DataFrame({'time': pn_arr,
                               'lmk': 'g-',
                               'direction': -1})

        # voicing based on g landmark
        peak_file = pd.concat((peak_p, peak_n), axis=0).sort_values(['time']) # would need to find a way to have a more reliable estmiate of voicing and unvoicing.
        peak_file['start'] = peak_file['time']
        peak_file['end'] = peak_file['time'].shift(periods=-1)
        peak_file['transition'] = peak_file['direction'].diff().shift(periods=-1)
        s_region = np.array(peak_file.loc[peak_file['transition'] == -2, ['start', 'end']]).astype(int)
        b_region = np.array(peak_file.loc[peak_file['transition'] != -2, ['start', 'end']]).astype(int)
        #print('b_region',b_region)
        # b_region = np.array(peak_file.loc[peak_file['transition'] == 2, ['start', 'end']]).astype(int) # strict for unvoiced

        voicing_g = pd.DataFrame(np.zeros((L, 8)))
        voicing_g.columns=['s_voicing', 'b_voicing', 'pos', 'neg', 's+', 'b+', 's-', 'b-']

        for i in range(s_region.shape[0]):
            voicing_g.loc[s_region[i, 0]:s_region[i, 1], 's_voicing'] = 1

        for i in range(b_region.shape[0]):
            voicing_g.loc[b_region[i, 0]:b_region[i, 1], 'b_voicing'] = 1

        # s and b landmarks
        band_list = list(localized_peak_p_dict.keys())[1:]
        # print('band_list', band_list)
        peaks_band_p = pd.DataFrame(np.zeros((L, 5)), columns=band_list)
        peaks_band_n = pd.DataFrame(np.zeros((L, 5)), columns=band_list)

        peaks_cdt_sb = pd.DataFrame(np.zeros((L, 2)), columns=['peaks_cdt_p', 'peaks_cdt_n'])

        for i, band in enumerate(band_list):
            peaks_band_p.loc[localized_peak_p_dict[band], band] = 1
            peaks_band_n.loc[localized_peak_n_dict[band], band] = 1

        aa_sb = peaks_band_p.sum(axis=1)[peaks_band_p.sum(axis=1) >= 1].index.tolist()
        bb_sb = peaks_band_n.sum(axis=1)[peaks_band_n.sum(axis=1) >= 1].index.tolist()
        for a in aa_sb:
            if peaks_band_p.loc[a-20:a+20, :].any().sum() >= 3: # look at 1 in each column before sum for band counts
                peaks_cdt_sb['peaks_cdt_p'].iloc[a] = 1
        for a in bb_sb:
            if peaks_band_n.loc[a-20:a+20, :].any().sum() >= 3: # look at 1 in each column before sum for band counts
                peaks_cdt_sb['peaks_cdt_n'].iloc[a] = 1

        #merge peaks if occuring within 20ms
        peaks_sb = peaks_cdt_sb.apply(lambda x: find_peaks(x, height=0.5, distance=20), axis=0)

        # find s+ or b+
        voicing_g.loc[peaks_sb['peaks_cdt_p'][0], ['pos']] = 1 # positive landmarks, s or b
        voicing_g.loc[peaks_sb['peaks_cdt_n'][0], ['neg']] = 1 # negative landmarks, s or b

        voicing_g['s+'] = voicing_g['s_voicing'] * voicing_g['pos']
        voicing_g['b+'] = voicing_g['b_voicing'] * voicing_g['pos']
        voicing_g['s-'] = voicing_g['s_voicing'] * voicing_g['neg']
        voicing_g['b-'] = voicing_g['b_voicing'] * voicing_g['neg']

        peaks_band_p.columns = [x + '_p' for x in band_list]
        peaks_band_n.columns = [x + '_n' for x in band_list]
        peaks_band = pd.concat([peaks_band_p, peaks_band_n], axis=1) # combine rises and falls together for f and v

        peaks_fv_p = peaks_band.loc[:, ['band_4_p', 'band_5_p', 'band_6_p', 'band_2_n', 'band_3_n']]
        peaks_fv_n = peaks_band.loc[:, ['band_4_n', 'band_5_n', 'band_6_n', 'band_2_p', 'band_3_p']]

        peaks_cdt_fv = pd.DataFrame(np.zeros((L, 2)), columns=['peaks_cdt_p', 'peaks_cdt_n'])
        aa_fv = peaks_fv_p.sum(axis=1)[peaks_fv_p.sum(axis=1) >= 1].index.tolist()
        bb_fv = peaks_fv_n.sum(axis=1)[peaks_fv_n.sum(axis=1) >= 1].index.tolist()
        for a in aa_fv:
            if peaks_fv_p.loc[a - 20:a + 20,
               :].any().sum() >= 3:  # look at 1 in each column before sum for band counts
                peaks_cdt_fv['peaks_cdt_p'].iloc[a] = 1
        for a in bb_fv:
            if peaks_fv_n.loc[a - 20:a + 20,
               :].any().sum() >= 3:  # look at 1 in each column before sum for band counts
                peaks_cdt_fv['peaks_cdt_n'].iloc[a] = 1

        # block out f, v when they overlap with s and b, which take priorities.
        cdt_sb = peaks_cdt_sb.apply(lambda x: find_peaks(x, height=0.5, distance=1), axis=0)
        for i, i_x in enumerate(cdt_sb.keys().tolist()):
            for ii_x in cdt_sb[i_x][0]:
                if ii_x + 20 < L:
                    peaks_cdt_fv.loc[ii_x - 20: ii_x + 20, i_x] = 0

        # merge peaks if occuring within 20ms
        peaks_fv = peaks_cdt_fv.apply(lambda x: find_peaks(x, height=0.5, distance=20), axis=0)

        voicing_fv = pd.DataFrame(np.zeros((L, 8)))
        voicing_fv.columns=['s_voicing', 'b_voicing', 'pos', 'neg', 'v+', 'f+', 'v-', 'f-']
        voicing_fv.loc[:, ['s_voicing', 'b_voicing']] = voicing_g.loc[:, ['s_voicing', 'b_voicing']]

        # find f and v
        voicing_fv.loc[peaks_fv['peaks_cdt_p'][0], ['pos']] = 1  # positive landmarks, f or v
        voicing_fv.loc[peaks_fv['peaks_cdt_n'][0], ['neg']] = 1  # negative landmarks, f or v

        voicing_fv['v+'] = voicing_fv['s_voicing'] * voicing_fv['pos']
        voicing_fv['f+'] = voicing_fv['b_voicing'] * voicing_fv['pos']
        voicing_fv['v-'] = voicing_fv['s_voicing'] * voicing_fv['neg']
        voicing_fv['f-'] = voicing_fv['b_voicing'] * voicing_fv['neg']


        landmarks['g+'] = pp_arr / 1000
        landmarks['g-'] = pn_arr / 1000
        landmarks['s+'] = np.array(voicing_g['s+'][voicing_g['s+'] == 1].index) / 1000
        landmarks['s-'] = np.array(voicing_g['s-'][voicing_g['s-'] == 1].index) / 1000
        landmarks['b+'] = np.array(voicing_g['b+'][voicing_g['b+'] == 1].index) / 1000
        landmarks['b-'] = np.array(voicing_g['b-'][voicing_g['b-'] == 1].index) / 1000
        landmarks['v+'] = np.array(voicing_fv['v+'][voicing_fv['v+'] == 1].index) / 1000
        landmarks['v-'] = np.array(voicing_fv['v-'][voicing_fv['v-'] == 1].index) / 1000
        landmarks['f+'] = np.array(voicing_fv['f+'][voicing_fv['f+'] == 1].index) / 1000
        landmarks['f-'] = np.array(voicing_fv['f-'][voicing_fv['f-'] == 1].index) / 1000


        return landmarks, pp_arr/1000, pn_arr/1000, peak_file

class P_landmark:
    def __init__(self, Landmarks_base_obj, thr):
        self.Landmarks_base_obj = Landmarks_base_obj(thr=thr)
        self.label = 'p'

    def find_P_landmark(self):
        [Fs, Y, T, Y_len, ms_s] = self.Landmarks_base_obj.wav_read_info()
        frame_shift = 10  # Frame shift in ms
        frame_L = 25  # Frame length in ms
        delta = int(Fs / 1000 * frame_shift)  # Number of samples per shift
        N = int(Fs / 1000 * frame_L)  # Number of samples per window
        n_frames = int(len(Y) / delta)  # Number of frames

            # Performing auto correlation on each frame
        Ef = np.array([])
        for fr in range(n_frames):
            if ((fr * delta) + N) < len(Y):
                x = np.array(Y[fr * delta: (fr * delta) + N])  # selected frame
                Rxx = np.array([])
                for el in range(N):
                    temp1 = 1 / (N - el) * sum(x[el:] * x[:N - el])
                    Rxx = np.append(Rxx, temp1)
                temp2 = sum(Rxx ** 2) / N
                Ef = np.append(Ef, temp2)

        shift = 10  # Shift of landmarks in ms
        shift = int(shift * Fs / 1000)  # Shift samples

        # Up-sampling to match input size
        up_Ef = np.array([])
        scale = int(len(Y) / len(Ef)) + 1
        for i in Ef:
            up_Ef = np.append(up_Ef, i * np.ones([1, scale]))
        up_Ef = pd.DataFrame(up_Ef)

        # Smoothing the correlation function
        up_Ef_sm = up_Ef.rolling(window=500, center=True).mean().fillna(0)
        up_Ef_sm = up_Ef_sm.rolling(window=1500, center=True).mean().fillna(0)

        # Normalizing the output and matching the length to Y
        up_Ef_sm_norm = (up_Ef_sm / np.max(up_Ef_sm)).iloc[:len(Y)]

        # Shifting
        up_Ef_sm_norm = pd.concat([up_Ef_sm_norm.iloc[shift:], pd.DataFrame(np.zeros([1, shift])).T], axis=0).reset_index()
        up_Ef_sm_norm.drop(["index"], axis=1, inplace=True)

        # Changing output to binary
        thr = np.mean(up_Ef_sm_norm[0]) / 30  # Threshold for onset/offset detection
        bin_up_Ef_sm_norm = np.array([0 if x <= thr else 1 for x in list(up_Ef_sm_norm[0])])

        # detecting jumps or onset/offset
        up_Ef_sm_norm_jumps = np.append(bin_up_Ef_sm_norm[1:] - bin_up_Ef_sm_norm[:-1], [0])

        # getting the index and time of positive and negative jumps
        get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]

        pLandmark_p_idx = get_indexes(1, up_Ef_sm_norm_jumps)
        pLandmark_p_time = T[np.array(pLandmark_p_idx)] * 1000

        pLandmark_n_idx = get_indexes(-1, up_Ef_sm_norm_jumps)
        pLandmark_n_time = T[np.array(pLandmark_n_idx)] * 1000
