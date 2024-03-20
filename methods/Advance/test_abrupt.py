# 用来测试abrupt函数对不对

from wavread_freq import wavread_freq
from hpfilt_std import hpfilt_std
from abrupt_events import abrupt_events
from band_pivots import pf_to_pvts
from band_pivots import atten_near_pvts
from band_pivots import fpf_to_fpvts
from band_pivots import band_pivots
import numpy as np
from smoothspecbands import smoothspecbands
from band_energies import demedfilt1up
from band_energies import band_energies
from abrupt_events import cons_pivots
from abrupt_events import abrupt_events
from abrupt_lms import contour_to_laryng
from abrupt_lms import move_after_gs
from abrupt_lms import merged_plms_voicing
from abrupt_lms import abrupt_lms

# audio_path = 'E://UNSW/project/Auto-Landmark-main/methods/Basic/test_short.wav'
# std_fs = 16000
# signal, sr = wavread_freq(audio_path,  std_fs) # signal 输入跟matlab一致
# hpsig = hpfilt_std(signal, std_fs)  # hpsig 输出数值与matlab一致，但是维度变成2维，输入的时候降一维
# hpsig = hpsig.reshape(1, -1)

# 此后hpsig做了smooth处理，跟matlab值不同了
# 我直接复制了matlab smooth 处理后的hpsig，方便观察abrupt里面的数据处理对错
# file_path = 'hpsig.txt'
# hpsig = np.loadtxt(file_path) # 是np 一维数组

MAX_WDW = 20
min_bands = 3
LF_EARLY = 10
LF_LATE = 8
# smpksf_path = 'smpksf.txt'
# smpksf = np.loadtxt(smpksf_path)

# pf_path  = 'pf.txt'
# pf = np.loadtxt(pf_path)

# pvts = pf_to_pvts(pf,MAX_WDW,min_bands,smpksf)
# print(pvts)
# pvts = np.loadtxt('pvts.txt')
#
# thresh = np.loadtxt('thresh.txt')
# fpf_path = 'fpf.txt'
# fpf = np.loadtxt(fpf_path)
# fpf = atten_near_pvts(fpf,pvts,MAX_WDW,thresh)
# print(smpksf)
# np.set_printoptions(threshold=np.inf)
# print(fpf)

# B_EN_ROR_C = np.loadtxt('B_EN_ROR_C.txt')
# B_EN_ROR_F = np.loadtxt('B_EN_ROR_F.txt')
#
# pvts, peaks = band_pivots(B_EN_ROR_C, B_EN_ROR_F)
# print(pvts)

# RATE = 16000
# STD_WSPACE = 16
# NDXRANGES = np.loadtxt('NDXRANGES.txt')
# sg = np.loadtxt('sg-impsg.txt')
# bands = smoothspecbands(sg,RATE,NDXRANGES,STD_WSPACE)
# print(bands)

# mfknl = 6
# sgm = np.loadtxt('sgm.txt')
# y = demedfilt1up(sgm,mfknl)

# # abrupt events 的测试代码
# age = 'adult'
# rate = 16000
# signal = np.loadtxt('SIGNAL.txt')
# PREV_RORTHR = np.nan
#
# lms, bandrate, wdwlen, rorthr, bands, bandsf = abrupt_events(SIGNAL=signal,RATE=rate,AGE=age,PREV_RORTHR=PREV_RORTHR,NOMED='')
# print(lms)

# PLUS_G = 2
# MINUS_G = 1
# VRATE = 125
# VOICING = np.loadtxt('VOICING.txt')
# vlms, gmarks = contour_to_laryng(VOICING,PLUS_G,MINUS_G,VRATE)
# #print(vlms)
#
# lms = np.loadtxt('lms.txt')
# a = [4,8]
# vary = 125
#
# lms, ndxspg1,ndxsmg1 = move_after_gs(lms,a,gmarks,vary, MINUS_G,PLUS_G)
# print(lms)

# VRATE = 125
# plms = np.loadtxt('plms.txt')
# ans = np.loadtxt('ans.txt')
# per = np.loadtxt('per.txt')
# mlp = merged_plms_voicing(plms,ans,per,VRATE)
# print(ans.dtype)
#print(mlp)

from wavread_freq import wavread_freq
hpsig = np.loadtxt('/home/561/xz4320/Auto-Landmark/methods/Advance/test_example/hpsig.txt')

# hpsig = wavread_freq('/g/data/wa66/Xiangyu/Data/TIMIT/TRAIN/DR1/FCJF0/SA1.WAV', 16000)
Fs = 16000
age = 'adult'
PREV_RPRTHR = None
# voicing = np.loadtxt('voicing.txt')
pts = 125

lm = abrupt_lms(hpsig, Fs, age, PREV_RPRTHR, '',None,pts)
print(lm)