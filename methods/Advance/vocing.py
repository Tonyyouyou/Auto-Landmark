from scipy.io.wavfile import read
import numpy as np

def calculate_voicing_basic(wav_path, frame_length=2048, hop_length=512, energy_threshold=0.01):
    # 读取WAV文件
    sample_rate, audio = read(wav_path)
    if audio.dtype != np.int16:
        audio = (audio * 32768).astype(np.int16)

    # 计算短时能量
    frames = [audio[i:i+frame_length] for i in range(0, len(audio), hop_length)]
    energy = np.array([np.sum(frame.astype(float)**2) for frame in frames])
    
    # 标准化能量
    normalized_energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))
    
    # 有声/无声判断
    voicing_status = np.where(normalized_energy > energy_threshold, 1, 0)
    return voicing_status

# 使用提供的音频文件路径
wav_path = '/home/561/xz4320/Auto-Landmark/methods/Advance/test_example/spx2.wav'

# 计算voicing状态
voicing_basic = calculate_voicing_basic(wav_path)

# 输出结果
print(voicing_basic[:200])  # 打印前200个点查看

print(voicing_basic)