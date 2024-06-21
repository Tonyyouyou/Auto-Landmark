import os
import sys

# 动态地将Auto-Landmark项目的根目录以及Basic方法的目录添加到sys.path中
project_root = '/home/561/xz4320/Auto-Landmark'
basic_methods_dir = os.path.join(project_root, 'methods', 'Basic')
sys.path.append(project_root)
sys.path.append(basic_methods_dir)  # 确保thinkdsp可以被导入
from Landmarks_func import extract_all_landmarks

full_path = '/g/data/wa66/Xiangyu/Data/TIMIT/TEST/DR1/FAKS0/SA1.WAV'
landmarks_dict = extract_all_landmarks(full_path,landmark_remove=None)
print(landmarks_dict)