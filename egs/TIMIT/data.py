import os
import sys

# 动态地将Auto-Landmark项目的根目录以及Basic方法的目录添加到sys.path中
project_root = '/home/561/xz4320/Auto-Landmark'
basic_methods_dir = os.path.join(project_root, 'methods', 'Basic')
sys.path.append(project_root)
sys.path.append(basic_methods_dir)  # 确保thinkdsp可以被导入
from Landmarks_func import extract_all_landmarks

def extract_landmarks_for_directory(root_dir, output_file_path):
    # 打开输出文件
    count = 0
    with open(output_file_path, 'w') as output_file:
        # 遍历根目录下的所有文件
        for dirpath, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith('.WAV'):
                    # 构建完整的文件路径
                    full_path = os.path.join(dirpath, filename)
           
                    # 提取landmarks
                    landmarks_dict = extract_all_landmarks(full_path,landmark_remove=None)
                    count += 1
                    print(count)
                    # 对landmarks按时间排序
                    landmarks_list = []
                    for landmark, timestamps in landmarks_dict.items():
                        for timestamp in timestamps:
                            landmarks_list.append((timestamp, landmark))

                    # Sort the list by timestamp
                    sorted_landmarks = sorted(landmarks_list, key=lambda x: x[0])

                    # 将文件路径和排序后的landmarks写入输出文件
                    output_line = f"{full_path}: " + " ".join([f"{time}:{landmark}" for time, landmark in sorted_landmarks])
                    output_file.write(output_line + "\n")

if __name__ == "__main__":
    root_directory = '/g/data/wa66/Xiangyu/Data/TIMIT/TEST'
    output_text_file = '/g/data/wa66/Xiangyu/Landmark_dataset/auto-landmark/basic/test_landmark_time.txt'
    extract_landmarks_for_directory(root_directory, output_text_file)
