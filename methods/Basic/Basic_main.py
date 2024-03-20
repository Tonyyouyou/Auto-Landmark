import os
import sys
import argparse
import yaml
from Landmarks_func import extract_all_landmarks

current_file_path = os.path.abspath(__file__)
auto_landmark_root = current_file_path
while os.path.basename(auto_landmark_root) != 'Auto-Landmark' and auto_landmark_root != '/':
    auto_landmark_root = os.path.dirname(auto_landmark_root)
data_dir_path = os.path.join(auto_landmark_root, 'data')

if data_dir_path not in sys.path:
    sys.path.append(data_dir_path)

from Arrange_landmark_output import process_result_dict_withtime, process_result_dict_notime


def parse_arguments():
    parser = argparse.ArgumentParser(description='Define Landmark you want to extract')
    parser.add_argument('--config', type=str, default='data_config.yaml', help='Path to the YAML config file')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    whole_landmark_support = ['g+', 'g-', 's+', 's-', 'b+', 'b-', 'v+','v-', 'f+', 'f-', 'p+', 'p-']

    segment_data_folder = config['data_folder']
    landmark_type = config['landmark_type']
    output_path = config['output_path']

    landmark_remove = list(set(whole_landmark_support) - set(landmark_type))

    landmark_result = {}

    audio_files_list = os.listdir(segment_data_folder)
    skip_file_list = [] 

    for file in audio_files_list:
        print(file)
        audio_file_path = os.path.join(segment_data_folder, file)
        try:
            audio_landmark = extract_all_landmarks(audio_file_path, landmark_remove)
            landmark_result[audio_file_path] = audio_landmark

        except Exception as e: 
            print(f"Failed to extract landmarks for {file}: {e}")
            skip_file_list.append(file)  

    process_result_dict_withtime(landmark_result, output_path)
    process_result_dict_notime(landmark_result, output_path)

    if skip_file_list:
        print("Failed files:", skip_file_list)
        failed_files_output_path = os.path.join(output_path, 'failed_files.txt')
        with open(failed_files_output_path, 'w') as f:
            for file_name in skip_file_list:
                f.write(f"{file_name}\n")

if __name__ == "__main__":
    main()