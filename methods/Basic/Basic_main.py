import os
from Landmarks_func import extract_all_landmarks

def parse_arguments():
    parser = argparse.ArgumentParser(description='Define Landmark you want to extract')
    parser.add_argument('--config', type=str, default='data_config.yaml', help='Path to the YAML config file')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    segment_data_folder = config['data_folder']
    landmark_type = config['landmark_type']

    audio_files = os.listdir(segment_data_folder)

if __name__ == "__main__":
    main()