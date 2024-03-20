import os 
import argparse
import yaml
def parse_arguments():
    parser = argparse.ArgumentParser(description='Define Landmark you want to extract')
    parser.add_argument('--config', type=str, default='data_config.yaml', help='Path to the YAML config file')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    landmark_type = config['landmark_type']
    print(landmark_type)


if __name__ == "__main__":
    main()