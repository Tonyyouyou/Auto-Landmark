import os
import argparse
import yaml
from pydub import AudioSegment

class segment_audio:
    def __init__(self, data_type, data_path, output_path):
        self.data_type = data_type
        self.data_path = data_path
        self.output_path = output_path

    def split_audio(self, file_path, start_time, end_time, output_path, output_name, format='wav'):
        #Input unit should be second

        audio = AudioSegment.from_file(file_path, format=format)
        
        start_time = start_time * 1000
        end_time = end_time * 1000

        extract = audio[start_time:end_time]
        output_path = os.path.join(output_path, output_name)
        extract.export(output_path, format="wav")


    def process_line_segment(self, line):
        parts = line.split()
        if len(parts) != 4:
            raise ValueError(f"Unexpected format in line: {line}")
        key = parts[0]
        value = (parts[1], float(parts[2]), float(parts[3]))
        return key, value

    def process_line_wavscp(self, line):
        parts = line.split()
        key = parts[0]
        value = parts[1]
        return key, value

    def parse_txt_to_dict(self, segment_file, wavscp_file):
        segment_dict = {}
        wav_dict = {}
        with open(segment_file, 'r') as f:
            for line in f:
                key, value = self.process_line_segment(line.strip())
                segment_dict[key] = value
        
        with open(wavscp_file, 'r') as f:
            for line in f:
                key, value = self.process_line_wavscp(line.strip())
                wav_dict[key] = value

        return segment_dict, wav_dict

    def segment_all(self):
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)
            
        segments_file = os.path.join(self.data_path, self.data_type,'segments')
        wavscp_file = os.path.join(self.data_path, self.data_type,'wav.scp')

        segment_dict, wav_dict = self.parse_txt_to_dict(segments_file, wavscp_file)

        for key, value in segment_dict.items():
            file_path = wav_dict[value[0]]
            start_time = value[1]
            end_time = value[2]
            output_name = key + '.wav'
            self.split_audio(file_path, start_time, end_time, self.output_path, output_name)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Description of your data prep')
    parser.add_argument('--config', type=str, default='data_config.yaml', help='Path to the YAML config file')
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    data_type = config['segments']['data_type']
    data_path = config['segments']['data_path']
    output_path = config['segments']['output_path']
    segments = segment_audio(data_type, data_path, output_path)
    segments.segment_all()


if __name__ == "__main__":
    main()