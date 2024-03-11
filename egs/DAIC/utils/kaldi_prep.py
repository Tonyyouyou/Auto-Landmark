import os
import pandas as pd
import argparse
import yaml

class data_prep:
    def __init__(self, data_type, data_folder_path, label_csv_path, output_path, dataset_name = 'DAIC'):
        self.data_folder_path = data_folder_path
        self.ID_label_frame = pd.read_csv(label_csv_path)
        self.output_path = output_path
        self.data_type = data_type
        self.dataset_name = dataset_name
    
    def ID_to_Label(self):
        self.pairs_list = list(zip(self.ID_label_frame['Participant_ID'], self.ID_label_frame['PHQ8_Binary']))
        label_dict = dict(zip(self.ID_label_frame['Participant_ID'], self.ID_label_frame['PHQ8_Binary']))

        self.id2label_path = os.path.join(self.output_path, 'id2label')

        return label_dict
         
    def gather_infomration(self):
        dataset_name = self.dataset_name
        label_dict = self.ID_to_Label()
        # data_list = os.listdir(self.data_folder_path)
        data_list = list(label_dict.keys())

        TEXT = {}   #uttid - transcript
        WAV_SCP = {} #wavid - path
        UTT2SPK = {} #uttid - speaker
        SEGMENTS = {} #uttid (wavid start_time end_time)
        UTT2LABEL = {} #uttid - label

        for filename in data_list:
            filename = str(filename)
            wavid =  filename + '_'+ dataset_name + "_wav"
            wav_path = os.path.join(self.data_folder_path,filename, filename + "_AUDIO.wav")
            WAV_SCP[wavid] = wav_path

            csv_file_path = os.path.join(self.data_folder_path,filename ,filename + "_TRANSCRIPT.csv")
            sample_information = pd.read_csv(csv_file_path, sep='\t')

            for index, row in sample_information.iterrows():
                uttid = filename + '_utt' + '_' + dataset_name + '_' + str(index)
                TEXT[uttid] = row['value']
                UTT2SPK[uttid] = row['speaker']
                SEGMENTS[uttid] = (wavid, row['start_time'], row['stop_time'])
                UTT2LABEL[uttid] = label_dict[int(filename)]
        
        return TEXT, WAV_SCP, UTT2SPK, SEGMENTS, UTT2LABEL

    def save_data(self):

        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

        TEXT, WAV_SCP, UTT2SPK, SEGMENTS, UTT2LABEL = self.gather_infomration()

        text_path = os.path.join(self.output_path, 'text')
        with open(text_path, 'w') as file:
            for key, value in TEXT.items():
                file.write(f"{key} {value}\n")
        
        wavscp_path = os.path.join(self.output_path, 'wav.scp')
        with open(wavscp_path, 'w') as file:
            for key, value in WAV_SCP.items():
                file.write(f"{key} {value}\n")
        
        utt2spk_path = os.path.join(self.output_path, 'utt2spk')
        with open(utt2spk_path, 'w') as file:
            for key, value in UTT2SPK.items():
                file.write(f"{key} {value}\n")
        
        segments_path = os.path.join(self.output_path, 'segments')
        with open(segments_path, 'w') as file:
            for key, (wavid, start_time, stop_time) in SEGMENTS.items():
                file.write(f"{key} {wavid} {start_time} {stop_time}\n")
        
        utt2label_path = os.path.join(self.output_path, 'utt2label')
        with open(utt2label_path, 'w') as file:
            for key, value in UTT2LABEL.items():
                file.write(f"{key} {value}\n")
        
        with open(self.id2label_path, 'w') as file:
            for pair in self.pairs_list:
                file.write(' '.join(map(str, pair)) + '\n') 

def parse_arguments():
    parser = argparse.ArgumentParser(description='Description of your data prep')
    parser.add_argument('--config', type=str, default='data_config.yaml', help='Path to the YAML config file')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_type = config['train']['data_type']
    data_folder_path = config['train']['data_folder_path']
    label_csv_path = config['train']['label_csv_path']
    output_path = config['train']['output_path']
    dataset_name = config['train']['dataset_name']

    train_dataset_class = data_prep(data_type, data_folder_path, label_csv_path, output_path)
    train_dataset_class.save_data()


    data_type = config['dev']['data_type']
    data_folder_path = config['dev']['data_folder_path']
    label_csv_path = config['dev']['label_csv_path']
    output_path = config['dev']['output_path']
    dataset_name = config['dev']['dataset_name']

    dev_dataset_class = data_prep(data_type, data_folder_path, label_csv_path, output_path)
    dev_dataset_class.save_data()

    # data_type = config['test']['data_type']
    # data_folder_path = config['test']['data_folder_path']
    # label_csv_path = config['test']['label_csv_path']
    # output_path = config['test']['output_path']
    # dataset_name = config['test']['dataset_name']

    # test_dataset_class = data_prep(data_type, data_folder_path, label_csv_path, output_path)
    # test_dataset_class.save_data()


if __name__ == "__main__":
    main()