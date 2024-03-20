import os

def get_ordered_landmarks(landmark_dict):
    all_landmarks = []
    for landmark, times in landmark_dict.items():
        all_landmarks.extend((time, landmark) for time in times)
    all_landmarks.sort()
    return all_landmarks

def process_result_dict_withtime(result_dict, output_path):
    output_path = os.path.join(output_path, 'landmark_output_time.txt')
    with open(output_path, 'w') as outfile:
        for file_path, landmark_dict in result_dict.items():
            ordered_landmarks = get_ordered_landmarks(landmark_dict)
            landmarks_str = ' '.join(f'{time}:{landmark}' for time, landmark in ordered_landmarks)
            outfile.write(f'{file_path}: {landmarks_str}\n')

def process_result_dict_notime(result_dict, output_path):
    output_path = os.path.join(output_path, 'landmark_output_notime.txt')
    with open(output_path, 'w') as outfile:
        for file_path, landmark_dict in result_dict.items():
            ordered_landmarks = get_ordered_landmarks(landmark_dict)
            landmarks_str = ' '.join(f'{landmark}' for _, landmark in ordered_landmarks)
            outfile.write(f'{file_path}: {landmarks_str}\n')