import librosa

def wavread_freq(file_path,  Fs):
    """
    Read a .wav audio file and return the audio signal and the sampling rate.

    Parameters:
        file_path (str): File path or file name.

    Returns:
        sig (numpy.array): Audio signal data.
        sr (int): Sampling rate.
    """
    try:
        # Use the librosa.load function to read the .wav audio file
        # sr=None means to keep the original sampling rate
        sig, sr = librosa.load(file_path, sr= Fs)

        # Return the read audio signal and sampling rate
        return sig, sr

    except FileNotFoundError:
        raise FileNotFoundError(f'File not found: {file_path}')