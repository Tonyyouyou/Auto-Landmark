import librosa
import numpy as np
from .wavread_freq import wavread_freq

def waveform_freq(SIGNAL, Fs):
    """
    Get the audio signal from a variable or load it from a file.

    Parameters:
        SIGNAL (numeric array or str): Can be a numeric array or file name (including path).
            If the filename is given with a wildcard (*), a dialog box will open to select or
            browse the file.
        Fs (int): Sampling rate [Hz] of the audio signal to be returned.

    Returns:
        arr (numeric array): Audio signal data. If SIGNAL is a filename FN, then [arr, fs0] = wavread_freq(FN, Fs).
            If SIGNAL is a numeric array, then arr = SIGNAL.
        fs0 (int): Original sampling rate of the audio signal. If SIGNAL is a file, then fs0 is the sampling rate
            from the file. If SIGNAL is a numeric array, then fs0 = Fs.
    """
    if isinstance(SIGNAL, str):  # If SIGNAL is a string
        if '*' in SIGNAL:  # If SIGNAL contains a wildcard *
            fname = librosa.core.time_freq.uigetfile(SIGNAL, 'Select audio file to read')  # Open dialog to choose audio file
        else:
            fname = SIGNAL  # Otherwise, use the given filename

        try:
            arr, fs0 = wavread_freq(fname,  Fs)  # Read audio signal and sampling rate from the file
        except FileNotFoundError:
            if fname == 0:
                raise ValueError('No file specified.')
            else:
                raise  # Raise other exceptions
    elif isinstance(SIGNAL, (int, float)):  # If SIGNAL is a numeric type
        arr = SIGNAL  # Use SIGNAL as audio signal data directly
        fs0 = Fs  # Use the given sampling rate Fs
    else:
        raise TypeError(f'waveform_freq: Unable to process SIGNAL: Must be numeric or string, not {type(SIGNAL)} type.')

    return arr, fs0  # Return audio signal data and sampling rate

# # Example 1: Read a .wav audio file and specify the sampling rate
# file_path = 'D:/python object/spx1.wav'
# desired_sampling_rate = 16000
# Audio_Signal, Sampling_Rate = waveform_freq(file_path, desired_sampling_rate)
# # Print the audio signal and sampling rate
# print('Audio Signal:', Audio_Signal)
# print('Sampling Rate:', Sampling_Rate)

# # Example 2: Get the audio signal from a variable
# # This is just an example, your actual audio data should be used here
# signal_data = np.array([-0.00152588, -0.0020752, -0.00183105, -0.01116943, -0.01113892, -0.01184082])
# # Assume your audio data has a sampling rate of 44100Hz
# desired_sampling_rate = 44100
# # Use signal_data as the audio signal data and desired_sampling_rate as the sampling rate
# Audio_Signal, Sampling_Rate = signal_data, desired_sampling_rate
# # Print the audio signal and sampling rate
# print('Audio Signal:', Audio_Signal)
# print('Sampling Rate:', Sampling_Rate)
