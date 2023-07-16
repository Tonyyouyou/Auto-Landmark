import soundfile as sf


def waveform_freq(SIGNAL, Fs):
    if SIGNAL == '?':
        print("[arr, fs0] = waveform_freq(SIGNAL_ARR|SIGNAL_FNAME, Fs)")
        return

    if isinstance(SIGNAL, str):
        if "*" in SIGNAL:
            fname = sf.file_dialog('Select audio file to read', SIGNAL)
        else:
            fname = SIGNAL
        try:
            arr, fs0 = sf.read_freq(fname, Fs)
        except ValueError:
            if fname == "":
                raise ValueError("No file specified.")
            else:
                raise
    elif isinstance(SIGNAL, (list, np.ndarray)):
        arr = SIGNAL
        fs0 = Fs
    else:
        raise ValueError(f"Unable to process SIGNAL: Must be numeric or string, not {type(SIGNAL)}.")

    return arr, fs0
