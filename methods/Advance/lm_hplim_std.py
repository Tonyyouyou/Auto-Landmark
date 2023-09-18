def lm_hplim_std(AGE='ADULT'):
    """
    Calculate the standard high-pass filter cutoff frequency for landmark speech signal processing.

    Parameters:
        AGE (str, optional): Age group ("ADULT" or "CHILD"). Defaults to "ADULT" for adults.

    Returns:
        hplim (int): Cutoff frequency of the high-pass filter in hertz (Hz).

    """

    # If the function is called with one input parameter and the parameter value is '?',
    # print information about the function syntax.
    if AGE == '?':
        print('hplim = lm_hplim_std(<"adult"|AGE|"child">)')
        return

    # Convert the input string to a character array
    if isinstance(AGE, str):
        AGE = AGE.strip().upper()

    # If the AGE parameter is not provided, default it to "ADULT"
    if not AGE:
        AGE = 'ADULT'

    if AGE == 'ADULT':  # For adult cases:
        hplim = 75  # [Hz]
    else:  # For infant or child cases:
        hplim = 150  # [Hz]

    return hplim