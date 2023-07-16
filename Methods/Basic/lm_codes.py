from lm_labels import lm_labels

def lm_codes(*args):
    outargs = [None] * (len(args) + 1)
    outargs[0] = float('nan')

    mapping = {
        'MINUS_G': '-g',
        'PLUS_G': '+g',
        'MINUS_B': '-b',
        'PLUS_B': '+b',
        'MINUS_S': '-s',
        'PLUS_S': '+s',
        'MINUS_F': '-f',
        'PLUS_F': '+f',
        'MINUS_V': '-v',
        'PLUS_V': '+v',
        'MINUS_P': '-p',
        'PLUS_P': '+p',
        'MINUS_J': '-j',
        'PLUS_J': '+j',
        'VOWEL': ' V',
        'FRICATION': ' F',
        'START_SEG': '+T',
        'END_SEG': '-T'
    }

    for arg in args:
        arg_upper = arg.upper()
        if arg_upper in mapping:
            outargs[1 + args.index(arg)] = lm_labels(mapping[arg_upper])

    return outargs[1:]
