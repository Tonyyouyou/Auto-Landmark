import numpy as np
from lm_labels import lm_labels

def lm_codes(*args):
    """
    Converts a landmark label or landmark character marker to a corresponding landmark index or landmark character marker.

    Parameters：
    *args：str or array_like
        The input to the landmark label or landmark character tag can be passed multiple string arguments or array arguments。

    Return：
    tuple
        The converted tuple of the landmark index or landmark character tag。
    """
    # 定义地标标签与索引的映射关系
    code_label_mapping = {
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

    # 输出plus_p有问题
    # array_args = [np.array([code_label_mapping[arg.upper()]]) if isinstance(arg, str) else arg for arg in args]

    # outargs = []
    # for arg in array_args:
    #     outargs.extend(lm_labels(arg))

    array_args = {value: index + 1 for index, value in enumerate(list(code_label_mapping.keys()))}
    outargs = []
    for arg in args:
        if arg in array_args:
            outargs.append(array_args[arg])
    return outargs

# # 示例：测试 lm_codes 函数
# output = lm_codes('PLUS_P','MINUS_P')
# print(output)
# output = lm_codes('MINUS_G', 'PLUS_G', 'MINUS_B', 'PLUS_B', 'MINUS_S', 'PLUS_S', 'MINUS_F', 'PLUS_F', 'MINUS_V', 'PLUS_V')
# print(output)