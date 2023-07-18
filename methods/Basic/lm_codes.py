import numpy as np
from .lm_labels import lm_labels

def lm_codes(*args):
    """
    将地标标签或地标字符标记转换为对应的地标索引或地标字符标记。

    参数：
    *args：str 或 array_like
        地标标签或地标字符标记的输入，可以传入多个字符串参数或数组参数。

    返回：
    tuple
        转换后的地标索引或地标字符标记的元组。
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

    # 将字符串参数转换为数组参数
    array_args = [np.array([code_label_mapping[arg.upper()]]) if isinstance(arg, str) else arg for arg in args]

    # 调用 lm_labels 函数，并将结果存储在列表中
    outargs = []
    for arg in array_args:
        outargs.extend(lm_labels(arg))

    # 将列表转换为元组并返回
    return outargs

# # 示例：测试 lm_codes 函数
# output = lm_codes('PLUS_G','MINUS_B')
# print(output)
# output = lm_codes('MINUS_G', 'PLUS_G', 'MINUS_B', 'PLUS_B', 'MINUS_S', 'PLUS_S', 'MINUS_F', 'PLUS_F', 'MINUS_V', 'PLUS_V')
# print(output)