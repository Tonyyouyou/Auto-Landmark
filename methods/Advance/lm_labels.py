import numpy as np

def lm_labels(input_array):
    dict = {
        "-g": 1,
        "+g": 2,
        "-b": 3,
        "+b": 4,
        "-s": 5,
        "+s": 6,
        "-f": 7,
        "+f": 8,
        "-v": 9,
        "+v": 10
    }
    result = []
    for i in input_array:
        if i in dict:
            # i is in the dictionary's keys
            result.append(dict[i])  # Append the corresponding value to the result list
        elif i in dict.values():
            # i is in the dictionary's values
            result.append(list(dict.keys())[list(dict.values()).index(i)])
        else:
            if i != "??":
                result.append("??")
            else:
                result.append(np.nan)

    return result

# input_array1 = [2, 1, np.nan, 4, np.inf]
# print(lm_labels(input_array1))

# input_array2 = ['+g', '-g', '??', '+b', '??']
# print(lm_labels(input_array2))