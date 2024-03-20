#!/bin/bash

# 初始化默认的方法为空字符串
METHOD=""

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --STAGE) STAGE="$2"; shift; shift ;;
        --DATA_YAML_FILE) DATA_YAML_FILE="$2"; shift; shift ;;
        --TRAIN_YAML_FILE) TRAIN_YAML_FILE="$2"; shift; shift ;;
        --METHOD) METHOD="$2"; shift; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
done

# 检查是否所有必要的参数都被设置
if [ -z "$STAGE" ] || [ -z "$DATA_YAML_FILE" ] && [ "$STAGE" -ne 3 ] || [ -z "$TRAIN_YAML_FILE" ] && [ "$STAGE" -eq 3 ]; then
    echo "Missing parameters."
    exit 1
fi

# 脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 计算Auto-Landmark根目录的路径
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# 根据STAGE执行不同的操作
case $STAGE in
    1)
        KALDI_PYTHON_SCRIPT="${PWD}/utils/kaldi_prep.py"
        echo "Executing command: python3 \"$KALDI_PYTHON_SCRIPT\" --config \"$DATA_YAML_FILE\""
        python3 "$KALDI_PYTHON_SCRIPT" --config "$DATA_YAML_FILE"
        ;;
    2)
        SEGMENT_PYTHON_SCRIPT="$ROOT_DIR/data/Audio_segment.py"
        echo "Executing command: python3 \"$SEGMENT_PYTHON_SCRIPT\" --config \"$TRAIN_YAML_FILE\""
        python3 "$SEGMENT_PYTHON_SCRIPT" --config "$TRAIN_YAML_FILE"
        ;;
    3)
        echo "Selected method for stage 3: $METHOD"
        if [ "$METHOD" = "basic" ]; then
            PYTHON_METHOD_FILE="$ROOT_DIR/methods/Basic/Basic_main.py"
            python3 $PYTHON_METHOD_FILE --config "$TRAIN_YAML_FILE"
        elif [ "$METHOD" = "advance" ]; then
            PYTHON_METHOD_FILE="$ROOT_DIR/methods/Advance/Advance_main.py"
        else
            echo "Unknown method: $METHOD"
            exit 2
        fi
        echo "Executing command: python3 \"$PYTHON_METHOD_FILE\" \"$TRAIN_YAML_FILE\""
        python3 "$PYTHON_METHOD_FILE" "$TRAIN_YAML_FILE"
        ;;
    *)
        echo "Unknown stage: $STAGE"
        exit 3
        ;;
esac
