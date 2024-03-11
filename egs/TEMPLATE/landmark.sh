#!/bin/bash

STAGE=$1
DATA_FILE=$2

# 脚本所在目录的绝对路径
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# 计算Auto-Landmark根目录的路径
ROOT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"


if [ -z "$STAGE" ] || [ -z "$PYTHON_SCRIPT" ] || [ -z "$DATA_FILE" ]; then
    echo "Usage: $0 <stage> <python_script> <yaml_file>"
    exit 1
fi

case $STAGE in
    1)
        KAKDI_PYTHON_SCRIPT="$ROOT_DIR/egs/TEMPLATE/utils/kaldi_prep.py"
        python $PYTHON_SCRIPT_PATH $DATA_FILE
        ;;
    2)
    
        SEGMENT_PYTHON_SCRIPT="$ROOT_DIR/data/Audio_segment.py"
        python $SEGMENT_PYTHON_SCRIPT $DATA_FILE
        ;;
    3)
      
        echo "Selected method for stage 3: $METHOD"
        if [ "$METHOD" = "basic" ]; then
         
            python basic_method.py $YAML_FILE
        elif [ "$METHOD" = "advance" ]; then
           
            python advance_method.py $YAML_FILE
        else
            echo "Unknown method: $METHOD"
            exit 2
        fi
        ;;
    *)
        echo "Unknown stage: $STAGE"
        exit 3
        ;;
esac