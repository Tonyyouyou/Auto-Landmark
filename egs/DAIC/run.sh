#!/bin/bash
DATA_YAML_FILE='/home/561/xz4320/Auto-Landmark/egs/DAIC/config/data_config.yaml'
TRAIN_YAML_FILE='/home/561/xz4320/Auto-Landmark/egs/DAIC/config/train_basic.yaml'
DEV_YAML_FILE='/home/561/xz4320/Auto-Landmark/egs/DAIC/config/dev_basic.yaml'

./landmark.sh \
    --STAGE 3 \
    --DATA_YAML_FILE $DATA_YAML_FILE \
    --TRAIN_YAML_FILE $TRAIN_YAML_FILE \
    --METHOD 'basic'



