#!/bin/bash

python ecg_classification_tf.py \
    --config_name ./output/CPSC2018/config.json \
    --model_name_or_path ./output/CPSC2018 \
    --output_dir ./output/CPSC2018_Classification \
    --train_file ./data/train \
    --validation_file ./data/val \
    --per_device_train_batch_size 8 \
    --num_train_epochs 10 \
    --overwrite_output_dir