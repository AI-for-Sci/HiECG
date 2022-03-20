#!/bin/bash

python analysis_ecg_pretrain_tf.py \
    --config_name ./output/CPSC2018/config.json \
    --output_dir ./output/CPSC2018 \
    --train_file ./data/ \
    --per_device_train_batch_size 4 \
    --overwrite_output_dir