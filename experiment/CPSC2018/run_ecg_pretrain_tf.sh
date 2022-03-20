#!/bin/bash

python ecg_pretrain_tf.py \
    --config_name ./models/config.json \
    --output_dir ./output/CPSC2018 \
    --train_file ./data/ \
    --per_device_train_batch_size 8 \
    --overwrite_output_dir