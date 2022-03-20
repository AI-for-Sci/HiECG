import os
import tensorflow as tf
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import wfdb


def load_dataset(data_path) -> tf.data.Dataset:
    """
    parse_function
    """

    def _parse_record(record):
        name_to_features = {
            'input_values': tf.io.FixedLenFeature([], tf.string),
            'labels': tf.io.VarLenFeature(tf.int64),
        }

        example = tf.io.parse_single_example(record, name_to_features)
        for name in ["labels"]:
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = tf.sparse.to_dense(t, default_value=0)

        for name in ["input_values"]:
            t = example[name]
            t = tf.io.parse_tensor(t, out_type=tf.float64)
            if t.dtype == tf.float64:
                t = tf.cast(t, tf.float32)
            example[name] = t

        print("input_values: ", example["input_values"])
        return example

    def _parse_mask(example):
        print("input_values: ", example["input_values"])
        print("labels: ", example["labels"])
        return example["input_values"], example["labels"]

    record_names = []
    if os.path.isdir(data_path):
        files = os.listdir(data_path)
        for file_name in files:
            if file_name.endswith(".tfrecord"):
                record_names.append(os.path.join(data_path, file_name))
    else:
        if data_path.endswith(".tfrecord"):
            record_names.append(data_path)

    print("record_names: ", record_names)

    dataset = tf.data.TFRecordDataset(record_names)
    dataset = dataset.map(map_func=_parse_record)
    # dataset = dataset.map(map_func=_parse_mask)

    return dataset


if __name__ == "__main__":
    data_dir = "../data"

    ds = load_dataset(data_dir)
    for record in ds.take(10):
        print(record)


