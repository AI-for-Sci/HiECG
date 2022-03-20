import os
import tensorflow as tf
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import wfdb


def scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise


def shift(sig, interval=20):
    for col in range(sig.shape[1]):
        offset = np.random.choice(range(-interval, interval))
        sig[:, col] += offset / 1000 
    return sig


def transform(sig, train=False):
    if train:
        if np.random.randn() > 0.5: sig = scaling(sig)
        if np.random.randn() > 0.5: sig = shift(sig)
    return sig


class ECGDataset(Dataset):
    def __init__(self, phase, data_dir, label_csv, folds, leads):
        super(ECGDataset, self).__init__()
        self.phase = phase
        df = pd.read_csv(label_csv)
        df = df[df['fold'].isin(folds)]
        self.data_dir = data_dir
        self.labels = df
        self.leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        if leads == 'all':
            self.use_leads = np.where(np.in1d(self.leads, self.leads))[0]
        else:
            self.use_leads = np.where(np.in1d(self.leads, leads))[0]
        self.nleads = len(self.use_leads)
        self.classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
        self.n_classes = len(self.classes)
        self.data_dict = {}
        self.label_dict = {}

    def __getitem__(self, index: int):
        row = self.labels.iloc[index]
        patient_id = row['patient_id']
        ecg_data, _ = wfdb.rdsamp(os.path.join(self.data_dir, patient_id))
        ecg_data = transform(ecg_data, self.phase == 'train')
        nsteps, _ = ecg_data.shape
        ecg_data = ecg_data[-15000:, self.use_leads]
        result = np.zeros((15000, self.nleads)) # 30 s, 500 Hz
        result[-nsteps:, :] = ecg_data
        if self.label_dict.get(patient_id):
            labels = self.label_dict.get(patient_id)
        else:
            labels = row[self.classes].to_numpy(dtype=np.float32)
            self.label_dict[patient_id] = labels
        return torch.from_numpy(result.transpose()).float(), torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.labels)

def split_data(seed=42):
    folds = range(1, 11)
    folds = np.random.RandomState(seed).permutation(folds)
    return folds[:8], folds[8:9], folds[9:]

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

if __name__ == "__main__":
    # Download data from https://www.dropbox.com/s/unicm8ulxt24vh8/CPSC.zip?dl=0
    label_csv = "../../../data/CPSC/labels.csv"
    data_dir = "../../../data/CPSC/"

    # Labels
    label_df = pd.read_csv(label_csv)

    train_folds, val_folds, test_folds = split_data()

    max_step = 0
    classes = ['SNR', 'AF', 'IAVB', 'LBBB', 'RBBB', 'PAC', 'PVC', 'STD', 'STE']
    leads = 'all'
    all_leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    if leads == 'all':
        use_leads = np.where(np.in1d(all_leads, all_leads))[0]
    else:
        use_leads = np.where(np.in1d(all_leads, leads))[0]

    num_leads = 12
    print("use_leads: ", use_leads)

    train_examples, val_examples, test_examples = [], [], []
    for index, row in label_df.iterrows():
        patient_id = row['patient_id']
        fold = row['fold']
        ecg_data, _ = wfdb.rdsamp(os.path.join(data_dir, patient_id))
        ecg_data = transform(ecg_data, True)
        nsteps, _ = ecg_data.shape
        y_data = row[classes].values

        x_data = ecg_data[-15000:, use_leads]
        # x_data = np.zeros((15000, num_leads), dtype=np.float)  # 30 s, 500 Hz
        # x_data[-nsteps:, :] = ecg_data

        def array_serialize(array):
            array = tf.io.serialize_tensor(array)
            return array


        x_data = (x_data - x_data.mean()) / np.sqrt(x_data.var() + 1e-7)
        x_data = array_serialize(x_data)
        features = tf.train.Features(
            feature={
                "input_values": _bytes_feature(x_data),
                "labels": tf.train.Feature(
                    int64_list=tf.train.Int64List(value=list(y_data)))
            }
        )
        tf_example = tf.train.Example(features=features)
        serialized = tf_example.SerializeToString()

        if fold in train_folds:
            train_examples.append(serialized)
        elif fold in val_folds:
            val_examples.append(serialized)
        else:
            test_examples.append(serialized)

        if nsteps > max_step:
            max_step = nsteps

    if len(train_examples) > 0:
        dest_file = '../data/train_examples.tfrecord'
        with tf.io.TFRecordWriter(dest_file) as writer:
            for example in train_examples:
                writer.write(example)

    if len(val_examples) > 0:
        dest_file = '../data/val_examples.tfrecord'
        with tf.io.TFRecordWriter(dest_file) as writer:
            for example in val_examples:
                writer.write(example)

    if len(test_examples) > 0:
        dest_file = '../data/test_examples.tfrecord'
        with tf.io.TFRecordWriter(dest_file) as writer:
            for example in test_examples:
                writer.write(example)

    print("max_step: ", max_step)
    # df = df[df['fold'].isin(folds)]