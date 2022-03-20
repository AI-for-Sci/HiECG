#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=masked-lm
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

# TODO Do multi-GPU and TPU tests and make sure the dataset length works as expected
# TODO Duplicate all changes over to the CLM script

import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

import datasets
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import sys

sys.path.append('../../')
import hiai.transformers
from hiai.transformers import (
    CONFIG_MAPPING,
    CONFIG_NAME,
    MODEL_FOR_MASKED_LM_MAPPING,
    TF2_WEIGHTS_NAME,
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TFAutoModelForMaskedLM,
    TFTrainingArguments,
    create_optimizer,
    set_seed,
    TFSignal2VecModel,
    Signal2VecConfig
)
from hiai.transformers.utils.versions import require_version

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# region Command-line arguments
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
                    "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": "Override some existing default config settings when a model is trained from scratch. Example: "
                    "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # dataset_name: Optional[str] = field(
    #     default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    # )
    # dataset_config_name: Optional[str] = field(
    #     default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    # )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=10000,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )

    def __post_init__(self):
        if self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        # else:
        #     if self.train_file is not None:
        #         extension = self.train_file.split(".")[-1]
        #         assert extension in ["tfrecord"], "`train_file` should be a tfrecord file."
        #     if self.validation_file is not None:
        #         extension = self.validation_file.split(".")[-1]
        #         assert extension in ["tfrecord"], "`validation_file` should be a tfrecord file."


# endregion


# region Helper classes
class SavePretrainedCallback(tf.keras.callbacks.Callback):
    # Hugging Face models have a save_pretrained() method that saves both the weights and the necessary
    # metadata to allow them to be loaded as a pretrained model in future. This is a simple Keras callback
    # that saves the model with this method after each epoch.
    def __init__(self, output_dir, **kwargs):
        super().__init__()
        self.output_dir = output_dir

    def on_epoch_end(self, epoch, logs=None):
        self.model.save_pretrained(self.output_dir)


# endregion

# region Data generator

def data_dict_process(input_values, labels):
    example = {}
    example["input_values"] = input_values
    example["labels"] = labels
    return example


def load_dataset(data_path, first_token_id=105, last_token_id=111, sequence_length=1024) -> tf.data.Dataset:
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
    dataset = dataset.map(map_func=_parse_mask)

    return dataset


# endregion

def simcse_loss(z1, z2, temperature=0.1):
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)

    step_batch_size = tf.shape(z1)[0]

    labels = tf.one_hot(tf.range(step_batch_size), step_batch_size)

    logits_aa = tf.matmul(z1, z2, transpose_b=True) / temperature

    loss_a = tf.nn.softmax_cross_entropy_with_logits(
        labels, tf.concat([logits_aa], 1))
    loss = loss_a  # tf.reduce_mean(loss_a)
    return loss


class ProjectionHeader(tf.keras.models.Model):
    """
    """

    def __init__(self, pooling='cls', drop_rate=0.1, hidden_size=128, hidden_norm=True, name='cls', **kwargs):
        super(ProjectionHeader, self).__init__(name=name)
        self.hidden_norm = hidden_norm
        self.hidden_size = hidden_size
        self.pooling = pooling

        self.avg_pool = tf.keras.layers.GlobalAveragePooling1D()
        self.fc1 = tf.keras.layers.Dense(units=hidden_size, activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.dropout = tf.keras.layers.Dropout(rate=drop_rate)
        self.fc2 = tf.keras.layers.Dense(units=hidden_size, activation='relu')

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                'name': self.name,
                'hidden_norm': self.hidden_norm,
                'hidden_size': self.hidden_size,
                'pooling': self.pooling
            }
        )
        return config

    def call(self, hidden_states):
        # Output
        if self.pooling == 'first-last-avg':
            outputs = [
                self.avg_pool(hidden_states[0]),
                self.avg_pool(hidden_states[-1])
            ]
            output = tf.keras.layers.Average()(outputs)
        elif self.pooling == 'last-avg':
            output = self.avg_pool(hidden_states[-1])
        elif self.pooling == 'cls':
            output = tf.keras.layers.Lambda(lambda x: x[:, 0])(hidden_states[-1])
        else:
            output = tf.keras.layers.Lambda(lambda x: x[:, 0])(hidden_states[-1])

        x = output
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def train_step(training_args, model, project_header, optimizer, input_values):
    with tf.GradientTape() as tape:
        z1 = model({"input_values": input_values}, output_hidden_states=True, training=True)
        z1 = project_header(z1['hidden_states'])
        z2 = model({"input_values": input_values}, output_hidden_states=True, training=True)
        z2 = project_header(z2['hidden_states'])

        loss = simcse_loss(z1, z2)
        loss = tf.nn.compute_average_loss(loss, global_batch_size=training_args.per_device_train_batch_size)

    vars = [model.trainable_variables,
            project_header.trainable_variables,
            ]
    grads = tape.gradient(loss, vars)

    for grad, var in zip(grads, vars):
        optimizer.apply_gradients(zip(grad, var))
    return loss


# @tf.function
def distributed_train_step(training_args, model, project_header, optimizer, input_values):
    """
    """
    # strategy = tf.distribute.MirroredStrategy()
    # strategy.experimental_run()
    per_replica_losses = training_args.strategy.run(train_step,
                                                    args=(
                                                        training_args,
                                                        model,
                                                        project_header,
                                                        optimizer,
                                                        input_values))
    distributed_loss = training_args.strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
    return distributed_loss


def main():
    # Set memory growth
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # region Argument Parsing
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.output_dir is not None:
        training_args.output_dir = Path(training_args.output_dir)
        os.makedirs(training_args.output_dir, exist_ok=True)

    if isinstance(training_args.strategy, tf.distribute.TPUStrategy) and not data_args.pad_to_max_length:
        logger.warning("We are training on TPU - forcing pad_to_max_length")
        data_args.pad_to_max_length = True
    # endregion

    # region Checkpoints
    # Detecting last checkpoint.
    checkpoint = None
    if len(os.listdir(training_args.output_dir)) > 0 and not training_args.overwrite_output_dir:
        config_path = training_args.output_dir / CONFIG_NAME
        weights_path = training_args.output_dir / TF2_WEIGHTS_NAME
        if config_path.is_file() and weights_path.is_file():
            checkpoint = training_args.output_dir
            logger.warning(
                f"Checkpoint detected, resuming training from checkpoint in {training_args.output_dir}. To avoid this"
                " behavior, change the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
        else:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to continue regardless."
            )
    # endregion

    # region Setup logging
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO)
    datasets.utils.logging.set_verbosity_warning()
    hiai.transformers.utils.logging.set_verbosity_info()
    # endregion

    # If passed along, set the training seed now.
    if training_args.seed is not None:
        set_seed(training_args.seed)

    # region Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if checkpoint is not None:
        config = AutoConfig.from_pretrained(checkpoint)
    elif model_args.config_name:
        # config = AutoConfig.from_pretrained(model_args.config_name)
        config = Signal2VecConfig.from_json_file(model_args.config_name)
    # elif model_args.model_name_or_path:
    #     config = AutoConfig.from_pretrained(model_args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # endregion

    if data_args.train_file is not None:
        train_dataset = load_dataset(data_args.train_file)
        train_dataset = train_dataset.shuffle(buffer_size=10000)
        train_dataset = train_dataset.padded_batch(batch_size=training_args.per_device_train_batch_size,
                                                   padded_shapes=([None, None], [None]),
                                                   padding_values=(0.0, 0), drop_remainder=True)
        # train_dataset = train_dataset.map(data_dict_process, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

        # strategy.scope创建并分发数据集
        train_dataset = training_args.strategy.experimental_distribute_dataset(train_dataset)

    num_replicas = training_args.strategy.num_replicas_in_sync

    # endregion

    with training_args.strategy.scope():
        # region Prepare model
        if checkpoint is not None:
            model = TFAutoModelForMaskedLM.from_pretrained(checkpoint, config=config)
        # elif model_args.model_name_or_path:
        #     model = TFAutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, config=config)
        else:
            logger.info("Training new model from scratch")
            model = TFSignal2VecModel.from_pretrained("./models/", config=config)

        # endregion

        # region Optimizer and loss
        batches_per_epoch = data_args.max_train_samples // (num_replicas * training_args.per_device_train_batch_size)
        optimizer, lr_schedule = create_optimizer(
            init_lr=training_args.learning_rate,
            num_train_steps=int(training_args.num_train_epochs * batches_per_epoch),
            num_warmup_steps=training_args.warmup_steps,
            adam_beta1=training_args.adam_beta1,
            adam_beta2=training_args.adam_beta2,
            adam_epsilon=training_args.adam_epsilon,
            weight_decay_rate=training_args.weight_decay,
        )

        project_header = ProjectionHeader(pooling='last-avg', hidden_size=256)

        # model.compile(optimizer=optimizer, loss={"loss": dummy_loss})
        # endregion

        # region Training and validation
        logger.info("***** Running training *****")
        logger.info(f"  Num Epochs = {training_args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {training_args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size = {training_args.per_device_train_batch_size * num_replicas}")

        train_loss = tf.keras.metrics.Mean(name='train_loss')

        for epoch in range(int(training_args.num_train_epochs)):
            train_loss.reset_states()
            start = time.process_time()
            total_loss = 0.0
            for step, (input_values, labels) in enumerate(train_dataset):
                distributed_loss = distributed_train_step(training_args, model, project_header, optimizer, input_values)
                # total_loss += distributed_loss

                train_loss(distributed_loss)

                if step % 100 == 0 and step > 0:
                    template = 'Epoch {}, Step {}, Loss: {:.6f}.'
                    print(template.format(epoch + 1,
                                          str(step),
                                          train_loss.result()))

            end = time.process_time()

            template = 'Epoch {}, Loss: {}, Times: {}.'
            print(template.format(epoch + 1,
                                  train_loss.result(),
                                  str(end - start)), )

            if training_args.output_dir is not None:
                model.save_pretrained(training_args.output_dir)
                project_header.save_weights(os.path.join(training_args.output_dir, 'project_header.h5'),
                                            save_format='h5')

        if training_args.output_dir is not None:
            model.save_pretrained(training_args.output_dir)
            project_header.save_weights(os.path.join(training_args.output_dir, 'project_header.h5'),
                                        save_format='h5')


if __name__ == "__main__":
    main()
