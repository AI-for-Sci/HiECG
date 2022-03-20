# flake8: noqa
# There's no way to ignore "F401 '...' imported but unused" warnings in this
# module, but to preserve other warnings. So, don't check this module at all.

# Copyright 2021 The HuggingFace Team. All rights reserved.
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
from typing import TYPE_CHECKING

from ...file_utils import _LazyModule, is_flax_available, is_tf_available, is_torch_available


_import_structure = {
    "configuration_signal2vec": ["SIGNAL_2_VEC_PRETRAINED_CONFIG_ARCHIVE_MAP", "Signal2VecConfig"],
    "feature_extraction_signal2vec": ["Signal2VecFeatureExtractor"],
    "processing_signal2vec": ["Signal2VecProcessor"],
    "tokenization_signal2vec": ["Signal2VecCTCTokenizer", "Signal2VecTokenizer"],
}

if is_torch_available():
    _import_structure["modeling_signal2vec"] = [
        "SIGNAL_2_VEC_PRETRAINED_MODEL_ARCHIVE_LIST",
        "Signal2VecForCTC",
        "Signal2VecForMaskedLM",
        "Signal2VecForPreTraining",
        "Signal2VecForSequenceClassification",
        "Signal2VecModel",
        "Signal2VecPreTrainedModel",
    ]

if is_tf_available():
    _import_structure["modeling_tf_signal2vec"] = [
        "TF_SIGNAL_2_VEC_PRETRAINED_MODEL_ARCHIVE_LIST",
        "TFSignal2VecForCTC",
        "TFSignal2VecModel",
        "TFSignal2VecPreTrainedModel",
    ]

if is_flax_available():
    _import_structure["modeling_flax_signal2vec"] = [
        "FlaxSignal2VecForCTC",
        "FlaxSignal2VecForPreTraining",
        "FlaxSignal2VecModel",
        "FlaxSignal2VecPreTrainedModel",
    ]


if TYPE_CHECKING:
    from .configuration_signal2vec import SIGNAL_2_VEC_PRETRAINED_CONFIG_ARCHIVE_MAP, Signal2VecConfig
    from .feature_extraction_signal2vec import Signal2VecFeatureExtractor
    from .processing_signal2vec import Signal2VecProcessor
    from .tokenization_signal2vec import Signal2VecCTCTokenizer, Signal2VecTokenizer

    if is_torch_available():
        from .modeling_signal2vec import (
            SIGNAL_2_VEC_PRETRAINED_MODEL_ARCHIVE_LIST,
            Signal2VecForCTC,
            Signal2VecForMaskedLM,
            Signal2VecForPreTraining,
            Signal2VecForSequenceClassification,
            Signal2VecModel,
            Signal2VecPreTrainedModel,
        )

    if is_tf_available():
        from .modeling_tf_signal2vec import (
            TF_WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST,
            TFSignal2VecForCTC,
            TFSignal2VecModel,
            TFSignal2VecPreTrainedModel,
        )

    if is_flax_available():
        from .modeling_flax_signal2vec import (
            FlaxSignal2VecForCTC,
            FlaxSignal2VecForPreTraining,
            FlaxSignal2VecModel,
            FlaxSignal2VecPreTrainedModel,
        )


else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
