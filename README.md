
1、安装- Installation

# CUDA
-- https://developer.nvidia.com/CUDA-toolkit-archive
sudo sh cuda_11.2.1_460.32.03_linux.run
-- https://developer.nvidia.com/rdp/cudnn-archive
sudo dpkg -i libcudnn8_8.1.0.77-1+cuda11.2_amd64.deb


conda create -n transformers python=3.8

-- Pytorch
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

-- JAX
--pip install --upgrade pip --user
--pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
--pip install --upgrade flax

-- transformers
pip install transformers==4.12.5

-- Tensorflow
conda install tensorflow=2.4
pip install tensorflow-gpu==2.4.1

-- Others
conda install pandas
conda install sentencepiece
pip install sklearn
pip install seqeval
pip install typeguard
pip install datasets
pip install gin_config
pip install soundfile





