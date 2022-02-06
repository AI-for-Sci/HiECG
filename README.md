
## Model Architecture

<img src="https://imgur.com/BIvuVUc.png" width="300">

> Deep neural network architecture for cardiac arrhythimas diagnosis.

## Requirement

### Dataset

The 12-lead ECG dataset used in this study is the CPSC2018 training dataset which is released by the 1st China Physiological Signal Challenge (CPSC) 2018 during the 7th International Conference on Biomedical Engineering and Biotechnology. Details of the CPSC2018 dataset can be found [here](https://bit.ly/3gus3D0). To access the processed data, click [here](https://www.dropbox.com/s/unicm8ulxt24vh8/CPSC.zip?dl=0).

### Software

- Python 3.7.4
- Matplotlib 3.1.1
- Numpy 1.17.2
- Pandas 0.25.2
- PyTorch 1.2.0
- Scikit-learn 0.21.3
- Scipy 1.3.1
- Shap 0.35.1
- Tqdm 4.36.1
- Wfdb 2.2.1

## Run

### Preprocessing

```sh
$ python preprocess.py --data-dir data/CPSC
```

### Baselines

```sh
$ python baselines.py --data-dir data/CPSC --classifier LR
```

### Deep model

```sh
$ python main.py --data-dir data/CPSC --leads all --use-gpu # training
$ python predict.py --data-dir data/CPSC --leads all --use-gpu # evaluation
```

### Interpretation

```sh
$ python shap_values.py --data-dir data/CPSC --use-gpu # visualizing shap values
```
