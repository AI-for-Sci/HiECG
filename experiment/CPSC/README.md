
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
