
## Run

### Preprocessing

```sh
$ python preprocess.py --data-dir data/baseline
```

### Baselines

```sh
$ python baselines.py --data-dir data/baseline --classifier LR
```

### Deep model

```sh
$ python main.py --data-dir data/baseline --leads all --use-gpu # training
$ python predict.py --data-dir data/baseline --leads all --use-gpu # evaluation
```

### Interpretation

```sh
$ python shap_values.py --data-dir data/baseline --use-gpu # visualizing shap values
```
