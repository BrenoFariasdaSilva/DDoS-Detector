<div align="center">

# [DDoS-Detector - Results.](https://github.com/BrenoFariasdaSilva/DDoS-Detector)

</div>

This document records verified empirical results for the current repository documentation update. Major evaluation results use only the CICDDoS2019 `01-12` combined-files multi-class Run 1 stacking cache. Single-file, binary, and one-vs-rest attack-class results are not used as headline results.

## Evaluation Context

Authoritative stacking result source for this update:

```text
Cache-Datasets_CICDDoS2019_01_12-Stacking_Classifiers_Results_Run_1.csv
```

Verified Run 1 schema:

```text
experiment_id, experiment_run, experiment_mode, execution_mode, data_source,
dataset, attack_types_combined, augmentation_ratio, feature_selection_enabled,
hyperparameters_enabled, data_augmentation_enabled, hyperparameter_mode,
feature_set, classifier_type, model_name, model, n_features, n_samples_train,
n_samples_test, accuracy, precision, recall, f1_score, fpr, fnr,
elapsed_time_s, cv_method, rfe_ranking, hyperparameters, features_list
```

Run 1 scope:

| Field | Value |
| --- | --- |
| Rows | 99 |
| Dataset identity | `Datasets/CICDDoS2019/01-12/` |
| Execution mode | `combined_files` |
| Experiment mode | `original_only` |
| Data source | `Original Combined Files` |
| Augmentation ratio | `0.0` |
| Class labels | BENIGN, DrDoS_DNS, DrDoS_LDAP, DrDoS_MSSQL, DrDoS_NTP, DrDoS_NetBIOS, DrDoS_SNMP, DrDoS_SSDP, DrDoS_UDP, Syn, TFTP, UDP-lag, WebDDoS |
| Train samples | 38,959,900 for most rows; 38,959,823 for LSTM rows |
| Test samples | 9,739,976 for most rows; 9,739,899 for LSTM rows |
| Stacking meta-classifier rows in this CSV | None; all rows have `classifier_type = Individual` |

## Feature-Selection Results

Feature-analysis files used:

```text
Feature_Analysis/Genetic_Algorithm/Genetic_Algorithm_Results.csv
Feature_Analysis/RFE/RFE_Run_Results.csv
Feature_Analysis/PCA/PCA_Results.csv
Feature_Analysis/Extra_Trees/Extra_Trees_Results.csv
```

### Feature-Selection Summary

| Method | Result Identity | Selected Representation | Test F1-Score | Test Accuracy | Test Precision | Test Recall | Test FPR | Test FNR | Runtime Fields |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Full | No reduction in `stacking.py` | 68-70 features in Run 1 rows | Reported in stacking table | Reported in stacking table | Reported in stacking table | Reported in stacking table | Reported in stacking table | Reported in stacking table | Per classifier in Run 1 |
| PCA | Best available sweep rows tie on rounded F1 | 8, 16, 24, 32, 48, 64 components | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.0104 to 0.0253 | 0.0000 | Feature extraction 3.38-4.14s; training 767.85-3340.42s; testing 78.69-81.34s |
| RFE | Single exported RFE result | 10 selected features | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.0014 | 0.0000 | Training 360s; testing 6s |
| Genetic Algorithm | Single exported best row | 20 selected features; 67-feature union across runs | 0.999997963 | 0.999997964 | 0.999997964 | 0.999997964 | 0.000000 | 0.002980626 | Feature extraction 1.54s; training 224.93s; testing 1.28s |
| Extra Trees | Ranked Extra-Trees-20 result | 20 selected features | 0.999997963 | 0.999997964 | 0.999997964 | 0.999997964 | 0.002978590 | 0.000002036 | Feature extraction 160s; training 160s; testing 3s; elapsed 3764s |

Feature-analysis metrics above are selector/evaluator artifacts for `DrDoS_DNS.csv`; they are not used as multi-class headline classifier results.

### Genetic Algorithm

Source row: `tool = Genetic Algorithm`, `run_index = best`, `model = RandomForestClassifier`, `cv_method = StratifiedKFold(n_splits=10)`, `train_test_split = 80%/20%`, `scaling = StandardScaler`.

Selected features:

```text
source port, destination port, total backward packets,
total length of bwd packets, fwd packet length mean,
bwd packet length max, bwd packet length min, flow iat min,
fwd iat total, bwd iat total, bwd iat mean, bwd iat std,
bwd packets/s, rst flag count, ack flag count, subflow bwd bytes,
init_win_bytes_backward, idle mean, idle max, inbound
```

### RFE

Source row: `tool = RFE`, `model = Random Forest`, `cv_method = StratifiedKFold(n_splits=10)`, `train_test_split = test_size=0.2`, `scaling = standard`.

Selected features:

```text
Source Port, Destination Port, Protocol, Total Backward Packets,
Flow Bytes/s, Bwd Header Length, Bwd Packets/s, Subflow Bwd Packets,
Init_Win_bytes_forward, Inbound
```

The RFE CSV also stores a 70-entry ranking list. Top ranking entries include Source Port, Destination Port, Flow Bytes/s, Bwd Packets/s, Subflow Bwd Packets, Init_Win_bytes_forward, Inbound, Protocol, and Bwd Header Length.

### PCA

Source rows: `tool = PCA`, `model = Random Forest`, `cv_method = StratifiedKFold(n_splits=10)`, `train_test_split = 80/20 split`, `scaling = StandardScaler`.

| Components | Explained Variance | CV F1-Score | Test F1-Score | Test FPR | Test FNR | Training Time (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 0.6227 | 0.9999 | 0.9999 | 0.0253 | 0.0000 | 767.85 |
| 16 | 0.8159 | 0.9999 | 0.9999 | 0.0163 | 0.0000 | 1501.27 |
| 24 | 0.9266 | 0.9999 | 0.9999 | 0.0149 | 0.0000 | 1638.52 |
| 32 | 0.9854 | 0.9999 | 0.9999 | 0.0119 | 0.0000 | 2039.12 |
| 48 | 0.9999 | 0.9999 | 0.9999 | 0.0104 | 0.0000 | 2688.14 |
| 64 | 0.9999 | 0.9999 | 0.9999 | 0.0134 | 0.0000 | 3340.42 |

`stacking.py` selects the best PCA component count from `PCA_Results.csv` and applies a PCA transformer for the `PCA Components` feature set.

### Extra Trees

Source row: `tool = Extra Trees`, `run_index = ranked`, `model = ExtraTreesClassifier`, `cv_method = StratifiedKFold(n_splits=10)`, `train_test_split = 80%/20%`, `scaling = none`, `n_estimators = 200`, `random_state = 42`, `n_jobs = 1`.

The file contains 70 ranked eligible features; 20 are selected.

| Rank | Feature | Importance |
| ---: | --- | ---: |
| 1 | Inbound | 0.206429 |
| 2 | Source Port | 0.137152 |
| 3 | Protocol | 0.078087 |
| 4 | URG Flag Count | 0.076847 |
| 5 | min_seg_size_forward | 0.041661 |
| 6 | Destination Port | 0.035838 |
| 7 | Down/Up Ratio | 0.033814 |
| 8 | Fwd Packet Length Min | 0.026007 |
| 9 | Min Packet Length | 0.020519 |
| 10 | Init_Win_bytes_forward | 0.019823 |

All selected Extra Trees features:

```text
Inbound, Source Port, Protocol, URG Flag Count, min_seg_size_forward,
Destination Port, Down/Up Ratio, Fwd Packet Length Min, Min Packet Length,
Init_Win_bytes_forward, CWE Flag Count, Bwd Packet Length Min,
ACK Flag Count, Fwd PSH Flags, Subflow Bwd Packets,
Avg Bwd Segment Size, Fwd Packet Length Mean, RST Flag Count,
Avg Fwd Segment Size, Bwd Packet Length Max
```

## Multi-Class Stacking Run 1 Results

These are the major evaluation results for the documentation. They use only the Run 1 combined-files multi-class cache listed in [Evaluation Context](#evaluation-context).

### Top Multi-Class Results By F1-Score

| Rank | Feature Set | Classifier | Hyperparameters | F1-Score | Accuracy | Precision | Recall | FPR | FNR | Features | Runtime (s) |
| ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | RFE Features | LSTM | Default | 0.894754 | 0.907271 | 0.909585 | 0.907271 | 0.007285 | 0.092729 | 10 | 9,909 |
| 2 | GA Features | LSTM | Default | 0.890487 | 0.899561 | 0.902714 | 0.899561 | 0.007567 | 0.100439 | 20 | 584 |
| 3 | Extra Trees Features | LSTM | Default | 0.882844 | 0.891874 | 0.895705 | 0.891874 | 0.008303 | 0.108126 | 20 | 14,407 |
| 4 | Full Features | Random Forest | Default | 0.876252 | 0.881026 | 0.876501 | 0.881026 | 0.009991 | 0.118974 | 70 | 8,879 |
| 5 | Full Features | Random Forest | Optimized | 0.875035 | 0.888302 | 0.888312 | 0.888302 | 0.009790 | 0.111698 | 70 | 2,890 |
| 6 | Full Features | XGBoost | Optimized | 0.873542 | 0.887141 | 0.887363 | 0.887141 | 0.010012 | 0.112859 | 68 | 10,404 |
| 7 | Full Features | XGBoost | Default | 0.872178 | 0.886155 | 0.886552 | 0.886155 | 0.009932 | 0.113845 | 68 | 3,441 |
| 8 | Full Features | LightGBM | Optimized | 0.867977 | 0.885235 | 0.887015 | 0.885235 | 0.010588 | 0.114765 | 68 | 3,479 |
| 9 | Full Features | Gradient Boosting | Default | 0.865378 | 0.883520 | 0.884832 | 0.883520 | 0.010129 | 0.116480 | 68 | 159,738 |
| 10 | PCA Components | Random Forest | Default | 0.864719 | 0.870194 | 0.863950 | 0.870194 | 0.011300 | 0.129806 | 48 | 147,773 |

Best F1-Score and best Accuracy are the same configuration in this Run 1 file: **RFE Features + LSTM + Default Hyperparameters**, with **0.894754 F1-Score** and **0.907271 Accuracy**.

### Best Result By Feature Set And Hyperparameter Mode

| Feature Set | Hyperparameters | Best Classifier | F1-Score | Accuracy | Precision | Recall | FPR | FNR | Runtime (s) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RFE Features | Default | LSTM | 0.894754 | 0.907271 | 0.909585 | 0.907271 | 0.007285 | 0.092729 | 9,909 |
| GA Features | Default | LSTM | 0.890487 | 0.899561 | 0.902714 | 0.899561 | 0.007567 | 0.100439 | 584 |
| Extra Trees Features | Default | LSTM | 0.882844 | 0.891874 | 0.895705 | 0.891874 | 0.008303 | 0.108126 | 14,407 |
| Full Features | Default | Random Forest | 0.876252 | 0.881026 | 0.876501 | 0.881026 | 0.009991 | 0.118974 | 8,879 |
| Full Features | Optimized | Random Forest | 0.875035 | 0.888302 | 0.888312 | 0.888302 | 0.009790 | 0.111698 | 2,890 |
| PCA Components | Default | Random Forest | 0.864719 | 0.870194 | 0.863950 | 0.870194 | 0.011300 | 0.129806 | 147,773 |
| PCA Components | Optimized | KNN | 0.858397 | 0.863974 | 0.857849 | 0.863974 | 0.013518 | 0.136026 | 36,082 |
| GA Features | Optimized | Random Forest | 0.852454 | 0.874489 | 0.877655 | 0.874489 | 0.011248 | 0.125511 | 2,884 |
| RFE Features | Optimized | Random Forest | 0.837992 | 0.863923 | 0.868017 | 0.863923 | 0.012668 | 0.136077 | 3,416 |
| Extra Trees Features | Optimized | LightGBM | 0.819714 | 0.841687 | 0.855692 | 0.841687 | 0.015079 | 0.158313 | 1,632 |

### Best Result By Classifier

| Classifier | Best Feature Set | Hyperparameters | F1-Score | Accuracy | Precision | Recall | FPR | FNR |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| LSTM | RFE Features | Default | 0.894754 | 0.907271 | 0.909585 | 0.907271 | 0.007285 | 0.092729 |
| Random Forest | Full Features | Default | 0.876252 | 0.881026 | 0.876501 | 0.881026 | 0.009991 | 0.118974 |
| XGBoost | Full Features | Optimized | 0.873542 | 0.887141 | 0.887363 | 0.887141 | 0.010012 | 0.112859 |
| LightGBM | Full Features | Optimized | 0.867977 | 0.885235 | 0.887015 | 0.885235 | 0.010588 | 0.114765 |
| Gradient Boosting | Full Features | Default | 0.865378 | 0.883520 | 0.884832 | 0.883520 | 0.010129 | 0.116480 |
| KNN | Full Features | Optimized | 0.858397 | 0.863974 | 0.857850 | 0.863974 | 0.013518 | 0.136026 |
| MLP (Neural Net) | Full Features | Default | 0.852958 | 0.867243 | 0.866822 | 0.867243 | 0.012304 | 0.132757 |
| Tabular ResNet | PCA Components | Default | 0.845181 | 0.872793 | 0.874795 | 0.872793 | 0.011865 | 0.127207 |
| FT-Transformer | PCA Components | Default | 0.840449 | 0.869851 | 0.870506 | 0.869851 | 0.012027 | 0.130149 |
| AutoEncoder | PCA Components | Default | 0.836106 | 0.865107 | 0.863043 | 0.865107 | 0.013417 | 0.134893 |
| Logistic Regression | PCA Components | Default | 0.714344 | 0.747789 | 0.716747 | 0.747789 | 0.064463 | 0.252211 |
| ResNet18 | PCA Components | Default | 0.654014 | 0.651344 | 0.784294 | 0.651344 | 0.034740 | 0.348656 |
| Nearest Centroid | Full Features | Optimized | 0.606335 | 0.576145 | 0.731149 | 0.576145 | 0.036436 | 0.423855 |

## Hyperparameter Optimization Artifact

The external hyperparameter-optimization artifact inspected for current optimized/default behavior was:

```text
Classifiers_Hyperparameters/Hyperparameter_Optimization_Results.csv
```

It contains 8 rows for GA-selected features (`n_features = 29`) and these optimized model identities: Random Forest, XGBoost, KNN, Gradient Boosting, LightGBM, MLP (Neural Net), Logistic Regression, and Nearest Centroid. SVM is supported by the code/config but was not present in this inspected result file.

| Model | Best CV F1-Score | Accuracy | Precision | Recall | False Positive Rate | False Negative Rate | Runtime (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Random Forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 25.52 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0006 | 0.0006 | 10.16 |
| KNN | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 9720.66 |
| Gradient Boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 961.16 |
| LightGBM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 5.33 |
| MLP (Neural Net) | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 0.0347 | 0.0347 | 194.18 |
| Logistic Regression | 0.9997 | 0.9998 | 0.9997 | 0.9998 | 0.1169 | 0.1169 | 14.09 |
| Nearest Centroid | 0.9997 | 0.9997 | 0.9997 | 0.9997 | 0.0722 | 0.0722 | 3.28 |

These hyperparameter results are not used as headline multi-class stacking results because the requested major evaluation source is the Run 1 combined-files stacking cache.

## Current Code-Backed Interpretation

- `stacking.py` supports both separate-file binary-style evaluation and combined-files multi-class evaluation. The Run 1 file used here is combined-files multi-class only.
- `stacking.py` can evaluate `StackingClassifier`, but the authoritative Run 1 cache inspected here contains only individual classifier rows.
- `stacking.py` supports Full, PCA, RFE, GA, Extra Trees, and explicit feature sets. Feature modes are included only when enabled and backed by usable artifacts, except Full and explicit features.
- `hyperparameters_enabled = False` means default estimator parameters from `config.yaml`/source defaults. `hyperparameters_enabled = True` means optimized parameters loaded from hyperparameter artifacts where matching rows exist.
- `stacking.py` AutoML is Optuna-based model and stacking configuration search with bounded search spaces; it is not documented here as a full general-purpose AutoML platform.
- Explainability and data augmentation are implemented, but the Run 1 source used for major results has `augmentation_ratio = 0.0` and no augmented-data headline result.

## Reproducibility Notes

- Major metrics in README.md and this file use the same Run 1 combined-files multi-class CSV and the same 6-decimal precision policy.
- Feature-selection metrics use the actual CSV files under `Feature_Analysis/`.
- No binary/single-class or one-vs-rest result is used as a major result.
- No personal absolute filesystem path is required to reproduce the documented repository workflow; place result artifacts under the corresponding dataset output directories when rerunning experiments.
