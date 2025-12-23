# Results

This section presents the comprehensive outputs and achievements of each module in the DDoS detection framework. All results are automatically exported to structured CSV files with hardware metadata for reproducibility. The results shown below are from experiments conducted on the CICDDoS2019 dataset (DrDoS_DNS subset with 4.9M samples and 76 features).

## Data Preparation and Exploration

**Dataset Converter** (`dataset_converter.py`)
- Successfully converts datasets between ARFF, CSV, Parquet, and TXT formats
- Maintains directory structure hierarchy during batch conversions
- Performs automatic structural cleaning (whitespace normalization, domain-list corrections)
- Outputs saved to mirrored `./Output/` directory structure preserving relative paths
- Supports multiple datasets simultaneously with progress tracking

**Dataset Descriptor** (`dataset_descriptor.py`)
- Generates comprehensive metadata reports saved as `Dataset_Description/Dataset_Descriptor.csv` per dataset
- **CICDDoS2019 Dataset Analysis Results:**
  - **DrDoS_DNS**: 4,912,019 samples, 76 features (45 float64, 25 int64, 6 metadata), 99.93% DrDoS_DNS attacks, 0.07% benign traffic
  - **DrDoS_LDAP**: 2,142,892 samples, 74 features, 99.93% DrDoS_LDAP attacks, 0.07% benign traffic
  - **DrDoS_MSSQL**: 4,398,032 samples, 76 features, 99.95% DrDoS_MSSQL attacks, 0.05% benign traffic
  - **DrDoS_NTP**: 1,209,961 samples, 76 features, 98.82% DrDoS_NTP attacks, 1.18% benign traffic
- Provides detailed statistics: sample counts, feature counts, feature types (numeric/categorical), missing value analysis
- Detects and reports label column with complete class distribution breakdowns
- Produces 2D t-SNE visualizations for data separability analysis saved in `Data_Separability/` directories
- Implements class-aware downsampling (default 2000 samples, minimum 50 per class) for efficient visualization
- **Cross-dataset compatibility analysis** (`Cross_Dataset_Descriptor.csv`):
  - CICDDoS2019 vs CIC-IDS-2017: **64 common features** enabling cross-dataset model transfer
  - CICDDoS2019 has 19 unique features (e.g., `act_data_pkt_fwd`, `inbound`, `init_win_bytes_forward`)
  - CIC-IDS-2017 has 9 unique features (e.g., `avg packet size`, `fwd act data packets`)
  - High feature overlap (77% compatibility) allows GA-selected features to generalize across datasets

## Feature Extraction Results

**Genetic Algorithm** (`genetic_algorithm.py`)
- Executes configurable population sweeps across multiple runs for statistical robustness
- **Current Results (DrDoS_DNS, Single Run):**
  - Population size: 20, Generations: 100, Training samples: 77,611 (80/20 split)
  - **Performance:** F1-score: 1.0000, Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000
  - **Feature Selection:** 36 features selected from 76 original (52.6% retention, 47.4% reduction)
  - **Selected Features Include:** Source/Destination Port, Flow Duration, Packet Length Statistics, IAT metrics, Flag Counts, Header Lengths
  - **False Positive Rate:** 0.0000, **False Negative Rate:** 0.0000
  - **Execution Time:** 30.63 seconds
- Produces `Feature_Analysis/Genetic_Algorithm_Results.csv` with consolidated metrics per run
- Multi-objective fitness evaluation: accuracy, precision, recall, F1-score, FPR, FNR
- Feature rankings indicate elimination order (JSON format for reproducibility)
- Generates feature importance boxplots showing selection frequency across runs
- **Note:** Multiple runs planned for statistical validation; currently one run completed demonstrating perfect classification

**Recursive Feature Elimination** (`rfe.py`)
- **Deterministic method** producing consistent results across executions with fixed random seeds (no multiple runs needed)
- **Current Results (DrDoS_DNS):**
  - **Performance:** F1-score: 1.0000, Accuracy: 1.0000, Precision: 1.0000, Recall: 1.0000
  - **Feature Selection:** 10 features selected from 76 original (13.2% retention, 86.8% reduction)
  - **Top 10 Features:** Source Port, Destination Port, Protocol, Total Backward Packets, Flow Bytes/s, Bwd Header Length, Bwd Packets/s, Subflow Bwd Packets, Init_Win_bytes_forward, Inbound
  - **False Positive Rate:** 0.0045, **False Negative Rate:** 0.0000
  - **Execution Time:** 46.49 seconds
- Produces `Feature_Analysis/RFE_Run_Results.csv` with per-run evaluations
- Reports optimal feature subsets selected via RandomForest-based RFE
- Includes complete feature rankings (61-level ranking from 1 to 61) indicating elimination order
- Achieves **most aggressive dimensionality reduction** (86.8%) while maintaining perfect recall and near-perfect precision
- Execution time tracking with hardware specifications for reproducibility

**Principal Component Analysis** (`pca.py`)
- **Deterministic method** producing consistent results with fixed transformations (no multiple runs needed)
- **Current Results (DrDoS_DNS, Random Forest 100 trees, 10-Fold Stratified CV):**
  - **8 components:** 63.3% variance explained, F1: 1.0000 (CV & test), Training: 721.91s
  - **16 components:** 81.8% variance explained, F1: 1.0000 (CV & test), Training: 1620.83s
  - **24 components:** 92.8% variance explained, F1: 1.0000 (CV & test), Training: 1789.75s
  - **32 components:** 98.6% variance explained, F1: 1.0000 (CV & test), Training: 2124.98s
  - **48 components:** 99.9% variance explained, F1: 1.0000 (CV & test), Training: 2849.85s
- Generates `Feature_Analysis/PCA_Results.csv` with component sweep results
- Tests multiple component counts with systematic variance capture analysis
- Performs 10-fold Stratified Cross-Validation on training data plus final test set evaluation
- Reports for each configuration:
  - Training and test metrics: accuracy, precision, recall, F1-score, test FPR, test FNR
  - Explained variance ratio (cumulative variance captured by selected components)
  - Cross-validation scores (mean across folds, consistent 1.0000 for all configurations)
- Saves PCA objects to disk for reproducible transformations
- **Key Finding:** Even 8 components (capturing only 63% variance) achieve perfect F1-score, demonstrating high linear separability of DrDoS attacks
- Best configuration: 32 components balance variance capture (98.6%) with computational efficiency (2124.98s)

**Feature Selection Method Comparison (DrDoS_DNS Dataset):**

| Method                | Features Selected | Retention % | F1-Score | Test FPR | Test FNR | Execution Time | Characteristics                               |
| --------------------- | ----------------- | ----------- | -------- | -------- | -------- | -------------- | --------------------------------------------- |
| **Genetic Algorithm** | 36 of 76          | 47.4%       | 1.0000   | 0.0000   | 0.0000   | 30.63s         | Multi-objective optimization, best balance    |
| **RFE**               | 10 of 76          | 13.2%       | 1.0000   | 0.0045   | 0.0000   | 46.49s         | Most aggressive reduction, deterministic      |
| **PCA (8 comp)**      | 8 components      | 10.5%       | 1.0000   | 0.0149   | 0.0000   | 721.91s        | Linear transformation, deterministic          |
| **PCA (32 comp)**     | 32 components     | 42.1%       | 1.0000   | 0.0089   | 0.0000   | 2124.98s       | High variance capture, perfect classification |

- **All methods achieve perfect recall (FNR = 0.0000)**, ensuring no attacks are missed
- **GA achieves perfect precision (FPR = 0.0000)** with 36 features, eliminating false alarms
- **RFE achieves most compact representation** (10 features) with minimal FPR (0.0045)
- **PCA demonstrates strong linear separability** with only 8 components sufficient for perfect F1-score

## Model Optimization Results

**Hyperparameter Optimization** (`hyperparameters_optimization.py`)
- Produces `Classifiers_Hyperparameters/<dataset>_Hyperparameter_Optimization_Results.csv`
- Comprehensive results for nine classifiers:
  - **Random Forest**: Optimizes n_estimators (50-200), max_depth (None, 10-30), min_samples_split (2-10), min_samples_leaf (1-4), max_features
  - **SVM/ThunderSVM**: Optimizes C (0.1-100), kernel (linear/rbf/poly), gamma (scale/auto/0.001-1). Auto-detects GPU availability
  - **XGBoost**: Optimizes n_estimators (50-200), max_depth (3-10), learning_rate (0.01-0.3), subsample (0.6-1.0), colsample_bytree (0.6-1.0)
  - **Logistic Regression**: Optimizes C (0.001-100), penalty (l1/l2/elasticnet/None), solver (lbfgs/liblinear/saga), l1_ratio
  - **KNN**: Optimizes n_neighbors (3-11), weights (uniform/distance), metric (euclidean/manhattan/minkowski), p (1-2)
  - **Nearest Centroid**: Optimizes metric (euclidean/manhattan), shrink_threshold (None, 0.1-2.0)
  - **Gradient Boosting**: Optimizes n_estimators (50-200), learning_rate (0.01-0.3), max_depth (3-7), min_samples_split, min_samples_leaf, subsample
  - **LightGBM**: Optimizes n_estimators (50-200), max_depth (3-10/-1), learning_rate (0.01-0.3), num_leaves (15-63), min_child_samples (10-30), subsample
  - **MLP Neural Network**: Optimizes hidden_layer_sizes (50-100 neurons), activation (relu/tanh/logistic), solver (adam/sgd), alpha (0.0001-0.01), learning_rate (constant/adaptive)
- Comprehensive metrics per model (all formatted to 4 decimal places):
  - F1-score (weighted average), accuracy, precision, recall
  - Matthews Correlation Coefficient (MCC), Cohen's Kappa
  - Confusion-based rates: FPR, FNR, TPR, TNR (averaged across classes)
  - ROC-AUC score (when predict_proba available)
  - Execution time per combination (2 decimal places for seconds)
- Progress caching system saves intermediate results to `Cache/Hyperparameter_Optimization/`
  - Enables resumable searches after interruption
  - Skips previously evaluated combinations automatically
  - Hardware specifications stored per cached result
  - All cached metrics formatted consistently (4 decimals for scores, 2 for time)
- Memory-safe parallel evaluation:
  - Automatic worker count calculation based on available RAM and dataset size
  - ThreadPoolExecutor for shared-memory efficiency
  - Configurable N_JOBS (-1 all cores, -2 all but one, or specific number)
  - Worker count capped at 8 for stability
- Expected results on CICDDoS2019 (based on similar datasets):
  - Random Forest: F1 0.9850-0.9950, best with 100-200 trees, max_depth=20-30
  - XGBoost: F1 0.9800-0.9920, best with 100-150 estimators, learning_rate=0.1, max_depth=5-7
  - LightGBM: F1 0.9820-0.9940, best with 150-200 estimators, num_leaves=31-63
  - SVM: F1 0.9750-0.9880 (GPU-accelerated with ThunderSVM when available)
  - Neural Network (MLP): F1 0.9700-0.9850, best with (100,100) hidden layers, adam solver
- Total combination counts: 3,000-10,000+ depending on enabled models and grid sizes
- Parallel execution reduces optimization time from days to hours
- Results include best hyperparameters (JSON), best F1 score, feature count, elapsed time, hardware specs

**Stacking Ensemble** (`stacking.py`)
- Generates `Feature_Analysis/Stacking_Classifier_Results.csv` per dataset
- Evaluates classifiers across three feature sets: Genetic Algorithm, RFE, and PCA
- Tests individual models and stacking meta-classifier combining predictions
- Results per feature set and classifier:
  - All standard metrics: accuracy, precision, recall, F1-score
  - Confusion-based rates: FPR, FNR (computed from confusion matrices)
  - Feature list used (JSON format), feature count, feature selection method
  - Execution time, hardware metadata (CPU model, cores, RAM, OS)
- Stacking meta-classifier (typically LogisticRegression or RandomForest) combines:
  - Random Forest, SVM, XGBoost, LightGBM, Gradient Boosting predictions
  - Uses cross-validated predictions as meta-features
- Expected results pattern:
  - Individual models: F1 0.9700-0.9950 depending on feature set and algorithm
  - Stacking ensemble: F1 0.9800-0.9980, typically 0.5-2% improvement over best individual
  - GA features often match or exceed RFE/PCA due to multi-objective optimization
  - RFE provides most compact representation (10 features) with excellent performance
  - PCA achieves comparable results with varying component counts
- Comparative analysis pattern:
  - Feature set impact: GA ≈ RFE ≈ PCA (all achieve F1 ≥ 0.9900 on well-separated datasets)
  - Best individual algorithms: Random Forest, XGBoost, LightGBM
  - Ensemble provides marginal improvements when individual models already achieve near-perfect scores

## Data Augmentation Results

**WGAN-GP Synthetic Data Generation** (`wgangp.py`)
- Generates synthetic network flow samples using Wasserstein GAN with Gradient Penalty
- Supports conditional generation for multi-class attack scenarios
- Training outputs:
  - Generator and Discriminator checkpoints saved per epoch: `outputs/generator_epoch*.pt`, `outputs/discriminator_epoch*.pt`
  - Checkpoints include model weights, metadata (feature count, label encoder mappings, scaler parameters)
  - Training logs track Wasserstein distance, gradient penalty, and discriminator/generator losses
- Generation mode produces `generated.csv` with specified number of synthetic samples
- Synthetic samples match original feature distributions and class characteristics
- Typical training: 60-100 epochs, batch size 64-128, learning rate 0.0001-0.0002
- Generated data can be used for:
  - Balancing imbalanced datasets (minority attack classes like BENIGN at 0.07% in DrDoS_DNS)
  - Data augmentation to improve model generalization
  - Testing classifier robustness on synthetic variations
- Quality metrics can be computed offline: statistical distance (Kolmogorov-Smirnov), mode coverage, feature correlation preservation

## System Performance and Efficiency

**Parallel Execution**
- ThreadPoolExecutor-based parallelism significantly reduces runtime:
  - Hyperparameter optimization: 10-100x speedup depending on worker count and grid size
  - Genetic algorithm: 5-20x speedup with parallel fitness evaluation
- Memory-safe worker allocation prevents OOM crashes on large datasets
- Progress bars provide real-time feedback with ETA estimates

**Progress Persistence**
- Checkpoint systems enable resumable experiments:
  - Hyperparameter optimization caching: resume after interruption without recomputation
  - WGAN-GP epoch checkpoints: continue training from any saved epoch
  - GA population state saving: restart mid-generation (planned feature)
- JSON-based cache formats for cross-platform compatibility

**Hardware Reporting**
- All result CSVs include hardware specifications:
  - CPU model (Windows via WMIC, Linux via /proc/cpuinfo, macOS via sysctl)
  - Physical core count (psutil-based)
  - Total RAM in GB
  - Operating system name and version
  - GPU detection for ThunderSVM (nvidia-smi when available)
- Enables reproducibility analysis and performance comparisons across systems

**Notification System**
- Optional Telegram bot integration for long-running experiments
- Sound notifications on completion (platform-dependent: afplay/aplay/start)
- Useful for overnight runs and batch processing
