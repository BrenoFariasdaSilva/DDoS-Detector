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
