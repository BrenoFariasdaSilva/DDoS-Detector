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
