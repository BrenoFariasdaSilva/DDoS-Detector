<div align="center">
  
# [DDoS-Detector.](https://github.com/BrenoFariasdaSilva/DDoS-Detector) <img src="https://github.com/BrenoFariasdaSilva/DDoS-Detector/blob/main/.assets/Icons/DDoS.png"  width="4%" height="4%">

</div>

<div align="center">
  
---

A machine learning framework for Distributed Denial of Service (DDoS) attack detection achieving **100% F1-score and 0% False Negative Rate (FNR)** on the `01-12/DrDoS_DNS.csv` subset of the CICDDoS2019 dataset, and demonstrating **100% F1-score** (multiple classifiers) on the `01-12/UDPLag.csv` subset. Features hyperparameter optimization across nine classifiers (Random Forest, SVM, XGBoost, Logistic Regression, KNN, Nearest Centroid, Gradient Boosting, LightGBM, and MLP), WGAN-GP for synthetic data generation, multi-method feature selection (Genetic Algorithms, RFE, PCA), and stacking ensemble evaluation. Validated on CICDDoS2019 benchmark datasets with full reproducibility and cross-platform support. The framework can also send progress updates and completion notifications (logs or short summaries) to a configured Telegram chat during long-running experiments.
  
---

</div>

<div align="center">

![GitHub Code Size in Bytes](https://img.shields.io/github/languages/code-size/BrenoFariasdaSilva/DDoS-Detector)
![Lines Of Code](https://raw.githubusercontent.com/BrenoFariasdaSilva/DDoS-Detector/image-data/badge.svg)
![GitHub Commits](https://img.shields.io/github/commit-activity/t/BrenoFariasdaSilva/DDoS-Detector/main)
![GitHub Last Commit](https://img.shields.io/github/last-commit/BrenoFariasdaSilva/DDoS-Detector)
![GitHub Forks](https://img.shields.io/github/forks/BrenoFariasdaSilva/DDoS-Detector)
![GitHub Language Count](https://img.shields.io/github/languages/count/BrenoFariasdaSilva/DDoS-Detector)
![GitHub License](https://img.shields.io/github/license/BrenoFariasdaSilva/DDoS-Detector)
![GitHub Stars](https://img.shields.io/github/stars/BrenoFariasdaSilva/DDoS-Detector)
![GitHub Contributors](https://img.shields.io/github/contributors/BrenoFariasdaSilva/DDoS-Detector)
![GitHub Created At](https://img.shields.io/github/created-at/BrenoFariasdaSilva/DDoS-Detector)
![wakatime](https://wakatime.com/badge/github/BrenoFariasdaSilva/DDoS-Detector.svg)

</div>

<div align="center">
  
![RepoBeats Statistics](https://repobeats.axiom.co/api/embed/deca67a753c6ad283c2b87e95f2b676767739706.svg "Repobeats analytics image")

</div>

## Table of Contents
- [DDoS-Detector. ](#ddos-detector-)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Project Architecture](#project-architecture)
    - [Key Technical Features](#key-technical-features)
    - [Workflow Integration](#workflow-integration)
  - [Setup](#setup)
    - [Git](#git)
        - [Linux](#linux)
        - [macOS](#macos)
        - [Windows](#windows)
    - [Clone the Repository](#clone-the-repository)
    - [Python, Pip and Venv](#python-pip-and-venv)
      - [Linux](#linux-1)
      - [macOS](#macos-1)
      - [Windows](#windows-1)
    - [Make](#make)
      - [Linux](#linux-2)
      - [macOS](#macos-2)
      - [Windows](#windows-2)
    - [Dependencies/Requirements](#dependenciesrequirements)
    - [Datasets](#datasets)
  - [Results](#results)
    - [Feature Selection Performance (`DrDoS_DNS`)](#feature-selection-performance-drdos_dns)
    - [Hyperparameter Optimization (`UDPLag`)](#hyperparameter-optimization-udplag)
  - [How to Cite?](#how-to-cite)
  - [Contributing](#contributing)
  - [Collaborators](#collaborators)
  - [License](#license)
    - [Apache License 2.0](#apache-license-20)

## Introduction

This project provides a complete end-to-end machine learning pipeline for DDoS (Distributed Denial of Service) attack detection and classification using network flow data. The framework integrates state-of-the-art techniques for data preprocessing, feature engineering, model optimization, and evaluation to achieve robust and accurate intrusion detection across multiple benchmark datasets.

### Project Architecture

The system is organized into several interconnected modules, each addressing a critical aspect of the machine learning workflow:

**1. Data Preparation and Exploration**
- **Dataset Converter** (`dataset_converter.py`): Multi-format conversion utility supporting ARFF, CSV, Parquet, and TXT formats. Performs lightweight structural cleaning and maintains directory hierarchy during conversion.
- **Dataset Descriptor** (`dataset_descriptor.py`): Generates comprehensive metadata reports including feature types, missing values, class distributions, and 2D t-SNE visualizations for data separability analysis. Produces cross-dataset compatibility reports comparing feature unions and intersections.

**2. Feature Engineering**
- **Genetic Algorithm** (`genetic_algorithm.py`): DEAP-based binary-mask genetic algorithm for optimal feature selection. Uses RandomForest-based fitness evaluation with multi-objective metrics (accuracy, precision, recall, F1, FPR, FNR). Supports population sweeps and exports consolidated results with feature importance rankings.
- **Recursive Feature Elimination** (`rfe.py`): Automated RFE workflow using RandomForestClassifier to iteratively eliminate less important features. Exports structured run results with feature rankings and performance metrics.
- **Principal Component Analysis** (`pca.py`): PCA-based dimensionality reduction with configurable component counts. Performs 10-fold Stratified CV evaluation and saves PCA objects for reproducibility.

**3. Data Augmentation**
- **WGAN-GP** (`wgangp.py`): Wasserstein Generative Adversarial Network with Gradient Penalty for generating synthetic network flow data. Implements conditional generation with residual blocks (DRCGAN-style architecture) for multi-class attack scenarios. Produces high-quality synthetic samples to balance datasets and augment training data.

**4. Model Optimization and Evaluation**
- **Hyperparameter Optimization** (`hyperparameters_optimization.py`): Comprehensive hyperparameter tuning for nine classifiers (Random Forest, SVM, XGBoost, Logistic Regression, KNN, Nearest Centroid, Gradient Boosting, LightGBM, MLP). Features parallel evaluation with ThreadPoolExecutor, progress caching, memory-safe worker allocation, and detailed metric tracking (F1, accuracy, precision, recall, MCC, Cohen's kappa, ROC-AUC, FPR, FNR, TPR, TNR).
- **Stacking Ensemble** (`stacking.py`): Evaluates individual classifiers and stacking meta-classifiers across GA, RFE, and PCA feature sets. Produces consolidated CSV results with hardware metadata for reproducibility.

**5. Utilities and Infrastructure**
- **Logger** (`Logger.py`): Dual-channel logger preserving ANSI color codes for terminal output while maintaining clean log files.
- **Telegram Bot** (`telegram_bot.py`): Notification system for long-running experiments, supporting message splitting for Telegram's character limits. It can send progress updates, status summaries and completion notifications (logs or short reports) to a configured Telegram chat during script execution.
- **Makefile**: Automation for all pipeline stages with cross-platform support (Windows, Linux, macOS) and detached execution modes.

### Key Technical Features

- **Multi-Dataset Support**: Designed for CICDDoS2019, CIC-IDS-2017, and compatible datasets with shared feature definitions
- **Feature Reusability**: GA-selected features can be reused across compatible datasets without retraining
- **Comprehensive Metrics**: Tracks standard metrics (accuracy, precision, recall, F1) plus confusion-based rates (FPR, FNR, TPR, TNR), MCC, Cohen's kappa, and ROC-AUC
- **Progress Persistence**: Checkpoint saving and caching for resumable long-running optimizations
- **Hardware Awareness**: Automatic detection of CPU model, cores, RAM, GPU availability (ThunderSVM), and memory-safe parallel worker allocation
- **Cross-Platform**: Unified codebase with OS-specific adaptations for sound notifications, path handling, and system information retrieval

### Workflow Integration

The typical pipeline execution follows this sequence:

1. **Download/Convert Datasets**: Obtain raw data using `download_datasets.sh` or convert existing formats with `dataset_converter.py`
2. **Describe Datasets**: Generate metadata reports and t-SNE visualizations using `dataset_descriptor.py`
3. **Feature Selection**: Run `genetic_algorithm.py`, `rfe.py`, and `pca.py` to extract optimal feature subsets
4. **Hyperparameter Tuning**: Optimize individual classifiers with `hyperparameters_optimization.py` using GA-selected features
5. **Ensemble Evaluation**: Compare stacking and individual models across feature sets with `stacking.py`
6. **Optional Augmentation**: Generate synthetic samples using `wgangp.py` for dataset balancing
7. **Results Analysis**: Consolidated CSV outputs in `Feature_Analysis/` and `Classifiers_Hyperparameters/` directories

This modular architecture enables researchers to execute the complete pipeline end-to-end or run individual components for targeted analysis.

## Setup

This section provides instructions for installing Git, Python, Pip, Make, then to clone the repository (if not done yet) and all required project dependencies. 

### Git

`git` is a distributed version control system that is widely used for tracking changes in source code during software development. In this project, `git` is used to download and manage the analyzed repositories, as well as to clone the project and its submodules. To install `git`, follow the instructions below based on your operating system:

##### Linux

To install `git` on Linux, run:

```bash
sudo apt install git -y # For Debian-based distributions (e.g., Ubuntu)
```

##### macOS

If you don't have Homebrew installed, you can install it by running the following command in your terminal:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

To install `git` on MacOS, you can use Homebrew:

```bash
brew install git
```

##### Windows

On Windows, you can download `git` from the official website [here](https://git-scm.com/downloads) and follow the installation instructions provided there.

### Clone the Repository

Now that git is installed, it's time to clone this repository with all required submodules, use:

``` bash
git clone --recurse-submodules https://github.com/BrenoFariasdaSilva/DDoS-Detector.git
```

If you clone without submodules (not recommended):

``` bash
git clone https://github.com/BrenoFariasdaSilva/DDoS-Detector
```

To initialize submodules manually:

``` bash
cd DDoS-Detector # Only if not in the repository root directory yet
git submodule init
git submodule update
```

### Python, Pip and Venv

You must have Python 3, Pip, and the `venv` module installed.

#### Linux

``` bash
sudo apt install python3 python3-pip python3-venv -y
```

#### macOS

``` bash
brew install python3
```

#### Windows

If you do not have Chocolatey installed, you can install it by running the following command in an **elevated PowerShell (Run as Administrator)**:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

Once Chocolatey is installed, you can install Python using:

``` bash
choco install python3
```

Or download the installer from the official website [here](https://www.python.org/downloads/windows/) and follow the installation instructions provided there. Make sure to check the option "Add Python to PATH" during installation ans restart your terminal/computer.

### Make 

`Make` is used to run automated tasks defined in the project's Makefile, such as setting up environments, executing scripts, and managing Python dependencies.

#### Linux

``` bash
sudo apt install make -y
```

#### macOS

``` bash
brew install make
```

#### Windows

Available via Cygwin, MSYS2, or WSL.

### Dependencies/Requirements

1. Install the project dependencies with the following command:

   ```bash
   cd DDoS-Detector # Only if not in the repository root directory yet
   make dependencies
   ```

   This command will create a virtual environment in the `.venv` folder and install all required dependencies listed in the `requirements.txt` file.

### Datasets

This repository includes a shell script `download_datasets.sh` (at the repository root) that can automatically download and extract several datasets used by the project. The current script downloads and extracts:

   - CICDDoS2019 (two CSV ZIPs: `CSV-01-12.zip` and `CSV-03-11.zip`) into `Datasets/CICDDoS2019`
   - CIC-IDS-2017 (the labelled flows ZIP) into `Datasets/CICIDS2017`

   The script behavior:
   - Creates the main `Datasets` directory if it does not exist.
   - For each configured dataset it creates the target directory (from `DATASET_DIRS`).
   - Downloads the configured ZIP file(s) with `wget -c` (resumable downloads).
   - Extracts each ZIP using `unzip -o` into the dataset directory.

   Requirements: `wget` and `unzip` must be installed and available on your PATH, and you must have an active internet connection.

   Usage (from the repository root):

   ```bash
   make download_datasets
   ```

   After successful run the files will be available under the `Datasets` subfolders, for example:

   - `Datasets/CICDDoS2019/CSV-01-12.zip` (unzipped CSV files will be inside this directory)
   - `Datasets/CICDDoS2019/CSV-03-11.zip`
   - `Datasets/CICIDS2017/GeneratedLabelledFlows.zip`

   Note: the ZIP filenames and the exact extraction layout depend on the upstream archive contents; check the target folder after extraction.

   Configuring the script
   - The script is driven by two associative arrays at the top of `download_datasets.sh`:
     - `DATASET_URLS`: maps a short dataset key to the download URL.
     - `DATASET_DIRS`: maps the same dataset key to the local target directory under `Datasets/`.

   - To select which datasets the script downloads, edit `download_datasets.sh` and comment/uncomment the relevant entries in `DATASET_URLS` (or remove entries you don't want). The script iterates over the keys present in `DATASET_URLS` and downloads whatever is configured there.

   - To change where a dataset is extracted, update the corresponding value in `DATASET_DIRS` for that key. Example:

   ```bash
   # in download_datasets.sh
   DATASET_URLS=( [CICDDoS2019_CSV_01_12]="http://.../CSV-01-12.zip" )
   DATASET_DIRS=( [CICDDoS2019_CSV_01_12]="Datasets/CICDDoS2019" )
   ```

   - To add another dataset: add a new key to `DATASET_URLS` with its URL, and add the same key to `DATASET_DIRS` with the desired target folder. Save the file and re-run `./download_datasets.sh`.

1. **Manually downloaded datasets**

   If you prefer to download datasets manually, create the `Datasets` directory (if needed):

   ```bash
   mkdir -p Datasets
   ```

   Then create a subfolder per dataset and place the downloaded CSV(s) or extracted files there. Example structure:

   ```text
   Datasets/
      CICDDoS2019/
         CSV-01-12/ # Extracted CSVs from the first archive
         CSV-03-11/ # Extracted CSVs from the second archive
      CICIDS2017/
         TrafficLabelling/ # Extracted CSVs
   ```

   Primary datasets used in this project:

   - https://www.unb.ca/cic/datasets/ddos-2019.html (CICDDoS2019)
   - https://www.unb.ca/cic/datasets/ids-2017.html (CIC-IDS-2017)

   These datasets were chosen because they share similar feature definitions. This allows feature subsets extracted via the Genetic Algorithm to be reused across multiple datasets, avoiding the need to retrain models from scratch for each dataset/file.

2. **Using other datasets**

   You may use additional datasets as long as they are compatible with the project's preprocessing pipeline. Good sources include:

   - Kaggle: https://www.kaggle.com/datasets
   - UCI: https://archive.ics.uci.edu/ml/index.php

   Ensure any new dataset is adapted to match the expected CSV format, column names (features), and label conventions used by the project. If necessary, use the provided dataset utilities (e.g., `dataset_converter.py` / `dataset_descriptor.py`) to convert or normalize new datasets to the project's expected format.

## Results

The following results demonstrate the framework's performance on the **CICDDoS2019** dataset, specifically the `DrDoS_DNS` and `UDPLag` subsets (01-12 capture).

### Feature Selection Performance (`DrDoS_DNS`)

The table below compares three feature selection strategies using a **Random Forest** classifier. The **Genetic Algorithm (GA)** identified a subset of 36 features that achieved optimal performance (0% FNR) significantly faster than PCA (which required 48 components) and with better minority-class detection than RFE (which missed 0.46% of attacks).

| Metric                        | RFE (10 Features) | PCA (48 Components) | Genetic Algorithm (36 Features) |
| :---------------------------- | :---------------: | :-----------------: | :-----------------------------: |
| **Accuracy**                  |      100.00%      |       100.00%       |           **100.00%**           |
| **Precision**                 |      100.00%      |       100.00%       |           **100.00%**           |
| **Recall**                    |      100.00%      |       100.00%       |           **100.00%**           |
| **F1-Score**                  |      100.00%      |       100.00%       |           **100.00%**           |
| **False Negative Rate (FNR)** |       0.46%       |        0.00%        |            **0.00%**            |
| **Execution Time**            |      53.89s       |      2849.85s       |           **30.63s**            |

### Hyperparameter Optimization (`UDPLag`)

The table below summarizes the best configurations and performance metrics for various classifiers on the `UDPLag` subset, using the 30 features selected by the Genetic Algorithm. Several models achieved perfect classification on this highly imbalanced multi-class subset.

| Model                   |  F1-Score   |  Accuracy   | Execution Time | Best Parameters (Partial)                     |
| :---------------------- | :---------: | :---------: | :------------: | :-------------------------------------------- |
| **Random Forest**       | **100.00%** | **100.00%** |     1.27s      | `n_estimators=50`, `max_depth=30`             |
| **XGBoost**             | **100.00%** | **100.00%** |     0.46s      | `n_estimators=50`, `max_depth=10`             |
| **LightGBM**            | **100.00%** | **100.00%** |     0.90s      | `n_estimators=50`, `num_leaves=31`            |
| **Gradient Boosting**   | **100.00%** | **100.00%** |     55.04s     | `n_estimators=200`, `max_depth=7`             |
| **KNN**                 | **100.00%** | **100.00%** |     60.49s     | `n_neighbors=9`, `weights=distance`           |
| **MLP**                 |   99.98%    |   99.98%    |     7.18s      | `hidden_layer_sizes=[100]`, `activation=relu` |
| **Logistic Regression** |   99.88%    |   99.89%    |     4.22s      | `C=100`, `solver=lbfgs`                       |
| **Nearest Centroid**    |   99.50%    |   99.43%    |     0.22s      | `metric=manhattan`                            |

**📊 For detailed experimental results, performance benchmarks, and feature listings, please see [RESULTS.md](RESULTS.md).**

## How to Cite?

If you use the DDoS-Detector in your research, please cite it using the following BibTeX entry:

```
@misc{softwareDDoS-Detector:2025,
  title = {A Framework for DDoS Attack Detection Using Hyperparameter Optimization, WGAN-GP–Based Data Augmentation, Feature Extraction via Genetic Algorithms, RFE, and PCA, with Ensemble Classifiers and Multi-Dataset Evaluation},
  author = {Breno Farias da Silva},
  year = {2025},
  howpublished = {https://github.com/BrenoFariasdaSilva/DDoS-Detector},
  note = {Accessed on October 6, 2026}
}
```

Additionally, a `main.bib` file is available in the root directory of this repository, in which contains the BibTeX entry for this project.

If you find this repository valuable, please don't forget to give it a ⭐ to show your support! Contributions are highly encouraged, whether by creating issues for feedback or submitting pull requests (PRs) to improve the project. For details on how to contribute, please refer to the [Contributing](#contributing) section below.

Thank you for your support and for recognizing the contribution of this tool to your work!

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**. If you have suggestions for improving the code, your insights will be highly welcome.
In order to contribute to this project, please follow the guidelines below or read the [CONTRIBUTING.md](CONTRIBUTING.md) file for more details on how to contribute to this project, as it contains information about the commit standards and the entire pull request process.
Please follow these guidelines to make your contributions smooth and effective:

1. **Set Up Your Environment**: Ensure you've followed the setup instructions in the [Setup](#setup) section to prepare your development environment.

2. **Make Your Changes**:
   - **Create a Branch**: `git checkout -b feature/YourFeatureName`
   - **Implement Your Changes**: Make sure to test your changes thoroughly.
   - **Commit Your Changes**: Use clear commit messages, for example:
     - For new features: `git commit -m "FEAT: Add some AmazingFeature"`
     - For bug fixes: `git commit -m "FIX: Resolve Issue #123"`
     - For documentation: `git commit -m "DOCS: Update README with new instructions"`
     - For refactorings: `git commit -m "REFACTOR: Enhance component for better aspect"`
     - For snapshots: `git commit -m "SNAPSHOT: Temporary commit to save the current state for later reference"`
   - See more about crafting commit messages in the [CONTRIBUTING.md](CONTRIBUTING.md) file.

3. **Submit Your Contribution**:
   - **Push Your Changes**: `git push origin feature/YourFeatureName`
   - **Open a Pull Request (PR)**: Navigate to the repository on GitHub and open a PR with a detailed description of your changes.

4. **Stay Engaged**: Respond to any feedback from the project maintainers and make necessary adjustments to your PR.

5. **Celebrate**: Once your PR is merged, celebrate your contribution to the project!

## Collaborators

We thank the following people who contributed to this project:

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/BrenoFariasdaSilva" title="Breno Farias da Silva (Founder)">
        <img src="https://github.com/BrenoFariasdaSilva.png" width="100px;" alt="Breno Farias da Silva (Founder)"/><br>
        <sub>
          <b>Breno Farias da Silva</b>
        </sub>
      </a>
    </td>
  </tr>
</table>

## License

### Apache License 2.0

This project is licensed under the [Apache License 2.0](LICENSE). This license permits use, modification, distribution, and sublicense of the code for both private and commercial purposes, provided that the original copyright notice and a disclaimer of warranty are included in all copies or substantial portions of the software. It also requires a clear attribution back to the original author(s) of the repository. For more details, see the [LICENSE](LICENSE) file in this repository.
