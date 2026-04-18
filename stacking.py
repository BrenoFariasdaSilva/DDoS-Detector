"""
================================================================================
Individual and Stacking Ensemble Classifier Evaluation Script (stacking.py)
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-12-01
Description :
    Orchestrates evaluation of individual classifiers and a stacking ensemble
    across multiple feature sets derived from the project's feature-analysis
    artifacts (Genetic Algorithm, RFE, PCA). The script centralizes dataset
    loading, preprocessing, feature-set assembly, per-model evaluation and the
    export of consolidated CSV results for downstream analysis.

Core capabilities:
    - Automatic loading and sanitization of CSV datasets (NaN/infinite removal)
    - Integration of GA, RFE and PCA outputs to build alternative feature sets
    - Scaling, optional PCA projection and selective feature subsetting
    - Evaluation of many classifiers (RF, SVM, XGBoost, LightGBM, etc.) and a
        stacking meta-classifier combining their predictions
    - Calculation of standard metrics (accuracy, precision, recall, F1) plus
        FPR/FNR and elapsed-time reporting
    - Export of `Stacking_Classifier_Results.csv` including `features_list`
        and hardware metadata for reproducibility
    - Utilities to discover feature-analysis files at file, parent or dataset level

Usage:
    - Configure `DATASETS` mapping or call `main()` directly.
    - Run: `python3 stacking.py` or via the repository Makefile target.

Outputs:
    - `Stacking_Classifier_Results.csv` (per-dataset `Feature_Analysis/` directory)
    - Terminal logs, optional Telegram notifications and sound on completion

Notes & conventions:
    - Input CSVs are expected under `Datasets/<DatasetName>/...` and the last
        column conventionally contains the target variable.
    - Feature-analysis artifacts are expected under `.../Feature_Analysis/`:
        `Genetic_Algorithm_Results.csv`, `RFE_Run_Results.csv`, `PCA_Results.csv`.
    - Defaults assume CSV input; Parquet support can be added as needed.
    - Toggle `VERBOSE = True` for additional diagnostic output.

Execution Modes:
    - `binary`: Standard binary classification where every attack is treated as
        a single positive class and non-attack as negative.
    - `multi-class`: Multi-class classification where each distinct attack type
        is treated as its own class label for per-attack evaluation.
    - `both`: Run both `binary` and `multi-class` evaluations; outputs are
        written under separate subfolders (`binary` and `multi-class`).
    - Default: `both` (configurable via CLI or `CONFIG['execution']['execution_mode']`).

TODOs:
    - Add CLI argument parsing for dataset paths and runtime flags.
    - Add native Parquet support and safer large-file streaming.
    - Add voting ensemble baseline and parallelize per-feature-set evaluations.

Dependencies:
    - Python >= 3.8
    - pandas, numpy, scikit-learn, colorama, lightgbm, xgboost
    - Optional: telegram_bot for notifications
"""

import arff  # Liac-arff, used to load ARFF files
import argparse  # For parsing command-line arguments
import ast  # For safely evaluating Python literals
import atexit  # For playing a sound when the program finishes
import concurrent.futures  # For parallel execution
import dataframe_image as dfi  # For exporting DataFrame styled tables as PNG images
import datetime  # For getting the current date and time
import gc  # For explicit garbage collection to reclaim memory from deleted objects
import glob  # For file pattern matching
import itertools  # For generating combinations of FS/HP/DA
import json  # Import json for handling JSON strings within the CSV
import lightgbm as lgb  # For LightGBM model
import math  # For mathematical operations
import matplotlib  # For plotting configuration
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt  # For creating t-SNE visualization plots
import numpy as np  # Import numpy for numerical operations
import optuna  # For Bayesian hyperparameter optimization (AutoML)
import os  # For running a command in the terminal
import pandas as pd  # Import pandas for data manipulation
import pickle  # For loading PCA objects
import platform  # For getting the operating system name
import psutil  # For checking system RAM
import re  # For regular expressions
import seaborn as sns  # For generating feature usage heatmaps
import shap  # For SHAP explainability analysis
import subprocess  # For running small system commands (sysctl/wmic)
import sys  # For system-specific parameters and functions
import telegram_bot as telegram_module  # For setting Telegram prefix and device info
import time  # For measuring execution time
import traceback  # For formatting and printing exception tracebacks
import yaml  # Import YAML library
from colorama import Style  # For terminal text styling
from joblib import dump, load  # For exporting and loading trained models and scalers
from lime.lime_tabular import LimeTabularExplainer  # Import LIME library
from Logger import Logger  # For logging output to both terminal and file
from pathlib import Path  # For handling file paths
from scipy.io import arff as scipy_arff  # Used to read ARFF files
from sklearn.decomposition import PCA  # For Principal Component Analysis
from sklearn.ensemble import (  # For ensemble models
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.inspection import permutation_importance  # Import permutation importance
from sklearn.linear_model import LogisticRegression  # For logistic regression model
from sklearn.manifold import TSNE  # For t-SNE dimensionality reduction
from sklearn.metrics import (  # For performance metrics
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split  # For splitting the dataset and CV
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid  # For k-nearest neighbors model
from sklearn.neural_network import MLPClassifier  # For neural network model
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For label encoding and feature scaling
from sklearn.svm import SVC  # For Support Vector Machine model
from sklearn.tree import DecisionTreeClassifier  # For Decision Tree classifier model
from telegram_bot import TelegramBot, send_exception_via_telegram, send_telegram_message, setup_global_exception_hook  # For sending progress messages to Telegram
from tqdm import tqdm  # For progress bars
from typing import Optional  # For optional typing hints
from xgboost import XGBClassifier  # For XGBoost classifier


# Macros:
class BackgroundColors:  # Colors for the terminal
    CYAN = "\033[96m"  # Cyan
    GREEN = "\033[92m"  # Green
    YELLOW = "\033[93m"  # Yellow
    RED = "\033[91m"  # Red
    BOLD = "\033[1m"  # Bold
    UNDERLINE = "\033[4m"  # Underline
    CLEAR_TERMINAL = "\033[H\033[J"  # Clear the terminal


# Global Configuration Container:
CONFIG = {}  # Will be initialized by initialize_config() - holds all runtime settings

# Telegram Bot Setup:
TELEGRAM_BOT = None  # Global Telegram bot instance (initialized in setup_telegram_bot)

# Logger Setup:
logger = None  # Will be initialized in initialize_logger()


# Functions Definitions:


def get_stacking_output_dir(dataset_file_path: str, config: dict) -> str:
    """
    Get the output directory for stacking results based on the dataset file path and configuration.
    
    :param dataset_file_path: The path to the dataset file being processed
    :param config: The configuration dictionary containing the stacking results directory setting
    :return: The resolved path to the stacking results directory for the given dataset
    """

    if not isinstance(dataset_file_path, (str, Path)):
        raise ValueError("dataset_file_path must be a path string or Path")
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")

    dataset_file = Path(dataset_file_path).resolve()
    dataset_root = dataset_file.parent

    stacking_cfg = config.get("stacking", {})
    results_dir = stacking_cfg.get("results_dir")
    if not isinstance(results_dir, str) or results_dir.strip() == "":
        raise ValueError("Missing or invalid config['stacking']['results_dir']")
    if os.path.isabs(results_dir):
        raise ValueError("config['stacking']['results_dir'] must be relative (not absolute)")

    stacking_dir = dataset_root / results_dir
    stacking_dir.mkdir(parents=True, exist_ok=True)

    try:
        common = os.path.commonpath([str(dataset_root.resolve()), str(stacking_dir.resolve())])
    except Exception:
        raise RuntimeError("Invalid paths provided for stacking directory validation")
    if common != str(dataset_root.resolve()):
        raise RuntimeError("Resolved stacking directory is not inside the dataset root")

    return str(stacking_dir.resolve())


def validate_output_path(base_dir: str, target_path: str) -> None:
    """
    Validate that the target_path is safely within the base_dir to prevent directory traversal issues.

    :param base_dir: The base directory that should contain the target path
    :param target_path: The target path to validate
    :return: None
    """
    
    if base_dir is None:
        raise RuntimeError("base_dir is None")
    
    base_dir = os.path.abspath(base_dir)
    target_path = os.path.abspath(target_path)
    
    try:
        common = os.path.commonpath([base_dir, target_path])
    except Exception:
        raise RuntimeError("Invalid paths provided for validation")
    
    if common != base_dir:
        raise RuntimeError("Unsafe path detected outside stacking output directory")


def parse_cli_args():
    """
    Parse command-line arguments for stacking pipeline.
    
    :return: Namespace object containing parsed arguments
    """

    try:
        parser = argparse.ArgumentParser(
            description="Run stacking classifier evaluation pipeline with optional AutoML and data augmentation testing."
        )  # Create argument parser
        parser.add_argument("--config", type=str, default=None, help="Path to config.yaml file")
        parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
        parser.add_argument("--skip-train-if-model-exists", dest="skip_train", action="store_true", help="Load existing models instead of retraining")
        parser.add_argument("--csv", type=str, default=None, help="Path to specific CSV file to process")
        parser.add_argument("--dataset-path", type=str, default=None, help="Path to a dataset directory or single CSV file to process")
        parser.add_argument("--enable-automl", dest="enable_automl", action="store_true", default=None, help="Enable AutoML pipeline method toggle")  # Enable AutoML via CLI
        parser.add_argument("--disable-automl", dest="enable_automl", action="store_false", help="Disable AutoML pipeline method toggle")  # Disable AutoML via CLI
        parser.add_argument("--automl-trials", type=int, default=None, help="Number of AutoML trials")
        parser.add_argument("--automl-stacking-trials", type=int, default=None, help="Number of stacking trials")
        parser.add_argument("--automl-timeout", type=int, default=None, help="AutoML timeout in seconds")
        parser.add_argument("--test-augmentation", action="store_true", help="Enable data augmentation testing")
        parser.add_argument("--no-test-augmentation", dest="test_augmentation", action="store_false", help="Disable data augmentation testing")
        parser.add_argument("--multi-class", action="store_true", help="Enable multi-class classification mode (combine all attacks)")
        parser.add_argument("--binary", dest="multi_class", action="store_false", help="Enable binary classification mode (default)")
        parser.add_argument("--both", action="store_true", help="Run both binary and multi-class pipelines sequentially")
        parser.add_argument("--stacking-results-dir", type=str, default=None, help="Directory to save stacking results (relative to dataset root)")
        parser.add_argument("--top-n-features", dest="top_n_features", type=int, default=None, help="Number of top features to show in heatmap (overrides config)")
        parser.add_argument("--enable-augmentation", dest="enable_augmentation", action="store_true", default=None, help="Enable data augmentation method toggle")
        parser.add_argument("--disable-augmentation", dest="enable_augmentation", action="store_false", help="Disable data augmentation method toggle")
        parser.add_argument("--enable-feature-selection", dest="enable_feature_selection", action="store_true", default=None, help="Enable feature selection method toggle")
        parser.add_argument("--disable-feature-selection", dest="enable_feature_selection", action="store_false", help="Disable feature selection method toggle")
        parser.add_argument("--enable-hyperparameters", dest="enable_hyperparameters", action="store_true", default=None, help="Enable hyperparameter optimization method toggle")
        parser.add_argument("--disable-hyperparameters", dest="enable_hyperparameters", action="store_false", help="Disable hyperparameter optimization method toggle")
        parser.add_argument("--low-memory", dest="low_memory", action="store_true", default=False, help="Enable low memory mode for pandas operations")  # Add low memory mode CLI argument
        parser.add_argument("--dataset-file-format", type=str, default=None, dest="dataset_file_format", help="File format for dataset files: arff, csv, parquet, txt")  # Dataset file format CLI override
        parser.add_argument("--augmentation-file-format", type=str, default=None, dest="augmentation_file_format", help="File format for augmentation files: arff, csv, parquet, txt")  # Augmentation file format CLI override
        
        return parser.parse_args()  # Return parsed arguments
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def get_default_stacking_config():
    """
    Return default stacking pipeline configuration section.

    :return: Dictionary containing default stacking configuration values
    """

    try:
        return {
            "results_dir": "Stacking",  # Output subdirectory name for stacking results
            "results_filename": "Stacking_Classifiers_Results.csv",  # Binary results CSV filename
            "multiclass_results_filename": "Stacking_Classifiers_MultiClass_Results.csv",  # Multi-class results CSV filename
            "augmentation_comparison_filename": "Data_Augmentation_Comparison_Results.csv",  # Augmentation comparison CSV filename
            "data_augmentation_suffix": "_data_augmented",  # File suffix for augmented data files
            "augmentation_ratios": [0.10, 0.25, 0.50, 0.75, 1.00],  # Ratios of augmented data to sample
            "hyperparameters_filename": "Hyperparameter_Optimization_Results.csv",  # Hyperparameter results CSV filename
            "cache_prefix": "Cache_",  # Prefix for cached model files
            "model_export_base": "Feature_Analysis/Stacking/Models/",  # Base directory for model exports
            "results_csv_columns": [
                "model", "dataset", "execution_mode", "attack_types_combined", "feature_set", "classifier_type", "model_name",
                "data_source", "experiment_id", "experiment_mode", "augmentation_ratio",
                "n_features", "n_samples_train", "n_samples_test", "accuracy",
                "precision", "recall", "f1_score", "fpr", "fnr", "elapsed_time_s",
                "cv_method", "top_features", "rfe_ranking", "hyperparameters",
                "features_list", "Hardware",
            ],  # Column names for results CSV export
            "top_n_features_heatmap": 15,  # Number of top features to show in heatmap
            "methods": {
                "augmentation": True,  # Enable data augmentation combination by default
                "feature_selection": True,  # Enable feature selection combination by default
                "hyperparameter_optimization": True,  # Enable hyperparameter optimization combination by default
                "automl": True,  # Enable AutoML pipeline by default
            },  # Method toggles for stacking pipeline
            "dataset_file_format": "csv",  # File format for dataset files: arff, csv, parquet, txt
            "augmentation_file_format": "csv",  # File format for augmentation files: arff, csv, parquet, txt
            "match_filenames_to_process": [""],  # Filename patterns to match for processing
            "ignore_files": ["Stacking_Classifiers_Results.csv"],  # Files to ignore during processing
            "ignore_dirs": [
                "Classifiers", "Classifiers_Hyperparameters", "Dataset_Description",
                "Data_Separability", "Feature_Analysis",
            ],  # Directories to ignore during processing
        }  # Return default stacking configuration
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_default_models_config():
    """
    Return default model hyperparameters configuration section.

    :return: Dictionary containing default model configuration values
    """

    try:
        return {
            "random_forest": {"n_estimators": 100, "random_state": 42},  # Random Forest default parameters
            "svm": {"kernel": "rbf", "probability": True, "random_state": 42},  # SVM default parameters
            "xgboost": {"eval_metric": "mlogloss", "random_state": 42},  # XGBoost default parameters
            "logistic_regression": {"max_iter": 1000, "random_state": 42},  # Logistic Regression default parameters
            "knn": {"n_neighbors": 5},  # KNN default parameters
            "gradient_boosting": {"random_state": 42},  # Gradient Boosting default parameters
            "lightgbm": {"force_row_wise": True, "min_gain_to_split": 0.01, "random_state": 42, "verbosity": -1},  # LightGBM default parameters
            "mlp": {"hidden_layer_sizes": (100,), "max_iter": 500, "random_state": 42},  # MLP Neural Net default parameters
            "stacking_meta": {"n_estimators": 50, "random_state": 42},  # Stacking meta-estimator default parameters
        }  # Return default models configuration
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_default_explainability_config():
    """
    Return default explainability configuration section.

    :return: Dictionary containing default explainability configuration values
    """

    try:
        return {
            "enabled": True,  # Whether explainability pipeline is enabled
            "shap": True,  # Enable SHAP explanations
            "lime": True,  # Enable LIME explanations
            "permutation_importance": True,  # Enable permutation importance analysis
            "feature_importance": True,  # Enable feature importance extraction
            "pdp": False,  # Disable partial dependence plots by default
            "ice": False,  # Disable individual conditional expectation by default
            "surrogate": False,  # Disable surrogate model explanations by default
            "max_display_features": 20,  # Maximum features to display in explainability outputs
            "lime_num_samples": 1000,  # Number of perturbation samples for LIME
            "lime_num_features": 10,  # Number of features to show in LIME explanations
            "shap_max_samples": 100,  # Maximum samples for SHAP computation
            "random_state": 42,  # Random state for reproducibility
            "output_subdir": "explainability",  # Subdirectory for explainability outputs
        }  # Return default explainability configuration
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_default_config():
    """
    Return default configuration dictionary for stacking pipeline.
    
    All configurable parameters with their default values matching
    genetic_algorithm.py and wgangp.py configuration architecture.
    
    :return: Dictionary containing default configuration values
    """

    try:
        return {
        "execution": {
            "verbose": False,
            "play_sound": True,
            "skip_train_if_model_exists": False,
            "low_memory": False,
            "csv_file": None,
            "process_entire_dataset": False,
            "test_data_augmentation": True,
            "execution_mode": "both",
            "classification_mode": "both",
        },
        "dataset": {
            "remove_zero_variance": True,
            "test_size": 0.2,
            "random_state": 42,
            "datasets": {
                "CICDDoS2019-Dataset": [
                    "./Datasets/CICDDoS2019/01-12/",
                    "./Datasets/CICDDoS2019/03-11/",
                ],
            },
        },
        "stacking": get_default_stacking_config(),  # Stacking pipeline configuration section
        "evaluation": {
            "n_jobs": -1,
            "threads_limit": 2,
            "cv_folds": 10,
            "random_state": 42,
            "ram_threshold_gb": 128,
        },
        "models": get_default_models_config(),  # Model hyperparameters configuration section
        "automl": {
            "enabled": False,
            "n_trials": 50,
            "stacking_trials": 20,
            "timeout": 3600,
            "cv_folds": 5,
            "random_state": 42,
            "stacking_top_n": 5,
            "results_filename": "AutoML_Results.csv",
        },
        "tsne": {
            "perplexity": 30,
            "random_state": 42,
            "n_iter": 1000,
            "figsize": [12, 10],
            "dpi": 300,
            "alpha": 0.6,
            "marker_size": 50,
        },
        "paths": {
            "logs_dir": "./Logs",
            "datasets_dir": None,
            "output_dir": "Feature_Analysis",
        },
        "sound": {
            "enabled": True,
            "commands": {
                "Darwin": "afplay",
                "Linux": "aplay",
                "Windows": "start",
            },
            "file": "./.assets/Sounds/NotificationSound.wav",
        },
        "logging": {
            "enabled": True,
            "clean": True,
        },
        "telegram": {
            "enabled": True,
            "verify_env": True,
        },
        "explainability": get_default_explainability_config(),  # Explainability configuration section
    }  # Return default configuration
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_config_file(config_path=None):
    """
    Load configuration from YAML file.
    
    :param config_path: Path to config.yaml file (None for default config.yaml)
    :return: Dictionary with loaded configuration or empty dict if not found
    """

    try:
        if config_path is None:  # If no path specified
            config_path = Path(__file__).parent / "config.yaml"  # Use default config.yaml
        else:  # Path was specified
            config_path = Path(config_path)  # Convert to Path object
        
        if not config_path.exists():  # If config file doesn't exist
            return {}  # Return empty dict
        
        try:  # Try to load YAML file
            with open(config_path, "r", encoding="utf-8") as f:  # Open config file
                config = yaml.safe_load(f)  # Load YAML safely
            return config or {}  # Return loaded config or empty dict
        except Exception as e:  # If loading fails
            print(f"{BackgroundColors.YELLOW}Warning: Failed to load config file {config_path}: {e}{Style.RESET_ALL}")
            return {}  # Return empty dict on error
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def deep_merge_dicts(base, override):
    """Recursively merge override dict into base dict.
    
    :param base: Base dictionary
    :param override: Override dictionary
    :return: Merged dictionary
    """

    try:
        result = dict(base)  # Copy base dictionary
        for key, value in override.items():  # Iterate override items
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):  # Both are dicts
                result[key] = deep_merge_dicts(result[key], value)  # Recursive merge
            else:  # Not both dicts
                result[key] = value  # Override value
        return result  # Return merged dictionary
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def merge_configs(defaults, file_config, cli_args):
    """
    Merge configurations with priority: CLI > file > defaults.
    
    :param defaults: Default configuration dictionary
    :param file_config: Configuration from file
    :param cli_args: Parsed CLI arguments
    :return: Merged configuration dictionary
    """

    try:
        config = deep_merge_dicts(defaults, file_config)  # Merge file config into defaults
        
        classification_mode = config.get("execution", {}).get("classification_mode", None)  # Read classification_mode from config if present
        if classification_mode is not None:  # If classification_mode was explicitly set in config
            config["execution"]["execution_mode"] = classification_mode  # Map classification_mode to execution_mode for internal use
        
        if cli_args is None:  # If no CLI args
            return config  # Return merged config
        
        if hasattr(cli_args, "verbose") and cli_args.verbose:  # Verbose flag
            config["execution"]["verbose"] = True
        if hasattr(cli_args, "skip_train") and cli_args.skip_train:  # Skip train flag
            config["execution"]["skip_train_if_model_exists"] = True
        if hasattr(cli_args, "csv") and cli_args.csv:  # CSV file override
            config["execution"]["csv_file"] = cli_args.csv
        if hasattr(cli_args, "enable_automl") and cli_args.enable_automl is not None:  # AutoML method toggle CLI override
            config.setdefault("stacking", {}).setdefault("methods", {})["automl"] = cli_args.enable_automl  # Apply AutoML toggle override
        if hasattr(cli_args, "automl_trials") and cli_args.automl_trials is not None:  # AutoML trials
            config["automl"]["n_trials"] = cli_args.automl_trials
        if hasattr(cli_args, "automl_stacking_trials") and cli_args.automl_stacking_trials is not None:  # Stacking trials
            config["automl"]["stacking_trials"] = cli_args.automl_stacking_trials
        if hasattr(cli_args, "automl_timeout") and cli_args.automl_timeout is not None:  # AutoML timeout
            config["automl"]["timeout"] = cli_args.automl_timeout
        if hasattr(cli_args, "test_augmentation"):  # Test augmentation flag (explicit True or False)
            config["execution"]["test_data_augmentation"] = cli_args.test_augmentation
        if hasattr(cli_args, "both") and cli_args.both:  # Both mode flag takes precedence
            config["execution"]["execution_mode"] = "both"
        elif hasattr(cli_args, "multi_class"):  # Multi-class mode flag
            config["execution"]["execution_mode"] = "multi-class" if cli_args.multi_class else "binary"

        if hasattr(cli_args, "top_n_features") and cli_args.top_n_features is not None:  # CLI override for top-N features
            if not isinstance(cli_args.top_n_features, int) or cli_args.top_n_features <= 0:  # Validate positive integer
                raise ValueError("--top-n-features must be an integer greater than 0")  # Raise for invalid value
            config.setdefault("stacking", {})["top_n_features_heatmap"] = cli_args.top_n_features  # Apply override to config

        if hasattr(cli_args, "stacking_results_dir") and cli_args.stacking_results_dir:
            config.setdefault("stacking", {})["results_dir"] = cli_args.stacking_results_dir

        if hasattr(cli_args, "dataset_path") and cli_args.dataset_path is not None:  # Dataset path CLI override
            config["execution"]["dataset_path"] = cli_args.dataset_path  # Store CLI dataset path override

        if hasattr(cli_args, "enable_augmentation") and cli_args.enable_augmentation is not None:  # Augmentation method toggle CLI override
            config.setdefault("stacking", {}).setdefault("methods", {})["augmentation"] = cli_args.enable_augmentation  # Apply augmentation toggle override

        if hasattr(cli_args, "enable_feature_selection") and cli_args.enable_feature_selection is not None:  # Feature selection method toggle CLI override
            config.setdefault("stacking", {}).setdefault("methods", {})["feature_selection"] = cli_args.enable_feature_selection  # Apply feature selection toggle override

        if hasattr(cli_args, "enable_hyperparameters") and cli_args.enable_hyperparameters is not None:  # Hyperparameter optimization method toggle CLI override
            config.setdefault("stacking", {}).setdefault("methods", {})["hyperparameter_optimization"] = cli_args.enable_hyperparameters  # Apply hyperparameter optimization toggle override

        if hasattr(cli_args, "low_memory") and cli_args.low_memory:  # Low memory CLI override
            config["execution"]["low_memory"] = True  # Apply low memory override to config

        if hasattr(cli_args, "dataset_file_format") and cli_args.dataset_file_format is not None:  # Dataset file format CLI override
            config.setdefault("stacking", {})["dataset_file_format"] = cli_args.dataset_file_format  # Apply dataset file format override to config

        if hasattr(cli_args, "augmentation_file_format") and cli_args.augmentation_file_format is not None:  # Augmentation file format CLI override
            config.setdefault("stacking", {})["augmentation_file_format"] = cli_args.augmentation_file_format  # Apply augmentation file format override to config
        
        return config  # Return final merged configuration
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def initialize_config(config_path=None, cli_args=None):
    """
    Initialize global CONFIG with merged configuration.
    
    :param config_path: Path to config file (None for default)
    :param cli_args: Parsed CLI arguments (None for no CLI overrides)
    :return: Merged configuration dictionary
    """

    try:
        defaults = get_default_config()  # Get default configuration
        file_config = load_config_file(config_path)  # Load file configuration
        config = merge_configs(defaults, file_config, cli_args)  # Merge all configurations
        
        global CONFIG  # Access global CONFIG
        CONFIG = config  # Set global CONFIG
        
        return config  # Return merged configuration
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def initialize_logger(config=None):
    """
    Initialize logger using configuration.
    
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        global logger  # Access global logger
        
        logs_dir = config.get("paths", {}).get("logs_dir", "./Logs")  # Get logs directory
        clean = config.get("logging", {}).get("clean", True)  # Get clean flag
        
        os.makedirs(logs_dir, exist_ok=True)  # Ensure logs directory exists
        log_path = Path(logs_dir) / f"{Path(__file__).stem}.log"  # Build log file path
        
        logger = Logger(str(log_path), clean=clean)  # Create Logger instance
        sys.stdout = logger  # Redirect stdout to logger
        sys.stderr = logger  # Redirect stderr to logger
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def verbose_output(true_string="", false_string="", config=None):
    """
    Output a message if verbose mode is enabled in configuration.

    :param true_string: The string to be outputted if verbose is enabled.
    :param false_string: The string to be outputted if verbose is disabled.
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None.
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        verbose = config.get("execution", {}).get("verbose", False)  # Get verbose flag
        
        if verbose and true_string != "":  # If verbose is True and a true_string was provided
            print(true_string)  # Output the true statement string
        elif false_string != "":  # If a false_string was provided
            print(false_string)  # Output the false statement string
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def verify_dot_env_file(config=None):
    """
    Verify if the .env file exists in the current directory.

    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: True if the .env file exists, False otherwise.
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        verify_env = config.get("telegram", {}).get("verify_env", True)  # Get verify_env flag
        
        if not verify_env:  # If verification is disabled
            return True  # Skip verification
        
        env_path = Path(__file__).parent / ".env"  # Path to the .env file
        if not env_path.exists():  # If the .env file does not exist
            print(f"{BackgroundColors.CYAN}.env{BackgroundColors.YELLOW} file not found at {BackgroundColors.CYAN}{env_path}{BackgroundColors.YELLOW}. Telegram messages may not be sent.{Style.RESET_ALL}")
            return False  # Return False

        return True  # Return True if the .env file exists
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def setup_telegram_bot(config=None):
    """
    Set up the Telegram bot for progress messages.

    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None.
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        telegram_enabled = config.get("telegram", {}).get("enabled", True)  # Get telegram enabled flag
        
        if not telegram_enabled:  # If Telegram is disabled
            return  # Skip setup
        
        verbose_output(
            f"{BackgroundColors.GREEN}Setting up Telegram bot for messages...{Style.RESET_ALL}", config=config
        )  # Output the verbose message

        verify_dot_env_file(config)  # Verify if the .env file exists

        global TELEGRAM_BOT  # Declare the module-global telegram_bot variable

        try:  # Try to initialize the Telegram bot
            TELEGRAM_BOT = TelegramBot()  # Initialize Telegram bot for progress messages
            telegram_module.TELEGRAM_DEVICE_INFO = f"{telegram_module.get_local_ip()} - {platform.system()}"  # Set device info string with IP and OS
            telegram_module.RUNNING_CODE = os.path.basename(__file__)  # Set currently running script name
        except Exception as e:
            print(f"{BackgroundColors.RED}Failed to initialize Telegram bot: {e}{Style.RESET_ALL}")  # Report initialization failure to terminal
            TELEGRAM_BOT = None  # Set to None if initialization fails
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def set_threads_limit_based_on_ram(config=None):
    """
    Sets threads limit to 1 if system RAM is below threshold to avoid memory issues.

    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Threads limit value
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Verifying system RAM to set threads_limit...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        threads_limit = config.get("evaluation", {}).get("threads_limit", 2)  # Get threads limit from config
        ram_threshold = config.get("evaluation", {}).get("ram_threshold_gb", 128)  # Get RAM threshold from config
        ram_gb = psutil.virtual_memory().total / (1024**3)  # Get total system RAM in GB

        if ram_gb <= ram_threshold:  # If RAM is less than or equal to threshold
            threads_limit = 1  # Set threads_limit to 1
            verbose_output(
                f"{BackgroundColors.YELLOW}System RAM is {ram_gb:.1f}GB (<={ram_threshold}GB). Setting threads_limit to 1.{Style.RESET_ALL}",
                config=config
            )
        
        return threads_limit  # Return the threads limit value
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def verify_filepath_exists(filepath):
    """
    Verify if a file or folder exists at the specified path.

    :param filepath: Path to the file or folder
    :return: True if the file or folder exists, False otherwise
    """

    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Verifying if the file or folder exists at the path: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}"
        )  # Output the verbose message

        return os.path.exists(filepath)  # Return True if the file or folder exists, False otherwise
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def get_files_to_process(directory_path, file_extension=".csv", config=None):
    """
    Collect all files with a given extension inside a directory (non-recursive).

    Performs validation, respects IGNORE_FILES, and optionally filters by
    MATCH_FILENAMES_TO_PROCESS when defined.

    :param directory_path: Path to the directory to scan
    :param file_extension: File extension to include (default: ".csv")
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Sorted list of matching file paths
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Getting all {BackgroundColors.CYAN}{file_extension}{BackgroundColors.GREEN} files in: {BackgroundColors.CYAN}{directory_path}{Style.RESET_ALL}",
            config=config
        )  # Verbose: starting file collection
        verify_filepath_exists(directory_path)  # Validate directory path exists

        if not os.path.isdir(directory_path):  # Verify if path is a valid directory
            verbose_output(
                f"{BackgroundColors.RED}Not a directory: {BackgroundColors.CYAN}{directory_path}{Style.RESET_ALL}",
                config=config
            )  # Verbose: invalid directory
            return []  # Return empty list for invalid paths

        match_filenames = config.get("stacking", {}).get("match_filenames_to_process", [""])  # Get match filenames from config
        ignore_files = config.get("stacking", {}).get("ignore_files", [])  # Get ignore files from config
        
        match_names = (
            set(match_filenames) if match_filenames not in ([], [""], [" "]) else None
        )  # Load match list or None
        if match_names:
            verbose_output(
                f"{BackgroundColors.GREEN}Filtering to filenames: {BackgroundColors.CYAN}{match_names}{Style.RESET_ALL}",
                config=config
            )  # Verbose: applying filename filter

        files = []  # Accumulator for valid files

        for item in os.listdir(directory_path):  # Iterate directory entries
            item_path = os.path.join(directory_path, item)  # Absolute path
            filename = os.path.basename(item_path)  # Extract just the filename

            if any(ignore == filename or ignore == item_path for ignore in ignore_files):  # Verify if file is in the ignore list
                verbose_output(
                    f"{BackgroundColors.YELLOW}Ignoring {BackgroundColors.CYAN}{filename}{BackgroundColors.YELLOW} (listed in IGNORE_FILES){Style.RESET_ALL}",
                    config=config
                )  # Verbose: ignoring file
                continue  # Skip ignored file

            if os.path.isfile(item_path) and item.lower().endswith(file_extension):  # File matches extension requirement
                if (
                    match_names is not None and filename not in match_names
                ):  # Filename not included in MATCH_FILENAMES_TO_PROCESS
                    verbose_output(
                        f"{BackgroundColors.YELLOW}Skipping {BackgroundColors.CYAN}{filename}{BackgroundColors.YELLOW} (not in MATCH_FILENAMES_TO_PROCESS){Style.RESET_ALL}",
                        config=config
                    )  # Verbose: skipping non-matching file
                    continue  # Skip this file
                files.append(item_path)  # Add file to result list

        return sorted(files)  # Return sorted list for deterministic output
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def get_dataset_name(input_path):
    """
    Extract the dataset name from CSVs path.

    :param input_path: Path to the CSVs files
    :return: Dataset name
    """

    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Extracting dataset name from CSV path: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}"
        )  # Output the verbose message

        datasets_pos = input_path.find("/Datasets/")  # Find the position of "/Datasets/" in the path
        if datasets_pos != -1:  # If "/Datasets/" is found in the path
            after_datasets = input_path[datasets_pos + len("/Datasets/") :]  # Get the substring after "/Datasets/"
            next_slash = after_datasets.find("/")  # Find the next "/"
            if next_slash != -1:  # If there is another "/"
                dataset_name = after_datasets[:next_slash]  # Take until the next "/"
            else:  # If there is no other "/"
                dataset_name = (
                    after_datasets.split("/")[0] if "/" in after_datasets else after_datasets
                )  # No more "/", take the first part if any
        else:  # If "/Datasets/" is not found in the path
            dataset_name = os.path.basename(input_path)  # Fallback to basename if "Datasets" not in path

        return dataset_name  # Return the dataset name
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def process_single_file(f, config=None):
    """
    Process a single dataset file: load, preprocess, and extract target and features.

    :param f: Path to the dataset CSV file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (df_clean, target_col, feat_cols) or None if invalid
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        verbose_output(
            f"{BackgroundColors.GREEN}Processing file: {BackgroundColors.CYAN}{f}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        df = load_dataset(f, config=config)  # Load the dataset from the file
        if df is None:  # If loading failed
            return None  # Return None
        
        remove_zero_variance = config.get("dataset", {}).get("remove_zero_variance", True)  # Get remove zero variance flag from config
        df_clean = preprocess_dataframe(df, remove_zero_variance=remove_zero_variance, config=config)  # Preprocess the dataframe

        del df  # Release raw dataframe to free memory after preprocessing
        gc.collect()  # Force garbage collection to reclaim memory from deleted raw dataframe

        if df_clean is None or df_clean.empty:  # If preprocessing failed or dataframe is empty
            return None  # Return None

        target_col = df_clean.columns[-1]  # Get the last column as target
        feat_cols = [c for c in df_clean.columns[:-1] if pd.api.types.is_numeric_dtype(df_clean[c])]  # Get numeric feature columns
        if not feat_cols:  # If no numeric features
            return None  # Return None

        return (df_clean, target_col, feat_cols)  # Return the processed data
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def handle_target_column_consistency(target_col_name, this_target, f, df_clean, config=None):
    """
    Handle target column consistency by renaming if necessary.

    :param target_col_name: Current target column name (or None)
    :param this_target: Target column name in this file
    :param f: File path for warning message
    :param df_clean: DataFrame to rename if needed
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (updated_target_col_name, updated_df_clean)
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        verbose_output(
            f"{BackgroundColors.GREEN}Checking target column consistency for: {BackgroundColors.CYAN}{f}{Style.RESET_ALL} (target: {BackgroundColors.CYAN}{this_target}{Style.RESET_ALL})...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        if target_col_name is None:  # If target column not set yet
            target_col_name = this_target  # Set it to this target
        elif this_target != target_col_name:  # If target column name differs
            print(f"{BackgroundColors.YELLOW}Warning: target column name mismatch: {f} uses {this_target} while others use {target_col_name}. Trying to proceed by renaming.{Style.RESET_ALL}")  # Print warning
            df_clean = df_clean.rename(columns={this_target: target_col_name})  # Rename the column

        return (target_col_name, df_clean)  # Return updated values
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def intersect_features(common_features, feat_cols, config=None):
    """
    Intersect features with common features set.

    :param common_features: Current common features set (or None)
    :param feat_cols: Feature columns in this file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Updated common features set
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        verbose_output(
            f"{BackgroundColors.GREEN}Intersecting features for current file...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        if common_features is None:  # If common features not set yet
            common_features = set(feat_cols)  # Initialize with this file's features
        else:  # Otherwise, intersect with existing common features
            common_features &= set(feat_cols)  # Update common features

        return common_features  # Return updated common features
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def find_common_features_and_target(processed_files, config=None):
    """
    Find common features and consistent target column from processed files.

    :param processed_files: List of (f, df_clean, target_col, feat_cols)
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (common_features, target_col_name, dfs) or (None, None, []) if invalid
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Finding common features and target column among processed files...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        dfs = []  # Initialize list to store valid dataframes
        common_features = None  # Initialize set for common features
        target_col_name = None  # Initialize target column name

        for f, df_clean, this_target, feat_cols in processed_files:  # Iterate over processed files
            target_col_name, df_clean = handle_target_column_consistency(target_col_name, this_target, f, df_clean, config=config)  # Handle target consistency
            common_features = intersect_features(common_features, feat_cols, config=config)  # Intersect features
            dfs.append((f, df_clean))  # Add the file and cleaned dataframe to the list

        if not dfs or not common_features:  # If no valid dataframes or no common features
            return (None, None, [])  # Return invalid

        if target_col_name is None:  # If no target column was found
            return (None, None, [])  # Return invalid

        common_features = sorted(list(common_features))  # Sort the common features list
        return (common_features, target_col_name, dfs)  # Return the results
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def create_reduced_dataframes(dfs, common_features, target_col_name, config=None):
    """
    Create reduced dataframes with only common features and target.

    :param dfs: List of (f, df_clean)
    :param common_features: List of common feature names
    :param target_col_name: Name of the target column
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: List of reduced dataframes
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        verbose_output(
            f"{BackgroundColors.GREEN}Creating reduced dataframes with common features and target...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        reduced_dfs = []  # Initialize list for reduced dataframes
        for idx, (f, df_clean) in enumerate(dfs):  # Iterate over valid dataframes with index for memory release
            cols_to_keep = [c for c in common_features if c in df_clean.columns]  # Get common features present in this df
            cols_to_keep.append(target_col_name)  # Add the target column
            reduced = df_clean.loc[:, cols_to_keep].copy()  # Create reduced dataframe with only common features and target
            reduced_dfs.append(reduced)  # Add to list

            dfs[idx] = (f, None)  # Release original full dataframe reference to free memory

        del dfs  # Release the dfs list reference after building all reduced dataframes
        gc.collect()  # Force garbage collection to reclaim memory from released original dataframes

        return reduced_dfs  # Return the reduced dataframes
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def combine_and_clean_dataframes(reduced_dfs, config=None):
    """
    Combine reduced dataframes, clean, and validate.

    :param reduced_dfs: List of reduced dataframes
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Combined dataframe or None if empty
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        verbose_output(
            f"{BackgroundColors.GREEN}Combining reduced dataframes and cleaning...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        combined = pd.concat(reduced_dfs, ignore_index=True)  # Concatenate all reduced dataframes

        del reduced_dfs  # Release list of reduced dataframes to free memory after concatenation
        gc.collect()  # Force garbage collection to reclaim memory from deleted reduced dataframes

        combined.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN in-place to save memory
        combined.dropna(inplace=True)  # Drop rows with NaN values in-place to avoid creating a copy

        if combined.empty:  # If combined dataframe is empty
            print(f"{BackgroundColors.RED}Combined dataset is empty after alignment and NaN removal.{Style.RESET_ALL}")  # Print error
            return None  # Return None

        return combined  # Return the combined dataframe
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def combine_dataset_files(files_list, config=None):
    """
    Load, preprocess and combine multiple dataset CSVs into a single DataFrame.

    :param files_list: List of dataset CSV file paths to combine
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Combined DataFrame with aligned features and target, or None if no compatible files found
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Combining dataset files: {BackgroundColors.CYAN}{files_list}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        processed_files = []  # Initialize list for processed file data
        for f in files_list:  # Iterate over each file in the list
            result = process_single_file(f, config=config)  # Process the single file
            if result is not None:  # If processing succeeded
                df_clean, target_col, feat_cols = result  # Unpack the result
                processed_files.append((f, df_clean, target_col, feat_cols))  # Add to processed list

        if not processed_files:  # If no files were processed successfully
            print(f"{BackgroundColors.RED}No compatible files found to combine for dataset: {files_list}.{Style.RESET_ALL}")  # Print error
            return None  # Return None

        common_features, target_col_name, dfs = find_common_features_and_target(processed_files, config=config)  # Find common features and target
        if common_features is None:  # If finding common features failed
            print(f"{BackgroundColors.RED}No valid target column found.{Style.RESET_ALL}")  # Print error
            del processed_files  # Release processed files to free memory on failure path
            gc.collect()  # Force garbage collection on failure path
            return None  # Return None

        del processed_files  # Release processed files list to free memory before creating reduced dataframes
        gc.collect()  # Force garbage collection to reclaim memory from released processed files

        reduced_dfs = create_reduced_dataframes(dfs, common_features, target_col_name, config=config)  # Create reduced dataframes
        combined = combine_and_clean_dataframes(reduced_dfs, config=config)  # Combine and clean the dataframes

        return combined, target_col_name  # Return the combined dataframe and target column name
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def extract_attack_label_from_path(file_path):
    """
    Extract attack type label from file path for multi-class classification.
    
    :param file_path: Path to the dataset CSV file
    :return: Attack type string extracted from filename or path
    """

    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Extracting attack label from path: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}"
        )  # Output the verbose message
        
        file_path_obj = Path(file_path)  # Create Path object from file path string
        filename_stem = file_path_obj.stem  # Extract filename without extension
        
        clean_stem = filename_stem.replace("_data_augmented", "").replace("_cleaned", "").replace("_processed", "")  # Remove common suffixes from stem
        
        attack_label = clean_stem  # Set attack label to cleaned filename stem
        
        verbose_output(
            f"{BackgroundColors.GREEN}Extracted attack label: {BackgroundColors.CYAN}{attack_label}{Style.RESET_ALL}"
        )  # Output extracted label
        
        return attack_label  # Return attack type label string
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def concat_files_into_multiclass_df(processed_files_with_labels, common_features_list, attack_types_set, config):
    """
    Align each processed file to the common feature set, assign attack type labels, concatenate into a single DataFrame, and clean infinite/NaN values.

    :param processed_files_with_labels: List of tuples (file_path, df_clean, target_col, feat_cols, attack_label).
    :param common_features_list: Sorted list of feature column names shared across all files.
    :param attack_types_set: Set of unique attack type strings used to build the final sorted list.
    :param config: Configuration dictionary passed through for verbose output.
    :return: Tuple of (combined_df, attack_types_list) on success, or None when the result is empty after cleaning.
    """

    combined_parts = []  # Initialize list to accumulate dataframe parts

    for idx, (f, df_clean, this_target, feat_cols, attack_label) in enumerate(processed_files_with_labels):  # Iterate over processed files with index
        df_subset = df_clean[common_features_list].copy()  # Select only common features as a copy for safe modification
        df_subset['attack_type'] = attack_label  # Add attack type column with the extracted label
        combined_parts.append(df_subset)  # Append to combined parts list

        processed_files_with_labels[idx] = (f, None, this_target, feat_cols, attack_label)  # Release original full dataframe reference to free memory

        verbose_output(
            f"{BackgroundColors.GREEN}Added {BackgroundColors.CYAN}{len(df_subset)}{BackgroundColors.GREEN} samples from {BackgroundColors.CYAN}{attack_label}{Style.RESET_ALL}",
            config=config
        )  # Output samples added message

    gc.collect()  # Force garbage collection to reclaim memory from released original dataframes

    combined_df = pd.concat(combined_parts, ignore_index=True)  # Concatenate all parts into single dataframe

    del combined_parts  # Release list of dataframe parts to free memory after concatenation
    gc.collect()  # Force garbage collection to reclaim memory from deleted parts list

    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN in-place to save memory
    combined_df.dropna(inplace=True)  # Drop rows with NaN values in-place to avoid creating a copy

    if combined_df.empty:  # If combined dataframe is empty after cleaning
        print(f"{BackgroundColors.RED}Combined multi-class dataset is empty after cleaning.{Style.RESET_ALL}")  # Print error
        return None  # Signal failure to caller

    attack_types_list = sorted(list(attack_types_set))  # Convert attack types set to sorted list
    print(
        f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Multi-class dataset created: {BackgroundColors.CYAN}{len(combined_df)} samples, {len(common_features_list)} features, {len(attack_types_list)} classes{Style.RESET_ALL}"
    )  # Print summary of combined dataset
    return combined_df, attack_types_list  # Return combined dataframe and attack types list


def compute_common_features_across_files(processed_files_with_labels, config):
    """
    Compute the intersection of feature columns across all processed files and determine the common target column name.

    :param processed_files_with_labels: List of tuples (file_path, df_clean, target_col, feat_cols, attack_label).
    :param config: Configuration dictionary passed through for verbose output.
    :return: Tuple of (common_features_list, target_col_name) on success, or None when no common features exist.
    """

    common_features = None  # Initialize common features set
    target_col_name = None  # Initialize target column name

    for f, df_clean, this_target, feat_cols, attack_label in processed_files_with_labels:  # Iterate over processed files
        if target_col_name is None:  # If target not yet set
            target_col_name = this_target  # Set target column name

        if common_features is None:  # If common features not yet initialized
            common_features = set(feat_cols)  # Initialize with first file's features
        else:  # If common features already initialized
            common_features = common_features.intersection(set(feat_cols))  # Intersect with current file's features

    if not common_features:  # If no common features found
        print(f"{BackgroundColors.RED}No common features found across files for multi-class combination.{Style.RESET_ALL}")  # Print error
        return None  # Signal failure to caller

    common_features_list = sorted(list(common_features))  # Convert to sorted list
    verbose_output(
        f"{BackgroundColors.GREEN}Common features for multi-class: {BackgroundColors.CYAN}{len(common_features_list)} features{Style.RESET_ALL}",
        config=config
    )  # Output common features count
    return common_features_list, target_col_name  # Return common features and target column name


def process_files_and_extract_labels(files_list, config):
    """
    Process each dataset file, extract its attack type label, and accumulate results into a list suitable for multi-class combination.

    :param files_list: List of dataset CSV file paths to process.
    :param config: Configuration dictionary passed through to processing helpers.
    :return: Tuple of (processed_files_with_labels, attack_types_set) on success, or None when no files could be processed.
    """

    processed_files_with_labels = []  # Initialize list for processed file data with attack labels
    attack_types_set = set()  # Initialize set to track unique attack types

    for f in files_list:  # Iterate over each file in the list
        result = process_single_file(f, config=config)  # Process the single file
        if result is not None:  # If processing succeeded
            df_clean, target_col, feat_cols = result  # Unpack the result
            attack_label = extract_attack_label_from_path(f)  # Extract attack type from filename
            attack_types_set.add(attack_label)  # Add attack type to set
            processed_files_with_labels.append((f, df_clean, target_col, feat_cols, attack_label))  # Add to processed list with label
        else:  # If processing failed
            verbose_output(
                f"{BackgroundColors.YELLOW}Skipping file due to processing failure: {BackgroundColors.CYAN}{f}{Style.RESET_ALL}",
                config=config
            )  # Output warning message

    if not processed_files_with_labels:  # If no files were processed successfully
        print(f"{BackgroundColors.RED}No compatible files found to combine for multi-class dataset.{Style.RESET_ALL}")  # Print error message
        return None  # Signal failure to caller

    print(
        f"{BackgroundColors.GREEN}Found {BackgroundColors.CYAN}{len(attack_types_set)}{BackgroundColors.GREEN} unique attack types for multi-class: {BackgroundColors.CYAN}{sorted(attack_types_set)}{Style.RESET_ALL}"
    )  # Print attack types summary
    return processed_files_with_labels, attack_types_set  # Return processed data and attack types set


def combine_files_for_multiclass(files_list, config=None):
    """
    Combine multiple dataset files into a single multi-class dataset.
    Each file represents a different attack type and becomes a unique class label.
    
    :param files_list: List of dataset CSV file paths to combine for multi-class
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (combined_df, attack_types_list, target_col_name) or (None, None, None) if failed
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        verbose_output(
            f"{BackgroundColors.GREEN}Combining files for multi-class classification: {BackgroundColors.CYAN}{len(files_list)} files{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message
        
        if not files_list:  # If files list is empty
            print(f"{BackgroundColors.RED}No files provided for multi-class combination.{Style.RESET_ALL}")  # Print error message
            return (None, None, None)  # Return None tuple
        
        process_result = process_files_and_extract_labels(files_list, config)  # Process files and extract attack labels
        if process_result is None:  # If processing failed
            return (None, None, None)  # Return failure tuple
        processed_files_with_labels, attack_types_set = process_result  # Unpack processed data and attack types set
        
        feature_result = compute_common_features_across_files(processed_files_with_labels, config)  # Compute common features across all files
        if feature_result is None:  # If no common features found
            del processed_files_with_labels  # Release processed files to free memory on failure path
            gc.collect()  # Force garbage collection on failure path
            return (None, None, None)  # Return failure tuple
        common_features_list, target_col_name = feature_result  # Unpack common features and target column name
        
        concat_result = concat_files_into_multiclass_df(processed_files_with_labels, common_features_list, attack_types_set, config)  # Concatenate files into multiclass dataframe

        del processed_files_with_labels  # Release individual dataframes to free memory after concatenation
        gc.collect()  # Force garbage collection to reclaim memory from released individual dataframes

        if concat_result is None:  # If concatenation failed
            return (None, None, None)  # Return failure tuple
        
        if not isinstance(concat_result, (list, tuple)) or len(concat_result) != 2:  # Verify concat_result is iterable and has exactly two items
            print(f"{BackgroundColors.RED}Unexpected return from concat_files_into_multiclass_df: {type(concat_result)}{Style.RESET_ALL}")  # Log unexpected return type for diagnosis
            err_val = ValueError("concat_files_into_multiclass_df returned unexpected format")  # Create ValueError to describe unexpected format
            send_exception_via_telegram(type(err_val), err_val, err_val.__traceback__)  # Notify via Telegram about unexpected format
            return (None, None, None)  # Return failure tuple due to unexpected format
        
        result = tuple(concat_result)  # Convert verified concat_result into tuple for safe unpacking
        combined_df, attack_types_list = result  # Unpack combined dataframe and attack types list after casting
        return (combined_df, attack_types_list, 'attack_type')  # Return combined dataframe, attack types list, and target column name
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def find_data_augmentation_file(original_file_path, config=None):
    """
    Find the corresponding data augmentation file for an original CSV file.
    Matches wgangp.py naming: <parent>/Data_Augmentation/<stem>_data_augmented<suffix>.

    :param original_file_path: Path to the original CSV file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Path to the augmented file if it exists, None otherwise
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Looking for data augmentation file for: {BackgroundColors.CYAN}{original_file_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        data_augmentation_suffix = config.get("execution", {}).get("results_suffix", None)  # Try execution-level suffix first
        if not data_augmentation_suffix:  # If not set at execution level
            data_augmentation_suffix = config.get("stacking", {}).get("data_augmentation_suffix", "_data_augmented")  # Fallback to stacking config
        original_path = Path(original_file_path)  # Create Path object from the original file path
        data_aug_dir = config.get("paths", {}).get("data_augmentation_dir", "Data_Augmentation")  # Use configured data augmentation base directory name
        data_aug_sample_dir = config.get("paths", {}).get("data_augmentation_sample_dir", "Samples")  # Use configured augmented samples subdirectory name
        augmented_dir = original_path.parent / data_aug_dir / data_aug_sample_dir  # Build data augmentation samples subdirectory path using config
        augmentation_format = config.get("stacking", {}).get("augmentation_file_format", "csv")  # Read configured augmentation file format
        augmentation_extension = resolve_format_extension(augmentation_format)  # Resolve format string to file extension
        augmented_filename = f"{original_path.stem}{data_augmentation_suffix}{augmentation_extension}"  # Build augmented filename with configured extension
        augmented_file = augmented_dir / augmented_filename  # Construct the full augmented file path

        if augmented_file.exists():  # If the expected augmented file exists at the constructed path
            verbose_output(
                f"{BackgroundColors.GREEN}Found augmented file: {BackgroundColors.CYAN}{augmented_file}{Style.RESET_ALL}",
                config=config
            )  # Output success message with the found path
            return str(augmented_file)  # Return the augmented file path as a string

        fallback_candidates = list(augmented_dir.glob(f"{original_path.stem}*{data_augmentation_suffix}*"))  # Search for any file matching stem+suffix pattern as fallback
        if fallback_candidates:  # If any fallback candidates were found via glob search
            verbose_output(
                f"{BackgroundColors.GREEN}Found augmented file via fallback glob: {BackgroundColors.CYAN}{fallback_candidates[0]}{Style.RESET_ALL}",
                config=config
            )  # Output fallback match message
            return str(fallback_candidates[0])  # Return the first matching fallback candidate

        verbose_output(
            f"{BackgroundColors.YELLOW}No augmented file found for: {BackgroundColors.CYAN}{original_file_path}{BackgroundColors.YELLOW}. Expected: {BackgroundColors.CYAN}{augmented_file}{Style.RESET_ALL}",
            config=config
        )  # Output warning with expected path for debugging
        return None  # Return None when no augmented file is found
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_augmented_files_for_multiclass(original_files_list, config=None):
    """
    Load augmented data files corresponding to original files for multi-class mode.
    
    :param original_files_list: List of original dataset CSV file paths
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: List of augmented file paths (None entries where augmented file not found)
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG as fallback

        verbose_output(  # Emit verbose startup message for multi-class augmentation loading
            f"{BackgroundColors.GREEN}Loading augmented files for multi-class mode: {BackgroundColors.CYAN}{len(original_files_list)} files{Style.RESET_ALL}",
            config=config,
        )

        augmented_files = []  # Prepare list to preserve alignment with originals (None = missing)
        found_count = 0  # Counter for how many augmented files were found

        for original_file in original_files_list:  # Iterate originals to locate augmented counterparts
            augmented_file = find_data_augmentation_file(original_file, config=config)  # Locate augmented file path or None
            if augmented_file is not None:  # If an augmented file exists for this original
                augmented_files.append(augmented_file)  # Append the found augmented file path
                found_count += 1  # Increment found counter
            else:  # If no augmented file exists for this original
                verbose_output(  # Emit a per-file informative warning to verbose output
                    f"{BackgroundColors.YELLOW}No augmented file found for: {BackgroundColors.CYAN}{original_file}{Style.RESET_ALL}",
                    config=config,
                )
                augmented_files.append(None)  # Preserve index alignment with None placeholder

        if found_count == 0:  # If none were found across all originals
            print(  # Print a consolidated warning about missing augmented files
                f"{BackgroundColors.YELLOW}No augmented files found for any original files in multi-class mode.{Style.RESET_ALL}"
            )
        else:  # If at least one augmented file was found
            print(  # Print a consolidated summary of found augmented files
                f"{BackgroundColors.GREEN}Found {BackgroundColors.CYAN}{found_count}/{len(original_files_list)}{BackgroundColors.GREEN} augmented files for multi-class mode.{Style.RESET_ALL}"
            )

        return augmented_files  # Return the list preserving None placeholders for missing entries
    except Exception as e:  # On any exception, ensure logging and notification then re-raise
        print(str(e))  # Print the exception string to terminal logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send the exception via Telegram with traceback
        raise  # Re-raise the original exception to preserve semantics


def merge_original_and_augmented(original_df, augmented_df, config=None):
    """
    Merge original and augmented dataframes by concatenating them.

    :param original_df: Original DataFrame
    :param augmented_df: Augmented DataFrame
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Merged DataFrame
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Merging original ({len(original_df)} rows) and augmented ({len(augmented_df)} rows) data{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        merged_df = pd.concat([original_df, augmented_df], ignore_index=True)  # Concatenate dataframes
        
        verbose_output(
            f"{BackgroundColors.GREEN}Merged dataset has {BackgroundColors.CYAN}{len(merged_df)}{BackgroundColors.GREEN} rows{Style.RESET_ALL}",
            config=config
        )  # Output the result

        return merged_df  # Return merged dataframe
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_class_metrics(y_true, y_pred):
    """
    Extract per-class and global classification metrics without recomputing predictions.

    :param y_true: True labels (array-like)
    :param y_pred: Predicted labels (array-like)
    :return: dict with structure specified by requirements
    """
    
    try:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)  # Generate report dict

        if not isinstance(report, dict):  # Ensure report has mapping semantics
            raise TypeError("classification_report did not return a dict as expected")  # Defensive type verification

        per_class = {}  # Prepare per-class metrics mapping
        for k, v in report.items():  # Iterate over report items (class labels and metrics)

            if k in ("accuracy", "macro avg", "weighted avg", "micro avg"):  # Skip aggregate keys
                continue  # Move to next item

            per_class[k] = {  # Populate per-class metric dict converting values to float
                "precision": float(v.get("precision", 0.0)),
                "recall": float(v.get("recall", 0.0)),
                "f1": float(v.get("f1-score", v.get("f1", 0.0))),
            }

        global_metrics = {  # Compute deterministic global metrics
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }

        return {"per_class": per_class, "global": global_metrics}  # Return structured metrics
    except Exception as e:  # On exception, log and notify then re-raise
        print(str(e))  # Print exception to terminal
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception details via Telegram
        raise  # Re-raise the exception to preserve original control flow
        return {"per_class": {}, "global": {}}


def ensure_figure_min_4k_and_save(fig=None, path=None, dpi=None, **kwargs):
    """
    Ensure a Matplotlib figure meets 4k minimum pixel dimensions and save it.

    :param fig: Matplotlib Figure instance or None to use current figure.
    :param path: Path where the image will be saved.
    :param dpi: DPI to use for saving; preserved if provided.
    :return: None
    """

    fig = fig or plt.gcf()  # Use current figure when none provided

    effective_dpi = dpi if dpi is not None else fig.get_dpi()  # Determine DPI for pixel computation

    width_inch, height_inch = fig.get_size_inches()  # Get current figure size in inches

    if width_inch * effective_dpi < 3840 or height_inch * effective_dpi < 2160:  # Verify if either dimension is below 4k
        new_w = max(width_inch, 3840.0 / effective_dpi)  # Compute required width in inches to reach 4k at effective DPI
        new_h = max(height_inch, 2160.0 / effective_dpi)  # Compute required height in inches to reach 4k at effective DPI
        fig.set_size_inches(new_w, new_h)  # Resize figure in inches while preserving DPI

    save_kwargs = dict(kwargs)  # Prepare kwargs for saving call

    if dpi is not None:  # Verify whether caller explicitly provided DPI
        save_kwargs["dpi"] = dpi  # Preserve caller DPI in save call

    resolved_fig = fig  # Ensure we close the exact figure used for saving
    if path is None:  # Verify that a valid path argument was provided
        raise ValueError("path must be provided to save the figure")  # Raise explicit error when path is missing to avoid passing None to savefig
    try:  # Save the figure to disk using provided kwargs
        resolved_fig.savefig(path, **save_kwargs)  # Save the figure to disk using provided kwargs
    finally:  # Ensure we close the figure to free memory regardless of save success
        plt.close(resolved_fig)  # Close the figure to free memory


def plot_per_class_metric(metric_dict: dict, metric_name: str, output_path: str, dpi: int, fmt: str) -> None:
    """
    Plot a per-class bar chart.

    :param metric_dict: mapping class_label -> {precision, recall, f1}
    :param metric_name: one of 'precision','recall','f1'
    :param output_path: full path to save file
    :param dpi: dpi integer > 0
    :param fmt: file format string
    """

    try:
        labels = sorted(metric_dict.keys(), key=lambda x: str(x))
        values = [float(metric_dict[l].get(metric_name, 0.0)) for l in labels]

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.4), 6))
        bars = ax.bar(labels, values, color="tab:blue", alpha=0.8)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(metric_name.capitalize())
        ax.set_xlabel("Class")
        ax.set_title(f"Stacking - {metric_name.capitalize()} per Class")
        ax.grid(axis="y", alpha=0.25)

        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2.0, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        ensure_figure_min_4k_and_save(fig=fig, path=output_path, dpi=dpi, format=fmt, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        try:
            plt.close()
        except Exception:
            pass
        raise


def plot_global_metrics(global_metrics: dict, output_path: str, dpi: int, fmt: str) -> None:
    """
    Plot a deterministic ordered global metrics bar chart.
    Order: Accuracy, Macro F1, Weighted F1, Macro Precision, Macro Recall
    
    :param global_metrics: dict with keys like 'accuracy', 'macro_f1', etc.
    :param output_path: full path to save file
    :param dpi: dpi integer > 0
    :param fmt: file format string
    :return: None (saves plot to file)
    """
    
    try:
        order = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
        labels = ["Accuracy", "Macro F1", "Weighted F1", "Macro Precision", "Macro Recall"]
        values = [float(global_metrics.get(k, 0.0)) for k in order]

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(labels, values, color="tab:green", alpha=0.85)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title("Stacking - Global Metrics")
        ax.grid(axis="y", alpha=0.25)

        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2.0, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        ensure_figure_min_4k_and_save(fig=fig, path=output_path, dpi=dpi, format=fmt, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        try:
            plt.close()
        except Exception:
            pass
        raise


def generate_and_save_metric_plots(y_true, y_pred, stacking_config: dict, resolved_results_dir: str) -> None:
    """
    Orchestrate extraction and saving of all required plots into
    <Feature_Analysis>/<stacking.results_dir>/Plots/

    :param y_true: true labels
    :param y_pred: predicted labels (must be the same predictions used to compute metrics)
    :param stacking_config: config dict for stacking (may contain 'plots')
    :param resolved_results_dir: absolute or project-relative path to stacking results directory
    """
    
    try:
        plots_cfg = stacking_config.get("plots", {}) if stacking_config is not None else {}
        enabled = plots_cfg.get("enabled", True)
        if not enabled:
            return

        dpi = plots_cfg.get("dpi", 1000)
        fmt = plots_cfg.get("format", "png")

        if not isinstance(dpi, int) or dpi <= 0:
            raise ValueError("plots.dpi must be an integer > 0")
        if fmt not in ("png", "jpg", "pdf", "svg"):
            raise ValueError("plots.format must be one of: png, jpg, pdf, svg")

        plots_dir = os.path.join(os.path.abspath(resolved_results_dir), config.get("paths", {}).get("plots_subdir", "Plots"))
        print(f"{BackgroundColors.GREEN}Saving metric plots to: {BackgroundColors.CYAN}{plots_dir}{Style.RESET_ALL}")

        validate_output_path(os.path.abspath(resolved_results_dir), os.path.abspath(plots_dir))
        os.makedirs(plots_dir, exist_ok=True)

        metrics = extract_class_metrics(y_true, y_pred)

        per = metrics.get("per_class", {})

        per_class_f1_path = os.path.join(plots_dir, f"per_class_f1.{fmt}")
        per_class_prec_path = os.path.join(plots_dir, f"per_class_precision.{fmt}")
        per_class_rec_path = os.path.join(plots_dir, f"per_class_recall.{fmt}")

        plot_per_class_metric(per, "f1", per_class_f1_path, dpi, fmt)
        plot_per_class_metric(per, "precision", per_class_prec_path, dpi, fmt)
        plot_per_class_metric(per, "recall", per_class_rec_path, dpi, fmt)

        global_path = os.path.join(plots_dir, f"global_metrics.{fmt}")
        plot_global_metrics(metrics.get("global", {}), global_path, dpi, fmt)

        verbose_output(f"[STACKING][PLOTS] Saved: {per_class_f1_path}, {per_class_prec_path}, {per_class_rec_path}, {global_path}")
    except Exception as e:
        verbose_output(f"{BackgroundColors.YELLOW}Plot generation failed: {e}{Style.RESET_ALL}")
        send_exception_via_telegram(type(e), e, e.__traceback__)
        return


def calculate_metric_improvement(original_value, augmented_value):
    """
    Calculate percentage improvement of a metric.

    :param original_value: Original metric value
    :param augmented_value: Augmented metric value
    :return: Percentage improvement (positive = better, negative = worse)
    """

    try:
        if original_value == 0:  # Avoid division by zero
            return 0.0 if augmented_value == 0 else float('inf')  # Return 0 if both are 0, inf if only original is 0
        
        improvement = ((augmented_value - original_value) / original_value) * 100  # Calculate percentage improvement
        return improvement  # Return improvement
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def generate_experiment_id(file_path, experiment_mode, augmentation_ratio=None):
    """
    Generates a unique experiment identifier for traceability in CSV results.

    :param file_path: Path to the dataset file being processed
    :param experiment_mode: Experiment mode string (e.g., 'original_only' or 'original_plus_augmented')
    :param augmentation_ratio: Augmentation ratio float (e.g., 0.10) or None for original-only mode
    :return: String experiment identifier combining timestamp, filename, mode and ratio
    """

    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Create a timestamp string for uniqueness
        file_stem = Path(file_path).stem  # Extract the filename stem without extension
        ratio_tag = f"_ratio{int(augmentation_ratio * 100)}" if augmentation_ratio is not None else ""  # Build ratio tag string or empty
        experiment_id = f"{timestamp}_{file_stem}_{experiment_mode}{ratio_tag}"  # Concatenate all parts into unique identifier

        return experiment_id  # Return the generated experiment identifier
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def validate_augmented_dataframe(original_df, augmented_df, file_path):
    """
    Validates that the augmented DataFrame is compatible with the original for merging.

    :param original_df: Original cleaned DataFrame to compare against
    :param augmented_df: Augmented DataFrame to validate
    :param file_path: File path string for error message context
    :return: True if augmented data is valid and compatible, False otherwise
    """

    try:
        if augmented_df is None or augmented_df.empty:  # Verify if augmented DataFrame is None or contains no rows
            print(
                f"{BackgroundColors.YELLOW}Warning: Augmented DataFrame is empty for {BackgroundColors.CYAN}{file_path}{BackgroundColors.YELLOW}. Skipping.{Style.RESET_ALL}"
            )  # Print warning about empty augmented data
            return False  # Return False for empty augmented data

        original_cols = set(original_df.columns)  # Get the set of column names from the original DataFrame
        augmented_cols = set(augmented_df.columns)  # Get the set of column names from the augmented DataFrame
        missing_cols = original_cols - augmented_cols  # Compute columns present in original but missing in augmented

        if missing_cols:  # If there are columns missing from the augmented DataFrame
            print(
                f"{BackgroundColors.YELLOW}Warning: Augmented data for {BackgroundColors.CYAN}{file_path}{BackgroundColors.YELLOW} is missing columns: {BackgroundColors.CYAN}{missing_cols}{BackgroundColors.YELLOW}. Skipping.{Style.RESET_ALL}"
            )  # Print warning listing the missing column names
            return False  # Return False due to column mismatch

        original_dtypes = original_df.select_dtypes(include=np.number).columns.tolist()  # Get list of numeric columns in original
        augmented_dtypes = augmented_df.select_dtypes(include=np.number).columns.tolist()  # Get list of numeric columns in augmented
        numeric_overlap = set(original_dtypes) & set(augmented_dtypes)  # Compute intersection of numeric columns

        if len(numeric_overlap) < 2:  # Verify if there are at least 2 overlapping numeric columns (features + target)
            print(
                f"{BackgroundColors.YELLOW}Warning: Augmented data for {BackgroundColors.CYAN}{file_path}{BackgroundColors.YELLOW} has insufficient numeric column overlap ({len(numeric_overlap)}). Skipping.{Style.RESET_ALL}"
            )  # Print warning about insufficient numeric overlap
            return False  # Return False due to insufficient numeric columns

        verbose_output(
            f"{BackgroundColors.GREEN}Augmented data validation passed for {BackgroundColors.CYAN}{file_path}{BackgroundColors.GREEN}: {len(augmented_df)} rows, {len(augmented_cols)} columns{Style.RESET_ALL}"
        )  # Output verbose message confirming validation success

        return True  # Return True indicating augmented data is valid and compatible
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def sample_augmented_by_ratio(augmented_df, original_df, ratio):
    """
    Samples rows from the augmented DataFrame proportional to the original dataset size.

    :param augmented_df: Full augmented DataFrame to sample from
    :param original_df: Original DataFrame used to determine sample count
    :param ratio: Float ratio (e.g., 0.10 means 10% of original size)
    :return: Sampled DataFrame with at most ratio * len(original_df) rows, or None on failure
    """

    try:
        n_original = len(original_df)  # Get the number of rows in the original dataset
        n_requested = max(1, int(round(ratio * n_original)))  # Calculate requested sample size capped at minimum 1 row
        n_available = len(augmented_df)  # Get the total number of rows available in augmented data

        if n_available == 0:  # Verify if augmented DataFrame has zero rows
            print(
                f"{BackgroundColors.YELLOW}Warning: Augmented DataFrame is empty. Cannot sample at ratio {ratio}.{Style.RESET_ALL}"
            )  # Print warning about empty augmented source
            return None  # Return None for empty augmented data

        n_sample = min(n_requested, n_available)  # Cap the sample size at the available number of augmented rows

        if n_sample < n_requested:  # Log a warning if capping occurred (fewer augmented rows than requested)
            verbose_output(
                f"{BackgroundColors.YELLOW}Augmented data has only {n_available} rows; requested {n_requested} (ratio={ratio}). Using all {n_available}.{Style.RESET_ALL}"
            )  # Warn that fewer rows than requested are available

        sampled_df = augmented_df.sample(n=n_sample, random_state=42, replace=False)  # Randomly sample n_sample rows with fixed seed for reproducibility

        verbose_output(
            f"{BackgroundColors.GREEN}Sampled {BackgroundColors.CYAN}{n_sample}{BackgroundColors.GREEN} augmented rows at ratio {BackgroundColors.CYAN}{ratio}{BackgroundColors.GREEN} (original has {n_original} rows){Style.RESET_ALL}"
        )  # Output verbose message confirming sampling details

        return sampled_df  # Return the sampled augmented DataFrame
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_tsne_output_directory(original_file_path, augmented_file_path):
    """
    Build output directory path for t-SNE plots preserving nested dataset structure.

    :param original_file_path: Path to original dataset file
    :param augmented_file_path: Path to augmented dataset file
    :return: Path object for t-SNE output directory
    """

    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Building t-SNE output directory for: {BackgroundColors.CYAN}{original_file_path}{Style.RESET_ALL}"
        )  # Output verbose message for directory creation

        original_path = Path(original_file_path)  # Create Path object for original file
        augmented_path = Path(augmented_file_path)  # Create Path object for augmented file

        datasets_keyword = "Datasets"  # Standard directory name in project structure
        relative_parts = []  # List to accumulate relative path components
        found_datasets = False  # Flag to track if Datasets directory was found

        for part in original_path.parts:  # Iterate through path components
            if found_datasets and part != original_path.name:  # After Datasets, before filename
                relative_parts.append(part)  # Add intermediate directories to relative path
            if part == datasets_keyword:  # Found the Datasets directory
                found_datasets = True  # Set flag to start collecting relative parts

        augmented_parent = augmented_path.parent  # Get parent directory of augmented file
        tsne_base = augmented_parent / "tsne_plots"  # Base directory for all t-SNE plots

        if relative_parts:  # If nested structure exists
            tsne_dir = tsne_base / Path(*relative_parts) / original_path.stem  # Preserve nested path
        else:  # Flat structure
            tsne_dir = tsne_base / original_path.stem  # Use filename stem only

        os.makedirs(tsne_dir, exist_ok=True)  # Create directory structure if it doesn't exist

        verbose_output(
            f"{BackgroundColors.GREEN}Created t-SNE directory: {BackgroundColors.CYAN}{tsne_dir}{Style.RESET_ALL}"
        )  # Output confirmation message

        return tsne_dir  # Return the output directory path
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def combine_and_label_augmentation_data(original_df, augmented_df=None, label_col=None):
    """
    Combine original and augmented data with source labels for t-SNE visualization.

    :param original_df: DataFrame with original data
    :param augmented_df: DataFrame with augmented data (None for original-only)
    :param label_col: Name of the label/class column
    :return: Combined DataFrame with composite labels for visualization
    """

    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Combining and labeling data for t-SNE visualization...{Style.RESET_ALL}"
        )  # Output verbose message for data combination

        if label_col is None:  # No label column specified
            label_col = original_df.columns[-1]  # Use last column as label

        df_orig = original_df.copy()  # Copy original DataFrame to avoid modifying input

        if label_col in df_orig.columns:  # If label column exists
            df_orig['tsne_label'] = df_orig[label_col].astype(str) + "_original"  # Create composite label
        else:  # No label column found
            df_orig['tsne_label'] = "original"  # Use simple source label

        if augmented_df is not None:  # If augmented data provided
            df_aug = augmented_df.copy()  # Copy augmented DataFrame to avoid modifying input

            if label_col in df_aug.columns:  # If label column exists
                df_aug['tsne_label'] = df_aug[label_col].astype(str) + "_augmented"  # Create composite label
            else:  # No label column found
                df_aug['tsne_label'] = "augmented"  # Use simple source label

            combined_df = pd.concat([df_orig, df_aug], ignore_index=True)  # Concatenate DataFrames
        else:  # Original only
            combined_df = df_orig  # Use original DataFrame only

        return combined_df  # Return combined DataFrame with composite labels
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def prepare_numeric_features_for_tsne(df, exclude_col='tsne_label'):
    """
    Extract and prepare numeric features from DataFrame for t-SNE.

    :param df: DataFrame with mixed features
    :param exclude_col: Column name to exclude from numeric extraction
    :return: Tuple (numeric_array, labels_array, success_flag)
    """

    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Preparing numeric features for t-SNE...{Style.RESET_ALL}"
        )  # Output verbose message for feature preparation

        if exclude_col in df.columns:  # If label column exists
            labels = df[exclude_col].values  # Extract labels as numpy array
            df_work = df.drop(columns=[exclude_col])  # Remove label column for numeric extraction
        else:  # No label column
            labels = np.array(['unknown'] * len(df))  # Create default labels
            df_work = df.copy()  # Use full DataFrame

        numeric_df = df_work.select_dtypes(include=np.number)  # Select only numeric columns

        if numeric_df.empty:  # No numeric columns found
            print(
                f"{BackgroundColors.YELLOW}Warning: No numeric features found for t-SNE generation.{Style.RESET_ALL}"
            )  # Print warning message
            return (None, None, False)  # Return failure tuple

        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
        numeric_df = numeric_df.fillna(numeric_df.median())  # Fill NaN with column median
        numeric_df = numeric_df.fillna(0)  # Fill remaining NaN with zero

        if numeric_df.shape[0] == 0 or numeric_df.shape[1] == 0:  # Empty result after cleaning
            print(
                f"{BackgroundColors.YELLOW}Warning: No valid numeric data remaining after cleaning.{Style.RESET_ALL}"
            )  # Print warning message
            return (None, None, False)  # Return failure tuple

        X = numeric_df.values  # Extract values as numpy array

        try:  # Attempt feature scaling
            scaler = StandardScaler()  # Initialize standard scaler
            X_scaled = scaler.fit_transform(X)  # Scale features to zero mean and unit variance
        except Exception as e:  # Scaling failed
            print(
                f"{BackgroundColors.YELLOW}Warning: Feature scaling failed: {e}. Using unscaled data.{Style.RESET_ALL}"
            )  # Print warning message
            X_scaled = X  # Use unscaled data as fallback

        return (X_scaled, labels, True)  # Return scaled features, labels, and success flag
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def compute_and_save_tsne_plot(X_scaled, labels, output_path, title, perplexity=30, random_state=42):
    """
    Compute t-SNE embedding and save visualization plot.

    :param X_scaled: Scaled numeric feature array
    :param labels: Array of labels for coloring
    :param output_path: Full path for saving the plot file
    :param title: Title for the plot
    :param perplexity: t-SNE perplexity parameter
    :param random_state: Random seed for reproducibility
    :return: True if successful, False otherwise
    """

    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Computing t-SNE embedding and saving plot...{Style.RESET_ALL}"
        )  # Output verbose message for t-SNE computation

        max_perplexity = (X_scaled.shape[0] - 1) // 3  # Maximum valid perplexity
        actual_perplexity = min(perplexity, max_perplexity)  # Use minimum of requested and maximum

        if actual_perplexity < 5:  # Perplexity too small for meaningful results
            print(
                f"{BackgroundColors.YELLOW}Warning: Sample size too small for t-SNE (n={X_scaled.shape[0]}).{Style.RESET_ALL}"
            )  # Print warning message
            return False  # Return failure flag

        tsne = TSNE(
            n_components=2,  # 2D embedding for visualization
            perplexity=actual_perplexity,  # Adjusted perplexity parameter
            random_state=random_state,  # Random seed for reproducibility
            n_iter=1000  # Number of iterations
        )  # Create t-SNE object

        X_embedded = tsne.fit_transform(X_scaled)  # Compute 2D embedding

        plt.figure(figsize=(12, 10))  # Create figure with specified size

        unique_labels = np.unique(labels)  # Extract unique label values
        n_labels = len(unique_labels)  # Count unique labels

        cmap = plt.cm.get_cmap("rainbow")  # Get rainbow colormap for distinct colors
        colors = cmap(np.linspace(0, 1, n_labels))  # Generate distinct colors from colormap

        for idx, label in enumerate(unique_labels):  # Iterate over unique labels
            mask = labels == label  # Create boolean mask for current label
            plt.scatter(
                X_embedded[mask, 0],  # X coordinates for current class
                X_embedded[mask, 1],  # Y coordinates for current class
                c=[colors[idx]],  # Color for current class
                label=label,  # Legend label
                alpha=0.6,  # Transparency
                edgecolors='k',  # Black edge color
                linewidth=0.5,  # Edge line width
                s=50  # Marker size
            )  # Plot scatter points for current class

        plt.title(title, fontsize=16, fontweight='bold')  # Set plot title
        plt.xlabel('t-SNE Component 1', fontsize=12)  # Set x-axis label
        plt.ylabel('t-SNE Component 2', fontsize=12)  # Set y-axis label
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)  # Add legend outside plot
        plt.grid(True, alpha=0.3)  # Add grid with transparency
        plt.tight_layout()  # Adjust layout to prevent label cutoff

        ensure_figure_min_4k_and_save(fig=plt.gcf(), path=output_path, dpi=300, bbox_inches='tight')  # Save figure with high resolution
        plt.close()  # Close figure to free memory

        print(
            f"{BackgroundColors.GREEN}t-SNE plot saved to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
        )  # Print success message

        return True  # Return success flag

    except Exception as e:  # t-SNE computation or plotting failed
        print(
            f"{BackgroundColors.RED}Error generating t-SNE plot: {e}{Style.RESET_ALL}"
        )  # Print error message
        send_exception_via_telegram(type(e), e, e.__traceback__)
        return False  # Return failure flag


def generate_augmentation_tsne_visualization(original_file, original_df, augmented_df=None, augmentation_ratio=None, experiment_mode="original_only"):
    """
    Generate t-SNE visualization for data augmentation experiment.

    :param original_file: Path to original dataset file
    :param original_df: DataFrame with original data
    :param augmented_df: DataFrame with augmented data (None for original-only)
    :param augmentation_ratio: Augmentation ratio (e.g., 0.50 for 50%)
    :param experiment_mode: Experiment mode string
    :return: None
    """

    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Generating t-SNE visualization for augmentation experiment...{Style.RESET_ALL}"
        )  # Output verbose message for t-SNE generation

        augmented_file = find_data_augmentation_file(original_file)  # Locate augmented data file
        if augmented_file is None:  # No augmented file found
            print(
                f"{BackgroundColors.YELLOW}Warning: Cannot generate t-SNE - augmented file path not found.{Style.RESET_ALL}"
            )  # Print warning message
            return  # Exit function early

        stacking_output_dir = get_stacking_output_dir(original_file, CONFIG)
        tsne_output_dir = Path(stacking_output_dir) / "Plots" / "tsne_plots" / Path(original_file).stem
        tsne_output_dir.mkdir(parents=True, exist_ok=True)

        combined_df = combine_and_label_augmentation_data(original_df, augmented_df)  # Prepare labeled data

        X_scaled, labels, success = prepare_numeric_features_for_tsne(combined_df, exclude_col='tsne_label')  # Extract and scale features

        if not success:  # Feature preparation failed
            print(
                f"{BackgroundColors.YELLOW}Warning: Skipping t-SNE generation due to feature preparation failure.{Style.RESET_ALL}"
            )  # Print warning message
            return  # Exit function early

        file_stem = Path(original_file).stem  # Extract filename without extension

        if experiment_mode == "original_only":  # Original-only experiment
            plot_filename = f"{file_stem}_original_only_tsne.png"  # Filename for original-only plot
            plot_title = f"t-SNE: {file_stem} (Original Only)"  # Title for original-only plot
        else:  # Original + augmented experiment
            ratio_pct = int(augmentation_ratio * 100) if augmentation_ratio else 0  # Convert ratio to percentage
            plot_filename = f"{file_stem}_augmented_{ratio_pct}pct_tsne.png"  # Filename for augmented plot
            plot_title = f"t-SNE: {file_stem} (Original + {ratio_pct}% Augmented)"  # Title for augmented plot

        output_path = tsne_output_dir / plot_filename  # Build full output path

        compute_and_save_tsne_plot(X_scaled, labels, str(output_path), plot_title)  # Generate and save visualization

        send_telegram_message(
            TELEGRAM_BOT, f"Generated t-SNE plot: {file_stem} ({experiment_mode}, ratio={augmentation_ratio})"
        )  # Send notification via Telegram
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def zebra(row):
    """
    Apply zebra-striping style to a DataFrame row for visual clarity in exported images.
    
    :param row: pandas Series representing a DataFrame row
    :return: List of CSS styles for each column in the row, alternating background colors
    """
    
    return ["background-color: #ffffff" if i % 2 == 0 else "background-color: #f2f2f2" for i in range(len(row))]  # Return CSS list per column


def apply_zebra_style(df):
    """
    Apply zebra-striping style to a pandas DataFrame using Styler.

    :param df: pandas DataFrame to style
    :return: pandas.io.formats.style.Styler styled object ready for export
    """

    try:

        styled = df.style.apply(zebra, axis=1)  # Create Styler with zebra striping applied
        return styled  # Return styled object
    except Exception as e:  # If styling fails
        raise  # Propagate exception without swallowing


def export_dataframe_image(styled_df, output_path):
    """
    Export a pandas Styler object to a PNG file using dataframe_image.

    :param styled_df: pandas Styler object to export
    :param output_path: Path to PNG file to write
    :return: None
    """

    try:
        out_path = Path(output_path)  # Create Path object for output
        parent = out_path.parent  # Get parent directory
        if not parent.exists():  # Ensure parent exists
            parent.mkdir(parents=True, exist_ok=True)  # Create parent directories if missing

        if not os.access(str(parent), os.W_OK):  # Verify write permission on parent
            raise PermissionError(f"Directory not writable: {parent}")  # Raise if not writable

        dfi.export(styled_df, str(out_path))  # Use dataframe_image to export styled DataFrame to PNG
        return  # Return None explicitly
    except Exception:
        raise  # Propagate any exception (do not swallow)


def generate_table_image_from_dataframe(df, csv_path):
    """
    Generate a zebra-striped PNG table image for a DataFrame, saved beside the CSV.

    :param df: pandas DataFrame in memory to render
    :param csv_path: Path to the CSV file (image will be same base name with .png)
    :return: Path to generated PNG image
    """
    try:
        png_path = str(Path(csv_path).with_suffix('.png'))  # Build PNG path by replacing extension
        styled = apply_zebra_style(df)  # Apply zebra styling to DataFrame
        export_dataframe_image(styled, png_path)  # Export styled DataFrame as PNG image
        return png_path  # Return path to generated PNG
    except Exception:
        raise  # Propagate exceptions to caller


def generate_csv_and_image(df, csv_path, is_visualizable=True, config=None, **to_csv_kwargs):
    """
    Save DataFrame to CSV and optionally generate a zebra-striped PNG image beside it.

    :param df: pandas DataFrame to save
    :param csv_path: Destination CSV file path
    :param is_visualizable: Whether to generate the PNG image (default True)
    :param to_csv_kwargs: Additional keyword args forwarded to pandas.DataFrame.to_csv
    :return: Path to saved CSV file
    """
    
    try:
        out_path = Path(csv_path)
        base = get_stacking_output_dir(csv_path, config or CONFIG)
        validate_output_path(base, str(out_path))

        df.to_csv(csv_path, **to_csv_kwargs)  # Save DataFrame to CSV with original kwargs
        if is_visualizable and len(df) <= 100:  # Generate image only when visualizable and within the safe row limit
            try:  # Guard PNG rendering to keep CSV persistence independent from image export
                generate_table_image_from_dataframe(df, csv_path)  # Generate PNG from in-memory DataFrame
            except Exception as _png_e:  # Contain PNG rendering failures locally
                print(f"{BackgroundColors.YELLOW}Warning: PNG generation failed for {Path(csv_path).name}: {_png_e}{Style.RESET_ALL}")  # Warn and continue without propagating PNG errors
        return csv_path  # Return CSV path
    except Exception:
        raise  # Propagate exceptions (no silent failures)


def aggregate_feature_usage(results_df, top_n=None):
    """
    Aggregate feature usage counts from a results DataFrame.

    :param results_df: DataFrame containing at least `features_list` and `model` columns
    :param top_n: Optional integer for number of top features to return (None => use config)
    :return: DataFrame indexed by feature with columns per model and a `total` column, limited to top-N features
    """
    
    try:
        if results_df is None or results_df.empty:  # Validate input presence
            raise ValueError("results_df is empty or None")  # Signal caller about empty input

        cfg_top_n = CONFIG.get("stacking", {}).get("top_n_features_heatmap", 15)  # Read default from config
        n = top_n if top_n is not None else cfg_top_n  # Determine effective top-N
        n = int(n)  # Ensure integer type for downstream logic

        counts = {}  # Dictionary to accumulate counts per model per feature

        for _, row in results_df.iterrows():  # Iterate rows to parse feature lists
            raw = row.get("features_list", None)  # Extract raw features_list value
            model_name = row.get("model", "AllModels")  # Extract model identifier or fallback

            if pd.isna(raw) or raw is None:  # Skip empty feature lists
                continue  # Nothing to count for this row

            features = []  # Prepare list of parsed feature names

            if isinstance(raw, str) and raw.strip().startswith("["):  # JSON-array string
                try:
                    features = json.loads(raw)  # Parse JSON list into Python list
                except Exception:
                    features = [f.strip() for f in raw.strip().strip('[]').split(",") if f.strip()]  # Fallback split
            elif isinstance(raw, str):
                features = [f.strip() for f in raw.split(",") if f.strip()]  # Comma-separated string parsing
            elif isinstance(raw, (list, tuple)):  # Already a list/tuple
                features = list(raw)  # Normalize to list
            else:
                features = [str(raw)]  # Convert unexpected types to single-item list

            for feat in features:  # Count each feature occurrence
                if feat == "":  # Skip empty strings
                    continue  # Ignore empty tokens
                counts.setdefault(feat, {}).setdefault(model_name, 0)  # Ensure nested dict entry exists
                counts[feat][model_name] += 1  # Increment count for this feature-model pair

        if not counts:  # If still empty after processing
            return pd.DataFrame()  # Return empty DataFrame to caller

        feature_index = sorted(list(counts.keys()))  # Sorted feature list for deterministic ordering
        model_names = sorted({m for feat in counts.values() for m in feat.keys()})  # Sorted model list

        matrix = []  # Rows for building DataFrame
        for feat in feature_index:  # Build each row of counts
            row_vals = [counts[feat].get(m, 0) for m in model_names]  # Get counts per model with zero fill
            matrix.append(row_vals)  # Append the row values

        df_counts = pd.DataFrame(matrix, index=feature_index, columns=model_names)  # Build counts DataFrame
        df_counts["total"] = df_counts.sum(axis=1)  # Compute total occurrences across models
        df_counts = df_counts.sort_values("total", ascending=False)  # Sort by total descending for top-N selection

        top_n_effective = min(n, len(df_counts))  # Cap top-N to available features
        return df_counts.head(top_n_effective)  # Return top-N slice
    except Exception:
        raise  # Propagate exceptions to caller (no silent failures)


def export_top_features_csv(feature_counts_df, csv_path, dataset_file=None):
    """
    Export aggregated top-N feature counts to CSV.

    :param feature_counts_df: DataFrame produced by aggregate_feature_usage()
    :param csv_path: Destination CSV path
    :return: Path to saved CSV file
    """
    
    try:
        if feature_counts_df is None or feature_counts_df.empty:  # Validate presence
            feature_counts_df = pd.DataFrame()  # Create empty DataFrame for consistent output

        out_path = Path(csv_path)  # Normalize to Path object
        out_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directory exists
        if dataset_file is None:
            raise RuntimeError("dataset_file must be provided to safely resolve stacking results directory")
        base = get_stacking_output_dir(dataset_file, CONFIG)
        validate_output_path(base, str(out_path))
        feature_counts_df.to_csv(str(out_path), index=True)  # Write DataFrame to CSV including index (feature names)
        return str(out_path)  # Return string path to caller
    except Exception:
        raise  # Propagate exceptions


def generate_feature_usage_heatmap(feature_counts_df, output_path, dataset_file=None):
    """
    Generate and save a heatmap PNG from feature counts DataFrame.

    :param feature_counts_df: DataFrame indexed by feature with per-model columns and a `total` column
    :param output_path: Destination PNG path
    :return: Path to saved PNG file
    """
    
    try:
        if feature_counts_df is None or feature_counts_df.empty:  # If no data, create a minimal empty heatmap
            fig, ax = plt.subplots(figsize=(6, 2))  # Create small figure for empty state
            ax.text(0.5, 0.5, "No features available", ha="center", va="center")  # Informative message
            ax.set_axis_off()  # Hide axes for empty message
            out_png = str(Path(output_path))  # Compute output path
            Path(out_png).parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            if dataset_file is None:
                raise RuntimeError("dataset_file must be provided to safely resolve stacking results directory")
            base = get_stacking_output_dir(dataset_file, CONFIG)
            validate_output_path(base, out_png)
            ensure_figure_min_4k_and_save(fig=fig, path=out_png, bbox_inches="tight")  # Save PNG to disk
            plt.close(fig)  # Close the figure to free memory
            return out_png  # Return path to generated PNG

        data = feature_counts_df.drop(columns=[c for c in feature_counts_df.columns if c == "total"], errors="ignore")  # Drop total for heatmap columns
        data = data.astype(int)  # Ensure integer type for plotting annotations

        n_rows = data.shape[0]  # Number of features (rows)
        n_cols = data.shape[1]  # Number of models (columns)
        height = max(2, 0.4 * n_rows)  # Heuristic height for readability
        width = max(4, 0.8 * n_cols)  # Heuristic width for readability

        fig, ax = plt.subplots(figsize=(width, height))  # Create figure with computed size
        sns.heatmap(data, annot=True, fmt="d", cmap="YlGnBu", cbar=True, ax=ax)  # Draw annotated heatmap
        ax.set_ylabel("")  # Keep y-axis label minimal
        ax.set_xlabel("")  # Keep x-axis label minimal
        plt.yticks(rotation=0)  # Ensure feature labels horizontal for readability
        plt.xticks(rotation=45, ha="right")  # Rotate column labels for space
        out_png = str(Path(output_path))  # Normalize output path to string
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directory exists
        plt.tight_layout()  # Tighten layout before saving
        ensure_figure_min_4k_and_save(fig=fig, path=out_png, dpi=300)  # Save figure to PNG with good resolution
        plt.close(fig)  # Close the figure to release resources
        return out_png  # Return saved PNG path
    except Exception:
        raise  # Propagate exceptions to caller


def save_augmentation_comparison_results(file_path, comparison_results, config=None):
    """
    Save data augmentation comparison results to CSV file.

    :param file_path: Path to the original CSV file being processed
    :param comparison_results: List of dictionaries containing comparison metrics
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        augmentation_comparison_filename = config.get("stacking", {}).get("augmentation_comparison_filename", "Data_Augmentation_Comparison_Results.csv")  # Get filename from config

        if not comparison_results:  # If no results to save
            return  # Exit early

        file_path_obj = Path(file_path)  # Create Path object
        feature_analysis_dir = file_path_obj.parent / "Feature_Analysis"  # Feature_Analysis directory
        os.makedirs(feature_analysis_dir, exist_ok=True)  # Ensure directory exists
        output_path = feature_analysis_dir / augmentation_comparison_filename  # Output file path

        df = pd.DataFrame(comparison_results)  # Convert results to DataFrame

        column_order = [
            "dataset",
            "feature_set",
            "classifier_type",
            "model_name",
            "data_source",
            "experiment_id",
            "experiment_mode",
            "augmentation_ratio",
            "n_features",
            "n_samples_train",
            "n_samples_test",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "fpr",
            "fnr",
            "training_time",
            "accuracy_improvement",
            "precision_improvement",
            "recall_improvement",
            "f1_score_improvement",
            "fpr_improvement",
            "fnr_improvement",
            "training_time_improvement",
            "features_list",
            "Hardware",
        ]  # Define desired column order with experiment traceability columns

        existing_columns = [col for col in column_order if col in df.columns]  # Filter to existing columns
        df = df[existing_columns]  # Reorder DataFrame columns

        df = add_hardware_column(df, existing_columns)  # Add hardware specifications column

        generate_csv_and_image(df, output_path, is_visualizable=True, index=False)  # Save CSV and generate PNG image
        print(
            f"{BackgroundColors.GREEN}Saved augmentation comparison results to {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
        )  # Output success message
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def find_local_feature_file(file_dir, filename, config=None):
    """
    Attempt to locate <file_dir>/Feature_Analysis/<filename>.

    :param file_dir: Directory to search within
    :param filename: Filename to search for
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: The matching path or None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Checking local Feature_Analysis in directory: {BackgroundColors.CYAN}{file_dir}{BackgroundColors.GREEN} for file: {BackgroundColors.CYAN}{filename}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        candidate = os.path.join(file_dir, "Feature_Analysis", filename)  # Construct candidate path

        if os.path.exists(candidate):  # If the candidate file exists
            return candidate  # Return the candidate path

        return None  # Not found
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def find_parent_feature_file(file_dir, filename, config=None):
    """
    Ascend parent directories searching for <parent>/Feature_Analysis/<filename>.

    :param file_dir: Directory to search within
    :param filename: Filename to search for
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: The matching path or None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Ascending parent directories from: {BackgroundColors.CYAN}{file_dir}{BackgroundColors.GREEN} searching for file: {BackgroundColors.CYAN}{filename}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        path = file_dir  # Start from the file's directory
        while True:  # Loop until break
            candidate = os.path.join(path, "Feature_Analysis", filename)  # Construct candidate path
            if os.path.exists(candidate):  # If the candidate file exists
                return candidate  # Return the candidate path

            parent = os.path.dirname(path)  # Get the parent directory
            if parent == path:  # If reached the root directory
                break  # Break the loop

            path = parent  # Move up to the parent directory

        return None  # Not found
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def find_dataset_level_feature_file(file_path, filename, config=None):
    """
    Try dataset-level search:

    - /.../Datasets/<dataset_name>/Feature_Analysis/<filename>
    - recursive search under dataset directory

    :param file_path: Path to the file
    :param filename: Filename to search for
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: The matching path or None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Searching dataset-level Feature_Analysis for file: {BackgroundColors.CYAN}{filename}{BackgroundColors.GREEN} related to file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        abs_path = os.path.abspath(file_path)  # Get absolute path of the input file
        parts = abs_path.split(os.sep)  # Split the path into parts

        if "Datasets" not in parts:  # If "Datasets" is not in the path parts
            return None  # Nothing to do

        idx = parts.index("Datasets")  # Get the index of "Datasets"
        if idx + 1 >= len(parts):  # If there is no dataset name after "Datasets"
            return None  # Nothing to do

        dataset_dir = os.sep.join(parts[: idx + 2])  # Construct the dataset directory path

        candidate = os.path.join(dataset_dir, "Feature_Analysis", filename)  # Construct candidate path for the direct path
        if os.path.exists(candidate):  # If the candidate file exists
            return candidate  # Return the candidate path

        matches = glob.glob(
            os.path.join(dataset_dir, "**", "Feature_Analysis", filename), recursive=True
        )  # Search recursively
        if matches:  # If matches are found
            return matches[0]  # Return the first match

        return None  # Not found
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def find_feature_file(file_path, filename, config=None):
    """
    Locate a feature-analysis CSV file related to `file_path`.

    Search order:
    - <file_dir>/Feature_Analysis/<filename>
    - ascend parent directories checking <parent>/Feature_Analysis/<filename>
    - dataset-level folder under `.../Datasets/<dataset_name>/Feature_Analysis/<filename>`
    - fallback: search under workspace ./Datasets/**/Feature_Analysis/<filename`

    :param file_path: Path to the file
    :param filename: Filename to search for
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: The matching path or None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Searching for feature analysis file: {BackgroundColors.CYAN}{filename}{BackgroundColors.GREEN} related to file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        file_dir = os.path.dirname(os.path.abspath(file_path))  # Get the directory of the input file

        result = find_local_feature_file(file_dir, filename, config=config)  # 1. Local Feature_Analysis in the same directory
        if result is not None:  # If found
            return result  # Return the result

        result = find_parent_feature_file(file_dir, filename, config=config)  # 2. Ascend parents checking for Feature_Analysis
        if result is not None:  # If found
            return result  # Return the result

        result = find_dataset_level_feature_file(file_path, filename, config=config)  # 3. Dataset-level Feature_Analysis
        if result is not None:  # If found
            return result  # Return the result

        print(
            f"{BackgroundColors.YELLOW}Warning: Feature analysis file {BackgroundColors.CYAN}{filename}{BackgroundColors.YELLOW} not found for dataset containing {BackgroundColors.CYAN}{file_path}{BackgroundColors.YELLOW}.{Style.RESET_ALL}"
        )  # Output the warning message

        return None  # Return None if not found
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_genetic_algorithm_features(file_path, config=None):
    """
    Extracts the features selected by the Genetic Algorithm from the corresponding
    "Genetic_Algorithm_Results.csv" file located in the "Feature_Analysis"
    subdirectory relative to the input file's directory.

    It specifically retrieves the 'best_features' (a JSON string) from the row
    where the 'run_index' is 'best', and returns it as a Python list.

    :param file_path: Full path to the current CSV file being processed (e.g., "./Datasets/.../DrDoS_DNS.csv").
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: List of features selected by the GA, or None if the file is not found or fails to load/parse.
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        file_dir = os.path.dirname(file_path)  # Determine the directory of the input file
        verbose_output(
            f"{BackgroundColors.GREEN}Extracting GA features for file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        ga_results_path = find_feature_file(file_path, "Genetic_Algorithm_Results.csv", config=config)  # Find the GA results file
        if ga_results_path is None:  # If the GA results file does not exist
            print(
                f"{BackgroundColors.YELLOW}Warning: GA results file not found for dataset containing {BackgroundColors.CYAN}{file_path}{BackgroundColors.YELLOW}. Skipping GA feature extraction for this file.{Style.RESET_ALL}"
            )
            return None  # Return None if the file does not exist

        try:  # Try to load the GA results
            low_memory = config.get("execution", {}).get("low_memory", False)  # Read low memory flag from config
            df = pd.read_csv(ga_results_path, usecols=["best_features", "run_index"], low_memory=low_memory)  # Load only the necessary columns
            df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names
            best_row = df[df["run_index"] == "best"].iloc[0]  # Get the row where run_index is 'best'
            best_features_json = best_row["best_features"]  # Get the JSON string of best features
            ga_features = json.loads(best_features_json)  # Parse the JSON string into a Python list

            verbose_output(
                f"{BackgroundColors.GREEN}Successfully extracted {BackgroundColors.CYAN}{len(ga_features)}{BackgroundColors.GREEN} GA features from the 'best' run.{Style.RESET_ALL}",
                config=config
            )  # Output the verbose message

            return ga_features  # Return the list of GA features
        except IndexError:  # If there is no 'best' run_index
            print(
                f"{BackgroundColors.RED}Error: 'best' run_index not found in GA results file at {BackgroundColors.CYAN}{ga_results_path}{Style.RESET_ALL}"
            )
            return None  # Return None if 'best' run_index is not found
        except Exception as e:  # If there is an error loading or parsing the file
            print(
                f"{BackgroundColors.RED}Error loading/parsing GA features from {BackgroundColors.CYAN}{ga_results_path}{BackgroundColors.RED}: {e}{Style.RESET_ALL}"
            )
            return None  # Return None if there was an error
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_principal_component_analysis_features(file_path, config=None):
    """
    Extracts the optimal number of Principal Components (n_components)
    from the "PCA_Results.csv" file located in the "Feature_Analysis"
    subdirectory relative to the input file's directory.

    The best result is determined by the highest 'cv_f1_score'.

    :param file_path: Full path to the current CSV file being processed (e.g., "./Datasets/.../DrDoS_DNS.csv").
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Integer representing the optimal number of components (n_components), or None if the file is not found or fails to load/parse.
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        file_dir = os.path.dirname(file_path)  # Determine the directory of the input file
        verbose_output(
            f"{BackgroundColors.GREEN}Extracting PCA features for file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        pca_results_path = find_feature_file(file_path, "PCA_Results.csv", config=config)  # Find the PCA results file
        if pca_results_path is None:  # If the PCA results file does not exist
            print(
                f"{BackgroundColors.YELLOW}Warning: PCA results file not found for dataset containing {BackgroundColors.CYAN}{file_path}{BackgroundColors.YELLOW}. Skipping PCA feature extraction for this file.{Style.RESET_ALL}"
            )
            return None  # Return None if the file does not exist

        try:  # Try to load the PCA results
            low_memory = config.get("execution", {}).get("low_memory", False)  # Read low memory flag from config
            df = pd.read_csv(pca_results_path, usecols=["n_components", "cv_f1_score"], low_memory=low_memory)  # Load only the necessary columns
            df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names

            if df.empty:  # Verify if the DataFrame is empty
                print(
                    f"{BackgroundColors.RED}Error: PCA results file at {BackgroundColors.CYAN}{pca_results_path}{BackgroundColors.RED} is empty.{Style.RESET_ALL}"
                )
                return None  # Return None if the file is empty

            best_row_index = df["cv_f1_score"].idxmax()  # Get the index of the row with the highest CV F1-Score
            best_n_components = df.loc[best_row_index, "n_components"]  # Get the optimal number of components

            verbose_output(
                f"{BackgroundColors.GREEN}Successfully extracted best PCA configuration. Optimal components: {BackgroundColors.CYAN}{best_n_components}{Style.RESET_ALL}"
            )  # Output the verbose message

            best_n_components = df.loc[best_row_index, "n_components"]  # Get the optimal number of components

            best_n_components_int = int(pd.to_numeric(best_n_components, errors="raise"))  # Ensure it's an integer

            return best_n_components_int  # Return the optimal number of components

        except KeyError as e:  # Handle missing columns
            print(
                f"{BackgroundColors.RED}Error: Required column {e} not found in PCA results file at {BackgroundColors.CYAN}{pca_results_path}{Style.RESET_ALL}"
            )
            return None  # Return None if required column is missing
        except Exception as e:  # Handle other errors (loading, parsing, etc.)
            print(
                f"{BackgroundColors.RED}Error loading/parsing PCA features from {BackgroundColors.CYAN}{pca_results_path}{BackgroundColors.RED}: {e}{Style.RESET_ALL}"
            )
            return None  # Return None if there was an error
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_recursive_feature_elimination_features(file_path, config=None):
    """
    Extracts the "top_features" list (Python literal string) from the first row of the
    "RFE_Run_Results.csv" file located in the "Feature_Analysis" subdirectory
    relative to the input file's directory.

    :param file_path: Full path to the current CSV file being processed (e.g., "./Datasets/.../DrDoS_DNS.csv").
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: List of top features selected by RFE from the first run, or None if the file is not found or fails to load/parse.
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        file_dir = os.path.dirname(file_path)  # Determine the directory of the input file
        verbose_output(
            f"{BackgroundColors.GREEN}Extracting RFE features for file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        rfe_runs_path = find_feature_file(file_path, "RFE_Run_Results.csv", config=config)  # Find the RFE runs file
        if rfe_runs_path is None:  # If the RFE runs file does not exist
            print(
                f"{BackgroundColors.YELLOW}Warning: RFE runs file not found for dataset containing {BackgroundColors.CYAN}{file_path}{BackgroundColors.YELLOW}. Skipping RFE feature extraction for this file.{Style.RESET_ALL}"
            )
            return None  # Return None if the file does not exist

        try:  # Try to load the RFE runs results
            low_memory = config.get("execution", {}).get("low_memory", False)  # Read low memory flag from config
            df = pd.read_csv(rfe_runs_path, usecols=["top_features"], low_memory=low_memory)  # Load only the "top_features" column
            df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names

            if not df.empty:  # Verify if the DataFrame is not empty
                top_features_raw = df.loc[0, "top_features"]  # Get the "top_features" from the first row

                top_features_str = str(top_features_raw)  # Ensure it's a string

                rfe_features = ast.literal_eval(top_features_str)  # Convert string to list

                verbose_output(
                    f"{BackgroundColors.GREEN}Successfully extracted RFE top features from Run 1. Total features: {BackgroundColors.CYAN}{len(rfe_features)}{Style.RESET_ALL}",
                    config=config
                )  # Output the verbose message

                return rfe_features  # Return the list of RFE features
            else:  # If the DataFrame is empty
                print(
                    f"{BackgroundColors.RED}Error: RFE runs file at {BackgroundColors.CYAN}{rfe_runs_path}{BackgroundColors.RED} is empty.{Style.RESET_ALL}"
                )
                return None  # Return None if the file is empty

        except Exception as e:  # If there is an error loading or parsing the file
            print(
                f"{BackgroundColors.RED}Error loading/parsing RFE features from {BackgroundColors.CYAN}{rfe_runs_path}{BackgroundColors.RED}: {e}{Style.RESET_ALL}"
            )
            return None  # Return None if there was an error
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def load_feature_selection_results(file_path, config=None):
    """
    Load GA, RFE and PCA feature selection artifacts for a given dataset file and
    print concise status messages.

    :param file_path: Path to the dataset CSV being processed.
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (ga_selected_features, pca_n_components, rfe_selected_features)
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        ga_selected_features = extract_genetic_algorithm_features(file_path, config=config)  # Extract GA features
        if ga_selected_features:  # If GA features were successfully extracted
            verbose_output(
                f"{BackgroundColors.GREEN}Genetic Algorithm Features successfully loaded for {BackgroundColors.CYAN}{os.path.basename(file_path)}{BackgroundColors.GREEN}. Total features: {BackgroundColors.CYAN}{len(ga_selected_features)}{Style.RESET_ALL}",
                config=config
            )
            verbose_output(
                f"{BackgroundColors.GREEN}Genetic Algorithm Selected Features: {BackgroundColors.CYAN}{ga_selected_features}{Style.RESET_ALL}",
                config=config
            )
        else:  # If GA features were not extracted
            print(
                f"{BackgroundColors.YELLOW}Proceeding without GA features for {BackgroundColors.CYAN}{os.path.basename(file_path)}{Style.RESET_ALL}"
            )

        pca_n_components = extract_principal_component_analysis_features(file_path, config=config)  # Extract PCA components
        if pca_n_components:  # If PCA components were successfully extracted
            verbose_output(
                f"{BackgroundColors.GREEN}PCA optimal components successfully loaded for {BackgroundColors.CYAN}{os.path.basename(file_path)}{BackgroundColors.GREEN}: {BackgroundColors.CYAN}{pca_n_components}{Style.RESET_ALL}",
                config=config
            )
            verbose_output(
                f"{BackgroundColors.GREEN}PCA Number of Components: {BackgroundColors.CYAN}{pca_n_components}{Style.RESET_ALL}",
                config=config
            )
        else:  # If PCA components were not extracted
            print(
                f"{BackgroundColors.YELLOW}Proceeding without PCA components for {BackgroundColors.CYAN}{os.path.basename(file_path)}{Style.RESET_ALL}"
            )

        rfe_selected_features = extract_recursive_feature_elimination_features(file_path, config=config)  # Extract RFE features
        if rfe_selected_features:  # If RFE features were successfully extracted
            verbose_output(
                f"{BackgroundColors.GREEN}RFE Features successfully loaded for {BackgroundColors.CYAN}{os.path.basename(file_path)}{BackgroundColors.GREEN}. Total features: {BackgroundColors.CYAN}{len(rfe_selected_features)}{Style.RESET_ALL}",
                config=config
            )
            verbose_output(
                f"{BackgroundColors.GREEN}RFE Selected Features: {BackgroundColors.CYAN}{rfe_selected_features}{Style.RESET_ALL}",
                config=config
            )
        else:  # If RFE features were not extracted
            print(
                f"{BackgroundColors.YELLOW}Proceeding without RFE features for {BackgroundColors.CYAN}{os.path.basename(file_path)}{Style.RESET_ALL}"
            )

        return ga_selected_features, pca_n_components, rfe_selected_features  # Return the extracted features
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def resolve_format_extension(file_format: str) -> str:
    """
    Resolve a file format string to its dot-prefixed file extension.

    :param file_format: File format string (arff, csv, parquet, txt).
    :return: Dot-prefixed file extension string.
    """

    try:
        extension_map = {  # Mapping of format strings to file extensions
            "arff": ".arff",
            "csv": ".csv",
            "parquet": ".parquet",
            "txt": ".txt",
        }
        return extension_map.get(file_format.lower(), ".csv")  # Return extension or fallback to .csv
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_arff_with_scipy(input_path: str, config=None) -> pd.DataFrame:
    """
    Load an ARFF file using scipy, decoding byte strings when necessary.

    :param input_path: Path to the ARFF file.
    :param config: Configuration dictionary (uses global CONFIG if None).
    :return: pandas DataFrame loaded from the ARFF file.
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Loading ARFF file with scipy: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        data, meta = scipy_arff.loadarff(input_path)  # Load the ARFF file using scipy
        df = pd.DataFrame(data)  # Convert the loaded data to a DataFrame

        for col in df.columns:  # Iterate through each column in the DataFrame
            if df[col].dtype == object:  # If column contains byte/string data
                df[col] = df[col].apply(
                    lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                )  # Decode bytes to strings

        return df  # Return the DataFrame with decoded strings
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_arff_with_liac(input_path: str, config=None) -> pd.DataFrame:
    """
    Load an ARFF file using the liac-arff library.

    :param input_path: Path to the ARFF file.
    :param config: Configuration dictionary (uses global CONFIG if None).
    :return: pandas DataFrame loaded from the ARFF file.
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Loading ARFF file with liac-arff: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        with open(input_path, "r", encoding="utf-8") as f:  # Open the ARFF file for reading
            data = arff.load(f)  # Load using liac-arff

        return pd.DataFrame(data["data"], columns=[attr[0] for attr in data["attributes"]])  # Convert to DataFrame
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_arff_file(input_path: str, config=None) -> pd.DataFrame:
    """
    Load an ARFF file, trying scipy first and falling back to liac-arff if needed.

    :param input_path: Path to the ARFF file.
    :param config: Configuration dictionary (uses global CONFIG if None).
    :return: pandas DataFrame loaded from the ARFF file.
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        try:  # Attempt to load using scipy
            return load_arff_with_scipy(input_path, config=config)  # Return DataFrame from scipy loader
        except Exception as e:  # If scipy fails, warn and try liac-arff
            verbose_output(
                f"{BackgroundColors.YELLOW}Warning: Failed to load ARFF with scipy ({e}). Trying liac-arff...{Style.RESET_ALL}",
                config=config
            )  # Output fallback warning message

            try:  # Attempt to load using liac-arff
                return load_arff_with_liac(input_path, config=config)  # Return DataFrame from liac loader
            except Exception as e2:  # If both loaders fail
                raise RuntimeError(f"Failed to load ARFF file with both scipy and liac-arff: {e2}")  # Raise combined error
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_parquet_file(input_path: str, config=None) -> pd.DataFrame:
    """
    Load a Parquet file into a pandas DataFrame.

    :param input_path: Path to the Parquet file.
    :param config: Configuration dictionary (uses global CONFIG if None).
    :return: pandas DataFrame loaded from the Parquet file.
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Loading Parquet file: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        return pd.read_parquet(input_path)  # Load and return the Parquet file as a DataFrame
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_txt_file(input_path: str, config=None) -> pd.DataFrame:
    """
    Load a TXT file into a pandas DataFrame, assuming tab-separated values.

    :param input_path: Path to the TXT file.
    :param config: Configuration dictionary (uses global CONFIG if None).
    :return: pandas DataFrame containing the loaded dataset.
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Loading TXT file: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        low_memory = config.get("execution", {}).get("low_memory", False)  # Read low memory flag from config
        df = pd.read_csv(input_path, sep="\t", low_memory=low_memory)  # Load TXT file using tab separator

        return df  # Return the DataFrame
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def dispatch_format_loader(file_path: str, file_format: str, config=None) -> pd.DataFrame:
    """
    Dispatch dataset loading to the appropriate loader based on the given file format.

    :param file_path: Path to the dataset file to load.
    :param file_format: File format string: arff, csv, parquet, or txt.
    :param config: Configuration dictionary (uses global CONFIG if None).
    :return: pandas DataFrame loaded from the file.
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        supported_formats = {"arff", "csv", "parquet", "txt"}  # Set of supported file formats
        if file_format not in supported_formats:  # If the format is not supported
            raise ValueError(f"Unsupported file format: '{file_format}'. Supported: {sorted(supported_formats)}")  # Raise error for unsupported format

        low_memory = config.get("execution", {}).get("low_memory", False)  # Read low memory flag from config

        if file_format == "arff":  # If the file format is ARFF
            return load_arff_file(file_path, config=config)  # Load and return ARFF file
        elif file_format == "csv":  # If the file format is CSV
            df = pd.read_csv(file_path, low_memory=low_memory)  # Load CSV file with configured memory mode
            return df  # Return the loaded CSV DataFrame
        elif file_format == "parquet":  # If the file format is Parquet
            return load_parquet_file(file_path, config=config)  # Load and return Parquet file
        elif file_format == "txt":  # If the file format is TXT
            return load_txt_file(file_path, config=config)  # Load and return TXT file
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_augmented_dataset(file_path: str, config=None) -> Optional[pd.DataFrame]:
    """
    Load an augmented dataset file and return a DataFrame.
    Supports csv, txt, parquet, and arff formats via stacking.augmentation_file_format.

    :param file_path: Path to the augmented dataset file.
    :param config: Configuration dictionary (uses global CONFIG if None).
    :return: DataFrame or None if file not found or invalid.
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"\n{BackgroundColors.GREEN}Loading augmented dataset from: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}",
            config=config
        )  # Output the loading augmented dataset message

        if not verify_filepath_exists(file_path):  # If the augmented dataset file does not exist
            print(f"{BackgroundColors.RED}Augmented dataset file not found: {file_path}{Style.RESET_ALL}")
            return None  # Return None

        file_format = config.get("stacking", {}).get("augmentation_file_format", "csv")  # Read configured augmentation file format

        verbose_output(
            f"{BackgroundColors.GREEN}Loading augmented dataset with format: {BackgroundColors.CYAN}{file_format}{Style.RESET_ALL}",
            config=config
        )  # Output the selected format message

        df = dispatch_format_loader(file_path, file_format, config=config)  # Dispatch to the appropriate loader

        df.columns = df.columns.str.strip()  # Clean column names by stripping leading/trailing whitespace

        if df.shape[1] < 2:  # If there are less than 2 columns
            print(f"{BackgroundColors.RED}Augmented dataset must have at least 1 feature and 1 target.{Style.RESET_ALL}")
            return None  # Return None

        return df  # Return the loaded DataFrame
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_dataset(file_path: str, config=None) -> Optional[pd.DataFrame]:
    """
    Load a dataset file and return a DataFrame.
    Supports csv, txt, parquet, and arff formats via stacking.dataset_file_format.

    :param file_path: Path to the dataset file.
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: DataFrame
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"\n{BackgroundColors.GREEN}Loading dataset from: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}",
            config=config
        )  # Output the loading dataset message

        if not verify_filepath_exists(file_path):  # If the dataset file does not exist
            print(f"{BackgroundColors.RED}Dataset file not found: {file_path}{Style.RESET_ALL}")
            return None  # Return None

        file_format = config.get("stacking", {}).get("dataset_file_format", "csv")  # Read configured dataset file format

        verbose_output(
            f"{BackgroundColors.GREEN}Loading dataset with format: {BackgroundColors.CYAN}{file_format}{Style.RESET_ALL}",
            config=config
        )  # Output the selected format message

        df = dispatch_format_loader(file_path, file_format, config=config)  # Dispatch to the appropriate loader

        df.columns = df.columns.str.strip()  # Clean column names by stripping leading/trailing whitespace

        if df.shape[1] < 2:  # If there are less than 2 columns
            print(f"{BackgroundColors.RED}Dataset must have at least 1 feature and 1 target.{Style.RESET_ALL}")
            return None  # Return None

        return df  # Return the loaded DataFrame
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def sanitize_feature_names(columns):
    r"""
    Sanitize column names by removing special JSON characters that LightGBM doesn't support.
    Replaces: { } [ ] : , " \ with underscores.

    :param columns: pandas Index or list of column names
    :return: list of sanitized column names
    """
    
    try:
        sanitized = []  # List to store sanitized column names
        
        for col in columns:  # Iterate over each column name
            clean_col = re.sub(r"[{}\[\]:,\"\\]", "_", str(col))  # Replace special characters with underscores
            clean_col = re.sub(r"_+", "_", clean_col)  # Replace multiple underscores with a single underscore
            clean_col = clean_col.strip("_")  # Remove leading/trailing underscores
            sanitized.append(clean_col)  # Add sanitized column name to the list
            
        return sanitized  # Return the list of sanitized column names
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def verify_selected_features_exist(selected_features: list, dataset_columns: list, method_name: str) -> list:
    """
    Verify that features selected by a feature-selection method exist in the dataset columns.

    - Prints a structured warning if any selected features are missing.
    - Returns the filtered list of valid features (preserving original order, removing duplicates).
    - Raises ValueError if ALL selected features are missing, or if inputs are invalid/empty.

    :param selected_features: list of selected feature names (strings)
    :param dataset_columns: list of dataset column names (strings)
    :param method_name: human-friendly method name (e.g., 'GA', 'RFE', 'PCA')
    :return: filtered list of valid feature names
    """

    if selected_features is None:
        raise ValueError(f"No selected features provided for {method_name}.")
    if not isinstance(selected_features, (list, tuple)):
        raise ValueError("selected_features must be a list or tuple")
    if not selected_features:
        raise ValueError(f"selected_features is empty for {method_name}")
    if dataset_columns is None or not isinstance(dataset_columns, (list, tuple)):
        raise ValueError("dataset_columns must be a non-empty list or tuple")
    if not dataset_columns:
        raise ValueError("dataset_columns is empty")

    mn = (method_name or "").strip().lower()
    is_pca = mn == "pca"
    if is_pca:
        if all(isinstance(f, str) and f.strip().upper().startswith("PC") for f in selected_features):
            return list(dict.fromkeys(selected_features))  # Preserve order, remove duplicates

    dataset_set = set(dataset_columns)
    valid_features = []
    seen = set()
    for f in selected_features:
        if f in dataset_set and f not in seen:
            valid_features.append(f)
            seen.add(f)

    missing = [f for f in selected_features if f not in dataset_set]

    if missing and len(valid_features) > 0:
        print(f"{BackgroundColors.YELLOW}Missing features detected for {BackgroundColors.CYAN}{method_name}{Style.RESET_ALL}:")
        for m in missing:
            print(f"- {m}")
        print(f"{BackgroundColors.GREEN}Proceeding with {len(valid_features)} valid features for {BackgroundColors.CYAN}{method_name}{Style.RESET_ALL}.")
        return valid_features

    if missing and len(valid_features) == 0:
        print(f"{BackgroundColors.RED}All selected features for {method_name} are missing from dataset columns.{Style.RESET_ALL}")
        for m in missing:
            print(f"- {m}")
        print(f"{BackgroundColors.YELLOW}Using 0 out of {len(selected_features)} selected features for {BackgroundColors.CYAN}{method_name}{Style.RESET_ALL}.{Style.RESET_ALL}")
        raise ValueError(f"All selected features for {method_name} are missing from dataset columns")

    return list(dict.fromkeys(selected_features))


def preprocess_dataframe(df, remove_zero_variance=True, config=None):
    """
    Preprocess a DataFrame by removing rows with NaN or infinite values and
    dropping zero-variance numeric features.

    :param df: pandas DataFrame to preprocess
    :param remove_zero_variance: whether to drop numeric columns with zero variance
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: cleaned DataFrame
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        if remove_zero_variance:  # If remove_zero_variance is set to True
            verbose_output(
                f"{BackgroundColors.GREEN}Preprocessing DataFrame: "
                f"{BackgroundColors.CYAN}normalizing and sanitizing column names, removing NaN/infinite rows, and dropping zero-variance numeric features"
                f"{BackgroundColors.GREEN}.{Style.RESET_ALL}",
                config=config
            )
        else:  # If remove_zero_variance is set to False
            verbose_output(
                f"{BackgroundColors.GREEN}Preprocessing DataFrame: "
                f"{BackgroundColors.CYAN}normalizing and sanitizing column names and removing NaN/infinite rows"
                f"{BackgroundColors.GREEN}.{Style.RESET_ALL}",
                config=config
            )

        if df is None:  # If the DataFrame is None
            return df  # Return None

        df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names
        
        df.columns = sanitize_feature_names(df.columns)  # Sanitize column names to remove special characters

        df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN in-place to avoid creating a copy
        df_clean = df.dropna()  # Drop rows containing NaN values into a new dataframe
        del df  # Release original dataframe reference to free memory after cleaning
        gc.collect()  # Force garbage collection to reclaim memory from deleted original dataframe

        if remove_zero_variance:  # If remove_zero_variance is set to True
            numeric_cols = df_clean.select_dtypes(include=["number"]).columns  # Select only numeric columns
            if len(numeric_cols) > 0:  # If there are numeric columns
                variances = df_clean[numeric_cols].var(axis=0, ddof=0)  # Calculate variances
                zero_var_cols = variances[variances == 0].index.tolist()  # Get columns with zero variance
                if zero_var_cols:  # If there are zero-variance columns
                    df_clean = df_clean.drop(columns=zero_var_cols)  # Drop zero-variance columns

        return df_clean  # Return the cleaned DataFrame
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def scale_and_split(X, y, test_size=0.2, random_state=42, config=None, X_augmented=None, y_augmented=None):
    """
    Scales the numeric features using StandardScaler and splits the data
    into training and testing sets. If augmented data is provided, it is merged ONLY into the training set AFTER splitting to ensure test set contains exclusively original samples.

    Note: The target variable 'y' is label-encoded before splitting.

    :param X: Features DataFrame (must contain numeric features).
    :param y: Target Series or array.
    :param test_size: Fraction of the data to reserve for the test set.
    :param random_state: Seed for the random split for reproducibility.
    :param config: Configuration dictionary (uses global CONFIG if None)
    :param X_augmented: Optional augmented features DataFrame to merge into training set only
    :param y_augmented: Optional augmented target Series/array to merge into training set only
    :return: Tuple (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Scaling features and splitting data (train/test ratio: {BackgroundColors.CYAN}{1-test_size}/{test_size}{BackgroundColors.GREEN})...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        y = pd.Series(y)  # Normalize target to pandas Series

        le = LabelEncoder()  # Initialize a LabelEncoder
        encoded_values: np.ndarray = np.asarray(le.fit_transform(y.to_numpy()), dtype=int)  # Encode target labels as integers

        y_encoded = pd.Series(encoded_values, index=y.index)  # Create a Series for the encoded target

        numeric_X = X.select_dtypes(include=np.number)  # Select only numeric columns for scaling
        non_numeric_X = X.select_dtypes(exclude=np.number)  # Identify non-numeric columns (to be dropped)

        if not non_numeric_X.empty:  # If non-numeric columns were found
            print(
                f"{BackgroundColors.YELLOW}Warning: Dropping non-numeric feature columns for scaling: {BackgroundColors.CYAN}{list(non_numeric_X.columns)}{Style.RESET_ALL}"
            )  # Warn about dropped columns

        if numeric_X.empty:  # If no numeric features remain
            raise ValueError(
                f"{BackgroundColors.RED}No numeric features found in X after filtering.{Style.RESET_ALL}"
            )  # Raise an error if X is empty

        X_train, X_test, y_train, y_test = train_test_split(
            numeric_X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded
        )  # Split the data into training and testing sets with stratification

        if X_augmented is not None and y_augmented is not None:  # If augmented data is provided for training enhancement
            verbose_output(
                f"{BackgroundColors.GREEN}Merging augmented data into training set ({len(X_augmented)} augmented samples)...{Style.RESET_ALL}",
                config=config
            )  # Output augmentation merge message
            
            y_augmented_series = pd.Series(y_augmented)  # Normalize augmented target to pandas Series
            le_augmented = LabelEncoder()  # Initialize LabelEncoder for augmented target
            le_augmented.classes_ = le.classes_  # Use same classes as original encoder for consistency
            encoded_augmented_values = np.asarray(le_augmented.transform(y_augmented_series.to_numpy()), dtype=int)  # Encode augmented target labels as integers
            y_augmented_encoded = pd.Series(encoded_augmented_values, index=y_augmented_series.index)  # Create Series for encoded augmented target
            
            numeric_X_augmented = X_augmented.select_dtypes(include=np.number)  # Select only numeric columns from augmented features
            
            X_train = pd.concat([X_train, numeric_X_augmented], ignore_index=True)  # Concatenate augmented features into training set
            y_train = pd.concat([y_train, y_augmented_encoded], ignore_index=True)  # Concatenate augmented target into training labels
            
            verbose_output(
                f"{BackgroundColors.GREEN}Training set expanded to {BackgroundColors.CYAN}{len(X_train)}{BackgroundColors.GREEN} samples (original + augmented){Style.RESET_ALL}",
                config=config
            )  # Output expanded training set size

        scaler = StandardScaler()  # Initialize the StandardScaler

        X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the training features (including augmented if provided)

        X_test_scaled = scaler.transform(X_test)  # Transform the testing features (original data only)

        verbose_output(
            f"{BackgroundColors.GREEN}Data split successful. Training set shape: {BackgroundColors.CYAN}{X_train_scaled.shape}{BackgroundColors.GREEN}. Testing set shape: {BackgroundColors.CYAN}{X_test_scaled.shape}{Style.RESET_ALL}",
            config=config
        )  # Output the successful split message

        return (
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            scaler,
        )  # Return scaled features, target, and the fitted scaler
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_models(config=None):
    """
    Initializes and returns a dictionary of models to train.

    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary of model name and instance
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Initializing models for training...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message
        
        n_jobs = config.get("evaluation", {}).get("n_jobs", -1)  # Get n_jobs from config
        random_state = config.get("evaluation", {}).get("random_state", 42)  # Get random_state from config
        
        rf_params = config.get("models", {}).get("random_forest", {})  # Random Forest params
        svm_params = config.get("models", {}).get("svm", {})  # SVM params
        xgb_params = config.get("models", {}).get("xgboost", {})  # XGBoost params
        lr_params = config.get("models", {}).get("logistic_regression", {})  # Logistic Regression params
        knn_params = config.get("models", {}).get("knn", {})  # KNN params
        gb_params = config.get("models", {}).get("gradient_boosting", {})  # Gradient Boosting params
        lgb_params = config.get("models", {}).get("lightgbm", {})  # LightGBM params
        mlp_params = config.get("models", {}).get("mlp", {})  # MLP params

        models = {  # Build full models dictionary with all classifiers
            "Random Forest": RandomForestClassifier(
                n_estimators=rf_params.get("n_estimators", 100),
                random_state=rf_params.get("random_state", random_state),
                n_jobs=n_jobs
            ),
            "SVM": SVC(
                kernel=svm_params.get("kernel", "rbf"),
                probability=svm_params.get("probability", True),
                random_state=svm_params.get("random_state", random_state)
            ),
            "XGBoost": XGBClassifier(
                eval_metric=xgb_params.get("eval_metric", "mlogloss"),
                random_state=xgb_params.get("random_state", random_state),
                n_jobs=n_jobs
            ),
            "Logistic Regression": LogisticRegression(
                max_iter=lr_params.get("max_iter", 1000),
                random_state=lr_params.get("random_state", random_state),
                n_jobs=n_jobs
            ),
            "KNN": KNeighborsClassifier(
                n_neighbors=knn_params.get("n_neighbors", 5),
                n_jobs=n_jobs
            ),
            "Nearest Centroid": NearestCentroid(),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=gb_params.get("random_state", random_state)
            ),
            "LightGBM": lgb.LGBMClassifier(
                force_row_wise=lgb_params.get("force_row_wise", True),
                min_gain_to_split=lgb_params.get("min_gain_to_split", 0.01),
                random_state=lgb_params.get("random_state", random_state),
                verbosity=lgb_params.get("verbosity", -1),
                n_jobs=n_jobs
            ),
            "MLP (Neural Net)": MLPClassifier(
                hidden_layer_sizes=mlp_params.get("hidden_layer_sizes", (100,)),
                max_iter=mlp_params.get("max_iter", 500),
                random_state=mlp_params.get("random_state", random_state)
            ),
        }

        classifiers_list = config.get("stacking", {}).get("enabled_classifiers", None)  # Read optional classifier filter list from config

        if classifiers_list is None:  # If no classifier filter key is present, preserve original behavior
            return models  # Return full models dictionary with no filtering applied

        unique_classifiers = list(dict.fromkeys(classifiers_list))  # Deduplicate classifiers list while preserving insertion order
        filtered_models = {k: v for k, v in models.items() if k in unique_classifiers}  # Retain only classifiers whose names appear in the filter list

        enabled_names = list(filtered_models.keys())  # Collect names of enabled classifiers for logging
        disabled_names = [k for k in models if k not in filtered_models]  # Collect names of disabled classifiers for logging

        verbose_output(
            f"[DEBUG] Enabled classifiers: {enabled_names}",
            config=config
        )  # Output the list of enabled classifiers
        verbose_output(
            f"[DEBUG] Disabled classifiers: {disabled_names}",
            config=config
        )  # Output the list of disabled classifiers

        return filtered_models  # Return filtered models dictionary, may be empty if all names were invalid or list was empty
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def extract_hyperparameter_optimization_results(csv_path, config=None):
    """
    Extract hyperparameter optimization results for a specific dataset file.

    Looks for the HYPERPARAMETERS_FILENAME file in the "Classifiers_Hyperparameters"
    subdirectory relative to the dataset CSV file. Filters results to match the
    current base_csv filename being processed.

    This function extracts **only the best hyperparameters** for each classifier
    that corresponds to the current file being processed.

    :param csv_path: Path to the dataset CSV file being processed.
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary mapping model names to their best hyperparameters, or None if not found.
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Looking for hyperparameter optimization results for: {BackgroundColors.CYAN}{csv_path}{Style.RESET_ALL}",
            config=config
        )  # Inform user which dataset we're searching for

        file_dir = os.path.dirname(csv_path)  # Directory containing the dataset file
        base_filename = os.path.basename(csv_path)  # Get the base filename (e.g., "UDPLag.csv")
        
        hyperparameters_filename = config.get("stacking", {}).get("hyperparameters_filename", "Hyperparameter_Optimization_Results.csv")  # Get filename from config

        hyperparams_path = os.path.join(
            file_dir, "Classifiers_Hyperparameters", hyperparameters_filename
        )  # Path to hyperparameter optimization results

        if not verify_filepath_exists(hyperparams_path):  # If the hyperparameters file does not exist
            verbose_output(
                f"{BackgroundColors.YELLOW}No hyperparameter optimization results found at: {BackgroundColors.CYAN}{hyperparams_path}{Style.RESET_ALL}",
                config=config
            )
            return None  # Return None if no optimization results found

        try:  # Try to load the CSV file
            low_memory = config.get("execution", {}).get("low_memory", False)  # Read low memory flag from config
            df = pd.read_csv(hyperparams_path, low_memory=low_memory)  # Load the CSV into a DataFrame
            df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names
        except Exception as e:  # If there is an error loading the CSV
            print(
                f"{BackgroundColors.RED}Failed to load hyperparameter optimization file {hyperparams_path}: {e}{Style.RESET_ALL}"
            )
            return {}  # Return empty dict on failure

        matching_rows = df[df["base_csv"] == base_filename]  # Filter by base_csv column

        if matching_rows.empty:  # If no matching rows found
            verbose_output(
                f"{BackgroundColors.YELLOW}No hyperparameter results found for file: {BackgroundColors.CYAN}{base_filename}{BackgroundColors.YELLOW} in {BackgroundColors.CYAN}{hyperparams_path}{Style.RESET_ALL}",
                config=config
            )
            return None  # Return None if no results for this file

        results = {}  # Initialize dictionary to hold parsed results per model
        for _, row in matching_rows.iterrows():  # Iterate over each matching row
            try:  # Try to parse each row
                model = row.get("model") or row.get("Model")  # Try common column names for model identifier
                if not model:  # If model identifier is missing
                    continue  # Skip invalid rows

                best_params_raw = (
                    row.get("best_params") or row.get("best_params_json") or row.get("best_params_str")
                )  # Try common column names for best_params
                best_params = None  # Default if parsing fails or value missing
                if isinstance(best_params_raw, str) and best_params_raw.strip():  # If best_params_raw is a non-empty string
                    try:  # Try to parse best_params as JSON first
                        best_params = json.loads(best_params_raw)  # Parse JSON string if possible
                    except Exception:  # If JSON parsing fails, try ast.literal_eval as a fallback
                        try:  # Try to parse using ast.literal_eval
                            best_params = ast.literal_eval(best_params_raw)  # Safely evaluate string to Python literal
                        except Exception:  # If both parsing attempts fail
                            best_params = None  # Leave as None if parsing fails

                results[str(model)] = {"best_params": best_params}  # Store parsed parameters only
            except Exception:  # Catch any unexpected errors during row parsing
                continue  # Skip problematic rows silently

        verbose_output(
            f"{BackgroundColors.GREEN}Loaded {BackgroundColors.CYAN}{len(results)}{BackgroundColors.GREEN} hyperparameter optimization results for {BackgroundColors.CYAN}{base_filename}{BackgroundColors.GREEN} from: {BackgroundColors.CYAN}{hyperparams_path}{Style.RESET_ALL}",
            config=config
        )
        return results  # Return the normalized results mapping
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def normalize(name):
    """
    Normalize a model name by removing non-alphanumeric characters and converting to lowercase.
    
    :param name: The model name to normalize.
    :return: Normalized model name string.
    """
    
    return "".join(
        [c.lower() for c in str(name) if c.isalnum()]
    )  # Normalize model name by removing non-alphanumeric characters and converting to lowercase


def apply_hyperparameters_to_models(hyperparams_map, models_map, config=None):
    """
    Apply hyperparameter mappings to instantiated models.

    This function attempts to match keys from the hyperparameter mapping
    to the model names in the provided models dictionary. Matching is
    attempted in the following order: exact match, case-insensitive
    match, and normalized (alphanumeric-only) match. When a matching
    entry is found and the corresponding value is a valid dictionary of
    hyperparameters, they are applied to the estimator using set_params.
    Errors during matching or parameter application are handled
    gracefully and reported via verbose_output.

    :param hyperparams_map: Mapping of model name -> hyperparameter dict
    :param models_map: Mapping of model name -> instantiated estimator
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Updated models_map with applied hyperparameters where possible
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Starting to apply hyperparameters to models...{Style.RESET_ALL}",
            config=config
        )  # Inform user that application is starting

        if not hyperparams_map:  # Nothing to apply
            return models_map  # Return models unchanged

        hp_keys = list(hyperparams_map.keys())  # List of provided model names
        hp_normalized = {k: normalize(k) for k in hp_keys}  # Normalized lookup for matching

        for model_name, model in models_map.items():  # Iterate over each instantiated model
            try:  # Try to match hyperparameters to this model
                params = None  # Default if not found

                if model_name in hyperparams_map:  # Attempt exact match
                    params = hyperparams_map[model_name]  # Exact match found
                else:  # Try case-insensitive and normalized matches
                    lower_matches = [k for k in hp_keys if k.lower() == model_name.lower()]  # Case-insensitive match
                    if lower_matches:  # If case-insensitive match found
                        params = hyperparams_map[lower_matches[0]]  # Use the matched parameters
                    else:  # Try normalized match
                        norm = normalize(model_name)  # Compute normalized name
                        norm_matches = [k for k, nk in hp_normalized.items() if nk == norm]  # Normalized match
                        if norm_matches:  # If normalized match found
                            params = hyperparams_map[norm_matches[0]]  # Use the matched parameters

                if params is None:  # No parameters for this model
                    continue  # Skip to next model

                if isinstance(params, str):  # Parameters stored as string
                    try:  # Try JSON decode
                        params = json.loads(params)  # Parse JSON if possible
                    except Exception:  # Fallback to literal evaluation
                        try:  # Try to parse using ast.literal_eval
                            params = ast.literal_eval(params)  # Safely evaluate string to Python literal
                        except Exception:  # If both parsing attempts fail
                            params = None  # Leave as None if parsing fails

                if not isinstance(params, dict):  # Ensure parameters are valid dictionary
                    verbose_output(
                        f"{BackgroundColors.YELLOW}Warning: Parsed hyperparameters for {BackgroundColors.CYAN}{model_name}{BackgroundColors.YELLOW} are not a dict. Skipping.{Style.RESET_ALL}",
                        config=config
                    )
                    continue  # Skip invalid parameter entries

                try:  # Try applying parameters
                    model.set_params(**params)  # Apply parameters to estimator
                    verbose_output(
                        f"{BackgroundColors.GREEN}Applied hyperparameters to {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}",
                        config=config
                    )  # Inform success
                except Exception as e:  # If applying fails
                    print(
                        f"{BackgroundColors.YELLOW}Failed to apply hyperparameters to {BackgroundColors.CYAN}{model_name}{BackgroundColors.YELLOW}: {e}{Style.RESET_ALL}"
                    )  # Warn user
            except Exception:  # Catch any unexpected errors for this model
                continue  # Skip problematic entries silently

        return models_map  # Return updated model mapping
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def load_pca_object(file_path, pca_n_components, config=None):
    """
    Loads a pre-fitted PCA object from a pickle file.

    :param file_path: Path to the dataset CSV file.
    :param pca_n_components: Number of PCA components to load.
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: PCA object if found, None otherwise.
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Loading the PCA Cache object with {BackgroundColors.CYAN}{pca_n_components}{BackgroundColors.GREEN} components from file {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        file_dir = os.path.dirname(file_path)  # Get the directory of the dataset
        pca_file = os.path.join(
            file_dir, "Cache", f"PCA_{pca_n_components}_components.pkl"
        )  # Construct the path to the PCA pickle file

        if not verify_filepath_exists(pca_file):  # Verify if the PCA file exists
            verbose_output(
                f"{BackgroundColors.YELLOW}PCA object file not found at {BackgroundColors.CYAN}{pca_file}{Style.RESET_ALL}",
                config=config
            )
            return None  # Return None if the file doesn't exist

        try:  # Try to load the PCA object
            with open(pca_file, "rb") as f:  # Open the PCA pickle file
                pca = pickle.load(f)  # Load the PCA object
            verbose_output(
                f"{BackgroundColors.GREEN}Successfully loaded PCA object from {BackgroundColors.CYAN}{pca_file}{Style.RESET_ALL}",
                config=config
            )
            return pca  # Return the loaded PCA object
        except Exception as e:  # Handle any errors during loading
            print(
                f"{BackgroundColors.RED}Error loading PCA object from {BackgroundColors.CYAN}{pca_file}{BackgroundColors.RED}: {e}{Style.RESET_ALL}"
            )
            return None  # Return None if there was an error
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def apply_pca_transformation(X_train_scaled, X_test_scaled, pca_n_components, file_path=None, config=None):
    """
    Applies Principal Component Analysis (PCA) transformation to the scaled training
    and testing datasets using the optimal number of components.

    First attempts to load a pre-fitted PCA object from disk. If not found,
    fits a new PCA model on the training data.

    :param X_train_scaled: Scaled training features (numpy array).
    :param X_test_scaled: Scaled testing features (numpy array).
    :param pca_n_components: Optimal number of components (integer), or None/0 if PCA is skipped.
    :param file_path: Path to the dataset CSV file (optional, for loading pre-fitted PCA).
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (X_train_pca, X_test_pca) - Transformed features, or (None, None).
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        X_train_pca = None  # Initialize PCA training features
        X_test_pca = None  # Initialize PCA testing features

        if pca_n_components is not None and pca_n_components > 0:  # If PCA components are specified
            verbose_output(
                f"{BackgroundColors.GREEN}Starting PCA transformation with {BackgroundColors.CYAN}{pca_n_components}{BackgroundColors.GREEN} components...{Style.RESET_ALL}",
                config=config
            )  # Output the verbose message

            n_features = X_train_scaled.shape[1]  # Get the number of features in the training set
            n_components = min(
                pca_n_components, n_features
            )  # Effective number of components cannot exceed number of features

            if n_components < pca_n_components:  # Verify if the component count was reduced
                print(
                    f"{BackgroundColors.YELLOW}Warning: Reduced PCA components from {pca_n_components} to {n_components} due to limited features ({n_features}).{Style.RESET_ALL}"
                )

            pca = None  # Initialize PCA object as None
            if file_path:  # Only attempt to load if file_path is provided
                pca = load_pca_object(file_path, n_components, config=config)  # Load pre-fitted PCA object

            if pca is None:  # If PCA object wasn't loaded, fit a new one
                verbose_output(
                    f"{BackgroundColors.GREEN}Fitting new PCA model with {BackgroundColors.CYAN}{n_components}{BackgroundColors.GREEN} components...{Style.RESET_ALL}",
                    config=config
                )
                pca = PCA(n_components=n_components)  # Initialize PCA with the effective number of components
                X_train_pca = pca.fit_transform(X_train_scaled)  # Fit and transform the training data
            else:  # PCA object was loaded successfully
                print(
                    f"{BackgroundColors.GREEN}Using pre-fitted PCA model with {BackgroundColors.CYAN}{n_components}{BackgroundColors.GREEN} components{Style.RESET_ALL}"
                )
                X_train_pca = pca.transform(X_train_scaled)  # Only transform the training data

            X_test_pca = pca.transform(X_test_scaled)  # Transform the testing data

            verbose_output(
                f"{BackgroundColors.GREEN}PCA applied successfully. Transformed data shape: {BackgroundColors.CYAN}{X_train_pca.shape}{Style.RESET_ALL}"
            )  # Output the transformed shape

        return X_train_pca, X_test_pca  # Return the transformed features
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_feature_subset(X_scaled, features, feature_names):
    """
    Returns a subset of features from the scaled feature set based on the provided feature names.
    Also returns the actual feature names that were successfully selected.

    :param X_scaled: Scaled features (numpy array).
    :param features: List of feature names to select.
    :param feature_names: List of all feature names corresponding to columns in X_scaled.
    :return: Tuple of (subset array, list of actual selected feature names)
    """
    
    try:
        if features:  # Only proceed if the list of selected features is NOT empty/None
            indices = []  # List to store indices of selected features
            selected_names = []  # List to store names of selected features
            for f in features:  # Iterate over each feature in the provided list
                if f in feature_names:  # Verify if the feature exists in the full feature list
                    indices.append(feature_names.index(f))  # Append the index of the feature
                    selected_names.append(f)  # Append the name of the feature
            return X_scaled[:, indices], selected_names  # Return the subset and actual names
        else:  # If no features are selected (or features is None)
            return np.empty((X_scaled.shape[0], 0)), []  # Return empty array and empty list
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def truncate_value(value):
    """
    Format a numeric value to 4 decimal places, or return None if not possible.
    
    :param value: Numeric value
    :return: Formatted string or None
    """
    
    try:
        try:  # Try to format the value
            if value is None:  # If value is None
                return None  # Return None
            v = float(value)  # Convert to float
            truncated = math.trunc(v * 10000) / 10000.0  # Truncate to 4 decimal places
            return f"{truncated:.4f}"  # Return formatted string
        except Exception:  # On failure
            return None  # Return None
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def export_model_and_scaler(model, scaler, dataset_name, model_name, feature_names, best_params=None, feature_set=None, dataset_csv_path=None, config=None):
    """
    Export model, scaler and metadata for stacking evaluations.
    
    :param model: Trained model to export
    :param scaler: Fitted scaler to export
    :param dataset_name: Name of dataset
    :param model_name: Name of model
    :param feature_names: List of feature names
    :param best_params: Best parameters from hyperparameter optimization
    :param feature_set: Feature set name (GA, RFE, PCA, etc.)
    :param dataset_csv_path: Path to dataset CSV file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        def safe_filename(name):
            return re.sub(r'[\\/*?:"<>|]', "_", str(name))

        if not dataset_csv_path:
            raise ValueError("dataset_csv_path is required to safely export models")

        file_path_obj = Path(dataset_csv_path)
        stacking_output_dir = get_stacking_output_dir(str(file_path_obj), config)

        export_dir = Path(stacking_output_dir) / "Models" / safe_filename(dataset_name)
        os.makedirs(export_dir, exist_ok=True)
        param_str = "_".join(f"{k}-{v}" for k, v in sorted(best_params.items())) if best_params else ""
        param_str = safe_filename(param_str)[:64]
        features_str = safe_filename("_".join(feature_names))[:64] if feature_names else "all_features"
        fs = safe_filename(feature_set) if feature_set else "all"
        base_name = f"{safe_filename(model_name)}__{fs}__{features_str}__{param_str}"
        model_path = os.path.join(str(export_dir), f"{base_name}_model.joblib")
        scaler_path = os.path.join(str(export_dir), f"{base_name}_scaler.joblib")
        try:
            validate_output_path(stacking_output_dir, model_path)
            dump(model, model_path)
            if scaler is not None:
                validate_output_path(stacking_output_dir, scaler_path)
                dump(scaler, scaler_path)
            meta = {
                "model_name": model_name,
                "feature_set": feature_set,
                "features": feature_names,
                "params": best_params,
            }
            meta_path = os.path.join(str(export_dir), f"{base_name}_meta.json")
            validate_output_path(stacking_output_dir, meta_path)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            verbose_output(f"Exported model to {model_path} and scaler to {scaler_path}")
        except Exception as e:
            print(f"{BackgroundColors.YELLOW}Warning: Failed to export model {model_name}: {e}{Style.RESET_ALL}")
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def compute_fpr_fnr(y_test, y_pred):
    """
    Computes the False Positive Rate and False Negative Rate from predictions.

    :param y_test: True target labels
    :param y_pred: Predicted target labels
    :return: Tuple (fpr, fnr) with 0.0 placeholders for multi-class problems
    """

    try:
        if len(np.unique(y_test)) == 2:  # Binary classification problem
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()  # Extract confusion matrix components
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # Calculate False Positive Rate with zero-division guard
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # Calculate False Negative Rate with zero-division guard
        else:  # Multi-class problem
            fpr = 0.0  # Use placeholder for multi-class FPR
            fnr = 0.0  # Use placeholder for multi-class FNR
        return (fpr, fnr)  # Return the FPR and FNR tuple
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def load_existing_model_if_available(model, model_name, dataset_file, feature_set, X_test, y_test, scaler, config=None):
    """
    Attempts to load a previously trained model from disk and evaluate it, skipping retraining.

    :param model: The classifier model object used as fallback when no saved model is found
    :param model_name: Name of the classifier for file pattern matching
    :param dataset_file: Path to the dataset file used to locate the saved models directory
    :param feature_set: Feature set name used to narrow the saved model file search
    :param X_test: Testing features for evaluation (numpy array)
    :param y_test: Testing target labels for metric computation
    :param scaler: Optional scaler object to load alongside the model from disk
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Metrics tuple (acc, prec, rec, f1, fpr, fnr, 0.0) if a model was loaded, else None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        models_dir = Path(dataset_file).parent / "Classifiers" / "Models"  # Build models directory path from dataset file location
        if not models_dir.exists():  # If models directory does not exist on disk
            return None  # No saved models available, signal caller to retrain

        safe_model = re.sub(r'[\\/*?:"<>|]', "_", str(model_name))  # Sanitize model name for filesystem pattern matching
        pattern = f"*{safe_model}*"  # Build default glob pattern using sanitized model name

        if feature_set:  # If feature set name was provided for narrowing the search
            safe_fs = re.sub(r'[\\/*?:"<>|]', "_", str(feature_set))  # Sanitize feature set name for filesystem pattern
            pattern = f"*{safe_model}*{safe_fs}*"  # Build narrowed pattern combining model and feature set names

        matches = list(models_dir.glob(f"{pattern}_model.joblib"))  # Search for matching model files using the pattern
        if not matches:  # If no matching model file found on disk
            return None  # No saved model available, signal caller to retrain

        try:  # Attempt to load the found model and evaluate it
            loaded = load(str(matches[0]))  # Load the serialized model from the first match
            model = loaded  # Replace in-memory model with the loaded one
            scaler_path = str(matches[0]).replace("_model.joblib", "_scaler.joblib")  # Build expected scaler path from model path
            if os.path.exists(scaler_path):  # If scaler file exists alongside the model
                scaler = load(scaler_path)  # Load the serialized scaler from disk
            verbose_output(f"Loaded existing model from {matches[0]}")  # Log that an existing model was loaded
            y_pred = model.predict(X_test)  # Predict labels using the loaded model on test features
            elapsed_time = 0.0  # Set elapsed time to zero since retraining was skipped
            acc = accuracy_score(y_test, y_pred)  # Calculate accuracy score on test predictions
            prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate weighted precision score
            rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate weighted recall score
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate weighted F1 score
            fpr, fnr = compute_fpr_fnr(y_test, y_pred)  # Compute false positive and false negative rates
            return (acc, prec, rec, f1, fpr, fnr, elapsed_time)  # Return the metrics tuple with zero elapsed time
        except Exception:  # If loading or evaluation fails
            verbose_output(f"Failed to load existing model for {model_name}; retraining.")  # Log fallback to retraining
            return None  # Signal caller to retrain from scratch
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def evaluate_individual_classifier(model, model_name, X_train, y_train, X_test, y_test, dataset_file=None, scaler=None, feature_names=None, feature_set=None, config=None):
    """
    Trains an individual classifier and evaluates its performance on the test set.

    :param model: The classifier model object to train.
    :param model_name: Name of the classifier (for logging).
    :param X_train: Training features (scaled numpy array).
    :param y_train: Training target labels (encoded Series/array).
    :param X_test: Testing features (scaled numpy array).
    :param y_test: Testing target labels (encoded Series/array).
    :param dataset_file: Path to dataset file
    :param scaler: Scaler object
    :param feature_names: List of feature names
    :param feature_set: Feature set name
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Metrics tuple (acc, prec, rec, f1, fpr, fnr, elapsed_time)
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        skip_train_if_model_exists = config.get("execution", {}).get("skip_train_if_model_exists", False)  # Get skip train flag from config

        verbose_output(
            f"{BackgroundColors.GREEN}Training {BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN}...{Style.RESET_ALL}",
            config=config,
        )  # Output the verbose message

        start_time = time.time()  # Record the start time

        if dataset_file is not None and skip_train_if_model_exists:  # If dataset file provided and skip-train flag enabled
            existing_metrics = load_existing_model_if_available(model, model_name, dataset_file, feature_set, X_test, y_test, scaler, config=config)  # Attempt to load a previously saved model from disk
            if existing_metrics is not None:  # If a valid existing model was found and used
                return existing_metrics  # Return cached metrics without retraining

        sys.stdout.flush()  # Flush stdout before model training to ensure logs are visible under nohup
        model.fit(X_train, y_train)  # Fit the model on the training data using its internal n_jobs parallelism

        y_pred = model.predict(X_test)  # Predict the labels for the test set

        elapsed_time = time.time() - start_time  # Calculate the total time elapsed

        acc = accuracy_score(y_test, y_pred)  # Calculate Accuracy
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate Precision
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate Recall
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate F1-Score

        fpr, fnr = compute_fpr_fnr(y_test, y_pred)  # Compute False Positive and False Negative rates

        verbose_output(
            f"{BackgroundColors.GREEN}{model_name} Accuracy: {BackgroundColors.CYAN}{truncate_value(acc)}{BackgroundColors.GREEN}, Time: {BackgroundColors.CYAN}{int(round(elapsed_time))}s{Style.RESET_ALL}"
        )  # Output result

        try:
            if dataset_file is not None:
                dataset_name = os.path.basename(os.path.dirname(dataset_file))
                export_model_and_scaler(model, scaler, dataset_name, model_name, feature_names or [], best_params=None, feature_set=feature_set, dataset_csv_path=dataset_file)
        except Exception:
            pass

        return (acc, prec, rec, f1, fpr, fnr, int(round(elapsed_time)))  # Return the metrics tuple
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def evaluate_stacking_classifier(model, X_train, y_train, X_test, y_test):
    """
    Trains the StackingClassifier model and evaluates its performance on the test set.

    :param model: The fitted StackingClassifier model object.
    :param X_train: Training features (pandas DataFrame or numpy array with feature names).
    :param y_train: Training target labels (encoded Series/array).
    :param X_test: Testing features (pandas DataFrame or numpy array with feature names).
    :param y_test: Testing target labels (encoded Series/array).
    :return: Metrics tuple (acc, prec, rec, f1, fpr, fnr, elapsed_time)
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Starting training and evaluation of Stacking Classifier...{Style.RESET_ALL}"
        )  # Output the verbose message

        start_time = time.time()  # Record the start time for timing training and prediction

        sys.stdout.flush()  # Flush stdout before stacking training to ensure logs are visible under nohup
        model.fit(X_train, y_train)  # Fit the stacking model on the training data (accepts DataFrame or array)

        y_pred = model.predict(X_test)  # Predict the labels for the test set

        elapsed_time = time.time() - start_time  # Calculate the total time elapsed

        acc = accuracy_score(y_test, y_pred)  # Calculate Accuracy
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate Precision (weighted)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate Recall (weighted)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate F1-Score (weighted)

        fpr, fnr = compute_fpr_fnr(y_test, y_pred)  # Compute False Positive and False Negative rates
        if len(np.unique(y_test)) != 2:  # For multi-class problems (FPR/FNR set to 0.0 placeholders)
            print(
                f"{BackgroundColors.YELLOW}Warning: Multi-class FPR/FNR calculation simplified to 0.0.{Style.RESET_ALL}"
            )  # Warning about simplification

        verbose_output(
            f"{BackgroundColors.GREEN}Evaluation complete. Accuracy: {BackgroundColors.CYAN}{truncate_value(acc)}{BackgroundColors.GREEN}, Time: {BackgroundColors.CYAN}{int(round(elapsed_time))}s{Style.RESET_ALL}"
        )  # Output the final result summary

        return (acc, prec, rec, f1, fpr, fnr, int(round(elapsed_time)), y_pred)  # Return the metrics tuple and predictions
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def sample_shap_test_data(X_test, y_test, max_samples, random_state):
    """
    Samples test data for SHAP computation when the test set exceeds the maximum sample size.

    :param X_test: Test features array to sample from
    :param y_test: Test labels array or Series to sample from
    :param max_samples: Maximum number of samples to use for SHAP computation
    :param random_state: Random seed for reproducible sampling
    :return: Tuple (X_test_sampled, y_test_sampled) with sampled or full test data
    """

    try:
        if len(X_test) > max_samples:  # If test set exceeds the sample limit
            np.random.seed(random_state)  # Set random seed for reproducible sampling
            sample_indices = np.random.choice(len(X_test), size=max_samples, replace=False)  # Draw random sample indices
            X_test_sampled = X_test[sample_indices]  # Slice test features to sampled indices
            y_test_sampled = y_test.iloc[sample_indices] if hasattr(y_test, 'iloc') else y_test[sample_indices]  # Slice test labels using iloc for Series or index for arrays
        else:  # Test set is within the sample limit
            X_test_sampled = X_test  # Use full test features without sampling
            y_test_sampled = y_test  # Use full test labels without sampling
        return (X_test_sampled, y_test_sampled)  # Return the sampled or full test data tuple
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def aggregate_mean_shap_importance(shap_values_summary, feature_names):
    """
    Computes mean absolute SHAP values and maps them to feature names.

    :param shap_values_summary: SHAP values array (2D: samples x features)
    :param feature_names: List of feature names corresponding to the SHAP value columns
    :return: Dictionary mapping each feature name to its mean absolute SHAP importance
    """

    try:
        shap_array = np.array(shap_values_summary)  # Convert SHAP values to numpy array for consistent operations
        mean_shap_values = np.mean(np.abs(shap_array), axis=0)  # Compute mean absolute SHAP value per feature across samples
        mean_shap_list = mean_shap_values.tolist() if hasattr(mean_shap_values, 'tolist') else list(mean_shap_values)  # Convert numpy array to plain Python list
        shap_importance = dict(zip(feature_names[:len(mean_shap_list)], mean_shap_list))  # Map feature names to their mean absolute importance values
        return shap_importance  # Return importance dictionary for downstream use
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def save_shap_summary_and_bar_plots(shap_values_summary, X_test_sampled, feature_names, output_dir, dataset_name, model_name, max_display):
    """
    Saves SHAP summary and bar plots to the output directory.

    :param shap_values_summary: SHAP values array used for plot generation
    :param X_test_sampled: Sampled test features for plot background data
    :param feature_names: List of feature names for axis labels
    :param output_dir: Directory path where plots will be saved
    :param dataset_name: Dataset name used in output filenames
    :param model_name: Model name used in output filenames
    :param max_display: Maximum number of features to display in each plot
    :return: None
    """

    try:
        try:  # Attempt to create SHAP summary plot
            plt.figure()  # Create new figure for summary plot
            shap.summary_plot(shap_values_summary, X_test_sampled, feature_names=feature_names[:len(feature_names)], max_display=max_display, show=False)  # Create summary plot
            summary_plot_path = os.path.join(output_dir, f"{dataset_name}_{model_name}_shap_summary.png")  # Build summary plot file path
            plt.tight_layout()  # Adjust layout for tight fit
            ensure_figure_min_4k_and_save(fig=plt.gcf(), path=summary_plot_path, dpi=300, bbox_inches='tight')  # Save with minimum 4K resolution
            plt.close()  # Close summary figure
        except Exception:  # If summary plot generation fails
            plt.close()  # Close figure to avoid resource leak

        try:  # Attempt to create SHAP bar plot
            plt.figure()  # Create new figure for bar plot
            shap.summary_plot(shap_values_summary, X_test_sampled, feature_names=feature_names[:len(feature_names)], max_display=max_display, plot_type="bar", show=False)  # Create bar plot
            bar_plot_path = os.path.join(output_dir, f"{dataset_name}_{model_name}_shap_bar.png")  # Build bar plot file path
            plt.tight_layout()  # Adjust layout for tight fit
            ensure_figure_min_4k_and_save(fig=plt.gcf(), path=bar_plot_path, dpi=300, bbox_inches='tight')  # Save with minimum 4K resolution
            plt.close()  # Close bar figure
        except Exception:  # If bar plot generation fails
            plt.close()  # Close figure to avoid resource leak
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def select_shap_explainer(model, X_test_sampled, random_state):
    """
    Selects and instantiates the appropriate SHAP explainer based on model type.

    :param model: Trained model object for which to build the explainer
    :param X_test_sampled: Sampled test features used for KernelExplainer background data
    :param random_state: Random seed used for KernelExplainer background sampling
    :return: Instantiated SHAP explainer object
    """

    try:
        model_type = model.__class__.__name__  # Get model class name for branch selection
        if model_type in ["RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier", "LightGBMClassifier", "ExtraTreesClassifier"]:  # Tree-based models
            return shap.TreeExplainer(model)  # Use TreeExplainer for tree-based models
        elif model_type in ["LogisticRegression", "LinearSVC", "SGDClassifier"]:  # Linear models
            return shap.LinearExplainer(model, X_test_sampled)  # Use LinearExplainer for linear models
        else:  # Other models that require a fallback explainer
            return shap.KernelExplainer(model.predict_proba, shap.sample(X_test_sampled, 50, random_state=random_state))  # Use KernelExplainer as fallback
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def generate_shap_explanations(model, X_test, y_test, feature_names, output_dir, model_name, dataset_name, execution_mode, config=None):
    """
    Generate SHAP explanations for a trained model.

    :param model: Trained model object
    :param X_test: Test features (numpy array or DataFrame)
    :param y_test: Test labels
    :param feature_names: List of feature names
    :param output_dir: Directory to save SHAP outputs
    :param model_name: Name of the model for labeling
    :param dataset_name: Name of the dataset
    :param execution_mode: Execution mode string ('binary' or 'multi-class')
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary with SHAP values and summary metrics or None if failed
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        try:  # Attempt to generate SHAP explanations
            verbose_output(
                f"{BackgroundColors.GREEN}Generating SHAP explanations for {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}",
                config=config
            )  # Log SHAP generation start

            shap_config = config.get("explainability", {})  # Get explainability config
            max_samples = shap_config.get("shap_max_samples", 100)  # Max samples for SHAP computation
            max_display = shap_config.get("max_display_features", 20)  # Max features to display
            random_state = shap_config.get("random_state", 42)  # Random state for sampling

            os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

            X_test_sampled, y_test_sampled = sample_shap_test_data(X_test, y_test, max_samples, random_state)  # Sample test data for SHAP computation

            explainer = select_shap_explainer(model, X_test_sampled, random_state)  # Select the appropriate SHAP explainer based on model type

            shap_values = explainer.shap_values(X_test_sampled)  # Compute SHAP values

            if isinstance(shap_values, list):  # Multi-class case
                shap_values_summary = shap_values[0] if len(shap_values) > 0 else shap_values  # Use first class for summary
            else:  # Binary or regression case
                shap_values_summary = shap_values  # Use SHAP values directly

            save_shap_summary_and_bar_plots(shap_values_summary, X_test_sampled, feature_names, output_dir, dataset_name, model_name, max_display)  # Save both SHAP summary and bar plots to disk

            shap_importance = aggregate_mean_shap_importance(shap_values_summary, feature_names)  # Aggregate mean absolute SHAP values into a feature importance dictionary

            verbose_output(
                f"{BackgroundColors.GREEN}SHAP explanations saved to {BackgroundColors.CYAN}{output_dir}{Style.RESET_ALL}",
                config=config
            )  # Log SHAP completion

            return {"shap_importance": shap_importance, "shap_values": shap_values}  # Return SHAP results

        except ImportError:  # If SHAP not installed
            print(f"{BackgroundColors.YELLOW}SHAP library not installed. Skipping SHAP explanations. Install with: pip install shap{Style.RESET_ALL}")  # Warn user
            return None  # Return None
        except Exception as e:  # If any other error
            verbose_output(
                f"{BackgroundColors.YELLOW}Failed to generate SHAP explanations for {model_name}: {e}{Style.RESET_ALL}",
                config=config
            )  # Log error
            return None  # Return None
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def generate_lime_explanations(model, X_test, y_test, feature_names, output_dir, model_name, dataset_name, execution_mode, config=None):
    """
    Generate LIME explanations for a trained model.

    :param model: Trained model object
    :param X_test: Test features (numpy array or DataFrame)
    :param y_test: Test labels
    :param feature_names: List of feature names
    :param output_dir: Directory to save LIME outputs
    :param model_name: Name of the model for labeling
    :param dataset_name: Name of the dataset
    :param execution_mode: Execution mode string ('binary' or 'multi-class')
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary with LIME explanations or None if failed
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        try:  # Attempt to generate LIME explanations
            verbose_output(
                f"{BackgroundColors.GREEN}Generating LIME explanations for {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}",
                config=config
            )  # Log LIME generation start

            lime_config = config.get("explainability", {})  # Get explainability config
            num_features = lime_config.get("lime_num_features", 10)  # Number of features in explanation
            num_samples = lime_config.get("lime_num_samples", 1000)  # Number of samples for LIME
            random_state = lime_config.get("random_state", 42)  # Random state for sampling

            os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

            mode = "classification"  # Default mode
            class_names = [str(c) for c in np.unique(y_test)]  # Get class names

            explainer = LimeTabularExplainer(
                X_test,
                feature_names=feature_names[:X_test.shape[1]],
                class_names=class_names,
                mode=mode,
                random_state=random_state
            )  # Initialize LIME explainer

            np.random.seed(random_state)  # Set random seed
            num_instances_to_explain = min(5, len(X_test))  # Explain up to 5 instances
            instance_indices = np.random.choice(len(X_test), size=num_instances_to_explain, replace=False)  # Sample indices

            lime_explanations = []  # List to store LIME explanations

            for idx in instance_indices:  # For each instance to explain
                instance = X_test[idx]  # Get instance
                explanation = explainer.explain_instance(
                    instance,
                    model.predict_proba,
                    num_features=num_features,
                    num_samples=num_samples
                )  # Generate LIME explanation

                try:  # Try to save explanation figure
                    fig = explanation.as_pyplot_figure()  # Get matplotlib figure
                    explanation_plot_path = os.path.join(output_dir, f"{dataset_name}_{model_name}_lime_instance_{idx}.png")  # Build plot path
                    plt.tight_layout()  # Adjust layout
                    ensure_figure_min_4k_and_save(fig=plt.gcf(), path=explanation_plot_path, dpi=300, bbox_inches='tight')  # Save plot
                    plt.close()  # Close plot
                except Exception:  # If plot save fails
                    plt.close()  # Close plot

                lime_explanations.append(explanation.as_list())  # Store explanation as list

            verbose_output(
                f"{BackgroundColors.GREEN}LIME explanations saved to {BackgroundColors.CYAN}{output_dir}{Style.RESET_ALL}",
                config=config
            )  # Log LIME completion

            return {"lime_explanations": lime_explanations}  # Return LIME results

        except ImportError:  # If LIME not installed
            print(f"{BackgroundColors.YELLOW}LIME library not installed. Skipping LIME explanations. Install with: pip install lime{Style.RESET_ALL}")  # Warn user
            return None  # Return None
        except Exception as e:  # If any other error
            verbose_output(
                f"{BackgroundColors.YELLOW}Failed to generate LIME explanations for {model_name}: {e}{Style.RESET_ALL}",
                config=config
            )  # Log error
            return None  # Return None
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def generate_permutation_importance(model, X_test, y_test, feature_names, output_dir, model_name, dataset_name, config=None):
    """
    Generate permutation feature importance for a trained model.

    :param model: Trained model object
    :param X_test: Test features (numpy array or DataFrame)
    :param y_test: Test labels
    :param feature_names: List of feature names
    :param output_dir: Directory to save permutation importance outputs
    :param model_name: Name of the model for labeling
    :param dataset_name: Name of the dataset
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary with permutation importance or None if failed
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        try:  # Attempt to generate permutation importance
            verbose_output(
                f"{BackgroundColors.GREEN}Computing permutation importance for {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}",
                config=config
            )  # Log permutation importance computation start

            explainer_config = config.get("explainability", {})  # Get explainability config
            random_state = explainer_config.get("random_state", 42)  # Random state for permutation
            n_jobs = config.get("evaluation", {}).get("n_jobs", -1)  # Number of parallel jobs

            os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

            perm_importance = permutation_importance(
                model,
                X_test,
                y_test,
                n_repeats=10,
                random_state=random_state,
                n_jobs=n_jobs
            )  # Compute permutation importance

            importances_mean = perm_importance['importances_mean']  # Extract mean importances from Bunch
            importances_std = perm_importance['importances_std']  # Extract std importances from Bunch
            sorted_indices = importances_mean.argsort()[::-1]  # Sort indices by descending importance
            sorted_features = [feature_names[i] for i in sorted_indices]  # Get sorted feature names
            sorted_importances = importances_mean[sorted_indices]  # Get sorted importances
            sorted_std = importances_std[sorted_indices]  # Get sorted standard deviations

            importance_dict = {}  # Initialize importance dictionary
            for feat, imp, std in zip(sorted_features, sorted_importances, sorted_std):  # For each feature
                importance_dict[feat] = {"mean": float(imp), "std": float(std)}  # Store importance and std

            resolved_fig = None  # Initialize resolved_fig to None before plotting to ensure it's defined for cleanup
            try:  # Try to create bar plot
                max_display = explainer_config.get("max_display_features", 20)  # Max features to display
                display_count = min(max_display, len(sorted_features))  # Number of features to display

                resolved_fig = None  # Initialize resolved_fig to None for proper exception handling
                plt.figure(figsize=(10, max(6, display_count * 0.3)))  # Create figure with dynamic height
                resolved_fig = plt.gcf()  # Get the current figure to ensure we can close it in case of exceptions
                y_pos = np.arange(display_count)  # Y positions for bars
                plt.barh(y_pos, sorted_importances[:display_count], xerr=sorted_std[:display_count], align='center', alpha=0.7, color='steelblue')  # Create horizontal bar plot with error bars
                plt.yticks(y_pos, sorted_features[:display_count])  # Set Y-axis labels
                plt.xlabel('Permutation Importance', fontsize=12)  # Set X-axis label
                plt.ylabel('Features', fontsize=12)  # Set Y-axis label
                plt.title(f'Permutation Importance - {model_name}\n{dataset_name}', fontsize=14)  # Set title
                plt.grid(axis='x', alpha=0.3)  # Add X-axis grid
                plt.tight_layout()  # Adjust layout
                perm_plot_path = os.path.join(output_dir, f"{dataset_name}_{model_name}_permutation_importance.png")  # Build plot path
                try:  # Ensure figure has at least 4k pixels before saving
                    resolved_fig.savefig(perm_plot_path, dpi=300, bbox_inches='tight')  # Save plot
                finally:  # Attempt to close the exact figure to free memory
                    plt.close(resolved_fig)  # Close the exact figure to free memory
            except Exception:  # If plot fails
                try:  # Attempt to close any figure that might be open to free memory
                    if resolved_fig is not None:  # If we have a reference to the figure, close it
                        plt.close(resolved_fig)  # Close the exact figure to free memory
                except Exception:  # If closing the figure fails, just pass
                    pass  # If plot creation or saving fails, we ignore the error and continue without the plot

            verbose_output(
                f"{BackgroundColors.GREEN}Permutation importance saved to {BackgroundColors.CYAN}{output_dir}{Style.RESET_ALL}",
                config=config
            )  # Log permutation importance completion

            return {"permutation_importance": importance_dict}  # Return permutation importance results

        except Exception as e:  # If any error
            verbose_output(
                f"{BackgroundColors.YELLOW}Failed to compute permutation importance for {model_name}: {e}{Style.RESET_ALL}",
                config=config
            )  # Log error
            return None  # Return None
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_linear_model_importance(model, feature_names, output_dir, model_name, dataset_name, config=None):
    """
    Extracts feature importances from linear models using the coef_ attribute as a proxy for importance.

    :param model: Trained linear model with a coef_ attribute
    :param feature_names: List of feature names corresponding to model input columns
    :param output_dir: Directory path where output artifacts will be created
    :param model_name: Name of the model used in log messages and file names
    :param dataset_name: Name of the dataset used in log messages and file names
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary with key 'model_importance' mapping feature names to absolute coefficient scores
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Extracting coefficients as feature importance for {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}",
            config=config
        )  # Log linear model importance extraction start

        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists before writing

        coef = model.coef_  # Read coefficient array from the linear model
        if len(coef.shape) > 1:  # Multi-class models have a 2D coefficient matrix
            importances = np.abs(coef).mean(axis=0)  # Average absolute coefficients across all classes
        else:  # Binary classification models have a 1D coefficient array
            importances = np.abs(coef[0] if coef.ndim > 1 else coef)  # Use absolute values of the single coefficient vector

        sorted_indices = np.argsort(importances)[::-1]  # Sort indices from highest to lowest absolute coefficient
        sorted_features = [feature_names[i] for i in sorted_indices]  # Reorder feature names by descending importance
        sorted_importances = importances[sorted_indices]  # Reorder importance values by descending importance

        importance_dict = dict(zip(sorted_features, sorted_importances.tolist()))  # Build feature-name-to-score mapping

        verbose_output(
            f"{BackgroundColors.GREEN}Model coefficients saved as importance for {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}",
            config=config
        )  # Log successful extraction completion

        return {"model_importance": importance_dict}  # Return the importance dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_tree_based_importance(model, feature_names, output_dir, model_name, dataset_name, config=None):
    """
    Extracts and plots feature importances from tree-based models using the feature_importances_ attribute.

    :param model: Trained tree-based model with a feature_importances_ attribute
    :param feature_names: List of feature names corresponding to model input columns
    :param output_dir: Directory path where the importance bar plot will be saved
    :param model_name: Name of the model used in plot titles and file names
    :param dataset_name: Name of the dataset used in plot titles and file names
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary with key 'model_importance' mapping feature names to importance scores
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Extracting built-in feature importance for {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}",
            config=config
        )  # Log tree-based importance extraction start

        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists before writing

        importances = model.feature_importances_  # Read raw importance scores from the model
        sorted_indices = np.argsort(importances)[::-1]  # Sort indices from highest to lowest importance
        sorted_features = [feature_names[i] for i in sorted_indices]  # Reorder feature names by descending importance
        sorted_importances = importances[sorted_indices]  # Reorder importance values by descending importance

        importance_dict = dict(zip(sorted_features, sorted_importances.tolist()))  # Build feature-name-to-score mapping

        try:  # Attempt to generate the importance bar plot
            explainer_config = config.get("explainability", {})  # Read explainability section from config
            max_display = explainer_config.get("max_display_features", 20)  # Maximum features shown in the plot
            display_count = min(max_display, len(sorted_features))  # Clamp display count to available features

            plt.figure(figsize=(10, max(6, display_count * 0.3)))  # Create figure sized relative to feature count
            y_pos = np.arange(display_count)  # Compute bar Y-axis positions
            plt.barh(y_pos, sorted_importances[:display_count], align='center', alpha=0.7, color='forestgreen')  # Draw horizontal bars
            plt.yticks(y_pos, sorted_features[:display_count])  # Label Y-axis with feature names
            plt.xlabel('Feature Importance', fontsize=12)  # Label X-axis
            plt.ylabel('Features', fontsize=12)  # Label Y-axis
            plt.title(f'Model Feature Importance - {model_name}\n{dataset_name}', fontsize=14)  # Set descriptive title
            plt.grid(axis='x', alpha=0.3)  # Add subtle X-axis grid lines
            plt.tight_layout()  # Adjust figure padding
            importance_plot_path = os.path.join(output_dir, f"{dataset_name}_{model_name}_feature_importance.png")  # Construct output file path
            ensure_figure_min_4k_and_save(fig=plt.gcf(), path=importance_plot_path, dpi=300, bbox_inches='tight')  # Save plot with minimum 4K resolution
            plt.close()  # Release figure resources
        except Exception:  # If plot generation fails
            plt.close()  # Close figure to avoid resource leak

        verbose_output(
            f"{BackgroundColors.GREEN}Model feature importance saved to {BackgroundColors.CYAN}{output_dir}{Style.RESET_ALL}",
            config=config
        )  # Log successful extraction and plot save

        return {"model_importance": importance_dict}  # Return the importance dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_model_feature_importance(model, feature_names, output_dir, model_name, dataset_name, config=None):
    """
    Extract built-in feature importance from model if supported.

    :param model: Trained model object
    :param feature_names: List of feature names
    :param output_dir: Directory to save feature importance outputs
    :param model_name: Name of the model for labeling
    :param dataset_name: Name of the dataset
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary with feature importance or None if not supported
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        try:  # Attempt to extract feature importance
            model_type = model.__class__.__name__  # Get model class name

            if hasattr(model, 'feature_importances_'):  # If model has built-in feature importance
                return extract_tree_based_importance(model, feature_names, output_dir, model_name, dataset_name, config=config)  # Delegate to tree-based importance extractor

            elif hasattr(model, 'coef_'):  # If linear model with coefficients
                return extract_linear_model_importance(model, feature_names, output_dir, model_name, dataset_name, config=config)  # Delegate to linear coefficient importance extractor

            else:  # Model does not support feature importance
                verbose_output(
                    f"{BackgroundColors.YELLOW}Model {model_name} does not support built-in feature importance{Style.RESET_ALL}",
                    config=config
                )  # Log unsupported model
                return None  # Return None

        except Exception as e:  # If any error
            verbose_output(
                f"{BackgroundColors.YELLOW}Failed to extract feature importance for {model_name}: {e}{Style.RESET_ALL}",
                config=config
            )  # Log error
            return None  # Return None
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def generate_combined_importance_report(shap_result, lime_result, perm_result, model_result, feature_names, output_dir, model_name, dataset_name, config=None):
    """
    Generate a combined importance report aggregating all explainability methods.

    :param shap_result: SHAP results dictionary or None
    :param lime_result: LIME results dictionary or None
    :param perm_result: Permutation importance results dictionary or None
    :param model_result: Model feature importance results dictionary or None
    :param feature_names: List of feature names
    :param output_dir: Directory to save combined report
    :param model_name: Name of the model for labeling
    :param dataset_name: Name of the dataset
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Path to saved report or None if failed
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        try:  # Attempt to generate combined report
            verbose_output(
                f"{BackgroundColors.GREEN}Generating combined importance report for {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}",
                config=config
            )  # Log report generation start

            os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

            report_data = []  # List to store report rows

            for feature in feature_names:  # For each feature
                row = {"feature": feature}  # Initialize row with feature name

                if shap_result and "shap_importance" in shap_result:  # If SHAP results available
                    row["shap_importance"] = shap_result["shap_importance"].get(feature, 0.0)  # Get SHAP importance
                else:  # If SHAP not available
                    row["shap_importance"] = np.nan  # Set to NaN

                if perm_result and "permutation_importance" in perm_result:  # If permutation results available
                    perm_dict = perm_result["permutation_importance"].get(feature, {})  # Get permutation dict for feature
                    row["permutation_importance_mean"] = perm_dict.get("mean", np.nan)  # Get mean importance
                    row["permutation_importance_std"] = perm_dict.get("std", np.nan)  # Get std

                else:  # If permutation not available
                    row["permutation_importance_mean"] = np.nan  # Set to NaN
                    row["permutation_importance_std"] = np.nan  # Set to NaN

                if model_result and "model_importance" in model_result:  # If model importance available
                    row["model_importance"] = model_result["model_importance"].get(feature, 0.0)  # Get model importance
                else:  # If model importance not available
                    row["model_importance"] = np.nan  # Set to NaN

                report_data.append(row)  # Add row to report data

            report_df = pd.DataFrame(report_data)  # Create DataFrame from report data

            importance_cols = [col for col in report_df.columns if col != "feature" and "std" not in col]  # Get importance columns
            if len(importance_cols) >= 2:  # If at least 2 importance methods available
                for col in importance_cols:  # For each importance column
                    report_df[f"{col}_rank"] = report_df[col].rank(ascending=False, na_option='bottom')  # Compute rank

                rank_cols = [f"{col}_rank" for col in importance_cols]  # Get rank column names
                report_df["average_rank"] = report_df[rank_cols].mean(axis=1)  # Compute average rank
                report_df["rank_std"] = report_df[rank_cols].std(axis=1)  # Compute rank standard deviation
                report_df["consistency_score"] = 1.0 / (1.0 + report_df["rank_std"])  # Compute consistency score (higher = more consistent)

                report_df = report_df.sort_values("average_rank")  # Sort by average rank

            report_path = os.path.join(output_dir, f"{dataset_name}_{model_name}_combined_importance.csv")  # Build report path
            generate_csv_and_image(report_df, report_path, is_visualizable=True, index=False)  # Save report CSV and generate PNG image

            verbose_output(
                f"{BackgroundColors.GREEN}Combined importance report saved to {BackgroundColors.CYAN}{report_path}{Style.RESET_ALL}",
                config=config
            )  # Log report completion

            return report_path  # Return path to saved report

        except Exception as e:  # If any error
            verbose_output(
                f"{BackgroundColors.YELLOW}Failed to generate combined importance report for {model_name}: {e}{Style.RESET_ALL}",
                config=config
            )  # Log error
            return None  # Return None
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_explainability_pipeline(model, model_name, X_test, y_test, feature_names, dataset_file, feature_set, execution_mode, config=None):
    """
    Run comprehensive explainability pipeline for a trained model.

    Generates SHAP, LIME, permutation importance, and combined reports.
    Only runs on ORIGINAL test data, never on augmented samples.

    :param model: Trained model object
    :param model_name: Name of the model for labeling
    :param X_test: Original test features (numpy array, NOT augmented)
    :param y_test: Original test labels (NOT augmented)
    :param feature_names: List of feature names
    :param dataset_file: Path to dataset file for output directory construction
    :param feature_set: Feature set name (e.g., "Full Features", "GA Features")
    :param execution_mode: Execution mode string ('binary' or 'multi-class')
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary with all explainability results or None if disabled
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        explainability_config = config.get("explainability", {})  # Get explainability config
        if not explainability_config.get("enabled", False):  # If explainability is disabled
            return None  # Return None immediately

        verbose_output(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Running explainability pipeline for {BackgroundColors.CYAN}{model_name} - {feature_set}{Style.RESET_ALL}",
            config=config
        )  # Log pipeline start

        dataset_name = Path(dataset_file).stem  # Get dataset name from file path
        output_subdir = explainability_config.get("output_subdir", "explainability")  # Get output subdirectory name
        stacking_output_dir = get_stacking_output_dir(dataset_file, config)
        base_output_dir = Path(stacking_output_dir) / output_subdir / execution_mode / dataset_name
        output_dir = base_output_dir / feature_set.replace(" ", "_") / model_name.replace(" ", "_")  # Build full output directory
        output_dir = str(output_dir)  # Convert Path to string

        all_results = {}  # Dictionary to store all explainability results

        if explainability_config.get("shap", True):  # If SHAP is enabled
            shap_result = generate_shap_explanations(
                model, X_test, y_test, feature_names, output_dir, model_name, dataset_name, execution_mode, config
            )  # Generate SHAP explanations
            if shap_result:  # If SHAP results available
                all_results.update(shap_result)  # Add SHAP results to all results

        if explainability_config.get("lime", True):  # If LIME is enabled
            lime_result = generate_lime_explanations(
                model, X_test, y_test, feature_names, output_dir, model_name, dataset_name, execution_mode, config
            )  # Generate LIME explanations
            if lime_result:  # If LIME results available
                all_results.update(lime_result)  # Add LIME results to all results

        if explainability_config.get("permutation_importance", True):  # If permutation importance is enabled
            perm_result = generate_permutation_importance(
                model, X_test, y_test, feature_names, output_dir, model_name, dataset_name, config
            )  # Generate permutation importance
            if perm_result:  # If permutation results available
                all_results.update(perm_result)  # Add permutation results to all results

        if explainability_config.get("feature_importance", True):  # If feature importance extraction is enabled
            model_result = extract_model_feature_importance(
                model, feature_names, output_dir, model_name, dataset_name, config
            )  # Extract model feature importance
            if model_result:  # If model importance available
                all_results.update(model_result)  # Add model importance to all results

        shap_res = all_results if "shap_importance" in all_results else None  # Get SHAP results or None
        lime_res = all_results if "lime_explanations" in all_results else None  # Get LIME results or None
        perm_res = all_results if "permutation_importance" in all_results else None  # Get permutation results or None
        model_res = all_results if "model_importance" in all_results else None  # Get model importance or None

        report_path = generate_combined_importance_report(
            shap_res, lime_res, perm_res, model_res, feature_names, output_dir, model_name, dataset_name, config
        )  # Generate combined report
        if report_path:  # If report generated successfully
            all_results["combined_report_path"] = report_path  # Add report path to results

        verbose_output(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Explainability pipeline completed for {BackgroundColors.CYAN}{model_name} - {feature_set}{Style.RESET_ALL}",
            config=config
        )  # Log pipeline completion

        return all_results  # Return all explainability results
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_hardware_specifications():
    """
    Returns system specs: real CPU model (Windows/Linux/macOS), physical cores,
    RAM in GB, and OS name/version.

    :return: Dictionary with keys: cpu_model, cores, ram_gb, os
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Fetching system specifications...{Style.RESET_ALL}"
        )  # Output the verbose message

        system = platform.system()  # Identify OS type

        try:  # Try to fetch real CPU model using OS-specific methods
            if system == "Windows":  # Windows: use WMIC
                out = subprocess.check_output("wmic cpu get Name", shell=True).decode(errors="ignore")  # Run WMIC
                cpu_model = out.strip().split("\n")[1].strip()  # Extract model line

            elif system == "Linux":  # Linux: read from /proc/cpuinfo
                cpu_model = "Unknown"  # Default
                with open("/proc/cpuinfo") as f:  # Open cpuinfo
                    for line in f:  # Iterate lines
                        if "model name" in line:  # Model name entry
                            cpu_model = line.split(":", 1)[1].strip()  # Extract name
                            break  # Stop after first match

            elif system == "Darwin":  # MacOS: use sysctl
                out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])  # Run sysctl
                cpu_model = out.decode().strip()  # Extract model string

            else:  # Unsupported OS
                cpu_model = "Unknown"  # Fallback

        except Exception:  # If any method fails
            cpu_model = "Unknown"  # Fallback on failure

        cores = psutil.cpu_count(logical=False)  # Physical core count
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)  # Total RAM in GB
        os_name = f"{platform.system()} {platform.release()}"  # OS name + version

        return {  # Build final dictionary
            "cpu_model": cpu_model,  # CPU model string
            "cores": cores,  # Physical cores
            "ram_gb": ram_gb,  # RAM in gigabytes
            "os": os_name,  # Operating system
        }
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def add_hardware_column(df, columns_order, column_name="Hardware"):
    """
    Ensure a hardware description column named by `column_name` exists on `df`
    and insert it into `columns_order` immediately after `elapsed_time_s`.

    :param df: pandas.DataFrame to mutate in-place
    :param columns_order: list of canonical column names to update
    :param column_name: hardware column name to add (default: "Hardware")
    :return: None
    """
    
    try:
        try:  # Try to get hardware specifications
            hardware_specs = get_hardware_specifications()  # Get system specs
            df[column_name] = (
                hardware_specs["cpu_model"]
                + " | Cores: "
                + str(hardware_specs["cores"])
                + " | RAM: "
                + str(hardware_specs["ram_gb"])
                + " GB | OS: "
                + hardware_specs["os"]
            )  # Add hardware specs column
        except Exception:  # If fetching specs fails
            df[column_name] = None  # Add column with None values

        if column_name not in columns_order:  # If the hardware column is not already in the order list
            insert_idx = (
                (columns_order.index("elapsed_time_s") + 1) if "elapsed_time_s" in columns_order else len(columns_order)
            )  # Determine insertion index
            columns_order.insert(insert_idx, column_name)  # Insert hardware column into the desired position
            
        return df  # Return the modified DataFrame
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def export_feature_artifacts(df, file_path_obj, stacking_dir, config=None):
    """
    Aggregates feature usage statistics, exports the top-features CSV, and generates the heatmap PNG.

    :param df: DataFrame containing stacking results with feature columns
    :param file_path_obj: Path object for the original CSV file used to derive base filenames
    :param stacking_dir: Path object for the Stacking output directory where artifacts are saved
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        try:  # Attempt feature usage aggregation
            feature_counts_df = aggregate_feature_usage(df, None)  # Aggregate feature usage from results DataFrame (None uses config)
        except ValueError as ve:
            print(f"{BackgroundColors.YELLOW}No features to aggregate: {ve}{Style.RESET_ALL}")  # Warn when no feature columns are present
            feature_counts_df = pd.DataFrame()  # Provide empty DataFrame to proceed without crashing
        except Exception as e:
            print(f"{BackgroundColors.RED}Feature aggregation failed: {e}{Style.RESET_ALL}")  # Log aggregation failure for debugging
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Send aggregation error via Telegram
            feature_counts_df = pd.DataFrame()  # Provide empty DataFrame to allow downstream code to continue

        try:  # Attempt to export top-features CSV and heatmap
            dataset_base = file_path_obj.stem  # Extract stem of filename for output file naming
            top_csv = stacking_dir / f"{dataset_base}_top_features.csv"  # Build path for top-features CSV
            top_png = stacking_dir / f"{dataset_base}_top_features.png"  # Build path for feature usage heatmap
            export_top_features_csv(feature_counts_df, str(top_csv), dataset_file=str(file_path_obj))  # Export aggregated feature counts to CSV
            generate_feature_usage_heatmap(feature_counts_df, str(top_png), dataset_file=str(file_path_obj))  # Generate heatmap PNG of feature usage frequencies
        except Exception as e:
            print(f"{BackgroundColors.RED}Failed to export top-features CSV or heatmap: {e}{Style.RESET_ALL}")  # Log export failure details
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Notify via Telegram but do not interrupt main flow
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def reorder_and_annotate_dataframe(df, config=None):
    """
    Reorders DataFrame columns according to configuration and appends the hardware column.

    :param df: DataFrame containing stacking results to reorder
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Reordered DataFrame with hardware column appended at the proper position
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        results_csv_columns = config.get("stacking", {}).get("results_csv_columns", [])  # Read desired column order from config
        column_order = list(results_csv_columns) if results_csv_columns else list(config.get("stacking", {}).get("results_csv_columns", []))  # Use config list as ordering reference

        existing_columns = [col for col in column_order if col in df.columns]  # Filter to only columns that exist in the DataFrame
        df = df[existing_columns + [c for c in df.columns if c not in existing_columns]]  # Reorder with configured columns first, then any remaining

        df = add_hardware_column(df, existing_columns)  # Append hardware annotation column at the configured position
        return df  # Return the reordered and annotated DataFrame
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def flatten_and_serialize_results(results_list):
    """
    Flattens result dictionaries, truncates numeric metrics, and JSON-serializes nested fields.

    :param results_list: List of result dictionaries from classifier evaluation
    :return: List of flattened and serialized row dictionaries ready for DataFrame construction
    """

    try:
        flat_rows = []  # Initialize list to collect flattened rows
        for res in results_list:  # Iterate over each result dictionary
            row = dict(res)  # Create a mutable shallow copy of the result dictionary

            for metric in ["accuracy", "precision", "recall", "f1_score", "fpr", "fnr"]:  # Iterate over numeric metric field names
                if metric in row and row[metric] is not None:  # If metric field is present and has a value
                    row[metric] = truncate_value(row[metric])  # Truncate to consistent decimal precision

            if "features_list" in row and not isinstance(row["features_list"], str):  # If features_list is not yet a JSON string
                row["features_list"] = json.dumps(row["features_list"])  # Serialize features list to JSON string
            if "top_features" in row and not isinstance(row["top_features"], str):  # If top_features is not yet a JSON string
                row["top_features"] = json.dumps(row["top_features"])  # Serialize top features to JSON string
            if "rfe_ranking" in row and row["rfe_ranking"] is not None and not isinstance(row["rfe_ranking"], str):  # If rfe_ranking is present and not yet serialized
                row["rfe_ranking"] = json.dumps(row["rfe_ranking"])  # Serialize RFE ranking to JSON string
            if "hyperparameters" in row and row["hyperparameters"] is not None and not isinstance(row["hyperparameters"], str):  # If hyperparameters is present and not yet serialized
                row["hyperparameters"] = json.dumps(row["hyperparameters"])  # Serialize hyperparameters dict to JSON string

            if "feature_selection_enabled" not in row:  # If FS flag missing from row
                row["feature_selection_enabled"] = False  # Default to False when undefined
            if "hyperparameters_enabled" not in row:  # If HP flag missing from row
                row["hyperparameters_enabled"] = False  # Default to False when undefined
            if "data_augmentation_enabled" not in row:  # If DA flag missing from row
                row["data_augmentation_enabled"] = False  # Default to False when undefined

            flat_rows.append(row)  # Append the processed row to the flat list
        return flat_rows  # Return the list of flattened and serialized rows
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def save_stacking_results(csv_path, results_list, config=None):
    """Save the stacking results to CSV file located in same dataset Feature_Analysis directory.

    Writes richer metadata fields matching RFE outputs: model, dataset, accuracy, precision,
    recall, f1_score, fpr, fnr, elapsed_time_s, cv_method, top_features, rfe_ranking,
    hyperparameters, features_list and Hardware.
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Preparing to save {BackgroundColors.CYAN}{len(results_list)}{BackgroundColors.GREEN} stacking results to CSV...{Style.RESET_ALL}",
            config=config
        )

        if not results_list:
            print(f"{BackgroundColors.YELLOW}Warning: No results provided to save.{Style.RESET_ALL}")
            return

        results_filename = config.get("stacking", {}).get("results_filename", "Stacking_Classifiers_Results.csv")  # Get results filename from config
        file_path_obj = Path(csv_path)
        feature_analysis_dir = file_path_obj.parent / "Feature_Analysis"
        os.makedirs(feature_analysis_dir, exist_ok=True)
        
        stacking_results_dir = config.get("stacking", {}).get("results_dir", "Stacking")
        stacking_dir = file_path_obj.parent / stacking_results_dir
        os.makedirs(stacking_dir, exist_ok=True)
        output_path = stacking_dir / results_filename

        flat_rows = flatten_and_serialize_results(results_list)  # Flatten and serialize all result rows into plain dicts

        df = pd.DataFrame(flat_rows)  # Construct results DataFrame from flattened rows

        df = reorder_and_annotate_dataframe(df, config=config)  # Reorder columns by config order and append hardware annotation

        try:
            generate_csv_and_image(df, str(output_path), is_visualizable=True, index=False, encoding="utf-8")  # Persist results CSV and generate PNG
            print(
                f"\n{BackgroundColors.GREEN}Stacking classifier results successfully saved to {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
            )  # Notify user of success

            export_feature_artifacts(df, file_path_obj, stacking_dir, config=config)  # Export feature usage CSV and heatmap to the Stacking directory
        except Exception as e:
            print(
                f"{BackgroundColors.RED}Failed to write Stacking Classifier CSV to {BackgroundColors.CYAN}{output_path}{BackgroundColors.RED}: {e}{Style.RESET_ALL}"
            )
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_cache_file_path(csv_path, config=None):
    """
    Generate the cache file path for a given dataset CSV path.

    :param csv_path: Path to the dataset CSV file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Path to the cache file
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Generating cache file path for: {BackgroundColors.CYAN}{csv_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        cache_prefix = config.get("stacking", {}).get("cache_prefix", "CACHE_")  # Get cache prefix from config
        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]  # Get base dataset name
        output_dir = f"{os.path.dirname(csv_path)}/Classifiers"  # Directory relative to the dataset

        stacking_output_dir = get_stacking_output_dir(csv_path, config)
        validate_output_path(stacking_output_dir, str(Path(output_dir)))
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
        cache_filename = f"{cache_prefix}{dataset_name}-Stacking_Classifiers_Results.csv"  # Cache filename
        cache_path = os.path.join(output_dir, cache_filename)  # Full cache file path

        return cache_path  # Return the cache file path
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def safe_load_json(val):
    """
    Safely load a JSON string if it's a string, otherwise return the value as is.
    
    :param val: The value to load as JSON if it's a string
    :return: The loaded JSON object if val is a string, None if val is Na
    """
    
    if pd.isna(val):  # Verify value is NaN before returning None
        return None  # Return None for NaN values
    
    if isinstance(val, str):  # If the value is a string, attempt to parse it as JSON
        try:  # Try to load the string as JSON
            return json.loads(val)  # Return the loaded JSON object
        except Exception:  # If parsing fails, return the original string value
            return val  # Return the original value if it's a string but not valid JSON
    
    return val  # For non-string values, return as is
            
                
def load_cache_results(csv_path, config=None):
    """
    Load cached results from the cache file if it exists.

    :param csv_path: Path to the dataset CSV file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary mapping (feature_set, model_name) to result entry
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        cache_path = get_cache_file_path(csv_path, config=config)  # Get the cache file path

        if not os.path.exists(cache_path):  # If cache file doesn't exist
            verbose_output(
                f"{BackgroundColors.YELLOW}No cache file found at: {BackgroundColors.CYAN}{cache_path}{Style.RESET_ALL}",
                config=config
            )  # Output the verbose message
            return {}  # Return empty dictionary

        verbose_output(
            f"{BackgroundColors.GREEN}Loading cached results from: {BackgroundColors.CYAN}{cache_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        try:  # Try to load the cache file
            low_memory = config.get("execution", {}).get("low_memory", False)  # Read low memory flag from config
            df_cache = pd.read_csv(cache_path, low_memory=low_memory)  # Read the cache file
            df_cache.columns = df_cache.columns.str.strip()  # Remove leading/trailing whitespace from column names
            cache_dict = {}  # Initialize cache dictionary

            for _, row in df_cache.iterrows():  # Iterate through each row
                feature_set = row.get("feature_set", "")  # Get feature set name
                model_name = row.get("model_name", "")  # Get model name
                cache_key = (feature_set, model_name)  # Create cache key tuple

                result_entry = {
                    "model": row.get("model", ""),
                    "dataset": row.get("dataset", ""),
                    "feature_set": feature_set,
                    "classifier_type": row.get("classifier_type", ""),
                    "model_name": model_name,
                    "data_source": row.get("data_source", ""),
                    "experiment_id": row.get("experiment_id", None),
                    "experiment_mode": row.get("experiment_mode", "original_only"),
                    "augmentation_ratio": float(row["augmentation_ratio"]) if "augmentation_ratio" in row and not pd.isna(row.get("augmentation_ratio")) else None,
                    "n_features": int(row["n_features"]) if "n_features" in row and not pd.isna(row["n_features"]) else None,
                    "n_samples_train": int(row["n_samples_train"]) if "n_samples_train" in row and not pd.isna(row["n_samples_train"]) else None,
                    "n_samples_test": int(row["n_samples_test"]) if "n_samples_test" in row and not pd.isna(row["n_samples_test"]) else None,
                    "accuracy": float(row["accuracy"]) if "accuracy" in row and not pd.isna(row["accuracy"]) else None,
                    "precision": float(row["precision"]) if "precision" in row and not pd.isna(row["precision"]) else None,
                    "recall": float(row["recall"]) if "recall" in row and not pd.isna(row["recall"]) else None,
                    "f1_score": float(row["f1_score"]) if "f1_score" in row and not pd.isna(row["f1_score"]) else None,
                    "fpr": float(row["fpr"]) if "fpr" in row and not pd.isna(row["fpr"]) else None,
                    "fnr": float(row["fnr"]) if "fnr" in row and not pd.isna(row["fnr"]) else None,
                    "elapsed_time_s": float(row["elapsed_time_s"]) if "elapsed_time_s" in row and not pd.isna(row["elapsed_time_s"]) else None,
                    "cv_method": row.get("cv_method", None),
                    "top_features": safe_load_json(row.get("top_features", None)),
                    "rfe_ranking": safe_load_json(row.get("rfe_ranking", None)),
                    "hyperparameters": safe_load_json(row.get("hyperparameters", None)),
                    "features_list": safe_load_json(row.get("features_list", None)),
                    "Hardware": row.get("Hardware", None),
                }

                cache_dict[cache_key] = result_entry

            print(f"{BackgroundColors.GREEN}Loaded cached results from: {BackgroundColors.CYAN}{cache_path}{Style.RESET_ALL}")
            return cache_dict

        except Exception as e:  # Catch any errors
            print(
                f"{BackgroundColors.YELLOW}Warning: Failed to save to cache {BackgroundColors.CYAN}{cache_path}{BackgroundColors.YELLOW}: {e}{Style.RESET_ALL}"
            )  # Print warning message
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def remove_cache_file(csv_path, config=None):
    """
    Remove the cache file after successful completion.

    :param csv_path: Path to the dataset CSV file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        cache_path = get_cache_file_path(csv_path, config=config)  # Get the cache file path

        if os.path.exists(cache_path):  # If cache file exists
            try:  # Try to remove the cache file
                os.remove(cache_path)  # Delete the cache file
                print(
                    f"{BackgroundColors.GREEN}Cache file removed: {BackgroundColors.CYAN}{cache_path}{Style.RESET_ALL}"
                )  # Print success message
            except Exception as e:  # Catch any errors
                print(
                    f"{BackgroundColors.YELLOW}Warning: Failed to remove cache file {BackgroundColors.CYAN}{cache_path}{BackgroundColors.YELLOW}: {e}{Style.RESET_ALL}"
                )  # Print warning message
        else:  # If cache file doesn't exist
            verbose_output(
                f"{BackgroundColors.YELLOW}No cache file to remove at: {BackgroundColors.CYAN}{cache_path}{Style.RESET_ALL}"
            )  # Output verbose message
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_automl_search_spaces():
    """
    Return hyperparameter search space definitions for all AutoML candidate models.

    :return: Dictionary mapping model names to their search space configurations.
    """
    
    try:
        return {  # Dictionary of model search spaces
            "Random Forest": {  # Random Forest search space
                "n_estimators": ("int", 50, 500),  # Number of trees range
                "max_depth": ("int_or_none", 3, 50),  # Max depth range or None
                "min_samples_split": ("int", 2, 20),  # Min samples to split
                "min_samples_leaf": ("int", 1, 10),  # Min samples per leaf
                "max_features": ("categorical", ["sqrt", "log2", None]),  # Feature selection method
            },
            "XGBoost": {  # XGBoost search space
                "n_estimators": ("int", 50, 500),  # Number of boosting rounds
                "max_depth": ("int", 3, 15),  # Max tree depth
                "learning_rate": ("float_log", 0.01, 0.3),  # Learning rate (log scale)
                "subsample": ("float", 0.5, 1.0),  # Row subsampling ratio
                "colsample_bytree": ("float", 0.5, 1.0),  # Column subsampling ratio
                "min_child_weight": ("int", 1, 10),  # Min child weight
                "reg_alpha": ("float_log", 1e-8, 10.0),  # L1 regularization
                "reg_lambda": ("float_log", 1e-8, 10.0),  # L2 regularization
            },
            "LightGBM": {  # LightGBM search space
                "n_estimators": ("int", 50, 500),  # Number of boosting rounds
                "max_depth": ("int", 3, 15),  # Max tree depth
                "learning_rate": ("float_log", 0.01, 0.3),  # Learning rate (log scale)
                "num_leaves": ("int", 15, 127),  # Number of leaves
                "min_child_samples": ("int", 5, 100),  # Min samples in leaf
                "subsample": ("float", 0.5, 1.0),  # Row subsampling ratio
                "colsample_bytree": ("float", 0.5, 1.0),  # Column subsampling ratio
                "reg_alpha": ("float_log", 1e-8, 10.0),  # L1 regularization
                "reg_lambda": ("float_log", 1e-8, 10.0),  # L2 regularization
            },
            "Logistic Regression": {  # Logistic Regression search space
                "C": ("float_log", 0.001, 100.0),  # Regularization parameter
                "solver": ("categorical", ["lbfgs", "saga"]),  # Optimization algorithm
                "max_iter": ("int", 500, 5000),  # Max iterations
            },
            "SVM": {  # SVM search space
                "C": ("float_log", 0.01, 100.0),  # Regularization parameter
                "kernel": ("categorical", ["rbf", "linear", "poly"]),  # Kernel function
                "gamma": ("categorical", ["scale", "auto"]),  # Kernel coefficient
            },
            "Extra Trees": {  # Extra Trees search space
                "n_estimators": ("int", 50, 500),  # Number of trees
                "max_depth": ("int_or_none", 3, 50),  # Max depth or None
                "min_samples_split": ("int", 2, 20),  # Min samples to split
                "min_samples_leaf": ("int", 1, 10),  # Min samples per leaf
                "max_features": ("categorical", ["sqrt", "log2", None]),  # Feature selection method
            },
            "Gradient Boosting": {  # Gradient Boosting search space
                "n_estimators": ("int", 50, 300),  # Number of boosting rounds
                "max_depth": ("int", 3, 10),  # Max tree depth
                "learning_rate": ("float_log", 0.01, 0.3),  # Learning rate
                "subsample": ("float", 0.5, 1.0),  # Row subsampling ratio
                "min_samples_split": ("int", 2, 20),  # Min samples to split
                "min_samples_leaf": ("int", 1, 10),  # Min samples per leaf
            },
            "MLP (Neural Net)": {  # MLP Neural Network search space
                "hidden_layer_sizes_0": ("int", 32, 256),  # First hidden layer size
                "hidden_layer_sizes_1": ("int", 0, 128),  # Second hidden layer size (0 means single layer)
                "learning_rate_init": ("float_log", 0.0001, 0.01),  # Initial learning rate
                "alpha": ("float_log", 1e-6, 0.01),  # L2 penalty (regularization)
                "max_iter": ("int", 200, 1000),  # Max iterations
                "activation": ("categorical", ["relu", "tanh"]),  # Activation function
            },
            "Decision Tree": {  # Decision Tree search space
                "max_depth": ("int_or_none", 3, 50),  # Max depth or None
                "min_samples_split": ("int", 2, 20),  # Min samples to split
                "min_samples_leaf": ("int", 1, 10),  # Min samples per leaf
                "criterion": ("categorical", ["gini", "entropy"]),  # Split criterion
                "max_features": ("categorical", ["sqrt", "log2", None]),  # Feature selection method
            },
            "KNN": {  # K-Nearest Neighbors search space
                "n_neighbors": ("int", 3, 25),  # Number of neighbors
                "weights": ("categorical", ["uniform", "distance"]),  # Weight function
                "metric": ("categorical", ["euclidean", "manhattan", "minkowski"]),  # Distance metric
            },
        }  # Return full search space dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def suggest_hyperparameters_for_model(trial, model_name, search_spaces):
    """
    Suggests hyperparameters for a given model using an Optuna trial.

    :param trial: Optuna trial object for parameter suggestion
    :param model_name: Name of the model to suggest hyperparameters for
    :param search_spaces: Dictionary of search space definitions
    :return: Dictionary of suggested hyperparameters
    """
    
    try:
        space = search_spaces.get(model_name, {})  # Get the search space for this model
        params = {}  # Initialize empty parameters dictionary

        for param_name, config in space.items():  # Iterate over each parameter definition
            param_type = config[0]  # Extract the parameter type

            if param_type == "int":  # Integer parameter
                params[param_name] = trial.suggest_int(param_name, config[1], config[2])  # Suggest integer value
            elif param_type == "float":  # Float parameter (uniform)
                params[param_name] = trial.suggest_float(param_name, config[1], config[2])  # Suggest float value
            elif param_type == "float_log":  # Float parameter (log scale)
                params[param_name] = trial.suggest_float(param_name, config[1], config[2], log=True)  # Suggest log-scaled float
            elif param_type == "categorical":  # Categorical parameter
                params[param_name] = trial.suggest_categorical(param_name, config[1])  # Suggest from categories
            elif param_type == "int_or_none":  # Integer or None parameter
                use_none = trial.suggest_categorical(f"{param_name}_none", [True, False])  # Decide whether to use None
                if use_none:  # If None is selected
                    params[param_name] = None  # Set parameter to None
                else:  # Otherwise suggest an integer
                    params[param_name] = trial.suggest_int(param_name, config[1], config[2])  # Suggest integer value

        return params  # Return the suggested parameters
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def create_model_from_params(model_name, params, config=None):
    """
    Creates a classifier instance from a model name and hyperparameters dictionary.

    :param model_name: Name of the classifier to instantiate
    :param params: Dictionary of hyperparameters to apply
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Instantiated classifier object
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        automl_random_state = config.get("automl", {}).get("random_state", 42)  # Get random state from config
        n_jobs = config.get("evaluation", {}).get("n_jobs", -1)  # Get n_jobs from config

        clean_params = {k: v for k, v in params.items() if not k.endswith("_none")}  # Copy params excluding _none flags

        if model_name == "Random Forest":  # Random Forest classifier
            return RandomForestClassifier(random_state=automl_random_state, n_jobs=n_jobs, **clean_params)  # Create RF instance
        elif model_name == "XGBoost":  # XGBoost classifier
            return XGBClassifier(eval_metric="mlogloss", random_state=automl_random_state, n_jobs=n_jobs, **clean_params)  # Create XGB instance
        elif model_name == "LightGBM":  # LightGBM classifier
            return lgb.LGBMClassifier(force_row_wise=True, random_state=automl_random_state, verbosity=-1, n_jobs=n_jobs, **clean_params)  # Create LGBM instance
        elif model_name == "Logistic Regression":  # Logistic Regression classifier
            return LogisticRegression(random_state=automl_random_state, n_jobs=n_jobs, **clean_params)  # Create LR instance with parallel jobs
        elif model_name == "SVM":  # Support Vector Machine classifier
            return SVC(probability=True, random_state=automl_random_state, **clean_params)  # Create SVM instance
        elif model_name == "Extra Trees":  # Extra Trees classifier
            return ExtraTreesClassifier(random_state=automl_random_state, n_jobs=n_jobs, **clean_params)  # Create ET instance
        elif model_name == "Gradient Boosting":  # Gradient Boosting classifier
            return GradientBoostingClassifier(random_state=automl_random_state, **clean_params)  # Create GB instance
        elif model_name == "MLP (Neural Net)":  # MLP Neural Network classifier
            hidden_0 = clean_params.pop("hidden_layer_sizes_0", 100)  # Extract first hidden layer size
            hidden_1 = clean_params.pop("hidden_layer_sizes_1", 0)  # Extract second hidden layer size
            hidden_layers = (hidden_0,) if hidden_1 == 0 else (hidden_0, hidden_1)  # Build hidden layer tuple
            return MLPClassifier(hidden_layer_sizes=hidden_layers, random_state=automl_random_state, **clean_params)  # Create MLP instance
        elif model_name == "Decision Tree":  # Decision Tree classifier
            return DecisionTreeClassifier(random_state=automl_random_state, **clean_params)  # Create DT instance
        elif model_name == "KNN":  # K-Nearest Neighbors classifier
            return KNeighborsClassifier(n_jobs=n_jobs, **clean_params)  # Create KNN instance
        else:  # Unknown model type
            raise ValueError(f"Unknown AutoML model name: {model_name}")  # Raise error for unknown model
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def automl_cross_validate_model(model, X_train, y_train, cv_folds, trial=None, config=None):
    """
    Performs stratified cross-validation on a model and returns mean F1 score.

    :param model: Classifier instance to evaluate
    :param X_train: Training features array
    :param y_train: Training target array (numpy)
    :param cv_folds: Number of cross-validation folds
    :param trial: Optional Optuna trial for intermediate reporting and pruning
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Mean cross-validated F1 score
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        automl_random_state = config.get("automl", {}).get("random_state", 42)  # Get random state from config

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=automl_random_state)  # Create stratified k-fold
        f1_scores = []  # Initialize list for F1 scores

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):  # Iterate over folds
            X_fold_train = X_train[train_idx]  # Get fold training features
            y_fold_train = y_train[train_idx]  # Get fold training target
            X_fold_val = X_train[val_idx]  # Get fold validation features
            y_fold_val = y_train[val_idx]  # Get fold validation target

            model.fit(X_fold_train, y_fold_train)  # Fit model on fold training data using its internal n_jobs parallelism
            y_pred = model.predict(X_fold_val)  # Predict on fold validation data
            fold_f1 = f1_score(y_fold_val, y_pred, average="weighted", zero_division=0)  # Calculate fold F1
            f1_scores.append(fold_f1)  # Append fold F1 score

            if trial is not None:  # If Optuna trial is provided
                trial.report(np.mean(f1_scores), fold_idx)  # Report intermediate value for pruning
                if trial.should_prune():  # Verify if trial should be pruned
                    raise optuna.exceptions.TrialPruned()  # Prune this trial

        return float(np.mean(f1_scores))  # Return mean F1 score across folds as Python float
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def automl_objective(trial, X_train, y_train, cv_folds, config=None):
    """
    Optuna objective function for automated model and hyperparameter selection.

    :param trial: Optuna trial object
    :param X_train: Training features array
    :param y_train: Training target array (numpy)
    :param cv_folds: Number of cross-validation folds
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Mean cross-validated F1 score (to maximize)
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        search_spaces = get_automl_search_spaces()  # Get all model search spaces
        model_names = list(search_spaces.keys())  # Get list of available model names

        model_name = trial.suggest_categorical("model_name", model_names)  # Select model type via trial
        params = suggest_hyperparameters_for_model(trial, model_name, search_spaces)  # Suggest hyperparameters

        try:  # Try to create and evaluate the model
            model = create_model_from_params(model_name, params, config=config)  # Create model instance from params
            mean_f1 = automl_cross_validate_model(model, X_train, y_train, cv_folds, trial, config=config)  # Cross-validate
            return mean_f1  # Return mean F1 score
        except optuna.exceptions.TrialPruned:  # Handle Optuna pruning
            raise  # Re-raise pruning exception
        except Exception as e:  # Handle other errors gracefully
            verbose_output(
                f"{BackgroundColors.YELLOW}AutoML trial failed for {model_name}: {e}{Style.RESET_ALL}",
                config=config,
            )  # Log the trial failure
            return 0.0  # Return zero score for failed trials
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_automl_model_search(X_train, y_train, file_path, config=None):
    """
    Runs Optuna-based AutoML model search to find optimal classifier and hyperparameters.

    :param X_train: Scaled training features (numpy array)
    :param y_train: Training target labels (numpy array)
    :param file_path: Path to the dataset file for logging
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (best_model_name, best_params, study) or (None, None, None) on failure
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        automl_n_trials = config.get("automl", {}).get("n_trials", 50)  # Get number of trials from config
        automl_timeout = config.get("automl", {}).get("timeout", 3600)  # Get timeout from config
        automl_cv_folds = config.get("automl", {}).get("cv_folds", 5)  # Get CV folds from config
        automl_random_state = config.get("automl", {}).get("random_state", 42)  # Get random state from config

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Starting AutoML model search with {BackgroundColors.CYAN}{automl_n_trials}{BackgroundColors.GREEN} trials...{Style.RESET_ALL}"
        )  # Output search start message

        optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress verbose Optuna logging

        sampler = optuna.samplers.TPESampler(seed=automl_random_state)  # Create TPE sampler with deterministic seed
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)  # Create median pruner for early stopping

        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner, study_name="automl_model_search"
        )  # Create Optuna study to maximize F1 score

        objective_fn = lambda trial: automl_objective(trial, X_train, y_train, automl_cv_folds, config=config)  # Create objective wrapper
        study.optimize(objective_fn, n_trials=automl_n_trials, timeout=automl_timeout, n_jobs=1)  # Run the optimization

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]  # Get completed trials

        if not completed_trials:  # If no trials completed successfully
            print(
                f"{BackgroundColors.RED}AutoML model search failed: no successful trials completed.{Style.RESET_ALL}"
            )  # Output failure message
            return (None, None, None)  # Return None tuple

        best_trial = study.best_trial  # Get the best trial
        best_model_name = best_trial.params.get("model_name", "Unknown")  # Extract best model name
        best_params = {
            k: v for k, v in best_trial.params.items() if k != "model_name" and not k.endswith("_none")
        }  # Extract best params excluding model_name and _none flags

        pruned_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])  # Count pruned trials

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}AutoML Best Model: {BackgroundColors.CYAN}{best_model_name}{Style.RESET_ALL}"
        )  # Output best model name
        print(
            f"{BackgroundColors.GREEN}Best CV F1 Score: {BackgroundColors.CYAN}{truncate_value(study.best_value)}{Style.RESET_ALL}"
        )  # Output best F1 score
        print(
            f"{BackgroundColors.GREEN}Best Parameters: {BackgroundColors.CYAN}{best_params}{Style.RESET_ALL}"
        )  # Output best parameters
        print(
            f"{BackgroundColors.GREEN}Trials: {BackgroundColors.CYAN}{len(completed_trials)} completed, {pruned_count} pruned{Style.RESET_ALL}"
        )  # Output trial statistics

        return (best_model_name, best_params, study)  # Return best model info and study object
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def automl_stacking_objective(trial, X_train, y_train, cv_folds, candidate_models, config=None):
    """
    Optuna objective function for optimizing stacking ensemble configuration.

    :param trial: Optuna trial object
    :param X_train: Training features array
    :param y_train: Training target array (numpy)
    :param cv_folds: Number of cross-validation folds
    :param candidate_models: Dictionary mapping model names to parameter dicts
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Mean cross-validated F1 score (to maximize)
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        automl_random_state = config.get("automl", {}).get("random_state", 42)  # Get random state from config
        n_jobs = config.get("evaluation", {}).get("n_jobs", -1)  # Get n_jobs from config

        model_names = list(candidate_models.keys())  # Get list of candidate model names
        selected_models = []  # Initialize list for selected base learners

        for name in model_names:  # Iterate over each candidate model
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")  # Sanitize name for Optuna parameter
            include = trial.suggest_categorical(f"use_{safe_name}", [True, False])  # Decide whether to include this model
            if include:  # If model is selected for inclusion
                selected_models.append(name)  # Add to selected list

        if len(selected_models) < 2:  # Need at least 2 base learners for stacking
            return 0.0  # Return zero score if insufficient base learners

        meta_learner_name = trial.suggest_categorical(
            "meta_learner", ["Logistic Regression", "Random Forest", "Gradient Boosting"]
        )  # Select meta-learner type
        n_cv_splits = trial.suggest_int("stacking_cv_splits", 3, min(cv_folds, 10))  # Select CV splits for stacking

        try:  # Try to build and evaluate stacking ensemble
            estimators = []  # Initialize base estimators list
            for name in selected_models:  # Build each selected base learner
                model_params = candidate_models[name]  # Get pre-optimized parameters
                model = create_model_from_params(name, model_params, config=config)  # Create model instance
                safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")  # Sanitize estimator name
                estimators.append((safe_name, model))  # Add to estimators list

            if meta_learner_name == "Logistic Regression":  # Logistic Regression meta-learner
                meta_model = LogisticRegression(max_iter=1000, random_state=automl_random_state, n_jobs=n_jobs)  # Create LR meta-learner with parallel jobs
            elif meta_learner_name == "Random Forest":  # Random Forest meta-learner
                meta_model = RandomForestClassifier(n_estimators=50, random_state=automl_random_state, n_jobs=n_jobs)  # Create RF meta-learner
            else:  # Gradient Boosting meta-learner
                meta_model = GradientBoostingClassifier(random_state=automl_random_state)  # Create GB meta-learner

            stacking = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_model,
                cv=StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=automl_random_state),
                n_jobs=1,
            )  # Create stacking classifier with sequential CV folds to prevent nested loky deadlock

            mean_f1 = automl_cross_validate_model(stacking, X_train, y_train, cv_folds, trial)  # Cross-validate stacking
            return mean_f1  # Return mean F1 score

        except optuna.exceptions.TrialPruned:  # Handle Optuna pruning
            raise  # Re-raise pruning exception
        except Exception as e:  # Handle other errors gracefully
            verbose_output(
                f"{BackgroundColors.YELLOW}AutoML stacking trial failed: {e}{Style.RESET_ALL}"
            )  # Log the failure
            return 0.0  # Return zero for failed trials
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_top_automl_models(study, top_n=5):
    """
    Extracts the top N unique models from an AutoML study based on F1 score.

    :param study: Completed Optuna study object
    :param top_n: Number of top models to extract
    :return: Dictionary mapping model names to their best parameters
    """
    
    try:
        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]  # Filter to completed trials only
        sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)  # Sort trials by score descending

        top_models = {}  # Initialize dictionary for top models

        for trial in sorted_trials:  # Iterate through sorted trials
            model_name = trial.params.get("model_name", "Unknown")  # Get model name from trial
            if model_name not in top_models:  # If this model type hasn't been added yet
                params = {
                    k: v for k, v in trial.params.items() if k != "model_name" and not k.endswith("_none")
                }  # Extract parameters
                top_models[model_name] = params  # Store best params for this model type
            if len(top_models) >= top_n:  # If we've collected enough models
                break  # Stop collecting

        return top_models  # Return dictionary of top models and their parameters
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_automl_stacking_search(X_train, y_train, model_study, file_path, config=None):
    """
    Runs Optuna-based optimization to find the best stacking ensemble configuration.

    :param X_train: Scaled training features (numpy array)
    :param y_train: Training target labels (numpy array)
    :param model_study: Completed Optuna study from model search
    :param file_path: Path to the dataset file for logging
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (best_stacking_config, stacking_study) or (None, None) on failure
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Starting AutoML stacking search with {BackgroundColors.CYAN}{config.get("automl", {}).get("stacking_trials", 20)}{BackgroundColors.GREEN} trials...{Style.RESET_ALL}"
        )  # Output search start message

        candidate_models = extract_top_automl_models(model_study, top_n=config.get("automl", {}).get("stacking_top_n", 5))  # Get top models from model search

        if len(candidate_models) < 2:  # If not enough candidate models
            print(
                f"{BackgroundColors.YELLOW}Not enough candidate models for stacking search. Need at least 2, got {len(candidate_models)}.{Style.RESET_ALL}"
            )  # Output warning
            return (None, None)  # Return None tuple

        print(
            f"{BackgroundColors.GREEN}Candidate base learners: {BackgroundColors.CYAN}{list(candidate_models.keys())}{Style.RESET_ALL}"
        )  # Output candidate models

        optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress verbose Optuna logging

        sampler = optuna.samplers.TPESampler(seed=config.get("automl", {}).get("random_state", 42) + 1)  # Create sampler with different seed
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)  # Create pruner

        stacking_study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner, study_name="automl_stacking_search"
        )  # Create Optuna study for stacking optimization

        objective_fn = lambda trial: automl_stacking_objective(
            trial, X_train, y_train, config.get("automl", {}).get("cv_folds", 5), candidate_models
        )  # Create stacking objective wrapper
        stacking_study.optimize(
            objective_fn, n_trials=config.get("automl", {}).get("stacking_trials", 20), timeout=config.get("automl", {}).get("timeout", 3600), n_jobs=1
        )  # Run stacking optimization

        completed = [
            t for t in stacking_study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]  # Get completed trials

        if not completed:  # If no stacking trials completed
            print(
                f"{BackgroundColors.YELLOW}AutoML stacking search: no successful trials.{Style.RESET_ALL}"
            )  # Output warning
            return (None, None)  # Return None tuple

        best_trial = stacking_study.best_trial  # Get best stacking trial
        best_config = {
            "meta_learner": best_trial.params.get("meta_learner"),  # Best meta-learner choice
            "stacking_cv_splits": best_trial.params.get("stacking_cv_splits"),  # Best CV splits
            "base_learners": [
                name for name in candidate_models.keys()
                if best_trial.params.get(f"use_{name.replace(' ', '_').replace('(', '').replace(')', '')}", False)
            ],  # Selected base learner names
            "base_learner_params": candidate_models,  # Parameters for each base learner
            "best_cv_f1": stacking_study.best_value,  # Best CV F1 score
        }  # Build best configuration dictionary

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}AutoML Best Stacking Config:{Style.RESET_ALL}"
        )  # Output header
        print(
            f"{BackgroundColors.GREEN}  Meta-learner: {BackgroundColors.CYAN}{best_config['meta_learner']}{Style.RESET_ALL}"
        )  # Output meta-learner
        print(
            f"{BackgroundColors.GREEN}  Base learners: {BackgroundColors.CYAN}{best_config['base_learners']}{Style.RESET_ALL}"
        )  # Output base learners
        print(
            f"{BackgroundColors.GREEN}  CV splits: {BackgroundColors.CYAN}{best_config['stacking_cv_splits']}{Style.RESET_ALL}"
        )  # Output CV splits
        print(
            f"{BackgroundColors.GREEN}  Best CV F1: {BackgroundColors.CYAN}{truncate_value(best_config['best_cv_f1'])}{Style.RESET_ALL}"
        )  # Output best F1

        return (best_config, stacking_study)  # Return best config and study
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_automl_stacking_model(best_config, config=None):
    """
    Builds a StackingClassifier from the best AutoML stacking configuration.

    :param best_config: Dictionary with best stacking configuration
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Configured StackingClassifier instance
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        estimators = []  # Initialize estimators list

        for name in best_config["base_learners"]:  # Iterate over selected base learners
            params = best_config["base_learner_params"].get(name, {})  # Get model parameters
            model = create_model_from_params(name, params)  # Create model instance
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")  # Sanitize estimator name
            estimators.append((safe_name, model))  # Add to estimators list

        meta_learner_name = best_config["meta_learner"]  # Get meta-learner name

        if meta_learner_name == "Logistic Regression":  # Logistic Regression meta-learner
            meta_model = LogisticRegression(max_iter=1000, random_state=config.get("automl", {}).get("random_state", 42), n_jobs=config.get("evaluation", {}).get("n_jobs", -1))  # Create LR with parallel jobs
        elif meta_learner_name == "Random Forest":  # Random Forest meta-learner
            meta_model = RandomForestClassifier(n_estimators=50, random_state=config.get("automl", {}).get("random_state", 42), n_jobs=config.get("evaluation", {}).get("n_jobs", -1))  # Create RF
        else:  # Gradient Boosting meta-learner
            meta_model = GradientBoostingClassifier(random_state=config.get("automl", {}).get("random_state", 42))  # Create GB

        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_model,
            cv=StratifiedKFold(
                n_splits=best_config["stacking_cv_splits"], shuffle=True, random_state=config.get("automl", {}).get("random_state", 42)
            ),
            n_jobs=1,
        )  # Create stacking classifier with sequential CV folds to prevent nested loky deadlock

        return stacking_model  # Return configured stacking model
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def evaluate_automl_model_on_test(model, model_name, X_train, y_train, X_test, y_test):
    """
    Trains and evaluates an AutoML-selected model on the held-out test set.

    :param model: Classifier instance to evaluate
    :param model_name: Name of the model for logging
    :param X_train: Training features array
    :param y_train: Training target labels
    :param X_test: Testing features array
    :param y_test: Testing target labels
    :return: Dictionary containing all evaluation metrics
    """
    
    try:
        start_time = time.time()  # Record start time

        sys.stdout.flush()  # Flush stdout before model training to ensure logs are visible under nohup
        model.fit(X_train, y_train)  # Train model on full training set using its internal n_jobs parallelism
        y_pred = model.predict(X_test)  # Generate predictions on test set

        elapsed = time.time() - start_time  # Calculate elapsed training time

        acc = accuracy_score(y_test, y_pred)  # Calculate accuracy
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate weighted precision
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate weighted recall
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate weighted F1 score

        roc_auc = None  # Initialize ROC-AUC as None
        try:  # Try to compute ROC-AUC
            if hasattr(model, "predict_proba"):  # If model supports probability predictions
                y_proba = model.predict_proba(X_test)  # Get probability predictions
                if len(np.unique(y_test)) == 2:  # Binary classification
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])  # Compute binary ROC-AUC
                else:  # Multi-class classification
                    roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")  # Compute multi-class ROC-AUC
        except Exception:  # If ROC-AUC computation fails
            roc_auc = None  # Keep as None

        if len(np.unique(y_test)) == 2:  # Binary classification metrics
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()  # Get confusion matrix components
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # Calculate false positive rate
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # Calculate false negative rate
        else:  # Multi-class (simplified)
            fpr = 0.0  # Placeholder FPR
            fnr = 0.0  # Placeholder FNR

        print(
            f"{BackgroundColors.GREEN}AutoML {model_name} Test Results - Acc: {BackgroundColors.CYAN}{truncate_value(acc)}{BackgroundColors.GREEN}, F1: {BackgroundColors.CYAN}{truncate_value(f1)}{BackgroundColors.GREEN}, ROC-AUC: {BackgroundColors.CYAN}{truncate_value(roc_auc)}{BackgroundColors.GREEN}, Time: {BackgroundColors.CYAN}{int(round(elapsed))}s{Style.RESET_ALL}"
        )  # Output test results

        return {  # Build and return metrics dictionary
            "accuracy": acc,  # Accuracy value
            "precision": prec,  # Precision value
            "recall": rec,  # Recall value
            "f1_score": f1,  # F1 score value
            "roc_auc": roc_auc,  # ROC-AUC value
            "fpr": fpr,  # False positive rate
            "fnr": fnr,  # False negative rate
            "elapsed_time_s": int(round(elapsed)),  # Elapsed time in seconds
        }  # Return metrics dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def export_automl_search_history(study, output_dir, study_name, dataset_file=None):
    """
    Exports the Optuna study trial history to a CSV file.

    :param study: Completed Optuna study object
    :param output_dir: Directory path for saving the export file
    :param study_name: Name prefix for the output file
    :return: Path to the exported CSV file
    """
    
    try:
        if dataset_file is not None:
            stacking_output_dir = get_stacking_output_dir(dataset_file, CONFIG)
        else:
            fa_parent = None
            for p in Path(output_dir).parents:
                if p.name == "Feature_Analysis":
                    fa_parent = str(p)
                    break
            if fa_parent is None:
                raise RuntimeError("Cannot derive Feature_Analysis root from output_dir for safe write")
            stacking_output_dir = str((Path(fa_parent).parent / CONFIG.get("stacking", {}).get("results_dir")).resolve())
        validate_output_path(stacking_output_dir, str(Path(output_dir)))
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

        trials_data = []  # Initialize list for trial data

        for trial in study.trials:  # Iterate over all trials
            trial_entry = {  # Build entry for this trial
                "trial_number": trial.number,  # Trial index number
                "value": trial.value if trial.value is not None else None,  # Objective value (F1 score)
                "state": trial.state.name,  # Trial state (COMPLETE, PRUNED, FAIL)
                "duration_s": (
                    trial.duration.total_seconds() if trial.duration else None
                ),  # Trial duration in seconds
            }  # Build basic trial entry
            trial_entry.update(trial.params)  # Add trial parameters to entry
            trials_data.append(trial_entry)  # Append to trials data list

        df = pd.DataFrame(trials_data)  # Convert trials data to DataFrame
        output_path = os.path.join(output_dir, f"{study_name}_search_history.csv")  # Build output file path
        generate_csv_and_image(df, output_path, is_visualizable=True, index=False)  # Save to CSV and generate PNG image

        print(
            f"{BackgroundColors.GREEN}AutoML search history exported to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
        )  # Output export confirmation

        return output_path  # Return the output file path
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def export_automl_best_config(best_model_name, best_params, test_metrics, stacking_config, output_dir, feature_names, dataset_file=None):
    """
    Exports the best AutoML configuration and metrics to a JSON file.

    :param best_model_name: Name of the best model found
    :param best_params: Best hyperparameters dictionary
    :param test_metrics: Test set evaluation metrics dictionary
    :param stacking_config: Best stacking configuration dictionary (or None)
    :param output_dir: Directory path for saving the export file
    :param feature_names: List of feature names used
    :return: Path to the exported JSON file
    """
    
    try:
        if dataset_file is not None:
            stacking_output_dir = get_stacking_output_dir(dataset_file, CONFIG)
        else:
            fa_parent = None
            for p in Path(output_dir).parents:
                if p.name == "Feature_Analysis":
                    fa_parent = str(p)
                    break
            if fa_parent is None:
                raise RuntimeError("Cannot derive Feature_Analysis root from output_dir for safe write")
            stacking_output_dir = str((Path(fa_parent).parent / CONFIG.get("stacking", {}).get("results_dir")).resolve())
        validate_output_path(stacking_output_dir, str(Path(output_dir)))
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

        export_config = {  # Build configuration export dictionary
            "best_individual_model": {  # Best individual model section
                "model_name": best_model_name,  # Model name
                "hyperparameters": best_params,  # Model hyperparameters
                "test_metrics": {
                    k: truncate_value(v) if isinstance(v, (int, float)) and v is not None else v
                    for k, v in test_metrics.items()
                },  # Test metrics with truncation
            },
            "best_stacking_config": stacking_config,  # Stacking configuration (may be None)
            "automl_settings": {  # AutoML settings used
                "n_trials": CONFIG.get("automl", {}).get("n_trials", 50),  # Number of model search trials
                "stacking_trials": CONFIG.get("automl", {}).get("stacking_trials", 20),  # Number of stacking search trials
                "cv_folds": CONFIG.get("automl", {}).get("cv_folds", 5),  # Cross-validation folds
                "timeout_s": CONFIG.get("automl", {}).get("timeout", 3600),  # Timeout in seconds
                "random_state": CONFIG.get("automl", {}).get("random_state", 42),  # Random seed used
            },
            "feature_names": feature_names,  # Features used in training
            "n_features": len(feature_names),  # Number of features
        }  # Build complete config dictionary

        output_path = os.path.join(output_dir, CONFIG.get("automl", {}).get("results_filename", "AutoML_Results.csv").replace(".csv", "_best_config.json"))  # Build output path

        if dataset_file is not None:
            stacking_output_dir = get_stacking_output_dir(dataset_file, CONFIG)
        else:
            fa_parent = None
            for p in Path(output_path).parents:
                if p.name == "Feature_Analysis":
                    fa_parent = str(p)
                    break
            if fa_parent is None:
                raise RuntimeError("Cannot derive Feature_Analysis root from output_path for safe write")
            stacking_output_dir = str((Path(fa_parent).parent / CONFIG.get("stacking", {}).get("results_dir")).resolve())
        validate_output_path(stacking_output_dir, output_path)

        with open(output_path, "w", encoding="utf-8") as f:  # Open file for writing
            json.dump(export_config, f, indent=2, default=str)  # Write JSON with indentation

        print(
            f"{BackgroundColors.GREEN}AutoML best configuration exported to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
        )  # Output export confirmation

        return output_path  # Return the output file path
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def export_automl_best_model(model, scaler, output_dir, model_name, feature_names, dataset_file=None):
    """
    Exports the best AutoML model and scaler to disk using joblib.

    :param model: Trained best model instance
    :param scaler: Fitted StandardScaler instance
    :param output_dir: Directory path for saving model files
    :param model_name: Name of the model for file naming
    :param feature_names: List of feature names for metadata
    :return: Tuple (model_path, scaler_path)
    """
    
    try:
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

        safe_name = re.sub(r'[\\/*?:"<>|() ]', "_", str(model_name))  # Sanitize model name for filename
        model_path = os.path.join(output_dir, f"AutoML_best_{safe_name}_model.joblib")  # Build model file path
        scaler_path = os.path.join(output_dir, f"AutoML_best_{safe_name}_scaler.joblib")  # Build scaler file path

        if dataset_file is not None:
            stacking_output_dir = get_stacking_output_dir(dataset_file, CONFIG)
        else:
            fa_parent = None
            for p in Path(output_dir).parents:
                if p.name == "Feature_Analysis":
                    fa_parent = str(p)
                    break
            if fa_parent is None:
                raise RuntimeError("Cannot derive Feature_Analysis root from output_dir for safe write")
            stacking_output_dir = str((Path(fa_parent).parent / CONFIG.get("stacking", {}).get("results_dir")).resolve())
        validate_output_path(stacking_output_dir, model_path)
        dump(model, model_path)  # Export model to disk

        if scaler is not None:  # If scaler is provided
            validate_output_path(stacking_output_dir, scaler_path)
            dump(scaler, scaler_path)  # Export scaler to disk

        meta_path = os.path.join(output_dir, f"AutoML_best_{safe_name}_meta.json")  # Build metadata file path
        meta = {  # Build metadata dictionary
            "model_name": model_name,  # Model name
            "features": feature_names,  # Feature names used
            "n_features": len(feature_names),  # Number of features
        }  # Metadata content

        validate_output_path(stacking_output_dir, meta_path)
        with open(meta_path, "w", encoding="utf-8") as f:  # Open metadata file
            json.dump(meta, f, indent=2)  # Write metadata JSON

        print(
            f"{BackgroundColors.GREEN}AutoML best model exported to: {BackgroundColors.CYAN}{model_path}{Style.RESET_ALL}"
        )  # Output export confirmation

        return (model_path, scaler_path)  # Return file paths
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_automl_results_list(best_model_name, best_params, individual_metrics, stacking_metrics, stacking_config, file_path, feature_names, n_train, n_test, config=None):
    """
    Builds the results list for AutoML CSV export matching existing results format.

    :param best_model_name: Name of the best individual model
    :param best_params: Best hyperparameters for the individual model
    :param individual_metrics: Metrics from individual model test evaluation
    :param stacking_metrics: Metrics from stacking model test evaluation (or None)
    :param stacking_config: Best stacking configuration (or None)
    :param file_path: Path to the dataset file
    :param feature_names: List of feature names
    :param n_train: Number of training samples
    :param n_test: Number of test samples
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: List of result dictionaries for CSV export
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        results = []  # Initialize results list

        individual_entry = {  # Build individual model result entry
            "model": best_model_name,  # Model class name
            "dataset": os.path.relpath(file_path),  # Dataset relative path
            "feature_set": "AutoML",  # Feature set label
            "classifier_type": "AutoML_Individual",  # Classifier type
            "model_name": f"AutoML_{best_model_name}",  # Prefixed model name
            "data_source": "Original",  # Data source label
            "experiment_id": None,  # No experiment ID for standalone AutoML runs
            "experiment_mode": "original_only",  # AutoML runs on original data only
            "augmentation_ratio": None,  # No augmentation ratio for AutoML
            "n_features": len(feature_names),  # Number of features
            "n_samples_train": n_train,  # Training sample count
            "n_samples_test": n_test,  # Test sample count
            "accuracy": truncate_value(individual_metrics["accuracy"]),  # Accuracy
            "precision": truncate_value(individual_metrics["precision"]),  # Precision
            "recall": truncate_value(individual_metrics["recall"]),  # Recall
            "f1_score": truncate_value(individual_metrics["f1_score"]),  # F1 score
            "fpr": truncate_value(individual_metrics["fpr"]),  # False positive rate
            "fnr": truncate_value(individual_metrics["fnr"]),  # False negative rate
            "elapsed_time_s": individual_metrics["elapsed_time_s"],  # Elapsed time
            "cv_method": f"Optuna({config.get("automl", {}).get("n_trials", 50)} trials, {config.get("automl", {}).get("cv_folds", 5)}-fold CV)",  # CV method description
            "top_features": json.dumps(feature_names),  # Feature names as JSON
            "rfe_ranking": None,  # No RFE ranking for AutoML
            "hyperparameters": json.dumps(best_params),  # Hyperparameters as JSON
            "features_list": feature_names,  # Feature names list
        }  # Individual model result entry
        results.append(individual_entry)  # Add to results list

        if stacking_metrics is not None and stacking_config is not None:  # If stacking results are available
            stacking_entry = {  # Build stacking result entry
                "model": "StackingClassifier",  # Model class name
                "dataset": os.path.relpath(file_path),  # Dataset relative path
                "feature_set": "AutoML",  # Feature set label
                "classifier_type": "AutoML_Stacking",  # Classifier type
                "model_name": "AutoML_StackingClassifier",  # Prefixed model name
                "data_source": "Original",  # Data source label
                "experiment_id": None,  # No experiment ID for standalone AutoML runs
                "experiment_mode": "original_only",  # AutoML runs on original data only
                "augmentation_ratio": None,  # No augmentation ratio for AutoML
                "n_features": len(feature_names),  # Number of features
                "n_samples_train": n_train,  # Training sample count
                "n_samples_test": n_test,  # Test sample count
                "accuracy": truncate_value(stacking_metrics["accuracy"]),  # Accuracy
                "precision": truncate_value(stacking_metrics["precision"]),  # Precision
                "recall": truncate_value(stacking_metrics["recall"]),  # Recall
                "f1_score": truncate_value(stacking_metrics["f1_score"]),  # F1 score
                "fpr": truncate_value(stacking_metrics["fpr"]),  # False positive rate
                "fnr": truncate_value(stacking_metrics["fnr"]),  # False negative rate
                "elapsed_time_s": stacking_metrics["elapsed_time_s"],  # Elapsed time
                "cv_method": f"Optuna({config.get("automl", {}).get("stacking_trials", 20)} trials, {config.get("automl", {}).get("cv_folds", 5)}-fold CV)",  # CV method description
                "top_features": json.dumps(feature_names),  # Feature names as JSON
                "rfe_ranking": None,  # No RFE ranking for AutoML
                "hyperparameters": json.dumps(stacking_config, default=str),  # Stacking config as JSON
                "features_list": feature_names,  # Feature names list
            }  # Stacking result entry
            results.append(stacking_entry)  # Add stacking to results list

        return results  # Return results list
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def save_automl_results(csv_path, results_list, config=None):
    """
    Saves AutoML results to a dedicated CSV file in the Feature_Analysis/AutoML directory.

    :param csv_path: Path to the original dataset CSV file
    :param results_list: List of result dictionaries to save
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        if not results_list:  # If no results to save
            return  # Exit early

        file_path_obj = Path(csv_path)  # Create Path object for dataset file
        automl_dir = file_path_obj.parent / "Feature_Analysis" / "AutoML"  # Build AutoML output directory
        os.makedirs(automl_dir, exist_ok=True)  # Ensure directory exists
        output_path = automl_dir / config.get("automl", {}).get("results_filename", "AutoML_Results.csv")  # Build output file path

        df = pd.DataFrame(results_list)  # Convert results to DataFrame
        column_order = list(config.get("stacking", {}).get("results_csv_columns", []))  # Use canonical column ordering
        existing_columns = [col for col in column_order if col in df.columns]  # Filter to existing columns
        df = df[existing_columns + [c for c in df.columns if c not in existing_columns]]  # Reorder columns

        df = add_hardware_column(df, existing_columns)  # Add hardware specifications column

        generate_csv_and_image(df, str(output_path), is_visualizable=True, index=False, encoding="utf-8")  # Save AutoML results CSV and generate PNG image

        print(
            f"\n{BackgroundColors.GREEN}AutoML results saved to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
        )  # Output save confirmation
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_automl_stacking_phase(X_train_scaled, y_train_arr, X_test_scaled, y_test_arr, model_study, file, config=None):
    """
    Run AutoML stacking search and evaluation phase if stacking trials are enabled.

    :param X_train_scaled: Scaled training features
    :param y_train_arr: Training target as numpy array
    :param X_test_scaled: Scaled test features
    :param y_test_arr: Test target as numpy array
    :param model_study: Optuna study from model search phase
    :param file: Path to the dataset file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple of (stacking_config, stacking_metrics, stacking_study)
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        stacking_config = None  # Initialize stacking config as None
        stacking_metrics = None  # Initialize stacking metrics as None
        stacking_study = None  # Initialize stacking study as None

        if config.get("automl", {}).get("stacking_trials", 20) > 0:  # If stacking search is enabled
            stacking_config, stacking_study = run_automl_stacking_search(
                X_train_scaled, y_train_arr, model_study, file, config=config
            )  # Run stacking search optimization

            if stacking_config is not None:  # If stacking search succeeded
                best_stacking_model = build_automl_stacking_model(stacking_config, config=config)  # Build best stacking model

                print(
                    f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Evaluating AutoML best stacking model on test set...{Style.RESET_ALL}"
                )  # Output evaluation message

                stacking_metrics = evaluate_automl_model_on_test(
                    best_stacking_model, "AutoML_Stacking", X_train_scaled, y_train_arr, X_test_scaled, y_test_arr
                )  # Evaluate stacking model on test set

        return stacking_config, stacking_metrics, stacking_study  # Return stacking results tuple
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def export_automl_pipeline_artifacts(model_study, stacking_study, stacking_config, stacking_metrics, best_model_name, best_params, individual_metrics, best_individual_model, scaler, file, feature_names, X_train_scaled, y_train_arr, automl_output_dir, config=None):
    """
    Export all AutoML pipeline artifacts including search history, configs, and models.

    :param model_study: Optuna study from model search phase
    :param stacking_study: Optuna study from stacking search phase or None
    :param stacking_config: Best stacking configuration or None
    :param stacking_metrics: Stacking evaluation metrics or None
    :param best_model_name: Name of the best individual model
    :param best_params: Best individual model parameters
    :param individual_metrics: Individual model evaluation metrics
    :param best_individual_model: Best individual model instance
    :param scaler: Fitted scaler for feature transformation
    :param file: Path to the dataset file
    :param feature_names: List of feature column names
    :param X_train_scaled: Scaled training features
    :param y_train_arr: Training target as numpy array
    :param automl_output_dir: Output directory for AutoML artifacts
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        export_automl_search_history(model_study, automl_output_dir, "model_search", dataset_file=file)  # Export model search history

        if stacking_study is not None:  # If stacking study exists
            export_automl_search_history(stacking_study, automl_output_dir, "stacking_search", dataset_file=file)  # Export stacking search history

        export_automl_best_config(
            best_model_name, best_params, individual_metrics, stacking_config, automl_output_dir, feature_names, dataset_file=file
        )  # Export best configuration to file

        export_automl_best_model(
            best_individual_model, scaler, automl_output_dir, best_model_name, feature_names, dataset_file=file
        )  # Export best individual model to file

        if stacking_config is not None and stacking_metrics is not None:  # If stacking was successful
            best_stacking_model_final = build_automl_stacking_model(stacking_config, config=config)  # Rebuild stacking model for export
            sys.stdout.flush()  # Flush stdout before stacking training to ensure logs are visible under nohup
            best_stacking_model_final.fit(X_train_scaled, y_train_arr)  # Fit stacking model on full training data
            export_automl_best_model(
                best_stacking_model_final, scaler, automl_output_dir, "AutoML_Stacking", feature_names, dataset_file=file
            )  # Export best stacking model to file
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_automl_model_search_and_eval(X_train_scaled, y_train_arr, X_test_scaled, y_test_arr, file):
    """
    Executes AutoML phase 1 model search, instantiates the best model, and evaluates it on the test set.

    :param X_train_scaled: Scaled training feature matrix
    :param y_train_arr: Training target labels as numpy array
    :param X_test_scaled: Scaled test feature matrix
    :param y_test_arr: Test target labels as numpy array
    :param file: Path to the dataset file for logging context
    :return: Tuple (best_model_name, best_params, model_study, best_individual_model, individual_metrics) or None if search fails
    """

    try:
        best_model_name, best_params, model_study = run_automl_model_search(X_train_scaled, y_train_arr, file)  # Run Optuna-based model search to find best model and hyperparameters

        if best_model_name is None:  # If model search found no valid model
            print(
                f"{BackgroundColors.RED}AutoML pipeline aborted: model search failed.{Style.RESET_ALL}"
            )  # Output failure message to terminal
            return None  # Signal caller to abort pipeline

        best_individual_model = create_model_from_params(best_model_name, best_params)  # Instantiate best model using found parameters

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Evaluating AutoML best individual model on test set...{Style.RESET_ALL}"
        )  # Output evaluation start message

        individual_metrics = evaluate_automl_model_on_test(
            best_individual_model, best_model_name, X_train_scaled, y_train_arr, X_test_scaled, y_test_arr
        )  # Evaluate the instantiated best individual model on the held-out test set

        return (best_model_name, best_params, model_study, best_individual_model, individual_metrics)  # Return all search and evaluation artifacts
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def prepare_automl_training_data(df, config=None):
    """
    Extracts features and target from DataFrame, validates class count, scales and splits data for AutoML.

    :param df: Preprocessed DataFrame with numeric features and a target in the last column
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (X_train_scaled, X_test_scaled, y_train_arr, y_test_arr, scaler) or None if target has fewer than 2 classes
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        X_full = df.select_dtypes(include=np.number).iloc[:, :-1]  # Extract all numeric columns except the last as features
        y = df.iloc[:, -1]  # Extract the last column as the target

        if len(np.unique(y)) < 2:  # If target has fewer than 2 unique classes
            print(
                f"{BackgroundColors.RED}AutoML: Target has only one class. Skipping.{Style.RESET_ALL}"
            )  # Output single-class error message
            return None  # Signal caller to abort the pipeline

        X_train_scaled, X_test_scaled, y_train, y_test, scaler = scale_and_split(X_full, y, config=config)  # Scale features and split into train/test sets

        y_train_arr = np.asarray(y_train)  # Convert training target to numpy array for model compatibility
        y_test_arr = np.asarray(y_test)  # Convert test target to numpy array for model compatibility

        return (X_train_scaled, X_test_scaled, y_train_arr, y_test_arr, scaler)  # Return prepared training data tuple
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def print_automl_pipeline_header(file):
    """
    Prints the formatted header block for the AutoML pipeline execution.

    :param file: Path to the dataset file whose basename is shown in the header
    :return: None
    """

    try:
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*80}{Style.RESET_ALL}"
        )  # Print top separator line
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}AutoML Pipeline - {BackgroundColors.CYAN}{os.path.basename(file)}{Style.RESET_ALL}"
        )  # Print pipeline title with dataset filename
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*80}{Style.RESET_ALL}\n"
        )  # Print bottom separator line
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_automl_pipeline(file, df, feature_names, data_source_label="Original", config=None):
    """
    Runs the complete AutoML pipeline: model search, stacking optimization, evaluation, and export.

    :param file: Path to the dataset file being processed
    :param df: Preprocessed DataFrame with features and target
    :param feature_names: List of feature column names
    :param data_source_label: Label identifying the data source
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary containing AutoML results, or None on failure
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        print_automl_pipeline_header(file)  # Print AutoML pipeline header with dataset filename

        automl_start = time.time()  # Record pipeline start time

        training_data = prepare_automl_training_data(df, config=config)  # Prepare scaled training and test splits, returns None for single-class targets
        if training_data is None:  # If preparation failed due to insufficient classes
            return None  # Abort pipeline early

        X_train_scaled, X_test_scaled, y_train_arr, y_test_arr, scaler = training_data  # Unpack prepared training data tuple

        send_telegram_message(TELEGRAM_BOT, f"Starting AutoML pipeline for {os.path.basename(file)}")  # Notify via Telegram

        search_result = run_automl_model_search_and_eval(X_train_scaled, y_train_arr, X_test_scaled, y_test_arr, file)  # Run model search and evaluate the best individual model
        if search_result is None:  # If model search or evaluation failed
            return None  # Abort pipeline

        best_model_name, best_params, model_study, best_individual_model, individual_metrics = search_result  # Unpack search and evaluation results

        stacking_config, stacking_metrics, stacking_study = run_automl_stacking_phase(
            X_train_scaled, y_train_arr, X_test_scaled, y_test_arr, model_study, file, config=config
        )  # Phase 2: Run stacking search and evaluation if enabled

        file_path_obj = Path(file)  # Create Path object for file
        automl_output_dir = str(file_path_obj.parent / "Feature_Analysis" / "AutoML")  # Build AutoML output directory

        export_automl_pipeline_artifacts(
            model_study, stacking_study, stacking_config, stacking_metrics,
            best_model_name, best_params, individual_metrics, best_individual_model,
            scaler, file, feature_names, X_train_scaled, y_train_arr, automl_output_dir, config=config,
        )  # Export all AutoML artifacts (search history, config, models)

        results_list = build_automl_results_list(
            best_model_name, best_params, individual_metrics, stacking_metrics, stacking_config,
            file, feature_names, len(y_train_arr), len(y_test_arr), config=config
        )  # Build results list for CSV

        save_automl_results(file, results_list, config=config)  # Save AutoML results to CSV

        automl_elapsed = time.time() - automl_start  # Calculate total AutoML pipeline time

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}AutoML pipeline completed in {BackgroundColors.CYAN}{calculate_execution_time(0, automl_elapsed)}{Style.RESET_ALL}"
        )  # Output completion message

        send_telegram_message(
            TELEGRAM_BOT, f"AutoML pipeline completed for {os.path.basename(file)} in {calculate_execution_time(0, automl_elapsed)}. Best model: {best_model_name} (F1: {truncate_value(individual_metrics['f1_score'])})"
        )  # Send Telegram notification

        return {  # Return AutoML results summary
            "best_model_name": best_model_name,  # Best model name
            "best_params": best_params,  # Best parameters
            "individual_metrics": individual_metrics,  # Individual model metrics
            "stacking_config": stacking_config,  # Stacking configuration
            "stacking_metrics": stacking_metrics,  # Stacking metrics
        }  # Return results dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def sanitize_and_verify_feature_selections(ga_selected_features, rfe_selected_features, feature_names, config=None):
    """
    Sanitize and verify GA and RFE feature selections against available features.

    :param ga_selected_features: Features selected by genetic algorithm
    :param rfe_selected_features: Features selected by RFE
    :param feature_names: List of available feature names
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple of (sanitized_ga_features, sanitized_rfe_features)
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        if ga_selected_features:  # If GA features are provided
            ga_selected_features = sanitize_feature_names(ga_selected_features)  # Sanitize GA feature names
        if rfe_selected_features:  # If RFE features are provided
            rfe_selected_features = sanitize_feature_names(rfe_selected_features)  # Sanitize RFE feature names

        try:  # Verify GA features exist in dataset
            if ga_selected_features:  # If GA features remain after sanitization
                ga_selected_features = verify_selected_features_exist(ga_selected_features, feature_names, "GA")  # Verify GA features exist
        except ValueError as e:  # All GA features missing from dataset
            verbose_output(str(e), config=config)  # Log warning about missing GA features
            ga_selected_features = []  # Reset GA features to empty list

        try:  # Verify RFE features exist in dataset
            if rfe_selected_features:  # If RFE features remain after sanitization
                rfe_selected_features = verify_selected_features_exist(rfe_selected_features, feature_names, "RFE")  # Verify RFE features exist
        except ValueError as e:  # All RFE features missing from dataset
            verbose_output(str(e), config=config)  # Log warning about missing RFE features
            rfe_selected_features = []  # Reset RFE features to empty list

        return ga_selected_features, rfe_selected_features  # Return cleaned and verified feature selections
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def prepare_evaluation_data_splits(df, df_augmented_for_training=None, config=None):
    """
    Prepare training/test data splits with optional augmented data merging.

    :param df: DataFrame with the original dataset
    :param df_augmented_for_training: Optional augmented DataFrame to merge into training set only
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple of (X_train_scaled, X_test_scaled, y_train, y_test, scaler) or None if single-class target
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        X_full = df.select_dtypes(include=np.number).iloc[:, :-1]  # Extract numeric feature columns excluding the last (target)
        y = df.iloc[:, -1]  # Extract target column as the last column

        if len(np.unique(y)) < 2:  # Verify if there is more than one class
            print(
                f"{BackgroundColors.RED}Target column has only one class. Cannot perform classification. Skipping.{Style.RESET_ALL}"
            )  # Output the error message
            return None  # Return None to signal classification is not possible

        if df_augmented_for_training is not None:  # If augmented data provided for training enhancement
            X_augmented = df_augmented_for_training.select_dtypes(include=np.number).iloc[:, :-1]  # Extract augmented features (numeric only)
            y_augmented = df_augmented_for_training.iloc[:, -1]  # Extract augmented target
            X_train_scaled, X_test_scaled, y_train, y_test, scaler = scale_and_split(
                X_full, y, config=config, X_augmented=X_augmented, y_augmented=y_augmented
            )  # Scale and split with augmented data merged into training set only
        else:  # No augmented data provided
            X_train_scaled, X_test_scaled, y_train, y_test, scaler = scale_and_split(
                X_full, y, config=config
            )  # Scale and split the data normally (original-only)

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler  # Return the prepared data splits
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_evaluation_stacking_model(base_models, config=None):
    """
    Build a StackingClassifier from base models excluding SVM.

    :param base_models: Dictionary of base models
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: StackingClassifier instance
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        estimators = [
            (name, model) for name, model in base_models.items() if name != "SVM"
        ]  # Define estimators list excluding SVM which is incompatible with stacking

        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=config.get("evaluation", {}).get("n_jobs", -1)),
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
            n_jobs=1,
        )  # Define the Stacking Classifier model with sequential CV folds to prevent nested loky deadlock

        return stacking_model  # Return the constructed stacking model
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def assemble_feature_sets(X_train_scaled, X_test_scaled, feature_names, ga_selected_features, pca_n_components, rfe_selected_features, file):
    """
    Build feature sets dictionary from full, GA, PCA, and RFE feature subsets.

    :param X_train_scaled: Scaled training features
    :param X_test_scaled: Scaled test features
    :param feature_names: List of all feature names
    :param ga_selected_features: GA selected features
    :param pca_n_components: Number of PCA components
    :param rfe_selected_features: RFE selected features
    :param file: Path to dataset file (for PCA artifact lookup)
    :return: Sorted dictionary mapping feature set names to (X_train, X_test, feature_names) tuples
    """

    try:
        X_train_pca, X_test_pca = apply_pca_transformation(
            X_train_scaled, X_test_scaled, pca_n_components, file
        )  # Apply PCA transformation if applicable

        X_train_ga, ga_actual_features = get_feature_subset(X_train_scaled, ga_selected_features, feature_names)  # Get GA feature subset for training
        X_test_ga, _ = get_feature_subset(X_test_scaled, ga_selected_features, feature_names)  # Get GA feature subset for testing

        X_train_rfe, rfe_actual_features = get_feature_subset(X_train_scaled, rfe_selected_features, feature_names)  # Get RFE feature subset for training
        X_test_rfe, _ = get_feature_subset(X_test_scaled, rfe_selected_features, feature_names)  # Get RFE feature subset for testing

        feature_sets = {  # Dictionary of feature sets to evaluate
            "Full Features": (X_train_scaled, X_test_scaled, feature_names),  # All features with names
            "GA Features": (X_train_ga, X_test_ga, ga_actual_features),  # GA subset with actual names
            "PCA Components": (
                (X_train_pca, X_test_pca, None) if X_train_pca is not None else None
            ),  # PCA components (only if PCA was applied)
            "RFE Features": (X_train_rfe, X_test_rfe, rfe_actual_features),  # RFE subset with actual names
        }  # Build the complete feature sets dictionary

        feature_sets = {
            k: v for k, v in feature_sets.items() if v is not None
        }  # Remove any None entries (e.g., PCA if not applied)
        feature_sets = dict(sorted(feature_sets.items()))  # Sort the feature sets by name

        return feature_sets  # Return the assembled and sorted feature sets dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_classifier_result_entry(model_class, file, execution_mode_str, attack_types_combined, feature_set_name, classifier_type, model_name, data_source_label, experiment_id, experiment_mode, augmentation_ratio, n_features, n_samples_train, n_samples_test, metrics_tuple, subset_feature_names, hyperparams_map=None):
    """
    Build a standardized result entry dictionary for classifier evaluation results.

    :param model_class: Class name of the model
    :param file: Path to the dataset file
    :param execution_mode_str: Execution mode string ('binary' or 'multi-class')
    :param attack_types_combined: List of attack types for multi-class or None
    :param feature_set_name: Name of the feature set used
    :param classifier_type: Type of classifier ('Individual' or 'Stacking')
    :param model_name: Name of the model
    :param data_source_label: Label for the data source
    :param experiment_id: Unique experiment identifier
    :param experiment_mode: Experiment mode string
    :param augmentation_ratio: Augmentation ratio or None
    :param n_features: Number of features used
    :param n_samples_train: Number of training samples
    :param n_samples_test: Number of test samples
    :param metrics_tuple: Tuple of (accuracy, precision, recall, f1, fpr, fnr, elapsed, ...)
    :param subset_feature_names: List of feature names used
    :param hyperparams_map: Dictionary mapping model names to hyperparameters
    :return: Dictionary containing the result entry
    """

    try:
        acc, prec, rec, f1, fpr, fnr, elapsed = metrics_tuple[:7]  # Unpack the first 7 metrics from the tuple
        return {
            "model": model_class,  # Model class name for identification
            "dataset": os.path.relpath(file),  # Relative dataset path for portability
            "execution_mode": execution_mode_str,  # Execution mode (binary or multi-class)
            "attack_types_combined": json.dumps(attack_types_combined) if attack_types_combined else None,  # JSON-serialized attack types or None
            "feature_set": feature_set_name,  # Name of the feature set evaluated
            "classifier_type": classifier_type,  # Classifier type (Individual or Stacking)
            "model_name": model_name,  # Model name for result identification
            "data_source": data_source_label,  # Data source label for experiment traceability
            "experiment_id": experiment_id,  # Unique experiment identifier
            "experiment_mode": experiment_mode,  # Experiment mode (original_only or original_plus_augmented)
            "augmentation_ratio": augmentation_ratio,  # Augmentation ratio or None for original-only
            "n_features": n_features,  # Number of features used in evaluation
            "n_samples_train": n_samples_train,  # Number of training samples
            "n_samples_test": n_samples_test,  # Number of test samples
            "accuracy": truncate_value(acc),  # Truncated accuracy metric
            "precision": truncate_value(prec),  # Truncated precision metric
            "recall": truncate_value(rec),  # Truncated recall metric
            "f1_score": truncate_value(f1),  # Truncated F1 score metric
            "fpr": truncate_value(fpr),  # Truncated false positive rate
            "fnr": truncate_value(fnr),  # Truncated false negative rate
            "elapsed_time_s": int(round(elapsed)),  # Rounded elapsed time in seconds
            "cv_method": f"StratifiedKFold(n_splits=10)",  # Cross-validation method description
            "top_features": json.dumps(subset_feature_names),  # JSON-serialized subset feature names
            "rfe_ranking": None,  # RFE ranking placeholder (not computed here)
            "hyperparameters": json.dumps(hyperparams_map.get(model_name)) if hyperparams_map and hyperparams_map.get(model_name) is not None else None,  # JSON-serialized hyperparameters or None
            "features_list": subset_feature_names,  # Raw list of feature names for downstream use
        }  # Return the constructed result entry dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def submit_classifier_evaluations_to_pool(executor, individual_models, current_combination, name, X_train_df, y_train, X_test_df, y_test, file, scaler, subset_feature_names, total_steps):
    """
    Submit each individual classifier to the thread pool executor and return the futures map alongside the updated combination counter.

    :param executor: Active ThreadPoolExecutor instance to submit evaluation tasks to.
    :param individual_models: Dictionary mapping model names to model instances.
    :param current_combination: Current global combination counter before submission begins.
    :param name: Name of the current feature set being evaluated.
    :param X_train_df: Training feature DataFrame with named columns.
    :param y_train: Training target labels.
    :param X_test_df: Test feature DataFrame with named columns.
    :param y_test: Test target labels.
    :param file: Path to the dataset file used for evaluation.
    :param scaler: Fitted scaler for dataset preprocessing.
    :param subset_feature_names: List of feature names for the current subset.
    :param total_steps: Total number of evaluation steps for Telegram progress messages.
    :return: Tuple of (future_to_model, current_combination) where future_to_model maps each submitted future to its (model_name, model_class, combination_index) metadata.
    """

    future_to_model = {}  # Map futures to (model_name, model_class, combination_index)
    for model_name, model in individual_models.items():  # Iterate over each individual model
        send_telegram_message(TELEGRAM_BOT, f"Starting combination {current_combination}/{total_steps}: {name} - {model_name}")  # Notify telegram about evaluation start
        future = executor.submit(
            evaluate_individual_classifier,
            model,
            model_name,
            X_train_df.values,
            y_train,
            X_test_df.values,
            y_test,
            file,
            scaler,
            subset_feature_names,
            name,
        )  # Submit evaluation task to thread pool using numpy arrays
        future_to_model[future] = (model_name, model.__class__.__name__, current_combination)  # Store future with metadata
        current_combination += 1  # Advance the global combination counter
    return future_to_model, current_combination  # Return futures map and updated combination counter


def collect_classifier_results_from_futures(future_to_model, individual_models, name, X_test_subset, X_train_n_cols, file, execution_mode_str, attack_types_combined, data_source_label, experiment_id, experiment_mode, augmentation_ratio, y_train, y_test, hyperparams_map, subset_feature_names, total_steps, progress_bar, config):
    """
    Collect results from completed classifier futures, build standardized result entries, and optionally run explainability for each model.

    :param future_to_model: Dictionary mapping futures to (model_name, model_class, combination_index) metadata.
    :param individual_models: Dictionary mapping model names to model instances, used for explainability retrieval.
    :param name: Name of the current feature set being evaluated.
    :param X_test_subset: Test feature array passed to the explainability pipeline.
    :param X_train_n_cols: Number of training columns used for result entry metadata.
    :param file: Path to the dataset file for result metadata and explainability output.
    :param execution_mode_str: Execution mode string ('binary' or 'multi-class').
    :param attack_types_combined: List of attack types for multi-class, or None for binary.
    :param data_source_label: Label identifying the data source for result traceability.
    :param experiment_id: Unique experiment identifier for traceability.
    :param experiment_mode: Experiment mode string ('original_only' or 'original_plus_augmented').
    :param augmentation_ratio: Augmentation ratio float, or None for original-only experiments.
    :param y_train: Training target labels used to compute training set size.
    :param y_test: Test target labels used for evaluation and size metadata.
    :param hyperparams_map: Dictionary mapping model names to their hyperparameter dictionaries.
    :param subset_feature_names: List of feature names for the current subset.
    :param total_steps: Total number of evaluation steps for Telegram progress messages.
    :param progress_bar: tqdm progress bar advanced after each completed future.
    :param config: Configuration dictionary used for explainability settings and verbose output.
    :return: Dictionary mapping (feature_set_name, model_name) tuples to standardized result entry dicts.
    """

    results_dict = {}  # Accumulate result entries for this feature set
    for future in concurrent.futures.as_completed(future_to_model):  # Process results as futures complete
        model_name, model_class, comb_idx = future_to_model[future]  # Retrieve model metadata for this future
        metrics = future.result()  # Collect metrics tuple from completed evaluation
        result_entry = build_classifier_result_entry(
            model_class, file, execution_mode_str, attack_types_combined, name, "Individual",
            model_name, data_source_label, experiment_id, experiment_mode, augmentation_ratio,
            X_train_n_cols, len(y_train), len(y_test), metrics, subset_feature_names,
            hyperparams_map=hyperparams_map,
        )  # Build standardized result entry for this individual classifier
        results_dict[(name, model_name)] = result_entry  # Store result keyed by (feature_set, model_name)
        send_telegram_message(TELEGRAM_BOT, f"Finished combination {comb_idx}/{total_steps}: {name} - {model_name} with F1: {truncate_value(metrics[3])} in {calculate_execution_time(0, metrics[6])}")  # Notify telegram about completion
        print(
            f"    {BackgroundColors.GREEN}{model_name} Accuracy: {BackgroundColors.CYAN}{truncate_value(metrics[0])}{Style.RESET_ALL}"
        )  # Output individual model accuracy
        progress_bar.update(1)  # Advance progress bar by one step

        if config.get("explainability", {}).get("enabled", False) and experiment_mode == "original_only":  # Only run explainability on original data
            try:  # Attempt to run explainability pipeline for this model
                trained_model = individual_models[model_name]  # Retrieve trained model object for explainability
                run_explainability_pipeline(
                    trained_model,
                    model_name,
                    X_test_subset,
                    y_test,
                    subset_feature_names,
                    file,
                    name,
                    execution_mode_str,
                    config
                )  # Run explainability pipeline on original test data only
            except Exception as e:  # If explainability fails
                verbose_output(
                    f"{BackgroundColors.YELLOW}Explainability failed for {model_name}: {e}{Style.RESET_ALL}",
                    config=config
                )  # Log error but continue evaluation
    return results_dict  # Return accumulated result entries


def run_individual_classifiers_for_feature_set(name, individual_models, X_train_df, y_train, X_test_df, y_test, X_test_subset, X_train_n_cols, file, execution_mode_str, attack_types_combined, data_source_label, experiment_id, experiment_mode, augmentation_ratio, hyperparams_map, scaler, subset_feature_names, total_steps, current_combination, progress_bar, config=None):
    """
    Evaluates all individual classifiers for a feature set sequentially, collects results, and runs explainability.

    :param name: Name of the current feature set being evaluated
    :param individual_models: Dictionary mapping model names to model objects
    :param X_train_df: Training feature DataFrame with named columns
    :param y_train: Training target labels
    :param X_test_df: Test feature DataFrame with named columns
    :param y_test: Test target labels
    :param X_test_subset: Test feature array for explainability pipeline input
    :param X_train_n_cols: Number of training columns (features) used for result entry metadata
    :param file: Path to the dataset file for export and result metadata
    :param execution_mode_str: Execution mode string ('binary' or 'multi-class')
    :param attack_types_combined: List of attack types for multi-class or None for binary
    :param data_source_label: Label identifying the data source for result traceability
    :param experiment_id: Unique experiment identifier for traceability
    :param experiment_mode: Experiment mode string ('original_only' or 'original_plus_augmented')
    :param augmentation_ratio: Augmentation ratio float or None for original-only experiments
    :param hyperparams_map: Dictionary mapping model names to their hyperparameter dicts
    :param scaler: Fitted scaler used for dataset preprocessing
    :param subset_feature_names: List of feature names for the current subset
    :param total_steps: Total number of evaluation steps for Telegram progress messages
    :param current_combination: Current combination index counter for progress messages
    :param progress_bar: tqdm progress bar to update after each model evaluation
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (results_dict, next_current_combination) where results_dict maps (name, model_name) to result entries
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        progress_bar.set_description(
            f"{data_source_label} - {name} (Individual)"
        )  # Update progress bar description for individual model evaluations

        results_dict = {}  # Accumulate result entries for this feature set

        for model_name, model in individual_models.items():  # Iterate over each individual model sequentially to prevent loky deadlock
            send_telegram_message(TELEGRAM_BOT, f"Starting combination {current_combination}/{total_steps}: {name} - {model_name}")  # Notify Telegram about evaluation start
            sys.stdout.flush()  # Flush stdout before each classifier to ensure logs are visible under nohup

            metrics = evaluate_individual_classifier(
                model,
                model_name,
                X_train_df.values,
                y_train,
                X_test_df.values,
                y_test,
                file,
                scaler,
                subset_feature_names,
                name,
                config=config,
            )  # Evaluate individual classifier sequentially using numpy arrays

            model_class = model.__class__.__name__  # Retrieve model class name for result entry
            result_entry = build_classifier_result_entry(
                model_class, file, execution_mode_str, attack_types_combined, name, "Individual",
                model_name, data_source_label, experiment_id, experiment_mode, augmentation_ratio,
                X_train_n_cols, len(y_train), len(y_test), metrics, subset_feature_names,
                hyperparams_map=hyperparams_map,
            )  # Build standardized result entry for this individual classifier
            results_dict[(name, model_name)] = result_entry  # Store result keyed by (feature_set, model_name)

            send_telegram_message(TELEGRAM_BOT, f"Finished combination {current_combination}/{total_steps}: {name} - {model_name} with F1: {truncate_value(metrics[3])} in {calculate_execution_time(0, metrics[6])}")  # Notify Telegram about completion
            print(
                f"    {BackgroundColors.GREEN}{model_name} Accuracy: {BackgroundColors.CYAN}{truncate_value(metrics[0])}{Style.RESET_ALL}"
            )  # Output individual model accuracy
            progress_bar.update(1)  # Advance progress bar by one step

            if config.get("explainability", {}).get("enabled", False) and experiment_mode == "original_only":  # Only run explainability on original data
                try:  # Attempt to run explainability pipeline for this model
                    run_explainability_pipeline(
                        model,
                        model_name,
                        X_test_subset,
                        y_test,
                        subset_feature_names,
                        file,
                        name,
                        execution_mode_str,
                        config
                    )  # Run explainability pipeline on original test data only
                except Exception as e:  # If explainability fails
                    verbose_output(
                        f"{BackgroundColors.YELLOW}Explainability failed for {model_name}: {e}{Style.RESET_ALL}",
                        config=config
                    )  # Log error but continue evaluation

            current_combination += 1  # Advance the global combination counter

        return (results_dict, current_combination)  # Return accumulated results and updated combination counter
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_stacking_evaluation_for_feature_set(name, stacking_model, X_train_df, y_train, X_test_df, y_test, X_test_subset, X_train_n_cols, file, execution_mode_str, attack_types_combined, data_source_label, experiment_id, experiment_mode, augmentation_ratio, scaler, subset_feature_names, total_steps, current_combination, progress_bar, config=None):
    """
    Evaluates the stacking classifier for one feature set, exports the model, generates metric plots, and returns the result entry.

    :param name: Name of the current feature set being evaluated
    :param stacking_model: Fitted stacking classifier model object
    :param X_train_df: Training feature DataFrame with named columns
    :param y_train: Training target labels
    :param X_test_df: Test feature DataFrame with named columns
    :param y_test: Test target labels
    :param X_test_subset: Test feature array for explainability pipeline input
    :param X_train_n_cols: Number of training columns used for result entry metadata
    :param file: Path to the dataset file for export and result metadata
    :param execution_mode_str: Execution mode string ('binary' or 'multi-class')
    :param attack_types_combined: List of attack types for multi-class or None for binary
    :param data_source_label: Label identifying the data source for result traceability
    :param experiment_id: Unique experiment identifier for traceability
    :param experiment_mode: Experiment mode string ('original_only' or 'original_plus_augmented')
    :param augmentation_ratio: Augmentation ratio float or None for original-only experiments
    :param scaler: Fitted scaler used for dataset preprocessing
    :param subset_feature_names: List of feature names for the current subset
    :param total_steps: Total number of evaluation steps for Telegram progress messages
    :param current_combination: Current combination index counter for progress messages
    :param progress_bar: tqdm progress bar to update after stacking evaluation
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (stacking_result_entry, next_current_combination) where stacking_result_entry is the standardized result dict
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        print(
            f"  {BackgroundColors.GREEN}Training {BackgroundColors.CYAN}Stacking Classifier{BackgroundColors.GREEN}...{Style.RESET_ALL}"
        )  # Announce the start of stacking classifier training and evaluation
        progress_bar.set_description(
            f"{data_source_label} - {name} (Stacking)"
        )  # Update progress bar description for stacking classifier

        send_telegram_message(TELEGRAM_BOT, f"Starting combination {current_combination}/{total_steps}: {name} - StackingClassifier")  # Notify Telegram about stacking evaluation start

        stacking_metrics = evaluate_stacking_classifier(
            stacking_model, X_train_df, y_train, X_test_df, y_test
        )  # Evaluate stacking model with DataFrames and retrieve metrics tuple

        try:
            dataset_name = os.path.basename(os.path.dirname(file))  # Get dataset directory name from file path
            export_model_and_scaler(stacking_model, scaler, dataset_name, "StackingClassifier", subset_feature_names, best_params=None, feature_set=name, dataset_csv_path=file)  # Export fitted stacking model and scaler to disk
        except Exception:  # If model export fails
            pass  # Continue without exporting

        s_y_pred = stacking_metrics[7] if len(stacking_metrics) > 7 else None  # Extract stacking predictions from metrics tuple for plot generation

        try:  # Attempt to generate metric plots for stacking model
            file_path_obj = Path(file)  # Create Path object for the dataset file
            feature_analysis_dir = file_path_obj.parent / "Feature_Analysis"  # Build Feature_Analysis directory path for outputs
            stacking_output_dir = get_stacking_output_dir(str(file_path_obj), config)  # Get stacking output directory path from config
            generate_and_save_metric_plots(y_test, s_y_pred, config.get("stacking", {}), stacking_output_dir)  # Generate and save metric plots to stacking output directory
        except Exception:  # If metric plot generation fails
            pass  # Continue without plotting

        stacking_result_entry = build_classifier_result_entry(
            stacking_model.__class__.__name__, file, execution_mode_str, attack_types_combined, name, "Stacking",
            "StackingClassifier", data_source_label, experiment_id, experiment_mode, augmentation_ratio,
            X_train_n_cols, len(y_train), len(y_test), stacking_metrics, subset_feature_names,
        )  # Build standardized result entry for the stacking classifier
        send_telegram_message(TELEGRAM_BOT, f"Finished combination {current_combination}/{total_steps}: {name} - StackingClassifier with F1: {truncate_value(stacking_metrics[3])} in {calculate_execution_time(0, stacking_metrics[6])}")  # Notify Telegram about stacking evaluation completion
        print(
            f"    {BackgroundColors.GREEN}Stacking Accuracy: {BackgroundColors.CYAN}{truncate_value(stacking_metrics[0])}{Style.RESET_ALL}"
        )  # Output stacking classifier accuracy
        progress_bar.update(1)  # Advance progress bar after stacking evaluation
        current_combination += 1  # Advance the global combination counter

        if config.get("explainability", {}).get("enabled", False) and experiment_mode == "original_only":  # Only run explainability on original data
            try:  # Attempt to run explainability pipeline for stacking model
                run_explainability_pipeline(
                    stacking_model,
                    "StackingClassifier",
                    X_test_subset,
                    y_test,
                    subset_feature_names,
                    file,
                    name,
                    execution_mode_str,
                    config
                )  # Run explainability pipeline on original test data only for stacking model
            except Exception as e:  # If explainability fails
                verbose_output(
                    f"{BackgroundColors.YELLOW}Explainability failed for StackingClassifier: {e}{Style.RESET_ALL}",
                    config=config
                )  # Log error but continue evaluation

        return (stacking_result_entry, current_combination)  # Return result entry and updated combination counter
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def print_dataset_evaluation_header(data_source_label):
    """
    Prints the formatted header block for a dataset evaluation run.

    :param data_source_label: Label identifying the data source being evaluated
    :return: None
    """

    try:
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*80}{Style.RESET_ALL}"
        )  # Print top separator line for evaluation section
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Evaluating on: {BackgroundColors.CYAN}{data_source_label} Data{Style.RESET_ALL}"
        )  # Print the data source label currently being evaluated
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*80}{Style.RESET_ALL}\n"
        )  # Print bottom separator line for evaluation section
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def setup_feature_set_names(name, X_train_subset, subset_feature_names_list):
    """
    Determine the list of feature names for the given feature set, generating PCA names or generic labels when needed.

    :param name: Name of the current feature set.
    :param X_train_subset: Training feature array used to determine the number of features.
    :param subset_feature_names_list: Pre-computed feature names list, or None/empty for generic generation.
    :return: List of feature name strings for the current subset.
    """

    if name == "PCA Components":  # If the feature set is pca components
        return [f"PC{i+1}" for i in range(X_train_subset.shape[1])]  # Generate pca component names
    return (
        subset_feature_names_list if subset_feature_names_list else [f"feature_{i}" for i in range(X_train_subset.shape[1])]
    )  # Use actual feature names or generate generic ones


def align_feature_names_to_array(X_subset, feature_names, label):
    """
    Align a feature name list to match the number of columns in a feature array.

    :param X_subset: Feature array whose column count is the authoritative shape.
    :param feature_names: List of feature names to align against the array shape.
    :param label: Label string used in the warning message to identify which array is being aligned.
    :return: Feature name list whose length exactly matches X_subset.shape[1].
    """

    n_features_array = X_subset.shape[1]  # Get the number of columns in the feature array
    n_features_names = len(feature_names)  # Get the number of provided feature names

    if n_features_array == n_features_names:  # Verify if sizes already match
        return feature_names  # Return unchanged list when sizes are equal

    print(
        f"{BackgroundColors.YELLOW}[WARNING] Feature name/array mismatch in {label}: "
        f"array has {n_features_array} columns, names list has {n_features_names} entries. "
        f"Applying safe alignment.{Style.RESET_ALL}"
    )  # Log the mismatch details and alignment action

    if n_features_names > n_features_array:  # Verify if names list is longer than array columns
        return feature_names[:n_features_array]  # Truncate names list to match array column count

    extra_names = [f"feature_{i}" for i in range(n_features_names, n_features_array)]  # Generate synthetic names for missing indices
    return feature_names + extra_names  # Extend names list with synthetic entries to match array column count


def convert_subset_to_dataframes(X_train_subset, X_test_subset, subset_feature_names):
    """
    Convert training and test feature arrays to DataFrames using the provided feature name list.

    :param X_train_subset: Training feature array to wrap in a DataFrame.
    :param X_test_subset: Test feature array to wrap in a DataFrame.
    :param subset_feature_names: List of column names to assign to both DataFrames.
    :return: Tuple of (X_train_df, X_test_df) DataFrames with named columns.
    """

    train_names = align_feature_names_to_array(X_train_subset, subset_feature_names, "X_train_subset")  # Align feature names to training array shape
    test_names = align_feature_names_to_array(X_test_subset, subset_feature_names, "X_test_subset")  # Align feature names to test array shape

    X_train_df = pd.DataFrame(X_train_subset, columns=train_names)  # Convert training features to DataFrame with aligned column names
    X_test_df = pd.DataFrame(X_test_subset, columns=test_names)  # Convert test features to DataFrame with aligned column names

    return X_train_df, X_test_df  # Return DataFrames with named columns


def initialize_evaluation_run_state(base_models, feature_sets, data_source_label):
    """
    Build the individual model map, compute total evaluation steps, create the progress bar, and initialize result containers.

    :param base_models: Dictionary of base model name to model instance pairs used for individual evaluation.
    :param feature_sets: Ordered dictionary of feature set name to (X_train_subset, X_test_subset, feature_names) tuples.
    :param data_source_label: String label for the current data source displayed in the progress bar description.
    :return: Tuple of (individual_models, total_steps, progress_bar, all_results, current_combination) ready for the evaluation loop.
    """

    individual_models = {
        k: v for k, v in base_models.items()
    }  # Use the base models (with hyperparameters applied) for individual evaluation

    total_steps = len(feature_sets) * (
        len(individual_models) + 1
    )  # Total steps: models + stacking per feature set

    progress_bar = tqdm(total=total_steps, desc=f"{data_source_label} Data", file=sys.stdout)  # Progress bar for all evaluations

    all_results = {}  # Dictionary to store results: (feature_set, model_name) -> result_entry

    current_combination = 1  # Counter for combination index

    return individual_models, total_steps, progress_bar, all_results, current_combination  # Return all initialized evaluation state variables


def evaluate_single_feature_set(
    idx,
    name,
    X_train_subset,
    X_test_subset,
    subset_feature_names_list,
    individual_models,
    stacking_model,
    y_train,
    y_test,
    file,
    execution_mode_str,
    attack_types_combined,
    data_source_label,
    experiment_id,
    experiment_mode,
    augmentation_ratio,
    hyperparams_map,
    scaler,
    total_steps,
    current_combination,
    progress_bar,
    config=None,
):
    """
    Evaluate all individual classifiers and the stacking model on one non-empty feature subset.

    :param idx: 1-based index of the current feature set in the evaluation loop for logging.
    :param name: Name string identifying the current feature set used for output labeling.
    :param X_train_subset: Training feature matrix restricted to the current feature subset.
    :param X_test_subset: Test feature matrix restricted to the current feature subset.
    :param subset_feature_names_list: List of feature names for the current subset or None to derive from array.
    :param individual_models: Dictionary of model name to model instance pairs for individual evaluation.
    :param stacking_model: Stacking classifier instance built from the base models.
    :param y_train: Training target vector used for fitting all classifiers.
    :param y_test: Test target vector used for evaluating all classifiers.
    :param file: Dataset file path used for model artifact export and result labeling.
    :param execution_mode_str: Execution mode string ('binary' or 'multi-class') selecting the evaluation strategy.
    :param attack_types_combined: List of attack type strings for multi-class mode or None for binary mode.
    :param data_source_label: Data source label string included in result entries for traceability.
    :param experiment_id: Unique experiment identifier string included in result entries for traceability.
    :param experiment_mode: Experiment mode string included in result entries for traceability.
    :param augmentation_ratio: Augmentation ratio float included in result entries or None for original-only.
    :param hyperparams_map: Dictionary mapping model names to best hyperparameter dicts applied before training.
    :param scaler: Fitted StandardScaler instance used to transform subsets when needed.
    :param total_steps: Total number of evaluation steps across all feature sets used by the progress bar.
    :param current_combination: 1-based overall combination counter updated for each model evaluated.
    :param progress_bar: tqdm progress bar instance advanced after each model evaluation.
    :param config: Optional configuration dictionary; falls back to global CONFIG when None.
    :return: Tuple of (individual_results, stacking_result_entry, current_combination) containing per-model result dicts and updated combination counter.
    """

    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Evaluating models on: {BackgroundColors.CYAN}{name} ({X_train_subset.shape[1]} features){Style.RESET_ALL}"
    )  # Output evaluation status

    subset_feature_names = setup_feature_set_names(name, X_train_subset, subset_feature_names_list)  # Determine feature names for this subset

    X_train_df, X_test_df = convert_subset_to_dataframes(X_train_subset, X_test_subset, subset_feature_names)  # Convert feature arrays to dataframes with named columns

    individual_results, current_combination = run_individual_classifiers_for_feature_set(
        name, individual_models, X_train_df, y_train, X_test_df, y_test,
        X_test_subset, X_train_subset.shape[1], file, execution_mode_str, attack_types_combined,
        data_source_label, experiment_id, experiment_mode, augmentation_ratio,
        hyperparams_map, scaler, subset_feature_names, total_steps, current_combination, progress_bar, config=config,
    )  # Evaluate all individual classifiers in parallel and collect their result entries

    stacking_result_entry, current_combination = run_stacking_evaluation_for_feature_set(
        name, stacking_model, X_train_df, y_train, X_test_df, y_test,
        X_test_subset, X_train_subset.shape[1], file, execution_mode_str, attack_types_combined,
        data_source_label, experiment_id, experiment_mode, augmentation_ratio,
        scaler, subset_feature_names, total_steps, current_combination, progress_bar, config=config,
    )  # Evaluate stacking classifier, export model artifacts, generate metric plots, and collect result entry

    return individual_results, stacking_result_entry, current_combination  # Return per-model results and updated combination counter to the caller


def evaluate_on_dataset(
    file,
    df,
    feature_names,
    ga_selected_features,
    pca_n_components,
    rfe_selected_features,
    base_models,
    data_source_label="Original",
    hyperparams_map=None,
    experiment_id=None,
    experiment_mode="original_only",
    augmentation_ratio=None,
    execution_mode_str="binary",
    attack_types_combined=None,
    df_augmented_for_training=None,
    config=None,
):
    """
    Evaluate classifiers on a single dataset with optional training-only augmentation.

    :param file: Path to the dataset file
    :param df: DataFrame with the original dataset (used for test set)
    :param feature_names: List of feature column names
    :param ga_selected_features: GA selected features
    :param pca_n_components: Number of PCA components
    :param rfe_selected_features: RFE selected features
    :param base_models: Dictionary of base models to evaluate
    :param data_source_label: Label for data source ("Original", "Original+Augmented@50%", etc.)
    :param hyperparams_map: Dictionary mapping model names to hyperparameter dicts
    :param experiment_id: Unique experiment identifier for traceability
    :param experiment_mode: Experiment mode string ('original_only' or 'original_plus_augmented')
    :param augmentation_ratio: Augmentation ratio float (e.g., 0.50) or None for original-only
    :param execution_mode_str: Execution mode string ('binary' or 'multi-class')
    :param attack_types_combined: List of attack types for multi-class or None for binary
    :param df_augmented_for_training: Optional augmented DataFrame to merge into training set only (test set remains original-only)
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary mapping (feature_set, model_name) to results
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        ga_selected_features, rfe_selected_features = sanitize_and_verify_feature_selections(
            ga_selected_features, rfe_selected_features, feature_names, config=config
        )  # Sanitize and verify GA/RFE feature selections against available features

        print_dataset_evaluation_header(data_source_label)  # Print formatted evaluation header for the current data source

        data_splits = prepare_evaluation_data_splits(df, df_augmented_for_training, config=config)  # Prepare training/test data splits with optional augmentation

        if data_splits is None:  # If data preparation failed (single-class target)
            return {}  # Return empty dictionary

        X_train_scaled, X_test_scaled, y_train, y_test, scaler = data_splits  # Unpack the data splits tuple

        stacking_model = build_evaluation_stacking_model(base_models, config=config)  # Build stacking classifier from base models

        feature_sets = assemble_feature_sets(
            X_train_scaled, X_test_scaled, feature_names, ga_selected_features, pca_n_components, rfe_selected_features, file
        )  # Assemble feature sets dictionary for evaluation

        individual_models, total_steps, progress_bar, all_results, current_combination = initialize_evaluation_run_state(
            base_models, feature_sets, data_source_label
        )  # Build individual model map, compute total steps, create progress bar, and initialize result containers

        for idx, (name, (X_train_subset, X_test_subset, subset_feature_names_list)) in enumerate(feature_sets.items(), start=1):
            if X_train_subset.shape[1] == 0:  # Verify if the subset is empty
                print(
                    f"{BackgroundColors.YELLOW}Warning: Skipping {name}. No features selected.{Style.RESET_ALL}"
                )  # Output warning
                progress_bar.update(len(individual_models) + 1)  # Skip all steps for this feature set
                continue  # Skip to the next set

            individual_results, stacking_result_entry, current_combination = evaluate_single_feature_set(
                idx, name, X_train_subset, X_test_subset, subset_feature_names_list,
                individual_models, stacking_model, y_train, y_test,
                file, execution_mode_str, attack_types_combined,
                data_source_label, experiment_id, experiment_mode, augmentation_ratio,
                hyperparams_map, scaler, total_steps, current_combination, progress_bar, config=config,
            )  # Evaluate all individual classifiers and stacking model on this non-empty feature subset

            all_results.update(individual_results)  # Merge this feature set's results into the global results dict
            all_results[(name, "StackingClassifier")] = stacking_result_entry  # Store stacking result with key

        progress_bar.close()  # Close progress bar
        return all_results  # Return dictionary of results
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def determine_files_to_process(csv_file, input_path, config=None):
    """
    Determines which files to process based on CLI override or directory scan.

    :param csv_file: Optional CSV file path from CLI argument
    :param input_path: Directory path to search for CSV files
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: List of file paths to process
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Determining files to process from path: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        if csv_file:  # If a specific CSV file is provided via CLI
            try:  # Attempt to validate CSV file path
                abs_csv = os.path.abspath(csv_file)  # Get absolute path of CSV file
                abs_input = os.path.abspath(input_path)  # Get absolute path of input directory
                if abs_csv.startswith(abs_input):  # If CSV file belongs to this input path
                    return [csv_file]  # Return list with single CSV file
                else:  # CSV override does not belong to this path
                    return []  # Return empty list to skip this path
            except Exception:  # If validation fails
                return []  # Return empty list on error
        else:  # No CLI override, scan directory for dataset files
            dataset_format = config.get("stacking", {}).get("dataset_file_format", "csv")  # Read configured dataset file format
            dataset_extension = resolve_format_extension(dataset_format)  # Resolve format string to file extension
            return get_files_to_process(input_path, file_extension=dataset_extension, config=config)  # Get list of dataset files to process
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def combine_dataset_if_needed(files_to_process, config=None):
    """
    Combines multiple dataset files into one if PROCESS_ENTIRE_DATASET is enabled.

    :param files_to_process: List of file paths to process
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (combined_df, combined_file_for_features, updated_files_list) or (None, None, files_to_process)
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Checking if dataset combination is needed...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message
        
        process_entire_dataset = config.get("execution", {}).get("process_entire_dataset", False)  # Get process entire dataset flag from config

        if process_entire_dataset and len(files_to_process) > 1:  # If combining is enabled and multiple files exist
            verbose_output(
                f"{BackgroundColors.GREEN}Attempting to combine {BackgroundColors.CYAN}{len(files_to_process)}{BackgroundColors.GREEN} dataset files...{Style.RESET_ALL}",
                config=config
            )  # Output the verbose message
            result = combine_dataset_files(files_to_process, config=config)  # Attempt to combine all files
            if result is not None:  # If combination was successful
                combined_df, combined_target_col = result  # Unpack the combined dataframe and target column
                combined_file_for_features = files_to_process[0]  # Use first file for feature selection metadata
                files_to_process = ["combined"]  # Replace file list with single "combined" entry
                return (combined_df, combined_file_for_features, files_to_process)  # Return combined data and updated file list
            else:  # If combination failed
                print(
                    f"{BackgroundColors.YELLOW}Warning: Could not combine dataset files. Processing individually.{Style.RESET_ALL}"
                )  # Output warning message

        return (None, None, files_to_process)  # Return original file list unchanged
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def load_and_preprocess_dataset(file, combined_df, config=None):
    """
    Loads and preprocesses a dataset file or uses combined dataframe.

    :param file: File path to load or "combined" keyword
    :param combined_df: Pre-combined dataframe (used if file == "combined")
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (df_cleaned, feature_names) or (None, None) if loading/preprocessing fails
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Loading and preprocessing dataset: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        if file == "combined":  # If using combined dataset
            df_original = combined_df  # Use the pre-combined dataframe
        else:  # Otherwise load from file
            df_original = load_dataset(file, config=config)  # Load the original dataset

        if df_original is None:  # If the dataset failed to load
            verbose_output(
                f"{BackgroundColors.RED}Failed to load dataset from: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}",
                config=config
            )  # Output the failure message
            return (None, None)  # Return None tuple
        
        remove_zero_variance = config.get("dataset", {}).get("remove_zero_variance", True)  # Get remove zero variance flag from config

        df_cleaned = preprocess_dataframe(df_original, remove_zero_variance=remove_zero_variance, config=config)  # Preprocess the DataFrame

        if df_cleaned is None or df_cleaned.empty:  # If the DataFrame is None or empty after preprocessing
            print(
                f"{BackgroundColors.RED}Dataset {BackgroundColors.CYAN}{file}{BackgroundColors.RED} empty after preprocessing. Skipping.{Style.RESET_ALL}"
            )  # Output error message
            return (None, None)  # Return None tuple

        feature_names = df_cleaned.select_dtypes(include=np.number).iloc[:, :-1].columns.tolist()  # Get numeric feature names excluding target

        return (df_cleaned, feature_names)  # Return cleaned dataframe and feature names
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def prepare_models_with_hyperparameters(file_path, config=None):
    """
    Prepares base models and applies hyperparameter optimization results if available.

    :param file_path: Path to the dataset file for loading hyperparameters
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (base_models, hp_params_map)
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Preparing models with hyperparameters for: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        base_models = get_models(config=config)  # Get the base models with default parameters

        hp_params_map = {}  # Initialize empty hyperparameters mapping
        hp_results_raw = extract_hyperparameter_optimization_results(file_path, config=config)  # Extract hyperparameter optimization results

        if hp_results_raw:  # If results were found, extract the params mapping and apply
            hp_params_map = {
                k: (v.get("best_params") if isinstance(v, dict) else v) for k, v in hp_results_raw.items()
            }  # Extract only the best_params mapping
            base_models = apply_hyperparameters_to_models(hp_params_map, base_models, config=config)  # Apply hyperparameters to base models
            verbose_output(
                f"{BackgroundColors.GREEN}Applied hyperparameters from optimization results{Style.RESET_ALL}",
                config=config
            )  # Output the verbose message

        return (base_models, hp_params_map)  # Return models and hyperparameters mapping
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def locate_and_verify_artifacts(file_path, config=None):
    """
    Locate and verify artifacts for feature selection, hyperparameters and data augmentation.

    :param file_path: Path to the dataset CSV file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dict with keys: 'ga', 'pca', 'rfe', 'hyperparams', 'augmented_file'
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Verifying artifacts for: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}",
            config=config,
        )  # Inform which file we're verifying

        artifacts = {}  # Prepare return mapping

        ga_features = None  # Placeholder for GA features
        try:
            ga_features = extract_genetic_algorithm_features(file_path, config=config)  # Try extracting GA features
        except Exception as e:  # Guard extraction to avoid raising
            print(f"{BackgroundColors.YELLOW}Warning: GA extraction failed for {file_path}: {e}{Style.RESET_ALL}")  # Warn on failure
            ga_features = None  # Normalize to None on failure
        artifacts["ga"] = ga_features  # Store GA features (or None)

        pca_n = None  # Placeholder for PCA components
        try:
            pca_n = extract_principal_component_analysis_features(file_path, config=config)  # Try extracting PCA n_components
        except Exception as e:  # Guard extraction
            print(f"{BackgroundColors.YELLOW}Warning: PCA extraction failed for {file_path}: {e}{Style.RESET_ALL}")  # Warn on failure
            pca_n = None  # Normalize to None
        artifacts["pca"] = pca_n  # Store PCA components (or None)

        rfe_sel = None  # Placeholder for RFE selected features
        try:
            rfe_sel = extract_recursive_feature_elimination_features(file_path, config=config)  # Try extracting RFE features
        except Exception as e:  # Guard extraction
            print(f"{BackgroundColors.YELLOW}Warning: RFE extraction failed for {file_path}: {e}{Style.RESET_ALL}")  # Warn on failure
            rfe_sel = None  # Normalize to None
        artifacts["rfe"] = rfe_sel  # Store RFE selection (or None)

        hyperparams = None  # Placeholder for hyperparameters mapping
        try:
            hyperparams = extract_hyperparameter_optimization_results(file_path, config=config)  # Try locating HP CSV and parsing
        except Exception as e:  # Guard extraction
            print(f"{BackgroundColors.YELLOW}Warning: Hyperparameter extraction failed for {file_path}: {e}{Style.RESET_ALL}")  # Warn on failure
            hyperparams = None  # Normalize to None
        artifacts["hyperparams"] = hyperparams  # Store HP mapping (or None)

        augmented = None  # Placeholder for augmented CSV path
        try:
            augmented = find_data_augmentation_file(file_path, config=config)  # Try locating augmented CSV using config-aware finder
            if augmented:  # If found
                # Also collect PNG image path with same stem if present
                aug_png = Path(augmented).with_suffix(".png")  # Build PNG path by changing suffix
                if aug_png.exists():  # If PNG exists
                    artifacts["augmented_image"] = str(aug_png)  # Store image path
        except Exception as e:  # Guard finder
            print(f"{BackgroundColors.YELLOW}Warning: Data augmentation lookup failed for {file_path}: {e}{Style.RESET_ALL}")  # Warn on failure
            augmented = None  # Normalize to None
        artifacts["augmented_file"] = augmented  # Store augmented CSV path (or None)

        return artifacts  # Return mapping of discovered artifacts
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_metrics_from_result(result):
    """
    Extracts metrics from a result dictionary into a list.

    :param result: Result dictionary containing metric keys
    :return: List of [accuracy, precision, recall, f1_score, fpr, fnr, elapsed_time_s]
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Extracting metrics from result dictionary...{Style.RESET_ALL}"
        )  # Output the verbose message

        return [
            result.get("accuracy", 0),  # Get accuracy or default to 0
            result.get("precision", 0),  # Get precision or default to 0
            result.get("recall", 0),  # Get recall or default to 0
            result.get("f1_score", 0),  # Get F1 score or default to 0
            result.get("fpr", 0),  # Get false positive rate or default to 0
            result.get("fnr", 0),  # Get false negative rate or default to 0
            result.get("elapsed_time_s", 0),  # Get elapsed time or default to 0
        ]  # Return list of metric values
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def calculate_all_improvements(orig_metrics, merged_metrics):
    """
    Calculates improvement percentages for all metrics comparing original vs merged data.

    :param orig_metrics: List of original metrics [accuracy, precision, recall, f1, fpr, fnr, time]
    :param merged_metrics: List of merged metrics [accuracy, precision, recall, f1, fpr, fnr, time]
    :return: Dictionary of improvement percentages for each metric
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Calculating metric improvements...{Style.RESET_ALL}"
        )  # Output the verbose message

        return {
            "accuracy": calculate_metric_improvement(orig_metrics[0], merged_metrics[0]),  # Calculate accuracy improvement
            "precision": calculate_metric_improvement(orig_metrics[1], merged_metrics[1]),  # Calculate precision improvement
            "recall": calculate_metric_improvement(orig_metrics[2], merged_metrics[2]),  # Calculate recall improvement
            "f1_score": calculate_metric_improvement(orig_metrics[3], merged_metrics[3]),  # Calculate F1 score improvement
            "fpr": calculate_metric_improvement(orig_metrics[4], merged_metrics[4]),  # Calculate FPR change (lower is better)
            "fnr": calculate_metric_improvement(orig_metrics[5], merged_metrics[5]),  # Calculate FNR change (lower is better)
            "training_time": calculate_metric_improvement(orig_metrics[6], merged_metrics[6]),  # Calculate time change (lower is better)
        }  # Return dictionary of improvements
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def print_model_comparison(feature_set, model_name, orig_metrics, aug_metrics, merged_metrics, improvements):
    """
    Prints detailed comparison of metrics for a single model across data sources.

    :param feature_set: Name of the feature set used
    :param model_name: Name of the model
    :param orig_metrics: List of original data metrics
    :param aug_metrics: List of augmented data metrics
    :param merged_metrics: List of merged data metrics
    :param improvements: Dictionary of improvement percentages
    :return: None
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Printing comparison for model: {BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN}, feature set: {BackgroundColors.CYAN}{feature_set}{Style.RESET_ALL}"
        )  # Output the verbose message

        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Feature Set: {BackgroundColors.CYAN}{feature_set}{BackgroundColors.GREEN} | Model: {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}"
        )  # Print header with feature set and model name

        print(f"  {BackgroundColors.YELLOW}Accuracy:{Style.RESET_ALL}")  # Print accuracy label
        print(
            f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {truncate_value(orig_metrics[0])} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {truncate_value(aug_metrics[0])} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {truncate_value(merged_metrics[0])} | {BackgroundColors.CYAN}Improvement: {improvements['accuracy']:+.2f}%{Style.RESET_ALL}"
        )  # Print accuracy comparison

        print(f"  {BackgroundColors.YELLOW}Precision:{Style.RESET_ALL}")  # Print precision label
        print(
            f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {truncate_value(orig_metrics[1])} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {truncate_value(aug_metrics[1])} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {truncate_value(merged_metrics[1])} | {BackgroundColors.CYAN}Improvement: {improvements['precision']:+.2f}%{Style.RESET_ALL}"
        )  # Print precision comparison

        print(f"  {BackgroundColors.YELLOW}Recall:{Style.RESET_ALL}")  # Print recall label
        print(
            f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {truncate_value(orig_metrics[2])} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {truncate_value(aug_metrics[2])} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {truncate_value(merged_metrics[2])} | {BackgroundColors.CYAN}Improvement: {improvements['recall']:+.2f}%{Style.RESET_ALL}"
        )  # Print recall comparison

        print(f"  {BackgroundColors.YELLOW}F1-Score:{Style.RESET_ALL}")  # Print F1 score label
        print(
            f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {truncate_value(orig_metrics[3])} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {truncate_value(aug_metrics[3])} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {truncate_value(merged_metrics[3])} | {BackgroundColors.CYAN}Improvement: {improvements['f1_score']:+.2f}%{Style.RESET_ALL}"
        )  # Print F1 score comparison

        print(f"  {BackgroundColors.YELLOW}FPR (lower is better):{Style.RESET_ALL}")  # Print FPR label
        print(
            f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {truncate_value(orig_metrics[4])} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {truncate_value(aug_metrics[4])} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {truncate_value(merged_metrics[4])} | {BackgroundColors.CYAN}Change: {improvements['fpr']:+.2f}%{Style.RESET_ALL}"
        )  # Print FPR comparison

        print(f"  {BackgroundColors.YELLOW}FNR (lower is better):{Style.RESET_ALL}")  # Print FNR label
        print(
            f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {truncate_value(orig_metrics[5])} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {truncate_value(aug_metrics[5])} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {truncate_value(merged_metrics[5])} | {BackgroundColors.CYAN}Change: {improvements['fnr']:+.2f}%{Style.RESET_ALL}"
        )  # Print FNR comparison

        print(f"  {BackgroundColors.YELLOW}Training Time (seconds, lower is better):{Style.RESET_ALL}")  # Print training time label
        print(
            f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {orig_metrics[6]:.2f}s | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {aug_metrics[6]:.2f}s | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {merged_metrics[6]:.2f}s | {BackgroundColors.CYAN}Change: {improvements['training_time']:+.2f}%{Style.RESET_ALL}\n"
        )  # Print training time comparison
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_comparison_result_entry(orig_result, feature_set, classifier_type, model_name, data_source, metrics, improvements, n_features_override=None, n_samples_train_override=None, n_samples_test_override=None, experiment_id=None, experiment_mode="original_only", augmentation_ratio=None):
    """
    Builds a single comparison result entry for CSV export.

    :param orig_result: Original result dictionary for base metadata
    :param feature_set: Name of the feature set
    :param classifier_type: Type of classifier (e.g., 'Individual' or 'Stacking')
    :param model_name: Name of the model
    :param data_source: Data source label (e.g., 'Original', 'Original+Augmented@50%')
    :param metrics: List of metrics [accuracy, precision, recall, f1, fpr, fnr, time]
    :param improvements: Dictionary of improvement percentages
    :param n_features_override: Override for n_features (optional)
    :param n_samples_train_override: Override for n_samples_train (optional)
    :param n_samples_test_override: Override for n_samples_test (optional)
    :param experiment_id: Unique experiment identifier for traceability
    :param experiment_mode: Experiment mode string ('original_only' or 'original_plus_augmented')
    :param augmentation_ratio: Augmentation ratio float or None
    :return: Dictionary containing comparison result entry
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Building comparison result entry for: {BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN}, data source: {BackgroundColors.CYAN}{data_source}{Style.RESET_ALL}"
        )  # Output the verbose message

        return {
            "dataset": orig_result["dataset"],  # Dataset name from original result
            "feature_set": feature_set,  # Feature set name
            "classifier_type": classifier_type,  # Classifier type
            "model_name": model_name,  # Model name
            "data_source": data_source,  # Data source label
            "experiment_id": experiment_id,  # Unique experiment identifier for traceability
            "experiment_mode": experiment_mode,  # Experiment mode (original_only or original_plus_augmented)
            "augmentation_ratio": augmentation_ratio,  # Augmentation ratio float or None
            "n_features": n_features_override if n_features_override is not None else orig_result["n_features"],  # Number of features
            "n_samples_train": n_samples_train_override if n_samples_train_override is not None else orig_result["n_samples_train"],  # Training samples count
            "n_samples_test": n_samples_test_override if n_samples_test_override is not None else orig_result["n_samples_test"],  # Test samples count
            "accuracy": metrics[0],  # Accuracy metric
            "precision": metrics[1],  # Precision metric
            "recall": metrics[2],  # Recall metric
            "f1_score": metrics[3],  # F1 score metric
            "fpr": metrics[4],  # False positive rate
            "fnr": metrics[5],  # False negative rate
            "training_time": metrics[6],  # Training time in seconds
            "accuracy_improvement": improvements.get("accuracy", 0.0),  # Accuracy improvement percentage
            "precision_improvement": improvements.get("precision", 0.0),  # Precision improvement percentage
            "recall_improvement": improvements.get("recall", 0.0),  # Recall improvement percentage
            "f1_score_improvement": improvements.get("f1_score", 0.0),  # F1 score improvement percentage
            "fpr_improvement": improvements.get("fpr", 0.0),  # FPR improvement percentage
            "fnr_improvement": improvements.get("fnr", 0.0),  # FNR improvement percentage
            "training_time_improvement": improvements.get("training_time", 0.0),  # Training time improvement percentage
            "features_list": orig_result["features_list"],  # List of feature names used
        }  # Return comparison result entry dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def generate_ratio_comparison_report(results_original, all_ratio_results):
    """
    Generates and prints comparison report for ratio-based data augmentation evaluation.
    Compares the original baseline against each augmentation ratio experiment.

    :param results_original: Dictionary of results from original data evaluation
    :param all_ratio_results: Dictionary mapping ratio (float) to results dictionary
    :return: List of comparison result entries for CSV export
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Generating ratio-based data augmentation comparison report...{Style.RESET_ALL}"
        )  # Output the verbose message

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*100}{Style.RESET_ALL}"
        )  # Print separator line for visual clarity
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}DATA AUGMENTATION RATIO-BASED COMPARISON REPORT{Style.RESET_ALL}"
        )  # Print report header title
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*100}{Style.RESET_ALL}\n"
        )  # Print closing separator line

        comparison_results = []  # Initialize list for comparison result entries
        no_improvements = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "fpr": 0.0, "fnr": 0.0, "training_time": 0.0}  # Zero improvements dict for original baseline entries

        for key in results_original.keys():  # Iterate through each feature_set/model combination from original results
            orig_result = results_original[key]  # Get the original baseline result entry
            feature_set = orig_result["feature_set"]  # Extract feature set name from result
            model_name = orig_result["model_name"]  # Extract model name from result
            classifier_type = orig_result["classifier_type"]  # Extract classifier type from result
            orig_metrics = extract_metrics_from_result(orig_result)  # Extract metrics list from original result
            orig_experiment_id = orig_result.get("experiment_id", None)  # Get experiment ID from original result

            comparison_results.append(
                build_comparison_result_entry(
                    orig_result, feature_set, classifier_type, model_name, "Original",
                    orig_metrics, no_improvements,
                    experiment_id=orig_experiment_id, experiment_mode="original_only", augmentation_ratio=None,
                )
            )  # Add original baseline entry to comparison results

            print(
                f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Feature Set: {BackgroundColors.CYAN}{feature_set}{BackgroundColors.GREEN} | Model: {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}"
            )  # Print header with feature set and model name
            print(
                f"  {BackgroundColors.GREEN}Original baseline - Acc: {BackgroundColors.CYAN}{truncate_value(orig_metrics[0])}{BackgroundColors.GREEN}, F1: {BackgroundColors.CYAN}{truncate_value(orig_metrics[3])}{Style.RESET_ALL}"
            )  # Print original baseline metrics summary

            for ratio in sorted(all_ratio_results.keys()):  # Iterate over each ratio in sorted order
                ratio_results = all_ratio_results[ratio]  # Get results dict for this ratio
                ratio_result = ratio_results.get(key)  # Get the matching result for this feature_set/model key

                if ratio_result is None:  # If no matching result exists for this ratio
                    continue  # Skip this ratio for this model/feature_set combination

                ratio_metrics = extract_metrics_from_result(ratio_result)  # Extract metrics list from ratio result
                improvements = calculate_all_improvements(orig_metrics, ratio_metrics)  # Calculate improvements vs original
                ratio_pct = int(ratio * 100)  # Convert float ratio to integer percentage for display
                ratio_experiment_id = ratio_result.get("experiment_id", None)  # Get experiment ID from ratio result

                comparison_results.append(
                    build_comparison_result_entry(
                        orig_result, feature_set, classifier_type, model_name,
                        f"Original+Augmented@{ratio_pct}%", ratio_metrics, improvements,
                        n_features_override=ratio_result.get("n_features"),
                        n_samples_train_override=ratio_result.get("n_samples_train"),
                        n_samples_test_override=ratio_result.get("n_samples_test"),
                        experiment_id=ratio_experiment_id, experiment_mode="original_plus_augmented",
                        augmentation_ratio=ratio,
                    )
                )  # Add ratio experiment entry with improvements to comparison results

                f1_improvement = improvements.get("f1_score", 0.0)  # Extract F1 improvement for display
                improvement_color = BackgroundColors.GREEN if f1_improvement >= 0 else BackgroundColors.RED  # Choose color based on improvement direction
                print(
                    f"  {BackgroundColors.YELLOW}@{ratio_pct}%:{Style.RESET_ALL} Acc: {BackgroundColors.CYAN}{truncate_value(ratio_metrics[0])}{Style.RESET_ALL}, F1: {BackgroundColors.CYAN}{truncate_value(ratio_metrics[3])}{Style.RESET_ALL}, F1 change: {improvement_color}{f1_improvement:+.2f}%{Style.RESET_ALL}"
                )  # Print ratio result metrics with F1 improvement indicator

        return comparison_results  # Return list of all comparison result entries for CSV export
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def load_and_validate_augmented_data(file, df_original_cleaned, config=None):
    """
    Locates, loads, preprocesses, and validates augmented data for the given file.

    :param file: Original file path used to locate the augmented counterpart
    :param df_original_cleaned: Cleaned original dataframe for validation reference
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Cleaned augmented dataframe, or None if loading or validation fails
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        augmented_file = find_data_augmentation_file(file, config=config)  # Look for augmented data file using wgangp.py naming convention

        if augmented_file is None:  # If no augmented file found at expected path
            print(
                f"\n{BackgroundColors.YELLOW}No augmented data found for this file. Skipping augmentation comparison.{Style.RESET_ALL}"
            )  # Print warning message about missing augmented file
            return None  # Signal caller that no augmented data is available

        df_augmented = load_augmented_dataset(augmented_file, config=config)  # Load the augmented dataset using configured augmentation format

        if df_augmented is None:  # If augmented dataset failed to load from disk
            print(
                f"{BackgroundColors.YELLOW}Warning: Failed to load augmented dataset from {BackgroundColors.CYAN}{augmented_file}{BackgroundColors.YELLOW}. Skipping.{Style.RESET_ALL}"
            )  # Print warning message about load failure
            return None  # Signal caller that loading failed

        df_augmented_cleaned = preprocess_dataframe(df_augmented)  # Preprocess the augmented dataframe with same pipeline as original

        if not validate_augmented_dataframe(df_original_cleaned, df_augmented_cleaned, file):  # Validate augmented data is compatible with original
            return None  # Signal caller that validation failed

        return df_augmented_cleaned  # Return the cleaned and validated augmented dataframe
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_single_ratio_experiment(file, df_original_cleaned, df_augmented_cleaned, feature_names, ga_selected_features, pca_n_components, rfe_selected_features, base_models, hp_params_map, ratio, ratio_idx, total_ratios, config=None):
    """
    Executes a single ratio-based augmentation experiment by sampling, visualizing, and evaluating.

    :param file: Original file path
    :param df_original_cleaned: Cleaned original dataframe
    :param df_augmented_cleaned: Cleaned augmented dataframe to sample from
    :param feature_names: List of feature names
    :param ga_selected_features: Features selected by genetic algorithm
    :param pca_n_components: Number of PCA components
    :param rfe_selected_features: Features selected by RFE
    :param base_models: Dictionary of base models
    :param hp_params_map: Hyperparameters mapping
    :param ratio: Current augmentation ratio (float between 0 and 1)
    :param ratio_idx: Current ratio index (1-based) for progress display
    :param total_ratios: Total number of ratios being evaluated
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Results list from evaluate_on_dataset, or None if sampling fails
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        ratio_pct = int(ratio * 100)  # Convert float ratio to integer percentage for display
        experiment_id = generate_experiment_id(file, "original_plus_augmented", ratio)  # Generate unique experiment ID for this ratio

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}[{ratio_idx}/{total_ratios}] Evaluating Original + Augmented@{ratio_pct}%{Style.RESET_ALL}"
        )  # Print progress indicator for current ratio experiment
        send_telegram_message(TELEGRAM_BOT, [f"[{ratio_idx}/{total_ratios}] Augmentation experiment: Original + Augmented@{ratio_pct}% | file: {os.path.basename(file)}"])  # Notify Telegram about per-ratio augmentation experiment progress

        df_sampled = sample_augmented_by_ratio(df_augmented_cleaned, df_original_cleaned, ratio)  # Sample augmented rows at the current ratio

        if df_sampled is None or df_sampled.empty:  # If sampling returned no valid data
            print(
                f"{BackgroundColors.YELLOW}Warning: Could not sample augmented data at ratio {ratio}. Skipping this ratio.{Style.RESET_ALL}"
            )  # Print warning about sampling failure
            return None  # Signal caller to skip this ratio

        data_source_label = f"Original+Augmented@{ratio_pct}%"  # Build descriptive data source label for CSV traceability

        print(
            f"{BackgroundColors.GREEN}Sampled augmented dataset: {BackgroundColors.CYAN}{len(df_sampled)} augmented samples at {ratio_pct}% ratio (will be merged into training set only){Style.RESET_ALL}"
        )  # Print sampled dataset size for transparency

        generate_augmentation_tsne_visualization(
            file, df_original_cleaned, df_sampled, ratio, "original_plus_augmented"
        )  # Generate t-SNE visualization for this augmentation ratio

        results_ratio = evaluate_on_dataset(
            file, df_original_cleaned, feature_names, ga_selected_features, pca_n_components,
            rfe_selected_features, base_models, data_source_label=data_source_label,
            hyperparams_map=hp_params_map, experiment_id=experiment_id,
            experiment_mode="original_plus_augmented", augmentation_ratio=ratio,
            execution_mode_str="binary", attack_types_combined=None,
            df_augmented_for_training=df_sampled
        )  # Evaluate all classifiers with augmented data in training only (test remains original-only)

        send_telegram_message(
            TELEGRAM_BOT, f"Completed augmentation ratio {ratio_pct}% for {os.path.basename(file)}"
        )  # Send Telegram notification for ratio completion

        del df_sampled  # Release sampled augmented data to free memory after evaluation
        gc.collect()  # Force garbage collection to reclaim memory from sampled data

        return results_ratio  # Return the evaluation results for this ratio
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def process_augmented_data_evaluation(file, df_original_cleaned, feature_names, ga_selected_features, pca_n_components, rfe_selected_features, base_models, hp_params_map, results_original, config=None):
    """
    Handles complete augmented data evaluation workflow with ratio-based experiments.
    For each ratio in config.get("stacking", {}).get("augmentation_ratios", [0.10, 0.25, 0.50, 0.75, 1.00]), samples augmented data proportionally,
    merges with original, evaluates classifiers, and compares against original baseline.

    :param file: Original file path
    :param df_original_cleaned: Cleaned original dataframe
    :param feature_names: List of feature names
    :param ga_selected_features: Features selected by genetic algorithm
    :param pca_n_components: Number of PCA components
    :param rfe_selected_features: Features selected by RFE
    :param base_models: Dictionary of base models
    :param hp_params_map: Hyperparameters mapping
    :param results_original: Results from original data evaluation
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Processing augmented data evaluation for: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}"
        )  # Output the verbose message

        df_augmented_cleaned = load_and_validate_augmented_data(file, df_original_cleaned, config=config)  # Load, preprocess, and validate augmented data

        if df_augmented_cleaned is None:  # If augmented data could not be loaded or validated
            return  # Exit function early when no valid augmented data is available

        augmentation_ratios = config.get("stacking", {}).get("augmentation_ratios", [0.10, 0.25, 0.50, 0.75, 1.00])  # Retrieve the list of ratios to evaluate

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*100}{Style.RESET_ALL}"
        )  # Print separator line for visual clarity
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}RATIO-BASED DATA AUGMENTATION EXPERIMENTS{Style.RESET_ALL}"
        )  # Print header for the ratio-based experiments section
        print(
            f"{BackgroundColors.GREEN}Ratios to evaluate: {BackgroundColors.CYAN}{[f'{int(r*100)}%' for r in augmentation_ratios]}{Style.RESET_ALL}"
        )  # Print the list of ratios that will be evaluated
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*100}{Style.RESET_ALL}\n"
        )  # Print closing separator line
        send_telegram_message(TELEGRAM_BOT, [f"[BINARY] Starting ratio-based augmentation experiments | file: {os.path.basename(file)} | ratios: {[f'{int(r * 100)}%' for r in augmentation_ratios]}"])  # Notify Telegram about augmentation ratio experiments start

        all_ratio_results = {}  # Dictionary to store results for each ratio: {ratio: results_dict}

        for ratio_idx, ratio in enumerate(augmentation_ratios, start=1):  # Iterate over each augmentation ratio
            results_ratio = run_single_ratio_experiment(
                file, df_original_cleaned, df_augmented_cleaned, feature_names,
                ga_selected_features, pca_n_components, rfe_selected_features,
                base_models, hp_params_map, ratio, ratio_idx, len(augmentation_ratios), config
            )  # Execute the experiment for this specific ratio

            if results_ratio is not None:  # If the ratio experiment produced valid results
                all_ratio_results[ratio] = results_ratio  # Store the results for this ratio in the results dictionary

        del df_augmented_cleaned  # Release augmented data to free memory after all ratio experiments
        gc.collect()  # Force garbage collection to reclaim memory from augmented data

        if not all_ratio_results:  # If no ratio experiments produced valid results
            print(
                f"{BackgroundColors.YELLOW}Warning: No ratio experiments completed successfully. Skipping comparison report.{Style.RESET_ALL}"
            )  # Print warning about no completed experiments
            return  # Exit function early when no results are available

        comparison_results = generate_ratio_comparison_report(results_original, all_ratio_results)  # Generate the comparison report across all ratios

        save_augmentation_comparison_results(file, comparison_results)  # Save comparison results to CSV file

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Data augmentation ratio-based comparison complete!{Style.RESET_ALL}"
        )  # Print success message indicating all ratio experiments are done
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def save_multiclass_results_to_csv(reference_file, results_list, config=None):
    """
    Save multi-class evaluation results to the Feature_Analysis directory.

    :param reference_file: Reference file path for determining output directory
    :param results_list: List of result dictionaries to save
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Path object of the Feature_Analysis directory for reuse
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        multiclass_results_filename = config.get("stacking", {}).get("multiclass_results_filename", "Stacking_Classifiers_MultiClass_Results.csv")  # Get multi-class results filename from config
        reference_file_path = Path(reference_file)  # Create Path object from reference file
        feature_analysis_dir = reference_file_path.parent / "Feature_Analysis"  # Build Feature_Analysis directory path
        os.makedirs(feature_analysis_dir, exist_ok=True)  # Ensure directory exists on disk
        multiclass_results_path = feature_analysis_dir / multiclass_results_filename  # Build full multi-class results file path

        save_stacking_results(str(multiclass_results_path), results_list, config=config)  # Save multi-class results to CSV

        return feature_analysis_dir  # Return Feature_Analysis directory path for reuse
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def save_multiclass_augmentation_comparison(results_original, all_ratio_results, feature_analysis_dir, config=None):
    """
    Generates a ratio comparison report and saves it to a CSV file in the feature analysis directory.

    :param results_original: Results dictionary from the original-data-only evaluation
    :param all_ratio_results: Dictionary mapping augmentation ratio floats to their evaluation result dicts
    :param feature_analysis_dir: Path object pointing to the Feature_Analysis output directory
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        if not all_ratio_results:  # If no ratio experiments produced results
            print(
                f"{BackgroundColors.YELLOW}No augmentation ratio experiments completed successfully for multi-class.{Style.RESET_ALL}"
            )  # Print warning about no completed experiments
            return  # Exit early since there is nothing to compare

        comparison_results = generate_ratio_comparison_report(results_original, all_ratio_results)  # Generate comparison report across all evaluated augmentation ratios

        augmentation_comparison_filename = config.get("stacking", {}).get("augmentation_comparison_filename", "Data_Augmentation_Comparison_Results.csv")  # Get base comparison filename from config
        multiclass_comparison_filename = augmentation_comparison_filename.replace(".csv", "_MultiClass.csv")  # Build multi-class-specific comparison filename
        multiclass_comparison_path = feature_analysis_dir / multiclass_comparison_filename  # Construct full output path inside Feature_Analysis directory

        save_augmentation_comparison_results(str(multiclass_comparison_path), comparison_results, config=config)  # Save comparison results to the CSV file

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Multi-class data augmentation ratio-based comparison complete!{Style.RESET_ALL}"
        )  # Print success message indicating comparison report has been saved
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_multiclass_augmentation_ratio_experiment(reference_file, combined_multiclass_df, combined_augmented_df, feature_names, ga_selected_features, pca_n_components, rfe_selected_features, base_models, hp_params_map, attack_types_list, ratio, ratio_idx, total_steps, dataset_name, config=None):
    """
    Runs a single ratio-based augmentation experiment for multi-class evaluation and returns the results.

    :param reference_file: Reference file path used for t-SNE visualization and experiment ID generation
    :param combined_multiclass_df: Combined original multi-class DataFrame with all attack types
    :param combined_augmented_df: Combined augmented DataFrame used as the augmentation source
    :param feature_names: List of feature column names for the dataset
    :param ga_selected_features: List of features selected by the genetic algorithm
    :param pca_n_components: Number of PCA components or None if PCA is disabled
    :param rfe_selected_features: List of features selected by RFE or None if disabled
    :param base_models: Dictionary mapping model names to model objects
    :param hp_params_map: Dictionary mapping model names to hyperparameter dicts
    :param attack_types_list: List of unique attack type labels for multi-class classification
    :param ratio: Augmentation ratio float between 0 and 1
    :param ratio_idx: 1-based index of this ratio in the augmentation schedule
    :param total_steps: Total number of evaluation steps for progress display
    :param dataset_name: Name of the dataset for Telegram notification messages
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Results dictionary from evaluate_on_dataset, or None if sampling failed
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        ratio_pct = int(ratio * 100)  # Convert ratio float to integer percentage for display

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}[{ratio_idx + 1}/{total_steps}] Evaluating with {ratio_pct}% augmented data (Multi-Class){Style.RESET_ALL}"
        )  # Print experiment step progress for this ratio

        df_sampled = sample_augmented_by_ratio(combined_augmented_df, combined_multiclass_df, ratio)  # Sample augmented data proportionally to the requested ratio

        if df_sampled is None:  # If sampling failed for this ratio
            print(
                f"{BackgroundColors.YELLOW}Failed to sample augmented data at ratio {ratio_pct}%. Skipping.{Style.RESET_ALL}"
            )  # Print warning about sampling failure
            return None  # Signal caller to skip this ratio

        data_source_label = f"Original+Augmented@{ratio_pct}%_MultiClass"  # Build data source label for result traceability
        experiment_id = generate_experiment_id(reference_file, "multiclass_original_plus_augmented", ratio)  # Generate unique experiment ID for this run

        print(
            f"{BackgroundColors.GREEN}Sampled augmented dataset: {BackgroundColors.CYAN}{len(df_sampled)} augmented samples at {ratio_pct}% ratio (will be merged into training set only){Style.RESET_ALL}"
        )  # Print sampled dataset size for transparency

        generate_augmentation_tsne_visualization(
            reference_file, combined_multiclass_df, df_sampled, ratio, "original_plus_augmented"
        )  # Generate t-SNE visualization comparing original and augmented distributions

        send_telegram_message(
            TELEGRAM_BOT, f"Starting multi-class augmentation ratio {ratio_pct}% for {dataset_name}"
        )  # Send Telegram notification for ratio experiment start

        results_ratio = evaluate_on_dataset(
            reference_file, combined_multiclass_df, feature_names, ga_selected_features, pca_n_components,
            rfe_selected_features, base_models, data_source_label=data_source_label,
            hyperparams_map=hp_params_map, experiment_id=experiment_id,
            experiment_mode="original_plus_augmented", augmentation_ratio=ratio,
            execution_mode_str="multi-class", attack_types_combined=attack_types_list,
            df_augmented_for_training=df_sampled
        )  # Evaluate all classifiers with augmented training data using original-only test set

        send_telegram_message(
            TELEGRAM_BOT, f"Completed multi-class augmentation ratio {ratio_pct}% for {dataset_name}"
        )  # Send Telegram notification for ratio experiment completion

        return results_ratio  # Return evaluation results for this ratio
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def print_multiclass_augmentation_header(augmentation_ratios):
    """
    Prints the formatted header block announcing ratio-based data augmentation experiments for multi-class mode.

    :param augmentation_ratios: List of augmentation ratio floats to be evaluated
    :return: None
    """

    try:
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*100}{Style.RESET_ALL}"
        )  # Print top separator line for the augmentation experiments section
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}RATIO-BASED DATA AUGMENTATION EXPERIMENTS (Multi-Class){Style.RESET_ALL}"
        )  # Print the section title for ratio-based multi-class augmentation
        print(
            f"{BackgroundColors.GREEN}Ratios to evaluate: {BackgroundColors.CYAN}{[f'{int(r*100)}%' for r in augmentation_ratios]}{Style.RESET_ALL}"
        )  # Print the list of augmentation ratios that will be evaluated
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*100}{Style.RESET_ALL}\n"
        )  # Print bottom separator line for the augmentation experiments section
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def load_and_combine_augmented_multiclass_files(original_files_list, config=None):
    """
    Loads augmented files for multi-class mode and combines them into a single DataFrame.

    :param original_files_list: List of original file paths used to find corresponding augmented files
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Combined augmented DataFrame on success, or None if loading or combination fails
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        augmented_files_list = load_augmented_files_for_multiclass(original_files_list, config=config)  # Load the list of augmented file paths

        if not augmented_files_list:  # If no augmented files were found
            print(
                f"{BackgroundColors.YELLOW}No augmented files found for multi-class mode. Skipping augmentation testing.{Style.RESET_ALL}"
            )  # Print warning about missing augmented files
            return None  # Signal caller to exit early

        combined_augmented_df, augmented_attack_types, augmented_target_col = combine_files_for_multiclass(augmented_files_list, config=config)  # Combine all augmented files into a single DataFrame

        if combined_augmented_df is None:  # If augmented file combination failed
            print(
                f"{BackgroundColors.YELLOW}Failed to combine augmented files for multi-class. Skipping augmentation testing.{Style.RESET_ALL}"
            )  # Print warning about combination failure
            return None  # Signal caller to exit early

        return combined_augmented_df  # Return the combined augmented DataFrame for ratio experiments
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def process_multiclass_augmentation_testing(reference_file, original_files_list, combined_multiclass_df, feature_names, ga_selected_features, pca_n_components, rfe_selected_features, base_models, hp_params_map, attack_types_list, results_original, augmentation_ratios, total_steps, feature_analysis_dir, dataset_name, config=None):
    """
    Process multi-class augmented data evaluation with ratio-based experiments.

    :param reference_file: Reference file path for feature metadata
    :param original_files_list: List of original file paths
    :param combined_multiclass_df: Combined multi-class DataFrame
    :param feature_names: List of feature column names
    :param ga_selected_features: GA selected features
    :param pca_n_components: Number of PCA components
    :param rfe_selected_features: RFE selected features
    :param base_models: Dictionary of base models
    :param hp_params_map: Hyperparameters mapping
    :param attack_types_list: List of unique attack type labels
    :param results_original: Results from original data evaluation
    :param augmentation_ratios: List of augmentation ratios to evaluate
    :param total_steps: Total number of evaluation steps for progress display
    :param feature_analysis_dir: Path to Feature_Analysis directory
    :param dataset_name: Name of the dataset being processed
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Processing multi-class augmented data evaluation...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        generate_augmentation_tsne_visualization(
            reference_file, combined_multiclass_df, None, None, "original_only"
        )  # Generate t-SNE visualization for original multi-class data only

        combined_augmented_df = load_and_combine_augmented_multiclass_files(original_files_list, config=config)  # Load and combine augmented files into a single DataFrame

        if combined_augmented_df is None:  # If loading or combining augmented files failed
            return  # Exit function early as signaled by the loading function

        print_multiclass_augmentation_header(augmentation_ratios)  # Print the section header for ratio-based multi-class augmentation experiments
        send_telegram_message(TELEGRAM_BOT, [f"[MULTI-CLASS] Starting ratio-based augmentation experiments | Dataset: {dataset_name} | ratios: {[f'{int(r * 100)}%' for r in augmentation_ratios]}"])  # Notify Telegram about multiclass augmentation ratio experiments start

        all_ratio_results = {}  # Dictionary to store results for each ratio: {ratio: results_dict}

        for ratio_idx, ratio in enumerate(augmentation_ratios, start=1):  # Iterate over each augmentation ratio
            results_ratio = run_multiclass_augmentation_ratio_experiment(
                reference_file, combined_multiclass_df, combined_augmented_df, feature_names,
                ga_selected_features, pca_n_components, rfe_selected_features, base_models,
                hp_params_map, attack_types_list, ratio, ratio_idx, total_steps, dataset_name, config=config,
            )  # Run evaluation for this ratio and retrieve results or None if sampling failed
            if results_ratio is None:  # If this ratio experiment failed at the sampling stage
                continue  # Skip to the next ratio
            all_ratio_results[ratio] = results_ratio  # Store the results for this ratio

        del combined_augmented_df  # Release combined augmented dataframe to free memory after all ratio experiments
        gc.collect()  # Force garbage collection to reclaim memory from augmented data

        save_multiclass_augmentation_comparison(results_original, all_ratio_results, feature_analysis_dir, config=config)  # Generate comparison report and save results to CSV
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def process_multiclass_evaluation(original_files_list, combined_multiclass_df, attack_types_list, dataset_name, config=None):
    """
    Process evaluation for multi-class classification mode with optional data augmentation.
    
    :param original_files_list: List of original file paths used for multi-class combination
    :param combined_multiclass_df: Combined multi-class DataFrame with 'attack_type' column
    :param attack_types_list: List of unique attack type labels
    :param dataset_name: Name of the dataset being processed
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        verbose_output(
            f"{BackgroundColors.GREEN}Processing multi-class evaluation for dataset: {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message
    
        reference_file = original_files_list[0] if original_files_list else "multiclass_combined"  # Get reference file for feature metadata
        
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}"
        )  # Print separator line
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing multi-class dataset: {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}"
        )  # Print dataset header
        print(
            f"{BackgroundColors.GREEN}Attack types: {BackgroundColors.CYAN}{attack_types_list}{Style.RESET_ALL}"
        )  # Print attack types
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}\n"
        )  # Print closing separator
        send_telegram_message(TELEGRAM_BOT, [f"Starting multi-class evaluation | Dataset: {dataset_name} | {len(attack_types_list)} attack types: {', '.join(str(a) for a in attack_types_list)}"])  # Notify Telegram about multiclass evaluation start

        ga_selected_features, pca_n_components, rfe_selected_features = load_feature_selection_results(
            reference_file, config=config
        )  # Load feature selection results

        methods_cfg = config.get("stacking", {}).get("methods", {})  # Retrieve method toggles from config

        if not methods_cfg.get("feature_selection", True):  # Verify if feature selection is disabled via toggle
            ga_selected_features = None  # Suppress GA features when feature selection is disabled
            pca_n_components = None  # Suppress PCA components when feature selection is disabled
            rfe_selected_features = None  # Suppress RFE features when feature selection is disabled
        
        feature_names = [col for col in combined_multiclass_df.columns if col != 'attack_type']  # Get feature column names
        
        verbose_output(
            f"{BackgroundColors.GREEN}Multi-class dataset features: {BackgroundColors.CYAN}{len(feature_names)} features{Style.RESET_ALL}",
            config=config
        )  # Output feature count

        if methods_cfg.get("hyperparameter_optimization", True):  # Verify if hyperparameter optimization is enabled via toggle
            base_models, hp_params_map = prepare_models_with_hyperparameters(reference_file, config=config)  # Prepare base models with hyperparameters
        else:  # Hyperparameter optimization is disabled via toggle
            base_models = get_models(config=config)  # Instantiate base models without hyperparameter optimization
            hp_params_map = {}  # Initialize empty hyperparameters mapping when disabled
        
        original_experiment_id = generate_experiment_id(reference_file, "multiclass_original_only")  # Generate unique experiment ID
        
        test_data_augmentation = config.get("execution", {}).get("test_data_augmentation", False)  # Get test data augmentation flag from config

        if not methods_cfg.get("augmentation", True):  # Verify if augmentation is disabled via toggle
            test_data_augmentation = False  # Override test data augmentation when augmentation method is disabled

        augmentation_ratios = config.get("stacking", {}).get("augmentation_ratios", [0.10, 0.25, 0.50, 0.75, 1.00])  # Get augmentation ratios from config
        
        total_steps = 1 + (len(augmentation_ratios) if test_data_augmentation else 0)  # Calculate total evaluation steps
        
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}[1/{total_steps}] Evaluating on ORIGINAL MULTI-CLASS data{Style.RESET_ALL}"
        )  # Print progress message with total step count
        
        results_original = evaluate_on_dataset(
            reference_file, combined_multiclass_df, feature_names, ga_selected_features, pca_n_components,
            rfe_selected_features, base_models, data_source_label="Original_MultiClass", hyperparams_map=hp_params_map,
            experiment_id=original_experiment_id, experiment_mode="original_only", augmentation_ratio=None,
            execution_mode_str="multi-class", attack_types_combined=attack_types_list
        )  # Evaluate on original multi-class data with execution mode tracking
        
        original_results_list = list(results_original.values())  # Convert results dict to list
        feature_analysis_dir = save_multiclass_results_to_csv(reference_file, original_results_list, config=config)  # Save multi-class results and get Feature_Analysis directory
        
        enable_automl = methods_cfg.get("automl", True)  # Resolve AutoML toggle from stacking methods config
        if enable_automl:  # If AutoML pipeline is enabled
            print(f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}[DEBUG] AutoML pipeline is ENABLED. Running AutoML for multi-class dataset.{Style.RESET_ALL}")  # Log AutoML execution start
            send_telegram_message(TELEGRAM_BOT, [f"Running AutoML pipeline for multi-class dataset: {dataset_name}"])  # Notify Telegram about AutoML pipeline execution
            run_automl_pipeline(reference_file, combined_multiclass_df, feature_names, data_source_label="Original_MultiClass", config=config)  # Run AutoML pipeline for multi-class
        else:  # AutoML pipeline is disabled via method toggle
            print(f"{BackgroundColors.YELLOW}[DEBUG] AutoML pipeline is DISABLED (stacking.methods.automl=false). Skipping AutoML for multi-class. Enable via config or --enable-automl flag.{Style.RESET_ALL}")  # Log AutoML skip reason
        
        if test_data_augmentation:  # If data augmentation testing is enabled
            process_multiclass_augmentation_testing(
                reference_file, original_files_list, combined_multiclass_df, feature_names,
                ga_selected_features, pca_n_components, rfe_selected_features, base_models,
                hp_params_map, attack_types_list, results_original, augmentation_ratios,
                total_steps, feature_analysis_dir, dataset_name, config=config,
            )  # Process multi-class augmented data evaluation with ratio experiments
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def print_file_processing_header(file, config=None):
    """
    Prints formatted header for file processing section.

    :param file: File path being processed
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Printing file processing header for: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}"
        )  # Print separator line
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing file: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}"
        )  # Print file being processed
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}\n"
        )  # Print separator line
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def process_single_file_evaluation(file, combined_df, combined_file_for_features, config=None):
    """
    Processes evaluation for a single file including feature loading, model preparation, and evaluation.

    :param file: File path to process
    :param combined_df: Combined dataframe (used if file == "combined")
    :param combined_file_for_features: File to use for feature selection metadata
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Starting single file evaluation for: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        print_file_processing_header(file, config=config)  # Print formatted header

        file_for_features = combined_file_for_features if file == "combined" else file  # Determine which file to use for feature selection metadata
        ga_selected_features, pca_n_components, rfe_selected_features = load_feature_selection_results(
            file_for_features, config=config
        )  # Load feature selection results

        df_original_cleaned, feature_names = load_and_preprocess_dataset(file, combined_df, config=config)  # Load and preprocess the dataset

        if df_original_cleaned is None:  # If loading or preprocessing failed
            return  # Exit function early

        base_models, hp_params_map = prepare_models_with_hyperparameters(file, config=config)  # Prepare base models with hyperparameters

        original_experiment_id = generate_experiment_id(file, "original_only")  # Generate unique experiment ID for the original-only evaluation

        test_data_augmentation = config.get("execution", {}).get("test_data_augmentation", False)  # Get test data augmentation flag from config
        augmentation_ratios = config.get("stacking", {}).get("augmentation_ratios", [0.10, 0.25, 0.50, 0.75, 1.00])  # Get augmentation ratios from config
        
        if test_data_augmentation:  # If data augmentation testing is enabled
            generate_augmentation_tsne_visualization(
                file, df_original_cleaned, None, None, "original_only"
            )  # Generate t-SNE visualization for original data only

        total_steps = 1 + (len(augmentation_ratios) if test_data_augmentation else 0)  # Calculate total evaluation steps
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}[1/{total_steps}] Evaluating on ORIGINAL data{Style.RESET_ALL}"
        )  # Print progress message with total step count
        results_original = evaluate_on_dataset(
            file, df_original_cleaned, feature_names, ga_selected_features, pca_n_components,
            rfe_selected_features, base_models, data_source_label="Original", hyperparams_map=hp_params_map,
            experiment_id=original_experiment_id, experiment_mode="original_only", augmentation_ratio=None,
            execution_mode_str="binary", attack_types_combined=None
        )  # Evaluate on original data with experiment traceability metadata

        original_results_list = list(results_original.values())  # Convert results dict to list
        save_stacking_results(file, original_results_list, config=config)  # Save original results to CSV

        methods_cfg_local = config.get("stacking", {}).get("methods", {})  # Retrieve method toggles from config
        enable_automl = methods_cfg_local.get("automl", True)  # Resolve AutoML toggle from stacking methods config
        if enable_automl:  # If AutoML pipeline is enabled
            print(f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}[DEBUG] AutoML pipeline is ENABLED. Running AutoML for binary dataset: {os.path.basename(file)}{Style.RESET_ALL}")  # Log AutoML execution start
            run_automl_pipeline(file, df_original_cleaned, feature_names, config=config)  # Run AutoML pipeline
        else:  # AutoML pipeline is disabled via method toggle
            print(f"{BackgroundColors.YELLOW}[DEBUG] AutoML pipeline is DISABLED (stacking.methods.automl=false). Skipping AutoML for {os.path.basename(file)}. Enable via config or --enable-automl flag.{Style.RESET_ALL}")  # Log AutoML skip reason

        if test_data_augmentation:  # If data augmentation testing is enabled
            process_augmented_data_evaluation(
                file, df_original_cleaned, feature_names, ga_selected_features, pca_n_components,
                rfe_selected_features, base_models, hp_params_map, results_original, config=config
            )  # Process augmented data evaluation workflow
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def annotate_results_with_combination_flags(results_list, feature_selection_enabled, hyperparameters_enabled, data_augmentation_enabled=False):
    """
    Annotate result entries with combination flags for FS, HP, and DA.

    :param results_list: List of result dictionaries to annotate
    :param feature_selection_enabled: Whether feature selection was enabled
    :param hyperparameters_enabled: Whether hyperparameters were enabled
    :param data_augmentation_enabled: Whether data augmentation was enabled
    :return: None (modifies results_list in place)
    """

    try:
        for row in results_list:  # For each result row in the list
            row["feature_selection_enabled"] = feature_selection_enabled  # Mark feature selection status
            row["hyperparameters_enabled"] = hyperparameters_enabled  # Mark hyperparameters status
            row["data_augmentation_enabled"] = data_augmentation_enabled  # Mark data augmentation status
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def save_results_with_optional_suffix(file, results_list, suffix, base_filename_key, fallback_filename, use_stacking_subdir=False, config=None):
    """
    Save results to CSV with optional suffix-based filename.

    :param file: File path for determining output directory
    :param results_list: List of result dictionaries to save
    :param suffix: Combination suffix string
    :param base_filename_key: Config key for base filename
    :param fallback_filename: Default filename if config key not found
    :param use_stacking_subdir: Whether to use Stacking subdirectory
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        use_suffix = config.get("stacking", {}).get("use_suffix_filenames", False)  # Verify if suffix filenames are configured
        if use_suffix:  # If suffix-based files desired
            base = config.get("stacking", {}).get(base_filename_key, fallback_filename)  # Get base filename from config
            out_name = base.replace('.csv', f"{suffix}.csv")  # Compose suffixed filename
            if use_stacking_subdir:  # If Stacking subdirectory should be used
                out_path = Path(file).parent / "Feature_Analysis" / "Stacking" / out_name  # Build path with Stacking subdir
            else:  # Without Stacking subdirectory
                out_path = Path(file).parent / "Feature_Analysis" / out_name  # Build path without Stacking subdir
            save_stacking_results(str(out_path), results_list, config=config)  # Save suffixed CSV
        else:  # Default single CSV with extra columns
            save_stacking_results(file, results_list, config=config)  # Save to default location
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def orchestrate_binary_combination(file, ga_sel, pca_n, rfe_sel, base_models, hp_params_map, hyperparameters_enabled, feature_selection_enabled, data_augmentation_enabled, suffix, config=None):
    """
    Orchestrate evaluation for a single binary combination of FS/HP/DA flags.

    :param file: Path to the dataset file
    :param ga_sel: GA selected features or None
    :param pca_n: PCA components or None
    :param rfe_sel: RFE selected features or None
    :param base_models: Dictionary of base models
    :param hp_params_map: Hyperparameters mapping
    :param hyperparameters_enabled: Whether hyperparameters were enabled
    :param feature_selection_enabled: Whether feature selection was enabled
    :param data_augmentation_enabled: Whether data augmentation was enabled
    :param suffix: Combination suffix string
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: True if evaluation succeeded, False otherwise
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        df_original, feature_names = load_and_preprocess_dataset(file, None, config=config)  # Load original dataset
        if df_original is None:  # If loading failed
            print(f"{BackgroundColors.YELLOW}Skipping file {file} (failed to load).{Style.RESET_ALL}")  # Warn about load failure
            return False  # Signal failure

        send_telegram_message(TELEGRAM_BOT, [f"[BINARY] Starting evaluation | file: {os.path.basename(file)} | FS: {'ON' if feature_selection_enabled else 'OFF'} | HP: {'ON' if hyperparameters_enabled else 'OFF'} | DA: {'ON' if data_augmentation_enabled else 'OFF'}"])  # Notify Telegram about binary evaluation start

        try:  # Protect the evaluation call
            results = evaluate_on_dataset(
                file, df_original, feature_names, ga_sel, pca_n, rfe_sel, base_models,
                data_source_label="Original",
                hyperparams_map=hp_params_map if hyperparameters_enabled else {},
                experiment_id=generate_experiment_id(file, "original_only"),
                experiment_mode="original_only", augmentation_ratio=None,
                execution_mode_str="binary", attack_types_combined=None,
                df_augmented_for_training=None,
            )  # Evaluate original-only binary classification
        except Exception as e:  # If evaluation fails
            print(f"{BackgroundColors.RED}Evaluation failed for {file} combo {suffix}: {e}{Style.RESET_ALL}")  # Log error
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception
            return False  # Signal failure

        results_list = list(results.values())  # Convert results dict to list
        annotate_results_with_combination_flags(results_list, feature_selection_enabled, hyperparameters_enabled, False)  # Annotate results with flags
        save_results_with_optional_suffix(
            file, results_list, suffix, "results_filename", "Stacking_Classifiers_Results.csv",
            use_stacking_subdir=True, config=config,
        )  # Save results with optional suffix

        if data_augmentation_enabled:  # If DA requested and available
            try:  # Protect augmentation processing
                process_augmented_data_evaluation(
                    file, df_original, feature_names, ga_sel, pca_n, rfe_sel, base_models,
                    hp_params_map if hyperparameters_enabled else {}, results, config=config,
                )  # Process augmented evaluations
            except Exception as e:  # If augmentation fails
                print(f"{BackgroundColors.YELLOW}Augmentation processing failed for {file} combo {suffix}: {e}{Style.RESET_ALL}")  # Warn
                send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception
                return False  # Signal failure

        del df_original, results  # Release evaluation data to free memory after binary combination completes
        gc.collect()  # Force garbage collection to reclaim memory from evaluation data

        return True  # Signal success
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def execute_original_multiclass_evaluation(files_to_process, ga_sel, pca_n, rfe_sel, base_models, hp_params_map, hyperparameters_enabled, feature_selection_enabled, suffix, config):
    """
    Combine files for multi-class classification, evaluate the original-only dataset, annotate results, and persist them to disk.

    :param files_to_process: List of dataset CSV file paths to combine.
    :param ga_sel: GA-selected feature list or None when feature selection is disabled.
    :param pca_n: Number of PCA components or None when PCA is disabled.
    :param rfe_sel: RFE-selected feature list or None when feature selection is disabled.
    :param base_models: Dictionary mapping model names to model instances.
    :param hp_params_map: Hyperparameter mapping applied when hyperparameters are enabled.
    :param hyperparameters_enabled: Whether hyperparameter optimization results should be applied.
    :param feature_selection_enabled: Whether feature selection is active for this combination.
    :param suffix: Combination suffix string appended to result filenames.
    :param config: Configuration dictionary used throughout evaluation helpers.
    :return: Tuple of ("ok", (combined_df, attack_types, feature_names)) on success, or ("break", None)/("continue", None) on failure.
    """

    try:  # Protect multiclass combine step
        combined_df, attack_types, target_col = combine_files_for_multiclass(files_to_process, config=config)  # Combine files for multiclass
    except Exception as e:  # If combining fails
        print(f"{BackgroundColors.RED}Failed to combine files for multi-class: {e}{Style.RESET_ALL}")  # Error message
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception
        return "break", None  # Signal to break out of combination loop

    if combined_df is None:  # If no combined df produced
        print(f"{BackgroundColors.YELLOW}No multi-class dataset available. Skipping multi-class orchestration.{Style.RESET_ALL}")  # Warn
        return "break", None  # Signal to break out of combination loop

    feature_names = [c for c in combined_df.columns if c != 'attack_type']  # Extract feature names excluding target

    try:  # Protect evaluation step
        results = evaluate_on_dataset(
            "multiclass_combined", combined_df, feature_names, ga_sel, pca_n, rfe_sel, base_models,
            data_source_label="Original_MultiClass",
            hyperparams_map=hp_params_map if hyperparameters_enabled else {},
            experiment_id=generate_experiment_id("multiclass_combined", "multiclass_original_only"),
            experiment_mode="original_only", augmentation_ratio=None,
            execution_mode_str="multi-class", attack_types_combined=attack_types,
            df_augmented_for_training=None,
        )  # Evaluate multi-class original dataset
    except Exception as e:  # If evaluation fails
        print(f"{BackgroundColors.RED}Multi-class evaluation failed for combo {suffix}: {e}{Style.RESET_ALL}")  # Error
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception
        return "continue", None  # Signal to continue with next combination

    results_list = list(results.values())  # Convert results dict to list
    annotate_results_with_combination_flags(results_list, feature_selection_enabled, hyperparameters_enabled, False)  # Annotate results
    save_results_with_optional_suffix(
        files_to_process[0], results_list, suffix, "multiclass_results_filename",
        "Stacking_Classifiers_MultiClass_Results.csv", use_stacking_subdir=False, config=config,
    )  # Save multi-class results with optional suffix
    return "ok", (combined_df, attack_types, feature_names)  # Return success with data payload


def execute_multiclass_augmentation(files_to_process, combined_df, attack_types, feature_names, ga_sel, pca_n, rfe_sel, base_models, hp_params_map, hyperparameters_enabled, feature_selection_enabled, suffix, config):
    """
    Run augmented data experiments for each configured augmentation ratio using the combined multi-class dataset.

    :param files_to_process: List of original dataset CSV file paths used to locate augmented counterparts.
    :param combined_df: Combined original multi-class DataFrame used as the test set baseline.
    :param attack_types: List of attack type strings for multi-class evaluation metadata.
    :param feature_names: List of feature column names shared across all combined files.
    :param ga_sel: GA-selected feature list or None when feature selection is disabled.
    :param pca_n: Number of PCA components or None when PCA is disabled.
    :param rfe_sel: RFE-selected feature list or None when feature selection is disabled.
    :param base_models: Dictionary mapping model names to model instances.
    :param hp_params_map: Hyperparameter mapping applied when hyperparameters are enabled.
    :param hyperparameters_enabled: Whether hyperparameter optimization results should be applied.
    :param feature_selection_enabled: Whether feature selection is active for this combination.
    :param suffix: Combination suffix string appended to result filenames.
    :param config: Configuration dictionary used throughout evaluation helpers.
    :return: "continue" when augmentation orchestration fails, or None on success.
    """

    try:  # Protect augmentation orchestration
        augmented_files_list = load_augmented_files_for_multiclass(files_to_process, config=config)  # Load augmented files per file
        if not augmented_files_list:  # If none found
            print(f"{BackgroundColors.YELLOW}No augmented files found for multi-class combo {suffix}. Skipping augmentation.{Style.RESET_ALL}")  # Warn
        else:  # Have augmented files to process
            combined_aug_df, _, _ = combine_files_for_multiclass(augmented_files_list, config=config)  # Combine augmented files
            if combined_aug_df is None:  # If combine failed
                print(f"{BackgroundColors.YELLOW}Failed to combine augmented files for multi-class combo {suffix}. Skipping.{Style.RESET_ALL}")  # Warn
            else:  # Proceed with ratio experiments
                for ratio in config.get("stacking", {}).get("augmentation_ratios", [0.10, 0.25, 0.50, 0.75, 1.00]):  # For each augmentation ratio
                    df_sampled = sample_augmented_by_ratio(combined_aug_df, combined_df, ratio)  # Sample augmented data
                    if df_sampled is None:  # If sampling failed
                        print(f"{BackgroundColors.YELLOW}Sampling failed for ratio {ratio} in combo {suffix}. Skipping ratio.{Style.RESET_ALL}")  # Warn
                        continue  # Next ratio
                    try:  # Evaluate with augmented training data
                        res = evaluate_on_dataset(
                            "multiclass_combined", combined_df, feature_names, ga_sel, pca_n, rfe_sel, base_models,
                            data_source_label=f"Original+Augmented@{int(ratio*100)}%_MultiClass",
                            hyperparams_map=hp_params_map if hyperparameters_enabled else {},
                            experiment_id=generate_experiment_id("multiclass_combined", "multiclass_original_plus_augmented", ratio),
                            experiment_mode="original_plus_augmented", augmentation_ratio=ratio,
                            execution_mode_str="multi-class", attack_types_combined=attack_types,
                            df_augmented_for_training=df_sampled,
                        )  # Evaluate augmented multi-class combination
                    except Exception as e:  # If evaluation failed
                        print(f"{BackgroundColors.YELLOW}Augmented evaluation failed for ratio {ratio} combo {suffix}: {e}{Style.RESET_ALL}")  # Warn
                        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception
                        continue  # Next ratio
                    res_list = list(res.values())  # Convert augmented results to list
                    annotate_results_with_combination_flags(res_list, feature_selection_enabled, hyperparameters_enabled, True)  # Annotate with da enabled
                    save_stacking_results(files_to_process[0], res_list, config=config)  # Save augmented results
                del combined_aug_df  # Release combined augmented dataframe to free memory after all ratios
                gc.collect()  # Force garbage collection to reclaim memory from augmented data
    except Exception as e:  # Catch augmentation orchestration errors
        print(f"{BackgroundColors.YELLOW}Multi-class augmentation orchestration failed for combo {suffix}: {e}{Style.RESET_ALL}")  # Warn
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception
        return "continue"  # Signal to continue with next combination
    return None  # Signal success


def orchestrate_multiclass_combination(files_to_process, ga_sel, pca_n, rfe_sel, base_models, hp_params_map, hyperparameters_enabled, feature_selection_enabled, data_augmentation_enabled, suffix, config=None):
    """
    Orchestrate evaluation for a single multi-class combination of FS/HP/DA flags.

    :param files_to_process: List of files to process
    :param ga_sel: GA selected features or None
    :param pca_n: PCA components or None
    :param rfe_sel: RFE selected features or None
    :param base_models: Dictionary of base models
    :param hp_params_map: Hyperparameters mapping
    :param hyperparameters_enabled: Whether hyperparameters were enabled
    :param feature_selection_enabled: Whether feature selection was enabled
    :param data_augmentation_enabled: Whether data augmentation was enabled
    :param suffix: Combination suffix string
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: "break" to abort mode loop, "continue" to skip combination, or None on success
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        status, data = execute_original_multiclass_evaluation(files_to_process, ga_sel, pca_n, rfe_sel, base_models, hp_params_map, hyperparameters_enabled, feature_selection_enabled, suffix, config)  # Combine files, evaluate original dataset, and persist results
        if status != "ok":  # If evaluation did not succeed
            return status  # Propagate flow control signal to caller

        if data is None:  # Verify that data payload is present before unpacking
            print(f"{BackgroundColors.RED}Multi-class evaluation returned no data for combo {suffix}.{Style.RESET_ALL}")
            return "break"  # Signal to break out of combination loop since we cannot proceed without the combined dataset and metadata

        combined_df, attack_types, feature_names = data  # Unpack evaluation results payload

        if data_augmentation_enabled:  # If augmentation requested
            augmentation_signal = execute_multiclass_augmentation(files_to_process, combined_df, attack_types, feature_names, ga_sel, pca_n, rfe_sel, base_models, hp_params_map, hyperparameters_enabled, feature_selection_enabled, suffix, config)  # Run augmented experiments for all configured ratios
            if augmentation_signal is not None:  # If augmentation returned a flow control signal
                return augmentation_signal  # Propagate signal to caller

        del combined_df, data  # Release multi-class evaluation data to free memory
        gc.collect()  # Force garbage collection to reclaim memory from multi-class combination

        return None  # Signal success (no flow control change needed)
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def orchestrate_all_combinations(input_path, dataset_name=None, config=None):
    """
    Orchestrate all 8 combinations of Feature Selection (FS), Hyperparameter
    Optimization (HP), and Data Augmentation (DA) for each execution mode.

    This function does not change internal evaluation logic; it calls existing
    evaluation functions with appropriate arguments per combination.

    :param input_path: Path containing dataset files to process.
    :param dataset_name: Optional dataset name for logging.
    :param config: Configuration dictionary (uses global CONFIG if None).
    :return: None
    """

    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG

    files_to_process = determine_files_to_process(config.get("execution", {}).get("csv_file", None), input_path, config=config)  # Determine files

    configured_mode = config.get("execution", {}).get("execution_mode", "both")  # Read configured mode
    modes = [configured_mode] if configured_mode in ("binary", "multi-class") else ["binary", "multi-class"]  # Modes list

    methods_cfg = config.get("stacking", {}).get("methods", {})  # Retrieve method toggles from config
    fs_toggle = methods_cfg.get("feature_selection", True)  # Resolve feature selection toggle from config
    hp_toggle = methods_cfg.get("hyperparameter_optimization", True)  # Resolve hyperparameter optimization toggle from config
    da_toggle = methods_cfg.get("augmentation", True)  # Resolve data augmentation toggle from config
    automl_toggle = methods_cfg.get("automl", True)  # Resolve AutoML toggle from config

    fs_options = [True, False] if fs_toggle else [False]  # Build feature selection iteration options based on toggle
    hp_options = [True, False] if hp_toggle else [False]  # Build hyperparameter optimization iteration options based on toggle
    da_options = [True, False] if da_toggle else [False]  # Build data augmentation iteration options based on toggle

    for mode in modes:  # For each mode
        for file in files_to_process:  # For each file in the input path
            try:  # Protect individual-file orchestration
                artifacts = locate_and_verify_artifacts(file, config=config)  # Locate feature/HP/DA artifacts
                fs_ga, fs_pca, fs_rfe = artifacts.get("ga"), artifacts.get("pca"), artifacts.get("rfe")  # Unpack feature artifacts
                fs_available = bool(fs_ga or fs_rfe or fs_pca)  # True when at least one artifact exists
                hp_raw = artifacts.get("hyperparams")  # Get hyperparameter mapping if present
                hp_available = bool(hp_raw)  # True if HP results exist
                aug_file = artifacts.get("augmented_file")  # Get augmented file path if present
                aug_available = bool(aug_file)  # True when augmented file exists

                for fs_flag, hp_flag, da_flag in itertools.product(fs_options, hp_options, da_options):  # Cartesian product of FS/HP/DA flags gated by method toggles
                    feature_selection_enabled = fs_flag and fs_available  # Only enabled if requested and available
                    hyperparameters_enabled = hp_flag and hp_available  # Only enabled if requested and available
                    data_augmentation_enabled = da_flag and aug_available  # Only enabled if requested and available

                    suffix = f"_fs_{'on' if feature_selection_enabled else 'off'}_hp_{'on' if hyperparameters_enabled else 'off'}_da_{'on' if data_augmentation_enabled else 'off'}"  # Suffix

                    print(f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}Orchestrating: mode={mode}, file={file}, combo={suffix}{Style.RESET_ALL}")  # Log orchestration
                    send_telegram_message(TELEGRAM_BOT, [f"[{mode.upper()}] Starting combination: {suffix} | file: {os.path.basename(file)}"])  # Notify Telegram about FS/HP/DA combination start

                    if hyperparameters_enabled:  # If we should use hyperparameters
                        base_models, hp_params_map = prepare_models_with_hyperparameters(file, config=config)  # Apply HP mapping
                    else:  # Hyperparameters disabled -> use defaults
                        base_models = get_models(config=config)  # Instantiate base models without HP
                        hp_params_map = {}  # Empty HP mapping

                    ga_sel = fs_ga if feature_selection_enabled else None  # GA features or None
                    pca_n = fs_pca if feature_selection_enabled else None  # PCA components or None
                    rfe_sel = fs_rfe if feature_selection_enabled else None  # RFE features or None

                    if mode == "binary":  # Binary evaluation per-file
                        if not orchestrate_binary_combination(
                            file, ga_sel, pca_n, rfe_sel, base_models, hp_params_map,
                            hyperparameters_enabled, feature_selection_enabled, data_augmentation_enabled, suffix, config=config,
                        ):  # If binary combination evaluation failed
                            continue  # Skip to next combination

                    elif mode == "multi-class":  # Multi-class orchestration
                        signal = orchestrate_multiclass_combination(
                            files_to_process, ga_sel, pca_n, rfe_sel, base_models, hp_params_map,
                            hyperparameters_enabled, feature_selection_enabled, data_augmentation_enabled, suffix, config=config,
                        )  # Orchestrate multi-class combination evaluation
                        if signal == "break":  # If combination cannot proceed for this mode
                            break  # Abort multi-class for this file list
                        elif signal == "continue":  # If combination should skip to next
                            continue  # Continue with next combination

                    else:  # Unknown mode (shouldn't happen)
                        print(f"{BackgroundColors.YELLOW}Unknown execution mode: {mode}{Style.RESET_ALL}")  # Warn
                        continue  # Skip

            except Exception as e:  # If per-file orchestration fails
                print(f"{BackgroundColors.RED}Orchestration failed for file {file}: {e}{Style.RESET_ALL}")  # Error
                send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception
                continue  # Continue with next file


def execute_both_mode_pipeline(files_to_process, local_dataset_name, config=None):
    """
    Execute both binary and multi-class classification pipelines sequentially.

    :param files_to_process: List of file paths to process
    :param local_dataset_name: Name of the dataset being processed
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}Execution Mode: BOTH (Binary + Multi-Class){Style.RESET_ALL}",
            config=config
        )  # Output execution mode

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}"
        )  # Print separator line
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}BOTH MODE: Running Binary and Multi-Class pipelines sequentially{Style.RESET_ALL}"
        )  # Print mode header
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}\n"
        )  # Print closing separator

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}[STEP 1/2] Executing BINARY Classification Pipeline{Style.RESET_ALL}\n"
        )  # Print binary step header

        combined_df, combined_file_for_features, files_for_binary = combine_dataset_if_needed(files_to_process, config=config)  # Combine dataset files if needed

        for file in files_for_binary:  # For each file to process in binary mode
            orchestrate_all_combinations(file, dataset_name=local_dataset_name, config=config)  # Orchestrate all combinations for binary mode

        del combined_df, combined_file_for_features, files_for_binary  # Release binary phase data to free memory before multi-class
        gc.collect()  # Force garbage collection to reclaim memory from binary phase before multi-class loading

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}✓ Binary pipeline complete{Style.RESET_ALL}\n"
        )  # Print binary completion message

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}[STEP 2/2] Executing MULTI-CLASS Classification Pipeline{Style.RESET_ALL}\n"
        )  # Print multi-class step header

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}"
        )  # Print separator line
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}MULTI-CLASS CLASSIFICATION MODE{Style.RESET_ALL}"
        )  # Print mode header
        print(
            f"{BackgroundColors.GREEN}Combining {BackgroundColors.CYAN}{len(files_to_process)}{BackgroundColors.GREEN} files into single multi-class dataset{Style.RESET_ALL}"
        )  # Print files count
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}\n"
        )  # Print closing separator

        combined_multiclass_df, attack_types_list, target_col_name = combine_files_for_multiclass(files_to_process, config=config)  # Combine files for multi-class

        if combined_multiclass_df is None:  # If combination failed
            print(
                f"{BackgroundColors.RED}Failed to create multi-class dataset. Skipping multi-class evaluation.{Style.RESET_ALL}"
            )  # Print error about combination failure
        else:  # If combination succeeded
            process_multiclass_evaluation(
                files_to_process, combined_multiclass_df, attack_types_list, local_dataset_name, config=config
            )  # Process multi-class evaluation workflow

            del combined_multiclass_df  # Release combined multi-class dataframe to free memory after evaluation
            gc.collect()  # Force garbage collection to reclaim memory from multi-class evaluation

            print(
                f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}✓ Multi-class pipeline complete{Style.RESET_ALL}\n"
            )  # Print multi-class completion message

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}"
        )  # Print final separator
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}✓ BOTH MODE COMPLETE: Binary and Multi-Class pipelines finished{Style.RESET_ALL}"
        )  # Print both mode completion message
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}\n"
        )  # Print final separator
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def execute_multiclass_mode_pipeline(files_to_process, local_dataset_name, config=None):
    """
    Execute multi-class classification pipeline only.

    :param files_to_process: List of file paths to process
    :param local_dataset_name: Name of the dataset being processed
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}Execution Mode: MULTI-CLASS{Style.RESET_ALL}",
            config=config
        )  # Output execution mode

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}"
        )  # Print separator line
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}MULTI-CLASS CLASSIFICATION MODE{Style.RESET_ALL}"
        )  # Print mode header
        print(
            f"{BackgroundColors.GREEN}Combining {BackgroundColors.CYAN}{len(files_to_process)}{BackgroundColors.GREEN} files into single multi-class dataset{Style.RESET_ALL}"
        )  # Print files count
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}\n"
        )  # Print closing separator

        combined_multiclass_df, attack_types_list, target_col_name = combine_files_for_multiclass(files_to_process, config=config)  # Combine files for multi-class

        if combined_multiclass_df is None:  # If combination failed
            print(
                f"{BackgroundColors.RED}Failed to create multi-class dataset. Skipping multi-class evaluation.{Style.RESET_ALL}"
            )  # Print error about combination failure
            return  # Exit function early

        process_multiclass_evaluation(
            files_to_process, combined_multiclass_df, attack_types_list, local_dataset_name, config=config
        )  # Process multi-class evaluation workflow

        del combined_multiclass_df  # Release combined multi-class dataframe to free memory after evaluation
        gc.collect()  # Force garbage collection to reclaim memory from multi-class evaluation
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def execute_binary_mode_pipeline(files_to_process, local_dataset_name, config=None):
    """
    Execute binary classification pipeline only.

    :param files_to_process: List of file paths to process
    :param local_dataset_name: Name of the dataset being processed
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}Execution Mode: BINARY{Style.RESET_ALL}",
            config=config
        )  # Output execution mode

        combined_df, combined_file_for_features, files_to_process = combine_dataset_if_needed(files_to_process, config=config)  # Combine dataset files if needed

        for file in files_to_process:  # For each file to process
            orchestrate_all_combinations(file, dataset_name=local_dataset_name, config=config)  # Orchestrate all combinations for binary mode
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def process_files_in_path(input_path, dataset_name, config=None):
    """
    Processes all files in a given input path including file discovery and dataset combination.

    :param input_path: Directory path containing files to process
    :param dataset_name: Name of the dataset being processed
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Processing files in path: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        if not verify_filepath_exists(input_path):  # If the input path does not exist
            verbose_output(
                f"{BackgroundColors.YELLOW}Skipping missing path: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}",
                config=config
            )  # Output skip message
            return  # Exit function early
        
        csv_file = config.get("execution", {}).get("csv_file", None)  # Get CSV file override from config
        execution_mode = config.get("execution", {}).get("execution_mode", "both")  # Get execution mode from config (binary/multi-class/both, default: both)

        print(f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}[DEBUG] Classification mode: {execution_mode}{Style.RESET_ALL}")  # Log the resolved classification mode
        send_telegram_message(TELEGRAM_BOT, [f"Classification mode: {execution_mode} | path: {os.path.relpath(input_path)}"])  # Notify Telegram about classification mode and relative input path

        files_to_process = determine_files_to_process(csv_file, input_path, config=config)  # Determine which files to process

        local_dataset_name = dataset_name or get_dataset_name(input_path)  # Use provided dataset name or infer from path

        if execution_mode == "both":  # If BOTH execution modes are enabled (run binary first, then multi-class)
            print(f"{BackgroundColors.CYAN}[DEBUG] Running binary and multi-class classification pipelines sequentially{Style.RESET_ALL}")  # Log both mode execution
            execute_both_mode_pipeline(files_to_process, local_dataset_name, config=config)  # Run binary + multi-class pipelines sequentially
        elif execution_mode == "multi-class":  # If multi-class execution mode is enabled
            print(f"{BackgroundColors.CYAN}[DEBUG] Running multiclass classification pipeline{Style.RESET_ALL}")  # Log multiclass pipeline start
            execute_multiclass_mode_pipeline(files_to_process, local_dataset_name, config=config)  # Run multi-class pipeline only
        else:  # If binary execution mode (default)
            print(f"{BackgroundColors.CYAN}[DEBUG] Running binary classification pipeline{Style.RESET_ALL}")  # Log binary pipeline start
            execute_binary_mode_pipeline(files_to_process, local_dataset_name, config=config)  # Run binary pipeline only
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def process_dataset_paths(dataset_name, paths, config=None):
    """
    Processes all paths for a given dataset.

    :param dataset_name: Name of the dataset
    :param paths: List of paths to process for this dataset
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Processing dataset: {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing dataset: {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}"
        )  # Print dataset name
        send_telegram_message(TELEGRAM_BOT, [f"Processing dataset: {dataset_name} ({len(paths)} path(s))"])  # Notify Telegram about dataset processing start

        for input_path in paths:  # For each path in the dataset's paths list
            process_files_in_path(input_path, dataset_name, config=config)  # Process all files in  this path
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_stacking_pipeline(config_path=None, **config_overrides):
    """
    Programmatic entry point for stacking classifier evaluation.
    
    Allows calling this module as a library with configuration overrides:
    
    Example:
        from stacking import run_stacking_pipeline
        
        # Using config file
        run_stacking_pipeline(config_path="custom_config.yaml")
        
        # Using direct parameter overrides
        run_stacking_pipeline(
            execution={\"verbose\": True, \"test_data_augmentation\": False},
            automl={\"enabled\": True, \"n_trials\": 100}
        )
    
    :param config_path: Path to configuration file (None for default config.yaml)
    :param config_overrides: Dictionary overrides for configuration
    :return: None
    """
    
    config = initialize_config(config_path=config_path, cli_args=None)  # Load base config
    
    for key, value in config_overrides.items():  # Iterate over provided overrides
        if isinstance(value, dict) and key in config:  # If override is dict and key exists
            config[key] = deep_merge_dicts(config[key], value)  # Deep merge override
        else:  # Direct override
            config[key] = value  # Set value directly
    
    initialize_logger(config=config)  # Setup logging
    
    main(config=config)  # Execute stacking pipeline


def to_seconds(obj):
    """
    Converts various time-like objects to seconds.
    
    :param obj: The object to convert (can be int, float, timedelta, datetime, etc.)
    :return: The equivalent time in seconds as a float, or None if conversion fails
    """
    
    try:
        if obj is None:  # None can't be converted
            return None  # Signal failure to convert
        if isinstance(obj, (int, float)):  # Already numeric (seconds or timestamp)
            return float(obj)  # Return as float seconds
        if hasattr(obj, "total_seconds"):  # Timedelta-like objects
            try:  # Attempt to call total_seconds()
                return float(obj.total_seconds())  # Use the total_seconds() method
            except Exception:
                pass  # Fallthrough on error
        if hasattr(obj, "timestamp"):  # Datetime-like objects
            try:  # Attempt to call timestamp()
                return float(obj.timestamp())  # Use timestamp() to get seconds since epoch
            except Exception:
                pass  # Fallthrough on error
        return None  # Couldn't convert
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def calculate_execution_time(start_time, finish_time=None):
    """
    Calculate the execution time and return a human-readable string.

    :param start_time: The start time or duration value (datetime, timedelta, or numeric seconds).
    :param finish_time: Optional finish time; if None, start_time is treated as the total duration.
    :return: Human-readable execution time string formatted as days, hours, minutes, and seconds.
    """
    
    try:
        if finish_time is None:  # Single-argument mode: start_time already represents duration or seconds
            total_seconds = to_seconds(start_time)  # Try to convert provided value to seconds
            if total_seconds is None:  # Conversion failed
                try:  # Attempt numeric coercion
                    total_seconds = float(start_time)  # Attempt numeric coercion
                except Exception:
                    total_seconds = 0.0  # Fallback to zero
        else:  # Two-argument mode: Compute difference finish_time - start_time
            st = to_seconds(start_time)  # Convert start to seconds if possible
            ft = to_seconds(finish_time)  # Convert finish to seconds if possible
            if st is not None and ft is not None:  # Both converted successfully
                total_seconds = ft - st  # Direct numeric subtraction
            else:  # Fallback to other methods
                try:  # Attempt to subtract (works for datetimes/timedeltas)
                    delta = finish_time - start_time  # Try subtracting (works for datetimes/timedeltas)
                    total_seconds = float(delta.total_seconds())  # Get seconds from the resulting timedelta
                except Exception:  # Subtraction failed
                    try:  # Final attempt: Numeric coercion
                        total_seconds = float(finish_time) - float(start_time)  # Final numeric coercion attempt
                    except Exception:  # Numeric coercion failed
                        total_seconds = 0.0  # Fallback to zero on failure

        if total_seconds is None:  # Ensure a numeric value
            total_seconds = 0.0  # Default to zero
        if total_seconds < 0:  # Normalize negative durations
            total_seconds = abs(total_seconds)  # Use absolute value

        days = int(total_seconds // 86400)  # Compute full days
        hours = int((total_seconds % 86400) // 3600)  # Compute remaining hours
        minutes = int((total_seconds % 3600) // 60)  # Compute remaining minutes
        seconds = int(total_seconds % 60)  # Compute remaining seconds

        if days > 0:  # Include days when present
            return f"{days}d {hours}h {minutes}m {seconds}s"  # Return formatted days+hours+minutes+seconds
        if hours > 0:  # Include hours when present
            return f"{hours}h {minutes}m {seconds}s"  # Return formatted hours+minutes+seconds
        if minutes > 0:  # Include minutes when present
            return f"{minutes}m {seconds}s"  # Return formatted minutes+seconds
        return f"{seconds}s"  # Fallback: only seconds
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def play_sound(config=None):
    """
    Play a sound when the program finishes and skip if the operating system is Windows.

    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None.
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        current_os = platform.system()  # Get the current operating system
        if current_os == "Windows":  # If the current operating system is Windows
            return  # Do nothing
        
        sound_enabled = config.get("sound", {}).get("enabled", True)  # Get sound enabled flag from config
        if not sound_enabled:  # If sound is disabled
            return  # Do nothing
        
        sound_file = config.get("sound", {}).get("file", "./.assets/Sounds/NotificationSound.wav")  # Get sound file from config
        sound_commands = config.get("sound", {}).get("commands", {})  # Get sound commands from config

        if verify_filepath_exists(sound_file):  # If the sound file exists
            if current_os in sound_commands:  # If the platform.system() is in the sound_commands dictionary
                os.system(f"{sound_commands[current_os]} {sound_file}")  # Play the sound
            else:  # If the platform.system() is not in the sound_commands dictionary
                print(
                    f"{BackgroundColors.RED}The {BackgroundColors.CYAN}{current_os}{BackgroundColors.RED} is not in the {BackgroundColors.CYAN}sound_commands dictionary{BackgroundColors.RED}. Please add it!{Style.RESET_ALL}"
                )
        else:  # If the sound file does not exist
            print(
                f"{BackgroundColors.RED}Sound file {BackgroundColors.CYAN}{sound_file}{BackgroundColors.RED} not found. Make sure the file exists.{Style.RESET_ALL}"
            )
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def validate_and_resolve_dataset_path(dataset_path: str, config: dict) -> dict:
    """
    Validate the CLI dataset path and return an overridden datasets dictionary.

    :param dataset_path: Path to a dataset directory or single CSV file from CLI
    :param config: Configuration dictionary for fallback dataset name resolution
    :return: Dictionary mapping dataset name to list of paths
    """

    try:
        resolved = os.path.abspath(dataset_path)  # Resolve to absolute path

        if not os.path.exists(resolved):  # Verify path existence before proceeding
            raise FileNotFoundError(f"Dataset path does not exist: {resolved}")  # Raise descriptive error for missing path

        dataset_name = get_dataset_name(resolved)  # Extract dataset name from resolved path

        if os.path.isfile(resolved):  # Verify if the path points to a single file
            if not resolved.lower().endswith(".csv"):  # Verify file has CSV extension
                raise ValueError(f"Dataset file must be a CSV: {resolved}")  # Raise error for non-CSV files
            print(f"{BackgroundColors.GREEN}[INFO] CLI --dataset-path resolved to single file: {BackgroundColors.CYAN}{resolved}{Style.RESET_ALL}")  # Log resolved file path
            config["execution"]["csv_file"] = resolved  # Set CSV file override for single-file processing
            return {dataset_name: [os.path.dirname(resolved)]}  # Return parent directory containing the file

        elif os.path.isdir(resolved):  # Verify if the path points to a directory
            print(f"{BackgroundColors.GREEN}[INFO] CLI --dataset-path resolved to directory: {BackgroundColors.CYAN}{resolved}{Style.RESET_ALL}")  # Log resolved directory path
            return {dataset_name: [resolved]}  # Return directory path as dataset entry

        else:  # Path exists but is neither file nor directory
            raise ValueError(f"Dataset path is neither a file nor directory: {resolved}")  # Raise error for unsupported path type
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def log_resolved_configuration(config: dict) -> None:
    """
    Log the resolved configuration for dataset path and method toggles.

    :param config: Configuration dictionary with resolved settings
    :return: None
    """

    try:
        dataset_path_cli = config.get("execution", {}).get("dataset_path", None)  # Retrieve CLI dataset path override
        methods_cfg = config.get("stacking", {}).get("methods", {})  # Retrieve method toggles from config

        fs_enabled = methods_cfg.get("feature_selection", True)  # Resolve feature selection toggle state
        hp_enabled = methods_cfg.get("hyperparameter_optimization", True)  # Resolve hyperparameter optimization toggle state
        da_enabled = methods_cfg.get("augmentation", True)  # Resolve data augmentation toggle state
        automl_enabled = methods_cfg.get("automl", True)  # Resolve AutoML toggle state

        if dataset_path_cli is not None:  # Verify if dataset path was overridden via CLI
            print(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}[INFO] Dataset path override (CLI): {BackgroundColors.CYAN}{dataset_path_cli}{Style.RESET_ALL}")  # Log CLI dataset path override
        else:  # Dataset path is from config.yaml or default
            print(f"{BackgroundColors.GREEN}[INFO] Dataset path source: {BackgroundColors.CYAN}config.yaml (default){Style.RESET_ALL}")  # Log config-based dataset path

        print(f"{BackgroundColors.GREEN}[INFO] Method toggle — Feature Selection: {BackgroundColors.CYAN}{fs_enabled}{Style.RESET_ALL}")  # Log feature selection toggle state
        print(f"{BackgroundColors.GREEN}[INFO] Method toggle — Hyperparameter Optimization: {BackgroundColors.CYAN}{hp_enabled}{Style.RESET_ALL}")  # Log hyperparameter optimization toggle state
        print(f"{BackgroundColors.GREEN}[INFO] Method toggle — Data Augmentation: {BackgroundColors.CYAN}{da_enabled}{Style.RESET_ALL}")  # Log data augmentation toggle state
        print(f"{BackgroundColors.GREEN}[INFO] Method toggle — AutoML: {BackgroundColors.CYAN}{automl_enabled}{Style.RESET_ALL}")  # Log AutoML toggle state

        overrides = []  # Initialize list for override source tracking

        if dataset_path_cli is not None:  # Verify if dataset path came from CLI
            overrides.append("dataset_path")  # Track dataset path as CLI override

        if any(k in methods_cfg for k in ("feature_selection", "hyperparameter_optimization", "augmentation", "automl")):  # Verify if any method toggle was explicitly set
            overrides.append("method_toggles")  # Track method toggles as override source

        if overrides:  # Verify if any overrides were detected
            print(f"{BackgroundColors.YELLOW}[INFO] Configuration overrides active for: {BackgroundColors.CYAN}{', '.join(overrides)}{Style.RESET_ALL}")  # Log active override sources
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def main(config=None):
    """
    Main function.

    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        print(
            f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Classifiers Stacking{BackgroundColors.GREEN} program!{Style.RESET_ALL}\n"
        )  # Output the welcome message

        log_resolved_configuration(config=config)  # Log resolved dataset path and method toggle states
        
        test_data_augmentation = config.get("execution", {}).get("test_data_augmentation", True)  # Get test augmentation flag from config
        augmentation_ratios = config.get("stacking", {}).get("augmentation_ratios", [0.10, 0.25, 0.50, 0.75, 1.00])  # Get augmentation ratios from config

        if test_data_augmentation:  # If data augmentation testing is enabled
            print(
                f"{BackgroundColors.BOLD}{BackgroundColors.YELLOW}Data Augmentation Testing: {BackgroundColors.CYAN}ENABLED{Style.RESET_ALL}"
            )  # Print augmentation enabled message
            print(
                f"{BackgroundColors.GREEN}Will evaluate Original vs Original+Augmented at ratios: {BackgroundColors.CYAN}{[f'{int(r*100)}%' for r in augmentation_ratios]}{Style.RESET_ALL}\n"
            )  # Print augmentation ratios to be evaluated

        start_time = datetime.datetime.now()  # Get the start time of the program

        setup_telegram_bot(config=config)  # Setup Telegram bot if configured

        _exec_mode = config.get("execution", {}).get("execution_mode", "both")  # Retrieve execution mode from config
        _dataset_path_cli = config.get("execution", {}).get("dataset_path", None)  # Retrieve CLI dataset path override
        _methods_cfg = config.get("stacking", {}).get("methods", {})  # Retrieve method toggles from config
        _fs_on = _methods_cfg.get("feature_selection", True)  # Resolve feature selection toggle
        _hp_on = _methods_cfg.get("hyperparameter_optimization", True)  # Resolve hyperparameter optimization toggle
        _da_on = _methods_cfg.get("augmentation", True)  # Resolve data augmentation toggle
        _automl_on = _methods_cfg.get("automl", True)  # Resolve AutoML toggle
        _start_lines = [
            f"Starting Classifiers Stacking at {start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Execution mode: {_exec_mode} | Dataset: {_dataset_path_cli if _dataset_path_cli else 'config.yaml (default)'} | Methods: Feature Selection: {'ON' if _fs_on else 'OFF'}, HP Optimization: {'ON' if _hp_on else 'OFF'}, Data Augmentation: {'ON' if _da_on else 'OFF'}, AutoML: {'ON' if _automl_on else 'OFF'}",
        ]
        
        if test_data_augmentation and _da_on:  # If augmentation testing is enabled and augmentation toggle is on
            _start_lines.append(f"Augmentation ratios: {[f'{int(r * 100)}%' for r in augmentation_ratios]}")  # Append augmentation ratios to start message
        
        send_telegram_message(TELEGRAM_BOT, _start_lines)  # Send detailed start message with full execution configuration

        threads_limit = set_threads_limit_based_on_ram(config=config)  # Adjust config.get("evaluation", {}).get("threads_limit", 2) based on system RAM
        
        dataset_path_override = config.get("execution", {}).get("dataset_path", None)  # Retrieve CLI --dataset-path override

        if dataset_path_override is not None:  # Verify if CLI dataset path was provided
            datasets = validate_and_resolve_dataset_path(dataset_path_override, config=config)  # Validate and resolve CLI dataset path into datasets dict
        else:  # No CLI override, use config.yaml datasets
            datasets = config.get("dataset", {}).get("datasets", {})  # Get datasets from config

        for dataset_name, paths in datasets.items():  # For each dataset in the datasets dictionary
            process_dataset_paths(dataset_name, paths, config=config)  # Process all paths for this dataset

        finish_time = datetime.datetime.now()  # Get the finish time of the program
        print(
            f"\n{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
        )  # Output the start and finish times
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
        )  # Output the end of the program message

        send_telegram_message(TELEGRAM_BOT, [f"Finished Classifiers Stacking at {finish_time.strftime('%Y-%m-%d %H:%M:%S')} | Execution time: {calculate_execution_time(start_time, finish_time)}"])  # Send Telegram message indicating finish
        
        play_sound_enabled = config.get("sound", {}).get("enabled", True)  # Get play sound flag from config
        if play_sound_enabled:  # If play sound is enabled
            atexit.register(play_sound, config=config)  # Register the play_sound function to be called when the program finishes
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


if __name__ == "__main__":
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """
    
    try:  # Protect top-level execution to ensure errors are reported and notified
        cli_args = parse_cli_args()  # Parse command-line arguments into namespace
        config = initialize_config(config_path=cli_args.config, cli_args=cli_args)  # Merge configuration from file and CLI
        initialize_logger(config=config)  # Initialize logger and redirect stdout/stderr to logger
        try:  # Run main and handle user interrupts separately
            main(config=config)  # Invoke main business logic for stacking pipeline
        except KeyboardInterrupt:  # Handle user-initiated interrupts with friendly notification
            try:  # Attempt graceful interrupt notification and cleanup
                print("Execution interrupted by user (KeyboardInterrupt)")  # Inform terminal about user interrupt
                send_telegram_message(TELEGRAM_BOT, ["Stacking pipeline interrupted by user (KeyboardInterrupt)"])  # Notify via Telegram about interrupt
            except Exception:  # Ignore failures sending interrupt notification to avoid masking the interrupt
                pass  # No-op on notification failure
            try:  # Best-effort logger flush/close during interrupt handling
                if logger is not None:  # Only flush/close if logger exists
                    logger.flush()  # Flush pending log writes
                    logger.close()  # Close the logger file handle
            except Exception:  # Ignore logger cleanup errors during interrupt handling
                pass  # Continue to re-raise the interrupt
            raise  # Re-raise KeyboardInterrupt to preserve original exit semantics
    except BaseException as e:  # Catch everything (including SystemExit) and report
        try:  # Try to log and notify about the fatal error
            print(f"Fatal error: {e}")  # Print the exception message to terminal for visibility
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback and message via Telegram
        except Exception:  # If notification fails, attempt to print traceback to stderr as fallback
            try:  # Attempt fallback traceback printing for diagnostics
                traceback.print_exc()  # Print full traceback to stderr as a fallback notification
            except Exception:  # Ignore failures of the fallback printing to avoid cascading errors
                pass  # No further fallback available
        try:  # Attempt best-effort logger cleanup after fatal error
            if logger is not None:  # Only flush/close if logger initialized
                logger.flush()  # Flush pending log writes
                logger.close()  # Close the logger file handle
        except Exception:  # Ignore logger cleanup errors to avoid masking the primary failure
            pass  # No-op on cleanup failure
        raise  # Re-raise the original exception to preserve non-zero exit code and behavior
