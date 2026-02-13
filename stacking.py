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

TODOs:
    - Add CLI argument parsing for dataset paths and runtime flags.
    - Add native Parquet support and safer large-file streaming.
    - Add voting ensemble baseline and parallelize per-feature-set evaluations.

Dependencies:
    - Python >= 3.8
    - pandas, numpy, scikit-learn, colorama, lightgbm, xgboost
    - Optional: telegram_bot for notifications
"""

import argparse  # For parsing command-line arguments
import ast  # For safely evaluating Python literals
import atexit  # For playing a sound when the program finishes
import concurrent.futures  # For parallel execution
import datetime  # For getting the current date and time
import glob  # For file pattern matching
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
import subprocess  # For running small system commands (sysctl/wmic)
import sys  # For system-specific parameters and functions
import telegram_bot as telegram_module  # For setting Telegram prefix and device info
import time  # For measuring execution time
from colorama import Style  # For terminal text styling
from joblib import dump, load  # For exporting and loading trained models and scalers
from Logger import Logger  # For logging output to both terminal and file
from pathlib import Path  # For handling file paths
from sklearn.decomposition import PCA  # For Principal Component Analysis
from sklearn.ensemble import (  # For ensemble models
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression  # For logistic regression model
from sklearn.manifold import TSNE  # For t-SNE dimensionality reduction
from sklearn.metrics import (  # For performance metrics
    accuracy_score,
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
from telegram_bot import TelegramBot, send_telegram_message  # For sending progress messages to Telegram
from tqdm import tqdm  # For progress bars
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


def initialize_logger(config=None):
    """
    Initialize logger using configuration.
    
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
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


def verbose_output(true_string="", false_string="", config=None):
    """
    Outputs a message if verbose mode is enabled in configuration.

    :param true_string: The string to be outputted if verbose is enabled.
    :param false_string: The string to be outputted if verbose is disabled.
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG
    
    verbose = config.get("execution", {}).get("verbose", False)  # Get verbose flag
    
    if verbose and true_string != "":  # If verbose is True and a true_string was provided
        print(true_string)  # Output the true statement string
    elif false_string != "":  # If a false_string was provided
        print(false_string)  # Output the false statement string


def verify_dot_env_file(config=None):
    """
    Verifies if the .env file exists in the current directory.

    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: True if the .env file exists, False otherwise
    """

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


def setup_telegram_bot(config=None):
    """
    Sets up the Telegram bot for progress messages.

    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
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
        telegram_module.TELEGRAM_DEVICE_INFO = f"{telegram_module.get_local_ip()} - {platform.system()}"
        telegram_module.RUNNING_CODE = os.path.basename(__file__)
    except Exception as e:
        print(f"{BackgroundColors.RED}Failed to initialize Telegram bot: {e}{Style.RESET_ALL}")
        TELEGRAM_BOT = None  # Set to None if initialization fails


def set_threads_limit_based_on_ram(config=None):
    """
    Sets threads limit to 1 if system RAM is below threshold to avoid memory issues.

    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Threads limit value
    """
    
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


def verify_filepath_exists(filepath):
    """
    Verify if a file or folder exists at the specified path.

    :param filepath: Path to the file or folder
    :return: True if the file or folder exists, False otherwise
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Verifying if the file or folder exists at the path: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}"
    )  # Output the verbose message

    return os.path.exists(filepath)  # Return True if the file or folder exists, False otherwise


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
    
    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG

    verbose_output(
        f"{BackgroundColors.GREEN}Getting all {BackgroundColors.CYAN}{file_extension}{BackgroundColors.GREEN} files in: {BackgroundColors.CYAN}{directory_path}{Style.RESET_ALL}",
        config=config
    )  # Verbose: starting file collection
    verify_filepath_exists(directory_path)  # Validate directory path exists

    if not os.path.isdir(directory_path):  # Check if path is a valid directory
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

        if any(ignore == filename or ignore == item_path for ignore in ignore_files):  # Check if file is in ignore list
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


def get_dataset_name(input_path):
    """
    Extract the dataset name from CSVs path.

    :param input_path: Path to the CSVs files
    :return: Dataset name
    """

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


def process_single_file(f, config=None):
    """
    Process a single dataset file: load, preprocess, and extract target and features.

    :param f: Path to the dataset CSV file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (df_clean, target_col, feat_cols) or None if invalid
    """
    
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
    if df_clean is None or df_clean.empty:  # If preprocessing failed or dataframe is empty
        return None  # Return None

    target_col = df_clean.columns[-1]  # Get the last column as target
    feat_cols = [c for c in df_clean.columns[:-1] if pd.api.types.is_numeric_dtype(df_clean[c])]  # Get numeric feature columns
    if not feat_cols:  # If no numeric features
        return None  # Return None

    return (df_clean, target_col, feat_cols)  # Return the processed data


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


def intersect_features(common_features, feat_cols, config=None):
    """
    Intersect features with common features set.

    :param common_features: Current common features set (or None)
    :param feat_cols: Feature columns in this file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Updated common features set
    """
    
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


def find_common_features_and_target(processed_files, config=None):
    """
    Find common features and consistent target column from processed files.

    :param processed_files: List of (f, df_clean, target_col, feat_cols)
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (common_features, target_col_name, dfs) or (None, None, []) if invalid
    """
    
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


def create_reduced_dataframes(dfs, common_features, target_col_name, config=None):
    """
    Create reduced dataframes with only common features and target.

    :param dfs: List of (f, df_clean)
    :param common_features: List of common feature names
    :param target_col_name: Name of the target column
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: List of reduced dataframes
    """
    
    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG
    
    verbose_output(
        f"{BackgroundColors.GREEN}Creating reduced dataframes with common features and target...{Style.RESET_ALL}",
        config=config
    )  # Output the verbose message

    reduced_dfs = []  # Initialize list for reduced dataframes
    for f, df_clean in dfs:  # Iterate over valid dataframes
        cols_to_keep = [c for c in common_features if c in df_clean.columns]  # Get common features present in this df
        cols_to_keep.append(target_col_name)  # Add the target column
        reduced = df_clean.loc[:, cols_to_keep].copy()  # Create reduced dataframe
        reduced_dfs.append(reduced)  # Add to list

    return reduced_dfs  # Return the reduced dataframes


def combine_and_clean_dataframes(reduced_dfs, config=None):
    """
    Combine reduced dataframes, clean, and validate.

    :param reduced_dfs: List of reduced dataframes
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Combined dataframe or None if empty
    """
    
    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG
    
    verbose_output(
        f"{BackgroundColors.GREEN}Combining reduced dataframes and cleaning...{Style.RESET_ALL}",
        config=config
    )  # Output the verbose message

    combined = pd.concat(reduced_dfs, ignore_index=True)  # Concatenate all reduced dataframes
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()  # Replace inf with nan and drop na

    if combined.empty:  # If combined dataframe is empty
        print(f"{BackgroundColors.RED}Combined dataset is empty after alignment and NaN removal.{Style.RESET_ALL}")  # Print error
        return None  # Return None

    return combined  # Return the combined dataframe


def combine_dataset_files(files_list, config=None):
    """
    Load, preprocess and combine multiple dataset CSVs into a single DataFrame.

    :param files_list: List of dataset CSV file paths to combine
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Combined DataFrame with aligned features and target, or None if no compatible files found
    """
    
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
        return None  # Return None

    reduced_dfs = create_reduced_dataframes(dfs, common_features, target_col_name, config=config)  # Create reduced dataframes
    combined = combine_and_clean_dataframes(reduced_dfs, config=config)  # Combine and clean the dataframes

    return combined, target_col_name  # Return the combined dataframe and target column name


def find_data_augmentation_file(original_file_path, config=None):
    """
    Find the corresponding data augmentation file for an original CSV file.
    Matches wgangp.py naming: <parent>/Data_Augmentation/<stem>_data_augmented<suffix>.

    :param original_file_path: Path to the original CSV file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Path to the augmented file if it exists, None otherwise
    """
    
    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG

    verbose_output(
        f"{BackgroundColors.GREEN}Looking for data augmentation file for: {BackgroundColors.CYAN}{original_file_path}{Style.RESET_ALL}",
        config=config
    )  # Output the verbose message

    data_augmentation_suffix = config.get("stacking", {}).get("data_augmentation_suffix", "_data_augmented")  # Get suffix from config
    original_path = Path(original_file_path)  # Create Path object from the original file path
    augmented_dir = original_path.parent / "Data_Augmentation"  # Build Data_Augmentation subdirectory path
    augmented_filename = f"{original_path.stem}{data_augmentation_suffix}{original_path.suffix}"  # Build augmented filename with wgangp.py suffix convention
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


def merge_original_and_augmented(original_df, augmented_df, config=None):
    """
    Merge original and augmented dataframes by concatenating them.

    :param original_df: Original DataFrame
    :param augmented_df: Augmented DataFrame
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Merged DataFrame
    """
    
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


def calculate_metric_improvement(original_value, augmented_value):
    """
    Calculate percentage improvement of a metric.

    :param original_value: Original metric value
    :param augmented_value: Augmented metric value
    :return: Percentage improvement (positive = better, negative = worse)
    """

    if original_value == 0:  # Avoid division by zero
        return 0.0 if augmented_value == 0 else float('inf')  # Return 0 if both are 0, inf if only original is 0
    
    improvement = ((augmented_value - original_value) / original_value) * 100  # Calculate percentage improvement
    return improvement  # Return improvement


def generate_experiment_id(file_path, experiment_mode, augmentation_ratio=None):
    """
    Generates a unique experiment identifier for traceability in CSV results.

    :param file_path: Path to the dataset file being processed
    :param experiment_mode: Experiment mode string (e.g., 'original_only' or 'original_plus_augmented')
    :param augmentation_ratio: Augmentation ratio float (e.g., 0.10) or None for original-only mode
    :return: String experiment identifier combining timestamp, filename, mode and ratio
    """

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Create a timestamp string for uniqueness
    file_stem = Path(file_path).stem  # Extract the filename stem without extension
    ratio_tag = f"_ratio{int(augmentation_ratio * 100)}" if augmentation_ratio is not None else ""  # Build ratio tag string or empty
    experiment_id = f"{timestamp}_{file_stem}_{experiment_mode}{ratio_tag}"  # Concatenate all parts into unique identifier

    return experiment_id  # Return the generated experiment identifier


def validate_augmented_dataframe(original_df, augmented_df, file_path):
    """
    Validates that the augmented DataFrame is compatible with the original for merging.

    :param original_df: Original cleaned DataFrame to compare against
    :param augmented_df: Augmented DataFrame to validate
    :param file_path: File path string for error message context
    :return: True if augmented data is valid and compatible, False otherwise
    """

    if augmented_df is None or augmented_df.empty:  # Check if augmented DataFrame is None or contains no rows
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

    if len(numeric_overlap) < 2:  # Check if there are at least 2 overlapping numeric columns (features + target)
        print(
            f"{BackgroundColors.YELLOW}Warning: Augmented data for {BackgroundColors.CYAN}{file_path}{BackgroundColors.YELLOW} has insufficient numeric column overlap ({len(numeric_overlap)}). Skipping.{Style.RESET_ALL}"
        )  # Print warning about insufficient numeric overlap
        return False  # Return False due to insufficient numeric columns

    verbose_output(
        f"{BackgroundColors.GREEN}Augmented data validation passed for {BackgroundColors.CYAN}{file_path}{BackgroundColors.GREEN}: {len(augmented_df)} rows, {len(augmented_cols)} columns{Style.RESET_ALL}"
    )  # Output verbose message confirming validation success

    return True  # Return True indicating augmented data is valid and compatible


def sample_augmented_by_ratio(augmented_df, original_df, ratio):
    """
    Samples rows from the augmented DataFrame proportional to the original dataset size.

    :param augmented_df: Full augmented DataFrame to sample from
    :param original_df: Original DataFrame used to determine sample count
    :param ratio: Float ratio (e.g., 0.10 means 10% of original size)
    :return: Sampled DataFrame with at most ratio * len(original_df) rows, or None on failure
    """

    n_original = len(original_df)  # Get the number of rows in the original dataset
    n_requested = max(1, int(round(ratio * n_original)))  # Calculate requested sample size capped at minimum 1 row
    n_available = len(augmented_df)  # Get the total number of rows available in augmented data

    if n_available == 0:  # Check if augmented DataFrame has zero rows
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


def build_tsne_output_directory(original_file_path, augmented_file_path):
    """
    Build output directory path for t-SNE plots preserving nested dataset structure.

    :param original_file_path: Path to original dataset file
    :param augmented_file_path: Path to augmented dataset file
    :return: Path object for t-SNE output directory
    """

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


def combine_and_label_augmentation_data(original_df, augmented_df=None, label_col=None):
    """
    Combine original and augmented data with source labels for t-SNE visualization.

    :param original_df: DataFrame with original data
    :param augmented_df: DataFrame with augmented data (None for original-only)
    :param label_col: Name of the label/class column
    :return: Combined DataFrame with composite labels for visualization
    """

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


def prepare_numeric_features_for_tsne(df, exclude_col='tsne_label'):
    """
    Extract and prepare numeric features from DataFrame for t-SNE.

    :param df: DataFrame with mixed features
    :param exclude_col: Column name to exclude from numeric extraction
    :return: Tuple (numeric_array, labels_array, success_flag)
    """

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

    verbose_output(
        f"{BackgroundColors.GREEN}Computing t-SNE embedding and saving plot...{Style.RESET_ALL}"
    )  # Output verbose message for t-SNE computation

    try:  # Attempt t-SNE computation
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

        plt.savefig(output_path, dpi=300, bbox_inches='tight')  # Save figure with high resolution
        plt.close()  # Close figure to free memory

        print(
            f"{BackgroundColors.GREEN}t-SNE plot saved to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
        )  # Print success message

        return True  # Return success flag

    except Exception as e:  # t-SNE computation or plotting failed
        print(
            f"{BackgroundColors.RED}Error generating t-SNE plot: {e}{Style.RESET_ALL}"
        )  # Print error message
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

    verbose_output(
        f"{BackgroundColors.GREEN}Generating t-SNE visualization for augmentation experiment...{Style.RESET_ALL}"
    )  # Output verbose message for t-SNE generation

    augmented_file = find_data_augmentation_file(original_file)  # Locate augmented data file
    if augmented_file is None:  # No augmented file found
        print(
            f"{BackgroundColors.YELLOW}Warning: Cannot generate t-SNE - augmented file path not found.{Style.RESET_ALL}"
        )  # Print warning message
        return  # Exit function early

    tsne_output_dir = build_tsne_output_directory(original_file, augmented_file)  # Create output directory

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


def save_augmentation_comparison_results(file_path, comparison_results, config=None):
    """
    Save data augmentation comparison results to CSV file.

    :param file_path: Path to the original CSV file being processed
    :param comparison_results: List of dictionaries containing comparison metrics
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

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

    # Define column order for better readability
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

    # Reorder columns (only include columns that exist)
    existing_columns = [col for col in column_order if col in df.columns]  # Filter to existing columns
    df = df[existing_columns]  # Reorder DataFrame columns

    df = add_hardware_column(df, existing_columns)  # Add hardware specifications column

    df.to_csv(output_path, index=False)  # Save to CSV file
    print(
        f"{BackgroundColors.GREEN}Saved augmentation comparison results to {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
    )  # Output success message


def find_local_feature_file(file_dir, filename, config=None):
    """
    Attempt to locate <file_dir>/Feature_Analysis/<filename>.

    :param file_dir: Directory to search within
    :param filename: Filename to search for
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: The matching path or None
    """
    
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


def find_parent_feature_file(file_dir, filename, config=None):
    """
    Ascend parent directories searching for <parent>/Feature_Analysis/<filename>.

    :param file_dir: Directory to search within
    :param filename: Filename to search for
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: The matching path or None
    """
    
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
        df = pd.read_csv(ga_results_path, usecols=["best_features", "run_index"])  # Load only the necessary columns
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
        df = pd.read_csv(pca_results_path, usecols=["n_components", "cv_f1_score"])  # Load only the necessary columns
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


def extract_recursive_feature_elimination_features(file_path, config=None):
    """
    Extracts the "top_features" list (Python literal string) from the first row of the
    "RFE_Run_Results.csv" file located in the "Feature_Analysis" subdirectory
    relative to the input file's directory.

    :param file_path: Full path to the current CSV file being processed (e.g., "./Datasets/.../DrDoS_DNS.csv").
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: List of top features selected by RFE from the first run, or None if the file is not found or fails to load/parse.
    """
    
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
        df = pd.read_csv(rfe_runs_path, usecols=["top_features"])  # Load only the "top_features" column
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


def load_feature_selection_results(file_path, config=None):
    """
    Load GA, RFE and PCA feature selection artifacts for a given dataset file and
    print concise status messages.

    :param file_path: Path to the dataset CSV being processed.
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (ga_selected_features, pca_n_components, rfe_selected_features)
    """
    
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


def load_dataset(csv_path, config=None):
    """
    Load CSV and return DataFrame.

    :param csv_path: Path to CSV dataset.
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: DataFrame
    """
    
    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG

    verbose_output(
        f"\n{BackgroundColors.GREEN}Loading dataset from: {BackgroundColors.CYAN}{csv_path}{Style.RESET_ALL}",
        config=config
    )  # Output the loading dataset message

    if not verify_filepath_exists(csv_path):  # If the CSV file does not exist
        print(f"{BackgroundColors.RED}CSV file not found: {csv_path}{Style.RESET_ALL}")
        return None  # Return None

    df = pd.read_csv(csv_path, low_memory=False)  # Load the dataset

    df.columns = df.columns.str.strip()  # Clean column names by stripping leading/trailing whitespace

    if df.shape[1] < 2:  # If there are less than 2 columns
        print(f"{BackgroundColors.RED}CSV must have at least 1 feature and 1 target.{Style.RESET_ALL}")
        return None  # Return None

    return df  # Return the loaded DataFrame


def sanitize_feature_names(columns):
    r"""
    Sanitize column names by removing special JSON characters that LightGBM doesn't support.
    Replaces: { } [ ] : , " \ with underscores.

    :param columns: pandas Index or list of column names
    :return: list of sanitized column names
    """
    
    sanitized = []  # List to store sanitized column names
    
    for col in columns:  # Iterate over each column name
        clean_col = re.sub(r"[{}\[\]:,\"\\]", "_", str(col))  # Replace special characters with underscores
        clean_col = re.sub(r"_+", "_", clean_col)  # Replace multiple underscores with a single underscore
        clean_col = clean_col.strip("_")  # Remove leading/trailing underscores
        sanitized.append(clean_col)  # Add sanitized column name to the list
        
    return sanitized  # Return the list of sanitized column names


def preprocess_dataframe(df, remove_zero_variance=True, config=None):
    """
    Preprocess a DataFrame by removing rows with NaN or infinite values and
    dropping zero-variance numeric features.

    :param df: pandas DataFrame to preprocess
    :param remove_zero_variance: whether to drop numeric columns with zero variance
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: cleaned DataFrame
    """
    
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

    df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()  # Remove rows with NaN or infinite values

    if remove_zero_variance:  # If remove_zero_variance is set to True
        numeric_cols = df_clean.select_dtypes(include=["number"]).columns  # Select only numeric columns
        if len(numeric_cols) > 0:  # If there are numeric columns
            variances = df_clean[numeric_cols].var(axis=0, ddof=0)  # Calculate variances
            zero_var_cols = variances[variances == 0].index.tolist()  # Get columns with zero variance
            if zero_var_cols:  # If there are zero-variance columns
                df_clean = df_clean.drop(columns=zero_var_cols)  # Drop zero-variance columns

    return df_clean  # Return the cleaned DataFrame


def scale_and_split(X, y, test_size=0.2, random_state=42, config=None):
    """
    Scales the numeric features using StandardScaler and splits the data
    into training and testing sets.

    Note: The target variable 'y' is label-encoded before splitting.

    :param X: Features DataFrame (must contain numeric features).
    :param y: Target Series or array.
    :param test_size: Fraction of the data to reserve for the test set.
    :param random_state: Seed for the random split for reproducibility.
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    
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

    scaler = StandardScaler()  # Initialize the StandardScaler

    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the training features

    X_test_scaled = scaler.transform(X_test)  # Transform the testing features

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


def get_models(config=None):
    """
    Initializes and returns a dictionary of models to train.

    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary of model name and instance
    """
    
    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG

    verbose_output(
        f"{BackgroundColors.GREEN}Initializing models for training...{Style.RESET_ALL}",
        config=config
    )  # Output the verbose message
    
    # Get configuration values
    n_jobs = config.get("evaluation", {}).get("n_jobs", -1)  # Get n_jobs from config
    random_state = config.get("evaluation", {}).get("random_state", 42)  # Get random_state from config
    
    # Get model-specific parameters from config
    rf_params = config.get("models", {}).get("random_forest", {})  # Random Forest params
    svm_params = config.get("models", {}).get("svm", {})  # SVM params
    xgb_params = config.get("models", {}).get("xgboost", {})  # XGBoost params
    lr_params = config.get("models", {}).get("logistic_regression", {})  # Logistic Regression params
    knn_params = config.get("models", {}).get("knn", {})  # KNN params
    gb_params = config.get("models", {}).get("gradient_boosting", {})  # Gradient Boosting params
    lgb_params = config.get("models", {}).get("lightgbm", {})  # LightGBM params
    mlp_params = config.get("models", {}).get("mlp", {})  # MLP params

    return {  # Dictionary of models to train
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
            random_state=lr_params.get("random_state", random_state)
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
        df = pd.read_csv(hyperparams_path)  # Load the CSV into a DataFrame
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
    
    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG

    verbose_output(
        f"{BackgroundColors.GREEN}Starting to apply hyperparameters to models...{Style.RESET_ALL}",
        config=config
    )  # Inform user that application is starting

    if not hyperparams_map:  # Nothing to apply
        return models_map  # Return models unchanged

    def _normalize(name):  # Convert to alphanumeric lowercase
        return "".join(
            [c.lower() for c in str(name) if c.isalnum()]
        )  # Normalize model name by removing non-alphanumeric characters and converting to lowercase

    hp_keys = list(hyperparams_map.keys())  # List of provided model names
    hp_normalized = {k: _normalize(k) for k in hp_keys}  # Normalized lookup for matching

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
                    norm = _normalize(model_name)  # Compute normalized name
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


def load_pca_object(file_path, pca_n_components, config=None):
    """
    Loads a pre-fitted PCA object from a pickle file.

    :param file_path: Path to the dataset CSV file.
    :param pca_n_components: Number of PCA components to load.
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: PCA object if found, None otherwise.
    """
    
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

    if not verify_filepath_exists(pca_file):  # Check if the PCA file exists
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


def get_feature_subset(X_scaled, features, feature_names):
    """
    Returns a subset of features from the scaled feature set based on the provided feature names.
    Also returns the actual feature names that were successfully selected.

    :param X_scaled: Scaled features (numpy array).
    :param features: List of feature names to select.
    :param feature_names: List of all feature names corresponding to columns in X_scaled.
    :return: Tuple of (subset array, list of actual selected feature names)
    """

    if features:  # Only proceed if the list of selected features is NOT empty/None
        indices = []  # List to store indices of selected features
        selected_names = []  # List to store names of selected features
        for f in features:  # Iterate over each feature in the provided list
            if f in feature_names:  # Check if the feature exists in the full feature list
                indices.append(feature_names.index(f))  # Append the index of the feature
                selected_names.append(f)  # Append the name of the feature
        return X_scaled[:, indices], selected_names  # Return the subset and actual names
    else:  # If no features are selected (or features is None)
        return np.empty((X_scaled.shape[0], 0)), []  # Return empty array and empty list


def truncate_value(value):
    """
    Format a numeric value to 4 decimal places, or return None if not possible.
    
    :param value: Numeric value
    :return: Formatted string or None
    """

    try:  # Try to format the value
        if value is None:  # If value is None
            return None  # Return None
        v = float(value)  # Convert to float
        truncated = math.trunc(v * 10000) / 10000.0  # Truncate to 4 decimal places
        return f"{truncated:.4f}"  # Return formatted string
    except Exception:  # On failure
        return None  # Return None


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
    
    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG
    
    model_export_base = config.get("stacking", {}).get("model_export_base", "Feature_Analysis/Stacking/Models/")  # Get model export base from config
    
    def safe_filename(name):
        return re.sub(r'[\\/*?:"<>|]', "_", str(name))

    # Prefer dataset-local export directory when a CSV path is provided
    if dataset_csv_path:
        file_path_obj = Path(dataset_csv_path)
        export_dir = file_path_obj.parent / "Classifiers" / "Models"
    else:
        export_dir = Path(model_export_base) / safe_filename(dataset_name)
    os.makedirs(export_dir, exist_ok=True)
    param_str = "_".join(f"{k}-{v}" for k, v in sorted(best_params.items())) if best_params else ""
    param_str = safe_filename(param_str)[:64]
    features_str = safe_filename("_".join(feature_names))[:64] if feature_names else "all_features"
    fs = safe_filename(feature_set) if feature_set else "all"
    base_name = f"{safe_filename(model_name)}__{fs}__{features_str}__{param_str}"
    model_path = os.path.join(str(export_dir), f"{base_name}_model.joblib")
    scaler_path = os.path.join(str(export_dir), f"{base_name}_scaler.joblib")
    try:
        dump(model, model_path)
        if scaler is not None:
            dump(scaler, scaler_path)
        meta = {
            "model_name": model_name,
            "feature_set": feature_set,
            "features": feature_names,
            "params": best_params,
        }
        meta_path = os.path.join(str(export_dir), f"{base_name}_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        verbose_output(f"Exported model to {model_path} and scaler to {scaler_path}")
    except Exception as e:
        print(f"{BackgroundColors.YELLOW}Warning: Failed to export model {model_name}: {e}{Style.RESET_ALL}")


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

    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG
    
    skip_train_if_model_exists = config.get("execution", {}).get("skip_train_if_model_exists", False)  # Get skip train flag from config

    verbose_output(
        f"{BackgroundColors.GREEN}Training {BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN}...{Style.RESET_ALL}",
        config=config,
    )  # Output the verbose message

    start_time = time.time()  # Record the start time

    # If requested, attempt to load an existing exported model instead of retraining
    if dataset_file is not None and skip_train_if_model_exists:
        try:
            models_dir = Path(dataset_file).parent / "Classifiers" / "Models"
            if models_dir.exists():
                safe_model = re.sub(r'[\\/*?:"<>|]', "_", str(model_name))
                pattern = f"*{safe_model}*"
                if feature_set:
                    safe_fs = re.sub(r'[\\/*?:"<>|]', "_", str(feature_set))
                    pattern = f"*{safe_model}*{safe_fs}*"
                matches = list(models_dir.glob(f"{pattern}_model.joblib"))
                if matches:
                    try:
                        loaded = load(str(matches[0]))
                        model = loaded
                        # Try load scaler with same base name
                        scaler_path = str(matches[0]).replace("_model.joblib", "_scaler.joblib")
                        if os.path.exists(scaler_path):
                            scaler = load(scaler_path)
                        verbose_output(f"Loaded existing model from {matches[0]}")
                        # Compute predictions and metrics without retraining
                        y_pred = model.predict(X_test)
                        elapsed_time = 0.0
                        acc = accuracy_score(y_test, y_pred)
                        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
                        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
                        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
                        if len(np.unique(y_test)) == 2:
                            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
                            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
                        else:
                            fpr = 0.0
                            fnr = 0.0
                        return (acc, prec, rec, f1, fpr, fnr, elapsed_time)
                    except Exception:
                        # Fallback to training if loading fails
                        verbose_output(f"Failed to load existing model for {model_name}; retraining.")
        except Exception:
            pass

    model.fit(X_train, y_train)  # Fit the model on the training data

    y_pred = model.predict(X_test)  # Predict the labels for the test set

    elapsed_time = time.time() - start_time  # Calculate the total time elapsed

    acc = accuracy_score(y_test, y_pred)  # Calculate Accuracy
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate Precision
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate Recall
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate F1-Score

    if len(np.unique(y_test)) == 2:  # Binary classification
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()  # Get confusion matrix components
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # Calculate FPR
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # Calculate FNR
    else:  # Multi-class
        fpr = 0.0  # Placeholder
        fnr = 0.0  # Placeholder

    verbose_output(
        f"{BackgroundColors.GREEN}{model_name} Accuracy: {BackgroundColors.CYAN}{truncate_value(acc)}{BackgroundColors.GREEN}, Time: {BackgroundColors.CYAN}{int(round(elapsed_time))}s{Style.RESET_ALL}"
    )  # Output result

    # Export trained model and scaler if dataset info is available
    try:
        if dataset_file is not None:
            dataset_name = os.path.basename(os.path.dirname(dataset_file))
            export_model_and_scaler(model, scaler, dataset_name, model_name, feature_names or [], best_params=None, feature_set=feature_set, dataset_csv_path=dataset_file)
    except Exception:
        pass

    return (acc, prec, rec, f1, fpr, fnr, int(round(elapsed_time)))  # Return the metrics tuple


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

    verbose_output(
        f"{BackgroundColors.GREEN}Starting training and evaluation of Stacking Classifier...{Style.RESET_ALL}"
    )  # Output the verbose message

    start_time = time.time()  # Record the start time for timing training and prediction

    model.fit(X_train, y_train)  # Fit the stacking model on the training data (accepts DataFrame or array)

    y_pred = model.predict(X_test)  # Predict the labels for the test set

    elapsed_time = time.time() - start_time  # Calculate the total time elapsed

    acc = accuracy_score(y_test, y_pred)  # Calculate Accuracy
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate Precision (weighted)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate Recall (weighted)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate F1-Score (weighted)

    if len(np.unique(y_test)) == 2:  # Verify if it's a binary classification problem
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()  # Get Confusion Matrix components
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # Calculate False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # Calculate False Negative Rate
    else:  # For multi-class (simplified approach, actual implementation is complex)
        fpr = 0.0  # Placeholder
        fnr = 0.0  # Placeholder
        print(
            f"{BackgroundColors.YELLOW}Warning: Multi-class FPR/FNR calculation simplified to 0.0.{Style.RESET_ALL}"
        )  # Warning about simplification

    verbose_output(
        f"{BackgroundColors.GREEN}Evaluation complete. Accuracy: {BackgroundColors.CYAN}{truncate_value(acc)}{BackgroundColors.GREEN}, Time: {BackgroundColors.CYAN}{int(round(elapsed_time))}s{Style.RESET_ALL}"
    )  # Output the final result summary

    return (acc, prec, rec, f1, fpr, fnr, int(round(elapsed_time)))  # Return the metrics tuple


def get_hardware_specifications():
    """
    Returns system specs: real CPU model (Windows/Linux/macOS), physical cores,
    RAM in GB, and OS name/version.

    :return: Dictionary with keys: cpu_model, cores, ram_gb, os
    """

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


def add_hardware_column(df, columns_order, column_name="Hardware"):
    """
    Ensure a hardware description column named by `column_name` exists on `df`
    and insert it into `columns_order` immediately after `elapsed_time_s`.

    :param df: pandas.DataFrame to mutate in-place
    :param columns_order: list of canonical column names to update
    :param column_name: hardware column name to add (default: "Hardware")
    :return: None
    """

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


def save_stacking_results(csv_path, results_list, config=None):
    """Save the stacking results to CSV file located in same dataset Feature_Analysis directory.

    Writes richer metadata fields matching RFE outputs: model, dataset, accuracy, precision,
    recall, f1_score, fpr, fnr, elapsed_time_s, cv_method, top_features, rfe_ranking,
    hyperparameters, features_list and Hardware.
    """
    
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
    stacking_dir = feature_analysis_dir / "Stacking"
    os.makedirs(stacking_dir, exist_ok=True)
    output_path = stacking_dir / results_filename

    flat_rows = []
    for res in results_list:
        row = dict(res)

        # Truncate metrics to 4 decimal places
        for metric in ["accuracy", "precision", "recall", "f1_score", "fpr", "fnr"]:
            if metric in row and row[metric] is not None:
                row[metric] = truncate_value(row[metric])

        # Serialize list-like fields into JSON strings for CSV stability
        if "features_list" in row and not isinstance(row["features_list"], str):
            row["features_list"] = json.dumps(row["features_list"])
        if "top_features" in row and not isinstance(row["top_features"], str):
            row["top_features"] = json.dumps(row["top_features"])
        if "rfe_ranking" in row and row["rfe_ranking"] is not None and not isinstance(
            row["rfe_ranking"], str
        ):
            row["rfe_ranking"] = json.dumps(row["rfe_ranking"])
        if "hyperparameters" in row and row["hyperparameters"] is not None and not isinstance(
            row["hyperparameters"], str
        ):
            row["hyperparameters"] = json.dumps(row["hyperparameters"])

        flat_rows.append(row)

    df = pd.DataFrame(flat_rows)

    # Use the canonical header constant for results CSV column ordering
    results_csv_columns = config.get("stacking", {}).get("results_csv_columns", [])  # Get columns from config
    column_order = list(results_csv_columns) if results_csv_columns else list(config.get("stacking", {}).get("results_csv_columns", []))  # Use config or fallback to global

    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns + [c for c in df.columns if c not in existing_columns]]

    df = add_hardware_column(df, existing_columns)

    try:
        df.to_csv(str(output_path), index=False, encoding="utf-8")
        print(
            f"\n{BackgroundColors.GREEN}Stacking classifier results successfully saved to {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
        )
    except Exception as e:
        print(
            f"{BackgroundColors.RED}Failed to write Stacking Classifier CSV to {BackgroundColors.CYAN}{output_path}{BackgroundColors.RED}: {e}{Style.RESET_ALL}"
        )


def get_cache_file_path(csv_path, config=None):
    """
    Generate the cache file path for a given dataset CSV path.

    :param csv_path: Path to the dataset CSV file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Path to the cache file
    """
    
    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG

    verbose_output(
        f"{BackgroundColors.GREEN}Generating cache file path for: {BackgroundColors.CYAN}{csv_path}{Style.RESET_ALL}",
        config=config
    )  # Output the verbose message

    cache_prefix = config.get("stacking", {}).get("cache_prefix", "CACHE_")  # Get cache prefix from config
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]  # Get base dataset name
    output_dir = f"{os.path.dirname(csv_path)}/Classifiers"  # Directory relative to the dataset
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    cache_filename = f"{cache_prefix}{dataset_name}-Stacking_Classifiers_Results.csv"  # Cache filename
    cache_path = os.path.join(output_dir, cache_filename)  # Full cache file path

    return cache_path  # Return the cache file path


def load_cache_results(csv_path, config=None):
    """
    Load cached results from the cache file if it exists.

    :param csv_path: Path to the dataset CSV file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary mapping (feature_set, model_name) to result entry
    """
    
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
        df_cache = pd.read_csv(cache_path)  # Read the cache file
        df_cache.columns = df_cache.columns.str.strip()  # Remove leading/trailing whitespace from column names
        cache_dict = {}  # Initialize cache dictionary

        for _, row in df_cache.iterrows():  # Iterate through each row
            feature_set = row.get("feature_set", "")  # Get feature set name
            model_name = row.get("model_name", "")  # Get model name
            cache_key = (feature_set, model_name)  # Create cache key tuple

            def _safe_load_json(val):
                if pd.isna(val):
                    return None
                if isinstance(val, str):
                    try:
                        return json.loads(val)
                    except Exception:
                        return val
                return val

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
                "top_features": _safe_load_json(row.get("top_features", None)),
                "rfe_ranking": _safe_load_json(row.get("rfe_ranking", None)),
                "hyperparameters": _safe_load_json(row.get("hyperparameters", None)),
                "features_list": _safe_load_json(row.get("features_list", None)),
                "Hardware": row.get("Hardware", None),
            }

            cache_dict[cache_key] = result_entry

        print(f"{BackgroundColors.GREEN}Loaded cached results from: {BackgroundColors.CYAN}{cache_path}{Style.RESET_ALL}")
        return cache_dict

    except Exception as e:  # Catch any errors
        print(
            f"{BackgroundColors.YELLOW}Warning: Failed to save to cache {BackgroundColors.CYAN}{cache_path}{BackgroundColors.YELLOW}: {e}{Style.RESET_ALL}"
        )  # Print warning message


def remove_cache_file(csv_path, config=None):
    """
    Remove the cache file after successful completion.

    :param csv_path: Path to the dataset CSV file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
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


def get_automl_search_spaces():
    """
    Returns hyperparameter search space definitions for all AutoML candidate models.

    :param: None
    :return: Dictionary mapping model names to their search space configurations
    """

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


def suggest_hyperparameters_for_model(trial, model_name, search_spaces):
    """
    Suggests hyperparameters for a given model using an Optuna trial.

    :param trial: Optuna trial object for parameter suggestion
    :param model_name: Name of the model to suggest hyperparameters for
    :param search_spaces: Dictionary of search space definitions
    :return: Dictionary of suggested hyperparameters
    """

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


def create_model_from_params(model_name, params, config=None):
    """
    Creates a classifier instance from a model name and hyperparameters dictionary.

    :param model_name: Name of the classifier to instantiate
    :param params: Dictionary of hyperparameters to apply
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Instantiated classifier object
    """

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
        return LogisticRegression(random_state=automl_random_state, **clean_params)  # Create LR instance
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

        model.fit(X_fold_train, y_fold_train)  # Fit model on fold training data
        y_pred = model.predict(X_fold_val)  # Predict on fold validation data
        fold_f1 = f1_score(y_fold_val, y_pred, average="weighted", zero_division=0)  # Calculate fold F1
        f1_scores.append(fold_f1)  # Append fold F1 score

        if trial is not None:  # If Optuna trial is provided
            trial.report(np.mean(f1_scores), fold_idx)  # Report intermediate value for pruning
            if trial.should_prune():  # Check if trial should be pruned
                raise optuna.exceptions.TrialPruned()  # Prune this trial

    return float(np.mean(f1_scores))  # Return mean F1 score across folds as Python float


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


def run_automl_model_search(X_train, y_train, file_path, config=None):
    """
    Runs Optuna-based AutoML model search to find optimal classifier and hyperparameters.

    :param X_train: Scaled training features (numpy array)
    :param y_train: Training target labels (numpy array)
    :param file_path: Path to the dataset file for logging
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (best_model_name, best_params, study) or (None, None, None) on failure
    """

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
            meta_model = LogisticRegression(max_iter=1000, random_state=automl_random_state)  # Create LR meta-learner
        elif meta_learner_name == "Random Forest":  # Random Forest meta-learner
            meta_model = RandomForestClassifier(n_estimators=50, random_state=automl_random_state, n_jobs=n_jobs)  # Create RF meta-learner
        else:  # Gradient Boosting meta-learner
            meta_model = GradientBoostingClassifier(random_state=automl_random_state)  # Create GB meta-learner

        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_model,
            cv=StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=automl_random_state),
            n_jobs=n_jobs,
        )  # Create stacking classifier

        mean_f1 = automl_cross_validate_model(stacking, X_train, y_train, cv_folds, trial)  # Cross-validate stacking
        return mean_f1  # Return mean F1 score

    except optuna.exceptions.TrialPruned:  # Handle Optuna pruning
        raise  # Re-raise pruning exception
    except Exception as e:  # Handle other errors gracefully
        verbose_output(
            f"{BackgroundColors.YELLOW}AutoML stacking trial failed: {e}{Style.RESET_ALL}"
        )  # Log the failure
        return 0.0  # Return zero for failed trials


def extract_top_automl_models(study, top_n=5):
    """
    Extracts the top N unique models from an AutoML study based on F1 score.

    :param study: Completed Optuna study object
    :param top_n: Number of top models to extract
    :return: Dictionary mapping model names to their best parameters
    """

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


def run_automl_stacking_search(X_train, y_train, model_study, file_path):
    """
    Runs Optuna-based optimization to find the best stacking ensemble configuration.

    :param X_train: Scaled training features (numpy array)
    :param y_train: Training target labels (numpy array)
    :param model_study: Completed Optuna study from model search
    :param file_path: Path to the dataset file for logging
    :return: Tuple (best_stacking_config, stacking_study) or (None, None) on failure
    """

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


def build_automl_stacking_model(best_config):
    """
    Builds a StackingClassifier from the best AutoML stacking configuration.

    :param best_config: Dictionary with best stacking configuration
    :return: Configured StackingClassifier instance
    """

    estimators = []  # Initialize estimators list

    for name in best_config["base_learners"]:  # Iterate over selected base learners
        params = best_config["base_learner_params"].get(name, {})  # Get model parameters
        model = create_model_from_params(name, params)  # Create model instance
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")  # Sanitize estimator name
        estimators.append((safe_name, model))  # Add to estimators list

    meta_learner_name = best_config["meta_learner"]  # Get meta-learner name

    if meta_learner_name == "Logistic Regression":  # Logistic Regression meta-learner
        meta_model = LogisticRegression(max_iter=1000, random_state=config.get("automl", {}).get("random_state", 42))  # Create LR
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
        n_jobs=config.get("evaluation", {}).get("n_jobs", -1),
    )  # Create stacking classifier with optimal configuration

    return stacking_model  # Return configured stacking model


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

    start_time = time.time()  # Record start time

    model.fit(X_train, y_train)  # Train model on full training set
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


def export_automl_search_history(study, output_dir, study_name):
    """
    Exports the Optuna study trial history to a CSV file.

    :param study: Completed Optuna study object
    :param output_dir: Directory path for saving the export file
    :param study_name: Name prefix for the output file
    :return: Path to the exported CSV file
    """

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
    df.to_csv(output_path, index=False)  # Save to CSV

    print(
        f"{BackgroundColors.GREEN}AutoML search history exported to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
    )  # Output export confirmation

    return output_path  # Return the output file path


def export_automl_best_config(best_model_name, best_params, test_metrics, stacking_config, output_dir, feature_names):
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

    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    config = {  # Build configuration export dictionary
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
            "n_trials": config.get("automl", {}).get("n_trials", 50),  # Number of model search trials
            "stacking_trials": config.get("automl", {}).get("stacking_trials", 20),  # Number of stacking search trials
            "cv_folds": config.get("automl", {}).get("cv_folds", 5),  # Cross-validation folds
            "timeout_s": config.get("automl", {}).get("timeout", 3600),  # Timeout in seconds
            "random_state": config.get("automl", {}).get("random_state", 42),  # Random seed used
        },
        "feature_names": feature_names,  # Features used in training
        "n_features": len(feature_names),  # Number of features
    }  # Build complete config dictionary

    output_path = os.path.join(output_dir, config.get("automl", {}).get("results_filename", "AutoML_Results.csv").replace(".csv", "_best_config.json"))  # Build output path

    with open(output_path, "w", encoding="utf-8") as f:  # Open file for writing
        json.dump(config, f, indent=2, default=str)  # Write JSON with indentation

    print(
        f"{BackgroundColors.GREEN}AutoML best configuration exported to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
    )  # Output export confirmation

    return output_path  # Return the output file path


def export_automl_best_model(model, scaler, output_dir, model_name, feature_names):
    """
    Exports the best AutoML model and scaler to disk using joblib.

    :param model: Trained best model instance
    :param scaler: Fitted StandardScaler instance
    :param output_dir: Directory path for saving model files
    :param model_name: Name of the model for file naming
    :param feature_names: List of feature names for metadata
    :return: Tuple (model_path, scaler_path)
    """

    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    safe_name = re.sub(r'[\\/*?:"<>|() ]', "_", str(model_name))  # Sanitize model name for filename
    model_path = os.path.join(output_dir, f"AutoML_best_{safe_name}_model.joblib")  # Build model file path
    scaler_path = os.path.join(output_dir, f"AutoML_best_{safe_name}_scaler.joblib")  # Build scaler file path

    dump(model, model_path)  # Export model to disk

    if scaler is not None:  # If scaler is provided
        dump(scaler, scaler_path)  # Export scaler to disk

    meta_path = os.path.join(output_dir, f"AutoML_best_{safe_name}_meta.json")  # Build metadata file path
    meta = {  # Build metadata dictionary
        "model_name": model_name,  # Model name
        "features": feature_names,  # Feature names used
        "n_features": len(feature_names),  # Number of features
    }  # Metadata content

    with open(meta_path, "w", encoding="utf-8") as f:  # Open metadata file
        json.dump(meta, f, indent=2)  # Write metadata JSON

    print(
        f"{BackgroundColors.GREEN}AutoML best model exported to: {BackgroundColors.CYAN}{model_path}{Style.RESET_ALL}"
    )  # Output export confirmation

    return (model_path, scaler_path)  # Return file paths


def build_automl_results_list(best_model_name, best_params, individual_metrics, stacking_metrics, stacking_config, file_path, feature_names, n_train, n_test):
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
    :return: List of result dictionaries for CSV export
    """

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


def save_automl_results(csv_path, results_list):
    """
    Saves AutoML results to a dedicated CSV file in the Feature_Analysis/AutoML directory.

    :param csv_path: Path to the original dataset CSV file
    :param results_list: List of result dictionaries to save
    :return: None
    """

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

    df.to_csv(str(output_path), index=False, encoding="utf-8")  # Save to CSV file

    print(
        f"\n{BackgroundColors.GREEN}AutoML results saved to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
    )  # Output save confirmation


def run_automl_pipeline(file, df, feature_names, data_source_label="Original"):
    """
    Runs the complete AutoML pipeline: model search, stacking optimization, evaluation, and export.

    :param file: Path to the dataset file being processed
    :param df: Preprocessed DataFrame with features and target
    :param feature_names: List of feature column names
    :param data_source_label: Label identifying the data source
    :return: Dictionary containing AutoML results, or None on failure
    """

    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*80}{Style.RESET_ALL}"
    )  # Print separator
    print(
        f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}AutoML Pipeline - {BackgroundColors.CYAN}{os.path.basename(file)}{Style.RESET_ALL}"
    )  # Print pipeline header
    print(
        f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*80}{Style.RESET_ALL}\n"
    )  # Print separator

    automl_start = time.time()  # Record pipeline start time

    X_full = df.select_dtypes(include=np.number).iloc[:, :-1]  # Extract numeric features
    y = df.iloc[:, -1]  # Extract target column

    if len(np.unique(y)) < 2:  # Check for at least 2 classes
        print(
            f"{BackgroundColors.RED}AutoML: Target has only one class. Skipping.{Style.RESET_ALL}"
        )  # Output error
        return None  # Return None

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = scale_and_split(X_full, y)  # Scale and split data

    y_train_arr = np.asarray(y_train)  # Convert training target to numpy array
    y_test_arr = np.asarray(y_test)  # Convert test target to numpy array

    send_telegram_message(TELEGRAM_BOT, f"Starting AutoML pipeline for {os.path.basename(file)}")  # Notify via Telegram

    best_model_name, best_params, model_study = run_automl_model_search(
        X_train_scaled, y_train_arr, file
    )  # Phase 1: Run model search

    if best_model_name is None:  # If model search failed
        print(
            f"{BackgroundColors.RED}AutoML pipeline aborted: model search failed.{Style.RESET_ALL}"
        )  # Output failure message
        return None  # Return None

    best_individual_model = create_model_from_params(best_model_name, best_params)  # Create best individual model

    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Evaluating AutoML best individual model on test set...{Style.RESET_ALL}"
    )  # Output evaluation message

    individual_metrics = evaluate_automl_model_on_test(
        best_individual_model, best_model_name, X_train_scaled, y_train_arr, X_test_scaled, y_test_arr
    )  # Evaluate best individual model on test set

    stacking_config = None  # Initialize stacking config as None
    stacking_metrics = None  # Initialize stacking metrics as None
    stacking_study = None  # Initialize stacking study as None

    if config.get("automl", {}).get("stacking_trials", 20) > 0:  # If stacking search is enabled
        stacking_config, stacking_study = run_automl_stacking_search(
            X_train_scaled, y_train_arr, model_study, file
        )  # Phase 2: Run stacking search

        if stacking_config is not None:  # If stacking search succeeded
            best_stacking_model = build_automl_stacking_model(stacking_config)  # Build best stacking model

            print(
                f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Evaluating AutoML best stacking model on test set...{Style.RESET_ALL}"
            )  # Output evaluation message

            stacking_metrics = evaluate_automl_model_on_test(
                best_stacking_model, "AutoML_Stacking", X_train_scaled, y_train_arr, X_test_scaled, y_test_arr
            )  # Evaluate stacking model on test set

    file_path_obj = Path(file)  # Create Path object for file
    automl_output_dir = str(file_path_obj.parent / "Feature_Analysis" / "AutoML")  # Build AutoML output directory

    export_automl_search_history(model_study, automl_output_dir, "model_search")  # Export model search history

    if stacking_study is not None:  # If stacking study exists
        export_automl_search_history(stacking_study, automl_output_dir, "stacking_search")  # Export stacking search history

    export_automl_best_config(
        best_model_name, best_params, individual_metrics, stacking_config, automl_output_dir, feature_names
    )  # Export best configuration

    export_automl_best_model(
        best_individual_model, scaler, automl_output_dir, best_model_name, feature_names
    )  # Export best individual model

    if stacking_config is not None and stacking_metrics is not None:  # If stacking was successful
        best_stacking_model_final = build_automl_stacking_model(stacking_config)  # Rebuild stacking model for export
        best_stacking_model_final.fit(X_train_scaled, y_train_arr)  # Fit stacking model on full training data
        export_automl_best_model(
            best_stacking_model_final, scaler, automl_output_dir, "AutoML_Stacking", feature_names
        )  # Export best stacking model

    results_list = build_automl_results_list(
        best_model_name, best_params, individual_metrics, stacking_metrics, stacking_config,
        file, feature_names, len(y_train), len(y_test)
    )  # Build results list for CSV

    save_automl_results(file, results_list)  # Save AutoML results to CSV

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
):
    """
    Evaluate classifiers on a single dataset (original or augmented).

    :param file: Path to the dataset file
    :param df: DataFrame with the dataset
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
    :return: Dictionary mapping (feature_set, model_name) to results
    """

    # Sanitize GA and RFE feature names to match the sanitized feature_names in the DataFrame
    if ga_selected_features:
        ga_selected_features = sanitize_feature_names(ga_selected_features)
    if rfe_selected_features:
        rfe_selected_features = sanitize_feature_names(rfe_selected_features)

    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*80}{Style.RESET_ALL}"
    )
    print(
        f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Evaluating on: {BackgroundColors.CYAN}{data_source_label} Data{Style.RESET_ALL}"
    )
    print(
        f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*80}{Style.RESET_ALL}\n"
    )

    X_full = df.select_dtypes(include=np.number).iloc[:, :-1]  # Features (numeric only)
    y = df.iloc[:, -1]  # Target

    if len(np.unique(y)) < 2:  # Verify if there is more than one class
        print(
            f"{BackgroundColors.RED}Target column has only one class. Cannot perform classification. Skipping.{Style.RESET_ALL}"
        )  # Output the error message
        return {}  # Return empty dictionary

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = scale_and_split(
        X_full, y
    )  # Scale and split the data

    estimators = [
        (name, model) for name, model in base_models.items() if name != "SVM"
    ]  # Define estimators (excluding SVM)

    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=config.get("evaluation", {}).get("n_jobs", -1)),
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        n_jobs=config.get("evaluation", {}).get("n_jobs", -1),
    )  # Define the Stacking Classifier model

    X_train_pca, X_test_pca = apply_pca_transformation(
        X_train_scaled, X_test_scaled, pca_n_components, file
    )  # Apply PCA transformation if applicable

    # Get feature subsets with actual selected feature names
    X_train_ga, ga_actual_features = get_feature_subset(X_train_scaled, ga_selected_features, feature_names)
    X_test_ga, _ = get_feature_subset(X_test_scaled, ga_selected_features, feature_names)
    
    X_train_rfe, rfe_actual_features = get_feature_subset(X_train_scaled, rfe_selected_features, feature_names)
    X_test_rfe, _ = get_feature_subset(X_test_scaled, rfe_selected_features, feature_names)

    feature_sets = {  # Dictionary of feature sets to evaluate
        "Full Features": (X_train_scaled, X_test_scaled, feature_names),  # All features with names
        "GA Features": (X_train_ga, X_test_ga, ga_actual_features),  # GA subset with actual names
        "PCA Components": (
            (X_train_pca, X_test_pca, None) if X_train_pca is not None else None
        ),  # PCA components (only if PCA was applied)
        "RFE Features": (X_train_rfe, X_test_rfe, rfe_actual_features),  # RFE subset with actual names
    }

    feature_sets = {
        k: v for k, v in feature_sets.items() if v is not None
    }  # Remove any None entries (e.g., PCA if not applied)
    feature_sets = dict(sorted(feature_sets.items()))  # Sort the feature sets by name

    individual_models = {
        k: v for k, v in base_models.items()
    }  # Use the base models (with hyperparameters applied) for individual evaluation
    total_steps = len(feature_sets) * (
        len(individual_models) + 1
    )  # Total steps: models + stacking per feature set
    progress_bar = tqdm(total=total_steps, desc=f"{data_source_label} Data", file=sys.stdout)  # Progress bar for all evaluations

    all_results = {}  # Dictionary to store results: (feature_set, model_name) -> result_entry

    current_combination = 1  # Counter for combination index

    for idx, (name, (X_train_subset, X_test_subset, subset_feature_names_list)) in enumerate(feature_sets.items(), start=1):
        if X_train_subset.shape[1] == 0:  # Verify if the subset is empty
            print(
                f"{BackgroundColors.YELLOW}Warning: Skipping {name}. No features selected.{Style.RESET_ALL}"
            )  # Output warning
            progress_bar.update(len(individual_models) + 1)  # Skip all steps for this feature set
            continue  # Skip to the next set

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Evaluating models on: {BackgroundColors.CYAN}{name} ({X_train_subset.shape[1]} features){Style.RESET_ALL}"
        )  # Output evaluation status

        if name == "PCA Components":  # If the feature set is PCA Components
            subset_feature_names = [
                f"PC{i+1}" for i in range(X_train_subset.shape[1])
            ]  # Generate PCA component names
        else:  # For other feature sets
            subset_feature_names = (
                subset_feature_names_list if subset_feature_names_list else [f"feature_{i}" for i in range(X_train_subset.shape[1])]
            )  # Use actual feature names or generate generic ones

        X_train_df = pd.DataFrame(
            X_train_subset, columns=subset_feature_names
        )  # Convert training features to DataFrame
        X_test_df = pd.DataFrame(
            X_test_subset, columns=subset_feature_names
        )  # Convert test features to DataFrame

        progress_bar.set_description(
            f"{data_source_label} - {name} (Individual)"
        )  # Update progress bar description
        
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=config.get("evaluation", {}).get("threads_limit", 2)
        ) as executor:  # Create a thread pool executor for parallel evaluation
            future_to_model = {}  # Dictionary to map futures to model names
            for model_name, model in individual_models.items():  # Iterate over each individual model
                send_telegram_message(TELEGRAM_BOT, f"Starting combination {current_combination}/{total_steps}: {name} - {model_name}")
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
                )  # Submit evaluation task to thread pool (using .values for numpy arrays)
                # Store both the model name and its class name for richer metadata
                future_to_model[future] = (model_name, model.__class__.__name__, current_combination)
                current_combination += 1
            
            for future in concurrent.futures.as_completed(future_to_model):  # As each evaluation completes
                model_name, model_class, comb_idx = future_to_model[future]  # Get metadata from mapping
                metrics = future.result()  # Get the metrics from the completed future
                # Flatten metrics into named fields and include extra metadata similar to rfe.py
                acc, prec, rec, f1, fpr, fnr, elapsed = metrics
                result_entry = {
                    "model": model_class,
                    "dataset": os.path.relpath(file),
                    "feature_set": name,
                    "classifier_type": "Individual",
                    "model_name": model_name,
                    "data_source": data_source_label,
                    "experiment_id": experiment_id,
                    "experiment_mode": experiment_mode,
                    "augmentation_ratio": augmentation_ratio,
                    "n_features": X_train_subset.shape[1],
                    "n_samples_train": len(y_train),
                    "n_samples_test": len(y_test),
                    "accuracy": truncate_value(acc),
                    "precision": truncate_value(prec),
                    "recall": truncate_value(rec),
                    "f1_score": truncate_value(f1),
                    "fpr": truncate_value(fpr),
                    "fnr": truncate_value(fnr),
                    "elapsed_time_s": int(round(elapsed)),
                    "cv_method": f"StratifiedKFold(n_splits=10)",
                    "top_features": json.dumps(subset_feature_names),
                    "rfe_ranking": None,
                    "hyperparameters": json.dumps(hyperparams_map.get(model_name)) if hyperparams_map and hyperparams_map.get(model_name) is not None else None,
                    "features_list": subset_feature_names,
                }  # Prepare result entry
                all_results[(name, model_name)] = result_entry  # Store result with key
                send_telegram_message(TELEGRAM_BOT, f"Finished combination {comb_idx}/{total_steps}: {name} - {model_name} with F1: {truncate_value(f1)} in {calculate_execution_time(0, elapsed)}")
                print(
                    f"    {BackgroundColors.GREEN}{model_name} Accuracy: {BackgroundColors.CYAN}{truncate_value(metrics[0])}{Style.RESET_ALL}"
                )  # Output accuracy
                progress_bar.update(1)  # Update progress after each model

        print(
            f"  {BackgroundColors.GREEN}Training {BackgroundColors.CYAN}Stacking Classifier{BackgroundColors.GREEN}...{Style.RESET_ALL}"
        )
        progress_bar.set_description(
            f"{data_source_label} - {name} (Stacking)"
        )  # Update progress bar description for stacking

        send_telegram_message(TELEGRAM_BOT, f"Starting combination {current_combination}/{total_steps}: {name} - StackingClassifier")

        stacking_metrics = evaluate_stacking_classifier(
            stacking_model, X_train_df, y_train, X_test_df, y_test
        )  # Evaluate stacking model with DataFrames

        # Export stacking model and scaler
        try:
            dataset_name = os.path.basename(os.path.dirname(file))
            export_model_and_scaler(stacking_model, scaler, dataset_name, "StackingClassifier", subset_feature_names, best_params=None, feature_set=name, dataset_csv_path=file)
        except Exception:
            pass

        # Flatten stacking metrics and include richer metadata
        s_acc, s_prec, s_rec, s_f1, s_fpr, s_fnr, s_elapsed = stacking_metrics
        stacking_result_entry = {
            "model": stacking_model.__class__.__name__,
            "dataset": os.path.relpath(file),
            "feature_set": name,
            "classifier_type": "Stacking",
            "model_name": "StackingClassifier",
            "data_source": data_source_label,
            "experiment_id": experiment_id,
            "experiment_mode": experiment_mode,
            "augmentation_ratio": augmentation_ratio,
            "n_features": X_train_subset.shape[1],
            "n_samples_train": len(y_train),
            "n_samples_test": len(y_test),
            "accuracy": truncate_value(s_acc),
            "precision": truncate_value(s_prec),
            "recall": truncate_value(s_rec),
            "f1_score": truncate_value(s_f1),
            "fpr": truncate_value(s_fpr),
            "fnr": truncate_value(s_fnr),
            "elapsed_time_s": int(round(s_elapsed)),
            "cv_method": f"StratifiedKFold(n_splits=10)",
            "top_features": json.dumps(subset_feature_names),
            "rfe_ranking": None,
            "hyperparameters": None,
            "features_list": subset_feature_names,
        }  # Prepare stacking result entry
        all_results[(name, "StackingClassifier")] = stacking_result_entry  # Store result with key
        send_telegram_message(TELEGRAM_BOT, f"Finished combination {current_combination}/{total_steps}: {name} - StackingClassifier with F1: {truncate_value(s_f1)} in {calculate_execution_time(0, s_elapsed)}")
        print(
            f"    {BackgroundColors.GREEN}Stacking Accuracy: {BackgroundColors.CYAN}{truncate_value(stacking_metrics[0])}{Style.RESET_ALL}"
        )  # Output accuracy
        progress_bar.update(1)  # Update progress after stacking
        current_combination += 1

    progress_bar.close()  # Close progress bar
    return all_results  # Return dictionary of results


def determine_files_to_process(csv_file, input_path, config=None):
    """
    Determines which files to process based on CLI override or directory scan.

    :param csv_file: Optional CSV file path from CLI argument
    :param input_path: Directory path to search for CSV files
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: List of file paths to process
    """
    
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
    else:  # No CLI override, scan directory for CSV files
        return get_files_to_process(input_path, file_extension=".csv", config=config)  # Get list of CSV files to process


def combine_dataset_if_needed(files_to_process, config=None):
    """
    Combines multiple dataset files into one if PROCESS_ENTIRE_DATASET is enabled.

    :param files_to_process: List of file paths to process
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (combined_df, combined_file_for_features, updated_files_list) or (None, None, files_to_process)
    """
    
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


def load_and_preprocess_dataset(file, combined_df, config=None):
    """
    Loads and preprocesses a dataset file or uses combined dataframe.

    :param file: File path to load or "combined" keyword
    :param combined_df: Pre-combined dataframe (used if file == "combined")
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (df_cleaned, feature_names) or (None, None) if loading/preprocessing fails
    """
    
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


def prepare_models_with_hyperparameters(file_path, config=None):
    """
    Prepares base models and applies hyperparameter optimization results if available.

    :param file_path: Path to the dataset file for loading hyperparameters
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (base_models, hp_params_map)
    """
    
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


def extract_metrics_from_result(result):
    """
    Extracts metrics from a result dictionary into a list.

    :param result: Result dictionary containing metric keys
    :return: List of [accuracy, precision, recall, f1_score, fpr, fnr, elapsed_time_s]
    """

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


def calculate_all_improvements(orig_metrics, merged_metrics):
    """
    Calculates improvement percentages for all metrics comparing original vs merged data.

    :param orig_metrics: List of original metrics [accuracy, precision, recall, f1, fpr, fnr, time]
    :param merged_metrics: List of merged metrics [accuracy, precision, recall, f1, fpr, fnr, time]
    :return: Dictionary of improvement percentages for each metric
    """

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


def generate_ratio_comparison_report(results_original, all_ratio_results):
    """
    Generates and prints comparison report for ratio-based data augmentation evaluation.
    Compares the original baseline against each augmentation ratio experiment.

    :param results_original: Dictionary of results from original data evaluation
    :param all_ratio_results: Dictionary mapping ratio (float) to results dictionary
    :return: List of comparison result entries for CSV export
    """

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


def process_augmented_data_evaluation(file, df_original_cleaned, feature_names, ga_selected_features, pca_n_components, rfe_selected_features, base_models, hp_params_map, results_original):
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
    :return: None
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Processing augmented data evaluation for: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}"
    )  # Output the verbose message

    augmented_file = find_data_augmentation_file(file)  # Look for augmented data file using wgangp.py naming convention

    if augmented_file is None:  # If no augmented file found at expected path
        print(
            f"\n{BackgroundColors.YELLOW}No augmented data found for this file. Skipping augmentation comparison.{Style.RESET_ALL}"
        )  # Print warning message about missing augmented file
        return  # Exit function early when no augmented file exists

    df_augmented = load_dataset(augmented_file)  # Load the augmented dataset from the discovered file

    if df_augmented is None:  # If augmented dataset failed to load from disk
        print(
            f"{BackgroundColors.YELLOW}Warning: Failed to load augmented dataset from {BackgroundColors.CYAN}{augmented_file}{BackgroundColors.YELLOW}. Skipping.{Style.RESET_ALL}"
        )  # Print warning message about load failure
        return  # Exit function early on load failure

    df_augmented_cleaned = preprocess_dataframe(df_augmented)  # Preprocess the augmented dataframe with same pipeline as original

    if not validate_augmented_dataframe(df_original_cleaned, df_augmented_cleaned, file):  # Validate augmented data is compatible with original
        return  # Exit function early if augmented data fails validation checks

    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*100}{Style.RESET_ALL}"
    )  # Print separator line for visual clarity
    print(
        f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}RATIO-BASED DATA AUGMENTATION EXPERIMENTS{Style.RESET_ALL}"
    )  # Print header for the ratio-based experiments section
    print(
        f"{BackgroundColors.GREEN}Ratios to evaluate: {BackgroundColors.CYAN}{[f'{int(r*100)}%' for r in config.get("stacking", {}).get("augmentation_ratios", [0.10, 0.25, 0.50, 0.75, 1.00])]}{Style.RESET_ALL}"
    )  # Print the list of ratios that will be evaluated
    print(
        f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*100}{Style.RESET_ALL}\n"
    )  # Print closing separator line

    all_ratio_results = {}  # Dictionary to store results for each ratio: {ratio: results_dict}

    for ratio_idx, ratio in enumerate(config.get("stacking", {}).get("augmentation_ratios", [0.10, 0.25, 0.50, 0.75, 1.00]), start=1):  # Iterate over each augmentation ratio
        ratio_pct = int(ratio * 100)  # Convert float ratio to integer percentage for display
        experiment_id = generate_experiment_id(file, "original_plus_augmented", ratio)  # Generate unique experiment ID for this ratio

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}[{ratio_idx}/{len(config.get("stacking", {}).get("augmentation_ratios", [0.10, 0.25, 0.50, 0.75, 1.00]))}] Evaluating Original + Augmented@{ratio_pct}%{Style.RESET_ALL}"
        )  # Print progress indicator for current ratio experiment

        df_sampled = sample_augmented_by_ratio(df_augmented_cleaned, df_original_cleaned, ratio)  # Sample augmented rows at the current ratio

        if df_sampled is None or df_sampled.empty:  # If sampling returned no valid data
            print(
                f"{BackgroundColors.YELLOW}Warning: Could not sample augmented data at ratio {ratio}. Skipping this ratio.{Style.RESET_ALL}"
            )  # Print warning about sampling failure
            continue  # Skip to the next ratio in the loop

        df_merged = merge_original_and_augmented(df_original_cleaned, df_sampled)  # Merge original data with sampled augmented data

        data_source_label = f"Original+Augmented@{ratio_pct}%"  # Build descriptive data source label for CSV traceability

        print(
            f"{BackgroundColors.GREEN}Merged dataset: {BackgroundColors.CYAN}{len(df_original_cleaned)} original + {len(df_sampled)} augmented = {len(df_merged)} total rows{Style.RESET_ALL}"
        )  # Print merged dataset size breakdown for transparency

        generate_augmentation_tsne_visualization(
            file, df_original_cleaned, df_sampled, ratio, "original_plus_augmented"
        )  # Generate t-SNE visualization for this augmentation ratio

        results_ratio = evaluate_on_dataset(
            file, df_merged, feature_names, ga_selected_features, pca_n_components,
            rfe_selected_features, base_models, data_source_label=data_source_label,
            hyperparams_map=hp_params_map, experiment_id=experiment_id,
            experiment_mode="original_plus_augmented", augmentation_ratio=ratio,
        )  # Evaluate all classifiers on the merged dataset with experiment metadata

        all_ratio_results[ratio] = results_ratio  # Store the results for this ratio in the results dictionary

        send_telegram_message(
            TELEGRAM_BOT, f"Completed augmentation ratio {ratio_pct}% for {os.path.basename(file)}"
        )  # Send Telegram notification for ratio completion

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


def print_file_processing_header(file, config=None):
    """
    Prints formatted header for file processing section.

    :param file: File path being processed
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
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


def process_single_file_evaluation(file, combined_df, combined_file_for_features, config=None):
    """
    Processes evaluation for a single file including feature loading, model preparation, and evaluation.

    :param file: File path to process
    :param combined_df: Combined dataframe (used if file == "combined")
    :param combined_file_for_features: File to use for feature selection metadata
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
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
    augmentation_ratios = config.get("execution", {}).get("augmentation_ratios", [])  # Get augmentation ratios from config
    
    if test_data_augmentation:  # If data augmentation testing is enabled
        generate_augmentation_tsne_visualization(
            file, df_original_cleaned, None, None, "original_only", config=config
        )  # Generate t-SNE visualization for original data only

    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}[1/{1 + len(augmentation_ratios) if test_data_augmentation else 1}] Evaluating on ORIGINAL data{Style.RESET_ALL}"
    )  # Print progress message with total step count
    results_original = evaluate_on_dataset(
        file, df_original_cleaned, feature_names, ga_selected_features, pca_n_components,
        rfe_selected_features, base_models, data_source_label="Original", hyperparams_map=hp_params_map,
        experiment_id=original_experiment_id, experiment_mode="original_only", augmentation_ratio=None,
        config=config
    )  # Evaluate on original data with experiment traceability metadata

    original_results_list = list(results_original.values())  # Convert results dict to list
    save_stacking_results(file, original_results_list, config=config)  # Save original results to CSV

    enable_automl = config.get("execution", {}).get("enable_automl", False)  # Get enable automl flag from config
    if enable_automl:  # If AutoML pipeline is enabled
        run_automl_pipeline(file, df_original_cleaned, feature_names, config=config)  # Run AutoML pipeline

    if test_data_augmentation:  # If data augmentation testing is enabled
        process_augmented_data_evaluation(
            file, df_original_cleaned, feature_names, ga_selected_features, pca_n_components,
            rfe_selected_features, base_models, hp_params_map, results_original, config=config
        )  # Process augmented data evaluation workflow


def process_files_in_path(input_path, dataset_name, config=None):
    """
    Processes all files in a given input path including file discovery and dataset combination.

    :param input_path: Directory path containing files to process
    :param dataset_name: Name of the dataset being processed
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
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

    files_to_process = determine_files_to_process(csv_file, input_path, config=config)  # Determine which files to process

    local_dataset_name = dataset_name or get_dataset_name(input_path)  # Use provided dataset name or infer from path

    combined_df, combined_file_for_features, files_to_process = combine_dataset_if_needed(files_to_process, config=config)  # Combine dataset files if needed

    for file in files_to_process:  # For each file to process
        process_single_file_evaluation(file, combined_df, combined_file_for_features, config=config)  # Process the single file evaluation


def process_dataset_paths(dataset_name, paths, config=None):
    """
    Processes all paths for a given dataset.

    :param dataset_name: Name of the dataset
    :param paths: List of paths to process for this dataset
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG

    verbose_output(
        f"{BackgroundColors.GREEN}Processing dataset: {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}",
        config=config
    )  # Output the verbose message

    print(
        f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing dataset: {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}"
    )  # Print dataset name

    for input_path in paths:  # For each path in the dataset's paths list
        process_files_in_path(input_path, dataset_name, config=config)  # Process all files in this path


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
    
    # Initialize configuration
    config = initialize_config(config_path=config_path, cli_args=None)  # Load base config
    
    # Apply programmatic overrides
    for key, value in config_overrides.items():  # Iterate over provided overrides
        if isinstance(value, dict) and key in config:  # If override is dict and key exists
            config[key] = deep_merge_dicts(config[key], value)  # Deep merge override
        else:  # Direct override
            config[key] = value  # Set value directly
    
    # Initialize logger
    initialize_logger(config=config)  # Setup logging
    
    # Run main pipeline
    main(config=config)  # Execute stacking pipeline


def to_seconds(obj):
    """
    Converts various time-like objects to seconds.
    
    :param obj: The object to convert (can be int, float, timedelta, datetime, etc.)
    :return: The equivalent time in seconds as a float, or None if conversion fails
    """
    
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


def calculate_execution_time(start_time, finish_time=None):
    """
    Calculates the execution time and returns a human-readable string.

    Accepts either:
    - Two datetimes/timedeltas: `calculate_execution_time(start, finish)`
    - A single timedelta or numeric seconds: `calculate_execution_time(delta)`
    - Two numeric timestamps (seconds): `calculate_execution_time(start_s, finish_s)`

    Returns a string like "1h 2m 3s".
    """

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


def play_sound(config=None):
    """
    Plays a sound when the program finishes and skips if the operating system is Windows.

    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
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


def main(config=None):
    """
    Main function.

    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG

    print(
        f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Classifiers Stacking{BackgroundColors.GREEN} program!{Style.RESET_ALL}\n"
    )  # Output the welcome message
    
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

    send_telegram_message(
        TELEGRAM_BOT, [f"Starting Classifiers Stacking at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"]
    )  # Send Telegram message indicating start

    threads_limit = set_threads_limit_based_on_ram(config=config)  # Adjust config.get("evaluation", {}).get("threads_limit", 2) based on system RAM
    
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


if __name__ == "__main__":
    # Parse CLI arguments
    cli_args = parse_cli_args()  # Parse command-line arguments
    
    # Initialize configuration with CLI overrides
    config = initialize_config(config_path=cli_args.config, cli_args=cli_args)  # Initialize config with file and CLI args
    
    # Initialize logger
    initialize_logger(config=config)  # Initialize logger with config
    
    # Run main function with config
    main(config=config)  # Run main with configuration
