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
import numpy as np  # Import numpy for numerical operations
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
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression  # For logistic regression model
from sklearn.metrics import (  # For performance metrics
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split  # For splitting the dataset and CV
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid  # For k-nearest neighbors model
from sklearn.neural_network import MLPClassifier  # For neural network model
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For label encoding and feature scaling
from sklearn.svm import SVC  # For Support Vector Machine model
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


# Execution Constants:
VERBOSE = False  # Set to True to output verbose messages
N_JOBS = -1  # Number of parallel jobs for GridSearchCV (-1 uses all processors)
THREADS_LIMIT = 2  # Number of threads for parallel evaluation of individual classifiers
TEST_DATA_AUGMENTATION = True  # Set to True to compare original vs augmented data performance
RESULTS_FILENAME = "Stacking_Classifiers_Results.csv"  # Filename for saving stacking results
AUGMENTATION_COMPARISON_FILENAME = "Data_Augmentation_Comparison_Results.csv"  # Filename for augmentation comparison results
RESULTS_CSV_COLUMNS = [  # Columns for the results CSV
    "model",
    "dataset",
    "feature_set",
    "classifier_type",
    "model_name",
    "data_source",
    "n_features",
    "n_samples_train",
    "n_samples_test",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "fpr",
    "fnr",
    "elapsed_time_s",
    "cv_method",
    "top_features",
    "rfe_ranking",
    "hyperparameters",
    "features_list",
    "Hardware",
]
MATCH_FILENAMES_TO_PROCESS = [""]  # List of specific filenames to search for a match (set to None to process all files)
IGNORE_FILES = [RESULTS_FILENAME]  # List of filenames to ignore when searching for datasets
IGNORE_DIRS = [
    "Classifiers",
    "Classifiers_Hyperparameters",
    "Dataset_Description",
    "Data_Separability",
    "Feature_Analysis",
]  # List of directory names to ignore when searching for datasets
HYPERPARAMETERS_FILENAME = "Hyperparameter_Optimization_Results.csv"  # Filename for hyperparameter optimization results
CACHE_PREFIX = "Cache_"  # Prefix for cache filenames
MODEL_EXPORT_BASE = "Feature_Analysis/Stacking/Models/"
SKIP_TRAIN_IF_MODEL_EXISTS = False  # If True, load exported models instead of retraining when available
CSV_FILE = None  # Optional CSV override from CLI
PROCESS_ENTIRE_DATASET = False  # Set to True to process all files in the dataset, False to only process the specified CSV_FILE

# Telegram Bot Setup:
TELEGRAM_BOT = None  # Global Telegram bot instance (initialized in setup_telegram_bot)

# Logger Setup:
logger = Logger(f"./Logs/{Path(__file__).stem}.log", clean=True)  # Create a Logger instance
sys.stdout = logger  # Redirect stdout to the logger
sys.stderr = logger  # Redirect stderr to the logger

# Sound Constants:
SOUND_COMMANDS = {
    "Darwin": "afplay",
    "Linux": "aplay",
    "Windows": "start",
}  # The commands to play a sound for each operating system
SOUND_FILE = "./.assets/Sounds/NotificationSound.wav"  # The path to the sound file

# RUN_FUNCTIONS:
RUN_FUNCTIONS = {
    "Play Sound": True,  # Set to True to play a sound when the program finishes
}

DATASETS = {  # Dictionary containing dataset paths and feature files
    "CICDDoS2019-Dataset": [  # List of paths to the CICDDoS2019 dataset
        "./Datasets/CICDDoS2019/01-12/",
        "./Datasets/CICDDoS2019/03-11/",
    ],
}

# Functions Definitions:


def verbose_output(true_string="", false_string=""):
    """
    Outputs a message if the VERBOSE constant is set to True.

    :param true_string: The string to be outputted if the VERBOSE constant is set to True.
    :param false_string: The string to be outputted if the VERBOSE constant is set to False.
    :return: None
    """

    if VERBOSE and true_string != "":  # If VERBOSE is True and a true_string was provided
        print(true_string)  # Output the true statement string
    elif false_string != "":  # If a false_string was provided
        print(false_string)  # Output the false statement string


def verify_dot_env_file():
    """
    Verifies if the .env file exists in the current directory.

    :return: True if the .env file exists, False otherwise
    """

    env_path = Path(__file__).parent / ".env"  # Path to the .env file
    if not env_path.exists():  # If the .env file does not exist
        print(f"{BackgroundColors.CYAN}.env{BackgroundColors.YELLOW} file not found at {BackgroundColors.CYAN}{env_path}{BackgroundColors.YELLOW}. Telegram messages may not be sent.{Style.RESET_ALL}")
        return False  # Return False

    return True  # Return True if the .env file exists


def setup_telegram_bot():
    """
    Sets up the Telegram bot for progress messages.

    :return: None
    """
    
    verbose_output(
        f"{BackgroundColors.GREEN}Setting up Telegram bot for messages...{Style.RESET_ALL}"
    )  # Output the verbose message

    verify_dot_env_file()  # Verify if the .env file exists

    global TELEGRAM_BOT  # Declare the module-global telegram_bot variable

    try:  # Try to initialize the Telegram bot
        TELEGRAM_BOT = TelegramBot()  # Initialize Telegram bot for progress messages
        telegram_module.TELEGRAM_DEVICE_INFO = f"{telegram_module.get_local_ip()} - {platform.system()}"
        telegram_module.RUNNING_CODE = os.path.basename(__file__)
    except Exception as e:
        print(f"{BackgroundColors.RED}Failed to initialize Telegram bot: {e}{Style.RESET_ALL}")
        TELEGRAM_BOT = None  # Set to None if initialization fails


def set_threads_limit_based_on_ram():
    """
    Sets THREADS_LIMIT to 1 if system RAM is <= 16GB to avoid memory issues.

    :param: None
    :return: None
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Verifying system RAM to set THREADS_LIMIT...{Style.RESET_ALL}"
    )  # Output the verbose message

    global THREADS_LIMIT  # Use the global THREADS_LIMIT variable

    ram_gb = psutil.virtual_memory().total / (1024**3)  # Get total system RAM in GB

    if ram_gb <= 128:  # If RAM is less than or equal to 128GB
        THREADS_LIMIT = 1  # Set THREADS_LIMIT to 1
        verbose_output(
            f"{BackgroundColors.YELLOW}System RAM is {ram_gb:.1f}GB (<=128GB). Setting THREADS_LIMIT to 1.{Style.RESET_ALL}"
        )


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


def get_files_to_process(directory_path, file_extension=".csv"):
    """
    Collect all files with a given extension inside a directory (non-recursive).

    Performs validation, respects IGNORE_FILES, and optionally filters by
    MATCH_FILENAMES_TO_PROCESS when defined.

    :param directory_path: Path to the directory to scan
    :param file_extension: File extension to include (default: ".csv")
    :return: Sorted list of matching file paths
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Getting all {BackgroundColors.CYAN}{file_extension}{BackgroundColors.GREEN} files in: {BackgroundColors.CYAN}{directory_path}{Style.RESET_ALL}"
    )  # Verbose: starting file collection
    verify_filepath_exists(directory_path)  # Validate directory path exists

    if not os.path.isdir(directory_path):  # Check if path is a valid directory
        verbose_output(
            f"{BackgroundColors.RED}Not a directory: {BackgroundColors.CYAN}{directory_path}{Style.RESET_ALL}"
        )  # Verbose: invalid directory
        return []  # Return empty list for invalid paths

    try:  # Attempt to read MATCH_FILENAMES_TO_PROCESS if defined
        match_names = (
            set(MATCH_FILENAMES_TO_PROCESS) if MATCH_FILENAMES_TO_PROCESS not in ([], [""], [" "]) else None
        )  # Load match list or None
        if match_names:
            verbose_output(
                f"{BackgroundColors.GREEN}Filtering to filenames: {BackgroundColors.CYAN}{match_names}{Style.RESET_ALL}"
            )  # Verbose: applying filename filter
    except NameError:  # MATCH_FILENAMES_TO_PROCESS not defined
        match_names = None  # No filtering will be applied

    files = []  # Accumulator for valid files

    for item in os.listdir(directory_path):  # Iterate directory entries
        item_path = os.path.join(directory_path, item)  # Absolute path
        filename = os.path.basename(item_path)  # Extract just the filename

        if any(ignore == filename or ignore == item_path for ignore in IGNORE_FILES):  # Check if file is in ignore list
            verbose_output(
                f"{BackgroundColors.YELLOW}Ignoring {BackgroundColors.CYAN}{filename}{BackgroundColors.YELLOW} (listed in IGNORE_FILES){Style.RESET_ALL}"
            )  # Verbose: ignoring file
            continue  # Skip ignored file

        if os.path.isfile(item_path) and item.lower().endswith(file_extension):  # File matches extension requirement
            if (
                match_names is not None and filename not in match_names
            ):  # Filename not included in MATCH_FILENAMES_TO_PROCESS
                verbose_output(
                    f"{BackgroundColors.YELLOW}Skipping {BackgroundColors.CYAN}{filename}{BackgroundColors.YELLOW} (not in MATCH_FILENAMES_TO_PROCESS){Style.RESET_ALL}"
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


def combine_dataset_files(files_list):
    """
    Load, preprocess and combine multiple dataset CSVs into a single DataFrame.

    :param files_list: List of dataset CSV file paths to combine
    :return: Combined DataFrame with aligned features and target, or None if no compatible files found
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Combining dataset files: {BackgroundColors.CYAN}{files_list}{Style.RESET_ALL}"
    )  # Output the verbose message

    processed_files = []  # Initialize list for processed file data
    for f in files_list:  # Iterate over each file in the list
        result = process_single_file(f)  # Process the single file
        if result is not None:  # If processing succeeded
            df_clean, target_col, feat_cols = result  # Unpack the result
            processed_files.append((f, df_clean, target_col, feat_cols))  # Add to processed list

    if not processed_files:  # If no files were processed successfully
        print(f"{BackgroundColors.RED}No compatible files found to combine for dataset: {files_list}.{Style.RESET_ALL}")  # Print error
        return None  # Return None

    common_features, target_col_name, dfs = find_common_features_and_target(processed_files)  # Find common features and target
    if common_features is None:  # If finding common features failed
        print(f"{BackgroundColors.RED}No valid target column found.{Style.RESET_ALL}")  # Print error
        return None  # Return None

    reduced_dfs = create_reduced_dataframes(dfs, common_features, target_col_name)  # Create reduced dataframes
    combined = combine_and_clean_dataframes(reduced_dfs)  # Combine and clean the dataframes

    return combined, target_col_name  # Return the combined dataframe and target column name


def find_data_augmentation_file(original_file_path):
    """
    Find the corresponding data augmentation file for an original CSV file.
    The augmented file is expected to be in ../Data_Augmentation/<same_filename>

    :param original_file_path: Path to the original CSV file
    :return: Path to the augmented file if it exists, None otherwise
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Looking for data augmentation file for: {BackgroundColors.CYAN}{original_file_path}{Style.RESET_ALL}"
    )  # Output the verbose message

    original_path = Path(original_file_path)  # Create Path object
    augmented_dir = original_path.parent / "Data_Augmentation"  # Data_Augmentation subdirectory
    augmented_file = augmented_dir / original_path.name  # Same filename in augmented directory

    if augmented_file.exists():  # If augmented file exists
        verbose_output(
            f"{BackgroundColors.GREEN}Found augmented file: {BackgroundColors.CYAN}{augmented_file}{Style.RESET_ALL}"
        )  # Output success message
        return str(augmented_file)  # Return path as string
    else:  # If augmented file doesn't exist
        verbose_output(
            f"{BackgroundColors.YELLOW}No augmented file found for: {BackgroundColors.CYAN}{original_file_path}{Style.RESET_ALL}"
        )  # Output warning
        return None  # Return None


def merge_original_and_augmented(original_df, augmented_df):
    """
    Merge original and augmented dataframes by concatenating them.

    :param original_df: Original DataFrame
    :param augmented_df: Augmented DataFrame
    :return: Merged DataFrame
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Merging original ({len(original_df)} rows) and augmented ({len(augmented_df)} rows) data{Style.RESET_ALL}"
    )  # Output the verbose message

    merged_df = pd.concat([original_df, augmented_df], ignore_index=True)  # Concatenate dataframes
    
    verbose_output(
        f"{BackgroundColors.GREEN}Merged dataset has {BackgroundColors.CYAN}{len(merged_df)}{BackgroundColors.GREEN} rows{Style.RESET_ALL}"
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


def save_augmentation_comparison_results(file_path, comparison_results):
    """
    Save data augmentation comparison results to CSV file.

    :param file_path: Path to the original CSV file being processed
    :param comparison_results: List of dictionaries containing comparison metrics
    :return: None
    """

    if not comparison_results:  # If no results to save
        return  # Exit early

    file_path_obj = Path(file_path)  # Create Path object
    feature_analysis_dir = file_path_obj.parent / "Feature_Analysis"  # Feature_Analysis directory
    os.makedirs(feature_analysis_dir, exist_ok=True)  # Ensure directory exists
    output_path = feature_analysis_dir / AUGMENTATION_COMPARISON_FILENAME  # Output file path

    df = pd.DataFrame(comparison_results)  # Convert results to DataFrame

    # Define column order for better readability
    column_order = [
        "dataset",
        "feature_set",
        "classifier_type",
        "model_name",
        "data_source",
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
    ]  # Define desired column order

    # Reorder columns (only include columns that exist)
    existing_columns = [col for col in column_order if col in df.columns]  # Filter to existing columns
    df = df[existing_columns]  # Reorder DataFrame columns

    df = add_hardware_column(df, existing_columns)  # Add hardware specifications column

    df.to_csv(output_path, index=False)  # Save to CSV file
    print(
        f"{BackgroundColors.GREEN}Saved augmentation comparison results to {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
    )  # Output success message


def find_local_feature_file(file_dir, filename):
    """
    Attempt to locate <file_dir>/Feature_Analysis/<filename>.

    :param file_dir: Directory to search within
    :param filename: Filename to search for
    :return: The matching path or None
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Checking local Feature_Analysis in directory: {BackgroundColors.CYAN}{file_dir}{BackgroundColors.GREEN} for file: {BackgroundColors.CYAN}{filename}{Style.RESET_ALL}"
    )  # Output the verbose message

    candidate = os.path.join(file_dir, "Feature_Analysis", filename)  # Construct candidate path

    if os.path.exists(candidate):  # If the candidate file exists
        return candidate  # Return the candidate path

    return None  # Not found


def find_parent_feature_file(file_dir, filename):
    """
    Ascend parent directories searching for <parent>/Feature_Analysis/<filename>.

    :param file_dir: Directory to search within
    :param filename: Filename to search for
    :return: The matching path or None
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Ascending parent directories from: {BackgroundColors.CYAN}{file_dir}{BackgroundColors.GREEN} searching for file: {BackgroundColors.CYAN}{filename}{Style.RESET_ALL}"
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


def find_dataset_level_feature_file(file_path, filename):
    """
    Try dataset-level search:

    - /.../Datasets/<dataset_name>/Feature_Analysis/<filename>
    - recursive search under dataset directory

    :param file_path: Path to the file
    :param filename: Filename to search for
    :return: The matching path or None
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Searching dataset-level Feature_Analysis for file: {BackgroundColors.CYAN}{filename}{BackgroundColors.GREEN} related to file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}"
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


def find_feature_file(file_path, filename):
    """
    Locate a feature-analysis CSV file related to `file_path`.

    Search order:
    - <file_dir>/Feature_Analysis/<filename>
    - ascend parent directories checking <parent>/Feature_Analysis/<filename>
    - dataset-level folder under `.../Datasets/<dataset_name>/Feature_Analysis/<filename>`
    - fallback: search under workspace ./Datasets/**/Feature_Analysis/<filename`

    :param file_path: Path to the file
    :param filename: Filename to search for
    :return: The matching path or None
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Searching for feature analysis file: {BackgroundColors.CYAN}{filename}{BackgroundColors.GREEN} related to file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}"
    )  # Output the verbose message

    file_dir = os.path.dirname(os.path.abspath(file_path))  # Get the directory of the input file

    result = find_local_feature_file(file_dir, filename)  # 1. Local Feature_Analysis in the same directory
    if result is not None:  # If found
        return result  # Return the result

    result = find_parent_feature_file(file_dir, filename)  # 2. Ascend parents checking for Feature_Analysis
    if result is not None:  # If found
        return result  # Return the result

    result = find_dataset_level_feature_file(file_path, filename)  # 3. Dataset-level Feature_Analysis
    if result is not None:  # If found
        return result  # Return the result

    print(
        f"{BackgroundColors.YELLOW}Warning: Feature analysis file {BackgroundColors.CYAN}{filename}{BackgroundColors.YELLOW} not found for dataset containing {BackgroundColors.CYAN}{file_path}{BackgroundColors.YELLOW}.{Style.RESET_ALL}"
    )  # Output the warning message

    return None  # Return None if not found


def extract_genetic_algorithm_features(file_path):
    """
    Extracts the features selected by the Genetic Algorithm from the corresponding
    "Genetic_Algorithm_Results.csv" file located in the "Feature_Analysis"
    subdirectory relative to the input file's directory.

    It specifically retrieves the 'best_features' (a JSON string) from the row
    where the 'run_index' is 'best', and returns it as a Python list.

    :param file_path: Full path to the current CSV file being processed (e.g., "./Datasets/.../DrDoS_DNS.csv").
    :return: List of features selected by the GA, or None if the file is not found or fails to load/parse.
    """

    file_dir = os.path.dirname(file_path)  # Determine the directory of the input file
    verbose_output(
        f"{BackgroundColors.GREEN}Extracting GA features for file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}"
    )  # Output the verbose message

    ga_results_path = find_feature_file(file_path, "Genetic_Algorithm_Results.csv")  # Find the GA results file
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
            f"{BackgroundColors.GREEN}Successfully extracted {BackgroundColors.CYAN}{len(ga_features)}{BackgroundColors.GREEN} GA features from the 'best' run.{Style.RESET_ALL}"
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


def extract_principal_component_analysis_features(file_path):
    """
    Extracts the optimal number of Principal Components (n_components)
    from the "PCA_Results.csv" file located in the "Feature_Analysis"
    subdirectory relative to the input file's directory.

    The best result is determined by the highest 'cv_f1_score'.

    :param file_path: Full path to the current CSV file being processed (e.g., "./Datasets/.../DrDoS_DNS.csv").
    :return: Integer representing the optimal number of components (n_components), or None if the file is not found or fails to load/parse.
    """

    file_dir = os.path.dirname(file_path)  # Determine the directory of the input file
    verbose_output(
        f"{BackgroundColors.GREEN}Extracting PCA features for file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}"
    )  # Output the verbose message

    pca_results_path = find_feature_file(file_path, "PCA_Results.csv")  # Find the PCA results file
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


def extract_recursive_feature_elimination_features(file_path):
    """
    Extracts the "top_features" list (Python literal string) from the first row of the
    "RFE_Run_Results.csv" file located in the "Feature_Analysis" subdirectory
    relative to the input file's directory.

    :param file_path: Full path to the current CSV file being processed (e.g., "./Datasets/.../DrDoS_DNS.csv").
    :return: List of top features selected by RFE from the first run, or None if the file is not found or fails to load/parse.
    """

    file_dir = os.path.dirname(file_path)  # Determine the directory of the input file
    verbose_output(
        f"{BackgroundColors.GREEN}Extracting RFE features for file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}"
    )  # Output the verbose message

    rfe_runs_path = find_feature_file(file_path, "RFE_Run_Results.csv")  # Find the RFE runs file
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
                f"{BackgroundColors.GREEN}Successfully extracted RFE top features from Run 1. Total features: {BackgroundColors.CYAN}{len(rfe_features)}{Style.RESET_ALL}"
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


def load_feature_selection_results(file_path):
    """
    Load GA, RFE and PCA feature selection artifacts for a given dataset file and
    print concise status messages.

    :param file_path: Path to the dataset CSV being processed.
    :return: Tuple (ga_selected_features, pca_n_components, rfe_selected_features)
    """

    ga_selected_features = extract_genetic_algorithm_features(file_path)  # Extract GA features
    if ga_selected_features:  # If GA features were successfully extracted
        verbose_output(
            f"{BackgroundColors.GREEN}Genetic Algorithm Features successfully loaded for {BackgroundColors.CYAN}{os.path.basename(file_path)}{BackgroundColors.GREEN}. Total features: {BackgroundColors.CYAN}{len(ga_selected_features)}{Style.RESET_ALL}"
        )
        verbose_output(
            f"{BackgroundColors.GREEN}Genetic Algorithm Selected Features: {BackgroundColors.CYAN}{ga_selected_features}{Style.RESET_ALL}"
        )
    else:  # If GA features were not extracted
        print(
            f"{BackgroundColors.YELLOW}Proceeding without GA features for {BackgroundColors.CYAN}{os.path.basename(file_path)}{Style.RESET_ALL}"
        )

    pca_n_components = extract_principal_component_analysis_features(file_path)  # Extract PCA components
    if pca_n_components:  # If PCA components were successfully extracted
        verbose_output(
            f"{BackgroundColors.GREEN}PCA optimal components successfully loaded for {BackgroundColors.CYAN}{os.path.basename(file_path)}{BackgroundColors.GREEN}: {BackgroundColors.CYAN}{pca_n_components}{Style.RESET_ALL}"
        )
        verbose_output(
            f"{BackgroundColors.GREEN}PCA Number of Components: {BackgroundColors.CYAN}{pca_n_components}{Style.RESET_ALL}"
        )
    else:  # If PCA components were not extracted
        print(
            f"{BackgroundColors.YELLOW}Proceeding without PCA components for {BackgroundColors.CYAN}{os.path.basename(file_path)}{Style.RESET_ALL}"
        )

    rfe_selected_features = extract_recursive_feature_elimination_features(file_path)  # Extract RFE features
    if rfe_selected_features:  # If RFE features were successfully extracted
        verbose_output(
            f"{BackgroundColors.GREEN}RFE Features successfully loaded for {BackgroundColors.CYAN}{os.path.basename(file_path)}{BackgroundColors.GREEN}. Total features: {BackgroundColors.CYAN}{len(rfe_selected_features)}{Style.RESET_ALL}"
        )
        verbose_output(
            f"{BackgroundColors.GREEN}RFE Selected Features: {BackgroundColors.CYAN}{rfe_selected_features}{Style.RESET_ALL}"
        )
    else:  # If RFE features were not extracted
        print(
            f"{BackgroundColors.YELLOW}Proceeding without RFE features for {BackgroundColors.CYAN}{os.path.basename(file_path)}{Style.RESET_ALL}"
        )

    return ga_selected_features, pca_n_components, rfe_selected_features  # Return the extracted features


def load_dataset(csv_path):
    """
    Load CSV and return DataFrame.

    :param csv_path: Path to CSV dataset.
    :return: DataFrame
    """

    verbose_output(
        f"\n{BackgroundColors.GREEN}Loading dataset from: {BackgroundColors.CYAN}{csv_path}{Style.RESET_ALL}"
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


def preprocess_dataframe(df, remove_zero_variance=True):
    """
    Preprocess a DataFrame by removing rows with NaN or infinite values and
    dropping zero-variance numeric features.

    :param df: pandas DataFrame to preprocess
    :param remove_zero_variance: whether to drop numeric columns with zero variance
    :return: cleaned DataFrame
    """

    if remove_zero_variance:  # If remove_zero_variance is set to True
        verbose_output(
            f"{BackgroundColors.GREEN}Preprocessing DataFrame: "
            f"{BackgroundColors.CYAN}normalizing and sanitizing column names, removing NaN/infinite rows, and dropping zero-variance numeric features"
            f"{BackgroundColors.GREEN}.{Style.RESET_ALL}"
        )
    else:  # If remove_zero_variance is set to False
        verbose_output(
            f"{BackgroundColors.GREEN}Preprocessing DataFrame: "
            f"{BackgroundColors.CYAN}normalizing and sanitizing column names and removing NaN/infinite rows"
            f"{BackgroundColors.GREEN}.{Style.RESET_ALL}"
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


def scale_and_split(X, y, test_size=0.2, random_state=42):
    """
    Scales the numeric features using StandardScaler and splits the data
    into training and testing sets.

    Note: The target variable 'y' is label-encoded before splitting.

    :param X: Features DataFrame (must contain numeric features).
    :param y: Target Series or array.
    :param test_size: Fraction of the data to reserve for the test set.
    :param random_state: Seed for the random split for reproducibility.
    :return: Tuple (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Scaling features and splitting data (train/test ratio: {BackgroundColors.CYAN}{1-test_size}/{test_size}{BackgroundColors.GREEN})...{Style.RESET_ALL}"
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
        f"{BackgroundColors.GREEN}Data split successful. Training set shape: {BackgroundColors.CYAN}{X_train_scaled.shape}{BackgroundColors.GREEN}. Testing set shape: {BackgroundColors.CYAN}{X_test_scaled.shape}{Style.RESET_ALL}"
    )  # Output the successful split message

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
    )  # Return scaled features, target, and the fitted scaler


def get_models():
    """
    Initializes and returns a dictionary of models to train.

    :param: None
    :return: Dictionary of model name and instance
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Initializing models for training...{Style.RESET_ALL}"
    )  # Output the verbose message

    return {  # Dictionary of models to train
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=N_JOBS),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=42, n_jobs=N_JOBS),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=N_JOBS),
        "Nearest Centroid": NearestCentroid(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "LightGBM": lgb.LGBMClassifier(
            force_row_wise=True, min_gain_to_split=0.01, random_state=42, verbosity=-1, n_jobs=N_JOBS
        ),
        "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    }


def extract_hyperparameter_optimization_results(csv_path):
    """
    Extract hyperparameter optimization results for a specific dataset file.

    Looks for the HYPERPARAMETERS_FILENAME file in the "Classifiers_Hyperparameters"
    subdirectory relative to the dataset CSV file. Filters results to match the
    current base_csv filename being processed.

    This function extracts **only the best hyperparameters** for each classifier
    that corresponds to the current file being processed.

    :param csv_path: Path to the dataset CSV file being processed.
    :return: Dictionary mapping model names to their best hyperparameters, or None if not found.
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Looking for hyperparameter optimization results for: {BackgroundColors.CYAN}{csv_path}{Style.RESET_ALL}"
    )  # Inform user which dataset we're searching for

    file_dir = os.path.dirname(csv_path)  # Directory containing the dataset file
    base_filename = os.path.basename(csv_path)  # Get the base filename (e.g., "UDPLag.csv")

    hyperparams_path = os.path.join(
        file_dir, "Classifiers_Hyperparameters", HYPERPARAMETERS_FILENAME
    )  # Path to hyperparameter optimization results

    if not verify_filepath_exists(hyperparams_path):  # If the hyperparameters file does not exist
        verbose_output(
            f"{BackgroundColors.YELLOW}No hyperparameter optimization results found at: {BackgroundColors.CYAN}{hyperparams_path}{Style.RESET_ALL}"
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
            f"{BackgroundColors.YELLOW}No hyperparameter results found for file: {BackgroundColors.CYAN}{base_filename}{BackgroundColors.YELLOW} in {BackgroundColors.CYAN}{hyperparams_path}{Style.RESET_ALL}"
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
        f"{BackgroundColors.GREEN}Loaded {BackgroundColors.CYAN}{len(results)}{BackgroundColors.GREEN} hyperparameter optimization results for {BackgroundColors.CYAN}{base_filename}{BackgroundColors.GREEN} from: {BackgroundColors.CYAN}{hyperparams_path}{Style.RESET_ALL}"
    )
    return results  # Return the normalized results mapping


def apply_hyperparameters_to_models(hyperparams_map, models_map):
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
    :return: Updated models_map with applied hyperparameters where possible
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Starting to apply hyperparameters to models...{Style.RESET_ALL}"
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
                    f"{BackgroundColors.YELLOW}Warning: Parsed hyperparameters for {BackgroundColors.CYAN}{model_name}{BackgroundColors.YELLOW} are not a dict. Skipping.{Style.RESET_ALL}"
                )
                continue  # Skip invalid parameter entries

            try:  # Try applying parameters
                model.set_params(**params)  # Apply parameters to estimator
                verbose_output(
                    f"{BackgroundColors.GREEN}Applied hyperparameters to {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}"
                )  # Inform success
            except Exception as e:  # If applying fails
                print(
                    f"{BackgroundColors.YELLOW}Failed to apply hyperparameters to {BackgroundColors.CYAN}{model_name}{BackgroundColors.YELLOW}: {e}{Style.RESET_ALL}"
                )  # Warn user
        except Exception:  # Catch any unexpected errors for this model
            continue  # Skip problematic entries silently

    return models_map  # Return updated model mapping


def load_pca_object(file_path, pca_n_components):
    """
    Loads a pre-fitted PCA object from a pickle file.

    :param file_path: Path to the dataset CSV file.
    :param pca_n_components: Number of PCA components to load.
    :return: PCA object if found, None otherwise.
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Loading the PCA Cache object with {BackgroundColors.CYAN}{pca_n_components}{BackgroundColors.GREEN} components from file {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}"
    )  # Output the verbose message

    file_dir = os.path.dirname(file_path)  # Get the directory of the dataset
    pca_file = os.path.join(
        file_dir, "Cache", f"PCA_{pca_n_components}_components.pkl"
    )  # Construct the path to the PCA pickle file

    if not verify_filepath_exists(pca_file):  # Check if the PCA file exists
        verbose_output(
            f"{BackgroundColors.YELLOW}PCA object file not found at {BackgroundColors.CYAN}{pca_file}{Style.RESET_ALL}"
        )
        return None  # Return None if the file doesn't exist

    try:  # Try to load the PCA object
        with open(pca_file, "rb") as f:  # Open the PCA pickle file
            pca = pickle.load(f)  # Load the PCA object
        verbose_output(
            f"{BackgroundColors.GREEN}Successfully loaded PCA object from {BackgroundColors.CYAN}{pca_file}{Style.RESET_ALL}"
        )
        return pca  # Return the loaded PCA object
    except Exception as e:  # Handle any errors during loading
        print(
            f"{BackgroundColors.RED}Error loading PCA object from {BackgroundColors.CYAN}{pca_file}{BackgroundColors.RED}: {e}{Style.RESET_ALL}"
        )
        return None  # Return None if there was an error


def apply_pca_transformation(X_train_scaled, X_test_scaled, pca_n_components, file_path=None):
    """
    Applies Principal Component Analysis (PCA) transformation to the scaled training
    and testing datasets using the optimal number of components.

    First attempts to load a pre-fitted PCA object from disk. If not found,
    fits a new PCA model on the training data.

    :param X_train_scaled: Scaled training features (numpy array).
    :param X_test_scaled: Scaled testing features (numpy array).
    :param pca_n_components: Optimal number of components (integer), or None/0 if PCA is skipped.
    :param file_path: Path to the dataset CSV file (optional, for loading pre-fitted PCA).
    :return: Tuple (X_train_pca, X_test_pca) - Transformed features, or (None, None).
    """

    X_train_pca = None  # Initialize PCA training features
    X_test_pca = None  # Initialize PCA testing features

    if pca_n_components is not None and pca_n_components > 0:  # If PCA components are specified
        verbose_output(
            f"{BackgroundColors.GREEN}Starting PCA transformation with {BackgroundColors.CYAN}{pca_n_components}{BackgroundColors.GREEN} components...{Style.RESET_ALL}"
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
            pca = load_pca_object(file_path, n_components)  # Load pre-fitted PCA object

        if pca is None:  # If PCA object wasn't loaded, fit a new one
            verbose_output(
                f"{BackgroundColors.GREEN}Fitting new PCA model with {BackgroundColors.CYAN}{n_components}{BackgroundColors.GREEN} components...{Style.RESET_ALL}"
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


def export_model_and_scaler(model, scaler, dataset_name, model_name, feature_names, best_params=None, feature_set=None, dataset_csv_path=None):
    """
    Export model, scaler and metadata for stacking evaluations.
    """
    
    def safe_filename(name):
        return re.sub(r'[\\/*?:"<>|]', "_", str(name))

    # Prefer dataset-local export directory when a CSV path is provided
    if dataset_csv_path:
        file_path_obj = Path(dataset_csv_path)
        export_dir = file_path_obj.parent / "Classifiers" / "Models"
    else:
        export_dir = Path(MODEL_EXPORT_BASE) / safe_filename(dataset_name)
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


def evaluate_individual_classifier(model, model_name, X_train, y_train, X_test, y_test, dataset_file=None, scaler=None, feature_names=None, feature_set=None):
    """
    Trains an individual classifier and evaluates its performance on the test set.

    :param model: The classifier model object to train.
    :param model_name: Name of the classifier (for logging).
    :param X_train: Training features (scaled numpy array).
    :param y_train: Training target labels (encoded Series/array).
    :param X_test: Testing features (scaled numpy array).
    :param y_test: Testing target labels (encoded Series/array).
    :return: Metrics tuple (acc, prec, rec, f1, fpr, fnr, elapsed_time)
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Training {BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN}...{Style.RESET_ALL}"
    )  # Output the verbose message

    start_time = time.time()  # Record the start time

    # If requested, attempt to load an existing exported model instead of retraining
    if dataset_file is not None and SKIP_TRAIN_IF_MODEL_EXISTS:
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


def save_stacking_results(csv_path, results_list):
    """Save the stacking results to CSV file located in same dataset Feature_Analysis directory.

    Writes richer metadata fields matching RFE outputs: model, dataset, accuracy, precision,
    recall, f1_score, fpr, fnr, elapsed_time_s, cv_method, top_features, rfe_ranking,
    hyperparameters, features_list and Hardware.
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Preparing to save {BackgroundColors.CYAN}{len(results_list)}{BackgroundColors.GREEN} stacking results to CSV...{Style.RESET_ALL}"
    )

    if not results_list:
        print(f"{BackgroundColors.YELLOW}Warning: No results provided to save.{Style.RESET_ALL}")
        return

    file_path_obj = Path(csv_path)
    feature_analysis_dir = file_path_obj.parent / "Feature_Analysis"
    os.makedirs(feature_analysis_dir, exist_ok=True)
    stacking_dir = feature_analysis_dir / "Stacking"
    os.makedirs(stacking_dir, exist_ok=True)
    output_path = stacking_dir / RESULTS_FILENAME

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
    column_order = list(RESULTS_CSV_COLUMNS)

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


def get_cache_file_path(csv_path):
    """
    Generate the cache file path for a given dataset CSV path.

    :param csv_path: Path to the dataset CSV file
    :return: Path to the cache file
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Generating cache file path for: {BackgroundColors.CYAN}{csv_path}{Style.RESET_ALL}"
    )  # Output the verbose message

    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]  # Get base dataset name
    output_dir = f"{os.path.dirname(csv_path)}/Classifiers"  # Directory relative to the dataset
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    cache_filename = f"{CACHE_PREFIX}{dataset_name}-Stacking_Classifiers_Results.csv"  # Cache filename
    cache_path = os.path.join(output_dir, cache_filename)  # Full cache file path

    return cache_path  # Return the cache file path


def load_cache_results(csv_path):
    """
    Load cached results from the cache file if it exists.

    :param csv_path: Path to the dataset CSV file
    :return: Dictionary mapping (feature_set, model_name) to result entry
    """

    cache_path = get_cache_file_path(csv_path)  # Get the cache file path

    if not os.path.exists(cache_path):  # If cache file doesn't exist
        verbose_output(
            f"{BackgroundColors.YELLOW}No cache file found at: {BackgroundColors.CYAN}{cache_path}{Style.RESET_ALL}"
        )  # Output the verbose message
        return {}  # Return empty dictionary

    verbose_output(
        f"{BackgroundColors.GREEN}Loading cached results from: {BackgroundColors.CYAN}{cache_path}{Style.RESET_ALL}"
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


def remove_cache_file(csv_path):
    """
    Remove the cache file after successful completion.

    :param csv_path: Path to the dataset CSV file
    :return: None
    """

    cache_path = get_cache_file_path(csv_path)  # Get the cache file path

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


def calculate_execution_time(start_time, finish_time=None):
    """
    Calculates the execution time and returns a human-readable string.

    Accepts either:
    - Two datetimes/timedeltas: `calculate_execution_time(start, finish)`
    - A single timedelta or numeric seconds: `calculate_execution_time(delta)`
    - Two numeric timestamps (seconds): `calculate_execution_time(start_s, finish_s)`

    Returns a string like "1h 2m 3s".
    """

    def _to_seconds(obj):  # Convert various time-like objects to seconds
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

    if finish_time is None:  # Single-argument mode: start_time already represents duration or seconds
        total_seconds = _to_seconds(start_time)  # Try to convert provided value to seconds
        if total_seconds is None:  # Conversion failed
            try:  # Attempt numeric coercion
                total_seconds = float(start_time)  # Attempt numeric coercion
            except Exception:
                total_seconds = 0.0  # Fallback to zero
    else:  # Two-argument mode: Compute difference finish_time - start_time
        st = _to_seconds(start_time)  # Convert start to seconds if possible
        ft = _to_seconds(finish_time)  # Convert finish to seconds if possible
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


def play_sound():
    """
    Plays a sound when the program finishes and skips if the operating system is Windows.

    :param: None
    :return: None
    """

    current_os = platform.system()  # Get the current operating system
    if current_os == "Windows":  # If the current operating system is Windows
        return  # Do nothing

    if verify_filepath_exists(SOUND_FILE):  # If the sound file exists
        if current_os in SOUND_COMMANDS:  # If the platform.system() is in the SOUND_COMMANDS dictionary
            os.system(f"{SOUND_COMMANDS[current_os]} {SOUND_FILE}")  # Play the sound
        else:  # If the platform.system() is not in the SOUND_COMMANDS dictionary
            print(
                f"{BackgroundColors.RED}The {BackgroundColors.CYAN}{current_os}{BackgroundColors.RED} is not in the {BackgroundColors.CYAN}SOUND_COMMANDS dictionary{BackgroundColors.RED}. Please add it!{Style.RESET_ALL}"
            )
    else:  # If the sound file does not exist
        print(
            f"{BackgroundColors.RED}Sound file {BackgroundColors.CYAN}{SOUND_FILE}{BackgroundColors.RED} not found. Make sure the file exists.{Style.RESET_ALL}"
        )


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
    :param data_source_label: Label for data source ("Original", "Augmented", or "Original+Augmented")
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
        final_estimator=RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=N_JOBS),
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        n_jobs=N_JOBS,
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
            max_workers=THREADS_LIMIT
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


def main():
    """
    Main function.

    :param: None
    :return: None
    """

    print(
        f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Classifiers Stacking{BackgroundColors.GREEN} program!{Style.RESET_ALL}\n"
    )  # Output the welcome message
    
    if TEST_DATA_AUGMENTATION:
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.YELLOW}Data Augmentation Testing: {BackgroundColors.CYAN}ENABLED{Style.RESET_ALL}"
        )
        print(
            f"{BackgroundColors.GREEN}Will compare performance on: Original vs Augmented vs Original+Augmented{Style.RESET_ALL}\n"
        )
    
    start_time = datetime.datetime.now()  # Get the start time of the program
    
    setup_telegram_bot()  # Setup Telegram bot if configured
    
    send_telegram_message(TELEGRAM_BOT, [f"Starting Classifiers Stacking at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"])  # Send Telegram message indicating start

    set_threads_limit_based_on_ram()  # Adjust THREADS_LIMIT based on system RAM

    for dataset_name, paths in DATASETS.items():  # For each dataset in the DATASETS dictionary
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing dataset: {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}"
        )
        for input_path in paths:  # For each path in the dataset's paths list
            if not verify_filepath_exists(input_path):  # If the input path does not exist
                verbose_output(
                    f"{BackgroundColors.YELLOW}Skipping missing path: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}"
                )
                continue  # Skip to the next path if the current one doesn't exist

            # Determine files to process; allow CLI CSV override
            if CSV_FILE:  # If a specific CSV file is provided via CLI
                try:
                    abs_csv = os.path.abspath(CSV_FILE)
                    abs_input = os.path.abspath(input_path)
                    if abs_csv.startswith(abs_input):
                        files_to_process = [CSV_FILE]
                    else:
                        # CSV override does not belong to this path; skip
                        files_to_process = []
                except Exception:
                    files_to_process = []
            else:
                files_to_process = get_files_to_process(
                    input_path, file_extension=".csv"
                )  # Get list of CSV files to process

            local_dataset_name = dataset_name or get_dataset_name(
                input_path
            )  # Use provided dataset name or infer from path

            combined_df = None
            combined_file_for_features = None

            if PROCESS_ENTIRE_DATASET and len(files_to_process) > 1:
                result = combine_dataset_files(files_to_process)
                if result is not None:
                    combined_df, combined_target_col = result
                    combined_file_for_features = files_to_process[0]
                    files_to_process = ["combined"]
                else:
                    print(f"{BackgroundColors.YELLOW}Warning: Could not combine dataset files. Processing individually.{Style.RESET_ALL}")
                    # files_to_process remains as is
            for file in files_to_process:  # For each file to process
                print(
                    f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}"
                )
                print(
                    f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing file: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}"
                )
                print(
                    f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}\n"
                )

                # Load feature selection results (same for all runs)
                file_for_features = combined_file_for_features if file == "combined" else file
                ga_selected_features, pca_n_components, rfe_selected_features = load_feature_selection_results(
                    file_for_features
                )  # Load feature selection results

                # Load original dataset
                if file == "combined":
                    df_original = combined_df
                else:
                    df_original = load_dataset(file)  # Load the original dataset

                if df_original is None:  # If the dataset failed to load
                    verbose_output(
                        f"{BackgroundColors.RED}Failed to load dataset from: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}"
                    )  # Output the failure message
                    continue  # Skip to the next file if loading failed

                df_original_cleaned = preprocess_dataframe(df_original)  # Preprocess the DataFrame

                if df_original_cleaned is None or df_original_cleaned.empty:  # If the DataFrame is None or empty after preprocessing
                    print(
                        f"{BackgroundColors.RED}Dataset {BackgroundColors.CYAN}{file}{BackgroundColors.RED} empty after preprocessing. Skipping.{Style.RESET_ALL}"
                    )
                    continue  # Skip to the next file if preprocessing failed

                feature_names = df_original_cleaned.select_dtypes(include=np.number).iloc[:, :-1].columns.tolist()  # Get feature names

                # Get base models (same for all runs)
                base_models = get_models()  # Get the base models

                # Prepare hyperparameters mapping (may remain empty)
                hp_params_map = {}
                hp_results_raw = extract_hyperparameter_optimization_results(
                    file
                )  # Extract hyperparameter optimization results
                if hp_results_raw:  # If results were found, extract the params mapping and apply
                    hp_params_map = {
                        k: (v.get("best_params") if isinstance(v, dict) else v) for k, v in hp_results_raw.items()
                    }  # Extract only the best_params mapping
                    base_models = apply_hyperparameters_to_models(
                        hp_params_map, base_models
                    )  # Apply hyperparameters to base models

                # ALWAYS evaluate on original data first
                print(
                    f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}[1/3] Evaluating on ORIGINAL data{Style.RESET_ALL}"
                )
                results_original = evaluate_on_dataset(
                    file,
                    df_original_cleaned,
                    feature_names,
                    ga_selected_features,
                    pca_n_components,
                    rfe_selected_features,
                    base_models,
                    data_source_label="Original",
                    hyperparams_map=hp_params_map,
                )

                # Save original results
                original_results_list = list(results_original.values())
                save_stacking_results(file, original_results_list)

                # If TEST_DATA_AUGMENTATION is enabled, also evaluate on augmented data
                if TEST_DATA_AUGMENTATION:
                    augmented_file = find_data_augmentation_file(file)
                    
                    if augmented_file is not None:
                        print(
                            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}[2/3] Evaluating on AUGMENTED data{Style.RESET_ALL}"
                        )
                        
                        df_augmented = load_dataset(augmented_file)
                        
                        if df_augmented is not None:
                            df_augmented_cleaned = preprocess_dataframe(df_augmented)
                            
                            if df_augmented_cleaned is not None and not df_augmented_cleaned.empty:
                                # Evaluate on augmented data only
                                results_augmented = evaluate_on_dataset(
                                    file,
                                    df_augmented_cleaned,
                                    feature_names,
                                    ga_selected_features,
                                    pca_n_components,
                                    rfe_selected_features,
                                    base_models,
                                    data_source_label="Augmented",
                                    hyperparams_map=hp_params_map,
                                )
                                
                                # Merge original + augmented data
                                print(
                                    f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}[3/3] Evaluating on ORIGINAL + AUGMENTED data{Style.RESET_ALL}"
                                )
                                df_merged = merge_original_and_augmented(df_original_cleaned, df_augmented_cleaned)
                                
                                results_merged = evaluate_on_dataset(
                                    file,
                                    df_merged,
                                    feature_names,
                                    ga_selected_features,
                                    pca_n_components,
                                    rfe_selected_features,
                                    base_models,
                                    data_source_label="Original+Augmented",
                                    hyperparams_map=hp_params_map,
                                )
                                
                                # Generate comparison report
                                print(
                                    f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*100}{Style.RESET_ALL}"
                                )
                                print(
                                    f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}DATA AUGMENTATION COMPARISON REPORT{Style.RESET_ALL}"
                                )
                                print(
                                    f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*100}{Style.RESET_ALL}\n"
                                )
                                
                                comparison_results = []
                                
                                # Compare metrics for each feature_set and model
                                for key in results_original.keys():
                                    orig_result = results_original[key]
                                    aug_result = results_augmented.get(key)
                                    merged_result = results_merged.get(key)
                                    
                                    feature_set = orig_result["feature_set"]
                                    model_name = orig_result["model_name"]
                                    classifier_type = orig_result["classifier_type"]
                                    
                                    # Extract metrics (accuracy, precision, recall, f1, fpr, fnr, time)
                                    orig_metrics = [
                                        orig_result.get("accuracy", 0),
                                        orig_result.get("precision", 0),
                                        orig_result.get("recall", 0),
                                        orig_result.get("f1_score", 0),
                                        orig_result.get("fpr", 0),
                                        orig_result.get("fnr", 0),
                                        orig_result.get("elapsed_time_s", 0),
                                    ]
                                    aug_metrics = [
                                        aug_result.get("accuracy", 0) if aug_result else 0,
                                        aug_result.get("precision", 0) if aug_result else 0,
                                        aug_result.get("recall", 0) if aug_result else 0,
                                        aug_result.get("f1_score", 0) if aug_result else 0,
                                        aug_result.get("fpr", 0) if aug_result else 0,
                                        aug_result.get("fnr", 0) if aug_result else 0,
                                        aug_result.get("elapsed_time_s", 0) if aug_result else 0,
                                    ]
                                    merged_metrics = [
                                        merged_result.get("accuracy", 0) if merged_result else 0,
                                        merged_result.get("precision", 0) if merged_result else 0,
                                        merged_result.get("recall", 0) if merged_result else 0,
                                        merged_result.get("f1_score", 0) if merged_result else 0,
                                        merged_result.get("fpr", 0) if merged_result else 0,
                                        merged_result.get("fnr", 0) if merged_result else 0,
                                        merged_result.get("elapsed_time_s", 0) if merged_result else 0,
                                    ]
                                    
                                    # Calculate improvements (Original+Augmented vs Original)
                                    acc_improvement = calculate_metric_improvement(orig_metrics[0], merged_metrics[0])
                                    prec_improvement = calculate_metric_improvement(orig_metrics[1], merged_metrics[1])
                                    rec_improvement = calculate_metric_improvement(orig_metrics[2], merged_metrics[2])
                                    f1_improvement = calculate_metric_improvement(orig_metrics[3], merged_metrics[3])
                                    # For FPR and FNR, lower is better, so negative change means improvement
                                    fpr_improvement = calculate_metric_improvement(orig_metrics[4], merged_metrics[4])
                                    fnr_improvement = calculate_metric_improvement(orig_metrics[5], merged_metrics[5])
                                    # For training time, lower is better, so negative change means improvement
                                    time_improvement = calculate_metric_improvement(orig_metrics[6], merged_metrics[6])
                                    
                                    # Print detailed comparison
                                    print(
                                        f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Feature Set: {BackgroundColors.CYAN}{feature_set}{BackgroundColors.GREEN} | Model: {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}"
                                    )
                                    print(f"  {BackgroundColors.YELLOW}Accuracy:{Style.RESET_ALL}")
                                    print(f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {truncate_value(orig_metrics[0])} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {truncate_value(aug_metrics[0])} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {truncate_value(merged_metrics[0])} | {BackgroundColors.CYAN}Improvement: {acc_improvement:+.2f}%{Style.RESET_ALL}")
                                    print(f"  {BackgroundColors.YELLOW}Precision:{Style.RESET_ALL}")
                                    print(f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {truncate_value(orig_metrics[1])} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {truncate_value(aug_metrics[1])} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {truncate_value(merged_metrics[1])} | {BackgroundColors.CYAN}Improvement: {prec_improvement:+.2f}%{Style.RESET_ALL}")
                                    print(f"  {BackgroundColors.YELLOW}Recall:{Style.RESET_ALL}")
                                    print(f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {truncate_value(orig_metrics[2])} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {truncate_value(aug_metrics[2])} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {truncate_value(merged_metrics[2])} | {BackgroundColors.CYAN}Improvement: {rec_improvement:+.2f}%{Style.RESET_ALL}")
                                    print(f"  {BackgroundColors.YELLOW}F1-Score:{Style.RESET_ALL}")
                                    print(f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {truncate_value(orig_metrics[3])} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {truncate_value(aug_metrics[3])} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {truncate_value(merged_metrics[3])} | {BackgroundColors.CYAN}Improvement: {f1_improvement:+.2f}%{Style.RESET_ALL}")
                                    print(f"  {BackgroundColors.YELLOW}FPR (lower is better):{Style.RESET_ALL}")
                                    print(f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {truncate_value(orig_metrics[4])} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {truncate_value(aug_metrics[4])} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {truncate_value(merged_metrics[4])} | {BackgroundColors.CYAN}Change: {fpr_improvement:+.2f}%{Style.RESET_ALL}")
                                    print(f"  {BackgroundColors.YELLOW}FNR (lower is better):{Style.RESET_ALL}")
                                    print(f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {truncate_value(orig_metrics[5])} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {truncate_value(aug_metrics[5])} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {truncate_value(merged_metrics[5])} | {BackgroundColors.CYAN}Change: {fnr_improvement:+.2f}%{Style.RESET_ALL}")
                                    print(f"  {BackgroundColors.YELLOW}Training Time (seconds, lower is better):{Style.RESET_ALL}")
                                    print(f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {orig_metrics[6]:.2f}s | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {aug_metrics[6]:.2f}s | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {merged_metrics[6]:.2f}s | {BackgroundColors.CYAN}Change: {time_improvement:+.2f}%{Style.RESET_ALL}\n")
                                    
                                    # Store comparison results for CSV export
                                    comparison_results.append({
                                        "dataset": orig_result["dataset"],
                                        "feature_set": feature_set,
                                        "classifier_type": classifier_type,
                                        "model_name": model_name,
                                        "data_source": "Original",
                                        "n_features": orig_result["n_features"],
                                        "n_samples_train": orig_result["n_samples_train"],
                                        "n_samples_test": orig_result["n_samples_test"],
                                        "accuracy": orig_metrics[0],
                                        "precision": orig_metrics[1],
                                        "recall": orig_metrics[2],
                                        "f1_score": orig_metrics[3],
                                        "fpr": orig_metrics[4],
                                        "fnr": orig_metrics[5],
                                        "training_time": orig_metrics[6],
                                        "accuracy_improvement": 0.0,
                                        "precision_improvement": 0.0,
                                        "recall_improvement": 0.0,
                                        "f1_score_improvement": 0.0,
                                        "fpr_improvement": 0.0,
                                        "fnr_improvement": 0.0,
                                        "training_time_improvement": 0.0,
                                        "features_list": orig_result["features_list"],
                                    })
                                    
                                    comparison_results.append({
                                        "dataset": orig_result["dataset"],
                                        "feature_set": feature_set,
                                        "classifier_type": classifier_type,
                                        "model_name": model_name,
                                        "data_source": "Augmented",
                                        "n_features": aug_result["n_features"] if aug_result else 0,
                                        "n_samples_train": aug_result["n_samples_train"] if aug_result else 0,
                                        "n_samples_test": aug_result["n_samples_test"] if aug_result else 0,
                                        "accuracy": aug_metrics[0],
                                        "precision": aug_metrics[1],
                                        "recall": aug_metrics[2],
                                        "f1_score": aug_metrics[3],
                                        "fpr": aug_metrics[4],
                                        "fnr": aug_metrics[5],
                                        "training_time": aug_metrics[6],
                                        "accuracy_improvement": 0.0,
                                        "precision_improvement": 0.0,
                                        "recall_improvement": 0.0,
                                        "f1_score_improvement": 0.0,
                                        "fpr_improvement": 0.0,
                                        "fnr_improvement": 0.0,
                                        "training_time_improvement": 0.0,
                                        "features_list": orig_result["features_list"],
                                    })
                                    
                                    comparison_results.append({
                                        "dataset": orig_result["dataset"],
                                        "feature_set": feature_set,
                                        "classifier_type": classifier_type,
                                        "model_name": model_name,
                                        "data_source": "Original+Augmented",
                                        "n_features": merged_result["n_features"] if merged_result else 0,
                                        "n_samples_train": merged_result["n_samples_train"] if merged_result else 0,
                                        "n_samples_test": merged_result["n_samples_test"] if merged_result else 0,
                                        "accuracy": merged_metrics[0],
                                        "precision": merged_metrics[1],
                                        "recall": merged_metrics[2],
                                        "f1_score": merged_metrics[3],
                                        "fpr": merged_metrics[4],
                                        "fnr": merged_metrics[5],
                                        "training_time": merged_metrics[6],
                                        "accuracy_improvement": acc_improvement,
                                        "precision_improvement": prec_improvement,
                                        "recall_improvement": rec_improvement,
                                        "f1_score_improvement": f1_improvement,
                                        "fpr_improvement": fpr_improvement,
                                        "fnr_improvement": fnr_improvement,
                                        "training_time_improvement": time_improvement,
                                        "features_list": orig_result["features_list"],
                                    })
                                
                                # Save comparison results
                                save_augmentation_comparison_results(file, comparison_results)
                                
                                print(
                                    f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}✓ Data augmentation comparison complete!{Style.RESET_ALL}"
                                )
                            else:
                                print(
                                    f"{BackgroundColors.YELLOW}Warning: Augmented dataset empty after preprocessing. Skipping augmentation comparison.{Style.RESET_ALL}"
                                )
                        else:
                            print(
                                f"{BackgroundColors.YELLOW}Warning: Failed to load augmented dataset. Skipping augmentation comparison.{Style.RESET_ALL}"
                            )
                    else:
                        print(
                            f"\n{BackgroundColors.YELLOW}No augmented data found for this file. Skipping augmentation comparison.{Style.RESET_ALL}"
                        )

    finish_time = datetime.datetime.now()  # Get the finish time of the program
    print(
        f"\n{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
    )  # Output the start and finish times
    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
    )  # Output the end of the program message

    send_telegram_message(TELEGRAM_BOT, [f"Finished Classifiers Stacking at {finish_time.strftime('%Y-%m-%d %H:%M:%S')} | Execution time: {calculate_execution_time(start_time, finish_time)}"])  # Send Telegram message indicating finish

    (
        atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None
    )  # Register the play_sound function to be called when the program finishes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run stacking pipeline and optionally load existing exported models."
    )
    parser.add_argument(
        "--skip-train-if-model-exists",
        dest="skip_train",
        action="store_true",
        help="If set, do not retrain; load existing exported artifacts and evaluate.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose output during the run.",
    )
    parser.add_argument(
        "--csv",
        dest="csv",
        type=str,
        default=None,
        help="Optional: path to dataset CSV to analyze. If omitted, uses the default in DATASETS.",
    )
    args = parser.parse_args()

    # Override module-level constants based on CLI flags
    SKIP_TRAIN_IF_MODEL_EXISTS = bool(args.skip_train)
    VERBOSE = bool(args.verbose)
    CSV_FILE = args.csv if args.csv else CSV_FILE

    main()  # Call the main function
