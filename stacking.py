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
