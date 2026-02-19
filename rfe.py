"""
================================================================================
Recursive Feature Elimination (RFE) Automation and Feature Analysis Tool (rfe.py)
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-10-07
Description :
    Utility to automate Recursive Feature Elimination (RFE) workflows for
    structured classification datasets. The module bundles safe dataset
    loading, preprocessing, scaling, RFE selection (Random Forest base),
    evaluation and export of structured run results for reproducible analysis.

Core features:
    - Safe CSV loading with column sanitization and basic validation
    - Z-score standardization of numeric features prior to RFE
    - RFE using RandomForestClassifier (configurable number of selected features)
    - Performance evaluation (accuracy, precision, recall, F1, FPR, FNR)
    - Export of run results to `Feature_Analysis/RFE_Run_Results.csv` with hardware metadata
    - Portable: skips platform-specific features on unsupported OS (e.g., sound on Windows)

Usage:
    - Set `csv_file` in `main()` or call `run_rfe(csv_path)` programmatically.
    - Run: `python3 rfe.py` or via the project Makefile target.

Outputs:
    - `Feature_Analysis/RFE_Run_Results.csv` summarizing each run
    - Per-run JSON fields for `top_features` and `rfe_ranking` inside the CSV

Notes & conventions:
    - The last column of the input CSV is treated as the target variable.
    - Only numeric columns (or coercible-to-numeric) are used for feature ranking.
    - Defaults: 80/20 train-test split, RandomForest with 100 trees, fixed seed
    - Toggle `VERBOSE = True` for detailed runtime logs useful during debugging

TODOs:
    - Add CLI argument parsing for dataset path, `n_select`, and parallel runs.
    - Support additional estimators (SVM, Gradient Boosting) and compare results.
    - Integrate automatic handling for categorical and missing data.
    - Add unit tests for preprocessing and metric computations.

Dependencies:
    - Python >= 3.9
    - pandas, numpy, scikit-learn, matplotlib, colorama
"""


import argparse  # For command-line argument parsing
import atexit  # For playing a sound when the program finishes
import csv  # For CSV quoting options
import dataframe_image as dfi  # For exporting DataFrame images (zebra-striped PNG)
import datetime  # For timestamping
import glob  # For finding exported model files
import json  # For saving lists and dicts as JSON strings
import math  # For mathematical operations
import numpy as np  # For numerical operations
import os  # For file and directory operations
import pandas as pd  # For data manipulation
import platform  # For getting the operating system name
import psutil  # For hardware information
import re  # For regular expressions
import subprocess  # For executing system commands
import sys  # For system-specific parameters and functions
import telegram_bot as telegram_module  # For setting Telegram prefix and device info
import time  # For measuring elapsed time
import yaml  # For loading configuration from YAML files
from colorama import Style  # For coloring the terminal
from joblib import dump, load  # For exporting and loading trained models and scalers
from Logger import Logger  # For logging output to both terminal and file
from pathlib import Path  # For handling file paths
from sklearn.ensemble import RandomForestClassifier  # For the Random Forest model
from sklearn.feature_selection import RFE  # For Recursive Feature Elimination
from sklearn.metrics import (  # For performance metrics
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split  # For train/test split and stratified K-Fold CV
from sklearn.preprocessing import StandardScaler  # For scaling the data (standardization)
from telegram_bot import TelegramBot, send_exception_via_telegram, send_telegram_message, setup_global_exception_hook  # For sending progress messages to Telegram
from typing import Any, Dict, Optional, Union  # For type hinting


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
SKIP_TRAIN_IF_MODEL_EXISTS = False  # If True, try loading exported models instead of retraining
CSV_FILE = "./Datasets/CICDDoS2019/01-12/DrDoS_DNS.csv"  # Path to the CSV dataset file (set in main)
RFE_RESULTS_CSV_COLUMNS = [  # Columns for the RFE results CSV
    "timestamp",
    "tool",
    "model",
    "dataset",
    "hyperparameters",
    "cv_method",
    "train_test_split",
    "scaling",
    "cv_accuracy",
    "cv_precision",
    "cv_recall",
    "cv_f1_score",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1_score",
    "test_fpr",
    "test_fnr",
    "feature_extraction_time_s",
    "training_time_s",
    "testing_time_s",
    "hardware",
    "top_features",
    "rfe_ranking",
]

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

# Functions Definitions:


setup_global_exception_hook()  # Set up global exception hook to send exceptions via Telegram


def verbose_output(true_string="", false_string=""):
    """
    Outputs a message if the VERBOSE constant is set to True.

    :param true_string: The string to be outputted if the VERBOSE constant is set to True.
    :param false_string: The string to be outputted if the VERBOSE constant is set to False.
    :return: None
    """
    
    try:
        if VERBOSE and true_string != "":  # If VERBOSE is True and a true_string was provided
            print(true_string)  # Output the true statement string
        elif false_string != "":  # If a false_string was provided
            print(false_string)  # Output the false statement string
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_config(config_path=None):
    """
    Load RFE-related configuration from config.yaml or config.yaml.example.

    Precedence: CLI args (handled in __main__) > config file > hard-coded defaults.

    :param config_path: Optional path to a YAML config file
    :return: dict with keys under 'rfe' merged with defaults
    """
    
    try:
        defaults = {
            "rfe": {
                "n_features_to_select": None,  # None means auto / majority selection
                "step": 1,
                "estimator": "random_forest",
                "random_state": 42,
            },
        }

        cfg_file = None
        if config_path:
            cfg_file = Path(config_path)
        else:
            candidate = Path(__file__).parent / "config.yaml"
            candidate_example = Path(__file__).parent / "config.yaml.example"
            if candidate.exists():
                cfg_file = candidate
            elif candidate_example.exists():
                cfg_file = candidate_example

        if cfg_file is None or not cfg_file.exists():
            verbose_output(f"RFE config file not found, using defaults: {defaults['rfe']}")
            return defaults

        with open(cfg_file, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}

        cfg = dict(defaults)
        if "rfe" in loaded and isinstance(loaded["rfe"], dict):
            cfg["rfe"].update(loaded["rfe"])

        return cfg
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def verify_dot_env_file():
    """
    Verifies if the .env file exists in the current directory.

    :return: True if the .env file exists, False otherwise
    """
    
    try:
        env_path = Path(__file__).parent / ".env"  # Path to the .env file
        if not env_path.exists():  # If the .env file does not exist
            print(f"{BackgroundColors.CYAN}.env{BackgroundColors.YELLOW} file not found at {BackgroundColors.CYAN}{env_path}{BackgroundColors.YELLOW}. Telegram messages may not be sent.{Style.RESET_ALL}")
            return False  # Return False

        return True  # Return True if the .env file exists
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def setup_telegram_bot():
    """
    Sets up the Telegram bot for progress messages.

    :return: None
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Setting up Telegram bot for messages...{Style.RESET_ALL}"
        )  # Output the verbose message

        verify_dot_env_file()  # Verify if the .env file exists

        global TELEGRAM_BOT  # Declare the module-global TELEGRAM_BOT variable

        try:  # Try to initialize the Telegram bot
            TELEGRAM_BOT = TelegramBot()  # Initialize Telegram bot for progress messages
            telegram_module.TELEGRAM_DEVICE_INFO = f"{telegram_module.get_local_ip()} - {platform.system()}"
            telegram_module.RUNNING_CODE = os.path.basename(__file__)
        except Exception as e:
            print(f"{BackgroundColors.RED}Failed to initialize Telegram bot: {e}{Style.RESET_ALL}")
            TELEGRAM_BOT = None  # Set to None if initialization fails
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def safe_filename(name):
    """
    Converts a string to a safe filename by replacing invalid characters with underscores.

    :param name: The original string
    :return: A safe filename string
    """
    
    try:
        return re.sub(r'[\\/*?:"<>|]', "_", name)  # Replace invalid characters with underscores
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


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
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def load_dataset(csv_path):
    """
    Load CSV and return DataFrame.

    :param csv_path: Path to CSV dataset.
    :return: DataFrame
    """
    
    try:
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
        
        send_telegram_message(TELEGRAM_BOT, [f"Dataset loaded from: {csv_path}"])  # Send message about dataset loading

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


def preprocess_dataframe(df, remove_zero_variance=True):
    """
    Preprocess a DataFrame by removing rows with NaN or infinite values and
    dropping zero-variance numeric features.

    :param df: pandas DataFrame to preprocess
    :param remove_zero_variance: whether to drop numeric columns with zero variance
    :return: cleaned DataFrame
    """
    
    try:
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
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def scale_and_split(X, y, test_size=0.2, random_state=42):
    """
    Scales numeric features and splits into train/test sets.

    :param X: Features DataFrame
    :param y: Target Series
    :param test_size: Proportion of the dataset to include in the test split
    :param random_state: Random seed for reproducibility
    :return: X_train, X_test, y_train, y_test, feature_columns
    """
    
    try:
        scaler = StandardScaler()  # Initialize the scaler

        X_numeric = X.select_dtypes(include=["number"]).copy()  # Pick numeric columns first
        if X_numeric.shape[1] == 0:  # No numeric columns detected
            coerced_cols = {}  # Dictionary to hold coerced numeric columns
            for col in X.columns:  # Try coercing each column to numeric
                coerced = pd.to_numeric(X[col], errors="coerce")  # Coerce invalid -> NaN
                if coerced.notna().sum() > 0:  # Keep columns that produced numeric values
                    coerced_cols[col] = coerced
            if coerced_cols:  # Build DataFrame from coerced columns
                X_numeric = pd.DataFrame(coerced_cols, index=X.index)  # Use original index
            else:  # Nothing numeric available -> cannot proceed
                raise ValueError(
                    "No numeric features found after preprocessing. Ensure the dataset contains numeric columns for RFE."
                )

        X_scaled = scaler.fit_transform(X_numeric.values)  # Scale the numeric array

        stratify_param = y if len(np.unique(y)) > 1 else None  # Avoid stratify for constant labels
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )  # Split into train/test sets

        return X_train, X_test, y_train, y_test, X_numeric.columns  # Return the split data and feature columns
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_rfe_selector(X_train, y_train, n_select=10, step=1, estimator_name="random_forest", random_state=42):
    """
    Runs RFE with RandomForestClassifier and returns the selector object.

    :param X_train: Training features
    :param y_train: Training target
    :param n_select: Number of features to select
    :param random_state: Random seed for reproducibility
    :return: selector (fitted RFE object), model, feature_extraction_time_s
    """
    
    try:
        estimator_name_l = (estimator_name or "random_forest").lower()
        if "random" in estimator_name_l:
            model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=N_JOBS)
        else:
            raise ValueError(f"Unsupported estimator '{estimator_name}'. Supported: 'random_forest'")

        n_features = X_train.shape[1]  # Get the number of features
        if n_select is None:
            n_select = n_features  # default to all (caller may reduce later)
        n_select = int(n_select)
        if n_select <= 0:
            raise ValueError(f"n_features_to_select must be > 0, got {n_select}")
        n_select = n_select if n_features >= n_select else n_features  # Adjust n_select if more than available features

        selector = RFE(model, n_features_to_select=n_select, step=int(step))  # Initialize RFE
        sel_start = time.perf_counter()  # Start perf_counter for selector fitting
        selector = selector.fit(X_train, y_train)  # Fit RFE (feature selection) as part of feature extraction
        sel_end = time.perf_counter()  # End perf_counter for selector fitting
        feature_extraction_time_s = round(sel_end - sel_start, 6)  # Compute selector fit duration rounded to 6 decimals

        return selector, model, feature_extraction_time_s  # Return the fitted selector, model and feature extraction time
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def compute_rfe_metrics(selector, X_train, X_test, y_train, y_test, random_state=42, estimator_name="random_forest"):
    """
    Computes performance metrics using the RFE-selected features.

    :param selector: Fitted RFE object
    :param X_train: Training features
    :param X_test: Testing features
    :param y_train: Training target
    :param y_test: Testing target
    :param random_state: Random seed for reproducibility
    :return: metrics tuple (acc, prec, rec, f1, fpr, fnr, training_time_s, testing_time_s)
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Computing performance metrics using RFE-selected features...{Style.RESET_ALL}"
        )  # Output the verbose message

        support = selector.support_  # Get the mask of selected features
        X_train_selected = X_train[:, support]  # Select training features
        X_test_selected = X_test[:, support]  # Select testing features

        estimator_name_l = (estimator_name or "random_forest").lower()
        if "random" in estimator_name_l:
            model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=N_JOBS)  # Initialize the model
        else:
            raise ValueError(f"Unsupported estimator '{estimator_name}' for compute_rfe_metrics. Supported: 'random_forest'")

        train_start = time.perf_counter()  # Start perf_counter immediately before model.fit (training window)
        model.fit(X_train_selected, y_train)  # Fit the model on selected features (training)
        train_end = time.perf_counter()  # End perf_counter immediately after model.fit
        training_time_s = round(train_end - train_start, 6)  # Compute training time rounded to 6 decimals

        test_start = time.perf_counter()  # Start perf_counter immediately before prediction (testing window)
        y_pred = model.predict(X_test_selected)  # Predict on selected test features (testing)
        acc = accuracy_score(y_test, y_pred)  # Calculate accuracy
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate precision
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate recall
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate F1-score

        if len(np.unique(y_test)) == 2:  # If binary classification
            cm = confusion_matrix(y_test, y_pred)  # Confusion matrix for observed labels
            if cm.shape == (2, 2):  # Expect 2x2 matrix for binary
                tn, fp, fn, tp = cm.ravel()  # Unpack
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Calculate false positive rate
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # Calculate false negative rate
            else:  # Fallback: compute rates from sums if unexpected shape
                total = cm.sum() if cm.size > 0 else 1
                fpr = float(cm.sum() - np.trace(cm)) / float(total) if total > 0 else 0
                fnr = fpr  # Fallback estimate when binary layout is unexpected
        else:  # For multi-class classification
            cm = confusion_matrix(y_test, y_pred)  # Confusion matrix for observed labels
            supports = cm.sum(axis=1)  # Support for each class
            fprs = []  # List to hold per-class FPR
            fnrs = []  # List to hold per-class FNR
            for i in range(cm.shape[0]):  # For each class
                tp = cm[i, i]  # True positives for class i
                fn = cm[i, :].sum() - tp  # False negatives: actual i but predicted not-i
                fp = cm[:, i].sum() - tp  # False positives: predicted i but actual not-i
                tn = cm.sum() - (tp + fp + fn)  # True negatives: everything else
                denom_fnr = (tp + fn) if (tp + fn) > 0 else 1  # Denominator for FNR (avoid div0)
                denom_fpr = (fp + tn) if (fp + tn) > 0 else 1  # Denominator for FPR (avoid div0)
                fnr_i = fn / denom_fnr  # Per-class false negative rate
                fpr_i = fp / denom_fpr  # Per-class false positive rate
                fprs.append((fpr_i, supports[i]))  # Store FPR with class support for weighting
                fnrs.append((fnr_i, supports[i]))  # Store FNR with class support for weighting
            total_support = float(supports.sum()) if supports.sum() > 0 else 1.0  # Total support across classes
            fpr = float(sum(v * s for v, s in fprs) / total_support)  # Weighted average FPR across classes
            fnr = float(sum(v * s for v, s in fnrs) / total_support)  # Weighted average FNR across classes
        test_end = time.perf_counter()  # End perf_counter immediately after metrics computed
        testing_time_s = round(test_end - test_start, 6)  # Compute testing time rounded to 6 decimals

        return (
            float(acc),  # Accuracy
            float(prec),  # Precision
            float(rec),  # Recall
            float(f1),  # F1-score
            float(fpr),  # False positive rate
            float(fnr),  # False negative rate
            float(training_time_s),  # Training time in seconds (rounded 6 decimals)
            float(testing_time_s),  # Testing time in seconds (rounded 6 decimals)
        )  # Return the metrics as Python floats
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_top_features(selector, X_columns):
    """
    Returns top selected features and their RFE rankings.

    :param selector: Fitted RFE object
    :param X_columns: Original feature column names
    :return: top_features list, rfe_ranking dict
    """
    
    try:
        rfe_ranking = {
            f: r for f, r in zip(X_columns, selector.ranking_)
        }  # Map normalized feature names to their RFE rankings
        rfe_ranking = {k: int(v) for k, v in rfe_ranking.items()}  # Convert numpy types to Python int
        top_features = [f for f, s in zip(X_columns, selector.support_) if s]  # List of top selected features

        return top_features, rfe_ranking  # Return the top features and their rankings
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def print_top_features(top_features, rfe_ranking):
    """
    Prints top features and their RFE rankings to the terminal.

    :param top_features: List of top features
    :param rfe_ranking: Dict mapping normalized feature names to RFE rankings
    """
    
    try:
        print(f"\n{BackgroundColors.BOLD}Top {len(top_features)} features selected by RFE:{Style.RESET_ALL}")

        for i, feat in enumerate(top_features, start=1):  # Print each top feature with its ranking
            rank_info = (
                f" {BackgroundColors.GREEN}(RFE ranking {BackgroundColors.CYAN}{rfe_ranking[feat]}{Style.RESET_ALL})"
                if feat in rfe_ranking
                else " (RFE ranking N/A)"
            )  # Get ranking info
            print(f"{i}. {feat}{rank_info}")  # Print the feature and its ranking
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def print_metrics(metrics_tuple):
    """
    Prints metrics for the current run to the terminal.

    :param metrics_tuple: Tuple of average metrics
    """
    
    try:
        print(f"\n{BackgroundColors.BOLD}Average Metrics:{Style.RESET_ALL}")
        print(f"  {BackgroundColors.GREEN}Accuracy: {BackgroundColors.CYAN}{truncate_value(metrics_tuple[0])}{Style.RESET_ALL}")
        print(f"  {BackgroundColors.GREEN}Precision: {BackgroundColors.CYAN}{truncate_value(metrics_tuple[1])}{Style.RESET_ALL}")
        print(f"  {BackgroundColors.GREEN}Recall: {BackgroundColors.CYAN}{truncate_value(metrics_tuple[2])}{Style.RESET_ALL}")
        print(f"  {BackgroundColors.GREEN}F1-Score: {BackgroundColors.CYAN}{truncate_value(metrics_tuple[3])}{Style.RESET_ALL}")
        print(
            f"  {BackgroundColors.GREEN}False Positive Rate (FPR): {BackgroundColors.CYAN}{truncate_value(metrics_tuple[4])}{Style.RESET_ALL}"
        )
        print(
            f"  {BackgroundColors.GREEN}False Negative Rate (FNR): {BackgroundColors.CYAN}{truncate_value(metrics_tuple[5])}{Style.RESET_ALL}"
        )
        try:
            if len(metrics_tuple) >= 8:  # If tuple has training and testing times
                displayed_elapsed = int(round(float(metrics_tuple[6]) + float(metrics_tuple[7])))  # Sum training+testing
            else:
                displayed_elapsed = int(round(float(metrics_tuple[6])))  # Fallback to the single elapsed value
        except Exception:
            displayed_elapsed = 0  # On error, show zero

        print(f"  {BackgroundColors.GREEN}Elapsed Time: {BackgroundColors.CYAN}{displayed_elapsed}s{Style.RESET_ALL}")
        
        send_telegram_message(
            TELEGRAM_BOT,
            [
                f"Average Metrics:\n"
                f"  Accuracy: {truncate_value(metrics_tuple[0])}\n"
                f"  Precision: {truncate_value(metrics_tuple[1])}\n"
                f"  Recall: {truncate_value(metrics_tuple[2])}\n"
                f"  F1-Score: {truncate_value(metrics_tuple[3])}\n"
                f"  False Positive Rate (FPR): {truncate_value(metrics_tuple[4])}\n"
                f"  False Negative Rate (FNR): {truncate_value(metrics_tuple[5])}\n"
                f"  Elapsed Time: {displayed_elapsed}s"
            ],
        )  # Send metrics to Telegram
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


def populate_hardware_column_and_order(df, column_name="hardware"):
    """
    Add a hardware-specs column to `df` and reorder columns so the hardware
    column appears immediately after `elapsed_time_s`.

    :param df: pandas DataFrame with RFE results
    :param column_name: name for the hardware column
    :return: reordered DataFrame with hardware column added
    """
    
    try:
        df_results = df.copy()  # Copy DataFrame to modify
        hardware_specs = get_hardware_specifications()  # Get system specs
        df_results[column_name] = (
            hardware_specs["cpu_model"]
            + " | Cores: "
            + str(hardware_specs["cores"])
            + " | RAM: "
            + str(hardware_specs["ram_gb"])
            + " GB | OS: "
            + hardware_specs["os"]
        )  # Add hardware specs column

        columns_order = [
            (column_name if str(c).lower() == "hardware" else c) for c in RFE_RESULTS_CSV_COLUMNS
        ]

        return df_results.reindex(columns=columns_order)  # Reorder columns
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def export_final_model(X_numeric, feature_columns, top_features, y_array, csv_path):
    """
    Train a final RandomForest on the full numeric dataset restricted to
    `top_features`, save model, scaler, and feature list to disk and
    return their paths.

    All newly added lines include inline comments explaining their purpose.
    
    :param X_numeric: DataFrame of numeric features
    :param feature_columns: Original feature column names
    :param top_features: List of top selected features
    :param y_array: Numpy array of target labels
    :param csv_path: Original CSV file path
    :return: model_path, scaler_path, features_path
    """
    
    try:
        scaler_full = StandardScaler()  # Create a scaler for full-data training
        X_full_scaled = scaler_full.fit_transform(X_numeric.values)  # Scale all numeric features
        sel_indices = [i for i, f in enumerate(feature_columns) if f in top_features]  # Get indices for top features
        X_final = X_full_scaled[:, sel_indices] if sel_indices else X_full_scaled  # Select columns or keep all if none
        final_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=N_JOBS)  # Instantiate final RF
        final_model.fit(X_final, y_array)  # Fit final model on entire dataset using selected features

        models_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis/RFE/Models/"  # Models output directory under RFE/Models subdir
        os.makedirs(models_dir, exist_ok=True)  # Ensure directory exists
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")  # Timestamp for filenames (YYYY_MM_DD-HH_MM_SS)
        base_name = safe_filename(Path(csv_path).stem)  # Safe base name from dataset path
        model_path = f"{models_dir}RFE-{base_name}-{timestamp}-model.joblib"  # Model file path
        scaler_path = f"{models_dir}RFE-{base_name}-{timestamp}-scaler.joblib"  # Scaler file path
        features_path = f"{models_dir}RFE-{base_name}-{timestamp}-features.json"  # Selected features file path
        params_path = f"{models_dir}RFE-{base_name}-{timestamp}-params.json"  # Hyperparameters file path
        dump(final_model, model_path)  # Save trained model to disk
        dump(scaler_full, scaler_path)  # Save fitted scaler to disk
        
        with open(features_path, "w", encoding="utf-8") as fh:  # Write selected features to json
            fh.write(json.dumps(top_features))  # Save feature list as JSON
            
        model_params = final_model.get_params()  # Get hyperparameters from trained estimator
        with open(params_path, "w", encoding="utf-8") as ph:  # Write params to json
            ph.write(json.dumps(model_params, default=str))  # Save params as JSON (default=str for non-serializable)

        print(f"{BackgroundColors.GREEN}Saved final model to {BackgroundColors.CYAN}{model_path}{Style.RESET_ALL}")  # Notify saved model
        print(f"{BackgroundColors.GREEN}Saved scaler to {BackgroundColors.CYAN}{scaler_path}{Style.RESET_ALL}")  # Notify saved scaler
        print(f"{BackgroundColors.GREEN}Saved params to {BackgroundColors.CYAN}{params_path}{Style.RESET_ALL}")  # Notify saved params

        return final_model, scaler_full, top_features, model_path, scaler_path, features_path, model_params, params_path  # Return objects, paths and params
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


def apply_zebra_style(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Apply zebra-striping style to a DataFrame using pandas Styler.

    :param df: DataFrame to style
    :return: pandas Styler with alternating row background colors
    """
    try:
        def _row_style(row):  # Define row-wise styling helper
            bg = "white" if (row.name % 2) == 0 else "#f2f2f2"  # white/light-gray alternation
            return [f"background-color: {bg};" for _ in row.index]  # Return style per cell preserving order

        styled = df.style.apply(_row_style, axis=1)  # Apply zebra styling row-wise to preserve column order
        styled = styled.set_table_attributes('style="border-collapse:collapse; width:100%;"')  # Tight table rendering
        styled = styled.set_properties(**{"border": "1px solid #ddd", "padding": "6px"})  # Add cell padding and border
        return styled  # Return the pandas Styler object
    except Exception as e:
        print(str(e))  # Print exception for visibility
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Notify via Telegram about the failure
        raise  # Propagate exception (no silent failures)


def export_dataframe_image(styled_df: pd.io.formats.style.Styler, output_path: Union[str, Path]):
    """
    Export a pandas Styler to a PNG image using dataframe_image.

    :param styled_df: pandas Styler to export
    :param output_path: Destination PNG file path
    :raises: Propagates exceptions from dataframe_image (no silent failures)
    """
    try:
        out_p = Path(output_path)  # Normalize output path to Path
        out_p.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directory exists
        dfi.export(styled_df, str(out_p))  # Use dataframe_image to export styled DataFrame to PNG
    except Exception as e:
        print(str(e))  # Print export error for visibility
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Notify via Telegram about export failure
        raise  # Propagate exception to caller


def generate_table_image_from_dataframe(df: pd.DataFrame, output_path: Union[str, Path]):
    """
    Generate a zebra-striped PNG table image from an in-memory DataFrame.

    :param df: In-memory DataFrame to render
    :param output_path: Target PNG path (same directory as CSV)
    :raises: PermissionError if directory is not writable, or other export errors
    """
    try:
        out_p = Path(output_path)  # Convert to Path for convenience
        parent = out_p.parent  # Get parent directory
        if not parent.exists():  # If parent doesn't exist
            parent.mkdir(parents=True, exist_ok=True)  # Try to create it
        if not os.access(str(parent), os.W_OK):  # Check parent directory is writable
            raise PermissionError(f"Directory not writable: {parent}")  # Raise on non-writable directory
        styled = apply_zebra_style(df)  # Create styled DataFrame with zebra stripes
        export_dataframe_image(styled, out_p)  # Export styled DataFrame to PNG
    except Exception:
        raise  # Propagate any exception (no silent failures)


def generate_csv_and_image(df: pd.DataFrame, csv_path: Union[str, Path], is_visualizable: bool = True):
    """
    Save DataFrame to CSV and optionally generate a corresponding zebra-striped PNG image.

    :param df: DataFrame to save
    :param csv_path: Destination CSV path
    :param is_visualizable: Whether to generate PNG image alongside CSV
    :raises: Propagates IO and export errors (no silent failures)
    """
    try:
        csv_p = Path(csv_path)  # Normalize csv_path to Path
        parent = csv_p.parent  # Directory for CSV
        parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directory exists
        if not os.access(str(parent), os.W_OK):  # Verify write permission on parent
            raise PermissionError(f"Directory not writable: {parent}")  # Raise when not writable
        df.to_csv(str(csv_p), index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)  # Persist DataFrame to CSV preserving column order and content
        if is_visualizable:  # Only generate image when flagged as visualizable
            png_path = csv_p.with_suffix('.png')  # Construct PNG path by replacing extension
            generate_table_image_from_dataframe(df, png_path)  # Generate PNG from in-memory DataFrame
    except Exception:
        raise  # Propagate exceptions to caller


def save_rfe_results(csv_path, run_results):
    """
    Saves results from RFE run to a structured CSV file.

    :param csv_path: Original CSV file path
    :param run_results: List of dicts containing results from the current run
    """
    
    try:
        verbose_output(f"{BackgroundColors.GREEN}Saving RFE run results to CSV...{Style.RESET_ALL}")

        runs = run_results if isinstance(run_results, list) else [run_results]

        rows = []
        for r in runs:
            data: Dict[str, Optional[Any]] = {c: None for c in RFE_RESULTS_CSV_COLUMNS}
            ts = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
            data["timestamp"] = ts
            data["tool"] = "RFE"
            model_obj = r.get("model") or r.get("estimator") or ""
            try:
                model_repr = model_obj if isinstance(model_obj, str) else getattr(model_obj, "__class__", type(model_obj)).__name__
            except Exception:
                model_repr = str(model_obj)

            if isinstance(model_repr, str) and "random" in model_repr.lower() and "forest" in model_repr.lower():
                data["model"] = "Random Forest"
            else:
                data["model"] = model_repr
            
            try:
                data["dataset"] = os.path.relpath(csv_path)
            except Exception:
                data["dataset"] = csv_path

            hyper = r.get("hyperparameters") or r.get("params") or {}
            try:
                data["hyperparameters"] = json.dumps(hyper, sort_keys=True, ensure_ascii=False)
            except Exception:
                data["hyperparameters"] = str(hyper)

            cv_method = r.get("cv_method") or r.get("cv")
            if not cv_method:
                n_splits = r.get("cv_n_splits") or r.get("n_splits")
                if n_splits:
                    cv_method = f"StratifiedKFold(n_splits={n_splits})"
            data["cv_method"] = cv_method

            data["train_test_split"] = r.get("train_test_split") or f"test_size={r.get('test_size', 0.2)}"
            data["scaling"] = r.get("scaling") or r.get("scaler") or r.get("preprocessing") or "standard"

            metric_map = {
                "cv_accuracy": ("cv", "accuracy"),
                "cv_precision": ("cv", "precision"),
                "cv_recall": ("cv", "recall"),
                "cv_f1_score": ("cv", "f1_score"),
                "test_accuracy": ("test", "accuracy"),
                "test_precision": ("test", "precision"),
                "test_recall": ("test", "recall"),
                "test_f1_score": ("test", "f1_score"),
                "test_fpr": ("test", "fpr"),
                "test_fnr": ("test", "fnr"),
            }

            for col, (section, key) in metric_map.items():
                val = r.get(col)
                if val is None:
                    sec = r.get(section)
                    if isinstance(sec, dict):
                        val = sec.get(key)

                if val is not None:
                    try:
                        data[col] = truncate_value(float(val))
                    except Exception:
                        data[col] = val

            feature_extraction_time = r.get("feature_extraction_time_s")  # Read feature extraction time if present
            try:
                data["feature_extraction_time_s"] = round(float(feature_extraction_time), 6) if feature_extraction_time is not None else None
            except Exception:
                data["feature_extraction_time_s"] = None

            training_time = r.get("training_time_s")  # Read training time from result dict
            try:
                data["training_time_s"] = round(float(training_time), 6) if training_time is not None else None  # Store as float rounded to 6 decimals
            except Exception:
                data["training_time_s"] = None  # On error leave as None

            testing_time = r.get("testing_time_s")  # Read testing time from result dict
            try:
                data["testing_time_s"] = round(float(testing_time), 6) if testing_time is not None else None  # Store as float rounded to 6 decimals
            except Exception:
                data["testing_time_s"] = None  # On error leave as None

            try:
                data["top_features"] = json.dumps(r.get("top_features") or [], ensure_ascii=False)
            except Exception:
                data["top_features"] = str(r.get("top_features") or [])

            try:
                data["rfe_ranking"] = json.dumps(r.get("rfe_ranking") or {}, sort_keys=True, ensure_ascii=False)
            except Exception:
                data["rfe_ranking"] = str(r.get("rfe_ranking") or {})

            rows.append(data)

        df_new = pd.DataFrame(rows, columns=RFE_RESULTS_CSV_COLUMNS)
        output_dir = os.path.join(os.path.dirname(csv_path), "Feature_Analysis", "RFE")
        os.makedirs(output_dir, exist_ok=True)
        run_csv_path = os.path.join(output_dir, "RFE_Run_Results.csv")

        if os.path.exists(run_csv_path):
            try:
                df_existing = pd.read_csv(run_csv_path, dtype=str)
                if "timestamp" not in df_existing.columns:
                    mtime = os.path.getmtime(run_csv_path)
                    back_ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d_%H_%M_%S")
                    df_existing["timestamp"] = back_ts

                for c in RFE_RESULTS_CSV_COLUMNS:
                    if c not in df_existing.columns:
                        df_existing[c] = None

                df_combined = pd.concat([df_existing[RFE_RESULTS_CSV_COLUMNS], df_new], ignore_index=True, sort=False)

                try:
                    df_combined["timestamp_dt"] = pd.to_datetime(df_combined["timestamp"], format="%Y-%m-%d_%H_%M_%S", errors="coerce")
                    df_combined = df_combined.sort_values(by="timestamp_dt", ascending=False)
                    df_combined = df_combined.drop(columns=["timestamp_dt"])
                except Exception:
                    df_combined = df_combined.sort_values(by="timestamp", ascending=False)

                df_out = df_combined.reset_index(drop=True)
            except Exception:
                df_out = df_new
        else:
            df_out = df_new

        df_out = populate_hardware_column_and_order(df_out, column_name="hardware")

        try:
            generate_csv_and_image(df_out, run_csv_path, is_visualizable=True)  # Save CSV and generate PNG image
            print(f"{BackgroundColors.GREEN}Run results saved to {BackgroundColors.CYAN}{run_csv_path}{Style.RESET_ALL}")
        except Exception as e:
            print(f"{BackgroundColors.RED}Failed to save run results to CSV: {e}{Style.RESET_ALL}")

        return run_csv_path
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def print_run_summary(run_results):
    """
    Print a concise run summary to the terminal.

    :param run_results: list containing a single run-results dict
    :return: None
    """
    
    try:
        if not run_results:
            return
        
        res = run_results[0]
        print(f"\n{BackgroundColors.BOLD}Run summary:{Style.RESET_ALL}")
        print(f"  Model: {res.get('model')}")
        print(f"  CV Method: {res.get('cv_method')}")
        print(
            f"  Accuracy: {truncate_value(res.get('test_accuracy', res.get('accuracy')))}  Precision: {truncate_value(res.get('test_precision', res.get('precision')))}  Recall: {truncate_value(res.get('test_recall', res.get('recall')))}  F1: {truncate_value(res.get('test_f1_score', res.get('f1_score')))}"
        )
        print(f"  FPR: {truncate_value(res.get('test_fpr', res.get('fpr')))}  FNR: {truncate_value(res.get('test_fnr', res.get('fnr')))}  Elapsed: {res.get('elapsed_time_s')}s")
        print(f"  Top features: {res.get('top_features')}")
        if res.get("hyperparameters"):
            try:
                hp = json.loads(res.get("hyperparameters")) if isinstance(res.get("hyperparameters"), str) else res.get("hyperparameters")
                print(f"  Hyperparameters: {json.dumps(hp)}")
            except Exception:
                print(f"  Hyperparameters: {res.get('hyperparameters')}")
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def load_exported_artifacts(csv_path):
    """Attempt to locate and load latest exported model, scaler and features for csv_path.

    :param csv_path: original dataset path used to name exported artifacts
    :return: (model, scaler, features) or None if not found
    """
    
    try:
        models_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis/RFE/Models/"  # Location where RFE artifacts are stored
        
        if not os.path.isdir(models_dir):
            return None  # No models directory

        base_name = safe_filename(Path(csv_path).stem)  # Safe base name
        pattern = os.path.join(models_dir, f"RFE-{base_name}-*-model.joblib")  # Glob pattern for RFE model files
        candidates = glob.glob(pattern)  # Find matching model files
        if not candidates:
            return None  # No exported models found

        latest_model = max(candidates, key=os.path.getmtime)  # Select most recent model file
        scaler_path = latest_model.replace("-model.joblib", "-scaler.joblib")  # Infer scaler path
        features_path = latest_model.replace("-model.joblib", "-features.json")  # Infer features path
        if not os.path.exists(scaler_path) or not os.path.exists(features_path):
            return None  # Incomplete artifact set

        try:
            model = load(latest_model)  # Load model with joblib
            scaler = load(scaler_path)  # Load scaler with joblib
            with open(features_path, "r", encoding="utf-8") as fh:
                features = json.load(fh)  # Load features list
            params = None
            params_path = latest_model.replace("-model.joblib", "-params.json")  # Infer params path
            if os.path.exists(params_path):
                try:
                    with open(params_path, "r", encoding="utf-8") as ph:
                        params = json.load(ph)  # Load hyperparameters if available
                except Exception:
                    params = None
            return model, scaler, features, params
        except Exception:
            return None  # Any loading error -> treat as not found
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def evaluate_exported_model(model, scaler, X_numeric, feature_columns, top_features, y_array):
    """Evaluate a loaded/trained model on the full numeric dataset and
    compute the same metrics used by the RFE pipeline.

    :return: tuple (acc, prec, rec, f1, fpr, fnr, testing_time_s)
    """
    
    try:
        test_start = time.perf_counter()  # Start perf_counter immediately before prediction (testing window)
        X_scaled = scaler.transform(X_numeric.values)  # Scale full numeric data with provided scaler
        sel_indices = [i for i, f in enumerate(feature_columns) if f in top_features]  # Indices for chosen features
        X_eval = X_scaled[:, sel_indices] if sel_indices else X_scaled  # Selected eval array
        y_pred = model.predict(X_eval)  # Model predictions on full dataset

        acc = accuracy_score(y_array, y_pred)  # Compute accuracy
        prec = precision_score(y_array, y_pred, average="weighted", zero_division=0)  # Precision
        rec = recall_score(y_array, y_pred, average="weighted", zero_division=0)  # Recall
        f1 = f1_score(y_array, y_pred, average="weighted", zero_division=0)  # F1 score

        if len(np.unique(y_array)) == 2:
            cm = confusion_matrix(y_array, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
            else:
                total = cm.sum() if cm.size > 0 else 1
                fpr = float(cm.sum() - np.trace(cm)) / float(total) if total > 0 else 0
                fnr = fpr
        else:
            cm = confusion_matrix(y_array, y_pred)
            supports = cm.sum(axis=1)
            fprs = []
            fnrs = []
            for i in range(cm.shape[0]):
                tp = cm[i, i]
                fn = cm[i, :].sum() - tp
                fp = cm[:, i].sum() - tp
                tn = cm.sum() - (tp + fp + fn)
                denom_fnr = (tp + fn) if (tp + fn) > 0 else 1
                denom_fpr = (fp + tn) if (fp + tn) > 0 else 1
                fnr_i = fn / denom_fnr
                fpr_i = fp / denom_fpr
                fprs.append((fpr_i, supports[i]))
                fnrs.append((fnr_i, supports[i]))
            total_support = float(supports.sum()) if supports.sum() > 0 else 1.0
            fpr = float(sum(v * s for v, s in fprs) / total_support)
            fnr = float(sum(v * s for v, s in fnrs) / total_support)

        test_end = time.perf_counter()  # End perf_counter immediately after evaluation metrics computed
        testing_time_s = round(test_end - test_start, 6)  # Compute testing time rounded to 6 decimals
        return float(acc), float(prec), float(rec), float(f1), float(fpr), float(fnr), float(testing_time_s)
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_final_model(csv_path, X_train, y_train, top_features, feature_columns):
    """
    Helper function to load or export the final model, scaler, and parameters.

    :param csv_path: Path to the CSV dataset file
    :param X_train: Training features DataFrame
    :param y_train: Training target array
    :param top_features: List of top features
    :param feature_columns: Feature column names
    :return: final_model, scaler_full, top_features, loaded_hyperparams
    """
    
    try:
        loaded_hyperparams = None
        if SKIP_TRAIN_IF_MODEL_EXISTS:
            loaded = load_exported_artifacts(csv_path)
            if loaded is not None:
                final_model, scaler_full, loaded_features, loaded_params = loaded
                top_features = loaded_features
                loaded_hyperparams = loaded_params
                print(f"{BackgroundColors.GREEN}Loaded exported model and scaler for {BackgroundColors.CYAN}{Path(csv_path).stem}{Style.RESET_ALL}")
            else:
                (
                    final_model,
                    scaler_full,
                    top_features,
                    _model_path,
                    _scaler_path,
                    _features_path,
                    model_params,
                    _params_path,
                ) = export_final_model(X_train, feature_columns, top_features, y_train, csv_path)
                loaded_hyperparams = model_params
        else:
            (
                final_model,
                scaler_full,
                top_features,
                _model_path,
                _scaler_path,
                _features_path,
                model_params,
                _params_path,
            ) = export_final_model(X_train, feature_columns, top_features, y_train, csv_path)
            loaded_hyperparams = model_params

        return final_model, scaler_full, top_features, loaded_hyperparams
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_run_results(final_model, csv_path, hyperparameters, cv_method, cv_metrics=None, test_metrics=None, training_time=None, top_features=None, rfe_ranking=None, feature_extraction_time=None):
    """
    Helper function to build the run_results dictionary.

    :param final_model: The trained model
    :param csv_path: Path to the CSV dataset file
    :param hyperparameters: Hyperparameters dict
    :param cv_method: CV method string
    :param cv_metrics: Tuple of CV metrics (optional)
    :param test_metrics: Tuple of test metrics
    :param training_time: Training time
    :param top_features: List of top features
    :param rfe_ranking: Sorted RFE ranking
    :param feature_extraction_time: Feature extraction time
    :return: List containing the results dict
    """
    
    try:
        result = {
            "model": final_model.__class__.__name__,
            "dataset": os.path.relpath(csv_path),
            "hyperparameters": json.dumps(hyperparameters),
            "cv_method": cv_method,
            "top_features": json.dumps(top_features),
            "rfe_ranking": json.dumps(rfe_ranking),
        }

        if cv_metrics is not None:
            result.update({
                "cv_accuracy": truncate_value(cv_metrics[0]),
                "cv_precision": truncate_value(cv_metrics[1]) or "0.0",
                "cv_recall": truncate_value(cv_metrics[2]) or "0.0",
                "cv_f1_score": truncate_value(cv_metrics[3]) or "0.0",
                "cv_fpr": truncate_value(cv_metrics[4]) or "0.0",
                "cv_fnr": truncate_value(cv_metrics[5]) or "0.0",
            })

        if test_metrics is not None:
            result.update({
                "test_accuracy": truncate_value(test_metrics[0]) or "0.0",
                "test_precision": truncate_value(test_metrics[1]) or "0.0",
                "test_recall": truncate_value(test_metrics[2]) or "0.0",
                "test_f1_score": truncate_value(test_metrics[3]) or "0.0",
                "test_fpr": truncate_value(test_metrics[4]) or "0.0",
                "test_fnr": truncate_value(test_metrics[5]) or "0.0",
                "testing_time_s": round(float(test_metrics[6]), 6),
            })

        if training_time is not None:
            result["training_time_s"] = round(float(training_time), 6)

        if feature_extraction_time is not None:
            result["feature_extraction_time_s"] = round(float(feature_extraction_time), 6)
        else:
            result["feature_extraction_time_s"] = round(0.0, 6)

        return [result]
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_results_with_hyperparams(final_model, csv_path, loaded_hyperparams, fallback_hyperparameters, cv_method, cv_metrics=None, test_metrics=None, training_time=None, top_features=None, rfe_ranking=None, feature_extraction_time=None):
    """
    Determine hyperparameters to save (prefer loaded_hyperparams, else model.get_params(), else fallback)
    and build the run_results via `build_run_results`.

    :param final_model: trained or loaded model
    :param csv_path: dataset path
    :param loaded_hyperparams: params loaded from params.json (or None)
    :param fallback_hyperparameters: fallback dict supplied by caller
    :param cv_method: string describing CV method
    :param cv_metrics: optional CV metrics tuple
    :param test_metrics: optional test metrics tuple
    :param training_time: optional training time
    :param top_features: optional list of top features
    :param rfe_ranking: optional rfe ranking mapping
    :param feature_extraction_time: optional feature extraction time
    :return: list containing a single run_results dict
    """
    
    try:
        if loaded_hyperparams is not None:
            hyperparams_to_save = loaded_hyperparams
        else:
            try:
                hyperparams_to_save = final_model.get_params()
            except Exception:
                hyperparams_to_save = fallback_hyperparameters or {}

        return build_run_results(
            final_model,
            csv_path,
            hyperparams_to_save,
            cv_method,
            cv_metrics=cv_metrics,
            test_metrics=test_metrics,
            training_time=training_time,
            top_features=top_features,
            rfe_ranking=rfe_ranking,
            feature_extraction_time=feature_extraction_time,
        )
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_rfe_fallback(csv_path, X_numeric, y_array, feature_columns, hyperparameters, n_features_to_select=None, step=1, estimator_name="random_forest", random_state=42):
    """
    Handles RFE for datasets with insufficient samples for stratified CV (fallback to single train/test split).

    :param csv_path: Path to the CSV dataset file
    :param X_numeric: Numeric features DataFrame
    :param y_array: Target array
    :param feature_columns: Feature column names
    :param hyperparameters: Hyperparameters dict
    :return: None
    """
    
    try:
        print(f"{BackgroundColors.YELLOW}Not enough samples per class for stratified CV; falling back to single train/test split.{Style.RESET_ALL}")
        send_telegram_message(TELEGRAM_BOT, f"RFE: Falling back to single train/test split for dataset {Path(csv_path).stem}")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_numeric.values, y_array, test_size=0.2, random_state=int(random_state), stratify=None
        )  # Perform a single non-stratified train/test split
        selector, model, feature_extraction_time_s = run_rfe_selector(
            X_train, y_train, n_select=n_features_to_select or X_train.shape[1], step=step, estimator_name=estimator_name, random_state=random_state
        )  # Run RFE on the single split and get feature extraction time
        metrics_tuple = compute_rfe_metrics(selector, X_train, X_test, y_train, y_test, random_state=random_state, estimator_name=estimator_name)  # Compute metrics on split (returns training and testing times)
        top_features, rfe_ranking = extract_top_features(selector, feature_columns)  # Extract selected features and rankings
        sorted_rfe_ranking = sorted(rfe_ranking.items(), key=lambda x: x[1])  # Sort features by ranking (ascending)

        print_metrics(metrics_tuple) if VERBOSE else None  # Optionally print metrics
        print_top_features(top_features, rfe_ranking) if VERBOSE else None  # Optionally print top features

        final_model, scaler_full, top_features, loaded_hyperparams = get_final_model(csv_path, X_numeric, y_array, top_features, feature_columns)  # Get final model/scaler
        eval_metrics = evaluate_exported_model(final_model, scaler_full, X_numeric, feature_columns, top_features, y_array)  # Evaluate on full dataset

        run_results = build_results_with_hyperparams(
            final_model,
            csv_path,
            loaded_hyperparams,
            hyperparameters,
            "single_train_test_split",
            test_metrics=eval_metrics,
            training_time=metrics_tuple[6],
            top_features=top_features,
            rfe_ranking=sorted_rfe_ranking,
            feature_extraction_time=feature_extraction_time_s,
        )  # Build results dict

        save_rfe_results(csv_path, run_results)  # Save fallback run results
        print_run_summary(run_results)  # Concise terminal summary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_rfe_cv(csv_path, X_numeric, y_array, feature_columns, hyperparameters, n_features_to_select=None, step=1, estimator_name="random_forest", random_state=42):
    """
    Handles RFE with stratified cross-validation.

    :param csv_path: Path to the CSV dataset file
    :param X_numeric: Numeric features DataFrame
    :param y_array: Target array
    :param feature_columns: Feature column names
    :param hyperparameters: Hyperparameters dict
    :return: None
    """
    
    try:
        verbose_output(f"{BackgroundColors.GREEN}Starting RFE with Stratified K-Fold Cross-Validation...{Style.RESET_ALL}")
        
        stratify_param = y_array if len(np.unique(y_array)) > 1 else None
        X_train_df, X_test_df, y_train_array, y_test_array = train_test_split(
            X_numeric, y_array, test_size=0.2, random_state=int(random_state), stratify=stratify_param
        )

        scaler_for_run = StandardScaler()  # Create scaler for the run
        scale_start = time.perf_counter()  # Start perf_counter immediately before scaling (feature extraction window begins)
        X_train_scaled = scaler_for_run.fit_transform(X_train_df.values)  # Fit scaler and transform training data (scaling)
        X_test_scaled = scaler_for_run.transform(X_test_df.values)  # Transform test data using fitted scaler
        scale_end = time.perf_counter()  # End perf_counter immediately after scaling (feature extraction window part)
        scaling_time_s = round(scale_end - scale_start, 6)  # Compute scaling time rounded to 6 decimals

        unique_tr, counts_tr = np.unique(y_train_array, return_counts=True)
        min_class_count_tr = counts_tr.min() if counts_tr.size > 0 else 0
        n_splits = min(10, len(y_train_array), min_class_count_tr)  # Up to 10 splits but not more than samples or smallest class in train
        n_splits = max(2, int(n_splits))  # Ensure at least 2 splits

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))  # Create stratified K-Fold iterator

        fold_metrics = []  # List to collect per-fold metric tuples
        fold_rankings = []  # List to collect per-fold ranking arrays
        fold_supports = []  # List to collect per-fold support masks
        total_elapsed = 0.0  # Accumulator for training times across folds
        total_feature_extraction = float(scaling_time_s)  # Accumulator for total feature extraction time (scaling + selector fits)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_train_scaled, y_train_array), start=1):
            verbose_output(f"{BackgroundColors.CYAN}Running fold {fold_idx}/{n_splits}{Style.RESET_ALL}")  # Optional fold progress

            X_train_fold = X_train_scaled[train_idx]
            X_test_fold = X_train_scaled[test_idx]
            y_train_fold = y_train_array[train_idx]
            y_test_fold = y_train_array[test_idx]

            selector, model, feat_time = run_rfe_selector(
                X_train_fold, y_train_fold, n_select=n_features_to_select or X_train_fold.shape[1], step=step, estimator_name=estimator_name, random_state=random_state
            )  # Fit RFE on this fold's training data and get selector fit time

            metrics_tuple = compute_rfe_metrics(
                selector, X_train_fold, X_test_fold, y_train_fold, y_test_fold, random_state=random_state, estimator_name=estimator_name
            )  # Compute metrics for this fold (returns training and testing times)
            fold_metrics.append(metrics_tuple)  # Append per-fold metrics tuple
            fold_rankings.append(selector.ranking_)  # Append per-fold ranking array
            fold_supports.append(selector.support_.astype(int))  # Append per-fold support mask as integers
            total_elapsed += metrics_tuple[6]  # Accumulate training time from this fold
            total_feature_extraction += float(feat_time)  # Accumulate selector fit time into total feature extraction

            send_telegram_message(
                TELEGRAM_BOT,
                f"RFE: Finished fold {fold_idx}/{n_splits} for dataset {Path(csv_path).stem} with F1: {truncate_value(metrics_tuple[3])} in {calculate_execution_time(0, metrics_tuple[6])}"
            )

        metrics_arr = np.array(fold_metrics)  # Convert list of tuples to numpy array
        mean_metrics = metrics_arr.mean(axis=0)  # Compute mean metric values across folds

        rankings_arr = np.vstack(fold_rankings)  # Shape: (n_folds, n_features) stack rankings
        mean_rankings = rankings_arr.mean(axis=0)  # Mean ranking per feature
        avg_rfe_ranking = {f: float(r) for f, r in zip(feature_columns, mean_rankings)}  # Map feature->avg rank

        supports_arr = np.vstack(fold_supports)  # Shape: (n_folds, n_features) stack support masks
        support_counts = supports_arr.sum(axis=0)  # Count how many folds selected each feature
        majority_threshold = (n_splits // 2) + 1  # Require strict majority to consider a feature selected
        top_features = [f for f, c in zip(feature_columns, support_counts) if c >= majority_threshold]  # Select majority-chosen features

        sorted_rfe_ranking = sorted(avg_rfe_ranking.items(), key=lambda x: x[1])  # Sort averaged rankings ascending

        final_model, scaler_full, top_features, loaded_hyperparams = get_final_model(csv_path, X_train_df, y_train_array, top_features, feature_columns)

        eval_metrics = evaluate_exported_model(final_model, scaler_full, X_test_df, feature_columns, top_features, y_test_array)

        run_results = build_results_with_hyperparams(
            final_model,
            csv_path,
            loaded_hyperparams,
            hyperparameters,
            f"StratifiedKFold(n_splits={n_splits})",
            cv_metrics=mean_metrics,
            test_metrics=eval_metrics,
            training_time=total_elapsed,
            top_features=top_features,
            rfe_ranking=sorted_rfe_ranking,
            feature_extraction_time=total_feature_extraction,
        )  # Build results dict

        print_metrics(tuple(mean_metrics)) if VERBOSE else None  # Optionally print aggregated metrics
        print_top_features(top_features, avg_rfe_ranking) if VERBOSE else None  # Optionally print aggregated top features and avg ranks

        save_rfe_results(csv_path, run_results)  # Save aggregated run results to CSV
        print_run_summary(run_results)  # Concise terminal summary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_rfe(csv_path, n_features_to_select=None, step=None, estimator_name=None, random_state=None):
    """
    Runs Recursive Feature Elimination on the provided dataset, prints the single
    set of top features selected, computes and prints performance metrics, and
    saves the structured results.

    :param csv_path: Path to the CSV dataset file
    :return: None
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Starting RFE analysis on dataset: {BackgroundColors.CYAN}{csv_path}{Style.RESET_ALL}"
        )

        hyperparameters = {}  # Default hyperparameters (to be extended later)

        cfg = get_config()
        rfe_cfg = cfg.get("rfe", {}) if isinstance(cfg, dict) else {}

        cfg_n_select = rfe_cfg.get("n_features_to_select")
        cfg_step = rfe_cfg.get("step", 1)
        cfg_estimator = rfe_cfg.get("estimator", "random_forest")
        cfg_random_state = int(rfe_cfg.get("random_state", 42))


        default_n_select = n_features_to_select if n_features_to_select is not None else cfg_n_select
        default_step = step if step is not None else cfg_step
        default_estimator = estimator_name if estimator_name is not None else cfg_estimator
        default_random_state = int(random_state) if random_state is not None else cfg_random_state

        df = load_dataset(csv_path)  # Load dataset

        if df is None:  # If loading failed
            return  # Exit the function

        cleaned_df = preprocess_dataframe(df)  # Preprocess the DataFrame

        X = cleaned_df.iloc[:, :-1]  # Features are all columns except the last
        y = cleaned_df.iloc[:, -1]  # Target is the last column

        if X is None or y is None:  # If feature or target extraction failed
            return  # Exit the function

        X_numeric = X.select_dtypes(include=["number"]).copy()  # Select only numeric features
        if X_numeric.shape[1] == 0:  # If no numeric features found
            coerced_cols = {}  # Dictionary to hold coerced numeric columns
            for col in X.columns:  # Iterate through all columns
                coerced = pd.to_numeric(X[col], errors="coerce")  # Attempt to coerce to numeric
                if coerced.notna().sum() > 0:  # If any values were successfully coerced
                    coerced_cols[col] = coerced  # Add to coerced columns
            if coerced_cols:  # If any columns were successfully coerced
                X_numeric = pd.DataFrame(coerced_cols, index=X.index)  # Create DataFrame from coerced columns
            else:  # If no columns could be coerced
                print(f"{BackgroundColors.RED}No numeric features found after preprocessing. Cannot run RFE.{Style.RESET_ALL}")
                return  # Exit the function

        feature_columns = X_numeric.columns  # Get the feature column names

        y_array = np.array(y)  # Convert target to numpy array
        unique, counts = np.unique(y_array, return_counts=True)  # Get unique classes and their counts
        min_class_count = counts.min() if counts.size > 0 else 0  # Minimum samples in any class

        if min_class_count < 2:  # If any class has fewer than 2 samples
            run_rfe_fallback(
                csv_path,
                X_numeric,
                y_array,
                feature_columns,
                hyperparameters,
                n_features_to_select=default_n_select,
                step=default_step,
                estimator_name=default_estimator,
                random_state=default_random_state,
            )  # Run fallback RFE
        else:  # If sufficient samples for stratified CV
            run_rfe_cv(
                csv_path,
                X_numeric,
                y_array,
                feature_columns,
                hyperparameters,
                n_features_to_select=default_n_select,
                step=default_step,
                estimator_name=default_estimator,
                random_state=default_random_state,
            )  # Run RFE with CV
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


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
    Calculates the execution time and returns a human-readable string.

    Accepts either:
    - Two datetimes/timedeltas: `calculate_execution_time(start, finish)`
    - A single timedelta or numeric seconds: `calculate_execution_time(delta)`
    - Two numeric timestamps (seconds): `calculate_execution_time(start_s, finish_s)`

    Returns a string like "1h 2m 3s".
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


def play_sound():
    """
    Plays a sound when the program finishes and skips if the operating system is Windows.

    :param: None
    :return: None
    """
    
    try:
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
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def main(n_features_to_select=None, step=None, estimator_name=None, random_state=None, config_path=None):
    """
    Main function.

    :param: None
    :return: None
    """
    
    try:
        print(
            f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Recursive Feature Elimination (RFE){BackgroundColors.GREEN} program!{Style.RESET_ALL}"
        )  # Output the welcome message
        start_time = datetime.datetime.now()  # Get the start time of the program
        
        setup_telegram_bot()  # Setup Telegram bot if configured

        dataset_name = os.path.splitext(os.path.basename(CSV_FILE))[0]  # Get dataset name for messages

        send_telegram_message(TELEGRAM_BOT, [f"Starting RFE Feature Selection on {dataset_name} at {start_time.strftime('%d/%m/%Y - %H:%M:%S')}"])  # Send starting message

        run_rfe(CSV_FILE, n_features_to_select=n_features_to_select, step=step, estimator_name=estimator_name, random_state=random_state)

        finish_time = datetime.datetime.now()  # Get the finish time of the program
        print(
            f"{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
        )  # Output the start and finish times
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
        )  # Output the end of the program message
        
        send_telegram_message(TELEGRAM_BOT, [f"RFE Feature Selection completed for {dataset_name}. Execution time: {calculate_execution_time(start_time, finish_time)}"])  # Send completion message

        atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None  # Register play_sound at exit if enabled
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RFE pipeline and optionally load existing exported models."
    )  # CLI description
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (overrides auto-detection)")
    parser.add_argument(
        "--skip-train-if-model-exists",
        dest="skip_train",
        action="store_true",
        help="If set, do not retrain; load existing exported artifacts and evaluate.",
    )  # Flag to skip training when exported artifacts exist
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose output during the run.",
    )  # Flag to enable verbose logging
    parser.add_argument(
        "--csv",
        dest="csv",
        type=str,
        default=None,
        help="Optional: path to dataset CSV to analyze. If omitted, uses the default in main().",
    )  # Optional CSV override
    parser.add_argument("--n_features_to_select", type=int, default=None, help="Number of features to select (overrides config)")
    parser.add_argument("--step", type=int, default=None, help="RFE step (number of features to remove per iteration)")
    parser.add_argument("--estimator", type=str, default=None, help="Estimator name for RFE (default: random_forest)")
    parser.add_argument("--random_state", type=int, default=None, help="Random seed for reproducibility (overrides config)")
    args = parser.parse_args()  # Parse CLI args

    SKIP_TRAIN_IF_MODEL_EXISTS = bool(args.skip_train)  # Respect the user's CLI choice
    VERBOSE = bool(args.verbose)  # Respect the user's request to print verbose output
    CSV_FILE = args.csv if args.csv else CSV_FILE  # Use provided CSV or default

    # Load config to determine defaults (if any)
    cfg = get_config(args.config)
    rfe_cfg = cfg.get("rfe", {}) if isinstance(cfg, dict) else {}

    # Resolve parameters precedence: CLI > config > defaults
    n_features_to_select = args.n_features_to_select if args.n_features_to_select is not None else rfe_cfg.get("n_features_to_select")
    step = args.step if args.step is not None else rfe_cfg.get("step", 1)
    estimator_name = args.estimator if args.estimator is not None else rfe_cfg.get("estimator", "random_forest")
    random_state = int(args.random_state) if args.random_state is not None else int(rfe_cfg.get("random_state", 42))

    # Log resolved sources
    print(f"RFE params: n_features_to_select={n_features_to_select} (cli/config/default), step={step}, estimator={estimator_name}, random_state={random_state}")

    main(n_features_to_select=n_features_to_select, step=step, estimator_name=estimator_name, random_state=random_state, config_path=args.config)
