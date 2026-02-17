"""
================================================================================
Principal Component Analysis (PCA) Feature Extraction & Evaluation Tool (pca.py)
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-11-21
Description :
    Utility to run PCA-based dimensionality reduction and evaluate downstream
    classification performance. The script bundles dataset loading, cleaning,
    scaling, PCA transformation, stratified cross-validation evaluation and
    export of consolidated results for easy comparison between configurations.

Core features:
    - Safe dataset loading and basic validation
    - Z-score standardization of numeric features prior to PCA
    - PCA transform with configurable `n_components` grid
    - 10-fold Stratified CV on the training set and final evaluation on a held-out test split
    - Aggregated metrics including Accuracy, Precision, Recall, F1, FPR and FNR
    - Export of `PCA_Results.csv` (per-dataset `Feature_Analysis/`) with hardware metadata
    - Optional parallel execution for multiple component configurations

Usage:
    - Configure `csv_file` in `main()` or call `run_pca_analysis(csv_path, ...)`
    - Adjust `n_components_list` to test desired component counts
    - Run: `python3 pca.py` or via the repository Makefile

Outputs:
    - `Feature_Analysis/PCA_Results.csv` (one row per configuration)
    - Saved PCA objects for reproducibility (optional)
    - Console summary and best-configuration selection by CV F1-score

Notes & conventions:
    - The code expects the last column to be the target variable.
    - Only numeric input columns are used for PCA (non-numeric columns are ignored).
    - Defaults: 80/20 train-test split, 10-fold Stratified CV on training data,
        RandomForest (100 trees) used for evaluation.
    - Toggle `VERBOSE = True` for extra diagnostic output.

TODOs:
    - Add CLI argument parsing for dataset path, `n_components_list`, `parallel` and `max_workers`.
    - Add visualization for explained variance and component loadings.
    - Provide incremental / out-of-core PCA for very large datasets.
    - Add unit tests for preprocessing and evaluation functions.

Dependencies:
    - Python >= 3.9
    - pandas, numpy, scikit-learn, colorama
"""

import argparse  # For command-line argument parsing
import atexit  # For playing a sound when the program finishes
import concurrent.futures  # For parallel execution
import datetime  # For timestamping
import glob  # For file pattern matching
import json  # For serializing hyperparameters and other metadata
import math  # For mathematical operations
import numpy as np  # For numerical operations
import os  # For file and directory operations
import pandas as pd  # For data manipulation
import pickle  # For serializing PCA objects
import platform  # For getting the operating system name
import psutil  # For system memory and CPU counts
import re  # For regex operations
import subprocess  # For fetching CPU model on some OSes
import sys  # For system-specific parameters and functions
import telegram_bot as telegram_module  # For setting Telegram prefix and device info
import time  # For measuring elapsed time
from colorama import Style  # For coloring the terminal
from joblib import dump  # For saving scalers and models
from Logger import Logger  # For logging output to both terminal and file
from pathlib import Path  # For handling file paths
from sklearn.decomposition import PCA  # For Principal Component Analysis
from sklearn.ensemble import RandomForestClassifier  # For the Random Forest model
from sklearn.metrics import (  # For performance metrics
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split, StratifiedKFold  # For splitting and cross-validation
from sklearn.preprocessing import StandardScaler  # For scaling the data (standardization)
from telegram_bot import TelegramBot, send_exception_via_telegram, send_telegram_message, setup_global_exception_hook  # For sending progress messages to Telegram
from tqdm import tqdm  # For progress bars


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
SKIP_TRAIN_IF_MODEL_EXISTS = False  # If True, try loading exported models instead of retraining
CSV_FILE = "./Datasets/CICDDoS2019/01-12/DrDoS_DNS.csv"  # Path to the CSV dataset file (set in main)
PCA_RESULTS_CSV_COLUMNS = [  # Columns for the PCA results CSV
    "timestamp",
    "tool",
    "model",
    "dataset",
    "hyperparameters",
    "cv_method",
    "train_test_split",
    "scaling",
    "n_components",
    "explained_variance",
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

        global TELEGRAM_BOT  # Declare the module-global telegram_bot variable

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
    :return: X_train, X_test, y_train, y_test, scaler
    """
    
    try:
        stratify_param = y if len(np.unique(y)) > 1 else None  # Determine stratify param
        X_train_df, X_test_df, y_train, y_test = train_test_split(  # Split dataset
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param  # Split args
        )  # End split

        start_scaling = time.perf_counter()  # Start high-resolution scaling timer
        scaler = StandardScaler()  # Create scaler instance
        X_train = scaler.fit_transform(X_train_df)  # Fit scaler and transform train
        X_test = scaler.transform(X_test_df)  # Transform test data with fitted scaler
        scaling_time = round(time.perf_counter() - start_scaling, 6)  # End scaling timer and round to 6 decimals
        try:  # Safely attach scaling time to scaler instance using setattr to avoid static attribute-access diagnostics
            setattr(scaler, "_scaling_time", scaling_time)  # Store scaling time as dynamic attribute on scaler
        except Exception:  # Preserve prior silent-failure behavior if attribute cannot be set
            pass  # No-op on failure

        return X_train, X_test, y_train, y_test, scaler  # Return the split data and scaler
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def apply_pca_and_evaluate(X_train, y_train, X_test, y_test, n_components, cv_folds=10, workers=1, scaling_time=0.0):
    """
    Applies PCA transformation and evaluates performance using 10-fold Stratified Cross-Validation

    :param X_train: Training features (scaled)
    :param y_train: Training target
    :param X_test: Testing features (scaled)
    :param y_test: Testing target
    :param n_components: Number of principal components to keep
    :param cv_folds: Number of cross-validation folds (default: 10)
    :return: Dictionary containing metrics, explained variance, and PCA object
    """
    
    try:
        if n_components <= 0:  # Validate n_components
            raise ValueError(f"n_components must be positive, got {n_components}")  # Raise on invalid
        if n_components > X_train.shape[1]:  # Validate n_components against number of features
            raise ValueError(  # Raise descriptive error
                f"n_components ({n_components}) cannot be greater than number of features ({X_train.shape[1]})"
            )  # End raise

        send_telegram_message(TELEGRAM_BOT, f"Starting PCA training for n_components={n_components}")  # Notify
        pca = PCA(n_components=n_components, random_state=42)  # Initialize PCA

        start_pca = time.perf_counter()  # Start PCA timer (feature extraction part)
        X_train_pca = pca.fit_transform(X_train)  # Fit PCA on training data and transform
        X_test_pca = pca.transform(X_test)  # Transform test data using the fitted PCA
        pca_time = round(time.perf_counter() - start_pca, 6)  # End PCA timer and round
        feature_extraction_time_s = round((scaling_time or 0.0) + pca_time, 6)  # Sum scaling and PCA times

        explained_variance = pca.explained_variance_ratio_.sum()  # Total explained variance ratio

        rf_n_jobs = -1 if workers == 1 else 1  # Set n_jobs for RandomForest based on workers
        model = RandomForestClassifier(  # Initialize Random Forest model
            n_estimators=100, random_state=42, n_jobs=rf_n_jobs
        )  # End model init

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)  # Stratified K-Fold cross-validator

        cv_accs, cv_precs, cv_recs, cv_f1s = [], [], [], []  # Lists to store CV metrics
        total_training_time = 0.0  # Accumulator for all model.fit durations
        total_testing_time = 0.0  # Accumulator for all prediction+metric durations

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_pca, y_train), start=1):  # Loop folds
            send_telegram_message(TELEGRAM_BOT, f"Starting CV fold {fold_idx}/{cv_folds} for n_components={n_components}")  # Notify
            X_train_fold = X_train_pca[train_idx]  # Training data for this fold
            X_val_fold = X_train_pca[val_idx]  # Validation data for this fold
            y_train_fold = (  # Support both Series and ndarray
                y_train.iloc[train_idx] if isinstance(y_train, pd.Series) else y_train[train_idx]
            )  # End y_train_fold
            y_val_fold = (  # Support both Series and ndarray for validation
                y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]
            )  # End y_val_fold

            start_fit = time.perf_counter()  # Start timer immediately before model.fit for this fold
            model.fit(X_train_fold, y_train_fold)  # Fit model on training fold
            fit_elapsed = round(time.perf_counter() - start_fit, 6)  # Stop timer immediately after fit and round
            total_training_time += fit_elapsed  # Accumulate training durations

            start_pred = time.perf_counter()  # Start timer immediately before prediction+metrics for this fold
            y_pred_fold = model.predict(X_val_fold)  # Predict on validation fold
            cv_accs.append(accuracy_score(y_val_fold, y_pred_fold))  # Calculate and store accuracy
            cv_precs.append(
                precision_score(y_val_fold, y_pred_fold, average="weighted", zero_division=0)
            )  # Calculate and store precision
            cv_recs.append(
                recall_score(y_val_fold, y_pred_fold, average="weighted", zero_division=0)
            )  # Calculate and store recall
            f1_fold = f1_score(y_val_fold, y_pred_fold, average="weighted", zero_division=0)  # Compute F1
            cv_f1s.append(f1_fold)  # Store F1
            pred_elapsed = round(time.perf_counter() - start_pred, 6)  # Stop timer after prediction+metrics and round
            total_testing_time += pred_elapsed  # Accumulate testing durations for CV
            send_telegram_message(TELEGRAM_BOT, f"Finished CV fold {fold_idx}/{cv_folds} for n_components={n_components} with F1: {truncate_value(f1_fold)}")  # Notify fold completion

        cv_acc_mean = np.mean(cv_accs)  # Mean CV accuracy
        cv_prec_mean = np.mean(cv_precs)  # Mean CV precision
        cv_rec_mean = np.mean(cv_recs)  # Mean CV recall
        cv_f1_mean = np.mean(cv_f1s)  # Mean CV f1

        send_telegram_message(TELEGRAM_BOT, f"Finished PCA training for n_components={n_components} with CV F1: {truncate_value(cv_f1_mean)}")  # Notify completion

        start_final_fit = time.perf_counter()  # Start timer immediately before final model.fit
        model.fit(X_train_pca, y_train)  # Fit model on full training data
        final_fit_elapsed = round(time.perf_counter() - start_final_fit, 6)  # Stop timer immediately after final fit and round
        total_training_time += final_fit_elapsed  # Add final fit duration to training total

        start_test = time.perf_counter()  # Start timer immediately before test prediction+metrics
        y_pred = model.predict(X_test_pca)  # Predict on test data
        acc = accuracy_score(y_test, y_pred)  # Calculate test accuracy
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate test precision
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate test recall
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate test f1
        fpr, fnr = 0, 0  # Initialize FPR and FNR
        unique_classes = np.unique(y_test)  # Get unique classes in the test set
        if len(unique_classes) == 2:  # If binary classification
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=unique_classes).ravel()  # Get confusion matrix
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Compute FPR
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # Compute FNR
        test_pred_elapsed = round(time.perf_counter() - start_test, 6)  # Stop timer after test prediction+metrics and round
        total_testing_time += test_pred_elapsed  # Add test prediction duration to testing total

        scaler_export = StandardScaler().fit(np.vstack([X_train, X_test]))  # Create scaler for export

        try:  # Try to get trained classifier parameters
            trained_classifier_params = model.get_params()  # Get model parameters
        except Exception:  # On failure
            trained_classifier_params = None  # Set to None

        return {
            "n_components": n_components,  # Components
            "explained_variance": explained_variance,  # Explained variance
            "cv_accuracy": cv_acc_mean,  # CV accuracy
            "cv_precision": cv_prec_mean,  # CV precision
            "cv_recall": cv_rec_mean,  # CV recall
            "cv_f1_score": cv_f1_mean,  # CV f1
            "test_accuracy": acc,  # Test accuracy
            "test_precision": prec,  # Test precision
            "test_recall": rec,  # Test recall
            "test_f1_score": f1,  # Test f1
            "test_fpr": fpr,  # Test FPR
            "test_fnr": fnr,  # Test FNR
            "feature_extraction_time_s": float(round(feature_extraction_time_s, 6)),  # Feature extraction time (scaling+PCA)
            "training_time_s": float(round(total_training_time, 6)),  # Training time (sum of all model.fit durations)
            "testing_time_s": float(round(total_testing_time, 6)),  # Testing time (sum of all prediction+metric durations)
            "pca_object": pca,  # PCA object
            "scaler": scaler_export,  # Scaler for export
            "trained_classifier": model,  # Trained classifier object
            "trained_classifier_params": trained_classifier_params,  # Trained classifier params
        }
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def print_pca_results(results):
    """
    Prints PCA results in a formatted way.

    :param results: Dictionary containing PCA evaluation results
    :return: None
    """
    
    try:
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}PCA Results (n_components={results['n_components']}):{Style.RESET_ALL}"
        )
        print(
            f"  {BackgroundColors.GREEN}Explained Variance Ratio: {BackgroundColors.CYAN}{truncate_value(results['explained_variance'])} ({truncate_value(results['explained_variance']*100)}%){Style.RESET_ALL}"
        )
        print(
            f"\n  {BackgroundColors.BOLD}{BackgroundColors.GREEN}10-Fold Cross-Validation Metrics (Training Set):{Style.RESET_ALL}"
        )
        print(
            f"  {BackgroundColors.GREEN}CV Accuracy: {BackgroundColors.CYAN}{truncate_value(results['cv_accuracy'])}{Style.RESET_ALL}"
        )
        print(
            f"  {BackgroundColors.GREEN}CV Precision: {BackgroundColors.CYAN}{truncate_value(results['cv_precision'])}{Style.RESET_ALL}"
        )
        print(f"  {BackgroundColors.GREEN}CV Recall: {BackgroundColors.CYAN}{truncate_value(results['cv_recall'])}{Style.RESET_ALL}")
        print(
            f"  {BackgroundColors.GREEN}CV F1-Score: {BackgroundColors.CYAN}{truncate_value(results['cv_f1_score'])}{Style.RESET_ALL}"
        )
        print(f"\n  {BackgroundColors.BOLD}{BackgroundColors.GREEN}Test Set Metrics:{Style.RESET_ALL}")
        print(
            f"  {BackgroundColors.GREEN}Test Accuracy: {BackgroundColors.CYAN}{truncate_value(results['test_accuracy'])}{Style.RESET_ALL}"
        )
        print(
            f"  {BackgroundColors.GREEN}Test Precision: {BackgroundColors.CYAN}{truncate_value(results['test_precision'])}{Style.RESET_ALL}"
        )
        print(
            f"  {BackgroundColors.GREEN}Test Recall: {BackgroundColors.CYAN}{truncate_value(results['test_recall'])}{Style.RESET_ALL}"
        )
        print(
            f"  {BackgroundColors.GREEN}Test F1-Score: {BackgroundColors.CYAN}{truncate_value(results['test_f1_score'])}{Style.RESET_ALL}"
        )
        print(f"  {BackgroundColors.GREEN}Test FPR: {BackgroundColors.CYAN}{truncate_value(results['test_fpr'])}{Style.RESET_ALL}")
        print(f"  {BackgroundColors.GREEN}Test FNR: {BackgroundColors.CYAN}{truncate_value(results['test_fnr'])}{Style.RESET_ALL}")
        print(
            f"  {BackgroundColors.GREEN}Training Time: {BackgroundColors.CYAN}{int(round(results['training_time_s']))}s  Testing Time: {BackgroundColors.CYAN}{int(round(results['testing_time_s']))}s{Style.RESET_ALL}"
        )
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


def populate_hardware_column(df, column_name="Hardware"):
    """
    Populate `df[column_name]` with a readable hardware description built from
    `get_hardware_specifications()`. On failure the column will be set to None.

    :param df: pandas.DataFrame to modify or reindex
    :param column_name: Name of the column to set (default: "Hardware")
    :return: DataFrame with hardware column ensured and positioned after `elapsed_time_s`
    """
    
    try:
        try:  # Try to fetch hardware specifications
            hardware_specs = get_hardware_specifications()  # Get system specs
            hardware_str = f"{hardware_specs.get('cpu_model','Unknown')} | Cores: {hardware_specs.get('cores', 'N/A')} | RAM: {hardware_specs.get('ram_gb',  'N/A')} GB | OS: {hardware_specs.get('os','Unknown')}"  # Build hardware string
            df[column_name] = hardware_str  # Set the hardware column
        except Exception:  # On any failure
            df[column_name] = None  # Set hardware column to None

        if "elapsed_time_s" in df.columns:  # If elapsed_time_s exists, reposition Hardware after it
            cols = list(df.columns)  # Get current columns
            if column_name in cols:  # If hardware column exists
                cols.remove(column_name)  # Remove it
            el_idx = (
                cols.index("elapsed_time_s") if "elapsed_time_s" in cols else len(cols) - 1
            )  # Find index of elapsed_time_s
            cols.insert(el_idx + 1, column_name)  # Insert hardware column after elapsed_time_s
            return df[cols]  # Reindex DataFrame with new column order

        return df  # Return DataFrame as-is if no repositioning needed
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
            
def save_pca_results(csv_path, all_results):
    """
    Saves PCA results to a single CSV file containing evaluation metadata
    and per-configuration metrics. This replaces separate JSON and CSV
    outputs with one machine-friendly CSV that includes the evaluation
    details repeated on each row for easy consumption by downstream tools.

    :param csv_path: Original CSV file path
    :param all_results: List of result dictionaries from different PCA configurations
    :return: None
    """
    
    try:
        output_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis/PCA/"  # Output directory under PCA subdir
        os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

        eval_model = "Random Forest"  # Evaluation model
        train_test_split = "80/20 split"  # Train/test split
        scaling = "StandardScaler"  # Scaling method

        rows = []  # List to store one normalized row per result
        
        evaluator_fallback = {"model": "RandomForestClassifier", "n_estimators": 100, "random_state": 42, "n_jobs": -1}
        cv_method = "StratifiedKFold(n_splits=10)"
        for results in all_results:
            eval_params = None
            trained_clf = results.get("trained_classifier")
            if trained_clf is not None:
                try:
                    eval_params = trained_clf.get_params()
                except Exception:
                    eval_params = None

            row = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"),
                "tool": "PCA",
                "model": eval_model,
                "dataset": os.path.relpath(csv_path),
                "hyperparameters": json.dumps(eval_params or evaluator_fallback, sort_keys=True, ensure_ascii=False, default=str),
                "cv_method": cv_method,
                "train_test_split": train_test_split,
                "scaling": scaling,
                "n_components": int(results.get("n_components")) if results.get("n_components") is not None else None,
                "explained_variance": truncate_value(results.get("explained_variance")),
                "cv_accuracy": truncate_value(results.get("cv_accuracy")),
                "cv_precision": truncate_value(results.get("cv_precision")),
                "cv_recall": truncate_value(results.get("cv_recall")),
                "cv_f1_score": truncate_value(results.get("cv_f1_score")),
                "test_accuracy": truncate_value(results.get("test_accuracy")),
                "test_precision": truncate_value(results.get("test_precision")),
                "test_recall": truncate_value(results.get("test_recall")),
                "test_f1_score": truncate_value(results.get("test_f1_score")),
                "test_fpr": truncate_value(results.get("test_fpr")),
                "test_fnr": truncate_value(results.get("test_fnr")),
                "training_time_s": results.get("training_time_s") if results.get("training_time_s") is not None else None,
                "testing_time_s": results.get("testing_time_s") if results.get("testing_time_s") is not None else None,
                "feature_extraction_time_s": results.get("feature_extraction_time_s") if results.get("feature_extraction_time_s") is not None else None,
            }
            rows.append(row)

        comparison_df = pd.DataFrame(rows)  # Create DataFrame from rows
        csv_output = f"{output_dir}/PCA_Results.csv"  # Output CSV path

        if os.path.exists(csv_output):
            try:
                df_existing = pd.read_csv(csv_output, dtype=str)
                if "timestamp" not in df_existing.columns:
                    mtime = os.path.getmtime(csv_output)
                    back_ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d_%H_%M_%S")
                    df_existing["timestamp"] = back_ts

                for c in PCA_RESULTS_CSV_COLUMNS:
                    if c not in df_existing.columns:
                        df_existing[c] = None

                df_combined = pd.concat([df_existing[PCA_RESULTS_CSV_COLUMNS], comparison_df], ignore_index=True, sort=False)

                try:
                    df_combined["timestamp_dt"] = pd.to_datetime(df_combined["timestamp"], format="%Y-%m-%d_%H_%M_%S", errors="coerce")
                    df_combined = df_combined.sort_values(by="timestamp_dt", ascending=False)
                    df_combined = df_combined.drop(columns=["timestamp_dt"])
                except Exception:
                    df_combined = df_combined.sort_values(by="timestamp", ascending=False)

                df_out = df_combined.reset_index(drop=True)
            except Exception:
                df_out = comparison_df
        else:
            df_out = comparison_df

        df_out = populate_hardware_column(df_out, column_name="hardware")  # Add hardware specs column (lowercase)

        try:
            df_out = df_out.reindex(columns=PCA_RESULTS_CSV_COLUMNS)
        except Exception:
            pass

        try:  # Attempt to save the CSV
            df_out.to_csv(csv_output, index=False)  # Save the DataFrame to CSV
            print(f"{BackgroundColors.GREEN}PCA results saved to {BackgroundColors.CYAN}{csv_output}{Style.RESET_ALL}")
        except Exception as e:  # Handle exceptions during file saving
            print(f"{BackgroundColors.RED}Failed to save PCA CSV: {e}{Style.RESET_ALL}")

        models_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis/PCA/Models/"
        os.makedirs(models_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        base_name = Path(csv_path).stem
        for results in all_results:
            n_comp = results["n_components"]
            pca_obj = results.get("pca_object")
            if pca_obj:
                pca_file = f"{models_dir}/PCA-{base_name}-{n_comp}c-{timestamp}.pkl"
                try:
                    with open(pca_file, "wb") as f:
                        pickle.dump(pca_obj, f)
                    verbose_output(f"{BackgroundColors.GREEN}PCA object saved to {BackgroundColors.CYAN}{pca_file}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{BackgroundColors.RED}Failed to save PCA object: {e}{Style.RESET_ALL}")
            scaler = results.get("scaler")
            clf = results.get("trained_classifier")
            if scaler is not None:
                scaler_path = f"{models_dir}/PCA-{base_name}-{n_comp}c-{timestamp}-scaler.joblib"
                try:
                    dump(scaler, scaler_path)
                    verbose_output(f"{BackgroundColors.GREEN}Scaler saved to {BackgroundColors.CYAN}{scaler_path}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{BackgroundColors.RED}Failed to save scaler: {e}{Style.RESET_ALL}")
            if clf is not None:
                model_path = f"{models_dir}/PCA-{base_name}-{n_comp}c-{timestamp}-model.joblib"
                try:
                    dump(clf, model_path)
                    verbose_output(f"{BackgroundColors.GREEN}Trained classifier saved to {BackgroundColors.CYAN}{model_path}{Style.RESET_ALL}")
                except Exception as e:
                    print(f"{BackgroundColors.RED}Failed to save classifier: {e}{Style.RESET_ALL}")

        verbose_output(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}PCA Configuration Comparison:{Style.RESET_ALL}")
        verbose_output(comparison_df.to_string(index=False))  # Output the comparison DataFrame
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_pca_analysis(csv_path, n_components_list=[8, 16, 24, 32, 48], parallel=True, max_workers=None):
    """
    Runs PCA analysis with different numbers of components and evaluates performance.

    :param csv_path: Path to the CSV dataset file
    :param n_components_list: List of component counts to test
    :return: None
    """
    
    try:
        global SKIP_TRAIN_IF_MODEL_EXISTS
        models_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis/PCA/Models/"
        base_name = Path(csv_path).stem
        timestamp = None
        if SKIP_TRAIN_IF_MODEL_EXISTS:
            found = False
            for n_comp in n_components_list:
                pattern = f"{models_dir}/PCA-{base_name}-{n_comp}c-*-model.joblib"
                matches = glob.glob(pattern)
                if matches:
                    print(f"{BackgroundColors.GREEN}Found exported model for n_components={n_comp}, skipping training.{Style.RESET_ALL}")
                    found = True
            if found:
                print(f"{BackgroundColors.GREEN}SKIP_TRAIN_IF_MODEL_EXISTS: At least one model exists, skipping retraining for all configs.{Style.RESET_ALL}")
                return

        df = load_dataset(csv_path)  # Load the dataset
        if df is None:  # If dataset loading failed
            return {}  # Return empty dictionary

        cleaned_df = preprocess_dataframe(df)  # Preprocess the DataFrame

        X = cleaned_df.select_dtypes(include=["number"]).iloc[:, :-1]  # Select numeric features (all columns except last)
        y = cleaned_df.iloc[:, -1]  # Select target variable (last column)

        n_components_list = [n for n in n_components_list if n > 0]  # Filter out non-positive component counts
        max_components = min(X.shape[1], max(n_components_list)) if n_components_list else 0  # Maximum valid components
        n_components_list = [n for n in n_components_list if n <= max_components]  # Filter valid component counts

        if not n_components_list:  # If no valid component counts remain
            print(
                f"{BackgroundColors.RED}No valid component counts. Dataset has only {X.shape[1]} features.{Style.RESET_ALL}"
            )
            return  # Exit the function

        print(f"\\n{BackgroundColors.CYAN}PCA Configuration:{Style.RESET_ALL}")
        print(
            f"  {BackgroundColors.GREEN}• Testing components: {BackgroundColors.CYAN}{n_components_list}{Style.RESET_ALL}"
        )
        print(
            f"  {BackgroundColors.GREEN}• Evaluation: {BackgroundColors.CYAN}10-Fold Stratified Cross-Validation{Style.RESET_ALL}"
        )
        print(f"  {BackgroundColors.GREEN}• Model: {BackgroundColors.CYAN}Random Forest (100 estimators){Style.RESET_ALL}")
        print(f"  {BackgroundColors.GREEN}• Split: {BackgroundColors.CYAN}80/20 (train/test){Style.RESET_ALL}\\n")

        X_train, X_test, y_train, y_test, scaler = scale_and_split(X, y)  # Scale and split the data

        all_results = []  # List to store all results

        executed_parallel = False  # Flag to track if parallel execution was successful

        if parallel and len(n_components_list) > 1:  # If parallel execution is enabled and multiple configurations
            try:  # Attempt parallel execution
                cpu_count = os.cpu_count() or 1  # Get the number of CPU cores
                workers = max_workers or min(len(n_components_list), cpu_count)  # Determine number of workers
                print(
                    f"{BackgroundColors.GREEN}Running {BackgroundColors.CYAN}PCA Analysis{BackgroundColors.GREEN} in Parallel with {BackgroundColors.CYAN}{workers}{BackgroundColors.GREEN} Worker(s)...{Style.RESET_ALL}"
                )

                results_map = {}  # Map to store results by n_components
                future_to_ncomp = {}  # Map each future to its n_components value
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=workers
                ) as executor:  # Create a process pool executor
                    for n_components in n_components_list:  # Loop over each number of components
                        scaling_time_val = getattr(scaler, "_scaling_time", 0.0)  # Retrieve scaling_time attached to scaler
                        fut = executor.submit(
                            apply_pca_and_evaluate, X_train, y_train, X_test, y_test, n_components, workers=workers, scaling_time=scaling_time_val
                        )  # Submit task to the executor with scaling_time
                        future_to_ncomp[fut] = n_components  # Map future to n_components

                    pbar = tqdm(
                        total=len(future_to_ncomp),
                        desc=f"{BackgroundColors.GREEN}PCA Analysis{Style.RESET_ALL}",
                        unit="config",
                        leave=False,
                    )  # Progress bar for tracking completion
                    try:  # Handle completion of futures
                        for fut in concurrent.futures.as_completed(future_to_ncomp):  # As each future completes
                            n = future_to_ncomp.get(fut)  # Get the corresponding n_components
                            pbar.set_description(
                                f"{BackgroundColors.GREEN}Processing PCA with {BackgroundColors.CYAN}{n}{BackgroundColors.GREEN} components{Style.RESET_ALL}"
                            )
                            try:  # Get the result from the future
                                res = fut.result()  # Get the result
                                results_map[res["n_components"]] = res  # Store result in the map
                                print_pca_results(res) if VERBOSE else None
                            except Exception as e:  # Handle exceptions from worker processes
                                print(f"{BackgroundColors.RED}Error in worker: {e}{Style.RESET_ALL}")
                            finally:  # Update the progress bar
                                pbar.update(1)  # Update progress bar
                    finally:  # Ensure progress bar is closed
                        pbar.close()  # Close the progress bar

                for n in n_components_list:  # Collect results in the original order
                    if n in results_map:  # If result exists for this n_components
                        all_results.append(results_map[n])  # Append to all_results
                executed_parallel = True  # Mark parallel execution as successful
            except Exception as e:  # Handle exceptions during parallel execution
                print(
                    f"{BackgroundColors.YELLOW}Parallel execution failed: {e}. Falling back to sequential execution.{Style.RESET_ALL}"
                )
                parallel = False  # Disable parallel for fallback

        if not executed_parallel:  # If parallel was not executed or failed, run sequential
            for n_components in tqdm(
                n_components_list, desc=f"{BackgroundColors.GREEN}PCA Analysis{Style.RESET_ALL}", unit="config"
            ):
                send_telegram_message(TELEGRAM_BOT, f"Starting PCA training for n_components={n_components}")
                print(
                    f"\\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Testing PCA with {BackgroundColors.CYAN}{n_components}{BackgroundColors.GREEN} components...{Style.RESET_ALL}"
                )
                comp_start = time.perf_counter()  # Start high-resolution timer for this component config
                scaling_time_val = getattr(scaler, "_scaling_time", 0.0)  # Retrieve scaling_time attached to scaler
                results = apply_pca_and_evaluate(
                    X_train, y_train, X_test, y_test, n_components, workers=1, scaling_time=scaling_time_val
                )  # Apply PCA and evaluate (single worker)
                comp_elapsed = time.perf_counter() - comp_start  # Compute elapsed duration
                send_telegram_message(TELEGRAM_BOT, f"Finished PCA training for n_components={n_components} with CV F1: {truncate_value(results['cv_f1_score'])} in {calculate_execution_time(comp_start, time.perf_counter())}")
                all_results.append(results)  # Append results to the list
                print_pca_results(results) if VERBOSE else None

        if not all_results:  # If no results were collected
            print(
                f"{BackgroundColors.RED}No results collected from PCA analysis. Verify for errors in worker processes.{Style.RESET_ALL}"
            )
            return  # Return

        save_pca_results(csv_path, all_results)  # Save all results to files

        best_result = max(all_results, key=lambda x: x["cv_f1_score"])  # Find the best configuration based on CV F1-Score

        print(f"\\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Best Configuration:{Style.RESET_ALL}")
        print(
            f"  {BackgroundColors.GREEN}n_components = {BackgroundColors.CYAN}{best_result['n_components']}{Style.RESET_ALL}"
        )
        print(
            f"  {BackgroundColors.GREEN}CV F1-Score = {BackgroundColors.CYAN}{truncate_value(best_result['cv_f1_score'])}{Style.RESET_ALL}"
        )
        print(
            f"  {BackgroundColors.GREEN}Explained Variance = {BackgroundColors.CYAN}{truncate_value(best_result['explained_variance'])}{Style.RESET_ALL}"
        )
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


def main():
    """
    Main function.

    :param: None
    :return: None
    """
    
    try:
        global VERBOSE, SKIP_TRAIN_IF_MODEL_EXISTS, CSV_FILE
        parser = argparse.ArgumentParser(description="PCA Feature Extraction & Evaluation Tool")
        parser.add_argument("--csv_file", type=str, default=CSV_FILE, help="Path to the CSV dataset file")
        parser.add_argument("--skip_train_if_model_exists", action="store_true", help="Skip training if exported model exists")
        parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
        parser.add_argument("--n_components_list", type=str, default="8,16,32,64", help="Comma-separated list of PCA component counts to test")
        parser.add_argument("--max_workers", type=int, default=-1, help="Number of parallel workers (default: -1)")
        args = parser.parse_args()

        VERBOSE = args.verbose
        SKIP_TRAIN_IF_MODEL_EXISTS = args.skip_train_if_model_exists
        CSV_FILE = args.csv_file
        n_components_list = [int(x) for x in args.n_components_list.split(",") if x.strip().isdigit()]
        max_workers = args.max_workers

        print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}PCA Feature Extraction{BackgroundColors.GREEN} program!{Style.RESET_ALL}")
        start_time = datetime.datetime.now()  # Get the start time of the program
        
        setup_telegram_bot()  # Setup Telegram bot if configured
        
        send_telegram_message(TELEGRAM_BOT, [f"Starting PCA Feature Extraction on {CSV_FILE} at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"])

        run_pca_analysis(CSV_FILE, n_components_list, max_workers=max_workers)  # Run the PCA analysis

        finish_time = datetime.datetime.now()  # Get the finish time of the program
        print(
            f"{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
        )  # Output the start and finish times
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
        )  # Output the end of the program message
        
        send_telegram_message(TELEGRAM_BOT, [f"PCA Feature Extraction completed on {CSV_FILE} at {finish_time.strftime('%Y-%m-%d %H:%M:%S')}.\nExecution time: {calculate_execution_time(start_time, finish_time)}"])

        (
            atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None
        )  # Register the play_sound function to be called at exit if enabled
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


if __name__ == "__main__":
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """

    main()
