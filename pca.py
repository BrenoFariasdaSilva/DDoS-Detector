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
import json  # For serializing hyperparameters and other metadata
import math  # For mathematical operations
import numpy as np  # For numerical operations
import os  # For file and directory operations
import pandas as pd  # For data manipulation
import pickle  # For serializing PCA objects
import platform  # For getting the operating system name
import psutil  # For system memory and CPU counts
import subprocess  # For fetching CPU model on some OSes
import sys  # For system-specific parameters and functions
import time  # For measuring elapsed time
from colorama import Style  # For coloring the terminal
from joblib import dump  # For saving scalers and models
from Logger import Logger  # For logging output to both terminal and file
from pathlib import Path  # For handling file paths
from sklearn.decomposition import PCA  # For Principal Component Analysis
from sklearn.ensemble import RandomForestClassifier  # For the Random Forest model
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)  # For performance metrics
from sklearn.model_selection import train_test_split, StratifiedKFold  # For splitting and cross-validation
from sklearn.preprocessing import StandardScaler  # For scaling the data (standardization)
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
    "elapsed_time_s",
    "hardware",
]

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


def verbose_output(true_string="", false_string=""):
    """
    Outputs a message if the VERBOSE constant is set to True.

    :param true_string: The string to be outputted if the VERBOSE constant is set to True.
    :param false_string: The string to be outputted if the VERBOSE constant is set to False.
    :return: None
    """

    if VERBOSE and true_string != "":  # If VERBOSE is True and a true_string was provided
        print(true_string)
    elif false_string != "":  # If a false_string was provided
        print(false_string)


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


def preprocess_dataframe(df, remove_zero_variance=True):
    """
    Preprocess a DataFrame by removing rows with NaN or infinite values and
    dropping zero-variance numeric features.

    :param df: pandas DataFrame to preprocess
    :param remove_zero_variance: whether to drop numeric columns with zero variance
    :return: cleaned DataFrame
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Preprocessing the DataFrame by removing NaN/infinite values and zero-variance features.{Style.RESET_ALL}"
    )  # Output the verbose message

    if df is None:  # If the DataFrame is None
        return df  # Return None

    # Strip whitespace from all column names
    df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names

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
    Scales numeric features and splits into train/test sets.

    :param X: Features DataFrame
    :param y: Target Series
    :param test_size: Proportion of the dataset to include in the test split
    :param random_state: Random seed for reproducibility
    :return: X_train, X_test, y_train, y_test, scaler
    """

    # Perform the train/test split first to avoid data leakage
    stratify_param = y if len(np.unique(y)) > 1 else None
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
    )

    # Fit the scaler on the training data only, then transform both partitions
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df)
    X_test = scaler.transform(X_test_df)

    return X_train, X_test, y_train, y_test, scaler  # Return the split data and scaler


def apply_pca_and_evaluate(X_train, y_train, X_test, y_test, n_components, cv_folds=10, workers=1):
    """
    Applies PCA transformation and evaluates performance using 10-fold Stratified Cross-Validation
    on the training set, then tests on the held-out test set.

    :param X_train: Training features (scaled)
    :param y_train: Training target
    :param X_test: Testing features (scaled)
    :param y_test: Testing target
    :param n_components: Number of principal components to keep
    :param cv_folds: Number of cross-validation folds (default: 10)
    :return: Dictionary containing metrics, explained variance, and PCA object
    """

    if n_components <= 0:  # Validate n_components
        raise ValueError(f"n_components must be positive, got {n_components}")
    if n_components > X_train.shape[1]:  # Validate n_components against number of features
        raise ValueError(
            f"n_components ({n_components}) cannot be greater than number of features ({X_train.shape[1]})"
        )

    pca = PCA(n_components=n_components, random_state=42)  # Initialize PCA

    X_train_pca = pca.fit_transform(X_train)  # Fit PCA on training data and transform
    X_test_pca = pca.transform(X_test)  # Transform test data using the fitted PCA

    explained_variance = pca.explained_variance_ratio_.sum()  # Total explained variance ratio

    rf_n_jobs = -1 if workers == 1 else 1  # Set n_jobs for RandomForest based on workers
    model = RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=rf_n_jobs
    )  # Initialize Random Forest model

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)  # Stratified K-Fold cross-validator

    cv_accs, cv_precs, cv_recs, cv_f1s = [], [], [], []  # Lists to store CV metrics

    for train_idx, val_idx in skf.split(X_train_pca, y_train):  # Loop over each fold
        X_train_fold = X_train_pca[train_idx]  # Training data for this fold
        X_val_fold = X_train_pca[val_idx]  # Validation data for this fold
        y_train_fold = (
            y_train.iloc[train_idx] if isinstance(y_train, pd.Series) else y_train[train_idx]
        )  # Training target for this fold
        y_val_fold = (
            y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx]
        )  # Validation target for this fold

        model.fit(X_train_fold, y_train_fold)  # Fit model on training fold
        y_pred_fold = model.predict(X_val_fold)  # Predict on validation fold

        cv_accs.append(accuracy_score(y_val_fold, y_pred_fold))  # Calculate and store metrics
        cv_precs.append(
            precision_score(y_val_fold, y_pred_fold, average="weighted", zero_division=0)
        )  # Calculate and store metrics
        cv_recs.append(
            recall_score(y_val_fold, y_pred_fold, average="weighted", zero_division=0)
        )  # Calculate and store metrics
        cv_f1s.append(
            f1_score(y_val_fold, y_pred_fold, average="weighted", zero_division=0)
        )  # Calculate and store metrics

    cv_acc_mean = np.mean(cv_accs)  # Mean CV metrics
    cv_prec_mean = np.mean(cv_precs)  # Mean CV metrics
    cv_rec_mean = np.mean(cv_recs)  # Mean CV metrics
    cv_f1_mean = np.mean(cv_f1s)  # Mean CV metrics

    start_time = time.time()  # Start timing for test set evaluation
    model.fit(X_train_pca, y_train)  # Fit model on full training data
    y_pred = model.predict(X_test_pca)  # Predict on test data
    elapsed_time = time.time() - start_time  # Elapsed time for test evaluation

    acc = accuracy_score(y_test, y_pred)  # Calculate test metrics
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate test metrics
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate test metrics
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate test metrics

    fpr, fnr = 0, 0  # Initialize FPR and FNR
    unique_classes = np.unique(y_test)  # Get unique classes in the test set
    if len(unique_classes) == 2:  # If binary classification
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=unique_classes).ravel()  # Get confusion matrix values
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Calculate False Positive Rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # Calculate False Negative Rate

    # Attach scaler and trained classifier for export
    scaler_export = StandardScaler().fit(np.vstack([X_train, X_test]))
    return {
        "n_components": n_components,
        "explained_variance": explained_variance,
        "cv_accuracy": cv_acc_mean,
        "cv_precision": cv_prec_mean,
        "cv_recall": cv_rec_mean,
        "cv_f1_score": cv_f1_mean,
        "test_accuracy": acc,
        "test_precision": prec,
        "test_recall": rec,
        "test_f1_score": f1,
        "test_fpr": fpr,
        "test_fnr": fnr,
        "elapsed_time_s": elapsed_time,
        "pca_object": pca,
        "scaler": scaler_export,
        "trained_classifier": model,
    }


def print_pca_results(results):
    """
    Prints PCA results in a formatted way.

    :param results: Dictionary containing PCA evaluation results
    :return: None
    """

    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}PCA Results (n_components={results['n_components']}):{Style.RESET_ALL}"
    )
    print(
        f"  {BackgroundColors.GREEN}Explained Variance Ratio: {BackgroundColors.CYAN}{results['explained_variance']:.4f} ({results['explained_variance']*100:.2f}%){Style.RESET_ALL}"
    )
    print(
        f"\n  {BackgroundColors.BOLD}{BackgroundColors.GREEN}10-Fold Cross-Validation Metrics (Training Set):{Style.RESET_ALL}"
    )
    print(
        f"  {BackgroundColors.GREEN}CV Accuracy: {BackgroundColors.CYAN}{results['cv_accuracy']:.4f}{Style.RESET_ALL}"
    )
    print(
        f"  {BackgroundColors.GREEN}CV Precision: {BackgroundColors.CYAN}{results['cv_precision']:.4f}{Style.RESET_ALL}"
    )
    print(f"  {BackgroundColors.GREEN}CV Recall: {BackgroundColors.CYAN}{results['cv_recall']:.4f}{Style.RESET_ALL}")
    print(
        f"  {BackgroundColors.GREEN}CV F1-Score: {BackgroundColors.CYAN}{results['cv_f1_score']:.4f}{Style.RESET_ALL}"
    )
    print(f"\n  {BackgroundColors.BOLD}{BackgroundColors.GREEN}Test Set Metrics:{Style.RESET_ALL}")
    print(
        f"  {BackgroundColors.GREEN}Test Accuracy: {BackgroundColors.CYAN}{results['test_accuracy']:.4f}{Style.RESET_ALL}"
    )
    print(
        f"  {BackgroundColors.GREEN}Test Precision: {BackgroundColors.CYAN}{results['test_precision']:.4f}{Style.RESET_ALL}"
    )
    print(
        f"  {BackgroundColors.GREEN}Test Recall: {BackgroundColors.CYAN}{results['test_recall']:.4f}{Style.RESET_ALL}"
    )
    print(
        f"  {BackgroundColors.GREEN}Test F1-Score: {BackgroundColors.CYAN}{results['test_f1_score']:.4f}{Style.RESET_ALL}"
    )
    print(f"  {BackgroundColors.GREEN}Test FPR: {BackgroundColors.CYAN}{results['test_fpr']:.4f}{Style.RESET_ALL}")
    print(f"  {BackgroundColors.GREEN}Test FNR: {BackgroundColors.CYAN}{results['test_fnr']:.4f}{Style.RESET_ALL}")
    print(
        f"  {BackgroundColors.GREEN}Elapsed Time: {BackgroundColors.CYAN}{results['elapsed_time_s']:.2f}s{Style.RESET_ALL}"
    )


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

        elif system == "Darwin":  # macOS: use sysctl
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


def populate_hardware_column(df, column_name="Hardware"):
    """
    Populate `df[column_name]` with a readable hardware description built from
    `get_hardware_specifications()`. On failure the column will be set to None.

    :param df: pandas.DataFrame to modify or reindex
    :param column_name: Name of the column to set (default: "Hardware")
    :return: DataFrame with hardware column ensured and positioned after `elapsed_time_s`
    """

    try:  # Try to fetch hardware specifications
        hardware_specs = get_hardware_specifications()  # Get system specs
        hardware_str = f"{hardware_specs.get('cpu_model','Unknown')} | Cores: {hardware_specs.get('cores', 'N/A')} | RAM: {hardware_specs.get('ram_gb', 'N/A')} GB | OS: {hardware_specs.get('os','Unknown')}"  # Build hardware string
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


def get_file_size_string(file_path):
    """
    Get the file size in human-readable format (KB, MB, GB).

    :param file_path: Path to the file
    :return: String representing the file size
    """

    file_size = os.path.getsize(file_path)  # Get the file size in bytes

    if file_size < 1024 * 1024:  # If less than 1 MB
        size_str = f"{file_size / 1024:.2f} KB"  # Size in KB
    elif file_size < 1024 * 1024 * 1024:  # If less than 1 GB
        size_str = f"{file_size / (1024 * 1024):.2f} MB"  # Size in MB
    else:  # Otherwise
        size_str = f"{file_size / (1024 * 1024 * 1024):.2f} GB"  # Size in GB

    return size_str  # Return the size string


def format_value(value):
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

    output_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis/PCA/"  # Output directory under PCA subdir
    os.makedirs(output_dir, exist_ok=True)  # Create output directory if it doesn't exist

    eval_model = "Random Forest"  # Evaluation model
    train_test_split = "80/20 split"  # Train/test split
    scaling = "StandardScaler"  # Scaling method

    rows = []  # List to store one normalized row per result
    evaluator_hyperparams = {"model": "RandomForestClassifier", "n_estimators": 100, "random_state": 42, "n_jobs": -1}
    cv_method = "StratifiedKFold(n_splits=10)"
    for results in all_results:

        # Build a single normalized row with canonical fields
        row = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"),
            "tool": "PCA",
            "model": eval_model,
            "dataset": os.path.relpath(csv_path),
            "hyperparameters": json.dumps(evaluator_hyperparams),
            "cv_method": cv_method,
            "train_test_split": train_test_split,
            "scaling": scaling,
            "n_components": int(results.get("n_components")) if results.get("n_components") is not None else None,
            "explained_variance": format_value(results.get("explained_variance")),
            "cv_accuracy": format_value(results.get("cv_accuracy")),
            "cv_precision": format_value(results.get("cv_precision")),
            "cv_recall": format_value(results.get("cv_recall")),
            "cv_f1_score": format_value(results.get("cv_f1_score")),
            "test_accuracy": format_value(results.get("test_accuracy")),
            "test_precision": format_value(results.get("test_precision")),
            "test_recall": format_value(results.get("test_recall")),
            "test_f1_score": format_value(results.get("test_f1_score")),
            "test_fpr": format_value(results.get("test_fpr")),
            "test_fnr": format_value(results.get("test_fnr")),
            "elapsed_time_s": int(results.get("elapsed_time_s")),
        }
        rows.append(row)

    comparison_df = pd.DataFrame(rows)  # Create DataFrame from rows
    csv_output = f"{output_dir}/PCA_Results.csv"  # Output CSV path

    # If existing file present, read and backfill timestamp if missing, then concat
    if os.path.exists(csv_output):
        try:
            df_existing = pd.read_csv(csv_output, dtype=str)
            # Backfill timestamp for legacy files without timestamp column
            if "timestamp" not in df_existing.columns:
                mtime = os.path.getmtime(csv_output)
                back_ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d_%H_%M_%S")
                df_existing["timestamp"] = back_ts

            # Ensure all expected columns exist in existing dataframe
            for c in PCA_RESULTS_CSV_COLUMNS:
                if c not in df_existing.columns:
                    df_existing[c] = None

            df_combined = pd.concat([df_existing[PCA_RESULTS_CSV_COLUMNS], comparison_df], ignore_index=True, sort=False)

            # Sort newest -> oldest by timestamp and reindex to canonical order
            try:
                df_combined["timestamp_dt"] = pd.to_datetime(df_combined["timestamp"], format="%Y-%m-%d_%H_%M_%S", errors="coerce")
                df_combined = df_combined.sort_values(by="timestamp_dt", ascending=False)
                df_combined = df_combined.drop(columns=["timestamp_dt"])
            except Exception:
                # fallback: string-sort (format chosen is lexicographically sortable)
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


def run_pca_analysis(csv_path, n_components_list=[8, 16, 24, 32, 48], parallel=True, max_workers=None):
    """
    Runs PCA analysis with different numbers of components and evaluates performance.

    :param csv_path: Path to the CSV dataset file
    :param n_components_list: List of component counts to test
    :return: None
    """

    # --- SKIP_TRAIN_IF_MODEL_EXISTS logic ---
    global SKIP_TRAIN_IF_MODEL_EXISTS
    models_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis/PCA/Models/"
    base_name = Path(csv_path).stem
    timestamp = None
    if SKIP_TRAIN_IF_MODEL_EXISTS:
        import glob
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

    print(f"\n{BackgroundColors.CYAN}PCA Configuration:{Style.RESET_ALL}")
    print(
        f"  {BackgroundColors.GREEN}• Testing components: {BackgroundColors.CYAN}{n_components_list}{Style.RESET_ALL}"
    )
    print(
        f"  {BackgroundColors.GREEN}• Evaluation: {BackgroundColors.CYAN}10-Fold Stratified Cross-Validation{Style.RESET_ALL}"
    )
    print(f"  {BackgroundColors.GREEN}• Model: {BackgroundColors.CYAN}Random Forest (100 estimators){Style.RESET_ALL}")
    print(f"  {BackgroundColors.GREEN}• Split: {BackgroundColors.CYAN}80/20 (train/test){Style.RESET_ALL}\n")

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
                    fut = executor.submit(
                        apply_pca_and_evaluate, X_train, y_train, X_test, y_test, n_components, workers=workers
                    )  # Submit task to the executor
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
            print(
                f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Testing PCA with {BackgroundColors.CYAN}{n_components}{BackgroundColors.GREEN} components...{Style.RESET_ALL}"
            )
            results = apply_pca_and_evaluate(
                X_train, y_train, X_test, y_test, n_components, workers=1
            )  # Apply PCA and evaluate (single worker)
            all_results.append(results)  # Append results to the list
            print_pca_results(results) if VERBOSE else None

    if not all_results:  # If no results were collected
        print(
            f"{BackgroundColors.RED}No results collected from PCA analysis. Verify for errors in worker processes.{Style.RESET_ALL}"
        )
        return  # Return

    save_pca_results(csv_path, all_results)  # Save all results to files

    best_result = max(all_results, key=lambda x: x["cv_f1_score"])  # Find the best configuration based on CV F1-Score

    print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Best Configuration:{Style.RESET_ALL}")
    print(
        f"  {BackgroundColors.GREEN}n_components = {BackgroundColors.CYAN}{best_result['n_components']}{Style.RESET_ALL}"
    )
    print(
        f"  {BackgroundColors.GREEN}CV F1-Score = {BackgroundColors.CYAN}{best_result['cv_f1_score']:.4f}{Style.RESET_ALL}"
    )
    print(
        f"  {BackgroundColors.GREEN}Explained Variance = {BackgroundColors.CYAN}{best_result['explained_variance']:.4f}{Style.RESET_ALL}"
    )


def calculate_execution_time(start_time, finish_time):
    """
    Calculates the execution time between start and finish times and formats it as hh:mm:ss.

    :param start_time: The start datetime object
    :param finish_time: The finish datetime object
    :return: String formatted as hh:mm:ss representing the execution time
    """

    delta = finish_time - start_time  # Calculate the time difference
    hours, remainder = divmod(delta.seconds, 3600)  # Calculate the hours, minutes and seconds
    minutes, seconds = divmod(remainder, 60)  # Calculate the minutes and seconds
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"  # Format the execution time


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


def main():
    """
    Main function.

    :param: None
    :return: None
    """
    
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

    run_pca_analysis(CSV_FILE, n_components_list, max_workers=max_workers)  # Run the PCA analysis

    finish_time = datetime.datetime.now()  # Get the finish time of the program
    print(
        f"{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
    )  # Output the start and finish times
    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
    )  # Output the end of the program message

    (
        atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None
    )  # Register the play_sound function to be called at exit if enabled


if __name__ == "__main__":
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """

    main()
