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
import datetime  # For timestamping
import json  # For saving lists and dicts as JSON strings
import math  # For mathematical operations
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical operations
import os  # For file and directory operations
import pandas as pd  # For data manipulation
import platform  # For getting the operating system name
import psutil  # For hardware information
import re  # For regular expressions
import subprocess  # For executing system commands
import sys  # For system-specific parameters and functions
import time  # For measuring elapsed time
from colorama import Style  # For coloring the terminal
from joblib import dump, load  # For exporting and loading trained models and scalers
import glob  # For finding exported model files
from Logger import Logger  # For logging output to both terminal and file
from pathlib import Path  # For handling file paths
from sklearn.ensemble import RandomForestClassifier  # For the Random Forest model
from sklearn.feature_selection import RFE  # For Recursive Feature Elimination
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)  # For performance metrics
from sklearn.model_selection import StratifiedKFold, train_test_split  # For train/test split and stratified K-Fold CV
from sklearn.preprocessing import StandardScaler  # For scaling the data (standardization)


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
    "training_time_s",
    "testing_time_s",
    "hardware",
    "top_features",
    "rfe_ranking",
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


def safe_filename(name):
    """
    Converts a string to a safe filename by replacing invalid characters with underscores.

    :param name: The original string
    :return: A safe filename string
    """

    return re.sub(r'[\\/*?:"<>|]', "_", name)  # Replace invalid characters with underscores


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
    :return: X_train, X_test, y_train, y_test, feature_columns
    """

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


def run_rfe_selector(X_train, y_train, n_select=10, random_state=42):
    """
    Runs RFE with RandomForestClassifier and returns the selector object.

    :param X_train: Training features
    :param y_train: Training target
    :param n_select: Number of features to select
    :param random_state: Random seed for reproducibility
    :return: selector (fitted RFE object)
    """

    model = RandomForestClassifier(
        n_estimators=100, random_state=random_state, n_jobs=N_JOBS
    )  # Initialize the Random Forest model
    n_features = X_train.shape[1]  # Get the number of features
    n_select = n_select if n_features >= n_select else n_features  # Adjust n_select if more than available features

    selector = RFE(model, n_features_to_select=n_select, step=1)  # Initialize RFE
    selector = selector.fit(X_train, y_train)  # Fit RFE

    return selector, model  # Return the fitted selector and model


def compute_rfe_metrics(selector, X_train, X_test, y_train, y_test, random_state=42):
    """
    Computes performance metrics using the RFE-selected features.

    :param selector: Fitted RFE object
    :param X_train: Training features
    :param X_test: Testing features
    :param y_train: Training target
    :param y_test: Testing target
    :param random_state: Random seed for reproducibility
    :return: metrics tuple (acc, prec, rec, f1, fpr, fnr, elapsed_time)
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Computing performance metrics using RFE-selected features...{Style.RESET_ALL}"
    )  # Output the verbose message

    support = selector.support_  # Get the mask of selected features
    X_train_selected = X_train[:, support]  # Select training features
    X_test_selected = X_test[:, support]  # Select testing features

    model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=N_JOBS)  # Initialize the model

    start_time = time.time()  # Start time measurement
    model.fit(X_train_selected, y_train)  # Fit the model on selected features
    y_pred = model.predict(X_test_selected)  # Predict on selected test features
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

    elapsed_time = time.time() - start_time  # Calculate elapsed time
    return (
        float(acc),
        float(prec),
        float(rec),
        float(f1),
        float(fpr),
        float(fnr),
        float(elapsed_time),
    )  # Return the metrics as Python floats


def extract_top_features(selector, X_columns):
    """
    Returns top selected features and their RFE rankings.

    :param selector: Fitted RFE object
    :param X_columns: Original feature column names
    :return: top_features list, rfe_ranking dict
    """

    rfe_ranking = {
        f: r for f, r in zip(X_columns, selector.ranking_)
    }  # Map normalized feature names to their RFE rankings
    rfe_ranking = {k: int(v) for k, v in rfe_ranking.items()}  # Convert numpy types to Python int
    top_features = [f for f, s in zip(X_columns, selector.support_) if s]  # List of top selected features

    return top_features, rfe_ranking  # Return the top features and their rankings


def print_top_features(top_features, rfe_ranking):
    """
    Prints top features and their RFE rankings to the terminal.

    :param top_features: List of top features
    :param rfe_ranking: Dict mapping normalized feature names to RFE rankings
    """

    print(f"\n{BackgroundColors.BOLD}Top {len(top_features)} features selected by RFE:{Style.RESET_ALL}")

    for i, feat in enumerate(top_features, start=1):  # Print each top feature with its ranking
        rank_info = (
            f" {BackgroundColors.GREEN}(RFE ranking {BackgroundColors.CYAN}{rfe_ranking[feat]}{Style.RESET_ALL})"
            if feat in rfe_ranking
            else " (RFE ranking N/A)"
        )  # Get ranking info
        print(f"{i}. {feat}{rank_info}")  # Print the feature and its ranking


def print_metrics(metrics_tuple):
    """
    Prints metrics for the current run to the terminal.

    :param metrics_tuple: Tuple of average metrics
    """

    print(f"\n{BackgroundColors.BOLD}Average Metrics:{Style.RESET_ALL}")
    print(f"  {BackgroundColors.GREEN}Accuracy: {BackgroundColors.CYAN}{metrics_tuple[0]:.4f}{Style.RESET_ALL}")
    print(f"  {BackgroundColors.GREEN}Precision: {BackgroundColors.CYAN}{metrics_tuple[1]:.4f}{Style.RESET_ALL}")
    print(f"  {BackgroundColors.GREEN}Recall: {BackgroundColors.CYAN}{metrics_tuple[2]:.4f}{Style.RESET_ALL}")
    print(f"  {BackgroundColors.GREEN}F1-Score: {BackgroundColors.CYAN}{metrics_tuple[3]:.4f}{Style.RESET_ALL}")
    print(
        f"  {BackgroundColors.GREEN}False Positive Rate (FPR): {BackgroundColors.CYAN}{metrics_tuple[4]:.4f}{Style.RESET_ALL}"
    )
    print(
        f"  {BackgroundColors.GREEN}False Negative Rate (FNR): {BackgroundColors.CYAN}{metrics_tuple[5]:.4f}{Style.RESET_ALL}"
    )
    print(f"  {BackgroundColors.GREEN}Elapsed Time: {BackgroundColors.CYAN}{metrics_tuple[6]:.2f}s{Style.RESET_ALL}")


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


def populate_hardware_column_and_order(df, column_name="hardware"):
    """
    Add a hardware-specs column to `df` and reorder columns so the hardware
    column appears immediately after `elapsed_time_s`.

    :param df: pandas DataFrame with RFE results
    :param column_name: name for the hardware column
    :return: reordered DataFrame with hardware column added
    """

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

    scaler_full = StandardScaler()  # create a scaler for full-data training
    X_full_scaled = scaler_full.fit_transform(X_numeric.values)  # scale all numeric features
    sel_indices = [i for i, f in enumerate(feature_columns) if f in top_features]  # get indices for top features
    X_final = X_full_scaled[:, sel_indices] if sel_indices else X_full_scaled  # select columns or keep all if none
    final_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=N_JOBS)  # instantiate final RF
    final_model.fit(X_final, y_array)  # fit final model on entire dataset using selected features

    models_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis/RFE/Models/"  # models output directory under RFE/Models subdir
    os.makedirs(models_dir, exist_ok=True)  # ensure directory exists
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")  # timestamp for filenames (YYYY_MM_DD-HH_MM_SS)
    base_name = safe_filename(Path(csv_path).stem)  # safe base name from dataset path
    model_path = f"{models_dir}RFE-{base_name}-{timestamp}-model.joblib"  # model file path
    scaler_path = f"{models_dir}RFE-{base_name}-{timestamp}-scaler.joblib"  # scaler file path
    features_path = f"{models_dir}RFE-{base_name}-{timestamp}-features.json"  # selected features file path
    params_path = f"{models_dir}RFE-{base_name}-{timestamp}-params.json"  # hyperparameters file path
    dump(final_model, model_path)  # save trained model to disk
    dump(scaler_full, scaler_path)  # save fitted scaler to disk
    
    with open(features_path, "w", encoding="utf-8") as fh:  # write selected features to json
        fh.write(json.dumps(top_features))  # save feature list as JSON
        
    # Save model hyperparameters (so we can reproduce training configuration)
    model_params = final_model.get_params()  # get hyperparameters from trained estimator
    with open(params_path, "w", encoding="utf-8") as ph:  # write params to json
        ph.write(json.dumps(model_params, default=str))  # save params as JSON (default=str for non-serializable)

    print(f"{BackgroundColors.GREEN}Saved final model to {BackgroundColors.CYAN}{model_path}{Style.RESET_ALL}")  # notify saved model
    print(f"{BackgroundColors.GREEN}Saved scaler to {BackgroundColors.CYAN}{scaler_path}{Style.RESET_ALL}")  # notify saved scaler
    print(f"{BackgroundColors.GREEN}Saved params to {BackgroundColors.CYAN}{params_path}{Style.RESET_ALL}")  # notify saved params

    return final_model, scaler_full, top_features, model_path, scaler_path, features_path, model_params, params_path  # return objects, paths and params


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


def save_rfe_results(csv_path, run_results):
    """
    Saves results from RFE run to a structured CSV file.

    :param csv_path: Original CSV file path
    :param run_results: List of dicts containing results from the current run
    """

    verbose_output(f"{BackgroundColors.GREEN}Saving RFE run results to CSV...{Style.RESET_ALL}")

    runs = run_results if isinstance(run_results, list) else [run_results]

    rows = []
    for r in runs:
        data = {c: None for c in RFE_RESULTS_CSV_COLUMNS}
        # Timestamp for this row (YYYY-MM-DD_HH_MM_SS)
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
                    data[col] = format_value(float(val))
                except Exception:
                    data[col] = val

        training_time = r.get("training_time_s")
        try:
            data["training_time_s"] = format_value(float(training_time)) if training_time is not None else None
        except Exception:
            data["training_time_s"] = None

        testing_time = r.get("testing_time_s")
        try:
            data["testing_time_s"] = format_value(float(testing_time)) if testing_time is not None else None
        except Exception:
            data["testing_time_s"] = None

        try:
            data["top_features"] = json.dumps(r.get("top_features") or [], ensure_ascii=False)
        except Exception:
            data["top_features"] = str(r.get("top_features") or [])

        try:
            data["rfe_ranking"] = json.dumps(r.get("rfe_ranking") or {}, sort_keys=True, ensure_ascii=False)
        except Exception:
            data["rfe_ranking"] = str(r.get("rfe_ranking") or {})

        rows.append(data)

    # Build DataFrame and save to CSV
    df_new = pd.DataFrame(rows, columns=RFE_RESULTS_CSV_COLUMNS)
    output_dir = os.path.join(os.path.dirname(csv_path), "Feature_Analysis", "RFE")
    os.makedirs(output_dir, exist_ok=True)
    run_csv_path = os.path.join(output_dir, "RFE_Run_Results.csv")

    # If existing file present, read and backfill timestamp if missing, then concat
    if os.path.exists(run_csv_path):
        try:
            df_existing = pd.read_csv(run_csv_path, dtype=str)
            # Backfill timestamp for legacy files without timestamp column
            if "timestamp" not in df_existing.columns:
                mtime = os.path.getmtime(run_csv_path)
                back_ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d_%H_%M_%S")
                df_existing["timestamp"] = back_ts

            # Ensure all expected columns exist in existing dataframe
            for c in RFE_RESULTS_CSV_COLUMNS:
                if c not in df_existing.columns:
                    df_existing[c] = None

            df_combined = pd.concat([df_existing[RFE_RESULTS_CSV_COLUMNS], df_new], ignore_index=True, sort=False)

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
            df_out = df_new
    else:
        df_out = df_new

    # Populate hardware metadata and enforce ordering
    df_out = populate_hardware_column_and_order(df_out, column_name="hardware")

    try:
        df_out.to_csv(run_csv_path, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
        print(f"{BackgroundColors.GREEN}Run results saved to {BackgroundColors.CYAN}{run_csv_path}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{BackgroundColors.RED}Failed to save run results to CSV: {e}{Style.RESET_ALL}")

    return run_csv_path


def print_run_summary(run_results):
    """
    Print a concise run summary to the terminal.

    :param run_results: list containing a single run-results dict
    :return: None
    """
    
    if not run_results:
        return
    
    res = run_results[0]
    print(f"\n{BackgroundColors.BOLD}Run summary:{Style.RESET_ALL}")
    print(f"  Model: {res.get('model')}")
    print(f"  CV Method: {res.get('cv_method')}")
    print(
        f"  Accuracy: {res.get('test_accuracy', res.get('accuracy')):.4f}  Precision: {res.get('test_precision', res.get('precision')):.4f}  Recall: {res.get('test_recall', res.get('recall')):.4f}  F1: {res.get('test_f1_score', res.get('f1_score')):.4f}"
    )
    print(f"  FPR: {res.get('test_fpr', res.get('fpr')):.4f}  FNR: {res.get('test_fnr', res.get('fnr')):.4f}  Elapsed: {res.get('elapsed_time_s')}s")
    print(f"  Top features: {res.get('top_features')}")
    if res.get("hyperparameters"):
        try:
            hp = json.loads(res.get("hyperparameters")) if isinstance(res.get("hyperparameters"), str) else res.get("hyperparameters")
            print(f"  Hyperparameters: {json.dumps(hp)}")
        except Exception:
            print(f"  Hyperparameters: {res.get('hyperparameters')}")


def load_exported_artifacts(csv_path):
    """Attempt to locate and load latest exported model, scaler and features for csv_path.

    :param csv_path: original dataset path used to name exported artifacts
    :return: (model, scaler, features) or None if not found
    """

    models_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis/RFE/Models/"  # location where RFE artifacts are stored
    
    if not os.path.isdir(models_dir):
        return None  # no models directory

    base_name = safe_filename(Path(csv_path).stem)  # safe base name
    pattern = os.path.join(models_dir, f"RFE-{base_name}-*-model.joblib")  # glob pattern for RFE model files
    candidates = glob.glob(pattern)  # find matching model files
    if not candidates:
        return None  # no exported models found

    # pick latest by modification time
    latest_model = max(candidates, key=os.path.getmtime)  # select most recent model file
    scaler_path = latest_model.replace("-model.joblib", "-scaler.joblib")  # infer scaler path
    features_path = latest_model.replace("-model.joblib", "-features.json")  # infer features path
    if not os.path.exists(scaler_path) or not os.path.exists(features_path):
        return None  # incomplete artifact set

    try:
        model = load(latest_model)  # load model with joblib
        scaler = load(scaler_path)  # load scaler with joblib
        with open(features_path, "r", encoding="utf-8") as fh:
            features = json.load(fh)  # load features list
        params = None
        params_path = latest_model.replace("-model.joblib", "-params.json")  # infer params path
        if os.path.exists(params_path):
            try:
                with open(params_path, "r", encoding="utf-8") as ph:
                    params = json.load(ph)  # load hyperparameters if available
            except Exception:
                params = None
        return model, scaler, features, params
    except Exception:
        return None  # any loading error -> treat as not found


def evaluate_exported_model(model, scaler, X_numeric, feature_columns, top_features, y_array):
    """Evaluate a loaded/trained model on the full numeric dataset and
    compute the same metrics used by the RFE pipeline.

    :return: tuple (acc, prec, rec, f1, fpr, fnr, elapsed_time)
    """

    start_time = time.time()  # measure prediction/eval time
    X_scaled = scaler.transform(X_numeric.values)  # scale full numeric data with provided scaler
    sel_indices = [i for i, f in enumerate(feature_columns) if f in top_features]  # indices for chosen features
    X_eval = X_scaled[:, sel_indices] if sel_indices else X_scaled  # selected eval array
    y_pred = model.predict(X_eval)  # model predictions on full dataset

    acc = accuracy_score(y_array, y_pred)  # compute accuracy
    prec = precision_score(y_array, y_pred, average="weighted", zero_division=0)  # precision
    rec = recall_score(y_array, y_pred, average="weighted", zero_division=0)  # recall
    f1 = f1_score(y_array, y_pred, average="weighted", zero_division=0)  # f1 score

    # compute FPR/FNR similarly to compute_rfe_metrics
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

    elapsed = time.time() - start_time
    return float(acc), float(prec), float(rec), float(f1), float(fpr), float(fnr), float(elapsed)


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


def build_run_results(final_model, csv_path, hyperparameters, cv_method, cv_metrics=None, test_metrics=None, training_time=None, top_features=None, rfe_ranking=None):
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
    :return: List containing the results dict
    """
    
    result = {
        "model": final_model.__class__.__name__,
        "dataset": os.path.relpath(csv_path),
        "hyperparameters": json.dumps(hyperparameters),
        "cv_method": cv_method,
        "top_features": json.dumps(top_features),
        "rfe_ranking": json.dumps(rfe_ranking),
    }

    if cv_metrics:
        result.update({
            "cv_accuracy": round(cv_metrics[0], 4),
            "cv_precision": round(cv_metrics[1], 4),
            "cv_recall": round(cv_metrics[2], 4),
            "cv_f1_score": round(cv_metrics[3], 4),
            "cv_fpr": round(cv_metrics[4], 4),
            "cv_fnr": round(cv_metrics[5], 4),
        })

    if test_metrics:
        result.update({
            "test_accuracy": round(test_metrics[0], 4),
            "test_precision": round(test_metrics[1], 4),
            "test_recall": round(test_metrics[2], 4),
            "test_f1_score": round(test_metrics[3], 4),
            "test_fpr": round(test_metrics[4], 4),
            "test_fnr": round(test_metrics[5], 4),
            "testing_time_s": float(test_metrics[6]),
        })

    if training_time is not None:
        result["training_time_s"] = float(training_time)

    return [result]


def run_rfe_fallback(csv_path, X_numeric, y_array, feature_columns, hyperparameters):
    """
    Handles RFE for datasets with insufficient samples for stratified CV (fallback to single train/test split).

    :param csv_path: Path to the CSV dataset file
    :param X_numeric: Numeric features DataFrame
    :param y_array: Target array
    :param feature_columns: Feature column names
    :param hyperparameters: Hyperparameters dict
    """
    
    print(f"{BackgroundColors.YELLOW}Not enough samples per class for stratified CV; falling back to single train/test split.{Style.RESET_ALL}")
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric.values, y_array, test_size=0.2, random_state=42, stratify=None
    )  # perform a single non-stratified train/test split
    selector, model = run_rfe_selector(X_train, y_train, random_state=42)  # run RFE on the single split
    metrics_tuple = compute_rfe_metrics(selector, X_train, X_test, y_train, y_test, random_state=42)  # compute metrics on split
    top_features, rfe_ranking = extract_top_features(selector, feature_columns)  # extract selected features and rankings
    sorted_rfe_ranking = sorted(rfe_ranking.items(), key=lambda x: x[1])  # sort features by ranking (ascending)

    print_metrics(metrics_tuple) if VERBOSE else None  # optionally print metrics
    print_top_features(top_features, rfe_ranking) if VERBOSE else None  # optionally print top features

    final_model, scaler_full, top_features, loaded_hyperparams = get_final_model(csv_path, X_numeric, y_array, top_features, feature_columns)  # get final model/scaler
    eval_metrics = evaluate_exported_model(final_model, scaler_full, X_numeric, feature_columns, top_features, y_array)  # evaluate on full dataset

    run_results = build_run_results(final_model, csv_path, hyperparameters, "single_train_test_split", test_metrics=eval_metrics, training_time=metrics_tuple[6], top_features=top_features, rfe_ranking=sorted_rfe_ranking)  # build results dict

    save_rfe_results(csv_path, run_results)  # save fallback run results
    print_run_summary(run_results)  # concise terminal summary


def run_rfe_cv(csv_path, X_numeric, y_array, feature_columns, hyperparameters):
    """
    Handles RFE with stratified cross-validation.

    :param csv_path: Path to the CSV dataset file
    :param X_numeric: Numeric features DataFrame
    :param y_array: Target array
    :param feature_columns: Feature column names
    :param hyperparameters: Hyperparameters dict
    """

    verbose_output(f"{BackgroundColors.GREEN}Starting RFE with Stratified K-Fold Cross-Validation...{Style.RESET_ALL}")
    
    stratify_param = y_array if len(np.unique(y_array)) > 1 else None
    X_train_df, X_test_df, y_train_array, y_test_array = train_test_split(
        X_numeric, y_array, test_size=0.2, random_state=42, stratify=stratify_param
    )

    # Fit a scaler on the training portion and transform both train/test.
    scaler_for_run = StandardScaler()
    X_train_scaled = scaler_for_run.fit_transform(X_train_df.values)
    X_test_scaled = scaler_for_run.transform(X_test_df.values)

    # Determine CV splits based on the training portion
    unique_tr, counts_tr = np.unique(y_train_array, return_counts=True)
    min_class_count_tr = counts_tr.min() if counts_tr.size > 0 else 0
    n_splits = min(10, len(y_train_array), min_class_count_tr)  # up to 10 splits but not more than samples or smallest class in train
    n_splits = max(2, int(n_splits))  # ensure at least 2 splits

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)  # create stratified K-Fold iterator

    fold_metrics = []  # list to collect per-fold metric tuples
    fold_rankings = []  # list to collect per-fold ranking arrays
    fold_supports = []  # list to collect per-fold support masks
    total_elapsed = 0.0  # accumulator for elapsed times across folds

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_train_scaled, y_train_array), start=1):
        verbose_output(f"{BackgroundColors.CYAN}Running fold {fold_idx}/{n_splits}{Style.RESET_ALL}")  # optional fold progress

        X_train_fold = X_train_scaled[train_idx]
        X_test_fold = X_train_scaled[test_idx]
        y_train_fold = y_train_array[train_idx]
        y_test_fold = y_train_array[test_idx]

        selector, model = run_rfe_selector(X_train_fold, y_train_fold, random_state=42)  # fit RFE on this fold's training data

        metrics_tuple = compute_rfe_metrics(selector, X_train_fold, X_test_fold, y_train_fold, y_test_fold, random_state=42)  # compute metrics for this fold
        fold_metrics.append(metrics_tuple)  # append per-fold metrics tuple
        fold_rankings.append(selector.ranking_)  # append per-fold ranking array
        fold_supports.append(selector.support_.astype(int))  # append per-fold support mask as integers
        total_elapsed += metrics_tuple[6]  # accumulate elapsed time from this fold

    # Aggregate metrics (mean across folds)
    metrics_arr = np.array(fold_metrics)  # convert list of tuples to numpy array
    mean_metrics = metrics_arr.mean(axis=0)  # compute mean metric values across folds

    # Aggregate rankings: mean rank per feature across folds
    rankings_arr = np.vstack(fold_rankings)  # shape: (n_folds, n_features) stack rankings
    mean_rankings = rankings_arr.mean(axis=0)  # mean ranking per feature
    avg_rfe_ranking = {f: float(r) for f, r in zip(feature_columns, mean_rankings)}  # map feature->avg rank

    # Aggregate supports to decide top features (selected in majority of folds)
    supports_arr = np.vstack(fold_supports)  # shape: (n_folds, n_features) stack support masks
    support_counts = supports_arr.sum(axis=0)  # count how many folds selected each feature
    majority_threshold = (n_splits // 2) + 1  # require strict majority to consider a feature selected
    top_features = [f for f, c in zip(feature_columns, support_counts) if c >= majority_threshold]  # select majority-chosen features

    sorted_rfe_ranking = sorted(avg_rfe_ranking.items(), key=lambda x: x[1])  # sort averaged rankings ascending

    # Get final model, scaler, and parameters
    final_model, scaler_full, top_features, loaded_hyperparams = get_final_model(csv_path, X_train_df, y_train_array, top_features, feature_columns)

    # Evaluate final_model (loaded or newly trained) on the held-out test set
    eval_metrics = evaluate_exported_model(final_model, scaler_full, X_test_df, feature_columns, top_features, y_test_array)
    run_results = build_run_results(final_model, csv_path, hyperparameters, f"StratifiedKFold(n_splits={n_splits})", cv_metrics=mean_metrics, test_metrics=eval_metrics, training_time=total_elapsed, top_features=top_features, rfe_ranking=sorted_rfe_ranking)  # build results dict

    print_metrics(tuple(mean_metrics)) if VERBOSE else None  # optionally print aggregated metrics
    print_top_features(top_features, avg_rfe_ranking) if VERBOSE else None  # optionally print aggregated top features and avg ranks

    save_rfe_results(csv_path, run_results)  # save aggregated run results to CSV
    print_run_summary(run_results)  # concise terminal summary


def run_rfe(csv_path):
    """
    Runs Recursive Feature Elimination on the provided dataset, prints the single
    set of top features selected, computes and prints performance metrics, and
    saves the structured results.

    :param csv_path: Path to the CSV dataset file
    :return: None
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Starting RFE analysis on dataset: {BackgroundColors.CYAN}{csv_path}{Style.RESET_ALL}"
    )

    hyperparameters = {  # Default hyperparameters for documentation
        "model": "RandomForestClassifier",
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": N_JOBS,
    }

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
        run_rfe_fallback(csv_path, X_numeric, y_array, feature_columns, hyperparameters)  # Run fallback RFE
    else:  # If sufficient samples for stratified CV
        run_rfe_cv(csv_path, X_numeric, y_array, feature_columns, hyperparameters)  # Run RFE with CV


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

    print(
        f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Recursive Feature Elimination (RFE){BackgroundColors.GREEN} program!{Style.RESET_ALL}"
    )  # Output the welcome message
    start_time = datetime.datetime.now()  # Get the start time of the program

    run_rfe(CSV_FILE)  # Run RFE on the specified CSV file

    finish_time = datetime.datetime.now()  # Get the finish time of the program
    print(
        f"{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
    )  # Output the start and finish times
    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
    )  # Output the end of the program message

    atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None  # Register play_sound at exit if enabled


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run RFE pipeline and optionally load existing exported models."
    )  # CLI description
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
    args = parser.parse_args()  # Parse CLI args

    # Override module-level constant based on CLI flag
    SKIP_TRAIN_IF_MODEL_EXISTS = bool(args.skip_train)  # Respect the user's CLI choice
    VERBOSE = bool(args.verbose)  # Respect the user's request to print verbose output
    CSV_FILE = args.csv if args.csv else CSV_FILE  # Use provided CSV or default

    main()  # Call the main function
