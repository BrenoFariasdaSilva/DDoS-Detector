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
   - pandas, numpy, scikit-learn, seaborn, matplotlib, colorama
"""

import atexit  # For playing a sound when the program finishes
import datetime  # For timestamping
import json  # For saving lists and dicts as JSON strings
import matplotlib.pyplot as plt  # For plotting
import numpy as np  # For numerical operations
import os  # For file and directory operations
import pandas as pd  # For data manipulation
import platform  # For getting the operating system name
import psutil  # For hardware information
import re  # For regular expressions
import seaborn as sns  # For advanced plots
import subprocess  # For executing system commands
import sys  # For system-specific parameters and functions
import time  # For measuring elapsed time
from colorama import Style  # For coloring the terminal
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
from sklearn.model_selection import train_test_split  # For splitting the data
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


def populate_hardware_column_and_order(df, column_name="Hardware"):
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
        "model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "fpr",
        "fnr",
        "elapsed_time_s",
        column_name,
        "top_features",
        "rfe_ranking",
    ]  # Define column order
    return df_results.reindex(columns=columns_order)  # Reorder columns


def save_rfe_results(csv_path, run_results):
    """
    Saves results from RFE run to a structured CSV file.

    :param csv_path: Original CSV file path
    : param run_results: List of dicts containing results from the current run
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Saving RFE Run Results to CSV...{Style.RESET_ALL}"
    )  # Output the verbose message

    output_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis/"  # Define output directory
    os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

    try:  # Try saving CSV
        df_run = pd.DataFrame(run_results)  # Create DataFrame
        df_run = populate_hardware_column_and_order(df_run, column_name="Hardware")
        run_csv_path = f"{output_dir}RFE_Run_Results.csv"  # CSV path
        df_run.to_csv(run_csv_path, index=False, encoding="utf-8")  # Write run results CSV
        print(
            f"{BackgroundColors.GREEN}Run results saved to {BackgroundColors.CYAN}{run_csv_path}{Style.RESET_ALL}"
        )  # Notify CSV saved
    except Exception as e:  # If saving CSV fails
        print(f"{BackgroundColors.RED}Failed to save run results to CSV: {e}{Style.RESET_ALL}")  # Print error


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
    )  # Output the verbose message

    df = load_dataset(csv_path)  # Load the dataset

    if df is None:  # If dataset loading failed
        return  # Return None (no need for empty dict)

    cleaned_df = preprocess_dataframe(df)  # Preprocess the DataFrame

    X = cleaned_df.iloc[:, :-1]  # Features DataFrame
    y = cleaned_df.iloc[:, -1]  # Target Series

    if X is None or y is None:  # If loading failed
        return  # Exit the function

    X_train, X_test, y_train, y_test, feature_columns = scale_and_split(X, y)  # Scale and split the data

    random_state = 42  # Fixed random state for reproducibility

    selector, model = run_rfe_selector(X_train, y_train, random_state=random_state)  # Run RFE to select top features
    metrics_tuple = compute_rfe_metrics(
        selector, X_train, X_test, y_train, y_test, random_state=random_state
    )  # Compute performance metrics (returns a tuple)
    top_features, rfe_ranking = extract_top_features(
        selector, feature_columns
    )  # Extract top features and their rankings

    sorted_rfe_ranking = sorted(
        rfe_ranking.items(), key=lambda x: x[1]
    )  # Sort features by ranking (ascending, lower is better), as list of [feature, ranking]

    run_results = [
        {  # Store results for this run
            "model": model.__class__.__name__,  # Model name
            "accuracy": round(metrics_tuple[0], 4),  # Accuracy
            "precision": round(metrics_tuple[1], 4),  # Precision
            "recall": round(metrics_tuple[2], 4),  # Recall
            "f1_score": round(metrics_tuple[3], 4),  # F1-Score
            "fpr": round(metrics_tuple[4], 4),  # False Positive Rate
            "fnr": round(metrics_tuple[5], 4),  # False Negative Rate
            "elapsed_time_s": round(metrics_tuple[6], 2),  # Elapsed time in seconds (2 decimals for time)
            "top_features": json.dumps(top_features),  # List of top features as JSON
            "rfe_ranking": json.dumps(sorted_rfe_ranking),  # Sorted RFE rankings list as JSON
        }
    ]

    print_metrics(metrics_tuple) if VERBOSE else None  # Print metrics to terminal
    print_top_features(top_features, rfe_ranking) if VERBOSE else None  # Print top features to terminal

    save_rfe_results(csv_path, run_results)  # Save structured results


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

    csv_file = "./Datasets/CICDDoS2019/01-12/DrDoS_DNS.csv"  # Path to the CSV file
    run_rfe(csv_file)  # Run RFE on the specified CSV file

    finish_time = datetime.datetime.now()  # Get the finish time of the program
    print(
        f"{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
    )  # Output the start and finish times
    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
    )  # Output the end of the program message

    atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None  # Register play_sound at exit if enabled


if __name__ == "__main__":
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """

    main()  # Call the main function
