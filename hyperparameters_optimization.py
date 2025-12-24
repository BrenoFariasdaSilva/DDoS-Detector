"""
================================================================================
Classifiers Hyperparameter Optimization
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-12-08
Description :
   This script performs hyperparameter optimization for multiple machine learning
   classifiers using GridSearchCV on DDoS detection datasets. It uses only the
   features selected by the Genetic Algorithm for optimal performance. The script
   evaluates Random Forest, SVM, XGBoost, Logistic Regression, KNN, Nearest Centroid,
   Gradient Boosting, LightGBM, and MLP Neural Network classifiers.

   Key features include:
      - Automatic loading of Genetic Algorithm selected features
      - Data preprocessing with NaN/infinite value removal and zero-variance filtering
      - Comprehensive hyperparameter search grids for each classifier
      - Cross-validation with stratified K-fold (cv=5) for robust evaluation
      - F1-score optimization (weighted average for multi-class problems)
      - Results saved to CSV with best parameters and cross-validation scores
      - Progress tracking with tqdm progress bars
      - Sound notification upon completion

Usage:
   - Configure `DATASETS` or edit `main()` to point to dataset directories.
   - Run: `python hyperparameters_optimization.py` (or integrate from other code).

Outputs:
   - Classifiers_Hyperparameters/<dataset>_Hyperparameter_Optimization_Results.csv
     containing best parameters, best CV F1 and timing for each model tested.

TODOs:
   - Add `argparse` to control dataset selection, CV folds, and search strategy
   - Add randomized/Bayesian search alternatives for large parameter grids
   - Improve resumability for long-running searches and better exception traces

Dependencies:
   - Python >= 3.8
   - pandas, numpy, scikit-learn, xgboost, lightgbm, tqdm, colorama
   - psutil (optional, used for hardware reporting)

Assumptions & Notes:
   - Input CSV: last column is the target, numeric features only are used
   - Genetic Algorithm results must be present under `Feature_Analysis/`
   - Outputs are written next to each processed dataset directory
"""

import argparse  # For command-line arguments
import atexit  # For playing a sound when the program finishes
import concurrent.futures  # For parallel execution with progress updates
import datetime  # For getting the current date and time
import json  # For handling JSON strings
import lightgbm as lgb  # For LightGBM model
import numpy as np  # For numerical operations
import os  # For running a command in the terminal
import pandas as pd  # For data manipulation
import platform  # For getting the operating system name
import psutil  # RAM and CPU core info
import subprocess  # WMIC call
import sys  # For system-specific parameters and functions
import time  # For measuring execution time
import warnings  # For suppressing warnings
from collections import OrderedDict  # For deterministic results column ordering when saving
from colorama import Style  # For coloring the terminal
from itertools import product  # For generating parameter combinations
from Logger import Logger  # For logging output to both terminal and file
from pathlib import Path  # For handling file paths
from sklearn.base import clone  # Import necessary modules for cloning
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # For ensemble models
from sklearn.linear_model import LogisticRegression  # For logistic regression model
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)  # For custom scoring metrics
from sklearn.model_selection import train_test_split  # For hyperparameter search and data splitting
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid  # For k-nearest neighbors model
from sklearn.neural_network import MLPClassifier  # For neural network model
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For label encoding and feature scaling
from sklearn.svm import SVC  # For Support Vector Machine model

try:  # Attempt to import ThunderSVM
    from thundersvm import SVC as ThunderSVC  # For ThunderSVM classifier

    THUNDERSVM_AVAILABLE = True  # Flag indicating ThunderSVM is available
except Exception as _th_err:  # Import failed
    ThunderSVC = None  # ThunderSVM not available
    THUNDERSVM_AVAILABLE = False  # Set flag to False
    print(f"Warning: ThunderSVM import failed ({type(_th_err).__name__}: {_th_err}). Falling back to sklearn.SVC.")  # Print warning message
from tqdm import tqdm  # For progress bars
from xgboost import XGBClassifier  # For XGBoost classifier

# Warnings:
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)  # Ignore pandas dtype warnings


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
VERBOSE = False  # Default verbose setting (can be overridden via CLI args)
N_JOBS = -2  # Number of parallel jobs (-1 uses all cores, -2 leaves one core free, or set specific number like 4)
RESULTS_FILENAME = "Hyperparameter_Optimization_Results.csv"  # Filename for saving results
CACHE_DIR = "Cache/Hyperparameter_Optimization"  # Directory for saving progress cache
ENABLE_PROGRESS_CACHE = True  # Set to False to disable progress caching and recompute all combinations
MATCH_FILENAMES_TO_PROCESS = [""]  # List of specific filenames to search for a match (set to None to process all files)
IGNORE_FILES = [RESULTS_FILENAME]  # List of filenames to ignore when searching for datasets
IGNORE_DIRS = [
    "Cache",
    "Dataset_Description",
    "Data_Separability",
    "Feature_Analysis",
]  # List of directory names to ignore when searching for datasets

# Enabled Models Configuration:
ENABLED_MODELS = [
    "Random Forest",
    "SVM",
    "XGBoost",
    "Logistic Regression",
    "KNN",
    "Nearest Centroid",
    "Gradient Boosting",
    "LightGBM",
    "MLP (Neural Net)",
]  # List of model names to run (comment out models you want to skip)

DATASETS = {  # Dictionary containing dataset paths and feature files
    "CICDDoS2019-Dataset": [  # List of paths to the CICDDoS2019 dataset
        "./Datasets/CICDDoS2019/01-12/",
        "./Datasets/CICDDoS2019/03-11/",
    ],
}

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

    if VERBOSE and true_string != "":  # If the VERBOSE constant is set to True and the true_string is set
        print(true_string)  # Output the true statement string
    elif false_string != "":  # If the false_string is set
        print(false_string)  # Output the false statement string


def parse_args(default_verbose=False):
    """
    Parse command-line arguments to set the VERBOSE constant.

    :param default_verbose: default boolean used when no flag is provided
    :return: resulting VERBOSE boolean
    """
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization runner")  # Create argument parser
    parser.add_argument(  # Add verbose flag
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose output (overrides VERBOSE)",
    )
    parser.add_argument(  # Add no-verbose flag
        "--no-verbose",
        dest="verbose",
        action="store_false",
        help="Disable verbose output (overrides VERBOSE)",
    )
    
    global VERBOSE  # Access the global VERBOSE constant
    parser.set_defaults(verbose=default_verbose)  # Set default value for verbose argument
    args = parser.parse_args()  # Parse command-line arguments
    VERBOSE = bool(args.verbose)  # Set the VERBOSE constant based on parsed argument
    return VERBOSE  # Return the resulting VERBOSE boolean


def iterate_dataset_directories():
    """
    Iterates over all dataset directories defined in DATASETS, skipping invalid and ignored directories.

    :param: None
    :return: Generator yielding (dataset_name, dirpath)
    """

    for dataset_name, paths in DATASETS.items():  # Iterate over datasets
        for dirpath in paths:  # Iterate configured paths
            if not os.path.isdir(dirpath):  # If path is not a directory
                verbose_output(
                    f"{BackgroundColors.YELLOW}Skipping non-directory path: {BackgroundColors.CYAN}{dirpath}{Style.RESET_ALL}"
                )  # Verbose notice
                continue  # Skip invalid path
            if os.path.basename(os.path.normpath(dirpath)) in IGNORE_DIRS:  # If path is in ignore list
                verbose_output(
                    f"{BackgroundColors.YELLOW}Ignoring directory per IGNORE_DIRS: {BackgroundColors.CYAN}{dirpath}{Style.RESET_ALL}"
                )  # Verbose notice
                continue  # Skip ignored directory
            yield dataset_name, dirpath  # Yield valid directory


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

    verbose_output(
        f"{BackgroundColors.GREEN}Extracting Genetic Algorithm selected features...{Style.RESET_ALL}"
    )  # Output the verbose message

    file_dir = os.path.dirname(file_path)  # Determine the directory of the input file
    ga_results_path = os.path.join(
        file_dir, "Feature_Analysis", "Genetic_Algorithm_Results.csv"
    )  # Construct the path to the consolidated GA results file

    verbose_output(
        f"{BackgroundColors.GREEN}Extracting GA features for file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}"
    )  # Output the verbose message

    if not verify_filepath_exists(ga_results_path):  # If the GA results file does not exist
        print(
            f"{BackgroundColors.YELLOW}GA results file not found: {BackgroundColors.CYAN}{ga_results_path}{Style.RESET_ALL}"
        )  # Print warning message
        return None  # Return None if the file doesn't exist

    try:  # Try to load and parse the GA results
        df = pd.read_csv(ga_results_path)  # Load the CSV file into a DataFrame
        best_row = df[df["run_index"] == "best"]  # Filter for the row with run_index == "best"
        if best_row.empty:  # If no "best" row is found
            print(
                f"{BackgroundColors.YELLOW}No 'best' run_index found in {BackgroundColors.CYAN}{ga_results_path}{Style.RESET_ALL}"
            )  # Print warning message
            return None  # Return None if the "best" row doesn't exist
        best_features_str = best_row.iloc[0]["best_features"]  # Get the best_features column value (JSON string)
        best_features = json.loads(best_features_str)  # Parse the JSON string into a Python list
        verbose_output(
            f"{BackgroundColors.GREEN}Loaded {BackgroundColors.CYAN}{len(best_features)}{BackgroundColors.GREEN} GA features{Style.RESET_ALL}"
        )  # Output verbose message about loaded features
        return best_features  # Return the list of best features
    except IndexError:  # If there's an issue accessing the row
        print(
            f"{BackgroundColors.RED}Error: Could not access 'best' row in {BackgroundColors.CYAN}{ga_results_path}{Style.RESET_ALL}"
        )  # Print error message
        return None  # Return None if there was an error
    except Exception as e:  # Catch any other exceptions
        print(f"{BackgroundColors.RED}Error loading GA features: {e}{Style.RESET_ALL}")  # Print error message
        return None  # Return None if there was an error


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
        print(f"{BackgroundColors.RED}CSV file not found: {csv_path}{Style.RESET_ALL}")  # Print error message
        return None  # Return None

    df = pd.read_csv(csv_path, low_memory=True)  # Load the dataset

    df.columns = df.columns.str.strip()  # Clean column names by stripping leading/trailing whitespace

    if df.shape[1] < 2:  # If there are less than 2 columns
        print(f"{BackgroundColors.RED}CSV must have at least 1 feature and 1 target.{Style.RESET_ALL}")  # Print error message
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

    le = LabelEncoder()  # Initialize a LabelEncoder
    y_encoded = pd.Series(
        le.fit_transform(y), index=y.index
    )  # Encode the target variable (essential for stratification)

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


def load_and_prepare_dataset(csv_path):
    """
    Loads, preprocesses, and prepares a dataset for model training and evaluation.

    :param csv_path: Path to the CSV dataset file
    :return: Tuple (X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names)
    """

    df = load_dataset(csv_path)  # Load dataset
    if df is None:  # If load failed
        print(f"{BackgroundColors.YELLOW}Failed to load dataset {csv_path}. Skipping file.{Style.RESET_ALL}")  # Print warning message
        return None  # Exit early

    df_clean = preprocess_dataframe(df)  # Preprocess DataFrame
    if df_clean is None or df_clean.empty:  # If preprocessing failed
        print(f"{BackgroundColors.YELLOW}Dataset preprocessing failed for {csv_path}. Skipping file.{Style.RESET_ALL}")  # Print warning message
        return None  # Exit early

    X = df_clean.iloc[:, :-1]  # Features
    y = df_clean.iloc[:, -1]  # Target

    print(
        f"{BackgroundColors.GREEN}Dataset loaded with {BackgroundColors.CYAN}{X.shape[0]}{BackgroundColors.GREEN} samples and {BackgroundColors.CYAN}{X.shape[1]}{BackgroundColors.GREEN} features{Style.RESET_ALL}"
    )  # Output dataset shape

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = scale_and_split(X, y)  # Split and scale

    feature_names = list(X.select_dtypes(include=np.number).columns)  # Numeric feature names

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names  # Return all components X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names


def get_feature_subset(X_scaled, features, feature_names):
    """
    Returns a subset of features from the scaled feature set based on the provided feature names.

    :param X_scaled: Scaled features (numpy array).
    :param features: List of feature names to select.
    :param feature_names: List of all feature names corresponding to columns in X_scaled.
    :return: Numpy array containing only the selected features, or an empty array if features is None/empty.
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Selecting subset of features based on GA selection...{Style.RESET_ALL}"
    )  # Output the verbose message

    if features:  # Only proceed if the list of selected features is NOT empty/None
        indices = [feature_names.index(f) for f in features if f in feature_names]  # Get indices of selected features
        return X_scaled[:, indices]  # Return the subset of features
    else:  # If no features are selected (or features is None)
        return np.empty((X_scaled.shape[0], 0))  # Return an empty array with correct number of rows


def _is_valid_combination(model_name_local, params_local):
    """
    Check if the given hyperparameter combination is valid for the specified model.
    
    :param model_name_local: Name of the model
    :param params_local: Dictionary of hyperparameters
    :return: True if the combination is valid, False otherwise
    """
    
    if model_name_local != "Logistic Regression":  # Only validate for Logistic Regression
        return True  # All combinations valid for other models

    solver = params_local.get("solver")  # Get solver parameter
    penalty = params_local.get("penalty")  # Get penalty parameter
    l1_ratio = params_local.get("l1_ratio", 0.0)  # Get l1_ratio parameter (default to 0.0)

    # lbfgs doesn't support l1 or elasticnet penalties
    if solver == "lbfgs" and penalty in ("l1", "elasticnet"):
        return False
    
    # elasticnet penalty requires saga solver
    if penalty == "elasticnet" and solver != "saga":
        return False
    
    # l1 penalty only works with saga and liblinear solvers
    if penalty == "l1" and solver not in ("saga", "liblinear"):
        return False
    
    # l1_ratio is only used with elasticnet penalty - filter out if used with other penalties
    if l1_ratio not in (None, 0.0) and penalty != "elasticnet":
        return False
    
    # When penalty is None, l1_ratio should not be specified
    if penalty is None and l1_ratio not in (None, 0.0):
        return False

    return True  # Valid combination


def get_cache_path(csv_path, model_name):
    """
    Generate cache file path for a specific dataset and model.

    :param csv_path: Path to the CSV dataset file
    :param model_name: Name of the model being optimized
    :return: Absolute path to the cache file
    """
    
    verbose_output(
        f"{BackgroundColors.GREEN}Generating cache path for model: {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}"
    )  # Output the verbose message

    csv_basename = os.path.splitext(os.path.basename(csv_path))[0]  # Get CSV filename without extension
    parent_dir = os.path.basename(os.path.dirname(csv_path))  # Get parent directory name
    safe_model_name = model_name.replace(" ", "_").replace("/", "_")  # Make model name filesystem-safe

    cache_subdir = os.path.join(CACHE_DIR, parent_dir, csv_basename)  # Cache subdirectory
    cache_file = os.path.join(cache_subdir, f"{safe_model_name}.json")  # Cache file path

    return cache_file  # Return the cache file path


def load_cached_results(csv_path, model_name):
    """
    Load cached hyperparameter optimization results for a specific dataset and model.

    :param csv_path: Path to the CSV dataset file
    :param model_name: Name of the model being optimized
    :return: Dictionary mapping parameter combinations (as JSON strings) to results, or empty dict if no cache
    """

    if not ENABLE_PROGRESS_CACHE:  # If caching is disabled
        return {}  # Caching disabled

    cache_file = get_cache_path(csv_path, model_name)  # Get cache file path

    if not os.path.exists(cache_file):  # If cache file does not exist
        verbose_output(
            f"{BackgroundColors.YELLOW}No cache found for {model_name} on {os.path.basename(csv_path)}{Style.RESET_ALL}"
        )
        return {}  # No cache file exists

    try:  # Try to load cached results
        with open(cache_file, "r") as f:  # Open cache file
            cached_data = json.load(f)  # Load JSON data
        verbose_output(
            f"{BackgroundColors.GREEN}Loaded {BackgroundColors.CYAN}{len(cached_data)}{BackgroundColors.GREEN} cached results for {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}"
        )
        return cached_data  # Return cached results
    except Exception as e:  # Catch any errors during loading
        print(
            f"{BackgroundColors.YELLOW}Warning: Failed to load cache for {model_name}: {e}{Style.RESET_ALL}"
        )
        return {}  # Return empty dict on error


def save_combination_result(csv_path, model_name, params, score, elapsed, cached_results, hardware_specs=None):
    """
    Save a single combination result to the cache file immediately.

    :param csv_path: Path to the CSV dataset file
    :param model_name: Name of the model being optimized
    :param params: Dictionary of hyperparameters for this combination
    :param score: F1 score achieved (can be a scalar or dict with 'f1_score' key)
    :param elapsed: Execution time in seconds
    :param cached_results: Dictionary of all cached results (will be updated and saved)
    :param hardware_specs: Dictionary of hardware specifications (optional)
    :return: None
    """

    if not ENABLE_PROGRESS_CACHE:  # If caching is disabled
        return  # Caching disabled

    cache_file = get_cache_path(csv_path, model_name)  # Get cache file path

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)  # Ensure cache directory exists

    params_key = json.dumps(params, sort_keys=True)  # Serialize parameters as JSON string

    if isinstance(score, dict):  # If score is a dictionary
        f1_score_value = score.get("f1_score", None)  # Extract F1 score
    else:  # If score is a scalar
        f1_score_value = score  # Use score directly

    cached_results[params_key] = {  # Update cached results
        "score": round(float(f1_score_value), 4) if f1_score_value is not None else None,  # F1 score formatted to 4 decimals
        "elapsed": round(float(elapsed), 2),  # Execution time formatted to 2 decimals
        "timestamp": datetime.datetime.now().isoformat(),  # Timestamp
    }

    if hardware_specs is not None:  # If hardware specifications are provided
        cached_results[params_key]["hardware"] = hardware_specs  # Add hardware specifications

    # Save to file
    try:  # Try to save cached results
        with open(cache_file, "w") as f:  # Open cache file for writing
            json.dump(cached_results, f, indent=2)  # Write JSON data
    except Exception as e:  # Catch any errors during saving
        verbose_output(
            f"{BackgroundColors.YELLOW}Warning: Failed to save cache: {e}{Style.RESET_ALL}"
        )


def detect_gpu_info():
    """
    Detect GPU brand/model using `nvidia-smi` (best-effort).

    :return: String with GPU brand/model (e.g., 'NVIDIA GeForce RTX 2080 Ti') or None if not detected
    """

    try:  # Try to detect GPU info via nvidia-smi
        res = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=False
        )  # Run nvidia-smi to get GPU list
        if res.returncode == 0 and res.stdout.strip():  # If command succeeded and output is non-empty
            first = res.stdout.strip().splitlines()[0]  # Get the first line of output
            if ":" in first:  # If the line contains a colon
                info = first.split(":", 1)[1].strip()  # Extract GPU info after the colon
                return info.split("(")[0].strip()  # Clean up info (remove parentheses)
    except Exception:  # Any error means no GPU info
        return None  # GPU info is not available

    return None  # GPU info is not available


def get_thundersvm_estimator():
    """
    Return a ThunderSVM estimator if available. Prioritize GPU when available
    (detected via `nvidia-smi`), otherwise configure ThunderSVM to use multiple
    CPU threads. Fall back to sklearn's SVC if ThunderSVM is not installed.
    """

    if not THUNDERSVM_AVAILABLE:  # If ThunderSVM is not available
        gpu_info = detect_gpu_info()  # Best-effort GPU brand/model detection

        if gpu_info:  # If GPU info was detected
            print(
                f"{BackgroundColors.YELLOW}ThunderSVM not available; falling back to sklearn.SVC. GPU detected: {BackgroundColors.CYAN}{gpu_info}{Style.RESET_ALL}"
            )
        else:  # If no GPU info was detected
            print(f"{BackgroundColors.YELLOW}ThunderSVM not available; falling back to sklearn.SVC.{Style.RESET_ALL}")

        return SVC(random_state=42, probability=True)  # Return sklearn's SVC as fallback

    gpu_available = False  # Assume no GPU by default
    try:  # Try to run nvidia-smi
        res = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=False
        )  # Run nvidia-smi to check for GPUs
        if res.returncode == 0 and res.stdout.strip():  # If command succeeded and output is non-empty
            gpu_available = True  # GPU is available
    except Exception:  # Any error means no GPU
        gpu_available = False  # GPU is not available

    if gpu_available:  # If a GPU is available
        try:  # Attempt to create ThunderSVM with GPU
            clf = ThunderSVC(random_state=42, probability=True, gpu_id=0)  # Try to use GPU with gpu_id=0
            verbose_output(
                f"{BackgroundColors.GREEN}Using ThunderSVM on GPU (gpu_id=0).{Style.RESET_ALL}"
            )  # Verbose message
            return clf  # Return the GPU-enabled classifier
        except TypeError:  # Constructor may not accept gpu_id; fall back to default constructor
            try:  # Attempt default ThunderSVM constructor
                clf = ThunderSVC(random_state=42, probability=True)  # Default constructor (may auto-detect GPU)
                verbose_output(
                    f"{BackgroundColors.GREEN}Using ThunderSVM (GPU preferred) with default constructor.{Style.RESET_ALL}"
                )  # Verbose message
                return clf  # Return the classifier
            except Exception:  # Any error means fall back to CPU
                pass  # Fall through to CPU handling

    cpu_threads = max(1, (os.cpu_count() or 2) - 1)  # Use all but one CPU core if no GPU is available
    for param in ("n_jobs", "nthread", "nthreads", "nproc", "threads"):  # Try common ThunderSVM CPU thread parameters
        try:  # Attempt to create ThunderSVM with CPU threads
            clf = ThunderSVC(random_state=42, probability=True, **{param: cpu_threads})  # Set CPU threads
            verbose_output(
                f"{BackgroundColors.GREEN}Using ThunderSVM on CPU with {cpu_threads} threads ({param}).{Style.RESET_ALL}"
            )  # Verbose message
            return clf  # Return the CPU-enabled classifier
        except Exception:  # Any error means try next parameter
            continue  # Try next parameter

    verbose_output(
        f"{BackgroundColors.YELLOW}Using ThunderSVM default CPU instantiation.{Style.RESET_ALL}"
    )  # Verbose message

    return ThunderSVC(random_state=42, probability=True)  # Default CPU ThunderSVM


def get_models_and_param_grids():
    """
    Returns a dictionary of models with their corresponding hyperparameter grids for GridSearchCV.

    :param: None
    :return: Dictionary with model names as keys and tuples (model_instance, param_grid) as values
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Initializing models and parameter grids for hyperparameter optimization...{Style.RESET_ALL}"
    )  # Output the verbose message

    all_models = {  # Dictionary of all available models and their parameter grids
        "Random Forest": (
            RandomForestClassifier(random_state=42, n_jobs=N_JOBS),  # Random Forest classifier
            {
                "n_estimators": [50, 100, 200],  # Number of trees in the forest
                "max_depth": [None, 10, 20, 30],  # Maximum depth of the tree
                "min_samples_split": [2, 5, 10],  # Minimum number of samples required to split an internal node
                "min_samples_leaf": [1, 2, 4],  # Minimum number of samples required to be at a leaf node
                "max_features": [
                    "sqrt",
                    "log2",
                    None,
                ],  # Number of features to consider when looking for the best split
            },
        ),
        "SVM": (
            get_thundersvm_estimator(),  # ThunderSVM (GPU preferred) or fallback to sklearn SVC
            {
                "C": [0.1, 1, 10, 100],  # Regularization parameter
                "kernel": ["linear", "rbf", "poly"],  # Kernel type to be used in the algorithm
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],  # Kernel coefficient
            },
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="mlogloss", random_state=42, n_jobs=N_JOBS),  # XGBoost classifier
            {
                "n_estimators": [50, 100, 200],  # Number of trees in the forest
                "max_depth": [3, 5, 7, 10],  # Maximum depth of the tree
                "learning_rate": [0.01, 0.1, 0.3],  # Step size shrinkage
                "subsample": [0.6, 0.8, 1.0],  # Subsample ratio of the training instances
                "colsample_bytree": [0.6, 0.8, 1.0],  # Subsample ratio of columns when constructing each tree
            },
        ),
        "Logistic Regression": (
            LogisticRegression(max_iter=5000, random_state=42, n_jobs=N_JOBS),  # Logistic Regression classifier
            {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
                "penalty": ["l1", "l2", "elasticnet", None],  # Norm used in the penalization
                "solver": ["lbfgs", "saga"],  # Algorithm to use in the optimization problem (liblinear removed)
                "l1_ratio": [0.0, 0.5, 1.0],  # The Elastic-Net mixing parameter
            },
        ),
        "KNN": (
            KNeighborsClassifier(n_jobs=N_JOBS),  # K-Nearest Neighbors classifier
            {
                "n_neighbors": [3, 5, 7, 9, 11],  # Number of neighbors to use
                "weights": ["uniform", "distance"],  # Weight function used in prediction
                "metric": ["euclidean", "manhattan", "minkowski"],  # Distance metric
                "p": [1, 2],  # Power parameter for the Minkowski metric
            },
        ),
        "Nearest Centroid": (
            NearestCentroid(),  # Nearest Centroid classifier
            {
                "metric": ["euclidean", "manhattan"],  # Distance metric
                "shrink_threshold": [None, 0.1, 0.5, 1.0, 2.0],  # Threshold for shrinking centroids
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),  # Gradient Boosting classifier
            {
                "n_estimators": [50, 100, 200],  # Number of boosting stages to be run
                "learning_rate": [0.01, 0.1, 0.3],  # Learning rate shrinks the contribution of each tree
                "max_depth": [3, 5, 7],  # Maximum depth of the individual regression estimators
                "min_samples_split": [2, 5, 10],  # Minimum number of samples required to split an internal node
                "min_samples_leaf": [1, 2, 4],  # Minimum number of samples required to be at a leaf node
                "subsample": [0.6, 0.8, 1.0],  # Subsample ratio of the training instances
            },
        ),
        "LightGBM": (
            lgb.LGBMClassifier(
                force_row_wise=True, random_state=42, verbosity=-1, n_jobs=N_JOBS
            ),  # LightGBM classifier
            {
                "n_estimators": [50, 100, 200],  # Number of boosting stages to be run
                "max_depth": [3, 5, 7, 10, -1],  # Maximum depth of the tree (-1 means no limit)
                "learning_rate": [0.01, 0.1, 0.3],  # Step size shrinkage
                "num_leaves": [15, 31, 63],  # Number of leaves in one tree
                "min_child_samples": [10, 20, 30],  # Minimum number of data needed in a child (leaf)
                "subsample": [0.6, 0.8, 1.0],  # Subsample ratio of the training instances
            },
        ),
        "MLP (Neural Net)": (
            MLPClassifier(max_iter=500, random_state=42),  # Multi-layer Perceptron classifier
            {
                "hidden_layer_sizes": [(50,), (100,), (100, 50), (100, 100)],  # Number of neurons in the hidden layers
                "activation": ["relu", "tanh", "logistic"],  # Activation function for the hidden layer
                "solver": ["adam", "sgd"],  # The solver for weight optimization
                "alpha": [0.0001, 0.001, 0.01],  # L2 penalty (regularization term) parameter
                "learning_rate": ["constant", "adaptive"],  # Learning rate schedule for weight updates
            },
        ),
    }


    enabled_models = {model_name: model_config for model_name, model_config in all_models.items() if model_name in ENABLED_MODELS}  # Filter enabled models

    if not enabled_models:  # If no models are enabled
        print(
            f"{BackgroundColors.RED}Error: No models enabled in ENABLED_MODELS configuration. Please enable at least one model.{Style.RESET_ALL}"
        )  # Print error message
        return {}  # Return empty dict

    disabled_models = [name for name in all_models.keys() if name not in ENABLED_MODELS]  # List of disabled models
    if disabled_models:  # If there are disabled models
        print(
            f"{BackgroundColors.YELLOW}Disabled models: {BackgroundColors.CYAN}{', '.join(disabled_models)}{Style.RESET_ALL}"
        )  # Print list of disabled models

    print(
        f"{BackgroundColors.GREEN}Enabled models ({len(enabled_models)}): {BackgroundColors.CYAN}{', '.join(enabled_models.keys())}{Style.RESET_ALL}"
    )  # Print list of enabled models

    return enabled_models  # Return enabled models and their parameter grids


def compute_total_param_combinations(models):
    """
    Computes the total number of hyperparameter combinations for all models
    and returns both the total count and a per-model combination dictionary.

    :param models: List of (model_name, (model_instance, param_grid))
    :return: Tuple (total_combinations_all_models, model_combinations_counts)
    """

    total_combinations_all_models = 0  # Initialize total combinations counter
    model_combinations_counts = {}  # Store per-model combination counts

    for model_name, (model, param_grid) in models:  # Iterate models
        if param_grid:  # If there is a param grid
            values = [v if isinstance(v, (list, tuple)) else [v] for v in param_grid.values()]  # Ensure lists
            count = len(list(product(*values)))  # Count combinations
        else:  # No hyperparameters
            count = 1  # Single combination

        model_combinations_counts[model_name] = count  # Store per-model count
        total_combinations_all_models += count  # Add to total

    return total_combinations_all_models, model_combinations_counts  # Return total and per-model counts


def compute_confusion_rates(cm):
    """
    Compute average FPR, FNR, TPR, TNR from a confusion matrix.

    :param cm: confusion matrix (numpy array)
    :return: tuple(mean_fpr, mean_fnr, mean_tpr, mean_tnr)
    """
    
    n_classes = cm.shape[0]  # Number Of classes from confusion matrix shape
    fpr_list, fnr_list, tpr_list, tnr_list = [], [], [], []  # Initialize lists to collect rates

    for i in range(n_classes):  # Iterate each class index
        tp = cm[i, i]  # True positives for class i
        fn = cm[i, :].sum() - tp  # False negatives for class i
        fp = cm[:, i].sum() - tp  # False positives for class i
        tn = cm.sum() - tp - fn - fp  # True negatives for class i

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False positive rate computation
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False negative rate computation
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True positive rate (recall) computation
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # True negative rate computation

        fpr_list.append(fpr)  # Collect FPR
        fnr_list.append(fnr)  # Collect FNR
        tpr_list.append(tpr)  # Collect TPR
        tnr_list.append(tnr)  # Collect TNR

    return float(np.mean(fpr_list)), float(np.mean(fnr_list)), float(np.mean(tpr_list)), float(
        np.mean(tnr_list)
    )  # Return mean rates as floats


def compute_metrics_from_predictions(model, y_true, y_pred, X=None):
    """
    Compute a standard metrics dict from true/predicted labels and model.

    :param model: trained estimator (may provide predict_proba)
    :param y_true: true labels (array-like)
    :param y_pred: predicted labels (array-like)
    :param X: feature matrix (optional, used for predict_proba)
    :return: dict with metrics
    """
    
    metrics = {}  # Container for computed metrics

    f1 = f1_score(y_true, y_pred, average="weighted")  # Weighted F1 score
    accuracy = accuracy_score(y_true, y_pred)  # Accuracy score
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)  # Precision
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)  # Recall
    mcc = matthews_corrcoef(y_true, y_pred)  # Matthews correlation coefficient
    kappa = cohen_kappa_score(y_true, y_pred)  # Cohen's kappa

    metrics.update({
        "f1_score": float(f1),  # Store F1 as float
        "accuracy": float(accuracy),  # Store accuracy
        "precision": float(precision),  # Store precision
        "recall": float(recall),  # Store recall
        "matthews_corrcoef": float(mcc),  # Store MCC
        "cohen_kappa": float(kappa),  # Store Cohen's kappa
    })

    try:  # Compute confusion matrix-based metrics
        cm = confusion_matrix(y_true, y_pred)  # Compute confusion matrix
        fpr, fnr, tpr, tnr = compute_confusion_rates(cm)  # Compute averaged rates
        metrics["false_positive_rate"] = fpr  # Add FPR to metrics
        metrics["false_negative_rate"] = fnr  # Add FNR
        metrics["true_positive_rate"] = tpr  # Add TPR
        metrics["true_negative_rate"] = tnr  # Add TNR
    except Exception:  # If confusion matrix computation fails
        pass  # Ignore if confusion-based metrics fail

    try:  # Attempt to compute ROC-AUC if possible
        if X is not None and hasattr(model, "predict_proba"):  # Check if model supports predict_proba
            y_pred_proba = model.predict_proba(X)  # Get predicted probabilities
            roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="weighted")  # Compute ROC-AUC
            metrics["roc_auc_score"] = float(roc_auc)  # Store ROC-AUC
    except Exception:  # If ROC-AUC computation fails
        pass  # Ignore ROC-AUC computation errors

    return metrics  # Return assembled metrics dict


def evaluate_single_combination(model, keys, combination, X_train, y_train):
    """
    Helper function to evaluate a single parameter combination.
    Designed to be called in parallel via ThreadPoolExecutor with memory safety.

    :param model: Clone of the model instance
    :param keys: Parameter names
    :param combination: Parameter values for this combination
    :param X_train: Training features
    :param y_train: Training labels
    :return: Tuple (current_params, metrics_dict, elapsed)
    """

    current_params = dict(zip(keys, combination))  # Build dict of current params
    start_time = time.time()  # Start timing
    metrics = None  # Initialize metrics as None
    try:  # Try to train and evaluate using extracted helpers for clarity
        model.set_params(**current_params)  # Apply hyperparameters to model
        model.fit(X_train, y_train)  # Fit model on training data
        y_pred = model.predict(X_train)  # Predict labels on training data

        metrics = compute_metrics_from_predictions(model, y_train, y_pred, X_train)  # Compute metrics from predictions

    except MemoryError:  # Catch memory errors specifically
        print(
            f"{BackgroundColors.RED}MemoryError with params {current_params}. Consider reducing dataset size or n_jobs.{Style.RESET_ALL}"
        )  # Print error message
        metrics = None  # Set metrics to None on memory error
    except KeyboardInterrupt:  # User interrupted execution
        raise  # Re-raise KeyboardInterrupt
    except Exception as e:  # Catch any other exceptions
        verbose_output(
            f"{BackgroundColors.YELLOW}Error evaluating params {current_params}: {type(e).__name__}: {e}{Style.RESET_ALL}"
        )  # Output verbose error message
        metrics = None  # Set metrics to None on error

    elapsed = time.time() - start_time  # Compute elapsed time

    return current_params, metrics, elapsed  # Return params, metrics and elapsed


def update_optimization_progress_bar(
    progress_bar,
    csv_path,
    model_name,
    param_grid=None,
    combo_current=None,
    combo_total=None,
    current=None,
    total_models=None,
    total_combinations=None,
    overall=None,
):
    """
    Updates a tqdm progress bar during hyperparameter optimization.

    Shows dataset reference, model name, progress index, and an optional compact
    summary of hyperparameters.

    :param progress_bar: tqdm progress bar instance
    :param csv_path: Path to dataset CSV
    :param model_name: Name of the model being optimized
    :param param_grid: Optional hyperparameter dictionary or summary
    :param combo_current: Current hyperparameter combination index (1-based)
    :param combo_total: Total hyperparameter combinations for the current model
    :param current: Current model index (1-based)
    :param total_models: Total number of models being optimized
    :param total_combinations: Total number of hyperparameter combinations across all models
    :return: None
    """

    if progress_bar is None:
        return  # Nothing to update when progress bar is None

    try:  # Protect against unexpected errors
        csv_name = os.path.basename(csv_path)  # Base filename from path
        parent_dir = os.path.basename(os.path.dirname(csv_path))  # Parent directory name
        if parent_dir and parent_dir.lower() != csv_name.lower():  # If parent differs from filename
            dataset_ref = f"{BackgroundColors.CYAN}{parent_dir}/{csv_name}{BackgroundColors.GREEN}"  # Show parent/filename
        else:  # Otherwise
            dataset_ref = f"{BackgroundColors.CYAN}{csv_name}{BackgroundColors.GREEN}"  # Show only filename

        idx_str = (
            f"{BackgroundColors.GREEN}[{BackgroundColors.CYAN}{current}/{total_models}{BackgroundColors.GREEN}]" if current is not None and total_models is not None else ""
        )  # Formatted model index string

        if combo_current is not None and combo_total is not None:  # If combination indices supplied
            combo_str = f" {BackgroundColors.GREEN}{{{BackgroundColors.CYAN}{combo_current}/{combo_total}{BackgroundColors.GREEN}}}"  # Combination progress
        else:  # No combination info
            combo_str = ""  # No combination info

        desc = f"{BackgroundColors.GREEN}Dataset: {dataset_ref}{BackgroundColors.GREEN} - {idx_str} {BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN}:{combo_str}"  # Base description for progress

        def _short(value, limit=30):  # Helper to create short string representations
            return str(value) if len(str(value)) <= limit else str(value)[: limit - 3] + "..."  # Truncate helper

        if isinstance(param_grid, dict):  # If param_grid is dict, prepare compact display
            parts = []  # List to accumulate parameter summaries
            for i, (k, v) in enumerate(param_grid.items()):  # Iterate items
                if i >= 4:
                    break  # Limit display to first 4 params
                try:
                    vals = list(v) if hasattr(v, "__iter__") and not isinstance(v, (str, bytes, dict)) else [v]
                    shown = ",".join([_short(x, 12) for x in vals[:4]])  # Show up to 4 values
                    if len(vals) > 4:
                        shown += f",+{len(vals)-4}"  # Indicate extras
                    parts.append(f"{BackgroundColors.GREEN}{_short(k,18)}{BackgroundColors.GREEN}:[{BackgroundColors.CYAN}{shown}{BackgroundColors.GREEN}]")
                except Exception:
                    parts.append(f"{BackgroundColors.GREEN}{_short(k,18)}{BackgroundColors.GREEN}:[{BackgroundColors.CYAN}{_short(v,12)}{BackgroundColors.GREEN}]")
            remaining = max(0, len(param_grid) - 4)  # Number of remaining params not displayed
            param_display = ", ".join(parts)  # Join parts for display
            if remaining > 0:
                param_display += f", {BackgroundColors.CYAN}+{remaining} more{BackgroundColors.GREEN}"  # Append remaining count
        else:
            param_display = _short(param_grid, 60)  # Otherwise, short representation

        desc = f"{desc} {BackgroundColors.GREEN}({param_display}){Style.RESET_ALL}"  # Append parameter display

        progress_bar.set_description(desc)  # Set progress bar description
        progress_bar.n = overall if overall is not None else (current or getattr(progress_bar, "n", 0))  # Set current count
        progress_bar.total = total_combinations  # Set total combinations
        progress_bar.refresh()  # Refresh the display

    except Exception:
        pass  # Silently ignore any errors during update


def compute_safe_n_jobs(X_train, y_train):
    """
    Compute a safe `n_jobs` value based on available memory and dataset size.
    
    :param X_train: Training feature matrix
    :param y_train: Training labels
    :return: Tuple (safe_n_jobs, available_memory_gb, data_size_gb
    """
    
    available_memory_gb = psutil.virtual_memory().available / (1024 ** 3)  # Available RAM in GB

    try:  # Try to compute dataset size
        data_size_gb = (X_train.nbytes + y_train.nbytes) / (1024 ** 3)  # Dataset size in GB
    except Exception:  # If computation fails
        data_size_gb = 0.0  # Default to 0 GB

    estimated_memory_per_worker = max(0.1, data_size_gb * 0.25)  # Estimate memory per worker (25% of dataset size, min 0.1 GB)

    physical_cores = psutil.cpu_count(logical=False) or os.cpu_count() or 1  # Number of physical CPU cores
    if N_JOBS == -1:  # All cores
        desired_workers = physical_cores  # Use all physical cores
    elif N_JOBS == -2:  # All but one core
        desired_workers = max(1, physical_cores - 1)  # Use all but one core
    elif N_JOBS > 0:  # Specific number of jobs
        desired_workers = N_JOBS  # Use specified number of jobs
    else:  # Invalid N_JOBS
        desired_workers = max(1, physical_cores - 1)  # Default to all but one core

    mem_based_cap = int(max(1, available_memory_gb / max(0.5, estimated_memory_per_worker)))  # Memory-based worker cap
    safe_n_jobs = int(min(desired_workers, physical_cores, mem_based_cap))  # Final safe n_jobs
    safe_n_jobs = max(1, min(safe_n_jobs, 64))  # Ensure at least 1 and at most 64

    return safe_n_jobs, available_memory_gb, data_size_gb  # Return computed values


def run_parallel_evaluation(
    model_name,
    model,
    keys,
    combinations_to_test,
    cached_results,
    cached_count,
    X_train,
    y_train,
    csv_path,
    progress_bar,
    total_combinations,
    model_index,
    total_combinations_all_models,
    total_models,
    hardware_specs,
    global_counter,
    best_score,
    best_params,
    best_elapsed,
    all_results,
    safe_n_jobs,
):
    """
    Run the ThreadPoolExecutor evaluation and return updated results.

    :param model_name: Name of the model being optimized
    :param model: Model instance to optimize
    :param keys: List of hyperparameter names
    :param combinations_to_test: List of hyperparameter combinations to evaluate
    :param cached_results: Dictionary of cached results
    :param cached_count: Number of cached combinations
    :param X_train: Training features
    :param y_train: Training labels
    :param csv_path: Path to CSV for progress description
    :param progress_bar: tqdm progress bar instance
    :param total_combinations: Total hyperparameter combinations for the current model
    :param model_index: Current model index (1-based)
    :param total_combinations_all_models: Total hyperparameter combinations across all models
    :param total_models: Total number of models being optimized
    :param hardware_specs: Dictionary of hardware specifications (optional)
    :param global_counter: Overall combination index across all models
    :param best_score: Current best F1 score
    :param best_params: Current best hyperparameters
    :param best_elapsed: Execution time of the best combination
    :param all_results: List of all results collected so far
    :param safe_n_jobs: Number of parallel workers to use
    :return: Tuple (best_params, best_score, best_elapsed, all_results, global_counter)
    """
    
    verbose_output(f"{BackgroundColors.GREEN}Starting parallel evaluation with {BackgroundColors.CYAN}{safe_n_jobs}{BackgroundColors.GREEN} workers...{Style.RESET_ALL}")  # Log start

    if progress_bar is not None and cached_count > 0:  # Advance progress for cached combos
        for _ in range(cached_count):  # For each cached combo
            try:  # Try to update progress bar
                progress_bar.update(1)  # Increment progress bar
            except Exception:  # Catch any errors
                pass  # Ignore progress update errors

    if len(combinations_to_test) == 0:  # Nothing to evaluate
        verbose_output(f"{BackgroundColors.GREEN}All combinations already cached for {BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN}. Skipping computation.{Style.RESET_ALL}")
        return best_params, best_score, best_elapsed, all_results, global_counter  # Return early

    with concurrent.futures.ThreadPoolExecutor(max_workers=safe_n_jobs) as executor:  # Run parallel evaluation
        future_to_params = {
            executor.submit(evaluate_single_combination, clone(model), keys, combo, X_train, y_train): combo
            for combo in combinations_to_test
        }  # Map futures to combos

        local_counter = cached_count  # Start local counter after cached items

        for future in concurrent.futures.as_completed(future_to_params):  # Process completed futures
            try:  # Try to get result
                current_params, metrics, elapsed = future.result()  # Get results
            except Exception as worker_err:  # Catch exceptions from worker
                combo = future_to_params.get(future)  # Find combo that failed
                current_params = dict(zip(keys, combo)) if combo is not None else {}  # Reconstruct params
                metrics = None  # No metrics on failure
                elapsed = 0.0  # No elapsed time
                print(f"{BackgroundColors.YELLOW}Warning: Combination {current_params} failed: {worker_err}{Style.RESET_ALL}")  # Warn

            if csv_path:  # Persist result if csv path provided
                save_combination_result(
                    csv_path, model_name, current_params, metrics, elapsed, cached_results, hardware_specs
                )  # Save result

                global_counter += 1  # Increment global progress
                local_counter += 1  # Increment local model progress

                if progress_bar is not None:
                    update_optimization_progress_bar(
                        progress_bar,
                        csv_path,
                        model_name,
                        param_grid=current_params,
                        combo_current=local_counter,
                        combo_total=total_combinations,
                        current=model_index,
                        total_combinations=total_combinations_all_models,
                        total_models=total_models,
                        overall=global_counter,
                    )  # Update progress description
                    try:
                        progress_bar.update(1)  # Advance progress
                    except Exception:
                        pass  # Ignore update errors

                result_entry = OrderedDict([("params", json.dumps(current_params)), ("execution_time", round(float(elapsed), 2))])  # Build result entry with formatted time
                if metrics is not None:
                    # Format all metrics to 4 decimal places
                    formatted_metrics = {k: round(float(v), 4) if v is not None else None for k, v in metrics.items()}
                    result_entry.update(formatted_metrics)  # Include formatted metrics
                all_results.append(result_entry)  # Append to list

                if metrics is not None and "f1_score" in metrics:  # Update best if improved
                    f1 = metrics["f1_score"]  # Extract F1 score
                    current_best_elapsed = (
                        next((r["execution_time"] for r in all_results if r.get("f1_score") == best_score), float("inf"))
                        if best_score != -float("inf")
                        else float("inf")
                    )  # Find current best elapsed
                    if (f1 > best_score) or (f1 == best_score and elapsed < current_best_elapsed):
                        best_score = f1
                        best_params = current_params
                        best_elapsed = elapsed
                        verbose_output(f"{BackgroundColors.GREEN}New best F1 score: {BackgroundColors.CYAN}{best_score:.4f}{BackgroundColors.GREEN} with params: {BackgroundColors.CYAN}{best_params}{Style.RESET_ALL}")
                        # Log new best

    return best_params, best_score, best_elapsed, all_results, global_counter  # Return updated results


def manual_grid_search(
    model_name,
    model,
    param_grid,
    X_train,
    y_train,
    progress_bar=None,
    csv_path=None,
    global_counter_start=0,
    total_combinations_all_models=None,
    model_index=None,
    total_models=None,
    hardware_specs=None,
):
    """
    Performs manual grid search hyperparameter optimization with integrated progress bar.
    Uses parallel processing via joblib to evaluate parameter combinations simultaneously,
    significantly speeding up optimization for all classifiers.

    Updates the progress bar description and counter for each parameter combination
    tested, showing both the current combination index of this model and the
    overall combination count across all models.

    :param model_name: Name of the model for logging
    :param model: Model instance to optimize
    :param param_grid: Dictionary of hyperparameters to search
    :param X_train: Training features
    :param y_train: Training labels
    :param progress_bar: Optional tqdm progress bar
    :param csv_path: Path to CSV for progress description
    :param global_counter_start: Starting counter of overall combination index
    :param total_combinations_all_models: Total number of parameter combinations across all models
    :param total_models: Total number of models being optimized
    :param hardware_specs: Dictionary of hardware specifications (optional)
    :return: Tuple (best_params, best_score, all_results, global_counter_end)
    """

    verbose_output(f"{BackgroundColors.GREEN}Manually optimizing {BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN} using parallel processing...{Style.RESET_ALL}")  # Log manual optimization start

    if not param_grid:
        return None, None, None, global_counter_start  # No hyperparameters to optimize

    keys = list(param_grid.keys())  # Parameter names
    values = [v if isinstance(v, (list, tuple)) else [v] for v in param_grid.values()]  # Ensure each value is iterable
    param_combinations = list(product(*values))  # Cartesian product of hyperparameter values
    total_combinations = len(param_combinations)  # Total combinations count

    cached_results = load_cached_results(csv_path, model_name) if csv_path else {}  # Load cached cache results
    cached_count = len(cached_results)  # Count cached entries

    combinations_to_test = []  # Accumulate combinations that are not cached
    for combo in param_combinations:  # Iterate all combos
        params_dict = dict(zip(keys, combo))  # Build params dict
        params_key = json.dumps(params_dict, sort_keys=True)  # Canonical JSON key
        if params_key not in cached_results and _is_valid_combination(model_name, params_dict):  # Skip if cached or invalid
            combinations_to_test.append(combo)  # Schedule for testing

    if cached_count > 0:  # Report cached count
        print(f"{BackgroundColors.GREEN}Found {BackgroundColors.CYAN}{cached_count}{BackgroundColors.GREEN} cached results for {BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN}. Testing {BackgroundColors.CYAN}{len(combinations_to_test)}{BackgroundColors.GREEN} remaining combinations.{Style.RESET_ALL}")

    best_score = -float("inf")  # Initialize best score sentinel
    best_params = None  # Placeholder for best params
    best_elapsed = 0.0  # Elapsed time for best params
    all_results = []  # List to collect results
    global_counter = global_counter_start  # Start global counter

    for params_key, cached_data in cached_results.items():  # Process cached results
        try:
            params_dict = json.loads(params_key)  # Parse params key
            f1 = cached_data.get("f1_score")  # Cached F1 if present
            elapsed = cached_data.get("elapsed", 0.0)  # Cached elapsed time

            result_entry = OrderedDict([("params", params_key), ("execution_time", round(float(elapsed), 2))])  # Base result entry with formatted time

            for key in [
                "f1_score",
                "accuracy",
                "precision",
                "recall",
                "false_positive_rate",
                "false_negative_rate",
                "true_positive_rate",
                "true_negative_rate",
                "matthews_corrcoef",
                "roc_auc_score",
                "cohen_kappa",
            ]:  # Metrics to include
                if key in cached_data:  # If metric present
                    result_entry[key] = round(float(cached_data[key]), 4) if cached_data[key] is not None else None  # Include cached metric formatted to 4 decimals
            
            if "f1_score" not in result_entry and f1 is not None:  # If missing F1 but have score
                result_entry["f1_score"] = round(float(f1), 4)  # Add F1 score formatted

            all_results.append(result_entry)  # Append cached result

            if f1 is not None and f1 > best_score:  # Update best from cache
                best_score = f1  # Update best score
                best_params = params_dict  # Update best params
                best_elapsed = elapsed  # Update best elapsed time

            global_counter += 1  # Increment global counter
        except Exception:  # Catch JSON parsing errors
            pass  # Ignore malformed cache entries

    safe_n_jobs, available_memory_gb, data_size_gb = compute_safe_n_jobs(X_train, y_train)  # Compute workers and sizes

    verbose_output(
        f"{BackgroundColors.GREEN}Using {BackgroundColors.CYAN}{safe_n_jobs}{BackgroundColors.GREEN} parallel workers (Available RAM: {BackgroundColors.CYAN}{available_memory_gb:.1f}GB{BackgroundColors.GREEN}, Dataset: {BackgroundColors.CYAN}{data_size_gb:.2f}GB{BackgroundColors.GREEN}){Style.RESET_ALL}"
    )  # Log resources

    best_params, best_score, best_elapsed, all_results, global_counter = run_parallel_evaluation(
        model_name,
        model,
        keys,
        combinations_to_test,
        cached_results,
        cached_count,
        X_train,
        y_train,
        csv_path,
        progress_bar,
        total_combinations,
        model_index,
        total_combinations_all_models,
        total_models,
        hardware_specs,
        global_counter,
        best_score,
        best_params,
        best_elapsed,
        all_results,
        safe_n_jobs,
    )  # Perform parallel evaluation

    verbose_output(f"{BackgroundColors.GREEN}Completed optimization for {BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN}. Best score: {BackgroundColors.CYAN}{best_score:.4f}{Style.RESET_ALL}")  # Log completion

    return best_params, best_score, best_elapsed, all_results, global_counter  # Return final results


def build_result_entry_from_best(csv_path, model_name, best_params, best_score, best_elapsed, all_results, X_train_ga):
    """
    Build the ordered result dict for the best found parameters for a model.
    
    :param csv_path: Path of the CSV file currently being processed
    :param model_name: Name of the model
    :param best_params: Best hyperparameters found
    :param best_score: Best F1 score achieved
    :param best_elapsed: Elapsed time for best parameters
    :param all_results: List of all results collected during optimization
    :param X_train_ga: Training feature matrix generated by the Genetic Algorithm
    :return: OrderedDict with result entry
    """
    
    elapsed_time = float(best_elapsed or 0.0)  # Elapsed time for best params

    best_result = None  # Placeholder for best result entry
    for result in all_results:  # Find the best result entry
        if result.get("params") == json.dumps(best_params):  # Match params
            best_result = result  # Store best result
            break  # Exit loop

    result_dict = OrderedDict(  # Build result dictionary
        [
            ("base_csv", os.path.basename(csv_path)),
            ("model", model_name),
            ("best_params", json.dumps(best_params)),
            ("best_cv_f1_score", round(float(best_score), 4) if best_score is not None else None),
            ("n_features", X_train_ga.shape[1]),
            ("feature_selection_method", "Genetic Algorithm"),
            ("dataset", os.path.basename(csv_path)),
            ("elapsed_time_s", round(float(elapsed_time), 2)),
        ]
    )

    if best_result:  # If best result found, include additional metrics
        for metric_key in [  # Metrics to include
            "accuracy",
            "precision",
            "recall",
            "false_positive_rate",
            "false_negative_rate",
            "true_positive_rate",
            "true_negative_rate",
            "matthews_corrcoef",
            "roc_auc_score",
            "cohen_kappa",
        ]:
            if metric_key in best_result:  # If metric present
                result_dict[metric_key] = round(float(best_result[metric_key]), 4) if best_result[metric_key] is not None else None  # Add to result dict formatted to 4 decimals

    return result_dict  # Return assembled result dict


def run_model_optimizations(models, csv_path, X_train_ga, y_train, dir_results_list):
    """
    Runs optimization for all configured ML models using a progress bar and manual grid search.

    :param models: List of (model_name, (model_instance, param_grid))
    :param csv_path: Path of the CSV file currently being processed
    :param X_train_ga: Training feature matrix generated by the Genetic Algorithm
    :param y_train: Training labels
    :param dir_results_list: Accumulator list for storing optimization results
    :return: None
    """

    total_combinations_all_models, model_combinations_counts = compute_total_param_combinations(
        models
    )  # Compute total combinations

    verbose_output(
        f"{BackgroundColors.GREEN}Starting hyperparameter optimizations for {BackgroundColors.CYAN}{len(models)}{BackgroundColors.GREEN} models with a total of {BackgroundColors.CYAN}{total_combinations_all_models}{BackgroundColors.GREEN} parameter combinations...{Style.RESET_ALL}"
    )  # Output verbose message

    hardware_specs = get_hardware_specifications()  # Fetch system specs
    verbose_output(
        f"{BackgroundColors.GREEN}Hardware: {BackgroundColors.CYAN}{hardware_specs['cpu_model']}{BackgroundColors.GREEN} | Cores: {BackgroundColors.CYAN}{hardware_specs['cores']}{BackgroundColors.GREEN} | RAM: {BackgroundColors.CYAN}{hardware_specs['ram_gb']}GB{Style.RESET_ALL}"
    )  # Output hardware info

    global_counter = 0  # Initialize global combination counter

    with tqdm(
        total=total_combinations_all_models,
        desc=f"{BackgroundColors.GREEN}Optimizing Models{Style.RESET_ALL}",
        unit="comb",
    ) as pbar:  # Progress bar
        for model_index, (model_name, (model, param_grid)) in enumerate(models, start=1):  # Iterate models with index
            try:  # Wrap each model optimization in try-except to prevent one failure from stopping all
                best_params, best_score, best_elapsed, all_results, global_counter = manual_grid_search(
                    model_name,
                    model,
                    param_grid,
                    X_train_ga,
                    y_train,
                    progress_bar=pbar,
                    csv_path=csv_path,
                    global_counter_start=global_counter,
                    total_combinations_all_models=total_combinations_all_models,
                    model_index=model_index,
                    total_models=len(models),
                    hardware_specs=hardware_specs,
                )  # Manual grid search instead of GridSearchCV
            except Exception as model_err:  # Catch any errors during model optimization
                print(
                    f"{BackgroundColors.RED}Error optimizing {model_name}: {model_err}{Style.RESET_ALL}"
                )  # Print error message
                continue  # Skip to next model

            if best_params is not None:  # If valid best parameters were found
                result_dict = build_result_entry_from_best(
                    csv_path, model_name, best_params, best_score, best_elapsed, all_results, X_train_ga
                )  # Build result entry
                dir_results_list.append(result_dict)  # Append to results list
            else:  # If no valid parameters were found
                print(
                    f"{BackgroundColors.YELLOW}Warning: No valid results for {model_name}{Style.RESET_ALL}"
                )  # Print warning message

            pbar.refresh()  # Refresh progress bar

        print()  # Line spacing after all models complete


def process_single_csv_file(csv_path, dir_results_list):
    """
    Processes a single CSV file: loads GA-selected features, prepares the dataset,
    applies GA-based column filtering, and runs model hyperparameter optimization.

    :param csv_path: Path to dataset CSV file
    :param dir_results_list: List to store optimization results for the directory
    :return: None
    """

    print(
        f"{BackgroundColors.GREEN}\nProcessing file: {BackgroundColors.CYAN}{csv_path}{Style.RESET_ALL}"
    )  # Output the file being processed

    print(
        f"{BackgroundColors.GREEN}Loading Genetic Algorithm selected features...{Style.RESET_ALL}"
    )  # Output loading message
    ga_selected_features = extract_genetic_algorithm_features(csv_path)  # Extract GA features
    if ga_selected_features is None or len(ga_selected_features) == 0:  # If no GA features found
        print(f"{BackgroundColors.YELLOW}No GA features found for {csv_path}. Skipping file.{Style.RESET_ALL}")  # Print warning
        return  # Exit early

    dataset_bundle = load_and_prepare_dataset(csv_path)  # Load, preprocess, split, scale
    if dataset_bundle is None:  # If loading/preprocessing failed
        return  # Exit early

    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = dataset_bundle  # Unpack dataset bundle

    print(f"{BackgroundColors.GREEN}Applying GA feature selection...{Style.RESET_ALL}")  # Output message
    X_train_ga = get_feature_subset(X_train_scaled, ga_selected_features, feature_names)  # GA train subset
    X_test_ga = get_feature_subset(X_test_scaled, ga_selected_features, feature_names)  # GA test subset

    print(
        f"{BackgroundColors.GREEN}Training set shape after GA feature selection: {BackgroundColors.CYAN}{X_train_ga.shape}{Style.RESET_ALL}"
    )  # Output shape
    print(
        f"{BackgroundColors.GREEN}Testing set shape after GA feature selection: {BackgroundColors.CYAN}{X_test_ga.shape}{Style.RESET_ALL}"
    )  # Output shape

    if X_train_ga.shape[1] == 0:  # If GA selects no features
        print(f"{BackgroundColors.YELLOW}No features selected by GA for {csv_path}. Skipping file.{Style.RESET_ALL}")  # Print warning
        return  # Exit early

    models_and_grids = get_models_and_param_grids()  # Get model grids

    start_idx = len(dir_results_list)  # Track result insertion index
    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Starting hyperparameter optimization for {BackgroundColors.CYAN}{len(models_and_grids)}{BackgroundColors.GREEN} models on {BackgroundColors.CYAN}{os.path.basename(csv_path)}{BackgroundColors.GREEN}...{Style.RESET_ALL}\n"
    )  # Output header

    models = list(models_and_grids.items())  # Convert dict to list

    run_model_optimizations(models, csv_path, X_train_ga, y_train, dir_results_list)  # Run optimizations

    added_slice = dir_results_list[start_idx:]  # Extract slice
    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Optimization Summary for {BackgroundColors.CYAN}{os.path.basename(csv_path)}{BackgroundColors.GREEN}:{Style.RESET_ALL}"
    )  # Summary header
    print(
        f"{BackgroundColors.GREEN}Total models optimized: {BackgroundColors.CYAN}{len(added_slice)}{Style.RESET_ALL}"
    )  # Output count

    if added_slice:  # If results exist
        best_model = max(added_slice, key=lambda x: x["best_cv_f1_score"])  # Best model
        print(
            f"{BackgroundColors.GREEN}Best model: {BackgroundColors.CYAN}{best_model['model']}{Style.RESET_ALL}"
        )  # Output model name
        print(
            f"{BackgroundColors.GREEN}Best CV F1 Score: {BackgroundColors.CYAN}{best_model['best_cv_f1_score']:.4f}{Style.RESET_ALL}"
        )  # Output best score


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


def save_optimization_results(csv_path, results_list):
    """
    Saves hyperparameter optimization results to a CSV file.

    :param csv_path: Path to the original dataset CSV file.
    :param results_list: List of dictionaries containing optimization results.
    :return: None
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Saving optimization results...{Style.RESET_ALL}"
    )  # Output the verbose message

    if not results_list:  # If the results list is empty
        print(f"{BackgroundColors.YELLOW}No results to save.{Style.RESET_ALL}")  # Print warning
        return  # Exit the function

    output_dir = f"{os.path.dirname(csv_path)}/Classifiers_Hyperparameters/"  # Directory to save outputs
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]  # Get the base name of the dataset
    output_path = os.path.join(output_dir, f"{dataset_name}_{RESULTS_FILENAME}")  # Full path to save the results CSV

    try:  # Try to save the results
        df_results = pd.DataFrame(results_list)  # Convert results list to DataFrame
        hardware_specs = get_hardware_specifications()  # Get system specs
        df_results["Hardware"] = (
            hardware_specs["cpu_model"]
            + " | Cores: "
            + str(hardware_specs["cores"])
            + " | RAM: "
            + str(hardware_specs["ram_gb"])
            + " GB | OS: "
            + hardware_specs["os"]
        )  # Add hardware specs column
        df_results.to_csv(output_path, index=False, encoding="utf-8")  # Save to CSV
        print(f"{BackgroundColors.GREEN}Results saved to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}")  # Print success message
    except Exception as e:  # Catch any errors during saving
        print(f"{BackgroundColors.RED}Error saving results: {e}{Style.RESET_ALL}")  # Print error message


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
            )  # Print error message
    else:  # If the sound file does not exist
        print(
            f"{BackgroundColors.RED}Sound file {BackgroundColors.CYAN}{SOUND_FILE}{BackgroundColors.RED} not found. Make sure the file exists.{Style.RESET_ALL}"
        )  # Print error message


def main():
    """
    Main function.

    :param: None
    :return: None
    """

    parse_args(VERBOSE)  # Parse command-line arguments

    print(
        f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Classifiers Hyperparameters Optimization{BackgroundColors.GREEN} program!{Style.RESET_ALL}",
        end="\n\n",
    )  # Output the welcome message
    start_time = datetime.datetime.now()  # Get the start time of the program

    for dataset_name, dirpath in iterate_dataset_directories():  # Iterate valid dataset directories
        csv_files = get_files_to_process(
            dirpath, file_extension=".csv"
        )  # Discover CSV files in this directory (non-recursive)
        if not csv_files:  # If no CSV files were discovered in this dirpath
            verbose_output(
                f"{BackgroundColors.YELLOW}No CSV files found in: {BackgroundColors.CYAN}{dirpath}{Style.RESET_ALL}"
            )  # Verbose notice
            continue  # Move to the next dirpath

        dir_results_list = []  # Aggregate results for all CSVs in this dirpath

        for csv_path in csv_files:  # Process each CSV file found in the current dirpath
            try:  # Process the current csv_path inside a try/except to continue on errors
                process_single_csv_file(csv_path, dir_results_list)  # Process CSV end-to-end
            except Exception as e:  # Catch any unhandled exceptions during CSV processing
                print(
                    f"{BackgroundColors.RED}Unhandled error processing {csv_path}: {e}{Style.RESET_ALL}"
                )  # Print the exception and continue
                continue  # Continue to the next CSV file

        if dir_results_list:  # If there are results
            rep_csv_path = os.path.join(
                dirpath, os.path.basename(os.path.normpath(dirpath))
            )  # Representative CSV base path
            save_optimization_results(rep_csv_path, dir_results_list)  # Save results

    finish_time = datetime.datetime.now()  # Get the finish time of the program
    print(
        f"\n{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
    )  # Output the start and finish times
    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
    )  # Output the end of the program message

    (
        atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None
    )  # Register the play_sound function to be called when the program finishes


if __name__ == "__main__":
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """

    main()  # Call the main function
