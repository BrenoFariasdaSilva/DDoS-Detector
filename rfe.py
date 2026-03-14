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
import traceback  # For formatting and printing exception tracebacks
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
from typing import Any, Dict, Optional, Union, Tuple, cast  # For type hinting


# Macros:
class BackgroundColors:  # Colors for the terminal
    CYAN = "\033[96m"  # Cyan
    GREEN = "\033[92m"  # Green
    YELLOW = "\033[93m"  # Yellow
    RED = "\033[91m"  # Red
    BOLD = "\033[1m"  # Bold
    UNDERLINE = "\033[4m"  # Underline
    CLEAR_TERMINAL = "\033[H\033[J"  # Clear the terminal


# Global state (initialized in main)
CONFIG: Dict[str, Any] = {}
CLI_ARGS: Dict[str, Any] = {}
TELEGRAM_BOT = None
logger = None

SOUND_COMMANDS = {
    "Darwin": "afplay",
    "Linux": "aplay",
    "Windows": "start",
}

SOUND_FILE = "./.assets/Sounds/NotificationSound.wav"

RUN_FUNCTIONS = {"Play Sound": True}

# Functions Definitions:


def verbose_output(true_string="", false_string=""):
    """
    Output a message if the VERBOSE constant is set to True.

    :param true_string: The string to be outputted if the VERBOSE constant is set to True.
    :param false_string: The string to be outputted if the VERBOSE constant is set to False.
    :return: None.
    """
    
    try:
        if is_verbose():
            if true_string:
                print(true_string)
        else:
            if false_string:
                print(false_string)
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_default_config() -> Dict[str, Any]:
    """
    Return the default configuration for the RFE tool.
    Must match config.yaml.example exactly.

    :return: Dict with default configuration values.
    """
    
    try:
        return {
            "rfe": {
                "execution": {
                    "verbose": False,
                    "skip_train_if_model_exists": False,
                    "dataset_path": None,
                },
                "model": {"estimator": "random_forest", "random_state": 42},
                "selection": {"n_features_to_select": 10, "step": 1},
                "cross_validation": {"n_folds": 10},
                "multiprocessing": {"n_jobs": -1, "cpu_processes": 1},
                "caching": {"enabled": True, "pickle_protocol": 4},
                "export": {
                    "results_dir": "Feature_Analysis/RFE",
                    "results_filename": "RFE_Results.csv",
                    "results_csv_columns": [
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
                },
            }
        }
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def load_config_file(path: Optional[str]) -> Dict[str, Any]:
    """
    Load YAML config file. Returns empty dict if none found.
    
    :param path: Optional path to the config file. If None, will look for config.yaml or config.yaml.example in the script directory.
    :return: Dict with the loaded configuration, or empty dict if no file found
    """
    
    try:
        candidate = None  # Initialize candidate path before resolution
        if path:  # Use the provided path if given
            candidate = Path(path)  # Convert provided path to Path object
        else:  # Auto-discover config file from script directory
            p = Path(__file__).parent  # Locate the directory containing this script
            c = p / "config.yaml"  # Build path to config.yaml
            e = p / "config.yaml.example"  # Build path to config.yaml.example
            if c.exists():  # Prefer config.yaml when it exists
                candidate = c  # Assign config.yaml as candidate
            elif e.exists():  # Fall back to config.yaml.example when present
                candidate = e  # Assign config.yaml.example as candidate

        if candidate is None or not candidate.exists():  # Return empty dict when no config file was found
            return {}  # No config file found

        with open(candidate, "r", encoding="utf-8") as fh:  # Open the located config file for reading
            return yaml.safe_load(fh) or {}  # Parse YAML and return dict (or empty dict if file is empty)
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def parse_cli_args() -> Dict[str, Any]:
    """
    Parse command-line arguments and return them as a dictionary.

    :return: Dict with the parsed command-line arguments.
    """
    
    try:
        parser = argparse.ArgumentParser(description="Run RFE pipeline")  # Create the argument parser
        parser.add_argument("--config", type=str, default=None)  # Config file path
        parser.add_argument("--dataset_path", type=str, default=None)  # Dataset path override
        parser.add_argument("--n_features_to_select", type=int, default=None)  # Number of features to select override
        parser.add_argument("--step", type=int, default=None)  # RFE step override
        parser.add_argument("--estimator", type=str, default=None)  # Estimator name override
        parser.add_argument("--random_state", type=int, default=None)  # Random state override
        parser.add_argument("--n_folds", type=int, default=None)  # Number of CV folds override
        parser.add_argument("--n_jobs", type=int, default=None)  # Number of parallel jobs override
        parser.add_argument("--cpu_processes", type=int, default=None)  # Number of CPU processes override
        parser.add_argument("--caching_enabled", type=lambda s: str(s).lower() in ("1", "true", "yes", "y"), default=None)  # Caching enabled override
        parser.add_argument("--pickle_protocol", type=int, default=None)  # Pickle protocol override
        parser.add_argument("--verbose", action="store_true")  # Verbose flag
        parser.add_argument("--skip_train_if_model_exists", action="store_true")  # Skip training flag
        parser.add_argument("--results_dir", type=str, default=None, help="Override results directory for RFE exports (overrides config.rfe.export.results_dir)")  # Results directory override
        parser.add_argument("--results_filename", type=str, default=None, help="Override results filename for RFE exports (overrides config.rfe.export.results_filename)")  # Results filename override
        args = parser.parse_args()  # Parse all arguments from sys.argv
        return vars(args)  # Return parsed arguments as a plain dict
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two dictionaries, with values from the override taking precedence.
    
    :param base: The base dictionary to merge into
    :param override: The dictionary with values to override in the base
    :return: A new dictionary resulting from the deep merge of base and override
    """
    
    try:
        result = dict(base)  # Start with a shallow copy of the base dictionary
        for k, v in (override or {}).items():  # Iterate over all keys in the override dict
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):  # Recursively merge nested dicts
                result[k] = deep_merge_dicts(result[k], v)  # Recurse into nested dict
            else:  # Non-dict values are replaced by the override value
                result[k] = v  # Assign override value directly
        return result  # Return the merged dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def validate_config_structure(config: Dict[str, Any]) -> None:
    """
    Validates the structure and types of the configuration dictionary. Raises ValueError if any required section or key is missing or has an invalid type.
    
    :param config: The configuration dictionary to validate
    :return: None
    """
    
    try:
        if not isinstance(config, dict):  # Validate top-level type
            raise ValueError("config must be a dict")  # Raise on invalid type
        if "rfe" not in config or not isinstance(config["rfe"], dict):  # Validate rfe section exists
            raise ValueError("Missing required top-level 'rfe' section in config")  # Raise on missing section
        r = config["rfe"]  # Extract the rfe section
        required = ["execution", "model", "selection", "cross_validation", "multiprocessing", "caching", "export"]  # Required sub-sections
        for key in required:  # Iterate required sub-section names
            if key not in r or not isinstance(r[key], dict):  # Validate each required sub-section
                raise ValueError(f"Missing required 'rfe.{key}' section in config or wrong type")  # Raise on missing required section
        cols = r["export"].get("results_csv_columns")  # Read the results CSV columns
        if not isinstance(cols, list) or not cols:  # Validate columns are a non-empty list
            raise ValueError("rfe.export.results_csv_columns must be a non-empty list in config")  # Raise on invalid columns
        sel = r["selection"]  # Extract selection section
        if sel.get("n_features_to_select") is not None:  # Validate n_features_to_select when provided
            if not isinstance(sel["n_features_to_select"], int) or sel["n_features_to_select"] <= 0:  # Validate positive int
                raise ValueError("rfe.selection.n_features_to_select must be an int > 0 or null")  # Raise on invalid value
        if not isinstance(sel.get("step", 1), int) or sel.get("step", 1) <= 0:  # Validate step is positive int
            raise ValueError("rfe.selection.step must be an int > 0")  # Raise on invalid step
        cv = r["cross_validation"].get("n_folds")  # Read n_folds
        if not isinstance(cv, int) or cv <= 1:  # Validate n_folds is int greater than 1
            raise ValueError("rfe.cross_validation.n_folds must be an int > 1")  # Raise on invalid folds
        mp = r["multiprocessing"]  # Extract multiprocessing section
        if not isinstance(mp.get("n_jobs"), int):  # Validate n_jobs type
            raise ValueError("rfe.multiprocessing.n_jobs must be an integer")  # Raise on invalid n_jobs
        if not isinstance(mp.get("cpu_processes"), int) or mp.get("cpu_processes") < 1:  # Validate cpu_processes is positive
            raise ValueError("rfe.multiprocessing.cpu_processes must be an int >= 1")  # Raise on invalid cpu_processes
        cache = r["caching"]  # Extract caching section
        if not isinstance(cache.get("enabled"), bool):  # Validate enabled is bool
            raise ValueError("rfe.caching.enabled must be a boolean")  # Raise on invalid enabled type
        pp = cache.get("pickle_protocol")  # Read pickle_protocol
        if not isinstance(pp, int) or not (0 <= pp <= 5):  # Validate pickle_protocol range
            raise ValueError("rfe.caching.pickle_protocol must be int between 0 and 5")  # Raise on invalid protocol
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_config(cli_args: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Returns the merged configuration dictionary and a dictionary of sources for each key in the merged config.
    
    :param cli_args: Optional dictionary of command-line arguments to override config values
    :return: Tuple of (merged_config, sources_dict) where merged_config is the final configuration dictionary and sources_dict maps config sections to their source ('default', 'file', 'cli')
    """
    
    defaults = get_default_config()
    file_cfg = load_config_file((cli_args or {}).get("config"))

    file_rfe = file_cfg.get("rfe") if isinstance(file_cfg, dict) and "rfe" in file_cfg else file_cfg
    merged = deep_merge_dicts(defaults["rfe"], file_rfe or {})


    cli = cli_args or {}
    exec_cfg = merged.get("execution", {})
    if cli.get("dataset_path") is not None:
        exec_cfg["dataset_path"] = cli.get("dataset_path")
    if cli.get("verbose"):
        exec_cfg["verbose"] = True
    if cli.get("skip_train_if_model_exists"):
        exec_cfg["skip_train_if_model_exists"] = True

    model_cfg = merged.get("model", {})
    if cli.get("estimator") is not None:
        model_cfg["estimator"] = cli.get("estimator")
    if cli.get("random_state") is not None:
        model_cfg["random_state"] = cli.get("random_state")

    sel = merged.get("selection", {})
    if cli.get("n_features_to_select") is not None:
        sel["n_features_to_select"] = cli.get("n_features_to_select")
    if cli.get("step") is not None:
        sel["step"] = cli.get("step")

    cv = merged.get("cross_validation", {})
    if cli.get("n_folds") is not None:
        cv["n_folds"] = cli.get("n_folds")

    mp = merged.get("multiprocessing", {})
    if cli.get("n_jobs") is not None:
        mp["n_jobs"] = cli.get("n_jobs")
    if cli.get("cpu_processes") is not None:
        mp["cpu_processes"] = cli.get("cpu_processes")

    cache = merged.get("caching", {})
    if cli.get("caching_enabled") is not None:
        cache["enabled"] = bool(cli.get("caching_enabled"))
    if cli.get("pickle_protocol") is not None:
        raw_pickle_protocol = cli.get("pickle_protocol")  # Retrieve raw CLI value for 'pickle_protocol' (Any | None)
        if raw_pickle_protocol is None:  # Ensure the retrieved CLI value is not None before conversion
            raise ValueError("pickle_protocol CLI argument is required")  # Raise to preserve strict semantics when missing
        if isinstance(raw_pickle_protocol, int):  # If the raw value is already an int
            cache["pickle_protocol"] = raw_pickle_protocol  # Assign the int value directly into cache
        elif isinstance(raw_pickle_protocol, str) and raw_pickle_protocol.strip() != "":  # If it's a non-empty string
            cache["pickle_protocol"] = int(raw_pickle_protocol)  # Safely convert numeric string to int and assign into cache
        else:  # Any other type is invalid for conversion
            raise ValueError("Invalid pickle_protocol CLI value; expected int or numeric string")  # Raise to avoid unsafe int() call

    if cli.get("results_dir") is not None:
        merged.setdefault("export", {})["results_dir"] = cli.get("results_dir")
    if cli.get("results_filename") is not None:
        merged.setdefault("export", {})["results_filename"] = cli.get("results_filename")

    merged["execution"] = exec_cfg
    merged["model"] = model_cfg
    merged["selection"] = sel
    merged["cross_validation"] = cv
    merged["multiprocessing"] = mp
    merged["caching"] = cache

    final = {"rfe": merged}
    validate_config_structure(final)
    sources = {"rfe": "merged"}
    return final, sources


def get_results_csv_columns(config: Optional[Dict[str, Any]] = None) -> list:
    """
    Get the list of columns to be used in the results CSV export from the configuration.
    
    :param config: Optional configuration dictionary to read the columns from. If None, will use the global CONFIG variable.
    :return: List of column names to be used in the results CSV export
    """
    
    cfg = config or CONFIG
    try:
        cols = cfg["rfe"]["export"]["results_csv_columns"]
    except Exception:
        raise ValueError("rfe.export.results_csv_columns missing from configuration")
    if not isinstance(cols, list) or not cols:
        raise ValueError("rfe.export.results_csv_columns must be a non-empty list")
    return cols


def is_verbose() -> bool:
    """
    Verify if verbose output is enabled based on the CLI arguments or configuration.

    :return: True if verbose output is enabled, False otherwise.
    """
    
    return bool((CLI_ARGS.get("verbose") if isinstance(CLI_ARGS, dict) else False) or (CONFIG.get("rfe", {}).get("execution", {}).get("verbose") if isinstance(CONFIG, dict) else False))


def verify_dot_env_file():
    """
    Verify if the .env file exists in the current directory.

    :return: True if the .env file exists, False otherwise.
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
    Set up the Telegram bot for progress messages.

    :return: None.
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Setting up Telegram bot for messages...{Style.RESET_ALL}"
        )  # Output the verbose message

        verify_dot_env_file()  # Verify if the .env file exists

        global TELEGRAM_BOT  # Declare the module-global TELEGRAM_BOT variable

        try:  # Try to initialize the Telegram bot
            TELEGRAM_BOT = TelegramBot()  # Initialize Telegram bot for progress messages
            telegram_module.TELEGRAM_DEVICE_INFO = f"{telegram_module.get_local_ip()} - {platform.system()}"  # Set device info string with IP and OS
            telegram_module.RUNNING_CODE = os.path.basename(__file__)  # Set currently running script name
        except Exception as e:
            print(f"{BackgroundColors.RED}Failed to initialize Telegram bot: {e}{Style.RESET_ALL}")  # Report initialization failure to terminal
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
            n_jobs_val = CONFIG.get("rfe", {}).get("multiprocessing", {}).get("n_jobs") if isinstance(CONFIG, dict) else -1
            model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=int(n_jobs_val) if n_jobs_val is not None else -1)
        else:
            raise ValueError(f"Unsupported estimator '{estimator_name}'. Supported: 'random_forest'")

        n_features = X_train.shape[1]  # Get the number of features
        if n_select is None:
            n_select = n_features  # Default to all (caller may reduce later)
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


def select_rfe_features(selector, X_train, X_test):
    """
    Apply the RFE support mask to select features from training and testing arrays.

    :param selector: Fitted RFE object containing the support mask.
    :param X_train: Training feature array.
    :param X_test: Testing feature array.
    :return: Tuple of (X_train_selected, X_test_selected) with only RFE-selected columns.
    """

    try:
        support = selector.support_  # Get the boolean mask of selected features
        X_train_selected = X_train[:, support]  # Apply mask to select training features
        X_test_selected = X_test[:, support]  # Apply mask to select testing features

        return X_train_selected, X_test_selected  # Return the selected feature arrays
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def initialize_rfe_estimator(estimator_name, random_state=42):
    """
    Initialize the classifier estimator for RFE metric computation.

    :param estimator_name: Name of the estimator to initialize.
    :param random_state: Random seed for reproducibility.
    :return: Initialized classifier model.
    """

    try:
        estimator_name_l = (estimator_name or "random_forest").lower()  # Normalize estimator name to lowercase

        if "random" in estimator_name_l:  # Verify if random forest estimator was requested
            n_jobs_val = CONFIG.get("rfe", {}).get("multiprocessing", {}).get("n_jobs") if isinstance(CONFIG, dict) else -1  # Retrieve n_jobs from config or default to -1
            model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=int(n_jobs_val) if n_jobs_val is not None else -1)  # Initialize the random forest classifier
        else:  # Unsupported estimator name
            raise ValueError(f"Unsupported estimator '{estimator_name}' for compute_rfe_metrics. Supported: 'random_forest'")  # Raise error for unsupported estimator

        return model  # Return the initialized model
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def train_rfe_estimator(model, X_train_selected, y_train):
    """
    Fit the estimator on the selected training features and measure training time.

    :param model: Classifier to train.
    :param X_train_selected: Training feature array with only RFE-selected columns.
    :param y_train: Training target array.
    :return: Tuple of (model, training_time_s).
    """

    try:
        train_start = time.perf_counter()  # Start perf_counter immediately before model.fit (training window)
        model.fit(X_train_selected, y_train)  # Fit the model on selected features (training)
        train_end = time.perf_counter()  # End perf_counter immediately after model.fit
        training_time_s = round(train_end - train_start, 6)  # Compute training time rounded to 6 decimals

        return model, training_time_s  # Return the trained model and its training time
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def compute_fpr_fnr(y_test, y_pred):
    """
    Compute false positive rate and false negative rate for binary or multi-class classification.

    :param y_test: True target labels.
    :param y_pred: Predicted target labels.
    :return: Tuple of (fpr, fnr) as Python floats.
    """

    try:
        if len(np.unique(y_test)) == 2:  # Verify if binary classification
            cm = confusion_matrix(y_test, y_pred)  # Confusion matrix for observed labels

            if cm.shape == (2, 2):  # Expect 2x2 matrix for binary
                tn, fp, fn, tp = cm.ravel()  # Unpack confusion matrix into tn, fp, fn, tp
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Calculate false positive rate
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # Calculate false negative rate
            else:  # Fallback: compute rates from sums if unexpected shape
                total = cm.sum() if cm.size > 0 else 1  # Total predictions for normalization
                fpr = float(cm.sum() - np.trace(cm)) / float(total) if total > 0 else 0  # Approximate FPR
                fnr = fpr  # Fallback estimate when binary layout is unexpected
        else:  # For multi-class classification
            cm = confusion_matrix(y_test, y_pred)  # Confusion matrix for observed labels
            supports = cm.sum(axis=1)  # Support for each class
            fprs = []  # List to hold per-class FPR
            fnrs = []  # List to hold per-class FNR

            for i in range(cm.shape[0]):  # Iterate over each class
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

        return float(fpr), float(fnr)  # Return FPR and FNR as Python floats
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def predict_and_compute_rfe_scores(model, X_test_selected, y_test):
    """
    Run model prediction on selected test features and compute all classification scores including timing.

    :param model: Trained classifier.
    :param X_test_selected: Test feature array with only RFE-selected columns.
    :param y_test: True target labels.
    :return: Tuple of (acc, prec, rec, f1, fpr, fnr, testing_time_s).
    """

    try:
        test_start = time.perf_counter()  # Start perf_counter immediately before prediction (testing window)
        y_pred = model.predict(X_test_selected)  # Predict on selected test features (testing)
        acc = accuracy_score(y_test, y_pred)  # Calculate accuracy
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate precision
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate recall
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate F1-score

        fpr, fnr = compute_fpr_fnr(y_test, y_pred)  # Compute false positive and false negative rates

        test_end = time.perf_counter()  # End perf_counter immediately after metrics computed
        testing_time_s = round(test_end - test_start, 6)  # Compute testing time rounded to 6 decimals

        return float(acc), float(prec), float(rec), float(f1), float(fpr), float(fnr), float(testing_time_s)  # Return all classification scores
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

        X_train_selected, X_test_selected = select_rfe_features(selector, X_train, X_test)  # Apply RFE support mask to training and testing feature arrays

        model = initialize_rfe_estimator(estimator_name, random_state)  # Initialize the classifier estimator for this RFE run

        model, training_time_s = train_rfe_estimator(model, X_train_selected, y_train)  # Fit the estimator and measure training time

        acc, prec, rec, f1, fpr, fnr, testing_time_s = predict_and_compute_rfe_scores(model, X_test_selected, y_test)  # Predict and compute all classification scores

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


def populate_hardware_column_and_order(df, config: Optional[Dict[str, Any]] = None, column_name="hardware"):
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

        cols = get_results_csv_columns(config)
        columns_order = [(column_name if str(c).lower() == "hardware" else c) for c in cols]
        return df_results.reindex(columns=columns_order)
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
        model_rs = CONFIG.get("rfe", {}).get("model", {}).get("random_state") if isinstance(CONFIG, dict) else 42
        n_jobs_val = CONFIG.get("rfe", {}).get("multiprocessing", {}).get("n_jobs") if isinstance(CONFIG, dict) else -1
        final_model = RandomForestClassifier(n_estimators=100, random_state=int(model_rs) if model_rs is not None else 42, n_jobs=int(n_jobs_val) if n_jobs_val is not None else -1)  # Instantiate final RF
        final_model.fit(X_final, y_array)  # Fit final model on entire dataset using selected features

        cfg_source = CONFIG if isinstance(CONFIG, dict) and CONFIG else get_default_config()
        rfe_cfg_local = cfg_source.get("rfe") if isinstance(cfg_source.get("rfe"), dict) else cfg_source
        export_cfg_local = (rfe_cfg_local or {}).get("export", {})
        results_dir_raw_local = export_cfg_local.get("results_dir") or os.path.join("Feature_Analysis", "RFE")
        if os.path.isabs(results_dir_raw_local):
            resolved_dir_local = os.path.abspath(os.path.expanduser(results_dir_raw_local))
        else:
            dataset_dir_local = os.path.dirname(csv_path) or "."
            resolved_dir_local = os.path.abspath(os.path.expanduser(os.path.join(dataset_dir_local, results_dir_raw_local)))
        models_dir = os.path.join(resolved_dir_local, "Models")
        
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

def row_style_for_zebra(row):
    """
    Top-level function to produce zebra row styles for pandas Styler.

    :param row: pandas Series representing a row
    :return: List[str] of CSS style strings for each cell
    """

    bg = "white" if (row.name % 2) == 0 else "#f2f2f2"  # White for even rows, light gray for odd rows
    return [f"background-color: {bg};" for _ in row.index]  # Return style for every column in the row
    

def apply_zebra_style(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Apply zebra-striping style to a DataFrame using pandas Styler.

    :param df: Input DataFrame to style
    :return: pandas Styler with zebra row background colors applied
    """
    
    try:
        styled = df.style.apply(row_style_for_zebra, axis=1)  # Apply zebra function row-wise using top-level function
        styled = styled.set_table_attributes('style="border-collapse:collapse; width:100%;"')  # Tight table style
        styled = cast(pd.io.formats.style.Styler, cast(Any, styled).set_properties(**{"border": "1px solid #ddd", "padding": "6px"}))  # Cell padding/border (cast to Any to satisfy typing)
        return styled  # Return the styled object
    except Exception as e:
        print(str(e))  # Print error for visibility
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Notify via Telegram
        raise  # Propagate error to caller


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
        dfi.export(cast(Any, styled_df), str(out_p))  # Use dataframe_image to export styled DataFrame to PNG (cast Styler to Any for static typing)
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
        if not os.access(str(parent), os.W_OK):  # Verify write permission on parent
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
        if is_visualizable and len(df) <= 100:  # Only generate image when visualizable and within the safe row limit
            try:  # Guard PNG rendering to keep CSV persistence independent from image export
                png_path = csv_p.with_suffix('.png')  # Construct PNG path by replacing extension
                generate_table_image_from_dataframe(df, png_path)  # Generate PNG from in-memory DataFrame
            except Exception as _png_e:  # Contain PNG rendering failures locally
                print(f"{BackgroundColors.YELLOW}Warning: PNG generation failed for {csv_p.name}: {_png_e}{Style.RESET_ALL}")  # Warn and continue without propagating PNG errors
    except Exception:
        raise  # Propagate exceptions to caller


def resolve_model_display_name(r: Dict[str, Any]) -> str:
    """
    Resolve the display name of the model from a run result dict.

    :param r: Run result dictionary from a single RFE run.
    :return: Friendly display name string for the model.
    """

    try:
        model_obj = r.get("model") or r.get("estimator") or ""  # Extract model object or name from result dict

        try:  # Safely resolve model name string from object
            model_repr = model_obj if isinstance(model_obj, str) else getattr(model_obj, "__class__", type(model_obj)).__name__  # Get class name if not already a string
        except Exception:  # On resolution failure
            model_repr = str(model_obj)  # Fallback to string representation of model object

        if isinstance(model_repr, str) and "random" in model_repr.lower() and "forest" in model_repr.lower():  # Verify if model is Random Forest
            return "Random Forest"  # Return friendly display name for Random Forest

        return model_repr  # Return raw representation for non-random-forest models
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_cv_method_string(r: Dict[str, Any]) -> Optional[str]:
    """
    Build the cross-validation method description string from a run result dict.

    :param r: Run result dictionary from a single RFE run.
    :return: CV method description string or None if not available.
    """

    try:
        cv_method = r.get("cv_method") or r.get("cv")  # Extract CV method string from result dict

        if not cv_method:  # Verify if CV method string was missing from result
            n_splits = r.get("cv_n_splits") or r.get("n_splits")  # Attempt to reconstruct from n_splits field
            if n_splits:  # Verify if n_splits value is available for reconstruction
                cv_method = f"StratifiedKFold(n_splits={n_splits})"  # Reconstruct CV method string from n_splits

        return cv_method  # Return the resolved or reconstructed CV method string
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_metric_values_into_row(r: Dict[str, Any], data: Dict[str, Optional[Any]], metric_map: Dict[str, tuple]) -> Dict[str, Optional[Any]]:
    """
    Extract and format metric values from a run result dict into a data row.

    :param r: Run result dictionary from a single RFE run.
    :param data: Result row dictionary to populate with metric values.
    :param metric_map: Mapping of CSV column names to (section, key) lookup tuples.
    :return: Updated data row dictionary with metric values populated.
    """

    try:
        for col, (section, key) in metric_map.items():  # Iterate over each metric column mapping
            val = r.get(col)  # Attempt direct lookup by column name first

            if val is None:  # Verify if direct lookup returned no value
                sec = r.get(section)  # Attempt nested section lookup using section key
                if isinstance(sec, dict):  # Verify the section value is a dict before accessing
                    val = sec.get(key)  # Extract metric value from nested section dict

            if val is not None:  # Verify a value was resolved before formatting
                try:  # Safely truncate numeric value to 4 decimal places
                    data[col] = truncate_value(float(val))  # Format as truncated float string
                except Exception:  # On float conversion or truncation failure
                    data[col] = val  # Preserve raw value when formatting fails

        return data  # Return updated data row with populated metric values
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_timing_values_into_row(r: Dict[str, Any], data: Dict[str, Optional[Any]]) -> Dict[str, Optional[Any]]:
    """
    Extract and format timing values from a run result dict into a data row.

    :param r: Run result dictionary containing timing fields.
    :param data: Result row dictionary to populate with timing values.
    :return: Updated data row dictionary with timing values populated.
    """

    try:
        feature_extraction_time = r.get("feature_extraction_time_s")  # Read feature extraction time if present in result

        try:  # Safely convert feature extraction time to rounded float
            data["feature_extraction_time_s"] = round(float(feature_extraction_time), 6) if feature_extraction_time is not None else None  # Store as float rounded to 6 decimals
        except Exception:  # On conversion failure
            data["feature_extraction_time_s"] = None  # Leave as None when conversion fails

        training_time = r.get("training_time_s")  # Read training time from result dict

        try:  # Safely convert training time to rounded float
            data["training_time_s"] = round(float(training_time), 6) if training_time is not None else None  # Store as float rounded to 6 decimals
        except Exception:  # On conversion failure
            data["training_time_s"] = None  # Leave as None when conversion fails

        testing_time = r.get("testing_time_s")  # Read testing time from result dict

        try:  # Safely convert testing time to rounded float
            data["testing_time_s"] = round(float(testing_time), 6) if testing_time is not None else None  # Store as float rounded to 6 decimals
        except Exception:  # On conversion failure
            data["testing_time_s"] = None  # Leave as None when conversion fails

        return data  # Return updated data row with populated timing values
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def serialize_feature_selection_fields(r: Dict[str, Any], data: Dict[str, Optional[Any]]) -> Dict[str, Optional[Any]]:
    """
    Serialize feature selection fields from a run result dict into a data row.

    :param r: Run result dictionary containing feature selection output fields.
    :param data: Result row dictionary to populate with serialized feature fields.
    :return: Updated data row dictionary with serialized feature selection fields populated.
    """

    try:
        try:  # Safely serialize top features list to JSON array
            data["top_features"] = json.dumps(r.get("top_features") or [], ensure_ascii=False)  # Serialize features to JSON array string
        except Exception:  # On serialization failure
            data["top_features"] = str(r.get("top_features") or [])  # Fallback to Python string representation

        try:  # Safely serialize RFE ranking dict to JSON object
            data["rfe_ranking"] = json.dumps(r.get("rfe_ranking") or {}, sort_keys=True, ensure_ascii=False)  # Serialize sorted JSON object string
        except Exception:  # On serialization failure
            data["rfe_ranking"] = str(r.get("rfe_ranking") or {})  # Fallback to Python string representation

        return data  # Return updated data row with serialized feature selection fields
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_result_row(r: Dict[str, Any], csv_path: str, cols: list) -> Dict[str, Optional[Any]]:
    """
    Build a single results row dictionary from a run result dict, CSV path and column list.

    :param r: Run result dictionary from a single RFE run.
    :param csv_path: Path to the original CSV dataset file.
    :param cols: List of column names for the results CSV.
    :return: Dict mapping column names to their formatted values.
    """

    try:
        data: Dict[str, Optional[Any]] = {c: None for c in cols}  # Initialize all columns to None
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")  # Generate current timestamp string
        data["timestamp"] = ts  # Assign timestamp to result row
        data["tool"] = "RFE"  # Assign tool identifier

        data["model"] = resolve_model_display_name(r)  # Resolve and assign friendly model display name

        try:  # Safely compute relative dataset path
            data["dataset"] = os.path.relpath(csv_path)  # Resolve relative path for portability
        except Exception:  # On relpath failure (e.g., different drives)
            data["dataset"] = csv_path  # Fallback to absolute path

        hyper = r.get("hyperparameters") or r.get("params") or {}  # Extract hyperparameters from result dict
        try:  # Safely serialize hyperparameters to JSON
            data["hyperparameters"] = json.dumps(hyper, sort_keys=True, ensure_ascii=False)  # Serialize sorted JSON string
        except Exception:  # On serialization failure
            data["hyperparameters"] = str(hyper)  # Fallback to string when JSON serialization fails

        data["cv_method"] = build_cv_method_string(r)  # Build and assign CV method description string

        data["train_test_split"] = r.get("train_test_split") or f"test_size={r.get('test_size', 0.2)}"  # Assign train/test split description
        data["scaling"] = r.get("scaling") or r.get("scaler") or r.get("preprocessing") or "standard"  # Assign scaling method identifier

        metric_map = {  # Mapping of result CSV columns to (section, key) lookup tuples
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

        data = extract_metric_values_into_row(r, data, metric_map)  # Extract and format all metric values into result row

        data = extract_timing_values_into_row(r, data)  # Extract and format all timing values into result row

        data = serialize_feature_selection_fields(r, data)  # Serialize feature selection fields into result row

        return data  # Return the completed result row dict
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def resolve_results_export_path(csv_path: str) -> Tuple[str, str]:
    """
    Resolve the absolute results directory and full CSV file path for RFE result export.

    :param csv_path: Path to the original CSV dataset file.
    :return: Tuple of (resolved_dir, run_csv_path).
    """

    try:
        cfg_source = CONFIG if isinstance(CONFIG, dict) and CONFIG else get_default_config()  # Use global config or fallback to defaults
        rfe_cfg = cfg_source.get("rfe") if isinstance(cfg_source.get("rfe"), dict) else cfg_source  # Extract RFE config section
        export_cfg = (rfe_cfg or {}).get("export", {})  # Extract export subsection
        results_dir_raw = export_cfg.get("results_dir")  # Read raw results directory setting
        results_filename = export_cfg.get("results_filename")  # Read results filename setting

        if not isinstance(results_dir_raw, str) or not results_dir_raw:  # Verify results_dir is a non-empty string
            raise ValueError("rfe.export.results_dir must be a non-empty string in configuration")  # Raise on invalid results directory
        if not isinstance(results_filename, str) or not results_filename:  # Verify results_filename is a non-empty string
            raise ValueError("rfe.export.results_filename must be a non-empty string in configuration")  # Raise on invalid filename
        if not results_filename.lower().endswith(".csv"):  # Verify filename ends with .csv
            raise ValueError("rfe.export.results_filename must end with .csv")  # Raise on invalid extension

        if os.path.isabs(results_dir_raw):  # Verify if the configured path is absolute
            resolved_dir = os.path.abspath(os.path.expanduser(results_dir_raw))  # Use absolute path directly
        else:  # Relative path: resolve relative to the dataset directory
            dataset_dir = os.path.dirname(csv_path) or "."  # Get dataset parent directory
            resolved_dir = os.path.abspath(os.path.expanduser(os.path.join(dataset_dir, results_dir_raw)))  # Join and resolve

        os.makedirs(resolved_dir, exist_ok=True)  # Ensure the results directory exists
        if not os.access(resolved_dir, os.W_OK):  # Verify write access to the results directory
            raise PermissionError(f"Directory not writable: {resolved_dir}")  # Raise on non-writable directory

        run_csv_path = os.path.join(resolved_dir, results_filename)  # Compose full CSV path
        print(f"{BackgroundColors.GREEN}Resolved results directory: {BackgroundColors.CYAN}{resolved_dir}{Style.RESET_ALL}")  # Log resolved directory to terminal

        return resolved_dir, run_csv_path  # Return resolved directory and full CSV path
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def load_and_merge_results_csv(run_csv_path: str, df_new: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Load an existing results CSV and merge it with the new results DataFrame.

    :param run_csv_path: Full path to the existing (or new) results CSV file.
    :param df_new: New results DataFrame to merge.
    :param cols: Ordered list of column names for the merged output.
    :return: Merged and sorted DataFrame ready for export.
    """

    try:
        if os.path.exists(run_csv_path):  # Verify if an existing results file is present
            try:  # Attempt to load and merge with existing data
                df_existing = pd.read_csv(run_csv_path, dtype=str)  # Load existing CSV with all columns as strings
                if "timestamp" not in df_existing.columns:  # Verify if timestamp column is missing
                    mtime = os.path.getmtime(run_csv_path)  # Get file modification time as fallback
                    back_ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d_%H_%M_%S")  # Format modification time as timestamp
                    df_existing["timestamp"] = back_ts  # Assign fallback timestamp to existing rows

                for c in cols:  # Iterate through expected columns
                    if c not in df_existing.columns:  # Verify if column is missing from existing data
                        df_existing[c] = None  # Add missing column with None values

                df_combined = pd.concat([df_existing[cols], df_new], ignore_index=True, sort=False)  # Concatenate existing and new results

                try:  # Attempt timestamp-based sorting
                    df_combined["timestamp_dt"] = pd.to_datetime(df_combined["timestamp"], format="%Y-%m-%d_%H_%M_%S", errors="coerce")  # Parse timestamp for sorting
                    df_combined = df_combined.sort_values(by="timestamp_dt", ascending=False)  # Sort by timestamp descending
                    df_combined = df_combined.drop(columns=["timestamp_dt"])  # Remove temporary sorting column
                except Exception:  # Fallback to lexicographic sort on string timestamp
                    df_combined = df_combined.sort_values(by="timestamp", ascending=False)  # Lexicographic fallback sort

                df_out = df_combined.reset_index(drop=True)  # Reset index after sorting
            except Exception:  # On any merge failure
                df_out = df_new  # Fall back to only new results
        else:  # No existing file: use only new results
            df_out = df_new  # Assign new results directly

        return df_out  # Return the merged and sorted DataFrame
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def save_rfe_results(csv_path, run_results):
    """
    Saves results from RFE run to a structured CSV file.

    :param csv_path: Original CSV file path.
    :param run_results: List of dicts containing results from the current run.
    :return: Path to the saved results CSV file.
    """

    try:
        verbose_output(f"{BackgroundColors.GREEN}Saving RFE run results to CSV...{Style.RESET_ALL}")

        runs = run_results if isinstance(run_results, list) else [run_results]

        rows = []
        for r in runs:
            cols = get_results_csv_columns(CONFIG if CONFIG else None)
            data = build_result_row(r, csv_path, cols)  # Build the complete row dict for this result
            rows.append(data)

        cols = get_results_csv_columns(CONFIG if CONFIG else None)
        df_new = pd.DataFrame(rows, columns=cols)

        resolved_dir, run_csv_path = resolve_results_export_path(csv_path)  # Resolve destination directory and full CSV path

        df_out = load_and_merge_results_csv(run_csv_path, df_new, cols)  # Load existing CSV and merge with new rows

        df_out = populate_hardware_column_and_order(df_out, CONFIG if CONFIG else None, column_name="hardware")

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
        cfg_source = CONFIG if isinstance(CONFIG, dict) and CONFIG else get_default_config()
        rfe_cfg_local = cfg_source.get("rfe") if isinstance(cfg_source.get("rfe"), dict) else cfg_source
        export_cfg_local = (rfe_cfg_local or {}).get("export", {})
        results_dir_raw_local = export_cfg_local.get("results_dir") or os.path.join("Feature_Analysis", "RFE")
        if os.path.isabs(results_dir_raw_local):
            resolved_dir_local = os.path.abspath(os.path.expanduser(results_dir_raw_local))
        else:
            dataset_dir_local = os.path.dirname(csv_path) or "."
            resolved_dir_local = os.path.abspath(os.path.expanduser(os.path.join(dataset_dir_local, results_dir_raw_local)))
        models_dir = os.path.join(resolved_dir_local, "Models")

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
    Function to load or export the final model, scaler, and parameters.

    :param csv_path: Path to the CSV dataset file
    :param X_train: Training features DataFrame
    :param y_train: Training target array
    :param top_features: List of top features
    :param feature_columns: Feature column names
    :return: final_model, scaler_full, top_features, loaded_hyperparams
    """
    
    try:
        loaded_hyperparams = None
        skip_train = CONFIG.get("rfe", {}).get("execution", {}).get("skip_train_if_model_exists") if isinstance(CONFIG, dict) else False
        if skip_train:
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
    Function to build the run_results dictionary.

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

        if is_verbose():
            print_metrics(metrics_tuple)
            print_top_features(top_features, rfe_ranking)

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


def split_and_scale_for_cv(X_numeric: pd.DataFrame, y_array: np.ndarray, random_state: int) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Split the dataset into train/test sets and scale the training features for cross-validation.

    :param X_numeric: Numeric features DataFrame.
    :param y_array: Target label array.
    :param random_state: Random seed for reproducibility.
    :return: Tuple of (X_train_df, X_test_df, y_train_array, y_test_array, X_train_scaled, X_test_scaled, scaling_time_s).
    """

    try:
        stratify_param = y_array if len(np.unique(y_array)) > 1 else None  # Avoid stratify for constant labels
        X_train_df, X_test_df, y_train_array, y_test_array = train_test_split(
            X_numeric, y_array, test_size=0.2, random_state=int(random_state), stratify=stratify_param
        )  # Split dataset into train/test portions

        scaler_for_run = StandardScaler()  # Create scaler for the run
        scale_start = time.perf_counter()  # Start perf_counter immediately before scaling (feature extraction window begins)
        X_train_scaled = scaler_for_run.fit_transform(X_train_df.values)  # Fit scaler and transform training data (scaling)
        X_test_scaled = scaler_for_run.transform(X_test_df.values)  # Transform test data using fitted scaler
        scale_end = time.perf_counter()  # End perf_counter immediately after scaling (feature extraction window part)
        scaling_time_s = round(scale_end - scale_start, 6)  # Compute scaling time rounded to 6 decimals

        return X_train_df, X_test_df, y_train_array, y_test_array, X_train_scaled, X_test_scaled, scaling_time_s  # Return all split and scaled artifacts
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def resolve_cv_n_splits(y_train_array: np.ndarray, n_folds: int) -> int:
    """
    Compute the effective number of CV splits based on available training data and class distribution.

    :param y_train_array: Training target array.
    :param n_folds: Configured maximum number of folds.
    :return: Effective number of CV splits as an int.
    """

    try:
        unique_tr, counts_tr = np.unique(y_train_array, return_counts=True)  # Get unique classes and their counts
        min_class_count_tr = counts_tr.min() if counts_tr.size > 0 else 0  # Minimum class count in training set

        n_splits = int(min(n_folds, len(y_train_array), min_class_count_tr))  # Effective splits bounded by folds, samples, and min class count
        if n_splits < 2:  # Verify sufficient splits for cross-validation
            raise ValueError(
                f"Effective number of CV splits is {n_splits}; must be >= 2 and <= number of training samples and smallest class count"
            )  # Raise on insufficient splits

        return n_splits  # Return the effective number of CV splits
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_cv_fold_loop(X_train_scaled: np.ndarray, y_train_array: np.ndarray, n_splits: int, n_features_to_select: Optional[int], step: int, estimator_name: str, random_state: int, csv_path: str, scaling_time_s: float) -> Tuple[list, list, list, float, float]:
    """
    Execute the stratified K-fold cross-validation loop and collect per-fold metrics, rankings, and timing.

    :param X_train_scaled: Scaled training feature array.
    :param y_train_array: Training target array.
    :param n_splits: Number of CV folds to execute.
    :param n_features_to_select: Number of features to select via RFE (or None for all).
    :param step: Number of features to remove at each RFE step.
    :param estimator_name: Name of the estimator to use for RFE.
    :param random_state: Random seed for reproducibility.
    :param csv_path: Path to the original CSV dataset file.
    :param scaling_time_s: Pre-computed scaling time to include in feature extraction accumulator.
    :return: Tuple of (fold_metrics, fold_rankings, fold_supports, total_elapsed, total_feature_extraction).
    """

    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(random_state))  # Initialize stratified K-fold splitter

        fold_metrics = []  # List to collect per-fold metric tuples
        fold_rankings = []  # List to collect per-fold ranking arrays
        fold_supports = []  # List to collect per-fold support masks
        total_elapsed = 0.0  # Accumulator for training times across folds
        total_feature_extraction = float(scaling_time_s)  # Accumulator for total feature extraction time (scaling + selector fits)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_train_scaled, y_train_array), start=1):  # Iterate over each fold
            verbose_output(f"{BackgroundColors.CYAN}Running fold {fold_idx}/{n_splits}{Style.RESET_ALL}")  # Optional fold progress

            X_train_fold = X_train_scaled[train_idx]  # Slice training data for this fold
            X_test_fold = X_train_scaled[test_idx]  # Slice testing data for this fold
            y_train_fold = y_train_array[train_idx]  # Slice training labels for this fold
            y_test_fold = y_train_array[test_idx]  # Slice testing labels for this fold

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
            )  # Notify fold completion via Telegram

        return fold_metrics, fold_rankings, fold_supports, total_elapsed, total_feature_extraction  # Return collected fold results and timing
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def aggregate_cv_fold_results(fold_metrics: list, fold_rankings: list, fold_supports: list, feature_columns, n_splits: int) -> Tuple[np.ndarray, Dict[str, float], list, list]:
    """
    Aggregate per-fold metrics, rankings, and support masks into mean metrics and majority-vote top features.

    :param fold_metrics: List of per-fold metric tuples.
    :param fold_rankings: List of per-fold ranking arrays.
    :param fold_supports: List of per-fold support mask arrays.
    :param feature_columns: Feature column names.
    :param n_splits: Number of CV folds executed.
    :return: Tuple of (mean_metrics, avg_rfe_ranking, top_features, sorted_rfe_ranking).
    """

    try:
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

        return mean_metrics, avg_rfe_ranking, top_features, sorted_rfe_ranking  # Return aggregated results
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_rfe_cv(csv_path, X_numeric, y_array, feature_columns, hyperparameters, n_features_to_select=None, step=1, estimator_name="random_forest", random_state=42):
    """
    Handles RFE with stratified cross-validation.

    :param csv_path: Path to the CSV dataset file.
    :param X_numeric: Numeric features DataFrame.
    :param y_array: Target array.
    :param feature_columns: Feature column names.
    :param hyperparameters: Hyperparameters dict.
    :param n_features_to_select: Number of features to select via RFE (or None for all).
    :param step: Number of features to remove at each RFE step.
    :param estimator_name: Name of the estimator to use for RFE.
    :param random_state: Random seed for reproducibility.
    :return: None.
    """

    try:
        verbose_output(f"{BackgroundColors.GREEN}Starting RFE with Stratified K-Fold Cross-Validation...{Style.RESET_ALL}")

        X_train_df, X_test_df, y_train_array, y_test_array, X_train_scaled, X_test_scaled, scaling_time_s = split_and_scale_for_cv(X_numeric, y_array, random_state)  # Split dataset and scale training features

        resolved_n = int(CONFIG.get("rfe", {}).get("cross_validation", {}).get("n_folds", 10))
        n_splits = resolve_cv_n_splits(y_train_array, resolved_n)  # Compute effective number of CV splits

        fold_metrics, fold_rankings, fold_supports, total_elapsed, total_feature_extraction = run_cv_fold_loop(
            X_train_scaled, y_train_array, n_splits, n_features_to_select, step, estimator_name, random_state, csv_path, scaling_time_s
        )  # Execute the stratified K-fold loop and collect per-fold results

        mean_metrics, avg_rfe_ranking, top_features, sorted_rfe_ranking = aggregate_cv_fold_results(
            fold_metrics, fold_rankings, fold_supports, feature_columns, n_splits
        )  # Aggregate per-fold metrics, rankings, and support masks into final results

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

        if is_verbose():
            print_metrics(tuple(mean_metrics))
            print_top_features(top_features, avg_rfe_ranking)

        save_rfe_results(csv_path, run_results)  # Save aggregated run results to CSV
        print_run_summary(run_results)  # Concise terminal summary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def resolve_rfe_run_params(n_features_to_select: Optional[int], step: Optional[int], estimator_name: Optional[str], random_state: Optional[int]) -> Tuple[Optional[int], int, str, int]:
    """
    Extract RFE execution parameters from the global CONFIG and apply CLI argument overrides.

    :param n_features_to_select: Optional CLI override for number of features to select.
    :param step: Optional CLI override for RFE step size.
    :param estimator_name: Optional CLI override for estimator name.
    :param random_state: Optional CLI override for random seed.
    :return: Tuple of (default_n_select, default_step, default_estimator, default_random_state).
    """

    try:
        rfe_cfg = CONFIG["rfe"]  # Access the validated RFE configuration block

        multi_cfg = rfe_cfg.get("multiprocessing", {})  # Extract multiprocessing config
        n_jobs = int(multi_cfg.get("n_jobs", -1))  # Read configured n_jobs value
        cpu_procs = int(multi_cfg.get("cpu_processes", 1))  # Read configured CPU process count

        cv_cfg = rfe_cfg.get("cross_validation", {})  # Extract cross-validation config
        n_folds = int(cv_cfg.get("n_folds", 10))  # Read configured fold count

        cache_cfg = rfe_cfg.get("caching", {})  # Extract caching config
        caching_enabled = bool(cache_cfg.get("enabled", True))  # Read caching enabled flag
        pickle_protocol = int(cache_cfg.get("pickle_protocol", 4))  # Read pickle protocol version

        sel_cfg = rfe_cfg.get("selection", {})  # Extract feature selection config
        cfg_n_select = sel_cfg.get("n_features_to_select")  # Read configured feature count
        cfg_step = sel_cfg.get("step", 1)  # Read configured RFE step size

        model_cfg = rfe_cfg.get("model", {})  # Extract model config
        cfg_estimator = model_cfg.get("estimator", "random_forest")  # Read configured estimator name
        cfg_random_state = int(model_cfg.get("random_state", 42))  # Read configured random state

        default_n_select = n_features_to_select if n_features_to_select is not None else cfg_n_select  # Prefer CLI override over config
        default_step = step if step is not None else cfg_step  # Prefer CLI override over config
        default_estimator = estimator_name if estimator_name is not None else cfg_estimator  # Prefer CLI override over config
        default_random_state = int(random_state) if random_state is not None else cfg_random_state  # Prefer CLI override over config

        return default_n_select, default_step, default_estimator, default_random_state  # Return resolved execution parameters
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_numeric_features(X: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Extract numeric feature columns from a DataFrame, attempting coercion for non-numeric columns.

    :param X: Feature DataFrame.
    :return: DataFrame containing only numeric features, or None if no numeric features are found.
    """

    try:
        X_numeric = X.select_dtypes(include=["number"]).copy()  # Select only numeric columns

        if X_numeric.shape[1] == 0:  # Verify if no numeric columns were detected
            coerced_cols = {}  # Dictionary to hold coerced numeric columns
            for col in X.columns:  # Iterate through all columns
                coerced = pd.to_numeric(X[col], errors="coerce")  # Attempt to coerce to numeric
                if coerced.notna().sum() > 0:  # Verify if any values were successfully coerced
                    coerced_cols[col] = coerced  # Add to coerced columns

            if coerced_cols:  # Verify if any columns were successfully coerced
                X_numeric = pd.DataFrame(coerced_cols, index=X.index)  # Create DataFrame from coerced columns
            else:  # No columns could be coerced
                print(f"{BackgroundColors.RED}No numeric features found after preprocessing. Cannot run RFE.{Style.RESET_ALL}")  # Log failure to find numeric features
                return None  # Return None to signal failure

        return X_numeric  # Return the numeric feature DataFrame
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_rfe(csv_path, n_features_to_select=None, step=None, estimator_name=None, random_state=None):
    """
    Runs Recursive Feature Elimination on the provided dataset, prints the single
    set of top features selected, computes and prints performance metrics, and
    saves the structured results.

    :param csv_path: Path to the CSV dataset file.
    :param n_features_to_select: Optional override for number of features to select.
    :param step: Optional override for RFE step size.
    :param estimator_name: Optional override for estimator name.
    :param random_state: Optional override for random seed.
    :return: None.
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Starting RFE analysis on dataset: {BackgroundColors.CYAN}{csv_path}{Style.RESET_ALL}"
        )

        hyperparameters = {}

        if not CONFIG or "rfe" not in CONFIG:
            raise ValueError("Configuration not initialized. Call get_config() and set CONFIG before running.")

        default_n_select, default_step, default_estimator, default_random_state = resolve_rfe_run_params(
            n_features_to_select, step, estimator_name, random_state
        )  # Extract config parameters and apply CLI overrides

        df = load_dataset(csv_path)  # Load dataset

        if df is None:  # If loading failed
            return  # Exit the function

        cleaned_df = preprocess_dataframe(df)  # Preprocess the DataFrame

        X = cleaned_df.iloc[:, :-1]  # Features are all columns except the last
        y = cleaned_df.iloc[:, -1]  # Target is the last column

        if X is None or y is None:  # If feature or target extraction failed
            return  # Exit the function

        X_numeric = extract_numeric_features(X)  # Extract numeric features, applying coercion if needed
        if X_numeric is None:  # Verify if numeric feature extraction failed
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


def play_sound():
    """
    Play a sound when the program finishes and skip if the operating system is Windows.

    :return: None.
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


def main(n_features_to_select=None, step=None, estimator_name=None, random_state=None, csv_path=None):
    """
    Main function.

    :param n_features_to_select: Number of top features to select via RFE; None uses config value.
    :param step: Number of features to remove per RFE iteration; None uses config value.
    :param estimator_name: Name of the estimator model to use; None uses config value.
    :param random_state: Random seed for reproducibility; None uses config value.
    :param csv_path: Path to the input CSV dataset file; None uses config value.
    :return: None.
    """
    
    try:
        print(
            f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Recursive Feature Elimination (RFE){BackgroundColors.GREEN} program!{Style.RESET_ALL}"
        )  # Output the welcome message
        start_time = datetime.datetime.now()  # Record program start time
        setup_telegram_bot()  # Initialize Telegram bot for progress notifications

        dataset_to_use = csv_path or CONFIG.get("rfe", {}).get("execution", {}).get("dataset_path")  # Resolve dataset path from argument or config
        if not dataset_to_use:  # Raise if no dataset path was provided
            raise ValueError("No dataset path provided. Use --dataset_path or set rfe.execution.dataset_path in config.")

        dataset_name = os.path.splitext(os.path.basename(dataset_to_use))[0]  # Extract dataset name from file path
        send_telegram_message(TELEGRAM_BOT, [f"Starting RFE Feature Selection on {dataset_name} at {start_time.strftime('%d/%m/%Y - %H:%M:%S')}"])  # Notify Telegram about program start
        run_rfe(dataset_to_use, n_features_to_select=n_features_to_select, step=step, estimator_name=estimator_name, random_state=random_state)  # Execute RFE feature selection pipeline

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
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """

    try:  # Protect top-level execution to ensure errors are reported and notified
        cli = parse_cli_args()  # Parse CLI arguments into a dict
        CLI_ARGS.update(cli)  # Update global CLI_ARGS with parsed values
        cfg, sources = get_config(cli)  # Merge configuration from defaults/file/CLI
        CONFIG.update(cfg)  # Populate global CONFIG with merged config

        logger = Logger(f"./Logs/{Path(__file__).stem}.log", clean=True)  # Initialize file logger for this script
        sys.stdout = logger  # Redirect stdout to logger to capture prints
        sys.stderr = logger  # Redirect stderr to logger to capture errors
        setup_global_exception_hook()  # Install global exception hook that forwards to Telegram

        dataset_arg = cli.get("dataset_path") or CONFIG.get("rfe", {}).get("execution", {}).get("dataset_path")  # Resolve dataset path from CLI or config
        if dataset_arg is None:  # Validate required dataset argument presence
            raise ValueError("dataset path must be provided via --dataset_path or config.rfe.execution.dataset_path")  # Raise if missing

        n_features = cli.get("n_features_to_select") if cli.get("n_features_to_select") is not None else None  # Extract optional n_features
        step_arg = cli.get("step") if cli.get("step") is not None else None  # Extract optional step value
        estimator_arg = cli.get("estimator") if cli.get("estimator") is not None else None  # Extract optional estimator override
        rs_arg = cli.get("random_state") if cli.get("random_state") is not None else None  # Extract optional random state

        try:  # Run main and handle user interrupts separately
            main(n_features_to_select=n_features, step=step_arg, estimator_name=estimator_arg, random_state=rs_arg, csv_path=dataset_arg)  # Invoke main business logic
        except KeyboardInterrupt:  # Handle user-initiated interrupts with friendly notification
            try:  # Attempt graceful interrupt notification and cleanup
                print("Execution interrupted by user (KeyboardInterrupt)")  # Inform terminal about user interrupt
                send_telegram_message(TELEGRAM_BOT, ["RFE execution interrupted by user (KeyboardInterrupt)"])  # Notify via Telegram about interrupt
            except Exception:  # Ignore failures sending interrupt notification to avoid masking the interrupt
                pass  # No-op on notification failure
            try:  # Best-effort logger flush/close during interrupt handling
                if logger is not None:  # Only flush/close if logger exists
                    logger.flush()  # Flush pending log writes
                    logger.close()  # Close the logger file handle
            except Exception:  # Ignore logger cleanup errors during interrupt handling
                pass  # Continue to re-raise the interrupt
            raise  # Re-raise KeyboardInterrupt to preserve original behavior and exit code
    except BaseException as e:  # Catch everything (including SystemExit) and report via Telegram
        try:  # Try to log and notify about the fatal error
            print(f"Fatal error: {e}")  # Print the exception message to terminal for visibility
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback and message via Telegram
        except Exception:  # If notification fails, attempt to print traceback to stderr as fallback
            try:  # Attempt fallback traceback printing for diagnostics
                traceback.print_exc()  # Print full traceback to stderr as a fallback notification
            except Exception:  # Ignore failures of the fallback printing to avoid cascading errors
                pass  # No further fallback available
        try:  # Attempt best-effort logger cleanup after fatal error
            if logger is not None:  # Only flush/close if logger exists
                logger.flush()  # Flush pending log writes
                logger.close()  # Close the logger file handle
        except Exception:  # Ignore logger cleanup errors to avoid masking the primary failure
            pass  # No-op on cleanup failure
        raise  # Re-raise the original exception to preserve non-zero exit code and behavior
