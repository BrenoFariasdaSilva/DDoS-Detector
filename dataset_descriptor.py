"""
================================================================================
Dataset Descriptor and Report Generator - dataset_descriptor.py
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-10-07

What this module does
    - Recursively scans a directory (or single CSV file) and collects all
        matching CSV datasets.
    - Extracts metadata and summaries per file: sample/feature counts,
        feature types, missing values, detected label column and class
        distributions.
    - Optionally generates a 2D t-SNE plot per file (`Data_Separability/`),
        using class-aware downsampling with a default target of 2000 (config
        in callers); small classes (default min 50) are preserved in full.
    - Optionally computes cross-dataset compatibility reports comparing
        feature unions/intersections between dataset groups (`CROSS_DATASET_VALIDATE`).

Key defaults and globals
    - File discovery default extension: .csv
    - Results saved under each dataset base directory in `RESULTS_DIR`
        (default: ./Dataset_Description/). The per-dataset CSV is named
        `Dataset_Descriptor.csv` (config: `RESULTS_FILENAME`).
    - Cross-group report: saved as `Cross_{RESULTS_FILENAME}` in each
        group's results directory when `CROSS_DATASET_VALIDATE = True`.
    - t-SNE: uses sklearn.manifold.TSNE and adapts to `n_iter`/`max_iter`
        parameter name differences across scikit-learn versions.

Behavioral notes & guarantees
    - Downsampling is class-aware: classes with >= `min_class_size` receive
        at least `min_class_size` samples when possible; classes with fewer
        samples are included entirely. Remaining budget is distributed
        proportionally using a fractional remainder method.
    - Numeric extraction tries `select_dtypes(include=["number"])` and
        attempts coercion of object/string columns to numeric when needed.
    - The script performs disk-space checks before writing large outputs.
    - The generator writes one cross-dataset CSV per dataset group and
        normalizes rows so the file's group appears as "Dataset A".

Usage
    - Run the script directly: `python3 dataset_descriptor.py` (adjust
        `DATASETS` constant or call `generate_dataset_report()` programmatically).

Dependencies
    - Python 3.9+
    - pandas, numpy, matplotlib, scikit-learn, tqdm, colorama

Limitations / TODO
    - Header detection and CSV parsing are pragmatic; malformed CSVs may
        require preprocessing.
    - Add CLI flags for `sample_size`, `min_class_size`, `CROSS_DATASET_VALIDATE`.
    - Consider structured logging instead of printing/redirecting stdout.
"""

import atexit  # For playing a sound when the program finishes
import dataframe_image as dfi  # For exporting DataFrame as PNG images
import datetime  # For timestamping
import gc  # For explicit garbage collection
import matplotlib.pyplot as plt  # For plotting t-SNE results
import numpy as np  # For numerical operations
import os  # For running a command in the terminal
import pandas as pd  # For data manipulation
import platform  # For getting the operating system name
import re  # For regex operations
import sys  # For system-specific parameters and functions
import telegram_bot as telegram_module  # For setting Telegram prefix and device info
import traceback  # For printing full exception tracebacks
import yaml  # For optional config.yaml loading when locating WGANGP outputs
import warnings  # For suppressing pandas warnings when requested
from colorama import Style  # For coloring the terminal
from inspect import signature  # For inspecting function signatures
from Logger import Logger  # For logging output to both terminal and file
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from pathlib import Path  # For handling file paths
from sklearn.manifold import TSNE  # For t-SNE dimensionality reduction
from sklearn.preprocessing import StandardScaler  # For feature scaling
from telegram_bot import TelegramBot, send_exception_via_telegram, send_telegram_message, setup_global_exception_hook  # For sending progress messages and exceptions to Telegram
from tqdm import tqdm  # For progress bars
from typing import Any, cast  # For type hinting
import argparse
import yaml
import os
import sys

setup_global_exception_hook()  # Install global exception handler to catch unhandled exceptions


# Macros:
class BackgroundColors:  # Colors for the terminal
    CYAN = "\033[96m"  # Cyan
    GREEN = "\033[92m"  # Green
    YELLOW = "\033[93m"  # Yellow
    RED = "\033[91m"  # Red
    BOLD = "\033[1m"  # Bold
    UNDERLINE = "\033[4m"  # Underline
    CLEAR_TERMINAL = "\033[H\033[J"  # Clear the terminal


# Execution Constants will be sourced from configuration (CLI > config.yaml > defaults)

SOUND_COMMANDS = {
    "Darwin": "afplay",
    "Linux": "aplay",
    "Windows": "start",
}

SOUND_FILE = "./.assets/Sounds/NotificationSound.wav"


def get_default_config() -> dict:
    """
    Return default configuration used by dataset_descriptor.py.
    
    :param None
    :return: A dictionary containing default configuration values for dataset_descriptor.py
    """
    return {
        "dataset_descriptor": {
            "include_preprocessing_metrics": True,
            "include_data_augmentation_info": True,
            "generate_table_image": True,
            "table_image_format": "png",
            "csv_output_suffix": "_description",
            "class_column_name": "Label",
            "dropna_before_analysis": False,
            "compute_class_distribution": True,
            "compute_feature_statistics": True,
            "round_decimals": 4,
        },
        "paths": {
            "dataset_description_subdir": "Dataset_Description",
            "preprocessing_summary_subdir": "Preprocessing_Summary",
            "logs_dir": "./Logs",
        },
        "execution": {
            "verbose": False,
            "cross_dataset_validate": True,
        },
        "datasets": {},
    }


def deep_merge_dicts(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base and return new dict.
    
    :param base: The base dictionary to merge into (not modified)
    :param override: The dictionary with override values (not modified)
    :return: A new dictionary resulting from deep merging override into base
    """
    
    result = dict(base)
    for k, v in (override or {}).items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge_dicts(result[k], v)
        else:
            result[k] = v
    return result


def load_config_file(path: str = "config.yaml") -> dict:
    """
    Load configuration from a YAML file if it exists, otherwise return an empty dict.

    :param path: Path to the YAML configuration file (default: "config.yaml")
    :return: Configuration dictionary loaded from the file, or empty dict if file doesn't exist or fails to load
    """
    
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception:
            return {}
    return {}


def parse_cli_args(argv=None) -> dict:
    """
    Parse CLI arguments and return a dictionary of config overrides.

    :param argv: List of command-line arguments (default: None, which uses sys.argv)
    :return: Dictionary of config overrides based on CLI arguments
    """
    
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--include_preprocessing_metrics", dest="include_preprocessing_metrics", action="store_true")
    parser.add_argument("--no-include_preprocessing_metrics", dest="include_preprocessing_metrics", action="store_false")
    parser.add_argument("--include_data_augmentation_info", dest="include_data_augmentation_info", action="store_true")
    parser.add_argument("--no-include_data_augmentation_info", dest="include_data_augmentation_info", action="store_false")
    parser.add_argument("--generate_table_image", dest="generate_table_image", action="store_true")
    parser.add_argument("--no-generate_table_image", dest="generate_table_image", action="store_false")
    parser.add_argument("--table_image_format", dest="table_image_format")
    parser.add_argument("--csv_output_suffix", dest="csv_output_suffix")
    parser.add_argument("--class_column_name", dest="class_column_name")
    parser.add_argument("--dropna_before_analysis", dest="dropna_before_analysis", action="store_true")
    parser.add_argument("--no-dropna_before_analysis", dest="dropna_before_analysis", action="store_false")
    parser.add_argument("--compute_class_distribution", dest="compute_class_distribution", action="store_true")
    parser.add_argument("--no-compute_class_distribution", dest="compute_class_distribution", action="store_false")
    parser.add_argument("--compute_feature_statistics", dest="compute_feature_statistics", action="store_true")
    parser.add_argument("--no-compute_feature_statistics", dest="compute_feature_statistics", action="store_false")
    parser.add_argument("--round_decimals", dest="round_decimals", type=int)
    parser.add_argument("--config", dest="config", default="config.yaml")
    args, _ = parser.parse_known_args(argv)
    return {k: v for k, v in vars(args).items() if v is not None}


def get_config(file_path: str = "config.yaml", cli_args: dict | None = None) -> dict:
    """
    Load and merge configuration with precedence CLI > config.yaml > defaults.

    :param file_path: Path to the configuration YAML file (default: "config.yaml")
    :param cli_args: Dictionary of CLI arguments that were parsed (optional)
    :return: Merged configuration dictionary
    """
    
    defaults = get_default_config()
    file_conf = load_config_file(file_path)
    merged = deep_merge_dicts(defaults, file_conf)

    if cli_args:
        dd = merged.setdefault("dataset_descriptor", {})
        for key in [
            "include_preprocessing_metrics",
            "include_data_augmentation_info",
            "generate_table_image",
            "table_image_format",
            "csv_output_suffix",
            "class_column_name",
            "dropna_before_analysis",
            "compute_class_distribution",
            "compute_feature_statistics",
            "round_decimals",
        ]:
            if key in cli_args and cli_args[key] is not None:
                dd[key] = cli_args[key]
        if "verbose" in cli_args and cli_args["verbose"] is not None:
            merged.setdefault("execution", {})["verbose"] = cli_args["verbose"]
    return merged


def init_runtime(config: dict):
    """
    Initialize runtime artifacts (logger) based on provided config.

    :param config: The merged configuration dictionary
    :param cli_args: The dictionary of CLI arguments that were parsed (optional)
    :return: A dictionary containing runtime artifacts and settings
    """

    validate_config_structure(config)

    logs_dir = config.get("paths", {}).get("logs_dir", "./Logs")
    os.makedirs(logs_dir, exist_ok=True)
    logger = Logger(os.path.join(logs_dir, f"{Path(__file__).stem}.log"), clean=True)

    runtime = {
        "logger": logger,
        "verbose": bool(config.get("execution", {}).get("verbose", False)),
        "results_dir": os.path.join(".", config.get("paths", {}).get("dataset_description_subdir", "Dataset_Description")),
        "results_filename_suffix": config.get("dataset_descriptor", {}).get("csv_output_suffix", "_description"),
        "ignore_files": list(config.get("paths", {}).get("ignore_files", []) or []),
        "ignore_dirs": list(config.get("paths", {}).get("ignore_dirs", []) or []),
    }
    return runtime


def log_config_sources(config: dict, cli_args: dict | None = None):
    """
    Log configuration values with their source (CLI/config/default).

    :param config: The merged configuration dictionary
    :param cli_args: The dictionary of CLI arguments that were
    :return: None
    """
    
    dd = config.get("dataset_descriptor", {})
    for k, v in dd.items():
        src = "config"
        if cli_args and k in cli_args and cli_args[k] is not None:
            src = "CLI"
        elif k not in (load_config_file().get("dataset_descriptor") or {}):
            src = "default"
        print(f"[CONFIG] {k} = {v} (source: {src})")


def validate_config_structure(config: dict):
    """
    Ensure required keys exist and have correct types for dataset_descriptor.

    :param config: The configuration dictionary to validate
    :return: None, raises ValueError if validation fails
    """
    
    if not isinstance(config, dict):
        raise ValueError("config must be a dictionary")
    dd = config.get("dataset_descriptor")
    if not isinstance(dd, dict):
        raise ValueError("config.dataset_descriptor must be a mapping")

    expected = {
        "include_preprocessing_metrics": bool,
        "include_data_augmentation_info": bool,
        "generate_table_image": bool,
        "table_image_format": str,
        "csv_output_suffix": str,
        "class_column_name": str,
        "dropna_before_analysis": bool,
        "compute_class_distribution": bool,
        "compute_feature_statistics": bool,
        "round_decimals": int,
    }
    
    for key, typ in expected.items():
        if key not in dd:
            raise ValueError(f"Missing required config key: dataset_descriptor.{key}")
        if not isinstance(dd[key], typ):
            raise ValueError(f"Invalid type for dataset_descriptor.{key}: expected {typ.__name__}")

    if dd["round_decimals"] < 0:
        raise ValueError("dataset_descriptor.round_decimals must be >= 0")

# Functions Definitions:


def verbose_output(true_string="", false_string=""):
    """
    Outputs a message if the VERBOSE constant is set to True.

    :param true_string: The string to be outputted if the VERBOSE constant is set to True.
    :param false_string: The string to be outputted if the VERBOSE constant is set to False.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_flag = os.environ.get("DD_DESCRIPTOR_VERBOSE", "False").lower() in ("1", "true", "yes")
        if verbose_flag and true_string != "":
            print(true_string)
        elif false_string != "":
            print(false_string)
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def verify_dot_env_file():
    """
    Verifies if the .env file exists in the current directory.

    :return: True if the .env file exists, False otherwise
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        env_path = Path(__file__).parent / ".env"  # Path to the .env file
        if not env_path.exists():  # If the .env file does not exist
            print(f"{BackgroundColors.CYAN}.env{BackgroundColors.YELLOW} file not found at {BackgroundColors.CYAN}{env_path}{BackgroundColors.YELLOW}. Telegram messages may not be sent.{Style.RESET_ALL}")
            return False  # Return False

        return True  # Return True if the .env file exists
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def setup_telegram_bot():
    """
    Sets up the Telegram bot for progress messages.

    :return: None
    """
    
    try:  # Wrap full function logic to ensure production-safe monitoring
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def verify_filepath_exists(filepath):
    """
    Verify if a file or folder exists at the specified path.

    :param filepath: Path to the file or folder
    :return: True if the file or folder exists, False otherwise
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Verifying if the file or folder exists at the path: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}"
        )  # Output the verbose message

        return os.path.exists(filepath)  # Return True if the file or folder exists, False otherwise
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def collect_matching_files(
    input_dir,
    file_format=".csv",
    ignore_files: list | None = None,
    ignore_dirs: list | None = None,
    config: dict | None = None,
):
    """
    Recursively collects all files in the specified directory and subdirectories
    that match the given file format and are not in the ignore list.

    :param input_dir: Directory to search
    :param file_format: File format to include (default: .csv)
    :param ignore_files: List of filenames to ignore
    :param ignore_dirs: List of directory names to ignore
    :return: Sorted list of matching file paths
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Collecting all files with format {BackgroundColors.CYAN}{file_format}{BackgroundColors.GREEN} in directory: {BackgroundColors.CYAN}{input_dir}{Style.RESET_ALL}"
        )  # Output the verbose message

        cfg = config or get_default_config()
        resolved_ignore_files = ignore_files if ignore_files is not None else list(cfg.get("paths", {}).get("ignore_files", []) or [])
        resolved_ignore_dirs = ignore_dirs if ignore_dirs is not None else list(cfg.get("paths", {}).get("ignore_dirs", ["Cache", "Data_Separability", "Dataset_Description", "Feature_Analysis"]) or ["Cache", "Data_Separability", "Dataset_Description", "Feature_Analysis"])

        ignore_files = set(os.path.normcase(f) for f in (resolved_ignore_files or []))
        ignore_dirs = set(os.path.normcase(d) for d in (resolved_ignore_dirs or []))

        matching_files = []  # List to store matching file paths

        for root, dirs, files in os.walk(input_dir):  # Walk through the directory
            try:  # Try to filter out ignored directories
                dirs[:] = [
                    d for d in dirs if os.path.normcase(d) not in ignore_dirs
                ]  # Modify dirs in-place to skip ignored directories
            except Exception:  # If an error occurs while filtering directories
                pass  # Ignore the error and continue

            for file in files:  # For each file
                if not file.endswith(file_format):  # Skip files that do not match the specified format
                    continue  # Continue to the next file

                basename_norm = os.path.normcase(file)  # Normalize the basename for case-insensitive comparison
                fullpath = os.path.join(root, file)  # Get the full file path
                fullpath_norm = os.path.normcase(fullpath)  # Normalize the full file path for case-insensitive comparison

                if basename_norm in ignore_files or fullpath_norm in ignore_files:  # If the file is in the ignore list
                    verbose_output(f"Skipping ignored file: {fullpath}")  # Output verbose message for ignored file
                    continue  # Continue to the next file

                matching_files.append(fullpath)  # Add the full file path to the list

        sorted_matching_files = sorted(set(matching_files))  # Remove duplicates and sort the list

        return sorted_matching_files  # Return the sorted list of matching files
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def build_headers_map(filepaths, low_memory=True):
    """
    Build a mapping of file path -> list of header columns.

    Attempts a lightweight header-only read (`pd.read_csv(..., nrows=0)`)
    and falls back to `load_dataset` if that fails.

    :param filepaths: Iterable of file paths
    :param low_memory: Passed to `load_dataset` when falling back
    :return: dict mapping filepath -> list of column names
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Building headers map for provided file paths.{Style.RESET_ALL}"
        )  # Output the verbose message

        headers = {}  # Dictionary that will map each filepath to its list of columns
        for fp in filepaths:  # Iterate over all given file paths
            try:  # Try header-only read
                df_headers = pd.read_csv(fp, nrows=0)  # Extract columns without loading file content
                df_headers.columns = df_headers.columns.str.strip()  # Remove leading/trailing whitespace from column names
                cols = df_headers.columns.tolist()  # Get column list
            except Exception:  # If header-only read fails
                df_tmp = load_dataset(fp, low_memory=low_memory)  # Load full dataset (slow fallback)
                cols = df_tmp.columns.tolist() if df_tmp is not None else []  # Extract columns if dataset loaded
            headers[fp] = cols  # Store the resolved column list for this file

        return headers  # Return filepath->headers mapping
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def compute_common_features(headers_map):
    """
    Compute the intersection of headers across all files and determine
    whether all headers match exactly (ignoring case and surrounding whitespace).

    :param headers_map: dict mapping filepath -> list of column names
    :return: tuple (common_features_set, headers_match_all_bool)
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        try:  # Normalize all headers to lowercase and strip whitespace
            normalized_sets = [
                set(col.strip().lower() for col in v) for v in headers_map.values() if v
            ]  # List of normalized header sets
            if normalized_sets:  # If there are valid header sets
                common = set.intersection(*normalized_sets)  # Compute intersection across all files
            else:  # If no headers available
                common = set()  # Empty intersection
        except Exception:  # Catch unexpected failures
            common = set()  # Fallback to empty set

        unique_normalized_sets = {
            frozenset(col.strip().lower() for col in v) for v in headers_map.values() if v
        }  # Unique normalized header sets
        match_all = len(unique_normalized_sets) <= 1  # True if all header sets are identical

        return common, match_all  # Return shared features and match flag
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_dataset(filepath, low_memory=True):
    """
    Loads a dataset from a CSV file.

    :param filepath: Path to the CSV file
    :param low_memory: Whether to use low memory mode (default: True)
    :return: Pandas DataFrame
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        try:  # Try to load the dataset
            with warnings.catch_warnings():  # Suppress DtypeWarning warnings
                warnings.simplefilter("ignore", pd.errors.DtypeWarning)  # Ignore DtypeWarning warnings
                df = pd.read_csv(filepath, low_memory=low_memory)  # Load the dataset
                df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names

            return df  # Return the DataFrame
        except Exception as e:  # If an error occurs
            print(f"{BackgroundColors.RED}Error loading {BackgroundColors.GREEN}{filepath}: {e}{Style.RESET_ALL}")
            return None  # Return None if an error occurs
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def sanitize_feature_names(columns):
    r"""
    Sanitize column names by removing special JSON characters that LightGBM doesn't support.
    Replaces: { } [ ] : , " \ with underscores.

    :param columns: pandas Index or list of column names
    :return: list of sanitized column names
    """
    
    try:  # Wrap full function logic to ensure production-safe monitoring
        sanitized = []  # List to store sanitized column names
        
        for col in columns:  # Iterate over each column name
            clean_col = re.sub(r"[{}\[\]:,\"\\]", "_", str(col))  # Replace special characters with underscores
            clean_col = re.sub(r"_+", "_", clean_col)  # Replace multiple underscores with a single underscore
            clean_col = clean_col.strip("_")  # Remove leading/trailing underscores
            sanitized.append(clean_col)  # Add sanitized column name to the list
            
        return sanitized  # Return the list of sanitized column names
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def preprocess_dataframe(df, remove_zero_variance=True):
    """
    Preprocess a DataFrame by removing rows with NaN or infinite values and
    dropping zero-variance numeric features.

    :param df: pandas DataFrame to preprocess
    :param remove_zero_variance: whether to drop numeric columns with zero variance
    :return: cleaned DataFrame
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def detect_label_column(columns):
    """
    Try to guess the label column based on common naming conventions.

    :param columns: List of column names
    :return: The name of the label column if found, else None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        candidates = ["label", "class", "target"]  # Common label column names

        for col in columns:  # First search for exact matches
            if col.lower() in candidates:  # Verify if the column name matches any candidate exactly
                return col  # Return the column name if found

        for col in columns:  # Second search for partial matches
            if "target" in col.lower() or "label" in col.lower():  # Verify if the column name contains any candidate
                return col  # Return the column name if found

        return None  # Return None if no label column is found
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def summarize_features(df):
    """
    Summarizes number of samples, features, and feature types.
    Ensures the sum of feature types matches the number of columns.

    :param df: pandas DataFrame
    :return: Tuple containing:
             n_samples, n_features, n_numeric, n_int, n_categorical, n_other, categorical columns string
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        n_samples, n_features = df.shape  # Get number of samples and features
        dtypes = df.dtypes  # Get data types of each column

        n_numeric = dtypes[dtypes == "float64"].count()  # Count float64 types
        n_int = dtypes[dtypes == "int64"].count() + dtypes[dtypes == "Int64"].count()  # Count int64 and Int64 types
        n_categorical = dtypes[dtypes.isin(["object", "category", "bool", "string"])].count()  # Count categorical types

        n_other = n_features - (n_numeric + n_int + n_categorical)  # Anything else goes to "other"

        categorical_cols = df.select_dtypes(
            include=["object", "category", "bool", "string"]
        ).columns.tolist()  # List of categorical columns
        categorical_cols_str = (
            ", ".join(categorical_cols) if categorical_cols else "None"
        )  # Create string of categorical columns or "None"

        return (
            n_samples,
            n_features,
            n_numeric,
            n_int,
            n_categorical,
            n_other,
            categorical_cols_str,
        )  # Return the summary values
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def summarize_missing_values(df):
    """
    Summarizes missing values for the dataset.

    :param df: The pandas DataFrame
    :return: Summary string of missing values
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        missing_vals = df.isnull().sum()  # Get count of missing values per column
        missing_summary = (
            ", ".join([f"{col} ({cnt})" for col, cnt in missing_vals.items() if cnt > 0])
            if missing_vals.sum() > 0
            else "None"
        )  # Create summary string or "None"

        return missing_summary  # Return the missing values summary
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def summarize_classes(df, label_col):
    """
    Summarizes classes and class distributions if a label column exists.

    :param df: The pandas DataFrame
    :param label_col: The name of the label column
    :return: Tuple containing string of classes and class distribution summary
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        if label_col and label_col in df.columns:  # If a label column exists
            classes = df[label_col].unique()  # Get unique classes
            classes_str = ", ".join(map(str, classes))  # Create string of classes
            class_counts = df[label_col].value_counts()  # Get counts of each class
            total = class_counts.sum()  # Total number of samples
            class_dist_list = [
                f"{cls}: {cnt} ({cnt/total*100:.2f}%)" for cls, cnt in class_counts.items()
            ]  # Create class distribution list
            class_dist_str = ", ".join(class_dist_list)  # Create class distribution string
            return classes_str, class_dist_str  # Return the classes and class distribution

        return "None", "None"  # Return "None" if no label column
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def coerce_numeric_columns(df):
    """
    Try to extract numeric columns from `df`. If no numeric columns are
    present, attempt to coerce object/string columns to numeric values.

    :param df: pandas DataFrame
    :return: DataFrame with numeric columns (may be empty)
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Extracting or coercing numeric columns from the DataFrame.{Style.RESET_ALL}"
        )  # Output the verbose message

        numeric_df = df.select_dtypes(include=["number"]).copy()  # Select numeric columns from the DataFrame
        if numeric_df.empty:  # If there are no numeric columns found
            obj_cols = df.select_dtypes(
                include=["object", "string"]
            ).columns.tolist()  # List object/string columns as candidates
            for c in obj_cols:  # Iterate over candidate object/string columns
                coerced = pd.to_numeric(df[c], errors="coerce")  # Attempt to coerce the column to numeric, invalid -> NaN
                if coerced.notna().sum() > 0:  # If coercion produced any non-NaN values
                    numeric_df[c] = coerced  # Add the coerced column to the numeric DataFrame

        return numeric_df  # Return the numeric-only DataFrame (may be empty)
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def fill_replace_and_drop(numeric_df):
    """
    Replace infinities, drop all-NaN columns, and fill remaining NaNs with
    the column median (or 0 when median is NaN).

    :param numeric_df: DataFrame with numeric columns
    :return: cleaned DataFrame (may be empty)
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Replacing {BackgroundColors.CYAN}infinities, dropping all-NaN columns, and filling NaNs{BackgroundColors.GREEN} with column medians.{Style.RESET_ALL}"
        )  # Output the verbose message

        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)  # Replace +/-infinity with NaN
        numeric_df = numeric_df.loc[:, numeric_df.notna().any(axis=0)]  # Drop columns that are entirely NaN
        if numeric_df.shape[1] == 0:  # If no columns remain after dropping
            return numeric_df  # Return the (empty) DataFrame

        for col in numeric_df.columns:  # Iterate over numeric columns
            med = numeric_df[col].median()  # Compute column median
            numeric_df[col] = numeric_df[col].fillna(0 if pd.isna(med) else med)  # Fill NaNs with median or 0

        return numeric_df  # Return cleaned numeric DataFrame
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def compute_initial_alloc(counts, min_per_class):
    """
    Compute initial per-class allocations capped by `min_per_class`.

    This helper computes the initial allocation for each class as the
    minimum of the class count and the requested `min_per_class` value and
    returns the allocation mapping together with the sum of those values.

    :param counts: pandas Series with per-class counts
    :param min_per_class: preferred minimum samples per class
    :return: Tuple (initial_alloc dict, s_min int)
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        initial = {c: min(int(counts[c]), int(min_per_class)) for c in counts.index}  # Compute min(count, min_per_class)
        s = sum(initial.values())  # Sum of initial allocations

        return initial, s  # Return tuple (initial_alloc, s_min)
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def allocate_with_min(initial_alloc, counts, max_samples):
    """
    Distribute remaining capacity after satisfying per-class minima.

    Starting from `initial_alloc` (which already enforces per-class minima),
    this helper distributes the remaining available capacity proportionally
    to classes that still have unused samples. It performs integer flooring
    and then distributes leftover units according to fractional remainders
    to produce a final integer allocation per class.

    :param initial_alloc: dict mapping class -> allocated minima
    :param counts: pandas Series with per-class counts
    :param max_samples: total maximum samples to allocate
    :return: dict mapping class -> final allocation
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        alloc = dict(initial_alloc)  # Start with initial allocations
        remaining_local = max_samples - sum(initial_alloc.values())  # Remaining capacity after minima
        rem_avail_local = {c: max(0, int(counts[c]) - alloc[c]) for c in counts.index}  # Remaining available per class
        total_rem_avail_local = sum(rem_avail_local.values())  # Total remaining available

        if total_rem_avail_local > 0 and remaining_local > 0:  # Only proceed if there is capacity to distribute
            float_add_local = {
                c: (remaining_local * rem_avail_local[c] / total_rem_avail_local) for c in counts.index
            }  # Proportional fractional add
            add_alloc_local = {c: int(float_add_local[c]) for c in counts.index}  # Base integer additional allocation
            assigned_local = sum(add_alloc_local.values())  # Sum of base additional allocations
            leftover_local = remaining_local - assigned_local  # Leftover after flooring
            remainders_local = sorted(
                counts.index, key=lambda c: (float_add_local[c] - add_alloc_local[c]), reverse=True
            )  # Order by fractional remainder

            for c in remainders_local:  # Distribute leftover one-by-one
                if leftover_local <= 0:  # Stop when no leftover remains
                    break  # Exit distribution
                if add_alloc_local[c] < rem_avail_local[c]:  # Only add if class can accept more
                    add_alloc_local[c] += 1  # Increment allocation for this class
                    leftover_local -= 1  # Decrease leftover count
            for c in counts.index:  # Finalize allocations applying available caps
                alloc[c] += min(add_alloc_local.get(c, 0), rem_avail_local[c])  # Cap addition by remaining available

        return alloc  # Return finalized allocations
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def proportional_alloc(counts, max_samples):
    """
    Compute a proportional allocation across classes when minima cannot be met.

    This helper computes a proportional distribution of `max_samples` across
    classes according to their relative counts. It floors fractional values
    to integers and then distributes leftover units by descending fractional
    remainder to ensure the total sums to `max_samples` (subject to class
    availability caps).

    :param counts: pandas Series with per-class counts
    :param max_samples: total maximum samples to allocate
    :return: dict mapping class -> final allocation
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        total_local = int(counts.sum())  # Total samples available across classes
        float_alloc_local = {
            c: (max_samples * int(counts[c]) / total_local) for c in counts.index
        }  # Fractional proportional allocation
        base_alloc_local = {c: int(float_alloc_local[c]) for c in counts.index}  # Base integer allocation
        assigned_local = sum(base_alloc_local.values())  # Sum of base allocations
        leftover_local = max_samples - assigned_local  # Leftover to distribute due to flooring
        remainders_local = sorted(
            counts.index, key=lambda c: (float_alloc_local[c] - base_alloc_local[c]), reverse=True
        )  # Order by fractional remainder

        for c in remainders_local:  # Distribute leftover one-by-one
            if leftover_local <= 0:  # Stop when leftover exhausted
                break  # Exit loop
            if base_alloc_local[c] < int(counts[c]):  # Only increase if class has remaining samples
                base_alloc_local[c] += 1  # Increment base allocation
                leftover_local -= 1  # Decrement leftover

        final_alloc_local = {c: min(int(counts[c]), base_alloc_local[c]) for c in counts.index}  # Cap by class availability

        return final_alloc_local  # Return proportional allocations
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def sample_indices_from_alloc(labels, allocations, random_state):
    """
    Draw indices from `labels` according to `allocations` using `random_state`.

    For each class in `allocations`, this helper selects the requested number
    of indices without replacement (or all available indices if the
    allocation exceeds availability). The selection is reproducible via the
    provided `random_state`.

    :param labels: pandas Series with class labels
    :param allocations: dict mapping class -> number of samples to draw
    :param random_state: integer seed for RNG reproducibility
    :return: list of sampled row indices
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        rng_local = np.random.RandomState(random_state)  # RNG for reproducibility
        sampled_indices_local = []  # Container for sampled indices

        for cls in allocations:  # Iterate classes in allocation order
            cls_idx_local = labels[labels == cls].index.to_list()  # Indices belonging to the class
            k_local = allocations.get(cls, 0)  # Number to sample for this class

            if k_local <= 0:  # Skip when zero allocation
                continue  # Continue to next class

            if k_local >= len(cls_idx_local):  # If allocation exceeds availability
                sampled_local = cls_idx_local  # Take all available indices
            else:  # Otherwise sample without replacement
                sampled_local = list(rng_local.choice(cls_idx_local, size=k_local, replace=False))  # Draw random sample

            sampled_indices_local.extend(sampled_local)  # Append sampled indices

        return sampled_indices_local  # Return list of sampled indices
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def prepare_numeric_dataset(filepath, low_memory=True, sample_size=5000, random_state=42):
    """
    Load CSV dataset, clean it, extract numeric features, optionally downsample,
    and return numeric DataFrame and labels.

    :param filepath: path to CSV file
    :param low_memory: whether to use low-memory mode when loading CSV
    :param sample_size: maximum number of rows to keep (downsampling threshold)
    :param random_state: random seed for reproducibility
    :return: tuple (numeric_df, labels) or (None, None) on failure
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        df = load_dataset(filepath, low_memory=low_memory)  # Load CSV into DataFrame
        if df is None:  # If loading failed
            return None, None  # Abort

        cleaned = preprocess_dataframe(df, remove_zero_variance=False)  # Basic cleaning
        if cleaned is None:  # If cleaning failed
            return None, None  # Abort

        numeric_df = coerce_numeric_columns(cleaned)  # Extract numeric features
        if numeric_df is None:  # If extraction failed
            return None, None  # Abort

        numeric_df = fill_replace_and_drop(numeric_df)  # Clean numeric frame
        if numeric_df is None:  # If cleaning failed
            return None, None  # Abort

        if numeric_df.shape[0] == 0 or numeric_df.shape[1] == 0:  # No numeric data
            return None, None  # Abort

        label_col = detect_label_column(cleaned.columns)  # Detect label column
        labels = cleaned[label_col] if label_col in cleaned.columns else None  # Extract labels if present

        if numeric_df.shape[0] > sample_size:  # Downsample if too many rows
            numeric_df, labels = downsample_with_class_awareness(
                numeric_df, labels, sample_size, random_state
            )  # Class-aware downsampling

        return numeric_df, labels  # Return numeric DataFrame and labels
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def prepare_numeric_dataset_from_df(df, sample_size=5000, random_state=42):
    """
    Prepare numeric DataFrame and labels from an already-loaded DataFrame.

    Mirrors `prepare_numeric_dataset` behaviour but operates on `df` to
    avoid rereading the CSV from disk.
    """
    
    try:
        if df is None:
            return None, None

        cleaned = preprocess_dataframe(df, remove_zero_variance=False)  # Basic cleaning
        if cleaned is None:
            return None, None

        numeric_df = coerce_numeric_columns(cleaned)  # Extract numeric features
        if numeric_df is None:
            return None, None

        numeric_df = fill_replace_and_drop(numeric_df)  # Clean numeric frame
        if numeric_df is None:
            return None, None

        if numeric_df.shape[0] == 0 or numeric_df.shape[1] == 0:
            return None, None

        label_col = detect_label_column(cleaned.columns)  # Detect label column
        labels = cleaned[label_col] if label_col in cleaned.columns else None

        if numeric_df.shape[0] > sample_size:
            numeric_df, labels = downsample_with_class_awareness(
                numeric_df, labels, sample_size, random_state
            )

        return numeric_df, labels
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def scale_features(numeric_df):
    """
    Standardize numeric features to zero mean and unit variance. Fall back to
    converting to float64 array if scaling fails.

    :param numeric_df: DataFrame with numeric features
    :return: Numpy array with scaled features
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Scaling numeric features to zero mean and unit variance.{Style.RESET_ALL}"
        )  # Output the verbose message

        try:  # Try scaling with sklearn StandardScaler
            scaler = StandardScaler()  # Create scaler instance
            X_scaled = scaler.fit_transform(numeric_df.values)  # Fit and transform numeric values
        except Exception:  # Fallback if scaling fails
            X_scaled = np.asarray(numeric_df.values, dtype=np.float64)  # Convert to a float64 numpy array

        return X_scaled  # Return the scaled array
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def allocate_remaining_budget(counts, allocations, remaining_budget):
    """
    Helper function to distribute remaining budget among classes proportionally,
    using the fractional remainder method.

    :param counts: pandas Series of class counts
    :param allocations: dict of current allocations
    :param remaining_budget: number of samples left to allocate
    :return: dict of updated allocations including remaining budget
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        extras = {
            cls: max(0, int(counts[cls]) - allocations.get(cls, 0)) for cls in counts.index
        }  # Extra samples available per class
        total_extras = sum(extras.values())  # Total extra samples available

        if total_extras > 0:  # Only proceed if there are extras to allocate
            float_alloc = {cls: remaining_budget * extras[cls] / total_extras for cls in extras}  # Fractional allocation
            base_alloc = {cls: int(float_alloc[cls]) for cls in float_alloc}  # Base integer allocation

            assigned = sum(base_alloc.values())  # Sum of base allocations
            leftover = remaining_budget - assigned  # Leftover samples to distribute

            remainders = sorted(
                extras.keys(), key=lambda c: (float_alloc[c] - base_alloc[c]), reverse=True
            )  # Order by fractional remainder
            for cls in remainders:  # Distribute leftover samples
                if leftover <= 0:  # Stop when no leftover remains
                    break  # Exit loop
                if base_alloc.get(cls, 0) < extras.get(cls, 0):  # Only allocate if class can accept more
                    base_alloc[cls] = base_alloc.get(cls, 0) + 1  # Increment allocation for this class
                    leftover -= 1  # Decrease leftover count

            for cls in counts.index:  # Finalize allocations applying available caps
                allocations[cls] = min(
                    int(counts[cls]), allocations.get(cls, 0) + base_alloc.get(cls, 0)
                )  # Cap addition by remaining available

        return allocations  # Return updated allocations
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def compute_class_aware_allocations(labels, sample_size, min_class_size=50):
    """
    Compute per-class sample allocations that preserve class distribution while
    ensuring small classes (< min_class_size) are fully included.

    Strategy:
    - Classes with fewer than min_class_size samples: include all samples
    - Remaining budget: distribute proportionally among larger classes
    - Use fractional remainder method to allocate leftover samples fairly

    :param labels: pandas Series containing class labels
    :param sample_size: maximum total samples to allocate
    :param min_class_size: threshold below which all class samples are included
    :return: dict mapping class -> number of samples to select
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        counts = labels.value_counts()  # Get per-class counts

        allocations = {
            cls: min(int(cnt), int(min_class_size)) for cls, cnt in counts.items()
        }  # Initial allocations for small classes

        remaining_budget = int(sample_size - sum(allocations.values()))  # Remaining samples to allocate

        if remaining_budget > 0:  # If there is remaining budget
            allocations = allocate_remaining_budget(counts, allocations, remaining_budget)  # Distribute remaining budget

        total_alloc = sum(allocations.values())  # Total allocated samples
        if total_alloc > sample_size:  # Safety check to reduce overallocation
            sorted_by_alloc = sorted(
                allocations.items(), key=lambda x: x[1], reverse=True
            )  # Sort classes by allocation descending
            i = 0  # Index for iteration
            while total_alloc > sample_size and i < len(sorted_by_alloc):  # While overallocation exists
                cls, cur = sorted_by_alloc[i]  # Current class and its allocation
                reducible = cur - min(
                    int(counts[cls]), int(min_class_size)
                )  # Amount that can be reduced without violating minima
                if reducible > 0:  # If there is reducible allocation
                    remove = min(reducible, total_alloc - sample_size)  # Amount to remove
                    allocations[cls] -= remove  # Reduce allocation
                    total_alloc -= remove  # Update total allocation
                i += 1  # Move to next class

        return allocations  # Return finalized allocations
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def sample_by_class_allocation(labels, allocations, random_state):
    """
    Sample row indices according to per-class allocations.

    For each class, randomly selects the allocated number of samples without
    replacement. If allocation exceeds available samples for a class, all
    samples from that class are included.

    :param labels: pandas Series containing class labels
    :param allocations: dict mapping class -> number of samples to select
    :param random_state: seed for reproducible random sampling
    :return: list of selected row indices
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        selected_idx = []  # Container for selected indices
        rng = np.random.RandomState(random_state)  # RNG for reproducibility

        for cls, k in allocations.items():  # Iterate allocations per class
            idxs = labels[labels == cls].index.to_list()  # All indices for this class
            if k >= len(idxs):  # If allocation >= available samples
                sampled_local = idxs  # Take all available indices
            else:  # Otherwise
                sampled_local = list(
                    rng.choice(idxs, size=k, replace=False)
                )  # Randomly choose k indices without replacement
            selected_idx.extend(sampled_local)  # Append sampled indices to selection

        return selected_idx  # Return the final list of indices
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def downsample_with_class_awareness(numeric_df, labels, sample_size, random_state):
    """
    Downsample dataset while preserving class distribution and ensuring
    small classes (< 50 samples) are fully represented.

    Falls back to random sampling if class-aware sampling fails or labels
    are unavailable.

    :param numeric_df: DataFrame containing numeric features
    :param labels: pandas Series with class labels (or None)
    :param sample_size: target number of samples after downsampling
    :param random_state: seed for reproducible sampling
    :return: tuple (downsampled numeric_df, downsampled labels or None)
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        if labels is None:  # No labels available -> random sampling
            return numeric_df.sample(n=sample_size, random_state=random_state), None  # Return random sample and no labels

        try:  # Try class-aware downsampling
            allocations = compute_class_aware_allocations(
                labels, sample_size, min_class_size=50
            )  # Compute per-class allocations
            selected_idx = sample_by_class_allocation(labels, allocations, random_state)  # Sample indices per allocations

            if not selected_idx:  # If selection failed or empty
                return numeric_df.sample(n=sample_size, random_state=random_state), None  # Fallback to random sampling

            return (
                numeric_df.loc[selected_idx].reset_index(drop=True),
                labels.loc[selected_idx].reset_index(drop=True),
            )  # Return downsampled DataFrame and labels
        except Exception:  # On any error, fallback to random sampling
            return numeric_df.sample(n=sample_size, random_state=random_state), None  # Return random sample and no labels
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def initialize_and_fit_tsne(X, n_components=2, perplexity=30, n_iter=1000, random_state=42):
    """
    Initialize t-SNE with proper parameters and compute 2D or 3D embedding.

    Handles compatibility with different TSNE versions by inspecting the constructor
    signature and setting 'n_iter' or 'max_iter' accordingly.

    :param X: numpy array of scaled numeric features
    :param n_components: number of dimensions for embedding (2 or 3)
    :param perplexity: t-SNE perplexity parameter
    :param n_iter: number of t-SNE optimization iterations
    :param random_state: random seed for reproducibility
    :return: numpy array of t-SNE embeddings (n_samples, n_components)
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        try:  # Inspect TSNE init signature for compatibility
            sig = signature(TSNE.__init__).parameters  # Get TSNE init signature
        except Exception:  # If inspection fails
            sig = {}  # Fallback to empty signature

        tsne_kwargs = {
            "n_components": n_components,
            "perplexity": perplexity,
            "random_state": random_state,
            "init": "pca",
        }  # Base t-SNE args
        if "n_iter" in sig:  # Check for n_iter parameter
            tsne_kwargs["n_iter"] = n_iter  # Set n_iter if supported
        elif "max_iter" in sig:  # Check for max_iter parameter
            tsne_kwargs["max_iter"] = n_iter  # Set max_iter if supported
        else:  # Neither parameter supported
            tsne_kwargs["max_iter"] = n_iter  # Default to max_iter

        tsne = TSNE(**tsne_kwargs)  # Initialize t-SNE with compatible args
        X_emb = tsne.fit_transform(X)  # Compute embedding
        return X_emb  # Return the embedding
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def save_tsne_plot(X_emb, labels, output_path, title):
    """
    Create and save a 2D t-SNE scatter plot.

    If labels are provided, points are colored by class with a legend.
    Otherwise, all points are plotted uniformly.

    :param X_emb: 2D numpy array of t-SNE embeddings (shape: [n_samples, 2])
    :param labels: pandas Series of class labels or None
    :param output_path: absolute path where PNG will be saved
    :param title: plot title string
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        plt.figure(figsize=(8, 6))  # Create matplotlib figure

        if labels is not None:  # Plot colored by class
            labels_ser = pd.Series(labels)  # Ensure labels are a pandas Series
            counts = labels_ser.value_counts()  # Count samples per class
            unique = list(labels_ser.unique())  # Unique class labels (preserve order)
            for cls in unique:  # Plot each class separately
                mask = labels_ser == cls  # Boolean mask for class
                plt.scatter(
                    X_emb[mask, 0], X_emb[mask, 1], label=f"{cls} ({int(counts.get(cls, 0))})", s=8
                )  # Scatter plot for class with count in label
            plt.legend(markerscale=2, fontsize="small")  # Add legend for classes
            try:  # Try to add counts text box
                counts_text = "\n".join([f"{str(c)}: {int(counts[c])}" for c in counts.index])  # Prepare counts text
                plt.gcf().text(
                    0.99,
                    0.01,
                    counts_text,
                    ha="right",
                    va="bottom",
                    fontsize=8,
                    bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
                )  # Add text box with counts
            except Exception:  # Ignore any errors in adding counts text
                pass  # Do nothing
        else:  # No labels provided
            plt.scatter(X_emb[:, 0], X_emb[:, 1], s=8)  # Plot all points uniformly

        plt.title(title)  # Set plot title
        plt.xlabel("t-SNE 1")  # X-axis label
        plt.ylabel("t-SNE 2")  # Y-axis label
        plt.tight_layout()  # Adjust layout
        plt.savefig(output_path, dpi=150)  # Save figure to disk
        plt.close()  # Close figure to free memory
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def save_tsne_3d_plot(X_emb, labels, output_path, title):
    """
    Create and save a 3D t-SNE scatter plot.

    If labels are provided, points are colored by class with a legend.
    Otherwise, all points are plotted uniformly.

    :param X_emb: 3D numpy array of t-SNE embeddings (shape: [n_samples, 3])
    :param labels: pandas Series of class labels or None
    :param output_path: absolute path where PNG will be saved
    :param title: plot title string
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        fig = plt.figure(figsize=(10, 8))  # Create matplotlib figure
        ax = fig.add_subplot(111, projection='3d')  # Create 3D axis

        if labels is not None:  # Plot colored by class
            labels_ser = pd.Series(labels)  # Ensure labels are a pandas Series
            counts = labels_ser.value_counts()  # Count samples per class
            unique = list(labels_ser.unique())  # Unique class labels (preserve order)
            for cls in unique:  # Plot each class separately
                mask = labels_ser == cls  # Boolean mask for class
                cast(Any, ax).scatter(
                    X_emb[mask, 0],
                    X_emb[mask, 1],
                    X_emb[mask, 2],
                    label=f"{cls} ({int(counts.get(cls, 0))})",
                    s=8,
                )  # 3D scatter plot for class with count in label
            ax.legend(markerscale=2, fontsize="small")  # Add legend for classes
        else:  # No labels provided
            cast(Any, ax).scatter(X_emb[:, 0], X_emb[:, 1], X_emb[:, 2], s=8)  # Plot all points uniformly

        ax.set_title(title)  # Set plot title
        ax.set_xlabel("t-SNE 1")  # X-axis label
        ax.set_ylabel("t-SNE 2")  # Y-axis label
        cast(Any, ax).set_zlabel("t-SNE 3")  # Z-axis label (cast to Any for typing)
        plt.tight_layout()  # Adjust layout
        plt.savefig(output_path, dpi=150)  # Save figure to disk
        plt.close()  # Close figure to free memory
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def generate_tsne_plot(
    filepath,
    df=None,
    low_memory=True,
    sample_size=5000,
    perplexity=30,
    n_iter=1000,
    random_state=42,
    output_dir=None,
    config: dict | None = None,
):
    """
    Generate and save both 2D and 3D t-SNE visualizations of a CSV dataset.

    This function loads a dataset, extracts numeric features, performs class-aware
    downsampling (if needed), computes 2D and 3D t-SNE embeddings, and saves both
    as PNG scatter plots with per-class coloring (if labels are detected).

    Downsampling strategy:
    - Classes with < 50 samples: all samples included
    - Larger classes: sampled proportionally to preserve distribution
    - Falls back to random sampling if class detection fails

    :param filepath: path to CSV file
    :param low_memory: whether to use low-memory mode when loading CSV
    :param sample_size: maximum number of rows to embed (reduces computation time)
    :param perplexity: t-SNE perplexity parameter (typically 5-50)
    :param n_iter: number of t-SNE optimization iterations
    :param random_state: random seed for reproducible results
    :param output_dir: directory for saving PNGs (defaults to dataset's RESULTS_DIR)
    :return: tuple of (2D_filename, 3D_filename) or (None, None) on failure
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Generating t-SNE plots (2D and 3D) for: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}"
        )  # Output start message for t-SNE generation

        try:  # Main try-catch block for overall failure handling
            if df is not None:
                numeric_df, labels = prepare_numeric_dataset_from_df(
                    df, sample_size=sample_size, random_state=random_state
                )
            else:
                numeric_df, labels = prepare_numeric_dataset(
                    filepath, low_memory, sample_size, random_state
                )  # Prepare numeric dataset
            if numeric_df is None:  # Abort if preparation failed
                return None, None  # Indicate failure

            X = scale_features(numeric_df)  # Scale features for t-SNE

            n_rows = X.shape[0]  # Number of rows after downsampling
            if n_rows <= max(3, int(perplexity) + 1):  # Check t-SNE feasibility
                return None, None  # Abort if too few samples for t-SNE

            if output_dir is None:  # Determine output directory
                cfg = config or get_default_config()
                tsne_subdir = cfg.get("paths", {}).get("data_separability_subdir", "Data_Separability")
                output_dir = os.path.join(os.path.dirname(os.path.abspath(filepath)), tsne_subdir)
            os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

            base = os.path.splitext(os.path.basename(filepath))[0]  # Base filename
            
            verbose_output(
                f"{BackgroundColors.GREEN}Computing 2D t-SNE embedding for: {BackgroundColors.CYAN}{base}{Style.RESET_ALL}"
            )  # Output 2D t-SNE message
            X_emb_2d = initialize_and_fit_tsne(X, n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)  # Compute 2D t-SNE embedding
            out_name_2d = f"TSNE_2D_{base}.png"  # 2D output PNG name
            out_path_2d = os.path.join(output_dir, out_name_2d)  # 2D absolute path
            save_tsne_plot(X_emb_2d, labels, out_path_2d, f"t-SNE 2D: {base}")  # Create and save 2D plot
            verbose_output(
                f"{BackgroundColors.GREEN}Saved 2D t-SNE plot to: {BackgroundColors.CYAN}{out_path_2d}{Style.RESET_ALL}"
            )  # Output success message
            
            verbose_output(
                f"{BackgroundColors.GREEN}Computing 3D t-SNE embedding for: {BackgroundColors.CYAN}{base}{Style.RESET_ALL}"
            )  # Output 3D t-SNE message
            X_emb_3d = initialize_and_fit_tsne(X, n_components=3, perplexity=perplexity, n_iter=n_iter, random_state=random_state)  # Compute 3D t-SNE embedding
            out_name_3d = f"TSNE_3D_{base}.png"  # 3D output PNG name
            out_path_3d = os.path.join(output_dir, out_name_3d)  # 3D absolute path
            save_tsne_3d_plot(X_emb_3d, labels, out_path_3d, f"t-SNE 3D: {base}")  # Create and save 3D plot
            verbose_output(
                f"{BackgroundColors.GREEN}Saved 3D t-SNE plot to: {BackgroundColors.CYAN}{out_path_3d}{Style.RESET_ALL}"
            )  # Output success message

            try:  # Try to delete DataFrame to free memory
                del numeric_df  # Free numeric DataFrame
            except Exception:  # Ignore any exceptions during deletion
                pass  # Do nothing
            gc.collect()  # Force garbage collection

            return out_name_2d, out_name_3d  # Return both saved filenames
        except Exception as e:  # Catch-all failure
            print(f"{BackgroundColors.RED}t-SNE generation failed for {filepath}: {e}{Style.RESET_ALL}")  # Verbose error
            return None, None  # Indicate failure
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def get_augmented_sample_count(original_csv_path, config=None) -> int:
    """
    Determine the total number of augmented samples produced for a given original
    CSV by inspecting the WGANGP output directory for that dataset.

    Detection logic:
    - Use `config` if provided; otherwise try to load `config.yaml` next to this file.
    - The augmented file is expected under: <original_parent>/Data_Augmentation/
        with filename: <stem>{results_suffix}{suffix} where `results_suffix` defaults
        to `_data_augmented` (per wgangp defaults).
    - If the exact-stem file exists and is a CSV, read and return its row count.
    - If not found, return 0 (no augmentation for this dataset).

    Raises RuntimeError if the augmented CSV exists but cannot be read (corrupted).
    """
    
    try:
        p = Path(original_csv_path)

        cfg = {}
        if config and isinstance(config, dict):
            cfg = config
        else:
            cfg_path = Path(__file__).parent / "config.yaml"
            if cfg_path.exists():
                try:
                    with open(cfg_path, "r", encoding="utf-8") as _f:
                        cfg = yaml.safe_load(_f) or {}
                except Exception:
                    cfg = {}

        data_aug_subdir = cfg.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation")
        results_suffix = cfg.get("execution", {}).get("results_suffix", "_data_augmented")

        data_aug_dir = p.parent / data_aug_subdir

        candidate = data_aug_dir / f"{p.stem}{results_suffix}{p.suffix}"

        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(candidate)
                return int(len(df)) if len(df) > 0 else 0
            except Exception as e:
                raise RuntimeError(f"Failed to read augmented CSV '{candidate}': {e}")

        return 0
    except Exception:
        raise


def get_dataset_file_info(filepath, df=None, low_memory=True):
    """
    Extracts dataset information from a CSV file and returns it as a dictionary.

    :param filepath: Path to the CSV file
    :param low_memory: Whether to use low memory mode when loading the CSV (default: True)
    :return: Dictionary containing dataset information
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Extracting dataset information from: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}"
        )  # Output start message for dataset info extraction

        if df is None:
            df = load_dataset(filepath, low_memory)  # Load the dataset

        if df is None:  # If the dataset could not be loaded
            return None  # Return None

        original_num_rows = len(df)  # Capture original number of rows immediately after read
        original_num_features = df.shape[1] if hasattr(df, "shape") else 0  # Capture original feature count

        cleaned_df = preprocess_dataframe(df)  # Preprocess the DataFrame

        rows_after_preprocessing = len(cleaned_df)  # Capture rows after preprocessing
        features_after_preprocessing = cleaned_df.shape[1] if hasattr(cleaned_df, "shape") else 0  # Capture features after preprocessing

        label_col = detect_label_column(cleaned_df.columns)  # Try to detect the label column
        n_samples, n_features, n_numeric, n_int, n_categorical, n_other, categorical_cols_str = summarize_features(
            cleaned_df
        )  # Summarize features
        missing_summary = summarize_missing_values(cleaned_df)  # Summarize missing values
        classes_str, class_dist_str = summarize_classes(cleaned_df, label_col)  # Summarize classes and distributions

        try:  # Try to get file size in GB
            size_bytes = os.path.getsize(filepath)  # Get file size in bytes
            size_gb = size_bytes / (1024**3)  # Convert bytes to gigabytes
            size_gb_str = f"{size_gb:.3f}"  # Format size to 3 decimal places
        except Exception:  # If an error occurs
            size_gb_str = "N/A"  # Set size to N/A if error occurs

        result = {  # Return the dataset information as a dictionary
            "Dataset Name": os.path.basename(filepath),
            "Size (GB)": size_gb_str,
            "Number of Samples": f"{n_samples:,}",  # Format with commas for readability
            "Number of Features": f"{n_features:,}",  # Format with commas for readability
            "original_num_rows": original_num_rows,  # Rows immediately after reading CSV
            "rows_after_preprocessing": rows_after_preprocessing,  # Rows after preprocessing
            "original_num_features": original_num_features,  # Features before preprocessing
            "features_after_preprocessing": features_after_preprocessing,  # Features after preprocessing
            "Feature Types": f"{n_numeric} numeric (float64), {n_int} integer (int64), {n_categorical} categorical (object/category/bool/string), {n_other} other",
            "Categorical Features (object/string)": categorical_cols_str,
            "Missing Values": missing_summary,
            "Classes": classes_str,
            "Class Distribution": class_dist_str,
        }

        try:
            aug_count = get_augmented_sample_count(filepath, None)
        except Exception:
            raise
        result["data_augmentation_samples"] = int(aug_count)

        return result  # Return the dataset information
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def get_file_common_and_extras(headers_map, filepath, common_features):
    """
    Return the sorted common features list and extra columns for a specific file, using normalized feature names (lowercase + strip).

    :param headers_map: dict mapping filepath -> list of column names
    :param filepath: path for which to compute extras
    :param common_features: set of features present in all files
    :return: tuple (common_list, extras_list)
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        file_cols = headers_map.get(filepath, [])  # Get headers for this file

        if file_cols is not None:  # Normalize file columns
            normalized_file_cols = set(col.strip().lower() for col in file_cols)  # Normalize file columns
            normalized_common = set(col.strip().lower() for col in common_features)  # Normalize common features
            extras = sorted(normalized_file_cols - normalized_common)  # Compute non-common extras
        else:  # If no columns found for this file
            extras = []  # No extras

        common_list = (
            sorted(col.strip().lower() for col in common_features) if common_features else []
        )  # Sorted normalized shared features

        return common_list, extras  # Return common + extras lists
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def write_report(report_rows, base_dir, output_filename, config: dict | None = None):
    """
    Writes the report rows to a CSV file.

    :param report_rows: List of dictionaries containing report data
    :param base_dir: Base directory for saving the report
    :param output_filename: Name of the output CSV file
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        report_df = pd.DataFrame(report_rows)  # Create a DataFrame from the report rows

        if "#" in report_df.columns:  # If the "#"" column exists
            cols = ["#"] + [c for c in report_df.columns if c != "#"]  # Move "#" to the front
            report_df = report_df[cols]  # Reorder columns

        cfg = config or get_default_config()
        results_subdir = cfg.get("paths", {}).get("dataset_description_subdir", "Dataset_Description")
        results_dir = os.path.join(base_dir, results_subdir)
        os.makedirs(results_dir, exist_ok=True)
        report_csv_path = os.path.join(results_dir, output_filename)
        generate_csv_and_image(report_df, report_csv_path, config=cfg)
        pass  # No-op here; preprocessing summary handling is performed by the caller to avoid globals
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def collect_preprocessing_metrics(filepath, original_num_rows, rows_after_preprocessing, original_num_features, features_after_preprocessing):
    """
    Collect preprocessing metrics for a single file and return a dict matching the required CSV schema.

    :param filepath: Path to the processed CSV file
    :param original_num_rows: Number of rows immediately after reading the CSV
    :param rows_after_preprocessing: Number of rows after preprocessing steps
    :param original_num_features: Number of features before preprocessing
    :param features_after_preprocessing: Number of features after preprocessing
    :return: Dict with keys matching preprocessing_summary.csv columns
    """

    try:  # Wrap logic to preserve existing error handling conventions
        filename = os.path.basename(filepath)  # Extract filename from filepath
        removed_rows = original_num_rows - rows_after_preprocessing  # Compute removed rows count
        removed_rows = removed_rows if removed_rows >= 0 else 0  # Clamp negative to zero for safety
        if original_num_rows > 0:  # If original rows non-zero compute proportion
            removed_rows_proportion = round(removed_rows / float(original_num_rows), 6)  # Compute proportion and round
        else:  # Avoid division by zero when original is zero
            removed_rows_proportion = 0.0  # Set proportion to 0.0 per spec

        removed_features = original_num_features - features_after_preprocessing  # Compute removed features count
        removed_features = removed_features if removed_features >= 0 else 0  # Clamp negative to zero for safety
        if original_num_features > 0:  # If original features non-zero compute proportion
            removed_features_proportion = round(removed_features / float(original_num_features), 6)  # Compute proportion and round
        else:  # Avoid division by zero when original is zero
            removed_features_proportion = 0.0  # Set proportion to 0.0 per spec

        return {  # Return metrics dict matching required output columns and order
            "filename": filename,  # Base filename
            "original_num_rows": int(original_num_rows),  # Cast to int for CSV
            "rows_after_preprocessing": int(rows_after_preprocessing),  # Cast to int
            "removed_rows": int(removed_rows),  # Cast to int
            "removed_rows_proportion": float(removed_rows_proportion),  # Float rounded to 6 decimals
            "original_num_features": int(original_num_features),  # Cast to int
            "features_after_preprocessing": int(features_after_preprocessing),  # Cast to int
            "removed_features": int(removed_features),  # Cast to int
            "removed_features_proportion": float(removed_features_proportion),  # Float rounded to 6 decimals
        }  # End dict
    except Exception as e:  # Preserve exception handling style
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def build_preprocessing_summary_dataframe(metrics_list):
    """
    Build a DataFrame for preprocessing summary from a list of metrics dicts.

    :param metrics_list: List of dicts produced by `collect_preprocessing_metrics`
    :return: pandas.DataFrame with fixed column order
    """

    try:  # Wrap function body for consistency with module style
        cols = [
            "filename",
            "original_num_rows",
            "rows_after_preprocessing",
            "removed_rows",
            "removed_rows_proportion",
            "original_num_features",
            "features_after_preprocessing",
            "removed_features",
            "removed_features_proportion",
        ]  # Define exact column order required by spec

        df = pd.DataFrame(metrics_list)  # Create DataFrame from provided metrics list
        for c in cols:  # Ensure all expected columns exist in DataFrame
            if c not in df.columns:  # If missing column
                df[c] = None  # Add column filled with None to preserve schema
        df = df[cols]  # Reorder columns to the required fixed order
        return df  # Return the prepared DataFrame
    except Exception as e:  # Preserve exception handling
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def save_preprocessing_summary_csv(df, base_dir, filename="preprocessing_summary.csv", config: dict | None = None):
    """
    Save the preprocessing summary DataFrame to the results directory for the given base_dir.

    :param df: DataFrame produced by `build_preprocessing_summary_dataframe`
    :param base_dir: Base directory where dataset results are stored
    :param filename: Output CSV filename (default: preprocessing_summary.csv)
    :return: Absolute path to saved CSV file
    """

    try:  # Wrap function body for robust error reporting per module conventions
        cfg = config or get_default_config()
        results_subdir = cfg.get("paths", {}).get("dataset_description_subdir", "Dataset_Description")
        results_dir = os.path.join(base_dir, results_subdir)
        if not os.path.exists(results_dir):
            os.makedirs(results_dir, exist_ok=True)
        out_path = os.path.join(results_dir, filename)
        generate_csv_and_image(df, out_path, config=cfg)
        return out_path
    except Exception as e:  # Preserve exception handling style
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def print_preprocessing_summary_table(df):
    """
    Print a formatted table of the preprocessing summary DataFrame to the terminal.

    :param df: DataFrame in the exact schema produced by `build_preprocessing_summary_dataframe`
    :return: None
    """

    try:  # Wrap printing to preserve module error handling conventions
        if df is None or df.empty:  # If DataFrame is empty or None
            print(f"{BackgroundColors.YELLOW}No preprocessing summary to display.{Style.RESET_ALL}")  # Inform the user
            return  # Nothing to print

        cols = [
            "filename",
            "original_num_rows",
            "rows_after_preprocessing",
            "removed_rows",
            "removed_rows_proportion",
            "original_num_features",
            "features_after_preprocessing",
            "removed_features",
            "removed_features_proportion",
        ]  # Column order for printing

        col_widths = {}  # Prepare dict to hold widths
        for c in cols:  # For each column compute width
            header_w = len(c)  # Header width
            max_data_w = max([len(str(x)) for x in df[c].fillna("")]) if c in df.columns and not df[c].isnull().all() else 0  # Max width of data
            col_widths[c] = max(header_w, max_data_w)  # Choose the max

        header_parts = []  # Parts for header
        for c in cols:  # For each column append formatted header
            header_parts.append(c.ljust(col_widths[c]))  # Left-justify header text
        header_line = " | ".join(header_parts)  # Join header parts with separators
        sep_line = "-" * len(header_line)  # Separator line of matching length

        print(header_line)  # Print header
        print(sep_line)  # Print separator

        for _, row in df.iterrows():  # Iterate DataFrame rows
            parts = []  # Parts for this row
            for c in cols:  # For each column format the cell
                val = row.get(c, "")  # Get value with fallback
                parts.append(str(val).ljust(col_widths[c]))  # Left-justify cell text
            print(" | ".join(parts))  # Print joined row
    except Exception as e:  # Preserve exception handling
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def apply_zebra_style(df):
    """
    Apply zebra-striping pandas Styler to the provided DataFrame.

    :param df: pandas.DataFrame to style
    :return: pandas.Styler with zebra styling applied
    """

    try:  # Wrap function body for consistent error handling
        def _stripe(row):  # Small helper to produce row-wise styles
            return [
                "background-color: #ffffff" if row.name % 2 == 0 else "background-color: #f2f2f2"
                for _ in row
            ]  # Return alternating colors per column in the row

        styled = df.style.apply(_stripe, axis=1)  # Apply zebra striping across rows
        return styled  # Return the styled DataFrame
    except Exception as e:  # Preserve exception handling style
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise exception to surface failure


def export_dataframe_image(styled_df, output_path):
    """
    Export a pandas.Styler to a PNG image using dataframe_image.

    :param styled_df: pandas.Styler object to export
    :param output_path: Path to write PNG image
    :return: None
    """

    try:  # Wrap to ensure exceptions bubble with logging consistent with module
        dfi.export(styled_df, output_path)  # Export styled DataFrame to PNG using dataframe_image
    except Exception as e:  # If export fails, do not swallow the error
        print(str(e))  # Print export error for logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to indicate failure


def generate_table_image_from_dataframe(df, output_path, config: dict | None = None):
    """
    Generate a zebra-striped PNG table image from a DataFrame and save to output_path.

    :param df: pandas.DataFrame to render
    :param output_path: Path for output PNG image
    :return: None
    """

    try:  # Wrap to preserve module's error handling conventions
        styled = apply_zebra_style(df)  # Create a styled DataFrame with zebra striping
        export_dataframe_image(styled, output_path)  # Export the styled DataFrame to PNG
    except Exception:  # Do not swallow exceptions here per spec
        raise  # Re-raise any exception to caller


def generate_csv_and_image(df, csv_path, config: dict | None = None):
    """
    Save a DataFrame to CSV and generate a corresponding PNG table image next to it.

    :param df: pandas.DataFrame to save and render
    :param csv_path: Full path for CSV output
    :return: Tuple (csv_path, image_path)
    """

    try:  # Wrap whole function for consistent error handling
        # Validate inputs
        if not isinstance(csv_path, str) or not csv_path:
            raise ValueError("csv_path must be a non-empty string")
        df.to_csv(csv_path, index=False)  # Persist DataFrame to CSV without index
        img_ext = (config or {}).get("dataset_descriptor", {}).get("table_image_format", "png")
        image_path = os.path.splitext(csv_path)[0] + f".{img_ext}"  # Build image path using configured format
        generate_table_image_from_dataframe(df, image_path, config=config)  # Generate image from DataFrame
        return csv_path, image_path  # Return both paths for caller use
    except Exception:
        raise


def generate_dataset_report(input_path, file_extension=".csv", low_memory=True, output_filename: str | None = None, config: dict | None = None):
    """
    Generates a CSV report for the specified input path.
    The Dataset Name column will include subdirectories if present.

    :param input_path: Directory or file path containing the dataset
    :param file_extension: File extension to filter (default: .csv)
    :param low_memory: Whether to use low memory mode when loading CSVs (default: True)
    :param output_filename: Name of the CSV file to save the report
    :return: True if the report was generated successfully, False otherwise
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        report_rows = []  # List to store report rows
        sorted_matching_files = []  # List to store matching files
        preprocessing_metrics = []  # List to collect per-file preprocessing metric dicts

        if os.path.isdir(input_path):  # If the input path is a directory
            print(
                f"{BackgroundColors.GREEN}Scanning directory {BackgroundColors.CYAN}{input_path}{BackgroundColors.GREEN} for {BackgroundColors.CYAN}{file_extension}{BackgroundColors.GREEN} files...{Style.RESET_ALL}"
            )  # Output scanning message
            sorted_matching_files = collect_matching_files(input_path, file_extension, config=config)  # Collect matching files
            base_dir = os.path.abspath(input_path)  # Get the absolute path of the base directory
        elif os.path.isfile(input_path) and input_path.endswith(file_extension):  # If the input path is a file
            print(
                f"{BackgroundColors.GREEN}Processing single file...{Style.RESET_ALL}"
            )  # Output processing single file message
            sorted_matching_files = [input_path]  # Only process this single file
            base_dir = os.path.dirname(os.path.abspath(input_path))  # Get the base directory of the file
        else:  # If the input path is neither a directory nor a valid file
            print(
                f"{BackgroundColors.RED}Input path is neither a directory nor a valid {file_extension} file: {input_path}{Style.RESET_ALL}"
            )  # Output the error message
            sorted_matching_files = []  # No files to process
            base_dir = os.path.abspath(input_path)  # Just use the input path as base_dir for error message

        cfg = config or get_default_config()

        if not sorted_matching_files:  # If no matching files were found
            print(f"{BackgroundColors.RED}No matching {file_extension} files found in: {input_path}{Style.RESET_ALL}")
            return False  # Exit the function

        cfg = config or get_default_config()
        if output_filename is None:
            output_filename = cfg.get("dataset_descriptor", {}).get("csv_output_suffix", "_description")

        file_dfs = {}  # filepath -> DataFrame (loaded once)
        headers_map = {}  # filepath -> list(columns)
        for fp in sorted_matching_files:
            df = load_dataset(fp, low_memory)
            if df is None:
                print(f"{BackgroundColors.YELLOW}Warning: failed to load {fp}; skipping.{Style.RESET_ALL}")
                continue
            file_dfs[fp] = df
            headers_map[fp] = list(df.columns)

        common_features, headers_match_all = compute_common_features(headers_map)

        progress = tqdm(
            sorted_matching_files,
            desc=f"{BackgroundColors.GREEN}Processing files{BackgroundColors.GREEN}",
            unit="file",
            ncols=100,
        )  # Create a progress bar with fixed width
        for idx, filepath in enumerate(progress, 1):  # Process each matching file
            file_basename = os.path.basename(filepath)  # Get the base filename
            progress.set_description(
                f"{BackgroundColors.GREEN}Processing file {BackgroundColors.CYAN}{idx}{BackgroundColors.GREEN}/{BackgroundColors.CYAN}{len(sorted_matching_files)}{BackgroundColors.GREEN}: {BackgroundColors.CYAN}{file_basename[:30]}{BackgroundColors.GREEN}"
            )  # Update progress bar description (truncate long names)
            info = get_dataset_file_info(filepath, df=file_dfs.get(filepath), low_memory=low_memory)  # Get dataset info (reuse in-memory DF)
            if info:  # If info was successfully retrieved
                relative_path = os.path.relpath(filepath, base_dir)  # Get path relative to base_dir
                info["Dataset Name"] = relative_path.replace(
                    "\\", "/"
                )  # Use relative path for Dataset Name and normalize slashes

                common_list, extras = get_file_common_and_extras(
                    headers_map, filepath, common_features
                )  # Get common and extra features for this file

                info["Headers Match All Files"] = (
                    "Yes" if headers_match_all else "No"
                )  # Indicate if headers match all files
                info["Common Features (in all files)"] = (
                    ", ".join(common_list) if common_list else "None"
                )  # Join common features into a string
                info["Extra Features (not in all files)"] = (
                    ", ".join(extras) if extras else "None"
                )  # Join extra features into a string

                tsne_out_subdir = cfg.get("paths", {}).get("data_separability_subdir", "Data_Separability")
                tsne_file = generate_tsne_plot(
                    filepath,
                    df=file_dfs.get(filepath),
                    low_memory=low_memory,
                    sample_size=2000,
                    output_dir=os.path.join(os.path.dirname(os.path.abspath(filepath)), tsne_out_subdir),
                    config=cfg,
                )  # Generate t-SNE plot (uses in-memory DF when available)
                info["t-SNE Plot"] = tsne_file if tsne_file else "None"  # Add t-SNE plot filename or "None"

                report_rows.append(info)  # Add the info to the report rows
                try:  # Collect preprocessing metrics for this file when available
                    metrics_row = collect_preprocessing_metrics(
                        filepath,  # File path being processed
                        info.get("original_num_rows", 0),  # Original rows captured earlier
                        info.get("rows_after_preprocessing", 0),  # Rows after preprocessing captured earlier
                        info.get("original_num_features", 0),  # Original features captured earlier
                        info.get("features_after_preprocessing", 0),  # Features after preprocessing captured earlier
                    )  # Create metrics row dict
                    preprocessing_metrics.append(metrics_row)  # Append metrics row to list for this directory
                except Exception as _pm:  # If metrics collection fails
                    print(f"{BackgroundColors.YELLOW}Warning: failed to collect preprocessing metrics for {file_basename}: {_pm}{Style.RESET_ALL}")  # Warn but continue

        if report_rows:  # If there are report rows to write
            for i, row in enumerate(report_rows, start=1):  # For each report row
                row["#"] = i  # Add the counter value

            write_report(report_rows, base_dir, output_filename, config=config)
            try:  # After writing main report, handle preprocessing summary generation if metrics collected
                if preprocessing_metrics:  # Only proceed when metrics were collected
                    pre_df = build_preprocessing_summary_dataframe(preprocessing_metrics)  # Build DataFrame from metrics list
                    out_path = save_preprocessing_summary_csv(pre_df, base_dir, config=config)  # Save CSV to results dir and get path
                    print(f"{BackgroundColors.GREEN}Saved preprocessing summary to {BackgroundColors.CYAN}{out_path}{Style.RESET_ALL}")  # Inform user of saved path
                    if os.environ.get("DD_DESCRIPTOR_VERBOSE", "False").lower() in ("1", "true", "yes"):
                        print_preprocessing_summary_table(pre_df)
            except Exception as _ps:  # If summary generation fails, warn but do not abort
                print(f"{BackgroundColors.YELLOW}Warning: failed to generate preprocessing summary: {_ps}{Style.RESET_ALL}")  # Warn and continue
            return True  # Return True indicating success
        else:  # If no report rows were generated
            return False  # Return False indicating failure
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def collect_group_files(paths, file_extension=".csv", config: dict | None = None):
    """
    Collect all matching files for a group of paths.

    :param paths: List of file or directory paths to search
    :param file_extension: File extension to filter (default: ".csv")
    :return: Sorted list of unique file paths
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Collecting {file_extension} files from specified paths...{Style.RESET_ALL}"
        )  # Output collection message

        files = []  # Initialize collection list

        for p in paths:  # Iterate over each path
            if os.path.isdir(p):  # If path is a directory
                files.extend(collect_matching_files(p, file_extension, config=config))  # Collect matching files
            elif os.path.isfile(p) and p.endswith(file_extension):  # If path is a file with correct extension
                files.append(p)  # Add file to list

        return sorted(set(files))  # Remove duplicates and sort
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def compute_group_features(files, low_memory=True):
    """
    Compute common and union features for a list of dataset files.

    :param files: List of dataset file paths
    :param low_memory: Whether to optimize memory when reading CSV headers
    :return: Tuple (common_features_set, union_features_set)
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Computing common and union features for dataset group...{Style.RESET_ALL}"
        )  # Output computation message

        if not files:  # No files, return empty sets
            return set(), set()  # Return empty sets

        headers_map = build_headers_map(files, low_memory=low_memory)  # Build headers map
        common_features, _ = compute_common_features(headers_map)  # Compute common features

        union_features = set()  # Initialize union set
        for cols in headers_map.values():  # Iterate over each file's columns
            if cols:  # If columns exist
                union_features.update(
                    [c.strip().lower() for c in cols]
                )  # Normalize features: strip whitespace and lowercase
        common_features = set(
            [c.strip().lower() for c in common_features]
        )  # Normalize common features: strip whitespace and lowercase

        return set(common_features), union_features  # Return both sets
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def generate_pairwise_report(group_info):
    """
    Generate pairwise comparison rows from group info.

    :param group_info: Dict mapping group_name -> {"files": [...], "common": set(), "union": set()}
    :return: List of dictionaries representing pairwise comparison rows
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        rows = []  # Initialize report row list
        group_names = list(group_info.keys())  # List of group names

        for i in range(len(group_names)):  # Iterate over first group
            for j in range(i + 1, len(group_names)):  # Iterate over second group avoiding duplicates
                a_name, b_name = group_names[i], group_names[j]  # Group names
                a_info, b_info = group_info[a_name], group_info[b_name]  # Group info

                if not a_info["files"] and not b_info["files"]:  # Skip if both have no files
                    continue  # Proceed to next pair

                common_between = sorted(a_info["union"] & b_info["union"])  # Features common to both groups
                extras_a = sorted(a_info["union"] - b_info["union"])  # Features in A not in B
                extras_b = sorted(b_info["union"] - a_info["union"])  # Features in B not in A

                row = {  # Construct row dictionary
                    "Dataset A": a_name,  # First dataset group name
                    "Dataset B": b_name,  # Second dataset group name
                    "Files in A": len(a_info["files"]),  # Number of files in A
                    "Files in B": len(b_info["files"]),  # Number of files in B
                    "Common Features (A ∩ B)": ", ".join(common_between) or "None",  # Common features between A and B
                    "Extra Features in A (A \\ B)": ", ".join(extras_a) or "None",  # Extra features in A
                    "Extra Features in B (B \\ A)": ", ".join(extras_b) or "None",  # Extra features in B
                }

                rows.append(row)  # Append to report rows

        return rows  # Return the list of report rows
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def adjust_rows_for_group(report_rows, group_name):
    """
    Adjust pairwise rows so that the target group always appears as Dataset A.

    :param report_rows: List of dictionaries representing pairwise report rows
    :param group_name: Target group to appear as Dataset A
    :return: List of adjusted report rows
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        adjusted = []  # Initialize adjusted row list

        for row in report_rows:  # Iterate over existing report rows
            if row["Dataset A"] == group_name:  # Already Dataset A
                adjusted.append(dict(row))  # Keep as-is
            elif row["Dataset B"] == group_name:  # Swap A <-> B
                swapped = {  # Construct swapped row
                    "Dataset A": row["Dataset B"],  # Swap Dataset A
                    "Dataset B": row["Dataset A"],  # Swap Dataset B
                    "Files in A": row["Files in B"],  # Swap file counts
                    "Files in B": row["Files in A"],  # Swap file counts
                    "Common Features (A ∩ B)": row["Common Features (A ∩ B)"],  # Keep common features
                    "Extra Features in A (A \\ B)": row["Extra Features in B (B \\ A)"],  # Swap extra features
                    "Extra Features in B (B \\ A)": row["Extra Features in A (A \\ B)"],  # Swap extra features
                }
                adjusted.append(swapped)  # Append swapped row
            else:  # Unrelated row, keep as-is
                adjusted.append(dict(row))  # Keep as-is

        return adjusted  # Return adjusted rows
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def generate_cross_dataset_report(datasets_dict, file_extension=".csv", low_memory=True, output_filename=None, config: dict | None = None):
    """
    Generate a cross-dataset feature-compatibility report comparing dataset
    groups defined in `datasets_dict`. Produces pairwise comparisons between
    dataset groups and writes a CSV report named `Cross_{RESULTS_FILENAME}` by
    default into the `RESULTS_DIR`.

    :param datasets_dict: dict mapping dataset group name -> list of paths
    :param file_extension: extension to search for (default: .csv)
    :param low_memory: passed to CSV loader when building headers
    :param output_filename: optional filename to write; defaults to Cross_{RESULTS_FILENAME}
    :return: True on success, False otherwise
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        cfg = config or get_default_config()
        if output_filename is None:  # If no output filename is provided
            suffix = cfg.get("dataset_descriptor", {}).get("csv_output_suffix", "_description")
            output_filename = f"Cross_{suffix.lstrip('_')}" if suffix else "Cross_dataset_descriptor.csv"

        group_info = {}  # Map group_name -> {"files": [...], "common": set(), "union": set()}
        for group_name, paths in datasets_dict.items():  # Iterate over dataset groups
            all_files = collect_group_files(paths, file_extension, config=cfg)  # Collect files for this group
            common_features, union_features = compute_group_features(all_files, low_memory=low_memory)  # Compute features

            group_info[group_name] = {
                "files": all_files,
                "common": set(common_features),
                "union": union_features,
            }  # Store group info

        report_rows = generate_pairwise_report(group_info)  # Generate pairwise report rows
        if not report_rows:  # If no report rows were generated
            return False  # Return False indicating failure

        saved_any = False  # Flag to track if any report was saved
        for group_name, info in group_info.items():  # Iterate over each group
            base_dir = (
                os.path.dirname(os.path.abspath(info["files"][0])) if info["files"] else os.getcwd()
            )  # Base dir from first file or current dir
            adjusted_rows = adjust_rows_for_group(report_rows, group_name)  # Adjust rows for this group
            try:  # Try to write the report
                write_report(adjusted_rows, base_dir, output_filename, config=cfg)  # Write the report
                saved_any = True  # Mark that at least one report was saved
            except Exception:  # Fail silently
                pass  # Do nothing on failure

        return saved_any  # Return whether any report was saved
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def to_seconds(obj):
    """
    Converts various time-like objects to seconds.
    
    :param obj: The object to convert (can be int, float, timedelta, datetime, etc.)
    :return: The equivalent time in seconds as a float, or None if conversion fails
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def calculate_execution_time(start_time, finish_time=None):
    """
    Calculates the execution time and returns a human-readable string.

    Accepts either:
    - Two datetimes/timedeltas: `calculate_execution_time(start, finish)`
    - A single timedelta or numeric seconds: `calculate_execution_time(delta)`
    - Two numeric timestamps (seconds): `calculate_execution_time(start_s, finish_s)`

    Returns a string like "1h 2m 3s".
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def play_sound():
    """
    Plays a sound when the program finishes and skips if the operating system is Windows.

    :param: None
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        cfg = {}
        try:
            cfg = get_config() or {}
        except Exception:
            cfg = {}

        sound_cfg = cfg.get("sound", {}) if isinstance(cfg, dict) else {}
        sound_file = sound_cfg.get("file", SOUND_FILE)
        sound_cmds = sound_cfg.get("commands", SOUND_COMMANDS)

        current_os = platform.system()  # Get the current operating system
        if current_os == "Windows":  # If the current operating system is Windows
            return  # Do nothing on Windows by default

        if verify_filepath_exists(sound_file):  # If the sound file exists
            if current_os in sound_cmds:  # Use commands from config or defaults
                os.system(f"{sound_cmds[current_os]} {sound_file}")  # Play the sound
            else:  # Unknown OS mapping
                print(
                    f"{BackgroundColors.RED}The {BackgroundColors.CYAN}{current_os}{BackgroundColors.RED} is not configured in sound.commands. Please add it!{Style.RESET_ALL}"
                )
        else:  # If the sound file does not exist
            print(
                f"{BackgroundColors.RED}Sound file {BackgroundColors.CYAN}{sound_file}{BackgroundColors.RED} not found. Make sure the file exists or set 'sound.file' in config.yaml.{Style.RESET_ALL}"
            )
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def main():
    """
    Main function.

    :param: None
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        # Parse CLI args and load configuration (no CLI parsing on import)
        cli_args = parse_cli_args()
        config = get_config(file_path=cli_args.get("config", "config.yaml"), cli_args=cli_args)

        # Initialize runtime (logger, flags) from config
        runtime = init_runtime(config)

        # Redirect stdout/stderr to logger for main runtime only
        sys_stdout_old = sys.stdout
        sys_stderr_old = sys.stderr
        sys.stdout = runtime["logger"]
        sys.stderr = runtime["logger"]

        # Export verbosity flag for helper functions that read it
        os.environ["DD_DESCRIPTOR_VERBOSE"] = str(runtime.get("verbose", False))

        print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Dataset Descriptor{BackgroundColors.GREEN}!{Style.RESET_ALL}")
        start_time = datetime.datetime.now()

        # Log resolved configuration for traceability
        log_config_sources(config, cli_args)

        # Setup optional Telegram bot and send start message
        setup_telegram_bot()
        send_telegram_message(TELEGRAM_BOT, [f"Starting Dataset Descriptor at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"])

        datasets = config.get("datasets", {}) or {}
        results_suffix = config.get("dataset_descriptor", {}).get("csv_output_suffix", "_description")

        for dataset_name, paths in datasets.items():
            print(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing dataset: {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}")
            safe_dataset_name = str(dataset_name).replace(" ", "_").replace("/", "_")

            for dir_path in paths:
                print(f"{BackgroundColors.GREEN}Location: {BackgroundColors.CYAN}{dir_path}{Style.RESET_ALL}")
                if not verify_filepath_exists(dir_path):
                    print(f"{BackgroundColors.RED}The specified input path does not exist: {BackgroundColors.CYAN}{dir_path}{Style.RESET_ALL}")
                    continue

                success = generate_dataset_report(dir_path, file_extension=".csv", low_memory=True, output_filename=None, config=config)
                if not success:
                    print(f"{BackgroundColors.RED}Failed to generate dataset report for: {BackgroundColors.CYAN}{dir_path}{Style.RESET_ALL}")
                else:
                    print(f"{BackgroundColors.GREEN}Report saved for {BackgroundColors.CYAN}{dataset_name}{BackgroundColors.GREEN} -> {BackgroundColors.CYAN}{results_suffix}{Style.RESET_ALL}")

        # Cross-dataset validation when configured
        if config.get("execution", {}).get("cross_dataset_validate", True) and len(datasets) > 1:
            try:
                send_telegram_message(TELEGRAM_BOT, "Starting cross-dataset validation...")
                success = generate_cross_dataset_report(datasets, file_extension=".csv", config=config)
                if success:
                    print(f"{BackgroundColors.GREEN}Cross-dataset report saved -> {BackgroundColors.CYAN}Cross_{results_suffix.lstrip('_')}{Style.RESET_ALL}")
                else:
                    print(f"{BackgroundColors.YELLOW}No cross-dataset comparisons generated (no files found).{Style.RESET_ALL}")
            except Exception as e:
                print(f"{BackgroundColors.RED}Cross-dataset validation failed: {e}{Style.RESET_ALL}")

        sys.stdout = sys_stdout_old
        sys.stderr = sys_stderr_old

        finish_time = datetime.datetime.now()  # Get the finish time of the program
        print(
            f"{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
        )  # Output the start and finish times
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
        )  # Output the end of the program message

        send_telegram_message(
            TELEGRAM_BOT,
            f"Dataset Descriptor finished. Execution time: {calculate_execution_time(start_time, finish_time)}",
        )  # Send Telegram message about program completion

        try:
            if config.get("execution", {}).get("play_sound", True):
                atexit.register(play_sound)
        except Exception:
            pass
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


if __name__ == "__main__":
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """

    try:  # Protect main execution to ensure errors are reported and notified
        main()  # Call the main function
    except Exception as e:  # Catch any unhandled exception from main
        print(str(e))  # Print the exception message to terminal for logs
        try:  # Attempt to send full exception via Telegram
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback and message
        except Exception:  # If sending the notification fails, print traceback to stderr
            traceback.print_exc()  # Print full traceback to stderr as final fallback
        raise  # Re-raise to avoid silent failure and preserve original crash behavior
