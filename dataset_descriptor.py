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
    - The script performs disk-space verification before writing large outputs.
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


import argparse  # For parsing CLI arguments
import atexit  # For playing a sound when the program finishes
import dataframe_image as dfi  # For exporting DataFrame as PNG images
import datetime  # For timestamping
import gc  # For explicit garbage collection
import math  # For ceiling division in batch count computation
import matplotlib.pyplot as plt  # For plotting t-SNE results
import multiprocessing as mp  # For explicit process and semaphore resource finalization
import numpy as np  # For numerical operations
import os  # For running a command in the terminal
import pandas as pd  # For data manipulation
import platform  # For getting the operating system name
import psutil  # Local import to avoid global dependency if not needed
import re  # For regex operations
import sys  # For system-specific parameters and functions
import telegram_bot as telegram_module  # For setting Telegram prefix and device info
import time  # Import time locally to perform retry timing and ensure dependency is available at runtime
import traceback  # For printing full exception tracebacks
import warnings  # For suppressing pandas warnings when requested
import yaml  # For optional config.yaml loading when locating WGANGP outputs
from colorama import Style  # For coloring the terminal
from inspect import signature  # For inspecting function signatures
from Logger import Logger  # For logging output to both terminal and file
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from pathlib import Path  # For handling file paths
from PIL import Image  # For verifying image dimensions and upscaling if necessary
from sklearn.manifold import TSNE  # For t-SNE dimensionality reduction
from sklearn.preprocessing import StandardScaler  # For feature scaling
from telegram_bot import TelegramBot, send_exception_via_telegram, send_telegram_message, setup_global_exception_hook  # For sending progress messages and exceptions to Telegram
from tqdm import tqdm  # For progress bars
from typing import Any, cast  # For type hinting


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


# Functions Definitions:


def get_default_config() -> dict:
    """
    Return default configuration used by dataset_descriptor.py.

    :return: A dictionary containing default configuration values for dataset_descriptor.py.
    """
    
    return {
        "dataset_descriptor": {
            "low_memory": True,
            "include_preprocessing_metrics": True,
            "include_data_augmentation_info": True,
            "generate_table_image": True,
            "table_image_format": "png",
            "table_image_timeout_ms": 30000,
            "csv_output_suffix": "_description",
            "class_column_name": "Label",
            "dropna_before_analysis": False,
            "compute_class_distribution": True,
            "compute_feature_statistics": True,
            "round_decimals": 4,
            "batch_threshold_gb": None,
        },
        "paths": {
            "dataset_description_subdir": "Dataset_Description",
            "preprocessing_summary_subdir": "Preprocessing_Summary",
            "data_augmentation_dir": "Data_Augmentation",
            "data_augmentation_sample_dir": "Samples",
            "logs_dir": "./Logs",
        },
        "execution": {
            "verbose": True,
            "cross_dataset_validate": True,
        },
        "datasets": {},
    }


def deep_merge_dicts(base: dict, override: dict) -> dict:
    """
    Recursively merge override into base and return new dict.

    :param base: The base dictionary to merge into (not modified).
    :param override: The dictionary with override values (not modified).
    :return: A new dictionary resulting from deep merging override into base.
    """

    result = dict(base)  # Create a shallow copy of base to avoid mutation
    for k, v in (override or {}).items():  # Iterate over override entries
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):  # Verify both values are nested dicts
            result[k] = deep_merge_dicts(result[k], v)  # Recursively merge nested dicts
        else:  # Non-dict values are overwritten directly
            result[k] = v  # Set override value in result
    return result  # Return the merged dictionary


def load_config_file(path: str = "config.yaml") -> dict:
    """
    Load configuration from a YAML file if it exists, otherwise return an empty dict.

    :param path: Path to the YAML configuration file (default: "config.yaml").
    :return: Configuration dictionary loaded from the file, or empty dict if file does not exist or fails to load.
    """

    if path and verify_filepath_exists(path):  # Verify the path is non-empty and the file exists
        try:  # Attempt to open and parse the YAML file
            with open(path, "r", encoding="utf-8") as f:  # Open config file for reading
                return yaml.safe_load(f) or {}  # Parse YAML safely and fallback to empty dict
        except Exception:  # Ignore any parse or read errors and return empty dict
            return {}  # Return empty dict on parse failure
    return {}  # Return empty dict when file path is missing or file does not exist


def parse_cli_args(argv=None) -> dict:
    """
    Parse CLI arguments and return a dictionary of config overrides.

    :param argv: List of command-line arguments (default: None, which uses sys.argv).
    :return: Dictionary of config overrides based on CLI arguments.
    """

    parser = argparse.ArgumentParser(add_help=False)  # Create argument parser without default help option
    low_memory_group = parser.add_mutually_exclusive_group()
    low_memory_group.add_argument("--low-memory", dest="low_memory", action="store_true", default=None)
    low_memory_group.add_argument("--no-low-memory", dest="low_memory", action="store_false", default=None)
    parser.add_argument("--include_preprocessing_metrics", dest="include_preprocessing_metrics", action="store_true", default=None)
    parser.add_argument("--no-include_preprocessing_metrics", dest="include_preprocessing_metrics", action="store_false", default=None)
    parser.add_argument("--include_data_augmentation_info", dest="include_data_augmentation_info", action="store_true", default=None)
    parser.add_argument("--no-include_data_augmentation_info", dest="include_data_augmentation_info", action="store_false", default=None)
    parser.add_argument("--generate_table_image", dest="generate_table_image", action="store_true", default=None)
    parser.add_argument("--no-generate_table_image", dest="generate_table_image", action="store_false", default=None)
    parser.add_argument("--table_image_format", dest="table_image_format", default=None)
    parser.add_argument("--csv_output_suffix", dest="csv_output_suffix", default=None)
    parser.add_argument("--class_column_name", dest="class_column_name", default=None)
    parser.add_argument("--dropna_before_analysis", dest="dropna_before_analysis", action="store_true", default=None)
    parser.add_argument("--no-dropna_before_analysis", dest="dropna_before_analysis", action="store_false", default=None)
    parser.add_argument("--compute_class_distribution", dest="compute_class_distribution", action="store_true", default=None)
    parser.add_argument("--no-compute_class_distribution", dest="compute_class_distribution", action="store_false", default=None)
    parser.add_argument("--compute_feature_statistics", dest="compute_feature_statistics", action="store_true", default=None)
    parser.add_argument("--no-compute_feature_statistics", dest="compute_feature_statistics", action="store_false", default=None)
    parser.add_argument("--round_decimals", dest="round_decimals", type=int, default=None)
    parser.add_argument("--batch_threshold_gb", dest="batch_threshold_gb", type=float, default=None)  # Threshold in GB above which batched loading is activated
    parser.add_argument("--config", dest="config", default="config.yaml")
    args, _ = parser.parse_known_args(argv)  # Parse known args and discard unknown entries
    return {k: v for k, v in vars(args).items() if v is not None}  # Return only non-None values as overrides


def get_config(file_path: str = "config.yaml", cli_args: dict | None = None) -> dict:
    """
    Load and merge configuration with precedence CLI > config.yaml > defaults.

    :param file_path: Path to the configuration YAML file (default: "config.yaml").
    :param cli_args: Dictionary of CLI arguments that were parsed (optional).
    :return: Merged configuration dictionary.
    """

    defaults = get_default_config()  # Load hard-coded default configuration
    file_conf = load_config_file(file_path)  # Load configuration from disk
    merged = deep_merge_dicts(defaults, file_conf)  # Deep-merge file config over defaults

    if cli_args:  # Apply CLI overrides when provided
        dd = merged.setdefault("dataset_descriptor", {})  # Access or create dataset_descriptor section
        for key in [
            "low_memory",
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
            "batch_threshold_gb",
        ]:  # Iterate over keys that may be overridden by CLI
            if key in cli_args and cli_args[key] is not None:  # Verify the CLI arg key is present and non-None
                dd[key] = cli_args[key]  # Apply the CLI override for this key
        if "verbose" in cli_args and cli_args["verbose"] is not None:  # Verify verbose CLI arg is present and non-None
            merged.setdefault("execution", {})["verbose"] = cli_args["verbose"]  # Apply verbose CLI override to execution section
    return merged  # Return the fully merged configuration dictionary


def resolve_low_memory(cli_args: "argparse.Namespace", config: dict) -> bool:
    """
    Resolve the effective low_memory boolean using CLI overrides and config.

    Priority: CLI (if provided) -> config.dataset_descriptor.low_memory -> default

    :param cli_args: argparse.Namespace or dict of parsed CLI args (may be partial)
    :param config: merged configuration dict
    :return: final boolean for low_memory
    """
    
    try:  # Wrap full function logic to ensure production-safe monitoring and fallback to defaults on any error
        cli_val = None  # Initialize cli_val to None to detect if it was provided in CLI arguments
        if isinstance(cli_args, dict):  # Support both argparse.Namespace and dict for CLI arguments to allow flexible calling contexts
            cli_val = cli_args.get("low_memory", None)  # Attempt to get low_memory from dict, defaulting to None if not present
        else:  # Assume argparse.Namespace and attempt to get low_memory attribute, defaulting to None if not present
            cli_val = getattr(cli_args, "low_memory", None)  # Attempt to get low_memory from argparse.Namespace, defaulting to None if not present

        if cli_val is not None:  # If CLI provided a value (even if False), it takes precedence over config and defaults
            return bool(cli_val)  # Return the CLI value as boolean when it is explicitly provided, even if it is False

        if isinstance(config, dict):  # Verify config is a dictionary before accessing nested keys to avoid type errors
            return bool(config.get("dataset_descriptor", {}).get("low_memory", get_default_config().get("dataset_descriptor", {}).get("low_memory", True)))  # Return the config value for low_memory if it exists, otherwise fallback to default config value

        return bool(get_default_config().get("dataset_descriptor", {}).get("low_memory", True))  # Fallback to default config value when config is not a dict or does not contain the expected keys
    except Exception:  # Catch any exception during resolution to ensure production safety and fallback to defaults
        return bool(get_default_config().get("dataset_descriptor", {}).get("low_memory", True))  # On any error during resolution, fallback to default config value for low_memory


def init_runtime(config: dict):
    """
    Initialize runtime artifacts (logger) based on provided config.

    :param config: The merged configuration dictionary.
    :return: A dictionary containing runtime artifacts and settings.
    """

    validate_config_structure(config)  # Validate required config keys before accessing them

    logs_dir = config.get("paths", {}).get("logs_dir", "./Logs")  # Resolve logs directory from config with default
    os.makedirs(logs_dir, exist_ok=True)  # Create logs directory if it does not exist
    logger = Logger(os.path.join(logs_dir, f"{Path(__file__).stem}.log"), clean=True)  # Initialize a fresh logger for this run

    runtime = {
        "logger": logger,  # Logger instance for output redirection
        "verbose": bool(config.get("execution", {}).get("verbose", False)),  # Verbose flag from config
        "results_dir": os.path.join(".", config.get("paths", {}).get("dataset_description_subdir", "Dataset_Description")),  # Results output directory
        "results_filename_suffix": config.get("dataset_descriptor", {}).get("csv_output_suffix", "_description"),  # CSV output filename suffix
        "ignore_files": list(config.get("paths", {}).get("ignore_files", []) or []),  # Files to ignore during scanning
        "ignore_dirs": list(config.get("execution", {}).get("ignore_dirs", []) or []),  # Directories to ignore during scanning
    }  # Assemble runtime dictionary with all required artifacts
    return runtime  # Return the fully assembled runtime dictionary


def log_config_sources(config: dict, cli_args: dict | None = None):
    """
    Log configuration values with their source (CLI/config/default).

    :param config: The merged configuration dictionary.
    :param cli_args: The dictionary of CLI arguments that were parsed (optional).
    :return: None.
    """

    dd = config.get("dataset_descriptor", {})  # Extract dataset_descriptor section from config for logging
    for k, v in dd.items():  # Iterate over each configuration key-value pair
        src = "config"  # Assume configuration file as the default source
        if cli_args and k in cli_args and cli_args[k] is not None:  # Verify if key was overridden via CLI
            src = "CLI"  # Mark source as CLI when overridden
        elif k not in (load_config_file().get("dataset_descriptor") or {}):  # Verify if key is absent from config file
            src = "default"  # Mark source as default when not present in config file
        print(f"{BackgroundColors.GREEN}[CONFIG] {BackgroundColors.CYAN}{k}{BackgroundColors.GREEN} = {BackgroundColors.CYAN}{v}{BackgroundColors.GREEN} (source: {BackgroundColors.CYAN}{src}{BackgroundColors.GREEN})")  # Log configuration key and value with colored output


def validate_config_structure(config: dict):
    """
    Ensure required keys exist and have correct types for dataset_descriptor.

    :param config: The configuration dictionary to validate.
    :return: None, raises ValueError if validation fails.
    """

    if not isinstance(config, dict):  # Verify config is a dictionary before accessing keys
        raise ValueError("config must be a dictionary")  # Raise when config is not a dict
    dd = config.get("dataset_descriptor")  # Extract the dataset_descriptor section
    if not isinstance(dd, dict):  # Verify dataset_descriptor section is a mapping
        raise ValueError("config.dataset_descriptor must be a mapping")  # Raise when section is missing or invalid

    expected = {
        "include_preprocessing_metrics": bool,  # Boolean flag for preprocessing metrics
        "include_data_augmentation_info": bool,  # Boolean flag for augmentation info
        "generate_table_image": bool,  # Boolean flag for table image generation
        "table_image_format": str,  # String format for table images
        "csv_output_suffix": str,  # String suffix for CSV output files
        "class_column_name": str,  # String name of the class column
        "dropna_before_analysis": bool,  # Boolean flag for NaN dropping
        "compute_class_distribution": bool,  # Boolean flag for class distribution computation
        "compute_feature_statistics": bool,  # Boolean flag for feature statistics computation
        "round_decimals": int,  # Integer number of decimal places for rounding
    }  # Expected key-to-type mapping for validation

    for key, typ in expected.items():  # Iterate over required keys and their expected types
        if key not in dd:  # Verify required key is present in the section
            raise ValueError(f"Missing required config key: dataset_descriptor.{key}")  # Raise when key is absent
        if not isinstance(dd[key], typ):  # Verify the value has the correct type
            raise ValueError(f"Invalid type for dataset_descriptor.{key}: expected {typ.__name__}")  # Raise on type mismatch

    if dd["round_decimals"] < 0:  # Verify round_decimals is non-negative
        raise ValueError("dataset_descriptor.round_decimals must be >= 0")  # Raise when value is negative

    if "batch_threshold_gb" in dd and dd["batch_threshold_gb"] is not None:  # Verify batch_threshold_gb type when explicitly provided
        if not isinstance(dd["batch_threshold_gb"], (int, float)):  # Verify batch_threshold_gb is numeric when present
            raise ValueError("dataset_descriptor.batch_threshold_gb must be a float or None")  # Raise on type mismatch for batch_threshold_gb
        if float(dd["batch_threshold_gb"]) <= 0.0:  # Verify batch_threshold_gb is positive when specified
            raise ValueError("dataset_descriptor.batch_threshold_gb must be greater than 0.0 when specified")  # Raise when threshold is non-positive


def verbose_output(true_string="", false_string=""):
    """
    Output a message based on whether verbose mode is enabled via environment variable.

    :param true_string: The string to be outputted if verbose mode is enabled.
    :param false_string: The string to be outputted if verbose mode is disabled.
    :return: None.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_flag = os.environ.get("DD_DESCRIPTOR_VERBOSE", "False").lower() in ("1", "true", "yes")  # Resolve verbose flag from environment variable
        if verbose_flag and true_string != "":  # Verify verbose is enabled and true_string is non-empty
            print(true_string)  # Output the true string when verbose mode is active
        elif false_string != "":  # Verify false_string is non-empty for non-verbose output
            print(false_string)  # Output the false string when verbose mode is inactive
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


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
            telegram_module.TELEGRAM_DEVICE_INFO = f"{telegram_module.get_local_ip()} - {platform.system()}"  # Set device info for Telegram notifications
            telegram_module.RUNNING_CODE = os.path.basename(__file__)  # Set the running code name for Telegram context
        except Exception as e:
            print(f"{BackgroundColors.RED}Failed to initialize Telegram bot: {e}{Style.RESET_ALL}")  # Report initialization failure to terminal
            TELEGRAM_BOT = None  # Set to None if initialization fails
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def resolve_entry_with_trailing_space(current_path: str, entry: str, stripped_part: str) -> str:
    """
    Resolve and optionally rename a directory entry with trailing spaces.

    :param current_path: Current directory path.
    :param entry: Directory entry name.
    :param stripped_part: Normalized target name without surrounding spaces.
    :return: Resolved path after optional rename.
    """

    try:  # Wrap full function logic to ensure safe execution
        resolved = os.path.join(current_path, entry)  # Build resolved path

        if entry != stripped_part:  # Verify trailing spaces exist
            corrected = os.path.join(current_path, stripped_part)  # Build corrected path
            try:  # Attempt to rename entry
                os.rename(resolved, corrected)  # Rename entry to stripped version
                verbose_output(true_string=f"{BackgroundColors.GREEN}Renamed: {BackgroundColors.CYAN}{resolved}{BackgroundColors.GREEN} -> {BackgroundColors.CYAN}{corrected}{Style.RESET_ALL}")  # Log rename
                resolved = corrected  # Update resolved path after rename
            except Exception:  # Handle rename failure
                verbose_output(true_string=f"{BackgroundColors.RED}Failed to rename: {BackgroundColors.CYAN}{resolved}{Style.RESET_ALL}")  # Log failure

        return resolved  # Return resolved path
    except Exception:  # Catch unexpected errors
        return os.path.join(current_path, entry)  # Return fallback resolved path


def resolve_full_trailing_space_path(filepath: str) -> str:
    """
    Resolve trailing space issues across all path components.

    :param filepath: Path to resolve potential trailing space mismatches.
    :return: Corrected full path if matches are found, otherwise original filepath.
    """

    try:  # Wrap full function logic to ensure safe execution
        verbose_output(true_string=f"{BackgroundColors.GREEN}Resolving full trailing space path for: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}")  # Log start

        if not isinstance(filepath, str) or not filepath:  # Verify filepath validity
            verbose_output(true_string=f"{BackgroundColors.YELLOW}Invalid filepath provided, skipping resolution.{Style.RESET_ALL}")  # Log invalid input
            return filepath  # Return original

        filepath = os.path.expanduser(filepath)  # Expand ~ to user directory
        parts = filepath.split(os.sep)  # Split path into components

        if not parts:  # Verify path parts exist
            return filepath  # Return original

        if filepath.startswith(os.sep):  # Handle absolute paths
            current_path = os.sep  # Start from root
            parts = parts[1:]  # Remove empty root part
        else:
            current_path = parts[0] if parts[0] else os.getcwd()  # Initialize base
            parts = parts[1:] if parts[0] else parts  # Adjust parts

        for part in parts:  # Iterate over each path component
            if part == "":  # Skip empty parts
                continue  # Continue iteration

            try:  # Attempt to list current directory
                entries = os.listdir(current_path) if os.path.isdir(current_path) else []  # List current directory entries
            except Exception:  # Handle failure to list directory contents
                verbose_output(true_string=f"{BackgroundColors.RED}Failed to list directory: {BackgroundColors.CYAN}{current_path}{Style.RESET_ALL}")  # Log failure
                return filepath  # Return original

            stripped_part = part.strip()  # Normalize current part
            match_found = False  # Initialize match flag

            for entry in entries:  # Iterate directory entries
                try:  # Attempt safe comparison for each entry
                    if entry.strip() == stripped_part:  # Compare stripped names
                        current_path = resolve_entry_with_trailing_space(current_path, entry, stripped_part)  # Resolve entry and update current path
                        match_found = True  # Mark match
                        break  # Stop searching
                except Exception:  # Handle any unexpected error during comparison
                    continue  # Continue on error

            if not match_found:  # If no match found for this segment
                verbose_output(true_string=f"{BackgroundColors.YELLOW}No match for segment: {BackgroundColors.CYAN}{part}{Style.RESET_ALL}")  # Log miss
                return filepath  # Return original

        return current_path  # Return fully resolved path

    except Exception:  # Catch unexpected errors to maintain stability
        verbose_output(true_string=f"{BackgroundColors.RED}Error resolving full path: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}")  # Log error
        return filepath  # Return original


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
        
        if not isinstance(filepath, str) or not filepath.strip():  # Verify for non-string or empty/whitespace-only input   
            verbose_output(true_string=f"{BackgroundColors.YELLOW}Invalid filepath provided, skipping existence verification.{Style.RESET_ALL}")  # Log invalid input
            return False  # Return False for invalid input

        if os.path.exists(filepath):  # Fast path: original input exists
            return True  # Return True immediately

        candidate = str(filepath).strip()  # Normalize input to string and strip surrounding whitespace

        if (candidate.startswith("'") and candidate.endswith("'")) or (
            candidate.startswith('"') and candidate.endswith('"')
        ):  # Handle quoted paths from config files
            candidate = candidate[1:-1].strip()  # Remove wrapping quotes and trim again

        candidate = os.path.expanduser(candidate)  # Expand ~ to user home directory
        candidate = os.path.normpath(candidate)  # Normalize path separators and structure

        if os.path.exists(candidate):  # Verify normalized candidate directly
            return True  # Return True if normalized path exists

        repo_dir = os.path.dirname(os.path.abspath(__file__))  # Resolve repository directory
        cwd = os.getcwd()  # Capture current working directory

        alt = candidate.lstrip(os.sep) if candidate.startswith(os.sep) else candidate  # Prepare relative-safe path

        repo_candidate = os.path.join(repo_dir, alt)  # Build repo-relative candidate
        cwd_candidate = os.path.join(cwd, alt)  # Build cwd-relative candidate

        for path_variant in (repo_candidate, cwd_candidate):  # Iterate alternative base paths
            try:
                normalized_variant = os.path.normpath(path_variant)  # Normalize variant
                if os.path.exists(normalized_variant):  # Verify existence
                    return True  # Return True if found
            except Exception:
                continue  # Continue safely on error

        try:  # Attempt absolute path resolution as fallback
            abs_candidate = os.path.abspath(candidate)  # Build absolute path
            if os.path.exists(abs_candidate):  # Verify existence
                return True  # Return True if found
        except Exception:
            pass  # Ignore resolution errors

        for path_variant in (candidate, repo_candidate, cwd_candidate):  # Attempt trailing-space resolution on all variants
            try:  # Attempt to resolve trailing space issues across path components for this variant
                resolved = resolve_full_trailing_space_path(path_variant)  # Resolve trailing space issues across path components
                if resolved != path_variant and os.path.exists(resolved):  # Verify resolved path exists
                    verbose_output(
                        f"{BackgroundColors.YELLOW}Resolved trailing space mismatch: {BackgroundColors.CYAN}{path_variant}{BackgroundColors.YELLOW} -> {BackgroundColors.CYAN}{resolved}{Style.RESET_ALL}"
                    )  # Log successful resolution
                    return True  # Return True if corrected path exists
            except Exception:  # Catch any exception during trailing space resolution   
                continue  # Continue safely on error

        return False  # Not found after all resolution strategies
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
    Recursively collect all files in the specified directory and subdirectories
    that match the given file format and are not in the ignore list.

    :param input_dir: Directory to search.
    :param file_format: File format to include (default: .csv).
    :param ignore_files: List of filenames to ignore.
    :param ignore_dirs: List of directory names to ignore.
    :param config: Optional configuration dictionary for ignore list resolution.
    :return: Sorted list of matching file paths.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Collecting all files with format {BackgroundColors.CYAN}{file_format}{BackgroundColors.GREEN} in directory: {BackgroundColors.CYAN}{input_dir}{Style.RESET_ALL}"
        )  # Output the verbose message

        cfg = config or get_default_config()
        resolved_ignore_files = ignore_files if ignore_files is not None else list(cfg.get("paths", {}).get("ignore_files", []) or [])
        resolved_ignore_dirs = ignore_dirs if ignore_dirs is not None else list(cfg.get("execution", {}).get("ignore_dirs", ["Cache", "Data_Separability", "Dataset_Description", "Feature_Analysis"]) or ["Cache", "Data_Separability", "Dataset_Description", "Feature_Analysis"])

        normalized_ignore_files: set[str] = {os.path.normcase(f) for f in (resolved_ignore_files or [])}  # Create normalized set of ignored filenames for fast membership verification
        normalized_ignore_dirs: set[str] = {os.path.normcase(d) for d in (resolved_ignore_dirs or [])}  # Create normalized set of ignored directory names for fast membership verification

        matching_files = []  # List to store matching file paths

        for root, dirs, files in os.walk(input_dir):  # Walk through the directory
            try:  # Try to filter out ignored directories
                dirs[:] = [
                    d for d in dirs if os.path.normcase(d) not in normalized_ignore_dirs
                ]  # Modify dirs in-place to skip ignored directories using normalized set
            except Exception:  # If an error occurs while filtering directories
                pass  # Ignore the error and continue

            for file in files:  # For each file
                if not file.endswith(file_format):  # Skip files that do not match the specified format
                    continue  # Continue to the next file

                basename_norm = os.path.normcase(file)  # Normalize the basename for case-insensitive comparison
                fullpath = os.path.join(root, file)  # Get the full file path
                fullpath_norm = os.path.normcase(fullpath)  # Normalize the full file path for case-insensitive comparison

                if basename_norm in normalized_ignore_files or fullpath_norm in normalized_ignore_files:  # If the file is in the ignore set
                    verbose_output(f"Skipping ignored file: {fullpath}")  # Output verbose message for ignored file
                    continue  # Continue to the next file

                matching_files.append(fullpath)  # Add the full file path to the list

        unique_files = list(set(matching_files))  # Remove duplicates while preserving no particular order

        files_with_size = []  # Prepare list to hold (path, size) tuples for robust sorting
        for f in unique_files:  # Iterate files to resolve their sizes
            try:  # Attempt to get file size and handle any filesystem issues gracefully
                size = os.path.getsize(f)  # Get the file size in bytes for sorting by magnitude
            except Exception:  # If size retrieval fails for any file
                size = 0  # Fallback to zero size to avoid breaking the sort when file is inaccessible
            files_with_size.append((f, size))  # Store tuple of file path and its size for later sorting

        sorted_matching_files = [p for p, _ in sorted(files_with_size, key=lambda x: (-x[1], x[0]))]  # Sort by size descending then by path for determinism

        return sorted_matching_files  # Return the list of matching files ordered from biggest to smallest by file size
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def build_headers_map(filepaths, low_memory=None):
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
                try:  # Attempt UTF-8 header read first to prefer standard encoding
                    df_headers = pd.read_csv(fp, nrows=0, encoding="utf-8")  # Read header with UTF-8 encoding
                except UnicodeDecodeError:  # Handle UTF-8 decode failures specifically
                    try:  # Attempt Latin-1 decoding as first fallback for legacy CSVs
                        df_headers = pd.read_csv(fp, nrows=0, encoding="latin1")  # Read header with Latin-1 encoding
                    except UnicodeDecodeError:  # Handle Latin-1 decode failures specifically
                        df_headers = pd.read_csv(fp, nrows=0, encoding="cp1252")  # Read header with CP1252 encoding as final fallback
                df_headers.columns = df_headers.columns.str.strip()  # Remove leading/trailing whitespace from column names
                cols = df_headers.columns.tolist()  # Get column list
            except Exception:  # If header-only read fails
                df_tmp = load_dataset(fp, low_memory=low_memory)  # Load full dataset (slow fallback)
                cols = df_tmp.columns.tolist() if df_tmp is not None else []  # Extract columns if dataset loaded
                del df_tmp  # Release fallback DataFrame immediately to reclaim memory
                gc.collect()  # Force garbage collection after releasing the full dataset
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


def count_csv_rows(filepath: str) -> int:
    """
    Count the total number of data rows in a CSV file using chunked reading.

    :param filepath: Path to the CSV file to count rows in.
    :return: Total number of data rows excluding the header, or 0 on failure.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        total = 0  # Initialize row counter for incremental accumulation

        try:  # Attempt chunked UTF-8 read for accurate row counting
            with warnings.catch_warnings():  # Suppress DtypeWarning during counting
                warnings.simplefilter("ignore", pd.errors.DtypeWarning)  # Ignore dtype mismatch warnings
                for chunk in pd.read_csv(filepath, encoding="utf-8", chunksize=10000, low_memory=True):  # Read fixed-size chunks using UTF-8 encoding
                    total += len(chunk)  # Accumulate row count from each chunk
                    del chunk  # Release each chunk immediately after counting to free memory
        except UnicodeDecodeError:  # Handle UTF-8 decode failures
            total = 0  # Reset counter before retrying with Latin-1 encoding
            try:  # Attempt chunked Latin-1 read as first fallback encoding
                with warnings.catch_warnings():  # Suppress DtypeWarning during Latin-1 counting
                    warnings.simplefilter("ignore", pd.errors.DtypeWarning)  # Ignore dtype mismatch warnings
                    for chunk in pd.read_csv(filepath, encoding="latin1", chunksize=10000, low_memory=True):  # Read chunks using Latin-1 encoding
                        total += len(chunk)  # Accumulate row count from each chunk
                        del chunk  # Release each chunk immediately after counting to free memory
            except UnicodeDecodeError:  # Handle Latin-1 decode failures
                total = 0  # Reset counter before final encoding attempt
                with warnings.catch_warnings():  # Suppress DtypeWarning during CP1252 counting
                    warnings.simplefilter("ignore", pd.errors.DtypeWarning)  # Ignore dtype mismatch warnings
                    for chunk in pd.read_csv(filepath, encoding="cp1252", chunksize=10000, low_memory=True):  # Read chunks using CP1252 encoding as final fallback
                        total += len(chunk)  # Accumulate row count from each chunk
                        del chunk  # Release each chunk immediately after counting to free memory

        gc.collect()  # Trigger garbage collection after counting loop to reclaim chunk memory
        return max(0, int(total))  # Return non-negative integer row count
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def get_file_size_gb(filepath: str) -> float:
    """
    Determine the size of a file in gigabytes.

    :param filepath: Path to the file to measure.
    :return: File size in gigabytes as a float, or 0.0 on failure.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        size_bytes = os.path.getsize(filepath)  # Retrieve raw file size in bytes from filesystem
        size_gb = float(size_bytes) / (1024.0 ** 3)  # Convert bytes to gigabytes using binary prefix
        return size_gb  # Return file size as a float in gigabytes
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def compute_batch_count(file_size_gb: float, threshold: float) -> int:
    """
    Compute the number of batches required based on file size and batch threshold.

    :param file_size_gb: File size in gigabytes.
    :param threshold: Batch threshold in gigabytes; each batch covers at most this many GB.
    :return: Number of batches as an integer, always at least 1.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        if threshold <= 0.0:  # Verify threshold is positive to avoid division by zero
            return 1  # Return single batch when threshold is non-positive
        batches = int(math.ceil(float(file_size_gb) / float(threshold)))  # Compute ceiling of file size divided by threshold
        return max(1, batches)  # Ensure at least one batch is always returned
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_dataset(filepath, low_memory=None):
    """
    Loads a dataset from a CSV file.

    :param filepath: Path to the CSV file
    :param low_memory: Whether to use low memory mode (default: True)
    :return: Pandas DataFrame
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        low_memory = True if low_memory is None else bool(low_memory)  # Default to True if low_memory is not provided
        
        try:  # Try to load the dataset
            with warnings.catch_warnings():  # Suppress DtypeWarning warnings
                warnings.simplefilter("ignore", pd.errors.DtypeWarning)  # Ignore DtypeWarning warnings
                try:  # Attempt to read file using UTF-8 as primary encoding
                    df = pd.read_csv(filepath, low_memory=low_memory, encoding="utf-8")  # Read dataset using UTF-8 encoding
                except UnicodeDecodeError:  # If UTF-8 decoding fails, try legacy encodings
                    try:  # Attempt to read file using Latin-1 as fallback encoding
                        df = pd.read_csv(filepath, low_memory=low_memory, encoding="latin1")  # Read dataset using Latin-1 encoding
                    except UnicodeDecodeError:  # If Latin-1 decoding also fails, try CP1252 as final fallback
                        df = pd.read_csv(filepath, low_memory=low_memory, encoding="cp1252")  # Read dataset using CP1252 encoding
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


def finalize_multiprocessing_resources() -> None:
    """
    Finalize multiprocessing children and tracked shared resources.

    :return: None.
    """

    try:  # Wrap function body to preserve exception-handling conventions
        try:  # Iterate active children to avoid orphan processes at interpreter shutdown
            for child in mp.active_children():  # Traverse active multiprocessing child processes
                try:  # Join quickly when a child is already exiting
                    child.join(timeout=0.2)  # Wait briefly for graceful child completion
                except Exception:  # Ignore join failures to preserve best-effort semantics
                    pass  # Continue with remaining children when join fails

                if child.is_alive():  # Determine if child remained alive after short grace period
                    try:  # Attempt forced termination when child does not exit naturally
                        child.terminate()  # Terminate child process to avoid orphan resources
                    except Exception:  # Ignore terminate failures to continue cleanup attempts
                        pass  # Continue cleanup even when termination raises
                    try:  # Attempt final join after termination request
                        child.join(timeout=0.5)  # Wait briefly for terminated child to finalize
                    except Exception:  # Ignore final join failures to preserve best-effort semantics
                        pass  # Continue cleanup sequence
        except Exception:  # Ignore failures while enumerating or joining children
            pass  # Continue with shared-resource tracker cleanup path

        gc.collect()  # Trigger garbage collection after resource finalization attempts
    except Exception as e:  # Catch outer exceptions to preserve module-wide telemetry behavior
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def configure_multiprocessing_startup() -> None:
    """
    Configure multiprocessing startup behavior for safe macOS process semantics.

    :return: None.
    """

    try:  # Wrap startup configuration with module-consistent error handling
        mp.set_start_method("spawn", force=True)  # Force spawn context to prevent macOS fork-related semaphore lifecycle conflicts
    except RuntimeError:  # Ignore context configuration errors when method is already fixed in this process
        pass  # Preserve startup flow when start method cannot be changed
    except Exception as e:  # Capture unexpected failures and keep telemetry behavior consistent
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
                f"{BackgroundColors.GREEN}Preprocessing DataFrame: "  # Verify intent message
                f"{BackgroundColors.CYAN}normalizing and sanitizing column names, removing NaN rows, removing infinite rows, and dropping zero-variance numeric features"
                f"{BackgroundColors.GREEN}.{Style.RESET_ALL}"
            )  # Emit verbose message when zero-variance removal is requested
        else:  # If remove_zero_variance is set to False
            verbose_output(
                f"{BackgroundColors.GREEN}Preprocessing DataFrame: "  # Verify intent message
                f"{BackgroundColors.CYAN}normalizing and sanitizing column names, removing NaN rows, and removing infinite rows"
                f"{BackgroundColors.GREEN}.{Style.RESET_ALL}"
            )  # Emit verbose message when zero-variance removal is not requested

        if df is None:  # If the DataFrame is None
            return df  # Return None when no DataFrame provided

        df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names

        df.columns = sanitize_feature_names(df.columns)  # Sanitize column names to remove special characters

        original_row_count = int(len(df))  # Record original row count for metric computations
        original_feature_count = int(df.shape[1]) if hasattr(df, "shape") else 0  # Record original feature count for metric computations
        preprocessing_metrics = {"nan_removal": {}, "infinite_removal": {}, "nan_inf": {}, "zero_variance": {}, "final": {}}  # Initialize structured per-step metrics container

        rows_before_nan_removal = int(len(df))  # Capture row count before NaN/null filtering step
        missing_mask = df.isna().any(axis=1)  # Identify rows that contain any NaN/null values
        df_no_nan = df.loc[~missing_mask]  # Remove rows that contain NaN/null values
        rows_after_nan_removal = int(len(df_no_nan))  # Capture row count after NaN/null filtering step
        removed_rows_nan = int(rows_before_nan_removal - rows_after_nan_removal)  # Compute rows removed by NaN/null filtering step
        if rows_before_nan_removal > 0:  # Guard division by zero for NaN/null removed-row proportion
            removed_rows_nan_proportion = round(float(removed_rows_nan) / float(rows_before_nan_removal), 6)  # Compute NaN/null removed-row proportion
        else:  # Handle empty datasets before NaN/null filtering
            removed_rows_nan_proportion = 0.0  # Set NaN/null removed-row proportion to zero
        preprocessing_metrics["nan_removal"] = {"rows_before_step": rows_before_nan_removal, "rows_after_step": rows_after_nan_removal, "removed_rows_step": removed_rows_nan, "removed_rows_step_proportion": removed_rows_nan_proportion}  # Store isolated NaN/null step metrics

        rows_before_inf_removal = int(len(df_no_nan))  # Capture row count before infinite-value filtering step
        numeric_cols = df_no_nan.select_dtypes(include=["number"]).columns  # Identify numeric columns to detect infinite values
        if len(numeric_cols) > 0:  # Only compute infinite masks when numeric columns exist
            inf_mask = df_no_nan[numeric_cols].isin([np.inf, -np.inf]).any(axis=1)  # Identify rows with any +/-inf in numeric columns
        else:  # When no numeric columns exist
            inf_mask = pd.Series([False] * len(df_no_nan), index=df_no_nan.index)  # Create false mask to avoid errors
        df_no_inf = df_no_nan.loc[~inf_mask]  # Remove rows that contain infinite values in numeric columns
        del df_no_nan  # Release NaN-filtered intermediate DataFrame to reduce memory footprint after infinite-value filtering is applied
        rows_after_inf_removal = int(len(df_no_inf))  # Capture row count after infinite-value filtering step
        removed_rows_inf = int(rows_before_inf_removal - rows_after_inf_removal)  # Compute rows removed by infinite-value filtering step
        if rows_before_inf_removal > 0:  # Guard division by zero for infinite-value removed-row proportion
            removed_rows_inf_proportion = round(float(removed_rows_inf) / float(rows_before_inf_removal), 6)  # Compute infinite-value removed-row proportion
        else:  # Handle empty datasets before infinite-value filtering
            removed_rows_inf_proportion = 0.0  # Set infinite-value removed-row proportion to zero
        preprocessing_metrics["infinite_removal"] = {"rows_before_step": rows_before_inf_removal, "rows_after_step": rows_after_inf_removal, "removed_rows_step": removed_rows_inf, "removed_rows_step_proportion": removed_rows_inf_proportion}  # Store isolated infinite-value step metrics

        removed_rows_nan_inf = int(original_row_count - rows_after_inf_removal)  # Compute rows removed by NaN+infinite filtering sequence
        if original_row_count > 0:  # Guard division by zero for NaN+infinite removed-row proportion
            removed_rows_nan_inf_proportion = round(float(removed_rows_nan_inf) / float(original_row_count), 6)  # Compute NaN+infinite removed-row proportion
        else:  # Handle empty original datasets
            removed_rows_nan_inf_proportion = 0.0  # Set NaN+infinite removed-row proportion to zero
        preprocessing_metrics["nan_inf"] = {"rows_before_step": original_row_count, "rows_after_step": rows_after_inf_removal, "removed_rows_step": removed_rows_nan_inf, "removed_rows_step_proportion": removed_rows_nan_inf_proportion}  # Store aggregated NaN+infinite step metrics

        df_clean = df_no_inf.replace([np.nan], np.nan)  # Ensure canonical NaN representation after removal steps
        del df_no_inf  # Release infinite-filtered intermediate DataFrame to reduce memory footprint after NaN normalization is applied
        features_before_zero_variance = int(df_clean.shape[1]) if hasattr(df_clean, "shape") else 0  # Capture feature count before zero-variance removal
        features_after_zero_variance = features_before_zero_variance  # Initialize feature count after zero-variance removal
        removed_zero_variance_features = 0  # Initialize removed zero-variance feature counter

        if remove_zero_variance:  # If zero-variance removal is requested
            numeric_cols_post = df_clean.select_dtypes(include=["number"]).columns  # Identify numeric columns for variance computation
            if len(numeric_cols_post) > 0:  # Only compute variances when numeric columns exist
                variances = df_clean[numeric_cols_post].var(axis=0, ddof=0)  # Calculate variances for numeric columns
                zero_var_cols = variances[variances == 0].index.tolist()  # Collect column names with zero variance
                if zero_var_cols:  # If there are zero-variance columns found
                    df_clean = df_clean.drop(columns=zero_var_cols)  # Drop the zero-variance columns from DataFrame
        features_after_zero_variance = int(df_clean.shape[1]) if hasattr(df_clean, "shape") else 0  # Capture feature count after zero-variance removal step
        removed_zero_variance_features = int(features_before_zero_variance - features_after_zero_variance)  # Compute removed zero-variance feature count
        removed_zero_variance_features = removed_zero_variance_features if removed_zero_variance_features >= 0 else 0  # Clamp negative removed feature count to zero
        if features_before_zero_variance > 0:  # Guard division by zero for zero-variance removed-feature proportion
            removed_zero_variance_features_proportion = round(float(removed_zero_variance_features) / float(features_before_zero_variance), 6)  # Compute zero-variance removed-feature proportion
        else:  # Handle empty feature spaces before zero-variance removal
            removed_zero_variance_features_proportion = 0.0  # Set zero-variance removed-feature proportion to zero
        preprocessing_metrics["zero_variance"] = {"features_before_step": features_before_zero_variance, "features_after_step": features_after_zero_variance, "removed_features_step": removed_zero_variance_features, "removed_features_step_proportion": removed_zero_variance_features_proportion}  # Store isolated zero-variance feature step metrics

        rows_after_preprocessing = int(len(df_clean))  # Capture final row count after preprocessing pipeline
        removed_rows_total = int(original_row_count - rows_after_preprocessing)  # Compute total removed rows across preprocessing pipeline
        removed_rows_total = removed_rows_total if removed_rows_total >= 0 else 0  # Clamp negative total removed rows to zero
        if original_row_count > 0:  # Guard division by zero for total removed-row proportion
            removed_rows_total_proportion = round(float(removed_rows_total) / float(original_row_count), 6)  # Compute total removed-row proportion
        else:  # Handle empty original datasets
            removed_rows_total_proportion = 0.0  # Set total removed-row proportion to zero
        preprocessing_metrics["final"] = {"rows_before_step": original_row_count, "rows_after_step": rows_after_preprocessing, "removed_rows_step": removed_rows_total, "removed_rows_step_proportion": removed_rows_total_proportion, "features_before_step": original_feature_count, "features_after_step": int(df_clean.shape[1]) if hasattr(df_clean, "shape") else 0}  # Store final aggregated preprocessing metrics
        df_clean.attrs["preprocessing_metrics"] = preprocessing_metrics  # Attach structured preprocessing metrics to DataFrame metadata

        return df_clean  # Return the cleaned DataFrame after sequential preprocessing
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
        candidates = ["label", "class", "target", "y", "category"]  # Common label column names to verify for exact matches

        for col in columns:  # First search for exact matches
            if col.lower() in candidates:  # Verify if the column name matches any candidate exactly
                return col  # Return the column name if found

        for col in columns:  # Second search for partial matches
            if "target" in col.lower() or "label" in col.lower():  # Verify if the column name contains any candidate
                return col  # Return the column name if found

        return ""  # Return empty string if no label column is found (fix type issue)
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


def extract_labels_info(df: pd.DataFrame) -> tuple:  # Define function to extract label info with type hints
    """
    Extract number of unique labels and label list.

    :param df: pandas DataFrame containing potential label column.
    :return: Tuple of (num_labels, labels_list).
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        label_col = detect_label_column(df.columns)  # Detect potential label column using existing heuristic
        if label_col and label_col in df.columns:  # If label column exists in the DataFrame
            labels = df[label_col].dropna().unique().tolist()  # Extract unique non-NaN labels preserving types
            labels_sorted = sorted(labels, key=lambda x: str(x))  # Sort labels deterministically by string representation
            return int(len(labels_sorted)), labels_sorted  # Return integer count and sorted list of labels
        return 0, []  # Return zero and empty list when no label column is detected
    except Exception as e:  # Catch any exception to preserve existing telemetry behavior
        print(str(e))  # Print exception to terminal for visibility
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception via Telegram hook for monitoring
        raise  # Re-raise exception to maintain original semantics


def format_labels_list(labels_list: list) -> str:  # Define function to format labels list into stable CSV string
    """
    Format labels list into stable CSV string.

    :param labels_list: List of labels to format.
    :return: String representation suitable for CSV.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        if not labels_list:  # If the provided labels list is empty
            return "[]"  # Return empty list representation for CSV
        escaped = ", ".join(repr(x) for x in labels_list)  # Create comma-separated repr string to preserve types and quoting
        return f"[{escaped}]"  # Return bracketed representation suitable for CSV ingestion
    except Exception as e:  # Catch any exception to preserve existing telemetry behavior
        print(str(e))  # Print exception to terminal for visibility
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception via Telegram hook for monitoring
        raise  # Re-raise exception to maintain original semantics


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

        numeric_df = df.select_dtypes(include=["number"])  # Select numeric columns from the DataFrame; select_dtypes returns a new object so an explicit copy is not needed
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

    This function computes the initial allocation for each class as the
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
    this function distributes the remaining available capacity proportionally
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

    This function computes a proportional distribution of `max_samples` across
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

    For each class in `allocations`, this function selects the requested number
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


def prepare_numeric_dataset(filepath, low_memory=None, sample_size=5000, random_state=42):
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
            del df  # Release the loaded DataFrame before returning from failed cleaning path
            return None, None  # Abort

        numeric_df = coerce_numeric_columns(cleaned)  # Extract numeric features
        if numeric_df is None:  # If extraction failed
            del cleaned  # Release cleaned DataFrame before returning from failed extraction path
            del df  # Release original DataFrame before returning from failed extraction path
            return None, None  # Abort

        numeric_df = fill_replace_and_drop(numeric_df)  # Clean numeric frame
        if numeric_df is None:  # If cleaning failed
            del cleaned  # Release cleaned DataFrame before returning from failed numeric cleanup path
            del df  # Release original DataFrame before returning from failed numeric cleanup path
            return None, None  # Abort

        if numeric_df.shape[0] == 0 or numeric_df.shape[1] == 0:  # No numeric data
            del numeric_df  # Release empty numeric DataFrame before returning
            del cleaned  # Release cleaned DataFrame before returning
            del df  # Release original DataFrame before returning
            return None, None  # Abort

        label_col = detect_label_column(cleaned.columns)  # Detect label column
        labels = cleaned[label_col] if label_col in cleaned.columns else None  # Extract labels if present

        if numeric_df.shape[0] > sample_size:  # Downsample if too many rows
            numeric_df, labels = downsample_with_class_awareness(
                numeric_df, labels, sample_size, random_state
            )  # Class-aware downsampling

        del cleaned  # Release cleaned DataFrame after deriving numeric frame and labels
        del df  # Release original DataFrame after deriving numeric frame and labels
        gc.collect()  # Trigger garbage collection after releasing large intermediate DataFrames

        return numeric_df, labels  # Return numeric DataFrame and labels
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def prepare_numeric_dataset_from_df(df, sample_size=5000, random_state=42):
    """
    Prepare numeric DataFrame and labels from an already-loaded DataFrame.

    :param df: Pandas DataFrame to prepare numeric features from.
    :param sample_size: Maximum number of rows to keep after optional downsampling.
    :param random_state: Random seed for reproducible downsampling.
    :return: Tuple of (numeric_df, labels) or (None, None) on failure.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        if df is None:  # Verify that a valid DataFrame was provided
            return None, None  # Abort when no DataFrame is available

        cleaned = preprocess_dataframe(df, remove_zero_variance=False)  # Apply basic cleaning without removing zero-variance columns
        if cleaned is None:  # Verify preprocessing did not fail
            return None, None  # Abort if preprocessing produced no valid output

        numeric_df = coerce_numeric_columns(cleaned)  # Extract or coerce numeric columns from cleaned DataFrame
        if numeric_df is None:  # Verify coercion did not fail
            del cleaned  # Release cleaned DataFrame before returning from failed coercion path
            return None, None  # Abort when no numeric columns could be obtained

        numeric_df = fill_replace_and_drop(numeric_df)  # Replace infinities, drop all-NaN columns and fill remaining NaNs
        if numeric_df is None:  # Verify fill and drop did not fail
            del cleaned  # Release cleaned DataFrame before returning from failed numeric cleanup path
            return None, None  # Abort when numeric frame became invalid after cleaning

        if numeric_df.shape[0] == 0 or numeric_df.shape[1] == 0:  # Verify numeric frame is not empty after all cleaning steps
            del numeric_df  # Release empty numeric DataFrame before returning
            del cleaned  # Release cleaned DataFrame before returning
            return None, None  # Abort when no usable rows or columns remain

        label_col = detect_label_column(cleaned.columns)  # Detect label column from cleaned DataFrame columns
        labels = cleaned[label_col] if label_col in cleaned.columns else None  # Extract labels when column is present

        if numeric_df.shape[0] > sample_size:  # Verify whether downsampling is required
            numeric_df, labels = downsample_with_class_awareness(
                numeric_df, labels, sample_size, random_state
            )  # Downsample while preserving class distribution

        del cleaned  # Release cleaned DataFrame after deriving numeric frame and labels
        gc.collect()  # Trigger garbage collection after releasing large intermediate DataFrame

        return numeric_df, labels  # Return numeric features and corresponding labels
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


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
    Function to distribute remaining budget among classes proportionally,
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
        if total_alloc > sample_size:  # Safety verification to reduce overallocation
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
        if "n_iter" in sig:  # Verifies for n_iter parameter
            tsne_kwargs["n_iter"] = n_iter  # Set n_iter if supported
        elif "max_iter" in sig:  # Verifies for max_iter parameter
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


def sanitize_plot_text(text: str) -> str:
    """
    Normalize text to safe UTF-8 for matplotlib rendering.

    :param text: Original text string.
    :return: Sanitized text string.
    """
    
    if text is None:  # Handle None inputs gracefully
        return ""  # Return empty string for None inputs
    
    try:  # Try explicit replacements and UTF-8 normalization
        sanitized = str(text).replace("\x96", "-")  # Replace CP1252 EN DASH with ASCII hyphen
        sanitized = sanitized.replace("–", "-")  # Replace Unicode EN DASH with ASCII hyphen
        sanitized = sanitized.replace("—", "-")  # Replace EM DASH with ASCII hyphen
        sanitized = sanitized.replace("\x92", "'")  # Replace CP1252 smart quote with apostrophe
        sanitized = sanitized.encode("utf-8", "ignore").decode("utf-8")  # Remove invalid UTF-8 sequences
        sanitized = "".join(ch for ch in sanitized if (ch.isprintable() or ch in "\t\n\r"))  # Remove control characters
        return sanitized  # Return cleaned text
    except Exception:  # In case of unexpected errors during sanitization
        try:  # Best-effort fallback to safe ASCII representation
            return str(text).encode("ascii", "ignore").decode("ascii")  # Fallback to ASCII-only string
        except Exception:  # If fallback also fails
            return ""  # Return empty string as ultimate fallback


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
        target_width_px = 3840  # Desired width in pixels for the output image (e.g., 3840 for 4K UHD)
        target_height_px = 2160  # Desired height in pixels for the output image (e.g., 2160 for 4K UHD)
        dpi_given = 1000  # High DPI to ensure the output image has the desired pixel dimensions when saved

        figsize = (target_width_px / float(dpi_given), target_height_px / float(dpi_given))  # Calculate figure size in inches to achieve target pixel dimensions at the given DPI
        fig = plt.figure(figsize=figsize)  # Create matplotlib figure (DPI preserved by savefig)

        try:  # Try plotting and saving to ensure figure is closed even if an error occurs
            if labels is not None:  # Plot colored by class
                labels_ser = pd.Series(labels)  # Ensure labels are a pandas Series
                counts = labels_ser.value_counts()  # Count samples per class
                unique = list(labels_ser.unique())  # Unique class labels (preserve order)
                for cls in unique:  # Plot each class separately
                    mask = labels_ser == cls  # Boolean mask for class
                    plt.scatter(X_emb[mask, 0], X_emb[mask, 1], label=sanitize_plot_text(f"{cls} ({int(counts.get(cls, 0))})"), s=8)  # Scatter plot for class with sanitized count label
                plt.legend(markerscale=2, fontsize="small")  # Add legend for classes
                try:  # Try to add counts text box
                    counts_text = "\n".join([f"{str(c)}: {int(counts[c])}" for c in counts.index])  # Prepare counts text
                    counts_text = sanitize_plot_text(counts_text)  # Sanitize counts text before rendering
                    fig.text(
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

            title = sanitize_plot_text(title)  # Sanitize incoming title before plotting
            plt.title(title)  # Set plot title
            plt.xlabel(sanitize_plot_text("t-SNE 1"))  # X-axis label sanitized
            plt.ylabel(sanitize_plot_text("t-SNE 2"))  # Y-axis label sanitized
            plt.tight_layout()  # Adjust layout
            try:  # Try saving the figure
                fig.savefig(output_path, dpi=300)  # Save figure to disk
            finally:  # Ensure the figure is closed to free memory
                plt.close(fig)  # Close the figure to free memory
        except Exception as e:  # Catch any exception during plotting or saving to ensure the figure is closed
            try:  # Attempt to close the figure if an error occurs during plotting/saving
                plt.close(fig)  # Close the figure to free memory
            except Exception:  # Ignore any exceptions that occur while trying to close the figure
                pass  # Do nothing if closing the figure fails
            raise  # Re-raise the original exception to be caught by the outer block for logging and Telegram alert
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

    fig = None
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
                    label=sanitize_plot_text(f"{cls} ({int(counts.get(cls, 0))})"),  # Sanitize class label before passing to matplotlib
                    s=8,
                )  # 3D scatter plot for class with count in label
            ax.legend(markerscale=2, fontsize="small")  # Add legend for classes
        else:  # No labels provided
            cast(Any, ax).scatter(X_emb[:, 0], X_emb[:, 1], X_emb[:, 2], s=8)  # Plot all points uniformly

        ax.set_title(sanitize_plot_text(title))  # Set sanitized plot title
        ax.set_xlabel(sanitize_plot_text("t-SNE 1"))  # X-axis label sanitized
        ax.set_ylabel(sanitize_plot_text("t-SNE 2"))  # Y-axis label sanitized
        cast(Any, ax).set_zlabel(sanitize_plot_text("t-SNE 3"))  # Z-axis label sanitized (cast to Any for typing)
        plt.tight_layout()  # Adjust layout to fit everything within the figure area
        try:  # Attempt to save the figure and guarantee the figure is closed
            fig.savefig(output_path, dpi=300)  # Save 3D t-SNE figure to disk at 300 DPI
        finally:  # Ensure figure is always closed to free memory
            plt.close(fig)  # Close the figure to free memory
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def generate_and_save_tsne_embeddings(X, labels, base, output_dir, perplexity, n_iter, random_state):
    """
    Compute 2D and 3D t-SNE embeddings for the scaled feature matrix and save both as PNG scatter plots.

    :param X: Scaled 2D NumPy array of feature values to embed.
    :param labels: Optional array-like of class labels used for per-class coloring in the plots.
    :param base: Base filename string derived from the source dataset file for output naming.
    :param output_dir: Directory path where the PNG output files are written.
    :param perplexity: t-SNE perplexity parameter controlling the effective local neighborhood size.
    :param n_iter: Number of optimization iterations for the t-SNE algorithm.
    :param random_state: Integer random seed for reproducible t-SNE results.
    :return: Tuple of (out_name_2d, out_name_3d) filename strings for the two saved PNG outputs.
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Computing 2D t-SNE embedding for: {BackgroundColors.CYAN}{base}{Style.RESET_ALL}"
    )  # Announce 2D t-SNE computation start
    X_emb_2d = initialize_and_fit_tsne(X, n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=random_state)  # Compute the 2D t-SNE embedding
    out_name_2d = f"TSNE_2D_{base}.png"  # Build the 2D output PNG filename
    out_path_2d = os.path.join(output_dir, out_name_2d)  # Construct the full absolute path for the 2D PNG
    save_tsne_plot(X_emb_2d, labels, out_path_2d, f"t-SNE 2D: {base}")  # Create and save the 2D scatter plot
    verbose_output(
        f"{BackgroundColors.GREEN}Saved 2D t-SNE plot to: {BackgroundColors.CYAN}{out_path_2d}{Style.RESET_ALL}"
    )  # Confirm successful 2D plot save

    verbose_output(
        f"{BackgroundColors.GREEN}Computing 3D t-SNE embedding for: {BackgroundColors.CYAN}{base}{Style.RESET_ALL}"
    )  # Announce 3D t-SNE computation start
    X_emb_3d = initialize_and_fit_tsne(X, n_components=3, perplexity=perplexity, n_iter=n_iter, random_state=random_state)  # Compute the 3D t-SNE embedding
    out_name_3d = f"TSNE_3D_{base}.png"  # Build the 3D output PNG filename
    out_path_3d = os.path.join(output_dir, out_name_3d)  # Construct the full absolute path for the 3D PNG
    save_tsne_3d_plot(X_emb_3d, labels, out_path_3d, f"t-SNE 3D: {base}")  # Create and save the 3D scatter plot
    verbose_output(
        f"{BackgroundColors.GREEN}Saved 3D t-SNE plot to: {BackgroundColors.CYAN}{out_path_3d}{Style.RESET_ALL}"
    )  # Confirm successful 3D plot save

    return out_name_2d, out_name_3d  # Return both saved PNG filenames to the caller


def resolve_tsne_output_directory(filepath, output_dir, config):
    """
    Resolve the t-SNE output directory from an explicit path or from the configuration, and create it.

    :param filepath: Path to the source CSV file used to derive the default output directory location.
    :param output_dir: Explicit output directory path when already known; None triggers config-based resolution.
    :param config: Configuration dictionary used to read the data separability subdirectory name; None loads the default.
    :return: Resolved absolute output directory path that is guaranteed to exist on disk.
    """

    if output_dir is None:  # Verify when no explicit directory was provided by the caller
        cfg = config or get_default_config()  # Load the default config when no external config was passed
        tsne_subdir = cfg.get("paths", {}).get("data_separability_subdir", "Data_Separability")  # Read the configured t-SNE subdirectory name with fallback
        output_dir = os.path.join(os.path.dirname(os.path.abspath(filepath)), tsne_subdir)  # Build the output path relative to the dataset's directory
    os.makedirs(output_dir, exist_ok=True)  # Create the output directory and any missing parents if they do not exist
    return output_dir  # Return the resolved and created output directory path


def generate_tsne_plot(
    filepath,
    df=None,
    low_memory=None,
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
            if df is not None:  # Verify whether a pre-loaded DataFrame was provided by the caller
                numeric_df, labels = prepare_numeric_dataset_from_df(
                    df, sample_size=sample_size, random_state=random_state
                )  # Prepare numeric DataFrame from the pre-loaded dataset
            else:  # No pre-loaded DataFrame, load from file path
                numeric_df, labels = prepare_numeric_dataset(
                    filepath, low_memory, sample_size, random_state
                )  # Prepare numeric dataset from disk
            if numeric_df is None:  # Abort if preparation failed
                return None, None  # Indicate failure

            X = scale_features(numeric_df)  # Scale features for t-SNE

            n_rows = X.shape[0]  # Number of rows after downsampling
            if n_rows <= max(3, int(perplexity) + 1):  # Verifies t-SNE feasibility
                return None, None  # Abort if too few samples for t-SNE

            output_dir = resolve_tsne_output_directory(filepath, output_dir, config)  # Resolve output directory from explicit path or config and create it

            base = os.path.splitext(os.path.basename(filepath))[0]  # Base filename
            
            out_name_2d, out_name_3d = generate_and_save_tsne_embeddings(X, labels, base, output_dir, perplexity, n_iter, random_state)  # Compute and save 2D and 3D t-SNE embeddings as PNG plots

            try:  # Try to delete DataFrame to free memory
                del numeric_df  # Free numeric DataFrame
            except Exception:  # Ignore any exceptions during deletion
                pass  # Do nothing
            try:  # Try to delete labels object to free memory
                del labels  # Free labels series/object
            except Exception:  # Ignore any exceptions during deletion
                pass  # Do nothing
            try:  # Try to delete scaled matrix to free memory
                del X  # Free scaled feature matrix
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
    Determine the total number of augmented samples produced for a given original CSV.

    :param original_csv_path: Path to the original CSV dataset file.
    :param config: Optional configuration dictionary for resolving augmentation directory paths.
    :return: Total number of augmented samples found, or 0 when no augmentation is present.
    """
    
    try:  # Wrap full function logic to ensure production-safe monitoring
        p = Path(original_csv_path)  # Convert original CSV path to a Path object

        cfg = {}  # Initialize empty config dict as base
        if config and isinstance(config, dict):  # Verify caller-provided config is a valid dict
            cfg = config  # Use the caller-provided config
        else:  # Load config from disk when caller did not provide one
            cfg_path = Path(__file__).parent / "config.yaml"  # Build path to config.yaml next to this file
            if cfg_path.exists():  # Verify config file exists before attempting read
                try:  # Attempt to load and parse the config file
                    with open(cfg_path, "r", encoding="utf-8") as _f:  # Open config file for reading
                        cfg = yaml.safe_load(_f) or {}  # Parse YAML safely with empty dict fallback
                except Exception:  # Ignore errors when loading config to preserve fallback behavior
                    cfg = {}  # Reset config to empty dict on read failure

        data_aug_dir = cfg.get("paths", {}).get("data_augmentation_dir", "Data_Augmentation")  # Resolve augmentation base directory from config
        data_aug_sample_dir = cfg.get("paths", {}).get("data_augmentation_sample_dir", "Samples")  # Resolve augmentation samples subdirectory from config
        results_suffix = cfg.get("execution", {}).get("results_suffix", "_data_augmented")  # Resolve results suffix from config

        aug_dir = p.parent / data_aug_dir / data_aug_sample_dir  # Build full augmentation directory path using both config variables

        candidate = aug_dir / f"{p.stem}{results_suffix}{p.suffix}"  # Build candidate augmented CSV path using full directory and stem

        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".csv":  # Verify augmented CSV exists, is a file, and has .csv extension
            try:  # Attempt chunked reading to avoid loading the entire augmented CSV into memory
                total_rows = 0  # Initialize row counter for incremental chunk accumulation
                try:  # Attempt chunked UTF-8 read to avoid loading the entire augmented CSV into memory
                    for chunk in pd.read_csv(candidate, encoding="utf-8", chunksize=10000):  # Read fixed-size chunks to limit peak RAM usage per file
                        total_rows += len(chunk)  # Accumulate row count from each individual chunk
                        del chunk  # Release each chunk immediately after counting to free memory
                except UnicodeDecodeError:  # Handle UTF-8 decode failures specifically
                    total_rows = 0  # Reset counter before retrying with fallback encoding
                    try:  # Attempt chunked Latin-1 read as first fallback for legacy CSVs
                        for chunk in pd.read_csv(candidate, encoding="latin1", chunksize=10000):  # Read fixed-size chunks using Latin-1 encoding
                            total_rows += len(chunk)  # Accumulate row count from each individual chunk
                            del chunk  # Release each chunk immediately after counting to free memory
                    except UnicodeDecodeError:  # Handle Latin-1 decode failures specifically
                        total_rows = 0  # Reset counter before retrying with CP1252 encoding
                        for chunk in pd.read_csv(candidate, encoding="cp1252", chunksize=10000):  # Read fixed-size chunks using CP1252 encoding as final fallback
                            total_rows += len(chunk)  # Accumulate row count from each individual chunk
                            del chunk  # Release each chunk immediately after counting to free memory
                return int(total_rows) if total_rows > 0 else 0  # Return total accumulated row count when read successfully
            except Exception as e:  # Raise RuntimeError when augmented CSV cannot be read
                raise RuntimeError(f"Failed to read augmented CSV '{candidate}': {e}")  # Propagate as RuntimeError with context

        return 0  # Return 0 when no augmented CSV was found for this dataset
    except Exception:  # Re-raise all exceptions to preserve original failure semantics
        raise  # Re-raise to preserve original failure semantics


def format_percentage(p: float) -> str:
    """
    Format a float as a percentage string without unnecessary trailing zeros.

    :param p: Float value to format.
    :return: Formatted percentage string.
    """
    
    string = f"{p:.4f}"  # Format with 4 decimal places
    string = string.rstrip("0").rstrip(".") if "." in string else string  # Trim trailing zeros and dot
    return string  # Return the cleaned percentage string


def build_class_distribution_string(counts: "pd.Series") -> str:
    """
    Build a dictionary-like class distribution string from a pandas Series of counts.

    The returned string follows the exact format used in dataset reports:
    "{ClassA: 100 (62.5%), ClassB: 50 (31.25%), ClassC: 10 (6.25%)}"

    :param counts: pandas Series where index are class labels and values are counts.
    :return: Formatted distribution string or None when counts is empty/None.
    """

    try:
        if counts is None or counts.empty:  # Return empty string when counts is missing or empty
            return ""  # No distribution can be built

        total = int(counts.sum())  # Compute total number of samples from counts
        if total == 0:  # Guard against division by zero when total is zero
            return ""  # No distribution to build for zero total

        counts_sorted = counts.sort_values(ascending=False)  # Sort classes by count descending

        parts = []  # Initialize list to accumulate formatted class parts
        for cls, cnt in counts_sorted.items():  # Iterate over (class, count) pairs in sorted order
            cls_key = str(cls)  # Convert class label to string for stable formatting
            pct = (float(cnt) / float(total)) * 100.0  # Compute percentage of total for this class
            pct_str = format_percentage(pct)  # Format percentage to trimmed string representation
            parts.append(f"{cls_key}: {int(cnt)} ({pct_str}%)")  # Append formatted entry to parts list

        return "{" + ", ".join(parts) + "}" if parts else ""  # Join parts into final dictionary-like string or return empty string
    except Exception as e:
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise  # Re-raise to preserve original failure semantics


def extract_classes_and_distribution(df: "pd.DataFrame") -> tuple:
    """
    Identify label column and extract classes and their distribution.

    :param df: pandas DataFrame to inspect for label column and classes.
    :return: Tuple (label_col_or_None, classes_str_or_None, class_dist_str_or_None).
    """

    try:  # Guard logic with module-standard exception handling
        if df is None or df.empty:  # Handle empty DataFrame edge case
            return "", "", ""  # No label and no class info

        found_col = detect_label_column(df.columns)  # Use existing detector to find label column when possible

        if not found_col:  # If detector failed, fallback heuristics
            last_col = df.columns[-1] if len(df.columns) > 0 else ""  # Determine last column when available
            if last_col and (
                pd.api.types.is_object_dtype(df[last_col].dtype)
                or getattr(pd.api.types, "is_categorical_dtype", lambda x: False)(df[last_col].dtype)
                or pd.api.types.is_bool_dtype(df[last_col].dtype)
            ):  # Accept last column as label when non-numeric
                found_col = last_col  # Use last column as fallback label

        if not found_col:  # When still no label candidate found
            return "", "", ""  # No class information available

        series = df[found_col]  # Reference label series directly to avoid copy
        counts = series.value_counts(dropna=False)  # Counts including NaN as a key when present
        total = int(counts.sum())  # Total samples as int

        if total == 0:  # Handle degenerate case with zero total
            return found_col, "", ""  # Return label column but no class info

        counts_sorted = counts.sort_values(ascending=False)  # Ensure highest-to-lowest ordering

        classes_list = [str(x) for x in counts_sorted.index.tolist()]  # Convert index values to strings
        classes_str = ", ".join(classes_list) if classes_list else ""  # Comma-separated classes string or empty string

        class_dist_str = build_class_distribution_string(counts_sorted)  # Build class distribution string in the exact format used in reports

        return found_col, classes_str, class_dist_str  # Return detected label col and formatted class info
    except Exception as e:  # Preserve exception reporting pattern
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def get_process_cpu_usage(process: psutil.Process) -> tuple[int, int, float]:
    """
    Compute CPU usage and logical CPU count for a given process.

    :param process: psutil process instance.
    :return: Tuple containing (used_cpu_cores, total_cpu_cores, cpu_percent)
    """

    cpu_times = process.cpu_times()  # Retrieve CPU time statistics for process

    cpu_percent = process.cpu_percent(interval=0.1)  # Compute CPU usage over short sampling interval

    total_cpu_cores = os.cpu_count() or 1  # Detect total system logical CPU cores

    cpu_affinity = process.cpu_affinity() if hasattr(process, "cpu_affinity") else None  # Retrieve CPU affinity if available

    used_cpu_cores = len(cpu_affinity) if cpu_affinity else total_cpu_cores  # Estimate active/assigned cores or fallback to total

    return used_cpu_cores, total_cpu_cores, cpu_percent  # Return CPU usage metrics


def format_bytes_to_best_unit(byte_value: int) -> tuple[float, str]:
    """
    Convert a byte value into the most appropriate human-readable unit.

    :param byte_value: Size in bytes to be converted.
    :return: Tuple containing the converted value and its unit label.
    """

    if byte_value < 0:  # Verify if byte value is negative before conversion
        byte_value = 0  # Normalize negative values to zero to avoid invalid memory representation

    kb_value = byte_value / 1024  # Compute kilobyte representation
    mb_value = kb_value / 1024  # Compute megabyte representation
    gb_value = mb_value / 1024  # Compute gigabyte representation
    tb_value = gb_value / 1024  # Compute terabyte representation

    if tb_value >= 1:  # Verify if value fits into terabyte range
        return tb_value, "TB"  # Return terabyte formatted value and unit

    if gb_value >= 1:  # Verify if value fits into gigabyte range
        return gb_value, "GB"  # Return gigabyte formatted value and unit

    if mb_value >= 1:  # Verify if value fits into megabyte range
        return mb_value, "MB"  # Return megabyte formatted value and unit

    if kb_value >= 1:  # Verify if value fits into kilobyte range
        return kb_value, "KB"  # Return kilobyte formatted value and unit

    return float(byte_value), "B"  # Return raw bytes when below kilobyte threshold


def report_resources_usage(stage: str, filepath: str) -> tuple[int, int, float, int, int, float]:
    """
    Report and return process memory usage, system RAM percentage, CPU core usage, and CPU utilization.

    Prints a verbose output line with CPU usage and memory usage for the given stage and file.

    :param stage: Description of the measurement stage (e.g., "Before loading").
    :param filepath: Path to the file being processed (for output context).
    :return: Tuple (rss_bytes, total_bytes, percent_used, used_cpu_cores, total_cpu_cores, cpu_percent)
    """

    process = psutil.Process(os.getpid())  # Get current process

    mem_bytes = process.memory_info().rss  # Resident Set Size in bytes

    total_bytes = psutil.virtual_memory().total  # Total system RAM in bytes

    percent = (mem_bytes / total_bytes) * 100 if total_bytes else 0.0  # Compute percent of RAM used

    mem_value, mem_unit = format_bytes_to_best_unit(mem_bytes)  # Convert memory usage to best unit

    used_cpu_cores, total_cpu_cores, cpu_percent = get_process_cpu_usage(process)  # Retrieve CPU metrics

    verbose_output(
        f"{BackgroundColors.YELLOW}[MEMORY]{Style.RESET_ALL} {stage} {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}: "
        f"{cpu_percent:.2f}% CPU using {used_cpu_cores}/{total_cpu_cores} cores | "
        f"{mem_value:.2f} {mem_unit} ({percent:.4f}%) RAM."
    )  # Print CPU first, then memory usage with structured format

    return mem_bytes, total_bytes, percent, used_cpu_cores, total_cpu_cores, cpu_percent  # Return full resource metrics


def get_dataset_file_info(filepath, df=None, low_memory=None):
    """
    Extract dataset information from a CSV file and return it as a dictionary.

    :param filepath: Path to the CSV file.
    :param df: Optional pre-loaded pandas DataFrame to avoid redundant disk reads.
    :param low_memory: Whether to use low memory mode when loading the CSV (default: True).
    :return: Dictionary containing dataset information.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Extracting dataset information from: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}"
        )  # Output start message for dataset info extraction
        send_telegram_message(TELEGRAM_BOT, [f"Extracting dataset information from: {os.path.basename(filepath)}"])  # Send Telegram notification indicating start of dataset info extraction
        
        mem_before, total_mem, percent_before, used_cores_before, total_cores_before, cpu_percent_before = report_resources_usage(f"Before Loading Dataset File", filepath)  # Capture full resource state before dataset load
        
        if df is None:
            df = load_dataset(filepath, low_memory)  # Load the dataset

        if df is None:  # If the dataset could not be loaded
            return None  # Return None
        
        mem_after, total_mem_after, percent_after, used_cores_after, total_cores_after, cpu_percent_after = report_resources_usage(f"After Loading Dataset File", filepath)  # Capture full resource state after dataset load

        original_num_rows = len(df)  # Capture original number of rows immediately after read
        original_num_features = df.shape[1] if hasattr(df, "shape") else 0  # Capture original feature count

        nan_mask = df.isna().any(axis=1)  # Build boolean mask for rows containing any NaN/null values
        rows_after_nan_removal = int((~nan_mask).sum())  # Capture rows remaining after NaN/null filtering via mask count
        removed_rows_nan = original_num_rows - rows_after_nan_removal  # Compute removed row count due to NaN/null filtering
        removed_rows_nan = removed_rows_nan if removed_rows_nan >= 0 else 0  # Clamp negative values to zero for safety
        if original_num_rows > 0:  # Verify original row count is non-zero before percentage division
            removed_rows_nan_proportion = round(removed_rows_nan / float(original_num_rows), 6)  # Compute rounded removed-row proportion for NaN/null filtering
        else:  # Handle zero-row datasets without division
            removed_rows_nan_proportion = 0.0  # Set removed-row proportion to zero when no rows are present

        numeric_cols_after_nan = df.select_dtypes(include=["number"]).columns  # Identify numeric columns to detect infinite values after NaN removal
        if len(numeric_cols_after_nan) > 0:  # Only compute infinite masks when numeric columns exist
            inf_mask_after_nan = df[numeric_cols_after_nan].isin([np.inf, -np.inf]).any(axis=1)  # Identify rows with any +/-inf in numeric columns
        else:  # When no numeric columns exist
            inf_mask_after_nan = pd.Series([False] * len(df), index=df.index)  # Create false mask to avoid errors
        valid_nan_mask = ~nan_mask  # Build non-NaN mask to reuse across metric computations
        valid_inf_mask = ~inf_mask_after_nan  # Build non-infinite mask to reuse across metric computations
        valid_nan_inf_mask = valid_nan_mask & valid_inf_mask  # Combine masks to represent rows surviving NaN and infinite filtering
        rows_after_nan_inf_removal = int(valid_nan_inf_mask.sum())  # Capture rows remaining after NaN+infinite filtering
        removed_rows_inf = rows_after_nan_removal - rows_after_nan_inf_removal  # Compute rows removed due to infinite filtering specifically
        rows_after_inf_removal = rows_after_nan_inf_removal  # Record rows after infinite removal (applied after NaN removal)
        removed_rows_nan_inf = original_num_rows - rows_after_nan_inf_removal  # Compute total rows removed by NaN+infinite filtering combined
        removed_rows_nan_inf = removed_rows_nan_inf if removed_rows_nan_inf >= 0 else 0  # Clamp negative values to zero for safety
        if original_num_rows > 0:  # Verify original row count is non-zero before percentage division
            removed_rows_nan_inf_proportion = round(removed_rows_nan_inf / float(original_num_rows), 6)  # Compute rounded removed-row proportion for NaN+infinite filtering
        else:  # Handle zero-row datasets without division
            removed_rows_nan_inf_proportion = 0.0  # Set removed-row proportion to zero when no rows are present

        numeric_cols_after_nan_inf = numeric_cols_after_nan  # Reuse numeric columns for zero-variance analysis after NaN+inf removals
        zero_var_cols = []  # Initialize zero-variance feature list
        if len(numeric_cols_after_nan_inf) > 0:  # Verify numeric features are available before variance computation
            df_after_nan_inf_numeric = df.loc[valid_nan_inf_mask, numeric_cols_after_nan_inf]  # Materialize filtered numeric frame only for variance computation
            variances_after_nan_inf = df_after_nan_inf_numeric.var(axis=0, ddof=0)  # Compute variance for each numeric feature after NaN+inf filtering
            zero_var_cols = variances_after_nan_inf[variances_after_nan_inf == 0].index.tolist()  # Collect numeric features with zero variance
            del df_after_nan_inf_numeric  # Release filtered numeric frame immediately after variance computation
            del variances_after_nan_inf  # Release variance series immediately after extracting zero-variance column names
        removed_zero_variance_features = len(zero_var_cols)  # Capture number of removed zero-variance numerical features
        if original_num_features > 0:  # Verify original feature count is non-zero before percentage division
            removed_zero_variance_features_proportion = round(removed_zero_variance_features / float(original_num_features), 6)  # Compute rounded removed-feature proportion for zero-variance removal
        else:  # Handle zero-feature datasets without division
            removed_zero_variance_features_proportion = 0.0  # Set removed-feature proportion to zero when no features are present

        dropped_non_informative_features = 0  # Preserve current behavior because non-informative identifier/metadata dropping is not applied in this module
        if original_num_features > 0:  # Verify original feature count is non-zero before percentage division
            dropped_non_informative_features_proportion = round(dropped_non_informative_features / float(original_num_features), 6)  # Compute rounded dropped-feature proportion for identifier/metadata step
        else:  # Handle zero-feature datasets without division
            dropped_non_informative_features_proportion = 0.0  # Set dropped-feature proportion to zero when no features are present

        features_after_zero_variance_removal = int(df.shape[1]) - int(removed_zero_variance_features)  # Compute feature count after zero-variance removal using original feature count baseline
        features_after_zero_variance_removal = features_after_zero_variance_removal if features_after_zero_variance_removal >= 0 else 0  # Clamp negative values to zero for safety

        cleaned_df = preprocess_dataframe(df)  # Preprocess the DataFrame
        del df  # Release original DataFrame local reference to reduce memory footprint after preprocessing produces the cleaned copy

        rows_after_preprocessing = len(cleaned_df)  # Capture rows after preprocessing
        features_after_preprocessing = cleaned_df.shape[1] if hasattr(cleaned_df, "shape") else 0  # Capture features after preprocessing
        preprocessing_step_metrics = cleaned_df.attrs.get("preprocessing_metrics", {}) if hasattr(cleaned_df, "attrs") else {}  # Capture structured preprocessing step metrics from DataFrame metadata

        label_col, classes_str, class_dist_str = extract_classes_and_distribution(cleaned_df)  # Identify label column and extract classes/distribution
        n_samples, n_features, n_numeric, n_int, n_categorical, n_other, categorical_cols_str = summarize_features(
            cleaned_df
        )  # Summarize features
        missing_summary = summarize_missing_values(cleaned_df)  # Summarize missing values

        num_labels, labels_list = extract_labels_info(cleaned_df)  # Extract number of unique labels and the sorted labels list

        labels_list_str = format_labels_list(labels_list)  # Format labels list into stable string for CSV inclusion

        feature_view_df = cleaned_df.drop(columns=[label_col], errors="ignore") if label_col else cleaned_df  # Build feature-only frame by excluding the label column when available
        numeric_feature_view = feature_view_df.select_dtypes(include=["number"])  # Extract numeric features for cast-to-float64/int64 accounting
        categorical_feature_view = feature_view_df.select_dtypes(exclude=["number"])  # Extract non-numeric features for categorical encoding accounting
        features_cast_to_float64_int64 = int((~numeric_feature_view.dtypes.isin([np.dtype("float64"), np.dtype("int64")])).sum()) if not numeric_feature_view.empty else 0  # Count numeric features whose dtypes differ from float64/int64
        features_encoded_categorical = int(categorical_feature_view.shape[1])  # Count categorical features that require ordinal or one-hot encoding
        features_transformed_for_experiment = int(features_cast_to_float64_int64) + int(features_encoded_categorical)  # Compute total transformed feature count for experiment-level preprocessing
        if features_after_preprocessing > 0:  # Verify post-preprocessing feature count is non-zero before percentage division
            features_transformed_for_experiment_proportion = round(features_transformed_for_experiment / float(features_after_preprocessing), 6)  # Compute rounded transformed-feature proportion after preprocessing
        else:  # Handle zero-feature datasets without division
            features_transformed_for_experiment_proportion = 0.0  # Set transformed-feature proportion to zero when no features are present

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
            "Number of Labels": f"{num_labels:,}",  # Format with commas for readability for label count
            "Labels List": labels_list_str,  # Stable string representation of detected labels
            "original_num_rows": original_num_rows,  # Rows immediately after reading CSV
            "rows_after_nan_removal": rows_after_nan_removal,  # Rows remaining after removing NaN/null values only
            "removed_rows_nan": removed_rows_nan,  # Rows removed by NaN/null filtering step
            "removed_rows_nan_proportion": removed_rows_nan_proportion,  # Proportion of rows removed by NaN/null filtering step
            "rows_after_inf_removal": rows_after_inf_removal,  # Rows remaining after removing infinite values only
            "removed_rows_inf": removed_rows_inf,  # Rows removed by infinite-value filtering step
            "rows_after_nan_inf_removal": rows_after_nan_inf_removal,  # Rows remaining after removing NaN+infinite values
            "removed_rows_nan_inf": removed_rows_nan_inf,  # Rows removed by combined NaN+infinite filtering step
            "removed_rows_nan_inf_proportion": removed_rows_nan_inf_proportion,  # Proportion of rows removed by combined NaN+infinite filtering step
            "rows_after_preprocessing": rows_after_preprocessing,  # Rows after preprocessing
            "original_num_features": original_num_features,  # Features before preprocessing
            "features_after_zero_variance_removal": features_after_zero_variance_removal,  # Features remaining after zero-variance numerical feature removal
            "removed_zero_variance_features": removed_zero_variance_features,  # Zero-variance numerical features removed in preprocessing
            "removed_zero_variance_features_proportion": removed_zero_variance_features_proportion,  # Proportion of zero-variance numerical features removed
            "features_after_preprocessing": features_after_preprocessing,  # Features after preprocessing
            "dropped_non_informative_features": dropped_non_informative_features,  # Non-informative identifier/metadata features removed in this module
            "dropped_non_informative_features_proportion": dropped_non_informative_features_proportion,  # Proportion of non-informative identifier/metadata features removed
            "features_transformed_for_experiment": features_transformed_for_experiment,  # Features transformed for dtype enforcement and categorical encoding per experiment
            "features_transformed_for_experiment_proportion": features_transformed_for_experiment_proportion,  # Proportion of transformed features for dtype enforcement and categorical encoding per experiment
            "features_cast_to_float64_int64": features_cast_to_float64_int64,  # Numeric features requiring cast to float64/int64
            "features_encoded_categorical": features_encoded_categorical,  # Categorical features requiring ordinal or one-hot encoding
            "preprocessing_metrics": preprocessing_step_metrics,  # Structured per-step preprocessing metrics for isolated CSV mapping
            "Feature Types": f"{n_numeric} numeric (float64), {n_int} integer (int64), {n_categorical} categorical (object/category/bool/string), {n_other} other",
            "Categorical Features (object/string)": categorical_cols_str,
            "Missing Values": missing_summary,
            "Classes": classes_str,
            "Class Distribution": class_dist_str,
        }

        try:  # Attempt to retrieve augmented sample count for this dataset
            aug_count = get_augmented_sample_count(filepath, None)  # Get augmented sample count from WGANGP output directory
        except Exception:  # Re-raise on failure to preserve original semantics
            raise  # Re-raise to preserve original failure semantics
        result["data_augmentation_samples"] = int(aug_count)  # Store integer augmented sample count in result dict

        del cleaned_df  # Release preprocessed DataFrame to reduce memory footprint after all metrics are extracted
        del nan_mask  # Release NaN mask to reduce retained memory after metrics extraction
        del inf_mask_after_nan  # Release infinite-value mask to reduce retained memory after metrics extraction
        del valid_nan_mask  # Release non-NaN mask to reduce retained memory after metrics extraction
        del valid_inf_mask  # Release non-infinite mask to reduce retained memory after metrics extraction
        del valid_nan_inf_mask  # Release combined valid-row mask to reduce retained memory after metrics extraction
        del feature_view_df  # Release feature-view DataFrame reference after derived metrics are computed
        del numeric_feature_view  # Release numeric feature view reference after dtype accounting
        del categorical_feature_view  # Release categorical feature view reference after dtype accounting
        gc.collect()  # Trigger garbage collection after releasing large intermediate objects

        verbose_output(f"{BackgroundColors.GREEN}Finished processing dataset: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}")  # Print message indicating completion of processing for dataset
        send_telegram_message(TELEGRAM_BOT, [f"Finished processing dataset: {os.path.basename(filepath)}"])  # Send Telegram notification indicating completion of processing for dataset
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


def reorder_report_columns(report_df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Reorder report DataFrame columns to place preprocessing columns at the end.

    :param report_df: DataFrame to reorder.
    :return: Reordered DataFrame.
    """

    try:  # Guard the helper with same exception handling pattern used across module
        preprocessing_keys = [  # Known preprocessing-related keys to move to the end when present
            "original_num_rows",
            "rows_after_nan_removal",
            "removed_rows_nan",
            "removed_rows_nan_proportion",
            "rows_after_inf_removal",
            "removed_rows_inf",
            "rows_after_nan_inf_removal",
            "removed_rows_nan_inf",
            "removed_rows_nan_inf_proportion",
            "rows_after_preprocessing",
            "original_num_features",
            "features_after_zero_variance_removal",
            "removed_zero_variance_features",
            "removed_zero_variance_features_proportion",
            "features_after_preprocessing",
            "dropped_non_informative_features",
            "dropped_non_informative_features_proportion",
            "features_transformed_for_experiment",
            "features_transformed_for_experiment_proportion",
            "features_cast_to_float64_int64",
            "features_encoded_categorical",
            "preprocessing_metrics",
        ]  # End preprocessing key list

        desired_front = [  # Desired primary header order before preprocessing columns
            "#",
            "Dataset Name",
            "Size (GB)",
            "Number of Samples",
            "Number of Features",
            "Feature Types",
            "Categorical Features (object/string)",
            "Missing Values",
            "Classes",
            "Class Distribution",
            "data_augmentation_samples",
            "Headers Match All Files",
            "Common Features (in all files)",
            "Extra Features (not in all files)",
            "t-SNE Plot",
        ]  # End desired front columns

        ordered_cols: list[str] = []  # Initialize ordered columns accumulator
        for column in desired_front:  # Iterate desired front list
            if column in report_df.columns and column not in ordered_cols:  # Add when present and not duplicated
                ordered_cols.append(column)  # Append desired front column when found

        for column in report_df.columns:  # Preserve original column discovery order for remaining non-preprocessing fields
            if column not in ordered_cols and column not in preprocessing_keys:  # Only include non-preprocessing and not-yet-added columns
                ordered_cols.append(column)  # Append the remaining non-preprocessing column

        for column in preprocessing_keys:  # Iterate known preprocessing keys to place them at the end
            if column in report_df.columns and column not in ordered_cols:  # Add only when present and not already added
                ordered_cols.append(column)  # Append preprocessing column

        if ordered_cols:  # If we built an ordering
            return report_df[[column for column in ordered_cols if column in report_df.columns]]  # Return reordered DataFrame
        return report_df  # Return original DataFrame when no ordering was computed
    except Exception as e:  # Preserve module-wide exception handling semantics
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def write_report(report_rows, base_dir, output_filename, config: dict | None = None):
    """
    Write the report rows to a CSV file.

    :param report_rows: List of dictionaries containing report data.
    :param base_dir: Base directory for saving the report.
    :param output_filename: Name of the output CSV file.
    :param config: Optional configuration dictionary for resolving output subdirectory.
    :return: None.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        report_df = pd.DataFrame(report_rows)  # Create a DataFrame from the report rows
        report_df = reorder_report_columns(report_df)  # Reorder DataFrame columns to place preprocessing metrics at the end for better readability
        cfg = config or get_default_config()
        results_subdir = cfg.get("paths", {}).get("dataset_description_subdir", "Dataset_Description")
        results_dir = os.path.join(base_dir, results_subdir)
        os.makedirs(results_dir, exist_ok=True)
        report_csv_path = os.path.join(results_dir, output_filename)
        generate_csv_and_image(report_df, report_csv_path, config=cfg)
        pass  # No-op here; preprocessing summary is handled by the caller
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def collect_preprocessing_metrics(
    filepath,
    original_num_rows,
    rows_after_preprocessing,
    original_num_features,
    features_after_preprocessing,
    rows_after_nan_inf_removal=0,
    removed_rows_nan_inf=0,
    removed_rows_nan_inf_proportion=0.0,
    features_after_zero_variance_removal=0,
    removed_zero_variance_features=0,
    removed_zero_variance_features_proportion=0.0,
    dropped_non_informative_features=0,
    dropped_non_informative_features_proportion=0.0,
    features_transformed_for_experiment=0,
    features_transformed_for_experiment_proportion=0.0,
    features_cast_to_float64_int64=0,
    features_encoded_categorical=0,
    preprocessing_step_metrics=None,
):
    """
    Collect preprocessing metrics for a single file and return a dict matching the required CSV schema.

    :param filepath: Path to the processed CSV file
    :param original_num_rows: Number of rows immediately after reading the CSV
    :param rows_after_preprocessing: Number of rows after preprocessing steps
    :param original_num_features: Number of features before preprocessing
    :param features_after_preprocessing: Number of features after preprocessing
    :param rows_after_nan_inf_removal: Number of rows remaining after removing NaN and infinite rows.
    :param removed_rows_nan_inf: Number of rows removed by NaN/infinite filtering.
    :param removed_rows_nan_inf_proportion: Proportion of rows removed by NaN/infinite filtering.
    :param features_after_zero_variance_removal: Number of features remaining after zero-variance removal.
    :param removed_zero_variance_features: Number of zero-variance numerical features removed.
    :param removed_zero_variance_features_proportion: Proportion of zero-variance numerical features removed.
    :param dropped_non_informative_features: Number of non-informative identifier/metadata features dropped.
    :param dropped_non_informative_features_proportion: Proportion of non-informative identifier/metadata features dropped.
    :param features_transformed_for_experiment: Number of features transformed for experiment encoding/casting.
    :param features_transformed_for_experiment_proportion: Proportion of features transformed for experiment encoding/casting.
    :param features_cast_to_float64_int64: Number of numeric features that require casting to float64/int64.
    :param features_encoded_categorical: Number of categorical features that require ordinal/one-hot encoding.
    :return: Dict with keys matching preprocessing_summary.csv columns.
    """

    try:  # Wrap logic to preserve existing error handling conventions
        filename = os.path.basename(filepath)  # Extract filename from filepath
        original_num_rows = int(original_num_rows) if original_num_rows is not None else 0  # Normalize original row count to integer
        rows_after_preprocessing = int(rows_after_preprocessing) if rows_after_preprocessing is not None else 0  # Normalize final row count to integer
        original_num_features = int(original_num_features) if original_num_features is not None else 0  # Normalize original feature count to integer
        features_after_preprocessing = int(features_after_preprocessing) if features_after_preprocessing is not None else 0  # Normalize final feature count to integer
        step_metrics = preprocessing_step_metrics if isinstance(preprocessing_step_metrics, dict) else {}  # Resolve optional structured per-step metrics container
        nan_inf_metrics = step_metrics.get("nan_inf", {}) if isinstance(step_metrics.get("nan_inf", {}), dict) else {}  # Resolve NaN+infinite step metrics from structured container
        zero_variance_metrics = step_metrics.get("zero_variance", {}) if isinstance(step_metrics.get("zero_variance", {}), dict) else {}  # Resolve zero-variance step metrics from structured container
        final_metrics = step_metrics.get("final", {}) if isinstance(step_metrics.get("final", {}), dict) else {}  # Resolve final aggregated metrics from structured container

        rows_after_nan_inf_value = nan_inf_metrics.get("rows_after_step", rows_after_nan_inf_removal)  # Resolve rows after NaN+infinite filtering from step metrics with fallback
        rows_after_nan_inf_value = int(rows_after_nan_inf_value) if rows_after_nan_inf_value is not None else 0  # Normalize rows after NaN+infinite filtering to integer
        removed_rows_nan_inf_value = nan_inf_metrics.get("removed_rows_step", removed_rows_nan_inf)  # Resolve removed rows for NaN+infinite filtering from step metrics with fallback
        removed_rows_nan_inf_value = int(removed_rows_nan_inf_value) if removed_rows_nan_inf_value is not None else 0  # Normalize removed rows for NaN+infinite filtering to integer
        removed_rows_nan_inf_value = removed_rows_nan_inf_value if removed_rows_nan_inf_value >= 0 else 0  # Clamp negative removed rows for NaN+infinite filtering to zero
        removed_rows_nan_inf_proportion_value = nan_inf_metrics.get("removed_rows_step_proportion", removed_rows_nan_inf_proportion)  # Resolve NaN+infinite removed-row proportion from step metrics with fallback
        removed_rows_nan_inf_proportion_value = round(float(removed_rows_nan_inf_proportion_value), 6) if removed_rows_nan_inf_proportion_value is not None else 0.0  # Normalize NaN+infinite removed-row proportion

        features_after_zero_variance_value = zero_variance_metrics.get("features_after_step", features_after_zero_variance_removal)  # Resolve features after zero-variance removal from step metrics with fallback
        features_after_zero_variance_value = int(features_after_zero_variance_value) if features_after_zero_variance_value is not None else 0  # Normalize features after zero-variance removal to integer
        removed_zero_variance_features_value = zero_variance_metrics.get("removed_features_step", removed_zero_variance_features)  # Resolve removed zero-variance features from step metrics with fallback
        removed_zero_variance_features_value = int(removed_zero_variance_features_value) if removed_zero_variance_features_value is not None else 0  # Normalize removed zero-variance features to integer
        removed_zero_variance_features_value = removed_zero_variance_features_value if removed_zero_variance_features_value >= 0 else 0  # Clamp negative removed zero-variance features to zero
        removed_zero_variance_features_proportion_value = zero_variance_metrics.get("removed_features_step_proportion", removed_zero_variance_features_proportion)  # Resolve zero-variance removed-feature proportion from step metrics with fallback
        removed_zero_variance_features_proportion_value = round(float(removed_zero_variance_features_proportion_value), 6) if removed_zero_variance_features_proportion_value is not None else 0.0  # Normalize zero-variance removed-feature proportion

        removed_rows = final_metrics.get("removed_rows_step", original_num_rows - rows_after_preprocessing)  # Resolve total removed rows from final step metrics with fallback
        removed_rows = int(removed_rows) if removed_rows is not None else 0  # Normalize total removed rows to integer
        removed_rows = removed_rows if removed_rows >= 0 else 0  # Clamp negative total removed rows to zero for safety
        if original_num_rows > 0:  # Guard division by zero for total removed-row proportion
            removed_rows_proportion = round(float(removed_rows) / float(original_num_rows), 6)  # Compute total removed-row proportion from normalized values
        else:  # Handle zero-row datasets without division
            removed_rows_proportion = 0.0  # Set total removed-row proportion to zero when no rows are present

        removed_features = original_num_features - features_after_preprocessing  # Compute total removed features count
        removed_features = int(removed_features) if removed_features is not None else 0  # Normalize total removed features to integer
        removed_features = removed_features if removed_features >= 0 else 0  # Clamp negative total removed features to zero for safety
        if original_num_features > 0:  # Guard division by zero for total removed-feature proportion
            removed_features_proportion = round(float(removed_features) / float(original_num_features), 6)  # Compute total removed-feature proportion from normalized values
        else:  # Handle zero-feature datasets without division
            removed_features_proportion = 0.0  # Set total removed-feature proportion to zero when no features are present

        return {  # Return metrics dict matching required output columns and order
            "filename": filename,  # Base filename
            "original_num_rows": int(original_num_rows),  # Cast to int for CSV
            "rows_after_nan_inf_removal": int(rows_after_nan_inf_value),  # Persist rows after NaN+infinite filtering only
            "removed_rows_nan_inf": int(removed_rows_nan_inf_value),  # Persist removed rows from NaN+infinite filtering only
            "removed_rows_nan_inf_proportion": float(removed_rows_nan_inf_proportion_value),  # Persist removed-row proportion from NaN+infinite filtering only
            "rows_after_preprocessing": int(rows_after_preprocessing),  # Cast to int
            "removed_rows": int(removed_rows),  # Cast to int
            "removed_rows_proportion": float(removed_rows_proportion),  # Float rounded to 6 decimals
            "original_num_features": int(original_num_features),  # Cast to int
            "features_after_zero_variance_removal": int(features_after_zero_variance_value),  # Persist features after zero-variance filtering only
            "removed_zero_variance_features": int(removed_zero_variance_features_value),  # Persist removed zero-variance features only
            "removed_zero_variance_features_proportion": float(removed_zero_variance_features_proportion_value),  # Persist removed-feature proportion from zero-variance filtering only
            "features_after_preprocessing": int(features_after_preprocessing),  # Cast to int
            "removed_features": int(removed_features),  # Cast to int
            "removed_features_proportion": float(removed_features_proportion),  # Float rounded to 6 decimals
            "dropped_non_informative_features": int(dropped_non_informative_features),  # Cast to int for CSV
            "dropped_non_informative_features_proportion": float(dropped_non_informative_features_proportion),  # Float rounded to 6 decimals
            "features_transformed_for_experiment": int(features_transformed_for_experiment),  # Cast to int for CSV
            "features_transformed_for_experiment_proportion": float(features_transformed_for_experiment_proportion),  # Float rounded to 6 decimals
            "features_cast_to_float64_int64": int(features_cast_to_float64_int64),  # Cast to int for CSV
            "features_encoded_categorical": int(features_encoded_categorical),  # Cast to int for CSV
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
            "rows_after_nan_inf_removal",
            "removed_rows_nan_inf",
            "removed_rows_nan_inf_proportion",
            "rows_after_preprocessing",
            "removed_rows",
            "removed_rows_proportion",
            "original_num_features",
            "features_after_zero_variance_removal",
            "removed_zero_variance_features",
            "removed_zero_variance_features_proportion",
            "features_after_preprocessing",
            "removed_features",
            "removed_features_proportion",
            "dropped_non_informative_features",
            "dropped_non_informative_features_proportion",
            "features_transformed_for_experiment",
            "features_transformed_for_experiment_proportion",
            "features_cast_to_float64_int64",
            "features_encoded_categorical",
        ]  # Define exact column order required by spec

        df = pd.DataFrame(metrics_list)  # Create DataFrame from provided metrics list
        for c in cols:  # Ensure all expected columns exist in DataFrame
            if c not in df.columns:  # If missing column
                df[c] = None  # Add column filled with None to preserve schema
        df = df[cols]  # Reorder columns to the required fixed order

        if not df.empty:  # Verify that there is at least one dataset row before computing averages
            numeric_cols = [c for c in cols if c != "filename"]  # Build numeric columns list by excluding filename
            avg_row: dict[str, Any] = {"filename": "AVERAGE"}  # Initialize the average row with a fixed label and explicit flexible value types
            for c in numeric_cols:  # Iterate over all numeric columns that require an average value
                series_num = pd.to_numeric(df[c], errors="coerce")  # Convert each column to numeric while coercing invalid values
                avg_row[c] = round(float(series_num.mean()), 6) if series_num.notna().any() else None  # Compute rounded mean only when valid values exist
            df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)  # Append the average row as the final record

        return df  # Return the prepared DataFrame
    except Exception as e:  # Preserve exception handling
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def save_preprocessing_summary_csv(df, base_dir, filename="preprocessing_summary.csv", config: dict | None = None):
    """
    Save the preprocessing summary DataFrame to the results directory for the given base_dir.

    :param df: DataFrame produced by `build_preprocessing_summary_dataframe`.
    :param base_dir: Base directory where dataset results are stored.
    :param filename: Output CSV filename (default: preprocessing_summary.csv).
    :param config: Optional configuration dictionary for resolving the output subdirectory.
    :return: Absolute path to the saved CSV file.
    """

    try:  # Wrap function body for robust error reporting per module conventions
        cfg = config or get_default_config()
        results_subdir = cfg.get("paths", {}).get("dataset_description_subdir", "Dataset_Description")
        results_dir = os.path.join(base_dir, results_subdir)
        if not verify_filepath_exists(results_dir):
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
            "rows_after_nan_inf_removal",
            "removed_rows_nan_inf",
            "removed_rows_nan_inf_proportion",
            "rows_after_preprocessing",
            "removed_rows",
            "removed_rows_proportion",
            "original_num_features",
            "features_after_zero_variance_removal",
            "removed_zero_variance_features",
            "removed_zero_variance_features_proportion",
            "features_after_preprocessing",
            "removed_features",
            "removed_features_proportion",
            "dropped_non_informative_features",
            "dropped_non_informative_features_proportion",
            "features_transformed_for_experiment",
            "features_transformed_for_experiment_proportion",
            "features_cast_to_float64_int64",
            "features_encoded_categorical",
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

def stripe(row):
    """
    Apply zebra striping to a DataFrame row for styling.

    :param row: pandas Series representing a DataFrame row.
    :return: List of CSS styles for each cell in the row to achieve zebra striping.
    """

    return [
        "background-color: #ffffff" if row.name % 2 == 0 else "background-color: #f2f2f2"
        for _ in row
    ]  # Return alternating background colors per column based on row index parity
    
            
def apply_zebra_style(df):
    """
    Apply zebra-striping pandas Styler to the provided DataFrame.

    :param df: pandas.DataFrame to style
    :return: pandas.Styler with zebra styling applied
    """

    try:  # Wrap function body for consistent error handling
        sanitized_df = df.copy()  # Make a shallow copy to avoid mutating caller DataFrame
        sanitized_df.columns = [sanitize_plot_text(str(c)) for c in sanitized_df.columns]  # Sanitize all column names to safe UTF-8
        try:  # Attempt to sanitize index labels when present to avoid glyph issues in table exports
            sanitized_df.index = sanitized_df.index.map(lambda x: sanitize_plot_text(str(x)) if pd.notnull(x) else x)  # Sanitize index entries
        except Exception:  # Ignore index sanitization errors to preserve original behavior
            pass  # Continue even if index mapping fails
        for col in list(sanitized_df.columns):  # Iterate over a static list of columns to sanitize values
            try:  # Guard per-column sanitization to avoid failing entire styling pipeline
                if sanitized_df[col].dtype == object or getattr(pd.api.types, "is_string_dtype", lambda x: False)(sanitized_df[col]):  # Detect string-like columns
                    sanitized_df[col] = sanitized_df[col].apply(lambda x: sanitize_plot_text(str(x)) if pd.notnull(x) else x)  # Sanitize each cell in string columns
            except Exception:  # Ignore individual column sanitization errors to preserve original behavior
                pass  # Continue processing remaining columns even if one fails
        styled = sanitized_df.style.apply(stripe, axis=1)  # Apply zebra striping across rows on sanitized DataFrame
        return styled  # Return the styled DataFrame
    except Exception as e:  # Preserve exception handling style
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise exception to surface failure


def upscale_image_if_needed(path, fallback=False):
    """
    This function verifies the dimensions of the image at the given path and upscales it if either dimension is below 4k (3840x2160).
    
    :param path: Absolute path to the image file to verify and potentially upscale
    :param fallback: Boolean indicating if this upscale is being attempted after a fallback export (for logging purposes)
    :return: None
    """
    
    try:  # Guard image operations to avoid raising from image processing
        with Image.open(path) as im:  # Open the output image for inspection and possible resizing
            w, h = im.size  # Capture current image width and height
            if w < 3840 or h < 2160:  # Verify if image is smaller than 4k thresholds
                target_w = max(3840, w)  # Compute target width ensuring at least 3840
                target_h = max(2160, h)  # Compute target height ensuring at least 2160
                scale = max(target_w / float(w), target_h / float(h))  # Compute scale factor to meet both dimensions
                new_size = (int(w * scale), int(h * scale))  # Compute new integer dimensions for resizing
                resample_filter = getattr(Image, "LANCZOS", None)  # Attempt to get LANCZOS attribute from PIL.Image
                if resample_filter is None:  # If LANCZOS attribute is not present on PIL.Image
                    resampling_enum = getattr(Image, "Resampling", None)  # Attempt to get Resampling enum from PIL.Image
                    resample_filter = getattr(resampling_enum, "LANCZOS", None) if resampling_enum is not None else None  # Use Resampling.LANCZOS if available else None
                if resample_filter is None:  # If no LANCZOS candidate was found
                    resample_filter = getattr(Image, "BICUBIC", None)  # Attempt to get BICUBIC attribute from PIL.Image
                    if resample_filter is None:  # If BICUBIC is not present on PIL.Image
                        resampling_enum = getattr(Image, "Resampling", None)  # Attempt to get Resampling enum from PIL.Image again
                        resample_filter = getattr(resampling_enum, "BICUBIC", None) if resampling_enum is not None else None  # Use Resampling.BICUBIC if available else None
                    if resample_filter is None:  # If still no BICUBIC candidate was found
                        resample_filter = getattr(Image, "NEAREST", 0)  # Fallback to Image.NEAREST constant via getattr with numeric default
                im_resized = im.resize(new_size, resample=resample_filter)  # Resize using chosen resample filter with explicit resample argument
                orig_dpi = im.info.get("dpi") if hasattr(im, "info") else None  # Retrieve original DPI metadata if available
                
                if orig_dpi:  # Verify if DPI metadata exists
                    im_resized.save(path, dpi=orig_dpi)  # Save resized image preserving original DPI
                else:
                    im_resized.save(path)  # Save resized image without explicit DPI metadata
                
                if fallback:  # Verify whether this upscale was triggered from fallback export
                    print(f"{BackgroundColors.GREEN}[DEBUG] Upscaled image to meet 4k (fallback): {BackgroundColors.CYAN}{path}{Style.RESET_ALL}")  # Log fallback upscale event with colored output
                else:  # Verify whether this upscale was a normal upscale
                    print(f"{BackgroundColors.GREEN}[DEBUG] Upscaled image to meet 4k: {BackgroundColors.CYAN}{path}{Style.RESET_ALL}")  # Log normal upscale event with colored output
    except Exception:  # Ignore any image processing errors to avoid cascading failures
        pass  # Continue silently on upscale failures to preserve original behavior


def attempt_matplotlib_export_fallback(styled_df, output_path, e_inner):
    """
    Attempt to export a styled DataFrame to PNG using pure matplotlib table rendering as a last-resort fallback.

    :param styled_df: pandas.Styler or DataFrame object to render as a table image.
    :param output_path: File system path where the rendered PNG will be written.
    :param e_inner: Exception from the previous fallback attempt, or None when a prior method succeeded.
    :return: None when export succeeded, or the last encountered exception when matplotlib rendering also failed.
    """

    if e_inner is None:  # Skip matplotlib fallback when a prior method already succeeded
        return None  # Prior method succeeded; no further fallback is needed
    try:  # Attempt pure matplotlib table rendering as the final deterministic fallback
        try:  # Extract the underlying DataFrame from the Styler when possible
            df_to_render = getattr(styled_df, "data", styled_df)  # Access the raw DataFrame from a Styler or use the object as-is
        except Exception:  # Fall back to using styled_df directly when attribute access fails
            df_to_render = styled_df  # Use the original styled_df when data extraction is unavailable
        fig = plt.figure(figsize=(12, 8))  # Create a matplotlib figure sized for a readable table layout
        ax = fig.add_subplot(111)  # Add a single subplot to host the table
        ax.axis("off")  # Disable axes to produce a table-only image without borders or ticks
        try:  # Build the table from DataFrame values and column labels directly
            table = ax.table(cellText=list(df_to_render.values), colLabels=list(df_to_render.columns), loc='center')  # Construct the matplotlib table from the DataFrame
        except Exception:  # Fall back to stringified values when direct construction fails
            table = ax.table(cellText=[[str(x) for x in row] for row in df_to_render.values], colLabels=[str(c) for c in df_to_render.columns], loc='center')  # Build table with all values converted to strings
        table.auto_set_font_size(False)  # Disable automatic font sizing for consistent appearance
        table.set_fontsize(6)  # Set a small font size to fit large tables within the figure bounds
        fig.tight_layout()  # Adjust the layout to fit the table within the figure area
        fig.savefig(output_path, dpi=300)  # Save the rendered table to disk at 300 DPI
        plt.close(fig)  # Close the figure immediately after saving to free memory
        if verify_filepath_exists(output_path):  # Verify the output file was actually created on disk
            print(f"{BackgroundColors.GREEN}[DEBUG] Exported image (matplotlib fallback): {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}")  # Log matplotlib fallback success
            print(f"{BackgroundColors.GREEN}[INFO] Table image successfully saved to: {BackgroundColors.CYAN}{os.path.abspath(output_path)}{Style.RESET_ALL}")  # Log the absolute save path
            return None  # Return None to signal matplotlib export succeeded
        else:  # File not present after the save attempt indicates a silent failure
            return RuntimeError("Matplotlib fallback failed to produce output file")  # Return explicit error for caller to raise
    except Exception as _e_matplot:  # Capture any exception during matplotlib rendering for final re-raise
        return _e_matplot  # Return exception to allow the caller to perform final error handling


def attempt_chrome_export_fallback(styled_df, output_path, e_inner, export_kwargs, timeout_ms):
    """
    Attempt to export a styled DataFrame to PNG using Chrome as the table conversion engine.

    :param styled_df: pandas.Styler object to export as a PNG image.
    :param output_path: File system path where the exported PNG will be written.
    :param e_inner: Exception from the previous Playwright attempt, or None when Playwright succeeded.
    :param export_kwargs: Pre-built keyword arguments dict from which Chrome kwargs will be derived.
    :param timeout_ms: Timeout in milliseconds to pass to the Chrome conversion engine.
    :return: None when export succeeded, or the last encountered exception when the Chrome export failed.
    """

    if e_inner is None:  # Skip Chrome fallback when Playwright already succeeded
        return None  # Playwright succeeded; no fallback needed
    try:  # Attempt Chrome-based dataframe_image export as the first deterministic fallback
        chrome_kwargs = dict(export_kwargs)  # Copy the existing kwargs to preserve all prior options
        chrome_kwargs["table_conversion"] = "chrome"  # Override conversion engine to Chrome
        chrome_kwargs["timeout"] = timeout_ms  # Pass the configured timeout to the Chrome conversion engine
        dfi.export(styled_df, output_path, **chrome_kwargs)  # Attempt PNG export using Chrome conversion
        print(f"{BackgroundColors.GREEN}[DEBUG] Exported image (chrome fallback): {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}")  # Log Chrome fallback export success
        upscale_image_if_needed(output_path, fallback=True)  # Upscale exported image after Chrome fallback
        print(f"{BackgroundColors.GREEN}[INFO] Table image successfully saved to: {BackgroundColors.CYAN}{os.path.abspath(output_path)}{Style.RESET_ALL}")  # Log absolute save path
        return None  # Return None to signal Chrome export succeeded
    except Exception as _e_chrome:  # Record Chrome fallback exception for downstream matplotlib fallback
        return _e_chrome  # Return exception to allow caller to attempt matplotlib fallback


def attempt_playwright_export_with_retry(styled_df, output_path, export_kwargs, timeout_ms):
    """
    Attempt to export a styled DataFrame to PNG using Playwright-based dataframe_image with bounded retries.

    :param styled_df: pandas.Styler object to export as a PNG image.
    :param output_path: File system path where the exported PNG will be written.
    :param export_kwargs: Pre-built keyword arguments dict for dfi.export including table_conversion and timeout.
    :param timeout_ms: Timeout in milliseconds used as the fallback kwargs timeout when signature inspection fails.
    :return: None when export succeeded, or the last encountered exception when all attempts failed.
    """

    max_attempts = 5  # Define the maximum number of bounded Playwright export attempts
    e_inner = None  # Track the last exception; None signals success
    for attempt in range(1, max_attempts + 1):  # Retry up to max_attempts times to handle transient failures
        try:  # Try exporting using dataframe_image with Playwright and configured kwargs
            dfi.export(styled_df, output_path, **export_kwargs)  # Export styled DataFrame to PNG using dataframe_image
            print(f"{BackgroundColors.GREEN}[DEBUG] Exported image: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}")  # Log successful export for diagnostics
            upscale_image_if_needed(output_path, fallback=False)  # Upscale exported image if below 4k
            print(f"{BackgroundColors.GREEN}[INFO] Table image successfully saved to: {BackgroundColors.CYAN}{os.path.abspath(output_path)}{Style.RESET_ALL}")  # Log absolute save path
            e_inner = None  # Clear last exception on success
            break  # Exit retry loop after successful export
        except TypeError:  # Handle dfi versions that raise TypeError for unexpected kwargs
            try:  # Attempt fallback export using only the minimal supported kwargs
                try:  # Inspect dfi.export signature to determine supported timeout parameter
                    _params_fallback = set(signature(dfi.export).parameters.keys())  # Get set of supported parameter names
                except Exception:  # If signature inspection fails, treat all params as unsupported
                    _params_fallback = set()  # Use empty set as conservative fallback
                kwargs_fb: dict[str, Any] = {"table_conversion": "playwright"}  # Build minimal fallback kwargs
                if "timeout" in _params_fallback:  # Attach timeout only when supported
                    kwargs_fb["timeout"] = timeout_ms  # Attach timeout using the supported parameter name
                dfi.export(styled_df, output_path, **kwargs_fb)  # Retry export with minimal kwargs
                print(f"{BackgroundColors.GREEN}[DEBUG] Exported image (fallback): {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}")  # Log fallback export success
                upscale_image_if_needed(output_path, fallback=True)  # Upscale exported image after fallback
                print(f"{BackgroundColors.GREEN}[INFO] Table image successfully saved to: {BackgroundColors.CYAN}{os.path.abspath(output_path)}{Style.RESET_ALL}")  # Log save path after fallback
                e_inner = None  # Clear last exception on success
                break  # Exit retry loop after successful fallback export
            except Exception as _inner_e:  # Capture fallback exception for downstream retry/raise handling
                e_inner = _inner_e  # Record inner fallback exception
        except Exception as _e_export:  # Capture general export exceptions for potential retry
            e_inner = _e_export  # Record exception for downstream retry/raise logic

        try:  # Import Playwright TimeoutError for precise timeout detection
            from playwright._impl._errors import TimeoutError as PlaywrightTimeoutError  # Import when available
        except Exception:  # Disable precise detection when import fails
            PlaywrightTimeoutError = None  # Set to None when import is unavailable

        if PlaywrightTimeoutError is not None and isinstance(e_inner, PlaywrightTimeoutError):  # Playwright timeout detected
            if attempt < max_attempts:  # Retry if attempts remain
                time.sleep(0.5)  # Brief pause before retry to allow transient conditions to clear
                print(f"{BackgroundColors.YELLOW}[WARNING] Playwright screenshot timeout, retrying export (attempt {attempt})...{Style.RESET_ALL}")  # Log retry
                continue  # Retry the export
            else:  # Max attempts exhausted; fall through to fallback strategy
                pass  # No-op; allow fallback to execute after loop
        else:  # Non-timeout exception; retry if attempts remain
            if attempt < max_attempts:  # Retry if attempts remain
                time.sleep(0.2)  # Brief pause before next retry
                continue  # Retry the export
            else:  # Attempts exhausted; fall through to fallback strategy
                pass  # No-op; allow fallback to execute after loop

    return e_inner  # Return None on success or the last exception on failure


def load_tableau_image_config():
    """
    Load the configuration file, resolve the table image timeout, and build the base dataframe_image export kwargs.

    :return: Tuple of (timeout_ms, export_kwargs) where timeout_ms is the configured timeout in milliseconds and export_kwargs is a dict pre-populated with the Playwright conversion option and timeout parameters.
    """

    cfg = load_config_file()  # Load configuration from config.yaml if present in the workspace
    timeout_ms = int((cfg or {}).get("dataset_descriptor", {}).get("table_image_timeout_ms", 30000))  # Determine timeout in milliseconds using config value with hardcoded fallback
    src = "config" if (cfg or {}).get("dataset_descriptor", {}).get("table_image_timeout_ms") is not None else "default"  # Identify whether the timeout came from config or the default value
    print(f"{BackgroundColors.GREEN}[CONFIG] table_image_timeout_ms = {BackgroundColors.CYAN}{timeout_ms}{Style.RESET_ALL} (source: {src})")  # Log the active timeout value and its source with colored terminal output
    export_kwargs: dict[str, Any] = {"table_conversion": "playwright"}  # Build base export kwargs with Playwright as the conversion engine
    export_kwargs["timeout"] = timeout_ms  # Inject the configured timeout so Playwright receives the correct value
    try:  # Inspect dfi.export signature to attach the matching screenshot timeout parameter name
        params = set(signature(dfi.export).parameters.keys())  # Retrieve the set of parameter names supported by dfi.export
        for _pname in ("screenshot_timeout", "timeout", "playwright_timeout", "playwright_screenshot_timeout"):  # Iterate candidate timeout parameter names from various dfi versions
            if _pname in params:  # Verify whether this candidate is present in the detected parameter set
                export_kwargs[_pname] = timeout_ms  # Attach the timeout using the first matching parameter name
                break  # Stop after the first supported parameter to avoid conflicting kwargs
    except Exception:  # Ignore signature inspection failures to preserve original behavior
        pass  # Continue without explicit screenshot timeout when inspection is unavailable
    return timeout_ms, export_kwargs  # Return the resolved timeout and fully built export kwargs dict


def export_dataframe_image(styled_df, output_path):
    """
    Export a pandas.Styler to a PNG image using dataframe_image.

    :param styled_df: pandas.Styler object to export
    :param output_path: Path to write PNG image
    :return: None
    """

    try:  # Wrap to ensure exceptions are handled and module logging conventions are preserved
        timeout_ms, export_kwargs = load_tableau_image_config()  # Load config, resolve the table image timeout, and build Playwright export kwargs

        e_inner = attempt_playwright_export_with_retry(styled_df, output_path, export_kwargs, timeout_ms)  # Attempt export with bounded Playwright retries and return last exception or None on success

        if e_inner is not None:  # If last Playwright/dfi attempt failed and no success occurred
            e_inner = attempt_chrome_export_fallback(styled_df, output_path, e_inner, export_kwargs, timeout_ms)  # Try Chrome as first deterministic fallback and update e_inner

        if e_inner is not None:  # If both Playwright and chrome fallbacks failed, attempt matplotlib rendering as last resort
            e_inner = attempt_matplotlib_export_fallback(styled_df, output_path, e_inner)  # Try matplotlib as final fallback and update e_inner

        if e_inner is not None:  # If all methods failed, re-raise the last encountered exception to be handled by outer block
            raise e_inner  # Re-raise last exception to preserve original outer logging and telemetry behavior
    except Exception as e:  # If export fails, log warning and continue without crashing
        try:  # Try to import Playwright-specific TimeoutError for precise detection
            from playwright._impl._errors import TimeoutError as PlaywrightTimeoutError  # Optional import of Playwright TimeoutError for specific handling
        except Exception:  # If import fails, ensure variable is defined for downstream verification logic
            PlaywrightTimeoutError = None  # Set to None when Playwright TimeoutError cannot be imported
        if PlaywrightTimeoutError is not None and isinstance(e, PlaywrightTimeoutError):  # Verify if exception is Playwright TimeoutError
            print(f"{BackgroundColors.YELLOW}[WARNING] Playwright screenshot timeout while exporting {BackgroundColors.CYAN}{output_path}{BackgroundColors.YELLOW}: {e}{Style.RESET_ALL}")  # Warn when Playwright timeout occurs with colored output
        else:  # General failure when not a Playwright TimeoutError
            print(f"{BackgroundColors.YELLOW}[WARNING] Failed to export image {BackgroundColors.CYAN}{output_path}{BackgroundColors.YELLOW}: {e}{Style.RESET_ALL}")  # Warn for general export failures with colored output
        try:  # Send exception trace via Telegram for observability
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram for remote debugging
        except Exception:  # Ignore failures when sending Telegram notifications to avoid cascading errors
            pass  # Ignore failures when sending Telegram notifications
        return  # Return gracefully to avoid terminating the program
    finally:  # Ensure multiprocessing and large object cleanup regardless of export outcome
        try:  # Attempt explicit multiprocessing resource finalization to avoid leaked semaphores
            finalize_multiprocessing_resources()  # Finalize active child processes and resource tracker state
        except Exception:  # Ignore cleanup failures to preserve non-fatal export semantics
            pass  # Continue gracefully when finalization fails
        try:  # Attempt to release styled object reference as soon as export flow ends
            del styled_df  # Delete styled DataFrame reference to reduce retained memory
        except Exception:  # Ignore delete failures to preserve behavior
            pass  # Continue gracefully when reference deletion fails
        gc.collect()  # Trigger garbage collection after export cleanup


def generate_table_image_from_dataframe(df, output_path, config: dict | None = None):
    """
    Generate a zebra-striped PNG table image from a DataFrame and save to output_path.

    :param df: pandas.DataFrame to render.
    :param output_path: Path for output PNG image.
    :param config: Optional configuration dictionary (reserved for future use).
    :return: None.
    """

    try:  # Wrap to preserve module's error handling conventions
        styled = apply_zebra_style(df)  # Create a styled DataFrame with zebra striping
        export_dataframe_image(styled, output_path)  # Export the styled DataFrame to PNG
    except Exception:  # Do not swallow exceptions here per spec
        raise  # Re-raise any exception to caller


def generate_csv_and_image(df, csv_path, config: dict | None = None):
    """
    Save a DataFrame to CSV and generate a corresponding PNG table image next to it.

    :param df: pandas.DataFrame to save and render.
    :param csv_path: Full path for CSV output.
    :param config: Optional configuration dictionary for resolving image format.
    :return: Tuple (csv_path, image_path).
    """

    try:  # Wrap to preserve module's error handling conventions
        if not isinstance(csv_path, str) or not csv_path:  # Verify csv_path is a non-empty string
            raise ValueError("csv_path must be a non-empty string")  # Raise when csv_path is missing or invalid
        df.to_csv(csv_path, index=False)  # Persist DataFrame to CSV without index
        img_ext = (config or {}).get("dataset_descriptor", {}).get("table_image_format", "png")  # Resolve table image format from config with png fallback
        image_path = os.path.splitext(csv_path)[0] + f".{img_ext}"  # Build image path using configured format
        if len(df) <= 100:  # Generate image only when DataFrame size is within the safe row limit
            try:  # Guard image rendering to preserve the already-written CSV on PNG export failure
                generate_table_image_from_dataframe(df, image_path, config=config)  # Generate image from DataFrame
            except Exception as e:  # Contain PNG export failure locally to avoid aborting the pipeline
                print(f"{BackgroundColors.YELLOW}[WARNING] Failed to generate table image for {BackgroundColors.CYAN}{os.path.basename(csv_path)}{BackgroundColors.YELLOW}: {e}{Style.RESET_ALL}")  # Warn and continue when PNG rendering fails
        return csv_path, image_path  # Return both paths for caller use
    except Exception:
        raise  # Re-raise to preserve original failure semantics


def finalize_and_write_report(report_rows, preprocessing_metrics, base_dir, output_filename, config):
    """
    Number report rows, write the report CSV, generate the preprocessing summary, and return the success flag.

    :param report_rows: List of per-file info dicts accumulated during the processing loop.
    :param preprocessing_metrics: List of per-file preprocessing metric dicts for summary generation.
    :param base_dir: Absolute base directory used as the output root for report and summary files.
    :param output_filename: Resolved output filename string ending with ".csv".
    :param config: Optional configuration dictionary forwarded to write_report and save_preprocessing_summary_csv.
    :return: True when the report was written successfully, False when no report rows were available.
    """

    if not report_rows:  # Return False immediately when the processing loop produced no data rows
        return False  # Signal failure to the caller when no rows were collected
    for i, row in enumerate(report_rows, start=1):  # Assign sequential row numbers starting at 1
        row["#"] = i  # Embed the counter directly into each row dict before writing
    write_report(report_rows, base_dir, output_filename, config=config)  # Persist all numbered rows as the main report CSV
    try:  # Generate the preprocessing summary separately to avoid aborting the main report on failure
        if preprocessing_metrics:  # Only generate a summary when per-file metrics were successfully collected
            pre_df = build_preprocessing_summary_dataframe(preprocessing_metrics)  # Build a DataFrame from the accumulated metrics list
            out_path = save_preprocessing_summary_csv(pre_df, base_dir, config=config)  # Save the summary CSV to the results directory
            print(f"{BackgroundColors.GREEN}Saved preprocessing summary to {BackgroundColors.CYAN}{out_path}{Style.RESET_ALL}")  # Inform the user of the saved summary path
            if os.environ.get("DD_DESCRIPTOR_VERBOSE", "False").lower() in ("1", "true", "yes"):  # Print table only in verbose mode
                print_preprocessing_summary_table(pre_df)  # Print the summary table to the terminal when verbose output is enabled
            del pre_df  # Release preprocessing summary DataFrame after saving and printing to reduce retained memory
    except Exception as _ps:  # Warn and continue when summary generation fails to preserve the main report
        print(f"{BackgroundColors.YELLOW}Warning: failed to generate preprocessing summary: {_ps}{Style.RESET_ALL}")  # Warn without aborting
    return True  # Return True to signal that the main report was written successfully


def resolve_output_filename(output_filename, cfg):
    """
    Resolve the output filename for the dataset report CSV, applying config defaults and ensuring a .csv extension.

    :param output_filename: Caller-provided filename string, or None to use a config-derived default.
    :param cfg: Configuration dictionary used to read the csv_output_suffix fallback value.
    :return: Resolved output filename string guaranteed to end with ".csv".
    """

    if output_filename is None:  # Use config-based suffix when no filename was provided by the caller
        output_filename = cfg.get("dataset_descriptor", {}).get("csv_output_suffix", "description")  # Read configured suffix with hardcoded fallback
    if not isinstance(output_filename, str):  # Convert non-string filename to string using config suffix as fallback
        output_filename = str(output_filename or cfg.get("dataset_descriptor", {}).get("csv_output_suffix", "_description"))  # Stringify with config fallback when value is falsy
    if not output_filename.lower().endswith(".csv"):  # Append .csv extension when absent
        output_filename = f"{output_filename}.csv"  # Ensure the filename always has the .csv extension
    return output_filename  # Return the fully resolved output filename


def collect_report_input_files(input_path, file_extension, config):
    """
    Determine the matching files and base directory from the provided input path.

    :param input_path: Directory or single file path to scan for matching dataset files.
    :param file_extension: File extension used to filter files when scanning a directory.
    :param config: Optional configuration dictionary passed through to collect_matching_files.
    :return: Tuple of (sorted_matching_files, base_dir) where sorted_matching_files is a list of absolute file paths and base_dir is the absolute base directory used for relative path computations.
    """

    if os.path.isdir(input_path):  # Scan the directory for all matching files
        print(
            f"{BackgroundColors.GREEN}Scanning directory {BackgroundColors.CYAN}{input_path}{BackgroundColors.GREEN} for {BackgroundColors.CYAN}{file_extension}{BackgroundColors.GREEN} files...{Style.RESET_ALL}"
        )  # Announce directory scan start
        sorted_matching_files = collect_matching_files(input_path, file_extension, config=config)  # Collect all matching files from the directory tree
        base_dir = os.path.abspath(input_path)  # Use the directory itself as the base for relative paths
    elif os.path.isfile(input_path) and input_path.endswith(file_extension):  # Single file provided
        print(
            f"{BackgroundColors.GREEN}Processing single file...{Style.RESET_ALL}"
        )  # Announce single file processing
        sorted_matching_files = [input_path]  # Wrap the single file in a list for uniform processing
        base_dir = os.path.dirname(os.path.abspath(input_path))  # Use the file's parent directory as base
    else:  # Input is neither a directory nor a valid matching file
        print(
            f"{BackgroundColors.RED}Input path is neither a directory nor a valid {file_extension} file: {input_path}{Style.RESET_ALL}"
        )  # Report the invalid input path
        sorted_matching_files = []  # No files to process when input is invalid
        base_dir = os.path.abspath(input_path)  # Preserve input path as base for any downstream error messages
    return sorted_matching_files, base_dir  # Return the collected files and resolved base directory


def enrich_file_info_with_metadata(info, filepath, base_dir, headers_map, common_features, headers_match_all, cfg, low_memory, df_current):
    """
    Populate a file info dictionary with relative path, header uniformity, common/extra feature lists, and t-SNE plot path.

    :param info: Mutable dictionary of dataset metadata fields populated in place by this function.
    :param filepath: Absolute path of the dataset file being processed.
    :param base_dir: Absolute base directory used to compute the relative path for the Dataset Name field.
    :param headers_map: Dictionary mapping file paths to their header lists for common/extra feature computation.
    :param common_features: Set of feature names present in every discovered file used for common/extra classification.
    :param headers_match_all: Boolean flag indicating whether all files share identical header sets.
    :param cfg: Configuration dictionary used to look up the t-SNE output subdirectory key.
    :param low_memory: Boolean flag passed to the t-SNE generator to control memory usage during plot generation.
    :param df_current: Already-loaded pandas DataFrame for the current file passed to the t-SNE generator to avoid a second disk read.
    :return: None (modifies info in place with Dataset Name, Headers Match All Files, Common Features, Extra Features, and t-SNE Plot fields).
    """

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

    tsne_out_subdir = cfg.get("paths", {}).get("data_separability_subdir", "Data_Separability")  # Read t-SNE output subdirectory name from configuration with default fallback
    tsne_file = generate_tsne_plot(
        filepath,
        df=df_current,
        low_memory=low_memory,
        sample_size=2000,
        output_dir=os.path.join(os.path.dirname(os.path.abspath(filepath)), tsne_out_subdir),
        config=cfg,
    )  # Generate t-SNE plot using the already-loaded DataFrame to avoid rereading from disk
    info["t-SNE Plot"] = tsne_file if tsne_file else "None"  # Add t-SNE plot filename or "None"


def append_preprocessing_metrics_safe(filepath, info, preprocessing_metrics, file_basename):
    """
    Collect preprocessing metrics for a processed file and append them to the metrics list, WARNING on failure.

    :param filepath: Absolute path of the dataset file used as identifier in the metrics row.
    :param info: Dataset metadata dictionary providing original and post-preprocessing row/feature counts.
    :param preprocessing_metrics: Mutable list to which the collected metrics row dictionary is appended.
    :param file_basename: Relative file path string used in the failure warning message for user context.
    :return: None (appends to preprocessing_metrics in place or prints a warning on failure).
    """

    try:  # Collect preprocessing metrics for this file when available
        metrics_row = collect_preprocessing_metrics(
            filepath,  # File path being processed
            info.get("original_num_rows", 0),  # Original rows captured earlier
            info.get("rows_after_preprocessing", 0),  # Rows after preprocessing captured earlier
            info.get("original_num_features", 0),  # Original features captured earlier
            info.get("features_after_preprocessing", 0),  # Features after preprocessing captured earlier
            info.get("rows_after_nan_inf_removal", 0),  # Rows after NaN/infinite removal step
            info.get("removed_rows_nan_inf", 0),  # Rows removed by NaN/infinite filtering step
            info.get("removed_rows_nan_inf_proportion", 0.0),  # Proportion of rows removed by NaN/infinite filtering step
            info.get("features_after_zero_variance_removal", 0),  # Features after zero-variance numerical feature removal step
            info.get("removed_zero_variance_features", 0),  # Zero-variance numerical features removed in preprocessing
            info.get("removed_zero_variance_features_proportion", 0.0),  # Proportion of zero-variance numerical features removed
            info.get("dropped_non_informative_features", 0),  # Non-informative identifier/metadata features removed in this module
            info.get("dropped_non_informative_features_proportion", 0.0),  # Proportion of non-informative identifier/metadata features removed
            info.get("features_transformed_for_experiment", 0),  # Features transformed for dtype enforcement and categorical encoding per experiment
            info.get("features_transformed_for_experiment_proportion", 0.0),  # Proportion of transformed features for dtype enforcement and categorical encoding per experiment
            info.get("features_cast_to_float64_int64", 0),  # Numeric features requiring cast to float64/int64
            info.get("features_encoded_categorical", 0),  # Categorical features requiring ordinal or one-hot encoding
            info.get("preprocessing_metrics", None),  # Structured per-step preprocessing metrics for isolated CSV mapping
        )  # Create metrics row dict
        preprocessing_metrics.append(metrics_row)  # Append metrics row to list for this directory
    except Exception as _pm:  # If metrics collection fails
        print(f"{BackgroundColors.YELLOW}Warning: failed to collect preprocessing metrics for {file_basename}: {_pm}{Style.RESET_ALL}")  # Warn without breaking the progress bar


def generate_dataset_report(input_path, file_extension=".csv", low_memory=None, output_filename: str | None = None, config: dict | None = None):
    """
    Generate a CSV report for the specified input path.
    The Dataset Name column will include subdirectories if present.

    :param input_path: Directory or file path containing the dataset.
    :param file_extension: File extension to filter (default: .csv).
    :param low_memory: Whether to use low memory mode when loading CSVs (default: True).
    :param output_filename: Name of the CSV file to save the report.
    :param config: Optional configuration dictionary for resolving paths and settings.
    :return: True if the report was generated successfully, False otherwise.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        report_rows = []  # List to store report rows
        sorted_matching_files = []  # List to store matching files
        preprocessing_metrics = []  # List to collect per-file preprocessing metric dicts

        sorted_matching_files, base_dir = collect_report_input_files(input_path, file_extension, config)  # Collect matching files and resolve base directory from the provided input path

        cfg = config or get_default_config()

        if not sorted_matching_files:  # If no matching files were found
            print(f"{BackgroundColors.RED}No matching {file_extension} files found in: {input_path}{Style.RESET_ALL}")
            return False  # Exit the function

        output_filename = resolve_output_filename(output_filename, cfg)  # Resolve the output filename, applying config defaults and ensuring a .csv extension

        headers_map = build_headers_map(sorted_matching_files, low_memory=low_memory)  # Build headers map using lightweight header-only reads to avoid loading all datasets into memory simultaneously
        common_features, headers_match_all = compute_common_features(headers_map)  # Compute shared features and header uniformity flag from the headers-only map

        progress = tqdm(
            sorted_matching_files,  # Iterate over sorted matching files list
            desc=f"{BackgroundColors.GREEN}Processing files{Style.RESET_ALL}",  # Description text remains green and reset styles
            unit="file",  # Use file as unit for progress
            ncols=100,  # Fixed progress bar width in characters
            colour="cyan",  # Set progress bar visualization color to cyan
        )  # Create a single in-place progress bar instance
        for idx, filepath in enumerate(progress, 1):  # Process each matching file
            file_basename = os.path.relpath(filepath, base_dir).replace("\\", "/")  # Get the file path relative to base_dir and normalize slashes
            py_dir = os.path.dirname(os.path.abspath(__file__))  # Resolve the directory of the running Python file for relative display path computation
            display_path = os.path.relpath(filepath, py_dir).replace("\\", "/")  # Compute path relative to the Python file's directory for display in the progress bar
            colored_desc = f"{BackgroundColors.GREEN}Processing {BackgroundColors.CYAN}{display_path}{Style.RESET_ALL}"  # Compose colored description using BackgroundColors while keeping length bounded
            progress.set_description(colored_desc)  # Update progress bar description with colored, truncated filename for inline display

            df_current = load_dataset(filepath, low_memory)  # Load one dataset at a time to minimize peak RAM usage
            if df_current is None:  # Verify that the dataset was loaded successfully
                print(f"{BackgroundColors.YELLOW}Warning: failed to load {filepath}; skipping.{Style.RESET_ALL}")  # Warn about the skipped file without breaking the progress bar
                continue  # Skip to the next file without accumulating a None entry

            info = get_dataset_file_info(filepath, df=df_current, low_memory=low_memory)  # Extract metadata using the already-loaded DataFrame to avoid a second full read
            if info:  # If info was successfully retrieved
                enrich_file_info_with_metadata(info, filepath, base_dir, headers_map, common_features, headers_match_all, cfg, low_memory, df_current)  # Populate Dataset Name, Headers Match, Common/Extra Features, and t-SNE Plot fields in place

                report_rows.append(info)  # Add the info to the report rows

                append_preprocessing_metrics_safe(filepath, info, preprocessing_metrics, file_basename)  # Collect and append preprocessing metrics with error-safe handling
                del info  # Release info reference after appending to report structures to reduce retention

            try:  # Attempt to release dataset memory to minimize peak RAM consumption
                del df_current  # Delete the current dataset reference to allow garbage collection
            except Exception:  # Ignore exceptions during cleanup to prevent masking processing errors
                pass  # Continue without cleanup on delete failure
            gc.collect()  # Force garbage collection to reclaim memory released by deleting the dataset

        return finalize_and_write_report(report_rows, preprocessing_metrics, base_dir, output_filename, config)  # Number rows, write report, generate preprocessing summary, and return success flag
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def collect_group_files(paths, file_extension=".csv", config: dict | None = None):
    """
    Collect all matching files for a group of paths.

    :param paths: List of file or directory paths to search.
    :param file_extension: File extension to filter (default: ".csv").
    :param config: Optional configuration dictionary passed to collect_matching_files.
    :return: Sorted list of unique file paths.
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

        unique_files = list(set(files))  # Remove duplicates while preserving no particular order

        files_with_size = []  # Prepare list to hold (path, size) tuples for robust sorting
        for f in unique_files:  # Iterate files to resolve their sizes
            try:  # Attempt to get file size and handle any filesystem issues gracefully
                size = os.path.getsize(f)  # Get the file size in bytes for sorting by magnitude
            except Exception:  # If size retrieval fails for any file
                size = 0  # Fallback to zero size to avoid breaking the sort when file is inaccessible
            files_with_size.append((f, size))  # Store tuple of file path and its size for later sorting

        return [p for p, _ in sorted(files_with_size, key=lambda x: (-x[1], x[0]))]  # Sort by size descending then by path for determinism
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def compute_group_features(files, low_memory=None):
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

                n_common = int(len(common_between))  # Integer count of common features
                n_extra_a = int(len(extras_a))  # Integer count of extra features in A
                n_extra_b = int(len(extras_b))  # Integer count of extra features in B

                row = {  # Construct row dictionary
                    "Dataset A": a_name,  # First dataset group name
                    "Dataset B": b_name,  # Second dataset group name
                    "Files in A": len(a_info["files"]),  # Number of files in A
                    "Files in B": len(b_info["files"]),  # Number of files in B
                    "N Common Features": n_common,  # Integer count of common features between A and B
                    "Common Features (A ∩ B)": ", ".join(common_between) or "None",  # Common features between A and B
                    "N Extra Features in A": n_extra_a,  # Integer count of extra features present in A but not in B
                    "Extra Features in A (A \\ B)": ", ".join(extras_a) or "None",  # Extra features in A
                    "N Extra Features in B": n_extra_b,  # Integer count of extra features present in B but not in A
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
                    "N Common Features": int(row.get("N Common Features", 0)),  # Keep common feature count unchanged on swap
                    "Common Features (A ∩ B)": row["Common Features (A ∩ B)"],  # Keep common features unchanged on swap
                    "N Extra Features in A": int(row.get("N Extra Features in B", 0)),  # Swap extra feature count so A count reflects former B count
                    "Extra Features in A (A \\ B)": row["Extra Features in B (B \\ A)"],  # Swap extra features so A receives former B extras
                    "N Extra Features in B": int(row.get("N Extra Features in A", 0)),  # Swap extra feature count so B count reflects former A count
                    "Extra Features in B (B \\ A)": row["Extra Features in A (A \\ B)"],  # Swap extra features so B receives former A extras
                }
                adjusted.append(swapped)  # Append swapped row
            else:  # Unrelated row, keep as-is
                adjusted.append(dict(row))  # Keep as-is

        return adjusted  # Return adjusted rows
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def generate_cross_dataset_report(datasets_dict, file_extension=".csv", low_memory=None, output_filename=None, config: dict | None = None):
    """
    Generate a cross-dataset feature-compatibility report comparing dataset
    groups defined in `datasets_dict`. Produces pairwise comparisons between
    dataset groups and writes a CSV report named `Cross_{RESULTS_FILENAME}` by
    default into the `RESULTS_DIR`.

    :param datasets_dict: Dict mapping dataset group name -> list of paths.
    :param file_extension: Extension to search for (default: .csv).
    :param low_memory: Passed to CSV loader when building headers.
    :param output_filename: Optional filename to write; defaults to Cross_{RESULTS_FILENAME}.
    :param config: Optional configuration dictionary for resolving output paths and settings.
    :return: True on success, False otherwise.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        cfg = config or get_default_config()
        if output_filename is None:  # If no output filename is provided
            suffix = cfg.get("dataset_descriptor", {}).get("csv_output_suffix", "_description")  # Get suffix from config or default
            output_filename = f"Cross_{suffix.lstrip('_')}" if suffix else "Cross_dataset_descriptor.csv"  # Build cross filename
        if not output_filename.lower().endswith(".csv"):  # Verify the output filename has a .csv extension
            output_filename = f"{output_filename}.csv"  # Append .csv extension when missing

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
    Calculate the execution time and return a human-readable string.

    :param start_time: The start time or duration value (datetime, timedelta, or numeric seconds).
    :param finish_time: Optional finish time; if None, start_time is treated as the total duration.
    :return: Human-readable execution time string formatted as days, hours, minutes, and seconds.
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
    Play a sound when the program finishes and skip if the operating system is Windows.

    :return: None.
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

    :return: None.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring

        cli_args_dict = parse_cli_args()  # Parse CLI arguments and load configuration as dict
        config = get_config(file_path=cli_args_dict.get("config", "config.yaml"), cli_args=cli_args_dict)  # Load and merge config with CLI overrides

        cli_args_ns = argparse.Namespace(**cli_args_dict)  # Convert dict to Namespace for type safety

        runtime = init_runtime(config)  # Initialize runtime artifacts including the logger

        sys_stdout_old = sys.stdout  # Save original stdout for later restoration
        sys_stderr_old = sys.stderr  # Save original stderr for later restoration
        sys.stdout = runtime["logger"]  # Redirect stdout to logger for this runtime session
        sys.stderr = runtime["logger"]  # Redirect stderr to logger for this runtime session

        os.environ["DD_DESCRIPTOR_VERBOSE"] = str(runtime.get("verbose", False))  # Export verbosity flag via environment for use in other functions

        print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Dataset Descriptor{BackgroundColors.GREEN}!{Style.RESET_ALL}")  # Print welcome message
        start_time = datetime.datetime.now()  # Capture program start time

        log_config_sources(config, cli_args_dict)  # Log resolved configuration values with their source
        low_memory = resolve_low_memory(cli_args_ns, config)  # Determine low memory mode based on CLI and config settings

        setup_telegram_bot()  # Initialize Telegram bot for progress notifications
        send_telegram_message(TELEGRAM_BOT, [f"Starting Dataset Descriptor at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"])  # Send start notification via Telegram

        datasets = config.get("dataset_descriptor", {}).get("datasets", {}) or config.get("datasets") or {}  # Resolve datasets mapping from config
        results_suffix = config.get("dataset_descriptor", {}).get("csv_output_suffix", "_description")  # Resolve output CSV suffix from config

        for dataset_name, paths in datasets.items():  # Iterate over configured dataset entries
            dataset_name = str(dataset_name).strip()  # Normalize dataset name by removing leading/trailing spaces
            verbose_output(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing dataset: {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}")  # Log dataset processing start
            safe_dataset_name = dataset_name.replace(" ", "_").replace("/", "_")  # Sanitize dataset name for safe filesystem use and remove leading/trailing spaces

            for dir_path in paths:  # Iterate over all configured paths for this dataset
                dir_path = str(dir_path).strip()  # Normalize directory path by removing leading/trailing spaces
                print(f"{BackgroundColors.GREEN}Location: {BackgroundColors.CYAN}{dir_path}{Style.RESET_ALL}")  # Print current directory path
                if not verify_filepath_exists(dir_path):  # Verify the configured path exists before processing
                    print(f"{BackgroundColors.RED}The specified input path does not exist: {BackgroundColors.CYAN}{dir_path}{Style.RESET_ALL}")  # Report missing path to terminal
                    continue  # Skip non-existing paths without aborting the full run

                success = generate_dataset_report(dir_path, file_extension=".csv", low_memory=low_memory, output_filename=None, config=config)  # Generate dataset report for this path
                if not success:  # Verify whether report generation succeeded
                    print(f"{BackgroundColors.RED}Failed to generate dataset report for: {BackgroundColors.CYAN}{dir_path}{Style.RESET_ALL}")  # Report failure for this path
                else:  # Report generation succeeded
                    print(f"{BackgroundColors.GREEN}Report saved for {BackgroundColors.CYAN}{safe_dataset_name}{BackgroundColors.GREEN} -> {BackgroundColors.CYAN}{results_suffix}{Style.RESET_ALL}")  # Confirm successful report save

        if config.get("execution", {}).get("cross_dataset_validate", True) and len(datasets) > 1:  # Verify cross-dataset validation is enabled and multiple datasets are configured
            try:  # Attempt cross-dataset validation with graceful failure handling
                send_telegram_message(TELEGRAM_BOT, "Starting cross-dataset validation...")  # Notify cross-dataset validation start via Telegram
                success = generate_cross_dataset_report(datasets, file_extension=".csv", low_memory=low_memory, config=config)  # Generate pairwise cross-dataset feature compatibility report
                if success:  # Verify whether cross-dataset report was saved
                    print(f"{BackgroundColors.GREEN}Cross-dataset report saved -> {BackgroundColors.CYAN}Cross_{results_suffix.lstrip('_')}{Style.RESET_ALL}")  # Confirm successful cross-dataset report save
                else:  # Cross-dataset report generation produced no output
                    print(f"{BackgroundColors.YELLOW}No cross-dataset comparisons generated (no files found).{Style.RESET_ALL}")  # Warn when no output was produced
            except Exception as e:  # Catch cross-dataset validation errors to avoid aborting the main run
                print(f"{BackgroundColors.RED}Cross-dataset validation failed: {e}{Style.RESET_ALL}")  # Report cross-dataset validation failure without re-raising

        sys.stdout = sys_stdout_old  # Restore original stdout after logging session
        sys.stderr = sys_stderr_old  # Restore original stderr after logging session

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

        try:  # Attempt to register sound notification at exit when configured
            if config.get("execution", {}).get("play_sound", True):  # Verify play_sound is enabled in config
                atexit.register(play_sound)  # Register play_sound to execute when the program exits
        except Exception:  # Ignore any errors during atexit registration to avoid crashing at exit
            pass  # Continue silently when atexit registration fails

        finalize_multiprocessing_resources()  # Finalize child processes and tracked shared resources before interpreter shutdown
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
        configure_multiprocessing_startup()  # Configure multiprocessing start method once before executing main flow
        main()  # Call the main function
    except Exception as e:  # Catch any unhandled exception from main
        print(str(e))  # Print the exception message to terminal for logs
        try:  # Attempt to send full exception via Telegram
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback and message
        except Exception:  # If sending the notification fails, print traceback to stderr
            traceback.print_exc()  # Print full traceback to stderr as final fallback
        raise  # Re-raise to avoid silent failure and preserve original crash behavior
