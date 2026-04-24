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

            - `Feature_Analysis/PCA_Results.csv` (one row per configuration)
            - Saved PCA objects for reproducibility (optional)
            - Console summary and best-configuration selection by CV F1-score

Notes & conventions:
    - The code expects the last column to be the target variable.
    - Only numeric input columns are used for PCA (non-numeric columns are ignored).
    - Defaults: 80/20 train-test split, 10-fold Stratified CV on training data,
    - Toggle `VERBOSE = True` for extra diagnostic output.
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
import dataframe_image as dfi  # For exporting DataFrame styled tables as PNG images
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
import traceback  # For formatting and printing exception tracebacks
import yaml  # For loading configuration from YAML files
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
from typing import Any, Union, cast  # For type hints used by functions


# Macros:
class BackgroundColors:  # Colors for the terminal
    CYAN = "\033[96m"  # Cyan
    GREEN = "\033[92m"  # Green
    YELLOW = "\033[93m"  # Yellow
    RED = "\033[91m"  # Red
    BOLD = "\033[1m"  # Bold
    UNDERLINE = "\033[4m"  # Underline
    CLEAR_TERMINAL = "\033[H\033[J"  # Clear the terminal


# Execution placeholders (no hard-coded runtime constants; populated at runtime)
VERBOSE = None
SKIP_TRAIN_IF_MODEL_EXISTS = None
CSV_FILE = None
N_JOBS = None
CROSS_N_FOLDS = None
CPU_PROCESSES = None
CACHING_ENABLED = None
PICKLE_PROTOCOL = None
CONFIG_FILE = None

# Telegram Bot Setup placeholder (initialized in runtime)
TELEGRAM_BOT = None

# Note: Logger and exception hooks are initialized at runtime inside main()

# Sound Constants:
SOUND_COMMANDS = {
    "Darwin": "afplay",
    "Linux": "aplay",
    "Windows": "start",
}  # The commands to play a sound for each operating system
SOUND_FILE = "./.assets/Sounds/NotificationSound.wav"  # The path to the sound file

# RUN_FUNCTIONS removed; sound/playback is controlled via configuration at runtime

# Functions Definitions:


# Do not call global exception hooks at import time; initialize in `main()` once config is loaded

def get_default_config() -> dict:
    """
    Return the default configuration dictionary for PCA analysis.

    This function centralizes all defaults so that configuration precedence
    (CLI > config.yaml > defaults) can be implemented reliably.

    :return: Default configuration dictionary for PCA analysis.
    """

    try:
        return {
            "pca": {
                "execution": {
                    "verbose": False,
                    "skip_train_if_model_exists": False,
                    "dataset_path": None,
                },
                "model": {
                    "estimator": "RandomForestClassifier",
                    "random_state": 42,
                },
                "dimensionality": {
                    "n_components": 8,
                    "n_components_list": [8, 16, 32, 64],
                },
                "preprocessing": {"scale_data": True, "remove_zero_variance": True},
                "cross_validation": {"n_folds": 10},
                "multiprocessing": {"n_jobs": -1, "cpu_processes": 1},
                "caching": {"enabled": True, "pickle_protocol": 4},
                "export": {
                    "results_dir": "Feature_Analysis/PCA",
                    "results_filename": "PCA_Results.csv",
                    "results_csv_columns": [
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
                },
            }
        }
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def load_config_file(path: str) -> dict:
    """
    Load YAML configuration from `path` and return as dict.

    Raises FileNotFoundError or yaml.YAMLError on parse errors so callers
    can handle validation explicitly.

    :param path: Absolute or relative path to the YAML configuration file.
    :return: Dictionary loaded from the YAML file, or an empty dict if path is falsy.
    """

    try:
        if not path:  # Return empty dict when no path is provided
            return {}  # Empty config for absent path
        p = Path(path)  # Build Path object from the string path
        if not p.exists():  # Verify the file exists before attempting to load it
            raise FileNotFoundError(f"Config file not found: {path}")  # Raise descriptive error for missing config
        with p.open("r", encoding="utf-8") as fh:  # Open the config file with UTF-8 encoding
            data = yaml.safe_load(fh) or {}  # Parse YAML and default to empty dict if file is empty
        if not isinstance(data, dict):  # Validate that the top-level structure is a mapping
            raise ValueError("Configuration file must contain a YAML mapping at top level")  # Raise on invalid structure
        return data  # Return the parsed config dict
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def parse_cli_args() -> dict:
    """
    Parse CLI arguments and return a plain dict of values. This is executed
    only when requested (e.g., inside `get_config()` or `main()`), never at
    import time.

    :return: Dictionary of parsed argument names mapped to their values.
    """

    try:
        parser = argparse.ArgumentParser(description="PCA Feature Extraction & Evaluation Tool")  # Initialize argument parser with description
        parser.add_argument("--config", type=str, default=None, help="Path to config.yaml (overrides auto-detection)")  # Config file path override
        parser.add_argument("--dataset_path", type=str, default=None, help="Path to the CSV dataset file")  # Dataset file path argument
        parser.add_argument("--n_components", type=int, default=None, help="Single number of PCA components to test (overrides n_components_list)")  # Single n_components override
        parser.add_argument("--n_components_list", type=str, default=None, help="Comma-separated list of PCA component counts to test")  # Comma-separated n_components list
        parser.add_argument("--random_state", type=int, default=None, help="Random seed for reproducibility (overrides config)")  # Random state override
        parser.add_argument("--scale_data", dest="scale_data", action="store_true", default=None, help="Enable scaling (overrides config)")  # Enable scaling flag
        parser.add_argument("--no-scale_data", dest="scale_data", action="store_false", help="Disable scaling (overrides config)")  # Disable scaling flag
        parser.add_argument("--remove_zero_variance", dest="remove_zero_variance", action="store_true", default=None, help="Remove zero-variance features (overrides config)")  # Enable zero-variance removal
        parser.add_argument("--no-remove_zero_variance", dest="remove_zero_variance", action="store_false", help="Do not remove zero-variance features (overrides config)")  # Disable zero-variance removal
        parser.add_argument("--max_workers", type=int, default=None, help="Number of parallel workers (overrides config)")  # Max workers override
        parser.add_argument("--n_folds", type=int, default=None, help="Number of CV folds (overrides config.cross_validation.n_folds)")  # CV folds override
        parser.add_argument("--n_jobs", type=int, default=None, help="Number of parallel jobs for estimators/CV (-1 uses all cores)")  # N_jobs override
        parser.add_argument("--cpu_processes", type=int, default=None, help="Number of CPU processes for multiprocessing (overrides config.multiprocessing.cpu_processes)")  # CPU processes override
        parser.add_argument("--caching_enabled", type=lambda s: str(s).lower() in ("1", "true", "yes", "y"), default=None, help="Enable/disable caching (true/false). Overrides config.caching.enabled")  # Caching enabled flag
        parser.add_argument("--pickle_protocol", type=int, default=None, help="Pickle protocol (0-5) to use when caching")  # Pickle protocol override
        parser.add_argument("--verbose", action="store_true", default=None, help="Enable verbose output (overrides config)")  # Verbose output flag
        parser.add_argument("--skip_train_if_model_exists", action="store_true", default=None, help="Skip training if exported model exists (overrides config)")  # Skip training flag
        parser.add_argument("--results_dir", type=str, default=None, help="Override results directory for PCA exports (overrides config.pca.export.results_dir)")  # Results directory override
        parser.add_argument("--results_filename", type=str, default=None, help="Override results filename for PCA exports (overrides config.pca.export.results_filename)")  # Results filename override

        args = parser.parse_args()  # Parse all CLI arguments
        return vars(args)  # Return parsed arguments as a plain dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def deep_merge_dicts(base: dict, override: dict) -> dict:
    """
    Recursively merge two dictionaries with `override` taking precedence.
    Returns a new merged dictionary.

    :param base: Base dictionary to merge into.
    :param override: Override dictionary whose values take precedence over base.
    :return: New dictionary with all keys from base overridden by values from override.
    """

    try:
        if not isinstance(base, dict):  # If base is not a dict, override replaces it entirely
            return override  # Return override directly when base is not a mapping
        result = dict(base)  # Create a shallow copy of the base dictionary
        for k, v in (override or {}).items():  # Iterate all override key-value pairs
            if k in result and isinstance(result[k], dict) and isinstance(v, dict):  # Recurse when both values are dicts
                result[k] = deep_merge_dicts(result[k], v)  # Recursively merge nested dicts
            else:  # For non-dict values, override takes priority
                result[k] = v  # Assign override value directly
        return result  # Return the merged dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def validate_config_structure(config: dict) -> None:
    """
    Validate the merged configuration structure and raise ValueError on issues.

    This enforces the required pca.export.results_csv_columns presence and
    performs light type/range verifications.

    :param config: Merged configuration dictionary to validate.
    :return: None
    """

    try:
        if not isinstance(config, dict):  # Validate that config is a dictionary
            raise ValueError("Configuration must be a mapping/dict")  # Raise on non-dict input
        if "pca" in config and isinstance(config.get("pca"), dict):  # Verify 'pca' section exists and is a dict
            pca_cfg = config.get("pca")  # Use the 'pca' section if it exists and is a dict
        else:  # Allow top-level keys if 'pca' section is missing, but still require 'export' subsection
            pca_cfg = config  # Treat config itself as the pca config when 'pca' key is absent

        if not isinstance(pca_cfg, dict):  # Validate that pca_cfg resolves to a dict
            raise ValueError("Missing or invalid 'pca' configuration section")  # Raise on invalid structure

        export_cfg = pca_cfg.get("export")  # Retrieve the export subsection
        if not isinstance(export_cfg, dict):  # Validate that export subsection exists and is a dict
            raise ValueError("Missing 'pca.export' configuration section")  # Raise on missing export section

        cols = export_cfg.get("results_csv_columns")  # Retrieve the CSV column header list
        if not isinstance(cols, list) or not cols:  # Validate that columns list is a non-empty list
            raise ValueError("'pca.export.results_csv_columns' must be a non-empty list defining the CSV header")  # Raise on invalid or empty columns

        n_folds = (pca_cfg.get("cross_validation") or {}).get("n_folds")  # Retrieve n_folds from cross_validation section
        if not isinstance(n_folds, int) or n_folds < 2:  # Validate n_folds is an integer >= 2
            raise ValueError("'pca.cross_validation.n_folds' must be an integer >= 2")  # Raise on invalid n_folds

        n_components = (pca_cfg.get("dimensionality") or {}).get("n_components")  # Retrieve n_components from dimensionality section
        if not isinstance(n_components, int) or n_components <= 0:  # Validate n_components is a positive integer
            raise ValueError("'pca.dimensionality.n_components' must be an integer > 0")  # Raise on invalid n_components
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def resolve_config_file_path(cli_config_arg) -> str | None:
    """
    Resolve the configuration file path using the CLI argument or auto-detection.

    :param cli_config_arg: Config path from CLI, or None to trigger auto-detection.
    :return: Resolved config file path string, or None if not found.
    """

    try:
        if cli_config_arg:  # Use CLI-provided path when available
            return cli_config_arg  # Return the CLI-specified config path directly

        candidate = Path(__file__).parent / "config.yaml"  # Auto-detect config.yaml in script directory
        candidate_example = Path(__file__).parent / "config.yaml.example"  # Verify for example config file as fallback

        if candidate.exists():  # Verify if the primary config file exists
            return str(candidate)  # Return the primary config file path
        elif candidate_example.exists():  # Verify if the example config file exists
            return str(candidate_example)  # Return the example config file path

        return None  # Return None when neither config file is found
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_execution_overrides(cli: dict) -> dict:
    """
    Build the execution section overrides from CLI arguments.

    :param cli: Dictionary of parsed CLI arguments.
    :return: Nested override dict for the execution section or empty dict.
    """

    try:
        exec_ov = {}  # Collect execution-related override values

        if cli.get("verbose") is not None:  # Verify if verbose was specified via CLI
            exec_ov.setdefault("execution", {})["verbose"] = bool(cli.get("verbose"))  # Store boolean verbose override
        if cli.get("skip_train_if_model_exists") is not None:  # Verify if skip flag was specified
            exec_ov.setdefault("execution", {})["skip_train_if_model_exists"] = bool(cli.get("skip_train_if_model_exists"))  # Store boolean skip override
        if cli.get("dataset_path") is not None:  # Verify if dataset path was specified
            exec_ov.setdefault("execution", {})["dataset_path"] = cli.get("dataset_path")  # Store dataset path override

        return exec_ov  # Return the collected execution overrides
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_model_overrides(cli: dict) -> dict:
    """
    Build the model section overrides from CLI arguments.

    :param cli: Dictionary of parsed CLI arguments.
    :return: Nested override dict for the model section or empty dict.
    """

    try:
        model_ov = {}  # Collect model-related override values
        raw_random_state: Any = cli.get("random_state")  # Retrieve raw CLI value for random_state (may be None)

        if raw_random_state is not None:  # Only process when CLI provided to preserve original behavior
            if not isinstance(raw_random_state, (int, str)):  # Validate acceptable input types
                raise TypeError("--random_state must be an int or string convertible to int")  # Raise on invalid type
            try:
                model_ov.setdefault("model", {})["random_state"] = int(raw_random_state)  # Safely convert to int
            except Exception as e:
                raise ValueError(f"Invalid --random_state value: {raw_random_state}") from e  # Raise with context

        return model_ov  # Return the collected model overrides
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_export_overrides(cli: dict) -> dict:
    """
    Build the export section overrides from CLI arguments.

    :param cli: Dictionary of parsed CLI arguments.
    :return: Nested override dict for the export section or empty dict.
    """

    try:
        export_ov = {}  # Collect export-related override values

        if cli.get("results_dir") is not None:  # Verify if results_dir was overridden via CLI
            export_ov.setdefault("export", {})["results_dir"] = cli.get("results_dir")  # Store results_dir override
        if cli.get("results_filename") is not None:  # Verify if results_filename was overridden via CLI
            export_ov.setdefault("export", {})["results_filename"] = cli.get("results_filename")  # Store results_filename override

        return export_ov  # Return the collected export overrides
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_dimensionality_overrides(cli: dict) -> dict:
    """
    Build the dimensionality section overrides from CLI arguments.

    :param cli: Dictionary of parsed CLI arguments.
    :return: Nested override dict for the dimensionality section or empty dict.
    """

    try:
        dim_ov = {}  # Collect dimensionality-related override values

        if cli.get("n_components") is not None:  # Verify if n_components was specified
            raw_ncomp: Any = cli.get("n_components")  # Retrieve raw CLI value for n_components (may be str/int)
            if raw_ncomp is None:  # Defensive guard; preserve original conditional logic
                pass  # No-op when raw_ncomp unexpectedly None (keeps original semantics)
            else:
                if not isinstance(raw_ncomp, (int, str)):  # Ensure convertible types
                    raise TypeError("--n_components must be an int or string convertible to int")  # Raise on invalid type
                try:
                    validated_ncomp: int = int(raw_ncomp)  # Convert to int after validation
                except Exception as e:
                    raise ValueError(f"Invalid --n_components value: {raw_ncomp}") from e  # Raise with context
                dim_ov.setdefault("dimensionality", {})["n_components"] = validated_ncomp  # Store validated int
                dim_ov.setdefault("dimensionality", {})["n_components_list"] = [validated_ncomp]  # Store single-item list
        elif cli.get("n_components_list") is not None:  # Verify if n_components_list was specified instead
            try:
                raw_list: Any = cli.get("n_components_list")  # Retrieve raw CLI value for n_components_list
                if not isinstance(raw_list, str):  # Ensure it is a comma-separated string as expected
                    raise TypeError("--n_components_list must be a comma-separated string of integers")  # Raise on wrong type
                parts = []  # Prepare list to collect validated ints
                for x in raw_list.split(","):  # Iterate substrings to validate and convert each
                    s = x.strip()  # Strip whitespace from the substring
                    if not s:  # Skip empty substrings
                        continue  # Continue to next substring
                    try:
                        parts.append(int(s))  # Convert validated substring to int and append
                    except Exception as e:
                        raise ValueError(f"Invalid integer in --n_components_list: {s}") from e  # Raise on bad value
            except Exception:
                raise ValueError("--n_components_list must be a comma-separated list of integers")  # Raise on parse failure
            dim_ov.setdefault("dimensionality", {})["n_components_list"] = parts  # Store validated list

        return dim_ov  # Return the collected dimensionality overrides
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_preprocessing_overrides(cli: dict) -> dict:
    """
    Build the preprocessing section overrides from CLI arguments.

    :param cli: Dictionary of parsed CLI arguments.
    :return: Nested override dict for the preprocessing section or empty dict.
    """

    try:
        prep_ov = {}  # Collect preprocessing-related override values

        if cli.get("scale_data") is not None:  # Verify if scale_data was specified via CLI
            prep_ov.setdefault("preprocessing", {})["scale_data"] = bool(cli.get("scale_data"))  # Store boolean scale_data override
        if cli.get("remove_zero_variance") is not None:  # Verify if remove_zero_variance was specified
            prep_ov.setdefault("preprocessing", {})["remove_zero_variance"] = bool(cli.get("remove_zero_variance"))  # Store boolean override

        return prep_ov  # Return the collected preprocessing overrides
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_cv_overrides(cli: dict) -> dict:
    """
    Build the cross-validation section overrides from CLI arguments.

    :param cli: Dictionary of parsed CLI arguments.
    :return: Nested override dict for the cross_validation section or empty dict.
    """

    try:
        cv_ov = {}  # Collect cross-validation-related override values
        raw_n_folds: Any = cli.get("n_folds")  # Retrieve raw CLI value for n_folds

        if raw_n_folds is not None:  # Only process when provided
            if not isinstance(raw_n_folds, (int, str)):  # Validate expected input types
                raise TypeError("--n_folds must be an int or string convertible to int")  # Raise on invalid type
            try:
                cv_ov.setdefault("cross_validation", {})["n_folds"] = int(raw_n_folds)  # Safely convert to int
            except Exception as e:
                raise ValueError(f"Invalid --n_folds value: {raw_n_folds}") from e  # Raise with context

        return cv_ov  # Return the collected cross-validation overrides
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_multiprocessing_overrides(cli: dict) -> dict:
    """
    Build the multiprocessing section overrides from CLI arguments.

    :param cli: Dictionary of parsed CLI arguments.
    :return: Nested override dict for the multiprocessing section or empty dict.
    """

    try:
        multi_ov = {}  # Collect multiprocessing-related override values
        raw_n_jobs: Any = cli.get("n_jobs")  # Retrieve raw CLI value for n_jobs

        if raw_n_jobs is not None:  # Only process when provided
            if not isinstance(raw_n_jobs, (int, str)):  # Validate acceptable types
                raise TypeError("--n_jobs must be an int or string convertible to int")  # Raise on invalid type
            try:
                multi_ov.setdefault("multiprocessing", {})["n_jobs"] = int(raw_n_jobs)  # Convert to int safely
            except Exception as e:
                raise ValueError(f"Invalid --n_jobs value: {raw_n_jobs}") from e  # Raise with context

        raw_cpu_processes: Any = cli.get("cpu_processes")  # Retrieve raw CLI value for cpu_processes

        if raw_cpu_processes is not None:  # Only process when provided
            if not isinstance(raw_cpu_processes, (int, str)):  # Validate acceptable types
                raise TypeError("--cpu_processes must be an int or string convertible to int")  # Raise on invalid type
            try:
                multi_ov.setdefault("multiprocessing", {})["cpu_processes"] = int(raw_cpu_processes)  # Convert to int safely
            except Exception as e:
                raise ValueError(f"Invalid --cpu_processes value: {raw_cpu_processes}") from e  # Raise with context

        return multi_ov  # Return the collected multiprocessing overrides
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_caching_overrides(cli: dict) -> dict:
    """
    Build the caching section overrides from CLI arguments.

    :param cli: Dictionary of parsed CLI arguments.
    :return: Nested override dict for the caching section or empty dict.
    """

    try:
        cache_ov = {}  # Collect caching-related override values

        if cli.get("caching_enabled") is not None:  # Verify if caching_enabled was provided
            cache_ov.setdefault("caching", {})["enabled"] = bool(cli.get("caching_enabled"))  # Store boolean caching override

        raw_pickle_protocol: Any = cli.get("pickle_protocol")  # Retrieve raw CLI value for pickle_protocol

        if raw_pickle_protocol is not None:  # Only process when provided
            if not isinstance(raw_pickle_protocol, (int, str)):  # Validate types
                raise TypeError("--pickle_protocol must be an int or string convertible to int")  # Raise on invalid type
            try:
                cache_ov.setdefault("caching", {})["pickle_protocol"] = int(raw_pickle_protocol)  # Safely convert to int
            except Exception as e:
                raise ValueError(f"Invalid --pickle_protocol value: {raw_pickle_protocol}") from e  # Raise with context

        return cache_ov  # Return the collected caching overrides
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_sources_map(file_cfg: dict, pca_overrides: dict) -> dict:
    """
    Build a sources map indicating the origin of each top-level PCA config section.

    :param file_cfg: The file-loaded configuration dictionary.
    :param pca_overrides: The CLI-derived nested override dictionary.
    :return: Dict mapping each pca config section to its origin ('cli', 'config', or 'default').
    """

    try:
        sources = {"pca": {}}  # Initialize the sources map with the pca key

        for top in ("execution", "model", "dimensionality", "preprocessing", "cross_validation", "multiprocessing", "caching", "export"):  # Iterate all tracked config sections
            src = "default"  # Start with default as the baseline origin
            if isinstance(file_cfg, dict) and file_cfg.get("pca", {}).get(top) is not None:  # Verify if section is present in file config
                src = "config"  # Mark as config-sourced when found in file
            if pca_overrides.get("pca", {}).get(top) is not None:  # Verify if section was overridden via CLI
                src = "cli"  # Mark as cli-sourced when CLI override exists
            sources["pca"][top] = src  # Store the origin label for this section

        return sources  # Return the completed sources map
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_config() -> tuple:
    """
    Produce the final configuration using precedence: CLI > config.yaml > defaults.

    Returns (config_dict, sources_dict) where sources_dict maps top-level keys
    to the origin ('cli'|'config'|'default').

    :return: Tuple of (config_dict, sources_dict) with merged configuration and origin map.
    """

    try:
        cli = parse_cli_args()  # Parse all CLI arguments into a flat dictionary

        cfg_path = resolve_config_file_path(cli.get("config"))  # Resolve configuration file path from CLI or auto-detection

        file_cfg = {}  # Initialize file config as empty
        if cfg_path:  # Verify if a config file path was resolved
            try:
                file_cfg = load_config_file(cfg_path)  # Load and parse the YAML config file
            except Exception as e:
                raise  # Re-raise loading errors for caller to handle

        defaults = get_default_config()  # Retrieve the complete set of default configuration values

        merged = deep_merge_dicts(defaults, file_cfg)  # Merge defaults with file config, file taking precedence

        pca_overrides = {}  # Prepare dict to collect CLI overrides for PCA section
        pca_overrides.setdefault("pca", {})  # Ensure top-level 'pca' mapping exists in overrides

        exec_ov = build_execution_overrides(cli)  # Build execution section overrides from CLI
        if exec_ov:  # Only update when overrides are present
            pca_overrides["pca"].update(exec_ov)  # Merge execution overrides into pca_overrides

        model_ov = build_model_overrides(cli)  # Build model section overrides from CLI
        if model_ov:  # Only update when overrides are present
            pca_overrides["pca"].update(model_ov)  # Merge model overrides into pca_overrides

        export_ov = build_export_overrides(cli)  # Build export section overrides from CLI
        if export_ov:  # Only update when overrides are present
            pca_overrides["pca"].update(export_ov)  # Merge export overrides into pca_overrides

        dim_ov = build_dimensionality_overrides(cli)  # Build dimensionality section overrides from CLI
        if dim_ov:  # Only update when overrides are present
            pca_overrides["pca"].update(dim_ov)  # Merge dimensionality overrides into pca_overrides

        prep_ov = build_preprocessing_overrides(cli)  # Build preprocessing section overrides from CLI
        if prep_ov:  # Only update when overrides are present
            pca_overrides["pca"].update(prep_ov)  # Merge preprocessing overrides into pca_overrides

        cv_ov = build_cv_overrides(cli)  # Build cross-validation section overrides from CLI
        if cv_ov:  # Only update when overrides are present
            pca_overrides["pca"].update(cv_ov)  # Merge cross-validation overrides into pca_overrides

        multi_ov = build_multiprocessing_overrides(cli)  # Build multiprocessing section overrides from CLI
        if multi_ov:  # Only update when overrides are present
            pca_overrides["pca"].update(multi_ov)  # Merge multiprocessing overrides into pca_overrides

        cache_ov = build_caching_overrides(cli)  # Build caching section overrides from CLI
        if cache_ov:  # Only update when overrides are present
            pca_overrides["pca"].update(cache_ov)  # Merge caching overrides into pca_overrides

        final = deep_merge_dicts(merged, pca_overrides)  # Merge merged config with CLI overrides

        validate_config_structure(final.get("pca", {}))  # Validate the final merged config structure before use

        sources = build_sources_map(file_cfg, pca_overrides)  # Build the origin map for all config sections

        return final, sources  # Return the final config dict and sources map
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def verbose_output(true_string="", false_string=""):
    """
    Output a message if the VERBOSE constant is set to True.

    :param true_string: The string to be outputted if the VERBOSE constant is set to True.
    :param false_string: The string to be outputted if the VERBOSE constant is set to False.
    :return: None.
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

        global TELEGRAM_BOT  # Declare the module-global telegram_bot variable

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
            )  # Log preprocessing start with zero-variance removal enabled
        else:  # If remove_zero_variance is set to False
            verbose_output(
                f"{BackgroundColors.GREEN}Preprocessing DataFrame: "
                f"{BackgroundColors.CYAN}normalizing and sanitizing column names and removing NaN/infinite rows"
                f"{BackgroundColors.GREEN}.{Style.RESET_ALL}"
            )  # Log preprocessing start without zero-variance removal

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


def scale_and_split(X, y, test_size=0.2, random_state=42, scale_data=True):
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

        scaler = None  # Initialize scaler to None before optional scaling step
        scaling_time = 0.0  # Initialize scaling time accumulator to zero
        if scale_data:  # Apply feature scaling only when scale_data flag is enabled
            start_scaling = time.perf_counter()  # Start high-resolution scaling timer
            scaler = StandardScaler()  # Create scaler instance
            X_train = scaler.fit_transform(X_train_df)  # Fit scaler and transform train
            X_test = scaler.transform(X_test_df)  # Transform test data with fitted scaler
            scaling_time = round(time.perf_counter() - start_scaling, 6)  # End scaling timer and round to 6 decimals
            try:  # Safely attach scaling time to scaler instance using setattr to avoid static attribute-access diagnostics
                setattr(scaler, "_scaling_time", scaling_time)  # Store scaling time as dynamic attribute on scaler
            except Exception:  # Preserve prior silent-failure behavior if attribute cannot be set
                pass  # No-op on failure
        else:
            X_train = X_train_df.values if hasattr(X_train_df, "values") else np.asarray(X_train_df)  # Convert training DataFrame to numpy array without scaling
            X_test = X_test_df.values if hasattr(X_test_df, "values") else np.asarray(X_test_df)  # Convert testing DataFrame to numpy array without scaling

        return X_train, X_test, y_train, y_test, scaler  # Return the split data and optional scaler
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def validate_pca_n_components(n_components: int, n_features: int) -> None:
    """
    Validate that n_components is a positive integer not exceeding the number of features.

    :param n_components: Number of principal components to validate.
    :param n_features: Number of available features in the training data.
    :return: None
    """

    try:
        if n_components <= 0:  # Verify n_components is positive
            raise ValueError(f"n_components must be positive, got {n_components}")  # Raise on invalid value

        if n_components > n_features:  # Verify n_components does not exceed feature count
            raise ValueError(  # Raise descriptive error
                f"n_components ({n_components}) cannot be greater than number of features ({n_features})"
            )  # End raise
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def fit_transform_pca(X_train, X_test, n_components: int, scaling_time: float, random_state: int) -> tuple:
    """
    Fit PCA on training data and transform both training and test sets.

    :param X_train: Scaled training feature array.
    :param X_test: Scaled test feature array.
    :param n_components: Number of principal components to retain.
    :param scaling_time: Elapsed scaling time to include in feature extraction timing.
    :param random_state: Random seed for PCA reproducibility.
    :return: Tuple of (X_train_pca, X_test_pca, pca, feature_extraction_time_s, explained_variance).
    """

    try:
        pca = PCA(n_components=n_components, random_state=random_state)  # Initialize PCA with requested components

        start_pca = time.perf_counter()  # Start PCA timer (feature extraction part)
        X_train_pca = pca.fit_transform(X_train)  # Fit PCA on training data and transform
        X_test_pca = pca.transform(X_test)  # Transform test data using the fitted PCA
        pca_time = round(time.perf_counter() - start_pca, 6)  # End PCA timer and round

        feature_extraction_time_s = round((scaling_time or 0.0) + pca_time, 6)  # Sum scaling and PCA times

        explained_variance = pca.explained_variance_ratio_.sum()  # Total explained variance ratio

        return X_train_pca, X_test_pca, pca, feature_extraction_time_s, explained_variance  # Return all PCA results
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_rf_classifier(random_state: int, workers: int) -> RandomForestClassifier:
    """
    Build and return a RandomForestClassifier with configured jobs and random state.

    :param random_state: Random seed for the classifier.
    :param workers: Number of parallel workers to determine n_jobs setting.
    :return: Configured RandomForestClassifier instance.
    """

    try:
        global N_JOBS
        rf_n_jobs = N_JOBS if (N_JOBS is not None) else (-1 if workers == 1 else 1)  # Determine n_jobs from globals or workers arg

        model = RandomForestClassifier(  # Build the classifier with validated parameters
            n_estimators=100, random_state=random_state, n_jobs=rf_n_jobs
        )  # End RandomForestClassifier

        return model  # Return the configured classifier
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_cv_folds(model: RandomForestClassifier, X_train_pca, y_train, skf: StratifiedKFold, n_components: int, n_folds: int) -> tuple:
    """
    Execute stratified cross-validation folds and accumulate per-fold metrics.

    :param model: Configured RandomForestClassifier instance.
    :param X_train_pca: PCA-transformed training feature array.
    :param y_train: Training target array or Series.
    :param skf: Configured StratifiedKFold splitter.
    :param n_components: Number of PCA components used for Telegram notifications.
    :param n_folds: Total number of CV folds used in Telegram notifications.
    :return: Tuple of (cv_accs, cv_precs, cv_recs, cv_f1s, total_training_time, total_testing_time).
    """

    try:
        cv_accs, cv_precs, cv_recs, cv_f1s = [], [], [], []  # Initialize per-fold metric accumulators
        total_training_time = 0.0  # Accumulator for all model.fit durations
        total_testing_time = 0.0  # Accumulator for all prediction+metric durations

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train_pca, y_train), start=1):  # Loop folds
            send_telegram_message(TELEGRAM_BOT, f"Starting CV fold {fold_idx}/{n_folds} for n_components={n_components}")  # Notify
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
            send_telegram_message(TELEGRAM_BOT, f"Finished CV fold {fold_idx}/{n_folds} for n_components={n_components} with F1: {f1_fold}")  # Notify fold completion

        return cv_accs, cv_precs, cv_recs, cv_f1s, total_training_time, total_testing_time  # Return all accumulated CV metrics
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def compute_test_metrics(model: RandomForestClassifier, X_train_pca, y_train, X_test_pca, y_test) -> tuple:
    """
    Fit the model on the full training set and evaluate on the test set.

    :param model: Configured RandomForestClassifier instance.
    :param X_train_pca: PCA-transformed full training feature array.
    :param y_train: Full training target array or Series.
    :param X_test_pca: PCA-transformed test feature array.
    :param y_test: Test target array or Series.
    :return: Tuple of (acc, prec, rec, f1, fpr, fnr, final_fit_elapsed, test_pred_elapsed).
    """

    try:
        start_final_fit = time.perf_counter()  # Start timer immediately before final model.fit
        model.fit(X_train_pca, y_train)  # Fit model on full training data
        final_fit_elapsed = round(time.perf_counter() - start_final_fit, 6)  # Stop timer immediately after final fit and round

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

        return acc, prec, rec, f1, fpr, fnr, final_fit_elapsed, test_pred_elapsed  # Return all test evaluation metrics
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def apply_pca_and_evaluate(X_train, y_train, X_test, y_test, n_components, cv_folds=10, workers=1, scaling_time=0.0, random_state=42):
    """
    Applies PCA transformation and evaluates performance using 10-fold Stratified Cross-Validation.

    :param X_train: Training features (scaled).
    :param y_train: Training target.
    :param X_test: Testing features (scaled).
    :param y_test: Testing target.
    :param n_components: Number of principal components to keep.
    :param cv_folds: Number of cross-validation folds (default: 10).
    :param workers: Number of parallel workers for job configuration.
    :param scaling_time: Elapsed scaling time included in feature extraction timing.
    :param random_state: Random seed for reproducibility.
    :return: Dictionary containing metrics, explained variance, and PCA object.
    """

    try:
        validate_pca_n_components(n_components, X_train.shape[1])  # Validate n_components before proceeding

        send_telegram_message(TELEGRAM_BOT, f"Starting PCA training for n_components={n_components}")  # Notify

        if random_state is None:
            random_state = 42  # Fall back to default random state when None

        X_train_pca, X_test_pca, pca, feature_extraction_time_s, explained_variance = fit_transform_pca(
            X_train, X_test, n_components, scaling_time, random_state
        )  # Fit and transform PCA on training and test data

        model = build_rf_classifier(random_state, workers)  # Initialize RF classifier with job configuration

        if not isinstance(cv_folds, int) or cv_folds < 2:  # Verify cv_folds is a valid integer >= 2
            raise ValueError(f"cv_folds must be integer >= 2, got {cv_folds}")  # Raise on invalid cv_folds
        if cv_folds > len(X_train):  # Verify cv_folds does not exceed sample count
            raise ValueError(f"cv_folds ({cv_folds}) cannot exceed number of training samples ({len(X_train)})")  # Raise on too many folds
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)  # Initialize stratified splitter

        cv_accs, cv_precs, cv_recs, cv_f1s, total_training_time, total_testing_time = run_cv_folds(
            model, X_train_pca, y_train, skf, n_components, cv_folds
        )  # Run all CV folds and collect per-fold metrics

        cv_acc_mean = np.mean(cv_accs)  # Mean CV accuracy
        cv_prec_mean = np.mean(cv_precs)  # Mean CV precision
        cv_rec_mean = np.mean(cv_recs)  # Mean CV recall
        cv_f1_mean = np.mean(cv_f1s)  # Mean CV f1

        send_telegram_message(TELEGRAM_BOT, f"Finished PCA training for n_components={n_components} with CV F1: {cv_f1_mean}")  # Notify completion

        acc, prec, rec, f1, fpr, fnr, final_fit_elapsed, test_pred_elapsed = compute_test_metrics(
            model, X_train_pca, y_train, X_test_pca, y_test
        )  # Evaluate on the held-out test set

        total_training_time += final_fit_elapsed  # Add final fit duration to training total
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
        )  # Print PCA results header with n_components
        print(
            f"  {BackgroundColors.GREEN}Explained Variance Ratio: {BackgroundColors.CYAN}{results['explained_variance']} ({results['explained_variance']*100}%){Style.RESET_ALL}"
        )  # Print explained variance ratio
        print(
            f"\n  {BackgroundColors.BOLD}{BackgroundColors.GREEN}10-Fold Cross-Validation Metrics (Training Set):{Style.RESET_ALL}"
        )  # Print cross-validation section header
        print(
            f"  {BackgroundColors.GREEN}CV Accuracy: {BackgroundColors.CYAN}{results['cv_accuracy']}{Style.RESET_ALL}"
        )  # Print CV accuracy
        print(
            f"  {BackgroundColors.GREEN}CV Precision: {BackgroundColors.CYAN}{results['cv_precision']}{Style.RESET_ALL}"
        )  # Print CV precision
        print(f"  {BackgroundColors.GREEN}CV Recall: {BackgroundColors.CYAN}{results['cv_recall']}{Style.RESET_ALL}")  # Print CV recall
        print(
            f"  {BackgroundColors.GREEN}CV F1-Score: {BackgroundColors.CYAN}{results['cv_f1_score']}{Style.RESET_ALL}"
        )  # Print CV F1-score
        print(f"\n  {BackgroundColors.BOLD}{BackgroundColors.GREEN}Test Set Metrics:{Style.RESET_ALL}")  # Print test set section header
        print(
            f"  {BackgroundColors.GREEN}Test Accuracy: {BackgroundColors.CYAN}{results['test_accuracy']}{Style.RESET_ALL}"
        )  # Print test accuracy
        print(
            f"  {BackgroundColors.GREEN}Test Precision: {BackgroundColors.CYAN}{results['test_precision']}{Style.RESET_ALL}"
        )  # Print test precision
        print(
            f"  {BackgroundColors.GREEN}Test Recall: {BackgroundColors.CYAN}{results['test_recall']}{Style.RESET_ALL}"
        )  # Print test recall
        print(
            f"  {BackgroundColors.GREEN}Test F1-Score: {BackgroundColors.CYAN}{results['test_f1_score']}{Style.RESET_ALL}"
        )  # Print test F1-score
        print(f"  {BackgroundColors.GREEN}Test FPR: {BackgroundColors.CYAN}{results['test_fpr']}{Style.RESET_ALL}")  # Print test false positive rate
        print(f"  {BackgroundColors.GREEN}Test FNR: {BackgroundColors.CYAN}{results['test_fnr']}{Style.RESET_ALL}")  # Print test false negative rate
        print(
            f"  {BackgroundColors.GREEN}Training Time: {BackgroundColors.CYAN}{int(round(results['training_time_s']))}s  Testing Time: {BackgroundColors.CYAN}{int(round(results['testing_time_s']))}s{Style.RESET_ALL}"
        )  # Print training and testing elapsed times
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

def row_style_for_zebra(row):
    """
    Top-level function to produce zebra row styles for pandas Styler.

    :param row: pandas Series representing a row
    :return: List[str] of CSS style strings for each cell
    """

    try:
        bg = "white" if (row.name % 2) == 0 else "#f2f2f2"  # White for even rows, light gray for odd rows
        return [f"background-color: {bg};" for _ in row.index]  # Return style for every column in the row
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


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
        dfi.export(cast(Any, styled_df), str(out_p))  # Use dataframe_image to export styled DataFrame to PNG with cast to satisfy static typing
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
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


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
        df.to_csv(str(csv_p), index=False)  # Persist DataFrame to CSV preserving column order and content
        if is_visualizable and len(df) <= 100:  # Only generate image when visualizable and within the safe row limit
            try:  # Guard PNG rendering to keep CSV persistence independent from image export
                png_path = csv_p.with_suffix('.png')  # Construct PNG path by replacing extension
                generate_table_image_from_dataframe(df, png_path)  # Generate PNG from in-memory DataFrame
            except Exception as _png_e:  # Contain PNG rendering failures locally
                print(f"{BackgroundColors.YELLOW}Warning: PNG generation failed for {csv_p.name}: {_png_e}{Style.RESET_ALL}")  # Warn and continue without propagating PNG errors
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise

        
def resolve_pca_output_dir(cfg: Any, csv_path: str) -> tuple:
    """
    Resolve and validate the output directory and CSV file path for PCA results.

    :param cfg: PCA configuration dictionary containing the export section.
    :param csv_path: Original input CSV dataset path used for relative path resolution.
    :return: Tuple of (resolved_dir, csv_output) as absolute path strings.
    """

    export_cfg = (cfg or {}).get("export", {})  # Retrieve the export sub-config safely
    results_dir_raw = export_cfg.get("results_dir")  # Read configured results directory path
    results_filename = export_cfg.get("results_filename")  # Read configured results filename

    if not isinstance(results_dir_raw, str) or not results_dir_raw:  # Verify results_dir is a non-empty string
        raise ValueError("'pca.export.results_dir' must be a non-empty string in configuration")  # Raise on invalid
    if not isinstance(results_filename, str) or not results_filename:  # Verify results_filename is a non-empty string
        raise ValueError("'pca.export.results_filename' must be a non-empty string in configuration")  # Raise on invalid
    if not results_filename.lower().endswith(".csv"):  # Verify results_filename ends with .csv
        raise ValueError("'pca.export.results_filename' must end with .csv")  # Raise on wrong extension

    if os.path.isabs(results_dir_raw):  # Verify if given path is absolute
        resolved_dir = os.path.abspath(os.path.expanduser(results_dir_raw))  # Expand and resolve absolute path
    else:  # Handle relative path resolution against dataset directory
        dataset_dir = os.path.dirname(csv_path) or "."  # Use dataset directory as base for relative paths
        resolved_dir = os.path.abspath(os.path.expanduser(os.path.join(dataset_dir, results_dir_raw)))  # Resolve relative path

    os.makedirs(resolved_dir, exist_ok=True)  # Create the output directory if it does not exist
    if not os.access(resolved_dir, os.W_OK):  # Verify write permission on the resolved directory
        raise PermissionError(f"Directory not writable: {resolved_dir}")  # Raise on non-writable directory

    csv_output = os.path.join(resolved_dir, results_filename)  # Build the full CSV output path

    return resolved_dir, csv_output  # Return the resolved output directory and CSV file path


def build_result_rows(all_results: list, csv_path: str) -> list:
    """
    Build a list of normalized row dictionaries from PCA result entries.

    :param all_results: List of per-configuration result dicts from apply_pca_and_evaluate.
    :param csv_path: Original input CSV dataset path for the dataset column value.
    :return: List of row dictionaries ready for DataFrame construction.
    """

    eval_model = "Random Forest"  # Evaluation model descriptor
    train_test_split_desc = "80/20 split"  # Train/test split descriptor
    scaling = "StandardScaler"  # Scaling method descriptor
    evaluator_fallback = {"model": "RandomForestClassifier", "n_estimators": 100, "random_state": 42, "n_jobs": -1}  # Fallback hyperparameters
    cv_method = "StratifiedKFold(n_splits=10)"  # Cross-validation method descriptor

    rows = []  # List to store one normalized row per result

    for results in all_results:  # Iterate over each PCA configuration result dict
        eval_params = None  # Initialize eval_params for this result
        trained_clf = results.get("trained_classifier")  # Retrieve trained classifier object if present
        if trained_clf is not None:  # Only attempt to extract params when classifier exists
            try:
                eval_params = trained_clf.get_params()  # Try to get estimator parameters
            except Exception:
                eval_params = None  # Fallback to None if inspection fails

        raw_n_components: Any = results.get("n_components")  # Retrieve raw n_components from result (may be None)
        if raw_n_components is None:  # If not present, keep None to match original behavior
            validated_n_components = None  # Preserve original None semantics when missing
        else:
            if not isinstance(raw_n_components, (int, str)):  # Validate acceptable input types
                raise TypeError(f"results['n_components'] must be int or str convertible to int, got {type(raw_n_components)}")  # Raise on invalid type
            try:
                validated_n_components = int(raw_n_components)  # Convert to int after validation
            except Exception as e:
                raise ValueError(f"Invalid n_components value in results: {raw_n_components}") from e  # Raise with context

        row = {  # Build the normalized row mapping for CSV export
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"),  # Current timestamp for this row
            "tool": "PCA",  # Tool name
            "model": eval_model,  # Model descriptor
            "dataset": os.path.relpath(csv_path),  # Relative dataset path
            "hyperparameters": json.dumps(eval_params or evaluator_fallback, sort_keys=True, ensure_ascii=False, default=str),  # JSON-encoded params
            "cv_method": cv_method,  # CV method string
            "train_test_split": train_test_split_desc,  # Train/test split descriptor
            "scaling": scaling,  # Scaling descriptor
            "n_components": validated_n_components,  # Validated integer or None
            "explained_variance": results.get("explained_variance"),  # Explained variance formatted
            "cv_accuracy": results.get("cv_accuracy"),  # CV accuracy preserved at full precision
            "cv_precision": results.get("cv_precision"),  # CV precision preserved at full precision
            "cv_recall": results.get("cv_recall"),  # CV recall preserved at full precision
            "cv_f1_score": results.get("cv_f1_score"),  # CV F1 preserved at full precision
            "test_accuracy": results.get("test_accuracy"),  # Test accuracy preserved at full precision
            "test_precision": results.get("test_precision"),  # Test precision preserved at full precision
            "test_recall": results.get("test_recall"),  # Test recall preserved at full precision
            "test_f1_score": results.get("test_f1_score"),  # Test F1 preserved at full precision
            "test_fpr": results.get("test_fpr"),  # Test FPR preserved at full precision
            "test_fnr": results.get("test_fnr"),  # Test FNR preserved at full precision
            "training_time_s": results.get("training_time_s") if results.get("training_time_s") is not None else None,  # Training time or None
            "testing_time_s": results.get("testing_time_s") if results.get("testing_time_s") is not None else None,  # Testing time or None
            "feature_extraction_time_s": results.get("feature_extraction_time_s") if results.get("feature_extraction_time_s") is not None else None,  # Feature extraction time or None
        }
        rows.append(row)  # Append the normalized row to the rows collection

    return rows  # Return the list of all normalized result rows


def merge_or_create_results_df(csv_output: str, comparison_df: pd.DataFrame, header: list) -> pd.DataFrame:
    """
    Merge new results with an existing CSV or return the new DataFrame directly.

    :param csv_output: Absolute path to the output CSV file to read from and merge into.
    :param comparison_df: DataFrame containing the new result rows to add.
    :param header: Ordered list of column names defining the CSV structure.
    :return: Merged and sorted DataFrame ready for saving.
    """

    try:
        if verify_filepath_exists(csv_output):  # Verify if a pre-existing CSV file was found
            try:
                df_existing = pd.read_csv(csv_output, dtype=str)  # Read existing CSV with all columns as strings
                if "timestamp" not in df_existing.columns:  # Verify if timestamp column is missing
                    mtime = os.path.getmtime(csv_output)  # Get file modification time as fallback timestamp
                    back_ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d_%H_%M_%S")  # Format modification time
                    df_existing["timestamp"] = back_ts  # Assign fallback timestamp to all existing rows
                for c in header:  # Iterate expected header columns
                    if c not in df_existing.columns:  # Verify if column is missing from existing data
                        df_existing[c] = None  # Add missing column with None values

                df_combined = pd.concat([df_existing[header], comparison_df], ignore_index=True, sort=False)  # Concatenate existing and new rows
                try:
                    df_combined["timestamp_dt"] = pd.to_datetime(df_combined["timestamp"], format="%Y-%m-%d_%H_%M_%S", errors="coerce")  # Parse timestamps for sorting
                    df_combined = df_combined.sort_values(by="timestamp_dt", ascending=False)  # Sort by parsed timestamp descending
                    df_combined = df_combined.drop(columns=["timestamp_dt"])  # Remove temporary sorting column
                except Exception:
                    df_combined = df_combined.sort_values(by="timestamp", ascending=False)  # Fallback sort by raw timestamp string

                return df_combined.reset_index(drop=True)  # Reset index after merge and sorting
            except Exception:
                return comparison_df  # Return new rows only when existing CSV read fails

        return comparison_df  # Return new rows only when no prior CSV exists
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def resolve_pickle_protocol(cfg: dict) -> int:
    """
    Resolve the pickle protocol to use for serializing model artifacts.

    :param cfg: PCA configuration dictionary containing the caching section.
    :return: Validated integer pickle protocol value.
    """

    try:
        if PICKLE_PROTOCOL is not None:  # Verify if global PICKLE_PROTOCOL was explicitly set
            return int(PICKLE_PROTOCOL)  # Return the explicitly configured global protocol

        cache_cfg: Any = (cfg or {}).get("caching")  # Safely retrieve 'caching' mapping from cfg

        if cache_cfg is None:  # Verify if caching section is missing
            return 4  # Return the default pickle protocol fallback

        raw_proto: Any = cache_cfg.get("pickle_protocol")  # Raw value from config (may be None)

        if raw_proto is None:  # Verify if protocol is unspecified in config
            return 4  # Return the default pickle protocol fallback
        if not isinstance(raw_proto, (int, str)):  # Validate acceptable types
            raise TypeError("cfg['caching']['pickle_protocol'] must be int or string convertible to int")  # Raise on invalid type
        try:
            return int(raw_proto)  # Convert validated value to int and return
        except Exception as e:
            raise ValueError(f"Invalid pickle_protocol value in config: {raw_proto}") from e  # Raise with context
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def save_pca_model_artifacts(all_results: list, models_dir: str, base_name: str, timestamp: str, cfg: Any) -> None:
    """
    Save PCA objects, scalers, and trained classifiers to disk for each configuration.

    :param all_results: List of per-configuration result dicts containing model artifacts.
    :param models_dir: Directory path where model artifacts will be saved.
    :param base_name: Base filename stem derived from the dataset path.
    :param timestamp: Formatted timestamp string used in artifact filenames.
    :param cfg: PCA configuration dictionary used for pickle protocol resolution.
    :return: None
    """

    try:
        os.makedirs(models_dir, exist_ok=True)  # Ensure the models directory exists before saving

        for results in all_results:  # Iterate over each PCA configuration result
            n_comp = results["n_components"]  # Retrieve the number of components for filename construction

            pca_obj = results.get("pca_object")  # Retrieve the PCA object if present
            if pca_obj:  # Only save when a PCA object exists
                pca_file = f"{models_dir}/PCA-{base_name}-{n_comp}c-{timestamp}.pkl"  # Build the PCA artifact filename
                try:
                    proto = resolve_pickle_protocol(cfg)  # Resolve pickle protocol from config
                    with open(pca_file, "wb") as f:  # Open file for binary write
                        pickle.dump(pca_obj, f, protocol=int(proto))  # Dump PCA object using validated protocol
                    verbose_output(f"{BackgroundColors.GREEN}PCA object saved to {BackgroundColors.CYAN}{pca_file}{Style.RESET_ALL}")  # Log successful PCA save
                except Exception as e:
                    print(f"{BackgroundColors.RED}Failed to save PCA object: {e}{Style.RESET_ALL}")  # Log PCA save failure

            scaler = results.get("scaler")  # Retrieve the scaler if present
            if scaler is not None:  # Only save when a scaler exists
                scaler_path = f"{models_dir}/PCA-{base_name}-{n_comp}c-{timestamp}-scaler.joblib"  # Build the scaler artifact filename
                try:
                    proto = resolve_pickle_protocol(cfg)  # Resolve pickle protocol from config
                    dump(scaler, scaler_path, protocol=int(proto))  # Dump scaler using validated protocol
                    verbose_output(f"{BackgroundColors.GREEN}Scaler saved to {BackgroundColors.CYAN}{scaler_path}{Style.RESET_ALL}")  # Log successful scaler save
                except Exception as e:
                    print(f"{BackgroundColors.RED}Failed to save scaler: {e}{Style.RESET_ALL}")  # Log scaler save failure

            clf = results.get("trained_classifier")  # Retrieve the trained classifier if present
            if clf is not None:  # Only save when a classifier exists
                model_path = f"{models_dir}/PCA-{base_name}-{n_comp}c-{timestamp}-model.joblib"  # Build the classifier artifact filename
                try:
                    proto = resolve_pickle_protocol(cfg)  # Resolve pickle protocol from config
                    dump(clf, model_path, protocol=int(proto))  # Dump classifier with validated protocol
                    verbose_output(f"{BackgroundColors.GREEN}Trained classifier saved to {BackgroundColors.CYAN}{model_path}{Style.RESET_ALL}")  # Log successful classifier save
                except Exception as e:
                    print(f"{BackgroundColors.RED}Failed to save classifier: {e}{Style.RESET_ALL}")  # Log classifier save failure
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def save_pca_results(csv_path, all_results, cfg=None):
    """
    Saves PCA results to a single CSV file containing evaluation metadata
    and per-configuration metrics. This replaces separate JSON and CSV
    outputs with one machine-friendly CSV that includes the evaluation
    details repeated on each row for easy consumption by downstream tools.

    :param csv_path: Original CSV file path.
    :param all_results: List of result dictionaries from different PCA configurations.
    :param cfg: PCA configuration dictionary.
    :return: None
    """

    try:
        resolved_dir, csv_output = resolve_pca_output_dir(cfg, csv_path)  # Resolve and validate the output directory and CSV path

        print(f"{BackgroundColors.GREEN}Exporting PCA results to CSV: {BackgroundColors.CYAN}{csv_output}{Style.RESET_ALL}")  # Log the export target path

        rows = build_result_rows(all_results, csv_path)  # Build normalized row dicts from all result entries
        comparison_df = pd.DataFrame(rows)  # Create DataFrame from rows

        header = None  # Initialize header to None before attempting lookup
        if cfg and isinstance(cfg, dict):  # Verify if cfg dict is available
            header = (cfg or {}).get("export", {}).get("results_csv_columns")  # Extract header from config
        if not header:  # Verify if header was found
            raise ValueError("PCA results CSV header is missing from configuration: pca.export.results_csv_columns")  # Raise on missing header

        df_out = merge_or_create_results_df(csv_output, comparison_df, header)  # Merge with existing CSV or use new rows
        df_out = populate_hardware_column(df_out, column_name="hardware")  # Add hardware specs column (lowercase)

        try:
            df_out = df_out.reindex(columns=header)  # Reorder columns to match the configured header
        except Exception:
            pass  # Preserve original silent-failure behavior on reindex error

        try:  # Attempt to save the CSV
            generate_csv_and_image(df_out, csv_output, is_visualizable=True)  # Save CSV and generate PNG image
            print(f"{BackgroundColors.GREEN}PCA results saved to {BackgroundColors.CYAN}{csv_output}{Style.RESET_ALL}")  # Log successful save
        except Exception as e:  # Handle exceptions during file saving
            print(f"{BackgroundColors.RED}Failed to save PCA CSV: {e}{Style.RESET_ALL}")  # Log file save failure

        models_dir = os.path.join(resolved_dir, "Models")  # Build the models subdirectory path
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")  # Generate timestamp for artifact filenames
        base_name = Path(csv_path).stem  # Extract the base name from the dataset path

        save_pca_model_artifacts(all_results, models_dir, base_name, timestamp, cfg)  # Save all model artifacts for each configuration

        verbose_output(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}PCA Configuration Comparison:{Style.RESET_ALL}")  # Log comparison header
        verbose_output(comparison_df.to_string(index=False))  # Output the comparison DataFrame
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def should_skip_existing_models(n_components_list: list, models_dir: str, base_name: str) -> bool:
    """
    Determine if training should be skipped because exported models already exist.

    :param n_components_list: List of n_components values to verify model existence for.
    :param models_dir: Directory path where model joblib files are stored.
    :param base_name: Base filename stem for matching stored model files.
    :return: True if at least one matching model exists and training should be skipped, False otherwise.
    """

    try:
        found = False  # Initialize found flag to False before scanning

        for n_comp in n_components_list:  # Iterate over each component count to scan for matching models
            pattern = f"{models_dir}/PCA-{base_name}-{n_comp}c-*-model.joblib"  # Build glob pattern for this n_components value
            matches = glob.glob(pattern)  # Search for matching model files in models directory
            if matches:  # Verify if any matching model files were found
                print(f"{BackgroundColors.GREEN}Found exported model for n_components={n_comp}, skipping training.{Style.RESET_ALL}")  # Log the found model
                found = True  # Mark that at least one model was found

        if found:  # Verify if any model was found across all component counts
            print(f"{BackgroundColors.GREEN}SKIP_TRAIN_IF_MODEL_EXISTS: At least one model exists, skipping retraining for all configs.{Style.RESET_ALL}")  # Log the skip decision

        return found  # Return whether any matching exported model was found
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def load_and_prepare_pca_dataset(csv_path: str, remove_zero_variance: bool) -> tuple | None:
    """
    Load the dataset, preprocess it, and extract feature and target arrays.

    :param csv_path: Path to the CSV dataset file.
    :param remove_zero_variance: Whether to drop zero-variance numeric features during preprocessing.
    :return: Tuple of (X, y) as DataFrame and Series, or None if loading fails.
    """

    try:
        df = load_dataset(csv_path)  # Load the raw dataset from CSV
        if df is None:  # Verify if dataset loading failed
            return None  # Return None to signal failure

        cleaned_df = preprocess_dataframe(df, remove_zero_variance=remove_zero_variance)  # Preprocess the DataFrame
        X = cleaned_df.select_dtypes(include=["number"]).iloc[:, :-1]  # Select numeric features (all columns except last)
        y = cleaned_df.iloc[:, -1]  # Select target variable (last column)

        return X, y  # Return the feature matrix and target vector
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def validate_and_filter_components(n_components_list: list, n_features: int) -> list:
    """
    Filter and validate the n_components list against the available feature count.

    :param n_components_list: Raw list of requested component counts.
    :param n_features: Number of available numeric features in the dataset.
    :return: Filtered list of valid component counts.
    """

    try:
        n_components_list = [n for n in n_components_list if n > 0]  # Filter out non-positive component counts
        max_components = min(n_features, max(n_components_list)) if n_components_list else 0  # Maximum valid components
        n_components_list = [n for n in n_components_list if n <= max_components]  # Filter valid component counts

        return n_components_list  # Return the filtered list of valid component counts
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def print_pca_configuration(n_components_list: list) -> None:
    """
    Print the PCA analysis configuration summary to the terminal.

    :param n_components_list: Validated list of component counts to be tested.
    :return: None
    """

    try:
        print(f"\n{BackgroundColors.CYAN}PCA Configuration:{Style.RESET_ALL}")  # Print configuration section header
        print(
            f"  {BackgroundColors.GREEN}• Testing components: {BackgroundColors.CYAN}{n_components_list}{Style.RESET_ALL}"
        )  # Print the list of components to be tested
        cv_display = CROSS_N_FOLDS if CROSS_N_FOLDS is not None else cfg.get("cross_validation", {}).get("n_folds", 10)  # Determine CV folds display value
        print(
            f"  {BackgroundColors.GREEN}• Evaluation: {BackgroundColors.CYAN}{cv_display}-Fold Stratified Cross-Validation{Style.RESET_ALL}"
        )  # Print the CV method description
        print(f"  {BackgroundColors.GREEN}• Model: {BackgroundColors.CYAN}Random Forest (100 estimators){Style.RESET_ALL}")  # Print model description
        print(f"  {BackgroundColors.GREEN}• Split: {BackgroundColors.CYAN}80/20 (train/test){Style.RESET_ALL}\n")  # Print split description
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def execute_pca_parallel(X_train, y_train, X_test, y_test, n_components_list: list, scaler, max_workers, random_state: int) -> tuple:
    """
    Execute PCA analysis on all component configurations using a process pool.

    :param X_train: Scaled training feature array.
    :param y_train: Training target array or Series.
    :param X_test: Scaled test feature array.
    :param y_test: Test target array or Series.
    :param n_components_list: List of component counts to test in parallel.
    :param scaler: Fitted scaler with attached scaling time attribute.
    :param max_workers: Maximum number of parallel workers or None to use CPU count.
    :param random_state: Random seed for reproducibility.
    :return: Tuple of (all_results, executed_parallel) where all_results is a list and executed_parallel is a bool.
    """

    all_results = []  # Initialize results list for this execution attempt
    executed_parallel = False  # Initialize flag tracking successful parallel execution

    try:  # Attempt parallel execution
        cpu_count = os.cpu_count() or 1  # Get the number of CPU cores
        workers = max_workers or min(len(n_components_list), cpu_count)  # Compute initial worker count
        global CPU_PROCESSES
        if CPU_PROCESSES is not None:  # Verify if a CPU process limit was configured
            workers = min(workers, int(CPU_PROCESSES))  # Cap workers at configured process limit
        print(
            f"{BackgroundColors.GREEN}Running {BackgroundColors.CYAN}PCA Analysis{BackgroundColors.GREEN} in Parallel with {BackgroundColors.CYAN}{workers}{BackgroundColors.GREEN} Worker(s)...{Style.RESET_ALL}"
        )  # Log the parallel execution start

        results_map = {}  # Map to store results by n_components
        future_to_ncomp = {}  # Map each future to its n_components value

        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:  # Create a process pool executor
            for n_components in n_components_list:  # Loop over each number of components
                scaling_time_val = getattr(scaler, "_scaling_time", 0.0)  # Retrieve scaling_time attached to scaler
                fut = executor.submit(
                    apply_pca_and_evaluate,
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    n_components,
                    cv_folds=CROSS_N_FOLDS if CROSS_N_FOLDS is not None else cfg.get("cross_validation", {}).get("n_folds", 10),
                    workers=workers,
                    scaling_time=scaling_time_val,
                    random_state=random_state,
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
                    )  # Update progress bar description
                    try:  # Get the result from the future
                        res = fut.result()  # Get the result
                        results_map[res["n_components"]] = res  # Store result in the map
                        print_pca_results(res) if VERBOSE else None  # Print results if verbose
                    except Exception as e:  # Handle exceptions from worker processes
                        print(f"{BackgroundColors.RED}Error in worker: {e}{Style.RESET_ALL}")  # Log worker error
                    finally:  # Update the progress bar
                        pbar.update(1)  # Update progress bar
            finally:  # Ensure progress bar is closed
                pbar.close()  # Close the progress bar

        for n in n_components_list:  # Collect results in the original order
            if n in results_map:  # Verify if result exists for this n_components
                all_results.append(results_map[n])  # Append to all_results
        executed_parallel = True  # Mark parallel execution as successful
    except Exception as e:  # Handle exceptions during parallel execution
        print(
            f"{BackgroundColors.YELLOW}Parallel execution failed: {e}. Falling back to sequential execution.{Style.RESET_ALL}"
        )  # Log the fallback decision

    return all_results, executed_parallel  # Return results list and execution success flag


def execute_pca_sequential(X_train, y_train, X_test, y_test, n_components_list: list, scaler, random_state: int) -> list:
    """
    Execute PCA analysis on all component configurations sequentially.

    :param X_train: Scaled training feature array.
    :param y_train: Training target array or Series.
    :param X_test: Scaled test feature array.
    :param y_test: Test target array or Series.
    :param n_components_list: List of component counts to test sequentially.
    :param scaler: Fitted scaler with attached scaling time attribute.
    :param random_state: Random seed for reproducibility.
    :return: List of result dicts, one per component count.
    """

    all_results = []  # Initialize results list for sequential execution

    for n_components in tqdm(
        n_components_list, desc=f"{BackgroundColors.GREEN}PCA Analysis{Style.RESET_ALL}", unit="config"
    ):  # Iterate with progress bar over each component count
        send_telegram_message(TELEGRAM_BOT, f"Starting PCA training for n_components={n_components}")  # Notify training start
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Testing PCA with {BackgroundColors.CYAN}{n_components}{BackgroundColors.GREEN} components...{Style.RESET_ALL}"
        )  # Log current component count being tested
        comp_start = time.perf_counter()  # Start high-resolution timer for this component config
        scaling_time_val = getattr(scaler, "_scaling_time", 0.0)  # Retrieve scaling_time attached to scaler
        results = apply_pca_and_evaluate(
            X_train,
            y_train,
            X_test,
            y_test,
            n_components,
            cv_folds=CROSS_N_FOLDS if CROSS_N_FOLDS is not None else cfg.get("cross_validation", {}).get("n_folds", 10),
            workers=1,
            scaling_time=scaling_time_val,
            random_state=random_state,
        )  # Apply PCA and evaluate (single worker)
        comp_elapsed = time.perf_counter() - comp_start  # Compute elapsed duration
        send_telegram_message(TELEGRAM_BOT, f"Finished PCA training for n_components={n_components} with CV F1: {results['cv_f1_score']} in {calculate_execution_time(comp_start, time.perf_counter())}")  # Notify completion with metrics
        all_results.append(results)  # Append results to the list
        print_pca_results(results) if VERBOSE else None  # Print verbose results if configured

    return all_results  # Return the list of all sequential evaluation results


def print_best_pca_configuration(all_results: list) -> None:
    """
    Identify and print the best PCA configuration based on CV F1-Score.

    :param all_results: List of result dicts from all evaluated PCA configurations.
    :return: None
    """

    try:
        best_result = max(all_results, key=lambda x: x["cv_f1_score"])  # Find the best configuration based on CV F1-Score

        print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Best Configuration:{Style.RESET_ALL}")  # Print best config header
        print(
            f"  {BackgroundColors.GREEN}n_components = {BackgroundColors.CYAN}{best_result['n_components']}{Style.RESET_ALL}"
        )  # Print best n_components
        print(
            f"  {BackgroundColors.GREEN}CV F1-Score = {BackgroundColors.CYAN}{best_result['cv_f1_score']}{Style.RESET_ALL}"
        )  # Print best CV F1-Score
        print(
            f"  {BackgroundColors.GREEN}Explained Variance = {BackgroundColors.CYAN}{best_result['explained_variance']}{Style.RESET_ALL}"
        )  # Print best explained variance
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_pca_analysis(csv_path, n_components_list=[8, 16, 24, 32, 48], parallel=True, max_workers=None, random_state=42, scale_data=True, remove_zero_variance=True):
    """
    Runs PCA analysis with different numbers of components and evaluates performance.

    :param csv_path: Path to the CSV dataset file.
    :param n_components_list: List of component counts to test.
    :param parallel: Whether to use parallel execution when multiple configurations exist.
    :param max_workers: Maximum number of parallel workers or None to auto-detect.
    :param random_state: Random seed for reproducibility.
    :param scale_data: Whether to scale data before PCA.
    :param remove_zero_variance: Whether to drop zero-variance features during preprocessing.
    :return: None
    """

    try:
        global SKIP_TRAIN_IF_MODEL_EXISTS

        export_cfg_local = (globals().get('cfg') or {}).get("export", {}) if 'cfg' in globals() else None  # Retrieve export config from globals if available
        results_dir_raw_local = None  # Initialize results dir to None before resolution
        if export_cfg_local:  # Verify if export config was resolved from globals
            results_dir_raw_local = export_cfg_local.get("results_dir")  # Read results directory from export config
        if not results_dir_raw_local:  # Verify if results dir is still unresolved
            results_dir_raw_local = os.path.join("Feature_Analysis", "PCA")  # Fall back to default results directory
        if os.path.isabs(results_dir_raw_local):  # Verify if the path is absolute
            resolved_models_base = os.path.abspath(os.path.expanduser(results_dir_raw_local))  # Resolve absolute path
        else:  # Handle relative path
            dataset_dir_local = os.path.dirname(csv_path) or "."  # Use dataset directory as relative base
            resolved_models_base = os.path.abspath(os.path.expanduser(os.path.join(dataset_dir_local, results_dir_raw_local)))  # Resolve relative path
        models_dir = os.path.join(resolved_models_base, "Models")  # Build the models directory path
        base_name = Path(csv_path).stem  # Extract base filename stem from dataset path

        if SKIP_TRAIN_IF_MODEL_EXISTS:  # Verify if skip-training flag is enabled
            if should_skip_existing_models(n_components_list, models_dir, base_name):  # Verify if any exported model already exists
                return  # Skip all training when at least one model is found

        result = load_and_prepare_pca_dataset(csv_path, remove_zero_variance)  # Load and preprocess the dataset
        if result is None:  # Verify if dataset loading failed
            return {}  # Return empty dictionary on load failure

        X, y = result  # Unpack the feature matrix and target vector

        n_components_list = validate_and_filter_components(n_components_list, X.shape[1])  # Filter and validate component counts

        if not n_components_list:  # Verify if any valid component counts remain after filtering
            print(
                f"{BackgroundColors.RED}No valid component counts. Dataset has only {X.shape[1]} features.{Style.RESET_ALL}"
            )  # Log the empty component list error
            return  # Exit the function

        print_pca_configuration(n_components_list)  # Print PCA configuration summary to terminal

        X_train, X_test, y_train, y_test, scaler = scale_and_split(
            X, y, random_state=random_state, scale_data=scale_data
        )  # Scale and split the data

        all_results = []  # List to store all results
        executed_parallel = False  # Flag to track if parallel execution was successful

        if parallel and len(n_components_list) > 1:  # Verify if parallel execution is enabled and multiple configurations exist
            parallel_results, executed_parallel = execute_pca_parallel(
                X_train, y_train, X_test, y_test, n_components_list, scaler, max_workers, random_state
            )  # Run parallel execution
            all_results.extend(parallel_results)  # Collect parallel results

        if not executed_parallel:  # Verify if parallel was not executed or failed
            all_results = execute_pca_sequential(
                X_train, y_train, X_test, y_test, n_components_list, scaler, random_state
            )  # Fall back to sequential execution

        if not all_results:  # Verify if no results were collected
            print(
                f"{BackgroundColors.RED}No results collected from PCA analysis. Verify for errors in worker processes.{Style.RESET_ALL}"
            )  # Log the empty results warning
            return  # Return without saving

        save_pca_results(csv_path, all_results, cfg)  # Save all results to files (header from config)

        print_best_pca_configuration(all_results)  # Print the best configuration found
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


def main():
    """
    Main function.

    :return: None.
    """
    
    try:
        merged_cfg, sources = get_config()  # Load and merge configuration sources

        global cfg, app_cfg  # Declare global config variables for module-wide access
        app_cfg = merged_cfg  # Store full merged config in global
        cfg = (merged_cfg or {}).get("pca", {}) if isinstance(merged_cfg, dict) else {}  # Extract PCA-specific config block

        logs_dir = app_cfg.get("paths", {}).get("logs_dir", "./Logs") if isinstance(app_cfg, dict) else "./Logs"  # Resolve logs directory from config
        os.makedirs(logs_dir, exist_ok=True)  # Create logs directory if missing
        logger = Logger(f"{logs_dir}/{Path(__file__).stem}.log", clean=app_cfg.get("logging", {}).get("clean", True) if isinstance(app_cfg, dict) else True)  # Initialize Logger for file and stdout capture
        sys.stdout = logger  # Redirect stdout to logger
        sys.stderr = logger  # Redirect stderr to logger

        setup_global_exception_hook()  # Register global exception notifier via Telegram

        global N_JOBS, CROSS_N_FOLDS, CPU_PROCESSES, CACHING_ENABLED, PICKLE_PROTOCOL, VERBOSE, SKIP_TRAIN_IF_MODEL_EXISTS, CSV_FILE  # Declare all global runtime constants for config update
        CROSS_N_FOLDS = (cfg or {}).get("cross_validation", {}).get("n_folds")  # Read cross-validation fold count from config
        N_JOBS = (cfg or {}).get("multiprocessing", {}).get("n_jobs")  # Read parallel job count from config
        CPU_PROCESSES = (cfg or {}).get("multiprocessing", {}).get("cpu_processes")  # Read CPU process count from config
        CACHING_ENABLED = (cfg or {}).get("caching", {}).get("enabled")  # Read caching enabled flag from config
        PICKLE_PROTOCOL = (cfg or {}).get("caching", {}).get("pickle_protocol")  # Read pickle protocol version from config
        VERBOSE = (cfg or {}).get("execution", {}).get("verbose", False)  # Read verbose output flag from config
        SKIP_TRAIN_IF_MODEL_EXISTS = (cfg or {}).get("execution", {}).get("skip_train_if_model_exists", False)  # Read skip-training flag from config
        CSV_FILE = (cfg or {}).get("execution", {}).get("dataset_path")  # Read dataset path from config

        if CSV_FILE is None:  # Raise if dataset path missing from config and CLI
            raise ValueError("Dataset path must be provided (CLI --dataset_path or config.pca.execution.dataset_path)")
        if not Path(CSV_FILE).exists():  # Raise if dataset file not found at given path
            raise ValueError(f"Dataset file not found: {CSV_FILE}")

        n_components_list = (cfg or {}).get("dimensionality", {}).get("n_components_list")  # Read list of n_components values from config
        if not isinstance(n_components_list, list) or not all(isinstance(x, int) and x > 0 for x in n_components_list):  # Validate n_components_list before proceeding
            raise ValueError("pca.dimensionality.n_components_list must be a list of positive integers")

        random_state = (cfg or {}).get("model", {}).get("random_state")  # Read random state seed from config
        scale_data = (cfg or {}).get("preprocessing", {}).get("scale_data", True)  # Read scale_data flag from config
        remove_zero_variance = (cfg or {}).get("preprocessing", {}).get("remove_zero_variance", True)  # Read zero-variance removal flag from config
        max_workers = None  # Default to None for automatic thread pool sizing

        print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}PCA Feature Extraction{BackgroundColors.GREEN} program!{Style.RESET_ALL}")  # Clear terminal and print welcome message
        start_time = datetime.datetime.now()  # Record program start time

        setup_telegram_bot()  # Initialize Telegram bot for progress notifications

        send_telegram_message(TELEGRAM_BOT, [f"Starting PCA Feature Extraction on {CSV_FILE} at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"])  # Notify Telegram about program start

        run_pca_analysis(
            CSV_FILE,
            n_components_list,
            max_workers=max_workers,
            random_state=random_state,
            scale_data=scale_data,
            remove_zero_variance=remove_zero_variance,
        )  # Execute PCA feature extraction pipeline

        finish_time = datetime.datetime.now()  # Record program finish time
        print(
            f"{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
        )  # Output start time, finish time, and total execution time
        print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}")  # Output program completion message

        send_telegram_message(TELEGRAM_BOT, [f"PCA Feature Extraction completed on {CSV_FILE} at {finish_time.strftime('%Y-%m-%d %H:%M:%S')}.\nExecution time: {calculate_execution_time(start_time, finish_time)}"])  # Notify Telegram about program completion

        try:
            top_play_sound = app_cfg.get("execution", {}).get("play_sound", False)  # Read play_sound flag from top-level app config
            if top_play_sound:  # Register sound effect on exit if enabled
                atexit.register(play_sound)  # Register play_sound to run at program exit
        except Exception:
            pass
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


if __name__ == "__main__":
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """

    try:  # Protect main execution to ensure errors are reported and notified
        main()  # Call the main function
    except KeyboardInterrupt:  # User-initiated interrupt
        try:  # Attempt friendly shutdown notification on interrupt
            print("Execution interrupted by user (KeyboardInterrupt)")  # Inform terminal about user interrupt
            send_telegram_message(TELEGRAM_BOT, ["PCA execution interrupted by user (KeyboardInterrupt)"])  # Notify via Telegram
        except Exception:  # Ignore notification failures during interrupt handling
            pass  # Continue to cleanup even if notification fails
        try:  # Attempt to flush/close logger for clean logs on interrupt
            if "logger" in globals() and globals().get("logger") is not None:  # Verify logger existence before flushing to avoid attribute errors
                try:  # Try flushing and closing logger
                    globals()["logger"].flush()  # Flush logger buffer
                    globals()["logger"].close()  # Close logger handle
                except Exception:  # Ignore flush/close errors
                    pass  # Swallow to continue shutdown
        except Exception:  # Ignore unexpected errors during logger cleanup
            pass  # Swallow to continue shutdown
        raise  # Re-raise KeyboardInterrupt to allow upstream handling
    except BaseException as e:  # Catch everything (including SystemExit) and report
        try:  # Try to log and notify about the fatal error
            print(f"Fatal error: {e}")  # Print the exception message to terminal for logs
            tb_str = traceback.format_exc()  # Preserve full traceback string for local printing and inspection
            try:  # Send detailed exception via existing notifier (reuse implementation)
                send_exception_via_telegram(type(e), e, e.__traceback__)  # Use existing notifier to send full traceback
            except Exception:  # If telegram send fails, attempt to print the traceback as fallback
                try:  # Try printing fallback traceback
                    traceback.print_exc()  # Print full traceback to stderr as fallback
                except Exception:  # If even printing fails, swallow
                    pass  # Swallow to avoid masking original error
        except Exception:  # If notification preparation fails, attempt to print traceback
            try:  # Try printing traceback if preparation failed
                traceback.print_exc()  # Print traceback to stderr
            except Exception:  # If printing fails, swallow
                pass  # Swallow to avoid further errors
        try:  # Attempt to flush and close logger to preserve logs on fatal errors
            if "logger" in globals() and globals().get("logger") is not None:  # Verify logger exists before flushing to preserve logs
                try:  # Try to flush/close logger
                    globals()["logger"].flush()  # Flush logger buffer
                    globals()["logger"].close()  # Close logger handle
                except Exception:  # Ignore logger cleanup failures
                    pass  # Continue after best-effort cleanup
            else:  # Fallback if logger unavailable
                try:  # Try to flush/close sys.stdout if possible
                    if hasattr(sys.stdout, "flush"):  # Verify stdout supports flush before calling
                        try:
                            sys.stdout.flush()  # Attempt to flush stdout
                        except Exception:  # Ignore flush errors
                            pass  # Continue cleanup
                    if hasattr(sys.stdout, "close"):  # Verify stdout supports close before calling
                        try:
                            sys.stdout.close()  # Attempt to close stdout
                        except Exception:  # Ignore close errors
                            pass  # Continue cleanup
                except Exception:  # Ignore any errors while handling stdout
                    pass  # Continue after best-effort cleanup
        except Exception:  # Ignore any logger cleanup failures
            pass  # Continue to re-raise after best-effort cleanup
        raise  # Re-raise exception to preserve exit semantics
