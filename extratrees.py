"""
================================================================================
Extra Trees Feature Selection Tool (extratrees.py)
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-07-31
Description :
    Generates an Extra-Trees-20 feature-selection representation for the DDoS
    detection pipeline. The selector fits only on the training partition,
    ranks all eligible numeric predictor columns by Extra Trees importance, and
    persists the complete ranked list for downstream stacking.py execution.
"""
if __name__ in {"__main__", "__mp_main__"}:
    try:
        from setproctitle import setproctitle
        setproctitle(f"DDoS-{__file__.rsplit('/', 1)[-1].rsplit('.', 1)[0]}")
    except ImportError:
        pass


import argparse  # Parse command-line overrides
import datetime  # Build timestamp metadata
import json  # Serialize configuration metadata
import math  # Validate finite numeric values
import os  # Resolve filesystem paths
from pathlib import Path  # Normalize paths
import platform  # Record hardware platform metadata
import re  # Sanitize feature names consistently
import sys  # Provide process exit behavior
import time  # Measure feature extraction duration
from typing import Any, Optional  # Provide type hints

import numpy as np  # Handle numeric arrays
import pandas as pd  # Load datasets and persist CSV results
import psutil  # Record hardware memory metadata
import yaml  # Load YAML configuration
from colorama import Style  # Match project terminal color reset style
from sklearn.ensemble import ExtraTreesClassifier  # Fit Extra Trees feature importances
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score  # Compute selector diagnostics
from sklearn.model_selection import StratifiedKFold, train_test_split  # Split before selector fitting and optional CV diagnostics
from tqdm import tqdm  # Show progress bars with ETA for iterable stages
from Logger import Logger, SAO_PAULO_TIMEZONE_NAME  # Reuse project logger for terminal and file output
from utils.process_name import set_runtime_process_name  # Apply optional htop-visible process identities.

try:  # Import optional Telegram utilities
    import telegram_bot as telegram_module  # Configure Telegram message prefixes consistently with stacking.py
    from telegram_bot import TelegramBot, send_exception_via_telegram, send_telegram_message  # Reuse existing Telegram delivery path
except Exception:  # Keep Extra Trees usable when Telegram dependencies are unavailable
    telegram_module = None  # Disable Telegram module integration
    TelegramBot = None  # Disable Telegram bot construction
    send_exception_via_telegram = None  # Disable Telegram exception forwarding
    send_telegram_message = None  # Disable Telegram message delivery


NON_FEATURE_COLUMNS = ("Unnamed: 0", "Flow ID", "Source IP", "Destination IP", "Timestamp")  # Define leakage-prone metadata columns
TARGET_COLUMN_ALIASES = ("Label", "label", "attack_type", "Attack Type", "Class", "class", "Target", "target")  # Define common target column names
DEFAULT_RESULTS_CSV_COLUMNS = ("timestamp", "tool", "run_index", "model", "dataset", "dataset_path", "hyperparameters", "cv_method", "train_test_split", "scaling", "cv_accuracy", "cv_precision", "cv_recall", "cv_f1_score", "cv_fpr", "cv_fnr", "test_accuracy", "test_precision", "test_recall", "test_f1_score", "test_fpr", "test_fnr", "feature_extraction_time_s", "training_time_s", "testing_time_s", "elapsed_run_time", "hardware", "best_features", "union_features_across_runs", "rfe_ranking", "feature_name", "original_feature_index", "extra_trees_importance", "importance_rank", "selected", "configured_selected_feature_count", "actual_selected_feature_count", "n_estimators", "random_state", "n_jobs", "n_train", "n_test", "source_feature_count", "eligible_feature_count", "target_column", "excluded_columns")  # Define default configurable export header
TELEGRAM_BOT = None  # Store optional Telegram bot instance for script-level notifications
logger = None  # Store optional script logger instance


class BackgroundColors:  # Match project color constants
    CYAN = "\033[96m"  # Cyan terminal color
    GREEN = "\033[92m"  # Green terminal color
    YELLOW = "\033[93m"  # Yellow terminal color
    RED = "\033[91m"  # Red terminal color
    BOLD = "\033[1m"  # Bold terminal style
    CLEAR_TERMINAL = "\033[H\033[J"  # Clear-terminal escape sequence


def get_default_config() -> dict:
    """
    Return default Extra Trees configuration.

    :return: Default configuration dictionary.
    """

    return {  # Return internal defaults before YAML and CLI overrides
        "execution": {"verbose": False, "dataset_path": "./Datasets/CICDDoS2019/01-12/DrDoS_DNS.csv", "low_memory": False, "combined_files": False},  # Define execution defaults
        "dataset": {"test_size": 0.2, "random_state": 42},  # Define split defaults
        "extra_trees": {  # Define selector defaults
            "selection": {"n_features_to_select": 20},  # Define Extra-Trees-20 default representation
            "cross_validation": {"enabled": True, "n_folds": 10},  # Define training-partition CV diagnostics
            "model": {"n_estimators": 200, "random_state": 42, "n_jobs": 1, "criterion": "gini", "max_features": "sqrt"},  # Define Extra Trees classifier defaults
            "export": {"results_dir": "Feature_Analysis/Extra_Trees", "results_filename": "Extra_Trees_Results.csv", "results_csv_columns": list(DEFAULT_RESULTS_CSV_COLUMNS)},  # Define export path and header defaults
        },
    }  # Complete default config


def load_config_file(config_path: Optional[str]) -> dict:
    """
    Load YAML configuration from disk.

    :param config_path: Optional YAML path.
    :return: Loaded configuration dictionary.
    """

    if not config_path:  # Treat absent config as empty override
        return {}  # Return empty config
    path = Path(config_path).expanduser()  # Normalize config path
    if not path.exists():  # Validate config file presence
        raise FileNotFoundError(f"Config file not found: {config_path}")  # Raise explicit missing-config error
    with path.open("r", encoding="utf-8") as handle:  # Open YAML config with stable encoding
        loaded = yaml.safe_load(handle) or {}  # Load YAML mapping or empty fallback
    if not isinstance(loaded, dict):  # Validate YAML root shape
        raise ValueError(f"Config file must contain a mapping: {config_path}")  # Raise explicit schema error
    return loaded  # Return loaded config


def deep_merge_dicts(base: dict, override: dict) -> dict:
    """
    Merge nested dictionaries without mutating inputs.

    :param base: Base configuration dictionary.
    :param override: Override configuration dictionary.
    :return: Merged configuration dictionary.
    """

    merged = dict(base)  # Copy base level
    for key, value in (override or {}).items():  # Iterate override keys
        if isinstance(value, dict) and isinstance(merged.get(key), dict):  # Merge nested mappings recursively
            merged[key] = deep_merge_dicts(merged[key], value)  # Store merged nested mapping
        else:  # Replace scalar or mismatched values
            merged[key] = value  # Store override value
    return merged  # Return merged mapping


def parse_cli_args() -> argparse.Namespace:
    """
    Parse Extra Trees command-line arguments.

    :return: Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description="Extra Trees feature selection for DDoS-Detector")  # Build CLI parser
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")  # Add config override
    parser.add_argument("--process-name", type=str, default=None, help="Process title displayed by htop and similar tools")  # Allow concurrent runs to have distinct operating-system identities.
    parser.add_argument("--dataset-path", "--dataset_path", dest="dataset_path", type=str, default=None, help="Path to dataset CSV")  # Add dataset override
    parser.add_argument("--combined-files", dest="combined_files", action="store_true", default=False, help="Treat dataset path as a directory of CSV files and combine them in memory")  # Add in-memory combined-files mode
    parser.add_argument("--n-features-to-select", dest="n_features_to_select", type=int, default=None, help="Number of Extra Trees features to select")  # Add selected-count override
    parser.add_argument("--n-estimators", dest="n_estimators", type=int, default=None, help="Number of Extra Trees estimators")  # Add estimator-count override
    parser.add_argument("--random-state", dest="random_state", type=int, default=None, help="Random seed")  # Add seed override
    parser.add_argument("--n-jobs", dest="n_jobs", type=int, default=None, help="Extra Trees worker threads")  # Add worker override
    parser.add_argument("--cv-folds", dest="cv_folds", type=int, default=None, help="Training-partition StratifiedKFold count for selector diagnostics")  # Add CV-fold override
    parser.add_argument("--disable-cv", dest="disable_cv", action="store_true", default=False, help="Disable training-partition CV diagnostics")  # Add CV-disable override
    parser.add_argument("--results-dir", dest="results_dir", type=str, default=None, help="Results directory relative to dataset directory")  # Add export-directory override
    parser.add_argument("--results-filename", dest="results_filename", type=str, default=None, help="Results CSV filename")  # Add export-filename override
    parser.add_argument("--results-csv-columns", dest="results_csv_columns", type=str, default=None, help="Comma-separated Extra Trees result CSV columns")  # Add configurable header override
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging")  # Add verbose override
    cli_args = parser.parse_args()  # Parse the Extra Trees arguments.
    set_runtime_process_name(cli_args.process_name, script_path=__file__)  # Apply the generated htop identity before configuration and logging initialization.
    return cli_args  # Return parsed CLI arguments.


def build_cli_overrides(cli_args: argparse.Namespace) -> dict:
    """
    Build configuration overrides from explicitly supplied CLI values.

    :param cli_args: Parsed CLI namespace.
    :return: CLI override configuration dictionary.
    """

    overrides: dict[str, Any] = {}  # Accumulate CLI overrides only when supplied
    if cli_args.verbose:  # Apply boolean verbose only when explicitly true
        overrides.setdefault("execution", {})["verbose"] = True  # Store verbose override
    if cli_args.dataset_path is not None:  # Apply dataset path only when supplied
        overrides.setdefault("execution", {})["dataset_path"] = cli_args.dataset_path  # Store dataset path override
    if cli_args.combined_files:  # Apply combined-files mode only when explicitly requested
        overrides.setdefault("execution", {})["combined_files"] = True  # Store in-memory directory-combine override
    if cli_args.n_features_to_select is not None:  # Apply selected-count override only when supplied
        overrides.setdefault("extra_trees", {}).setdefault("selection", {})["n_features_to_select"] = cli_args.n_features_to_select  # Store selected-count override
    if cli_args.n_estimators is not None:  # Apply estimator-count override only when supplied
        overrides.setdefault("extra_trees", {}).setdefault("model", {})["n_estimators"] = cli_args.n_estimators  # Store estimator-count override
    if cli_args.random_state is not None:  # Apply seed override only when supplied
        overrides.setdefault("extra_trees", {}).setdefault("model", {})["random_state"] = cli_args.random_state  # Store model seed override
        overrides.setdefault("dataset", {})["random_state"] = cli_args.random_state  # Store split seed override
    if cli_args.n_jobs is not None:  # Apply worker override only when supplied
        overrides.setdefault("extra_trees", {}).setdefault("model", {})["n_jobs"] = cli_args.n_jobs  # Store worker override
    if cli_args.cv_folds is not None:  # Apply CV-fold override only when supplied
        overrides.setdefault("extra_trees", {}).setdefault("cross_validation", {})["n_folds"] = cli_args.cv_folds  # Store CV-fold override
    if cli_args.disable_cv:  # Apply CV-disable override only when supplied
        overrides.setdefault("extra_trees", {}).setdefault("cross_validation", {})["enabled"] = False  # Store CV-disable override
    if cli_args.results_dir is not None:  # Apply results directory override only when supplied
        overrides.setdefault("extra_trees", {}).setdefault("export", {})["results_dir"] = cli_args.results_dir  # Store results directory override
    if cli_args.results_filename is not None:  # Apply results filename override only when supplied
        overrides.setdefault("extra_trees", {}).setdefault("export", {})["results_filename"] = cli_args.results_filename  # Store results filename override
    if cli_args.results_csv_columns is not None:  # Apply CSV header override only when supplied
        overrides.setdefault("extra_trees", {}).setdefault("export", {})["results_csv_columns"] = [column.strip() for column in cli_args.results_csv_columns.split(",") if column.strip()]  # Store parsed CSV header override
    return overrides  # Return CLI-only overrides


def resolve_config_file_path(cli_config_arg: Optional[str]) -> Optional[str]:
    """
    Resolve configuration file path.

    :param cli_config_arg: CLI config path or None.
    :return: Resolved config path or None.
    """

    if cli_config_arg:  # Prefer explicit CLI config path
        return cli_config_arg  # Return supplied config path
    if Path("config.yaml").exists():  # Prefer runtime config in working directory
        return "config.yaml"  # Return runtime config path
    if Path("config.yaml.example").exists():  # Fall back to example config for local validation
        return "config.yaml.example"  # Return example config path
    return None  # Return absent config path


def get_config(cli_args: argparse.Namespace) -> dict:
    """
    Resolve effective Extra Trees configuration.

    :param cli_args: Parsed CLI namespace.
    :return: Effective configuration dictionary.
    """

    defaults = get_default_config()  # Load internal defaults
    config_path = resolve_config_file_path(cli_args.config)  # Resolve YAML path
    file_config = load_config_file(config_path) if config_path else {}  # Load YAML overrides when available
    merged = deep_merge_dicts(defaults, file_config)  # Apply YAML over defaults
    scoped_extra_trees_config = merged.get("extra_trees", {})  # Read Extra Trees scoped YAML settings
    if isinstance(scoped_extra_trees_config.get("execution"), dict):  # Apply Extra Trees scoped execution settings
        merged["execution"] = deep_merge_dicts(merged.get("execution", {}), scoped_extra_trees_config.get("execution", {}))  # Merge scoped execution settings into effective script execution
    if isinstance(scoped_extra_trees_config.get("dataset"), dict):  # Apply Extra Trees scoped dataset settings
        merged["dataset"] = deep_merge_dicts(merged.get("dataset", {}), scoped_extra_trees_config.get("dataset", {}))  # Merge scoped dataset settings into effective script split configuration
    merged = deep_merge_dicts(merged, build_cli_overrides(cli_args))  # Apply explicit CLI over YAML
    validate_config(merged)  # Validate resolved settings
    return merged  # Return effective config


def validate_positive_int(value: Any, source: str) -> int:
    """
    Validate a positive integer configuration value.

    :param value: Raw configuration value.
    :param source: User-facing configuration source name.
    :return: Validated integer value.
    """

    if isinstance(value, bool) or not isinstance(value, int):  # Reject booleans and non-integers
        raise ValueError(f"{source} must be an integer greater than or equal to 1")  # Raise explicit integer error
    if value < 1:  # Reject zero and negative values
        raise ValueError(f"{source} must be an integer greater than or equal to 1")  # Raise explicit range error
    return int(value)  # Return normalized integer


def validate_n_jobs(value: Any, source: str) -> int:
    """
    Validate an Extra Trees worker count.

    :param value: Raw worker count.
    :param source: User-facing configuration source name.
    :return: Validated worker count.
    """

    if isinstance(value, bool) or not isinstance(value, int):  # Reject booleans and non-integers
        raise ValueError(f"{source} must be a non-zero integer")  # Raise explicit integer error
    if value == 0:  # Reject unsupported zero workers
        raise ValueError(f"{source} must be a non-zero integer")  # Raise explicit zero error
    return int(value)  # Return normalized worker count


def validate_config(config: dict) -> None:
    """
    Validate effective Extra Trees configuration.

    :param config: Effective configuration dictionary.
    :return: None.
    """

    selection_cfg = config.get("extra_trees", {}).get("selection", {})  # Read selection configuration
    model_cfg = config.get("extra_trees", {}).get("model", {})  # Read model configuration
    cv_cfg = config.get("extra_trees", {}).get("cross_validation", {})  # Read cross-validation configuration
    export_cfg = config.get("extra_trees", {}).get("export", {})  # Read export configuration
    dataset_cfg = config.get("dataset", {})  # Read dataset configuration
    execution_cfg = config.get("execution", {})  # Read execution configuration
    validate_positive_int(selection_cfg.get("n_features_to_select", 20), "extra_trees.selection.n_features_to_select")  # Validate selected-feature count
    validate_positive_int(model_cfg.get("n_estimators", 200), "extra_trees.model.n_estimators")  # Validate estimator count
    validate_n_jobs(model_cfg.get("n_jobs", 1), "extra_trees.model.n_jobs")  # Validate worker count
    validate_positive_int(cv_cfg.get("n_folds", 3), "extra_trees.cross_validation.n_folds")  # Validate CV-fold count
    if not isinstance(execution_cfg.get("combined_files", False), bool):  # Validate combined-files mode type
        raise ValueError("execution.combined_files must be true or false")  # Raise explicit combined-files mode error
    if not isinstance(cv_cfg.get("enabled", True), bool):  # Validate CV enablement type
        raise ValueError("extra_trees.cross_validation.enabled must be true or false")  # Raise explicit CV enablement error
    if not isinstance(dataset_cfg.get("random_state", 42), int) or isinstance(dataset_cfg.get("random_state", 42), bool):  # Validate split seed type
        raise ValueError("dataset.random_state must be an integer")  # Raise explicit split-seed error
    if not isinstance(model_cfg.get("random_state", 42), int) or isinstance(model_cfg.get("random_state", 42), bool):  # Validate model seed type
        raise ValueError("extra_trees.model.random_state must be an integer")  # Raise explicit model-seed error
    test_size = float(dataset_cfg.get("test_size", 0.2))  # Normalize split fraction
    if not math.isfinite(test_size) or test_size <= 0.0 or test_size >= 1.0:  # Validate split fraction range
        raise ValueError("dataset.test_size must be greater than 0 and less than 1")  # Raise explicit split range error
    results_filename = export_cfg.get("results_filename", "Extra_Trees_Results.csv")  # Read output filename
    if not isinstance(results_filename, str) or not results_filename.lower().endswith(".csv"):  # Validate CSV filename
        raise ValueError("extra_trees.export.results_filename must end with .csv")  # Raise explicit filename error
    results_columns = export_cfg.get("results_csv_columns", list(DEFAULT_RESULTS_CSV_COLUMNS))  # Read configured result columns
    if not isinstance(results_columns, list) or not results_columns or any(not isinstance(column, str) or not column.strip() for column in results_columns):  # Validate result header shape
        raise ValueError("extra_trees.export.results_csv_columns must be a non-empty list of column names")  # Raise explicit header error
    required_loader_columns = {"feature_name", "extra_trees_importance", "importance_rank", "selected"}  # Define stacking loader-required columns
    missing_loader_columns = required_loader_columns - {column.strip() for column in results_columns}  # Identify loader-required columns absent from configured header
    if missing_loader_columns:  # Reject headers that stacking.py cannot consume
        raise ValueError(f"extra_trees.export.results_csv_columns must include {sorted(missing_loader_columns)}")  # Raise explicit unusable-header error


def sanitize_feature_names(columns: Any) -> list[str]:
    r"""
    Sanitize feature names using the stacking.py column convention.

    :param columns: Raw column names.
    :return: Sanitized feature names.
    """

    sanitized = []  # Accumulate sanitized names
    for column in columns:  # Iterate raw names
        clean_column = re.sub(r"[{}\[\]:,\"\\]", "_", str(column))  # Replace LightGBM-sensitive characters
        clean_column = re.sub(r"_+", "_", clean_column)  # Collapse repeated underscores
        clean_column = clean_column.strip("_")  # Remove edge underscores
        sanitized.append(clean_column)  # Store sanitized name
    return sanitized  # Return sanitized names


def preprocess_dataframe(dataframe: pd.DataFrame, remove_zero_variance: bool = True) -> pd.DataFrame:
    """
    Preprocess a DataFrame for Extra Trees feature selection.

    :param dataframe: Input dataset DataFrame.
    :param remove_zero_variance: Whether zero-variance numeric columns are removed.
    :return: Cleaned dataset DataFrame.
    """

    original_rows = int(dataframe.shape[0])  # Record source row count before sanitation
    original_columns = int(dataframe.shape[1])  # Record source column count before sanitation
    print(f"{BackgroundColors.GREEN}[PREPROCESS] Starting sanitation for {BackgroundColors.CYAN}{original_rows}{BackgroundColors.GREEN} rows and {BackgroundColors.CYAN}{original_columns}{BackgroundColors.GREEN} columns.{Style.RESET_ALL}")  # Log preprocessing start
    dataframe.columns = sanitize_feature_names([str(column).strip() for column in dataframe.columns])  # Apply stacking-compatible column sanitization
    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN in-place
    numeric_columns = dataframe.select_dtypes(include=["number"]).columns  # Resolve numeric columns for finite-range sanitation
    if len(numeric_columns) > 0:  # Sanitize numeric predictors when present
        float32_limit = np.finfo(np.float32).max  # Match sklearn tree validation limit
        oversized_rows = pd.Series(False, index=dataframe.index)  # Track rows with values too large for sklearn trees
        for column in tqdm(numeric_columns, desc=f"{BackgroundColors.GREEN}Extra Trees preprocessing{Style.RESET_ALL}", unit="column", colour="green"):  # Scan numeric columns with colored ETA
            oversized_rows |= dataframe[column].abs().gt(float32_limit)  # Mark rows exceeding the float32 tree limit
        if oversized_rows.any():  # Remove rows that sklearn trees cannot fit
            print(f"{BackgroundColors.YELLOW}[PREPROCESS] Removing {BackgroundColors.CYAN}{int(oversized_rows.sum())}{BackgroundColors.YELLOW} row(s) exceeding sklearn tree float32 limits.{Style.RESET_ALL}")  # Log oversized-row removal
            dataframe.drop(index=dataframe.index[oversized_rows], inplace=True)  # Drop oversized rows in-place
    rows_after_range = int(dataframe.shape[0])  # Record row count after range sanitation
    dataframe.dropna(inplace=True)  # Drop rows containing NaN values after infinity sanitation
    rows_after_na = int(dataframe.shape[0])  # Record row count after NaN removal
    if remove_zero_variance and len(numeric_columns) > 0:  # Remove constant numeric columns when configured
        numeric_columns = dataframe.select_dtypes(include=["number"]).columns  # Refresh numeric columns after row cleanup
        variances = dataframe[numeric_columns].var(axis=0, ddof=0)  # Calculate numeric feature variances
        zero_variance_columns = variances[variances == 0].index.tolist()  # Resolve constant numeric columns
        if zero_variance_columns:  # Remove constant columns only when found
            print(f"{BackgroundColors.YELLOW}[PREPROCESS] Removing {BackgroundColors.CYAN}{len(zero_variance_columns)}{BackgroundColors.YELLOW} zero-variance numeric column(s).{Style.RESET_ALL}")  # Log zero-variance removal
            dataframe.drop(columns=zero_variance_columns, inplace=True)  # Drop zero-variance columns in-place
    if dataframe.empty:  # Reject datasets emptied by sanitation
        raise ValueError("Extra Trees preprocessing removed all rows; dataset contains no valid finite samples")  # Raise explicit empty-data error
    print(f"{BackgroundColors.GREEN}[PREPROCESS] Finished sanitation: {BackgroundColors.CYAN}{int(dataframe.shape[0])}{BackgroundColors.GREEN} rows, {BackgroundColors.CYAN}{int(dataframe.shape[1])}{BackgroundColors.GREEN} columns retained. Dropped rows: range={BackgroundColors.CYAN}{original_rows - rows_after_range}{BackgroundColors.GREEN}, nan={BackgroundColors.CYAN}{rows_after_range - rows_after_na}{Style.RESET_ALL}")  # Log preprocessing result
    return dataframe  # Return cleaned dataframe


def normalize_feature_name(name: Any) -> str:
    """
    Normalize feature names for metadata exclusion.

    :param name: Raw feature name.
    :return: Normalized feature name.
    """

    text = str(name)  # Convert feature name to string
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\u00a0\u2060\u2028\u2029]", "", text)  # Remove invisible characters
    text = re.sub(r" +", " ", text.strip())  # Normalize visible whitespace
    return text.lower()  # Return lowercase comparison key


def resolve_target_column(columns: list[str]) -> str:
    """
    Resolve target column from dataset columns.

    :param columns: Dataset column names.
    :return: Target column name.
    """

    for alias in TARGET_COLUMN_ALIASES:  # Prefer known label aliases
        if alias in columns:  # Match an existing target alias
            return alias  # Return detected target column
    return columns[-1]  # Fall back to last-column convention


def resolve_output_paths(config: dict, csv_path: str) -> tuple[Path, Path]:
    """
    Resolve Extra Trees output directory and CSV path.

    :param config: Effective configuration dictionary.
    :param csv_path: Dataset CSV path.
    :return: Output directory and CSV path.
    """

    export_cfg = config.get("extra_trees", {}).get("export", {})  # Read export configuration
    results_dir_raw = export_cfg.get("results_dir", "Feature_Analysis/Extra_Trees")  # Read configured output directory
    results_filename = export_cfg.get("results_filename", "Extra_Trees_Results.csv")  # Read configured output filename
    dataset_path = Path(csv_path).expanduser().resolve()  # Resolve dataset path
    dataset_dir = dataset_path if dataset_path.is_dir() else dataset_path.parent  # Resolve dataset directory for file or combined-directory input
    output_dir = Path(results_dir_raw).expanduser()  # Normalize configured output directory
    if not output_dir.is_absolute():  # Resolve relative exports beside dataset
        output_dir = dataset_dir / output_dir  # Build dataset-relative export directory
    return output_dir.resolve(), (output_dir / results_filename).resolve()  # Return resolved paths


def list_combined_dataset_files(dataset_dir: Path) -> list[Path]:
    """
    List CSV files for in-memory combined-files Extra Trees mode.

    :param dataset_dir: Dataset directory.
    :return: Ordered CSV file paths.
    """

    files = sorted(path for path in dataset_dir.glob("*.csv") if path.is_file())  # Resolve deterministic top-level CSV inputs
    if not files:  # Reject empty directories
        raise FileNotFoundError(f"No CSV files found in dataset directory: {dataset_dir}")  # Raise explicit combined-files error
    return files  # Return ordered CSV files


def resolve_common_columns(csv_files: list[Path], low_memory: bool) -> list[str]:
    """
    Resolve common CSV columns in first-file order.

    :param csv_files: Ordered CSV files.
    :param low_memory: Pandas low-memory mode.
    :return: Common column names.
    """

    headers = [(path, list(pd.read_csv(path, nrows=0, low_memory=low_memory).columns)) for path in csv_files]  # Read headers without loading rows
    common_columns = list(headers[0][1])  # Preserve first-file column order
    for _, columns in headers[1:]:  # Intersect remaining headers
        column_set = set(columns)  # Build lookup for this file
        common_columns = [column for column in common_columns if column in column_set]  # Keep only shared columns
    if not common_columns:  # Reject incompatible CSV files
        raise ValueError(f"No common columns across {len(csv_files)} CSV files in combined-files mode")  # Raise explicit schema error
    return common_columns  # Return shared columns


def load_combined_dataset(dataset_dir: Path, config: dict) -> pd.DataFrame:
    """
    Load a directory of CSV files into one in-memory DataFrame.

    :param dataset_dir: Dataset directory.
    :param config: Effective configuration dictionary.
    :return: Combined DataFrame.
    """

    low_memory = bool(config.get("execution", {}).get("low_memory", False))  # Resolve pandas low-memory mode
    csv_files = list_combined_dataset_files(dataset_dir)  # Resolve source CSV files
    common_columns = resolve_common_columns(csv_files, low_memory)  # Align files to common schema
    print(f"{BackgroundColors.GREEN}[LOAD] Combining {BackgroundColors.CYAN}{len(csv_files)}{BackgroundColors.GREEN} CSV file(s) from {BackgroundColors.CYAN}{dataset_dir}{BackgroundColors.GREEN} in memory.{Style.RESET_ALL}")  # Log combined-files load start
    print(f"{BackgroundColors.GREEN}[LOAD] Common columns retained: {BackgroundColors.CYAN}{len(common_columns)}{Style.RESET_ALL}")  # Log common schema width
    frames = []  # Accumulate in-memory source frames
    for csv_file in tqdm(csv_files, desc=f"{BackgroundColors.GREEN}Extra Trees combined CSV load{Style.RESET_ALL}", unit="file", colour="green"):  # Load each source file with progress
        frame = pd.read_csv(csv_file, usecols=common_columns, low_memory=low_memory)  # Load only aligned common columns
        frames.append(frame)  # Store frame for in-memory concatenation
    dataframe = pd.concat(frames, ignore_index=True, copy=False)  # Combine every source frame without writing an intermediate CSV
    print(f"{BackgroundColors.GREEN}[LOAD] Loaded combined raw dataset shape: {BackgroundColors.CYAN}{dataframe.shape[0]} rows x {dataframe.shape[1]} columns{Style.RESET_ALL}")  # Log combined shape
    return dataframe  # Return combined frame


def load_dataset(csv_path: str, config: dict) -> pd.DataFrame:
    """
    Load and sanitize the dataset.

    :param csv_path: Dataset CSV path.
    :param config: Effective configuration dictionary.
    :return: Loaded DataFrame.
    """

    path = Path(csv_path).expanduser()  # Normalize dataset path
    if not path.exists():  # Validate dataset presence
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")  # Raise explicit dataset error
    low_memory = bool(config.get("execution", {}).get("low_memory", False))  # Resolve pandas low-memory mode
    if bool(config.get("execution", {}).get("combined_files", False)):  # Use directory-based in-memory combined mode when requested
        if not path.is_dir():  # Require a directory for combined-files mode
            raise ValueError(f"Combined-files mode requires --dataset-path to be a directory: {csv_path}")  # Raise explicit mode/path mismatch
        dataframe = load_combined_dataset(path, config)  # Load and concatenate source CSV files in memory
    else:  # Use legacy single-CSV mode
        if path.is_dir():  # Reject accidental directory input in single-file mode
            raise ValueError(f"Dataset path is a directory; pass --combined-files to combine CSV files in memory: {csv_path}")  # Raise explicit mode hint
        print(f"{BackgroundColors.GREEN}[LOAD] Reading dataset from {BackgroundColors.CYAN}{path}{Style.RESET_ALL}")  # Log dataset load start
        dataframe = pd.read_csv(path, low_memory=low_memory)  # Load dataset CSV
        print(f"{BackgroundColors.GREEN}[LOAD] Loaded raw dataset shape: {BackgroundColors.CYAN}{dataframe.shape[0]} rows x {dataframe.shape[1]} columns{Style.RESET_ALL}")  # Log raw dataset shape
    dataframe = preprocess_dataframe(dataframe, remove_zero_variance=bool(config.get("dataset", {}).get("remove_zero_variance", True)))  # Apply GA-aligned dataframe sanitation
    if dataframe.shape[1] < 2:  # Validate predictor plus target columns
        raise ValueError("Dataset must contain at least one feature column and one target column")  # Raise explicit dataset shape error
    print(f"{BackgroundColors.GREEN}[LOAD] Dataset ready for Extra Trees: {BackgroundColors.CYAN}{dataframe.shape[0]} rows x {dataframe.shape[1]} columns{Style.RESET_ALL}")  # Log cleaned dataset shape
    return dataframe  # Return loaded dataframe


def prepare_training_data(dataframe: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, list[str], dict]:
    """
    Prepare training-only data for Extra Trees ranking.

    :param dataframe: Loaded dataset DataFrame.
    :param config: Effective configuration dictionary.
    :return: Training predictors, training labels, test predictors, test labels, feature names, and split metadata.
    """

    print(f"{BackgroundColors.GREEN}[SPLIT] Preparing train/test data and eligible feature list.{Style.RESET_ALL}")  # Log split preparation start
    target_column = resolve_target_column(list(dataframe.columns))  # Resolve target column
    excluded = {normalize_feature_name(column) for column in NON_FEATURE_COLUMNS}  # Build leakage-prone metadata exclusion set
    excluded_present = [column for column in dataframe.columns if column != target_column and normalize_feature_name(column) in excluded]  # Record leakage-prone columns present in source data
    feature_columns = [column for column in dataframe.columns if column != target_column and normalize_feature_name(column) not in excluded]  # Select eligible predictor names
    numeric_columns = [column for column in feature_columns if pd.api.types.is_numeric_dtype(dataframe[column])]  # Keep numeric predictors only
    requested_count = int(config.get("extra_trees", {}).get("selection", {}).get("n_features_to_select", 20))  # Resolve requested selection count
    print(f"{BackgroundColors.GREEN}[SPLIT] Target column: {BackgroundColors.CYAN}{target_column}{BackgroundColors.GREEN} | Numeric eligible features: {BackgroundColors.CYAN}{len(numeric_columns)}{BackgroundColors.GREEN} | Requested selected features: {BackgroundColors.CYAN}{requested_count}{Style.RESET_ALL}")  # Log target and feature counts
    if len(numeric_columns) < requested_count:  # Validate enough eligible features exist
        raise ValueError(f"Extra Trees requested {requested_count} features but only {len(numeric_columns)} eligible numeric features are available")  # Raise explicit feature-count error
    y = dataframe[target_column].to_numpy(copy=False)  # Read labels without renaming or grouping
    if np.unique(y).shape[0] < 2:  # Validate classification target
        raise ValueError("Extra Trees feature selection requires at least two target classes")  # Raise explicit class-count error
    split_random_state = int(config.get("dataset", {}).get("random_state", 42))  # Resolve split seed
    test_size = float(config.get("dataset", {}).get("test_size", 0.2))  # Resolve held-out fraction
    row_indices = np.arange(dataframe.shape[0], dtype=np.int64)  # Build row index vector for split
    train_indices, test_indices = train_test_split(row_indices, test_size=test_size, random_state=split_random_state, stratify=y)  # Split before fitting selector
    X_train = dataframe.loc[dataframe.index[train_indices], numeric_columns]  # Select training predictors only
    y_train = y[train_indices]  # Select training labels only
    X_test = dataframe.loc[dataframe.index[test_indices], numeric_columns]  # Select held-out predictors only for selector diagnostics
    y_test = y[test_indices]  # Select held-out labels only for selector diagnostics
    metadata = {"target_column": target_column, "n_train": int(len(train_indices)), "n_test": int(len(test_indices)), "source_feature_count": int(dataframe.shape[1] - 1), "eligible_feature_count": int(len(numeric_columns)), "excluded_columns": excluded_present, "test_size": float(test_size)}  # Record split and eligibility metadata
    print(f"{BackgroundColors.GREEN}[SPLIT] Train rows: {BackgroundColors.CYAN}{len(train_indices)}{BackgroundColors.GREEN} | Test rows: {BackgroundColors.CYAN}{len(test_indices)}{BackgroundColors.GREEN} | Test size: {BackgroundColors.CYAN}{test_size}{Style.RESET_ALL}")  # Log split result
    return X_train, y_train, X_test, y_test, numeric_columns, metadata  # Return selector inputs and held-out diagnostics data


def build_extra_trees_selector(config: dict) -> ExtraTreesClassifier:
    """
    Build the configured Extra Trees classifier.

    :param config: Effective configuration dictionary.
    :return: ExtraTreesClassifier instance.
    """

    model_cfg = config.get("extra_trees", {}).get("model", {})  # Read model configuration
    return ExtraTreesClassifier(n_estimators=int(model_cfg.get("n_estimators", 200)), random_state=int(model_cfg.get("random_state", 42)), n_jobs=int(model_cfg.get("n_jobs", 1)), criterion=str(model_cfg.get("criterion", "gini")), max_features=model_cfg.get("max_features", "sqrt"))  # Return configured selector


def fit_extra_trees_rankings(X_train: pd.DataFrame, y_train: np.ndarray, feature_names: list[str], config: dict) -> tuple[pd.DataFrame, float, dict, ExtraTreesClassifier]:
    """
    Fit Extra Trees and build ranked feature rows.

    :param X_train: Training predictor DataFrame.
    :param y_train: Training labels.
    :param feature_names: Ordered eligible feature names.
    :param config: Effective configuration dictionary.
    :return: Ranked DataFrame, elapsed seconds, model parameters, and fitted selector.
    """

    selector = build_extra_trees_selector(config)  # Build configured Extra Trees selector
    print(f"{BackgroundColors.GREEN}[TRAIN] Fitting final Extra Trees selector on {BackgroundColors.CYAN}{X_train.shape[0]}{BackgroundColors.GREEN} rows x {BackgroundColors.CYAN}{X_train.shape[1]}{BackgroundColors.GREEN} features with {BackgroundColors.CYAN}{selector.n_estimators}{BackgroundColors.GREEN} trees.{Style.RESET_ALL}")  # Log final selector fit start
    start = time.perf_counter()  # Start selector timing
    selector.fit(X_train.to_numpy(copy=False), y_train)  # Fit selector on training rows only
    elapsed = round(time.perf_counter() - start, 6)  # Resolve selector fit duration
    print(f"{BackgroundColors.GREEN}[TRAIN] Final Extra Trees selector fitted in {BackgroundColors.CYAN}{int(round(elapsed))}s{Style.RESET_ALL}")  # Log final selector fit completion
    importances = np.asarray(selector.feature_importances_, dtype=np.float64)  # Read fitted importances
    order = np.lexsort((np.arange(importances.shape[0]), -importances))  # Rank by importance descending then original index
    ranks = np.empty_like(order)  # Allocate rank vector
    ranks[order] = np.arange(1, order.shape[0] + 1)  # Assign one-based ranks
    requested_count = int(config.get("extra_trees", {}).get("selection", {}).get("n_features_to_select", 20))  # Resolve configured selected-feature count
    selected_indices = set(int(index) for index in order[:requested_count])  # Resolve selected feature indexes
    rows = []  # Accumulate full ranked feature rows
    for original_index, feature_name in enumerate(feature_names):  # Preserve original feature index in output rows
        rows.append({"feature_name": feature_name, "original_feature_index": int(original_index), "extra_trees_importance": float(importances[original_index]), "importance_rank": int(ranks[original_index]), "selected": bool(original_index in selected_indices)})  # Store one feature row
    ranked = pd.DataFrame(rows).sort_values(["importance_rank", "original_feature_index"], ascending=[True, True], kind="mergesort").reset_index(drop=True)  # Persist rows in ranking order
    return ranked, elapsed, selector.get_params(deep=True), selector  # Return ranked rows and fitted selector configuration


def calculate_weighted_error_rates(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:  # Calculate weighted selector error rates
    """
    Calculate weighted false-positive and false-negative rates.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: Weighted false-positive rate and false-negative rate.
    """

    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))  # Build stable label universe from observed values
    matrix = confusion_matrix(y_true, y_pred, labels=labels)  # Build confusion matrix across all observed labels
    supports = matrix.sum(axis=1)  # Calculate per-class support
    total_support = float(supports.sum()) if supports.sum() > 0 else 1.0  # Resolve nonzero denominator for weighted rates
    fpr_values = []  # Accumulate per-class false-positive rates
    fnr_values = []  # Accumulate per-class false-negative rates
    for index in range(matrix.shape[0]):  # Iterate every label position
        true_positive = matrix[index, index]  # Resolve true positives for this label
        false_negative = matrix[index, :].sum() - true_positive  # Resolve false negatives for this label
        false_positive = matrix[:, index].sum() - true_positive  # Resolve false positives for this label
        true_negative = matrix.sum() - (true_positive + false_positive + false_negative)  # Resolve true negatives for this label
        fpr_denominator = false_positive + true_negative if false_positive + true_negative > 0 else 1  # Resolve FPR denominator
        fnr_denominator = true_positive + false_negative if true_positive + false_negative > 0 else 1  # Resolve FNR denominator
        fpr_values.append((float(false_positive / fpr_denominator), float(supports[index])))  # Store weighted FPR contribution
        fnr_values.append((float(false_negative / fnr_denominator), float(supports[index])))  # Store weighted FNR contribution
    weighted_fpr = float(sum(value * support for value, support in fpr_values) / total_support)  # Calculate weighted FPR
    weighted_fnr = float(sum(value * support for value, support in fnr_values) / total_support)  # Calculate weighted FNR
    return weighted_fpr, weighted_fnr  # Return weighted error rates


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:  # Calculate selector classification metrics
    """
    Calculate weighted classification metrics.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :return: Dictionary containing accuracy, precision, recall, F1, FPR, and FNR.
    """

    fpr, fnr = calculate_weighted_error_rates(y_true, y_pred)  # Calculate weighted error rates
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)), "f1_score": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)), "fpr": fpr, "fnr": fnr}  # Return weighted selector metrics


def evaluate_selector_metrics(selector: ExtraTreesClassifier, X_test: pd.DataFrame, y_test: np.ndarray) -> tuple[dict, float]:  # Evaluate held-out selector diagnostics
    """
    Evaluate the fitted selector model on held-out data.

    :param selector: Fitted Extra Trees selector.
    :param X_test: Held-out predictor DataFrame.
    :param y_test: Held-out labels.
    :return: Test metrics dictionary and testing duration seconds.
    """

    print(f"{BackgroundColors.GREEN}[TEST] Evaluating fitted selector on held-out rows: {BackgroundColors.CYAN}{X_test.shape[0]}{Style.RESET_ALL}")  # Log held-out evaluation start
    start = time.perf_counter()  # Start held-out prediction timing
    y_pred = selector.predict(X_test.to_numpy(copy=False))  # Predict held-out rows without refitting selector
    elapsed = round(time.perf_counter() - start, 6)  # Resolve held-out prediction duration
    metrics = calculate_classification_metrics(y_test, y_pred)  # Calculate held-out selector metrics
    print(f"{BackgroundColors.GREEN}[TEST] Held-out metrics: F1={BackgroundColors.CYAN}{metrics.get('f1_score')}{BackgroundColors.GREEN}, FPR={BackgroundColors.CYAN}{metrics.get('fpr')}{BackgroundColors.GREEN}, FNR={BackgroundColors.CYAN}{metrics.get('fnr')}{BackgroundColors.GREEN}, time={BackgroundColors.CYAN}{int(round(elapsed))}s{Style.RESET_ALL}")  # Log held-out metrics
    return metrics, elapsed  # Return metrics and timing


def evaluate_cross_validation_metrics(X_train: pd.DataFrame, y_train: np.ndarray, config: dict) -> tuple[dict, str]:  # Evaluate training-only CV diagnostics
    """
    Evaluate selector diagnostics with training-partition CV.

    :param X_train: Training predictor DataFrame.
    :param y_train: Training labels.
    :param config: Effective configuration dictionary.
    :return: Mean CV metrics dictionary and CV method label.
    """

    cv_cfg = config.get("extra_trees", {}).get("cross_validation", {})  # Read CV configuration
    if not bool(cv_cfg.get("enabled", True)):  # Respect disabled CV diagnostics
        print(f"{BackgroundColors.YELLOW}[CV] Training-partition cross-validation disabled by configuration.{Style.RESET_ALL}")  # Log disabled CV
        send_telegram_notice(TELEGRAM_BOT, "[EXTRA TREES CV] Cross-validation disabled by configuration")  # Send disabled-CV notice
        return {"accuracy": None, "precision": None, "recall": None, "f1_score": None, "fpr": None, "fnr": None}, "disabled"  # Return missing CV metrics when disabled
    requested_folds = int(cv_cfg.get("n_folds", 3))  # Resolve configured fold count
    _, class_counts = np.unique(y_train, return_counts=True)  # Count labels in training partition only
    effective_folds = min(requested_folds, int(class_counts.min())) if class_counts.size else 0  # Resolve feasible stratified fold count
    if effective_folds < 2:  # Require at least two folds for CV diagnostics
        print(f"{BackgroundColors.YELLOW}[CV] Skipping CV because training class support is insufficient.{Style.RESET_ALL}")  # Log insufficient CV support
        send_telegram_notice(TELEGRAM_BOT, "[EXTRA TREES CV] Cross-validation skipped: insufficient training class support")  # Send skipped-CV notice
        return {"accuracy": None, "precision": None, "recall": None, "f1_score": None, "fpr": None, "fnr": None}, "insufficient_training_class_support"  # Return missing CV metrics when class support is too small
    print(f"{BackgroundColors.GREEN}[CV] Starting training-partition StratifiedKFold with {BackgroundColors.CYAN}{effective_folds}{BackgroundColors.GREEN} fold(s). Requested folds: {BackgroundColors.CYAN}{requested_folds}{BackgroundColors.GREEN} | Initial CV ETA: {BackgroundColors.CYAN}unavailable{Style.RESET_ALL}")  # Log CV start
    send_telegram_notice(TELEGRAM_BOT, f"[EXTRA TREES CV START] Starting training-partition StratifiedKFold | Effective folds: {effective_folds} | Requested folds: {requested_folds} | Training rows: {X_train.shape[0]} | Features: {X_train.shape[1]} | Initial CV ETA: unavailable")  # Send CV start notice
    splitter = StratifiedKFold(n_splits=effective_folds, shuffle=True, random_state=int(config.get("dataset", {}).get("random_state", 42)))  # Build training-only stratified splitter
    fold_metrics = []  # Accumulate fold metric dictionaries
    X_values = X_train.to_numpy(copy=False)  # Reuse CPU-backed training values for fold slicing
    fold_iterator = tqdm(splitter.split(X_values, y_train), total=effective_folds, desc=f"{BackgroundColors.GREEN}Extra Trees CV{Style.RESET_ALL}", unit="fold", colour="green")  # Show colored CV fold progress with ETA
    cv_started_at = time.perf_counter()  # Start current CV timing for ETA reporting
    for fold_number, (train_index, validation_index) in enumerate(fold_iterator, start=1):  # Iterate training-only CV folds
        fold_selector = build_extra_trees_selector(config)  # Build an isolated selector for this fold
        fold_start = time.perf_counter()  # Start fold timing
        fold_selector.fit(X_values[train_index], y_train[train_index])  # Fit fold selector on fold-training rows only
        fold_pred = fold_selector.predict(X_values[validation_index])  # Predict fold-validation rows only
        fold_metric = calculate_classification_metrics(y_train[validation_index], fold_pred)  # Calculate fold validation metrics
        fold_metrics.append(fold_metric)  # Store fold validation metrics
        fold_elapsed = time.perf_counter() - fold_start  # Resolve fold elapsed seconds
        remaining_folds = max(effective_folds - fold_number, 0)  # Resolve remaining fold count
        cv_eta_label = "complete" if remaining_folds == 0 else format_duration_seconds((time.perf_counter() - cv_started_at) / fold_number * remaining_folds)  # Estimate current CV ETA from completed folds
        fold_iterator.set_postfix(f1=f"{fold_metric.get('f1_score'):.6f}", elapsed=f"{int(round(fold_elapsed))}s", eta=cv_eta_label)  # Update progress bar metrics
        print(f"{BackgroundColors.GREEN}[CV] Fold {BackgroundColors.CYAN}{fold_number}/{effective_folds}{BackgroundColors.GREEN} complete: F1={BackgroundColors.CYAN}{fold_metric.get('f1_score')}{BackgroundColors.GREEN}, time={BackgroundColors.CYAN}{int(round(fold_elapsed))}s{BackgroundColors.GREEN}, CV ETA={BackgroundColors.CYAN}{cv_eta_label}{Style.RESET_ALL}")  # Log fold completion with current CV ETA
        send_telegram_notice(TELEGRAM_BOT, format_cv_metrics_notice(f"[EXTRA TREES CV] Fold {fold_number}/{effective_folds}", fold_metric, fold_elapsed, cv_eta_label))  # Send per-fold CV result notice with current CV ETA
    metric_names = ("accuracy", "precision", "recall", "f1_score", "fpr", "fnr")  # Define exported metric names
    cv_metrics = {name: float(np.mean([metrics[name] for metrics in fold_metrics])) for name in metric_names}  # Average fold metrics
    print(f"{BackgroundColors.GREEN}[CV] Completed CV: F1={BackgroundColors.CYAN}{cv_metrics.get('f1_score')}{BackgroundColors.GREEN}, FPR={BackgroundColors.CYAN}{cv_metrics.get('fpr')}{BackgroundColors.GREEN}, FNR={BackgroundColors.CYAN}{cv_metrics.get('fnr')}{Style.RESET_ALL}")  # Log CV completion
    send_telegram_notice(TELEGRAM_BOT, format_cv_metrics_notice(f"[EXTRA TREES CV] Completed {effective_folds} fold(s)", cv_metrics))  # Send aggregate CV result notice
    return cv_metrics, f"StratifiedKFold(n_splits={effective_folds})"  # Return CV metrics and method label


def get_hardware_specifications() -> dict:
    """
    Return compact hardware metadata.

    :return: Hardware metadata dictionary.
    """

    return {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(), "memory_gb": round(psutil.virtual_memory().total / (1024 ** 3), 3)}  # Return compact host metadata


def build_results_dataframe(ranked: pd.DataFrame, config: dict, csv_path: str, selector_params: dict, selector_elapsed: float, testing_elapsed: float, cv_metrics: dict, test_metrics: dict, cv_method: str, split_metadata: dict, started_at: datetime.datetime, finished_at: datetime.datetime) -> pd.DataFrame:  # Build configured Extra Trees CSV rows
    """
    Build the persisted Extra Trees results DataFrame.

    :param ranked: Ranked feature rows.
    :param config: Effective configuration dictionary.
    :param csv_path: Dataset CSV path.
    :param selector_params: Fitted selector parameters.
    :param selector_elapsed: Selector fitting duration.
    :param testing_elapsed: Held-out selector prediction duration.
    :param cv_metrics: Training-partition CV metrics.
    :param test_metrics: Held-out test metrics.
    :param cv_method: CV method label.
    :param split_metadata: Split and dataset metadata.
    :param started_at: Execution start time.
    :param finished_at: Execution finish time.
    :return: Results DataFrame.
    """

    requested_count = int(config.get("extra_trees", {}).get("selection", {}).get("n_features_to_select", 20))  # Resolve configured selected-feature count
    actual_count = int(ranked["selected"].sum())  # Count selected rows
    if actual_count != requested_count:  # Validate exact selection count
        raise ValueError(f"Extra Trees selected {actual_count} features but expected {requested_count}")  # Raise exact-count error
    ranked = ranked.copy()  # Copy ranked rows before metadata assignment
    ranked.insert(0, "timestamp", finished_at.isoformat())  # Store completion timestamp
    ranked.insert(1, "tool", "Extra Trees")  # Store tool identity
    ranked.insert(2, "run_index", "ranked")  # Store ranked-row marker
    ranked.insert(3, "model", "ExtraTreesClassifier")  # Store selector model name
    ranked.insert(4, "dataset", Path(csv_path).stem)  # Store dataset stem
    ranked.insert(5, "dataset_path", os.path.relpath(csv_path))  # Store relative dataset path
    ranked.insert(6, "hyperparameters", json.dumps(selector_params, default=str, sort_keys=True))  # Store selector parameters
    ranked.insert(7, "cv_method", cv_method)  # Store CV method label
    ranked.insert(8, "train_test_split", f"{1.0 - float(split_metadata['test_size']):.0%}/{float(split_metadata['test_size']):.0%}")  # Store split ratio
    ranked.insert(9, "scaling", "none")  # Store tree-safety scaling marker
    ranked["cv_accuracy"] = cv_metrics.get("accuracy")  # Store CV accuracy diagnostic
    ranked["cv_precision"] = cv_metrics.get("precision")  # Store CV precision diagnostic
    ranked["cv_recall"] = cv_metrics.get("recall")  # Store CV recall diagnostic
    ranked["cv_f1_score"] = cv_metrics.get("f1_score")  # Store CV F1 diagnostic
    ranked["cv_fpr"] = cv_metrics.get("fpr")  # Store CV FPR diagnostic
    ranked["cv_fnr"] = cv_metrics.get("fnr")  # Store CV FNR diagnostic
    ranked["test_accuracy"] = test_metrics.get("accuracy")  # Store held-out test accuracy diagnostic
    ranked["test_precision"] = test_metrics.get("precision")  # Store held-out test precision diagnostic
    ranked["test_recall"] = test_metrics.get("recall")  # Store held-out test recall diagnostic
    ranked["test_f1_score"] = test_metrics.get("f1_score")  # Store held-out test F1 diagnostic
    ranked["test_fpr"] = test_metrics.get("fpr")  # Store held-out test FPR diagnostic
    ranked["test_fnr"] = test_metrics.get("fnr")  # Store held-out test FNR diagnostic
    ranked["configured_selected_feature_count"] = requested_count  # Store configured selected-feature count
    ranked["actual_selected_feature_count"] = actual_count  # Store actual selected-feature count
    ranked["n_estimators"] = int(selector_params.get("n_estimators", 0))  # Store estimator count
    ranked["random_state"] = selector_params.get("random_state")  # Store model seed
    ranked["n_jobs"] = selector_params.get("n_jobs")  # Store worker count
    ranked["n_train"] = int(split_metadata["n_train"])  # Store training row count
    ranked["n_test"] = int(split_metadata["n_test"])  # Store held-out row count
    ranked["source_feature_count"] = int(split_metadata["source_feature_count"])  # Store source predictor count
    ranked["eligible_feature_count"] = int(split_metadata["eligible_feature_count"])  # Store eligible predictor count
    ranked["target_column"] = split_metadata["target_column"]  # Store target column name
    ranked["excluded_columns"] = json.dumps(split_metadata["excluded_columns"], ensure_ascii=False)  # Store excluded predictor metadata
    ranked["feature_extraction_time_s"] = int(round(float(selector_elapsed)))  # Store selector fit duration as integer seconds
    ranked["training_time_s"] = int(round(float(selector_elapsed)))  # Store selector training duration as integer seconds
    ranked["testing_time_s"] = int(round(float(testing_elapsed)))  # Store held-out prediction duration as integer seconds
    ranked["elapsed_run_time"] = int(round((finished_at - started_at).total_seconds()))  # Store full script duration as integer seconds
    ranked["hardware"] = json.dumps(get_hardware_specifications(), default=str, sort_keys=True)  # Store hardware metadata
    selected_features = ranked.loc[ranked["selected"], "feature_name"].tolist()  # Resolve ranked selected features for GA-compatible summary columns
    ranked["best_features"] = json.dumps(selected_features, ensure_ascii=False)  # Store selected features in GA-compatible payload column
    ranked["union_features_across_runs"] = json.dumps(selected_features, ensure_ascii=False)  # Store single-run selected feature union
    ranked["rfe_ranking"] = None  # Preserve GA-compatible column with no RFE ranking for Extra Trees
    configured_columns = config.get("extra_trees", {}).get("export", {}).get("results_csv_columns", list(DEFAULT_RESULTS_CSV_COLUMNS))  # Read configured result header
    for column in configured_columns:  # Ensure every configured column exists before export
        if column not in ranked.columns:  # Add configured columns absent from generated metadata
            ranked[column] = None  # Store project missing-value marker through pandas
    return ranked[configured_columns]  # Return complete results DataFrame in configured column order


def save_results(results: pd.DataFrame, csv_output: Path) -> None:
    """
    Persist Extra Trees results CSV.

    :param results: Results DataFrame.
    :param csv_output: Output CSV path.
    :return: None.
    """

    csv_output.parent.mkdir(parents=True, exist_ok=True)  # Create output directory
    temporary_path = csv_output.with_suffix(csv_output.suffix + ".tmp")  # Build same-directory temporary path
    print(f"{BackgroundColors.GREEN}[EXPORT] Writing ranked Extra Trees CSV to {BackgroundColors.CYAN}{csv_output}{Style.RESET_ALL}")  # Log export start
    results.to_csv(temporary_path, index=False)  # Write complete ranked CSV atomically staged
    os.replace(temporary_path, csv_output)  # Replace final CSV atomically
    print(f"{BackgroundColors.GREEN}[EXPORT] CSV saved successfully with {BackgroundColors.CYAN}{len(results)}{BackgroundColors.GREEN} ranked feature row(s).{Style.RESET_ALL}")  # Log export completion


def print_extra_trees_summary(csv_output: Path, selected_count: int, eligible_count: int, cv_metrics: dict, test_metrics: dict, elapsed_seconds: float) -> None:  # Print colored Extra Trees execution summary
    """
    Print colored Extra Trees execution summary.

    :param csv_output: Persisted results CSV path.
    :param selected_count: Number of selected features.
    :param eligible_count: Number of eligible features.
    :param cv_metrics: Training-partition CV metrics.
    :param test_metrics: Held-out test metrics.
    :param elapsed_seconds: Total execution time in seconds.
    :return: None.
    """

    print(f"\n{BackgroundColors.GREEN}{'=' * 80}{Style.RESET_ALL}")  # Print summary separator
    print(f"{BackgroundColors.GREEN}Extra Trees feature selection completed{Style.RESET_ALL}")  # Print summary title
    print(f"{BackgroundColors.GREEN}Results CSV: {BackgroundColors.CYAN}{csv_output}{Style.RESET_ALL}")  # Print output path
    print(f"{BackgroundColors.GREEN}Selected features: {BackgroundColors.CYAN}{selected_count}{BackgroundColors.GREEN} of {BackgroundColors.CYAN}{eligible_count}{Style.RESET_ALL}")  # Print selected-feature count
    print(f"{BackgroundColors.GREEN}CV F1: {BackgroundColors.CYAN}{cv_metrics.get('f1_score')}{BackgroundColors.GREEN} | CV FPR: {BackgroundColors.CYAN}{cv_metrics.get('fpr')}{BackgroundColors.GREEN} | CV FNR: {BackgroundColors.CYAN}{cv_metrics.get('fnr')}{Style.RESET_ALL}")  # Print CV diagnostics
    print(f"{BackgroundColors.GREEN}Test F1: {BackgroundColors.CYAN}{test_metrics.get('f1_score')}{BackgroundColors.GREEN} | Test FPR: {BackgroundColors.CYAN}{test_metrics.get('fpr')}{BackgroundColors.GREEN} | Test FNR: {BackgroundColors.CYAN}{test_metrics.get('fnr')}{Style.RESET_ALL}")  # Print held-out diagnostics
    print(f"{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{int(round(elapsed_seconds))}s{Style.RESET_ALL}")  # Print elapsed time
    print(f"{BackgroundColors.GREEN}{'=' * 80}{Style.RESET_ALL}\n")  # Print summary separator


def calculate_execution_time(start_time: datetime.datetime, finish_time: datetime.datetime) -> str:
    """
    Format elapsed execution time.

    :param start_time: Program start datetime.
    :param finish_time: Program finish datetime.
    :return: Human-readable elapsed execution time.
    """

    total_seconds = int(round((finish_time - start_time).total_seconds()))  # Convert elapsed duration to whole seconds
    total_seconds = max(total_seconds, 0)  # Prevent negative output from clock drift
    hours, remainder = divmod(total_seconds, 3600)  # Split elapsed hours
    minutes, seconds = divmod(remainder, 60)  # Split elapsed minutes and seconds
    return f"{hours:02d}h {minutes:02d}m {seconds:02d}s"  # Return formatted runtime


def format_duration_seconds(total_seconds: float) -> str:
    """
    Format a duration in seconds.

    :param total_seconds: Duration in seconds.
    :return: Human-readable duration text.
    """

    seconds = max(int(round(total_seconds)), 0)  # Normalize duration to non-negative whole seconds
    hours, remainder = divmod(seconds, 3600)  # Split elapsed hours
    minutes, seconds = divmod(remainder, 60)  # Split elapsed minutes and seconds
    if hours:  # Include hours only when needed
        return f"{hours}h {minutes}m {seconds}s"  # Return hour-minute-second duration
    if minutes:  # Include minutes only when needed
        return f"{minutes}m {seconds}s"  # Return minute-second duration
    return f"{seconds}s"  # Return second-only duration


def format_metric_value(value: Any) -> str:
    """
    Format a metric value for Telegram notices.

    :param value: Raw metric value.
    :return: Readable metric text.
    """

    if value is None:  # Preserve unavailable metrics explicitly
        return "unavailable"  # Return unavailable marker
    try:  # Format numeric metrics compactly
        return f"{float(value):.6f}"  # Return fixed precision metric
    except Exception:  # Preserve unexpected metric values without hiding the notice
        return str(value)  # Return fallback text


def format_extra_trees_config_notice(config: dict, dataset_path: str) -> str:
    """
    Format the Extra Trees configuration Telegram notice.

    :param config: Effective configuration dictionary.
    :param dataset_path: Dataset path.
    :return: Telegram configuration notice text.
    """

    execution_cfg = config.get("execution", {})  # Read execution configuration
    selection_cfg = config.get("extra_trees", {}).get("selection", {})  # Read feature-selection configuration
    model_cfg = config.get("extra_trees", {}).get("model", {})  # Read model configuration
    cv_cfg = config.get("extra_trees", {}).get("cross_validation", {})  # Read cross-validation configuration
    export_cfg = config.get("extra_trees", {}).get("export", {})  # Read export configuration
    mode = "Combined files in memory" if bool(execution_cfg.get("combined_files", False)) else "Single CSV"  # Resolve execution mode label
    return f"[EXTRA TREES CONFIG] Dataset: {dataset_path} | Mode: {mode} | Selected features: {selection_cfg.get('n_features_to_select', 20)} | Estimators: {model_cfg.get('n_estimators', 200)} | n_jobs: {model_cfg.get('n_jobs', 1)} | Random state: {model_cfg.get('random_state', 42)} | CV: {'Enabled' if bool(cv_cfg.get('enabled', True)) else 'Disabled'} | CV folds: {cv_cfg.get('n_folds', 10)} | Results: {export_cfg.get('results_dir', 'Feature_Analysis/Extra_Trees')}/{export_cfg.get('results_filename', 'Extra_Trees_Results.csv')}"  # Return compact config summary


def format_cv_metrics_notice(prefix: str, metrics: dict, elapsed_seconds: Optional[float] = None, eta_label: Optional[str] = None) -> str:
    """
    Format cross-validation metrics for Telegram notices.

    :param prefix: Notice prefix.
    :param metrics: Metric mapping.
    :param elapsed_seconds: Optional elapsed seconds.
    :param eta_label: Optional current CV ETA label.
    :return: Telegram CV metrics notice text.
    """

    elapsed = f" | Time: {int(round(elapsed_seconds))}s" if elapsed_seconds is not None else ""  # Format optional elapsed time
    eta = f" | CV ETA: {eta_label}" if eta_label else ""  # Format optional current-CV ETA
    return f"{prefix} | Accuracy: {format_metric_value(metrics.get('accuracy'))} | Precision: {format_metric_value(metrics.get('precision'))} | Recall: {format_metric_value(metrics.get('recall'))} | F1: {format_metric_value(metrics.get('f1_score'))} | FPR: {format_metric_value(metrics.get('fpr'))} | FNR: {format_metric_value(metrics.get('fnr'))}{elapsed}{eta}"  # Return compact CV notice


def setup_telegram_bot(config: dict) -> Any:
    """
    Set up the Telegram bot for Extra Trees notifications.

    :param config: Effective configuration dictionary.
    :return: Telegram bot instance or None.
    """

    if not config.get("telegram", {}).get("enabled", True):  # Respect shared Telegram enablement
        return None  # Skip Telegram setup when disabled
    if TelegramBot is None or telegram_module is None:  # Skip Telegram setup when imports failed
        return None  # Return no bot when Telegram support is unavailable
    print(f"{BackgroundColors.GREEN}Setting up Telegram bot for messages...{Style.RESET_ALL}")  # Match stacking.py setup log
    bot = TelegramBot()  # Initialize the shared Telegram bot implementation
    telegram_module.TELEGRAM_DEVICE_INFO = f"{telegram_module.get_local_ip()} - {platform.system()}"  # Set device prefix like stacking.py
    telegram_module.RUNNING_CODE = os.path.basename(__file__)  # Set script name prefix like stacking.py
    telegram_module.TELEGRAM_BOT = bot  # Share bot with Telegram exception handler
    return bot  # Return configured Telegram bot


def send_telegram_notice(bot: Any, messages: Any) -> None:
    """
    Send an Extra Trees Telegram notification when available.

    :param bot: Telegram bot instance or None.
    :param messages: Message text or list of message strings.
    :return: None.
    """

    if send_telegram_message is None:  # Skip notification when Telegram support is unavailable
        return None  # Return without side effects
    send_telegram_message(bot, messages)  # Reuse existing guarded Telegram delivery
    return None  # Return explicit None


def initialize_logger(config: dict) -> Any:
    """
    Initialize Extra Trees terminal and file logging.

    :param config: Effective configuration dictionary.
    :return: Logger instance.
    """

    logs_dir = config.get("paths", {}).get("logs_dir", "./Logs")  # Resolve shared logs directory
    clean_log = bool(config.get("logging", {}).get("clean", True))  # Resolve shared log-cleaning policy
    os.makedirs(logs_dir, exist_ok=True)  # Create logs directory when missing
    log_path = Path(logs_dir) / f"{Path(__file__).stem}.log"  # Build script log path
    logger_instance = Logger(str(log_path), clean=clean_log, timestamp_timezone=SAO_PAULO_TIMEZONE_NAME)  # Create timestamped project logger
    sys.stdout = logger_instance  # Redirect stdout to terminal and log file
    sys.stderr = logger_instance  # Redirect stderr and tqdm output to terminal and log file
    print(f"{BackgroundColors.GREEN}[LOGGING] Writing Extra Trees logs to {BackgroundColors.CYAN}{log_path}{Style.RESET_ALL}")  # Log file target
    return logger_instance  # Return active logger


def run_extra_trees_feature_selection(config: dict, csv_path: str) -> Path:
    """
    Run Extra Trees feature selection and persist results.

    :param config: Effective configuration dictionary.
    :param csv_path: Dataset CSV path.
    :return: Persisted results CSV path.
    """

    started_at = datetime.datetime.now(datetime.timezone.utc)  # Record start timestamp
    output_dir, csv_output = resolve_output_paths(config, csv_path)  # Resolve output locations
    print(f"{BackgroundColors.GREEN}Starting Extra Trees feature selection for {BackgroundColors.CYAN}{csv_path}{Style.RESET_ALL}")  # Print colored start message
    print(f"{BackgroundColors.GREEN}[CONFIG] Results directory: {BackgroundColors.CYAN}{output_dir}{BackgroundColors.GREEN} | Results file: {BackgroundColors.CYAN}{csv_output.name}{Style.RESET_ALL}")  # Log resolved export configuration
    dataframe = load_dataset(csv_path, config)  # Load dataset once
    X_train, y_train, X_test, y_test, feature_names, split_metadata = prepare_training_data(dataframe, config)  # Prepare training-only selector inputs and held-out diagnostics data
    print(f"{BackgroundColors.GREEN}[MEMORY] Releasing raw cleaned DataFrame before CV and fitting.{Style.RESET_ALL}")  # Log dataframe release
    del dataframe  # Release full dataframe before fitting selector
    cv_metrics, cv_method = evaluate_cross_validation_metrics(X_train, y_train, config)  # Evaluate selector CV diagnostics on training partition only
    ranked, selector_elapsed, selector_params, selector = fit_extra_trees_rankings(X_train, y_train, feature_names, config)  # Fit selector and rank features
    test_metrics, testing_elapsed = evaluate_selector_metrics(selector, X_test, y_test)  # Evaluate fitted selector on held-out split
    print(f"{BackgroundColors.GREEN}[MEMORY] Releasing train/test matrices and fitted selector before CSV assembly.{Style.RESET_ALL}")  # Log matrix release
    del X_train, y_train, X_test, y_test, selector  # Release data and fitted selector before CSV assembly
    finished_at = datetime.datetime.now(datetime.timezone.utc)  # Record finish timestamp
    print(f"{BackgroundColors.GREEN}[RESULTS] Building ranked feature-result rows.{Style.RESET_ALL}")  # Log results assembly start
    results = build_results_dataframe(ranked, config, csv_path, selector_params, selector_elapsed, testing_elapsed, cv_metrics, test_metrics, cv_method, split_metadata, started_at, finished_at)  # Build output rows
    save_results(results, csv_output)  # Persist ranked results
    print_extra_trees_summary(csv_output, int(results["selected"].sum()), int(split_metadata["eligible_feature_count"]), cv_metrics, test_metrics, (finished_at - started_at).total_seconds())  # Print colored completion summary
    send_telegram_notice(TELEGRAM_BOT, f"Finished Extra Trees result: Extra-Trees-{int(results['selected'].sum())} - Dataset: {Path(csv_path).stem} - Selected {int(results['selected'].sum())}/{int(split_metadata['eligible_feature_count'])} - Test F1: {test_metrics.get('f1_score')} - CV F1: {cv_metrics.get('f1_score')} in {calculate_execution_time(started_at, finished_at)}")  # Send result-style notification
    return csv_output  # Return output CSV path


def main() -> None:
    """
    Execute Extra Trees feature selection from CLI.

    :return: None.
    """
    
    start_time = datetime.datetime.now()  # Record program start time
    cli_args = parse_cli_args()  # Parse CLI arguments
    config = get_config(cli_args)  # Resolve effective configuration
    set_runtime_process_name(getattr(cli_args, "process_name", None), script_path=__file__, config=config)  # Re-apply with merged config so config.yaml can override only when CLI omitted the process name.
    global logger  # Use module-level logger instance
    logger = initialize_logger(config)  # Initialize file logging after configuration resolution
    print(
        f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Extra Trees for Feature Selection{BackgroundColors.GREEN} Program!{Style.RESET_ALL}\n"
    )  # Output the welcome message
    dataset_path = config.get("execution", {}).get("dataset_path")  # Resolve dataset path
    if not dataset_path:  # Validate dataset path
        raise ValueError("execution.dataset_path must be provided")  # Raise explicit missing-dataset error
    global TELEGRAM_BOT  # Use module-level Telegram bot instance
    TELEGRAM_BOT = setup_telegram_bot(config)  # Initialize Telegram notifications when configured
    send_telegram_notice(TELEGRAM_BOT, f"Starting Extra Trees feature selection at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")  # Send startup notification
    send_telegram_notice(TELEGRAM_BOT, format_extra_trees_config_notice(config, str(dataset_path)))  # Send startup configuration notification
    run_extra_trees_feature_selection(config, str(dataset_path))  # Run feature selection workflow
    finish_time = datetime.datetime.now()  # Record program finish time
    print(
        f"\n{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
    )  # Output start, finish, and elapsed time
    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
    )  # Output program-finished message
    send_telegram_notice(TELEGRAM_BOT, [f"Finished Extra Trees feature selection at {finish_time.strftime('%Y-%m-%d %H:%M:%S')} | Execution time: {calculate_execution_time(start_time, finish_time)}"])  # Send finish notification
    


if __name__ == "__main__":  # Execute only when called as a script
    try:  # Surface clean CLI failures
        main()  # Run CLI entry point
    except Exception as exc:  # Report fatal error without hiding traceback
        print(str(exc), file=sys.stderr)  # Print concise failure to stderr
        if send_exception_via_telegram is not None:  # Forward fatal errors when Telegram support is available
            send_exception_via_telegram(type(exc), exc, exc.__traceback__)  # Send full traceback through existing Telegram path
        raise  # Preserve traceback and nonzero exit
