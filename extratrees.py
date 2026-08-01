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
from sklearn.ensemble import ExtraTreesClassifier  # Fit Extra Trees feature importances
from sklearn.model_selection import train_test_split  # Split before selector fitting


NON_FEATURE_COLUMNS = ("Unnamed: 0", "Flow ID", "Source IP", "Destination IP", "Timestamp")  # Define leakage-prone metadata columns
TARGET_COLUMN_ALIASES = ("Label", "label", "attack_type", "Attack Type", "Class", "class", "Target", "target")  # Define common target column names


def get_default_config() -> dict:
    """
    Return default Extra Trees configuration.

    :return: Default configuration dictionary.
    """

    return {  # Return internal defaults before YAML and CLI overrides
        "execution": {"verbose": False, "dataset_path": "./Datasets/CICDDoS2019/01-12/DrDoS_DNS.csv", "low_memory": False},  # Define execution defaults
        "dataset": {"test_size": 0.2, "random_state": 42},  # Define split defaults
        "extra_trees": {  # Define selector defaults
            "selection": {"n_features_to_select": 20},  # Define Extra-Trees-20 default representation
            "model": {"n_estimators": 200, "random_state": 42, "n_jobs": 1, "criterion": "gini", "max_features": "sqrt"},  # Define Extra Trees classifier defaults
            "export": {"results_dir": "Feature_Analysis/Extra_Trees", "results_filename": "Extra_Trees_Results.csv"},  # Define export path defaults
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
    parser.add_argument("--dataset-path", "--dataset_path", dest="dataset_path", type=str, default=None, help="Path to dataset CSV")  # Add dataset override
    parser.add_argument("--n-features-to-select", dest="n_features_to_select", type=int, default=None, help="Number of Extra Trees features to select")  # Add selected-count override
    parser.add_argument("--n-estimators", dest="n_estimators", type=int, default=None, help="Number of Extra Trees estimators")  # Add estimator-count override
    parser.add_argument("--random-state", dest="random_state", type=int, default=None, help="Random seed")  # Add seed override
    parser.add_argument("--n-jobs", dest="n_jobs", type=int, default=None, help="Extra Trees worker threads")  # Add worker override
    parser.add_argument("--results-dir", dest="results_dir", type=str, default=None, help="Results directory relative to dataset directory")  # Add export-directory override
    parser.add_argument("--results-filename", dest="results_filename", type=str, default=None, help="Results CSV filename")  # Add export-filename override
    parser.add_argument("--verbose", action="store_true", default=False, help="Enable verbose logging")  # Add verbose override
    return parser.parse_args()  # Return parsed CLI arguments


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
    if cli_args.n_features_to_select is not None:  # Apply selected-count override only when supplied
        overrides.setdefault("extra_trees", {}).setdefault("selection", {})["n_features_to_select"] = cli_args.n_features_to_select  # Store selected-count override
    if cli_args.n_estimators is not None:  # Apply estimator-count override only when supplied
        overrides.setdefault("extra_trees", {}).setdefault("model", {})["n_estimators"] = cli_args.n_estimators  # Store estimator-count override
    if cli_args.random_state is not None:  # Apply seed override only when supplied
        overrides.setdefault("extra_trees", {}).setdefault("model", {})["random_state"] = cli_args.random_state  # Store model seed override
        overrides.setdefault("dataset", {})["random_state"] = cli_args.random_state  # Store split seed override
    if cli_args.n_jobs is not None:  # Apply worker override only when supplied
        overrides.setdefault("extra_trees", {}).setdefault("model", {})["n_jobs"] = cli_args.n_jobs  # Store worker override
    if cli_args.results_dir is not None:  # Apply results directory override only when supplied
        overrides.setdefault("extra_trees", {}).setdefault("export", {})["results_dir"] = cli_args.results_dir  # Store results directory override
    if cli_args.results_filename is not None:  # Apply results filename override only when supplied
        overrides.setdefault("extra_trees", {}).setdefault("export", {})["results_filename"] = cli_args.results_filename  # Store results filename override
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
    export_cfg = config.get("extra_trees", {}).get("export", {})  # Read export configuration
    dataset_cfg = config.get("dataset", {})  # Read dataset configuration
    validate_positive_int(selection_cfg.get("n_features_to_select", 20), "extra_trees.selection.n_features_to_select")  # Validate selected-feature count
    validate_positive_int(model_cfg.get("n_estimators", 200), "extra_trees.model.n_estimators")  # Validate estimator count
    validate_n_jobs(model_cfg.get("n_jobs", 1), "extra_trees.model.n_jobs")  # Validate worker count
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
    dataset_dir = Path(csv_path).expanduser().resolve().parent  # Resolve dataset directory
    output_dir = Path(results_dir_raw).expanduser()  # Normalize configured output directory
    if not output_dir.is_absolute():  # Resolve relative exports beside dataset
        output_dir = dataset_dir / output_dir  # Build dataset-relative export directory
    return output_dir.resolve(), (output_dir / results_filename).resolve()  # Return resolved paths


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
    dataframe = pd.read_csv(path, low_memory=low_memory)  # Load dataset CSV
    dataframe.columns = sanitize_feature_names([str(column).strip() for column in dataframe.columns])  # Apply stacking-compatible column sanitization
    if dataframe.shape[1] < 2:  # Validate predictor plus target columns
        raise ValueError("Dataset must contain at least one feature column and one target column")  # Raise explicit dataset shape error
    return dataframe  # Return loaded dataframe


def prepare_training_data(dataframe: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, np.ndarray, list[str], dict]:
    """
    Prepare training-only data for Extra Trees ranking.

    :param dataframe: Loaded dataset DataFrame.
    :param config: Effective configuration dictionary.
    :return: Training predictors, training labels, feature names, and split metadata.
    """

    target_column = resolve_target_column(list(dataframe.columns))  # Resolve target column
    excluded = {normalize_feature_name(column) for column in NON_FEATURE_COLUMNS}  # Build leakage-prone metadata exclusion set
    excluded_present = [column for column in dataframe.columns if column != target_column and normalize_feature_name(column) in excluded]  # Record leakage-prone columns present in source data
    feature_columns = [column for column in dataframe.columns if column != target_column and normalize_feature_name(column) not in excluded]  # Select eligible predictor names
    numeric_columns = [column for column in feature_columns if pd.api.types.is_numeric_dtype(dataframe[column])]  # Keep numeric predictors only
    requested_count = int(config.get("extra_trees", {}).get("selection", {}).get("n_features_to_select", 20))  # Resolve requested selection count
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
    metadata = {"target_column": target_column, "n_train": int(len(train_indices)), "n_test": int(len(test_indices)), "source_feature_count": int(dataframe.shape[1] - 1), "eligible_feature_count": int(len(numeric_columns)), "excluded_columns": excluded_present, "test_size": float(test_size)}  # Record split and eligibility metadata
    return X_train, y_train, numeric_columns, metadata  # Return training-only selector inputs


def build_extra_trees_selector(config: dict) -> ExtraTreesClassifier:
    """
    Build the configured Extra Trees classifier.

    :param config: Effective configuration dictionary.
    :return: ExtraTreesClassifier instance.
    """

    model_cfg = config.get("extra_trees", {}).get("model", {})  # Read model configuration
    return ExtraTreesClassifier(n_estimators=int(model_cfg.get("n_estimators", 200)), random_state=int(model_cfg.get("random_state", 42)), n_jobs=int(model_cfg.get("n_jobs", 1)), criterion=str(model_cfg.get("criterion", "gini")), max_features=model_cfg.get("max_features", "sqrt"))  # Return configured selector


def fit_extra_trees_rankings(X_train: pd.DataFrame, y_train: np.ndarray, feature_names: list[str], config: dict) -> tuple[pd.DataFrame, float, dict]:
    """
    Fit Extra Trees and build ranked feature rows.

    :param X_train: Training predictor DataFrame.
    :param y_train: Training labels.
    :param feature_names: Ordered eligible feature names.
    :param config: Effective configuration dictionary.
    :return: Ranked DataFrame, elapsed seconds, and model parameters.
    """

    selector = build_extra_trees_selector(config)  # Build configured Extra Trees selector
    start = time.perf_counter()  # Start selector timing
    selector.fit(X_train.to_numpy(copy=False), y_train)  # Fit selector on training rows only
    elapsed = round(time.perf_counter() - start, 6)  # Resolve selector fit duration
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
    return ranked, elapsed, selector.get_params(deep=True)  # Return ranked rows and fitted configuration


def get_hardware_specifications() -> dict:
    """
    Return compact hardware metadata.

    :return: Hardware metadata dictionary.
    """

    return {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count(), "memory_gb": round(psutil.virtual_memory().total / (1024 ** 3), 3)}  # Return compact host metadata


def build_results_dataframe(ranked: pd.DataFrame, config: dict, csv_path: str, selector_params: dict, selector_elapsed: float, split_metadata: dict, started_at: datetime.datetime, finished_at: datetime.datetime) -> pd.DataFrame:
    """
    Build the persisted Extra Trees results DataFrame.

    :param ranked: Ranked feature rows.
    :param config: Effective configuration dictionary.
    :param csv_path: Dataset CSV path.
    :param selector_params: Fitted selector parameters.
    :param selector_elapsed: Selector fitting duration.
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
    ranked.insert(7, "cv_method", "train_test_split")  # Store split method
    ranked.insert(8, "train_test_split", f"{1.0 - float(split_metadata['test_size']):.0%}/{float(split_metadata['test_size']):.0%}")  # Store split ratio
    ranked.insert(9, "scaling", "none")  # Store tree-safety scaling marker
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
    ranked["feature_extraction_time_s"] = round(float(selector_elapsed), 6)  # Store selector fit duration
    ranked["elapsed_run_time"] = round((finished_at - started_at).total_seconds(), 6)  # Store full script duration
    ranked["hardware"] = json.dumps(get_hardware_specifications(), default=str, sort_keys=True)  # Store hardware metadata
    return ranked  # Return complete results DataFrame


def save_results(results: pd.DataFrame, csv_output: Path) -> None:
    """
    Persist Extra Trees results CSV.

    :param results: Results DataFrame.
    :param csv_output: Output CSV path.
    :return: None.
    """

    csv_output.parent.mkdir(parents=True, exist_ok=True)  # Create output directory
    temporary_path = csv_output.with_suffix(csv_output.suffix + ".tmp")  # Build same-directory temporary path
    results.to_csv(temporary_path, index=False)  # Write complete ranked CSV atomically staged
    os.replace(temporary_path, csv_output)  # Replace final CSV atomically


def run_extra_trees_feature_selection(config: dict, csv_path: str) -> Path:
    """
    Run Extra Trees feature selection and persist results.

    :param config: Effective configuration dictionary.
    :param csv_path: Dataset CSV path.
    :return: Persisted results CSV path.
    """

    started_at = datetime.datetime.now(datetime.timezone.utc)  # Record start timestamp
    output_dir, csv_output = resolve_output_paths(config, csv_path)  # Resolve output locations
    dataframe = load_dataset(csv_path, config)  # Load dataset once
    X_train, y_train, feature_names, split_metadata = prepare_training_data(dataframe, config)  # Prepare training-only selector inputs
    del dataframe  # Release full dataframe before fitting selector
    ranked, selector_elapsed, selector_params = fit_extra_trees_rankings(X_train, y_train, feature_names, config)  # Fit selector and rank features
    del X_train, y_train  # Release training data before CSV assembly
    finished_at = datetime.datetime.now(datetime.timezone.utc)  # Record finish timestamp
    results = build_results_dataframe(ranked, config, csv_path, selector_params, selector_elapsed, split_metadata, started_at, finished_at)  # Build output rows
    save_results(results, csv_output)  # Persist ranked results
    print(f"Extra Trees feature selection saved to {csv_output}")  # Report output path
    print(f"Selected features: {int(results['selected'].sum())} of {int(results['eligible_feature_count'].iloc[0])}")  # Report selected feature count
    return csv_output  # Return output CSV path


def main() -> None:
    """
    Execute Extra Trees feature selection from CLI.

    :return: None.
    """
    
    cli_args = parse_cli_args()  # Parse CLI arguments
    config = get_config(cli_args)  # Resolve effective configuration
    dataset_path = config.get("execution", {}).get("dataset_path")  # Resolve dataset path
    if not dataset_path:  # Validate dataset path
        raise ValueError("execution.dataset_path must be provided")  # Raise explicit missing-dataset error
    run_extra_trees_feature_selection(config, str(dataset_path))  # Run feature selection workflow


if __name__ == "__main__":  # Execute only when called as a script
    try:  # Surface clean CLI failures
        main()  # Run CLI entry point
    except Exception as exc:  # Report fatal error without hiding traceback
        print(str(exc), file=sys.stderr)  # Print concise failure to stderr
        raise  # Preserve traceback and nonzero exit
