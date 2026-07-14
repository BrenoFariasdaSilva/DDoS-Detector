"""
================================================================================
Individual and Stacking Ensemble Classifier Evaluation Script (stacking.py)
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-12-01
Description :
    Orchestrates evaluation of individual classifiers and a stacking ensemble
    across multiple feature sets derived from the project's feature-analysis
    artifacts (Genetic Algorithm, RFE, PCA). The script centralizes dataset
    loading, preprocessing, feature-set assembly, per-model evaluation and the
    export of consolidated CSV results for downstream analysis.

Core capabilities:
    - Automatic loading and sanitization of CSV datasets (NaN/infinite removal)
    - Integration of GA, RFE and PCA outputs to build alternative feature sets
    - Scaling, optional PCA projection and selective feature subsetting
    - Evaluation of many classifiers (RF, SVM, XGBoost, LightGBM, etc.) and a
        stacking meta-classifier combining their predictions
    - Calculation of standard metrics (accuracy, precision, recall, F1) plus
        FPR/FNR and elapsed-time reporting
    - Export of `Stacking_Classifier_Results.csv` including `features_list`
        and hardware metadata for reproducibility
    - Utilities to discover feature-analysis files at file, parent or dataset level

Usage:
    - Configure `DATASETS` mapping or call `main()` directly.
    - Run: `python3 stacking.py` or via the repository Makefile target.

Outputs:
    - `Stacking_Classifier_Results.csv` (per-dataset `Feature_Analysis/` directory)
    - Terminal logs, optional Telegram notifications and sound on completion

Notes & conventions:
    - Input CSVs are expected under `Datasets/<DatasetName>/...` and the last
        column conventionally contains the target variable.
    - Feature-analysis artifacts are expected under `.../Feature_Analysis/`:
        `Genetic_Algorithm_Results.csv`, `RFE_Run_Results.csv`, `PCA_Results.csv`.
    - Defaults assume CSV input; Parquet support can be added as needed.
    - Toggle `VERBOSE = True` for additional diagnostic output.

Execution Modes:
    - `separate_files`: Separate files evaluation where every attack is treated as
        a single positive class and non-attack as negative.
    - `combined_files`: Combined files evaluation where each distinct attack type
        is treated as its own class label for per-attack evaluation.
    - `both`: Run both `separate_files` and `combined_files` evaluations; outputs are
        written under separate subfolders (`separate_files` and `combined_files`).
    - Default: `both` (configurable via CLI or `CONFIG['execution']['execution_mode']`).

TODOs:
    - Add CLI argument parsing for dataset paths and runtime flags.
    - Add native Parquet support and safer large-file streaming.
    - Add voting ensemble baseline and parallelize per-feature-set evaluations.

Dependencies:
    - Python >= 3.8
    - pandas, numpy, scikit-learn, colorama, lightgbm, xgboost
    - Optional: telegram_bot for notifications
"""

import arff  # Liac-arff, used to load ARFF files
import argparse  # For parsing command-line arguments
import ast  # For safely evaluating Python literals
import atexit  # For playing a sound when the program finishes
import concurrent.futures  # For parallel execution
import dataframe_image as dfi  # For exporting DataFrame styled tables as PNG images
import datetime  # For getting the current date and time
import gc  # For explicit garbage collection to reclaim memory from deleted objects
import glob  # For file pattern matching
import hashlib  # Build stable digests for compact classifier metadata
import importlib  # Import importlib for dynamic module import
import json  # Import json for handling JSON strings within the CSV
import lightgbm as lgb  # For LightGBM model
import math  # For mathematical operations
import matplotlib  # For plotting configuration
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt  # For creating t-SNE visualization plots
import multiprocessing as mp  # For process-isolated explainability execution
import numpy as np  # Import numpy for numerical operations
import optuna  # For Bayesian hyperparameter optimization (AutoML)
import os  # For running a command in the terminal
import pandas as pd  # Import pandas for data manipulation
import pickle  # For loading PCA objects
import platform  # For getting the operating system name
import psutil  # For verifying system RAM
import queue as queue_module  # For queue timeout exceptions from explainability status channels
import re  # For regular expressions
import seaborn as sns  # For generating feature usage heatmaps
import shutil  # For removing temporary feature-source spill directories
import shap  # For SHAP explainability analysis
import subprocess  # For running small system commands (sysctl/wmic)
import sys  # For system-specific parameters and functions
import telegram_bot as telegram_module  # For setting Telegram prefix and device info
import threading  # For low-overhead training RAM sampling
import time  # For measuring execution time
import tempfile  # For atomic cache file writes using temp-file and rename strategy
import traceback  # For formatting and printing exception tracebacks
import tracemalloc  # Capture optional Python allocation snapshots at phase boundaries
import uuid  # Generate collision-resistant watcher run identifiers
import yaml  # Import YAML library
from colorama import Style  # For terminal text styling
from joblib import dump, load  # For exporting and loading trained models and scalers
from lime.lime_tabular import LimeTabularExplainer  # Import LIME library
from Logger import Logger  # For logging output to both terminal and file
from pathlib import Path  # For handling file paths
from scipy.io import arff as scipy_arff  # Used to read ARFF files
from sklearn.base import clone  # Clone estimator prototypes before each atomic fit
from sklearn.decomposition import PCA  # For Principal Component Analysis
from sklearn.ensemble import (  # For ensemble models
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.inspection import permutation_importance  # Import permutation importance
from sklearn.linear_model import LogisticRegression  # For logistic regression model
from sklearn.manifold import TSNE  # For t-SNE dimensionality reduction
from sklearn.metrics import (  # For performance metrics
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split  # For splitting the dataset and CV
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid  # For k-nearest neighbors model
from sklearn.neural_network import MLPClassifier  # For neural network model
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For label encoding and feature scaling
from sklearn.svm import SVC  # For Support Vector Machine model
from sklearn.tree import DecisionTreeClassifier  # For Decision Tree classifier model
from telegram_bot import TelegramBot, send_exception_via_telegram, send_telegram_message, setup_global_exception_hook  # For sending progress messages to Telegram
from threadpoolctl import threadpool_limits  # For narrowly limiting BLAS and OpenMP threads during feature extraction
from tqdm import tqdm  # For progress bars
from typing import Any, Callable, Optional, List, Tuple, cast  # For optional and collection typing hints
from xgboost import XGBClassifier  # For XGBoost classifier


# Macros:
class BackgroundColors:  # Colors for the terminal
    CYAN = "\033[96m"  # Cyan
    GREEN = "\033[92m"  # Green
    YELLOW = "\033[93m"  # Yellow
    RED = "\033[91m"  # Red
    BOLD = "\033[1m"  # Bold
    UNDERLINE = "\033[4m"  # Underline
    CLEAR_TERMINAL = "\033[H\033[J"  # Clear the terminal


# Global Configuration Container:
CONFIG = {}  # Will be initialized by initialize_config() - holds all runtime settings

# Telegram Bot Setup:
TELEGRAM_BOT = None  # Global Telegram bot instance (initialized in setup_telegram_bot)

# Logger Setup:
logger = None  # Will be initialized in initialize_logger()
EXPLAINABILITY_PROCESSES: List[dict] = []  # Tracks process-isolated explainability jobs until finalization.
EXPLAINABILITY_RAM_THRESHOLD_PERCENT = 50.0  # Defines the strict RAM ceiling for asynchronous explainability dispatch.
TRAINING_RAM_SAMPLE_INTERVAL_SECONDS = 1.0  # Defines the classifier training RAM polling interval.
EXPLAINABILITY_PROCESS_START_TIMEOUT_SECONDS = 10.0  # Defines the maximum wait for child initialization status.
EXPLAINABILITY_PROCESS_JOIN_TIMEOUT_SECONDS = 5.0  # Defines bounded join polling for live explainability children.
MEMORY_WATCHER_PROCESS: Optional[subprocess.Popen] = None  # Holds the single watcher sidecar process for this stacking run
MEMORY_WATCHER_RUN_DIR: Optional[str] = None  # Holds the unique watcher output directory for this stacking run
MEMORY_WATCHER_PHASE_STATE_PATH: Optional[str] = None  # Holds the atomic phase-state path shared with the watcher
MEMORY_WATCHER_EVENT_COUNTER = 0  # Counts emitted phase-state events for watcher ordering
MEMORY_WATCHER_TRACEMALLOC_PREVIOUS: Optional[tracemalloc.Snapshot] = None  # Holds previous tracemalloc snapshot for concise diffs
MEMORY_WATCHER_FINALIZED = False  # Prevents duplicate terminal watcher events


# Functions Definitions:


def get_memory_watcher_config(config: Optional[dict] = None) -> dict:  # Resolve memory watcher configuration
    """
    Resolve the memory watcher configuration with safe defaults.

    :param config: Runtime configuration dictionary.
    :return: Memory watcher configuration dictionary.
    """

    if config is None:  # Use global configuration when none is provided
        config = CONFIG  # Read global configuration
    default_cfg = {"enabled": False, "sample_interval_seconds": 2, "system_memory_threshold_percent": 90, "process_rss_threshold_gb": None, "capture_process_tree": True, "capture_tracemalloc": False, "tracemalloc_frame_depth": 25, "output_directory": "Logs/Memory_Watch", "keep_watcher_after_target_exit_seconds": 5}  # Define watcher defaults
    runtime_cfg = config.get("memory_watcher", {}) if isinstance(config, dict) else {}  # Read configured watcher section
    if isinstance(runtime_cfg, dict):  # Merge configured watcher values
        default_cfg.update(runtime_cfg)  # Apply runtime watcher overrides
    return default_cfg  # Return resolved watcher configuration


def memory_watcher_enabled(config: Optional[dict] = None) -> bool:  # Resolve watcher enabled flag
    """
    Return whether the memory watcher is enabled.

    :param config: Runtime configuration dictionary.
    :return: True when the watcher is enabled.
    """

    cfg = get_memory_watcher_config(config)  # Read resolved watcher configuration
    return bool(cfg.get("enabled", False))  # Return enabled state


def sanitize_memory_watcher_value(value: Any, depth: int = 0) -> Any:  # Normalize compact watcher metadata
    """
    Convert values into compact JSON-safe metadata for the watcher state file.

    :param value: Value to normalize.
    :param depth: Current recursion depth.
    :return: JSON-safe compact value.
    """

    if depth > 4:  # Bound nested metadata depth
        return str(value)  # Return string representation at depth limit
    if value is None or isinstance(value, (str, int, float, bool)):  # Preserve simple JSON scalars
        return value  # Return scalar unchanged
    if isinstance(value, Path):  # Normalize pathlib paths
        return str(value)  # Return path string
    if isinstance(value, np.generic):  # Normalize NumPy scalar values
        return value.item()  # Return Python scalar value
    if isinstance(value, (list, tuple, set)):  # Normalize bounded sequences
        return [sanitize_memory_watcher_value(item, depth + 1) for item in list(value)[:60]]  # Return compact list metadata
    if isinstance(value, dict):  # Normalize bounded mappings
        return {str(k): sanitize_memory_watcher_value(v, depth + 1) for k, v in list(value.items())[:80]}  # Return compact mapping metadata
    return str(value)  # Return safe string representation


def get_classifier_n_jobs(model: Any) -> Optional[Any]:  # Read estimator n_jobs setting when exposed
    """
    Read an estimator n_jobs parameter when it exists.

    :param model: Estimator object.
    :return: n_jobs value or None.
    """

    try:  # Read estimator parameters safely
        if hasattr(model, "get_params"):  # Use sklearn-style parameter access when available
            params = model.get_params(deep=False)  # Read shallow estimator parameters
            if isinstance(params, dict) and "n_jobs" in params:  # Detect exposed n_jobs parameter
                return params.get("n_jobs")  # Return n_jobs parameter
    except Exception:  # Keep metadata best-effort
        return None  # Return unavailable n_jobs value
    return None  # Return unavailable n_jobs value


def get_classifier_params_digest(model: Any) -> dict:  # Build stable classifier parameter digest
    """
    Build a stable digest for estimator parameters without storing models or arrays.

    :param model: Estimator object.
    :return: Digest metadata dictionary.
    """

    try:  # Read and normalize estimator parameters
        params = model.get_params(deep=False) if hasattr(model, "get_params") else {}  # Read shallow parameters when available
        normalized = sanitize_memory_watcher_value(params)  # Normalize parameter values for JSON
        encoded = json.dumps(normalized, sort_keys=True, default=str)  # Serialize normalized parameters
        digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()  # Compute stable parameter digest
        return {"digest": digest, "parameter_count": len(params) if isinstance(params, dict) else None}  # Return compact digest metadata
    except Exception as exc:  # Keep diagnostics from interrupting training
        digest = hashlib.sha256(str(model).encode("utf-8", errors="replace")).hexdigest()  # Compute fallback digest from model string
        return {"digest": digest, "parameter_count": None, "error": str(exc)}  # Return fallback digest metadata


def read_system_ram_percent() -> float:
    """
    Read current system RAM usage percentage.

    :return: Current system RAM usage percentage.
    """

    return float(psutil.virtual_memory().percent)  # Return the current system RAM usage percentage from psutil.


def sample_training_ram_usage(stop_event: threading.Event, samples: List[float], interval_seconds: float) -> None:
    """
    Sample system RAM usage until classifier training stops.

    :param stop_event: Event used to stop RAM sampling.
    :param samples: Mutable list receiving RAM usage percentages.
    :param interval_seconds: Sampling interval in seconds.
    :return: None.
    """

    while not stop_event.wait(interval_seconds):  # Continue sampling until the caller stops training monitoring.
        samples.append(read_system_ram_percent())  # Store the current system RAM usage percentage for this classifier.


def start_training_ram_monitor(interval_seconds: float) -> dict:
    """
    Start low-overhead RAM sampling for one classifier training period.

    :param interval_seconds: Sampling interval in seconds.
    :return: Monitor state dictionary used to stop sampling.
    """

    samples: List[float] = [read_system_ram_percent()]  # Record an immediate RAM sample before classifier fit starts.
    stop_event = threading.Event()  # Create a stop event owned by this classifier training monitor.
    safe_interval = max(float(interval_seconds), 0.1)  # Bound the interval away from busy waiting.
    thread = threading.Thread(target=sample_training_ram_usage, args=(stop_event, samples, safe_interval), name="training-ram-monitor", daemon=True)  # Create one daemon sampler thread for this classifier fit.
    thread.start()  # Start RAM sampling immediately before classifier fit.
    return {"stop_event": stop_event, "samples": samples, "thread": thread, "interval_seconds": safe_interval}  # Return monitor state to the caller.


def stop_training_ram_monitor(monitor: Optional[dict]) -> dict:
    """
    Stop RAM sampling and summarize one classifier training period.

    :param monitor: Monitor state dictionary returned by start_training_ram_monitor.
    :return: Dictionary containing latest, average, sample count, and thread state.
    """

    if monitor is None:  # Preserve callers that did not start monitoring.
        current_percent = read_system_ram_percent()  # Read a fallback RAM value when no monitor exists.
        return {"latest_percent": current_percent, "average_percent": current_percent, "sample_count": 1, "thread_alive": False}  # Return a one-sample fallback summary.
    samples = monitor.get("samples", [])  # Read samples collected for this classifier.
    samples.append(read_system_ram_percent())  # Record the terminal RAM sample immediately after classifier fit exits.
    stop_event = cast(Any, monitor.get("stop_event"))  # Read the stop event from monitor state.
    thread = monitor.get("thread")  # Read the sampler thread from monitor state.
    interval_seconds = float(monitor.get("interval_seconds", TRAINING_RAM_SAMPLE_INTERVAL_SECONDS))  # Read the effective sampling interval.
    if stop_event is not None:  # Verify that the monitor stop event is present.
        stop_event.set()  # Signal the sampler thread to stop.
    if isinstance(thread, threading.Thread):  # Verify that the monitor thread is valid.
        thread.join(timeout=max(interval_seconds + 1.0, 1.0))  # Wait briefly for sampler cleanup without delaying training flow indefinitely.
    numeric_samples = [float(value) for value in samples if isinstance(value, (int, float)) and not math.isnan(float(value))]  # Keep numeric RAM samples only.
    if not numeric_samples:  # Use a fallback sample if all collected values were invalid.
        numeric_samples = [read_system_ram_percent()]  # Read current RAM usage as the fallback sample.
    latest_percent = float(numeric_samples[-1])  # Resolve latest observed RAM usage for this classifier.
    average_percent = float(sum(numeric_samples) / len(numeric_samples))  # Compute arithmetic mean RAM usage for this classifier training period.
    thread_alive = bool(isinstance(thread, threading.Thread) and thread.is_alive())  # Record whether sampler cleanup completed.
    return {"latest_percent": latest_percent, "average_percent": average_percent, "sample_count": len(numeric_samples), "thread_alive": thread_alive}  # Return classifier RAM summary.


def store_training_ram_stats(stats_holder: Optional[dict], stats: dict) -> None:
    """
    Store RAM statistics in the caller-owned classifier state holder.

    :param stats_holder: Mutable state holder supplied by the classifier loop.
    :param stats: RAM statistics produced by stop_training_ram_monitor.
    :return: None.
    """

    if stats_holder is not None:  # Preserve optional caller state ownership.
        stats_holder.clear()  # Remove stale classifier RAM values before storing this classifier's values.
        stats_holder.update(stats)  # Store the RAM statistics for the matching classifier.


def resolve_training_ram_percent(stats: Optional[dict], key: str) -> Optional[float]:
    """
    Resolve one RAM percentage value from classifier training statistics.

    :param stats: RAM statistics dictionary for one classifier.
    :param key: Statistic key to read.
    :return: RAM percentage value, or None when unavailable.
    """

    if not isinstance(stats, dict):  # Treat missing or invalid statistics as unavailable.
        return None  # Return unavailable percentage.
    value = stats.get(key)  # Read the requested RAM statistic.
    if value is None:  # Treat missing values as unavailable.
        return None  # Return unavailable percentage.
    return float(value)  # Return the RAM percentage as a float.


def format_ram_percent(value: Optional[float]) -> str:
    """
    Format a RAM percentage value for logs.

    :param value: RAM percentage value.
    :return: Formatted RAM percentage string.
    """

    if value is None:  # Render unavailable values explicitly.
        return "unavailable"  # Return the unavailable marker.
    return f"{float(value):.2f}%"  # Return RAM usage formatted with two decimal places.


def build_memory_phase_metadata(config: Optional[dict] = None, **metadata: Any) -> dict:  # Build compact phase-state metadata
    """
    Build a compact phase-state record for the watcher.

    :param config: Runtime configuration dictionary.
    :param metadata: Phase-specific metadata values.
    :return: Phase-state metadata dictionary.
    """

    cfg = config if isinstance(config, dict) else CONFIG  # Resolve runtime configuration
    watcher_cfg = get_memory_watcher_config(cfg)  # Read watcher configuration
    event = {"run_id": watcher_cfg.get("run_id"), "main_pid": os.getpid(), "execution_mode": cfg.get("execution", {}).get("execution_mode") if isinstance(cfg, dict) else None, "watcher_run_dir": watcher_cfg.get("run_directory")}  # Build shared metadata base
    try:  # Add process/system memory snapshot only when watcher events are already being emitted
        process_info = psutil.Process(os.getpid()).memory_info()  # Read current process memory counters
        virtual_memory = psutil.virtual_memory()  # Read system RAM counters
        swap_memory = psutil.swap_memory()  # Read system swap counters
        event["process_rss_gb"] = round(process_info.rss / (1024 ** 3), 4)  # Store resident set size in GiB
        event["process_vms_gb"] = round(process_info.vms / (1024 ** 3), 4)  # Store virtual memory size in GiB
        event["system_available_gb"] = round(virtual_memory.available / (1024 ** 3), 4)  # Store currently available system RAM
        event["system_memory_percent"] = virtual_memory.percent  # Store system memory pressure percentage
        event["swap_used_gb"] = round(swap_memory.used / (1024 ** 3), 4)  # Store current swap usage in GiB
        event["swap_percent"] = swap_memory.percent  # Store swap pressure percentage
    except Exception:  # Keep watcher metadata best-effort
        pass  # Do not let diagnostics interrupt model execution
    for key, value in metadata.items():  # Merge caller metadata
        event[key] = sanitize_memory_watcher_value(value)  # Store normalized metadata value
    return event  # Return compact phase-state metadata


def write_memory_tracemalloc_report(phase: str, event_id: int, config: Optional[dict] = None) -> None:  # Write optional tracemalloc report
    """
    Write a concise tracemalloc report for selected phase boundaries.

    :param phase: Phase name.
    :param event_id: Phase event sequence number.
    :param config: Runtime configuration dictionary.
    :return: None.
    """

    global MEMORY_WATCHER_TRACEMALLOC_PREVIOUS  # Update previous snapshot for diff reports
    try:  # Keep tracemalloc failures diagnostic-only
        cfg = get_memory_watcher_config(config)  # Read watcher configuration
        phases = {"startup", "watcher_started", "before_classifier_fit", "after_classifier_fit", "after_prediction_and_metrics", "before_cache_persist", "after_cache_persist", "before_explainability_schedule", "after_explainability_schedule", "before_memory_cleanup", "after_memory_cleanup", "memory_error", "model_error", "final_export", "normal_completion", "abnormal_completion"}  # Define snapshot phase boundaries
        if not cfg.get("capture_tracemalloc", False):  # Skip when tracemalloc mode is disabled
            return  # Leave without report
        if phase not in phases:  # Limit snapshots to selected phase boundaries
            return  # Leave without report
        if not tracemalloc.is_tracing():  # Skip when tracing was not started
            return  # Leave without report
        run_dir = cfg.get("run_directory") or MEMORY_WATCHER_RUN_DIR  # Resolve watcher run directory
        if not run_dir:  # Skip when no output directory is available
            return  # Leave without report
        snapshot = tracemalloc.take_snapshot()  # Capture Python allocation snapshot
        safe_phase = re.sub(r"[^A-Za-z0-9_.-]+", "_", phase)  # Normalize phase for filename
        report_path = os.path.join(run_dir, f"tracemalloc_{event_id:06d}_{safe_phase}.txt")  # Resolve report path
        previous_snapshot = MEMORY_WATCHER_TRACEMALLOC_PREVIOUS  # Read previous snapshot
        stats = snapshot.compare_to(previous_snapshot, "lineno")[:20] if previous_snapshot is not None else snapshot.statistics("lineno")[:20]  # Build concise allocation stats
        with open(report_path, "w", encoding="utf-8") as file_obj:  # Write report file
            file_obj.write(f"phase={phase}\nevent_id={event_id}\ntimestamp={datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")  # Write report header
            file_obj.write("Limitation: tracemalloc primarily explains Python-level allocations and may not explain native NumPy or scikit-learn allocations.\n")  # Write tracemalloc limitation
            for stat in stats:  # Write top allocation rows
                file_obj.write(f"{stat}\n")  # Write allocation stat row
        MEMORY_WATCHER_TRACEMALLOC_PREVIOUS = snapshot  # Store snapshot for next diff
    except Exception as exc:  # Keep training alive if tracemalloc fails
        try:  # Best-effort warning output
            print(f"{BackgroundColors.YELLOW}[WARNING] Tracemalloc report failed for phase {phase}: {exc}{Style.RESET_ALL}")  # Log tracemalloc failure
        except Exception:  # Ignore logging failure
            pass  # Continue training


def write_memory_phase_event(phase: str, config: Optional[dict] = None, **metadata: Any) -> None:  # Write atomic phase-state event
    """
    Write the latest compact phase-state JSON atomically for the watcher process.

    :param phase: Phase name.
    :param config: Runtime configuration dictionary.
    :param metadata: Compact phase metadata.
    :return: None.
    """

    global MEMORY_WATCHER_EVENT_COUNTER  # Increment shared phase event counter
    try:  # Keep diagnostics from interrupting model execution
        if not memory_watcher_enabled(config):  # Skip phase writes when watcher is disabled
            return  # Leave without diagnostics
        state_path = MEMORY_WATCHER_PHASE_STATE_PATH  # Read shared phase-state path
        if not state_path:  # Skip until watcher startup resolves the state path
            return  # Leave without diagnostics
        MEMORY_WATCHER_EVENT_COUNTER += 1  # Increment monotonic event counter
        event_id = MEMORY_WATCHER_EVENT_COUNTER  # Capture current event id
        event = build_memory_phase_metadata(config, **metadata)  # Build compact metadata payload
        event["event_id"] = event_id  # Store phase event id
        event["timestamp"] = datetime.datetime.now(datetime.timezone.utc).isoformat()  # Store phase timestamp
        event["phase"] = phase  # Store phase name
        tmp_path = f"{state_path}.{os.getpid()}.{event_id}.tmp"  # Build same-directory temporary path
        with open(tmp_path, "w", encoding="utf-8") as file_obj:  # Write temporary phase state
            json.dump(sanitize_memory_watcher_value(event), file_obj, sort_keys=True)  # Serialize compact phase state
            file_obj.write("\n")  # Terminate JSON file
            file_obj.flush()  # Flush Python buffer
            os.fsync(file_obj.fileno())  # Flush phase-state data to disk
        os.replace(tmp_path, state_path)  # Atomically publish latest phase state
        phase_events_path = os.path.join(os.path.dirname(state_path), "phase_events.jsonl")  # Resolve phase events JSONL path
        with open(phase_events_path, "a", encoding="utf-8") as events_file:  # Append an ordered main-process phase event
            events_file.write(json.dumps({"timestamp": event["timestamp"], "event_type": "phase_event", "phase_metadata": sanitize_memory_watcher_value(event)}, sort_keys=True) + "\n")  # Write compact phase event row
            events_file.flush()  # Flush phase event row promptly
        write_memory_tracemalloc_report(phase, event_id, config=config)  # Write optional tracemalloc report
    except Exception as exc:  # Swallow watcher write failures
        try:  # Best-effort warning output
            print(f"{BackgroundColors.YELLOW}[WARNING] Memory watcher phase write failed for phase {phase}: {exc}{Style.RESET_ALL}")  # Log phase write failure
        except Exception:  # Ignore logging failure
            pass  # Continue training


def create_memory_watcher_run_directory(config: Optional[dict] = None) -> Optional[str]:  # Create unique watcher run directory
    """
    Create a unique watcher output directory.

    :param config: Runtime configuration dictionary.
    :return: Absolute run directory path or None.
    """

    try:  # Keep directory creation best-effort
        cfg = get_memory_watcher_config(config)  # Read watcher configuration
        project_root = Path(__file__).resolve().parent  # Resolve repository root for this script
        output_directory = cfg.get("output_directory", "Logs/Memory_Watch")  # Read configured output directory
        base_dir = Path(output_directory) if os.path.isabs(str(output_directory)) else project_root / str(output_directory)  # Resolve project-relative output directory
        base_dir.mkdir(parents=True, exist_ok=True)  # Ensure base output directory exists
        for _ in range(100):  # Try bounded collision-resistant names
            run_id = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{uuid.uuid4().hex[:8]}"  # Build unique run identifier
            run_dir = base_dir / run_id  # Resolve run directory
            try:  # Attempt exclusive directory creation
                run_dir.mkdir(parents=False, exist_ok=False)  # Create run directory without overwrite
                if isinstance(config, dict):  # Store resolved watcher paths in runtime config
                    config.setdefault("memory_watcher", {})["run_id"] = run_id  # Store watcher run id
                    config.setdefault("memory_watcher", {})["run_directory"] = str(run_dir)  # Store watcher run directory
                    config.setdefault("memory_watcher", {})["phase_state_path"] = str(run_dir / "phase_state.json")  # Store phase-state file path
                return str(run_dir)  # Return unique run directory
            except FileExistsError:  # Retry on rare name collision
                continue  # Generate another run id
        raise RuntimeError("Unable to create a unique memory watcher run directory")  # Raise after bounded attempts
    except Exception as exc:  # Keep pipeline alive when diagnostics cannot start
        print(f"{BackgroundColors.YELLOW}[WARNING] Failed to create memory watcher directory: {exc}{Style.RESET_ALL}")  # Log directory creation failure
        return None  # Signal unavailable watcher directory


def start_memory_watcher(config: Optional[dict] = None) -> Optional[subprocess.Popen]:  # Start one watcher sidecar for this run
    """
    Start exactly one independent memory watcher process for this top-level run.

    :param config: Runtime configuration dictionary.
    :return: Watcher process object or None.
    """

    global MEMORY_WATCHER_PROCESS, MEMORY_WATCHER_RUN_DIR, MEMORY_WATCHER_PHASE_STATE_PATH, MEMORY_WATCHER_FINALIZED, MEMORY_WATCHER_TRACEMALLOC_PREVIOUS  # Update watcher lifecycle globals
    try:  # Keep watcher startup best-effort
        if not memory_watcher_enabled(config):  # Skip startup when watcher is disabled
            return None  # Return without watcher
        if MEMORY_WATCHER_PROCESS is not None:  # Prevent duplicate watcher sidecars
            return MEMORY_WATCHER_PROCESS  # Return existing watcher process
        cfg = get_memory_watcher_config(config)  # Read watcher configuration
        run_dir = create_memory_watcher_run_directory(config)  # Create unique watcher run directory
        if not run_dir:  # Abort watcher startup when directory creation failed
            return None  # Return without watcher
        phase_state_path = get_memory_watcher_config(config).get("phase_state_path") or os.path.join(run_dir, "phase_state.json")  # Resolve phase-state file path
        MEMORY_WATCHER_RUN_DIR = run_dir  # Store watcher run directory globally
        MEMORY_WATCHER_PHASE_STATE_PATH = phase_state_path  # Store phase-state path globally
        MEMORY_WATCHER_FINALIZED = False  # Reset terminal event state for this run
        MEMORY_WATCHER_TRACEMALLOC_PREVIOUS = None  # Reset tracemalloc diff state for this run
        if cfg.get("capture_tracemalloc", False):  # Start tracemalloc only when explicitly enabled
            try:  # Keep tracemalloc startup best-effort
                frame_depth = int(cfg.get("tracemalloc_frame_depth", 25) or 25)  # Resolve frame depth
                if not tracemalloc.is_tracing():  # Start tracing only once
                    tracemalloc.start(frame_depth)  # Start Python allocation tracing
            except Exception as exc:  # Preserve training when tracemalloc cannot start
                print(f"{BackgroundColors.YELLOW}[WARNING] Failed to start tracemalloc: {exc}{Style.RESET_ALL}")  # Log tracemalloc startup failure
        write_memory_phase_event("startup", config=config, event_outcome="started")  # Publish startup phase before watcher launch
        watcher_path = Path(__file__).resolve().parent / "Scripts" / "memory_watcher.py"  # Resolve authoritative watcher script path
        if not watcher_path.exists():  # Abort when watcher script is missing
            print(f"{BackgroundColors.YELLOW}[WARNING] Memory watcher script not found: {watcher_path}{Style.RESET_ALL}")  # Log missing watcher script
            return None  # Return without watcher
        target_process = psutil.Process(os.getpid())  # Resolve current process identity
        target_create_time = float(target_process.create_time())  # Read target creation timestamp
        command = [sys.executable, str(watcher_path), "--target-pid", str(os.getpid()), "--target-create-time", str(target_create_time), "--run-dir", run_dir, "--phase-state-path", phase_state_path, "--sample-interval-seconds", str(cfg.get("sample_interval_seconds", 2)), "--system-memory-threshold-percent", str(cfg.get("system_memory_threshold_percent", 90)), "--keep-after-target-exit-seconds", str(cfg.get("keep_watcher_after_target_exit_seconds", 5))]  # Build watcher command without target script name
        if cfg.get("process_rss_threshold_gb") is not None:  # Add optional process RSS threshold
            command.extend(["--process-rss-threshold-gb", str(cfg.get("process_rss_threshold_gb"))])  # Add process RSS threshold arguments
        if cfg.get("capture_process_tree", True):  # Add process-tree capture flag
            command.append("--capture-process-tree")  # Enable recursive child process rows
        if cfg.get("capture_tracemalloc", False):  # Add tracemalloc mode flag for summary parity
            command.append("--capture-tracemalloc")  # Mark tracemalloc mode in watcher summary
        popen_kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL, "stdin": subprocess.DEVNULL, "cwd": str(Path(__file__).resolve().parent), "close_fds": True}  # Build detached process settings
        windows_fallback_creationflags = None  # Track fallback Windows flags when breakaway is rejected
        if platform.system() == "Windows":  # Use Windows process-group flags when available
            windows_base_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)  # Build baseline detached Windows flags
            windows_breakaway_flag = getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)  # Read optional job-breakaway flag
            popen_kwargs["creationflags"] = windows_base_flags | windows_breakaway_flag  # Request detached Windows watcher process with job breakaway when available
            windows_fallback_creationflags = windows_base_flags  # Store fallback flags without job breakaway
        else:  # Use POSIX session isolation when available
            popen_kwargs["start_new_session"] = True  # Request separate session for watcher process
        try:  # Launch watcher with preferred detachment settings
            MEMORY_WATCHER_PROCESS = subprocess.Popen(command, **popen_kwargs)  # Launch watcher without waiting
        except OSError:  # Retry Windows launch if job breakaway is unavailable
            if platform.system() == "Windows" and windows_fallback_creationflags is not None:  # Use fallback Windows flags only on Windows
                popen_kwargs["creationflags"] = windows_fallback_creationflags  # Remove job-breakaway flag for fallback launch
                MEMORY_WATCHER_PROCESS = subprocess.Popen(command, **popen_kwargs)  # Launch watcher with baseline detached flags
            else:  # Re-raise non-Windows launch failures
                raise  # Preserve original startup failure behavior
        write_memory_phase_event("watcher_started", config=config, watcher_pid=MEMORY_WATCHER_PROCESS.pid, watcher_command_program=os.path.basename(str(watcher_path)), watcher_arguments_contain_target_script=any("stacking.py" in str(part) for part in command), event_outcome="started")  # Publish watcher-started phase
        print(f"{BackgroundColors.GREEN}[DEBUG] Memory watcher enabled. Output directory: {BackgroundColors.CYAN}{run_dir}{Style.RESET_ALL}")  # Log watcher output path
        return MEMORY_WATCHER_PROCESS  # Return watcher process object
    except Exception as exc:  # Preserve model execution on watcher startup failure
        print(f"{BackgroundColors.YELLOW}[WARNING] Failed to start memory watcher: {exc}{Style.RESET_ALL}")  # Log watcher startup failure
        return None  # Return without watcher


def finalize_memory_watcher(config: Optional[dict] = None, phase: str = "normal_completion", **metadata: Any) -> None:  # Publish terminal watcher phase
    """
    Publish terminal watcher phase metadata without waiting for the watcher sidecar.

    :param config: Runtime configuration dictionary.
    :param phase: Terminal phase name.
    :param metadata: Additional terminal metadata.
    :return: None.
    """

    global MEMORY_WATCHER_FINALIZED  # Prevent duplicate final events
    try:  # Keep final diagnostics best-effort
        if not memory_watcher_enabled(config):  # Skip when watcher is disabled
            return  # Leave without diagnostics
        if MEMORY_WATCHER_FINALIZED:  # Avoid duplicate terminal phase writes
            return  # Leave without duplicate event
        write_memory_phase_event(phase, config=config, **metadata)  # Publish terminal phase state
        MEMORY_WATCHER_FINALIZED = True  # Mark terminal phase written
    except Exception:  # Never fail main cleanup due to watcher finalization
        pass  # Continue process shutdown


def resolve_path_represents_directory(dataset_path: str) -> bool:
    """
    Return whether a dataset path represents a directory.

    :param dataset_path: Dataset path to classify.
    :return: True when the path represents a directory.
    """

    path_text = str(dataset_path).strip()  # Normalize incoming path text.
    normalized_text = path_text.replace("\\", "/")  # Normalize separators for suffix tests.
    if normalized_text.endswith("/"):  # Treat trailing separators as directory intent.
        return True  # Return directory intent.
    path_obj = Path(path_text)  # Build path object for filesystem metadata.
    if path_obj.exists():  # Use filesystem metadata when available.
        return path_obj.is_dir()  # Return directory status from the filesystem.
    return Path(path_text).suffix == ""  # Infer directory intent for extensionless paths.


def resolve_dataset_root_path(dataset_path: str) -> Path:
    """
    Resolve dataset root directory from a file or directory path.

    :param dataset_path: Dataset file or directory path.
    :return: Dataset root directory path.
    """

    path_obj = Path(str(dataset_path).strip())  # Normalize incoming path text into a path object.
    if resolve_path_represents_directory(str(path_obj)):  # Use the directory itself for directory identities.
        return path_obj.resolve()  # Return resolved dataset directory.
    return path_obj.resolve().parent  # Return resolved parent directory for file identities.


def resolve_canonical_dataset_identity(dataset_path: str, is_directory: bool) -> str:
    """
    Resolve canonical dataset identity path.

    :param dataset_path: Dataset path to normalize.
    :param is_directory: Whether dataset_path identifies a directory.
    :return: Canonical dataset identity path.
    """

    path_text = str(dataset_path).strip()  # Normalize incoming path text.
    if not path_text:  # Reject empty dataset identity inputs.
        raise ValueError("Dataset identity path cannot be empty")  # Raise explicit identity failure.
    normalized_path = os.path.normpath(os.path.expanduser(path_text))  # Normalize local path structure.
    absolute_path = os.path.abspath(normalized_path)  # Resolve absolute path for relative conversion.
    try:  # Convert to project-relative identity when possible.
        identity_path = os.path.relpath(absolute_path)  # Preserve existing relative result convention.
    except ValueError:  # Fall back for paths on different Windows drives.
        identity_path = normalized_path  # Preserve normalized input when relative conversion fails.
    identity_path = identity_path.replace("\\", "/")  # Normalize separators to project CSV convention.
    while identity_path.startswith("./"):  # Remove duplicate leading current-directory markers.
        identity_path = identity_path[2:]  # Trim one leading current-directory marker.
    identity_path = re.sub(r"/+", "/", identity_path).rstrip("/")  # Collapse duplicate separators and trim trailing separators.
    if is_directory:  # Preserve directory identity with one trailing separator.
        identity_path = f"{identity_path}/"  # Append canonical directory separator.
    return identity_path  # Return canonical dataset identity.


def build_filename_safe_dataset_identity(dataset_identity: str) -> str:
    """
    Build filename-safe text from a dataset identity.

    :param dataset_identity: Canonical dataset identity path.
    :return: Filename-safe dataset identity text.
    """

    identity_text = str(dataset_identity).strip().replace("\\", "/")  # Normalize identity text and separators.
    while identity_text.startswith("./"):  # Remove duplicate leading current-directory markers.
        identity_text = identity_text[2:]  # Trim one leading current-directory marker.
    identity_text = re.sub(r"/+", "/", identity_text).strip("/")  # Collapse duplicate separators and trim edges.
    filename_identity = re.sub(r"[^A-Za-z0-9_]+", "_", identity_text)  # Convert path and delimiter characters to underscores.
    filename_identity = re.sub(r"_+", "_", filename_identity).strip("_")  # Collapse duplicate underscores after sanitization.
    if not filename_identity:  # Reject empty filename identity outputs.
        raise ValueError("Dataset filename identity cannot be empty")  # Raise explicit filename identity failure.
    return filename_identity  # Return filename-safe identity.


def resolve_combined_files_dataset_identity(original_files_list: List[str]) -> str:
    """
    Resolve combined-files dataset directory identity.

    :param original_files_list: Source file paths used by combined-files mode.
    :return: Canonical combined dataset directory identity.
    """

    if not original_files_list:  # Require source files to derive a combined directory identity.
        raise ValueError("Combined files dataset identity requires at least one source file")  # Raise explicit combined identity failure.
    source_directories = [os.path.dirname(os.path.abspath(str(file_path))) for file_path in original_files_list]  # Resolve source directories without changing file order.
    combined_directory = os.path.commonpath(source_directories)  # Resolve the shared dataset directory from active source files.
    return resolve_canonical_dataset_identity(combined_directory, True)  # Return canonical directory identity.


def get_stacking_output_dir(dataset_file_path: str, config: dict) -> str:
    """
    Get the output directory for stacking results based on the dataset file or directory path and configuration.
    
    :param dataset_file_path: The path to the dataset file or directory being processed
    :param config: The configuration dictionary containing the stacking results directory setting
    :return: The resolved path to the stacking results directory for the given dataset
    """

    if not isinstance(dataset_file_path, (str, Path)):
        raise ValueError("dataset_file_path must be a path string or Path")
    if not isinstance(config, dict):
        raise ValueError("config must be a dict")

    dataset_root = resolve_dataset_root_path(str(dataset_file_path))  # Resolve dataset root from file or directory input.

    stacking_cfg = config.get("stacking", {})
    results_dir = stacking_cfg.get("results_dir")
    if not isinstance(results_dir, str) or results_dir.strip() == "":
        raise ValueError("Missing or invalid config['stacking']['results_dir']")
    if os.path.isabs(results_dir):
        raise ValueError("config['stacking']['results_dir'] must be relative (not absolute)")

    stacking_dir = dataset_root / results_dir
    stacking_dir.mkdir(parents=True, exist_ok=True)

    try:
        common = os.path.commonpath([str(dataset_root.resolve()), str(stacking_dir.resolve())])
    except Exception:
        raise RuntimeError("Invalid paths provided for stacking directory validation")
    if common != str(dataset_root.resolve()):
        raise RuntimeError("Resolved stacking directory is not inside the dataset root")

    return str(stacking_dir.resolve())


def validate_output_path(base_dir: str, target_path: str) -> None:
    """
    Validate that the target_path is safely within the base_dir to prevent directory traversal issues.

    :param base_dir: The base directory that should contain the target path
    :param target_path: The target path to validate
    :return: None
    """
    
    if base_dir is None:
        raise RuntimeError("base_dir is None")
    
    base_dir = os.path.abspath(base_dir)
    target_path = os.path.abspath(target_path)
    
    try:
        common = os.path.commonpath([base_dir, target_path])
    except Exception:
        raise RuntimeError("Invalid paths provided for validation")
    
    if common != base_dir:
        raise RuntimeError("Unsafe path detected outside stacking output directory")


def parse_cli_args():
    """
    Parse command-line arguments for stacking pipeline.
    
    :return: Namespace object containing parsed arguments
    """

    try:
        parser = argparse.ArgumentParser(
            description="Run stacking classifier evaluation pipeline with optional AutoML and data augmentation testing."
        )  # Create argument parser
        parser.add_argument("--config", type=str, default=None, help="Path to config.yaml file")
        parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
        parser.add_argument("--skip-train-if-model-exists", dest="skip_train", action="store_true", help="Load existing models instead of retraining")
        parser.add_argument("--csv", type=str, default=None, help="Path to specific CSV file to process")
        parser.add_argument("--dataset-path", type=str, default=None, help="Path to a dataset directory or single CSV file to process")
        parser.add_argument("--enable-automl", dest="enable_automl", action="store_true", default=None, help="Enable AutoML pipeline method toggle")  # Enable AutoML via CLI
        parser.add_argument("--disable-automl", dest="enable_automl", action="store_false", help="Disable AutoML pipeline method toggle")  # Disable AutoML via CLI
        parser.add_argument("--automl-trials", type=int, default=None, help="Number of AutoML trials")
        parser.add_argument("--automl-stacking-trials", type=int, default=None, help="Number of stacking trials")
        parser.add_argument("--automl-timeout", type=int, default=None, help="AutoML timeout in seconds")
        parser.add_argument("--test-augmentation", action="store_true", help="Enable data augmentation testing")
        parser.add_argument("--no-test-augmentation", dest="test_augmentation", action="store_false", help="Disable data augmentation testing")
        parser.add_argument("--combined-files", dest="combined_files_flag", action="store_true", default=None, help="Enable combined files evaluation mode (directory-based batch processing)")  # Sets combined files evaluation mode
        parser.add_argument("--separate-files", dest="combined_files_flag", action="store_false", help="Enable separate files evaluation mode (single-file processing)")  # Sets separate files evaluation mode
        parser.add_argument("--both", action="store_true", help="Run both separate_files and combined_files pipelines sequentially")  # Runs both modes in sequence
        parser.add_argument("--stacking-results-dir", type=str, default=None, help="Directory to save stacking results (relative to dataset root)")
        parser.add_argument("--top-n-features", dest="top_n_features", type=int, default=None, help="Number of top features to show in heatmap (overrides config)")
        parser.add_argument("--enable-augmentation", dest="enable_augmentation", action="store_true", default=None, help="Enable data augmentation method toggle")
        parser.add_argument("--disable-augmentation", dest="enable_augmentation", action="store_false", help="Disable data augmentation method toggle")
        parser.add_argument("--enable-feature-selection", dest="enable_feature_selection", action="store_true", default=None, help="Enable feature selection method toggle")
        parser.add_argument("--disable-feature-selection", dest="enable_feature_selection", action="store_false", help="Disable feature selection method toggle")
        parser.add_argument("--enable-hyperparameters", dest="enable_hyperparameters", action="store_true", default=None, help="Enable hyperparameter optimization method toggle")
        parser.add_argument("--disable-hyperparameters", dest="enable_hyperparameters", action="store_false", help="Disable hyperparameter optimization method toggle")
        parser.add_argument("--enable-stacking", dest="enable_stacking", action="store_true", default=None, help="Enable stacking classifier evaluation")
        parser.add_argument("--disable-stacking", dest="enable_stacking", action="store_false", help="Disable stacking classifier evaluation")
        parser.add_argument("--n-jobs", dest="n_jobs", type=int, default=None, help="Override evaluation.n_jobs for estimators that support parallel fitting (-1 uses all processors; 1 is memory-safe)",)
        parser.add_argument("--feature-extraction-n-jobs", dest="feature_extraction_n_jobs", type=int, default=None, help="Override evaluation.feature_extraction_n_jobs for feature extraction/transformation stages such as PCA, not classifier training (-1 uses available CPUs; 1 is memory-safe)")  # Add the independent feature extraction thread override
        parser.add_argument("--low-memory", dest="low_memory", action="store_true", default=False, help="Enable low memory mode for pandas operations")  # Add low memory mode CLI argument
        parser.add_argument("--dataset-file-format", type=str, default=None, dest="dataset_file_format", help="File format for dataset files: arff, csv, parquet, txt")  # Dataset file format CLI override
        parser.add_argument("--augmentation-file-format", type=str, default=None, dest="augmentation_file_format", help="File format for augmentation files: arff, csv, parquet, txt")  # Augmentation file format CLI override
        parser.add_argument("--feature-sets", type=str, default=None, dest="feature_sets", help="Comma-separated feature set strategies to enable: full,pca,rfe,ga (overrides config toggles)")  # Feature set strategies CLI override
        parser.add_argument("--features", type=str, default=None, dest="explicit_features", help="Comma-separated explicit feature names")  # Explicit feature list CLI override
        parser.add_argument("--enable-memory-watcher", dest="enable_memory_watcher", action="store_true", default=None, help="Enable low-overhead memory watcher diagnostics")  # Enable watcher diagnostics via CLI
        parser.add_argument("--disable-memory-watcher", dest="enable_memory_watcher", action="store_false", help="Disable memory watcher diagnostics")  # Disable watcher diagnostics via CLI
        parser.add_argument("--memory-watch-interval-seconds", type=float, default=None, dest="memory_watch_interval_seconds", help="Memory watcher sampling interval in seconds")  # Override watcher sampling interval
        parser.add_argument("--memory-watch-threshold-percent", type=float, default=None, dest="memory_watch_threshold_percent", help="System memory pressure threshold percent")  # Override watcher system threshold
        parser.add_argument("--enable-memory-tracemalloc", dest="enable_memory_tracemalloc", action="store_true", default=False, help="Enable optional tracemalloc reports at selected phase boundaries")  # Enable optional Python allocation reports
        
        return parser.parse_args()  # Return parsed arguments
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def get_default_stacking_config():
    """
    Return default stacking pipeline configuration section.

    :return: Dictionary containing default stacking configuration values
    """

    try:
        return {
            "results_dir": "Stacking",  # Output subdirectory name for stacking results
            "results_filename": "Stacking_Classifiers_Results.csv",  # Separate files evaluation results CSV filename
            "cache_results_subdir": "Cache_Results",  # Cache results subdirectory inside the stacking results directory
            "combined_files_results_filename": "Stacking_Classifiers_CombinedFiles_Results.csv",  # Combined files evaluation results CSV filename
            "augmentation_comparison_filename": "Data_Augmentation_Comparison_Results.csv",  # Augmentation comparison CSV filename
            "data_augmentation_suffix": "_data_augmented",  # File suffix for augmented data files
            "augmentation_ratios": [0.25, 0.50, 0.75, 1.00],  # Ratios of augmented data to sample
            "hyperparameters_filename": "Hyperparameter_Optimization_Results.csv",  # Hyperparameter results CSV filename
            "cache_prefix": "Cache_",  # Prefix for cached model files
            "model_export_base": "Feature_Analysis/Stacking/Models/",  # Base directory for model exports
            "results_csv_columns": [
                "experiment_id", "experiment_mode", "execution_mode", "data_source",
                "dataset", "attack_types_combined", "augmentation_ratio",
                "feature_selection_enabled", "hyperparameters_enabled", "data_augmentation_enabled", "hyperparameter_mode",
                "feature_set", "classifier_type", "model_name", "model",
                "n_features", "n_samples_train", "n_samples_test",
                "accuracy", "precision", "recall", "f1_score", "fpr", "fnr", "elapsed_time_s",
                "cv_method", "top_features", "rfe_ranking", "hyperparameters", "features_list", "Hardware",
            ],  # Column names for results CSV export
            "cache_results_csv_columns": [
                "experiment_id", "experiment_mode", "execution_mode", "data_source",
                "dataset", "attack_types_combined", "augmentation_ratio",
                "feature_selection_enabled", "hyperparameters_enabled", "data_augmentation_enabled", "hyperparameter_mode",
                "feature_set", "classifier_type", "model_name", "model",
                "n_features", "n_samples_train", "n_samples_test",
                "accuracy", "precision", "recall", "f1_score", "fpr", "fnr", "elapsed_time_s",
                "cv_method", "rfe_ranking", "hyperparameters", "features_list",
            ],  # Column names for temporary cache CSV export
            "top_n_features_heatmap": 15,  # Number of top features to show in heatmap
            "combined_files_evaluation": True,  # Default: combined files evaluation enabled; False = separate files evaluation
            "methods": {
                "augmentation": True,  # Enable data augmentation combination by default
                "feature_selection": True,  # Enable feature selection combination by default
                "hyperparameter_optimization": True,  # Enable hyperparameter optimization combination by default
                "automl": True,  # Enable AutoML pipeline by default
                "stacking": True,  # Enable stacking classifier evaluation by default
            },  # Method toggles for stacking pipeline
            "dataset_file_format": "csv",  # File format for dataset files: arff, csv, parquet, txt
            "augmentation_file_format": "csv",  # File format for augmentation files: arff, csv, parquet, txt
            "match_filenames_to_process": [""],  # Filename patterns to match for processing
            "ignore_files": ["Stacking_Classifiers_Results.csv"],  # Files to ignore during processing
            "ignore_dirs": [
                "Classifiers", "Classifiers_Hyperparameters", "Dataset_Description",
                "Data_Separability", "Feature_Analysis",
            ],  # Directories to ignore during processing
            "feature_sets_config": {
                "use_full": True,  # Enable Full Features strategy (all dataset features)
                "use_pca": True,  # Enable PCA Components strategy (dimensionality reduction)
                "use_rfe": True,  # Enable RFE Features strategy (recursive feature elimination)
                "use_ga": True,  # Enable GA Features strategy (genetic algorithm selection)
                "explicit_features": [],  # Optional explicit feature list; when non-empty, added as an additional "Explicit Features" set alongside all enabled strategies. Use this carefully, as when analyzing multiple datasets, you must put features common between the datasets.
            },  # Configurable feature set strategies for evaluation
            "memory_management": {
                "spill_full_feature_arrays_after_full_eval": True,  # Spill full scaled source matrices to disk-backed memmaps after the full-feature baseline when later feature modes remain.
                "spill_min_array_nbytes": 1073741824,  # Minimum combined train/test source matrix bytes before spilling is used.
                "spill_directory": None,  # Optional directory for temporary spill files; null uses the dataset Stacking/Array_Cache directory.
            },  # Memory lifecycle controls for large feature-selection grids
        }  # Return default stacking configuration
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_default_models_config():
    """
    Return default model hyperparameters configuration section.

    :return: Dictionary containing default model configuration values
    """

    try:
        return {
            "random_forest": {"n_estimators": 100, "random_state": 42},  # Random Forest default parameters
            "svm": {"kernel": "rbf", "probability": True, "random_state": 42},  # SVM default parameters
            "xgboost": {"eval_metric": "mlogloss", "random_state": 42},  # XGBoost default parameters
            "logistic_regression": {"max_iter": 1000, "random_state": 42},  # Logistic Regression default parameters
            "knn": {"n_neighbors": 5},  # KNN default parameters
            "gradient_boosting": {"random_state": 42},  # Gradient Boosting default parameters
            "lightgbm": {"force_row_wise": True, "min_gain_to_split": 0.01, "random_state": 42, "verbosity": -1},  # LightGBM default parameters
            "mlp": {"hidden_layer_sizes": (100,), "max_iter": 500, "random_state": 42},  # MLP Neural Net default parameters
            "stacking_meta": {"n_estimators": 50, "random_state": 42},  # Stacking meta-estimator default parameters
        }  # Return default models configuration
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_default_explainability_config():
    """
    Return default explainability configuration section.

    :return: Dictionary containing default explainability configuration values
    """

    try:
        return {
            "enabled": True,  # Whether explainability pipeline is enabled
            "shap": True,  # Enable SHAP explanations
            "lime": True,  # Enable LIME explanations
            "permutation_importance": True,  # Enable permutation importance analysis
            "feature_importance": True,  # Enable feature importance extraction
            "pdp": False,  # Disable partial dependence plots by default
            "ice": False,  # Disable individual conditional expectation by default
            "surrogate": False,  # Disable surrogate model explanations by default
            "max_display_features": 20,  # Maximum features to display in explainability outputs
            "lime_num_samples": 1000,  # Number of perturbation samples for LIME
            "lime_num_features": 10,  # Number of features to show in LIME explanations
            "shap_max_samples": 100,  # Maximum samples for SHAP computation
            "perm_max_samples": 5000,  # Maximum samples for permutation importance computation to prevent OOM on large datasets
            "background": False,  # Retain legacy setting while RAM-gated process dispatch is selected automatically
            "random_state": 42,  # Random state for reproducibility
            "output_subdir": "explainability",  # Subdirectory for explainability outputs
        }  # Return default explainability configuration
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_default_config():
    """
    Return default configuration dictionary for stacking pipeline.
    
    All configurable parameters with their default values matching
    genetic_algorithm.py and wgangp.py configuration architecture.
    
    :return: Dictionary containing default configuration values
    """

    try:
        return {
        "execution": {
            "verbose": False,
            "play_sound": True,
            "skip_train_if_model_exists": False,
            "low_memory": False,
            "csv_file": None,
            "process_entire_dataset": False,
            "test_data_augmentation": True,
            "execution_mode": "both",
            "classification_mode": "both",
        },
        "dataset": {
            "remove_zero_variance": True,
            "test_size": 0.2,
            "random_state": 42,
            "datasets": {
                "CICDDoS2019-Dataset": [
                    "./Datasets/CICDDoS2019/01-12/",
                    "./Datasets/CICDDoS2019/03-11/",
                ],
            },
        },
        "stacking": get_default_stacking_config(),  # Stacking pipeline configuration section
        "evaluation": {
            "n_jobs": 1,
            "feature_extraction_n_jobs": 1,  # Use one thread for feature extraction by default without changing classifier training parallelism
            "threads_limit": 2,
            "cv_folds": 10,
            "random_state": 42,
            "ram_threshold_gb": 128,
        },
        "models": get_default_models_config(),  # Model hyperparameters configuration section
        "automl": {
            "enabled": False,
            "n_trials": 50,
            "stacking_trials": 20,
            "timeout": 3600,
            "cv_folds": 5,
            "random_state": 42,
            "stacking_top_n": 5,
            "results_filename": "AutoML_Results.csv",
        },
        "tsne": {
            "perplexity": 30,
            "random_state": 42,
            "n_iter": 1000,
            "figsize": [12, 10],
            "dpi": 300,
            "alpha": 0.6,
            "marker_size": 50,
        },
        "paths": {
            "logs_dir": "./Logs",
            "datasets_dir": None,
            "output_dir": "Feature_Analysis",
        },
        "sound": {
            "enabled": True,
            "commands": {
                "Darwin": "afplay",
                "Linux": "aplay",
                "Windows": "start",
            },
            "file": "./.assets/Sounds/NotificationSound.wav",
        },
        "logging": {
            "enabled": True,
            "clean": True,
        },
        "memory_watcher": {
            "enabled": False,  # Keep watcher disabled for normal stacking runs by default
            "sample_interval_seconds": 2,  # Sample process and system memory every two seconds
            "system_memory_threshold_percent": 90,  # Emit threshold events at high system memory pressure
            "process_rss_threshold_gb": None,  # Leave process RSS threshold disabled unless configured
            "capture_process_tree": True,  # Record actual recursive child process rows
            "capture_tracemalloc": False,  # Keep Python allocation tracing disabled by default
            "tracemalloc_frame_depth": 25,  # Use bounded stack depth for optional tracemalloc snapshots
            "output_directory": "Logs/Memory_Watch",  # Store watcher runs under project logs
            "keep_watcher_after_target_exit_seconds": 5,  # Give sidecar time to write terminal records
        },
        "telegram": {
            "enabled": True,
            "verify_env": True,
        },
        "explainability": get_default_explainability_config(),  # Explainability configuration section
    }  # Return default configuration
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_config_file(config_path=None):
    """
    Load configuration from YAML file.
    
    :param config_path: Path to config.yaml file (None for default config.yaml)
    :return: Dictionary with loaded configuration or empty dict if not found
    """

    try:
        if config_path is None:  # If no path specified
            config_path = Path(__file__).parent / "config.yaml"  # Use default config.yaml
        else:  # Path was specified
            config_path = Path(config_path)  # Convert to Path object
        
        if not config_path.exists():  # If config file doesn't exist
            return {}  # Return empty dict
        
        try:  # Try to load YAML file
            with open(config_path, "r", encoding="utf-8") as f:  # Open config file
                config = yaml.safe_load(f)  # Load YAML safely
            return config or {}  # Return loaded config or empty dict
        except Exception as e:  # If loading fails
            print(f"{BackgroundColors.YELLOW}Warning: Failed to load config file {config_path}: {e}{Style.RESET_ALL}")
            return {}  # Return empty dict on error
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def deep_merge_dicts(base, override):
    """Recursively merge override dict into base dict.
    
    :param base: Base dictionary
    :param override: Override dictionary
    :return: Merged dictionary
    """

    try:
        result = dict(base)  # Copy base dictionary
        for key, value in override.items():  # Iterate override items
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):  # Both are dicts
                result[key] = deep_merge_dicts(result[key], value)  # Recursive merge
            else:  # Not both dicts
                result[key] = value  # Override value
        return result  # Return merged dictionary
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def validate_feature_extraction_n_jobs(value: Any, source: str = "evaluation.feature_extraction_n_jobs") -> int:  # Validate the independent feature extraction thread setting.
    """
    Validate a feature extraction thread setting.

    :param value: Configured feature extraction thread value.
    :param source: User-facing setting name used in validation errors.
    :return: Validated integer value.
    """

    if isinstance(value, bool) or not isinstance(value, int) or value == 0 or value < -1:  # Accept only positive integers or the existing all-CPU convention.
        raise ValueError(f"{source} must be -1 or an integer greater than 0")  # Reject invalid feature extraction parallelism values.
    return value  # Return the validated independent feature extraction setting.


def merge_configs(defaults, file_config, cli_args):
    """
    Merge configurations with priority: CLI > file > defaults.
    
    :param defaults: Default configuration dictionary
    :param file_config: Configuration from file
    :param cli_args: Parsed CLI arguments
    :return: Merged configuration dictionary
    """

    try:
        config = deep_merge_dicts(defaults, file_config)  # Merge file config into defaults
        
        classification_mode = config.get("execution", {}).get("classification_mode", None)  # Read classification_mode from config if present
        if classification_mode is not None:  # If classification_mode was explicitly set in config
            _LEGACY_MODES = {"binary": "separate_files", "multiclass": "combined_files", "multi-class": "combined_files"}  # Backward compatibility map for legacy execution mode names
            config["execution"]["execution_mode"] = _LEGACY_MODES.get(classification_mode, classification_mode)  # Normalize legacy mode name to canonical value before assigning
        
        if cli_args is None:  # If no CLI args
            validate_feature_extraction_n_jobs(config.get("evaluation", {}).get("feature_extraction_n_jobs", 1))  # Validate the effective file or default feature extraction setting.
            return config  # Return merged config
        
        if hasattr(cli_args, "verbose") and cli_args.verbose:  # Verbose flag
            config["execution"]["verbose"] = True
        if hasattr(cli_args, "skip_train") and cli_args.skip_train:  # Skip train flag
            config["execution"]["skip_train_if_model_exists"] = True
        if hasattr(cli_args, "csv") and cli_args.csv:  # CSV file override
            config["execution"]["csv_file"] = cli_args.csv
        if hasattr(cli_args, "enable_automl") and cli_args.enable_automl is not None:  # AutoML method toggle CLI override
            config.setdefault("stacking", {}).setdefault("methods", {})["automl"] = cli_args.enable_automl  # Apply AutoML toggle override
        if hasattr(cli_args, "automl_trials") and cli_args.automl_trials is not None:  # AutoML trials
            config["automl"]["n_trials"] = cli_args.automl_trials
        if hasattr(cli_args, "automl_stacking_trials") and cli_args.automl_stacking_trials is not None:  # Stacking trials
            config["automl"]["stacking_trials"] = cli_args.automl_stacking_trials
        if hasattr(cli_args, "automl_timeout") and cli_args.automl_timeout is not None:  # AutoML timeout
            config["automl"]["timeout"] = cli_args.automl_timeout
        if hasattr(cli_args, "test_augmentation"):  # Test augmentation flag (explicit True or False)
            config["execution"]["test_data_augmentation"] = cli_args.test_augmentation
        if hasattr(cli_args, "both") and cli_args.both:  # Both mode flag takes precedence
            config["execution"]["execution_mode"] = "both"
        else:
            cfg_combined_files = config.get("stacking", {}).get("combined_files_evaluation", True)  # Read combined_files_evaluation from stacking config (config.yaml), defaulting to True
            if hasattr(cli_args, "combined_files_flag") and cli_args.combined_files_flag is not None:  # CLI explicitly set (--combined-files or --separate-files)
                effective_combined_files = cli_args.combined_files_flag  # CLI overrides config.yaml
            else:
                effective_combined_files = cfg_combined_files  # Fall back to config.yaml value
            config["execution"]["execution_mode"] = "combined_files" if effective_combined_files else "separate_files"  # Map boolean flag to execution_mode string

        if hasattr(cli_args, "top_n_features") and cli_args.top_n_features is not None:  # CLI override for top-N features
            if not isinstance(cli_args.top_n_features, int) or cli_args.top_n_features <= 0:  # Validate positive integer
                raise ValueError("--top-n-features must be an integer greater than 0")  # Raise for invalid value
            config.setdefault("stacking", {})["top_n_features_heatmap"] = cli_args.top_n_features  # Apply override to config

        if hasattr(cli_args, "stacking_results_dir") and cli_args.stacking_results_dir:
            config.setdefault("stacking", {})["results_dir"] = cli_args.stacking_results_dir

        if hasattr(cli_args, "dataset_path") and cli_args.dataset_path is not None:  # Dataset path CLI override
            config["execution"]["dataset_path"] = cli_args.dataset_path  # Store CLI dataset path override

        if hasattr(cli_args, "enable_augmentation") and cli_args.enable_augmentation is not None:  # Augmentation method toggle CLI override
            config.setdefault("stacking", {}).setdefault("methods", {})["augmentation"] = cli_args.enable_augmentation  # Apply augmentation toggle override

        if hasattr(cli_args, "enable_feature_selection") and cli_args.enable_feature_selection is not None:  # Feature selection method toggle CLI override
            config.setdefault("stacking", {}).setdefault("methods", {})["feature_selection"] = cli_args.enable_feature_selection  # Apply feature selection toggle override

        if hasattr(cli_args, "enable_hyperparameters") and cli_args.enable_hyperparameters is not None:  # Hyperparameter optimization method toggle CLI override
            config.setdefault("stacking", {}).setdefault("methods", {})["hyperparameter_optimization"] = cli_args.enable_hyperparameters  # Apply hyperparameter optimization toggle override

        if hasattr(cli_args, "enable_stacking") and cli_args.enable_stacking is not None:  # Stacking classifier evaluation toggle CLI override
            config.setdefault("stacking", {}).setdefault("methods", {})["stacking"] = cli_args.enable_stacking  # Apply stacking toggle override

        if hasattr(cli_args, "n_jobs") and cli_args.n_jobs is not None:  # Evaluation n_jobs CLI override
            if cli_args.n_jobs == 0:  # scikit-learn/joblib do not accept zero workers
                raise ValueError(
                    "--n-jobs must be a non-zero integer; use -1 for all processors or 1 for memory-safe execution"
                )  # Raise explicit validation error
            config.setdefault("evaluation", {})["n_jobs"] = cli_args.n_jobs  # Apply n_jobs override to estimator construction config

        if hasattr(cli_args, "feature_extraction_n_jobs") and cli_args.feature_extraction_n_jobs is not None:  # Feature extraction n_jobs CLI override.
            validated_feature_extraction_n_jobs = validate_feature_extraction_n_jobs(cli_args.feature_extraction_n_jobs, "--feature-extraction-n-jobs")  # Validate the independent CLI value.
            config.setdefault("evaluation", {})["feature_extraction_n_jobs"] = validated_feature_extraction_n_jobs  # Apply the feature extraction override without changing classifier n_jobs.

        if hasattr(cli_args, "low_memory") and cli_args.low_memory:  # Low memory CLI override
            config["execution"]["low_memory"] = True  # Apply low memory override to config

        if hasattr(cli_args, "dataset_file_format") and cli_args.dataset_file_format is not None:  # Dataset file format CLI override
            config.setdefault("stacking", {})["dataset_file_format"] = cli_args.dataset_file_format  # Apply dataset file format override to config

        if hasattr(cli_args, "augmentation_file_format") and cli_args.augmentation_file_format is not None:  # Augmentation file format CLI override
            config.setdefault("stacking", {})["augmentation_file_format"] = cli_args.augmentation_file_format  # Apply augmentation file format override to config

        if hasattr(cli_args, "feature_sets") and cli_args.feature_sets is not None:  # Feature set strategies CLI override
            allowed = {"full", "pca", "rfe", "ga"}  # Valid strategy identifier names
            strategies = {s.strip().lower() for s in cli_args.feature_sets.split(",") if s.strip()}  # Parse and normalize comma-separated strategy names
            invalid = strategies - allowed  # Identify any unrecognized strategy names
            if invalid:  # If any invalid strategy names were provided
                raise ValueError(f"Invalid --feature-sets values: {invalid}. Valid options are: {allowed}")  # Raise with explicit invalid name details
            fsc = config.setdefault("stacking", {}).setdefault("feature_sets_config", {})  # Access or create feature_sets_config section
            fsc["use_full"] = "full" in strategies  # Set full features toggle from parsed strategies
            fsc["use_pca"] = "pca" in strategies  # Set PCA toggle from parsed strategies
            fsc["use_rfe"] = "rfe" in strategies  # Set RFE toggle from parsed strategies
            fsc["use_ga"] = "ga" in strategies  # Set GA toggle from parsed strategies

        if hasattr(cli_args, "explicit_features") and cli_args.explicit_features is not None:  # Explicit features CLI override
            explicit_list = [f.strip() for f in cli_args.explicit_features.split(",") if f.strip()]  # Parse and strip comma-separated feature names
            if not explicit_list:  # If the parsed list is empty after stripping whitespace
                raise ValueError("--features must provide at least one non-empty feature name")  # Raise with clear error message
            config.setdefault("stacking", {}).setdefault("feature_sets_config", {})["explicit_features"] = explicit_list  # Store parsed explicit feature list in config

        if hasattr(cli_args, "enable_memory_watcher") and cli_args.enable_memory_watcher is not None:  # Memory watcher CLI toggle
            config.setdefault("memory_watcher", {})["enabled"] = cli_args.enable_memory_watcher  # Apply watcher enabled override

        if hasattr(cli_args, "memory_watch_interval_seconds") and cli_args.memory_watch_interval_seconds is not None:  # Memory watcher interval CLI override
            config.setdefault("memory_watcher", {})["sample_interval_seconds"] = cli_args.memory_watch_interval_seconds  # Apply watcher interval override

        if hasattr(cli_args, "memory_watch_threshold_percent") and cli_args.memory_watch_threshold_percent is not None:  # Memory watcher threshold CLI override
            config.setdefault("memory_watcher", {})["system_memory_threshold_percent"] = cli_args.memory_watch_threshold_percent  # Apply watcher system threshold override

        if hasattr(cli_args, "enable_memory_tracemalloc") and cli_args.enable_memory_tracemalloc:  # Optional tracemalloc CLI toggle
            config.setdefault("memory_watcher", {})["capture_tracemalloc"] = True  # Enable optional Python allocation reports

        validate_feature_extraction_n_jobs(config.get("evaluation", {}).get("feature_extraction_n_jobs", 1))  # Validate the effective feature extraction setting after CLI precedence is applied.
        return config  # Return final merged configuration
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def initialize_config(config_path=None, cli_args=None):
    """
    Initialize global CONFIG with merged configuration.
    
    :param config_path: Path to config file (None for default)
    :param cli_args: Parsed CLI arguments (None for no CLI overrides)
    :return: Merged configuration dictionary
    """

    try:
        defaults = get_default_config()  # Get default configuration
        file_config = load_config_file(config_path)  # Load file configuration
        config = merge_configs(defaults, file_config, cli_args)  # Merge all configurations
        
        global CONFIG  # Access global CONFIG
        CONFIG = config  # Set global CONFIG
        
        return config  # Return merged configuration
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def initialize_logger(config=None):
    """
    Initialize logger using configuration.
    
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        global logger  # Access global logger
        
        logs_dir = config.get("paths", {}).get("logs_dir", "./Logs")  # Get logs directory
        clean = config.get("logging", {}).get("clean", True)  # Get clean flag
        
        os.makedirs(logs_dir, exist_ok=True)  # Ensure logs directory exists
        log_path = Path(logs_dir) / f"{Path(__file__).stem}.log"  # Build log file path
        
        logger = Logger(str(log_path), clean=clean)  # Create Logger instance
        sys.stdout = logger  # Redirect stdout to logger
        sys.stderr = logger  # Redirect stderr to logger
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def verbose_output(true_string="", false_string="", config=None):
    """
    Output a message if verbose mode is enabled in configuration.

    :param true_string: The string to be outputted if verbose is enabled.
    :param false_string: The string to be outputted if verbose is disabled.
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None.
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        verbose = config.get("execution", {}).get("verbose", False)  # Get verbose flag
        
        if verbose and true_string != "":  # If verbose is True and a true_string was provided
            print(true_string)  # Output the true statement string
        elif false_string != "":  # If a false_string was provided
            print(false_string)  # Output the false statement string
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def verify_dot_env_file(config=None):
    """
    Verify if the .env file exists in the current directory.

    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: True if the .env file exists, False otherwise.
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        verify_env = config.get("telegram", {}).get("verify_env", True)  # Get verify_env flag
        
        if not verify_env:  # If verification is disabled
            return True  # Skip verification
        
        env_path = Path(__file__).parent / ".env"  # Path to the .env file
        if not env_path.exists():  # If the .env file does not exist
            print(f"{BackgroundColors.CYAN}.env{BackgroundColors.YELLOW} file not found at {BackgroundColors.CYAN}{env_path}{BackgroundColors.YELLOW}. Telegram messages may not be sent.{Style.RESET_ALL}")
            return False  # Return False

        return True  # Return True if the .env file exists
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def setup_telegram_bot(config=None):
    """
    Set up the Telegram bot for progress messages.

    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None.
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        telegram_enabled = config.get("telegram", {}).get("enabled", True)  # Get telegram enabled flag
        
        if not telegram_enabled:  # If Telegram is disabled
            return  # Skip setup
        
        verbose_output(
            f"{BackgroundColors.GREEN}Setting up Telegram bot for messages...{Style.RESET_ALL}", config=config
        )  # Output the verbose message

        verify_dot_env_file(config)  # Verify if the .env file exists

        global TELEGRAM_BOT  # Declare the module-global telegram_bot variable

        try:  # Try to initialize the Telegram bot
            TELEGRAM_BOT = TelegramBot()  # Initialize Telegram bot for progress messages
            telegram_module.TELEGRAM_DEVICE_INFO = f"{telegram_module.get_local_ip()} - {platform.system()}"  # Set device info string with IP and OS
            telegram_module.RUNNING_CODE = os.path.basename(__file__)  # Set currently running script name
        except Exception as e:
            print(f"{BackgroundColors.RED}Failed to initialize Telegram bot: {e}{Style.RESET_ALL}")  # Report initialization failure to terminal
            TELEGRAM_BOT = None  # Set to None if initialization fails
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def set_threads_limit_based_on_ram(config=None):
    """
    Sets threads limit to 1 if system RAM is below threshold to avoid memory issues.

    Also enforces n_jobs=1 in config when RAM is constrained or low_memory is active,
    so all downstream model instantiation respects the safe parallelism ceiling and
    prevents OOM-induced process termination during parallel model fitting.

    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Threads limit value
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Verifying system RAM to set threads_limit...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        threads_limit = config.get("evaluation", {}).get("threads_limit", 2)  # Get threads limit from config
        ram_threshold = config.get("evaluation", {}).get("ram_threshold_gb", 32)  # Get RAM threshold from config
        ram_gb = psutil.virtual_memory().total / (1024**3)  # Get total system RAM in GB

        if ram_gb <= ram_threshold:  # If RAM is less than or equal to threshold
            threads_limit = 1  # Set threads_limit to 1
            verbose_output(
                f"{BackgroundColors.YELLOW}System RAM is {ram_gb:.1f}GB (<={ram_threshold}GB). Setting threads_limit to 1.{Style.RESET_ALL}",
                config=config
            )

        low_memory = config.get("execution", {}).get("low_memory", False)  # Read low memory flag to determine if model-level parallelism must also be restricted
        current_n_jobs = config.get("evaluation", {}).get("n_jobs", 1)  # Read the currently configured n_jobs value before any override
        effective_n_jobs = 1 if (threads_limit == 1 or low_memory) else current_n_jobs  # Force n_jobs to 1 when RAM-constrained or low_memory is active to prevent OOM during parallel model fitting
        config.setdefault("evaluation", {})["n_jobs"] = effective_n_jobs  # Write effective n_jobs back into config so all downstream model instantiation respects the safe value

        if effective_n_jobs != current_n_jobs:  # Verify if n_jobs was actually changed to issue a visible diagnostic
            reason = "low_memory=True" if low_memory else f"ram_gb={ram_gb:.1f}GB (<={ram_threshold}GB)"  # Determine the specific reason that triggered the n_jobs override
            print(f"{BackgroundColors.YELLOW}[RESOURCE GUARD] n_jobs overridden: {BackgroundColors.CYAN}{current_n_jobs}{BackgroundColors.YELLOW} -> {BackgroundColors.CYAN}{effective_n_jobs}{BackgroundColors.YELLOW} (reason: {reason}). Prevents OOM during parallel model fitting.{Style.RESET_ALL}")  # Inform operator that n_jobs was limited for OOM safety
            send_telegram_message(TELEGRAM_BOT, f"[RESOURCE GUARD] n_jobs overridden: {current_n_jobs} -> {effective_n_jobs} (reason: {reason}). Prevents OOM during parallel model fitting.")  # Notify Telegram about the resource guard activation for remote monitoring

        return threads_limit  # Return the threads limit value
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def resolve_feature_extraction_n_jobs(config=None) -> Tuple[int, int, Optional[str]]:  # Resolve requested and RAM-safe feature extraction thread counts.
    """
    Resolve requested and effective feature extraction thread counts.

    :param config: Configuration dictionary (uses global CONFIG if None).
    :return: Tuple of requested value, effective value, and optional adjustment reason.
    """

    if config is None:  # Use global configuration when no explicit configuration is supplied.
        config = CONFIG  # Assign the global runtime configuration.

    requested_n_jobs = validate_feature_extraction_n_jobs(config.get("evaluation", {}).get("feature_extraction_n_jobs", 1))  # Read and validate the independent feature extraction setting.
    available_cpus = max(1, os.cpu_count() or 1)  # Resolve a safe positive CPU availability value.
    effective_n_jobs = available_cpus if requested_n_jobs == -1 else min(requested_n_jobs, available_cpus)  # Resolve all-CPU mode and prevent CPU oversubscription.
    adjustment_reasons = []  # Accumulate only reasons that reduce the requested feature extraction capacity.
    if requested_n_jobs > available_cpus:  # Detect a fixed request above available CPU capacity.
        adjustment_reasons.append(f"requested value exceeds {available_cpus} available CPUs")  # Record the CPU-capacity reduction reason.

    low_memory = bool(config.get("execution", {}).get("low_memory", False))  # Read the existing low-memory safety setting.
    ram_threshold = float(config.get("evaluation", {}).get("ram_threshold_gb", 32))  # Read the existing classifier RAM threshold without changing it.
    ram_gb = psutil.virtual_memory().total / (1024**3)  # Read total system RAM using the existing project mechanism.
    ram_constrained = ram_gb <= ram_threshold  # Apply the existing RAM threshold comparison to feature extraction.
    if effective_n_jobs != 1 and (low_memory or ram_constrained):  # Force sequential feature extraction under the existing memory safety conditions.
        effective_n_jobs = 1  # Reduce PCA numerical work to one thread for RAM safety.
        adjustment_reasons.append("low_memory=True" if low_memory else f"ram_gb={ram_gb:.1f}GB (<={ram_threshold:g}GB)")  # Record the exact memory safety reason.

    adjustment_reason = "; ".join(adjustment_reasons) if adjustment_reasons else None  # Format an optional concise adjustment reason.
    return requested_n_jobs, effective_n_jobs, adjustment_reason  # Return independent requested and effective feature extraction values.


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


def get_files_to_process(directory_path, file_extension=".csv", config=None):
    """
    Collect all files with a given extension inside a directory (non-recursive).

    Performs validation, respects IGNORE_FILES, and optionally filters by
    MATCH_FILENAMES_TO_PROCESS when defined.

    :param directory_path: Path to the directory to scan
    :param file_extension: File extension to include (default: ".csv")
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Sorted list of matching file paths
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Getting all {BackgroundColors.CYAN}{file_extension}{BackgroundColors.GREEN} files in: {BackgroundColors.CYAN}{directory_path}{Style.RESET_ALL}",
            config=config
        )  # Verbose: starting file collection
        verify_filepath_exists(directory_path)  # Validate directory path exists

        if not os.path.isdir(directory_path):  # Verify if path is a valid directory
            verbose_output(
                f"{BackgroundColors.RED}Not a directory: {BackgroundColors.CYAN}{directory_path}{Style.RESET_ALL}",
                config=config
            )  # Verbose: invalid directory
            return []  # Return empty list for invalid paths

        match_filenames = config.get("stacking", {}).get("match_filenames_to_process", [""])  # Get match filenames from config
        ignore_files = config.get("stacking", {}).get("ignore_files", [])  # Get ignore files from config
        
        match_names = (
            set(match_filenames) if match_filenames not in ([], [""], [" "]) else None
        )  # Load match list or None
        if match_names:
            verbose_output(
                f"{BackgroundColors.GREEN}Filtering to filenames: {BackgroundColors.CYAN}{match_names}{Style.RESET_ALL}",
                config=config
            )  # Verbose: applying filename filter

        files = []  # Accumulator for valid files

        for item in os.listdir(directory_path):  # Iterate directory entries
            item_path = os.path.join(directory_path, item)  # Absolute path
            filename = os.path.basename(item_path)  # Extract just the filename

            if any(ignore == filename or ignore == item_path for ignore in ignore_files):  # Verify if file is in the ignore list
                verbose_output(
                    f"{BackgroundColors.YELLOW}Ignoring {BackgroundColors.CYAN}{filename}{BackgroundColors.YELLOW} (listed in IGNORE_FILES){Style.RESET_ALL}",
                    config=config
                )  # Verbose: ignoring file
                continue  # Skip ignored file

            if os.path.isfile(item_path) and item.lower().endswith(file_extension):  # File matches extension requirement
                if (
                    match_names is not None and filename not in match_names
                ):  # Filename not included in MATCH_FILENAMES_TO_PROCESS
                    verbose_output(
                        f"{BackgroundColors.YELLOW}Skipping {BackgroundColors.CYAN}{filename}{BackgroundColors.YELLOW} (not in MATCH_FILENAMES_TO_PROCESS){Style.RESET_ALL}",
                        config=config
                    )  # Verbose: skipping non-matching file
                    continue  # Skip this file
                files.append(item_path)  # Add file to result list

        return sorted(files)  # Return sorted list for deterministic output
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def order_files_by_size_descending(files_list: List[str], config: Optional[dict] = None) -> List[str]:
    """
    Order dataset files by filesystem size descending.

    :param files_list: List of dataset file paths.
    :param config: Configuration dictionary used for verbose output.
    :return: List of dataset file paths ordered by size descending.
    """

    try:  # Protect size ordering so project logging and Telegram alerts remain consistent
        if config is None:  # Use global configuration when no explicit configuration is supplied
            config = CONFIG  # Assign global configuration reference

        resolved_paths = {}  # Store resolved paths for deterministic equal-size ordering
        file_sizes = {}  # Store one filesystem size lookup per path
        for file_path in files_list:  # Iterate filtered file paths in caller-provided order
            resolved_path = os.path.abspath(file_path)  # Resolve the full path before querying filesystem size
            resolved_paths[file_path] = resolved_path  # Store the resolved path for later tie ordering
            try:  # Read the actual filesystem size for this file
                file_sizes[file_path] = os.path.getsize(resolved_path)  # Store byte size from the resolved path
            except Exception as e:  # Handle size-read errors with path-specific logging
                print(f"{BackgroundColors.RED}[ERROR] Failed to read file size for {BackgroundColors.CYAN}{resolved_path}{BackgroundColors.RED}: {e}{Style.RESET_ALL}")  # Log size-read failure before aborting
                raise  # Preserve failure instead of silently dropping files or guessing order

        ordered_files = sorted(files_list, key=lambda file_path: (-file_sizes[file_path], resolved_paths[file_path]))  # Sort by size descending and resolved path ascending for ties
        return ordered_files  # Return ordered paths using the caller-provided path strings
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def get_dataset_name(input_path):
    """
    Extract the dataset name from CSVs path.

    :param input_path: Path to the CSVs files
    :return: Dataset name
    """

    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Extracting dataset name from CSV path: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}"
        )  # Output the verbose message

        datasets_pos = input_path.find("/Datasets/")  # Find the position of "/Datasets/" in the path
        if datasets_pos != -1:  # If "/Datasets/" is found in the path
            after_datasets = input_path[datasets_pos + len("/Datasets/") :]  # Get the substring after "/Datasets/"
            next_slash = after_datasets.find("/")  # Find the next "/"
            if next_slash != -1:  # If there is another "/"
                dataset_name = after_datasets[:next_slash]  # Take until the next "/"
            else:  # If there is no other "/"
                dataset_name = (
                    after_datasets.split("/")[0] if "/" in after_datasets else after_datasets
                )  # No more "/", take the first part if any
        else:  # If "/Datasets/" is not found in the path
            dataset_name = os.path.basename(input_path)  # Fallback to basename if "Datasets" not in path

        return dataset_name  # Return the dataset name
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def detect_label_column(columns):
    """
    Try to guess the label column based on common naming conventions.

    :param columns: List of column names.
    :return: The name of the label column if found, else None.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        candidates = ["label", "class", "target", "y", "category"]  # Common label column names to verify for exact matches

        for col in columns:  # First search for exact matches
            if col.lower() in candidates:  # Verify match against candidate set
                return col  # Return detected label column

        for col in columns:  # Second search for partial matches
            if "target" in col.lower() or "label" in col.lower():  # Verify partial match condition
                return col  # Return detected label column

        return None  # Return None if no label column is found
    except Exception as e:  # Catch any exception for safe logging
        print(str(e))  # Output error for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Notify via Telegram
        raise  # Preserve original failure behavior


def process_single_file(f, config=None, remove_zero_variance=None):
    """
    Process a single dataset file: load, preprocess, and extract target and features.

    :param f: Path to the dataset CSV file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :param remove_zero_variance: Whether to remove zero-variance columns before returning.
    :return: Tuple (df_clean, target_col, feat_cols) or None if invalid
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        progress_idx = config.get("execution", {}).get("progress_index", None)  # Get progress index from config when provided
        progress_total = config.get("execution", {}).get("progress_total", None)  # Get progress total from config when provided
        if progress_idx is not None and progress_total is not None:  # If both progress values are present
            verbose_output(f"{BackgroundColors.GREEN}Processing file [{progress_idx}/{progress_total}]: {BackgroundColors.CYAN}{f}{Style.RESET_ALL}", config=config)  # Output the verbose message with index
        else:  # If no progress metadata is available
            verbose_output(f"{BackgroundColors.GREEN}Processing file: {BackgroundColors.CYAN}{f}{Style.RESET_ALL}", config=config)  # Output the verbose message without index

        write_memory_phase_event("before_each_source_file_load", config=config, dataset_source=f, dataset_identity=os.path.basename(str(f)), event_outcome="starting")  # Publish source-file load start
        df = load_dataset(f, config=config)  # Load the dataset from the file
        if df is None:  # If loading failed
            write_memory_phase_event("after_each_source_file_load", config=config, dataset_source=f, dataset_identity=os.path.basename(str(f)), event_outcome="load_failed")  # Publish failed source-file load
            return None  # Return None
        
        if remove_zero_variance is None:
            remove_zero_variance = config.get("dataset", {}).get("remove_zero_variance", True)  # Preserve configured behavior outside the stacking split flow.
        df_clean = preprocess_dataframe(df, remove_zero_variance=remove_zero_variance, config=config)  # Preprocess the dataframe

        del df  # Release raw dataframe to free memory after preprocessing
        gc.collect()  # Force garbage collection to reclaim memory from deleted raw dataframe

        if df_clean is None or df_clean.empty:  # If preprocessing failed or dataframe is empty
            write_memory_phase_event("after_each_source_file_load", config=config, dataset_source=f, dataset_identity=os.path.basename(str(f)), event_outcome="empty_after_preprocessing")  # Publish empty source-file result
            return None  # Return None

        target_col = detect_label_column(df_clean.columns.tolist())  # Detect label column using naming conventions
        if target_col is None:  # If label column was not detected
            target_col = df_clean.columns[-1]  # Fall back to last column as target
        feat_cols = [c for c in df_clean.columns if c != target_col and pd.api.types.is_numeric_dtype(df_clean[c])]  # Get numeric feature columns excluding label column
        if not feat_cols:  # If no numeric features
            write_memory_phase_event("after_each_source_file_load", config=config, dataset_source=f, dataset_identity=os.path.basename(str(f)), row_count=len(df_clean), feature_count=0, event_outcome="no_numeric_features")  # Publish unusable source-file result
            return None  # Return None

        write_memory_phase_event("after_each_source_file_load", config=config, dataset_source=f, dataset_identity=os.path.basename(str(f)), row_count=len(df_clean), feature_count=len(feat_cols), attack_scope=target_col, event_outcome="loaded")  # Publish successful source-file load
        return (df_clean, target_col, feat_cols)  # Return the processed data
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def handle_target_column_consistency(target_col_name, this_target, f, df_clean, config=None):
    """
    Handle target column consistency by renaming if necessary.

    :param target_col_name: Current target column name (or None)
    :param this_target: Target column name in this file
    :param f: File path for warning message
    :param df_clean: DataFrame to rename if needed
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (updated_target_col_name, updated_df_clean)
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        verbose_output(
            f"{BackgroundColors.GREEN}Verifying target column consistency for: {BackgroundColors.CYAN}{f}{Style.RESET_ALL} (target: {BackgroundColors.CYAN}{this_target}{Style.RESET_ALL})...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        if target_col_name is None:  # If target column not set yet
            target_col_name = this_target  # Set it to this target
        elif this_target != target_col_name:  # If target column name differs
            print(f"{BackgroundColors.YELLOW}Warning: target column name mismatch: {f} uses {this_target} while others use {target_col_name}. Trying to proceed by renaming.{Style.RESET_ALL}")  # Print warning
            df_clean = df_clean.rename(columns={this_target: target_col_name})  # Rename the column

        return (target_col_name, df_clean)  # Return updated values
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def intersect_features(common_features, feat_cols, config=None):
    """
    Intersect features with common features set.

    :param common_features: Current common features set (or None)
    :param feat_cols: Feature columns in this file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Updated common features set
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        verbose_output(
            f"{BackgroundColors.GREEN}Intersecting features for current file...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        if common_features is None:  # If common features not set yet
            common_features = set(feat_cols)  # Initialize with this file's features
        else:  # Otherwise, intersect with existing common features
            common_features &= set(feat_cols)  # Update common features

        return common_features  # Return updated common features
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def find_common_features_and_target(processed_files, config=None):
    """
    Find common features and consistent target column from processed files.

    :param processed_files: List of (f, df_clean, target_col, feat_cols)
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (common_features, target_col_name, dfs) or (None, None, []) if invalid
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Finding common features and target column among processed files...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        dfs = []  # Initialize list to store valid dataframes
        common_features = None  # Initialize set for common features
        target_col_name = None  # Initialize target column name

        for f, df_clean, this_target, feat_cols in processed_files:  # Iterate over processed files
            target_col_name, df_clean = handle_target_column_consistency(target_col_name, this_target, f, df_clean, config=config)  # Handle target consistency
            common_features = intersect_features(common_features, feat_cols, config=config)  # Intersect features
            dfs.append((f, df_clean))  # Add the file and cleaned dataframe to the list

        if not dfs or not common_features:  # If no valid dataframes or no common features
            return (None, None, [])  # Return invalid

        if target_col_name is None:  # If no target column was found
            return (None, None, [])  # Return invalid

        common_features = sorted(list(common_features))  # Sort the common features list
        return (common_features, target_col_name, dfs)  # Return the results
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def create_reduced_dataframes(dfs, common_features, target_col_name, config=None):
    """
    Create reduced dataframes with only common features and target.

    :param dfs: List of (f, df_clean)
    :param common_features: List of common feature names
    :param target_col_name: Name of the target column
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: List of reduced dataframes
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        verbose_output(
            f"{BackgroundColors.GREEN}Creating reduced dataframes with common features and target...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        reduced_dfs = []  # Initialize list for reduced dataframes
        for idx, (f, df_clean) in enumerate(dfs):  # Iterate over valid dataframes with index for memory release
            cols_to_keep = [c for c in common_features if c in df_clean.columns]  # Get common features present in this df
            cols_to_keep.append(target_col_name)  # Add the target column
            reduced = df_clean.loc[:, cols_to_keep].copy()  # Create reduced dataframe with only common features and target
            reduced_dfs.append(reduced)  # Add to list

            dfs[idx] = (f, None)  # Release original full dataframe reference to free memory

        del dfs  # Release the dfs list reference after building all reduced dataframes
        gc.collect()  # Force garbage collection to reclaim memory from released original dataframes

        return reduced_dfs  # Return the reduced dataframes
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def combine_and_clean_dataframes(reduced_dfs, config=None):
    """
    Combine reduced dataframes, clean, and validate.

    :param reduced_dfs: List of reduced dataframes
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Combined dataframe or None if empty
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        verbose_output(
            f"{BackgroundColors.GREEN}Combining reduced dataframes and cleaning...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        combined = pd.concat(reduced_dfs, ignore_index=True)  # Concatenate all reduced dataframes

        del reduced_dfs  # Release list of reduced dataframes to free memory after concatenation
        gc.collect()  # Force garbage collection to reclaim memory from deleted reduced dataframes

        combined.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN in-place to save memory
        combined.dropna(inplace=True)  # Drop rows with NaN values in-place to avoid creating a copy

        if combined.empty:  # If combined dataframe is empty
            print(f"{BackgroundColors.RED}Combined dataset is empty after alignment and NaN removal.{Style.RESET_ALL}")  # Print error
            return None  # Return None

        return combined  # Return the combined dataframe
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def combine_dataset_files(files_list, config=None):
    """
    Load, preprocess and combine multiple dataset CSVs into a single DataFrame.

    :param files_list: List of dataset CSV file paths to combine
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Combined DataFrame with aligned features and target, or None if no compatible files found
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Combining dataset files: {BackgroundColors.CYAN}{files_list}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        processed_files = []  # Initialize list for processed file data
        total_files = len(files_list)  # Compute total number of files for progress reporting
        for idx, f in enumerate(files_list, start=1):  # Iterate over each file with index for progress
            cfg = dict(config)  # Create shallow copy of config to avoid mutating caller config
            exec_cfg = dict(cfg.get("execution", {}))  # Copy nested execution config for modification
            exec_cfg["progress_index"] = idx  # Set current file index in execution config
            exec_cfg["progress_total"] = total_files  # Set total files count in execution config
            cfg["execution"] = exec_cfg  # Assign modified execution config back into config copy
            result = process_single_file(f, config=cfg)  # Process the single file with progress info in config
            if result is not None:  # If processing succeeded
                df_clean, target_col, feat_cols = result  # Unpack the result
                processed_files.append((f, df_clean, target_col, feat_cols))  # Add to processed list

        if not processed_files:  # If no files were processed successfully
            print(f"{BackgroundColors.RED}No compatible files found to combine for dataset: {files_list}.{Style.RESET_ALL}")  # Print error
            return None  # Return None

        common_features, target_col_name, dfs = find_common_features_and_target(processed_files, config=config)  # Find common features and target
        if common_features is None:  # If finding common features failed
            print(f"{BackgroundColors.RED}No valid target column found.{Style.RESET_ALL}")  # Print error
            del processed_files  # Release processed files to free memory on failure path
            gc.collect()  # Force garbage collection on failure path
            return None  # Return None

        del processed_files  # Release processed files list to free memory before creating reduced dataframes
        gc.collect()  # Force garbage collection to reclaim memory from released processed files

        reduced_dfs = create_reduced_dataframes(dfs, common_features, target_col_name, config=config)  # Create reduced dataframes
        combined = combine_and_clean_dataframes(reduced_dfs, config=config)  # Combine and clean the dataframes

        return combined, target_col_name  # Return the combined dataframe and target column name
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def extract_attack_label_from_path(file_path):
    """
    Extract attack type label from file path for combined files evaluation.
    
    :param file_path: Path to the dataset CSV file
    :return: Attack type string extracted from filename or path
    """

    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Extracting attack label from path: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}"
        )  # Output the verbose message
        
        file_path_obj = Path(file_path)  # Create Path object from file path string
        filename_stem = file_path_obj.stem  # Extract filename without extension
        
        clean_stem = filename_stem.replace("_data_augmented", "").replace("_cleaned", "").replace("_processed", "")  # Remove common suffixes from stem
        
        attack_label = clean_stem  # Set attack label to cleaned filename stem
        
        verbose_output(
            f"{BackgroundColors.GREEN}Extracted attack label: {BackgroundColors.CYAN}{attack_label}{Style.RESET_ALL}"
        )  # Output extracted label
        
        return attack_label  # Return attack type label string
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def concat_files_into_combined_files_df(processed_files_with_labels, common_features_list, attack_types_set, config):
    """
    Align each processed file to the common feature set, assign attack type labels, concatenate into a single DataFrame, and clean infinite/NaN values.

    :param processed_files_with_labels: List of tuples (file_path, df_clean, target_col, feat_cols, attack_label).
    :param common_features_list: Sorted list of feature column names shared across all files.
    :param attack_types_set: Set of unique attack type strings used to build the final sorted list.
    :param config: Configuration dictionary passed through for verbose output.
    :return: Tuple of (combined_df, attack_types_list) on success, or None when the result is empty after cleaning.
    """

    combined_parts = []  # Initialize list to accumulate dataframe parts
    
    total_samples = sum(len(df_clean) for _, df_clean, _, _, _ in processed_files_with_labels)  # Calculate total samples across all processed files for percentage reporting

    for idx, (f, df_clean, this_target, feat_cols, attack_label) in enumerate(processed_files_with_labels):  # Iterate over processed files with index
        df_subset = df_clean[common_features_list].copy()  # Select only common features as a copy for safe modification
        df_subset["attack_type"] = df_clean[this_target].values  # Preserve original label column values to maintain benign and attack class integrity
        combined_parts.append(df_subset)  # Append to combined parts list

        processed_files_with_labels[idx] = (f, None, this_target, feat_cols, attack_label)  # Release original full dataframe reference to free memory

        samples_count = len(df_subset)  # Get sample count contributed by the current attack type
        samples_percentage = (samples_count / total_samples) * 100 if total_samples else 0.0  # Calculate percentage contribution to the final combined dataset
        
        verbose_output(
            f"{BackgroundColors.GREEN}Added {BackgroundColors.CYAN}{samples_count}{BackgroundColors.GREEN} samples from {BackgroundColors.CYAN}{attack_label}{BackgroundColors.GREEN} ({BackgroundColors.CYAN}{samples_percentage:.2f}%{BackgroundColors.GREEN} of total){Style.RESET_ALL}",
            config=config
        )  # Output samples added message including percentage contribution

    gc.collect()  # Force garbage collection to reclaim memory from released original dataframes

    write_memory_phase_event("before_combined_dataframe_creation", config=config, source_file_count=len(processed_files_with_labels), feature_count=len(common_features_list), attack_scope=sorted(list(attack_types_set)), train_sample_count=total_samples, event_outcome="starting")  # Publish combined dataframe creation start
    combined_df = pd.concat(combined_parts, ignore_index=True)  # Concatenate all parts into single dataframe

    del combined_parts  # Release list of dataframe parts to free memory after concatenation
    gc.collect()  # Force garbage collection to reclaim memory from deleted parts list

    combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN in-place to save memory
    combined_df.dropna(inplace=True)  # Drop rows with NaN values in-place to avoid creating a copy

    if combined_df.empty:  # If combined dataframe is empty after cleaning
        print(f"{BackgroundColors.RED}Combined files evaluation dataset is empty after cleaning.{Style.RESET_ALL}")  # Print error
        write_memory_phase_event("after_combined_dataframe_creation", config=config, source_file_count=len(processed_files_with_labels), feature_count=len(common_features_list), attack_scope=sorted(list(attack_types_set)), event_outcome="empty_after_cleaning")  # Publish failed combined dataframe creation
        return None  # Signal failure to caller

    attack_types_list = sorted(list(attack_types_set))  # Convert attack types set to sorted list
    write_memory_phase_event("after_combined_dataframe_creation", config=config, source_file_count=len(processed_files_with_labels), row_count=len(combined_df), feature_count=len(common_features_list), attack_scope=attack_types_list, event_outcome="created")  # Publish successful combined dataframe creation
    print(
        f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Combined files evaluation dataset created: {BackgroundColors.CYAN}{len(combined_df)} samples, {len(common_features_list)} features, {len(attack_types_list)} classes{Style.RESET_ALL}"
    )  # Print summary of combined dataset
    return combined_df, attack_types_list  # Return combined dataframe and attack types list


def compute_common_features_across_files(processed_files_with_labels, config):
    """
    Compute the intersection of feature columns across all processed files and determine the common target column name.

    :param processed_files_with_labels: List of tuples (file_path, df_clean, target_col, feat_cols, attack_label).
    :param config: Configuration dictionary passed through for verbose output.
    :return: Tuple of (common_features_list, target_col_name) on success, or None when no common features exist.
    """

    common_features = None  # Initialize common features set
    target_col_name = None  # Initialize target column name

    for f, df_clean, this_target, feat_cols, attack_label in processed_files_with_labels:  # Iterate over processed files
        if target_col_name is None:  # If target not yet set
            target_col_name = this_target  # Set target column name

        if common_features is None:  # If common features not yet initialized
            common_features = set(feat_cols)  # Initialize with first file's features
        else:  # If common features already initialized
            common_features = common_features.intersection(set(feat_cols))  # Intersect with current file's features

    if not common_features:  # If no common features found
        print(f"{BackgroundColors.RED}No common features found across files for combined files evaluation.{Style.RESET_ALL}")  # Print error
        return None  # Signal failure to caller

    common_features_list = sorted(list(common_features))  # Convert to sorted list
    verbose_output(
        f"{BackgroundColors.GREEN}Common features for combined files evaluation: {BackgroundColors.CYAN}{len(common_features_list)} features{Style.RESET_ALL}",
        config=config
    )  # Output common features count
    return common_features_list, target_col_name  # Return common features and target column name


def process_files_and_extract_labels(files_list, config, remove_zero_variance=False):
    """
    Process each dataset file, extract its attack type label, and accumulate results into a list suitable for combined files evaluation.

    :param files_list: List of dataset CSV file paths to process.
    :param config: Configuration dictionary passed through to processing helpers.
    :param remove_zero_variance: Whether to remove zero-variance columns while loading each source.
    :return: Tuple of (processed_files_with_labels, attack_types_set) on success, or None when no files could be processed.
    """

    processed_files_with_labels = []  # Initialize list for processed file data with attack labels
    attack_types_set = set()  # Initialize set to track unique attack types

    total_files = len(files_list)  # Compute total number of files for progress reporting
    for idx, f in enumerate(files_list, start=1):  # Iterate over each file with index for progress
        cfg = dict(config)  # Create shallow copy of config to avoid mutating caller config
        exec_cfg = dict(cfg.get("execution", {}))  # Copy nested execution config for modification
        exec_cfg["progress_index"] = idx  # Set current file index in execution config
        exec_cfg["progress_total"] = total_files  # Set total files count in execution config
        cfg["execution"] = exec_cfg  # Assign modified execution config back into config copy
        result = process_single_file(f, config=cfg, remove_zero_variance=remove_zero_variance)  # Preserve original training schema when loading augmented testing sources.
        if result is not None:  # If processing succeeded
            df_clean, target_col, feat_cols = result  # Unpack the result
            attack_label = extract_attack_label_from_path(f)  # Extract attack type from filename for tuple reference
            classes = df_clean[target_col].unique()  # Extract all unique classes from the label column
            for cls in classes:  # Iterate over all unique classes found in this file
                attack_types_set.add(str(cls))  # Add each class to the global attack types set
            processed_files_with_labels.append((f, df_clean, target_col, feat_cols, attack_label))  # Add to processed list with label
        else:  # If processing failed
            verbose_output(
                f"{BackgroundColors.YELLOW}Skipping file due to processing failure: {BackgroundColors.CYAN}{f}{Style.RESET_ALL}",
                config=cfg
            )  # Output warning message including progress when available

    if not processed_files_with_labels:  # If no files were processed successfully
        print(f"{BackgroundColors.RED}No compatible files found to combine for combined files evaluation dataset.{Style.RESET_ALL}")  # Print error message
        return None  # Signal failure to caller

    print(
        f"{BackgroundColors.GREEN}Found {BackgroundColors.CYAN}{len(attack_types_set)}{BackgroundColors.GREEN} unique attack types for combined files evaluation: {BackgroundColors.CYAN}{sorted(attack_types_set)}{Style.RESET_ALL}"
    )  # Print attack types summary
    return processed_files_with_labels, attack_types_set  # Return processed data and attack types set


def combine_files_for_combined_evaluation(files_list, config=None, remove_zero_variance=False):
    """
    Combine multiple dataset files into a single combined files evaluation dataset.
    Each file represents a different attack type and becomes a unique class label.
    
    :param files_list: List of dataset CSV file paths to combine for combined files evaluation
    :param config: Configuration dictionary (uses global CONFIG if None)
    :param remove_zero_variance: Whether to remove zero-variance columns while loading each source.
    :return: Tuple (combined_df, attack_types_list, target_col_name) or (None, None, None) if failed
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        verbose_output(
            f"{BackgroundColors.GREEN}Combining files for combined files evaluation: {BackgroundColors.CYAN}{len(files_list)} files{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message
        
        if not files_list:  # If files list is empty
            print(f"{BackgroundColors.RED}No files provided for combined files evaluation combination.{Style.RESET_ALL}")  # Print error message
            return (None, None, None)  # Return None tuple
        
        process_result = process_files_and_extract_labels(files_list, config, remove_zero_variance=remove_zero_variance)  # Process files while preserving the requested feature schema.
        if process_result is None:  # If processing failed
            return (None, None, None)  # Return failure tuple
        processed_files_with_labels, attack_types_set = process_result  # Unpack processed data and attack types set
        
        feature_result = compute_common_features_across_files(processed_files_with_labels, config)  # Compute common features across all files
        if feature_result is None:  # If no common features found
            del processed_files_with_labels  # Release processed files to free memory on failure path
            gc.collect()  # Force garbage collection on failure path
            return (None, None, None)  # Return failure tuple
        common_features_list, target_col_name = feature_result  # Unpack common features and target column name
        
        concat_result = concat_files_into_combined_files_df(processed_files_with_labels, common_features_list, attack_types_set, config)  # Concatenate files into combined files evaluation dataframe

        del processed_files_with_labels  # Release individual dataframes to free memory after concatenation
        gc.collect()  # Force garbage collection to reclaim memory from released individual dataframes

        if concat_result is None:  # If concatenation failed
            return (None, None, None)  # Return failure tuple
        
        if not isinstance(concat_result, (list, tuple)) or len(concat_result) != 2:  # Verify concat_result is iterable and has exactly two items
            print(f"{BackgroundColors.RED}Unexpected return from concat_files_into_combined_files_df: {type(concat_result)}{Style.RESET_ALL}")  # Log unexpected return type for diagnosis
            err_val = ValueError("concat_files_into_combined_files_df returned unexpected format")  # Create ValueError to describe unexpected format
            send_exception_via_telegram(type(err_val), err_val, err_val.__traceback__)  # Notify via Telegram about unexpected format
            return (None, None, None)  # Return failure tuple due to unexpected format
        
        result = tuple(concat_result)  # Convert verified concat_result into tuple for safe unpacking
        combined_df, attack_types_list = result  # Unpack combined dataframe and attack types list after casting
        return (combined_df, attack_types_list, 'attack_type')  # Return combined dataframe, attack types list, and target column name
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def find_data_augmentation_file(original_file_path, config=None):
    """
    Find the corresponding data augmentation artifact for an original dataset file.
    Matches configured naming: <parent>/Data_Augmentation/Samples/<stem><augmentation suffix><format extension>.

    :param original_file_path: Path to the original CSV file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Path to the augmented file if it exists, None otherwise
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Looking for data augmentation file for: {BackgroundColors.CYAN}{original_file_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        data_augmentation_suffix = config.get("stacking", {}).get("data_augmentation_suffix", "_data_augmented")  # Read the augmentation-specific suffix without using generic result artifact naming
        original_path = Path(original_file_path)  # Create Path object from the original file path
        data_aug_dir = config.get("paths", {}).get("data_augmentation_dir", "Data_Augmentation")  # Use configured data augmentation base directory name
        data_aug_sample_dir = config.get("paths", {}).get("data_augmentation_sample_dir", "Samples")  # Use configured augmented samples subdirectory name
        augmented_dir = original_path.parent / data_aug_dir / data_aug_sample_dir  # Build data augmentation samples subdirectory path using config
        augmentation_format = config.get("stacking", {}).get("augmentation_file_format", "csv")  # Read configured augmentation file format
        augmentation_extension = resolve_format_extension(augmentation_format)  # Resolve format string to file extension
        augmented_filename = f"{original_path.stem}{data_augmentation_suffix}{augmentation_extension}"  # Build augmented filename with configured extension
        augmented_file = augmented_dir / augmented_filename  # Construct the full augmented file path

        if augmented_file.is_file():  # Verify the configured augmented artifact is a regular file
            verbose_output(
                f"{BackgroundColors.GREEN}Found augmented file: {BackgroundColors.CYAN}{augmented_file}{Style.RESET_ALL}",
                config=config
            )  # Output success message with the found path
            return str(augmented_file)  # Return the augmented file path as a string

        verbose_output(
            f"{BackgroundColors.YELLOW}No augmented file found for: {BackgroundColors.CYAN}{original_file_path}{BackgroundColors.YELLOW}. Expected: {BackgroundColors.CYAN}{augmented_file}{Style.RESET_ALL}",
            config=config
        )  # Output warning with expected path for debugging
        return None  # Return None when no augmented file is found
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_augmented_files_for_combined_evaluation(original_files_list, config=None):
    """
    Load augmented data files corresponding to original files for combined files evaluation mode.
    
    :param original_files_list: List of original dataset CSV file paths
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: List of augmented file paths (None entries where augmented file not found)
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG as fallback

        verbose_output(  # Emit verbose startup message for combined files evaluation augmentation loading
            f"{BackgroundColors.GREEN}Loading augmented files for combined files evaluation mode: {BackgroundColors.CYAN}{len(original_files_list)} files{Style.RESET_ALL}",
            config=config,
        )

        augmented_files = []  # Prepare list to preserve alignment with originals (None = missing)
        found_count = 0  # Counter for how many augmented files were found
        total_originals = len(original_files_list)  # Compute total count for progress

        for idx, original_file in enumerate(original_files_list, start=1):  # Iterate originals with index for progress
            augmented_file = find_data_augmentation_file(original_file, config=config)  # Locate augmented file path or None
            if augmented_file is not None:  # If an augmented file exists for this original
                augmented_files.append(augmented_file)  # Append the found augmented file path
                found_count += 1  # Increment found counter
            else:  # If no augmented file exists for this original
                verbose_output(  # Emit a per-file informative warning to verbose output including progress
                    f"{BackgroundColors.YELLOW}No augmented file found for: {BackgroundColors.CYAN}[{idx}/{total_originals}] {original_file}{Style.RESET_ALL}",
                    config=config,
                )  # Output missing-augmented-file warning with progress indicator
                augmented_files.append(None)  # Preserve index alignment with None placeholder

        if found_count == 0:  # If none were found across all originals
            print(  # Print a consolidated warning about missing augmented files
                f"{BackgroundColors.YELLOW}No augmented files found for any original files in combined files evaluation mode.{Style.RESET_ALL}"
            )
        else:  # If at least one augmented file was found
            print(  # Print a consolidated summary of found augmented files
                f"{BackgroundColors.GREEN}Found {BackgroundColors.CYAN}{found_count}/{len(original_files_list)}{BackgroundColors.GREEN} augmented files for combined files evaluation mode.{Style.RESET_ALL}"
            )

        return augmented_files  # Return the list preserving None placeholders for missing entries
    except Exception as e:  # On any exception, ensure logging and notification then re-raise
        print(str(e))  # Print the exception string to terminal logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send the exception via Telegram with traceback
        raise  # Re-raise the original exception to preserve semantics


def merge_original_and_augmented(original_df, augmented_df, config=None):
    """
    Merge original and augmented dataframes by concatenating them.

    :param original_df: Original DataFrame
    :param augmented_df: Augmented DataFrame
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Merged DataFrame
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Merging original ({len(original_df)} rows) and augmented ({len(augmented_df)} rows) data{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        merged_df = pd.concat([original_df, augmented_df], ignore_index=True)  # Concatenate dataframes
        
        verbose_output(
            f"{BackgroundColors.GREEN}Merged dataset has {BackgroundColors.CYAN}{len(merged_df)}{BackgroundColors.GREEN} rows{Style.RESET_ALL}",
            config=config
        )  # Output the result

        return merged_df  # Return merged dataframe
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_class_metrics(y_true, y_pred):
    """
    Extract per-class and global classification metrics without recomputing predictions.

    :param y_true: True labels (array-like)
    :param y_pred: Predicted labels (array-like)
    :return: dict with structure specified by requirements
    """
    
    try:
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=cast(Any, 0))  # Generate report dict

        if not isinstance(report, dict):  # Ensure report has mapping semantics
            raise TypeError("classification_report did not return a dict as expected")  # Defensive type verification

        per_class = {}  # Prepare per-class metrics mapping
        for k, v in report.items():  # Iterate over report items (class labels and metrics)

            if k in ("accuracy", "macro avg", "weighted avg", "micro avg"):  # Skip aggregate keys
                continue  # Move to next item

            per_class[k] = {  # Populate per-class metric dict converting values to float
                "precision": float(v.get("precision", 0.0)),
                "recall": float(v.get("recall", 0.0)),
                "f1": float(v.get("f1-score", v.get("f1", 0.0))),
            }

        global_metrics = {  # Compute deterministic global metrics
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=cast(Any, 0))),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=cast(Any, 0))),
            "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=cast(Any, 0))),
            "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=cast(Any, 0))),
        }

        return {"per_class": per_class, "global": global_metrics}  # Return structured metrics
    except Exception as e:  # On exception, log and notify then re-raise
        print(str(e))  # Print exception to terminal
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception details via Telegram
        raise  # Re-raise the exception to preserve original control flow
        return {"per_class": {}, "global": {}}


def ensure_figure_min_4k_and_save(fig=None, path=None, dpi=None, **kwargs):
    """
    Ensure a Matplotlib figure meets 4k minimum pixel dimensions and save it.

    :param fig: Matplotlib Figure instance or None to use current figure.
    :param path: Path where the image will be saved.
    :param dpi: DPI to use for saving; preserved if provided.
    :return: None
    """

    fig = fig or plt.gcf()  # Use current figure when none provided

    effective_dpi = dpi if dpi is not None else fig.get_dpi()  # Determine DPI for pixel computation

    width_inch, height_inch = fig.get_size_inches()  # Get current figure size in inches

    if width_inch * effective_dpi < 3840 or height_inch * effective_dpi < 2160:  # Verify if either dimension is below 4k
        new_w = max(width_inch, 3840.0 / effective_dpi)  # Compute required width in inches to reach 4k at effective DPI
        new_h = max(height_inch, 2160.0 / effective_dpi)  # Compute required height in inches to reach 4k at effective DPI
        fig.set_size_inches(new_w, new_h)  # Resize figure in inches while preserving DPI

    save_kwargs = dict(kwargs)  # Prepare kwargs for saving call

    if dpi is not None:  # Verify whether caller explicitly provided DPI
        save_kwargs["dpi"] = dpi  # Preserve caller DPI in save call

    resolved_fig = fig  # Ensure we close the exact figure used for saving
    if path is None:  # Verify that a valid path argument was provided
        raise ValueError("path must be provided to save the figure")  # Raise explicit error when path is missing to avoid passing None to savefig
    try:  # Save the figure to disk using provided kwargs
        resolved_fig.savefig(path, **save_kwargs)  # Save the figure to disk using provided kwargs
    finally:  # Ensure we close the figure to free memory regardless of save success
        plt.close(resolved_fig)  # Close the figure to free memory


def plot_per_class_metric(metric_dict: dict, metric_name: str, output_path: str, dpi: int, fmt: str) -> None:
    """
    Plot a per-class bar chart.

    :param metric_dict: mapping class_label -> {precision, recall, f1}
    :param metric_name: one of 'precision','recall','f1'
    :param output_path: full path to save file
    :param dpi: dpi integer > 0
    :param fmt: file format string
    """

    try:
        labels = sorted(metric_dict.keys(), key=lambda x: str(x))
        values = [float(metric_dict[l].get(metric_name, 0.0)) for l in labels]

        fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.4), 6))
        bars = ax.bar(labels, values, color="tab:blue", alpha=0.8)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel(metric_name.capitalize())
        ax.set_xlabel("Class")
        ax.set_title(f"Stacking - {metric_name.capitalize()} per Class")
        ax.grid(axis="y", alpha=0.25)

        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2.0, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        ensure_figure_min_4k_and_save(fig=fig, path=output_path, dpi=dpi, format=fmt, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        try:
            plt.close()
        except Exception:
            pass
        raise


def plot_global_metrics(global_metrics: dict, output_path: str, dpi: int, fmt: str) -> None:
    """
    Plot a deterministic ordered global metrics bar chart.
    Order: Accuracy, Macro F1, Weighted F1, Macro Precision, Macro Recall
    
    :param global_metrics: dict with keys like 'accuracy', 'macro_f1', etc.
    :param output_path: full path to save file
    :param dpi: dpi integer > 0
    :param fmt: file format string
    :return: None (saves plot to file)
    """
    
    try:
        order = ["accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"]
        labels = ["Accuracy", "Macro F1", "Weighted F1", "Macro Precision", "Macro Recall"]
        values = [float(global_metrics.get(k, 0.0)) for k in order]

        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(labels, values, color="tab:green", alpha=0.85)
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title("Stacking - Global Metrics")
        ax.grid(axis="y", alpha=0.25)

        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2.0, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()
        ensure_figure_min_4k_and_save(fig=fig, path=output_path, dpi=dpi, format=fmt, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        try:
            plt.close()
        except Exception:
            pass
        raise


def generate_and_save_metric_plots(y_true, y_pred, stacking_config: dict, resolved_results_dir: str) -> None:
    """
    Orchestrate extraction and saving of all required plots into
    <Feature_Analysis>/<stacking.results_dir>/Plots/

    :param y_true: true labels
    :param y_pred: predicted labels (must be the same predictions used to compute metrics)
    :param stacking_config: config dict for stacking (may contain 'plots')
    :param resolved_results_dir: absolute or project-relative path to stacking results directory
    """
    
    try:
        plots_cfg = stacking_config.get("plots", {}) if stacking_config is not None else {}
        enabled = plots_cfg.get("enabled", True)
        if not enabled:
            return

        dpi = plots_cfg.get("dpi", 1000)
        fmt = plots_cfg.get("format", "png")

        if not isinstance(dpi, int) or dpi <= 0:
            raise ValueError("plots.dpi must be an integer > 0")
        if fmt not in ("png", "jpg", "pdf", "svg"):
            raise ValueError("plots.format must be one of: png, jpg, pdf, svg")

        plots_dir = os.path.join(os.path.abspath(resolved_results_dir), config.get("paths", {}).get("plots_subdir", "Plots"))
        print(f"{BackgroundColors.GREEN}Saving metric plots to: {BackgroundColors.CYAN}{plots_dir}{Style.RESET_ALL}")

        validate_output_path(os.path.abspath(resolved_results_dir), os.path.abspath(plots_dir))
        os.makedirs(plots_dir, exist_ok=True)

        metrics = extract_class_metrics(y_true, y_pred)

        per = metrics.get("per_class", {})

        per_class_f1_path = os.path.join(plots_dir, f"per_class_f1.{fmt}")
        per_class_prec_path = os.path.join(plots_dir, f"per_class_precision.{fmt}")
        per_class_rec_path = os.path.join(plots_dir, f"per_class_recall.{fmt}")

        plot_per_class_metric(per, "f1", per_class_f1_path, dpi, fmt)
        plot_per_class_metric(per, "precision", per_class_prec_path, dpi, fmt)
        plot_per_class_metric(per, "recall", per_class_rec_path, dpi, fmt)

        global_path = os.path.join(plots_dir, f"global_metrics.{fmt}")
        plot_global_metrics(metrics.get("global", {}), global_path, dpi, fmt)

        verbose_output(f"[STACKING][PLOTS] Saved: {per_class_f1_path}, {per_class_prec_path}, {per_class_rec_path}, {global_path}")
    except Exception as e:
        verbose_output(f"{BackgroundColors.YELLOW}Plot generation failed: {e}{Style.RESET_ALL}")
        send_exception_via_telegram(type(e), e, e.__traceback__)
        return


def calculate_metric_improvement(original_value, augmented_value):
    """
    Calculate percentage improvement of a metric.

    :param original_value: Original metric value
    :param augmented_value: Augmented metric value
    :return: Percentage improvement (positive = better, negative = worse)
    """

    try:
        if original_value == 0:  # Avoid division by zero
            return 0.0 if augmented_value == 0 else float('inf')  # Return 0 if both are 0, inf if only original is 0
        
        improvement = ((augmented_value - original_value) / original_value) * 100  # Calculate percentage improvement
        return improvement  # Return improvement
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def generate_experiment_id(file_path, experiment_mode, augmentation_ratio=None):
    """
    Generates a unique experiment identifier for traceability in CSV results.

    :param file_path: Path to the dataset file being processed
    :param experiment_mode: Experiment mode string (e.g., 'original_only' or 'original_training_augmented_testing')
    :param augmentation_ratio: Augmentation ratio float (e.g., 0.25) or None for original-only mode
    :return: String experiment identifier combining timestamp, filename, mode and ratio
    """

    try:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Create a timestamp string for uniqueness
        path_is_directory = resolve_path_represents_directory(str(file_path))  # Resolve whether experiment identity comes from a directory.
        file_stem = build_filename_safe_dataset_identity(resolve_canonical_dataset_identity(str(file_path), True)) if path_is_directory else Path(file_path).stem  # Resolve experiment filename identity by path scope.
        ratio_tag = f"_ratio{int(augmentation_ratio * 100)}" if augmentation_ratio is not None else ""  # Build ratio tag string or empty
        experiment_id = f"{timestamp}_{file_stem}_{experiment_mode}{ratio_tag}"  # Concatenate all parts into unique identifier

        return experiment_id  # Return the generated experiment identifier
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def validate_augmented_dataframe(original_df, augmented_df, file_path):
    """
    Validates that the augmented DataFrame is compatible with the original for merging.

    :param original_df: Original cleaned DataFrame to compare against
    :param augmented_df: Augmented DataFrame to validate
    :param file_path: File path string for error message context
    :return: True if augmented data is valid and compatible, False otherwise
    """

    try:
        if augmented_df is None or augmented_df.empty:  # Verify if augmented DataFrame is None or contains no rows
            print(
                f"{BackgroundColors.YELLOW}Warning: Augmented DataFrame is empty for {BackgroundColors.CYAN}{file_path}{BackgroundColors.YELLOW}. Skipping.{Style.RESET_ALL}"
            )  # Print warning about empty augmented data
            return False  # Return False for empty augmented data

        original_cols = set(original_df.columns)  # Get the set of column names from the original DataFrame
        augmented_cols = set(augmented_df.columns)  # Get the set of column names from the augmented DataFrame
        missing_cols = original_cols - augmented_cols  # Compute columns present in original but missing in augmented

        if missing_cols:  # If there are columns missing from the augmented DataFrame
            print(
                f"{BackgroundColors.YELLOW}Warning: Augmented data for {BackgroundColors.CYAN}{file_path}{BackgroundColors.YELLOW} is missing columns: {BackgroundColors.CYAN}{missing_cols}{BackgroundColors.YELLOW}. Skipping.{Style.RESET_ALL}"
            )  # Print warning listing the missing column names
            return False  # Return False due to column mismatch

        original_dtypes = original_df.select_dtypes(include=np.number).columns.tolist()  # Get list of numeric columns in original
        augmented_dtypes = augmented_df.select_dtypes(include=np.number).columns.tolist()  # Get list of numeric columns in augmented
        numeric_overlap = set(original_dtypes) & set(augmented_dtypes)  # Compute intersection of numeric columns

        if len(numeric_overlap) < 2:  # Verify if there are at least 2 overlapping numeric columns (features + target)
            print(
                f"{BackgroundColors.YELLOW}Warning: Augmented data for {BackgroundColors.CYAN}{file_path}{BackgroundColors.YELLOW} has insufficient numeric column overlap ({len(numeric_overlap)}). Skipping.{Style.RESET_ALL}"
            )  # Print warning about insufficient numeric overlap
            return False  # Return False due to insufficient numeric columns

        verbose_output(
            f"{BackgroundColors.GREEN}Augmented data validation passed for {BackgroundColors.CYAN}{file_path}{BackgroundColors.GREEN}: {len(augmented_df)} rows, {len(augmented_cols)} columns{Style.RESET_ALL}"
        )  # Output verbose message confirming validation success

        return True  # Return True indicating augmented data is valid and compatible
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def sample_augmented_by_ratio(augmented_df, original_df, ratio):
    """
    Samples rows from the augmented DataFrame proportional to the original dataset size.

    :param augmented_df: Full augmented DataFrame to sample from
    :param original_df: Original DataFrame used to determine sample count
    :param ratio: Float ratio (e.g., 0.25 means 25% of original size)
    :return: Sampled DataFrame with at most ratio * len(original_df) rows, or None on failure
    """

    try:
        n_original = int(original_df) if isinstance(original_df, (int, np.integer)) else len(original_df)  # Resolve original row count from dataframe or precomputed count.
        n_requested = max(1, int(round(ratio * n_original)))  # Calculate requested sample size capped at minimum 1 row
        n_available = len(augmented_df)  # Get the total number of rows available in augmented data

        if n_available == 0:  # Verify if augmented DataFrame has zero rows
            print(
                f"{BackgroundColors.YELLOW}Warning: Augmented DataFrame is empty. Cannot sample at ratio {ratio}.{Style.RESET_ALL}"
            )  # Print warning about empty augmented source
            return None  # Return None for empty augmented data

        n_sample = min(n_requested, n_available)  # Cap the sample size at the available number of augmented rows

        if n_sample < n_requested:  # Log a warning if capping occurred (fewer augmented rows than requested)
            verbose_output(
                f"{BackgroundColors.YELLOW}Augmented data has only {n_available} rows; requested {n_requested} (ratio={ratio}). Using all {n_available}.{Style.RESET_ALL}"
            )  # Warn that fewer rows than requested are available

        sampled_df = augmented_df.sample(n=n_sample, random_state=42, replace=False)  # Randomly sample n_sample rows with fixed seed for reproducibility

        verbose_output(
            f"{BackgroundColors.GREEN}Sampled {BackgroundColors.CYAN}{n_sample}{BackgroundColors.GREEN} augmented rows at ratio {BackgroundColors.CYAN}{ratio}{BackgroundColors.GREEN} (original has {n_original} rows){Style.RESET_ALL}"
        )  # Output verbose message confirming sampling details

        return sampled_df  # Return the sampled augmented DataFrame
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_tsne_output_directory(original_file_path, augmented_file_path):
    """
    Build output directory path for t-SNE plots preserving nested dataset structure.

    :param original_file_path: Path to original dataset file
    :param augmented_file_path: Path to augmented dataset file
    :return: Path object for t-SNE output directory
    """

    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Building t-SNE output directory for: {BackgroundColors.CYAN}{original_file_path}{Style.RESET_ALL}"
        )  # Output verbose message for directory creation

        original_path = Path(original_file_path)  # Create Path object for original file
        augmented_path = Path(augmented_file_path)  # Create Path object for augmented file

        datasets_keyword = "Datasets"  # Standard directory name in project structure
        relative_parts = []  # List to accumulate relative path components
        found_datasets = False  # Flag to track if Datasets directory was found

        for part in original_path.parts:  # Iterate through path components
            if found_datasets and part != original_path.name:  # After Datasets, before filename
                relative_parts.append(part)  # Add intermediate directories to relative path
            if part == datasets_keyword:  # Found the Datasets directory
                found_datasets = True  # Set flag to start collecting relative parts

        augmented_parent = augmented_path.parent  # Get parent directory of augmented file
        tsne_base = augmented_parent / "tsne_plots"  # Base directory for all t-SNE plots

        if relative_parts:  # If nested structure exists
            tsne_dir = tsne_base / Path(*relative_parts) / original_path.stem  # Preserve nested path
        else:  # Flat structure
            tsne_dir = tsne_base / original_path.stem  # Use filename stem only

        os.makedirs(tsne_dir, exist_ok=True)  # Create directory structure if it doesn't exist

        verbose_output(
            f"{BackgroundColors.GREEN}Created t-SNE directory: {BackgroundColors.CYAN}{tsne_dir}{Style.RESET_ALL}"
        )  # Output confirmation message

        return tsne_dir  # Return the output directory path
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def combine_and_label_augmentation_data(original_df, augmented_df=None, label_col=None):
    """
    Combine original and augmented data with source labels for t-SNE visualization.

    :param original_df: DataFrame with original data
    :param augmented_df: DataFrame with augmented data (None for original-only)
    :param label_col: Name of the label/class column
    :return: Combined DataFrame with composite labels for visualization
    """

    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Combining and labeling data for t-SNE visualization...{Style.RESET_ALL}"
        )  # Output verbose message for data combination

        if label_col is None:  # No label column specified
            label_col = original_df.columns[-1]  # Use last column as label

        df_orig = original_df.copy()  # Copy original DataFrame to avoid modifying input

        if label_col in df_orig.columns:  # If label column exists
            df_orig['tsne_label'] = df_orig[label_col].astype(str) + "_original"  # Create composite label
        else:  # No label column found
            df_orig['tsne_label'] = "original"  # Use simple source label

        if augmented_df is not None:  # If augmented data provided
            df_aug = augmented_df.copy()  # Copy augmented DataFrame to avoid modifying input

            if label_col in df_aug.columns:  # If label column exists
                df_aug['tsne_label'] = df_aug[label_col].astype(str) + "_augmented"  # Create composite label
            else:  # No label column found
                df_aug['tsne_label'] = "augmented"  # Use simple source label

            combined_df = pd.concat([df_orig, df_aug], ignore_index=True)  # Concatenate DataFrames
        else:  # Original only
            combined_df = df_orig  # Use original DataFrame only

        return combined_df  # Return combined DataFrame with composite labels
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def prepare_numeric_features_for_tsne(df, exclude_col='tsne_label'):
    """
    Extract and prepare numeric features from DataFrame for t-SNE.

    :param df: DataFrame with mixed features
    :param exclude_col: Column name to exclude from numeric extraction
    :return: Tuple (numeric_array, labels_array, success_flag)
    """

    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Preparing numeric features for t-SNE...{Style.RESET_ALL}"
        )  # Output verbose message for feature preparation

        if exclude_col in df.columns:  # If label column exists
            labels = df[exclude_col].values  # Extract labels as numpy array
            df_work = df.drop(columns=[exclude_col])  # Remove label column for numeric extraction
        else:  # No label column
            labels = np.array(['unknown'] * len(df))  # Create default labels
            df_work = df.copy()  # Use full DataFrame

        numeric_df = df_work.select_dtypes(include=np.number)  # Select only numeric columns

        if numeric_df.empty:  # No numeric columns found
            print(
                f"{BackgroundColors.YELLOW}Warning: No numeric features found for t-SNE generation.{Style.RESET_ALL}"
            )  # Print warning message
            return (None, None, False)  # Return failure tuple

        numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)  # Replace infinities with NaN
        numeric_df = numeric_df.fillna(numeric_df.median())  # Fill NaN with column median
        numeric_df = numeric_df.fillna(0)  # Fill remaining NaN with zero

        if numeric_df.shape[0] == 0 or numeric_df.shape[1] == 0:  # Empty result after cleaning
            print(
                f"{BackgroundColors.YELLOW}Warning: No valid numeric data remaining after cleaning.{Style.RESET_ALL}"
            )  # Print warning message
            return (None, None, False)  # Return failure tuple

        X = numeric_df.values  # Extract values as numpy array

        try:  # Attempt feature scaling
            scaler = StandardScaler()  # Initialize standard scaler
            X_scaled = scaler.fit_transform(X)  # Scale features to zero mean and unit variance
        except Exception as e:  # Scaling failed
            print(
                f"{BackgroundColors.YELLOW}Warning: Feature scaling failed: {e}. Using unscaled data.{Style.RESET_ALL}"
            )  # Print warning message
            X_scaled = X  # Use unscaled data as fallback

        return (X_scaled, labels, True)  # Return scaled features, labels, and success flag
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def compute_and_save_tsne_plot(X_scaled, labels, output_path, title, perplexity=30, random_state=42):
    """
    Compute t-SNE embedding and save visualization plot.

    :param X_scaled: Scaled numeric feature array
    :param labels: Array of labels for coloring
    :param output_path: Full path for saving the plot file
    :param title: Title for the plot
    :param perplexity: t-SNE perplexity parameter
    :param random_state: Random seed for reproducibility
    :return: True if successful, False otherwise
    """

    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Computing t-SNE embedding and saving plot...{Style.RESET_ALL}"
        )  # Output verbose message for t-SNE computation

        max_perplexity = (X_scaled.shape[0] - 1) // 3  # Maximum valid perplexity
        actual_perplexity = min(perplexity, max_perplexity)  # Use minimum of requested and maximum

        if actual_perplexity < 5:  # Perplexity too small for meaningful results
            print(
                f"{BackgroundColors.YELLOW}Warning: Sample size too small for t-SNE (n={X_scaled.shape[0]}).{Style.RESET_ALL}"
            )  # Print warning message
            return False  # Return failure flag

        tsne = TSNE(
            n_components=2,  # 2D embedding for visualization
            perplexity=actual_perplexity,  # Adjusted perplexity parameter
            random_state=random_state,  # Random seed for reproducibility
            max_iter=1000  # Number of iterations
        )  # Create t-SNE object

        X_embedded = tsne.fit_transform(X_scaled)  # Compute 2D embedding

        plt.figure(figsize=(12, 10))  # Create figure with specified size

        unique_labels = np.unique(labels)  # Extract unique label values
        n_labels = len(unique_labels)  # Count unique labels

        cmap = plt.cm.get_cmap("rainbow")  # Get rainbow colormap for distinct colors
        colors = cmap(np.linspace(0, 1, n_labels))  # Generate distinct colors from colormap

        for idx, label in enumerate(unique_labels):  # Iterate over unique labels
            mask = labels == label  # Create boolean mask for current label
            plt.scatter(
                X_embedded[mask, 0],  # X coordinates for current class
                X_embedded[mask, 1],  # Y coordinates for current class
                c=[colors[idx]],  # Color for current class
                label=label,  # Legend label
                alpha=0.6,  # Transparency
                edgecolors='k',  # Black edge color
                linewidth=0.5,  # Edge line width
                s=50  # Marker size
            )  # Plot scatter points for current class

        plt.title(title, fontsize=16, fontweight='bold')  # Set plot title
        plt.xlabel('t-SNE Component 1', fontsize=12)  # Set x-axis label
        plt.ylabel('t-SNE Component 2', fontsize=12)  # Set y-axis label
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)  # Add legend outside plot
        plt.grid(True, alpha=0.3)  # Add grid with transparency
        plt.tight_layout()  # Adjust layout to prevent label cutoff

        ensure_figure_min_4k_and_save(fig=plt.gcf(), path=output_path, dpi=300, bbox_inches='tight')  # Save figure with high resolution
        plt.close()  # Close figure to free memory

        print(
            f"{BackgroundColors.GREEN}t-SNE plot saved to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
        )  # Print success message

        return True  # Return success flag

    except Exception as e:  # t-SNE computation or plotting failed
        print(
            f"{BackgroundColors.RED}Error generating t-SNE plot: {e}{Style.RESET_ALL}"
        )  # Print error message
        send_exception_via_telegram(type(e), e, e.__traceback__)
        return False  # Return failure flag


def generate_augmentation_tsne_visualization(original_file, original_df, augmented_df=None, augmentation_ratio=None, experiment_mode="original_only"):
    """
    Generate t-SNE visualization for data augmentation experiment.

    :param original_file: Path to original dataset file
    :param original_df: DataFrame with original data
    :param augmented_df: DataFrame with augmented data (None for original-only)
    :param augmentation_ratio: Augmentation ratio (e.g., 0.50 for 50%)
    :param experiment_mode: Experiment mode string
    :return: None
    """

    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Generating t-SNE visualization for augmentation experiment...{Style.RESET_ALL}"
        )  # Output verbose message for t-SNE generation

        augmented_file = find_data_augmentation_file(original_file)  # Locate augmented data file
        if augmented_file is None:  # No augmented file found
            print(
                f"{BackgroundColors.YELLOW}Warning: Cannot generate t-SNE - augmented file path not found.{Style.RESET_ALL}"
            )  # Print warning message
            return  # Exit function early

        stacking_output_dir = get_stacking_output_dir(original_file, CONFIG)
        tsne_output_dir = Path(stacking_output_dir) / "Plots" / "tsne_plots" / Path(original_file).stem
        tsne_output_dir.mkdir(parents=True, exist_ok=True)

        combined_df = combine_and_label_augmentation_data(original_df, augmented_df)  # Prepare labeled data

        X_scaled, labels, success = prepare_numeric_features_for_tsne(combined_df, exclude_col='tsne_label')  # Extract and scale features

        if not success:  # Feature preparation failed
            print(
                f"{BackgroundColors.YELLOW}Warning: Skipping t-SNE generation due to feature preparation failure.{Style.RESET_ALL}"
            )  # Print warning message
            return  # Exit function early

        file_stem = Path(original_file).stem  # Extract filename without extension

        if experiment_mode == "original_only":  # Original-only experiment
            plot_filename = f"{file_stem}_original_only_tsne.png"  # Filename for original-only plot
            plot_title = f"t-SNE: {file_stem} (Original Only)"  # Title for original-only plot
        else:  # Original versus augmented testing experiment
            ratio_pct = int(augmentation_ratio * 100) if augmentation_ratio else 0  # Convert ratio to percentage
            plot_filename = f"{file_stem}_augmented_{ratio_pct}pct_tsne.png"  # Filename for augmented plot
            plot_title = f"t-SNE: {file_stem} (Original vs {ratio_pct}% Augmented Test)"  # Title for augmented-testing plot

        output_path = tsne_output_dir / plot_filename  # Build full output path

        compute_and_save_tsne_plot(X_scaled, labels, str(output_path), plot_title)  # Generate and save visualization

    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def zebra(row):
    """
    Apply zebra-striping style to a DataFrame row for visual clarity in exported images.
    
    :param row: pandas Series representing a DataFrame row
    :return: List of CSS styles for each column in the row, alternating background colors
    """
    
    return ["background-color: #ffffff" if i % 2 == 0 else "background-color: #f2f2f2" for i in range(len(row))]  # Return CSS list per column


def apply_zebra_style(df):
    """
    Apply zebra-striping style to a pandas DataFrame using Styler.

    :param df: pandas DataFrame to style
    :return: pandas.io.formats.style.Styler styled object ready for export
    """

    try:

        styled = df.style.apply(zebra, axis=1)  # Create Styler with zebra striping applied
        return styled  # Return styled object
    except Exception as e:  # If styling fails
        raise  # Propagate exception without swallowing


def is_playwright_installed() -> bool:
    """
    Verify if Playwright package is importable.

    :return: Boolean indicating whether Playwright is installed.
    """

    try:  # Attempt dynamic import of Playwright to verify installation
        importlib.import_module("playwright")  # Try to import the Playwright package
        return True  # Return True when Playwright is importable
    except Exception:  # Handle import failure gracefully
        return False  # Return False when Playwright is not available


def install_playwright_if_missing() -> None:
    """
    Install Playwright package into the active Python interpreter when missing.

    :return: None
    """

    try:  # Wrap installation attempt in try/except to avoid interrupting pipeline
        if is_playwright_installed():  # Verify if Playwright already installed
            return  # Return early when Playwright is already present

        subprocess.run([sys.executable, "-m", "pip", "install", "playwright"], check=False)  # Install Playwright using the active interpreter
        return  # Return after attempting installation
    except Exception as e:  # Handle any errors during Playwright installation
        print(str(e))  # Print the exception string for diagnostics
        try:  # Attempt to notify via Telegram if configured
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception via Telegram hook
        except Exception:  # Swallow any Telegram send errors silently
            pass  # No-op when Telegram notification fails


def install_playwright_chromium() -> None:
    """
    Install Playwright Chromium browser using the current Python interpreter.

    :return: None
    """

    try:  # Attempt to install Chromium using Playwright module invocation
        subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=False)  # Invoke Playwright install via the active interpreter
        return  # Return after attempting Chromium installation
    except Exception as e:  # Handle installation errors without raising
        print(str(e))  # Print the exception string for diagnostics
        try:  # Attempt to notify via Telegram if configured
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception via Telegram hook
        except Exception:  # Swallow any Telegram send errors silently
            pass  # No-op when Telegram notification fails


def ensure_playwright_chromium() -> None:
    """
    Ensure Playwright and Chromium are installed for PNG exports.

    :return: None
    """

    try:  # Orchestrate Playwright installation steps safely
        install_playwright_if_missing()  # Install Playwright into the active interpreter when needed
        install_playwright_chromium()  # Install Chromium via Playwright using the active interpreter
        return  # Return after attempting orchestration
    except Exception as e:  # Handle unexpected orchestration errors gracefully
        print(str(e))  # Print the exception string for diagnostics
        try:  # Attempt to notify via Telegram if configured
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception via Telegram hook
        except Exception:  # Swallow any Telegram send errors silently
            pass  # No-op when Telegram notification fails


def export_dataframe_image(styled_df, output_path):
    """
    Export a pandas Styler object to a PNG file using dataframe_image.

    :param styled_df: pandas Styler object to export
    :param output_path: Path to PNG file to write
    :return: None
    """

    try:
        out_path = Path(output_path)  # Create Path object for output
        parent = out_path.parent  # Get parent directory
        if not parent.exists():  # Ensure parent exists
            parent.mkdir(parents=True, exist_ok=True)  # Create parent directories if missing

        if not os.access(str(parent), os.W_OK):  # Verify write permission on parent
            raise PermissionError(f"Directory not writable: {parent}")  # Raise if not writable

        ensure_playwright_chromium()  # Ensure Playwright Chromium is installed before exporting PNG
        dfi.export(styled_df, str(out_path))  # Use dataframe_image to export styled DataFrame to PNG
        return  # Return None explicitly
    except Exception:
        raise  # Propagate any exception (do not swallow)


def generate_table_image_from_dataframe(df, csv_path):
    """
    Generate a zebra-striped PNG table image for a DataFrame, saved beside the CSV.

    :param df: pandas DataFrame in memory to render
    :param csv_path: Path to the CSV file (image will be same base name with .png)
    :return: Path to generated PNG image
    """
    try:
        png_path = str(Path(csv_path).with_suffix('.png'))  # Build PNG path by replacing extension
        styled = apply_zebra_style(df)  # Apply zebra styling to DataFrame
        export_dataframe_image(styled, png_path)  # Export styled DataFrame as PNG image
        return png_path  # Return path to generated PNG
    except Exception:
        raise  # Propagate exceptions to caller


def generate_csv_and_image(df, csv_path, is_visualizable=True, config=None, **to_csv_kwargs):
    """
    Save DataFrame to CSV and optionally generate a zebra-striped PNG image beside it.

    :param df: pandas DataFrame to save
    :param csv_path: Destination CSV file path
    :param is_visualizable: Whether to generate the PNG image (default True)
    :param to_csv_kwargs: Additional keyword args forwarded to pandas.DataFrame.to_csv
    :return: Path to saved CSV file
    """
    
    try:
        out_path = Path(csv_path).resolve()  # Resolve to absolute path to normalize any traversal sequences
        base = str(out_path.parent)  # Derive safe base from the resolved output file's own parent directory
        validate_output_path(base, str(out_path))  # Validate target is within its own resolved parent directory

        df.to_csv(csv_path, **to_csv_kwargs)  # Save DataFrame to CSV with original kwargs
        if is_visualizable and len(df) <= 100:  # Generate image only when visualizable and within the safe row limit
            try:  # Guard PNG rendering to keep CSV persistence independent from image export
                generate_table_image_from_dataframe(df, csv_path)  # Generate PNG from in-memory DataFrame
            except Exception as _png_e:  # Contain PNG rendering failures locally
                print(f"{BackgroundColors.YELLOW}Warning: PNG generation failed for {Path(csv_path).name}: {_png_e}{Style.RESET_ALL}")  # Warn and continue without propagating PNG errors
        return csv_path  # Return CSV path
    except Exception:
        raise  # Propagate exceptions (no silent failures)


def aggregate_feature_usage(results_df, top_n=None):
    """
    Aggregate feature usage counts from a results DataFrame.

    :param results_df: DataFrame containing at least `features_list` and `model` columns
    :param top_n: Optional integer for number of top features to return (None => use config)
    :return: DataFrame indexed by feature with columns per model and a `total` column, limited to top-N features
    """
    
    try:
        if results_df is None or results_df.empty:  # Validate input presence
            raise ValueError("results_df is empty or None")  # Signal caller about empty input

        cfg_top_n = CONFIG.get("stacking", {}).get("top_n_features_heatmap", 15)  # Read default from config
        n = top_n if top_n is not None else cfg_top_n  # Determine effective top-N
        n = int(n)  # Ensure integer type for downstream logic

        counts = {}  # Dictionary to accumulate counts per model per feature

        for _, row in results_df.iterrows():  # Iterate rows to parse feature lists
            raw = row.get("features_list", None)  # Extract raw features_list value
            model_name = row.get("model", "AllModels")  # Extract model identifier or fallback

            if pd.isna(raw) or raw is None:  # Skip empty feature lists
                continue  # Nothing to count for this row

            features = []  # Prepare list of parsed feature names

            if isinstance(raw, str) and raw.strip().startswith("["):  # JSON-array string
                try:
                    features = json.loads(raw)  # Parse JSON list into Python list
                except Exception:
                    features = [f.strip() for f in raw.strip().strip('[]').split(",") if f.strip()]  # Fallback split
            elif isinstance(raw, str):
                features = [f.strip() for f in raw.split(",") if f.strip()]  # Comma-separated string parsing
            elif isinstance(raw, (list, tuple)):  # Already a list/tuple
                features = list(raw)  # Normalize to list
            else:
                features = [str(raw)]  # Convert unexpected types to single-item list

            for feat in features:  # Count each feature occurrence
                if feat == "":  # Skip empty strings
                    continue  # Ignore empty tokens
                counts.setdefault(feat, {}).setdefault(model_name, 0)  # Ensure nested dict entry exists
                counts[feat][model_name] += 1  # Increment count for this feature-model pair

        if not counts:  # If still empty after processing
            return pd.DataFrame()  # Return empty DataFrame to caller

        feature_index = sorted(list(counts.keys()))  # Sorted feature list for deterministic ordering
        model_names = sorted({m for feat in counts.values() for m in feat.keys()})  # Sorted model list

        matrix = []  # Rows for building DataFrame
        for feat in feature_index:  # Build each row of counts
            row_vals = [counts[feat].get(m, 0) for m in model_names]  # Get counts per model with zero fill
            matrix.append(row_vals)  # Append the row values

        df_counts = pd.DataFrame(matrix, index=pd.Index(feature_index), columns=pd.Index(model_names))  # Build counts DataFrame
        df_counts["total"] = df_counts.sum(axis=1)  # Compute total occurrences across models
        df_counts = df_counts.sort_values("total", ascending=False)  # Sort by total descending for top-N selection

        top_n_effective = min(n, len(df_counts))  # Cap top-N to available features
        return df_counts.head(top_n_effective)  # Return top-N slice
    except Exception:
        raise  # Propagate exceptions to caller (no silent failures)


def export_top_features_csv(feature_counts_df, csv_path, dataset_file=None):
    """
    Export aggregated top-N feature counts to CSV.

    :param feature_counts_df: DataFrame produced by aggregate_feature_usage()
    :param csv_path: Destination CSV path
    :return: Path to saved CSV file
    """
    
    try:
        if feature_counts_df is None or feature_counts_df.empty:  # Validate presence
            feature_counts_df = pd.DataFrame()  # Create empty DataFrame for consistent output

        out_path = Path(csv_path)  # Normalize to Path object
        out_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directory exists
        if dataset_file is None:
            raise RuntimeError("dataset_file must be provided to safely resolve stacking results directory")
        base = get_stacking_output_dir(dataset_file, CONFIG)
        validate_output_path(base, str(out_path))
        feature_counts_df.to_csv(str(out_path), index=True)  # Write DataFrame to CSV including index (feature names)
        return str(out_path)  # Return string path to caller
    except Exception:
        raise  # Propagate exceptions


def generate_feature_usage_heatmap(feature_counts_df, output_path, dataset_file=None):
    """
    Generate and save a heatmap PNG from feature counts DataFrame.

    :param feature_counts_df: DataFrame indexed by feature with per-model columns and a `total` column
    :param output_path: Destination PNG path
    :return: Path to saved PNG file
    """
    
    try:
        if feature_counts_df is None or feature_counts_df.empty:  # If no data, create a minimal empty heatmap
            fig, ax = plt.subplots(figsize=(6, 2))  # Create small figure for empty state
            ax.text(0.5, 0.5, "No features available", ha="center", va="center")  # Informative message
            ax.set_axis_off()  # Hide axes for empty message
            out_png = str(Path(output_path))  # Compute output path
            Path(out_png).parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
            if dataset_file is None:
                raise RuntimeError("dataset_file must be provided to safely resolve stacking results directory")
            base = get_stacking_output_dir(dataset_file, CONFIG)
            validate_output_path(base, out_png)
            ensure_figure_min_4k_and_save(fig=fig, path=out_png, bbox_inches="tight")  # Save PNG to disk
            plt.close(fig)  # Close the figure to free memory
            return out_png  # Return path to generated PNG

        data = feature_counts_df.drop(columns=[c for c in feature_counts_df.columns if c == "total"], errors="ignore")  # Drop total for heatmap columns
        data = data.astype(int)  # Ensure integer type for plotting annotations

        n_rows = data.shape[0]  # Number of features (rows)
        n_cols = data.shape[1]  # Number of models (columns)
        height = max(2, 0.4 * n_rows)  # Heuristic height for readability
        width = max(4, 0.8 * n_cols)  # Heuristic width for readability

        fig, ax = plt.subplots(figsize=(width, height))  # Create figure with computed size
        sns.heatmap(data, annot=True, fmt="d", cmap="YlGnBu", cbar=True, ax=ax)  # Draw annotated heatmap
        ax.set_ylabel("")  # Keep y-axis label minimal
        ax.set_xlabel("")  # Keep x-axis label minimal
        plt.yticks(rotation=0)  # Ensure feature labels horizontal for readability
        plt.xticks(rotation=45, ha="right")  # Rotate column labels for space
        out_png = str(Path(output_path))  # Normalize output path to string
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directory exists
        plt.tight_layout()  # Tighten layout before saving
        ensure_figure_min_4k_and_save(fig=fig, path=out_png, dpi=300)  # Save figure to PNG with good resolution
        plt.close(fig)  # Close the figure to release resources
        return out_png  # Return saved PNG path
    except Exception:
        raise  # Propagate exceptions to caller


def save_augmentation_comparison_results(file_path, comparison_results, config=None):
    """
    Save data augmentation comparison results to CSV file.

    :param file_path: Path to the original CSV file being processed
    :param comparison_results: List of dictionaries containing comparison metrics
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        augmentation_comparison_filename = config.get("stacking", {}).get("augmentation_comparison_filename", "Data_Augmentation_Comparison_Results.csv")  # Get filename from config

        if not comparison_results:  # If no results to save
            return  # Exit early

        file_path_obj = Path(file_path)  # Create Path object
        feature_analysis_dir = file_path_obj.parent / "Feature_Analysis"  # Feature_Analysis directory
        os.makedirs(feature_analysis_dir, exist_ok=True)  # Ensure directory exists
        output_path = feature_analysis_dir / augmentation_comparison_filename  # Output file path

        df = pd.DataFrame(comparison_results)  # Convert results to DataFrame

        column_order = [
            "dataset",
            "feature_set",
            "hyperparameter_mode",
            "classifier_type",
            "model_name",
            "data_source",
            "experiment_id",
            "experiment_mode",
            "augmentation_ratio",
            "n_features",
            "n_samples_train",
            "n_samples_test",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "fpr",
            "fnr",
            "training_time",
            "accuracy_improvement",
            "precision_improvement",
            "recall_improvement",
            "f1_score_improvement",
            "fpr_improvement",
            "fnr_improvement",
            "training_time_improvement",
            "features_list",
            "Hardware",
        ]  # Define desired column order with experiment traceability columns

        existing_columns = [col for col in column_order if col in df.columns]  # Filter to existing columns
        df = df[existing_columns]  # Reorder DataFrame columns

        df = add_hardware_column(df, existing_columns)  # Add hardware specifications column

        generate_csv_and_image(df, output_path, is_visualizable=True, index=False)  # Save CSV and generate PNG image
        print(
            f"{BackgroundColors.GREEN}Saved augmentation comparison results to {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
        )  # Output success message
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def find_local_feature_file(file_dir, filename, config=None):
    """
    Attempt to locate <file_dir>/Feature_Analysis/<filename>.

    :param file_dir: Directory to search within
    :param filename: Filename to search for
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: The matching path or None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Verifying local Feature_Analysis/ in directory: {BackgroundColors.CYAN}{file_dir}{BackgroundColors.GREEN} for file: {BackgroundColors.CYAN}{filename}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        candidate = os.path.join(file_dir, "Feature_Analysis", filename)  # Construct candidate path

        if verify_filepath_exists(candidate):  # If the candidate file exists
            return candidate  # Return the candidate path

        return None  # Not found
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def find_parent_feature_file(file_dir, filename, config=None):
    """
    Ascend parent directories searching for <parent>/Feature_Analysis/<filename>.

    :param file_dir: Directory to search within
    :param filename: Filename to search for
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: The matching path or None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Ascending parent directories from: {BackgroundColors.CYAN}{file_dir}{BackgroundColors.GREEN} searching for file: {BackgroundColors.CYAN}{filename}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        path = file_dir  # Start from the file's directory
        while True:  # Loop until break
            candidate = os.path.join(path, "Feature_Analysis", filename)  # Construct candidate path
            if verify_filepath_exists(candidate):  # If the candidate file exists
                return candidate  # Return the candidate path

            parent = os.path.dirname(path)  # Get the parent directory
            if parent == path:  # If reached the root directory
                break  # Break the loop

            path = parent  # Move up to the parent directory

        return None  # Not found
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def find_dataset_level_feature_file(file_path, filename, config=None):
    """
    Try dataset-level search:

    - /.../Datasets/<dataset_name>/Feature_Analysis/<filename>
    - recursive search under dataset directory

    :param file_path: Path to the file
    :param filename: Filename to search for
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: The matching path or None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Searching dataset-level Feature_Analysis for file: {BackgroundColors.CYAN}{filename}{BackgroundColors.GREEN} related to file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        abs_path = os.path.abspath(file_path)  # Get absolute path of the input file
        parts = abs_path.split(os.sep)  # Split the path into parts

        if "Datasets" not in parts:  # If "Datasets" is not in the path parts
            return None  # Nothing to do

        idx = parts.index("Datasets")  # Get the index of "Datasets"
        if idx + 1 >= len(parts):  # If there is no dataset name after "Datasets"
            return None  # Nothing to do

        dataset_dir = os.sep.join(parts[: idx + 2])  # Construct the dataset directory path

        candidate = os.path.join(dataset_dir, "Feature_Analysis", filename)  # Construct candidate path for the direct path
        if verify_filepath_exists(candidate):  # If the candidate file exists
            return candidate  # Return the candidate path

        matches = glob.glob(
            os.path.join(dataset_dir, "**", "Feature_Analysis", filename), recursive=True
        )  # Search recursively
        if matches:  # If matches are found
            return matches[0]  # Return the first match

        return None  # Not found
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def find_feature_file(file_path, filename, config=None):
    """
    Locate a feature-analysis CSV file related to `file_path`.

    Search order:
    - <file_dir>/Feature_Analysis/<filename>
    - ascend parent directories verifying <parent>/Feature_Analysis/<filename>
    - dataset-level folder under `.../Datasets/<dataset_name>/Feature_Analysis/<filename>`
    - fallback: search under workspace ./Datasets/**/Feature_Analysis/<filename>

    :param file_path: Path to the file
    :param filename: Filename to search for
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: The matching path or None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Searching for feature analysis file: {BackgroundColors.CYAN}{filename}{BackgroundColors.GREEN} related to file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        file_dir = os.path.dirname(os.path.abspath(file_path))  # Get the directory of the input file

        result = find_local_feature_file(file_dir, filename, config=config)  # 1. Local Feature_Analysis in the same directory
        if result is not None:  # If found
            return result  # Return the result

        result = find_parent_feature_file(file_dir, filename, config=config)  # 2. Ascend parents verifying for Feature_Analysis
        if result is not None:  # If found
            return result  # Return the result

        result = find_dataset_level_feature_file(file_path, filename, config=config)  # 3. Dataset-level Feature_Analysis
        if result is not None:  # If found
            return result  # Return the result

        print(
            f"{BackgroundColors.YELLOW}Warning: Feature analysis file {BackgroundColors.CYAN}{filename}{BackgroundColors.YELLOW} not found for dataset containing {BackgroundColors.CYAN}{file_path}{BackgroundColors.YELLOW}.{Style.RESET_ALL}"
        )  # Output the warning message

        return None  # Return None if not found
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_genetic_algorithm_features(file_path, config=None):
    """
    Extracts the features selected by the Genetic Algorithm from the corresponding
    "Genetic_Algorithm_Results.csv" file located in the "Feature_Analysis"
    subdirectory relative to the input file's directory.

    It specifically retrieves the 'best_features' (a JSON string) from the row
    where the 'run_index' is 'best', and returns it as a Python list.

    :param file_path: Full path to the current CSV file being processed (e.g., "./Datasets/.../DrDoS_DNS.csv").
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: List of features selected by the GA, or None if the file is not found or fails to load/parse.
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        file_dir = os.path.dirname(file_path)  # Determine the directory of the input file
        verbose_output(
            f"{BackgroundColors.GREEN}Extracting GA features for file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        ga_results_path = find_feature_file(file_path, "Genetic_Algorithm/Genetic_Algorithm_Results.csv", config=config)  # Find the GA results file
        if ga_results_path is None:  # If the GA results file does not exist
            print(
                f"{BackgroundColors.YELLOW}Warning: GA results file not found for dataset containing {BackgroundColors.CYAN}{file_path}{BackgroundColors.YELLOW}. Skipping GA feature extraction for this file.{Style.RESET_ALL}"
            )
            return None  # Return None if the file does not exist

        print(f"{BackgroundColors.GREEN}[INFO] GA feature file found: {BackgroundColors.CYAN}{ga_results_path}{Style.RESET_ALL}")  # Log the resolved GA results file path

        try:  # Try to load the GA results
            low_memory = config.get("execution", {}).get("low_memory", False)  # Read low memory flag from config
            df = pd.read_csv(ga_results_path, low_memory=low_memory)  # Load the full GA results schema to support robust fallback selection
            df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names

            if "best_features" not in df.columns:  # Validate required feature payload column presence before row selection
                print(
                    f"{BackgroundColors.RED}Error: 'best_features' column not found in GA results file at {BackgroundColors.CYAN}{ga_results_path}{Style.RESET_ALL}"
                )  # Report schema mismatch when mandatory best_features column is missing
                return None  # Return None when GA results schema cannot provide selected features

            selected_row = None  # Initialize selected row as None before applying selection rules
            selected_source = ""  # Track selected-row strategy for diagnostic logging

            if "run_index" in df.columns:  # Prefer explicit best-run marker when run_index column exists
                run_index_norm = df["run_index"].fillna("").astype(str).str.strip().str.lower()  # Normalize run_index values for resilient matching
                best_rows = df[run_index_norm == "best"]  # Select rows explicitly marked as best run
                if not best_rows.empty:  # Continue only when at least one explicit best row exists
                    selected_row = best_rows.iloc[0]  # Select the first explicit best row preserving legacy behavior
                    selected_source = "run_index=best"  # Record legacy selection source for diagnostics

            if selected_row is None:  # Apply fallback logic only when explicit best run marker is unavailable
                best_features_norm = df["best_features"].fillna("").astype(str).str.strip()  # Normalize best_features payload for emptiness filtering
                candidate_rows = df[best_features_norm != ""]  # Keep only rows that contain a non-empty best_features payload

                metric_column = None  # Initialize metric column name used for fallback ranking
                for candidate_metric in ["cv_f1_score", "test_f1_score", "f1_score", "cv_fnr", "test_fnr", "cv_fpr", "test_fpr"]:  # Try known GA metric fields in priority order
                    if candidate_metric in candidate_rows.columns:  # Use first metric field present in the candidate rows
                        metric_column = candidate_metric  # Store detected metric field for row ranking
                        break  # Stop scanning once a valid ranking metric is found

                if metric_column is not None and not candidate_rows.empty:  # Rank fallback candidates when at least one metric column exists
                    ranked_rows = candidate_rows.copy()  # Copy candidate rows before numeric conversion and sorting
                    ranked_rows[metric_column] = pd.to_numeric(ranked_rows[metric_column], errors="coerce")  # Convert ranking metric to numeric values for deterministic ordering
                    ascending_order = metric_column.lower().endswith("fpr") or metric_column.lower().endswith("fnr")  # Use ascending order for error-rate metrics and descending for score metrics
                    ranked_rows = cast(Any, ranked_rows).sort_values(by=metric_column, ascending=ascending_order, na_position="last")  # Sort rows by the resolved metric with NaN values placed last
                    if not ranked_rows.empty:  # Continue only when ranking produced at least one usable row
                        selected_row = ranked_rows.iloc[0]  # Select the top-ranked fallback row
                        selected_source = f"fallback_metric={metric_column}"  # Record metric-based fallback source for diagnostics

                if selected_row is None and not candidate_rows.empty:  # Fallback to first valid payload row when no ranking metric was available
                    selected_row = candidate_rows.iloc[0]  # Select first non-empty best_features row as final fallback
                    selected_source = "fallback_first_non_empty_best_features"  # Record positional fallback source for diagnostics

            if selected_row is None:  # Abort when no row can provide a valid best_features payload
                print(
                    f"{BackgroundColors.RED}Error: 'best' run_index not found in GA results file at {BackgroundColors.CYAN}{ga_results_path}{BackgroundColors.RED}, and no fallback row with non-empty 'best_features' was found.{Style.RESET_ALL}"
                )  # Report final selection failure with explicit fallback outcome
                return None  # Return None when no usable GA feature row is available

            best_features_raw = selected_row["best_features"]  # Read serialized best_features payload from the selected row
            parsed_features = best_features_raw  # Initialize parsed payload with raw value before decoding
            if isinstance(parsed_features, str):  # Decode serialized payload when best_features is stored as string
                decoded_value = parsed_features.strip()  # Normalize serialized payload whitespace before decoding
                for _ in range(2):  # Decode at most twice to support nested serialized payloads
                    if not isinstance(decoded_value, str):  # Stop decoding when payload is no longer a string
                        break  # Exit decode loop once payload becomes structured data
                    if decoded_value == "":  # Convert empty payload string into an empty list container
                        decoded_value = []  # Map empty payload to empty list for downstream normalization
                        break  # Stop decode loop after empty payload normalization
                    try:  # Prefer JSON decoding because GA writer exports JSON strings
                        decoded_value = json.loads(decoded_value)  # Decode JSON payload into Python structure
                        continue  # Continue decode loop for possible nested serialization
                    except Exception:
                        pass  # Fall through to Python-literal decode when JSON parsing fails
                    try:  # Support legacy literal string payloads from older exports
                        decoded_value = ast.literal_eval(decoded_value)  # Decode Python literal payload safely into Python structure
                        continue  # Continue decode loop for possible nested serialization
                    except Exception:
                        break  # Stop decode loop when no decoder can parse the remaining string
                parsed_features = decoded_value  # Store decoded payload for final normalization

            if isinstance(parsed_features, (tuple, set)):  # Normalize tuple/set payloads into ordered list container
                parsed_features = list(parsed_features)  # Convert tuple/set payload into list for downstream compatibility
            elif isinstance(parsed_features, np.ndarray):  # Normalize numpy array payloads into Python list container
                parsed_features = parsed_features.tolist()  # Convert numpy payload into list for downstream compatibility
            elif isinstance(parsed_features, str):  # Normalize single feature string payload into one-item list
                parsed_features = [parsed_features]  # Wrap single feature name string into list container

            if not isinstance(parsed_features, list):  # Validate final decoded payload type before returning features
                print(
                    f"{BackgroundColors.RED}Error: Invalid GA 'best_features' payload type ({type(parsed_features).__name__}) in {BackgroundColors.CYAN}{ga_results_path}{Style.RESET_ALL}"
                )  # Report unsupported payload type to preserve explicit diagnostics
                return None  # Return None when decoded payload is not a feature list

            ga_features = [str(f) for f in parsed_features]  # Normalize all feature names to strings for downstream consumers

            verbose_output(
                f"{BackgroundColors.GREEN}Successfully extracted {BackgroundColors.CYAN}{len(ga_features)}{BackgroundColors.GREEN} GA features using {BackgroundColors.CYAN}{selected_source}{BackgroundColors.GREEN}.{Style.RESET_ALL}",
                config=config
            )  # Output the verbose message

            return ga_features  # Return the list of GA features
        except Exception as e:  # If there is an error loading or parsing the file
            print(
                f"{BackgroundColors.RED}Error loading/parsing GA features from {BackgroundColors.CYAN}{ga_results_path}{BackgroundColors.RED}: {e}{Style.RESET_ALL}"
            )
            return None  # Return None if there was an error
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_principal_component_analysis_features(file_path, config=None):
    """
    Extracts the optimal number of Principal Components (n_components)
    from the "PCA_Results.csv" file located in the "Feature_Analysis"
    subdirectory relative to the input file's directory.

    Selection is based on the safest available PCA ranking logic:
      1. Highest test_f1_score
      2. Lowest test_fnr
      3. Lowest test_fpr
      4. Highest explained_variance
      5. Lowest n_components

    If the PCA results file does not contain the full metric set, the
    function falls back to choosing the row with the highest cv_f1_score.

    :param file_path: Full path to the current CSV file being processed (e.g., "./Datasets/.../DrDoS_DNS.csv").
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Integer representing the optimal number of components (n_components), or None if the file is not found or fails to load/parse.
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        file_dir = os.path.dirname(file_path)  # Determine the directory of the input file
        verbose_output(
            f"{BackgroundColors.GREEN}Extracting PCA features for file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        pca_results_path = find_feature_file(file_path, "PCA/PCA_Results.csv", config=config)  # Find the PCA results file
        if pca_results_path is None:  # If the PCA results file does not exist
            print(
                f"{BackgroundColors.YELLOW}Warning: PCA results file not found for dataset containing {BackgroundColors.CYAN}{file_path}{BackgroundColors.YELLOW}. Skipping PCA feature extraction for this file.{Style.RESET_ALL}"
            )
            return None  # Return None if the file does not exist

        print(f"{BackgroundColors.GREEN}[INFO] PCA_Results.csv found at: {BackgroundColors.CYAN}{pca_results_path}{Style.RESET_ALL}")  # Identify the analysis CSV resolved for component selection.

        try:  # Try to load the PCA results
            low_memory = config.get("execution", {}).get("low_memory", False)  # Read low memory flag from config
            df = pd.read_csv(pca_results_path, low_memory=low_memory)  # Load the PCA results file
            df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names
            print(f"{BackgroundColors.GREEN}[INFO] PCA analysis results loaded from: {BackgroundColors.CYAN}{pca_results_path}{Style.RESET_ALL}")  # Confirm successful CSV parsing before ranking configurations.

            if df.empty:  # Verify if the DataFrame is empty
                print(
                    f"{BackgroundColors.RED}Error: PCA results file at {BackgroundColors.CYAN}{pca_results_path}{BackgroundColors.RED} is empty.{Style.RESET_ALL}"
                )
                return None  # Return None if the DataFrame is empty

            required_columns = ["n_components", "cv_f1_score"]  # Required columns for PCA selection
            for col in required_columns:  # Ensure the required columns are present
                if col not in df.columns:
                    raise KeyError(col)  # Raise if a required column is missing

            metric_columns = ["test_f1_score", "test_fnr", "test_fpr", "explained_variance"]  # Metrics for safe ranking
            if all(col in df.columns for col in metric_columns):  # Use the full ranking only if all metrics exist
                df = df.copy()  # Make a copy before numeric conversion
                df["n_components"] = pd.to_numeric(df["n_components"], errors="raise")  # Normalize component counts
                df["test_f1_score"] = pd.to_numeric(df["test_f1_score"], errors="raise")  # Normalize test F1 scores
                df["test_fnr"] = pd.to_numeric(df["test_fnr"], errors="raise")  # Normalize test FNR
                df["test_fpr"] = pd.to_numeric(df["test_fpr"], errors="raise")  # Normalize test FPR
                df["explained_variance"] = pd.to_numeric(df["explained_variance"], errors="raise")  # Normalize explained variance

                sorted_df = df.sort_values(
                    by=["test_f1_score", "test_fnr", "test_fpr", "explained_variance", "n_components"],
                    ascending=[False, True, True, False, True],
                    kind="mergesort",
                )  # Rank candidates with stable sorting and explicit tie-breakers
                best_n_components = sorted_df.iloc[0]["n_components"]  # Select the best row after ranking
            else:  # Fallback to CV F1 when the full metric set is unavailable
                best_row_index = df["cv_f1_score"].idxmax()  # Find the highest CV F1-Score row
                best_n_components = df.loc[best_row_index, "n_components"]  # Select n_components from the best CV F1 row

            verbose_output(
                f"{BackgroundColors.GREEN}Successfully extracted best PCA configuration. Optimal components: {BackgroundColors.CYAN}{best_n_components}{Style.RESET_ALL}"
            )  # Output the verbose message

            best_n_components_int = int(cast(Any, pd.to_numeric(best_n_components, errors="raise")))  # Ensure it's an integer
            print(f"{BackgroundColors.GREEN}[INFO] PCA optimal component count selected from PCA_Results.csv: {BackgroundColors.CYAN}{best_n_components_int}{Style.RESET_ALL}")  # Report the selected component count separately from transformer loading.

            return best_n_components_int  # Return the optimal number of components

        except KeyError as e:  # Handle missing columns
            print(
                f"{BackgroundColors.RED}Error: Required column {e} not found in PCA results file at {BackgroundColors.CYAN}{pca_results_path}{Style.RESET_ALL}"
            )
            return None  # Return None if required column is missing
        except Exception as e:  # Handle other errors (loading, parsing, etc.)
            print(
                f"{BackgroundColors.RED}Error loading/parsing PCA features from {BackgroundColors.CYAN}{pca_results_path}{BackgroundColors.RED}: {e}{Style.RESET_ALL}"
            )
            return None  # Return None if there was an error
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_recursive_feature_elimination_features(file_path, config=None):
    """
    Extracts the "top_features" list (Python literal string) from the first row of the
    "RFE_Run_Results.csv" file located in the "Feature_Analysis" subdirectory
    relative to the input file's directory.

    :param file_path: Full path to the current CSV file being processed (e.g., "./Datasets/.../DrDoS_DNS.csv").
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: List of top features selected by RFE from the first run, or None if the file is not found or fails to load/parse.
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        file_dir = os.path.dirname(file_path)  # Determine the directory of the input file
        verbose_output(
            f"{BackgroundColors.GREEN}Extracting RFE features for file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        rfe_runs_path = find_feature_file(file_path, "RFE/RFE_Run_Results.csv", config=config)  # Find the RFE runs file
        if rfe_runs_path is None:  # If the RFE runs file does not exist
            print(
                f"{BackgroundColors.YELLOW}Warning: RFE runs file not found for dataset containing {BackgroundColors.CYAN}{file_path}{BackgroundColors.YELLOW}. Skipping RFE feature extraction for this file.{Style.RESET_ALL}"
            )
            return None  # Return None if the file does not exist

        print(f"{BackgroundColors.GREEN}[INFO] RFE feature file found: {BackgroundColors.CYAN}{rfe_runs_path}{Style.RESET_ALL}")  # Log the resolved RFE runs file path

        try:  # Try to load the RFE runs results
            low_memory = config.get("execution", {}).get("low_memory", False)  # Read low memory flag from config
            df = pd.read_csv(rfe_runs_path, usecols=cast(Any, ["top_features"]), low_memory=low_memory)  # Load only the "top_features" column
            df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names

            if not df.empty:  # Verify if the DataFrame is not empty
                top_features_raw = df.loc[0, "top_features"]  # Get the "top_features" from the first row

                rfe_features = top_features_raw  # Keep raw value and normalize below to avoid string/char splitting
                if isinstance(rfe_features, str):  # If serialized, decode safely and support double-encoded payloads
                    decoded_value = rfe_features.strip()  # Normalize whitespace around serialized payload
                    for _ in range(2):  # Decode at most twice to support nested JSON string payloads
                        if not isinstance(decoded_value, str):  # Stop when payload is no longer a string
                            break
                        if decoded_value == "":  # Empty serialized payload maps to an empty feature list
                            decoded_value = []
                            break
                        try:  # Prefer JSON decoding because RFE exports JSON strings
                            decoded_value = json.loads(decoded_value)
                            continue
                        except Exception:
                            pass
                        try:  # Fallback for legacy Python-literal serialized payloads
                            decoded_value = ast.literal_eval(decoded_value)
                            continue
                        except Exception:
                            break
                    rfe_features = decoded_value  # Store decoded payload for type normalization

                if isinstance(rfe_features, (tuple, set)):  # Normalize tuple/set payloads to list for downstream compatibility
                    rfe_features = list(rfe_features)
                elif isinstance(rfe_features, np.ndarray):  # Normalize numpy array payloads to list for downstream compatibility
                    rfe_features = rfe_features.tolist()
                elif isinstance(rfe_features, str):  # Defensive fallback: treat single feature name as one-item list
                    rfe_features = [rfe_features]

                if not isinstance(rfe_features, list):  # Validate normalized payload type before returning
                    raise ValueError(f"Invalid RFE top_features payload type: {type(rfe_features).__name__}")

                rfe_features = [str(f) for f in rfe_features]  # Ensure all feature names are strings

                verbose_output(
                    f"{BackgroundColors.GREEN}Successfully extracted RFE top features from Run 1. Total features: {BackgroundColors.CYAN}{len(rfe_features)}{Style.RESET_ALL}",
                    config=config
                )  # Output the verbose message

                return rfe_features  # Return the list of RFE features
            else:  # If the DataFrame is empty
                print(
                    f"{BackgroundColors.RED}Error: RFE runs file at {BackgroundColors.CYAN}{rfe_runs_path}{BackgroundColors.RED} is empty.{Style.RESET_ALL}"
                )
                return None  # Return None if the file is empty

        except Exception as e:  # If there is an error loading or parsing the file
            print(
                f"{BackgroundColors.RED}Error loading/parsing RFE features from {BackgroundColors.CYAN}{rfe_runs_path}{BackgroundColors.RED}: {e}{Style.RESET_ALL}"
            )
            return None  # Return None if there was an error
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def load_feature_selection_results(file_path, config=None):
    """
    Load GA, RFE and PCA feature selection artifacts for a given dataset file and
    print concise status messages.

    :param file_path: Path to the dataset CSV being processed.
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (ga_selected_features, pca_n_components, rfe_selected_features)
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        ga_selected_features = extract_genetic_algorithm_features(file_path, config=config)  # Extract GA features
        if ga_selected_features:  # If GA features were successfully extracted
            verbose_output(
                f"{BackgroundColors.GREEN}Genetic Algorithm Features successfully loaded for {BackgroundColors.CYAN}{os.path.basename(file_path)}{BackgroundColors.GREEN}. Total features: {BackgroundColors.CYAN}{len(ga_selected_features)}{Style.RESET_ALL}",
                config=config
            )
            verbose_output(
                f"{BackgroundColors.GREEN}Genetic Algorithm Selected Features: {BackgroundColors.CYAN}{ga_selected_features}{Style.RESET_ALL}",
                config=config
            )
        else:  # If GA features were not extracted
            print(
                f"{BackgroundColors.YELLOW}Proceeding without GA features for {BackgroundColors.CYAN}{os.path.basename(file_path)}{Style.RESET_ALL}"
            )

        pca_n_components = extract_principal_component_analysis_features(file_path, config=config)  # Extract PCA components
        if pca_n_components:  # If PCA components were successfully extracted
            verbose_output(
                f"{BackgroundColors.GREEN}PCA optimal components successfully loaded for {BackgroundColors.CYAN}{os.path.basename(file_path)}{BackgroundColors.GREEN}: {BackgroundColors.CYAN}{pca_n_components}{Style.RESET_ALL}",
                config=config
            )
            verbose_output(
                f"{BackgroundColors.GREEN}PCA Number of Components: {BackgroundColors.CYAN}{pca_n_components}{Style.RESET_ALL}",
                config=config
            )
        else:  # If PCA components were not extracted
            print(
                f"{BackgroundColors.YELLOW}Proceeding without PCA components for {BackgroundColors.CYAN}{os.path.basename(file_path)}{Style.RESET_ALL}"
            )

        rfe_selected_features = extract_recursive_feature_elimination_features(file_path, config=config)  # Extract RFE features
        if rfe_selected_features:  # If RFE features were successfully extracted
            verbose_output(
                f"{BackgroundColors.GREEN}RFE Features successfully loaded for {BackgroundColors.CYAN}{os.path.basename(file_path)}{BackgroundColors.GREEN}. Total features: {BackgroundColors.CYAN}{len(rfe_selected_features)}{Style.RESET_ALL}",
                config=config
            )
            verbose_output(
                f"{BackgroundColors.GREEN}RFE Selected Features: {BackgroundColors.CYAN}{rfe_selected_features}{Style.RESET_ALL}",
                config=config
            )
        else:  # If RFE features were not extracted
            print(
                f"{BackgroundColors.YELLOW}Proceeding without RFE features for {BackgroundColors.CYAN}{os.path.basename(file_path)}{Style.RESET_ALL}"
            )

        return ga_selected_features, pca_n_components, rfe_selected_features  # Return the extracted features
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def resolve_format_extension(file_format: str) -> str:
    """
    Resolve a file format string to its dot-prefixed file extension.

    :param file_format: File format string (arff, csv, parquet, txt).
    :return: Dot-prefixed file extension string.
    """

    try:
        extension_map = {  # Mapping of format strings to file extensions
            "arff": ".arff",
            "csv": ".csv",
            "parquet": ".parquet",
            "txt": ".txt",
        }
        return extension_map.get(file_format.lower(), ".csv")  # Return extension or fallback to .csv
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_arff_with_scipy(input_path: str, config=None) -> pd.DataFrame:
    """
    Load an ARFF file using scipy, decoding byte strings when necessary.

    :param input_path: Path to the ARFF file.
    :param config: Configuration dictionary (uses global CONFIG if None).
    :return: pandas DataFrame loaded from the ARFF file.
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Loading ARFF file with scipy: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        data, meta = scipy_arff.loadarff(input_path)  # Load the ARFF file using scipy
        df = pd.DataFrame(data)  # Convert the loaded data to a DataFrame

        for col in df.columns:  # Iterate through each column in the DataFrame
            if df[col].dtype == object:  # If column contains byte/string data
                df[col] = df[col].apply(
                    lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
                )  # Decode bytes to strings

        return df  # Return the DataFrame with decoded strings
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_arff_with_liac(input_path: str, config=None) -> pd.DataFrame:
    """
    Load an ARFF file using the liac-arff library.

    :param input_path: Path to the ARFF file.
    :param config: Configuration dictionary (uses global CONFIG if None).
    :return: pandas DataFrame loaded from the ARFF file.
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Loading ARFF file with liac-arff: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        with open(input_path, "r", encoding="utf-8") as f:  # Open the ARFF file for reading
            data = arff.load(f)  # Load using liac-arff

        return pd.DataFrame(data["data"], columns=pd.Index([attr[0] for attr in data["attributes"]]))  # Convert to DataFrame
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_arff_file(input_path: str, config=None) -> pd.DataFrame:
    """
    Load an ARFF file, trying scipy first and falling back to liac-arff if needed.

    :param input_path: Path to the ARFF file.
    :param config: Configuration dictionary (uses global CONFIG if None).
    :return: pandas DataFrame loaded from the ARFF file.
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        try:  # Attempt to load using scipy
            return load_arff_with_scipy(input_path, config=config)  # Return DataFrame from scipy loader
        except Exception as e:  # If scipy fails, warn and try liac-arff
            verbose_output(
                f"{BackgroundColors.YELLOW}Warning: Failed to load ARFF with scipy ({e}). Trying liac-arff...{Style.RESET_ALL}",
                config=config
            )  # Output fallback warning message

            try:  # Attempt to load using liac-arff
                return load_arff_with_liac(input_path, config=config)  # Return DataFrame from liac loader
            except Exception as e2:  # If both loaders fail
                raise RuntimeError(f"Failed to load ARFF file with both scipy and liac-arff: {e2}")  # Raise combined error
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_parquet_file(input_path: str, config=None) -> pd.DataFrame:
    """
    Load a Parquet file into a pandas DataFrame.

    :param input_path: Path to the Parquet file.
    :param config: Configuration dictionary (uses global CONFIG if None).
    :return: pandas DataFrame loaded from the Parquet file.
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Loading Parquet file: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        return pd.read_parquet(input_path)  # Load and return the Parquet file as a DataFrame
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_txt_file(input_path: str, config=None) -> pd.DataFrame:
    """
    Load a TXT file into a pandas DataFrame, assuming tab-separated values.

    :param input_path: Path to the TXT file.
    :param config: Configuration dictionary (uses global CONFIG if None).
    :return: pandas DataFrame containing the loaded dataset.
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Loading TXT file: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        low_memory = config.get("execution", {}).get("low_memory", False)  # Read low memory flag from config
        df = pd.read_csv(input_path, sep="\t", low_memory=low_memory)  # Load TXT file using tab separator

        return df  # Return the DataFrame
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def dispatch_format_loader(file_path: str, file_format: str, config=None) -> pd.DataFrame:
    """
    Dispatch dataset loading to the appropriate loader based on the given file format.

    :param file_path: Path to the dataset file to load.
    :param file_format: File format string: arff, csv, parquet, or txt.
    :param config: Configuration dictionary (uses global CONFIG if None).
    :return: pandas DataFrame loaded from the file.
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        supported_formats = {"arff", "csv", "parquet", "txt"}  # Set of supported file formats
        if file_format not in supported_formats:  # If the format is not supported
            raise ValueError(f"Unsupported file format: '{file_format}'. Supported: {sorted(supported_formats)}")  # Raise error for unsupported format

        low_memory = config.get("execution", {}).get("low_memory", False)  # Read low memory flag from config

        if file_format == "arff":  # If the file format is ARFF
            return load_arff_file(file_path, config=config)  # Load and return ARFF file
        elif file_format == "csv":  # If the file format is CSV
            df = pd.read_csv(file_path, low_memory=low_memory)  # Load CSV file with configured memory mode
            return df  # Return the loaded CSV DataFrame
        elif file_format == "parquet":  # If the file format is Parquet
            return load_parquet_file(file_path, config=config)  # Load and return Parquet file
        elif file_format == "txt":  # If the file format is TXT
            return load_txt_file(file_path, config=config)  # Load and return TXT file
        return pd.DataFrame()  # Return empty DataFrame to satisfy return contract
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics

def load_augmented_dataset(file_path: str, config=None) -> Optional[pd.DataFrame]:
    """
    Load an augmented dataset file and return a DataFrame.
    Supports csv, txt, parquet, and arff formats via stacking.augmentation_file_format.

    :param file_path: Path to the augmented dataset file.
    :param config: Configuration dictionary (uses global CONFIG if None).
    :return: DataFrame or None if file not found or invalid.
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"\n{BackgroundColors.GREEN}Loading augmented dataset from: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}",
            config=config
        )  # Output the loading augmented dataset message

        if not verify_filepath_exists(file_path):  # If the augmented dataset file does not exist
            print(f"{BackgroundColors.RED}Augmented dataset file not found: {file_path}{Style.RESET_ALL}")
            return None  # Return None

        file_format = config.get("stacking", {}).get("augmentation_file_format", "csv")  # Read configured augmentation file format

        verbose_output(
            f"{BackgroundColors.GREEN}Loading augmented dataset with format: {BackgroundColors.CYAN}{file_format}{Style.RESET_ALL}",
            config=config
        )  # Output the selected format message

        df = dispatch_format_loader(file_path, file_format, config=config)  # Dispatch to the appropriate loader

        df.columns = df.columns.str.strip()  # Clean column names by stripping leading/trailing whitespace

        if df.shape[1] < 2:  # If there are less than 2 columns
            print(f"{BackgroundColors.RED}Augmented dataset must have at least 1 feature and 1 target.{Style.RESET_ALL}")
            return None  # Return None

        return df  # Return the loaded DataFrame
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_dataset(file_path: str, config=None) -> Optional[pd.DataFrame]:
    """
    Load a dataset file and return a DataFrame.
    Supports csv, txt, parquet, and arff formats via stacking.dataset_file_format.

    :param file_path: Path to the dataset file.
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: DataFrame
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Loading dataset from: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}",
            config=config
        )  # Output the loading dataset message

        if not verify_filepath_exists(file_path):  # If the dataset file does not exist
            print(f"{BackgroundColors.RED}Dataset file not found: {file_path}{Style.RESET_ALL}")
            return None  # Return None

        file_format = config.get("stacking", {}).get("dataset_file_format", "csv")  # Read configured dataset file format

        verbose_output(
            f"{BackgroundColors.GREEN}Loading dataset with format: {BackgroundColors.CYAN}{file_format}{Style.RESET_ALL}",
            config=config
        )  # Output the selected format message

        df = dispatch_format_loader(file_path, file_format, config=config)  # Dispatch to the appropriate loader

        df.columns = df.columns.str.strip()  # Clean column names by stripping leading/trailing whitespace

        if df.shape[1] < 2:  # If there are less than 2 columns
            print(f"{BackgroundColors.RED}Dataset must have at least 1 feature and 1 target.{Style.RESET_ALL}")
            return None  # Return None

        return df  # Return the loaded DataFrame
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def sanitize_feature_name(name: str) -> str:
    """
    Normalize a single feature name for whitespace-agnostic, case-insensitive comparison only.

    :param name: Raw feature name string to normalize for comparison.
    :return: Normalized lowercase feature name string suitable for comparison only.
    """

    try:
        if not isinstance(name, str):  # Verify the input is a string before processing
            name = str(name)  # Convert non-string input to string for safe normalization
        name = re.sub(r"[\u200b\u200c\u200d\ufeff\u00a0\u2060\u2028\u2029]", "", name)  # Remove invisible and zero-width unicode characters
        name = name.strip()  # Strip leading and trailing whitespace from the name
        name = re.sub(r" +", " ", name)  # Collapse multiple consecutive spaces into a single space
        return name.lower()  # Return lowercase form for case-insensitive comparison only
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
    
    try:
        if isinstance(columns, str):  # Guard against raw string payloads to avoid character-by-character sanitization
            columns = [columns]  # Treat single string as one feature name

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


def verify_selected_features_exist(selected_features: list, dataset_columns: list, method_name: str) -> list:
    """
    Verify that features selected by a feature-selection method exist in the dataset columns.

    :param selected_features: list of selected feature names (strings)
    :param dataset_columns: list of dataset column names (strings)
    :param method_name: human-friendly method name (e.g., 'GA', 'RFE', 'PCA')
    :return: filtered list of valid feature names mapped to original dataset column names
    """

    if selected_features is None:  # Verify selected_features is not None before processing
        raise ValueError(f"No selected features provided for {method_name}.")  # Raise with method context for diagnostics
    if not isinstance(selected_features, (list, tuple)):  # Verify selected_features is a list or tuple
        raise ValueError("selected_features must be a list or tuple")  # Raise with type mismatch details
    if not selected_features:  # Verify selected_features is not empty
        raise ValueError(f"selected_features is empty for {method_name}")  # Raise with method context for diagnostics
    if dataset_columns is None or not isinstance(dataset_columns, (list, tuple)):  # Verify dataset_columns is a valid sequence
        raise ValueError("dataset_columns must be a non-empty list or tuple")  # Raise with type mismatch details
    if not dataset_columns:  # Verify dataset_columns is not empty
        raise ValueError("dataset_columns is empty")  # Raise with empty list details

    mn = (method_name or "").strip().lower()  # Normalize method name for PCA detection
    is_pca = mn == "pca"  # Determine if this is a PCA feature set
    if is_pca:  # If PCA, return component names as-is since they are synthetic
        if all(isinstance(f, str) and f.strip().upper().startswith("PC") for f in selected_features):  # Verify all selected are valid PC component names
            return list(dict.fromkeys(selected_features))  # Preserve order and remove duplicates for PCA components

    sanitized_col_map = {sanitize_feature_name(col): col for col in dataset_columns}  # Build sanitized-to-original column mapping for whitespace-agnostic comparison
    valid_features = []  # Accumulate valid features mapped to original dataset column names
    seen_sanitized = set()  # Track sanitized keys already resolved to prevent duplicates
    for f in selected_features:  # Iterate over each selected feature for sanitized lookup
        sf = sanitize_feature_name(f)  # Normalize selected feature name for comparison
        if sf in sanitized_col_map and sf not in seen_sanitized:  # Verify sanitized match exists and not yet resolved
            valid_features.append(sanitized_col_map[sf])  # Append original dataset column name for this feature
            seen_sanitized.add(sf)  # Mark this sanitized key as seen to prevent duplicates

    missing = [f for f in selected_features if sanitize_feature_name(f) not in sanitized_col_map]  # Identify features absent even after sanitized comparison

    if missing and len(valid_features) > 0:  # If partial mismatch detected (some valid, some missing)
        print(f"{BackgroundColors.YELLOW}[WARNING] Missing features detected for {BackgroundColors.CYAN}{method_name}{BackgroundColors.YELLOW}:{Style.RESET_ALL}")  # Print partial-missing warning header
        for m in missing:  # Iterate over each missing feature name
            print(f"  - {m}")  # Print individual missing feature name
        print(f"{BackgroundColors.GREEN}Proceeding with {BackgroundColors.CYAN}{len(valid_features)}{BackgroundColors.GREEN} valid features for {BackgroundColors.CYAN}{method_name}{BackgroundColors.GREEN}.{Style.RESET_ALL}")  # Print valid-features continuation message
        return valid_features  # Return only the valid features mapped to original dataset column names

    if missing and len(valid_features) == 0:  # If all features are missing from dataset columns
        print(f"{BackgroundColors.RED}[WARNING] All selected features for {BackgroundColors.CYAN}{method_name}{BackgroundColors.RED} are missing from dataset columns.{Style.RESET_ALL}")  # Print all-missing warning
        for m in missing:  # Iterate over each missing feature name
            print(f"  - {m}")  # Print individual missing feature name
        print(f"{BackgroundColors.YELLOW}Using 0 out of {BackgroundColors.CYAN}{len(selected_features)}{BackgroundColors.YELLOW} selected features for {BackgroundColors.CYAN}{method_name}{BackgroundColors.YELLOW}.{Style.RESET_ALL}")  # Print zero-valid fallback message
        raise ValueError(f"All selected features for {method_name} are missing from dataset columns")  # Raise to signal complete mismatch

    return list(dict.fromkeys(valid_features))  # Preserve order and remove any residual duplicates from valid features


def preprocess_dataframe(df, remove_zero_variance=True, config=None):
    """
    Preprocess a DataFrame by removing rows with NaN or infinite values and
    dropping zero-variance numeric features.

    :param df: pandas DataFrame to preprocess
    :param remove_zero_variance: whether to drop numeric columns with zero variance
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: cleaned DataFrame
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        write_memory_phase_event("before_preprocessing", config=config, row_count=len(df) if df is not None else None, feature_count=len(df.columns) if df is not None else None, event_outcome="starting")  # Publish preprocessing start
        if remove_zero_variance:  # If remove_zero_variance is set to True
            verbose_output(
                f"{BackgroundColors.GREEN}Preprocessing DataFrame: "
                f"{BackgroundColors.CYAN}normalizing and sanitizing column names, removing NaN/infinite rows, and dropping zero-variance numeric features"
                f"{BackgroundColors.GREEN}.{Style.RESET_ALL}",
                config=config
            )
        else:  # If remove_zero_variance is set to False
            verbose_output(
                f"{BackgroundColors.GREEN}Preprocessing DataFrame: "
                f"{BackgroundColors.CYAN}normalizing and sanitizing column names and removing NaN/infinite rows"
                f"{BackgroundColors.GREEN}.{Style.RESET_ALL}",
                config=config
            )

        if df is None:  # If the DataFrame is None
            write_memory_phase_event("after_preprocessing", config=config, row_count=None, feature_count=None, event_outcome="none_input")  # Publish preprocessing none-input outcome
            return df  # Return None

        df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names
        
        df.columns = sanitize_feature_names(df.columns)  # Sanitize column names to remove special characters

        df.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN in-place to avoid creating a copy
        df_clean = df.dropna()  # Drop rows containing NaN values into a new dataframe
        del df  # Release original dataframe reference to free memory after cleaning
        gc.collect()  # Force garbage collection to reclaim memory from deleted original dataframe

        if remove_zero_variance:  # If remove_zero_variance is set to True
            numeric_cols = df_clean.select_dtypes(include=["number"]).columns  # Select only numeric columns
            if len(numeric_cols) > 0:  # If there are numeric columns
                variances = df_clean[numeric_cols].var(axis=0, ddof=0)  # Calculate variances
                zero_var_cols = variances[variances == 0].index.tolist()  # Get columns with zero variance
                if zero_var_cols:  # If there are zero-variance columns
                    df_clean = df_clean.drop(columns=zero_var_cols)  # Drop zero-variance columns

        write_memory_phase_event("after_preprocessing", config=config, row_count=len(df_clean), feature_count=len(df_clean.columns), event_outcome="completed")  # Publish preprocessing completion
        return df_clean  # Return the cleaned DataFrame
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def scale_and_split(X, y, test_size=0.2, random_state=42, config=None):
    """
    Scales the numeric features using StandardScaler and splits the data
    into training and testing sets.

    Note: The target encoder is fitted after splitting, using original training labels only.

    :param X: Features DataFrame (must contain numeric features).
    :param y: Target Series or array.
    :param test_size: Fraction of the data to reserve for the test set.
    :param random_state: Seed for the random split for reproducibility.
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder)
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Scaling features and splitting data (train/test ratio: {BackgroundColors.CYAN}{1-test_size}/{test_size}{BackgroundColors.GREEN})...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        y_series = y if isinstance(y, pd.Series) else pd.Series(y)  # Normalize target to pandas Series only when needed

        if isinstance(X, pd.DataFrame):  # If features are a DataFrame
            non_numeric_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]  # Detect non-numeric feature columns for warning and filtering
            if non_numeric_cols:  # If non-numeric columns were found
                print(
                    f"{BackgroundColors.YELLOW}Warning: Dropping non-numeric feature columns for scaling: {BackgroundColors.CYAN}{non_numeric_cols}{Style.RESET_ALL}"
                )  # Warn about dropped columns
                numeric_X = X.select_dtypes(include=np.number)  # Select only numeric columns when needed
                if numeric_X.empty:  # If no numeric features remain after filtering
                    raise ValueError(
                        f"{BackgroundColors.RED}No numeric features found in X after filtering.{Style.RESET_ALL}"
                    )  # Raise an error if no numeric features remain
                X_values = numeric_X.to_numpy(copy=False)  # Convert numeric DataFrame to numpy view when possible
                del numeric_X  # Release temporary numeric DataFrame reference to reduce peak memory
            else:  # If all columns are already numeric
                X_values = X.to_numpy(copy=False)  # Convert directly to numpy view when possible without extra DataFrame filtering copy
        else:  # If features are already array-like
            X_values = np.asarray(X)  # Normalize features to numpy array

        if X_values.ndim != 2 or X_values.shape[1] == 0:  # If no usable feature columns are available
            raise ValueError(
                f"{BackgroundColors.RED}No numeric features found in X after filtering.{Style.RESET_ALL}"
            )  # Raise an error if X has no valid feature columns

        write_memory_phase_event("before_train_test_split", config=config, train_sample_count=None, test_sample_count=None, feature_count=X_values.shape[1], event_outcome="starting")  # Publish train/test split start
        target_values = y_series.to_numpy(copy=False)  # Reuse the original target values for stratified splitting before fitting the encoder.
        sample_indices = np.arange(target_values.shape[0], dtype=np.int64)  # Build sample index array for index-based stratified splitting
        train_idx, test_idx = train_test_split(
            sample_indices, test_size=test_size, random_state=random_state, stratify=target_values
        )  # Split sample indices to avoid DataFrame duplication during train/test partitioning

        X_train = X_values[train_idx]  # Materialize training features from index split
        X_test = X_values[test_idx]  # Materialize test features from index split
        le = LabelEncoder()  # Initialize the target encoder after isolating the original training rows.
        y_train = np.asarray(le.fit_transform(target_values[train_idx]), dtype=np.int64)  # Fit label encoding only on original training labels.
        y_test = np.asarray(le.transform(target_values[test_idx]), dtype=np.int64)  # Transform original testing labels without modifying the encoder.
        write_memory_phase_event("after_train_test_split", config=config, train_sample_count=len(y_train), test_sample_count=len(y_test), feature_count=X_train.shape[1], event_outcome="completed")  # Publish train/test split completion

        del sample_indices, train_idx, test_idx, X_values  # Release split helper arrays and base feature view before augmentation/scaling
        gc.collect()  # Force garbage collection to reclaim memory from released split helpers

        write_memory_phase_event("before_scaling", config=config, train_sample_count=len(y_train), test_sample_count=len(y_test), feature_count=X_train.shape[1], event_outcome="starting")  # Publish scaling start
        scaler = StandardScaler()  # Initialize the StandardScaler

        X_train_scaled = np.asarray(scaler.fit_transform(X_train))  # Fit and transform original training features only.

        X_test_scaled = np.asarray(scaler.transform(X_test))  # Transform the testing features (original data only)
        write_memory_phase_event("after_scaling", config=config, train_sample_count=len(y_train), test_sample_count=len(y_test), feature_count=X_train_scaled.shape[1], event_outcome="completed")  # Publish scaling completion

        del X_train, X_test, y_series  # Release large pre-scaled matrices and temporary target Series after scaling
        gc.collect()  # Force garbage collection to reclaim memory from released pre-scaled matrices

        verbose_output(
            f"{BackgroundColors.GREEN}Data split successful. Training set shape: {BackgroundColors.CYAN}{X_train_scaled.shape}{BackgroundColors.GREEN}. Testing set shape: {BackgroundColors.CYAN}{X_test_scaled.shape}{Style.RESET_ALL}",
            config=config
        )  # Output the successful split message

        return (
            X_train_scaled,
            X_test_scaled,
            y_train,
            y_test,
            scaler,
            le,
        )  # Return scaled features, targets, and fitted preprocessing.
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_models(config=None):
    """
    Initializes and returns a dictionary of models to train.

    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary of model name and instance
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Initializing models for training...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message
        
        n_jobs = config.get("evaluation", {}).get("n_jobs", 1)  # Get n_jobs from config
        random_state = config.get("evaluation", {}).get("random_state", 42)  # Get random_state from config
        
        rf_params = config.get("models", {}).get("random_forest", {})  # Random Forest params
        svm_params = config.get("models", {}).get("svm", {})  # SVM params
        xgb_params = config.get("models", {}).get("xgboost", {})  # XGBoost params
        lr_params = config.get("models", {}).get("logistic_regression", {})  # Logistic Regression params
        knn_params = config.get("models", {}).get("knn", {})  # KNN params
        gb_params = config.get("models", {}).get("gradient_boosting", {})  # Gradient Boosting params
        lgb_params = config.get("models", {}).get("lightgbm", {})  # LightGBM params
        mlp_params = config.get("models", {}).get("mlp", {})  # MLP params

        models = {  # Build full models dictionary with all classifiers
            "Random Forest": RandomForestClassifier(
                n_estimators=rf_params.get("n_estimators", 100),
                random_state=rf_params.get("random_state", random_state),
                n_jobs=n_jobs
            ),
            "SVM": SVC(
                kernel=svm_params.get("kernel", "rbf"),
                probability=svm_params.get("probability", True),
                random_state=svm_params.get("random_state", random_state)
            ),
            "XGBoost": XGBClassifier(
                eval_metric=xgb_params.get("eval_metric", "mlogloss"),
                random_state=xgb_params.get("random_state", random_state),
                n_jobs=n_jobs
            ),
            "Logistic Regression": LogisticRegression(
                max_iter=lr_params.get("max_iter", 1000),
                random_state=lr_params.get("random_state", random_state),
                n_jobs=n_jobs
            ),
            "KNN": KNeighborsClassifier(
                n_neighbors=knn_params.get("n_neighbors", 5),
                n_jobs=n_jobs
            ),
            "Nearest Centroid": NearestCentroid(),
            "Gradient Boosting": GradientBoostingClassifier(
                random_state=gb_params.get("random_state", random_state)
            ),
            "LightGBM": lgb.LGBMClassifier(
                force_row_wise=lgb_params.get("force_row_wise", True),
                min_gain_to_split=lgb_params.get("min_gain_to_split", 0.01),
                random_state=lgb_params.get("random_state", random_state),
                verbosity=lgb_params.get("verbosity", -1),
                n_jobs=n_jobs
            ),
            "MLP (Neural Net)": MLPClassifier(
                hidden_layer_sizes=mlp_params.get("hidden_layer_sizes", (100,)),
                max_iter=mlp_params.get("max_iter", 500),
                random_state=mlp_params.get("random_state", random_state)
            ),
        }

        classifiers_list = config.get("stacking", {}).get("enabled_classifiers", None)  # Read optional classifier filter list from config

        if classifiers_list is None:  # If no classifier filter key is present, preserve original behavior
            return models  # Return full models dictionary with no filtering applied

        unique_classifiers = list(dict.fromkeys(classifiers_list))  # Deduplicate classifiers list while preserving insertion order
        filtered_models = {k: v for k, v in models.items() if k in unique_classifiers}  # Retain only classifiers whose names appear in the filter list

        enabled_names = list(filtered_models.keys())  # Collect names of enabled classifiers for logging
        disabled_names = [k for k in models if k not in filtered_models]  # Collect names of disabled classifiers for logging

        verbose_output(
            f"[DEBUG] Enabled classifiers: {enabled_names}",
            config=config
        )  # Output the list of enabled classifiers
        verbose_output(
            f"[DEBUG] Disabled classifiers: {disabled_names}",
            config=config
        )  # Output the list of disabled classifiers

        return filtered_models  # Return filtered models dictionary, may be empty if all names were invalid or list was empty
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def filter_matching_hyperparams(df: pd.DataFrame, csv_path: str, config: dict) -> pd.DataFrame:
    """
    Filter hyperparameter optimization rows by normalized dataset path for combined files evaluation mode.

    :param df: DataFrame loaded from the hyperparameter optimization CSV file.
    :param csv_path: Full path to the dataset CSV file being processed.
    :param config: Configuration dictionary for verbose output.
    :return: Filtered DataFrame of matching rows, or empty DataFrame if no match found.
    """

    try:
        if "dataset_path" not in df.columns:  # Combined-files matching requires an explicit dataset directory identity.
            return pd.DataFrame()  # Return no matches when the artifact cannot identify its dataset path.

        repository_root = Path(__file__).resolve().parent  # Resolve relative artifact paths from the repository root used by the application.
        runtime_path = Path(str(csv_path).strip()).expanduser()  # Normalize the representative runtime path without changing the raw input.
        if not runtime_path.is_absolute():  # Resolve relative runtime inputs using the same repository-root convention.
            runtime_path = repository_root / runtime_path  # Build the complete runtime path before directory normalization.
        normalized_runtime_path = os.path.normcase(str(resolve_dataset_root_path(str(runtime_path))))  # Normalize the representative file to its complete dataset-directory path.

        normalized_artifact_paths = []  # Store comparison-only paths without modifying the dataframe's raw dataset_path values.
        for raw_dataset_path in df["dataset_path"]:  # Normalize each artifact dataset directory deterministically.
            if pd.isna(raw_dataset_path) or not str(raw_dataset_path).strip():  # Ignore missing dataset identities.
                normalized_artifact_paths.append(None)  # Preserve row alignment for non-matching missing values.
                continue  # Continue with the next artifact row.
            artifact_path = Path(str(raw_dataset_path).strip()).expanduser()  # Normalize user-home syntax and redundant path text.
            if not artifact_path.is_absolute():  # Resolve relative artifact paths from the verified repository root.
                artifact_path = repository_root / artifact_path  # Build the complete artifact dataset path.
            normalized_artifact_paths.append(os.path.normcase(str(resolve_dataset_root_path(str(artifact_path)))))  # Normalize directory intent, dot components, and trailing separators.

        normalized_dataset_paths = pd.Series(normalized_artifact_paths, index=df.index, dtype="object")  # Align comparison-only normalized paths with the original rows.
        path_match = cast(pd.DataFrame, df.loc[normalized_dataset_paths == normalized_runtime_path, :])  # Compare complete normalized dataset paths for equality.
        verbose_output(
            f"[DEBUG] filter_matching_hyperparams: matched {len(path_match)} row(s) by normalized dataset_path",
            config=config
        )  # Log the deterministic combined-files match count.
        return path_match  # Return only exact normalized dataset-path matches.
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_hyperparameter_optimization_results(csv_path, config=None):
    """
    Extract hyperparameter optimization results for a specific dataset file.

    Looks for the HYPERPARAMETERS_FILENAME file in the "Classifiers_Hyperparameters"
    subdirectory relative to the dataset CSV file. Matches the normalized dataset
    path in combined mode and the current base_csv filename in separate mode.

    This function extracts **only the best hyperparameters** for each classifier
    that corresponds to the current file being processed.

    :param csv_path: Path to the dataset CSV file being processed.
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary mapping model names to their best hyperparameters, or None if not found.
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Looking for hyperparameter optimization results for: {BackgroundColors.CYAN}{csv_path}{Style.RESET_ALL}",
            config=config
        )  # Inform user which dataset we're searching for

        file_dir = os.path.dirname(csv_path)  # Directory containing the dataset file
        base_filename = os.path.basename(csv_path)  # Get the base filename (e.g., "UDPLag.csv")
        
        hyperparameters_filename = config.get("stacking", {}).get("hyperparameters_filename", "Hyperparameter_Optimization_Results.csv")  # Get filename from config

        hyperparams_path = os.path.join(
            file_dir, "Classifiers_Hyperparameters", hyperparameters_filename
        )  # Path to hyperparameter optimization results

        if not verify_filepath_exists(hyperparams_path):  # If the hyperparameters file does not exist
            verbose_output(
                f"{BackgroundColors.YELLOW}No hyperparameter optimization results found at: {BackgroundColors.CYAN}{hyperparams_path}{Style.RESET_ALL}",
                config=config
            )
            return None  # Return None if no optimization results found

        try:  # Try to load the CSV file
            low_memory = config.get("execution", {}).get("low_memory", False)  # Read low memory flag from config
            df = pd.read_csv(hyperparams_path, low_memory=low_memory)  # Load the CSV into a DataFrame
            df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names
        except Exception as e:  # If there is an error loading the CSV
            print(
                f"{BackgroundColors.RED}Failed to load hyperparameter optimization file {hyperparams_path}: {e}{Style.RESET_ALL}"
            )
            return {}  # Return empty dict on failure

        combined_files_enabled = config.get("stacking", {}).get("combined_files_evaluation", True)  # Read combined_files_evaluation mode from stacking config
        verbose_output(
            f"[DEBUG] extract_hyperparameter_optimization_results: mode={'combined files evaluation' if combined_files_enabled else 'separate files evaluation'}",
            config=config
        )  # Log selected execution mode

        if combined_files_enabled:  # Route to normalized dataset-path matching for combined files evaluation mode
            verbose_output(
                "[DEBUG] extract_hyperparameter_optimization_results: using normalized dataset_path matching strategy",
                config=config
            )  # Log selected matching strategy
            matching_rows = filter_matching_hyperparams(df, csv_path, config)  # Match the combined dataset directory independently of the representative base CSV
        else:  # Separate files evaluation mode: only filter by base_csv column
            verbose_output(
                "[DEBUG] extract_hyperparameter_optimization_results: using legacy base_csv matching strategy",
                config=config
            )  # Log selected matching strategy
            matching_rows = df[df["base_csv"] == base_filename]  # Filter by base_csv column (legacy separate files evaluation mode only)

        verbose_output(
            f"[DEBUG] extract_hyperparameter_optimization_results: matched {len(matching_rows)} row(s)",
            config=config
        )  # Log total number of matched rows

        if matching_rows.empty:  # If no matching rows found
            verbose_output(
                f"{BackgroundColors.YELLOW}No hyperparameter results found for file: {BackgroundColors.CYAN}{base_filename}{BackgroundColors.YELLOW} in {BackgroundColors.CYAN}{hyperparams_path}{Style.RESET_ALL}",
                config=config
            )
            return None  # Return None if no results for this file

        results = {}  # Initialize dictionary to hold parsed results per model
        for _, row in matching_rows.iterrows():  # Iterate over each matching row
            try:  # Try to parse each row
                model = row.get("model") or row.get("Model")  # Try common column names for model identifier
                if not model:  # If model identifier is missing
                    continue  # Skip invalid rows

                best_params_raw = (
                    row.get("best_params") or row.get("best_params_json") or row.get("best_params_str")
                )  # Try common column names for best_params
                best_params = None  # Default if parsing fails or value missing
                if isinstance(best_params_raw, str) and best_params_raw.strip():  # If best_params_raw is a non-empty string
                    try:  # Try to parse best_params as JSON first
                        best_params = json.loads(best_params_raw)  # Parse JSON string if possible
                    except Exception:  # If JSON parsing fails, try ast.literal_eval as a fallback
                        try:  # Try to parse using ast.literal_eval
                            best_params = ast.literal_eval(best_params_raw)  # Safely evaluate string to Python literal
                        except Exception:  # If both parsing attempts fail
                            best_params = None  # Leave as None if parsing fails

                results[str(model)] = {"best_params": best_params}  # Store parsed parameters only
            except Exception:  # Catch any unexpected errors during row parsing
                continue  # Skip problematic rows silently

        verbose_output(
            f"{BackgroundColors.GREEN}Loaded {BackgroundColors.CYAN}{len(results)}{BackgroundColors.GREEN} hyperparameter optimization results for {BackgroundColors.CYAN}{base_filename}{BackgroundColors.GREEN} from: {BackgroundColors.CYAN}{hyperparams_path}{Style.RESET_ALL}",
            config=config
        )
        return results  # Return the normalized results mapping
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def normalize(name):
    """
    Normalize a model name by removing non-alphanumeric characters and converting to lowercase.
    
    :param name: The model name to normalize.
    :return: Normalized model name string.
    """
    
    return "".join(
        [c.lower() for c in str(name) if c.isalnum()]
    )  # Normalize model name by removing non-alphanumeric characters and converting to lowercase


def apply_hyperparameters_to_models(hyperparams_map, models_map, config=None):
    """
    Apply hyperparameter mappings to instantiated models.

    This function attempts to match keys from the hyperparameter mapping
    to the model names in the provided models dictionary. Matching is
    attempted in the following order: exact match, case-insensitive
    match, and normalized (alphanumeric-only) match. When a matching
    entry is found and the corresponding value is a valid dictionary of
    hyperparameters, they are applied to the estimator using set_params.
    Errors during matching or parameter application are handled
    gracefully and reported via verbose_output.

    :param hyperparams_map: Mapping of model name -> hyperparameter dict
    :param models_map: Mapping of model name -> instantiated estimator
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Updated models_map with applied hyperparameters where possible
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Starting to apply hyperparameters to models...{Style.RESET_ALL}",
            config=config
        )  # Inform user that application is starting

        if not hyperparams_map:  # Nothing to apply
            return models_map  # Return models unchanged

        hp_keys = list(hyperparams_map.keys())  # List of provided model names
        hp_normalized = {k: normalize(k) for k in hp_keys}  # Normalized lookup for matching

        for model_name, model in models_map.items():  # Iterate over each instantiated model
            try:  # Try to match hyperparameters to this model
                params = None  # Default if not found

                if model_name in hyperparams_map:  # Attempt exact match
                    params = hyperparams_map[model_name]  # Exact match found
                else:  # Try case-insensitive and normalized matches
                    lower_matches = [k for k in hp_keys if k.lower() == model_name.lower()]  # Case-insensitive match
                    if lower_matches:  # If case-insensitive match found
                        params = hyperparams_map[lower_matches[0]]  # Use the matched parameters
                    else:  # Try normalized match
                        norm = normalize(model_name)  # Compute normalized name
                        norm_matches = [k for k, nk in hp_normalized.items() if nk == norm]  # Normalized match
                        if norm_matches:  # If normalized match found
                            params = hyperparams_map[norm_matches[0]]  # Use the matched parameters

                if params is None:  # No parameters for this model
                    continue  # Skip to next model

                if isinstance(params, str):  # Parameters stored as string
                    try:  # Try JSON decode
                        params = json.loads(params)  # Parse JSON if possible
                    except Exception:  # Fallback to literal evaluation
                        try:  # Try to parse using ast.literal_eval
                            params = ast.literal_eval(params)  # Safely evaluate string to Python literal
                        except Exception:  # If both parsing attempts fail
                            params = None  # Leave as None if parsing fails

                if not isinstance(params, dict):  # Ensure parameters are valid dictionary
                    verbose_output(
                        f"{BackgroundColors.YELLOW}Warning: Parsed hyperparameters for {BackgroundColors.CYAN}{model_name}{BackgroundColors.YELLOW} are not a dict. Skipping.{Style.RESET_ALL}",
                        config=config
                    )
                    continue  # Skip invalid parameter entries

                try:  # Try applying parameters
                    guarded_n_jobs = config.get("evaluation", {}).get("n_jobs", 1)  # Read the currently effective n_jobs to preserve any resource guard overrides
                    if guarded_n_jobs == 1 and "n_jobs" in params:  # Prevent hyperparameter CSV from overriding the resource guard n_jobs=1 setting
                        params = {k: v for k, v in params.items() if k != "n_jobs"}  # Strip n_jobs from params to preserve the RAM-based safety limit
                        verbose_output(
                            f"{BackgroundColors.YELLOW}[RESOURCE GUARD] Stripped n_jobs from hyperparameters for {BackgroundColors.CYAN}{model_name}{BackgroundColors.YELLOW} to preserve n_jobs=1 safety limit.{Style.RESET_ALL}",
                            config=config
                        )  # Inform operator that n_jobs was removed from hyperparameters for OOM safety
                    model.set_params(**params)  # Apply parameters to estimator
                    verbose_output(
                        f"{BackgroundColors.GREEN}Applied hyperparameters to {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}",
                        config=config
                    )  # Inform success
                except Exception as e:  # If applying fails
                    print(
                        f"{BackgroundColors.YELLOW}Failed to apply hyperparameters to {BackgroundColors.CYAN}{model_name}{BackgroundColors.YELLOW}: {e}{Style.RESET_ALL}"
                    )  # Warn user
            except Exception:  # Catch any unexpected errors for this model
                continue  # Skip problematic entries silently

        return models_map  # Return updated model mapping
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def resolve_hyperparameters_for_model(model_name: str, hyperparams_map: dict) -> Optional[dict]:
    """
    Resolve optimized parameters for one classifier.

    :param model_name: Configured classifier name.
    :param hyperparams_map: Mapping of artifact classifier names to parameter dictionaries.
    :return: Parameter dictionary for the classifier, or None when unavailable.
    """

    if not hyperparams_map:  # Return no optimized parameters when the artifact mapping is empty.
        return None  # Return no optimized parameters.

    params = hyperparams_map.get(model_name, None)  # Prefer exact classifier-name matches from the artifact.
    if params is None:  # Try case-insensitive matching when an exact key is unavailable.
        lower_matches = [key for key in hyperparams_map.keys() if str(key).lower() == model_name.lower()]  # Build case-insensitive candidate keys.
        params = hyperparams_map[lower_matches[0]] if lower_matches else None  # Use the first case-insensitive match when present.

    if params is None:  # Try normalized matching when case-insensitive matching is unavailable.
        normalized_model_name = normalize(model_name)  # Normalize the configured classifier name.
        normalized_matches = [key for key in hyperparams_map.keys() if normalize(key) == normalized_model_name]  # Build normalized candidate keys.
        params = hyperparams_map[normalized_matches[0]] if normalized_matches else None  # Use the first normalized match when present.

    if isinstance(params, str):  # Decode serialized parameter dictionaries from CSV artifacts.
        try:  # Prefer JSON for exported artifact payloads.
            params = json.loads(params)  # Decode JSON parameters.
        except Exception:  # Fall back to Python literal parsing for legacy payloads.
            try:  # Parse legacy literal dictionaries safely.
                params = ast.literal_eval(params)  # Decode Python-literal parameters.
            except Exception:  # Treat undecodable payloads as unavailable.
                params = None  # Normalize invalid serialized parameters.

    if not isinstance(params, dict):  # Reject missing or invalid parameter payloads.
        return None  # Return no optimized parameters for this classifier.

    return dict(params)  # Return a defensive copy of the optimized parameters.


def build_optimized_hyperparameter_models(file_path: str, config: Optional[dict]) -> Tuple[dict, dict]:
    """
    Build optimized models for classifiers with valid artifacts.

    :param file_path: Dataset file path used to locate the hyperparameter artifact.
    :param config: Configuration dictionary, or None to use the global configuration.
    :return: Tuple of optimized model mapping and applied parameter mapping.
    """

    if config is None:  # Use global configuration when no configuration is provided.
        config = CONFIG  # Assign the global configuration reference.

    optimized_models = {}  # Accumulate classifiers with valid optimized parameters.
    optimized_params = {}  # Accumulate applied parameter dictionaries by classifier.
    active_models = get_models(config=config)  # Resolve enabled classifiers through the existing filtering logic.
    hp_results_raw = extract_hyperparameter_optimization_results(file_path, config=config)  # Load matching hyperparameter artifact rows for this context.

    if not hp_results_raw:  # Warn when no artifact data exists for the requested optimized branch.
        print(f"{BackgroundColors.YELLOW}[WARNING] Optimized hyperparameter branch skipped for {BackgroundColors.CYAN}{file_path}{BackgroundColors.YELLOW}: no matching artifact rows were found.{Style.RESET_ALL}")  # Report missing optimized artifact rows.
        return optimized_models, optimized_params  # Return empty optimized mappings.

    hp_params_map = {key: (value.get("best_params") if isinstance(value, dict) else value) for key, value in hp_results_raw.items()}  # Extract best parameter payloads by artifact classifier name.

    for model_name, model in active_models.items():  # Iterate active classifiers in configured order.
        params = resolve_hyperparameters_for_model(model_name, hp_params_map)  # Resolve valid parameters for this exact classifier.
        if params is None:  # Skip optimized mode for classifiers without valid parameters.
            print(f"{BackgroundColors.YELLOW}[WARNING] Optimized hyperparameter branch skipped for {BackgroundColors.CYAN}{model_name}{BackgroundColors.YELLOW}: no valid matching optimized-parameter artifact exists for {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}")  # Report classifier-specific skip reason.
            continue  # Continue with the next classifier.

        guarded_n_jobs = config.get("evaluation", {}).get("n_jobs", 1)  # Read the effective n_jobs resource setting.
        if guarded_n_jobs == 1 and "n_jobs" in params:  # Preserve resource guard when optimized artifacts include n_jobs.
            params = {key: value for key, value in params.items() if key != "n_jobs"}  # Remove n_jobs from optimized parameters.
            print(f"{BackgroundColors.YELLOW}[RESOURCE GUARD] Stripped n_jobs from optimized hyperparameters for {BackgroundColors.CYAN}{model_name}{BackgroundColors.YELLOW} to preserve n_jobs=1 safety limit.{Style.RESET_ALL}")  # Report resource-guard parameter removal.

        try:  # Apply the optimized parameters to the classifier instance.
            model.set_params(**params)  # Apply optimized parameters.
        except Exception as e:  # Skip this optimized classifier when parameters are incompatible.
            print(f"{BackgroundColors.YELLOW}[WARNING] Optimized hyperparameter branch skipped for {BackgroundColors.CYAN}{model_name}{BackgroundColors.YELLOW}: parameters could not be applied ({e}).{Style.RESET_ALL}")  # Report incompatible optimized parameters.
            continue  # Continue with the next classifier.

        optimized_models[model_name] = model  # Store the optimized classifier instance.
        optimized_params[model_name] = params  # Store the applied optimized parameters.

    if not optimized_models:  # Warn when every optimized classifier branch was rejected.
        print(f"{BackgroundColors.YELLOW}[WARNING] Optimized hyperparameter branch skipped for {BackgroundColors.CYAN}{file_path}{BackgroundColors.YELLOW}: no enabled classifier had valid applicable optimized parameters.{Style.RESET_ALL}")  # Report that optimized mode is unavailable.

    return optimized_models, optimized_params  # Return only classifiers with verified optimized parameters.


def build_stacking_pca_cache_context(file_path: str, source_files: List[str], feature_names: List[Any], X_train_scaled: Any, X_test_scaled: Any, scaler: Any, n_components: int, evaluation_identity: dict) -> dict:  # Build deterministic provenance for one fitted stacking PCA transformer.
    """
    Build deterministic provenance for one fitted stacking PCA transformer.

    :param file_path: Dataset file or directory identity used by stacking.
    :param source_files: Ordered source file paths used to build the evaluated dataset.
    :param feature_names: Ordered numeric input feature names before PCA.
    :param X_train_scaled: Scaled training matrix used to fit PCA.
    :param X_test_scaled: Scaled testing matrix transformed by PCA.
    :param scaler: Fitted scaler that produced the scaled matrices.
    :param n_components: Effective PCA component count.
    :param evaluation_identity: Split and experiment identity metadata.
    :return: JSON-compatible provenance dictionary used for cache identity and validation.
    """

    if not source_files:  # Require exact source-file provenance before cache reuse can be enabled.
        raise ValueError("PCA cache provenance requires an ordered non-empty source file list")  # Reject incomplete dataset provenance.
    normalized_features = [str(feature) for feature in feature_names]  # Normalize ordered feature names for deterministic comparison.
    if len(normalized_features) != int(X_train_scaled.shape[1]):  # Require feature metadata to describe every PCA input column.
        raise ValueError(f"PCA cache feature metadata has {len(normalized_features)} names for {X_train_scaled.shape[1]} columns")  # Reject ambiguous feature order.

    source_metadata = []  # Accumulate ordered source-file filesystem metadata.
    for source_index, source_file in enumerate(source_files):  # Preserve source order because combined row order affects the split and fitted PCA state.
        source_path = Path(str(source_file)).expanduser().resolve()  # Resolve one source path without changing its identity.
        if not source_path.is_file():  # Require every provenance source to exist as a regular file.
            raise FileNotFoundError(f"PCA cache source file is unavailable: {source_path}")  # Reject incomplete source provenance.
        source_stat = source_path.stat()  # Read stable filesystem metadata for compatibility validation.
        source_metadata.append({"index": source_index, "path": str(source_path), "size": int(source_stat.st_size), "mtime_ns": int(source_stat.st_mtime_ns)})  # Store ordered path, size, and nanosecond modification time.

    scaler_state = {  # Capture fitted scaling state without serializing a duplicate scaler artifact.
        "mean_": getattr(scaler, "mean_", None),  # Store the fitted per-feature means.
        "scale_": getattr(scaler, "scale_", None),  # Store the fitted per-feature scaling factors.
        "var_": getattr(scaler, "var_", None),  # Store the fitted per-feature variances.
        "n_samples_seen_": getattr(scaler, "n_samples_seen_", None),  # Store the scaler fitting sample count.
    }  # Complete the fitted scaler-state payload.
    normalized_scaler_state = normalize_metadata_for_json(scaler_state)  # Normalize fitted scaler arrays for deterministic hashing.
    scaler_state_digest = hashlib.sha256(json.dumps(normalized_scaler_state, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()  # Fingerprint the fitted scaler state used before PCA.
    expected_pca = PCA(n_components=int(n_components))  # Build the exact unfitted PCA configuration expected by this flow.
    sklearn_module = sys.modules.get("sklearn")  # Resolve the already imported scikit-learn package metadata.
    dataset_path = str(Path(str(file_path)).expanduser().resolve())  # Resolve the evaluated dataset identity to an absolute path.

    cache_context = {  # Assemble the complete compatibility identity without creation-time fields.
        "artifact_type": "stacking_fitted_pca_transformer",  # Identify the artifact as a stacking-fitted PCA transformer.
        "schema_version": 1,  # Identify the supported metadata schema.
        "dataset_path": dataset_path,  # Store the resolved stacking dataset identity.
        "source_files": source_metadata,  # Store ordered source-file provenance.
        "feature_names": normalized_features,  # Store exact pre-PCA feature order.
        "feature_count": int(X_train_scaled.shape[1]),  # Store pre-PCA input dimensionality.
        "n_components": int(n_components),  # Store effective PCA output dimensionality.
        "train_sample_count": int(X_train_scaled.shape[0]),  # Store the PCA fitting sample count.
        "test_sample_count": int(X_test_scaled.shape[0]),  # Store the transformed test sample count.
        "train_dtype": str(np.asarray(X_train_scaled).dtype),  # Store training matrix numeric representation.
        "test_dtype": str(np.asarray(X_test_scaled).dtype),  # Store testing matrix numeric representation.
        "scaling_method": f"{scaler.__class__.__module__}.{scaler.__class__.__name__}",  # Store the fitted scaler class identity.
        "scaler_params": normalize_metadata_for_json(scaler.get_params(deep=False) if hasattr(scaler, "get_params") else {}),  # Store scaler constructor semantics.
        "scaler_state_digest": scaler_state_digest,  # Store the fitted scaler-state fingerprint.
        "split_identity": normalize_metadata_for_json(evaluation_identity),  # Store split and data-variant semantics.
        "random_state": int(evaluation_identity.get("random_state", 42)),  # Store the split random seed explicitly.
        "pca_class": f"{PCA.__module__}.{PCA.__name__}",  # Store the expected transformer class identity.
        "pca_params": normalize_metadata_for_json(expected_pca.get_params(deep=False)),  # Store exact PCA constructor semantics.
        "sklearn_version": str(getattr(sklearn_module, "__version__", "unknown")),  # Store the scikit-learn compatibility version.
        "numpy_version": str(np.__version__),  # Store the NumPy compatibility version.
    }  # Complete the deterministic PCA cache context.

    return cache_context  # Return the complete deterministic compatibility context.


def resolve_stacking_pca_artifact_paths(file_path: str, cache_context: dict, config: dict) -> dict:  # Resolve dataset-local fitted PCA artifact paths using the stacking model convention.
    """
    Resolve dataset-local fitted PCA artifact paths using the stacking model convention.

    :param file_path: Dataset file or directory identity used by stacking.
    :param cache_context: Deterministic PCA compatibility context.
    :param config: Runtime configuration dictionary.
    :return: Dictionary containing stacking, model, metadata, transformer, and legacy paths.
    """

    dataset_root = resolve_dataset_root_path(str(file_path))  # Resolve the dataset-local root that owns the Stacking directory.
    dataset_identity = resolve_canonical_dataset_identity(str(dataset_root), True)  # Build the same canonical directory identity used by stacking artifacts.
    dataset_slug = build_filename_safe_dataset_identity(dataset_identity)  # Generate the existing dataset slug convention.
    stacking_directory = Path(get_stacking_output_dir(str(file_path), config)).resolve()  # Resolve the dataset-local Stacking directory.
    models_directory = (stacking_directory / "Models" / dataset_slug).resolve()  # Resolve the dataset-specific fitted model directory.
    cache_identity = hashlib.sha256(json.dumps(cache_context, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()  # Derive a stable identity that separates data variants and split contexts.
    artifact_slug = f"{int(cache_context['n_components'])}_components_{cache_identity[:12]}"  # Build a short stable PCA feature slug for coexistence across compatible variants.
    artifact_base = f"PCA Transformer__PCA Components__{artifact_slug}__"  # Match the existing model, feature-set, and feature-slug delimiter convention.
    metadata_path = (models_directory / f"{artifact_base}_meta.json").resolve()  # Build the companion provenance metadata path.
    transformer_path = (models_directory / f"{artifact_base}_transformer.joblib").resolve()  # Build the fitted PCA transformer joblib path.
    legacy_path = (dataset_root / "Cache" / f"PCA_{int(cache_context['n_components'])}_components.pkl").resolve()  # Resolve the rejected legacy bare pickle path for explicit diagnostics.
    shared_legacy_path = (dataset_root.parent / "Cache" / f"PCA_{int(cache_context['n_components'])}_components.pkl").resolve()  # Resolve the historical parent-shared bare pickle path for rejection diagnostics.
    legacy_paths = [str(legacy_path)]  # Start with the corrected dataset-local legacy location.
    if shared_legacy_path != legacy_path:  # Avoid duplicate diagnostics when both legacy path forms resolve identically.
        legacy_paths.append(str(shared_legacy_path))  # Include the historical shared-parent location without treating it as reusable.
    validate_output_path(str(stacking_directory), str(models_directory))  # Verify the model directory remains inside the dataset-local Stacking root.
    validate_output_path(str(stacking_directory), str(metadata_path))  # Verify the metadata path remains inside the dataset-local Stacking root.
    validate_output_path(str(stacking_directory), str(transformer_path))  # Verify the transformer path remains inside the dataset-local Stacking root.

    artifact_paths = {  # Assemble resolved dataset-local artifact paths and identities.
        "stacking_directory": str(stacking_directory),  # Store the dataset-specific Stacking directory.
        "models_directory": str(models_directory),  # Store the dataset-slug model directory.
        "dataset_slug": dataset_slug,  # Store the existing filename-safe dataset identity.
        "cache_identity": cache_identity,  # Store the complete deterministic cache fingerprint.
        "metadata_path": str(metadata_path),  # Store the companion metadata path.
        "transformer_path": str(transformer_path),  # Store the fitted transformer joblib path.
        "legacy_path": str(legacy_path),  # Store the rejected bare pickle path for diagnostics.
        "legacy_paths": legacy_paths,  # Store every known bare pickle location for explicit rejection diagnostics.
    }  # Complete the artifact-path payload.

    return artifact_paths  # Return all resolved artifact paths and identity fields.


def calculate_file_sha256(file_path: str) -> str:  # Calculate a streaming SHA-256 digest for one artifact file.
    """
    Calculate a streaming SHA-256 digest for one artifact file.

    :param file_path: Artifact file path to digest.
    :return: Lowercase SHA-256 hexadecimal digest.
    """

    digest = hashlib.sha256()  # Initialize the artifact integrity digest.
    with open(file_path, "rb") as file_obj:  # Open the artifact for bounded streaming reads.
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):  # Read one MiB chunks without loading the whole artifact into memory.
            digest.update(chunk)  # Add the current artifact bytes to the digest.
    return digest.hexdigest()  # Return the completed artifact digest.


def write_stacking_pca_artifact_atomically(value: Any, destination_path: str, artifact_format: str) -> None:  # Write one PCA cache artifact through a same-directory atomic replacement.
    """
    Write one PCA cache artifact through a same-directory atomic replacement.

    :param value: Python object or metadata dictionary to persist.
    :param destination_path: Final artifact path.
    :param artifact_format: Serialization format, either joblib or json.
    :return: None.
    """

    destination = Path(destination_path)  # Normalize the final artifact path.
    destination.parent.mkdir(parents=True, exist_ok=True)  # Create the dataset-specific model directory before writing.
    temporary_fd, temporary_path = tempfile.mkstemp(dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".tmp")  # Allocate the temporary file beside the final artifact.
    os.close(temporary_fd)  # Close the raw descriptor before opening it through the selected serializer.
    try:  # Ensure temporary files are removed after success or failure.
        if artifact_format == "joblib":  # Serialize fitted estimators with the existing joblib dependency.
            with open(temporary_path, "wb") as file_obj:  # Open the temporary artifact for binary serialization.
                dump(value, file_obj)  # Serialize the fitted PCA transformer into the temporary file.
                file_obj.flush()  # Flush Python buffers before publishing the artifact.
                os.fsync(file_obj.fileno())  # Flush artifact bytes to the filesystem before replacement.
        elif artifact_format == "json":  # Serialize provenance metadata as readable JSON.
            with open(temporary_path, "w", encoding="utf-8") as file_obj:  # Open the temporary metadata file with explicit encoding.
                json.dump(value, file_obj, indent=2, sort_keys=True, allow_nan=False)  # Serialize deterministic standards-compliant metadata.
                file_obj.flush()  # Flush Python buffers before publishing the metadata.
                os.fsync(file_obj.fileno())  # Flush metadata bytes to the filesystem before replacement.
        else:  # Reject unknown serialization formats.
            raise ValueError(f"Unsupported PCA artifact format: {artifact_format}")  # Prevent ambiguous artifact writes.
        os.replace(temporary_path, destination)  # Atomically publish the completed artifact in the same directory.
    finally:  # Remove only an unpublished temporary file.
        if os.path.exists(temporary_path):  # Verify whether a temporary artifact remains after replacement or failure.
            os.unlink(temporary_path)  # Remove the incomplete temporary artifact.


def validate_stacking_pca_cache_metadata(metadata: Any, expected_context: dict, artifact_paths: dict) -> Optional[str]:  # Validate provenance metadata before any transformer deserialization.
    """
    Validate provenance metadata before any transformer deserialization.

    :param metadata: Parsed PCA cache metadata payload.
    :param expected_context: Current deterministic PCA compatibility context.
    :param artifact_paths: Current expected artifact paths and cache identity.
    :return: Rejection reason string, or None when metadata is compatible.
    """

    if not isinstance(metadata, dict):  # Require a JSON object as the metadata root.
        return "metadata root is not a JSON object"  # Reject malformed metadata roots.
    for field_name, expected_value in expected_context.items():  # Compare every deterministic provenance field before loading joblib content.
        if metadata.get(field_name) != expected_value:  # Reject the first incompatible provenance field with an exact reason.
            return f"metadata field '{field_name}' does not match the current PCA input context"  # Report the incompatible field without deserializing the transformer.
    if metadata.get("cache_identity") != artifact_paths.get("cache_identity"):  # Require metadata to identify the exact artifact filename variant.
        return "cache_identity does not match the expected artifact identity"  # Reject metadata copied between cache variants.
    if metadata.get("stacking_directory") != artifact_paths.get("stacking_directory"):  # Require the recorded Stacking directory to match the current dataset root.
        return "stacking_directory does not match the current dataset artifact root"  # Reject relocated or cross-dataset metadata.
    if metadata.get("models_directory") != artifact_paths.get("models_directory"):  # Require the recorded dataset model directory to match.
        return "models_directory does not match the current dataset slug directory"  # Reject metadata from another dataset slug.
    if metadata.get("transformer_artifact") != os.path.basename(str(artifact_paths.get("transformer_path"))):  # Require the referenced joblib filename to match the expected path.
        return "transformer_artifact does not match the expected joblib filename"  # Reject filename substitution before deserialization.
    if metadata.get("scaler_artifact", "missing") is not None:  # The current PCA flow must use the freshly fitted pre-PCA scaler rather than a duplicate scaler artifact.
        return "scaler_artifact must be null for the current pre-scaled PCA flow"  # Reject incompatible scaling artifact semantics.
    transformer_sha256 = metadata.get("transformer_sha256")  # Read the expected transformer integrity digest.
    if not isinstance(transformer_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", transformer_sha256) is None:  # Require a valid lowercase SHA-256 value.
        return "transformer_sha256 is missing or invalid"  # Reject metadata without usable artifact integrity data.
    if not isinstance(metadata.get("transformer_size_bytes"), int) or metadata.get("transformer_size_bytes", 0) <= 0:  # Require a positive serialized artifact size.
        return "transformer_size_bytes is missing or invalid"  # Reject incomplete artifact metadata.
    if not isinstance(metadata.get("created_at"), str) or not metadata.get("created_at"):  # Require a creation timestamp for provenance.
        return "created_at is missing or invalid"  # Reject incomplete provenance metadata.
    if "fitted_svd_solver" not in metadata:  # Require the fitted PCA solver selected by scikit-learn.
        return "fitted_svd_solver is missing"  # Reject metadata without fitted solver provenance.
    return None  # Accept metadata only after every compatibility requirement passes.


def validate_loaded_stacking_pca_transformer(transformer: Any, expected_context: dict, metadata: dict) -> Optional[str]:  # Validate the deserialized fitted PCA object against current provenance.
    """
    Validate the deserialized fitted PCA object against current provenance.

    :param transformer: Deserialized fitted PCA object.
    :param expected_context: Current deterministic PCA compatibility context.
    :param metadata: Validated cache metadata payload.
    :return: Rejection reason string, or None when the transformer is compatible.
    """

    if not isinstance(transformer, PCA):  # Require the exact estimator family used by the stacking PCA flow.
        return f"loaded object type is {type(transformer).__name__}, expected PCA"  # Reject unrelated joblib content.
    if normalize_metadata_for_json(transformer.get_params(deep=False)) != expected_context.get("pca_params"):  # Require constructor parameters to match current PCA semantics.
        return "loaded PCA parameters do not match the current PCA configuration"  # Reject parameter drift.
    if int(getattr(transformer, "n_components_", -1)) != int(expected_context.get("n_components", -1)):  # Require the fitted output dimensionality to match.
        return "loaded PCA fitted component count does not match"  # Reject fitted dimensionality drift.
    if int(getattr(transformer, "n_features_in_", -1)) != int(expected_context.get("feature_count", -1)):  # Require the fitted input width to match ordered feature provenance.
        return "loaded PCA input feature count does not match"  # Reject fitted input-width drift.
    components = getattr(transformer, "components_", None)  # Read the fitted component matrix for structural validation.
    expected_shape = (int(expected_context.get("n_components", -1)), int(expected_context.get("feature_count", -1)))  # Build the required component matrix shape.
    if not isinstance(components, np.ndarray) or components.shape != expected_shape:  # Require a complete fitted component matrix.
        return f"loaded PCA component matrix shape does not match {expected_shape}"  # Reject incomplete or malformed fitted state.
    if metadata.get("fitted_svd_solver") != getattr(transformer, "_fit_svd_solver", None):  # Require the serialized fitted solver to match metadata.
        return "loaded PCA fitted solver does not match metadata"  # Reject inconsistent fitted-state provenance.
    return None  # Accept the loaded transformer only after structural validation passes.


def load_stacking_pca_cache(expected_context: dict, artifact_paths: dict, config: Optional[dict] = None) -> Optional[PCA]:  # Load a fitted PCA transformer only after strict metadata and integrity validation.
    """
    Load a fitted PCA transformer only after strict metadata and integrity validation.

    :param expected_context: Current deterministic PCA compatibility context.
    :param artifact_paths: Current expected artifact paths and cache identity.
    :param config: Runtime configuration dictionary.
    :return: Compatible fitted PCA transformer, or None when reuse is unsafe.
    """

    metadata_path = str(artifact_paths["metadata_path"])  # Read the expected companion metadata path.
    transformer_path = str(artifact_paths["transformer_path"])  # Read the expected fitted transformer path.
    legacy_paths = [str(path) for path in artifact_paths.get("legacy_paths", [artifact_paths["legacy_path"]])]  # Read every known rejected legacy bare pickle path.
    for legacy_path in legacy_paths:  # Inspect legacy locations only for explicit rejection diagnostics.
        if os.path.exists(legacy_path):  # Detect an old bare pickle without opening or deserializing it.
            print(f"{BackgroundColors.YELLOW}[WARNING] Legacy bare PCA pickle rejected because provenance metadata is unavailable: {BackgroundColors.CYAN}{legacy_path}{Style.RESET_ALL}")  # Explain why the legacy cache is never reused.
    if not os.path.isfile(metadata_path):  # Require companion provenance before considering any joblib artifact.
        orphan_text = " An orphan transformer joblib was also found and rejected." if os.path.isfile(transformer_path) else ""  # Describe an unsafe transformer without metadata.
        print(f"{BackgroundColors.YELLOW}[INFO] Compatible fitted PCA cache not found because metadata is missing at: {BackgroundColors.CYAN}{metadata_path}{BackgroundColors.YELLOW}.{orphan_text}{Style.RESET_ALL}")  # Report the exact cache miss reason.
        return None  # Fit a new transformer when metadata is unavailable.
    try:  # Parse metadata before loading executable joblib content.
        with open(metadata_path, "r", encoding="utf-8") as file_obj:  # Open the companion provenance metadata.
            metadata = json.load(file_obj)  # Parse the metadata JSON payload.
    except Exception as exc:  # Treat corrupted metadata as a safe cache miss.
        print(f"{BackgroundColors.YELLOW}[WARNING] Fitted PCA cache rejected because metadata could not be parsed at {BackgroundColors.CYAN}{metadata_path}{BackgroundColors.YELLOW}: {exc}{Style.RESET_ALL}")  # Report corrupted metadata without stopping evaluation.
        return None  # Fit a new transformer after metadata corruption.

    rejection_reason = validate_stacking_pca_cache_metadata(metadata, expected_context, artifact_paths)  # Validate path-independent provenance before deserialization.
    if rejection_reason is not None:  # Reject stale or incompatible metadata explicitly.
        print(f"{BackgroundColors.YELLOW}[WARNING] Fitted PCA cache rejected: {rejection_reason}. Metadata: {BackgroundColors.CYAN}{metadata_path}{Style.RESET_ALL}")  # Report the exact metadata incompatibility.
        return None  # Fit a new transformer after metadata rejection.
    if not os.path.isfile(transformer_path):  # Require the metadata-referenced transformer artifact.
        print(f"{BackgroundColors.YELLOW}[WARNING] Fitted PCA cache rejected because transformer joblib is missing at: {BackgroundColors.CYAN}{transformer_path}{Style.RESET_ALL}")  # Report the missing serialized transformer.
        return None  # Fit a new transformer when joblib content is unavailable.
    actual_size = int(os.path.getsize(transformer_path))  # Read the serialized transformer size before deserialization.
    if actual_size != int(metadata["transformer_size_bytes"]):  # Require the artifact size recorded during atomic save.
        print(f"{BackgroundColors.YELLOW}[WARNING] Fitted PCA cache rejected because transformer size changed: expected {metadata['transformer_size_bytes']}, found {actual_size}.{Style.RESET_ALL}")  # Report artifact truncation or replacement.
        return None  # Fit a new transformer after integrity failure.
    actual_sha256 = calculate_file_sha256(transformer_path)  # Calculate the transformer integrity digest before joblib loading.
    if actual_sha256 != metadata["transformer_sha256"]:  # Require exact artifact bytes written with the validated metadata.
        print(f"{BackgroundColors.YELLOW}[WARNING] Fitted PCA cache rejected because transformer SHA-256 does not match metadata at: {BackgroundColors.CYAN}{transformer_path}{Style.RESET_ALL}")  # Report byte-level artifact corruption.
        return None  # Fit a new transformer after integrity failure.
    try:  # Deserialize only after metadata and artifact integrity validation succeed.
        transformer = load(transformer_path)  # Load the fitted PCA transformer through the existing joblib dependency.
    except Exception as exc:  # Treat corrupted or incompatible joblib content as a safe cache miss.
        print(f"{BackgroundColors.YELLOW}[WARNING] Fitted PCA cache rejected because transformer joblib could not be loaded at {BackgroundColors.CYAN}{transformer_path}{BackgroundColors.YELLOW}: {exc}{Style.RESET_ALL}")  # Report deserialization failure without stopping evaluation.
        return None  # Fit a new transformer after joblib corruption.
    rejection_reason = validate_loaded_stacking_pca_transformer(transformer, expected_context, metadata)  # Validate fitted estimator type, parameters, and learned state.
    if rejection_reason is not None:  # Reject incompatible deserialized state.
        print(f"{BackgroundColors.YELLOW}[WARNING] Fitted PCA cache rejected: {rejection_reason}. Transformer: {BackgroundColors.CYAN}{transformer_path}{Style.RESET_ALL}")  # Report the exact fitted-state incompatibility.
        return None  # Fit a new transformer after fitted-state rejection.
    print(f"{BackgroundColors.GREEN}[INFO] Compatible fitted PCA transformer cache loaded from: {BackgroundColors.CYAN}{transformer_path}{Style.RESET_ALL}")  # Confirm safe reuse after all validation stages.
    return cast(PCA, transformer)  # Return the strictly validated fitted PCA transformer.


def save_stacking_pca_cache(transformer: PCA, cache_context: dict, artifact_paths: dict, config: Optional[dict] = None) -> bool:  # Atomically save a fitted PCA transformer and companion provenance metadata.
    """
    Atomically save a fitted PCA transformer and companion provenance metadata.

    :param transformer: Fitted PCA transformer to persist.
    :param cache_context: Deterministic PCA compatibility context.
    :param artifact_paths: Resolved dataset-local artifact paths.
    :param config: Runtime configuration dictionary.
    :return: True when transformer and metadata were published successfully.
    """

    try:  # Keep cache persistence best-effort so scientific evaluation can continue after write failures.
        fitted_solver = getattr(transformer, "_fit_svd_solver", None)  # Record the solver selected by scikit-learn during fitting.
        fitted_metadata = {"fitted_svd_solver": fitted_solver}  # Build the fitted-state metadata required by estimator validation.
        rejection_reason = validate_loaded_stacking_pca_transformer(transformer, cache_context, fitted_metadata)  # Validate the fitted object before writing any final artifact.
        if rejection_reason is not None:  # Refuse to persist incomplete or incompatible fitted state.
            raise ValueError(rejection_reason)  # Route invalid fitted state through the non-fatal save warning.
        models_directory = str(artifact_paths["models_directory"])  # Read the dataset-specific Stacking model directory.
        Path(models_directory).mkdir(parents=True, exist_ok=True)  # Create the existing model artifact directory convention.
        transformer_path = str(artifact_paths["transformer_path"])  # Read the final transformer joblib path.
        metadata_path = str(artifact_paths["metadata_path"])  # Read the final companion metadata path.
        write_stacking_pca_artifact_atomically(transformer, transformer_path, "joblib")  # Publish the fitted transformer atomically before its metadata.
        transformer_size = int(os.path.getsize(transformer_path))  # Record the published transformer size for integrity validation.
        transformer_sha256 = calculate_file_sha256(transformer_path)  # Record the published transformer SHA-256 before metadata publication.
        metadata = dict(cache_context)  # Copy deterministic provenance without mutating caller-owned context.
        metadata.update({  # Add creation, path, integrity, and fitted-state metadata.
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),  # Store an explicit UTC artifact creation timestamp.
            "stacking_directory": str(artifact_paths["stacking_directory"]),  # Store the resolved dataset-local Stacking directory.
            "models_directory": models_directory,  # Store the resolved dataset-slug model directory.
            "cache_identity": str(artifact_paths["cache_identity"]),  # Store the deterministic provenance fingerprint.
            "transformer_artifact": os.path.basename(transformer_path),  # Store the exact companion transformer filename.
            "transformer_size_bytes": transformer_size,  # Store serialized transformer size for integrity validation.
            "transformer_sha256": transformer_sha256,  # Store serialized transformer content integrity.
            "scaler_artifact": None,  # Record that no duplicate scaler artifact is required by the pre-scaled PCA flow.
            "fitted_svd_solver": fitted_solver,  # Store the solver selected during PCA fitting.
        })  # Complete fitted artifact metadata without changing deterministic context fields.
        write_stacking_pca_artifact_atomically(metadata, metadata_path, "json")  # Publish metadata atomically only after the transformer is complete.
        print(f"{BackgroundColors.GREEN}[INFO] Provenance-bearing fitted PCA cache saved successfully. Metadata: {BackgroundColors.CYAN}{metadata_path}{BackgroundColors.GREEN}. Transformer: {BackgroundColors.CYAN}{transformer_path}{Style.RESET_ALL}")  # Confirm both final artifacts were published.
        return True  # Report successful cache persistence.
    except Exception as exc:  # Preserve PCA evaluation when artifact persistence fails.
        print(f"{BackgroundColors.YELLOW}[WARNING] Failed to save fitted PCA transformer cache; evaluation will continue with the in-memory transformer: {exc}{Style.RESET_ALL}")  # Report the non-fatal cache save failure.
        return False  # Report that no reusable cache was published.


def apply_pca_transformation(X_train_scaled, X_test_scaled, pca_n_components, file_path=None, config=None, feature_names=None, scaler=None, source_files=None, cache_context=None):  # Apply or safely reuse the fitted PCA transformation for one evaluation split.
    """
    Applies Principal Component Analysis (PCA) transformation to the scaled training
    and testing datasets using the optimal number of components.

    First attempts to load a pre-fitted PCA object from disk. If not found,
    fits a new PCA model on the training data.

    :param X_train_scaled: Scaled training features (numpy array).
    :param X_test_scaled: Scaled testing features (numpy array).
    :param pca_n_components: Optimal number of components (integer), or None/0 if PCA is skipped.
    :param file_path: Path to the dataset CSV file (optional, for loading pre-fitted PCA).
    :param config: Configuration dictionary (uses global CONFIG if None)
    :param feature_names: Ordered numeric feature names supplied to the scaled matrices.
    :param scaler: Fitted scaler that produced the scaled matrices.
    :param source_files: Ordered source files used to construct the evaluated data.
    :param cache_context: Split and experiment identity metadata for compatibility validation.
    :return: Tuple (X_train_pca, X_test_pca, pca) - Transformed features and fitted transformer, or null values.
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        X_train_pca = None  # Initialize PCA training features
        X_test_pca = None  # Initialize PCA testing features
        pca = None  # Initialize the fitted PCA transformer returned for classifier persistence.

        if pca_n_components is not None and pca_n_components > 0:  # If PCA components are specified
            n_features = X_train_scaled.shape[1]  # Get the number of features in the training set
            n_components = min(
                pca_n_components, n_features
            )  # Effective number of components cannot exceed number of features

            if n_components < pca_n_components:  # Verify if the component count was reduced
                print(
                    f"{BackgroundColors.YELLOW}Warning: Reduced PCA components from {pca_n_components} to {n_components} due to limited features ({n_features}).{Style.RESET_ALL}"
                )

            dataset_reference = str(file_path) if file_path else "unspecified dataset"  # Resolve the dataset identity shown in stage logs and Telegram.
            dataset_scope = "combined dataset" if file_path and resolve_path_represents_directory(str(file_path)) else "dataset"  # Describe directory-backed evaluation accurately.
            resolved_source_files = list(source_files) if source_files is not None else ([str(file_path)] if file_path and not resolve_path_represents_directory(str(file_path)) else [])  # Use exact caller provenance or the single evaluated dataset file.
            pca_cache_context = None  # Initialize cache provenance as unavailable until every required field is built.
            pca_artifact_paths = None  # Initialize artifact paths as unavailable until cache identity succeeds.
            try:  # Disable cache reuse safely when complete provenance cannot be built.
                pca_cache_context = build_stacking_pca_cache_context(str(file_path), resolved_source_files, list(feature_names or []), X_train_scaled, X_test_scaled, scaler, n_components, dict(cache_context or {}))  # Build strict dataset, split, scaling, and PCA compatibility provenance.
                pca_artifact_paths = resolve_stacking_pca_artifact_paths(str(file_path), pca_cache_context, config)  # Resolve artifacts under the existing dataset-specific Stacking model directory.
            except Exception as exc:  # Continue with a fresh in-memory fit when provenance is incomplete.
                print(f"{BackgroundColors.YELLOW}[WARNING] Fitted PCA cache reuse is unavailable because provenance could not be built: {exc}{Style.RESET_ALL}")  # Report why safe cache reuse is disabled.

            metadata_path_text = str(pca_artifact_paths["metadata_path"]) if pca_artifact_paths is not None else "unavailable"  # Resolve metadata path text for stage logging.
            transformer_path_text = str(pca_artifact_paths["transformer_path"]) if pca_artifact_paths is not None else "unavailable"  # Resolve transformer path text for stage logging.
            requested_feature_extraction_n_jobs, effective_feature_extraction_n_jobs, feature_extraction_adjustment_reason = resolve_feature_extraction_n_jobs(config=config)  # Resolve independent requested and RAM-safe PCA thread counts.
            print(f"{BackgroundColors.GREEN}[INFO] Starting PCA feature extraction for {dataset_scope} using {BackgroundColors.CYAN}{n_components}{BackgroundColors.GREEN} components. PCA_Results.csv selected: {BackgroundColors.CYAN}{pca_n_components}{BackgroundColors.GREEN}. Dataset: {BackgroundColors.CYAN}{dataset_reference}{Style.RESET_ALL}")  # Announce the selected and effective component counts before transformation.
            print(f"{BackgroundColors.GREEN}[INFO] Feature extraction n_jobs requested: {BackgroundColors.CYAN}{requested_feature_extraction_n_jobs}{Style.RESET_ALL}")  # Report the user-controlled feature extraction request.
            print(f"{BackgroundColors.GREEN}[INFO] Feature extraction n_jobs effective: {BackgroundColors.CYAN}{effective_feature_extraction_n_jobs}{Style.RESET_ALL}")  # Report the effective thread limit used for PCA numerical work.
            if feature_extraction_adjustment_reason is not None:  # Report any CPU or RAM safety reduction once at the PCA stage.
                print(f"{BackgroundColors.YELLOW}[WARNING] Feature extraction n_jobs reduced from {BackgroundColors.CYAN}{requested_feature_extraction_n_jobs}{BackgroundColors.YELLOW} to {BackgroundColors.CYAN}{effective_feature_extraction_n_jobs}{BackgroundColors.YELLOW} because {feature_extraction_adjustment_reason}.{Style.RESET_ALL}")  # Explain why the effective feature extraction value differs.
            print(f"{BackgroundColors.GREEN}[INFO] Applying feature extraction thread limit during PCA fit/transform: {BackgroundColors.CYAN}{effective_feature_extraction_n_jobs}{Style.RESET_ALL}")  # Announce the narrowly scoped numerical thread limit.
            print(f"{BackgroundColors.GREEN}[INFO] Expected fitted PCA cache metadata: {BackgroundColors.CYAN}{metadata_path_text}{Style.RESET_ALL}")  # Report the companion provenance path before lookup.
            print(f"{BackgroundColors.GREEN}[INFO] Expected fitted PCA transformer: {BackgroundColors.CYAN}{transformer_path_text}{Style.RESET_ALL}")  # Report the joblib transformer path before lookup.
            telegram_adjustment_text = f"\nAdjustment reason: {feature_extraction_adjustment_reason}." if feature_extraction_adjustment_reason is not None else ""  # Format an optional RAM or CPU adjustment line without adding another Telegram message.
            send_telegram_message(TELEGRAM_BOT, f"PCA feature extraction/transformation started for {dataset_scope}.\nComponents selected from PCA_Results.csv: {pca_n_components}.\nComponents used: {n_components}.\nRequested feature extraction jobs: {requested_feature_extraction_n_jobs}.\nEffective feature extraction jobs: {effective_feature_extraction_n_jobs}.{telegram_adjustment_text}\nDataset: {dataset_reference}\nFitted PCA metadata will be attempted at: {metadata_path_text}\nA compatible fitted transformer will be reused before fitting when provenance matches.")  # Send one stage-level PCA notification through the existing Telegram path.

            pca = load_stacking_pca_cache(pca_cache_context, pca_artifact_paths, config=config) if pca_cache_context is not None and pca_artifact_paths is not None else None  # Load only a strictly compatible provenance-bearing fitted transformer.
            if pca is None:  # Fit PCA only when no compatible fitted transformer cache is available.
                print(f"{BackgroundColors.GREEN}[INFO] Compatible fitted PCA transformer cache was not loaded. Fitting PCA and transforming the training matrix now using {BackgroundColors.CYAN}{n_components}{BackgroundColors.GREEN} components.{Style.RESET_ALL}")  # State why computation begins after cache lookup.
                pca = PCA(n_components=n_components)  # Initialize PCA with the effective number of components.
                with threadpool_limits(limits=effective_feature_extraction_n_jobs):  # Limit only PCA BLAS and OpenMP work without changing process-global defaults permanently.
                    X_train_pca = pca.fit_transform(X_train_scaled)  # Fit exact PCA on training data only and transform that same matrix.
                if pca_cache_context is not None and pca_artifact_paths is not None:  # Persist only when complete provenance and safe dataset-local paths exist.
                    print(f"{BackgroundColors.GREEN}[INFO] Saving provenance-bearing fitted PCA transformer under: {BackgroundColors.CYAN}{pca_artifact_paths['models_directory']}{Style.RESET_ALL}")  # Announce the existing Stacking model artifact directory.
                    save_stacking_pca_cache(pca, pca_cache_context, pca_artifact_paths, config=config)  # Atomically save transformer and companion metadata without changing evaluation semantics.
            else:  # Reuse the validated fitted transformer without fitting again.
                with threadpool_limits(limits=effective_feature_extraction_n_jobs):  # Limit only cached PCA training transformation numerical work.
                    X_train_pca = pca.transform(X_train_scaled)  # Transform the current training matrix using the compatible cached fitted PCA.

            with threadpool_limits(limits=effective_feature_extraction_n_jobs):  # Limit only PCA testing transformation numerical work.
                X_test_pca = pca.transform(X_test_scaled)  # Transform the testing data with the same fitted PCA state.

            print(f"{BackgroundColors.GREEN}[INFO] PCA feature extraction completed. Transformed training data shape: {BackgroundColors.CYAN}{X_train_pca.shape}{BackgroundColors.GREEN}. Transformed test data shape: {BackgroundColors.CYAN}{X_test_pca.shape}{Style.RESET_ALL}")  # Report both dense transformed matrix shapes before model evaluation.

        return X_train_pca, X_test_pca, pca  # Return transformed features and the fitted preprocessing transformer.
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_feature_subset(X_scaled, features, feature_names):
    """
    Returns a subset of features from the scaled feature set based on the provided feature names.
    Also returns the actual feature names that were successfully selected.

    :param X_scaled: Scaled features (numpy array).
    :param features: List of feature names to select.
    :param feature_names: List of all feature names corresponding to columns in X_scaled.
    :return: Tuple of (subset array, list of actual selected feature names)
    """
    
    try:
        if features:  # Only proceed if the list of selected features is NOT empty/None
            indices = []  # List to store indices of selected features
            selected_names = []  # List to store names of selected features
            feature_index_map = {feature_name: idx for idx, feature_name in enumerate(feature_names)}  # Build one positional lookup table for deterministic column selection
            for f in features:  # Iterate over each feature in the provided list
                if f in feature_index_map:  # Verify if the feature exists in the full feature list
                    indices.append(feature_index_map[f])  # Append the index of the feature
                    selected_names.append(f)  # Append the name of the feature
            subset = np.take(X_scaled, np.asarray(indices, dtype=np.intp), axis=1)  # Materialize selected columns in C-contiguous order instead of NumPy's column-advanced F-contiguous copy
            if not subset.flags.c_contiguous:  # Defensive guard for array subclasses or unusual input layouts
                subset = np.ascontiguousarray(subset)  # Ensure downstream estimators do not create an additional order-conversion copy
            return subset, selected_names  # Return the subset and actual names
        else:  # If no features are selected (or features is None)
            return np.empty((X_scaled.shape[0], 0)), []  # Return empty array and empty list
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_array_memory_metadata(prefix: str, array: Any) -> dict:
    """
    Build compact memory/layout metadata for a feature matrix.

    :param prefix: Prefix for emitted metadata keys.
    :param array: Array-like object to inspect.
    :return: Dictionary containing shape, dtype, bytes, and layout flags.
    """

    metadata: dict[str, Any] = {}  # Accumulate JSON-safe metadata fields
    try:
        arr = np.asarray(array)  # Normalize array-like objects without requesting a copy
        metadata[f"{prefix}_array_type"] = type(array).__name__  # Store original container type
        metadata[f"{prefix}_shape"] = list(arr.shape)  # Store matrix shape
        metadata[f"{prefix}_dtype"] = str(arr.dtype)  # Store dtype string
        metadata[f"{prefix}_nbytes"] = int(arr.nbytes)  # Store exact byte count
        metadata[f"{prefix}_nbytes_gb"] = round(arr.nbytes / (1024 ** 3), 4)  # Store byte count in GiB
        metadata[f"{prefix}_c_contiguous"] = bool(arr.flags.c_contiguous)  # Store C-contiguous flag
        metadata[f"{prefix}_f_contiguous"] = bool(arr.flags.f_contiguous)  # Store F-contiguous flag
        metadata[f"{prefix}_writeable"] = bool(arr.flags.writeable)  # Store writeability flag
        metadata[f"{prefix}_base_type"] = type(arr.base).__name__ if arr.base is not None else None  # Store base owner type when present
    except Exception as exc:  # Keep diagnostics best-effort
        metadata[f"{prefix}_metadata_error"] = str(exc)  # Report metadata failure without interrupting training
    return metadata  # Return compact array metadata


def get_array_nbytes(array: Any) -> int:
    """
    Return the byte footprint of an array-like object.

    :param array: Array-like object to inspect.
    :return: Number of bytes occupied by the array data buffer, or zero when unavailable.
    """

    try:
        return int(np.asarray(array).nbytes)  # Return NumPy-reported data bytes without requesting a copy
    except Exception:
        return 0  # Treat unknown objects as zero-byte for spill threshold decisions


def resolve_feature_source_spill_base_directory(file: str, config: Optional[dict] = None) -> Path:
    """
    Resolve the base directory used for temporary feature-source memmap files.

    :param file: Dataset file or directory identity for output-root resolution.
    :param config: Runtime configuration dictionary.
    :return: Existing base directory for temporary spill subdirectories.
    """

    if config is None:  # Use global configuration when no explicit configuration is supplied
        config = CONFIG  # Assign global configuration
    memory_cfg = config.get("stacking", {}).get("memory_management", {})  # Read memory management settings
    configured_directory = memory_cfg.get("spill_directory", None)  # Read optional explicit spill directory
    if configured_directory:  # If operator configured a spill directory
        base_directory = Path(str(configured_directory)).expanduser()  # Resolve configured directory path
    else:  # Use the dataset's stacking output area by default
        base_directory = Path(get_stacking_output_dir(str(file), config)) / "Array_Cache"  # Build dataset-local temporary array cache directory
    base_directory.mkdir(parents=True, exist_ok=True)  # Ensure base directory exists
    return base_directory  # Return base directory path


def spill_array_to_memmap(array: Any, temp_dir: str, name: str) -> Tuple[np.memmap, str]:
    """
    Copy one feature-source array into a temporary C-contiguous memmap.

    :param array: Source array to spill.
    :param temp_dir: Temporary directory that owns the memmap file.
    :param name: Logical array name used in the file name.
    :return: Tuple of memmap array and backing file path.
    """

    source = np.asarray(array)  # Normalize source without changing dtype
    if source.ndim != 2:  # Feature-source arrays must be two-dimensional matrices
        raise ValueError(f"Feature source array {name} must be 2D, got shape {source.shape}")  # Raise explicit shape error
    file_path = os.path.join(temp_dir, f"{name}_{uuid.uuid4().hex}.dat")  # Build collision-resistant backing file path
    memmap_array = np.memmap(file_path, dtype=source.dtype, mode="w+", shape=source.shape, order="C")  # Allocate disk-backed C-contiguous matrix
    memmap_array[:] = source  # Copy exact source values into the memmap without dtype conversion
    memmap_array.flush()  # Flush backing bytes before releasing the source array
    return memmap_array, file_path  # Return memmap and backing file path


def cleanup_feature_source_arrays(feature_source_arrays: Optional[dict], config: Optional[dict] = None) -> None:
    """
    Release feature-source arrays and remove any temporary spill files.

    :param feature_source_arrays: Mutable holder created by evaluate_on_dataset.
    :param config: Runtime configuration dictionary.
    :return: None.
    """

    if not feature_source_arrays:  # Nothing to clean
        return  # Exit early
    spill_temp_dir = feature_source_arrays.get("spill_temp_dir")  # Resolve temporary spill directory if one was created
    try:
        feature_source_arrays["X_train_scaled"] = None  # Drop train source reference before deleting memmap files
        feature_source_arrays["X_test_scaled"] = None  # Drop test source reference before deleting memmap files
        gc.collect()  # Reclaim ndarray or memmap objects before filesystem cleanup
        if spill_temp_dir and os.path.isdir(spill_temp_dir):  # Remove only this run's unique temporary directory
            shutil.rmtree(spill_temp_dir, ignore_errors=True)  # Delete temporary memmap files best-effort
    except Exception as exc:  # Keep cleanup best-effort
        print(f"{BackgroundColors.YELLOW}[WARNING] Failed to clean feature-source spill files at {spill_temp_dir}: {exc}{Style.RESET_ALL}")  # Log cleanup failure
    finally:
        feature_source_arrays.clear()  # Clear holder so repeated cleanup is a no-op


def maybe_spill_feature_source_arrays(feature_source_arrays: dict, file: str, config: Optional[dict] = None) -> None:
    """
    Spill full scaled source matrices to disk-backed memmaps after Full Features.

    :param feature_source_arrays: Mutable holder containing X_train_scaled and X_test_scaled.
    :param file: Dataset file or directory identity for resolving spill location.
    :param config: Runtime configuration dictionary.
    :return: None.
    """

    if config is None:  # Use global configuration when no explicit configuration is supplied
        config = CONFIG  # Assign global configuration
    memory_cfg = config.get("stacking", {}).get("memory_management", {})  # Read memory management settings
    if not bool(memory_cfg.get("spill_full_feature_arrays_after_full_eval", True)):  # Honor operator disable switch
        return  # Leave arrays resident in memory
    if feature_source_arrays.get("spilled_to_memmap", False):  # Avoid duplicate spilling
        return  # Already spilled

    X_train_source = feature_source_arrays.get("X_train_scaled")  # Read current full train source matrix
    X_test_source = feature_source_arrays.get("X_test_scaled")  # Read current full test source matrix
    if X_train_source is None or X_test_source is None:  # Nothing usable to spill
        return  # Exit early
    total_nbytes = get_array_nbytes(X_train_source) + get_array_nbytes(X_test_source)  # Compute combined source matrix footprint
    min_nbytes = int(memory_cfg.get("spill_min_array_nbytes", 1073741824) or 0)  # Resolve threshold with zero as explicit always-spill
    if total_nbytes < min_nbytes:  # Skip small datasets where disk spill adds no value
        return  # Leave arrays in memory

    temp_dir = None  # Track temp dir for failure cleanup
    try:
        base_directory = resolve_feature_source_spill_base_directory(file, config=config)  # Resolve spill base directory
        temp_dir = tempfile.mkdtemp(prefix="FeatureSource_", dir=str(base_directory))  # Create unique spill directory for this evaluation slice
        X_train_memmap, train_path = spill_array_to_memmap(X_train_source, temp_dir, "X_train_scaled")  # Spill train source matrix
        X_test_memmap, test_path = spill_array_to_memmap(X_test_source, temp_dir, "X_test_scaled")  # Spill test source matrix
        feature_source_arrays["X_train_scaled"] = X_train_memmap  # Replace in-memory train matrix with memmap-backed matrix
        feature_source_arrays["X_test_scaled"] = X_test_memmap  # Replace in-memory test matrix with memmap-backed matrix
        feature_source_arrays["spilled_to_memmap"] = True  # Mark holder as spilled
        feature_source_arrays["spill_temp_dir"] = temp_dir  # Store unique spill directory for cleanup
        feature_source_arrays["spill_paths"] = [train_path, test_path]  # Store backing paths for diagnostics
        del X_train_source, X_test_source  # Release old in-memory source references held by this helper
        gc.collect()  # Reclaim old in-memory full matrices before the next feature subset is materialized

        event_metadata = {"dataset_source": file, "spill_temp_dir": temp_dir, "spilled_total_nbytes": total_nbytes, "spilled_total_nbytes_gb": round(total_nbytes / (1024 ** 3), 4), "event_outcome": "completed"}  # Build compact spill event metadata
        if memory_watcher_enabled(config):  # Add array layout details only when diagnostics are active
            event_metadata.update(build_array_memory_metadata("feature_source_train", feature_source_arrays["X_train_scaled"]))  # Add train memmap metadata
            event_metadata.update(build_array_memory_metadata("feature_source_test", feature_source_arrays["X_test_scaled"]))  # Add test memmap metadata
        write_memory_phase_event("after_feature_source_spill", config=config, **event_metadata)  # Publish spill completion event
        verbose_output(f"{BackgroundColors.GREEN}[MEMORY] Spilled full feature source matrices ({total_nbytes / (1024 ** 3):.2f} GiB) to temporary memmaps at {BackgroundColors.CYAN}{temp_dir}{Style.RESET_ALL}", config=config)  # Log spill path when verbose
    except Exception as exc:  # Keep spill failure non-fatal so experiments preserve behavior
        if temp_dir and os.path.isdir(temp_dir):  # Clean failed partial spill directory
            shutil.rmtree(temp_dir, ignore_errors=True)  # Remove partial memmap files best-effort
        write_memory_phase_event("after_feature_source_spill", config=config, dataset_source=file, spilled_total_nbytes=total_nbytes, event_outcome=f"failed:{exc}")  # Publish failed spill event when watcher is active
        print(f"{BackgroundColors.YELLOW}[WARNING] Feature-source memmap spill failed; continuing with in-memory arrays: {exc}{Style.RESET_ALL}")  # Warn operator that memory-saving spill is unavailable


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


def build_stacking_model_artifact_context(dataset_csv_path: str, source_files: List[str], execution_mode_str: str, attack_types_combined: Any, target_column: str, model_name: str, model: Any, feature_set: str, input_feature_names: List[Any], model_feature_names: List[Any], label_classes: List[Any], transformer: Any, hyperparameters_enabled: bool) -> dict:
    """
    Build the deterministic compatibility identity for one original-trained classifier.

    :param dataset_csv_path: Dataset file or directory identity.
    :param source_files: Ordered original source files used to build the dataset.
    :param execution_mode_str: Separate-files or combined-files execution mode.
    :param attack_types_combined: Combined classification scope, or None.
    :param target_column: Positional target column name.
    :param model_name: Configured classifier name.
    :param model: Unfitted or fitted classifier exposing its effective parameters.
    :param feature_set: Feature-set and hyperparameter-mode identity.
    :param input_feature_names: Ordered numeric schema before fitted preprocessing.
    :param model_feature_names: Ordered schema presented to the classifier.
    :param label_classes: Ordered classes represented by the fitted label encoder.
    :param transformer: Optional fitted or equivalently configured PCA transformer.
    :param hyperparameters_enabled: Whether optimized hyperparameters are active.
    :return: JSON-compatible artifact compatibility context.
    """

    if not source_files:  # Require original source provenance for model reuse.
        raise ValueError("Classifier artifact provenance requires original source files")
    source_metadata = []  # Accumulate ordered original-file metadata.
    for source_index, source_file in enumerate(source_files):
        source_path = Path(str(source_file)).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Classifier artifact source file is unavailable: {source_path}")
        source_stat = source_path.stat()
        source_metadata.append({"index": source_index, "path": str(source_path), "size": int(source_stat.st_size), "mtime_ns": int(source_stat.st_mtime_ns)})
    sklearn_module = sys.modules.get("sklearn")
    model_params = model.get_params(deep=True) if hasattr(model, "get_params") else {}
    transformer_metadata = None
    if transformer is not None:
        transformer_metadata = {
            "class": f"{transformer.__class__.__module__}.{transformer.__class__.__name__}",
            "params": normalize_metadata_for_json(transformer.get_params(deep=False) if hasattr(transformer, "get_params") else {}),
        }
    return {
        "artifact_type": "stacking_original_trained_classifier",
        "schema_version": 1,
        "training_data": "original_only",
        "dataset_path": str(Path(str(dataset_csv_path)).expanduser().resolve()),
        "source_files": source_metadata,
        "execution_mode": str(execution_mode_str),
        "attack_types": normalize_metadata_for_json(attack_types_combined),
        "target_column": str(target_column),
        "input_feature_names": [str(feature) for feature in input_feature_names],
        "model_feature_names": [str(feature) for feature in model_feature_names],
        "feature_set": str(feature_set),
        "hyperparameter_mode": "optimized" if hyperparameters_enabled else "default",
        "model_name": str(model_name),
        "model_class": f"{model.__class__.__module__}.{model.__class__.__name__}",
        "model_params": normalize_metadata_for_json(model_params),
        "scaler_class": f"{StandardScaler.__module__}.{StandardScaler.__name__}",
        "scaler_params": normalize_metadata_for_json(StandardScaler().get_params(deep=False)),
        "label_encoder_class": f"{LabelEncoder.__module__}.{LabelEncoder.__name__}",
        "label_classes": normalize_metadata_for_json(label_classes),
        "transformer": transformer_metadata,
        "pre_split_feature_removal": False,
        "test_size": 0.2,
        "random_state": 42,
        "stratified": True,
        "sklearn_version": str(getattr(sklearn_module, "__version__", "unknown")),
        "numpy_version": str(np.__version__),
    }


def resolve_stacking_model_artifact_paths(dataset_name: str, dataset_csv_path: str, model_name: str, feature_set: str, artifact_context: dict, config: dict) -> dict:
    """
    Resolve deterministic dataset-local paths for one classifier artifact.

    :param dataset_name: Existing dataset folder identity used by model exports.
    :param dataset_csv_path: Dataset file or directory identity.
    :param model_name: Configured classifier name.
    :param feature_set: Feature-set and hyperparameter-mode identity.
    :param artifact_context: Deterministic compatibility context.
    :param config: Runtime configuration dictionary.
    :return: Model, metadata, directory, and identity paths.
    """

    stacking_output_dir = get_stacking_output_dir(dataset_csv_path, config)
    models_directory = (Path(stacking_output_dir) / "Models" / re.sub(r'[\\/*?:"<>|]', "_", str(dataset_name))).resolve()
    artifact_identity = hashlib.sha256(json.dumps(artifact_context, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
    safe_model_name = re.sub(r'[\\/*?:"<>|]', "_", str(model_name))
    safe_feature_set = re.sub(r'[\\/*?:"<>|]', "_", str(feature_set))
    artifact_base = f"{safe_model_name}__{safe_feature_set}__{artifact_identity[:16]}"
    model_path = (models_directory / f"{artifact_base}_model.joblib").resolve()
    metadata_path = (models_directory / f"{artifact_base}_meta.json").resolve()
    validate_output_path(stacking_output_dir, str(models_directory))
    validate_output_path(stacking_output_dir, str(model_path))
    validate_output_path(stacking_output_dir, str(metadata_path))
    return {
        "stacking_output_dir": str(Path(stacking_output_dir).resolve()),
        "models_directory": str(models_directory),
        "artifact_identity": artifact_identity,
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
    }


def export_model_and_scaler(model, scaler, dataset_name, model_name, feature_set=None, dataset_csv_path=None, config=None, artifact_context=None, label_encoder=None, transformer=None):
    """
    Export model, scaler and metadata for stacking evaluations.
    
    :param model: Trained model to export
    :param scaler: Fitted scaler to export
    :param dataset_name: Name of dataset
    :param model_name: Name of model
    :param feature_set: Feature set name (GA, RFE, PCA, etc.)
    :param dataset_csv_path: Path to dataset CSV file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :param artifact_context: Deterministic original-training compatibility context
    :param label_encoder: Fitted original-training label encoder
    :param transformer: Optional fitted original-training PCA transformer
    :return: Persisted classifier artifact path
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        if not dataset_csv_path:
            raise ValueError("dataset_csv_path is required to safely export models")
        if not isinstance(artifact_context, dict) or scaler is None or label_encoder is None:
            raise ValueError("Complete classifier artifact context and fitted preprocessing are required")
        artifact_paths = resolve_stacking_model_artifact_paths(dataset_name, str(dataset_csv_path), model_name, str(feature_set), artifact_context, config)
        bundle = {
            "model": model,
            "scaler": scaler,
            "label_encoder": label_encoder,
            "transformer": transformer,
            "input_feature_names": list(artifact_context["input_feature_names"]),
            "model_feature_names": list(artifact_context["model_feature_names"]),
            "artifact_identity": artifact_paths["artifact_identity"],
        }
        write_stacking_pca_artifact_atomically(bundle, artifact_paths["model_path"], "joblib")
        model_size = int(os.path.getsize(artifact_paths["model_path"]))
        scaler_state = normalize_metadata_for_json({"mean_": scaler.mean_, "scale_": scaler.scale_, "var_": scaler.var_, "n_samples_seen_": scaler.n_samples_seen_})
        metadata = dict(artifact_context)
        metadata.update({
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "artifact_identity": artifact_paths["artifact_identity"],
            "model_artifact": os.path.basename(artifact_paths["model_path"]),
            "model_size_bytes": model_size,
            "model_sha256": calculate_file_sha256(artifact_paths["model_path"]),
            "train_sample_count": int(getattr(scaler, "n_samples_seen_", 0)),
            "scaler_state_digest": hashlib.sha256(json.dumps(scaler_state, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest(),
        })
        write_stacking_pca_artifact_atomically(metadata, artifact_paths["metadata_path"], "json")
        relative_model = os.path.relpath(artifact_paths["model_path"], artifact_paths["stacking_output_dir"])
        verbose_output(f"{BackgroundColors.GREEN}Exported original-trained classifier and fitted preprocessing to {BackgroundColors.CYAN}{relative_model}{Style.RESET_ALL}", config=config)
        return artifact_paths["model_path"]
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def compute_fpr_fnr(y_test, y_pred):
    """
    Computes the False Positive Rate and False Negative Rate from predictions.

    :param y_test: True target labels
    :param y_pred: Predicted target labels
    :return: Tuple (fpr, fnr) computed for binary or multiclass predictions
    """

    try:
        y_test_arr = np.asarray(y_test)  # Normalize true labels to numpy array for deterministic metric computation
        y_pred_arr = np.asarray(y_pred)  # Normalize predicted labels to numpy array for deterministic metric computation
        unique_labels = np.unique(y_test_arr)  # Resolve unique labels from true labels to branch binary versus multiclass logic
        if len(unique_labels) == 2:  # Verify binary classification case for classic confusion-matrix decomposition
            tn, fp, fn, tp = confusion_matrix(y_test_arr, y_pred_arr).ravel()  # Extract binary confusion matrix components
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # Calculate binary False Positive Rate with zero-division guard
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # Calculate binary False Negative Rate with zero-division guard
        else:  # Handle multiclass classification with one-vs-rest aggregation
            labels_union = np.unique(np.concatenate((y_test_arr, y_pred_arr)))  # Build deterministic label universe including unseen predictions
            cm = confusion_matrix(y_test_arr, y_pred_arr, labels=labels_union)  # Build multiclass confusion matrix over full label universe
            tp = np.diag(cm).astype(float)  # Extract per-class true positives from confusion matrix diagonal
            fp = (cm.sum(axis=0) - np.diag(cm)).astype(float)  # Compute per-class false positives from predicted-column totals
            fn = (cm.sum(axis=1) - np.diag(cm)).astype(float)  # Compute per-class false negatives from true-row totals
            tn = (cm.sum() - (tp + fp + fn)).astype(float)  # Compute per-class true negatives using one-vs-rest decomposition
            fpr_per_class = np.divide(fp, fp + tn, out=np.zeros_like(fp, dtype=float), where=(fp + tn) > 0)  # Compute per-class FPR with safe division
            fnr_per_class = np.divide(fn, fn + tp, out=np.zeros_like(fn, dtype=float), where=(fn + tp) > 0)  # Compute per-class FNR with safe division
            support = cm.sum(axis=1).astype(float)  # Use per-class true-label support for weighted aggregation
            support_sum = float(support.sum())  # Compute total support to guard weighted averaging
            if support_sum > 0.0:  # Verify weighted averaging denominator is valid
                fpr = float(np.average(fpr_per_class, weights=support))  # Aggregate multiclass FPR as support-weighted one-vs-rest mean
                fnr = float(np.average(fnr_per_class, weights=support))  # Aggregate multiclass FNR as support-weighted one-vs-rest mean
            else:  # Handle degenerate empty-support case defensively
                fpr = 0.0  # Default multiclass FPR to zero when support is unavailable
                fnr = 0.0  # Default multiclass FNR to zero when support is unavailable
        return (fpr, fnr)  # Return the FPR and FNR tuple
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def load_existing_model_if_available(model_name, dataset_file, dataset_name, feature_set, artifact_context, config=None):
    """
    Load one strictly compatible original-trained classifier and its preprocessing.

    :param model_name: Configured classifier name
    :param dataset_file: Dataset file or directory identity
    :param dataset_name: Existing dataset folder identity used by model exports
    :param feature_set: Feature-set and hyperparameter-mode identity
    :param artifact_context: Expected deterministic compatibility context
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple containing the loaded artifact bundle or None and a factual rejection reason
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        artifact_paths = resolve_stacking_model_artifact_paths(dataset_name, str(dataset_file), model_name, str(feature_set), artifact_context, config)
        metadata_path = artifact_paths["metadata_path"]
        model_path = artifact_paths["model_path"]
        if not os.path.isfile(metadata_path) or not os.path.isfile(model_path):
            return None, "expected classifier artifact or metadata file is missing"
        try:
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
        except Exception as exc:
            return None, f"classifier metadata is unreadable: {exc}"
        if not isinstance(metadata, dict):
            return None, "classifier metadata root is not a JSON object"
        for field_name, expected_value in artifact_context.items():
            if metadata.get(field_name) != expected_value:
                return None, f"classifier metadata field '{field_name}' is incompatible"
        if metadata.get("artifact_identity") != artifact_paths["artifact_identity"]:
            return None, "classifier artifact identity is incompatible"
        if metadata.get("model_artifact") != os.path.basename(model_path):
            return None, "classifier metadata references a different model file"
        try:
            if int(metadata.get("model_size_bytes", -1)) != int(os.path.getsize(model_path)):
                return None, "classifier artifact size is incompatible"
            if metadata.get("model_sha256") != calculate_file_sha256(model_path):
                return None, "classifier artifact SHA-256 is incompatible"
            bundle = load(model_path)
        except Exception as exc:
            return None, f"classifier artifact is unreadable: {exc}"
        if not isinstance(bundle, dict):
            return None, "classifier artifact root is not a dictionary"
        required_components = ("model", "scaler", "label_encoder", "transformer", "input_feature_names", "model_feature_names", "artifact_identity")
        if any(component not in bundle for component in required_components):
            return None, "classifier artifact is missing required inference components"
        if bundle.get("artifact_identity") != artifact_paths["artifact_identity"]:
            return None, "classifier bundle identity is incompatible"
        if bundle.get("input_feature_names") != artifact_context.get("input_feature_names") or bundle.get("model_feature_names") != artifact_context.get("model_feature_names"):
            return None, "classifier bundle feature schema is incompatible"
        loaded_model = bundle.get("model")
        loaded_model_class = f"{loaded_model.__class__.__module__}.{loaded_model.__class__.__name__}"
        loaded_model_params = normalize_metadata_for_json(loaded_model.get_params(deep=True) if hasattr(loaded_model, "get_params") else {})
        if loaded_model_class != artifact_context.get("model_class") or loaded_model_params != artifact_context.get("model_params"):
            return None, "classifier bundle model configuration is incompatible"
        if not isinstance(bundle.get("scaler"), StandardScaler) or not isinstance(bundle.get("label_encoder"), LabelEncoder):
            return None, "classifier bundle preprocessing types are incompatible"
        loaded_scaler = bundle["scaler"]
        loaded_scaler_state = normalize_metadata_for_json({"mean_": loaded_scaler.mean_, "scale_": loaded_scaler.scale_, "var_": loaded_scaler.var_, "n_samples_seen_": loaded_scaler.n_samples_seen_})
        loaded_scaler_digest = hashlib.sha256(json.dumps(loaded_scaler_state, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
        if metadata.get("scaler_state_digest") != loaded_scaler_digest:
            return None, "classifier bundle scaler state is incompatible"
        if normalize_metadata_for_json(bundle["label_encoder"].classes_) != artifact_context.get("label_classes"):
            return None, "classifier bundle label encoding is incompatible"
        expected_transformer = artifact_context.get("transformer")
        if (expected_transformer is None) != (bundle.get("transformer") is None):
            return None, "classifier bundle transformer presence is incompatible"
        if bundle.get("transformer") is not None:
            loaded_transformer = bundle["transformer"]
            loaded_transformer_metadata = {"class": f"{loaded_transformer.__class__.__module__}.{loaded_transformer.__class__.__name__}", "params": normalize_metadata_for_json(loaded_transformer.get_params(deep=False) if hasattr(loaded_transformer, "get_params") else {})}
            if loaded_transformer_metadata != expected_transformer:
                return None, "classifier bundle transformer configuration is incompatible"
        verbose_output(f"{BackgroundColors.GREEN}Loaded compatible original-trained classifier from {BackgroundColors.CYAN}{model_path}{Style.RESET_ALL}", config=config)
        return bundle, None
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def evaluate_individual_classifier(model, model_name, X_train, y_train, X_test, y_test, dataset_file=None, scaler=None, feature_names=None, feature_set=None, config=None, phase_metadata=None, training_ram_stats=None, fit_model=True):  # Evaluate one classifier with optional watcher metadata and RAM statistics
    """
    Trains an individual classifier and evaluates its performance on the test set.

    :param model: The classifier model object to train.
    :param model_name: Name of the classifier (for logging).
    :param X_train: Training features (scaled numpy array).
    :param y_train: Training target labels (encoded Series/array).
    :param X_test: Testing features (scaled numpy array).
    :param y_test: Testing target labels (encoded Series/array).
    :param dataset_file: Path to dataset file
    :param scaler: Scaler object
    :param feature_names: List of feature names
    :param feature_set: Feature set name
    :param config: Configuration dictionary (uses global CONFIG if None)
    :param phase_metadata: Optional compact watcher metadata for this classifier
    :param training_ram_stats: Mutable holder receiving classifier training RAM statistics
    :param fit_model: Whether to fit on original training data before prediction
    :return: Metrics tuple (acc, prec, rec, f1, fpr, fnr, elapsed_time)
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        phase_metadata = dict(phase_metadata or {})  # Normalize watcher metadata for safe reuse
        training_ram_monitor = None  # Initialize classifier RAM monitor state for exception-safe cleanup.
        
        try:  # Attempt to obtain only the model's own parameters for logging
            params_raw = model.get_params(deep=False) if hasattr(model, "get_params") else {}  # Get top-level estimator params only
        except Exception:  # On any error retrieving params
            params_raw = {}  # Fallback to empty dict when parameters cannot be read

        try:  # Build a compact, truncated string representation of parameters for safe logging
            filtered_params = {}  # Prepare a filtered dict that replaces nested estimators with class names
            for k, v in (params_raw.items() if isinstance(params_raw, dict) else []):  # Iterate over top-level params
                if hasattr(v, "fit") or hasattr(v, "predict"):  # Detect estimator-like objects and replace with class name
                    try:
                        filtered_params[k] = v.__class__.__name__  # Use class name for estimator objects for readability
                    except Exception:
                        filtered_params[k] = str(type(v))  # Fallback to type string when class name retrieval fails
                else:
                    filtered_params[k] = v  # Preserve simple values as-is

            # filtered_params = list(filtered_params.items())[:6]  # Limit to first N items to avoid excessive output
            params_snapshot = ", ".join(f"{k}={v}" for k, v in filtered_params.items())  # Join key=value pairs for display
            # if len(params_snapshot) > 240:  # Truncate overly long snapshots for readability
            #     params_snapshot = params_snapshot[:237] + "..."  # Truncate and append ellipsis
        except Exception:  # On failure during snapshot formatting
            params_snapshot = ""  # Use empty string when formatting fails

        verbose_output(  # Output the verbose message including a compact classifier parameter snapshot
            f"{BackgroundColors.GREEN}{'Training' if fit_model else 'Evaluating loaded'} {BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN} Params:{BackgroundColors.CYAN}{params_snapshot}{Style.RESET_ALL}",  # Message with compact params snapshot
            config=config,  # Configuration used for verbose output routing
        )

        start_time = time.time()  # Record the start time

        if fit_model:  # Fit only during the original-data training lifecycle.
            sys.stdout.flush()  # Flush stdout before model training to ensure logs are visible under nohup
            training_ram_monitor = start_training_ram_monitor(TRAINING_RAM_SAMPLE_INTERVAL_SECONDS)  # Start RAM monitoring immediately before classifier fit.
            try:  # Ensure RAM monitoring stops even when classifier fit fails.
                model.fit(X_train, y_train)  # Fit the model on original training data only.
            finally:  # Stop RAM monitoring immediately after classifier fit exits.
                classifier_ram_stats = stop_training_ram_monitor(training_ram_monitor)  # Summarize RAM usage across this classifier fit.
                store_training_ram_stats(training_ram_stats, classifier_ram_stats)  # Associate RAM statistics with this classifier only.
                training_ram_monitor = None  # Clear monitor state after stop processing.
            if classifier_ram_stats.get("thread_alive", False):  # Verify whether sampler thread cleanup completed.
                verbose_output(f"{BackgroundColors.YELLOW}[WARNING] RAM monitor thread remained alive after training {model_name}.{Style.RESET_ALL}", config=config)  # Log monitor cleanup anomaly without changing classifier results.
            params_digest = get_classifier_params_digest(model)  # Build fitted classifier parameter digest
            fit_metadata = dict(phase_metadata)  # Copy caller watcher metadata
            fit_metadata.update({"dataset_identity": os.path.basename(str(dataset_file)) if dataset_file is not None else fit_metadata.get("dataset_identity"), "classifier_name": model_name, "classifier_params_digest": params_digest, "classifier_params_reference": f"sha256:{params_digest.get('digest')}", "train_sample_count": len(y_train), "test_sample_count": len(y_test), "feature_count": X_train.shape[1] if hasattr(X_train, "shape") and len(X_train.shape) > 1 else fit_metadata.get("feature_count"), "n_jobs": get_classifier_n_jobs(model), "event_outcome": "fit_completed"})  # Build fit completion watcher metadata
            write_memory_phase_event("after_classifier_fit", config=config, **fit_metadata)  # Publish classifier fit completion
        else:
            store_training_ram_stats(training_ram_stats, stop_training_ram_monitor(None))  # Preserve RAM-stat shape without starting a training monitor.

        y_pred = model.predict(X_test)  # Predict the labels for the test set

        elapsed_time = time.time() - start_time  # Calculate the total time elapsed

        acc = accuracy_score(y_test, y_pred)  # Calculate Accuracy
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=cast(Any, 0))  # Calculate Precision
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=cast(Any, 0))  # Calculate Recall
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=cast(Any, 0))  # Calculate F1-Score

        fpr, fnr = compute_fpr_fnr(y_test, y_pred)  # Compute False Positive and False Negative rates

        human_time = calculate_execution_time(elapsed_time)  # Convert elapsed duration to human-readable string using helper
        total_seconds = int(round(elapsed_time))  # Reuse elapsed_time as total seconds for reporting
        train_seconds = total_seconds if fit_model else 0  # Report zero training time for persisted-model evaluation.
        exec_seconds = total_seconds  # Reuse total seconds as execution time when only one timer exists
        evaluation_mode = None  # Initialize evaluation mode string from config or global CONFIG
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG as fallback
        mode_raw = config.get("execution", {}).get("execution_mode")  # Obtain execution mode from config if present
        if mode_raw:  # If an execution mode string exists in config
            evaluation_mode = mode_raw.replace("_", " ").title().replace(" ", "")  # Normalize to CamelCase style
        if evaluation_mode is None:  # If still not resolved
            evaluation_mode = "SeparateFiles"  # Default to SeparateFiles when unknown
        msg = f"{BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN}: Mode {BackgroundColors.CYAN}{evaluation_mode}{BackgroundColors.GREEN} | F1-Score {BackgroundColors.CYAN}{f1}{BackgroundColors.GREEN} | Accuracy: {BackgroundColors.CYAN}{acc}{BackgroundColors.GREEN} | Precision: {BackgroundColors.CYAN}{prec}{BackgroundColors.GREEN} | Recall: {BackgroundColors.CYAN}{rec}{BackgroundColors.GREEN} | FPR: {BackgroundColors.CYAN}{fpr}{BackgroundColors.GREEN} | FNR: {BackgroundColors.CYAN}{fnr}{BackgroundColors.GREEN} | Training Time: {BackgroundColors.CYAN}{int(train_seconds)}s{BackgroundColors.GREEN} | Execution Time: {BackgroundColors.CYAN}{int(exec_seconds)}s{BackgroundColors.GREEN} | Total Time: {BackgroundColors.CYAN}{human_time}{BackgroundColors.GREEN} ({BackgroundColors.CYAN}{int(total_seconds)}s{BackgroundColors.GREEN}){Style.RESET_ALL}"  # Build final formatted classifier summary using raw floats for metrics and integer times
        print(msg)  # Print the summary message to console

        return (acc, prec, rec, f1, fpr, fnr, int(round(elapsed_time)))  # Return the metrics tuple
    except MemoryError as e:  # Handle classifier memory errors with a diagnostic phase
        error_metadata = dict(phase_metadata or {})  # Copy watcher metadata for memory error
        error_metadata.update({"dataset_identity": os.path.basename(str(dataset_file)) if dataset_file is not None else error_metadata.get("dataset_identity"), "classifier_name": model_name, "train_sample_count": len(y_train) if y_train is not None else error_metadata.get("train_sample_count"), "test_sample_count": len(y_test) if y_test is not None else error_metadata.get("test_sample_count"), "feature_count": X_train.shape[1] if hasattr(X_train, "shape") and len(X_train.shape) > 1 else error_metadata.get("feature_count"), "n_jobs": get_classifier_n_jobs(model), "event_outcome": str(e)})  # Build memory error watcher metadata
        write_memory_phase_event("memory_error", config=config, **error_metadata)  # Publish classifier memory error
        try:
            model = None  # Release the partially fitted estimator before heavy error reporting
            gc.collect()  # Reclaim estimator-owned memory best-effort
        except Exception:
            pass  # Do not mask the original MemoryError
        print(str(e))  # Print memory error to terminal logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send memory error via Telegram
        raise  # Preserve original MemoryError behavior
    except Exception as e:
        error_metadata = dict(phase_metadata or {})  # Copy watcher metadata for model error
        error_metadata.update({"dataset_identity": os.path.basename(str(dataset_file)) if dataset_file is not None else error_metadata.get("dataset_identity"), "classifier_name": model_name, "train_sample_count": len(y_train) if y_train is not None else error_metadata.get("train_sample_count"), "test_sample_count": len(y_test) if y_test is not None else error_metadata.get("test_sample_count"), "feature_count": X_train.shape[1] if hasattr(X_train, "shape") and len(X_train.shape) > 1 else error_metadata.get("feature_count"), "n_jobs": get_classifier_n_jobs(model), "event_outcome": str(e)})  # Build model error watcher metadata
        write_memory_phase_event("model_error", config=config, **error_metadata)  # Publish classifier model error
        try:
            model = None  # Release the partially fitted estimator before heavy error reporting
            gc.collect()  # Reclaim estimator-owned memory best-effort
        except Exception:
            pass  # Do not mask the original exception
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def evaluate_stacking_classifier(model, X_train, y_train, X_test, y_test, config=None, training_ram_stats=None, fit_model=True):
    """
    Trains the StackingClassifier model and evaluates its performance on the test set.

    :param model: The fitted StackingClassifier model object.
    :param X_train: Training features (pandas DataFrame or numpy array with feature names).
    :param y_train: Training target labels (encoded Series/array).
    :param X_test: Testing features (pandas DataFrame or numpy array with feature names).
    :param y_test: Testing target labels (encoded Series/array).
    :param config: Configuration dictionary (uses global CONFIG if None).
    :param training_ram_stats: Mutable holder receiving classifier training RAM statistics.
    :param fit_model: Whether to fit on original training data before prediction.
    :return: Metrics tuple (acc, prec, rec, f1, fpr, fnr, elapsed_time)
    """
    
    try:
        if config is None:  # Use global configuration when no explicit configuration is supplied.
            config = CONFIG  # Assign global CONFIG fallback.
        training_ram_monitor = None  # Initialize stacking RAM monitor state for exception-safe cleanup.
        verbose_output(
            f"{BackgroundColors.GREEN}Starting {'training and evaluation' if fit_model else 'persisted-model evaluation'} of Stacking Classifier...{Style.RESET_ALL}"
        )  # Output the verbose message

        start_time = time.time()  # Record the start time for timing training and prediction

        if fit_model:  # Fit only during the original-data training lifecycle.
            sys.stdout.flush()  # Flush stdout before stacking training to ensure logs are visible under nohup
            training_ram_monitor = start_training_ram_monitor(TRAINING_RAM_SAMPLE_INTERVAL_SECONDS)  # Start RAM monitoring immediately before stacking fit.
            try:  # Ensure RAM monitoring stops even when stacking fit fails.
                model.fit(X_train, y_train)  # Fit the stacking model on original training data only.
            finally:  # Stop RAM monitoring immediately after stacking fit exits.
                stacking_ram_summary = stop_training_ram_monitor(training_ram_monitor)  # Summarize RAM usage across this stacking fit.
                store_training_ram_stats(training_ram_stats, stacking_ram_summary)  # Associate RAM statistics with this stacking classifier only.
                training_ram_monitor = None  # Clear monitor state after stop processing.
            if stacking_ram_summary.get("thread_alive", False):  # Verify whether sampler thread cleanup completed.
                verbose_output(f"{BackgroundColors.YELLOW}[WARNING] RAM monitor thread remained alive after training StackingClassifier.{Style.RESET_ALL}", config=config)  # Log monitor cleanup anomaly without changing stacking results.
        else:
            store_training_ram_stats(training_ram_stats, stop_training_ram_monitor(None))  # Preserve RAM-stat shape without starting a training monitor.

        y_pred = model.predict(X_test)  # Predict the labels for the test set

        elapsed_time = time.time() - start_time  # Calculate the total time elapsed

        acc = accuracy_score(y_test, y_pred)  # Calculate Accuracy
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=cast(Any, 0))  # Calculate Precision (weighted)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=cast(Any, 0))  # Calculate Recall (weighted)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=cast(Any, 0))  # Calculate F1-Score (weighted)

        fpr, fnr = compute_fpr_fnr(y_test, y_pred)  # Compute False Positive and False Negative rates for binary or multiclass predictions

        human_time = calculate_execution_time(elapsed_time)  # Convert elapsed duration to human-readable string using helper
        total_seconds = int(round(elapsed_time))  # Reuse elapsed_time as total seconds for reporting
        train_seconds = total_seconds if fit_model else 0  # Report zero training time for persisted-model evaluation.
        exec_seconds = total_seconds  # Reuse total seconds as execution time when only one timer exists
        evaluation_mode = None  # Initialize evaluation mode string from runtime config
        mode_raw = config.get("execution", {}).get("execution_mode")  # Obtain execution mode from runtime config if present
        if mode_raw:  # If an execution mode string exists in CONFIG
            evaluation_mode = mode_raw.replace("_", " ").title().replace(" ", "")  # Normalize to CamelCase style
        if evaluation_mode is None:  # If still not resolved
            evaluation_mode = "SeparateFiles"  # Default to SeparateFiles when unknown
        msg = f"{BackgroundColors.CYAN}StackingClassifier{BackgroundColors.GREEN}: Mode {BackgroundColors.CYAN}{evaluation_mode}{BackgroundColors.GREEN} | F1-Score {BackgroundColors.CYAN}{f1}{BackgroundColors.GREEN} | Accuracy: {BackgroundColors.CYAN}{acc}{BackgroundColors.GREEN} | Precision: {BackgroundColors.CYAN}{prec}{BackgroundColors.GREEN} | Recall: {BackgroundColors.CYAN}{rec}{BackgroundColors.GREEN} | FPR: {BackgroundColors.CYAN}{fpr}{BackgroundColors.GREEN} | FNR: {BackgroundColors.CYAN}{fnr}{BackgroundColors.GREEN} | Training Time: {BackgroundColors.CYAN}{int(train_seconds)}s{BackgroundColors.GREEN} | Execution Time: {BackgroundColors.CYAN}{int(exec_seconds)}s{BackgroundColors.GREEN} | Total Time: {BackgroundColors.CYAN}{human_time}{BackgroundColors.GREEN} ({BackgroundColors.CYAN}{int(total_seconds)}s{BackgroundColors.GREEN}){Style.RESET_ALL}"  # Build final formatted stacking summary with colors using raw floats and integer times
        print(msg)  # Print the summary message to console

        return (acc, prec, rec, f1, fpr, fnr, int(round(elapsed_time)), y_pred)  # Return the metrics tuple and predictions
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def sample_shap_test_data(X_test, y_test, max_samples, random_state):
    """
    Samples test data for SHAP computation when the test set exceeds the maximum sample size.

    :param X_test: Test features array to sample from
    :param y_test: Test labels array or Series to sample from
    :param max_samples: Maximum number of samples to use for SHAP computation
    :param random_state: Random seed for reproducible sampling
    :return: Tuple (X_test_sampled, y_test_sampled) with sampled or full test data
    """

    try:
        if len(X_test) > max_samples:  # If test set exceeds the sample limit
            rng = np.random.default_rng(random_state)  # Create explicit RNG to avoid using global RNG
            sample_indices = rng.choice(len(X_test), size=max_samples, replace=False)  # Draw reproducible sample indices from RNG
            if hasattr(X_test, 'iloc') and hasattr(X_test, 'iloc'):  # Verify pandas DataFrame/Series slicing is applicable
                X_test_sampled = X_test.iloc[sample_indices]  # Slice test features via iloc for DataFrame/Series
            else:  # Fallback to numpy-style indexing for arrays
                X_test_sampled = X_test[sample_indices]  # Slice test features via numpy indexing
            if hasattr(y_test, 'iloc'):  # If y_test is a pandas Series
                y_test_sampled = y_test.iloc[sample_indices]  # Slice labels via iloc for pandas Series
            else:  # If y_test is numpy array-like
                y_test_sampled = y_test[sample_indices]  # Slice labels via numpy indexing for arrays
        else:  # Test set is within the sample limit
            X_test_sampled = X_test  # Use full test features without sampling
            y_test_sampled = y_test  # Use full test labels without sampling
        return (X_test_sampled, y_test_sampled)  # Return the sampled or full test data tuple
    except Exception as e:  # Handle unexpected errors
        print(str(e))  # Print the exception string for diagnostics
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception details via Telegram if configured
        raise  # Re-raise the exception to preserve original behavior


def aggregate_mean_shap_importance(shap_values_summary, feature_names):
    """
    Computes mean absolute SHAP values and maps them to feature names.

    :param shap_values_summary: SHAP values array (2D: samples x features)
    :param feature_names: List of feature names corresponding to the SHAP value columns
    :return: Dictionary mapping each feature name to its mean absolute SHAP importance
    """

    try:
        shap_array = np.array(shap_values_summary)  # Convert SHAP values to numpy array for consistent operations
        mean_shap_values = np.mean(np.abs(shap_array), axis=0)  # Compute mean absolute SHAP value per feature across samples
        mean_shap_list = mean_shap_values.tolist() if hasattr(mean_shap_values, 'tolist') else list(mean_shap_values)  # Convert numpy array to plain Python list
        if isinstance(feature_names, (list, tuple)) and len(feature_names) == len(mean_shap_list):  # Verify feature name length matches SHAP values
            shap_importance = dict(zip(feature_names, mean_shap_list))  # Map feature names exactly to their mean absolute importance values
        else:  # If mismatch detected
            shap_importance = {f"f{idx}": val for idx, val in enumerate(mean_shap_list)}  # Fall back to index-based feature keys preserving values
        return shap_importance  # Return importance dictionary for downstream use
    except Exception as e:  # Handle unexpected errors
        print(str(e))  # Print the exception string for diagnostics
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception details via Telegram if configured
        raise  # Re-raise the exception to preserve original behavior


def save_shap_summary_and_bar_plots(shap_values_summary, X_test_sampled, feature_names, output_dir, dataset_name, model_name, max_display):
    """
    Saves SHAP summary and bar plots to the output directory.

    :param shap_values_summary: SHAP values array used for plot generation
    :param X_test_sampled: Sampled test features for plot background data
    :param feature_names: List of feature names for axis labels
    :param output_dir: Directory path where plots will be saved
    :param dataset_name: Dataset name used in output filenames
    :param model_name: Model name used in output filenames
    :param max_display: Maximum number of features to display in each plot
    :return: None
    """

    try:
        # Validate that X_test_sampled is non-empty and shaped correctly before plotting
        if X_test_sampled is None or (hasattr(X_test_sampled, '__len__') and len(X_test_sampled) == 0):  # Verify sampled data is present
            return  # Exit early if there is no data to plot

        # Ensure feature_names matches X_test_sampled column count exactly when possible
        if hasattr(X_test_sampled, 'shape') and hasattr(X_test_sampled, 'ndim'):
            ncols = X_test_sampled.shape[1] if X_test_sampled.ndim == 2 else (len(feature_names) if hasattr(feature_names, '__len__') else None)  # Determine number of feature columns
        else:
            ncols = len(feature_names) if hasattr(feature_names, '__len__') else None  # Fallback to provided feature_names length

        if ncols is not None and hasattr(feature_names, '__len__') and ncols != len(feature_names):  # If mismatch detected
            feature_names_to_use = list(feature_names)[:ncols]  # Truncate or expand feature names defensively to match column count
        else:
            feature_names_to_use = feature_names  # Use feature names as provided when consistent

        try:  # Attempt to create SHAP summary plot
            plt.figure()  # Create new figure for summary plot
            shap.summary_plot(shap_values_summary, X_test_sampled, feature_names=feature_names_to_use, max_display=max_display, show=False)  # Create summary plot with explicit feature names
            summary_plot_path = os.path.join(output_dir, f"{dataset_name}_{model_name}_shap_summary.png")  # Build summary plot file path
            plt.tight_layout()  # Adjust layout for tight fit
            ensure_playwright_chromium()  # Ensure Playwright Chromium is installed before saving PNG
            ensure_figure_min_4k_and_save(fig=plt.gcf(), path=summary_plot_path, dpi=300, bbox_inches='tight')  # Save with minimum 4K resolution
            plt.close()  # Close summary figure
        except Exception as e:  # If summary plot generation fails
            plt.close()  # Close figure to avoid resource leak
            raise e  # Re-raise so outer handler can notify via Telegram

        try:  # Attempt to create SHAP bar plot
            plt.figure()  # Create new figure for bar plot
            shap.summary_plot(shap_values_summary, X_test_sampled, feature_names=feature_names_to_use, max_display=max_display, plot_type="bar", show=False)  # Create bar plot with explicit feature names
            bar_plot_path = os.path.join(output_dir, f"{dataset_name}_{model_name}_shap_bar.png")  # Build bar plot file path
            plt.tight_layout()  # Adjust layout for tight fit
            ensure_playwright_chromium()  # Ensure Playwright Chromium is installed before saving PNG
            ensure_figure_min_4k_and_save(fig=plt.gcf(), path=bar_plot_path, dpi=300, bbox_inches='tight')  # Save with minimum 4K resolution
            plt.close()  # Close bar figure
        except Exception as e:  # If bar plot generation fails
            plt.close()  # Close figure to avoid resource leak
            raise e  # Re-raise so outer handler can notify via Telegram
    except Exception as e:  # Handle unexpected errors
        print(str(e))  # Print the exception string for diagnostics
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception details via Telegram if configured
        raise  # Re-raise the exception to preserve original behavior


def supports_predict_proba(model):  # Verify model supports predict_proba at module level
    """
    Verify if model supports predict_proba.

    :param model: Model instance.
    :return: Boolean indicating support.
    """

    return hasattr(model, "predict_proba")  # Verify presence of predict_proba method


def get_shap_prediction_function(model):  # Resolve prediction function for SHAP/LIME at module level
    """
    Resolve prediction function for SHAP based on model capabilities.

    :param model: Model instance.
    :return: Callable prediction function.
    """

    if supports_predict_proba(model):  # Verify if model supports predict_proba
        return model.predict_proba  # Use probability predictions when available

    return model.predict  # Fallback to class prediction when probabilities unavailable


def build_kernel_explainer(model, X_test_sampled, random_state):
    """
    Build a SHAP KernelExplainer with a bounded, reproducible background sample.

    :param model: Trained model object.
    :param X_test_sampled: Sampled test features used to derive background data.
    :param random_state: Random seed used for deterministic background sampling.
    :return: Instantiated shap.KernelExplainer.
    """

    rng = np.random.default_rng(random_state)  # Create explicit RNG for deterministic background sampling
    bkg_size = min(50, len(X_test_sampled)) if hasattr(X_test_sampled, "__len__") else 50  # Determine background sample size defensively
    indices = rng.choice(len(X_test_sampled), size=bkg_size, replace=False)  # Draw reproducible background indices
    if hasattr(X_test_sampled, "iloc"):  # If sampled data is a pandas object
        background = X_test_sampled.iloc[indices]  # Slice background via iloc
    else:  # Otherwise assume numpy-like array
        background = X_test_sampled[indices]  # Slice background via numpy indexing
    prediction_fn = get_shap_prediction_function(model)  # Resolve SHAP-compatible prediction callable
    return shap.KernelExplainer(prediction_fn, background)  # Build and return KernelExplainer


def select_shap_explainer(model, X_test_sampled, random_state):
    """
    Selects and instantiates the appropriate SHAP explainer based on model type.

    :param model: Trained model object for which to build the explainer
    :param X_test_sampled: Sampled test features used for KernelExplainer background data
    :param random_state: Random seed used for KernelExplainer background sampling
    :return: Instantiated SHAP explainer object
    """

    try:
        model_type = model.__class__.__name__  # Get model class name for branch selection
        n_classes = getattr(model, "n_classes_", None)  # Resolve fitted class count when available
        if model_type in ["RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier", "LightGBMClassifier", "ExtraTreesClassifier"]:  # Tree-based models
            if model_type == "GradientBoostingClassifier" and n_classes is not None and int(n_classes) > 2:  # SHAP TreeExplainer does not support multiclass GradientBoostingClassifier
                return build_kernel_explainer(model, X_test_sampled, random_state)  # Fallback to model-agnostic KernelExplainer for multiclass GB
            try:  # Try fast tree explainer first for supported tree models
                return shap.TreeExplainer(model)  # Use TreeExplainer for supported tree-based models
            except Exception as e:  # If SHAP tree path fails for a known unsupported case
                err = str(e).lower()  # Normalize exception string for safe matching
                if model_type == "GradientBoostingClassifier" and "only supported for binary classification" in err:  # Explicit SHAP multiclass GB limitation
                    return build_kernel_explainer(model, X_test_sampled, random_state)  # Fallback to KernelExplainer when SHAP rejects multiclass GB
                raise  # Re-raise unknown errors to preserve failure visibility
        elif model_type in ["LogisticRegression", "LinearSVC", "SGDClassifier"]:  # Linear models
            return shap.LinearExplainer(model, X_test_sampled)  # Use LinearExplainer for linear models
        else:  # Other models that require a fallback explainer
            return build_kernel_explainer(model, X_test_sampled, random_state)  # Use KernelExplainer with bounded deterministic background sampling
    except Exception as e:  # Handle unexpected errors
        print(str(e))  # Print the exception string for diagnostics
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception details via Telegram if configured
        raise  # Re-raise the exception to preserve original behavior


def build_shap_progress_description(model_name, dataset_name, explainer_name):
    """
    Build a concise contextual SHAP progress-bar description.

    :param model_name: Name of the model being explained.
    :param dataset_name: Name of the dataset being explained.
    :param explainer_name: Name of the SHAP explainer in use.
    :return: Context-rich progress-bar description string.
    """

    try:
        model_label = str(model_name) if model_name else "UnknownModel"  # Normalize model label for progress output.
        dataset_label = str(dataset_name) if dataset_name else "UnknownDataset"  # Normalize dataset label for progress output.
        explainer_label = str(explainer_name) if explainer_name else "SHAP"  # Normalize explainer label for progress output.
        return f"SHAP {explainer_label} | {model_label} | {dataset_label}"  # Return concise contextual description for SHAP progress output.
    except Exception as e:
        print(str(e))  # Print the exception string for diagnostics.
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception details via Telegram if configured.
        raise  # Re-raise the exception to preserve original behavior.


def resolve_shap_progress_target(shap_callable) -> Tuple[Optional[str], Optional[Any], Optional[str], Optional[Callable[..., Any]]]:
    """
    Resolve the runtime tqdm symbol used by a SHAP callable when exposed.

    :param shap_callable: Bound SHAP callable used to compute SHAP values.
    :return: Tuple describing the patch target and original tqdm callable, or empty values when unavailable.
    """

    try:
        method_globals = getattr(shap_callable, "__globals__", {})  # Access callable globals for runtime tqdm resolution.
        direct_tqdm = method_globals.get("tqdm", None)  # Resolve direct tqdm symbol from callable globals when present.
        if callable(direct_tqdm):  # Verify direct tqdm symbol is callable before using it.
            return ("globals_dict", method_globals, "tqdm", direct_tqdm)  # Return direct globals-based patch target and original tqdm callable.
        for value in method_globals.values():  # Iterate global values to locate module-style tqdm exposure when used by SHAP.
            module_name = getattr(value, "__name__", "") if value is not None else ""  # Resolve module-like name defensively for tqdm filtering.
            tqdm_attr = getattr(value, "tqdm", None) if value is not None else None  # Resolve nested tqdm attribute when a module wrapper is used.
            if callable(tqdm_attr) and "tqdm" in str(module_name).lower():  # Verify nested tqdm attribute belongs to a tqdm-related module.
                return ("module_attr", value, "tqdm", tqdm_attr)  # Return module-attribute patch target and original tqdm callable.
        return (None, None, None, None)  # Return empty patch metadata when SHAP does not expose a runtime tqdm hook.
    except Exception as e:
        print(str(e))  # Print the exception string for diagnostics.
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception details via Telegram if configured.
        raise  # Re-raise the exception to preserve original behavior.


def create_shap_progress_wrapper(tqdm_callable, progress_desc, progress_phase):
    """
    Create a tqdm wrapper that injects SHAP-specific context without changing iteration semantics.

    :param tqdm_callable: Original tqdm callable resolved from SHAP runtime globals.
    :param progress_desc: Description string to inject into the progress bar.
    :param progress_phase: Postfix string describing the current SHAP phase.
    :return: Wrapped tqdm callable.
    """

    def wrapped_tqdm(*args, **kwargs):
        kwargs.setdefault("desc", progress_desc)  # Inject contextual description only when SHAP did not already provide one.
        kwargs.setdefault("file", sys.stdout)  # Route progress output through the configured stdout logger.
        progress_bar = tqdm_callable(*args, **kwargs)  # Delegate progress-bar construction to the original tqdm callable.
        if hasattr(progress_bar, "set_postfix_str") and progress_phase:  # Verify postfix support before appending SHAP phase metadata.
            progress_bar.set_postfix_str(progress_phase, refresh=False)  # Append concise SHAP phase metadata without forcing a redraw.
        return progress_bar  # Return the original tqdm instance with injected contextual metadata.

    return wrapped_tqdm  # Return wrapped tqdm callable for temporary SHAP runtime patching.


def compute_shap_values_with_context(explainer, X_test_for_shap, progress_desc, progress_phase):
    """
    Compute SHAP values while temporarily injecting contextual progress metadata when SHAP exposes tqdm.

    :param explainer: Instantiated SHAP explainer object.
    :param X_test_for_shap: Test feature matrix passed to SHAP.
    :param progress_desc: Description string for the SHAP progress bar.
    :param progress_phase: Postfix string describing the current SHAP phase.
    :return: SHAP values returned by the explainer.
    """

    try:
        patch_kind, patch_owner, patch_attr, original_tqdm = resolve_shap_progress_target(explainer.shap_values)  # Resolve SHAP runtime tqdm target when exposed by the installed version.
        if original_tqdm is None or patch_owner is None or patch_attr is None:  # Verify whether SHAP exposes a complete patchable tqdm hook before attempting runtime injection.
            return explainer.shap_values(X_test_for_shap)  # Compute SHAP values directly when no patchable tqdm hook exists.
        wrapped_tqdm = create_shap_progress_wrapper(original_tqdm, progress_desc, progress_phase)  # Build contextual tqdm wrapper around the original SHAP progress constructor.
        try:
            if patch_kind == "globals_dict":  # Verify whether SHAP uses a direct globals-based tqdm symbol.
                patch_owner[patch_attr] = wrapped_tqdm  # Replace SHAP globals-based tqdm symbol temporarily.
            else:
                setattr(patch_owner, patch_attr, wrapped_tqdm)  # Replace SHAP module-attribute tqdm symbol temporarily.
            return explainer.shap_values(X_test_for_shap)  # Compute SHAP values while the contextual tqdm wrapper is active.
        finally:
            if patch_kind == "globals_dict":  # Verify whether the patched tqdm symbol lives in SHAP callable globals.
                patch_owner[patch_attr] = original_tqdm  # Restore the original SHAP globals-based tqdm symbol after computation.
            else:
                setattr(patch_owner, patch_attr, original_tqdm)  # Restore the original SHAP module-attribute tqdm symbol after computation.
    except Exception as e:
        print(str(e))  # Print the exception string for diagnostics.
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception details via Telegram if configured.
        raise  # Re-raise the exception to preserve original behavior.


def generate_shap_explanations(model, X_test, y_test, feature_names, output_dir, model_name, dataset_name, execution_mode, config=None):
    """
    Generate SHAP explanations for a trained model.

    :param model: Trained model object
    :param X_test: Test features (numpy array or DataFrame)
    :param y_test: Test labels
    :param feature_names: List of feature names
    :param output_dir: Directory to save SHAP outputs
    :param model_name: Name of the model for labeling
    :param dataset_name: Name of the dataset
    :param execution_mode: Execution mode string ('separate_files' or 'combined_files')
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary with SHAP values and summary metrics or None if failed
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        try:  # Attempt to generate SHAP explanations
            verbose_output(
                f"{BackgroundColors.GREEN}Generating SHAP explanations for {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}",
                config=config
            )  # Log SHAP generation start

            shap_config = config.get("explainability", {})  # Get explainability config
            max_samples = shap_config.get("shap_max_samples", 100)  # Max samples for SHAP computation
            max_display = shap_config.get("max_display_features", 20)  # Max features to display
            random_state = shap_config.get("random_state", 42)  # Random state for sampling

            os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

            X_test_sampled, y_test_sampled = sample_shap_test_data(X_test, y_test, max_samples, random_state)  # Sample test data for SHAP computation

            if hasattr(X_test_sampled, 'columns'):  # If sampled data is a pandas DataFrame
                sampled_feature_names = list(X_test_sampled.columns)  # Extract feature names from DataFrame columns to ensure exact ordering
                X_test_for_shap = X_test_sampled.values  # Convert DataFrame to numpy array for SHAP where required
            else:  # If sampled data is numpy array-like
                X_test_for_shap = X_test_sampled  # Use numpy array directly for SHAP computations
                sampled_feature_names = feature_names if hasattr(feature_names, '__len__') else [f"f{i}" for i in range(X_test_for_shap.shape[1])]  # Build feature name list defensively

            if hasattr(X_test_for_shap, 'size') and X_test_for_shap.size == 0:  # Verify sampled data is not empty
                verbose_output(f"{BackgroundColors.YELLOW}No data available for SHAP explanations for {model_name}.{Style.RESET_ALL}", config=config)  # Log empty data situation
                return None  # Return None when there is nothing to explain

            if np.isnan(np.sum(X_test_for_shap)) or np.isinf(np.sum(X_test_for_shap)):  # Detect NaN or infinite entries in numeric array
                verbose_output(f"{BackgroundColors.YELLOW}NaN/Inf detected in SHAP input data for {model_name}; skipping SHAP.{Style.RESET_ALL}", config=config)  # Log problematic data state
                return None  # Return None to avoid SHAP failures when data invalid

            explainer = select_shap_explainer(model, X_test_sampled, random_state)  # Select the appropriate SHAP explainer based on model type
            explainer_name = explainer.__class__.__name__  # Resolve explainer class name for logging, Telegram, and progress context.
            progress_desc = build_shap_progress_description(model_name, dataset_name, explainer_name)  # Build contextual SHAP progress-bar description from execution metadata.
            progress_phase = f"samples={len(X_test_for_shap)} | mode={execution_mode}"  # Build concise SHAP progress postfix from available execution metadata.

            verbose_output(
                f"{BackgroundColors.GREEN}Using {BackgroundColors.CYAN}{explainer_name}{BackgroundColors.GREEN} for SHAP on {BackgroundColors.CYAN}{len(X_test_for_shap)}{BackgroundColors.GREEN} sampled rows from {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}",
                config=config
            )  # Log the resolved SHAP explainer and sampled row count before computation.
            send_telegram_message(TELEGRAM_BOT, f"Starting SHAP explanations for {dataset_name} - {model_name} using {explainer_name} on {len(X_test_for_shap)} sample(s)")  # Notify Telegram that SHAP computation is starting with available execution context.

            shap_values = compute_shap_values_with_context(explainer, X_test_for_shap, progress_desc, progress_phase)  # Compute SHAP values using the prepared numpy input while preserving SHAP math and improving progress context.

            if isinstance(shap_values, list):  # Combined files evaluation case for multi-class classifiers
                shap_values_summary = shap_values[0] if len(shap_values) > 0 else shap_values  # Use first class for summary to preserve existing behavior
            else:  # Separate files evaluation or regression case
                shap_values_summary = shap_values  # Use SHAP values directly for single-output explainers

            save_shap_summary_and_bar_plots(shap_values_summary, X_test_for_shap, sampled_feature_names, output_dir, dataset_name, model_name, max_display)  # Save both SHAP summary and bar plots to disk

            shap_importance = aggregate_mean_shap_importance(shap_values_summary, sampled_feature_names)  # Aggregate mean absolute SHAP values into a feature importance dictionary

            verbose_output(
                f"{BackgroundColors.GREEN}SHAP explanations saved to {BackgroundColors.CYAN}{output_dir}{Style.RESET_ALL}",
                config=config
            )  # Log SHAP completion
            send_telegram_message(TELEGRAM_BOT, f"Finished SHAP explanations for {dataset_name} - {model_name} using {explainer_name}; outputs saved to {output_dir}")  # Notify Telegram that SHAP computation finished successfully with available execution context.

            return {"shap_importance": shap_importance, "shap_values": shap_values}  # Return SHAP results

        except ImportError:  # If SHAP not installed
            print(f"{BackgroundColors.YELLOW}SHAP library not installed. Skipping SHAP explanations. Install with: pip install shap{Style.RESET_ALL}")  # Warn user about missing dependency
            return None  # Return None
        except Exception as e:  # If any other error
            verbose_output(
                f"{BackgroundColors.YELLOW}Failed to generate SHAP explanations for {model_name}: {e}{Style.RESET_ALL}",
                config=config
            )  # Log error
            return None  # Return None
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def generate_lime_explanations(model, X_test, y_test, feature_names, output_dir, model_name, dataset_name, execution_mode, config=None):
    """
    Generate LIME explanations for a trained model.

    :param model: Trained model object
    :param X_test: Test features (numpy array or DataFrame)
    :param y_test: Test labels
    :param feature_names: List of feature names
    :param output_dir: Directory to save LIME outputs
    :param model_name: Name of the model for labeling
    :param dataset_name: Name of the dataset
    :param execution_mode: Execution mode string ('separate_files' or 'combined_files')
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary with LIME explanations or None if failed
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        try:  # Attempt to generate LIME explanations
            verbose_output(
                f"{BackgroundColors.GREEN}Generating LIME explanations for {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}",
                config=config
            )  # Log LIME generation start

            lime_config = config.get("explainability", {})  # Get explainability config
            num_features = lime_config.get("lime_num_features", 10)  # Number of features in explanation
            num_samples = lime_config.get("lime_num_samples", 1000)  # Number of samples for LIME
            random_state = lime_config.get("random_state", 42)  # Random state for sampling

            os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

            mode = "classification"  # Default mode
            class_names = [str(c) for c in np.unique(y_test)]  # Get class names

            # Prepare feature names to match X_test columns exactly for LIME
            if hasattr(X_test, 'shape') and hasattr(feature_names, '__len__') and X_test.shape[1] == len(feature_names):
                lime_feature_names = feature_names  # Use provided feature_names when they exactly match X_test columns
            elif hasattr(X_test, 'columns'):
                lime_feature_names = list(X_test.columns)  # Extract feature names from DataFrame columns when available
            else:
                lime_feature_names = feature_names[: X_test.shape[1]] if hasattr(feature_names, '__len__') else [f"f{i}" for i in range(X_test.shape[1])]  # Defensive fallback feature names

            explainer = LimeTabularExplainer(
                X_test,
                feature_names=lime_feature_names,
                class_names=class_names,
                mode=mode,
                random_state=random_state
            )  # Initialize LIME explainer

            predictor_fn = get_shap_prediction_function(model)  # Resolve prediction function for LIME usage

            rng = np.random.default_rng(random_state)  # Create explicit RNG to avoid global seeding
            num_instances_to_explain = min(5, len(X_test))  # Explain up to 5 instances
            instance_indices = rng.choice(len(X_test), size=num_instances_to_explain, replace=False)  # Sample indices reproducibly

            lime_explanations = []  # List to store LIME explanations

            for idx in instance_indices:  # For each instance to explain
                instance = X_test[idx]  # Get instance
                explanation = explainer.explain_instance(
                    instance,
                    predictor_fn,
                    num_features=num_features,
                    num_samples=num_samples
                )  # Generate LIME explanation using resolved prediction function

                try:  # Try to save explanation figure
                    fig = explanation.as_pyplot_figure()  # Get matplotlib figure
                    explanation_plot_path = os.path.join(output_dir, f"{dataset_name}_{model_name}_lime_instance_{idx}.png")  # Build plot path
                    plt.tight_layout()  # Adjust layout
                    ensure_playwright_chromium()  # Ensure Playwright Chromium is installed before saving PNG
                    ensure_figure_min_4k_and_save(fig=plt.gcf(), path=explanation_plot_path, dpi=300, bbox_inches='tight')  # Save plot
                    plt.close()  # Close plot
                except Exception:  # If plot save fails
                    plt.close()  # Close plot

                lime_explanations.append(explanation.as_list())  # Store explanation as list

            verbose_output(
                f"{BackgroundColors.GREEN}LIME explanations saved to {BackgroundColors.CYAN}{output_dir}{Style.RESET_ALL}",
                config=config
            )  # Log LIME completion

            return {"lime_explanations": lime_explanations}  # Return LIME results

        except ImportError:  # If LIME not installed
            print(f"{BackgroundColors.YELLOW}LIME library not installed. Skipping LIME explanations. Install with: pip install lime{Style.RESET_ALL}")  # Warn user
            return None  # Return None
        except Exception as e:  # If any other error
            verbose_output(
                f"{BackgroundColors.YELLOW}Failed to generate LIME explanations for {model_name}: {e}{Style.RESET_ALL}",
                config=config
            )  # Log error
            return None  # Return None
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def generate_permutation_importance(model, X_test, y_test, feature_names, output_dir, model_name, dataset_name, config=None):
    """
    Generate permutation feature importance for a trained model.

    :param model: Trained model object
    :param X_test: Test features (numpy array or DataFrame)
    :param y_test: Test labels
    :param feature_names: List of feature names
    :param output_dir: Directory to save permutation importance outputs
    :param model_name: Name of the model for labeling
    :param dataset_name: Name of the dataset
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary with permutation importance or None if failed
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        try:  # Attempt to generate permutation importance
            verbose_output(
                f"{BackgroundColors.GREEN}Computing permutation importance for {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}",
                config=config
            )  # Log permutation importance computation start

            explainer_config = config.get("explainability", {})  # Get explainability config
            random_state = explainer_config.get("random_state", 42)  # Random state for permutation
            perm_max_samples = explainer_config.get("perm_max_samples", 5000)  # Cap test samples to prevent OOM on large datasets

            os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

            X_test_perm, y_test_perm = sample_shap_test_data(X_test, y_test, perm_max_samples, random_state)  # Sample test data to bounded size to prevent OOM during permutation importance

            perm_importance = permutation_importance(
                model,
                X_test_perm,  # Use sampled subset to prevent OOM on datasets with tens of millions of rows
                y_test_perm,  # Use labels aligned to sampled subset
                n_repeats=10,
                random_state=random_state,
                n_jobs=1  # Force single-threaded execution to prevent nested parallelism explosion with tree ensemble models
            )  # Compute permutation importance

            importances_mean = perm_importance['importances_mean']  # Extract mean importances from Bunch
            importances_std = perm_importance['importances_std']  # Extract std importances from Bunch
            sorted_indices = importances_mean.argsort()[::-1]  # Sort indices by descending importance
            sorted_features = [feature_names[i] for i in sorted_indices]  # Get sorted feature names
            sorted_importances = importances_mean[sorted_indices]  # Get sorted importances
            sorted_std = importances_std[sorted_indices]  # Get sorted standard deviations

            importance_dict = {}  # Initialize importance dictionary
            for feat, imp, std in zip(sorted_features, sorted_importances, sorted_std):  # For each feature
                importance_dict[feat] = {"mean": float(imp), "std": float(std)}  # Store importance and std

            resolved_fig = None  # Initialize resolved_fig to None before plotting to ensure it's defined for cleanup
            try:  # Try to create bar plot
                max_display = explainer_config.get("max_display_features", 20)  # Max features to display
                display_count = min(max_display, len(sorted_features))  # Number of features to display

                resolved_fig = None  # Initialize resolved_fig to None for proper exception handling
                plt.figure(figsize=(10, max(6, display_count * 0.3)))  # Create figure with dynamic height
                resolved_fig = plt.gcf()  # Get the current figure to ensure we can close it in case of exceptions
                y_pos = np.arange(display_count)  # Y positions for bars
                plt.barh(y_pos, sorted_importances[:display_count], xerr=sorted_std[:display_count], align='center', alpha=0.7, color='steelblue')  # Create horizontal bar plot with error bars
                plt.yticks(y_pos, sorted_features[:display_count])  # Set Y-axis labels
                plt.xlabel('Permutation Importance', fontsize=12)  # Set X-axis label
                plt.ylabel('Features', fontsize=12)  # Set Y-axis label
                plt.title(f'Permutation Importance - {model_name}\n{dataset_name}', fontsize=14)  # Set title
                plt.grid(axis='x', alpha=0.3)  # Add X-axis grid
                plt.tight_layout()  # Adjust layout
                perm_plot_path = os.path.join(output_dir, f"{dataset_name}_{model_name}_permutation_importance.png")  # Build plot path
                try:  # Ensure figure has at least 4k pixels before saving
                    resolved_fig.savefig(perm_plot_path, dpi=300, bbox_inches='tight')  # Save plot
                finally:  # Attempt to close the exact figure to free memory
                    plt.close(resolved_fig)  # Close the exact figure to free memory
            except Exception:  # If plot fails
                try:  # Attempt to close any figure that might be open to free memory
                    if resolved_fig is not None:  # If we have a reference to the figure, close it
                        plt.close(resolved_fig)  # Close the exact figure to free memory
                except Exception:  # If closing the figure fails, just pass
                    pass  # If plot creation or saving fails, we ignore the error and continue without the plot

            verbose_output(
                f"{BackgroundColors.GREEN}Permutation importance saved to {BackgroundColors.CYAN}{output_dir}{Style.RESET_ALL}",
                config=config
            )  # Log permutation importance completion

            return {"permutation_importance": importance_dict}  # Return permutation importance results

        except Exception as e:  # If any error
            verbose_output(
                f"{BackgroundColors.YELLOW}Failed to compute permutation importance for {model_name}: {e}{Style.RESET_ALL}",
                config=config
            )  # Log error
            return None  # Return None
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_linear_model_importance(model, feature_names, output_dir, model_name, dataset_name, config=None):
    """
    Extracts feature importances from linear models using the coef_ attribute as a proxy for importance.

    :param model: Trained linear model with a coef_ attribute
    :param feature_names: List of feature names corresponding to model input columns
    :param output_dir: Directory path where output artifacts will be created
    :param model_name: Name of the model used in log messages and file names
    :param dataset_name: Name of the dataset used in log messages and file names
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary with key 'model_importance' mapping feature names to absolute coefficient scores
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Extracting coefficients as feature importance for {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}",
            config=config
        )  # Log linear model importance extraction start

        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists before writing

        coef = model.coef_  # Read coefficient array from the linear model
        if len(coef.shape) > 1:  # Combined files evaluation models have a 2D coefficient matrix
            importances = np.abs(coef).mean(axis=0)  # Average absolute coefficients across all classes
        else:  # Separate files evaluation models have a 1D coefficient array
            importances = np.abs(coef[0] if coef.ndim > 1 else coef)  # Use absolute values of the single coefficient vector

        sorted_indices = np.argsort(importances)[::-1]  # Sort indices from highest to lowest absolute coefficient
        sorted_features = [feature_names[i] for i in sorted_indices]  # Reorder feature names by descending importance
        sorted_importances = importances[sorted_indices]  # Reorder importance values by descending importance

        importance_dict = dict(zip(sorted_features, sorted_importances.tolist()))  # Build feature-name-to-score mapping

        verbose_output(
            f"{BackgroundColors.GREEN}Model coefficients saved as importance for {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}",
            config=config
        )  # Log successful extraction completion

        return {"model_importance": importance_dict}  # Return the importance dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_tree_based_importance(model, feature_names, output_dir, model_name, dataset_name, config=None):
    """
    Extracts and plots feature importances from tree-based models using the feature_importances_ attribute.

    :param model: Trained tree-based model with a feature_importances_ attribute
    :param feature_names: List of feature names corresponding to model input columns
    :param output_dir: Directory path where the importance bar plot will be saved
    :param model_name: Name of the model used in plot titles and file names
    :param dataset_name: Name of the dataset used in plot titles and file names
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary with key 'model_importance' mapping feature names to importance scores
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Extracting built-in feature importance for {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}",
            config=config
        )  # Log tree-based importance extraction start

        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists before writing

        importances = model.feature_importances_  # Read raw importance scores from the model
        sorted_indices = np.argsort(importances)[::-1]  # Sort indices from highest to lowest importance
        sorted_features = [feature_names[i] for i in sorted_indices]  # Reorder feature names by descending importance
        sorted_importances = importances[sorted_indices]  # Reorder importance values by descending importance

        importance_dict = dict(zip(sorted_features, sorted_importances.tolist()))  # Build feature-name-to-score mapping

        try:  # Attempt to generate the importance bar plot
            explainer_config = config.get("explainability", {})  # Read explainability section from config
            max_display = explainer_config.get("max_display_features", 20)  # Maximum features shown in the plot
            display_count = min(max_display, len(sorted_features))  # Clamp display count to available features

            plt.figure(figsize=(10, max(6, display_count * 0.3)))  # Create figure sized relative to feature count
            y_pos = np.arange(display_count)  # Compute bar Y-axis positions
            plt.barh(y_pos, sorted_importances[:display_count], align='center', alpha=0.7, color='forestgreen')  # Draw horizontal bars
            plt.yticks(y_pos, sorted_features[:display_count])  # Label Y-axis with feature names
            plt.xlabel('Feature Importance', fontsize=12)  # Label X-axis
            plt.ylabel('Features', fontsize=12)  # Label Y-axis
            plt.title(f'Model Feature Importance - {model_name}\n{dataset_name}', fontsize=14)  # Set descriptive title
            plt.grid(axis='x', alpha=0.3)  # Add subtle X-axis grid lines
            plt.tight_layout()  # Adjust figure padding
            importance_plot_path = os.path.join(output_dir, f"{dataset_name}_{model_name}_feature_importance.png")  # Construct output file path
            ensure_figure_min_4k_and_save(fig=plt.gcf(), path=importance_plot_path, dpi=300, bbox_inches='tight')  # Save plot with minimum 4K resolution
            plt.close()  # Release figure resources
        except Exception:  # If plot generation fails
            plt.close()  # Close figure to avoid resource leak

        verbose_output(
            f"{BackgroundColors.GREEN}Model feature importance saved to {BackgroundColors.CYAN}{output_dir}{Style.RESET_ALL}",
            config=config
        )  # Log successful extraction and plot save

        return {"model_importance": importance_dict}  # Return the importance dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_model_feature_importance(model, feature_names, output_dir, model_name, dataset_name, config=None):
    """
    Extract built-in feature importance from model if supported.

    :param model: Trained model object
    :param feature_names: List of feature names
    :param output_dir: Directory to save feature importance outputs
    :param model_name: Name of the model for labeling
    :param dataset_name: Name of the dataset
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary with feature importance or None if not supported
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        try:  # Attempt to extract feature importance
            model_type = model.__class__.__name__  # Get model class name

            if hasattr(model, 'feature_importances_'):  # If model has built-in feature importance
                return extract_tree_based_importance(model, feature_names, output_dir, model_name, dataset_name, config=config)  # Delegate to tree-based importance extractor

            elif hasattr(model, 'coef_'):  # If linear model with coefficients
                return extract_linear_model_importance(model, feature_names, output_dir, model_name, dataset_name, config=config)  # Delegate to linear coefficient importance extractor

            else:  # Model does not support feature importance
                verbose_output(
                    f"{BackgroundColors.YELLOW}Model {model_name} does not support built-in feature importance{Style.RESET_ALL}",
                    config=config
                )  # Log unsupported model
                return None  # Return None

        except Exception as e:  # If any error
            verbose_output(
                f"{BackgroundColors.YELLOW}Failed to extract feature importance for {model_name}: {e}{Style.RESET_ALL}",
                config=config
            )  # Log error
            return None  # Return None
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def generate_combined_importance_report(shap_result, lime_result, perm_result, model_result, feature_names, output_dir, model_name, dataset_name, config=None):
    """
    Generate a combined importance report aggregating all explainability methods.

    :param shap_result: SHAP results dictionary or None
    :param lime_result: LIME results dictionary or None
    :param perm_result: Permutation importance results dictionary or None
    :param model_result: Model feature importance results dictionary or None
    :param feature_names: List of feature names
    :param output_dir: Directory to save combined report
    :param model_name: Name of the model for labeling
    :param dataset_name: Name of the dataset
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Path to saved report or None if failed
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        try:  # Attempt to generate combined report
            verbose_output(
                f"{BackgroundColors.GREEN}Generating combined importance report for {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}",
                config=config
            )  # Log report generation start

            os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

            report_data = []  # List to store report rows

            for feature in feature_names:  # For each feature
                row = {"feature": feature}  # Initialize row with feature name

                if shap_result and "shap_importance" in shap_result:  # If SHAP results available
                    row["shap_importance"] = shap_result["shap_importance"].get(feature, 0.0)  # Get SHAP importance
                else:  # If SHAP not available
                    row["shap_importance"] = np.nan  # Set to NaN

                if perm_result and "permutation_importance" in perm_result:  # If permutation results available
                    perm_dict = perm_result["permutation_importance"].get(feature, {})  # Get permutation dict for feature
                    row["permutation_importance_mean"] = perm_dict.get("mean", np.nan)  # Get mean importance
                    row["permutation_importance_std"] = perm_dict.get("std", np.nan)  # Get std

                else:  # If permutation not available
                    row["permutation_importance_mean"] = np.nan  # Set to NaN
                    row["permutation_importance_std"] = np.nan  # Set to NaN

                if model_result and "model_importance" in model_result:  # If model importance available
                    row["model_importance"] = model_result["model_importance"].get(feature, 0.0)  # Get model importance
                else:  # If model importance not available
                    row["model_importance"] = np.nan  # Set to NaN

                report_data.append(row)  # Add row to report data

            report_df = pd.DataFrame(report_data)  # Create DataFrame from report data

            importance_cols = [col for col in report_df.columns if col != "feature" and "std" not in col]  # Get importance columns
            if len(importance_cols) >= 2:  # If at least 2 importance methods available
                for col in importance_cols:  # For each importance column
                    report_df[f"{col}_rank"] = report_df[col].rank(ascending=False, na_option='bottom')  # Compute rank

                rank_cols = [f"{col}_rank" for col in importance_cols]  # Get rank column names
                report_df["average_rank"] = report_df[rank_cols].mean(axis=1)  # Compute average rank
                report_df["rank_std"] = report_df[rank_cols].std(axis=1)  # Compute rank standard deviation
                report_df["consistency_score"] = 1.0 / (1.0 + report_df["rank_std"])  # Compute consistency score (higher = more consistent)

                report_df = report_df.sort_values("average_rank")  # Sort by average rank

            report_path = os.path.join(output_dir, f"{dataset_name}_{model_name}_combined_importance.csv")  # Build report path
            generate_csv_and_image(report_df, report_path, is_visualizable=True, index=False)  # Save report CSV and generate PNG image

            verbose_output(
                f"{BackgroundColors.GREEN}Combined importance report saved to {BackgroundColors.CYAN}{report_path}{Style.RESET_ALL}",
                config=config
            )  # Log report completion

            return report_path  # Return path to saved report

        except Exception as e:  # If any error
            verbose_output(
                f"{BackgroundColors.YELLOW}Failed to generate combined importance report for {model_name}: {e}{Style.RESET_ALL}",
                config=config
            )  # Log error
            return None  # Return None
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_explainability_dataset_label(input_path: str, execution_mode: str, config=None) -> str:
    """
    Build a mode-aware dataset label for explainability outputs.

    :param input_path: Path to the dataset file or dataset directory used in the experiment.
    :param execution_mode: Execution mode string, e.g., 'combined_files' or 'single_file'.
    :return: Constructed dataset label string suitable for directory and file names.
    """

    try:
        p = Path(input_path)  # Convert input_path to Path object for manipulations.
        base = get_dataset_name(str(input_path))  # Get base dataset name using existing function.
        parts = p.parts  # Extract path segments for post-processing.
        label = base  # Initialize label with the base dataset name.
        if execution_mode == "combined_files":  # Combined mode: derive dataset-level label.
            try:
                idx = parts.index(base)  # Locate base dataset name position in path parts.
            except ValueError:
                idx = None  # Fallback when base not present in path parts.
            if idx is not None:  # Proceed only when base was found.
                last_part = parts[-1]  # Inspect last segment to determine if it's a filename.
                is_file_like = "." in last_part  # Heuristic: dot in segment suggests a filename.
                if is_file_like:
                    remaining = parts[idx+1:-1]  # Exclude filename from remaining segments.
                else:
                    remaining = parts[idx+1:]  # Include all trailing directory parts when no file present.
                if remaining:  # If there are trailing segments, append them with underscores.
                    label = base + "_" + "_".join([seg for seg in remaining])  # Build final label by joining segments.
        else:
            label = p.stem  # Single-file mode: use filename stem as label.
        return label  # Return assembled label.
    except Exception as e:
        return Path(input_path).stem  # Fallback to stem on unexpected errors.


def run_explainability_pipeline(model, model_name, X_test, y_test, feature_names, dataset_file, feature_set, execution_mode, config=None):
    """
    Run comprehensive explainability pipeline for a trained model.

    Generates SHAP, LIME, permutation importance, and combined reports.
    Only runs on ORIGINAL test data, never on augmented samples.

    :param model: Trained model object
    :param model_name: Name of the model for labeling
    :param X_test: Original test features (numpy array, NOT augmented)
    :param y_test: Original test labels (NOT augmented)
    :param feature_names: List of feature names
    :param dataset_file: Path to dataset file for output directory construction
    :param feature_set: Feature set name (e.g., "Full Features", "GA Features")
    :param execution_mode: Execution mode string ('separate_files' or 'combined_files')
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary with all explainability results or None if disabled
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        explainability_config = config.get("explainability", {})  # Get explainability config
        if not explainability_config.get("enabled", False):  # If explainability is disabled
            return None  # Return None immediately

        verbose_output(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Running explainability pipeline for {BackgroundColors.CYAN}{model_name} - {feature_set}{Style.RESET_ALL}",
            config=config
        )  # Log pipeline start

        dataset_label = get_explainability_dataset_label(dataset_file, execution_mode, config)  # Build mode-aware dataset label
        output_subdir = explainability_config.get("output_subdir", "Explainability")  # Get output subdirectory name
        base_output_dir = Path(".") / output_subdir / execution_mode / dataset_label  # Use relative path root and mode-aware label
        output_dir = base_output_dir / feature_set.replace(" ", "_") / model_name.replace(" ", "_")  # Build full output directory
        output_dir = str(output_dir)  # Convert Path to string
        Path(output_dir).mkdir(parents=True, exist_ok=True)  # Ensure output directories exist before writing

        all_results = {}  # Dictionary to store all explainability results

        if explainability_config.get("shap", True):  # If SHAP is enabled
            shap_result = generate_shap_explanations(
                model, X_test, y_test, feature_names, output_dir, model_name, dataset_label, execution_mode, config
            )  # Generate SHAP explanations
            if shap_result:  # If SHAP results available
                all_results.update(shap_result)  # Add SHAP results to all results

        if explainability_config.get("lime", True):  # If LIME is enabled
            lime_result = generate_lime_explanations(
                model, X_test, y_test, feature_names, output_dir, model_name, dataset_label, execution_mode, config
            )  # Generate LIME explanations
            if lime_result:  # If LIME results available
                all_results.update(lime_result)  # Add LIME results to all results

        if explainability_config.get("permutation_importance", True):  # If permutation importance is enabled
            perm_result = generate_permutation_importance(
                model, X_test, y_test, feature_names, output_dir, model_name, dataset_label, config
            )  # Generate permutation importance
            if perm_result:  # If permutation results available
                all_results.update(perm_result)  # Add permutation results to all results

        if explainability_config.get("feature_importance", True):  # If feature importance extraction is enabled
            model_result = extract_model_feature_importance(
                model, feature_names, output_dir, model_name, dataset_label, config
            )  # Extract model feature importance
            if model_result:  # If model importance available
                all_results.update(model_result)  # Add model importance to all results

        shap_res = all_results if "shap_importance" in all_results else None  # Get SHAP results or None
        lime_res = all_results if "lime_explanations" in all_results else None  # Get LIME results or None
        perm_res = all_results if "permutation_importance" in all_results else None  # Get permutation results or None
        model_res = all_results if "model_importance" in all_results else None  # Get model importance or None

        report_path = generate_combined_importance_report(
            shap_res, lime_res, perm_res, model_res, feature_names, output_dir, model_name, dataset_label, config
        )  # Generate combined report
        if report_path:  # If report generated successfully
            all_results["combined_report_path"] = report_path  # Add report path to results

        verbose_output(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Explainability pipeline completed for {BackgroundColors.CYAN}{model_name} - {feature_set}{Style.RESET_ALL}",
            config=config
        )  # Log pipeline completion

        return all_results  # Return all explainability results
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def copy_explainability_value(value: Any) -> Any:
    """
    Copy an explainability input value for background ownership.

    :param value: Value that should be detached from later caller mutations.
    :return: Copied value when possible, otherwise the original value.
    """

    if value is None:  # Preserve None values without allocation.
        return None  # Return None directly.
    if hasattr(value, "copy"):  # Use native copy semantics for dataframe, series, and array objects.
        try:  # Prefer pandas-style deep copy when supported.
            return value.copy(deep=True)  # Copy pandas-style objects with deep ownership.
        except TypeError:  # Fall back for numpy-style copy signatures.
            return value.copy()  # Copy numpy-style objects with native ownership.
    if isinstance(value, (list, tuple, dict)):  # Use serialization for common mutable containers.
        return pickle.loads(pickle.dumps(value))  # Return a detached serialized copy.
    return value  # Return immutable or opaque values unchanged.


def copy_explainability_model(model: Any) -> Any:
    """
    Copy fitted model state for background explainability execution.

    :param model: Fitted model object to detach from later training mutations.
    :return: Serialized copy of the fitted model.
    """

    return pickle.loads(pickle.dumps(model))  # Snapshot fitted model state before any later refit can mutate it.


def snapshot_explainability_inputs(model: Any, X_test: Any, y_test: Any, feature_names: Any, config: dict) -> Tuple[Any, Any, Any, List[Any], dict]:
    """
    Snapshot explainability inputs before dispatching background execution.

    :param model: Fitted model object for explainability.
    :param X_test: Test feature matrix for explainability.
    :param y_test: Test labels for explainability.
    :param feature_names: Feature names associated with the test feature matrix.
    :param config: Configuration dictionary used by explainability routines.
    :return: Tuple containing detached model, test data, labels, feature names, and config.
    """

    model_snapshot = copy_explainability_model(model)  # Snapshot fitted model state for background ownership.
    X_test_snapshot = copy_explainability_value(X_test)  # Copy test feature data before caller reuse or release.
    y_test_snapshot = copy_explainability_value(y_test)  # Copy test labels before caller reuse or release.
    feature_names_snapshot = list(feature_names) if feature_names is not None else []  # Snapshot feature metadata as an independent list.
    config_snapshot = pickle.loads(pickle.dumps(config))  # Snapshot configuration so later mutations do not affect queued work.
    return model_snapshot, X_test_snapshot, y_test_snapshot, feature_names_snapshot, config_snapshot  # Return the detached explainability inputs.


def execute_explainability_job(model: Any, model_name: str, X_test: Any, y_test: Any, feature_names: List[Any], dataset_file: str, feature_set: str, execution_mode: str, config: dict) -> Optional[dict]:
    """
    Run one explainability job and propagate its failures.

    :param model: Detached fitted model for explainability.
    :param model_name: Model name used for labels and filenames.
    :param X_test: Detached test feature matrix.
    :param y_test: Detached test labels.
    :param feature_names: Detached feature name list.
    :param dataset_file: Dataset file path used for output directory construction.
    :param feature_set: Feature set label used for output directory construction.
    :param execution_mode: Execution mode string used for output directory construction.
    :param config: Detached configuration dictionary used by explainability routines.
    :return: Explainability result dictionary, or None when explainability is disabled.
    """

    try:  # Run the existing explainability implementation under one failure boundary.
        return run_explainability_pipeline(model, model_name, X_test, y_test, feature_names, dataset_file, feature_set, execution_mode, config)  # Run the existing explainability pipeline.
    except Exception as e:  # Log explainability failure before propagating it.
        print(str(e))  # Print the error through the existing logging path.
        verbose_output(f"{BackgroundColors.YELLOW}Explainability failed for {model_name}: {e}{Style.RESET_ALL}", config=config)  # Log failure using the existing verbose pattern.
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send failure details through the existing Telegram exception path.
        raise  # Preserve explainability failure propagation.


def prepare_explainability_child_config(config: dict) -> dict:
    """
    Prepare a process-safe configuration snapshot for explainability.

    :param config: Configuration dictionary used by explainability routines.
    :return: Detached configuration dictionary for a child process.
    """

    config_snapshot = pickle.loads(pickle.dumps(config))  # Detach configuration before it crosses the process boundary.
    config_snapshot.setdefault("logging", {})["clean"] = False  # Prevent child logger initialization from truncating the parent log.
    return config_snapshot  # Return child-safe configuration snapshot.


def run_explainability_process(status_queue: Any, model: Any, model_name: str, X_test: Any, y_test: Any, feature_names: List[Any], dataset_file: str, feature_set: str, execution_mode: str, config: dict) -> None:
    """
    Run explainability inside a child process and report lifecycle status.

    :param status_queue: Multiprocessing queue used to report child status.
    :param model: Detached fitted model for explainability.
    :param model_name: Model name used for labels and filenames.
    :param X_test: Detached test feature matrix.
    :param y_test: Detached test labels.
    :param feature_names: Detached feature name list.
    :param dataset_file: Dataset file path used for output directory construction.
    :param feature_set: Feature set label used for output directory construction.
    :param execution_mode: Execution mode string used for output directory construction.
    :param config: Detached configuration dictionary used by explainability routines.
    :return: None.
    """

    global CONFIG  # Allow the child process to reuse the standard CONFIG fallback.
    global logger  # Allow child logger initialization to update the module logger.
    try:  # Initialize child-side runtime state before reporting startup.
        CONFIG = config  # Set child CONFIG so existing fallback logic remains valid.
        if logger is None:  # Initialize child logging only when the child has no logger.
            initialize_logger(config=config)  # Attach child stdout and stderr to the existing log file in append mode.
        setup_telegram_bot(config=config)  # Initialize Telegram in the child when configured.
        status_queue.put({"status": "started", "pid": os.getpid(), "model_name": model_name, "feature_set": feature_set})  # Confirm successful child startup before heavy explainability begins.
        result = execute_explainability_job(model, model_name, X_test, y_test, feature_names, dataset_file, feature_set, execution_mode, config)  # Run the existing explainability pipeline without reimplementation.
        result_keys = sorted(result.keys()) if isinstance(result, dict) else []  # Summarize result shape without sending large artifacts through the queue.
        status_queue.put({"status": "success", "pid": os.getpid(), "model_name": model_name, "feature_set": feature_set, "result_keys": result_keys})  # Report successful explainability completion.
    except BaseException as e:  # Report any child failure before preserving the nonzero exit.
        status_queue.put({"status": "failure", "pid": os.getpid(), "model_name": model_name, "feature_set": feature_set, "exception_type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()})  # Send structured child failure details to the parent.
        try:  # Preserve existing Telegram exception reporting for child failures.
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Notify Telegram about the child failure.
        except Exception:  # Avoid masking the original child failure during notification.
            pass  # Preserve the original child failure.
        raise  # Preserve nonzero child process exit status.
    finally:  # Flush child logs before process exit.
        try:  # Flush child logger when it exists.
            if logger is not None:  # Verify child logger exists before flushing.
                logger.flush()  # Flush child log writes.
        except Exception:  # Avoid masking child process status during logger flush.
            pass  # Preserve the original child process status.


def drain_explainability_status_queue(status_queue: Any) -> List[dict]:
    """
    Drain queued child process status messages without blocking.

    :param status_queue: Multiprocessing queue containing child status messages.
    :return: List of status dictionaries.
    """

    statuses: List[dict] = []  # Collect every available child status message.
    while True:  # Drain until the queue reports no immediately available message.
        try:  # Read one status message without blocking.
            statuses.append(status_queue.get_nowait())  # Append the available child status message.
        except queue_module.Empty:  # Stop when the queue is empty.
            break  # Leave the draining loop.
    return statuses  # Return drained status messages.


def await_explainability_process_start(process_record: dict, timeout_seconds: float, config: Optional[dict]) -> dict:
    """
    Wait for a child process to confirm explainability startup.

    :param process_record: Process record containing process and status queue.
    :param timeout_seconds: Maximum startup wait in seconds.
    :param config: Configuration dictionary used for logging.
    :return: Startup status dictionary.
    """

    process = process_record["process"]  # Read child process from process record.
    status_queue = process_record["status_queue"]  # Read child status queue from process record.
    deadline = time.time() + float(timeout_seconds)  # Compute absolute startup deadline.
    while time.time() < deadline:  # Wait until child startup succeeds, fails, or times out.
        try:  # Poll child status with a short timeout.
            status = status_queue.get(timeout=0.2)  # Wait briefly for a startup status message.
            process_record.setdefault("statuses", []).append(status)  # Preserve startup status for finalization logs.
            if status.get("status") == "started":  # Verify successful child startup status.
                return status  # Return startup status to the scheduler.
            if status.get("status") == "failure":  # Treat startup failure as unsafe asynchronous dispatch.
                raise RuntimeError(f"Explainability child failed before startup confirmation for {process_record.get('model_name')}: {status.get('message')}")  # Raise structured startup failure.
        except queue_module.Empty:  # No status arrived during this poll interval.
            pass  # Continue waiting until deadline or process exit.
        if process.exitcode is not None:  # Detect child exit before startup confirmation.
            raise RuntimeError(f"Explainability child exited before startup confirmation for {process_record.get('model_name')} with exitcode {process.exitcode}")  # Raise failed startup status.
    raise TimeoutError(f"Explainability child did not confirm startup for {process_record.get('model_name')} within {timeout_seconds:.1f}s")  # Raise startup timeout.


def finalize_explainability_process_record(process_record: dict, config: Optional[dict], terminate: bool) -> None:
    """
    Join one explainability process and surface its final status.

    :param process_record: Process record containing process, queue, and metadata.
    :param config: Configuration dictionary used for logging.
    :param terminate: Whether to terminate a live child during abnormal shutdown.
    :return: None.
    """

    process = process_record["process"]  # Read child process from process record.
    model_name = process_record.get("model_name", "unknown")  # Resolve model name for logs.
    feature_set = process_record.get("feature_set", "unknown")  # Resolve feature set for logs.
    if terminate and process.is_alive():  # Terminate live child during abnormal shutdown.
        print(f"{BackgroundColors.YELLOW}[WARNING] Terminating asynchronous explainability child PID {process.pid} for {model_name} - {feature_set}.{Style.RESET_ALL}")  # Log abnormal child termination.
        process.terminate()  # Ask the child process to terminate.
        process.join(timeout=EXPLAINABILITY_PROCESS_JOIN_TIMEOUT_SECONDS)  # Wait briefly after terminate.
        if process.is_alive():  # Escalate only when the child ignored terminate.
            process.kill()  # Force-stop the child process.
            process.join(timeout=EXPLAINABILITY_PROCESS_JOIN_TIMEOUT_SECONDS)  # Reap the force-stopped child.
    else:  # Normal finalization waits for child completion.
        while process.is_alive():  # Wait for explainability completion without busy waiting.
            process.join(timeout=EXPLAINABILITY_PROCESS_JOIN_TIMEOUT_SECONDS)  # Reap progress in bounded intervals.
    process.join(timeout=0.0)  # Reap final process status after it has stopped.
    statuses = list(process_record.get("statuses", []))  # Start with statuses consumed during startup confirmation.
    statuses.extend(drain_explainability_status_queue(process_record["status_queue"]))  # Add any completion or failure statuses.
    process_record["statuses"] = statuses  # Store complete statuses for diagnostics.
    exitcode = process.exitcode  # Read final child exit code.
    status_names = [str(status.get("status")) for status in statuses]  # Extract compact status names for logs.
    success_seen = any(status.get("status") == "success" for status in statuses)  # Detect successful explainability completion.
    failure_statuses = [status for status in statuses if status.get("status") == "failure"]  # Collect structured child failures.
    verbose_output(f"{BackgroundColors.GREEN}[DEBUG] Explainability child finalized: model={model_name}, feature_set={feature_set}, pid={process.pid}, exitcode={exitcode}, statuses={status_names}{Style.RESET_ALL}", config=config)  # Log final child process status.
    try:  # Release queue feeder resources after draining statuses.
        process_record["status_queue"].close()  # Close the multiprocessing queue in the parent.
        process_record["status_queue"].join_thread()  # Join queue feeder resources.
    except Exception:  # Keep queue cleanup best-effort after child has ended.
        pass  # Preserve child failure reporting.
    try:  # Release Process resources after join.
        process.close()  # Close process object resources after reaping.
    except Exception:  # Keep process object cleanup best-effort.
        pass  # Preserve child failure reporting.
    if terminate:  # Abnormal shutdown cleanup should not mask the original program failure.
        return  # Return after cleanup during abnormal shutdown.
    if failure_statuses:  # Surface structured child failure during normal finalization.
        failure = failure_statuses[-1]  # Use the latest failure status for the error message.
        raise RuntimeError(f"Explainability child failed for {model_name} - {feature_set}: {failure.get('exception_type')}: {failure.get('message')}")  # Raise child failure before successful program completion.
    if exitcode not in (0, None):  # Surface nonzero child exit without a structured failure message.
        raise RuntimeError(f"Explainability child exited with code {exitcode} for {model_name} - {feature_set}")  # Raise nonzero child exit.
    if not success_seen:  # Require explicit child success on normal finalization.
        raise RuntimeError(f"Explainability child ended without success status for {model_name} - {feature_set}")  # Raise missing success status.


def reap_finished_explainability_processes(config: Optional[dict] = None) -> None:
    """
    Reap completed asynchronous explainability processes.

    :param config: Configuration dictionary used for logging.
    :return: None.
    """

    if config is None:  # Use global configuration when no configuration is supplied.
        config = CONFIG  # Preserve existing CONFIG fallback behavior.
    tracked_processes = list(EXPLAINABILITY_PROCESSES)  # Snapshot tracked process records before mutation.
    EXPLAINABILITY_PROCESSES.clear()  # Rebuild the active process list after reaping completed records.
    for process_record in tracked_processes:  # Inspect each tracked explainability child.
        process = process_record.get("process")  # Read process object from record.
        if process is not None and process.is_alive():  # Keep live child records for later finalization.
            EXPLAINABILITY_PROCESSES.append(process_record)  # Preserve active child process record.
        else:  # Reap completed child process immediately.
            finalize_explainability_process_record(process_record, config=config, terminate=False)  # Join and validate completed child process status.


def start_explainability_process(model: Any, model_name: str, X_test: Any, y_test: Any, feature_names: Any, dataset_file: str, feature_set: str, execution_mode: str, config: dict) -> dict:
    """
    Start one asynchronous explainability process after snapshotting inputs.

    :param model: Fitted model object from the completed classifier result.
    :param model_name: Model name used for labels and filenames.
    :param X_test: Test feature matrix for explainability.
    :param y_test: Test labels for explainability.
    :param feature_names: Feature names associated with the test feature matrix.
    :param dataset_file: Dataset file path used for output directory construction.
    :param feature_set: Feature set label used for output directory construction.
    :param execution_mode: Execution mode string used for output directory construction.
    :param config: Configuration dictionary used by explainability routines.
    :return: Process record for the started child process.
    """

    model_snapshot, X_test_snapshot, y_test_snapshot, feature_names_snapshot, config_snapshot = snapshot_explainability_inputs(model, X_test, y_test, feature_names, config)  # Snapshot only explainability inputs before process dispatch.
    config_snapshot = prepare_explainability_child_config(config_snapshot)  # Prepare child configuration without truncating logs.
    context = mp.get_context("spawn")  # Use spawn context for deterministic process isolation across platforms.
    status_queue = context.Queue()  # Create a status queue owned by the same multiprocessing context.
    process = context.Process(target=run_explainability_process, args=(status_queue, model_snapshot, model_name, X_test_snapshot, y_test_snapshot, feature_names_snapshot, dataset_file, feature_set, execution_mode, config_snapshot), name=f"Explainability-{model_name}")  # Build child process for existing explainability pipeline.
    process.daemon = False  # Keep child non-daemonic so it can finish artifact writes and queue status.
    process_record = {"process": process, "status_queue": status_queue, "model_name": model_name, "feature_set": feature_set, "statuses": []}  # Build process record before startup.
    try:  # Ensure failed startup leaves no live child behind.
        process.start()  # Start the child process.
        if process.pid is None:  # Treat missing PID as failed startup.
            raise RuntimeError(f"Explainability child did not provide a PID for {model_name} - {feature_set}")  # Raise unsafe startup status.
        start_status = await_explainability_process_start(process_record, EXPLAINABILITY_PROCESS_START_TIMEOUT_SECONDS, config)  # Wait for explicit child startup confirmation.
        process_record["started_status"] = start_status  # Store startup status for diagnostics.
        return process_record  # Return confirmed process record.
    except Exception:  # Reap any partially started child before synchronous fallback.
        try:  # Terminate partial child process if it is still alive.
            if process.is_alive():  # Verify process liveness before termination.
                process.terminate()  # Terminate unsafe partial child startup.
                process.join(timeout=EXPLAINABILITY_PROCESS_JOIN_TIMEOUT_SECONDS)  # Reap terminated partial child.
                if process.is_alive():  # Escalate if terminate did not stop the child.
                    process.kill()  # Force-stop unsafe partial child startup.
                    process.join(timeout=EXPLAINABILITY_PROCESS_JOIN_TIMEOUT_SECONDS)  # Reap force-stopped partial child.
            else:  # Process already exited during startup failure.
                process.join(timeout=EXPLAINABILITY_PROCESS_JOIN_TIMEOUT_SECONDS)  # Reap exited partial child.
        except Exception:  # Preserve original startup failure if cleanup fails.
            pass  # Preserve startup failure propagation.
        try:  # Release partial status queue resources.
            status_queue.close()  # Close partial startup status queue.
            status_queue.join_thread()  # Join partial startup queue feeder.
        except Exception:  # Preserve original startup failure if queue cleanup fails.
            pass  # Preserve startup failure propagation.
        try:  # Release partial Process resources after cleanup.
            process.close()  # Close partial child process object.
        except Exception:  # Preserve original startup failure if process close fails.
            pass  # Preserve startup failure propagation.
        raise  # Propagate startup failure to trigger synchronous fallback.


def schedule_explainability_job(model: Any, model_name: str, X_test: Any, y_test: Any, feature_names: Any, dataset_file: str, feature_set: str, execution_mode: str, config: Optional[dict], experiment_mode: str, hyperparameters_enabled: bool, training_ram_stats: Optional[dict] = None) -> Optional[dict]:
    """
    Dispatch explainability after classifier result completion.

    :param model: Fitted model object from the completed classifier result.
    :param model_name: Model name used for labels and filenames.
    :param X_test: Test feature matrix for explainability.
    :param y_test: Test labels for explainability.
    :param feature_names: Feature names associated with the test feature matrix.
    :param dataset_file: Dataset file path used for output directory construction.
    :param feature_set: Feature set label used for output directory construction.
    :param execution_mode: Execution mode string used for output directory construction.
    :param config: Configuration dictionary used by explainability routines.
    :param experiment_mode: Experiment mode string controlling original-only explainability.
    :param hyperparameters_enabled: Whether this classifier result used optimized hyperparameters.
    :param training_ram_stats: RAM statistics collected during classifier training.
    :return: Dispatch outcome dictionary, or None when explainability is disabled for this classifier.
    """

    if config is None:  # Use global configuration when no configuration is provided.
        config = CONFIG  # Preserve existing CONFIG fallback behavior.
    explainability_config = config.get("explainability", {})  # Read explainability settings from configuration.
    if not explainability_config.get("enabled", False):  # Preserve disabled-run behavior.
        return None  # Return without dispatch when explainability is disabled.
    if experiment_mode != "original_only":  # Preserve original-only explainability behavior.
        return None  # Return without dispatch for augmented experiments.
    hyperparameter_label = "Optimized Hyperparameters" if hyperparameters_enabled else "Default Hyperparameters"  # Resolve HP label for isolated output paths.
    explainability_feature_set = f"{feature_set} - {hyperparameter_label}"  # Isolate explainability artifacts across default and optimized HP runs.
    average_training_ram = resolve_training_ram_percent(training_ram_stats, "average_percent")  # Read average classifier training RAM usage.
    latest_training_ram = resolve_training_ram_percent(training_ram_stats, "latest_percent")  # Read latest classifier training RAM usage.
    dispatch_ram = read_system_ram_percent()  # Read current system RAM usage at the exact explainability dispatch point.
    threshold = EXPLAINABILITY_RAM_THRESHOLD_PERCENT  # Resolve strict asynchronous RAM threshold.
    can_dispatch_async = average_training_ram is not None and average_training_ram < threshold and dispatch_ram < threshold  # Apply strict RAM criteria for asynchronous explainability.
    if can_dispatch_async:  # Attempt process-isolated explainability only when RAM criteria are satisfied.
        reap_finished_explainability_processes(config=config)  # Reap any completed child before considering new dispatch.
        if EXPLAINABILITY_PROCESSES:  # Enforce one active asynchronous explainability process at a time.
            verbose_output(f"{BackgroundColors.GREEN}[DEBUG] Waiting for active asynchronous explainability before dispatching {model_name}: active={len(EXPLAINABILITY_PROCESSES)}{Style.RESET_ALL}", config=config)  # Log bounded concurrency synchronization.
            finalize_pending_explainability_jobs(config=config, terminate=False)  # Wait for existing child before starting another child.
        try:  # Start process-isolated explainability and fall back to synchronous on unsafe startup.
            process_record = start_explainability_process(model, model_name, X_test, y_test, feature_names, dataset_file, explainability_feature_set, execution_mode, config)  # Start child and wait for startup confirmation.
            EXPLAINABILITY_PROCESSES.append(process_record)  # Track child for later finalization.
            verbose_output(f"{BackgroundColors.GREEN}[DEBUG] Explainability dispatch: model={model_name}, feature_set={explainability_feature_set}, avg_training_ram={format_ram_percent(average_training_ram)}, latest_training_ram={format_ram_percent(latest_training_ram)}, dispatch_ram={format_ram_percent(dispatch_ram)}, mode=asynchronous, pid={process_record['process'].pid}{Style.RESET_ALL}", config=config)  # Log asynchronous dispatch decision.
            return {"mode": "asynchronous", "pid": process_record["process"].pid, "average_training_ram": average_training_ram, "latest_training_ram": latest_training_ram, "dispatch_ram": dispatch_ram}  # Return asynchronous dispatch metadata.
        except Exception as e:  # Fall back to synchronous explainability when process startup is unsafe.
            print(f"{BackgroundColors.YELLOW}[WARNING] Asynchronous explainability could not start for {model_name} - {explainability_feature_set}: {e}. Falling back to synchronous execution.{Style.RESET_ALL}")  # Log fallback reason using existing warning style.
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Report unsafe asynchronous startup through existing exception notification.
    else:  # Use synchronous explainability when RAM criteria are not satisfied.
        reason = "training_average_unavailable" if average_training_ram is None else "ram_threshold_not_met"  # Resolve synchronous dispatch reason.
        verbose_output(f"{BackgroundColors.GREEN}[DEBUG] Explainability dispatch: model={model_name}, feature_set={explainability_feature_set}, avg_training_ram={format_ram_percent(average_training_ram)}, latest_training_ram={format_ram_percent(latest_training_ram)}, dispatch_ram={format_ram_percent(dispatch_ram)}, threshold={format_ram_percent(threshold)}, mode=synchronous, reason={reason}{Style.RESET_ALL}", config=config)  # Log synchronous dispatch decision.
    execute_explainability_job(model, model_name, X_test, y_test, list(feature_names) if feature_names is not None else [], dataset_file, explainability_feature_set, execution_mode, config)  # Run explainability before the next classifier starts.
    gc.collect()  # Reclaim explainability temporaries before returning to the classifier loop.
    verbose_output(f"{BackgroundColors.GREEN}Completed synchronous explainability for {BackgroundColors.CYAN}{model_name} - {explainability_feature_set}{Style.RESET_ALL}", config=config)  # Log synchronous explainability completion.
    return {"mode": "synchronous", "pid": None, "average_training_ram": average_training_ram, "latest_training_ram": latest_training_ram, "dispatch_ram": dispatch_ram}  # Return synchronous dispatch metadata.


def finalize_pending_explainability_jobs(config: Optional[dict] = None, terminate: bool = False) -> None:
    """
    Wait for asynchronous explainability processes and surface failures.

    :param config: Configuration dictionary used for logging.
    :param terminate: Whether live children should be terminated during abnormal shutdown.
    :return: None.
    """

    if config is None:  # Use global configuration when no configuration is provided.
        config = CONFIG  # Preserve existing CONFIG fallback behavior.
    tracked_processes = list(EXPLAINABILITY_PROCESSES)  # Snapshot tracked child processes before waiting.
    if not tracked_processes:  # Skip finalization when no child processes are tracked.
        return  # Return immediately when there is nothing to finalize.
    EXPLAINABILITY_PROCESSES.clear()  # Prevent duplicate finalization of the same child records.
    verbose_output(f"{BackgroundColors.GREEN}Waiting for {len(tracked_processes)} asynchronous explainability process(es) to finish before shutdown.{Style.RESET_ALL}", config=config)  # Log explainability finalization.
    finalization_errors: List[Exception] = []  # Collect child finalization errors for propagation.
    for process_record in tracked_processes:  # Finalize every tracked child process.
        try:  # Join one child and validate its final status.
            finalize_explainability_process_record(process_record, config=config, terminate=terminate)  # Reap one child process and report failure status.
        except Exception as e:  # Collect child finalization errors.
            finalization_errors.append(e)  # Preserve child finalization error.
            print(str(e))  # Print finalization error through existing logging path.
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Send finalization failure details through Telegram.
            if terminate:  # Do not mask an earlier abnormal parent failure.
                continue  # Continue cleanup for remaining child processes.
    if finalization_errors and not terminate:  # Surface child failures before successful program completion.
        raise RuntimeError("; ".join(str(error) for error in finalization_errors))  # Raise combined child finalization failure.


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


def add_hardware_column(df, columns_order, column_name="Hardware"):
    """
    Ensure a hardware description column named by `column_name` exists on `df`
    and insert it into `columns_order` immediately after `elapsed_time_s`.

    :param df: pandas.DataFrame to mutate in-place
    :param columns_order: list of canonical column names to update
    :param column_name: hardware column name to add (default: "Hardware")
    :return: None
    """
    
    try:
        try:  # Try to get hardware specifications
            hardware_specs = get_hardware_specifications()  # Get system specs
            df[column_name] = (
                hardware_specs["cpu_model"]
                + " | Cores: "
                + str(hardware_specs["cores"])
                + " | RAM: "
                + str(hardware_specs["ram_gb"])
                + " GB | OS: "
                + hardware_specs["os"]
            )  # Add hardware specs column
        except Exception:  # If fetching specs fails
            df[column_name] = None  # Add column with None values

        if column_name not in columns_order:  # If the hardware column is not already in the order list
            insert_idx = (
                (columns_order.index("elapsed_time_s") + 1) if "elapsed_time_s" in columns_order else len(columns_order)
            )  # Determine insertion index
            columns_order.insert(insert_idx, column_name)  # Insert hardware column into the desired position
            
        return df  # Return the modified DataFrame
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def export_feature_artifacts(df, file_path_obj, stacking_dir, config=None):
    """
    Aggregates feature usage statistics, exports the top-features CSV, and generates the heatmap PNG.

    :param df: DataFrame containing stacking results with feature columns
    :param file_path_obj: Path object for the original CSV file used to derive base filenames
    :param stacking_dir: Path object for the Stacking output directory where artifacts are saved
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        try:  # Attempt feature usage aggregation
            feature_counts_df = aggregate_feature_usage(df, None)  # Aggregate feature usage from results DataFrame (None uses config)
        except ValueError as ve:
            print(f"{BackgroundColors.YELLOW}No features to aggregate: {ve}{Style.RESET_ALL}")  # Warn when no feature columns are present
            feature_counts_df = pd.DataFrame()  # Provide empty DataFrame to proceed without crashing
        except Exception as e:
            print(f"{BackgroundColors.RED}Feature aggregation failed: {e}{Style.RESET_ALL}")  # Log aggregation failure for debugging
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Send aggregation error via Telegram
            feature_counts_df = pd.DataFrame()  # Provide empty DataFrame to allow downstream code to continue

        try:  # Attempt to export top-features CSV and heatmap
            path_is_directory = resolve_path_represents_directory(str(file_path_obj))  # Resolve whether artifact identity comes from a directory.
            dataset_base = build_filename_safe_dataset_identity(resolve_canonical_dataset_identity(str(file_path_obj), True)) if path_is_directory else file_path_obj.stem  # Resolve artifact filename identity by path scope.
            top_csv = stacking_dir / f"{dataset_base}_top_features.csv"  # Build path for top-features CSV
            top_png = stacking_dir / f"{dataset_base}_top_features.png"  # Build path for feature usage heatmap
            export_top_features_csv(feature_counts_df, str(top_csv), dataset_file=str(file_path_obj))  # Export aggregated feature counts to CSV
            generate_feature_usage_heatmap(feature_counts_df, str(top_png), dataset_file=str(file_path_obj))  # Generate heatmap PNG of feature usage frequencies
        except Exception as e:
            print(f"{BackgroundColors.RED}Failed to export top-features CSV or heatmap: {e}{Style.RESET_ALL}")  # Log export failure details
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Notify via Telegram but do not interrupt main flow
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_stacking_results_csv_columns(config: Optional[dict] = None) -> List[str]:
    """
    Resolve the configured final stacking results column order.

    :param config: Configuration dictionary, or None to use the global configuration.
    :return: Ordered list of final stacking results column names.
    """

    if config is None:  # Use global configuration when no configuration is provided.
        config = CONFIG  # Assign the global configuration reference.

    configured_columns = config.get("stacking", {}).get("results_csv_columns")  # Read configured final results columns.
    if isinstance(configured_columns, list) and configured_columns:  # Verify that a non-empty configured list is available.
        return list(configured_columns)  # Return a defensive copy of the configured order.

    return list(get_default_stacking_config()["results_csv_columns"])  # Return the default final results order.


def get_cache_results_csv_columns(config: Optional[dict] = None) -> List[str]:
    """
    Resolve the configured temporary cache results column order.

    :param config: Configuration dictionary, or None to use the global configuration.
    :return: Ordered list of temporary cache results column names.
    """

    if config is None:  # Use global configuration when no configuration is provided.
        config = CONFIG  # Assign the global configuration reference.

    configured_columns = config.get("stacking", {}).get("cache_results_csv_columns")  # Read configured temporary cache columns.
    if isinstance(configured_columns, list) and configured_columns:  # Verify that a non-empty configured list is available.
        return list(configured_columns)  # Return a defensive copy of the configured cache order.

    return list(get_default_stacking_config()["cache_results_csv_columns"])  # Return the default temporary cache order.


def resolve_boolean_value(value: Any, default: bool = False) -> bool:
    """
    Normalize scalar boolean-like values to a Python bool.

    :param value: Value to normalize into a boolean.
    :param default: Boolean value returned for missing or unrecognized input.
    :return: Normalized boolean value.
    """

    if isinstance(value, bool):  # Preserve native boolean values.
        return value  # Return the native boolean unchanged.

    if value is None:  # Treat None as missing.
        return default  # Return the supplied default for missing input.

    try:  # Normalize pandas missing scalars without rejecting lists or arrays.
        if bool(cast(Any, pd.isna(value))):  # Verify whether the scalar is missing.
            return default  # Return the supplied default for missing pandas values.
    except Exception:  # Continue when pandas returns a non-scalar missing mask.
        pass  # Preserve legacy tolerance for list-like values.

    if isinstance(value, (int, float)):  # Normalize numeric flags.
        return bool(value)  # Return numeric truthiness.

    if isinstance(value, str):  # Normalize common serialized flags.
        normalized_value = value.strip().lower()  # Normalize whitespace and case.
        if normalized_value in ("true", "1", "yes", "y", "on", "optimized", "optimized hyperparameters"):  # Recognize truthy strings.
            return True  # Return True for recognized truthy strings.
        if normalized_value in ("false", "0", "no", "n", "off", "default", "default hyperparameters", "", "none", "nan", "null"):  # Recognize falsy strings.
            return False  # Return False for recognized falsy strings.

    return default  # Return the default for unrecognized values.


def has_serialized_value(value: Any) -> bool:
    """
    Determine whether a serialized metadata field carries a meaningful value.

    :param value: Serialized or native metadata value to inspect.
    :return: True when the value contains meaningful metadata, otherwise False.
    """

    if value is None:  # Treat None as missing metadata.
        return False  # Return False for missing metadata.

    try:  # Normalize pandas missing scalars without rejecting containers.
        if bool(cast(Any, pd.isna(value))):  # Verify whether the scalar is missing.
            return False  # Return False for missing pandas values.
    except Exception:  # Continue when pandas returns a non-scalar missing mask.
        pass  # Preserve tolerance for list-like metadata values.

    if isinstance(value, str):  # Normalize serialized metadata strings.
        normalized_value = value.strip()  # Remove surrounding whitespace.
        return normalized_value not in ("", "None", "none", "nan", "NaN", "null", "{}", "[]")  # Return whether the string carries data.

    return True  # Treat remaining native values as meaningful metadata.


def normalize_metadata_for_json(value: Any, depth: int = 0) -> Any:
    """
    Normalize metadata values into deterministic JSON-compatible structures.

    :param value: Metadata value to normalize.
    :param depth: Current recursion depth.
    :return: JSON-compatible normalized value.
    """

    if depth > 8:  # Bound recursive estimator and container traversal.
        return f"{value.__class__.__module__}.{value.__class__.__name__}"  # Return deterministic class identity at the depth limit.

    if value is None:  # Preserve null metadata values.
        return None  # Return JSON null.

    try:  # Normalize pandas missing scalars before scalar serialization.
        if bool(cast(Any, pd.isna(value))):  # Verify whether the scalar is missing.
            return None  # Return JSON null for pandas missing scalars.
    except Exception:  # Continue when pandas returns a non-scalar mask.
        pass  # Preserve container and estimator processing.

    if isinstance(value, bool):  # Preserve native booleans before integer handling.
        return value  # Return boolean value unchanged.

    if isinstance(value, int):  # Preserve native integers.
        return value  # Return integer value unchanged.

    if isinstance(value, float):  # Normalize native floats.
        return value if math.isfinite(value) else None  # Return finite floats and map non-finite values to JSON null.

    if isinstance(value, str):  # Preserve strings.
        return value  # Return string value unchanged.

    if isinstance(value, np.generic):  # Normalize NumPy scalar values.
        return normalize_metadata_for_json(value.item(), depth + 1)  # Return normalized Python scalar.

    if isinstance(value, np.ndarray):  # Normalize NumPy arrays.
        return normalize_metadata_for_json(value.tolist(), depth + 1)  # Return normalized nested list.

    if isinstance(value, Path):  # Normalize pathlib paths.
        return str(value)  # Return path text.

    if isinstance(value, (datetime.datetime, datetime.date)):  # Normalize datetime values.
        return value.isoformat()  # Return ISO-8601 text.

    if isinstance(value, dict):  # Normalize mappings with deterministic key order.
        return {str(key): normalize_metadata_for_json(item_value, depth + 1) for key, item_value in sorted(value.items(), key=lambda item: str(item[0]))}  # Return normalized mapping.

    if isinstance(value, (list, tuple)):  # Normalize ordered collections.
        return [normalize_metadata_for_json(item, depth + 1) for item in value]  # Return normalized list preserving order.

    if isinstance(value, set):  # Normalize unordered collections deterministically.
        normalized_items = [normalize_metadata_for_json(item, depth + 1) for item in value]  # Normalize every set item.
        return sorted(normalized_items, key=lambda item: json.dumps(item, sort_keys=True, default=str))  # Return deterministically ordered set values.

    class_name = f"{value.__class__.__module__}.{value.__class__.__name__}"  # Build deterministic object class identity.

    if isinstance(value, type):  # Normalize class objects.
        return f"{value.__module__}.{value.__qualname__}"  # Return deterministic class path.

    if callable(value):  # Normalize callable objects without memory addresses.
        return f"{getattr(value, '__module__', value.__class__.__module__)}.{getattr(value, '__qualname__', value.__class__.__name__)}"  # Return callable identity text.

    if hasattr(value, "get_params"):  # Normalize estimator-like objects through their exposed parameters.
        try:  # Read shallow parameters to avoid recursive estimator expansion loops.
            estimator_params = value.get_params(deep=False)  # Read top-level estimator parameters.
            return {"estimator_class": class_name, "parameters": normalize_metadata_for_json(estimator_params, depth + 1)}  # Return estimator class and normalized parameters.
        except Exception:  # Fall back to class identity when estimator parameters are unavailable.
            return {"estimator_class": class_name}  # Return deterministic estimator class identity.

    try:  # Extract public object attributes when available.
        public_attrs = {key: attr_value for key, attr_value in vars(value).items() if not str(key).startswith("_") and not callable(attr_value)}  # Collect public non-callable attributes.
    except Exception:  # Use class identity for objects without inspectable attributes.
        public_attrs = {}  # Normalize unavailable attributes to an empty mapping.

    if public_attrs:  # Preserve meaningful public object state when present.
        return {"object_class": class_name, "attributes": normalize_metadata_for_json(public_attrs, depth + 1)}  # Return class identity plus normalized attributes.

    return {"object_class": class_name}  # Return deterministic class identity for opaque objects.


def serialize_effective_estimator_parameters(model: Any) -> str:
    """
    Serialize effective estimator parameters into deterministic JSON.

    :param model: Estimator object whose effective parameters should be serialized.
    :return: JSON string containing normalized estimator parameters.
    """

    try:  # Prefer sklearn-style parameter extraction when available.
        params = model.get_params(deep=True) if hasattr(model, "get_params") else {}  # Read effective estimator parameters.
    except Exception:  # Fall back to deterministic model identity when parameter extraction fails.
        params = {"estimator_class": f"{model.__class__.__module__}.{model.__class__.__name__}"}  # Store deterministic estimator class identity.

    normalized_params = normalize_metadata_for_json(params)  # Normalize parameters into JSON-compatible values.
    return json.dumps(normalized_params, sort_keys=True, allow_nan=False)  # Return deterministic valid JSON.


def serialize_result_hyperparameters(model_name: str, hyperparams_map: Optional[dict] = None, effective_hyperparameters: Any = None) -> Optional[str]:
    """
    Serialize row hyperparameters from the evaluated estimator or applied mapping.

    :param model_name: Configured classifier name used for mapping lookup.
    :param hyperparams_map: Optional model-name mapping of applied optimized parameters.
    :param effective_hyperparameters: Optional effective parameters read from the evaluated estimator.
    :return: JSON string with row hyperparameters, or None when no source exists.
    """

    if effective_hyperparameters is not None:  # Prefer parameters read from the estimator that was evaluated.
        if isinstance(effective_hyperparameters, str):  # Preserve pre-serialized estimator parameters.
            return effective_hyperparameters  # Return existing JSON payload.
        normalized_effective = normalize_metadata_for_json(effective_hyperparameters)  # Normalize native estimator parameters.
        return json.dumps(normalized_effective, sort_keys=True, allow_nan=False)  # Return deterministic valid JSON.

    if hyperparams_map and hyperparams_map.get(model_name) is not None:  # Fall back to the applied parameter mapping for legacy callers.
        normalized_mapped = normalize_metadata_for_json(hyperparams_map.get(model_name))  # Normalize mapped parameters before serialization.
        return json.dumps(normalized_mapped, sort_keys=True, allow_nan=False)  # Return deterministic valid JSON.

    return None  # Return no payload when no parameter source exists.


def resolve_persisted_augmentation_ratio(experiment_mode: Any, augmentation_ratio: Any) -> Any:
    """
    Resolve the persisted augmentation ratio for one result row.

    :param experiment_mode: Experiment mode metadata for the row.
    :param augmentation_ratio: Raw augmentation ratio metadata for the row.
    :return: Numeric 0.0 for baseline rows, preserved ratio for augmented rows, or None when unavailable.
    """

    normalized_mode = str(experiment_mode or "").strip().lower()  # Normalize experiment mode text.
    if normalized_mode == "original_only":  # Detect baseline rows explicitly.
        return 0.0  # Persist baseline rows with a numeric zero ratio.

    if not has_serialized_value(augmentation_ratio):  # Preserve missing ratios only for non-baseline rows.
        return None  # Return missing ratio for validation or legacy recovery.

    try:  # Normalize numeric ratio payloads.
        return float(augmentation_ratio)  # Return numeric ratio value.
    except Exception:  # Preserve non-numeric legacy payloads without guessing.
        return augmentation_ratio  # Return original ratio payload.


def resolve_persisted_feature_selection_enabled(feature_set: Any, feature_selection_enabled: Any) -> bool:
    """
    Resolve row-level feature-selection metadata.

    :param feature_set: Effective feature-set label for the evaluated row.
    :param feature_selection_enabled: Raw feature-selection flag from caller or cache.
    :return: False for Full Features rows, otherwise the normalized caller flag.
    """

    if str(feature_set or "").strip().lower() == "full features":  # Detect the all-feature baseline mode.
        return False  # Full Features does not apply a selection strategy.

    return resolve_boolean_value(feature_selection_enabled, False)  # Preserve normalized caller semantics for non-baseline feature sets.


def resolve_persisted_data_augmentation_enabled(experiment_mode: Any, augmentation_ratio: Any, data_augmentation_enabled: Any) -> bool:
    """
    Resolve row-level data-augmentation metadata.

    :param experiment_mode: Experiment mode metadata for the row.
    :param augmentation_ratio: Augmentation ratio metadata for the row.
    :param data_augmentation_enabled: Raw data-augmentation flag from caller or cache.
    :return: Boolean indicating whether generated data was used for the row.
    """

    normalized_mode = str(experiment_mode or "").strip().lower()  # Normalize experiment mode text.
    if normalized_mode == "original_only":  # Detect baseline rows explicitly.
        return False  # Baseline rows do not use generated data.

    if normalized_mode in ("original_plus_augmented", "original_training_augmented_testing"):  # Detect legacy training augmentation and current augmented testing rows.
        return True  # Augmented rows use generated data.

    resolved_ratio = resolve_persisted_augmentation_ratio(experiment_mode, augmentation_ratio)  # Resolve ratio for legacy rows without explicit mode.
    try:  # Interpret numeric legacy ratios when mode metadata is unavailable.
        ratio_active = float(resolved_ratio) > 0.0 if has_serialized_value(resolved_ratio) else False  # Treat only positive numeric ratios as augmented.
    except Exception:  # Fall back to payload presence for non-numeric legacy ratios.
        ratio_active = has_serialized_value(resolved_ratio)  # Treat meaningful legacy ratio payload as augmented.

    return resolve_boolean_value(data_augmentation_enabled, ratio_active)  # Preserve explicit flag when recognized, otherwise use ratio-derived semantics.


def reorder_and_annotate_dataframe(df, config=None):
    """
    Reorders DataFrame columns according to configuration and appends the hardware column.

    :param df: DataFrame containing stacking results to reorder
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Reordered DataFrame with hardware column appended at the proper position
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        column_order = get_stacking_results_csv_columns(config)  # Resolve the final results column order from configuration.
        df = add_hardware_column(df, column_order)  # Populate the hardware column before the final ordering pass.
        existing_columns = [col for col in column_order if col in df.columns]  # Select configured columns that exist in the DataFrame.
        df = df[existing_columns + [c for c in df.columns if c not in existing_columns]]  # Reorder configured columns first and preserve any extra columns after them.

        return df  # Return the reordered and annotated DataFrame
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def flatten_and_serialize_results(results_list):
    """
    Flattens result dictionaries, truncates numeric metrics, and JSON-serializes nested fields.

    :param results_list: List of result dictionaries from classifier evaluation
    :return: List of flattened and serialized row dictionaries ready for DataFrame construction
    """

    try:
        flat_rows = []  # Initialize list to collect flattened rows
        for res in results_list:  # Iterate over each result dictionary
            row = dict(res)  # Create a mutable shallow copy of the result dictionary
            row["augmentation_ratio"] = resolve_persisted_augmentation_ratio(row.get("experiment_mode", "original_only"), row.get("augmentation_ratio", None))  # Normalize baseline ratio metadata before DataFrame construction.
            row["feature_selection_enabled"] = resolve_persisted_feature_selection_enabled(row.get("feature_set", ""), row.get("feature_selection_enabled", False))  # Normalize row-level feature-selection metadata.
            row["data_augmentation_enabled"] = resolve_persisted_data_augmentation_enabled(row.get("experiment_mode", "original_only"), row.get("augmentation_ratio", None), row.get("data_augmentation_enabled", False))  # Normalize row-level data-augmentation metadata.

            for metric in ["accuracy", "precision", "recall", "f1_score", "fpr", "fnr"]:  # Iterate over numeric metric field names
                if metric in row and row[metric] is not None:  # If metric field is present and has a value
                    row[metric] = row[metric]  # Preserve full float precision for classification metrics

            if "features_list" not in row and "top_features" in row:  # Populate features_list from legacy top_features when only the duplicate field exists.
                row["features_list"] = row["top_features"]  # Preserve feature metadata under the canonical field.
            if "features_list" in row and not isinstance(row["features_list"], str):  # If features_list is not yet a JSON string
                row["features_list"] = json.dumps(row["features_list"])  # Serialize features list to JSON string
            if "top_features" in row and not isinstance(row["top_features"], str):  # If top_features is not yet a JSON string
                row["top_features"] = json.dumps(row["top_features"])  # Serialize top features to JSON string
            if "rfe_ranking" in row and row["rfe_ranking"] is not None and not isinstance(row["rfe_ranking"], str):  # If rfe_ranking is present and not yet serialized
                row["rfe_ranking"] = json.dumps(row["rfe_ranking"])  # Serialize RFE ranking to JSON string
            if "hyperparameters" in row and row["hyperparameters"] is not None and not isinstance(row["hyperparameters"], str):  # If hyperparameters is present and not yet serialized
                row["hyperparameters"] = json.dumps(normalize_metadata_for_json(row["hyperparameters"]), sort_keys=True, allow_nan=False)  # Serialize hyperparameters as deterministic valid JSON.

            if "hyperparameter_mode" not in row or not has_serialized_value(row["hyperparameter_mode"]):  # Populate explicit HP mode when the row does not carry one.
                row["hyperparameter_mode"] = "Optimized Hyperparameters" if resolve_boolean_value(row.get("hyperparameters_enabled"), False) else "Default Hyperparameters"  # Derive HP mode from the semantic HP flag only.
            row["feature_selection_enabled"] = resolve_persisted_feature_selection_enabled(row.get("feature_set", ""), row.get("feature_selection_enabled", False))  # Reapply feature-selection semantics after legacy defaults.
            row["hyperparameters_enabled"] = row["hyperparameter_mode"] == "Optimized Hyperparameters"  # Keep HP boolean aligned with explicit HP mode
            row["data_augmentation_enabled"] = resolve_persisted_data_augmentation_enabled(row.get("experiment_mode", "original_only"), row.get("augmentation_ratio", None), row.get("data_augmentation_enabled", False))  # Reapply data-augmentation semantics after legacy defaults.

            flat_rows.append(row)  # Append the processed row to the flat list
        return flat_rows  # Return the list of flattened and serialized rows
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def save_stacking_results(csv_path, results_list, config=None):
    """Save the stacking results to CSV file located in same dataset Feature_Analysis directory.

    Writes richer metadata fields matching RFE outputs: model, dataset, accuracy, precision,
    recall, f1_score, fpr, fnr, elapsed_time_s, cv_method, top_features, rfe_ranking,
    hyperparameters, features_list and Hardware.
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Preparing to save {BackgroundColors.CYAN}{len(results_list)}{BackgroundColors.GREEN} stacking results to CSV...{Style.RESET_ALL}",
            config=config
        )

        if not results_list:
            print(f"{BackgroundColors.YELLOW}Warning: No results provided to save.{Style.RESET_ALL}")
            return

        results_filename = config.get("stacking", {}).get("results_filename", "Stacking_Classifiers_Results.csv")  # Get results filename from config
        file_path_obj = Path(csv_path)
        dataset_root = resolve_dataset_root_path(str(csv_path))  # Resolve output root from file or directory input.
        feature_analysis_dir = dataset_root / "Feature_Analysis"  # Build Feature_Analysis path from the dataset root.
        os.makedirs(feature_analysis_dir, exist_ok=True)
        
        stacking_results_dir = config.get("stacking", {}).get("results_dir", "Stacking")
        stacking_dir = dataset_root / stacking_results_dir  # Build stacking output path from the dataset root.
        os.makedirs(stacking_dir, exist_ok=True)
        output_path = stacking_dir / results_filename

        flat_rows = flatten_and_serialize_results(results_list)  # Flatten and serialize all result rows into plain dicts

        df = pd.DataFrame(flat_rows)  # Construct results DataFrame from flattened rows

        df = reorder_and_annotate_dataframe(df, config=config)  # Reorder columns by config order and append hardware annotation

        try:
            write_memory_phase_event("final_export", config=config, dataset_source=csv_path, output_path=str(output_path), row_count=len(df), event_outcome="starting")  # Publish final export start
            generate_csv_and_image(df, str(output_path), is_visualizable=True, index=False, encoding="utf-8")  # Persist results CSV and generate PNG
            write_memory_phase_event("final_export", config=config, dataset_source=csv_path, output_path=str(output_path), row_count=len(df), event_outcome="saved")  # Publish final export completion
            print(
                f"\n{BackgroundColors.GREEN}Stacking classifier results successfully saved to {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
            )  # Notify user of success

            export_feature_artifacts(df, file_path_obj, stacking_dir, config=config)  # Export feature usage CSV and heatmap to the Stacking directory
        except Exception as e:
            write_memory_phase_event("final_export", config=config, dataset_source=csv_path, output_path=str(output_path), row_count=len(df) if 'df' in locals() else None, event_outcome=f"failed:{e}")  # Publish final export failure
            print(
                f"{BackgroundColors.RED}Failed to write Stacking Classifier CSV to {BackgroundColors.CYAN}{output_path}{BackgroundColors.RED}: {e}{Style.RESET_ALL}"
            )
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_cache_file_path(csv_path, config=None):
    """
    Generate the cache file path for a given dataset CSV path.

    :param csv_path: Path to the dataset CSV file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Path to the cache file
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Generating cache file path for: {BackgroundColors.CYAN}{csv_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        cache_prefix = config.get("stacking", {}).get("cache_prefix", "CACHE_")  # Get cache prefix from config
        cache_results_subdir = config.get("stacking", {}).get("cache_results_subdir", "Cache_Results")  # Get cache results subdirectory from config
        path_is_directory = resolve_path_represents_directory(str(csv_path))  # Resolve whether cache identity comes from a directory.
        if path_is_directory:  # Use directory identity for combined-files cache names.
            dataset_identity = resolve_canonical_dataset_identity(str(csv_path), True)  # Resolve canonical directory identity.
            dataset_name = build_filename_safe_dataset_identity(dataset_identity)  # Build filename-safe directory identity.
            cache_filename_prefix = f"{cache_prefix.rstrip('_-')}-"  # Use hyphen delimiter for path-derived cache identity.
        else:  # Preserve single-file cache naming behavior.
            dataset_name = os.path.splitext(os.path.basename(csv_path))[0]  # Get base dataset name.
            cache_filename_prefix = cache_prefix  # Preserve configured single-file cache prefix.
        stacking_output_dir = get_stacking_output_dir(csv_path, config)  # Get stacking results directory from config-aware path resolver
        output_dir = os.path.join(stacking_output_dir, cache_results_subdir)  # Build cache directory inside the stacking results directory

        validate_output_path(stacking_output_dir, str(Path(output_dir).resolve()))  # Verify cache directory is within the stacking results directory to prevent directory traversal
        os.makedirs(output_dir, exist_ok=True)  # Ensure the cache directory exists
        cache_filename = f"{cache_filename_prefix}{dataset_name}-Stacking_Classifiers_Results.csv"  # Build cache filename.
        cache_path = os.path.join(output_dir, cache_filename)  # Full cache file path

        return cache_path  # Return the cache file path
    except Exception as e:  # Catch any unexpected exceptions during cache path generation
        print(str(e))  # Log the error message for debugging
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send the exception details via Telegram for monitoring
        raise  # Re-raise the exception to allow upstream handling if necessary


def safe_load_json(val):
    """
    Safely load a JSON string if it's a string, otherwise return the value as is.
    
    :param val: The value to load as JSON if it's a string
    :return: The loaded JSON object if val is a string, None if val is Na
    """
    
    if pd.isna(val):  # Verify value is NaN before returning None
        return None  # Return None for NaN values
    
    if isinstance(val, str):  # If the value is a string, attempt to parse it as JSON
        try:  # Try to load the string as JSON
            return json.loads(val)  # Return the loaded JSON object
        except Exception:  # If parsing fails, return the original string value
            return val  # Return the original value if it's a string but not valid JSON
    
    return val  # For non-string values, return as is
            
                
def load_cache_results(csv_path, config=None, notify_discovery: bool = True):  # Load cache rows with optional operator discovery notification.
    """
    Load cached results from the cache file if it exists.

    :param csv_path: Path to the dataset CSV file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary mapping full resume cache key tuple to result entry
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        cache_path = get_cache_file_path(csv_path, config=config)  # Get the cache file path

        if not verify_filepath_exists(cache_path):  # If cache file doesn't exist
            verbose_output(
                f"{BackgroundColors.YELLOW}No cache file found at: {BackgroundColors.CYAN}{cache_path}{Style.RESET_ALL}",
                config=config
            )  # Output the verbose message
            return {}  # Return empty dictionary

        if notify_discovery:  # Emit discovery notifications only for operator-facing resume loads.
            print(f"{BackgroundColors.GREEN}Resume cache file found at: {BackgroundColors.CYAN}{cache_path}{Style.RESET_ALL}")  # Print cache discovery and exact location when requested.

        verbose_output(
            f"{BackgroundColors.GREEN}Loading cached results from: {BackgroundColors.CYAN}{cache_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        try:  # Try to load the cache file
            low_memory = config.get("execution", {}).get("low_memory", False)  # Read low memory flag from config
            df_cache = pd.read_csv(cache_path, low_memory=low_memory)  # Read the cache file
            df_cache = prepare_cache_dataframe(df_cache, config=config)  # Normalize legacy and current cache schemas before resume use
            cache_dict = {}  # Initialize cache dictionary

            for _, row in df_cache.iterrows():  # Iterate through each row
                def cache_row_value(column_name, default=None):
                    value = row.get(column_name, default)  # Retrieve a scalar value from the cache row
                    return default if value is None or bool(cast(Any, pd.isna(value))) else value  # Normalize missing pandas scalars to the provided default

                feature_set = str(cache_row_value("feature_set", ""))  # Get feature set name
                model_name = str(cache_row_value("model_name", ""))  # Get model name
                execution_mode_row = str(cache_row_value("execution_mode", "separate_files"))  # Get execution mode from cached row
                data_source_row = str(cache_row_value("data_source", "Original"))  # Get data source label from cached row
                experiment_mode_row = str(cache_row_value("experiment_mode", "original_only"))  # Get experiment mode from cached row
                aug_ratio_value = cache_row_value("augmentation_ratio", None)  # Retrieve augmentation ratio when present
                aug_ratio_row = resolve_persisted_augmentation_ratio(experiment_mode_row, aug_ratio_value)  # Normalize augmentation ratio from cached row
                attack_types_raw_row = safe_load_json(cache_row_value("attack_types_combined", None))  # Load attack types from cached row
                attack_types_list_row = attack_types_raw_row if isinstance(attack_types_raw_row, list) else None  # Normalize attack types to list or None
                hyperparameter_mode_row = str(cache_row_value("hyperparameter_mode", "Default Hyperparameters"))  # Recover explicit hyperparameter mode
                hyperparameters_enabled_row = resolve_boolean_value(cache_row_value("hyperparameters_enabled", False), hyperparameter_mode_row == "Optimized Hyperparameters")  # Normalize cached hyperparameter mode to the boolean used by resume keys
                cache_key = build_resume_cache_key(execution_mode_row, data_source_row, experiment_mode_row, aug_ratio_row, attack_types_list_row, feature_set, model_name, hyperparameters_enabled_row)  # Build full resume cache key from all distinguishing dimensions

                result_entry = {
                    "model": cache_row_value("model", ""),
                    "dataset": cache_row_value("dataset", ""),
                    "execution_mode": execution_mode_row,
                    "attack_types_combined": cache_row_value("attack_types_combined", None),
                    "feature_set": feature_set,
                    "hyperparameter_mode": hyperparameter_mode_row,
                    "classifier_type": cache_row_value("classifier_type", ""),
                    "model_name": model_name,
                    "data_source": data_source_row,
                    "experiment_id": cache_row_value("experiment_id", None),
                    "experiment_mode": experiment_mode_row,
                    "augmentation_ratio": aug_ratio_row,
                    "feature_selection_enabled": resolve_persisted_feature_selection_enabled(feature_set, cache_row_value("feature_selection_enabled", False)),  # Recover normalized FS flag from cache metadata
                    "hyperparameters_enabled": hyperparameters_enabled_row,  # Recover normalized HP flag from cache metadata
                    "data_augmentation_enabled": resolve_persisted_data_augmentation_enabled(experiment_mode_row, aug_ratio_row, cache_row_value("data_augmentation_enabled", False)),  # Recover normalized DA flag from cache metadata
                    "n_features": int(n_features_value) if (n_features_value := cache_row_value("n_features", None)) is not None else None,
                    "n_samples_train": int(n_samples_train_value) if (n_samples_train_value := cache_row_value("n_samples_train", None)) is not None else None,
                    "n_samples_test": int(n_samples_test_value) if (n_samples_test_value := cache_row_value("n_samples_test", None)) is not None else None,
                    "accuracy": float(accuracy_value) if (accuracy_value := cache_row_value("accuracy", None)) is not None else None,
                    "precision": float(precision_value) if (precision_value := cache_row_value("precision", None)) is not None else None,
                    "recall": float(recall_value) if (recall_value := cache_row_value("recall", None)) is not None else None,
                    "f1_score": float(f1_score_value) if (f1_score_value := cache_row_value("f1_score", None)) is not None else None,
                    "fpr": float(fpr_value) if (fpr_value := cache_row_value("fpr", None)) is not None else None,
                    "fnr": float(fnr_value) if (fnr_value := cache_row_value("fnr", None)) is not None else None,
                    "elapsed_time_s": float(elapsed_time_value) if (elapsed_time_value := cache_row_value("elapsed_time_s", None)) is not None else None,
                    "cv_method": cache_row_value("cv_method", None),
                    "top_features": safe_load_json(cache_row_value("features_list", None)),  # Rebuild duplicate final-export field from canonical cache feature metadata
                    "rfe_ranking": safe_load_json(cache_row_value("rfe_ranking", None)),
                    "hyperparameters": safe_load_json(cache_row_value("hyperparameters", None)),
                    "features_list": safe_load_json(cache_row_value("features_list", None)),
                    "Hardware": cache_row_value("Hardware", None),
                }

                cache_dict[cache_key] = result_entry

            print(f"{BackgroundColors.GREEN}Loaded cached results from: {BackgroundColors.CYAN}{cache_path}{Style.RESET_ALL}")
            return cache_dict

        except Exception as e:  # Catch any errors reading the cache file
            print(
                f"{BackgroundColors.YELLOW}Warning: Failed to load from cache {BackgroundColors.CYAN}{cache_path}{BackgroundColors.YELLOW}: {e}{Style.RESET_ALL}"
            )  # Print warning message about cache read failure
            return {}  # Return empty dict to signal no cached results are available
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def remove_cache_file(csv_path, config=None):
    """
    Remove the cache file after successful completion.

    :param csv_path: Path to the dataset CSV file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        cache_path = get_cache_file_path(csv_path, config=config)  # Get the cache file path

        if verify_filepath_exists(cache_path):  # If cache file exists
            try:  # Try to remove the cache file
                os.remove(cache_path)  # Delete the cache file
                print(
                    f"{BackgroundColors.GREEN}Cache file removed: {BackgroundColors.CYAN}{cache_path}{Style.RESET_ALL}"
                )  # Print success message
            except Exception as e:  # Catch any errors
                print(
                    f"{BackgroundColors.YELLOW}Warning: Failed to remove cache file {BackgroundColors.CYAN}{cache_path}{BackgroundColors.YELLOW}: {e}{Style.RESET_ALL}"
                )  # Print warning message
        else:  # If cache file doesn't exist
            verbose_output(
                f"{BackgroundColors.YELLOW}No cache file to remove at: {BackgroundColors.CYAN}{cache_path}{Style.RESET_ALL}"
            )  # Output verbose message
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_resume_cache_key(execution_mode_str: str, data_source_label: str, experiment_mode: str, augmentation_ratio, attack_types_combined, feature_set: str, model_name: str, hyperparameters_enabled: bool = False) -> tuple:
    """
    Build a deterministic, collision-free resume cache key for one evaluation unit.

    :param execution_mode_str: Execution mode string (e.g., 'separate_files' or 'combined_files')
    :param data_source_label: Data source label (e.g., 'Original' or 'Augmented@50%')
    :param experiment_mode: Experiment mode string (e.g., 'original_only' or 'original_training_augmented_testing')
    :param augmentation_ratio: Augmentation ratio float or None for original-only experiments
    :param attack_types_combined: List of attack type strings for combined files mode or None
    :param feature_set: Feature set name string (e.g., 'Full Features' or 'GA Features')
    :param model_name: Model name string (e.g., 'Random Forest' or 'StackingClassifier')
    :return: Hashable tuple uniquely identifying the evaluation unit across all dimensions
    """

    try:  # Build a deterministic cache key tuple from all relevant dimensions, using JSON serialization for complex fields to ensure hashability and uniqueness
        attack_key = json.dumps(sorted(str(a) for a in attack_types_combined), sort_keys=True) if attack_types_combined else "None"  # Serialize attack types deterministically or use sentinel string
        resolved_ratio = resolve_persisted_augmentation_ratio(experiment_mode, augmentation_ratio)  # Normalize baseline ratios before identity construction.
        ratio_key = str(resolved_ratio) if resolved_ratio is not None else "None"  # Serialize augmentation ratio as string or use sentinel
        hyperparameter_mode_key = "optimized" if hyperparameters_enabled else "default"  # Distinguish default and optimized evaluations so resume never crosses HP modes
        return (execution_mode_str, data_source_label, experiment_mode, ratio_key, attack_key, hyperparameter_mode_key, feature_set, model_name)  # Return tuple covering all dimensions that uniquely identify an evaluation unit
    except Exception as e:  # Catch any unexpected errors in key construction
        print(str(e))  # Log the error message for debugging
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send the exception details via Telegram for monitoring
        raise  # Re-raise the exception to allow upstream handling if necessary


def resolve_hyperparameter_mode_from_row(row: Any) -> str:
    """
    Resolve the explicit hyperparameter mode for a cache row.

    :param row: Mapping-like row containing cache metadata.
    :return: Explicit hyperparameter mode label.
    """

    mode_value = row.get("hyperparameter_mode", None)  # Read explicit HP mode when present.
    if has_serialized_value(mode_value):  # Use explicit HP mode when it carries a value.
        return str(mode_value).strip()  # Return normalized explicit HP mode text.

    enabled_value = row.get("hyperparameters_enabled", False)  # Read legacy HP boolean flag.
    enabled = resolve_boolean_value(enabled_value, False)  # Derive optimized mode from the semantic HP flag only.

    return "Optimized Hyperparameters" if enabled else "Default Hyperparameters"  # Return the resolved HP mode label.


def build_cache_identity_from_row(row: Any) -> tuple:
    """
    Build the resume identity tuple for a normalized cache row.

    :param row: Mapping-like row containing cache metadata.
    :return: Hashable resume identity tuple.
    """

    attack_types_raw = safe_load_json(row.get("attack_types_combined", None))  # Decode serialized attack type metadata.
    attack_types_list = attack_types_raw if isinstance(attack_types_raw, list) else None  # Normalize attack type metadata to a list or None.
    augmentation_ratio_value = row.get("augmentation_ratio", None)  # Read augmentation ratio metadata.
    augmentation_ratio = resolve_persisted_augmentation_ratio(row.get("experiment_mode", "original_only"), augmentation_ratio_value)  # Normalize augmentation ratio for the resume key.
    hyperparameter_mode = resolve_hyperparameter_mode_from_row(row)  # Resolve explicit HP mode.
    hyperparameters_enabled = hyperparameter_mode == "Optimized Hyperparameters"  # Convert HP mode to the boolean identity dimension.

    return build_resume_cache_key(
        str(row.get("execution_mode", "separate_files")),
        str(row.get("data_source", "Original")),
        str(row.get("experiment_mode", "original_only")),
        augmentation_ratio,
        attack_types_list,
        str(row.get("feature_set", "")),
        str(row.get("model_name", "")),
        hyperparameters_enabled,
    )  # Return the existing resume identity tuple.


def normalize_cache_dataframe(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """
    Normalize cache rows from current and legacy temporary cache schemas.

    :param df: Cache DataFrame to normalize.
    :param config: Configuration dictionary, or None to use the global configuration.
    :return: Cache DataFrame using the canonical temporary cache schema.
    """

    if config is None:  # Use global configuration when no configuration is provided.
        config = CONFIG  # Assign the global configuration reference.

    normalized_df = df.copy()  # Work on a copy to avoid mutating caller-owned DataFrames.
    normalized_df.columns = normalized_df.columns.str.strip()  # Normalize column names before schema migration.
    cache_columns = get_cache_results_csv_columns(config)  # Resolve canonical cache column order.
    methods_cfg = config.get("stacking", {}).get("methods", {})  # Read active stacking method toggles.
    feature_selection_default = bool(methods_cfg.get("feature_selection", True))  # Resolve default FS flag for current execution context.

    if "features_list" not in normalized_df.columns and "top_features" in normalized_df.columns:  # Migrate duplicate legacy feature metadata.
        normalized_df["features_list"] = normalized_df["top_features"]  # Store legacy top_features payload under the canonical field.
    elif "features_list" in normalized_df.columns and "top_features" in normalized_df.columns:  # Fill missing canonical feature metadata from legacy duplicate payloads.
        missing_features_mask = normalized_df["features_list"].isna()  # Locate rows with missing canonical feature metadata.
        normalized_df.loc[missing_features_mask, "features_list"] = normalized_df.loc[missing_features_mask, "top_features"]  # Fill canonical feature metadata from legacy duplicate data.

    if "hyperparameter_mode" not in normalized_df.columns:  # Add explicit HP mode column for legacy cache files.
        normalized_df["hyperparameter_mode"] = None  # Initialize HP mode before row-wise resolution.

    if "hyperparameters_enabled" not in normalized_df.columns:  # Add HP boolean column for legacy or partial rows.
        normalized_df["hyperparameters_enabled"] = False  # Initialize HP boolean before row-wise resolution.

    if "hyperparameters" not in normalized_df.columns:  # Add HP metadata column when absent.
        normalized_df["hyperparameters"] = None  # Preserve canonical schema when no HP metadata exists.

    if "augmentation_ratio" not in normalized_df.columns:  # Add augmentation ratio when absent.
        normalized_df["augmentation_ratio"] = None  # Preserve canonical schema when no augmentation ratio exists.

    if "feature_selection_enabled" not in normalized_df.columns:  # Add FS flag when absent.
        normalized_df["feature_selection_enabled"] = feature_selection_default  # Initialize FS flag from current method context.

    if "data_augmentation_enabled" not in normalized_df.columns:  # Add DA flag when absent.
        normalized_df["data_augmentation_enabled"] = False  # Initialize DA flag before row-level resolution.

    normalized_df["hyperparameter_mode"] = normalized_df.apply(resolve_hyperparameter_mode_from_row, axis=1)  # Resolve HP mode deterministically for every row.
    normalized_df["hyperparameters_enabled"] = normalized_df["hyperparameter_mode"].map(lambda value: value == "Optimized Hyperparameters")  # Keep HP boolean synchronized with HP mode.
    normalized_df["augmentation_ratio"] = normalized_df.apply(lambda row: resolve_persisted_augmentation_ratio(row.get("experiment_mode", "original_only"), row.get("augmentation_ratio", None)), axis=1)  # Normalize baseline and augmented ratio metadata.
    normalized_df["feature_selection_enabled"] = normalized_df.apply(lambda row: resolve_persisted_feature_selection_enabled(row.get("feature_set", ""), row.get("feature_selection_enabled", feature_selection_default)), axis=1)  # Normalize row-level FS metadata.
    normalized_df["data_augmentation_enabled"] = normalized_df.apply(lambda row: resolve_persisted_data_augmentation_enabled(row.get("experiment_mode", "original_only"), row.get("augmentation_ratio", None), row.get("data_augmentation_enabled", False)), axis=1)  # Normalize row-level DA metadata.

    for column in cache_columns:  # Ensure every canonical cache column exists.
        if column not in normalized_df.columns:  # Add missing canonical cache columns.
            normalized_df[column] = None  # Initialize missing canonical values with None.

    normalized_df = normalized_df[cache_columns]  # Apply canonical cache column order and drop duplicate legacy-only fields.

    return normalized_df  # Return normalized cache rows.


def deduplicate_cache_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate cache rows using the resume identity tuple.

    :param df: Normalized cache DataFrame.
    :return: Deduplicated cache DataFrame.
    """

    if df.empty:  # Return immediately when no cache rows exist.
        return df  # Preserve empty DataFrame schema.

    deduped_df = df.copy()  # Work on a copy before adding the transient identity column.
    deduped_df["cache_identity"] = deduped_df.apply(build_cache_identity_from_row, axis=1)  # Build one identity per cache row.
    deduped_df = deduped_df.drop_duplicates(subset=["cache_identity"], keep="last")  # Keep the most recent row for each resume identity.
    deduped_df = deduped_df.drop(columns=["cache_identity"])  # Remove transient identity column before writing.

    return deduped_df  # Return deduplicated cache rows.


def prepare_cache_dataframe(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """
    Normalize, deduplicate, and order temporary cache rows.

    :param df: Cache DataFrame to prepare for reading or writing.
    :param config: Configuration dictionary, or None to use the global configuration.
    :return: Prepared cache DataFrame using the canonical schema.
    """

    normalized_df = normalize_cache_dataframe(df, config=config)  # Normalize current and legacy cache schemas.
    deduped_df = deduplicate_cache_dataframe(normalized_df)  # Remove duplicate rows by resume identity.
    ordered_df = deduped_df[get_cache_results_csv_columns(config)]  # Enforce canonical cache column order.

    return ordered_df  # Return prepared cache rows.


def validate_cache_result_payload(result_entry: dict, config: Optional[dict] = None) -> None:  # Validate one cache row before any disk write.
    """
    Validate that one completed atomic result can form a complete canonical cache row.

    :param result_entry: Fully computed classifier result entry.
    :param config: Configuration dictionary, or None to use the global configuration.
    :return: None.
    """

    if config is None:  # Use global configuration when no configuration is provided.
        config = CONFIG  # Assign the global configuration reference.

    flat_rows = flatten_and_serialize_results([result_entry])  # Serialize the atomic result through the production result flattener.
    if not flat_rows:  # Reject empty serialization before touching the cache file.
        raise ValueError("Cache result serialization produced no rows")  # Raise a persistence-blocking validation error.

    row_df = prepare_cache_dataframe(pd.DataFrame([flat_rows[0]]), config=config)  # Normalize the row through the production cache schema.
    cache_columns = get_cache_results_csv_columns(config)  # Resolve the canonical temporary cache column order.
    missing_columns = [column for column in cache_columns if column not in row_df.columns]  # Locate missing canonical cache columns.
    if missing_columns:  # Reject rows that cannot satisfy the configured cache schema.
        raise ValueError(f"Cache result row missing required columns: {missing_columns}")  # Raise with precise missing column names.

    row_values = row_df.iloc[0].to_dict()  # Convert the normalized row to scalar metadata for required-field validation.
    required_payload_columns = [  # List the fields required to identify and recover a completed atomic classifier result.
        "experiment_id",  # Require the existing experiment identifier for traceability.
        "experiment_mode",  # Require original-only versus augmented mode.
        "execution_mode",  # Require separate-files versus combined-files mode.
        "data_source",  # Require the source label that separates original and augmented rows.
        "dataset",  # Require dataset identity metadata in the persisted payload.
        "augmentation_ratio",  # Require explicit numeric baseline or augmented ratio metadata.
        "feature_set",  # Require feature-set identity.
        "hyperparameter_mode",  # Require default versus optimized hyperparameter identity.
        "classifier_type",  # Require individual versus stacking classifier grouping.
        "model_name",  # Require classifier identity.
        "model",  # Require concrete estimator class metadata.
        "n_features",  # Require feature-count metadata.
        "n_samples_train",  # Require training sample count.
        "n_samples_test",  # Require test sample count.
        "accuracy",  # Require completed accuracy metric.
        "precision",  # Require completed precision metric.
        "recall",  # Require completed recall metric.
        "f1_score",  # Require completed F1 metric.
        "fpr",  # Require completed false-positive-rate metric.
        "fnr",  # Require completed false-negative-rate metric.
        "elapsed_time_s",  # Require elapsed-time metric.
        "cv_method",  # Require evaluation method metadata.
        "hyperparameters",  # Require effective estimator parameter metadata.
        "features_list",  # Require complete feature payload for resume and exports.
    ]  # Complete required payload list.
    missing_values = [column for column in required_payload_columns if not has_serialized_value(row_values.get(column, None))]  # Locate required fields without meaningful values.
    if missing_values:  # Reject incomplete result rows before disk persistence.
        raise ValueError(f"Cache result row missing required values: {missing_values}")  # Raise with precise missing value names.

    if str(row_values.get("execution_mode", "")) == "combined_files" and not has_serialized_value(row_values.get("attack_types_combined", None)):  # Require combined attack scope when combined-files mode is active.
        raise ValueError("Cache result row missing combined attack scope")  # Raise before writing an ambiguous combined-files row.
    if str(row_values.get("experiment_mode", "")) in ("original_plus_augmented", "original_training_augmented_testing") and not has_serialized_value(row_values.get("augmentation_ratio", None)):  # Require the exact ratio for augmented rows.
        raise ValueError("Cache result row missing augmentation ratio")  # Raise before writing an ambiguous augmented row.


def sync_cache_file_data(cache_file: Any) -> None:  # Synchronize temp-file contents where the platform supports it.
    """
    Flush and synchronize a temporary cache file before replacement.

    :param cache_file: Open writable file object.
    :return: None.
    """

    cache_file.flush()  # Flush Python buffers before the rename boundary.
    sync_function = getattr(os, "fsync", None)  # Resolve the filesystem sync primitive when available.
    if sync_function is None:  # Allow platforms without fsync to rely on close semantics.
        return  # Return after the explicit flush.
    sync_function(cache_file.fileno())  # Synchronize temp-file bytes before replacement.


def sync_cache_parent_directory(cache_path: str) -> None:  # Synchronize rename metadata where the platform supports directory descriptors.
    """
    Synchronize the parent directory after an atomic cache replacement when supported.

    :param cache_path: Final cache CSV path.
    :return: None.
    """

    sync_function = getattr(os, "fsync", None)  # Resolve the filesystem sync primitive when available.
    if sync_function is None or os.name == "nt":  # Use a guarded no-op on platforms without portable directory sync.
        return  # Return after the file replacement has completed.

    directory_fd = None  # Store the parent directory descriptor for cleanup.
    try:  # Attempt POSIX directory metadata synchronization.
        directory_fd = os.open(os.path.dirname(os.path.abspath(cache_path)), os.O_RDONLY)  # Open the parent directory for metadata sync.
        sync_function(directory_fd)  # Synchronize the directory entry created by os.replace.
    except OSError:  # Treat unsupported directory synchronization as nonfatal portability behavior.
        return  # Preserve the successful file replacement when directory sync is unsupported.
    finally:  # Close the directory descriptor when one was opened.
        if directory_fd is not None:  # Verify that the descriptor exists before closing.
            os.close(directory_fd)  # Close the parent directory descriptor.


def verify_cache_result_persisted(cache_ref_file: str, resume_key: tuple, config: Optional[dict] = None) -> None:  # Verify the real resume loader can discover the written identity.
    """
    Reload the cache through the production resume path and verify the written identity.

    :param cache_ref_file: Dataset file path used to derive the cache file location.
    :param resume_key: Resume identity expected after persistence.
    :param config: Configuration dictionary, or None to use the global configuration.
    :return: None.
    """

    if config is None:  # Use global configuration when no configuration is provided.
        config = CONFIG  # Assign the global configuration reference.

    persisted_cache = load_cache_results(cache_ref_file, config=config, notify_discovery=False)  # Reload through the production cache loader used by resume without operator notification.
    if resume_key not in persisted_cache:  # Reject persistence that cannot be recovered by the real loader.
        raise RuntimeError(f"Persisted cache result is not recoverable for identity: {resume_key}")  # Raise a persistence-blocking recovery error.


def persist_cache_result_entry(cache_ref_file: Optional[str], result_entry: dict, cache_dict: Optional[dict], config: Optional[dict] = None) -> None:
    """
    Persist one atomic result and register it in the in-memory resume cache.

    :param cache_ref_file: Dataset file path used to derive the cache file location, or None to skip persistence.
    :param result_entry: Fully computed classifier result entry.
    :param cache_dict: Mutable in-memory resume cache keyed by resume identity, or None.
    :param config: Configuration dictionary, or None to use the global configuration.
    :return: None.
    """

    if cache_ref_file is None:  # Skip persistence when no cache reference is available.
        return  # Return without writing.

    if config is None:  # Use global configuration when no configuration is provided.
        config = CONFIG  # Assign the global configuration reference.

    methods_cfg = config.get("stacking", {}).get("methods", {})  # Read active method toggles for cache metadata.
    result_entry["augmentation_ratio"] = resolve_persisted_augmentation_ratio(result_entry.get("experiment_mode", "original_only"), result_entry.get("augmentation_ratio", None))  # Persist explicit baseline or augmented ratio metadata.
    result_entry["feature_selection_enabled"] = resolve_persisted_feature_selection_enabled(result_entry.get("feature_set", ""), methods_cfg.get("feature_selection", True))  # Persist row-level FS context with the temporary row.
    result_entry["hyperparameters_enabled"] = result_entry.get("hyperparameter_mode") == "Optimized Hyperparameters"  # Persist HP boolean derived from explicit HP mode.
    result_entry["data_augmentation_enabled"] = resolve_persisted_data_augmentation_enabled(result_entry.get("experiment_mode", "original_only"), result_entry.get("augmentation_ratio", None), result_entry.get("data_augmentation_enabled", False))  # Persist row-level DA context.
    validate_cache_result_payload(result_entry, config=config)  # Validate the complete normalized cache payload before writing.
    resume_key = build_cache_identity_from_row(result_entry)  # Build the same identity used during resume loading.

    if cache_dict is not None and resume_key in cache_dict:  # Avoid duplicate writes when the same identity is already registered in memory.
        return  # Return without appending a duplicate cache row.

    try:  # Persist and verify the row before any in-memory cache registration.
        save_cache_result_entry(cache_ref_file, result_entry, config=config)  # Persist the fully computed atomic result immediately.
        verify_cache_result_persisted(cache_ref_file, resume_key, config=config)  # Verify the real resume loader can recover the exact identity.
    except Exception as e:  # If cache persistence or recovery verification fails.
        print(f"{BackgroundColors.RED}Failed to persist cache result for identity {resume_key}: {e}{Style.RESET_ALL}")  # Log the affected atomic identity clearly.
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Notify through the existing exception reporting path.
        raise  # Propagate the failure so later combinations do not proceed as cached.

    if cache_dict is not None:  # Register successful writes for the remainder of the current process.
        cache_dict[resume_key] = result_entry  # Store the result under its resume identity.


def save_cache_result_entry(csv_path: str, result_entry: dict, config=None) -> None:
    """
    Atomically append a single result entry to the cache CSV file.

    Uses a temp-file and rename strategy for atomicity on POSIX systems. If the
    cache file does not exist, a new file with a header row is created. If it
    already exists, the entry is appended as a new row without repeating the header.

    :param csv_path: Path to the dataset CSV file used to derive the cache file path
    :param result_entry: Dictionary containing the classifier evaluation result to persist
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        cache_path = get_cache_file_path(csv_path, config=config)  # Derive cache file path from the dataset file path

        flat_rows = flatten_and_serialize_results([result_entry])  # Flatten and serialize entry using same pipeline as final results
        if not flat_rows:  # If serialization produced no rows
            return  # Exit without writing anything

        row_dict = flat_rows[0]  # Extract the single flattened row dictionary
        row_df = prepare_cache_dataframe(pd.DataFrame([row_dict]), config=config)  # Normalize the new row to the canonical cache schema

        try:  # Attempt atomic write to cache file
            cache_dir = os.path.dirname(cache_path)  # Get directory containing the cache file
            os.makedirs(cache_dir, exist_ok=True)  # Ensure cache directory exists before writing
            cache_file_exists = os.path.isfile(cache_path)  # Verify if cache file already exists to decide whether to write header
            if cache_file_exists:  # Read and normalize existing cache rows before appending
                low_memory = config.get("execution", {}).get("low_memory", False)  # Read low memory flag from config
                existing_df = pd.read_csv(cache_path, low_memory=low_memory)  # Load existing cache rows for schema migration
                combined_df = pd.concat([existing_df, row_df], ignore_index=True)  # Append the new atomic result to existing cache rows
            else:  # Initialize a new cache DataFrame when no cache file exists
                combined_df = row_df.copy()  # Use the normalized row as the complete cache content
            combined_df = prepare_cache_dataframe(combined_df, config=config)  # Normalize, deduplicate, and order the complete cache content
            tmp_fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")  # Create temp file in same directory for atomic rename
            try:  # Write to temp file before atomic rename
                with os.fdopen(tmp_fd, "w", encoding="utf-8", newline="") as tmp_f:  # Open temp file descriptor for writing
                    tmp_f.write(combined_df.to_csv(index=False, header=True))  # Write canonical cache content with a single header row
                    sync_cache_file_data(tmp_f)  # Flush and synchronize temp-file bytes before replacement
                os.replace(tmp_path, cache_path)  # Atomically replace the cache file after the temp file is complete
                sync_cache_parent_directory(cache_path)  # Synchronize parent-directory metadata where supported
            except Exception:  # If write or rename fails
                try:  # Attempt to clean up temp file
                    os.unlink(tmp_path)  # Remove temp file to avoid leftover artifacts
                except Exception:  # If temp file cleanup fails
                    pass  # Ignore cleanup failure to avoid masking the original error
                raise  # Re-raise original write failure
        except Exception as e:  # If any cache write operation fails
            verbose_output(
                f"{BackgroundColors.YELLOW}Warning: Failed to save result to cache {BackgroundColors.CYAN}{cache_path}{BackgroundColors.YELLOW}: {e}{Style.RESET_ALL}",
                config=config,
            )  # Log cache write failure before propagating the exception
            raise  # Propagate cache write failure so completed atomic results are not treated as safely persisted
    except Exception as e:  # Catch any unexpected errors in the cache saving process
        print(str(e))  # Log the error message for debugging
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send the exception details via Telegram for monitoring
        raise  # Re-raise the exception to allow upstream handling if necessary


def get_automl_search_spaces():
    """
    Return hyperparameter search space definitions for all AutoML candidate models.

    :return: Dictionary mapping model names to their search space configurations.
    """
    
    try:
        return {  # Dictionary of model search spaces
            "Random Forest": {  # Random Forest search space
                "n_estimators": ("int", 50, 500),  # Number of trees range
                "max_depth": ("int_or_none", 3, 50),  # Max depth range or None
                "min_samples_split": ("int", 2, 20),  # Min samples to split
                "min_samples_leaf": ("int", 1, 10),  # Min samples per leaf
                "max_features": ("categorical", ["sqrt", "log2", None]),  # Feature selection method
            },
            "XGBoost": {  # XGBoost search space
                "n_estimators": ("int", 50, 500),  # Number of boosting rounds
                "max_depth": ("int", 3, 15),  # Max tree depth
                "learning_rate": ("float_log", 0.01, 0.3),  # Learning rate (log scale)
                "subsample": ("float", 0.5, 1.0),  # Row subsampling ratio
                "colsample_bytree": ("float", 0.5, 1.0),  # Column subsampling ratio
                "min_child_weight": ("int", 1, 10),  # Min child weight
                "reg_alpha": ("float_log", 1e-8, 10.0),  # L1 regularization
                "reg_lambda": ("float_log", 1e-8, 10.0),  # L2 regularization
            },
            "LightGBM": {  # LightGBM search space
                "n_estimators": ("int", 50, 500),  # Number of boosting rounds
                "max_depth": ("int", 3, 15),  # Max tree depth
                "learning_rate": ("float_log", 0.01, 0.3),  # Learning rate (log scale)
                "num_leaves": ("int", 15, 127),  # Number of leaves
                "min_child_samples": ("int", 5, 100),  # Min samples in leaf
                "subsample": ("float", 0.5, 1.0),  # Row subsampling ratio
                "colsample_bytree": ("float", 0.5, 1.0),  # Column subsampling ratio
                "reg_alpha": ("float_log", 1e-8, 10.0),  # L1 regularization
                "reg_lambda": ("float_log", 1e-8, 10.0),  # L2 regularization
            },
            "Logistic Regression": {  # Logistic Regression search space
                "C": ("float_log", 0.001, 100.0),  # Regularization parameter
                "solver": ("categorical", ["lbfgs", "saga"]),  # Optimization algorithm
                "max_iter": ("int", 500, 5000),  # Max iterations
            },
            "SVM": {  # SVM search space
                "C": ("float_log", 0.01, 100.0),  # Regularization parameter
                "kernel": ("categorical", ["rbf", "linear", "poly"]),  # Kernel function
                "gamma": ("categorical", ["scale", "auto"]),  # Kernel coefficient
            },
            "Extra Trees": {  # Extra Trees search space
                "n_estimators": ("int", 50, 500),  # Number of trees
                "max_depth": ("int_or_none", 3, 50),  # Max depth or None
                "min_samples_split": ("int", 2, 20),  # Min samples to split
                "min_samples_leaf": ("int", 1, 10),  # Min samples per leaf
                "max_features": ("categorical", ["sqrt", "log2", None]),  # Feature selection method
            },
            "Gradient Boosting": {  # Gradient Boosting search space
                "n_estimators": ("int", 50, 300),  # Number of boosting rounds
                "max_depth": ("int", 3, 10),  # Max tree depth
                "learning_rate": ("float_log", 0.01, 0.3),  # Learning rate
                "subsample": ("float", 0.5, 1.0),  # Row subsampling ratio
                "min_samples_split": ("int", 2, 20),  # Min samples to split
                "min_samples_leaf": ("int", 1, 10),  # Min samples per leaf
            },
            "MLP (Neural Net)": {  # MLP Neural Network search space
                "hidden_layer_sizes_0": ("int", 32, 256),  # First hidden layer size
                "hidden_layer_sizes_1": ("int", 0, 128),  # Second hidden layer size (0 means single layer)
                "learning_rate_init": ("float_log", 0.0001, 0.01),  # Initial learning rate
                "alpha": ("float_log", 1e-6, 0.01),  # L2 penalty (regularization)
                "max_iter": ("int", 200, 1000),  # Max iterations
                "activation": ("categorical", ["relu", "tanh"]),  # Activation function
            },
            "Decision Tree": {  # Decision Tree search space
                "max_depth": ("int_or_none", 3, 50),  # Max depth or None
                "min_samples_split": ("int", 2, 20),  # Min samples to split
                "min_samples_leaf": ("int", 1, 10),  # Min samples per leaf
                "criterion": ("categorical", ["gini", "entropy"]),  # Split criterion
                "max_features": ("categorical", ["sqrt", "log2", None]),  # Feature selection method
            },
            "KNN": {  # K-Nearest Neighbors search space
                "n_neighbors": ("int", 3, 25),  # Number of neighbors
                "weights": ("categorical", ["uniform", "distance"]),  # Weight function
                "metric": ("categorical", ["euclidean", "manhattan", "minkowski"]),  # Distance metric
            },
        }  # Return full search space dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def suggest_hyperparameters_for_model(trial, model_name, search_spaces):
    """
    Suggests hyperparameters for a given model using an Optuna trial.

    :param trial: Optuna trial object for parameter suggestion
    :param model_name: Name of the model to suggest hyperparameters for
    :param search_spaces: Dictionary of search space definitions
    :return: Dictionary of suggested hyperparameters
    """
    
    try:
        space = search_spaces.get(model_name, {})  # Get the search space for this model
        params = {}  # Initialize empty parameters dictionary

        for param_name, config in space.items():  # Iterate over each parameter definition
            param_type = config[0]  # Extract the parameter type

            if param_type == "int":  # Integer parameter
                params[param_name] = trial.suggest_int(param_name, config[1], config[2])  # Suggest integer value
            elif param_type == "float":  # Float parameter (uniform)
                params[param_name] = trial.suggest_float(param_name, config[1], config[2])  # Suggest float value
            elif param_type == "float_log":  # Float parameter (log scale)
                params[param_name] = trial.suggest_float(param_name, config[1], config[2], log=True)  # Suggest log-scaled float
            elif param_type == "categorical":  # Categorical parameter
                params[param_name] = trial.suggest_categorical(param_name, config[1])  # Suggest from categories
            elif param_type == "int_or_none":  # Integer or None parameter
                use_none = trial.suggest_categorical(f"{param_name}_none", [True, False])  # Decide whether to use None
                if use_none:  # If None is selected
                    params[param_name] = None  # Set parameter to None
                else:  # Otherwise suggest an integer
                    params[param_name] = trial.suggest_int(param_name, config[1], config[2])  # Suggest integer value

        return params  # Return the suggested parameters
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def create_model_from_params(model_name, params, config=None):
    """
    Creates a classifier instance from a model name and hyperparameters dictionary.

    :param model_name: Name of the classifier to instantiate
    :param params: Dictionary of hyperparameters to apply
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Instantiated classifier object
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        automl_random_state = config.get("automl", {}).get("random_state", 42)  # Get random state from config
        n_jobs = config.get("evaluation", {}).get("n_jobs", 1)  # Get n_jobs from config

        clean_params = {k: v for k, v in params.items() if not k.endswith("_none")}  # Copy params excluding _none flags

        if model_name == "Random Forest":  # Random Forest classifier
            return RandomForestClassifier(random_state=automl_random_state, n_jobs=n_jobs, **clean_params)  # Create RF instance
        elif model_name == "XGBoost":  # XGBoost classifier
            return XGBClassifier(eval_metric="mlogloss", random_state=automl_random_state, n_jobs=n_jobs, **clean_params)  # Create XGB instance
        elif model_name == "LightGBM":  # LightGBM classifier
            return lgb.LGBMClassifier(force_row_wise=True, random_state=automl_random_state, verbosity=-1, n_jobs=n_jobs, **clean_params)  # Create LGBM instance
        elif model_name == "Logistic Regression":  # Logistic Regression classifier
            return LogisticRegression(random_state=automl_random_state, n_jobs=n_jobs, **clean_params)  # Create LR instance with parallel jobs
        elif model_name == "SVM":  # Support Vector Machine classifier
            return SVC(probability=True, random_state=automl_random_state, **clean_params)  # Create SVM instance
        elif model_name == "Extra Trees":  # Extra Trees classifier
            return ExtraTreesClassifier(random_state=automl_random_state, n_jobs=n_jobs, **clean_params)  # Create ET instance
        elif model_name == "Gradient Boosting":  # Gradient Boosting classifier
            return GradientBoostingClassifier(random_state=automl_random_state, **clean_params)  # Create GB instance
        elif model_name == "MLP (Neural Net)":  # MLP Neural Network classifier
            hidden_0 = clean_params.pop("hidden_layer_sizes_0", 100)  # Extract first hidden layer size
            hidden_1 = clean_params.pop("hidden_layer_sizes_1", 0)  # Extract second hidden layer size
            hidden_layers = (hidden_0,) if hidden_1 == 0 else (hidden_0, hidden_1)  # Build hidden layer tuple
            return MLPClassifier(hidden_layer_sizes=hidden_layers, random_state=automl_random_state, **clean_params)  # Create MLP instance
        elif model_name == "Decision Tree":  # Decision Tree classifier
            return DecisionTreeClassifier(random_state=automl_random_state, **clean_params)  # Create DT instance
        elif model_name == "KNN":  # K-Nearest Neighbors classifier
            return KNeighborsClassifier(n_jobs=n_jobs, **clean_params)  # Create KNN instance
        else:  # Unknown model type
            raise ValueError(f"Unknown AutoML model name: {model_name}")  # Raise error for unknown model
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def automl_cross_validate_model(model, X_train, y_train, cv_folds, trial=None, config=None):
    """
    Performs stratified cross-validation on a model and returns mean F1 score.

    :param model: Classifier instance to evaluate
    :param X_train: Training features array
    :param y_train: Training target array (numpy)
    :param cv_folds: Number of cross-validation folds
    :param trial: Optional Optuna trial for intermediate reporting and pruning
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Mean cross-validated F1 score
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        automl_random_state = config.get("automl", {}).get("random_state", 42)  # Get random state from config

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=automl_random_state)  # Create stratified k-fold
        f1_scores = []  # Initialize list for F1 scores

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):  # Iterate over folds
            X_fold_train = X_train[train_idx]  # Get fold training features
            y_fold_train = y_train[train_idx]  # Get fold training target
            X_fold_val = X_train[val_idx]  # Get fold validation features
            y_fold_val = y_train[val_idx]  # Get fold validation target

            model.fit(X_fold_train, y_fold_train)  # Fit model on fold training data using its internal n_jobs parallelism
            y_pred = model.predict(X_fold_val)  # Predict on fold validation data
            fold_f1 = f1_score(y_fold_val, y_pred, average="weighted", zero_division=cast(Any, 0))  # Calculate fold F1
            f1_scores.append(fold_f1)  # Append fold F1 score

            if trial is not None:  # If Optuna trial is provided
                trial.report(np.mean(f1_scores), fold_idx)  # Report intermediate value for pruning
                if trial.should_prune():  # Verify if trial should be pruned
                    raise optuna.exceptions.TrialPruned()  # Prune this trial

        return float(np.mean(f1_scores))  # Return mean F1 score across folds as Python float
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def automl_objective(trial, X_train, y_train, cv_folds, config=None):
    """
    Optuna objective function for automated model and hyperparameter selection.

    :param trial: Optuna trial object
    :param X_train: Training features array
    :param y_train: Training target array (numpy)
    :param cv_folds: Number of cross-validation folds
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Mean cross-validated F1 score (to maximize)
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        search_spaces = get_automl_search_spaces()  # Get all model search spaces
        model_names = list(search_spaces.keys())  # Get list of available model names

        model_name = trial.suggest_categorical("model_name", model_names)  # Select model type via trial
        params = suggest_hyperparameters_for_model(trial, model_name, search_spaces)  # Suggest hyperparameters

        try:  # Try to create and evaluate the model
            model = create_model_from_params(model_name, params, config=config)  # Create model instance from params
            mean_f1 = automl_cross_validate_model(model, X_train, y_train, cv_folds, trial, config=config)  # Cross-validate
            return mean_f1  # Return mean F1 score
        except optuna.exceptions.TrialPruned:  # Handle Optuna pruning
            raise  # Re-raise pruning exception
        except Exception as e:  # Handle other errors gracefully
            verbose_output(
                f"{BackgroundColors.YELLOW}AutoML trial failed for {model_name}: {e}{Style.RESET_ALL}",
                config=config,
            )  # Log the trial failure
            return 0.0  # Return zero score for failed trials
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_automl_model_search(X_train, y_train, file_path, config=None):
    """
    Runs Optuna-based AutoML model search to find optimal classifier and hyperparameters.

    :param X_train: Scaled training features (numpy array)
    :param y_train: Training target labels (numpy array)
    :param file_path: Path to the dataset file for logging
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (best_model_name, best_params, study) or (None, None, None) on failure
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        automl_n_trials = config.get("automl", {}).get("n_trials", 50)  # Get number of trials from config
        automl_timeout = config.get("automl", {}).get("timeout", 3600)  # Get timeout from config
        automl_cv_folds = config.get("automl", {}).get("cv_folds", 5)  # Get CV folds from config
        automl_random_state = config.get("automl", {}).get("random_state", 42)  # Get random state from config

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Starting AutoML model search with {BackgroundColors.CYAN}{automl_n_trials}{BackgroundColors.GREEN} trials...{Style.RESET_ALL}"
        )  # Output search start message

        optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress verbose Optuna logging

        sampler = optuna.samplers.TPESampler(seed=automl_random_state)  # Create TPE sampler with deterministic seed
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)  # Create median pruner for early stopping

        study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner, study_name="automl_model_search"
        )  # Create Optuna study to maximize F1 score

        objective_fn = lambda trial: automl_objective(trial, X_train, y_train, automl_cv_folds, config=config)  # Create objective wrapper
        study.optimize(objective_fn, n_trials=automl_n_trials, timeout=automl_timeout, n_jobs=1)  # Run the optimization

        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]  # Get completed trials

        if not completed_trials:  # If no trials completed successfully
            print(
                f"{BackgroundColors.RED}AutoML model search failed: no successful trials completed.{Style.RESET_ALL}"
            )  # Output failure message
            return (None, None, None)  # Return None tuple

        best_trial = study.best_trial  # Get the best trial
        best_model_name = best_trial.params.get("model_name", "Unknown")  # Extract best model name
        best_params = {
            k: v for k, v in best_trial.params.items() if k != "model_name" and not k.endswith("_none")
        }  # Extract best params excluding model_name and _none flags

        pruned_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])  # Count pruned trials

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}AutoML Best Model: {BackgroundColors.CYAN}{best_model_name}{Style.RESET_ALL}"
        )  # Output best model name
        print(
            f"{BackgroundColors.GREEN}Best CV F1 Score: {BackgroundColors.CYAN}{study.best_value}{Style.RESET_ALL}"
        )  # Output best F1 score using raw float precision
        print(
            f"{BackgroundColors.GREEN}Best Parameters: {BackgroundColors.CYAN}{best_params}{Style.RESET_ALL}"
        )  # Output best parameters
        print(
            f"{BackgroundColors.GREEN}Trials: {BackgroundColors.CYAN}{len(completed_trials)} completed, {pruned_count} pruned{Style.RESET_ALL}"
        )  # Output trial statistics

        return (best_model_name, best_params, study)  # Return best model info and study object
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def automl_stacking_objective(trial, X_train, y_train, cv_folds, candidate_models, config=None):
    """
    Optuna objective function for optimizing stacking ensemble configuration.

    :param trial: Optuna trial object
    :param X_train: Training features array
    :param y_train: Training target array (numpy)
    :param cv_folds: Number of cross-validation folds
    :param candidate_models: Dictionary mapping model names to parameter dicts
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Mean cross-validated F1 score (to maximize)
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        automl_random_state = config.get("automl", {}).get("random_state", 42)  # Get random state from config
        n_jobs = config.get("evaluation", {}).get("n_jobs", 1)  # Get n_jobs from config

        model_names = list(candidate_models.keys())  # Get list of candidate model names
        selected_models = []  # Initialize list for selected base learners

        for name in model_names:  # Iterate over each candidate model
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")  # Sanitize name for Optuna parameter
            include = trial.suggest_categorical(f"use_{safe_name}", [True, False])  # Decide whether to include this model
            if include:  # If model is selected for inclusion
                selected_models.append(name)  # Add to selected list

        if len(selected_models) < 2:  # Need at least 2 base learners for stacking
            return 0.0  # Return zero score if insufficient base learners

        meta_learner_name = trial.suggest_categorical(
            "meta_learner", ["Logistic Regression", "Random Forest", "Gradient Boosting"]
        )  # Select meta-learner type
        n_cv_splits = trial.suggest_int("stacking_cv_splits", 3, min(cv_folds, 10))  # Select CV splits for stacking

        try:  # Try to build and evaluate stacking ensemble
            estimators = []  # Initialize base estimators list
            for name in selected_models:  # Build each selected base learner
                model_params = candidate_models[name]  # Get pre-optimized parameters
                model = create_model_from_params(name, model_params, config=config)  # Create model instance
                safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")  # Sanitize estimator name
                estimators.append((safe_name, model))  # Add to estimators list

            if meta_learner_name == "Logistic Regression":  # Logistic Regression meta-learner
                meta_model = LogisticRegression(max_iter=1000, random_state=automl_random_state, n_jobs=n_jobs)  # Create LR meta-learner with parallel jobs
            elif meta_learner_name == "Random Forest":  # Random Forest meta-learner
                meta_model = RandomForestClassifier(n_estimators=50, random_state=automl_random_state, n_jobs=n_jobs)  # Create RF meta-learner
            else:  # Gradient Boosting meta-learner
                meta_model = GradientBoostingClassifier(random_state=automl_random_state)  # Create GB meta-learner

            stacking = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_model,
                cv=StratifiedKFold(n_splits=n_cv_splits, shuffle=True, random_state=automl_random_state),
                n_jobs=config.get("evaluation", {}).get("n_jobs", 1),
            )  # Create stacking classifier with sequential CV folds to prevent nested loky deadlock

            mean_f1 = automl_cross_validate_model(stacking, X_train, y_train, cv_folds, trial)  # Cross-validate stacking
            return mean_f1  # Return mean F1 score

        except optuna.exceptions.TrialPruned:  # Handle Optuna pruning
            raise  # Re-raise pruning exception
        except Exception as e:  # Handle other errors gracefully
            verbose_output(
                f"{BackgroundColors.YELLOW}AutoML stacking trial failed: {e}{Style.RESET_ALL}"
            )  # Log the failure
            return 0.0  # Return zero for failed trials
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_top_automl_models(study, top_n=5):
    """
    Extracts the top N unique models from an AutoML study based on F1 score.

    :param study: Completed Optuna study object
    :param top_n: Number of top models to extract
    :return: Dictionary mapping model names to their best parameters
    """
    
    try:
        completed = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]  # Filter to completed trials only
        sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)  # Sort trials by score descending

        top_models = {}  # Initialize dictionary for top models

        for trial in sorted_trials:  # Iterate through sorted trials
            model_name = trial.params.get("model_name", "Unknown")  # Get model name from trial
            if model_name not in top_models:  # If this model type hasn't been added yet
                params = {
                    k: v for k, v in trial.params.items() if k != "model_name" and not k.endswith("_none")
                }  # Extract parameters
                top_models[model_name] = params  # Store best params for this model type
            if len(top_models) >= top_n:  # If we've collected enough models
                break  # Stop collecting

        return top_models  # Return dictionary of top models and their parameters
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_automl_stacking_search(X_train, y_train, model_study, file_path, config=None):
    """
    Runs Optuna-based optimization to find the best stacking ensemble configuration.

    :param X_train: Scaled training features (numpy array)
    :param y_train: Training target labels (numpy array)
    :param model_study: Completed Optuna study from model search
    :param file_path: Path to the dataset file for logging
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (best_stacking_config, stacking_study) or (None, None) on failure
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Starting AutoML stacking search with {BackgroundColors.CYAN}{config.get("automl", {}).get("stacking_trials", 20)}{BackgroundColors.GREEN} trials...{Style.RESET_ALL}"
        )  # Output search start message

        candidate_models = extract_top_automl_models(model_study, top_n=config.get("automl", {}).get("stacking_top_n", 5))  # Get top models from model search

        if len(candidate_models) < 2:  # If not enough candidate models
            print(
                f"{BackgroundColors.YELLOW}Not enough candidate models for stacking search. Need at least 2, got {len(candidate_models)}.{Style.RESET_ALL}"
            )  # Output warning
            return (None, None)  # Return None tuple

        print(
            f"{BackgroundColors.GREEN}Candidate base learners: {BackgroundColors.CYAN}{list(candidate_models.keys())}{Style.RESET_ALL}"
        )  # Output candidate models

        optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress verbose Optuna logging

        sampler = optuna.samplers.TPESampler(seed=config.get("automl", {}).get("random_state", 42) + 1)  # Create sampler with different seed
        pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)  # Create pruner

        stacking_study = optuna.create_study(
            direction="maximize", sampler=sampler, pruner=pruner, study_name="automl_stacking_search"
        )  # Create Optuna study for stacking optimization

        objective_fn = lambda trial: automl_stacking_objective(
            trial, X_train, y_train, config.get("automl", {}).get("cv_folds", 5), candidate_models
        )  # Create stacking objective wrapper
        stacking_study.optimize(
            objective_fn, n_trials=config.get("automl", {}).get("stacking_trials", 20), timeout=config.get("automl", {}).get("timeout", 3600), n_jobs=1
        )  # Run stacking optimization

        completed = [
            t for t in stacking_study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]  # Get completed trials

        if not completed:  # If no stacking trials completed
            print(
                f"{BackgroundColors.YELLOW}AutoML stacking search: no successful trials.{Style.RESET_ALL}"
            )  # Output warning
            return (None, None)  # Return None tuple

        best_trial = stacking_study.best_trial  # Get best stacking trial
        best_config = {
            "meta_learner": best_trial.params.get("meta_learner"),  # Best meta-learner choice
            "stacking_cv_splits": best_trial.params.get("stacking_cv_splits"),  # Best CV splits
            "base_learners": [
                name for name in candidate_models.keys()
                if best_trial.params.get(f"use_{name.replace(' ', '_').replace('(', '').replace(')', '')}", False)
            ],  # Selected base learner names
            "base_learner_params": candidate_models,  # Parameters for each base learner
            "best_cv_f1": stacking_study.best_value,  # Best CV F1 score
        }  # Build best configuration dictionary

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}AutoML Best Stacking Config:{Style.RESET_ALL}"
        )  # Output header
        print(
            f"{BackgroundColors.GREEN}  Meta-learner: {BackgroundColors.CYAN}{best_config['meta_learner']}{Style.RESET_ALL}"
        )  # Output meta-learner
        print(
            f"{BackgroundColors.GREEN}  Base learners: {BackgroundColors.CYAN}{best_config['base_learners']}{Style.RESET_ALL}"
        )  # Output base learners
        print(
            f"{BackgroundColors.GREEN}  CV splits: {BackgroundColors.CYAN}{best_config['stacking_cv_splits']}{Style.RESET_ALL}"
        )  # Output CV splits
        print(
            f"{BackgroundColors.GREEN}  Best CV F1: {BackgroundColors.CYAN}{best_config['best_cv_f1']}{Style.RESET_ALL}"
        )  # Output best F1 using raw float precision

        return (best_config, stacking_study)  # Return best config and study
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_automl_stacking_model(best_config, config=None):
    """
    Builds a StackingClassifier from the best AutoML stacking configuration.

    :param best_config: Dictionary with best stacking configuration
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Configured StackingClassifier instance
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        estimators = []  # Initialize estimators list

        for name in best_config["base_learners"]:  # Iterate over selected base learners
            params = best_config["base_learner_params"].get(name, {})  # Get model parameters
            model = create_model_from_params(name, params)  # Create model instance
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")  # Sanitize estimator name
            estimators.append((safe_name, model))  # Add to estimators list

        meta_learner_name = best_config["meta_learner"]  # Get meta-learner name

        if meta_learner_name == "Logistic Regression":  # Logistic Regression meta-learner
            meta_model = LogisticRegression(max_iter=1000, random_state=config.get("automl", {}).get("random_state", 42), n_jobs=config.get("evaluation", {}).get("n_jobs", 1))  # Create LR with parallel jobs
        elif meta_learner_name == "Random Forest":  # Random Forest meta-learner
            meta_model = RandomForestClassifier(n_estimators=50, random_state=config.get("automl", {}).get("random_state", 42), n_jobs=config.get("evaluation", {}).get("n_jobs", 1))  # Create RF
        else:  # Gradient Boosting meta-learner
            meta_model = GradientBoostingClassifier(random_state=config.get("automl", {}).get("random_state", 42))  # Create GB

        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_model,
            cv=StratifiedKFold(
                n_splits=best_config["stacking_cv_splits"], shuffle=True, random_state=config.get("automl", {}).get("random_state", 42)
            ),
            n_jobs=config.get("evaluation", {}).get("n_jobs", 1),
        )  # Create stacking classifier with sequential CV folds to prevent nested loky deadlock

        return stacking_model  # Return configured stacking model
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def evaluate_automl_model_on_test(model, model_name, X_train, y_train, X_test, y_test):
    """
    Trains and evaluates an AutoML-selected model on the held-out test set.

    :param model: Classifier instance to evaluate
    :param model_name: Name of the model for logging
    :param X_train: Training features array
    :param y_train: Training target labels
    :param X_test: Testing features array
    :param y_test: Testing target labels
    :return: Dictionary containing all evaluation metrics
    """
    
    try:
        start_time = time.time()  # Record start time

        sys.stdout.flush()  # Flush stdout before model training to ensure logs are visible under nohup
        model.fit(X_train, y_train)  # Train model on full training set using its internal n_jobs parallelism
        y_pred = model.predict(X_test)  # Generate predictions on test set

        elapsed = time.time() - start_time  # Calculate elapsed training time

        acc = accuracy_score(y_test, y_pred)  # Calculate accuracy
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=cast(Any, 0))  # Calculate weighted precision
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=cast(Any, 0))  # Calculate weighted recall
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=cast(Any, 0))  # Calculate weighted F1 score

        roc_auc = None  # Initialize ROC-AUC as None
        try:  # Try to compute ROC-AUC
            if hasattr(model, "predict_proba"):  # If model supports probability predictions
                y_proba = model.predict_proba(X_test)  # Get probability predictions
                if len(np.unique(y_test)) == 2:  # Separate files evaluation classification
                    roc_auc = roc_auc_score(y_test, y_proba[:, 1])  # Compute separate files evaluation ROC-AUC
                else:  # Combined files evaluation classification
                    roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")  # Compute combined files evaluation ROC-AUC
        except Exception:  # If ROC-AUC computation fails
            roc_auc = None  # Keep as None

        fpr, fnr = compute_fpr_fnr(y_test, y_pred)  # Compute false positive and false negative rates for binary or multiclass predictions

        total_seconds = int(round(elapsed))  # Reuse elapsed as total seconds for reporting
        train_seconds = total_seconds  # Reuse total seconds as training time when only one timer exists
        exec_seconds = total_seconds  # Reuse total seconds as execution time when only one timer exists
        if len(np.unique(y_test)) == 2:  # Determine evaluation mode from label cardinality when CONFIG not available
            evaluation_mode = "SeparateFiles"  # Use SeparateFiles for binary classification
        else:  # For multiclass predictions
            evaluation_mode = "MultiClass"  # Use MultiClass for multi-class evaluation
        msg = f"{BackgroundColors.GREEN}{model_name}: Mode {evaluation_mode} | F1-Score {BackgroundColors.CYAN}{f1}{BackgroundColors.GREEN} | Accuracy: {BackgroundColors.CYAN}{acc}{BackgroundColors.GREEN} | Precision: {BackgroundColors.CYAN}{prec}{BackgroundColors.GREEN} | Recall: {BackgroundColors.CYAN}{rec}{BackgroundColors.GREEN} | FPR: {BackgroundColors.CYAN}{fpr}{BackgroundColors.GREEN} | FNR: {BackgroundColors.CYAN}{fnr}{BackgroundColors.GREEN} | Training Time: {BackgroundColors.CYAN}{int(train_seconds)}s{BackgroundColors.GREEN} | Execution Time: {BackgroundColors.CYAN}{int(exec_seconds)}s{BackgroundColors.GREEN} | Total Time: {BackgroundColors.CYAN}{calculate_execution_time(elapsed)} ({total_seconds}s){Style.RESET_ALL}"  # Build colored summary using raw floats for metrics and integer times
        print(msg)  # Output test results to console
        send_telegram_message(TELEGRAM_BOT, msg)  # Send identical message to Telegram for remote monitoring of ratio experiment results

        return {  # Build and return metrics dictionary
            "accuracy": acc,  # Accuracy value
            "precision": prec,  # Precision value
            "recall": rec,  # Recall value
            "f1_score": f1,  # F1 score value
            "roc_auc": roc_auc,  # ROC-AUC value
            "fpr": fpr,  # False positive rate
            "fnr": fnr,  # False negative rate
            "elapsed_time_s": int(round(elapsed)),  # Elapsed time in seconds
        }  # Return metrics dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def export_automl_search_history(study, output_dir, study_name, dataset_file=None):
    """
    Exports the Optuna study trial history to a CSV file.

    :param study: Completed Optuna study object
    :param output_dir: Directory path for saving the export file
    :param study_name: Name prefix for the output file
    :return: Path to the exported CSV file
    """
    
    try:
        if dataset_file is not None:
            stacking_output_dir = get_stacking_output_dir(dataset_file, CONFIG)
        else:
            fa_parent = None
            for p in Path(output_dir).parents:
                if p.name == "Feature_Analysis":
                    fa_parent = str(p)
                    break
            if fa_parent is None:
                raise RuntimeError("Cannot derive Feature_Analysis root from output_dir for safe write")
            stacking_output_dir = str((Path(fa_parent).parent / CONFIG.get("stacking", {}).get("results_dir")).resolve())
        validate_output_path(stacking_output_dir, str(Path(output_dir)))
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

        trials_data = []  # Initialize list for trial data

        for trial in study.trials:  # Iterate over all trials
            trial_entry = {  # Build entry for this trial
                "trial_number": trial.number,  # Trial index number
                "value": trial.value if trial.value is not None else None,  # Objective value (F1 score)
                "state": trial.state.name,  # Trial state (COMPLETE, PRUNED, FAIL)
                "duration_s": (
                    trial.duration.total_seconds() if trial.duration else None
                ),  # Trial duration in seconds
            }  # Build basic trial entry
            trial_entry.update(trial.params)  # Add trial parameters to entry
            trials_data.append(trial_entry)  # Append to trials data list

        df = pd.DataFrame(trials_data)  # Convert trials data to DataFrame
        output_path = os.path.join(output_dir, f"{study_name}_search_history.csv")  # Build output file path
        generate_csv_and_image(df, output_path, is_visualizable=True, index=False)  # Save to CSV and generate PNG image

        print(
            f"{BackgroundColors.GREEN}AutoML search history exported to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
        )  # Output export confirmation

        return output_path  # Return the output file path
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def export_automl_best_config(best_model_name, best_params, test_metrics, stacking_config, output_dir, feature_names, dataset_file=None):
    """
    Exports the best AutoML configuration and metrics to a JSON file.

    :param best_model_name: Name of the best model found
    :param best_params: Best hyperparameters dictionary
    :param test_metrics: Test set evaluation metrics dictionary
    :param stacking_config: Best stacking configuration dictionary (or None)
    :param output_dir: Directory path for saving the export file
    :param feature_names: List of feature names used
    :return: Path to the exported JSON file
    """
    
    try:
        if dataset_file is not None:
            stacking_output_dir = get_stacking_output_dir(dataset_file, CONFIG)
        else:
            fa_parent = None
            for p in Path(output_dir).parents:
                if p.name == "Feature_Analysis":
                    fa_parent = str(p)
                    break
            if fa_parent is None:
                raise RuntimeError("Cannot derive Feature_Analysis root from output_dir for safe write")
            stacking_output_dir = str((Path(fa_parent).parent / CONFIG.get("stacking", {}).get("results_dir")).resolve())
        validate_output_path(stacking_output_dir, str(Path(output_dir)))
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

        export_config = {  # Build configuration export dictionary
            "best_individual_model": {  # Best individual model section
                "model_name": best_model_name,  # Model name
                "hyperparameters": best_params,  # Model hyperparameters
                "test_metrics": {
                    k: v if not (isinstance(v, (int, float)) and v is None) else v
                    for k, v in test_metrics.items()
                },  # Test metrics preserved with raw numeric precision
            },
            "best_stacking_config": stacking_config,  # Stacking configuration (may be None)
            "automl_settings": {  # AutoML settings used
                "n_trials": CONFIG.get("automl", {}).get("n_trials", 50),  # Number of model search trials
                "stacking_trials": CONFIG.get("automl", {}).get("stacking_trials", 20),  # Number of stacking search trials
                "cv_folds": CONFIG.get("automl", {}).get("cv_folds", 5),  # Cross-validation folds
                "timeout_s": CONFIG.get("automl", {}).get("timeout", 3600),  # Timeout in seconds
                "random_state": CONFIG.get("automl", {}).get("random_state", 42),  # Random seed used
            },
            "feature_names": feature_names,  # Features used in training
            "n_features": len(feature_names),  # Number of features
        }  # Build complete config dictionary

        output_path = os.path.join(output_dir, CONFIG.get("automl", {}).get("results_filename", "AutoML_Results.csv").replace(".csv", "_best_config.json"))  # Build output path

        if dataset_file is not None:
            stacking_output_dir = get_stacking_output_dir(dataset_file, CONFIG)
        else:
            fa_parent = None
            for p in Path(output_path).parents:
                if p.name == "Feature_Analysis":
                    fa_parent = str(p)
                    break
            if fa_parent is None:
                raise RuntimeError("Cannot derive Feature_Analysis root from output_path for safe write")
            stacking_output_dir = str((Path(fa_parent).parent / CONFIG.get("stacking", {}).get("results_dir")).resolve())
        validate_output_path(stacking_output_dir, output_path)

        with open(output_path, "w", encoding="utf-8") as f:  # Open file for writing
            json.dump(export_config, f, indent=2, default=str)  # Write JSON with indentation

        print(
            f"{BackgroundColors.GREEN}AutoML best configuration exported to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
        )  # Output export confirmation

        return output_path  # Return the output file path
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def export_automl_best_model(model, scaler, output_dir, model_name, feature_names, dataset_file=None):
    """
    Exports the best AutoML model and scaler to disk using joblib.

    :param model: Trained best model instance
    :param scaler: Fitted StandardScaler instance
    :param output_dir: Directory path for saving model files
    :param model_name: Name of the model for file naming
    :param feature_names: List of feature names for metadata
    :return: Tuple (model_path, scaler_path)
    """
    
    try:
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

        safe_name = re.sub(r'[\\/*?:"<>|() ]', "_", str(model_name))  # Sanitize model name for filename
        model_path = os.path.join(output_dir, f"AutoML_best_{safe_name}_model.joblib")  # Build model file path
        scaler_path = os.path.join(output_dir, f"AutoML_best_{safe_name}_scaler.joblib")  # Build scaler file path

        if dataset_file is not None:
            stacking_output_dir = get_stacking_output_dir(dataset_file, CONFIG)
        else:
            fa_parent = None
            for p in Path(output_dir).parents:
                if p.name == "Feature_Analysis":
                    fa_parent = str(p)
                    break
            if fa_parent is None:
                raise RuntimeError("Cannot derive Feature_Analysis root from output_dir for safe write")
            stacking_output_dir = str((Path(fa_parent).parent / CONFIG.get("stacking", {}).get("results_dir")).resolve())
        validate_output_path(stacking_output_dir, model_path)
        dump(model, model_path)  # Export model to disk

        if scaler is not None:  # If scaler is provided
            validate_output_path(stacking_output_dir, scaler_path)
            dump(scaler, scaler_path)  # Export scaler to disk

        meta_path = os.path.join(output_dir, f"AutoML_best_{safe_name}_meta.json")  # Build metadata file path
        meta = {  # Build metadata dictionary
            "model_name": model_name,  # Model name
            "features": feature_names,  # Feature names used
            "n_features": len(feature_names),  # Number of features
        }  # Metadata content

        validate_output_path(stacking_output_dir, meta_path)
        with open(meta_path, "w", encoding="utf-8") as f:  # Open metadata file
            json.dump(meta, f, indent=2)  # Write metadata JSON

        print(
            f"{BackgroundColors.GREEN}AutoML best model exported to: {BackgroundColors.CYAN}{model_path}{Style.RESET_ALL}"
        )  # Output export confirmation

        return (model_path, scaler_path)  # Return file paths
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_automl_results_list(best_model_name, best_params, individual_metrics, stacking_metrics, stacking_config, file_path, feature_names, n_train, n_test, config=None):
    """
    Builds the results list for AutoML CSV export matching existing results format.

    :param best_model_name: Name of the best individual model
    :param best_params: Best hyperparameters for the individual model
    :param individual_metrics: Metrics from individual model test evaluation
    :param stacking_metrics: Metrics from stacking model test evaluation (or None)
    :param stacking_config: Best stacking configuration (or None)
    :param file_path: Path to the dataset file
    :param feature_names: List of feature names
    :param n_train: Number of training samples
    :param n_test: Number of test samples
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: List of result dictionaries for CSV export
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        results = []  # Initialize results list
        path_is_directory = resolve_path_represents_directory(str(file_path))  # Resolve whether AutoML identity comes from a directory.
        dataset_identity = resolve_canonical_dataset_identity(str(file_path), True) if path_is_directory else os.path.relpath(file_path)  # Resolve AutoML result-row dataset identity by path scope.

        individual_entry = {  # Build individual model result entry
            "model": best_model_name,  # Model class name
            "dataset": dataset_identity,  # Store resolved AutoML dataset identity.
            "feature_set": "AutoML",  # Feature set label
            "classifier_type": "AutoML_Individual",  # Classifier type
            "model_name": f"AutoML_{best_model_name}",  # Prefixed model name
            "data_source": "Original",  # Data source label
            "experiment_id": None,  # No experiment ID for standalone AutoML runs
            "experiment_mode": "original_only",  # AutoML runs on original data only
            "augmentation_ratio": resolve_persisted_augmentation_ratio("original_only", None),  # Persist AutoML baseline ratio as numeric zero.
            "n_features": len(feature_names),  # Number of features
            "n_samples_train": n_train,  # Training sample count
            "n_samples_test": n_test,  # Test sample count
                "accuracy": individual_metrics["accuracy"],  # Accuracy as raw float
                "precision": individual_metrics["precision"],  # Precision as raw float
                "recall": individual_metrics["recall"],  # Recall as raw float
                "f1_score": individual_metrics["f1_score"],  # F1 score as raw float
                "fpr": individual_metrics["fpr"],  # False positive rate as raw float
                "fnr": individual_metrics["fnr"],  # False negative rate as raw float
            "elapsed_time_s": individual_metrics["elapsed_time_s"],  # Elapsed time
            "cv_method": f"Optuna({config.get("automl", {}).get("n_trials", 50)} trials, {config.get("automl", {}).get("cv_folds", 5)}-fold CV)",  # CV method description
            "top_features": json.dumps(feature_names),  # Feature names as JSON
            "rfe_ranking": None,  # No RFE ranking for AutoML
            "hyperparameters": json.dumps(normalize_metadata_for_json(best_params), sort_keys=True, allow_nan=False),  # Hyperparameters as deterministic valid JSON.
            "features_list": feature_names,  # Feature names list
        }  # Individual model result entry
        results.append(individual_entry)  # Add to results list

        if stacking_metrics is not None and stacking_config is not None:  # If stacking results are available
            stacking_entry = {  # Build stacking result entry
                "model": "StackingClassifier",  # Model class name
                "dataset": dataset_identity,  # Store resolved AutoML dataset identity.
                "feature_set": "AutoML",  # Feature set label
                "classifier_type": "AutoML_Stacking",  # Classifier type
                "model_name": "AutoML_StackingClassifier",  # Prefixed model name
                "data_source": "Original",  # Data source label
                "experiment_id": None,  # No experiment ID for standalone AutoML runs
                "experiment_mode": "original_only",  # AutoML runs on original data only
                "augmentation_ratio": resolve_persisted_augmentation_ratio("original_only", None),  # Persist AutoML baseline ratio as numeric zero.
                "n_features": len(feature_names),  # Number of features
                "n_samples_train": n_train,  # Training sample count
                "n_samples_test": n_test,  # Test sample count
                "accuracy": stacking_metrics["accuracy"],  # Accuracy as raw float
                "precision": stacking_metrics["precision"],  # Precision as raw float
                "recall": stacking_metrics["recall"],  # Recall as raw float
                "f1_score": stacking_metrics["f1_score"],  # F1 score as raw float
                "fpr": stacking_metrics["fpr"],  # False positive rate as raw float
                "fnr": stacking_metrics["fnr"],  # False negative rate as raw float
                "elapsed_time_s": stacking_metrics["elapsed_time_s"],  # Elapsed time
                "cv_method": f"Optuna({config.get("automl", {}).get("stacking_trials", 20)} trials, {config.get("automl", {}).get("cv_folds", 5)}-fold CV)",  # CV method description
                "top_features": json.dumps(feature_names),  # Feature names as JSON
                "rfe_ranking": None,  # No RFE ranking for AutoML
                "hyperparameters": json.dumps(normalize_metadata_for_json(stacking_config), sort_keys=True, allow_nan=False),  # Stacking config as deterministic valid JSON.
                "features_list": feature_names,  # Feature names list
            }  # Stacking result entry
            results.append(stacking_entry)  # Add stacking to results list

        return results  # Return results list
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def save_automl_results(csv_path, results_list, config=None):
    """
    Saves AutoML results to a dedicated CSV file in the Feature_Analysis/AutoML directory.

    :param csv_path: Path to the original dataset CSV file
    :param results_list: List of result dictionaries to save
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        if not results_list:  # If no results to save
            return  # Exit early

        dataset_root = resolve_dataset_root_path(str(csv_path))  # Resolve AutoML output root from file or directory input.
        automl_dir = dataset_root / "Feature_Analysis" / "AutoML"  # Build AutoML output directory from dataset root.
        os.makedirs(automl_dir, exist_ok=True)  # Ensure directory exists
        output_path = automl_dir / config.get("automl", {}).get("results_filename", "AutoML_Results.csv")  # Build output file path

        df = pd.DataFrame(results_list)  # Convert results to DataFrame
        column_order = list(config.get("stacking", {}).get("results_csv_columns", []))  # Use canonical column ordering
        existing_columns = [col for col in column_order if col in df.columns]  # Filter to existing columns
        df = df[existing_columns + [c for c in df.columns if c not in existing_columns]]  # Reorder columns

        df = add_hardware_column(df, existing_columns)  # Add hardware specifications column

        generate_csv_and_image(df, str(output_path), is_visualizable=True, index=False, encoding="utf-8")  # Save AutoML results CSV and generate PNG image

        print(
            f"\n{BackgroundColors.GREEN}AutoML results saved to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
        )  # Output save confirmation
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_automl_stacking_phase(X_train_scaled, y_train_arr, X_test_scaled, y_test_arr, model_study, file, config=None):
    """
    Run AutoML stacking search and evaluation phase if stacking trials are enabled.

    :param X_train_scaled: Scaled training features
    :param y_train_arr: Training target as numpy array
    :param X_test_scaled: Scaled test features
    :param y_test_arr: Test target as numpy array
    :param model_study: Optuna study from model search phase
    :param file: Path to the dataset file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple of (stacking_config, stacking_metrics, stacking_study)
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        stacking_config = None  # Initialize stacking config as None
        stacking_metrics = None  # Initialize stacking metrics as None
        stacking_study = None  # Initialize stacking study as None

        if config.get("automl", {}).get("stacking_trials", 20) > 0:  # If stacking search is enabled
            stacking_config, stacking_study = run_automl_stacking_search(
                X_train_scaled, y_train_arr, model_study, file, config=config
            )  # Run stacking search optimization

            if stacking_config is not None:  # If stacking search succeeded
                best_stacking_model = build_automl_stacking_model(stacking_config, config=config)  # Build best stacking model

                print(
                    f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Evaluating AutoML best stacking model on test set...{Style.RESET_ALL}"
                )  # Output evaluation message

                stacking_metrics = evaluate_automl_model_on_test(
                    best_stacking_model, "AutoML_Stacking", X_train_scaled, y_train_arr, X_test_scaled, y_test_arr
                )  # Evaluate stacking model on test set

        return stacking_config, stacking_metrics, stacking_study  # Return stacking results tuple
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def export_automl_pipeline_artifacts(model_study, stacking_study, stacking_config, stacking_metrics, best_model_name, best_params, individual_metrics, best_individual_model, scaler, file, feature_names, X_train_scaled, y_train_arr, automl_output_dir, config=None):
    """
    Export all AutoML pipeline artifacts including search history, configs, and models.

    :param model_study: Optuna study from model search phase
    :param stacking_study: Optuna study from stacking search phase or None
    :param stacking_config: Best stacking configuration or None
    :param stacking_metrics: Stacking evaluation metrics or None
    :param best_model_name: Name of the best individual model
    :param best_params: Best individual model parameters
    :param individual_metrics: Individual model evaluation metrics
    :param best_individual_model: Best individual model instance
    :param scaler: Fitted scaler for feature transformation
    :param file: Path to the dataset file
    :param feature_names: List of feature column names
    :param X_train_scaled: Scaled training features
    :param y_train_arr: Training target as numpy array
    :param automl_output_dir: Output directory for AutoML artifacts
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        export_automl_search_history(model_study, automl_output_dir, "model_search", dataset_file=file)  # Export model search history

        if stacking_study is not None:  # If stacking study exists
            export_automl_search_history(stacking_study, automl_output_dir, "stacking_search", dataset_file=file)  # Export stacking search history

        export_automl_best_config(
            best_model_name, best_params, individual_metrics, stacking_config, automl_output_dir, feature_names, dataset_file=file
        )  # Export best configuration to file

        export_automl_best_model(
            best_individual_model, scaler, automl_output_dir, best_model_name, feature_names, dataset_file=file
        )  # Export best individual model to file

        if stacking_config is not None and stacking_metrics is not None:  # If stacking was successful
            best_stacking_model_final = build_automl_stacking_model(stacking_config, config=config)  # Rebuild stacking model for export
            sys.stdout.flush()  # Flush stdout before stacking training to ensure logs are visible under nohup
            best_stacking_model_final.fit(X_train_scaled, y_train_arr)  # Fit stacking model on full training data
            export_automl_best_model(
                best_stacking_model_final, scaler, automl_output_dir, "AutoML_Stacking", feature_names, dataset_file=file
            )  # Export best stacking model to file
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_automl_model_search_and_eval(X_train_scaled, y_train_arr, X_test_scaled, y_test_arr, file):
    """
    Executes AutoML phase 1 model search, instantiates the best model, and evaluates it on the test set.

    :param X_train_scaled: Scaled training feature matrix
    :param y_train_arr: Training target labels as numpy array
    :param X_test_scaled: Scaled test feature matrix
    :param y_test_arr: Test target labels as numpy array
    :param file: Path to the dataset file for logging context
    :return: Tuple (best_model_name, best_params, model_study, best_individual_model, individual_metrics) or None if search fails
    """

    try:
        best_model_name, best_params, model_study = run_automl_model_search(X_train_scaled, y_train_arr, file)  # Run Optuna-based model search to find best model and hyperparameters

        if best_model_name is None:  # If model search found no valid model
            print(
                f"{BackgroundColors.RED}AutoML pipeline aborted: model search failed.{Style.RESET_ALL}"
            )  # Output failure message to terminal
            return None  # Signal caller to abort pipeline

        best_individual_model = create_model_from_params(best_model_name, best_params)  # Instantiate best model using found parameters

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Evaluating AutoML best individual model on test set...{Style.RESET_ALL}"
        )  # Output evaluation start message

        individual_metrics = evaluate_automl_model_on_test(
            best_individual_model, best_model_name, X_train_scaled, y_train_arr, X_test_scaled, y_test_arr
        )  # Evaluate the instantiated best individual model on the held-out test set

        return (best_model_name, best_params, model_study, best_individual_model, individual_metrics)  # Return all search and evaluation artifacts
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def prepare_automl_training_data(df, config=None):
    """
    Extracts features and target from DataFrame, validates class count, scales and splits data for AutoML.

    :param df: Preprocessed DataFrame with numeric features and a target in the last column
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (X_train_scaled, X_test_scaled, y_train_arr, y_test_arr, scaler) or None if target has fewer than 2 classes
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        X_full = df.select_dtypes(include=np.number).iloc[:, :-1]  # Extract all numeric columns except the last as features
        y = df.iloc[:, -1]  # Extract the last column as the target

        if len(np.unique(y)) < 2:  # If target has fewer than 2 unique classes
            print(
                f"{BackgroundColors.RED}AutoML: Target has only one class. Skipping.{Style.RESET_ALL}"
            )  # Output single-class error message
            return None  # Signal caller to abort the pipeline

        X_train_scaled, X_test_scaled, y_train, y_test, scaler, _ = scale_and_split(X_full, y, config=config)  # Scale features and split into train/test sets

        y_train_arr = np.asarray(y_train)  # Convert training target to numpy array for model compatibility
        y_test_arr = np.asarray(y_test)  # Convert test target to numpy array for model compatibility

        return (X_train_scaled, X_test_scaled, y_train_arr, y_test_arr, scaler)  # Return prepared training data tuple
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def print_automl_pipeline_header(file):
    """
    Prints the formatted header block for the AutoML pipeline execution.

    :param file: Path to the dataset file whose basename is shown in the header
    :return: None
    """

    try:
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*80}{Style.RESET_ALL}"
        )  # Print top separator line
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}AutoML Pipeline - {BackgroundColors.CYAN}{os.path.basename(file)}{Style.RESET_ALL}"
        )  # Print pipeline title with dataset filename
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*80}{Style.RESET_ALL}\n"
        )  # Print bottom separator line
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_automl_pipeline(file, df, feature_names, data_source_label="Original", config=None):
    """
    Runs the complete AutoML pipeline: model search, stacking optimization, evaluation, and export.

    :param file: Path to the dataset file being processed
    :param df: Preprocessed DataFrame with features and target
    :param feature_names: List of feature column names
    :param data_source_label: Label identifying the data source
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dictionary containing AutoML results, or None on failure
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        print_automl_pipeline_header(file)  # Print AutoML pipeline header with dataset filename

        automl_start = time.time()  # Record pipeline start time

        training_data = prepare_automl_training_data(df, config=config)  # Prepare scaled training and test splits, returns None for single-class targets
        if training_data is None:  # If preparation failed due to insufficient classes
            return None  # Abort pipeline early

        X_train_scaled, X_test_scaled, y_train_arr, y_test_arr, scaler = training_data  # Unpack prepared training data tuple

        send_telegram_message(TELEGRAM_BOT, f"Starting AutoML pipeline for {os.path.basename(file)}")  # Notify via Telegram

        search_result = run_automl_model_search_and_eval(X_train_scaled, y_train_arr, X_test_scaled, y_test_arr, file)  # Run model search and evaluate the best individual model
        if search_result is None:  # If model search or evaluation failed
            return None  # Abort pipeline

        best_model_name, best_params, model_study, best_individual_model, individual_metrics = search_result  # Unpack search and evaluation results

        stacking_config, stacking_metrics, stacking_study = run_automl_stacking_phase(
            X_train_scaled, y_train_arr, X_test_scaled, y_test_arr, model_study, file, config=config
        )  # Phase 2: Run stacking search and evaluation if enabled

        dataset_root = resolve_dataset_root_path(str(file))  # Resolve AutoML artifact root from file or directory input.
        automl_output_dir = str(dataset_root / "Feature_Analysis" / "AutoML")  # Build AutoML output directory from dataset root.

        export_automl_pipeline_artifacts(
            model_study, stacking_study, stacking_config, stacking_metrics,
            best_model_name, best_params, individual_metrics, best_individual_model,
            scaler, file, feature_names, X_train_scaled, y_train_arr, automl_output_dir, config=config,
        )  # Export all AutoML artifacts (search history, config, models)

        results_list = build_automl_results_list(
            best_model_name, best_params, individual_metrics, stacking_metrics, stacking_config,
            file, feature_names, len(y_train_arr), len(y_test_arr), config=config
        )  # Build results list for CSV

        save_automl_results(file, results_list, config=config)  # Save AutoML results to CSV

        automl_elapsed = time.time() - automl_start  # Calculate total AutoML pipeline time

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}AutoML pipeline completed in {BackgroundColors.CYAN}{calculate_execution_time(0, automl_elapsed)}{Style.RESET_ALL}"
        )  # Output completion message

        send_telegram_message(
            TELEGRAM_BOT, f"AutoML pipeline completed for {os.path.basename(file)} in {calculate_execution_time(0, automl_elapsed)}. Best model: {best_model_name} (F1: {individual_metrics['f1_score']})"
        )  # Send Telegram notification with raw F1 float precision

        return {  # Return AutoML results summary
            "best_model_name": best_model_name,  # Best model name
            "best_params": best_params,  # Best parameters
            "individual_metrics": individual_metrics,  # Individual model metrics
            "stacking_config": stacking_config,  # Stacking configuration
            "stacking_metrics": stacking_metrics,  # Stacking metrics
        }  # Return results dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def sanitize_and_verify_feature_selections(ga_selected_features, rfe_selected_features, feature_names, config=None):
    """
    Sanitize and verify GA and RFE feature selections against available features.

    :param ga_selected_features: Features selected by genetic algorithm
    :param rfe_selected_features: Features selected by RFE
    :param feature_names: List of available feature names
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple of (sanitized_ga_features, sanitized_rfe_features)
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        if ga_selected_features:  # If GA features are provided
            ga_selected_features = sanitize_feature_names(ga_selected_features)  # Sanitize GA feature names
        if rfe_selected_features:  # If RFE features are provided
            rfe_selected_features = sanitize_feature_names(rfe_selected_features)  # Sanitize RFE feature names

        try:  # Verify GA features exist in dataset
            if ga_selected_features:  # If GA features remain after sanitization
                ga_selected_features = verify_selected_features_exist(ga_selected_features, feature_names, "GA")  # Verify GA features exist
        except ValueError as e:  # All GA features missing from dataset
            verbose_output(str(e), config=config)  # Log warning about missing GA features
            ga_selected_features = []  # Reset GA features to empty list

        try:  # Verify RFE features exist in dataset
            if rfe_selected_features:  # If RFE features remain after sanitization
                rfe_selected_features = verify_selected_features_exist(rfe_selected_features, feature_names, "RFE")  # Verify RFE features exist
        except ValueError as e:  # All RFE features missing from dataset
            verbose_output(str(e), config=config)  # Log warning about missing RFE features
            rfe_selected_features = []  # Reset RFE features to empty list

        return ga_selected_features, rfe_selected_features  # Return cleaned and verified feature selections
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def prepare_evaluation_data_splits(df, config=None):
    """
    Prepare original-only training and testing data splits.

    :param df: DataFrame with the original dataset
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple of (X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder) or None if single-class target
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        feature_df = df.iloc[:, :-1]  # Build feature view excluding the target column positionally to avoid drop-copy amplification
        X_full = feature_df.select_dtypes(include=np.number)  # Select only numeric feature columns from the feature view
        y = df.iloc[:, -1]  # Extract target column as the last column

        del feature_df  # Release temporary feature DataFrame view reference before split/scaling
        gc.collect()  # Force garbage collection to reclaim memory from released feature view references

        if len(np.unique(y)) < 2:  # Verify if there is more than one class
            print(
                f"{BackgroundColors.RED}Target column has only one class. Cannot perform classification. Skipping.{Style.RESET_ALL}"
            )  # Output the error message
            return None  # Return None to signal classification is not possible

        X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder = scale_and_split(
            X_full, y, config=config
        )  # Split and fit preprocessing on original training data only.

        del X_full, y  # Release original split inputs after scale_and_split returns
        gc.collect()  # Force garbage collection to reclaim memory from released original split inputs

        return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder  # Return the prepared original-only data splits and preprocessing.
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_evaluation_stacking_model(base_models, config=None):
    """
    Build a StackingClassifier from base models excluding SVM.

    :param base_models: Dictionary of base models
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: StackingClassifier instance
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        estimators = [
            (name, model) for name, model in base_models.items() if name != "SVM"
        ]  # Define estimators list excluding SVM which is incompatible with stacking

        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=config.get("evaluation", {}).get("n_jobs", 1)),
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
            n_jobs=config.get("evaluation", {}).get("n_jobs", 1),
        )  # Define the Stacking Classifier model with sequential CV folds to prevent nested loky deadlock

        return stacking_model  # Return the constructed stacking model
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def assemble_feature_sets(X_train_scaled, X_test_scaled, feature_names, ga_selected_features, pca_n_components, rfe_selected_features, file, feature_sets_config=None, config=None):
    """
    Build feature sets dictionary from configurable feature selection strategies.

    Supports an additive explicit feature set alongside independent per-strategy
    toggles (full, GA, PCA, RFE). When explicit_features is non-empty it is
    included as an additional entry; all strategy toggles remain fully independent.
    To evaluate ONLY the explicit feature set the caller must disable all toggles.

    :param X_train_scaled: Scaled training features
    :param X_test_scaled: Scaled test features
    :param feature_names: List of all feature names
    :param ga_selected_features: GA selected features
    :param pca_n_components: Number of PCA components
    :param rfe_selected_features: RFE selected features
    :param file: Path to dataset file (for PCA artifact lookup)
    :param feature_sets_config: Optional dict controlling which strategies to use
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Sorted dictionary mapping feature set names to (X_train, X_test, feature_names) tuples
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        if feature_sets_config is None:  # If no feature sets config provided
            feature_sets_config = {}  # Use empty dict as fallback

        explicit_features = feature_sets_config.get("explicit_features", []) or []  # Retrieve explicit features list (empty list if not set or None)
        use_full = feature_sets_config.get("use_full", True)  # Toggle for full features strategy (default: enabled)
        use_ga = feature_sets_config.get("use_ga", True)  # Toggle for GA features strategy (default: enabled)
        use_pca = feature_sets_config.get("use_pca", True)  # Toggle for PCA features strategy (default: enabled)
        use_rfe = feature_sets_config.get("use_rfe", True)  # Toggle for RFE features strategy (default: enabled)

        verbose_output(
            f"{BackgroundColors.GREEN}Feature strategies enabled: Full={use_full}, PCA={use_pca}, RFE={use_rfe}, GA={use_ga}.{Style.RESET_ALL}", config=config
        )  # Log which feature strategies are active for this evaluation

        feature_sets = {}  # Initialize empty feature sets dictionary
        feature_signatures = set()  # Track semantic feature-set identities to prevent duplicate equivalent runs

        if use_full:  # Include full features set before optional subsets so it remains the baseline
            full_signature = ("features", tuple(sorted(sanitize_feature_name(f) for f in feature_names)))  # Build order-insensitive full-feature identity
            feature_sets["Full Features"] = (X_train_scaled, X_test_scaled, feature_names)  # Add all features with names as the baseline mode
            feature_signatures.add(full_signature)  # Register full-feature identity before optional feature modes

        if explicit_features:  # If explicit feature set is provided as an additive strategy
            verbose_output(
                f"{BackgroundColors.GREEN}Explicit feature set enabled with {len(explicit_features)} feature(s).{Style.RESET_ALL}", config=config
            )  # Log explicit feature set activation with feature count

            feature_names_list = list(feature_names)  # Normalize feature names to list for safe index lookup
            sanitized_col_map = {sanitize_feature_name(col): col for col in feature_names_list}  # Build sanitized-to-original column mapping for whitespace-agnostic comparison
            verbose_output(
                f"{BackgroundColors.GREEN}Sanitized column name mapping built with {BackgroundColors.CYAN}{len(sanitized_col_map)}{BackgroundColors.GREEN} entries for explicit feature comparison.{Style.RESET_ALL}", config=config
            )  # Log sanitized mapping size for diagnostics

            valid_explicit = []  # Accumulate valid explicit features mapped to original dataset column names
            missing_explicit = []  # Accumulate explicit feature names not found in dataset columns
            seen_sanitized = set()  # Track sanitized keys already resolved to prevent duplicates
            for ef in explicit_features:  # Iterate over each requested explicit feature for sanitized lookup
                sef = sanitize_feature_name(ef)  # Normalize explicit feature name for comparison
                if sef in sanitized_col_map and sef not in seen_sanitized:  # Verify sanitized match exists and not yet resolved
                    valid_explicit.append(sanitized_col_map[sef])  # Append original dataset column name for this feature
                    seen_sanitized.add(sef)  # Mark sanitized key as seen to prevent duplicates
                elif sef not in sanitized_col_map:  # If no match found even after sanitization
                    missing_explicit.append(ef)  # Record original feature name as missing

            if missing_explicit and not valid_explicit:  # If ALL explicit features are absent from dataset columns
                raise ValueError(
                    f"Explicit features not found in dataset columns: {missing_explicit}"
                )  # Raise with explicit mismatch details to preserve original error semantics

            if missing_explicit and valid_explicit:  # If PARTIAL mismatch detected (some valid, some missing)
                print(f"{BackgroundColors.YELLOW}[WARNING] Explicit feature(s) not found in dataset columns and will be skipped: {BackgroundColors.CYAN}{missing_explicit}{Style.RESET_ALL}")  # Log missing explicit features as warning
                print(f"{BackgroundColors.GREEN}Proceeding with {BackgroundColors.CYAN}{len(valid_explicit)}{BackgroundColors.GREEN} valid explicit feature(s): {BackgroundColors.CYAN}{valid_explicit}{Style.RESET_ALL}")  # Log valid features proceeding with original column names

            if valid_explicit:  # If at least one explicit feature resolved to a valid original dataset column
                verbose_output(
                    f"{BackgroundColors.GREEN}Resolved explicit features to original column names: {BackgroundColors.CYAN}{valid_explicit}{Style.RESET_ALL}", config=config
                )  # Log resolved original column names for diagnostics
                explicit_signature = ("features", tuple(sorted(sanitize_feature_name(f) for f in valid_explicit)))  # Build order-insensitive feature identity for duplicate prevention
                if explicit_signature not in feature_signatures:  # Add explicit mode only when it is semantically distinct
                    feature_indices = [feature_names_list.index(f) for f in valid_explicit]  # Map each resolved original feature name to its column index
                    X_train_explicit = X_train_scaled[:, feature_indices]  # Extract training columns matching the resolved explicit feature list
                    X_test_explicit = X_test_scaled[:, feature_indices]  # Extract test columns matching the resolved explicit feature list
                    feature_sets["Explicit Features"] = (X_train_explicit, X_test_explicit, valid_explicit)  # Add explicit feature set using original column names as an additional strategy entry
                    feature_signatures.add(explicit_signature)  # Register explicit feature identity after adding the mode
                else:  # Duplicate explicit feature set
                    print(f"{BackgroundColors.YELLOW}[WARNING] Explicit Features skipped because it is equivalent to an existing feature-set mode.{Style.RESET_ALL}")  # Report duplicate feature-set suppression

        if use_pca:  # Compute PCA transformation only when PCA strategy is enabled
            try:  # Apply PCA transformation while preserving the remaining feature-set modes on failure
                X_train_pca, X_test_pca, _ = apply_pca_transformation(
                    X_train_scaled, X_test_scaled, pca_n_components, file, config=config
                )  # Apply PCA transformation if applicable
            except Exception as e:  # Skip PCA mode when transformation fails
                print(f"{BackgroundColors.YELLOW}[WARNING] PCA Components skipped because transformation failed for {BackgroundColors.CYAN}{file}{BackgroundColors.YELLOW}: {e}{Style.RESET_ALL}")  # Report PCA transformation failure
                X_train_pca, X_test_pca = None, None  # Suppress PCA mode after failed transformation
        else:  # PCA strategy is disabled
            X_train_pca, X_test_pca = None, None  # Skip PCA transformation when strategy is disabled

        if use_ga:  # Compute GA subset only when GA strategy is enabled
            X_train_ga, ga_actual_features = get_feature_subset(X_train_scaled, ga_selected_features, feature_names)  # Get GA feature subset for training
            X_test_ga, _ = get_feature_subset(X_test_scaled, ga_selected_features, feature_names)  # Get GA feature subset for testing
        else:  # GA strategy is disabled
            X_train_ga, X_test_ga, ga_actual_features = None, None, []  # Skip GA subset computation when strategy is disabled

        if use_rfe:  # Compute RFE subset only when RFE strategy is enabled
            X_train_rfe, rfe_actual_features = get_feature_subset(X_train_scaled, rfe_selected_features, feature_names)  # Get RFE feature subset for training
            X_test_rfe, _ = get_feature_subset(X_test_scaled, rfe_selected_features, feature_names)  # Get RFE feature subset for testing
        else:  # RFE strategy is disabled
            X_train_rfe, X_test_rfe, rfe_actual_features = None, None, []  # Skip RFE subset computation when strategy is disabled

        if use_ga and X_train_ga is not None and X_train_ga.shape[1] > 0:  # Include GA subset only when the artifact produced at least one usable feature
            ga_signature = ("features", tuple(sorted(sanitize_feature_name(f) for f in ga_actual_features)))  # Build order-insensitive GA feature identity
            if ga_signature not in feature_signatures:  # Add GA only when it is semantically distinct
                feature_sets["GA Features"] = (X_train_ga, X_test_ga, ga_actual_features)  # GA subset with actual selected names
                feature_signatures.add(ga_signature)  # Register GA feature identity
            else:  # Duplicate GA feature set
                print(f"{BackgroundColors.YELLOW}[WARNING] GA Features skipped because it is equivalent to an existing feature-set mode.{Style.RESET_ALL}")  # Report duplicate feature-set suppression

        if use_pca and X_train_pca is not None and X_test_pca is not None:  # Include PCA components only when transformation produced both matrices
            pca_signature = ("pca", int(X_train_pca.shape[1]))  # Build transformed component-space identity
            if pca_signature not in feature_signatures:  # Add PCA only when this component-space identity is distinct
                feature_sets["PCA Components"] = (X_train_pca, X_test_pca, None)  # PCA components with synthetic names generated later
                feature_signatures.add(pca_signature)  # Register PCA component identity

        if use_rfe and X_train_rfe is not None and X_train_rfe.shape[1] > 0:  # Include RFE subset only when the artifact produced at least one usable feature
            rfe_signature = ("features", tuple(sorted(sanitize_feature_name(f) for f in rfe_actual_features)))  # Build order-insensitive RFE feature identity
            if rfe_signature not in feature_signatures:  # Add RFE only when it is semantically distinct
                feature_sets["RFE Features"] = (X_train_rfe, X_test_rfe, rfe_actual_features)  # RFE subset with actual selected names
                feature_signatures.add(rfe_signature)  # Register RFE feature identity
            else:  # Duplicate RFE feature set
                print(f"{BackgroundColors.YELLOW}[WARNING] RFE Features skipped because it is equivalent to an existing feature-set mode.{Style.RESET_ALL}")  # Report duplicate feature-set suppression

        feature_sets = {
            k: v for k, v in feature_sets.items() if v is not None
        }  # Remove any None entries (e.g., PCA if not applied)
        feature_sets = dict(sorted(feature_sets.items(), key=lambda item: (item[0] != "Full Features", item[0])))  # Keep the required full-feature baseline first, then sort implemented selection modes

        verbose_output(
            f"{BackgroundColors.GREEN}Feature strategy: Assembled {len(feature_sets)} set(s): {list(feature_sets.keys())}.{Style.RESET_ALL}", config=config
        )  # Log the final list of assembled feature sets

        return feature_sets  # Return the assembled and sorted feature sets dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def iterate_feature_sets_sequentially(feature_source_arrays: dict, feature_names: List[Any], ga_selected_features: Any, pca_n_components: Any, rfe_selected_features: Any, file: str, feature_sets_config: dict, config: Optional[dict], scaler: Any = None, source_files: Optional[List[str]] = None, pca_cache_context: Optional[dict] = None, pca_input_feature_names: Optional[List[Any]] = None) -> Any:  # Yield feature matrices while carrying exact PCA cache provenance.
    """
    Yield feature-set matrices one at a time.

    :param feature_source_arrays: Mutable holder containing the full scaled train/test source matrices.
    :param feature_names: Complete ordered feature-name list.
    :param ga_selected_features: GA-selected feature list or None.
    :param pca_n_components: PCA component count or None.
    :param rfe_selected_features: RFE-selected feature list or None.
    :param file: Dataset path used for PCA artifact loading.
    :param feature_sets_config: Feature-set strategy configuration.
    :param config: Runtime configuration dictionary.
    :param scaler: Fitted scaler that produced the full scaled source matrices.
    :param source_files: Ordered source files used to build this evaluation dataset.
    :param pca_cache_context: Split and experiment identity used for PCA cache validation.
    :param pca_input_feature_names: Exact ordered numeric feature names consumed by PCA.
    :return: Iterator yielding feature-set matrices, names, and fitted feature transformer.
    """

    if config is None:  # Use global configuration when no explicit configuration is supplied.
        config = CONFIG  # Assign global configuration reference.
    if feature_sets_config is None:  # Use an empty feature-set configuration when absent.
        feature_sets_config = {}  # Assign empty configuration mapping.

    explicit_features = feature_sets_config.get("explicit_features", []) or []  # Retrieve optional explicit feature list.
    use_full = feature_sets_config.get("use_full", True)  # Resolve full-feature strategy toggle.
    use_ga = feature_sets_config.get("use_ga", True)  # Resolve GA strategy toggle.
    use_pca = feature_sets_config.get("use_pca", True)  # Resolve PCA strategy toggle.
    use_rfe = feature_sets_config.get("use_rfe", True)  # Resolve RFE strategy toggle.
    feature_signatures: set[Tuple[str, Any]] = set()  # Track semantic feature-set identities for duplicate suppression.
    pending_mode_count = len(list_grid_feature_modes(ga_selected_features, pca_n_components, rfe_selected_features, feature_names, config=config))  # Count modes to decide whether source matrices are still needed after Full Features.

    if use_full:  # Yield full features first to preserve baseline ordering.
        full_signature = ("features", tuple(sorted(sanitize_feature_name(feature) for feature in feature_names)))  # Build full-feature identity.
        feature_signatures.add(full_signature)  # Register full-feature identity before optional modes.
        yield "Full Features", feature_source_arrays["X_train_scaled"], feature_source_arrays["X_test_scaled"], feature_names, None  # Yield full-feature matrices without copying.
        if pending_mode_count > 1:  # If later feature modes need the full source matrices, spill them before materializing subset copies.
            maybe_spill_feature_source_arrays(feature_source_arrays, file, config=config)  # Replace large in-memory full matrices with disk-backed memmaps when configured.

    if explicit_features:  # Resolve explicit feature mode only when configured.
        feature_names_list = list(feature_names)  # Normalize feature names for positional lookup.
        sanitized_col_map = {sanitize_feature_name(column): column for column in feature_names_list}  # Build sanitized feature lookup.
        valid_explicit = []  # Accumulate valid explicit features in requested order.
        missing_explicit = []  # Accumulate explicit features absent from the dataset.
        seen_sanitized = set()  # Track already emitted sanitized feature names.
        for explicit_feature in explicit_features:  # Iterate requested explicit features.
            sanitized_feature = sanitize_feature_name(explicit_feature)  # Normalize explicit feature name for lookup.
            if sanitized_feature in sanitized_col_map and sanitized_feature not in seen_sanitized:  # Verify the feature is present and not duplicated.
                valid_explicit.append(sanitized_col_map[sanitized_feature])  # Add the original dataset column name.
                seen_sanitized.add(sanitized_feature)  # Mark sanitized feature as emitted.
            elif sanitized_feature not in sanitized_col_map:  # Detect requested features absent from the dataset.
                missing_explicit.append(explicit_feature)  # Store missing explicit feature name.
        if missing_explicit and not valid_explicit:  # Preserve existing explicit-feature failure behavior.
            raise ValueError(f"Explicit features not found in dataset columns: {missing_explicit}")  # Raise when every explicit feature is absent.
        if missing_explicit and valid_explicit:  # Report partial explicit-feature mismatch.
            print(f"{BackgroundColors.YELLOW}[WARNING] Explicit feature(s) not found in dataset columns and will be skipped: {BackgroundColors.CYAN}{missing_explicit}{Style.RESET_ALL}")  # Log missing explicit features.
            print(f"{BackgroundColors.GREEN}Proceeding with {BackgroundColors.CYAN}{len(valid_explicit)}{BackgroundColors.GREEN} valid explicit feature(s): {BackgroundColors.CYAN}{valid_explicit}{Style.RESET_ALL}")  # Log valid explicit features.
        if valid_explicit:  # Yield explicit mode only when at least one feature is valid.
            explicit_signature = ("features", tuple(sorted(sanitize_feature_name(feature) for feature in valid_explicit)))  # Build explicit feature identity.
            if explicit_signature not in feature_signatures:  # Suppress duplicate explicit mode.
                X_train_explicit, explicit_actual_features = get_feature_subset(feature_source_arrays["X_train_scaled"], valid_explicit, feature_names)  # Materialize explicit training subset for this mode.
                X_test_explicit, _ = get_feature_subset(feature_source_arrays["X_test_scaled"], valid_explicit, feature_names)  # Materialize explicit test subset for this mode.
                feature_signatures.add(explicit_signature)  # Register explicit feature identity.
                try:
                    yield "Explicit Features", X_train_explicit, X_test_explicit, explicit_actual_features, None  # Yield explicit feature matrices.
                finally:
                    del X_train_explicit, X_test_explicit  # Release explicit subset arrays after caller finishes or aborts this mode.
                    gc.collect()  # Reclaim explicit subset memory before the next feature mode.
            else:  # Report duplicate explicit mode suppression.
                print(f"{BackgroundColors.YELLOW}[WARNING] Explicit Features skipped because it is equivalent to an existing feature-set mode.{Style.RESET_ALL}")  # Log duplicate explicit mode.

    if use_ga:  # Resolve GA feature mode after explicit features.
        X_train_ga, ga_actual_features = get_feature_subset(feature_source_arrays["X_train_scaled"], ga_selected_features, feature_names)  # Materialize GA training subset for this mode.
        X_test_ga, _ = get_feature_subset(feature_source_arrays["X_test_scaled"], ga_selected_features, feature_names)  # Materialize GA test subset for this mode.
        if X_train_ga is not None and X_train_ga.shape[1] > 0:  # Yield GA only when at least one feature exists.
            ga_signature = ("features", tuple(sorted(sanitize_feature_name(feature) for feature in ga_actual_features)))  # Build GA feature identity.
            if ga_signature not in feature_signatures:  # Suppress duplicate GA mode.
                feature_signatures.add(ga_signature)  # Register GA feature identity.
                try:
                    yield "GA Features", X_train_ga, X_test_ga, ga_actual_features, None  # Yield GA feature matrices.
                finally:
                    del X_train_ga, X_test_ga  # Release GA subset arrays after caller finishes or aborts this mode.
                    gc.collect()  # Reclaim GA subset memory before the next feature mode.
            else:  # Report duplicate GA mode suppression.
                print(f"{BackgroundColors.YELLOW}[WARNING] GA Features skipped because it is equivalent to an existing feature-set mode.{Style.RESET_ALL}")  # Log duplicate GA mode.
                del X_train_ga, X_test_ga  # Release duplicate GA subset arrays immediately.
                gc.collect()  # Reclaim duplicate GA subset memory.

    if use_pca:  # Resolve PCA feature mode after GA to preserve sorted evaluation order.
        try:  # Preserve existing PCA failure tolerance.
            X_train_pca, X_test_pca, pca_transformer = apply_pca_transformation(feature_source_arrays["X_train_scaled"], feature_source_arrays["X_test_scaled"], pca_n_components, file, config=config, feature_names=pca_input_feature_names if pca_input_feature_names is not None else feature_names, scaler=scaler, source_files=source_files, cache_context=pca_cache_context)  # Materialize PCA matrices with strict source, scaling, and split provenance.
        except Exception as e:  # Skip PCA mode when transformation fails.
            print(f"{BackgroundColors.YELLOW}[WARNING] PCA Components skipped because transformation failed for {BackgroundColors.CYAN}{file}{BackgroundColors.YELLOW}: {e}{Style.RESET_ALL}")  # Log PCA transformation failure.
            X_train_pca, X_test_pca, pca_transformer = None, None, None  # Suppress PCA mode after failure.
        if X_train_pca is not None and X_test_pca is not None:  # Yield PCA only when both matrices exist.
            pca_signature = ("pca", int(X_train_pca.shape[1]))  # Build PCA component identity.
            if pca_signature not in feature_signatures:  # Suppress duplicate PCA mode.
                feature_signatures.add(pca_signature)  # Register PCA component identity.
                try:
                    print(f"{BackgroundColors.GREEN}[INFO] Starting model evaluation on PCA Components with {BackgroundColors.CYAN}{X_train_pca.shape[1]}{BackgroundColors.GREEN} features.{Style.RESET_ALL}")  # Mark the boundary between PCA transformation and classifier evaluation.
                    yield "PCA Components", X_train_pca, X_test_pca, None, pca_transformer  # Yield PCA matrices with synthetic names and fitted preprocessing.
                finally:
                    del X_train_pca, X_test_pca  # Release PCA matrices after caller finishes or aborts this mode.
                    gc.collect()  # Reclaim PCA memory before the next feature mode.
            else:
                del X_train_pca, X_test_pca  # Release duplicate PCA matrices immediately.
                gc.collect()  # Reclaim duplicate PCA memory.

    if use_rfe:  # Resolve RFE feature mode last to preserve sorted evaluation order.
        X_train_rfe, rfe_actual_features = get_feature_subset(feature_source_arrays["X_train_scaled"], rfe_selected_features, feature_names)  # Materialize RFE training subset for this mode.
        X_test_rfe, _ = get_feature_subset(feature_source_arrays["X_test_scaled"], rfe_selected_features, feature_names)  # Materialize RFE test subset for this mode.
        if X_train_rfe is not None and X_train_rfe.shape[1] > 0:  # Yield RFE only when at least one feature exists.
            rfe_signature = ("features", tuple(sorted(sanitize_feature_name(feature) for feature in rfe_actual_features)))  # Build RFE feature identity.
            if rfe_signature not in feature_signatures:  # Suppress duplicate RFE mode.
                feature_signatures.add(rfe_signature)  # Register RFE feature identity.
                try:
                    yield "RFE Features", X_train_rfe, X_test_rfe, rfe_actual_features, None  # Yield RFE feature matrices.
                finally:
                    del X_train_rfe, X_test_rfe  # Release RFE subset arrays after caller finishes or aborts this mode.
                    gc.collect()  # Reclaim RFE subset memory.
            else:  # Report duplicate RFE mode suppression.
                print(f"{BackgroundColors.YELLOW}[WARNING] RFE Features skipped because it is equivalent to an existing feature-set mode.{Style.RESET_ALL}")  # Log duplicate RFE mode.
                del X_train_rfe, X_test_rfe  # Release duplicate RFE subset arrays immediately.
                gc.collect()  # Reclaim duplicate RFE subset memory.


def build_classifier_result_entry(model_class, file, execution_mode_str, attack_types_combined, feature_set_name, classifier_type, model_name, data_source_label, experiment_id, experiment_mode, augmentation_ratio, n_features, n_samples_train, n_samples_test, metrics_tuple, subset_feature_names, hyperparams_map=None, hyperparameters_enabled=False, effective_hyperparameters=None):  # Build a standardized classifier result entry.
    """
    Build a standardized result entry dictionary for classifier evaluation results.

    :param model_class: Class name of the model
    :param file: Path to the dataset file
    :param execution_mode_str: Execution mode string ('separate_files' or 'combined_files')
    :param attack_types_combined: List of attack types for combined files evaluation or None
    :param feature_set_name: Name of the feature set used
    :param classifier_type: Type of classifier ('Individual' or 'Stacking')
    :param model_name: Name of the model
    :param data_source_label: Label for the data source
    :param experiment_id: Unique experiment identifier
    :param experiment_mode: Experiment mode string
    :param augmentation_ratio: Augmentation ratio or None
    :param n_features: Number of features used
    :param n_samples_train: Number of training samples
    :param n_samples_test: Number of test samples
    :param metrics_tuple: Tuple of (accuracy, precision, recall, f1, fpr, fnr, elapsed, ...)
    :param subset_feature_names: List of feature names used
    :param hyperparams_map: Dictionary mapping model names to hyperparameters
    :param effective_hyperparameters: Effective parameters read from the estimator evaluated for this row
    :return: Dictionary containing the result entry
    """

    try:
        acc, prec, rec, f1, fpr, fnr, elapsed = metrics_tuple[:7]  # Unpack the first 7 metrics from the tuple
        dataset_identity = resolve_canonical_dataset_identity(file, True) if execution_mode_str == "combined_files" else os.path.relpath(file)  # Resolve result-row dataset identity by execution mode.
        persisted_augmentation_ratio = resolve_persisted_augmentation_ratio(experiment_mode, augmentation_ratio)  # Resolve explicit baseline or augmented ratio metadata.
        serialized_hyperparameters = serialize_result_hyperparameters(model_name, hyperparams_map=hyperparams_map, effective_hyperparameters=effective_hyperparameters)  # Resolve effective estimator parameters for CSV persistence.
        return {
            "model": model_class,  # Model class name for identification
            "dataset": dataset_identity,  # Store the resolved dataset identity for CSV exports
            "execution_mode": execution_mode_str,  # Execution mode (separate_files or combined_files)
            "attack_types_combined": json.dumps(attack_types_combined) if attack_types_combined else None,  # JSON-serialized attack types or None
            "feature_set": feature_set_name,  # Name of the feature set evaluated
            "hyperparameter_mode": "Optimized Hyperparameters" if hyperparameters_enabled else "Default Hyperparameters",  # Explicit HP mode for result separation and resume safety
            "classifier_type": classifier_type,  # Classifier type (Individual or Stacking)
            "model_name": model_name,  # Model name for result identification
            "data_source": data_source_label,  # Data source label for experiment traceability
            "experiment_id": experiment_id,  # Unique experiment identifier
            "experiment_mode": experiment_mode,  # Persist original-only or augmented-testing semantics.
            "augmentation_ratio": persisted_augmentation_ratio,  # Augmentation ratio with 0.0 for original-only rows
            "n_features": n_features,  # Number of features used in evaluation
            "n_samples_train": n_samples_train,  # Number of training samples
            "n_samples_test": n_samples_test,  # Number of test samples
            "accuracy": acc,  # Accuracy as raw float
            "precision": prec,  # Precision as raw float
            "recall": rec,  # Recall as raw float
            "f1_score": f1,  # F1 score as raw float
            "fpr": fpr,  # False positive rate as raw float
            "fnr": fnr,  # False negative rate as raw float
            "elapsed_time_s": int(round(elapsed)),  # Rounded elapsed time in seconds
            "cv_method": f"StratifiedKFold(n_splits=10)",  # Cross-validation method description
            "top_features": json.dumps(subset_feature_names),  # JSON-serialized subset feature names
            "rfe_ranking": None,  # RFE ranking placeholder (not computed here)
            "hyperparameters": serialized_hyperparameters,  # JSON-serialized effective estimator hyperparameters
            "features_list": subset_feature_names,  # Raw list of feature names for downstream use
        }  # Return the constructed result entry dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_telegram_combination_header(name, model_name, augmentation_ratio=None, hyperparameters_enabled=False):
    """
    Build a Telegram combination header reflecting the current evaluation configuration.

    :param name: Feature set name for the current evaluation.
    :param model_name: Model name being evaluated.
    :param augmentation_ratio: Augmentation ratio float or None when augmentation is not active.
    :param hyperparameters_enabled: Whether optimized hyperparameters are active for this run.
    :return: Human-readable combination header string.
    """
    
    feature_label = "PCA" if name == "PCA Components" else name  # Normalize PCA wording for compact progress labels
    hyperparameter_label = "Optimized Hyperparameters" if hyperparameters_enabled else "Default Hyperparameters"  # Build explicit HP mode label
    augmentation_label = f"Augmented Test Ratio = {augmentation_ratio:.2f}" if augmentation_ratio is not None else "Original Test Data"  # Build the isolated testing-mode label.
    parts = [feature_label, hyperparameter_label, augmentation_label, model_name]  # Build full progress label in the expected order
    
    return " - ".join(parts)


def submit_classifier_evaluations_to_pool(executor, individual_models, current_combination, name, X_train_df, y_train, X_test_df, y_test, file, scaler, subset_feature_names, total_steps, augmentation_ratio=None, hyperparameters_enabled=False):
    """
    Submit each individual classifier to the thread pool executor and return the futures map alongside the updated combination counter.

    :param executor: Active ThreadPoolExecutor instance to submit evaluation tasks to.
    :param individual_models: Dictionary mapping model names to model instances.
    :param current_combination: Current global combination counter before submission begins.
    :param name: Name of the current feature set being evaluated.
    :param X_train_df: Training feature DataFrame with named columns.
    :param y_train: Training target labels.
    :param X_test_df: Test feature DataFrame with named columns.
    :param y_test: Test target labels.
    :param file: Path to the dataset file used for evaluation.
    :param scaler: Fitted scaler for dataset preprocessing.
    :param subset_feature_names: List of feature names for the current subset.
    :param total_steps: Total number of evaluation steps for Telegram progress messages.
    :return: Tuple of (future_to_model, current_combination) where future_to_model maps each submitted future to its (model_name, model_class, combination_index) metadata.
    """

    future_to_model = {}  # Map futures to (model_name, model_class, combination_index)
    for model_name, model in individual_models.items():  # Iterate over each individual model
        artifact_feature_set = f"{name} - {'Optimized Hyperparameters' if hyperparameters_enabled else 'Default Hyperparameters'}"  # Keep exported/resumed model artifacts isolated by HP mode
        future = executor.submit(
            evaluate_individual_classifier,
            model,
            model_name,
            X_train_df.values,
            y_train,
            X_test_df.values,
            y_test,
            file,
            scaler,
            subset_feature_names,
            artifact_feature_set,
        )  # Submit evaluation task to thread pool using numpy arrays
        future_to_model[future] = (model_name, model.__class__.__name__, current_combination)  # Store future with metadata
        current_combination += 1  # Advance the global combination counter
    return future_to_model, current_combination  # Return futures map and updated combination counter


def collect_classifier_results_from_futures(future_to_model, individual_models, name, X_test_subset, X_train_n_cols, file, execution_mode_str, attack_types_combined, data_source_label, experiment_id, experiment_mode, augmentation_ratio, y_train, y_test, hyperparams_map, subset_feature_names, total_steps, progress_bar, config):
    """
    Collect results from completed classifier futures, build standardized result entries, and optionally run explainability for each model.

    :param future_to_model: Dictionary mapping futures to (model_name, model_class, combination_index) metadata.
    :param individual_models: Dictionary mapping model names to model instances, used for explainability retrieval.
    :param name: Name of the current feature set being evaluated.
    :param X_test_subset: Test feature array passed to the explainability pipeline.
    :param X_train_n_cols: Number of training columns used for result entry metadata.
    :param file: Path to the dataset file for result metadata and explainability output.
    :param execution_mode_str: Execution mode string ('separate_files' or 'combined_files').
    :param attack_types_combined: List of attack types for combined files evaluation, or None for separate files evaluation.
    :param data_source_label: Label identifying the data source for result traceability.
    :param experiment_id: Unique experiment identifier for traceability.
    :param experiment_mode: Experiment mode string ('original_only' or 'original_training_augmented_testing').
    :param augmentation_ratio: Augmentation ratio float, or None for original-only experiments.
    :param y_train: Training target labels used to compute training set size.
    :param y_test: Test target labels used for evaluation and size metadata.
    :param hyperparams_map: Dictionary mapping model names to their hyperparameter dictionaries.
    :param subset_feature_names: List of feature names for the current subset.
    :param total_steps: Total number of evaluation steps for Telegram progress messages.
    :param progress_bar: tqdm progress bar advanced after each completed future.
    :param config: Configuration dictionary used for explainability settings and verbose output.
    :return: Dictionary mapping (feature_set_name, model_name) tuples to standardized result entry dicts.
    """

    results_dict = {}  # Accumulate result entries for this feature set
    for future in concurrent.futures.as_completed(future_to_model):  # Process results as futures complete
        model_name, model_class, comb_idx = future_to_model[future]  # Retrieve model metadata for this future
        metrics = future.result()  # Collect metrics tuple from completed evaluation
        result_entry = build_classifier_result_entry(
            model_class, file, execution_mode_str, attack_types_combined, name, "Individual",
            model_name, data_source_label, experiment_id, experiment_mode, augmentation_ratio,
            X_train_n_cols, len(y_train), len(y_test), metrics, subset_feature_names,
            hyperparams_map=hyperparams_map, hyperparameters_enabled=bool(hyperparams_map),
            effective_hyperparameters=serialize_effective_estimator_parameters(individual_models[model_name]),
        )  # Build standardized result entry for this individual classifier
        results_dict[(name, model_name)] = result_entry  # Store result keyed by (feature_set, model_name)
        progress_bar.update(1)  # Advance progress bar by one step

        if config.get("explainability", {}).get("enabled", False) and experiment_mode == "original_only":  # Only queue explainability on original data
            try:  # Attempt to dispatch explainability for this model.
                trained_model = individual_models[model_name]  # Retrieve trained model object for snapshotting
                schedule_explainability_job(trained_model, model_name, X_test_subset, y_test, subset_feature_names, file, name, execution_mode_str, config, experiment_mode, bool(hyperparams_map), training_ram_stats=None)  # Dispatch synchronous explainability when legacy future path lacks RAM statistics.
            except Exception as e:  # If explainability queueing fails
                verbose_output(
                    f"{BackgroundColors.YELLOW}Explainability failed for {model_name}: {e}{Style.RESET_ALL}",
                    config=config
                )  # Log explainability failure before propagation.
                raise  # Preserve explainability failure propagation.
    return results_dict  # Return accumulated result entries


def recover_cached_individual_classifier_result(cache_dict, execution_mode_str, data_source_label, experiment_mode, augmentation_ratio, attack_types_combined, feature_set_name, model_name, results_dict, current_combination, total_steps, progress_bar, hyperparameters_enabled=False, expected_n_features=None, expected_feature_names=None, expected_n_samples_train=None, expected_n_samples_test=None):  # Recover a cached classifier result only when its evaluated data shape matches.
    """
    Recover one cached individual-classifier result (if present), emit resume logs, and advance progress counters.

    :param cache_dict: Dictionary of cached result entries keyed by resume cache key.
    :param execution_mode_str: Current execution mode string used to build resume cache keys.
    :param data_source_label: Data source label used to build resume cache keys.
    :param experiment_mode: Experiment mode string used to build resume cache keys.
    :param augmentation_ratio: Augmentation ratio used to build resume cache keys.
    :param attack_types_combined: Attack types list for combined mode key disambiguation.
    :param feature_set_name: Current feature set name used to build resume cache keys.
    :param model_name: Current model name used to build resume cache keys.
    :param results_dict: Mutable result dictionary to receive recovered entries.
    :param current_combination: Current combination counter value.
    :param total_steps: Total number of evaluation steps for progress messaging.
    :param progress_bar: tqdm progress bar instance to advance when recovered.
    :param hyperparameters_enabled: Whether the active evaluation uses optimized hyperparameters.
    :param expected_n_features: Feature count required for safe cache recovery.
    :param expected_feature_names: Ordered feature names required for safe cache recovery.
    :param expected_n_samples_train: Training sample count required for safe cache recovery.
    :param expected_n_samples_test: Test sample count required for safe cache recovery.
    :return: Tuple (recovered, next_current_combination) where recovered indicates if cache was hit.
    """

    if not cache_dict:  # Skip recovery when cache dictionary is empty or unavailable
        return (False, current_combination)  # Signal no cache hit and unchanged counter

    resume_key = build_resume_cache_key(
        execution_mode_str,
        data_source_label,
        experiment_mode,
        augmentation_ratio,
        attack_types_combined,
        feature_set_name,
        model_name,
        hyperparameters_enabled,
    )  # Build full resume cache key for this evaluation unit

    if resume_key not in cache_dict:  # Skip recovery when no cached result exists for this key
        return (False, current_combination)  # Signal cache miss and unchanged counter

    cached_result = cache_dict[resume_key]  # Retrieve cached result entry for this classifier
    cached_n_features = cached_result.get("n_features", None)  # Read the feature count persisted with the cached evaluation.
    cached_feature_names = cached_result.get("features_list", None)  # Read the ordered feature names persisted with the cached evaluation.
    cached_n_samples_train = cached_result.get("n_samples_train", None)  # Read the cached training sample count.
    cached_n_samples_test = cached_result.get("n_samples_test", None)  # Read the cached test sample count.
    normalized_expected_features = [str(feature) for feature in expected_feature_names] if expected_feature_names is not None else None  # Normalize active feature names for deterministic comparison.
    normalized_cached_features = [str(feature) for feature in cached_feature_names] if isinstance(cached_feature_names, list) else None  # Normalize cached feature names when the payload is a list.
    feature_count_matches = expected_n_features is None or (cached_n_features is not None and int(cached_n_features) == int(expected_n_features))  # Require the active PCA dimensionality or subset width to match.
    feature_names_match = normalized_expected_features is None or normalized_cached_features == normalized_expected_features  # Require the exact ordered feature identity to match.
    train_count_matches = expected_n_samples_train is None or (cached_n_samples_train is not None and int(cached_n_samples_train) == int(expected_n_samples_train))  # Require the active training sample count to match.
    test_count_matches = expected_n_samples_test is None or (cached_n_samples_test is not None and int(cached_n_samples_test) == int(expected_n_samples_test))  # Require the active test sample count to match.

    if not all((feature_count_matches, feature_names_match, train_count_matches, test_count_matches)):  # Reject stale cache rows that share a broad resume key but represent different evaluated data.
        del cache_dict[resume_key]  # Remove the stale in-memory identity so the recomputed result can be persisted.
        print(f"{BackgroundColors.YELLOW}[RESUME] Ignored incompatible cached combination for {BackgroundColors.CYAN}{feature_set_name} - {model_name}{BackgroundColors.YELLOW}: active shape or feature identity differs from saved partial progress.{Style.RESET_ALL}")  # Explain why this classifier will be recomputed.
        return (False, current_combination)  # Signal cache rejection without advancing the progress counter.

    results_dict[(feature_set_name, model_name)] = cached_result  # Reuse cached entry without recomputation
    combination_header = build_telegram_combination_header(feature_set_name, model_name, augmentation_ratio, hyperparameters_enabled)  # Build full recovered combination label
    print(
        f"{BackgroundColors.YELLOW}[RESUME] Recovered combination {current_combination}/{total_steps}: {combination_header} from saved partial progress (no recomputation performed).{Style.RESET_ALL}"
    )  # Log recovered combination to stdout for visibility

    cached_execution_mode = cached_result.get("execution_mode", execution_mode_str)  # Resolve execution mode from cached entry for log consistency
    evaluation_mode = str(cached_execution_mode).replace("_", " ").title().replace(" ", "") if cached_execution_mode else "SeparateFiles"  # Normalize execution mode to CamelCase style
    acc = cached_result.get("accuracy", "N/A")  # Recover accuracy from cached result for full metrics log
    prec = cached_result.get("precision", "N/A")  # Recover precision from cached result for full metrics log
    rec = cached_result.get("recall", "N/A")  # Recover recall from cached result for full metrics log
    f1 = cached_result.get("f1_score", "N/A")  # Recover F1 score from cached result for full metrics log
    fpr = cached_result.get("fpr", "N/A")  # Recover FPR from cached result for full metrics log
    fnr = cached_result.get("fnr", "N/A")  # Recover FNR from cached result for full metrics log
    cached_elapsed = cached_result.get("elapsed_time_s", 0)  # Recover elapsed time from cached result for full metrics log
    cached_elapsed = int(round(float(cached_elapsed))) if cached_elapsed is not None else 0  # Normalize elapsed seconds to integer for display
    cached_human_time = calculate_execution_time(cached_elapsed)  # Format elapsed seconds to human-readable duration string
    print(
        f"{BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN}: Mode {BackgroundColors.CYAN}{evaluation_mode}{BackgroundColors.GREEN} | F1-Score {BackgroundColors.CYAN}{f1}{BackgroundColors.GREEN} | Accuracy: {BackgroundColors.CYAN}{acc}{BackgroundColors.GREEN} | Precision: {BackgroundColors.CYAN}{prec}{BackgroundColors.GREEN} | Recall: {BackgroundColors.CYAN}{rec}{BackgroundColors.GREEN} | FPR: {BackgroundColors.CYAN}{fpr}{BackgroundColors.GREEN} | FNR: {BackgroundColors.CYAN}{fnr}{BackgroundColors.GREEN} | Training Time: {BackgroundColors.CYAN}{cached_elapsed}s{BackgroundColors.GREEN} | Execution Time: {BackgroundColors.CYAN}{cached_elapsed}s{BackgroundColors.GREEN} | Total Time: {BackgroundColors.CYAN}{cached_human_time}{BackgroundColors.GREEN} ({BackgroundColors.CYAN}{cached_elapsed}s{BackgroundColors.GREEN}){Style.RESET_ALL}"
    )  # Print a full metrics summary for recovered cached classifier as if freshly evaluated

    progress_bar.update(1)  # Advance progress bar even for skipped cached evaluations
    current_combination += 1  # Advance the global combination counter for skipped evaluations

    return (True, current_combination)  # Signal cache hit and return updated counter


def run_individual_classifiers_for_feature_set(name, individual_models, X_train_df, y_train, X_test_df, y_test, X_test_subset, X_train_n_cols, file, execution_mode_str, attack_types_combined, data_source_label, experiment_id, experiment_mode, augmentation_ratio, hyperparams_map, scaler, label_encoder, transformer, input_feature_names, target_column, source_files, subset_feature_names, total_steps, current_combination, progress_bar, config=None, cache_dict=None, cache_ref_file=None, hyperparameters_enabled=False):
    """
    Evaluates all individual classifiers for a feature set sequentially, collects results, and runs explainability.

    :param name: Name of the current feature set being evaluated
    :param individual_models: Dictionary mapping model names to model objects
    :param X_train_df: Training feature DataFrame with named columns
    :param y_train: Training target labels
    :param X_test_df: Test feature DataFrame with named columns
    :param y_test: Test target labels
    :param X_test_subset: Test feature array for explainability pipeline input
    :param X_train_n_cols: Number of training columns (features) used for result entry metadata
    :param file: Path to the dataset file for export and result metadata
    :param execution_mode_str: Execution mode string ('separate_files' or 'combined_files')
    :param attack_types_combined: List of attack types for combined files evaluation or None for separate files evaluation
    :param data_source_label: Label identifying the data source for result traceability
    :param experiment_id: Unique experiment identifier for traceability
    :param experiment_mode: Experiment mode string ('original_only' or 'original_training_augmented_testing')
    :param augmentation_ratio: Augmentation ratio float or None for original-only experiments
    :param hyperparams_map: Dictionary mapping model names to their hyperparameter dicts
    :param scaler: Fitted scaler used for dataset preprocessing
    :param label_encoder: Label encoder fitted on original training labels
    :param transformer: Optional PCA transformer fitted on original training features
    :param input_feature_names: Ordered numeric schema before preprocessing
    :param target_column: Positional target column name
    :param source_files: Ordered original source files used for model identity
    :param subset_feature_names: List of feature names for the current subset
    :param total_steps: Total number of evaluation steps for Telegram progress messages
    :param current_combination: Current combination index counter for progress messages
    :param progress_bar: tqdm progress bar to update after each model evaluation
    :param config: Configuration dictionary (uses global CONFIG if None)
    :param cache_dict: Dictionary of previously cached results keyed by resume cache key for skip-if-cached logic.
    :param cache_ref_file: File path used when deriving the cache file location for atomic cache writes.
    :return: Tuple (results_dict, next_current_combination) where results_dict maps (name, model_name) to result entries
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        progress_bar.set_description(
            f"{data_source_label} - {name} (Individual)"
        )  # Update progress bar description for individual model evaluations

        results_dict = {}  # Accumulate result entries for this feature set
        X_train_values = X_train_df.to_numpy(copy=False) if hasattr(X_train_df, "to_numpy") else np.asarray(X_train_df)  # Reuse one no-copy train array view for all classifiers in this feature set
        X_test_values = X_test_df.to_numpy(copy=False) if hasattr(X_test_df, "to_numpy") else np.asarray(X_test_df)  # Reuse one no-copy test array view for all classifiers in this feature set

        for model_name, model in individual_models.items():  # Iterate over each individual model sequentially to prevent loky deadlock
            recovered, current_combination = recover_cached_individual_classifier_result(cache_dict, execution_mode_str, data_source_label, experiment_mode, augmentation_ratio, attack_types_combined, name, model_name, results_dict, current_combination, total_steps, progress_bar, hyperparameters_enabled=hyperparameters_enabled, expected_n_features=X_train_n_cols, expected_feature_names=subset_feature_names, expected_n_samples_train=len(y_train), expected_n_samples_test=len(y_test))  # Recover only rows produced with the same active data shape and ordered features.
            if recovered:  # Skip recomputation when cache recovery succeeds
                continue  # Move to next model because this one has already been recovered
            active_model = clone(model)  # Clone the estimator prototype so fitted state is not retained across atomic classifiers
            artifact_feature_set = f"{name} - {'Optimized Hyperparameters' if hyperparameters_enabled else 'Default Hyperparameters'}"  # Resolve HP-isolated artifact feature-set label
            dataset_name = build_filename_safe_dataset_identity(resolve_canonical_dataset_identity(str(file), True)) if execution_mode_str == "combined_files" else os.path.basename(os.path.dirname(file))  # Resolve model export dataset folder by execution mode.
            artifact_context = build_stacking_model_artifact_context(file, source_files, execution_mode_str, attack_types_combined, target_column, model_name, active_model, artifact_feature_set, input_feature_names, subset_feature_names, list(label_encoder.classes_), transformer, hyperparameters_enabled)  # Build ratio-independent original-training artifact identity.
            combination_header = build_telegram_combination_header(name, model_name, augmentation_ratio, hyperparameters_enabled)  # Build full progress label for this classifier
            progress_bar.set_description(combination_header)  # Update progress bar with the complete active configuration
            sys.stdout.flush()  # Flush stdout before each classifier to ensure logs are visible under nohup
            phase_params_digest = get_classifier_params_digest(active_model)  # Build compact classifier parameter digest
            phase_cache_key = build_resume_cache_key(execution_mode_str, data_source_label, experiment_mode, augmentation_ratio, attack_types_combined, name, model_name, hyperparameters_enabled)  # Build cache identity source for diagnostics
            phase_cache_digest = hashlib.sha256(json.dumps(phase_cache_key, sort_keys=True, default=str).encode("utf-8")).hexdigest()  # Build stable cache identity digest
            phase_metadata = {"dataset_identity": os.path.basename(str(file)), "dataset_source": file, "execution_mode": execution_mode_str, "attack_scope": attack_types_combined, "data_source": data_source_label, "experiment_mode": experiment_mode, "augmentation_ratio": augmentation_ratio, "feature_set_name": name, "hyperparameter_mode": "Optimized Hyperparameters" if hyperparameters_enabled else "Default Hyperparameters", "classifier_name": model_name, "classifier_params_digest": phase_params_digest, "classifier_params_reference": f"sha256:{phase_params_digest.get('digest')}", "train_sample_count": len(y_train), "test_sample_count": len(y_test), "feature_count": X_train_n_cols, "n_jobs": get_classifier_n_jobs(active_model), "cache_identity": phase_cache_digest, "cache_reference": cache_ref_file, "combination_index": current_combination, "total_combinations": total_steps}  # Build compact phase metadata for this classifier
            if memory_watcher_enabled(config):  # Add concrete matrix layout diagnostics only when the watcher is active
                phase_metadata.update(build_array_memory_metadata("X_train", X_train_values))  # Record train matrix shape/dtype/layout
                phase_metadata.update(build_array_memory_metadata("X_test", X_test_values))  # Record test matrix shape/dtype/layout
            write_memory_phase_event("before_classifier_fit", config=config, **phase_metadata, event_outcome="starting")  # Publish classifier fit start
            training_ram_stats = {}  # Hold RAM statistics for this classifier fit only.

            metrics = evaluate_individual_classifier(
                active_model,  # Use the fitted clone for this atomic classifier.
                model_name,
                X_train_values,
                y_train,
                X_test_values,
                y_test,
                file,
                scaler,
                subset_feature_names,
                artifact_feature_set,  # Use the HP-specific artifact label.
                config=config,
                phase_metadata=phase_metadata,  # Pass compact watcher context into fit completion and error events
                training_ram_stats=training_ram_stats,  # Capture per-classifier RAM statistics for explainability scheduling.
            )  # Evaluate individual classifier sequentially using HP-isolated model artifact names
            write_memory_phase_event("after_prediction_and_metrics", config=config, **phase_metadata, accuracy=metrics[0], precision=metrics[1], recall=metrics[2], f1_score=metrics[3], event_outcome="metrics_completed")  # Publish prediction and metrics completion

            model_class = active_model.__class__.__name__  # Retrieve model class name for result entry
            effective_hyperparameters = serialize_effective_estimator_parameters(active_model)  # Serialize effective estimator parameters after fitting.
            result_entry = build_classifier_result_entry(
                model_class, file, execution_mode_str, attack_types_combined, name, "Individual",
                model_name, data_source_label, experiment_id, experiment_mode, augmentation_ratio,
                X_train_n_cols, len(y_train), len(y_test), metrics, subset_feature_names,
                hyperparams_map=hyperparams_map,
                hyperparameters_enabled=hyperparameters_enabled,
                effective_hyperparameters=effective_hyperparameters,
            )  # Build standardized result entry for this individual classifier
            write_memory_phase_event("before_cache_persist", config=config, **phase_metadata, event_outcome="starting")  # Publish cache persistence start
            persist_cache_result_entry(cache_ref_file, result_entry, cache_dict, config=config)  # Persist this atomic classifier result immediately and register its resume identity
            write_memory_phase_event("after_cache_persist", config=config, **phase_metadata, event_outcome="persisted")  # Publish cache persistence completion

            results_dict[(name, model_name)] = result_entry  # Store result keyed by feature set and model only after durable cache verification

            export_model_and_scaler(active_model, scaler, dataset_name, model_name, feature_set=artifact_feature_set, dataset_csv_path=file, config=config, artifact_context=artifact_context, label_encoder=label_encoder, transformer=transformer)  # Atomically persist the original-trained classifier and fitted preprocessing.
            pass  # Verify removal of duplicate individual model accuracy print
            progress_bar.update(1)  # Advance progress bar by one step

            write_memory_phase_event("before_explainability_schedule", config=config, **phase_metadata, event_outcome="starting")  # Publish explainability scheduling start
            explainability_outcome = "skipped"  # Track explainability scheduling outcome
            if config.get("explainability", {}).get("enabled", False) and experiment_mode == "original_only":  # Only queue explainability on original data
                try:  # Attempt to dispatch explainability for this model.
                    explainability_dispatch = schedule_explainability_job(active_model, model_name, X_test_subset, y_test, subset_feature_names, file, name, execution_mode_str, config, experiment_mode, hyperparameters_enabled, training_ram_stats=training_ram_stats)  # Dispatch explainability according to RAM-gated process policy.
                    explainability_outcome = f"scheduled:{explainability_dispatch.get('mode', 'none')}" if isinstance(explainability_dispatch, dict) else "scheduled:none"  # Track selected explainability execution mode.
                except Exception as e:  # If explainability queueing fails
                    explainability_outcome = f"failed:{e}"  # Track explainability scheduling failure
                    verbose_output(
                        f"{BackgroundColors.YELLOW}Explainability failed for {model_name}: {e}{Style.RESET_ALL}",
                        config=config
                    )  # Log explainability failure before propagation.
                    raise  # Preserve explainability failure propagation.
            write_memory_phase_event("after_explainability_schedule", config=config, **phase_metadata, event_outcome=explainability_outcome)  # Publish explainability scheduling completion

            current_combination += 1  # Advance the global combination counter
            write_memory_phase_event("before_memory_cleanup", config=config, **phase_metadata, event_outcome="starting")  # Publish per-classifier cleanup start
            del active_model, artifact_context, metrics, training_ram_stats  # Release fitted estimator, artifact metadata, metrics, and RAM stats before the next classifier
            gc.collect()  # Reclaim estimator-owned memory before the next atomic classifier
            write_memory_phase_event("after_memory_cleanup", config=config, **phase_metadata, event_outcome="completed")  # Publish per-classifier cleanup completion

        del X_train_values, X_test_values  # Drop local array views before returning feature-set results
        return (results_dict, current_combination)  # Return accumulated results and updated combination counter
    except Exception as e:
        try:
            if "active_model" in locals():  # Release any partially fitted estimator before re-raising
                del active_model
            if "metrics" in locals():  # Release metric tuple if the failure happened after evaluation
                del metrics
            if "training_ram_stats" in locals():  # Release classifier RAM statistics if allocated before failure
                del training_ram_stats
            if "X_train_values" in locals():  # Release train array view held by this function frame
                del X_train_values
            if "X_test_values" in locals():  # Release test array view held by this function frame
                del X_test_values
            gc.collect()  # Reclaim any released estimator-owned memory before error reporting
        except Exception:
            pass  # Do not mask the primary failure
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_stacking_evaluation_for_feature_set(name, stacking_model, X_train_df, y_train, X_test_df, y_test, X_test_subset, X_train_n_cols, file, execution_mode_str, attack_types_combined, data_source_label, experiment_id, experiment_mode, augmentation_ratio, scaler, label_encoder, transformer, input_feature_names, target_column, source_files, subset_feature_names, total_steps, current_combination, progress_bar, config=None, cache_dict=None, cache_ref_file=None, hyperparameters_enabled=False):
    """
    Evaluates the stacking classifier for one feature set, exports the model, generates metric plots, and returns the result entry.

    :param name: Name of the current feature set being evaluated
    :param stacking_model: Unfitted stacking classifier prototype
    :param X_train_df: Training feature DataFrame with named columns
    :param y_train: Training target labels
    :param X_test_df: Test feature DataFrame with named columns
    :param y_test: Test target labels
    :param X_test_subset: Test feature array for explainability pipeline input
    :param X_train_n_cols: Number of training columns used for result entry metadata
    :param file: Path to the dataset file for export and result metadata
    :param execution_mode_str: Execution mode string ('separate_files' or 'combined_files')
    :param attack_types_combined: List of attack types for combined files evaluation or None for separate files evaluation
    :param data_source_label: Label identifying the data source for result traceability
    :param experiment_id: Unique experiment identifier for traceability
    :param experiment_mode: Experiment mode string ('original_only' or 'original_training_augmented_testing')
    :param augmentation_ratio: Augmentation ratio float or None for original-only experiments
    :param scaler: Fitted scaler used for dataset preprocessing
    :param label_encoder: Label encoder fitted on original training labels
    :param transformer: Optional PCA transformer fitted on original training features
    :param input_feature_names: Ordered numeric schema before preprocessing
    :param target_column: Positional target column name
    :param source_files: Ordered original source files used for model identity
    :param subset_feature_names: List of feature names for the current subset
    :param total_steps: Total number of evaluation steps for Telegram progress messages
    :param current_combination: Current combination index counter for progress messages
    :param progress_bar: tqdm progress bar to update after stacking evaluation
    :param config: Configuration dictionary (uses global CONFIG if None)
    :param cache_dict: Dictionary of previously cached results keyed by resume cache key for skip-if-cached logic.
    :param cache_ref_file: File path used when deriving the cache file location for atomic cache writes.
    :return: Tuple (stacking_result_entry, next_current_combination) where stacking_result_entry is the standardized result dict
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        if cache_dict:  # Verify if a cache dictionary is available for resume
            resume_key = build_resume_cache_key(execution_mode_str, data_source_label, experiment_mode, augmentation_ratio, attack_types_combined, name, "StackingClassifier", hyperparameters_enabled)  # Build the full resume cache key for the stacking classifier
            if resume_key in cache_dict:  # Verify if the stacking result is already cached from a previous run
                cached_result = cache_dict[resume_key]  # Retrieve cached stacking result entry for full resume logging
                combination_header = build_telegram_combination_header(name, "StackingClassifier", augmentation_ratio, hyperparameters_enabled)  # Build full recovered stacking combination label
                print(f"{BackgroundColors.YELLOW}[RESUME] Recovered combination {current_combination}/{total_steps}: {combination_header} from saved partial progress (no recomputation performed).{Style.RESET_ALL}")  # Log recovered stacking combination to stdout for visibility

                cached_execution_mode = cached_result.get("execution_mode", execution_mode_str)  # Resolve execution mode from cached entry for log consistency
                evaluation_mode = str(cached_execution_mode).replace("_", " ").title().replace(" ", "") if cached_execution_mode else "SeparateFiles"  # Normalize execution mode to CamelCase style
                acc = cached_result.get("accuracy", "N/A")  # Recover accuracy from cached stacking result for full metrics log
                prec = cached_result.get("precision", "N/A")  # Recover precision from cached stacking result for full metrics log
                rec = cached_result.get("recall", "N/A")  # Recover recall from cached stacking result for full metrics log
                f1 = cached_result.get("f1_score", "N/A")  # Recover F1 score from cached stacking result for full metrics log
                fpr = cached_result.get("fpr", "N/A")  # Recover FPR from cached stacking result for full metrics log
                fnr = cached_result.get("fnr", "N/A")  # Recover FNR from cached stacking result for full metrics log
                cached_elapsed = cached_result.get("elapsed_time_s", 0)  # Recover elapsed time from cached stacking result for full metrics log
                cached_elapsed = int(round(float(cached_elapsed))) if cached_elapsed is not None else 0  # Normalize elapsed seconds to integer for display
                cached_human_time = calculate_execution_time(cached_elapsed)  # Format elapsed seconds to human-readable duration string
                print(
                    f"{BackgroundColors.CYAN}StackingClassifier{BackgroundColors.GREEN}: Mode {BackgroundColors.CYAN}{evaluation_mode}{BackgroundColors.GREEN} | F1-Score {BackgroundColors.CYAN}{f1}{BackgroundColors.GREEN} | Accuracy: {BackgroundColors.CYAN}{acc}{BackgroundColors.GREEN} | Precision: {BackgroundColors.CYAN}{prec}{BackgroundColors.GREEN} | Recall: {BackgroundColors.CYAN}{rec}{BackgroundColors.GREEN} | FPR: {BackgroundColors.CYAN}{fpr}{BackgroundColors.GREEN} | FNR: {BackgroundColors.CYAN}{fnr}{BackgroundColors.GREEN} | Training Time: {BackgroundColors.CYAN}{cached_elapsed}s{BackgroundColors.GREEN} | Execution Time: {BackgroundColors.CYAN}{cached_elapsed}s{BackgroundColors.GREEN} | Total Time: {BackgroundColors.CYAN}{cached_human_time}{BackgroundColors.GREEN} ({BackgroundColors.CYAN}{cached_elapsed}s{BackgroundColors.GREEN}){Style.RESET_ALL}"
                )  # Print a full metrics summary for recovered cached stacking classifier as if freshly evaluated

                progress_bar.update(1)  # Advance progress bar even for skipped cached evaluations
                current_combination += 1  # Advance the global combination counter for skipped evaluations
                return (cached_result, current_combination)  # Return the cached stacking result entry without re-running the evaluation

        active_stacking_model = clone(stacking_model)  # Isolate fitted stacking state to this atomic feature-set evaluation.

        try:  # Attempt to obtain a compact snapshot of stacking model parameters for logging
            params_raw = active_stacking_model.get_params() if hasattr(active_stacking_model, "get_params") else {}  # Get model parameters when available
        except Exception:  # On any error retrieving params
            params_raw = {}  # Fallback to empty dict when parameters cannot be read

        try:  # Build a compact, truncated string representation of parameters for safe printing
            items = list(params_raw.items())[:6]  # Limit to first N items to avoid excessive output
            params_snapshot = ", ".join(f"{k}={v}" for k, v in items)  # Join key=value pairs for display
            if len(params_snapshot) > 240:  # Truncate overly long snapshots for readability
                params_snapshot = params_snapshot[:237] + "..."  # Truncate and append ellipsis
        except Exception:  # On failure during snapshot formatting
            params_snapshot = ""  # Use empty string when formatting fails

        print(
            f"  {BackgroundColors.GREEN}Training {BackgroundColors.CYAN}Stacking Classifier{BackgroundColors.GREEN}...{Style.RESET_ALL} Params: {params_snapshot}"
        )  # Announce the start of stacking classifier training and evaluation with compact parameter snapshot
        combination_header = build_telegram_combination_header(name, "StackingClassifier", augmentation_ratio, hyperparameters_enabled)  # Build full progress label for stacking classifier
        progress_bar.set_description(combination_header)  # Update progress bar with the complete active configuration

        phase_params_digest = get_classifier_params_digest(active_stacking_model)  # Build compact stacking parameter digest
        phase_cache_key = build_resume_cache_key(execution_mode_str, data_source_label, experiment_mode, augmentation_ratio, attack_types_combined, name, "StackingClassifier", hyperparameters_enabled)  # Build stacking cache identity source for diagnostics
        phase_cache_digest = hashlib.sha256(json.dumps(phase_cache_key, sort_keys=True, default=str).encode("utf-8")).hexdigest()  # Build stable stacking cache identity digest
        phase_metadata = {"dataset_identity": os.path.basename(str(file)), "dataset_source": file, "execution_mode": execution_mode_str, "attack_scope": attack_types_combined, "data_source": data_source_label, "experiment_mode": experiment_mode, "augmentation_ratio": augmentation_ratio, "feature_set_name": name, "hyperparameter_mode": "Optimized Hyperparameters" if hyperparameters_enabled else "Default Hyperparameters", "classifier_name": "StackingClassifier", "classifier_params_digest": phase_params_digest, "classifier_params_reference": f"sha256:{phase_params_digest.get('digest')}", "train_sample_count": len(y_train), "test_sample_count": len(y_test), "feature_count": X_train_n_cols, "n_jobs": get_classifier_n_jobs(active_stacking_model), "cache_identity": phase_cache_digest, "cache_reference": cache_ref_file, "combination_index": current_combination, "total_combinations": total_steps}  # Build compact phase metadata for stacking
        write_memory_phase_event("before_classifier_fit", config=config, **phase_metadata, event_outcome="starting")  # Publish stacking fit start
        stacking_ram_stats = {}  # Hold RAM statistics for this stacking fit only.

        stacking_metrics = evaluate_stacking_classifier(
            active_stacking_model, X_train_df, y_train, X_test_df, y_test, config=config, training_ram_stats=stacking_ram_stats
        )  # Evaluate stacking model with DataFrames and retrieve metrics tuple
        write_memory_phase_event("after_classifier_fit", config=config, **phase_metadata, event_outcome="fit_and_prediction_completed")  # Publish stacking fit completion
        write_memory_phase_event("after_prediction_and_metrics", config=config, **phase_metadata, accuracy=stacking_metrics[0], precision=stacking_metrics[1], recall=stacking_metrics[2], f1_score=stacking_metrics[3], event_outcome="metrics_completed")  # Publish stacking metrics completion

        s_y_pred = stacking_metrics[7] if len(stacking_metrics) > 7 else None  # Extract stacking predictions from metrics tuple for plot generation

        try:  # Attempt to generate metric plots for stacking model
            file_path_obj = Path(file)  # Create Path object for the dataset file
            feature_analysis_dir = file_path_obj.parent / "Feature_Analysis"  # Build Feature_Analysis directory path for outputs
            stacking_output_dir = get_stacking_output_dir(str(file_path_obj), config)  # Get stacking output directory path from config
            generate_and_save_metric_plots(y_test, s_y_pred, config.get("stacking", {}), stacking_output_dir)  # Generate and save metric plots to stacking output directory
        except Exception:  # If metric plot generation fails
            pass  # Continue without plotting

        stacking_result_entry = build_classifier_result_entry(
            active_stacking_model.__class__.__name__, file, execution_mode_str, attack_types_combined, name, "Stacking",
            "StackingClassifier", data_source_label, experiment_id, experiment_mode, augmentation_ratio,
            X_train_n_cols, len(y_train), len(y_test), stacking_metrics, subset_feature_names,
            hyperparameters_enabled=hyperparameters_enabled,
            effective_hyperparameters=serialize_effective_estimator_parameters(active_stacking_model),
        )  # Build standardized result entry for the stacking classifier

        write_memory_phase_event("before_cache_persist", config=config, **phase_metadata, event_outcome="starting")  # Publish stacking cache persistence start
        persist_cache_result_entry(cache_ref_file, stacking_result_entry, cache_dict, config=config)  # Persist this atomic stacking result immediately and register its resume identity
        write_memory_phase_event("after_cache_persist", config=config, **phase_metadata, event_outcome="persisted")  # Publish stacking cache persistence completion

        dataset_name = build_filename_safe_dataset_identity(resolve_canonical_dataset_identity(str(file), True)) if execution_mode_str == "combined_files" else os.path.basename(os.path.dirname(file))  # Resolve stacking export dataset folder by execution mode.
        artifact_feature_set = f"{name} - {'Optimized Hyperparameters' if hyperparameters_enabled else 'Default Hyperparameters'}"  # Keep stacking artifacts isolated by HP mode
        artifact_context = build_stacking_model_artifact_context(file, source_files, execution_mode_str, attack_types_combined, target_column, "StackingClassifier", active_stacking_model, artifact_feature_set, input_feature_names, subset_feature_names, list(label_encoder.classes_), transformer, hyperparameters_enabled)  # Build ratio-independent original-training artifact identity.
        export_model_and_scaler(active_stacking_model, scaler, dataset_name, "StackingClassifier", feature_set=artifact_feature_set, dataset_csv_path=file, config=config, artifact_context=artifact_context, label_encoder=label_encoder, transformer=transformer)  # Atomically persist fitted stacking and preprocessing.
        pass  # Verify removal of duplicate stacking classifier accuracy print
        progress_bar.update(1)  # Advance progress bar after stacking evaluation
        current_combination += 1  # Advance the global combination counter

        write_memory_phase_event("before_explainability_schedule", config=config, **phase_metadata, event_outcome="starting")  # Publish stacking explainability scheduling start
        explainability_outcome = "skipped"  # Track stacking explainability scheduling outcome
        if config.get("explainability", {}).get("enabled", False) and experiment_mode == "original_only":  # Only queue explainability on original data
            try:  # Attempt to dispatch explainability for stacking model.
                explainability_dispatch = schedule_explainability_job(active_stacking_model, "StackingClassifier", X_test_subset, y_test, subset_feature_names, file, name, execution_mode_str, config, experiment_mode, hyperparameters_enabled, training_ram_stats=stacking_ram_stats)  # Dispatch explainability according to RAM-gated process policy.
                explainability_outcome = f"scheduled:{explainability_dispatch.get('mode', 'none')}" if isinstance(explainability_dispatch, dict) else "scheduled:none"  # Track selected explainability execution mode.
            except Exception as e:  # If explainability queueing fails
                explainability_outcome = f"failed:{e}"  # Track stacking explainability scheduling failure
                verbose_output(
                    f"{BackgroundColors.YELLOW}Explainability failed for StackingClassifier: {e}{Style.RESET_ALL}",
                    config=config
                )  # Log explainability failure before propagation.
                raise  # Preserve explainability failure propagation.
        write_memory_phase_event("after_explainability_schedule", config=config, **phase_metadata, event_outcome=explainability_outcome)  # Publish stacking explainability scheduling completion
        write_memory_phase_event("before_memory_cleanup", config=config, **phase_metadata, event_outcome="starting")  # Publish stacking cleanup start
        del active_stacking_model, artifact_context, stacking_ram_stats  # Release fitted stacking, artifact metadata, and RAM statistics before returning.
        gc.collect()  # Reclaim any released stacking temporaries before returning
        write_memory_phase_event("after_memory_cleanup", config=config, **phase_metadata, event_outcome="completed")  # Publish stacking cleanup completion

        return (stacking_result_entry, current_combination)  # Return result entry and updated combination counter
    except MemoryError as e:  # Handle stacking memory errors with diagnostic phase
        if "active_stacking_model" in locals():  # Release fitted stacking state after a memory failure.
            del active_stacking_model  # Drop the fitted model reference before error reporting.
            gc.collect()  # Reclaim released stacking state.
        if "stacking_ram_stats" in locals():  # Release stacking RAM statistics if allocated before memory failure.
            del stacking_ram_stats  # Drop RAM statistics reference before error reporting.
        write_memory_phase_event("memory_error", config=config, classifier_name="StackingClassifier", feature_set_name=name, train_sample_count=len(y_train) if y_train is not None else None, test_sample_count=len(y_test) if y_test is not None else None, feature_count=X_train_n_cols, event_outcome=str(e))  # Publish stacking memory error
        print(str(e))  # Print memory error to terminal logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send memory error via Telegram
        raise  # Preserve original MemoryError behavior
    except Exception as e:
        if "active_stacking_model" in locals():  # Release fitted stacking state after an evaluation failure.
            del active_stacking_model  # Drop the fitted model reference before error reporting.
            gc.collect()  # Reclaim released stacking state.
        if "stacking_ram_stats" in locals():  # Release stacking RAM statistics if allocated before failure.
            del stacking_ram_stats  # Drop RAM statistics reference before error reporting.
        write_memory_phase_event("model_error", config=config, classifier_name="StackingClassifier", feature_set_name=name, train_sample_count=len(y_train) if y_train is not None else None, test_sample_count=len(y_test) if y_test is not None else None, feature_count=X_train_n_cols, event_outcome=str(e))  # Publish stacking model error
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def print_dataset_evaluation_header(data_source_label, evaluation_plan, execution_mode_str, attack_types_combined):
    """
    Prints the formatted header block for a dataset evaluation run.

    :param data_source_label: Label identifying the data source being evaluated
    :param evaluation_plan: Ordered runtime combinations represented by the progress bar
    :param execution_mode_str: Execution mode shared by every listed combination
    :param attack_types_combined: Attack types shared by combined-files combinations
    :return: None
    """

    try:
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*80}{Style.RESET_ALL}"
        )  # Print top separator line for evaluation section
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Evaluating on: {BackgroundColors.CYAN}{data_source_label} Data{Style.RESET_ALL}"
        )  # Print the data source label currently being evaluated
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*80}{Style.RESET_ALL}\n"
        )  # Print bottom separator line for evaluation section

        total_combinations = len(evaluation_plan)  # Use the exact ordered plan length displayed by the progress bar
        print(f"Evaluation plan: {total_combinations} combinations\n")  # Print the exact full-grid combination count
        print(f"Execution Mode: {execution_mode_str}")  # Print the execution-mode identity shared by the plan
        if attack_types_combined:  # Print the combined attack scope when it participates in cache identity
            print(f"Attack Types: {', '.join(str(attack_type) for attack_type in attack_types_combined)}")  # Print the ordered attack-type scope once for the full plan
        print()  # Separate shared identity fields from ordered combinations

        for combination_index, (feature_set, hyperparameters_enabled, augmentation_ratio, classifier) in enumerate(evaluation_plan, start=1):  # Print combinations in their exact execution order
            hyperparameter_label = "Optimized Hyperparameters" if hyperparameters_enabled else "Default Hyperparameters"  # Resolve the active hyperparameter label
            augmentation_label = f"{augmentation_ratio * 100:g}%" if augmentation_ratio is not None else "None"  # Resolve the augmented-test ratio.
            testing_data_label = "Augmented Data" if augmentation_ratio is not None else "Original Data"  # Resolve the isolated testing source.
            print(f"[{combination_index}/{total_combinations}] Feature Set: {feature_set} | Hyperparameters: {hyperparameter_label} | Training Data: Original Data | Testing Data: {testing_data_label} | Augmented Test Ratio: {augmentation_label} | Classifier: {classifier}")  # Print one complete ordered combination identity.

        feature_sets = list(dict.fromkeys(combination[0] for combination in evaluation_plan))  # Preserve first-occurrence feature-set order for the Telegram summary
        hyperparameter_modes = list(dict.fromkeys("Optimized Hyperparameters" if combination[1] else "Default Hyperparameters" for combination in evaluation_plan))  # Preserve first-occurrence runnable hyperparameter order
        augmentation_modes = list(dict.fromkeys(f"{combination[2] * 100:g}%" if combination[2] is not None else "None" for combination in evaluation_plan))  # Preserve first-occurrence augmented-test order.
        classifiers = list(dict.fromkeys(combination[3] for combination in evaluation_plan))  # Preserve first-occurrence enabled-classifier order

        telegram_summary = "\n".join([
            f"Evaluation plan: {data_source_label} Data",
            f"Total combinations: {total_combinations}",
            f"Feature sets: {', '.join(feature_sets)}",
            f"Hyperparameters: {', '.join(hyperparameter_modes)}",
            "Training data: Original Data",
            "Testing data: Original Data and ratio-selected Augmented Data",
            f"Augmented test ratios: {', '.join(augmentation_modes)}",
            f"Classifiers: {', '.join(classifiers)}",
            "Detailed ordered plan written to the application log.",
        ])  # Build one condensed Telegram summary from the ordered runtime plan
        send_telegram_message(TELEGRAM_BOT, telegram_summary)  # Send one guarded, length-protected summary for this evaluation section
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def setup_feature_set_names(name, X_train_subset, subset_feature_names_list):
    """
    Determine the list of feature names for the given feature set, generating PCA names or generic labels when needed.

    :param name: Name of the current feature set.
    :param X_train_subset: Training feature array used to determine the number of features.
    :param subset_feature_names_list: Pre-computed feature names list, or None/empty for generic generation.
    :return: List of feature name strings for the current subset.
    """

    if name == "PCA Components":  # If the feature set is pca components
        return [f"PC{i+1}" for i in range(X_train_subset.shape[1])]  # Generate pca component names
    return (
        subset_feature_names_list if subset_feature_names_list else [f"feature_{i}" for i in range(X_train_subset.shape[1])]
    )  # Use actual feature names or generate generic ones


def align_feature_names_to_array(X_subset, feature_names, label):
    """
    Align a feature name list to match the number of columns in a feature array.

    :param X_subset: Feature array whose column count is the authoritative shape.
    :param feature_names: List of feature names to align against the array shape.
    :param label: Label string used in the warning message to identify which array is being aligned.
    :return: Feature name list whose length exactly matches X_subset.shape[1].
    """

    n_features_array = X_subset.shape[1]  # Get the number of columns in the feature array
    n_features_names = len(feature_names)  # Get the number of provided feature names

    if n_features_array == n_features_names:  # Verify if sizes already match
        return feature_names  # Return unchanged list when sizes are equal

    print(
        f"{BackgroundColors.YELLOW}[WARNING] Feature name/array mismatch in {label}: "
        f"array has {n_features_array} columns, names list has {n_features_names} entries. "
        f"Applying safe alignment.{Style.RESET_ALL}"
    )  # Log the mismatch details and alignment action

    if n_features_names > n_features_array:  # Verify if names list is longer than array columns
        return feature_names[:n_features_array]  # Truncate names list to match array column count

    extra_names = [f"feature_{i}" for i in range(n_features_names, n_features_array)]  # Generate synthetic names for missing indices
    return feature_names + extra_names  # Extend names list with synthetic entries to match array column count


def convert_subset_to_dataframes(X_train_subset, X_test_subset, subset_feature_names):
    """
    Convert training and test feature arrays to DataFrames using the provided feature name list.

    :param X_train_subset: Training feature array to wrap in a DataFrame.
    :param X_test_subset: Test feature array to wrap in a DataFrame.
    :param subset_feature_names: List of column names to assign to both DataFrames.
    :return: Tuple of (X_train_df, X_test_df) DataFrames with named columns.
    """

    train_names = align_feature_names_to_array(X_train_subset, subset_feature_names, "X_train_subset")  # Align feature names to training array shape
    test_names = align_feature_names_to_array(X_test_subset, subset_feature_names, "X_test_subset")  # Align feature names to test array shape

    X_train_df = pd.DataFrame(X_train_subset, columns=train_names, copy=False)  # Wrap training features without requesting an additional copy
    X_test_df = pd.DataFrame(X_test_subset, columns=test_names, copy=False)  # Wrap test features without requesting an additional copy

    return X_train_df, X_test_df  # Return DataFrames with named columns


def initialize_evaluation_run_state(base_models, feature_sets, data_source_label, stacking_enabled=True):
    """
    Build the individual model map, compute total evaluation steps, create the progress bar, and initialize result containers.

    :param base_models: Dictionary of base model name to model instance pairs used for individual evaluation.
    :param feature_sets: Ordered dictionary of feature set name to (X_train_subset, X_test_subset, feature_names) tuples.
    :param data_source_label: String label for the current data source displayed in the progress bar description.
    :return: Tuple of (individual_models, total_steps, progress_bar, all_results, current_combination) ready for the evaluation loop.
    """

    individual_models = {
        k: v for k, v in base_models.items()
    }  # Use the base models (with hyperparameters applied) for individual evaluation

    total_steps = len(feature_sets) * len(individual_models)  # Total steps: individual models only by default
    if stacking_enabled:  # Count one extra step per feature set only when stacking is enabled
        total_steps += len(feature_sets)

    progress_bar = tqdm(total=total_steps, desc=f"{data_source_label} Data", file=sys.stdout)  # Progress bar for all evaluations

    all_results = {}  # Dictionary to store results: (feature_set, model_name) -> result_entry

    current_combination = 1  # Counter for combination index

    return individual_models, total_steps, progress_bar, all_results, current_combination  # Return all initialized evaluation state variables


def evaluate_single_feature_set(
    idx,
    name,
    X_train_subset,
    X_test_subset,
    subset_feature_names_list,
    individual_models,
    stacking_model,
    y_train,
    y_test,
    file,
    execution_mode_str,
    attack_types_combined,
    data_source_label,
    experiment_id,
    experiment_mode,
    augmentation_ratio,
    hyperparams_map,
    scaler,
    label_encoder,
    transformer,
    input_feature_names,
    target_column,
    source_files,
    total_steps,
    current_combination,
    progress_bar,
    stacking_enabled=True,
    config=None,
    cache_dict=None,
    cache_ref_file=None,
    hyperparameters_enabled=False,
):
    """
    Evaluate all individual classifiers and the stacking model on one non-empty feature subset.

    :param idx: 1-based index of the current feature set in the evaluation loop for logging.
    :param name: Name string identifying the current feature set used for output labeling.
    :param X_train_subset: Training feature matrix restricted to the current feature subset.
    :param X_test_subset: Test feature matrix restricted to the current feature subset.
    :param subset_feature_names_list: List of feature names for the current subset or None to derive from array.
    :param individual_models: Dictionary of model name to model instance pairs for individual evaluation.
    :param stacking_model: Stacking classifier instance built from the base models.
    :param y_train: Training target vector used for fitting all classifiers.
    :param y_test: Test target vector used for evaluating all classifiers.
    :param file: Dataset file path used for model artifact export and result labeling.
    :param execution_mode_str: Execution mode string ('separate_files' or 'combined_files') selecting the evaluation strategy.
    :param attack_types_combined: List of attack type strings for combined files evaluation mode or None for separate files evaluation mode.
    :param data_source_label: Data source label string included in result entries for traceability.
    :param experiment_id: Unique experiment identifier string included in result entries for traceability.
    :param experiment_mode: Experiment mode string included in result entries for traceability.
    :param augmentation_ratio: Augmentation ratio float included in result entries or None for original-only.
    :param hyperparams_map: Dictionary mapping model names to best hyperparameter dicts applied before training.
    :param scaler: Fitted StandardScaler instance used to transform subsets when needed.
    :param label_encoder: Label encoder fitted on original training labels.
    :param transformer: Optional PCA transformer fitted on original training features.
    :param input_feature_names: Ordered numeric schema before preprocessing.
    :param target_column: Positional target column name.
    :param source_files: Ordered original source files used for model identity.
    :param total_steps: Total number of evaluation steps across all feature sets used by the progress bar.
    :param current_combination: 1-based overall combination counter updated for each model evaluated.
    :param progress_bar: tqdm progress bar instance advanced after each model evaluation.
    :param config: Optional configuration dictionary; falls back to global CONFIG when None.
    :param cache_dict: Dictionary of previously cached results keyed by resume cache key for skip-if-cached logic.
    :param cache_ref_file: File path used when deriving the cache file location for atomic cache writes.
    :return: Tuple of (individual_results, stacking_result_entry, current_combination) containing per-model result dicts and updated combination counter.
    """

    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Evaluating models on: {BackgroundColors.CYAN}{name} ({X_train_subset.shape[1]} features){Style.RESET_ALL}"
    )  # Output evaluation status

    subset_feature_names = setup_feature_set_names(name, X_train_subset, subset_feature_names_list)  # Determine feature names for this subset

    X_train_df, X_test_df = convert_subset_to_dataframes(X_train_subset, X_test_subset, subset_feature_names)  # Convert feature arrays to dataframes with named columns

    individual_results, current_combination = run_individual_classifiers_for_feature_set(
        name, individual_models, X_train_df, y_train, X_test_df, y_test,
        X_test_subset, X_train_subset.shape[1], file, execution_mode_str, attack_types_combined,
        data_source_label, experiment_id, experiment_mode, augmentation_ratio,
        hyperparams_map, scaler, label_encoder, transformer, input_feature_names, target_column, source_files, subset_feature_names, total_steps, current_combination, progress_bar, config=config,
        cache_dict=cache_dict, cache_ref_file=cache_ref_file, hyperparameters_enabled=hyperparameters_enabled,
    )  # Evaluate all individual classifiers and collect their result entries with resume support

    stacking_result_entry = None
    if stacking_enabled:  # Skip stacking entirely when disabled to save RAM/CPU
        stacking_result_entry, current_combination = run_stacking_evaluation_for_feature_set(
            name, stacking_model, X_train_df, y_train, X_test_df, y_test,
            X_test_subset, X_train_subset.shape[1], file, execution_mode_str, attack_types_combined,
            data_source_label, experiment_id, experiment_mode, augmentation_ratio,
            scaler, label_encoder, transformer, input_feature_names, target_column, source_files, subset_feature_names, total_steps, current_combination, progress_bar, config=config,
            cache_dict=cache_dict, cache_ref_file=cache_ref_file, hyperparameters_enabled=hyperparameters_enabled,
        )  # Evaluate stacking classifier, export model artifacts, generate metric plots, and collect result entry with resume support

    return individual_results, stacking_result_entry, current_combination  # Return per-model results and updated combination counter to the caller


def evaluate_on_dataset(
    file,
    df,
    feature_names,
    ga_selected_features,
    pca_n_components,
    rfe_selected_features,
    base_models,
    data_source_label="Original",
    hyperparams_map=None,
    experiment_id=None,
    experiment_mode="original_only",
    augmentation_ratio=None,
    execution_mode_str="separate_files",
    attack_types_combined=None,
    df_augmented_for_testing=None,
    config=None,
    cache_ref_file=None,
    hyperparameters_enabled=None,
    grid_progress=None,
    source_files=None,  # Preserve ordered dataset provenance for PCA cache validation.
    artifact_recovery_target=None,
):
    """
    Train on original data or evaluate persisted original-trained models on augmented data.
    :param file: Path to the dataset file
    :param df: DataFrame with the original dataset (used for test set)
    :param feature_names: List of feature column names
    :param ga_selected_features: GA selected features
    :param pca_n_components: Number of PCA components
    :param rfe_selected_features: RFE selected features
    :param base_models: Dictionary of base models to evaluate
    :param data_source_label: Label for the original or augmented testing source
    :param hyperparams_map: Dictionary mapping model names to hyperparameter dicts
    :param experiment_id: Unique experiment identifier for traceability
    :param experiment_mode: Experiment mode string ('original_only' or 'original_training_augmented_testing')
    :param augmentation_ratio: Augmentation ratio float (e.g., 0.50) or None for original-only
    :param execution_mode_str: Execution mode string ('separate_files' or 'combined_files')
    :param attack_types_combined: List of attack types for combined files evaluation or None for separate files evaluation
    :param df_augmented_for_testing: Optional sampled augmented DataFrame used only for persisted-model testing
    :param config: Configuration dictionary (uses global CONFIG if None)
    :param cache_ref_file: Override file path to use when computing the cache file location (required when file is 'combined_files_combined')
    :param hyperparameters_enabled: Explicit hyperparameter mode flag for progress labels and result flow
    :param source_files: Ordered original source files used to construct the training dataset.
    :param artifact_recovery_target: Optional feature-set and classifier identity for original-only artifact recovery.
    :return: Dictionary mapping (feature_set, model_name) to results
    """

    feature_source_arrays = None  # Initialized before try so failure cleanup can run safely even during early setup errors
    feature_sets_iter = None  # Initialized before try so generator cleanup is always safe
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        if hyperparameters_enabled is None:  # If caller did not provide an explicit HP mode flag
            hyperparameters_enabled = bool(hyperparams_map)  # Preserve legacy inference for older callers

        stacking_enabled = config.get("stacking", {}).get("methods", {}).get("stacking", True)  # Resolve stacking toggle from config
        if artifact_recovery_target is not None:
            stacking_enabled = artifact_recovery_target[1] == "StackingClassifier"  # Restrict recovery to the missing classifier identity.

        ga_selected_features, rfe_selected_features = sanitize_and_verify_feature_selections(
            ga_selected_features, rfe_selected_features, feature_names, config=config
        )  # Sanitize and verify GA/RFE feature selections against available features

        feature_sets_config = dict(config.get("stacking", {}).get("feature_sets_config", {}))  # Copy feature strategy config so grid-specific enforcement cannot mutate global configuration
        if config.get("stacking", {}).get("methods", {}).get("feature_selection", True):  # Feature selection enabled: full features remain the required baseline alongside configured methods
            feature_sets_config["use_full"] = True  # Always include the full-feature baseline in the evaluation grid
        else:  # Feature selection disabled: only the full-feature baseline may be generated
            feature_sets_config = {"use_full": True, "use_pca": False, "use_rfe": False, "use_ga": False, "explicit_features": []}  # Suppress every selection strategy without mutating CLI/config state

        feature_mode_names = [artifact_recovery_target[0]] if artifact_recovery_target is not None else list_grid_feature_modes(ga_selected_features, pca_n_components, rfe_selected_features, feature_names, config=config)  # Resolve the full plan or one missing artifact mode.
        planned_models = base_models if artifact_recovery_target is None else ({artifact_recovery_target[1]: base_models[artifact_recovery_target[1]]} if artifact_recovery_target[1] != "StackingClassifier" else {})  # Keep recovery progress limited to the missing artifact.
        if grid_progress is None:  # Build the exact standalone plan for the current data and HP slice
            evaluation_plan = build_evaluation_plan([(bool(hyperparameters_enabled), planned_models, hyperparams_map or {})], [augmentation_ratio], feature_mode_names, stacking_enabled)  # Build the standalone ordered runtime combinations
        else:  # Reuse the complete ordered plan that created the shared full-grid progress bar
            evaluation_plan = grid_progress["evaluation_plan"]  # Reuse the authoritative full-grid plan source

        if artifact_recovery_target is None and (grid_progress is None or not grid_progress.get("plan_printed", False)):
            print_dataset_evaluation_header(data_source_label, evaluation_plan, execution_mode_str, attack_types_combined)  # Print the detailed local plan and one condensed Telegram summary.
            if grid_progress is not None:
                grid_progress["plan_printed"] = True  # Prevent per-ratio plan messages for the shared grid.

        pca_input_feature_names = [str(column) for column in df.iloc[:, :-1].select_dtypes(include=np.number).columns]  # Capture the exact ordered numeric columns consumed by scaling and PCA.
        target_column_name = str(df.columns[-1])  # Capture the positional target column used by the split flow.
        original_sample_count = int(len(df))  # Record the original sample population before splitting.
        if source_files is not None:  # Use caller-supplied provenance for combined or explicitly sourced evaluation data.
            pca_source_files = [str(source_file) for source_file in source_files]  # Preserve the caller's source order for deterministic combined-row provenance.
        elif not resolve_path_represents_directory(str(file)):  # Infer the unambiguous original single-file source.
            pca_source_files = [str(file)]  # Use the evaluated file itself as complete source provenance.
        else:  # Refuse inference for combined directories with undisclosed original source files.
            pca_source_files = []  # Disable persistent PCA reuse when source provenance cannot be proven.
        pca_cache_context = {  # Record the exact split and data-variant semantics that determine PCA fitting input.
            "execution_mode": str(execution_mode_str),  # Store separate-file or combined-file execution semantics.
            "data_source": "Original",  # PCA is always fitted for the original-data training identity.
            "experiment_mode": "original_only",  # PCA never receives augmented samples.
            "augmentation_ratio": None,  # Exclude evaluation-only augmentation ratios from fitted preprocessing identity.
            "target_column": target_column_name,  # Store the positional target column identity.
            "original_sample_count": original_sample_count,  # Store the original population size before splitting.
            "augmented_sample_count": 0,  # Record that fitted preprocessing receives no augmented samples.
            "test_size": 0.2,  # Store the fixed split ratio used by prepare_evaluation_data_splits.
            "random_state": 42,  # Store the fixed split seed used by prepare_evaluation_data_splits.
            "stratified": True,  # Store the stratified split semantics used by scale_and_split.
            "augmentation_merged_into_training_after_split": False,  # Record original-only fitted preprocessing semantics.
            "attack_types": normalize_metadata_for_json(attack_types_combined),  # Store combined-mode label scope when available.
        }  # Complete the split and evaluation identity payload.

        effective_cache_ref = None if artifact_recovery_target is not None else (cache_ref_file if cache_ref_file is not None else (file if file != "combined_files_combined" else None))  # Keep internal artifact recovery out of result-cache persistence.
        cache_dict = {}  # Initialize empty cache dictionary as fallback when no cache file exists
        if effective_cache_ref is not None and artifact_recovery_target is None:
            try:
                cache_dict = load_cache_results(effective_cache_ref, config=config)  # Load results keyed by the complete semantic identity.
                if cache_dict:
                    print(f"{BackgroundColors.GREEN}Resume: loaded {BackgroundColors.CYAN}{len(cache_dict)}{BackgroundColors.GREEN} cached result(s) from previous run.{Style.RESET_ALL}")
            except Exception:
                cache_dict = {}

        stacking_model = build_evaluation_stacking_model(base_models, config=config) if stacking_enabled else None  # Build only an unfitted stacking prototype.
        individual_models = {key: value for key, value in base_models.items() if artifact_recovery_target is None or (artifact_recovery_target[1] != "StackingClassifier" and key == artifact_recovery_target[1])}  # Restrict recovery without retaining fitted estimators.
        if grid_progress is None:
            total_steps = len(evaluation_plan)
            progress_bar = tqdm(total=total_steps, desc=f"{data_source_label} Data", file=sys.stdout)
            all_results = {}
            current_combination = 1
        else:
            total_steps = grid_progress["total_steps"]
            progress_bar = grid_progress["progress_bar"]
            all_results = {}
            current_combination = grid_progress["current_combination"]

        if experiment_mode == "original_training_augmented_testing":
            if df_augmented_for_testing is None or augmentation_ratio is None:
                raise ValueError("Augmented testing requires sampled augmented data and an augmentation ratio")
            augmented_features = df_augmented_for_testing.iloc[:, :-1].select_dtypes(include=np.number)
            missing_augmented_features = [feature for feature in pca_input_feature_names if feature not in augmented_features.columns]
            if missing_augmented_features:
                raise ValueError(f"Augmented testing data is missing original feature columns: {missing_augmented_features}")
            X_augmented_original_schema = augmented_features[pca_input_feature_names].to_numpy(copy=False)
            y_augmented_raw = df_augmented_for_testing.iloc[:, -1].to_numpy(copy=False)
            expected_label_classes = np.unique(df.iloc[:, -1].to_numpy(copy=False)).tolist()
            original_train_count = int(original_sample_count - math.ceil(original_sample_count * 0.2))
            feature_lookup = {sanitize_feature_name(feature): str(feature) for feature in pca_input_feature_names}
            explicit_features = feature_sets_config.get("explicit_features", []) or []
            selected_features_by_mode = {
                "Full Features": list(pca_input_feature_names),
                "Explicit Features": list(dict.fromkeys(feature_lookup[sanitize_feature_name(feature)] for feature in explicit_features if sanitize_feature_name(feature) in feature_lookup)),
                "GA Features": [feature_lookup[sanitize_feature_name(feature)] for feature in (ga_selected_features or []) if sanitize_feature_name(feature) in feature_lookup],
                "PCA Components": [f"PC{index + 1}" for index in range(min(int(pca_n_components or 0), len(pca_input_feature_names)))],
                "RFE Features": [feature_lookup[sanitize_feature_name(feature)] for feature in (rfe_selected_features or []) if sanitize_feature_name(feature) in feature_lookup],
            }
            dataset_name = build_filename_safe_dataset_identity(resolve_canonical_dataset_identity(str(file), True)) if execution_mode_str == "combined_files" else os.path.basename(os.path.dirname(file))
            evaluation_models = list(individual_models.items()) + ([('StackingClassifier', stacking_model)] if stacking_enabled else [])
            for name in feature_mode_names:
                subset_feature_names = selected_features_by_mode[name]
                expected_transformer = PCA(n_components=len(subset_feature_names)) if name == "PCA Components" else None
                for model_name, model_prototype in evaluation_models:
                    recovered, current_combination = recover_cached_individual_classifier_result(cache_dict, execution_mode_str, data_source_label, experiment_mode, augmentation_ratio, attack_types_combined, name, model_name, all_results, current_combination, total_steps, progress_bar, hyperparameters_enabled=hyperparameters_enabled, expected_n_features=len(subset_feature_names), expected_feature_names=subset_feature_names, expected_n_samples_train=original_train_count, expected_n_samples_test=len(y_augmented_raw))
                    if recovered:
                        continue
                    progress_bar.set_description(build_telegram_combination_header(name, model_name, augmentation_ratio, hyperparameters_enabled))  # Display the exact augmented-test combination being evaluated.
                    artifact_feature_set = f"{name} - {'Optimized Hyperparameters' if hyperparameters_enabled else 'Default Hyperparameters'}"
                    artifact_context = build_stacking_model_artifact_context(file, pca_source_files, execution_mode_str, attack_types_combined, target_column_name, model_name, model_prototype, artifact_feature_set, pca_input_feature_names, subset_feature_names, expected_label_classes, expected_transformer, hyperparameters_enabled)
                    artifact_bundle, rejection_reason = load_existing_model_if_available(model_name, file, dataset_name, artifact_feature_set, artifact_context, config=config)
                    if artifact_bundle is None:
                        print(f"{BackgroundColors.YELLOW}[WARNING] Retraining {BackgroundColors.CYAN}{model_name}{BackgroundColors.YELLOW} on original data only because {rejection_reason}.{Style.RESET_ALL}")
                        recovery_results = evaluate_on_dataset(file, df, feature_names, ga_selected_features, pca_n_components, rfe_selected_features, base_models, data_source_label="Original", hyperparams_map=hyperparams_map, experiment_id=generate_experiment_id(file, "original_only"), experiment_mode="original_only", augmentation_ratio=None, execution_mode_str=execution_mode_str, attack_types_combined=attack_types_combined, config=config, cache_ref_file=cache_ref_file, hyperparameters_enabled=hyperparameters_enabled, source_files=pca_source_files, artifact_recovery_target=(name, model_name))
                        del recovery_results
                        gc.collect()
                        artifact_bundle, rejection_reason = load_existing_model_if_available(model_name, file, dataset_name, artifact_feature_set, artifact_context, config=config)
                        if artifact_bundle is None:
                            raise RuntimeError(f"Original-only artifact recovery failed for {name} - {model_name}: {rejection_reason}")
                    loaded_model = artifact_bundle["model"]
                    X_augmented_scaled = np.asarray(artifact_bundle["scaler"].transform(X_augmented_original_schema))
                    if artifact_bundle["transformer"] is not None:
                        X_augmented_model = np.asarray(artifact_bundle["transformer"].transform(X_augmented_scaled))
                    else:
                        model_feature_indices = [artifact_bundle["input_feature_names"].index(feature) for feature in artifact_bundle["model_feature_names"]]
                        X_augmented_model = X_augmented_scaled[:, model_feature_indices]
                    y_augmented = np.asarray(artifact_bundle["label_encoder"].transform(y_augmented_raw), dtype=np.int64)
                    X_augmented_df = pd.DataFrame(X_augmented_model, columns=subset_feature_names)
                    training_ram_stats = {}
                    if model_name == "StackingClassifier":
                        metrics = evaluate_stacking_classifier(loaded_model, None, None, X_augmented_df, y_augmented, config=config, training_ram_stats=training_ram_stats, fit_model=False)
                        classifier_type = "Stacking"
                    else:
                        metrics = evaluate_individual_classifier(loaded_model, model_name, None, None, X_augmented_model, y_augmented, file, artifact_bundle["scaler"], subset_feature_names, artifact_feature_set, config=config, training_ram_stats=training_ram_stats, fit_model=False)
                        classifier_type = "Individual"
                    result_entry = build_classifier_result_entry(loaded_model.__class__.__name__, file, execution_mode_str, attack_types_combined, name, classifier_type, model_name, data_source_label, experiment_id, experiment_mode, augmentation_ratio, len(subset_feature_names), original_train_count, len(y_augmented), metrics, subset_feature_names, hyperparams_map=hyperparams_map, hyperparameters_enabled=hyperparameters_enabled, effective_hyperparameters=serialize_effective_estimator_parameters(loaded_model))
                    persist_cache_result_entry(effective_cache_ref, result_entry, cache_dict, config=config)
                    all_results[(name, model_name)] = result_entry
                    progress_bar.update(1)
                    current_combination += 1
                    del loaded_model, artifact_bundle, artifact_context, X_augmented_scaled, X_augmented_model, X_augmented_df, y_augmented, metrics, training_ram_stats
                    gc.collect()
            if grid_progress is None:
                progress_bar.close()
            else:
                grid_progress["current_combination"] = current_combination
            del X_augmented_original_schema, y_augmented_raw, augmented_features, df_augmented_for_testing
            gc.collect()
            return all_results

        data_splits = prepare_evaluation_data_splits(df, config=config)  # Prepare original-only training and testing data splits.

        if data_splits is None:  # If data preparation failed (single-class target)
            return {}  # Return empty dictionary

        X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder = data_splits  # Unpack original-only splits and fitted preprocessing.
        feature_source_arrays = {"X_train_scaled": X_train_scaled, "X_test_scaled": X_test_scaled, "spilled_to_memmap": False}  # Transfer full source matrices into a mutable lifecycle holder.
        del data_splits, X_train_scaled, X_test_scaled  # Remove duplicate local references so the holder can truly release or spill the full matrices.
        del df  # Release the original dataframe after split and scaling before classifier fitting.
        gc.collect()  # Reclaim released dataframe memory before feature-set materialization.

        feature_sets_iter = iterate_feature_sets_sequentially(feature_source_arrays, feature_names, ga_selected_features, pca_n_components, rfe_selected_features, file, feature_sets_config, config, scaler=scaler, source_files=pca_source_files, pca_cache_context=pca_cache_context, pca_input_feature_names=pca_input_feature_names)  # Create the lazy feature-set iterator with exact PCA source, feature, scaler, and split provenance.
        for idx, (name, X_train_subset, X_test_subset, subset_feature_names_list, transformer) in enumerate(feature_sets_iter, start=1):  # Evaluate one materialized feature set at a time
            if artifact_recovery_target is not None and name != artifact_recovery_target[0]:
                continue  # Materialize only as needed until the missing feature-set artifact is reached.
            if X_train_subset.shape[1] == 0:  # Verify if the subset is empty
                print(
                    f"{BackgroundColors.YELLOW}Warning: Skipping {name}. No features selected.{Style.RESET_ALL}"
                )  # Output warning
                write_memory_phase_event("before_feature_set_evaluation", config=config, dataset_identity=os.path.basename(str(file)), dataset_source=file, execution_mode=execution_mode_str, attack_scope=attack_types_combined, data_source=data_source_label, experiment_mode=experiment_mode, augmentation_ratio=augmentation_ratio, feature_set_name=name, hyperparameter_mode="Optimized Hyperparameters" if hyperparameters_enabled else "Default Hyperparameters", train_sample_count=len(y_train), test_sample_count=len(y_test), feature_count=0, event_outcome="skipped_empty")  # Publish skipped feature-set start
                write_memory_phase_event("after_feature_set_evaluation", config=config, dataset_identity=os.path.basename(str(file)), dataset_source=file, execution_mode=execution_mode_str, attack_scope=attack_types_combined, data_source=data_source_label, experiment_mode=experiment_mode, augmentation_ratio=augmentation_ratio, feature_set_name=name, hyperparameter_mode="Optimized Hyperparameters" if hyperparameters_enabled else "Default Hyperparameters", train_sample_count=len(y_train), test_sample_count=len(y_test), feature_count=0, event_outcome="skipped_empty")  # Publish skipped feature-set completion
                progress_bar.update(len(individual_models) + (1 if stacking_enabled else 0))  # Skip exactly the steps represented by this empty feature set
                current_combination += len(individual_models) + (1 if stacking_enabled else 0)  # Keep shared combination numbering aligned with skipped steps
                continue  # Skip to the next set

            feature_phase_metadata = {"dataset_identity": os.path.basename(str(file)), "dataset_source": file, "execution_mode": execution_mode_str, "attack_scope": attack_types_combined, "data_source": data_source_label, "experiment_mode": experiment_mode, "augmentation_ratio": augmentation_ratio, "feature_set_name": name, "hyperparameter_mode": "Optimized Hyperparameters" if hyperparameters_enabled else "Default Hyperparameters", "train_sample_count": len(y_train), "test_sample_count": len(y_test), "feature_count": X_train_subset.shape[1]}  # Build reusable feature-set metadata
            if memory_watcher_enabled(config):  # Add exact matrix layout only when watcher diagnostics are active
                feature_phase_metadata.update(build_array_memory_metadata("X_train_subset", X_train_subset))  # Add train subset memory metadata
                feature_phase_metadata.update(build_array_memory_metadata("X_test_subset", X_test_subset))  # Add test subset memory metadata
            write_memory_phase_event("before_feature_set_evaluation", config=config, **feature_phase_metadata, event_outcome="starting")  # Publish feature-set evaluation start
            individual_results, stacking_result_entry, current_combination = evaluate_single_feature_set(
                idx, name, X_train_subset, X_test_subset, subset_feature_names_list,
                individual_models, stacking_model, y_train, y_test,
                file, execution_mode_str, attack_types_combined,
                data_source_label, experiment_id, experiment_mode, augmentation_ratio,
                hyperparams_map, scaler, label_encoder, transformer, pca_input_feature_names, target_column_name, pca_source_files, total_steps, current_combination, progress_bar, stacking_enabled=stacking_enabled, config=config,
                cache_dict=cache_dict, cache_ref_file=effective_cache_ref, hyperparameters_enabled=hyperparameters_enabled,
            )  # Evaluate all individual classifiers and stacking model on this non-empty feature subset with resume support
            write_memory_phase_event("after_feature_set_evaluation", config=config, **feature_phase_metadata, event_outcome="completed")  # Publish feature-set evaluation completion

            all_results.update(individual_results)  # Merge this feature set's results into the global results dict
            if stacking_result_entry is not None:  # Store stacking result only when stacking evaluation was enabled
                all_results[(name, "StackingClassifier")] = stacking_result_entry  # Store stacking result with key
            del X_train_subset, X_test_subset, subset_feature_names_list, transformer, individual_results, stacking_result_entry  # Release feature-set arrays, transformer reference, and transient results after this mode
            gc.collect()  # Reclaim feature-set memory before constructing the next mode

        if grid_progress is None:  # Close only progress bars created by this standalone evaluation
            progress_bar.close()  # Close local progress bar
        else:  # Persist the next counter for the following HP/augmentation grid slice
            grid_progress["current_combination"] = current_combination  # Advance the shared full-grid combination counter
        if feature_sets_iter is not None and hasattr(feature_sets_iter, "close"):  # Close the generator before deleting source memmaps
            feature_sets_iter.close()  # Trigger generator-side subset cleanup if not already exhausted
        cleanup_feature_source_arrays(feature_source_arrays, config=config)  # Release full source arrays or temporary memmaps before returning result metadata
        feature_source_arrays = None  # Prevent duplicate cleanup in the outer finally
        del y_train, y_test, scaler, label_encoder  # Release labels and fitted preprocessing before returning result metadata
        gc.collect()  # Reclaim split/scaled arrays after all feature modes complete
        return all_results  # Return dictionary of results
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise
    finally:
        try:  # Always close the active feature iterator before source cleanup
            if feature_sets_iter is not None and hasattr(feature_sets_iter, "close"):  # Verify iterator supports close
                feature_sets_iter.close()  # Trigger generator-side cleanup on failures and interrupts
        except Exception:
            pass  # Do not mask the primary exception
        try:  # Always release feature source arrays and spill files
            cleanup_feature_source_arrays(feature_source_arrays, config=config)  # Cleanup is a no-op when already completed
        except Exception:
            pass  # Do not mask the primary exception


def determine_files_to_process(csv_file, input_path, config=None):
    """
    Determines which files to process based on CLI override or directory scan.

    :param csv_file: Optional CSV file path from CLI argument
    :param input_path: Directory path to search for CSV files
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: List of file paths to process
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Determining files to process from path: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        if csv_file:  # If a specific CSV file is provided via CLI
            try:  # Attempt to validate CSV file path
                abs_csv = os.path.abspath(csv_file)  # Get absolute path of CSV file
                abs_input = os.path.abspath(input_path)  # Get absolute path of input directory
                if abs_csv.startswith(abs_input):  # If CSV file belongs to this input path
                    return [csv_file]  # Return list with single CSV file
                else:  # CSV override does not belong to this path
                    return []  # Return empty list to skip this path
            except Exception:  # If validation fails
                return []  # Return empty list on error
        else:  # No CLI override, scan directory for dataset files
            dataset_format = config.get("stacking", {}).get("dataset_file_format", "csv")  # Read configured dataset file format
            dataset_extension = resolve_format_extension(dataset_format)  # Resolve format string to file extension
            return get_files_to_process(input_path, file_extension=dataset_extension, config=config)  # Get list of dataset files to process
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def combine_dataset_if_needed(files_to_process, config=None):
    """
    Combines multiple dataset files into one if PROCESS_ENTIRE_DATASET is enabled.

    :param files_to_process: List of file paths to process
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (combined_df, combined_file_for_features, updated_files_list) or (None, None, files_to_process)
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Verifying if dataset combination is needed...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message
        
        process_entire_dataset = config.get("execution", {}).get("process_entire_dataset", False)  # Get process entire dataset flag from config

        if process_entire_dataset and len(files_to_process) > 1:  # If combining is enabled and multiple files exist
            verbose_output(
                f"{BackgroundColors.GREEN}Attempting to combine {BackgroundColors.CYAN}{len(files_to_process)}{BackgroundColors.GREEN} dataset files...{Style.RESET_ALL}",
                config=config
            )  # Output the verbose message
            result = combine_dataset_files(files_to_process, config=config)  # Attempt to combine all files
            if result is not None:  # If combination was successful
                combined_df, combined_target_col = result  # Unpack the combined dataframe and target column
                combined_file_for_features = files_to_process[0]  # Use first file for feature selection metadata
                files_to_process = ["combined"]  # Replace file list with single "combined" entry
                return (combined_df, combined_file_for_features, files_to_process)  # Return combined data and updated file list
            else:  # If combination failed
                print(
                    f"{BackgroundColors.YELLOW}Warning: Could not combine dataset files. Processing individually.{Style.RESET_ALL}"
                )  # Output warning message

        return (None, None, files_to_process)  # Return original file list unchanged
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def load_and_preprocess_dataset(file, combined_df, config=None):
    """
    Loads and preprocesses a dataset file or uses combined dataframe.

    :param file: File path to load or "combined" keyword
    :param combined_df: Pre-combined dataframe (used if file == "combined")
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (df_cleaned, feature_names) or (None, None) if loading/preprocessing fails
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Loading and preprocessing dataset: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        if file == "combined":  # If using combined dataset
            df_original = combined_df  # Use the pre-combined dataframe
        else:  # Otherwise load from file
            df_original = load_dataset(file, config=config)  # Load the original dataset

        if df_original is None:  # If the dataset failed to load
            verbose_output(
                f"{BackgroundColors.RED}Failed to load dataset from: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}",
                config=config
            )  # Output the failure message
            return (None, None)  # Return None tuple
        
        df_cleaned = preprocess_dataframe(df_original, remove_zero_variance=False, config=config)  # Avoid learning a feature-removal state before the original train/test split.

        if df_cleaned is None or df_cleaned.empty:  # If the DataFrame is None or empty after preprocessing
            print(
                f"{BackgroundColors.RED}Dataset {BackgroundColors.CYAN}{file}{BackgroundColors.RED} empty after preprocessing. Skipping.{Style.RESET_ALL}"
            )  # Output error message
            return (None, None)  # Return None tuple

        feature_names = df_cleaned.iloc[:, :-1].select_dtypes(include=np.number).columns.tolist()  # Preserve every ordered numeric feature regardless of target dtype.

        return (df_cleaned, feature_names)  # Return cleaned dataframe and feature names
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def prepare_models_with_hyperparameters(file_path, config=None):
    """
    Prepares base models and applies hyperparameter optimization results if available.

    :param file_path: Path to the dataset file for loading hyperparameters
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (base_models, hp_params_map)
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Preparing models with hyperparameters for: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        base_models = get_models(config=config)  # Get the base models with default parameters

        hp_params_map = {}  # Initialize empty hyperparameters mapping
        hp_results_raw = extract_hyperparameter_optimization_results(file_path, config=config)  # Extract hyperparameter optimization results

        if hp_results_raw:  # If results were found, extract the params mapping and apply
            hp_params_map = {
                k: (v.get("best_params") if isinstance(v, dict) else v) for k, v in hp_results_raw.items()
            }  # Extract only the best_params mapping
            base_models = apply_hyperparameters_to_models(hp_params_map, base_models, config=config)  # Apply hyperparameters to base models
            verbose_output(
                f"{BackgroundColors.GREEN}Applied hyperparameters from optimization results{Style.RESET_ALL}",
                config=config
            )  # Output the verbose message

        return (base_models, hp_params_map)  # Return models and hyperparameters mapping
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def locate_and_verify_artifacts(file_path, config=None):
    """
    Locate and verify artifacts for feature selection, hyperparameters and data augmentation.

    :param file_path: Path to the dataset CSV file
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Dict with keys: 'ga', 'pca', 'rfe', 'hyperparams', 'augmented_file'
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Verifying artifacts for: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}",
            config=config,
        )  # Inform which file we're verifying

        artifacts = {}  # Prepare return mapping

        ga_features = None  # Placeholder for GA features
        try:
            ga_features = extract_genetic_algorithm_features(file_path, config=config)  # Try extracting GA features
        except Exception as e:  # Guard extraction to avoid raising
            print(f"{BackgroundColors.YELLOW}Warning: GA extraction failed for {file_path}: {e}{Style.RESET_ALL}")  # Warn on failure
            ga_features = None  # Normalize to None on failure
        artifacts["ga"] = ga_features  # Store GA features (or None)

        pca_n = None  # Placeholder for PCA components
        try:
            pca_n = extract_principal_component_analysis_features(file_path, config=config)  # Try extracting PCA n_components
        except Exception as e:  # Guard extraction
            print(f"{BackgroundColors.YELLOW}Warning: PCA extraction failed for {file_path}: {e}{Style.RESET_ALL}")  # Warn on failure
            pca_n = None  # Normalize to None
        artifacts["pca"] = pca_n  # Store PCA components (or None)

        rfe_sel = None  # Placeholder for RFE selected features
        try:
            rfe_sel = extract_recursive_feature_elimination_features(file_path, config=config)  # Try extracting RFE features
        except Exception as e:  # Guard extraction
            print(f"{BackgroundColors.YELLOW}Warning: RFE extraction failed for {file_path}: {e}{Style.RESET_ALL}")  # Warn on failure
            rfe_sel = None  # Normalize to None
        artifacts["rfe"] = rfe_sel  # Store RFE selection (or None)

        hyperparams = None  # Placeholder for hyperparameters mapping
        try:
            hyperparams = extract_hyperparameter_optimization_results(file_path, config=config)  # Try locating HP CSV and parsing
        except Exception as e:  # Guard extraction
            print(f"{BackgroundColors.YELLOW}Warning: Hyperparameter extraction failed for {file_path}: {e}{Style.RESET_ALL}")  # Warn on failure
            hyperparams = None  # Normalize to None
        artifacts["hyperparams"] = hyperparams  # Store HP mapping (or None)

        augmented = None  # Placeholder for augmented CSV path
        try:
            augmented = find_data_augmentation_file(file_path, config=config)  # Try locating augmented CSV using config-aware finder
            if augmented:  # If found
                # Also collect PNG image path with same stem if present
                aug_png = Path(augmented).with_suffix(".png")  # Build PNG path by changing suffix
                if aug_png.exists():  # If PNG exists
                    artifacts["augmented_image"] = str(aug_png)  # Store image path
        except Exception as e:  # Guard finder
            print(f"{BackgroundColors.YELLOW}Warning: Data augmentation lookup failed for {file_path}: {e}{Style.RESET_ALL}")  # Warn on failure
            augmented = None  # Normalize to None
        artifacts["augmented_file"] = augmented  # Store augmented CSV path (or None)

        return artifacts  # Return mapping of discovered artifacts
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_metrics_from_result(result):
    """
    Extracts metrics from a result dictionary into a list.

    :param result: Result dictionary containing metric keys
    :return: List of [accuracy, precision, recall, f1_score, fpr, fnr, elapsed_time_s]
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Extracting metrics from result dictionary...{Style.RESET_ALL}"
        )  # Output the verbose message

        return [
            result.get("accuracy", 0),  # Get accuracy or default to 0
            result.get("precision", 0),  # Get precision or default to 0
            result.get("recall", 0),  # Get recall or default to 0
            result.get("f1_score", 0),  # Get F1 score or default to 0
            result.get("fpr", 0),  # Get false positive rate or default to 0
            result.get("fnr", 0),  # Get false negative rate or default to 0
            result.get("elapsed_time_s", 0),  # Get elapsed time or default to 0
        ]  # Return list of metric values
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def calculate_all_improvements(orig_metrics, merged_metrics):
    """
    Calculates improvement percentages for all metrics comparing original vs merged data.

    :param orig_metrics: List of original metrics [accuracy, precision, recall, f1, fpr, fnr, time]
    :param merged_metrics: List of merged metrics [accuracy, precision, recall, f1, fpr, fnr, time]
    :return: Dictionary of improvement percentages for each metric
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Calculating metric improvements...{Style.RESET_ALL}"
        )  # Output the verbose message

        return {
            "accuracy": calculate_metric_improvement(orig_metrics[0], merged_metrics[0]),  # Calculate accuracy improvement
            "precision": calculate_metric_improvement(orig_metrics[1], merged_metrics[1]),  # Calculate precision improvement
            "recall": calculate_metric_improvement(orig_metrics[2], merged_metrics[2]),  # Calculate recall improvement
            "f1_score": calculate_metric_improvement(orig_metrics[3], merged_metrics[3]),  # Calculate F1 score improvement
            "fpr": calculate_metric_improvement(orig_metrics[4], merged_metrics[4]),  # Calculate FPR change (lower is better)
            "fnr": calculate_metric_improvement(orig_metrics[5], merged_metrics[5]),  # Calculate FNR change (lower is better)
            "training_time": calculate_metric_improvement(orig_metrics[6], merged_metrics[6]),  # Calculate time change (lower is better)
        }  # Return dictionary of improvements
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def print_model_comparison(feature_set, model_name, orig_metrics, aug_metrics, merged_metrics, improvements):
    """
    Prints detailed comparison of metrics for a single model across data sources.

    :param feature_set: Name of the feature set used
    :param model_name: Name of the model
    :param orig_metrics: List of original data metrics
    :param aug_metrics: List of augmented data metrics
    :param merged_metrics: List of merged data metrics
    :param improvements: Dictionary of improvement percentages
    :return: None
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Printing comparison for model: {BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN}, feature set: {BackgroundColors.CYAN}{feature_set}{Style.RESET_ALL}"
        )  # Output the verbose message

        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Feature Set: {BackgroundColors.CYAN}{feature_set}{BackgroundColors.GREEN} | Model: {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}"
        )  # Print header with feature set and model name

        acc_label = f"  {BackgroundColors.YELLOW}Accuracy:{Style.RESET_ALL}"  # Build accuracy label
        print(acc_label)  # Print accuracy label
        acc_values = f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {orig_metrics[0]} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {aug_metrics[0]} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {merged_metrics[0]} | {BackgroundColors.CYAN}Improvement: {improvements['accuracy']}%{Style.RESET_ALL}"  # Build accuracy comparison line using raw floats
        print(acc_values)  # Print accuracy comparison

        prec_label = f"  {BackgroundColors.YELLOW}Precision:{Style.RESET_ALL}"  # Build precision label
        print(prec_label)  # Print precision label
        prec_values = f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {orig_metrics[1]} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {aug_metrics[1]} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {merged_metrics[1]} | {BackgroundColors.CYAN}Improvement: {improvements['precision']}%{Style.RESET_ALL}"  # Build precision comparison line using raw floats
        print(prec_values)  # Print precision comparison

        recall_label = f"  {BackgroundColors.YELLOW}Recall:{Style.RESET_ALL}"  # Build recall label
        print(recall_label)  # Print recall label
        recall_values = f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {orig_metrics[2]} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {aug_metrics[2]} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {merged_metrics[2]} | {BackgroundColors.CYAN}Improvement: {improvements['recall']}%{Style.RESET_ALL}"  # Build recall comparison line using raw floats
        print(recall_values)  # Print recall comparison

        f1_label = f"  {BackgroundColors.YELLOW}F1-Score:{Style.RESET_ALL}"  # Build F1 score label
        print(f1_label)  # Print F1 score label
        f1_values = f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {orig_metrics[3]} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {aug_metrics[3]} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {merged_metrics[3]} | {BackgroundColors.CYAN}Improvement: {improvements['f1_score']}%{Style.RESET_ALL}"  # Build F1 score comparison using raw floats
        print(f1_values)  # Print F1 score comparison

        fpr_label = f"  {BackgroundColors.YELLOW}FPR (lower is better):{Style.RESET_ALL}"  # Build FPR label
        print(fpr_label)  # Print FPR label
        fpr_values = f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {orig_metrics[4]} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {aug_metrics[4]} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {merged_metrics[4]} | {BackgroundColors.CYAN}Change: {improvements['fpr']}%{Style.RESET_ALL}"  # Build FPR comparison using raw floats
        print(fpr_values)  # Print FPR comparison

        fnr_label = f"  {BackgroundColors.YELLOW}FNR (lower is better):{Style.RESET_ALL}"  # Build FNR label
        print(fnr_label)  # Print FNR label
        fnr_values = f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {orig_metrics[5]} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {aug_metrics[5]} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {merged_metrics[5]} | {BackgroundColors.CYAN}Change: {improvements['fnr']}%{Style.RESET_ALL}"  # Build FNR comparison using raw floats
        print(fnr_values)  # Print FNR comparison

        time_label = f"  {BackgroundColors.YELLOW}Training Time (seconds, lower is better):{Style.RESET_ALL}"  # Build training time label
        print(time_label)  # Print training time label
        time_values = f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {int(orig_metrics[6])}s | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {int(aug_metrics[6])}s | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {int(merged_metrics[6])}s | {BackgroundColors.CYAN}Change: {improvements['training_time']}%{Style.RESET_ALL}\n"  # Build training time comparison line using integer seconds
        print(time_values)  # Print training time comparison
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_comparison_result_entry(orig_result, feature_set, classifier_type, model_name, data_source, metrics, improvements, n_features_override=None, n_samples_train_override=None, n_samples_test_override=None, experiment_id=None, experiment_mode="original_only", augmentation_ratio=None):
    """
    Builds a single comparison result entry for CSV export.

    :param orig_result: Original result dictionary for base metadata
    :param feature_set: Name of the feature set
    :param classifier_type: Type of classifier (e.g., 'Individual' or 'Stacking')
    :param model_name: Name of the model
    :param data_source: Data source label (e.g., 'Original', 'Augmented@50%')
    :param metrics: List of metrics [accuracy, precision, recall, f1, fpr, fnr, time]
    :param improvements: Dictionary of improvement percentages
    :param n_features_override: Override for n_features (optional)
    :param n_samples_train_override: Override for n_samples_train (optional)
    :param n_samples_test_override: Override for n_samples_test (optional)
    :param experiment_id: Unique experiment identifier for traceability
    :param experiment_mode: Experiment mode string ('original_only' or 'original_training_augmented_testing')
    :param augmentation_ratio: Augmentation ratio float or None
    :return: Dictionary containing comparison result entry
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Building comparison result entry for: {BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN}, data source: {BackgroundColors.CYAN}{data_source}{Style.RESET_ALL}"
        )  # Output the verbose message

        return {
            "dataset": orig_result["dataset"],  # Dataset name from original result
            "feature_set": feature_set,  # Feature set name
            "classifier_type": classifier_type,  # Classifier type
            "model_name": model_name,  # Model name
            "data_source": data_source,  # Data source label
            "experiment_id": experiment_id,  # Unique experiment identifier for traceability
            "experiment_mode": experiment_mode,  # Persist original-only or augmented-testing semantics.
            "augmentation_ratio": resolve_persisted_augmentation_ratio(experiment_mode, augmentation_ratio),  # Augmentation ratio with 0.0 for original-only rows
            "n_features": n_features_override if n_features_override is not None else orig_result["n_features"],  # Number of features
            "n_samples_train": n_samples_train_override if n_samples_train_override is not None else orig_result["n_samples_train"],  # Training samples count
            "n_samples_test": n_samples_test_override if n_samples_test_override is not None else orig_result["n_samples_test"],  # Test samples count
            "accuracy": metrics[0],  # Accuracy metric
            "precision": metrics[1],  # Precision metric
            "recall": metrics[2],  # Recall metric
            "f1_score": metrics[3],  # F1 score metric
            "fpr": metrics[4],  # False positive rate
            "fnr": metrics[5],  # False negative rate
            "training_time": metrics[6],  # Training time in seconds
            "accuracy_improvement": improvements.get("accuracy", 0.0),  # Accuracy improvement percentage
            "precision_improvement": improvements.get("precision", 0.0),  # Precision improvement percentage
            "recall_improvement": improvements.get("recall", 0.0),  # Recall improvement percentage
            "f1_score_improvement": improvements.get("f1_score", 0.0),  # F1 score improvement percentage
            "fpr_improvement": improvements.get("fpr", 0.0),  # FPR improvement percentage
            "fnr_improvement": improvements.get("fnr", 0.0),  # FNR improvement percentage
            "training_time_improvement": improvements.get("training_time", 0.0),  # Training time improvement percentage
            "features_list": orig_result["features_list"],  # List of feature names used
        }  # Return comparison result entry dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def generate_ratio_comparison_report(results_original, all_ratio_results, config=None):
    """
    Generates and prints comparison report for ratio-based data augmentation evaluation.
    Compares the original baseline against each augmentation ratio experiment.

    :param results_original: Dictionary of results from original data evaluation
    :param all_ratio_results: Dictionary mapping ratio (float) to results dictionary
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: List of comparison result entries for CSV export
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Generating ratio-based data augmentation comparison report...{Style.RESET_ALL}"
        )  # Output the verbose message

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*100}{Style.RESET_ALL}"
        )  # Print separator line for visual clarity
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}DATA AUGMENTATION RATIO-BASED COMPARISON REPORT{Style.RESET_ALL}"
        )  # Print report header title
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*100}{Style.RESET_ALL}\n"
        )  # Print closing separator line

        comparison_results = []  # Initialize list for comparison result entries
        no_improvements = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "fpr": 0.0, "fnr": 0.0, "training_time": 0.0}  # Zero improvements dict for original baseline entries

        for key in results_original.keys():  # Iterate through each feature_set/model combination from original results
            orig_result = results_original[key]  # Get the original baseline result entry
            feature_set = orig_result["feature_set"]  # Extract feature set name from result
            model_name = orig_result["model_name"]  # Extract model name from result
            classifier_type = orig_result["classifier_type"]  # Extract classifier type from result
            orig_metrics = extract_metrics_from_result(orig_result)  # Extract metrics list from original result
            orig_experiment_id = orig_result.get("experiment_id", None)  # Get experiment ID from original result

            comparison_results.append(
                build_comparison_result_entry(
                    orig_result, feature_set, classifier_type, model_name, "Original",
                    orig_metrics, no_improvements,
                    experiment_id=orig_experiment_id, experiment_mode="original_only", augmentation_ratio=None,
                )
            )  # Add original baseline entry to comparison results

            header_msg = f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Feature Set: {BackgroundColors.CYAN}{feature_set}{BackgroundColors.GREEN} | Model: {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}"  # Build header with feature set and model name
            print(header_msg)  # Print header with feature set and model name
            cfg = config if config is not None else CONFIG  # Resolve config or fallback to global CONFIG
            mode_raw = cfg.get("execution", {}).get("execution_mode")  # Obtain execution mode from config if present
            evaluation_mode = mode_raw.replace("_", " ").title().replace(" ", "") if mode_raw else "SeparateFiles"  # Normalize or default
            total_seconds_orig = int(round(orig_metrics[6]))  # Total seconds from original metrics
            human_time_orig = calculate_execution_time(0, total_seconds_orig)  # Human-readable original elapsed time
            msg = f"{BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN}: Mode {BackgroundColors.YELLOW}{evaluation_mode}{BackgroundColors.GREEN} | F1-Score {BackgroundColors.CYAN}{orig_metrics[3]}{BackgroundColors.GREEN} | Accuracy: {BackgroundColors.CYAN}{orig_metrics[0]}{BackgroundColors.GREEN} | Precision: {BackgroundColors.CYAN}{orig_metrics[1]}{BackgroundColors.GREEN} | Recall: {BackgroundColors.CYAN}{orig_metrics[2]}{BackgroundColors.GREEN} | FPR: {BackgroundColors.CYAN}{orig_metrics[4]}{BackgroundColors.GREEN} | FNR: {BackgroundColors.CYAN}{orig_metrics[5]}{BackgroundColors.GREEN} | Training Time: {BackgroundColors.CYAN}{int(total_seconds_orig)}s{BackgroundColors.GREEN} | Execution Time: {BackgroundColors.CYAN}{int(total_seconds_orig)}s{BackgroundColors.GREEN} | Total Time: {BackgroundColors.CYAN}{human_time_orig} ({total_seconds_orig}s){Style.RESET_ALL}"  # Build colored original baseline summary using raw floats and integer times
            print(msg)  # Print original baseline metrics summary

            for ratio in sorted(all_ratio_results.keys()):  # Iterate over each ratio in sorted order
                ratio_results = all_ratio_results[ratio]  # Get results dict for this ratio
                ratio_result = ratio_results.get(key)  # Get the matching result for this feature_set/model key

                if ratio_result is None:  # If no matching result exists for this ratio
                    continue  # Skip this ratio for this model/feature_set combination

                ratio_metrics = extract_metrics_from_result(ratio_result)  # Extract metrics list from ratio result
                improvements = calculate_all_improvements(orig_metrics, ratio_metrics)  # Calculate improvements vs original
                ratio_pct = int(ratio * 100)  # Convert float ratio to integer percentage for display
                ratio_experiment_id = ratio_result.get("experiment_id", None)  # Get experiment ID from ratio result

                comparison_results.append(
                    build_comparison_result_entry(
                        orig_result, feature_set, classifier_type, model_name,
                        f"Augmented@{ratio_pct}%", ratio_metrics, improvements,
                        n_features_override=ratio_result.get("n_features"),
                        n_samples_train_override=ratio_result.get("n_samples_train"),
                        n_samples_test_override=ratio_result.get("n_samples_test"),
                        experiment_id=ratio_experiment_id, experiment_mode="original_training_augmented_testing",
                        augmentation_ratio=ratio,
                    )
                )  # Add ratio experiment entry with improvements to comparison results

                f1_improvement = improvements.get("f1_score", 0.0)  # Extract F1 improvement for display
                improvement_color = BackgroundColors.GREEN if f1_improvement >= 0 else BackgroundColors.RED  # Choose color based on improvement direction
                total_seconds_ratio = int(round(ratio_metrics[6]))  # Total seconds from ratio experiment metrics
                human_time_ratio = calculate_execution_time(0, total_seconds_ratio)  # Human-readable ratio elapsed time
                msg = f"{BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN}: Mode {BackgroundColors.YELLOW}{evaluation_mode}{BackgroundColors.GREEN} | F1-Score {BackgroundColors.CYAN}{ratio_metrics[3]}{BackgroundColors.GREEN} | Accuracy: {BackgroundColors.CYAN}{ratio_metrics[0]}{BackgroundColors.GREEN} | Precision: {BackgroundColors.CYAN}{ratio_metrics[1]}{BackgroundColors.GREEN} | Recall: {BackgroundColors.CYAN}{ratio_metrics[2]}{BackgroundColors.GREEN} | FPR: {BackgroundColors.CYAN}{ratio_metrics[4]}{BackgroundColors.GREEN} | FNR: {BackgroundColors.CYAN}{ratio_metrics[5]}{BackgroundColors.GREEN} | Training Time: {BackgroundColors.CYAN}{int(total_seconds_ratio)}s{BackgroundColors.GREEN} | Execution Time: {BackgroundColors.CYAN}{int(total_seconds_ratio)}s{BackgroundColors.GREEN} | Total Time: {BackgroundColors.CYAN}{human_time_ratio} ({total_seconds_ratio}s){Style.RESET_ALL}"  # Build colored ratio summary using raw floats and integer times
                print(msg)  # Print ratio result metrics with F1 improvement indicator

        return comparison_results  # Return list of all comparison result entries for CSV export
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def load_and_validate_augmented_data(file, df_original_cleaned, config=None):
    """
    Locates, loads, preprocesses, and validates augmented data for the given file.

    :param file: Original file path used to locate the augmented counterpart
    :param df_original_cleaned: Cleaned original dataframe for validation reference
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Cleaned augmented dataframe, or None if loading or validation fails
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        augmented_file = find_data_augmentation_file(file, config=config)  # Look for augmented data file using wgangp.py naming convention

        if augmented_file is None:  # If no augmented file found at expected path
            print(
                f"\n{BackgroundColors.YELLOW}No augmented data found for this file. Skipping augmentation comparison.{Style.RESET_ALL}"
            )  # Print warning message about missing augmented file
            return None  # Signal caller that no augmented data is available

        df_augmented = load_augmented_dataset(augmented_file, config=config)  # Load the augmented dataset using configured augmentation format

        if df_augmented is None:  # If augmented dataset failed to load from disk
            print(
                f"{BackgroundColors.YELLOW}Warning: Failed to load augmented dataset from {BackgroundColors.CYAN}{augmented_file}{BackgroundColors.YELLOW}. Skipping.{Style.RESET_ALL}"
            )  # Print warning message about load failure
            return None  # Signal caller that loading failed

        df_augmented_cleaned = preprocess_dataframe(df_augmented, remove_zero_variance=False, config=config)  # Preserve the original fitted feature schema for inference-only augmentation.

        if not validate_augmented_dataframe(df_original_cleaned, df_augmented_cleaned, file):  # Validate augmented data is compatible with original
            return None  # Signal caller that validation failed

        return df_augmented_cleaned  # Return the cleaned and validated augmented dataframe
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_single_ratio_experiment(file, df_original_cleaned, df_augmented_cleaned, feature_names, ga_selected_features, pca_n_components, rfe_selected_features, base_models, hp_params_map, ratio, ratio_idx, total_ratios, config=None):
    """
    Executes a single ratio-based augmentation experiment by sampling, visualizing, and evaluating.

    :param file: Original file path
    :param df_original_cleaned: Cleaned original dataframe
    :param df_augmented_cleaned: Cleaned augmented dataframe to sample from
    :param feature_names: List of feature names
    :param ga_selected_features: Features selected by genetic algorithm
    :param pca_n_components: Number of PCA components
    :param rfe_selected_features: Features selected by RFE
    :param base_models: Dictionary of base models
    :param hp_params_map: Hyperparameters mapping
    :param ratio: Current augmentation ratio (float between 0 and 1)
    :param ratio_idx: Current ratio index (1-based) for progress display
    :param total_ratios: Total number of ratios being evaluated
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Results list from evaluate_on_dataset, or None if sampling fails
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        ratio_pct = int(ratio * 100)  # Convert float ratio to integer percentage for display
        experiment_id = generate_experiment_id(file, "original_training_augmented_testing", ratio)  # Generate unique experiment ID for this ratio

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}[{ratio_idx}/{total_ratios}] Evaluating persisted original-trained models on Augmented@{ratio_pct}%{Style.RESET_ALL}"
        )  # Print progress indicator for current ratio experiment
        df_sampled = sample_augmented_by_ratio(df_augmented_cleaned, df_original_cleaned, ratio)  # Sample augmented rows at the current ratio

        if df_sampled is None or df_sampled.empty:  # If sampling returned no valid data
            print(
                f"{BackgroundColors.YELLOW}Warning: Could not sample augmented data at ratio {ratio}. Skipping this ratio.{Style.RESET_ALL}"
            )  # Print warning about sampling failure
            return None  # Signal caller to skip this ratio

        data_source_label = f"Augmented@{ratio_pct}%"  # Build descriptive augmented-testing label for CSV traceability

        print(
            f"{BackgroundColors.GREEN}Sampled augmented test dataset: {BackgroundColors.CYAN}{len(df_sampled)} samples at {ratio_pct}% ratio{Style.RESET_ALL}"
        )  # Print sampled dataset size for transparency

        generate_augmentation_tsne_visualization(
            file, df_original_cleaned, df_sampled, ratio, "original_training_augmented_testing"
        )  # Generate t-SNE visualization for this augmentation ratio

        results_ratio = evaluate_on_dataset(
            file, df_original_cleaned, feature_names, ga_selected_features, pca_n_components,
            rfe_selected_features, base_models, data_source_label=data_source_label,
            hyperparams_map=hp_params_map, experiment_id=experiment_id,
            experiment_mode="original_training_augmented_testing", augmentation_ratio=ratio,
            execution_mode_str="separate_files", attack_types_combined=None,
            df_augmented_for_testing=df_sampled, config=config,
        )  # Evaluate persisted original-trained classifiers on augmented samples only.

        del df_sampled  # Release sampled augmented data to free memory after evaluation
        gc.collect()  # Force garbage collection to reclaim memory from sampled data

        return results_ratio  # Return the evaluation results for this ratio
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def process_augmented_data_evaluation(file, df_original_cleaned, feature_names, ga_selected_features, pca_n_components, rfe_selected_features, base_models, hp_params_map, results_original, config=None):
    """
    Handles complete augmented data evaluation workflow with ratio-based experiments.
    For each ratio in config.get("stacking", {}).get("augmentation_ratios", [0.25, 0.50, 0.75, 1.00]), samples augmented data proportionally,
    merges with original, evaluates classifiers, and compares against original baseline.

    :param file: Original file path
    :param df_original_cleaned: Cleaned original dataframe
    :param feature_names: List of feature names
    :param ga_selected_features: Features selected by genetic algorithm
    :param pca_n_components: Number of PCA components
    :param rfe_selected_features: Features selected by RFE
    :param base_models: Dictionary of base models
    :param hp_params_map: Hyperparameters mapping
    :param results_original: Results from original data evaluation
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Processing augmented data evaluation for: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}"
        )  # Output the verbose message

        df_augmented_cleaned = load_and_validate_augmented_data(file, df_original_cleaned, config=config)  # Load, preprocess, and validate augmented data

        if df_augmented_cleaned is None:  # If augmented data could not be loaded or validated
            return  # Exit function early when no valid augmented data is available

        augmentation_ratios = config.get("stacking", {}).get("augmentation_ratios", [0.25, 0.50, 0.75, 1.00])  # Retrieve the list of ratios to evaluate

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*100}{Style.RESET_ALL}"
        )  # Print separator line for visual clarity
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}RATIO-BASED DATA AUGMENTATION EXPERIMENTS{Style.RESET_ALL}"
        )  # Print header for the ratio-based experiments section
        print(
            f"{BackgroundColors.GREEN}Ratios to evaluate: {BackgroundColors.CYAN}{[f'{int(r*100)}%' for r in augmentation_ratios]}{Style.RESET_ALL}"
        )  # Print the list of ratios that will be evaluated
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*100}{Style.RESET_ALL}\n"
        )  # Print closing separator line
        send_telegram_message(TELEGRAM_BOT, [f"[SEPARATE_FILES] Starting ratio-based augmentation experiments | file: {os.path.basename(file)} | ratios: {[f'{int(r * 100)}%' for r in augmentation_ratios]}"])  # Notify Telegram about augmentation ratio experiments start

        all_ratio_results = {}  # Dictionary to store results for each ratio: {ratio: results_dict}

        for ratio_idx, ratio in enumerate(augmentation_ratios, start=1):  # Iterate over each augmentation ratio
            results_ratio = run_single_ratio_experiment(
                file, df_original_cleaned, df_augmented_cleaned, feature_names,
                ga_selected_features, pca_n_components, rfe_selected_features,
                base_models, hp_params_map, ratio, ratio_idx, len(augmentation_ratios), config
            )  # Execute the experiment for this specific ratio

            if results_ratio is not None:  # If the ratio experiment produced valid results
                all_ratio_results[ratio] = results_ratio  # Store the results for this ratio in the results dictionary

        del df_augmented_cleaned  # Release augmented data to free memory after all ratio experiments
        gc.collect()  # Force garbage collection to reclaim memory from augmented data

        if not all_ratio_results:  # If no ratio experiments produced valid results
            print(
                f"{BackgroundColors.YELLOW}Warning: No ratio experiments completed successfully. Skipping comparison report.{Style.RESET_ALL}"
            )  # Print warning about no completed experiments
            return  # Exit function early when no results are available

        comparison_results = generate_ratio_comparison_report(results_original, all_ratio_results, config=config)  # Generate the comparison report across all ratios

        save_augmentation_comparison_results(file, comparison_results)  # Save comparison results to CSV file

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Data augmentation ratio-based comparison complete!{Style.RESET_ALL}"
        )  # Print success message indicating all ratio experiments are done
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def save_combined_files_results_to_csv(reference_file, results_list, config=None):
    """
    Save combined files evaluation results to the Feature_Analysis directory.

    :param reference_file: Reference file path for determining output directory
    :param results_list: List of result dictionaries to save
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Path object of the Feature_Analysis directory for reuse
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        combined_files_results_filename = config.get("stacking", {}).get("combined_files_results_filename", "Stacking_Classifiers_CombinedFiles_Results.csv")  # Get combined files evaluation results filename from config
        reference_file_path = Path(reference_file)  # Create Path object from reference file
        dataset_root = resolve_dataset_root_path(str(reference_file_path))  # Resolve Feature_Analysis root from file or directory input.
        feature_analysis_dir = dataset_root / "Feature_Analysis"  # Build Feature_Analysis directory path from dataset root.
        os.makedirs(feature_analysis_dir, exist_ok=True)  # Ensure directory exists on disk

        combined_config = dict(config)  # Copy configuration so combined export can override only the result filename
        combined_stacking_config = dict(combined_config.get("stacking", {}))  # Copy stacking configuration before filename override
        combined_stacking_config["results_filename"] = combined_files_results_filename  # Use the configured combined-files result filename
        combined_config["stacking"] = combined_stacking_config  # Store the filename override in the copied configuration
        save_stacking_results(str(reference_file_path), results_list, config=combined_config)  # Save combined files evaluation results with the combined filename

        return feature_analysis_dir  # Return Feature_Analysis directory path for reuse
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def save_combined_files_augmentation_comparison(results_original, all_ratio_results, feature_analysis_dir, config=None, comparison_results=None):
    """
    Generates a ratio comparison report and saves it to a CSV file in the feature analysis directory.

    :param results_original: Results dictionary from the original-data-only evaluation
    :param all_ratio_results: Dictionary mapping augmentation ratio floats to their evaluation result dicts
    :param feature_analysis_dir: Path object pointing to the Feature_Analysis output directory
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        if comparison_results is None and not all_ratio_results:  # If no prebuilt comparisons or ratio experiments produced results
            print(
                f"{BackgroundColors.YELLOW}No augmentation ratio experiments completed successfully for combined files evaluation.{Style.RESET_ALL}"
            )  # Print warning about no completed experiments
            return  # Exit early since there is nothing to compare

        if comparison_results is None:  # Build comparisons from result mappings for legacy callers
            comparison_results = generate_ratio_comparison_report(results_original, all_ratio_results, config=config)  # Generate comparison report across all evaluated augmentation ratios

        augmentation_comparison_filename = config.get("stacking", {}).get("augmentation_comparison_filename", "Data_Augmentation_Comparison_Results.csv")  # Get base comparison filename from config
        combined_files_comparison_filename = augmentation_comparison_filename.replace(".csv", "_CombinedFiles.csv")  # Build combined files evaluation-specific comparison filename
        combined_files_comparison_path = feature_analysis_dir / combined_files_comparison_filename  # Construct full output path inside Feature_Analysis directory

        save_augmentation_comparison_results(str(combined_files_comparison_path), comparison_results, config=config)  # Save comparison results to the CSV file

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Combined files evaluation data augmentation ratio-based comparison complete!{Style.RESET_ALL}"
        )  # Print success message indicating comparison report has been saved
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_combined_files_augmentation_ratio_experiment(reference_file, combined_files_df, combined_augmented_df, feature_names, ga_selected_features, pca_n_components, rfe_selected_features, base_models, hp_params_map, attack_types_list, ratio, ratio_idx, total_steps, dataset_name, config=None):
    """
    Runs a single ratio-based augmentation experiment for combined files evaluation and returns the results.

    :param reference_file: Reference file path used for t-SNE visualization and experiment ID generation
    :param combined_files_df: Combined original combined files evaluation DataFrame with all attack types
    :param combined_augmented_df: Combined augmented DataFrame used as the augmentation source
    :param feature_names: List of feature column names for the dataset
    :param ga_selected_features: List of features selected by the genetic algorithm
    :param pca_n_components: Number of PCA components or None if PCA is disabled
    :param rfe_selected_features: List of features selected by RFE or None if disabled
    :param base_models: Dictionary mapping model names to model objects
    :param hp_params_map: Dictionary mapping model names to hyperparameter dicts
    :param attack_types_list: List of unique attack type labels for combined files evaluation
    :param ratio: Augmentation ratio float between 0 and 1
    :param ratio_idx: 1-based index of this ratio in the augmentation schedule
    :param total_steps: Total number of evaluation steps for progress display
    :param dataset_name: Name of the dataset for Telegram notification messages
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Results dictionary from evaluate_on_dataset, or None if sampling failed
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        ratio_pct = int(ratio * 100)  # Convert ratio float to integer percentage for display

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}[{ratio_idx + 1}/{total_steps}] Evaluating with {ratio_pct}% augmented data (Combined Files Evaluation){Style.RESET_ALL}"
        )  # Print experiment step progress for this ratio

        df_sampled = sample_augmented_by_ratio(combined_augmented_df, combined_files_df, ratio)  # Sample augmented data proportionally to the requested ratio

        if df_sampled is None:  # If sampling failed for this ratio
            print(
                f"{BackgroundColors.YELLOW}Failed to sample augmented data at ratio {ratio_pct}%. Skipping.{Style.RESET_ALL}"
            )  # Print warning about sampling failure
            return None  # Signal caller to skip this ratio

        data_source_label = f"Augmented@{ratio_pct}%_CombinedFiles"  # Build data source label for result traceability
        experiment_id = generate_experiment_id(reference_file, "combined_files_original_training_augmented_testing", ratio)  # Generate unique experiment ID for this run

        print(
            f"{BackgroundColors.GREEN}Sampled augmented test dataset: {BackgroundColors.CYAN}{len(df_sampled)} samples at {ratio_pct}% ratio{Style.RESET_ALL}"
        )  # Print sampled dataset size for transparency

        generate_augmentation_tsne_visualization(
            reference_file, combined_files_df, df_sampled, ratio, "original_training_augmented_testing"
        )  # Generate t-SNE visualization comparing original and augmented distributions

        results_ratio = evaluate_on_dataset(
            reference_file, combined_files_df, feature_names, ga_selected_features, pca_n_components,
            rfe_selected_features, base_models, data_source_label=data_source_label,
            hyperparams_map=hp_params_map, experiment_id=experiment_id,
            experiment_mode="original_training_augmented_testing", augmentation_ratio=ratio,
            execution_mode_str="combined_files", attack_types_combined=attack_types_list,
            df_augmented_for_testing=df_sampled, config=config,
        )  # Evaluate persisted original-trained classifiers on augmented samples only.

        return results_ratio  # Return evaluation results for this ratio
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def print_combined_files_augmentation_header(augmentation_ratios):
    """
    Prints the formatted header block announcing ratio-based data augmentation experiments for combined files evaluation mode.

    :param augmentation_ratios: List of augmentation ratio floats to be evaluated
    :return: None
    """

    try:
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*100}{Style.RESET_ALL}"
        )  # Print top separator line for the augmentation experiments section
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}RATIO-BASED DATA AUGMENTATION EXPERIMENTS (Combined Files Evaluation){Style.RESET_ALL}"
        )  # Print the section title for ratio-based combined files evaluation augmentation
        print(
            f"{BackgroundColors.GREEN}Ratios to evaluate: {BackgroundColors.CYAN}{[f'{int(r*100)}%' for r in augmentation_ratios]}{Style.RESET_ALL}"
        )  # Print the list of augmentation ratios that will be evaluated
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*100}{Style.RESET_ALL}\n"
        )  # Print bottom separator line for the augmentation experiments section
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def load_and_combine_augmented_combined_files(original_files_list, config=None):
    """
    Loads augmented files for combined files evaluation mode and combines them into a single DataFrame.

    :param original_files_list: List of original file paths used to find corresponding augmented files
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Combined augmented DataFrame on success, or None if loading or combination fails
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        augmented_files_list = load_augmented_files_for_combined_evaluation(original_files_list, config=config)  # Load the list of augmented file paths

        if not augmented_files_list:  # If no augmented files were found
            print(
                f"{BackgroundColors.YELLOW}No augmented files found for combined files evaluation mode. Skipping augmentation testing.{Style.RESET_ALL}"
            )  # Print warning about missing augmented files
            return None  # Signal caller to exit early

        combined_augmented_df, augmented_attack_types, augmented_target_col = combine_files_for_combined_evaluation(augmented_files_list, config=config, remove_zero_variance=False)  # Preserve the original fitted schema in augmented testing data.

        if combined_augmented_df is None:  # If augmented file combination failed
            print(
                f"{BackgroundColors.YELLOW}Failed to combine augmented files for combined files evaluation. Skipping augmentation testing.{Style.RESET_ALL}"
            )  # Print warning about combination failure
            return None  # Signal caller to exit early

        return combined_augmented_df  # Return the combined augmented DataFrame for ratio experiments
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def process_combined_files_augmentation_testing(reference_file, original_files_list, combined_files_df, feature_names, ga_selected_features, pca_n_components, rfe_selected_features, base_models, hp_params_map, attack_types_list, results_original, augmentation_ratios, total_steps, feature_analysis_dir, dataset_name, config=None):
    """
    Process combined files evaluation augmented data with ratio-based experiments.

    :param reference_file: Reference file path for feature metadata
    :param original_files_list: List of original file paths
    :param combined_files_df: Combined files evaluation DataFrame
    :param feature_names: List of feature column names
    :param ga_selected_features: GA selected features
    :param pca_n_components: Number of PCA components
    :param rfe_selected_features: RFE selected features
    :param base_models: Dictionary of base models
    :param hp_params_map: Hyperparameters mapping
    :param attack_types_list: List of unique attack type labels
    :param results_original: Results from original data evaluation
    :param augmentation_ratios: List of augmentation ratios to evaluate
    :param total_steps: Total number of evaluation steps for progress display
    :param feature_analysis_dir: Path to Feature_Analysis directory
    :param dataset_name: Name of the dataset being processed
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Processing combined files evaluation augmented data...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message
        combined_dataset_identity = resolve_combined_files_dataset_identity(original_files_list)  # Resolve canonical combined directory identity.
        combined_dataset_reference = combined_dataset_identity.rstrip("/")  # Use directory path text without trailing separator for path APIs.

        generate_augmentation_tsne_visualization(
            combined_dataset_reference, combined_files_df, None, None, "original_only"  # Use directory identity for combined visualization.
        )  # Generate t-SNE visualization for original combined files evaluation data only

        combined_augmented_df = load_and_combine_augmented_combined_files(original_files_list, config=config)  # Load and combine augmented files into a single DataFrame

        if combined_augmented_df is None:  # If loading or combining augmented files failed
            return  # Exit function early as signaled by the loading function

        print_combined_files_augmentation_header(augmentation_ratios)  # Print the section header for ratio-based combined files evaluation augmentation experiments
        send_telegram_message(
            TELEGRAM_BOT,
            build_telegram_pipeline_summary(
                config,
                dataset_name=dataset_name,
                classification_mode="combined_files",
            ) + [f"[COMBINED_FILES] Starting ratio-based augmentation experiments | Dataset: {dataset_name} | ratios: {[f'{int(r * 100)}%' for r in augmentation_ratios]}"]
        )  # Notify Telegram about combined files evaluation augmentation ratio experiments start with full pipeline summary

        all_ratio_results = {}  # Dictionary to store results for each ratio: {ratio: results_dict}

        for ratio_idx, ratio in enumerate(augmentation_ratios, start=1):  # Iterate over each augmentation ratio
            results_ratio = run_combined_files_augmentation_ratio_experiment(
                combined_dataset_reference, combined_files_df, combined_augmented_df, feature_names,  # Use directory identity for ratio evaluation.
                ga_selected_features, pca_n_components, rfe_selected_features, base_models,
                hp_params_map, attack_types_list, ratio, ratio_idx, total_steps, dataset_name, config=config,
            )  # Run evaluation for this ratio and retrieve results or None if sampling failed
            if results_ratio is None:  # If this ratio experiment failed at the sampling stage
                continue  # Skip to the next ratio
            all_ratio_results[ratio] = results_ratio  # Store the results for this ratio

        del combined_augmented_df  # Release combined augmented dataframe to free memory after all ratio experiments
        gc.collect()  # Force garbage collection to reclaim memory from augmented data

        save_combined_files_augmentation_comparison(results_original, all_ratio_results, feature_analysis_dir, config=config)  # Generate comparison report and save results to CSV
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def compute_class_distribution(combined_df: pd.DataFrame, target_col: str) -> List[Tuple[str, int, float]]:
    """
    Compute sorted class distribution for a target column.

    :param combined_df: Combined DataFrame containing the target column.
    :param target_col: Name of the target column to count values for.
    :return: List of tuples (label, count, percentage) sorted descending by count.
    """

    try:
        counts = combined_df[target_col].value_counts(dropna=False)  # Get value counts including NaN as a category
        counts_sorted = counts.sort_values(ascending=False)  # Sort counts descending to show most frequent classes first
        total_samples = int(counts_sorted.sum())  # Compute total number of samples across all classes
        distribution = [(str(label), int(count), (int(count) / total_samples) * 100) for label, count in counts_sorted.items()]  # Build list of (label, count, percentage) tuples
        return distribution  # Return the structured distribution result
    except Exception as e:  # On exception, propagate for outer handlers to log and notify
        raise


def process_combined_files_evaluation(original_files_list, combined_files_df, attack_types_list, dataset_name, config=None):
    """
    Process evaluation for combined files evaluation mode with optional data augmentation.
    
    :param original_files_list: List of original file paths used for combined files evaluation
    :param combined_files_df: Combined files evaluation DataFrame with 'attack_type' column
    :param attack_types_list: List of unique attack type labels
    :param dataset_name: Name of the dataset being processed
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG
        
        verbose_output(
            f"{BackgroundColors.GREEN}Processing combined files evaluation for dataset: {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message
    
        reference_file = original_files_list[0] if original_files_list else "combined_files_combined"  # Get source file for feature metadata.
        combined_dataset_identity = resolve_combined_files_dataset_identity(original_files_list)  # Resolve canonical combined directory identity.
        combined_dataset_reference = combined_dataset_identity.rstrip("/")  # Use directory path text without trailing separator for path APIs.
        
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}"
        )  # Print separator line
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing combined files evaluation dataset: {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}"
        )  # Print dataset header
        distribution = compute_class_distribution(combined_files_df, 'attack_type')  # Compute distribution from combined dataframe
        print(f"{BackgroundColors.GREEN}Attack types:{Style.RESET_ALL}")  # Print attack types header
        for label, count, percentage in distribution:  # Iterate over distribution entries sorted by count
            print(f"- {BackgroundColors.CYAN}{label}{BackgroundColors.GREEN}: {BackgroundColors.CYAN}{count:,}{BackgroundColors.GREEN} ({percentage:.2f}% of total){Style.RESET_ALL}")  # Print each class name, sample count, and percentage
        print(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}\n")  # Print closing separator
        send_telegram_message(
            TELEGRAM_BOT,
            build_telegram_pipeline_summary(
                config,
                dataset_name=dataset_name,
                classification_mode="combined_files",
                attack_types_list=attack_types_list,
                include_attack_types=True,
            ) + [f"Starting combined files evaluation | Dataset: {dataset_name}"]
        )  # Notify Telegram about combined files evaluation start with full pipeline summary

        ga_selected_features, pca_n_components, rfe_selected_features = load_feature_selection_results(
            reference_file, config=config
        )  # Load feature selection results

        methods_cfg = config.get("stacking", {}).get("methods", {})  # Retrieve method toggles from config

        if not methods_cfg.get("feature_selection", True):  # Verify if feature selection is disabled via toggle
            ga_selected_features = None  # Suppress GA features when feature selection is disabled
            pca_n_components = None  # Suppress PCA components when feature selection is disabled
            rfe_selected_features = None  # Suppress RFE features when feature selection is disabled
        
        feature_names = [col for col in combined_files_df.columns if col != 'attack_type']  # Get feature column names
        ga_selected_features, rfe_selected_features = sanitize_and_verify_feature_selections(ga_selected_features, rfe_selected_features, feature_names, config=config)  # Normalize artifacts before grid counting so the denominator matches assembled feature modes
        
        verbose_output(
            f"{BackgroundColors.GREEN}Combined files evaluation dataset features: {BackgroundColors.CYAN}{len(feature_names)} features{Style.RESET_ALL}",
            config=config
        )  # Output feature count
        original_sample_count = int(len(combined_files_df))  # Preserve original row count for augmentation sampling after dataframe release.
        combined_files_df_holder = [combined_files_df]  # Transfer the original combined dataframe so later calls can consume it without caller retention.
        combined_files_df = None  # Release this frame's direct original combined dataframe reference before classifier fitting.
        gc.collect()  # Reclaim the released direct dataframe reference before the first evaluation slice.

        default_models = get_models(config=config)  # Create untouched default/current parameter model objects
        hp_runs = [(False, default_models, {})]  # Default hyperparameters always form the first complete grid
        if methods_cfg.get("hyperparameter_optimization", True):  # Add optimized mode only when the method is enabled
            optimized_models, optimized_params = build_optimized_hyperparameter_models(reference_file, config=config)  # Build only classifiers with valid optimized artifacts
            if optimized_models:  # Add optimized mode only when at least one classifier has verified optimized parameters
                hp_runs.append((True, optimized_models, optimized_params))  # Keep optimized estimators isolated from defaults

        augmentation_enabled = methods_cfg.get("augmentation", True) and config.get("execution", {}).get("test_data_augmentation", False)  # Both existing toggles must enable augmentation modes
        augmented_files_list = load_augmented_files_for_combined_evaluation(original_files_list, config=config) if augmentation_enabled else []  # Resolve augmented file paths without loading their dataframes
        augmentation_file_paths = [path for path in augmented_files_list if path is not None]  # Filter missing augmented-file placeholders before deferred loading
        augmentation_ratios = config.get("stacking", {}).get("augmentation_ratios", [0.25, 0.50, 0.75, 1.00]) if augmentation_file_paths else []  # Generate ratio modes only when augmentation files exist
        feature_mode_names = list_grid_feature_modes(ga_selected_features, pca_n_components, rfe_selected_features, feature_names, config=config)  # Resolve actual feature modes in evaluation order
        evaluation_plan = build_evaluation_plan(hp_runs, [None] + list(augmentation_ratios), feature_mode_names, methods_cfg.get("stacking", True))  # Build the exact default-first, original-first full-grid order
        total_steps = len(evaluation_plan)  # Use the ordered runtime plan as the exact full-grid denominator
        grid_progress = create_grid_progress(total_steps, f"{dataset_name} Combined Grid")  # Share one counter across HP and augmentation modes
        grid_progress["evaluation_plan"] = evaluation_plan  # Reuse the authoritative plan in every evaluation section sharing this progress bar
        all_grid_results = []  # Accumulate every result row for a single consolidated export
        all_comparison_results = []  # Accumulate augmentation comparisons across both HP modes

        try:  # Ensure the shared progress bar is closed after the complete grid
            for hyperparameters_enabled, base_models, hp_params_map in hp_runs:  # Finish the default grid before beginning optimized runs
                hp_label = "Optimized Hyperparameters" if hyperparameters_enabled else "Default Hyperparameters"  # Build active HP label
                send_telegram_message(TELEGRAM_BOT, [f"[COMBINED_FILES] Starting {hp_label} grid | Dataset: {dataset_name}"])  # Announce active HP mode
                if combined_files_df_holder:  # Use the initially combined dataframe for the first original-only slice.
                    original_df_for_run = combined_files_df_holder.pop()  # Consume the initially combined dataframe exactly once.
                else:  # Rebuild the original combined dataframe only for later HP slices that need it.
                    original_df_for_run, _, _ = combine_files_for_combined_evaluation(original_files_list, config=config)  # Recombine from the same ordered input files for this HP slice.
                    if original_df_for_run is None:  # Abort this HP slice when recombination fails.
                        raise RuntimeError("Failed to rebuild original combined files dataframe for later HP slice")  # Raise explicit failure instead of fitting on missing data.
                original_df_for_run_holder = [original_df_for_run]  # Transfer this HP slice dataframe into evaluation without retaining a caller reference.
                del original_df_for_run  # Release direct HP slice dataframe reference before model evaluation.
                gc.collect()  # Reclaim released direct dataframe references before fitting starts.
                results_original = evaluate_on_dataset(
                    combined_dataset_reference, original_df_for_run_holder.pop(), feature_names, ga_selected_features, pca_n_components,  # Use directory identity for original combined evaluation.
                    rfe_selected_features, base_models, data_source_label="Original Combined Files", hyperparams_map=hp_params_map,
                    experiment_id=generate_experiment_id(combined_dataset_reference, "combined_files_original_only"), experiment_mode="original_only", augmentation_ratio=None,  # Build original experiment identity from combined directory.
                    execution_mode_str="combined_files", attack_types_combined=attack_types_list, config=config,
                    hyperparameters_enabled=hyperparameters_enabled, grid_progress=grid_progress,
                    source_files=original_files_list,  # Preserve the ordered original CSV provenance used to build this combined dataset.
                )  # Evaluate the no-augmentation feature/classifier slice
                del original_df_for_run_holder  # Release the empty HP slice transfer holder after evaluation returns.
                gc.collect()  # Reclaim any released original-only dataframe references before augmentation handling.
                original_results_list = list(results_original.values())  # Convert baseline results for annotation and export
                annotate_results_with_combination_flags(original_results_list, methods_cfg.get("feature_selection", True), hyperparameters_enabled, False)  # Mark active grid dimensions
                all_grid_results.extend(original_results_list)  # Preserve baseline rows beside later grid slices

                ratio_results = {}  # Collect this HP mode's ratio results for comparison reporting
                for ratio in augmentation_ratios:  # Evaluate each configured augmentation ratio separately
                    combined_augmented_df, _, _ = combine_files_for_combined_evaluation(augmentation_file_paths, config=config, remove_zero_variance=False)  # Combine augmented testing files without learning a feature-removal state.
                    if combined_augmented_df is None:  # Skip this ratio when augmented recombination fails.
                        print(f"{BackgroundColors.YELLOW}Failed to combine augmented files for combined files evaluation ratio {ratio}. Skipping augmentation ratio for {hp_label}.{Style.RESET_ALL}")  # Report skipped augmentation ratio.
                        continue  # Move to the next configured ratio.
                    df_sampled = sample_augmented_by_ratio(combined_augmented_df, original_sample_count, ratio)  # Sample augmented rows using the preserved original row count.
                    del combined_augmented_df  # Release the full augmented combined dataframe before original data is rebuilt.
                    gc.collect()  # Reclaim augmented source memory before ratio evaluation.
                    if df_sampled is None or df_sampled.empty:  # Skip unusable ratio samples
                        if df_sampled is not None:  # Release an empty sampled dataframe before continuing.
                            del df_sampled  # Release unusable sampled dataframe.
                            gc.collect()  # Reclaim unusable sampled dataframe memory.
                        continue  # Move to the next configured ratio
                    ratio_original_df, _, _ = combine_files_for_combined_evaluation(original_files_list, config=config)  # Rebuild original combined dataframe after augmented source release.
                    if ratio_original_df is None:  # Skip this ratio when original recombination fails.
                        print(f"{BackgroundColors.YELLOW}Failed to rebuild original combined files dataframe for ratio {ratio}. Skipping augmentation ratio for {hp_label}.{Style.RESET_ALL}")  # Report skipped ratio due to missing original data.
                        del df_sampled  # Release sampled augmented rows when original data is unavailable.
                        gc.collect()  # Reclaim sampled augmented rows before continuing.
                        continue  # Move to the next configured ratio.
                    ratio_original_df_holder = [ratio_original_df]  # Transfer ratio original dataframe into evaluation without retaining a caller reference.
                    df_sampled_holder = [df_sampled]  # Transfer sampled augmented dataframe into evaluation without retaining a caller reference.
                    del ratio_original_df, df_sampled  # Release direct ratio dataframe references before model evaluation.
                    gc.collect()  # Reclaim direct ratio dataframe references before fitting starts.
                    results_ratio = evaluate_on_dataset(
                        combined_dataset_reference, ratio_original_df_holder.pop(), feature_names, ga_selected_features, pca_n_components,  # Use directory identity for augmented combined evaluation.
                        rfe_selected_features, base_models, data_source_label=f"Augmented@{int(ratio * 100)}%_CombinedFiles", hyperparams_map=hp_params_map,
                        experiment_id=generate_experiment_id(combined_dataset_reference, "combined_files_original_training_augmented_testing", ratio), experiment_mode="original_training_augmented_testing", augmentation_ratio=ratio,  # Build augmented-testing experiment identity from combined directory.
                        execution_mode_str="combined_files", attack_types_combined=attack_types_list, df_augmented_for_testing=df_sampled_holder.pop(),
                        config=config, hyperparameters_enabled=hyperparameters_enabled, grid_progress=grid_progress,
                        source_files=original_files_list,  # Preserve only original training-source provenance in model identity.
                    )  # Evaluate persisted models on this augmented-only test ratio.
                    del ratio_original_df_holder, df_sampled_holder  # Release empty transfer holders after ratio evaluation returns.
                    gc.collect()  # Reclaim released ratio holder references before result aggregation.
                    ratio_results_list = list(results_ratio.values())  # Convert ratio results for annotation and export
                    annotate_results_with_combination_flags(ratio_results_list, methods_cfg.get("feature_selection", True), hyperparameters_enabled, True)  # Mark active grid dimensions
                    all_grid_results.extend(ratio_results_list)  # Preserve ratio rows in the consolidated export
                    ratio_results[ratio] = results_ratio  # Retain ratio result mapping for comparisons

                if ratio_results:  # Preserve existing augmentation comparison output for this HP mode
                    comparison_results = generate_ratio_comparison_report(results_original, ratio_results, config=config)  # Compare this HP mode's ratios against its matching baseline
                    for comparison_row in comparison_results:  # Annotate comparison rows so both HP modes remain distinguishable
                        comparison_row["hyperparameter_mode"] = hp_label  # Store the active HP mode in the comparison export
                    all_comparison_results.extend(comparison_results)  # Preserve comparisons until the complete grid is ready to save
        finally:  # Close shared grid progress even if a classifier evaluation fails
            grid_progress["progress_bar"].close()  # Close the one full-grid progress bar

        feature_analysis_dir = save_combined_files_results_to_csv(combined_dataset_reference, all_grid_results, config=config)  # Save the complete grid using directory identity.
        if all_comparison_results:  # Save all HP modes together so optimized comparisons cannot overwrite defaults
            save_combined_files_augmentation_comparison(None, None, feature_analysis_dir, config=config, comparison_results=all_comparison_results)  # Preserve existing combined comparison filename and metrics
        
        enable_automl = methods_cfg.get("automl", True)  # Resolve AutoML toggle from stacking methods config
        if enable_automl:  # If AutoML pipeline is enabled
            print(f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}[DEBUG] AutoML pipeline is ENABLED. Running AutoML for combined files evaluation dataset.{Style.RESET_ALL}")  # Log AutoML execution start
            send_telegram_message(TELEGRAM_BOT, [f"Running AutoML pipeline for combined files evaluation dataset: {dataset_name}"])  # Notify Telegram about AutoML pipeline execution
            automl_combined_df, _, _ = combine_files_for_combined_evaluation(original_files_list, config=config)  # Rebuild original combined dataframe only when AutoML is enabled.
            if automl_combined_df is None:  # Skip AutoML when original recombination fails.
                print(f"{BackgroundColors.YELLOW}[DEBUG] AutoML skipped because the original combined files dataframe could not be rebuilt.{Style.RESET_ALL}")  # Log AutoML skip reason.
            else:  # Run AutoML with the rebuilt dataframe.
                run_automl_pipeline(combined_dataset_reference, automl_combined_df, feature_names, data_source_label="Original Combined Files", config=config)  # Run AutoML pipeline using directory identity.
                del automl_combined_df  # Release AutoML combined dataframe after AutoML pipeline returns.
                gc.collect()  # Reclaim AutoML combined dataframe memory before cache cleanup.
        else:  # AutoML pipeline is disabled via method toggle
            print(f"{BackgroundColors.YELLOW}[DEBUG] AutoML pipeline is DISABLED (stacking.methods.automl=false). Skipping AutoML for combined files evaluation. Enable via config or --enable-automl flag.{Style.RESET_ALL}")  # Log AutoML skip reason
        
        try:  # Attempt to remove the cache file now that all combined files evaluation results are safely persisted
            remove_cache_file(combined_dataset_reference, config)  # Remove the directory-based cache once the full combined files evaluation is confirmed complete
        except Exception:  # If cache removal fails for any reason
            pass  # Proceed without failing the run since the CSV output is already written
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def print_file_processing_header(file, config=None):
    """
    Prints formatted header for file processing section.

    :param file: File path being processed
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        progress_idx = config.get("execution", {}).get("progress_index", None)  # Get progress index from config when provided
        progress_total = config.get("execution", {}).get("progress_total", None)  # Get progress total from config when provided
        if progress_idx is not None and progress_total is not None:  # If both progress values are present
            verbose_output(f"{BackgroundColors.GREEN}Printing file processing header for: {BackgroundColors.CYAN}[{progress_idx}/{progress_total}]: {file}{Style.RESET_ALL}", config=config)  # Output the verbose message with index
        else:  # If no progress metadata is available
            verbose_output(f"{BackgroundColors.GREEN}Printing file processing header for: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}", config=config)  # Output the verbose message without index

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}"
        )  # Print separator line
        if progress_idx is not None and progress_total is not None:  # If both progress values are present
            print(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing file [{progress_idx}/{progress_total}]: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}")  # Print file being processed with index
        else:  # If no progress metadata is available
            print(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing file: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}")  # Print file being processed without index
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}\n"
        )  # Print separator line
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def process_single_file_evaluation(file, combined_df, combined_file_for_features, config=None):
    """
    Processes evaluation for a single file including feature loading, model preparation, and evaluation.

    :param file: File path to process
    :param combined_df: Combined dataframe (used if file == "combined")
    :param combined_file_for_features: File to use for feature selection metadata
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Starting single file evaluation for: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        print_file_processing_header(file, config=config)  # Print formatted header

        file_for_features = combined_file_for_features if file == "combined" else file  # Determine which file to use for feature selection metadata
        ga_selected_features, pca_n_components, rfe_selected_features = load_feature_selection_results(
            file_for_features, config=config
        )  # Load feature selection results

        df_original_cleaned, feature_names = load_and_preprocess_dataset(file, combined_df, config=config)  # Load and preprocess the dataset

        if df_original_cleaned is None:  # If loading or preprocessing failed
            return  # Exit function early

        base_models, hp_params_map = prepare_models_with_hyperparameters(file, config=config)  # Prepare base models with hyperparameters

        original_experiment_id = generate_experiment_id(file, "original_only")  # Generate unique experiment ID for the original-only evaluation

        test_data_augmentation = config.get("execution", {}).get("test_data_augmentation", False)  # Get test data augmentation flag from config
        augmentation_ratios = config.get("stacking", {}).get("augmentation_ratios", [0.25, 0.50, 0.75, 1.00])  # Get augmentation ratios from config
        
        if test_data_augmentation:  # If data augmentation testing is enabled
            generate_augmentation_tsne_visualization(
                file, df_original_cleaned, None, None, "original_only"
            )  # Generate t-SNE visualization for original data only

        total_steps = 1 + (len(augmentation_ratios) if test_data_augmentation else 0)  # Calculate total evaluation steps
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}[1/{total_steps}] Evaluating on ORIGINAL data{Style.RESET_ALL}"
        )  # Print progress message with total step count
        results_original = evaluate_on_dataset(
            file, df_original_cleaned, feature_names, ga_selected_features, pca_n_components,
            rfe_selected_features, base_models, data_source_label="Original", hyperparams_map=hp_params_map,
            experiment_id=original_experiment_id, experiment_mode="original_only", augmentation_ratio=None,
            execution_mode_str="separate_files", attack_types_combined=None, config=config,
        )  # Evaluate on original data with experiment traceability metadata and explicit config propagation

        original_results_list = list(results_original.values())  # Convert results dict to list
        save_stacking_results(file, original_results_list, config=config)  # Save original results to CSV

        methods_cfg_local = config.get("stacking", {}).get("methods", {})  # Retrieve method toggles from config
        enable_automl = methods_cfg_local.get("automl", True)  # Resolve AutoML toggle from stacking methods config
        if enable_automl:  # If AutoML pipeline is enabled
            print(f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}[DEBUG] AutoML pipeline is ENABLED. Running AutoML for separate files evaluation dataset: {os.path.basename(file)}{Style.RESET_ALL}")  # Log AutoML execution start
            run_automl_pipeline(file, df_original_cleaned, feature_names, config=config)  # Run AutoML pipeline
        else:  # AutoML pipeline is disabled via method toggle
            print(f"{BackgroundColors.YELLOW}[DEBUG] AutoML pipeline is DISABLED (stacking.methods.automl=false). Skipping AutoML for {os.path.basename(file)}. Enable via config or --enable-automl flag.{Style.RESET_ALL}")  # Log AutoML skip reason

        if test_data_augmentation:  # If data augmentation testing is enabled
            process_augmented_data_evaluation(
                file, df_original_cleaned, feature_names, ga_selected_features, pca_n_components,
                rfe_selected_features, base_models, hp_params_map, results_original, config=config
            )  # Process augmented data evaluation workflow
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def annotate_results_with_combination_flags(results_list, feature_selection_enabled, hyperparameters_enabled, data_augmentation_enabled=False):
    """
    Annotate result entries with combination flags for FS, HP, and DA.

    :param results_list: List of result dictionaries to annotate
    :param feature_selection_enabled: Whether feature selection was enabled
    :param hyperparameters_enabled: Whether hyperparameters were enabled
    :param data_augmentation_enabled: Whether data augmentation was enabled
    :return: None (modifies results_list in place)
    """

    try:
        for row in results_list:  # For each result row in the list
            row["augmentation_ratio"] = resolve_persisted_augmentation_ratio(row.get("experiment_mode", "original_only"), row.get("augmentation_ratio", None))  # Persist explicit baseline or augmented ratio metadata.
            row["feature_selection_enabled"] = resolve_persisted_feature_selection_enabled(row.get("feature_set", ""), feature_selection_enabled)  # Mark row-level feature selection status.
            row["hyperparameters_enabled"] = hyperparameters_enabled  # Mark hyperparameters status
            row["hyperparameter_mode"] = "Optimized Hyperparameters" if hyperparameters_enabled else "Default Hyperparameters"  # Preserve the explicit HP mode in exported rows
            row["data_augmentation_enabled"] = resolve_persisted_data_augmentation_enabled(row.get("experiment_mode", "original_only"), row.get("augmentation_ratio", None), data_augmentation_enabled)  # Mark row-level data augmentation status.
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def list_grid_feature_modes(ga_selected_features: Optional[List[Any]], pca_n_components: Optional[int], rfe_selected_features: Optional[List[Any]], feature_names: List[Any], config: Optional[dict] = None) -> List[str]:
    """
    List the feature modes that the sequential evaluation iterator will generate.

    :param ga_selected_features: Selected feature names produced by GA, if available
    :param pca_n_components: Number of PCA components to use, if available
    :param rfe_selected_features: Selected feature names produced by RFE, if available
    :param feature_names: List of available feature names in the dataset
    :param config: Optional configuration dictionary; uses global CONFIG when None
    :return: Ordered feature mode names that will be generated
    """

    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG

    methods_cfg = config.get("stacking", {}).get("methods", {})  # Read feature-selection method toggle
    if not methods_cfg.get("feature_selection", True):  # Disabled feature selection always produces only the required full-feature baseline
        return ["Full Features"]  # Return only the full-feature baseline

    feature_sets_config = config.get("stacking", {}).get("feature_sets_config", {})  # Read enabled feature strategies
    feature_signatures: set[Tuple[str, Any]] = set()  # Track semantic feature-set identities to mirror assembly behavior
    full_signature = ("features", tuple(sorted(sanitize_feature_name(name) for name in feature_names)))  # Build full-feature identity
    feature_signatures.add(full_signature)  # Register full-feature baseline identity
    feature_modes = ["Full Features"]  # Full Features is always the baseline grid mode

    explicit_features = feature_sets_config.get("explicit_features", []) or []  # Read optional explicit feature strategy
    sanitized_col_map = {sanitize_feature_name(name): name for name in feature_names}  # Build normalized feature lookup
    valid_explicit = list(dict.fromkeys(sanitized_col_map[sanitize_feature_name(name)] for name in explicit_features if sanitize_feature_name(name) in sanitized_col_map))  # Resolve valid explicit features without duplicates
    if valid_explicit:  # Count explicit mode only when at least one requested feature exists
        explicit_signature = ("features", tuple(sorted(sanitize_feature_name(name) for name in valid_explicit)))  # Build explicit feature identity
        if explicit_signature not in feature_signatures:  # Count explicit only when semantically distinct
            feature_signatures.add(explicit_signature)  # Register explicit feature identity
            feature_modes.append("Explicit Features")  # Add the distinct explicit feature mode

    if feature_sets_config.get("use_ga", True) and ga_selected_features:  # Count GA only when enabled and backed by a non-empty artifact
        ga_signature = ("features", tuple(sorted(sanitize_feature_name(name) for name in ga_selected_features)))  # Build GA feature identity
        if ga_signature not in feature_signatures:  # Count GA only when semantically distinct
            feature_signatures.add(ga_signature)  # Register GA feature identity
            feature_modes.append("GA Features")  # Add the distinct GA feature mode

    if feature_sets_config.get("use_pca", True) and pca_n_components:  # Count PCA only when enabled and backed by a valid component count
        pca_signature = ("pca", int(min(pca_n_components, len(feature_names))))  # Build PCA component-space identity
        if pca_signature not in feature_signatures:  # Count PCA only when distinct from existing modes
            feature_signatures.add(pca_signature)  # Register PCA component identity
            feature_modes.append("PCA Components")  # Add the PCA feature mode

    if feature_sets_config.get("use_rfe", True) and rfe_selected_features:  # Count RFE only when enabled and backed by a non-empty artifact
        rfe_signature = ("features", tuple(sorted(sanitize_feature_name(name) for name in rfe_selected_features)))  # Build RFE feature identity
        if rfe_signature not in feature_signatures:  # Count RFE only when semantically distinct
            feature_signatures.add(rfe_signature)  # Register RFE feature identity
            feature_modes.append("RFE Features")  # Add the distinct RFE feature mode

    return feature_modes  # Return the exact ordered feature modes expected from sequential evaluation


def build_evaluation_plan(hp_runs: List[Tuple[bool, dict, dict]], augmentation_modes: List[Optional[float]], feature_mode_names: List[str], stacking_enabled: bool) -> List[Tuple[str, bool, Optional[float], str]]:
    """
    Build the exact ordered combinations represented by one evaluation progress bar.

    :param hp_runs: Ordered runnable hyperparameter modes and their model mappings.
    :param augmentation_modes: Ordered augmentation ratios with None representing original-only data.
    :param feature_mode_names: Ordered feature modes produced by the evaluation iterator.
    :param stacking_enabled: Whether the stacking classifier runs after individual classifiers.
    :return: Ordered tuples of feature set, hyperparameter mode, augmentation ratio, and classifier.
    """

    evaluation_plan = []  # Accumulate combinations in the existing nested-loop execution order
    for hyperparameters_enabled, models_map, _ in hp_runs:  # Preserve default-first runnable hyperparameter order
        classifier_names = list(models_map.keys()) + (["StackingClassifier"] if stacking_enabled else [])  # Preserve enabled individual-classifier order followed by stacking
        for augmentation_ratio in augmentation_modes:  # Preserve original-first configured augmentation order
            for feature_mode_name in feature_mode_names:  # Preserve the sequential feature-mode iterator order
                for classifier_name in classifier_names:  # Preserve individual-classifier order followed by stacking
                    evaluation_plan.append((feature_mode_name, hyperparameters_enabled, augmentation_ratio, classifier_name))  # Store one existing runtime combination identity

    return evaluation_plan  # Return the authoritative ordered progress plan


def create_grid_progress(total_steps, description):
    """
    Create mutable shared progress state for one complete HP/augmentation/feature/classifier grid.

    :param total_steps: Total number of grid steps expected in the progress bar
    :param description: Description displayed beside the tqdm progress bar
    :return: Dictionary containing total steps, current combination counter, and tqdm progress bar
    """

    return {
        "total_steps": total_steps,
        "current_combination": 1,
        "progress_bar": tqdm(total=total_steps, desc=description, file=sys.stdout),
    }  # Return shared denominator, counter, and progress bar


def save_results_with_optional_suffix(file, results_list, suffix, base_filename_key, fallback_filename, use_stacking_subdir=False, config=None):
    """
    Save results to CSV with optional suffix-based filename.

    :param file: File path for determining output directory
    :param results_list: List of result dictionaries to save
    :param suffix: Combination suffix string
    :param base_filename_key: Config key for base filename
    :param fallback_filename: Default filename if config key not found
    :param use_stacking_subdir: Whether to use Stacking subdirectory
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        use_suffix = config.get("stacking", {}).get("use_suffix_filenames", False)  # Verify if suffix filenames are configured
        if use_suffix:  # If suffix-based files desired
            base = config.get("stacking", {}).get(base_filename_key, fallback_filename)  # Get base filename from config
            out_name = base.replace('.csv', f"{suffix}.csv")  # Compose suffixed filename
            dataset_root = resolve_dataset_root_path(str(file))  # Resolve suffixed export root from file or directory input.
            if use_stacking_subdir:  # If Stacking subdirectory should be used
                out_path = dataset_root / "Feature_Analysis" / "Stacking" / out_name  # Build path with Stacking subdir from dataset root.
            else:  # Without Stacking subdirectory
                out_path = dataset_root / "Feature_Analysis" / out_name  # Build path without Stacking subdir from dataset root.
            save_stacking_results(str(out_path), results_list, config=config)  # Save suffixed CSV
        else:  # Default single CSV with extra columns
            save_stacking_results(file, results_list, config=config)  # Save to default location
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def orchestrate_binary_combination(file, ga_sel, pca_n, rfe_sel, base_models, hp_params_map, hyperparameters_enabled, feature_selection_enabled, data_augmentation_enabled, suffix, config=None):
    """
    Orchestrate evaluation for a single separate files evaluation combination of FS/HP/DA flags.

    :param file: Path to the dataset file
    :param ga_sel: GA selected features or None
    :param pca_n: PCA components or None
    :param rfe_sel: RFE selected features or None
    :param base_models: Dictionary of base models
    :param hp_params_map: Hyperparameters mapping
    :param hyperparameters_enabled: Whether hyperparameters were enabled
    :param feature_selection_enabled: Whether feature selection was enabled
    :param data_augmentation_enabled: Whether data augmentation was enabled
    :param suffix: Combination suffix string
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: True if evaluation succeeded, False otherwise
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        df_original, feature_names = load_and_preprocess_dataset(file, None, config=config)  # Load original dataset independently without combining
        if df_original is None:  # If loading failed
            print(f"{BackgroundColors.YELLOW}Skipping file {file} (failed to load).{Style.RESET_ALL}")  # Warn about load failure
            return False  # Signal failure

        target_col = detect_label_column(df_original.columns.tolist())  # Detect target column for class distribution logging
        if target_col:  # If target column was found
            distribution = compute_class_distribution(df_original, target_col)  # Compute class distribution for this file
            print(f"{BackgroundColors.GREEN}[SEPARATE_FILES] Class distribution for {os.path.basename(file)}:{Style.RESET_ALL}")  # Print distribution header
            for label, count, percentage in distribution:  # Iterate over each class, count, and percentage
                print(f"  - {BackgroundColors.CYAN}{label}{BackgroundColors.GREEN}: {BackgroundColors.CYAN}{count:,}{BackgroundColors.GREEN} ({percentage:.2f}% of total){Style.RESET_ALL}")  # Print class name, sample count, and percentage

        send_telegram_message(TELEGRAM_BOT, [f"[SEPARATE_FILES] Starting evaluation | file: {os.path.basename(file)} | FS: {'ON' if feature_selection_enabled else 'OFF'} | HP: {'ON' if hyperparameters_enabled else 'OFF'} | DA: {'ON' if data_augmentation_enabled else 'OFF'}"])  # Notify Telegram about separate files evaluation start

        try:  # Protect the evaluation call
            results = evaluate_on_dataset(
                file, df_original, feature_names, ga_sel, pca_n, rfe_sel, base_models,
                data_source_label="Original",
                hyperparams_map=hp_params_map if hyperparameters_enabled else {},
                experiment_id=generate_experiment_id(file, "original_only"),
                experiment_mode="original_only", augmentation_ratio=None,
                execution_mode_str="separate_files", attack_types_combined=None,
                df_augmented_for_testing=None, config=config, hyperparameters_enabled=hyperparameters_enabled,
            )  # Evaluate original-only separate files evaluation with explicit config propagation
        except Exception as e:  # If evaluation fails
            print(f"{BackgroundColors.RED}Evaluation failed for {file} combo {suffix}: {e}{Style.RESET_ALL}")  # Log error
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception
            return False  # Signal failure

        results_list = list(results.values())  # Convert results dict to list
        annotate_results_with_combination_flags(results_list, feature_selection_enabled, hyperparameters_enabled, False)  # Annotate results with flags
        save_results_with_optional_suffix(
            file, results_list, suffix, "results_filename", "Stacking_Classifiers_Results.csv",
            use_stacking_subdir=True, config=config,
        )  # Save results with optional suffix

        if data_augmentation_enabled:  # If DA requested and available
            try:  # Protect augmentation processing
                process_augmented_data_evaluation(
                    file, df_original, feature_names, ga_sel, pca_n, rfe_sel, base_models,
                    hp_params_map if hyperparameters_enabled else {}, results, config=config,
                )  # Process augmented evaluations
            except Exception as e:  # If augmentation fails
                print(f"{BackgroundColors.YELLOW}Augmentation processing failed for {file} combo {suffix}: {e}{Style.RESET_ALL}")  # Warn
                send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception
                return False  # Signal failure

        del df_original, results  # Release evaluation data to free memory after separate files evaluation combination completes
        gc.collect()  # Force garbage collection to reclaim memory from evaluation data

        try:  # Attempt to remove the cache file now that all results are safely persisted to CSV
            remove_cache_file(file, config)  # Remove the per-file cache once the full evaluation is confirmed complete
        except Exception:  # If cache removal fails for any reason
            pass  # Proceed without failing the run since the CSV output is already written

        return True  # Signal success
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def execute_original_combined_files_evaluation(files_to_process, ga_sel, pca_n, rfe_sel, base_models, hp_params_map, hyperparameters_enabled, feature_selection_enabled, suffix, config):
    """
    Combine files for combined files evaluation, evaluate the original-only dataset, annotate results, and persist them to disk.

    :param files_to_process: List of dataset CSV file paths to combine.
    :param ga_sel: GA-selected feature list or None when feature selection is disabled.
    :param pca_n: Number of PCA components or None when PCA is disabled.
    :param rfe_sel: RFE-selected feature list or None when feature selection is disabled.
    :param base_models: Dictionary mapping model names to model instances.
    :param hp_params_map: Hyperparameter mapping applied when hyperparameters are enabled.
    :param hyperparameters_enabled: Whether hyperparameter optimization results should be applied.
    :param feature_selection_enabled: Whether feature selection is active for this combination.
    :param suffix: Combination suffix string appended to result filenames.
    :param config: Configuration dictionary used throughout evaluation helpers.
    :return: Tuple of ("ok", (combined_df, attack_types, feature_names)) on success, or ("break", None)/("continue", None) on failure.
    """

    try:  # Protect combined files evaluation combine step
        combined_df, attack_types, target_col = combine_files_for_combined_evaluation(files_to_process, config=config)  # Combine files for combined files evaluation
    except Exception as e:  # If combining fails
        print(f"{BackgroundColors.RED}Failed to combine files for combined files evaluation: {e}{Style.RESET_ALL}")  # Error message
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception
        return "break", None  # Signal to break out of combination loop

    if combined_df is None:  # If no combined df produced
        print(f"{BackgroundColors.YELLOW}No combined files evaluation dataset available. Skipping combined files evaluation orchestration.{Style.RESET_ALL}")  # Warn
        return "break", None  # Signal to break out of combination loop

    feature_names = [c for c in combined_df.columns if c != 'attack_type']  # Extract feature names excluding target
    combined_dataset_identity = resolve_combined_files_dataset_identity(files_to_process)  # Resolve canonical combined directory identity.
    combined_dataset_reference = combined_dataset_identity.rstrip("/")  # Use directory path text without trailing separator for path APIs.

    try:  # Protect evaluation step
        results = evaluate_on_dataset(
            combined_dataset_reference, combined_df, feature_names, ga_sel, pca_n, rfe_sel, base_models,  # Use directory identity for legacy combined evaluation.
            data_source_label="Original Combined Files",  # Normalized data source label for log and CSV output
            hyperparams_map=hp_params_map if hyperparameters_enabled else {},
            experiment_id=generate_experiment_id(combined_dataset_reference, "combined_files_original_only"),  # Build legacy experiment identity from combined directory.
            experiment_mode="original_only", augmentation_ratio=None,
            execution_mode_str="combined_files", attack_types_combined=attack_types,
            df_augmented_for_testing=None,
            cache_ref_file=combined_dataset_reference, config=config, hyperparameters_enabled=hyperparameters_enabled,  # Use directory identity for cache resume.
            source_files=files_to_process,  # Preserve ordered original CSV provenance for legacy combined evaluation.
        )  # Evaluate combined files evaluation original dataset with directory cache reference and explicit config propagation
    except Exception as e:  # If evaluation fails
        print(f"{BackgroundColors.RED}Combined files evaluation failed for combo {suffix}: {e}{Style.RESET_ALL}")  # Error
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception
        return "continue", None  # Signal to continue with next combination

    results_list = list(results.values())  # Convert results dict to list
    annotate_results_with_combination_flags(results_list, feature_selection_enabled, hyperparameters_enabled, False)  # Annotate results
    save_results_with_optional_suffix(
        combined_dataset_reference, results_list, suffix, "combined_files_results_filename",  # Use directory identity for legacy combined export.
        "Stacking_Classifiers_CombinedFiles_Results.csv", use_stacking_subdir=False, config=config,
    )  # Save combined files evaluation results with optional suffix
    return "ok", (combined_df, attack_types, feature_names)  # Return success with data payload


def execute_combined_files_augmentation(files_to_process, combined_df, attack_types, feature_names, ga_sel, pca_n, rfe_sel, base_models, hp_params_map, hyperparameters_enabled, feature_selection_enabled, suffix, config):
    """
    Run augmented data experiments for each configured augmentation ratio using the combined files evaluation dataset.

    :param files_to_process: List of original dataset CSV file paths used to locate augmented counterparts.
    :param combined_df: Combined original combined files evaluation DataFrame used as the test set baseline.
    :param attack_types: List of attack type strings for combined files evaluation metadata.
    :param feature_names: List of feature column names shared across all combined files.
    :param ga_sel: GA-selected feature list or None when feature selection is disabled.
    :param pca_n: Number of PCA components or None when PCA is disabled.
    :param rfe_sel: RFE-selected feature list or None when feature selection is disabled.
    :param base_models: Dictionary mapping model names to model instances.
    :param hp_params_map: Hyperparameter mapping applied when hyperparameters are enabled.
    :param hyperparameters_enabled: Whether hyperparameter optimization results should be applied.
    :param feature_selection_enabled: Whether feature selection is active for this combination.
    :param suffix: Combination suffix string appended to result filenames.
    :param config: Configuration dictionary used throughout evaluation helpers.
    :return: "continue" when augmentation orchestration fails, or None on success.
    """

    try:  # Protect augmentation orchestration
        combined_dataset_identity = resolve_combined_files_dataset_identity(files_to_process)  # Resolve canonical combined directory identity.
        combined_dataset_reference = combined_dataset_identity.rstrip("/")  # Use directory path text without trailing separator for path APIs.
        augmented_files_list = load_augmented_files_for_combined_evaluation(files_to_process, config=config)  # Load augmented files per file
        if not augmented_files_list:  # If none found
            print(f"{BackgroundColors.YELLOW}No augmented files found for combined files evaluation combo {suffix}. Skipping augmentation.{Style.RESET_ALL}")  # Warn
        else:  # Have augmented files to process
            combined_aug_df, _, _ = combine_files_for_combined_evaluation(augmented_files_list, config=config, remove_zero_variance=False)  # Combine augmented testing files without learning a feature-removal state.
            if combined_aug_df is None:  # If combine failed
                print(f"{BackgroundColors.YELLOW}Failed to combine augmented files for combined files evaluation combo {suffix}. Skipping.{Style.RESET_ALL}")  # Warn
            else:  # Proceed with ratio experiments
                for ratio in config.get("stacking", {}).get("augmentation_ratios", [0.25, 0.50, 0.75, 1.00]):  # For each augmentation ratio
                    df_sampled = sample_augmented_by_ratio(combined_aug_df, combined_df, ratio)  # Sample augmented data
                    if df_sampled is None:  # If sampling failed
                        print(f"{BackgroundColors.YELLOW}Sampling failed for ratio {ratio} in combo {suffix}. Skipping ratio.{Style.RESET_ALL}")  # Warn
                        continue  # Next ratio
                    try:  # Evaluate one augmented-only testing ratio.
                        res = evaluate_on_dataset(
                            combined_dataset_reference, combined_df, feature_names, ga_sel, pca_n, rfe_sel, base_models,  # Use directory identity for legacy augmented evaluation.
                            data_source_label=f"Augmented@{int(ratio*100)}%_CombinedFiles",
                            hyperparams_map=hp_params_map if hyperparameters_enabled else {},
                            experiment_id=generate_experiment_id(combined_dataset_reference, "combined_files_original_training_augmented_testing", ratio),  # Build legacy augmented-testing experiment identity from combined directory.
                            experiment_mode="original_training_augmented_testing", augmentation_ratio=ratio,
                            execution_mode_str="combined_files", attack_types_combined=attack_types,
                            df_augmented_for_testing=df_sampled,
                            cache_ref_file=combined_dataset_reference, config=config, hyperparameters_enabled=hyperparameters_enabled,  # Use directory identity for cache resume.
                            source_files=list(files_to_process),  # Preserve only original training-source provenance.
                        )  # Evaluate persisted original-trained models on augmented samples.
                    except Exception as e:  # If evaluation failed
                        print(f"{BackgroundColors.YELLOW}Augmented evaluation failed for ratio {ratio} combo {suffix}: {e}{Style.RESET_ALL}")  # Warn
                        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception
                        continue  # Next ratio
                    res_list = list(res.values())  # Convert augmented results to list
                    annotate_results_with_combination_flags(res_list, feature_selection_enabled, hyperparameters_enabled, True)  # Annotate with da enabled
                    save_stacking_results(combined_dataset_reference, res_list, config=config)  # Save augmented results using directory identity
                del combined_aug_df  # Release combined augmented dataframe to free memory after all ratios
                gc.collect()  # Force garbage collection to reclaim memory from augmented data
    except Exception as e:  # Catch augmentation orchestration errors
        print(f"{BackgroundColors.YELLOW}Combined files evaluation augmentation orchestration failed for combo {suffix}: {e}{Style.RESET_ALL}")  # Warn
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception
        return "continue"  # Signal to continue with next combination
    return None  # Signal success


def orchestrate_combined_files_combination(files_to_process, ga_sel, pca_n, rfe_sel, base_models, hp_params_map, hyperparameters_enabled, feature_selection_enabled, data_augmentation_enabled, suffix, config=None):
    """
    Orchestrate evaluation for a single combined files evaluation combination of FS/HP/DA flags.

    :param files_to_process: List of files to process
    :param ga_sel: GA selected features or None
    :param pca_n: PCA components or None
    :param rfe_sel: RFE selected features or None
    :param base_models: Dictionary of base models
    :param hp_params_map: Hyperparameters mapping
    :param hyperparameters_enabled: Whether hyperparameters were enabled
    :param feature_selection_enabled: Whether feature selection was enabled
    :param data_augmentation_enabled: Whether data augmentation was enabled
    :param suffix: Combination suffix string
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: "break" to abort mode loop, "continue" to skip combination, or None on success
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        status, data = execute_original_combined_files_evaluation(files_to_process, ga_sel, pca_n, rfe_sel, base_models, hp_params_map, hyperparameters_enabled, feature_selection_enabled, suffix, config)  # Combine files, evaluate original dataset, and persist results
        if status != "ok":  # If evaluation did not succeed
            return status  # Propagate flow control signal to caller

        if data is None:  # Verify that data payload is present before unpacking
            print(f"{BackgroundColors.RED}Combined files evaluation returned no data for combo {suffix}.{Style.RESET_ALL}")
            return "break"  # Signal to break out of combination loop since we cannot proceed without the combined dataset and metadata

        combined_df, attack_types, feature_names = data  # Unpack evaluation results payload

        if data_augmentation_enabled:  # If augmentation requested
            augmentation_signal = execute_combined_files_augmentation(files_to_process, combined_df, attack_types, feature_names, ga_sel, pca_n, rfe_sel, base_models, hp_params_map, hyperparameters_enabled, feature_selection_enabled, suffix, config)  # Run augmented experiments for all configured ratios
            if augmentation_signal is not None:  # If augmentation returned a flow control signal
                return augmentation_signal  # Propagate signal to caller

        del combined_df, data  # Release combined files evaluation data to free memory
        gc.collect()  # Force garbage collection to reclaim memory from combined files evaluation combination

        try:  # Attempt to remove the cache file now that all results are safely persisted to CSV
            combined_dataset_identity = resolve_combined_files_dataset_identity(files_to_process)  # Resolve canonical combined directory identity.
            remove_cache_file(combined_dataset_identity.rstrip("/"), config)  # Remove the directory-based cache once the full combined files evaluation is confirmed complete
        except Exception:  # If cache removal fails for any reason
            pass  # Proceed without failing the run since the CSV output is already written

        return None  # Signal success (no flow control change needed)
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def orchestrate_all_combinations(input_path, dataset_name=None, config=None):
    """
    Orchestrate the complete feature/HP/augmentation/classifier grid for separate files evaluation.

    :param input_path: Path containing dataset files to process.
    :param dataset_name: Optional dataset name for logging.
    :param config: Configuration dictionary (uses global CONFIG if None).
    :return: None
    """

    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG

    files_to_process = determine_files_to_process(config.get("execution", {}).get("csv_file", None), input_path, config=config)  # Determine files

    methods_cfg = config.get("stacking", {}).get("methods", {})  # Retrieve method toggles from config
    fs_toggle = methods_cfg.get("feature_selection", True)  # Resolve feature selection toggle from config
    hp_toggle = methods_cfg.get("hyperparameter_optimization", True)  # Resolve hyperparameter optimization toggle from config
    da_toggle = methods_cfg.get("augmentation", True)  # Resolve data augmentation toggle from config
    stacking_enabled = methods_cfg.get("stacking", True)  # Resolve stacking classifier toggle
    augmentation_requested = da_toggle and config.get("execution", {}).get("test_data_augmentation", False)  # Both existing augmentation toggles must permit ratio modes

    total_files = len(files_to_process)  # Compute total number of files for progress reporting
    for idx, file in enumerate(files_to_process, start=1):  # Evaluate each file independently
        grid_progress = None  # Initialize shared progress state for safe cleanup
        try:  # Protect individual-file orchestration
            artifacts = locate_and_verify_artifacts(file, config=config)  # Locate feature, HP, and augmentation artifacts
            ga_sel = artifacts.get("ga") if fs_toggle else None  # Use GA artifact only when feature selection is enabled
            pca_n = artifacts.get("pca") if fs_toggle else None  # Use PCA artifact only when feature selection is enabled
            rfe_sel = artifacts.get("rfe") if fs_toggle else None  # Use RFE artifact only when feature selection is enabled

            df_original, feature_names = load_and_preprocess_dataset(file, None, config=config)  # Load the original dataset once for every grid slice
            if df_original is None:  # Skip files that cannot be loaded
                continue  # Move to the next file
            ga_sel, rfe_sel = sanitize_and_verify_feature_selections(ga_sel, rfe_sel, feature_names, config=config)  # Normalize artifacts before grid counting so the denominator matches assembled feature modes

            default_models = get_models(config=config)  # Create an untouched default/current parameter model map
            hp_runs = [(False, default_models, {})]  # Default hyperparameters always form the first complete grid
            if hp_toggle:  # Add optimized mode only when the method is enabled
                optimized_models, optimized_params = build_optimized_hyperparameter_models(file, config=config)  # Build only classifiers with valid optimized artifacts
                if optimized_models:  # Add optimized mode only when at least one classifier has verified optimized parameters
                    hp_runs.append((True, optimized_models, optimized_params))  # Keep optimized models isolated from default model objects

            df_augmented = load_and_validate_augmented_data(file, df_original, config=config) if augmentation_requested and artifacts.get("augmented_file") else None  # Load compatible augmentation data only when enabled
            augmentation_ratios = config.get("stacking", {}).get("augmentation_ratios", [0.25, 0.50, 0.75, 1.00]) if df_augmented is not None else []  # Generate ratio modes only when augmentation is usable
            feature_mode_names = list_grid_feature_modes(ga_sel, pca_n, rfe_sel, feature_names, config=config)  # Resolve actual feature modes in evaluation order
            evaluation_plan = build_evaluation_plan(hp_runs, [None] + list(augmentation_ratios), feature_mode_names, stacking_enabled)  # Build the exact default-first, original-first full-grid order
            total_steps = len(evaluation_plan)  # Use the ordered runtime plan as the exact full-grid denominator
            grid_progress = create_grid_progress(total_steps, f"{os.path.basename(file)} Grid")  # Share one counter across every HP and augmentation mode
            grid_progress["evaluation_plan"] = evaluation_plan  # Reuse the authoritative plan in every evaluation section sharing this progress bar
            all_grid_results = []  # Accumulate every grid result for one consolidated export
            all_comparison_results = []  # Accumulate augmentation comparisons across both HP modes

            print(f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}Orchestrating full grid: file=[{idx}/{total_files}] {file}, combinations={total_steps}{Style.RESET_ALL}")  # Log exact generated grid size

            for hyperparameters_enabled, base_models, hp_params_map in hp_runs:  # Complete default grid before starting the optimized grid
                hp_label = "Optimized Hyperparameters" if hyperparameters_enabled else "Default Hyperparameters"  # Build explicit active HP label
                send_telegram_message(TELEGRAM_BOT, [f"[SEPARATE_FILES] Starting {hp_label} grid | file: {os.path.basename(file)}"])  # Announce active HP grid

                results_original = evaluate_on_dataset(
                    file, df_original, feature_names, ga_sel, pca_n, rfe_sel, base_models,
                    data_source_label="Original", hyperparams_map=hp_params_map,
                    experiment_id=generate_experiment_id(file, "original_only"), experiment_mode="original_only", augmentation_ratio=None,
                    execution_mode_str="separate_files", attack_types_combined=None, config=config,
                    hyperparameters_enabled=hyperparameters_enabled, grid_progress=grid_progress,
                )  # Evaluate every feature/classifier combination without augmentation
                original_list = list(results_original.values())  # Convert baseline results for annotation and export
                annotate_results_with_combination_flags(original_list, fs_toggle, hyperparameters_enabled, False)  # Mark baseline grid dimensions
                all_grid_results.extend(original_list)  # Preserve baseline rows beside optimized and augmented rows

                ratio_results = {}  # Collect ratio results for the existing comparison export
                for ratio in augmentation_ratios:  # Evaluate each configured augmentation ratio as its own grid mode
                    df_sampled = sample_augmented_by_ratio(df_augmented, df_original, ratio)  # Sample the active augmentation ratio
                    if df_sampled is None or df_sampled.empty:  # Skip ratios that cannot produce augmented test data.
                        continue  # Move to the next configured ratio
                    results_ratio = evaluate_on_dataset(
                        file, df_original, feature_names, ga_sel, pca_n, rfe_sel, base_models,
                        data_source_label=f"Augmented@{int(ratio * 100)}%", hyperparams_map=hp_params_map,
                        experiment_id=generate_experiment_id(file, "original_training_augmented_testing", ratio), experiment_mode="original_training_augmented_testing", augmentation_ratio=ratio,
                        execution_mode_str="separate_files", attack_types_combined=None, df_augmented_for_testing=df_sampled,
                        config=config, hyperparameters_enabled=hyperparameters_enabled, grid_progress=grid_progress,
                    )  # Evaluate the same feature/classifier grid for this augmentation mode
                    ratio_list = list(results_ratio.values())  # Convert ratio results for annotation and export
                    annotate_results_with_combination_flags(ratio_list, fs_toggle, hyperparameters_enabled, True)  # Mark active grid dimensions
                    all_grid_results.extend(ratio_list)  # Preserve ratio rows in the consolidated grid export
                    ratio_results[ratio] = results_ratio  # Retain ratio results for comparison reporting
                    del df_sampled  # Release sampled augmentation rows after the ratio evaluation
                    gc.collect()  # Reclaim ratio-specific memory

                if ratio_results:  # Preserve the existing augmentation comparison report behavior
                    comparison_results = generate_ratio_comparison_report(results_original, ratio_results, config=config)  # Compare this HP mode's ratios against its matching baseline
                    for comparison_row in comparison_results:  # Annotate comparison rows so default and optimized metrics remain distinguishable
                        comparison_row["hyperparameter_mode"] = hp_label  # Store the active HP mode in the existing comparison export
                    all_comparison_results.extend(comparison_results)  # Preserve comparisons until the complete grid is ready to save

            save_stacking_results(file, all_grid_results, config=config)  # Save the complete default-first grid once so later modes cannot overwrite earlier rows
            if all_comparison_results:  # Save all HP modes together so optimized comparisons cannot overwrite default comparisons
                save_augmentation_comparison_results(file, all_comparison_results, config=config)  # Preserve the existing comparison filename and metrics
            remove_cache_file(file, config=config)  # Remove resume cache only after the complete grid is safely exported
            if df_augmented is not None:  # Release optional augmented dataset after both HP grids finish
                del df_augmented  # Drop augmented dataframe reference
            del df_original  # Release original dataset after complete file grid
            gc.collect()  # Reclaim dataset memory
        except Exception as e:  # If per-file orchestration fails
            print(f"{BackgroundColors.RED}Orchestration failed for file {file}: {e}{Style.RESET_ALL}")  # Error
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception
            continue  # Continue with next file
        finally:  # Always close the shared grid progress bar
            if grid_progress is not None:  # Verify progress state was created
                grid_progress["progress_bar"].close()  # Close the one full-grid progress bar


def execute_both_mode_pipeline(files_to_process, local_dataset_name, config=None):
    """
    Execute both separate files evaluation and combined files evaluation classification pipelines sequentially.

    :param files_to_process: List of file paths to process
    :param local_dataset_name: Name of the dataset being processed
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}Execution Mode: BOTH (Separate Files + Combined Files){Style.RESET_ALL}",
            config=config
        )  # Output execution mode

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}"
        )  # Print separator line
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}BOTH MODE: Running Separate Files and Combined Files pipelines sequentially{Style.RESET_ALL}"
        )  # Print mode header
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}\n"
        )  # Print closing separator

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}[STEP 1/2] Executing SEPARATE FILES Classification Pipeline{Style.RESET_ALL}\n"
        )  # Print separate files evaluation step header

        for file_idx, file in enumerate(files_to_process, start=1):  # Iterate each file independently to avoid data leakage between files
            print(f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}[DEBUG] [SEPARATE_FILES] [{file_idx}/{len(files_to_process)}] Evaluating file: {os.path.basename(file)}{Style.RESET_ALL}")  # Log file index and name
            send_telegram_message(TELEGRAM_BOT, [f"[SEPARATE_FILES] [{file_idx}/{len(files_to_process)}] Evaluating: {os.path.basename(file)}"])  # Notify Telegram about per-file evaluation start
            orchestrate_all_combinations(file, dataset_name=local_dataset_name, config=config)  # Orchestrate all combinations for separate files evaluation mode

        gc.collect()  # Force garbage collection to reclaim memory from separate files evaluation phase before combined files evaluation loading

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}✓ Separate files evaluation pipeline complete{Style.RESET_ALL}\n"
        )  # Print separate files evaluation completion message

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}[STEP 2/2] Executing COMBINED FILES EVALUATION Classification Pipeline{Style.RESET_ALL}\n"
        )  # Print combined files evaluation step header

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}"
        )  # Print separator line
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}COMBINED FILES EVALUATION MODE{Style.RESET_ALL}"
        )  # Print mode header
        print(
            f"{BackgroundColors.GREEN}Combining {BackgroundColors.CYAN}{len(files_to_process)}{BackgroundColors.GREEN} files into single combined files evaluation dataset{Style.RESET_ALL}"
        )  # Print files count
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}\n"
        )  # Print closing separator

        combined_files_df, attack_types_list, target_col_name = combine_files_for_combined_evaluation(files_to_process, config=config)  # Combine files for combined files evaluation

        if combined_files_df is None:  # If combination failed
            print(
                f"{BackgroundColors.RED}Failed to create combined files evaluation dataset. Skipping combined files evaluation.{Style.RESET_ALL}"
            )  # Print error about combination failure
        else:  # If combination succeeded
            combined_files_df_holder = [combined_files_df]  # Transfer the combined dataframe so the caller can release its direct reference.
            combined_files_df = None  # Release caller-side combined dataframe reference before nested model evaluation.
            gc.collect()  # Reclaim caller-side dataframe references before combined evaluation starts.
            process_combined_files_evaluation(
                files_to_process, combined_files_df_holder.pop(), attack_types_list, local_dataset_name, config=config
            )  # Process combined files evaluation workflow
            del combined_files_df_holder  # Release the empty transfer holder after combined evaluation returns.

            del combined_files_df  # Release combined files evaluation dataframe to free memory after evaluation
            gc.collect()  # Force garbage collection to reclaim memory from combined files evaluation

            print(
                f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}✓ Combined files evaluation pipeline complete{Style.RESET_ALL}\n"
            )  # Print combined files evaluation completion message

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}"
        )  # Print final separator
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}✓ BOTH MODE COMPLETE: Separate Files and Combined Files pipelines finished{Style.RESET_ALL}"
        )  # Print both mode completion message
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}\n"
        )  # Print final separator
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def execute_combined_files_mode_pipeline(files_to_process, local_dataset_name, config=None):
    """
    Execute combined files evaluation classification pipeline only.

    :param files_to_process: List of file paths to process
    :param local_dataset_name: Name of the dataset being processed
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}Execution Mode: COMBINED FILES EVALUATION{Style.RESET_ALL}",
            config=config
        )  # Output execution mode

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}"
        )  # Print separator line
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}COMBINED FILES EVALUATION MODE{Style.RESET_ALL}"
        )  # Print mode header
        print(
            f"{BackgroundColors.GREEN}Combining {BackgroundColors.CYAN}{len(files_to_process)}{BackgroundColors.GREEN} files into single combined files evaluation dataset{Style.RESET_ALL}"
        )  # Print files count
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}\n"
        )  # Print closing separator

        ordered_files_to_process = order_files_by_size_descending(files_to_process, config=config)  # Order combined files by filesystem size before loading
        write_memory_phase_event("after_combined_file_discovery", config=config, dataset_identity=local_dataset_name, source_file_count=len(ordered_files_to_process), source_files=[os.path.basename(str(path)) for path in ordered_files_to_process], event_outcome="ordered_largest_to_smallest")  # Publish ordered combined-file list
        combined_files_df, attack_types_list, target_col_name = combine_files_for_combined_evaluation(ordered_files_to_process, config=config)  # Combine files for combined files evaluation

        if combined_files_df is None:  # If combination failed
            print(
                f"{BackgroundColors.RED}Failed to create combined files evaluation dataset. Skipping combined files evaluation.{Style.RESET_ALL}"
            )  # Print error about combination failure
            return  # Exit function early

        combined_files_df_holder = [combined_files_df]  # Transfer the combined dataframe through a one-item holder so the caller can release its direct reference.
        combined_files_df = None  # Release caller-side combined dataframe reference before entering model evaluation.
        gc.collect()  # Reclaim caller-side dataframe references before nested evaluation starts.
        process_combined_files_evaluation(
            ordered_files_to_process, combined_files_df_holder.pop(), attack_types_list, local_dataset_name, config=config
        )  # Process combined files evaluation workflow with ordered files preserved.
        del combined_files_df_holder  # Release the empty transfer holder after evaluation returns.

        del combined_files_df  # Release the empty combined dataframe placeholder after evaluation
        gc.collect()  # Force garbage collection to reclaim memory from combined files evaluation
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def execute_binary_mode_pipeline(files_to_process, local_dataset_name, config=None):
    """
    Execute separate files evaluation classification pipeline only.

    :param files_to_process: List of file paths to process
    :param local_dataset_name: Name of the dataset being processed
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}Execution Mode: SEPARATE FILES EVALUATION{Style.RESET_ALL}",
            config=config
        )  # Output execution mode

        for file_idx, file in enumerate(files_to_process, start=1):  # Iterate each file independently to avoid data leakage between files
            print(f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}[DEBUG] [SEPARATE_FILES] [{file_idx}/{len(files_to_process)}] Evaluating file: {os.path.basename(file)}{Style.RESET_ALL}")  # Log file index and name
            send_telegram_message(TELEGRAM_BOT, [f"[SEPARATE_FILES] [{file_idx}/{len(files_to_process)}] Evaluating: {os.path.basename(file)}"])  # Notify Telegram about per-file evaluation start
            orchestrate_all_combinations(file, dataset_name=local_dataset_name, config=config)  # Orchestrate all combinations for separate files evaluation mode
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def process_files_in_path(input_path, dataset_name, config=None):
    """
    Processes all files in a given input path including file discovery and dataset combination.

    :param input_path: Directory path containing files to process
    :param dataset_name: Name of the dataset being processed
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        verbose_output(
            f"{BackgroundColors.GREEN}Processing files in path: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        if not verify_filepath_exists(input_path):  # If the input path does not exist
            verbose_output(
                f"{BackgroundColors.YELLOW}Skipping missing path: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}",
                config=config
            )  # Output skip message
            return  # Exit function early
        
        csv_file = config.get("execution", {}).get("csv_file", None)  # Get CSV file override from config
        execution_mode = config.get("execution", {}).get("execution_mode", "both")  # Get execution mode from config (separate_files/combined_files/both, default: both)

        print(f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}[DEBUG] Classification mode: {execution_mode}{Style.RESET_ALL}")  # Log the resolved classification mode
        send_telegram_message(
            TELEGRAM_BOT,
            build_telegram_pipeline_summary(
                config,
                dataset_path=os.path.relpath(input_path),
                dataset_name=dataset_name,
                classification_mode=execution_mode,
            ),
        )  # Notify Telegram about the full pipeline for this path

        write_memory_phase_event("before_combined_file_discovery", config=config, dataset_source=input_path, dataset_identity=dataset_name, event_outcome="starting")  # Publish file discovery start
        files_to_process = determine_files_to_process(csv_file, input_path, config=config)  # Determine which files to process
        write_memory_phase_event("after_combined_file_discovery", config=config, dataset_source=input_path, dataset_identity=dataset_name, source_file_count=len(files_to_process), source_files=[os.path.basename(str(path)) for path in files_to_process], event_outcome="completed")  # Publish file discovery completion

        local_dataset_name = dataset_name or get_dataset_name(input_path)  # Use provided dataset name or infer from path

        if execution_mode == "both":  # If BOTH execution modes are enabled (run separate files first, then combined files)
            print(f"{BackgroundColors.CYAN}[DEBUG] Running separate files and combined files classification pipelines sequentially{Style.RESET_ALL}")  # Log both mode execution
            execute_both_mode_pipeline(files_to_process, local_dataset_name, config=config)  # Run separate files + combined files pipelines sequentially
        elif execution_mode == "combined_files":  # If combined files evaluation execution mode is enabled
            print(f"{BackgroundColors.CYAN}[DEBUG] Running combined files evaluation classification pipeline{Style.RESET_ALL}")  # Log combined files evaluation pipeline start
            execute_combined_files_mode_pipeline(files_to_process, local_dataset_name, config=config)  # Run combined files evaluation pipeline only
        else:  # If separate files evaluation execution mode (default)
            print(f"{BackgroundColors.CYAN}[DEBUG] Running separate files evaluation classification pipeline{Style.RESET_ALL}")  # Log separate files evaluation pipeline start
            execute_binary_mode_pipeline(files_to_process, local_dataset_name, config=config)  # Run separate files evaluation pipeline only
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def process_dataset_paths(dataset_name, paths, config=None):
    """
    Processes all paths for a given dataset.

    :param dataset_name: Name of the dataset
    :param paths: List of paths to process for this dataset
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing dataset: {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}"
        )  # Print dataset name
        send_telegram_message(
            TELEGRAM_BOT,
            [f"Processing dataset: {dataset_name} ({len(paths)} path(s))"] + build_telegram_pipeline_summary(config, dataset_name=dataset_name),
        )  # Notify Telegram about dataset processing start with full pipeline summary

        for input_path in paths:  # For each path in the dataset's paths list
            process_files_in_path(input_path, dataset_name, config=config)  # Process all files in  this path
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_stacking_pipeline(config_path=None, **config_overrides):
    """
    Programmatic entry point for stacking classifier evaluation.
    
    Allows calling this module as a library with configuration overrides:
    
    Example:
        from stacking import run_stacking_pipeline
        
        # Using config file
        run_stacking_pipeline(config_path="custom_config.yaml")
        
        # Using direct parameter overrides
        run_stacking_pipeline(
            execution={\"verbose\": True, \"test_data_augmentation\": False},
            automl={\"enabled\": True, \"n_trials\": 100}
        )
    
    :param config_path: Path to configuration file (None for default config.yaml)
    :param config_overrides: Dictionary overrides for configuration
    :return: None
    """
    
    config = initialize_config(config_path=config_path, cli_args=None)  # Load base config
    
    for key, value in config_overrides.items():  # Iterate over provided overrides
        if isinstance(value, dict) and key in config:  # If override is dict and key exists
            config[key] = deep_merge_dicts(config[key], value)  # Deep merge override
        else:  # Direct override
            config[key] = value  # Set value directly
    
    initialize_logger(config=config)  # Setup logging
    
    main(config=config)  # Execute stacking pipeline


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


def play_sound(config=None):
    """
    Play a sound when the program finishes and skip if the operating system is Windows.

    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None.
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        current_os = platform.system()  # Get the current operating system
        if current_os == "Windows":  # If the current operating system is Windows
            return  # Do nothing
        
        sound_enabled = config.get("sound", {}).get("enabled", True)  # Get sound enabled flag from config
        if not sound_enabled:  # If sound is disabled
            return  # Do nothing
        
        sound_file = config.get("sound", {}).get("file", "./.assets/Sounds/NotificationSound.wav")  # Get sound file from config
        sound_commands = config.get("sound", {}).get("commands", {})  # Get sound commands from config

        if verify_filepath_exists(sound_file):  # If the sound file exists
            if current_os in sound_commands:  # If the platform.system() is in the sound_commands dictionary
                os.system(f"{sound_commands[current_os]} {sound_file}")  # Play the sound
            else:  # If the platform.system() is not in the sound_commands dictionary
                print(
                    f"{BackgroundColors.RED}The {BackgroundColors.CYAN}{current_os}{BackgroundColors.RED} is not in the {BackgroundColors.CYAN}sound_commands dictionary{BackgroundColors.RED}. Please add it!{Style.RESET_ALL}"
                )
        else:  # If the sound file does not exist
            print(
                f"{BackgroundColors.RED}Sound file {BackgroundColors.CYAN}{sound_file}{BackgroundColors.RED} not found. Make sure the file exists.{Style.RESET_ALL}"
            )
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def validate_and_resolve_dataset_path(dataset_path: str, config: dict) -> dict:
    """
    Validate the CLI dataset path and return an overridden datasets dictionary.

    :param dataset_path: Path to a dataset directory or single CSV file from CLI
    :param config: Configuration dictionary for fallback dataset name resolution
    :return: Dictionary mapping dataset name to list of paths
    """

    try:
        resolved = os.path.abspath(dataset_path)  # Resolve to absolute path

        if not verify_filepath_exists(resolved):  # Verify path existence before proceeding
            raise FileNotFoundError(f"Dataset path does not exist: {resolved}")  # Raise descriptive error for missing path

        dataset_name = get_dataset_name(resolved)  # Extract dataset name from resolved path

        if os.path.isfile(resolved):  # Verify if the path points to a single file
            if not resolved.lower().endswith(".csv"):  # Verify file has CSV extension
                raise ValueError(f"Dataset file must be a CSV: {resolved}")  # Raise error for non-CSV files
            print(f"{BackgroundColors.GREEN}[INFO] CLI --dataset-path resolved to single file: {BackgroundColors.CYAN}{resolved}{Style.RESET_ALL}")  # Log resolved file path
            config["execution"]["csv_file"] = resolved  # Set CSV file override for single-file processing
            return {dataset_name: [os.path.dirname(resolved)]}  # Return parent directory containing the file

        elif os.path.isdir(resolved):  # Verify if the path points to a directory
            print(f"{BackgroundColors.GREEN}[INFO] CLI --dataset-path resolved to directory: {BackgroundColors.CYAN}{resolved}{Style.RESET_ALL}")  # Log resolved directory path
            return {dataset_name: [resolved]}  # Return directory path as dataset entry

        else:  # Path exists but is neither file nor directory
            raise ValueError(f"Dataset path is neither a file nor directory: {resolved}")  # Raise error for unsupported path type
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def log_resolved_configuration(config: dict) -> None:
    """
    Log the resolved configuration for dataset path and method toggles.

    :param config: Configuration dictionary with resolved settings
    :return: None
    """

    try:
        dataset_path_cli = config.get("execution", {}).get("dataset_path", None)  # Retrieve CLI dataset path override
        methods_cfg = config.get("stacking", {}).get("methods", {})  # Retrieve method toggles from config

        fs_enabled = methods_cfg.get("feature_selection", True)  # Resolve feature selection toggle state
        hp_enabled = methods_cfg.get("hyperparameter_optimization", True)  # Resolve hyperparameter optimization toggle state
        da_enabled = methods_cfg.get("augmentation", True)  # Resolve data augmentation toggle state
        automl_enabled = methods_cfg.get("automl", True)  # Resolve AutoML toggle state

        if dataset_path_cli is not None:  # Verify if dataset path was overridden via CLI
            print(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}[INFO] Dataset path override (CLI): {BackgroundColors.CYAN}{dataset_path_cli}{Style.RESET_ALL}")  # Log CLI dataset path override
        else:  # Dataset path is from config.yaml or default
            print(f"{BackgroundColors.GREEN}[INFO] Dataset path source: {BackgroundColors.CYAN}config.yaml (default){Style.RESET_ALL}")  # Log config-based dataset path

        print(f"{BackgroundColors.GREEN}[INFO] Method toggle — Feature Selection: {BackgroundColors.CYAN}{fs_enabled}{Style.RESET_ALL}")  # Log feature selection toggle state
        print(f"{BackgroundColors.GREEN}[INFO] Method toggle — Hyperparameter Optimization: {BackgroundColors.CYAN}{hp_enabled}{Style.RESET_ALL}")  # Log hyperparameter optimization toggle state
        print(f"{BackgroundColors.GREEN}[INFO] Method toggle — Data Augmentation: {BackgroundColors.CYAN}{da_enabled}{Style.RESET_ALL}")  # Log data augmentation toggle state
        print(f"{BackgroundColors.GREEN}[INFO] Method toggle — AutoML: {BackgroundColors.CYAN}{automl_enabled}{Style.RESET_ALL}")  # Log AutoML toggle state

        feature_sets_cfg = config.get("stacking", {}).get("feature_sets_config", {})  # Get resolved feature sets configuration
        full_features_flag = feature_sets_cfg.get("use_full", True)  # Resolve Full Features flag from feature sets config
        pca_flag = feature_sets_cfg.get("use_pca", True)  # Resolve PCA Components flag from feature sets config
        rfe_flag = feature_sets_cfg.get("use_rfe", True)  # Resolve RFE Features flag from feature sets config
        ga_flag = feature_sets_cfg.get("use_ga", True)  # Resolve GA Features flag from feature sets config
        explicit_features = feature_sets_cfg.get("explicit_features", [])  # Retrieve explicit features list from feature sets config

        print(f"{BackgroundColors.GREEN}[INFO] Feature set — Full Features: {BackgroundColors.CYAN}{full_features_flag}{Style.RESET_ALL}")  # Log Full Features state
        print(f"{BackgroundColors.GREEN}[INFO] Feature set — PCA Components: {BackgroundColors.CYAN}{pca_flag}{Style.RESET_ALL}")  # Log PCA Components state
        print(f"{BackgroundColors.GREEN}[INFO] Feature set — RFE Features: {BackgroundColors.CYAN}{rfe_flag}{Style.RESET_ALL}")  # Log RFE Features state
        print(f"{BackgroundColors.GREEN}[INFO] Feature set — GA Features: {BackgroundColors.CYAN}{ga_flag}{Style.RESET_ALL}")  # Log GA Features state
        if explicit_features:  # If explicit features list is non-empty
            print(f"{BackgroundColors.GREEN}[INFO] Feature set — Explicit Features: Enabled ({BackgroundColors.CYAN}{len(explicit_features)} features{BackgroundColors.GREEN}){Style.RESET_ALL}")  # Log explicit features enabled with count
        else:  # If explicit features list is empty or not provided
            print(f"{BackgroundColors.GREEN}[INFO] Feature set — Explicit Features: Disabled{Style.RESET_ALL}")  # Log explicit features disabled

        overrides = []  # Initialize list for override source tracking

        if dataset_path_cli is not None:  # Verify if dataset path came from CLI
            overrides.append("dataset_path")  # Track dataset path as CLI override

        if any(k in methods_cfg for k in ("feature_selection", "hyperparameter_optimization", "augmentation", "automl")):  # Verify if any method toggle was explicitly set
            overrides.append("method_toggles")  # Track method toggles as override source

        if overrides:  # Verify if any overrides were detected
            print(f"{BackgroundColors.YELLOW}[INFO] Configuration overrides active for: {BackgroundColors.CYAN}{', '.join(overrides)}{Style.RESET_ALL}")  # Log active override sources
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def build_telegram_pipeline_summary(config: Optional[dict], dataset_path: Optional[str] = None, dataset_name: Optional[str] = None, classification_mode: Optional[str] = None, attack_types_list: Optional[List[Any]] = None, include_attack_types: bool = False) -> List[str]:
    """
    Build a Telegram message bundle describing the pipeline that will run.

    :param config: Configuration dictionary.
    :param dataset_path: Optional dataset path being processed.
    :param dataset_name: Optional dataset name being processed.
    :param classification_mode: Optional classification mode string.
    :param attack_types_list: Optional list of attack types for combined-files evaluation.
    :param include_attack_types: Whether to include attack type details when provided.
    :return: List of Telegram message lines.
    """

    try:
        if config is None:
            config = CONFIG

        execution_cfg = config.get("execution", {})
        stacking_cfg = config.get("stacking", {})
        methods_cfg = stacking_cfg.get("methods", {})
        feature_sets_cfg = stacking_cfg.get("feature_sets_config", {})

        enabled_classifiers = stacking_cfg.get("enabled_classifiers", []) or []
        feature_methods = []
        if feature_sets_cfg.get("use_full", True):
            feature_methods.append("Full")
        if feature_sets_cfg.get("use_pca", True):
            feature_methods.append("PCA")
        if feature_sets_cfg.get("use_rfe", True):
            feature_methods.append("RFE")
        if feature_sets_cfg.get("use_ga", True):
            feature_methods.append("GA")

        explicit_features = feature_sets_cfg.get("explicit_features", []) or []
        if explicit_features:
            feature_methods.append(f"Explicit({len(explicit_features)})")

        feature_selection_enabled = methods_cfg.get("feature_selection", True)
        hyperparameters_enabled = methods_cfg.get("hyperparameter_optimization", True)
        augmentation_enabled = methods_cfg.get("augmentation", True)
        automl_enabled = methods_cfg.get("automl", True)
        stacking_enabled = methods_cfg.get("stacking", True)
        test_data_augmentation = execution_cfg.get("test_data_augmentation", True)
        augmentation_ratios = stacking_cfg.get("augmentation_ratios", [0.25, 0.50, 0.75, 1.00])

        dataset_display = dataset_path if dataset_path else "config.yaml (default)"
        lines = [
            f"Execution mode: {execution_cfg.get('execution_mode', 'both')} | Dataset: {dataset_display}",
            f"Dataset name: {dataset_name}" if dataset_name else None,
            f"Classification mode: {classification_mode}" if classification_mode else None,
            f"Methods: Feature Selection: {'ON' if feature_selection_enabled else 'OFF'}, Hyperparameters: {'ON' if hyperparameters_enabled else 'OFF'}, Data Augmentation: {'ON' if augmentation_enabled else 'OFF'}, AutoML: {'ON' if automl_enabled else 'OFF'}, Stacking: {'ON' if stacking_enabled else 'OFF'}",
            f"Classifiers: {', '.join(enabled_classifiers) if enabled_classifiers else 'None'}",
            f"Feature extraction methods: {', '.join(feature_methods) if feature_methods else 'None'}",
        ]

        if augmentation_enabled and test_data_augmentation:
            lines.append(f"Data augmentation ratios: {[f'{int(r * 100)}%' for r in augmentation_ratios]}")

        if include_attack_types and attack_types_list:
            lines.append(f"Attack types: {len(attack_types_list)} attack types: {', '.join(str(a) for a in attack_types_list)}")

        return [line for line in lines if line]
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def main(config=None):
    """
    Main function.

    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    normal_completion = False  # Track whether the program reached successful explainability finalization.
    try:
        if config is None:  # If no config provided
            config = CONFIG  # Use global CONFIG

        print(
            f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Classifiers Stacking{BackgroundColors.GREEN} program!{Style.RESET_ALL}\n"
        )  # Output the welcome message

        log_resolved_configuration(config=config)  # Log resolved dataset path and method toggle states
        start_memory_watcher(config=config)  # Start the single watcher sidecar before expensive source-file loading
        
        test_data_augmentation = config.get("execution", {}).get("test_data_augmentation", True)  # Get test augmentation flag from config
        augmentation_ratios = config.get("stacking", {}).get("augmentation_ratios", [0.25, 0.50, 0.75, 1.00])  # Get augmentation ratios from config

        if test_data_augmentation:  # If data augmentation testing is enabled
            print(
                f"{BackgroundColors.BOLD}{BackgroundColors.YELLOW}Data Augmentation Testing: {BackgroundColors.CYAN}ENABLED{Style.RESET_ALL}"
            )  # Print augmentation enabled message
            print(
                f"{BackgroundColors.GREEN}Will train on Original and evaluate Original vs Augmented tests at ratios: {BackgroundColors.CYAN}{[f'{int(r*100)}%' for r in augmentation_ratios]}{Style.RESET_ALL}\n"
            )  # Print augmentation ratios to be evaluated

        start_time = datetime.datetime.now()  # Get the start time of the program

        setup_telegram_bot(config=config)  # Setup Telegram bot if configured

        _exec_mode = config.get("execution", {}).get("execution_mode", "both")  # Retrieve execution mode from config
        _dataset_path_cli = config.get("execution", {}).get("dataset_path", None)  # Retrieve CLI dataset path override
        _methods_cfg = config.get("stacking", {}).get("methods", {})  # Retrieve method toggles from config
        _start_lines = [f"Starting Classifiers Stacking at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"]
        _start_lines.extend(
            build_telegram_pipeline_summary(
                config,
                dataset_path=_dataset_path_cli,
                classification_mode=_exec_mode,
            )
        )
        
        send_telegram_message(TELEGRAM_BOT, _start_lines)  # Send detailed start message with full execution configuration

        threads_limit = set_threads_limit_based_on_ram(config=config)  # Adjust config.get("evaluation", {}).get("threads_limit", 2) based on system RAM
        
        dataset_path_override = config.get("execution", {}).get("dataset_path", None)  # Retrieve CLI --dataset-path override

        if dataset_path_override is not None:  # Verify if CLI dataset path was provided
            datasets = validate_and_resolve_dataset_path(dataset_path_override, config=config)  # Validate and resolve CLI dataset path into datasets dict
        else:  # No CLI override, use config.yaml datasets
            datasets = config.get("dataset", {}).get("datasets", {})  # Get datasets from config

        for dataset_name, paths in datasets.items():  # For each dataset in the datasets dictionary
            dataset_name = str(dataset_name).strip()  # Normalize dataset name by removing leading and trailing spaces
            paths = [p.strip() if isinstance(p, str) else p for p in paths] if isinstance(paths, list) else paths  # Normalize paths list by stripping each string entry
            process_dataset_paths(dataset_name, paths, config=config)  # Process all paths for this dataset

        finalize_pending_explainability_jobs(config=config, terminate=False)  # Wait for asynchronous explainability artifacts before final program reporting
        normal_completion = True  # Mark successful explainability finalization before final success reporting.

        finish_time = datetime.datetime.now()  # Get the finish time of the program
        print(
            f"\n{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
        )  # Output the start and finish times
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
        )  # Output the end of the program message

        send_telegram_message(TELEGRAM_BOT, [f"Finished Classifiers Stacking at {finish_time.strftime('%Y-%m-%d %H:%M:%S')} | Execution time: {calculate_execution_time(start_time, finish_time)}"])  # Send Telegram message indicating finish
        finalize_memory_watcher(config=config, phase="normal_completion", event_outcome="completed")  # Publish normal watcher terminal phase
        
        play_sound_enabled = config.get("sound", {}).get("enabled", True)  # Get play sound flag from config
        if play_sound_enabled:  # If play sound is enabled
            atexit.register(play_sound, config=config)  # Register the play_sound function to be called when the program finishes
    except MemoryError as e:  # Handle top-level memory errors with watcher terminal metadata
        write_memory_phase_event("memory_error", config=config, event_outcome=str(e))  # Publish top-level memory error phase
        finalize_memory_watcher(config=config, phase="abnormal_completion", event_outcome=f"memory_error:{e}")  # Publish abnormal watcher terminal phase
        print(str(e))  # Print memory error to terminal logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send memory error via Telegram
        raise  # Preserve original MemoryError behavior
    except Exception as e:
        finalize_memory_watcher(config=config, phase="abnormal_completion", event_outcome=str(e))  # Publish abnormal watcher terminal phase
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise
    finally:  # Always finalize queued explainability work before leaving main
        finalize_memory_watcher(config=config, phase="abnormal_completion", event_outcome="leaving_main_without_normal_completion")  # Publish abnormal terminal phase if no earlier terminal event exists
        finalize_pending_explainability_jobs(config=config, terminate=not normal_completion)  # Finalize or terminate asynchronous explainability work before main exits


if __name__ == "__main__":
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """
    
    try:  # Protect top-level execution to ensure errors are reported and notified
        cli_args = parse_cli_args()  # Parse command-line arguments into namespace
        config = initialize_config(config_path=cli_args.config, cli_args=cli_args)  # Merge configuration from file and CLI
        initialize_logger(config=config)  # Initialize logger and redirect stdout/stderr to logger
        try:  # Run main and handle user interrupts separately
            main(config=config)  # Invoke main business logic for stacking pipeline
        except KeyboardInterrupt:  # Handle user-initiated interrupts with friendly notification
            try:  # Attempt graceful interrupt notification and cleanup
                print("Execution interrupted by user (KeyboardInterrupt)")  # Inform terminal about user interrupt
                send_telegram_message(TELEGRAM_BOT, ["Stacking pipeline interrupted by user (KeyboardInterrupt)"])  # Notify via Telegram about interrupt
            except Exception:  # Ignore failures sending interrupt notification to avoid masking the interrupt
                pass  # No-op on notification failure
            try:  # Best-effort logger flush/close during interrupt handling
                if logger is not None:  # Only flush/close if logger exists
                    logger.flush()  # Flush pending log writes
                    logger.close()  # Close the logger file handle
            except Exception:  # Ignore logger cleanup errors during interrupt handling
                pass  # Continue to re-raise the interrupt
            raise  # Re-raise KeyboardInterrupt to preserve original exit semantics
    except BaseException as e:  # Catch everything (including SystemExit) and report
        try:  # Try to log and notify about the fatal error
            print(f"Fatal error: {e}")  # Print the exception message to terminal for visibility
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback and message via Telegram
        except Exception:  # If notification fails, attempt to print traceback to stderr as fallback
            try:  # Attempt fallback traceback printing for diagnostics
                traceback.print_exc()  # Print full traceback to stderr as a fallback notification
            except Exception:  # Ignore failures of the fallback printing to avoid cascading errors
                pass  # No further fallback available
        try:  # Attempt best-effort logger cleanup after fatal error
            if logger is not None:  # Only flush/close if logger initialized
                logger.flush()  # Flush pending log writes
                logger.close()  # Close the logger file handle
        except Exception:  # Ignore logger cleanup errors to avoid masking the primary failure
            pass  # No-op on cleanup failure
        raise  # Re-raise the original exception to preserve non-zero exit code and behavior
