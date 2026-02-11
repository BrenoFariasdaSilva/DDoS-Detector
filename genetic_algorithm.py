"""
================================================================================
Genetic Algorithm Feature Selection (genetic_algorithm.py)
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-10-07

Purpose:
    Runs a DEAP-based genetic algorithm to perform feature selection for
    classification tasks. The pipeline includes dataset loading and cleaning,
    scaling, GA setup and execution, fitness evaluation (RandomForest by
    default), and result export/analysis (CSV summaries, feature boxplots,
    and optional Telegram notifications).

Highlights:
    - Binary-mask GA using DEAP with configurable population sweep support
    - Fitness returns multi-metrics: accuracy, precision, recall, F1, FPR, FNR
    - Exports consolidated results to Feature_Analysis/ and saves plots
    - Optional Telegram progress notifications and background resource monitor

Usage:
    - Configure the CSV path in `main()` or call the functions programmatically.
    - The script assumes CSV format where the last column is the target and
        only numeric features are used by the GA pipeline.

Outputs:
    - Feature_Analysis/Genetic_Algorithm_Results.csv (consolidated results)
    - Per-dataset feature summaries and boxplots in Feature_Analysis/

Notes & TODOs:
    - Add argparse/CLI for run-time configuration (sample paths, generations,
        population sizes, runs). Currently `main()` contains the defaults.
    - Improve reproducibility (seed propagation), CV strategy, and parallelism.
    - Add more robust categorical handling, imputation, and unit tests.

Dependencies:
    Python >= 3.9 and: pandas, numpy, scikit-learn, deap, tqdm, matplotlib,
    seaborn, colorama. Optional: psutil, python-telegram-bot.
"""

import argparse  # For command-line argument parsing
import atexit  # For playing a sound when the program finishes
import csv  # For writing metrics/features CSVs
import datetime  # For timestamping
import glob  # For file pattern matching
import hashlib  # For generating state identifiers
import json  # For structured JSON output and parsing
import math  # For mathematical operations
import matplotlib.pyplot as plt  # For plotting graphs
import multiprocessing  # For parallel fitness evaluation
import numpy as np  # For numerical operations
import os  # For running a command in the terminal
import pandas as pd  # For data manipulation
import pickle  # For caching preprocessed data
import platform  # For getting the operating system name
import random  # For random number generation
import re  # For sanitizing filenames
import seaborn as sns  # For enhanced plotting
import shutil  # For verifying disk usage
import subprocess  # For running small system commands (sysctl/wmic)
import sys  # For system-specific parameters and functions
import telegram_bot as telegram_module  # For setting Telegram prefix and device info
import threading  # For optional background resource monitor
import time  # For measuring execution time
from colorama import Style  # For coloring the terminal
from deap import algorithms, base, creator, tools  # For the genetic algorithm
from functools import partial  # For creating partial functions
from joblib import dump, load  # For exporting and importing models
from Logger import Logger  # For logging output to both terminal and file
from pathlib import Path  # For handling file paths
from sklearn.ensemble import RandomForestClassifier  # For the machine learning model
from sklearn.metrics import (  # For model evaluation
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split  # For splitting the dataset and cross-validation
from sklearn.preprocessing import StandardScaler  # For feature scaling
from telegram_bot import TelegramBot, send_telegram_message  # For Telegram notifications
from tqdm import tqdm  # For progress bars
from typing import Any, Callable  # For type hints

psutil = (
    __import__("psutil") if __import__("importlib").util.find_spec("psutil") else None
)  # Import psutil if available, otherwise set to None


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
SKIP_TRAIN_IF_MODEL_EXISTS = False  # If True, try loading exported models instead of retraining
RUNS = 5  # Number of runs for Genetic Algorithm analysis
EARLY_STOP_ACC_THRESHOLD = 0.75  # Minimum acceptable accuracy for an individual
EARLY_STOP_FOLDS = 3  # Number of folds to verify before early stopping
N_JOBS = -1  # Number of parallel jobs for GridSearchCV (-1 uses all processors)
CPU_PROCESSES = 1  # Initial number of worker processes; can be updated by monitor
FILES_TO_IGNORE = [""]  # List of files to ignore during processing
GA_GENERATIONS_COMPLETED = 0  # Updated by GA loop to inform monitor when some generations have run
RESOURCE_MONITOR_LAST_FILE = None  # Path of the file currently being processed (monitor uses this)
RESOURCE_MONITOR_UPDATED_FOR_CURRENT_FILE = False  # Whether monitor already applied an update for the current file
RESUME_PROGRESS = True  # When True, attempt to resume progress from saved state files
PROGRESS_STATE_DIR_NAME = "ga_progress"  # Subfolder under Feature_Analysis to store progress files
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL  # Pickle protocol to use when saving state
GA_RESULTS_CSV_COLUMNS = [  # Columns for the results CSV
    "timestamp",
    "tool",
    "run_index",
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
    "elapsed_run_time",
    "hardware",
    "best_features",
    "rfe_ranking",
]

# Telegram Bot Setup:
TELEGRAM_BOT = None  # Global Telegram bot instance (initialized in setup_telegram_bot)

# Logger Setup:
logger = Logger(f"./Logs/{Path(__file__).stem}.log", clean=True)  # Create a Logger instance
sys.stdout = logger  # Redirect stdout to the logger
sys.stderr = logger  # Redirect stderr to the logger

# Fitness Cache:
fitness_cache = {}  # Cache for fitness results to avoid re-evaluating same feature masks

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

# Functions Definition


def verbose_output(true_string="", false_string=""):
    """
    Outputs a message if the VERBOSE constant is set to True.

    :param true_string: The string to be outputted if the VERBOSE constant is set to True.
    :param false_string: The string to be outputted if the VERBOSE constant is set to False.
    :return: None
    """

    if VERBOSE and true_string != "":  # If VERBOSE is True and a true_string was provided
        print(true_string)  # Output the true statement string
    elif false_string != "":  # If a false_string was provided
        print(false_string)  # Output the false statement string


def verify_dot_env_file():
    """
    Verifies if the .env file exists in the current directory.

    :return: True if the .env file exists, False otherwise
    """

    env_path = Path(__file__).parent / ".env"  # Path to the .env file
    if not env_path.exists():  # If the .env file does not exist
        print(f"{BackgroundColors.CYAN}.env{BackgroundColors.YELLOW} file not found at {BackgroundColors.CYAN}{env_path}{BackgroundColors.YELLOW}. Telegram messages may not be sent.{Style.RESET_ALL}")
        return False  # Return False

    return True  # Return True if the .env file exists


def setup_telegram_bot():
    """
    Sets up the Telegram bot for progress messages.

    :return: None
    """
    
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


def load_exported_artifacts(csv_path):
    """Attempt to locate and load latest exported model, scaler and features for csv_path.

    :param csv_path: original dataset path used to name exported artifacts
    :return: (model, scaler, features, params) or None if not found
    """
    
    models_dir = os.path.join(os.path.dirname(csv_path), "Feature_Analysis", "Genetic_Algorithm", "Models")
    if not os.path.isdir(models_dir):
        return None
    base_name = re.sub(r'[^A-Za-z0-9_.-]+', '_', os.path.splitext(os.path.basename(csv_path))[0])
    pattern = os.path.join(models_dir, f"GA-{base_name}-*-model.joblib")
    candidates = glob.glob(pattern)
    if not candidates:
        return None
    latest_model = max(candidates, key=os.path.getmtime)
    scaler_path = latest_model.replace("-model.joblib", "-scaler.joblib")
    features_path = latest_model.replace("-model.joblib", "-features.json")
    params_path = latest_model.replace("-model.joblib", "-params.json")
    if not (os.path.exists(scaler_path) and os.path.exists(features_path)):
        return None
    try:
        model = load(latest_model)
        scaler = load(scaler_path)
        with open(features_path, "r", encoding="utf-8") as fh:
            features = json.load(fh)
        params = None
        if os.path.exists(params_path):
            try:
                with open(params_path, "r", encoding="utf-8") as ph:
                    params = json.load(ph)
            except Exception:
                params = None
        return model, scaler, features, params
    except Exception:
        return None


def get_files_to_process(directory_path, file_extension=".csv"):
    """
    Get all of the specified files in a directory (non-recursive).

    :param directory_path: Path to the directory to search
    :param file_extension: File extension to filter (default: .csv)
    :return: List of files with the specified extension
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Getting all {BackgroundColors.CYAN}{file_extension}{BackgroundColors.GREEN} files in the directory: {BackgroundColors.CYAN}{directory_path}{Style.RESET_ALL}"
    )  # Output the verbose message

    verify_filepath_exists(directory_path)  # Verify if the directory exists

    if not os.path.isdir(directory_path):  # If the path is not a directory
        verbose_output(
            f"{BackgroundColors.RED}The specified path is not a directory: {BackgroundColors.CYAN}{directory_path}{Style.RESET_ALL}"
        )  # Output the verbose message
        return []  # Return an empty list

    files = []  # List to store the files

    for item in os.listdir(directory_path):  # List all items in the directory
        item_path = os.path.join(directory_path, item)  # Get the full path of the item
        filename = os.path.basename(item_path)  # Get the filename

        if any(
            ignore and (ignore == filename or ignore == item_path) for ignore in FILES_TO_IGNORE
        ):  # If the file is in the FILES_TO_IGNORE list
            verbose_output(
                f"{BackgroundColors.YELLOW}Ignoring file {BackgroundColors.CYAN}{filename}{BackgroundColors.YELLOW} listed in FILES_TO_IGNORE{Style.RESET_ALL}"
            )
            continue  # Skip this file

        if os.path.isfile(item_path) and item.lower().endswith(
            file_extension
        ):  # If the item is a file and has the specified extension
            files.append(item_path)  # Add the file to the list

    return sorted(files)  # Return sorted list for consistency


def get_dataset_name(input_path):
    """
    Extract the dataset name from CSVs path.

    :param input_path: Path to the CSVs files
    :return: Dataset name
    """

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


def get_logical_cpu_count():
    """
    Get logical CPU count, preferring psutil when available.

    :return: integer logical CPU count (>=1)
    """

    try:  # Try to obtain CPU count via psutil (preferred) or os.cpu_count (fallback)
        return (
            int(psutil.cpu_count(logical=True)) if psutil and psutil.cpu_count() else int(os.cpu_count() or 1)
        )  # Return logical CPU count
    except Exception:  # On any exception while querying CPU count
        return max(1, int(os.cpu_count() or 1))  # Fallback to at least 1 logical CPU


def compute_reserved_cpus(total_cpus, reserve_cpu_frac):
    """
    Compute how many CPUs to reserve for system and main process.

    :param total_cpus: total logical CPUs
    :param reserve_cpu_frac: fraction to reserve
    :return: reserved CPU count (>=1)
    """

    try:  # Try to compute reserved CPUs from fraction
        return max(1, int(total_cpus * float(reserve_cpu_frac)))  # Return reserved CPUs
    except Exception:  # If computation fails for any reason
        return 1  # Default to reserving 1 CPU


def compute_cpu_bound(total_cpus, reserved, min_procs):
    """
    Compute CPU-based upper bound for worker processes.

    :param total_cpus: total logical CPUs
    :param reserved: CPUs reserved for system/main
    :param min_procs: minimum allowed worker processes
    :return: cpu_bound (>= min_procs)
    """

    return max(min_procs, total_cpus - reserved)  # Compute CPU-based bound


def compute_memory_bound(reserve_mem_frac):
    """
    Compute memory-based upper bound for worker processes using psutil.

    :param reserve_mem_frac: fraction of memory to keep free
    :return: integer mem_bound or None if not computable
    """

    if not psutil:  # If psutil is not available
        return None  # Memory bound cannot be computed
    try:  # Try to compute memory-based bound
        vm = psutil.virtual_memory()  # Get virtual memory statistics
        avail = int(vm.available)  # Available memory in bytes
        my_rss = max(1, int(psutil.Process().memory_info().rss))  # Current process RSS in bytes (per-worker estimate)
        usable = int(avail * (1.0 - float(reserve_mem_frac)))  # Memory usable after reserving fraction
        return max(1, usable // my_rss)  # Return memory-based maximum number of workers
    except Exception:  # On any error while obtaining memory info
        return None  # Indicate memory bound not available


def compute_optimal_processes(reserve_cpu_frac=0.15, reserve_mem_frac=0.15, min_procs=1, max_procs=None):
    """
    Compute a conservative number of worker processes based on current CPU and
    memory availability.

    Returns an integer number of workers >= min_procs. Uses `psutil` when
    available; falls back to CPU-count heuristics otherwise.
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Computing optimal number of worker processes...{Style.RESET_ALL}"
    )  # Output the verbose message

    total_cpus = get_logical_cpu_count()  # Get logical CPU count (psutil/os fallback)

    reserved = compute_reserved_cpus(total_cpus, reserve_cpu_frac)  # Compute reserved CPUs from fraction
    cpu_bound = compute_cpu_bound(total_cpus, reserved, min_procs)  # Compute CPU-based bound for workers

    mem_bound = compute_memory_bound(reserve_mem_frac)  # Compute memory-based bound (or None)
    if mem_bound is None:  # If memory-based bound not available
        mem_bound = cpu_bound  # Fallback to CPU-based bound

    candidate = min(cpu_bound, mem_bound)  # Start with the smaller of CPU and memory bounds
    if max_procs is not None:  # If an explicit maximum is provided
        try:  # Try to apply the explicit maximum
            candidate = min(candidate, int(max_procs))  # Respect user-provided max_procs
        except Exception:  # Ignore invalid max_procs values
            pass  # Do nothing

    candidate = max(min_procs, int(candidate))  # Ensure candidate is not below min_procs

    try:  # Try to cap candidate by os.cpu_count for safety
        candidate = min(candidate, int(os.cpu_count() or candidate))  # Cap to logical CPUs reported by OS
    except Exception:  # If os.cpu_count fails for any reason
        pass  # Keep current candidate value

    try:  # Try to output verbose message and a visible print for auditing
        print(
            f"{BackgroundColors.GREEN}Optimal worker suggestion: {candidate} (total_cpus={total_cpus}, reserved={reserved}, cpu_bound={cpu_bound}, mem_bound={mem_bound}){Style.RESET_ALL}"
        )  # Verbose output
        print(
            f"{BackgroundColors.GREEN}[Resource Monitor] Using {BackgroundColors.CYAN}{candidate}{BackgroundColors.GREEN} worker(s), cpu={BackgroundColors.CYAN}{total_cpus}{BackgroundColors.GREEN}, reserved={BackgroundColors.CYAN}{reserved}{BackgroundColors.GREEN}, cpu_bound={BackgroundColors.CYAN}{cpu_bound}{BackgroundColors.GREEN}, mem_bound={BackgroundColors.CYAN}{mem_bound}{Style.RESET_ALL}"
        )  # Visible runtime message
    except Exception:  # Do not fail on any print errors
        pass  # Silently ignore printing failures

    return candidate  # Return final computed candidate number of worker processes


def start_resource_monitor(
    interval_seconds=30,
    reserve_cpu_frac=0.15,
    reserve_mem_frac=0.15,
    min_procs=1,
    max_procs=None,
    min_gens_before_update=10,
    daemon=True,
):
    """
    Spawn a daemon thread that updates global `CPU_PROCESSES` periodically.

    The thread computes a safe worker count using `compute_optimal_processes`
    and assigns it to the module-level `CPU_PROCESSES`. The thread swallows
    exceptions and retries after `interval_seconds`.
    """

    def monitor():  # Monitoring thread function
        global CPU_PROCESSES, RESOURCE_MONITOR_UPDATED_FOR_CURRENT_FILE, RESOURCE_MONITOR_LAST_FILE
        while True:  # Infinite loop
            try:  # If we've already updated for the current file, wait for a new file
                if RESOURCE_MONITOR_UPDATED_FOR_CURRENT_FILE:  # Only one update per file
                    time.sleep(1)  # Sleep briefly until new file arrives
                    continue  # Skip computing suggestions until new file
                if GA_GENERATIONS_COMPLETED < int(min_gens_before_update):  # If not enough generations yet
                    time.sleep(1)  # Sleep briefly and re-verify
                    continue  # Skip computing suggestions until threshold met
            except Exception:  # If GA_GENERATIONS_COMPLETED or flags are missing or invalid
                pass  # Proceed to compute (best-effort)

            try:  # Try to compute a suggested worker count
                suggested = compute_optimal_processes(
                    reserve_cpu_frac=reserve_cpu_frac,
                    reserve_mem_frac=reserve_mem_frac,
                    min_procs=min_procs,
                    max_procs=max_procs,
                )  # Compute candidate
                if suggested and suggested != CPU_PROCESSES:  # If suggestion differs from current
                    CPU_PROCESSES = suggested  # Update module-level worker count
                RESOURCE_MONITOR_UPDATED_FOR_CURRENT_FILE = True  # Mark that we've updated for this file
            except Exception:  # On any computation or assignment error
                pass  # Ignore and retry on next loop
            try:  # Sleep for configured interval before next verify
                time.sleep(max(1, int(interval_seconds)))  # Sleep at least 1 second
            except Exception:  # If sleep is interrupted or invalid
                time.sleep(5)  # Fallback sleep

    t = threading.Thread(target=monitor, daemon=daemon, name="ga-resource-monitor")  # Create the monitoring thread
    t.start()  # Start the thread

    return t  # Return the Thread object


def start_resource_monitor_safe(*args, **kwargs):
    """
    Safe wrapper to start the resource monitor: swallow any exceptions so
    callers (e.g., `main`) don't need to handle psutil or threading issues.

    Usage: `start_resource_monitor_safe()` (calls `start_resource_monitor` with defaults).
    Returns the Thread object when started, or None on failure.
    """

    try:  # Try to start the resource monitor
        return start_resource_monitor(*args, **kwargs)  # Start the resource monitor
    except Exception:  # If any exception occurs
        return None  # Return None


def signal_new_file(file_path):
    """
    Notify the resource monitor that processing of a new file has started.

    This resets the per-file monitor flag so the monitor will perform one
    sizing update (after `min_gens_before_update` generations) for the
    newly-started file. It also resets `GA_GENERATIONS_COMPLETED` to 0 so
    the monitor waits again for the configured number of generations.

    :param file_path: path of the file that will be processed
    :return: None
    """

    global RESOURCE_MONITOR_LAST_FILE, RESOURCE_MONITOR_UPDATED_FOR_CURRENT_FILE, GA_GENERATIONS_COMPLETED  # Use global variables
    try:  # Try to signal the new file
        RESOURCE_MONITOR_LAST_FILE = file_path  # Update the last file being processed
        RESOURCE_MONITOR_UPDATED_FOR_CURRENT_FILE = False  # Reset the per-file update flag
        GA_GENERATIONS_COMPLETED = 0  # Reset generations completed for the new file
    except Exception:  # Ignore any errors during signaling
        pass  # Do nothing


def update_progress_bar(
    progress_bar,
    dataset_name,
    csv_path,
    pop_size=None,
    max_pop=None,
    gen=None,
    n_generations=None,
    run=None,
    runs=None,
    progress_state=None,
):
    """
    Update a tqdm `progress_bar` description and postfix consistently.

    :param progress_bar: tqdm progress bar instance (or None)
    :param dataset_name: Name of the dataset
    :param csv_path: Path to the CSV file
    :param pop_size: Current population size (optional)
    :param max_pop: Maximum population size (optional)
    :param n_generations: Number of generations (optional)
    :param run: Current run index (1-based) (optional)
    :param runs: Total runs (optional)
    :param progress_state: Optional dict with keys "current_it" and "total_it" to show iterations
    :return: None
    """

    if progress_bar is None:  # If no progress bar is provided
        return  # Do nothing
    try:  # Try to update the progress bar
        # Build run info as part of description (not postfix)
        run_str = (
            f"{BackgroundColors.GREEN}Run {BackgroundColors.CYAN}{run}{BackgroundColors.GREEN}/{BackgroundColors.CYAN}{runs}{BackgroundColors.GREEN}"
            if run is not None and runs is not None
            else None
        )

        csv_basename = os.path.basename(csv_path)  # Get the CSV filename
        parent_dir = os.path.basename(os.path.dirname(csv_path))  # Get parent directory name
        if (
            parent_dir.lower() != (dataset_name or "").lower()
        ):  # If parent directory differs from dataset_name (case-insensitive)
            csv_filename = f"{BackgroundColors.CYAN}{parent_dir}{BackgroundColors.GREEN}/{BackgroundColors.CYAN}{csv_basename}"  # Include parent directory
        else:  # If parent directory is same as dataset_name
            csv_filename = csv_basename  # Use only basename

        if run_str:  # If run string is provided
            base = f"{BackgroundColors.CYAN}{dataset_name}{BackgroundColors.GREEN} Dataset - {BackgroundColors.CYAN}{csv_filename}{BackgroundColors.GREEN}: {run_str}{Style.RESET_ALL}"  # Base description with run info
        else:  # If no run string
            base = f"{BackgroundColors.CYAN}{dataset_name}{BackgroundColors.GREEN} Dataset - {BackgroundColors.CYAN}{csv_filename}{Style.RESET_ALL}"  # Base description without run info

        details = []  # List to hold detail strings (pop, gen)
        if pop_size is not None:  # If population size is provided
            if max_pop is not None:  # If maximum population size is also provided
                details.append(
                    f"{BackgroundColors.GREEN}Pop {BackgroundColors.CYAN}{pop_size}{BackgroundColors.GREEN}/{BackgroundColors.CYAN}{max_pop}"
                )  # Show current/max population
            else:  # If only current population size is provided
                details.append(
                    f"{BackgroundColors.GREEN}Pop {BackgroundColors.CYAN}{pop_size}"
                )  # Show current population only

        if gen is not None and n_generations is not None:  # If generation and total generations are provided
            details.append(
                f"{BackgroundColors.GREEN}Gen {BackgroundColors.CYAN}{gen}{BackgroundColors.GREEN}/{BackgroundColors.CYAN}{n_generations}"
            )  # Show current/total generations
        elif gen is not None:  # If only generation is provided
            details.append(f"{BackgroundColors.GREEN}Gen {BackgroundColors.CYAN}{gen}")  # Show current generation only
        elif n_generations is not None:  # If only total generations is provided
            details.append(
                f"{BackgroundColors.GREEN}Gen {BackgroundColors.CYAN}{n_generations}"
            )  # Show total generations only
        if details:  # If there are any details to show
            detail_str = ", ".join(details)  # Join details with commas
            desc = f"{base}{BackgroundColors.GREEN} - {detail_str}{Style.RESET_ALL}"
        else:  # If no details
            desc = base  # Just use the base description

        if progress_state and isinstance(progress_state, dict):  # If progress_state dict is provided
            try:  # Try to extract iteration info
                current_it = int(progress_state.get("current_it", 0))  # Current iteration
                total_it = int(progress_state.get("total_it", 0))  # Total iterations
                desc = f"{desc} [{BackgroundColors.CYAN}{current_it}{BackgroundColors.GREEN}/{BackgroundColors.CYAN}{total_it}{BackgroundColors.GREEN} iterations]{Style.RESET_ALL}"  # Append iteration info
            except Exception:  # Silently ignore iteration info extraction failures
                pass  # Do nothing

        progress_bar.set_description(desc)  # Update the progress bar description
        progress_bar.refresh()  # Refresh the progress bar display
    except Exception:  # Silently ignore progress bar update failures
        pass  # Do nothing


def normalize_feature_name(name):
    """
    Normalize feature name by stripping whitespace, replacing double spaces with single spaces, and lowercasing.

    :param name: The feature name to normalize.
    :return: Normalized feature name
    """

    return name.strip().replace("  ", " ").lower()  # Strip whitespace, replace double spaces, and lowercase


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
    df.columns = [
        normalize_feature_name(col) for col in df.columns
    ]  # Normalize feature names by stripping, replacing spaces, and lowercasing

    if df.shape[1] < 2:  # If there are less than 2 columns
        print(f"{BackgroundColors.RED}CSV must have at least 1 feature and 1 target.{Style.RESET_ALL}")
        return None  # Return None

    return df  # Return the loaded DataFrame


def print_loaded_artifacts_info(csv_path, features, params):
    """
    Display information about loaded model artifacts to the console.

    :param csv_path: Path to the CSV dataset file
    :param features: List of selected feature names
    :param params: Dictionary of model parameters (can be None)
    :return: None
    """

    dataset_stem = Path(csv_path).stem  # Extract dataset name from path
    verbose_output(f"{BackgroundColors.GREEN}Loaded exported model and scaler for {BackgroundColors.CYAN}{dataset_stem}{Style.RESET_ALL}")  # Notify successful load
    
    feature_count = len(features)  # Count number of features
    verbose_output(f"{BackgroundColors.GREEN}Selected features ({feature_count}): {BackgroundColors.CYAN}{features}{Style.RESET_ALL}")  # Display feature list
    
    if params is not None:  # Verify if parameters were loaded
        params_json = json.dumps(params, default=str)  # Serialize parameters to JSON string
        verbose_output(f"{BackgroundColors.GREEN}Model parameters: {BackgroundColors.CYAN}{params_json}{Style.RESET_ALL}")  # Display model configuration


def prepare_test_data_for_loaded_model(csv_path, features):
    """
    Execute the full dataset pipeline and select features for evaluation.

    :param csv_path: Path to the CSV dataset file
    :param features: List of feature names to select from the dataset
    :return: Tuple of (X_test_selected, y_test) or None on any failure
    """

    df = load_dataset(csv_path)  # Load raw dataset from CSV
    if df is None:  # Verify if loading failed
        return None  # Exit early on load failure

    cleaned_df = preprocess_dataframe(df)  # Clean and preprocess the dataframe
    if cleaned_df is None or cleaned_df.empty:  # Verify if preprocessing failed or resulted in empty data
        return None  # Exit early on preprocessing failure

    split_data = split_dataset(cleaned_df, csv_path)  # Split into train/test sets
    if split_data is None or split_data[0] is None:  # Verify if splitting failed
        return None  # Exit early on split failure

    X_train, X_test, y_train, y_test, feature_names = split_data  # Unpack split results
    
    sel_indices = [i for i, f in enumerate(feature_names) if f in features]  # Map loaded feature names to column indices
    if not sel_indices:  # Verify if no matching features were found
        return None  # Exit early if feature mapping failed

    X_test_sel = X_test[:, sel_indices]  # Select only the relevant feature columns from test data
    
    return (X_test_sel, y_test)  # Return selected test data and labels


def evaluate_and_display_loaded_model(model, X_test_sel, y_test):
    """
    Evaluate a loaded model on test data and display formatted metrics.

    :param model: The loaded machine learning model
    :param X_test_sel: Test features (with selected columns only)
    :param y_test: Test labels
    :return: None
    """

    try:  # Wrap evaluation in exception handler
        eval_m, testing_time = evaluate_final_on_test(model, X_test_sel, y_test)  # Compute test metrics
        
        if eval_m and eval_m[0] is not None:  # Verify if evaluation returned valid metrics
            print(f"\n{BackgroundColors.GREEN}Test Metrics (loaded model):{Style.RESET_ALL}")  # Print metrics header
            print(f"   {BackgroundColors.GREEN}Accuracy:  {BackgroundColors.CYAN}{truncate_value(eval_m[0])}{Style.RESET_ALL}")  # Display accuracy
            print(f"   {BackgroundColors.GREEN}Precision: {BackgroundColors.CYAN}{truncate_value(eval_m[1])}{Style.RESET_ALL}")  # Display precision
            print(f"   {BackgroundColors.GREEN}Recall:    {BackgroundColors.CYAN}{truncate_value(eval_m[2])}{Style.RESET_ALL}")  # Display recall
            print(f"   {BackgroundColors.GREEN}F1-Score:  {BackgroundColors.CYAN}{truncate_value(eval_m[3])}{Style.RESET_ALL}")  # Display F1 score
            print(f"   {BackgroundColors.GREEN}FPR:       {BackgroundColors.CYAN}{truncate_value(eval_m[4])}{Style.RESET_ALL}")  # Display false positive rate
            print(f"   {BackgroundColors.GREEN}FNR:       {BackgroundColors.CYAN}{truncate_value(eval_m[5])}{Style.RESET_ALL}")  # Display false negative rate
    
    except Exception as e:  # Catch any evaluation errors
        print(f"{BackgroundColors.YELLOW}Could not evaluate loaded model: {e}{Style.RESET_ALL}")  # Display error message


def handle_skip_train_if_model_exists(csv_path):
    """
    Search for existing exported model artifacts and handle loading and evaluation if they exist.

    :param csv_path: Path to the CSV dataset file
    :return: True if artifacts loaded and caller should return, False otherwise
    """

    loaded = load_exported_artifacts(csv_path)  # Attempt to load exported model artifacts
    if loaded is None:  # Verify if artifacts were not found or loading failed
        return False  # Signal that normal GA training should proceed

    model, scaler, features, params = loaded  # Unpack loaded artifacts
    
    print_loaded_artifacts_info(csv_path, features, params)  # Display loaded model information
    
    test_data = prepare_test_data_for_loaded_model(csv_path, features)  # Execute dataset pipeline and select features
    if test_data is not None:  # Verify if dataset preparation succeeded
        X_test_sel, y_test = test_data  # Unpack test data
        evaluate_and_display_loaded_model(model, X_test_sel, y_test)  # Evaluate and display metrics
    
    return True  # Signal that artifacts were loaded and caller should return early


def sanitize_feature_names(columns):
    r"""
    Sanitize column names by removing special JSON characters that LightGBM doesn't support.
    Replaces: { } [ ] : , " \ with underscores.

    :param columns: pandas Index or list of column names
    :return: list of sanitized column names
    """
    
    sanitized = []  # List to store sanitized column names
    
    for col in columns:  # Iterate over each column name
        clean_col = re.sub(r"[{}\[\]:,\"\\]", "_", str(col))  # Replace special characters with underscores
        clean_col = re.sub(r"_+", "_", clean_col)  # Replace multiple underscores with a single underscore
        clean_col = clean_col.strip("_")  # Remove leading/trailing underscores
        sanitized.append(clean_col)  # Add sanitized column name to the list
        
    return sanitized  # Return the list of sanitized column names


def preprocess_dataframe(df, remove_zero_variance=True):
    """
    Preprocess a DataFrame by removing rows with NaN or infinite values and
    dropping zero-variance numeric features.

    :param df: pandas DataFrame to preprocess
    :param remove_zero_variance: whether to drop numeric columns with zero variance
    :return: cleaned DataFrame
    """

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


def cache_preprocessed_data(result, cache_file, csv_path):
    """
    Cache the preprocessed data to a pickle file, verifying disk space first.
    Also, compare and display size reduction compared to the original CSV.

    :param result: The tuple to cache (X_train_scaled, X_test_scaled, y_train_np, y_test_np, X.columns)
    :param cache_file: Path to the cache file.
    :param csv_path: Path to the original CSV file for size comparison.
    :return: None
    """

    X_train_scaled, X_test_scaled, y_train_np, y_test_np, X_columns = result  # Unpack the result tuple
    estimated_size = (
        X_train_scaled.nbytes
        + X_test_scaled.nbytes
        + y_train_np.nbytes
        + y_test_np.nbytes
        + len(pickle.dumps(X_columns))
    )  # Estimate the size of the data to cache
    cache_dir = os.path.dirname(cache_file)  # Get the directory of the cache file
    total, used, free = shutil.disk_usage(cache_dir)  # Get disk usage information
    if free < estimated_size * 1.1:  # 10% margin
        print(
            f"{BackgroundColors.YELLOW}Warning: Insufficient disk space for caching ({estimated_size / (1024**3):.2f} GB needed, {free / (1024**3):.2f} GB free). Skipping cache save.{Style.RESET_ALL}"
        )  # Output warning message
        return  # Return without saving
    else:  # If there is enough space
        with open(cache_file, "wb") as f:  # Open cache file for writing
            pickle.dump(result, f)  # Dump the result to cache file
        verbose_output(
            f"{BackgroundColors.GREEN}Saved preprocessed data to cache {cache_file}.{Style.RESET_ALL}"
        )  # Output the verbose message

        pickle_size = os.path.getsize(cache_file)  # Get the size of the pickle file
        csv_size = os.path.getsize(csv_path)  # Get the size of the original CSV file
        if csv_size > 0:  # If CSV size is available
            reduction = (csv_size - pickle_size) / csv_size * 100  # Calculate reduction percentage
            print(
                f"{BackgroundColors.GREEN}Size comparison: CSV {csv_size / (1024**3):.2f} GB, Pickle {pickle_size / (1024**3):.2f} GB. Reduction: {reduction:.1f}%{Style.RESET_ALL}"
            )  # Output size comparison
        else:  # If CSV size is not available
            print(
                f"{BackgroundColors.YELLOW}Could not compare sizes: CSV size unknown.{Style.RESET_ALL}"
            )  # Output warning message


def split_dataset(df, csv_path, test_size=0.2):
    """
    Split dataset into training and testing sets.

    :param df: DataFrame to split.
    :param csv_path: Path to the CSV file for caching.
    :param test_size: Proportion of the dataset to include in the test split.
    :return: X_train, X_test, y_train, y_test
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Splitting dataset into training and testing sets with test size = {test_size}.{Style.RESET_ALL}"
    )  # Output the verbose message

    cache_file = csv_path.replace(
        ".csv", f"_cache_test{test_size}.pkl"
    )  # Cache file path, including test_size for uniqueness

    if os.path.exists(cache_file):  # If cache exists
        verbose_output(
            f"{BackgroundColors.GREEN}Loading cached preprocessed data from {cache_file}.{Style.RESET_ALL}"
        )  # Output loading message
        with open(cache_file, "rb") as f:  # Open cache file
            return pickle.load(f)  # Load and return cached data

    X = df.iloc[:, :-1].select_dtypes(include=["number"])  # Select only numeric features
    y = df.iloc[:, -1]  # Target variable
    if y.dtype == object or y.dtype == "category":  # If the target variable is categorical
        y, _ = pd.factorize(y)  # Factorize the target variable

    X = X.replace([np.inf, -np.inf], np.nan).dropna()  # Remove rows with NaN or infinite values
    y = (
        y.loc[X.index] if isinstance(y, pd.Series) else pd.Series(y, index=df.index).loc[X.index]
    )  # Align y with cleaned X

    if X.empty:  # If no numeric features remain after cleaning
        print(f"{BackgroundColors.RED}No valid numeric features remain after cleaning.{Style.RESET_ALL}")
        return None, None, None, None, None  # Return None values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )  # Split the dataset (stratified to preserve class proportions)

    scaler = StandardScaler()  # Initialize the scaler
    X_train_scaled = scaler.fit_transform(X_train)  # Fit scaler on training set and transform
    X_test_scaled = scaler.transform(X_test)  # Transform test set with the same scaler

    y_train_np = np.array(y_train)  # Convert y_train and y_test to numpy arrays for fast indexing
    y_test_np = np.array(y_test)  # Convert y_train and y_test to numpy arrays for fast indexing

    result = X_train_scaled, X_test_scaled, y_train_np, y_test_np, X.columns  # Prepare result tuple
    cache_preprocessed_data(result, cache_file, csv_path)  # Cache the preprocessed data with size comparison
    return result  # Return the splits and feature names


def print_ga_parameters(min_pop, max_pop, n_generations, feature_count):
    """
    Print the genetic algorithm parameters in verbose output.

    :param min_pop: Minimum population size.
    :param max_pop: Maximum population size.
    :param n_generations: Number of generations per run.
    :param feature_count: Number of features in the dataset.
    :return: None
    """

    print(f"{BackgroundColors.GREEN}Genetic Algorithm Parameters:{Style.RESET_ALL}")
    print(f"  {BackgroundColors.GREEN}Population sizes: {BackgroundColors.CYAN}{min_pop} to {max_pop}{Style.RESET_ALL}")
    print(f"  {BackgroundColors.GREEN}Generations per run: {BackgroundColors.CYAN}{n_generations}{Style.RESET_ALL}")
    print(f"  {BackgroundColors.GREEN}Number of features: {BackgroundColors.CYAN}{feature_count}{Style.RESET_ALL}")
    print(f"  {BackgroundColors.GREEN}Crossover probability: {BackgroundColors.CYAN}0.5{Style.RESET_ALL}")
    print(f"  {BackgroundColors.GREEN}Mutation probability: {BackgroundColors.CYAN}0.05{Style.RESET_ALL}")
    print(f"  {BackgroundColors.GREEN}Tournament size: {BackgroundColors.CYAN}3{Style.RESET_ALL}")
    print(
        f"  {BackgroundColors.GREEN}Fitness evaluation: {BackgroundColors.CYAN}10-fold Stratified CV on training set{Style.RESET_ALL}"
    )
    print(
        f"  {BackgroundColors.GREEN}Base estimator: {BackgroundColors.CYAN}RandomForestClassifier (n_estimators=100, n_jobs={N_JOBS}){Style.RESET_ALL}"
    )
    print(f"  {BackgroundColors.GREEN}Optimization goal: {BackgroundColors.CYAN}Maximize F1-Score{Style.RESET_ALL}")
    print("")  # Empty line for spacing


def prepare_sweep_data(csv_path, dataset_name, min_pop, max_pop, n_generations):
    """
    Load and preprocess dataset for GA sweep.

    :param csv_path: Path to the CSV dataset.
    :param dataset_name: Name of the dataset.
    :param min_pop: Minimum population size.
    :param max_pop: Maximum population size.
    :param n_generations: Number of generations.
    :return: Tuple (X_train, X_test, y_train, y_test, feature_names) or None if failed.
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Preparing dataset '{dataset_name}' for Genetic Algorithm sweep.{Style.RESET_ALL}"
    )  # Output the verbose message

    df = load_dataset(csv_path)  # Load dataset
    if df is None:  # If loading failed
        return None  # Exit early

    cleaned_df = preprocess_dataframe(df)  # Preprocess dataset

    if cleaned_df is None or cleaned_df.empty:  # If preprocessing failed or dataset is empty
        print(f"{BackgroundColors.RED}Dataset empty after preprocessing. Exiting.{Style.RESET_ALL}")
        return None  # Exit early

    X_train, X_test, y_train, y_test, feature_names = split_dataset(cleaned_df, csv_path)  # Split dataset
    if X_train is None:  # If splitting failed
        return None  # Exit early

    (
        print_ga_parameters(min_pop, max_pop, n_generations, len(feature_names) if feature_names is not None else 0)
        if VERBOSE
        else None
    )  # Print GA parameters if verbose

    train_count = len(y_train) if y_train is not None else 0  # Count training samples
    test_count = len(y_test) if y_test is not None else 0  # Count testing samples
    verbose_output(
        f"  {BackgroundColors.GREEN}  Dataset: {BackgroundColors.CYAN}{dataset_name} - {train_count} training / {test_count} testing  (80/20){Style.RESET_ALL}\n"
    )

    return X_train, X_test, y_train, y_test, feature_names  # Return prepared data


def compute_progress_state(min_pop, max_pop, n_generations, runs, progress_bar, folds=10):
    """
    Compute an estimated progress_state dictionary for the population sweep.

    The function returns a dict with keys:
      - current_it: starting at 0
      - total_it: estimated total number of classifier instantiations

    The estimation assumes each individual evaluation runs `folds` classifier
    instantiations (10-fold CV by default) and includes one re-evaluation of
    the best individual per run.

    :param min_pop: minimum population size
    :param max_pop: maximum population size
    :param n_generations: number of generations per run
    :param runs: number of runs per population size
    :param progress_bar: if falsy, function returns None
    :param folds: CV folds per evaluation (default 10)
    :return: dict or None
    """

    if not progress_bar:  # If no progress bar is provided
        return None  # Return None

    try:  # Try to compute the progress state
        n_pop_values = max_pop - min_pop + 1  # Number of population sizes to evaluate
        sum_pop_sizes = (min_pop + max_pop) * n_pop_values // 2  # Sum of population sizes (arithmetic series)
        total_individual_evals = runs * (
            n_generations * sum_pop_sizes + n_pop_values * 1
        )  # Total individual evaluations including best re-evaluations
        total_it = int(total_individual_evals * folds)  # Total classifier instantiations
        return {"current_it": 0, "total_it": total_it}  # Return the progress state dictionary
    except Exception:  # If any error occurs
        return {"current_it": 0, "total_it": 0}  # Return a default progress state


def compute_state_id(csv_path, pop_size, n_generations, cxpb, mutpb, run, folds, test_frac=None):
    """
    Compute a deterministic short id for a particular run configuration.

    :param csv_path: path to dataset CSV
    :param pop_size: population size
    :param n_generations: number of generations
    :param cxpb: crossover probability
    :param mutpb: mutation probability
    :param run: run index
    :param folds: CV folds
    :param test_frac: train/test fraction
    :return: hex string id or None on error
    """

    try:  # Attempt to compute the state id
        key = f"{csv_path}|pop{pop_size}|gens{n_generations}|cx{cxpb}|mut{mutpb}|run{run}|folds{folds}|test{test_frac}"  # Create a unique key string from run parameters
        return hashlib.sha256(
            key.encode("utf-8")
        ).hexdigest()  # Compute SHA256 hash of the key and return as hex string
    except Exception:  # If any error occurs during computation
        return None  # Return None to indicate failure


def state_file_paths(output_dir, state_id):
    """
    Return (gen_state_path, run_state_path) for a given state id and ensure dir exists.

    :param output_dir: base output directory
    :param state_id: deterministic id for run
    :return: tuple(gen_path, run_path)
    """

    state_dir = os.path.join(output_dir, PROGRESS_STATE_DIR_NAME)  # Construct the state directory path
    try:  # Try to create the state directory if it doesn't exist
        os.makedirs(state_dir, exist_ok=True)  # Create the directory, ignoring if it already exists
    except Exception:  # If directory creation fails
        pass  # Do nothing
    return os.path.join(state_dir, f"{state_id}_gen.pkl"), os.path.join(
        state_dir, f"{state_id}_run.pkl"
    )  # Return paths for generation and run state files


def load_run_result(output_dir, state_id):
    """
    Load a previously saved run result if present.

    :param output_dir: base output directory
    :param state_id: deterministic id for run
    :return: deserialized result or None
    """

    try:  # Attempt to load the run result
        _, run_path = state_file_paths(output_dir, state_id)  # Get the path for the run state file
        if not os.path.exists(run_path):  # Verify if the file exists
            return None  # Return None if file does not exist
        with open(run_path, "rb") as f:  # Open the file for reading in binary mode
            return pickle.load(f)  # Deserialize and return the result
    except Exception:  # If any error occurs during loading
        return None  # Return None to indicate failure


def load_cached_run_if_any(
    output_dir, csv_path, pop_size, n_generations, cxpb, mutpb, run, folds, y_train=None, y_test=None
):
    """
    Verify for and load a previously saved run result for the given parameters.

    :param output_dir: base Feature_Analysis output directory
    :param csv_path: dataset CSV path
    :param pop_size: population size
    :param n_generations: number of generations
    :param cxpb: crossover probability
    :param mutpb: mutation probability
    :param run: run index
    :param folds: CV folds
    :param y_train: training labels (optional, used to compute test_frac)
    :param y_test: testing labels (optional, used to compute test_frac)
    :return: tuple (result or None, state_id or None)
    """

    try:  # Attempt to verify for cached run result
        n_train = len(y_train) if y_train is not None else 0  # Get number of training samples
        n_test = len(y_test) if y_test is not None else 0  # Get number of test samples
        test_frac = (
            float(n_test) / float(n_train + n_test) if (n_train + n_test) > 0 else None
        )  # Calculate test fraction
        state_id = compute_state_id(
            csv_path or "", pop_size, n_generations, cxpb, mutpb, run, folds, test_frac=test_frac
        )  # Compute state id for the run
        if RESUME_PROGRESS and state_id is not None:  # If resume is enabled and state_id exists
            prev = load_run_result(output_dir, state_id)  # Load previous run result
            if prev:  # If previous result exists
                try:  # Try to log the cached result message
                    verbose_output(
                        f"{BackgroundColors.GREEN}Found cached run result for run {run} (state id {state_id[:8]}). Skipping execution.{Style.RESET_ALL}"
                    )  # Output cached result message
                except Exception:  # If logging fails
                    pass  # Do nothing
                return prev, state_id  # Return the cached result and state_id
    except Exception:  # If any error occurs during verifying
        pass  # Do nothing

    return None, None  # Return None for both result and state_id


def setup_genetic_algorithm(n_features, population_size=None, pool=None):
    """
    Setup DEAP Genetic Algorithm: creator, toolbox, population, and Hall of Fame.
    DEAP is a library for evolutionary algorithms in Python.

    :param n_features: Number of features in dataset
    :param population_size: Size of the population (default: n_features // 4, minimum 10)
    :param pool: Optional existing multiprocessing.Pool to reuse (avoids creating a new one)
    :return: toolbox, population, hall_of_fame
    """

    if population_size is None:  # If population_size is not provided
        population_size = max(n_features // 4, 10)  # Default to 1/4 of n_features, but at least 10

    verbose_output(
        f"{BackgroundColors.GREEN}Setting up Genetic Algorithm with {n_features} features and population size {population_size}.{Style.RESET_ALL}"
    )  # Output the verbose message

    FitnessMax = getattr(creator, "FitnessMax", None)  # Get or create FitnessMax
    if FitnessMax is None:
        FitnessMax = creator.create("FitnessMax", base.Fitness, weights=(1.0,))

    Individual = getattr(creator, "Individual", None)  # Get or create Individual
    if Individual is None:
        Individual = creator.create("Individual", list, fitness=FitnessMax)

    toolbox: Any = base.Toolbox()  # Toolbox (typed Any to avoid analyzer confusion)

    def _attr_bool() -> int:  # Binary attribute generator
        return random.randint(0, 1)

    toolbox.register("attr_bool", random.randint, 0, 1)

    individual_factory: Callable[[], Any] = partial(tools.initRepeat, Individual, _attr_bool, n_features)  # Individual factory and registration
    toolbox.register("individual", individual_factory)

    toolbox.register("population", tools.initRepeat, list, individual_factory)  # Population factory and registration

    toolbox.register("mate", tools.cxTwoPoint)  # Crossover operator
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # Mutation operator
    toolbox.register("select", tools.selTournament, tournsize=3)  # Selection operator

    if pool is None:  # If no external pool was provided, create one
        if CPU_PROCESSES is None:  # If CPU_PROCESSES is not set
            pool = multiprocessing.Pool()  # Create a multiprocessing pool with all available CPUs
        else:  # If CPU_PROCESSES is set
            pool = multiprocessing.Pool(
                processes=CPU_PROCESSES
            )  # Create a multiprocessing pool with specified number of CPUs
    toolbox.register("map", pool.map)  # Register parallel map for fitness evaluation

    population = toolbox.population(n=population_size)  # Create the initial population
    hof = tools.HallOfFame(1)  # Hall of Fame to store the best individual

    return toolbox, population, hof  # Return the toolbox, population, and Hall of Fame


def instantiate_estimator(estimator_cls=None):
    """
    Instantiate a classifier. If estimator_cls is None, use RandomForestClassifier.

    :param estimator_cls: Class of the estimator to instantiate (or None)
    :return: instantiated estimator
    """

    if estimator_cls is None:  # If no estimator class is provided
        return RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=N_JOBS
        )  # Return a default RandomForestClassifier

    try:  # Try to instantiate the provided estimator class
        return estimator_cls()  # Instantiate with default parameters
    except Exception:  # If instantiation fails
        return RandomForestClassifier(
            n_estimators=100, random_state=42, n_jobs=N_JOBS
        )  # Fallback to default RandomForestClassifier


def evaluate_individual(
    individual,
    X_train,
    y_train,
    X_test,
    y_test,
    estimator_cls=None,
):
    """
    Evaluate the fitness of an individual solution using 10-fold Stratified Cross-Validation
    on the training set only (não combina train+test para evitar data leakage).

    :param individual: A list representing the individual solution (binary mask for feature selection).
    :param X_train: Training feature set.
    :param y_train: Training target variable.
    :param X_test: Testing feature set (unused during CV, but kept for compatibility).
    :param y_test: Testing target variable (unused during CV, but kept for compatibility).
    :param estimator_cls: Classifier class to use (default: RandomForestClassifier).
    :return: Tuple containing CV accuracy, precision, recall, F1-score, FPR, FNR, test accuracy, precision, recall, F1-score, FPR, FNR
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Evaluating individual: {BackgroundColors.CYAN}{individual}{Style.RESET_ALL}"
    )  # Output the verbose message

    if sum(individual) == 0:  # If no features are selected
        return 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1  # Return worst possible scores for CV and test

    mask_tuple = tuple(individual)  # Convert individual to tuple for hashing
    if mask_tuple in fitness_cache:  # Verify if already evaluated
        return fitness_cache[mask_tuple]  # Return cached result

    mask = np.array(individual, dtype=bool)  # Create boolean mask from individual
    X_train_sel = X_train[:, mask]  # Select features based on the mask

    n_cv_folds = 10  # Number of CV folds
    metrics = np.empty((n_cv_folds, 6), dtype=float)  # Pre-allocate metrics array for each fold: [acc, prec, rec, f1, fpr, fnr]
    fold_count = 0  # Track how many folds actually ran

    try:  # Try to create StratifiedKFold splits
        skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)  # 10-fold Stratified CV
        splits = list(skf.split(X_train_sel, y_train))  # Generate splits
    except Exception:  # If StratifiedKFold fails (e.g., too few samples per class)
        print(
            f"{BackgroundColors.YELLOW}Warning: StratifiedKFold failed, falling back to simple train/test split for evaluation due to {str(Exception)}{Style.RESET_ALL}"
        )  # Output warning message
        X_test_sel = X_test[:, mask]  # Select features from test set
        model = instantiate_estimator(estimator_cls)  # Instantiate the model
        model.fit(X_train_sel, y_train)  # Fit the model on the training set
        y_pred = model.predict(X_test_sel)  # Predict on the test set

        acc = accuracy_score(y_test, y_pred)  # Calculate accuracy
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate precision
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate recall
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate F1-score

        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))  # Confusion matrix
        tn = cm[0, 0] if cm.shape == (2, 2) else 0  # True negatives
        fp = cm[0, 1] if cm.shape == (2, 2) else 0  # False positives
        fn = cm[1, 0] if cm.shape == (2, 2) else 0  # False negatives
        tp = cm[1, 1] if cm.shape == (2, 2) else 0  # True positives

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False negative rate

        return acc, prec, rec, f1, fpr, fnr, acc, prec, rec, f1, fpr, fnr  # Return metrics for both CV and test

    y_train_np = np.array(y_train)  # Convert y_train to numpy array for fast indexing
    early_stop_triggered = False  # Flag for early stopping

    for fold_idx, (train_idx, val_idx) in enumerate(splits):  # For each fold
        model = instantiate_estimator(estimator_cls)  # Instantiate the model
        y_train_fold = y_train_np[train_idx]  # Get training fold labels
        y_val_fold = y_train_np[val_idx]  # Get validation fold labels
        model.fit(X_train_sel[train_idx], y_train_fold)  # Fit the model on the training fold
        y_pred = model.predict(X_train_sel[val_idx])  # Predict on the validation fold

        acc = accuracy_score(y_val_fold, y_pred)  # Calculate accuracy
        prec = precision_score(y_val_fold, y_pred, average="weighted", zero_division=0)  # Calculate precision
        rec = recall_score(y_val_fold, y_pred, average="weighted", zero_division=0)  # Calculate recall
        f1 = f1_score(y_val_fold, y_pred, average="weighted", zero_division=0)  # Calculate F1-score

        cm = confusion_matrix(y_val_fold, y_pred, labels=np.unique(y_val_fold))  # Confusion matrix
        tn = cm[0, 0] if cm.shape == (2, 2) else 0  # True negatives
        fp = cm[0, 1] if cm.shape == (2, 2) else 0  # False positives
        fn = cm[1, 0] if cm.shape == (2, 2) else 0  # False negatives
        tp = cm[1, 1] if cm.shape == (2, 2) else 0  # True positives

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False negative rate

        metrics[fold_count] = [acc, prec, rec, f1, fpr, fnr]  # Write metrics directly to pre-allocated array
        fold_count += 1  # Increment fold counter

        if (
            fold_idx < EARLY_STOP_FOLDS and acc < EARLY_STOP_ACC_THRESHOLD
        ):  # Early stopping: If accuracy is below threshold in first few folds, break
            early_stop_triggered = True  # Set flag
            break  # Stop evaluating further folds for this individual

    means = np.mean(metrics[:fold_count], axis=0) if fold_count > 0 else np.zeros(6)  # Calculate means for completed folds only
    acc, prec, rec, f1, fpr, fnr = means  # Unpack mean metrics

    test_acc, test_prec, test_rec, test_f1, test_fpr, test_fnr = 0, 0, 0, 0, 0, 0  # Placeholder test metrics

    result = acc, prec, rec, f1, fpr, fnr, test_acc, test_prec, test_rec, test_f1, test_fpr, test_fnr  # Prepare result tuple
    fitness_cache[mask_tuple] = result  # Cache the result
    
    return result  # Return vectorized average metrics


def ga_fitness(ind, fitness_func):
    """
    Global fitness function for GA evaluation to avoid pickle issues with local functions.

    :param ind: Individual to evaluate
    :param fitness_func: Partial function for evaluation
    :return: Tuple with F1-score
    """

    return (fitness_func(ind)[3],)  # Return only the F1-score for GA optimization


def evaluate_individual_with_test(individual, X_train, y_train, X_test, y_test, estimator_cls=None):
    """
    Evaluate an individual with FULL test-set metrics. Use this only for the
    final best individual re-evaluation (not during GA evolution loop).

    This computes 10-fold CV metrics AND trains a model on full training
    data to produce test-set metrics — unlike evaluate_individual() which
    defers test evaluation to avoid waste during the GA loop.

    :param individual: Binary mask (list) for feature selection.
    :param X_train: Training feature set (numpy array).
    :param y_train: Training target variable (numpy array).
    :param X_test: Testing feature set (numpy array).
    :param y_test: Testing target variable (numpy array).
    :param estimator_cls: Classifier class to use (default: RandomForestClassifier).
    :return: 12-element tuple: (cv_acc, cv_prec, cv_rec, cv_f1, cv_fpr, cv_fnr,
             test_acc, test_prec, test_rec, test_f1, test_fpr, test_fnr)
    """

    if sum(individual) == 0:  # If no features are selected
        return 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1  # Return worst possible scores

    cv_result = evaluate_individual(individual, X_train, y_train, X_test, y_test, estimator_cls)  # Get CV metrics from the standard evaluation function
    cv_acc, cv_prec, cv_rec, cv_f1, cv_fpr, cv_fnr = cv_result[:6]  # Extract CV metrics

    mask = np.array(individual, dtype=bool)  # Create boolean mask
    X_train_sel = X_train[:, mask]  # Select training features
    X_test_sel = X_test[:, mask]  # Select test features

    model = instantiate_estimator(estimator_cls)  # Instantiate model
    model.fit(X_train_sel, y_train)  # Train on full training set
    y_pred_test = model.predict(X_test_sel)  # Predict on test set

    test_acc = accuracy_score(y_test, y_pred_test)  # Calculate test accuracy
    test_prec = precision_score(y_test, y_pred_test, average="weighted", zero_division=0)  # Calculate test precision
    test_rec = recall_score(y_test, y_pred_test, average="weighted", zero_division=0)  # Calculate test recall
    test_f1 = f1_score(y_test, y_pred_test, average="weighted", zero_division=0)  # Calculate test F1-score
    cm = confusion_matrix(y_test, y_pred_test, labels=np.unique(y_test))  # Confusion matrix for test set
    tn = cm[0, 0] if cm.shape == (2, 2) else 0  # True negatives
    fp = cm[0, 1] if cm.shape == (2, 2) else 0  # False positives
    fn = cm[1, 0] if cm.shape == (2, 2) else 0  # False negatives
    tp = cm[1, 1] if cm.shape == (2, 2) else 0  # True positives
    test_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Calculate test false positive rate
    test_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # Calculate test false negative rate

    return cv_acc, cv_prec, cv_rec, cv_f1, cv_fpr, cv_fnr, test_acc, test_prec, test_rec, test_f1, test_fpr, test_fnr  # Return combined CV and test metrics


def load_generation_state(output_dir, state_id):
    """
    Load generation state if present, returning the payload or None.

    :param output_dir: base output directory
    :param state_id: deterministic id for run
    :return: payload dict or None
    """

    try:  # Attempt to load the generation state
        gen_path, _ = state_file_paths(output_dir, state_id)  # Get the path for the generation state file
        if not os.path.exists(gen_path):  # Verify if the file exists
            return None  # Return None if file does not exist
        with open(gen_path, "rb") as f:  # Open the file for reading in binary mode
            return pickle.load(f)  # Deserialize and return the payload
    except Exception:  # If any error occurs during loading
        return None  # Return None to indicate failure


def recreate_population_from_lists(toolbox, pop_lists):
    """
    Create DEAP individual objects from plain lists using registered toolbox.individual.

    :param toolbox: DEAP toolbox with `individual` registered
    :param pop_lists: iterable of bit-lists
    :return: list of individuals or empty list on error
    """

    population = []  # Initialize an empty list for the population
    try:  # Attempt to recreate the population
        for bits in pop_lists:  # Iterate over each bit list in pop_lists
            ind = toolbox.individual()  # Create a new individual using the toolbox
            for i, b in enumerate(bits):  # Iterate over each bit in the list
                try:  # Try to convert the bit to int
                    ind[i] = int(b)  # Set the individual's gene to the integer value
                except Exception:  # If conversion fails
                    ind[i] = b  # Set the individual's gene to the original value
            ind.fitness.values = ()  # Initialize fitness values as empty tuple
            population.append(ind)  # Add the individual to the population
    except Exception:  # If any error occurs during recreation
        return []  # Return an empty list
    return population  # Return the recreated population


def load_and_apply_generation_state(toolbox, population, output_dir, state_id, run=None):
    """
    Load a saved GA generation state and apply it to the provided population.

    :param toolbox: DEAP toolbox used to recreate individuals
    :param population: population list to modify in-place
    :param output_dir: directory where generation state files live
    :param state_id: deterministic id for the run state
    :param run: optional run index for logging
    :return: tuple (start_gen:int, fitness_history:list)
    """

    start_gen = 1  # Initialize starting generation to 1
    fitness_history = []  # Initialize fitness history as empty list
    if RESUME_PROGRESS and state_id is not None:  # Verify if resume is enabled and state_id is provided
        try:  # Attempt to load and apply the state
            payload = load_generation_state(output_dir, state_id)  # Load the generation state payload
            if payload:  # If payload exists
                pop_lists = payload.get("population_lists")  # Get the population lists from payload
                if pop_lists:  # If population lists exist
                    recreated = recreate_population_from_lists(toolbox, pop_lists)  # Recreate population from lists
                    if recreated:  # If recreation succeeded
                        population[:] = recreated  # Replace the population with recreated one
                    fitness_history = payload.get("fitness_history", [])  # Get fitness history from payload
                loaded_gen = int(payload.get("gen", 0))  # Get the loaded generation number
                start_gen = loaded_gen + 1 if loaded_gen >= 1 else 1  # Set start_gen to loaded_gen + 1 or 1
                try:  # Try to log the resume message
                    verbose_output(
                        f"{BackgroundColors.GREEN}Resuming GA from generation {start_gen} for run {run} (state id {state_id[:8]}){Style.RESET_ALL}"
                    )  # Output resume message
                except Exception:  # If logging fails
                    pass  # Do nothing
        except Exception:  # If any error occurs during loading/applying
            pass  # Do nothing

    return start_gen, fitness_history  # Return the starting generation and fitness history


def save_generation_state(output_dir, state_id, gen, population, hof_best, fitness_history):
    """
    Persist minimal generation state to disk (lists only).

    :param output_dir: base output directory
    :param state_id: deterministic id for run
    :param gen: generation number
    :param population: list of individuals
    :param hof_best: best individual list or None
    :param fitness_history: list of fitness history
    :return: None
    """

    try:  # Attempt to save the generation state
        gen_path, _ = state_file_paths(output_dir, state_id)  # Get the path for the generation state file
        payload = {  # Prepare the payload dictionary
            "gen": int(gen),  # Current generation number
            "population_lists": [list(ind) for ind in population],  # List of population individuals as lists
            "hof_best": list(hof_best) if hof_best is not None else None,  # Best individual from hall of fame
            "fitness_history": (
                list(fitness_history) if fitness_history is not None else []
            ),  # History of fitness values
        }  # End of payload dictionary
        with open(gen_path, "wb") as f:  # Open the file for writing in binary mode
            pickle.dump(payload, f, protocol=PICKLE_PROTOCOL)  # Serialize and save the payload
    except Exception:  # If any error occurs during saving
        pass  # Do nothing


def run_genetic_algorithm_loop(
    toolbox,
    population,
    hof,
    X_train,
    y_train,
    X_test,
    y_test,
    n_generations=100,
    show_progress=False,
    progress_bar=None,
    dataset_name=None,
    csv_path=None,
    pop_size=None,
    max_pop=None,
    cxpb=0.5,
    mutpb=0.2,
    run=None,
    runs=None,
    progress_state=None,
):
    """
    Run Genetic Algorithm generations with a tqdm progress bar.

    :param toolbox: DEAP toolbox with registered functions.
    :param population: Initial population.
    :param hof: Hall of Fame to store the best individual.
    :param X_train: Training feature set.
    :param y_train: Training target variable.
    :param X_test: Testing feature set.
    :param y_test: Testing target variable.
    :param n_generations: Number of generations to run.
    :return: best individual
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Running Genetic Algorithm for {n_generations} generations.{Style.RESET_ALL}"
    )  # Output the verbose message

    fitness_func = partial(
        evaluate_individual, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
    )  # Partial function for evaluation
    toolbox.register("evaluate", partial(ga_fitness, fitness_func=fitness_func))  # Register the global fitness function

    global GA_GENERATIONS_COMPLETED  # To track completed generations
    best_fitness = None  # Track the best fitness value
    gens_without_improvement = 0  # Counter for generations with no improvement
    early_stop_gens = 10  # Number of generations to wait for improvement before stopping

    folds = 10  # Number of folds used in cross-validation

    output_dir = (
        f"{os.path.dirname(csv_path)}/Feature_Analysis" if csv_path else os.path.join(".", "Feature_Analysis")
    )  # Output directory for Feature_Analysis outputs
    state_id = compute_state_id(
        csv_path or "", pop_size or 0, n_generations, cxpb, mutpb, run or 0, folds, test_frac=None
    )  # Deterministic state id for resume/caching
    start_gen = 1  # Starting generation index
    fitness_history = []  # List to record best fitness per generation for convergence plot
    start_gen, fitness_history = load_and_apply_generation_state(
        toolbox, population, output_dir, state_id, run=run
    )  # Load and apply saved generation state if available, updating start generation and fitness history

    gen_range = (
        tqdm(range(start_gen, n_generations + 1), desc=f"{BackgroundColors.GREEN}Generations{Style.RESET_ALL}")
        if show_progress
        else range(start_gen, n_generations + 1)
    )  # Create generation range with progress bar if show_progress is enabled, otherwise use plain range
    gens_ran = 0  # Track how many generations were actually executed
    for gen in gen_range:  # Loop for the specified number of generations
        (
            update_progress_bar(
                progress_bar,
                dataset_name or "",
                csv_path or "",
                pop_size=pop_size,
                max_pop=max_pop,
                gen=gen,
                n_generations=n_generations,
                run=run,
                runs=runs,
                progress_state=progress_state,
            )
            if progress_bar
            else None
        )  # Update progress bar if provided

        offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)  # Apply crossover and mutation

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]  # Filter to only invalid individuals
        if invalid_ind:  # If there are individuals to evaluate
            invalid_fits = list(toolbox.map(toolbox.evaluate, invalid_ind))  # Evaluate only invalid offspring
            for ind, fit in zip(invalid_ind, invalid_fits):  # Assign fitness to evaluated individuals
                ind.fitness.values = fit  # Set the fitness value

        if progress_state and isinstance(progress_state, dict):  # Update progress state if provided
            try:  # Try to update progress state
                progress_state["current_it"] = int(progress_state.get("current_it", 0)) + len(invalid_ind) * folds
                (
                    update_progress_bar(
                        progress_bar,
                        dataset_name or "",
                        csv_path or "",
                        pop_size=pop_size,
                        max_pop=max_pop,
                        gen=gen,
                        n_generations=n_generations,
                        run=run,
                        runs=runs,
                        progress_state=progress_state,
                    )
                    if progress_bar
                    else None
                )  # Update progress bar if provided
            except Exception:  # Silently ignore progress update errors
                pass  # Do nothing

        population[:] = toolbox.select(offspring, k=len(population))  # Select the next generation population
        hof.update(population)  # Update the Hall of Fame

        if hof and len(hof) > 0:  # If hall of fame has a best individual
            if hof[0] not in population:  # If the best individual is not in the new population
                population[-1] = hof[0]  # Replace the worst individual with the hall-of-fame best

        current_best_fitness = (
            hof[0].fitness.values[0] if hof and hof[0].fitness.values else None
        )  # Get current best fitness
        try:  # Try to append best fitness to history
            fitness_history.append(
                float(current_best_fitness) if current_best_fitness is not None else np.nan
            )  # Record best fitness
        except Exception:  # If conversion fails
            fitness_history.append(np.nan)  # Record NaN
        if best_fitness is None or (current_best_fitness is not None and current_best_fitness > best_fitness):
            best_fitness = current_best_fitness  # Update best fitness
            gens_without_improvement = 0  # Reset counter
        else:  # If no improvement in best fitness
            gens_without_improvement += 1  # Increment counter

            if gens_without_improvement >= early_stop_gens:  # Verify early-stop condition
                print(
                    f"{BackgroundColors.YELLOW}Early stopping: No improvement in best fitness for {early_stop_gens} generations. Stopping at generation {gen}.{Style.RESET_ALL}"
                )  # Print early stopping message
                gens_ran = gen  # Record how many generations were executed before early stopping
                GA_GENERATIONS_COMPLETED = int(gen)  # Update global variable
                break  # Stop the loop early

        send_telegram_message(TELEGRAM_BOT, [
            f"Pop Size {pop_size}: Generation {gen}/{n_generations}, Best F1-Score: {truncate_value(best_fitness)}"
        ], show_progress and gen % max(1, n_generations // 100) == 0)  # Send periodic updates to Telegram telegram_bot

        gens_ran = gen  # Update gens_ran each generation
        GA_GENERATIONS_COMPLETED = int(gen)  # Update global variable
        gens_ran = gen if gens_ran == 0 else gens_ran  # Ensure gens_ran is set correctly if no early stopping occurred

        try:  # Persist per-generation progress so runs can be resumed (every 10 gens to reduce I/O)
            if RESUME_PROGRESS and state_id is not None and (gen % 10 == 0 or gen == n_generations):  # Save every 10 generations or at the end
                save_generation_state(
                    output_dir, state_id, gen, population, hof[0] if hof and len(hof) > 0 else None, fitness_history
                )
        except Exception:  # If saving fails
            pass  # Do nothing

    if hasattr(toolbox, "map") and hasattr(toolbox.map, "close"):  # If using multiprocessing pool
        pass  # Pool lifecycle is now managed by the caller (run_population_sweep) to allow reuse

    return hof[0], gens_ran, fitness_history  # Return the best individual, gens ran and fitness history


def adjust_progress_for_early_stop(progress_state, n_generations, pop_size, gens_ran, folds):
    """
    Adjust `progress_state` when a GA run finishes early.

    This subtracts planned-but-not-executed classifier instantiations from
    `progress_state['total_it']` and increments `current_it` for the final
    re-evaluation (which is always performed once per run).

    :param progress_state: dict with keys `current_it` and `total_it`.
    :param n_generations: configured total generations for the run.
    :param pop_size: population size used for the run.
    :param gens_ran: number of generations actually executed.
    :param folds: number of CV folds (classifier instantiations per evaluation).
    :return: None (mutates `progress_state` in-place)
    """

    if not (progress_state and isinstance(progress_state, dict)):  # Validate progress_state
        return  # Nothing to do if invalid

    try:  # Try to compute planned vs actual evaluations
        planned = int(n_generations) * int(pop_size) + 1  # Planned evaluations: generations * pop_size + 1 re-eval
        actual = int(gens_ran) * int(pop_size) + 1  # Actual evaluations: generations run * pop_size + 1 re-eval
        delta = max(0, planned - actual)  # Number of generation-evaluations saved by early stopping
        progress_state["total_it"] = (
            int(progress_state.get("total_it", 0)) - delta * folds
        )  # Reduce total iterations by saved evaluations
    except Exception:  # Silently ignore failures during adjustment
        pass  # Do nothing on error

    try:  # Update current_it by the single re-evaluation performed after GA (folds classifiers)
        progress_state["current_it"] = (
            int(progress_state.get("current_it", 0)) + folds
        )  # Increment current_it for final re-eval
    except Exception:  # Silently ignore failures when updating current_it
        pass  # Do nothing on error


def safe_filename(name):
    """
    Sanitize a string to be safe for use as a filename.

    :param name: The string to be sanitized.
    :return: A sanitized string safe for use as a filename.
    """

    return re.sub(r'[\\/*?:"<>|]', "_", name)  # Replace invalid filename characters with underscores


def plot_ga_convergence(
    csv_path, pop_size, run, fitness_history, dataset_name=None, n_generations=None, cxpb=0.5, mutpb=0.2
):
    """
    Plot and save the GA convergence curve (best fitness per generation) for a
    specific run and population size.

    :param csv_path: Path to the dataset CSV (used to determine output directory)
    :param pop_size: Population size used in this run
    :param run: Run index (1-based)
    :param fitness_history: List of best fitness values per generation
    :param dataset_name: Optional dataset name for title/filename
    :return: Path to saved image
    """

    output_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis"  # Directory to save outputs
    os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

    base_dataset_name = (
        safe_filename(os.path.splitext(os.path.basename(csv_path))[0])
        if not dataset_name
        else safe_filename(dataset_name)
    )  # Base name of the dataset
    gens_part = f"gens{int(n_generations)}" if n_generations is not None else "gensNA"
    cx_part = f"cx{int(cxpb*100)}"
    mut_part = f"mut{int(mutpb*100)}"
    fig_path = os.path.join(
        output_dir, f"{base_dataset_name}_run{run}_pop{pop_size}_{gens_part}_{cx_part}_{mut_part}_convergence.png"
    )  # Path to save the figure

    try:  # Try to plot and save the figure
        plt.figure(figsize=(8, 4))  # Create a matplotlib figure
        gens = list(range(1, len(fitness_history) + 1))  # Generation numbers
        plt.plot(gens, fitness_history, marker="o", linestyle="-", color="#1f77b4")  # Plot the fitness history
        plt.xlabel("Generation")  # X-axis label
        plt.ylabel("Best F1-Score")  # Y-axis label
        plt.title(f"GA Convergence - {base_dataset_name} (run={run}, pop={pop_size}, cx={cxpb}, mut={mutpb})")  # Title
        plt.grid(True, linestyle="--", alpha=0.5)  # Grid
        plt.tight_layout()  # Adjust layout
        plt.savefig(fig_path, dpi=150)  # Save the figure
        plt.close()  # Close the plot to free memory
        verbose_output(
            f"{BackgroundColors.GREEN}Saved GA convergence plot to {BackgroundColors.CYAN}{fig_path}{Style.RESET_ALL}"
        )  # Notify user
        return fig_path  # Return the path to the saved figure
    except Exception as e:  # If any error occurs during plotting
        try:  # Try to close the plot if open
            plt.close()  # Close the plot to free memory
        except Exception:  # Ignore errors during plot closing
            pass  # Do nothing
        raise  # Reraise the original exception


def save_run_result(output_dir, state_id, result):
    """
    Save a completed run result so future identical runs can be skipped.

    :param output_dir: base output directory
    :param state_id: deterministic id for run
    :param result: serializable run result
    :return: None
    """

    try:  # Attempt to save the run result
        _, run_path = state_file_paths(output_dir, state_id)  # Get the path for the run state file
        with open(run_path, "wb") as f:  # Open the file for writing in binary mode
            pickle.dump(result, f, protocol=PICKLE_PROTOCOL)  # Serialize and save the result
    except Exception:  # If any error occurs during saving
        pass  # Do nothing


def cleanup_state_for_id(output_dir, state_id):
    """
    Remove progress files for a finished run/generation (best-effort).

    :param output_dir: base output directory
    :param state_id: deterministic id for run
    :return: None
    """

    try:  # Attempt to clean up state files
        gen_path, run_path = state_file_paths(output_dir, state_id)  # Get paths for generation and run state files
        for p in (gen_path, run_path):  # Iterate over the file paths
            try:  # Try to remove each file
                if os.path.exists(p):  # Verify if the file exists
                    os.remove(p)  # Remove the file
            except Exception:  # If removal fails
                pass  # Do nothing
    except Exception:  # If any error occurs during cleanup
        pass  # Do nothing


def run_single_ga_iteration(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    pop_size,
    n_generations,
    cxpb,
    mutpb,
    run,
    runs,
    dataset_name,
    csv_path,
    max_pop,
    progress_bar,
    progress_state,
    folds,
    shared_pool=None,
):
    """
    Execute one GA run for a specific population size.

    :param X_train: Training features.
    :param y_train: Training labels.
    :param X_test: Test features.
    :param y_test: Test labels.
    :param feature_names: List of feature names.
    :param pop_size: Population size for this iteration.
    :param n_generations: Number of generations.
    :param cxpb: Crossover probability.
    :param mutpb: Mutation probability.
    :param run: Current run number (1-based).
    :param runs: Total number of runs.
    :param dataset_name: Dataset name.
    :param csv_path: Path to CSV.
    :param max_pop: Maximum population size.
    :param progress_bar: Progress bar instance.
    :param progress_state: Progress state dict.
    :param folds: Number of CV folds.
    :return: Dict with best_ind, metrics, best_features or None if failed.
    """

    iteration_start_time = time.time()  # Start tracking total iteration time

    feature_count = len(feature_names) if feature_names is not None else 0  # Count of features

    (
        update_progress_bar(
            progress_bar,
            dataset_name,
            csv_path,
            pop_size=pop_size,
            max_pop=max_pop,
            n_generations=n_generations,
            run=run,
            runs=runs,
            progress_state=progress_state,
        )
        if progress_bar
        else None
    )  # Update progress bar if provided

    output_dir = (
        f"{os.path.dirname(csv_path)}/Feature_Analysis" if csv_path else os.path.join(".", "Feature_Analysis")
    )  # Output directory for Feature_Analysis outputs
    cached, state_id = load_cached_run_if_any(
        output_dir, csv_path, pop_size, n_generations, cxpb, mutpb, run, folds, y_train, y_test
    )  # Try to load cached run result and state id
    if cached:  # If cached result found
        return cached  # Return cached result immediately

    toolbox, population, hof = setup_genetic_algorithm(feature_count, pop_size, pool=shared_pool)  # Setup GA components, reusing shared pool
    best_ind, gens_ran, fitness_history = run_genetic_algorithm_loop(
        toolbox,
        population,
        hof,
        X_train,
        y_train,
        X_test,
        y_test,
        n_generations,
        show_progress=False,
        progress_bar=progress_bar,
        dataset_name=dataset_name,
        csv_path=csv_path,
        pop_size=pop_size,
        max_pop=max_pop,
        cxpb=cxpb,
        mutpb=mutpb,
        run=run,
        runs=runs,
        progress_state=progress_state,
    )  # Run GA loop and get generations actually run and fitness history

    if best_ind is None:  # If GA failed
        return None  # Exit early

    metrics = evaluate_individual_with_test(best_ind, X_train, y_train, X_test, y_test)  # Evaluate best individual with full test metrics

    adjust_progress_for_early_stop(
        progress_state, n_generations, pop_size, gens_ran, folds
    )  # Update progress_state in helper
    (
        update_progress_bar(
            progress_bar,
            dataset_name,
            csv_path,
            pop_size=pop_size,
            max_pop=max_pop,
            n_generations=n_generations,
            run=run,
            runs=runs,
            progress_state=progress_state,
        )
        if progress_bar
        else None
    )  # Update progress bar

    best_features = [
        f for f, bit in zip(feature_names if feature_names is not None else [], best_ind) if bit == 1
    ]  # Extract best features

    iteration_elapsed_time = time.time() - iteration_start_time  # Calculate total iteration time
    metrics_with_iteration_time = metrics + (iteration_elapsed_time,)  # Add total iteration time as 7th element

    try:  # Try to generate GA convergence plot
        plot_ga_convergence(
            csv_path, pop_size, run, fitness_history, dataset_name, n_generations=n_generations, cxpb=cxpb, mutpb=mutpb
        )  # Generate convergence plot
    except Exception as e:  # On any plotting error
        verbose_output(
            f"{BackgroundColors.YELLOW}Failed to generate GA convergence plot: {e}{Style.RESET_ALL}"
        )  # Log warning

    result = {
        "best_ind": best_ind,
        "metrics": metrics_with_iteration_time,
        "best_features": best_features,
    }  # Build result dict

    try:  # Try to save run result
        if RESUME_PROGRESS and state_id is not None:  # If resume is enabled and state_id exists
            save_run_result(output_dir, state_id, result)  # Save the run result
            cleanup_state_for_id(output_dir, state_id)  # Cleanup state files
    except Exception:  # On any saving error
        pass  # Do nothing

    return result  # Return results


def aggregate_sweep_results(results, min_pop, max_pop, dataset_name):
    """
    Aggregate results per population size and find best.

    :param results: Dict mapping pop_size to runs list.
    :param min_pop: Minimum population size.
    :param max_pop: Maximum population size.
    :param dataset_name: Dataset name.
    :return: Tuple (best_score, best_result, best_metrics, results_dict).
    """

    best_score = -1  # Initialize best score
    best_result = None  # Initialize best result
    best_metrics = None  # Initialize best metrics

    for pop_size in range(min_pop, max_pop + 1):  # For each population size
        runs_list = results[pop_size]["runs"]  # Get runs list
        if not runs_list:  # If no runs
            results.pop(pop_size, None)  # Remove entry
            continue  # Skip to next population size

        all_metrics = [r["metrics"] for r in runs_list]  # Collect all metrics
        avg_metrics = tuple(np.mean(all_metrics, axis=0))  # Compute average metrics

        feature_sets = [set(r["best_features"]) for r in runs_list]  # Collect feature sets
        common_features = set.intersection(*feature_sets) if feature_sets else set()  # Find common features

        results[pop_size]["avg_metrics"] = avg_metrics  # Store average metrics
        results[pop_size]["common_features"] = common_features  # Store common features

        f1_avg = avg_metrics[3]  # Average F1-Score
        if f1_avg > best_score:  # If this is the best score so far
            best_score = f1_avg  # Update best score
            best_metrics = avg_metrics  # Update best metrics
            best_result = (pop_size, runs_list, common_features)  # Update best result

        print(
            f"{BackgroundColors.GREEN}Pop {BackgroundColors.CYAN}{pop_size}{BackgroundColors.GREEN}: AVG F1 {BackgroundColors.GREEN}{truncate_value(f1_avg)}{BackgroundColors.GREEN}, Common Features {BackgroundColors.CYAN}{len(common_features)}{Style.RESET_ALL}"
        )  # Print summary
        for i, run_data in enumerate(runs_list):  # For each run
            unique = set(run_data["best_features"]) - common_features  # Unique features
            print(
                f"  {BackgroundColors.GREEN}Run {BackgroundColors.CYAN}{i+1}{BackgroundColors.GREEN}: {BackgroundColors.GREEN}unique features {BackgroundColors.CYAN}{len(unique)}{Style.RESET_ALL}"
            )  # Print unique features count

        send_telegram_message(TELEGRAM_BOT, [
            f"Completed {len(runs_list)} runs for population size {pop_size} on {dataset_name} -> AVG F1-Score: {truncate_value(f1_avg)}"
        ])  # Send progress message

    return best_score, best_result, best_metrics, results  # Return aggregated results


def print_metrics(metrics):
    """
    Print performance metrics.

    :param metrics: Dictionary or tuple containing evaluation metrics.
    :return: None
    """

    if not metrics:  # If metrics is None or empty
        return  # Do nothing

    cv_acc, cv_prec, cv_rec, cv_f1, cv_fpr, cv_fnr, test_acc, test_prec, test_rec, test_f1, test_fpr, test_fnr = metrics
    print(
        f"\n{BackgroundColors.GREEN}CV Performance Metrics for the Random Forest Classifier using the best feature subset:{Style.RESET_ALL}"
    )
    print(f"   {BackgroundColors.GREEN}Accuracy: {BackgroundColors.CYAN}{truncate_value(cv_acc)}{Style.RESET_ALL}")
    print(f"   {BackgroundColors.GREEN}Precision: {BackgroundColors.CYAN}{truncate_value(cv_prec)}{Style.RESET_ALL}")
    print(f"   {BackgroundColors.GREEN}Recall: {BackgroundColors.CYAN}{truncate_value(cv_rec)}{Style.RESET_ALL}")
    print(f"   {BackgroundColors.GREEN}F1-Score: {BackgroundColors.CYAN}{truncate_value(cv_f1)}{Style.RESET_ALL}")
    print(f"   {BackgroundColors.GREEN}False Positive Rate (FPR): {BackgroundColors.CYAN}{truncate_value(cv_fpr)}{Style.RESET_ALL}")
    print(f"   {BackgroundColors.GREEN}False Negative Rate (FNR): {BackgroundColors.CYAN}{truncate_value(cv_fnr)}{Style.RESET_ALL}")
    print(
        f"\n{BackgroundColors.GREEN}Test Performance Metrics for the Random Forest Classifier using the best feature subset:{Style.RESET_ALL}"
    )
    print(f"   {BackgroundColors.GREEN}Accuracy: {BackgroundColors.CYAN}{truncate_value(test_acc)}{Style.RESET_ALL}")
    print(f"   {BackgroundColors.GREEN}Precision: {BackgroundColors.CYAN}{truncate_value(test_prec)}{Style.RESET_ALL}")
    print(f"   {BackgroundColors.GREEN}Recall: {BackgroundColors.CYAN}{truncate_value(test_rec)}{Style.RESET_ALL}")
    print(f"   {BackgroundColors.GREEN}F1-Score: {BackgroundColors.CYAN}{truncate_value(test_f1)}{Style.RESET_ALL}")
    print(f"   {BackgroundColors.GREEN}False Positive Rate (FPR): {BackgroundColors.CYAN}{truncate_value(test_fpr)}{Style.RESET_ALL}")
    print(f"   {BackgroundColors.GREEN}False Negative Rate (FNR): {BackgroundColors.CYAN}{truncate_value(test_fnr)}{Style.RESET_ALL}")


def extract_rfe_ranking(csv_path):
    """
    Extract RFE rankings from the RFE results file.

    :param csv_path: Path to the original CSV file for saving outputs.
    :return: Dictionary of feature names and their RFE rankings.
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Extracting RFE rankings from results file.{Style.RESET_ALL}"
    )  # Output the verbose message

    rfe_ranking = {}  # Dictionary to store feature names and their RFE rankings
    dir_path = os.path.dirname(csv_path)  # Directory that contains Feature_Analysis
    json_path = f"{dir_path}/Feature_Analysis/RFE_Summary.json"  # Path to new JSON summary
    csv_path_runs = f"{dir_path}/Feature_Analysis/RFE_Runs_Summary.csv"  # Path to runs summary CSV
    legacy_txt = f"{dir_path}/Feature_Analysis/RFE_results_RandomForestClassifier.txt"  # Legacy TXT path (fallback)

    if verify_filepath_exists(json_path):  # If JSON summary exists
        try:  # Attempt to parse JSON
            with open(json_path, "r", encoding="utf-8") as jf:  # Open JSON file
                data = json.load(jf)  # Load JSON content
            if isinstance(data, dict):  # Ensure data is a dictionary
                if "rfe_ranking" in data and isinstance(data["rfe_ranking"], dict):  # Rfe_ranking key
                    rfe_ranking = data["rfe_ranking"]  # Use provided ranking
                elif (
                    "per_run" in data and isinstance(data["per_run"], list) and len(data["per_run"]) > 0
                ):  # Per_run list exists
                    first = data["per_run"][0]  # First run entry
                    if (
                        isinstance(first, dict) and "ranking" in first and isinstance(first["ranking"], dict)
                    ):  # Ranking dict
                        rfe_ranking = first["ranking"]  # Use ranking
        except Exception as e:  # If parsing JSON fails
            print(
                f"{BackgroundColors.YELLOW}Failed to parse RFE JSON summary: {str(e)}. Skipping RFE ranking extraction.{Style.RESET_ALL}"
            )  # Warn user
            return rfe_ranking  # Return whatever we have (likely empty)
        return rfe_ranking  # Return ranking extracted from JSON

    if verify_filepath_exists(csv_path_runs):  # If CSV summary exists
        try:  # Attempt to parse CSV
            with open(csv_path_runs, "r", encoding="utf-8") as cf:  # Open CSV file
                reader = csv.DictReader(cf)  # Use DictReader to parse headered CSV
                for row in reader:  # Iterate rows
                    for key, val in row.items():  # For each column value
                        if val and isinstance(val, str) and val.strip().startswith("{"):  # Looks like JSON
                            try:  # Try parse JSON string
                                parsed = json.loads(val)  # Parse JSON string
                                if isinstance(parsed, dict) and all(
                                    isinstance(k, str) for k in parsed.keys()
                                ):  # Likely ranking
                                    rfe_ranking = parsed  # Use parsed dict as ranking
                                    return rfe_ranking  # Return early
                            except Exception:  # Ignore parse errors and continue
                                pass
        except Exception as e:  # If reading CSV fails
            print(
                f"{BackgroundColors.YELLOW}Failed to parse RFE CSV summary: {str(e)}. Skipping RFE ranking extraction.{Style.RESET_ALL}"
            )  # Warn user

    if not verify_filepath_exists(legacy_txt):  # If no legacy file either
        print(
            f"{BackgroundColors.YELLOW}RFE results not found (tried JSON/CSV/TXT). Skipping RFE ranking extraction.{Style.RESET_ALL}"
        )  # Notify user
        return rfe_ranking  # Return empty dictionary

    try:  # Attempt to parse TXT file
        with open(legacy_txt, "r", encoding="utf-8") as f:  # Open legacy TXT
            lines = f.readlines()  # Read lines
        for line in lines:  # Iterate lines
            line = line.strip()  # Strip whitespace
            if not line:  # Skip empty lines
                continue  # Continue
            m = re.match(r"^\s*(\d+)\.?\s+(.+?)\s*$", line)  # Try numeric prefix
            if m:  # If matched numbered list
                rank = int(m.group(1))  # Get rank number
                feat = m.group(2).strip()  # Get feature name
                rfe_ranking[feat] = rank  # Store ranking
    except Exception as e:  # If parsing fails
        print(
            f"{BackgroundColors.YELLOW}Failed to parse legacy RFE TXT: {str(e)}. Returning empty ranking.{Style.RESET_ALL}"
        )  # Notify user

    return rfe_ranking  # Return the RFE rankings dictionary


def extract_elapsed_from_metrics(metrics, index=6):
    """
    Safely extract an elapsed-time value from a metrics tuple.

    :param metrics: Metrics tuple (acc, prec, rec, f1, fpr, fnr, elapsed) or None
    :param index: Index within the tuple where elapsed time is expected (default 6)
    :return: elapsed value (float) if available, otherwise None
    """

    if not metrics:  # If metrics is None or falsy
        return None  # Return None
    try:  # Try to extract elapsed time
        if isinstance(metrics, (list, tuple)) and len(metrics) > index:  # Verify index is valid
            return metrics[index]  # Return elapsed time
    except Exception:  # On any error
        pass  # Ignore errors

    return None  # Return None if extraction fails


def build_base_row(
    csv_path,
    best_pop_size,
    n_generations,
    n_train,
    n_test,
    test_frac,
    rfe_ranking,
    elapsed_time_s=None,
    cxpb=None,
    mutpb=None,
):
    """
    Build the base dictionary row used for the consolidated GA CSV output.

    :param csv_path: Path to the original dataset CSV.
    :param best_pop_size: Population size that produced the best result.
    :param n_generations: Number of generations used in the GA.
    :param n_train: Number of training samples.
    :param n_test: Number of testing samples.
    :param test_frac: Train/test fraction.
    :param rfe_ranking: Dictionary with RFE rankings (will be JSON-encoded).
    :return: Dictionary representing the base row for CSV output.
    """

    return {  # Base row for CSV
        "dataset": os.path.splitext(os.path.basename(csv_path))[0],  # Dataset name
        "dataset_path": csv_path,  # Dataset path
        "run_index": "best",  # Indicates best run
        "population_size": best_pop_size,  # Population size
        "n_generations": n_generations,  # Number of generations
        "cxpb": cxpb,  # Crossover probability used in GA
        "mutpb": mutpb,  # Mutation probability used in GA
        "n_train": n_train,  # Number of training samples
        "n_test": n_test,  # Number of testing samples
        "train_test_ratio": test_frac,  # Train/test fraction
        "elapsed_time_s": (
            int(round(elapsed_time_s)) if elapsed_time_s is not None else None
        ),  # Elapsed seconds for best-model training
        "rfe_ranking": json.dumps(rfe_ranking, ensure_ascii=False),  # RFE ranking as JSON string
    }


def truncate_value(value):
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


def normalize_elapsed_column_df(df):
    """
    Normalize elapsed time column name to `elapsed_time_s` if the legacy
    `elapsed_time` column is present.

    :param df: pandas DataFrame
    :return: DataFrame with `elapsed_time_s` column
    """

    if (
        "elapsed_time" in df.columns and "elapsed_time_s" not in df.columns
    ):  # If legacy column present and new column missing
        df["elapsed_time_s"] = df["elapsed_time"]  # Copy legacy elapsed_time into the new elapsed_time_s column
        df.drop(columns=["elapsed_time"], inplace=True)  # Remove the legacy elapsed_time column to avoid duplication
    return df  # Return DataFrame with normalized elapsed time column


def load_existing_results(csv_out):
    """
    Load an existing consolidated CSV if present, returning an empty
    DataFrame on error or when file is missing.

    :param csv_out: path to consolidated CSV
    :return: pandas.DataFrame
    """

    if os.path.exists(csv_out):  # If the consolidated CSV exists
        try:  # Try to read the file into a DataFrame
            df = pd.read_csv(csv_out, dtype=object)  # Read CSV preserving types as object
            df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names
            return df  # Return the DataFrame
        except Exception:  # On any read error
            return pd.DataFrame()  # Return empty DataFrame as a fallback
    return pd.DataFrame()  # File not present — return empty DataFrame


def merge_replace_existing(df_existing, df_new):
    """
    Merge `df_new` into `df_existing` using replace-by-`dataset_path`
    semantics: any existing rows whose `dataset_path` appears in `df_new`
    are removed before appending `df_new`.

    :param df_existing: existing DataFrame (may be empty)
    :param df_new: incoming DataFrame with new rows
    :return: merged DataFrame
    """

    if (
        not df_existing.empty and "dataset_path" in df_new.columns
    ):  # If existing rows present and incoming rows include dataset_path
        new_paths = df_new["dataset_path"].unique().tolist()  # Compute unique dataset paths from incoming rows
        df_existing = df_existing[
            ~df_existing["dataset_path"].isin(new_paths)
        ]  # Remove any existing rows that match those paths
    df_combined = pd.concat(
        [df_existing, df_new], ignore_index=True, sort=False
    )  # Concatenate remaining existing rows with new rows
    return df_combined  # Return the merged DataFrame


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

        elif system == "Darwin":  # MacOS: use sysctl
            out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])  # Run sysctl
            cpu_model = out.decode().strip()  # Extract model string

        else:  # Unsupported OS
            cpu_model = "Unknown"  # Fallback

    except Exception:  # If any method fails
        cpu_model = "Unknown"  # Fallback on failure

    cores = psutil.cpu_count(logical=False) if psutil else None  # Physical core count
    ram_gb = round(psutil.virtual_memory().total / (1024**3), 1) if psutil else None  # Total RAM in GB
    os_name = f"{platform.system()} {platform.release()}"  # OS name + version

    return {  # Build final dictionary
        "cpu_model": cpu_model,  # CPU model string
        "cores": cores,  # Physical cores
        "ram_gb": ram_gb,  # RAM in gigabytes
        "os": os_name,  # Operating system
    }


def populate_hardware_column(df, column_name="hardware"):
    """
    Populate `df[column_name]` with a readable hardware description built from
    `get_hardware_specifications()`. On failure the column will be set to None.

    :param df: pandas.DataFrame to modify in-place
    :param column_name: Name of the column to set (default: "hardware")
    :return: The modified DataFrame
    """

    try:  # Try to fetch hardware specifications
        hardware_specs = get_hardware_specifications()  # Get system specs
        hardware_str = (  # Build readable hardware string
            f"{hardware_specs.get('cpu_model','Unknown')} | Cores: {hardware_specs.get('cores', 'N/A')}"
            f" | RAM: {hardware_specs.get('ram_gb', 'N/A')} GB | OS: {hardware_specs.get('os','Unknown')}"
        )
        df[column_name] = hardware_str  # Set the hardware column
    except Exception:  # On any failure
        df[column_name] = None  # Set hardware column to None

    return df  # Return the modified DataFrame


def ensure_expected_columns(df_combined, columns):
    """
    Ensure the expected columns exist on the combined DataFrame; add
    missing columns with None values.

    :param df_combined: pandas.DataFrame
    :param columns: list of expected column names
    :return: DataFrame with ensured columns
    """

    for column in columns:  # Iterate over expected column names
        if column not in df_combined.columns:  # If the column is missing from the DataFrame
            df_combined[column] = None  # Add the missing column and fill with None
    return df_combined  # Return DataFrame with ensured columns


def run_index_sort(val):
    """
    Convert run_index values into sortable numeric keys where "best"
    sorts before numeric indices.

    :param val: run_index value (string or numeric)
    :return: numeric sort key
    """

    try:  # Try to normalize and parse the run index
        s = str(val).strip()  # Convert the value to string and strip whitespace
        if s.lower() == "best":  # If value is the literal 'best'
            return -1  # Force 'best' to sort before numeric indices
        return int(float(s))  # Convert numeric-like strings to integer for sorting
    except Exception:  # On any parsing error
        return 10**9  # Use a very large number to push malformed values to the end


def sort_run_index_first(df_combined):
    """
    Sort by `dataset`, `dataset_path` and a numeric-coded `run_index`
    where the string "best" is forced to come before numeric runs.

    :param df_combined: pandas.DataFrame
    :return: sorted DataFrame
    """

    df_combined["run_index_sort"] = df_combined["run_index"].apply(run_index_sort)  # Create temporary numeric sort key
    df_combined.sort_values(
        by=["dataset", "dataset_path", "run_index_sort"], inplace=True, ascending=[True, True, True]
    )  # Sort by dataset, path, then run order
    df_combined.drop(columns=["run_index_sort"], inplace=True)  # Remove temporary sort key column
    return df_combined  # Return the sorted DataFrame


def write_consolidated_csv(rows, output_dir):
    """
    Write the consolidated GA results rows to a CSV file inside `output_dir`.

    :param rows: List of dictionaries representing rows.
    :param output_dir: Directory where `Genetic_Algorithm_Results.csv` will be saved.
    :return: None
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Writing consolidated Genetic Algorithm results CSV.{Style.RESET_ALL}"
    )  # Output the verbose message

    try:  # Try to write the consolidated CSV
        os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

        df_new = pd.DataFrame(rows)  # Create DataFrame from provided rows

        df_new = normalize_elapsed_column_df(df_new)  # Normalize legacy elapsed_time to elapsed_time_s in new rows

        csv_out = os.path.join(output_dir, "Genetic_Algorithm_Results.csv")  # Build path for consolidated CSV

        if os.path.exists(csv_out):
            df_existing = pd.read_csv(csv_out, dtype=str)
            if "timestamp" not in df_existing.columns:
                mtime = os.path.getmtime(csv_out)
                back_ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d_%H_%M_%S")
                df_existing["timestamp"] = back_ts
            for c in GA_RESULTS_CSV_COLUMNS:
                if c not in df_existing.columns:
                    df_existing[c] = None
            df_combined = pd.concat([df_existing[GA_RESULTS_CSV_COLUMNS], df_new], ignore_index=True, sort=False)
            try:
                df_combined["timestamp_dt"] = pd.to_datetime(df_combined["timestamp"], format="%Y-%m-%d_%H_%M_%S", errors="coerce")
                df_combined = df_combined.sort_values(by="timestamp_dt", ascending=False)
                df_combined = df_combined.drop(columns=["timestamp_dt"])
            except Exception:
                df_combined = df_combined.sort_values(by="timestamp", ascending=False)
            df_out = df_combined.reset_index(drop=True)
        else:
            df_out = df_new

        df_out = populate_hardware_column(df_out, column_name="hardware")  # Populate hardware column using system specs

        df_out = ensure_expected_columns(df_out, GA_RESULTS_CSV_COLUMNS)  # Add any missing expected columns with None values

        df_out = df_out[GA_RESULTS_CSV_COLUMNS]  # Reorder columns into the canonical order

        for col in df_out.columns:  # Iterate over all columns
            col_l = col.lower()  # Lowercase column name for case-insensitive verification
            if "time" in col_l or "elapsed" in col_l or col in ("hardware", "timestamp"):  # Skip time-related and special columns
                df_out[col] = df_out[col].apply(lambda v: int(v) if pd.notnull(v) and str(v).isdigit() else v)  # Convert time-related columns to int if possible
            try:  # Try to truncate values in the column
                df_out[col] = df_out[col].apply(lambda v: truncate_value(v) if pd.notnull(v) else v)  # Truncate numeric values
            except Exception:  # On any error
                pass  # Ignore errors and continue

        df_out.to_csv(csv_out, index=False, encoding="utf-8")  # Persist the consolidated CSV to disk
        print(
            f"\n{BackgroundColors.GREEN}Genetic Algorithm consolidated results saved to {BackgroundColors.CYAN}{csv_out}{Style.RESET_ALL}"
        )  # Notify user of success
    except Exception as e:
        print(
            f"{BackgroundColors.RED}Failed to write consolidated GA CSV: {str(e)}{Style.RESET_ALL}"
        )  # Print failure message with exception


def determine_best_features_and_ranking(best_ind, feature_names, csv_path):
    """
    Determine the selected feature names for a binary GA individual and
    extract RFE rankings from existing results.

    :param best_ind: Sequence (list/array) representing the GA individual (0/1 mask).
    :param feature_names: List of feature names aligned with `best_ind`.
    :param csv_path: Path to the dataset CSV (used to locate RFE summary files).
    :return: Tuple `(best_feats, ranking)` where `best_feats` is a list of selected
             feature names and `ranking` is the RFE ranking dictionary (may be empty).
    """

    # Build list of features selected by the binary mask
    best_feats = [f for f, bit in zip(feature_names if feature_names is not None else [], best_ind) if bit == 1]
    # Try to extract an existing RFE ranking for additional metadata
    ranking = extract_rfe_ranking(csv_path)
    # Verbose output for user
    print(f"\n{BackgroundColors.GREEN}Best features subset found: {BackgroundColors.CYAN}{best_feats}{Style.RESET_ALL}")
    return best_feats, ranking


def determine_rf_metrics(metrics_in):
    """
    Normalize a metrics tuple/sequence to the expected RF-metrics slice used
    downstream by `save_results`. The consolidated CSV expects up to 12
    metric values (cv + test metrics); this helper ensures the returned
    value is either a 12-element-like slice or `None` when not available.

    :param metrics_in: Metrics sequence or None.
    :return: Sliced metrics (first 12 elements) or None.
    """

    if metrics_in is not None:
        # Only keep first 12 values if available (cv metrics + test metrics)
        return metrics_in[:12] if len(metrics_in) >= 12 else None
    return None


def maybe_evaluate_on_test(rf_m, best_ind_local, X, y, X_test, y_test):
    """
    If RF metrics are not provided, optionally evaluate the best individual on
    the provided test set to produce test metrics.

    :param rf_m: Existing RF metrics (may be None).
    :param best_ind_local: GA individual (binary mask) to evaluate.
    :param X: Full feature matrix used for training (numpy array or DataFrame).
    :param y: Training labels.
    :param X_test: Optional test feature matrix.
    :param y_test: Optional test labels.
    :return: RF metrics tuple (possibly produced by `evaluate_individual`) or the original `rf_m`.
    """

    # Only perform evaluation if metrics are missing and a test set is available
    if rf_m is None and X_test is not None and y_test is not None:
        return evaluate_individual(best_ind_local, X, y, X_test, y_test)
    return rf_m


def prepare_output_paths_and_base(csv_path, rf_metrics, best_pop_size, n_generations, rfe_ranking, y, y_test, cxpb, mutpb):
    """
    Prepare filesystem paths and base CSV row metadata for the GA run.

    This helper computes train/test counts and fraction, extracts a base
    elapsed time from provided RF metrics, constructs the canonical base row
    used in the consolidated CSV and creates a timestamped models directory
    and filenames for the model, scaler, features and params artifacts.

    :param csv_path: Path to the original dataset CSV file.
    :param rf_metrics: RF metrics tuple (used to extract an elapsed time if present).
    :param best_pop_size: Population size that produced the best result.
    :param n_generations: Number of generations used by GA (for metadata).
    :param rfe_ranking: RFE ranking dictionary to include in base row.
    :param y: Training labels (used to compute `n_train`).
    :param y_test: Optional test labels (used to compute `n_test`).
    :param cxpb: Crossover probability used (included in base row).
    :param mutpb: Mutation probability used (included in base row).
    :return: Tuple containing computed values and prepared paths:
             (n_train, n_test, test_frac, elapsed_base, base_row,
              models_dir, ts, base_name, model_path, scaler_path,
              features_path, params_path)
    """

    # Compute counts and test fraction
    n_train_local = len(y) if y is not None else None
    n_test_local = len(y_test) if y_test is not None else None
    test_frac_local = None
    if n_train_local is not None and n_test_local is not None and (n_train_local + n_test_local) > 0:
        test_frac_local = float(n_test_local) / float(n_train_local + n_test_local)

    # Extract elapsed seconds from RF metrics (if present)
    elapsed_base = extract_elapsed_from_metrics(rf_metrics)
    # Build the canonical base row used by consolidated CSV
    base_row_local = build_base_row(
        csv_path,
        best_pop_size,
        n_generations,
        n_train_local,
        n_test_local,
        test_frac_local,
        rfe_ranking,
        elapsed_time_s=elapsed_base,
        cxpb=cxpb,
        mutpb=mutpb,
    )

    # Prepare model artifact directory and filenames
    models_dir_local = f"{os.path.dirname(csv_path)}/Feature_Analysis/Genetic_Algorithm/Models/"
    os.makedirs(models_dir_local, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    base_name_local = re.sub(r'[^A-Za-z0-9_.-]+', '_', os.path.splitext(os.path.basename(csv_path))[0])
    model_path_local = os.path.join(models_dir_local, f"GA-{base_name_local}-{ts}-model.joblib")
    scaler_path_local = os.path.join(models_dir_local, f"GA-{base_name_local}-{ts}-scaler.joblib")
    features_path_local = os.path.join(models_dir_local, f"GA-{base_name_local}-{ts}-features.json")
    params_path_local = os.path.join(models_dir_local, f"GA-{base_name_local}-{ts}-params.json")

    return (
        n_train_local,
        n_test_local,
        test_frac_local,
        elapsed_base,
        base_row_local,
        models_dir_local,
        ts,
        base_name_local,
        model_path_local,
        scaler_path_local,
        features_path_local,
        params_path_local,
    )


def train_and_save_final_model(best_feats_local, X, y, feature_names, X_test, model_path_local, scaler_path_local, features_path_local, params_path_local):
    """
    Train a RandomForestClassifier on the selected feature subset, persist the
    trained model, scaler, selected-features list and model parameters.

    :param best_feats_local: List of selected feature names.
    :param X: Full feature matrix (DataFrame or ndarray).
    :param y: Training labels.
    :param feature_names: List of all feature names corresponding to columns in `X`.
    :param X_test: Optional test feature matrix (used to select test columns consistently).
    :param model_path_local: Path where the trained model `.joblib` will be saved.
    :param scaler_path_local: Path where the fitted scaler `.joblib` will be saved.
    :param features_path_local: Path where the JSON file with `best_feats_local` will be saved.
    :param params_path_local: Path where the model `get_params()` JSON will be saved.
    :return: Tuple `(model_local, model_params_local, training_time_local, X_test_selected_local)`.
    """

    df_features_local = prepare_feature_dataframe(X, feature_names)
    scaler_local = StandardScaler()
    X_scaled_local = scaler_local.fit_transform(df_features_local.values)
    sel_indices_local = [i for i, f in enumerate(feature_names) if f in best_feats_local]
    X_final_local = X_scaled_local[:, sel_indices_local] if sel_indices_local else X_scaled_local
    X_test_selected_local = X_test[:, sel_indices_local] if sel_indices_local and X_test is not None else X_test
    model_local = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=N_JOBS)
    start_train_local = time.time()
    model_local.fit(X_final_local, y)
    training_time_local = time.time() - start_train_local

    dump(model_local, model_path_local)
    dump(scaler_local, scaler_path_local)
    with open(features_path_local, "w", encoding="utf-8") as fh:
        json.dump(best_feats_local, fh)
    model_params_local = model_local.get_params()
    with open(params_path_local, "w", encoding="utf-8") as ph:
        json.dump(model_params_local, ph, default=str)

    print(f"{BackgroundColors.GREEN}Saved final model to {BackgroundColors.CYAN}{model_path_local}{Style.RESET_ALL}")
    print(f"{BackgroundColors.GREEN}Saved scaler to {BackgroundColors.CYAN}{scaler_path_local}{Style.RESET_ALL}")
    print(f"{BackgroundColors.GREEN}Saved params to {BackgroundColors.CYAN}{params_path_local}{Style.RESET_ALL}")

    return model_local, model_params_local, training_time_local, X_test_selected_local


def evaluate_final_on_test(model_local, X_test_selected_local, y_test):
    """
    Evaluate a trained classifier on the provided test set and compute a
    standard set of metrics (accuracy, precision, recall, f1, fpr, fnr,
    testing_time_seconds).

    :param model_local: Trained classifier with a `predict` method.
    :param X_test_selected_local: Test features selected to match training subset.
    :param y_test: True test labels.
    :return: Tuple `(metrics_tuple, testing_time_seconds)` where `metrics_tuple` is
             `(acc, prec, rec, f1, fpr, fnr, testing_time)` or a tuple of Nones on error.
    """

    eval_m = None
    testing_time_local = None
    try:
        start_test_local = time.time()
        y_pred_local = model_local.predict(X_test_selected_local)
        acc_local = accuracy_score(y_test, y_pred_local)
        prec_local = precision_score(y_test, y_pred_local, average="weighted", zero_division=0)
        rec_local = recall_score(y_test, y_pred_local, average="weighted", zero_division=0)
        f1_local = f1_score(y_test, y_pred_local, average="weighted", zero_division=0)
        if len(np.unique(y_test)) == 2:
            cm_local = confusion_matrix(y_test, y_pred_local)
            if cm_local.shape == (2, 2):
                tn_local, fp_local, fn_local, tp_local = cm_local.ravel()
                fpr_local = fp_local / (fp_local + tn_local) if (fp_local + tn_local) > 0 else 0
                fnr_local = fn_local / (fn_local + tp_local) if (fn_local + tp_local) > 0 else 0
            else:
                total_local = cm_local.sum() if cm_local.size > 0 else 1
                fpr_local = float(cm_local.sum() - np.trace(cm_local)) / float(total_local) if total_local > 0 else 0
                fnr_local = fpr_local
        else:
            cm_local = confusion_matrix(y_test, y_pred_local)
            supports_local = cm_local.sum(axis=1)
            fprs_local = []
            fnrs_local = []
            for i_local in range(cm_local.shape[0]):
                tp_l = cm_local[i_local, i_local]
                fn_l = cm_local[i_local, :].sum() - tp_l
                fp_l = cm_local[:, i_local].sum() - tp_l
                tn_l = cm_local.sum() - (tp_l + fp_l + fn_l)
                denom_fnr_l = (tp_l + fn_l) if (tp_l + fn_l) > 0 else 1
                denom_fpr_l = (fp_l + tn_l) if (fp_l + tn_l) > 0 else 1
                fnr_i_l = fn_l / denom_fnr_l
                fpr_i_l = fp_l / denom_fpr_l
                fprs_local.append((fpr_i_l, supports_local[i_local]))
                fnrs_local.append((fnr_i_l, supports_local[i_local]))
            total_support_local = float(supports_local.sum()) if supports_local.sum() > 0 else 1.0
            fpr_local = float(sum(v * s for v, s in fprs_local) / total_support_local)
            fnr_local = float(sum(v * s for v, s in fnrs_local) / total_support_local)
        testing_time_local = time.time() - start_test_local
        eval_m = (acc_local, prec_local, rec_local, f1_local, fpr_local, fnr_local, testing_time_local)
    except Exception:
        eval_m = (None, None, None, None, None, None, None)
    return eval_m, testing_time_local


def build_and_write_run_results(
    ts,
    model_local,
    model_params_local,
    training_time_local,
    testing_time_local,
    rf_metrics_local,
    test_frac_local,
    elapsed_run_time,
    n_generations,
    best_pop_size,
    best_features,
    rfe_ranking,
    output_dir,
    csv_path,
):
    """
    Build the consolidated run row dictionary for the best GA individual and
    write it to the consolidated CSV via `write_consolidated_csv`.

    :param ts: Timestamp string used for this run artifacts.
    :param model_local: Trained model instance.
    :param model_params_local: Dictionary of model hyperparameters (`get_params()`).
    :param training_time_local: Training duration in seconds.
    :param testing_time_local: Testing duration in seconds (or None).
    :param rf_metrics_local: RF metrics tuple (cv/test metrics slice) or None.
    :param test_frac_local: Fraction of data used for testing (float) or None.
    :param elapsed_run_time: Total GA elapsed time in seconds (or None).
    :param n_generations: Number of generations used (or None).
    :param best_pop_size: Population size used for best run (or None).
    :param best_features: List of selected feature names.
    :param rfe_ranking: RFE ranking dictionary.
    :param output_dir: Directory where consolidated CSV is stored.
    :param csv_path: Original dataset CSV path (used for dataset name/path metadata).
    :return: None
    """

    cv_method_local = "StratifiedKFold(n_splits=10)" if n_generations is not None or best_pop_size is not None else "train_test_split"

    run_row = {
        "timestamp": ts,
        "tool": "Genetic Algorithm",
        "run_index": "best",
        "model": model_local.__class__.__name__,
        "dataset": os.path.relpath(csv_path),
        "hyperparameters": json.dumps(model_params_local, default=str) if model_params_local is not None else None,
        "cv_method": cv_method_local,
        "train_test_split": f"{1-test_frac_local:.0%}/{test_frac_local:.0%}" if test_frac_local is not None else "80/20",
        "scaling": "StandardScaler",
        "cv_accuracy": truncate_value(rf_metrics_local[0]) if rf_metrics_local and len(rf_metrics_local) > 0 else None,
        "cv_precision": truncate_value(rf_metrics_local[1]) if rf_metrics_local and len(rf_metrics_local) > 1 else None,
        "cv_recall": truncate_value(rf_metrics_local[2]) if rf_metrics_local and len(rf_metrics_local) > 2 else None,
        "cv_f1_score": truncate_value(rf_metrics_local[3]) if rf_metrics_local and len(rf_metrics_local) > 3 else None,
        "cv_fpr": truncate_value(rf_metrics_local[4]) if rf_metrics_local and len(rf_metrics_local) > 4 else None,
        "cv_fnr": truncate_value(rf_metrics_local[5]) if rf_metrics_local and len(rf_metrics_local) > 5 else None,
        "test_accuracy": truncate_value(rf_metrics_local[6]) if rf_metrics_local and len(rf_metrics_local) > 6 else None,
        "test_precision": truncate_value(rf_metrics_local[7]) if rf_metrics_local and len(rf_metrics_local) > 7 else None,
        "test_recall": truncate_value(rf_metrics_local[8]) if rf_metrics_local and len(rf_metrics_local) > 8 else None,
        "test_f1_score": truncate_value(rf_metrics_local[9]) if rf_metrics_local and len(rf_metrics_local) > 9 else None,
        "test_fpr": truncate_value(rf_metrics_local[10]) if rf_metrics_local and len(rf_metrics_local) > 10 else None,
        "test_fnr": truncate_value(rf_metrics_local[11]) if rf_metrics_local and len(rf_metrics_local) > 11 else None,
        "training_time_s": int(round(training_time_local)) if training_time_local is not None else None,
        "testing_time_s": int(round(testing_time_local)) if testing_time_local is not None else None,
        "elapsed_run_time": int(round(elapsed_run_time)) if elapsed_run_time is not None else None,
        "hardware": json.dumps(get_hardware_specifications()),
        "best_features": json.dumps(best_features),
        "rfe_ranking": json.dumps(rfe_ranking),
    }
    
    write_consolidated_csv([run_row], output_dir)

def save_results(
    best_ind,
    feature_names,
    X,
    y,
    csv_path,
    metrics=None,
    X_test=None,
    y_test=None,
    n_generations=None,
    best_pop_size=None,
    runs_list=None,
    cxpb=None,
    mutpb=None,
    elapsed_run_time=None,
):
    """
    Persist the GA best-result information to disk (consolidated CSV and auxiliary files).

    This function performs the saving responsibilities previously embedded inside
    `save_and_analyze_results`: it determines the selected features, extracts RFE
    rankings, optionally re-evaluates the best individual on a provided test set,
    builds the consolidated CSV rows and writes them to disk.

    :param best_ind: Best individual from the Genetic Algorithm (binary mask/list).
    :param feature_names: List of feature names corresponding to bits in `best_ind`.
    :param X: Feature set (DataFrame or numpy array) used during GA/training.
    :param y: Target variable (Series or array) used during GA/training.
    :param csv_path: Path to the original CSV file for saving outputs.
    :param metrics: Optional precomputed metrics tuple for the best individual.
    :param X_test: Optional test features to evaluate the best individual if `metrics` is None.
    :param y_test: Optional test labels to evaluate the best individual if `metrics` is None.
    :param n_generations: Number of GA generations used (for metadata only).
    :param best_pop_size: Population size that yielded the best result (for metadata only).
    :param runs_list: Optional list of per-run results (each a dict with keys 'metrics','best_features' or 'best_ind').
    :param cxpb: Crossover probability used in the GA (for metadata only).
    :param mutpb: Mutation probability used in the GA (for metadata only).
    :param elapsed_run_time: Optional total elapsed time for the GA run (for metadata only
    :return: Dictionary with saved metadata: {
                "best_features": list,
                "rf_metrics": tuple or None,
                "output_dir": str,
                "rfe_ranking": dict,
                "n_train": int or None,
                "n_test": int or None,
                "test_frac": float or None,
                "n_generations": int or None,
                "best_pop_size": int or None,
                "runs_list": list or None
             }
    """

    best_features, rfe_ranking = determine_best_features_and_ranking(best_ind, feature_names, csv_path)
    rf_metrics = determine_rf_metrics(metrics)
    rf_metrics = maybe_evaluate_on_test(rf_metrics, best_ind, X, y, X_test, y_test)
    (
        n_train,
        n_test,
        test_frac,
        elapsed_for_base,
        base_row,
        models_dir,
        ts,
        base_name,
        model_path,
        scaler_path,
        features_path,
        params_path,
    ) = prepare_output_paths_and_base(csv_path, rf_metrics, best_pop_size, n_generations, rfe_ranking, y, y_test, cxpb, mutpb)

    model_local, model_params, training_time_s, X_test_selected = train_and_save_final_model(
        best_features, X, y, feature_names, X_test, model_path, scaler_path, features_path, params_path
    )
    
    eval_metrics, testing_time_s = evaluate_final_on_test(model_local, X_test_selected, y_test)
    
    output_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis/"  # Directory to save outputs
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    build_and_write_run_results(
        ts,
        model_local,
        model_params,
        training_time_s,
        testing_time_s,
        rf_metrics,
        test_frac,
        elapsed_run_time,
        n_generations,
        best_pop_size,
        best_features,
        rfe_ranking,
        output_dir,
        csv_path,
    )

    return {
        "best_features": best_features,
        "rf_metrics": rf_metrics,
        "output_dir": output_dir,
        "rfe_ranking": rfe_ranking,
        "n_train": n_train,
        "n_test": n_test,
        "test_frac": test_frac,
        "n_generations": n_generations,
        "best_pop_size": best_pop_size,
        "runs_list": runs_list,
        "elapsed_time_s": int(round(elapsed_for_base)) if elapsed_for_base is not None else None,
        "model_path": model_path,
        "scaler_path": scaler_path,
        "features_path": features_path,
        "params_path": params_path,
    }


def prepare_feature_dataframe(X, feature_names):
    """
    Ensure features are available as a pandas DataFrame with appropriate column names.

    :param X: Feature matrix (DataFrame or numpy array).
    :param feature_names: Optional iterable of feature names.
    :return: pandas.DataFrame with feature columns.
    """

    if not isinstance(X, pd.DataFrame):  # If X is not a pandas DataFrame
        try:  # Try to create a DataFrame with original feature names
            df_features = pd.DataFrame(X, columns=list(feature_names))  # Create DataFrame with original feature names
        except Exception:  # If creating DataFrame with original feature names fails
            df_features = pd.DataFrame(X)  # Create DataFrame without original feature names
            df_features.columns = [f"feature_{i}" for i in range(df_features.shape[1])]  # Generic feature names
    else:  # If X is already a pandas DataFrame
        df_features = X.copy()  # Use the DataFrame as is

    return df_features  # Return the prepared DataFrame


def analyze_top_features(df, y, top_features, csv_path="."):
    """
    Analyze and visualize the top features.

    :param df: DataFrame containing the features.
    :param y: Target variable.
    :param top_features: List of top feature names.
    :param csv_path: Path to the original CSV file for saving outputs.
    :return: None
    """

    df_analysis = df[top_features].copy()  # Create a copy of the DataFrame with only the top features
    df_analysis["Target"] = pd.Series(y, index=df_analysis.index).astype(
        str
    )  # Add the target variable to the DataFrame

    output_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis"  # Directory to save outputs
    os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

    base_dataset_name = os.path.splitext(os.path.basename(csv_path))[0]  # Base name of the dataset

    summary = df_analysis.groupby("Target")[top_features].agg(
        ["mean", "std"]
    )  # Calculate mean and std for each feature grouped by target
    summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]  # Flatten MultiIndex columns

    summary_csv_path = f"{output_dir}/{base_dataset_name}_feature_summary.csv"  # Path to save the summary CSV
    summary.to_csv(summary_csv_path, encoding="utf-8")  # Save the summary to a CSV file
    print(
        f"\n{BackgroundColors.GREEN}Features summary saved to {BackgroundColors.CYAN}{summary_csv_path}{Style.RESET_ALL}"
    )  # Notify user

    for feature in top_features:  # For each top feature
        plt.figure(figsize=(8, 5))  # Create a new figure
        sns.boxplot(x="Target", y=feature, data=df_analysis, hue="Target", palette="Set2", dodge=False)  # Boxplot
        plt.title(f"Distribution of '{feature}' by class")  # Title
        plt.xlabel("Traffic Type")  # X-axis label
        plt.ylabel(feature)  # Y-axis label
        plt.tight_layout()  # Adjust layout
        plt.savefig(f"{output_dir}/{base_dataset_name}-{safe_filename(feature)}.png")  # Save the plot
        plt.close()  # Close the plot to free memory


def analyze_results(saved_info, X, y, feature_names, csv_path):
    """
    Analyze and visualize results that were previously saved by `save_results`.

    :param saved_info: Dictionary returned from `save_results` (must contain key "best_features").
    :param X: Feature set (DataFrame or numpy array) used during GA/training.
    :param y: Target variable (Series or array) used during GA/training.
    :param feature_names: List of original feature names used to construct the DataFrame.
    :param csv_path: Path to the original CSV file for saving outputs (used by analyzers).
    :return: None
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Analyzing saved results from Genetic Algorithm feature selection.{Style.RESET_ALL}"
    )  # Output the verbose message

    best_features = saved_info.get("best_features", []) if isinstance(saved_info, dict) else []  # Extract best features
    if not best_features:  # Nothing to analyze
        return  # Exit early

    df_features = prepare_feature_dataframe(X, feature_names)  # Prepare DataFrame for analysis

    if not isinstance(y, pd.Series):  # If y is not a pandas Series
        try:  # Try to create a Series with original indices
            y_series = pd.Series(y, index=df_features.index)  # Create Series with original indices
        except Exception:  # If creating Series with original indices fails
            y_series = pd.Series(y)  # Create Series without original indices
    else:  # If y is already a pandas Series
        y_series = (
            y.reindex(df_features.index) if not df_features.index.equals(y.index) else y
        )  # Align indices if necessary

    analyze_top_features(
        df_features, y_series, best_features, csv_path=csv_path
    )  # Analyze and visualize the top features


def run_population_sweep(
    dataset_name,
    csv_path,
    n_generations=200,
    min_pop=20,
    max_pop=20,
    cxpb=0.5,
    mutpb=0.01,
    runs=RUNS,
    progress_bar=None,
):
    """
    Executes a genetic algorithm (GA) for feature selection across multiple population sizes and runs.

    This function performs a "population sweep," testing different population sizes
    to identify the set of features that maximizes classification performance
    (F1-Score) on the training dataset using 10-fold Stratified Cross-Validation.
    For each population size, it runs the GA multiple times to verify for divergence.

    :param dataset_name: Name of the dataset being processed.
    :param csv_path: Path to the CSV dataset.
    :param n_generations: Number of generations to run the GA for each population size.
    :param min_pop: Minimum population size to test.
    :param max_pop: Maximum population size to test.
    :param cxpb: Crossover probability for the GA.
    :param mutpb: Mutation probability for the GA.
    :param runs: Number of runs for each population size.
    :param progress_bar: Optional tqdm progress bar instance to update with progress.
    :return: Dictionary mapping population sizes to their results including runs and divergence.
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Starting population sweep for dataset {BackgroundColors.CYAN}{dataset_name}{BackgroundColors.GREEN} from size {min_pop} to {max_pop}, running {n_generations} generations and {runs} runs each.{Style.RESET_ALL}"
    )

    send_telegram_message(TELEGRAM_BOT, [
        f"Starting population sweep for dataset {dataset_name} from size {min_pop} to {max_pop}"
    ])  # Send start message

    data = prepare_sweep_data(csv_path, dataset_name, min_pop, max_pop, n_generations)  # Prepare dataset
    if data is None:  # If preparation failed
        return {}  # Exit early

    X_train, X_test, y_train, y_test, feature_names = data  # Unpack prepared data

    folds = 10  # Number of CV folds
    progress_state = compute_progress_state(
        min_pop, max_pop, n_generations, runs, progress_bar, folds=folds
    )  # Compute progress state for tracking

    results = {}  # Dictionary to hold results per population size
    for p in range(min_pop, max_pop + 1):  # For each population size
        results[p] = {"runs": [], "avg_metrics": None, "common_features": set()}  # Initialize results entry

    shared_pool = multiprocessing.Pool(processes=CPU_PROCESSES if CPU_PROCESSES else None)  # Create a shared multiprocessing pool for parallel GA runs

    start_run_time = time.time()  # Start timing the entire run process
    for run in range(runs):  # For each run
        for pop_size in range(min_pop, max_pop + 1):  # For each population size
            send_telegram_message(TELEGRAM_BOT, [
                f"Run {run + 1}/{runs} - population size {pop_size}/{max_pop}"
            ])  # Send start message for this run and population size
            start_pop_time = time.time()  # Start timing this population size iteration
            result = run_single_ga_iteration(
                X_train,
                y_train,
                X_test,
                y_test,
                feature_names,
                pop_size,
                n_generations,
                cxpb,
                mutpb,
                run + 1,
                runs,
                dataset_name,
                csv_path,
                max_pop,
                progress_bar,
                progress_state,
                folds,
                shared_pool=shared_pool,
            )  # Run GA iteration
            elapsed_pop_time = time.time() - start_pop_time  # Calculate elapsed time for this population size
            if result:  # If result is valid
                results[pop_size]["runs"].append(result)  # Append result to runs list

            # Send completion message with timing
            send_telegram_message(
                TELEGRAM_BOT,
                f"Completed run {run + 1}/{runs} - population size {pop_size}/{max_pop} in {int(elapsed_pop_time)}s"
            )

    try:  # Close the shared pool after all GA iterations are complete
        shared_pool.close()  # Signal no more work will be submitted
        shared_pool.join()  # Wait for all workers to finish
    except Exception:  # If closing the pool fails (e.g., if it was already closed or if an error occurred)
        pass  # Best-effort cleanup

    best_score, best_result, best_metrics, results = aggregate_sweep_results(
        results, min_pop, max_pop, dataset_name
    )  # Aggregate results and find best

    elapsed_run_time = time.time() - start_run_time  # Calculate elapsed time for the entire run process
    
    if best_result:  # If a best result was found
        best_pop_size, runs_list, common_features = best_result  # Unpack the best result
        print(
            f"\n{BackgroundColors.GREEN}Best population size: {BackgroundColors.CYAN}{best_pop_size}{Style.RESET_ALL}"
        )
        print(
            f"{BackgroundColors.GREEN}Common features across runs: {BackgroundColors.CYAN}{len(common_features)}{Style.RESET_ALL}"
        )
        print_metrics(best_metrics) if VERBOSE else None  # Print metrics if VERBOSE is enabled
        best_run = max(runs_list, key=lambda r: r["metrics"][3])  # Select the run with the best F1-Score
        best_ind = best_run["best_ind"]  # Get the best individual from the best run
        best_metrics = best_run["metrics"]  # Get the metrics from the best run
        saved = save_results(
            best_ind,
            feature_names,
            X_train,
            y_train,
            csv_path,
            metrics=best_metrics,
            X_test=X_test,
            y_test=y_test,
            n_generations=n_generations,
            best_pop_size=best_pop_size,
            runs_list=runs_list,
            cxpb=cxpb,
            mutpb=mutpb,
            elapsed_run_time=elapsed_run_time,
        )  # Save the best results
        analyze_results(saved, X_train, y_train, feature_names, csv_path)  # Analyze the saved results
    else:  # If no valid result was found
        print(f"{BackgroundColors.RED}No valid results found during the sweep.{Style.RESET_ALL}")

    send_telegram_message(TELEGRAM_BOT, [f"Population sweep completed for {dataset_name}"])  # Send completion message

    return results  # Return the results dictionary


def to_seconds(obj):
    """
    Converts various time-like objects to seconds.
    
    :param obj: The object to convert (can be int, float, timedelta, datetime, etc.)
    :return: The equivalent time in seconds as a float, or None if conversion fails
    """
    
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


def calculate_execution_time(start_time, finish_time=None):
    """
    Calculates the execution time and returns a human-readable string.

    Accepts either:
    - Two datetimes/timedeltas: `calculate_execution_time(start, finish)`
    - A single timedelta or numeric seconds: `calculate_execution_time(delta)`
    - Two numeric timestamps (seconds): `calculate_execution_time(start_s, finish_s)`

    Returns a string like "1h 2m 3s".
    """

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


def play_sound():
    """
    Plays a sound when the program finishes and skips if the operating system is Windows.
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

    :return: None
    """

    parser = argparse.ArgumentParser(
        description="Run Genetic Algorithm feature selection and optionally load existing exported models."
    )
    parser.add_argument(
        "--skip-train-if-model-exists",
        dest="skip_train",
        action="store_true",
        help="If set, do not retrain; load existing exported artifacts and evaluate.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Enable verbose output during the run.",
    )
    parser.add_argument(
        "--csv",
        dest="csv",
        type=str,
        default=None,
        help="Optional: path to dataset CSV to analyze. If omitted, uses the default in main().",
    )
    args = parser.parse_args()

    global SKIP_TRAIN_IF_MODEL_EXISTS, VERBOSE
    SKIP_TRAIN_IF_MODEL_EXISTS = bool(args.skip_train)
    VERBOSE = bool(args.verbose)
    csv_path = args.csv if args.csv else "./Datasets/CICDDoS2019/01-12/DrDoS_DNS.csv"

    print(
        f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Genetic Algorithm Feature Selection{BackgroundColors.GREEN} program!{Style.RESET_ALL}",
        end="\n\n",
    )
    start_time = datetime.datetime.now()

    setup_telegram_bot()  # Setup Telegram bot if configured

    # Example GA params (could be extended to CLI)
    n_generations = 200
    min_pop = 20
    max_pop = 20
    cxpb = 0.5
    mutpb = 0.01
    runs = RUNS
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    send_telegram_message(TELEGRAM_BOT, [f"Starting Genetic Algorithm Feature Selection for {dataset_name} at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"])  # Send start message

    if SKIP_TRAIN_IF_MODEL_EXISTS:  # Verify if model loading is enabled
        should_return = handle_skip_train_if_model_exists(csv_path)  # Execute model loading workflow
        if should_return:  # Verify if artifacts were loaded successfully
            return  # Exit early to skip GA training

    # Run the GA pipeline as usual
    sweep_results = run_population_sweep(
        dataset_name,
        csv_path,
        n_generations=n_generations,
        min_pop=min_pop,
        max_pop=max_pop,
        cxpb=cxpb,
        mutpb=mutpb,
        runs=runs,
        progress_bar=None,
    )

    if VERBOSE and sweep_results:
        print(
            f"\n{BackgroundColors.GREEN}Detailed sweep results by population size:{Style.RESET_ALL}"
        )
        for pop_size, features in sweep_results.items():
            print(
                f"  Pop {pop_size}: {len(features)} features -> {features}"
            )

    finish_time = datetime.datetime.now()
    print(
        f"{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
    )
    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
    )
    
    send_telegram_message(TELEGRAM_BOT, [f"Genetic Algorithm feature selection completed for {dataset_name}. Execution time: {calculate_execution_time(start_time, finish_time)}"])  # Send completion message
    
    atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None


if __name__ == "__main__":
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """

    main()  # Call the main function
