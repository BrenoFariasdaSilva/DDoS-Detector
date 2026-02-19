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
    - Fully externally configurable via config files, CLI args, or programmatic calls

Configuration:
    ALL execution parameters are externally configurable through:
    1. Configuration file (config.yaml or .env) - PRIMARY method
    2. Command-line arguments - Override config file values
    3. Programmatic execution - Pass config dictionary to run_genetic_algorithm()
    
    Configuration priority: CLI args > config file > defaults
    See config.yaml.example and CONFIGURATION_GUIDE.md for details.

Usage:
    CLI with configuration file:
        python genetic_algorithm.py --config config.yaml --csv-path ./data.csv
    
    CLI with arguments only:
        python genetic_algorithm.py --runs 10 --n-generations 300 --verbose
    
    Programmatic execution:
        from genetic_algorithm import run_genetic_algorithm, get_default_config
        config = get_default_config()
        config["execution"]["runs"] = 10
        results = run_genetic_algorithm(config=config, csv_path="./data.csv")

Outputs:
    - Feature_Analysis/Genetic_Algorithm_Results.csv (consolidated results)
    - Per-dataset feature summaries and boxplots in Feature_Analysis/
    - Optional convergence plots and model artifacts

Dependencies:
    Python >= 3.9 and: pandas, numpy, scikit-learn, deap, tqdm, matplotlib,
    seaborn, colorama, pyyaml. Optional: psutil, python-telegram-bot, python-dotenv.
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
import yaml  # For system-specific parameters and functions
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
from telegram_bot import TelegramBot, send_exception_via_telegram, send_telegram_message, setup_global_exception_hook  # For Telegram notifications
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


# Global Configuration (initialized from config file/CLI/defaults):
CONFIG: dict[str, Any] = {}  # Global configuration dictionary (initialized in main/run_genetic_algorithm)

# Runtime State Variables (DO NOT configure these):
CPU_PROCESSES = None  # Number of CPU processes for multiprocessing (dynamically updated by monitor)
GA_GENERATIONS_COMPLETED = 0  # Updated by GA loop to inform monitor when some generations have run
RESOURCE_MONITOR_LAST_FILE = None  # Path of the file currently being processed (monitor uses this)
RESOURCE_MONITOR_UPDATED_FOR_CURRENT_FILE = False  # Whether monitor already applied an update for the current file

# Telegram Bot Setup:
TELEGRAM_BOT = None  # Global Telegram bot instance (initialized in setup_telegram_bot)

# Logger Setup:
logger = None  # Global logger instance (initialized in initialize_logger)

# Fitness Cache:
fitness_cache = {}  # Cache for fitness results to avoid re-evaluating same feature masks
fitness_cache_lock = threading.Lock()  # Thread lock for fitness cache

# Thread Locks for Global Variables:
global_state_lock = threading.Lock()  # Lock for CPU_PROCESSES, GA_GENERATIONS_COMPLETED, etc.
csv_write_lock = threading.Lock()  # Lock for CSV write operations to prevent race conditions

# Functions Definitions:


setup_global_exception_hook()  # Set up global exception hook to catch unhandled exceptions and send Telegram alerts

def parse_cli_args():
    """
    Parse command-line arguments for genetic algorithm configuration.

    :return: Namespace object containing parsed arguments
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        parser = argparse.ArgumentParser(
            description="Genetic Algorithm for Feature Selection",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )  # Create argument parser

        parser.add_argument("--csv-path", type=str, help="Path to CSV dataset file or directory")  # CSV path argument
        parser.add_argument("--files-to-ignore", type=str, nargs="*", help="List of files to ignore")  # Files to ignore
        parser.add_argument("--test-size", type=float, help="Test set size (0.0-1.0)")  # Test size argument

        parser.add_argument("--verbose", action="store_true", help="Enable verbose output")  # Verbose flag
        parser.add_argument("--runs", type=int, help="Number of GA runs")  # Number of runs
        parser.add_argument("--skip-train", action="store_true", help="Skip training if model exists")  # Skip training flag
        parser.add_argument("--no-resume", action="store_true", help="Disable progress resumption")  # Disable resume flag
        parser.add_argument("--no-sound", action="store_true", help="Disable sound notification")  # Disable sound flag

        parser.add_argument("--n-generations", type=int, help="Number of GA generations")  # Generations argument
        parser.add_argument("--min-pop", type=int, help="Minimum population size")  # Minimum population
        parser.add_argument("--max-pop", type=int, help="Maximum population size")  # Maximum population
        parser.add_argument("--cxpb", type=float, help="Crossover probability")  # Crossover probability
        parser.add_argument("--mutpb", type=float, help="Mutation probability")  # Mutation probability

        parser.add_argument("--cv-folds", type=int, help="Number of cross-validation folds")  # CV folds argument

        parser.add_argument("--early-stop-acc", type=float, help="Early stop accuracy threshold")  # Early stop threshold
        parser.add_argument("--early-stop-folds", type=int, help="Early stop folds")  # Early stop folds
        parser.add_argument("--early-stop-gens", type=int, help="Early stop generations")  # Early stop generations

        parser.add_argument("--n-jobs", type=int, help="Number of parallel jobs (-1 for all)")  # N jobs argument
        parser.add_argument("--cpu-processes", type=int, help="Initial CPU processes")  # CPU processes argument

        parser.add_argument(
            "--telegram-progress-pct",
            type=int,
            help="Send Telegram progress updates every N percent of generations (default from config)",
        )  # Telegram progress percent

        parser.add_argument("--no-monitor", action="store_true", help="Disable resource monitoring")  # Disable monitor
        parser.add_argument("--monitor-interval", type=int, help="Monitor interval in seconds")  # Monitor interval

        parser.add_argument("--config", type=str, help="Path to configuration file (YAML or .env)")  # Config file path

        return parser.parse_args()  # Parse and return arguments
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def get_default_config():
    """
    Returns the default configuration dictionary for the genetic algorithm.

    :return: Dictionary containing all default configuration parameters
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        return {
        "execution": {
            "verbose": False,  # Set to True to output verbose messages
            "skip_train_if_model_exists": False,  # If True, try loading exported models instead of retraining
            "runs": 5,  # Number of runs for Genetic Algorithm analysis
            "resume_progress": True,  # When True, attempt to resume progress from saved state files
            "progress_save_interval": 10,  # Save progress every N generations
            "play_sound": True,  # Set to True to play a sound when the program finishes
        },
        "telegram": {
            "enabled": True,  # Enable Telegram progress notifications
            "progress_pct": 10,  # Send Telegram updates every N percent of total generations (default 10%)
        },
        "dataset": {
            "files_to_ignore": [],  # List of files to ignore during processing
            "test_size": 0.2,  # Test set size for train-test split
            "min_test_fraction": 0.05,  # Minimum acceptable test fraction
            "max_test_fraction": 0.50,  # Maximum acceptable test fraction
            "remove_zero_variance": True,  # Remove zero-variance features during preprocessing
        },
        "genetic_algorithm": {
            "n_generations": 200,  # Number of generations for GA
            "min_pop": 20,  # Minimum population size
            "max_pop": 20,  # Maximum population size
            "cxpb": 0.5,  # Crossover probability
            "mutpb": 0.01,  # Mutation probability
        },
        "early_stop": {
            "acc_threshold": 0.75,  # Minimum acceptable accuracy for an individual
            "folds": 3,  # Number of folds to verify before early stopping
            "generations": 10,  # Number of generations without improvement before early stop
        },
        "cross_validation": {
            "n_folds": 10,  # Number of cross-validation folds
        },
        "multiprocessing": {
            "n_jobs": -1,  # Number of parallel jobs for GridSearchCV (-1 uses all processors)
            "cpu_processes": None,  # Initial number of worker processes; None -> use all available CPUs
        },
        "resource_monitor": {
            "enabled": True,  # Enable resource monitoring
            "interval_seconds": 30,  # Interval between monitoring cycles in seconds
            "reserve_cpu_frac": 0.15,  # Fraction of CPU reserved from worker allocation
            "reserve_mem_frac": 0.15,  # Fraction of memory reserved from worker allocation
            "min_procs": 1,  # Minimum number of processes allowed
            "max_procs": None,  # Maximum number of processes allowed
            "min_gens_before_update": 10,  # Minimum GA generations before updating workers
            "daemon": True,  # Whether the monitoring thread runs as daemon
        },
        "model": {
            "estimator": "RandomForestClassifier",  # Default estimator for GA fitness evaluation
            "random_state": None,  # Random state for reproducibility (None for non-deterministic)
        },
        "caching": {
            "enabled": True,  # Enable fitness caching
            "pickle_protocol": pickle.HIGHEST_PROTOCOL,  # Pickle protocol to use when saving state
        },
        "progress": {
            "state_dir_name": "ga_progress",  # Subfolder under Feature_Analysis to store progress files
        },
        "export": {
            "results_csv_columns": [  # Columns for the results CSV
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
                "num_features_selected",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1_score",
                "test_fpr",
                "test_fnr",
                "feature_extraction_time_s",
                "training_time_s",
                "testing_time_s",
                "elapsed_run_time",
                "hardware",
                "best_features",
                "rfe_ranking",
            ],
        },
        "sound": {
            "commands": {  # The commands to play a sound for each operating system
                "Darwin": "afplay",
                "Linux": "aplay",
                "Windows": "start",
            },
            "file": "./.assets/Sounds/NotificationSound.wav",  # The path to the sound file
        },
        "paths": {
            "datasets_dir": None,  # Directory containing datasets (None for auto-detect)
            "output_dir": "Feature_Analysis",  # Output directory for results
            "logs_dir": "./Logs",  # Directory for log files
        },
        }  # Return the default configuration dictionary
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_config_file(config_path=None):
    """
    Load configuration from YAML or .env file.

    :param config_path: Path to configuration file (None for auto-detect)
    :return: Dictionary with loaded configuration or empty dict if not found
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        if config_path is None:  # If no path provided, try default locations
            possible_paths = ["config.yaml", "config.yml", ".env"]  # List of possible config file paths
            for path in possible_paths:  # Iterate through possible paths
                if os.path.exists(path):  # Verify if file exists
                    config_path = path  # Use this path
                    break  # Stop searching

        if config_path is None or not os.path.exists(config_path):  # If no config file found
            return {}  # Return empty dictionary

        try:  # Attempt to load the configuration file
            with open(config_path, "r") as f:  # Open configuration file
                if config_path.endswith((".yaml", ".yml")):  # If YAML file
                    return yaml.safe_load(f) or {}  # Load and return YAML content
                elif config_path.endswith(".env"):  # If .env file
                    from dotenv import dotenv_values  # Import dotenv parser
                    return dotenv_values(config_path)  # Load and return .env content
        except Exception as e:  # If loading fails
            print(f"{BackgroundColors.YELLOW}Warning: Failed to load config from {config_path}: {e}{Style.RESET_ALL}")  # Output warning
            return {}  # Return empty dictionary

        return {}  # Return empty dictionary as fallback
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def deep_merge_dicts(base, override):
    """
    Recursively merge override dictionary into base dictionary.

    :param base: Base dictionary
    :param override: Override dictionary
    :return: Merged dictionary
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        result = base.copy()  # Create copy of base dictionary
        for key, value in override.items():  # Iterate through override items
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):  # If both are dicts
                result[key] = deep_merge_dicts(result[key], value)  # Recursively merge
            else:  # Otherwise
                result[key] = value  # Override the value
        return result  # Return merged dictionary
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def merge_configs(defaults, file_config, cli_args):
    """
    Merge configuration from defaults, config file, and CLI arguments.
    Priority: CLI args > config file > defaults.

    :param defaults: Default configuration dictionary
    :param file_config: Configuration loaded from file
    :param cli_args: Parsed command-line arguments
    :return: Merged configuration dictionary
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        config = deep_merge_dicts(defaults, file_config)  # Start with defaults, merge file config

        if cli_args.verbose:  # If verbose flag set
            config["execution"]["verbose"] = True  # Enable verbose mode

        if cli_args.runs is not None:  # If runs specified
            config["execution"]["runs"] = cli_args.runs  # Override runs

        if cli_args.skip_train:  # If skip train flag set
            config["execution"]["skip_train_if_model_exists"] = True  # Enable skip training

        if cli_args.no_resume:  # If no resume flag set
            config["execution"]["resume_progress"] = False  # Disable resume

        if cli_args.no_sound:  # If no sound flag set
            config["execution"]["play_sound"] = False  # Disable sound

        if cli_args.files_to_ignore is not None:  # If files to ignore specified
            config["dataset"]["files_to_ignore"] = cli_args.files_to_ignore  # Override files to ignore

        if cli_args.test_size is not None:  # If test size specified
            config["dataset"]["test_size"] = cli_args.test_size  # Override test size

        if cli_args.n_generations is not None:  # If generations specified
            config["genetic_algorithm"]["n_generations"] = cli_args.n_generations  # Override generations

        if cli_args.min_pop is not None:  # If min population specified
            config["genetic_algorithm"]["min_pop"] = cli_args.min_pop  # Override min population

        if cli_args.max_pop is not None:  # If max population specified
            config["genetic_algorithm"]["max_pop"] = cli_args.max_pop  # Override max population

        if cli_args.cxpb is not None:  # If crossover probability specified
            config["genetic_algorithm"]["cxpb"] = cli_args.cxpb  # Override crossover probability

        if cli_args.mutpb is not None:  # If mutation probability specified
            config["genetic_algorithm"]["mutpb"] = cli_args.mutpb  # Override mutation probability

        if cli_args.cv_folds is not None:  # If CV folds specified
            config["cross_validation"]["n_folds"] = cli_args.cv_folds  # Override CV folds

        if cli_args.early_stop_acc is not None:  # If early stop accuracy specified
            config["early_stop"]["acc_threshold"] = cli_args.early_stop_acc  # Override early stop threshold

        if cli_args.early_stop_folds is not None:  # If early stop folds specified
            config["early_stop"]["folds"] = cli_args.early_stop_folds  # Override early stop folds

        if cli_args.early_stop_gens is not None:  # If early stop generations specified
            config["early_stop"]["generations"] = cli_args.early_stop_gens  # Override early stop generations

        if cli_args.n_jobs is not None:  # If n jobs specified
            config["multiprocessing"]["n_jobs"] = cli_args.n_jobs  # Override n jobs

        if cli_args.cpu_processes is not None:  # If CPU processes specified
            config["multiprocessing"]["cpu_processes"] = cli_args.cpu_processes  # Override CPU processes

        if cli_args.no_monitor:  # If no monitor flag set
            config["resource_monitor"]["enabled"] = False  # Disable resource monitor

        if cli_args.monitor_interval is not None:  # If monitor interval specified
            config["resource_monitor"]["interval_seconds"] = cli_args.monitor_interval  # Override monitor interval

        if hasattr(cli_args, "telegram_progress_pct") and cli_args.telegram_progress_pct is not None:
            try:
                config["telegram"]["progress_pct"] = int(cli_args.telegram_progress_pct)
            except Exception:
                pass

        return config  # Return merged configuration
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def initialize_config(config_path=None, cli_args=None):
    """
    Initialize global configuration from defaults, file, and CLI arguments.

    :param config_path: Path to configuration file (None for auto-detect)
    :param cli_args: Parsed CLI arguments (None to parse from sys.argv)
    :return: Merged configuration dictionary
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        defaults = get_default_config()  # Get default configuration
        file_config = load_config_file(config_path)  # Load configuration from file

        if cli_args is None:  # If no CLI args provided
            cli_args = parse_cli_args()  # Parse from command line

        config = merge_configs(defaults, file_config, cli_args)  # Merge all configuration sources

        return config  # Return final configuration
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def initialize_logger(config):
    """
    Initialize the logger based on configuration.

    :param config: Configuration dictionary
    :return: Logger instance
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        logs_dir = config["paths"]["logs_dir"]  # Get logs directory from config
        log_file = f"{logs_dir}/{Path(__file__).stem}.log"  # Construct log file path

        logger_instance = Logger(log_file, clean=True)  # Create logger instance
        sys.stdout = logger_instance  # Redirect stdout to logger
        sys.stderr = logger_instance  # Redirect stderr to logger

        return logger_instance  # Return logger instance
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def verbose_output(true_string="", false_string=""):
    """
    Outputs a message if the VERBOSE constant is set to True.

    :param true_string: The string to be outputted if the VERBOSE constant is set to True.
    :param false_string: The string to be outputted if the VERBOSE constant is set to False.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        global CONFIG  # Access global configuration

        verbose = CONFIG.get("execution", {}).get("verbose", False) if CONFIG else False  # Get verbose setting from config

        if verbose and true_string != "":  # If verbose is True and a true_string was provided
            print(true_string)  # Output the true statement string
        elif false_string != "":  # If a false_string was provided
            print(false_string)  # Output the false statement string
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
            telegram_module.TELEGRAM_DEVICE_INFO = f"{telegram_module.get_local_ip()} - {platform.system()}"  # Set device info string with IP and OS
            telegram_module.RUNNING_CODE = os.path.basename(__file__)  # Set currently running script name
        except Exception as e:
            print(f"{BackgroundColors.RED}Failed to initialize Telegram bot: {str(e)}{Style.RESET_ALL}")
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


def load_exported_artifacts(csv_path):
    """
    Attempt to locate and load latest exported model, scaler and features for csv_path.

    :param csv_path: original dataset path used to name exported artifacts
    :return: (model, scaler, features, params) or None if not found
    """
    
    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Attempting to load exported model artifacts for dataset: {BackgroundColors.CYAN}{csv_path}{Style.RESET_ALL}"
        )  # Output the verbose message
        
        models_dir = os.path.join(os.path.dirname(csv_path), "Feature_Analysis", "Genetic_Algorithm", "Models")  # Construct path to models directory
        if not os.path.isdir(models_dir):  # Verify if models directory exists
            return None  # Return None if directory doesn't exist
        base_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(os.path.basename(csv_path))[0])  # Sanitize dataset basename for filename
        pattern = os.path.join(models_dir, f"GA-{base_name}-*-model.joblib")  # Build glob pattern to find model files
        candidates = glob.glob(pattern)  # Search for matching model files
        if not candidates:  # Verify if any models were found
            return None  # Return None if no models found
        latest_model = max(candidates, key=os.path.getmtime)  # Select most recently modified model file
        scaler_path = latest_model.replace("-model.joblib", "-scaler.joblib")  # Derive scaler file path from model path
        features_path = latest_model.replace("-model.joblib", "-features.json")  # Derive features file path from model path
        params_path = latest_model.replace("-model.joblib", "-params.json")  # Derive params file path from model path
        if not (os.path.exists(scaler_path) and os.path.exists(features_path)):  # Verify both scaler and features files exist
            return None  # Return None if required files are missing
        try:  # Attempt to load all model artifacts
            model = load(latest_model)  # Load trained model from joblib file
            scaler = load(scaler_path)  # Load fitted scaler from joblib file
            with open(features_path, "r", encoding="utf-8") as fh:  # Open features JSON file
                features = json.load(fh)  # Parse selected features list from JSON
            params = None  # Initialize params as None
            if os.path.exists(params_path):  # Verify if params file exists
                try:  # Attempt to load params
                    with open(params_path, "r", encoding="utf-8") as ph:  # Open params JSON file
                        params = json.load(ph)  # Parse model parameters from JSON
                except Exception:  # Ignore params loading errors
                    params = None  # Keep params as None on error
            return model, scaler, features, params  # Return all loaded artifacts as tuple
        except Exception:  # Catch any loading errors
            return None  # Return None if artifact loading fails
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def get_files_to_process(directory_path, file_extension=".csv"):
    """
    Get all of the specified files in a directory (non-recursive).

    :param directory_path: Path to the directory to search
    :param file_extension: File extension to filter (default: .csv)
    :return: List of files with the specified extension
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
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
                ignore and (ignore == filename or ignore == item_path) for ignore in CONFIG["dataset"]["files_to_ignore"]
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

    try:  # Wrap full function logic to ensure production-safe monitoring
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


def get_logical_cpu_count():
    """
    Get logical CPU count, preferring psutil when available.

    :return: integer logical CPU count (>=1)
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        try:  # Try to obtain CPU count via psutil (preferred) or os.cpu_count (fallback)
            return (
                int(psutil.cpu_count(logical=True)) if psutil and psutil.cpu_count() else int(os.cpu_count() or 1)
            )  # Return logical CPU count
        except Exception:  # On any exception while querying CPU count
            return max(1, int(os.cpu_count() or 1))  # Fallback to at least 1 logical CPU
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def compute_reserved_cpus(total_cpus, reserve_cpu_frac):
    """
    Compute how many CPUs to reserve for system and main process.

    :param total_cpus: total logical CPUs
    :param reserve_cpu_frac: fraction to reserve
    :return: reserved CPU count (>=1)
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        try:  # Try to compute reserved CPUs from fraction
            return max(1, int(total_cpus * float(reserve_cpu_frac)))  # Return reserved CPUs
        except Exception:  # If computation fails for any reason
            return 1  # Default to reserving 1 CPU
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def compute_cpu_bound(total_cpus, reserved, min_procs):
    """
    Compute CPU-based upper bound for worker processes.

    :param total_cpus: total logical CPUs
    :param reserved: CPUs reserved for system/main
    :param min_procs: minimum allowed worker processes
    :return: cpu_bound (>= min_procs)
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        return max(min_procs, total_cpus - reserved)  # Compute CPU-based bound
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def compute_memory_bound(reserve_mem_frac):
    """
    Compute memory-based upper bound for worker processes using psutil.

    :param reserve_mem_frac: fraction of memory to keep free
    :return: integer mem_bound or None if not computable
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def compute_optimal_processes(reserve_cpu_frac=0.15, reserve_mem_frac=0.15, min_procs=1, max_procs=None):
    """
    Compute a conservative number of worker processes based on current CPU and
    memory availability.

    Returns an integer number of workers >= min_procs. Uses "psutil" when
    available; falls back to CPU-count heuristics otherwise.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def monitor(
    interval_seconds,
    reserve_cpu_frac,
    reserve_mem_frac,
    min_procs,
    max_procs,
    min_gens_before_update,
):
    """
    Periodically monitor system resources and update CPU_PROCESSES when appropriate.

    :param interval_seconds: Interval between monitoring cycles in seconds.
    :param reserve_cpu_frac: Fraction of CPU reserved from worker allocation.
    :param reserve_mem_frac: Fraction of memory reserved from worker allocation.
    :param min_procs: Minimum number of processes allowed.
    :param max_procs: Maximum number of processes allowed.
    :param min_gens_before_update: Minimum GA generations before updating workers.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        global CPU_PROCESSES, RESOURCE_MONITOR_UPDATED_FOR_CURRENT_FILE, RESOURCE_MONITOR_LAST_FILE

        while True:  # Infinite monitoring loop
            try:  # Verify if update conditions are met
                with global_state_lock:  # Thread-safe access to global flags
                    already_updated = RESOURCE_MONITOR_UPDATED_FOR_CURRENT_FILE  # Whether update already occurred
                    gens_completed = GA_GENERATIONS_COMPLETED  # Number of completed GA generations

                if already_updated:  # Only one update per file
                    time.sleep(1)  # Sleep briefly until new file arrives
                    continue  # Skip computing suggestions until new file

                if gens_completed < int(min_gens_before_update):  # If not enough generations yet
                    time.sleep(1)  # Sleep briefly and re-verify later
                    continue  # Skip computing suggestions until threshold met

            except Exception:  # If flags or generation counters are unavailable
                pass  # Proceed with best-effort execution

            try:  # Attempt to compute suggested worker count
                suggested = compute_optimal_processes(
                    reserve_cpu_frac=reserve_cpu_frac,
                    reserve_mem_frac=reserve_mem_frac,
                    min_procs=min_procs,
                    max_procs=max_procs,
                )  # Compute optimal process count

                if suggested and suggested != CPU_PROCESSES:  # If suggestion differs from current
                    with global_state_lock:  # Thread-safe update
                        CPU_PROCESSES = suggested  # Update global worker count
                        RESOURCE_MONITOR_UPDATED_FOR_CURRENT_FILE = True  # Mark update applied

            except Exception:  # Ignore computation failures
                pass  # Retry on next monitoring cycle

            try:  # Sleep before next monitoring iteration
                time.sleep(max(1, int(interval_seconds)))  # Ensure minimum sleep of 1 second
            except Exception:  # Handle invalid or interrupted sleep
                time.sleep(5)  # Fallback sleep duration
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


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
Start the resource monitoring thread with the specified configuration.

    :param interval_seconds: Interval between monitoring cycles in seconds.
    :param reserve_cpu_frac: Fraction of CPU reserved from worker allocation.
    :param reserve_mem_frac: Fraction of memory reserved from worker allocation.
    :param min_procs: Minimum number of processes allowed.
    :param max_procs: Maximum number of processes allowed.
    :param min_gens_before_update: Minimum GA generations before updating workers.
    :param daemon: Whether the monitoring thread runs as daemon.
    :return: Thread object
    """

    t = threading.Thread(
        target=monitor,
        args=(
            interval_seconds,
            reserve_cpu_frac,
            reserve_mem_frac,
            min_procs,
            max_procs,
            min_gens_before_update,
        ),  # Pass monitoring configuration parameters
        daemon=daemon,
        name="ga-resource-monitor",
    )  # Create the monitoring thread

    t.start()  # Start the monitoring thread

    return t  # Return the Thread object


def start_resource_monitor_safe(*args, **kwargs):
    """
    Safe wrapper to start the resource monitor: swallow any exceptions so
    callers (e.g., "main") don't need to handle psutil or threading issues.

    Usage: "start_resource_monitor_safe()" (calls "start_resource_monitor" with defaults).
    Returns the Thread object when started, or None on failure.
    """

    try:  # Wrap full function logic to ensure production-safe execution
        try:  # Try to start the resource monitor
            return start_resource_monitor(*args, **kwargs)  # Start the resource monitor
        except Exception:  # If any exception occurs
            return None  # Return None
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def signal_new_file(file_path):
    """
    Notify the resource monitor that processing of a new file has started.

    This resets the per-file monitor flag so the monitor will perform one
    sizing update (after "min_gens_before_update" generations) for the
    newly-started file. It also resets "GA_GENERATIONS_COMPLETED" to 0 so
    the monitor waits again for the configured number of generations.

    :param file_path: path of the file that will be processed
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe execution
        global RESOURCE_MONITOR_LAST_FILE, RESOURCE_MONITOR_UPDATED_FOR_CURRENT_FILE, GA_GENERATIONS_COMPLETED  # Use global variables
        try:  # Try to signal the new file
            with global_state_lock:  # Thread-safe global variable updates
                RESOURCE_MONITOR_LAST_FILE = file_path  # Update the last file being processed
                RESOURCE_MONITOR_UPDATED_FOR_CURRENT_FILE = False  # Reset the per-file update flag
                GA_GENERATIONS_COMPLETED = 0  # Reset generations completed for the new file
        except Exception:  # Ignore any errors during signaling
            pass  # Do nothing
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


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
    Update a tqdm "progress_bar" description and postfix consistently.

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

    try:  # Wrap full function logic to ensure production-safe execution
        if progress_bar is None:  # If no progress bar is provided
            return  # Do nothing
        try:  # Try to update the progress bar
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def clear_fitness_cache():
    """
    Clear the fitness cache to prevent memory leaks and cross-dataset contamination.

    :return: None
    """
    
    try:  # Wrap full function logic to ensure production-safe execution
        verbose_output(
            f"{BackgroundColors.GREEN}Clearing fitness cache...{Style.RESET_ALL}"
        )

        global fitness_cache  # Use the global fitness_cache variable
        
        with fitness_cache_lock:  # Thread-safe cache clearing
            fitness_cache.clear()  # Clear the fitness cache
            verbose_output(
                f"{BackgroundColors.GREEN}Fitness cache cleared ({BackgroundColors.CYAN}memory leak prevention{BackgroundColors.GREEN}).{Style.RESET_ALL}"
            )
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def normalize_feature_name(name):
    """
    Normalize feature name by stripping whitespace, replacing double spaces with single spaces, and lowercasing.

    :param name: The feature name to normalize.
    :return: Normalized feature name
    """

    try:  # Wrap full function logic to ensure production-safe execution
        return name.strip().replace("  ", " ").lower()  # Strip whitespace, replace double spaces, and lowercase
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_dataset(csv_path):
    """
    Load CSV and return DataFrame.

    :param csv_path: Path to CSV dataset.
    :return: DataFrame
    """

    try:  # Wrap full function logic to ensure production-safe execution
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def print_loaded_artifacts_info(csv_path, features, params):
    """
    Display information about loaded model artifacts to the console.

    :param csv_path: Path to the CSV dataset file
    :param features: List of selected feature names
    :param params: Dictionary of model parameters (can be None)
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe execution
        dataset_stem = Path(csv_path).stem  # Extract dataset name from path
        print(f"{BackgroundColors.GREEN}Loaded exported model and scaler for {BackgroundColors.CYAN}{dataset_stem}{Style.RESET_ALL}")  # Notify successful load
        
        feature_count = len(features)  # Count number of features
        print(f"{BackgroundColors.GREEN}Selected features ({feature_count}): {BackgroundColors.CYAN}{features}{Style.RESET_ALL}")  # Display feature list
        
        if params is not None:  # Verify if parameters were loaded
            params_json = json.dumps(params, default=str)  # Serialize parameters to JSON string
            print(f"{BackgroundColors.GREEN}Model parameters: {BackgroundColors.CYAN}{params_json}{Style.RESET_ALL}")  # Display model configuration
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def prepare_test_data_for_loaded_model(csv_path, features):
    """
    Execute the full dataset pipeline and select features for evaluation.

    :param csv_path: Path to the CSV dataset file
    :param features: List of feature names to select from the dataset
    :return: Tuple of (X_test_selected, y_test) or None on any failure
    """

    try:  # Wrap full function logic to ensure production-safe execution
        df = load_dataset(csv_path)  # Load raw dataset from CSV
        if df is None:  # Verify if loading failed
            return None  # Exit early on load failure

        cleaned_df = preprocess_dataframe(df, remove_zero_variance=CONFIG["dataset"]["remove_zero_variance"])  # Clean and preprocess the dataframe
        if cleaned_df is None or cleaned_df.empty:  # Verify if preprocessing failed or resulted in empty data
            return None  # Exit early on preprocessing failure

        split_data = split_dataset(cleaned_df, csv_path, test_size=CONFIG["dataset"]["test_size"])  # Split into train/test sets
        if split_data is None or split_data[0] is None:  # Verify if splitting failed
            return None  # Exit early on split failure

        if isinstance(split_data, (list, tuple)) and len(split_data) == 6:
            X_train, X_test, y_train, y_test, feature_names, _scaling_time = split_data
        else:
            print(f"{BackgroundColors.RED}Unexpected split_dataset output format. Expected 6 elements but got {len(split_data) if isinstance(split_data, (list, tuple)) else 'non-list/tuple'}{Style.RESET_ALL}")
            return None  # Exit early on unexpected split format
        
        sel_indices = [i for i, f in enumerate(feature_names) if f in features]  # Map loaded feature names to column indices
        if not sel_indices:  # Verify if no matching features were found
            return None  # Exit early if feature mapping failed

        X_test_sel = X_test[:, sel_indices]  # Select only the relevant feature columns from test data
        
        return (X_test_sel, y_test)  # Return selected test data and labels
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def evaluate_and_display_loaded_model(model, X_test_sel, y_test):
    """
    Evaluate a loaded model on test data and display formatted metrics.

    :param model: The loaded machine learning model
    :param X_test_sel: Test features (with selected columns only)
    :param y_test: Test labels
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe execution
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def handle_skip_train_if_model_exists(csv_path):
    """
    Search for existing exported model artifacts and handle loading and evaluation if they exist.

    :param csv_path: Path to the CSV dataset file
    :return: True if artifacts loaded and caller should return, False otherwise
    """

    try:  # Wrap full function logic to ensure production-safe execution
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
    
    try:  # Wrap full function logic to ensure production-safe execution
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

    try:  # Wrap full function logic to ensure production-safe execution
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


def cache_preprocessed_data(result, cache_file, csv_path):
    """
    Cache the preprocessed data to a pickle file, verifying disk space first.
    Also, compare and display size reduction compared to the original CSV.

    :param result: The tuple to cache (X_train_scaled, X_test_scaled, y_train_np, y_test_np, X.columns)
    :param cache_file: Path to the cache file.
    :param csv_path: Path to the original CSV file for size comparison.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe execution
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def split_dataset(df, csv_path, test_size=0.2):
    """
    Split dataset into training and testing sets.

    :param df: DataFrame to split.
    :param csv_path: Path to the CSV file for caching.
    :param test_size: Proportion of the dataset to include in the test split.
    :return: X_train, X_test, y_train, y_test
    """

    try:  # Wrap full function logic to ensure production-safe execution
        min_test_frac = CONFIG["dataset"]["min_test_fraction"]  # Get min test fraction from config
        max_test_frac = CONFIG["dataset"]["max_test_fraction"]  # Get max test fraction from config
        
        if not (min_test_frac <= test_size <= max_test_frac):  # Validate test_size and clamp if out of bounds
            print(
                f"{BackgroundColors.YELLOW}Warning: test_size={test_size} outside valid range [{min_test_frac}, {max_test_frac}]. Clamping to valid range.{Style.RESET_ALL}"
            )
            test_size = max(min_test_frac, min(max_test_frac, test_size))  # Clamp test_size to valid range

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
        start_scale = time.perf_counter()  # High-resolution start for scaling
        X_train_scaled = scaler.fit_transform(X_train)  # Fit scaler on training set and transform
        X_test_scaled = scaler.transform(X_test)  # Transform test set with the same scaler
        scaling_time = time.perf_counter() - start_scale  # Calculate scaling duration
        try:  # Try to attach scaling time to scaler instance using setattr to avoid static attribute access warnings
            setattr(scaler, "_scaling_time", round(float(scaling_time), 6))  # Set dynamic attribute safely
        except Exception:  # On any error, silently ignore to preserve original behavior
            pass  # No-op on failure

        y_train_np = np.array(y_train)  # Convert y_train and y_test to numpy arrays for fast indexing
        y_test_np = np.array(y_test)  # Convert y_train and y_test to numpy arrays for fast indexing

        result = X_train_scaled, X_test_scaled, y_train_np, y_test_np, X.columns, round(float(scaling_time), 6)  # Prepare result tuple (include scaling time)
        cache_preprocessed_data(result, cache_file, csv_path)  # Cache the preprocessed data with size comparison
        return result  # Return the splits and feature names
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def print_ga_parameters(min_pop, max_pop, n_generations, feature_count):
    """
    Print the genetic algorithm parameters in verbose output.

    :param min_pop: Minimum population size.
    :param max_pop: Maximum population size.
    :param n_generations: Number of generations per run.
    :param feature_count: Number of features in the dataset.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe execution
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
            f"  {BackgroundColors.GREEN}Base estimator: {BackgroundColors.CYAN}RandomForestClassifier (n_estimators=100, n_jobs={CONFIG['multiprocessing']['n_jobs']}){Style.RESET_ALL}"
        )
        print(f"  {BackgroundColors.GREEN}Optimization goal: {BackgroundColors.CYAN}Maximize F1-Score{Style.RESET_ALL}")
        print("")  # Empty line for spacing
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


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

    try:  # Wrap full function logic to ensure production-safe execution
        verbose_output(
            f"{BackgroundColors.GREEN}Preparing dataset '{dataset_name}' for Genetic Algorithm sweep.{Style.RESET_ALL}"
        )  # Output the verbose message

        df = load_dataset(csv_path)  # Load dataset
        if df is None:  # If loading failed
            return None  # Exit early

        cleaned_df = preprocess_dataframe(df, remove_zero_variance=CONFIG["dataset"]["remove_zero_variance"])  # Preprocess dataset

        if cleaned_df is None or cleaned_df.empty:  # If preprocessing failed or dataset is empty
            print(f"{BackgroundColors.RED}Dataset empty after preprocessing. Exiting.{Style.RESET_ALL}")
            return None  # Exit early

        split_res = split_dataset(cleaned_df, csv_path, test_size=CONFIG["dataset"]["test_size"])  # Split dataset (may include scaling time)
        if split_res is None or split_res[0] is None:  # If splitting failed
            return None  # Exit early
        if isinstance(split_res, (list, tuple)) and len(split_res) == 6:
            X_train, X_test, y_train, y_test, feature_names, split_scaling_time = split_res
        else:
            X_train, X_test, y_train, y_test, feature_names = split_res
            split_scaling_time = None
        if X_train is None:  # If splitting failed
            return None  # Exit early

        (
            print_ga_parameters(min_pop, max_pop, n_generations, len(feature_names) if feature_names is not None else 0)
            if CONFIG.get("execution", {}).get("verbose", False)
            else None
        )  # Print GA parameters if verbose

        train_count = len(y_train) if y_train is not None else 0  # Count training samples
        test_count = len(y_test) if y_test is not None else 0  # Count testing samples
        verbose_output(
            f"  {BackgroundColors.GREEN}  Dataset: {BackgroundColors.CYAN}{dataset_name} - {train_count} training / {test_count} testing  (80/20){Style.RESET_ALL}\n"
        )

        return X_train, X_test, y_train, y_test, feature_names  # Return prepared data
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def compute_progress_state(min_pop, max_pop, n_generations, runs, progress_bar, folds=10):
    """
    Compute an estimated progress_state dictionary for the population sweep.

    The function returns a dict with keys:
      - current_it: starting at 0
      - total_it: estimated total number of classifier instantiations

    The estimation assumes each individual evaluation runs "folds" classifier
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

    try:  # Wrap full function logic to ensure production-safe execution
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


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

    try:  # Wrap full function logic to ensure production-safe execution
        try:  # Attempt to compute the state id
            key = f"{csv_path}|pop{pop_size}|gens{n_generations}|cx{cxpb}|mut{mutpb}|run{run}|folds{folds}|test{test_frac}"  # Create a unique key string from run parameters
            return hashlib.sha256(
                key.encode("utf-8")
            ).hexdigest()  # Compute SHA256 hash of the key and return as hex string
        except Exception:  # If any error occurs during computation
            return None  # Return None to indicate failure
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def state_file_paths(output_dir, state_id):
    """
    Return (gen_state_path, run_state_path) for a given state id and ensure dir exists.

    :param output_dir: base output directory
    :param state_id: deterministic id for run
    :return: tuple(gen_path, run_path)
    """

    try:  # Wrap full function logic to ensure production-safe execution
        state_dir = os.path.join(output_dir, CONFIG["progress"]["state_dir_name"])  # Construct the state directory path
        try:  # Try to create the state directory if it doesn't exist
            os.makedirs(state_dir, exist_ok=True)  # Create the directory, ignoring if it already exists
        except Exception:  # If directory creation fails
            pass  # Do nothing
        return os.path.join(state_dir, f"{state_id}_gen.pkl"), os.path.join(
            state_dir, f"{state_id}_run.pkl"
        )  # Return paths for generation and run state files
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_run_result(output_dir, state_id):
    """
    Load a previously saved run result if present.

    :param output_dir: base output directory
    :param state_id: deterministic id for run
    :return: deserialized result or None
    """

    try:  # Wrap full function logic to ensure production-safe execution
        try:  # Attempt to load the run result
            _, run_path = state_file_paths(output_dir, state_id)  # Get the path for the run state file
            if not os.path.exists(run_path):  # Verify if the file exists
                return None  # Return None if file does not exist
            with open(run_path, "rb") as f:  # Open the file for reading in binary mode
                return pickle.load(f)  # Deserialize and return the result
        except Exception:  # If any error occurs during loading
            return None  # Return None to indicate failure
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


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

    try:  # Wrap full function logic to ensure production-safe execution
        try:  # Attempt to verify for cached run result
            n_train = len(y_train) if y_train is not None else 0  # Get number of training samples
            n_test = len(y_test) if y_test is not None else 0  # Get number of test samples
            test_frac = (
                float(n_test) / float(n_train + n_test) if (n_train + n_test) > 0 else None
            )  # Calculate test fraction
            state_id = compute_state_id(
                csv_path or "", pop_size, n_generations, cxpb, mutpb, run, folds, test_frac=test_frac
            )  # Compute state id for the run
            if CONFIG["execution"]["resume_progress"] and state_id is not None:  # If resume is enabled and state_id exists
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def setup_genetic_algorithm(n_features, population_size=None, pool=None):
    """
    Setup DEAP Genetic Algorithm: creator, toolbox, population, and Hall of Fame.
    DEAP is a library for evolutionary algorithms in Python.

    :param n_features: Number of features in dataset
    :param population_size: Size of the population (default: n_features // 4, minimum 10)
    :param pool: Optional existing multiprocessing.Pool to reuse (avoids creating a new one)
    :return: toolbox, population, hall_of_fame
    """
    
    try:
        if population_size is None:  # If population_size is not provided
            population_size = max(n_features // 4, 10)  # Default to 1/4 of n_features, but at least 10

        verbose_output(
            f"{BackgroundColors.GREEN}Setting up Genetic Algorithm with {n_features} features and population size {population_size}.{Style.RESET_ALL}"
        )  # Output the verbose message

        if not hasattr(creator, "FitnessMulti"):  # If FitnessMulti class doesn't exist in creator
            creator.create(
                "FitnessMulti",
                base.Fitness,
                weights=(1.0, 1.0),
            )  # Create FitnessMulti with 2 objectives: maximize F1-score and minimize feature count
        FitnessMulti = getattr(creator, "FitnessMulti")  # Get FitnessMulti from creator namespace safely

        if not hasattr(creator, "Individual"):  # If Individual class doesn't exist in creator
            creator.create("Individual", list, fitness=FitnessMulti)  # Create Individual as list with FitnessMulti attribute
        Individual = getattr(creator, "Individual")  # Get Individual from creator namespace safely

        toolbox: Any = base.Toolbox()  # Toolbox (typed Any to avoid analyzer confusion)

        def _attr_bool() -> int:  # Binary attribute generator
            return random.randint(0, 1)  # Return random 0 or 1 for binary gene

        toolbox.register("attr_bool", random.randint, 0, 1)  # Register binary attribute generator in toolbox

        individual_factory: Callable[[], Any] = partial(tools.initRepeat, Individual, _attr_bool, n_features)  # Individual factory and registration
        toolbox.register("individual", individual_factory)  # Register individual factory in toolbox

        toolbox.register("population", tools.initRepeat, list, individual_factory)  # Population factory and registration

        toolbox.register("mate", tools.cxTwoPoint)  # Crossover operator
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)  # Mutation operator
        toolbox.register("select", tools.selNSGA2)  # Selection operator using NSGA-II for multi-objective Pareto-based optimization

        if pool is None:  # If no external pool was provided, create one
            with global_state_lock:  # Thread-safe read of CPU_PROCESSES
                cpu_procs = CPU_PROCESSES  # Read CPU_PROCESSES value
            if cpu_procs is None:  # If CPU_PROCESSES is not set
                pool = multiprocessing.Pool()  # Create a multiprocessing pool with all available CPUs
            else:  # If CPU_PROCESSES is set
                pool = multiprocessing.Pool(
                    processes=cpu_procs
                )  # Create a multiprocessing pool with specified number of CPUs
        toolbox.register("map", pool.map)  # Register parallel map for fitness evaluation

        population = toolbox.population(n=population_size)  # Create the initial population
        hof = tools.HallOfFame(1)  # Hall of Fame to store the best individual

        return toolbox, population, hof  # Return the toolbox, population, and Hall of Fame
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def instantiate_estimator(estimator_cls=None):
    """
    Instantiate a classifier. If estimator_cls is None, use RandomForestClassifier.

    :param estimator_cls: Class of the estimator to instantiate (or None)
    :return: instantiated estimator
    """
    
    try:
        try:  # Try to read multiprocessing configuration from global CONFIG
            mp_cfg = CONFIG.get("multiprocessing", {})  # Get multiprocessing config or empty dict
        except Exception:  # If CONFIG access fails for any reason
            mp_cfg = {}  # Fallback to empty config

        ga_parallel = (
            int(mp_cfg.get("cpu_processes", 1)) if isinstance(mp_cfg.get("cpu_processes", 1), int) else 1
        )  # Determine GA worker count from config
        if ga_parallel > 1:  # If GA uses multiple processes
            estimator_n_jobs = 1  # Force single-threaded estimator to avoid nested loky
        else:  # If GA is not parallelized across processes
            estimator_n_jobs = CONFIG.get("model", {}).get("n_jobs", 1)  # Use configured model n_jobs

        if estimator_cls is None:  # If no estimator class provided by caller
            return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=estimator_n_jobs)  # Return default RandomForest

        try:  # Try instantiating the provided estimator class
            try:  # Try to construct with n_jobs parameter when supported
                return estimator_cls(n_jobs=estimator_n_jobs)  # Instantiate estimator with n_jobs
            except TypeError:  # If provided class doesn't accept n_jobs
                return estimator_cls()  # Instantiate using no-arg constructor
        except Exception:  # If instantiation fails for any other reason
            return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=estimator_n_jobs)  # Fallback to RandomForest
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def evaluate_individual(
    individual,
    X_train,
    y_train,
    estimator_cls=None,
):
    """
    Evaluate the fitness of an individual solution using N_CV_FOLDS-fold Stratified Cross-Validation
    on the training set only (não combina train+test para evitar data leakage).

    :param individual: A list representing the individual solution (binary mask for feature selection).
    :param X_train: Training feature set.
    :param y_train: Training target variable.
    :param estimator_cls: Classifier class to use (default: RandomForestClassifier).
    :return: Tuple containing CV accuracy, precision, recall, F1-score, FPR, FNR, test accuracy, precision, recall, F1-score, FPR, FNR
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Evaluating individual: {BackgroundColors.CYAN}{individual}{Style.RESET_ALL}"
        )  # Output the verbose message

        num_features_selected = sum(individual)  # Count number of selected features for multi-objective optimization

        if num_features_selected == 0:  # If no features are selected
            return 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, num_features_selected  # Return worst possible scores for CV and test, with feature count

        mask_tuple = tuple(individual)  # Convert individual to tuple for hashing
        
        with fitness_cache_lock:  # Thread-safe cache access
            if mask_tuple in fitness_cache:  # Verify if already evaluated
                return fitness_cache[mask_tuple]  # Return cached result

        mask = np.array(individual, dtype=bool)  # Create boolean mask from individual
        X_train_sel = X_train[:, mask]  # Select features based on the mask

        n_cv_folds = CONFIG.get("cross_validation", {}).get("n_folds", 10)  # Use configurable constant
        metrics = np.empty((n_cv_folds, 6), dtype=float)  # Pre-allocate metrics array for each fold: [acc, prec, rec, f1, fpr, fnr]
        fold_count = 0  # Track how many folds actually ran

        try:  # Try to create StratifiedKFold splits
            skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=42)  # n_cv_folds-fold Stratified CV
            splits = list(skf.split(X_train_sel, y_train))  # Generate splits
        except Exception as e:  # If StratifiedKFold fails (e.g., too few samples per class)
            print(
                f"{BackgroundColors.YELLOW}Warning: StratifiedKFold failed ({type(e).__name__}: {str(e)}), using simple holdout validation on training data only.{Style.RESET_ALL}"
            )  # Output warning message
            
            from sklearn.model_selection import train_test_split as holdout_split  # Import holdout split function
            
            X_train_fold, X_val_fold, y_train_fold, y_val_fold = holdout_split(
                X_train_sel, y_train, test_size=0.2, random_state=42, stratify=y_train
            )  # Split training data into train/validation
            model = instantiate_estimator(estimator_cls)  # Instantiate the model
            model.fit(X_train_fold, y_train_fold)  # Fit on train fold
            y_pred = model.predict(X_val_fold)  # Predict on validation fold (NOT test set)

            acc = accuracy_score(y_val_fold, y_pred)  # Calculate accuracy on validation fold
            prec = precision_score(y_val_fold, y_pred, average="weighted", zero_division=0)  # Calculate precision
            rec = recall_score(y_val_fold, y_pred, average="weighted", zero_division=0)  # Calculate recall
            f1 = f1_score(y_val_fold, y_pred, average="weighted", zero_division=0)  # Calculate F1-score

            cm = confusion_matrix(y_val_fold, y_pred, labels=np.unique(y_val_fold))  # Confusion matrix on validation
            tn = cm[0, 0] if cm.shape == (2, 2) else 0  # True negatives
            fp = cm[0, 1] if cm.shape == (2, 2) else 0  # False positives
            fn = cm[1, 0] if cm.shape == (2, 2) else 0  # False negatives
            tp = cm[1, 1] if cm.shape == (2, 2) else 0  # True positives

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False negative rate

            result = acc, prec, rec, f1, fpr, fnr, 0, 0, 0, 0, 0, 0, num_features_selected  # Return validation metrics as CV, placeholder for test, with feature count
            with fitness_cache_lock:  # Thread-safe cache write
                fitness_cache[mask_tuple] = result  # Cache the result for this mask
            return result  # Return the result for holdout validation

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
                fold_idx < CONFIG["early_stop"]["folds"] and acc < CONFIG["early_stop"]["acc_threshold"]
            ):  # Early stopping: If accuracy is below threshold in first few folds, break
                early_stop_triggered = True  # Set flag
                break  # Stop evaluating further folds for this individual

        if early_stop_triggered:  # When early stopping triggers, return worst-case fitness
            result = 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, num_features_selected  # Worst possible scores with feature count
            with fitness_cache_lock:  # Thread-safe cache write
                fitness_cache[mask_tuple] = result  # Cache the worst-case result for this mask
            return result  # Return the worst-case result due to early stopping

        means = np.mean(metrics[:fold_count], axis=0) if fold_count > 0 else np.zeros(6)  # Calculate means for completed folds only
        acc, prec, rec, f1, fpr, fnr = means  # Unpack mean metrics

        test_acc, test_prec, test_rec, test_f1, test_fpr, test_fnr = 0, 0, 0, 0, 0, 0  # Placeholder test metrics

        result = acc, prec, rec, f1, fpr, fnr, test_acc, test_prec, test_rec, test_f1, test_fpr, test_fnr, num_features_selected  # Prepare result tuple with feature count
        
        with fitness_cache_lock:  # Thread-safe cache write
            fitness_cache[mask_tuple] = result  # Cache the result
        
        return result  # Return vectorized average metrics with feature count for multi-objective optimization
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def ga_fitness(ind, fitness_func):
    """
    Global fitness function for GA multi-objective evaluation to avoid pickle issues with local functions.

    :param ind: Individual to evaluate
    :param fitness_func: Partial function for evaluation
    :return: Tuple with (F1-score, -num_features) for multi-objective optimization (maximize F1, minimize features)
    """
    
    try:
        evaluation_result = fitness_func(ind)  # Evaluate individual to get full metrics tuple
        f1_score = evaluation_result[3]  # Extract F1-score (index 3 in metrics tuple)
        num_features = evaluation_result[12]  # Extract number of selected features (index 12 in extended metrics tuple)
        return (f1_score, -num_features)  # Return multi-objective fitness: maximize F1-score, minimize feature count (via negative)
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


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
    
    try:
        if sum(individual) == 0:  # If no features are selected
            return 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1  # Return worst possible scores

        cv_result = evaluate_individual(individual, X_train, y_train, estimator_cls)  # Get CV metrics from the standard evaluation function (only 4 params)
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_generation_state(output_dir, state_id):
    """
    Load generation state if present, returning the payload or None.

    :param output_dir: base output directory
    :param state_id: deterministic id for run
    :return: payload dict or None
    """
    
    try:
        try:  # Attempt to load the generation state
            gen_path, _ = state_file_paths(output_dir, state_id)  # Get the path for the generation state file
            if not os.path.exists(gen_path):  # Verify if the file exists
                return None  # Return None if file does not exist
            with open(gen_path, "rb") as f:  # Open the file for reading in binary mode
                return pickle.load(f)  # Deserialize and return the payload
        except Exception:  # If any error occurs during loading
            return None  # Return None to indicate failure
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def recreate_population_from_lists(toolbox, pop_lists):
    """
    Create DEAP individual objects from plain lists using registered toolbox.individual.

    :param toolbox: DEAP toolbox with "individual" registered
    :param pop_lists: iterable of bit-lists
    :return: list of individuals or empty list on error
    """
    
    try:
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


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
    
    try:
        start_gen = 1  # Initialize starting generation to 1
        fitness_history = []  # Initialize fitness history as empty list
        if CONFIG["execution"]["resume_progress"] and state_id is not None:  # Verify if resume is enabled and state_id is provided
            try:  # Attempt to load and apply the state
                payload = load_generation_state(output_dir, state_id)  # Load the generation state payload
                if payload:  # If payload exists
                    pop_lists = payload.get("population_lists")  # Get the population lists from payload
                    if pop_lists:  # If population lists exist
                        recreated = recreate_population_from_lists(toolbox, pop_lists)  # Recreate population from lists
                        if recreated:  # If recreation succeeded
                            population[:] = recreated  # Replace the population with recreated one

                        fitness_history_raw = payload.get("fitness_history", [])  # Get fitness history from payload
                        if isinstance(fitness_history_raw, dict):  # If new format (dict with extended history)
                            fitness_history = fitness_history_raw  # Use as-is
                        elif isinstance(fitness_history_raw, list):  # If old format (list of F1 scores only)
                            fitness_history = {"best_f1": fitness_history_raw}  # Convert to dict format for backward compatibility
                        else:  # If unexpected format
                            fitness_history = {}  # Initialize as empty dict

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

        return start_gen, fitness_history  # Return the starting generation and fitness history (dict or empty)
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def save_generation_state(output_dir, state_id, gen, population, hof_best, history_data):
    """
    Persist minimal generation state to disk (lists only) including extended history data.

    :param output_dir: base output directory
    :param state_id: deterministic id for run
    :param gen: generation number
    :param population: list of individuals
    :param hof_best: best individual list or None
    :param history_data: dict containing all history metrics (best_f1, best_features, etc.) or list for backward compatibility
    :return: None
    """
    
    try:
        try:  # Attempt to save the generation state
            gen_path, _ = state_file_paths(output_dir, state_id)  # Get the path for the generation state file

            if isinstance(history_data, dict):  # If new format (extended history)
                fitness_history_to_save = history_data  # Save the full dict
            elif isinstance(history_data, list):  # If old format (just F1 scores)
                fitness_history_to_save = {"best_f1": history_data}  # Wrap in dict for consistency
            else:  # If unexpected format
                fitness_history_to_save = {}  # Save empty dict

            payload = {  # Prepare the payload dictionary
                "gen": int(gen),  # Current generation number
                "population_lists": [list(ind) for ind in population],  # List of population individuals as lists
                "hof_best": list(hof_best) if hof_best is not None else None,  # Best individual from hall of fame
                "fitness_history": fitness_history_to_save,  # Extended history data (dict format)
            }  # End of payload dictionary
            with open(gen_path, "wb") as f:  # Open the file for writing in binary mode
                pickle.dump(payload, f, protocol=CONFIG["caching"]["pickle_protocol"])  # Serialize and save the payload
        except Exception:  # If any error occurs during saving
            pass  # Do nothing
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def run_genetic_algorithm_loop(
    toolbox,
    population,
    hof,
    X_train,
    y_train,
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
    :param n_generations: Number of generations to run.
    :param show_progress: Whether to show the tqdm progress bar.
    :param progress_bar: Optional tqdm progress bar instance to update.
    :param dataset_name: Optional dataset name for progress bar display.
    :param csv_path: Optional CSV path for progress bar display.
    :param pop_size: Optional population size for progress bar display.
    :param max_pop: Optional max population size for progress bar display.
    :param cxpb: Crossover probability.
    :param mutpb: Mutation probability.
    :param run: Optional run index for progress bar display.
    :param runs: Optional total runs for progress bar display.
    :param progress_state: Optional dict to track progress state across calls.
    :return: best individual
    """

    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Running Genetic Algorithm for {n_generations} generations.{Style.RESET_ALL}"
        )  # Output the verbose message

        fitness_func = partial(
            evaluate_individual, X_train=X_train, y_train=y_train
        )  # Partial function for evaluation
        toolbox.register("evaluate", partial(ga_fitness, fitness_func=fitness_func))  # Register the global fitness function

        global GA_GENERATIONS_COMPLETED  # To track completed generations
        best_fitness = None  # Track the best fitness value
        gens_without_improvement = 0  # Counter for generations with no improvement
        early_stop_gens = CONFIG.get("early_stop", {}).get("generations", 10)  # Use configured constant

        folds = CONFIG.get("cross_validation", {}).get("n_folds", 10)  # Use configured constant for CV folds

        output_dir = (
            f"{os.path.dirname(csv_path)}/Feature_Analysis" if csv_path else os.path.join(".", "Feature_Analysis")
        )  # Output directory for Feature_Analysis outputs
        state_id = compute_state_id(
            csv_path or "", pop_size or 0, n_generations, cxpb, mutpb, run or 0, folds, test_frac=None
        )  # Deterministic state id for resume/caching
        start_gen = 1  # Starting generation index

        start_gen, loaded_history = load_and_apply_generation_state(
            toolbox, population, output_dir, state_id, run=run
        )  # Load and apply saved generation state if available, updating start generation and history

        fitness_history = loaded_history.get("best_f1", []) if isinstance(loaded_history, dict) else []  # Best F1 scores
        best_features_history = loaded_history.get("best_features", []) if isinstance(loaded_history, dict) else []  # Best feature counts
        avg_f1_history = loaded_history.get("avg_f1", []) if isinstance(loaded_history, dict) else []  # Population avg F1
        avg_features_history = loaded_history.get("avg_features", []) if isinstance(loaded_history, dict) else []  # Population avg features
        pareto_size_history = loaded_history.get("pareto_size", []) if isinstance(loaded_history, dict) else []  # Pareto front sizes
        hypervolume_history = loaded_history.get("hypervolume", []) if isinstance(loaded_history, dict) else []  # Hypervolume values
        diversity_history = loaded_history.get("diversity", []) if isinstance(loaded_history, dict) else []  # Diversity values

        gen_range = (
            tqdm(range(start_gen, n_generations + 1), desc=f"{BackgroundColors.GREEN}Generations{Style.RESET_ALL}")
            if show_progress
            else range(start_gen, n_generations + 1)
        )  # Create generation range with progress bar if show_progress is enabled, otherwise use plain range
        gens_ran = 0  # Track how many generations were actually executed
        # Telegram progress notification settings (percent-based milestones)
        telegram_cfg = CONFIG.get("telegram", {}) if isinstance(CONFIG, dict) else {}
        telegram_enabled = bool(telegram_cfg.get("enabled", True))
        try:
            telegram_progress_pct = int(telegram_cfg.get("progress_pct", 10))
        except Exception:
            telegram_progress_pct = 10
        last_telegram_block = -1  # Track last percentage block notified so we don't repeat

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

            population[:] = toolbox.select(offspring, k=len(population))  # Select the next generation population using NSGA-II multi-objective selection
            hof.update(population)  # Update the Hall of Fame

            if hof and len(hof) > 0:  # If hall of fame has a best individual
                if hof[0] not in population:  # If the best individual is not in the new population
                    population[-1] = hof[0]  # Replace the worst individual with the hall-of-fame best

            current_best_fitness_f1 = (
                hof[0].fitness.values[0] if hof and hof[0].fitness.values else None
            )  # Get current best F1-score (first objective) from multi-objective fitness
            current_best_num_features = (
                -hof[0].fitness.values[1] if hof and hof[0].fitness.values and len(hof[0].fitness.values) > 1 else None
            )  # Get current number of features (negated second objective) from multi-objective fitness
            try:  # Try to append best fitness to history
                fitness_history.append(
                    float(current_best_fitness_f1) if current_best_fitness_f1 is not None else np.nan
                )  # Record best F1-score for convergence tracking
            except Exception:  # If conversion fails
                fitness_history.append(np.nan)  # Record NaN

            try:  # Track number of features in best individual
                best_features_history.append(
                    int(current_best_num_features) if current_best_num_features is not None else 0
                )  # Record best feature count
            except Exception:  # If tracking fails
                best_features_history.append(0)  # Record zero

            try:  # Track population average F1 score
                valid_f1_scores = [ind.fitness.values[0] for ind in population if ind.fitness.valid and len(ind.fitness.values) > 0]  # Extract valid F1 scores from population
                avg_f1 = sum(valid_f1_scores) / len(valid_f1_scores) if valid_f1_scores else 0.0  # Calculate average F1
                avg_f1_history.append(float(avg_f1))  # Record average F1 score
            except Exception:  # If calculation fails
                avg_f1_history.append(0.0)  # Record zero

            try:  # Track population average feature count
                valid_feature_counts = [-ind.fitness.values[1] for ind in population if ind.fitness.valid and len(ind.fitness.values) > 1]  # Extract valid feature counts (negate second objective)
                avg_features = sum(valid_feature_counts) / len(valid_feature_counts) if valid_feature_counts else 0.0  # Calculate average feature count
                avg_features_history.append(float(avg_features))  # Record average feature count
            except Exception:  # If calculation fails
                avg_features_history.append(0.0)  # Record zero

            try:  # Track Pareto front size
                pareto_front = extract_pareto_front(population)  # Extract Pareto front from population
                pareto_size_history.append(len(pareto_front))  # Record Pareto front size
            except Exception:  # If extraction fails
                pareto_size_history.append(0)  # Record zero

            try:  # Track hypervolume metric
                pareto_front = extract_pareto_front(population)  # Extract Pareto front (recompute if needed or reuse from above)
                hv = calculate_hypervolume(pareto_front)  # Calculate hypervolume for Pareto front
                if hv is None:  # If hypervolume calculation returns None, record zero
                    hypervolume_history.append(0.0)  # Record zero
                else:  # If hypervolume is valid, record the value
                    try:  # Try to convert hypervolume to float for recording
                        hypervolume_history.append(float(hv))  # Record hypervolume
                    except Exception:  # If conversion fails
                        hypervolume_history.append(0.0)
            except Exception:  # If calculation fails
                hypervolume_history.append(0.0)  # Record zero

            try:  # Track population diversity
                diversity = calculate_population_diversity(population)  # Calculate population diversity
                diversity_history.append(float(diversity))  # Record diversity metric
            except Exception:  # If calculation fails
                diversity_history.append(0.0)  # Record zero

            if best_fitness is None or (current_best_fitness_f1 is not None and current_best_fitness_f1 > best_fitness):
                best_fitness = current_best_fitness_f1  # Update best F1-score
                gens_without_improvement = 0  # Reset counter
            else:  # If no improvement in best F1-score
                gens_without_improvement += 1  # Increment counter

                if gens_without_improvement >= early_stop_gens:  # Verify early-stop condition
                    print(
                        f"{BackgroundColors.YELLOW}Early stopping: No improvement in best fitness for {early_stop_gens} generations. Stopping at generation {gen}.{Style.RESET_ALL}"
                    )  # Print early stopping message
                    gens_ran = gen  # Record how many generations were executed before early stopping
                    with global_state_lock:  # Thread-safe update
                        GA_GENERATIONS_COMPLETED = int(gen)  # Update global variable
                    break  # Stop the loop early

            # Decide whether to send a Telegram update based on percent milestones
            should_send = False
            if telegram_enabled and show_progress and telegram_progress_pct > 0:
                try:
                    # Compute current percent (0-100)
                    current_pct = (gen * 100) // max(1, int(n_generations))
                    current_block = current_pct // telegram_progress_pct
                    if current_block > last_telegram_block:
                        should_send = True
                        last_telegram_block = current_block
                except Exception:
                    should_send = False

            send_telegram_message(
                TELEGRAM_BOT,
                [
                    f"Pop Size {pop_size}: Generation {gen}/{n_generations}, Best F1-Score: {truncate_value(best_fitness)}, Features: {int(current_best_num_features) if current_best_num_features is not None else 'N/A'}"
                ],
                should_send,
            )  # Send periodic updates to Telegram with multi-objective fitness values at configured percent milestones

            gens_ran = gen  # Update gens_ran each generation
            with global_state_lock:  # Thread-safe update to GA_GENERATIONS_COMPLETED
                GA_GENERATIONS_COMPLETED = int(gen)  # Update global variable
            gens_ran = gen if gens_ran == 0 else gens_ran  # Ensure gens_ran is set correctly if no early stopping occurred

            try:  # Persist per-generation progress so runs can be resumed (every N gens to reduce I/O)
                if CONFIG["execution"]["resume_progress"] and state_id is not None and (gen % CONFIG["execution"]["progress_save_interval"] == 0 or gen == n_generations):  # Use configured interval
                    current_history_data = {
                        "best_f1": fitness_history,  # Best F1 scores up to current generation
                        "best_features": best_features_history,  # Best feature counts up to current generation
                        "avg_f1": avg_f1_history,  # Average F1 scores up to current generation
                        "avg_features": avg_features_history,  # Average feature counts up to current generation
                        "pareto_size": pareto_size_history,  # Pareto sizes up to current generation
                        "hypervolume": hypervolume_history,  # Hypervolume values up to current generation
                        "diversity": diversity_history,  # Diversity values up to current generation
                    }  # Consolidated history for state persistence
                    save_generation_state(
                        output_dir, state_id, gen, population, hof[0] if hof and len(hof) > 0 else None, current_history_data
                    )
            except Exception:  # If saving fails
                pass  # Do nothing

            history_data = {
                "best_f1": fitness_history,  # Best F1 score per generation (backward compatible)
                "best_features": best_features_history,  # Best feature count per generation
                "avg_f1": avg_f1_history,  # Population average F1 per generation
                "avg_features": avg_features_history,  # Population average feature count per generation
                "pareto_size": pareto_size_history,  # Pareto front size per generation
                "hypervolume": hypervolume_history,  # Hypervolume metric per generation
                "diversity": diversity_history,  # Population diversity per generation
            }  # Consolidated history dictionary for plotting

            return hof[0], gens_ran, history_data  # Return the best individual, gens ran and comprehensive history data
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def adjust_progress_for_early_stop(progress_state, n_generations, pop_size, gens_ran, folds):
    """
    Adjust "progress_state" when a GA run finishes early.

    This subtracts planned-but-not-executed classifier instantiations from
    "progress_state["total_it"]" and increments "current_it" for the final
    re-evaluation (which is always performed once per run).

    :param progress_state: dict with keys "current_it" and "total_it".
    :param n_generations: configured total generations for the run.
    :param pop_size: population size used for the run.
    :param gens_ran: number of generations actually executed.
    :param folds: number of CV folds (classifier instantiations per evaluation).
    :return: None (mutates "progress_state" in-place)
    """
    
    try:
        if not (progress_state and isinstance(progress_state, dict)):  # Validate progress_state
            return  # Nothing to do if invalid

        try:  # Each generation evaluates pop_size individuals
            planned_evals = int(n_generations) * int(pop_size)  # Planned evaluations: generations * pop_size
            actual_evals = int(gens_ran) * int(pop_size)  # Actual evaluations: generations run * pop_size
            saved_evals = max(0, planned_evals - actual_evals)  # Number of individual evaluations saved by early stopping
            progress_state["total_it"] = max(
                0, int(progress_state.get("total_it", 0)) - saved_evals * folds
            )  # Reduce total iterations by saved evaluations * folds
        except Exception:  # Silently ignore failures during adjustment
            pass  # Do nothing on error

            try:  # Update current_it by the single re-evaluation performed after GA (folds classifiers)
                progress_state["current_it"] = (
                    int(progress_state.get("current_it", 0)) + folds
                )  # Increment current_it for final re-eval
            except Exception:  # Silently ignore failures when updating current_it
                pass  # Do nothing on error
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def safe_filename(name):
    """
    Sanitize a string to be safe for use as a filename.

    :param name: The string to be sanitized.
    :return: A sanitized string safe for use as a filename.
    """
    
    try:
        if not name:  # Handle empty/None input
            return "unnamed"  # Return a default name for empty input
        
        sanitized = re.sub(r'[\\/*?:"<>|]', "_", str(name))  # Replace invalid filename characters with underscores
        
        max_len = 200  # Conservative limit leaving room for extensions
        if len(sanitized) > max_len:  # Truncate if too long to avoid filesystem issues
            sanitized = sanitized[:max_len]  # Truncate to max_len characters
        
        sanitized = sanitized.strip(". ")  # Remove leading/trailing dots and spaces which can cause issues on some filesystems
        
        return sanitized if sanitized else "unnamed"  # Ensure we don't return an empty string
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def extract_pareto_front(population):
    """
    Extract the Pareto front (non-dominated individuals) from a population.

    :param population: List of DEAP individuals with multi-objective fitness values
    :param: List of non-dominated individuals forming the Pareto front
    """
    
    try:
        if not population:  # If population is empty
            return []  # Return empty list

        pareto_front = []  # Initialize Pareto front as empty list
        try:  # Attempt to extract Pareto front
            for candidate in population:  # Iterate over each individual in population
                if not candidate.fitness.valid:  # Skip individuals without valid fitness
                    continue  # Move to next individual
                is_dominated = False  # Flag to check if candidate is dominated
                for other in population:  # Compare candidate with all other individuals
                    if not other.fitness.valid or candidate is other:  # Skip invalid or same individual
                        continue  # Move to next individual
                    if tools.emo.isDominated(candidate.fitness.values, other.fitness.values):  # Check if candidate is dominated by other
                        is_dominated = True  # Set dominated flag
                        break  # Stop checking
                if not is_dominated:  # If candidate is not dominated
                        pareto_front.append(candidate)  # Add to Pareto front
        except Exception:  # If any error occurs
            return []  # Return empty list

        return pareto_front  # Return the Pareto front
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def calculate_hypervolume(pareto_front, reference_point=(0.0, -100.0)):
    """
    Calculate hypervolume metric for a Pareto front (simplified 2D implementation).

    :param pareto_front: List of individuals in Pareto front with 2-objective fitness
    :param reference_point: Reference point (f1_min, -num_features_max) for hypervolume calculation
    :return: Hypervolume value as float, or 0.0 if calculation fails
    """
    
    try:
        if not pareto_front:  # If Pareto front is empty
            return 0.0  # Return zero hypervolume

        try:  # Attempt hypervolume calculation
            points = []  # List to store valid fitness points
            for ind in pareto_front:  # Iterate over Pareto front individuals
                if ind.fitness.valid and len(ind.fitness.values) >= 2:  # Verify valid 2-objective fitness
                    points.append((ind.fitness.values[0], ind.fitness.values[1]))  # Extract (f1, -features) tuple

            if len(points) < 1:  # If no valid points
                return 0.0  # Return zero hypervolume

            points = sorted(points, key=lambda p: p[0], reverse=True)  # Sort points by F1 score descending for sweep algorithm
            hypervolume = 0.0  # Initialize hypervolume accumulator
            prev_f1 = reference_point[0]  # Initialize previous F1 from reference point

            for f1, neg_features in points:  # Sweep through sorted points
                width = f1 - prev_f1  # Width of hypervolume slice
                height = neg_features - reference_point[1]  # Height of hypervolume slice
                if width > 0 and height > 0:  # Only add positive contributions
                    hypervolume += width * height  # Add slice area to total hypervolume
                prev_f1 = f1  # Update previous F1 for next slice

                return hypervolume  # Return calculated hypervolume
        except Exception:  # If any error occurs
            return 0.0  # Return zero hypervolume
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def calculate_population_diversity(population):
    """
    Calculate population diversity using average pairwise Hamming distance.

    :param population: List of DEAP individuals (binary feature masks)
    :return: Average Hamming distance as float, or 0.0 if calculation fails
    """
    
    try:
        if not population or len(population) < 2:  # If population too small
            return 0.0  # Return zero diversity

        try:  # Attempt diversity calculation
            total_distance = 0.0  # Initialize total distance accumulator
            comparisons = 0  # Initialize comparison counter

            for i in range(len(population)):  # Iterate over first individual
                for j in range(i + 1, len(population)):  # Iterate over second individual (avoid duplicates)
                    ind1, ind2 = population[i], population[j]  # Get pair of individuals
                    if len(ind1) != len(ind2):  # Skip if different lengths
                        continue  # Move to next pair
                    hamming_dist = sum(g1 != g2 for g1, g2 in zip(ind1, ind2))  # Calculate Hamming distance
                    total_distance += hamming_dist  # Add to total
                    comparisons += 1  # Increment comparison counter

            if comparisons > 0:  # If valid comparisons made
                return total_distance / comparisons  # Return average Hamming distance
            else:  # If no valid comparisons
                    return 0.0  # Return zero diversity
        except Exception:  # If any error occurs
            return 0.0  # Return zero diversity
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


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
    
    try:
        output_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis"  # Directory to save outputs
        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

        base_dataset_name = (
            safe_filename(os.path.splitext(os.path.basename(csv_path))[0])
            if not dataset_name
            else safe_filename(dataset_name)
        )  # Base name of the dataset
        gens_part = f"gens{int(n_generations)}" if n_generations is not None else "gensNA"  # Format generations part for filename
        cx_part = f"cx{int(cxpb*100)}"  # Format crossover probability as percentage for filename
        mut_part = f"mut{int(mutpb*100)}"  # Format mutation probability as percentage for filename
        fig_path = os.path.join(
            output_dir, f"{base_dataset_name}_run{run}_pop{pop_size}_{gens_part}_{cx_part}_{mut_part}_convergence.png"
        )  # Path to save the figure

        try:  # Try to plot and save the figure
            plt.figure(figsize=(8, 4))  # Create a matplotlib figure with 8x4 dimensions
            gens = list(range(1, len(fitness_history) + 1))  # Generate list of generation numbers starting from 1
            plt.plot(gens, fitness_history, marker="o", linestyle="-", color="#1f77b4")  # Plot fitness history with blue line and circle markers
            plt.xlabel("Generation")  # Set X-axis label
            plt.ylabel("Best F1-Score")  # Set Y-axis label
            plt.title(f"GA Convergence - {base_dataset_name} (run={run}, pop={pop_size}, cx={cxpb}, mut={mutpb})")  # Set plot title with run parameters
            plt.grid(True, linestyle="--", alpha=0.5)  # Add dashed grid with 50% transparency
            plt.tight_layout()  # Adjust subplot parameters for tight layout
            plt.savefig(fig_path, dpi=150)  # Save figure to file with 150 DPI resolution
            plt.close()  # Close plot to free memory resources
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def generate_convergence_plots(
    history_data,
    csv_path,
    pop_size,
    run,
    dataset_name=None,
    n_generations=None,
    cxpb=0.5,
    mutpb=0.2,
):
    """
    Generate comprehensive convergence and progress visualization plots for a single GA run.

    :param history_data: Dict containing all tracking histories (best_f1, best_features, avg_f1, avg_features, pareto_size, hypervolume, diversity)
    :param csv_path: Path to the dataset CSV (used to determine output directory)
    :param pop_size: Population size used in this run
    :param run: Run index (1-based)
    :param dataset_name: Optional dataset name for title/filename
    :param n_generations: Number of generations configured
    :param cxpb: Crossover probability
    :param mutpb: Mutation probability
    :return: List of paths to all saved plots
    """
    
    try:
        output_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis"  # Base directory for outputs
        os.makedirs(output_dir, exist_ok=True)  # Ensure base directory exists

        base_dataset_name = (
            safe_filename(os.path.splitext(os.path.basename(csv_path))[0])
            if not dataset_name
            else safe_filename(dataset_name)
        )  # Sanitized base name of the dataset

        gens_part = f"gens{int(n_generations)}" if n_generations is not None else "gensNA"  # Format generations for filename
        cx_part = f"cx{int(cxpb*100)}"  # Format crossover probability as percentage
        mut_part = f"mut{int(mutpb*100)}"  # Format mutation probability as percentage
        run_id = f"{base_dataset_name}_run{run}_pop{pop_size}_{gens_part}_{cx_part}_{mut_part}"  # Unique run identifier

        plot_output_dir = os.path.join(output_dir, "ga_progress_plots", run_id)  # Subdirectory for this run's plots
        os.makedirs(plot_output_dir, exist_ok=True)  # Ensure plot output directory exists

        saved_plots = []  # List to track all saved plot paths

        try:  # Attempt to generate all convergence plots
            best_f1_history = history_data.get("best_f1", [])  # Best F1 score per generation
            best_features_history = history_data.get("best_features", [])  # Best feature count per generation
            avg_f1_history = history_data.get("avg_f1", [])  # Average F1 score per generation
            avg_features_history = history_data.get("avg_features", [])  # Average feature count per generation
            pareto_size_history = history_data.get("pareto_size", [])  # Pareto front size per generation
            hypervolume_history = history_data.get("hypervolume", [])  # Hypervolume per generation
            diversity_history = history_data.get("diversity", [])  # Diversity per generation

            n_gens = max(
                len(best_f1_history),
                len(best_features_history),
                len(avg_f1_history),
                len(avg_features_history),
                len(pareto_size_history),
                len(hypervolume_history),
                len(diversity_history),
            )  # Determine actual number of generations from available data

            if n_gens == 0:  # If no history data available
                verbose_output(
                    f"{BackgroundColors.YELLOW}No history data available for convergence plots{Style.RESET_ALL}"
                )  # Log warning
                return []  # Return empty list

            generations = list(range(1, n_gens + 1))  # Generate list of generation numbers starting from 1

            if best_f1_history:  # If best F1 history exists
                try:  # Try to create plot
                    plt.figure(figsize=(10, 6))  # Create figure with 10x6 dimensions
                    plt.plot(generations[: len(best_f1_history)], best_f1_history, marker="o", linestyle="-", color="#1f77b4", linewidth=2, markersize=4)  # Plot best F1 with blue line
                    plt.xlabel("Generation", fontsize=12)  # Set X-axis label
                    plt.ylabel("Best F1-Score", fontsize=12)  # Set Y-axis label
                    plt.title(f"Best F1-Score Convergence\\n{base_dataset_name} (run={run}, pop={pop_size})", fontsize=14)  # Set plot title
                    plt.grid(True, linestyle="--", alpha=0.5)  # Add grid with 50% transparency
                    plt.tight_layout()  # Adjust layout
                    plot_path = os.path.join(plot_output_dir, "01_best_f1_convergence.png")  # Define plot path
                    plt.savefig(plot_path, dpi=150)  # Save figure with 150 DPI
                    plt.close()  # Close plot to free memory
                    saved_plots.append(plot_path)  # Add to saved plots list
                except Exception:  # If plotting fails
                    plt.close()  # Close plot to free memory

            if best_features_history:  # If best features history exists
                try:  # Try to create plot
                    plt.figure(figsize=(10, 6))  # Create figure with 10x6 dimensions
                    plt.plot(generations[: len(best_features_history)], best_features_history, marker="s", linestyle="-", color="#ff7f0e", linewidth=2, markersize=4)  # Plot feature count with orange line
                    plt.xlabel("Generation", fontsize=12)  # Set X-axis label
                    plt.ylabel("Number of Selected Features", fontsize=12)  # Set Y-axis label
                    plt.title(f"Feature Count Evolution\\n{base_dataset_name} (run={run}, pop={pop_size})", fontsize=14)  # Set plot title
                    plt.grid(True, linestyle="--", alpha=0.5)  # Add grid
                    plt.tight_layout()  # Adjust layout
                    plot_path = os.path.join(plot_output_dir, "02_feature_count_evolution.png")  # Define plot path
                    plt.savefig(plot_path, dpi=150)  # Save figure
                    plt.close()  # Close plot
                    saved_plots.append(plot_path)  # Add to saved plots list
                except Exception:  # If plotting fails
                    plt.close()  # Close plot

            if pareto_size_history:  # If Pareto size history exists
                try:  # Try to create plot
                    plt.figure(figsize=(10, 6))  # Create figure
                    plt.plot(generations[: len(pareto_size_history)], pareto_size_history, marker="^", linestyle="-", color="#2ca02c", linewidth=2, markersize=4)  # Plot Pareto size with green line
                    plt.xlabel("Generation", fontsize=12)  # Set X-axis label
                    plt.ylabel("Pareto Front Size", fontsize=12)  # Set Y-axis label
                    plt.title(f"Pareto Front Size Evolution\\n{base_dataset_name} (run={run}, pop={pop_size})", fontsize=14)  # Set plot title
                    plt.grid(True, linestyle="--", alpha=0.5)  # Add grid
                    plt.tight_layout()  # Adjust layout
                    plot_path = os.path.join(plot_output_dir, "03_pareto_front_size.png")  # Define plot path
                    plt.savefig(plot_path, dpi=150)  # Save figure
                    plt.close()  # Close plot
                    saved_plots.append(plot_path)  # Add to saved plots list
                except Exception:  # If plotting fails
                    plt.close()  # Close plot

            if avg_f1_history:  # If average F1 history exists
                try:  # Try to create plot
                    plt.figure(figsize=(10, 6))  # Create figure
                    plt.plot(generations[: len(avg_f1_history)], avg_f1_history, marker="d", linestyle="-", color="#d62728", linewidth=2, markersize=4)  # Plot average F1 with red line
                    plt.xlabel("Generation", fontsize=12)  # Set X-axis label
                    plt.ylabel("Population Average F1-Score", fontsize=12)  # Set Y-axis label
                    plt.title(f"Population Average F1-Score\\n{base_dataset_name} (run={run}, pop={pop_size})", fontsize=14)  # Set plot title
                    plt.grid(True, linestyle="--", alpha=0.5)  # Add grid
                    plt.tight_layout()  # Adjust layout
                    plot_path = os.path.join(plot_output_dir, "04_avg_f1_evolution.png")  # Define plot path
                    plt.savefig(plot_path, dpi=150)  # Save figure
                    plt.close()  # Close plot
                    saved_plots.append(plot_path)  # Add to saved plots list
                except Exception:  # If plotting fails
                    plt.close()  # Close plot

            if avg_features_history:  # If average features history exists
                try:  # Try to create plot
                    plt.figure(figsize=(10, 6))  # Create figure
                    plt.plot(generations[: len(avg_features_history)], avg_features_history, marker="v", linestyle="-", color="#9467bd", linewidth=2, markersize=4)  # Plot average features with purple line
                    plt.xlabel("Generation", fontsize=12)  # Set X-axis label
                    plt.ylabel("Population Average Feature Count", fontsize=12)  # Set Y-axis label
                    plt.title(f"Population Average Feature Count\\n{base_dataset_name} (run={run}, pop={pop_size})", fontsize=14)  # Set plot title
                    plt.grid(True, linestyle="--", alpha=0.5)  # Add grid
                    plt.tight_layout()  # Adjust layout
                    plot_path = os.path.join(plot_output_dir, "05_avg_feature_count_evolution.png")  # Define plot path
                    plt.savefig(plot_path, dpi=150)  # Save figure
                    plt.close()  # Close plot
                    saved_plots.append(plot_path)  # Add to saved plots list
                except Exception:  # If plotting fails
                    plt.close()  # Close plot

            if hypervolume_history:  # If hypervolume history exists
                try:  # Try to create plot
                    plt.figure(figsize=(10, 6))  # Create figure
                    plt.plot(generations[: len(hypervolume_history)], hypervolume_history, marker="*", linestyle="-", color="#8c564b", linewidth=2, markersize=6)  # Plot hypervolume with brown line
                    plt.xlabel("Generation", fontsize=12)  # Set X-axis label
                    plt.ylabel("Hypervolume", fontsize=12)  # Set Y-axis label
                    plt.title(f"Hypervolume Evolution (Pareto Quality)\\n{base_dataset_name} (run={run}, pop={pop_size})", fontsize=14)  # Set plot title
                    plt.grid(True, linestyle="--", alpha=0.5)  # Add grid
                    plt.tight_layout()  # Adjust layout
                    plot_path = os.path.join(plot_output_dir, "06_hypervolume_evolution.png")  # Define plot path
                    plt.savefig(plot_path, dpi=150)  # Save figure
                    plt.close()  # Close plot
                    saved_plots.append(plot_path)  # Add to saved plots list
                except Exception:  # If plotting fails
                    plt.close()  # Close plot

            if diversity_history:  # If diversity history exists
                try:  # Try to create plot
                    plt.figure(figsize=(10, 6))  # Create figure
                    plt.plot(generations[: len(diversity_history)], diversity_history, marker="p", linestyle="-", color="#e377c2", linewidth=2, markersize=4)  # Plot diversity with pink line
                    plt.xlabel("Generation", fontsize=12)  # Set X-axis label
                    plt.ylabel("Population Diversity (Avg Hamming Distance)", fontsize=12)  # Set Y-axis label
                    plt.title(f"Population Diversity Evolution\\n{base_dataset_name} (run={run}, pop={pop_size})", fontsize=14)  # Set plot title
                    plt.grid(True, linestyle="--", alpha=0.5)  # Add grid
                    plt.tight_layout()  # Adjust layout
                    plot_path = os.path.join(plot_output_dir, "07_diversity_evolution.png")  # Define plot path
                    plt.savefig(plot_path, dpi=150)  # Save figure
                    plt.close()  # Close plot
                    saved_plots.append(plot_path)  # Add to saved plots list
                except Exception:  # If plotting fails
                    plt.close()  # Close plot

            if best_f1_history and best_features_history:  # If both histories exist
                try:  # Try to create plot
                    fig, ax1 = plt.subplots(figsize=(10, 6))  # Create figure with primary axis
                    ax2 = ax1.twinx()  # Create secondary Y-axis

                    line1 = ax1.plot(generations[: len(best_f1_history)], best_f1_history, marker="o", linestyle="-", color="#1f77b4", linewidth=2, markersize=4, label="Best F1-Score")  # Plot F1 on primary axis
                    line2 = ax2.plot(generations[: len(best_features_history)], best_features_history, marker="s", linestyle="--", color="#ff7f0e", linewidth=2, markersize=4, label="Feature Count")  # Plot features on secondary axis

                    ax1.set_xlabel("Generation", fontsize=12)  # Set X-axis label
                    ax1.set_ylabel("Best F1-Score", fontsize=12, color="#1f77b4")  # Set primary Y-axis label
                    ax2.set_ylabel("Number of Selected Features", fontsize=12, color="#ff7f0e")  # Set secondary Y-axis label
                    ax1.tick_params(axis="y", labelcolor="#1f77b4")  # Color primary Y-axis ticks
                    ax2.tick_params(axis="y", labelcolor="#ff7f0e")  # Color secondary Y-axis ticks

                    lines = line1 + line2  # Combine lines for legend
                    labels = [str(l.get_label()) for l in lines]  # Extract labels as strings
                    ax1.legend(lines, labels, loc="best")  # Add legend

                    plt.title(f"Multi-Objective Convergence\\n{base_dataset_name} (run={run}, pop={pop_size})", fontsize=14)  # Set plot title
                    ax1.grid(True, linestyle="--", alpha=0.5)  # Add grid
                    plt.tight_layout()  # Adjust layout
                    plot_path = os.path.join(plot_output_dir, "08_multi_objective_convergence.png")  # Define plot path
                    plt.savefig(plot_path, dpi=150)  # Save figure
                    plt.close()  # Close plot
                    saved_plots.append(plot_path)  # Add to saved plots list
                except Exception:  # If plotting fails
                    plt.close()  # Close plot

            if saved_plots:  # If any plots were saved
                verbose_output(
                    f"{BackgroundColors.GREEN}Generated {len(saved_plots)} convergence plots in {BackgroundColors.CYAN}{plot_output_dir}{Style.RESET_ALL}"
                )  # Notify user

            return saved_plots  # Return list of saved plot paths

        except Exception as e:  # If any error occurs during plotting
            try:  # Try to close any open plots
                plt.close("all")  # Close all plots
            except Exception:  # Ignore errors during cleanup
                pass  # Do nothing
                verbose_output(
                    f"{BackgroundColors.YELLOW}Failed to generate convergence plots: {e}{Style.RESET_ALL}"
                )  # Log warning
                return []  # Return empty list
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def save_run_result(output_dir, state_id, result):
    """
    Save a completed run result so future identical runs can be skipped.

    :param output_dir: base output directory
    :param state_id: deterministic id for run
    :param result: serializable run result
    :return: None
    """
    
    try:
        try:  # Attempt to save the run result
            _, run_path = state_file_paths(output_dir, state_id)  # Get the path for the run state file
            with open(run_path, "wb") as f:  # Open the file for writing in binary mode
                pickle.dump(result, f, protocol=CONFIG["caching"]["pickle_protocol"])  # Serialize and save the result
        except Exception:  # If any error occurs during saving
            pass  # Do nothing
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def cleanup_state_for_id(output_dir, state_id):
    """
    Remove progress files for a finished run/generation (best-effort).

    :param output_dir: base output directory
    :param state_id: deterministic id for run
    :return: None
    """
    
    try:
        try:  # Attempt to clean up state files
            gen_path, run_path = state_file_paths(output_dir, state_id)  # Get paths for generation and run state files
            for p in (gen_path, run_path):  # Iterate over the file paths
                try:  # Safe delete without exists() verify to avoid TOCTOU
                    os.remove(p)  # Try to remove the file directly
                except FileNotFoundError:  # File already gone
                    pass  # Success - file doesn't exist
                except Exception:  # Any other error
                    pass  # Best-effort, ignore
        except Exception:  # If any error occurs during cleanup
            pass  # Do nothing
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


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
    
    try:
        iteration_start_time = time.perf_counter()  # Start tracking total iteration time

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
        best_ind, gens_ran, history_data = run_genetic_algorithm_loop(
            toolbox=toolbox,
            population=population,
            hof=hof,
            X_train=X_train,
            y_train=y_train,
            n_generations=n_generations,
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
        )  # Run GA loop and get generations actually run and comprehensive history data

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

        iteration_elapsed_time = time.perf_counter() - iteration_start_time  # Calculate total iteration time
        metrics_with_iteration_time = metrics + (iteration_elapsed_time,)  # Add total iteration time as 7th element

        try:  # Try to generate comprehensive convergence plots (if function exists)
            if 'generate_convergence_plots' in globals():  # Check if plotting function is available
                generate_convergence_plots(
                    history_data, csv_path, pop_size, run, dataset_name, n_generations=n_generations, cxpb=cxpb, mutpb=mutpb
                )  # Generate all convergence and progress plots
        except Exception as e:  # On any plotting error
            verbose_output(
                f"{BackgroundColors.YELLOW}Failed to generate convergence plots: {e}{Style.RESET_ALL}"
            )  # Log warning

        result = {
            "best_ind": best_ind,
            "metrics": metrics_with_iteration_time,
            "best_features": best_features,
            "history_data": history_data,  # Include comprehensive history data for multi-run aggregation
            "gens_ran": gens_ran,  # Include generations actually executed (for early stopping tracking)
        }  # Build result dict with extended tracking data

        try:  # Try to save run result
            if CONFIG["execution"]["resume_progress"] and state_id is not None:  # If resume is enabled and state_id exists
                save_run_result(output_dir, state_id, result)  # Save the run result
                cleanup_state_for_id(output_dir, state_id)  # Cleanup state files
        except Exception:  # On any saving error
            pass  # Do nothing

        return result  # Return results
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def aggregate_sweep_results(results, min_pop, max_pop, dataset_name):
    """
    Aggregate results per population size and find best.

    :param results: Dict mapping pop_size to runs list.
    :param min_pop: Minimum population size.
    :param max_pop: Maximum population size.
    :param dataset_name: Dataset name.
    :return: Tuple (best_score, best_result, best_metrics, results_dict).
    """
    
    try:
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics

def compute_convergence_generation(history_f1, threshold_pct=0.95):
    """
    Compute the generation at which the algorithm reached a specified percentage of its best F1 score.

    :param history_f1: List of best F1 scores per generation
    :param threshold_pct: Percentage of best score to consider as convergence (default: 0.95 = 95%)
    :return: Generation number where convergence was reached, or len(history_f1) if never reached
    """
    
    try:
        if not history_f1:  # If no history data
            return 0  # Return 0 as fallback

        try:  # Attempt to compute convergence generation
            best_f1 = max(history_f1)  # Find the best F1 score achieved
            if best_f1 <= 0:  # If best F1 is zero or negative
                return len(history_f1)  # Return total generations as fallback

            threshold = best_f1 * threshold_pct  # Compute threshold value
            for gen, f1 in enumerate(history_f1, start=1):  # Iterate through history with 1-based generation numbers
                if f1 >= threshold:  # If this generation reached the threshold
                        return gen  # Return the generation number

                return len(history_f1)  # Return total generations if threshold never reached
        except Exception:  # If calculation fails
            return len(history_f1) if history_f1 else 0  # Return length or 0
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def aggregate_run_metrics(run_result):
    """
    Extract and aggregate metrics from a single run result for comparison table.

    :param run_result: Dict containing run results (best_ind, metrics, best_features, history_data, gens_ran)
    :return: Dict with aggregated metrics or None if data unavailable
    """
    
    try:
        if not run_result:  # If no result provided
            return None  # Return None

        try:  # Attempt to extract and compute metrics
            metrics = run_result.get("metrics", [])  # Get metrics tuple
            history_data = run_result.get("history_data", {})  # Get history data dict
            gens_ran = run_result.get("gens_ran", 0)  # Get generations executed

            best_f1_final = metrics[3] if len(metrics) > 3 else 0.0  # Final F1 score from test evaluation
            features_final = len(run_result.get("best_features", []))  # Final feature count

            if isinstance(history_data, dict):  # If history data is available as dict
                best_f1_history = history_data.get("best_f1", [])  # Best F1 per generation
                best_features_history = history_data.get("best_features", [])  # Best features per generation
                avg_f1_history = history_data.get("avg_f1", [])  # Population avg F1 per generation
                avg_features_history = history_data.get("avg_features", [])  # Population avg features per generation
                pareto_size_history = history_data.get("pareto_size", [])  # Pareto front size per generation
                hypervolume_history = history_data.get("hypervolume", [])  # Hypervolume per generation
                diversity_history = history_data.get("diversity", [])  # Diversity per generation
            else:  # If history data not available or wrong format
                best_f1_history = []  # Empty list
                best_features_history = []  # Empty list
                avg_f1_history = []  # Empty list
                avg_features_history = []  # Empty list
                pareto_size_history = []  # Empty list
                hypervolume_history = []  # Empty list
                diversity_history = []  # Empty list

            avg_population_f1 = np.mean(avg_f1_history) if avg_f1_history else 0.0  # Mean of population averages
            avg_feature_count = np.mean(avg_features_history) if avg_features_history else 0.0  # Mean of feature counts
            pareto_size_final = pareto_size_history[-1] if pareto_size_history else 0  # Final Pareto front size
            hypervolume_final = hypervolume_history[-1] if hypervolume_history else 0.0  # Final hypervolume
            convergence_gen = compute_convergence_generation(best_f1_history, threshold_pct=0.95)  # Generation reaching 95% of best
            convergence_gen_safe = int(convergence_gen) if convergence_gen is not None else 0  # Ensure int type; default 0 when None

            return {
                "best_f1_final": float(best_f1_final),  # Final best F1 score
                "features_final": int(features_final),  # Final feature count
                "avg_population_f1": float(avg_population_f1),  # Average population F1 across generations
                "avg_feature_count": float(avg_feature_count),  # Average feature count across generations
                "pareto_size_final": int(pareto_size_final),  # Final Pareto front size
                "hypervolume_final": float(hypervolume_final),  # Final hypervolume metric
                "convergence_gen": int(convergence_gen_safe),  # Generation reaching convergence (0 if unknown)
                "gens_executed": int(gens_ran),  # Total generations actually executed
                }  # Return aggregated metrics dict

        except Exception:  # If any processing fails
            return None  # Return None
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def generate_run_comparison_table(results_dict, csv_path, dataset_name, min_pop, max_pop, n_generations, cxpb, mutpb):
    """
    Generate a CSV comparison table aggregating metrics across all runs and population sizes.

    :param results_dict: Dict mapping pop_size to {"runs": [...], "avg_metrics": ..., "common_features": ...}
    :param csv_path: Path to the dataset CSV (used to determine output directory)
    :param dataset_name: Name of the dataset
    :param min_pop: Minimum population size tested
    :param max_pop: Maximum population size tested
    :param n_generations: Number of generations configured
    :param cxpb: Crossover probability used
    :param mutpb: Mutation probability used
    :return: Path to the saved comparison CSV file, or None if generation failed
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Generating multi-run comparison table for {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}"
        )  # Log generation start

        try:  # Attempt to generate comparison table
            output_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis"  # Base output directory
            comparison_dir = os.path.join(output_dir, "ga_run_comparisons")  # Subdirectory for comparison tables
            os.makedirs(comparison_dir, exist_ok=True)  # Ensure directory exists

            comparison_rows = []  # List to store comparison data rows
            for pop_size in range(min_pop, max_pop + 1):  # For each population size
                if pop_size not in results_dict:  # If no results for this population size
                    continue  # Skip to next

                runs_list = results_dict[pop_size].get("runs", [])  # Get runs list
                for run_idx, run_result in enumerate(runs_list, start=1):  # For each run (1-based indexing)
                    aggregated = aggregate_run_metrics(run_result)  # Aggregate metrics from this run
                    if aggregated is None:  # If aggregation failed
                        continue  # Skip this run

                    row = {
                        "dataset": dataset_name,  # Dataset name
                        "pop_size": pop_size,  # Population size
                        "run_id": run_idx,  # Run identifier (1-based)
                        "n_generations": n_generations,  # Configured generations
                        "cxpb": cxpb,  # Crossover probability
                        "mutpb": mutpb,  # Mutation probability
                        "best_f1_final": aggregated["best_f1_final"],  # Final best F1 score
                        "features_final": aggregated["features_final"],  # Final feature count
                        "avg_population_f1": aggregated["avg_population_f1"],  # Average population F1
                        "avg_feature_count": aggregated["avg_feature_count"],  # Average feature count
                        "pareto_size_final": aggregated["pareto_size_final"],  # Final Pareto size
                        "hypervolume_final": aggregated["hypervolume_final"],  # Final hypervolume
                        "convergence_gen": aggregated["convergence_gen"],  # Convergence generation
                        "gens_executed": aggregated["gens_executed"],  # Generations executed
                    }  # Build row dict
                    comparison_rows.append(row)  # Add row to list

            if not comparison_rows:  # If no rows generated
                verbose_output(
                    f"{BackgroundColors.YELLOW}No comparison data available to generate table{Style.RESET_ALL}"
                )  # Log warning
                return None  # Return None

            df_comparison = pd.DataFrame(comparison_rows)  # Create DataFrame from rows

            df_comparison = df_comparison.sort_values(["pop_size", "run_id"]).reset_index(drop=True)  # Sort and reset index

            base_dataset_name = safe_filename(os.path.splitext(os.path.basename(csv_path))[0])  # Sanitized dataset name
            comparison_filename = f"{base_dataset_name}_multi_run_comparison.csv"  # Deterministic filename
            comparison_path = os.path.join(comparison_dir, comparison_filename)  # Full path

            df_comparison.to_csv(comparison_path, index=False)  # Save DataFrame to CSV without index

            verbose_output(
                f"{BackgroundColors.GREEN}Saved multi-run comparison table to {BackgroundColors.CYAN}{comparison_path}{Style.RESET_ALL}"
            )  # Log success

            print(f"\n{BackgroundColors.GREEN}{'='*80}{Style.RESET_ALL}")
            print(f"{BackgroundColors.GREEN}Multi-Run Comparison Summary for {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}")
            print(f"{BackgroundColors.GREEN}{'='*80}{Style.RESET_ALL}")
            print(f"{BackgroundColors.GREEN}Total Runs Analyzed: {BackgroundColors.CYAN}{len(comparison_rows)}{Style.RESET_ALL}")
            print(f"{BackgroundColors.GREEN}Population Sizes: {BackgroundColors.CYAN}{min_pop} - {max_pop}{Style.RESET_ALL}")
            print(f"{BackgroundColors.GREEN}Mean Best F1: {BackgroundColors.CYAN}{df_comparison['best_f1_final'].mean():.4f}{Style.RESET_ALL}")
            print(f"{BackgroundColors.GREEN}Std Best F1: {BackgroundColors.CYAN}{df_comparison['best_f1_final'].std():.4f}{Style.RESET_ALL}")
            print(f"{BackgroundColors.GREEN}Mean Final Features: {BackgroundColors.CYAN}{df_comparison['features_final'].mean():.2f}{Style.RESET_ALL}")
            print(f"{BackgroundColors.GREEN}Mean Convergence Gen: {BackgroundColors.CYAN}{df_comparison['convergence_gen'].mean():.1f}{Style.RESET_ALL}")
            print(f"{BackgroundColors.GREEN}{'='*80}{Style.RESET_ALL}\n")

            return comparison_path  # Return path to saved file

        except Exception as e:  # If any error occurs
            verbose_output(
                f"{BackgroundColors.YELLOW}Failed to generate comparison table: {e}{Style.RESET_ALL}"
            )  # Log error
            return None  # Return None
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def generate_multi_run_comparison_plots(results_dict, csv_path, dataset_name, min_pop, max_pop, n_generations, cxpb, mutpb):
    """
    Generate comprehensive multi-run comparison visualization plots.

    Creates overlay plots, distribution plots, bar charts, aggregated convergence
    plots, and optional advanced visualizations for comparing multiple GA runs.

    :param results_dict: Dict mapping pop_size to {"runs": [...], "avg_metrics": ..., "common_features": ...}
    :param csv_path: Path to the dataset CSV (used to determine output directory)
    :param dataset_name: Name of the dataset
    :param min_pop: Minimum population size tested
    :param max_pop: Maximum population size tested
    :param n_generations: Number of generations configured
    :param cxpb: Crossover probability used
    :param mutpb: Mutation probability used
    :return: List of paths to all saved comparison plots
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Generating multi-run comparison plots for {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}"
        )  # Log generation start

        try:  # Attempt to generate all comparison plots
            output_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis"  # Base output directory
            comparison_dir = os.path.join(output_dir, "ga_run_comparisons", "multi_run_plots")  # Subdirectory for comparison plots
            os.makedirs(comparison_dir, exist_ok=True)  # Ensure directory exists

            base_dataset_name = safe_filename(os.path.splitext(os.path.basename(csv_path))[0])  # Sanitized dataset name
            saved_plots = []  # List to track all saved plot paths

            all_runs_data = []  # List to store all runs data for comparison
            for pop_size in range(min_pop, max_pop + 1):  # For each population size
                if pop_size not in results_dict:  # If no results for this population size
                    continue  # Skip to next

                runs_list = results_dict[pop_size].get("runs", [])  # Get runs list
                for run_idx, run_result in enumerate(runs_list, start=1):  # For each run (1-based indexing)
                    history_data = run_result.get("history_data", {})  # Get history data
                    metrics = run_result.get("metrics", [])  # Get final metrics
                    
                    if not isinstance(history_data, dict):  # If history data not available
                        continue  # Skip this run

                    run_info = {
                        "pop_size": pop_size,  # Population size
                        "run_id": run_idx,  # Run identifier
                        "history_data": history_data,  # History data dict
                        "final_f1": metrics[3] if len(metrics) > 3 else 0.0,  # Final F1 score
                        "final_features": len(run_result.get("best_features", [])),  # Final feature count
                    }  # Build run info dict
                    all_runs_data.append(run_info)  # Add to list

            if not all_runs_data:  # If no run data available
                verbose_output(
                    f"{BackgroundColors.YELLOW}No run history data available for comparison plots{Style.RESET_ALL}"
                )  # Log warning
                return []  # Return empty list

            total_runs = len(all_runs_data)  # Count total runs
            verbose_output(
                f"{BackgroundColors.GREEN}Processing {BackgroundColors.CYAN}{total_runs}{BackgroundColors.GREEN} runs for comparison visualization{Style.RESET_ALL}"
            )  # Log run count

            try:  # Try to create overlay plot
                plt.figure(figsize=(12, 7))  # Create figure with larger dimensions
                cmap = plt.cm.get_cmap('tab10' if total_runs <= 10 else 'tab20')  # Get colormap based on number of runs
                colors = [cmap(i / min(total_runs, 20 if total_runs > 10 else 10)) for i in range(total_runs)]  # Generate color list

                for idx, run_data in enumerate(all_runs_data):  # For each run
                    best_f1_history = run_data["history_data"].get("best_f1", [])  # Get best F1 history
                    if not best_f1_history:  # If no history data
                        continue  # Skip this run

                    generations = list(range(1, len(best_f1_history) + 1))  # Generate generation numbers
                    color = colors[idx % len(colors)]  # Select color from palette
                    label = f"Pop{run_data['pop_size']}_Run{run_data['run_id']}"  # Create label

                    plt.plot(generations, best_f1_history, linestyle="-", color=color, linewidth=1.5, alpha=0.7, label=label)  # Plot run

                plt.xlabel("Generation", fontsize=12)  # Set X-axis label
                plt.ylabel("Best F1-Score", fontsize=12)  # Set Y-axis label
                plt.title(f"Multi-Run Best F1-Score Overlay\\n{dataset_name} ({total_runs} runs)", fontsize=14)  # Set plot title
                plt.grid(True, linestyle="--", alpha=0.3)  # Add grid with reduced alpha
                if total_runs <= 15:  # If 15 runs or fewer
                    plt.legend(loc="best", fontsize=8, ncol=2)  # Add legend with 2 columns
                plt.tight_layout()  # Adjust layout
                plot_path = os.path.join(comparison_dir, f"{base_dataset_name}_overlay_best_f1.png")  # Define plot path
                plt.savefig(plot_path, dpi=150)  # Save figure
                plt.close()  # Close plot
                saved_plots.append(plot_path)  # Add to saved plots list
            except Exception:  # If plotting fails
                plt.close()  # Close plot

            try:  # Try to create distribution plot
                final_f1_scores = [run["final_f1"] for run in all_runs_data]  # Extract final F1 scores
                
                plt.figure(figsize=(10, 6))  # Create figure
                plt.hist(final_f1_scores, bins=min(20, max(5, total_runs // 2)), color="#2ca02c", alpha=0.7, edgecolor="black")  # Plot histogram
                plt.axvline(float(np.mean(final_f1_scores)), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(final_f1_scores):.4f}")  # Add mean line
                plt.axvline(float(np.median(final_f1_scores)), color="blue", linestyle="--", linewidth=2, label=f"Median: {np.median(final_f1_scores):.4f}")  # Add median line
                plt.xlabel("Final Best F1-Score", fontsize=12)  # Set X-axis label
                plt.ylabel("Frequency", fontsize=12)  # Set Y-axis label
                plt.title(f"Distribution of Final Best F1-Scores\\n{dataset_name} ({total_runs} runs)", fontsize=14)  # Set plot title
                plt.legend(loc="best")  # Add legend
                plt.grid(True, linestyle="--", alpha=0.3, axis="y")  # Add Y-axis grid
                plt.tight_layout()  # Adjust layout
                plot_path = os.path.join(comparison_dir, f"{base_dataset_name}_dist_final_f1.png")  # Define plot path
                plt.savefig(plot_path, dpi=150)  # Save figure
                plt.close()  # Close plot
                saved_plots.append(plot_path)  # Add to saved plots list
            except Exception:  # If plotting fails
                plt.close()  # Close plot

            try:  # Try to create distribution plot
                final_features = [run["final_features"] for run in all_runs_data]  # Extract final feature counts
                
                plt.figure(figsize=(10, 6))  # Create figure
                plt.hist(final_features, bins=min(20, max(5, total_runs // 2)), color="#ff7f0e", alpha=0.7, edgecolor="black")  # Plot histogram
                plt.axvline(float(np.mean(final_features)), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(final_features):.1f}")  # Add mean line
                plt.axvline(float(np.median(final_features)), color="blue", linestyle="--", linewidth=2, label=f"Median: {np.median(final_features):.0f}")  # Add median line
                plt.xlabel("Final Feature Count", fontsize=12)  # Set X-axis label
                plt.ylabel("Frequency", fontsize=12)  # Set Y-axis label
                plt.title(f"Distribution of Final Feature Counts\\n{dataset_name} ({total_runs} runs)", fontsize=14)  # Set plot title
                plt.legend(loc="best")  # Add legend
                plt.grid(True, linestyle="--", alpha=0.3, axis="y")  # Add Y-axis grid
                plt.tight_layout()  # Adjust layout
                plot_path = os.path.join(comparison_dir, f"{base_dataset_name}_dist_final_features.png")  # Define plot path
                plt.savefig(plot_path, dpi=150)  # Save figure
                plt.close()  # Close plot
                saved_plots.append(plot_path)  # Add to saved plots list
            except Exception:  # If plotting fails
                plt.close()  # Close plot

            try:  # Try to create distribution plot
                final_hypervolumes = []  # List to store hypervolume values
                for run in all_runs_data:  # For each run
                    hv_history = run["history_data"].get("hypervolume", [])  # Get hypervolume history
                    if hv_history:  # If hypervolume data exists
                        final_hypervolumes.append(hv_history[-1])  # Add final hypervolume

                if final_hypervolumes:  # If any hypervolume data collected
                    plt.figure(figsize=(10, 6))  # Create figure
                    plt.hist(final_hypervolumes, bins=min(20, max(5, len(final_hypervolumes) // 2)), color="#8c564b", alpha=0.7, edgecolor="black")  # Plot histogram
                    plt.axvline(float(np.mean(final_hypervolumes)), color="red", linestyle="--", linewidth=2, label=f"Mean: {np.mean(final_hypervolumes):.4f}")  # Add mean line
                    plt.axvline(float(np.median(final_hypervolumes)), color="blue", linestyle="--", linewidth=2, label=f"Median: {np.median(final_hypervolumes):.4f}")  # Add median line
                    plt.xlabel("Final Hypervolume", fontsize=12)  # Set X-axis label
                    plt.ylabel("Frequency", fontsize=12)  # Set Y-axis label
                    plt.title(f"Distribution of Final Hypervolume\\n{dataset_name} ({len(final_hypervolumes)} runs)", fontsize=14)  # Set plot title
                    plt.legend(loc="best")  # Add legend
                    plt.grid(True, linestyle="--", alpha=0.3, axis="y")  # Add Y-axis grid
                    plt.tight_layout()  # Adjust layout
                    plot_path = os.path.join(comparison_dir, f"{base_dataset_name}_dist_final_hypervolume.png")  # Define plot path
                    plt.savefig(plot_path, dpi=150)  # Save figure
                    plt.close()  # Close plot
                    saved_plots.append(plot_path)  # Add to saved plots list
            except Exception:  # If plotting fails
                plt.close()  # Close plot

            try:  # Try to create bar chart
                run_labels = [f"P{run['pop_size']}_R{run['run_id']}" for run in all_runs_data]  # Create run labels
                final_f1_scores = [run["final_f1"] for run in all_runs_data]  # Extract final F1 scores
                
                plt.figure(figsize=(max(12, total_runs * 0.4), 6))  # Create figure with width scaled to number of runs
                bars = plt.bar(range(len(run_labels)), final_f1_scores, color="#1f77b4", alpha=0.7, edgecolor="black")  # Create bar chart
                
                sorted_f1 = sorted(final_f1_scores)  # Sort F1 scores
                threshold_high = sorted_f1[int(0.75 * len(sorted_f1))] if len(sorted_f1) > 4 else max(sorted_f1)  # Top 25% threshold
                threshold_low = sorted_f1[int(0.25 * len(sorted_f1))] if len(sorted_f1) > 4 else min(sorted_f1)  # Bottom 25% threshold
                
                for idx, (bar, f1) in enumerate(zip(bars, final_f1_scores)):  # For each bar
                    if f1 >= threshold_high:  # If in top 25%
                        bar.set_color("#2ca02c")  # Green
                    elif f1 <= threshold_low:  # If in bottom 25%
                        bar.set_color("#d62728")  # Red

                plt.xlabel("Run ID", fontsize=12)  # Set X-axis label
                plt.ylabel("Final Best F1-Score", fontsize=12)  # Set Y-axis label
                plt.title(f"Final Best F1-Score per Run\\n{dataset_name} ({total_runs} runs)", fontsize=14)  # Set plot title
                plt.xticks(range(len(run_labels)), run_labels, rotation=45, ha="right", fontsize=8)  # Set X-axis ticks with rotation
                plt.grid(True, linestyle="--", alpha=0.3, axis="y")  # Add Y-axis grid
                plt.tight_layout()  # Adjust layout
                plot_path = os.path.join(comparison_dir, f"{base_dataset_name}_bar_final_f1.png")  # Define plot path
                plt.savefig(plot_path, dpi=150)  # Save figure
                plt.close()  # Close plot
                saved_plots.append(plot_path)  # Add to saved plots list
            except Exception:  # If plotting fails
                plt.close()  # Close plot

            try:  # Try to create bar chart
                run_labels = [f"P{run['pop_size']}_R{run['run_id']}" for run in all_runs_data]  # Create run labels
                final_features = [run["final_features"] for run in all_runs_data]  # Extract final feature counts
                
                plt.figure(figsize=(max(12, total_runs * 0.4), 6))  # Create figure with width scaled to number of runs
                plt.bar(range(len(run_labels)), final_features, color="#ff7f0e", alpha=0.7, edgecolor="black")  # Create bar chart
                plt.xlabel("Run ID", fontsize=12)  # Set X-axis label
                plt.ylabel("Final Feature Count", fontsize=12)  # Set Y-axis label
                plt.title(f"Final Feature Count per Run\\n{dataset_name} ({total_runs} runs)", fontsize=14)  # Set plot title
                plt.xticks(range(len(run_labels)), run_labels, rotation=45, ha="right", fontsize=8)  # Set X-axis ticks with rotation
                plt.grid(True, linestyle="--", alpha=0.3, axis="y")  # Add Y-axis grid
                plt.tight_layout()  # Adjust layout
                plot_path = os.path.join(comparison_dir, f"{base_dataset_name}_bar_final_features.png")  # Define plot path
                plt.savefig(plot_path, dpi=150)  # Save figure
                plt.close()  # Close plot
                saved_plots.append(plot_path)  # Add to saved plots list
            except Exception:  # If plotting fails
                plt.close()  # Close plot

            try:  # Try to create bar chart
                run_info_with_hv = []  # List to store runs with hypervolume data
                for run in all_runs_data:  # For each run
                    hv_history = run["history_data"].get("hypervolume", [])  # Get hypervolume history
                    if hv_history:  # If hypervolume data exists
                        run_info_with_hv.append({
                            "label": f"P{run['pop_size']}_R{run['run_id']}",  # Run label
                            "hypervolume": hv_history[-1],  # Final hypervolume
                        })  # Add to list

                if run_info_with_hv:  # If any runs have hypervolume data
                    run_labels_hv = [r["label"] for r in run_info_with_hv]  # Extract labels
                    hypervolumes = [r["hypervolume"] for r in run_info_with_hv]  # Extract hypervolumes
                    
                    plt.figure(figsize=(max(12, len(run_labels_hv) * 0.4), 6))  # Create figure
                    plt.bar(range(len(run_labels_hv)), hypervolumes, color="#8c564b", alpha=0.7, edgecolor="black")  # Create bar chart
                    plt.xlabel("Run ID", fontsize=12)  # Set X-axis label
                    plt.ylabel("Final Hypervolume", fontsize=12)  # Set Y-axis label
                    plt.title(f"Final Hypervolume per Run\\n{dataset_name} ({len(run_labels_hv)} runs)", fontsize=14)  # Set plot title
                    plt.xticks(range(len(run_labels_hv)), run_labels_hv, rotation=45, ha="right", fontsize=8)  # Set X-axis ticks
                    plt.grid(True, linestyle="--", alpha=0.3, axis="y")  # Add Y-axis grid
                    plt.tight_layout()  # Adjust layout
                    plot_path = os.path.join(comparison_dir, f"{base_dataset_name}_bar_final_hypervolume.png")  # Define plot path
                    plt.savefig(plot_path, dpi=150)  # Save figure
                    plt.close()  # Close plot
                    saved_plots.append(plot_path)  # Add to saved plots list
            except Exception:  # If plotting fails
                plt.close()  # Close plot

            try:  # Try to create aggregated convergence plot
                all_f1_histories = []  # List to store all F1 histories
                max_gens = 0  # Track maximum generation count
                
                for run in all_runs_data:  # For each run
                    best_f1_history = run["history_data"].get("best_f1", [])  # Get best F1 history
                    if best_f1_history:  # If history exists
                        all_f1_histories.append(best_f1_history)  # Add to list
                        max_gens = max(max_gens, len(best_f1_history))  # Update max generations

                if all_f1_histories and max_gens > 0:  # If any F1 histories collected
                    aligned_histories = []  # List to store aligned histories
                    for history in all_f1_histories:  # For each history
                        if len(history) < max_gens:  # If shorter than max
                            padded = history + [history[-1]] * (max_gens - len(history))  # Pad with last value
                            aligned_histories.append(padded)  # Add padded history
                        else:  # If already at max length
                            aligned_histories.append(history)  # Add as is

                    histories_array = np.array(aligned_histories)  # Convert to numpy array
                    mean_f1 = np.mean(histories_array, axis=0)  # Compute mean per generation
                    std_f1 = np.std(histories_array, axis=0)  # Compute std dev per generation
                    generations = list(range(1, max_gens + 1))  # Generate generation numbers

                    plt.figure(figsize=(12, 7))  # Create figure
                    plt.plot(generations, mean_f1, color="#1f77b4", linewidth=2.5, label="Mean Best F1")  # Plot mean
                    plt.fill_between(generations, mean_f1 - std_f1, mean_f1 + std_f1, color="#1f77b4", alpha=0.2, label="± 1 Std Dev")  # Fill std dev region
                    plt.xlabel("Generation", fontsize=12)  # Set X-axis label
                    plt.ylabel("Best F1-Score", fontsize=12)  # Set Y-axis label
                    plt.title(f"Aggregated Best F1-Score Convergence (Mean ± Std Dev)\\n{dataset_name} ({len(all_f1_histories)} runs)", fontsize=14)  # Set plot title
                    plt.legend(loc="best", fontsize=10)  # Add legend
                    plt.grid(True, linestyle="--", alpha=0.3)  # Add grid
                    plt.tight_layout()  # Adjust layout
                    plot_path = os.path.join(comparison_dir, f"{base_dataset_name}_aggregated_convergence.png")  # Define plot path
                    plt.savefig(plot_path, dpi=150)  # Save figure
                    plt.close()  # Close plot
                    saved_plots.append(plot_path)  # Add to saved plots list
            except Exception:  # If plotting fails
                plt.close()  # Close plot

            try:  # Try to create scatter plot
                final_f1_scores = [run["final_f1"] for run in all_runs_data]  # Extract final F1 scores
                final_features = [run["final_features"] for run in all_runs_data]  # Extract final feature counts
                colors_pop = [run["pop_size"] for run in all_runs_data]  # Extract population sizes for coloring
                
                plt.figure(figsize=(10, 7))  # Create figure
                scatter = plt.scatter(final_features, final_f1_scores, c=colors_pop, cmap="viridis", s=100, alpha=0.6, edgecolors="black", linewidth=1)  # Create scatter plot
                plt.colorbar(scatter, label="Population Size")  # Add colorbar
                plt.xlabel("Final Feature Count", fontsize=12)  # Set X-axis label
                plt.ylabel("Final Best F1-Score", fontsize=12)  # Set Y-axis label
                plt.title(f"Best F1-Score vs Feature Count\\n{dataset_name} ({total_runs} runs)", fontsize=14)  # Set plot title
                plt.grid(True, linestyle="--", alpha=0.3)  # Add grid
                plt.tight_layout()  # Adjust layout
                plot_path = os.path.join(comparison_dir, f"{base_dataset_name}_scatter_f1_vs_features.png")  # Define plot path
                plt.savefig(plot_path, dpi=150)  # Save figure
                plt.close()  # Close plot
                saved_plots.append(plot_path)  # Add to saved plots list
            except Exception:  # If plotting fails
                plt.close()  # Close plot

            try:  # Try to create correlation heatmap
                metrics_data = []  # List to store metrics for each run
                for run in all_runs_data:  # For each run
                    history = run["history_data"]  # Get history data
                    hv_history = history.get("hypervolume", [])  # Get hypervolume history
                    div_history = history.get("diversity", [])  # Get diversity history
                    
                    metrics_row = {
                        "Final F1": run["final_f1"],  # Final F1 score
                        "Final Features": run["final_features"],  # Final feature count
                        "Final Hypervolume": hv_history[-1] if hv_history else np.nan,  # Final hypervolume
                        "Final Diversity": div_history[-1] if div_history else np.nan,  # Final diversity
                    }  # Build metrics row
                    metrics_data.append(metrics_row)  # Add to list

                df_metrics = pd.DataFrame(metrics_data)  # Create DataFrame
                df_metrics = df_metrics.dropna(axis=1, how="all")  # Drop columns with all NaN values
                
                if df_metrics.shape[1] >= 2:  # If at least 2 metrics available
                    correlation_matrix = df_metrics.corr()  # Compute correlation matrix
                    
                    plt.figure(figsize=(8, 6))  # Create figure
                    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, vmin=-1, vmax=1, square=True, linewidths=1, cbar_kws={"shrink": 0.8})  # Create heatmap
                    plt.title(f"Metric Correlation Heatmap\\n{dataset_name} ({total_runs} runs)", fontsize=14)  # Set plot title
                    plt.tight_layout()  # Adjust layout
                    plot_path = os.path.join(comparison_dir, f"{base_dataset_name}_correlation_heatmap.png")  # Define plot path
                    plt.savefig(plot_path, dpi=150)  # Save figure
                    plt.close()  # Close plot
                    saved_plots.append(plot_path)  # Add to saved plots list
            except Exception:  # If plotting fails
                plt.close()  # Close plot

            try:  # Try to create boxplot
                if max_pop > min_pop:  # If multiple population sizes tested
                    pop_sizes = sorted(set(run["pop_size"] for run in all_runs_data))  # Get unique population sizes
                    f1_by_pop = {}  # Dict to store F1 scores by population size
                    
                    for pop in pop_sizes:  # For each population size
                        f1_by_pop[pop] = [run["final_f1"] for run in all_runs_data if run["pop_size"] == pop]  # Collect F1 scores
                    
                    plt.figure(figsize=(10, 6))  # Create figure
                    box_positions = list(range(1, len(f1_by_pop) + 1))  # Create box positions
                    plt.boxplot(list(f1_by_pop.values()), positions=box_positions, patch_artist=True, boxprops=dict(facecolor="#1f77b4", alpha=0.7))  # Create boxplot
                    plt.xticks(box_positions, [f"Pop {p}" for p in f1_by_pop.keys()], fontsize=10)  # Set X-axis tick labels
                    plt.xlabel("Population Size", fontsize=12)  # Set X-axis label
                    plt.ylabel("Final Best F1-Score", fontsize=12)  # Set Y-axis label
                    plt.title(f"F1-Score Distribution by Population Size\\n{dataset_name}", fontsize=14)  # Set plot title
                    plt.grid(True, linestyle="--", alpha=0.3, axis="y")  # Add Y-axis grid
                    plt.tight_layout()  # Adjust layout
                    plot_path = os.path.join(comparison_dir, f"{base_dataset_name}_boxplot_f1_by_pop.png")  # Define plot path
                    plt.savefig(plot_path, dpi=150)  # Save figure
                    plt.close()  # Close plot
                    saved_plots.append(plot_path)  # Add to saved plots list
            except Exception:  # If plotting fails
                plt.close()  # Close plot

            if saved_plots:  # If any plots were saved
                verbose_output(
                    f"{BackgroundColors.GREEN}Generated {len(saved_plots)} multi-run comparison plots in {BackgroundColors.CYAN}{comparison_dir}{Style.RESET_ALL}"
                )  # Notify user
                print(f"\n{BackgroundColors.GREEN}{'='*80}{Style.RESET_ALL}")
                print(f"{BackgroundColors.GREEN}Multi-Run Visualization Summary{Style.RESET_ALL}")
                print(f"{BackgroundColors.GREEN}{'='*80}{Style.RESET_ALL}")
                print(f"{BackgroundColors.GREEN}Total Plots Generated: {BackgroundColors.CYAN}{len(saved_plots)}{Style.RESET_ALL}")
                print(f"{BackgroundColors.GREEN}Output Directory: {BackgroundColors.CYAN}{comparison_dir}{Style.RESET_ALL}")
                print(f"{BackgroundColors.GREEN}{'='*80}{Style.RESET_ALL}\n")

            return saved_plots  # Return list of saved plot paths

        except Exception as e:  # If any error occurs during generation
            try:  # Try to close any open plots
                plt.close("all")  # Close all plots
            except Exception:  # Ignore errors during cleanup
                pass  # Do nothing
            verbose_output(
                f"{BackgroundColors.YELLOW}Failed to generate multi-run comparison plots: {e}{Style.RESET_ALL}"
            )  # Log warning
            return []  # Return empty list
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def print_metrics(metrics):
    """
    Print performance metrics including multi-objective fitness values.

    :param metrics: Dictionary or tuple containing evaluation metrics.
    :return: None
    """
    
    try:
        if not metrics:  # If metrics is None or empty
            return  # Do nothing

        cv_acc, cv_prec, cv_rec, cv_f1, cv_fpr, cv_fnr, test_acc, test_prec, test_rec, test_f1, test_fpr, test_fnr = metrics[:12] if len(metrics) >= 12 else metrics  # Unpack first 12 metrics (backward compatible)
        num_features = metrics[12] if len(metrics) > 12 else 0  # Extract number of selected features if present
        print(
            f"\n{BackgroundColors.GREEN}CV Performance Metrics for the Random Forest Classifier using the best feature subset ({int(num_features)} features):{Style.RESET_ALL}"
        )
        print(f"   {BackgroundColors.GREEN}Accuracy: {BackgroundColors.CYAN}{truncate_value(cv_acc)}{Style.RESET_ALL}")
        print(f"   {BackgroundColors.GREEN}Precision: {BackgroundColors.CYAN}{truncate_value(cv_prec)}{Style.RESET_ALL}")
        print(f"   {BackgroundColors.GREEN}Recall: {BackgroundColors.CYAN}{truncate_value(cv_rec)}{Style.RESET_ALL}")
        print(f"   {BackgroundColors.GREEN}F1-Score: {BackgroundColors.CYAN}{truncate_value(cv_f1)}{Style.RESET_ALL}")
        print(f"   {BackgroundColors.GREEN}False Positive Rate (FPR): {BackgroundColors.CYAN}{truncate_value(cv_fpr)}{Style.RESET_ALL}")
        print(f"   {BackgroundColors.GREEN}False Negative Rate (FNR): {BackgroundColors.CYAN}{truncate_value(cv_fnr)}{Style.RESET_ALL}")
        print(
            f"\n{BackgroundColors.GREEN}Test Performance Metrics for the Random Forest Classifier using the best feature subset ({int(num_features)} features):{Style.RESET_ALL}"
        )
        print(f"   {BackgroundColors.GREEN}Accuracy: {BackgroundColors.CYAN}{truncate_value(test_acc)}{Style.RESET_ALL}")
        print(f"   {BackgroundColors.GREEN}Precision: {BackgroundColors.CYAN}{truncate_value(test_prec)}{Style.RESET_ALL}")
        print(f"   {BackgroundColors.GREEN}Recall: {BackgroundColors.CYAN}{truncate_value(test_rec)}{Style.RESET_ALL}")
        print(f"   {BackgroundColors.GREEN}F1-Score: {BackgroundColors.CYAN}{truncate_value(test_f1)}{Style.RESET_ALL}")
        print(f"   {BackgroundColors.GREEN}False Positive Rate (FPR): {BackgroundColors.CYAN}{truncate_value(test_fpr)}{Style.RESET_ALL}")
        print(f"   {BackgroundColors.GREEN}False Negative Rate (FNR): {BackgroundColors.CYAN}{truncate_value(test_fnr)}{Style.RESET_ALL}")
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def extract_rfe_ranking(csv_path):
    """
    Extract RFE rankings from the RFE results file.

    :param csv_path: Path to the original CSV file for saving outputs.
    :return: Dictionary of feature names and their RFE rankings.
    """
    
    try:
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics

def extract_elapsed_from_metrics(metrics, index=6):
    """
    Safely extract an elapsed-time value from a metrics tuple.

    :param metrics: Metrics tuple (acc, prec, rec, f1, fpr, fnr, elapsed) or None
    :param index: Index within the tuple where elapsed time is expected (default 6)
    :return: elapsed value (float) if available, otherwise None
    """
    
    try:
        if not metrics:  # If metrics is None or falsy
            return None  # Return None
        try:  # Try to extract elapsed time
            if isinstance(metrics, (list, tuple)) and len(metrics) > index:  # Verify index is valid
                return metrics[index]  # Return elapsed time
        except Exception:  # On any error
            pass  # Ignore errors

        return None  # Return None if extraction fails
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


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

    try:
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def normalize_elapsed_column_df(df):
    """
    Normalize elapsed time column name to "elapsed_time_s" if the legacy
    "elapsed_time" column is present.

    :param df: pandas DataFrame
    :return: DataFrame with "elapsed_time_s" column
    """

    try:
        if (
            "elapsed_time" in df.columns and "elapsed_time_s" not in df.columns
        ):  # If legacy column present and new column missing
            df["elapsed_time_s"] = df["elapsed_time"]  # Copy legacy elapsed_time into the new elapsed_time_s column
            df.drop(columns=["elapsed_time"], inplace=True)  # Remove the legacy elapsed_time column to avoid duplication
        return df  # Return DataFrame with normalized elapsed time column
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_existing_results(csv_out):
    """
    Load an existing consolidated CSV if present, returning an empty
    DataFrame on error or when file is missing.

    :param csv_out: path to consolidated CSV
    :return: pandas.DataFrame
    """

    try:
        if os.path.exists(csv_out):  # If the consolidated CSV exists
            try:  # Try to read the file into a DataFrame
                df = pd.read_csv(csv_out, dtype=object)  # Read CSV preserving types as object
                df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names
                return df  # Return the DataFrame
            except Exception:  # On any read error
                return pd.DataFrame()  # Return empty DataFrame as a fallback
        return pd.DataFrame()  # File not present — return empty DataFrame
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def merge_replace_existing(df_existing, df_new):
    """
    Merge "df_new" into "df_existing" using replace-by-"dataset_path"
    semantics: any existing rows whose "dataset_path" appears in "df_new"
    are removed before appending "df_new".

    :param df_existing: existing DataFrame (may be empty)
    :param df_new: incoming DataFrame with new rows
    :return: merged DataFrame
    """

    try:
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


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

        cores = psutil.cpu_count(logical=False) if psutil else None  # Physical core count
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1) if psutil else None  # Total RAM in GB
        os_name = f"{platform.system()} {platform.release()}"  # OS name + version

        return {  # Build final dictionary
            "cpu_model": cpu_model,  # CPU model string
            "cores": cores,  # Physical cores
            "ram_gb": ram_gb,  # RAM in gigabytes
            "os": os_name,  # Operating system
        }
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def populate_hardware_column(df, column_name="hardware"):
    """
    Populate "df[column_name]" with a readable hardware description built from
    "get_hardware_specifications()". On failure the column will be set to None.

    :param df: pandas.DataFrame to modify in-place
    :param column_name: Name of the column to set (default: "hardware")
    :return: The modified DataFrame
    """

    try:
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def ensure_expected_columns(df_combined, columns):
    """
    Ensure the expected columns exist on the combined DataFrame; add
    missing columns with None values.

    :param df_combined: pandas.DataFrame
    :param columns: list of expected column names
    :return: DataFrame with ensured columns
    """

    try:
        for column in columns:  # Iterate over expected column names
            if column not in df_combined.columns:  # If the column is missing from the DataFrame
                df_combined[column] = None  # Add the missing column and fill with None
        return df_combined  # Return DataFrame with ensured columns
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def run_index_sort(val):
    """
    Convert run_index values into sortable numeric keys where "best"
    sorts before numeric indices.

    :param val: run_index value (string or numeric)
    :return: numeric sort key
    """

    try:
        try:  # Try to normalize and parse the run index
            s = str(val).strip()  # Convert the value to string and strip whitespace
            if s.lower() == "best":  # If value is the literal "best"
                return -1  # Force "best" to sort before numeric indices
            return int(float(s))  # Convert numeric-like strings to integer for sorting
        except Exception:  # On any parsing error
            return 10**9  # Use a very large number to push malformed values to the end
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def sort_run_index_first(df_combined):
    """
    Sort by "dataset", "dataset_path" and a numeric-coded "run_index"
    where the string "best" is forced to come before numeric runs.

    :param df_combined: pandas.DataFrame
    :return: sorted DataFrame
    """

    try:
        df_combined["run_index_sort"] = df_combined["run_index"].apply(run_index_sort)  # Create temporary numeric sort key
        df_combined.sort_values(
            by=["dataset", "dataset_path", "run_index_sort"], inplace=True, ascending=[True, True, True]
        )  # Sort by dataset, path, then run order
        df_combined.drop(columns=["run_index_sort"], inplace=True)  # Remove temporary sort key column
        return df_combined  # Return the sorted DataFrame
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def write_consolidated_csv(rows, output_dir):
    """
    Write the consolidated GA results rows to a CSV file inside "output_dir".

    :param rows: List of dictionaries representing rows.
    :param output_dir: Directory where "Genetic_Algorithm_Results.csv" will be saved.
    :return: None
    """

    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Writing consolidated Genetic Algorithm results CSV.{Style.RESET_ALL}"
        )  # Output the verbose message

        with csv_write_lock:  # Thread-safe CSV write operation
            try:  # Try to write the consolidated CSV
                os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

                df_new = pd.DataFrame(rows)  # Create DataFrame from provided rows

                df_new = normalize_elapsed_column_df(df_new)  # Normalize legacy elapsed_time to elapsed_time_s in new rows

                csv_out = os.path.join(output_dir, "Genetic_Algorithm_Results.csv")  # Build path for consolidated CSV

                if os.path.exists(csv_out):  # If the consolidated CSV already exists, we need to merge with existing data
                    df_existing = pd.read_csv(csv_out, dtype=str)  # Read existing CSV as strings to avoid type issues
                    if "timestamp" not in df_existing.columns:  # If the existing CSV is missing the "timestamp" column, add it with a default value based on file modification time
                        print(
                            f"{BackgroundColors.YELLOW}Warning: Existing CSV missing 'timestamp' column. Adding default timestamps.{Style.RESET_ALL}"
                        )
                        mtime = os.path.getmtime(csv_out)  # Get file modification time
                        back_ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d_%H_%M_%S")  # Format timestamp based on file modification time
                        df_existing["timestamp"] = back_ts  # Add the timestamp column with the default value
                    for c in CONFIG["export"]["results_csv_columns"]:  # Ensure all expected columns are present in the existing DataFrame, adding any missing ones with None values
                        if c not in df_existing.columns:  # If an expected column is missing
                            df_existing[c] = None  # Add the missing column with None values
                    df_combined = pd.concat([df_existing[CONFIG["export"]["results_csv_columns"]], df_new], ignore_index=True, sort=False)  # Combine existing and new DataFrames, ensuring column order
                    try:  # Try to sort by timestamp if the column exists and is properly formatted
                        df_combined["timestamp_dt"] = pd.to_datetime(df_combined["timestamp"], format="%Y-%m-%d_%H_%M_%S", errors="coerce")  # Parse timestamp into datetime, coercing errors to NaT
                        df_combined = df_combined.sort_values(by="timestamp_dt", ascending=False)  # Sort by the parsed datetime column in descending order (newest first)
                        df_combined = df_combined.drop(columns=["timestamp_dt"])  # Remove the temporary datetime column after sorting
                    except Exception:  # If sorting by timestamp fails (e.g., due to parsing issues), fall back to sorting by dataset and run_index
                        df_combined = df_combined.sort_values(by="timestamp", ascending=False)  # Sort by timestamp as a string (may not be perfectly chronological if formatting is inconsistent)
                    df_out = df_combined.reset_index(drop=True)  # Reset index after sorting
                else:  # No existing CSV, so we can just use the new DataFrame as is
                    df_out = df_new  # Use the new DataFrame directly

                df_out = populate_hardware_column(df_out, column_name="hardware")  # Populate hardware column using system specs

                df_out = ensure_expected_columns(df_out, CONFIG["export"]["results_csv_columns"])  # Add any missing expected columns with None values

                df_out = df_out[CONFIG["export"]["results_csv_columns"]]  # Reorder columns into the canonical order

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
            except Exception as e:  # On any error during the write process
                print(
                    f"{BackgroundColors.RED}Failed to write consolidated GA CSV: {str(e)}{Style.RESET_ALL}"
                )  # Print failure message with exception
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def determine_best_features_and_ranking(best_ind, feature_names, csv_path):
    """
    Determine the selected feature names for a binary GA individual and
    extract RFE rankings from existing results.

    :param best_ind: Sequence (list/array) representing the GA individual (0/1 mask).
    :param feature_names: List of feature names aligned with "best_ind".
    :param csv_path: Path to the dataset CSV (used to locate RFE summary files).
    :return: Tuple "(best_feats, ranking)" where "best_feats" is a list of selected
             feature names and "ranking" is the RFE ranking dictionary (may be empty).
    """

    try:
        best_feats = [f for f, bit in zip(feature_names if feature_names is not None else [], best_ind) if bit == 1]  # Build list of features selected by the binary mask
        ranking = extract_rfe_ranking(csv_path)  # Try to extract an existing RFE ranking for additional metadata
        print(f"\n{BackgroundColors.GREEN}Best features subset found: {BackgroundColors.CYAN}{best_feats}{Style.RESET_ALL}")  # Verbose output for user
        return best_feats, ranking  # Return selected features list and ranking dictionary
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def determine_rf_metrics(metrics_in):
    """
    Normalize a metrics tuple/sequence to the expected RF-metrics slice used
    downstream by "save_results". The consolidated CSV expects up to 12
    metric values (cv + test metrics); this helper ensures the returned
    value is either a 12-element-like slice or "None" when not available.

    :param metrics_in: Metrics sequence or None.
    :return: Sliced metrics (first 12 elements) or None.
    """

    try:
        if metrics_in is not None:  # Verify if metrics are provided
            return metrics_in[:12] if len(metrics_in) >= 12 else None  # Only keep first 12 values if available (cv metrics + test metrics)
        return None  # Return None if no metrics provided
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


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
    :return: RF metrics tuple (possibly produced by "evaluate_individual") or the original "rf_m".
    """

    try:
        if rf_m is None and X_test is not None and y_test is not None:  # Only perform evaluation if metrics are missing and a test set is available
            return evaluate_individual(best_ind_local, X, y, X_test)  # Evaluate individual on test set to generate metrics
        return rf_m  # Return existing metrics if already available
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


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
    :param y: Training labels (used to compute "n_train").
    :param y_test: Optional test labels (used to compute "n_test").
    :param cxpb: Crossover probability used (included in base row).
    :param mutpb: Mutation probability used (included in base row).
    :return: Tuple containing computed values and prepared paths:
             (n_train, n_test, test_frac, elapsed_base, base_row,
              models_dir, ts, base_name, model_path, scaler_path,
              features_path, params_path)
    """

    try:
        n_train_local = len(y) if y is not None else None  # Compute counts and test fraction
        n_test_local = len(y_test) if y_test is not None else None  # Compute test sample count
        test_frac_local = None  # Initialize test fraction as None
        if n_train_local is not None and n_test_local is not None and (n_train_local + n_test_local) > 0:  # Verify both counts are valid
            test_frac_local = float(n_test_local) / float(n_train_local + n_test_local)  # Calculate test set fraction

        elapsed_base = extract_elapsed_from_metrics(rf_metrics)  # Extract elapsed seconds from RF metrics (if present)
        base_row_local = build_base_row(  # Build the canonical base row used by consolidated CSV
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

        models_dir_local = f"{os.path.dirname(csv_path)}/Feature_Analysis/Genetic_Algorithm/Models/"  # Prepare model artifact directory and filenames
        os.makedirs(models_dir_local, exist_ok=True)  # Create models directory if it doesn't exist
        ts = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")  # Generate timestamp string for filenames
        base_name_local = re.sub(r"[^A-Za-z0-9_.-]+", "_", os.path.splitext(os.path.basename(csv_path))[0])  # Sanitize dataset name for use in filename
        model_path_local = os.path.join(models_dir_local, f"GA-{base_name_local}-{ts}-model.joblib")  # Construct full path for model file
        scaler_path_local = os.path.join(models_dir_local, f"GA-{base_name_local}-{ts}-scaler.joblib")  # Construct full path for scaler file
        features_path_local = os.path.join(models_dir_local, f"GA-{base_name_local}-{ts}-features.json")  # Construct full path for features file
        params_path_local = os.path.join(models_dir_local, f"GA-{base_name_local}-{ts}-params.json")  # Construct full path for params file

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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def train_and_save_final_model(best_feats_local, X, y, feature_names, X_test, model_path_local, scaler_path_local, features_path_local, params_path_local):
    """
    Train a RandomForestClassifier on the selected feature subset, persist the
    trained model, scaler, selected-features list and model parameters.

    :param best_feats_local: List of selected feature names.
    :param X: Full feature matrix (DataFrame or ndarray).
    :param y: Training labels.
    :param feature_names: List of all feature names corresponding to columns in "X".
    :param X_test: Optional test feature matrix (used to select test columns consistently).
    :param model_path_local: Path where the trained model ".joblib" will be saved.
    :param scaler_path_local: Path where the fitted scaler ".joblib" will be saved.
    :param features_path_local: Path where the JSON file with "best_feats_local" will be saved.
    :param params_path_local: Path where the model "get_params()" JSON will be saved.
    :return: Tuple "(model_local, model_params_local, training_time_local, X_test_selected_local)".
    """

    try:
        df_features_local = prepare_feature_dataframe(X, feature_names)  # Convert feature matrix to DataFrame with column names
        scaler_local = StandardScaler()  # Initialize standard scaler for feature normalization
        start_scale_local = time.perf_counter()
        X_scaled_local = scaler_local.fit_transform(df_features_local.values)  # Fit scaler on features and transform to normalized values
        scaling_time_local = time.perf_counter() - start_scale_local
        try:  # Attach scaling time to scaler_local using setattr to avoid Pylance attribute warning
            setattr(scaler_local, "_scaling_time", round(float(scaling_time_local), 6))  # Store scaling time on scaler
        except Exception:  # If setting attribute fails, continue without raising
            pass  # Preserve prior silent failure behavior
        sel_indices_local = [i for i, f in enumerate(feature_names) if f in best_feats_local]  # Get indices of selected features
        X_final_local = X_scaled_local[:, sel_indices_local] if sel_indices_local else X_scaled_local  # Select only chosen feature columns from scaled data
        X_test_selected_local = X_test[:, sel_indices_local] if sel_indices_local and X_test is not None else X_test  # Select same feature columns from test data
        model_local = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=CONFIG["multiprocessing"]["n_jobs"])  # Instantiate Random Forest classifier with 100 trees
        start_train_local = time.perf_counter()  # Record training start time
        model_local.fit(X_final_local, y)  # Train model on selected features
        training_time_local = time.perf_counter() - start_train_local  # Calculate training duration

        dump(model_local, model_path_local)  # Serialize and save trained model to disk
        dump(scaler_local, scaler_path_local)  # Serialize and save fitted scaler to disk
        with open(features_path_local, "w", encoding="utf-8") as fh:  # Open features file for writing
            json.dump(best_feats_local, fh)  # Write selected features list as JSON
        model_params_local = model_local.get_params()  # Extract model hyperparameters
        with open(params_path_local, "w", encoding="utf-8") as ph:  # Open params file for writing
            json.dump(model_params_local, ph, default=str)  # Write model parameters as JSON with string fallback

        print(f"{BackgroundColors.GREEN}Saved final model to {BackgroundColors.CYAN}{model_path_local}{Style.RESET_ALL}")
        print(f"{BackgroundColors.GREEN}Saved scaler to {BackgroundColors.CYAN}{scaler_path_local}{Style.RESET_ALL}")
        print(f"{BackgroundColors.GREEN}Saved params to {BackgroundColors.CYAN}{params_path_local}{Style.RESET_ALL}")

        return model_local, model_params_local, training_time_local, X_test_selected_local, round(float(scaling_time_local), 6)
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def evaluate_final_on_test(model_local, X_test_selected_local, y_test):
    """
    Evaluate a trained classifier on the provided test set and compute a
    standard set of metrics (accuracy, precision, recall, f1, fpr, fnr,
    testing_time_seconds).

    :param model_local: Trained classifier with a "predict" method.
    :param X_test_selected_local: Test features selected to match training subset.
    :param y_test: True test labels.
    :return: Tuple "(metrics_tuple, testing_time_seconds)" where "metrics_tuple" is
             "(acc, prec, rec, f1, fpr, fnr, testing_time)" or a tuple of Nones on error.
    """

    try:
        eval_m = None  # Initialize evaluation metrics as None
        testing_time_local = None  # Initialize testing time as None
        try:  # Attempt to evaluate model on test set
            start_test_local = time.perf_counter()  # Record testing start time
            y_pred_local = model_local.predict(X_test_selected_local)  # Generate predictions on test set
            acc_local = accuracy_score(y_test, y_pred_local)  # Calculate accuracy metric
            prec_local = precision_score(y_test, y_pred_local, average="weighted", zero_division=0)  # Calculate weighted precision metric
            rec_local = recall_score(y_test, y_pred_local, average="weighted", zero_division=0)  # Calculate weighted recall metric
            f1_local = f1_score(y_test, y_pred_local, average="weighted", zero_division=0)  # Calculate weighted F1 score
            if len(np.unique(y_test)) == 2:  # Verify if binary classification problem
                cm_local = confusion_matrix(y_test, y_pred_local)  # Generate confusion matrix for binary classification
                if cm_local.shape == (2, 2):  # Verify if standard 2x2 binary confusion matrix
                    tn_local, fp_local, fn_local, tp_local = cm_local.ravel()  # Extract true negatives, false positives, false negatives, true positives
                    fpr_local = fp_local / (fp_local + tn_local) if (fp_local + tn_local) > 0 else 0  # Calculate false positive rate
                    fnr_local = fn_local / (fn_local + tp_local) if (fn_local + tp_local) > 0 else 0  # Calculate false negative rate
                else:  # Handle non-standard confusion matrix shape
                    total_local = cm_local.sum() if cm_local.size > 0 else 1  # Calculate total predictions
                    fpr_local = float(cm_local.sum() - np.trace(cm_local)) / float(total_local) if total_local > 0 else 0  # Estimate FPR from matrix
                    fnr_local = fpr_local  # Use FPR as FNR estimate
            else:  # Handle multi-class classification
                cm_local = confusion_matrix(y_test, y_pred_local)  # Generate confusion matrix for multi-class
                supports_local = cm_local.sum(axis=1)  # Calculate support (samples per class)
                fprs_local = []  # Initialize list for per-class false positive rates
                fnrs_local = []  # Initialize list for per-class false negative rates
                for i_local in range(cm_local.shape[0]):  # Iterate over each class
                    tp_l = cm_local[i_local, i_local]  # Extract true positives for class i
                    fn_l = cm_local[i_local, :].sum() - tp_l  # Calculate false negatives for class i
                    fp_l = cm_local[:, i_local].sum() - tp_l  # Calculate false positives for class i
                    tn_l = cm_local.sum() - (tp_l + fp_l + fn_l)  # Calculate true negatives for class i
                    denom_fnr_l = (tp_l + fn_l) if (tp_l + fn_l) > 0 else 1  # Calculate FNR denominator with zero protection
                    denom_fpr_l = (fp_l + tn_l) if (fp_l + tn_l) > 0 else 1  # Calculate FPR denominator with zero protection
                    fnr_i_l = fn_l / denom_fnr_l  # Calculate false negative rate for class i
                    fpr_i_l = fp_l / denom_fpr_l  # Calculate false positive rate for class i
                    fprs_local.append((fpr_i_l, supports_local[i_local]))  # Store FPR with class support
                    fnrs_local.append((fnr_i_l, supports_local[i_local]))  # Store FNR with class support
                total_support_local = float(supports_local.sum()) if supports_local.sum() > 0 else 1.0  # Calculate total support across all classes
                fpr_local = float(sum(v * s for v, s in fprs_local) / total_support_local)  # Calculate weighted average FPR
                fnr_local = float(sum(v * s for v, s in fnrs_local) / total_support_local)  # Calculate weighted average FNR
            testing_time_local = time.perf_counter() - start_test_local  # Calculate testing duration
            eval_m = (acc_local, prec_local, rec_local, f1_local, fpr_local, fnr_local, testing_time_local)  # Package all metrics into tuple
        except Exception:  # Catch any evaluation errors
            eval_m = (None, None, None, None, None, None, None)  # Return tuple of None values on error
        return eval_m, testing_time_local  # Return metrics tuple and testing time
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


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
    feature_extraction_time_s=None,
):
    """
    Build the consolidated run row dictionary for the best GA individual and
    write it to the consolidated CSV via "write_consolidated_csv".

    :param ts: Timestamp string used for this run artifacts.
    :param model_local: Trained model instance.
    :param model_params_local: Dictionary of model hyperparameters ("get_params()").
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

    try:
        cv_method_local = "StratifiedKFold(n_splits=10)" if n_generations is not None or best_pop_size is not None else "train_test_split"  # Determine CV method used based on GA execution

        run_row = {  # Build consolidated run row dictionary
            "timestamp": ts,  # Timestamp of model training
            "tool": "Genetic Algorithm",  # Tool name identifier
            "run_index": "best",  # Run index indicating best result
            "model": model_local.__class__.__name__,  # Model class name
            "dataset": os.path.relpath(csv_path),  # Relative path to dataset
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
            "feature_extraction_time_s": round(float(feature_extraction_time_s), 6) if feature_extraction_time_s is not None else None,
            "training_time_s": round(float(training_time_local), 6) if training_time_local is not None else None,
            "testing_time_s": round(float(testing_time_local), 6) if testing_time_local is not None else None,
            "elapsed_run_time": round(float(elapsed_run_time), 6) if elapsed_run_time is not None else None,
            "hardware": json.dumps(get_hardware_specifications()),
            "best_features": json.dumps(best_features),
            "rfe_ranking": json.dumps(rfe_ranking),
        }
        
        write_consolidated_csv([run_row], output_dir)
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics

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
    "save_and_analyze_results": it determines the selected features, extracts RFE
    rankings, optionally re-evaluates the best individual on a provided test set,
    builds the consolidated CSV rows and writes them to disk.

    :param best_ind: Best individual from the Genetic Algorithm (binary mask/list).
    :param feature_names: List of feature names corresponding to bits in "best_ind".
    :param X: Feature set (DataFrame or numpy array) used during GA/training.
    :param y: Target variable (Series or array) used during GA/training.
    :param csv_path: Path to the original CSV file for saving outputs.
    :param metrics: Optional precomputed metrics tuple for the best individual.
    :param X_test: Optional test features to evaluate the best individual if "metrics" is None.
    :param y_test: Optional test labels to evaluate the best individual if "metrics" is None.
    :param n_generations: Number of GA generations used (for metadata only).
    :param best_pop_size: Population size that yielded the best result (for metadata only).
    :param runs_list: Optional list of per-run results (each a dict with keys "metrics","best_features" or "best_ind").
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

    try:
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

        model_local, model_params, training_time_s, X_test_selected, final_scaling_time = train_and_save_final_model(
            best_features, X, y, feature_names, X_test, model_path, scaler_path, features_path, params_path
        )
        
        eval_metrics, testing_time_s = evaluate_final_on_test(model_local, X_test_selected, y_test)
        
        output_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis/"  # Directory to save outputs
        os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist

        try:
            feature_extraction_time_s = round(float(final_scaling_time), 6) if final_scaling_time is not None else None
        except Exception:
            feature_extraction_time_s = None

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
            feature_extraction_time_s,
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def prepare_feature_dataframe(X, feature_names):
    """
    Ensure features are available as a pandas DataFrame with appropriate column names.

    :param X: Feature matrix (DataFrame or numpy array).
    :param feature_names: Optional iterable of feature names.
    :return: pandas.DataFrame with feature columns.
    """

    try:
        if not isinstance(X, pd.DataFrame):  # If X is not a pandas DataFrame
            try:  # Try to create a DataFrame with original feature names
                df_features = pd.DataFrame(X, columns=list(feature_names))  # Create DataFrame with original feature names
            except Exception:  # If creating DataFrame with original feature names fails
                df_features = pd.DataFrame(X)  # Create DataFrame without original feature names
                df_features.columns = [f"feature_{i}" for i in range(df_features.shape[1])]  # Generic feature names
        else:  # If X is already a pandas DataFrame
            df_features = X.copy()  # Use the DataFrame as is

        return df_features  # Return the prepared DataFrame
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def analyze_top_features(df, y, top_features, csv_path="."):
    """
    Analyze and visualize the top features.

    :param df: DataFrame containing the features.
    :param y: Target variable.
    :param top_features: List of top feature names.
    :param csv_path: Path to the original CSV file for saving outputs.
    :return: None
    """

    try:
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def analyze_results(saved_info, X, y, feature_names, csv_path):
    """
    Analyze and visualize results that were previously saved by "save_results".

    :param saved_info: Dictionary returned from "save_results" (must contain key "best_features").
    :param X: Feature set (DataFrame or numpy array) used during GA/training.
    :param y: Target variable (Series or array) used during GA/training.
    :param feature_names: List of original feature names used to construct the DataFrame.
    :param csv_path: Path to the original CSV file for saving outputs (used by analyzers).
    :return: None
    """

    try:
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def run_population_sweep(
    dataset_name,
    csv_path,
    n_generations=200,
    min_pop=20,
    max_pop=20,
    cxpb=0.5,
    mutpb=0.01,
    runs=None,
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

    try:
        if runs is None:  # If runs is not provided, use the configured default
            runs = CONFIG.get("execution", {}).get("runs", 5)  # Default to configured runs or 5

        verbose_output(
            f"{BackgroundColors.GREEN}Starting population sweep for dataset {BackgroundColors.CYAN}{dataset_name}{BackgroundColors.GREEN} from size {BackgroundColors.CYAN}{min_pop}{BackgroundColors.GREEN} to {BackgroundColors.CYAN}{max_pop}{BackgroundColors.GREEN}, running {BackgroundColors.CYAN}{n_generations}{BackgroundColors.GREEN} generations and {BackgroundColors.CYAN}{runs}{BackgroundColors.GREEN} runs each.{Style.RESET_ALL}"
        )

        clear_fitness_cache()  # Clear any existing fitness cache to ensure fresh evaluations for the sweep

        send_telegram_message(TELEGRAM_BOT, [
            f"Starting population sweep for dataset {dataset_name} from size {min_pop} to {max_pop}"
        ])  # Send start message

        data = prepare_sweep_data(csv_path, dataset_name, min_pop, max_pop, n_generations)  # Prepare dataset
        if data is None:  # If preparation failed
            return {}  # Exit early

        X_train, X_test, y_train, y_test, feature_names = data  # Unpack prepared data

        folds = CONFIG.get("cross_validation", {}).get("n_folds", 10)  # Use configured constant for CV folds
        
        progress_state = compute_progress_state(
            min_pop, max_pop, n_generations, runs, progress_bar, folds=folds
        )  # Compute progress state for tracking

        results = {}  # Dictionary to hold results per population size
        for p in range(min_pop, max_pop + 1):  # For each population size
            results[p] = {"runs": [], "avg_metrics": None, "common_features": set()}  # Initialize results entry

        with global_state_lock:  # Thread-safe read of CPU_PROCESSES
            cpu_procs = CPU_PROCESSES  # Read CPU_PROCESSES value
        shared_pool = multiprocessing.Pool(processes=cpu_procs if cpu_procs else None)  # Create a shared multiprocessing pool for parallel GA runs

        start_run_time = time.perf_counter()  # Start timing the entire run process
        for run in range(runs):  # For each run
            for pop_size in range(min_pop, max_pop + 1):  # For each population size
                send_telegram_message(TELEGRAM_BOT, [
                    f"Run {run + 1}/{runs} - population size {pop_size}/{max_pop}"
                ])  # Send start message for this run and population size
                start_pop_time = time.perf_counter()  # Start timing this population size iteration
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
                elapsed_pop_time = time.perf_counter() - start_pop_time  # Calculate elapsed time for this population size
                if result:  # If result is valid
                    results[pop_size]["runs"].append(result)  # Append result to runs list

                f1_score = result.get("metrics", [None]*4)[3] if result else None  # Extract F1-Score from metrics if available
                f1_msg = f" - F1: {f1_score:.4f}" if f1_score is not None else ""  # Prepare F1 score message if available
                send_telegram_message(
                    TELEGRAM_BOT,
                    f"Completed run {run + 1}/{runs} - population size {pop_size}/{max_pop} {f1_msg} in {calculate_execution_time(elapsed_pop_time)}"
                )  # Send completion message for this run and population size (no int cast on formatted time)

        try:  # Close the shared pool after all GA iterations are complete
            shared_pool.close()  # Signal no more work will be submitted
            shared_pool.join()  # Wait for all workers to finish
        except Exception:  # If closing the pool fails (e.g., if it was already closed or if an error occurred)
            pass  # Best-effort cleanup

        best_score, best_result, best_metrics, results = aggregate_sweep_results(
            results, min_pop, max_pop, dataset_name
        )  # Aggregate results and find best

        try:  # Attempt to generate comparison table
            generate_run_comparison_table(
                results, csv_path, dataset_name, min_pop, max_pop, n_generations, cxpb, mutpb
            )  # Generate CSV comparison table with aggregated metrics
        except Exception as e:  # If table generation fails
            verbose_output(
                f"{BackgroundColors.YELLOW}Skipping comparison table generation due to error: {e}{Style.RESET_ALL}"
            )  # Log warning but continue

        try:  # Attempt to generate comparison plots
            generate_multi_run_comparison_plots(
                results, csv_path, dataset_name, min_pop, max_pop, n_generations, cxpb, mutpb
            )  # Generate multi-run comparison visualization plots
        except Exception as e:  # If plot generation fails
            verbose_output(
                f"{BackgroundColors.YELLOW}Skipping multi-run comparison plots due to error: {e}{Style.RESET_ALL}"
            )  # Log warning but continue

        elapsed_run_time = time.perf_counter() - start_run_time  # Calculate elapsed time for the entire run process
        
        if best_result:  # If a best result was found
            best_pop_size, runs_list, common_features = best_result  # Unpack the best result
            print(
                f"\n{BackgroundColors.GREEN}Best population size: {BackgroundColors.CYAN}{best_pop_size}{Style.RESET_ALL}"
            )
            print(
                f"{BackgroundColors.GREEN}Common features across runs: {BackgroundColors.CYAN}{len(common_features)}{Style.RESET_ALL}"
            )
            print_metrics(best_metrics) if CONFIG.get("execution", {}).get("verbose", False) else None  # Print metrics if VERBOSE is enabled
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def calculate_execution_time(start_time, finish_time=None):
    """
    Calculates the execution time and returns a human-readable string.

    Accepts either:
    - Two datetimes/timedeltas: "calculate_execution_time(start, finish)"
    - A single timedelta or numeric seconds: "calculate_execution_time(delta)"
    - Two numeric timestamps (seconds): "calculate_execution_time(start_s, finish_s)"

    Returns a string like "1h 2m 3s".
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def play_sound():
    """
    Plays a sound when the program finishes and skips if the operating system is Windows.
    :return: None
    """

    try:
        current_os = platform.system()  # Get the current operating system
        if current_os == "Windows":  # If the current operating system is Windows
            return  # Do nothing

        sound_file = CONFIG["sound"]["file"]  # Get sound file from config
        if verify_filepath_exists(sound_file):  # If the sound file exists
            if current_os in CONFIG["sound"]["commands"]:  # If the platform.system() is in the sound commands dictionary
                os.system(f"{CONFIG['sound']['commands'][current_os]} {sound_file}")  # Play the sound
            else:  # If the platform.system() is not in the sound commands dictionary
                print(
                    f"{BackgroundColors.RED}The {BackgroundColors.CYAN}{current_os}{BackgroundColors.RED} is not in the {BackgroundColors.CYAN}sound commands dictionary{BackgroundColors.RED}. Please add it!{Style.RESET_ALL}"
                )
        else:  # If the sound file does not exist
            print(
                f"{BackgroundColors.RED}Sound file {BackgroundColors.CYAN}{sound_file}{BackgroundColors.RED} not found. Make sure the file exists.{Style.RESET_ALL}"
            )
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def run_genetic_algorithm(config=None, csv_path=None):
    """
    Execute genetic algorithm feature selection with provided configuration.
    This function is the main orchestration entry point for programmatic execution.

    :param config: Configuration dictionary (uses defaults if None)
    :param csv_path: Path to CSV dataset (uses config/default if None)
    :return: Dictionary with sweep results
    """

    try:
        global CONFIG, CPU_PROCESSES  # Use global configuration and CPU_PROCESSES

        if config is not None:  # If configuration provided
            CONFIG = config  # Use provided configuration
        elif not CONFIG:  # If no configuration set (empty dict)
            CONFIG = get_default_config()  # Use defaults

        with global_state_lock:  # Thread-safe initialization
            if CPU_PROCESSES is None or CPU_PROCESSES == 1:  # If not initialized or default
                CPU_PROCESSES = CONFIG["multiprocessing"]["cpu_processes"]  # Set from config

        if csv_path is None:  # If no path provided
            csv_path = "./Datasets/CICDDoS2019/01-12/DrDoS_DNS.csv"  # Use default

        n_generations = CONFIG.get("genetic_algorithm", {}).get("n_generations", 200)  # Get generations from config
        min_pop = CONFIG.get("genetic_algorithm", {}).get("min_pop", 20)  # Get min population from config
        max_pop = CONFIG.get("genetic_algorithm", {}).get("max_pop", 20)  # Get max population from config
        cxpb = CONFIG.get("genetic_algorithm", {}).get("cxpb", 0.5)  # Get crossover probability from config
        mutpb = CONFIG.get("genetic_algorithm", {}).get("mutpb", 0.01)  # Get mutation probability from config
        runs = CONFIG.get("execution", {}).get("runs", 5)  # Get number of runs from config
        skip_train = CONFIG.get("execution", {}).get("skip_train_if_model_exists", False)  # Get skip train flag from config

        dataset_name = os.path.splitext(os.path.basename(csv_path))[0]  # Extract dataset name

        print(
            f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Genetic Algorithm Feature Selection{BackgroundColors.GREEN} program!{Style.RESET_ALL}",
            end="\n\n",
        )  # Print welcome message

        start_time = datetime.datetime.now()  # Record start time

        setup_telegram_bot()  # Setup Telegram bot

        send_telegram_message(
            TELEGRAM_BOT,
            [f"Starting Genetic Algorithm Feature Selection for {dataset_name} at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"]
        )  # Send start message

        if skip_train:  # If skip training enabled
            should_return = handle_skip_train_if_model_exists(csv_path)  # Try loading existing model
            if should_return:  # If model loaded successfully
                return {}  # Exit early

        if CONFIG.get("resource_monitor", {}).get("enabled", True):  # If resource monitor enabled
            start_resource_monitor_safe(
                interval_seconds=CONFIG.get("resource_monitor", {}).get("interval_seconds", 30),
                reserve_cpu_frac=CONFIG.get("resource_monitor", {}).get("reserve_cpu_frac", 0.15),
                reserve_mem_frac=CONFIG.get("resource_monitor", {}).get("reserve_mem_frac", 0.15),
                min_procs=CONFIG.get("resource_monitor", {}).get("min_procs", 1),
                max_procs=CONFIG.get("resource_monitor", {}).get("max_procs", None),
                min_gens_before_update=CONFIG.get("resource_monitor", {}).get("min_gens_before_update", 10),
                daemon=CONFIG.get("resource_monitor", {}).get("daemon", True),
            )  # Start resource monitor thread

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
        )  # Execute population sweep

        verbose = CONFIG.get("execution", {}).get("verbose", False)  # Get verbose flag
        if verbose and sweep_results:  # If verbose and results exist
            print(f"\n{BackgroundColors.GREEN}Detailed sweep results by population size:{Style.RESET_ALL}")  # Print header
            for pop_size, features in sweep_results.items():  # Iterate results
                print(f"  Pop {pop_size}: {len(features)} features -> {features}")  # Print details

        finish_time = datetime.datetime.now()  # Record finish time
        print(
            f"{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
        )  # Print timing info
        print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}")  # Print completion

        send_telegram_message(
            TELEGRAM_BOT,
            [f"Genetic Algorithm feature selection completed for {dataset_name}. Execution time: {calculate_execution_time(start_time, finish_time)}"]
        )  # Send completion message

        if CONFIG.get("execution", {}).get("play_sound", True):  # If sound enabled
            atexit.register(play_sound)  # Register sound callback

        return sweep_results  # Return results
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def main():
    """
    Main function for CLI execution. Initializes configuration and executes GA.

    :return: None
    """

    try:
        global CONFIG, logger  # Declare global variables

        cli_args = parse_cli_args()  # Parse command-line arguments

        CONFIG = initialize_config(config_path=cli_args.config if hasattr(cli_args, "config") else None, cli_args=cli_args)  # Initialize merged configuration

        logger = initialize_logger(CONFIG)  # Initialize logger with configuration

        global CPU_PROCESSES  # Declare global CPU_PROCESSES
        with global_state_lock:  # Thread-safe initialization
            CPU_PROCESSES = CONFIG["multiprocessing"]["cpu_processes"]  # Set initial CPU processes from config

        if hasattr(cli_args, "csv_path") and cli_args.csv_path:  # If CSV path provided via CLI, use it; otherwise, use config or default
            csv_path = cli_args.csv_path  # Use CSV path from CLI arguments if provided
        else:  # If no CSV path provided via CLI, verify config and then default
            csv_path = CONFIG.get("paths", {}).get("csv_path")  # Use CSV path from config if available
        if not csv_path:  # If no CSV path provided via CLI or config, use default
            csv_path = "./Datasets/CICDDoS2019/01-12/DrDoS_DNS.csv"  # Default CSV path

        run_genetic_algorithm(config=CONFIG, csv_path=csv_path)  # Run genetic algorithm with configuration
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


if __name__ == "__main__":
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """

    main()  # Call the main function
