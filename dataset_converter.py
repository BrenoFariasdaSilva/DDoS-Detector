"""
================================================================================
Multi-Format Dataset Converter (dataset_converter.py)
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-05-31

Short Description:
    Command-line utility that discovers datasets (ARFF, CSV, Parquet, TXT)
    under an input directory, applies lightweight structural cleaning to
    text-based formats, loads them into pandas DataFrames, and writes
    converted outputs (ARFF, CSV, Parquet, TXT) to a mirrored `Output`
    directory structure.

Defaults & Behavior:
    - Default input directory: ./Input
    - Default output directory: ./Output
    - Supported input formats: .arff, .csv, .parquet, .txt
    - Cleaning: minimal whitespace/domain-list normalization for ARFF/CSV/TXT
    - Parquet files are rewritten via `fastparquet` for consistency
    - Conversion preserves directory hierarchy relative to `Input`
    - Optional completion sound (platform-dependent)

Usage:
    - Run interactively:
        python3 dataset_converter.py
    - Or pass CLI args: `-i/--input`, `-o/--output`, `-f/--formats`, `-v/--verbose`

Dependencies (non-exhaustive):
    - Python 3.8+
    - pandas, fastparquet, scipy, liac-arff (arff), colorama, tqdm

Notes and Caveats:
    - The converter performs pragmatic cleaning only; do not rely on it to
        fully sanitize malformed CSVs.
    - The script uses both `scipy` and `liac-arff` as fallbacks for ARFF.
    - Disk-space checks are performed before writing outputs.
    - The module expects UTF-8 encoded text files.

TODOs (short):
    - Add unit tests and more robust CSV parsing
    - Add optional parallel conversion mode for large workloads
    - Provide more granular CLI control for cleaning rules
"""

import argparse  # For parsing command-line arguments
import arff  # Liac-arff, used to save ARFF files
import atexit  # For playing a sound when the program finishes
import datetime  # For timestamping
import io  # For in-memory file operations
import os  # For running commands in the terminal
import pandas as pd  # For handling CSV and TXT file formats
import platform  # For getting the operating system name
import shutil  # For checking disk usage
import sys  # For system-specific parameters and functions
import telegram_bot as telegram_module  # For setting Telegram prefix and device info
import traceback  # For printing tracebacks on exceptions
import yaml  # For loading configuration from YAML file
from colorama import Style  # For coloring the terminal output
from fastparquet import ParquetFile  # For handling Parquet file format
from Logger import Logger  # For logging output to both terminal and file
from pathlib import Path  # For handling file paths
from scipy.io import arff as scipy_arff  # Used to read ARFF files
from telegram_bot import TelegramBot, send_exception_via_telegram, send_telegram_message, setup_global_exception_hook  # For Telegram utilities and global exception hook
from tqdm import tqdm  # For showing a progress bar


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
DEFAULTS = None  # Will hold the default configuration loaded from YAML or hardcoded defaults

# Telegram Bot Setup:
TELEGRAM_BOT = None  # Global Telegram bot instance (initialized in setup_telegram_bot)

# Logger Setup:
logger = Logger(f"./Logs/{Path(__file__).stem}.log", clean=True)  # Create a Logger instance
sys.stdout = logger  # Redirect stdout to the logger
sys.stderr = logger  # Redirect stderr to the logger

# Sound Constants:
SOUND_COMMANDS = {"Darwin": "afplay", "Linux": "aplay", "Windows": "start"}  # Sound play command
SOUND_FILE = "./.assets/Sounds/NotificationSound.wav"  # Notification sound path

# RUN_FUNCTIONS:
RUN_FUNCTIONS = {
    "Play Sound": True,  # Set to True to play a sound when the program finishes
}


# Functions Definitions:


setup_global_exception_hook()  # Set global exception hook to shared Telegram handler


def get_default_config() -> dict:  # Return default configuration for dataset_converter
    """
    Default dataset_converter configuration.

    :return: Dictionary with default configuration values.
    """

    return {
        "dataset_converter": {
            "verbose": False,  # Whether to enable verbose messages
            "input_directory": "./Input",  # Default input directory
            "output_directory": "./Output",  # Default output directory
            "ignore_dirs": ["Results"],  # Directories to ignore
            "ignore_files": [],  # File substrings to ignore
        }
    }


def load_config_file(path: str = "config.yaml") -> dict:  # Load YAML config if exists
    """
    Load configuration from YAML file if present.

    :param path: Path to YAML file.
    :return: Loaded configuration dictionary or empty dict.
    """

    try:  # Wrap file loading to report errors and fallback gracefully
        if os.path.exists(path):  # Verify path existence
            with open(path, "r", encoding="utf-8") as fh:  # Open file for reading
                data = yaml.safe_load(fh) or {}  # Parse YAML safely
                return data  # Return parsed config
    except Exception as e:  # On error, log and notify via Telegram then return empty dict
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        return {}  # Return empty dict on error

    return {}  # Default empty dict when file not found


def initialize_defaults() -> None:
    """
    Initialize DEFAULTS by loading defaults and merging with config.yaml.

    :return: None
    """

    try:  # Wrap initialization logic to ensure production-safe monitoring
        global DEFAULTS  # Declare that we will assign to the module-global DEFAULTS
        
        defaults = get_default_config()  # Load hard-coded default configuration
        cfg = load_config_file()  # Load configuration from disk (config.yaml)
        
        if cfg and isinstance(cfg, dict) and "dataset_converter" in cfg:  # Verify presence of dataset_converter section
            try:  # Attempt to merge nested dataset_converter values
                defaults_dataset = defaults.get("dataset_converter", {})  # Extract defaults subsection
                file_dataset = cfg.get("dataset_converter", {})  # Extract file subsection
                defaults_dataset.update(file_dataset)  # Merge file subsection into defaults subsection
                defaults["dataset_converter"] = defaults_dataset  # Assign merged subsection back into defaults
            except Exception:  # If nested merge fails, fall back to shallow overlay
                defaults.update(cfg)  # Overlay top-level keys with file config
        DEFAULTS = defaults  # Set the module-global DEFAULTS to the merged configuration
        # Initialize runtime flags from DEFAULTS
        try:
            global VERBOSE  # Declare that we will assign to the module-global VERBOSE
            VERBOSE = bool(DEFAULTS.get("dataset_converter", {}).get("verbose", False))  # Set VERBOSE based on DEFAULTS, defaulting to False if not specified
        except Exception:  # Catch any issues with accessing DEFAULTS and ensure VERBOSE is set to a boolean
            VERBOSE = False  # Default to False if there was an issue accessing the verbose setting in DEFAULTS
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
        if VERBOSE and true_string != "":  # If VERBOSE is True and a true_string was provided
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
            telegram_module.TELEGRAM_DEVICE_INFO = f"{telegram_module.get_local_ip()} - {platform.system()}"
            telegram_module.RUNNING_CODE = os.path.basename(__file__)
        except Exception as e:
            print(f"{BackgroundColors.RED}Failed to initialize Telegram bot: {e}{Style.RESET_ALL}")
            TELEGRAM_BOT = None  # Set to None if initialization fails
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def parse_cli_arguments():
    """
    Parse command-line arguments for the dataset converter.

    :return: Parsed ArgumentParser namespace.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Parsing command-line arguments...{Style.RESET_ALL}"
        )  # Output the verbose message

        parser = argparse.ArgumentParser(
            description="Multi-Format Dataset Converter: convert ARFF/CSV/Parquet/TXT datasets"
        )  # Create the argument parser

        parser.add_argument(
            "-i", "--input", type=str, help="Input path (file or directory). If not provided, uses ./Input"
        )  # Input path argument
        parser.add_argument(
            "-o", "--output", type=str, help="Output directory. If not provided, uses ./Output"
        )  # Output directory argument
        parser.add_argument(
            "-f",
            "--formats",
            type=str,
            help="Comma-separated output formats to produce (arff,csv,parquet,txt). If not provided, all formats are produced",
        )  # Output formats argument
        parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")  # Verbose mode flag

        return parser.parse_args()  # Return parsed CLI arguments
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
            true_string=f"{BackgroundColors.GREEN}Verifying if the file or folder exists at the path: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}"
        )  # Output the verbose message

        return os.path.exists(filepath)  # Return True if the file or folder exists, False otherwise
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def extract_input_paths_from_datasets(dmap: dict) -> list:  # Define a nested helper to extract candidate paths
    """
    Extract input path candidates from datasets mapping.

    :param dmap: Datasets mapping from configuration.
    :return: List of candidate input path strings.
    """

    try:  # Wrap helper logic to ensure production-safe monitoring
        if not dmap or not isinstance(dmap, dict):  # Verify mapping is a dict
            return []  # Return empty list when mapping is missing or invalid
        candidates = []  # Initialize list of candidate paths
        
        for key in sorted(dmap.keys()):  # Iterate deterministically over mapping keys
            val = dmap.get(key)  # Retrieve the mapping value for the current key
            if isinstance(val, str):  # If the mapping value is a string path
                cleaned = val.strip() if isinstance(val, str) else val  # Strip surrounding whitespace from the path
                if cleaned:  # Only add non-empty cleaned paths
                    candidates.append(cleaned)  # Add the cleaned string path to candidates
            elif isinstance(val, (list, tuple)):  # If the mapping value is a list/tuple of paths
                for p in val:  # Iterate each candidate path in the sequence
                    cleaned = p.strip() if isinstance(p, str) else p  # Strip surrounding whitespace from each candidate
                    if cleaned:  # Only add non-empty cleaned candidates
                        candidates.append(cleaned)  # Add the cleaned candidate to list
            elif isinstance(val, dict):  # If the mapping value is a nested dict
                single = val.get("path") or val.get("input")  # Extract a single path candidate from known keys
                if isinstance(single, str):  # If the single candidate is a string
                    cleaned = single.strip()  # Strip surrounding whitespace from the single candidate
                    if cleaned:  # Only add non-empty cleaned single candidate
                        candidates.append(cleaned)  # Add the single candidate to the list
                multi = val.get("paths") or val.get("inputs")  # Extract multi-paths from known keys

                if isinstance(multi, (list, tuple)):  # If multi-paths is a sequence
                    for candidate in multi:  # Iterate provided multi-path entries
                        cleaned = candidate.strip() if isinstance(candidate, str) else candidate  # Strip whitespace from each multi candidate
                        if cleaned:  # Only add non-empty cleaned entries
                            candidates.append(cleaned)  # Append each cleaned candidate to the list
        
        return candidates  # Return collected candidate paths
    except Exception as e:  # Catch exceptions inside helper
        print(str(e))  # Print helper exception to terminal for logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send helper exception via Telegram
        raise  # Re-raise to preserve failure semantics


def validate_and_prepare_input_paths(paths: list) -> list:  # Define a nested helper to validate and create inputs
    """
    Validate candidate input paths and ensure directories exist.

    :param paths: Candidate input path list.
    :return: List of validated input paths.
    """

    try:  # Wrap helper logic to ensure production-safe monitoring
        valid = []  # Initialize list for validated existing paths
        for p in paths:  # Iterate provided candidate paths
            p_str = str(p).strip() if p is not None else ""  # Strip surrounding whitespace and coerce to string
            if not p_str:  # Skip empty or None entries after cleaning
                continue  # Continue to next candidate when value is falsy
            if verify_filepath_exists(p_str):  # Verify candidate exists on filesystem
                valid.append(p_str)  # Add existing cleaned path to validated list
            else:  # If candidate does not exist, do NOT create input directories automatically
                verbose_output(f"{BackgroundColors.YELLOW}Configured input path does not exist, skipping: {BackgroundColors.CYAN}{p_str}{Style.RESET_ALL}")  # Informative verbose message when configured input is missing
                continue  # Skip non-existing configured input paths without creating them
        
        return valid  # Return the list of validated paths
    except Exception as e:  # Catch exceptions inside helper
        print(str(e))  # Print helper exception to terminal for logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send helper exception via Telegram
        raise  # Re-raise to preserve failure semantics


def resolve_output_path(arg_output: str, cfg_section: dict) -> str:  # Define a nested helper to resolve output directory
    """
    Resolve the output directory path from CLI argument or configuration.

    :param arg_output: Output path provided via CLI.
    :param cfg_section: The dataset_converter configuration section.
    :return: Resolved output path string.
    """

    try:  # Wrap helper logic to ensure production-safe monitoring
        output_default = cfg_section.get("output_directory", "./Output") or "./Output"  # Determine configured default
        out = arg_output if arg_output else output_default  # Choose CLI-provided output or fallback default
        
        if not verify_filepath_exists(out):  # Verify output path existence
            create_directories(out)  # Create output directory when missing
            
        return out  # Return the resolved output path
    except Exception as e:  # Catch exceptions inside helper
        print(str(e))  # Print helper exception to terminal for logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send helper exception via Telegram
        raise  # Re-raise to preserve failure semantics


def resolve_io_paths(args):
    """
    Resolve and validate input/output paths from CLI arguments.

    :param args: Parsed CLI arguments.
    :return: Tuple (input_path, output_path).
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Resolving input/output paths...{Style.RESET_ALL}"
        )  # Output the verbose message

        cfg = DEFAULTS.get("dataset_converter", {}) if DEFAULTS else {}  # Get dataset_converter config safely
        datasets_cfg = cfg.get("datasets", {})  # Resolve datasets mapping from config

        input_candidates = [args.input] if args.input else extract_input_paths_from_datasets(datasets_cfg)  # Build initial candidate list from CLI or config
        resolved_inputs = validate_and_prepare_input_paths(input_candidates)  # Validate and prepare candidate input paths
        out_path = resolve_output_path(args.output if hasattr(args, "output") else None, cfg)  # Resolve output path using helper

        if not resolved_inputs:  # If no validated input paths were found
            print(f"{BackgroundColors.RED}No input path available from CLI or configuration datasets{Style.RESET_ALL}")  # Report missing input paths
            return None, None  # Return failure when no inputs are available

        return resolved_inputs, out_path  # Return validated input list and resolved output path
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def configure_verbose_mode(args):
    """
    Enable verbose output mode when requested via CLI.

    :param args: Parsed CLI arguments.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        if args.verbose:  # If verbose mode requested
            global VERBOSE  # Use global variable
            VERBOSE = True  # Enable verbose mode
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def create_directories(directory_name):
    """
    Creates a directory if it does not exist.

    :param directory_name: Name of the directory to be created.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        if not directory_name:  # Empty string or None
            print(f"{BackgroundColors.YELLOW}Warning: create_directories called with empty path; skipping{Style.RESET_ALL}")
            return  # Skip when no valid directory name provided

        verbose_output(
            f"{BackgroundColors.GREEN}Creating directory: {BackgroundColors.CYAN}{directory_name}{Style.RESET_ALL}"
        )  # Output the verbose message

        if not os.path.exists(directory_name):  # If the directory does not exist
            os.makedirs(directory_name)  # Create the directory
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def get_dataset_files(directory=None):
    """
    Get all dataset files in the specified directory and its subdirectories.

    :param directory: Path to the directory to search for dataset files.
    :return: List of paths to dataset files.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Searching for dataset files in: {BackgroundColors.CYAN}{directory}{Style.RESET_ALL}"
        )  # Output the verbose message

        dataset_files = []  # List to store discovered dataset file paths
        cfg = DEFAULTS.get("dataset_converter", {}) if DEFAULTS else {}  # Get dataset_converter settings from DEFAULTS
        ignore_list = cfg.get("ignore_dirs", ["Results"])  # Get ignore directories list from configuration
        
        if directory:  # If a specific directory argument provided
            roots = [directory]  # Use the provided directory as single root to scan
        else:  # If no directory argument provided
            datasets_map = cfg.get("datasets", {})  # Retrieve datasets mapping from configuration
            roots = []  # Initialize roots list for scanning
            
            for v in datasets_map.values():  # Iterate over dataset groups in mapping
                if isinstance(v, (list, tuple)):  # If mapping value is a list of paths
                    for candidate in v:  # Iterate candidate paths inside list
                        roots.append(candidate)  # Add candidate path to roots list
                elif isinstance(v, str):  # If mapping value is a single path string
                    roots.append(v)  # Add single path to roots list
        
        for root in roots:  # Iterate roots to walk through filesystem
            if not root:  # If root is empty string or None
                continue  # Skip empty root entries safely
            
            for r, dirs, files in os.walk(root):  # Walk the directory tree starting at root
                if any(ignore_word.lower() in r.lower() for ignore_word in ignore_list):  # If the current path contains ignored directory names
                    continue  # Skip ignored directories
                for file in files:  # Iterate files in the current directory
                    if os.path.splitext(file)[1].lower() in [".arff", ".csv", ".txt", ".parquet"]:  # If file extension is supported
                        dataset_files.append(os.path.join(r, file))  # Append full file path to results list
        
        return dataset_files  # Return collected dataset file paths
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def scan_top_level_for_supported_files(input_directory: str) -> list:
    """
    Scan the directory itself for supported extensions.

    :param input_directory: Directory path to scan.
    :return: List of supported files found directly under the directory.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        supported_exts = {".arff", ".csv", ".parquet", ".txt"}  # Supported file extensions set
        direct_files = []  # Container for files found directly under the directory
        if os.path.isdir(input_directory):  # Verify the path is a directory before listing
            for entry in os.listdir(input_directory):  # Iterate entries directly under the directory
                candidate = os.path.join(input_directory, entry)  # Build candidate full path
                if os.path.isfile(candidate) and os.path.splitext(entry)[1].lower() in supported_exts:  # Verify file and extension
                    direct_files.append(candidate)  # Add matching file to direct_files
        return direct_files  # Return the directly found files (may be empty)
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def scan_immediate_subdirs_for_files(input_directory: str) -> list:
    """
    Scan each immediate subdirectory for dataset files and return first non-empty result.

    :param input_directory: Directory path whose immediate children will be scanned.
    :return: List of dataset files found in the first child directory containing supported files.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        if not os.path.isdir(input_directory):  # Verify the input path is a directory before exploring children
            return []  # Return empty list when input is not a directory
        for entry in os.listdir(input_directory):  # Iterate child entries to explore subdirectories
            child = os.path.join(input_directory, entry)  # Build child path
            if os.path.isdir(child):  # Only consider child directories
                child_files = get_dataset_files(child)  # Attempt recursive discovery in the child directory
                if child_files:  # If any files were discovered in the child
                    return child_files  # Return the first non-empty child discovery
        return []  # Return empty list when no child directories contain dataset files
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def resolve_dataset_files(input_directory):
    """
    Resolve dataset files from a directory or a single file path.

    :param input_directory: Input directory or single file path.
    :return: List of dataset file paths.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        if os.path.isfile(input_directory):  # If the input_directory is actually a file
            return [input_directory]  # Return a single-item list containing the file path

        files = get_dataset_files(input_directory)  # Attempt to recursively discover dataset files under the directory
        if files:  # If recursive discovery returned any files
            return files  # Return discovered files immediately

        direct_files = scan_top_level_for_supported_files(input_directory)  # Scan the directory itself for supported extensions
        if direct_files:  # If direct files were found in the top-level directory
            return direct_files  # Return the directly found files

        child_files = scan_immediate_subdirs_for_files(input_directory)  # Scan each immediate subdirectory separately to handle unusual mounts
        if child_files:  # If any files were discovered in an immediate child directory
            return child_files  # Return the first non-empty child discovery

        return []  # Return empty list when no dataset files could be located
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def resolve_formats(formats):
    """
    Normalize and validate the list of output formats.

    :param formats: List or string of formats.
    :return: Cleaned list of formats.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        if formats is None:  # If no specific formats were provided
            return ["arff", "csv", "parquet", "txt"]  # Default to all supported formats

        if isinstance(formats, str):  # If provided as CSV string
            return [f.strip().lower().lstrip(".") for f in formats.split(",") if f.strip()]  # Split and clean

        return [f.strip().lower().lstrip(".") for f in formats if isinstance(f, str)]  # Clean list
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def resolve_destination_directory(input_directory, input_path, output_directory):
    """
    Determine where converted files should be saved.

    :param input_directory: Source directory.
    :param input_path: Path of the current file.
    :param output_directory: Base output directory.
    :return: Destination directory path.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        if os.path.isfile(input_directory):  # If converting a single file
            return output_directory  # Save directly

        rel_dir = os.path.relpath(os.path.dirname(input_path), input_directory)  # Subdir path
        return os.path.join(output_directory, rel_dir) if rel_dir != "." else output_directory  # Preserve structure
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def get_free_space_bytes(path):
    """
    Return the number of free bytes available on the filesystem
    containing the specified path.

    :param path: File or directory path to inspect.
    :return: Free space in bytes.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        target = path if os.path.isdir(path) else os.path.dirname(path) or "."  # Resolve target directory

        verbose_output(
            f"{BackgroundColors.GREEN}Verifying free space at: {BackgroundColors.CYAN}{target}{Style.RESET_ALL}"
        )  # Output verbose message

        try:  # Try to retrieve disk usage
            usage = shutil.disk_usage(target)  # Get disk usage statistics
            return usage.free  # Return free space
        except Exception as e:  # Catch any errors
            verbose_output(
                f"{BackgroundColors.RED}Failed to retrieve disk usage for {target}: {e}{Style.RESET_ALL}"
            )  # Log error
            return 0  # Fallback to zero
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def format_size_units(size_bytes):
    """
    Format a byte size into a human-readable string with appropriate units.

    :param size_bytes: Size in bytes.
    :return: Formatted size string.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        if size_bytes is None:  # If size_bytes is None
            return "0 Bytes"  # Return 0 Bytes

        try:  # Try to convert to float
            size = float(size_bytes)  # Convert to float
        except Exception:  # Catch conversion errors
            return str(size_bytes)  # Return original value as string

        for unit in ("TB", "GB", "MB", "KB"):  # Iterate through units
            if size >= 1024**4 and unit == "TB":  # Terabytes
                return f"{size / (1024 ** 4):.2f} TB"  # Return formatted string
            if size >= 1024**3 and unit == "GB":  # Gigabytes
                return f"{size / (1024 ** 3):.2f} GB"  # Return formatted string
            if size >= 1024**2 and unit == "MB":  # Megabytes
                return f"{size / (1024 ** 2):.2f} MB"  # Return formatted string
            if size >= 1024**1 and unit == "KB":  # Kilobytes
                return f"{size / 1024:.2f} KB"  # Return formatted string

        return f"{int(size)} Bytes"  # Return bytes if less than 1 KB
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def has_enough_space_for_path(path, required_bytes):
    """
    Verify whether the filesystem containing the specified path has at least
    the required number of free bytes.

    :param path: Path where free space must be evaluated.
    :param required_bytes: Minimum number of bytes required.
    :return: True if there is enough free space, otherwise False.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        parent = os.path.dirname(path) or "."  # Determine the directory to inspect

        verbose_output(
            f"{BackgroundColors.GREEN}Evaluating free space for: {BackgroundColors.CYAN}{parent}{Style.RESET_ALL}"
        )  # Output verbose message

        free = get_free_space_bytes(parent)  # Retrieve free space
        free_str = format_size_units(free)  # Format free space
        req_str = format_size_units(required_bytes)  # Format required space
        verbose_output(
            f"{BackgroundColors.GREEN}Free space: {BackgroundColors.CYAN}{free_str}{BackgroundColors.GREEN}; required: {BackgroundColors.CYAN}{req_str}{Style.RESET_ALL}"
        )  # Log details

        return free >= required_bytes  # Return comparison result
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def ensure_enough_space(path, required_bytes):
    """
    Ensure that the filesystem has enough space to write the required number of bytes.

    :param path: Destination file path to verify.
    :param required_bytes: Number of bytes required for writing.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        if not has_enough_space_for_path(path, required_bytes):  # Verify free space for the write operation
            free = get_free_space_bytes(os.path.dirname(path) or ".")
            raise OSError(
                f"Not enough disk space to write {path}. Free: {format_size_units(free)}; required: {format_size_units(required_bytes)}"
            )
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def estimate_bytes_arff(df, overhead, attributes):
    """
    Estimate required bytes for ARFF output by serializing to an in-memory buffer.

    :param df: pandas DataFrame.
    :param overhead: Additional bytes for headers/metadata.
    :param attributes: List of attributes for ARFF serialization.
    :return: Integer number of required bytes.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        try:  # Attempt ARFF serialization
            buf = io.StringIO()  # In-memory text buffer

            arff_dict = {  # Create a dictionary to hold the ARFF data
                "description": "",  # Description of the dataset (can be left empty)
                "relation": "converted_data",  # Name of the relation (dataset)
                "attributes": attributes,  # List of attributes with their names and types
                "data": df.values.tolist(),  # Convert the DataFrame values to a list of lists for ARFF data
            }

            arff.dump(arff_dict, buf)  # Dump ARFF data into the buffer

            return max(1024, len(buf.getvalue().encode("utf-8")) + overhead)  # Return estimated size with overhead
        except Exception:  # Fallback: estimate via CSV
            return max(1024, int(df.memory_usage(deep=True).sum()))  # Estimate size based on DataFrame memory usage
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def estimate_bytes_csv(df, overhead):
    """
    Estimate required bytes to write a CSV file using an in-memory buffer.

    :param df: pandas DataFrame.
    :param overhead: Additional bytes for headers/metadata.
    :return: Integer number of required bytes.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        try:  # Attempt CSV serialization
            buf = io.StringIO()  # In-memory text buffer
            df.to_csv(buf, index=False)  # Serialize DataFrame to CSV
            return max(1024, len(buf.getvalue().encode("utf-8")) + overhead)  # Return estimated size with overhead

        except Exception:  # Fallback to memory usage
            return max(1024, int(df.memory_usage(deep=True).sum()))  # Estimate size based on DataFrame memory usage
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def estimate_bytes_parquet(df):
    """
    Estimate required bytes for Parquet output using DataFrame memory size.

    :param df: pandas DataFrame.
    :return: Integer number of required bytes.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        return max(1024, int(df.memory_usage(deep=True).sum()))  # Estimate size based on DataFrame memory usage
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def estimate_bytes_from_lines(lines, overhead):
    """
    Estimate required bytes for plain-text lines (UTF-8 encoded).

    :param lines: List of text lines.
    :param overhead: Additional bytes for headers/metadata.
    :return: Integer number of required bytes.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        return max(1024, sum(len((ln or "").encode("utf-8")) for ln in lines) + overhead)  # Estimate byte size
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def clean_parquet_file(input_path, cleaned_path):
    """
    Cleans Parquet files by rewriting them without any textual cleaning,

    :param input_path: Path to the input Parquet file.
    :param cleaned_path: Path where the rewritten Parquet file will be saved.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        df = pd.read_parquet(input_path, engine="fastparquet", low_memory=False)  # Read parquet into DataFrame

        required_bytes = estimate_bytes_parquet(df)  # Estimate bytes needed for cleaned Parquet
        ensure_enough_space(cleaned_path, required_bytes)  # Ensure enough space to write the cleaned file

        df.to_parquet(cleaned_path, index=False)  # Write DataFrame back to parquet at destination
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def clean_arff_lines(lines):
    """
    Cleans ARFF files by removing unnecessary spaces in @attribute domain lists.

    :param lines: List of lines read from the ARFF file.
    :return: List of cleaned lines with sanitized domain values.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        cleaned_lines = []  # List to store cleaned lines

        for line in lines:  # Iterate through each line of the ARFF file
            if (
                line.strip().lower().startswith("@attribute") and "{" in line and "}" in line
            ):  # Verify if the line defines a domain list
                parts = line.split("{")  # Split before domain
                before = parts[0]  # Content before the domain
                domain = parts[1].split("}")[0]  # Extract domain content
                after = line.split("}")[1]  # Content after domain

                cleaned_domain = ",".join([val.strip() for val in domain.split(",")])  # Strip spaces inside domain list
                cleaned_line = f"{before}{{{cleaned_domain}}}{after}"  # Construct cleaned line
                cleaned_lines.append(cleaned_line)  # Add cleaned attribute line
            else:  # If the line is not an attribute definition
                cleaned_lines.append(line)  # Keep non-attribute lines unchanged

        return cleaned_lines  # Return the list of cleaned lines
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def clean_csv_or_txt_lines(lines):
    """
    Cleans TXT and CSV files by removing unnecessary spaces around comma-separated values.

    :param lines: List of lines read from the file.
    :return: List of cleaned lines with sanitized comma-separated values.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        cleaned_lines = []  # List to store cleaned lines

        for line in lines:  # Iterate through each line
            values = line.strip().split(",")  # Split the line on commas
            cleaned_values = [val.strip() for val in values]  # Strip whitespace
            cleaned_line = ",".join(cleaned_values) + "\n"  # Join cleaned values and add newline
            cleaned_lines.append(cleaned_line)  # Add cleaned line

        return cleaned_lines  # Return the list of cleaned lines
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def estimate_bytes_for_lines(lines):
    """
    Estimate the number of bytes a list of text lines will occupy when
    encoded as UTF-8.

    :param lines: List of text lines to measure.
    :return: Estimated byte size.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Estimating UTF-8 byte size for provided lines...{Style.RESET_ALL}"
        )  # Output verbose message

        return sum(len((ln or "").encode("utf-8")) for ln in lines)  # Compute and return byte size
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def write_cleaned_lines_to_file(cleaned_path, cleaned_lines):
    """
    Writes cleaned lines to a specified file.

    :param cleaned_path: Path to the file where cleaned lines will be written.
    :param cleaned_lines: List of cleaned lines to write to the file.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        required_bytes = estimate_bytes_for_lines(cleaned_lines)  # Estimate bytes needed for cleaned lines
        ensure_enough_space(cleaned_path, required_bytes)  # Ensure enough space to write the cleaned file

        with open(cleaned_path, "w", encoding="utf-8") as f:  # Open the cleaned file path for writing
            f.writelines(cleaned_lines)  # Write all cleaned lines to the output file
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def clean_file(input_path, cleaned_path):
    """
    Cleans ARFF, TXT, CSV, and Parquet files by removing unnecessary spaces in
    comma-separated values or domains. For Parquet files, it rewrites the file
    directly without textual cleaning.

    :param input_path: Path to the input file (.arff, .txt, .csv, .parquet).
    :param cleaned_path: Path to save the cleaned file.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        file_extension = os.path.splitext(input_path)[1].lower()  # Get the file extension of the input file

        verbose_output(
            f"{BackgroundColors.GREEN}Cleaning file: {BackgroundColors.CYAN}{input_path}{BackgroundColors.GREEN} and saving to {BackgroundColors.CYAN}{cleaned_path}{Style.RESET_ALL}"
        )  # Output the verbose message

        if file_extension == ".parquet":  # Handle parquet files separately (binary format)
            clean_parquet_file(input_path, cleaned_path)  # Clean parquet file
            return  # Exit early after handling parquet

        with open(input_path, "r", encoding="utf-8") as f:  # Open the input file for reading
            lines = f.readlines()  # Read all lines from the file

        if file_extension == ".arff":  # Cleaning logic for ARFF files
            cleaned_lines = clean_arff_lines(lines)  # Clean ARFF lines
        elif file_extension in [".txt", ".csv"]:  # Cleaning logic for TXT and CSV files
            cleaned_lines = clean_csv_or_txt_lines(lines)  # Clean TXT/CSV lines
        else:  # If the file extension is not supported
            raise ValueError(
                f"{BackgroundColors.RED}Unsupported file extension: {BackgroundColors.CYAN}{file_extension}{Style.RESET_ALL}"
            )  # Raise error for unsupported formats

        write_cleaned_lines_to_file(cleaned_path, cleaned_lines)  # Write cleaned lines to the cleaned file path
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_arff_with_scipy(input_path):
    """
    Attempt to load an ARFF file using scipy. Decodes byte strings when necessary.

    :param input_path: Path to the ARFF file.
    :return: pandas DataFrame loaded from the ARFF file.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
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


def load_arff_with_liac(input_path):
    """
    Load an ARFF file using the liac-arff library.

    :param input_path: Path to the ARFF file.
    :return: pandas DataFrame loaded from the ARFF file.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        with open(input_path, "r", encoding="utf-8") as f:  # Open the ARFF file for reading
            data = arff.load(f)  # Load using liac-arff

        return pd.DataFrame(data["data"], columns=[attr[0] for attr in data["attributes"]])  # Convert to DataFrame
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_arff_file(input_path):
    """
    Load an ARFF file, trying scipy first and falling back to liac-arff if needed.

    :param input_path: Path to the ARFF file.
    :return: pandas DataFrame loaded from the ARFF file.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        try:  # Try loading using scipy
            return load_arff_with_scipy(input_path)
        except Exception as e:  # If scipy fails, warn and try liac-arff
            verbose_output(
                f"{BackgroundColors.YELLOW}Warning: Failed to load ARFF with scipy ({e}). Trying with liac-arff...{Style.RESET_ALL}"
            )

            try:  # Try loading using liac-arff
                return load_arff_with_liac(input_path)
            except Exception as e2:  # If both fail, raise an error
                raise RuntimeError(f"Failed to load ARFF file with both scipy and liac-arff: {e2}")
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_csv_file(input_path):
    """
    Load a CSV file into a pandas DataFrame.

    :param input_path: Path to the CSV file.
    :return: pandas DataFrame containing the loaded dataset.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        df = pd.read_csv(input_path, low_memory=False)  # Load the CSV file
        df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names
        return df  # Return the DataFrame
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_parquet_file(input_path):
    """
    Load a Parquet file into a pandas DataFrame.

    :param input_path: Path to the Parquet file.
    :return: pandas DataFrame loaded from the Parquet file.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        pf = ParquetFile(input_path)  # Load the Parquet file using fastparquet
        return pf.to_pandas()  # Convert the Parquet file to a pandas DataFrame
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_txt_file(input_path):
    """
    Load a TXT file into a pandas DataFrame, assuming tab-separated values.

    :param input_path: Path to the TXT file.
    :return: pandas DataFrame containing the loaded dataset.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        df = pd.read_csv(input_path, sep="\t", low_memory=False)  # Load TXT file using tab separator
        df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names
        return df  # Return the DataFrame
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def load_dataset(input_path):
    """
    Load a dataset from a file in CSV, ARFF, TXT, or Parquet format into a pandas DataFrame.

    :param input_path: Path to the input dataset file.
    :return: pandas DataFrame containing the dataset.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Loading dataset from: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}"
        )  # Output the verbose message

        _, ext = os.path.splitext(input_path)  # Get the file extension of the input file
        ext = ext.lower()  # Convert the file extension to lowercase

        if ext == ".arff":  # If the file is in ARFF format
            df = load_arff_file(input_path)
        elif ext == ".csv":  # If the file is in CSV format
            df = load_csv_file(input_path)
        elif ext == ".parquet":  # If the file is in Parquet format
            df = load_parquet_file(input_path)
        elif ext == ".txt":  # If the file is in TXT format
            df = load_txt_file(input_path)
        else:  # Unsupported file format
            raise ValueError(f"Unsupported file format: {ext}")

        return df  # Return the loaded DataFrame
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def convert_to_arff(df, output_path):
    """
    Convert a pandas DataFrame to ARFF format and save it to the specified output path.

    :param df: pandas DataFrame to be converted.
    :param output_path: Path to save the converted ARFF file.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Converting DataFrame to ARFF format and saving to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
        )  # Output the verbose message

        attributes = [(col, "STRING") for col in df.columns]  # Define all attributes as strings
        df = df.astype(str)  # Ensure all values are strings

        arff_dict = {  # Create a dictionary to hold the ARFF data
            "description": "",  # Description of the dataset (can be left empty)
            "relation": "converted_data",  # Name of the relation (dataset)
            "attributes": attributes,  # List of attributes with their names and types
            "data": df.values.tolist(),  # Convert the DataFrame values to a list of lists for ARFF data
        }

        bytes_needed = estimate_bytes_arff(df, 512, attributes)  # Estimate size needed for ARFF output

        ensure_enough_space(output_path, bytes_needed)  # Ensure enough space to write the ARFF file

        with open(output_path, "w") as f:  # Open the output file for writing
            arff.dump(arff_dict, f)  # Dump the ARFF data into the file using liac-arff
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def convert_to_csv(df, output_path):
    """
    Convert a pandas DataFrame to CSV format and save it to the specified output path.

    :param df: pandas DataFrame to be converted.
    :param output_path: Path to save the converted CSV file.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Converting DataFrame to CSV format and saving to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
        )  # Output the verbose message

        bytes_needed = estimate_bytes_csv(df, overhead=512)  # Estimate size needed for CSV output
        ensure_enough_space(output_path, bytes_needed)  # Ensure enough space to write the CSV file

        df.to_csv(
            output_path, index=False
        )  # Save the DataFrame to the specified output path in CSV format, without the index
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def convert_to_parquet(df, output_path):
    """
    Convert a pandas DataFrame to PARQUET format and save it to the specified output path.

    :param df: pandas DataFrame to be converted.
    :param output_path: Path to save the converted PARQUET file.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Converting DataFrame to PARQUET format and saving to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
        )

        bytes_needed = estimate_bytes_parquet(df)  # Estimate size needed for PARQUET output
        ensure_enough_space(output_path, bytes_needed)  # Ensure enough space to write the PARQUET file

        df.to_parquet(
            output_path, index=False
        )  # Save the DataFrame to the specified output path in PARQUET format, without the index
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def convert_to_txt(df, output_path):
    """
    Convert a pandas DataFrame to TXT format and save it to the specified output path.

    :param df: pandas DataFrame to be converted.
    :param output_path: Path to save the converted TXT file.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Converting DataFrame to TXT format and saving to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
        )  # Output the verbose message

        try:  # Try to estimate size by dumping to a string buffer
            buf = io.StringIO()  # Create an in-memory string buffer
            df.to_csv(buf, sep="\t", index=False)  # Dump DataFrame to TXT in the buffer using tab as separator
            lines = buf.getvalue().splitlines()  # Get lines from the buffer
        except Exception:  # Fallback if dumping fails
            lines = None  # Set lines to None to use memory usage estimation

        if lines is not None:  # If lines were successfully obtained
            bytes_needed = estimate_bytes_from_lines(lines, overhead=512)  # Estimate size based on lines
        else:  # Fallback to memory usage estimation
            bytes_needed = estimate_bytes_parquet(df)  # Estimate size based on DataFrame memory usage

        ensure_enough_space(output_path, bytes_needed)  # Ensure enough space to write the TXT file

        df.to_csv(
            output_path, sep="\t", index=False
        )  # Save the DataFrame to the specified output path in TXT format, using tab as the separator and without the index
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def resolve_datasets_cfg(cfg: dict) -> dict:
    """
    Resolve datasets mapping from configuration.

    :param cfg: Configuration dictionary containing dataset mappings.
    :return: Mapping of dataset names to path lists or empty dict.
    """

    try:  # Wrap resolution logic to ensure production-safe monitoring
        datasets_cfg = cfg.get("datasets", {})  # Retrieve the datasets mapping from configuration
        if not isinstance(datasets_cfg, dict):  # Verify datasets_cfg is a mapping of dataset names to path lists
            return {}  # Return an empty mapping when datasets configuration is invalid
        return datasets_cfg  # Return the resolved datasets mapping when valid
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def process_dataset_paths(ds_paths: list, context: dict, cfg: dict) -> None:
    """
    Process configured paths for a single dataset entry.

    :param ds_paths: Iterable of paths configured for the dataset.
    :param context: Processing context dictionary with runtime values.
    :param cfg: Full configuration dictionary for fallback values.
    :return: None
    """

    try:  # Wrap processing logic to ensure production-safe monitoring
        if isinstance(ds_paths, str):  # Verify ds_paths is a single string path
            ds_paths = [ds_paths]  # Wrap single string into a list for uniform processing
        elif not isinstance(ds_paths, (list, tuple)):  # Verify dataset entry is an iterable of paths
            return  # Return early when ds_paths is not a valid iterable

        for ds_path in ds_paths:  # Iterate configured paths for this dataset entry
            effective_input = ds_path  # Set effective input directory for this configured path
            out_dir = context.get("output_directory")  # Retrieve output_directory from context which may be None
            effective_output_base = str(out_dir) if out_dir else os.path.join(str(ds_path), cfg.get("output_directory", "Converted"))  # Determine per-dataset base output directory as a string
            dataset_files = resolve_dataset_files(effective_input)  # Resolve dataset files for this configured path
            if not dataset_files:  # If no files found for this configured path
                print(f"{BackgroundColors.RED}No dataset files found in {BackgroundColors.CYAN}{effective_input}{Style.RESET_ALL}")  # Print error message when no files found
                continue  # Continue to next configured path when empty

            formats_list = resolve_formats(context.get("formats"))  # Normalize and validate output formats for this path
            len_dataset_files = len(dataset_files)  # Count files to process for progress reporting

            pbar = tqdm(dataset_files, desc=f"{BackgroundColors.CYAN}Converting {BackgroundColors.CYAN}{len_dataset_files}{BackgroundColors.GREEN} {'file' if len_dataset_files == 1 else 'files'}{Style.RESET_ALL}", unit="file", colour="green", total=len_dataset_files, leave=False, dynamic_ncols=True)  # Create a single-line progress bar for the conversion process

            for idx, input_path in enumerate(pbar, start=1):  # Iterate files for this configured path with index
                process_dataset_file(idx, len_dataset_files, input_path, effective_input, effective_output_base, formats_list, pbar)  # Delegate per-file processing to helper
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def process_dataset_file(idx: int, len_dataset_files: int, input_path: str, effective_input: str, effective_output_base: str, formats_list: list, pbar) -> None:
    """
    Process a single dataset file: clean, load and convert to requested formats.

    :param idx: Index of the file in the current processing batch.
    :param len_dataset_files: Total number of files in the current batch.
    :param input_path: Path to the input file being processed.
    :param effective_input: Effective input directory used for relative calculations.
    :param effective_output_base: Base output directory for converted files.
    :param formats_list: List of output formats to generate for this file.
    :param pbar: Progress bar instance used to display per-file progress.
    :return: None
    """

    try:  # Wrap per-file logic to ensure production-safe monitoring
        send_telegram_message(TELEGRAM_BOT, f"Converting file [{idx}/{len_dataset_files}]: {input_path}")  # Notify progress via Telegram before processing

        file = os.path.basename(str(input_path))  # Extract the file name from the full path
        name, ext = os.path.splitext(file)  # Split file name into base name and extension
        ext = ext.lower()  # Normalize extension to lowercase for matching

        formats_list = resolve_formats(formats_list) if formats_list is not None else []  # Normalize formats_list to a list when possible

        if pbar is not None:  # Verify progress bar instance exists before calling set_description
            try:  # Attempt to compute a relative path for the description
                rel = os.path.relpath(input_path, effective_input) if effective_input and os.path.isdir(effective_input) else os.path.basename(input_path)  # Compute relative path when possible
            except Exception:  # Fallback to basename on error
                rel = os.path.basename(input_path)  # Use basename when relpath fails
            pbar.set_description(f"{BackgroundColors.GREEN}Processing {BackgroundColors.CYAN}{rel}{Style.RESET_ALL}")  # Update the progress bar description with the relative path

        if ext not in [".arff", ".csv", ".parquet", ".txt"]:  # Skip unsupported file types early
            return  # Return early to the caller when unsupported extension

        dest_dir = resolve_destination_directory(effective_input, input_path, effective_output_base)  # Determine destination directory preserving relative structure
        create_directories(dest_dir)  # Ensure destination directory exists before writing

        cleaned_path = os.path.join(str(dest_dir), f"{name}{ext}")  # Path for saving the cleaned file prior to conversion
        clean_file(input_path, cleaned_path)  # Clean the file before conversion to normalize content

        df = load_dataset(cleaned_path)  # Load the cleaned dataset into a DataFrame for conversion
        if "arff" in formats_list:  # If ARFF format is requested for output
            convert_to_arff(df, os.path.join(str(dest_dir), f"{name}.arff"))  # Convert and save as ARFF format
        if "csv" in formats_list:  # If CSV format is requested for output
            convert_to_csv(df, os.path.join(str(dest_dir), f"{name}.csv"))  # Convert and save as CSV format
        if "parquet" in formats_list:  # If Parquet format is requested for output
            convert_to_parquet(df, os.path.join(str(dest_dir), f"{name}.parquet"))  # Convert and save as Parquet format
        if "txt" in formats_list:  # If TXT format is requested for output
            convert_to_txt(df, os.path.join(str(dest_dir), f"{name}.txt"))  # Convert and save as TXT format

        print()  # Print a newline for better readability between files in terminal output
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs in case of per-file failure
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram when per-file failure occurs
        raise  # Re-raise to preserve original failure semantics for upstream handling


def process_configured_datasets(context: dict) -> None:
    """
    Process datasets defined in configuration mapping.

    :param context: Dictionary with processing context values.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        cfg = context.get("cfg", {})  # Retrieve configuration section from context for processing
        datasets_cfg = resolve_datasets_cfg(cfg)  # Resolve and validate datasets mapping from config
        if not datasets_cfg:  # Verify datasets_cfg is a mapping of dataset names to path lists
            return  # Return immediately when configured datasets are not present or invalid

        for ds_name, ds_paths in datasets_cfg.items():  # Iterate each dataset entry in configuration mapping
            process_dataset_paths(ds_paths, context, cfg)  # Delegate per-dataset processing to helper function
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs in case of top-level failure
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram for top-level failures
        raise  # Re-raise to preserve original failure semantics for callers


def prepare_processing_context(context: dict) -> tuple:
    """
    Prepare common processing context values.

    :param context: Processing context dictionary with runtime values.
    :return: Tuple containing (cfg, input_directory, output_directory).
    """

    try:  # Wrap helper logic to ensure production-safe monitoring
        cfg = context.get("cfg", {})  # Retrieve configuration section from context for processing
        input_directory, output_directory = prepare_input_context(context, cfg)  # Prepare input and output directories for processing
        return cfg, input_directory, output_directory  # Return prepared context values
    except Exception as e:  # Catch exceptions inside helper
        print(str(e))  # Print helper exception to terminal for logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send helper exception via Telegram
        raise  # Re-raise to preserve failure semantics


def get_and_verify_dataset_files(input_directory: str, cfg: dict) -> tuple:
    """
    Gather dataset files and verify non-empty, printing message on empty.

    :param input_directory: Path to the input directory to scan for datasets.
    :param cfg: Configuration dictionary used for fallback values.
    :return: Tuple containing (dataset_files_list, len_dataset_files).
    """

    try:  # Wrap helper logic to ensure production-safe monitoring
        dataset_files, len_dataset_files = gather_dataset_files(input_directory, cfg)  # Gather dataset files and their count for processing
        if not dataset_files:  # If no dataset files were found
            print(f"{BackgroundColors.RED}No dataset files found in {BackgroundColors.CYAN}{input_directory}{Style.RESET_ALL}")  # Print error message when directory is empty
            return [], 0  # Return empty results to signal caller to exit early
        return dataset_files, len_dataset_files  # Return discovered dataset files and their count
    except Exception as e:  # Catch exceptions inside helper
        print(str(e))  # Print helper exception to terminal for logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send helper exception via Telegram
        raise  # Re-raise to preserve failure semantics


def create_progress_bar(dataset_files: list, len_dataset_files: int):
    """
    Create a progress bar for the conversion process.

    :param dataset_files: List of dataset files to display in the progress bar.
    :param len_dataset_files: Total number of dataset files for progress reporting.
    :return: A tqdm progress bar instance.
    """

    try:  # Wrap helper logic to ensure production-safe monitoring
        pbar = tqdm(dataset_files, desc=f"{BackgroundColors.CYAN}Converting {BackgroundColors.CYAN}{len_dataset_files}{BackgroundColors.GREEN} {'file' if len_dataset_files == 1 else 'files'}{Style.RESET_ALL}", unit="file", colour="green", total=len_dataset_files, leave=False, dynamic_ncols=True)  # Create a single-line progress bar for the conversion process
        return pbar  # Return the created progress bar instance
    except Exception as e:  # Catch exceptions inside helper
        print(str(e))  # Print helper exception to terminal for logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send helper exception via Telegram
        raise  # Re-raise to preserve failure semantics


def iterate_and_process_with_pbar(pbar, input_directory: str, output_directory: str, formats_list: list, len_dataset_files: int) -> None:
    """
    Iterate progress bar and delegate per-file processing to the per-file helper.

    :param pbar: Progress bar instance iterating dataset files.
    :param input_directory: Source input directory used for relative calculations.
    :param output_directory: Base output directory for converted files.
    :param formats_list: List of output formats to generate for this run.
    :param len_dataset_files: Total number of files in the current batch.
    :return: None
    """

    try:  # Wrap helper logic to ensure production-safe monitoring
        for idx, input_path in enumerate(pbar, start=1):  # Iterate through each dataset file with index
            process_single_input_file(idx, {"input_path": input_path, "input_directory": input_directory, "output_directory": output_directory, "formats_list": formats_list, "len_dataset_files": len_dataset_files, "pbar": pbar})  # Delegate per-file work to helper
    except Exception as e:  # Catch exceptions inside helper
        print(str(e))  # Print helper exception to terminal for logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send helper exception via Telegram
        raise  # Re-raise to preserve failure semantics


def process_input_directory(context: dict) -> None:
    """
    Process a single explicit input directory for conversion.

    :param context: Dictionary with processing context values.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        cfg, input_directory, output_directory = prepare_processing_context(context)  # Prepare context and directories for processing

        dataset_files, len_dataset_files = get_and_verify_dataset_files(input_directory, cfg)  # Gather dataset files and verify non-empty
        if not dataset_files:  # If no dataset files were found after verification
            return  # Exit early when helper signaled empty discovery

        formats_list = resolve_formats(context.get("formats"))  # Normalize and validate output formats for the run

        pbar = create_progress_bar(dataset_files, len_dataset_files)  # Create a progress bar for the conversion process

        iterate_and_process_with_pbar(pbar, input_directory, output_directory, formats_list, len_dataset_files)  # Iterate and process all files using helper
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs when top-level failure occurs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram for top-level failures
        raise  # Re-raise to preserve original failure semantics


def prepare_input_context(context: dict, cfg: dict) -> tuple:
    """
    Prepare input and output directory values from context and configuration.

    :param context: Processing context dictionary with runtime values.
    :param cfg: Configuration dictionary used for fallback defaults.
    :return: Tuple containing (input_directory, output_directory).
    """

    input_directory = context.get("input_directory")  # Retrieve provided input_directory from context
    output_directory = context.get("output_directory")  # Retrieve provided output_directory from context
    
    if not output_directory:  # If output directory is not provided, get it from defaults
        output_directory = cfg.get("output_directory", "Converted")  # Default to 'Converted' when not specified
    
    return input_directory, output_directory  # Return prepared input and output directories


def gather_dataset_files(input_directory: str, cfg: dict) -> tuple:
    """
    Gather dataset files from the input directory and return them with count.

    :param input_directory: Path to the input directory to scan for datasets.
    :param cfg: Configuration dictionary used for fallback values.
    :return: Tuple containing (dataset_files_list, len_dataset_files).
    """

    dataset_files = resolve_dataset_files(input_directory)  # Get all dataset files from the input directory
    len_dataset_files = len(dataset_files)  # Get the number of dataset files found
    
    return dataset_files, len_dataset_files  # Return both the list and its length


def process_single_input_file(idx: int, params: dict) -> None:
    """
    Process a single input file: clean, load and convert to requested formats.

    :param idx: Index of the file in the current processing batch.
    :param params: Dictionary with keys: input_path, input_directory, output_directory, formats_list, len_dataset_files, pbar.
    :return: None
    """

    try:  # Wrap per-file logic to ensure production-safe monitoring
        input_path = params.get("input_path")  # Extract input_path from params for this file
        input_directory = params.get("input_directory")  # Extract input_directory from params for relative pathing
        output_directory = params.get("output_directory")  # Extract output_directory from params for writing outputs
        formats_list = params.get("formats_list")  # Extract formats_list from params to know desired outputs
        len_dataset_files = params.get("len_dataset_files")  # Extract total count for progress messages
        pbar = params.get("pbar")  # Extract progress bar instance for updates

        send_telegram_message(TELEGRAM_BOT, f"Converting file [{idx}/{len_dataset_files}]: {input_path}")  # Notify progress via Telegram for current file

        file = os.path.basename(str(input_path))  # Extract the file name from the full path
        name, ext = os.path.splitext(file)  # Split file name into base name and extension
        ext = ext.lower()  # Normalize extension to lowercase for matching

        formats_list = resolve_formats(formats_list) if formats_list is not None else []  # Normalize formats_list to a list when possible

        if pbar is not None:  # Verify progress bar instance exists before calling set_description
            try:  # Attempt to compute relative path for description
                rel = os.path.relpath(input_path, input_directory) if input_directory and os.path.isdir(input_directory) else os.path.basename(input_path)  # Compute relative path when possible
            except Exception:  # Fallback to basename on error
                rel = os.path.basename(input_path)  # Use basename when relpath fails
            pbar.set_description(f"{BackgroundColors.GREEN}Processing {BackgroundColors.CYAN}{rel}{Style.RESET_ALL}")  # Update the progress bar description with the relative path

        if ext not in [".arff", ".csv", ".parquet", ".txt"]:  # Skip unsupported file types early
            return  # Return early to the caller when unsupported extension

        dest_dir = resolve_destination_directory(input_directory, input_path, output_directory)  # Determine destination directory for converted files
        create_directories(dest_dir)  # Ensure destination directory exists before writing

        cleaned_path = os.path.join(str(dest_dir), f"{name}{ext}")  # Path for saving the cleaned file prior to conversion
        clean_file(input_path, cleaned_path)  # Clean the file before conversion to normalize content

        df = load_dataset(cleaned_path)  # Load the cleaned dataset into a DataFrame for conversion
        if "arff" in formats_list:  # If ARFF format is requested for output
            convert_to_arff(df, os.path.join(str(dest_dir), f"{name}.arff"))  # Convert and save as ARFF format
        if "csv" in formats_list:  # If CSV format is requested for output
            convert_to_csv(df, os.path.join(str(dest_dir), f"{name}.csv"))  # Convert and save as CSV format
        if "parquet" in formats_list:  # If Parquet format is requested for output
            convert_to_parquet(df, os.path.join(str(dest_dir), f"{name}.parquet"))  # Convert and save as Parquet format
        if "txt" in formats_list:  # If TXT format is requested for output
            convert_to_txt(df, os.path.join(str(dest_dir), f"{name}.txt"))  # Convert and save as TXT format

        print()  # Print a newline for better readability between files in terminal output
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert for per-file failures
        print(str(e))  # Print error to terminal for server logs in case of per-file failure
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram when per-file failure occurs
        raise  # Re-raise to preserve original failure semantics for upstream handling


def batch_convert(input_directory=None, output_directory=None, formats=None):
    """
    Batch converts dataset files from the input directory into multiple output formats

    :param input_directory: Path to the input directory containing dataset files.
    :param output_directory: Path to the output directory where converted files will be saved.
    :param formats: List of output formats to generate (e.g., ["arff", "csv"]). If None, all formats are generated.
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(f"{BackgroundColors.GREEN}Batch converting dataset files from {BackgroundColors.CYAN}{input_directory}{BackgroundColors.GREEN} to {BackgroundColors.CYAN}{output_directory}{Style.RESET_ALL}")  # Output the verbose message

        cfg = DEFAULTS.get("dataset_converter", {}) if DEFAULTS else {}  # Get default configuration for dataset converter if available

        if not input_directory:  # Verify if no input_directory argument was given
            context = {"cfg": cfg, "output_directory": output_directory, "formats": formats}  # Build context dictionary for configured datasets processing
            process_configured_datasets(context)  # Process datasets defined in configuration
            return  # Completed processing configured datasets, exit early

        context = {"cfg": cfg, "input_directory": input_directory, "output_directory": output_directory, "formats": formats}  # Build context dictionary for input-directory processing
        process_input_directory(context)  # Process the explicit input directory
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
    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
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
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def main():
    """
    Main function.

    :return: None
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        print(
            f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Multi-Format Dataset Converter{BackgroundColors.GREEN}!{Style.RESET_ALL}\n"
        )  # Output the Welcome message
        start_time = datetime.datetime.now()  # Get the start time of the program
        
        initialize_defaults()  # Initialize DEFAULTS from get_default_config() and config.yaml
        
        setup_telegram_bot()  # Setup Telegram bot if configured
        
        args = parse_cli_arguments()  # Parse CLI arguments

        input_paths, output_path = resolve_io_paths(args)  # Resolve and validate paths, returning list of inputs
        if input_paths is None or output_path is None:  # If either resolution failed
            return  # Exit early when inputs/outputs are invalid

        send_telegram_message(TELEGRAM_BOT, f"Multi-Format Dataset Converter started for input: {', '.join(input_paths)} and output: {output_path} at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")  # Notify start via Telegram for all inputs

        configure_verbose_mode(args)  # Enable verbose mode if requested

        for input_path in input_paths:  # Iterate through each resolved input path
            batch_convert(input_path, output_path, formats=args.formats if args.formats else None)  # Perform batch conversion per input path

        finish_time = datetime.datetime.now()  # Get the finish time of the program
        print(
            f"{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
        )  # Output the start and finish times
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
        )  # Output the end of the program message
        
        send_telegram_message(TELEGRAM_BOT, f"Multi-Format Dataset Converter started for input: {input_path} and output: {output_path} finished. Execution time: {calculate_execution_time(start_time, finish_time)}.")  # Notify finish via Telegram

        (
            atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None
        )  # Register the play_sound function to be called when the program exits if RUN_FUNCTIONS["Play Sound"] is True
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
