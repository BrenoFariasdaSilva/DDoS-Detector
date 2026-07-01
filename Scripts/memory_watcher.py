"""
================================================================================
<PROJECT OR SCRIPT TITLE>
================================================================================
Author      : Breno Farias da Silva
Created     : <YYYY-MM-DD>
Description :
    <Provide a concise and complete overview of what this script does.>
    <Mention its purpose, scope, and relevance to the larger project.>

    Key features include:
        - <Feature 1 — e.g., automatic data loading and preprocessing>
        - <Feature 2 — e.g., model training and evaluation>
        - <Feature 3 — e.g., visualization or report generation>
        - <Feature 4 — e.g., logging or notification system>
        - <Feature 5 — e.g., integration with other modules or datasets>

Usage:
    1. <Explain any configuration steps before running, such as editing variables or paths.>
    2. <Describe how to execute the script — typically via Makefile or Python.>
        $ make <target>   or   $ python <script_name>.py
    3. <List what outputs are expected or where results are saved.>

Outputs:
    - <Output file or directory 1 — e.g., results.csv>
    - <Output file or directory 2 — e.g., Feature_Analysis/plots/>
    - <Output file or directory 3 — e.g., logs/output.txt>

TODOs:
    - <Add a task or improvement — e.g., implement CLI argument parsing.>
    - <Add another improvement — e.g., extend support to Parquet files.>
    - <Add optimization — e.g., parallelize evaluation loop.>
    - <Add robustness — e.g., error handling or data validation.>

Dependencies:
    - Python >= <version>
    - <Library 1 — e.g., pandas>
    - <Library 2 — e.g., numpy>
    - <Library 3 — e.g., scikit-learn>
    - <Library 4 — e.g., matplotlib, seaborn, tqdm, colorama>

Assumptions & Notes:
    - <List any key assumptions — e.g., last column is the target variable.>
    - <Mention data format — e.g., CSV files only.>
    - <Mention platform or OS-specific notes — e.g., sound disabled on Windows.>
    - <Note on output structure or reusability.>
"""

import argparse  # Parse watcher command-line arguments
import atexit  # For playing a sound when the program finishes
import datetime  # For getting the current date and time
import json  # Write compact JSONL and summary records
import os  # For running a command in the terminal
import platform  # For getting the operating system name
import sys  # For system-specific parameters and functions
import tempfile  # Write atomic summary and terminal records
import time  # Drive low-overhead sample intervals
from colorama import Style  # For coloring the terminal
from pathlib import Path  # For handling file paths
from typing import Any, Dict, List, Optional, Tuple  # Type watcher records and state

import psutil  # Inspect target process, child processes, and system memory

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Resolve repository root for local imports
if str(PROJECT_ROOT) not in sys.path:  # Ensure root modules are importable when launched from Scripts
    sys.path.insert(0, str(PROJECT_ROOT))  # Add repository root before importing project utilities

from Logger import Logger  # For logging output to both terminal and file


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

# Logger Setup:
logger = Logger(f"./Logs/{Path(__file__).stem}.log", clean=True)  # Create a Logger instance
sys.stdout = logger  # Redirect stdout to the logger
sys.stderr = logger  # Redirect stderr to the logger

# Sound Constants:
SOUND_COMMANDS = {
    "Darwin": "afplay",
    "Linux": "aplay",
    "Windows": "start",
}  # The commands to play a sound for each operating system
SOUND_FILE = "./.assets/Sounds/NotificationSound.wav"  # The path to the sound file

# RUN_FUNCTIONS:
RUN_FUNCTIONS = {
    "Play Sound": False,  # Disable completion sound for background watcher runs
}

# Functions Definitions:


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
        raise  # Re-raise to preserve original failure semantics


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

    :param: None
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


def parse_watcher_args():  # Parse watcher arguments without target script names
    """
    Parse command-line arguments for the independent memory watcher.

    :return: Namespace containing watcher runtime settings.
    """

    parser = argparse.ArgumentParser(description="Sample memory, swap, process, and phase state for a target PID.")  # Create watcher parser
    parser.add_argument("--target-pid", type=int, required=True, help="Target process PID")  # Accept target PID
    parser.add_argument("--target-create-time", type=float, default=None, help="Target process creation timestamp")  # Accept target identity marker
    parser.add_argument("--run-dir", type=str, required=True, help="Unique watcher run directory")  # Accept output directory
    parser.add_argument("--phase-state-path", type=str, required=True, help="Atomic phase-state JSON file path")  # Accept phase-state path
    parser.add_argument("--sample-interval-seconds", type=float, default=2.0, help="Sampling interval in seconds")  # Accept sample interval
    parser.add_argument("--system-memory-threshold-percent", type=float, default=90.0, help="System memory pressure threshold percent")  # Accept system threshold
    parser.add_argument("--process-rss-threshold-gb", type=float, default=None, help="Target RSS threshold in GB")  # Accept optional process RSS threshold
    parser.add_argument("--capture-process-tree", action="store_true", default=False, help="Record recursive child process rows")  # Enable process-tree rows
    parser.add_argument("--capture-tracemalloc", action="store_true", default=False, help="Record tracemalloc mode in summary")  # Record tracemalloc mode
    parser.add_argument("--keep-after-target-exit-seconds", type=float, default=5.0, help="Seconds to remain alive after target disappears")  # Accept terminal grace period
    return parser.parse_args()  # Return parsed watcher arguments


def utc_timestamp():  # Build UTC timestamp strings
    """
    Return the current UTC timestamp in ISO-8601 format.

    :return: ISO-8601 UTC timestamp.
    """

    return datetime.datetime.now(datetime.timezone.utc).isoformat()  # Return timezone-aware timestamp


def sanitize_json_value(value: Any, depth: int = 0) -> Any:  # Normalize values for compact JSON output
    """
    Convert arbitrary values into compact JSON-safe objects.

    :param value: Value to normalize.
    :param depth: Current recursion depth.
    :return: JSON-safe value.
    """

    if depth > 4:  # Limit nested metadata expansion
        return str(value)  # Return string representation at depth limit
    if value is None or isinstance(value, (str, int, float, bool)):  # Allow scalar JSON values
        return value  # Return scalar unchanged
    if isinstance(value, Path):  # Normalize Path values
        return str(value)  # Return path string
    if isinstance(value, (list, tuple, set)):  # Normalize sequence values
        return [sanitize_json_value(item, depth + 1) for item in list(value)[:50]]  # Return bounded normalized list
    if isinstance(value, dict):  # Normalize mapping values
        return {str(k): sanitize_json_value(v, depth + 1) for k, v in list(value.items())[:80]}  # Return bounded normalized mapping
    return str(value)  # Return safe string representation


def append_jsonl(filepath: str, record: Dict[str, Any]) -> bool:  # Append a JSONL record best-effort
    """
    Append one JSON record to a JSONL file.

    :param filepath: Destination JSONL file.
    :param record: Record to write.
    :return: True when the write completed.
    """

    try:  # Protect watcher output persistence
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure destination directory exists
        with open(filepath, "a", encoding="utf-8") as file_obj:  # Open JSONL file in append mode
            file_obj.write(json.dumps(sanitize_json_value(record), sort_keys=True) + "\n")  # Write normalized JSON row
            file_obj.flush()  # Flush row promptly for killed-target diagnostics
        return True  # Report successful write
    except Exception as exc:  # Keep watcher alive on output errors
        print(f"{BackgroundColors.YELLOW}[WARNING] Failed to append JSONL record to {filepath}: {exc}{Style.RESET_ALL}")  # Log persistence failure
        return False  # Report failed write


def atomic_write_json(filepath: str, record: Dict[str, Any]) -> bool:  # Write JSON atomically best-effort
    """
    Atomically write a JSON document.

    :param filepath: Destination JSON file.
    :param record: JSON-safe record to write.
    :return: True when the replacement completed.
    """

    tmp_path = None  # Track temporary file path for cleanup
    try:  # Protect atomic JSON persistence
        os.makedirs(os.path.dirname(filepath), exist_ok=True)  # Ensure destination directory exists
        fd, tmp_path = tempfile.mkstemp(prefix=".tmp_memory_watcher_", suffix=".json", dir=os.path.dirname(filepath))  # Create temporary JSON file
        with os.fdopen(fd, "w", encoding="utf-8") as file_obj:  # Open temporary file descriptor
            json.dump(sanitize_json_value(record), file_obj, indent=2, sort_keys=True)  # Write normalized summary JSON
            file_obj.write("\n")  # Terminate JSON with newline
            file_obj.flush()  # Flush Python buffer
            os.fsync(file_obj.fileno())  # Flush file data to disk where supported
        os.replace(tmp_path, filepath)  # Atomically replace destination file
        return True  # Report successful atomic write
    except Exception as exc:  # Keep watcher alive on summary errors
        print(f"{BackgroundColors.YELLOW}[WARNING] Failed to write JSON file {filepath}: {exc}{Style.RESET_ALL}")  # Log atomic write failure
        return False  # Report failed write
    finally:  # Remove leftover temporary file
        if tmp_path and os.path.exists(tmp_path):  # Verify temporary path remains after failure
            try:  # Protect temporary cleanup
                os.remove(tmp_path)  # Remove leftover temporary file
            except Exception:  # Ignore cleanup failure
                pass  # Leave filesystem state unchanged on cleanup failure


def load_phase_state(phase_state_path: str, previous_state: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:  # Load latest atomic phase state
    """
    Load the latest valid phase-state JSON, preserving the previous state on transient failures.

    :param phase_state_path: Path to atomic phase-state JSON.
    :param previous_state: Last valid state.
    :return: Tuple of state and optional error string.
    """

    try:  # Protect phase-state reads
        if not phase_state_path or not os.path.exists(phase_state_path):  # Treat absent state as transient
            return previous_state, None  # Return previous state without error
        with open(phase_state_path, "r", encoding="utf-8") as file_obj:  # Open phase-state file
            state = json.load(file_obj)  # Parse JSON state
        if isinstance(state, dict):  # Accept only mapping state
            return state, None  # Return parsed phase state
        return previous_state, "phase_state_not_mapping"  # Preserve previous state on unexpected shape
    except Exception as exc:  # Preserve watcher process on partial or malformed JSON
        return previous_state, str(exc)  # Return previous state with read error


def get_phase_identity(phase_state: Dict[str, Any]) -> Dict[str, Any]:  # Extract compact phase identity
    """
    Extract compact identity fields for peak and threshold records.

    :param phase_state: Current phase-state metadata.
    :return: Compact identity mapping.
    """

    fields = ["event_id", "run_id", "timestamp", "phase", "execution_mode", "dataset_identity", "dataset_source", "attack_scope", "data_source", "experiment_mode", "augmentation_ratio", "feature_set_name", "hyperparameter_mode", "classifier_name", "classifier_params_digest", "classifier_params_reference", "n_jobs", "cache_identity", "cache_reference", "event_outcome"]  # Define compact identity fields
    return {field: phase_state.get(field) for field in fields if field in phase_state}  # Return present identity fields


def emit_phase_event(run_dir: str, event_type: str, phase_state: Dict[str, Any], details: Optional[Dict[str, Any]] = None) -> None:  # Emit watcher event row
    """
    Write a watcher event to phase_events.jsonl.

    :param run_dir: Watcher run directory.
    :param event_type: Watcher event type.
    :param phase_state: Current phase-state metadata.
    :param details: Additional event details.
    :return: None.
    """

    event = {"timestamp": utc_timestamp(), "event_type": event_type, "phase_metadata": get_phase_identity(phase_state), "details": details or {}}  # Build compact event row
    append_jsonl(os.path.join(run_dir, "phase_events.jsonl"), event)  # Append phase event row


def summarize_cmdline(process: psutil.Process) -> Dict[str, Any]:  # Summarize command line safely
    """
    Return a compact, bounded command-line summary for a process.

    :param process: Process object to inspect.
    :return: Safe command-line summary.
    """

    try:  # Protect command-line access
        cmdline = process.cmdline()  # Read process command-line list
        first_arg = os.path.basename(cmdline[0]) if cmdline else None  # Capture executable basename only
        return {"argc": len(cmdline), "program": first_arg, "contains_target_script_literal": any("stacking.py" in str(arg) for arg in cmdline)}  # Return bounded command-line metadata
    except Exception as exc:  # Handle access-denied and disappearing children
        return {"argc": None, "program": None, "error": str(exc)}  # Return explicit unavailable summary


def read_process_memory(process: psutil.Process) -> Dict[str, Optional[int]]:  # Read memory metrics with optional USS and swap
    """
    Read process memory metrics.

    :param process: Process object to inspect.
    :return: Memory metric mapping.
    """

    rss_bytes = None  # Initialize RSS as unavailable
    vms_bytes = None  # Initialize VMS as unavailable
    uss_bytes = None  # Initialize USS as unavailable
    swap_bytes = None  # Initialize process swap as unavailable
    try:  # Read standard memory metrics
        mem_info = process.memory_info()  # Read RSS and VMS metrics
        rss_bytes = int(getattr(mem_info, "rss", 0))  # Store RSS bytes
        vms_bytes = int(getattr(mem_info, "vms", 0))  # Store VMS bytes
    except Exception:  # Keep unavailable values as null
        pass  # Preserve null metrics
    try:  # Read extended memory metrics when supported
        full_info = process.memory_full_info()  # Read USS and swap metrics when available
        uss_bytes = int(getattr(full_info, "uss", uss_bytes)) if getattr(full_info, "uss", None) is not None else uss_bytes  # Store USS bytes when available
        swap_bytes = int(getattr(full_info, "swap", swap_bytes)) if getattr(full_info, "swap", None) is not None else swap_bytes  # Store swap bytes when available
    except Exception:  # Keep extended metrics unavailable on this platform
        pass  # Preserve null extended metrics
    return {"rss_bytes": rss_bytes, "vms_bytes": vms_bytes, "uss_bytes": uss_bytes, "swap_bytes": swap_bytes}  # Return process memory metrics


def get_target_process(target_pid: int, expected_create_time: Optional[float]) -> Tuple[Optional[psutil.Process], Optional[str], Optional[float]]:  # Resolve target identity
    """
    Resolve the target process and verify its creation timestamp when provided.

    :param target_pid: Target PID.
    :param expected_create_time: Expected process creation timestamp.
    :return: Tuple of process, optional status, and actual creation timestamp.
    """

    try:  # Resolve process by PID
        process = psutil.Process(target_pid)  # Create process handle
        actual_create_time = float(process.create_time())  # Read creation timestamp
        if expected_create_time is not None and abs(actual_create_time - expected_create_time) > 1.0:  # Detect PID reuse by creation time
            return None, "target_identity_mismatch", actual_create_time  # Treat mismatched PID as disappeared target
        return process, None, actual_create_time  # Return verified process handle
    except psutil.NoSuchProcess:  # Target PID is gone
        return None, "target_disappeared", None  # Return disappeared status
    except Exception as exc:  # Preserve watcher loop on access errors
        return None, f"target_unavailable:{exc}", None  # Return unavailable status


def collect_child_process_rows(process: psutil.Process, timestamp: str, phase_state: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], int, int, int, Optional[str]]:  # Collect recursive child process metrics
    """
    Collect metrics for actual child processes only.

    :param process: Target process.
    :param timestamp: Sample timestamp.
    :param phase_state: Current phase metadata.
    :return: Rows, direct count, recursive count, non-deduplicated RSS aggregate, optional error.
    """

    rows = []  # Initialize process-tree rows
    direct_count = 0  # Initialize direct child count
    recursive_count = 0  # Initialize recursive child count
    nondedup_rss = 0  # Initialize non-deduplicated recursive RSS aggregate
    try:  # Read actual child processes
        direct_children = process.children(recursive=False)  # Read direct child PIDs
        recursive_children = process.children(recursive=True)  # Read recursive child PIDs
        direct_count = len(direct_children)  # Store direct child count
        recursive_count = len(recursive_children)  # Store recursive child count
        for child in recursive_children:  # Iterate recursive child processes
            memory = read_process_memory(child)  # Read child memory metrics
            nondedup_rss += int(memory.get("rss_bytes") or 0)  # Accumulate non-deduplicated RSS
            try:  # Read child status
                child_status = child.status()  # Get child process status
            except Exception as exc:  # Handle disappearing child
                child_status = f"unavailable:{exc}"  # Store explicit unavailable status
            try:  # Read child name
                child_name = child.name()  # Get child process name
            except Exception as exc:  # Handle inaccessible child name
                child_name = f"unavailable:{exc}"  # Store explicit unavailable name
            try:  # Read child thread count
                child_threads = child.num_threads()  # Count native threads for child process
            except Exception:  # Keep unavailable thread count null
                child_threads = None  # Store null child thread count
            try:  # Read parent PID
                parent_pid = child.ppid()  # Read parent PID
            except Exception:  # Keep unavailable parent PID null
                parent_pid = None  # Store null parent PID
            rows.append({"timestamp": timestamp, "target_pid": process.pid, "child_pid": child.pid, "parent_pid": parent_pid, "name": child_name, "status": child_status, "rss_bytes": memory.get("rss_bytes"), "vms_bytes": memory.get("vms_bytes"), "thread_count": child_threads, "cmdline_summary": summarize_cmdline(child), "phase_metadata": get_phase_identity(phase_state)})  # Append child row
        return rows, direct_count, recursive_count, nondedup_rss, None  # Return collected process-tree metrics
    except Exception as exc:  # Preserve watcher loop on process-tree errors
        return rows, direct_count, recursive_count, nondedup_rss, str(exc)  # Return partial process-tree metrics and error


def build_sample_record(args: argparse.Namespace, process: psutil.Process, phase_state: Dict[str, Any], phase_error: Optional[str]) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Optional[str]]:  # Build one sample row
    """
    Build one memory sample record and optional process-tree rows.

    :param args: Watcher arguments.
    :param process: Target process.
    :param phase_state: Current phase-state metadata.
    :param phase_error: Phase-state read error, if any.
    :return: Tuple of sample record, child rows, and optional sampling error.
    """

    timestamp = utc_timestamp()  # Capture sample timestamp
    sample_error = None  # Initialize sample error
    child_rows = []  # Initialize child process rows
    direct_count = None  # Initialize direct child count
    recursive_count = None  # Initialize recursive child count
    nondedup_child_rss = None  # Initialize child RSS aggregate
    process_tree_error = None  # Initialize process-tree error
    memory = read_process_memory(process)  # Read target memory metrics
    try:  # Read target status
        status = process.status()  # Get target status
    except Exception as exc:  # Capture status failure
        status = f"unavailable:{exc}"  # Store explicit unavailable status
        sample_error = str(exc)  # Store sampling error
    try:  # Read CPU percent
        cpu_percent = process.cpu_percent(interval=None)  # Read non-blocking process CPU percent
    except Exception as exc:  # Capture CPU failure
        cpu_percent = None  # Store null CPU percent
        sample_error = str(exc)  # Store sampling error
    try:  # Read thread count
        thread_count = process.num_threads()  # Count main process native threads
    except Exception as exc:  # Capture thread-count failure
        thread_count = None  # Store null thread count
        sample_error = str(exc)  # Store sampling error
    try:  # Read system memory pressure
        virtual_memory = psutil.virtual_memory()  # Read system virtual memory snapshot
        available_memory = int(getattr(virtual_memory, "available", 0))  # Store available system memory bytes
        memory_percent = float(getattr(virtual_memory, "percent", 0.0))  # Store system memory used percent
    except Exception as exc:  # Capture system memory failure
        available_memory = None  # Store null available memory
        memory_percent = None  # Store null memory percent
        sample_error = str(exc)  # Store sampling error
    try:  # Read system swap pressure
        swap_memory = psutil.swap_memory()  # Read system swap snapshot
        swap_percent = float(getattr(swap_memory, "percent", 0.0))  # Store system swap used percent
    except Exception as exc:  # Capture swap failure
        swap_percent = None  # Store null swap percent
        sample_error = str(exc)  # Store sampling error
    if args.capture_process_tree:  # Record child process rows only when enabled
        child_rows, direct_count, recursive_count, nondedup_child_rss, process_tree_error = collect_child_process_rows(process, timestamp, phase_state)  # Collect process-tree rows
        if process_tree_error:  # Preserve process-tree error in sample
            sample_error = process_tree_error  # Store process-tree error as sample error
    else:  # Still count children without row expansion when disabled
        try:  # Read child counts without process-tree row emission
            direct_count = len(process.children(recursive=False))  # Count direct child processes
            recursive_count = len(process.children(recursive=True))  # Count recursive child processes
        except Exception as exc:  # Capture child-count failure
            sample_error = str(exc)  # Store child-count error
    sample = {"timestamp": timestamp, "target_pid": args.target_pid, "target_create_time": args.target_create_time, "target_status": status, "target_cpu_percent": cpu_percent, "target_rss_bytes": memory.get("rss_bytes"), "target_vms_bytes": memory.get("vms_bytes"), "target_uss_bytes": memory.get("uss_bytes"), "target_swap_bytes": memory.get("swap_bytes"), "target_thread_count": thread_count, "direct_child_process_count": direct_count, "recursive_child_process_count": recursive_count, "recursive_child_process_nondedup_rss_bytes": nondedup_child_rss, "system_available_memory_bytes": available_memory, "system_memory_used_percent": memory_percent, "system_swap_used_percent": swap_percent, "phase_metadata": get_phase_identity(phase_state), "phase_state_error": phase_error, "sampling_error": sample_error}  # Build memory sample record
    return sample, child_rows, sample_error  # Return sample and process-tree rows


def update_peak(summary: Dict[str, Any], sample: Dict[str, Any], phase_state: Dict[str, Any], run_dir: str) -> None:  # Update peak metrics and emit meaningful peak events
    """
    Update summary peak fields and emit bounded peak events.

    :param summary: Mutable summary state.
    :param sample: Current sample record.
    :param phase_state: Current phase-state metadata.
    :param run_dir: Watcher run directory.
    :return: None.
    """

    peak_rules = [("target_rss_bytes", "peak_target_rss_bytes", 64 * 1024 * 1024), ("target_vms_bytes", "peak_target_vms_bytes", 64 * 1024 * 1024), ("target_uss_bytes", "peak_target_uss_bytes", 64 * 1024 * 1024), ("system_memory_used_percent", "peak_system_memory_used_percent", 1.0), ("system_swap_used_percent", "peak_system_swap_used_percent", 1.0), ("target_thread_count", "peak_target_thread_count", 1), ("recursive_child_process_count", "peak_recursive_child_process_count", 1)]  # Define meaningful peak increments
    for sample_key, peak_key, increment in peak_rules:  # Iterate peak rule definitions
        value = sample.get(sample_key)  # Read sample value
        if value is None:  # Skip unavailable values
            continue  # Continue to next metric
        previous = summary.get(peak_key)  # Read previous peak
        if previous is None or value > previous:  # Update stored peak on every higher value
            summary[peak_key] = value  # Store new peak value
            summary[f"{peak_key}_phase"] = get_phase_identity(phase_state)  # Store phase identity at peak
            if previous is None or value >= previous + increment:  # Emit only meaningful peak event increments
                event_details = {"metric": sample_key, "previous": previous, "value": value, "rss_vms_warning": "RSS and VMS do not identify the exact allocation source."}  # Build peak event details
                emit_phase_event(run_dir, "peak", phase_state, event_details)  # Write peak event row
                summary.setdefault("peak_events", []).append({"timestamp": utc_timestamp(), "metric": sample_key, "previous": previous, "value": value, "phase_metadata": get_phase_identity(phase_state)})  # Store summary peak event


def update_thresholds(args: argparse.Namespace, summary: Dict[str, Any], sample: Dict[str, Any], phase_state: Dict[str, Any], threshold_state: Dict[str, bool], run_dir: str) -> None:  # Update threshold state transitions
    """
    Emit threshold-enter and threshold-exit events only at transitions.

    :param args: Watcher arguments.
    :param summary: Mutable summary state.
    :param sample: Current sample record.
    :param phase_state: Current phase-state metadata.
    :param threshold_state: Mutable threshold active flags.
    :param run_dir: Watcher run directory.
    :return: None.
    """

    rules = []  # Initialize threshold rules
    if args.system_memory_threshold_percent is not None:  # Enable system memory threshold when configured
        rules.append(("system_memory", sample.get("system_memory_used_percent"), float(args.system_memory_threshold_percent), "percent"))  # Add system memory threshold rule
    if args.process_rss_threshold_gb is not None:  # Enable process RSS threshold when configured
        rules.append(("process_rss", sample.get("target_rss_bytes"), float(args.process_rss_threshold_gb) * 1024 * 1024 * 1024, "bytes"))  # Add process RSS threshold rule
    for name, value, threshold, unit in rules:  # Iterate threshold rules
        if value is None:  # Skip unavailable metrics
            continue  # Continue to next threshold
        active = value >= threshold  # Resolve threshold active state
        previous_active = threshold_state.get(name, False)  # Read prior active state
        if active and not previous_active:  # Emit enter event only once per crossing
            threshold_state[name] = True  # Mark threshold active
            event_details = {"threshold_name": name, "state": "enter", "value": value, "threshold": threshold, "unit": unit}  # Build threshold-enter details
            emit_phase_event(run_dir, "threshold_enter", phase_state, event_details)  # Write threshold-enter event row
            summary.setdefault("threshold_crossings", []).append({"timestamp": utc_timestamp(), "threshold_name": name, "state": "enter", "value": value, "threshold": threshold, "unit": unit, "phase_metadata": get_phase_identity(phase_state)})  # Store threshold-enter summary
        elif not active and previous_active:  # Emit exit event only once per crossing
            threshold_state[name] = False  # Mark threshold inactive
            event_details = {"threshold_name": name, "state": "exit", "value": value, "threshold": threshold, "unit": unit}  # Build threshold-exit details
            emit_phase_event(run_dir, "threshold_exit", phase_state, event_details)  # Write threshold-exit event row
            summary.setdefault("threshold_crossings", []).append({"timestamp": utc_timestamp(), "threshold_name": name, "state": "exit", "value": value, "threshold": threshold, "unit": unit, "phase_metadata": get_phase_identity(phase_state)})  # Store threshold-exit summary


def build_initial_summary(args: argparse.Namespace) -> Dict[str, Any]:  # Build initial summary state
    """
    Build mutable summary metadata for the watcher run.

    :param args: Watcher arguments.
    :return: Summary dictionary.
    """

    return {"target_pid": args.target_pid, "target_create_time": args.target_create_time, "watcher_pid": os.getpid(), "start_timestamp": utc_timestamp(), "end_timestamp": None, "target_completion_status": "running", "total_samples": 0, "peak_target_rss_bytes": None, "peak_target_vms_bytes": None, "peak_target_uss_bytes": None, "peak_system_memory_used_percent": None, "peak_system_swap_used_percent": None, "peak_target_thread_count": None, "peak_direct_child_process_count": None, "peak_recursive_child_process_count": None, "threshold_crossings": [], "peak_events": [], "sampling_errors": [], "tracemalloc_enabled": bool(args.capture_tracemalloc), "native_thread_note": "Native library threads are represented in target_thread_count when the OS exposes them through psutil.", "sleeping_entry_note": "Sleeping htop entries are not classified from appearance alone; child PIDs are recorded only from the OS process tree.", "rss_vms_warning": "RSS and VMS do not identify the exact allocation source, and non-deduplicated RSS aggregates can double count shared pages."}  # Return summary state


def write_final_summary(run_dir: str, summary: Dict[str, Any], status: str) -> None:  # Write terminal summary JSON
    """
    Write summary.json for the watcher run.

    :param run_dir: Watcher run directory.
    :param summary: Summary state.
    :param status: Final target completion status.
    :return: None.
    """

    summary["end_timestamp"] = utc_timestamp()  # Store watcher end timestamp
    summary["target_completion_status"] = status  # Store final target status
    atomic_write_json(os.path.join(run_dir, "summary.json"), summary)  # Write final summary atomically


def run_memory_watcher(args: argparse.Namespace) -> None:  # Run the independent watcher loop
    """
    Run the watcher until the target disappears and terminal files are written.

    :param args: Parsed watcher arguments.
    :return: None.
    """

    run_dir = os.path.abspath(args.run_dir)  # Resolve run directory
    os.makedirs(run_dir, exist_ok=True)  # Ensure watcher run directory exists
    memory_samples_path = os.path.join(run_dir, "memory_samples.jsonl")  # Resolve sample output path
    process_tree_path = os.path.join(run_dir, "process_tree.jsonl")  # Resolve process-tree output path
    phase_state = {}  # Initialize latest phase state
    last_phase_event_id = None  # Track emitted phase-state event IDs
    threshold_state = {"system_memory": False, "process_rss": False}  # Track active threshold states
    summary = build_initial_summary(args)  # Build mutable summary state
    missing_since = None  # Track first target disappearance time
    final_status = "target_disappeared"  # Default final status when loop exits
    print(f"{BackgroundColors.GREEN}[DEBUG] Memory watcher started. Target PID: {BackgroundColors.CYAN}{args.target_pid}{Style.RESET_ALL}")  # Log watcher start
    process, identity_status, _ = get_target_process(args.target_pid, args.target_create_time)  # Resolve target once before loop
    if process is not None:  # Prime CPU percent for non-blocking deltas
        try:  # Protect CPU priming
            process.cpu_percent(interval=None)  # Prime psutil CPU percent baseline
        except Exception:  # Ignore CPU priming failure
            pass  # Continue without CPU baseline
    while True:  # Sample until target terminal condition is confirmed
        phase_state, phase_error = load_phase_state(args.phase_state_path, phase_state)  # Read latest compact phase state
        current_event_id = phase_state.get("event_id")  # Read current phase event ID
        if current_event_id is not None and current_event_id != last_phase_event_id:  # Emit phase-state event on change
            emit_phase_event(run_dir, "phase_state", phase_state, {"phase_state_path": args.phase_state_path})  # Record observed phase transition
            last_phase_event_id = current_event_id  # Remember emitted phase event ID
        process, identity_status, actual_create_time = get_target_process(args.target_pid, args.target_create_time)  # Resolve target for this sample
        if process is None:  # Target is absent or no longer matches identity
            if missing_since is None:  # Record first disappearance only once
                missing_since = time.time()  # Store disappearance timestamp
                final_status = identity_status or "target_disappeared"  # Store final disappearance status
                emit_phase_event(run_dir, "target_disappeared", phase_state, {"status": final_status, "actual_create_time": actual_create_time})  # Record target terminal event
            if time.time() - missing_since >= max(0.0, float(args.keep_after_target_exit_seconds)):  # Wait configured grace period after disappearance
                break  # Exit watcher loop after terminal grace period
            time.sleep(max(0.1, float(args.sample_interval_seconds)))  # Sleep before re-reading terminal state
            continue  # Continue terminal grace loop
        missing_since = None  # Reset disappearance timer while target exists
        sample, child_rows, sample_error = build_sample_record(args, process, phase_state, phase_error)  # Build one sample record
        append_jsonl(memory_samples_path, sample)  # Write memory sample row
        if args.capture_process_tree:  # Persist child process rows when enabled
            for child_row in child_rows:  # Iterate actual child process rows
                append_jsonl(process_tree_path, child_row)  # Write process-tree row
        summary["total_samples"] = int(summary.get("total_samples", 0)) + 1  # Increment total sample count
        if sample.get("direct_child_process_count") is not None:  # Update direct child process peak
            previous_direct = summary.get("peak_direct_child_process_count")  # Read previous direct child peak
            summary["peak_direct_child_process_count"] = max(previous_direct or 0, int(sample.get("direct_child_process_count") or 0))  # Store direct child peak
        if sample_error or phase_error:  # Store sampling or phase errors
            summary.setdefault("sampling_errors", []).append({"timestamp": sample.get("timestamp"), "sampling_error": sample_error, "phase_state_error": phase_error})  # Append bounded error record
        update_peak(summary, sample, phase_state, run_dir)  # Update peak summary and events
        update_thresholds(args, summary, sample, phase_state, threshold_state, run_dir)  # Update threshold transitions
        time.sleep(max(0.1, float(args.sample_interval_seconds)))  # Sleep until next sample interval
    write_final_summary(run_dir, summary, final_status)  # Write terminal summary JSON


def main():
    """
    Main function.

    :param: None
    :return: None
    """

    args = parse_watcher_args()  # Parse watcher-specific command-line arguments
    start_time = datetime.datetime.now()  # Get watcher start time
    try:  # Run watcher and keep terminal summary reliable
        run_memory_watcher(args)  # Execute watcher sampling loop
    finally:  # Preserve template timing output
        finish_time = datetime.datetime.now()  # Get watcher finish time
        print(f"{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}")  # Output watcher timing
        print(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}")  # Output watcher completion message
        (atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None)  # Register sound only when explicitly enabled


if __name__ == "__main__":
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """

    main()  # Call the main function
