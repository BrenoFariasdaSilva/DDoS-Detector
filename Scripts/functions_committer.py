"""
================================================================================
Auto-Committer for Python Functions
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-02-07
Description :
    Automates staged commits for functions between two markers inside a Python file.
    This script reads a target Python file, extracts all functions between specified
    start and end function names, removes them temporarily, and then re-adds them
    one-by-one in reverse order (bottom to top), creating a Git commit for each.

    Key features include:
        - Reads and parses Python files to extract top-level function definitions
        - Identifies functions between configurable start and end markers
        - Removes all target functions and re-adds them incrementally
        - Automates Git add, commit, and optionally push operations per function
        - Standardizes function separators for consistent code formatting

Usage:
    1. Configure the constants in the CONFIG section (FILE_PATH, START_FUNCTION, END_FUNCTION, COMMIT_PREFIX).
    2. IMPORTANT: Make a backup or work on a test branch first.
    3. Execute the script via Makefile or Python:
        $ make run   or   $ python functions_committer.py
    4. The script will create one Git commit per function with descriptive messages.

Outputs:
    - Modified Python file with reformatted function separators
    - Individual Git commits for each function between the markers
    - Execution log in ./Logs/functions_committer.log

TODOs:
    - Add CLI argument parsing for dynamic configuration
    - Implement support for nested function definitions
    - Add dry-run mode to preview changes without committing
    - Extend to support multiple file processing in batch

Dependencies:
    - Python >= 3.7
    - re (built-in)
    - subprocess (built-in)
    - pathlib (built-in)
    - colorama
    - datetime (built-in)
    - os (built-in)
    - platform (built-in)
    - sys (built-in)

Assumptions & Notes:
    - Assumes functions are top-level definitions (no nested defs)
    - Requires a working Git repository with proper configuration
    - Functions are identified using regex pattern matching
    - Separator between functions is standardized to 3 newlines (2 empty lines)
    - Sound notification disabled on Windows platform
"""

import atexit  # For playing a sound when the program finishes
import datetime  # For getting the current date and time
import os  # For running a command in the terminal
import platform  # For getting the operating system name
import re  # For regular expression pattern matching
import subprocess  # For running Git commands
import sys  # For system-specific parameters and functions
import time  # For sleeping between commits
from colorama import Style  # For coloring the terminal
from pathlib import Path  # For handling file paths

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)  # Project root directory
if PROJECT_ROOT not in sys.path:  # Ensure project root is in sys.path
    sys.path.insert(0, PROJECT_ROOT)  # Insert at the beginning
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
FILE_PATH = Path("./class_block_functions_committer.py")  # Path to the target Python file
START_FUNCTION = "validate_markers"  # Name of the first function to include
END_FUNCTION = "extract_methods_between"  # Name of the last function to include
COMMIT_PREFIX = "FEAT: Adding the"  # Prefix for Git commit messages
FUNCTION_SEPARATOR = "\n\n\n"  # Standardized separator between top-level functions (3 newlines -> 2 empty lines)

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
    "Play Sound": True,  # Set to True to play a sound when the program finishes
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


def validate_markers(start_name, end_name, text):
    """
    Validate that START_FUNCTION and END_FUNCTION are provided and exist in the file.

    :param start_name: The name of the first function to include
    :param end_name: The name of the last function to include
    :param text: The full text content of the Python file
    :return: True if validation passes, False otherwise
    """
    
    verbose_output(f"{BackgroundColors.GREEN}Validating START_FUNCTION and END_FUNCTION markers...{Style.RESET_ALL}")

    problems = []  # List to collect validation problems
    if not start_name or not end_name:  # If either marker is not set
        problems.append("Both START_FUNCTION and END_FUNCTION must be set to function names.")  # Add problem if markers are not set

    pattern = r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\("  # Regex pattern to find top-level function names
    found_names = [m.group(1) for m in re.finditer(pattern, text, flags=re.MULTILINE)]  # Extract all top-level function names from the file

    if start_name and start_name not in found_names:  # If START_FUNCTION is set but not found in the file
        problems.append(f"START_FUNCTION '{start_name}' not found in {FILE_PATH.name}.")  # Add problem if START_FUNCTION is not found
    if end_name and end_name not in found_names:  # If END_FUNCTION is set but not found in the file
        problems.append(f"END_FUNCTION '{end_name}' not found in {FILE_PATH.name}.")  # Add problem if END_FUNCTION is not found

    if problems:  # If there are any validation problems, print them and return False
        print(f"{BackgroundColors.RED}Validation error with START/END function markers:{Style.RESET_ALL}")
        for p in problems:  # Print each problem in the validation
            print(f"{BackgroundColors.YELLOW}- {p}{Style.RESET_ALL}")

        if found_names:  # If there are any top-level functions found, list them to help the user
            print(f"{BackgroundColors.GREEN}Available top-level functions in the file:{BackgroundColors.CYAN} {', '.join(found_names)}{Style.RESET_ALL}")
        else:  # If no top-level functions were found, inform the user
            print(f"{BackgroundColors.YELLOW}No top-level functions were detected in {FILE_PATH.name}.{Style.RESET_ALL}")

        print(f"{BackgroundColors.GREEN}Please set `START_FUNCTION` and `END_FUNCTION` to valid function names before running the script.{Style.RESET_ALL}")
        return False  # Validation failed

    return True  # Validation passed


def run_git_commit(function_name: str):
    """
    Executes Git add and commit commands for the target file.

    :param function_name: The name of the function being committed
    :return: None
    """

    commit_msg = f"{COMMIT_PREFIX} {function_name} function to {FILE_PATH.name}"  # Create the commit message

    verbose_output(f"{BackgroundColors.GREEN}Running Git add for: {BackgroundColors.CYAN}{FILE_PATH}{Style.RESET_ALL}")

    absolute_file_path = FILE_PATH.resolve()  # Resolve FILE_PATH to an absolute path
    git_dir = absolute_file_path.parent  # Get the directory containing the file

    subprocess.run(["git", "-C", str(git_dir), "add", str(absolute_file_path)], check=True)  # Stage the file

    verbose_output(f"{BackgroundColors.GREEN}Running Git commit with message: {BackgroundColors.CYAN}{commit_msg}{Style.RESET_ALL}")

    subprocess.run(["git", "-C", str(git_dir), "commit", "-m", commit_msg], check=True)  # Commit the changes


def extract_functions_between(text, start_name, end_name):
    """
    Extracts all top-level functions between two specified function names.

    :param text: The full text content of the Python file
    :param start_name: The name of the first function to include
    :param end_name: The name of the last function to include
    :return: Tuple of (prefix_text, suffix_text, list_of_functions)
             Each function in the list is a tuple: (name, code, start_pos, end_pos)
    """

    verbose_output(f"{BackgroundColors.GREEN}Extracting functions between {BackgroundColors.CYAN}{start_name}{BackgroundColors.GREEN} and {BackgroundColors.CYAN}{end_name}{Style.RESET_ALL}")  # Log extraction start
    
    pattern = r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(.*?\)\s*(?:->.*?)?\s*:.*?(?=^def\s|\Z)"  # Regex pattern for top-level defs with optional return type annotations.
    name_matches = list(re.finditer(pattern, text, flags=re.MULTILINE))  # Find all name occurrences
    funcs = []  # Container for (name, code, start, end)
    for i, m in enumerate(name_matches):
        func_name = m.group(1)  # Extract function name from match group
        start_pos = m.start()  # Start index of this function definition in the text
        end_pos = name_matches[i + 1].start() if i + 1 < len(name_matches) else len(text)  # End index (next def start or EOF)
        func_code = text[start_pos:end_pos]  # Slice function code block from start to end
        funcs.append((func_name, func_code, start_pos, end_pos))  # Append tuple for downstream processing

    # Locate start and end indices; raise informative error if missing
    try:
        start_idx = next(i for i, f in enumerate(funcs) if f[0] == start_name)  # Find index of START function
    except StopIteration:
        available = [f[0] for f in funcs]  # Collect available function names for error message
        raise ValueError(f"START function '{start_name}' not found. Available: {available}")  # Inform caller
    try:
        end_idx = next(i for i, f in enumerate(funcs) if f[0] == end_name)  # Find index of END function
    except StopIteration:
        available = [f[0] for f in funcs]  # Collect available function names for error message
        raise ValueError(f"END function '{end_name}' not found. Available: {available}")  # Inform caller

    selected = funcs[start_idx:end_idx + 1]  # Select functions inclusively between markers
    prefix = text[: selected[0][2]] if selected else text  # Text before the first selected function
    suffix = text[selected[-1][3] :] if selected else ""  # Text after the last selected function
    
    verbose_output(f"{BackgroundColors.GREEN}Found {BackgroundColors.CYAN}{len(selected)}{BackgroundColors.GREEN} functions between markers{Style.RESET_ALL}")  # Log count
    
    return prefix, suffix, selected  # Return the components


def write_file(path, content):
    """
    Writes content to a file with UTF-8 encoding and ensures it ends with exactly one empty line.

    :param path: Path object pointing to the target file
    :param content: String content to write to the file
    :return: None
    """

    verbose_output(f"{BackgroundColors.GREEN}Writing content to file: {BackgroundColors.CYAN}{path}{Style.RESET_ALL}")
    
    content = content.rstrip() + "\n"  # Remove trailing whitespace and add exactly one newline
    
    path.write_text(content, encoding="utf-8")  # Write the content to the file


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


def main():
    """
    Main function.

    :param: None
    :return: None
    """

    print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Functions Committer{BackgroundColors.GREEN} program!{Style.RESET_ALL}", end="\n\n")  # Output the welcome message
    
    start_time = datetime.datetime.now()  # Get the start time of the program
    
    file_path_str = str(FILE_PATH)  # Convert FILE_PATH to string for logging
    
    if not verify_filepath_exists(file_path_str):  # If the target file does not exist, output an error message and exit
        print(f"{BackgroundColors.RED}Error: Target file {BackgroundColors.CYAN}{file_path_str}{BackgroundColors.RED} not found!{Style.RESET_ALL}")  # Output error message
        return  # Exit the function
    
    original_text = FILE_PATH.read_text(encoding="utf-8")  # Read the original file content

    if not validate_markers(START_FUNCTION, END_FUNCTION, original_text):  # If the markers are not valid, exit the function
        return  # Exit the function if validation fails

    print(f"{BackgroundColors.GREEN}Extracting functions between {BackgroundColors.CYAN}{START_FUNCTION}{BackgroundColors.GREEN} and {BackgroundColors.CYAN}{END_FUNCTION}{Style.RESET_ALL}")  # Output extraction message
    
    prefix, suffix, functions = extract_functions_between(original_text, START_FUNCTION, END_FUNCTION)  # Extract the functions between markers
    
    verbose_output(f"{BackgroundColors.YELLOW}Removing {BackgroundColors.CYAN}{len(functions)}{BackgroundColors.YELLOW} functions from the file...{Style.RESET_ALL}")  # Output removal message
    
    write_file(FILE_PATH, prefix + suffix)  # Write the file without the target functions
    
    verbose_output(f"{BackgroundColors.GREEN}Functions removed. Starting staged commits...{Style.RESET_ALL}")  # Output staged commits start message
    
    current_body = ""  # Initialize the current body content
    total_functions = len(functions)  # Get the total number of functions
    
    for index, (name, code, *_) in enumerate(reversed(functions), start=1):  # Iterate through functions in reverse order with index
        code = code.strip("\n")  # Remove all surrounding blank lines safely
        
        if current_body:  # If there is already content in the body
            current_body = code + FUNCTION_SEPARATOR + current_body.lstrip("\n")  # Add function with separator
        else:  # If this is the first function being added
            current_body = code + FUNCTION_SEPARATOR  # Add function with separator at the end
            
        new_content = prefix + current_body + suffix  # Construct the new file content
        
        write_file(FILE_PATH, new_content)  # Write the updated content to the file
        
        verbose_output(f"{BackgroundColors.BOLD}[{BackgroundColors.YELLOW}{index}{BackgroundColors.CYAN}/{BackgroundColors.YELLOW}{total_functions}{BackgroundColors.BOLD}]{Style.RESET_ALL} {BackgroundColors.GREEN}Committing function: {BackgroundColors.CYAN}{name}{Style.RESET_ALL}")  # Output commit message with progress indicator
        
        run_git_commit(name)  # Run Git add and commit
        
        time.sleep(3)  # Optional: Sleep for a short time between commits to avoid overwhelming the system (adjust as needed)

    git_dir = FILE_PATH.resolve().parent  # Get the directory containing the file
    subprocess.run(["git", "-C", str(git_dir), "push"], check=True)  # Push all commits to the remote repository
    
    print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}All {len(functions)} functions committed successfully!{Style.RESET_ALL}", end="\n\n")  # Output success message

    finish_time = datetime.datetime.now()  # Get the finish time of the program
    print(
        f"{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
    )  # Output the start and finish times
    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
    )  # Output the end of the program message
    (
        atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None
    )  # Register the play_sound function to be called when the program finishes


if __name__ == "__main__":
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """

    main()  # Call the main function
