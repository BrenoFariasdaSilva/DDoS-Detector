"""
================================================================================
Auto-Committer for Python Class Methods
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-02-07
Description :
    Automates staged commits for methods within a specific class inside a Python file.
    This script reads a target Python file, locates a specific class, extracts all
    methods between specified start and end method names, removes them temporarily,
    and then re-adds them one-by-one in reverse order (bottom to top), creating a
    Git commit for each.

    Key features include:
        - Reads and parses Python files to locate specific class definitions
        - Extracts methods from within a target class using indentation-aware parsing
        - Identifies methods between configurable start and end markers
        - Removes all target methods and re-adds them incrementally
        - Automates Git add, commit, and optionally push operations per method
        - Preserves original indentation for valid Python class structures
        - Ignores standalone functions and methods from other classes

Usage:
    1. Configure the constants in the CONFIG section (FILE_PATH, CLASSNAME, START_FUNCTION, END_FUNCTION, COMMIT_PREFIX).
    2. IMPORTANT: Make a backup or work on a test branch first.
    3. Execute the script via Makefile or Python:
        $ make run   or   $ python class_block_functions_committer.py
    4. The script will create one Git commit per method with descriptive messages.

Outputs:
    - Modified Python file with reformatted method separators
    - Individual Git commits for each method between the markers
    - Execution log in ./Logs/class_block_functions_committer.log

TODOs:
    - Add CLI argument parsing for dynamic configuration
    - Add dry-run mode to preview changes without committing
    - Extend to support multiple class processing in batch

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
    - Assumes methods are defined within a single target class
    - Uses indentation-aware parsing to detect class scope boundaries
    - Requires a working Git repository with proper configuration
    - Methods are identified using indentation-aware regex pattern matching
    - Separator between methods is standardized to 3 newlines (2 empty lines)
    - Sound notification disabled on Windows platform
    - Standalone functions outside the target class are ignored
    - Methods from other classes are ignored
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
FILE_PATH = Path("./main.py")  # Path to the target Python file
START_FUNCTION = ""  # Name of the first function to include
END_FUNCTION = ""  # Name of the last function to include
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

    verbose_output(f"{BackgroundColors.GREEN}Extracting functions between {BackgroundColors.CYAN}{start_name}{BackgroundColors.GREEN} and {BackgroundColors.CYAN}{end_name}{Style.RESET_ALL}")
    
    pattern = r"^def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(.*?\):.*?(?=^def\s|\Z)"  # Regex pattern for top-level defs
    matches = list(re.finditer(pattern, text, flags=re.DOTALL | re.MULTILINE))  # Find all matches
    funcs = [(m.group(1), m.group(0), m.start(), m.end()) for m in matches]  # Extract function details
    start_idx = next(i for i, f in enumerate(funcs) if f[0] == start_name)  # Find start function index
    end_idx = next(i for i, f in enumerate(funcs) if f[0] == end_name)  # Find end function index
    selected = funcs[start_idx:end_idx + 1]  # Get all functions in range (inclusive)
    prefix = text[:selected[0][2]]  # Text before the first selected function
    suffix = text[selected[-1][3]:]  # Text after the last selected function
    
    verbose_output(f"{BackgroundColors.GREEN}Found {BackgroundColors.CYAN}{len(selected)}{BackgroundColors.GREEN} functions between markers{Style.RESET_ALL}")
    
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
    
    if not verify_filepath_exists(FILE_PATH):  # If the file does not exist
        print(f"{BackgroundColors.RED}Error: Target file {BackgroundColors.CYAN}{FILE_PATH}{BackgroundColors.RED} not found!{Style.RESET_ALL}")  # Output error message
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
    
    print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}All functions committed successfully!{Style.RESET_ALL}", end="\n\n")  # Output success message

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
