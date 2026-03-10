"""
================================================================================
Function Size Reporter - function_size_reporter.py
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-02-07
Description :
    Parses a Python source file using the AST module to detect all function
    definitions, compute their sizes, and generate a structured JSON report.

    Key features include:
        - Reads and parses Python files using the built-in ast module
        - Detects all class blocks and their methods
        - Detects all top-level functions
        - Detects all nested functions (functions defined inside another function)
        - Computes start line, end line, and size for each function block
        - Generates a structured JSON report sorted by function size (biggest first)

Usage:
    1. Configure FILE_PATH in the Execution Constants section to target a Python file.
    2. Execute the script:
        $ python function_size_reporter.py
    3. The script generates function_size_report.json in the current directory.

Outputs:
    - function_size_report.json containing the full function size analysis
    - Execution log in ./Logs/function_size_reporter.log

TODOs:
    - Add CLI argument parsing for dynamic configuration
    - Add support for batch processing of multiple files
    - Add threshold filtering to report only functions above a given size

Dependencies:
    - Python >= 3.8
    - ast (built-in)
    - json (built-in)
    - atexit (built-in)
    - datetime (built-in)
    - os (built-in)
    - platform (built-in)
    - sys (built-in)
    - colorama
    - pathlib (built-in)

Assumptions & Notes:
    - Requires Python >= 3.8 for ast.end_lineno attribute support
    - Function size is defined as end_lineno - lineno + 1
    - Sound notification disabled on Windows platform
"""


import ast  # For parsing Python source files into an abstract syntax tree
import atexit  # For playing a sound when the program finishes
import datetime  # For getting the current date and time
import json  # For serializing the structured report to a JSON file
import os  # For interacting with the filesystem
import platform  # For getting the operating system name
import sys  # For system-specific parameters and functions
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
FILE_PATH = Path("./main.py")  # Path to the target Python file to analyze
OUTPUT_FILE = Path("./function_size_report.json")  # Path to the output JSON report file

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

    if VERBOSE and true_string != "":  # Verify if verbose logging is enabled and true_string is not empty
        print(true_string)  # Log the true_string if verbose logging is enabled
    elif false_string != "":  # Verify if false_string is not empty
        print(false_string)  # Log the false_string if verbose logging is not enabled


def verify_filepath_exists(filepath):
    """
    Verify if a file or folder exists at the specified path.

    :param filepath: Path to the file or folder.
    :return: True if the file or folder exists, False otherwise.
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Verifying if the file or folder exists at the path: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}"
    )  # Log the filepath verification message

    return os.path.exists(filepath)  # Return True if the file or folder exists, False otherwise


def save_report(report: dict, output_path: Path) -> None:
    """
    Serializes and saves the function size report to a JSON file on disk.

    :param report: The complete function size report dictionary to serialize.
    :param output_path: Path object pointing to the output JSON file location.
    :return: None
    """

    verbose_output(f"{BackgroundColors.GREEN}Saving report to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}")  # Log the report save operation

    with open(output_path, "w", encoding="utf-8") as report_file:  # Open the output file for writing with UTF-8 encoding
        json.dump(report, report_file, indent=2, ensure_ascii=False)  # Serialize the report dictionary with indentation

    print(f"{BackgroundColors.GREEN}Report saved to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}")  # Log the successful save confirmation


def to_seconds(obj):
    """
    Converts various time-like objects to seconds.

    :param obj: The object to convert (can be int, float, timedelta, datetime, etc.).
    :return: The equivalent time in seconds as a float, or None if conversion fails.
    """

    if obj is None:  # Verify if the object is None before attempting conversion
        return None  # Return None to signal failure to convert
    if isinstance(obj, (int, float)):  # Verify if the object is already numeric
        return float(obj)  # Return as float seconds directly
    if hasattr(obj, "total_seconds"):  # Verify if the object is a timedelta-like type
        try:  # Attempt to call total_seconds on the object
            return float(obj.total_seconds())  # Return total seconds from the timedelta
        except Exception:  # Handle conversion error and fall through
            pass  # Fall through on conversion error
    if hasattr(obj, "timestamp"):  # Verify if the object is a datetime-like type
        try:  # Attempt to call timestamp on the object
            return float(obj.timestamp())  # Return seconds since epoch from the datetime
        except Exception:  # Handle conversion error and fall through
            pass  # Fall through on conversion error
    return None  # Return None if no conversion was successful


def calculate_execution_time(start_time, finish_time=None):
    """
    Calculates the execution time between two time values and returns a human-readable string.

    :param start_time: The start time value (datetime, timedelta, int, or float).
    :param finish_time: The finish time value (datetime, timedelta, int, or float), or None for single-argument mode.
    :return: A human-readable string representation of the elapsed time such as "1h 2m 3s".
    """

    if finish_time is None:  # Verify if operating in single-argument mode
        total_seconds = to_seconds(start_time)  # Attempt to convert the single value to seconds
        if total_seconds is None:  # Verify if conversion returned None
            try:  # Attempt numeric coercion as fallback
                total_seconds = float(start_time)  # Coerce the value to float seconds
            except Exception:  # Handle coercion failure
                total_seconds = 0.0  # Default to zero on coercion failure
    else:  # Operate in two-argument mode to compute the difference
        st = to_seconds(start_time)  # Convert start time to seconds
        ft = to_seconds(finish_time)  # Convert finish time to seconds
        if st is not None and ft is not None:  # Verify both conversions succeeded
            total_seconds = ft - st  # Compute elapsed seconds by direct subtraction
        else:  # Fall back to alternative subtraction methods
            try:  # Attempt datetime/timedelta subtraction
                delta = finish_time - start_time  # Subtract the two time values
                total_seconds = float(delta.total_seconds())  # Extract seconds from the resulting timedelta
            except Exception:  # Handle subtraction failure
                try:  # Attempt final numeric coercion
                    total_seconds = float(finish_time) - float(start_time)  # Coerce both values and subtract
                except Exception:  # Handle numeric coercion failure
                    total_seconds = 0.0  # Default to zero on final failure

    if total_seconds is None:  # Verify total_seconds is not None before proceeding
        total_seconds = 0.0  # Default to zero if conversion produced None
    if total_seconds < 0:  # Verify total_seconds is non-negative
        total_seconds = abs(total_seconds)  # Use the absolute value for negative durations

    days = int(total_seconds // 86400)  # Compute full days from total seconds
    hours = int((total_seconds % 86400) // 3600)  # Compute remaining hours after days
    minutes = int((total_seconds % 3600) // 60)  # Compute remaining minutes after hours
    seconds = int(total_seconds % 60)  # Compute remaining seconds after minutes

    if days > 0:  # Verify if the duration includes full days
        return f"{days}d {hours}h {minutes}m {seconds}s"  # Return formatted days+hours+minutes+seconds string
    if hours > 0:  # Verify if the duration includes full hours
        return f"{hours}h {minutes}m {seconds}s"  # Return formatted hours+minutes+seconds string
    if minutes > 0:  # Verify if the duration includes full minutes
        return f"{minutes}m {seconds}s"  # Return formatted minutes+seconds string
    return f"{seconds}s"  # Return formatted seconds-only string


def play_sound():
    """
    Plays a sound when the program finishes and skips if the operating system is Windows.

    :param: None
    :return: None
    """

    current_os = platform.system()  # Retrieve the current operating system name
    if current_os == "Windows":  # Verify if the OS is Windows
        return  # Skip sound playback on Windows

    if verify_filepath_exists(SOUND_FILE):  # Verify if the sound file exists
        if current_os in SOUND_COMMANDS:  # Verify if the OS has a mapped sound command
            os.system(f"{SOUND_COMMANDS[current_os]} {SOUND_FILE}")  # Execute the OS-specific sound command
        else:  # Handle unsupported operating system for sound playback
            print(
                f"{BackgroundColors.RED}The {BackgroundColors.CYAN}{current_os}{BackgroundColors.RED} is not in the {BackgroundColors.CYAN}SOUND_COMMANDS dictionary{BackgroundColors.RED}. Please add it!{Style.RESET_ALL}"
            )  # Log the unsupported OS error message
    else:  # Handle missing sound file
        print(
            f"{BackgroundColors.RED}Sound file {BackgroundColors.CYAN}{SOUND_FILE}{BackgroundColors.RED} not found. Make sure the file exists.{Style.RESET_ALL}"
        )  # Log the missing sound file error message


def main():
    """
    Main function.

    :param: None
    :return: None
    """

    print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Function Size Reporter{BackgroundColors.GREEN} program!{Style.RESET_ALL}", end="\n\n")  # Output the welcome message

    start_time = datetime.datetime.now()  # Get the start time of the program

    if not verify_filepath_exists(FILE_PATH):  # Verify if the target file exists
        print(f"{BackgroundColors.RED}Error: Target file {BackgroundColors.CYAN}{FILE_PATH}{BackgroundColors.RED} not found!{Style.RESET_ALL}")  # Log the missing file error
        return  # Exit the main function early

    print(f"{BackgroundColors.GREEN}Analyzing file: {BackgroundColors.CYAN}{FILE_PATH}{Style.RESET_ALL}")  # Log the analysis start message

    source_text = read_source_file(FILE_PATH)  # Read the source file content
    tree = parse_ast_tree(source_text, FILE_PATH)  # Parse the source text into an AST tree
    report = build_report(tree)  # Build the complete function size report

    save_report(report, OUTPUT_FILE)  # Save the report to the output JSON file

    total = report["total_functions"]  # Retrieve the total function count from the report
    class_count = sum(len(v) for v in report["classes"].values())  # Compute the total class method count
    top_count = len(report["top-level functions"])  # Retrieve the top-level function count
    nested_count = len(report["nested functions"])  # Retrieve the nested function count

    print(f"{BackgroundColors.GREEN}Total functions detected: {BackgroundColors.CYAN}{total}{Style.RESET_ALL}")  # Log the total function count
    print(f"{BackgroundColors.GREEN}Class methods: {BackgroundColors.CYAN}{class_count}{Style.RESET_ALL}")  # Log the class method count
    print(f"{BackgroundColors.GREEN}Top-level functions: {BackgroundColors.CYAN}{top_count}{Style.RESET_ALL}")  # Log the top-level function count
    print(f"{BackgroundColors.GREEN}Nested functions: {BackgroundColors.CYAN}{nested_count}{Style.RESET_ALL}")  # Log the nested function count

    finish_time = datetime.datetime.now()  # Get the finish time of the program
    print(
        f"{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
    )  # Log the execution timing information
    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
    )  # Log the program completion message
    (
        atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None
    )  # Register the play_sound function to be called when the program finishes


if __name__ == "__main__":
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """

    main()  # Call the main function
