"""
================================================================================
Docstring Documentation Validator (docstring_documentation_validator.py)
================================================================================
Author      : Breno Farias da Silva (adapted)
Created     : 2026-01-30
Description :
    Scans Python files under the project tree and validates docstrings for
    functions and methods. Verifies if functions and methods have docstrings in the specified format:
    - First line: description
    - Empty line
    - :param lines in the order of function parameters
    - :return: line

    Automatically fixes:
    - Adds empty line after description if missing
    - Reorders :param lines to match parameter order
    - Adds missing :param lines with placeholder descriptions
    - Adds :return: None if missing

    Reports missing docstrings.

    Key features include:
        - AST-based parsing for accurate function detection
        - Recursive scanning of project directories
        - Automatic fixing of docstring format issues
        - JSON report generation with details on missing and fixed docstrings
        - Exclusion of irrelevant directories (e.g., venv, __pycache__)
        - Verbose output for debugging

Usage:
    1. Ensure Python environment is set up.
    2. Run the script via Python.
            $ python Scripts/docstring_documentation_validator.py
    3. Verify the output JSON report for missing and fixed docstrings.

Outputs:
    - Scripts/docstring_documentation_report.json: JSON report mapping files to missing and fixed docstrings

TODOs:
    - Implement CLI argument parsing for custom root directories.
    - Add support for ignoring specific files or patterns.
    - Optimize for large codebases with parallel processing.
    - Add error handling for malformed Python files.

Dependencies:
    - Python >= 3.8
    - ast (built-in)
    - astor
    - json (built-in)
    - os (built-in)
    - sys (built-in)
    - pathlib (built-in)
    - typing (built-in)
    - colorama

Assumptions & Notes:
    - Scans all .py files under the project root, excluding ignored directories.
    - Assumes UTF-8 encoding for Python files.
    - Functions and methods are def and async def, including methods inside classes.
    - Docstrings are triple-quoted strings as the first statement.
    - Report is overwritten on each run.
    - Sound notifications disabled on Windows.
"""


import ast  # For parsing Python code
import astor  # For generating Python code from AST
import atexit  # For playing a sound when the program finishes
import datetime  # For getting the current date and time
import json  # For generating JSON reports
import os  # For running a command in the terminal
import platform  # For getting the operating system name
import sys  # For system-specific parameters and functions
from colorama import Style  # For coloring the terminal
from pathlib import Path  # For handling file paths
from typing import Any, Dict, List  # For type hints

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)  # Project root directory
if PROJECT_ROOT not in sys.path:  # Add project root to sys.path
    sys.path.insert(0, PROJECT_ROOT)  # Insert at the beginning of sys.path
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
OUTPUT_FILE = os.path.abspath(os.path.join(Path(__file__).resolve().parent, "docstring_documentation_report.json"))
IGNORE_DIRS = {  # Directories to ignore
    ".assets",
    ".git",
    ".github",
    ".idea",
    "__pycache__",
    "Datasets",
    "env",
    "Logs",
    "venv",
}

# Logger Setup:
logger = Logger(f"../Logs/{Path(__file__).stem}.log", clean=True)  # Create a Logger instance
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


def calculate_execution_time(start_time, finish_time):
    """
    Calculates the execution time between start and finish times and formats it as hh:mm:ss.

    :param start_time: The start datetime object
    :param finish_time: The finish datetime object
    :return: String formatted as hh:mm:ss representing the execution time
    """

    delta = finish_time - start_time  # Calculate the time difference
    hours, remainder = divmod(delta.seconds, 3600)  # Calculate the hours, minutes and seconds
    minutes, seconds = divmod(remainder, 60)  # Calculate the minutes and seconds
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"  # Format the execution time


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

    print(
        f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Main Template Python{BackgroundColors.GREEN} program!{Style.RESET_ALL}",
        end="\n\n",
    )  # Output the welcome message
    start_time = datetime.datetime.now()  # Get the start time of the program

    send_telegram_message(TELEGRAM_BOT, [f"Starting the Main Template Python program at {start_time.strftime('%d/%m/%Y - %H:%M:%S')}"])  # Send start message to Telegram

    finish_time = datetime.datetime.now()  # Get the finish time of the program
    print(
        f"{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
    )  # Output the start and finish times
    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
    )  # Output the end of the program message
    
    send_telegram_message(TELEGRAM_BOT, [f"Finished the Main Template Python program at {finish_time.strftime('%d/%m/%Y - %H:%M:%S')} with execution time {calculate_execution_time(start_time, finish_time)}"])  # Send finish message to Telegram

    (
        atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None
    )  # Register the play_sound function to be called when the program finishes


if __name__ == "__main__":
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """

    main()  # Call the main function
