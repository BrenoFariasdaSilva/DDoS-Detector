"""
================================================================================
Python Function Order Validator (function_order_validator.py)
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-01-27
Description :
    This script scans Python files under a specified root directory (ROOT_DIR)
    and verifies whether functions are defined in the correct order according
    to the call hierarchy: each called function must appear above the function
    that calls it.

    Key features include:
        - AST-based parsing for precise detection of function definitions and calls
        - Recursive scanning of Python files (skips directories in IGNORE_DIRS)
        - JSON report generation listing function order violations
        - Integration with logging and terminal output
        - Cross-platform handling and sound notification on completion

Usage:
    1. Edit ROOT_DIR if necessary to point to the target directory.
    2. Execute the script:
        $ python validate_function_order.py
    3. Verify the generated JSON report for function order violations.

Outputs:
    - Scripts/function_order_report.json — structured report of order violations

TODOs:
    - Add CLI arguments for root directory and output path
    - Include function line numbers in the report
    - Add auto-correction or refactoring suggestion
    - Add logging instead of print statements

Dependencies:
    - Python >= 3.8
    - Standard library only (os, sys, ast, json, pathlib, typing, datetime, atexit, colorama)

Assumptions & Notes:
    - ROOT_DIR contains Python source files to scan
    - Files in IGNORE_DIRS are skipped
    - The report only includes functions defined after they are called
"""

import ast  # For parsing Python code into an AST
import atexit  # For playing a sound when the program finishes
import datetime  # For getting the current date and time
import json  # For saving the function order report
import os  # For running a command in the terminal
import platform  # For getting the operating system name
import sys  # For system-specific parameters and functions
from colorama import Style  # For coloring the terminal
from Logger import Logger  # For logging output to both terminal and file
from pathlib import Path  # For handling file paths
from typing import Any, Dict, List  # For type hinting


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
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)  # Project root directory
if PROJECT_ROOT not in sys.path:  # Ensure project root is in sys.path
    sys.path.insert(0, PROJECT_ROOT)  # Insert at the beginning
from Logger import Logger  # For logging output to both terminal and file

ROOT_DIR = str(Path(__file__).resolve().parent / "..")  # Directory to scan
IGNORE_DIRS = {  # Directories to ignore during the scan
    ".assets", ".git", ".github", ".idea", "__pycache__",
    "Datasets", "env", "Logs", "venv",
}
OUTPUT_FILE = os.path.join(ROOT_DIR, "Scripts", "functions_order_report.json")  # Output JSON file path

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


# Class Definitions:

class FunctionASTVisitor(ast.NodeVisitor):
    """
    AST visitor class to collect function definitions and function calls
    from a Python source file.

    Attributes:
        defined_funcs (List[str]): Names of functions defined in the file.
        called_funcs_map (Dict[str, List[str]]): Map of function names to functions they call.
        current_function (str | None): Tracks the function currently being visited.
    """

    def __init__(self):
        """
        Initializes the FunctionASTVisitor instance with empty lists and dictionaries
        to store function definitions and call relationships.
        """
        
        self.defined_funcs: List[str] = []  # List to store defined function names
        self.called_funcs_map: Dict[str, List[str]] = {}  # Map function -> functions it calls
        self.current_function: str | None = None  # Currently visited function

    def visit_FunctionDef(self, node):
        """
        Visits a function definition node in the AST.

        Records the function name, initializes its call list, and traverses child nodes.

        :param node: ast.FunctionDef node representing a function definition
        :return: None
        """
        
        func_name = node.name  # Get the function name
        self.defined_funcs.append(func_name)  # Add the function name to the defined functions list
        self.current_function = func_name  # Set the current function being visited
        self.called_funcs_map[func_name] = []  # Initialize the call list for this function
        self.generic_visit(node)  # Visit child nodes
        self.current_function = None  # Reset the current function

    def visit_Call(self, node):
        """
        Visits a function call node in the AST.

        If the call is a simple function call (not a method), adds it to the current function's call list.

        :param node: ast.Call node representing a function call
        :return: None
        """
        
        if isinstance(node.func, ast.Name) and self.current_function:  # If the call is a simple function call and we are inside a function
            self.called_funcs_map[self.current_function].append(node.func.id)  # Add the called function to the current function's call list
        self.generic_visit(node)  # Visit child nodes
        

# Functions Definitions:


def verbose_output(true_string="", false_string=""):
    """
    Outputs a message if the VERBOSE constant is set to True.

    :param true_string: The string to be outputted if the VERBOSE constant is set to True.
    :param false_string: The string to be outputted if the VERBOSE constant is set to False.
    :return: None
    """

    if VERBOSE and true_string != "":  # If the VERBOSE constant is set to True and the true_string is set
        print(true_string)  # Output the true statement string
    elif false_string != "":  # If the false_string is set
        print(false_string)  # Output the false statement string


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

    violations = detect_function_order_violations(ROOT_DIR)  # Detect function order violations
    write_order_report(violations)  # Write the function order report

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
