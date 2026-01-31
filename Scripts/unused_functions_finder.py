"""
================================================================================
Python Unused Functions Detector (unused_functions_finder.py
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-01-27
Description :
    This script scans Python files under a specified root directory (ROOT_DIR)
    and detects functions that are defined but never used within the project.

    Key features include:
        - AST-based parsing for precise detection of function definitions and calls
        - Recursive scanning of Python files (skips directories in IGNORE_DIRS)
        - JSON report generation listing unused functions
        - Integration with logging and terminal output
        - Cross-platform handling and sound notification on completion

Usage:
    1. Edit ROOT_DIR if necessary to point to the target directory.
    2. Execute the script:
        $ python detect_unused_functions.py
    3. Verify the generated JSON report for unused functions.

Outputs:
    - Scripts/unused_functions.json — structured report of unused functions

TODOs:
    - Add CLI arguments for root directory and output path
    - Improve detection across multiple modules and imported functions
    - Add logging instead of print statements
    - Include function docstrings and line numbers in the report

Dependencies:
    - Python >= 3.8
    - Standard library only (os, sys, ast, json, pathlib, typing, datetime, atexit, colorama)

Assumptions & Notes:
    - ROOT_DIR contains Python source files to scan
    - Files in IGNORE_DIRS are skipped
    - The JSON report only includes functions defined and never called
"""


import ast  # For parsing Python code into an AST
import atexit  # For playing a sound when the program finishes
import datetime  # For getting the current date and time
import json  # For saving the unused functions report
import os  # For running a command in the terminal
import platform  # For getting the operating system name
import sys  # For system-specific parameters and functions
from colorama import Style  # For coloring the terminal
from pathlib import Path  # For handling file paths
from typing import Dict, List  # For type hinting

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
ROOT_DIR = str(Path(__file__).resolve().parent / "..")  # Directory to scan
IGNORE_DIRS = {  # Directories to ignore during the scan
    ".assets", ".git", ".github", ".idea", "__pycache__",
    "Datasets", "env", "Logs", "venv",
}
OUTPUT_FILE = os.path.join(ROOT_DIR, "Scripts", "unused_functions_report.json")  # Output JSON file path

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

    This class traverses the abstract syntax tree (AST) of a Python file,
    recording:
        - All function definitions (names of functions defined in the file)
        - All function calls (names of functions invoked in the file)

    Attributes:
        defined_funcs (List[str]): Names of functions defined in the file.
        called_funcs (List[str]): Names of functions called in the file.
    """

    def __init__(self):
        """
        Initializes the FunctionASTVisitor instance with empty lists for storing
        function definitions and calls.
        """
        
        self.defined_funcs: List[str] = []  # List to store defined function names
        self.called_funcs: List[str] = []  # List to store called function names

    def visit_FunctionDef(self, node):
        """
        Visits each function definition node in the AST.

        Adds the function name to the 'defined_funcs' list and
        continues traversing child nodes.

        :param node: ast.FunctionDef node representing a function definition
        :return: None
        """
        
        if node.name != "__init__"  :  # Exclude __init__ methods from unused check
            self.defined_funcs.append(node.name)  # Record the defined function name
        self.generic_visit(node)  # Continue traversing child nodes

    def visit_ClassDef(self, node):
        """
        Visits each class definition node in the AST.

        Checks if the class has an __init__ method and records the class name
        in classes_with_init set.

        :param node: ast.ClassDef node representing a class definition
        :return: None
        """
        
        self.generic_visit(node)  # Continue traversing child nodes

    def visit_Call(self, node):
        """
        Visits each function call node in the AST.

        If the call is a simple function call (not a method or attribute),
        adds the function name to the 'called_funcs' list. Also handles
        special cases like atexit.register where a function is passed as
        an argument. Then continues traversing child nodes.

        :param node: ast.Call node representing a function call
        :return: None
        """
        
        if isinstance(node.func, ast.Name):  # Only simple function calls
            self.called_funcs.append(node.func.id)  # Record the called function name
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'atexit' and node.func.attr == "register":  # Special case for atexit.register
            if node.args and isinstance(node.args[0], ast.Name):  # Ensure there is at least one argument and it's a simple name
                self.called_funcs.append(node.args[0].id)  # Record the function passed to atexit.register
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == 'executor' and node.func.attr == "submit":  # Special case for executor.submit
            if node.args and isinstance(node.args[0], ast.Name):  # Ensure there is at least one argument and it's a simple name
                self.called_funcs.append(node.args[0].id)  # Record the function passed to executor.submit
        self.generic_visit(node)  # Continue traversing child nodes


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


def is_ignored(path: str) -> bool:
    """
    Determines whether a given path should be ignored based on the IGNORE_DIRS set.

    :param path: Path to a file or directory
    :return: True if any part of the path matches a directory in IGNORE_DIRS, False otherwise
    """
    
    verbose_output(
        f"{BackgroundColors.GREEN}Verifying if the path should be ignored: {BackgroundColors.CYAN}{path}{Style.RESET_ALL}"
    )  # Output the verbose message
    
    parts = set(Path(path).parts)  # Split the path into its components
    return bool(parts.intersection(IGNORE_DIRS))  # Return True if any part is in IGNORE_DIRS


def collect_python_files(root_dir: str) -> List[str]:
    """
    Recursively collects all Python (.py) files under a root directory, skipping ignored directories.

    :param root_dir: Root directory to scan
    :return: List of absolute paths to Python files found under root_dir
    """
    
    verbose_output(
        f"{BackgroundColors.GREEN}Collecting Python files under root directory: {BackgroundColors.CYAN}{root_dir}{Style.RESET_ALL}"
    )  # Output the verbose message
    
    py_files = []  # Initialize list to store Python files

    for dirpath, dirnames, filenames in os.walk(root_dir):  # Walk through all directories and files under root_dir
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]  # Skip ignored directories

        for filename in filenames:  # Iterate over files
            if filename.endswith(".py"):  # Only consider Python files
                full_path = os.path.join(dirpath, filename)  # Get absolute path
                if os.path.isfile(full_path) and not is_ignored(full_path):  # Skip ignored paths
                    py_files.append(full_path)  # Add to list

    return py_files  # Return all Python files found


def extract_functions_and_calls(file_path: str) -> Dict[str, List[str]]:
    """
    Parses a Python file to extract all function definitions and function calls.

    Utilizes the FunctionASTVisitor class to traverse the AST of the file.

    :param file_path: Path to the Python file to parse
    :return: Dictionary with keys:
        - 'defined': list of function names defined in the file
        - 'called': list of function names called in the file
    """
    
    verbose_output(
        f"{BackgroundColors.GREEN}Extracting functions and calls from file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}"
    )  # Output the verbose message
    
    with open(file_path, "r", encoding="utf-8") as f:  # Open file with UTF-8 encoding
        source = f.read()  # Read the entire file content

    tree = ast.parse(source, filename=file_path)  # Parse the file into an AST

    visitor = FunctionASTVisitor()  # Instantiate the AST visitor
    visitor.visit(tree)  # Traverse the AST

    return {"defined": visitor.defined_funcs, "called": visitor.called_funcs}  # Return results


def detect_unused_functions(root_dir: str) -> Dict[str, List[str]]:
    """
    Detects functions that are defined but never called in a Python project.

    :param root_dir: Root directory of the project to scan
    :return: Dictionary mapping relative file paths to lists of unused function names
    """
    
    verbose_output(
        f"{BackgroundColors.GREEN}Detecting unused functions under root directory: {BackgroundColors.CYAN}{root_dir}{Style.RESET_ALL}"
    )  # Output the verbose message
    
    unused_map = {}  # Map to store unused functions per file
    all_defined = {}  # Map of relative file paths to defined function names
    all_called = set()  # Set of all called function names in the project

    py_files = collect_python_files(root_dir)  # Collect all Python files
    for f in py_files:  # Process each Python file
        funcs = extract_functions_and_calls(f)  # Extract defined and called functions
        rel_path = os.path.relpath(f, root_dir).replace("\\", "/")  # Compute relative path
        all_defined[rel_path] = funcs["defined"]  # Store defined functions
        all_called.update(funcs["called"])  # Add called functions to global set

    for rel_path, funcs in all_defined.items():  # Verify each file's defined functions
        unused = [fn for fn in funcs if fn not in all_called]  # Functions never called
        if unused:  # Only include files with unused functions
            unused_map[rel_path] = unused

    return unused_map  # Return the unused functions map


def write_unused_functions_report(unused_map: Dict[str, List[str]]):
    """
    Writes a JSON report of unused functions if any are found.

    :param unused_map: Dictionary mapping relative file paths to unused function names
    :return: None
    """
    
    verbose_output(
        f"{BackgroundColors.GREEN}Writing unused functions report to: {BackgroundColors.CYAN}{OUTPUT_FILE}{Style.RESET_ALL}"
    )  # Output the verbose message
    
    if not unused_map:  # Verify if any unused functions were detected
        print("No unused functions found.")
        return  # Exit if nothing to report

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:  # Open output JSON file
        json.dump(unused_map, f, indent=4)  # Write the map with indentation

    print(f"Unused functions report generated: {OUTPUT_FILE}")  # Notify user


def calculate_execution_time(start_time, finish_time):
    """
    Calculates the execution time between start and finish times and formats it dynamically.

    :param start_time: The start datetime object
    :param finish_time: The finish datetime object
    :return: String formatted with appropriate units (s, m, h, d) representing the execution time
    """

    delta = finish_time - start_time  # Calculate the time difference
    total_seconds = delta.total_seconds()  # Get total seconds

    days = int(total_seconds // 86400)  # Calculate days
    hours = int((total_seconds % 86400) // 3600)  # Calculate hours
    minutes = int((total_seconds % 3600) // 60)  # Calculate minutes
    seconds = int(total_seconds % 60)  # Calculate seconds

    if days > 0:  # If there are days
        return f"{days}d {hours}h {minutes}m {seconds}s"  # Format with days, hours, minutes, seconds
    elif hours > 0:  # If there are hours
        return f"{hours}h {minutes}m {seconds}s"  # Format with hours, minutes, seconds
    elif minutes > 0:  # If there are minutes
        return f"{minutes}m {seconds}s"  # Format with minutes and seconds
    else:  # Only seconds
        return f"{seconds}s"  # Format with seconds


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
        f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Unused Functions Finder{BackgroundColors.GREEN} program!{Style.RESET_ALL}",
        end="\n\n",
    )  # Output the welcome message
    start_time = datetime.datetime.now()  # Get the start time of the program

    unused = detect_unused_functions(ROOT_DIR)  # Detect unused functions
    write_unused_functions_report(unused)  # Get the unused functions report

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
