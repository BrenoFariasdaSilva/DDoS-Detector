"""
================================================================================
Imports Placement Verifier (imports_placement.py)
================================================================================
Author      : Breno Farias da Silva (adapted)
Created     : 2026-01-30
Description :
    Scans Python files under the project tree and detects import statements
    that occur inside function or class bodies. All imports should be at the
    module top-level header; this tool reports nested imports with file and
    location details so they can be moved to the header.

    Key features include:
        - AST-based parsing for accurate import detection
        - Recursive scanning of project directories
        - JSON report generation with detailed location info
        - Exclusion of irrelevant directories (e.g., venv, __pycache__)
        - Verbose output for debugging

Usage:
    1. Ensure Python environment is set up.
    2. Run the script via Python.
            $ python Scripts/imports_placement.py
    3. Verify the output JSON report for any nested imports found.

Outputs:
    - Scripts/imports_placement_report.json: JSON report mapping files to nested imports

TODOs:
    - Implement CLI argument parsing for custom root directories.
    - Add support for ignoring specific files or patterns.
    - Optimize for large codebases with parallel processing.
    - Add error handling for malformed Python files.

Dependencies:
    - Python >= 3.8
    - ast (built-in)
    - json (built-in)
    - os (built-in)
    - sys (built-in)
    - pathlib (built-in)
    - typing (built-in)
    - colorama

Assumptions & Notes:
    - Scans all .py files under the project root, excluding ignored directories.
    - Assumes UTF-8 encoding for Python files.
    - Nested imports are those inside functions, async functions, or classes.
    - Report is overwritten on each run.
    - Sound notifications disabled on Windows.
"""


import ast  # For parsing Python code
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
OUTPUT_FILE = os.path.abspath(os.path.join(Path(__file__).resolve().parent, "imports_placement_report.json"))
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


# Classes Definitions:


class ImportPlacementVisitor(ast.NodeVisitor):
    """
    AST visitor that records imports that appear inside functions or classes.
    """

    def __init__(self):
        """
        Initialize the ImportPlacementVisitor.

        :param: None
        :return: None
        """
        
        self.container_stack: List[Dict[str, Any]] = []  # Stack to track current container (function/class)
        self.nested_imports: List[Dict[str, Any]] = []  # List to store detected nested imports

    def record_import(self, node: ast.AST, names: List[str], module: str | None = None):
        """
        Record a nested import with its details.

        :param node: The AST node of the import
        :param names: List of imported names
        :param module: The module name for from imports, None for regular imports
        :return: None
        """
        
        container = self.container_stack[-1] if self.container_stack else None  # Get the current container
        entry = {
            "lineno": getattr(node, "lineno", None),  # Line number of the import
            "container_type": container["type"] if container else None,  # Type of container (function/class)
            "container_name": container["name"] if container else None,  # Name of the container
            "module": module,  # Module name
            "names": names,  # List of imported names
        }
        self.nested_imports.append(entry)  # Add to the list of nested imports

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Visit a function definition node.

        :param node: The FunctionDef AST node
        :return: None
        """
        
        self.container_stack.append({"type": "function", "name": node.name})  # Push function to stack
        self.generic_visit(node)  # Visit child nodes
        self.container_stack.pop()  # Pop function from stack

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """
        Visit an async function definition node.

        :param node: The AsyncFunctionDef AST node
        :return: None
        """
        
        self.container_stack.append({"type": "async_function", "name": node.name})  # Push async function to stack
        self.generic_visit(node)  # Visit child nodes
        self.container_stack.pop()  # Pop async function from stack

    def visit_ClassDef(self, node: ast.ClassDef):
        """
        Visit a class definition node.

        :param node: The ClassDef AST node
        :return: None
        """
        
        self.container_stack.append({"type": "class", "name": node.name})  # Push class to stack
        self.generic_visit(node)  # Visit child nodes
        self.container_stack.pop()  # Pop class from stack

    def visit_Import(self, node: ast.Import):
        """
        Visit an import statement node.

        :param node: The Import AST node
        :return: None
        """
        
        if self.container_stack:  # If inside a container
            names = [alias.name for alias in node.names]  # Extract imported names
            self.record_import(node, names, module=None)  # Record the nested import

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """
        Visit a from import statement node.

        :param node: The ImportFrom AST node
        :return: None
        """
        
        if self.container_stack:  # If inside a container
            module = node.module  # Get the module name
            names = [alias.name for alias in node.names]  # Extract imported names
            self.record_import(node, names, module=module)  # Record the nested import


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
    Verify if a path should be ignored based on IGNORE_DIRS.

    :param path: The file path to verify
    :return: True if the path contains any ignored directory, False otherwise
    """
    
    parts = set(Path(path).parts)  # Get the path parts as a set
    return bool(parts.intersection(IGNORE_DIRS))  # Check if any part intersects with ignored directories


def collect_python_files(root_dir: str) -> List[str]:
    """
    Collect all Python files under the root directory, excluding ignored directories.

    :param root_dir: The root directory to scan
    :return: A sorted list of Python file paths
    """
    
    verbose_output(true_string=f"{BackgroundColors.GREEN}Collecting Python files under: {BackgroundColors.CYAN}{root_dir}{Style.RESET_ALL}")  # Output the verbose message
    
    py_files: List[str] = []  # Initialize list for Python files
    for dirpath, dirnames, filenames in os.walk(root_dir):  # Walk through the directory tree
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]  # Modify dirnames in place to skip ignored dirs
        
        for fname in filenames:  # Iterate over filenames in current directory
            if not fname.endswith(".py"):  # Skip non-Python files
                continue  # Continue to next file
            
            full = os.path.join(dirpath, fname)  # Construct full file path
            
            if not os.path.isfile(full) or is_ignored(full):  # Skip if not a file or in ignored path
                continue  # Continue to next file
            
            py_files.append(full)  # Add valid Python file to list
            
    return sorted(py_files)  # Return sorted list of Python files


def analyze_file(path: str) -> List[Dict[str, Any]]:
    """
    Analyze a Python file for nested imports using AST.

    :param path: The path to the Python file
    :return: A list of nested import entries
    """
    
    verbose_output(true_string=f"{BackgroundColors.GREEN}Analyzing file: {BackgroundColors.CYAN}{path}{Style.RESET_ALL}")  # Output the verbose message
    
    try:  # Try to read the file
        with open(path, "r", encoding="utf-8") as fh:  # Open file for reading with UTF-8 encoding
            src = fh.read()  # Read the entire source code
    except Exception:
        return []  # Return empty list if file cannot be read
    
    try:  # Parse the source code into an AST tree
        tree = ast.parse(src, filename=path)  # Parse the source code into an AST tree
    except SyntaxError:
        return []  # Return empty list if syntax error in file
    
    visitor = ImportPlacementVisitor()  # Create an instance of the visitor
    visitor.visit(tree)  # Visit the AST tree to detect nested imports
    
    return visitor.nested_imports  # Return the list of nested imports found


def write_report(report: Dict[str, Any], out_path: str):
    """
    Write the report to a JSON file.

    :param report: The report dictionary
    :param out_path: The output file path
    :return: None
    """
    
    verbose_output(true_string=f"{BackgroundColors.GREEN}Writing report to: {BackgroundColors.CYAN}{out_path}{Style.RESET_ALL}")  # Output the verbose message
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)  # Ensure the output directory exists
    
    with open(out_path, "w", encoding="utf-8") as fh:  # Open file for writing with UTF-8 encoding
        json.dump(report, fh, indent=4)  # Write the report as formatted JSON


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
        f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Imports Placement Verifier{BackgroundColors.GREEN} program!{Style.RESET_ALL}",
        end="\n\n",
    )  # Output the welcome message

    start_time = datetime.datetime.now()  # Get the start time of the program

    files = collect_python_files(PROJECT_ROOT)  # Collect all Python files under the project root
    report: Dict[str, Any] = {}  # Initialize the report dictionary
    
    for current_file in files:  # Iterate over each Python file
        nested = analyze_file(current_file)  # Analyze the file for nested imports
        
        if nested:  # If nested imports were found
            rel = os.path.relpath(current_file, PROJECT_ROOT).replace("\\", "/")  # Get relative path and normalize separators
            report[rel] = nested  # Add to report
            verbose_output(true_string=f"Found {len(nested)} nested import(s) in: {rel}")
        
    write_report(report, OUTPUT_FILE)  # Write the report to the output file
    
    if report:  # If there are nested imports in the report
        print(f"{BackgroundColors.YELLOW}Nested imports detected in {len(report)} file(s). Report: {BackgroundColors.CYAN}{OUTPUT_FILE}{Style.RESET_ALL}")
    else:  # If there are no nested imports in the report
        print(f"{BackgroundColors.GREEN}No nested imports detected.{Style.RESET_ALL}")

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
