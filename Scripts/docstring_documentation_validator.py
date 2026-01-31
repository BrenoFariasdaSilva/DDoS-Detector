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


# Classes Definitions:


class DocstringValidatorVisitor(ast.NodeVisitor):
    """
    AST visitor that validates and fixes docstrings for functions.
    """

    def __init__(self):
        """
        Initialize the DocstringValidatorVisitor.

        :param self: The instance of the class
        :return: None
        """
        
        self.missing: List[Dict[str, Any]] = []  # Initialize the list to store missing docstrings
        self.fixed: List[Dict[str, Any]] = []  # Initialize the list to store fixed docstrings

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """
        Visit a function definition node.

        :param self: The instance of the class
        :param node: The FunctionDef AST node
        :return: None
        """
        
        self.verify_docstring(node, "function")  # Verify the docstring for the function

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """
        Visit an async function definition node.

        :param self: The instance of the class
        :param node: The AsyncFunctionDef AST node
        :return: None
        """
        
        self.verify_docstring(node, "async_function")  # Verify the docstring for the async function

    def visit_ClassDef(self, node: ast.ClassDef):
        """
        Visit a class definition node to verify methods inside the class.

        :param self: The instance of the class
        :param node: The ClassDef AST node
        :return: None
        """
        
        self.generic_visit(node)  # Visit all nodes inside the class

    def handle_missing_docstring(self, node, func_type):
        """
        Handle the case where a docstring is missing.

        :param self: The instance of the class
        :param node: The function AST node
        :param func_type: Type of function ("function" or "async_function")
        :return: None
        """
        
        self.missing.append({  # Append the missing docstring details to the list
            "name": node.name,  # Add the name of the function
            "lineno": node.lineno,  # Add the line number of the function
            "type": func_type  # Add the type of the function
        })

    def ensure_single_line_delimiters(self, node, docstring, func_type):
        """
        Ensure that docstring delimiters are on a single line.

        :param self: The instance of the class
        :param node: The function AST node
        :param docstring: The docstring to verify
        :param func_type: Type of function ("function" or "async_function")
        :return: None
        """
        
        if "\n" in docstring or "\r" in docstring:  # Verify if the docstring contains newlines
            docstring_lines = docstring.splitlines()  # Split the docstring into lines
            fixed_docstring = f'"""{docstring_lines[0]}\n' + "\n".join(docstring_lines[1:]) + '"""'  # Fix the delimiters
            self.fixed.append({  # Append the fixed docstring details to the list
                "name": node.name,  # Add the name of the function
                "lineno": node.lineno,  # Add the line number of the function
                "type": func_type,  # Add the type of the function
                "fixed_docstring": fixed_docstring  # Add the fixed docstring
            })
            node.body.insert(0, ast.Expr(value=ast.Str(s=fixed_docstring)))  # Insert the fixed docstring into the node

    def ensure_empty_line_after_docstring(self, node):
        """
        Ensure that there is an empty line after the docstring.

        :param self: The instance of the class
        :param node: The function AST node
        :return: None
        """
        
        if not node.body or not isinstance(node.body[0], ast.Expr) or not isinstance(node.body[0].value, ast.Str):
            return  # Return if the node body is empty or the first node is not a docstring

        next_node = node.body[1] if len(node.body) > 1 else None  # Get the next node after the docstring
        if next_node and not isinstance(next_node, ast.Expr):  # Verify if the next node is not an expression
            node.body.insert(1, ast.Expr(value=ast.Str(s="")))  # Insert an empty line after the docstring

    def fix_docstring_if_needed(self, node, parsed, func_type):
        """
        Fix the docstring if it does not match the expected format.

        :param self: The instance of the class
        :param node: The function AST node
        :param parsed: The parsed docstring
        :param func_type: Type of function ("function" or "async_function")
        :return: None
        """
        
        param_names = [arg.arg for arg in node.args.args]  # Get the parameter names from the function arguments
        if parsed["param_names"] != param_names or parsed["add_empty"] or not parsed["return_line"]:  # Verify if the docstring needs fixing
            fixed_docstring = fix_docstring(parsed, param_names)  # Fix the docstring
            self.fixed.append({  # Append the fixed docstring details to the list
                "name": node.name,  # Add the name of the function
                "lineno": node.lineno,  # Add the line number of the function
                "type": func_type,  # Add the type of the function
                "fixed_docstring": fixed_docstring  # Add the fixed docstring
            })
            node.body.insert(0, ast.Expr(value=ast.Str(s=fixed_docstring)))  # Insert the fixed docstring into the node

    def verify_docstring(self, node, func_type):
        """
        Verify and fix the docstring for a function node.

        :param self: The instance of the class
        :param node: The function AST node
        :param func_type: Type of function ("function" or "async_function")
        :return: None
        """
        
        docstring = ast.get_docstring(node)  # Get the docstring of the function
        if not docstring:  # Verify if the docstring is missing
            self.handle_missing_docstring(node, func_type)  # Handle the missing docstring
            return  # Return as there is no docstring to verify

        self.ensure_single_line_delimiters(node, docstring, func_type)  # Ensure the docstring delimiters are on a single line
        self.ensure_empty_line_after_docstring(node)  # Ensure there is an empty line after the docstring

        parsed = parse_docstring(docstring)  # Parse the docstring
        if not parsed:  # Verify if the docstring could not be parsed
            self.handle_missing_docstring(node, func_type)  # Handle the missing docstring
            return  # Return as the docstring could not be parsed

        self.fix_docstring_if_needed(node, parsed, func_type)  # Fix the docstring if needed


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


def analyze_file(path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Analyze a Python file for docstring issues using AST.

    :param path: The path to the Python file
    :return: Dict with "missing" and "fixed" lists
    """

    verbose_output(true_string=
        f"{BackgroundColors.GREEN}Analyzing file: {BackgroundColors.CYAN}{path}{Style.RESET_ALL}"
    )  # Output the verbose message

    try:
        with open(path, "r", encoding="utf-8") as fh:  # Open the file for reading with UTF-8 encoding
            src = fh.read()  # Read the file content
    except Exception:  # Handle any exception that occurs while opening the file
        return {"missing": [], "fixed": []}  # Return empty lists for missing and fixed

    try:
        tree = ast.parse(src, filename=path)  # Parse the source code into an AST
    except SyntaxError:  # Handle syntax errors in the source code
        return {"missing": [], "fixed": []}  # Return empty lists for missing and fixed

    visitor = DocstringValidatorVisitor()  # Create an instance of the docstring validator visitor
    visitor.visit(tree)  # Visit the AST nodes to validate docstrings

    if visitor.fixed:  # If there are fixed docstrings
        try:
            new_src = astor.to_source(tree)  # Convert the modified AST back to source code
            with open(path, "w", encoding="utf-8") as fh:  # Open the file for writing with UTF-8 encoding
                fh.write(new_src)  # Write the modified source code back to the file
        except Exception as e:  # Handle any exception that occurs while writing the file
            verbose_output(true_string=
                f"{BackgroundColors.RED}Failed to write fixed file {path}: {e}{Style.RESET_ALL}"
            )  # Output the error message

    return {"missing": visitor.missing, "fixed": visitor.fixed}  # Return the lists of missing and fixed docstrings




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
        f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Docstring Documentation Validator{BackgroundColors.GREEN} program!{Style.RESET_ALL}",
        end="\n\n",
    )  # Output the welcome message
    start_time = datetime.datetime.now()  # Get the start time of the program

    files = collect_python_files(PROJECT_ROOT)  # Collect all Python files under the project root
    report: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}  # Initialize the report dictionary

    for current_file in files:  # Iterate over each Python file
        result = analyze_file(current_file)  # Analyze the current file
        if result["missing"] or result["fixed"]:  # Check if there are missing or fixed docstrings
            rel = os.path.relpath(current_file, PROJECT_ROOT).replace("\\", "/")  # Get the relative path
            report[rel] = result  # Add the result to the report
            verbose_output(
                true_string=
                f"Found issues in: {rel} (missing: {len(result['missing'])}, fixed: {len(result['fixed'])})"
            )  # Output verbose information about the issues

    write_report(report, OUTPUT_FILE)  # Write the report to the output file

    total_missing = sum(len(r["missing"]) for r in report.values())  # Calculate total missing docstrings
    total_fixed = sum(len(r["fixed"]) for r in report.values())  # Calculate total fixed docstrings

    if report:  # If there are any issues found
        print(
            f"{BackgroundColors.YELLOW}Docstring issues detected in {len(report)} file(s). Missing: {total_missing}, Fixed: {total_fixed}. Report: {BackgroundColors.CYAN}{OUTPUT_FILE}{Style.RESET_ALL}"
        )
    else:  # If no issues were found
        print(
            f"{BackgroundColors.GREEN}No docstring issues detected.{Style.RESET_ALL}"
        )

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
