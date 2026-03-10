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
    1. Configure TARGET_FILE_PATH in the Execution Constants section to target a Python file.
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
TARGET_FILE_PATH = Path("./main.py")  # Path to the target Python file to analyze
OUTPUT_FILE = None  # Output path computed at runtime from TARGET_FILE_PATH

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


def read_source_file(filepath: Path) -> str:
    """
    Reads and returns the full text content of a Python source file.

    :param filepath: Path object pointing to the target Python source file.
    :return: The full text content of the file as a UTF-8 decoded string.
    """

    verbose_output(f"{BackgroundColors.GREEN}Reading source file: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}")  # Log the file read operation

    return filepath.read_text(encoding="utf-8")  # Read and return the file content with UTF-8 encoding


def parse_ast_tree(source_text: str, filepath: Path) -> ast.Module:
    """
    Parses Python source text into an AST module node.

    :param source_text: The full text content of the Python source file.
    :param filepath: Path object used for reporting the filename during parsing.
    :return: The root ast.Module node of the parsed abstract syntax tree.
    """

    verbose_output(f"{BackgroundColors.GREEN}Parsing AST for: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}")  # Log the AST parsing operation

    return ast.parse(source_text, filename=str(filepath))  # Parse the source text and return the module node


def build_parent_map(tree: ast.Module) -> dict:
    """
    Builds a mapping from each AST node's id to its direct parent node.

    :param tree: The root ast.Module node of the abstract syntax tree.
    :return: A dictionary mapping id(child_node) to its direct parent AST node.
    """

    verbose_output(f"{BackgroundColors.GREEN}Building AST parent map...{Style.RESET_ALL}")  # Log the parent map construction

    parent_map = {}  # Initialize empty dictionary to store parent relationships
    
    for parent_node in ast.walk(tree):  # Walk every node in the AST tree
        for child_node in ast.iter_child_nodes(parent_node):  # Iterate over immediate children of the current node
            parent_map[id(child_node)] = parent_node  # Map the child node's id to its parent node

    return parent_map  # Return the completed parent map


def get_node_end_lineno(node: ast.AST) -> int:
    """
    Return the best-effort end line number for an AST node.

    :param node: The AST node to inspect.
    :return: The estimated end line number for the node.
    """

    max_lineno = getattr(node, "end_lineno", None)  # Try direct end_lineno attribute first
   
    if max_lineno is None:  # Verify if end_lineno was not provided by the parser
        max_lineno = getattr(node, "lineno", 0)  # Fallback to the node.lineno or zero when missing
    
    for child in ast.walk(node):  # Iterate all descendant nodes to find the largest line number
        child_end = getattr(child, "end_lineno", None)  # Attempt to read child's end_lineno attribute
        
        if child_end is not None:  # Verify the child provides an explicit end_lineno
            if child_end > max_lineno:  # Compare child's end with current maximum
                max_lineno = child_end  # Update maximum when child's end is larger
        else:  # Handle children that only expose lineno
            child_ln = getattr(child, "lineno", None)  # Attempt to read child's lineno attribute
            if child_ln is not None and child_ln > max_lineno:  # Verify and compare child's lineno
                max_lineno = child_ln  # Update maximum when child's lineno is larger

    return int(max_lineno)  # Return the computed maximum as an integer


def collect_class_methods(tree: ast.Module, parent_map: dict) -> dict:
    """
    Collects all method definitions inside class blocks from the AST.

    :param tree: The root ast.Module node of the abstract syntax tree.
    :param parent_map: A dictionary mapping id(node) to its direct parent AST node.
    :return: A dictionary mapping "Class ClassName" keys to sorted lists of method metadata.
    """

    verbose_output(f"{BackgroundColors.GREEN}Collecting class methods from AST...{Style.RESET_ALL}")  # Log the class method collection operation

    classes = {}  # Initialize empty dictionary to store class method data
    
    for node in ast.walk(tree):  # Walk every node in the AST tree
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):  # Verify if the node is a function definition
            parent_node = parent_map.get(id(node))  # Retrieve the parent node for the current function
            if isinstance(parent_node, ast.ClassDef):  # Verify if the parent is a class definition
                end_ln = get_node_end_lineno(node)  # Get best-effort end line for the function node
                size = end_ln - node.lineno + 1  # Compute function size as the line span using best-effort end
                entry = {  # Build the method metadata dictionary
                    "function_name": node.name,  # Store the method name
                    "function_size": size,  # Store the computed size in lines
                    "start_line": node.lineno,  # Store the starting line number
                    "end_line": end_ln,  # Store the computed ending line number
                }
                class_key = f"Class {parent_node.name}"  # Build the class key string
                classes.setdefault(class_key, []).append(entry)  # Append the entry to the class key list

    for class_key in classes:  # Iterate over each class key to sort its methods
        classes[class_key].sort(key=lambda e: e["function_size"], reverse=True)  # Sort methods by size descending

    return classes  # Return the completed class methods dictionary


def collect_top_level_functions(tree: ast.Module, parent_map: dict) -> list:
    """
    Collects all function definitions that are direct children of the module node.

    :param tree: The root ast.Module node of the abstract syntax tree.
    :param parent_map: A dictionary mapping id(node) to its direct parent AST node.
    :return: A list of top-level function metadata dictionaries sorted by size descending.
    """

    verbose_output(f"{BackgroundColors.GREEN}Collecting top-level functions from AST...{Style.RESET_ALL}")  # Log the top-level function collection operation

    top_level = []  # Initialize empty list to store top-level function data
    
    for node in ast.walk(tree):  # Walk every node in the AST tree
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):  # Verify if the node is a function definition
            parent_node = parent_map.get(id(node))  # Retrieve the parent node for the current function
            if isinstance(parent_node, ast.Module):  # Verify if the parent is the module root
                end_ln = get_node_end_lineno(node)  # Get best-effort end line for the function node
                size = end_ln - node.lineno + 1  # Compute function size as the line span using best-effort end
                entry = {  # Build the function metadata dictionary
                    "function_name": node.name,  # Store the function name
                    "function_size": size,  # Store the computed size in lines
                    "start_line": node.lineno,  # Store the starting line number
                    "end_line": end_ln,  # Store the computed ending line number
                }
                top_level.append(entry)  # Append the entry to the top-level list

    top_level.sort(key=lambda e: e["function_size"], reverse=True)  # Sort top-level functions by size descending

    return top_level  # Return the sorted list of top-level functions


def collect_nested_functions(tree: ast.Module, parent_map: dict) -> list:
    """
    Collects all function definitions that are direct children of another function definition.

    :param tree: The root ast.Module node of the abstract syntax tree.
    :param parent_map: A dictionary mapping id(node) to its direct parent AST node.
    :return: A list of nested function metadata dictionaries sorted by size descending.
    """

    verbose_output(f"{BackgroundColors.GREEN}Collecting nested functions from AST...{Style.RESET_ALL}")  # Log the nested function collection operation

    nested = []  # Initialize empty list to store nested function data
    
    for node in ast.walk(tree):  # Walk every node in the AST tree
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):  # Verify if the node is a function definition
            parent_node = parent_map.get(id(node))  # Retrieve the parent node for the current function
            if isinstance(parent_node, (ast.FunctionDef, ast.AsyncFunctionDef)):  # Verify if the parent is also a function definition
                end_ln = get_node_end_lineno(node)  # Get best-effort end line for the nested function node
                size = end_ln - node.lineno + 1  # Compute function size as the line span using best-effort end
                entry = {  # Build the nested function metadata dictionary
                    "nested_function_name": node.name,  # Store the nested function name
                    "parent_function_name": parent_node.name,  # Store the parent function name
                    "function_size": size,  # Store the computed size in lines
                    "start_line": node.lineno,  # Store the starting line number
                    "end_line": end_ln,  # Store the computed ending line number
                }
                nested.append(entry)  # Append the entry to the nested list

    nested.sort(key=lambda e: e["function_size"], reverse=True)  # Sort nested functions by size descending

    return nested  # Return the sorted list of nested functions


def build_report(tree: ast.Module) -> dict:
    """
    Builds the complete JSON report structure from the parsed AST.

    :param tree: The root ast.Module node of the abstract syntax tree.
    :return: A dictionary representing the complete function size report.
    """

    verbose_output(f"{BackgroundColors.GREEN}Building function size report...{Style.RESET_ALL}")  # Log the report build operation

    parent_map = build_parent_map(tree)  # Build the parent map for the entire AST tree
    classes = collect_class_methods(tree, parent_map)  # Collect all class method metadata
    top_level = collect_top_level_functions(tree, parent_map)  # Collect all top-level function metadata
    nested = collect_nested_functions(tree, parent_map)  # Collect all nested function metadata

    total_methods = sum(len(methods) for methods in classes.values())  # Compute the total method count across all classes
    total_functions = total_methods + len(top_level) + len(nested)  # Compute the grand total across all categories

    report = {  # Build the complete report dictionary
        "total_functions": total_functions,  # Store the total function count
        "classes": classes,  # Store the class methods data
        "top-level functions": top_level,  # Store the top-level functions data
        "nested functions": nested,  # Store the nested functions data
    }

    return report  # Return the completed report dictionary


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

    if not verify_filepath_exists(TARGET_FILE_PATH):  # Verify if the target file exists
        print(f"{BackgroundColors.RED}Error: Target file {BackgroundColors.CYAN}{TARGET_FILE_PATH}{BackgroundColors.RED} not found!{Style.RESET_ALL}")  # Log the missing file error
        return  # Exit the main function early

    print(f"{BackgroundColors.GREEN}Analyzing file: {BackgroundColors.CYAN}{TARGET_FILE_PATH}{Style.RESET_ALL}")  # Log the analysis start message

    source_text = read_source_file(TARGET_FILE_PATH)  # Read the source file content
    tree = parse_ast_tree(source_text, TARGET_FILE_PATH)  # Parse the source text into an AST tree
    report = build_report(tree)  # Build the complete function size report

    report = {"filename": TARGET_FILE_PATH.name, **report}  # Prepend filename as first entry in the final report
    save_report(report, Path(OUTPUT_FILE))  # Save the report to the output JSON file

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
