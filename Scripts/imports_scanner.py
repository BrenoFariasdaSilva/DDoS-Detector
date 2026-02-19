"""
================================================================================
Imports Scanner
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-02-19
Description :
    Scans Python source files in the current directory and under a `Scripts/`
    subdirectory and extracts top-level imported libraries using the standard
    library `ast` module. Only top-level module names are collected (for
    example, `pandas.core.frame` -> `pandas`) and relative imports are ignored.
    The script prints counts and a sorted unique list of detected libraries and
    preserves the project's logger redirection, execution timing, and sound
    behaviors.

    Key features include:
        - Scans all .py files in the current directory
        - Recursively scans all .py files under Scripts/
        - Parses source with `ast` to discover `import` and `from ... import ...`
        - Collects only top-level module names and ignores relative imports
        - Skips files with SyntaxError and continues scanning
        - Prints counts and a sorted unique list of libraries

Usage:
    1. Optionally create a `Scripts/` folder to include additional scripts.
    2. Run the script with Python:
        $ python main.py
    3. Inspect printed output and the log file in `./Logs/`.

Outputs:
    - Printed number of Python files scanned
    - Printed number of unique libraries detected
    - Printed sorted list of detected top-level libraries
    - Log file at `./Logs/main.log`

TODOs:
    - Add CLI flags to choose scan paths and toggle sound output
    - Optionally export detected libraries to a requirements-style file

Dependencies:
    - Python >= 3.8
    - ast (stdlib)
    - os (stdlib)
    - pathlib (stdlib)
    - colorama
    - Logger (project local module)

Assumptions & Notes:
    - Relative imports (for example `from . import x`) are ignored.
    - If `Scripts/` does not exist scanning continues silently.
    - Files with `SyntaxError` are skipped safely and do not abort execution.
"""


import ast  # For parsing Python source code
import atexit  # For playing a sound when the program finishes
import datetime  # For getting the current date and time
import os  # For running a command in the terminal
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


def extract_imports_from_file(filepath):
    """
    Extract top-level import module names from a Python file using ast.

    :param filepath: Path to the Python file
    :return: A set with top-level module names
    """

    try:  # Try to open and read the file
        text = Path(filepath).read_text(encoding="utf-8")  # Read file contents as text
    except Exception:  # If reading fails
        return set()  # Return empty set on error

    try:  # Try to parse the file into an AST
        node = ast.parse(text, filename=str(filepath))  # Parse source into AST
    except SyntaxError:  # If the file contains syntax errors
        return set()  # Skip files with syntax errors
    except Exception:  # Any other parsing error
        return set()  # Return empty set on error

    libs = set()  # Prepare a set to collect libraries
    for child in ast.walk(node):  # Walk the AST nodes
        if isinstance(child, ast.Import):  # Handle 'import x' nodes
            for alias in child.names:  # Iterate over imported aliases
                name = alias.name.split(".")[0]  # Take the top-level module name
                if name:  # If a name was found
                    libs.add(name)  # Add it to the set
        elif isinstance(child, ast.ImportFrom):  # Handle 'from x import y' nodes
            if getattr(child, "level", 0) and child.level > 0:  # Skip relative imports
                continue  # Continue to next node when relative import
            module = child.module  # Get the module attribute from the node
            if module:  # If module is present (not a relative bare import)
                name = module.split(".")[0]  # Take the top-level module name
                if name:  # If a name was found
                    libs.add(name)  # Add it to the set

    return libs  # Return the collected libraries


def collect_python_files():
    """
    Collect .py files in the current directory and recursively inside Scripts/.

    :return: A list of pathlib.Path objects for found Python files
    """

    project_root = Path(__file__).resolve().parent.parent  # Project root directory (parent of Scripts)
    files = []  # Initialize the list of files found

    for p in project_root.glob("*.py"):  # Gather .py files in the project root
        files.append(p)  # Append the found file to the list

    scripts_dir = project_root / "Scripts"  # Reference the Scripts directory under project root
    if scripts_dir.exists() and scripts_dir.is_dir():  # Only traverse when Scripts exists
        for p in scripts_dir.rglob("*.py"):  # Recursively gather .py files under Scripts/
            files.append(p)  # Append each found script file to the list

    local_dir = Path(__file__).parent  # Directory containing this script
    if local_dir.exists() and local_dir.is_dir():  # Ensure the local dir exists
        for p in local_dir.glob("*.py"):  # Gather immediate .py files in the local Scripts dir
            if p not in files:  # Avoid duplicates
                files.append(p)  # Append each local script file to the list

    return files  # Return the list of collected Python files


def build_relative_files_list_string(files):
    """
    Build a project-root-relative, alphabetically-sorted, comma-separated
    single-line representation of collected Python files.

    :param files: Iterable of pathlib.Path objects to format
    :return: A single-line string with comma-separated relative paths
    """

    project_root = Path(__file__).resolve().parent.parent  # Project root directory
    rel_paths = []  # Prepare list of relative path strings
    for fp in files:  # Iterate over provided file paths
        try:  # Try to compute a path relative to the project root
            rel = fp.relative_to(project_root)  # Relative path from project root
        except Exception:  # If relative conversion fails, fall back to the original path
            rel = fp  # Use absolute/path fallback
        rel_paths.append(str(rel))  # Append the stringified path to the list
    rel_paths.sort(key=lambda s: s.lower())  # Sort case-insensitively for deterministic output
    return ", ".join(rel_paths)  # Return the joined single-line string


def get_all_libraries():
    """
    Scan collected Python files and return the set of unique top-level libraries.

    :return: Tuple (number_of_files_scanned, set_of_library_names)
    """

    files = collect_python_files()  # Collect Python files to scan
    all_libs = set()  # Initialize the set of all libraries found
    for fp in files:  # Iterate over each file path
        libs = extract_imports_from_file(fp)  # Extract imports from the file
        all_libs.update(libs)  # Merge found libraries into the global set

    return len(files), all_libs  # Return count of files and the set of libraries


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

    print(
        f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Imports Scanner{BackgroundColors.GREEN} program!{Style.RESET_ALL}",
        end="\n",
    )  # Output the welcome message
    start_time = datetime.datetime.now()  # Get the start time of the program
    
    verbose_output(
        f"{BackgroundColors.GREEN}Scanning for Python files and extracting imports...{Style.RESET_ALL}",
        "",
    )  # Optional verbose message about starting the scan
    
    num_files_scanned, libraries = get_all_libraries()  # Scan files and collect libraries

    if num_files_scanned == 0:  # If no Python files were found
        print(
            f"{BackgroundColors.YELLOW}Warning: No Python files found in the current directory or Scripts/ directory.{Style.RESET_ALL}"
        )
    else:  # When one or more Python files were found
        print(
            f"{BackgroundColors.GREEN}Files scanned: {BackgroundColors.CYAN}{num_files_scanned}{BackgroundColors.GREEN}:{Style.RESET_ALL}"
        )
        files = collect_python_files()  # Collect the file Path objects to format output
        joined = build_relative_files_list_string(files)  # Build the formatted files list string
        print(f"{BackgroundColors.CYAN}[{joined}]{Style.RESET_ALL}")  # Print the single-line list of files

    unique_lib_count = len(libraries)  # Compute number of unique libraries found
    if unique_lib_count == 0:  # If no libraries were detected
        print(
            f"{BackgroundColors.YELLOW}No libraries detected in the scanned Python files.{Style.RESET_ALL}"
        )
    else:  # When one or more libraries were detected
        print(
            f"{BackgroundColors.GREEN}Libraries detected: {BackgroundColors.CYAN}{unique_lib_count}{Style.RESET_ALL}"
        )

    if unique_lib_count > 0:  # If there are libraries to list
        sorted_libs = sorted(libraries, key=lambda s: s.lower())  # Sort names case-insensitively
        print(f"{BackgroundColors.GREEN}Library list:{Style.RESET_ALL}")  # Header for the library list
        libs_lines = [f"  {BackgroundColors.CYAN}{lib}{Style.RESET_ALL}" for lib in sorted_libs]  # Build colored lines
        print("\n".join(libs_lines))  # Print all libraries joined by single newlines (no empty lines)

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
