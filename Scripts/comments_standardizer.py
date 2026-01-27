"""
================================================================================
Python Comment Standardizer (comments_standardizer.py)
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-01-27
Description :
    This script standardizes Python comments in .py files located under the
    `ROOT_DIR` (recursively). It skips directories listed in `IGNORE_DIRS`.

	It detects both full-line and inline comments and enforces:
		- Exactly one space after the "#" symbol.
		- Capitalization of the first letter of the comment text.
	It uses Python"s tokenize module to avoid modifying "#" characters inside
	strings and to safely handle inline comments.

	Key features include:
		- Token-based parsing of Python files (safe and precise).
		- Support for full-line and inline comments.
        - Recursive directory scanning (skips paths in `IGNORE_DIRS`).
		- Automatic in-place modification of files.
		- Robust handling of edge cases (empty comments, indentation, etc.).

Usage:
	1. Edit ROOT_DIR if necessary to point to the target directory.
	2. Execute the script:
        $ python comments_standardizer.py
	3. All .py files in the root directory will be updated in place.

Outputs:
	- Modified .py files with standardized comments in the target directory.

TODOs:
	- Add recursive directory traversal option.
	- Add dry-run mode to preview changes without modifying files.
	- Add CLI arguments for directory selection.
	- Add logging of modified files and lines.

Dependencies:
	- Python >= 3.8
	- Standard library only (os, tokenize, io, pathlib, datetime, etc.)

Assumptions & Notes:
    - Processes .py files under `ROOT_DIR` recursively, skipping directories
      listed in `IGNORE_DIRS`.
	- Comments inside strings are not modified.
	- Shebangs and encoding comments are preserved safely by tokenize.
	- Files are rewritten only if modifications are detected.
"""

import atexit  # For playing a sound when the program finishes
import datetime  # For getting the current date and time
import os  # For running a command in the terminal
import platform  # For getting the operating system name
import sys  # For system-specific parameters and functions
import tokenize  # For safe Python token parsing
from io import BytesIO  # For tokenizing byte streams
from colorama import Style  # For coloring the terminal
from pathlib import Path  # For handling file paths

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

# Root directory to process (non-recursive):
ROOT_DIR = str(Path(__file__).resolve().parent / "..")  # Parent directory of this script
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

    if VERBOSE and true_string != "":  # If the VERBOSE constant is set to True and the true_string is set
        print(true_string)  # Output the true statement string
    elif false_string != "":  # If the false_string is set
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


def standardize_comment(raw: str) -> str:
    """
    Receives a full comment token starting with "#".
    Ensures exactly one space after "#" and capitalizes the first letter of the text.

    :param raw: Original comment token string.
    :return: Standardized comment string.
    """

    body = raw[1:].strip()  # Remove "#" and trim spaces
    if not body:  # Empty comment
        return "#"  # Return just "#"
    return "# " + body[0].upper() + body[1:]  # Standardize comment


def process_file(file_path: str) -> None:
    """
    Process a single Python file, standardizing its comments.

    :param file_path: Path to the .py file.
    :return: None
    """

    with open(file_path, "rb") as f:  # Open file in binary mode for tokenization
        source = f.read()  # Read file content

    tokens = list(tokenize.tokenize(BytesIO(source).readline))  # Tokenize the source code
    modified = False  # Track if any modifications were made
    new_tokens = []  # List to hold modified tokens

    for tok in tokens:  # Iterate through all tokens
        if tok.type == tokenize.COMMENT:  # Verify if the token is a comment
            new_comment = standardize_comment(tok.string)  # Standardize the comment

            # Determine whether this is a full-line comment or inline.
            line = tok.line or ""  # Get the full line of code
            is_full_line = line.strip().startswith("#")  # Verify if it's a full-line comment

            tok_str = new_comment  # Start with the standardized comment

            if not is_full_line and "#" in line:  # Inline comment
                hash_idx = line.find("#")  # Find index of '#'
                spaces_before = 0  # Count spaces before '#'
                i = hash_idx - 1  # Start checking before '#'
                while i >= 0 and line[i] == " ":  # Count spaces before '#'
                    spaces_before += 1  # Increment space count
                    i -= 1  # Move left

                if spaces_before == 1:  # Exactly one space before '#'
                    tok_str = " " + new_comment  # Prepend single space

            if tok_str != tok.string:  # If the comment was modified
                tok = tokenize.TokenInfo(  # Create a new modified token
                    tok.type,
                    tok_str,
                    tok.start,
                    tok.end,
                    tok.line,
                )
                modified = True  # Mark file as modified

        new_tokens.append(tok)  # Append token (modified or not)

    if modified:  # Rewrite file only if changes were made
        new_source = tokenize.untokenize(new_tokens)  # Rebuild source code
        data = new_source if isinstance(new_source, (bytes, bytearray)) else new_source.encode("utf-8")  # Ensure bytes
        with open(file_path, "wb") as f:  # Write updated source code back to file
            f.write(data)  # Write new content as bytes
        verbose_output(
            f"{BackgroundColors.GREEN}Updated comments in: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}"
        )


def run_comment_standardization():
    """
    Run comment standardization on all .py files in ROOT_DIR (non-recursive).

    :return: None
    """

    if not verify_filepath_exists(ROOT_DIR):  # Validate root directory existence
        print(
            f"{BackgroundColors.RED}Directory not found: {BackgroundColors.CYAN}{ROOT_DIR}{Style.RESET_ALL}"
        )
        return  # Exit if directory does not exist

    for dirpath, dirnames, filenames in os.walk(ROOT_DIR):  # Walk through the directory
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]  # Skip ignored directories

        for filename in filenames:  # Process each file
            if not filename.endswith(".py"):  # Skip non-.py files
                continue  # Continue to next file
            file_path = os.path.join(dirpath, filename)  # Get full file path
            if os.path.isfile(file_path):  # Ensure it's a file
                parts = set(Path(file_path).parts)  # Get path parts
                if parts.intersection(IGNORE_DIRS):  # Verify if any part is in IGNORE_DIRS
                    continue  # Skip ignored directories
                process_file(file_path)  # Process the Python file


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
        f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Python Comment Standardizer{BackgroundColors.GREEN} program!{Style.RESET_ALL}",
        end="\n\n",
    )
    start_time = datetime.datetime.now()  # Get the start time of the program

    run_comment_standardization()  # Run the comment standardization

    finish_time = datetime.datetime.now()  # Get the finish time of the program
    print(
        f"{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n"
        f"{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n"
        f"{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
    )
    print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}")

    (atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None)  # Register the play_sound function to be called when the program finishes


if __name__ == "__main__":
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """

    main()  # Call the main function
