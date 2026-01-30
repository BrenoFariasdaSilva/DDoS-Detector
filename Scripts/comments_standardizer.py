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
from colorama import Style  # For coloring the terminal
from io import BytesIO  # For tokenizing byte streams
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

def verbose_output(true_string="", false_string="", telegram_bot=None):
    """
    Outputs a message if the VERBOSE constant is set to True.

    :param true_string: The string to be outputted if the VERBOSE constant is set to True.
    :param false_string: The string to be outputted if the VERBOSE constant is set to False.
    :param telegram_bot: Optional TelegramBot instance to send the message to Telegram.
    :return: None
    """

    if VERBOSE and true_string != "":  # If VERBOSE is True and a true_string was provided
        print(true_string)  # Output the true statement string
        if telegram_bot is not None:  # If a Telegram bot instance was provided
            send_telegram_message(telegram_bot, [true_string])  # Send the true_string to Telegram
    elif false_string != "":  # If a false_string was provided
        print(false_string)  # Output the false statement string
        if telegram_bot is not None:  # If a Telegram bot instance was provided
            send_telegram_message(telegram_bot, [false_string])  # Send the false_string to Telegram


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


def read_python_file(file_path: str) -> bytes:
    """
    Read a Python file in binary mode.

    :param file_path: Path to the .py file.
    :return: File content as bytes.
    """

    with open(file_path, "rb") as f:  # Open file in binary mode
        return f.read()  # Return file content


def decode_source_to_text(source: bytes) -> str:
    """
    Decode the source bytes to a UTF-8 string, with fallback error handling.

    :param source: The source code as bytes.
    :return: The decoded text as a string.
    """

    try:
        text = source.decode("utf-8")  # Attempt to decode as UTF-8
    except Exception:
        text = source.decode("utf-8", errors="replace")  # Fallback with replacement characters
    return text  # Return the decoded text


def tokenize_source(source: bytes):
    """
    Tokenize Python source code.

    :param source: Python source code as bytes.
    :return: List of tokens.
    """

    return list(tokenize.tokenize(BytesIO(source).readline))  # Tokenize source code


def build_string_spans(tokens) -> list:
    """
    Build a list of string token spans to identify string literals.

    :param tokens: List of tokens from tokenize.
    :return: List of tuples (start_row, start_col, end_row, end_col) for string tokens.
    """

    string_spans = []  # Initialize list for string spans
    for t in tokens:  # Iterate through all tokens
        if t.type == tokenize.STRING:  # Check if token is a string
            srow, scol = t.start  # Get start row and column
            erow, ecol = t.end  # Get end row and column
            string_spans.append((srow, scol, erow, ecol))  # Append span to list
    return string_spans  # Return the list of string spans


def is_comment_in_string(tok, string_spans: list) -> bool:
    """
    Check if a comment token is located inside a string literal.

    :param tok: The comment token.
    :param string_spans: List of string spans.
    :return: True if the comment is inside a string, False otherwise.
    """

    crow, ccol = tok.start  # Get comment's row and column
    for srow, scol, erow, ecol in string_spans:  # Iterate through string spans
        if srow <= crow <= erow:  # Check if comment row is within string span
            if srow == erow:  # Single-line string
                if scol <= ccol < ecol:  # Check if column is within span
                    return True  # Comment is inside string
            else:  # Multi-line string
                if (srow < crow < erow) or (crow == srow and ccol >= scol) or (
                    crow == erow and ccol < ecol
                ):  # Check conditions for multi-line
                    return True  # Comment is inside string
    return False  # Comment is not inside string


def process_comment_line(original_line: str, tok_string: str, start_col: int) -> str:
    """
    Process a single comment line, standardizing the comment and adjusting whitespace.

    :param original_line: The original line containing the comment.
    :param tok_string: The original comment token string.
    :param start_col: The column index where the comment starts (position of "#").
    :return: The modified line with standardized comment.
    """

    standardized = capitalize_comment_text(tok_string)  # Standardize the comment text

    hash_idx = start_col  # Use the token"s start column for "#"

    prefix = original_line[:hash_idx]  # Get text before "#"
    suffix = original_line[hash_idx + len(tok_string) :]  # Get text after the comment

    is_full_line = original_line.strip().startswith("#")  # Check if it's a full-line comment

    if is_full_line:  # For full-line comments
        new_line = prefix + standardized + suffix  # Replace comment
    else:  # For inline comments
        spaces_before = 0  # Count spaces before "#"
        i = hash_idx - 1  # Start from before "#"
        while i >= 0 and original_line[i] == " ":  # Count spaces
            spaces_before += 1  # Increment count
            i -= 1  # Move left
        code_before = original_line[:hash_idx - spaces_before]  # Get code before spaces
        new_line = code_before + "  " + standardized + suffix  # Ensure exactly two spaces

    return new_line  # Return the modified line


def capitalize_comment_text(raw_comment: str) -> str:
    """
    Standardize the text of a Python comment by ensuring capitalization and spacing after "#".

    :param raw_comment: Original comment token string.
    :return: Standardized comment string.
    """

    body = raw_comment[1:].strip()  # Remove "#"" and trim surrounding whitespace
    if not body:  # Handle empty comments like "#""
        return "#"
    return "# " + body[0].upper() + body[1:]  # Capitalize first letter and ensure one space


def write_modified_file(file_path: str, lines: list, modified: bool) -> None:
    """
    Write the modified lines back to the file if changes were made.

    :param file_path: Path to the file.
    :param lines: List of modified lines.
    :param modified: Flag indicating if modifications occurred.
    :return: None
    """

    if modified:  # Only write if modified
        new_text = "".join(lines)  # Join lines into text
        data = new_text.encode("utf-8")  # Encode to bytes
        with open(file_path, "wb") as f:  # Open file in binary mode
            f.write(data)  # Write data
        verbose_output(  # Output success message
            f"{BackgroundColors.GREEN}Updated comments in: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}"
        )


def process_file(file_path: str) -> None:
    """
    Process a single Python file, standardizing its comments.

    :param file_path: Path to the .py file.
    :return: None
    """

    source = read_python_file(file_path)  # Read file content as bytes
    text = decode_source_to_text(source)  # Decode to text

    tokens = tokenize_source(source)  # Tokenize source code

    lines = text.splitlines(keepends=True)  # Prepare mutable list of lines
    modified = False  # Track modifications

    string_spans = build_string_spans(tokens)  # Build string spans

    for tok in tokens:  # Iterate through tokens
        if tok.type != tokenize.COMMENT:  # Skip non-comment tokens
            continue  # Continue to next token

        line_no = tok.start[0]  # Get line number
        if line_no < 1 or line_no > len(lines):  # Validate line number
            continue  # Skip invalid line numbers

        if is_comment_in_string(tok, string_spans):  # Skip if in string
            continue  # Continue to next token

        original_line = lines[line_no - 1]  # Get original line
        new_line = process_comment_line(original_line, tok.string, tok.start[1])  # Process line

        if new_line != original_line:  # If changed
            lines[line_no - 1] = new_line  # Update line
            modified = True  # Mark as modified

    write_modified_file(file_path, lines, modified)  # Write back if modified


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
