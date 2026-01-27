"""
================================================================================
Python Comment Standardizer (renamer.py)
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-01-27
Description :
   This script standardizes Python comments in .py files located in the root
   directory of the DDoS-Detector project (non-recursive).

   It detects both full-line and inline comments and enforces:
      - Exactly one space after the '#' symbol.
      - Capitalization of the first letter of the comment text.

   Key features include:
      - Token-based parsing of Python files.
      - Support for full-line and inline comments.
      - Non-recursive directory scanning.
      - Safe modification of Python source code.

Usage:
   1. Edit ROOT_DIR if necessary.
   2. Run the script:
         $ python renamer.py

Outputs:
   - Updated .py files with standardized comments.

Dependencies:
   - Python >= 3.8
"""

import atexit
import datetime
import os
import platform
import sys
import tokenize
from io import BytesIO
from colorama import Style
from Logger import Logger
from pathlib import Path


# Macros:
class BackgroundColors:
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    CLEAR_TERMINAL = "\033[H\033[J"


# Execution Constants:
VERBOSE = False

# Root directory to process (non-recursive):
ROOT_DIR = r"D:\Backup\GitHub\Public\DDoS-Detector"

# Logger Setup:
logger = Logger(f"./Logs/{Path(__file__).stem}.log", clean=True)
sys.stdout = logger
sys.stderr = logger

# Sound Constants:
SOUND_COMMANDS = {
    "Darwin": "afplay",
    "Linux": "aplay",
    "Windows": "start",
}
SOUND_FILE = "./.assets/Sounds/NotificationSound.wav"

# RUN_FUNCTIONS:
RUN_FUNCTIONS = {
    "Play Sound": True,
}


def verbose_output(true_string="", false_string=""):
    """
    Outputs messages depending on the VERBOSE flag.
    """

    if VERBOSE and true_string != "":
        print(true_string)
    elif false_string != "":
        print(false_string)


def verify_filepath_exists(filepath):
    """
    Verify if a file or directory exists.
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Verifying path: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}"
    )
    return os.path.exists(filepath)


def calculate_execution_time(start_time, finish_time):
    """
    Calculate execution time in hh:mm:ss format.
    """

    delta = finish_time - start_time
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def play_sound():
    """
    Play a notification sound when execution finishes.
    """

    current_os = platform.system()
    if current_os == "Windows":
        return

    if verify_filepath_exists(SOUND_FILE):
        if current_os in SOUND_COMMANDS:
            os.system(f"{SOUND_COMMANDS[current_os]} {SOUND_FILE}")
        else:
            print(
                f"{BackgroundColors.RED}OS not supported: {BackgroundColors.CYAN}{current_os}{Style.RESET_ALL}"
            )
    else:
        print(
            f"{BackgroundColors.RED}Sound file not found: {BackgroundColors.CYAN}{SOUND_FILE}{Style.RESET_ALL}"
        )


def standardize_comment(raw: str) -> str:
    """
    Standardize a Python comment token.

    :param raw: Original comment string.
    :return: Standardized comment string.
    """

    body = raw[1:].strip()  # Remove '#' and surrounding whitespace
    if not body:  # Handle empty comments like '#'
        return "#"
    return "# " + body[0].upper() + body[1:]  # Ensure one space and capitalization


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
        if tok.type == tokenize.COMMENT:  # Check if the token is a comment
            new_comment = standardize_comment(tok.string)  # Standardize the comment
            if new_comment != tok.string:  # Check if modification is needed
                tok = tokenize.TokenInfo(  # Create a new modified token
                    tok.type,
                    new_comment,
                    tok.start,
                    tok.end,
                    tok.line,
                )
                modified = True  # Mark file as modified
        new_tokens.append(tok)  # Append token (modified or not)

    if modified:  # Rewrite file only if changes were made
        new_source = tokenize.untokenize(new_tokens)  # Rebuild source code
        with open(file_path, "wb") as f:  # Write updated source code back to file
            f.write(new_source)
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
        return

    for filename in os.listdir(ROOT_DIR):  # Iterate over files in root directory
        if filename.endswith(".py"):  # Filter Python files
            file_path = os.path.join(ROOT_DIR, filename)
            if os.path.isfile(file_path):  # Ensure it's a file
                process_file(file_path)  # Process the file


def main():
    """
    Main function.
    """

    print(
        f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}"
        f"Welcome to the {BackgroundColors.CYAN}Python Comment Standardizer{BackgroundColors.GREEN} program!"
        f"{Style.RESET_ALL}",
        end="\n\n",
    )

    start_time = datetime.datetime.now()  # Record start time

    run_comment_standardization()  # Execute core logic

    finish_time = datetime.datetime.now()  # Record finish time
    print(
        f"{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n"
        f"{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n"
        f"{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
    )
    print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}")

    (atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None)


if __name__ == "__main__":
    main()
