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
import re  # For splitting comment text while preserving whitespace
import string  # For punctuation classification used in token analysis
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


def resolve_entry_with_trailing_space(current_path: str, entry: str, stripped_part: str) -> str:
    """
    Resolve and optionally rename a directory entry with trailing spaces.

    :param current_path: Current directory path.
    :param entry: Directory entry name.
    :param stripped_part: Normalized target name without surrounding spaces.
    :return: Resolved path after optional rename.
    """

    try:  # Wrap full function logic to ensure safe execution
        resolved = os.path.join(current_path, entry)  # Build resolved path

        if entry != stripped_part:  # Verify trailing spaces exist
            corrected = os.path.join(current_path, stripped_part)  # Build corrected path
            try:  # Attempt to rename entry
                os.rename(resolved, corrected)  # Rename entry to stripped version
                verbose_output(true_string=f"{BackgroundColors.GREEN}Renamed: {BackgroundColors.CYAN}{resolved}{BackgroundColors.GREEN} -> {BackgroundColors.CYAN}{corrected}{Style.RESET_ALL}")  # Log rename
                resolved = corrected  # Update resolved path after rename
            except Exception:  # Handle rename failure
                verbose_output(true_string=f"{BackgroundColors.RED}Failed to rename: {BackgroundColors.CYAN}{resolved}{Style.RESET_ALL}")  # Log failure

        return resolved  # Return resolved path
    except Exception:  # Catch unexpected errors
        return os.path.join(current_path, entry)  # Return fallback resolved path


def resolve_full_trailing_space_path(filepath: str) -> str:
    """
    Resolve trailing space issues across all path components.

    :param filepath: Path to resolve potential trailing space mismatches.
    :return: Corrected full path if matches are found, otherwise original filepath.
    """

    try:  # Wrap full function logic to ensure safe execution
        verbose_output(true_string=f"{BackgroundColors.GREEN}Resolving full trailing space path for: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}")  # Log start

        if not isinstance(filepath, str) or not filepath:  # Verify filepath validity
            verbose_output(true_string=f"{BackgroundColors.YELLOW}Invalid filepath provided, skipping resolution.{Style.RESET_ALL}")  # Log invalid input
            return filepath  # Return original

        filepath = os.path.expanduser(filepath)  # Expand ~ to user directory
        parts = filepath.split(os.sep)  # Split path into components

        if not parts:  # Verify path parts exist
            return filepath  # Return original

        if filepath.startswith(os.sep):  # Handle absolute paths
            current_path = os.sep  # Start from root
            parts = parts[1:]  # Remove empty root part
        else:
            current_path = parts[0] if parts[0] else os.getcwd()  # Initialize base
            parts = parts[1:] if parts[0] else parts  # Adjust parts

        for part in parts:  # Iterate over each path component
            if part == "":  # Skip empty parts
                continue  # Continue iteration

            try:  # Attempt to list current directory
                entries = os.listdir(current_path) if os.path.isdir(current_path) else []  # List current directory entries
            except Exception:  # Handle failure to list directory contents
                verbose_output(true_string=f"{BackgroundColors.RED}Failed to list directory: {BackgroundColors.CYAN}{current_path}{Style.RESET_ALL}")  # Log failure
                return filepath  # Return original

            stripped_part = part.strip()  # Normalize current part
            match_found = False  # Initialize match flag

            for entry in entries:  # Iterate directory entries
                try:  # Attempt safe comparison for each entry
                    if entry.strip() == stripped_part:  # Compare stripped names
                        current_path = resolve_entry_with_trailing_space(current_path, entry, stripped_part)  # Resolve entry and update current path
                        match_found = True  # Mark match
                        break  # Stop searching
                except Exception:  # Handle any unexpected error during comparison
                    continue  # Continue on error

            if not match_found:  # If no match found for this segment
                verbose_output(true_string=f"{BackgroundColors.YELLOW}No match for segment: {BackgroundColors.CYAN}{part}{Style.RESET_ALL}")  # Log miss
                return filepath  # Return original

        return current_path  # Return fully resolved path

    except Exception:  # Catch unexpected errors to maintain stability
        verbose_output(true_string=f"{BackgroundColors.RED}Error resolving full path: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}")  # Log error
        return filepath  # Return original


def verify_filepath_exists(filepath):
    """
    Verify if a file or folder exists at the specified path.

    :param filepath: Path to the file or folder
    :return: True if the file or folder exists, False otherwise
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Verifying if the file or folder exists at the path: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}"
        )  # Output the verbose message
        
        if not isinstance(filepath, str) or not filepath.strip():  # Verify for non-string or empty/whitespace-only input   
            verbose_output(true_string=f"{BackgroundColors.YELLOW}Invalid filepath provided, skipping existence verification.{Style.RESET_ALL}")  # Log invalid input
            return False  # Return False for invalid input

        if os.path.exists(filepath):  # Fast path: original input exists
            return True  # Return True immediately

        candidate = str(filepath).strip()  # Normalize input to string and strip surrounding whitespace

        if (candidate.startswith("'") and candidate.endswith("'")) or (
            candidate.startswith('"') and candidate.endswith('"')
        ):  # Handle quoted paths from config files
            candidate = candidate[1:-1].strip()  # Remove wrapping quotes and trim again

        candidate = os.path.expanduser(candidate)  # Expand ~ to user home directory
        candidate = os.path.normpath(candidate)  # Normalize path separators and structure

        if os.path.exists(candidate):  # Verify normalized candidate directly
            return True  # Return True if normalized path exists

        repo_dir = os.path.dirname(os.path.abspath(__file__))  # Resolve repository directory
        cwd = os.getcwd()  # Capture current working directory

        alt = candidate.lstrip(os.sep) if candidate.startswith(os.sep) else candidate  # Prepare relative-safe path

        repo_candidate = os.path.join(repo_dir, alt)  # Build repo-relative candidate
        cwd_candidate = os.path.join(cwd, alt)  # Build cwd-relative candidate

        for path_variant in (repo_candidate, cwd_candidate):  # Iterate alternative base paths
            try:
                normalized_variant = os.path.normpath(path_variant)  # Normalize variant
                if os.path.exists(normalized_variant):  # Verify existence
                    return True  # Return True if found
            except Exception:
                continue  # Continue safely on error

        try:  # Attempt absolute path resolution as fallback
            abs_candidate = os.path.abspath(candidate)  # Build absolute path
            if os.path.exists(abs_candidate):  # Verify existence
                return True  # Return True if found
        except Exception:
            pass  # Ignore resolution errors

        for path_variant in (candidate, repo_candidate, cwd_candidate):  # Attempt trailing-space resolution on all variants
            try:  # Attempt to resolve trailing space issues across path components for this variant
                resolved = resolve_full_trailing_space_path(path_variant)  # Resolve trailing space issues across path components
                if resolved != path_variant and os.path.exists(resolved):  # Verify resolved path exists
                    verbose_output(
                        f"{BackgroundColors.YELLOW}Resolved trailing space mismatch: {BackgroundColors.CYAN}{path_variant}{BackgroundColors.YELLOW} -> {BackgroundColors.CYAN}{resolved}{Style.RESET_ALL}"
                    )  # Log successful resolution
                    return True  # Return True if corrected path exists
            except Exception:  # Catch any exception during trailing space resolution   
                continue  # Continue safely on error

        return False  # Not found after all resolution strategies
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


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
        if t.type == tokenize.STRING:  # Verifies if token is a string
            srow, scol = t.start  # Get start row and column
            erow, ecol = t.end  # Get end row and column
            string_spans.append((srow, scol, erow, ecol))  # Append span to list
    return string_spans  # Return the list of string spans


def is_comment_in_string(tok, string_spans: list) -> bool:
    """
    Verifies if a comment token is located inside a string literal.

    :param tok: The comment token.
    :param string_spans: List of string spans.
    :return: True if the comment is inside a string, False otherwise.
    """

    crow, ccol = tok.start  # Get comment's row and column
    for srow, scol, erow, ecol in string_spans:  # Iterate through string spans
        if srow <= crow <= erow:  # Verifies if comment row is within string span
            if srow == erow:  # Single-line string
                if scol <= ccol < ecol:  # Verifies if column is within span
                    return True  # Comment is inside string
            else:  # Multi-line string
                if (srow < crow < erow) or (crow == srow and ccol >= scol) or (
                    crow == erow and ccol < ecol
                ):  # Verifies conditions for multi-line
                    return True  # Comment is inside string
    return False  # Comment is not inside string


def is_all_upper(text: str) -> bool:
    """
    Verify whether text has letters and all letter characters are uppercase.

    :param text: Text to evaluate.
    :return: True if all letters are uppercase and at least one letter exists.
    """

    letters = [char for char in text if char.isalpha()]  # Collect only alphabetic characters
    
    if not letters:  # Verify if there are letters to evaluate
        return False  # Return False when no letters are present
    
    return all(char.isupper() for char in letters)  # Return True when all letters are uppercase


def is_all_lower(text: str) -> bool:
    """
    Verify whether text has letters and all letter characters are lowercase.

    :param text: Text to evaluate.
    :return: True if all letters are lowercase and at least one letter exists.
    """

    letters = [char for char in text if char.isalpha()]  # Collect only alphabetic characters
    
    if not letters:  # Verify if there are letters to evaluate
        return False  # Return False when no letters are present
    
    return all(char.islower() for char in letters)  # Return True when all letters are lowercase


def is_protected(token: str, strip_chars: str) -> bool:
    """
    Determine whether a token must be preserved exactly.

    :param token: Token to evaluate.
    :param strip_chars: Characters to strip from the token for analysis.
    :return: True when token is protected and must not be changed.
    """

    if "_" in token:  # Verify if token contains underscore
        return True  # Preserve identifiers with underscore
    if "-" in token:  # Verify if token contains hyphen
        return True  # Preserve hyphenated identifiers
    if any(ch.isdigit() for ch in token):  # Verify if token contains digits
        return True  # Preserve numeric tokens

    core = token.strip(strip_chars)  # Remove common surrounding punctuation for letter analysis

    alpha = "".join(ch for ch in core if ch.isalpha())  # Extract alphabetic characters
    if not alpha:  # Verify if no alphabetic characters remain
        return True  # Preserve tokens without letters (numbers/punctuation)

    if len(alpha) > 1 and alpha.isupper():  # Verify all-letters uppercase acronyms
        return True  # Preserve acronyms in uppercase form

    if any(c.isupper() for c in alpha) and any(c.islower() for c in alpha):  # Mixed case detection
        return True  # Preserve mixed-case identifiers

    return False  # Token is natural language and may be normalized


def transform_natural(token: str, capitalize_first: bool) -> str:
    """
    Transform a natural-language token to sentence-case form.

    :param token: Token to transform.
    :param capitalize_first: Whether to capitalize the first alphabetic character.
    :return: Transformed token.
    """

    chars = list(token)  # Operate on character list to preserve punctuation
    first_alpha_seen = False  # Track first alphabetic character inside this token

    for i, ch in enumerate(chars):  # Iterate characters to apply casing rules
        if ch.isalpha():  # Identify alphabetic characters only
            if not first_alpha_seen and capitalize_first:  # Capitalize the first alphabetic if requested
                chars[i] = ch.upper()  # Uppercase first alphabetic
                first_alpha_seen = True  # Mark that first alphabetic was handled
            else:
                chars[i] = ch.lower()  # Lowercase other alphabetic characters

    return "".join(chars)  # Rebuild token preserving non-letter characters


def to_sentence_case(text: str) -> str:
    """
    Convert text to sentence case using the first alphabetic character.

    :param text: Text to normalize.
    :return: Text converted to sentence case.
    """
    normalized = text  # Preserve provided text exactly for whitespace

    if not normalized:  # Verify if text is empty
        return normalized  # Return immediately when empty

    parts = re.split(r"(\s+)", normalized)  # Split text preserving whitespace separators

    strip_chars = "".join(ch for ch in string.punctuation if ch not in ("_", "-"))  # Punctuation to strip for core analysis (preserve underscores and hyphens)

    out_parts = []  # Collect transformed parts preserving original separators
    first_natural_found = False  # Track whether first natural-language word was already processed

    for part in parts:  # Iterate parts including whitespace separators
        if part.isspace():  # Preserve whitespace separators exactly
            out_parts.append(part)  # Append separator
            continue  # Move to next part

        if is_protected(part, strip_chars):  # Preserve protected tokens as-is
            out_parts.append(part)  # Append protected token unchanged
            continue  # Move to next part

        if not first_natural_found:  # First natural token must be capitalized
            out_parts.append(transform_natural(part, True))  # Capitalize first natural token
            first_natural_found = True  # Mark that first natural token has been handled
        else:
            out_parts.append(transform_natural(part, False))  # Lowercase subsequent natural tokens

    return "".join(out_parts)  # Reconstruct and return the normalized comment body


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

    is_full_line = original_line.strip().startswith("#")  # Verifies if it's a full-line comment

    if is_full_line:  # For full-line comments
        new_line = prefix + standardized + suffix  # Replace comment
    else:  # For inline comments
        body = standardized[1:].strip()  # Extract normalized inline comment body without the hash symbol
        if is_all_upper(body) or is_all_lower(body):  # If the comment body is all uppercase or all lowercase
            standardized = "# " + to_sentence_case(body)  # Convert inline comment text to sentence case

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
