"""
================================================================================
YAML configuration sections updater - configuration_sections.py
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-03-01
Description :
    Utilities for loading, normalizing and rebuilding the project's `config.yaml`.
    This module provides functions that:
        - Parse an existing `config.yaml` and extract inline comments and original
          quoting for list items.
        - Recursively sort mapping keys and separate top-level sections into
          "general" and per-script (python) sections.
        - Reconstruct an annotated, human-readable `config.yaml` and write a
          synchronized `config.yaml.example`.

Usage:
    Run the script to regenerate `config.yaml` and `config.yaml.example` from
    the repository configuration. It can also be used programmatically by
    calling `write_configs(config_path, example_path)`.

    $ python Scripts/configuration_sections.py

Outputs:
    - Updated `config.yaml` at repository root.
    - Synchronized `config.yaml.example` at repository root.
    - Log file written to `Logs/configuration_sections.log`.

TODOs:
    - Add CLI options to target alternate config files and control verbosity.
    - Add unit tests for comment extraction and list-quoting edge cases.
    - Consider supporting configurable indentation widths and YAML anchors.

Dependencies:
    - Python 3.8+
    - pyyaml
    - colorama

Assumptions & Notes:
    - The comment extractor assumes two-space indentation for nesting levels.
    - The script preserves inline comments and whether list items were quoted
      in the original file when reconstructing YAML blocks.
    - Sound notification is skipped on Windows by design.
"""


import atexit  # For playing a sound when the program finishes
import datetime  # For getting the current date and time
import os  # For running a command in the terminal
import platform  # For getting the operating system name
import re  # For parsing inline comments from existing YAML files
import sys  # For system-specific parameters and functions
import yaml  # YAML loader and dumper
from collections import OrderedDict  # Ordered mapping for deterministic output
from colorama import Style  # For coloring the terminal
from pathlib import Path  # For handling file paths
from typing import cast, Iterable, Any  # Cast and Iterable and Any for safe type coercion
from typing import Dict, Tuple  # Type hints for function signatures


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
ROOT_DIR = str(Path(__file__).resolve().parent / "..")  # Directory to scan


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


def HandleMappingKey(line: str, stack: list[str]) -> bool:
    """
    Handle a YAML mapping key line and update the stack accordingly.

    :param line: The raw line from the YAML file to inspect.
    :param stack: The stack tracking nested mapping keys by level.
    :return: True when the line contained a mapping key, False otherwise.
    """

    m = re.match(r"^(\s*)([^:\n]+):(?:\s*)$", line)  # Match indented mapping key lines like 'key:'
    
    if m:  # When a mapping key line without a value is detected
        indent = len(m.group(1))  # Measure leading spaces to infer nesting
        level = indent // 2  # Compute YAML nesting level using two-space indent convention
        key = m.group(2).strip()  # Extract the mapping key name from the match
        UpdateStackForKey(stack, level, key)  # Update stack with key and truncate deeper levels
        return True  # Indicate the line was handled as a mapping key
    
    m2 = re.match(r"^(\s*)([^:]+):\s*(.+)$", line)  # Match single-line 'key: value' entries
    
    if m2:  # When a single-line mapping with a value is detected
        indent = len(m2.group(1))  # Measure leading spaces for nesting computation
        level = indent // 2  # Compute the nesting level using two-space indentation
        key = m2.group(2).strip()  # Extract the mapping key before the ':'
        UpdateStackForKey(stack, level, key)  # Update stack with key and truncate deeper levels
        return True  # Indicate the line was handled as a single-line mapping
    
    return False  # Indicate the line did not represent a mapping key


def HandleInlineComment(before: str, comment: str, stack: list[str], comments: dict) -> None:
    """
    Process the portion before an inline comment and record the comment for the key path.

    :param before: The text before the inline '#' marker on the line.
    :param comment: The inline comment text to associate with the key path.
    :param stack: The stack tracking nested keys by indentation level.
    :param comments: The mapping to populate from key-path tuples to comment strings.
    :return: None.
    """

    m = re.match(r"^(\s*)([^:]+):(?:\s*(.*))?$", before)  # Parse the leading 'key:' portion before '#'
    
    if not m:  # If parsing fails, do not record anything for this line
        return None  # Return None to indicate no action was performed
    
    indent = len(m.group(1))  # Determine indentation width in spaces
    level = indent // 2  # Compute the YAML nesting level using two-space indent
    key = m.group(2).strip()  # Extract the mapping key name
    
    EnsureStackLength(stack, level)  # Ensure the stack has room for this nesting level
    
    stack[level] = key  # Record the key at the computed level in the stack
    del stack[level + 1 :]  # Truncate any deeper levels beyond the current one
    path = tuple([k for k in stack if k])  # Build a tuple path from non-empty stack entries
    
    if path:  # Only record when path is non-empty
        comments[path] = comment  # Associate the extracted comment with the key-path tuple


def extract_inline_comments(file_path: str, unused: object) -> tuple:
    """
    Extract inline comments from an existing YAML file preserving their mapping to dotted key paths.

    :param file_path: Path to the YAML file to parse for inline comments.
    :param unused: Placeholder parameter to preserve two-argument signature.
    :return: Mapping from tuple key-path (e.g., ("execution","verbose")) to comment string.
    """

    comments: dict = {}  # Initialize mapping for extracted inline comments
    quoted_items: dict = {}  # Initialize mapping for recording quoted list-item flags per key-path
    p = Path(file_path)  # Convert to Path for operations
    if not p.exists():  # Verify file exists before attempting to read
        return comments, quoted_items  # Return empty maps when file missing

    with p.open("r", encoding="utf-8") as fh:  # Open original YAML for reading
        stack: list[str] = []  # Stack to track current key path by indentation level
        for raw_line in fh:  # Iterate each line in the original file
            line = raw_line.rstrip("\n")  # Strip newline but preserve trailing spaces for comment detection
            if not line.strip():  # Skip empty lines silently
                continue  # Continue to next line when blank
            stripped = line.lstrip(" ")  # Left-trim spaces for content inspection
            if stripped.startswith("#"):  # Skip full-line comments (not inline)
                continue  # Continue to next line when full-line comment
            hash_idx = line.find("#")  # Find inline comment marker position if present
            if hash_idx == -1:  # No inline comment on this line
                
                mlist = re.match(r"^(\s*)-\s*(.*)$", line)  # Match list item lines like '- value'
                if mlist:  # When a list item without inline comment is present
                    item = mlist.group(2).strip()  # Extract the scalar portion after '-'
                    path = tuple([k for k in stack if k])  # Build current key-path tuple from stack
                    quoted = False  # Default: not quoted
                    if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):  # Already quoted in original
                        quoted = True  # Mark as originally quoted
                    quoted_items.setdefault(path, []).append(quoted)  # Record quoted flag for this list item index
                    handled = HandleMappingKey(line, stack)  # Try to handle mapping key or single-line mapping (no-op for list items)
                    if handled:  # If mapping key was handled update next line
                        continue  # Continue to next line when mapping handled
                    continue  # Continue to next line after recording list-item quote flag
                handled = HandleMappingKey(line, stack)  # Try to handle mapping key or single-line mapping
                if handled:  # If mapping key was handled update next line
                    continue  # Continue to next line when mapping handled
                continue  # Continue to next line when no inline comment to record
            comment = line[hash_idx + 1 :].strip()  # Extract the comment text after '#'
            before = line[:hash_idx].rstrip()  # Portion before the inline comment
            
            mlist2 = re.match(r"^(\s*)-\s*(.*)$", before)  # Match list-item before '#' (inline-comment on list item)
            if mlist2:  # When inline comment is attached to a list item
                item = mlist2.group(2).strip()  # Extract the scalar portion after '-'
                path = tuple([k for k in stack if k])  # Build current key-path tuple from stack
                quoted = False  # Default: not quoted
                if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):  # Already quoted in original
                    quoted = True  # Mark as originally quoted
                quoted_items.setdefault(path, []).append(quoted)  # Record quoted flag for this list item index
            HandleInlineComment(before, comment, stack, comments)  # Process and record the inline comment for the key path
    return comments, quoted_items  # Return both comment mapping and quoted-items mapping keyed by tuple path


def insert_comments_in_yaml_block(block_text: str, comments_map: dict, quoted_items_map: dict) -> str:
    """
    Insert inline comments into a YAML text block using a mapping of key-path tuples to comment strings.

    :param block_text: YAML text produced by safe_dump for a single top-level section.
    :param comments_map: Mapping from tuple key-path (e.g., ("execution","verbose")) to comment string.
    :param quoted_items_map: Mapping from tuple key-path to list of booleans indicating whether each list item was originally quoted.
    :return: Annotated YAML block with inline comments inserted and standardized spacing.
    """

    out_lines: list[str] = []  # Accumulate modified lines
    stack: list[str] = []  # Track current key path by indentation level while iterating lines
    list_counters: dict = {}  # Track current index for list items per path

    for raw in block_text.splitlines():  # Process block line-by-line
        line = raw.rstrip()  # Remove trailing newline and spaces for clean processing
        m_key_val = re.match(r"^(\s*)([^:\n]+):\s*(.*)$", line)  # Match 'key:' or 'key: value' lines

        if m_key_val:  # Handle mapping lines where a key is present
            indent = len(m_key_val.group(1))  # Number of leading spaces
            level = indent // 2  # Compute nesting level using two-space indentation standard
            key = m_key_val.group(2).strip()  # Extract the key name

            if len(stack) <= level:
                while len(stack) <= level:
                    stack.append("")  # Extend stack to required level

            stack[level] = key  # Set key for current level in stack
            stack = stack[: level + 1]  # Truncate deeper entries
            path = tuple([k for k in stack if k])  # Build tuple path for comment lookup

            left = m_key_val.group(0).rstrip()  # Left portion including key and value
            comment = comments_map.get(path)  # Look up comment for this path

            if comment is not None:  # If a preserved comment exists for this key path
                annotated = f"{left}  # {comment}"  # Append two spaces then hash and comment text
                out_lines.append(annotated)  # Add annotated line to output
                continue  # Continue to next input line after adding annotated line

        # If this line is a list item, decide quoting based on original file's quoted-items info
        m_list = re.match(r"^(\s*)-\s+(.*)$", line)  # Match list items like '- value'
        if m_list:  # When current line is a list item
            path = tuple([k for k in stack if k])  # Parent key-path for this list
            idx = list_counters.get(path, 0)  # Index of this list item under its parent path
            orig_list = quoted_items_map.get(path)  # Original quoted flags list for this path
            if orig_list is not None and idx < len(orig_list) and orig_list[idx]:  # If original had a quoted entry at this index
                item = m_list.group(2).strip()  # Extract scalar after '-'
                if not ((item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'"))):  # If currently not quoted
                    line = f"{m_list.group(1)}- \"{item}\""  # Force double-quoting to preserve original style
            else:
                line = QuoteListItemIfNeeded(line)  # Otherwise apply heuristic quoting when useful
            list_counters[path] = idx + 1  # Increment index for next list item under this path

        out_lines.append(line)  # Append the possibly modified or original line to output

    return "\n".join(out_lines)  # Return the reconstructed block with injected comments


def write_configs(config_path: str, example_path: str) -> bool:
    """
    Write the sorted configuration into config.yaml and sync to config.yaml.example.

    :param config_path: Path to the destination config.yaml file.
    :param example_path: Path to the destination config.yaml.example file.
    :return: True when both files are written successfully, False otherwise.
    """

    general, python = load_and_prepare_sections(config_path)  # Load, sort, and separate configuration sections
    header_top = build_header_top()  # Build top-level header block
    general_header = build_general_header()  # Build general sections header block
    
    try:  # Attempt to write configuration files
        final_text = build_final_text(general, python)  # Build final configuration text
        config_file = Path(config_path)  # Convert config_path to Path for write
        
        with config_file.open("w", encoding="utf-8") as fh:  # Open config.yaml for writing
            fh.write(final_text)  # Write the final merged YAML text
        example_file = Path(example_path)  # Convert example_path to Path for write
        
        with example_file.open("w", encoding="utf-8") as fh2:  # Open config.yaml.example for writing
            fh2.write(final_text)  # Write identical YAML text to example file
        
        return True  # Indicate success when both writes complete
    except Exception as exc:  # Handle exceptions raised during file write operations
        print(f"Failed to write config files: {exc}")  # Log error message when write fails
        return False  # Indicate failure when exception occurs


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
        f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}YAML Configuration Sections Updater{BackgroundColors.GREEN} program!{Style.RESET_ALL}",
        end="\n\n",
    )  # Output the welcome message
    
    start_time = datetime.datetime.now()  # Get the start time of the program
    
    try:  # Attempt the configuration read, sort, and write operations
        repo_root = Path(__file__).resolve().parent.parent  # Determine repository root from this script location
        config_path = repo_root / "config.yaml"  # Build path to config.yaml in repository root
        example_path = repo_root / "config.yaml.example"  # Build path to config.yaml.example in repository root
        success = write_configs(str(config_path), str(example_path))  # Invoke writer to update both files
        if success:  # Verify writer returned success
            print(f"Config files updated: {config_path} and {example_path}")  # Inform user that files were updated
        else:  # Writer indicated failure
            print(f"Failed to update config files")  # Inform user that update failed
    except Exception as err:  # Catch unexpected exceptions during processing
        print(f"Error while processing config files: {err}")  # Print the raised exception message

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
