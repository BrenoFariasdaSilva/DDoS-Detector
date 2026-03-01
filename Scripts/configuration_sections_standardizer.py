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


def get_python_filenames(base_dir: str, scripts_dir: str) -> list:
    """
    Get python filenames without extension.

    :param base_dir: Base directory path to search for .py files.
    :param scripts_dir: Scripts directory path to search for .py files.
    :return: List of python filenames without the .py extension.
    """

    base_path = Path(base_dir)  # Convert base_dir to Path for filesystem operations
    scripts_path = Path(scripts_dir)  # Convert scripts_dir to Path for filesystem operations
    py_names = set()  # Initialize a set to collect unique python filenames
    
    if base_path.exists() and base_path.is_dir():  # Verify base_path exists and is a directory
        for p in base_path.glob("*.py"):  # Iterate over Python files in base_path
            py_names.add(p.stem)  # Add the filename without extension to the set
    
    if scripts_path.exists() and scripts_path.is_dir():  # Verify scripts_path exists and is a directory
        for p in scripts_path.glob("*.py"):  # Iterate over Python files in scripts_path
            py_names.add(p.stem)  # Add the filename without extension to the set
    
    return sorted(py_names)  # Return an alphabetically sorted list of python filenames


def load_yaml(file_path: str, unused: object) -> dict:
    """
    Load a YAML file and return its content as a dictionary.

    :param file_path: Path to the YAML file to load.
    :param unused: Placeholder parameter to preserve two-argument signature.
    :return: Parsed YAML content as a dictionary (empty dict if file missing).
    """

    path = Path(file_path)  # Convert file_path to Path for operations
    
    if not path.exists():  # Verify the file exists before reading
        return {}  # Return empty dict when file is not present
    
    with path.open("r", encoding="utf-8") as fh:  # Open the YAML file for reading
        data = yaml.safe_load(fh) or {}  # Load YAML safely and default to empty dict
        
    return data  # Return the parsed YAML structure


def sort_recursive(obj: object, unused: object) -> object:
    """
    Recursively sort dictionary keys alphabetically and preserve nested structure.

    :param obj: The object to sort (dict, list, or scalar).
    :param unused: Placeholder parameter to preserve two-argument signature.
    :return: A new object with dict keys sorted recursively.
    """

    if isinstance(obj, dict):  # Verify the object is a dictionary
        ordered = OrderedDict()  # Create an ordered mapping for deterministic key order
        
        for key in sorted(obj.keys()):  # Iterate over sorted keys alphabetically
            ordered[key] = sort_recursive(obj[key], None)  # Recursively sort nested values
        return ordered  # Return the ordered mapping
    
    if isinstance(obj, list):  # Verify the object is a list
        return [sort_recursive(item, None) for item in obj]  # Recursively sort each list item
    
    return obj  # Return scalars unchanged


def separate_sections(config_dict: object, py_list: list) -> tuple:
    """
    Separate top-level YAML sections into general and python-specific groups.

    :param config_dict: The parsed YAML top-level mapping as an object.
    :param py_list: The list of python filenames (without .py) for matching.
    :return: Tuple containing (general_sections_dict, python_sections_dict).
    """

    py_set = set(py_list)  # Convert list of python names to a set for fast membership checks
    general = OrderedDict()  # Initialize ordered dict for general sections
    python = OrderedDict()  # Initialize ordered dict for python-specific sections
    if not isinstance(config_dict, dict):  # Verify config_dict is a mapping
        try:
            items_attr = getattr(config_dict, "items", None)  # Retrieve the items attribute if present
            if callable(items_attr):  # Verify items attribute is callable to iterate mapping pairs
                try:
                    pairs_called = items_attr()  # Call items() to obtain key-value pairs
                except Exception:
                    pairs_called = []  # Fallback to empty when call fails
                try:
                    if isinstance(pairs_called, (list, tuple, set)):  # Verify common iterable containers
                        iter_pairs = list(pairs_called)  # Convert container to list for iteration
                    elif hasattr(pairs_called, "__iter__"):  # Verify object implements iteration protocol
                        iter_pairs = list(cast(Iterable, pairs_called))  # Cast to Iterable and convert to list
                    else:
                        iter_pairs = []  # Fallback to empty list when not iterable
                except Exception:
                    iter_pairs = []  # Fallback to empty list when conversion to list fails
                try:
                    config_dict = dict(iter_pairs)  # Build dict from iterable pairs
                except Exception:
                    config_dict = {}  # Fallback to empty dict when dict construction fails
            else:
                config_dict = {}  # Fallback to empty dict when no items() available
        except Exception:
            config_dict = {}  # Fallback to empty dict when coercion fails
    for key in sorted(config_dict.keys()):  # Iterate top-level keys in alphabetical order
        if key in py_set:  # Verify if the key matches a python filename exactly
            python[key] = config_dict[key]  # Assign to python-specific mapping when matched
        else:
            general[key] = config_dict[key]  # Otherwise assign to general mapping
    return general, python  # Return the two grouped ordered mappings


def load_and_prepare_sections(config_path: str) -> Tuple[Dict, Dict]:
    """
    Load YAML configuration, sort it recursively, and separate sections.

    :param config_path: Path to the destination config.yaml file.
    :return: Tuple containing general and python-specific sections.
    """

    repo_root = Path(config_path).parent  # Determine repository root from config_path
    raw = load_yaml(str(repo_root / "config.yaml"), None)  # Load original YAML content
    sorted_raw = sort_recursive(raw, None)  # Recursively sort keys in the loaded YAML
    py_names = get_python_filenames(str(repo_root), str(repo_root / "Scripts"))  # Discover python filenames
    general, python = separate_sections(sorted_raw, py_names)  # Separate sections into groups and coerce types
    
    return general, python  # Return separated configuration sections


def build_header_top() -> str:
    """
    Build the top-level configuration header block.

    :param: No parameters.
    :return: Header string for top of configuration file.
    """

    header_top = (
        "# ================================================================================\n"
        "# DDoS-Detector — Unified Configuration File\n"
        "# ================================================================================\n\n"
    )  # Construct top-level header block
    
    return header_top  # Return top-level header string


def build_general_header() -> str:
    """
    Build the general sections header block.

    :param: No parameters.
    :return: Header string for general sections.
    """

    general_header = (
        "# ==============================================================================\n"
        "# General Sections — Used by multiple scripts\n"
        "# ==============================================================================\n\n"
    )  # Construct general sections header block
    return general_header  # Return general sections header string


def build_python_header(sec: str) -> str:
    """
    Build the python-specific section header block.

    :param sec: Section name.
    :return: Header string for a python-specific section.
    """

    py_header = (
        "# ==============================================================================\n"
        f"# {sec.upper()} — Used by {sec}.py\n"
        "# ==============================================================================\n\n"
    )  # Construct python-specific header block
    
    return py_header  # Return python-specific header string


def convert_ordered_dicts(obj: Any) -> Any:
    """
    Recursively convert OrderedDict instances to plain dicts.

    :param obj: Object to convert (can be OrderedDict, dict, list, or scalar).
    :return: Converted object with OrderedDict replaced by dict (or original scalar).
    """

    if isinstance(obj, OrderedDict):  # Verify the object is an OrderedDict
        new = {}  # Create a plain dict to preserve insertion order
        for k, v in obj.items():  # Iterate over mapping items in order
            new[k] = convert_ordered_dicts(v)  # Recurse to convert nested structures
        return new  # Return the converted plain dict
    
    if isinstance(obj, dict):  # Verify the object is a dict (non-OrderedDict)
        new = {}  # Create a new dict for converted contents
        for k, v in obj.items():  # Iterate over mapping items
            new[k] = convert_ordered_dicts(v)  # Recurse to convert nested structures
        return new  # Return the converted dict
    
    if isinstance(obj, list):  # Verify the object is a list container
        converted_list = [convert_ordered_dicts(i) for i in obj]  # Recurse convert each list element
        return converted_list  # Return the converted list
    
    return obj  # Return scalar or unsupported type unchanged


def build_final_text(general: Dict, python: Dict) -> str:
    """
    Build final YAML configuration text with headers and sections.

    :param general: Dictionary of general sections.
    :param python: Dictionary of python-specific sections.
    :return: Final formatted YAML configuration text.
    """

    try:  # Attempt to extract inline comments and quoted-items info from repository config.yaml
        repo_config_path = Path(__file__).resolve().parent.parent / "config.yaml"  # Compute path to repo config.yaml
        comments_map, quoted_items_map = extract_inline_comments(str(repo_config_path), None)  # Extract inline comments and quote flags keyed by tuple path
    except Exception:  # If extraction fails, fallback to empty mappings
        comments_map = {}  # Use empty comments map when extraction fails
        quoted_items_map = {}  # Use empty quoted-items map when extraction fails

    out_lines = []  # Initialize list to accumulate output text parts
    out_lines.append(build_header_top())  # Append top-level header block
    
    if len(general) > 0:  # Verify presence of general sections before writing header
        out_lines.append(build_general_header())  # Append general sections header when present
        for sec in general.keys():  # Iterate sorted general section names
            conv = convert_ordered_dicts(general[sec])  # Convert OrderedDicts to plain dicts recursively
            raw_block = yaml.safe_dump({sec: conv}, sort_keys=False, default_flow_style=False)  # Dump the section mapping to YAML text
            annotated_block = insert_comments_in_yaml_block(raw_block, comments_map, quoted_items_map)  # Inject preserved inline comments into dumped YAML block
            out_lines.append(annotated_block + "\n")  # Append the annotated YAML block and one blank line after the section
    
    for sec in python.keys():  # Iterate python-specific section names in sorted order
        out_lines.append(build_python_header(sec))  # Append python-specific header before section
        conv = convert_ordered_dicts(python[sec])  # Convert OrderedDicts to plain dicts recursively
        raw_block = yaml.safe_dump({sec: conv}, sort_keys=False, default_flow_style=False)  # Dump the python-specific section to YAML text
        annotated_block = insert_comments_in_yaml_block(raw_block, comments_map, quoted_items_map)  # Inject preserved inline comments into dumped YAML block
        out_lines.append(annotated_block + "\n")  # Append the annotated YAML block and one blank line after the section
    
    final_text = "".join(out_lines)  # Join all parts into a single string
    
    return final_text  # Return fully constructed YAML text


def EnsureStackLength(stack: list[str], level: int) -> None:
    """
    Ensure the stack has a slot for the given level.

    :param stack: The stack of keys indexed by indentation level.
    :param level: The indentation level that must be addressable.
    :return: None.
    """

    while len(stack) <= level:  # Verify stack length and loop until level is addressable
        stack.append("")  # Append placeholder entry so the stack is indexable at level


def UpdateStackForKey(stack: list[str], level: int, key: str) -> None:
    """
    Update the stack with a key at the specified level and truncate deeper levels.

    :param stack: The stack of keys indexed by indentation level.
    :param level: The indentation level where the key belongs.
    :param key: The mapping key to record at the specified level.
    :return: None.
    """

    EnsureStackLength(stack, level)  # Ensure the stack has room for the specified level
    stack[level] = key  # Record the provided key at the computed level in the stack
    del stack[level + 1 :]  # Truncate any deeper levels beyond the current one


def QuoteListItemIfNeeded(line: str) -> str:
    """
    Quote YAML list item scalars when they represent filesystem paths or contain spaces.

    :param line: The YAML line to inspect (may be a list item like '- ./path').
    :return: The possibly-updated YAML line with the scalar quoted when necessary.
    """

    m = re.match(r"^(\s*)-\s+(.*)$", line)  # Match indented list item lines like '- value'
    
    if not m:  # If not a list item, return original line unchanged
        return line  # No change required when line is not a list entry
    
    indent = m.group(1)  # Leading indentation for the list item
    item = m.group(2).strip()  # Extract the scalar portion after '-'
    
    if not item:  # Empty item requires no quoting
        return line  # Return original when item is empty
    
    if (item.startswith('"') and item.endswith('"')) or (item.startswith("'") and item.endswith("'")):  # Already quoted
        return line  # Preserve already-quoted scalars as-is
    
    needs_quote = False  # Default: do not quote
    
    if item.startswith('./') or item.startswith('.\\'):  # Filesystem relative path patterns
        needs_quote = True  # Quote relative filesystem paths to preserve leading ./ or .\
    
    if ' ' in item:  # Items containing spaces need quoting
        needs_quote = True  # Quote scalars that include whitespace
    
    if item.endswith('/') and item != '/':  # Trailing slash likely a directory path
        needs_quote = True  # Quote directory-like scalars to preserve trailing slash
    
    if needs_quote:  # When quoting is required
        quoted = f'"{item}"'  # Build double-quoted scalar preserving content
        return f"{indent}- {quoted}"  # Return reconstructed list line with quotes
    
    return line  # Return original line when quoting not necessary


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
