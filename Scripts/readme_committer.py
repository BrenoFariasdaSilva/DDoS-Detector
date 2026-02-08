"""
================================================================================
Auto-Committer for README Sections
================================================================================
Author      : Breno Farias da Silva
Created     : 2026-02-07
Description :
    Automates staged commits for sections between two markers inside a README file.
    This script reads a target README file, extracts all sections between specified
    start and end section names (marked with level 2 headers), removes them temporarily,
    and then re-adds them one-by-one in reverse order (bottom to top), creating a Git
    commit for each section.

    Key features include:
        - Reads and parses README files to extract level 2 header sections (## Section Name)
        - Identifies sections between configurable start and end markers
        - Removes all target sections and re-adds them incrementally
        - Automates Git add, commit, and optionally push operations per section
        - Standardizes section separators for consistent documentation formatting

Usage:
    1. Configure the constants in the Configuration Constants section (FILE_PATH, START_SECTION, END_SECTION, COMMIT_PREFIX).
    2. IMPORTANT: Make a backup or work on a test branch first.
    3. Execute the script via Makefile or Python:
            $ make run   or   $ python readme_committer.py
    4. The script will create one Git commit per section with descriptive messages.

Outputs:
    - Modified README file with reformatted section separators
    - Individual Git commits for each section between the markers
    - Execution log in ./Logs/readme_committer.log

TODOs:
    - Add CLI argument parsing for dynamic configuration
    - Implement support for different header levels (###, ####, etc.)
    - Add dry-run mode to preview changes without committing
    - Extend to support multiple file processing in batch

Dependencies:
    - Python >= 3.7
    - re (built-in)
    - subprocess (built-in)
    - pathlib (built-in)
    - colorama
    - datetime (built-in)
    - os (built-in)
    - platform (built-in)
    - sys (built-in)

Assumptions & Notes:
    - Assumes sections are marked with ## (level 2 headers)
    - Requires a working Git repository with proper configuration
    - Sections are identified using regex pattern matching
    - Separator between sections is standardized to 2 newlines (1 empty line)
    - Sound notification disabled on Windows platform
"""

import atexit  # For playing a sound when the program finishes
import datetime  # For getting the current date and time
import os  # For running a command in the terminal
import platform  # For getting the operating system name
import re  # For regular expression pattern matching
import subprocess  # For running Git commands
import sys  # For system-specific parameters and functions
import time  # For sleeping between commits
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
FILE_PATH = Path("./README.md")  # Path to the target README file
START_SECTION = ""  # Name of the first section to include
END_SECTION = ""  # Name of the last section to include
COMMIT_PREFIX = "DOCS: Adding the"  # Prefix for Git commit messages
SECTION_SEPARATOR = "\n\n"  # Standardized separator between sections (2 newlines -> 1 empty line)

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


def extract_section_names(text):
    """
    Extracts all level 2 section names from the README text.

    :param text: The full text content of the README file
    :return: List of section names found in the file
    """

    pattern = r"^##\s+([^\n]+)"  # Regex pattern to match level 2 headers
    found_names = [m.group(1).strip() for m in re.finditer(pattern, text, flags=re.MULTILINE)]  # Extract all section names
    
    return found_names  # Return the list of section names


def verify_markers_exist(start_name, end_name, found_names):
    """
    Verifies if start and end markers exist in the list of found section names.

    :param start_name: The name of the first section to include
    :param end_name: The name of the last section to include
    :param found_names: List of section names found in the file
    :return: List of validation problems, empty if no problems found
    """

    problems = []  # List to collect validation problems
    
    if not start_name or not end_name:  # If either marker is not set
        problems.append("Both START_SECTION and END_SECTION must be set to section names.")  # Add problem if markers are not set

    if start_name and start_name not in found_names:  # If START_SECTION is set but not found in the file
        problems.append(f"START_SECTION '{start_name}' not found in {FILE_PATH.name}.")  # Add problem if START_SECTION is not found
    
    if end_name and end_name not in found_names:  # If END_SECTION is set but not found in the file
        problems.append(f"END_SECTION '{end_name}' not found in {FILE_PATH.name}.")  # Add problem if END_SECTION is not found
    
    return problems  # Return the list of problems


def display_validation_errors(problems, found_names):
    """
    Displays validation errors and available sections to help the user.

    :param problems: List of validation problem messages
    :param found_names: List of section names found in the file
    :return: None
    """

    print(f"{BackgroundColors.RED}Validation error with START/END section markers:{Style.RESET_ALL}")
    
    for p in problems:  # Print each problem in the validation
        print(f"{BackgroundColors.YELLOW}- {p}{Style.RESET_ALL}")

    if found_names:  # If there are any sections found, list them to help the user
        print(f"{BackgroundColors.GREEN}Available level-2 sections in the file:{BackgroundColors.CYAN} {', '.join(found_names)}{Style.RESET_ALL}")
    else:  # If no sections were found, inform the user
        print(f"{BackgroundColors.YELLOW}No level-2 sections were detected in {FILE_PATH.name}.{Style.RESET_ALL}")

    print(f"{BackgroundColors.GREEN}Please set `START_SECTION` and `END_SECTION` to valid section names before running the script.{Style.RESET_ALL}")


def validate_markers(start_name, end_name, text):
    """
    Validate that START_SECTION and END_SECTION are provided and exist in the file.

    :param start_name: The name of the first section to include
    :param end_name: The name of the last section to include
    :param text: The full text content of the README file
    :return: True if validation passes, False otherwise
    """
    
    verbose_output(f"{BackgroundColors.GREEN}Validating START_SECTION and END_SECTION markers...{Style.RESET_ALL}")

    found_names = extract_section_names(text)  # Extract all level-2 section names from the file
    problems = verify_markers_exist(start_name, end_name, found_names)  # Verify if the markers exist in the found names

    if problems:  # If there are any validation problems, display them and return False
        display_validation_errors(problems, found_names)  # Display validation errors and available sections
        return False  # Validation failed

    return True  # Validation passed


def run_git_commit(section_name: str):
    """
    Executes Git add and commit commands for the target file.

    :param section_name: The name of the section being committed
    :return: None
    """

    commit_msg = f"{COMMIT_PREFIX} {section_name} section to {FILE_PATH.name}"  # Create the commit message

    verbose_output(f"{BackgroundColors.GREEN}Running Git add for: {BackgroundColors.CYAN}{FILE_PATH}{Style.RESET_ALL}")

    absolute_file_path = FILE_PATH.resolve()  # Resolve FILE_PATH to an absolute path
    git_dir = absolute_file_path.parent  # Get the directory containing the file

    subprocess.run(["git", "-C", str(git_dir), "add", str(FILE_PATH)], check=True)  # Stage the file

    verbose_output(f"{BackgroundColors.GREEN}Running Git commit with message: {BackgroundColors.CYAN}{commit_msg}{Style.RESET_ALL}")

    subprocess.run(["git", "-C", str(git_dir), "commit", "-m", commit_msg], check=True)  # Commit the changes


def extract_sections_between(text, start_name, end_name):
    """
    Extracts all level 2 header sections between two specified section names.

    :param text: The full text content of the README file
    :param start_name: The name of the first section to include
    :param end_name: The name of the last section to include
    :return: Tuple of (prefix_text, suffix_text, list_of_sections)
             Each section in the list is a tuple: (name, content, start_pos, end_pos)
    """

    verbose_output(f"{BackgroundColors.GREEN}Extracting sections between {BackgroundColors.CYAN}{start_name}{BackgroundColors.GREEN} and {BackgroundColors.CYAN}{end_name}{Style.RESET_ALL}")
    
    pattern = r"^##\s+([^\n]+).*?(?=^##\s|\Z)"  # Regex pattern for level 2 headers
    matches = list(re.finditer(pattern, text, flags=re.DOTALL | re.MULTILINE))  # Find all matches
    sections = [(m.group(1).strip(), m.group(0), m.start(), m.end()) for m in matches]  # Extract section details
    start_idx = next(i for i, s in enumerate(sections) if s[0] == start_name)  # Find start section index
    end_idx = next(i for i, s in enumerate(sections) if s[0] == end_name)  # Find end section index
    selected = sections[start_idx:end_idx + 1]  # Get all sections in range (inclusive)
    prefix = text[:selected[0][2]]  # Text before the first selected section
    suffix = text[selected[-1][3]:]  # Text after the last selected section
    
    verbose_output(f"{BackgroundColors.GREEN}Found {BackgroundColors.CYAN}{len(selected)}{BackgroundColors.GREEN} sections between markers{Style.RESET_ALL}")
    
    return prefix, suffix, selected  # Return the components


def write_file(path, content):
    """
    Writes content to a file with UTF-8 encoding and ensures it ends with exactly one empty line.

    :param path: Path object pointing to the target file
    :param content: String content to write to the file
    :return: None
    """

    verbose_output(f"{BackgroundColors.GREEN}Writing content to file: {BackgroundColors.CYAN}{path}{Style.RESET_ALL}")
    
    content = content.rstrip() + "\n"  # Remove trailing whitespace and add exactly one newline
    
    path.write_text(content, encoding="utf-8")  # Write the content to the file


def extract_subsections(section_content):
    """
    Extracts all level 3 subsections (###) from a section's content.

    :param section_content: The full content of a level 2 section
    :return: List of tuples (subsection_name, subsection_content) or empty list if no subsections
    """

    verbose_output(f"{BackgroundColors.GREEN}Verifying for subsections within section...{Style.RESET_ALL}")
    
    pattern = r"^###\s+([^\n]+).*?(?=^###\s|\Z)"  # Regex pattern for level 3 headers
    matches = list(re.finditer(pattern, section_content, flags=re.DOTALL | re.MULTILINE))  # Find all subsection matches
    
    if not matches:  # If no subsections found
        verbose_output(f"{BackgroundColors.YELLOW}No subsections found in this section.{Style.RESET_ALL}")
        return []  # Return empty list
    
    subsections = [(m.group(1).strip(), m.group(0)) for m in matches]  # Extract subsection details (name, content)
    
    verbose_output(f"{BackgroundColors.GREEN}Found {BackgroundColors.CYAN}{len(subsections)}{BackgroundColors.GREEN} subsections{Style.RESET_ALL}")
    
    return subsections  # Return the list of subsections


def extract_section_intro(section_body):
    """
    Extracts the introductory text from a section body before the first subsection.

    :param section_body: The body content of the section without the header
    :return: The introductory text, or empty string if no intro found
    """

    first_subsection_pos = section_body.find("###")  # Find the position of the first subsection marker
    
    if first_subsection_pos > 0:  # If there is content before the first subsection
        return section_body[:first_subsection_pos].rstrip("\n")  # Extract and return the introductory text with trailing newlines removed
    
    return ""  # Return empty string if no introductory text exists


def build_base_section_content(section_header, section_intro):
    """
    Builds the base section content including header and introductory text.

    :param section_header: The section header (e.g., "## Section Name")
    :param section_intro: The introductory text before subsections
    :return: String containing the base section content
    """

    base_content = section_header  # Start with the section header
    
    if section_intro:  # If there is introductory text
        base_content += "\n\n" + section_intro  # Add the intro with proper spacing
    
    return base_content  # Return the constructed base content


def build_section_with_subsections(base_content, subsections, end_index):
    """
    Builds section content with subsections up to the specified index.

    :param base_content: The base section content (header + intro)
    :param subsections: List of tuples containing (subsection_name, subsection_content)
    :param end_index: The index up to which subsections should be included (inclusive)
    :return: String containing the section with all subsections up to end_index
    """

    section_content = base_content  # Start with the base section content
    
    for i in range(end_index):  # Iterate through subsections up to the specified index
        sub_content = subsections[i][1].rstrip("\n")  # Get subsection content and remove trailing blank lines
        section_content += "\n\n" + sub_content  # Add the subsection with proper spacing
    
    return section_content  # Return the constructed section content


def execute_git_commit_for_subsection(subsection_name):
    """
    Executes Git add and commit commands for a subsection.

    :param subsection_name: The name of the subsection being committed
    :return: None
    """

    commit_msg = f"{COMMIT_PREFIX} {subsection_name} subsection to {FILE_PATH.name}"  # Create the commit message
    
    verbose_output(f"{BackgroundColors.GREEN}Running Git add for: {BackgroundColors.CYAN}{FILE_PATH}{Style.RESET_ALL}")
    
    absolute_file_path = FILE_PATH.resolve()  # Resolve FILE_PATH to an absolute path
    git_dir = absolute_file_path.parent  # Get the directory containing the file
    
    subprocess.run(["git", "-C", str(git_dir), "add", str(FILE_PATH)], check=True)  # Stage the file
    
    verbose_output(f"{BackgroundColors.GREEN}Running Git commit with message: {BackgroundColors.CYAN}{commit_msg}{Style.RESET_ALL}")
    
    subprocess.run(["git", "-C", str(git_dir), "commit", "-m", commit_msg], check=True)  # Commit the changes


def commit_section_with_subsections(name, section_header, section_body, subsections, prefix, suffix, current_body, commit_count):
    """
    Commits a section that contains subsections by committing each subsection separately.

    This function processes sections with level 3 headers (###) by committing each subsection
    individually in order (top to bottom). The section header and any introductory text
    before the first subsection are preserved with each subsection commit.

    :param name: The name of the parent section
    :param section_header: The reconstructed section header (e.g., "## Section Name")
    :param section_body: The body content of the section without the header
    :param subsections: List of tuples containing (subsection_name, subsection_content)
    :param prefix: Text before all selected sections in the document
    :param suffix: Text after all selected sections in the document
    :param current_body: The current accumulated body content being built
    :param commit_count: Current commit counter value
    :return: Tuple of (updated_current_body, updated_commit_count)
    """

    print(f"{BackgroundColors.YELLOW}Section '{BackgroundColors.CYAN}{name}{BackgroundColors.YELLOW}' contains {BackgroundColors.CYAN}{len(subsections)}{BackgroundColors.YELLOW} subsections. Committing each separately...{Style.RESET_ALL}")

    section_intro = extract_section_intro(section_body)  # Extract introductory text before the first subsection
    base_section_content = build_base_section_content(section_header, section_intro)  # Build the base section content with header and intro
    previous_sections_body = current_body  # Store the body content of all previously committed sections

    for subsection_index, (subsection_name, subsection_content) in enumerate(subsections, start=1):  # Process each subsection in order (top to bottom)
        current_section_content = build_section_with_subsections(base_section_content, subsections, subsection_index)  # Build section with all subsections up to current index
        current_body = previous_sections_body + current_section_content + SECTION_SEPARATOR if previous_sections_body else current_section_content + SECTION_SEPARATOR  # Combine previous sections with current section
        
        new_content = prefix + current_body + suffix  # Combine prefix, current body, and suffix
        write_file(FILE_PATH, new_content)  # Write the updated content to the file

        commit_count += 1  # Increment the commit counter
        verbose_output(f"{BackgroundColors.BOLD}[{BackgroundColors.YELLOW}{commit_count}{BackgroundColors.BOLD}]{Style.RESET_ALL} {BackgroundColors.GREEN}Committing subsection: {BackgroundColors.CYAN}{subsection_name}{BackgroundColors.GREEN} (from section {BackgroundColors.CYAN}{name}{BackgroundColors.GREEN}){Style.RESET_ALL}")

        execute_git_commit_for_subsection(subsection_name)  # Execute Git add and commit for this subsection

    return current_body, commit_count  # Return the updated body and commit count


def commit_whole_section(name, content, prefix, suffix, current_body, commit_count):
    """
    Commits an entire section as a single commit when it contains no subsections.

    This function handles sections that do not have level 3 headers (###) by committing
    the entire section content at once. This is the simpler case where no subsection
    splitting is needed.

    :param name: The name of the section being committed
    :param content: The full content of the section including header and body
    :param prefix: Text before all selected sections in the document
    :param suffix: Text after all selected sections in the document
    :param current_body: The current accumulated body content being built
    :param commit_count: Current commit counter value
    :return: Tuple of (updated_current_body, updated_commit_count)
    """

    content = content.rstrip("\n")  # Remove trailing blank lines from the section content

    current_body = current_body + content + SECTION_SEPARATOR if current_body else content + SECTION_SEPARATOR  # Add the section to the document body

    new_content = prefix + current_body + suffix  # Combine prefix, current body, and suffix
    write_file(FILE_PATH, new_content)  # Write the updated content to the file

    commit_count += 1  # Increment the commit counter
    verbose_output(f"{BackgroundColors.BOLD}[{BackgroundColors.YELLOW}{commit_count}{BackgroundColors.BOLD}]{Style.RESET_ALL} {BackgroundColors.GREEN}Committing section: {BackgroundColors.CYAN}{name}{Style.RESET_ALL}")

    run_git_commit(name)  # Commit the section

    return current_body, commit_count  # Return the updated body and commit count


def process_all_sections(sections, prefix, suffix):
    """
    Processes all sections and commits them incrementally with their subsections.

    :param sections: List of section tuples (name, content, start_pos, end_pos)
    :param prefix: Text before all selected sections in the document
    :param suffix: Text after all selected sections in the document
    :return: Total number of commits made
    """

    current_body = ""  # Initialize the current body content
    total_sections = len(sections)  # Get the total number of sections
    commit_count = 0  # Initialize commit counter

    verbose_output(f"{BackgroundColors.GREEN}Processing {BackgroundColors.CYAN}{total_sections}{BackgroundColors.GREEN} sections...{Style.RESET_ALL}")
    
    for section_index, (name, content, *_) in enumerate(sections, start=1):  # Process each section in order (top to bottom)
        section_header = f"## {name}"  # Construct the section header
        section_body = content[len(section_header):].strip("\n")  # Extract the section body by removing the header

        subsections = extract_subsections(section_body)  # Verify if the section contains level 3 subsections (###)

        if subsections:  # If subsections are present, commit each subsection separately
            current_body, commit_count = commit_section_with_subsections(
                name, section_header, section_body, subsections,
                prefix, suffix, current_body, commit_count
            )
        else:  # If no subsections are present, commit the entire section as one unit
            current_body, commit_count = commit_whole_section(
                name, content, prefix, suffix, current_body, commit_count
            )
            
        time.sleep(3)  # Sleep for a short time between commits to ensure proper Git history
    
    return commit_count  # Return the total number of commits made


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

    print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Auto-Committer for README Sections{BackgroundColors.GREEN} program!{Style.RESET_ALL}", end="\n\n")  # Output the welcome message
    
    start_time = datetime.datetime.now()  # Get the start time of the program
    
    if not verify_filepath_exists(FILE_PATH):  # If the file does not exist
        print(f"{BackgroundColors.RED}Error: Target file {BackgroundColors.CYAN}{FILE_PATH}{BackgroundColors.RED} not found!{Style.RESET_ALL}")  # Output error message
        return  # Exit the function

    original_text = FILE_PATH.read_text(encoding="utf-8")  # Read the original file content

    if not validate_markers(START_SECTION, END_SECTION, original_text):  # If the START_SECTION and END_SECTION markers are not valid
        return  # Exit the function
    
    print(f"{BackgroundColors.GREEN}Extracting sections between {BackgroundColors.CYAN}{START_SECTION}{BackgroundColors.GREEN} and {BackgroundColors.CYAN}{END_SECTION}{Style.RESET_ALL}")  # Output extraction message
    
    prefix, suffix, sections = extract_sections_between(original_text, START_SECTION, END_SECTION)  # Extract the sections between markers
    
    verbose_output(f"{BackgroundColors.YELLOW}Removing {BackgroundColors.CYAN}{len(sections)}{BackgroundColors.YELLOW} sections from the file...{Style.RESET_ALL}")  # Output removal message
    
    write_file(FILE_PATH, prefix + suffix)  # Write the file without the target sections
    
    verbose_output(f"{BackgroundColors.GREEN}Sections removed. Starting staged commits...{Style.RESET_ALL}")  # Output staged commits start message
    
    commit_count = process_all_sections(sections, prefix, suffix)  # Process all sections and commit them incrementally
        
    print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}All sections and subsections committed successfully! Total commits: {BackgroundColors.CYAN}{commit_count}{Style.RESET_ALL}", end="\n\n")  # Output success message with commit count

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
