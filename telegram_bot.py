"""
================================================================================
<PROJECT OR SCRIPT TITLE>
================================================================================
Author      : Breno Farias da Silva
Created     : <YYYY-MM-DD>
Description :
   <Provide a concise and complete overview of what this script does.>
   <Mention its purpose, scope, and relevance to the larger project.>

   Key features include:
      - <Feature 1 — e.g., automatic data loading and preprocessing>
      - <Feature 2 — e.g., model training and evaluation>
      - <Feature 3 — e.g., visualization or report generation>
      - <Feature 4 — e.g., logging or notification system>
      - <Feature 5 — e.g., integration with other modules or datasets>

Usage:
   1. <Explain any configuration steps before running, such as editing variables or paths.>
   2. <Describe how to execute the script — typically via Makefile or Python.>
         $ make <target>   or   $ python <script_name>.py
   3. <List what outputs are expected or where results are saved.>

Outputs:
   - <Output file or directory 1 — e.g., results.csv>
   - <Output file or directory 2 — e.g., Feature_Analysis/plots/>
   - <Output file or directory 3 — e.g., logs/output.txt>

TODOs:
   - <Add a task or improvement — e.g., implement CLI argument parsing.>
   - <Add another improvement — e.g., extend support to Parquet files.>
   - <Add optimization — e.g., parallelize evaluation loop.>
   - <Add robustness — e.g., error handling or data validation.>

Dependencies:
   - Python >= <version>
   - <Library 1 — e.g., pandas>
   - <Library 2 — e.g., numpy>
   - <Library 3 — e.g., scikit-learn>
   - <Library 4 — e.g., matplotlib, seaborn, tqdm, colorama>

Assumptions & Notes:
   - <List any key assumptions — e.g., last column is the target variable.>
   - <Mention data format — e.g., CSV files only.>
   - <Mention platform or OS-specific notes — e.g., sound disabled on Windows.>
   - <Note on output structure or reusability.>
"""

import atexit # For playing a sound when the program finishes
import os # For running a command in the terminal
import platform # For getting the operating system name
from colorama import Style # For coloring the terminal

# Macros:
class BackgroundColors: # Colors for the terminal
   CYAN = "\033[96m" # Cyan
   GREEN = "\033[92m" # Green
   YELLOW = "\033[93m" # Yellow
   RED = "\033[91m" # Red
   BOLD = "\033[1m" # Bold
   UNDERLINE = "\033[4m" # Underline
   CLEAR_TERMINAL = "\033[H\033[J" # Clear the terminal

# Execution Constants:
VERBOSE = False # Set to True to output verbose messages

# Sound Constants:
SOUND_COMMANDS = {"Darwin": "afplay", "Linux": "aplay", "Windows": "start"} # The commands to play a sound for each operating system
SOUND_FILE = "./.assets/Sounds/NotificationSound.wav" # The path to the sound file

# RUN_FUNCTIONS:
RUN_FUNCTIONS = {
   "Play Sound": True, # Set to True to play a sound when the program finishes
}

# Functions Definitions:

def verbose_output(true_string="", false_string=""):
   """
   Outputs a message if the VERBOSE constant is set to True.

   :param true_string: The string to be outputted if the VERBOSE constant is set to True.
   :param false_string: The string to be outputted if the VERBOSE constant is set to False.
   :return: None
   """

   if VERBOSE and true_string != "": # If the VERBOSE constant is set to True and the true_string is set
      print(true_string) # Output the true statement string
   elif false_string != "": # If the false_string is set
      print(false_string) # Output the false statement string

def verify_filepath_exists(filepath):
   """
   Verify if a file or folder exists at the specified path.

   :param filepath: Path to the file or folder
   :return: True if the file or folder exists, False otherwise
   """

   verbose_output(f"{BackgroundColors.GREEN}Verifying if the file or folder exists at the path: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}") # Output the verbose message

   return os.path.exists(filepath) # Return True if the file or folder exists, False otherwise

def play_sound():
   """
   Plays a sound when the program finishes and skips if the operating system is Windows.

   :param: None
   :return: None
   """

   current_os = platform.system() # Get the current operating system
   if current_os == "Windows": # If the current operating system is Windows
      return # Do nothing

   if verify_filepath_exists(SOUND_FILE): # If the sound file exists
      if current_os in SOUND_COMMANDS: # If the platform.system() is in the SOUND_COMMANDS dictionary
         os.system(f"{SOUND_COMMANDS[current_os]} {SOUND_FILE}") # Play the sound
      else: # If the platform.system() is not in the SOUND_COMMANDS dictionary
         print(f"{BackgroundColors.RED}The {BackgroundColors.CYAN}{current_os}{BackgroundColors.RED} is not in the {BackgroundColors.CYAN}SOUND_COMMANDS dictionary{BackgroundColors.RED}. Please add it!{Style.RESET_ALL}")
   else: # If the sound file does not exist
      print(f"{BackgroundColors.RED}Sound file {BackgroundColors.CYAN}{SOUND_FILE}{BackgroundColors.RED} not found. Make sure the file exists.{Style.RESET_ALL}")

def main():
   """
   Main function.

   :param: None
   :return: None
   """

   print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Main Template Python{BackgroundColors.GREEN} program!{Style.RESET_ALL}", end="\n\n") # Output the welcome message

   print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}") # Output the end of the program message

   atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None # Register the play_sound function to be called when the program finishes

if __name__ == "__main__":
   """
   This is the standard boilerplate that calls the main() function.

   :return: None
   """

   main() # Call the main function
