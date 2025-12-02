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
import datetime # For getting the current date and time
import os # For running a command in the terminal
import platform # For getting the operating system name
from colorama import Style # For coloring the terminal
from telegram_bot import TelegramBot # For Telegram notifications

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
      
def get_files_to_process(directory_path, file_extension=".csv"):
   """
   Get all of the specified files in a directory (non-recursive).
   
   :param directory_path: Path to the directory to search
   :param file_extension: File extension to filter (default: .csv)
   :return: List of files with the specified extension
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Getting all {BackgroundColors.CYAN}{file_extension}{BackgroundColors.GREEN} files in the directory: {BackgroundColors.CYAN}{directory_path}{Style.RESET_ALL}") # Output the verbose message

   verify_filepath_exists(directory_path) # Verify if the directory exists

   if not os.path.isdir(directory_path): # If the path is not a directory
      verbose_output(f"{BackgroundColors.RED}The specified path is not a directory: {BackgroundColors.CYAN}{directory_path}{Style.RESET_ALL}") # Output the verbose message
      return [] # Return an empty list

   files = [] # List to store the files

   for item in os.listdir(directory_path): # List all items in the directory
      item_path = os.path.join(directory_path, item) # Get the full path of the item
      
      if os.path.isfile(item_path) and item.lower().endswith(file_extension): # If the item is a file and has the specified extension
         files.append(item_path) # Add the file to the list

   return sorted(files) # Return sorted list for consistency

def get_dataset_name(input_path):
   """
   Extract the dataset name from CSVs path.

   :param input_path: Path to the CSVs files
   :return: Dataset name
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Extracting dataset name from CSV path: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}") # Output the verbose message
   
   datasets_pos = input_path.find("/Datasets/") # Find the position of "/Datasets/" in the path
   if datasets_pos != -1: # If "/Datasets/" is found in the path
      after_datasets = input_path[datasets_pos + len("/Datasets/"):] # Get the substring after "/Datasets/"
      next_slash = after_datasets.find("/") # Find the next "/"
      if next_slash != -1: # If there is another "/"
         dataset_name = after_datasets[:next_slash] # Take until the next "/"
      else: # If there is no other "/"
         dataset_name = after_datasets.split("/")[0] if "/" in after_datasets else after_datasets # No more "/", take the first part if any
   else: # If "/Datasets/" is not found in the path
      dataset_name = os.path.basename(input_path) # Fallback to basename if "Datasets" not in path

   return dataset_name # Return the dataset name

def calculate_execution_time(start_time, finish_time):
   """
   Calculates the execution time between start and finish times and formats it as hh:mm:ss.

   :param start_time: The start datetime object
   :param finish_time: The finish datetime object
   :return: String formatted as hh:mm:ss representing the execution time
   """

   delta = finish_time - start_time # Calculate the time difference
   hours, remainder = divmod(delta.seconds, 3600) # Calculate the hours, minutes and seconds
   minutes, seconds = divmod(remainder, 60) # Calculate the minutes and seconds
   return f"{hours:02d}:{minutes:02d}:{seconds:02d}" # Format the execution time

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

   print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Classifiers Stacking{BackgroundColors.GREEN} program!{Style.RESET_ALL}\n") # Output the welcome message 
   start_time = datetime.datetime.now() # Get the start time of the program
   
   input_path = "./Datasets/CICDDoS2019/01-12/" # Path to the input dataset directory
   files_to_process = get_files_to_process(input_path, file_extension=".csv") # Get list of CSV files to process
   files_to_process = ["./Datasets/CICDDoS2019/01-12/DrDoS_DNS.csv"] # For testing purposes, process only this file
   
   dataset_name = get_dataset_name(input_path) # Get the dataset name from the input path
   
   bot = TelegramBot() # Initialize Telegram bot for notifications
   
   # Your code goes here

   finish_time = datetime.datetime.now() # Get the finish time of the program
   print(f"{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}") # Output the start and finish times
   print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}") # Output the end of the program message

   atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None # Register the play_sound function to be called when the program finishes

if __name__ == "__main__":
   """
   This is the standard boilerplate that calls the main() function.

   :return: None
   """

   main() # Call the main function
