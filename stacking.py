"""
================================================================================
Classifiers Stacking
================================================================================
Author      : Breno Farias da Silva
Created     : <YYYY-MM-DD>
Description :
   This script is designed to load and process network traffic data (DDoS datasets)
   for subsequent analysis using machine learning classifiers, focusing on ensemble
   methods like stacking. It handles file processing, extracts dataset names, and
   is intended to integrate feature selection results from a Genetic Algorithm.

   Key features include:
      - Automatic data loading and preprocessing (planned)
      - Integration of Genetic Algorithm (GA) feature selection results.
      - Utilities for path validation and file listing.
      - Execution time calculation and sound notification upon completion.
      - Telegram bot integration for notifications.

Usage:
   1. Ensure the dataset path and feature analysis files are correctly structured.
   2. Execute the script:
         $ python <script_name>.py
   3. Check terminal output for processing logs and execution time.

Outputs:
   - Processed data files (planned)
   - Log messages to the terminal.
   - Telegram notifications (if configured).

TODOs:
   - Implement and call function to extract the genetic algorithm features.
   - Implement data loading and preprocessing logic.
   - Implement classifier training and evaluation (Stacking, Voting).
   - Implement CLI argument parsing for paths and parameters.
   - Extend support to Parquet files.

Dependencies:
   - Python >= 3.8
   - pandas
   - numpy
   - scikit-learn
   - colorama
   - telegram_bot (assumed custom module)

Assumptions & Notes:
   - The script assumes a directory structure like "Datasets/<DatasetName>/<SubDir>/<File.csv>".
   - The GA feature file is located at "<InputFileDir>/Feature_Analysis/Genetic_Algorithm_Results_features.csv".
   - CSV files are the primary data format.
   - The `telegram_bot` module is available in the environment.
"""

import atexit # For playing a sound when the program finishes
import datetime # For getting the current date and time
import json # Import json for handling JSON strings within the CSV
import os # For running a command in the terminal
import platform # For getting the operating system name
import pandas as pd # Import pandas for data manipulation
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

def extract_genetic_algorithm_features(file_path):
   """
   Extracts the features selected by the Genetic Algorithm from the corresponding
   "Genetic_Algorithm_Results.csv" file located in the "Feature_Analysis"
   subdirectory relative to the input file's directory.

   It specifically retrieves the 'best_features' (a JSON string) from the row
   where the 'run_index' is 'best', and returns it as a Python list.

   :param file_path: Full path to the current CSV file being processed (e.g., "./Datasets/.../DrDoS_DNS.csv").
   :return: List of features selected by the GA, or None if the file is not found or fails to load/parse.
   """
   
   file_dir = os.path.dirname(file_path) # Determine the directory of the input file
   ga_results_path = os.path.join(file_dir, "Feature_Analysis", "Genetic_Algorithm_Results.csv") # Construct the path to the consolidated GA results file
   
   verbose_output(f"{BackgroundColors.GREEN}Extracting GA features for file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}") # Output the verbose message

   if not verify_filepath_exists(ga_results_path): # Verify if the GA results file exists
      print(f"{BackgroundColors.YELLOW}Warning: GA results file not found at {BackgroundColors.CYAN}{ga_results_path}{BackgroundColors.YELLOW}. Skipping GA feature extraction for this file.{Style.RESET_ALL}")
      return None # Return None if the file does not exist

   try: # Try to load the GA results
      df = pd.read_csv(ga_results_path, usecols=["best_features", "run_index"]) # Load only the necessary columns
      best_row = df[df["run_index"] == "best"].iloc[0] # Get the row where run_index is 'best'
      best_features_json = best_row["best_features"] # Get the JSON string of best features
      ga_features = json.loads(best_features_json) # Parse the JSON string into a Python list
      
      verbose_output(f"{BackgroundColors.GREEN}Successfully extracted {BackgroundColors.CYAN}{len(ga_features)}{BackgroundColors.GREEN} GA features from the 'best' run.{Style.RESET_ALL}") # Output the verbose message
      
      return ga_features # Return the list of GA features
   except IndexError: # If there is no 'best' run_index
      print(f"{BackgroundColors.RED}Error: 'best' run_index not found in GA results file at {BackgroundColors.CYAN}{ga_results_path}{Style.RESET_ALL}")
      return None # Return None if 'best' run_index is not found
   except Exception as e: # If there is an error loading or parsing the file
      print(f"{BackgroundColors.RED}Error loading/parsing GA features from {BackgroundColors.CYAN}{ga_results_path}{BackgroundColors.RED}: {e}{Style.RESET_ALL}")
      return None # Return None if there was an error

def extract_principal_component_analysis_features(file_path):
   """
   Extracts the optimal number of Principal Components (n_components)
   from the "PCA_Results.csv" file located in the "Feature_Analysis"
   subdirectory relative to the input file's directory.

   The best result is determined by the highest 'cv_f1_score'.

   :param file_path: Full path to the current CSV file being processed (e.g., "./Datasets/.../DrDoS_DNS.csv").
   :return: Integer representing the optimal number of components (n_components), or None if the file is not found or fails to load/parse.
   """
   
   file_dir = os.path.dirname(file_path) # Determine the directory of the input file
   pca_results_path = os.path.join(file_dir, "Feature_Analysis", "PCA_Results.csv") # Construct the path to the PCA results file
   
   verbose_output(f"{BackgroundColors.GREEN}Extracting PCA features for file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}") # Output the verbose message

   if not verify_filepath_exists(pca_results_path): # Verify if the PCA results file exists
      print(f"{BackgroundColors.YELLOW}Warning: PCA results file not found at {BackgroundColors.CYAN}{pca_results_path}{BackgroundColors.YELLOW}. Skipping PCA feature extraction for this file.{Style.RESET_ALL}")
      return None # Return None if the file does not exist

   try: # Try to load the PCA results
      df = pd.read_csv(pca_results_path, usecols=["n_components", "cv_f1_score"]) # Load only the necessary columns
      
      if df.empty: # Verify if the DataFrame is empty
         print(f"{BackgroundColors.RED}Error: PCA results file at {BackgroundColors.CYAN}{pca_results_path}{BackgroundColors.RED} is empty.{Style.RESET_ALL}")
         return None # Return None if the file is empty
          
      best_row_index = df["cv_f1_score"].idxmax() # Get the index of the row with the highest CV F1-Score
      best_n_components = df.loc[best_row_index, "n_components"] # Get the optimal number of components
      
      verbose_output(f"{BackgroundColors.GREEN}Successfully extracted best PCA configuration. Optimal components: {BackgroundColors.CYAN}{best_n_components}{Style.RESET_ALL}") # Output the verbose message
      
      return int(best_n_components) # Return the optimal number of principal components
      
   except KeyError as e: # Handle missing columns
      print(f"{BackgroundColors.RED}Error: Required column {e} not found in PCA results file at {BackgroundColors.CYAN}{pca_results_path}{Style.RESET_ALL}")
      return None # Return None if required column is missing
   except Exception as e: # Handle other errors (loading, parsing, etc.)
      print(f"{BackgroundColors.RED}Error loading/parsing PCA features from {BackgroundColors.CYAN}{pca_results_path}{BackgroundColors.RED}: {e}{Style.RESET_ALL}")
      return None # Return None if there was an error

def extract_recursive_feature_elimination_features(file_path):
   """
   Extracts the "top_features" list (JSON string) from the first row of the
   "RFE_Run_Results.csv" file located in the "Feature_Analysis" subdirectory
   relative to the input file's directory.

   :param file_path: Full path to the current CSV file being processed (e.g., "./Datasets/.../DrDoS_DNS.csv").
   :return: List of top features selected by RFE from the first run, or None if the file is not found or fails to load/parse.
   """
   
   file_dir = os.path.dirname(file_path) # Determine the directory of the input file
   rfe_runs_path = os.path.join(file_dir, "Feature_Analysis", "RFE_Run_Results.csv") # Construct the path to the RFE runs file
   
   verbose_output(f"{BackgroundColors.GREEN}Extracting RFE features for file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}") # Output the verbose message

   if not verify_filepath_exists(rfe_runs_path): # Verify if the RFE runs file exists
      print(f"{BackgroundColors.YELLOW}Warning: RFE runs file not found at {BackgroundColors.CYAN}{rfe_runs_path}{BackgroundColors.YELLOW}. Skipping RFE feature extraction for this file.{Style.RESET_ALL}")
      return None # Return None if the file does not exist

   try: # Try to load the RFE runs results
      df = pd.read_csv(rfe_runs_path, usecols=["top_features"]) # Load only the "top_features" column
      
      if not df.empty: # Verify if the DataFrame is not empty
         top_features_json = df.loc[0, "top_features"] # Get the JSON string from the first row
         rfe_features = json.loads(top_features_json) # Parse the JSON string into a Python list
         
         verbose_output(f"{BackgroundColors.GREEN}Successfully extracted RFE top features from Run 1. Total features: {BackgroundColors.CYAN}{len(rfe_features)}{Style.RESET_ALL}") # Output the verbose message
         
         return rfe_features # Return the list of RFE features
      else: # If the DataFrame is empty
         print(f"{BackgroundColors.RED}Error: RFE runs file at {BackgroundColors.CYAN}{rfe_runs_path}{BackgroundColors.RED} is empty.{Style.RESET_ALL}")
         return None # Return None if the file is empty
         
   except Exception as e: # If there is an error loading or parsing the file
      print(f"{BackgroundColors.RED}Error loading/parsing RFE features from {BackgroundColors.CYAN}{rfe_runs_path}{BackgroundColors.RED}: {e}{Style.RESET_ALL}")
      return None # Return None if there was an error

def load_feature_selection_results(file_path):
   """
   Load GA, RFE and PCA feature selection artifacts for a given dataset file and
   print concise status messages.

   :param file_path: Path to the dataset CSV being processed.
   :return: Tuple (ga_selected_features, pca_n_components, rfe_selected_features)
   """

   ga_selected_features = extract_genetic_algorithm_features(file_path) # Extract GA features
   if ga_selected_features: # If GA features were successfully extracted
      print(f"{BackgroundColors.GREEN}GA Features successfully loaded for {BackgroundColors.CYAN}{os.path.basename(file_path)}{BackgroundColors.GREEN}. Total features: {BackgroundColors.CYAN}{len(ga_selected_features)}{Style.RESET_ALL}")
   else: # If GA features were not extracted
      print(f"{BackgroundColors.YELLOW}Proceeding without GA features for {BackgroundColors.CYAN}{os.path.basename(file_path)}{Style.RESET_ALL}")
      
   pca_n_components = extract_principal_component_analysis_features(file_path) # Extract PCA components
   if pca_n_components: # If PCA components were successfully extracted
      print(f"{BackgroundColors.GREEN}PCA optimal components successfully loaded for {BackgroundColors.CYAN}{os.path.basename(file_path)}{BackgroundColors.GREEN}: {BackgroundColors.CYAN}{pca_n_components}{Style.RESET_ALL}")
   else: # If PCA components were not extracted
      print(f"{BackgroundColors.YELLOW}Proceeding without PCA components for {BackgroundColors.CYAN}{os.path.basename(file_path)}{Style.RESET_ALL}")

   rfe_selected_features = extract_recursive_feature_elimination_features(file_path) # Extract RFE features
   if rfe_selected_features: # If RFE features were successfully extracted
      print(f"{BackgroundColors.GREEN}RFE Features successfully loaded for {BackgroundColors.CYAN}{os.path.basename(file_path)}{BackgroundColors.GREEN}. Total features: {BackgroundColors.CYAN}{len(rfe_selected_features)}{Style.RESET_ALL}")
   else: # If RFE features were not extracted
      print(f"{BackgroundColors.YELLOW}Proceeding without RFE features for {BackgroundColors.CYAN}{os.path.basename(file_path)}{Style.RESET_ALL}")

   return ga_selected_features, pca_n_components, rfe_selected_features # Return the extracted features

def load_dataset(csv_path):
   """
   Load CSV and return DataFrame.

   :param csv_path: Path to CSV dataset.
   :return: DataFrame
   """

   verbose_output(f"\n{BackgroundColors.GREEN}Loading dataset from: {BackgroundColors.CYAN}{csv_path}{Style.RESET_ALL}") # Output the loading dataset message

   if not verify_filepath_exists(csv_path): # If the CSV file does not exist
      print(f"{BackgroundColors.RED}CSV file not found: {csv_path}{Style.RESET_ALL}")
      return None # Return None

   df = pd.read_csv(csv_path, low_memory=False) # Load the dataset

   df.columns = df.columns.str.strip() # Clean column names by stripping leading/trailing whitespace

   if df.shape[1] < 2: # If there are less than 2 columns
      print(f"{BackgroundColors.RED}CSV must have at least 1 feature and 1 target.{Style.RESET_ALL}")
      return None # Return None

   return df # Return the loaded DataFrame

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
   
   for file in files_to_process: # For each file to process
      print(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing file: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}") # Output the file being processed
      
      ga_selected_features, pca_n_components, rfe_selected_features = load_feature_selection_results(file) # Load feature selection results
      
      # TODO: Implement data loading and preprocessing logic here
      
      # TODO: Implement classifiers stacking and evaluation logic here
      
      # TODO: Send Telegram notification about the processing status for every loop/iteration
      
      # TODO: Implement saving of results in a CSV file.

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
