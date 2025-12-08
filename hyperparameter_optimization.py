"""
================================================================================
Classifiers Hyperparameter Optimization
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-12-06
Description :
   This script performs hyperparameter optimization for multiple machine learning
   classifiers using GridSearchCV on DDoS detection datasets. It uses only the
   features selected by the Genetic Algorithm for optimal performance. The script
   evaluates Random Forest, SVM, XGBoost, Logistic Regression, KNN, Nearest Centroid,
   Gradient Boosting, LightGBM, and MLP Neural Network classifiers.

   Key features include:
      - Automatic loading of Genetic Algorithm selected features
      - Data preprocessing with NaN/infinite value removal and zero-variance filtering
      - Comprehensive hyperparameter search grids for each classifier
      - Cross-validation with stratified K-fold (cv=5) for robust evaluation
      - F1-score optimization (weighted average for multi-class problems)
      - Results saved to CSV with best parameters and cross-validation scores
      - Progress tracking with tqdm progress bars
      - Sound notification upon completion

Usage:
   1. Configure the dataset path in the main() function (csv_path variable).
   2. Execute the script via Python:
         $ python hyperparameter_optimization.py
   3. Check the output CSV file for optimized hyperparameters and scores.

Outputs:
   - Hyperparameter_Optimization_Results.csv: Contains best parameters and CV scores for each model
   - Terminal logs showing optimization progress and execution time
   - Sound notification when processing completes

TODOs:
   - Add CLI argument parsing for dataset path and output directory
   - Implement Bayesian optimization for more efficient search
   - Add support for custom parameter grids
   - Extend to support additional classifiers
   - Add visualization of hyperparameter importance

Dependencies:
   - Python >= 3.8
   - pandas
   - numpy
   - scikit-learn
   - lightgbm
   - xgboost
   - tqdm
   - colorama

Assumptions & Notes:
   - Dataset CSV files with last column as target variable
   - Genetic Algorithm results available in Feature_Analysis/Genetic_Algorithm_Results.csv
   - Numeric features only (non-numeric features are dropped)
   - Uses weighted F1-score for multi-class classification
   - Sound notification skipped on Windows platform
"""

import atexit # For playing a sound when the program finishes
import datetime # For getting the current date and time
import json # For handling JSON strings
import lightgbm as lgb # For LightGBM model
import numpy as np # For numerical operations
import os # For running a command in the terminal
import pandas as pd # For data manipulation
import platform # For getting the operating system name
from colorama import Style # For coloring the terminal
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier # For ensemble models
from sklearn.linear_model import LogisticRegression # For logistic regression model
from sklearn.metrics import make_scorer, f1_score # For custom scoring metrics
from sklearn.model_selection import GridSearchCV, train_test_split # For hyperparameter search and data splitting
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid # For k-nearest neighbors model
from sklearn.neural_network import MLPClassifier # For neural network model
from sklearn.preprocessing import LabelEncoder, StandardScaler # For label encoding and feature scaling
from sklearn.svm import SVC # For Support Vector Machine model
from tqdm import tqdm # For progress bars
from xgboost import XGBClassifier # For XGBoost classifier

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
CV_FOLDS = 5 # Number of cross-validation folds for GridSearchCV
N_JOBS = -1 # Number of parallel jobs for GridSearchCV (-1 uses all processors)
RESULTS_FILENAME = "Hyperparameter_Optimization_Results.csv" # Filename for saving results
IGNORE_FILES = [RESULTS_FILENAME] # List of filenames to ignore when searching for datasets
IGNORE_DIRS = ["Cache", "Dataset_Description", "Data_Separability", "Feature_Analysis"] # List of directory names to ignore when searching for datasets

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
      filename = os.path.basename(item_path) # Get the filename
      
      if any(ignore and (ignore == filename or ignore == item_path) for ignore in IGNORE_FILES): # If the file is in the IGNORE_FILES list
         verbose_output(f"{BackgroundColors.YELLOW}Ignoring file {BackgroundColors.CYAN}{filename}{BackgroundColors.YELLOW} listed in IGNORE_FILES{Style.RESET_ALL}")
         continue # Skip this file
      
      if os.path.isfile(item_path) and item.lower().endswith(file_extension): # If the item is a file and has the specified extension
         files.append(item_path) # Add the file to the list

   return sorted(files) # Return sorted list for consistency

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

   print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Main Template Python{BackgroundColors.GREEN} program!{Style.RESET_ALL}", end="\n\n") # Output the welcome message
   start_time = datetime.datetime.now() # Get the start time of the program
   
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
