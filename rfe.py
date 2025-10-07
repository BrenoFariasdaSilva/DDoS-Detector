"""
================================================================================
Recursive Feature Elimination (RFE) Automation and Feature Analysis Tool
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-10-07
Description :
   This script automates the process of performing Recursive Feature Elimination (RFE)
   on structured datasets to identify the most relevant features for classification tasks.
   It provides a fully integrated pipeline — from dataset loading and preprocessing
   to feature ranking, visualization, and export of analysis reports.

   Core functionalities include:
      - Dataset validation and safe file handling
      - Standardization of numeric features using z-score normalization
      - Recursive Feature Elimination (RFE) with Random Forest as the base estimator
      - Generation of ranked feature lists with visual and statistical summaries
      - Boxplot-based visualization of top features by class distribution
      - Cross-platform sound notification upon completion

Usage:
   1. Set the `csv_file` variable inside the `main()` function to the dataset path.
   2. Run the script using:
      $ make main
   3. The program will automatically:
      - Load and clean the dataset
      - Run RFE to select the most relevant features
      - Save results and visualizations to the `Feature_Analysis/` directory
      - Optionally play a notification sound when finished

Output:
   - Text report (`RFE_results_<Model>.txt`) summarizing feature rankings.
   - CSV summary of top features with mean and standard deviation per class.
   - Boxplot visualizations for each selected feature stored in `Feature_Analysis/`.

TODOs:
   - Add support for additional estimators (e.g., SVM, Gradient Boosting).
   - Integrate evaluation metrics (F1-score, accuracy, precision, recall, FPR, FNR)
     directly after feature selection.
   - Incorporate correlation analysis to remove redundant features.
   - Extend preprocessing to handle categorical and missing data automatically.
   - Implement CLI argument parsing for dataset paths and configuration options.
   - Add parallel RFE runs with different feature subset sizes (1, 2, 5, 10, 15, 20, 25).

Dependencies:
   - Python >= 3.9
   - pandas, numpy, seaborn, matplotlib, scikit-learn, colorama

Notes:
   - The last column of the dataset is assumed to be the target variable.
   - Only numeric columns are considered for RFE processing.
   - Sound playback is skipped on Windows platforms by default.
"""

import atexit # For playing a sound when the program finishes
import os # For file and directory operations
import numpy as np # For numerical operations
import pandas as pd # For data manipulation
import matplotlib.pyplot as plt # For plotting
import re # For regular expressions
import seaborn as sns # For advanced plots
import platform # For getting the operating system name
from colorama import Style # For coloring the terminal
from sklearn.model_selection import train_test_split # For splitting the data
from sklearn.preprocessing import StandardScaler # For scaling the data (standardization)
from sklearn.feature_selection import RFE # For Recursive Feature Elimination
from sklearn.ensemble import RandomForestClassifier # For the Random Forest model

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
   
   if VERBOSE and true_string != "": # If VERBOSE is True and a true_string was provided
      print(true_string)
   elif false_string != "": # If a false_string was provided
      print(false_string)

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

def safe_filename(name):
   """
   Converts a string to a safe filename by replacing invalid characters with underscores.

   :param name: The original string
   :return: A safe filename string
   """

   return re.sub(r'[\\/*?:"<>|]', "_", name) # Replace invalid characters with underscores

def analyze_top_features(df, y, top_features, csv_path="."):
   """
   Analyze distribution of top features for each class and save plots + CSV summary.
   Numeric values are rounded to 3 decimal places.

   :param df: The DataFrame containing the features
   :param y: The target variable (class labels)
   :param top_features: List of top feature names to analyze
   :param csv_path: Path to the original CSV file (used for naming output files)
   :return: None
   """

   df_analysis = df[top_features].copy() # Create a copy of the DataFrame with only the top features
   df_analysis["Target"] = pd.Series(y, index=df_analysis.index).astype(str) # Add the target column as string for better plotting

   output_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis/" # Define output directory
   os.makedirs(output_dir, exist_ok=True) # Create directory if it doesn't exist

   base_dataset_name = os.path.splitext(os.path.basename(csv_path))[0] # Base name of the dataset without extension

   summary = df_analysis.groupby("Target")[top_features].agg(["mean", "std"]) # Group by target and calculate mean and std
   summary.columns = [f"{col}_{stat}" for col, stat in summary.columns] # Flatten MultiIndex columns
   summary = summary.round(3) # Round to 3 decimal places

   summary_csv_path = f"{output_dir}/{base_dataset_name}_feature_summary.csv" # Define summary CSV path
   summary.to_csv(summary_csv_path, encoding="utf-8") # Save summary to CSV
   print(f"Feature summary saved to: {summary_csv_path}") # Output the path to the summary CSV

   for feature in top_features: # Plot distribution for each top feature
      plt.figure(figsize=(8, 5)) # Set figure size
      sns.boxplot(x="Target", y=feature, data=df_analysis, hue="Target", palette="Set2", dodge=False) # Boxplot
      plt.title(f"Distribution of '{feature}' by class") # Set title
      plt.xlabel("Traffic Type") # Set x-axis label
      plt.ylabel(feature) # Set y-axis label
      plt.tight_layout() # Adjust layout
      plt.savefig(f"{output_dir}/{base_dataset_name}-{safe_filename(feature)}.png") # Save plot
      plt.close() # Close plot to free memory

def main():
   """
   Main function.

   :return: None
   """
   
   print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Recursive Feature Elimination (RFE){BackgroundColors.GREEN} program!{Style.RESET_ALL}") # Output the welcome message

   csv_file = "./DDoS/CICDDoS2019/01-12/DrDoS_DNS.csv" # Path to the CSV file
   run_rfe(csv_file) # Run RFE on the specified CSV file

   print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}") # Output the end of the program message

   atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None # Register play_sound at exit if enabled

if __name__ == "__main__":
   """
   This is the standard boilerplate that calls the main() function.

   :return: None
   """
   
   main() # Call the main function
