"""
================================================================================
Principal Component Analysis (PCA) Feature Extraction & Evaluation Tool
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-11-08
Description :
   This script automates the process of performing Principal Component Analysis (PCA)
   for dimensionality reduction on structured datasets for classification tasks.
   It provides a fully integrated pipeline from dataset loading and preprocessing
   to PCA transformation, model evaluation with cross-validation, and export of results.

   Core functionalities include:
      - Dataset validation and safe file handling
      - Standardization of numeric features using z-score normalization
      - PCA transformation with configurable number of components (8, 16, 24, 32)
      - 10-fold Stratified Cross-Validation for robust performance evaluation
      - Comprehensive metrics: Accuracy, Precision, Recall, F1-Score, FPR, FNR
      - Generation of results files with performance metrics and explained variance
      - Comparison of different PCA configurations
      - Cross-platform sound notification upon completion

Usage:
   1. Set the `csv_file` variable inside the `main()` function to the dataset path.
   2. Configure the `n_components_list` to test different dimensionality reductions.
   3. Run the script using:
      $ python pca.py
      or via Makefile if configured
   4. The program will automatically:
      - Load and clean the dataset
      - Test PCA with different numbers of components (8, 16, 24, 32)
      - Perform 10-fold CV for each configuration
      - Save results and comparison to the `Feature_Analysis/` directory
      - Optionally play a notification sound when finished

Output:
   - Text report (`PCA_Results.txt`) summarizing performance for each configuration
   - CSV comparison of all PCA configurations
   - Console output with detailed metrics and explained variance ratios

TODOs:
   - Add visualization of explained variance ratio
   - Implement automatic selection of optimal number of components
   - Add support for kernel PCA (non-linear dimensionality reduction)
   - Integrate feature importance analysis for original features
   - Add parallel processing for multiple PCA configurations
   - Implement CLI argument parsing for dataset paths and configuration options
   - Add incremental PCA support for large datasets

Dependencies:
   - Python >= 3.9
   - pandas, numpy, scikit-learn, colorama

Notes:
   - The last column of the dataset is assumed to be the target variable.
   - Only numeric columns are considered for PCA processing.
   - PCA components are linear combinations of original features.
   - Results use 10-fold Stratified Cross-Validation on training data only.
"""

import atexit # For playing a sound when the program finishes
import os # For file and directory operations
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
	
	print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}PCA Feature Extraction{BackgroundColors.GREEN} program!{Style.RESET_ALL}")
	
	csv_file = "./Datasets/DDoS/CICDDoS2019/01-12/DrDoS_DNS.csv" # Path to the CSV dataset file
	n_components_list = [8, 16, 24, 32] # List of PCA component counts to test
	
	run_pca_analysis(csv_file, n_components_list) # Run the PCA analysis
	
	print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}")
	
	atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None # Register the play_sound function to be called at exit if enabled

if __name__ == "__main__":
	"""
	This is the standard boilerplate that calls the main() function.

	:return: None
	"""

	main()
