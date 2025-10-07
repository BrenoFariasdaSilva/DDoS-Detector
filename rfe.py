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

def main():
   """
   Main function.

   :return: None
   """
   
   pass

if __name__ == "__main__":
   """
   This is the standard boilerplate that calls the main() function.

   :return: None
   """
   
   main() # Call the main function
