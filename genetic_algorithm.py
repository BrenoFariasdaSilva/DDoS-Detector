"""
================================================================================
Genetic Algorithm Feature Selection & Analysis
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-10-07
Description :
   This script runs a DEAP-based Genetic Algorithm (GA) to perform feature
   selection for classification problems. It provides an end-to-end pipeline:
   dataset loading and cleaning, scaling, GA setup and execution, candidate
   evaluation (with a Random Forest base estimator), and post-hoc analysis
   (RFE ranking correlation, CSV summaries and boxplot visualizations).

   Key features include:
      - DEAP-based GA for binary-mask feature selection
      - Fitness evaluation using a RandomForest and returning multi-metrics
        (accuracy, precision, recall, F1, FPR, FNR, elapsed time)
      - Population sweep support (run GA over a range of population sizes)
      - Integration with previously-computed RFE rankings for cross-checking
      - Exports: best-subset text file, CSV summaries and per-feature boxplots
      - Progress bars via tqdm and safe filename handling for outputs
      - Cross-platform completion notification (optional sound)

Usage:
   1. Configure the dataset:
      - Edit the `csv_file` variable in the `main()` function to point to
         the CSV dataset you want to analyze (the script assumes the last
         column is the target and numeric features are used).
   2. Optionally tune GA parameters in `main()` or call sites:
      - n_generations, min_pop, max_pop, population_size, train_test_ratio
   3. Run the pipeline via the project's Makefile:
      $ make main
      (Makefile is expected to setup env / deps and execute this script.)
   NOTE:
      - If you prefer not to use the Makefile, you can run the module/script
        directly from Python in your dev environment, but the recommended
        workflow for the project is `make main`.

Outputs:
   - Feature_Analysis/Genetic_Algorithm_results.txt  (best subset + RFE cross-info)
   - Feature_Analysis/<dataset>_feature_summary.csv  (mean/std per class for selected features)
   - Feature_Analysis/<dataset>-<feature>.png         (boxplots for top features)
   - Console summary of best subsets per population size (when sweeping)

TODOs:
   - Add CLI argument parsing (argparse) to avoid editing `main()` for different runs.
   - Add cross-validation or nested CV to make fitness evaluation more robust.
   - Support multi-objective optimization (e.g., F1 vs. model training time).
   - Parallelize individual evaluations (joblib / dask) to speed up GA fitness calls.
   - Save and version best individuals (pickle/JSON) and GA run metadata.
   - Implement reproducible seeding across DEAP, numpy, random and sklearn.
   - Add automatic handling of categorical features and missing-value imputation.
   - Add early stopping and convergence checks to the GA loop.
   - Produce a machine-readable summary (JSON) of final metrics and selected features.
   - Add unit tests for core functions (fitness evaluation, GA setup, I/O).

Dependencies:
   - Python >= 3.9
   - pandas, numpy, scikit-learn, deap, tqdm, matplotlib, seaborn, colorama

Assumptions & Notes:
   - Dataset format: CSV, last column = target. Only numeric features are used.
   - RFE results (if present) are read from `Feature_Analysis/RFE_results_RandomForestClassifier.txt`.
   - Sound notification is skipped on Windows by default.
   - The script uses RandomForestClassifier as the default evaluator; change as needed.
   - Inspect output directories (`Feature_Analysis/`) after runs for artifacts.
"""

import atexit # For playing a sound when the program finishes
import matplotlib.pyplot as plt # For plotting graphs
import numpy as np # For numerical operations
import os # For running a command in the terminal
import pandas as pd # For data manipulation
import platform # For getting the operating system name
import random # For random number generation
import re # For sanitizing filenames
import seaborn as sns # For enhanced plotting
import time # For measuring execution time
from colorama import Style # For coloring the terminal
from deap import base, creator, tools, algorithms # For the genetic algorithm
from tqdm import tqdm # For progress bars
from sklearn.ensemble import RandomForestClassifier # For the machine learning model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix # For model evaluation
from sklearn.model_selection import train_test_split # For splitting the dataset
from sklearn.preprocessing import StandardScaler # For feature scaling

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

# Functions Definition

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

def load_dataset(csv_path):
   """
   Load CSV and return DataFrame.

   :param csv_path: Path to CSV dataset.
   :return: DataFrame
   """

   print(f"\n{BackgroundColors.GREEN}Loading dataset from: {BackgroundColors.CYAN}{csv_path}{Style.RESET_ALL}") # Output the loading dataset message

   if not verify_filepath_exists(csv_path): # If the CSV file does not exist
      print(f"{BackgroundColors.RED}CSV file not found: {csv_path}{Style.RESET_ALL}")
      return None # Return None

   df = pd.read_csv(csv_path, low_memory=False) # Load the dataset

   if df.shape[1] < 2: # If there are less than 2 columns
      print(f"{BackgroundColors.RED}CSV must have at least 1 feature and 1 target.{Style.RESET_ALL}")
      return None # Return None

   return df # Return the loaded DataFrame

def split_dataset(df, train_test_ratio=0.2):
   """
   Split dataset into training and testing sets.

   :param df: DataFrame to split.
   :param train_test_ratio: Proportion of the dataset to include in the test split.
   :return: X_train, X_test, y_train, y_test
   """

   X = df.iloc[:, :-1].select_dtypes(include=["number"]) # Select only numeric features
   y = df.iloc[:, -1] # Target variable
   if y.dtype == object or y.dtype == "category": # If the target variable is categorical
      y, _ = pd.factorize(y) # Factorize the target variable

   X = X.replace([np.inf, -np.inf], np.nan).dropna() # Remove rows with NaN or infinite values
   y = y[X.index] # Align y with cleaned X

   if X.empty: # If no numeric features remain after cleaning
      print(f"{BackgroundColors.RED}No valid numeric features remain after cleaning.{Style.RESET_ALL}")
      return None, None, None, None, None # Return None values

   scaler = StandardScaler() # Initialize the scaler
   X_scaled = scaler.fit_transform(X) # Scale the features
   X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=train_test_ratio, random_state=42) # Split the dataset

   return X_train, X_test, y_train, y_test, X.columns # Return the split data and feature names

def main():
   """
   Main function.

   :return: None
   :return: None
   """

   print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Genetic Algorithm Feature Selection{BackgroundColors.GREEN} program!{Style.RESET_ALL}")

   csv_file = "./DDoS/CICDDoS2019/01-12/DrDoS_DNS.csv"

   sweep_results = run_population_sweep(csv_file, n_generations=20, min_pop=3, max_pop=30, train_test_ratio=0.2)

   print(f"\n{BackgroundColors.GREEN}Summary of best features by population size:{Style.RESET_ALL}") # Print summary of results
   for pop_size, features in sweep_results.items(): # For each population size and its best features
      print(f"  Pop {pop_size}: {len(features)} features -> {features}") # Print the population size and the best features

   print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}") # Output the end of the program message

   atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None # Register the play_sound function to be called when the program finishes

if __name__ == "__main__":
   """
   This is the standard boilerplate that calls the main() function.

   :return: None
   """

   main() # Call the main function
