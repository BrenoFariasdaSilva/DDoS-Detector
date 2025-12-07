#!/usr/bin/env python3
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
      - Telegram bot notifications for progress updates

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
   - python-telegram-bot
   - python-dotenv

Assumptions & Notes:
   - Dataset format: CSV, last column = target. Only numeric features are used.
   - RFE results (if present) are read from `Feature_Analysis/RFE_results_RandomForestClassifier.txt`.
   - Sound notification is skipped on Windows by default.
   - The script uses RandomForestClassifier as the default evaluator; change as needed.
   - Inspect output directories (`Feature_Analysis/`) after runs for artifacts.
   - .env file must be present with TELEGRAM_API_KEY and CHAT_ID for notifications
"""

import atexit # For playing a sound when the program finishes
import csv # For writing metrics/features CSVs
import datetime # For timestamping
import json # For structured JSON output and parsing
import matplotlib.pyplot as plt # For plotting graphs
import multiprocessing # For parallel fitness evaluation
import numpy as np # For numerical operations
import os # For running a command in the terminal
import pandas as pd # For data manipulation
import pickle # For caching preprocessed data
import platform # For getting the operating system name
import random # For random number generation
import re # For sanitizing filenames
import seaborn as sns # For enhanced plotting
import shutil # For checking disk usage
import time # For measuring execution time
import warnings # For suppressing warnings
from colorama import Style # For coloring the terminal
from deap import base, creator, tools, algorithms # For the genetic algorithm
from functools import partial # For creating partial functions
from sklearn.ensemble import RandomForestClassifier # For the machine learning model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix # For model evaluation
from sklearn.model_selection import train_test_split, StratifiedKFold # For splitting the dataset and cross-validation
from sklearn.preprocessing import StandardScaler # For feature scaling
from telegram_bot import TelegramBot # For Telegram notifications
from tqdm import tqdm # For progress bars

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
RUNS = 5 # Number of runs for Genetic Algorithm analysis
EARLY_STOP_ACC_THRESHOLD = 0.75 # Minimum acceptable accuracy for an individual
EARLY_STOP_FOLDS = 3 # Number of folds to check before early stopping
CPU_PROCESSES = 1 # Initial number of worker processes; can be updated by monitor
FILES_TO_IGNORE = [""] # List of files to ignore during processing
GA_GENERATIONS_COMPLETED = 0 # Updated by GA loop to inform monitor when some generations have run

# Fitness Cache:
fitness_cache = {} # Cache for fitness results to avoid re-evaluating same feature masks

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
      
      if any(ignore and (ignore == filename or ignore == item_path) for ignore in FILES_TO_IGNORE): # If the file is in the FILES_TO_IGNORE list
         verbose_output(f"{BackgroundColors.YELLOW}Ignoring file {BackgroundColors.CYAN}{filename}{BackgroundColors.YELLOW} listed in FILES_TO_IGNORE{Style.RESET_ALL}")
         continue # Skip this file
      
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

def update_progress_bar(progress_bar, dataset_name, csv_path, pop_size=None, max_pop=None, gen=None, n_generations=None, run=None, runs=None, progress_state=None):
   """
   Update a tqdm `progress_bar` description and postfix consistently.

   :param progress_bar: tqdm progress bar instance (or None)
   :param dataset_name: Name of the dataset
   :param csv_path: Path to the CSV file
   :param pop_size: Current population size (optional)
   :param max_pop: Maximum population size (optional)
   :param n_generations: Number of generations (optional)
   :param run: Current run index (1-based) (optional)
   :param runs: Total runs (optional)
   :param progress_state: Optional dict with keys "current_it" and "total_it" to show iterations
   :return: None
   """

   if progress_bar is None: # If no progress bar is provided
      return # Do nothing
   try: # Try to update the progress bar
      # Build run info as part of description (not postfix)
      run_str = f"{BackgroundColors.GREEN}Run {BackgroundColors.CYAN}{run}{BackgroundColors.GREEN}/{BackgroundColors.CYAN}{runs}{BackgroundColors.GREEN}" if run is not None and runs is not None else None
      
      csv_basename = os.path.basename(csv_path) # Get the CSV filename
      parent_dir = os.path.basename(os.path.dirname(csv_path)) # Get parent directory name
      if parent_dir.lower() != (dataset_name or "").lower(): # If parent directory differs from dataset_name (case-insensitive)
         csv_filename = f"{BackgroundColors.CYAN}{parent_dir}{BackgroundColors.GREEN}/{BackgroundColors.CYAN}{csv_basename}" # Include parent directory
      else: # If parent directory is same as dataset_name
         csv_filename = csv_basename # Use only basename
      
      if run_str: # If run string is provided
         base = f"{BackgroundColors.CYAN}{dataset_name}{BackgroundColors.GREEN} Dataset - {BackgroundColors.CYAN}{csv_filename}{BackgroundColors.GREEN}: {run_str}{Style.RESET_ALL}" # Base description with run info
      else: # If no run string
         base = f"{BackgroundColors.CYAN}{dataset_name}{BackgroundColors.GREEN} Dataset - {BackgroundColors.CYAN}{csv_filename}{Style.RESET_ALL}" # Base description without run info
      
      details = [] # List to hold detail strings (pop, gen)
      if pop_size is not None: # If population size is provided
         if max_pop is not None: # If maximum population size is also provided
            details.append(f"{BackgroundColors.GREEN}Pop {BackgroundColors.CYAN}{pop_size}{BackgroundColors.GREEN}/{BackgroundColors.CYAN}{max_pop}") # Show current/max population
         else: # If only current population size is provided
            details.append(f"{BackgroundColors.GREEN}Pop {BackgroundColors.CYAN}{pop_size}") # Show current population only

      if gen is not None and n_generations is not None: # If generation and total generations are provided
         details.append(f"{BackgroundColors.GREEN}Gen {BackgroundColors.CYAN}{gen}{BackgroundColors.GREEN}/{BackgroundColors.CYAN}{n_generations}") # Show current/total generations
      elif gen is not None: # If only generation is provided
         details.append(f"{BackgroundColors.GREEN}Gen {BackgroundColors.CYAN}{gen}") # Show current generation only
      elif n_generations is not None: # If only total generations is provided
         details.append(f"{BackgroundColors.GREEN}Gen {BackgroundColors.CYAN}{n_generations}") # Show total generations only
      if details: # If there are any details to show
         detail_str = ", ".join(details) # Join details with commas
         desc = f"{base}{BackgroundColors.GREEN} - {detail_str}{Style.RESET_ALL}"
      else: # If no details
         desc = base # Just use the base description

      if progress_state and isinstance(progress_state, dict): # If progress_state dict is provided
         try: # Try to extract iteration info
            current_it = int(progress_state.get("current_it", 0)) # Current iteration
            total_it = int(progress_state.get("total_it", 0)) # Total iterations
            desc = f"{desc} [{BackgroundColors.CYAN}{current_it}{BackgroundColors.GREEN}/{BackgroundColors.CYAN}{total_it}{BackgroundColors.GREEN} iterations]{Style.RESET_ALL}" # Append iteration info
         except Exception: # Silently ignore iteration info extraction failures
            pass # Do nothing

      progress_bar.set_description(desc) # Update the progress bar description
      progress_bar.refresh() # Refresh the progress bar display
   except Exception: # Silently ignore progress bar update failures
      pass # Do nothing

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

def preprocess_dataframe(df, remove_zero_variance=True):
   """
   Preprocess a DataFrame by removing rows with NaN or infinite values and
   dropping zero-variance numeric features.

   :param df: pandas DataFrame to preprocess
   :param remove_zero_variance: whether to drop numeric columns with zero variance
   :return: cleaned DataFrame
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Preprocessing the DataFrame by removing NaN/infinite values and zero-variance features.{Style.RESET_ALL}") # Output the verbose message

   if df is None: # If the DataFrame is None
      return df # Return None

   df_clean = df.replace([np.inf, -np.inf], np.nan).dropna() # Remove rows with NaN or infinite values

   if remove_zero_variance: # If remove_zero_variance is set to True
      numeric_cols = df_clean.select_dtypes(include=["number"]).columns # Select only numeric columns
      if len(numeric_cols) > 0: # If there are numeric columns
         variances = df_clean[numeric_cols].var(axis=0, ddof=0) # Calculate variances
         zero_var_cols = variances[variances == 0].index.tolist() # Get columns with zero variance
         if zero_var_cols: # If there are zero-variance columns
            df_clean = df_clean.drop(columns=zero_var_cols) # Drop zero-variance columns

   return df_clean # Return the cleaned DataFrame

def cache_preprocessed_data(result, cache_file, csv_path):
   """
   Cache the preprocessed data to a pickle file, checking disk space first.
   Also, compare and display size reduction compared to the original CSV.

   :param result: The tuple to cache (X_train_scaled, X_test_scaled, y_train_np, y_test_np, X.columns)
   :param cache_file: Path to the cache file.
   :param csv_path: Path to the original CSV file for size comparison.
   :return: None
   """
   
   X_train_scaled, X_test_scaled, y_train_np, y_test_np, X_columns = result # Unpack the result tuple
   estimated_size = (X_train_scaled.nbytes + X_test_scaled.nbytes + 
                     y_train_np.nbytes + y_test_np.nbytes + 
                     len(pickle.dumps(X_columns))) # Estimate the size of the data to cache
   cache_dir = os.path.dirname(cache_file) # Get the directory of the cache file
   total, used, free = shutil.disk_usage(cache_dir) # Get disk usage information
   if free < estimated_size * 1.1: # 10% margin
      print(f"{BackgroundColors.YELLOW}Warning: Insufficient disk space for caching ({estimated_size / (1024**3):.2f} GB needed, {free / (1024**3):.2f} GB free). Skipping cache save.{Style.RESET_ALL}") # Output warning message
      return # Return without saving
   else: # If there is enough space
      with open(cache_file, "wb") as f: # Open cache file for writing
         pickle.dump(result, f) # Dump the result to cache file
      verbose_output(f"{BackgroundColors.GREEN}Saved preprocessed data to cache {cache_file}.{Style.RESET_ALL}") # Output the verbose message

      # Compare sizes
      pickle_size = os.path.getsize(cache_file) # Get the size of the pickle file
      csv_size = os.path.getsize(csv_path) # Get the size of the original CSV file
      if csv_size > 0: # If CSV size is available
         reduction = (csv_size - pickle_size) / csv_size * 100 # Calculate reduction percentage
         print(f"{BackgroundColors.GREEN}Size comparison: CSV {csv_size / (1024**3):.2f} GB, Pickle {pickle_size / (1024**3):.2f} GB. Reduction: {reduction:.1f}%{Style.RESET_ALL}") # Output size comparison
      else: # If CSV size is not available
         print(f"{BackgroundColors.YELLOW}Could not compare sizes: CSV size unknown.{Style.RESET_ALL}") # Output warning message

def split_dataset(df, csv_path, test_size=0.2):
   """
   Split dataset into training and testing sets.

   :param df: DataFrame to split.
   :param csv_path: Path to the CSV file for caching.
   :param test_size: Proportion of the dataset to include in the test split.
   :return: X_train, X_test, y_train, y_test
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Splitting dataset into training and testing sets with test size = {test_size}.{Style.RESET_ALL}") # Output the verbose message

   cache_file = csv_path.replace(".csv", f"_cache_test{test_size}.pkl") # Cache file path, including test_size for uniqueness

   if os.path.exists(cache_file): # If cache exists
      verbose_output(f"{BackgroundColors.GREEN}Loading cached preprocessed data from {cache_file}.{Style.RESET_ALL}") # Output loading message
      with open(cache_file, "rb") as f: # Open cache file
         return pickle.load(f) # Load and return cached data

   X = df.iloc[:, :-1].select_dtypes(include=["number"]) # Select only numeric features
   y = df.iloc[:, -1] # Target variable
   if y.dtype == object or y.dtype == "category": # If the target variable is categorical
      y, _ = pd.factorize(y) # Factorize the target variable

   X = X.replace([np.inf, -np.inf], np.nan).dropna() # Remove rows with NaN or infinite values
   y = y.loc[X.index] if isinstance(y, pd.Series) else pd.Series(y, index=df.index).loc[X.index] # Align y with cleaned X

   if X.empty: # If no numeric features remain after cleaning
      print(f"{BackgroundColors.RED}No valid numeric features remain after cleaning.{Style.RESET_ALL}")
      return None, None, None, None, None # Return None values

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42) # Split the dataset

   scaler = StandardScaler() # Initialize the scaler
   X_train_scaled = scaler.fit_transform(X_train) # Fit scaler on training set and transform
   X_test_scaled = scaler.transform(X_test) # Transform test set with the same scaler

   y_train_np = np.array(y_train) # Convert y_train and y_test to numpy arrays for fast indexing
   y_test_np = np.array(y_test) # Convert y_train and y_test to numpy arrays for fast indexing

   result = X_train_scaled, X_test_scaled, y_train_np, y_test_np, X.columns # Prepare result tuple
   cache_preprocessed_data(result, cache_file, csv_path) # Cache the preprocessed data with size comparison
   return result # Return the splits and feature names

def print_ga_parameters(min_pop, max_pop, n_generations, feature_count):
   """
   Print the genetic algorithm parameters in verbose output.

   :param min_pop: Minimum population size.
   :param max_pop: Maximum population size.
   :param n_generations: Number of generations per run.
   :param feature_count: Number of features in the dataset.
   :return: None
   """
   
   print(f"{BackgroundColors.GREEN}Genetic Algorithm Parameters:{Style.RESET_ALL}")
   print(f"  {BackgroundColors.GREEN}Population sizes: {BackgroundColors.CYAN}{min_pop} to {max_pop}{Style.RESET_ALL}")
   print(f"  {BackgroundColors.GREEN}Generations per run: {BackgroundColors.CYAN}{n_generations}{Style.RESET_ALL}")
   print(f"  {BackgroundColors.GREEN}Number of features: {BackgroundColors.CYAN}{feature_count}{Style.RESET_ALL}")
   print(f"  {BackgroundColors.GREEN}Crossover probability: {BackgroundColors.CYAN}0.5{Style.RESET_ALL}")
   print(f"  {BackgroundColors.GREEN}Mutation probability: {BackgroundColors.CYAN}0.05{Style.RESET_ALL}")
   print(f"  {BackgroundColors.GREEN}Tournament size: {BackgroundColors.CYAN}3{Style.RESET_ALL}")
   print(f"  {BackgroundColors.GREEN}Fitness evaluation: {BackgroundColors.CYAN}10-fold Stratified CV on training set{Style.RESET_ALL}")
   print(f"  {BackgroundColors.GREEN}Base estimator: {BackgroundColors.CYAN}RandomForestClassifier (n_estimators=100, n_jobs=1){Style.RESET_ALL}")
   print(f"  {BackgroundColors.GREEN}Optimization goal: {BackgroundColors.CYAN}Maximize F1-Score{Style.RESET_ALL}")
   print("") # Empty line for spacing

def compute_progress_state(min_pop, max_pop, n_generations, runs, progress_bar, folds=10):
   """
   Compute an estimated progress_state dictionary for the population sweep.

   The function returns a dict with keys:
     - current_it: starting at 0
     - total_it: estimated total number of classifier instantiations

   The estimation assumes each individual evaluation runs `folds` classifier
   instantiations (10-fold CV by default) and includes one re-evaluation of
   the best individual per run.

   :param min_pop: minimum population size
   :param max_pop: maximum population size
   :param n_generations: number of generations per run
   :param runs: number of runs per population size
   :param progress_bar: if falsy, function returns None
   :param folds: CV folds per evaluation (default 10)
   :return: dict or None
   """
   
   if not progress_bar: # If no progress bar is provided
      return None # Return None

   try: # Try to compute the progress state
      n_pop_values = (max_pop - min_pop + 1) # Number of population sizes to evaluate
      sum_pop_sizes = (min_pop + max_pop) * n_pop_values // 2 # Sum of population sizes (arithmetic series)
      total_individual_evals = runs * (n_generations * sum_pop_sizes + n_pop_values * 1) # Total individual evaluations including best re-evaluations
      total_it = int(total_individual_evals * folds) # Total classifier instantiations
      return {"current_it": 0, "total_it": total_it} # Return the progress state dictionary
   except Exception: # If any error occurs
      return {"current_it": 0, "total_it": 0} # Return a default progress state

def setup_genetic_algorithm(n_features, population_size=None):
   """
   Setup DEAP Genetic Algorithm: creator, toolbox, population, and Hall of Fame.
   DEAP is a library for evolutionary algorithms in Python.

   :param n_features: Number of features in dataset
   :param population_size: Size of the population (default: n_features // 4, minimum 10)
   :return: toolbox, population, hall_of_fame
   """
   
   if population_size is None: # If population_size is not provided
      population_size = max(n_features // 4, 10) # Default to 1/4 of n_features, but at least 10
   
   verbose_output(f"{BackgroundColors.GREEN}Setting up Genetic Algorithm with {n_features} features and population size {population_size}.{Style.RESET_ALL}") # Output the verbose message

   # Avoid re-creating FitnessMax and Individual
   if not hasattr(creator, "FitnessMax"): # If FitnessMax is not already created
      creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximize F1-score
   if not hasattr(creator, "Individual"): # If Individual is not already created
      creator.create("Individual", list, fitness=creator.FitnessMax) # Individual = list with fitness

   toolbox = base.Toolbox() # Create a toolbox
   toolbox.register("attr_bool", random.randint, 0, 1) # Attribute generator (0 or 1)
   toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features) # Individual generator
   toolbox.register("population", tools.initRepeat, list, toolbox.individual) # Population generator

   toolbox.register("mate", tools.cxTwoPoint) # Crossover operator
   toolbox.register("mutate", tools.mutFlipBit, indpb=0.05) # Mutation operator
   toolbox.register("select", tools.selTournament, tournsize=3) # Selection operator

   if CPU_PROCESSES is None: # If CPU_PROCESSES is not set
      pool = multiprocessing.Pool() # Create a multiprocessing pool with all available CPUs
   else: # If CPU_PROCESSES is set
      pool = multiprocessing.Pool(processes=CPU_PROCESSES) # Create a multiprocessing pool with specified number of CPUs 
   toolbox.register("map", pool.map) # Register parallel map for fitness evaluation

   population = toolbox.population(n=population_size) # Create the initial population
   hof = tools.HallOfFame(1) # Hall of Fame to store the best individual

   return toolbox, population, hof # Return the toolbox, population, and Hall of Fame

def instantiate_estimator(estimator_cls=None):
   """
   Instantiate a classifier. If estimator_cls is None, use RandomForestClassifier.

   :param estimator_cls: Class of the estimator to instantiate (or None)
   :return: instantiated estimator
   """

   if estimator_cls is None: # If no estimator class is provided
      return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1) # Return a default RandomForestClassifier

   try: # Try to instantiate the provided estimator class
      return estimator_cls() # Instantiate with default parameters
   except Exception: # If instantiation fails
      return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1) # Fallback to default RandomForestClassifier

def evaluate_individual(individual, X_train, y_train, X_test, y_test, estimator_cls=None, progress_bar=None, dataset_name=None, csv_path=None, pop_size=None, max_pop=None, gen=None, n_generations=None, run=None, runs=None, progress_state=None):
   """
   Evaluate the fitness of an individual solution using 10-fold Stratified Cross-Validation
   on the training set only (não combina train+test para evitar data leakage).

   :param individual: A list representing the individual solution (binary mask for feature selection).
   :param X_train: Training feature set.
   :param y_train: Training target variable.
   :param X_test: Testing feature set (unused during CV, but kept for compatibility).
   :param y_test: Testing target variable (unused during CV, but kept for compatibility).
   :param estimator_cls: Classifier class to use (default: RandomForestClassifier).
   :param progress_bar: Optional tqdm progress bar instance.
   :param dataset_name: Optional dataset name for progress display.
   :param csv_path: Optional CSV path for progress display.
   :param pop_size: Optional population size for progress display.
   :param max_pop: Optional maximum population size for progress display.
   :param gen: Optional current generation for progress display.
   :param n_generations: Optional total generations for progress display.
   :param run: Optional current run for progress display.
   :param runs: Optional total runs for progress display.
   :param progress_state: Optional progress state dict for tracking iterations.
   :return: Tuple containing accuracy, precision, recall, F1-score, FPR, FNR
   """

   verbose_output(f"{BackgroundColors.GREEN}Evaluating individual: {BackgroundColors.CYAN}{individual}{Style.RESET_ALL}") # Output the verbose message

   if sum(individual) == 0: # If no features are selected
      return 0, 0, 0, 0, 1, 1 # Return worst possible scores

   mask_tuple = tuple(individual) # Convert individual to tuple for hashing
   if mask_tuple in fitness_cache: # Verify if already evaluated
      return fitness_cache[mask_tuple] # Return cached result

   mask = np.array(individual, dtype=bool) # Create boolean mask from individual
   X_train_sel = X_train[:, mask] # Select features based on the mask

   metrics = np.empty((0, 6), dtype=float) # Will hold metrics for each fold: [acc, prec, rec, f1, fpr, fnr]

   try: # Try to create StratifiedKFold splits
      skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42) # 10-fold Stratified CV
      splits = list(skf.split(X_train_sel, y_train)) # Generate splits
   except Exception: # If StratifiedKFold fails (e.g., too few samples per class)
      print(f"{BackgroundColors.YELLOW}Warning: StratifiedKFold failed, falling back to simple train/test split for evaluation due to {str(Exception)}{Style.RESET_ALL}") # Output warning message
      X_test_sel = X_test[:, mask] # Select features from test set
      model = instantiate_estimator(estimator_cls) # Instantiate the model
      model.fit(X_train_sel, y_train) # Fit the model on the training set
      y_pred = model.predict(X_test_sel) # Predict on the test set

      acc = accuracy_score(y_test, y_pred) # Calculate accuracy
      prec = precision_score(y_test, y_pred, average="weighted", zero_division=0) # Calculate precision
      rec = recall_score(y_test, y_pred, average="weighted", zero_division=0) # Calculate recall
      f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0) # Calculate F1-score

      cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test)) # Confusion matrix
      tn = cm[0, 0] if cm.shape == (2, 2) else 0 # True negatives
      fp = cm[0, 1] if cm.shape == (2, 2) else 0 # False positives
      fn = cm[1, 0] if cm.shape == (2, 2) else 0 # False negatives
      tp = cm[1, 1] if cm.shape == (2, 2) else 0 # True positives

      fpr = fp / (fp + tn) if (fp + tn) > 0 else 0 # False positive rate
      fnr = fn / (fn + tp) if (fn + tp) > 0 else 0 # False negative rate

      return acc, prec, rec, f1, fpr, fnr # Return metrics

   y_train_np = np.array(y_train) # Convert y_train to numpy array for fast indexing
   early_stop_triggered = False # Flag for early stopping

   for fold_idx, (train_idx, val_idx) in enumerate(splits): # For each fold
      model = instantiate_estimator(estimator_cls) # Instantiate the model
      y_train_fold = y_train_np[train_idx] # Get training fold labels
      y_val_fold = y_train_np[val_idx] # Get validation fold labels
      model.fit(X_train_sel[train_idx], y_train_fold) # Fit the model on the training fold
      y_pred = model.predict(X_train_sel[val_idx]) # Predict on the validation fold

      acc = accuracy_score(y_val_fold, y_pred) # Calculate accuracy
      prec = precision_score(y_val_fold, y_pred, average="weighted", zero_division=0) # Calculate precision
      rec = recall_score(y_val_fold, y_pred, average="weighted", zero_division=0) # Calculate recall
      f1 = f1_score(y_val_fold, y_pred, average="weighted", zero_division=0) # Calculate F1-score

      cm = confusion_matrix(y_val_fold, y_pred, labels=np.unique(y_val_fold)) # Confusion matrix
      tn = cm[0, 0] if cm.shape == (2, 2) else 0 # True negatives
      fp = cm[0, 1] if cm.shape == (2, 2) else 0 # False positives
      fn = cm[1, 0] if cm.shape == (2, 2) else 0 # False negatives
      tp = cm[1, 1] if cm.shape == (2, 2) else 0 # True positives

      fpr = fp / (fp + tn) if (fp + tn) > 0 else 0 # False positive rate
      fnr = fn / (fn + tp) if (fn + tp) > 0 else 0 # False negative rate

      metrics = np.vstack((metrics, np.array([acc, prec, rec, f1, fpr, fnr], dtype=float))) # Append metrics using NumPy

      if fold_idx < EARLY_STOP_FOLDS and acc < EARLY_STOP_ACC_THRESHOLD: # Early stopping: If accuracy is below threshold in first few folds, break
         early_stop_triggered = True # Set flag
         break # Stop evaluating further folds for this individual

   means = np.mean(metrics, axis=0) if metrics.shape[0] > 0 else np.zeros(6) # Calculate means for each metric
   acc, prec, rec, f1, fpr, fnr = means # Unpack mean metrics

   result = acc, prec, rec, f1, fpr, fnr # Prepare result tuple
   fitness_cache[mask_tuple] = result # Cache the result
   return result # Return vectorized average metrics

def ga_fitness(ind, fitness_func):
   """
   Global fitness function for GA evaluation to avoid pickle issues with local functions.
   
   :param ind: Individual to evaluate
   :param fitness_func: Partial function for evaluation
   :return: Tuple with F1-score
   """
   
   return (fitness_func(ind)[3],) # Return only the F1-score for GA optimization

def run_genetic_algorithm_loop(bot, toolbox, population, hof, X_train, y_train, X_test, y_test, n_generations=100, show_progress=False, progress_bar=None, dataset_name=None, csv_path=None, pop_size=None, max_pop=None, run=None, runs=None, progress_state=None):
   """
   Run Genetic Algorithm generations with a tqdm progress bar.

   :param bot: TelegramBot instance for sending messages.
   :param toolbox: DEAP toolbox with registered functions.
   :param population: Initial population.
   :param hof: Hall of Fame to store the best individual.
   :param X_train: Training feature set.
   :param y_train: Training target variable.
   :param X_test: Testing feature set.
   :param y_test: Testing target variable.
   :param n_generations: Number of generations to run.
   :return: best individual
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Running Genetic Algorithm for {n_generations} generations.{Style.RESET_ALL}") # Output the verbose message

   fitness_func = partial(evaluate_individual, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test) # Partial function for evaluation
   toolbox.register("evaluate", partial(ga_fitness, fitness_func=fitness_func)) # Register the global fitness function

   global GA_GENERATIONS_COMPLETED # To track completed generations
   best_fitness = None # Track the best fitness value
   gens_without_improvement = 0 # Counter for generations with no improvement
   early_stop_gens = 10 # Number of generations to wait for improvement before stopping

   folds = 10 # Number of folds used in cross-validation
   gen_range = tqdm(range(1, n_generations + 1), desc=f"{BackgroundColors.GREEN}Generations{Style.RESET_ALL}") if show_progress else range(1, n_generations + 1)
   gens_ran = 0 # Track how many generations were actually executed
   for gen in gen_range: # Loop for the specified number of generations
      update_progress_bar(progress_bar, dataset_name or "", csv_path or "", pop_size=pop_size, max_pop=max_pop, gen=gen, n_generations=n_generations, run=run, runs=runs, progress_state=progress_state) if progress_bar else None # Update progress bar if provided

      offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2) # Apply crossover and mutation
      fits = list(toolbox.map(toolbox.evaluate, offspring)) # Evaluate the offspring in parallel
      
      if progress_state and isinstance(progress_state, dict): # Update progress state if provided
         try: # Try to update progress state
            progress_state["current_it"] = int(progress_state.get("current_it", 0)) + len(offspring) * folds
            update_progress_bar(progress_bar, dataset_name or "", csv_path or "", pop_size=pop_size, max_pop=max_pop, gen=gen, n_generations=n_generations, run=run, runs=runs, progress_state=progress_state) if progress_bar else None # Update progress bar if provided
         except Exception: # Silently ignore progress update errors
            pass # Do nothing

      for ind, fit in zip(offspring, fits): # Assign fitness values
         ind.fitness.values = fit # Set the fitness value

      population[:] = toolbox.select(offspring, k=len(population)) # Select the next generation population
      hof.update(population) # Update the Hall of Fame

      current_best_fitness = hof[0].fitness.values[0] if hof and hof[0].fitness.values else None # Get current best fitness
      if best_fitness is None or (current_best_fitness is not None and current_best_fitness > best_fitness):
         best_fitness = current_best_fitness # Update best fitness
         gens_without_improvement = 0 # Reset counter
      else:
         gens_without_improvement += 1 # Increment counter

         if gens_without_improvement >= early_stop_gens: # Verify early-stop condition
            print(f"{BackgroundColors.YELLOW}Early stopping: No improvement in best fitness for {early_stop_gens} generations. Stopping at generation {gen}.{Style.RESET_ALL}")
            gens_ran = gen # Record how many generations were executed before early stopping
            GA_GENERATIONS_COMPLETED = int(gen) # Update global variable
            break # Stop the loop early

      if bot.TELEGRAM_BOT_TOKEN and bot.CHAT_ID and show_progress and gen % max(1, n_generations // 100) == 0: # Send periodic updates to Telegram in every ~1% of generations
         try: # Try to send message
            bot.send_messages([f"GA Progress: Generation {gen}/{n_generations}, Best F1-Score: {best_fitness:.4f}"]) # Send message to Telegram bot
         except Exception: # Silently ignore Telegram errors
            pass # Do nothing

      gens_ran = gen # Update gens_ran each generation
      GA_GENERATIONS_COMPLETED = int(gen) # Update global variable

   gens_ran = gen if gens_ran == 0 else gens_ran # Ensure gens_ran is set correctly if no early stopping occurred

   if hasattr(toolbox, "map") and hasattr(toolbox.map, "close"): # If using multiprocessing pool
      toolbox.map.close() # Close the pool
      toolbox.map.join() # Join the pool

   return hof[0], gens_ran # Return the best individual and number of generations actually run

def safe_filename(name):
   """
   Sanitize a string to be safe for use as a filename.

   :param name: The string to be sanitized.
   :return: A sanitized string safe for use as a filename.
   """

   return re.sub(r'[\\/*?:"<>|]', "_", name) # Replace invalid filename characters with underscores

def analyze_top_features(df, y, top_features, csv_path="."):
   """
   Analyze and visualize the top features.

   :param df: DataFrame containing the features.
   :param y: Target variable.
   :param top_features: List of top feature names.
   :param csv_path: Path to the original CSV file for saving outputs.
   :return: None
   """

   df_analysis = df[top_features].copy() # Create a copy of the DataFrame with only the top features
   df_analysis["Target"] = pd.Series(y, index=df_analysis.index).astype(str) # Add the target variable to the DataFrame

   output_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis" # Directory to save outputs
   os.makedirs(output_dir, exist_ok=True) # Create the directory if it doesn't exist

   base_dataset_name = os.path.splitext(os.path.basename(csv_path))[0] # Base name of the dataset

   summary = df_analysis.groupby("Target")[top_features].agg(["mean", "std"]) # Calculate mean and std for each feature grouped by target
   summary.columns = [f"{col}_{stat}" for col, stat in summary.columns] # Flatten MultiIndex columns
   summary = summary.round(3) # Round to 3 decimal places

   summary_csv_path = f"{output_dir}/{base_dataset_name}_feature_summary.csv" # Path to save the summary CSV
   summary.to_csv(summary_csv_path, encoding="utf-8") # Save the summary to a CSV file
   print(f"\n{BackgroundColors.GREEN}Features summary saved to {BackgroundColors.CYAN}{summary_csv_path}{Style.RESET_ALL}") # Notify user

   for feature in top_features: # For each top feature
      plt.figure(figsize=(8, 5)) # Create a new figure
      sns.boxplot(x="Target", y=feature, data=df_analysis, hue="Target", palette="Set2", dodge=False) # Boxplot
      plt.title(f"Distribution of '{feature}' by class") # Title
      plt.xlabel("Traffic Type") # X-axis label
      plt.ylabel(feature) # Y-axis label
      plt.tight_layout() # Adjust layout
      plt.savefig(f"{output_dir}/{base_dataset_name}-{safe_filename(feature)}.png") # Save the plot
      plt.close() # Close the plot to free memory

def normalize_feature_name(name):
   """
   Normalize feature name by stripping whitespace and replacing double spaces with single spaces.

   :param name: The feature name to normalize.
   :return: Normalized feature name
   """

   return name.strip().replace("  ", " ") # Strip whitespace and replace double spaces with single spaces

def extract_rfe_ranking(csv_path):
   """
   Extract RFE rankings from the RFE results file.

   :param csv_path: Path to the original CSV file for saving outputs.
   :return: Dictionary of feature names and their RFE rankings.
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Extracting RFE rankings from results file.{Style.RESET_ALL}") # Output the verbose message

   rfe_ranking = {} # Dictionary to store feature names and their RFE rankings
   dir_path = os.path.dirname(csv_path) # Directory that contains Feature_Analysis
   json_path = f"{dir_path}/Feature_Analysis/RFE_Summary.json" # Path to new JSON summary
   csv_path_runs = f"{dir_path}/Feature_Analysis/RFE_Runs_Summary.csv" # Path to runs summary CSV
   legacy_txt = f"{dir_path}/Feature_Analysis/RFE_results_RandomForestClassifier.txt" # Legacy TXT path (fallback)

   if verify_filepath_exists(json_path): # If JSON summary exists
      try: # Attempt to parse JSON
         with open(json_path, "r", encoding="utf-8") as jf: # Open JSON file
            data = json.load(jf) # Load JSON content
         if isinstance(data, dict): # Ensure data is a dictionary
            if "rfe_ranking" in data and isinstance(data["rfe_ranking"], dict): # rfe_ranking key
               rfe_ranking = data["rfe_ranking"] # Use provided ranking
            elif "per_run" in data and isinstance(data["per_run"], list) and len(data["per_run"])>0: # per_run list exists
               first = data["per_run"][0] # First run entry
               if isinstance(first, dict) and "ranking" in first and isinstance(first["ranking"], dict): # ranking dict
                  rfe_ranking = first["ranking"] # Use ranking
      except Exception as e: # If parsing JSON fails
         print(f"{BackgroundColors.YELLOW}Failed to parse RFE JSON summary: {str(e)}. Skipping RFE ranking extraction.{Style.RESET_ALL}") # Warn user
         return rfe_ranking # Return whatever we have (likely empty)
      return rfe_ranking # Return ranking extracted from JSON

   if verify_filepath_exists(csv_path_runs): # If CSV summary exists
      try: # Attempt to parse CSV
         with open(csv_path_runs, "r", encoding="utf-8") as cf: # Open CSV file
            reader = csv.DictReader(cf) # Use DictReader to parse headered CSV
            for row in reader: # Iterate rows
               for key, val in row.items(): # For each column value
                  if val and isinstance(val, str) and val.strip().startswith("{"): # Looks like JSON
                     try: # Try parse JSON string
                        parsed = json.loads(val) # Parse JSON string
                        if isinstance(parsed, dict) and all(isinstance(k, str) for k in parsed.keys()): # Likely ranking
                           rfe_ranking = parsed # Use parsed dict as ranking
                           return rfe_ranking # Return early
                     except Exception: # ignore parse errors and continue
                        pass
      except Exception as e: # If reading CSV fails
         print(f"{BackgroundColors.YELLOW}Failed to parse RFE CSV summary: {str(e)}. Skipping RFE ranking extraction.{Style.RESET_ALL}") # Warn user

   if not verify_filepath_exists(legacy_txt): # If no legacy file either
      print(f"{BackgroundColors.YELLOW}RFE results not found (tried JSON/CSV/TXT). Skipping RFE ranking extraction.{Style.RESET_ALL}") # Notify user
      return rfe_ranking # Return empty dictionary

   try: # Attempt to parse TXT file
      with open(legacy_txt, "r", encoding="utf-8") as f: # Open legacy TXT
         lines = f.readlines() # Read lines
      for line in lines: # Iterate lines
         line = line.strip() # Strip whitespace
         if not line: # Skip empty lines
            continue # Continue
         m = re.match(r"^\s*(\d+)\.?\s+(.+?)\s*$", line) # Try numeric prefix
         if m: # If matched numbered list
            rank = int(m.group(1)) # Get rank number
            feat = m.group(2).strip() # Get feature name
            rfe_ranking[feat] = rank # Store ranking
   except Exception as e: # If parsing fails
      print(f"{BackgroundColors.YELLOW}Failed to parse legacy RFE TXT: {str(e)}. Returning empty ranking.{Style.RESET_ALL}") # Notify user

   return rfe_ranking # Return the RFE rankings dictionary

def print_metrics(metrics):
   """
   Print performance metrics.

   :param metrics: Dictionary or tuple containing evaluation metrics.
   :return: None
   """

   if not metrics: # If metrics is None or empty
      return # Do nothing
   
   acc, prec, rec, f1, fpr, fnr, elapsed_time = metrics
   print(f"\n{BackgroundColors.GREEN}Performance Metrics for the Random Forest Classifier using the best feature subset:{Style.RESET_ALL}")
   print(f"   {BackgroundColors.GREEN}Accuracy: {BackgroundColors.CYAN}{acc:.4f}{Style.RESET_ALL}")
   print(f"   {BackgroundColors.GREEN}Precision: {BackgroundColors.CYAN}{prec:.4f}{Style.RESET_ALL}")
   print(f"   {BackgroundColors.GREEN}Recall: {BackgroundColors.CYAN}{rec:.4f}{Style.RESET_ALL}")
   print(f"   {BackgroundColors.GREEN}F1-Score: {BackgroundColors.CYAN}{f1:.4f}{Style.RESET_ALL}")
   print(f"   {BackgroundColors.GREEN}False Positive Rate (FPR): {BackgroundColors.CYAN}{fpr:.4f}{Style.RESET_ALL}")
   print(f"   {BackgroundColors.GREEN}False Negative Rate (FNR): {BackgroundColors.CYAN}{fnr:.4f}{Style.RESET_ALL}")
   print(f"   {BackgroundColors.GREEN}Elapsed Time (s): {BackgroundColors.CYAN}{int(elapsed_time)}{Style.RESET_ALL}")

def metrics_to_dict(metrics):
   """
   Convert a metrics tuple to a standardized dictionary.

   :param metrics: Metrics tuple in the form (accuracy, precision, recall, f1, fpr, fnr, elapsed_time)
             or None.
   :return: Dictionary with float values for each metric or None when input is falsy.
   """
   
   if not metrics: # If metrics is None or falsy, return explicit keys with None
      return {"accuracy": None, "precision": None, "recall": None, "f1_score": None, "fpr": None, "fnr": None, "elapsed_time_s": None}

   acc, prec, rec, f1, fpr, fnr, elapsed = metrics # Unpack metrics

   return {
      "accuracy": float(round(acc, 4)), # Accuracy
      "precision": float(round(prec, 4)), # Precision
      "recall": float(round(rec, 4)), # Recall
      "f1_score": float(round(f1, 4)), # F1 Score
      "fpr": float(round(fpr, 4)), # False Positive Rate
      "fnr": float(round(fnr, 4)), # False Negative Rate
      "elapsed_time": int(round(elapsed)) # Elapsed time in seconds
   }

def build_base_row(csv_path, best_pop_size, n_generations, n_train, n_test, test_frac, rfe_ranking):
   """
   Build the base dictionary row used for the consolidated GA CSV output.

   :param csv_path: Path to the original dataset CSV.
   :param best_pop_size: Population size that produced the best result.
   :param n_generations: Number of generations used in the GA.
   :param n_train: Number of training samples.
   :param n_test: Number of testing samples.
   :param test_frac: Train/test fraction.
   :param rfe_ranking: Dictionary with RFE rankings (will be JSON-encoded).
   :return: Dictionary representing the base row for CSV output.
   """
   
   timestamp = datetime.datetime.now().isoformat() # Current timestamp
   
   return { # Base row for CSV
      "dataset": os.path.splitext(os.path.basename(csv_path))[0], # Dataset name
      "dataset_path": csv_path, # Dataset path
      "population_size": best_pop_size, # Population size
      "n_generations": n_generations, # Number of generations
      "n_train": n_train, # Number of training samples
      "n_test": n_test, # Number of testing samples
      "train_test_ratio": test_frac, # Train/test fraction
      "timestamp": timestamp, # Timestamp
      "run_index": "best", # Indicates best run
      "rfe_ranking": json.dumps(rfe_ranking, ensure_ascii=False) # RFE ranking as JSON string
   }

def build_rows_list(rf_metrics, best_features, runs_list, feature_names, base_row):
   """
   Build the list of rows (dictionaries) that will be written to the consolidated GA CSV.

   :param rf_metrics: Metrics tuple for the best RandomForest evaluation (or None).
   :param best_features: List of features selected by the GA best individual.
   :param runs_list: Optional list of per-run dictionaries with keys 'metrics', 'best_features', 'best_ind'.
   :param feature_names: Original feature names list (used if per-run data includes binary masks).
   :param base_row: Base row dictionary produced by `build_base_row`.
   :return: List of dictionaries ready to be converted into a DataFrame.
   """
   
   rows = [] # List to hold CSV rows
   rf_row = dict(base_row) # Create Random Forest row
   rf_row.update({"classifier": "RandomForest",}) # Set classifier
   rf_row.update(metrics_to_dict(rf_metrics)) # Add RF metrics
   rf_row.update({"best_features": json.dumps(best_features, ensure_ascii=False),}) # Add best features as JSON string
   rows.append(rf_row) # Add RF row to rows

   if runs_list: # If multiple runs data is provided
      for idx, run_data in enumerate(runs_list, start=1): # For each run
         run_metrics = run_data.get("metrics") if run_data.get("metrics") is not None else None
         run_features = run_data.get("best_features") if run_data.get("best_features") is not None else [f for f, bit in zip(feature_names if feature_names is not None else [], run_data.get("best_ind", [])) if bit == 1] # Extract features for this run
         run_row = dict(base_row) # Create row for this run

         run_row.update({"classifier": "RandomForest"}) # Set classifier
         run_row.update(metrics_to_dict(run_metrics)) # Add run metrics
         run_row.update({"best_features": json.dumps(run_features, ensure_ascii=False), "run_index": idx}) # Add best features and run index
         rows.append(run_row) # Add run row to rows

   return rows # Return consolidated rows

def normalize_elapsed_column_df(df):
   """
   Normalize elapsed time column name to `elapsed_time_s` if the legacy
   `elapsed_time` column is present.

   :param df: pandas DataFrame
   :return: DataFrame with `elapsed_time_s` column
   """
   
   if "elapsed_time" in df.columns and "elapsed_time_s" not in df.columns: # If legacy column present and new column missing
      df["elapsed_time_s"] = df["elapsed_time"] # Copy legacy elapsed_time into the new elapsed_time_s column
      df.drop(columns=["elapsed_time"], inplace=True) # Remove the legacy elapsed_time column to avoid duplication
   return df # Return DataFrame with normalized elapsed time column

def load_existing_results(csv_out):
   """
   Load an existing consolidated CSV if present, returning an empty
   DataFrame on error or when file is missing.

   :param csv_out: path to consolidated CSV
   :return: pandas.DataFrame
   """
   
   if os.path.exists(csv_out): # If the consolidated CSV exists
      try: # Try to read the file into a DataFrame
         return pd.read_csv(csv_out, dtype=object) # Read CSV preserving types as object
      except Exception: # On any read error
         return pd.DataFrame() # Return empty DataFrame as a fallback
   return pd.DataFrame() # File not present — return empty DataFrame

def merge_replace_existing(df_existing, df_new):
   """
   Merge `df_new` into `df_existing` using replace-by-`dataset_path`
   semantics: any existing rows whose `dataset_path` appears in `df_new`
   are removed before appending `df_new`.

   :param df_existing: existing DataFrame (may be empty)
   :param df_new: incoming DataFrame with new rows
   :return: merged DataFrame
   """
   
   if not df_existing.empty and "dataset_path" in df_new.columns: # If existing rows present and incoming rows include dataset_path
      new_paths = df_new["dataset_path"].unique().tolist() # Compute unique dataset paths from incoming rows
      df_existing = df_existing[~df_existing["dataset_path"].isin(new_paths)] # Remove any existing rows that match those paths
   df_combined = pd.concat([df_existing, df_new], ignore_index=True, sort=False) # Concatenate remaining existing rows with new rows
   return df_combined # Return the merged DataFrame

def ensure_expected_columns(df_combined, columns):
   """
   Ensure the expected columns exist on the combined DataFrame; add
   missing columns with None values.

   :param df_combined: pandas.DataFrame
   :param columns: list of expected column names
   :return: DataFrame with ensured columns
   """
   
   for column in columns: # Iterate over expected column names
      if column not in df_combined.columns: # If the column is missing from the DataFrame
         df_combined[column] = None # Add the missing column and fill with None
   return df_combined # Return DataFrame with ensured columns

def sort_run_index_first(df_combined):
   """
   Sort by `dataset`, `dataset_path` and a numeric-coded `run_index`
   where the string "best" is forced to come before numeric runs.

   :param df_combined: pandas.DataFrame
   :return: sorted DataFrame
   """
   
   def run_index_sort(val): # Helper that maps run_index values to sortable integers
      try: # Try to normalize and parse the run index
         s = str(val).strip() # Convert the value to string and strip whitespace
         if s.lower() == "best": # If value is the literal 'best'
            return -1 # Force 'best' to sort before numeric indices
         return int(float(s)) # Convert numeric-like strings to integer for sorting
      except Exception: # On any parsing error
         return 10**9 # Use a very large number to push malformed values to the end

   df_combined["run_index_sort"] = df_combined["run_index"].apply(run_index_sort) # Create temporary numeric sort key
   df_combined.sort_values(by=["dataset", "dataset_path", "run_index_sort"], inplace=True, ascending=[True, True, True]) # Sort by dataset, path, then run order
   df_combined.drop(columns=["run_index_sort"], inplace=True) # Remove temporary sort key column
   return df_combined # Return the sorted DataFrame

def adjust_progress_for_early_stop(progress_state, n_generations, pop_size, gens_ran, folds):
   """
   Adjust `progress_state` when a GA run finishes early.

   This subtracts planned-but-not-executed classifier instantiations from
   `progress_state['total_it']` and increments `current_it` for the final
   re-evaluation (which is always performed once per run).

   :param progress_state: dict with keys `current_it` and `total_it`.
   :param n_generations: configured total generations for the run.
   :param pop_size: population size used for the run.
   :param gens_ran: number of generations actually executed.
   :param folds: number of CV folds (classifier instantiations per evaluation).
   :return: None (mutates `progress_state` in-place)
   """
   
   if not (progress_state and isinstance(progress_state, dict)): # Validate progress_state
      return # Nothing to do if invalid

   try: # Try to compute planned vs actual evaluations
      planned = int(n_generations) * int(pop_size) + 1 # Planned evaluations: generations * pop_size + 1 re-eval
      actual = int(gens_ran) * int(pop_size) + 1 # Actual evaluations: generations run * pop_size + 1 re-eval
      delta = max(0, planned - actual) # Number of generation-evaluations saved by early stopping
      progress_state["total_it"] = int(progress_state.get("total_it", 0)) - delta * folds # Reduce total iterations by saved evaluations
   except Exception: # Silently ignore failures during adjustment
      pass # Do nothing on error

   try: # Update current_it by the single re-evaluation performed after GA (folds classifiers)
      progress_state["current_it"] = int(progress_state.get("current_it", 0)) + folds # Increment current_it for final re-eval
   except Exception: # Silently ignore failures when updating current_it
      pass # Do nothing on error

def write_consolidated_csv(rows, output_dir):
   """
   Write the consolidated GA results rows to a CSV file inside `output_dir`.

   :param rows: List of dictionaries representing rows.
   :param output_dir: Directory where `Genetic_Algorithm_Results.csv` will be saved.
   :return: None
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Writing consolidated Genetic Algorithm results CSV.{Style.RESET_ALL}") # Output the verbose message
   
   try: # Try to write the consolidated CSV
      os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

      df_new = pd.DataFrame(rows) # Create DataFrame from provided rows

      df_new = normalize_elapsed_column_df(df_new) # Normalize legacy elapsed_time to elapsed_time_s in new rows

      csv_out = os.path.join(output_dir, "Genetic_Algorithm_Results.csv") # Build path for consolidated CSV

      df_existing = load_existing_results(csv_out) # Load existing consolidated CSV if available
      df_existing = normalize_elapsed_column_df(df_existing) # Normalize legacy elapsed_time to elapsed_time_s in existing rows

      if df_existing.empty: # If there is no existing consolidated CSV
         df_combined = df_new.copy() # Use the new DataFrame as the combined result
      else: # If existing consolidated CSV rows were loaded
         df_combined = merge_replace_existing(df_existing, df_new) # Merge new rows, replacing by dataset_path where needed

      columns = [ # Define canonical column order for the consolidated CSV
         "dataset",
         "dataset_path",
         "population_size",
         "n_generations",
         "n_train",
         "n_test",
         "train_test_ratio",
         "timestamp",
         "run_index",
         "classifier",
         "accuracy",
         "precision",
         "recall",
         "f1_score",
         "fpr",
         "fnr",
         "elapsed_time_s",
         "best_features",
         "rfe_ranking",
      ]

      df_combined = ensure_expected_columns(df_combined, columns) # Add any missing expected columns with None values
      df_combined = sort_run_index_first(df_combined) # Sort so that 'best' run_index appears first for each dataset/dataset_path

      df_combined = df_combined[columns] # Reorder columns into the canonical order
      df_combined.to_csv(csv_out, index=False, encoding="utf-8") # Persist the consolidated CSV to disk
      print(f"\n{BackgroundColors.GREEN}Genetic Algorithm consolidated results saved to {BackgroundColors.CYAN}{csv_out}{Style.RESET_ALL}") # Notify user of success
   except Exception as e:
      print(f"{BackgroundColors.RED}Failed to write consolidated GA CSV: {str(e)}{Style.RESET_ALL}") # Print failure message with exception

def prepare_feature_dataframe(X, feature_names):
   """
   Ensure features are available as a pandas DataFrame with appropriate column names.

   :param X: Feature matrix (DataFrame or numpy array).
   :param feature_names: Optional iterable of feature names.
   :return: pandas.DataFrame with feature columns.
   """
   
   if not isinstance(X, pd.DataFrame): # If X is not a pandas DataFrame
      try: # Try to create a DataFrame with original feature names
         df_features = pd.DataFrame(X, columns=list(feature_names)) # Create DataFrame with original feature names
      except Exception: # If creating DataFrame with original feature names fails
         df_features = pd.DataFrame(X) # Create DataFrame without original feature names
         df_features.columns = [f"feature_{i}" for i in range(df_features.shape[1])] # Generic feature names
   else: # If X is already a pandas DataFrame
      df_features = X.copy() # Use the DataFrame as is
      
   return df_features # Return the prepared DataFrame

def save_results(best_ind, feature_names, X, y, csv_path, metrics=None, X_test=None, y_test=None, n_generations=None, best_pop_size=None, runs_list=None):
   """
   Persist the GA best-result information to disk (consolidated CSV and auxiliary files).

   This function performs the saving responsibilities previously embedded inside
   `save_and_analyze_results`: it determines the selected features, extracts RFE
   rankings, optionally re-evaluates the best individual on a provided test set,
   builds the consolidated CSV rows and writes them to disk.

   :param best_ind: Best individual from the Genetic Algorithm (binary mask/list).
   :param feature_names: List of feature names corresponding to bits in `best_ind`.
   :param X: Feature set (DataFrame or numpy array) used during GA/training.
   :param y: Target variable (Series or array) used during GA/training.
   :param csv_path: Path to the original CSV file for saving outputs.
   :param metrics: Optional precomputed metrics tuple for the best individual.
   :param X_test: Optional test features to evaluate the best individual if `metrics` is None.
   :param y_test: Optional test labels to evaluate the best individual if `metrics` is None.
   :param n_generations: Number of GA generations used (for metadata only).
   :param best_pop_size: Population size that yielded the best result (for metadata only).
   :param runs_list: Optional list of per-run results (each a dict with keys 'metrics','best_features' or 'best_ind').
   :return: Dictionary with saved metadata: {
               "best_features": list,
               "rf_metrics": tuple or None,
               "output_dir": str,
               "rfe_ranking": dict,
               "n_train": int or None,
               "n_test": int or None,
               "test_frac": float or None,
               "n_generations": int or None,
               "best_pop_size": int or None,
               "runs_list": list or None
            }
   """

   best_features = [f for f, bit in zip(feature_names if feature_names is not None else [], best_ind) if bit == 1] # Extract best features
   rfe_ranking = extract_rfe_ranking(csv_path) # Extract RFE rankings

   print(f"\n{BackgroundColors.GREEN}Best features subset found: {BackgroundColors.CYAN}{best_features}{Style.RESET_ALL}")

   dataset_name = os.path.splitext(os.path.basename(csv_path))[0] # Get the base name of the dataset
   output_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis/" # Directory to save outputs
   os.makedirs(output_dir, exist_ok=True) # Create the directory if it doesn't exist

   rf_metrics = metrics if metrics is not None else None # Use provided metrics if available

   if rf_metrics is None and X_test is not None and y_test is not None: # If no metrics provided, evaluate on test set
      rf_metrics = evaluate_individual(best_ind, X, y, X_test, y_test) # Evaluate best individual

   n_train = len(y) if y is not None else None # Number of training samples
   n_test = len(y_test) if y_test is not None else None # Number of testing samples

   test_frac = None # Initialize train/test fraction
   if n_train is not None and n_test is not None and (n_train + n_test) > 0: # If both lengths are valid
      test_frac = float(n_test) / float(n_train + n_test) # Calculate train/test fraction

   base_row = build_base_row(csv_path, best_pop_size, n_generations, n_train, n_test, test_frac, rfe_ranking) # Base row for CSV
   rows = build_rows_list(rf_metrics, best_features, runs_list, feature_names, base_row) # Build rows from metrics and runs
   write_consolidated_csv(rows, output_dir) # Persist consolidated CSV

   return { # Return saved metadata
      "best_features": best_features,
      "rf_metrics": rf_metrics,
      "output_dir": output_dir,
      "rfe_ranking": rfe_ranking,
      "n_train": n_train,
      "n_test": n_test,
      "test_frac": test_frac,
      "n_generations": n_generations,
      "best_pop_size": best_pop_size,
      "runs_list": runs_list,
   }

def analyze_results(saved_info, X, y, feature_names, csv_path):
   """
   Analyze and visualize results that were previously saved by `save_results`.

   :param saved_info: Dictionary returned from `save_results` (must contain key "best_features").
   :param X: Feature set (DataFrame or numpy array) used during GA/training.
   :param y: Target variable (Series or array) used during GA/training.
   :param feature_names: List of original feature names used to construct the DataFrame.
   :param csv_path: Path to the original CSV file for saving outputs (used by analyzers).
   :return: None
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Analyzing saved results from Genetic Algorithm feature selection.{Style.RESET_ALL}") # Output the verbose message

   best_features = saved_info.get("best_features", []) if isinstance(saved_info, dict) else [] # Extract best features
   if not best_features: # Nothing to analyze
      return # Exit early

   df_features = prepare_feature_dataframe(X, feature_names) # Prepare DataFrame for analysis

   if not isinstance(y, pd.Series): # If y is not a pandas Series
      try: # Try to create a Series with original indices
         y_series = pd.Series(y, index=df_features.index) # Create Series with original indices
      except Exception: # If creating Series with original indices fails
         y_series = pd.Series(y) # Create Series without original indices
   else: # If y is already a pandas Series
      y_series = y.reindex(df_features.index) if not df_features.index.equals(y.index) else y # Align indices if necessary

   analyze_top_features(df_features, y_series, best_features, csv_path=csv_path) # Analyze and visualize the top features

def prepare_sweep_data(csv_path, dataset_name, min_pop, max_pop, n_generations):
   """
   Load and preprocess dataset for GA sweep.
   
   :param csv_path: Path to the CSV dataset.
   :param dataset_name: Name of the dataset.
   :param min_pop: Minimum population size.
   :param max_pop: Maximum population size.
   :param n_generations: Number of generations.
   :return: Tuple (X_train, X_test, y_train, y_test, feature_names) or None if failed.
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Preparing dataset '{dataset_name}' for Genetic Algorithm sweep.{Style.RESET_ALL}") # Output the verbose message
   
   df = load_dataset(csv_path) # Load dataset
   if df is None: # If loading failed
      return None # Exit early
   
   cleaned_df = preprocess_dataframe(df) # Preprocess dataset
   
   if cleaned_df is None or cleaned_df.empty: # If preprocessing failed or dataset is empty
      print(f"{BackgroundColors.RED}Dataset empty after preprocessing. Exiting.{Style.RESET_ALL}")
      return None # Exit early
   
   X_train, X_test, y_train, y_test, feature_names = split_dataset(cleaned_df, csv_path) # Split dataset
   if X_train is None: # If splitting failed
      return None # Exit early
   
   print_ga_parameters(min_pop, max_pop, n_generations, len(feature_names) if feature_names is not None else 0) if VERBOSE else None # Print GA parameters if verbose
   
   train_count = len(y_train) if y_train is not None else 0 # Count training samples
   test_count = len(y_test) if y_test is not None else 0 # Count testing samples
   verbose_output(f"  {BackgroundColors.GREEN}  Dataset: {BackgroundColors.CYAN}{dataset_name} - {train_count} training / {test_count} testing  (80/20){Style.RESET_ALL}\n")
   
   return X_train, X_test, y_train, y_test, feature_names # Return prepared data

def run_single_ga_iteration(bot, X_train, y_train, X_test, y_test, feature_names, pop_size, n_generations, run, runs, dataset_name, csv_path, max_pop, progress_bar, progress_state, folds):
   """
   Execute one GA run for a specific population size.
   
   :param bot: TelegramBot instance.
   :param X_train: Training features.
   :param y_train: Training labels.
   :param X_test: Test features.
   :param y_test: Test labels.
   :param feature_names: List of feature names.
   :param pop_size: Population size for this iteration.
   :param n_generations: Number of generations.
   :param run: Current run number (1-based).
   :param runs: Total number of runs.
   :param dataset_name: Dataset name.
   :param csv_path: Path to CSV.
   :param max_pop: Maximum population size.
   :param progress_bar: Progress bar instance.
   :param progress_state: Progress state dict.
   :param folds: Number of CV folds.
   :return: Dict with best_ind, metrics, best_features or None if failed.
   """

   iteration_start_time = time.time() # Start tracking total iteration time
   
   feature_count = len(feature_names) if feature_names is not None else 0 # Count of features
   
   update_progress_bar(progress_bar, dataset_name, csv_path, pop_size=pop_size, max_pop=max_pop, n_generations=n_generations, run=run, runs=runs, progress_state=progress_state) if progress_bar else None # Update progress bar if provided
   
   toolbox, population, hof = setup_genetic_algorithm(feature_count, pop_size) # Setup GA components
   best_ind, gens_ran = run_genetic_algorithm_loop(bot, toolbox, population, hof, X_train, y_train, X_test, y_test, n_generations, show_progress=False, progress_bar=progress_bar, dataset_name=dataset_name, csv_path=csv_path, pop_size=pop_size, max_pop=max_pop, run=run, runs=runs, progress_state=progress_state) # Run GA loop and get generations actually run
   
   if best_ind is None: # If GA failed
      return None # Exit early
   
   metrics = evaluate_individual(best_ind, X_train, y_train, X_test, y_test) # Evaluate best individual
   
   adjust_progress_for_early_stop(progress_state, n_generations, pop_size, gens_ran, folds) # Update progress_state in helper
   update_progress_bar(progress_bar, dataset_name, csv_path, pop_size=pop_size, max_pop=max_pop, n_generations=n_generations, run=run, runs=runs, progress_state=progress_state) if progress_bar else None # Update progress bar
   
   best_features = [f for f, bit in zip(feature_names if feature_names is not None else [], best_ind) if bit == 1] # Extract best features
   
   iteration_elapsed_time = time.time() - iteration_start_time # Calculate total iteration time
   metrics_with_iteration_time = metrics + (iteration_elapsed_time,) # Add total iteration time as 7th element
   
   return {"best_ind": best_ind, "metrics": metrics_with_iteration_time, "best_features": best_features} # Return results

def aggregate_sweep_results(results, min_pop, max_pop, bot, dataset_name):
   """
   Aggregate results per population size and find best.
   
   :param results: Dict mapping pop_size to runs list.
   :param min_pop: Minimum population size.
   :param max_pop: Maximum population size.
   :param bot: TelegramBot instance.
   :param dataset_name: Dataset name.
   :return: Tuple (best_score, best_result, best_metrics, results_dict).
   """
   
   best_score = -1 # Initialize best score
   best_result = None # Initialize best result
   best_metrics = None # Initialize best metrics
   
   for pop_size in range(min_pop, max_pop + 1): # For each population size
      runs_list = results[pop_size]["runs"] # Get runs list
      if not runs_list: # If no runs
         results.pop(pop_size, None) # Remove entry
         continue # Skip to next population size
      
      all_metrics = [r["metrics"] for r in runs_list] # Collect all metrics
      avg_metrics = tuple(np.mean(all_metrics, axis=0)) # Compute average metrics
      
      feature_sets = [set(r["best_features"]) for r in runs_list] # Collect feature sets
      common_features = set.intersection(*feature_sets) if feature_sets else set() # Find common features
      
      results[pop_size]["avg_metrics"] = avg_metrics # Store average metrics
      results[pop_size]["common_features"] = common_features # Store common features
      
      f1_avg = avg_metrics[3] # Average F1-Score
      if f1_avg > best_score: # If this is the best score so far
         best_score = f1_avg # Update best score
         best_metrics = avg_metrics # Update best metrics
         best_result = (pop_size, runs_list, common_features) # Update best result
      
      print(f"{BackgroundColors.GREEN}Pop {BackgroundColors.CYAN}{pop_size}{BackgroundColors.GREEN}: AVG F1 {BackgroundColors.GREEN}{f1_avg:.4f}{BackgroundColors.GREEN}, Common Features {BackgroundColors.CYAN}{len(common_features)}{Style.RESET_ALL}") # Print summary
      for i, run_data in enumerate(runs_list): # For each run
         unique = set(run_data["best_features"]) - common_features # Unique features
         print(f"  {BackgroundColors.GREEN}Run {BackgroundColors.CYAN}{i+1}{BackgroundColors.GREEN}: {BackgroundColors.GREEN}unique features {BackgroundColors.CYAN}{len(unique)}{Style.RESET_ALL}") # Print unique features count
      
      if bot.TELEGRAM_BOT_TOKEN and bot.CHAT_ID: # If Telegram is configured
         try: # Try to send message
            bot.send_messages([f"Completed {len(runs_list)} runs for population size **{pop_size}** on **{dataset_name}** -> **Avg F1: {f1_avg:.4f}**"]) # Send progress message
         except (Exception, RuntimeWarning): # Silently ignore Telegram errors and warnings
            pass # Do nothing
   
   return best_score, best_result, best_metrics, results # Return aggregated results

def run_population_sweep(bot, dataset_name, csv_path, n_generations=100, min_pop=20, max_pop=20, runs=RUNS, progress_bar=None):
   """
   Executes a genetic algorithm (GA) for feature selection across multiple population sizes and runs.

   This function performs a "population sweep," testing different population sizes
   to identify the set of features that maximizes classification performance
   (F1-Score) on the training dataset using 10-fold Stratified Cross-Validation.
   For each population size, it runs the GA multiple times to verify for divergence.

   :param bot: TelegramBot instance for sending notifications.
   :param dataset_name: Name of the dataset being processed.
   :param csv_path: Path to the CSV dataset.
   :param n_generations: Number of generations to run the GA for each population size.
   :param min_pop: Minimum population size to test.
   :param max_pop: Maximum population size to test.
   :param runs: Number of runs for each population size.
   :param progress_bar: Optional tqdm progress bar instance to update with progress.
   :return: Dictionary mapping population sizes to their results including runs and divergence.
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Starting population sweep for dataset {BackgroundColors.CYAN}{dataset_name}{BackgroundColors.GREEN} from size {min_pop} to {max_pop}, running {n_generations} generations and {runs} runs each.{Style.RESET_ALL}")

   if bot.TELEGRAM_BOT_TOKEN and bot.CHAT_ID: # If Telegram is configured
      try: # Try to send message
         bot.send_messages([f"Starting population sweep for dataset **{dataset_name}** from size **{min_pop}** to **{max_pop}**"]) # Send start message
      except (Exception, RuntimeWarning): # Silently ignore Telegram errors and warnings
         pass # Do nothing

   data = prepare_sweep_data(csv_path, dataset_name, min_pop, max_pop, n_generations) # Prepare dataset
   if data is None: # If preparation failed
      return {} # Exit early
   
   X_train, X_test, y_train, y_test, feature_names = data # Unpack prepared data
   
   folds = 10 # Number of CV folds
   progress_state = compute_progress_state(min_pop, max_pop, n_generations, runs, progress_bar, folds=folds) # Compute progress state for tracking
   
   results = {} # Dictionary to hold results per population size
   for p in range(min_pop, max_pop + 1): # For each population size
      results[p] = {"runs": [], "avg_metrics": None, "common_features": set()} # Initialize results entry
   
   for run in range(runs): # For each run
      for pop_size in range(min_pop, max_pop + 1): # For each population size
         result = run_single_ga_iteration(bot, X_train, y_train, X_test, y_test, feature_names, pop_size, n_generations, run+1, runs, dataset_name, csv_path, max_pop, progress_bar, progress_state, folds) # Run GA iteration
         if result: # If result is valid
            results[pop_size]["runs"].append(result) # Append result to runs list
   
   best_score, best_result, best_metrics, results = aggregate_sweep_results(results, min_pop, max_pop, bot, dataset_name) # Aggregate results and find best

   if best_result: # If a best result was found
      best_pop_size, runs_list, common_features = best_result # Unpack the best result
      print(f"\n{BackgroundColors.GREEN}Best population size: {BackgroundColors.CYAN}{best_pop_size}{Style.RESET_ALL}")
      print(f"{BackgroundColors.GREEN}Common features across runs: {BackgroundColors.CYAN}{len(common_features)}{Style.RESET_ALL}")
      print_metrics(best_metrics) if VERBOSE else None # Print metrics if VERBOSE is enabled
      best_run = max(runs_list, key=lambda r: r["metrics"][3]) # Select the run with the best F1-Score
      best_ind = best_run["best_ind"] # Get the best individual from the best run
      best_metrics = best_run["metrics"] # Get the metrics from the best run
      saved = save_results(best_ind, feature_names, X_train, y_train, csv_path, metrics=best_metrics, X_test=X_test, y_test=y_test, n_generations=n_generations, best_pop_size=best_pop_size, runs_list=runs_list) # Save the best results
      analyze_results(saved, X_train, y_train, feature_names, csv_path) # Analyze the saved results
   else: # If no valid result was found
      print(f"{BackgroundColors.RED}No valid results found during the sweep.{Style.RESET_ALL}")

   if bot.TELEGRAM_BOT_TOKEN and bot.CHAT_ID: # If Telegram is configured
      try: # Try to send message
         bot.send_messages([f"Population sweep completed for **{dataset_name}**"]) # Send completion message
      except (Exception, RuntimeWarning): # Silently ignore Telegram errors and warnings
         pass # Do nothing

   return results # Return the results dictionary

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

   :return: None
   :return: None
   """

   warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited.*") # Suppress RuntimeWarning for unawaited coroutines from Telegram bot

   print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Genetic Algorithm Feature Selection{BackgroundColors.GREEN} program!{Style.RESET_ALL}", end="\n\n") # Output the welcome message
   start_time = datetime.datetime.now() # Get the start time of the program

   input_path = "./Datasets/CICDDoS2019/01-12/" # Path to the input dataset directory
   files_to_process = get_files_to_process(input_path, file_extension=".csv") # Get list of CSV files to process
   
   dataset_name = get_dataset_name(input_path) # Get the dataset name from the input path
   
   bot = TelegramBot() # Initialize Telegram bot for notifications

   progress_bar = tqdm(files_to_process, desc=f"{BackgroundColors.GREEN}Datasets{Style.RESET_ALL}", unit="file") # Progress bar for files to process
   for file in progress_bar: # For each file to process
      update_progress_bar(progress_bar, dataset_name, file) # Update the description to show the dataset and filename consistently

      sweep_results = run_population_sweep(bot, dataset_name, file, n_generations=100, min_pop=20, max_pop=20, runs=RUNS, progress_bar=progress_bar) # Run population sweep

      if VERBOSE and sweep_results: # If VERBOSE is True and there are results
         print(f"\n{BackgroundColors.GREEN}Detailed sweep results by population size:{Style.RESET_ALL}") # Print detailed results
         for pop_size, features in sweep_results.items(): # For each population size and its best features
            print(f"  Pop {pop_size}: {len(features)} features -> {features}") # Print the population size and the best features
   
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
