"""
================================================================================
Classifiers Hyperparameter Optimization
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-12-08
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
   - Configure `DATASETS` or edit `main()` to point to dataset directories.
   - Run: `python hyperparameters_optimization.py` (or integrate from other code).

Outputs:
   - Classifiers_Hyperparameters/<dataset>_Hyperparameter_Optimization_Results.csv
     containing best parameters, best CV F1 and timing for each model tested.

TODOs:
   - Add `argparse` to control dataset selection, CV folds, and search strategy
   - Add randomized/Bayesian search alternatives for large parameter grids
   - Improve resumability for long-running searches and better exception traces

Dependencies:
   - Python >= 3.8
   - pandas, numpy, scikit-learn, xgboost, lightgbm, tqdm, colorama
   - psutil (optional, used for hardware reporting)

Assumptions & Notes:
   - Input CSV: last column is the target, numeric features only are used
   - Genetic Algorithm results must be present under `Feature_Analysis/`
   - Outputs are written next to each processed dataset directory
"""

import atexit # For playing a sound when the program finishes
import concurrent.futures # For parallel execution with progress updates
import datetime # For getting the current date and time
import json # For handling JSON strings
import lightgbm as lgb # For LightGBM model
import numpy as np # For numerical operations
import os # For running a command in the terminal
import pandas as pd # For data manipulation
import platform # For getting the operating system name
import psutil # RAM and CPU core info
import shutil # For file operations
import statistics # For calculating statistics
import subprocess # WMIC call
import sys # For system-specific parameters and functions
import tempfile # For creating temporary files and directories
import threading # For threading operations
import time # For measuring execution time
import warnings # For suppressing warnings
from collections import OrderedDict # For deterministic results column ordering when saving
from colorama import Style # For coloring the terminal
from itertools import product # For generating parameter combinations
from joblib import Parallel, delayed # For parallel processing of parameter combinations
from Logger import Logger # For logging output to both terminal and file
from pathlib import Path # For handling file paths
from sklearn.base import clone # Import necessary modules for cloning
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier # For ensemble models
from sklearn.linear_model import LogisticRegression # For logistic regression model
from sklearn.metrics import make_scorer, f1_score # For custom scoring metrics
from sklearn.model_selection import GridSearchCV, train_test_split # For hyperparameter search and data splitting
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid # For k-nearest neighbors model
from sklearn.neural_network import MLPClassifier # For neural network model
from sklearn.preprocessing import LabelEncoder, StandardScaler # For label encoding and feature scaling
from sklearn.svm import SVC # For Support Vector Machine model
try: # Attempt to import ThunderSVM
   from thundersvm import SVC as ThunderSVC # For ThunderSVM classifier
   THUNDERSVM_AVAILABLE = True # Flag indicating ThunderSVM is available
except Exception as _th_err: # Import failed
   ThunderSVC = None # ThunderSVM not available
   THUNDERSVM_AVAILABLE = False
   print(f"Warning: ThunderSVM import failed ({type(_th_err).__name__}: {_th_err}). Falling back to sklearn.SVC.")
from tqdm import tqdm # For progress bars
from xgboost import XGBClassifier # For XGBoost classifier

# Warnings:
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning) # Ignore pandas dtype warnings

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
N_JOBS = -2 # Number of parallel jobs (-1 uses all cores, -2 leaves one core free, or set specific number like 4)
MAX_PARALLEL_MEMORY = '1G' # Maximum memory per joblib worker (e.g., '500M', '1G') to prevent excessive RAM usage
PRE_DISPATCH = '2*n_jobs' # Number of batches to pre-dispatch to workers (controls memory usage)
RESULTS_FILENAME = "Hyperparameter_Optimization_Results.csv" # Filename for saving results
MATCH_FILENAMES_TO_PROCESS = [""] # List of specific filenames to search for a match (set to None to process all files)
IGNORE_FILES = [RESULTS_FILENAME] # List of filenames to ignore when searching for datasets
IGNORE_DIRS = ["Cache", "Dataset_Description", "Data_Separability", "Feature_Analysis"] # List of directory names to ignore when searching for datasets

DATASETS = { # Dictionary containing dataset paths and feature files
	"CICDDoS2019-Dataset": [ # List of paths to the CICDDoS2019 dataset
		"./Datasets/CICDDoS2019/01-12/",
		"./Datasets/CICDDoS2019/03-11/",
   ],
}

# Logger Setup:
logger = Logger(f"./Logs/{Path(__file__).stem}.log", clean=True) # Create a Logger instance
sys.stdout = logger # Redirect stdout to the logger
sys.stderr = logger # Redirect stderr to the logger

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

def iterate_dataset_directories():
   """
   Iterates over all dataset directories defined in DATASETS, skipping invalid and ignored directories.

   :param: None
   :return: Generator yielding (dataset_name, dirpath)
   """
   
   for dataset_name, paths in DATASETS.items(): # Iterate over datasets
      for dirpath in paths: # Iterate configured paths
         if not os.path.isdir(dirpath): # If path is not a directory
            verbose_output(f"{BackgroundColors.YELLOW}Skipping non-directory path: {BackgroundColors.CYAN}{dirpath}{Style.RESET_ALL}") # Verbose notice
            continue # Skip invalid path
         if os.path.basename(os.path.normpath(dirpath)) in IGNORE_DIRS: # If path is in ignore list
            verbose_output(f"{BackgroundColors.YELLOW}Ignoring directory per IGNORE_DIRS: {BackgroundColors.CYAN}{dirpath}{Style.RESET_ALL}") # Verbose notice
            continue # Skip ignored directory
         yield dataset_name, dirpath # Yield valid directory

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
   Collect all files with a given extension inside a directory (non-recursive).

   Performs validation, respects IGNORE_FILES, and optionally filters by
   MATCH_FILENAMES_TO_PROCESS when defined.

   :param directory_path: Path to the directory to scan
   :param file_extension: File extension to include (default: ".csv")
   :return: Sorted list of matching file paths
   """

   verbose_output(f"{BackgroundColors.GREEN}Getting all {BackgroundColors.CYAN}{file_extension}{BackgroundColors.GREEN} files in: {BackgroundColors.CYAN}{directory_path}{Style.RESET_ALL}") # Verbose: starting file collection
   verify_filepath_exists(directory_path) # Validate directory path exists

   if not os.path.isdir(directory_path): # Check if path is a valid directory
      verbose_output(f"{BackgroundColors.RED}Not a directory: {BackgroundColors.CYAN}{directory_path}{Style.RESET_ALL}") # Verbose: invalid directory
      return [] # Return empty list for invalid paths

   try: # Attempt to read MATCH_FILENAMES_TO_PROCESS if defined
      match_names = set(MATCH_FILENAMES_TO_PROCESS) if MATCH_FILENAMES_TO_PROCESS not in ([], [""], [" "]) else None # Load match list or None
      if match_names: verbose_output(f"{BackgroundColors.GREEN}Filtering to filenames: {BackgroundColors.CYAN}{match_names}{Style.RESET_ALL}") # Verbose: applying filename filter
   except NameError: # MATCH_FILENAMES_TO_PROCESS not defined
      match_names = None # No filtering will be applied

   files = [] # Accumulator for valid files

   for item in os.listdir(directory_path): # Iterate directory entries
      item_path = os.path.join(directory_path, item) # Absolute path
      filename = os.path.basename(item_path) # Extract just the filename

      if any(ignore == filename or ignore == item_path for ignore in IGNORE_FILES): # Check if file is in ignore list
         verbose_output(f"{BackgroundColors.YELLOW}Ignoring {BackgroundColors.CYAN}{filename}{BackgroundColors.YELLOW} (listed in IGNORE_FILES){Style.RESET_ALL}") # Verbose: ignoring file
         continue # Skip ignored file

      if os.path.isfile(item_path) and item.lower().endswith(file_extension): # File matches extension requirement
         if match_names is not None and filename not in match_names: # Filename not included in MATCH_FILENAMES_TO_PROCESS
            verbose_output(f"{BackgroundColors.YELLOW}Skipping {BackgroundColors.CYAN}{filename}{BackgroundColors.YELLOW} (not in MATCH_FILENAMES_TO_PROCESS){Style.RESET_ALL}") # Verbose: skipping non-matching file
            continue # Skip this file
         files.append(item_path) # Add file to result list

   return sorted(files) # Return sorted list for deterministic output

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
   
   verbose_output(f"{BackgroundColors.GREEN}Extracting Genetic Algorithm selected features...{Style.RESET_ALL}") # Output the verbose message
   
   file_dir = os.path.dirname(file_path) # Determine the directory of the input file
   ga_results_path = os.path.join(file_dir, "Feature_Analysis", "Genetic_Algorithm_Results.csv") # Construct the path to the consolidated GA results file
   
   verbose_output(f"{BackgroundColors.GREEN}Extracting GA features for file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}") # Output the verbose message

   if not verify_filepath_exists(ga_results_path): # If the GA results file does not exist
      print(f"{BackgroundColors.YELLOW}GA results file not found: {BackgroundColors.CYAN}{ga_results_path}{Style.RESET_ALL}")
      return None # Return None if the file doesn't exist

   try: # Try to load and parse the GA results
      df = pd.read_csv(ga_results_path) # Load the CSV file into a DataFrame
      best_row = df[df["run_index"] == "best"] # Filter for the row with run_index == "best"
      if best_row.empty: # If no "best" row is found
         print(f"{BackgroundColors.YELLOW}No 'best' run_index found in {BackgroundColors.CYAN}{ga_results_path}{Style.RESET_ALL}")
         return None # Return None if the "best" row doesn't exist
      best_features_str = best_row.iloc[0]["best_features"] # Get the best_features column value (JSON string)
      best_features = json.loads(best_features_str) # Parse the JSON string into a Python list
      verbose_output(f"{BackgroundColors.GREEN}Loaded {BackgroundColors.CYAN}{len(best_features)}{BackgroundColors.GREEN} GA features{Style.RESET_ALL}")
      return best_features # Return the list of best features
   except IndexError: # If there's an issue accessing the row
      print(f"{BackgroundColors.RED}Error: Could not access 'best' row in {BackgroundColors.CYAN}{ga_results_path}{Style.RESET_ALL}")
      return None # Return None if there was an error
   except Exception as e: # Catch any other exceptions
      print(f"{BackgroundColors.RED}Error loading GA features: {e}{Style.RESET_ALL}")
      return None # Return None if there was an error

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

   df = pd.read_csv(csv_path, low_memory=True) # Load the dataset

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

def scale_and_split(X, y, test_size=0.2, random_state=42):
   """
   Scales the numeric features using StandardScaler and splits the data
   into training and testing sets.

   Note: The target variable 'y' is label-encoded before splitting.

   :param X: Features DataFrame (must contain numeric features).
   :param y: Target Series or array.
   :param test_size: Fraction of the data to reserve for the test set.
   :param random_state: Seed for the random split for reproducibility.
   :return: Tuple (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Scaling features and splitting data (train/test ratio: {BackgroundColors.CYAN}{1-test_size}/{test_size}{BackgroundColors.GREEN})...{Style.RESET_ALL}") # Output the verbose message
   
   le = LabelEncoder() # Initialize a LabelEncoder
   y_encoded = pd.Series(le.fit_transform(y), index=y.index) # Encode the target variable (essential for stratification)

   numeric_X = X.select_dtypes(include=np.number) # Select only numeric columns for scaling
   non_numeric_X = X.select_dtypes(exclude=np.number) # Identify non-numeric columns (to be dropped)
   
   if not non_numeric_X.empty: # If non-numeric columns were found
      print(f"{BackgroundColors.YELLOW}Warning: Dropping non-numeric feature columns for scaling: {BackgroundColors.CYAN}{list(non_numeric_X.columns)}{Style.RESET_ALL}") # Warn about dropped columns
       
   if numeric_X.empty: # If no numeric features remain
      raise ValueError(f"{BackgroundColors.RED}No numeric features found in X after filtering.{Style.RESET_ALL}") # Raise an error if X is empty

   X_train, X_test, y_train, y_test = train_test_split(numeric_X, y_encoded, test_size=test_size, random_state=random_state, stratify=y_encoded) # Split the data into training and testing sets with stratification
   
   scaler = StandardScaler() # Initialize the StandardScaler
   
   X_train_scaled = scaler.fit_transform(X_train) # Fit and transform the training features
   
   X_test_scaled = scaler.transform(X_test) # Transform the testing features

   verbose_output(f"{BackgroundColors.GREEN}Data split successful. Training set shape: {BackgroundColors.CYAN}{X_train_scaled.shape}{BackgroundColors.GREEN}. Testing set shape: {BackgroundColors.CYAN}{X_test_scaled.shape}{Style.RESET_ALL}") # Output the successful split message
   
   return X_train_scaled, X_test_scaled, y_train, y_test, scaler # Return scaled features, target, and the fitted scaler

def load_and_prepare_dataset(csv_path):
   """
   Loads, preprocesses, and prepares a dataset for model training and evaluation.

   :param csv_path: Path to the CSV dataset file
   :return: Tuple (X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names)
   """
   
   df = load_dataset(csv_path) # Load dataset
   if df is None: # If load failed
      print(f"{BackgroundColors.YELLOW}Failed to load dataset {csv_path}. Skipping file.{Style.RESET_ALL}")
      return None # Exit early

   df_clean = preprocess_dataframe(df) # Preprocess DataFrame
   if df_clean is None or df_clean.empty: # If preprocessing failed
      print(f"{BackgroundColors.YELLOW}Dataset preprocessing failed for {csv_path}. Skipping file.{Style.RESET_ALL}")
      return None # Exit early

   X = df_clean.iloc[:, :-1] # Features
   y = df_clean.iloc[:, -1] # Target

   print(f"{BackgroundColors.GREEN}Dataset loaded with {BackgroundColors.CYAN}{X.shape[0]}{BackgroundColors.GREEN} samples and {BackgroundColors.CYAN}{X.shape[1]}{BackgroundColors.GREEN} features{Style.RESET_ALL}") # Output dataset shape

   X_train_scaled, X_test_scaled, y_train, y_test, scaler = scale_and_split(X, y) # Split and scale

   feature_names = list(X.select_dtypes(include=np.number).columns) # Numeric feature names

   return X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names

def get_feature_subset(X_scaled, features, feature_names):
   """
   Returns a subset of features from the scaled feature set based on the provided feature names.
   
   :param X_scaled: Scaled features (numpy array).
   :param features: List of feature names to select.
   :param feature_names: List of all feature names corresponding to columns in X_scaled.
   :return: Numpy array containing only the selected features, or an empty array if features is None/empty.
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Selecting subset of features based on GA selection...{Style.RESET_ALL}") # Output the verbose message
   
   if features: # Only proceed if the list of selected features is NOT empty/None
      indices = [feature_names.index(f) for f in features if f in feature_names]
      return X_scaled[:, indices] # Return the subset of features
   else: # If no features are selected (or features is None)
      return np.empty((X_scaled.shape[0], 0)) # Return an empty array with correct number of rows

def detect_gpu_info():
   """
   Detect GPU brand/model using `nvidia-smi` (best-effort).

   :return: String with GPU brand/model (e.g., 'NVIDIA GeForce RTX 2080 Ti') or None if not detected
   """

   try: # Try to detect GPU info via nvidia-smi
      res = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False) # Run nvidia-smi to get GPU list
      if res.returncode == 0 and res.stdout.strip(): # If command succeeded and output is non-empty
         first = res.stdout.strip().splitlines()[0] # Get the first line of output
         if ":" in first: # If the line contains a colon
            info = first.split(":", 1)[1].strip() # Extract GPU info after the colon
            return info.split("(")[0].strip() # Clean up info (remove parentheses)
   except Exception: # Any error means no GPU info
      return None # GPU info is not available

   return None # GPU info is not available

def get_thundersvm_estimator():
   """
   Return a ThunderSVM estimator if available. Prioritize GPU when available
   (detected via `nvidia-smi`), otherwise configure ThunderSVM to use multiple
   CPU threads. Fall back to sklearn's SVC if ThunderSVM is not installed.
   """
   
   if not THUNDERSVM_AVAILABLE: # If ThunderSVM is not available
      gpu_info = detect_gpu_info() # Best-effort GPU brand/model detection

      if gpu_info: # If GPU info was detected
         print(f"{BackgroundColors.YELLOW}ThunderSVM not available; falling back to sklearn.SVC. GPU detected: {BackgroundColors.CYAN}{gpu_info}{Style.RESET_ALL}")
      else: # If no GPU info was detected
         print(f"{BackgroundColors.YELLOW}ThunderSVM not available; falling back to sklearn.SVC.{Style.RESET_ALL}")

      return SVC(random_state=42, probability=True) # Return sklearn's SVC as fallback

   gpu_available = False # Assume no GPU by default
   try: # Try to run nvidia-smi
      res = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False) # Run nvidia-smi to check for GPUs
      if res.returncode == 0 and res.stdout.strip(): # If command succeeded and output is non-empty
         gpu_available = True # GPU is available
   except Exception: # Any error means no GPU
      gpu_available = False # GPU is not available

   if gpu_available: # If a GPU is available
      try: # Attempt to create ThunderSVM with GPU
         clf = ThunderSVC(random_state=42, probability=True, gpu_id=0) # Try to use GPU with gpu_id=0
         verbose_output(f"{BackgroundColors.GREEN}Using ThunderSVM on GPU (gpu_id=0).{Style.RESET_ALL}") # Verbose message
         return clf # Return the GPU-enabled classifier
      except TypeError: # Constructor may not accept gpu_id; fall back to default constructor
         try: # Attempt default ThunderSVM constructor
            clf = ThunderSVC(random_state=42, probability=True) # Default constructor (may auto-detect GPU)
            verbose_output(f"{BackgroundColors.GREEN}Using ThunderSVM (GPU preferred) with default constructor.{Style.RESET_ALL}") # Verbose message
            return clf # Return the classifier
         except Exception: # Any error means fall back to CPU
            pass # Fall through to CPU handling

   cpu_threads = max(1, (os.cpu_count() or 2) - 1) # Use all but one CPU core if no GPU is available
   for param in ("n_jobs", "nthread", "nthreads", "nproc", "threads"): # Try common ThunderSVM CPU thread parameters
      try: # Attempt to create ThunderSVM with CPU threads
         clf = ThunderSVC(random_state=42, probability=True, **{param: cpu_threads}) # Set CPU threads
         verbose_output(f"{BackgroundColors.GREEN}Using ThunderSVM on CPU with {cpu_threads} threads ({param}).{Style.RESET_ALL}") # Verbose message
         return clf # Return the CPU-enabled classifier
      except Exception: # Any error means try next parameter
         continue # Try next parameter

   verbose_output(f"{BackgroundColors.YELLOW}Using ThunderSVM default CPU instantiation.{Style.RESET_ALL}") # Verbose message
   
   return ThunderSVC(random_state=42, probability=True) # Default CPU ThunderSVM

def get_models_and_param_grids():
   """
   Returns a dictionary of models with their corresponding hyperparameter grids for GridSearchCV.

   :param: None
   :return: Dictionary with model names as keys and tuples (model_instance, param_grid) as values
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Initializing models and parameter grids for hyperparameter optimization...{Style.RESET_ALL}") # Output the verbose message
   
   return { # Dictionary of models and their parameter grids
      "Random Forest": (
         RandomForestClassifier(random_state=42, n_jobs=N_JOBS), # Random Forest classifier
         {
            "n_estimators": [50, 100, 200], # Number of trees in the forest
            "max_depth": [None, 10, 20, 30], # Maximum depth of the tree
            "min_samples_split": [2, 5, 10], # Minimum number of samples required to split an internal node
            "min_samples_leaf": [1, 2, 4], # Minimum number of samples required to be at a leaf node
            "max_features": ["sqrt", "log2", None] # Number of features to consider when looking for the best split
         }
      ),
      "SVM": (
         get_thundersvm_estimator(), # ThunderSVM (GPU preferred) or fallback to sklearn SVC
         {
            "C": [0.1, 1, 10, 100], # Regularization parameter
            "kernel": ["linear", "rbf", "poly"], # Kernel type to be used in the algorithm
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1] # Kernel coefficient
         }
      ),
      "XGBoost": (
         XGBClassifier(eval_metric="mlogloss", random_state=42, n_jobs=N_JOBS), # XGBoost classifier
         {
            "n_estimators": [50, 100, 200], # Number of trees in the forest
            "max_depth": [3, 5, 7, 10], # Maximum depth of the tree
            "learning_rate": [0.01, 0.1, 0.3], # Step size shrinkage
            "subsample": [0.6, 0.8, 1.0], # Subsample ratio of the training instances
            "colsample_bytree": [0.6, 0.8, 1.0] # Subsample ratio of columns when constructing each tree
         }
      ),
      "Logistic Regression": (
         LogisticRegression(max_iter=1000, random_state=42, n_jobs=N_JOBS), # Logistic Regression classifier
         {
            "C": [0.001, 0.01, 0.1, 1, 10, 100], # Inverse of regularization strength
            "penalty": ["l1", "l2", "elasticnet", None], # Norm used in the penalization
            "solver": ["lbfgs", "liblinear", "saga"], # Algorithm to use in the optimization problem
            "l1_ratio": [0.0, 0.5, 1.0] # The Elastic-Net mixing parameter
         }
      ),
      "KNN": (
         KNeighborsClassifier(n_jobs=N_JOBS), # K-Nearest Neighbors classifier
         {
            "n_neighbors": [3, 5, 7, 9, 11], # Number of neighbors to use
            "weights": ["uniform", "distance"], # Weight function used in prediction
            "metric": ["euclidean", "manhattan", "minkowski"], # Distance metric
            "p": [1, 2] # Power parameter for the Minkowski metric
         }
      ),
      "Nearest Centroid": (
         NearestCentroid(), # Nearest Centroid classifier
         {
            "metric": ["euclidean", "manhattan"], # Distance metric
            "shrink_threshold": [None, 0.1, 0.5, 1.0, 2.0] # Threshold for shrinking centroids
         }
      ),
      "Gradient Boosting": (
         GradientBoostingClassifier(random_state=42), # Gradient Boosting classifier
         {
            "n_estimators": [50, 100, 200], # Number of boosting stages to be run
            "learning_rate": [0.01, 0.1, 0.3], # Learning rate shrinks the contribution of each tree
            "max_depth": [3, 5, 7], # Maximum depth of the individual regression estimators
            "min_samples_split": [2, 5, 10], # Minimum number of samples required to split an internal node
            "min_samples_leaf": [1, 2, 4], # Minimum number of samples required to be at a leaf node
            "subsample": [0.6, 0.8, 1.0] # Subsample ratio of the training instances
         }
      ),
      "LightGBM": (
         lgb.LGBMClassifier(force_row_wise=True, random_state=42, verbosity=-1, n_jobs=N_JOBS), # LightGBM classifier
         {
            "n_estimators": [50, 100, 200], # Number of boosting stages to be run
            "max_depth": [3, 5, 7, 10, -1], # Maximum depth of the tree (-1 means no limit)
            "learning_rate": [0.01, 0.1, 0.3], # Step size shrinkage
            "num_leaves": [15, 31, 63], # Number of leaves in one tree
            "min_child_samples": [10, 20, 30], # Minimum number of data needed in a child (leaf)
            "subsample": [0.6, 0.8, 1.0] # Subsample ratio of the training instances
         }
      ),
      "MLP (Neural Net)": (
         MLPClassifier(max_iter=500, random_state=42), # Multi-layer Perceptron classifier
         {
            "hidden_layer_sizes": [(50,), (100,), (100, 50), (100, 100)], # Number of neurons in the hidden layers
            "activation": ["relu", "tanh", "logistic"], # Activation function for the hidden layer
            "solver": ["adam", "sgd"], # The solver for weight optimization
            "alpha": [0.0001, 0.001, 0.01], # L2 penalty (regularization term) parameter
            "learning_rate": ["constant", "adaptive"] # Learning rate schedule for weight updates
         }
      )
   }

def compute_total_param_combinations(models):
   """
   Computes the total number of hyperparameter combinations for all models
   and returns both the total count and a per-model combination dictionary.

   :param models: List of (model_name, (model_instance, param_grid))
   :return: Tuple (total_combinations_all_models, model_combinations_counts)
   """

   total_combinations_all_models = 0 # Initialize total combinations counter
   model_combinations_counts = {} # Store per-model combination counts

   for model_name, (model, param_grid) in models: # Iterate models
      if param_grid: # If there is a param grid
         keys = list(param_grid.keys()) # Parameter names
         values = [v if isinstance(v, (list, tuple)) else [v] for v in param_grid.values()] # Ensure lists
         count = len(list(product(*values))) # Count combinations
      else: # No hyperparameters
         count = 1 # Single combination

      model_combinations_counts[model_name] = count # Store per-model count
      total_combinations_all_models += count # Add to total

   return total_combinations_all_models, model_combinations_counts # Return total and per-model counts

def measure_resource_usage_for_combination(model, keys, combination, X, y, sample_interval=0.05):
	"""
	Run a single parameter combination and sample memory and CPU usage
	in a background thread. Returns memory delta (bytes), average CPU percent, and elapsed time.
	Memory delta = peak_mem_during_training - baseline_mem_before_training
	"""
	
	proc = psutil.Process(os.getpid()) # Current process
	baseline_mem = proc.memory_info().rss # Baseline memory before training
	mem_samples = [] # Memory usage samples during training
	cpu_samples = [] # CPU usage samples
	stop_evt = threading.Event() # Event to stop monitoring thread

	def monitor(): # Monitoring thread function
		try: # Try to monitor resources
			psutil.cpu_percent(interval=None) # Initialize CPU percent measurement
			time.sleep(0.01) # Small delay for CPU measurement initialization
			while not stop_evt.is_set(): # Loop until stop event is set
				try: # Try to sample memory and CPU
					mem_samples.append(proc.memory_info().rss) # Sample memory usage (RSS)
					cpu_samples.append(psutil.cpu_percent(interval=sample_interval)) # Sample CPU percent
				except Exception: # Catch sampling errors
					break # Exit monitoring loop on error
		except Exception: # Catch initialization errors
			return # Exit monitoring thread on error

	t = threading.Thread(target=monitor, daemon=True) # Start monitoring thread
	t.start() # Start the thread

	start = time.time() # Start timing
	try: # Try to run the model with the parameter combination
		model.set_params(**dict(zip(keys, combination))) # Apply hyperparameters
		model.fit(X, y) # Train model
	except Exception: # Catch any errors during training
		pass # Ignore errors for resource measurement
	finally: # Ensure monitoring thread is stopped
		stop_evt.set() # Signal monitoring thread to stop
		t.join(timeout=1.0) # Wait for thread to finish

	elapsed = time.time() - start # Measure execution time
	peak_mem = max(mem_samples) if mem_samples else proc.memory_info().rss # Peak memory during training
	mem_delta = max(0, peak_mem - baseline_mem) # Memory increase during training (non-negative)
	avg_cpu = statistics.mean(cpu_samples) if cpu_samples else 0.0 # Average CPU percent
	return mem_delta, avg_cpu, elapsed # Return memory delta, CPU percent, and elapsed time

def evaluate_single_combination(model, keys, combination, X_train, y_train):
   """
   Helper function to evaluate a single parameter combination.
   Designed to be called in parallel via joblib with memory safety.

   :param model: Clone of the model instance
   :param keys: Parameter names
   :param combination: Parameter values for this combination
   :param X_train: Training features
   :param y_train: Training labels
   :return: Tuple (current_params, score, elapsed)
   """
   
   current_params = dict(zip(keys, combination)) # Build dict of current params
   start_time = time.time() # Start timing
   
   try: # Try to train and evaluate
      model.set_params(**current_params) # Apply hyperparameters
      model.fit(X_train, y_train) # Train model
      y_pred = model.predict(X_train) # Predict on training set
      score = f1_score(y_train, y_pred, average="weighted") # Compute weighted F1 score
   except MemoryError: # Catch memory errors specifically
      print(f"{BackgroundColors.RED}MemoryError with params {current_params}. Consider reducing dataset size or n_jobs.{Style.RESET_ALL}")
      score = None # Mark score as None
   except Exception as e: # Catch any other errors during training/evaluation
      score = None # Mark score as None
   
   elapsed = time.time() - start_time # Measure execution time
   
   return current_params, score, elapsed # Return results

def evaluate_single_combination_from_files(model, keys, combination, X_path, y_path):
   """
   Worker wrapper that loads X/y from disk using mmap and calls
   `evaluate_single_combination`. This avoids copying large arrays
   into each worker process when using ProcessPoolExecutor.
   """
   
   try: # Try to load data with memory mapping
      X = np.load(X_path, mmap_mode="r") # Load features
      y = np.load(y_path, mmap_mode="r") # Load labels
   except Exception: # Catch any loading errors
      current_params = dict(zip(keys, combination)) # Build dict of current params
      return current_params, None, 0.0 # Return failure result

   return evaluate_single_combination(model, keys, combination, X, y) # Call evaluation function

def update_optimization_progress_bar(progress_bar, csv_path, model_name, param_grid=None, combo_current=None, combo_total=None, current=None, total_models=None, total_combinations=None, overall=None):
   """
   Updates a tqdm progress bar during hyperparameter optimization.

   Shows dataset reference, model name, progress index, and an optional compact
   summary of hyperparameters.

   :param progress_bar: tqdm progress bar instance
   :param csv_path: Path to dataset CSV
   :param model_name: Name of the model being optimized
   :param param_grid: Optional hyperparameter dictionary or summary
   :param combo_current: Current hyperparameter combination index (1-based)
   :param combo_total: Total hyperparameter combinations for the current model
   :param current: Current model index (1-based)
   :param total_models: Total number of models being optimized
   :param total_combinations: Total number of hyperparameter combinations across all models
   :return: None
   """

   if progress_bar is None: return # Nothing to update if progress bar is None

   try: # Protect against unexpected errors
      csv_name = os.path.basename(csv_path) # Extract CSV filename
      parent_dir = os.path.basename(os.path.dirname(csv_path)) # Extract parent directory
      if parent_dir and parent_dir.lower() != csv_name.lower(): # Parent differs from filename
         dataset_ref = f"{BackgroundColors.CYAN}{parent_dir}/{csv_name}{BackgroundColors.GREEN}" # Show parent/filename
      else: # Parent same as filename or empty
         dataset_ref = f"{BackgroundColors.CYAN}{csv_name}{BackgroundColors.GREEN}" # Show only filename

      idx_str = f"{BackgroundColors.GREEN}[{BackgroundColors.CYAN}{current}/{total_models}{BackgroundColors.GREEN}]" if current is not None and total_models is not None else "" # Model index string

      if combo_current is not None and combo_total is not None: # If combination indices are provided
         combo_str = f" {BackgroundColors.GREEN}{{{BackgroundColors.CYAN}{combo_current}/{combo_total}{BackgroundColors.GREEN}}}" # Combination index string
      else: # No combination indices
         combo_str = "" # Empty combination string

      desc = f"{BackgroundColors.GREEN}Dataset: {dataset_ref}{BackgroundColors.GREEN} - {idx_str} {BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN}:{combo_str}" # Base description

      def _short(value, limit=30): return str(value) if len(str(value)) <= limit else str(value)[:limit-3] + "..." # Helper to truncate strings

      if isinstance(param_grid, dict): # If param_grid is a dictionary
         parts = [] # Collect formatted hyperparameters
         for i, (k, v) in enumerate(param_grid.items()): # Iterate over hyperparameters
            if i >= 4: break # Limit to 4 params for compactness
            try: # Try to format the hyperparameter
               vals = list(v) if hasattr(v, '__iter__') and not isinstance(v, (str, bytes, dict)) else [v] # Convert iterable values to list
               shown = ",".join([_short(x, 12) for x in vals[:4]]) # Show up to 4 values per hyperparameter
               if len(vals) > 4: shown += f",+{len(vals)-4}" # Indicate remaining values
               parts.append(f"{BackgroundColors.GREEN}{_short(k,18)}{BackgroundColors.GREEN}:[{BackgroundColors.CYAN}{shown}{BackgroundColors.GREEN}]") # Append formatted hyperparameter
            except Exception: # Fallback on error
               parts.append(f"{BackgroundColors.GREEN}{_short(k,18)}{BackgroundColors.GREEN}:[{BackgroundColors.CYAN}{_short(v,12)}{BackgroundColors.GREEN}]") # Fallback formatting
         remaining = max(0, len(param_grid)-4) # Count remaining parameters
         param_display = ", ".join(parts) # Join formatted parameters
         if remaining > 0: param_display += f", {BackgroundColors.CYAN}+{remaining} more{BackgroundColors.GREEN}" # Show remaining count
      else: # If param_grid is not a dictionary
         param_display = _short(param_grid, 60) # Truncate string or other type

      desc = f"{desc} {BackgroundColors.GREEN}({param_display}){Style.RESET_ALL}" # Append parameter display to description

      progress_bar.set_description(desc) # Update progress bar description
      progress_bar.n = overall if overall is not None else (current or getattr(progress_bar, 'n', 0))
      progress_bar.total = total_combinations # Ensure total is correct
      progress_bar.refresh() # Force refresh of the progress bar

   except Exception: pass # Silently ignore any errors during update

def manual_grid_search(model_name, model, param_grid, X_train, y_train, progress_bar=None, csv_path=None, global_counter_start=0, total_combinations_all_models=None, model_index=None, total_models=None):
   """
   Performs manual grid search hyperparameter optimization with integrated progress bar.
   Uses parallel processing via joblib to evaluate parameter combinations simultaneously,
   significantly speeding up optimization for all classifiers.

   Updates the progress bar description and counter for each parameter combination
   tested, showing both the current combination index of this model and the
   overall combination count across all models.

   :param model_name: Name of the model for logging
   :param model: Model instance to optimize
   :param param_grid: Dictionary of hyperparameters to search
   :param X_train: Training features
   :param y_train: Training labels
   :param progress_bar: Optional tqdm progress bar
   :param csv_path: Path to CSV for progress description
   :param global_counter_start: Starting counter of overall combination index
   :param total_combinations_all_models: Total number of parameter combinations across all models
   :param total_models: Total number of models being optimized
   :return: Tuple (best_params, best_score, all_results, global_counter_end)
   """

   verbose_output(f"{BackgroundColors.GREEN}Manually optimizing {BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN} using parallel processing...{Style.RESET_ALL}") # Output the verbose message

   if not param_grid: return None, None, None, global_counter_start # No hyperparameters to optimize

   keys = list(param_grid.keys()) # Parameter names
   values = [v if isinstance(v, (list, tuple)) else [v] for v in param_grid.values()] # Ensure values are lists
   param_combinations = list(product(*values)) # Cartesian product
   total_combinations = len(param_combinations) # Total number of combinations for this model

   best_score = -float("inf") # Initialize best score
   best_params = None # Initialize best parameters
   best_elapsed = 0.0 # Execution time for best parameters (seconds)
   all_results = [] # Store results for all combinations
   global_counter = global_counter_start # Initialize global counter
   
   available_memory_gb = psutil.virtual_memory().available / (1024**3) # Available RAM in GB
   data_size_gb = (X_train.nbytes + y_train.nbytes) / (1024**3) # Dataset size in GB
   
   estimated_memory_per_worker = data_size_gb * 1.2 # 1.20x for model overhead
   safe_n_jobs = max(1, min(abs(N_JOBS) if N_JOBS < 0 else N_JOBS, int(available_memory_gb / max(0.5, estimated_memory_per_worker)))) # Calculate safe n_jobs based on memory
   
   if N_JOBS == -2: # Leave one core free
      safe_n_jobs = min(safe_n_jobs, max(1, psutil.cpu_count(logical=False) - 1)) # Use all but one physical core
   elif N_JOBS == -1: # Use all cores but still respect memory limits
      safe_n_jobs = min(safe_n_jobs, psutil.cpu_count(logical=False)) # Use all physical cores
   
   verbose_output(f"{BackgroundColors.GREEN}Using {BackgroundColors.CYAN}{safe_n_jobs}{BackgroundColors.GREEN} parallel workers (Available RAM: {BackgroundColors.CYAN}{available_memory_gb:.1f}GB{BackgroundColors.GREEN}, Dataset: {BackgroundColors.CYAN}{data_size_gb:.2f}GB{BackgroundColors.GREEN}){Style.RESET_ALL}")
   
   try: # Benchmark one combination to refine n_jobs based on actual resource usage
      if len(param_combinations) > 1: # Only benchmark if multiple combinations exist
         verbose_output(f"{BackgroundColors.GREEN}Benchmarking One Parameter Combination to Estimate Resource Usage for {BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN}...{Style.RESET_ALL}")
         sample_combo = param_combinations[0] # Take the first combination as a sample
         mem_delta_bytes, avg_cpu_percent, sample_elapsed = measure_resource_usage_for_combination(clone(model), keys, sample_combo, X_train, y_train) # Measure resource usage

         per_worker_mem_gb = max(0.1, mem_delta_bytes / (1024**3)) * 1.05 # 5% safety margin

         cores_per_worker = max(0.25, avg_cpu_percent / 100.0) # Estimate cores per worker (min 0.25 = 25% of one core)

         total_cores = psutil.cpu_count(logical=True) or 1 # Use logical cores
         physical_cores = psutil.cpu_count(logical=False) or total_cores # Fallback to logical if physical unavailable

         if isinstance(N_JOBS, int) and N_JOBS > 0: # If N_JOBS is a positive integer
            configured_cpu_limit = min(N_JOBS, total_cores) # Limit to N_JOBS
         elif N_JOBS == -2: # Leave one logical core free
            configured_cpu_limit = max(1, total_cores - 1) # Use all but one logical core
         elif N_JOBS == -1: # Use all logical cores
            configured_cpu_limit = total_cores # Use all logical cores
         else: # Default behavior
            configured_cpu_limit = max(1, total_cores - 1) # Default: leave one core free

         try: # Calculate max workers based on configured CPU limit and per-worker core estimate
            max_workers_cpu = max(1, int(max(1, configured_cpu_limit) / cores_per_worker))
         except Exception: # Fallback on error
            max_workers_cpu = max(1, configured_cpu_limit) # Fallback to configured CPU limit

         usable_memory_gb = available_memory_gb * 0.90 # Reserve 10% for OS
         max_workers_mem = max(1, int(usable_memory_gb / per_worker_mem_gb)) # Max workers based on memory

         computed_safe = min(max_workers_cpu, max_workers_mem) # Initial safe n_jobs based on CPU and memory

         if isinstance(N_JOBS, int) and N_JOBS > 0: # If N_JOBS is a positive integer
            computed_safe = min(computed_safe, N_JOBS) # Limit to N_JOBS
         elif N_JOBS == -2: # Leave one logical core free
            computed_safe = min(computed_safe, max(1, total_cores - 1)) # Use all but one logical core
         elif N_JOBS == -1: # Use all cores but still respect memory limits
            computed_safe = min(computed_safe, total_cores) # Use all logical cores

         safe_n_jobs = max(1, int(computed_safe)) # Final safe n_jobs
         verbose_output(f"{BackgroundColors.GREEN}Measured Sample:\n  - Peak Memory Increase: {BackgroundColors.CYAN}{mem_delta_bytes/(1024**3):.2f} GB{BackgroundColors.GREEN} (Per-Worker Est.: {BackgroundColors.CYAN}{per_worker_mem_gb:.2f} GB{BackgroundColors.GREEN}, Incl. Safety)\n  - AVG CPU During Sample: {BackgroundColors.CYAN}{avg_cpu_percent:.1f}%{BackgroundColors.GREEN} → Cores/Worker: {BackgroundColors.CYAN}{cores_per_worker:.2f}{BackgroundColors.GREEN}\n  - System: Logical Cores {BackgroundColors.CYAN}{total_cores}{BackgroundColors.GREEN}, Usable RAM {BackgroundColors.CYAN}{usable_memory_gb:.1f} GB{BackgroundColors.GREEN}\n  - Max Workers (CPU / RAM): {BackgroundColors.CYAN}{max_workers_cpu}{BackgroundColors.GREEN} / {BackgroundColors.CYAN}{max_workers_mem}{BackgroundColors.GREEN} → Using {BackgroundColors.CYAN}{safe_n_jobs}{BackgroundColors.GREEN} Workers{Style.RESET_ALL}")
   except Exception as benchmark_err: # If benchmarking fails, continue with previous safe_n_jobs calculation
      verbose_output(f"{BackgroundColors.YELLOW}Benchmarking failed ({benchmark_err}), using initial estimate{Style.RESET_ALL}")
   tmp_dir = tempfile.mkdtemp(prefix="hpopt_") # Temporary directory for memory-mapped files
   X_path = os.path.join(tmp_dir, "X_train.npy") # Path for X_train
   y_path = os.path.join(tmp_dir, "y_train.npy") # Path for y_train
   try: # Ensure temporary files are cleaned up
      np.save(X_path, X_train) # Save X_train to disk
      np.save(y_path, y_train) # Save y_train to disk

      with concurrent.futures.ProcessPoolExecutor(max_workers=safe_n_jobs) as executor: # Use ProcessPoolExecutor for parallel evaluation
         future_to_params = {executor.submit(evaluate_single_combination_from_files, clone(model), keys, combination, X_path, y_path): combination for combination in param_combinations} # Submit all combinations
         local_counter = 0 # Local combination counter for this model

         for future in concurrent.futures.as_completed(future_to_params): # Iterate as each future completes
            try: # Try to get result
               current_params, score, elapsed = future.result() # Get result from future
            except Exception: # Catch any errors from the worker
               combo = future_to_params.get(future) # Get the combination that caused the error
               current_params = dict(zip(keys, combo)) if combo is not None else {} # Build current params dict
               score = None # Mark score as None
               elapsed = 0.0 # Mark elapsed as 0.0

            global_counter += 1 # Increment overall combination counter
            local_counter += 1 # Increment per-model combination counter

            if progress_bar is not None and csv_path is not None: # Update progress bar if available
               update_optimization_progress_bar(progress_bar, csv_path, model_name, param_grid=current_params, combo_current=local_counter, combo_total=total_combinations, current=model_index, total_combinations=total_combinations_all_models, total_models=total_models, overall=global_counter) # Update progress bar description
               try: # Safely update progress bar
                  progress_bar.update(1) # Increment progress bar
               except Exception: # Ignore progress bar update errors
                  pass # Ignore errors

            all_results.append(OrderedDict([("params", json.dumps(current_params)), ("score", score), ("execution_time", elapsed)])) # Store result

            if score is not None: # If score is valid
               current_best_elapsed = next((r["execution_time"] for r in all_results if r["score"] == best_score), float("inf")) if best_score != -float("inf") else float("inf") # Get elapsed time for current best score
               if (score > best_score) or (score == best_score and elapsed < current_best_elapsed): # Verify for new best (higher score or same score but faster)
                  best_score = score # Update best score
                  best_params = current_params # Update best parameters
                  best_elapsed = elapsed # Save execution time for best params
                  verbose_output(f"{BackgroundColors.GREEN}New best score: {BackgroundColors.CYAN}{best_score:.4f}{BackgroundColors.GREEN} with params: {BackgroundColors.CYAN}{best_params}{Style.RESET_ALL}") # Log new best
   finally: # Cleanup temporary directory
      try: # Remove temporary directory and files
         shutil.rmtree(tmp_dir) # Delete temporary directory
      except Exception: # Ignore cleanup errors
         pass # Ignore errors during cleanup

   return best_params, best_score, best_elapsed, all_results, global_counter # Return best results, elapsed for best, all combinations, and final global counter

def run_model_optimizations(models, csv_path, X_train_ga, y_train, dir_results_list):
   """
   Runs optimization for all configured ML models using a progress bar and manual grid search.

   :param models: List of (model_name, (model_instance, param_grid))
   :param csv_path: Path of the CSV file currently being processed
   :param X_train_ga: Training feature matrix generated by the Genetic Algorithm
   :param y_train: Training labels
   :param dir_results_list: Accumulator list for storing optimization results
   :return: None
   """

   total_combinations_all_models, model_combinations_counts = compute_total_param_combinations(models) # Compute total combinations

   verbose_output(f"{BackgroundColors.GREEN}Starting hyperparameter optimizations for {BackgroundColors.CYAN}{len(models)}{BackgroundColors.GREEN} models with a total of {BackgroundColors.CYAN}{total_combinations_all_models}{BackgroundColors.GREEN} parameter combinations...{Style.RESET_ALL}") # Output verbose message

   global_counter = 0 # Initialize global combination counter

   with tqdm(total=total_combinations_all_models, desc=f"{BackgroundColors.GREEN}Optimizing Models{Style.RESET_ALL}", unit="comb") as pbar: # Progress bar
      for model_index, (model_name, (model, param_grid)) in enumerate(models, start=1): # Iterate models with index
         best_params, best_score, best_elapsed, all_results, global_counter = manual_grid_search(model_name, model, param_grid, X_train_ga, y_train, progress_bar=pbar, csv_path=csv_path, global_counter_start=global_counter, total_combinations_all_models=total_combinations_all_models, model_index=model_index, total_models=len(models)) # Manual grid search instead of GridSearchCV

         if best_params is not None: # If optimization succeeded
            elapsed_time = float(best_elapsed or 0.0) # Elapsed time for best params
            dir_results_list.append(OrderedDict([ # Append only the best result
               ("base_csv", os.path.basename(csv_path)), # Base CSV filename
               ("model", model_name), # Model name
               ("best_params", json.dumps(best_params)), # Best parameters as JSON string
               ("best_cv_f1_score", best_score), # Best F1 score
               ("n_features", X_train_ga.shape[1]), # Number of features after GA selection
               ("feature_selection_method", "Genetic Algorithm"), # Feature selection method
               ("dataset", os.path.basename(csv_path)), # Dataset filename
               ("elapsed_time_s", round(float(elapsed_time), 2)) # Elapsed training time (seconds) for best params
            ])) # End of append

      print() # Line spacing between models

def process_single_csv_file(csv_path, dir_results_list):
   """
   Processes a single CSV file: loads GA-selected features, prepares the dataset,
   applies GA-based column filtering, and runs model hyperparameter optimization.

   :param csv_path: Path to dataset CSV file
   :param dir_results_list: List to store optimization results for the directory
   :return: None
   """
   
   print(f"{BackgroundColors.GREEN}\nProcessing file: {BackgroundColors.CYAN}{csv_path}{Style.RESET_ALL}") # Output the file being processed

   print(f"{BackgroundColors.GREEN}Loading Genetic Algorithm selected features...{Style.RESET_ALL}") # Output loading message
   ga_selected_features = extract_genetic_algorithm_features(csv_path) # Extract GA features
   if ga_selected_features is None or len(ga_selected_features) == 0: # If no GA features found
      print(f"{BackgroundColors.YELLOW}No GA features found for {csv_path}. Skipping file.{Style.RESET_ALL}")
      return # Exit early

   print(f"{BackgroundColors.GREEN}Loaded {BackgroundColors.CYAN}{len(ga_selected_features)}{BackgroundColors.GREEN} GA selected features{Style.RESET_ALL}") # Output feature count

   dataset_bundle = load_and_prepare_dataset(csv_path) # Load, preprocess, split, scale
   if dataset_bundle is None: # If loading/preprocessing failed
      return # Exit early

   X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = dataset_bundle # Unpack dataset bundle

   print(f"{BackgroundColors.GREEN}Applying GA feature selection...{Style.RESET_ALL}") # Output message
   X_train_ga = get_feature_subset(X_train_scaled, ga_selected_features, feature_names) # GA train subset
   X_test_ga = get_feature_subset(X_test_scaled, ga_selected_features, feature_names) # GA test subset

   print(f"{BackgroundColors.GREEN}Training set shape after GA feature selection: {BackgroundColors.CYAN}{X_train_ga.shape}{Style.RESET_ALL}") # Output shape
   print(f"{BackgroundColors.GREEN}Testing set shape after GA feature selection: {BackgroundColors.CYAN}{X_test_ga.shape}{Style.RESET_ALL}") # Output shape

   if X_train_ga.shape[1] == 0: # If GA selects no features
      print(f"{BackgroundColors.YELLOW}No features selected by GA for {csv_path}. Skipping file.{Style.RESET_ALL}")
      return # Exit early

   models_and_grids = get_models_and_param_grids() # Get model grids

   start_idx = len(dir_results_list) # Track result insertion index
   print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Starting hyperparameter optimization for {BackgroundColors.CYAN}{len(models_and_grids)}{BackgroundColors.GREEN} models on {BackgroundColors.CYAN}{os.path.basename(csv_path)}{BackgroundColors.GREEN}...{Style.RESET_ALL}\n") # Output header

   models = list(models_and_grids.items()) # Convert dict to list
   total_models = len(models) # Count models

   run_model_optimizations(models, csv_path, X_train_ga, y_train, dir_results_list) # Run optimizations

   added_slice = dir_results_list[start_idx:] # Extract slice
   print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Optimization Summary for {BackgroundColors.CYAN}{os.path.basename(csv_path)}{BackgroundColors.GREEN}:{Style.RESET_ALL}") # Summary header
   print(f"{BackgroundColors.GREEN}Total models optimized: {BackgroundColors.CYAN}{len(added_slice)}{Style.RESET_ALL}") # Output count

   if added_slice: # If results exist
      best_model = max(added_slice, key=lambda x: x["best_cv_f1_score"]) # Best model
      print(f"{BackgroundColors.GREEN}Best model: {BackgroundColors.CYAN}{best_model['model']}{Style.RESET_ALL}") # Output model
      print(f"{BackgroundColors.GREEN}Best CV F1 Score: {BackgroundColors.CYAN}{best_model['best_cv_f1_score']:.4f}{Style.RESET_ALL}") # Output score

def get_hardware_specifications():
   """
   Returns system specs: real CPU model (Windows/Linux/macOS), physical cores,
   RAM in GB, and OS name/version.
   
   :return: Dictionary with keys: cpu_model, cores, ram_gb, os
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Fetching system specifications...{Style.RESET_ALL}") # Output the verbose message
   
   system = platform.system() # Identify OS type

   try: # Try to fetch real CPU model using OS-specific methods
      if system == "Windows": # Windows: use WMIC
         out = subprocess.check_output("wmic cpu get Name", shell=True).decode(errors="ignore") # Run WMIC
         cpu_model = out.strip().split("\n")[1].strip() # Extract model line

      elif system == "Linux": # Linux: read from /proc/cpuinfo
         cpu_model = "Unknown" # Default
         with open("/proc/cpuinfo") as f: # Open cpuinfo
            for line in f: # Iterate lines
               if "model name" in line: # Model name entry
                  cpu_model = line.split(":",1)[1].strip() # Extract name
                  break # Stop after first match

      elif system == "Darwin": # macOS: use sysctl
         out = subprocess.check_output(["sysctl","-n","machdep.cpu.brand_string"]) # Run sysctl
         cpu_model = out.decode().strip() # Extract model string

      else: # Unsupported OS
         cpu_model = "Unknown" # Fallback

   except Exception: # If any method fails
      cpu_model = "Unknown" # Fallback on failure

   cores = psutil.cpu_count(logical=False) # Physical core count
   ram_gb = round(psutil.virtual_memory().total / (1024**3), 1) # Total RAM in GB
   os_name = f"{platform.system()} {platform.release()}" # OS name + version

   return { # Build final dictionary
      "cpu_model": cpu_model, # CPU model string
      "cores": cores, # Physical cores
      "ram_gb": ram_gb, # RAM in gigabytes
      "os": os_name # Operating system
   }

def save_optimization_results(csv_path, results_list):
   """
   Saves hyperparameter optimization results to a CSV file.

   :param csv_path: Path to the original dataset CSV file.
   :param results_list: List of dictionaries containing optimization results.
   :return: None
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Saving optimization results...{Style.RESET_ALL}") # Output the verbose message
   
   if not results_list: # If the results list is empty
      print(f"{BackgroundColors.YELLOW}No results to save.{Style.RESET_ALL}")
      return # Exit the function
   
   output_dir = f"{os.path.dirname(csv_path)}/Classifiers_Hyperparameters/" # Directory to save outputs
   os.makedirs(output_dir, exist_ok=True) # Create the output directory if it doesn't exist
   dataset_name = os.path.splitext(os.path.basename(csv_path))[0] # Get the base name of the dataset
   output_path = os.path.join(output_dir, f"{dataset_name}_{RESULTS_FILENAME}") # Full path to save the results CSV
   
   try: # Try to save the results
      df_results = pd.DataFrame(results_list) # Convert results list to DataFrame
      hardware_specs = get_hardware_specifications() # Get system specs
      df_results["Hardware"] = hardware_specs["cpu_model"] + " | Cores: " + str(hardware_specs["cores"]) + " | RAM: " + str(hardware_specs["ram_gb"]) + " GB | OS: " + hardware_specs["os"] # Add hardware specs column
      df_results.to_csv(output_path, index=False, encoding="utf-8") # Save to CSV
      print(f"{BackgroundColors.GREEN}Results saved to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}")
   except Exception as e: # Catch any errors during saving
      print(f"{BackgroundColors.RED}Error saving results: {e}{Style.RESET_ALL}")

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

   print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Classifiers Hyperparameters Optimization{BackgroundColors.GREEN} program!{Style.RESET_ALL}", end="\n\n") # Output the welcome message
   start_time = datetime.datetime.now() # Get the start time of the program

   for dataset_name, dirpath in iterate_dataset_directories(): # Iterate valid dataset directories
      csv_files = get_files_to_process(dirpath, file_extension=".csv") # Discover CSV files in this directory (non-recursive)
      if not csv_files: # If no CSV files were discovered in this dirpath
         verbose_output(f"{BackgroundColors.YELLOW}No CSV files found in: {BackgroundColors.CYAN}{dirpath}{Style.RESET_ALL}") # Verbose notice
         continue # Move to the next dirpath

      dir_results_list = [] # Aggregate results for all CSVs in this dirpath

      for csv_path in csv_files: # Process each CSV file found in the current dirpath
         try: # Process the current csv_path inside a try/except to continue on errors
            process_single_csv_file(csv_path, dir_results_list) # Process CSV end-to-end
         except Exception as e: # Catch any unhandled exceptions during CSV processing
            print(f"{BackgroundColors.RED}Unhandled error processing {csv_path}: {e}{Style.RESET_ALL}") # Print the exception and continue
            continue # Continue to the next CSV file

      if dir_results_list: # If there are results
         rep_csv_path = os.path.join(dirpath, os.path.basename(os.path.normpath(dirpath))) # Representative CSV base path
         save_optimization_results(rep_csv_path, dir_results_list) # Save results

   finish_time = datetime.datetime.now() # Get the finish time of the program
   print(f"\n{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}") # Output the start and finish times
   print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}") # Output the end of the program message

   atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None # Register the play_sound function to be called when the program finishes

if __name__ == "__main__":
   """
   This is the standard boilerplate that calls the main() function.

   :return: None
   """

   main() # Call the main function
