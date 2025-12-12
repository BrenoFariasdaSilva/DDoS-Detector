#!/usr/bin/env python3
"""
================================================================================
Dataset Descriptor and Report Generator
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-10-07
Description :
   This script automates the process of scanning datasets within a directory,
   extracting relevant statistical and structural information, and compiling
   the results into a comprehensive CSV report. It supports recursive
   directory traversal and detailed feature analysis for educational or
   research purposes in data-centric projects.

   Key features include:
      - Recursive discovery of dataset files by extension (default: .csv)
      - Detection of label columns based on naming conventions
      - Summary of sample and feature counts, feature types, and missing values
      - Class distribution analysis for classification datasets
      - Structured report generation in CSV format
      - Optional sound notification upon completion (cross-platform support)

Usage:
   Modify the variable `input_path` in the `main()` function to point to
   the dataset directory or single CSV file you wish to analyze.
   Then, simply run:
      $ make dataset_descriptor

TODOs:
   - Add CLI argument parsing for input paths, file extensions, and options.
   - Implement progress bars for large dataset scans.
   - Extend format support to ARFF, Parquet, and JSON.
   - Add summary statistics (mean, std, min, max) for numeric features.
   - Include timestamped report filenames for multiple runs.
   - Improve verbosity control and structured logging.

Dependencies:
   - Python >= 3.9
   - pandas, colorama

Output:
   - A CSV file named `_dataset_descriptor.csv` saved in the dataset directory,
   containing metadata and summary statistics for each discovered dataset.
"""

import atexit # For playing a sound when the program finishes
import datetime # For timestamping
import gc # For explicit garbage collection
import matplotlib.pyplot as plt # For plotting t-SNE results
import numpy as np # For numerical operations
import os # For running a command in the terminal
import pandas as pd # For data manipulation
import platform # For getting the operating system name
import sys # For system-specific parameters and functions
import warnings # For suppressing pandas warnings when requested
from colorama import Style # For coloring the terminal
from Logger import Logger # For logging output to both terminal and file
from pathlib import Path # For handling file paths
from sklearn.manifold import TSNE # For t-SNE dimensionality reduction
from sklearn.preprocessing import StandardScaler # For feature scaling
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

DATASETS = { # Dictionary containing dataset paths and feature files
	"CICDDoS2019-Dataset": [ # List of paths to the CICDDoS2019 dataset
		"./Datasets/CICDDoS2019/01-12/",
		"./Datasets/CICDDoS2019/03-11/",
   ],
   "CICDDoS2017-Dataset": [ # List of paths to the CICDDoS2017 dataset
      "./Datasets/CIC-IDS2017/Converted/",
   ],
}

CROSS_DATASET_VALIDATE = True # Set to True to perform cross-dataset validation between the datasets defined in DATASETS

RESULTS_DIR = "./Dataset_Description/" # Directory to save the results
RESULTS_FILENAME = "Dataset_Descriptor.csv" # Filename for the results CSV

IGNORE_FILES = [RESULTS_FILENAME] # List of filenames to ignore when searching for datasets
IGNORE_DIRS = ["Cache", "Data_Separability", "Dataset_Description", "Feature_Analysis"] # List of directory names to ignore when searching for datasets

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

def collect_matching_files(input_dir, file_format=".csv", ignore_files=IGNORE_FILES, ignore_dirs=IGNORE_DIRS,):
   """
   Recursively collects all files in the specified directory and subdirectories
   that match the given file format and are not in the ignore list.

   :param input_dir: Directory to search
   :param file_format: File format to include (default: .csv)
   :param ignore_files: List of filenames to ignore
   :param ignore_dirs: List of directory names to ignore
   :return: Sorted list of matching file paths
   """

   verbose_output(f"{BackgroundColors.GREEN}Collecting all files with format {BackgroundColors.CYAN}{file_format}{BackgroundColors.GREEN} in directory: {BackgroundColors.CYAN}{input_dir}{Style.RESET_ALL}") # Output the verbose message

   ignore_files = set(os.path.normcase(f) for f in (ignore_files or [])) # Normalize ignore files for case-insensitive comparison
   ignore_dirs = set(os.path.normcase(d) for d in (ignore_dirs or [])) # Normalize ignore directories for case-insensitive comparison

   matching_files = [] # List to store matching file paths

   for root, dirs, files in os.walk(input_dir): # Walk through the directory
      try: # Try to filter out ignored directories
         dirs[:] = [d for d in dirs if os.path.normcase(d) not in ignore_dirs] # Modify dirs in-place to skip ignored directories
      except Exception: # If an error occurs while filtering directories
         pass # Ignore the error and continue

      for file in files: # For each file
         if not file.endswith(file_format): # Skip files that do not match the specified format
            continue # Continue to the next file

         basename_norm = os.path.normcase(file) # Normalize the basename for case-insensitive comparison
         fullpath = os.path.join(root, file) # Get the full file path
         fullpath_norm = os.path.normcase(fullpath) # Normalize the full file path for case-insensitive comparison

         if basename_norm in ignore_files or fullpath_norm in ignore_files: # If the file is in the ignore list
            verbose_output(f"Skipping ignored file: {fullpath}") # Output verbose message for ignored file
            continue # Continue to the next file

         matching_files.append(fullpath) # Add the full file path to the list

   sorted_matching_files = sorted(set(matching_files)) # Remove duplicates and sort the list
   
   return sorted_matching_files # Return the sorted list of matching files

def build_headers_map(filepaths, low_memory=True):
   """
   Build a mapping of file path -> list of header columns.

   Attempts a lightweight header-only read (`pd.read_csv(..., nrows=0)`)
   and falls back to `load_dataset` if that fails.

   :param filepaths: Iterable of file paths
   :param low_memory: Passed to `load_dataset` when falling back
   :return: dict mapping filepath -> list of column names
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Building headers map for provided file paths.{Style.RESET_ALL}") # Output the verbose message

   headers = {} # Dictionary that will map each filepath to its list of columns
   for fp in filepaths: # Iterate over all given file paths
      try: # Try header-only read
         cols = pd.read_csv(fp, nrows=0).columns.tolist() # Extract columns without loading file content
      except Exception: # If header-only read fails
         df_tmp = load_dataset(fp, low_memory=low_memory) # Load full dataset (slow fallback)
         cols = df_tmp.columns.tolist() if df_tmp is not None else [] # Extract columns if dataset loaded
      headers[fp] = cols # Store the resolved column list for this file

   return headers # Return filepath->headers mapping

def compute_common_features(headers_map):
   """
   Compute the intersection of headers across all files and determine
   whether all headers match exactly (ignoring case and surrounding whitespace).

   :param headers_map: dict mapping filepath -> list of column names
   :return: tuple (common_features_set, headers_match_all_bool)
   """

   try: # Normalize all headers to lowercase and strip whitespace
      normalized_sets = [set(col.strip().lower() for col in v) for v in headers_map.values() if v] # List of normalized header sets
      if normalized_sets: # If there are valid header sets
         common = set.intersection(*normalized_sets) # Compute intersection across all files
      else: # If no headers available
         common = set() # Empty intersection
   except Exception: # Catch unexpected failures
      common = set() # Fallback to empty set

   unique_normalized_sets = {frozenset(col.strip().lower() for col in v) for v in headers_map.values() if v} # Unique normalized header sets
   match_all = len(unique_normalized_sets) <= 1 # True if all header sets are identical

   return common, match_all # Return shared features and match flag

def load_dataset(filepath, low_memory=True):
   """
   Loads a dataset from a CSV file.
   
   :param filepath: Path to the CSV file
   :param low_memory: Whether to use low memory mode (default: True)
   :return: Pandas DataFrame
   """

   try: # Try to load the dataset
      with warnings.catch_warnings(): # Suppress DtypeWarning warnings
         warnings.simplefilter("ignore", pd.errors.DtypeWarning) # Ignore DtypeWarning warnings
         df = pd.read_csv(filepath, low_memory=low_memory) # Load the dataset

      return df # Return the DataFrame
   except Exception as e: # If an error occurs
      print(f"{BackgroundColors.RED}Error loading {BackgroundColors.GREEN}{filepath}: {e}{Style.RESET_ALL}")
      return None # Return None if an error occurs

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

def detect_label_column(columns):
   """
   Try to guess the label column based on common naming conventions.
   
   :param columns: List of column names
   :return: The name of the label column if found, else None
   """

   candidates = ["label", "class", "target"] # Common label column names

   for col in columns: # First search for exact matches
      if col.lower() in candidates: # Verify if the column name matches any candidate exactly
         return col # Return the column name if found

   for col in columns: # Second search for partial matches
      if "target" in col.lower() or "label" in col.lower(): # Verify if the column name contains any candidate
         return col # Return the column name if found

   return None # Return None if no label column is found

def summarize_features(df):
   """
   Summarizes number of samples, features, and feature types.
   Ensures the sum of feature types matches the number of columns.

   :param df: pandas DataFrame
   :return: Tuple containing:
            n_samples, n_features, n_numeric, n_int, n_categorical, n_other, categorical columns string
   """

   n_samples, n_features = df.shape # Get number of samples and features
   dtypes = df.dtypes # Get data types of each column

   n_numeric = dtypes[dtypes == "float64"].count() # Count float64 types
   n_int = dtypes[dtypes == "int64"].count() + dtypes[dtypes == "Int64"].count() # Count int64 and Int64 types
   n_categorical = dtypes[dtypes.isin(["object", "category", "bool", "string"])].count() # Count categorical types

   n_other = n_features - (n_numeric + n_int + n_categorical) # Anything else goes to "other"

   categorical_cols = df.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist() # List of categorical columns
   categorical_cols_str = ", ".join(categorical_cols) if categorical_cols else "None" # Create string of categorical columns or "None"

   return n_samples, n_features, n_numeric, n_int, n_categorical, n_other, categorical_cols_str # Return the summary values

def summarize_missing_values(df):
   """
   Summarizes missing values for the dataset.

   :param df: The pandas DataFrame
   :return: Summary string of missing values
   """

   missing_vals = df.isnull().sum() # Get count of missing values per column
   missing_summary = ", ".join([f"{col} ({cnt})" for col, cnt in missing_vals.items() if cnt > 0]) if missing_vals.sum() > 0 else "None" # Create summary string or "None"

   return missing_summary # Return the missing values summary

def summarize_classes(df, label_col):
   """
   Summarizes classes and class distributions if a label column exists.

   :param df: The pandas DataFrame
   :param label_col: The name of the label column
   :return: Tuple containing string of classes and class distribution summary
   """

   if label_col and label_col in df.columns: # If a label column exists
      classes = df[label_col].unique() # Get unique classes
      classes_str = ", ".join(map(str, classes)) # Create string of classes
      class_counts = df[label_col].value_counts() # Get counts of each class
      total = class_counts.sum() # Total number of samples
      class_dist_list = [f"{cls}: {cnt} ({cnt/total*100:.2f}%)" for cls, cnt in class_counts.items()] # Create class distribution list
      class_dist_str = ", ".join(class_dist_list) # Create class distribution string
      return classes_str, class_dist_str # Return the classes and class distribution
   
   return "None", "None" # Return "None" if no label column

def coerce_numeric_columns(df):
   """
   Try to extract numeric columns from `df`. If no numeric columns are
   present, attempt to coerce object/string columns to numeric values.

   :param df: pandas DataFrame
   :return: DataFrame with numeric columns (may be empty)
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Extracting or coercing numeric columns from the DataFrame.{Style.RESET_ALL}") # Output the verbose message

   numeric_df = df.select_dtypes(include=["number"]).copy() # Select numeric columns from the DataFrame
   if numeric_df.empty: # If there are no numeric columns found
      obj_cols = df.select_dtypes(include=["object", "string"]).columns.tolist() # List object/string columns as candidates
      for c in obj_cols: # Iterate over candidate object/string columns
         coerced = pd.to_numeric(df[c], errors="coerce") # Attempt to coerce the column to numeric, invalid -> NaN
         if coerced.notna().sum() > 0: # If coercion produced any non-NaN values
            numeric_df[c] = coerced # Add the coerced column to the numeric DataFrame

   return numeric_df # Return the numeric-only DataFrame (may be empty)

def fill_replace_and_drop(numeric_df):
   """
   Replace infinities, drop all-NaN columns, and fill remaining NaNs with
   the column median (or 0 when median is NaN).

   :param numeric_df: DataFrame with numeric columns
   :return: cleaned DataFrame (may be empty)
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Replacing infinities, dropping all-NaN columns, and filling NaNs with column medians.{Style.RESET_ALL}") # Output the verbose message

   numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan) # Replace +/-infinity with NaN
   numeric_df = numeric_df.loc[:, numeric_df.notna().any(axis=0)] # Drop columns that are entirely NaN
   if numeric_df.shape[1] == 0: # If no columns remain after dropping
      return numeric_df # Return the (empty) DataFrame

   for col in numeric_df.columns: # Iterate over numeric columns
      med = numeric_df[col].median() # Compute column median
      numeric_df[col] = numeric_df[col].fillna(0 if pd.isna(med) else med) # Fill NaNs with median or 0

   return numeric_df # Return cleaned numeric DataFrame

def compute_initial_alloc(counts, min_per_class):
   """
   Compute initial per-class allocations capped by `min_per_class`.

   This helper computes the initial allocation for each class as the
   minimum of the class count and the requested `min_per_class` value and
   returns the allocation mapping together with the sum of those values.

   :param counts: pandas Series with per-class counts
   :param min_per_class: preferred minimum samples per class
   :return: Tuple (initial_alloc dict, s_min int)
   """
   
   initial = {c: min(int(counts[c]), int(min_per_class)) for c in counts.index} # Compute min(count, min_per_class)
   s = sum(initial.values()) # Sum of initial allocations
   
   return initial, s # Return tuple (initial_alloc, s_min)

def allocate_with_min(initial_alloc, counts, max_samples):
   """
   Distribute remaining capacity after satisfying per-class minima.

   Starting from `initial_alloc` (which already enforces per-class minima),
   this helper distributes the remaining available capacity proportionally
   to classes that still have unused samples. It performs integer flooring
   and then distributes leftover units according to fractional remainders
   to produce a final integer allocation per class.

   :param initial_alloc: dict mapping class -> allocated minima
   :param counts: pandas Series with per-class counts
   :param max_samples: total maximum samples to allocate
   :return: dict mapping class -> final allocation
   """
   
   alloc = dict(initial_alloc) # Start with initial allocations
   remaining_local = max_samples - sum(initial_alloc.values()) # Remaining capacity after minima
   rem_avail_local = {c: max(0, int(counts[c]) - alloc[c]) for c in counts.index} # Remaining available per class
   total_rem_avail_local = sum(rem_avail_local.values()) # Total remaining available
   
   if total_rem_avail_local > 0 and remaining_local > 0: # Only proceed if there is capacity to distribute
      float_add_local = {c: (remaining_local * rem_avail_local[c] / total_rem_avail_local) for c in counts.index} # Proportional fractional add
      add_alloc_local = {c: int(float_add_local[c]) for c in counts.index} # Base integer additional allocation
      assigned_local = sum(add_alloc_local.values()) # Sum of base additional allocations
      leftover_local = remaining_local - assigned_local # Leftover after flooring
      remainders_local = sorted(counts.index, key=lambda c: (float_add_local[c] - add_alloc_local[c]), reverse=True) # Order by fractional remainder
      
      for c in remainders_local: # Distribute leftover one-by-one
         if leftover_local <= 0: # Stop when no leftover remains
            break # Exit distribution
         if add_alloc_local[c] < rem_avail_local[c]: # Only add if class can accept more
            add_alloc_local[c] += 1 # Increment allocation for this class
            leftover_local -= 1 # Decrease leftover count
      for c in counts.index: # Finalize allocations applying available caps
         alloc[c] += min(add_alloc_local.get(c, 0), rem_avail_local[c]) # Cap addition by remaining available
         
   return alloc # Return finalized allocations

def proportional_alloc(counts, max_samples):
   """
   Compute a proportional allocation across classes when minima cannot be met.

   This helper computes a proportional distribution of `max_samples` across
   classes according to their relative counts. It floors fractional values
   to integers and then distributes leftover units by descending fractional
   remainder to ensure the total sums to `max_samples` (subject to class
   availability caps).

   :param counts: pandas Series with per-class counts
   :param max_samples: total maximum samples to allocate
   :return: dict mapping class -> final allocation
   """
   
   total_local = int(counts.sum()) # Total samples available across classes
   float_alloc_local = {c: (max_samples * int(counts[c]) / total_local) for c in counts.index} # Fractional proportional allocation
   base_alloc_local = {c: int(float_alloc_local[c]) for c in counts.index} # Base integer allocation
   assigned_local = sum(base_alloc_local.values()) # Sum of base allocations
   leftover_local = max_samples - assigned_local # Leftover to distribute due to flooring
   remainders_local = sorted(counts.index, key=lambda c: (float_alloc_local[c] - base_alloc_local[c]), reverse=True) # Order by fractional remainder
   
   for c in remainders_local: # Distribute leftover one-by-one
      if leftover_local <= 0: # Stop when leftover exhausted
         break # Exit loop
      if base_alloc_local[c] < int(counts[c]): # Only increase if class has remaining samples
         base_alloc_local[c] += 1 # Increment base allocation
         leftover_local -= 1 # Decrement leftover
   
   final_alloc_local = {c: min(int(counts[c]), base_alloc_local[c]) for c in counts.index} # Cap by class availability
   
   return final_alloc_local # Return proportional allocations

def sample_indices_from_alloc(labels, allocations, random_state):
   """
   Draw indices from `labels` according to `allocations` using `random_state`.

   For each class in `allocations`, this helper selects the requested number
   of indices without replacement (or all available indices if the
   allocation exceeds availability). The selection is reproducible via the
   provided `random_state`.

   :param labels: pandas Series with class labels
   :param allocations: dict mapping class -> number of samples to draw
   :param random_state: integer seed for RNG reproducibility
   :return: list of sampled row indices
   """
   
   rng_local = np.random.RandomState(random_state) # RNG for reproducibility
   sampled_indices_local = [] # Container for sampled indices
   
   for cls in allocations: # Iterate classes in allocation order
      cls_idx_local = labels[labels == cls].index.to_list() # Indices belonging to the class
      k_local = allocations.get(cls, 0) # Number to sample for this class
      
      if k_local <= 0: # Skip when zero allocation
         continue # Continue to next class
      
      if k_local >= len(cls_idx_local): # If allocation exceeds availability
         sampled_local = cls_idx_local # Take all available indices
      else: # Otherwise sample without replacement
         sampled_local = list(rng_local.choice(cls_idx_local, size=k_local, replace=False)) # Draw random sample
         
      sampled_indices_local.extend(sampled_local) # Append sampled indices
   
   return sampled_indices_local # Return list of sampled indices

def stratified_sample(numeric_df, labels, max_samples, random_state=42, min_per_class=50):
   """
   Downsample numeric features and labels to at most `max_samples` rows while
   attempting to ensure a minimum number of samples per class.

   The function will try to allocate up to `min_per_class` samples to each
   class, then distribute remaining capacity proportionally. If the minimum
   cannot be satisfied for every class the allocation falls back to a
   proportional distribution.

   :param numeric_df: DataFrame with numeric features
   :param labels: pandas Series aligned with `numeric_df`
   :param max_samples: Maximum total samples to return
   :param random_state: Seed for reproducible sampling (default: 42)
   :param min_per_class: Preferred minimum samples per class (default: 50)
   :return: Tuple (sampled_numeric_df, sampled_labels)
   """

   verbose_output(f"{BackgroundColors.GREEN}Stratified sampling to a maximum of {max_samples} samples while preserving class proportions and ensuring min {min_per_class} per class when possible.{Style.RESET_ALL}") # Verbose message

   n_rows = len(numeric_df) # total rows available
   if n_rows <= max_samples: # nothing to do
      return numeric_df.reset_index(drop=True), labels.reset_index(drop=True) # Return original DataFrame and labels

   counts = labels.value_counts() # per-class counts
   classes = list(counts.index) # list of class labels
   total = int(counts.sum()) # total available samples

   initial_alloc, s_min = compute_initial_alloc(counts, min_per_class) # compute initial allocations and their sum

   allocations = {c: 0 for c in classes} # Final allocations per class

   if s_min <= max_samples: # Can satisfy minimum for all classes
      allocations.update(allocate_with_min(initial_alloc, counts, max_samples)) # Apply min-aware allocation
   else: # Cannot satisfy minimum for all classes; allocate proportionally
      allocations = proportional_alloc(counts, max_samples) # Apply proportional allocation

   sampled_idx = sample_indices_from_alloc(labels, allocations, random_state) # Sample indices according to allocations

   if len(sampled_idx) > max_samples: # Safety check to guard against slight over-allocation
      sampled_idx = sampled_idx[:max_samples] # Trim to max_samples if exceeded

   return numeric_df.loc[sampled_idx].reset_index(drop=True), labels.loc[sampled_idx].reset_index(drop=True) # Return sampled DataFrame and labels

def scale_features(numeric_df):
   """
   Standardize numeric features to zero mean and unit variance. Fall back to
   converting to float64 array if scaling fails.
   
   :param numeric_df: DataFrame with numeric features
   :return: Numpy array with scaled features
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Scaling numeric features to zero mean and unit variance.{Style.RESET_ALL}") # Output the verbose message

   try: # Try scaling with sklearn StandardScaler
      scaler = StandardScaler() # Create scaler instance
      X_scaled = scaler.fit_transform(numeric_df.values) # Fit and transform numeric values
   except Exception: # Fallback if scaling fails
      X_scaled = np.asarray(numeric_df.values, dtype=np.float64) # Convert to a float64 numpy array
   
   return X_scaled # Return the scaled array

def allocate_remaining_budget(counts, allocations, remaining_budget):
   """
   Helper function to distribute remaining budget among classes proportionally,
   using the fractional remainder method.

   :param counts: pandas Series of class counts
   :param allocations: dict of current allocations
   :param remaining_budget: number of samples left to allocate
   :return: dict of updated allocations including remaining budget
   """
   extras = {cls: max(0, int(counts[cls]) - allocations.get(cls, 0)) for cls in counts.index} # Extra samples available per class
   total_extras = sum(extras.values()) # Total extra samples available

   if total_extras > 0: # Only proceed if there are extras to allocate
      float_alloc = {cls: remaining_budget * extras[cls] / total_extras for cls in extras} # Fractional allocation
      base_alloc = {cls: int(float_alloc[cls]) for cls in float_alloc} # Base integer allocation

      assigned = sum(base_alloc.values()) # Sum of base allocations
      leftover = remaining_budget - assigned # Leftover samples to distribute

      remainders = sorted(extras.keys(), key=lambda c: (float_alloc[c] - base_alloc[c]), reverse=True) # Order by fractional remainder
      for cls in remainders: # Distribute leftover samples
         if leftover <= 0: # Stop when no leftover remains
            break # Exit loop
         if base_alloc.get(cls, 0) < extras.get(cls, 0): # Only allocate if class can accept more
            base_alloc[cls] = base_alloc.get(cls, 0) + 1 # Increment allocation for this class
            leftover -= 1 # Decrease leftover count

      for cls in counts.index: # Finalize allocations applying available caps
         allocations[cls] = min(int(counts[cls]), allocations.get(cls, 0) + base_alloc.get(cls, 0)) # Cap addition by remaining available

   return allocations # Return updated allocations

def compute_class_aware_allocations(labels, sample_size, min_class_size=50):
   """
   Compute per-class sample allocations that preserve class distribution while
   ensuring small classes (< min_class_size) are fully included.

   Strategy:
   - Classes with fewer than min_class_size samples: include all samples
   - Remaining budget: distribute proportionally among larger classes
   - Use fractional remainder method to allocate leftover samples fairly

   :param labels: pandas Series containing class labels
   :param sample_size: maximum total samples to allocate
   :param min_class_size: threshold below which all class samples are included
   :return: dict mapping class -> number of samples to select
   """

   counts = labels.value_counts() # Get per-class counts

   allocations = {cls: min(int(cnt), int(min_class_size)) for cls, cnt in counts.items()} # Initial allocations for small classes

   remaining_budget = int(sample_size - sum(allocations.values())) # Remaining samples to allocate

   if remaining_budget > 0: # If there is remaining budget
      allocations = allocate_remaining_budget(counts, allocations, remaining_budget) # Distribute remaining budget

   total_alloc = sum(allocations.values()) # Total allocated samples
   if total_alloc > sample_size: # Safety check to reduce overallocation
      sorted_by_alloc = sorted(allocations.items(), key=lambda x: x[1], reverse=True) # Sort classes by allocation descending
      i = 0 # Index for iteration
      while total_alloc > sample_size and i < len(sorted_by_alloc): # While overallocation exists
         cls, cur = sorted_by_alloc[i] # Current class and its allocation
         reducible = cur - min(int(counts[cls]), int(min_class_size)) # Amount that can be reduced without violating minima
         if reducible > 0: # If there is reducible allocation
            remove = min(reducible, total_alloc - sample_size) # Amount to remove
            allocations[cls] -= remove # Reduce allocation
            total_alloc -= remove # Update total allocation
         i += 1 # Move to next class

   return allocations # Return finalized allocations

def sample_by_class_allocation(labels, allocations, random_state):
   """
   Sample row indices according to per-class allocations.

   For each class, randomly selects the allocated number of samples without
   replacement. If allocation exceeds available samples for a class, all
   samples from that class are included.

   :param labels: pandas Series containing class labels
   :param allocations: dict mapping class -> number of samples to select
   :param random_state: seed for reproducible random sampling
   :return: list of selected row indices
   """

   selected_idx = [] # Container for selected indices
   rng = np.random.RandomState(random_state) # RNG for reproducibility
   
   for cls, k in allocations.items(): # Iterate allocations per class
      idxs = labels[labels == cls].index.to_list() # All indices for this class
      if k >= len(idxs): # If allocation >= available samples
         sampled_local = idxs # Take all available indices
      else: # Otherwise
         sampled_local = list(rng.choice(idxs, size=k, replace=False)) # Randomly choose k indices without replacement
      selected_idx.extend(sampled_local) # Append sampled indices to selection
   
   return selected_idx # Return the final list of indices

def downsample_with_class_awareness(numeric_df, labels, sample_size, random_state):
   """
   Downsample dataset while preserving class distribution and ensuring
   small classes (< 50 samples) are fully represented.

   Falls back to random sampling if class-aware sampling fails or labels
   are unavailable.

   :param numeric_df: DataFrame containing numeric features
   :param labels: pandas Series with class labels (or None)
   :param sample_size: target number of samples after downsampling
   :param random_state: seed for reproducible sampling
   :return: tuple (downsampled numeric_df, downsampled labels or None)
   """

   if labels is None: # No labels available -> random sampling
      return numeric_df.sample(n=sample_size, random_state=random_state), None # Return random sample and no labels

   try: # Try class-aware downsampling
      allocations = compute_class_aware_allocations(labels, sample_size, min_class_size=50) # Compute per-class allocations
      selected_idx = sample_by_class_allocation(labels, allocations, random_state) # Sample indices per allocations
      
      if not selected_idx: # If selection failed or empty
         return numeric_df.sample(n=sample_size, random_state=random_state), None # Fallback to random sampling
      
      return (numeric_df.loc[selected_idx].reset_index(drop=True), labels.loc[selected_idx].reset_index(drop=True)) # Return downsampled DataFrame and labels
   except Exception: # On any error, fallback to random sampling
      return numeric_df.sample(n=sample_size, random_state=random_state), None # Return random sample and no labels

def initialize_and_fit_tsne(X, perplexity=30, n_iter=1000, random_state=42):
   """
   Initialize t-SNE with proper parameters and compute 2D embedding.

   Handles compatibility with different TSNE versions by inspecting the constructor
   signature and setting 'n_iter' or 'max_iter' accordingly.

   :param X: numpy array of scaled numeric features
   :param perplexity: t-SNE perplexity parameter
   :param n_iter: number of t-SNE optimization iterations
   :param random_state: random seed for reproducibility
   :return: 2D numpy array of t-SNE embeddings
   """
   
   try: # Inspect TSNE init signature for compatibility
      from inspect import signature # Import signature function
      sig = signature(TSNE.__init__).parameters # Get TSNE init signature
   except Exception: # If inspection fails
      sig = {} # Fallback to empty signature

   tsne_kwargs = {"n_components": 2, "perplexity": perplexity, "random_state": random_state, "init": "pca"} # Base t-SNE args
   if "n_iter" in sig: # Check for n_iter parameter
      tsne_kwargs["n_iter"] = n_iter # Set n_iter if supported
   elif "max_iter" in sig: # Check for max_iter parameter
      tsne_kwargs["max_iter"] = n_iter # Set max_iter if supported
   else: # Neither parameter supported
      tsne_kwargs["max_iter"] = n_iter # Default to max_iter

   tsne = TSNE(**tsne_kwargs) # Initialize t-SNE with compatible args
   X_emb = tsne.fit_transform(X) # Compute embedding
   return X_emb # Return the 2D embedding

def save_tsne_plot(X_emb, labels, output_path, title):
   """
   Create and save a 2D t-SNE scatter plot.

   If labels are provided, points are colored by class with a legend.
   Otherwise, all points are plotted uniformly.

   :param X_emb: 2D numpy array of t-SNE embeddings (shape: [n_samples, 2])
   :param labels: pandas Series of class labels or None
   :param output_path: absolute path where PNG will be saved
   :param title: plot title string
   :return: None
   """

   plt.figure(figsize=(8, 6)) # Create matplotlib figure
   
   if labels is not None: # Plot colored by class
      labels_ser = pd.Series(labels) # Ensure labels are a pandas Series
      counts = labels_ser.value_counts() # Count samples per class
      unique = list(labels_ser.unique()) # Unique class labels (preserve order)
      for cls in unique: # Plot each class separately
         mask = (labels_ser == cls) # Boolean mask for class
         plt.scatter(X_emb[mask, 0], X_emb[mask, 1], label=f"{cls} ({int(counts.get(cls, 0))})", s=8) # Scatter plot for class with count in label
      plt.legend(markerscale=2, fontsize="small") # Add legend for classes
      try: # Try to add counts text box
         counts_text = "\n".join([f"{str(c)}: {int(counts[c])}" for c in counts.index]) # Prepare counts text
         plt.gcf().text(0.99, 0.01, counts_text, ha="right", va="bottom", fontsize=8, bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")) # Add text box with counts
      except Exception: # Ignore any errors in adding counts text
         pass # Do nothing
   else: # No labels provided
      plt.scatter(X_emb[:, 0], X_emb[:, 1], s=8) # Plot all points uniformly
   
   plt.title(title) # Set plot title
   plt.xlabel("t-SNE 1") # X-axis label
   plt.ylabel("t-SNE 2") # Y-axis label
   plt.tight_layout() # Adjust layout
   plt.savefig(output_path, dpi=150) # Save figure to disk
   plt.close() # Close figure to free memory

def generate_tsne_plot(filepath, low_memory=True, sample_size=5000, perplexity=30, n_iter=1000, random_state=42, output_dir=None):
   """
   Generate and save a 2D t-SNE visualization of a CSV dataset.

   This function loads a dataset, extracts numeric features, performs class-aware
   downsampling (if needed), computes a 2D t-SNE embedding, and saves the result
   as a PNG scatter plot with per-class coloring (if labels are detected).

   Downsampling strategy:
   - Classes with < 50 samples: all samples included
   - Larger classes: sampled proportionally to preserve distribution
   - Falls back to random sampling if class detection fails

   :param filepath: path to CSV file
   :param low_memory: whether to use low-memory mode when loading CSV
   :param sample_size: maximum number of rows to embed (reduces computation time)
   :param perplexity: t-SNE perplexity parameter (typically 5-50)
   :param n_iter: number of t-SNE optimization iterations
   :param random_state: random seed for reproducible results
   :param output_dir: directory for saving PNG (defaults to dataset's RESULTS_DIR)
   :return: basename of saved PNG file, or None on failure
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Generating t-SNE plot for: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}") # Output start message for t-SNE generation

   try: # Main try-catch block for overall failure handling
      df = load_dataset(filepath, low_memory=low_memory) # Load CSV into DataFrame
      if df is None: # If loading failed
         return None # Abort

      cleaned = preprocess_dataframe(df, remove_zero_variance=False) # Basic cleaning
      if cleaned is None: # If cleaning failed
         return None # Abort

      numeric_df = coerce_numeric_columns(cleaned) # Extract numeric features
      if numeric_df is None: # If extraction failed
         return None # Abort
      
      numeric_df = fill_replace_and_drop(numeric_df) # Clean numeric frame
      if numeric_df is None: # If cleaning failed
         return None # Abort

      if numeric_df.shape[0] == 0 or numeric_df.shape[1] == 0: # No numeric data
         return None # Abort

      label_col = detect_label_column(cleaned.columns) # Detect label column
      labels = cleaned[label_col] if label_col in cleaned.columns else None # Extract labels if present

      if numeric_df.shape[0] > sample_size: # Downsample if too many rows
         numeric_df, labels = downsample_with_class_awareness(numeric_df, labels, sample_size, random_state) # Class-aware downsampling

      X = scale_features(numeric_df) # Scale features for t-SNE

      n_rows = X.shape[0] # Number of rows after downsampling
      if n_rows <= max(3, int(perplexity) + 1): # Check t-SNE feasibility
         return None # Abort if too few samples for t-SNE

      X_emb = initialize_and_fit_tsne(X, perplexity, n_iter, random_state) # Compute t-SNE embedding

      if output_dir is None: # Determine output directory
         output_dir = os.path.join(os.path.dirname(os.path.abspath(filepath)), "Data_Separability") # Default RESULTS_DIR under file folder
      os.makedirs(output_dir, exist_ok=True) # Ensure directory exists

      base = os.path.splitext(os.path.basename(filepath))[0] # Base filename
      out_name = f"TSNE_3D_{base}.png" # Output PNG name
      out_path = os.path.join(output_dir, out_name) # Absolute path

      save_tsne_plot(X_emb, labels, out_path, f"t-SNE: {base}") # Create and save plot

      try: # Try to delete DataFrame to free memory
         del df # Free raw DataFrame
      except Exception: # Ignore any exceptions during deletion
         pass # Do nothing
      gc.collect() # Force garbage collection

      return out_name # Return saved filename
   except Exception as e: # Catch-all failure
      print(f"{BackgroundColors.RED}t-SNE generation failed for {filepath}: {e}{Style.RESET_ALL}") # Verbose error
      return None # Indicate failure

def get_dataset_file_info(filepath, low_memory=True):
   """
   Extracts dataset information from a CSV file and returns it as a dictionary.

   :param filepath: Path to the CSV file
   :param low_memory: Whether to use low memory mode when loading the CSV (default: True)
   :return: Dictionary containing dataset information
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Extracting dataset information from: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}") # Output start message for dataset info extraction

   df = load_dataset(filepath, low_memory) # Load the dataset
   
   if df is None: # If the dataset could not be loaded
      return None # Return None
   
   cleaned_df = preprocess_dataframe(df) # Preprocess the DataFrame

   label_col = detect_label_column(cleaned_df.columns) # Try to detect the label column
   n_samples, n_features, n_numeric, n_int, n_categorical, n_other, categorical_cols_str = summarize_features(cleaned_df) # Summarize features
   missing_summary = summarize_missing_values(cleaned_df) # Summarize missing values
   classes_str, class_dist_str = summarize_classes(cleaned_df, label_col) # Summarize classes and distributions
   
   try: # Try to get file size in GB
      size_bytes = os.path.getsize(filepath) # Get file size in bytes
      size_gb = size_bytes / (1024 ** 3) # Convert bytes to gigabytes
      size_gb_str = f"{size_gb:.3f}" # Format size to 3 decimal places
   except Exception: # If an error occurs
      size_gb_str = "N/A" # Set size to N/A if error occurs

   result = { # Return the dataset information as a dictionary
      "Dataset Name": os.path.basename(filepath),
      "Size (GB)": size_gb_str,
      "Number of Samples": f"{n_samples:,}", # Format with commas for readability
      "Number of Features": f"{n_features:,}", # Format with commas for readability
      "Feature Types": f"{n_numeric} numeric (float64), {n_int} integer (int64), {n_categorical} categorical (object/category/bool/string), {n_other} other",
      "Categorical Features (object/string)": categorical_cols_str,
      "Missing Values": missing_summary,
      "Classes": classes_str,
      "Class Distribution": class_dist_str,
   }

   try: # Try to delete the DataFrame
      del df # Delete the DataFrame
   except Exception: # Ignore any exceptions during deletion
      pass # Do nothing
   gc.collect() # Force garbage collection

   return result # Return the dataset information

def get_file_common_and_extras(headers_map, filepath, common_features):
   """
   Return the sorted common features list and extra columns for a specific file, using normalized feature names (lowercase + strip).

   :param headers_map: dict mapping filepath -> list of column names
   :param filepath: path for which to compute extras
   :param common_features: set of features present in all files
   :return: tuple (common_list, extras_list)
   """

   file_cols = headers_map.get(filepath, []) # Get headers for this file

   if file_cols is not None: # Normalize file columns
      normalized_file_cols = set(col.strip().lower() for col in file_cols) # Normalize file columns
      normalized_common = set(col.strip().lower() for col in common_features) # Normalize common features
      extras = sorted(normalized_file_cols - normalized_common) # Compute non-common extras
   else: # If no columns found for this file
      extras = [] # No extras

   common_list = sorted(col.strip().lower() for col in common_features) if common_features else [] # Sorted normalized shared features

   return common_list, extras # Return common + extras lists

def write_report(report_rows, base_dir, output_filename):
   """
   Writes the report rows to a CSV file.

   :param report_rows: List of dictionaries containing report data
   :param base_dir: Base directory for saving the report
   :param output_filename: Name of the output CSV file
   :return: None
   """

   report_df = pd.DataFrame(report_rows) # Create a DataFrame from the report rows

   if "#" in report_df.columns: # If the "#"" column exists
      cols = ["#"] + [c for c in report_df.columns if c != "#"] # Move "#" to the front
      report_df = report_df[cols] # Reorder columns

   results_dir = os.path.join(base_dir, RESULTS_DIR) # Create results directory path
   os.makedirs(results_dir, exist_ok=True) # Create results directory if it doesn't exist
   report_csv_path = os.path.join(results_dir, output_filename) # Path to save the report CSV
   report_df.to_csv(report_csv_path, index=False) # Save the report to a CSV file

def generate_dataset_report(input_path, file_extension=".csv", low_memory=True, output_filename=RESULTS_FILENAME):
   """
   Generates a CSV report for the specified input path.
   The Dataset Name column will include subdirectories if present.

   :param input_path: Directory or file path containing the dataset
   :param file_extension: File extension to filter (default: .csv)
   :param low_memory: Whether to use low memory mode when loading CSVs (default: True)
   :param output_filename: Name of the CSV file to save the report
   :return: True if the report was generated successfully, False otherwise
   """

   report_rows = [] # List to store report rows
   sorted_matching_files = [] # List to store matching files

   if os.path.isdir(input_path): # If the input path is a directory
      print(f"{BackgroundColors.GREEN}Scanning directory {BackgroundColors.CYAN}{input_path}{BackgroundColors.GREEN} for {BackgroundColors.CYAN}{file_extension}{BackgroundColors.GREEN} files...{Style.RESET_ALL}") # Output scanning message
      sorted_matching_files = collect_matching_files(input_path, file_extension) # Collect matching files
      base_dir = os.path.abspath(input_path) # Get the absolute path of the base directory
   elif os.path.isfile(input_path) and input_path.endswith(file_extension): # If the input path is a file
      print(f"{BackgroundColors.GREEN}Processing single file...{Style.RESET_ALL}") # Output processing single file message
      sorted_matching_files = [input_path] # Only process this single file
      base_dir = os.path.dirname(os.path.abspath(input_path)) # Get the base directory of the file
   else: # If the input path is neither a directory nor a valid file
      print(f"{BackgroundColors.RED}Input path is neither a directory nor a valid {file_extension} file: {input_path}{Style.RESET_ALL}") # Output the error message
      sorted_matching_files = [] # No files to process
      base_dir = os.path.abspath(input_path) # Just use the input path as base_dir for error message

   if not sorted_matching_files: # If no matching files were found
      print(f"{BackgroundColors.RED}No matching {file_extension} files found in: {input_path}{Style.RESET_ALL}")
      return False # Exit the function

   headers_map = build_headers_map(sorted_matching_files, low_memory=low_memory) # Build headers map for all matching files
   common_features, headers_match_all = compute_common_features(headers_map) # Compute common features and header match status

   progress = tqdm(sorted_matching_files, desc=f"{BackgroundColors.GREEN}Processing files{Style.RESET_ALL}", unit="file") # Create a progress bar
   for idx, filepath in enumerate(progress, 1): # Process each matching file
      progress.set_description(f"{BackgroundColors.GREEN}Processing file {BackgroundColors.CYAN}{idx}/{len(sorted_matching_files)}{BackgroundColors.GREEN}: {BackgroundColors.CYAN}{os.path.basename(filepath)}{Style.RESET_ALL}") # Update progress bar description
      info = get_dataset_file_info(filepath, low_memory) # Get dataset info
      if info: # If info was successfully retrieved
         relative_path = os.path.relpath(filepath, base_dir) # Get path relative to base_dir
         info["Dataset Name"] = relative_path.replace("\\", "/") # Use relative path for Dataset Name and normalize slashes

         common_list, extras = get_file_common_and_extras(headers_map, filepath, common_features) # Get common and extra features for this file

         info["Headers Match All Files"] = "Yes" if headers_match_all else "No" # Indicate if headers match all files
         info["Common Features (in all files)"] = ", ".join(common_list) if common_list else "None" # Join common features into a string
         info["Extra Features (not in all files)"] = ", ".join(extras) if extras else "None" # Join extra features into a string

         tsne_file = generate_tsne_plot(filepath, low_memory=low_memory, sample_size=2000, output_dir=os.path.join(os.path.dirname(os.path.abspath(filepath)), "Data_Separability")) # Generate t-SNE plot
         info["t-SNE Plot"] = tsne_file if tsne_file else "None" # Add t-SNE plot filename or "None"

         report_rows.append(info) # Add the info to the report rows

   if report_rows: # If there are report rows to write
      for i, row in enumerate(report_rows, start=1): # For each report row
         row["#"] = i # Add the counter value

      write_report(report_rows, base_dir, output_filename)
      return True # Return True indicating success
   else: # If no report rows were generated
      return False # Return False indicating failure

def collect_group_files(paths, file_extension=".csv"):
   """
   Collect all matching files for a group of paths.

   :param paths: List of file or directory paths to search
   :param file_extension: File extension to filter (default: ".csv")
   :return: Sorted list of unique file paths
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Collecting {file_extension} files from specified paths...{Style.RESET_ALL}") # Output collection message

   files = [] # Initialize collection list

   for p in paths: # Iterate over each path
      if os.path.isdir(p): # If path is a directory
         files.extend(collect_matching_files(p, file_extension)) # Collect matching files
      elif os.path.isfile(p) and p.endswith(file_extension): # If path is a file with correct extension
         files.append(p) # Add file to list

   return sorted(set(files)) # Remove duplicates and sort

def compute_group_features(files, low_memory=True):
   """
   Compute common and union features for a list of dataset files.

   :param files: List of dataset file paths
   :param low_memory: Whether to optimize memory when reading CSV headers
   :return: Tuple (common_features_set, union_features_set)
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Computing common and union features for dataset group...{Style.RESET_ALL}") # Output computation message

   if not files: # No files, return empty sets
      return set(), set() # Return empty sets

   headers_map = build_headers_map(files, low_memory=low_memory) # Build headers map
   common_features, _ = compute_common_features(headers_map) # Compute common features

   union_features = set() # Initialize union set
   for cols in headers_map.values(): # Iterate over each file's columns
      if cols: # If columns exist
         union_features.update([c.strip().lower() for c in cols]) # Normalize features: strip whitespace and lowercase
   common_features = set([c.strip().lower() for c in common_features]) # Normalize common features: strip whitespace and lowercase

   return set(common_features), union_features # Return both sets

def generate_pairwise_report(group_info):
   """
   Generate pairwise comparison rows from group info.

   :param group_info: Dict mapping group_name -> {"files": [...], "common": set(), "union": set()}
   :return: List of dictionaries representing pairwise comparison rows
   """

   rows = [] # Initialize report row list
   group_names = list(group_info.keys()) # List of group names

   for i in range(len(group_names)): # Iterate over first group
      for j in range(i + 1, len(group_names)): # Iterate over second group avoiding duplicates
         a_name, b_name = group_names[i], group_names[j] # Group names
         a_info, b_info = group_info[a_name], group_info[b_name] # Group info

         if not a_info["files"] and not b_info["files"]: # Skip if both have no files
            continue # Proceed to next pair

         common_between = sorted(a_info["union"] & b_info["union"]) # Features common to both groups
         extras_a = sorted(a_info["union"] - b_info["union"]) # Features in A not in B
         extras_b = sorted(b_info["union"] - a_info["union"]) # Features in B not in A

         row = { # Construct row dictionary
            "Dataset A": a_name, # First dataset group name
            "Dataset B": b_name, # Second dataset group name
            "Files in A": len(a_info["files"]), # Number of files in A
            "Files in B": len(b_info["files"]), # Number of files in B
            "Common Features (A ∩ B)": ", ".join(common_between) or "None", # Common features between A and B
            "Extra Features in A (A \\ B)": ", ".join(extras_a) or "None", # Extra features in A
            "Extra Features in B (B \\ A)": ", ".join(extras_b) or "None", # Extra features in B
         }

         rows.append(row) # Append to report rows

   return rows # Return the list of report rows

def adjust_rows_for_group(report_rows, group_name):
   """
   Adjust pairwise rows so that the target group always appears as Dataset A.

   :param report_rows: List of dictionaries representing pairwise report rows
   :param group_name: Target group to appear as Dataset A
   :return: List of adjusted report rows
   """

   adjusted = [] # Initialize adjusted row list

   for row in report_rows: # Iterate over existing report rows
      if row["Dataset A"] == group_name: # Already Dataset A
         adjusted.append(dict(row)) # Keep as-is
      elif row["Dataset B"] == group_name: # Swap A <-> B
         swapped = { # Construct swapped row
            "Dataset A": row["Dataset B"], # Swap Dataset A
            "Dataset B": row["Dataset A"], # Swap Dataset B
            "Files in A": row["Files in B"], # Swap file counts
            "Files in B": row["Files in A"], # Swap file counts
            "Common Features (A ∩ B)": row["Common Features (A ∩ B)"], # Keep common features
            "Extra Features in A (A \\ B)": row["Extra Features in B (B \\ A)"], # Swap extra features
            "Extra Features in B (B \\ A)": row["Extra Features in A (A \\ B)"], # Swap extra features
         }
         adjusted.append(swapped) # Append swapped row
      else: # Unrelated row, keep as-is
         adjusted.append(dict(row)) # Keep as-is

   return adjusted # Return adjusted rows

def generate_cross_dataset_report(datasets_dict, file_extension=".csv", low_memory=True, output_filename=None):
   """
   Generate a cross-dataset feature-compatibility report comparing dataset
   groups defined in `datasets_dict`. Produces pairwise comparisons between
   dataset groups and writes a CSV report named `Cross_{RESULTS_FILENAME}` by
   default into the `RESULTS_DIR`.

   :param datasets_dict: dict mapping dataset group name -> list of paths
   :param file_extension: extension to search for (default: .csv)
   :param low_memory: passed to CSV loader when building headers
   :param output_filename: optional filename to write; defaults to Cross_{RESULTS_FILENAME}
   :return: True on success, False otherwise
   """

   if output_filename is None: # If no output filename is provided
      output_filename = f"Cross_{RESULTS_FILENAME}" # Default to Cross_{RESULTS_FILENAME}

   group_info = {} # Map group_name -> {"files": [...], "common": set(), "union": set()}
   for group_name, paths in datasets_dict.items(): # Iterate over dataset groups
      all_files = collect_group_files(paths, file_extension) # Collect files for this group
      common_features, union_features = compute_group_features(all_files, low_memory=low_memory) # Compute features

      group_info[group_name] = {"files": all_files, "common": set(common_features), "union": union_features} # Store group info

   report_rows = generate_pairwise_report(group_info) # Generate pairwise report rows
   if not report_rows: # If no report rows were generated
      return False # Return False indicating failure

   saved_any = False # Flag to track if any report was saved
   for group_name, info in group_info.items(): # Iterate over each group
      base_dir = os.path.dirname(os.path.abspath(info["files"][0])) if info["files"] else os.getcwd() # Base dir from first file or current dir
      adjusted_rows = adjust_rows_for_group(report_rows, group_name) # Adjust rows for this group
      try: # Try to write the report
         write_report(adjusted_rows, base_dir, output_filename) # Write the report
         saved_any = True # Mark that at least one report was saved
      except Exception: # Fail silently
         pass # Do nothing on failure

   return saved_any # Return whether any report was saved

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

   print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Dataset Descriptor{BackgroundColors.GREEN}!{Style.RESET_ALL}", end="\n\n") # Output the Welcome message
   start_time = datetime.datetime.now() # Get the start time of the program

   for dataset_name, paths in DATASETS.items(): # For each dataset in the DATASETS dictionary
      print(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing dataset: {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}")
      safe_dataset_name = str(dataset_name).replace(" ", "_").replace("/", "_") # Create a safe dataset name for filenames

      for dir_path in paths: # For each path in the list of paths for the dataset
         print(f" {BackgroundColors.GREEN}Location: {BackgroundColors.CYAN}{dir_path}{Style.RESET_ALL}")
         if not verify_filepath_exists(dir_path): # Verify path exists
            print(f"{BackgroundColors.RED}The specified input path does not exist: {BackgroundColors.CYAN}{dir_path}{Style.RESET_ALL}")
            continue # Skip to next configured path
         
         success = generate_dataset_report(dir_path, file_extension=".csv", low_memory=True, output_filename=RESULTS_FILENAME) # Generate the dataset report
         if not success: # If the report was not generated successfully
            print(f"{BackgroundColors.RED}Failed to generate dataset report for: {BackgroundColors.CYAN}{dir_path}{Style.RESET_ALL}")
         else: # If the report was generated successfully
            print(f"{BackgroundColors.GREEN}Report saved for {BackgroundColors.CYAN}{dataset_name}{BackgroundColors.GREEN} -> {BackgroundColors.CYAN}{RESULTS_FILENAME}{Style.RESET_ALL}")
   
   if CROSS_DATASET_VALIDATE and len(DATASETS) > 1: # If cross-dataset validation is enabled and there is more than one dataset
      try: # Try to generate the cross-dataset report
         success = generate_cross_dataset_report(DATASETS, file_extension=".csv") # Generate the cross-dataset report
         if success: # If the report was generated successfully
            print(f"{BackgroundColors.GREEN}Cross-dataset report saved -> {BackgroundColors.CYAN}Cross_{RESULTS_FILENAME}{Style.RESET_ALL}")
         else: # If no comparisons were generated
            print(f"{BackgroundColors.YELLOW}No cross-dataset comparisons generated (no files found).{Style.RESET_ALL}")
      except Exception as e: # If an error occurs during cross-dataset validation
         print(f"{BackgroundColors.RED}Cross-dataset validation failed: {e}{Style.RESET_ALL}")

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
