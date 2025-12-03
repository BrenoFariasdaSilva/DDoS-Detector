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
import gc # For explicit garbage collection
import numpy as np # For numerical operations
import os # For running a command in the terminal
import pandas as pd # For data manipulation
import warnings # For suppressing pandas warnings when requested
import platform # For getting the operating system name
from colorama import Style # For coloring the terminal
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
   ]
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

def collect_matching_files(input_dir, file_format=".csv", ignore_files=None):
   """
   Recursively collects all files in the specified directory and subdirectories
   that match the given file format and are not in the ignore list.

   :param input_dir: Directory to search
   :param file_format: File format to include (default: .csv)
   :param ignore_files: List of filenames to ignore
   :return: Sorted list of matching file paths
   """

   ignore_files = ignore_files or [] # Default to empty list if None
   matching_files = [] # List to store matching file paths
   
   for root, _, files in os.walk(input_dir): # Walk through the directory
      for file in files: # For each file
         if file.endswith(file_format) and file not in ignore_files: # Verify if it matches the format and is not ignored
            matching_files.append(os.path.join(root, file)) # Add the full file path to the list
   
   sorted_matching_files = sorted(set(matching_files)) # Remove duplicates and sort the list
   return sorted_matching_files # Return the sorted list of matching files

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
# t-SNE related functionality removed. We no longer compute embeddings or
# separability scores; related plotting and preprocessing helpers were removed
# to simplify the dataset descriptor output.

def get_dataset_info(filepath, low_memory=True):
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
   
   # t-SNE embedding and separability computation removed.

   result = { # Return the dataset information as a dictionary
      "Dataset Name": os.path.basename(filepath),
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

   report_csv_path = os.path.join(base_dir, output_filename) # Path to save the report CSV
   report_df.to_csv(report_csv_path, index=False) # Save the report to a CSV file

def generate_dataset_report(input_path, file_extension=".csv", low_memory=True, output_filename="_dataset_descriptor.csv"):
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
   ignore_files = [output_filename] # Ignore the output CSV itself
   sorted_matching_files = [] # List to store matching files

   if os.path.isdir(input_path): # If the input path is a directory
      print(f"{BackgroundColors.GREEN}Scanning directory {BackgroundColors.CYAN}{input_path}{BackgroundColors.GREEN} for {BackgroundColors.CYAN}{file_extension}{BackgroundColors.GREEN} files...{Style.RESET_ALL}") # Output scanning message
      sorted_matching_files = collect_matching_files(input_path, file_extension, ignore_files) # Collect matching files
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

   progress = tqdm(sorted_matching_files, desc=f"{BackgroundColors.GREEN}Processing files{Style.RESET_ALL}", unit="file") # Create a progress bar
   for idx, filepath in enumerate(progress, 1): # Process each matching file
      progress.set_description(f"{BackgroundColors.GREEN}Processing file {BackgroundColors.CYAN}{idx}/{len(sorted_matching_files)}{BackgroundColors.GREEN}: {BackgroundColors.CYAN}{os.path.basename(filepath)}{Style.RESET_ALL}") # Update progress bar description
      info = get_dataset_info(filepath, low_memory) # Get dataset info
      if info: # If info was successfully retrieved
         relative_path = os.path.relpath(filepath, base_dir) # Get path relative to base_dir
         info["Dataset Name"] = relative_path.replace("\\", "/") # Use relative path for Dataset Name and normalize slashes
         report_rows.append(info) # Add the info to the report rows

   if report_rows: # If there are report rows to write
      for i, row in enumerate(report_rows, start=1): # For each report row
         row["#"] = i # Add the counter value

      write_report(report_rows, base_dir, output_filename)
      return True # Return True indicating success
   else: # If no report rows were generated
      return False # Return False indicating failure

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

   for dataset_name, paths in DATASETS.items(): # For each dataset in the DATASETS dictionary
      print(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing dataset: {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}")
      safe_dataset_name = str(dataset_name).replace(" ", "_").replace("/", "_") # Create a safe dataset name for filenames

      for input_path in paths: # For each path in the list of paths for the dataset
         print(f" {BackgroundColors.GREEN}Location: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}")
         if not verify_filepath_exists(input_path): # Verify path exists
            print(f"{BackgroundColors.RED}The specified input path does not exist: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}")
            continue # Skip to next configured path

         output_filename = f"_dataset_descriptor.csv" # Create output filename based on dataset name
         
         success = generate_dataset_report(input_path, file_extension=".csv", low_memory=True, output_filename=output_filename) # Generate the dataset report
         if not success: # If the report was not generated successfully
            print(f"{BackgroundColors.RED}Failed to generate dataset report for: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}")
         else: # If the report was generated successfully
            print(f"{BackgroundColors.GREEN}Report saved for {BackgroundColors.CYAN}{dataset_name}{BackgroundColors.GREEN} -> {BackgroundColors.CYAN}{output_filename}{Style.RESET_ALL}")

   print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}") # Output the end of the program message

   atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None # Register the play_sound function to be called when the program finishes

if __name__ == "__main__":
   """
   This is the standard boilerplate that calls the main() function.

   :return: None
   """

   main() # Call the main function
