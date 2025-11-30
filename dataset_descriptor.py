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
import matplotlib.pyplot as plt # For plotting
import numpy as np # For numerical operations
import os # For running a command in the terminal
import pandas as pd # For data manipulation
import platform # For getting the operating system name
from colorama import Style # For coloring the terminal
from itertools import combinations # For computing pairwise combinations
from sklearn.manifold import TSNE # For data separability analysis
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

def get_files_to_process(directory_path, file_extension=".csv"):
   """
   Get all of the specified files in a directory (non-recursive).
   
   :param directory_path: Path to the directory to search
   :param file_extension: File extension to filter (default: .csv)
   :return: List of files with the specified extension
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Getting all {BackgroundColors.CYAN}{file_extension}{BackgroundColors.GREEN} files in the directory: {BackgroundColors.CYAN}{directory_path}{Style.RESET_ALL}") # Output the verbose message

   if not os.path.isdir(directory_path): # If the path is not a directory
      verbose_output(f"{BackgroundColors.RED}The specified path is not a directory: {BackgroundColors.CYAN}{directory_path}{Style.RESET_ALL}") # Output the verbose message
      return [] # Return an empty list

   files = [] # List to store the files

   for item in os.listdir(directory_path): # List all items in the directory
      item_path = os.path.join(directory_path, item) # Get the full path of the item
      
      if os.path.isfile(item_path) and item.lower().endswith(file_extension): # If the item is a file and has the specified extension
         files.append(item_path) # Add the file to the list

   return sorted(files) # Return sorted list for consistency

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

def load_dataset(filepath, low_memory=False):
   """
   Loads a dataset from a CSV file.
   
   :param filepath: Path to the CSV file
   :param low_memory: Whether to use low memory mode (default: False)
   :return: Pandas DataFrame
   """

   try: # Try to load the dataset
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

def compute_tsne_separability(df, label_col, random_state=42):
   """
   Computes a basic t-SNE separability score for the dataset.

   The score is based on the average Euclidean distance between class centroids
   in the 2D t-SNE projection. A higher value indicates better class separability.

   :param df: pandas DataFrame
   :param label_col: Name of the label column
   :param random_state: Random seed for reproducibility (default: 42)
   :return: Float separability score, or "N/A" if not applicable
   """

   verbose_output(f"{BackgroundColors.GREEN}Computing t-SNE separability score...{Style.RESET_ALL}") # Output verbose message

   if label_col is None or label_col not in df.columns: # If no label column is found
      return "N/A" # Return "N/A"

   numeric_df = df.select_dtypes(include=["float64", "int64", "Int64"]) # Select numeric columns
   if numeric_df.empty: # If there are no numeric features
      return "N/A" # Return "N/A"

   try: # Try to compute t-SNE
      tsne = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto") # Initialize t-SNE
      tsne_result = tsne.fit_transform(numeric_df.fillna(0)) # Fit and transform numeric features

      temp_df = pd.DataFrame(tsne_result, columns=["TSNE1", "TSNE2"]) # Create DataFrame with t-SNE results
      temp_df[label_col] = df[label_col].values # Add label column

      centroids = temp_df.groupby(label_col)[["TSNE1", "TSNE2"]].mean() # Compute class centroids
      if len(centroids) < 2: # If less than 2 classes exist
         return "N/A" # Return "N/A"

      distances = [np.linalg.norm(c1 - c2) for c1, c2 in combinations(centroids.values, 2)] # Compute distances
      separability_score = float(np.mean(distances)) # Average distance as score

      return round(separability_score, 4) # Return rounded score
   except Exception as e: # Handle exceptions gracefully
      verbose_output(f"{BackgroundColors.RED}t-SNE separability computation failed: {e}{Style.RESET_ALL}") # Output verbose message
      return "N/A" # Return "N/A" if computation fails

def save_tsne_plot(df, label_col, dataset_name, dataset_dir, random_state=42):
   """
   Generates and saves a 3D t-SNE scatter plot colored by class labels.

   The plot visually represents class separability in a 3D embedding space.

   :param df: pandas DataFrame
   :param label_col: Label column name
   :param dataset_name: Dataset filename (used for naming the PNG)
   :param dataset_dir: Directory of the dataset
   :param random_state: Random seed for reproducibility
   """

   if label_col is None or label_col not in df.columns:
      return # Skip if no labels present

   numeric_df = df.select_dtypes(include=["float64", "int64", "Int64"]) # Select numeric columns
   if numeric_df.empty:
      return # Skip if no numeric features

   try: # Try to generate t-SNE plot
      tsne = TSNE(n_components=3, random_state=random_state, init="pca", learning_rate="auto") # Initialize t-SNE
      tsne_result = tsne.fit_transform(numeric_df.fillna(0)) # Fit and transform numeric features

      tsne_df = pd.DataFrame(tsne_result, columns=["TSNE1", "TSNE2", "TSNE3"]) # Create DataFrame with t-SNE results
      tsne_df[label_col] = df[label_col].values # Add label column

      fig = plt.figure(figsize=(8, 6)) # Create figure
      ax = fig.add_subplot(111, projection="3d") # Add 3D subplot

      for label in tsne_df[label_col].unique(): # Plot each class separately
         subset = tsne_df[tsne_df[label_col] == label] # Subset for the class
         ax.scatter(subset["TSNE1"], subset["TSNE2"], subset["TSNE3"], s=30, alpha=0.8, label=str(label)) # Scatter plot for the class

      ax.set_title(f"3D t-SNE Class Distribution – {dataset_name}", fontsize=12) # Set title
      ax.set_xlabel("TSNE1") # Set x-axis label
      ax.set_ylabel("TSNE2") # Set y-axis label
      ax.set_zlabel("TSNE3") # Set z-axis label
      ax.legend(title=label_col, loc="best", fontsize=8) # Add legend
      plt.tight_layout() # Adjust layout

      output_dir = os.path.join(dataset_dir, "Data_Separability") # Set output directory to dataset path with Data_Separability subdir
      os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists
      image_path = os.path.join(output_dir, f"TSNE_3D_{os.path.splitext(dataset_name)[0]}.png") # Image path
      plt.savefig(image_path, dpi=150) # Save the figure
      plt.close() # Close the plot to free memory

      verbose_output(f"{BackgroundColors.GREEN}3D t-SNE plot saved: {image_path}{Style.RESET_ALL}")
   except Exception as e: # Handle exceptions gracefully
      verbose_output(f"{BackgroundColors.RED}3D t-SNE plot generation failed: {e}{Style.RESET_ALL}")

def get_dataset_info(filepath, low_memory=False):
   """
   Extracts dataset information from a CSV file and returns it as a dictionary.

   :param filepath: Path to the CSV file
   :param low_memory: Whether to use low memory mode when loading the CSV (default: False)
   :return: Dictionary containing dataset information
   """

   df = load_dataset(filepath, low_memory) # Load the dataset
   
   if df is None: # If the dataset could not be loaded
      return None # Return None
   
   cleaned_df = preprocess_dataframe(df) # Preprocess the DataFrame

   label_col = detect_label_column(cleaned_df.columns) # Try to detect the label column
   n_samples, n_features, n_numeric, n_int, n_categorical, n_other, categorical_cols_str = summarize_features(cleaned_df) # Summarize features
   missing_summary = summarize_missing_values(cleaned_df) # Summarize missing values
   classes_str, class_dist_str = summarize_classes(cleaned_df, label_col) # Summarize classes and distributions
   tsne_separability = compute_tsne_separability(cleaned_df, label_col) # Compute t-SNE separability score
   save_tsne_plot(cleaned_df, label_col, os.path.basename(filepath), os.path.dirname(filepath)) # Generate and save 3D t-SNE visualization

   result = { # Return the dataset information as a dictionary
      "Dataset Name": os.path.basename(filepath),
      "Number of Samples": f"{n_samples:,}", # Format with commas for readability
      "Number of Features": f"{n_features:,}", # Format with commas for readability
      "Feature Types": f"{n_numeric} numeric (float64), {n_int} integer (int64), {n_categorical} categorical (object/category/bool/string), {n_other} other",
      "Categorical Features (object/string)": categorical_cols_str,
      "Missing Values": missing_summary,
      "Classes": classes_str,
      "Class Distribution": class_dist_str,
      "t-SNE Separability Score": tsne_separability,
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
   report_csv_path = os.path.join(base_dir, output_filename) # Path to save the report CSV
   report_df.to_csv(report_csv_path, index=False) # Save the report to a CSV file

def generate_dataset_report(input_path, file_extension=".csv", low_memory=False, output_filename="_dataset_descriptor.csv"):
   """
   Generates a CSV report for the specified input path.
   The Dataset Name column will include subdirectories if present.

   :param input_path: Directory or file path containing the dataset
   :param file_extension: File extension to filter (default: .csv)
   :param low_memory: Whether to use low memory mode when loading CSVs (default: False)
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
      write_report(report_rows, base_dir, output_filename)
      print(f"\n{BackgroundColors.GREEN}Report saved to: {BackgroundColors.CYAN}{output_filename}{Style.RESET_ALL}") # Output the path to the saved report
      return True # Return True indicating success
   else: # If no report rows were generated
      print(f"\n{BackgroundColors.RED}No valid CSV files found in the specified path: {input_path}{Style.RESET_ALL}") # Output the error message
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

         output_filename = f"_{safe_dataset_name}_dataset_descriptor.csv" # Create output filename based on dataset name
         
         success = generate_dataset_report(input_path, file_extension=".csv", low_memory=False, output_filename=output_filename) # Generate the dataset report
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
