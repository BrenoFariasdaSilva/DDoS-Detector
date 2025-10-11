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
import matplotlib.pyplot as plt # For plotting
import numpy as np # For numerical operations
import os # For file and directory operations
import pandas as pd # For data manipulation
import platform # For getting the operating system name
import re # For regular expressions
import seaborn as sns # For advanced plots
import time # For measuring elapsed time
from colorama import Style # For coloring the terminal
from sklearn.ensemble import RandomForestClassifier # For the Random Forest model
from sklearn.feature_selection import RFE # For Recursive Feature Elimination
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix # For performance metrics
from sklearn.model_selection import train_test_split # For splitting the data
from sklearn.preprocessing import StandardScaler # For scaling the data (standardization)

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

def safe_filename(name):
   """
   Converts a string to a safe filename by replacing invalid characters with underscores.

   :param name: The original string
   :return: A safe filename string
   """

   return re.sub(r'[\\/*?:"<>|]', "_", name) # Replace invalid characters with underscores

def verify_filepath_exists(filepath):
   """
   Verify if a file or folder exists at the specified path.

   :param filepath: Path to the file or folder
   :return: True if the file or folder exists, False otherwise
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Verifying if the file or folder exists at the path: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}") # Output the verbose message
   return os.path.exists(filepath) # Return True if the file or folder exists, False otherwise

def load_and_clean_data(csv_path):
   """
   Loads the CSV dataset, selects numeric features, encodes target if necessary,
   and drops invalid values.

   :param csv_path: Path to the CSV dataset file
   :return: X (DataFrame of numeric features), y (target Series)
   """

   if not verify_filepath_exists(csv_path): # If the CSV file does not exist
      print(f"{BackgroundColors.RED}CSV file not found: {csv_path}{Style.RESET_ALL}")
      return None, None # Return None if file not found

   print(f"\n{BackgroundColors.GREEN}Loading {BackgroundColors.CYAN}{csv_path}{BackgroundColors.GREEN} CSV dataset file...{Style.RESET_ALL}")
   df = pd.read_csv(csv_path, low_memory=False) # Load the dataset

   if df.shape[1] < 2: # If there are less than 2 columns
      print(f"{BackgroundColors.RED}CSV must contain at least one feature column and one target column.{Style.RESET_ALL}")
      return None, None # Return None if not enough columns

   X = df.iloc[:, :-1] # All columns except the last
   y = df.iloc[:, -1] # Last column as target

   if y.dtype == object or y.dtype.name == "category": # If target is categorical
      y, _ = pd.factorize(y) # Encode target labels as integers

   X = X.select_dtypes(include=["number"]).replace([np.inf, -np.inf], np.nan).dropna() # Keep only numeric columns and drop rows with NaN or infinite values
   y = y[X.index] # Align target with cleaned features

   if X.empty: # If no numeric features remain
      print(f"{BackgroundColors.RED}No valid numeric features remain after cleaning.{Style.RESET_ALL}")
      return None, None # Return None if no valid features

   return X, y # Return features and target

def scale_and_split(X, y, test_size=0.2, random_state=42):
   """
   Scales numeric features and splits into train/test sets.

   :param X: Features DataFrame
   :param y: Target Series
   :param test_size: Proportion of the dataset to include in the test split
   :param random_state: Random seed for reproducibility
   :return: X_train, X_test, y_train, y_test
   """

   scaler = StandardScaler() # Initialize the scaler
   X_scaled = scaler.fit_transform(X) # Scale the features
   X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state) # Split into train/test sets

   return X_train, X_test, y_train, y_test # Return the split data

def run_rfe_selector(X_train, y_train, n_select=10):
   """
   Runs RFE with RandomForestClassifier and returns the selector object.

   :param X_train: Training features
   :param y_train: Training target
   :param n_select: Number of features to select
   :return: selector (fitted RFE object)
   """

   model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # Initialize the Random Forest model
   n_features = X_train.shape[1] # Get the number of features
   n_select = n_select if n_features >= n_select else n_features # Adjust n_select if more than available features

   selector = RFE(model, n_features_to_select=n_select, step=1) # Initialize RFE
   selector = selector.fit(X_train, y_train) # Fit RFE

   return selector, model # Return the fitted selector and model

def compute_rfe_metrics(selector, model, X_train, X_test, y_train, y_test):
   """
   Computes performance metrics using the RFE-selected estimator.

   :param selector: Fitted RFE object
   :param model: Base estimator used in RFE
   :param X_train: Training features
   :param X_test: Testing features
   :param y_train: Training target
   :param y_test: Testing target
   :return: metrics tuple (acc, prec, rec, f1, fpr, fnr, elapsed_time)
   """

   start_time = time.time() # Start time measurement
   y_pred = selector.estimator_.fit(X_train, y_train).predict(X_test) # Fit and predict using the RFE-selected estimator
   acc = accuracy_score(y_test, y_pred) # Calculate accuracy
   prec = precision_score(y_test, y_pred, average="weighted", zero_division=0) # Calculate precision
   rec = recall_score(y_test, y_pred, average="weighted", zero_division=0) # Calculate recall
   f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0) # Calculate F1-score

   if len(np.unique(y_test)) == 2: # If binary classification
      tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel() # Get confusion matrix values
      fpr = fp / (fp + tn) if (fp + tn) > 0 else 0 # Calculate false positive rate
      fnr = fn / (fn + tp) if (fn + tp) > 0 else 0 # Calculate false negative rate
   else: # For multi-class classification
      fpr, fnr = 0, 0 # Set FPR and FNR to 0

   elapsed_time = time.time() - start_time # Calculate elapsed time
   return acc, prec, rec, f1, fpr, fnr, elapsed_time # Return the metrics

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

def run_rfe(csv_path):
   """
   Runs Recursive Feature Elimination on the provided dataset, prints top features,
   and saves structured results including optional performance metrics.

   :param csv_path: Path to the CSV dataset file
   :return: None
   """

   X, y = load_and_clean_data(csv_path) # Load and clean the dataset

   if X is None or y is None: # If loading failed
      return # Exit the function

   X_train, X_test, y_train, y_test = scale_and_split(X, y) # Scale and split the data
   selector, model = run_rfe_selector(X_train, y_train) # Run RFE to select top features

   metrics = compute_rfe_metrics(selector, model, X_train, X_test, y_train, y_test) # Compute performance metrics

   output_file = f"{os.path.dirname(csv_path)}/Feature_Analysis/RFE_results_{model.__class__.__name__}.txt" # Define output file path
   os.makedirs(os.path.dirname(output_file), exist_ok=True) # Create directory if it doesn't exist

   with open(output_file, "w", encoding="utf-8") as f: # Write results to file
      header = f"RFE Results with {n_select} Selected Features and the classifier {model.__class__.__name__}:\n" # Header for the results file
      f.write(header) # Write header to file

      for idx, (col_idx, feature, status, ranking) in enumerate(results, start=1): # Write each result to the file
         plain_status = "Selected" if "Selected" in status else "Not Selected" # Plain text status
         line = f"{idx} - Column {col_idx}: {feature}: {plain_status} (Rank {ranking})" # Format line
         f.write(line + "\n") # Write line to file

   top_features = [feature for (_, feature, status, ranking) in results if "Selected" in status] # List of top features

   if top_features: # Analyze top features if any were selected
      analyze_top_features(X, y, top_features, csv_path=csv_path) # Analyze top features

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
   
   print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Recursive Feature Elimination (RFE){BackgroundColors.GREEN} program!{Style.RESET_ALL}") # Output the welcome message

   csv_file = "./Datasets/DDoS/CICDDoS2019/01-12/DrDoS_DNS.csv" # Path to the CSV file
   run_rfe(csv_file) # Run RFE on the specified CSV file

   print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}") # Output the end of the program message

   atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None # Register play_sound at exit if enabled

if __name__ == "__main__":
   """
   This is the standard boilerplate that calls the main() function.

   :return: None
   """
   
   main() # Call the main function
