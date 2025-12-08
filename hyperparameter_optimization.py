"""
================================================================================
Classifiers Hyperparameter Optimization
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-12-06
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
   1. Configure the dataset path in the main() function (csv_path variable).
   2. Execute the script via Python:
         $ python hyperparameter_optimization.py
   3. Check the output CSV file for optimized hyperparameters and scores.

Outputs:
   - Hyperparameter_Optimization_Results.csv: Contains best parameters and CV scores for each model
   - Terminal logs showing optimization progress and execution time
   - Sound notification when processing completes

TODOs:
   - Add CLI argument parsing for dataset path and output directory
   - Implement Bayesian optimization for more efficient search
   - Add support for custom parameter grids
   - Extend to support additional classifiers
   - Add visualization of hyperparameter importance

Dependencies:
   - Python >= 3.8
   - pandas
   - numpy
   - scikit-learn
   - lightgbm
   - xgboost
   - tqdm
   - colorama

Assumptions & Notes:
   - Dataset CSV files with last column as target variable
   - Genetic Algorithm results available in Feature_Analysis/Genetic_Algorithm_Results.csv
   - Numeric features only (non-numeric features are dropped)
   - Uses weighted F1-score for multi-class classification
   - Sound notification skipped on Windows platform
"""

import atexit # For playing a sound when the program finishes
import datetime # For getting the current date and time
import json # For handling JSON strings
import lightgbm as lgb # For LightGBM model
import numpy as np # For numerical operations
import os # For running a command in the terminal
import pandas as pd # For data manipulation
import platform # For getting the operating system name
from colorama import Style # For coloring the terminal
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier # For ensemble models
from sklearn.linear_model import LogisticRegression # For logistic regression model
from sklearn.metrics import make_scorer, f1_score # For custom scoring metrics
from sklearn.model_selection import GridSearchCV, train_test_split # For hyperparameter search and data splitting
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid # For k-nearest neighbors model
from sklearn.neural_network import MLPClassifier # For neural network model
from sklearn.preprocessing import LabelEncoder, StandardScaler # For label encoding and feature scaling
from sklearn.svm import SVC # For Support Vector Machine model
from tqdm import tqdm # For progress bars
from xgboost import XGBClassifier # For XGBoost classifier

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
N_JOBS = -1 # Number of parallel jobs for GridSearchCV (-1 uses all processors)
RESULTS_FILENAME = "Hyperparameter_Optimization_Results.csv" # Filename for saving results
IGNORE_FILES = [RESULTS_FILENAME] # List of filenames to ignore when searching for datasets
IGNORE_DIRS = ["Cache", "Dataset_Description", "Data_Separability", "Feature_Analysis"] # List of directory names to ignore when searching for datasets

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
      
      if any(ignore and (ignore == filename or ignore == item_path) for ignore in IGNORE_FILES): # If the file is in the IGNORE_FILES list
         verbose_output(f"{BackgroundColors.YELLOW}Ignoring file {BackgroundColors.CYAN}{filename}{BackgroundColors.YELLOW} listed in IGNORE_FILES{Style.RESET_ALL}")
         continue # Skip this file
      
      if os.path.isfile(item_path) and item.lower().endswith(file_extension): # If the item is a file and has the specified extension
         files.append(item_path) # Add the file to the list

   return sorted(files) # Return sorted list for consistency

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

def get_models_and_param_grids():
   """
   Returns a dictionary of models with their corresponding hyperparameter grids for GridSearchCV.

   :param: None
   :return: Dictionary with model names as keys and tuples (model_instance, param_grid) as values
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Initializing models and parameter grids for hyperparameter optimization...{Style.RESET_ALL}") # Output the verbose message
   
   return { # Dictionary of models and their parameter grids
      "Random Forest": (
         RandomForestClassifier(random_state=42), # Random Forest classifier
         {
            "n_estimators": [50, 100, 200], # Number of trees in the forest
            "max_depth": [None, 10, 20, 30], # Maximum depth of the tree
            "min_samples_split": [2, 5, 10], # Minimum number of samples required to split an internal node
            "min_samples_leaf": [1, 2, 4], # Minimum number of samples required to be at a leaf node
            "max_features": ["sqrt", "log2", None] # Number of features to consider when looking for the best split
         }
      ),
      "SVM": (
         SVC(random_state=42, probability=True), # Enable probability estimates for SVC
         {
            "C": [0.1, 1, 10, 100], # Regularization parameter
            "kernel": ["linear", "rbf", "poly"], # Kernel type to be used in the algorithm
            "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1] # Kernel coefficient
         }
      ),
      "XGBoost": (
         XGBClassifier(eval_metric="mlogloss", random_state=42), # XGBoost classifier
         {
            "n_estimators": [50, 100, 200], # Number of trees in the forest
            "max_depth": [3, 5, 7, 10], # Maximum depth of the tree
            "learning_rate": [0.01, 0.1, 0.3], # Step size shrinkage
            "subsample": [0.6, 0.8, 1.0], # Subsample ratio of the training instances
            "colsample_bytree": [0.6, 0.8, 1.0] # Subsample ratio of columns when constructing each tree
         }
      ),
      "Logistic Regression": (
         LogisticRegression(max_iter=1000, random_state=42), # Logistic Regression classifier
         {
            "C": [0.001, 0.01, 0.1, 1, 10, 100], # Inverse of regularization strength
            "penalty": ["l1", "l2", "elasticnet", None], # Norm used in the penalization
            "solver": ["lbfgs", "liblinear", "saga"], # Algorithm to use in the optimization problem
            "l1_ratio": [0.0, 0.5, 1.0] # The Elastic-Net mixing parameter
         }
      ),
      "KNN": (
         KNeighborsClassifier(), # K-Nearest Neighbors classifier
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
         lgb.LGBMClassifier(force_row_wise=True, random_state=42, verbosity=-1), # LightGBM classifier
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

def optimize_model(model_name, model, param_grid, X_train, y_train):
   """
   Performs hyperparameter optimization for a single model using GridSearchCV.

   :param model_name: Name of the model (for logging).
   :param model: The classifier model instance.
   :param param_grid: Dictionary of hyperparameters to search.
   :param X_train: Training features (scaled numpy array).
   :param y_train: Training target labels (encoded).
   :return: Tuple (best_params, best_score, cv_results) or (None, None, None) if optimization fails
   """
   
   verbose_output(f"{BackgroundColors.GREEN}Optimizing {BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN}...{Style.RESET_ALL}") # Output the verbose message
   
   try: # Try to perform grid search
      f1_scorer = make_scorer(f1_score, average="weighted") # Create F1 scorer for multi-class problems
      
      grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=CV_FOLDS, scoring=f1_scorer, n_jobs=N_JOBS, verbose=0, error_score="raise") # Initialize GridSearchCV
      
      grid_search.fit(X_train, y_train) # Fit the grid search
      
      best_params = grid_search.best_params_ # Get the best parameters
      best_score = grid_search.best_score_ # Get the best cross-validation score
      
      print(f"{BackgroundColors.GREEN}{model_name} - Best CV F1 Score: {BackgroundColors.CYAN}{best_score:.4f}{Style.RESET_ALL}")
      verbose_output(f"{BackgroundColors.GREEN}Best parameters: {BackgroundColors.CYAN}{best_params}{Style.RESET_ALL}")
      
      return best_params, best_score, grid_search.cv_results_ # Return optimization results
      
   except Exception as e: # Catch any errors during optimization
      print(f"{BackgroundColors.RED}Error optimizing {model_name}: {e}{Style.RESET_ALL}")
      return None, None, None # Return None values if optimization failed

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
   
   file_dir = os.path.dirname(csv_path) # Get the directory of the dataset
   output_path = os.path.join(file_dir, RESULTS_FILENAME) # Construct output path
   
   try: # Try to save the results
      df_results = pd.DataFrame(results_list) # Convert results list to DataFrame
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

   print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Main Template Python{BackgroundColors.GREEN} program!{Style.RESET_ALL}", end="\n\n") # Output the welcome message
   start_time = datetime.datetime.now() # Get the start time of the program
   
   # Your code goes here

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
