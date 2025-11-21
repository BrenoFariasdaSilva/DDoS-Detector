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
import numpy as np # For numerical operations
import os # For file and directory operations
import pandas as pd # For data manipulation
import platform # For getting the operating system name
import time # For measuring elapsed time
from colorama import Style # For coloring the terminal
from sklearn.decomposition import PCA # For Principal Component Analysis
from sklearn.ensemble import RandomForestClassifier # For the Random Forest model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix # For performance metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score # For splitting and cross-validation
from sklearn.preprocessing import StandardScaler # For scaling the data (standardization)
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

	df.columns = df.columns.str.strip() # Clean column names by stripping leading/trailing whitespace

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

	print(f"{BackgroundColors.GREEN}Dataset loaded: {BackgroundColors.CYAN}{X.shape[0]} samples, {X.shape[1]} features{Style.RESET_ALL}")
	
	return X, y # Return features and target

def scale_and_split(X, y, test_size=0.2, random_state=42):
	"""
	Scales numeric features and splits into train/test sets.

	:param X: Features DataFrame
	:param y: Target Series
	:param test_size: Proportion of the dataset to include in the test split
	:param random_state: Random seed for reproducibility
	:return: X_train, X_test, y_train, y_test, scaler
	"""

	scaler = StandardScaler() # Initialize the scaler
	X_scaled = scaler.fit_transform(X) # Scale the features
	X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=random_state, stratify=y) # Split into train/test sets

	return X_train, X_test, y_train, y_test, scaler # Return the split data and scaler

def apply_pca_and_evaluate(X_train, y_train, X_test, y_test, n_components, cv_folds=10):
	"""
	Applies PCA transformation and evaluates performance using 10-fold Stratified Cross-Validation
	on the training set, then tests on the held-out test set.

	:param X_train: Training features (scaled)
	:param y_train: Training target
	:param X_test: Testing features (scaled)
	:param y_test: Testing target
	:param n_components: Number of principal components to keep
	:param cv_folds: Number of cross-validation folds (default: 10)
	:return: Dictionary containing metrics, explained variance, and PCA object
	"""

	pca = PCA(n_components=n_components, random_state=42) # Initialize PCA
	
	X_train_pca = pca.fit_transform(X_train) # Fit PCA on training data and transform
	X_test_pca = pca.transform(X_test) # Transform test data using the fitted PCA
 
	explained_variance = pca.explained_variance_ratio_.sum() # Total explained variance ratio
	
	model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # Initialize Random Forest model
	
	print(f"{BackgroundColors.GREEN}  Running 10-fold Stratified CV on training data...{Style.RESET_ALL}")
	skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42) # Stratified K-Fold cross-validator
	
	cv_accs, cv_precs, cv_recs, cv_f1s = [], [], [], [] # Lists to store CV metrics
	
	for train_idx, val_idx in tqdm(skf.split(X_train_pca, y_train), total=cv_folds, desc="  CV Folds", leave=False): # Loop over each fold
		X_train_fold = X_train_pca[train_idx] # Training data for this fold
		X_val_fold = X_train_pca[val_idx] # Validation data for this fold
		y_train_fold = y_train.iloc[train_idx] if isinstance(y_train, pd.Series) else y_train[train_idx] # Training target for this fold
		y_val_fold = y_train.iloc[val_idx] if isinstance(y_train, pd.Series) else y_train[val_idx] # Validation target for this fold
		
		model.fit(X_train_fold, y_train_fold) # Fit model on training fold
		y_pred_fold = model.predict(X_val_fold) # Predict on validation fold
		
		cv_accs.append(accuracy_score(y_val_fold, y_pred_fold)) # Calculate and store metrics
		cv_precs.append(precision_score(y_val_fold, y_pred_fold, average="weighted", zero_division=0)) # Calculate and store metrics
		cv_recs.append(recall_score(y_val_fold, y_pred_fold, average="weighted", zero_division=0)) # Calculate and store metrics
		cv_f1s.append(f1_score(y_val_fold, y_pred_fold, average="weighted", zero_division=0)) # Calculate and store metrics
	
	cv_acc_mean = np.mean(cv_accs) # Mean CV metrics
	cv_prec_mean = np.mean(cv_precs) # Mean CV metrics
	cv_rec_mean = np.mean(cv_recs) # Mean CV metrics
	cv_f1_mean = np.mean(cv_f1s) # Mean CV metrics
	
	start_time = time.time() # Start timing for test set evaluation
	model.fit(X_train_pca, y_train) # Fit model on full training data
	y_pred = model.predict(X_test_pca) # Predict on test data
	elapsed_time = time.time() - start_time # Elapsed time for test evaluation
	
	acc = accuracy_score(y_test, y_pred) # Calculate test metrics
	prec = precision_score(y_test, y_pred, average="weighted", zero_division=0) # Calculate test metrics
	rec = recall_score(y_test, y_pred, average="weighted", zero_division=0) # Calculate test metrics
	f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0) # Calculate test metrics
	
	fpr, fnr = 0, 0 # Initialize FPR and FNR
	unique_classes = np.unique(y_test) # Get unique classes in the test set
	if len(unique_classes) == 2: # If binary classification
		tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=unique_classes).ravel() # Get confusion matrix values
		fpr = fp / (fp + tn) if (fp + tn) > 0 else 0 # Calculate False Positive Rate
		fnr = fn / (fn + tp) if (fn + tp) > 0 else 0 # Calculate False Negative Rate
	
	return { # Return all results in a dictionary
		"n_components": n_components,
		"explained_variance": explained_variance,
		"cv_accuracy": cv_acc_mean,
		"cv_precision": cv_prec_mean,
		"cv_recall": cv_rec_mean,
		"cv_f1_score": cv_f1_mean,
		"test_accuracy": acc,
		"test_precision": prec,
		"test_recall": rec,
		"test_f1_score": f1,
		"test_fpr": fpr,
		"test_fnr": fnr,
		"elapsed_time": elapsed_time,
		"pca_object": pca
	}

def print_pca_results(results):
	"""
	Prints PCA results in a formatted way.

	:param results: Dictionary containing PCA evaluation results
	:return: None
	"""
	
	print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}PCA Results (n_components={results['n_components']}):{Style.RESET_ALL}")
	print(f"  {BackgroundColors.GREEN}Explained Variance Ratio: {BackgroundColors.CYAN}{results['explained_variance']:.4f} ({results['explained_variance']*100:.2f}%){Style.RESET_ALL}")
	print(f"\n  {BackgroundColors.BOLD}10-Fold Cross-Validation Metrics (Training Set):{Style.RESET_ALL}")
	print(f"  {BackgroundColors.GREEN}CV Accuracy: {BackgroundColors.CYAN}{results['cv_accuracy']:.4f}{Style.RESET_ALL}")
	print(f"  {BackgroundColors.GREEN}CV Precision: {BackgroundColors.CYAN}{results['cv_precision']:.4f}{Style.RESET_ALL}")
	print(f"  {BackgroundColors.GREEN}CV Recall: {BackgroundColors.CYAN}{results['cv_recall']:.4f}{Style.RESET_ALL}")
	print(f"  {BackgroundColors.GREEN}CV F1-Score: {BackgroundColors.CYAN}{results['cv_f1_score']:.4f}{Style.RESET_ALL}")
	print(f"\n  {BackgroundColors.BOLD}Test Set Metrics:{Style.RESET_ALL}")
	print(f"  {BackgroundColors.GREEN}Test Accuracy: {BackgroundColors.CYAN}{results['test_accuracy']:.4f}{Style.RESET_ALL}")
	print(f"  {BackgroundColors.GREEN}Test Precision: {BackgroundColors.CYAN}{results['test_precision']:.4f}{Style.RESET_ALL}")
	print(f"  {BackgroundColors.GREEN}Test Recall: {BackgroundColors.CYAN}{results['test_recall']:.4f}{Style.RESET_ALL}")
	print(f"  {BackgroundColors.GREEN}Test F1-Score: {BackgroundColors.CYAN}{results['test_f1_score']:.4f}{Style.RESET_ALL}")
	print(f"  {BackgroundColors.GREEN}Test FPR: {BackgroundColors.CYAN}{results['test_fpr']:.4f}{Style.RESET_ALL}")
	print(f"  {BackgroundColors.GREEN}Test FNR: {BackgroundColors.CYAN}{results['test_fnr']:.4f}{Style.RESET_ALL}")
	print(f"  {BackgroundColors.GREEN}Elapsed Time: {BackgroundColors.CYAN}{results['elapsed_time']:.2f}s{Style.RESET_ALL}")

def run_pca_analysis(csv_path, n_components_list=[8, 16, 24, 32]):
	"""
	Runs PCA analysis with different numbers of components and evaluates performance.

	:param csv_path: Path to the CSV dataset file
	:param n_components_list: List of component counts to test
	:return: None
	"""

	X, y = load_and_clean_data(csv_path) # Load and clean the dataset
	
	if X is None or y is None: # If loading failed
		return # Exit the function
	
	max_components = min(X.shape[1], max(n_components_list)) # Maximum valid components
	n_components_list = [n for n in n_components_list if n <= max_components] # Filter valid component counts
	
	if not n_components_list: # If no valid component counts remain
		print(f"{BackgroundColors.RED}No valid component counts. Dataset has only {X.shape[1]} features.{Style.RESET_ALL}")
		return # Exit the function
	
	print(f"\n{BackgroundColors.CYAN}PCA Configuration:{Style.RESET_ALL}")
	print(f"  {BackgroundColors.GREEN}• Testing components: {BackgroundColors.CYAN}{n_components_list}{Style.RESET_ALL}")
	print(f"  {BackgroundColors.GREEN}• Evaluation: {BackgroundColors.CYAN}10-Fold Stratified Cross-Validation{Style.RESET_ALL}")
	print(f"  {BackgroundColors.GREEN}• Model: {BackgroundColors.CYAN}Random Forest (100 estimators){Style.RESET_ALL}")
	print(f"  {BackgroundColors.GREEN}• Split: {BackgroundColors.CYAN}80/20 (train/test){Style.RESET_ALL}\n")
	
	X_train, X_test, y_train, y_test, scaler = scale_and_split(X, y) # Scale and split the data
	
	all_results = [] # List to store all results
	
	for n_components in tqdm(n_components_list, desc=f"{BackgroundColors.GREEN}PCA Analysis{Style.RESET_ALL}", unit="config"): # Loop over each number of components
		print(f"\n{BackgroundColors.BOLD}Testing PCA with {BackgroundColors.CYAN}{n_components}{BackgroundColors.GREEN} components...{Style.RESET_ALL}")
		
		results = apply_pca_and_evaluate(X_train, y_train, X_test, y_test, n_components) # Apply PCA and evaluate
		all_results.append(results) # Store the results
		print_pca_results(results) if VERBOSE else None # Print results if VERBOSE is True
	
	save_pca_results(csv_path, all_results) # Save all results to files
	
	best_result = max(all_results, key=lambda x: x['cv_f1_score']) # Find the best configuration based on CV F1-Score
 
	print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Best Configuration:{Style.RESET_ALL}")
	print(f"  {BackgroundColors.GREEN}n_components = {BackgroundColors.CYAN}{best_result['n_components']}{Style.RESET_ALL}")
	print(f"  {BackgroundColors.GREEN}CV F1-Score = {BackgroundColors.CYAN}{best_result['cv_f1_score']:.4f}{Style.RESET_ALL}")
	print(f"  {BackgroundColors.GREEN}Explained Variance = {BackgroundColors.CYAN}{best_result['explained_variance']:.4f}{Style.RESET_ALL}")

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
