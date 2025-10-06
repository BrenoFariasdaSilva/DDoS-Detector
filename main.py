# pip install colorama pandas scikit-learn xgboost lightgbm shap lime matplotlib numpy

#@TODO: Add cross validation
#@TODO: add threads for each algorithm training?
#@TODO: search and review the code for overfitting issues
#@TODO: Seleção de Atributos com Base em Correlação + Regressão Lasso
# 1. Remover atributos com baixa variância (quase constantes)
# 2. Analisar a correlação com a variável target
# 3. Remover atributos altamente correlacionados entre si (multicolinearidade)
# 4. Aplicar regressão Lasso (com L1 regularization)
#@TODO: Implementar a função de seleção de atributos
#@TODO: Adicionar testes para a função de seleção de atributos

import arff as liac_arff # For loading ARFF files
import atexit # For registering a function to run at exit
import lightgbm as lgb # For LightGBM model
import numpy as np # For numerical operations
import os # For running a command in the terminal
import pandas as pd # For data manipulation and analysis
import platform # For detecting the operating system
import shap # For SHAP value explanations
import time # For measuring time taken by operations
from colorama import Style # For terminal text styling
from lime.lime_tabular import LimeTabularExplainer # For LIME explanations
from sklearn.ensemble import RandomForestClassifier # For Random Forest model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier # For Gradient Boosting model
from sklearn.linear_model import LogisticRegression # For logistic regression model
from sklearn.metrics import classification_report, confusion_matrix # For evaluating model performance
from sklearn.model_selection import train_test_split # For splitting the dataset into training and testing sets
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid # For k-nearest neighbors model
from sklearn.neural_network import MLPClassifier # For neural network model
from sklearn.preprocessing import LabelEncoder, StandardScaler # For preprocessing data
from sklearn.svm import SVC # For Support Vector Machine model
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

# Execution Constants
OUTPUT_DIR = f"Results" # Directory to save results
VERBOSE = False # Set to True for verbose output
DATASETS = { # Dictionary containing dataset directory paths with their training and testing file paths
	"./CIC-IDS2017-Dataset": [
		"./CIC-IDS2017-Dataset/Infiltration-Thursday-no-metadata.parquet",
		"./CIC-IDS2017-Dataset/Infiltration-Thursday-no-metadata.parquet"
	],
	"./Cybersecurity-Intrusion-Detection-Dataset": [
		"./Cybersecurity-Intrusion-Detection-Dataset/cybersecurity_intrusion_data.arff",
		"./Cybersecurity-Intrusion-Detection-Dataset/cybersecurity_intrusion_data.arff"
	],
	"./KDD-Dataset": [
		"./KDD-Dataset/KDDTrain+.arff",
		"./KDD-Dataset/KDDTest+.arff"
	],
	"./LUFlow-Dataset": [
		"./LUFlow-Dataset/2022/06/2022.06.14/2022.06.14.csv",
		"./LUFlow-Dataset/2022/06/2022.06.14/2022.06.14.csv"
	],
	"./UNSW-NB15-Dataset": [
		"./UNSW-NB15-Dataset/UNSW_NB15_training-set.parquet",
		"./UNSW-NB15-Dataset/UNSW_NB15_testing-set.parquet"
	]
}

# Constants
SOUND_COMMANDS = {"Darwin": "afplay", "Linux": "aplay", "Windows": "start"} # Commands to play sound on different platforms
SOUND_FILE = "./.assets/Sounds/NotificationSound.wav" # Path to the sound file
RUN_FUNCTIONS = { # Dictionary containing information about the functions to run/not run
	"Play Sound": True
}

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

def detect_label_column(columns, common_names=None):
	"""
	Detects the label column from a list of common label names.

	:param columns: List of DataFrame column names
	:param common_names: List of common label column names (optional)
	:return: The detected label column name or None if not found
	"""

	verbose_output(f"{BackgroundColors.GREEN}Detecting label column from the provided columns: {BackgroundColors.CYAN}{columns}{Style.RESET_ALL}") # Output the verbose message

	if common_names is None: # If common_names is not provided, use a default list
		common_names = ["class", "label", "target", "outcome", "result", "attack_detected"]

	columns_lower = [col.lower() for col in columns] # Convert all column names to lowercase for case-insensitive comparison

	for name in common_names: # Iterate through each common name
		if name.lower() in columns_lower: # If the common name is found in the columns
			index = columns_lower.index(name.lower()) # Get the index of the common name in the columns
			return columns[index] # Return the original column name (not lowercase) at that index

	return None # If no common name is found, return None

def load_arff_file_safely(path):
	"""
	Loads an ARFF file with preprocessing to sanitize nominal attribute definitions by removing extra spaces inside curly braces (e.g., { 'A', 'B' } → {'A','B'}).

	:param path: Path to the ARFF file
	:return: Dictionary parsed from ARFF content
	"""

	verbose_output(f"{BackgroundColors.GREEN}Loading ARFF file: {BackgroundColors.CYAN}{path}{Style.RESET_ALL}") # Output the verbose message

	with open(path, "r") as f: # Open the ARFF file in read mode
		lines = f.readlines() # Read all lines from the ARFF file

	cleaned_lines = [] # List to store the cleaned lines
	for line in lines: # Iterate through each line in the ARFF file
		if "@attribute" in line and "{" in line and "}" in line: # If the line contains an attribute definition with braces
			before_brace, brace_content = line.split("{", 1) # Split the line into parts before the first brace
			values, after_brace = brace_content.split("}", 1) # Split the line into parts before and after the braces
			values = ",".join([v.strip() for v in values.split(",")]) # Remove spaces around values
			line = f"{before_brace}{{{values}}}{after_brace}" # Reconstruct the line with cleaned values

		cleaned_lines.append(line) # Append the cleaned line to the list

	with open(path, "w") as f: # Open the ARFF file in write mode
		f.writelines(cleaned_lines) # Write the cleaned lines back to the ARFF file

	return liac_arff.loads("".join(cleaned_lines)) # Parse ARFF content into dictionary

def load_file(file_path):
	"""
	Loads a file based on its extension.

	:param file_path: Path to the file
	:return: DataFrame with loaded data
	"""

	verbose_output(f"{BackgroundColors.GREEN}Loading file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}") # Output the verbose message

	ext = file_path.lower().split(".")[-1] # Get file extension

	if ext == "arff": # ARFF file
		verbose_output(f"{BackgroundColors.GREEN}Loading data from ARFF file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}")
		arff_data = load_arff_file_safely(file_path) # Load ARFF data using safe loader
		df = pd.DataFrame(arff_data["data"], columns=[attr[0] for attr in arff_data["attributes"]]) # Create DataFrame with correct column names
	elif ext in ["csv", "txt"]: # CSV or TXT
		verbose_output(f"{BackgroundColors.GREEN}Loading data from CSV/TXT file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}")
		df = pd.read_csv(file_path) # Load CSV or TXT file into DataFrame
	elif ext == "parquet": # Parquet file
		verbose_output(f"{BackgroundColors.GREEN}Loading data from Parquet file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}")
		df = pd.read_parquet(file_path) # Load Parquet file into DataFrame
	else:
		raise ValueError(f"{BackgroundColors.RED}Unsupported file extension: {ext}{Style.RESET_ALL}")

	return df # Return the loaded DataFrame

def load_and_prepare_data(training_data_path=None, testing_data_path=None):
	"""
	Loads and prepares the training and testing data from the input files.
	Supports Parquet, ARFF, CSV, and TXT formats.

	:param training_data_path: Path to the training file
	:param testing_data_path: Path to the testing file
	:return: Tuple (train_df, test_df, split_required)
	"""

	verbose_output(f"{BackgroundColors.GREEN}Loading and preparing data...{Style.RESET_ALL}") # Output the verbose message

	if training_data_path is None or testing_data_path is None: # If either path is missing
		raise ValueError(f"{BackgroundColors.RED}Both training_data_path and testing_data_path must be provided.{Style.RESET_ALL}")

	split_required = os.path.abspath(training_data_path) == os.path.abspath(testing_data_path) # Normalize to absolute paths and determine if the same file is used for both training and testing

	if split_required: # If the same file is used for both training and testing
		verbose_output(f"{BackgroundColors.YELLOW}The same file was provided for training and testing: {BackgroundColors.CYAN}{training_data_path}{BackgroundColors.YELLOW}. Performing automatic split into training and testing sets.{Style.RESET_ALL}")

	train_df = load_file(training_data_path) # Load training file
	test_df = None if split_required else load_file(testing_data_path) # Load testing file only if different

	return train_df, test_df, split_required # Return dataframes and split flag

def preprocess_features(df, label_col=None, ref_columns=None, scaler=None, label_encoder=None):
	"""
	Applies one-hot encoding and scaling to features.
	One-hot encoding is a process of converting categorical variables into a format that can be provided to machine learning algorithms to do a better job in prediction. It creates binary columns for each category in the categorical variable, where each column represents the presence or absence of that category.
	Automatically detects the label column if not provided.
	Can align columns to a reference set and reuse scaler and label encoder.

	:param df: DataFrame with features and labels
	:param label_col: Name of the label column (optional)
	:param ref_columns: Reference columns for aligning one-hot encoded features (optional)
	:param scaler: Fitted StandardScaler to reuse (optional)
	:param label_encoder: Fitted LabelEncoder to reuse (optional)
	:return: Tuple (X_scaled, y, feature_names, X_encoded, scaler, label_encoder)
	"""

	verbose_output(f"{BackgroundColors.GREEN}Using label column: {label_col}{Style.RESET_ALL}")

	if label_col is None: # If label_col is not provided, automatically detect it
		label_col = detect_label_column(df.columns) # Detect the label column from the DataFrame columns
		if label_col is None: # If no label column is detected
			raise ValueError(f"{BackgroundColors.RED}No label column detected in the DataFrame. Please provide a valid label column name.{Style.RESET_ALL}")

	df = df.dropna(subset=[label_col]) # Remove rows with NaN in the label column
	y = df[label_col] # Extract label column (keep original)
	X = df.drop(label_col, axis=1) # Extract features only

	X_encoded = pd.get_dummies(X) # Apply one-hot encoding only to features

	X_encoded = X_encoded.dropna() # Drop rows with NaNs in features
	y = y.loc[X_encoded.index] # Align y with the cleaned X_encoded

	if ref_columns is not None: # Align one-hot encoded columns to reference if given
		X_encoded = X_encoded.reindex(columns=ref_columns, fill_value=0)

	if scaler is None: # Initialize scaler if not provided
		scaler = StandardScaler() # Create a new StandardScaler instance
		X_scaled_array = scaler.fit_transform(X_encoded) # Fit and transform the features using the scaler
	else:
		X_scaled_array = scaler.transform(X_encoded) # Transform using existing scaler

	X_scaled = pd.DataFrame(X_scaled_array, columns=X_encoded.columns, index=X_encoded.index) # Convert scaled array back to DataFrame with column names and original indices

	if not pd.api.types.is_numeric_dtype(y): # Verify if label is not already numeric
		if label_encoder is None: # Initialize label encoder if not provided
			label_encoder = LabelEncoder() # Create a new LabelEncoder instance
			y = label_encoder.fit_transform(y) # Transform the labels to numeric format
		else: # If label encoder is provided, use it to transform labels
			y = label_encoder.transform(y) # Use existing label encoder to transform labels

	return X_scaled, y, X_encoded.columns, X_encoded, scaler, label_encoder # Return scaled features, labels, feature names, one-hot DataFrame, scaler, and label encoder

def split_data(train_df, test_df, split_required, label_col=None):
	"""
	Handles splitting of data if needed, or uses provided train/test datasets.

	:param train_df: DataFrame for training data
	:param test_df: DataFrame for testing data (can be None if split_required is True)
	:param split_required: Boolean indicating if train/test split is required
	:param label_col: Label column name (optional)
	:return: Tuple (X_train, X_test, y_train, y_test, feature_names)
	"""

	verbose_output(f"{BackgroundColors.GREEN}Preparing data for training and testing...{Style.RESET_ALL}")

	if split_required: # Same file for train and test -> split internally after preprocessing
		X_scaled, y, feature_names, _, scaler, label_encoder = preprocess_features(train_df, label_col) # Preprocess features and labels
		verbose_output(f"{BackgroundColors.GREEN}Splitting dataset into train/test sets...{Style.RESET_ALL}") 
		X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)
	else: # Different files for train and test -> preprocess separately
		X_train_scaled, y_train, feature_names_train, _, scaler, label_encoder = preprocess_features(train_df, label_col) # Preprocess training features and labels
		X_test_scaled, y_test, feature_names_test, _, _, _ = preprocess_features(test_df, label_col, ref_columns=feature_names_train, scaler=scaler, label_encoder=label_encoder) # For test set, reuse scaler and label encoder, and align columns to train's feature names

		if list(feature_names_train) != list(feature_names_test): # Verify if feature names in train and test sets match
			raise ValueError(f"{BackgroundColors.RED}Mismatch in feature columns between training and testing datasets.{Style.RESET_ALL}")

		X_train, X_test = X_train_scaled, X_test_scaled # Assign preprocessed features to train and test sets
		feature_names = feature_names_train # Use training feature names as reference

	return X_train, X_test, y_train, y_test, feature_names # Return the training and testing features and labels, along with feature names

def main():
	"""
	Main function to run the machine learning pipeline on multiple datasets.

	:param: None
	:return: None
	"""

	pass

if __name__ == "__main__":
	"""
	This is the standard boilerplate that calls the main() function.

	:return: None
	"""

	main() # Call the main function
