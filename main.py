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
