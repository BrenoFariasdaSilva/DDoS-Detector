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
