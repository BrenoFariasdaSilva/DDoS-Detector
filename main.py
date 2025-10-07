#@TODO: Add cross validation
#@TODO: Add threads for each algorithm training?
#@TODO: Search and review the code for overfitting issues
#@TODO: Correlation-Based Feature Selection + Lasso Regression
# 1. Remove features with low variance (nearly constant)
# 2. Analyze the correlation with the target variable
# 3. Remove features that are highly correlated with each other (multicollinearity)
# 4. Apply Lasso regression (with L1 regularization)
#@TODO: Implement the feature selection function
#@TODO: Add tests for the feature selection function

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

def get_models():
	"""
	Initializes and returns a dictionary of models to train.

	:param: None
	:return: Dictionary of model name and instance
	"""

	verbose_output(f"{BackgroundColors.GREEN}Initializing models for training...{Style.RESET_ALL}") # Output the verbose message

	return { # Dictionary of models to train
		"Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
		"SVM": SVC(kernel="rbf", probability=True, random_state=42),
		"XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=42),
		"Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
		"KNN": KNeighborsClassifier(n_neighbors=5),
		"Nearest Centroid": NearestCentroid(),
		"Gradient Boosting": GradientBoostingClassifier(random_state=42),
		"LightGBM": lgb.LGBMClassifier(force_row_wise=True, min_gain_to_split=0.01, random_state=42, verbosity=-1),
		"MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
		# Few-Shot Learning
		# Contrastive Learning
	}

def format_duration(seconds):
	"""
	Formats a duration in seconds into a human-readable string.

	:param seconds: Duration in seconds (float or int)
	:return: Formatted string with the duration in appropriate units (seconds, minutes, or hours)
	"""

	if seconds < 60: # Less than one minute
		return f"{seconds:.2f}s"
	elif seconds < 3600: # Less than one hour
		minutes = int(seconds // 60)
		remaining_seconds = seconds % 60
		return f"{minutes}m {int(remaining_seconds)}s"
	else: # One hour or more
		hours = int(seconds // 3600)
		remaining_minutes = int((seconds % 3600) // 60)
		remaining_seconds = seconds % 60
		return f"{hours}h {remaining_minutes}m {int(remaining_seconds)}s"

def train_model(model, X_train, y_train, index, model_name, dataset_name):
	"""
	Trains a single model with timing and verbose output.

	:param model: Model instance to train
	:param X_train: Training features
	:param y_train: Training labels
	:param index: Model index for display
	:param model_name: Name of the model
	:param dataset_name: Name of the dataset
	:return: Trained model and training duration
	"""

	start_time = time.time() # Start timer for training duration in seconds
	model.fit(X_train, y_train) # Train the model
	duration = [time.time() - start_time, format_duration(time.time() - start_time)] # List to store the duration of training in seconds and in string

	print(f"{BackgroundColors.GREEN}	- Training {BackgroundColors.CYAN}{index:02d} - {model_name}{BackgroundColors.GREEN} for the dataset {BackgroundColors.CYAN}{dataset_name}{BackgroundColors.GREEN} done in {BackgroundColors.CYAN}{duration[1]}{Style.RESET_ALL}")

	return model, duration # Return trained model and duration

def build_extended_metrics(conf_matrix, labels, duration_str):
	"""
	Builds extended metrics for each class.

	:param conf_matrix: Confusion matrix
	:param labels: List of class labels
	:param duration_str: String of the duration of the model training
	:return: DataFrame with extended metrics per class
	"""

	verbose_output(f"{BackgroundColors.GREEN}Building extended metrics from confusion matrix...{Style.RESET_ALL}") # Output the verbose message

	metrics_list = [] # List to store metrics for each class

	for i, label in enumerate(labels): # Iterate through each label
		TP = conf_matrix[i, i] # True Positives for the class
		FN = np.sum(conf_matrix[i, :]) - TP # False Negatives for the class
		FP = np.sum(conf_matrix[:, i]) - TP # False Positives for the class
		TN = np.sum(conf_matrix) - (TP + FP + FN) # True Negatives for the class

		support = TP + FN # Support for the class (number of true instances)
		accuracy = round((TP + TN) / np.sum(conf_matrix), 2) if np.sum(conf_matrix) > 0 else 0 # Accuracy for the class
		precision = round(TP / (TP + FP), 2) if (TP + FP) > 0 else 0 # Precision for the class
		recall = round(TP / (TP + FN), 2) if (TP + FN) > 0 else 0 # Recall for the class
		f1 = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) > 0 else 0 # F1-Score for the class

		metrics_list.append({ # Append the metrics for the class to the list
			"Class": label,
			"Training Duration": duration_str, # Format the training duration
			"Correct (TP)": TP,
			"Wrong (FN)": FN,
			"False Positives (FP)": FP,
			"True Negatives (TN)": TN,
			"Support": support,
			"Accuracy (per class)": accuracy,
			"Precision": precision,
			"Recall": recall,
			"F1-Score": f1,
		})

	return pd.DataFrame(metrics_list) # Return a DataFrame with the extended metrics for each class

def evaluate_model(model, X_test, y_test, duration_str):
	"""
	Evaluates the model on the test data.

	:param model: Trained model
	:param X_test: Test features
	:param y_test: Test labels
	:param duration_str: String of the duration of the model training
	:return: Tuple containing:
				- Classification report dictionary
				- Extended metrics DataFrame
	"""

	verbose_output(f"{BackgroundColors.GREEN}Evaluating model: {model.__class__.__name__}...{Style.RESET_ALL}") # Output the verbose message

	preds = model.predict(X_test) # Make predictions on the test data

	report = classification_report(y_test, preds, output_dict=True, zero_division=0) # Generate classification report as dictionary

	conf_matrix = confusion_matrix(y_test, preds) # Generate confusion matrix

	metrics_df = build_extended_metrics(conf_matrix, model.classes_, duration_str) # Build extended metrics DataFrame from confusion matrix

	return report, metrics_df # Return the report and extended metrics

def save_results(report, metrics_df, results_dir, index, model_name):
	"""
	Saves the classification report and extended metrics to disk.

	:param report: Classification report dictionary
	:param metrics_df: DataFrame with extended metrics
	:param results_dir: Directory of the results of the dataset
	:param index: Index of the model
	:param model_name: Name of the model
	"""

	verbose_output(f"{BackgroundColors.GREEN}Saving results for {model_name}...{Style.RESET_ALL}")

	if not os.path.exists(results_dir): # If the results directory does not exist
		os.makedirs(results_dir) # Create the results directory

	filename_base = f"{results_dir}/{index:02d}-{model_name.replace(' ', '_').replace('(', '').replace(')', '')}" # Base filename for saving results

	pd.DataFrame(report).transpose().to_csv(f"{filename_base}-Classification_report.csv", float_format="%.2f", index_label="Class") # Save classification report
	metrics_df.to_csv(f"{filename_base}-Extended_metrics.csv", index=False, float_format="%.2f") # Save extended confusion matrix

def extract_average_metrics(metrics_df, dataset_name, model_name):
	"""
	Extracts the average row from the extended metrics dataframe.

	:param metrics_df: DataFrame with extended metrics
	:param dataset_name: Name of the dataset
	:param model_name: Name of the model
	:return: Dictionary with average metrics for summary
	"""

	verbose_output(f"{BackgroundColors.GREEN}Extracting average metrics for {model_name} on the {dataset_name} dataset...{Style.RESET_ALL}") # Output the verbose message

	avg_row = metrics_df.iloc[-1] # Get the last row which contains the average metrics

	return { # Return a dictionary with the average metrics
		"Dataset": dataset_name,
		"Training Duration": avg_row["Training Duration"],
		"Model": model_name,
		"Correct (TP)": int(avg_row["Correct (TP)"]),
		"Wrong (FN)": int(avg_row["Wrong (FN)"]),
		"False Positives (FP)": int(avg_row["False Positives (FP)"]),
		"True Negatives (TN)": int(avg_row["True Negatives (TN)"]),
		"Support": int(avg_row["Support"]),
		"Accuracy (per class)": round(float(avg_row["Accuracy (per class)"]), 2),
		"Precision": round(float(avg_row["Precision"]), 2),
		"Recall": round(float(avg_row["Recall"]), 2),
		"F1-Score": round(float(avg_row["F1-Score"]), 2)
	}

def get_model_results_file_header():
	"""
	Returns the header for the model results CSV file.

	:return: List of column names for the model results CSV
	"""

	verbose_output(true_string=f"{BackgroundColors.GREEN}Getting header for model results CSV file...{Style.RESET_ALL}") # Verbose output indicating the header retrieval

	return [ # List of column names for the model results CSV
		"Dataset", # Name of the dataset
		"Model", # Name of the ML model
		"Training Duration", # Duration of training the model
		"Correct (TP)", # True Positives
		"Wrong (FN)", # False Negatives
		"False Positives (FP)", # False Positives
		"True Negatives (TN)", # True Negatives
		"Support", # Total number of samples for the class
		"Accuracy (per class)", # Accuracy for the class (TP + TN) / Total
		"Precision", # Precision score
		"Recall", # Recall score
		"F1-Score" # F1-score
	]

def generate_overall_performance_summary(all_model_scores, output_path="."):
	"""
	Generates an overall performance summary CSV combining all datasets and models with detailed metrics.

	:param all_model_scores: List of dictionaries with model scores.
	:param output_path: Path where the summary CSV will be saved.
	:return: None
	"""

	verbose_output(true_string=f"{BackgroundColors.GREEN}Generating overall performance summary...{Style.RESET_ALL}") # Print start message

	columns = get_model_results_file_header() # Get the header for the model results CSV file

	formatted_scores = [] # Initialize list to store reformatted model score dictionaries

	for entry in all_model_scores: # Iterate through each model score entry
		dataset_name = entry.get("Dataset", "Unknown") # Use the dataset from entry or fallback to "Unknown"
		model_name = entry.get("Model", "") # Get the full model name from the entry
		if "-" in model_name: # If the model name has a prefix like "03-XGBoost"
			model_name = model_name.split("-", 1)[-1].replace("_", " ").strip() # Remove numeric prefix and clean name

		formatted_scores.append({ # Create a dictionary aligned with the defined column structure
			"Dataset": dataset_name, # Dataset name
			"Model": model_name, # Cleaned model name
			"Training Duration": entry.get("Training Duration", 0), # Training duration formatted
			"Correct (TP)": entry.get("Correct (TP)", ""), # True Positives
			"Wrong (FN)": entry.get("Wrong (FN)", ""), # False Negatives
			"False Positives (FP)": entry.get("False Positives (FP)", ""), # False Positives
			"True Negatives (TN)": entry.get("True Negatives (TN)", ""), # True Negatives
			"Support": entry.get("Support", ""), # Support (samples)
			"Accuracy (per class)": entry.get("Accuracy (per class)", ""), # Accuracy
			"Precision": entry.get("Precision", ""), # Precision
			"Recall": entry.get("Recall", ""), # Recall
			"F1-Score": entry.get("F1-Score", "") # F1-Score
		})

	formatted_scores = sorted(formatted_scores, key=lambda x: (x["Dataset"], -float(x["F1-Score"]))) # Sort first by Dataset name (alphabetically), then by F1-Score (descending within each dataset)

	output_df = pd.DataFrame(formatted_scores, columns=columns) # Create a DataFrame using the sorted scores and defined column order
	os.makedirs(os.path.join(output_path, OUTPUT_DIR)) if not os.path.exists(os.path.join(output_path, OUTPUT_DIR)) else None # Ensure the output directory exists
	output_file = os.path.join(output_path, f"{OUTPUT_DIR}/Overall_Performance.csv") # Define the output file path
	output_df.to_csv(output_file, index=False) # Save the DataFrame to CSV without including the index

	verbose_output(true_string=f"{BackgroundColors.GREEN}Overall performance summary saved to: {BackgroundColors.CYAN}{output_file}{Style.RESET_ALL}") # Print success message with file path

def train_and_evaluate_models(X_train, X_test, y_train, y_test, dataset_dir, dataset_name):
	"""
	Trains and evaluates multiple models.

	:param X_train: Training features
	:param X_test: Testing features
	:param y_train: Training labels
	:param y_test: Testing labels
	:param dataset_dir: Directory of the dataset for saving files
	:param dataset_name: Name of the dataset for saving files
	:return: Tuple containing:
				- Dictionary of trained models
				- List of dicts with extracted model metrics for summary (in memory)
	"""

	verbose_output(f"{BackgroundColors.GREEN}Training and evaluating models for the {BackgroundColors.CYAN}{dataset_name}{BackgroundColors.GREEN} dataset...{Style.RESET_ALL}")

	models = get_models() # Dictionary of models to train
	model_metrics_list = [] # List to store metrics dicts for each model, for later summary
	results_dir = os.path.join(dataset_dir, "Results") # Directory to save results for the current dataset

	for index, (name, model) in enumerate(models.items(), start=1): # Iterate through each model with an index starting from 1
		model, duration = train_model(model, X_train, y_train, index, name, dataset_name) # Train the model and get duration
		report, metrics_df = evaluate_model(model, X_test, y_test, duration[1]) # Evaluate model using reusable function
		save_results(report, metrics_df, results_dir, index, name) # Save reports and metrics

		avg_row = metrics_df.iloc[-1] # Get the last row (average metrics)

		metrics_dict = { # Compose in-memory summary dictionary
			"Dataset": dataset_name,
			"Model": name,
			"Training Duration": duration[1],
			"Correct (TP)": int(avg_row["Correct (TP)"]),
			"Wrong (FN)": int(avg_row["Wrong (FN)"]),
			"False Positives (FP)": int(avg_row["False Positives (FP)"]),
			"True Negatives (TN)": int(avg_row["True Negatives (TN)"]),
			"Support": int(avg_row["Support"]),
			"Accuracy (per class)": round(float(avg_row["Accuracy (per class)"]), 2),
			"Precision": round(float(avg_row["Precision"]), 2),
			"Recall": round(float(avg_row["Recall"]), 2),
			"F1-Score": round(float(avg_row["F1-Score"]), 2)
		}

		model_metrics_list.append(metrics_dict) # Add to summary list

	generate_overall_performance_summary(model_metrics_list, output_path=results_dir) # Generate overall performance summary for all models for the current dataset

	return models, model_metrics_list # Return trained models and metrics

def explain_predictions_with_tree_shap(model, X_train, X_test, feature_names, model_name="TreeModel"):
	"""
	Explains predictions using SHAP's TreeExplainer.
	:param model: Trained model
	:param X_train: Training features
	:param X_test: Testing features
	:param feature_names: Names of the features
	:param model_name: Name of the model for saving files
	:return: None
	"""

	print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Explaining Predictions with TreeExplainer...{Style.RESET_ALL}")
	X_explain = X_test[:5] # Select the first 5 instances for explanation

	explainer = shap.TreeExplainer(model) # Create a SHAP TreeExplainer for the model
	shap_values = explainer.shap_values(X_explain) # Calculate SHAP values for the selected instances

	for i in range(len(X_explain)): # Iterate through each instance
		shap_val = shap_values[i] # Get SHAP values for the instance
		feat_val = X_explain.iloc[i].values.flatten() # Get feature values for the instance

		if len(feature_names) != len(shap_val) or len(shap_val) != len(feat_val): # Verify if lengths match
			print(f"[Erro] Comprimentos incompatíveis na instância {i+1}") 
			continue # Skip this instance if lengths do not match

		shap_df = pd.DataFrame({ # Create a DataFrame for SHAP values
			"feature": feature_names,
			"shap_value": shap_val.flatten(),
			"feature_value": feat_val
		})
		shap_df.to_csv(f"{model_name}_tree_shap_instance_{i+1}.csv", index=False, float_format="%.2f") # Save SHAP values to CSV

def explain_predictions_with_shap(model, X_train, X_test, feature_names):
	"""
	Explains model predictions using SHAP values.
	:param model: Trained model
	:param X_train: Training features
	:param X_test: Testing features
	:param feature_names: Names of the features
	:return: None
	"""

	verbose_output(f"{BackgroundColors.GREEN}Explaining predictions with SHAP...{Style.RESET_ALL}")

	X_explain = X_test[:5] # Select the first 5 instances for explanation
	print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Explaining Predictions with SHAP...{Style.RESET_ALL}")

	explainer = shap.Explainer(model, X_train, feature_names=feature_names) # Create a SHAP explainer for the model
	shap_values = explainer(X_explain) # Calculate SHAP values for the selected instances

	for i in range(len(X_explain)): # Iterate through each instance
		shap_val = shap_values[i].values # Get SHAP values for the instance
		feat_val = shap_values[i].data # Get feature values for the instance

		shap_val = shap_val.flatten() # Flatten the SHAP values
		feat_val = feat_val.flatten() # Flatten the feature values

		if len(feature_names) != len(shap_val) or len(shap_val) != len(feat_val): # Verify if lengths match
			print(f"[Erro] Comprimentos incompatíveis na instância {i+1}:")
			print(f" - feature_names: {len(feature_names)}")
			print(f" - shap_value: {len(shap_val)}")
			print(f" - feature_value: {len(feat_val)}")
			continue # Skip this instance if lengths do not match

		shap_df = pd.DataFrame({ # Create a DataFrame for SHAP values
			"feature": feature_names,
			"shap_value": shap_val,
			"feature_value": feat_val
		})

		shap_df.to_csv(f"shap_values_instance_{i+1}.csv", index=False) # Save SHAP values to CSV

def explain_predictions_with_lime(model, X_train, X_test, feature_names, model_name="Model"):
	"""
	Explains model predictions using LIME.
	:param model: Trained model
	:param X_train: Training features
	:param X_test: Testing features
	:param feature_names: Names of the features
	:param model_name: Name of the model for saving files
	:return: None
	"""

	print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Explaining Predictions with LIME...{Style.RESET_ALL}")
	X_explain = X_test[:5] # Select the first 5 instances for explanation

	explainer = LimeTabularExplainer( # Create a LIME explainer for tabular data
		training_data=X_train.values, # Training data for the explainer
		feature_names=feature_names, # Names of the features
		class_names=[str(c) for c in model.classes_] if hasattr(model, "classes_") else ["Class 0", "Class 1"], # Class names for the model
		mode="classification" # Mode of the explainer (classification or regression)
	)

	for i in range(len(X_explain)): # Iterate through each instance
		exp = explainer.explain_instance( # Explain the instance using LIME
			data_row=X_explain.iloc[i].values, # Data row to explain
			predict_fn=model.predict_proba, # Prediction function for the model
			num_features=len(feature_names) # Number of features to include in the explanation
		)
		lime_df = pd.DataFrame(exp.as_list(), columns=["feature", "weight"]) # Create a DataFrame for LIME explanation
		lime_df.to_csv(f"{model_name}_lime_instance_{i+1}.csv", index=False, float_format="%.2f") # Save LIME explanation to CSV

def explain_with_multiple_methods(model, X_train, X_test, feature_names, model_name="Model"):
	"""
	Explains model predictions using multiple methods: SHAP, TreeExplainer, and LIME.
	:param model: Trained model
	:param X_train: Training features
	:param X_test: Testing features
	:param feature_names: Names of the features
	:param model_name: Name of the model for saving files
	:return: None
	"""

	print(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Iniciando explicações para {model_name}...{Style.RESET_ALL}")

	if "xgboost" in str(type(model)).lower() or "randomforest" in str(type(model)).lower(): # If the model is XGBoost or Random Forest
		explain_predictions_with_tree_shap(model, X_train, X_test, feature_names, model_name=model_name) # Explain predictions using SHAP's TreeExplainer
	else: # For other models
		explain_predictions_with_shap(model, X_train, X_test, feature_names) # Explain predictions using SHAP values

	explain_predictions_with_lime(model, X_train, X_test, feature_names, model_name=model_name) # Explain predictions using LIME, as it works with any model

def play_sound():
	"""
	Plays a sound when the program finishes.

	:return: None
	"""

	if verify_filepath_exists(SOUND_FILE): # If the sound file exists
		if platform.system() in SOUND_COMMANDS: # If the platform.system() is in the SOUND_COMMANDS dictionary
			os.system(f"{SOUND_COMMANDS[platform.system()]} {SOUND_FILE}") # Play the sound
		else: # If the platform.system() is not in the SOUND_COMMANDS dictionary
			print(f"{BackgroundColors.RED}The {BackgroundColors.CYAN}platform.system(){BackgroundColors.RED} is not in the {BackgroundColors.CYAN}SOUND_COMMANDS dictionary{BackgroundColors.RED}. Please add it!{Style.RESET_ALL}")
	else: # If the sound file does not exist
		print(f"{BackgroundColors.RED}Sound file {BackgroundColors.CYAN}{SOUND_FILE}{BackgroundColors.RED} not found. Make sure the file exists.{Style.RESET_ALL}")

def main():
	"""
	Main function to run the machine learning pipeline on multiple datasets.

	:param: None
	:return: None
	"""

	print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Starting Machine Learning Pipeline...{Style.RESET_ALL}\n") # Print the start message and clear the terminal

	sorted_datasets = sorted(DATASETS.items()) # Sort datasets alphabetically by keys

	all_model_scores = [] # List to store all models' performance metrics across all datasets

	for index, (dataset_key, (training_file_path, testing_file_path)) in enumerate(sorted_datasets, start=1): # Enumerate through sorted datasets with index starting from 1
		dataset_name = os.path.basename(dataset_key) # Get the dataset name from the directory path

		print(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing dataset {BackgroundColors.CYAN}{index}/{len(sorted_datasets)}{BackgroundColors.GREEN}: {BackgroundColors.CYAN}{dataset_name}{BackgroundColors.GREEN}{Style.RESET_ALL}")

		if not verify_filepath_exists(training_file_path) or not verify_filepath_exists(testing_file_path): # If either training or testing file does not exist
			print(f"{BackgroundColors.RED}Missing input files for {dataset_name}. Skipping.{Style.RESET_ALL}")
			continue # Skip to the next dataset if files are missing

		train_df, test_df, split_required = load_and_prepare_data(training_file_path, testing_file_path) # Load and prepare the training and testing data
		X_train, X_test, y_train, y_test, feature_names = split_data(train_df, test_df, split_required) # Split the data into training and testing sets, and preprocess features

		models, dataset_model_scores = train_and_evaluate_models(X_train, X_test, y_train, y_test, dataset_key, dataset_name) # Train and evaluate models on the dataset, returning trained models and their performance metrics

		# for model_name, model in models.items(): # Iterate through each trained model
			# print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Explaining predictions for {model_name} on {dataset_name}...{Style.RESET_ALL}")
			# explain_with_multiple_methods(model, X_train, X_test, feature_names, model_name=model_name) # Explain model predictions using multiple methods

		all_model_scores.extend(dataset_model_scores) if dataset_model_scores else None # Extend the list of all model scores with the current dataset's scores if available

		print(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Pipeline for {BackgroundColors.CYAN}{dataset_name}{BackgroundColors.GREEN} finished successfully.{Style.RESET_ALL}\n")

	generate_overall_performance_summary(all_model_scores) if all_model_scores else None # Generate overall performance summary if there are any model scores

	print(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}All datasets processed. Overall analysis finished.{Style.RESET_ALL}")

	atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None # Register the play_sound function to be called at exit if enabled

if __name__ == "__main__":
	"""
	This is the standard boilerplate that calls the main() function.

	:return: None
	"""

	main() # Call the main function
