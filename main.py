"""
================================================================================
Modular DDoS Detection Evaluation Framework (main.py)
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-10-06
Description :
    This module implements a modular evaluation framework for DDoS
    detection datasets. It centralizes dataset loading, preprocessing,
    training and evaluation of multiple classifiers, and exports detailed
    reports for reproducible experiments. The pipeline supports ARFF, CSV,
    TXT and Parquet inputs and integrates explainability via SHAP and LIME.

    Key features and behavior:
        - Automatic detection of label column and common non-feature fields
        - Optional feature selection support (Genetic Algorithm / RFE / PCA)
        - Stratified train/test splits or k-fold cross-validation
        - Per-dataset Results directory with classification and extended metrics
        - SHAP / LIME explanation exports for selected models
        - Robust ARFF loading with nominal value sanitization

    Important configuration constants in this file:
        - `SAMPLE_SIZE` (None|int): limit rows for quicker tests (None = full)
        - `OUTPUT_DIR`: base directory for aggregated outputs
        - `VERBOSE`: enable additional stdout diagnostics
        - `RUN_FUNCTIONS` controls optional behaviors (e.g., play sound)
        - Downsampling and t-SNE helpers enforce a per-class minimum of
            samples when requested (default min ≈ 50) to preserve minority classes

Usage:
    Configure the `DATASETS` mapping with dataset file paths and optional
    feature files, then run via the provided Makefile target or directly:

        python3 main.py
        or
        make main

Outputs:
    - Per-model CSV reports: `NN-ModelName-Classification_report.csv`
    - Per-model extended metrics: `NN-ModelName-Extended_metrics.csv`
    - Aggregated `Overall_Performance*.csv` under the dataset Results folder

Notes:
    - The script is designed for reproducible experiments; results are saved
        per-dataset under a `Results` subfolder.
    - The code includes compatibility workarounds for different versions of
        common libraries (e.g., sklearn t-SNE parameter name differences).
    - For debugging t-SNE and downsampling behaviour, set `VERBOSE = True`.

TODOs:
    - Add CLI flags for `SAMPLE_SIZE`, `min_class_size`, and toggles such as
        `CROSS_DATASET_VALIDATE` to make experiments configurable from the CLI.
    - Add automated unit tests for preprocessing and feature-selection flows.
    - Optionally parallelize model training and add resumable checkpoints.

Dependencies:
    - Python >= 3.9
    - pandas, numpy, scikit-learn, lightgbm, xgboost, shap, lime, colorama

Outputs (summary):
    - Model performance reports (.csv)
    - Extended confusion matrix with detailed class metrics
    - Aggregated performance summaries by dataset and algorithm
"""

import arff as liac_arff  # For loading ARFF files
import atexit  # For registering a function to run at exit
import dataframe_image as dfi  # For exporting DataFrame styled tables as PNG images
import datetime  # For timestamping
import lightgbm as lgb  # For LightGBM model
import math  # For mathematical operations
import numpy as np  # For numerical operations
import os  # For running a command in the terminal
import pandas as pd  # For data manipulation and analysis
import platform  # For detecting the operating system
import re  # For regular expressions
import shap  # For SHAP value explanations
import sys  # For system-specific parameters and functions
import telegram_bot as telegram_module  # For setting Telegram prefix and device info
import time  # For measuring time taken by operations
from colorama import Style  # For terminal text styling
from lime.lime_tabular import LimeTabularExplainer  # For LIME explanations
from Logger import Logger  # For logging output to both terminal and file
from pathlib import Path  # For handling file paths
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # For Gradient Boosting model
from sklearn.linear_model import LogisticRegression  # For logistic regression model
from sklearn.metrics import classification_report, confusion_matrix  # For evaluating model performance
from sklearn.model_selection import (  # For splitting the dataset into training and testing sets
    cross_val_score,
    StratifiedKFold,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid  # For k-nearest neighbors model
from sklearn.neural_network import MLPClassifier  # For neural network model
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For preprocessing data
from sklearn.svm import SVC  # For Support Vector Machine model
from telegram_bot import TelegramBot, send_exception_via_telegram, send_telegram_message, setup_global_exception_hook  # For sending progress messages to Telegram
from tqdm import tqdm  # For progress bars
from xgboost import XGBClassifier  # For XGBoost classifier


# Macros:
class BackgroundColors:  # Colors for the terminal
    CYAN = "\033[96m"  # Cyan
    GREEN = "\033[92m"  # Green
    YELLOW = "\033[93m"  # Yellow
    RED = "\033[91m"  # Red
    BOLD = "\033[1m"  # Bold
    UNDERLINE = "\033[4m"  # Underline
    CLEAR_TERMINAL = "\033[H\033[J"  # Clear the terminal


# Execution Constants
OUTPUT_DIR = f"./Results"  # Directory to save results
VERBOSE = False  # Set to True for verbose output
SAMPLE_SIZE = None  # Set to an integer (e.g., 100000) to sample data for faster testing, None to use full dataset
N_JOBS = -1  # Number of parallel jobs for GridSearchCV (-1 uses all processors)
DATASETS = {  # Dictionary containing dataset paths and feature files
    "CICDDoS2019-Dataset": {
        "train": "./Datasets/CICDDoS2019/01-12/DrDoS_DNS.csv",
        "test": "./Datasets/CICDDoS2019/01-12/DrDoS_DNS.csv",
        "features": ["./Datasets/CICDDoS2019/01-12/Feature_Analysis/Genetic_Algorithm_Results.txt"],
    }
}

# Telegram Bot Setup:
TELEGRAM_BOT = None  # Global Telegram bot instance (initialized in setup_telegram_bot)

# Logger Setup:
logger = Logger(f"./Logs/{Path(__file__).stem}.log", clean=True)  # Create a Logger instance
sys.stdout = logger  # Redirect stdout to the logger
sys.stderr = logger  # Redirect stderr to the logger

# Constants
SOUND_COMMANDS = {
    "Darwin": "afplay",
    "Linux": "aplay",
    "Windows": "start",
}  # Commands to play sound on different platforms
SOUND_FILE = "./.assets/Sounds/NotificationSound.wav"  # Path to the sound file
RUN_FUNCTIONS = {"Play Sound": True}  # Dictionary containing information about the functions to run/not run

# Functions Definitions:


setup_global_exception_hook()  # Set up global exception hook to send exceptions via Telegram


def verbose_output(true_string="", false_string=""):
    """
    Outputs a message if the VERBOSE constant is set to True.

    :param true_string: The string to be outputted if the VERBOSE constant is set to True.
    :param false_string: The string to be outputted if the VERBOSE constant is set to False.
    :return: None
    """
    
    try:
        if VERBOSE and true_string != "":  # If VERBOSE is True and a true_string was provided
            print(true_string)  # Output the true statement string
        elif false_string != "":  # If a false_string was provided
            print(false_string)  # Output the false statement string
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def verify_dot_env_file():
    """
    Verifies if the .env file exists in the current directory.

    :return: True if the .env file exists, False otherwise
    """
    
    try:
        env_path = Path(__file__).parent / ".env"  # Path to the .env file
        if not env_path.exists():  # If the .env file does not exist
            print(f"{BackgroundColors.CYAN}.env{BackgroundColors.YELLOW} file not found at {BackgroundColors.CYAN}{env_path}{BackgroundColors.YELLOW}. Telegram messages may not be sent.{Style.RESET_ALL}")
            return False  # Return False

        return True  # Return True if the .env file exists
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def setup_telegram_bot():
    """
    Sets up the Telegram bot for progress messages.

    :return: None
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Setting up Telegram bot for messages...{Style.RESET_ALL}"
        )  # Output the verbose message

        verify_dot_env_file()  # Verify if the .env file exists

        global TELEGRAM_BOT  # Declare the module-global telegram_bot variable

        try:  # Try to initialize the Telegram bot
            TELEGRAM_BOT = TelegramBot()  # Initialize Telegram bot for progress messages
            telegram_module.TELEGRAM_DEVICE_INFO = f"{telegram_module.get_local_ip()} - {platform.system()}"
            telegram_module.RUNNING_CODE = os.path.basename(__file__)
        except Exception as e:
            print(f"{BackgroundColors.RED}Failed to initialize Telegram bot: {e}{Style.RESET_ALL}")
            TELEGRAM_BOT = None  # Set to None if initialization fails
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def verify_filepath_exists(filepath):
    """
    Verify if a file or folder exists at the specified path.

    :param filepath: Path to the file or folder
    :return: True if the file or folder exists, False otherwise
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Verifying if the file or folder exists at the path: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}"
        )  # Output the verbose message

        return os.path.exists(filepath)  # Return True if the file or folder exists, False otherwise
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_features_from_file(file_path, start_line_keyword="Best Feature Subset using"):
    """
    Extracts feature names from a feature analysis results file (Genetic Algorithm or RFE).

    :param file_path: Path to the feature results file
    :param start_line_keyword: String that indicates where to start collecting features
                                                                            e.g., "Best Feature Subset using Genetic Algorithm"
                                                                                            "Best Feature Subset using Recursive Feature Elimination"
    :return: List of feature names
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Extracting features from file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}"
        )  # Output the verbose message

        features = []  # List to store feature names
        start_collecting = False  # Flag to indicate when to start collecting features

        with open(file_path, "r") as f:  # Open the feature results file in read mode
            for line in f:  # Iterate through each line in the file
                line = line.strip()  # Remove leading/trailing whitespace
                if start_line_keyword in line:  # Start collecting after this line
                    start_collecting = True  # Set flag to start collecting features
                    continue  # Skip the line with the keyword

                if start_collecting:  # If we are in the feature collection section
                    if not line:  # If the line is empty
                        break  # Stop at first empty line after features section

                    match = re.match(
                        r"\d+\.\s+(.*?)(\s+\(RFE ranking \d+\))?$", line
                    )  # Match lines like: "1. Feature Name (RFE ranking X)"

                    if match:  # If the line matches the expected format
                        feature_name = match.group(1).strip()  # Extract the feature name
                        features.append(feature_name)  # Add the feature name to the list

        return features  # Return the list of feature names
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def detect_label_column(columns, common_names=None):
    """
    Detects the label column from a list of common label names.
    Handles columns with leading/trailing whitespace.

    :param columns: List of DataFrame column names
    :param common_names: List of common label column names (optional)
    :return: The detected label column name or None if not found
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Detecting label column from the provided columns: {BackgroundColors.CYAN}{columns}{Style.RESET_ALL}"
        )  # Output the verbose message

        if common_names is None:  # If common_names is not provided, use a default list
            common_names = ["class", "label", "target", "outcome", "result", "attack_detected"]

        columns_lower = [
            (col.strip().lower(), col) for col in columns
        ]  # Convert all column names to lowercase and strip whitespace

        for name in common_names:  # Iterate through each common name
            for col_lower, col_original in columns_lower:  # Verify each column
                if name.lower() == col_lower:  # If the common name matches the stripped column name
                    return col_original  # Return the original column name (with spaces if present)

        return None  # If no common name is found, return None
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def load_arff_file_safely(path):
    """
    Loads an ARFF file with preprocessing to sanitize nominal attribute definitions by removing extra spaces inside curly braces (e.g., { 'A', 'B' } → {'A','B'}).

    :param path: Path to the ARFF file
    :return: Dictionary parsed from ARFF content
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Loading ARFF file: {BackgroundColors.CYAN}{path}{Style.RESET_ALL}"
        )  # Output the verbose message

        with open(path, "r") as f:  # Open the ARFF file in read mode
            lines = f.readlines()  # Read all lines from the ARFF file

        cleaned_lines = []  # List to store the cleaned lines
        for line in lines:  # Iterate through each line in the ARFF file
            if (
                "@attribute" in line and "{" in line and "}" in line
            ):  # If the line contains an attribute definition with braces
                before_brace, brace_content = line.split("{", 1)  # Split the line into parts before the first brace
                values, after_brace = brace_content.split("}", 1)  # Split the line into parts before and after the braces
                values = ",".join([v.strip() for v in values.split(",")])  # Remove spaces around values
                line = f"{before_brace}{{{values}}}{after_brace}"  # Reconstruct the line with cleaned values

            cleaned_lines.append(line)  # Append the cleaned line to the list

        with open(path, "w") as f:  # Open the ARFF file in write mode
            f.writelines(cleaned_lines)  # Write the cleaned lines back to the ARFF file

        return liac_arff.loads("".join(cleaned_lines))  # Parse ARFF content into dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def load_file(file_path):
    """
    Loads a file based on its extension.

    :param file_path: Path to the file
    :return: DataFrame with loaded data
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Loading file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}"
        )  # Output the verbose message

        ext = file_path.lower().split(".")[-1]  # Get file extension

        if ext == "arff":  # ARFF file
            verbose_output(
                f"{BackgroundColors.GREEN}Loading data from ARFF file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}"
            )
            arff_data = load_arff_file_safely(file_path)  # Load ARFF data using safe loader
            df = pd.DataFrame(
                arff_data["data"], columns=[attr[0] for attr in arff_data["attributes"]]
            )  # Create DataFrame with correct column names
        elif ext in ["csv", "txt"]:  # CSV or TXT
            verbose_output(
                f"{BackgroundColors.GREEN}Loading data from CSV/TXT file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}"
            )
            df = pd.read_csv(
                file_path, low_memory=False
            )  # Load CSV or TXT file into DataFrame with low_memory=False to avoid dtype warning
            df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names
        elif ext == "parquet":  # Parquet file
            verbose_output(
                f"{BackgroundColors.GREEN}Loading data from Parquet file: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}"
            )
            df = pd.read_parquet(file_path)  # Load Parquet file into DataFrame
        else:  # Unsupported file extension
            raise ValueError(f"{BackgroundColors.RED}Unsupported file extension: {ext}{Style.RESET_ALL}")

        df.columns = df.columns.str.strip()  # Clean column names by stripping leading/trailing whitespacee

        return df  # Return the loaded DataFrame
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def load_and_prepare_data(training_data_path=None, testing_data_path=None):
    """
    Loads and prepares the training and testing data from the input files.
    Supports Parquet, ARFF, CSV, and TXT formats.

    :param training_data_path: Path to the training file
    :param testing_data_path: Path to the testing file
    :return: Tuple (train_df, test_df, split_required)
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Loading and preparing data...{Style.RESET_ALL}"
        )  # Output the verbose message

        if training_data_path is None or testing_data_path is None:  # If either path is missing
            raise ValueError(
                f"{BackgroundColors.RED}Both training_data_path and testing_data_path must be provided.{Style.RESET_ALL}"
            )

        split_required = os.path.abspath(training_data_path) == os.path.abspath(
            testing_data_path
        )  # Normalize to absolute paths and determine if the same file is used for both training and testing

        if split_required:  # If the same file is used for both training and testing
            verbose_output(
                f"{BackgroundColors.GREEN}The same file was provided for training and testing: {BackgroundColors.CYAN}{training_data_path}{BackgroundColors.GREEN}. Performing automatic split into training and testing sets.{Style.RESET_ALL}"
            )

        train_df = load_file(training_data_path)  # Load training file
        test_df = None if split_required else load_file(testing_data_path)  # Load testing file only if different

        if SAMPLE_SIZE is not None and len(train_df) > SAMPLE_SIZE:  # If SAMPLE_SIZE is set and training data exceeds it
            train_df = train_df.sample(n=SAMPLE_SIZE, random_state=42)  # Sample training data
            if test_df is not None and len(test_df) > SAMPLE_SIZE:  # If testing data exists and exceeds SAMPLE_SIZE
                test_df = test_df.sample(n=SAMPLE_SIZE, random_state=42)  # Sample testing data

        return train_df, test_df, split_required  # Return dataframes and split flag
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def preprocess_features(df, label_col=None, ref_columns=None, scaler=None, label_encoder=None, selected_features=None):
    """
    Applies one-hot encoding and scaling to features.
    Optionally allows selecting only a subset of features.

    :param df: DataFrame with raw features
    :param label_col: Name of the label column
    :param ref_columns: Reference columns for reindexing (optional)
    :param scaler: Pre-fitted scaler (optional)
    :param label_encoder: Pre-fitted label encoder (optional)
    :param selected_features: List of selected features to keep (optional)
    :return: Tuple (X_scaled, y, feature_names, X_encoded, scaler, label_encoder)
    """
    
    try:
        verbose_output(f"{BackgroundColors.GREEN}Preprocessing features and labels...{Style.RESET_ALL}")
        verbose_output(
            f"\t{BackgroundColors.GREEN}Label Column: {BackgroundColors.CYAN}{label_col}{BackgroundColors.GREEN}{Style.RESET_ALL}"
        )
        verbose_output(
            f"\t{BackgroundColors.GREEN}Dataframe Shape: {BackgroundColors.CYAN}{df.shape}{BackgroundColors.GREEN}{Style.RESET_ALL}"
        )
        verbose_output(
            f"\tSelected Features {BackgroundColors.CYAN}{len(selected_features) if selected_features else 'All'}{BackgroundColors.GREEN}{Style.RESET_ALL}"
        )

        if label_col is None:  # If label_col is not provided, attempt to detect it
            label_col = detect_label_column(df.columns)  # Detect label column
            if label_col is None:  # If no label column is detected
                raise ValueError(f"{BackgroundColors.RED}No label column detected in the DataFrame.{Style.RESET_ALL}")

        df = df.dropna(subset=[label_col])  # Drop rows where label is NaN
        y = df[label_col]  # Extract labels
        X = df.drop(label_col, axis=1)  # Extract features

        non_feature_cols = [
            "Unnamed: 0",
            "Flow ID",
            "Source IP",
            "Destination IP",
            "Timestamp",
        ]  # Common non-feature columns
        cols_to_drop = [col for col in non_feature_cols if col in X.columns]  # Identify columns to drop
        if cols_to_drop:  # If there are columns to drop
            X = X.drop(columns=cols_to_drop)  # Drop non-feature columns

        if selected_features is not None:  # If selected_features is provided
            missing_features = [f for f in selected_features if f not in X.columns]  # Identify missing features
            if missing_features:  # If there are missing features
                print(
                    f"{BackgroundColors.RED}Warning: {len(missing_features)} features from selection file not found in dataset{Style.RESET_ALL}"
                )
                print(f"{BackgroundColors.RED}Missing features: {missing_features}{Style.RESET_ALL}")
                raise ValueError(
                    f"{BackgroundColors.RED}Some selected features not found in the DataFrame!{Style.RESET_ALL}"
                )

            existing_features = [f for f in selected_features if f in X.columns]  # Keep only features that exist
            if not existing_features:  # If no selected features exist in the DataFrame
                raise ValueError(f"{BackgroundColors.RED}No selected features found in the DataFrame!{Style.RESET_ALL}")
            X = X[existing_features]  # Keep only selected features that exist

        X = X.replace([np.inf, -np.inf], np.nan)  # Replace infinity with NaN
        initial_rows = len(X)  # Store initial number of rows
        X = X.dropna()  # Drop rows with NaN values
        rows_dropped = initial_rows - len(X)  # Calculate number of rows dropped
        if rows_dropped > 0:  # If any rows were dropped
            verbose_output(
                f"{BackgroundColors.YELLOW}Warning: Dropped {rows_dropped} rows due to NaN/inf values in features.{Style.RESET_ALL}"
            )

        y = y.loc[X.index]  # Align y with cleaned X

        X_encoded = pd.get_dummies(X)  # One-hot encode categorical features
        X_encoded = X_encoded.dropna()  # Drop any rows with NaN values after encoding
        y = y.loc[X_encoded.index]  # Align y with encoded X

        if ref_columns is not None:  # If reference columns are provided
            X_encoded = X_encoded.reindex(columns=ref_columns, fill_value=0)  # Reindex to match reference columns
            verbose_output(f"{BackgroundColors.GREEN}Reindexed features to match reference columns.{Style.RESET_ALL}")

        if scaler is None:  # If scaler is not provided, fit a new one
            scaler = StandardScaler()  # Initialize StandardScaler
            X_scaled_array = scaler.fit_transform(X_encoded)  # Fit and transform the data
        else:  # If scaler is provided, use it to transform
            X_scaled_array = scaler.transform(X_encoded)  # Transform the data with existing scaler

        X_scaled = pd.DataFrame(
            X_scaled_array, columns=X_encoded.columns, index=X_encoded.index
        )  # Create DataFrame from scaled array

        if not pd.api.types.is_numeric_dtype(y):  # If labels are not numeric
            if label_encoder is None:  # If label encoder is not provided, fit a new one
                label_encoder = LabelEncoder()  # Initialize LabelEncoder
                y = label_encoder.fit_transform(y)  # Fit and transform the labels
            else:  # If label encoder is provided, use it to transform
                y = label_encoder.transform(y)  # Transform the labels with existing encoder

        return X_scaled, y, X_encoded.columns, X_encoded, scaler, label_encoder  # Return processed data and objects
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def split_data(train_df, test_df, split_required, label_col=None, selected_features=None):
    """
    Splits the data into training and testing sets.

    :param train_df: DataFrame with training data
    :param test_df: DataFrame with testing data
    :param split_required: Boolean indicating if a split is required
    :param label_col: Name of the label column
    :param selected_features: List of selected features to keep (optional)
    :return: Tuple (X_train, X_test, y_train, y_test, feature_names)
    """
    
    try:
        if split_required:  # If a split is required
            verbose_output(f"{BackgroundColors.GREEN}Splitting data into train/test sets (75/25)...{Style.RESET_ALL}")
            if label_col is None:
                label_col = detect_label_column(train_df.columns)
                if label_col is None:
                    raise ValueError(f"{BackgroundColors.RED}No label column detected for stratified split.{Style.RESET_ALL}")

            df_clean = train_df.dropna(subset=[label_col])

            train_df_part, test_df_part = train_test_split(
                df_clean, test_size=0.25, random_state=42, stratify=df_clean[label_col]
            )

            X_train_scaled, y_train, feature_names, _, scaler, label_encoder = preprocess_features(
                train_df_part, label_col, selected_features=selected_features
            )

            X_test_scaled, y_test, feature_names_test, _, _, _ = preprocess_features(
                test_df_part,
                label_col,
                ref_columns=feature_names,
                scaler=scaler,
                label_encoder=label_encoder,
                selected_features=selected_features,
            )

            if list(feature_names) != list(feature_names_test):
                raise ValueError(
                    f"{BackgroundColors.RED}Mismatch in feature columns between training and testing partitions after preprocessing.{Style.RESET_ALL}"
                )

            X_train, X_test = X_train_scaled, X_test_scaled
            verbose_output(
                f"{BackgroundColors.GREEN}Split and preprocessing complete. Train: {X_train.shape}, Test: {X_test.shape}{Style.RESET_ALL}"
            )
        else:  # If no split is required
            X_train_scaled, y_train, feature_names_train, _, scaler, label_encoder = preprocess_features(
                train_df, label_col, selected_features=selected_features
            )  # Preprocess training features
            X_test_scaled, y_test, feature_names_test, _, _, _ = preprocess_features(
                test_df,
                label_col,
                ref_columns=feature_names_train,
                scaler=scaler,
                label_encoder=label_encoder,
                selected_features=selected_features,
            )  # Preprocess testing features with reference columns and existing scaler/encoder

            if list(feature_names_train) != list(feature_names_test):  # Verify that feature columns match
                raise ValueError(
                    f"{BackgroundColors.RED}Mismatch in feature columns between training and testing datasets.{Style.RESET_ALL}"
                )

            X_train, X_test = X_train_scaled, X_test_scaled  # Assign training and testing features
            feature_names = feature_names_train  # Use training feature names

        return X_train, X_test, y_train, y_test, feature_names  # Return split data and feature names
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_models():
    """
    Initializes and returns a dictionary of models to train.

    :param: None
    :return: Dictionary of model name and instance
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Initializing models for training...{Style.RESET_ALL}"
        )  # Output the verbose message

        return {  # Dictionary of models to train
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=N_JOBS),
            "SVM": SVC(kernel="rbf", probability=True, random_state=42),
            "XGBoost": XGBClassifier(eval_metric="mlogloss", random_state=42, n_jobs=N_JOBS),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=N_JOBS),
            "Nearest Centroid": NearestCentroid(),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42),
            "LightGBM": lgb.LGBMClassifier(
                force_row_wise=True, min_gain_to_split=0.01, random_state=42, verbosity=-1, n_jobs=N_JOBS
            ),
            "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
            # "Few-Shot Learning": FewShotLearning(),
            # "Contrastive Learning": ContrastiveLearning()
        }
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def format_duration(seconds):
    """
    Formats a duration in seconds into a human-readable string.

    :param seconds: Duration in seconds (float or int)
    :return: Formatted string with the duration in appropriate units (seconds, minutes, or hours)
    """
    
    try:
        if seconds < 60:  # Less than one minute
            return f"{seconds:.2f}s"
        elif seconds < 3600:  # Less than one hour
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {int(remaining_seconds)}s"
        else:  # One hour or more
            hours = int(seconds // 3600)
            remaining_minutes = int((seconds % 3600) // 60)
            remaining_seconds = seconds % 60
            return f"{hours}h {remaining_minutes}m {int(remaining_seconds)}s"
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


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
    
    try:
        start_time = time.time()  # Start timer for training duration in seconds
        model.fit(X_train, y_train)  # Train the model
        duration = [
            time.time() - start_time,
            format_duration(time.time() - start_time),
        ]  # List to store the duration of training in seconds and in string

        print(
            f"{BackgroundColors.GREEN}  - Training {BackgroundColors.CYAN}{index:02d} - {model_name}{BackgroundColors.GREEN} for the dataset {BackgroundColors.CYAN}{dataset_name}{BackgroundColors.GREEN} done in {BackgroundColors.CYAN}{duration[1]}{Style.RESET_ALL}"
        )

        return model, duration  # Return trained model and duration
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def truncate_value(value):
    """
    Format a numeric value to 4 decimal places, or return None if not possible.

    :param value: Numeric value
    :return: Formatted string or None
    """
    
    try:
        try:  # Try to format the value
            if value is None:  # If value is None
                return None  # Return None
            v = float(value)  # Convert to float
            truncated = math.trunc(v * 10000) / 10000.0  # Truncate to 4 decimal places
            return f"{truncated:.4f}"  # Return formatted string
        except Exception:  # On failure
            return None  # Return None
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def build_extended_metrics(conf_matrix, labels, duration_str):
    """
    Builds extended metrics for each class.

    :param conf_matrix: Confusion matrix
    :param labels: List of class labels
    :param duration_str: String of the duration of the model training
    :return: DataFrame with extended metrics per class
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Building extended metrics from confusion matrix...{Style.RESET_ALL}"
        )  # Output the verbose message

        metrics_list = []  # List to store metrics for each class

        for i, label in enumerate(labels):  # Iterate through each label
            TP = conf_matrix[i, i]  # True Positives for the class
            FN = np.sum(conf_matrix[i, :]) - TP  # False Negatives for the class
            FP = np.sum(conf_matrix[:, i]) - TP  # False Positives for the class
            TN = np.sum(conf_matrix) - (TP + FP + FN)  # True Negatives for the class

            support = TP + FN  # Support for the class (number of true instances)
            acc_val = (TP + TN) / np.sum(conf_matrix) if np.sum(conf_matrix) > 0 else 0  # Accuracy value
            prec_val = TP / (TP + FP) if (TP + FP) > 0 else 0  # Precision value
            rec_val = TP / (TP + FN) if (TP + FN) > 0 else 0  # Recall value
            f1_val = 2 * prec_val * rec_val / (prec_val + rec_val) if (prec_val + rec_val) > 0 else 0  # F1-Score value
            accuracy = truncate_value(acc_val)  # Truncated accuracy
            precision = truncate_value(prec_val)  # Truncated precision
            recall = truncate_value(rec_val)  # Truncated recall
            f1 = truncate_value(f1_val)  # Truncated F1-Score

            metrics_list.append(
                {  # Append the metrics for the class to the list
                    "Class": label,
                    "Training Duration": duration_str,  # Format the training duration
                    "Correct (TP)": TP,
                    "Wrong (FN)": FN,
                    "False Positives (FP)": FP,
                    "True Negatives (TN)": TN,
                    "Support": support,
                    "Accuracy (per class)": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-Score": f1,
                }
            )

        return pd.DataFrame(metrics_list)  # Return a DataFrame with the extended metrics for each class
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


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
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Evaluating model: {model.__class__.__name__}...{Style.RESET_ALL}"
        )  # Output the verbose message

        preds = model.predict(X_test)  # Make predictions on the test data

        report = classification_report(
            y_test, preds, output_dict=True, zero_division=0
        )  # Generate classification report as dictionary

        conf_matrix = confusion_matrix(y_test, preds)  # Generate confusion matrix

        metrics_df = build_extended_metrics(
            conf_matrix, model.classes_, duration_str
        )  # Build extended metrics DataFrame from confusion matrix

        return report, metrics_df  # Return the report and extended metrics
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def apply_zebra_style(df):
    """
    Apply zebra-striping style to a pandas DataFrame using Styler.

    :param df: pandas DataFrame to style
    :return: pandas.io.formats.style.Styler styled object ready for export
    """
    try:
        def _zebra(row):  # Define row-wise zebra function
            return ["background-color: #ffffff" if i % 2 == 0 else "background-color: #f2f2f2" for i in range(len(row))]  # Return CSS list per column

        styled = df.style.apply(_zebra, axis=1)  # Create Styler with zebra striping applied
        return styled  # Return styled object
    except Exception as e:  # If styling fails
        raise  # Propagate exception without swallowing


def export_dataframe_image(styled_df, output_path):
    """
    Export a pandas Styler object to a PNG file using dataframe_image.

    :param styled_df: pandas Styler object to export
    :param output_path: Path to PNG file to write
    :return: None
    """
    import os  # Local import to keep top-level imports minimal
    from pathlib import Path  # Local import for Path handling

    try:
        out_path = Path(output_path)  # Create Path object for output
        parent = out_path.parent  # Get parent directory
        if not parent.exists():  # Ensure parent exists
            parent.mkdir(parents=True, exist_ok=True)  # Create parent directories if missing

        if not os.access(str(parent), os.W_OK):  # Check write permission on parent
            raise PermissionError(f"Directory not writable: {parent}")  # Raise if not writable

        dfi.export(styled_df, str(out_path))  # Use dataframe_image to export styled DataFrame to PNG
        return  # Return None explicitly
    except Exception:
        raise  # Propagate any exception (do not swallow)


def generate_table_image_from_dataframe(df, csv_path):
    """
    Generate a zebra-striped PNG table image for a DataFrame, saved beside the CSV.

    :param df: pandas DataFrame in memory to render
    :param csv_path: Path to the CSV file (image will be same base name with .png)
    :return: Path to generated PNG image
    """
    try:
        png_path = str(Path(csv_path).with_suffix('.png'))  # Build PNG path by replacing extension
        styled = apply_zebra_style(df)  # Apply zebra styling to DataFrame
        export_dataframe_image(styled, png_path)  # Export styled DataFrame as PNG image
        return png_path  # Return path to generated PNG
    except Exception:
        raise  # Propagate exceptions to caller


def generate_csv_and_image(df, csv_path, is_visualizable=True, **to_csv_kwargs):
    """
    Save DataFrame to CSV and optionally generate a zebra-striped PNG image beside it.

    :param df: pandas DataFrame to save
    :param csv_path: Destination CSV file path
    :param is_visualizable: Whether to generate the PNG image (default True)
    :param to_csv_kwargs: Additional keyword args forwarded to pandas.DataFrame.to_csv
    :return: Path to saved CSV file
    """
    try:
        df.to_csv(csv_path, **to_csv_kwargs)  # Save DataFrame to CSV with original kwargs
        if is_visualizable:  # Generate image only if flagged
            generate_table_image_from_dataframe(df, csv_path)  # Generate PNG from in-memory DataFrame
        return csv_path  # Return CSV path
    except Exception:
        raise  # Propagate exceptions (no silent failures)


def save_results(report, metrics_df, results_dir, index, model_name, feat_extraction_method=""):
    """
    Saves the classification report and extended metrics to disk.

    :param report: Classification report dictionary
    :param metrics_df: DataFrame with extended metrics
    :param results_dir: Directory of the results of the dataset
    :param index: Index of the model
    :param model_name: Name of the model
    :param feat_extraction_method: Optional suffix to add to filename (e.g., feature selection method)
    """
    
    try:
        verbose_output(f"{BackgroundColors.GREEN}Saving results for {model_name}...{Style.RESET_ALL}")

        if not os.path.exists(results_dir):  # If the results directory does not exist
            os.makedirs(results_dir)  # Create the results directory

        model_name_clean = model_name.replace(" ", "_").replace("(", "").replace(")", "")  # Clean model name for filename
        filename_base = (
            f"{results_dir}/{index:02d}-{model_name_clean}{feat_extraction_method}"  # Base filename for saving results
        )

        pd.DataFrame(report).transpose().to_csv(
            f"{filename_base}-Classification_report.csv", float_format="%.4f", index_label="Class"
        )  # Save classification report
        metrics_df.to_csv(
            f"{filename_base}-Extended_metrics.csv", index=False, float_format="%.2f"
        )  # Save extended confusion matrix
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def extract_average_metrics(metrics_df, dataset_name, model_name):
    """
    Extracts the average row from the extended metrics dataframe.

    :param metrics_df: DataFrame with extended metrics
    :param dataset_name: Name of the dataset
    :param model_name: Name of the model
    :return: Dictionary with average metrics for summary
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Extracting average metrics for {model_name} on the {dataset_name} dataset...{Style.RESET_ALL}"
        )  # Output the verbose message

        avg_row = metrics_df.iloc[-1]  # Get the last row which contains the average metrics

        return {  # Return a dictionary with the average metrics
            "Dataset": dataset_name,
            "Training Duration": avg_row["Training Duration"],
            "Model": model_name,
            "Correct (TP)": int(avg_row["Correct (TP)"]),
            "Wrong (FN)": int(avg_row["Wrong (FN)"]),
            "False Positives (FP)": int(avg_row["False Positives (FP)"]),
            "True Negatives (TN)": int(avg_row["True Negatives (TN)"]),
            "Support": int(avg_row["Support"]),
            "Accuracy (per class)": avg_row["Accuracy (per class)"],
            "Precision": avg_row["Precision"],
            "Recall": avg_row["Recall"],
            "F1-Score": avg_row["F1-Score"],
        }
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_model_results_file_header():
    """
    Returns the header for the model results CSV file.

    :return: List of column names for the model results CSV
    """
    
    try:
        verbose_output(
            true_string=f"{BackgroundColors.GREEN}Getting header for model results CSV file...{Style.RESET_ALL}"
        )  # Verbose output indicating the header retrieval

        return [  # List of column names for the model results CSV
            "Dataset",  # Name of the dataset
            "Model",  # Name of the ML model
            "Training Duration",  # Duration of training the model
            "Correct (TP)",  # True Positives
            "Wrong (FN)",  # False Negatives
            "False Positives (FP)",  # False Positives
            "True Negatives (TN)",  # True Negatives
            "Support",  # Total number of samples for the class
            "Accuracy (per class)",  # Accuracy for the class (TP + TN) / Total
            "Precision",  # Precision score
            "Recall",  # Recall score
            "F1-Score",  # F1-score
        ]
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def generate_overall_performance_summary(all_model_scores, output_path=".", feat_extraction_method=""):
    """
    Generates an overall performance summary CSV combining all datasets and models with detailed metrics.

    :param all_model_scores: List of dictionaries with model scores.
    :param output_path: Path where the summary CSV will be saved.
    :param feat_extraction_method: Optional suffix to add to filename (e.g., feature selection method)
    :return: None
    """
    
    try:
        verbose_output(
            true_string=f"{BackgroundColors.GREEN}Generating overall performance summary...{Style.RESET_ALL}"
        )  # Print start message

        columns = get_model_results_file_header()  # Get the header for the model results CSV file

        formatted_scores = []  # Initialize list to store reformatted model score dictionaries

        for entry in all_model_scores:  # Iterate through each model score entry
            dataset_name = entry.get("Dataset", "Unknown")  # Use the dataset from entry or fallback to "Unknown"
            model_name = entry.get("Model", "")  # Get the full model name from the entry
            if "-" in model_name:  # If the model name has a prefix like "03-XGBoost"
                model_name = model_name.split("-", 1)[-1].replace("_", " ").strip()  # Remove numeric prefix and clean name

            formatted_scores.append(
                {  # Create a dictionary aligned with the defined column structure
                    "Dataset": dataset_name,  # Dataset name
                    "Model": model_name,  # Cleaned model name
                    "Training Duration": entry.get("Training Duration", 0),  # Training duration formatted
                    "Correct (TP)": entry.get("Correct (TP)", ""),  # True Positives
                    "Wrong (FN)": entry.get("Wrong (FN)", ""),  # False Negatives
                    "False Positives (FP)": entry.get("False Positives (FP)", ""),  # False Positives
                    "True Negatives (TN)": entry.get("True Negatives (TN)", ""),  # True Negatives
                    "Support": entry.get("Support", ""),  # Support (samples)
                    "Accuracy (per class)": entry.get("Accuracy (per class)", ""),  # Accuracy
                    "Precision": entry.get("Precision", ""),  # Precision
                    "Recall": entry.get("Recall", ""),  # Recall
                    "F1-Score": entry.get("F1-Score", ""),  # F1-Score
                }
            )

        formatted_scores = sorted(
            formatted_scores, key=lambda x: (x["Dataset"], -float(x["F1-Score"]))
        )  # Sort first by Dataset name (alphabetically), then by F1-Score (descending within each dataset)

        output_df = pd.DataFrame(
            formatted_scores, columns=columns
        )  # Create a DataFrame using the sorted scores and defined column order
        os.makedirs(output_path, exist_ok=True)  # Ensure the output directory exists
        output_file = os.path.join(
            output_path, f"Overall_Performance{feat_extraction_method}.csv"
        )  # Define the output file path with feature suffix
        output_df.to_csv(output_file, index=False)  # Save the DataFrame to CSV without including the index

        verbose_output(
            true_string=f"{BackgroundColors.GREEN}Overall performance summary saved to: {BackgroundColors.CYAN}{output_file}{Style.RESET_ALL}"
        )  # Print success message with file path
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def generate_feature_selection_comparison(
    baseline_metrics, feature_selected_metrics, output_path=".", feat_extraction_method=""
):
    """
    Generates a comparison CSV showing performance with and without feature selection.

    :param baseline_metrics: List of dictionaries with baseline model scores (without feature selection)
    :param feature_selected_metrics: List of dictionaries with feature-selected model scores
    :param output_path: Path where the comparison CSV will be saved
    :param feat_extraction_method: String indicating feature selection method
    :return: None
    """
    
    try:
        verbose_output(true_string=f"{BackgroundColors.GREEN}Generating feature selection comparison...{Style.RESET_ALL}")

        comparison_data = []  # Create comparison data

        for baseline in baseline_metrics:  # Iterate through each baseline metric
            model_name = baseline.get("Model", "")  # Get model name
            dataset_name = baseline.get("Dataset", "")  # Get dataset name

            feature_selected = next(
                (
                    fs
                    for fs in feature_selected_metrics
                    if fs.get("Model") == model_name and fs.get("Dataset") == dataset_name
                ),
                None,
            )  # Find matching feature-selected model

            if feature_selected:  # If a matching feature-selected model is found
                baseline_f1 = float(baseline.get("F1-Score", 0))  # Get baseline F1-Score
                feature_f1 = float(feature_selected.get("F1-Score", 0))  # Get feature-selected F1-Score
                improvement = (
                    ((feature_f1 - baseline_f1) / baseline_f1 * 100) if baseline_f1 > 0 else 0
                )  # Calculate percentage improvement

                comparison_data.append(
                    {  # Append comparison data
                        "Dataset": dataset_name,  # Dataset name
                        "Model": model_name,  # Model name
                        "F1-Score (Baseline)": truncate_value(baseline_f1),  # Truncated baseline F1-Score
                        "F1-Score (Feat. Selection)": truncate_value(feature_f1),  # Truncated feature-selected F1-Score
                        "Improvement (%)": truncate_value(improvement),  # Truncated improvement percentage
                        "Training Duration (Baseline)": baseline.get("Training Duration", ""),  # Baseline training duration
                        "Training Duration (Feat. Selection)": feature_selected.get(
                            "Training Duration", ""
                        ),  # Feature-selected training duration
                        "Features Used": feature_selected.get("Features Count", "N/A"),  # Number of features used
                    }
                )

        comparison_data = sorted(
            comparison_data, key=lambda x: (x["Dataset"], -x["Improvement (%)"])
        )  # Sort by improvement (descending)

        comparison_df = pd.DataFrame(comparison_data)  # Create DataFrame from comparison data
        os.makedirs(output_path, exist_ok=True)  # Ensure output directory exists
        output_file = os.path.join(
            output_path, f"{feat_extraction_method}-Feature_Selection_Comparison.csv"
        )  # Define output file path
        comparison_df.to_csv(output_file, index=False)  # Save comparison DataFrame to CSV

        print(
            f"{BackgroundColors.GREEN}Feature selection comparison saved to: {BackgroundColors.CYAN}{output_file}{Style.RESET_ALL}"
        )
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def cross_validate_model(model, X, y, cv=10, scoring="f1_weighted"):
    """
    Performs k-fold cross-validation for a given model.

    :param model: Model instance to evaluate
    :param X: Features
    :param y: Labels
    :param cv: Number of cross-validation folds (default is 10)
    :param scoring: Scoring metric to use (default is 'f1_weighted')
    :return: Returns mean and std of the chosen scoring metric.
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Performing {cv}-fold cross-validation for {model.__class__.__name__}...{Style.RESET_ALL}"
        )

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)  # Stratified K-Fold cross-validator
        scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)  # Perform cross-validation and get scores

        return scores.mean(), scores.std()  # Return mean and std of scores
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def train_and_evaluate_models(
    X_train,
    X_test,
    y_train,
    y_test,
    dataset_dir,
    dataset_name,
    use_cv=True,
    selected_features=None,
    features_file=None,
    return_metrics_only=False,
):
    """
    Trains and evaluates multiple models on the provided dataset.

    :param X_train: Training features
    :param X_test: Testing features
    :param y_train: Training labels
    :param y_test: Testing labels
    :param dataset_dir: Directory of the dataset
    :param dataset_name: Name of the dataset
    :param use_cv: Whether to use cross-validation (default is True)
    :param selected_features: List of selected features (if any)
    :param features_file: Path to the features selection file (if any)
    :param return_metrics_only: If True, only returns metrics without saving results
    :return: Tuple containing:
                            - Dictionary of trained models
                            - List of model metrics dictionaries
    """
    
    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Starting training and evaluation of models using {'cross-validation' if use_cv else 'train/test split'}...{Style.RESET_ALL}"
        )

        models = get_models()  # Get the dictionary of models to train
        model_metrics_list = []  # List to store metrics for each model
        results_dir = os.path.join(dataset_dir, "Results")  # Directory to save results

        feat_extraction_method = ""  # Suffix for feature extraction method
        if selected_features is not None and features_file:  # If feature selection is used and a features file is provided
            features_basename = os.path.basename(features_file)  # Get the base name of the features file
            features_name = os.path.splitext(features_basename)[0]  # Remove file extension
            if "Genetic" in features_name or "GA" in features_name:  # If genetic algorithm is indicated in the filename
                feat_extraction_method = "-GA"  # Use GA suffix
            elif "RFE" in features_name:  # If Recursive Feature Elimination is indicated in the filename
                feat_extraction_method = "-RFE"  # Use RFE suffix
            elif "PCA" in features_name:  # If PCA is indicated in the filename
                feat_extraction_method = "-PCA"  # Use PCA suffix
            else:  # Use first part of the filename (up to 15 chars) as suffix
                feat_extraction_method = f"-{features_name[:15]}"  # Generic suffix

        total_models = len(models)  # Total number of models to train
        model_pbar = tqdm(
            models.items(),
            total=total_models,
            desc=f"Training models for {dataset_name}",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            ncols=100,
        )

        for index, (name, model) in enumerate(model_pbar, start=1):  # Iterate through each model with progress bar
            model_pbar.set_description(f"[{index}/{total_models}] {name}")  # Update progress bar description
            send_telegram_message(TELEGRAM_BOT, [f"Training model {index}/{total_models}: {name} for dataset {dataset_name}"])
            if use_cv:  # If cross-validation is to be used
                mean_score, std_score = cross_validate_model(model, X_train, y_train)  # Perform cross-validation
                duration = [0, "CV"]  # CV is not timed here
                tqdm.write(
                    f"{BackgroundColors.GREEN}✓ {name} - F1: {truncate_value(mean_score)} ± {truncate_value(std_score)}{Style.RESET_ALL}"
                )  # Output cross-validation results
            else:  # If not using cross-validation
                start_time = time.time()  # Start timer for training duration
                model.fit(X_train, y_train)  # Train the model
                duration = [
                    time.time() - start_time,
                    format_duration(time.time() - start_time),
                ]  # List to store the duration of training in seconds and in string
                tqdm.write(
                    f"{BackgroundColors.GREEN}✓ {name} trained in {duration[1]}{Style.RESET_ALL}"
                )  # Output training duration

                report, metrics_df = evaluate_model(model, X_test, y_test, duration[1])  # Evaluate the model
                if not return_metrics_only:  # If not only returning metrics, save the results
                    save_results(
                        report, metrics_df, results_dir, index, name, feat_extraction_method=feat_extraction_method
                    )  # Save the results to disk
                avg_row = metrics_df.iloc[-1]  # Get the average metrics row
                metrics_dict = {  # Create a dictionary with average metrics
                    "Dataset": dataset_name,
                    "Model": name,
                    "Training Duration": duration[1],
                    "Correct (TP)": int(avg_row["Correct (TP)"]),
                    "Wrong (FN)": int(avg_row["Wrong (FN)"]),
                    "False Positives (FP)": int(avg_row["False Positives (FP)"]),
                    "True Negatives (TN)": int(avg_row["True Negatives (TN)"]),
                    "Support": int(avg_row["Support"]),
                    "Accuracy (per class)": avg_row["Accuracy (per class)"],
                    "Precision": avg_row["Precision"],
                    "Recall": avg_row["Recall"],
                    "F1-Score": avg_row["F1-Score"],
                    "Features Count": len(selected_features) if selected_features is not None else "All",
                }
                model_metrics_list.append(metrics_dict)  # Append metrics dictionary to the list

        if not use_cv and not return_metrics_only:  # If not using cross-validation and not only returning metrics
            generate_overall_performance_summary(
                model_metrics_list, output_path=results_dir, feat_extraction_method=feat_extraction_method
            )  # Generate overall performance summary

        if (
            selected_features is not None and model_metrics_list and not return_metrics_only
        ):  # If feature selection is used and metrics are available
            print(
                f"{BackgroundColors.GREEN}Models trained with {BackgroundColors.CYAN}{len(selected_features)}{BackgroundColors.GREEN} selected features{Style.RESET_ALL}"
            )

        return models, model_metrics_list  # Return trained models and their metrics
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


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
    
    try:
        verbose_output(f"{BackgroundColors.GREEN}Explaining Predictions with SHAP TreeExplainer...{Style.RESET_ALL}")
        X_explain = X_test[:5]  # Select the first 5 instances for explanation

        explainer = shap.TreeExplainer(model)  # Create a SHAP TreeExplainer for the model
        shap_values = explainer.shap_values(X_explain)  # Calculate SHAP values for the selected instances

        for i in range(len(X_explain)):  # Iterate through each instance
            shap_val = shap_values[i]  # Get SHAP values for the instance
            feat_val = X_explain.iloc[i].values.flatten()  # Get feature values for the instance

            if len(feature_names) != len(shap_val) or len(shap_val) != len(feat_val):  # Verify if lengths match
                verbose_output(f"{BackgroundColors.RED}Lengths do not match for instance {i+1}:{Style.RESET_ALL}")
                continue  # Skip this instance if lengths do not match

            shap_df = pd.DataFrame(
                {"feature": feature_names, "shap_value": shap_val.flatten(), "feature_value": feat_val}
            )  # Create a DataFrame for SHAP values
            shap_df.to_csv(
                f"{model_name}_tree_shap_instance_{i+1}.csv", index=False, float_format="%.2f"
            )  # Save SHAP values to CSV
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def explain_predictions_with_shap(model, X_train, X_test, feature_names):
    """
    Explains model predictions using SHAP values.

    :param model: Trained model
    :param X_train: Training features
    :param X_test: Testing features
    :param feature_names: Names of the features
    :return: None
    """
    
    try:
        verbose_output(f"{BackgroundColors.GREEN}Explaining predictions with SHAP Explainer...{Style.RESET_ALL}")

        X_explain = X_test[:5]  # Select the first 5 instances for explanation

        explainer = shap.Explainer(model, X_train, feature_names=feature_names)  # Create a SHAP explainer for the model
        shap_values = explainer(X_explain)  # Calculate SHAP values for the selected instances

        for i in range(len(X_explain)):  # Iterate through each instance
            shap_val = shap_values[i].values  # Get SHAP values for the instance
            feat_val = shap_values[i].data  # Get feature values for the instance

            shap_val = shap_val.flatten()  # Flatten the SHAP values
            feat_val = feat_val.flatten()  # Flatten the feature values

            if len(feature_names) != len(shap_val) or len(shap_val) != len(feat_val):  # Verify if lengths match
                verbose_output(f"{BackgroundColors.RED}Lengths do not match for instance {i+1}:{Style.RESET_ALL}")
                continue  # Skip this instance if lengths do not match

            shap_df = pd.DataFrame(
                {"feature": feature_names, "shap_value": shap_val, "feature_value": feat_val}
            )  # Create a DataFrame for SHAP values

            shap_df.to_csv(f"shap_values_instance_{i+1}.csv", index=False)  # Save SHAP values to CSV
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


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
    
    try:
        verbose_output(f"{BackgroundColors.GREEN}Explaining Predictions with LIME...{Style.RESET_ALL}")

        X_explain = X_test[:5]  # Select the first 5 instances for explanation

        explainer = LimeTabularExplainer(  # Create a LIME explainer for tabular data
            training_data=X_train.values,  # Training data for the explainer
            feature_names=feature_names,  # Names of the features
            class_names=(
                [str(c) for c in model.classes_] if hasattr(model, "classes_") else ["Class 0", "Class 1"]
            ),  # Class names for the model
            mode="classification",  # Mode of the explainer (classification or regression)
        )

        for i in range(len(X_explain)):  # Iterate through each instance
            exp = explainer.explain_instance(  # Explain the instance using LIME
                data_row=X_explain.iloc[i].values,  # Data row to explain
                predict_fn=model.predict_proba,  # Prediction function for the model
                num_features=len(feature_names),  # Number of features to include in the explanation
            )
            lime_df = pd.DataFrame(exp.as_list(), columns=["feature", "weight"])  # Create a DataFrame for LIME explanation
            lime_df.to_csv(
                f"{model_name}_lime_instance_{i+1}.csv", index=False, float_format="%.2f"
            )  # Save LIME explanation to CSV
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


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
    
    try:
        print(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Iniciando explicações para {model_name}...{Style.RESET_ALL}")

        if (
            "xgboost" in str(type(model)).lower() or "randomforest" in str(type(model)).lower()
        ):  # If the model is XGBoost or Random Forest
            explain_predictions_with_tree_shap(
                model, X_train, X_test, feature_names, model_name=model_name
            )  # Explain predictions using SHAP's TreeExplainer
        else:  # For other models
            explain_predictions_with_shap(model, X_train, X_test, feature_names)  # Explain predictions using SHAP values

        explain_predictions_with_lime(
            model, X_train, X_test, feature_names, model_name=model_name
        )  # Explain predictions using LIME, as it works with any model
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def to_seconds(obj):
    """
    Converts various time-like objects to seconds.
    
    :param obj: The object to convert (can be int, float, timedelta, datetime, etc.)
    :return: The equivalent time in seconds as a float, or None if conversion fails
    """
    
    try:
        if obj is None:  # None can't be converted
            return None  # Signal failure to convert
        if isinstance(obj, (int, float)):  # Already numeric (seconds or timestamp)
            return float(obj)  # Return as float seconds
        if hasattr(obj, "total_seconds"):  # Timedelta-like objects
            try:  # Attempt to call total_seconds()
                return float(obj.total_seconds())  # Use the total_seconds() method
            except Exception:
                pass  # Fallthrough on error
        if hasattr(obj, "timestamp"):  # Datetime-like objects
            try:  # Attempt to call timestamp()
                return float(obj.timestamp())  # Use timestamp() to get seconds since epoch
            except Exception:
                pass  # Fallthrough on error
        return None  # Couldn't convert
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def calculate_execution_time(start_time, finish_time=None):
    """
    Calculates the execution time and returns a human-readable string.

    Accepts either:
    - Two datetimes/timedeltas: `calculate_execution_time(start, finish)`
    - A single timedelta or numeric seconds: `calculate_execution_time(delta)`
    - Two numeric timestamps (seconds): `calculate_execution_time(start_s, finish_s)`

    Returns a string like "1h 2m 3s".
    """
    
    try:
        if finish_time is None:  # Single-argument mode: start_time already represents duration or seconds
            total_seconds = to_seconds(start_time)  # Try to convert provided value to seconds
            if total_seconds is None:  # Conversion failed
                try:  # Attempt numeric coercion
                    total_seconds = float(start_time)  # Attempt numeric coercion
                except Exception:
                    total_seconds = 0.0  # Fallback to zero
        else:  # Two-argument mode: Compute difference finish_time - start_time
            st = to_seconds(start_time)  # Convert start to seconds if possible
            ft = to_seconds(finish_time)  # Convert finish to seconds if possible
            if st is not None and ft is not None:  # Both converted successfully
                total_seconds = ft - st  # Direct numeric subtraction
            else:  # Fallback to other methods
                try:  # Attempt to subtract (works for datetimes/timedeltas)
                    delta = finish_time - start_time  # Try subtracting (works for datetimes/timedeltas)
                    total_seconds = float(delta.total_seconds())  # Get seconds from the resulting timedelta
                except Exception:  # Subtraction failed
                    try:  # Final attempt: Numeric coercion
                        total_seconds = float(finish_time) - float(start_time)  # Final numeric coercion attempt
                    except Exception:  # Numeric coercion failed
                        total_seconds = 0.0  # Fallback to zero on failure

        if total_seconds is None:  # Ensure a numeric value
            total_seconds = 0.0  # Default to zero
        if total_seconds < 0:  # Normalize negative durations
            total_seconds = abs(total_seconds)  # Use absolute value

        days = int(total_seconds // 86400)  # Compute full days
        hours = int((total_seconds % 86400) // 3600)  # Compute remaining hours
        minutes = int((total_seconds % 3600) // 60)  # Compute remaining minutes
        seconds = int(total_seconds % 60)  # Compute remaining seconds

        if days > 0:  # Include days when present
            return f"{days}d {hours}h {minutes}m {seconds}s"  # Return formatted days+hours+minutes+seconds
        if hours > 0:  # Include hours when present
            return f"{hours}h {minutes}m {seconds}s"  # Return formatted hours+minutes+seconds
        if minutes > 0:  # Include minutes when present
            return f"{minutes}m {seconds}s"  # Return formatted minutes+seconds
        return f"{seconds}s"  # Fallback: only seconds
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def play_sound():
    """
    Plays a sound when the program finishes.

    :param: None
    :return: None
    """
    
    try:
        if verify_filepath_exists(SOUND_FILE):  # If the sound file exists
            if platform.system() in SOUND_COMMANDS:  # If the platform.system() is in the SOUND_COMMANDS dictionary
                os.system(f"{SOUND_COMMANDS[platform.system()]} {SOUND_FILE}")  # Play the sound
            else:  # If the platform.system() is not in the SOUND_COMMANDS dictionary
                print(
                    f"{BackgroundColors.RED}The {BackgroundColors.CYAN}platform.system(){BackgroundColors.RED} is not in the {BackgroundColors.CYAN}SOUND_COMMANDS dictionary{BackgroundColors.RED}. Please add it!{Style.RESET_ALL}"
                )
        else:  # If the sound file does not exist
            print(
                f"{BackgroundColors.RED}Sound file {BackgroundColors.CYAN}{SOUND_FILE}{BackgroundColors.RED} not found. Make sure the file exists.{Style.RESET_ALL}"
            )
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def main(use_cv=False, extract_features=True, compare_feature_selection=None):
    """
    Main function to run the machine learning pipeline on multiple datasets.

    :param use_cv: Boolean flag to indicate whether to use cross-validation instead of train/test split (default is False)
    :param extract_features: Boolean flag to indicate whether to extract features from FEATURES_FILE (default is True)
    :param compare_feature_selection: Boolean flag to compare performance with/without feature selection (default is True if extract_features is True)
    :return: None
    """
    
    try:
        print(
            f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}DDoS Detector{BackgroundColors.GREEN} program!{Style.RESET_ALL}",
            end="\n\n",
        )
        start_time = datetime.datetime.now()  # Get the start time of the program
        
        setup_telegram_bot()  # Setup Telegram bot if configured
        
        send_telegram_message(TELEGRAM_BOT, [f"{BackgroundColors.GREEN}DDoS Detector main.py execution started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}"])

        sorted_datasets = sorted(DATASETS.items())  # Sort datasets alphabetically by keys

        features_to_use = None  # Initialize selected features holder
        features_file_used = None  # Initialize features file reference holder

        if compare_feature_selection is None:  # If compare_feature_selection is not set
            compare_feature_selection = extract_features  # Enable only if extracting features

        all_model_scores = []  # Overall model performance list
        baseline_metrics = []  # Metrics from baseline (no Feature Selection)
        feature_selected_metrics = []  # Metrics from Feature Selection-enabled runs

        for index, (dataset_key, dataset_info) in enumerate(sorted_datasets, start=1):  # Iterate datasets
            training_file_path = dataset_info.get("train")  # Get train file
            testing_file_path = dataset_info.get("test")  # Get test file
            feature_files = dataset_info.get("features", [])  # Get features list

            dataset_dir = os.path.dirname(str(training_file_path))  # Get the directory of the dataset file
            dataset_name = os.path.splitext(os.path.basename(str(training_file_path)))[
                0
            ]  # Extract dataset name (casts to str)

            print(
                f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing dataset {BackgroundColors.CYAN}{index}/{len(sorted_datasets)}{BackgroundColors.GREEN}: {BackgroundColors.CYAN}{dataset_name}{BackgroundColors.GREEN}{Style.RESET_ALL}"
            )

            send_telegram_message(TELEGRAM_BOT, [f"Processing dataset {index}/{len(sorted_datasets)}: {dataset_name}"])

            if not verify_filepath_exists(training_file_path):  # Verify train file
                print(f"{BackgroundColors.RED}Missing training file for {dataset_name}. Skipping.{Style.RESET_ALL}")
                continue  # Skip dataset if missing train file

            if not verify_filepath_exists(testing_file_path):  # Verify test file
                testing_file_path = training_file_path  # Fallback to train file

            features_to_use = None  # Reset feature list per dataset
            features_file_used = None  # Reset last-used features file

            if extract_features and feature_files:  # If Feature Selection is enabled and features exist
                merged_features = []  # Initialize merged features list
                for feature_file in feature_files:  # Loop through all feature files
                    if verify_filepath_exists(feature_file):  # Verify file existence
                        feats = get_features_from_file(feature_file)  # Load features
                        merged_features.extend(feats)  # Add to merge list
                        features_file_used = feature_file  # Track last loaded file
                    else:  # File not found
                        print(f"{BackgroundColors.RED}Feature file not found: {feature_file}{Style.RESET_ALL}")
                features_to_use = list(dict.fromkeys(merged_features)) if merged_features else None  # Remove duplicates

                if features_to_use:  # Print info if features loaded
                    verbose_output(
                        f"{BackgroundColors.GREEN}Loaded {BackgroundColors.CYAN}{len(features_to_use)}{BackgroundColors.GREEN} features from multiple feature files.{Style.RESET_ALL}\n"
                    )

            train_df, test_df, split_required = load_and_prepare_data(
                training_file_path, testing_file_path
            )  # Load and preprocess dataset

            label_col = detect_label_column(train_df.columns)  # Detect label column automatically
            if label_col is None:  # If not detected
                print(f"{BackgroundColors.RED}No label column detected in {dataset_name}.{Style.RESET_ALL}")
                print(
                    f"{BackgroundColors.RED}Available columns: {BackgroundColors.CYAN}{list(train_df.columns)}{BackgroundColors.RED}...{Style.RESET_ALL}"
                )
                print(
                    f"{BackgroundColors.RED}Total columns: {BackgroundColors.CYAN}{len(train_df.columns)}{Style.RESET_ALL}"
                )
                label_col = input(
                    f"{BackgroundColors.GREEN}Please enter the label column name for {dataset_name}: {Style.RESET_ALL}"
                ).strip()  # Ask user for label column
                if label_col == "":  # Skip if empty input
                    continue  # Skip dataset
                while label_col not in train_df.columns:  # Validate column name
                    print(f"{BackgroundColors.RED}Invalid label column name. Please try again.{Style.RESET_ALL}")
                    label_col = input(
                        f"{BackgroundColors.GREEN}Please enter the label column name for {dataset_name}: {Style.RESET_ALL}"
                    ).strip()  # Ask again

            if compare_feature_selection and features_to_use is not None:  # Run baseline if needed
                verbose_output(f"{BackgroundColors.GREEN}Running baseline (without feature selection)...{Style.RESET_ALL}")
                X_train_base, X_test_base, y_train_base, y_test_base, feature_names_base = split_data(
                    train_df, test_df, split_required, label_col=label_col, selected_features=None
                )  # Split without Feature Selection
                verbose_output(f"{BackgroundColors.GREEN}Training baseline models...{Style.RESET_ALL}")
                _, baseline_scores = train_and_evaluate_models(
                    X_train_base,
                    X_test_base,
                    y_train_base,
                    y_test_base,
                    dataset_dir,
                    dataset_name,
                    use_cv=use_cv,
                    selected_features=None,
                    features_file=None,
                    return_metrics_only=True,
                )  # Train baseline
                baseline_metrics.extend(baseline_scores)  # Store baseline metrics

            verbose_output(f"{BackgroundColors.GREEN}Preparing data with feature selection...{Style.RESET_ALL}")
            X_train, X_test, y_train, y_test, feature_names = split_data(
                train_df, test_df, split_required, label_col=label_col, selected_features=features_to_use
            )  # Split with Feature Selection
            verbose_output(f"{BackgroundColors.GREEN}Training models with feature selection...{Style.RESET_ALL}")
            models, dataset_model_scores = train_and_evaluate_models(
                X_train,
                X_test,
                y_train,
                y_test,
                dataset_dir,
                dataset_name,
                use_cv=use_cv,
                selected_features=features_to_use,
                features_file=features_file_used,
            )  # Train Feature Selection models

            all_model_scores.extend(dataset_model_scores) if dataset_model_scores else None  # Add to overall scores

            if compare_feature_selection and features_to_use is not None:  # If comparing Feature Selection is enabled
                feature_selected_metrics.extend(dataset_model_scores)  # Store Feature Selection metrics

        feat_extraction_method = ""  # Initialize method tag
        if features_to_use is not None and features_file_used:  # Map features file to method
            features_basename = os.path.basename(features_file_used)  # Extract base filename
            features_name = os.path.splitext(features_basename)[0]  # Remove extension
            if "Genetic" in features_name or "GA" in features_name:  # If Genetic Algorithm indicated
                feat_extraction_method = "GA"  # Genetic Algorithm tag
            elif "RFE" in features_name:
                feat_extraction_method = "RFE"  # RFE tag
            elif "PCA" in features_name:
                feat_extraction_method = "PCA"  # PCA tag
            else:  # Generic case
                feat_extraction_method = f"-{features_name[:15]}"  # Generic prefix

        (
            generate_overall_performance_summary(all_model_scores, feat_extraction_method=feat_extraction_method)
            if all_model_scores
            else None
        )  # Generate overall summary

        if (
            compare_feature_selection and baseline_metrics and feature_selected_metrics
        ):  # If comparing Feature Selection is enabled and metrics exist
            comparison_output_path = os.path.join(os.getcwd(), OUTPUT_DIR)  # Path for comparison report
            generate_feature_selection_comparison(
                baseline_metrics,
                feature_selected_metrics,
                output_path=comparison_output_path,
                feat_extraction_method=feat_extraction_method,
            )  # Generate Feature Selection comparison

        finish_time = datetime.datetime.now()  # Get the finish time of the program
        print(
            f"{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
        )  # Output the start and finish times
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
        )  # Output the end of the program message
        
        send_telegram_message(TELEGRAM_BOT, [f"{BackgroundColors.GREEN}DDoS Detector main.py execution finished. Execution time: {calculate_execution_time(start_time, finish_time)}"])

        atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None  # Register sound on exit if enabled
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


if __name__ == "__main__":
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """

    main()  # Call the main function
