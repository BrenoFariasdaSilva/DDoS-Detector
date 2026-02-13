"""
================================================================================
Individual and Stacking Ensemble Classifier Evaluation Script (stacking.py)
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-12-01
Description :
    Orchestrates evaluation of individual classifiers and a stacking ensemble
    across multiple feature sets derived from the project's feature-analysis
    artifacts (Genetic Algorithm, RFE, PCA). The script centralizes dataset
    loading, preprocessing, feature-set assembly, per-model evaluation and the
    export of consolidated CSV results for downstream analysis.

Core capabilities:
    - Automatic loading and sanitization of CSV datasets (NaN/infinite removal)
    - Integration of GA, RFE and PCA outputs to build alternative feature sets
    - Scaling, optional PCA projection and selective feature subsetting
    - Evaluation of many classifiers (RF, SVM, XGBoost, LightGBM, etc.) and a
        stacking meta-classifier combining their predictions
    - Calculation of standard metrics (accuracy, precision, recall, F1) plus
        FPR/FNR and elapsed-time reporting
    - Export of `Stacking_Classifier_Results.csv` including `features_list`
        and hardware metadata for reproducibility
    - Utilities to discover feature-analysis files at file, parent or dataset level

Usage:
    - Configure `DATASETS` mapping or call `main()` directly.
    - Run: `python3 stacking.py` or via the repository Makefile target.

Outputs:
    - `Stacking_Classifier_Results.csv` (per-dataset `Feature_Analysis/` directory)
    - Terminal logs, optional Telegram notifications and sound on completion

Notes & conventions:
    - Input CSVs are expected under `Datasets/<DatasetName>/...` and the last
        column conventionally contains the target variable.
    - Feature-analysis artifacts are expected under `.../Feature_Analysis/`:
        `Genetic_Algorithm_Results.csv`, `RFE_Run_Results.csv`, `PCA_Results.csv`.
    - Defaults assume CSV input; Parquet support can be added as needed.
    - Toggle `VERBOSE = True` for additional diagnostic output.

TODOs:
    - Add CLI argument parsing for dataset paths and runtime flags.
    - Add native Parquet support and safer large-file streaming.
    - Add voting ensemble baseline and parallelize per-feature-set evaluations.

Dependencies:
    - Python >= 3.8
    - pandas, numpy, scikit-learn, colorama, lightgbm, xgboost
    - Optional: telegram_bot for notifications
"""

import argparse  # For parsing command-line arguments
import ast  # For safely evaluating Python literals
import atexit  # For playing a sound when the program finishes
import concurrent.futures  # For parallel execution
import datetime  # For getting the current date and time
import glob  # For file pattern matching
import json  # Import json for handling JSON strings within the CSV
import lightgbm as lgb  # For LightGBM model
import math  # For mathematical operations
import matplotlib  # For plotting configuration
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt  # For creating t-SNE visualization plots
import numpy as np  # Import numpy for numerical operations
import optuna  # For Bayesian hyperparameter optimization (AutoML)
import os  # For running a command in the terminal
import pandas as pd  # Import pandas for data manipulation
import pickle  # For loading PCA objects
import platform  # For getting the operating system name
import psutil  # For checking system RAM
import re  # For regular expressions
import subprocess  # For running small system commands (sysctl/wmic)
import sys  # For system-specific parameters and functions
import telegram_bot as telegram_module  # For setting Telegram prefix and device info
import time  # For measuring execution time
from colorama import Style  # For terminal text styling
from joblib import dump, load  # For exporting and loading trained models and scalers
from Logger import Logger  # For logging output to both terminal and file
from pathlib import Path  # For handling file paths
from sklearn.decomposition import PCA  # For Principal Component Analysis
from sklearn.ensemble import (  # For ensemble models
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression  # For logistic regression model
from sklearn.manifold import TSNE  # For t-SNE dimensionality reduction
from sklearn.metrics import (  # For performance metrics
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split  # For splitting the dataset and CV
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid  # For k-nearest neighbors model
from sklearn.neural_network import MLPClassifier  # For neural network model
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For label encoding and feature scaling
from sklearn.svm import SVC  # For Support Vector Machine model
from sklearn.tree import DecisionTreeClassifier  # For Decision Tree classifier model
from telegram_bot import TelegramBot, send_telegram_message  # For sending progress messages to Telegram
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


# Global Configuration Container:
CONFIG = {}  # Will be initialized by initialize_config() - holds all runtime settings

# Telegram Bot Setup:
TELEGRAM_BOT = None  # Global Telegram bot instance (initialized in setup_telegram_bot)

# Logger Setup:
logger = None  # Will be initialized in initialize_logger()

# Functions Definitions:


def run_automl_stacking_search(X_train, y_train, model_study, file_path):
    """
    Runs Optuna-based optimization to find the best stacking ensemble configuration.

    :param X_train: Scaled training features (numpy array)
    :param y_train: Training target labels (numpy array)
    :param model_study: Completed Optuna study from model search
    :param file_path: Path to the dataset file for logging
    :return: Tuple (best_stacking_config, stacking_study) or (None, None) on failure
    """

    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Starting AutoML stacking search with {BackgroundColors.CYAN}{config.get("automl", {}).get("stacking_trials", 20)}{BackgroundColors.GREEN} trials...{Style.RESET_ALL}"
    )  # Output search start message

    candidate_models = extract_top_automl_models(model_study, top_n=config.get("automl", {}).get("stacking_top_n", 5))  # Get top models from model search

    if len(candidate_models) < 2:  # If not enough candidate models
        print(
            f"{BackgroundColors.YELLOW}Not enough candidate models for stacking search. Need at least 2, got {len(candidate_models)}.{Style.RESET_ALL}"
        )  # Output warning
        return (None, None)  # Return None tuple

    print(
        f"{BackgroundColors.GREEN}Candidate base learners: {BackgroundColors.CYAN}{list(candidate_models.keys())}{Style.RESET_ALL}"
    )  # Output candidate models

    optuna.logging.set_verbosity(optuna.logging.WARNING)  # Suppress verbose Optuna logging

    sampler = optuna.samplers.TPESampler(seed=config.get("automl", {}).get("random_state", 42) + 1)  # Create sampler with different seed
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)  # Create pruner

    stacking_study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner, study_name="automl_stacking_search"
    )  # Create Optuna study for stacking optimization

    objective_fn = lambda trial: automl_stacking_objective(
        trial, X_train, y_train, config.get("automl", {}).get("cv_folds", 5), candidate_models
    )  # Create stacking objective wrapper
    stacking_study.optimize(
        objective_fn, n_trials=config.get("automl", {}).get("stacking_trials", 20), timeout=config.get("automl", {}).get("timeout", 3600), n_jobs=1
    )  # Run stacking optimization

    completed = [
        t for t in stacking_study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]  # Get completed trials

    if not completed:  # If no stacking trials completed
        print(
            f"{BackgroundColors.YELLOW}AutoML stacking search: no successful trials.{Style.RESET_ALL}"
        )  # Output warning
        return (None, None)  # Return None tuple

    best_trial = stacking_study.best_trial  # Get best stacking trial
    best_config = {
        "meta_learner": best_trial.params.get("meta_learner"),  # Best meta-learner choice
        "stacking_cv_splits": best_trial.params.get("stacking_cv_splits"),  # Best CV splits
        "base_learners": [
            name for name in candidate_models.keys()
            if best_trial.params.get(f"use_{name.replace(' ', '_').replace('(', '').replace(')', '')}", False)
        ],  # Selected base learner names
        "base_learner_params": candidate_models,  # Parameters for each base learner
        "best_cv_f1": stacking_study.best_value,  # Best CV F1 score
    }  # Build best configuration dictionary

    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}AutoML Best Stacking Config:{Style.RESET_ALL}"
    )  # Output header
    print(
        f"{BackgroundColors.GREEN}  Meta-learner: {BackgroundColors.CYAN}{best_config['meta_learner']}{Style.RESET_ALL}"
    )  # Output meta-learner
    print(
        f"{BackgroundColors.GREEN}  Base learners: {BackgroundColors.CYAN}{best_config['base_learners']}{Style.RESET_ALL}"
    )  # Output base learners
    print(
        f"{BackgroundColors.GREEN}  CV splits: {BackgroundColors.CYAN}{best_config['stacking_cv_splits']}{Style.RESET_ALL}"
    )  # Output CV splits
    print(
        f"{BackgroundColors.GREEN}  Best CV F1: {BackgroundColors.CYAN}{truncate_value(best_config['best_cv_f1'])}{Style.RESET_ALL}"
    )  # Output best F1

    return (best_config, stacking_study)  # Return best config and study


def build_automl_stacking_model(best_config):
    """
    Builds a StackingClassifier from the best AutoML stacking configuration.

    :param best_config: Dictionary with best stacking configuration
    :return: Configured StackingClassifier instance
    """

    estimators = []  # Initialize estimators list

    for name in best_config["base_learners"]:  # Iterate over selected base learners
        params = best_config["base_learner_params"].get(name, {})  # Get model parameters
        model = create_model_from_params(name, params)  # Create model instance
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")  # Sanitize estimator name
        estimators.append((safe_name, model))  # Add to estimators list

    meta_learner_name = best_config["meta_learner"]  # Get meta-learner name

    if meta_learner_name == "Logistic Regression":  # Logistic Regression meta-learner
        meta_model = LogisticRegression(max_iter=1000, random_state=config.get("automl", {}).get("random_state", 42))  # Create LR
    elif meta_learner_name == "Random Forest":  # Random Forest meta-learner
        meta_model = RandomForestClassifier(n_estimators=50, random_state=config.get("automl", {}).get("random_state", 42), n_jobs=config.get("evaluation", {}).get("n_jobs", -1))  # Create RF
    else:  # Gradient Boosting meta-learner
        meta_model = GradientBoostingClassifier(random_state=config.get("automl", {}).get("random_state", 42))  # Create GB

    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        cv=StratifiedKFold(
            n_splits=best_config["stacking_cv_splits"], shuffle=True, random_state=config.get("automl", {}).get("random_state", 42)
        ),
        n_jobs=config.get("evaluation", {}).get("n_jobs", -1),
    )  # Create stacking classifier with optimal configuration

    return stacking_model  # Return configured stacking model


def evaluate_automl_model_on_test(model, model_name, X_train, y_train, X_test, y_test):
    """
    Trains and evaluates an AutoML-selected model on the held-out test set.

    :param model: Classifier instance to evaluate
    :param model_name: Name of the model for logging
    :param X_train: Training features array
    :param y_train: Training target labels
    :param X_test: Testing features array
    :param y_test: Testing target labels
    :return: Dictionary containing all evaluation metrics
    """

    start_time = time.time()  # Record start time

    model.fit(X_train, y_train)  # Train model on full training set
    y_pred = model.predict(X_test)  # Generate predictions on test set

    elapsed = time.time() - start_time  # Calculate elapsed training time

    acc = accuracy_score(y_test, y_pred)  # Calculate accuracy
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate weighted precision
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate weighted recall
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)  # Calculate weighted F1 score

    roc_auc = None  # Initialize ROC-AUC as None
    try:  # Try to compute ROC-AUC
        if hasattr(model, "predict_proba"):  # If model supports probability predictions
            y_proba = model.predict_proba(X_test)  # Get probability predictions
            if len(np.unique(y_test)) == 2:  # Binary classification
                roc_auc = roc_auc_score(y_test, y_proba[:, 1])  # Compute binary ROC-AUC
            else:  # Multi-class classification
                roc_auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")  # Compute multi-class ROC-AUC
    except Exception:  # If ROC-AUC computation fails
        roc_auc = None  # Keep as None

    if len(np.unique(y_test)) == 2:  # Binary classification metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()  # Get confusion matrix components
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # Calculate false positive rate
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # Calculate false negative rate
    else:  # Multi-class (simplified)
        fpr = 0.0  # Placeholder FPR
        fnr = 0.0  # Placeholder FNR

    print(
        f"{BackgroundColors.GREEN}AutoML {model_name} Test Results - Acc: {BackgroundColors.CYAN}{truncate_value(acc)}{BackgroundColors.GREEN}, F1: {BackgroundColors.CYAN}{truncate_value(f1)}{BackgroundColors.GREEN}, ROC-AUC: {BackgroundColors.CYAN}{truncate_value(roc_auc)}{BackgroundColors.GREEN}, Time: {BackgroundColors.CYAN}{int(round(elapsed))}s{Style.RESET_ALL}"
    )  # Output test results

    return {  # Build and return metrics dictionary
        "accuracy": acc,  # Accuracy value
        "precision": prec,  # Precision value
        "recall": rec,  # Recall value
        "f1_score": f1,  # F1 score value
        "roc_auc": roc_auc,  # ROC-AUC value
        "fpr": fpr,  # False positive rate
        "fnr": fnr,  # False negative rate
        "elapsed_time_s": int(round(elapsed)),  # Elapsed time in seconds
    }  # Return metrics dictionary


def export_automl_search_history(study, output_dir, study_name):
    """
    Exports the Optuna study trial history to a CSV file.

    :param study: Completed Optuna study object
    :param output_dir: Directory path for saving the export file
    :param study_name: Name prefix for the output file
    :return: Path to the exported CSV file
    """

    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    trials_data = []  # Initialize list for trial data

    for trial in study.trials:  # Iterate over all trials
        trial_entry = {  # Build entry for this trial
            "trial_number": trial.number,  # Trial index number
            "value": trial.value if trial.value is not None else None,  # Objective value (F1 score)
            "state": trial.state.name,  # Trial state (COMPLETE, PRUNED, FAIL)
            "duration_s": (
                trial.duration.total_seconds() if trial.duration else None
            ),  # Trial duration in seconds
        }  # Build basic trial entry
        trial_entry.update(trial.params)  # Add trial parameters to entry
        trials_data.append(trial_entry)  # Append to trials data list

    df = pd.DataFrame(trials_data)  # Convert trials data to DataFrame
    output_path = os.path.join(output_dir, f"{study_name}_search_history.csv")  # Build output file path
    df.to_csv(output_path, index=False)  # Save to CSV

    print(
        f"{BackgroundColors.GREEN}AutoML search history exported to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
    )  # Output export confirmation

    return output_path  # Return the output file path


def export_automl_best_config(best_model_name, best_params, test_metrics, stacking_config, output_dir, feature_names):
    """
    Exports the best AutoML configuration and metrics to a JSON file.

    :param best_model_name: Name of the best model found
    :param best_params: Best hyperparameters dictionary
    :param test_metrics: Test set evaluation metrics dictionary
    :param stacking_config: Best stacking configuration dictionary (or None)
    :param output_dir: Directory path for saving the export file
    :param feature_names: List of feature names used
    :return: Path to the exported JSON file
    """

    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    config = {  # Build configuration export dictionary
        "best_individual_model": {  # Best individual model section
            "model_name": best_model_name,  # Model name
            "hyperparameters": best_params,  # Model hyperparameters
            "test_metrics": {
                k: truncate_value(v) if isinstance(v, (int, float)) and v is not None else v
                for k, v in test_metrics.items()
            },  # Test metrics with truncation
        },
        "best_stacking_config": stacking_config,  # Stacking configuration (may be None)
        "automl_settings": {  # AutoML settings used
            "n_trials": config.get("automl", {}).get("n_trials", 50),  # Number of model search trials
            "stacking_trials": config.get("automl", {}).get("stacking_trials", 20),  # Number of stacking search trials
            "cv_folds": config.get("automl", {}).get("cv_folds", 5),  # Cross-validation folds
            "timeout_s": config.get("automl", {}).get("timeout", 3600),  # Timeout in seconds
            "random_state": config.get("automl", {}).get("random_state", 42),  # Random seed used
        },
        "feature_names": feature_names,  # Features used in training
        "n_features": len(feature_names),  # Number of features
    }  # Build complete config dictionary

    output_path = os.path.join(output_dir, config.get("automl", {}).get("results_filename", "AutoML_Results.csv").replace(".csv", "_best_config.json"))  # Build output path

    with open(output_path, "w", encoding="utf-8") as f:  # Open file for writing
        json.dump(config, f, indent=2, default=str)  # Write JSON with indentation

    print(
        f"{BackgroundColors.GREEN}AutoML best configuration exported to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
    )  # Output export confirmation

    return output_path  # Return the output file path


def export_automl_best_model(model, scaler, output_dir, model_name, feature_names):
    """
    Exports the best AutoML model and scaler to disk using joblib.

    :param model: Trained best model instance
    :param scaler: Fitted StandardScaler instance
    :param output_dir: Directory path for saving model files
    :param model_name: Name of the model for file naming
    :param feature_names: List of feature names for metadata
    :return: Tuple (model_path, scaler_path)
    """

    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    safe_name = re.sub(r'[\\/*?:"<>|() ]', "_", str(model_name))  # Sanitize model name for filename
    model_path = os.path.join(output_dir, f"AutoML_best_{safe_name}_model.joblib")  # Build model file path
    scaler_path = os.path.join(output_dir, f"AutoML_best_{safe_name}_scaler.joblib")  # Build scaler file path

    dump(model, model_path)  # Export model to disk

    if scaler is not None:  # If scaler is provided
        dump(scaler, scaler_path)  # Export scaler to disk

    meta_path = os.path.join(output_dir, f"AutoML_best_{safe_name}_meta.json")  # Build metadata file path
    meta = {  # Build metadata dictionary
        "model_name": model_name,  # Model name
        "features": feature_names,  # Feature names used
        "n_features": len(feature_names),  # Number of features
    }  # Metadata content

    with open(meta_path, "w", encoding="utf-8") as f:  # Open metadata file
        json.dump(meta, f, indent=2)  # Write metadata JSON

    print(
        f"{BackgroundColors.GREEN}AutoML best model exported to: {BackgroundColors.CYAN}{model_path}{Style.RESET_ALL}"
    )  # Output export confirmation

    return (model_path, scaler_path)  # Return file paths


def build_automl_results_list(best_model_name, best_params, individual_metrics, stacking_metrics, stacking_config, file_path, feature_names, n_train, n_test):
    """
    Builds the results list for AutoML CSV export matching existing results format.

    :param best_model_name: Name of the best individual model
    :param best_params: Best hyperparameters for the individual model
    :param individual_metrics: Metrics from individual model test evaluation
    :param stacking_metrics: Metrics from stacking model test evaluation (or None)
    :param stacking_config: Best stacking configuration (or None)
    :param file_path: Path to the dataset file
    :param feature_names: List of feature names
    :param n_train: Number of training samples
    :param n_test: Number of test samples
    :return: List of result dictionaries for CSV export
    """

    results = []  # Initialize results list

    individual_entry = {  # Build individual model result entry
        "model": best_model_name,  # Model class name
        "dataset": os.path.relpath(file_path),  # Dataset relative path
        "feature_set": "AutoML",  # Feature set label
        "classifier_type": "AutoML_Individual",  # Classifier type
        "model_name": f"AutoML_{best_model_name}",  # Prefixed model name
        "data_source": "Original",  # Data source label
        "experiment_id": None,  # No experiment ID for standalone AutoML runs
        "experiment_mode": "original_only",  # AutoML runs on original data only
        "augmentation_ratio": None,  # No augmentation ratio for AutoML
        "n_features": len(feature_names),  # Number of features
        "n_samples_train": n_train,  # Training sample count
        "n_samples_test": n_test,  # Test sample count
        "accuracy": truncate_value(individual_metrics["accuracy"]),  # Accuracy
        "precision": truncate_value(individual_metrics["precision"]),  # Precision
        "recall": truncate_value(individual_metrics["recall"]),  # Recall
        "f1_score": truncate_value(individual_metrics["f1_score"]),  # F1 score
        "fpr": truncate_value(individual_metrics["fpr"]),  # False positive rate
        "fnr": truncate_value(individual_metrics["fnr"]),  # False negative rate
        "elapsed_time_s": individual_metrics["elapsed_time_s"],  # Elapsed time
        "cv_method": f"Optuna({config.get("automl", {}).get("n_trials", 50)} trials, {config.get("automl", {}).get("cv_folds", 5)}-fold CV)",  # CV method description
        "top_features": json.dumps(feature_names),  # Feature names as JSON
        "rfe_ranking": None,  # No RFE ranking for AutoML
        "hyperparameters": json.dumps(best_params),  # Hyperparameters as JSON
        "features_list": feature_names,  # Feature names list
    }  # Individual model result entry
    results.append(individual_entry)  # Add to results list

    if stacking_metrics is not None and stacking_config is not None:  # If stacking results are available
        stacking_entry = {  # Build stacking result entry
            "model": "StackingClassifier",  # Model class name
            "dataset": os.path.relpath(file_path),  # Dataset relative path
            "feature_set": "AutoML",  # Feature set label
            "classifier_type": "AutoML_Stacking",  # Classifier type
            "model_name": "AutoML_StackingClassifier",  # Prefixed model name
            "data_source": "Original",  # Data source label
            "experiment_id": None,  # No experiment ID for standalone AutoML runs
            "experiment_mode": "original_only",  # AutoML runs on original data only
            "augmentation_ratio": None,  # No augmentation ratio for AutoML
            "n_features": len(feature_names),  # Number of features
            "n_samples_train": n_train,  # Training sample count
            "n_samples_test": n_test,  # Test sample count
            "accuracy": truncate_value(stacking_metrics["accuracy"]),  # Accuracy
            "precision": truncate_value(stacking_metrics["precision"]),  # Precision
            "recall": truncate_value(stacking_metrics["recall"]),  # Recall
            "f1_score": truncate_value(stacking_metrics["f1_score"]),  # F1 score
            "fpr": truncate_value(stacking_metrics["fpr"]),  # False positive rate
            "fnr": truncate_value(stacking_metrics["fnr"]),  # False negative rate
            "elapsed_time_s": stacking_metrics["elapsed_time_s"],  # Elapsed time
            "cv_method": f"Optuna({config.get("automl", {}).get("stacking_trials", 20)} trials, {config.get("automl", {}).get("cv_folds", 5)}-fold CV)",  # CV method description
            "top_features": json.dumps(feature_names),  # Feature names as JSON
            "rfe_ranking": None,  # No RFE ranking for AutoML
            "hyperparameters": json.dumps(stacking_config, default=str),  # Stacking config as JSON
            "features_list": feature_names,  # Feature names list
        }  # Stacking result entry
        results.append(stacking_entry)  # Add stacking to results list

    return results  # Return results list


def save_automl_results(csv_path, results_list):
    """
    Saves AutoML results to a dedicated CSV file in the Feature_Analysis/AutoML directory.

    :param csv_path: Path to the original dataset CSV file
    :param results_list: List of result dictionaries to save
    :return: None
    """

    if not results_list:  # If no results to save
        return  # Exit early

    file_path_obj = Path(csv_path)  # Create Path object for dataset file
    automl_dir = file_path_obj.parent / "Feature_Analysis" / "AutoML"  # Build AutoML output directory
    os.makedirs(automl_dir, exist_ok=True)  # Ensure directory exists
    output_path = automl_dir / config.get("automl", {}).get("results_filename", "AutoML_Results.csv")  # Build output file path

    df = pd.DataFrame(results_list)  # Convert results to DataFrame
    column_order = list(config.get("stacking", {}).get("results_csv_columns", []))  # Use canonical column ordering
    existing_columns = [col for col in column_order if col in df.columns]  # Filter to existing columns
    df = df[existing_columns + [c for c in df.columns if c not in existing_columns]]  # Reorder columns

    df = add_hardware_column(df, existing_columns)  # Add hardware specifications column

    df.to_csv(str(output_path), index=False, encoding="utf-8")  # Save to CSV file

    print(
        f"\n{BackgroundColors.GREEN}AutoML results saved to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}"
    )  # Output save confirmation


def run_automl_pipeline(file, df, feature_names, data_source_label="Original"):
    """
    Runs the complete AutoML pipeline: model search, stacking optimization, evaluation, and export.

    :param file: Path to the dataset file being processed
    :param df: Preprocessed DataFrame with features and target
    :param feature_names: List of feature column names
    :param data_source_label: Label identifying the data source
    :return: Dictionary containing AutoML results, or None on failure
    """

    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*80}{Style.RESET_ALL}"
    )  # Print separator
    print(
        f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}AutoML Pipeline - {BackgroundColors.CYAN}{os.path.basename(file)}{Style.RESET_ALL}"
    )  # Print pipeline header
    print(
        f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*80}{Style.RESET_ALL}\n"
    )  # Print separator

    automl_start = time.time()  # Record pipeline start time

    X_full = df.select_dtypes(include=np.number).iloc[:, :-1]  # Extract numeric features
    y = df.iloc[:, -1]  # Extract target column

    if len(np.unique(y)) < 2:  # Check for at least 2 classes
        print(
            f"{BackgroundColors.RED}AutoML: Target has only one class. Skipping.{Style.RESET_ALL}"
        )  # Output error
        return None  # Return None

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = scale_and_split(X_full, y)  # Scale and split data

    y_train_arr = np.asarray(y_train)  # Convert training target to numpy array
    y_test_arr = np.asarray(y_test)  # Convert test target to numpy array

    send_telegram_message(TELEGRAM_BOT, f"Starting AutoML pipeline for {os.path.basename(file)}")  # Notify via Telegram

    best_model_name, best_params, model_study = run_automl_model_search(
        X_train_scaled, y_train_arr, file
    )  # Phase 1: Run model search

    if best_model_name is None:  # If model search failed
        print(
            f"{BackgroundColors.RED}AutoML pipeline aborted: model search failed.{Style.RESET_ALL}"
        )  # Output failure message
        return None  # Return None

    best_individual_model = create_model_from_params(best_model_name, best_params)  # Create best individual model

    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Evaluating AutoML best individual model on test set...{Style.RESET_ALL}"
    )  # Output evaluation message

    individual_metrics = evaluate_automl_model_on_test(
        best_individual_model, best_model_name, X_train_scaled, y_train_arr, X_test_scaled, y_test_arr
    )  # Evaluate best individual model on test set

    stacking_config = None  # Initialize stacking config as None
    stacking_metrics = None  # Initialize stacking metrics as None
    stacking_study = None  # Initialize stacking study as None

    if config.get("automl", {}).get("stacking_trials", 20) > 0:  # If stacking search is enabled
        stacking_config, stacking_study = run_automl_stacking_search(
            X_train_scaled, y_train_arr, model_study, file
        )  # Phase 2: Run stacking search

        if stacking_config is not None:  # If stacking search succeeded
            best_stacking_model = build_automl_stacking_model(stacking_config)  # Build best stacking model

            print(
                f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Evaluating AutoML best stacking model on test set...{Style.RESET_ALL}"
            )  # Output evaluation message

            stacking_metrics = evaluate_automl_model_on_test(
                best_stacking_model, "AutoML_Stacking", X_train_scaled, y_train_arr, X_test_scaled, y_test_arr
            )  # Evaluate stacking model on test set

    file_path_obj = Path(file)  # Create Path object for file
    automl_output_dir = str(file_path_obj.parent / "Feature_Analysis" / "AutoML")  # Build AutoML output directory

    export_automl_search_history(model_study, automl_output_dir, "model_search")  # Export model search history

    if stacking_study is not None:  # If stacking study exists
        export_automl_search_history(stacking_study, automl_output_dir, "stacking_search")  # Export stacking search history

    export_automl_best_config(
        best_model_name, best_params, individual_metrics, stacking_config, automl_output_dir, feature_names
    )  # Export best configuration

    export_automl_best_model(
        best_individual_model, scaler, automl_output_dir, best_model_name, feature_names
    )  # Export best individual model

    if stacking_config is not None and stacking_metrics is not None:  # If stacking was successful
        best_stacking_model_final = build_automl_stacking_model(stacking_config)  # Rebuild stacking model for export
        best_stacking_model_final.fit(X_train_scaled, y_train_arr)  # Fit stacking model on full training data
        export_automl_best_model(
            best_stacking_model_final, scaler, automl_output_dir, "AutoML_Stacking", feature_names
        )  # Export best stacking model

    results_list = build_automl_results_list(
        best_model_name, best_params, individual_metrics, stacking_metrics, stacking_config,
        file, feature_names, len(y_train), len(y_test)
    )  # Build results list for CSV

    save_automl_results(file, results_list)  # Save AutoML results to CSV

    automl_elapsed = time.time() - automl_start  # Calculate total AutoML pipeline time

    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}AutoML pipeline completed in {BackgroundColors.CYAN}{calculate_execution_time(0, automl_elapsed)}{Style.RESET_ALL}"
    )  # Output completion message

    send_telegram_message(
        TELEGRAM_BOT, f"AutoML pipeline completed for {os.path.basename(file)} in {calculate_execution_time(0, automl_elapsed)}. Best model: {best_model_name} (F1: {truncate_value(individual_metrics['f1_score'])})"
    )  # Send Telegram notification

    return {  # Return AutoML results summary
        "best_model_name": best_model_name,  # Best model name
        "best_params": best_params,  # Best parameters
        "individual_metrics": individual_metrics,  # Individual model metrics
        "stacking_config": stacking_config,  # Stacking configuration
        "stacking_metrics": stacking_metrics,  # Stacking metrics
    }  # Return results dictionary


def evaluate_on_dataset(
    file,
    df,
    feature_names,
    ga_selected_features,
    pca_n_components,
    rfe_selected_features,
    base_models,
    data_source_label="Original",
    hyperparams_map=None,
    experiment_id=None,
    experiment_mode="original_only",
    augmentation_ratio=None,
):
    """
    Evaluate classifiers on a single dataset (original or augmented).

    :param file: Path to the dataset file
    :param df: DataFrame with the dataset
    :param feature_names: List of feature column names
    :param ga_selected_features: GA selected features
    :param pca_n_components: Number of PCA components
    :param rfe_selected_features: RFE selected features
    :param base_models: Dictionary of base models to evaluate
    :param data_source_label: Label for data source ("Original", "Original+Augmented@50%", etc.)
    :param hyperparams_map: Dictionary mapping model names to hyperparameter dicts
    :param experiment_id: Unique experiment identifier for traceability
    :param experiment_mode: Experiment mode string ('original_only' or 'original_plus_augmented')
    :param augmentation_ratio: Augmentation ratio float (e.g., 0.50) or None for original-only
    :return: Dictionary mapping (feature_set, model_name) to results
    """

    # Sanitize GA and RFE feature names to match the sanitized feature_names in the DataFrame
    if ga_selected_features:
        ga_selected_features = sanitize_feature_names(ga_selected_features)
    if rfe_selected_features:
        rfe_selected_features = sanitize_feature_names(rfe_selected_features)

    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*80}{Style.RESET_ALL}"
    )
    print(
        f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Evaluating on: {BackgroundColors.CYAN}{data_source_label} Data{Style.RESET_ALL}"
    )
    print(
        f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*80}{Style.RESET_ALL}\n"
    )

    X_full = df.select_dtypes(include=np.number).iloc[:, :-1]  # Features (numeric only)
    y = df.iloc[:, -1]  # Target

    if len(np.unique(y)) < 2:  # Verify if there is more than one class
        print(
            f"{BackgroundColors.RED}Target column has only one class. Cannot perform classification. Skipping.{Style.RESET_ALL}"
        )  # Output the error message
        return {}  # Return empty dictionary

    X_train_scaled, X_test_scaled, y_train, y_test, scaler = scale_and_split(
        X_full, y
    )  # Scale and split the data

    estimators = [
        (name, model) for name, model in base_models.items() if name != "SVM"
    ]  # Define estimators (excluding SVM)

    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=config.get("evaluation", {}).get("n_jobs", -1)),
        cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=42),
        n_jobs=config.get("evaluation", {}).get("n_jobs", -1),
    )  # Define the Stacking Classifier model

    X_train_pca, X_test_pca = apply_pca_transformation(
        X_train_scaled, X_test_scaled, pca_n_components, file
    )  # Apply PCA transformation if applicable

    # Get feature subsets with actual selected feature names
    X_train_ga, ga_actual_features = get_feature_subset(X_train_scaled, ga_selected_features, feature_names)
    X_test_ga, _ = get_feature_subset(X_test_scaled, ga_selected_features, feature_names)
    
    X_train_rfe, rfe_actual_features = get_feature_subset(X_train_scaled, rfe_selected_features, feature_names)
    X_test_rfe, _ = get_feature_subset(X_test_scaled, rfe_selected_features, feature_names)

    feature_sets = {  # Dictionary of feature sets to evaluate
        "Full Features": (X_train_scaled, X_test_scaled, feature_names),  # All features with names
        "GA Features": (X_train_ga, X_test_ga, ga_actual_features),  # GA subset with actual names
        "PCA Components": (
            (X_train_pca, X_test_pca, None) if X_train_pca is not None else None
        ),  # PCA components (only if PCA was applied)
        "RFE Features": (X_train_rfe, X_test_rfe, rfe_actual_features),  # RFE subset with actual names
    }

    feature_sets = {
        k: v for k, v in feature_sets.items() if v is not None
    }  # Remove any None entries (e.g., PCA if not applied)
    feature_sets = dict(sorted(feature_sets.items()))  # Sort the feature sets by name

    individual_models = {
        k: v for k, v in base_models.items()
    }  # Use the base models (with hyperparameters applied) for individual evaluation
    total_steps = len(feature_sets) * (
        len(individual_models) + 1
    )  # Total steps: models + stacking per feature set
    progress_bar = tqdm(total=total_steps, desc=f"{data_source_label} Data", file=sys.stdout)  # Progress bar for all evaluations

    all_results = {}  # Dictionary to store results: (feature_set, model_name) -> result_entry

    current_combination = 1  # Counter for combination index

    for idx, (name, (X_train_subset, X_test_subset, subset_feature_names_list)) in enumerate(feature_sets.items(), start=1):
        if X_train_subset.shape[1] == 0:  # Verify if the subset is empty
            print(
                f"{BackgroundColors.YELLOW}Warning: Skipping {name}. No features selected.{Style.RESET_ALL}"
            )  # Output warning
            progress_bar.update(len(individual_models) + 1)  # Skip all steps for this feature set
            continue  # Skip to the next set

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Evaluating models on: {BackgroundColors.CYAN}{name} ({X_train_subset.shape[1]} features){Style.RESET_ALL}"
        )  # Output evaluation status

        if name == "PCA Components":  # If the feature set is PCA Components
            subset_feature_names = [
                f"PC{i+1}" for i in range(X_train_subset.shape[1])
            ]  # Generate PCA component names
        else:  # For other feature sets
            subset_feature_names = (
                subset_feature_names_list if subset_feature_names_list else [f"feature_{i}" for i in range(X_train_subset.shape[1])]
            )  # Use actual feature names or generate generic ones

        X_train_df = pd.DataFrame(
            X_train_subset, columns=subset_feature_names
        )  # Convert training features to DataFrame
        X_test_df = pd.DataFrame(
            X_test_subset, columns=subset_feature_names
        )  # Convert test features to DataFrame

        progress_bar.set_description(
            f"{data_source_label} - {name} (Individual)"
        )  # Update progress bar description
        
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=config.get("evaluation", {}).get("threads_limit", 2)
        ) as executor:  # Create a thread pool executor for parallel evaluation
            future_to_model = {}  # Dictionary to map futures to model names
            for model_name, model in individual_models.items():  # Iterate over each individual model
                send_telegram_message(TELEGRAM_BOT, f"Starting combination {current_combination}/{total_steps}: {name} - {model_name}")
                future = executor.submit(
                    evaluate_individual_classifier,
                    model,
                    model_name,
                    X_train_df.values,
                    y_train,
                    X_test_df.values,
                    y_test,
                    file,
                    scaler,
                    subset_feature_names,
                    name,
                )  # Submit evaluation task to thread pool (using .values for numpy arrays)
                # Store both the model name and its class name for richer metadata
                future_to_model[future] = (model_name, model.__class__.__name__, current_combination)
                current_combination += 1
            
            for future in concurrent.futures.as_completed(future_to_model):  # As each evaluation completes
                model_name, model_class, comb_idx = future_to_model[future]  # Get metadata from mapping
                metrics = future.result()  # Get the metrics from the completed future
                # Flatten metrics into named fields and include extra metadata similar to rfe.py
                acc, prec, rec, f1, fpr, fnr, elapsed = metrics
                result_entry = {
                    "model": model_class,
                    "dataset": os.path.relpath(file),
                    "feature_set": name,
                    "classifier_type": "Individual",
                    "model_name": model_name,
                    "data_source": data_source_label,
                    "experiment_id": experiment_id,
                    "experiment_mode": experiment_mode,
                    "augmentation_ratio": augmentation_ratio,
                    "n_features": X_train_subset.shape[1],
                    "n_samples_train": len(y_train),
                    "n_samples_test": len(y_test),
                    "accuracy": truncate_value(acc),
                    "precision": truncate_value(prec),
                    "recall": truncate_value(rec),
                    "f1_score": truncate_value(f1),
                    "fpr": truncate_value(fpr),
                    "fnr": truncate_value(fnr),
                    "elapsed_time_s": int(round(elapsed)),
                    "cv_method": f"StratifiedKFold(n_splits=10)",
                    "top_features": json.dumps(subset_feature_names),
                    "rfe_ranking": None,
                    "hyperparameters": json.dumps(hyperparams_map.get(model_name)) if hyperparams_map and hyperparams_map.get(model_name) is not None else None,
                    "features_list": subset_feature_names,
                }  # Prepare result entry
                all_results[(name, model_name)] = result_entry  # Store result with key
                send_telegram_message(TELEGRAM_BOT, f"Finished combination {comb_idx}/{total_steps}: {name} - {model_name} with F1: {truncate_value(f1)} in {calculate_execution_time(0, elapsed)}")
                print(
                    f"    {BackgroundColors.GREEN}{model_name} Accuracy: {BackgroundColors.CYAN}{truncate_value(metrics[0])}{Style.RESET_ALL}"
                )  # Output accuracy
                progress_bar.update(1)  # Update progress after each model

        print(
            f"  {BackgroundColors.GREEN}Training {BackgroundColors.CYAN}Stacking Classifier{BackgroundColors.GREEN}...{Style.RESET_ALL}"
        )
        progress_bar.set_description(
            f"{data_source_label} - {name} (Stacking)"
        )  # Update progress bar description for stacking

        send_telegram_message(TELEGRAM_BOT, f"Starting combination {current_combination}/{total_steps}: {name} - StackingClassifier")

        stacking_metrics = evaluate_stacking_classifier(
            stacking_model, X_train_df, y_train, X_test_df, y_test
        )  # Evaluate stacking model with DataFrames

        # Export stacking model and scaler
        try:
            dataset_name = os.path.basename(os.path.dirname(file))
            export_model_and_scaler(stacking_model, scaler, dataset_name, "StackingClassifier", subset_feature_names, best_params=None, feature_set=name, dataset_csv_path=file)
        except Exception:
            pass

        # Flatten stacking metrics and include richer metadata
        s_acc, s_prec, s_rec, s_f1, s_fpr, s_fnr, s_elapsed = stacking_metrics
        stacking_result_entry = {
            "model": stacking_model.__class__.__name__,
            "dataset": os.path.relpath(file),
            "feature_set": name,
            "classifier_type": "Stacking",
            "model_name": "StackingClassifier",
            "data_source": data_source_label,
            "experiment_id": experiment_id,
            "experiment_mode": experiment_mode,
            "augmentation_ratio": augmentation_ratio,
            "n_features": X_train_subset.shape[1],
            "n_samples_train": len(y_train),
            "n_samples_test": len(y_test),
            "accuracy": truncate_value(s_acc),
            "precision": truncate_value(s_prec),
            "recall": truncate_value(s_rec),
            "f1_score": truncate_value(s_f1),
            "fpr": truncate_value(s_fpr),
            "fnr": truncate_value(s_fnr),
            "elapsed_time_s": int(round(s_elapsed)),
            "cv_method": f"StratifiedKFold(n_splits=10)",
            "top_features": json.dumps(subset_feature_names),
            "rfe_ranking": None,
            "hyperparameters": None,
            "features_list": subset_feature_names,
        }  # Prepare stacking result entry
        all_results[(name, "StackingClassifier")] = stacking_result_entry  # Store result with key
        send_telegram_message(TELEGRAM_BOT, f"Finished combination {current_combination}/{total_steps}: {name} - StackingClassifier with F1: {truncate_value(s_f1)} in {calculate_execution_time(0, s_elapsed)}")
        print(
            f"    {BackgroundColors.GREEN}Stacking Accuracy: {BackgroundColors.CYAN}{truncate_value(stacking_metrics[0])}{Style.RESET_ALL}"
        )  # Output accuracy
        progress_bar.update(1)  # Update progress after stacking
        current_combination += 1

    progress_bar.close()  # Close progress bar
    return all_results  # Return dictionary of results


def determine_files_to_process(csv_file, input_path, config=None):
    """
    Determines which files to process based on CLI override or directory scan.

    :param csv_file: Optional CSV file path from CLI argument
    :param input_path: Directory path to search for CSV files
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: List of file paths to process
    """
    
    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG

    verbose_output(
        f"{BackgroundColors.GREEN}Determining files to process from path: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}",
        config=config
    )  # Output the verbose message

    if csv_file:  # If a specific CSV file is provided via CLI
        try:  # Attempt to validate CSV file path
            abs_csv = os.path.abspath(csv_file)  # Get absolute path of CSV file
            abs_input = os.path.abspath(input_path)  # Get absolute path of input directory
            if abs_csv.startswith(abs_input):  # If CSV file belongs to this input path
                return [csv_file]  # Return list with single CSV file
            else:  # CSV override does not belong to this path
                return []  # Return empty list to skip this path
        except Exception:  # If validation fails
            return []  # Return empty list on error
    else:  # No CLI override, scan directory for CSV files
        return get_files_to_process(input_path, file_extension=".csv", config=config)  # Get list of CSV files to process


def combine_dataset_if_needed(files_to_process, config=None):
    """
    Combines multiple dataset files into one if PROCESS_ENTIRE_DATASET is enabled.

    :param files_to_process: List of file paths to process
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (combined_df, combined_file_for_features, updated_files_list) or (None, None, files_to_process)
    """
    
    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG

    verbose_output(
        f"{BackgroundColors.GREEN}Checking if dataset combination is needed...{Style.RESET_ALL}",
        config=config
    )  # Output the verbose message
    
    process_entire_dataset = config.get("execution", {}).get("process_entire_dataset", False)  # Get process entire dataset flag from config

    if process_entire_dataset and len(files_to_process) > 1:  # If combining is enabled and multiple files exist
        verbose_output(
            f"{BackgroundColors.GREEN}Attempting to combine {BackgroundColors.CYAN}{len(files_to_process)}{BackgroundColors.GREEN} dataset files...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message
        result = combine_dataset_files(files_to_process, config=config)  # Attempt to combine all files
        if result is not None:  # If combination was successful
            combined_df, combined_target_col = result  # Unpack the combined dataframe and target column
            combined_file_for_features = files_to_process[0]  # Use first file for feature selection metadata
            files_to_process = ["combined"]  # Replace file list with single "combined" entry
            return (combined_df, combined_file_for_features, files_to_process)  # Return combined data and updated file list
        else:  # If combination failed
            print(
                f"{BackgroundColors.YELLOW}Warning: Could not combine dataset files. Processing individually.{Style.RESET_ALL}"
            )  # Output warning message

    return (None, None, files_to_process)  # Return original file list unchanged


def load_and_preprocess_dataset(file, combined_df, config=None):
    """
    Loads and preprocesses a dataset file or uses combined dataframe.

    :param file: File path to load or "combined" keyword
    :param combined_df: Pre-combined dataframe (used if file == "combined")
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (df_cleaned, feature_names) or (None, None) if loading/preprocessing fails
    """
    
    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG

    verbose_output(
        f"{BackgroundColors.GREEN}Loading and preprocessing dataset: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}",
        config=config
    )  # Output the verbose message

    if file == "combined":  # If using combined dataset
        df_original = combined_df  # Use the pre-combined dataframe
    else:  # Otherwise load from file
        df_original = load_dataset(file, config=config)  # Load the original dataset

    if df_original is None:  # If the dataset failed to load
        verbose_output(
            f"{BackgroundColors.RED}Failed to load dataset from: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}",
            config=config
        )  # Output the failure message
        return (None, None)  # Return None tuple
    
    remove_zero_variance = config.get("dataset", {}).get("remove_zero_variance", True)  # Get remove zero variance flag from config

    df_cleaned = preprocess_dataframe(df_original, remove_zero_variance=remove_zero_variance, config=config)  # Preprocess the DataFrame

    if df_cleaned is None or df_cleaned.empty:  # If the DataFrame is None or empty after preprocessing
        print(
            f"{BackgroundColors.RED}Dataset {BackgroundColors.CYAN}{file}{BackgroundColors.RED} empty after preprocessing. Skipping.{Style.RESET_ALL}"
        )  # Output error message
        return (None, None)  # Return None tuple

    feature_names = df_cleaned.select_dtypes(include=np.number).iloc[:, :-1].columns.tolist()  # Get numeric feature names excluding target

    return (df_cleaned, feature_names)  # Return cleaned dataframe and feature names


def prepare_models_with_hyperparameters(file_path, config=None):
    """
    Prepares base models and applies hyperparameter optimization results if available.

    :param file_path: Path to the dataset file for loading hyperparameters
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: Tuple (base_models, hp_params_map)
    """
    
    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG

    verbose_output(
        f"{BackgroundColors.GREEN}Preparing models with hyperparameters for: {BackgroundColors.CYAN}{file_path}{Style.RESET_ALL}",
        config=config
    )  # Output the verbose message

    base_models = get_models(config=config)  # Get the base models with default parameters

    hp_params_map = {}  # Initialize empty hyperparameters mapping
    hp_results_raw = extract_hyperparameter_optimization_results(file_path, config=config)  # Extract hyperparameter optimization results

    if hp_results_raw:  # If results were found, extract the params mapping and apply
        hp_params_map = {
            k: (v.get("best_params") if isinstance(v, dict) else v) for k, v in hp_results_raw.items()
        }  # Extract only the best_params mapping
        base_models = apply_hyperparameters_to_models(hp_params_map, base_models, config=config)  # Apply hyperparameters to base models
        verbose_output(
            f"{BackgroundColors.GREEN}Applied hyperparameters from optimization results{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

    return (base_models, hp_params_map)  # Return models and hyperparameters mapping


def extract_metrics_from_result(result):
    """
    Extracts metrics from a result dictionary into a list.

    :param result: Result dictionary containing metric keys
    :return: List of [accuracy, precision, recall, f1_score, fpr, fnr, elapsed_time_s]
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Extracting metrics from result dictionary...{Style.RESET_ALL}"
    )  # Output the verbose message

    return [
        result.get("accuracy", 0),  # Get accuracy or default to 0
        result.get("precision", 0),  # Get precision or default to 0
        result.get("recall", 0),  # Get recall or default to 0
        result.get("f1_score", 0),  # Get F1 score or default to 0
        result.get("fpr", 0),  # Get false positive rate or default to 0
        result.get("fnr", 0),  # Get false negative rate or default to 0
        result.get("elapsed_time_s", 0),  # Get elapsed time or default to 0
    ]  # Return list of metric values


def calculate_all_improvements(orig_metrics, merged_metrics):
    """
    Calculates improvement percentages for all metrics comparing original vs merged data.

    :param orig_metrics: List of original metrics [accuracy, precision, recall, f1, fpr, fnr, time]
    :param merged_metrics: List of merged metrics [accuracy, precision, recall, f1, fpr, fnr, time]
    :return: Dictionary of improvement percentages for each metric
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Calculating metric improvements...{Style.RESET_ALL}"
    )  # Output the verbose message

    return {
        "accuracy": calculate_metric_improvement(orig_metrics[0], merged_metrics[0]),  # Calculate accuracy improvement
        "precision": calculate_metric_improvement(orig_metrics[1], merged_metrics[1]),  # Calculate precision improvement
        "recall": calculate_metric_improvement(orig_metrics[2], merged_metrics[2]),  # Calculate recall improvement
        "f1_score": calculate_metric_improvement(orig_metrics[3], merged_metrics[3]),  # Calculate F1 score improvement
        "fpr": calculate_metric_improvement(orig_metrics[4], merged_metrics[4]),  # Calculate FPR change (lower is better)
        "fnr": calculate_metric_improvement(orig_metrics[5], merged_metrics[5]),  # Calculate FNR change (lower is better)
        "training_time": calculate_metric_improvement(orig_metrics[6], merged_metrics[6]),  # Calculate time change (lower is better)
    }  # Return dictionary of improvements


def print_model_comparison(feature_set, model_name, orig_metrics, aug_metrics, merged_metrics, improvements):
    """
    Prints detailed comparison of metrics for a single model across data sources.

    :param feature_set: Name of the feature set used
    :param model_name: Name of the model
    :param orig_metrics: List of original data metrics
    :param aug_metrics: List of augmented data metrics
    :param merged_metrics: List of merged data metrics
    :param improvements: Dictionary of improvement percentages
    :return: None
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Printing comparison for model: {BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN}, feature set: {BackgroundColors.CYAN}{feature_set}{Style.RESET_ALL}"
    )  # Output the verbose message

    print(
        f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Feature Set: {BackgroundColors.CYAN}{feature_set}{BackgroundColors.GREEN} | Model: {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}"
    )  # Print header with feature set and model name

    print(f"  {BackgroundColors.YELLOW}Accuracy:{Style.RESET_ALL}")  # Print accuracy label
    print(
        f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {truncate_value(orig_metrics[0])} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {truncate_value(aug_metrics[0])} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {truncate_value(merged_metrics[0])} | {BackgroundColors.CYAN}Improvement: {improvements['accuracy']:+.2f}%{Style.RESET_ALL}"
    )  # Print accuracy comparison

    print(f"  {BackgroundColors.YELLOW}Precision:{Style.RESET_ALL}")  # Print precision label
    print(
        f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {truncate_value(orig_metrics[1])} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {truncate_value(aug_metrics[1])} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {truncate_value(merged_metrics[1])} | {BackgroundColors.CYAN}Improvement: {improvements['precision']:+.2f}%{Style.RESET_ALL}"
    )  # Print precision comparison

    print(f"  {BackgroundColors.YELLOW}Recall:{Style.RESET_ALL}")  # Print recall label
    print(
        f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {truncate_value(orig_metrics[2])} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {truncate_value(aug_metrics[2])} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {truncate_value(merged_metrics[2])} | {BackgroundColors.CYAN}Improvement: {improvements['recall']:+.2f}%{Style.RESET_ALL}"
    )  # Print recall comparison

    print(f"  {BackgroundColors.YELLOW}F1-Score:{Style.RESET_ALL}")  # Print F1 score label
    print(
        f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {truncate_value(orig_metrics[3])} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {truncate_value(aug_metrics[3])} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {truncate_value(merged_metrics[3])} | {BackgroundColors.CYAN}Improvement: {improvements['f1_score']:+.2f}%{Style.RESET_ALL}"
    )  # Print F1 score comparison

    print(f"  {BackgroundColors.YELLOW}FPR (lower is better):{Style.RESET_ALL}")  # Print FPR label
    print(
        f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {truncate_value(orig_metrics[4])} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {truncate_value(aug_metrics[4])} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {truncate_value(merged_metrics[4])} | {BackgroundColors.CYAN}Change: {improvements['fpr']:+.2f}%{Style.RESET_ALL}"
    )  # Print FPR comparison

    print(f"  {BackgroundColors.YELLOW}FNR (lower is better):{Style.RESET_ALL}")  # Print FNR label
    print(
        f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {truncate_value(orig_metrics[5])} | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {truncate_value(aug_metrics[5])} | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {truncate_value(merged_metrics[5])} | {BackgroundColors.CYAN}Change: {improvements['fnr']:+.2f}%{Style.RESET_ALL}"
    )  # Print FNR comparison

    print(f"  {BackgroundColors.YELLOW}Training Time (seconds, lower is better):{Style.RESET_ALL}")  # Print training time label
    print(
        f"    {BackgroundColors.GREEN}Original:{BackgroundColors.CYAN} {orig_metrics[6]:.2f}s | {BackgroundColors.YELLOW}Augmented:{BackgroundColors.CYAN} {aug_metrics[6]:.2f}s | {BackgroundColors.BOLD}Original+Augmented:{BackgroundColors.CYAN} {merged_metrics[6]:.2f}s | {BackgroundColors.CYAN}Change: {improvements['training_time']:+.2f}%{Style.RESET_ALL}\n"
    )  # Print training time comparison


def build_comparison_result_entry(orig_result, feature_set, classifier_type, model_name, data_source, metrics, improvements, n_features_override=None, n_samples_train_override=None, n_samples_test_override=None, experiment_id=None, experiment_mode="original_only", augmentation_ratio=None):
    """
    Builds a single comparison result entry for CSV export.

    :param orig_result: Original result dictionary for base metadata
    :param feature_set: Name of the feature set
    :param classifier_type: Type of classifier (e.g., 'Individual' or 'Stacking')
    :param model_name: Name of the model
    :param data_source: Data source label (e.g., 'Original', 'Original+Augmented@50%')
    :param metrics: List of metrics [accuracy, precision, recall, f1, fpr, fnr, time]
    :param improvements: Dictionary of improvement percentages
    :param n_features_override: Override for n_features (optional)
    :param n_samples_train_override: Override for n_samples_train (optional)
    :param n_samples_test_override: Override for n_samples_test (optional)
    :param experiment_id: Unique experiment identifier for traceability
    :param experiment_mode: Experiment mode string ('original_only' or 'original_plus_augmented')
    :param augmentation_ratio: Augmentation ratio float or None
    :return: Dictionary containing comparison result entry
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Building comparison result entry for: {BackgroundColors.CYAN}{model_name}{BackgroundColors.GREEN}, data source: {BackgroundColors.CYAN}{data_source}{Style.RESET_ALL}"
    )  # Output the verbose message

    return {
        "dataset": orig_result["dataset"],  # Dataset name from original result
        "feature_set": feature_set,  # Feature set name
        "classifier_type": classifier_type,  # Classifier type
        "model_name": model_name,  # Model name
        "data_source": data_source,  # Data source label
        "experiment_id": experiment_id,  # Unique experiment identifier for traceability
        "experiment_mode": experiment_mode,  # Experiment mode (original_only or original_plus_augmented)
        "augmentation_ratio": augmentation_ratio,  # Augmentation ratio float or None
        "n_features": n_features_override if n_features_override is not None else orig_result["n_features"],  # Number of features
        "n_samples_train": n_samples_train_override if n_samples_train_override is not None else orig_result["n_samples_train"],  # Training samples count
        "n_samples_test": n_samples_test_override if n_samples_test_override is not None else orig_result["n_samples_test"],  # Test samples count
        "accuracy": metrics[0],  # Accuracy metric
        "precision": metrics[1],  # Precision metric
        "recall": metrics[2],  # Recall metric
        "f1_score": metrics[3],  # F1 score metric
        "fpr": metrics[4],  # False positive rate
        "fnr": metrics[5],  # False negative rate
        "training_time": metrics[6],  # Training time in seconds
        "accuracy_improvement": improvements.get("accuracy", 0.0),  # Accuracy improvement percentage
        "precision_improvement": improvements.get("precision", 0.0),  # Precision improvement percentage
        "recall_improvement": improvements.get("recall", 0.0),  # Recall improvement percentage
        "f1_score_improvement": improvements.get("f1_score", 0.0),  # F1 score improvement percentage
        "fpr_improvement": improvements.get("fpr", 0.0),  # FPR improvement percentage
        "fnr_improvement": improvements.get("fnr", 0.0),  # FNR improvement percentage
        "training_time_improvement": improvements.get("training_time", 0.0),  # Training time improvement percentage
        "features_list": orig_result["features_list"],  # List of feature names used
    }  # Return comparison result entry dictionary


def generate_ratio_comparison_report(results_original, all_ratio_results):
    """
    Generates and prints comparison report for ratio-based data augmentation evaluation.
    Compares the original baseline against each augmentation ratio experiment.

    :param results_original: Dictionary of results from original data evaluation
    :param all_ratio_results: Dictionary mapping ratio (float) to results dictionary
    :return: List of comparison result entries for CSV export
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Generating ratio-based data augmentation comparison report...{Style.RESET_ALL}"
    )  # Output the verbose message

    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*100}{Style.RESET_ALL}"
    )  # Print separator line for visual clarity
    print(
        f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}DATA AUGMENTATION RATIO-BASED COMPARISON REPORT{Style.RESET_ALL}"
    )  # Print report header title
    print(
        f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*100}{Style.RESET_ALL}\n"
    )  # Print closing separator line

    comparison_results = []  # Initialize list for comparison result entries
    no_improvements = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1_score": 0.0, "fpr": 0.0, "fnr": 0.0, "training_time": 0.0}  # Zero improvements dict for original baseline entries

    for key in results_original.keys():  # Iterate through each feature_set/model combination from original results
        orig_result = results_original[key]  # Get the original baseline result entry
        feature_set = orig_result["feature_set"]  # Extract feature set name from result
        model_name = orig_result["model_name"]  # Extract model name from result
        classifier_type = orig_result["classifier_type"]  # Extract classifier type from result
        orig_metrics = extract_metrics_from_result(orig_result)  # Extract metrics list from original result
        orig_experiment_id = orig_result.get("experiment_id", None)  # Get experiment ID from original result

        comparison_results.append(
            build_comparison_result_entry(
                orig_result, feature_set, classifier_type, model_name, "Original",
                orig_metrics, no_improvements,
                experiment_id=orig_experiment_id, experiment_mode="original_only", augmentation_ratio=None,
            )
        )  # Add original baseline entry to comparison results

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Feature Set: {BackgroundColors.CYAN}{feature_set}{BackgroundColors.GREEN} | Model: {BackgroundColors.CYAN}{model_name}{Style.RESET_ALL}"
        )  # Print header with feature set and model name
        print(
            f"  {BackgroundColors.GREEN}Original baseline - Acc: {BackgroundColors.CYAN}{truncate_value(orig_metrics[0])}{BackgroundColors.GREEN}, F1: {BackgroundColors.CYAN}{truncate_value(orig_metrics[3])}{Style.RESET_ALL}"
        )  # Print original baseline metrics summary

        for ratio in sorted(all_ratio_results.keys()):  # Iterate over each ratio in sorted order
            ratio_results = all_ratio_results[ratio]  # Get results dict for this ratio
            ratio_result = ratio_results.get(key)  # Get the matching result for this feature_set/model key

            if ratio_result is None:  # If no matching result exists for this ratio
                continue  # Skip this ratio for this model/feature_set combination

            ratio_metrics = extract_metrics_from_result(ratio_result)  # Extract metrics list from ratio result
            improvements = calculate_all_improvements(orig_metrics, ratio_metrics)  # Calculate improvements vs original
            ratio_pct = int(ratio * 100)  # Convert float ratio to integer percentage for display
            ratio_experiment_id = ratio_result.get("experiment_id", None)  # Get experiment ID from ratio result

            comparison_results.append(
                build_comparison_result_entry(
                    orig_result, feature_set, classifier_type, model_name,
                    f"Original+Augmented@{ratio_pct}%", ratio_metrics, improvements,
                    n_features_override=ratio_result.get("n_features"),
                    n_samples_train_override=ratio_result.get("n_samples_train"),
                    n_samples_test_override=ratio_result.get("n_samples_test"),
                    experiment_id=ratio_experiment_id, experiment_mode="original_plus_augmented",
                    augmentation_ratio=ratio,
                )
            )  # Add ratio experiment entry with improvements to comparison results

            f1_improvement = improvements.get("f1_score", 0.0)  # Extract F1 improvement for display
            improvement_color = BackgroundColors.GREEN if f1_improvement >= 0 else BackgroundColors.RED  # Choose color based on improvement direction
            print(
                f"  {BackgroundColors.YELLOW}@{ratio_pct}%:{Style.RESET_ALL} Acc: {BackgroundColors.CYAN}{truncate_value(ratio_metrics[0])}{Style.RESET_ALL}, F1: {BackgroundColors.CYAN}{truncate_value(ratio_metrics[3])}{Style.RESET_ALL}, F1 change: {improvement_color}{f1_improvement:+.2f}%{Style.RESET_ALL}"
            )  # Print ratio result metrics with F1 improvement indicator

    return comparison_results  # Return list of all comparison result entries for CSV export


def process_augmented_data_evaluation(file, df_original_cleaned, feature_names, ga_selected_features, pca_n_components, rfe_selected_features, base_models, hp_params_map, results_original):
    """
    Handles complete augmented data evaluation workflow with ratio-based experiments.
    For each ratio in config.get("stacking", {}).get("augmentation_ratios", [0.10, 0.25, 0.50, 0.75, 1.00]), samples augmented data proportionally,
    merges with original, evaluates classifiers, and compares against original baseline.

    :param file: Original file path
    :param df_original_cleaned: Cleaned original dataframe
    :param feature_names: List of feature names
    :param ga_selected_features: Features selected by genetic algorithm
    :param pca_n_components: Number of PCA components
    :param rfe_selected_features: Features selected by RFE
    :param base_models: Dictionary of base models
    :param hp_params_map: Hyperparameters mapping
    :param results_original: Results from original data evaluation
    :return: None
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Processing augmented data evaluation for: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}"
    )  # Output the verbose message

    augmented_file = find_data_augmentation_file(file)  # Look for augmented data file using wgangp.py naming convention

    if augmented_file is None:  # If no augmented file found at expected path
        print(
            f"\n{BackgroundColors.YELLOW}No augmented data found for this file. Skipping augmentation comparison.{Style.RESET_ALL}"
        )  # Print warning message about missing augmented file
        return  # Exit function early when no augmented file exists

    df_augmented = load_dataset(augmented_file)  # Load the augmented dataset from the discovered file

    if df_augmented is None:  # If augmented dataset failed to load from disk
        print(
            f"{BackgroundColors.YELLOW}Warning: Failed to load augmented dataset from {BackgroundColors.CYAN}{augmented_file}{BackgroundColors.YELLOW}. Skipping.{Style.RESET_ALL}"
        )  # Print warning message about load failure
        return  # Exit function early on load failure

    df_augmented_cleaned = preprocess_dataframe(df_augmented)  # Preprocess the augmented dataframe with same pipeline as original

    if not validate_augmented_dataframe(df_original_cleaned, df_augmented_cleaned, file):  # Validate augmented data is compatible with original
        return  # Exit function early if augmented data fails validation checks

    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*100}{Style.RESET_ALL}"
    )  # Print separator line for visual clarity
    print(
        f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}RATIO-BASED DATA AUGMENTATION EXPERIMENTS{Style.RESET_ALL}"
    )  # Print header for the ratio-based experiments section
    print(
        f"{BackgroundColors.GREEN}Ratios to evaluate: {BackgroundColors.CYAN}{[f'{int(r*100)}%' for r in config.get("stacking", {}).get("augmentation_ratios", [0.10, 0.25, 0.50, 0.75, 1.00])]}{Style.RESET_ALL}"
    )  # Print the list of ratios that will be evaluated
    print(
        f"{BackgroundColors.BOLD}{BackgroundColors.CYAN}{'='*100}{Style.RESET_ALL}\n"
    )  # Print closing separator line

    all_ratio_results = {}  # Dictionary to store results for each ratio: {ratio: results_dict}

    for ratio_idx, ratio in enumerate(config.get("stacking", {}).get("augmentation_ratios", [0.10, 0.25, 0.50, 0.75, 1.00]), start=1):  # Iterate over each augmentation ratio
        ratio_pct = int(ratio * 100)  # Convert float ratio to integer percentage for display
        experiment_id = generate_experiment_id(file, "original_plus_augmented", ratio)  # Generate unique experiment ID for this ratio

        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}[{ratio_idx}/{len(config.get("stacking", {}).get("augmentation_ratios", [0.10, 0.25, 0.50, 0.75, 1.00]))}] Evaluating Original + Augmented@{ratio_pct}%{Style.RESET_ALL}"
        )  # Print progress indicator for current ratio experiment

        df_sampled = sample_augmented_by_ratio(df_augmented_cleaned, df_original_cleaned, ratio)  # Sample augmented rows at the current ratio

        if df_sampled is None or df_sampled.empty:  # If sampling returned no valid data
            print(
                f"{BackgroundColors.YELLOW}Warning: Could not sample augmented data at ratio {ratio}. Skipping this ratio.{Style.RESET_ALL}"
            )  # Print warning about sampling failure
            continue  # Skip to the next ratio in the loop

        df_merged = merge_original_and_augmented(df_original_cleaned, df_sampled)  # Merge original data with sampled augmented data

        data_source_label = f"Original+Augmented@{ratio_pct}%"  # Build descriptive data source label for CSV traceability

        print(
            f"{BackgroundColors.GREEN}Merged dataset: {BackgroundColors.CYAN}{len(df_original_cleaned)} original + {len(df_sampled)} augmented = {len(df_merged)} total rows{Style.RESET_ALL}"
        )  # Print merged dataset size breakdown for transparency

        generate_augmentation_tsne_visualization(
            file, df_original_cleaned, df_sampled, ratio, "original_plus_augmented"
        )  # Generate t-SNE visualization for this augmentation ratio

        results_ratio = evaluate_on_dataset(
            file, df_merged, feature_names, ga_selected_features, pca_n_components,
            rfe_selected_features, base_models, data_source_label=data_source_label,
            hyperparams_map=hp_params_map, experiment_id=experiment_id,
            experiment_mode="original_plus_augmented", augmentation_ratio=ratio,
        )  # Evaluate all classifiers on the merged dataset with experiment metadata

        all_ratio_results[ratio] = results_ratio  # Store the results for this ratio in the results dictionary

        send_telegram_message(
            TELEGRAM_BOT, f"Completed augmentation ratio {ratio_pct}% for {os.path.basename(file)}"
        )  # Send Telegram notification for ratio completion

    if not all_ratio_results:  # If no ratio experiments produced valid results
        print(
            f"{BackgroundColors.YELLOW}Warning: No ratio experiments completed successfully. Skipping comparison report.{Style.RESET_ALL}"
        )  # Print warning about no completed experiments
        return  # Exit function early when no results are available

    comparison_results = generate_ratio_comparison_report(results_original, all_ratio_results)  # Generate the comparison report across all ratios

    save_augmentation_comparison_results(file, comparison_results)  # Save comparison results to CSV file

    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Data augmentation ratio-based comparison complete!{Style.RESET_ALL}"
    )  # Print success message indicating all ratio experiments are done


def print_file_processing_header(file, config=None):
    """
    Prints formatted header for file processing section.

    :param file: File path being processed
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG

    verbose_output(
        f"{BackgroundColors.GREEN}Printing file processing header for: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}",
        config=config
    )  # Output the verbose message

    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}"
    )  # Print separator line
    print(
        f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing file: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}"
    )  # Print file being processed
    print(
        f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*100}{Style.RESET_ALL}\n"
    )  # Print separator line


def process_single_file_evaluation(file, combined_df, combined_file_for_features, config=None):
    """
    Processes evaluation for a single file including feature loading, model preparation, and evaluation.

    :param file: File path to process
    :param combined_df: Combined dataframe (used if file == "combined")
    :param combined_file_for_features: File to use for feature selection metadata
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG

    verbose_output(
        f"{BackgroundColors.GREEN}Starting single file evaluation for: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}",
        config=config
    )  # Output the verbose message

    print_file_processing_header(file, config=config)  # Print formatted header

    file_for_features = combined_file_for_features if file == "combined" else file  # Determine which file to use for feature selection metadata
    ga_selected_features, pca_n_components, rfe_selected_features = load_feature_selection_results(
        file_for_features, config=config
    )  # Load feature selection results

    df_original_cleaned, feature_names = load_and_preprocess_dataset(file, combined_df, config=config)  # Load and preprocess the dataset

    if df_original_cleaned is None:  # If loading or preprocessing failed
        return  # Exit function early

    base_models, hp_params_map = prepare_models_with_hyperparameters(file, config=config)  # Prepare base models with hyperparameters

    original_experiment_id = generate_experiment_id(file, "original_only")  # Generate unique experiment ID for the original-only evaluation

    test_data_augmentation = config.get("execution", {}).get("test_data_augmentation", False)  # Get test data augmentation flag from config
    augmentation_ratios = config.get("execution", {}).get("augmentation_ratios", [])  # Get augmentation ratios from config
    
    if test_data_augmentation:  # If data augmentation testing is enabled
        generate_augmentation_tsne_visualization(
            file, df_original_cleaned, None, None, "original_only", config=config
        )  # Generate t-SNE visualization for original data only

    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.CYAN}[1/{1 + len(augmentation_ratios) if test_data_augmentation else 1}] Evaluating on ORIGINAL data{Style.RESET_ALL}"
    )  # Print progress message with total step count
    results_original = evaluate_on_dataset(
        file, df_original_cleaned, feature_names, ga_selected_features, pca_n_components,
        rfe_selected_features, base_models, data_source_label="Original", hyperparams_map=hp_params_map,
        experiment_id=original_experiment_id, experiment_mode="original_only", augmentation_ratio=None,
        config=config
    )  # Evaluate on original data with experiment traceability metadata

    original_results_list = list(results_original.values())  # Convert results dict to list
    save_stacking_results(file, original_results_list, config=config)  # Save original results to CSV

    enable_automl = config.get("execution", {}).get("enable_automl", False)  # Get enable automl flag from config
    if enable_automl:  # If AutoML pipeline is enabled
        run_automl_pipeline(file, df_original_cleaned, feature_names, config=config)  # Run AutoML pipeline

    if test_data_augmentation:  # If data augmentation testing is enabled
        process_augmented_data_evaluation(
            file, df_original_cleaned, feature_names, ga_selected_features, pca_n_components,
            rfe_selected_features, base_models, hp_params_map, results_original, config=config
        )  # Process augmented data evaluation workflow


def process_files_in_path(input_path, dataset_name, config=None):
    """
    Processes all files in a given input path including file discovery and dataset combination.

    :param input_path: Directory path containing files to process
    :param dataset_name: Name of the dataset being processed
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG

    verbose_output(
        f"{BackgroundColors.GREEN}Processing files in path: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}",
        config=config
    )  # Output the verbose message

    if not verify_filepath_exists(input_path):  # If the input path does not exist
        verbose_output(
            f"{BackgroundColors.YELLOW}Skipping missing path: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}",
            config=config
        )  # Output skip message
        return  # Exit function early
    
    csv_file = config.get("execution", {}).get("csv_file", None)  # Get CSV file override from config

    files_to_process = determine_files_to_process(csv_file, input_path, config=config)  # Determine which files to process

    local_dataset_name = dataset_name or get_dataset_name(input_path)  # Use provided dataset name or infer from path

    combined_df, combined_file_for_features, files_to_process = combine_dataset_if_needed(files_to_process, config=config)  # Combine dataset files if needed

    for file in files_to_process:  # For each file to process
        process_single_file_evaluation(file, combined_df, combined_file_for_features, config=config)  # Process the single file evaluation


def process_dataset_paths(dataset_name, paths, config=None):
    """
    Processes all paths for a given dataset.

    :param dataset_name: Name of the dataset
    :param paths: List of paths to process for this dataset
    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG

    verbose_output(
        f"{BackgroundColors.GREEN}Processing dataset: {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}",
        config=config
    )  # Output the verbose message

    print(
        f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing dataset: {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}"
    )  # Print dataset name

    for input_path in paths:  # For each path in the dataset's paths list
        process_files_in_path(input_path, dataset_name, config=config)  # Process all files in this path


def run_stacking_pipeline(config_path=None, **config_overrides):
    """
    Programmatic entry point for stacking classifier evaluation.
    
    Allows calling this module as a library with configuration overrides:
    
    Example:
        from stacking import run_stacking_pipeline
        
        # Using config file
        run_stacking_pipeline(config_path="custom_config.yaml")
        
        # Using direct parameter overrides
        run_stacking_pipeline(
            execution={\"verbose\": True, \"test_data_augmentation\": False},
            automl={\"enabled\": True, \"n_trials\": 100}
        )
    
    :param config_path: Path to configuration file (None for default config.yaml)
    :param config_overrides: Dictionary overrides for configuration
    :return: None
    """
    
    # Initialize configuration
    config = initialize_config(config_path=config_path, cli_args=None)  # Load base config
    
    # Apply programmatic overrides
    for key, value in config_overrides.items():  # Iterate over provided overrides
        if isinstance(value, dict) and key in config:  # If override is dict and key exists
            config[key] = deep_merge_dicts(config[key], value)  # Deep merge override
        else:  # Direct override
            config[key] = value  # Set value directly
    
    # Initialize logger
    initialize_logger(config=config)  # Setup logging
    
    # Run main pipeline
    main(config=config)  # Execute stacking pipeline


def to_seconds(obj):
    """
    Converts various time-like objects to seconds.
    
    :param obj: The object to convert (can be int, float, timedelta, datetime, etc.)
    :return: The equivalent time in seconds as a float, or None if conversion fails
    """
    
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


def calculate_execution_time(start_time, finish_time=None):
    """
    Calculates the execution time and returns a human-readable string.

    Accepts either:
    - Two datetimes/timedeltas: `calculate_execution_time(start, finish)`
    - A single timedelta or numeric seconds: `calculate_execution_time(delta)`
    - Two numeric timestamps (seconds): `calculate_execution_time(start_s, finish_s)`

    Returns a string like "1h 2m 3s".
    """

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


def play_sound(config=None):
    """
    Plays a sound when the program finishes and skips if the operating system is Windows.

    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG

    current_os = platform.system()  # Get the current operating system
    if current_os == "Windows":  # If the current operating system is Windows
        return  # Do nothing
    
    sound_enabled = config.get("sound", {}).get("enabled", True)  # Get sound enabled flag from config
    if not sound_enabled:  # If sound is disabled
        return  # Do nothing
    
    sound_file = config.get("sound", {}).get("file", "./.assets/Sounds/NotificationSound.wav")  # Get sound file from config
    sound_commands = config.get("sound", {}).get("commands", {})  # Get sound commands from config

    if verify_filepath_exists(sound_file):  # If the sound file exists
        if current_os in sound_commands:  # If the platform.system() is in the sound_commands dictionary
            os.system(f"{sound_commands[current_os]} {sound_file}")  # Play the sound
        else:  # If the platform.system() is not in the sound_commands dictionary
            print(
                f"{BackgroundColors.RED}The {BackgroundColors.CYAN}{current_os}{BackgroundColors.RED} is not in the {BackgroundColors.CYAN}sound_commands dictionary{BackgroundColors.RED}. Please add it!{Style.RESET_ALL}"
            )
    else:  # If the sound file does not exist
        print(
            f"{BackgroundColors.RED}Sound file {BackgroundColors.CYAN}{sound_file}{BackgroundColors.RED} not found. Make sure the file exists.{Style.RESET_ALL}"
        )


def main(config=None):
    """
    Main function.

    :param config: Configuration dictionary (uses global CONFIG if None)
    :return: None
    """
    
    if config is None:  # If no config provided
        config = CONFIG  # Use global CONFIG

    print(
        f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Classifiers Stacking{BackgroundColors.GREEN} program!{Style.RESET_ALL}\n"
    )  # Output the welcome message
    
    test_data_augmentation = config.get("execution", {}).get("test_data_augmentation", True)  # Get test augmentation flag from config
    augmentation_ratios = config.get("stacking", {}).get("augmentation_ratios", [0.10, 0.25, 0.50, 0.75, 1.00])  # Get augmentation ratios from config

    if test_data_augmentation:  # If data augmentation testing is enabled
        print(
            f"{BackgroundColors.BOLD}{BackgroundColors.YELLOW}Data Augmentation Testing: {BackgroundColors.CYAN}ENABLED{Style.RESET_ALL}"
        )  # Print augmentation enabled message
        print(
            f"{BackgroundColors.GREEN}Will evaluate Original vs Original+Augmented at ratios: {BackgroundColors.CYAN}{[f'{int(r*100)}%' for r in augmentation_ratios]}{Style.RESET_ALL}\n"
        )  # Print augmentation ratios to be evaluated

    start_time = datetime.datetime.now()  # Get the start time of the program

    setup_telegram_bot(config=config)  # Setup Telegram bot if configured

    send_telegram_message(
        TELEGRAM_BOT, [f"Starting Classifiers Stacking at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"]
    )  # Send Telegram message indicating start

    threads_limit = set_threads_limit_based_on_ram(config=config)  # Adjust config.get("evaluation", {}).get("threads_limit", 2) based on system RAM
    
    datasets = config.get("dataset", {}).get("datasets", {})  # Get datasets from config

    for dataset_name, paths in datasets.items():  # For each dataset in the datasets dictionary
        process_dataset_paths(dataset_name, paths, config=config)  # Process all paths for this dataset

    finish_time = datetime.datetime.now()  # Get the finish time of the program
    print(
        f"\n{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
    )  # Output the start and finish times
    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
    )  # Output the end of the program message

    send_telegram_message(TELEGRAM_BOT, [f"Finished Classifiers Stacking at {finish_time.strftime('%Y-%m-%d %H:%M:%S')} | Execution time: {calculate_execution_time(start_time, finish_time)}"])  # Send Telegram message indicating finish
    
    play_sound_enabled = config.get("sound", {}).get("enabled", True)  # Get play sound flag from config
    if play_sound_enabled:  # If play sound is enabled
        atexit.register(play_sound, config=config)  # Register the play_sound function to be called when the program finishes


if __name__ == "__main__":
    # Parse CLI arguments
    cli_args = parse_cli_args()  # Parse command-line arguments
    
    # Initialize configuration with CLI overrides
    config = initialize_config(config_path=cli_args.config, cli_args=cli_args)  # Initialize config with file and CLI args
    
    # Initialize logger
    initialize_logger(config=config)  # Initialize logger with config
    
    # Run main function with config
    main(config=config)  # Run main with configuration
