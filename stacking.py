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
