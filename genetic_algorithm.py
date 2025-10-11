"""
================================================================================
Genetic Algorithm Feature Selection & Analysis
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-10-07
Description :
   This script runs a DEAP-based Genetic Algorithm (GA) to perform feature
   selection for classification problems. It provides an end-to-end pipeline:
   dataset loading and cleaning, scaling, GA setup and execution, candidate
   evaluation (with a Random Forest base estimator), and post-hoc analysis
   (RFE ranking correlation, CSV summaries and boxplot visualizations).

   Key features include:
      - DEAP-based GA for binary-mask feature selection
      - Fitness evaluation using a RandomForest and returning multi-metrics
        (accuracy, precision, recall, F1, FPR, FNR, elapsed time)
      - Population sweep support (run GA over a range of population sizes)
      - Integration with previously-computed RFE rankings for cross-checking
      - Exports: best-subset text file, CSV summaries and per-feature boxplots
      - Progress bars via tqdm and safe filename handling for outputs
      - Cross-platform completion notification (optional sound)

Usage:
   1. Configure the dataset:
      - Edit the `csv_file` variable in the `main()` function to point to
         the CSV dataset you want to analyze (the script assumes the last
         column is the target and numeric features are used).
   2. Optionally tune GA parameters in `main()` or call sites:
      - n_generations, min_pop, max_pop, population_size, train_test_ratio
   3. Run the pipeline via the project's Makefile:
      $ make main
      (Makefile is expected to setup env / deps and execute this script.)
   NOTE:
      - If you prefer not to use the Makefile, you can run the module/script
        directly from Python in your dev environment, but the recommended
        workflow for the project is `make main`.

Outputs:
   - Feature_Analysis/Genetic_Algorithm_results.txt  (best subset + RFE cross-info)
   - Feature_Analysis/<dataset>_feature_summary.csv  (mean/std per class for selected features)
   - Feature_Analysis/<dataset>-<feature>.png         (boxplots for top features)
   - Console summary of best subsets per population size (when sweeping)

TODOs:
   - Add CLI argument parsing (argparse) to avoid editing `main()` for different runs.
   - Add cross-validation or nested CV to make fitness evaluation more robust.
   - Support multi-objective optimization (e.g., F1 vs. model training time).
   - Parallelize individual evaluations (joblib / dask) to speed up GA fitness calls.
   - Save and version best individuals (pickle/JSON) and GA run metadata.
   - Implement reproducible seeding across DEAP, numpy, random and sklearn.
   - Add automatic handling of categorical features and missing-value imputation.
   - Add early stopping and convergence checks to the GA loop.
   - Produce a machine-readable summary (JSON) of final metrics and selected features.
   - Add unit tests for core functions (fitness evaluation, GA setup, I/O).

Dependencies:
   - Python >= 3.9
   - pandas, numpy, scikit-learn, deap, tqdm, matplotlib, seaborn, colorama

Assumptions & Notes:
   - Dataset format: CSV, last column = target. Only numeric features are used.
   - RFE results (if present) are read from `Feature_Analysis/RFE_results_RandomForestClassifier.txt`.
   - Sound notification is skipped on Windows by default.
   - The script uses RandomForestClassifier as the default evaluator; change as needed.
   - Inspect output directories (`Feature_Analysis/`) after runs for artifacts.
"""

import atexit # For playing a sound when the program finishes
import matplotlib.pyplot as plt # For plotting graphs
import numpy as np # For numerical operations
import os # For running a command in the terminal
import pandas as pd # For data manipulation
import platform # For getting the operating system name
import random # For random number generation
import re # For sanitizing filenames
import seaborn as sns # For enhanced plotting
import time # For measuring execution time
from colorama import Style # For coloring the terminal
from deap import base, creator, tools, algorithms # For the genetic algorithm
from tqdm import tqdm # For progress bars
from sklearn.ensemble import RandomForestClassifier # For the machine learning model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix # For model evaluation
from sklearn.model_selection import train_test_split # For splitting the dataset
from sklearn.preprocessing import StandardScaler # For feature scaling

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

# Functions Definition

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

def load_dataset(csv_path):
   """
   Load CSV and return DataFrame.

   :param csv_path: Path to CSV dataset.
   :return: DataFrame
   """

   print(f"\n{BackgroundColors.GREEN}Loading dataset from: {BackgroundColors.CYAN}{csv_path}{Style.RESET_ALL}") # Output the loading dataset message

   if not verify_filepath_exists(csv_path): # If the CSV file does not exist
      print(f"{BackgroundColors.RED}CSV file not found: {csv_path}{Style.RESET_ALL}")
      return None # Return None

   df = pd.read_csv(csv_path, low_memory=False) # Load the dataset

   if df.shape[1] < 2: # If there are less than 2 columns
      print(f"{BackgroundColors.RED}CSV must have at least 1 feature and 1 target.{Style.RESET_ALL}")
      return None # Return None

   return df # Return the loaded DataFrame

def split_dataset(df, train_test_ratio=0.2):
   """
   Split dataset into training and testing sets.

   :param df: DataFrame to split.
   :param train_test_ratio: Proportion of the dataset to include in the test split.
   :return: X_train, X_test, y_train, y_test
   """

   X = df.iloc[:, :-1].select_dtypes(include=["number"]) # Select only numeric features
   y = df.iloc[:, -1] # Target variable
   if y.dtype == object or y.dtype == "category": # If the target variable is categorical
      y, _ = pd.factorize(y) # Factorize the target variable

   X = X.replace([np.inf, -np.inf], np.nan).dropna() # Remove rows with NaN or infinite values
   y = y[X.index] # Align y with cleaned X

   if X.empty: # If no numeric features remain after cleaning
      print(f"{BackgroundColors.RED}No valid numeric features remain after cleaning.{Style.RESET_ALL}")
      return None, None, None, None, None # Return None values

   scaler = StandardScaler() # Initialize the scaler
   X_scaled = scaler.fit_transform(X) # Scale the features
   X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=train_test_ratio, random_state=42) # Split the dataset

   return X_train, X_test, y_train, y_test, X.columns # Return the split data and feature names

def setup_genetic_algorithm(n_features, population_size=30):
   """
   Setup DEAP Genetic Algorithm: creator, toolbox, population, and Hall of Fame.
   DEAP is a library for evolutionary algorithms in Python.

   :param n_features: Number of features in dataset
   :param population_size: Size of the population
   :return: toolbox, population, hall_of_fame
   """

   # Avoid re-creating FitnessMax and Individual
   if not hasattr(creator, "FitnessMax"): # If FitnessMax is not already created
      creator.create("FitnessMax", base.Fitness, weights=(1.0,)) # Maximize F1-score
   if not hasattr(creator, "Individual"): # If Individual is not already created
      creator.create("Individual", list, fitness=creator.FitnessMax) # Individual = list with fitness

   toolbox = base.Toolbox() # Create a toolbox
   toolbox.register("attr_bool", random.randint, 0, 1) # Attribute generator (0 or 1)
   toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_features) # Individual generator
   toolbox.register("population", tools.initRepeat, list, toolbox.individual) # Population generator

   toolbox.register("mate", tools.cxTwoPoint) # Crossover operator
   toolbox.register("mutate", tools.mutFlipBit, indpb=0.05) # Mutation operator
   toolbox.register("select", tools.selTournament, tournsize=3) # Selection operator

   population = toolbox.population(n=population_size) # Create the initial population
   hof = tools.HallOfFame(1) # Hall of Fame to store the best individual

   return toolbox, population, hof # Return the toolbox, population, and Hall of Fame

def evaluate_individual(individual, X_train, y_train, X_test, y_test):
   """
   Evaluate the fitness of an individual solution.

   :param individual: A list representing the individual solution (binary mask for feature selection).
   :param X_train: Training feature set.
   :param y_train: Training target variable.
   :param X_test: Testing feature set.
   :param y_test: Testing target variable.
   :return: Tuple containing accuracy, precision, recall, F1-score, FPR,
   """

   if sum(individual) == 0: # If no features are selected, return worst possible scores
      return 0, 0, 0, 0, 1, 1, float("inf") # Accuracy, Precision, Recall, F1-score, FPR, FNR, Time

   selected_idx = [i for i, bit in enumerate(individual) if bit == 1] # Indices of selected features
   X_train_sel = X_train[:, selected_idx] # Select features for training set
   X_test_sel = X_test[:, selected_idx] # Select features for testing set

   start_time = time.time() # Start time measurement
   model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1) # Initialize the model
   model.fit(X_train_sel, y_train) # Train the model
   y_pred = model.predict(X_test_sel) # Make predictions
   elapsed_time = time.time() - start_time # Calculate elapsed time

   acc = accuracy_score(y_test, y_pred) # Calculate accuracy
   prec = precision_score(y_test, y_pred, average="weighted", zero_division=0) # Calculate precision
   rec = recall_score(y_test, y_pred, average="weighted", zero_division=0) # Calculate recall
   f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0) # Calculate F1-score

   cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test)) # Confusion matrix
   tn = cm[0, 0] if cm.shape == (2, 2) else 0 # True negatives
   fp = cm[0, 1] if cm.shape == (2, 2) else 0 # False positives
   fn = cm[1, 0] if cm.shape == (2, 2) else 0 # False negatives
   tp = cm[1, 1] if cm.shape == (2, 2) else 0 # True positives

   fpr = fp / (fp + tn) if (fp + tn) > 0 else 0 # False positive rate
   fnr = fn / (fn + tp) if (fn + tp) > 0 else 0 # False negative rate

   return acc, prec, rec, f1, fpr, fnr, elapsed_time # Return all metrics

def run_genetic_algorithm_loop(toolbox, population, hof, X_train, y_train, X_test, y_test, n_generations=20):
   """
   Run Genetic Algorithm generations with a tqdm progress bar.

   :param toolbox: DEAP toolbox with registered functions.
   :param population: Initial population.
   :param hof: Hall of Fame to store the best individual.
   :param X_train: Training feature set.
   :param y_train: Training target variable.
   :param X_test: Testing feature set.
   :param y_test: Testing target variable.
   :param n_generations: Number of generations to run.
   :return: best individual
   """

   def fitness(ind): # Fitness function for DEAP
      acc, prec, rec, f1, fpr, fnr, t = evaluate_individual(ind, X_train, y_train, X_test, y_test) # Evaluate the individual
      return (f1,) # Return F1-score as the fitness value

   toolbox.register("evaluate", fitness) # Register the fitness function

   for _ in range(1, n_generations + 1): # Loop for the specified number of generations
      offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2) # Apply crossover and mutation
      fits = list(map(toolbox.evaluate, offspring)) # Evaluate the offspring

      for ind, fit in zip(offspring, fits): # Assign fitness values
         ind.fitness.values = fit # Set the fitness value
      
      population[:] = toolbox.select(offspring, k=len(population)) # Select the next generation population
      hof.update(population) # Update the Hall of Fame

   return hof[0] # Return the best individual from the Hall of Fame

def safe_filename(name):
   """
   Sanitize a string to be safe for use as a filename.

   :param name: The string to be sanitized.
   :return: A sanitized string safe for use as a filename.
   """

   return re.sub(r'[\\/*?:"<>|]', "_", name) # Replace invalid filename characters with underscores

def analyze_top_features(df, y, top_features, csv_path="."):
   """
   Analyze and visualize the top features.

   :param df: DataFrame containing the features.
   :param y: Target variable.
   :param top_features: List of top feature names.
   :param csv_path: Path to the original CSV file for saving outputs.
   :return: None
   """

   df_analysis = df[top_features].copy() # Create a copy of the DataFrame with only the top features
   df_analysis["Target"] = pd.Series(y, index=df_analysis.index).astype(str) # Add the target variable to the DataFrame

   output_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis" # Directory to save outputs
   os.makedirs(output_dir, exist_ok=True) # Create the directory if it doesn't exist

   base_dataset_name = os.path.splitext(os.path.basename(csv_path))[0] # Base name of the dataset

   summary = df_analysis.groupby("Target")[top_features].agg(["mean", "std"]) # Calculate mean and std for each feature grouped by target
   summary.columns = [f"{col}_{stat}" for col, stat in summary.columns] # Flatten MultiIndex columns
   summary = summary.round(3) # Round to 3 decimal places

   summary_csv_path = f"{output_dir}/{base_dataset_name}_feature_summary.csv" # Path to save the summary CSV
   summary.to_csv(summary_csv_path, encoding="utf-8") # Save the summary to a CSV file
   print(f"\n{BackgroundColors.GREEN}Features summary saved to {BackgroundColors.CYAN}{summary_csv_path}{Style.RESET_ALL}") # Notify user

   for feature in top_features: # For each top feature
      plt.figure(figsize=(8, 5)) # Create a new figure
      sns.boxplot(x="Target", y=feature, data=df_analysis, hue="Target", palette="Set2", dodge=False) # Boxplot
      plt.title(f"Distribution of '{feature}' by class") # Title
      plt.xlabel("Traffic Type") # X-axis label
      plt.ylabel(feature) # Y-axis label
      plt.tight_layout() # Adjust layout
      plt.savefig(f"{output_dir}/{base_dataset_name}-{safe_filename(feature)}.png") # Save the plot
      plt.close() # Close the plot to free memory

def normalize_feature_name(name):
   """
   Normalize feature name by stripping whitespace and replacing double spaces with single spaces.

   :param name: The feature name to normalize.
   :return: Normalized feature name
   """

   return name.strip().replace("  ", " ") # Strip whitespace and replace double spaces with single spaces

def run_genetic_algorithm_feature_selection(df, n_generations=20, population_size=30, train_test_ratio=0.2):
   """
   Run the genetic algorithm for feature selection on the given DataFrame.

   :param df: DataFrame to process.
   :param n_generations: Number of generations for the Genetic Algorithm.
   :param population_size: Population size for the Genetic Algorithm.
   :param train_test_ratio: Proportion of the dataset to include in the test split.
   :return: Best individual, feature names, training and testing sets (X_train, X_test, y_train, y_test).
   """

   if df is None: # If loading the dataset failed
      return [] # Return an empty list

   X_train, X_test, y_train, y_test, feature_names = split_dataset(df, train_test_ratio) # Split the dataset

   if X_train is None or X_test is None or y_train is None or y_test is None: # If splitting the dataset failed
      return [] # Return an empty list

   toolbox, population, hof = setup_genetic_algorithm(len(feature_names), population_size) # Setup the Genetic Algorithm
   best_ind = run_genetic_algorithm_loop(toolbox, population, hof, X_train, y_train, X_test, y_test, n_generations) # Run the Genetic Algorithm loop

   return best_ind, feature_names, X_train, X_test, y_train, y_test # Return all required values

def extract_rfe_ranking(csv_path):
   """
   Extract RFE rankings from the RFE results file.

   :param csv_path: Path to the original CSV file for saving outputs.
   :return: Dictionary of feature names and their RFE rankings.
   """

   rfe_ranking = {} # Dictionary to store feature names and their RFE rankings
   rfe_file_path = f"{os.path.dirname(csv_path)}/Feature_Analysis/RFE_results_RandomForestClassifier.txt" # Path to the RFE results file

   if not verify_filepath_exists(rfe_file_path): # If the RFE results file does not exist
      print(f"{BackgroundColors.YELLOW}RFE results file not found: {rfe_file_path}. Skipping RFE ranking extraction.{Style.RESET_ALL}")
      return rfe_ranking # Return empty dictionary
   
   with open(rfe_file_path, "r") as f: # Open the RFE results file
      lines = f.readlines() # Read all lines
      for line in lines: # For each line
         match = re.match(r"\d+\s+-\s+Column\s+\d+:\s+(.+?)\s*:\s+Selected\s+\(Rank\s+(\d+)\)", line) # Match the line with regex
         if match: # If the line matches the regex
            feature_name = normalize_feature_name(match.group(1)) # Normalize the feature name
            rank = int(match.group(2).strip()) # Extract the rank
            rfe_ranking[feature_name] = rank # Add to the dictionary
               
   return rfe_ranking # Return the RFE rankings dictionary

def print_metrics(metrics):
   """
   Print performance metrics.

   :param metrics: Dictionary or tuple containing evaluation metrics.
   :return: None
   """

   if not metrics: # If metrics is None or empty
      return # Do nothing
   
   acc, prec, rec, f1, fpr, fnr, elapsed_time = metrics
   print(f"\n{BackgroundColors.GREEN}Performance Metrics for the Random Forest Classifier using the best feature subset:{Style.RESET_ALL}")
   print(f"   {BackgroundColors.GREEN}Accuracy: {BackgroundColors.CYAN}{acc:.4f}{Style.RESET_ALL}")
   print(f"   {BackgroundColors.GREEN}Precision: {BackgroundColors.CYAN}{prec:.4f}{Style.RESET_ALL}")
   print(f"   {BackgroundColors.GREEN}Recall: {BackgroundColors.CYAN}{rec:.4f}{Style.RESET_ALL}")
   print(f"   {BackgroundColors.GREEN}F1-Score: {BackgroundColors.CYAN}{f1:.4f}{Style.RESET_ALL}")
   print(f"   {BackgroundColors.GREEN}False Positive Rate (FPR): {BackgroundColors.CYAN}{fpr:.4f}{Style.RESET_ALL}")
   print(f"   {BackgroundColors.GREEN}False Negative Rate (FNR): {BackgroundColors.CYAN}{fnr:.4f}{Style.RESET_ALL}")
   print(f"   {BackgroundColors.GREEN}Elapsed Time (s): {BackgroundColors.CYAN}{elapsed_time:.2f}{Style.RESET_ALL}")

def write_best_features_to_file(best_features, rfe_ranking, results_file, metrics=None):
   """
   Write the best features and their RFE rankings to a results file.

   :param best_features: List of best features selected by the Genetic Algorithm.
   :param rfe_ranking: Dictionary of feature names and their RFE rankings.
   :param results_file: Path to the results file for saving outputs.
   :param metrics: Dictionary or tuple containing evaluation metrics.
   :return: None
   """

   with open(results_file, "w") as f: # Open the results file for writing
      acc, prec, rec, f1, fpr, fnr, elapsed_time = metrics # Unpack metrics
      f.write("Performance Metrics for the Random Forest Classifier using the best feature subset from the Genetic Algorithm:\n") # Header for metrics section
      f.write(f"Accuracy: {acc:.4f}\n")
      f.write(f"Precision: {prec:.4f}\n")
      f.write(f"Recall: {rec:.4f}\n")
      f.write(f"F1-Score: {f1:.4f}\n")
      f.write(f"False Positive Rate (FPR): {fpr:.4f}\n")
      f.write(f"False Negative Rate (FNR): {fnr:.4f}\n")
      f.write(f"Elapsed Time (s): {elapsed_time:.2f}\n")

      f.write("\n\nBest Feature Subset using Genetic Algorithm:\n") # Write header
      for i, feat in enumerate(best_features, start=1): # For each best feature
         feat_norm = normalize_feature_name(feat) # Normalize the feature name
         rank_info = f" (RFE ranking {rfe_ranking[feat_norm]})" if feat_norm in rfe_ranking else " (RFE ranking N/A)" # Get RFE ranking info
         f.write(f"{i}. {feat_norm}{rank_info}\n") # Write feature and its RFE ranking

      print(f"\n{BackgroundColors.GREEN}Best features and metrics saved to {BackgroundColors.CYAN}{results_file}{Style.RESET_ALL}") # Notify user

def save_best_features(best_features, rfe_ranking, csv_path, metrics=None):
   """
   Save the best features and their RFE rankings to a results file.

   :param best_features: List of best features selected by the Genetic Algorithm.
   :param rfe_ranking: Dictionary of feature names and their RFE rankings.
   :param csv_path: Path to the original CSV file for saving outputs.
   :param metrics: Dictionary or tuple containing evaluation metrics.
   :return: None
   """

   output_dir = f"{os.path.dirname(csv_path)}/Feature_Analysis/"  # Directory to save outputs
   os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't exist
   results_file = f"{output_dir}/Genetic_Algorithm_Results.txt"  # Path to save the results file

   write_best_features_to_file(best_features, rfe_ranking, results_file, metrics=metrics) # Delegate writing to helper function

def save_and_analyze_results(best_ind, feature_names, X, y, csv_path, metrics=None):
   """
   Save best feature subset and analyze them.

   :param best_ind: Best individual from the Genetic Algorithm.
   :param feature_names: List of feature names.
   :param X: Feature set (DataFrame or numpy array).
   :param y: Target variable (Series or array).
   :param csv_path: Path to the original CSV file for saving outputs.
   :param metrics: Dictionary or tuple containing evaluation metrics.
   :return: List of best features
   """

   best_features = [f for f, bit in zip(feature_names, best_ind) if bit == 1] # Extract best features
   rfe_ranking = extract_rfe_ranking(csv_path) # Extract RFE rankings

   print(f"\n{BackgroundColors.GREEN}Best features subset found: {BackgroundColors.CYAN}{best_features}{Style.RESET_ALL}")

   save_best_features(best_features, rfe_ranking, csv_path, metrics=metrics) # Save best features, rankings, and metrics

   if best_features: # If there are best features to analyze
      if not isinstance(X, pd.DataFrame): # If X is not a pandas DataFrame
         try: # Try to create a DataFrame with original feature names
            df_features = pd.DataFrame(X, columns=list(feature_names)) # Create DataFrame with original feature names
         except Exception: # If creating DataFrame with original feature names fails
            df_features = pd.DataFrame(X) # Create DataFrame without original feature names
            df_features.columns = [f"feature_{i}" for i in range(df_features.shape[1])] # Generic feature names
      else: # If X is already a pandas DataFrame
         df_features = X.copy() # Use the DataFrame as is

      if not isinstance(y, pd.Series): # If y is not a pandas Series
         try: # Try to create a Series with original indices
            y_series = pd.Series(y, index=df_features.index) # Create Series with original indices
         except Exception: # If creating Series with original indices fails
            y_series = pd.Series(y) # Create Series without original indices
      else: # If y is already a pandas Series
         y_series = y.reindex(df_features.index) if not df_features.index.equals(y.index) else y # Align indices if necessary

      analyze_top_features(df_features, y_series, best_features, csv_path=csv_path) # Analyze and visualize the top features

   return best_features # Return the list of best features

def run_population_sweep(csv_path, n_generations=20, min_pop=3, max_pop=30, train_test_ratio=0.2):
   """
   Run the genetic algorithm for feature selection with population sizes from min_pop to max_pop.
   Show a progress bar and only save the best result overall.
   """

   best_score = -1 # Initialize best score
   best_result = None # Initialize best result
   best_metrics = None # Initialize best metrics
   results = {} # Dictionary to store results for each population size

   df = load_dataset(csv_path) # Load the dataset
   for pop_size in tqdm(range(min_pop, max_pop + 1), desc=f"{BackgroundColors.GREEN}Population Sweep{Style.RESET_ALL}", unit="pop"):
      ga_result = run_genetic_algorithm_feature_selection(df, n_generations=n_generations, population_size=pop_size, train_test_ratio=train_test_ratio) # Run the Genetic Algorithm feature selection

      if ga_result is None: # If running the Genetic Algorithm failed
         continue # Skip to the next population size

      best_ind, feature_names, X_train, X_test, y_train, y_test = ga_result # Unpack the result
      best_features = [f for f, bit in zip(feature_names, best_ind) if bit == 1] # Extract best features
      results[pop_size] = best_features # Store the best features for this population size

      metrics = evaluate_individual(best_ind, X_train, y_train, X_test, y_test) # Evaluate the best individual
      acc, prec, rec, f1, fpr, fnr, elapsed_time = metrics # Unpack the metrics

      if f1 > best_score: # If the F1-score is better than the best score
         best_score = f1 # Update the best score
         best_metrics = metrics # Update the best metrics
         best_result = (best_ind, feature_names, X_train, X_test, y_train, y_test) # Update the best result

   if best_result: # After the sweep, if we have a best result
      best_ind, feature_names, X_train, X_test, y_train, y_test = best_result # Unpack the best result
      print_metrics(best_metrics) # Print the best performance metrics
      save_and_analyze_results(best_ind, feature_names, X_train, y_train, csv_path) # Save and analyze the best results

   return results # Return the results dictionary

def play_sound():
   """
   Plays a sound when the program finishes and skips if the operating system is Windows.
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

   :return: None
   :return: None
   """

   print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Genetic Algorithm Feature Selection{BackgroundColors.GREEN} program!{Style.RESET_ALL}")

   csv_file = "./Datasets/DDoS/CICDDoS2019/01-12/DrDoS_DNS.csv"

   sweep_results = run_population_sweep(csv_file, n_generations=20, min_pop=5, max_pop=30, train_test_ratio=0.2)

   print(f"\n{BackgroundColors.GREEN}Summary of best features by population size:{Style.RESET_ALL}") # Print summary of results
   for pop_size, features in sweep_results.items(): # For each population size and its best features
      print(f"  Pop {pop_size}: {len(features)} features -> {features}") # Print the population size and the best features

   print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}") # Output the end of the program message

   atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None # Register the play_sound function to be called when the program finishes

if __name__ == "__main__":
   """
   This is the standard boilerplate that calls the main() function.

   :return: None
   """

   main() # Call the main function
