import atexit # For playing a sound when the program finishes
import os # For running a command in the terminal
import pandas as pd # For data manipulation
import platform # For getting the operating system name
from colorama import Style # For coloring the terminal

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

def collect_matching_files(input_dir, file_format=".csv", ignore_files=None):
   """
   Recursively collects all files in the specified directory and subdirectories
   that match the given file format and are not in the ignore list.

   :param input_dir: Directory to search
   :param file_format: File format to include (default: .csv)
   :param ignore_files: List of filenames to ignore
   :return: Sorted list of matching file paths
   """

   ignore_files = ignore_files or [] # Default to empty list if None
   matching_files = [] # List to store matching file paths
   
   for root, _, files in os.walk(input_dir): # Walk through the directory
      for file in files: # For each file
         if file.endswith(file_format) and file not in ignore_files: # Verify if it matches the format and is not ignored
            matching_files.append(os.path.join(root, file)) # Add the full file path to the list
   
   sorted_matching_files = sorted(set(matching_files)) # Remove duplicates and sort the list
   return sorted_matching_files # Return the sorted list of matching files

def load_dataset(filepath, low_memory=True):
   """
   Loads a dataset from a CSV file.
   
   :param filepath: Path to the CSV file
   :param low_memory: Whether to use low memory mode (default: True)
   :return: Pandas DataFrame
   """

   try: # Try to load the dataset
      df = pd.read_csv(filepath, low_memory=low_memory) # Load the dataset
      return df # Return the DataFrame
   except Exception as e:
      print(f"{BackgroundColors.RED}Error loading {BackgroundColors.GREEN}{filepath}: {e}{Style.RESET_ALL}")
      return None

def detect_label_column(columns):
   """
   Try to guess the label column based on common naming conventions.
   
   :param columns: List of column names
   :return: The name of the label column if found, else None
   """

   candidates = ["label", "class", "target"]

   for col in columns: # First search for exact matches
      if col.lower() in candidates: # Verify if the column name matches any candidate exactly
         return col # Return the column name if found

   for col in columns: # Second search for partial matches
      if "target" in col.lower() or "label" in col.lower(): # Verify if the column name contains any candidate
         return col # Return the column name if found

   return None # Return None if no label column is found

def summarize_features(df):
   """
   Summarizes number of samples, features, and feature types.
   Ensures the sum of feature types matches the number of columns.

   :param df: pandas DataFrame
   :return: Tuple containing:
            n_samples, n_features, n_numeric, n_int, n_categorical, n_other, categorical columns string
   """

   n_samples, n_features = df.shape # Get number of samples and features
   dtypes = df.dtypes # Get data types of each column

   n_numeric = dtypes[dtypes == "float64"].count() # Count float64 types
   n_int = dtypes[dtypes == "int64"].count() + dtypes[dtypes == "Int64"].count() # Count int64 and Int64 types
   n_categorical = dtypes[dtypes.isin(["object", "category", "bool", "string"])].count() # Count categorical types

   n_other = n_features - (n_numeric + n_int + n_categorical) # Anything else goes to "other"

   categorical_cols = df.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist() # List of categorical columns
   categorical_cols_str = ", ".join(categorical_cols) if categorical_cols else "None" # Create string of categorical columns or "None"

   return n_samples, n_features, n_numeric, n_int, n_categorical, n_other, categorical_cols_str # Return the summary values

def summarize_missing_values(df):
   """
   Summarizes missing values for the dataset.

   :param df: The pandas DataFrame
   :return: Summary string of missing values
   """

   missing_vals = df.isnull().sum() # Get count of missing values per column
   missing_summary = ", ".join([f"{col} ({cnt})" for col, cnt in missing_vals.items() if cnt > 0]) if missing_vals.sum() > 0 else "None" # Create summary string or "None"

   return missing_summary # Return the missing values summary

def summarize_classes(df, label_col):
   """
   Summarizes classes and class distributions if a label column exists.

   :param df: The pandas DataFrame
   :param label_col: The name of the label column
   :return: Tuple containing string of classes and class distribution summary
   """

   if label_col and label_col in df.columns: # If a label column exists
      classes = df[label_col].unique() # Get unique classes
      classes_str = ", ".join(map(str, classes)) # Create string of classes
      class_counts = df[label_col].value_counts() # Get counts of each class
      total = class_counts.sum() # Total number of samples
      class_dist_list = [f"{cls}: {cnt} ({cnt/total*100:.2f}%)" for cls, cnt in class_counts.items()] # Create class distribution list
      class_dist_str = ", ".join(class_dist_list) # Create class distribution string
      return classes_str, class_dist_str # Return the classes and class distribution
   
   return "None", "None" # Return "None" if no label column

def get_dataset_info(filepath, low_memory=True):
   """
   Extracts dataset information from a CSV file and returns it as a dictionary.

   :param filepath: Path to the CSV file
   :param low_memory: Whether to use low memory mode when loading the CSV (default:True)
   :return: Dictionary containing dataset information
   """

   df = load_dataset(filepath, low_memory) # Load the dataset
   
   if df is None: # If the dataset could not be loaded
      return None # Return None

   label_col = detect_label_column(df.columns) # Try to detect the label column
   n_samples, n_features, n_numeric, n_int, n_categorical, n_other, categorical_cols_str = summarize_features(df) # Summarize features
   missing_summary = summarize_missing_values(df) # Summarize missing values
   classes_str, class_dist_str = summarize_classes(df, label_col) # Summarize classes and distributions

   return { # Return the dataset information as a dictionary
      "Dataset Name": os.path.basename(filepath),
      "Number of Samples": n_samples,
      "Number of Features": n_features,
      "Feature Types": f"{n_numeric} numeric (float64), {n_int} integer (int64), {n_categorical} categorical (object/category/bool/string), {n_other} other",
      "Categorical Features (object/string)": categorical_cols_str,
      "Missing Values": missing_summary,
      "Classes": classes_str,
      "Class Distribution": class_dist_str
   }

def write_report(report_rows, base_dir, output_filename):
   """
   Writes the report rows to a CSV file.

   :param report_rows: List of dictionaries containing report data
   :param base_dir: Base directory for saving the report
   :param output_filename: Name of the output CSV file
   :return: None
   """

   report_df = pd.DataFrame(report_rows) # Create a DataFrame from the report rows
   report_csv_path = os.path.join(base_dir, output_filename) # Path to save the report CSV
   report_df.to_csv(report_csv_path, index=False) # Save the report to a CSV file

def main():
   """
   Main function.

   :return: None
   """

   print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Dataset Descriptor{BackgroundColors.GREEN}!{Style.RESET_ALL}", end="\n\n") # Output the Welcome message

   input_path = "./DDoS/CICDDoS2019/03-11/" # Path to the CSV file

   if verify_filepath_exists(input_path): # Verify if the input path exists
      generate_dataset_report(input_path=input_path, low_memory=True) # Generate the dataset report
   else: # If the input path does not exist
      print(f"{BackgroundColors.RED}Input path does not exist: {input_path}{Style.RESET_ALL}") # Output the error message

   print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}") # Output the end of the program message

   atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None # Register the play_sound function to be called when the program finishes

if __name__ == "__main__":
   """
   This is the standard boilerplate that calls the main() function.

   :return: None
   """

   main() # Call the main function
