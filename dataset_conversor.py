import arff # liac-arff, used to save ARFF files
import atexit # For playing a sound when the program finishes
import os # For running commands in the terminal
import pandas as pd # For handling CSV and TXT file formats
import platform # For getting the operating system name
from fastparquet import ParquetFile # For handling Parquet file format
from colorama import Style # For coloring the terminal output
from scipy.io import arff as scipy_arff # used to read ARFF files
from tqdm import tqdm # For showing a progress bar

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
INPUT_DIRECTORY = "./Input" # Input directory path
OUTPUT_DIRECTORY = "./Output" # Output directory path
IGNORE_DIRECTORY_NAMED_WITH = ["Results"] # List of directory names to ignore if they have any of this words in their name

# Sound Constants:
SOUND_COMMANDS = {"Darwin": "afplay", "Linux": "aplay", "Windows": "start"} # Sound play command
SOUND_FILE = "./.assets/Sounds/NotificationSound.wav" # Notification sound path

# RUN_FUNCTIONS:
RUN_FUNCTIONS = {
   "Play Sound": True, # Set to True to play a sound when the program finishes
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

   verbose_output(true_string=f"{BackgroundColors.GREEN}Verifying if the file or folder exists at the path: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}") # Output the verbose message

   return os.path.exists(filepath) # Return True if the file or folder exists, False otherwise

def create_directories(directory_name):
   """
   Creates a directory if it does not exist.

   :param directory_name: Name of the directory to be created.
   :return: None
   """

   verbose_output(f"{BackgroundColors.GREEN}Creating directory: {BackgroundColors.CYAN}{directory_name}{Style.RESET_ALL}") # Output the verbose message

   if not os.path.exists(directory_name): # If the directory does not exist
      os.makedirs(directory_name) # Create the directory

def get_dataset_files(directory=INPUT_DIRECTORY):
   """
   Get all dataset files in the specified directory and its subdirectories.

   :param directory: Path to the directory to search for dataset files.
   :return: List of paths to dataset files.
   """

   verbose_output(f"{BackgroundColors.GREEN}Searching for dataset files in: {BackgroundColors.CYAN}{directory}{Style.RESET_ALL}") # Output the verbose message

   dataset_files = [] # List to store paths of dataset files
   for root, dirs, files in os.walk(directory): # Walk through the directory and its subdirectories
      if any(ignore_word.lower() in root.lower() for ignore_word in IGNORE_DIRECTORY_NAMED_WITH): # Verify if the current directory contains any of the ignore words
         continue # If the current directory contains any of the ignore words, skip it

      for file in files: # Iterate through files in the current directory
         if os.path.splitext(file)[1].lower() in [".arff", ".csv", ".txt", ".parquet"]: # Verify if the file has a valid extension
            dataset_files.append(os.path.join(root, file)) # Append the full path of the file to the list

   return dataset_files # Return the list of dataset files

def clean_parquet_file(input_path, cleaned_path):
   """
   Cleans Parquet files by rewriting them without any textual cleaning,

   :param input_path: Path to the input Parquet file.
   :param cleaned_path: Path where the rewritten Parquet file will be saved.
   :return: None
   """

   pass # Parquet files are binary and do not require textual cleaning, so we just read and write them

def clean_arff_lines(lines):
   """
   Cleans ARFF files by removing unnecessary spaces in @attribute domain lists.

   :param lines: List of lines read from the ARFF file.
   :return: List of cleaned lines with sanitized domain values.
   """

   cleaned_lines = [] # List to store cleaned lines

   for line in lines: # Iterate through each line of the ARFF file
      if line.strip().lower().startswith("@attribute") and "{" in line and "}" in line: # Check if the line defines a domain list
         parts = line.split("{") # Split before domain
         before = parts[0] # Content before the domain
         domain = parts[1].split("}")[0] # Extract domain content
         after = line.split("}")[1] # Content after domain

         cleaned_domain = ",".join([val.strip() for val in domain.split(",")]) # Strip spaces inside domain list
         cleaned_line = f"{before}{{{cleaned_domain}}}{after}" # Construct cleaned line
         cleaned_lines.append(cleaned_line) # Add cleaned attribute line
      else: # If the line is not an attribute definition
         cleaned_lines.append(line) # Keep non-attribute lines unchanged

   return cleaned_lines # Return the list of cleaned lines

def clean_csv_or_txt_lines(lines):
   """
   Cleans TXT and CSV files by removing unnecessary spaces around comma-separated values.

   :param lines: List of lines read from the file.
   :return: List of cleaned lines with sanitized comma-separated values.
   """

   cleaned_lines = [] # List to store cleaned lines

   for line in lines: # Iterate through each line
      values = line.strip().split(",") # Split the line on commas
      cleaned_values = [val.strip() for val in values] # Strip whitespace
      cleaned_line = ",".join(cleaned_values) + "\n" # Join cleaned values and add newline
      cleaned_lines.append(cleaned_line) # Add cleaned line

   return cleaned_lines # Return the list of cleaned lines

def write_cleaned_lines_to_file(cleaned_path, cleaned_lines):
   """
   Writes cleaned lines to a specified file.

   :param cleaned_path: Path to the file where cleaned lines will be written.
   :param cleaned_lines: List of cleaned lines to write to the file.
   :return: None
   """

   with open(cleaned_path, "w", encoding="utf-8") as f: # Open the cleaned file path for writing
      f.writelines(cleaned_lines) # Write all cleaned lines to the output file

def clean_file(input_path, cleaned_path):
   """
   Cleans ARFF, TXT, CSV, and Parquet files by removing unnecessary spaces in
   comma-separated values or domains. For Parquet files, it rewrites the file
   directly without textual cleaning.

   :param input_path: Path to the input file (.arff, .txt, .csv, .parquet).
   :param cleaned_path: Path to save the cleaned file.
   :return: None
   """

   file_extension = os.path.splitext(input_path)[1].lower() # Get the file extension of the input file

   verbose_output(f"{BackgroundColors.GREEN}Cleaning file: {BackgroundColors.CYAN}{input_path}{BackgroundColors.GREEN} and saving to {BackgroundColors.CYAN}{cleaned_path}{Style.RESET_ALL}") # Output the verbose message

   if file_extension == ".parquet": # Handle parquet files separately (binary format)
      clean_parquet_file(input_path, cleaned_path) # Clean parquet file
      return # Exit early after handling parquet

   with open(input_path, "r", encoding="utf-8") as f: # Open the input file for reading
      lines = f.readlines() # Read all lines from the file

   if file_extension == ".arff": # Cleaning logic for ARFF files
      cleaned_lines = clean_arff_lines(lines) # Clean ARFF lines
   elif file_extension in [".txt", ".csv"]: # Cleaning logic for TXT and CSV files
      cleaned_lines = clean_csv_or_txt_lines(lines) # Clean TXT/CSV lines
   else: # If the file extension is not supported
      raise ValueError(f"{BackgroundColors.RED}Unsupported file extension: {BackgroundColors.CYAN}{file_extension}{Style.RESET_ALL}") # Raise error for unsupported formats

   write_cleaned_lines_to_file(cleaned_path, cleaned_lines) # Write cleaned lines to the cleaned file path

def load_arff_with_scipy(input_path):
   """
   Attempt to load an ARFF file using scipy. Decodes byte strings when necessary.

   :param input_path: Path to the ARFF file.
   :return: pandas DataFrame loaded from the ARFF file.
   """

   data, meta = scipy_arff.loadarff(input_path) # Load the ARFF file using scipy
   df = pd.DataFrame(data) # Convert the loaded data to a DataFrame

   for col in df.columns: # Iterate through each column in the DataFrame
      if df[col].dtype == object: # If column contains byte/string data
         df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x) # Decode bytes to strings

   return df # Return the DataFrame with decoded strings

def load_arff_with_liac(input_path):
   """
   Load an ARFF file using the liac-arff library.

   :param input_path: Path to the ARFF file.
   :return: pandas DataFrame loaded from the ARFF file.
   """

   with open(input_path, "r", encoding="utf-8") as f: # Open the ARFF file for reading
      data = arff.load(f) # Load using liac-arff

   return pd.DataFrame(data["data"], columns=[attr[0] for attr in data["attributes"]]) # Convert to DataFrame

def load_arff_file(input_path):
   """
   Load an ARFF file, trying scipy first and falling back to liac-arff if needed.

   :param input_path: Path to the ARFF file.
   :return: pandas DataFrame loaded from the ARFF file.
   """

   try: # Try loading using scipy
      return load_arff_with_scipy(input_path)
   except Exception as e: # If scipy fails, warn and try liac-arff
      verbose_output(f"{BackgroundColors.YELLOW}Warning: Failed to load ARFF with scipy ({e}). Trying with liac-arff...{Style.RESET_ALL}")

      try: # Try loading using liac-arff
         return load_arff_with_liac(input_path)
      except Exception as e2: # If both fail, raise an error
         raise RuntimeError(f"Failed to load ARFF file with both scipy and liac-arff: {e2}")

def load_csv_file(input_path):
   """
   Load a CSV file into a pandas DataFrame.

   :param input_path: Path to the CSV file.
   :return: pandas DataFrame containing the loaded dataset.
   """

   return pd.read_csv(input_path) # Load the CSV file

def load_parquet_file(input_path):
   """
   Load a Parquet file into a pandas DataFrame.

   :param input_path: Path to the Parquet file.
   :return: pandas DataFrame loaded from the Parquet file.
   """

   pf = ParquetFile(input_path) # Load the Parquet file using fastparquet
   return pf.to_pandas() # Convert the Parquet file to a pandas DataFrame

def load_txt_file(input_path):
   """
   Load a TXT file into a pandas DataFrame, assuming tab-separated values.

   :param input_path: Path to the TXT file.
   :return: pandas DataFrame containing the loaded dataset.
   """

   return pd.read_csv(input_path, sep="\t") # Load TXT file using tab separator

def load_dataset(input_path):
   """
   Load a dataset from a file in CSV, ARFF, TXT, or Parquet format into a pandas DataFrame.

   :param input_path: Path to the input dataset file.
   :return: pandas DataFrame containing the dataset.
   """

   verbose_output(f"{BackgroundColors.GREEN}Loading dataset from: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}") # Output the verbose message

   _, ext = os.path.splitext(input_path) # Get the file extension of the input file
   ext = ext.lower() # Convert the file extension to lowercase

   if ext == ".arff": # If the file is in ARFF format
      df = load_arff_file(input_path)
   elif ext == ".csv": # If the file is in CSV format
      df = load_csv_file(input_path)
   elif ext == ".parquet": # If the file is in Parquet format
      df = load_parquet_file(input_path)
   elif ext == ".txt": # If the file is in TXT format
      df = load_txt_file(input_path)
   else: # Unsupported file format
      raise ValueError(f"Unsupported file format: {ext}")

   return df # Return the loaded DataFrame

def convert_to_arff(df, output_path):
   """
   Convert a pandas DataFrame to ARFF format and save it to the specified output path.

   :param df: pandas DataFrame to be converted.
   :param output_path: Path to save the converted ARFF file.
   :return: None
   """

   verbose_output(f"{BackgroundColors.GREEN}Converting DataFrame to ARFF format and saving to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}") # Output the verbose message

   attributes = [(col, "STRING") for col in df.columns] # Define all attributes as strings
   df = df.astype(str) # Ensure all values are strings

   arff_dict = { # Create a dictionary to hold the ARFF data
      "description": "", # Description of the dataset (can be left empty)
      "relation": "converted_data", # Name of the relation (dataset)
      "attributes": attributes, # List of attributes with their names and types
      "data": df.values.tolist(), # Convert the DataFrame values to a list of lists for ARFF data
   }

   with open(output_path, "w") as f: # Open the output file for writing
      arff.dump(arff_dict, f) # Dump the ARFF data into the file using liac-arff

def convert_to_csv(df, output_path):
   """
   Convert a pandas DataFrame to CSV format and save it to the specified output path.

   :param df: pandas DataFrame to be converted.
   :param output_path: Path to save the converted CSV file.
   :return: None
   """

   verbose_output(f"{BackgroundColors.GREEN}Converting DataFrame to CSV format and saving to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}") # Output the verbose message

   df.to_csv(output_path, index=False) # Save the DataFrame to the specified output path in CSV format, without the index

def convert_to_parquet(df, output_path):
   """
   Convert a pandas DataFrame to PARQUET format and save it to the specified output path.

   :param df: pandas DataFrame to be converted.
   :param output_path: Path to save the converted PARQUET file.
   :return: None
   """

   verbose_output(f"{BackgroundColors.GREEN}Converting DataFrame to PARQUET format and saving to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}")

   df.to_parquet(output_path, index=False) # Save the DataFrame to the specified output path in PARQUET format, without the index

def convert_to_txt(df, output_path):
   """
   Convert a pandas DataFrame to TXT format and save it to the specified output path.

   :param df: pandas DataFrame to be converted.
   :param output_path: Path to save the converted TXT file.
   :return: None
   """

   verbose_output(f"{BackgroundColors.GREEN}Converting DataFrame to TXT format and saving to: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}") # Output the verbose message

   df.to_csv(output_path, sep="\t", index=False) # Save the DataFrame to the specified output path in TXT format, using tab as the separator and without the index

def batch_convert(input_directory=INPUT_DIRECTORY, output_directory=OUTPUT_DIRECTORY):
   """
   Batch converts dataset files from the input directory into multiple output formats
   (ARFF, CSV, TXT, Parquet). Ensures the output directory exists, cleans input files,
   loads them into a DataFrame, and writes all supported conversions.

   :param input_directory: Path to the input directory containing dataset files.
   :param output_directory: Path to the output directory where converted files will be saved.
   :return: None
   """

   verbose_output(f"{BackgroundColors.GREEN}Batch converting dataset files from {BackgroundColors.CYAN}{input_directory}{BackgroundColors.GREEN} to {BackgroundColors.CYAN}{output_directory}{Style.RESET_ALL}") # Output the verbose message

   dataset_files = get_dataset_files(input_directory) # Get all dataset files in the input directory

   if not dataset_files: # If no dataset files were found
      print(f"{BackgroundColors.RED}No dataset files found in {BackgroundColors.CYAN}{input_directory}{Style.RESET_ALL}") # Print error message
      return # Exit early if there are no files to convert

   pbar = tqdm(dataset_files, desc=f"{BackgroundColors.CYAN}Converting {BackgroundColors.CYAN}{len(dataset_files)}{BackgroundColors.GREEN} {'file' if len(dataset_files) == 1 else 'files'}{Style.RESET_ALL}", unit="file", colour="green", total=len(dataset_files)) # Create a progress bar for the conversion process
   for input_path in pbar: # Iterate through each dataset file
      file = os.path.basename(input_path) # Extract the file name from the full path
      name, ext = os.path.splitext(file) # Split file name into base name and extension
      ext = ext.lower() # Normalize extension to lowercase

      pbar.set_postfix_str(f"{BackgroundColors.GREEN}Processing {BackgroundColors.CYAN}{name}{ext}{Style.RESET_ALL}") # Display current file in progress bar

      if ext not in [".arff", ".csv", ".parquet", ".txt"]: # Skip unsupported file types
         continue # Move to the next file

      cleaned_path = os.path.join(output_directory, f"{name}{ext}") # Path for saving the cleaned file
      clean_file(input_path, cleaned_path) # Clean the file before conversion

      df = load_dataset(cleaned_path) # Load the cleaned dataset into a DataFrame

      if ext != ".arff": # Convert to ARFF if not already ARFF
         convert_to_arff(df, os.path.join(output_directory, f"{name}.arff")) # Write ARFF file
      if ext != ".csv": # Convert to CSV if not already CSV
         convert_to_csv(df, os.path.join(output_directory, f"{name}.csv")) # Write CSV file
      if ext != ".parquet": # Convert to Parquet if not already Parquet
         convert_to_parquet(df, os.path.join(output_directory, f"{name}.parquet")) # Write Parquet file
      if ext != ".txt": # Convert to TXT if not already TXT
         convert_to_txt(df, os.path.join(output_directory, f"{name}.txt")) # Write TXT file
      
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
   """
   
   print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Multi-Format Dataset Converter{BackgroundColors.GREEN}!{Style.RESET_ALL}\n") # Output the Welcome message

   if not verify_filepath_exists(INPUT_DIRECTORY): # If the input directory does not exist
      create_directories(INPUT_DIRECTORY) # Create input folder if it does not exist
      print(f"{BackgroundColors.RED}Input folder does not exist: {INPUT_DIRECTORY}{Style.RESET_ALL}")
      return # Exit the program if the input folder does not exist
   
   if not verify_filepath_exists(OUTPUT_DIRECTORY): # If the output directory does not exist
      create_directories(OUTPUT_DIRECTORY) # Create the output directory

   batch_convert(INPUT_DIRECTORY, OUTPUT_DIRECTORY) # Batch convert dataset files from the input directory to multiple formats in the output directory

   print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Conversions completed!{Style.RESET_ALL}") # Output the completion message

   atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None # Register the play_sound function to be called when the program exits if RUN_FUNCTIONS["Play Sound"] is True

if __name__ == "__main__":
   """
   This is the standard boilerplate that calls the main() function.

   :return: None
   """

   main() # Call the main function
