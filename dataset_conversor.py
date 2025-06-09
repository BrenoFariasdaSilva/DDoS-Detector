import arff # liac-arff, used to save ARFF files
import atexit # For playing a sound when the program finishes
import os # For running commands in the terminal
import pandas as pd # For handling CSV and TXT file formats
import platform # For getting the operating system name
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
         if os.path.splitext(file)[1].lower() in [".arff", ".csv", ".txt"]: # Verify if the file has a valid extension
            dataset_files.append(os.path.join(root, file)) # Append the full path of the file to the list

   return dataset_files # Return the list of dataset files

def clean_file(input_path, cleaned_path):
   """
   Cleans ARFF, TXT, and CSV files by removing extra spaces in comma-separated values or domains.

   :param input_path: Path to the input file (.arff, .txt, .csv).
   :param cleaned_path: Path to save the cleaned file.
   :return: None
   """

   file_extension = os.path.splitext(input_path)[1].lower() # Get the file extension of the input file

   verbose_output(f"{BackgroundColors.GREEN}Cleaning file: {BackgroundColors.CYAN}{input_path}{BackgroundColors.GREEN} and saving to {BackgroundColors.CYAN}{cleaned_path}{Style.RESET_ALL}")

   with open(input_path, "r") as f: # Open the input file for reading
      lines = f.readlines() # Read all lines from the input file

   cleaned_lines = [] # List to store cleaned lines

   if file_extension == ".arff": # If the file is in ARFF format
      for line in lines: # Iterate through each line in the file
         if line.strip().lower().startswith("@attribute") and "{" in line and "}" in line: # If the line is an attribute definition with a domain
            parts = line.split("{") # Split the line into parts at the first occurrence of "{"
            before = parts[0] # Get the part before the domain
            domain = parts[1].split("}")[0] # Get the domain part (between "{" and "}")
            after = line.split("}")[1] # Get the part after the domain

            cleaned_domain = ",".join([val.strip() for val in domain.split(",")]) # Clean the domain by removing extra spaces around comma-separated values
            cleaned_line = f"{before}{{{cleaned_domain}}}{after}" # Reconstruct the cleaned line with the cleaned domain
            cleaned_lines.append(cleaned_line) # Append the cleaned line to the list
         else: # If the line is not an attribute definition with a domain
            cleaned_lines.append(line) # Append the line as is to the cleaned lines list
   elif file_extension in [".txt", ".csv"]: # If the file is in TXT or CSV format
      for line in lines: # Iterate through each line in the file
         cleaned_line = ",".join([val.strip() for val in line.strip().split(",")]) + "\n" # Clean the line by removing extra spaces around comma-separated values
         cleaned_lines.append(cleaned_line) # Append the cleaned line to the cleaned lines list
   else: # If the file extension is not supported
      raise ValueError(f"Unsupported file extension: {file_extension}") # Raise an error for unsupported file extensions

   with open(cleaned_path, "w") as f: # Open the cleaned file for writing
      f.writelines(cleaned_lines) # Write the cleaned lines to the cleaned file

def load_dataset(input_path):
   """
   Load a dataset from a file in CSV, ARFF, or TXT format into a pandas DataFrame.

   :param input_path: Path to the input dataset file.
   :return: pandas DataFrame containing the dataset.
   """

   verbose_output(f"{BackgroundColors.GREEN}Loading dataset from: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}") # Output the verbose message

   _, ext = os.path.splitext(input_path) # Get the file extension of the input file
   ext = ext.lower() # Convert the file extension to lowercase

   if ext == ".csv": # If the file is in CSV format
      df = pd.read_csv(input_path) # Load the CSV file into a pandas DataFrame

   elif ext == ".arff": # If the file is in ARFF format
      try: # Try to load the ARFF file using scipy
         data, meta = scipy_arff.loadarff(input_path) # Try loading the ARFF file using scipy
         df = pd.DataFrame(data) # Convert the loaded data to a pandas DataFrame
         for col in df.columns: # Iterate through each column in the DataFrame
            if df[col].dtype == object: # If the column data type is object (usually for string data)
               df[col] = df[col].apply(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x) # Decode bytes to strings if necessary
      except Exception as e: # If there is an error loading the ARFF file with scipy
         verbose_output(f"{BackgroundColors.YELLOW}Warning: Failed to load ARFF with scipy ({e}). Trying with liac-arff...{Style.RESET_ALL}")
         try: # Try to load the ARFF file using liac-arff
            with open(input_path, "r", encoding="utf-8") as f:
               data = arff.load(f) # Load the ARFF file using liac-arff
            df = pd.DataFrame(data["data"], columns=[attr[0] for attr in data["attributes"]]) # Create DataFrame from liac-arff output
         except Exception as e2: # If there is an error loading the ARFF file with liac-arff
            raise RuntimeError(f"Failed to load ARFF file with both scipy and liac-arff: {e2}")

   elif ext == ".txt": # If the file is in TXT format
      df = pd.read_csv(input_path, sep="\t") # Load the TXT file into a pandas DataFrame using tab as the separator

   else: # If the file extension is not supported
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
   Batch convert dataset files from the input directory to multiple formats (ARFF, CSV, TXT) in the output directory.

   :param INPUT_DIRECTORY: Path to the input directory containing dataset files.
   :param OUTPUT_DIRECTORY: Path to the output directory where converted files will be saved.
   :return: None
   """

   verbose_output(f"{BackgroundColors.GREEN}Batch converting dataset files from {BackgroundColors.CYAN}{input_directory}{BackgroundColors.GREEN} to {BackgroundColors.CYAN}{output_directory}{Style.RESET_ALL}") # Output the verbose message

   dataset_files = get_dataset_files(input_directory) # Get all dataset files from the input directory

   if not dataset_files: # If no dataset files are found
      print(f"{BackgroundColors.RED}No dataset files found in {BackgroundColors.CYAN}{input_directory}{Style.RESET_ALL}")
      return # Exit the function if no dataset files are found

   pbar = tqdm(dataset_files, desc=f"{BackgroundColors.CYAN}Converting files{Style.RESET_ALL}", unit="file", colour="green") # Create a progress bar for the dataset files
   for input_path in pbar: # Iterate through each dataset file
      file = os.path.basename(input_path) # Get the base name of the file
      name, ext = os.path.splitext(file) # Split the file name into name and extension
      ext = ext.lower() # Convert the file extension to lowercase

      pbar.set_postfix_str(f"{BackgroundColors.GREEN}Processing {BackgroundColors.CYAN}{name}{ext}{Style.RESET_ALL}") # Update the progress bar with the current file being processed

      if ext not in [".csv", ".arff", ".txt"]: # If the file extension is not supported
         continue # Skip the file if it is not in CSV, ARFF, or TXT format

      cleaned_path = os.path.join(OUTPUT_DIRECTORY, f"{name}{ext}") # Create the cleaned file path in the output directory
      clean_file(input_path, cleaned_path) # Clean the input file and save it to the cleaned path
      
      df = load_dataset(cleaned_path) # Load the cleaned dataset file into a pandas DataFrame
      convert_to_arff(df, os.path.join(OUTPUT_DIRECTORY, f"{name}.arff")) if ext != ".arff" else None # Convert the DataFrame to ARFF format and save it if the file is not already in ARFF format
      convert_to_csv(df, os.path.join(OUTPUT_DIRECTORY, f"{name}.csv")) if ext != ".csv" else None # Convert the DataFrame to CSV format and save it if the file is not already in CSV format
      convert_to_txt(df, os.path.join(OUTPUT_DIRECTORY, f"{name}.txt")) if ext != ".txt" else None # Convert the DataFrame to TXT format and save it if the file is not already in TXT format

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
