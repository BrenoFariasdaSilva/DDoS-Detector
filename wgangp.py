"""
================================================================================
WGAN-GP Conditional Generator for DDoS Flow Features (CICDDoS2019)
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-11-21
Description :
   Implements a DRCGAN-like conditional Wasserstein GAN with Gradient Penalty
   for tabular flow features from CICDDoS2019 dataset. This module enables
   generating synthetic network flow data conditioned on attack labels.

   Key features include:
      - CSV loader with automatic scaling and label encoding
      - Conditional generator with residual blocks (DRC-style architecture)
      - MLP discriminator (critic) for Wasserstein loss
      - WGAN-GP training loop with gradient penalty for stability
      - Checkpoint saving and synthetic sample generation to CSV
      - Support for multi-class conditional generation

Usage:
   1. Prepare a CSV file with network flow features and labels.
   2. Train the model using the train mode:
         $ python wgangp.py --mode train --csv_path data.csv --epochs 60
   3. Generate synthetic samples using a trained checkpoint:
         $ python wgangp.py --mode gen --checkpoint outputs/generator_epoch60.pt --n_samples 1000

Outputs:
   - outputs/generator_epoch*.pt — Saved generator checkpoints with metadata
   - outputs/discriminator_epoch*.pt — Saved discriminator checkpoints
   - generated.csv — Generated synthetic flow samples (via --mode gen)

TODOs:
   - Implement learning rate scheduling for better convergence
   - Add support for different activation functions
   - Extend feature importance analysis for generated data
   - Add data quality metrics (statistical distance, mode coverage)
   - Implement multi-GPU training support

Dependencies:
   - Python >= 3.9
   - torch >= 1.9.0
   - numpy
   - pandas
   - scikit-learn

Assumptions & Notes:
   - CSV should contain feature columns and a label column
   - Features are automatically scaled using StandardScaler
   - Labels are encoded via LabelEncoder (categorical to integer)
   - Output features are inverse-transformed to original scale
   - CUDA is used if available; use --force_cpu to disable

================================================================================
"""

import argparse # For CLI argument parsing
import atexit # For playing a sound when the program finishes
import numpy as np # Numerical operations
import os # For running a command in the terminal
import pandas as pd # For CSV handling
import platform # For getting the operating system name
import random # For reproducibility
import torch # PyTorch core
import torch.nn as nn # Neural network modules
from colorama import Style # For coloring the terminal
from sklearn.preprocessing import StandardScaler, LabelEncoder # For data preprocessing
from torch import autograd # For gradient penalty
from torch.utils.data import Dataset, DataLoader # Dataset and DataLoader
from typing import Any, List, Optional # For Any type hint

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

def play_sound():
   """
   Plays a sound when the program finishes and skips if the operating system is Windows.

   :param: None
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

   :param: None
   :return: None
   """

   print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}WGAN-GP Data Augmentation{BackgroundColors.GREEN} program!{Style.RESET_ALL}", end="\n\n") # Output the welcome message
   
   args = parse_args() # Parse command-line arguments
   if args.mode == "train": # If training mode is selected
      assert args.csv_path is not None, "Training requires --csv_path" # Ensure CSV path is provided
      train(args) # Run training function
   elif args.mode == "gen": # If generation mode is selected
      assert args.checkpoint is not None, "Generation requires --checkpoint" # Ensure checkpoint is provided
      generate(args) # Run generation function

   print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}") # Output the end of the program message

   atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None # Register the play_sound function to be called when the program finishes

if __name__ == "__main__":
   """
   This is the standard boilerplate that calls the main() function.

   :return: None
   """

   main() # Call the main function
