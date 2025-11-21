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

# Classes Definitions:

class CSVFlowDataset(Dataset):
   """
   Initialize the CSVFlowDataset.
   This class loads flow data from a CSV file, applies scaling to features,

   :param csv_path: Path to CSV file containing flows and labels.
   :param label_col: Column name that contains the class labels.
   :param feature_cols: Optional list of feature column names. If None, all columns except label_col are used.
   :param scaler: Optional pre-fitted StandardScaler to use for features.
   :param label_encoder: Optional pre-fitted LabelEncoder to transform labels.
   :param fit_scaler: If True and scaler is None, fit a new StandardScaler on the data.
   :return: None
   """

   def __init__( # Begin constructor for initializing the dataset
      self, # Instance reference
      csv_path: str, # Path pointing to the CSV file
      label_col: str, # Column containing class labels
      feature_cols: Optional[List[str]] = None, # Optional list of selected features
      scaler: Optional[StandardScaler] = None, # Optional feature scaler
      label_encoder: Optional[LabelEncoder] = None, # Optional label encoder
      fit_scaler: bool = True # Whether to fit scaler on data
   ): # Close constructor signature
      df = pd.read_csv(csv_path) # Load CSV file into a DataFrame

      if feature_cols is None: # When user does not specify features
         feature_cols = [c for c in df.columns if c != label_col] # Select every column except label

      self.label_col = label_col # Save label column name
      self.feature_cols = feature_cols # Save list of feature columns

      self.labels_raw = df[label_col].values.astype(str) # Extract raw labels and convert to strings

      self.labels: Any # Must be Any or Pylance will error

      if label_encoder is None: # If no label encoder is given
         self.label_encoder = LabelEncoder() # Create a fresh label encoder
         self.labels = self.label_encoder.fit_transform(self.labels_raw) # Fit encoder and encode labels
      else: # If encoder is provided
         self.label_encoder = label_encoder # Store provided encoder
         self.labels = self.label_encoder.transform(self.labels_raw) # Encode labels with given encoder

      X = df[feature_cols].values.astype(np.float32) # Extract features and cast to float32

      if scaler is None: # If no scaler is provided
         self.scaler = StandardScaler() # Instantiate a default scaler
         if fit_scaler: # Fit scaler when requested
            self.X = self.scaler.fit_transform(X) # Fit and transform features
         else: # Do not fit scaler
            self.X = self.scaler.transform(X) # Only transform features
      else: # Scaler is provided
         self.scaler = scaler # Store provided scaler
         self.X = self.scaler.transform(X) # Transform features with external scaler

      self.n_classes = len(self.label_encoder.classes_) # Count number of unique classes
      self.feature_dim = self.X.shape[1] # Determine dimensionality of features

   def __len__(self): # Return number of samples in the dataset
      return len(self.X) # Return number of feature vectors

   def __getitem__(self, idx): # Fetch one item by index
      x = self.X[idx] # Get feature row
      y = int(self.labels[idx]) # Get encoded label
      return x, y # Return (features, label)

class ResidualBlockFC(nn.Module):
   """
   Simple fully-connected residual block used in the generator.
   
   :param dim: input and output dimensionality of the block
   """
   
   def __init__(self, dim): # Constructor taking the input/output dimension
      """
      Simple residual fully-connected block used in the generator.

      :param dim: input and output dimensionality of the block
      """
      super().__init__() # Initialize the parent nn.Module class

      self.net = nn.Sequential( # Define the residual transformation path
         nn.Linear(dim, dim), # First linear projection
         nn.BatchNorm1d(dim), # Normalize activations
         nn.LeakyReLU(0.2, inplace=True), # Apply nonlinearity
         nn.Linear(dim, dim), # Second linear projection
         nn.BatchNorm1d(dim), # Second batch normalization
      ) # End of sequential block

      self.act = nn.LeakyReLU(0.2, inplace=True) # Activation after merging residual shortcut

   def forward(self, x): # Forward computation of the block
      out = self.net(x) # Compute residual branch output
      out = out + x # Apply skip connection
      return self.act(out) # Apply activation to merged result

class Generator(nn.Module):
   """
   Conditional generator: input z + label embedding (one-hot or embedding), outputs feature vector.
   Uses residual blocks internally (DRC-style).
   
   :param latent_dim: dimensionality of input noise vector
   """

   def __init__( # Constructor defining generator architecture
      self,
      latent_dim: int, # Dimensionality of latent noise vector
      feature_dim: int, # Dimensionality of generated features
      n_classes: int, # Number of label classes
      hidden_dims: Optional[List[int]] = None, # Optional hidden layer configuration
      embed_dim: int = 32, # Size of label embedding
      n_resblocks: int = 3 # Number of residual blocks to apply
   ): # End constructor signature
      """
      Conditional generator that maps (z, y) -> feature vector.

      :param latent_dim: dimensionality of noise vector z
      :param feature_dim: dimensionality of output feature vector
      :param n_classes: number of conditioning classes
      :param hidden_dims: list of hidden layer sizes for initial MLP
      :param embed_dim: size of label embedding
      :param n_resblocks: number of residual blocks to apply
      """

      super().__init__() # Initialize module internals

      if hidden_dims is None: # Use default architecture if none given
         hidden_dims = [256, 512] # Default MLP layer widths

      self.latent_dim = latent_dim # Store latent input size
      self.feature_dim = feature_dim # Store output size
      self.n_classes = n_classes # Store number of classes
      self.embed = nn.Embedding(n_classes, embed_dim) # Create label embedding table

      input_dim = latent_dim + embed_dim # Combined dimension of noise + embedding
      layers = [] # Container for MLP layers
      prev = input_dim # Track previous layer width

      for h in hidden_dims: # Build MLP layers
         layers.append(nn.Linear(prev, h)) # Add linear layer
         layers.append(nn.BatchNorm1d(h)) # Normalize activations
         layers.append(nn.LeakyReLU(0.2, inplace=True)) # Apply activation
         prev = h # Update width tracker

      res_dim = prev # Width entering residual blocks
      self.pre = nn.Sequential(*layers) # Store assembled MLP

      self.resblocks = nn.ModuleList( # Build list of residual blocks
         [ResidualBlockFC(res_dim) for _ in range(n_resblocks)] # Create required count of blocks
      ) # End block list

      self.out = nn.Sequential( # Output mapping layer
         nn.Linear(res_dim, feature_dim), # Final linear projection
      ) # End output block

   def forward(self, z, y): # Compute generator output
      y_e = self.embed(y) # Convert class ID to embedding
      x = torch.cat([z, y_e], dim=1) # Concatenate noise and embedding
      x = self.pre(x) # Process through MLP
      for b in self.resblocks: # Loop through residual blocks
         x = b(x) # Apply block
      out = self.out(x) # Produce final feature vector
      return out # Return generated sample

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
