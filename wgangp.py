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

class Discriminator(nn.Module):
   """
   Conditional critic/discriminator: takes feature vector concatenated with label embedding.
   Returns scalar score (Wasserstein critic).
   
   :param feature_dim: dimensionality of input feature vectors
   """

   def __init__( # Constructor for critic network
      self,
      feature_dim: int, # Dimensionality of feature vectors
      n_classes: int, # Number of class labels
      hidden_dims: Optional[List[int]] = None, # Optional critic architecture
      embed_dim: int = 32 # Embedding dimension for labels
   ): # End signature
      """
      Conditional critic/discriminator network that scores (x, y).

      :param feature_dim: dimensionality of input feature vector
      :param n_classes: number of classes for conditioning
      :param hidden_dims: list of hidden layer sizes
      :param embed_dim: dimensionality of label embedding
      """

      super().__init__() # Initialize discriminator internals

      if hidden_dims is None: # Assign default architecture when unspecified
         hidden_dims = [512, 256, 128] # Standard critic hierarchy

      self.embed = nn.Embedding(n_classes, embed_dim) # Store label embedding table

      input_dim = feature_dim + embed_dim # Combined input dimension
      layers = [] # List to accumulate layers
      prev = input_dim # Initialize previous width

      for h in hidden_dims: # Build critic layers
         layers.append(nn.Linear(prev, h)) # Linear transformation
         layers.append(nn.LeakyReLU(0.2, inplace=True)) # Activation function
         prev = h # Update width tracker

      layers.append(nn.Linear(prev, 1)) # Output layer producing scalar score
      self.net = nn.Sequential(*layers) # Create critic network

   def forward(self, x, y): # Compute critic score
      y_e = self.embed(y) # Convert label to embedding
      inp = torch.cat([x, y_e], dim=1) # Join features with embedding
      return self.net(inp).squeeze(1) # Produce scalar score

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

def set_seed(seed: int):
   """
   Sets random seeds for reproducibility across all libraries.

   :param seed: The seed value to use for all random number generators
   :return: None
   """

   random.seed(seed) # Set Python random seed for reproducibility
   np.random.seed(seed) # Set NumPy random seed for reproducibility
   torch.manual_seed(seed) # Set PyTorch CPU seed for reproducibility
   torch.cuda.manual_seed_all(seed) # Set CUDA seed for all devices

def train(args):
   """
   Train the WGAN-GP model using the provided arguments.

   :param args: parsed arguments namespace containing training configuration
   :return: None
   """

   device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu") # Select device for training
   set_seed(args.seed) # Set random seed for reproducibility

   dataset = CSVFlowDataset(args.csv_path, label_col=args.label_col, feature_cols=args.feature_cols) # Load dataset from CSV
   dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=4) # Create dataloader for batching

   feature_dim = dataset.feature_dim # Get feature dimensionality from dataset
   n_classes = dataset.n_classes # Get number of label classes from dataset

   G = Generator(latent_dim=args.latent_dim, feature_dim=feature_dim, n_classes=n_classes,
      hidden_dims=args.g_hidden, embed_dim=args.embed_dim, n_resblocks=args.n_resblocks).to(device) # Initialize generator model
   D = Discriminator(feature_dim=feature_dim, n_classes=n_classes,
      hidden_dims=args.d_hidden, embed_dim=args.embed_dim).to(device) # Initialize discriminator model

   opt_D = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(args.beta1, args.beta2)) # Create optimizer for discriminator
   opt_G = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(args.beta1, args.beta2)) # Create optimizer for generator

   fixed_noise = torch.randn(args.sample_batch, args.latent_dim, device=device) # Generate fixed noise for inspection
   fixed_labels = torch.randint(0, n_classes, (args.sample_batch,), device=device) # Generate fixed labels for inspection

   os.makedirs(args.out_dir, exist_ok=True) # Ensure output directory exists
   step = 0 # Initialize global step counter

   for epoch in range(args.epochs): # Loop over epochs
      for real_x_np, labels_np in dataloader: # Loop over batches in dataloader
         real_x = real_x_np.to(device) # Move real features to device
         labels = labels_np.to(device, dtype=torch.long) # Move labels to device and set type

         loss_D = torch.tensor(0.0, device=device) # Initialize discriminator loss
         gp = torch.tensor(0.0, device=device) # Initialize gradient penalty
         for _ in range(args.critic_steps): # Train discriminator multiple steps
            z = torch.randn(args.batch_size, args.latent_dim, device=device) # Sample noise for discriminator step
            fake_x = G(z, labels).detach() # Generate fake samples and detach for discriminator
            d_real = D(real_x, labels) # Get discriminator score for real samples
            d_fake = D(fake_x, labels) # Get discriminator score for fake samples
            gp = gradient_penalty(D, real_x, fake_x, labels, device) # Compute gradient penalty

            loss_D = d_fake.mean() - d_real.mean() + args.lambda_gp * gp # Calculate WGAN-GP discriminator loss

            opt_D.zero_grad() # Zero discriminator gradients
            loss_D.backward() # Backpropagate discriminator loss
            opt_D.step() # Update discriminator parameters

         z = torch.randn(args.batch_size, args.latent_dim, device=device) # Sample noise for generator step
         gen_labels = torch.randint(0, n_classes, (args.batch_size,), device=device) # Sample labels for generator
         fake_x = G(z, gen_labels) # Generate fake samples with generator
         g_loss = -D(fake_x, gen_labels).mean() # Calculate generator loss

         opt_G.zero_grad() # Zero generator gradients
         g_loss.backward() # Backpropagate generator loss
         opt_G.step() # Update generator parameters

         if step % args.log_interval == 0: # Log training progress periodically
            print(f"[Epoch {epoch}/{args.epochs}] step {step} | loss_D: {loss_D.item():.4f} | loss_G: {g_loss.item():.4f} | gp: {gp.item():.4f}") # Print training status
         step += 1 # Increment global step counter

      if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1: # Save checkpoints periodically
         g_path = os.path.join(args.out_dir, f"generator_epoch{epoch+1}.pt") # Path for generator checkpoint
         d_path = os.path.join(args.out_dir, f"discriminator_epoch{epoch+1}.pt") # Path for discriminator checkpoint
         torch.save({
            "epoch": epoch + 1, # Save current epoch number
            "state_dict": G.state_dict(), # Save generator state dict
            "scaler": dataset.scaler, # Save scaler for inverse transform
            "label_encoder": dataset.label_encoder, # Save label encoder for mapping
            "args": vars(args) # Save training arguments
         }, g_path) # Save generator checkpoint to disk
         torch.save({
            "epoch": epoch + 1, # Save current epoch number
            "state_dict": D.state_dict(), # Save discriminator state dict
            "args": vars(args) # Save training arguments
         }, d_path) # Save discriminator checkpoint to disk
         torch.save(G.state_dict(), os.path.join(args.out_dir, "generator_latest.pt")) # Save latest generator weights
         print(f"Saved generator to {g_path}") # Print checkpoint save message

   print("Training finished.") # Print final training completion message

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
