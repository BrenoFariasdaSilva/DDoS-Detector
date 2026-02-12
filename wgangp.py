"""
================================================================================
Wasserstein Generative Adversarial Networks with Gradient Penalty (WGAN-GP) for CICDDoS2019 Tabular Flow Features (wgangp.py)
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
"""

import argparse  # For CLI argument parsing
import atexit  # For playing a sound when the program finishes
import datetime  # For tracking execution time
import json  # For saving/loading metrics history
import matplotlib.pyplot as plt  # For plotting training metrics
import numpy as np  # Numerical operations
import os  # For running a command in the terminal
import pandas as pd  # For CSV handling
import platform  # For getting the operating system name
import random  # For reproducibility
import sys  # For system-specific parameters and functions
import telegram_bot as telegram_module  # For setting Telegram prefix and device info
import torch  # PyTorch core
import torch.nn as nn  # Neural network modules
import traceback  # For printing tracebacks on exceptions
from colorama import Style  # For coloring the terminal
from contextlib import nullcontext  # For null context manager
from Logger import Logger  # For logging output to both terminal and file
from pathlib import Path  # For handling file paths
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For data preprocessing
from telegram_bot import TelegramBot, send_telegram_message  # For sending progress messages to Telegram
from torch import autograd  # For gradient penalty
from torch.utils.data import DataLoader, Dataset  # Dataset and DataLoader
from tqdm import tqdm  # For progress bar visualization
from typing import Any, List, Optional, cast  # For Any type hint and cast

# Prefer CUDA autocast when available; provide a safe fallback context manager
try:
    from torch.amp.autocast_mode import autocast as _torch_autocast
except Exception:
    _torch_autocast = None


def autocast(device_type: str, enabled: bool = True):
    """Return an autocast context manager when enabled on CUDA, else a nullcontext.

    This avoids referencing `torch.amp.autocast` directly (Pylance warning) and
    supports environments without CUDA.
    """

    if enabled and device_type == "cuda" and _torch_autocast is not None:
        return _torch_autocast(device_type)
    return nullcontext()


# Macros:
class BackgroundColors:  # Colors for the terminal
    CYAN = "\033[96m"  # Cyan
    GREEN = "\033[92m"  # Green
    YELLOW = "\033[93m"  # Yellow
    RED = "\033[91m"  # Red
    BOLD = "\033[1m"  # Bold
    UNDERLINE = "\033[4m"  # Underline
    CLEAR_TERMINAL = "\033[H\033[J"  # Clear the terminal


# Execution Constants:
VERBOSE = False  # Set to True to output verbose messages
RESULTS_SUFFIX = "_data_augmented"  # Suffix to add to generated filenames
MATCH_FILENAMES_TO_PROCESS = [""]  # List of specific filenames to search for a match (set to None to process all files)
IGNORE_FILES = [RESULTS_SUFFIX]  # List of filename substrings to ignore when searching for datasets
IGNORE_DIRS = [
    "Classifiers",
    "Classifiers_Hyperparameters",
    "Dataset_Description",
    "Data_Separability",
    "Feature_Analysis",
]  # List of directory names to ignore when searching for datasets

DATASETS = {  # Dictionary containing dataset paths and feature files
    "CICDDoS2019-Dataset": [  # List of paths to the CICDDoS2019 dataset
        "./Datasets/CICDDoS2019/01-12/",
        "./Datasets/CICDDoS2019/03-11/",
    ],
}

# Telegram Bot Setup:
TELEGRAM_BOT = None  # Global Telegram bot instance (initialized in setup_telegram_bot)

# Logger Setup:
logger = Logger(f"./Logs/{Path(__file__).stem}.log", clean=True)  # Create a Logger instance
sys.stdout = logger  # Redirect stdout to the logger
sys.stderr = logger  # Redirect stderr to the logger

# Sound Constants:
SOUND_COMMANDS = {
    "Darwin": "afplay",
    "Linux": "aplay",
    "Windows": "start",
}  # The commands to play a sound for each operating system
SOUND_FILE = "./.assets/Sounds/NotificationSound.wav"  # The path to the sound file

# RUN_FUNCTIONS:
RUN_FUNCTIONS = {
    "Play Sound": True,  # Set to True to play a sound when the program finishes
}

# Functions Definitions:

def detect_label_column(columns):
    """
    Try to guess the label column based on common naming conventions.

    :param columns: List of column names
    :return: The name of the label column if found, else None
    """

    candidates = ["label", "class", "target"]  # Common label column names

    for col in columns:  # First search for exact matches
        if col.lower() in candidates:  # Verify if the column name matches any candidate exactly
            return col  # Return the column name if found

    for col in columns:  # Second search for partial matches
        if "target" in col.lower() or "label" in col.lower():  # Verify if the column name contains any candidate
            return col  # Return the column name if found

    return None  # Return None if no label column is found


def preprocess_dataframe(df, label_col, remove_zero_variance=True):
    """
    Preprocess a DataFrame by:
    1. Selecting only numeric feature columns (excluding label)
    2. Removing rows with NaN or infinite values
    3. Optionally dropping zero-variance numeric features

    :param df: pandas DataFrame to preprocess
    :param label_col: name of the label column to preserve
    :param remove_zero_variance: whether to drop numeric columns with zero variance
    :return: cleaned DataFrame with only numeric features and the label column
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Preprocessing DataFrame: selecting numeric features, removing NaN/inf, handling zero-variance.{Style.RESET_ALL}"
    )  # Output verbose message

    if df is None:  # If the DataFrame is None
        return df  # Return None

    # Strip whitespace from all column names
    df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names

    # Separate label column
    labels = df[label_col].copy()  # Save labels

    # Select only numeric columns (excluding label)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # Get numeric column names
    if label_col in numeric_cols:  # If label is numeric, remove it from features
        numeric_cols.remove(label_col)  # Remove label from feature list

    verbose_output(
        f"{BackgroundColors.GREEN}Found {len(numeric_cols)} numeric feature columns out of {len(df.columns)-1} total features.{Style.RESET_ALL}"
    )  # Output count

    # Create DataFrame with only numeric features
    df_numeric = df[numeric_cols].copy()  # Select numeric features

    # Replace infinite values with NaN, then drop rows with NaN
    df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
    valid_mask = ~df_numeric.isna().any(axis=1)  # Mask for rows without NaN

    df_clean = df_numeric[valid_mask].copy()  # Keep only valid rows
    labels_clean = labels[valid_mask].copy()  # Keep corresponding labels

    rows_dropped = len(df) - len(df_clean)  # Calculate dropped rows
    if rows_dropped > 0:  # If rows were dropped
        verbose_output(
            f"{BackgroundColors.YELLOW}Dropped {rows_dropped} rows with NaN/inf values ({rows_dropped/len(df)*100:.2f}%).{Style.RESET_ALL}"
        )  # Output warning

    # Remove zero-variance features if requested
    if remove_zero_variance and len(df_clean) > 0:  # If removal enabled and data remains
        variances = df_clean.var(axis=0, ddof=0)  # Calculate column variances
        zero_var_cols = variances[variances == 0].index.tolist()  # Get zero-variance columns
        if zero_var_cols:  # If zero-variance columns exist
            verbose_output(
                f"{BackgroundColors.YELLOW}Dropping {len(zero_var_cols)} zero-variance columns.{Style.RESET_ALL}"
            )  # Output warning
            df_clean = df_clean.drop(columns=zero_var_cols)  # Drop zero-variance columns

    # Add label column back
    df_clean[label_col] = labels_clean.values  # Restore labels

    return df_clean  # Return cleaned DataFrame


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

    def __init__(  # Begin constructor for initializing the dataset
        self,  # Instance reference
        csv_path: str,  # Path pointing to the CSV file
        label_col: str,  # Column containing class labels
        feature_cols: Optional[List[str]] = None,  # Optional list of selected features
        scaler: Optional[StandardScaler] = None,  # Optional feature scaler
        label_encoder: Optional[LabelEncoder] = None,  # Optional label encoder
        fit_scaler: bool = True,  # Whether to fit scaler on data
    ):  # Close constructor signature
        df = pd.read_csv(csv_path, low_memory=False)  # Load CSV file into a DataFrame with low_memory=False to avoid DtypeWarning
        
        # Strip whitespace from column names immediately after loading
        df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names

        # Auto-detect label column if the specified one doesn't exist
        if label_col not in df.columns:  # If the specified label column is not found
            detected_col = detect_label_column(df.columns)  # Try to detect the label column
            if detected_col is not None:  # If a label column was detected
                print(f"{BackgroundColors.YELLOW}Warning: Label column '{label_col}' not found. Using detected column: '{detected_col}'{Style.RESET_ALL}")  # Warn user
                label_col = detected_col  # Use the detected column
            else:  # If no label column was detected
                raise ValueError(f"Label column '{label_col}' not found in CSV. Available columns: {list(df.columns)}")  # Raise error

        # Preprocess DataFrame: select numeric features, remove NaN/inf, handle zero-variance
        df = preprocess_dataframe(df, label_col, remove_zero_variance=True)  # Clean and filter DataFrame

        if len(df) == 0:  # If all rows were dropped
            raise ValueError(f"No valid data remaining after preprocessing {csv_path}")  # Raise error

        # Get available numeric feature columns (excluding label)
        available_features = [c for c in df.columns if c != label_col]  # List numeric features

        if feature_cols is None:  # When user does not specify features
            feature_cols = available_features  # Use all available numeric features
        else:  # User specified features
            # Filter to only include features that exist and are numeric
            feature_cols = [c for c in feature_cols if c in available_features]  # Keep valid features
            if not feature_cols:  # If no valid features remain
                raise ValueError(f"None of the specified feature columns are numeric or available in {csv_path}")  # Raise error

        self.label_col = label_col  # Save label column name
        self.feature_cols = feature_cols  # Save list of feature columns

        # Ensure labels_raw is a plain numpy array of Python strings to satisfy type checkers
        self.labels_raw: np.ndarray = np.asarray(df[label_col].values, dtype=str)

        self.labels: Any  # Must be Any or Pylance will error

        # Use a normalized numpy array for LabelEncoder input
        labels_arr = np.asarray(self.labels_raw, dtype=str)

        if label_encoder is None:  # If no label encoder is given
            self.label_encoder = LabelEncoder()  # Create a fresh label encoder
            self.labels = self.label_encoder.fit_transform(labels_arr)  # Fit encoder and encode labels
        else:  # If encoder is provided
            self.label_encoder = label_encoder  # Store provided encoder
            self.labels = self.label_encoder.transform(labels_arr)  # Encode labels with given encoder

        X = df[feature_cols].values.astype(np.float32)  # Extract features and cast to float32

        if scaler is None:  # If no scaler is provided
            self.scaler = StandardScaler()  # Instantiate a default scaler
            if fit_scaler:  # Fit scaler when requested
                self.X = self.scaler.fit_transform(X)  # Fit and transform features
            else:  # Do not fit scaler
                self.X = self.scaler.transform(X)  # Only transform features
        else:  # Scaler is provided
            self.scaler = scaler  # Store provided scaler
            self.X = self.scaler.transform(X)  # Transform features with external scaler

        self.n_classes = len(self.label_encoder.classes_)  # Count number of unique classes
        self.feature_dim = self.X.shape[1]  # Determine dimensionality of features

    def __len__(self):  # Return number of samples in the dataset
        return len(self.X)  # Return number of feature vectors

    def __getitem__(self, idx):  # Fetch one item by index
        x = self.X[idx]  # Get feature row
        y = int(self.labels[idx])  # Get encoded label
        return x, y  # Return (features, label)


class ResidualBlockFC(nn.Module):
    """
    Simple fully-connected residual block used in the generator.

    :param dim: input and output dimensionality of the block
    """

    def __init__(self, dim):  # Constructor taking the input/output dimension
        """
        Initialize a residual fully-connected block for the generator.

        :param dim: Input and output dimensionality of the block
        :return: None
        """

        super().__init__()  # Initialize the parent nn.Module class

        self.net = nn.Sequential(  # Define the residual transformation path
            nn.Linear(dim, dim),  # First linear projection
            nn.BatchNorm1d(dim),  # Normalize activations
            nn.LeakyReLU(0.2, inplace=True),  # Apply nonlinearity
            nn.Linear(dim, dim),  # Second linear projection
            nn.BatchNorm1d(dim),  # Second batch normalization
        )  # End of sequential block

        self.act = nn.LeakyReLU(0.2, inplace=True)  # Activation after merging residual shortcut

    def forward(self, x):  # Forward computation of the block
        """
        Perform forward pass through the residual block.

        :param x: Input tensor of shape (batch_size, dim)
        :return: Output tensor after residual connection and activation
        """

        out = self.net(x)  # Compute residual branch output
        out = out + x  # Apply skip connection
        return self.act(out)  # Apply activation to merged result


class Generator(nn.Module):
    """
    Conditional generator: input z + label embedding (one-hot or embedding), outputs feature vector.
    Uses residual blocks internally (DRC-style).

    :param latent_dim: dimensionality of input noise vector
    """

    def __init__(
        self,
        latent_dim: int,
        feature_dim: int,
        n_classes: int,
        hidden_dims: Optional[List[int]] = None,
        embed_dim: int = 32,
        n_resblocks: int = 3,
    ):
        """
        Initialize conditional generator that maps (z, y) -> feature vector.

        :param latent_dim: Dimensionality of noise vector z
        :param feature_dim: Dimensionality of output feature vector
        :param n_classes: Number of conditioning classes
        :param hidden_dims: List of hidden layer sizes for initial MLP (default: [256, 512])
        :param embed_dim: Size of label embedding (default: 32)
        :param n_resblocks: Number of residual blocks to apply (default: 3)
        :return: None
        """

        super().__init__()  # Initialize module internals

        if hidden_dims is None:  # Use default architecture if none given
            hidden_dims = [256, 512]  # Default MLP layer widths

        self.latent_dim = latent_dim  # Store latent input size
        self.feature_dim = feature_dim  # Store output size
        self.n_classes = n_classes  # Store number of classes
        self.embed = nn.Embedding(n_classes, embed_dim)  # Create label embedding table

        input_dim = latent_dim + embed_dim  # Combined dimension of noise + embedding
        layers = []  # Container for MLP layers
        prev = input_dim  # Track previous layer width

        for h in hidden_dims:  # Build MLP layers
            layers.append(nn.Linear(prev, h))  # Add linear layer
            layers.append(nn.BatchNorm1d(h))  # Normalize activations
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # Apply activation
            prev = h  # Update width tracker

        res_dim = prev  # Width entering residual blocks
        self.pre = nn.Sequential(*layers)  # Store assembled MLP

        self.resblocks = nn.ModuleList(  # Build list of residual blocks
            [ResidualBlockFC(res_dim) for _ in range(n_resblocks)]  # Create required count of blocks
        )  # End block list

        self.out = nn.Sequential(  # Output mapping layer
            nn.Linear(res_dim, feature_dim),  # Final linear projection
        )  # End output block

    def forward(self, z, y):  # Compute generator output
        """
        Generate synthetic features conditioned on labels.

        :param z: Noise tensor of shape (batch_size, latent_dim)
        :param y: Label tensor of shape (batch_size,) containing class indices
        :return: Generated feature tensor of shape (batch_size, feature_dim)
        """

        y_e = self.embed(y)  # Convert class ID to embedding
        x = torch.cat([z, y_e], dim=1)  # Concatenate noise and embedding
        x = self.pre(x)  # Process through MLP
        for b in self.resblocks:  # Loop through residual blocks
            x = b(x)  # Apply block
        out = self.out(x)  # Produce final feature vector
        return out  # Return generated sample


class Discriminator(nn.Module):
    """
    Conditional critic/discriminator: takes feature vector concatenated with label embedding.
    Returns scalar score (Wasserstein critic).

    :param feature_dim: dimensionality of input feature vectors
    """

    def __init__(
        self,
        feature_dim: int,
        n_classes: int,
        hidden_dims: Optional[List[int]] = None,
        embed_dim: int = 32,
    ):
        """
        Initialize conditional critic/discriminator network that scores (x, y).

        :param feature_dim: Dimensionality of input feature vector
        :param n_classes: Number of classes for conditioning
        :param hidden_dims: List of hidden layer sizes (default: [512, 256, 128])
        :param embed_dim: Dimensionality of label embedding (default: 32)
        :return: None
        """

        super().__init__()  # Initialize discriminator internals

        if hidden_dims is None:  # Assign default architecture when unspecified
            hidden_dims = [512, 256, 128]  # Standard critic hierarchy

        self.embed = nn.Embedding(n_classes, embed_dim)  # Store label embedding table

        input_dim = feature_dim + embed_dim  # Combined input dimension
        layers = []  # List to accumulate layers
        prev = input_dim  # Initialize previous width

        for h in hidden_dims:  # Build critic layers
            layers.append(nn.Linear(prev, h))  # Linear transformation
            layers.append(nn.LeakyReLU(0.2, inplace=True))  # Activation function
            prev = h  # Update width tracker

        layers.append(nn.Linear(prev, 1))  # Output layer producing scalar score
        self.net = nn.Sequential(*layers)  # Create critic network

    def forward(self, x, y):  # Compute critic score
        """
        Compute critic score for input features conditioned on labels.

        :param x: Feature tensor of shape (batch_size, feature_dim)
        :param y: Label tensor of shape (batch_size,) containing class indices
        :return: Scalar critic score tensor of shape (batch_size,)
        """

        y_e = self.embed(y)  # Convert label to embedding
        inp = torch.cat([x, y_e], dim=1)  # Join features with embedding
        return self.net(inp).squeeze(1)  # Produce scalar score


# Functions Definitions:


def verbose_output(true_string="", false_string=""):
    """
    Outputs a message if the VERBOSE constant is set to True.

    :param true_string: The string to be outputted if the VERBOSE constant is set to True.
    :param false_string: The string to be outputted if the VERBOSE constant is set to False.
    :return: None
    """

    if VERBOSE and true_string != "":  # If VERBOSE is True and a true_string was provided
        print(true_string)  # Output the true statement string
    elif false_string != "":  # If a false_string was provided
        print(false_string)  # Output the false statement string


def verify_dot_env_file():
    """
    Verifies if the .env file exists in the current directory.

    :return: True if the .env file exists, False otherwise
    """

    env_path = Path(__file__).parent / ".env"  # Path to the .env file
    if not env_path.exists():  # If the .env file does not exist
        print(f"{BackgroundColors.CYAN}.env{BackgroundColors.YELLOW} file not found at {BackgroundColors.CYAN}{env_path}{BackgroundColors.YELLOW}. Telegram messages may not be sent.{Style.RESET_ALL}")
        return False  # Return False

    return True  # Return True if the .env file exists


def setup_telegram_bot():
    """
    Sets up the Telegram bot for progress messages.

    :return: None
    """
    
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


def parse_args():
    """
    Parse command-line arguments and return namespace.

    :return: parsed arguments namespace
    """

    p = argparse.ArgumentParser(description="DRCGAN-like WGAN-GP for CICDDoS2019 features")  # Create argument parser
    p.add_argument("--mode", choices=["train", "gen", "both"], default="both", help="Mode: train, gen, or both (default: both)")  # Add mode argument
    p.add_argument(
        "--csv_path", type=str, default=None, help="Path to CSV (training data) - required for training."
    )  # Add CSV path argument
    p.add_argument(
        "--label_col", type=str, default="Label", help="Column name for class label"
    )  # Add label column argument
    p.add_argument(
        "--feature_cols",
        nargs="+",
        default=None,
        help="List of feature column names (if omitted, use all except label)",
    )  # Add feature columns argument
    p.add_argument(
        "--out_dir", type=str, default="outputs", help="Where to save models/logs"
    )  # Add output directory argument
    p.add_argument("--epochs", type=int, default=60)  # Add epochs argument
    p.add_argument("--batch_size", type=int, default=64)  # Add batch size argument
    p.add_argument("--latent_dim", type=int, default=100)  # Add latent dimension argument
    p.add_argument("--g_hidden", nargs="+", type=int, default=[256, 512])  # Add generator hidden layers argument
    p.add_argument(
        "--d_hidden", nargs="+", type=int, default=[512, 256, 128]
    )  # Add discriminator hidden layers argument
    p.add_argument("--embed_dim", type=int, default=32)  # Add embedding dimension argument
    p.add_argument("--n_resblocks", type=int, default=3)  # Add number of residual blocks argument
    p.add_argument("--critic_steps", type=int, default=5)  # Add critic steps argument
    p.add_argument("--lr", type=float, default=1e-4)  # Add learning rate argument
    p.add_argument("--beta1", type=float, default=0.5)  # Add beta1 argument for Adam optimizer
    p.add_argument("--beta2", type=float, default=0.9)  # Add beta2 argument for Adam optimizer
    p.add_argument("--lambda_gp", type=float, default=10.0, dest="lambda_gp")  # Add gradient penalty lambda argument
    p.add_argument("--seed", type=int, default=42)  # Add seed argument for reproducibility
    p.add_argument("--save_every", type=int, default=5)  # Add save frequency argument
    p.add_argument("--log_interval", type=int, default=50)  # Add log interval argument
    p.add_argument("--sample_batch", type=int, default=16)  # Add sample batch argument
    p.add_argument("--force_cpu", action="store_true")  # Add force CPU argument
    p.add_argument(
        "--checkpoint", type=str, default=None, help="Path to generator checkpoint for generation"
    )  # Add checkpoint argument
    p.add_argument(
        "--n_samples",
        type=float,
        default=1.0,
        help="Number of samples to generate. If > 1, absolute count. If <= 1, percentage of training data per class (1.0 = 100%%, default: 1.0)"
    )  # Add number of samples argument (supports both absolute and percentage modes)
    p.add_argument(
        "--label", type=int, default=None, help="If set, generate samples for this class id only"
    )  # Add label argument for generation
    p.add_argument("--out_file", type=str, default="generated.csv")  # Add output file argument
    p.add_argument("--gen_batch_size", type=int, default=256)  # Add generation batch size argument
    p.add_argument(
        "--feature_dim", type=int, default=None, help="If known, supply feature dim"
    )  # Add feature dimension argument
    p.add_argument("--gen_only", action="store_true")  # Add generation only flag
    p.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers (default: 8)")  # Add num_workers argument
    p.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision for faster training")  # Add AMP flag
    p.add_argument("--compile", action="store_true", help="Use torch.compile() for faster execution (PyTorch 2.0+)")  # Add compile flag
    p.add_argument("--from_scratch", action="store_true", help="Force training from scratch, ignoring existing checkpoints")  # Add from-scratch flag
    return p.parse_args()  # Parse arguments and return namespace


def set_seed(seed: int):
    """
    Sets random seeds for reproducibility across all libraries.

    :param seed: The seed value to use for all random number generators
    :return: None
    """

    random.seed(seed)  # Set Python random seed for reproducibility
    np.random.seed(seed)  # Set NumPy random seed for reproducibility
    torch.manual_seed(seed)  # Set PyTorch CPU seed for reproducibility
    torch.cuda.manual_seed_all(seed)  # Set CUDA seed for all devices


def verify_filepath_exists(filepath):
    """
    Verify if a file or folder exists at the specified path.

    :param filepath: Path to the file or folder
    :return: True if the file or folder exists, False otherwise
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Verifying if the file or folder exists at the path: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}"
    )  # Output the verbose message

    return os.path.exists(filepath)  # Return True if the file or folder exists, False otherwise


def get_files_to_process(directory_path, file_extension=".csv"):
    """
    Collect all files with a given extension inside a directory (non-recursive).

    Performs validation, respects IGNORE_FILES, and optionally filters by
    MATCH_FILENAMES_TO_PROCESS when defined.

    :param directory_path: Path to the directory to scan
    :param file_extension: File extension to include (default: ".csv")
    :return: Sorted list of matching file paths
    """

    verbose_output(
        f"{BackgroundColors.GREEN}Getting all {BackgroundColors.CYAN}{file_extension}{BackgroundColors.GREEN} files in: {BackgroundColors.CYAN}{directory_path}{Style.RESET_ALL}"
    )  # Verbose: starting file collection
    verify_filepath_exists(directory_path)  # Validate directory path exists

    if not os.path.isdir(directory_path):  # Check if path is a valid directory
        verbose_output(
            f"{BackgroundColors.RED}Not a directory: {BackgroundColors.CYAN}{directory_path}{Style.RESET_ALL}"
        )  # Verbose: invalid directory
        return []  # Return empty list for invalid paths

    try:  # Attempt to read MATCH_FILENAMES_TO_PROCESS if defined
        match_names = (
            set(MATCH_FILENAMES_TO_PROCESS) if MATCH_FILENAMES_TO_PROCESS not in ([], [""], [" "]) else None
        )  # Load match list or None
        if match_names:  # If filtering is to be applied
            verbose_output(
                f"{BackgroundColors.GREEN}Filtering to filenames: {BackgroundColors.CYAN}{match_names}{Style.RESET_ALL}"
            )  # Verbose: applying filename filter
    except NameError:  # MATCH_FILENAMES_TO_PROCESS not defined
        match_names = None  # No filtering will be applied

    files = []  # Accumulator for valid files

    for item in os.listdir(directory_path):  # Iterate directory entries
        item_path = os.path.join(directory_path, item)  # Absolute path
        filename = os.path.basename(item_path)  # Extract just the filename

        if any(ignore == filename or ignore == item_path for ignore in IGNORE_FILES):  # Check if file is in ignore list
            verbose_output(
                f"{BackgroundColors.YELLOW}Ignoring {BackgroundColors.CYAN}{filename}{BackgroundColors.YELLOW} (listed in IGNORE_FILES){Style.RESET_ALL}"
            )  # Verbose: ignoring file
            continue  # Skip ignored file

        if os.path.isfile(item_path) and item.lower().endswith(file_extension):  # File matches extension requirement
            if (
                match_names is not None and filename not in match_names
            ):  # Filename not included in MATCH_FILENAMES_TO_PROCESS
                verbose_output(
                    f"{BackgroundColors.YELLOW}Skipping {BackgroundColors.CYAN}{filename}{BackgroundColors.YELLOW} (not in MATCH_FILENAMES_TO_PROCESS){Style.RESET_ALL}"
                )  # Verbose: skipping non-matching file
                continue  # Skip this file
            files.append(item_path)  # Add file to result list

    return sorted(files)  # Return sorted list for deterministic output


def gradient_penalty(critic, real_samples, fake_samples, labels, device):
    """
    Compute the WGAN-GP gradient penalty.

    :param critic: critic network callable that accepts (samples, labels)
    :param real_samples: tensor of real samples (B, feature_dim)
    :param fake_samples: tensor of fake samples (B, feature_dim)
    :param labels: tensor of integer labels (B,)
    :param device: torch device to run computations on
    :return: scalar gradient penalty term
    """

    batch_size = real_samples.size(0)  # Get batch size from real samples
    alpha = torch.rand(batch_size, 1, device=device)  # Sample random interpolation factors
    alpha = alpha.expand_as(real_samples)  # Expand alpha to match feature shape
    interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)  # Create interpolated samples
    interpolates.requires_grad_(True)  # Enable gradients for interpolated samples
    d_interpolates = critic(interpolates, labels)  # Get critic scores for interpolated samples
    grad_outputs = torch.ones_like(d_interpolates, device=device)  # Create gradient outputs tensor

    grads = autograd.grad(  # Compute gradients of critic outputs with respect to interpolates
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[
        0
    ]  # Get gradients tensor
    grads = grads.view(batch_size, -1)  # Flatten gradients per sample
    grad_norm = torch.sqrt(torch.sum(grads**2, dim=1) + 1e-12)  # Compute L2 norm of gradients
    gp = ((grad_norm - 1) ** 2).mean()  # Calculate gradient penalty term
    return gp  # Return scalar gradient penalty


def plot_training_metrics(metrics_history, out_dir, filename="training_metrics.png"):
    """
    Plot training metrics and save to output directory.

    :param metrics_history: dictionary containing lists of metrics over training
    :param out_dir: directory to save plots
    :param filename: name of the plot file (default: "training_metrics.png")
    :return: None
    """

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Create 2x3 subplot grid
    fig.suptitle("WGAN-GP Training Metrics", fontsize=16, fontweight="bold")  # Add main title

    steps = metrics_history["steps"]  # Get step numbers

    # Plot 1: Discriminator Loss
    axes[0, 0].plot(steps, metrics_history["loss_D"], color="blue", linewidth=1.5, alpha=0.7)  # Plot loss_D
    axes[0, 0].set_title("Discriminator Loss (WGAN)", fontweight="bold")  # Set subplot title
    axes[0, 0].set_xlabel("Training Step")  # Set x-axis label
    axes[0, 0].set_ylabel("Loss D")  # Set y-axis label
    axes[0, 0].grid(True, alpha=0.3)  # Add grid

    # Plot 2: Generator Loss
    axes[0, 1].plot(steps, metrics_history["loss_G"], color="red", linewidth=1.5, alpha=0.7)  # Plot loss_G
    axes[0, 1].set_title("Generator Loss (WGAN)", fontweight="bold")  # Set subplot title
    axes[0, 1].set_xlabel("Training Step")  # Set x-axis label
    axes[0, 1].set_ylabel("Loss G")  # Set y-axis label
    axes[0, 1].grid(True, alpha=0.3)  # Add grid

    # Plot 3: Gradient Penalty
    axes[0, 2].plot(steps, metrics_history["gp"], color="green", linewidth=1.5, alpha=0.7)  # Plot gradient penalty
    axes[0, 2].set_title("Gradient Penalty", fontweight="bold")  # Set subplot title
    axes[0, 2].set_xlabel("Training Step")  # Set x-axis label
    axes[0, 2].set_ylabel("GP")  # Set y-axis label
    axes[0, 2].grid(True, alpha=0.3)  # Add grid

    # Plot 4: Critic Scores (Real vs Fake)
    axes[1, 0].plot(steps, metrics_history["D_real"], label="E[D(real)]", color="darkblue", linewidth=1.5, alpha=0.7)  # Plot real scores
    axes[1, 0].plot(steps, metrics_history["D_fake"], label="E[D(fake)]", color="darkred", linewidth=1.5, alpha=0.7)  # Plot fake scores
    axes[1, 0].set_title("Critic Scores (Real vs Fake)", fontweight="bold")  # Set subplot title
    axes[1, 0].set_xlabel("Training Step")  # Set x-axis label
    axes[1, 0].set_ylabel("Critic Score")  # Set y-axis label
    axes[1, 0].legend(loc="best")  # Add legend
    axes[1, 0].grid(True, alpha=0.3)  # Add grid

    # Plot 5: Wasserstein Distance Estimate
    axes[1, 1].plot(steps, metrics_history["wasserstein"], color="purple", linewidth=1.5, alpha=0.7)  # Plot Wasserstein distance
    axes[1, 1].set_title("Wasserstein Distance Estimate", fontweight="bold")  # Set subplot title
    axes[1, 1].set_xlabel("Training Step")  # Set x-axis label
    axes[1, 1].set_ylabel("W-Distance")  # Set y-axis label
    axes[1, 1].grid(True, alpha=0.3)  # Add grid

    # Plot 6: Combined Loss View
    axes[1, 2].plot(steps, metrics_history["loss_D"], label="Loss D", color="blue", linewidth=1.5, alpha=0.6)  # Plot loss_D
    axes[1, 2].plot(steps, metrics_history["loss_G"], label="Loss G", color="red", linewidth=1.5, alpha=0.6)  # Plot loss_G
    axes[1, 2].set_title("Combined Loss View", fontweight="bold")  # Set subplot title
    axes[1, 2].set_xlabel("Training Step")  # Set x-axis label
    axes[1, 2].set_ylabel("Loss")  # Set y-axis label
    axes[1, 2].legend(loc="best")  # Add legend
    axes[1, 2].grid(True, alpha=0.3)  # Add grid

    plt.tight_layout()  # Adjust spacing between subplots
    plot_path = os.path.join(out_dir, filename)  # Define plot save path using custom filename
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")  # Save figure to file
    print(f"{BackgroundColors.GREEN}Training metrics plot saved to: {BackgroundColors.CYAN}{plot_path}{Style.RESET_ALL}")  # Print save message
    plt.close()  # Close figure to free memory


def train(args):
    """
    Train the WGAN-GP model using the provided arguments.

    :param args: parsed arguments namespace containing training configuration
    :return: None
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    )  # Select device for training
    set_seed(args.seed)  # Set random seed for reproducibility

    send_telegram_message(TELEGRAM_BOT, f"Starting WGAN-GP training on {Path(args.csv_path).name} for {args.epochs} epochs")

    # Print optimization settings
    print(f"{BackgroundColors.GREEN}Device: {BackgroundColors.CYAN}{device.type.upper()}{Style.RESET_ALL}")
    if args.use_amp and device.type == 'cuda':
        print(f"{BackgroundColors.GREEN}Using Automatic Mixed Precision (AMP) for faster training{Style.RESET_ALL}")
    if args.compile:
        print(f"{BackgroundColors.GREEN}Using torch.compile() for optimized execution{Style.RESET_ALL}")

    dataset = CSVFlowDataset(
        args.csv_path, label_col=args.label_col, feature_cols=args.feature_cols
    )  # Load dataset from CSV
    
    # Optimized DataLoader settings for better performance
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=args.num_workers,  # Configurable number of workers
        pin_memory=True if device.type == 'cuda' else False,  # Faster CPU->GPU transfer
        persistent_workers=True if args.num_workers > 0 else False,  # Keep workers alive between epochs
        prefetch_factor=2 if args.num_workers > 0 else None,  # Prefetch batches for better GPU utilization
    )  # Create dataloader for batching

    feature_dim = dataset.feature_dim  # Get feature dimensionality from dataset
    n_classes = dataset.n_classes  # Get number of label classes from dataset

    G = Generator(
        latent_dim=args.latent_dim,
        feature_dim=feature_dim,
        n_classes=n_classes,
        hidden_dims=args.g_hidden,
        embed_dim=args.embed_dim,
        n_resblocks=args.n_resblocks,
    ).to(
        device
    )  # Initialize generator model
    D = Discriminator(
        feature_dim=feature_dim, n_classes=n_classes, hidden_dims=args.d_hidden, embed_dim=args.embed_dim
    ).to(
        device
    )  # Initialize discriminator model

    # Apply torch.compile() for faster execution (PyTorch 2.0+)
    if args.compile:
        try:
            G = torch.compile(G, mode="reduce-overhead")  # Compile generator
            D = torch.compile(D, mode="reduce-overhead")  # Compile discriminator
            print(f"{BackgroundColors.GREEN}Models compiled successfully{Style.RESET_ALL}")
        except Exception as e:
            print(f"{BackgroundColors.YELLOW}torch.compile() not available or failed: {e}{Style.RESET_ALL}")

    # Initialize mixed precision scaler for AMP
    scaler = torch.cuda.amp.GradScaler() if args.use_amp and device.type == 'cuda' else None

    # Create optimizers for generator and discriminator
    opt_D = torch.optim.Adam(
        cast(Any, D).parameters(), lr=args.lr, betas=(args.beta1, args.beta2)
    )  # Create optimizer for discriminator
    opt_G = torch.optim.Adam(
        cast(Any, G).parameters(), lr=args.lr, betas=(args.beta1, args.beta2)
    )  # Create optimizer for generator

    fixed_noise = torch.randn(args.sample_batch, args.latent_dim, device=device)  # Generate fixed noise for inspection
    fixed_labels = torch.randint(
        0, n_classes, (args.sample_batch,), device=device
    )  # Generate fixed labels for inspection

    os.makedirs(args.out_dir, exist_ok=True)  # Ensure output directory exists
    step = 0  # Initialize global step counter
    start_epoch = 0  # Initialize starting epoch

    # Initialize metrics tracking
    metrics_history = {
        "steps": [],  # Training step numbers
        "loss_D": [],  # Discriminator loss values
        "loss_G": [],  # Generator loss values
        "gp": [],  # Gradient penalty values
        "D_real": [],  # Average critic score for real samples
        "D_fake": [],  # Average critic score for fake samples
        "wasserstein": [],  # Estimated Wasserstein distance (D_real - D_fake)
    }  # Dictionary to store training metrics

    # Automatically check for existing checkpoints for this specific CSV file
    if not args.from_scratch and args.csv_path:  # If not forcing from scratch and CSV path provided
        csv_path_obj = Path(args.csv_path)  # Create Path object from csv_path
        checkpoint_dir = csv_path_obj.parent / "Data_Augmentation" / "Checkpoints"  # Expected checkpoint directory
        checkpoint_prefix = csv_path_obj.stem  # Expected filename prefix
        
        if checkpoint_dir.exists():  # If checkpoint directory exists
            # Find all generator checkpoints for this specific file
            checkpoint_files = sorted(checkpoint_dir.glob(f"{checkpoint_prefix}_generator_epoch*.pt"))  # Find matching checkpoints
            
            if checkpoint_files:  # If checkpoints found for this file
                g_checkpoint_path = checkpoint_files[-1]  # Get latest checkpoint
                # Extract epoch number from filename
                epoch_num = g_checkpoint_path.stem.split("epoch")[-1]  # Extract epoch number
                d_checkpoint_path = checkpoint_dir / f"{checkpoint_prefix}_discriminator_epoch{epoch_num}.pt"  # Build discriminator path
                
                print(f"{BackgroundColors.CYAN}Found existing checkpoints for {csv_path_obj.name}{Style.RESET_ALL}")
                print(f"{BackgroundColors.CYAN}Attempting to resume from epoch {epoch_num}...{Style.RESET_ALL}")
                
                # Load generator checkpoint
                if g_checkpoint_path.exists():  # If generator checkpoint exists
                    try:  # Try to load checkpoint
                        print(f"{BackgroundColors.GREEN}Loading generator checkpoint: {g_checkpoint_path.name}{Style.RESET_ALL}")
                        g_checkpoint = torch.load(g_checkpoint_path, map_location=device, weights_only=False)  # Load generator checkpoint with sklearn objects
                        cast(Any, G).load_state_dict(g_checkpoint["state_dict"])  # Restore generator weights
                        start_epoch = g_checkpoint["epoch"]  # Set starting epoch
                        
                        # Load optimizer state if available
                        if "opt_G_state" in g_checkpoint:  # If optimizer state saved
                            opt_G.load_state_dict(g_checkpoint["opt_G_state"])  # Restore generator optimizer
                            print(f"{BackgroundColors.GREEN}✓ Restored generator optimizer state{Style.RESET_ALL}")
                        
                        # Load metrics history from checkpoint or separate JSON file
                        metrics_loaded = False  # Flag to track if metrics were loaded
                        if "metrics_history" in g_checkpoint:  # If metrics history saved in checkpoint
                            metrics_history = g_checkpoint["metrics_history"]  # Restore metrics from checkpoint
                            step = metrics_history["steps"][-1] if metrics_history["steps"] else 0  # Restore step counter
                            metrics_loaded = True  # Mark as loaded
                            print(f"{BackgroundColors.GREEN}✓ Restored metrics history from checkpoint ({len(metrics_history['steps'])} steps){Style.RESET_ALL}")
                        else:  # Try loading from separate JSON file
                            metrics_json_path = checkpoint_dir / f"{checkpoint_prefix}_metrics_history.json"  # Path to metrics JSON
                            if metrics_json_path.exists():  # If JSON file exists
                                try:  # Try to load metrics
                                    with open(metrics_json_path, 'r') as f:  # Open file for reading
                                        metrics_history = json.load(f)  # Load metrics from JSON
                                    step = metrics_history["steps"][-1] if metrics_history["steps"] else 0  # Restore step counter
                                    metrics_loaded = True  # Mark as loaded
                                    print(f"{BackgroundColors.GREEN}✓ Restored metrics history from JSON file ({len(metrics_history['steps'])} steps){Style.RESET_ALL}")
                                except Exception as e:  # If loading fails
                                    print(f"{BackgroundColors.YELLOW}⚠ Warning: Failed to load metrics from JSON: {e}{Style.RESET_ALL}")
                        
                        # Load AMP scaler state if available
                        if scaler is not None and "scaler_state" in g_checkpoint:  # If using AMP and scaler state saved
                            scaler.load_state_dict(g_checkpoint["scaler_state"])  # Restore scaler state
                            print(f"{BackgroundColors.GREEN}✓ Restored AMP scaler state{Style.RESET_ALL}")
                        
                        # Load discriminator checkpoint
                        if d_checkpoint_path.exists():  # If discriminator checkpoint exists
                            print(f"{BackgroundColors.GREEN}Loading discriminator checkpoint: {d_checkpoint_path.name}{Style.RESET_ALL}")
                            d_checkpoint = torch.load(d_checkpoint_path, map_location=device, weights_only=False)  # Load discriminator checkpoint
                            cast(Any, D).load_state_dict(d_checkpoint["state_dict"])  # Restore discriminator weights
                            
                            # Load optimizer state if available
                            if "opt_D_state" in d_checkpoint:  # If optimizer state saved
                                opt_D.load_state_dict(d_checkpoint["opt_D_state"])  # Restore discriminator optimizer
                                print(f"{BackgroundColors.GREEN}✓ Restored discriminator optimizer state{Style.RESET_ALL}")
                        else:  # Discriminator checkpoint not found
                            print(f"{BackgroundColors.YELLOW}⚠ Warning: Discriminator checkpoint not found{Style.RESET_ALL}")
                        
                        # Check if training metrics plot exists after loading checkpoint
                        plot_dir = csv_path_obj.parent / "Data_Augmentation"  # Plot directory
                        plot_filename = csv_path_obj.stem + "_training_metrics.png"  # Plot filename
                        plot_path = plot_dir / plot_filename  # Full plot path
                        
                        if not plot_path.exists():  # If plot doesn't exist
                            # Check if we have metrics to generate the plot
                            if metrics_loaded and len(metrics_history.get("steps", [])) > 0:  # If metrics available
                                print(f"{BackgroundColors.YELLOW}Training metrics plot not found, generating from metrics history...{Style.RESET_ALL}")
                                os.makedirs(plot_dir, exist_ok=True)  # Ensure directory exists
                                plot_training_metrics(metrics_history, str(plot_dir), plot_filename)  # Generate plot
                                print(f"{BackgroundColors.GREEN}✓ Generated training metrics plot: {plot_filename}{Style.RESET_ALL}")
                            else:  # No metrics available
                                print(f"{BackgroundColors.YELLOW}⚠ Warning: Training metrics plot not found and no metrics history available to generate it{Style.RESET_ALL}")
                        else:
                            print(f"{BackgroundColors.GREEN}✓ Training metrics plot already exists{Style.RESET_ALL}")
                        
                        print(f"{BackgroundColors.GREEN}✓ Resuming training from epoch {start_epoch} (step {step}){Style.RESET_ALL}")
                    except Exception as e:  # If loading fails
                        print(f"{BackgroundColors.YELLOW}⚠ Failed to load checkpoint: {e}{Style.RESET_ALL}")
                        print(f"{BackgroundColors.YELLOW}⚠ Starting training from scratch{Style.RESET_ALL}")
                        start_epoch = 0  # Reset to start from beginning
                        step = 0  # Reset step counter
            else:  # No checkpoints found for this file
                print(f"{BackgroundColors.CYAN}No existing checkpoints found for {csv_path_obj.name}{Style.RESET_ALL}")
                print(f"{BackgroundColors.CYAN}Starting training from scratch{Style.RESET_ALL}")
        else:  # Checkpoint directory doesn't exist
            print(f"{BackgroundColors.CYAN}No checkpoint directory found{Style.RESET_ALL}")
            print(f"{BackgroundColors.CYAN}Starting training from scratch{Style.RESET_ALL}")
    elif args.from_scratch:  # If user explicitly requested from scratch
        print(f"{BackgroundColors.CYAN}--from_scratch flag set, ignoring existing checkpoints{Style.RESET_ALL}")
        print(f"{BackgroundColors.CYAN}Starting training from scratch{Style.RESET_ALL}")

    for epoch in range(start_epoch, args.epochs):  # Loop over epochs starting from resume point
        # Create progress bar for current epoch using original stdout to prevent multiple lines
        pbar = tqdm(
            dataloader, 
            desc=f"{BackgroundColors.CYAN}Epoch {epoch+1}/{args.epochs}{Style.RESET_ALL}", 
            unit="batch",
            file=sys.stdout,  # Use stdout before Logger redirection
            ncols=None,  # Auto-detect terminal width
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'  # Custom format
        )
        
        for real_x_np, labels_np in pbar:  # Loop over batches in dataloader with progress bar
            real_x = real_x_np.to(device)  # Move real features to device
            labels = labels_np.to(device, dtype=torch.long)  # Move labels to device and set type

            loss_D = torch.tensor(0.0, device=device)  # Initialize discriminator loss
            gp = torch.tensor(0.0, device=device)  # Initialize gradient penalty
            d_real_score = torch.tensor(0.0, device=device)  # Initialize real score tracker
            d_fake_score = torch.tensor(0.0, device=device)  # Initialize fake score tracker
            
            # Train discriminator with optional mixed precision
            for _ in range(args.critic_steps):  # Train discriminator multiple steps
                with autocast(device.type, enabled=(scaler is not None)):  # Enable AMP if available
                    z = torch.randn(args.batch_size, args.latent_dim, device=device)  # Sample noise for discriminator step
                    fake_x = G(z, labels).detach()  # Generate fake samples and detach for discriminator
                    d_real = D(real_x, labels)  # Get discriminator score for real samples
                    d_fake = D(fake_x, labels)  # Get discriminator score for fake samples
                    gp = gradient_penalty(D, real_x, fake_x, labels, device)  # Compute gradient penalty
                    loss_D = d_fake.mean() - d_real.mean() + args.lambda_gp * gp  # Calculate WGAN-GP discriminator loss

                opt_D.zero_grad()  # Zero discriminator gradients
                if scaler is not None:  # If using mixed precision
                    scaler.scale(loss_D).backward()  # Scale loss and backpropagate
                    scaler.step(opt_D)  # Update discriminator parameters with scaled gradients
                    scaler.update()  # Update scaler for next iteration
                else:  # Standard precision
                    loss_D.backward()  # Backpropagate discriminator loss
                    opt_D.step()  # Update discriminator parameters

                # Track scores for the last critic step
                d_real_score = d_real.mean()  # Store average real score
                d_fake_score = d_fake.mean()  # Store average fake score

            # Train generator with optional mixed precision
            with autocast(device.type, enabled=(scaler is not None)):  # Enable AMP if available
                z = torch.randn(args.batch_size, args.latent_dim, device=device)  # Sample noise for generator step
                gen_labels = torch.randint(0, n_classes, (args.batch_size,), device=device)  # Sample labels for generator
                fake_x = G(z, gen_labels)  # Generate fake samples with generator
                g_loss = -D(fake_x, gen_labels).mean()  # Calculate generator loss

            opt_G.zero_grad()  # Zero generator gradients
            if scaler is not None:  # If using mixed precision
                scaler.scale(g_loss).backward()  # Scale loss and backpropagate
                scaler.step(opt_G)  # Update generator parameters with scaled gradients
                scaler.update()  # Update scaler for next iteration
            else:  # Standard precision
                g_loss.backward()  # Backpropagate generator loss
                opt_G.step()  # Update generator parameters

            # Update progress bar description with current metrics (colored)
            pbar.set_description(
                f"{BackgroundColors.CYAN}Epoch {epoch+1}/{args.epochs}{Style.RESET_ALL} | "
                f"{BackgroundColors.YELLOW}step {step}{Style.RESET_ALL} | "
                f"{BackgroundColors.RED}loss_D: {loss_D.item():.4f}{Style.RESET_ALL} | "
                f"{BackgroundColors.GREEN}loss_G: {g_loss.item():.4f}{Style.RESET_ALL} | "
                f"gp: {gp.item():.4f} | "
                f"D(real): {d_real_score.item():.4f} | "
                f"D(fake): {d_fake_score.item():.4f}"
            )
            
            # Track metrics every log_interval steps
            if step % args.log_interval == 0:  # Log training progress periodically
                # Calculate Wasserstein distance estimate
                wasserstein_dist = (d_real_score - d_fake_score).item()  # Compute W-distance
                
                # Store metrics
                metrics_history["steps"].append(step)  # Record step number
                metrics_history["loss_D"].append(loss_D.item())  # Record discriminator loss
                metrics_history["loss_G"].append(g_loss.item())  # Record generator loss
                metrics_history["gp"].append(gp.item())  # Record gradient penalty
                metrics_history["D_real"].append(d_real_score.item())  # Record real score
                metrics_history["D_fake"].append(d_fake_score.item())  # Record fake score
                metrics_history["wasserstein"].append(wasserstein_dist)  # Record Wasserstein distance
            step += 1  # Increment global step counter

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:  # Save checkpoints periodically
            # Determine checkpoint output directory based on input CSV location
            if args.csv_path:  # If CSV path is provided
                csv_path_obj = Path(args.csv_path)  # Create Path object from csv_path
                checkpoint_dir = csv_path_obj.parent / "Data_Augmentation" / "Checkpoints"  # Create Checkpoints subdirectory
                os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure directory exists
                checkpoint_prefix = csv_path_obj.stem  # Use input filename as prefix
            else:  # No CSV path, use default out_dir
                checkpoint_dir = Path(args.out_dir) / "Checkpoints"  # Create Checkpoints subdirectory in out_dir
                os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure directory exists
                checkpoint_prefix = "model"  # Default prefix
            
            g_path = checkpoint_dir / f"{checkpoint_prefix}_generator_epoch{epoch+1}.pt"  # Path for generator checkpoint
            d_path = checkpoint_dir / f"{checkpoint_prefix}_discriminator_epoch{epoch+1}.pt"  # Path for discriminator checkpoint
            
            # Calculate class distribution for percentage-based generation
            unique_labels, label_counts = np.unique(dataset.labels, return_counts=True)  # Get class distribution
            class_distribution = dict(zip(unique_labels.tolist(), label_counts.tolist()))  # Create label:count mapping
            
            # Prepare generator checkpoint with full training state
            g_checkpoint = {
                "epoch": epoch + 1,  # Save current epoch number
                "state_dict": cast(Any, G).state_dict(),  # Save generator state dict
                "opt_G_state": cast(Any, opt_G).state_dict(),  # Save generator optimizer state
                "scaler": dataset.scaler,  # Save scaler for inverse transform
                "label_encoder": dataset.label_encoder,  # Save label encoder for mapping
                "feature_cols": dataset.feature_cols,  # Save feature column names for generation
                "class_distribution": class_distribution,  # Save class distribution for percentage-based generation
                "metrics_history": metrics_history,  # Save metrics history for resume
                "args": vars(args),  # Save training arguments
            }
            # Add AMP scaler state if using mixed precision
            if scaler is not None:  # If using AMP
                g_checkpoint["scaler_state"] = scaler.state_dict()  # Save scaler state
            
            torch.save(g_checkpoint, str(g_path))  # Save generator checkpoint to disk
            
            # Prepare discriminator checkpoint
            d_checkpoint = {
                "epoch": epoch + 1,  # Save current epoch number
                "state_dict": cast(Any, D).state_dict(),  # Save discriminator state dict
                "opt_D_state": cast(Any, opt_D).state_dict(),  # Save discriminator optimizer state
                "args": vars(args),  # Save training arguments
            }
            torch.save(d_checkpoint, str(d_path))  # Save discriminator checkpoint to disk
            latest_path = checkpoint_dir / f"{checkpoint_prefix}_generator_latest.pt"  # Path for latest generator
            torch.save(cast(Any, G).state_dict(), str(latest_path))  # Save latest generator weights
            
            # Save metrics history to separate JSON file for easy loading
            metrics_path = checkpoint_dir / f"{checkpoint_prefix}_metrics_history.json"  # Path for metrics JSON
            with open(metrics_path, 'w') as f:  # Open file for writing
                json.dump(metrics_history, f, indent=2)  # Save metrics as JSON
            print(f"{BackgroundColors.GREEN}Saved metrics history to {BackgroundColors.CYAN}{metrics_path}{Style.RESET_ALL}")  # Print metrics save message
            print(f"{BackgroundColors.GREEN}Saved generator to {BackgroundColors.CYAN}{g_path}{Style.RESET_ALL}")  # Print checkpoint save message
    
    print(f"{BackgroundColors.GREEN}Training finished!{Style.RESET_ALL}")  # Print training completion message
    
    # Plot training metrics
    if len(metrics_history["steps"]) > 0:  # If metrics were collected
        print(f"{BackgroundColors.GREEN}Generating training metrics plots...{Style.RESET_ALL}")  # Print plotting message
        # Determine plot output directory based on input CSV location
        if args.csv_path:  # If CSV path is provided
            csv_path_obj = Path(args.csv_path)  # Create Path object from csv_path
            plot_dir = csv_path_obj.parent / "Data_Augmentation"  # Create Data_Augmentation subdirectory
            os.makedirs(plot_dir, exist_ok=True)  # Ensure directory exists
            # Save plot with same base name as input file
            plot_filename = csv_path_obj.stem + "_training_metrics.png"  # Use input filename for plot
            # Temporarily modify out_dir for plotting
            original_out_dir = args.out_dir  # Save original out_dir
            args.out_dir = str(plot_dir)  # Set out_dir to Data_Augmentation
            # Update metrics history to use custom filename
            temp_metrics = metrics_history.copy()  # Copy metrics
            plot_training_metrics(temp_metrics, str(plot_dir), plot_filename)  # Create and save plots
            args.out_dir = original_out_dir  # Restore original out_dir
        else:  # No CSV path, use default out_dir
            plot_training_metrics(metrics_history, args.out_dir, "training_metrics.png")  # Create and save plots

    send_telegram_message(TELEGRAM_BOT, f"Finished WGAN-GP training on {Path(args.csv_path).name} after {args.epochs} epochs")


def generate(args):
    """
    Generate synthetic samples from a saved generator checkpoint.

    :param args: parsed arguments namespace containing generation options
    :return: None
    """

    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
    )  # Select device for generation

    send_telegram_message(TELEGRAM_BOT, f"Starting WGAN-GP generation from {Path(args.checkpoint).name}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)  # Load checkpoint from disk with sklearn objects
    args_ck = ckpt.get("args", {})  # Retrieve saved arguments from checkpoint
    scaler = ckpt.get("scaler", None)  # Try to get scaler from checkpoint
    label_encoder = ckpt.get("label_encoder", None)  # Try to get label encoder from checkpoint
    feature_cols = ckpt.get("feature_cols", None)  # Try to get feature column names from checkpoint
    class_distribution = ckpt.get("class_distribution", None)  # Try to get class distribution from checkpoint

    if scaler is None or label_encoder is None or feature_cols is None or (args.n_samples <= 1.0 and class_distribution is None):  # If critical data missing
        if args.csv_path is None:  # Verify if CSV path is provided
            raise RuntimeError(
                "Checkpoint missing scaler/label_encoder/feature_cols/class_distribution. Provide --csv_path to reconstruct them."
            )  # Raise error if not
        tmp_ds = CSVFlowDataset(
            args.csv_path, label_col=args.label_col, feature_cols=args.feature_cols
        )  # Rebuild dataset to get scaler, encoder, feature names, and class distribution
        scaler = tmp_ds.scaler  # Use scaler from rebuilt dataset
        label_encoder = tmp_ds.label_encoder  # Use label encoder from rebuilt dataset
        feature_cols = tmp_ds.feature_cols  # Use feature column names from rebuilt dataset
        if args.n_samples < 1.0:  # If percentage mode, calculate class distribution
            unique_labels, label_counts = np.unique(tmp_ds.labels, return_counts=True)  # Get class distribution
            class_distribution = dict(zip(unique_labels.tolist(), label_counts.tolist()))  # Create label:count mapping

    if args.feature_dim is not None:  # If feature dimension is provided
        feature_dim = args.feature_dim  # Use provided feature dimension
    else:
        mean_attr = getattr(scaler, "mean_", None) if scaler is not None else None
        if mean_attr is not None:
            mean_arr = np.asarray(mean_attr)
            if mean_arr.ndim == 0:
                raise RuntimeError(
                    "Scaler.mean_ is scalar; unable to infer feature dimension. Provide --feature_dim."
                )
            feature_dim = int(mean_arr.shape[0])  # Infer feature dimension from scaler
        else:
            raise RuntimeError(
                "Unable to determine feature dimension; provide --feature_dim or a checkpoint with scaler."
            )  # Raise error if not available
    n_classes = len(label_encoder.classes_)  # Get number of classes from label encoder

    G = Generator(
        latent_dim=args.latent_dim,
        feature_dim=feature_dim,
        n_classes=n_classes,
        hidden_dims=args.g_hidden,
        embed_dim=args.embed_dim,
        n_resblocks=args.n_resblocks,
    ).to(
        device
    )  # Initialize generator model
    G.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)  # Load generator weights from checkpoint
    G.eval()  # Set generator to evaluation mode

    # Determine number of samples to generate (supports both absolute count and percentage)
    if args.n_samples <= 1.0:  # Percentage mode: generate percentage of training data per class (1.0 == 100%)
        if class_distribution is None:  # If class distribution not available
            raise RuntimeError(
                "Percentage-based generation requires class_distribution in checkpoint or --csv_path to calculate it."
            )  # Raise error
        print(f"{BackgroundColors.CYAN}Generating {args.n_samples*100:.1f}% of training data per class (min 10 samples for small classes){Style.RESET_ALL}")
        if args.label is not None:  # If specific label requested
            if args.label not in class_distribution:  # Verify label exists
                raise ValueError(f"Label {args.label} not found in training data class distribution")  # Raise error
            original_count = class_distribution[args.label]  # Get original class count
            calculated = int(original_count * args.n_samples)  # Calculate percentage-based count
            # For small classes (<100 instances), ensure at least 10 samples are generated
            final_count = max(10 if original_count < 100 else 1, calculated)  # Apply minimum threshold
            n_per_class = {args.label: final_count}  # Store final count
        else:  # Generate for all classes
            n_per_class = {}  # Initialize dictionary
            for label, original_count in class_distribution.items():  # For each class
                calculated = int(original_count * args.n_samples)  # Calculate percentage-based count
                # For small classes (<100 instances), ensure at least 10 samples are generated
                final_count = max(10 if original_count < 100 else 1, calculated)  # Apply minimum threshold
                n_per_class[label] = final_count  # Store final count
        labels = []  # List to build label array
        for label, count in n_per_class.items():  # For each class
            labels.extend([label] * count)  # Repeat label by count
        labels = np.array(labels, dtype=np.int64)  # Convert to array
        n = len(labels)  # Total number of samples
        print(f"{BackgroundColors.GREEN}Total samples to generate: {BackgroundColors.CYAN}{n}{Style.RESET_ALL}")
        for label, count in n_per_class.items():  # Print per-class breakdown
            class_name = label_encoder.inverse_transform([label])[0]  # Get class name
            print(f"{BackgroundColors.GREEN}  - Class '{class_name}': {BackgroundColors.CYAN}{count}{BackgroundColors.GREEN} samples{Style.RESET_ALL}")
    else:  # Absolute count mode: generate exact number of samples
        n = int(args.n_samples)  # Convert to integer
        print(f"{BackgroundColors.CYAN}Generating {n} samples (absolute count){Style.RESET_ALL}")
        if args.label is not None:  # If a specific label is requested
            labels = np.array([args.label] * n, dtype=np.int64)  # Create array of repeated label
        else:
            labels = np.random.randint(0, n_classes, size=(n,), dtype=np.int64)  # Sample labels uniformly

    batch_size = args.gen_batch_size  # Set generation batch size
    all_fake = []  # List to store generated feature batches
    all_labels = []  # List to store corresponding labels
    with torch.no_grad():  # Disable gradient computation for generation
        for i in range(0, n, batch_size):  # Loop over batches for generation
            b = min(batch_size, n - i)  # Calculate current batch size
            z = torch.randn(b, args.latent_dim, device=device)  # Sample noise for batch
            y = torch.from_numpy(labels[i : i + b]).to(device, dtype=torch.long)  # Convert labels to tensor
            fake = G(z, y).cpu().numpy()  # Generate fake samples and move to CPU
            all_fake.append(fake)  # Append generated features to list
            all_labels.append(labels[i : i + b])  # Append labels to list

    X_fake = np.vstack(all_fake)  # Stack all generated feature batches
    Y_fake = np.concatenate(all_labels)  # Concatenate all label arrays

    X_orig = scaler.inverse_transform(X_fake)  # Inverse transform features to original scale

    # Use feature column names from checkpoint (preserves original feature names)
    df = pd.DataFrame(X_orig, columns=feature_cols)  # Create DataFrame with original feature names
    df[args.label_col] = label_encoder.inverse_transform(Y_fake)  # Map integer labels back to original strings
    df.to_csv(args.out_file, index=False)  # Save generated data to CSV file
    print(f"{BackgroundColors.GREEN}Saved {BackgroundColors.CYAN}{n}{BackgroundColors.GREEN} generated samples to {BackgroundColors.CYAN}{args.out_file}{Style.RESET_ALL}")  # Print completion message

    send_telegram_message(TELEGRAM_BOT, f"Finished WGAN-GP generation, saved {n} samples to {Path(args.out_file).name}")


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


def play_sound():
    """
    Plays a sound when the program finishes and skips if the operating system is Windows.

    :param: None
    :return: None
    """

    current_os = platform.system()  # Get the current operating system
    if current_os == "Windows":  # If the current operating system is Windows
        return  # Do nothing

    if verify_filepath_exists(SOUND_FILE):  # If the sound file exists
        if current_os in SOUND_COMMANDS:  # If the platform.system() is in the SOUND_COMMANDS dictionary
            os.system(f"{SOUND_COMMANDS[current_os]} {SOUND_FILE}")  # Play the sound
        else:  # If the platform.system() is not in the SOUND_COMMANDS dictionary
            print(
                f"{BackgroundColors.RED}The {BackgroundColors.CYAN}{current_os}{BackgroundColors.RED} is not in the {BackgroundColors.CYAN}SOUND_COMMANDS dictionary{BackgroundColors.RED}. Please add it!{Style.RESET_ALL}"
            )
    else:  # If the sound file does not exist
        print(
            f"{BackgroundColors.RED}Sound file {BackgroundColors.CYAN}{SOUND_FILE}{BackgroundColors.RED} not found. Make sure the file exists.{Style.RESET_ALL}"
        )


def main():
    """
    Main function.

    :param: None
    :return: None
    """

    print(
        f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}WGAN-GP Data Augmentation{BackgroundColors.GREEN} program!{Style.RESET_ALL}"
    )  # Output the welcome message
    
    start_time = datetime.datetime.now()  # Get the start time of the program
    
    setup_telegram_bot()  # Setup Telegram bot if configured
    
    send_telegram_message(TELEGRAM_BOT, [f"Starting WGAN-GP Data Augmentation at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"])  # Send Telegram notification

    args = parse_args()  # Parse command-line arguments
    
    if args.csv_path is not None:  # Single file mode (csv_path provided):
        # Set output file path if using default
        if args.out_file == "generated.csv" and args.mode in ["gen", "both"]:  # If using default output file
            csv_path_obj = Path(args.csv_path)  # Create Path object from csv_path
            data_aug_dir = csv_path_obj.parent / "Data_Augmentation"  # Create Data_Augmentation subdirectory path
            os.makedirs(data_aug_dir, exist_ok=True)  # Ensure Data_Augmentation directory exists
            output_filename = f"{csv_path_obj.stem}{RESULTS_SUFFIX}{csv_path_obj.suffix}"  # Use input name with _data_augmented suffix
            args.out_file = str(data_aug_dir / output_filename)  # Set output file path to Data_Augmentation subdirectory
        
        if args.mode == "train":
            train(args)  # Train the model
        elif args.mode == "gen":
            assert args.checkpoint is not None, "Generation requires --checkpoint"  # Ensure checkpoint is provided
            generate(args)  # Generate synthetic samples
        elif args.mode == "both":
            print(f"{BackgroundColors.GREEN}[1/2] Training model...{Style.RESET_ALL}")
            train(args)  # Train the model
            
            # Set checkpoint path to the last saved model (dataset-specific)
            csv_path_obj = Path(args.csv_path)
            checkpoint_prefix = csv_path_obj.stem  # Use CSV filename as prefix
            checkpoint_dir = csv_path_obj.parent / "Data_Augmentation" / "Checkpoints"
            checkpoint_path = checkpoint_dir / f"{checkpoint_prefix}_generator_epoch{args.epochs}.pt"
            if not checkpoint_path.exists():
                # Try to find the latest checkpoint for this specific file
                checkpoints = sorted(checkpoint_dir.glob(f"{checkpoint_prefix}_generator_epoch*.pt"))
                if checkpoints:  # If checkpoints found
                    checkpoint_path = checkpoints[-1]  # Use the latest checkpoint
                else:  # No checkpoints found
                    raise FileNotFoundError(f"No generator checkpoint found for {csv_path_obj.name} in {checkpoint_dir}")
            
            args.checkpoint = str(checkpoint_path)
            print(f"\n{BackgroundColors.CYAN}[2/2] Generating samples from {checkpoint_path.name}...{Style.RESET_ALL}")
            print(f"{BackgroundColors.GREEN}Output will be saved to: {BackgroundColors.CYAN}{args.out_file}{Style.RESET_ALL}")
            generate(args)  # Generate synthetic samples
    
    # Batch processing mode (no csv_path provided):
    else:
        print(
            f"{BackgroundColors.GREEN}No CSV path provided. Processing datasets in batch mode...{Style.RESET_ALL}"
        )  # Notify batch mode
        
        for dataset_name, paths in DATASETS.items():  # For each dataset in the DATASETS dictionary
            print(
                f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing dataset: {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}"
            )
            for input_path in paths:  # For each path in the dataset's paths list
                if not verify_filepath_exists(input_path):  # If the input path does not exist
                    verbose_output(
                        f"{BackgroundColors.YELLOW}Skipping missing path: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}"
                    )
                    continue  # Skip to the next path if the current one doesn't exist

                files_to_process = get_files_to_process(
                    input_path, file_extension=".csv"
                )  # Get list of CSV files to process
                
                for file in files_to_process:  # For each file to process
                    print(
                        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*80}{Style.RESET_ALL}"
                    )
                    print(
                        f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing file: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}"
                    )
                    print(
                        f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*80}{Style.RESET_ALL}\n"
                    )
                    
                    # Set output file path: Data_Augmentation subdirectory with same filename
                    csv_path_obj = Path(file)
                    data_aug_dir = csv_path_obj.parent / "Data_Augmentation"  # Create Data_Augmentation subdirectory path
                    os.makedirs(data_aug_dir, exist_ok=True)  # Ensure Data_Augmentation directory exists
                    output_filename = f"{csv_path_obj.stem}{RESULTS_SUFFIX}{csv_path_obj.suffix}"  # Use input name with RESULTS_SUFFIX
                    args.out_file = str(data_aug_dir / output_filename)  # Set output file path to Data_Augmentation subdirectory
                    args.csv_path = file  # Set CSV path to current file
                    
                    try:
                        if args.mode == "train":
                            train(args)  # Train the model only
                        elif args.mode == "gen":
                            assert args.checkpoint is not None, "Generation requires --checkpoint"
                            generate(args)  # Generate synthetic samples only
                        elif args.mode == "both":
                            print(f"{BackgroundColors.GREEN}[1/2] Training model on {BackgroundColors.CYAN}{csv_path_obj.name}{BackgroundColors.GREEN}...{Style.RESET_ALL}")
                            train(args)  # Train the model
                            
                            # Set checkpoint path to the last saved model (dataset-specific)
                            checkpoint_prefix = csv_path_obj.stem  # Use CSV filename as prefix
                            checkpoint_dir = data_aug_dir / "Checkpoints"  # Checkpoints in Data_Augmentation/Checkpoints
                            checkpoint_path = checkpoint_dir / f"{checkpoint_prefix}_generator_epoch{args.epochs}.pt"
                            if not checkpoint_path.exists():
                                # Try to find the latest checkpoint for this specific file
                                checkpoints = sorted(checkpoint_dir.glob(f"{checkpoint_prefix}_generator_epoch*.pt"))
                                if checkpoints:
                                    checkpoint_path = checkpoints[-1]
                                else:
                                    raise FileNotFoundError(f"No generator checkpoint found for {csv_path_obj.name} in {checkpoint_dir}")
                            
                            args.checkpoint = str(checkpoint_path)
                            print(f"\n{BackgroundColors.CYAN}[2/2] Generating samples from {checkpoint_path.name}...{Style.RESET_ALL}")
                            print(f"{BackgroundColors.GREEN}Output will be saved to: {BackgroundColors.CYAN}{args.out_file}{Style.RESET_ALL}")
                            generate(args)  # Generate synthetic samples
                            
                    except Exception as e:
                        print(
                            f"{BackgroundColors.RED}Error processing {BackgroundColors.CYAN}{file}{BackgroundColors.RED}: {e}{Style.RESET_ALL}"
                        )  # Print error message
                        traceback.print_exc()  # Print full traceback
                        continue  # Continue to next file
        
        print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Batch processing completed!{Style.RESET_ALL}")

    finish_time = datetime.datetime.now()  # Get the finish time of the program
    print(
        f"\n{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
    )  # Output the start and finish times
    print(
        f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
    )  # Output the end of the program message

    send_telegram_message(TELEGRAM_BOT, [f"WGAN-GP Data Augmentation finished. Execution time: {calculate_execution_time(start_time, finish_time)}"])

    (
        atexit.register(play_sound) if RUN_FUNCTIONS["Play Sound"] else None
    )  # Register the play_sound function to be called when the program finishes


if __name__ == "__main__":
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """

    main()  # Call the main function
