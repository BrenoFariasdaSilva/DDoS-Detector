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
import csv  # For writing per-directory results CSV
import dataframe_image as dfi  # For exporting DataFrame to PNG images
import datetime  # For tracking execution time
import json  # For saving/loading metrics history
import matplotlib.pyplot as plt  # For plotting training metrics
import numpy as np  # Numerical operations
import os  # For running a command in the terminal
import pandas as pd  # For CSV handling
import platform  # For getting the operating system name
import random  # For reproducibility
import subprocess  # For running small system commands to query hardware
import sys  # For system-specific parameters and functions
import telegram_bot as telegram_module  # For setting Telegram prefix and device info
import time  # For elapsed time tracking
import torch  # PyTorch core
import torch.nn as nn  # Neural network modules
import traceback  # For printing tracebacks on exceptions
import yaml  # For loading configuration files
from colorama import Style  # For coloring the terminal
from contextlib import nullcontext  # For null context manager
from Logger import Logger  # For logging output to both terminal and file
from pathlib import Path  # For handling file paths
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For data preprocessing
from telegram_bot import TelegramBot, send_exception_via_telegram, send_telegram_message, setup_global_exception_hook  # For Telegram utilities and global exception hook
from torch import autograd  # For gradient penalty
from torch.utils.data import DataLoader, Dataset  # Dataset and DataLoader
from tqdm import tqdm  # For progress bar visualization
from typing import Any, Dict, List, Optional, Union, cast  # For Any type hint and cast

psutil = (
    __import__("psutil") if __import__("importlib").util.find_spec("psutil") else None
)  # Import psutil if available, otherwise set to None

# Prefer CUDA autocast when available; provide a safe fallback context manager
try:  # Attempt to import torch.amp.autocast for mixed precision support
    from torch.amp.autocast_mode import autocast as _torch_autocast  # Import autocast for mixed precision
except Exception as e:  # If import fails (e.g., CUDA not available), handle and notify
    print(str(e))  # Print import error to terminal for visibility
    try:  # Try to notify via Telegram about the import failure
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full import error via Telegram via shared handler
    except Exception:  # If notification fails, ignore to avoid recursion during import
        pass  # Ignore Telegram send errors during import fallback
    _torch_autocast = None  # Set to None to indicate autocast is unavailable


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
CONFIG = None  # Will be initialized by load_configuration() - holds all runtime settings

# Results CSV Handles Registry:
RESULTS_CSV_HANDLES = {}  # Registry mapping results CSV path -> (file_obj, csv_writer) for progressive writes

# Processed files registry to avoid duplicate processing in a single run
PROCESSED_FILES = set()  # Set of absolute file paths already processed in this execution

# Telegram Bot Setup:
TELEGRAM_BOT = None  # Global Telegram bot instance (initialized in setup_telegram_bot)

# Logger Setup:
logger = None  # Will be initialized in initialize_logger()


# Functions Definitions:

setup_global_exception_hook()  # Set global exception hook to shared Telegram handler


def get_default_config():
    """
    Return the default configuration dictionary for WGAN-GP.

    This defines all configurable parameters with their default values.
    These defaults match config.yaml.example structure and can be overridden
    by config.yaml file or CLI arguments.

    :return: Dictionary containing default configuration values
    """

    try:
        return {  # Begin configuration dictionary
        "execution": {  # Execution control parameters
            "verbose": False,  # Enable verbose output messages
            "play_sound": True,  # Play sound notification when complete
            "results_suffix": "_data_augmented",  # Suffix to add to generated filenames
            "match_filenames_to_process": [""],  # List of specific filenames to search for a match
            "ignore_files": ["_data_augmented"],  # List of filename substrings to ignore
            "ignore_dirs": [  # List of directory names to ignore when searching for datasets
                "Classifiers",
                "Classifiers_Hyperparameters",
                "Dataset_Description",
                "Data_Separability",
                "Feature_Analysis",
            ],
        },
        "wgangp": {  # WGAN-GP core configuration
            "mode": "both",  # Mode: train, gen, or both
            "csv_path": None,  # Path to CSV training data
            "label_col": "Label",  # Column name for class label
            "feature_cols": None,  # List of feature column names (None = use all)
            "seed": 42,  # Random seed for reproducibility
            "force_cpu": False,  # Force CPU usage even if CUDA available
            "from_scratch": False,  # Force training from scratch, ignore checkpoints
        },
        "dataset": {  # Dataset configuration
            "remove_zero_variance": True,  # Remove zero-variance features during preprocessing
            "label_candidates": ["label", "class", "target"],  # Common label column names for auto-detection
            "datasets": {  # Dictionary containing dataset paths
                "CICDDoS2019-Dataset": [  # List of paths to the CICDDoS2019 dataset
                    "./Datasets/CICDDoS2019/01-12/",
                    "./Datasets/CICDDoS2019/03-11/",
                ],
            },
        },
        "training": {  # Training hyperparameters
            "epochs": 60,  # Number of training epochs
            "batch_size": 64,  # Training batch size
            "critic_steps": 5,  # Number of discriminator updates per generator update
            "lr": 1e-4,  # Learning rate for both networks
            "beta1": 0.5,  # Adam optimizer beta1 parameter
            "beta2": 0.9,  # Adam optimizer beta2 parameter
            "lambda_gp": 10.0,  # Gradient penalty coefficient
            "save_every": 5,  # Save checkpoints every N epochs
            "log_interval": 50,  # Log metrics every N steps
            "sample_batch": 16,  # Number of samples for fixed noise generation
            "use_amp": False,  # Use automatic mixed precision
            "compile": False,  # Use torch.compile() for faster execution
        },
        "generator": {  # Generator architecture
            "latent_dim": 100,  # Dimensionality of noise vector
            "hidden_dims": [256, 512],  # Hidden layer sizes
            "embed_dim": 32,  # Label embedding dimension
            "n_resblocks": 3,  # Number of residual blocks
            "leaky_relu_alpha": 0.2,  # LeakyReLU negative slope
        },
        "discriminator": {  # Discriminator architecture
            "hidden_dims": [512, 256, 128],  # Hidden layer sizes
            "embed_dim": 32,  # Label embedding dimension
            "leaky_relu_alpha": 0.2,  # LeakyReLU negative slope
        },
        "gradient_penalty": {  # Gradient penalty configuration
            "epsilon": 1e-12,  # Small constant for numerical stability
        },
        "generation": {  # Sample generation parameters
            "checkpoint": None,  # Path to generator checkpoint
            "n_samples": 1.0,  # Number/percentage of samples to generate
            "label": None,  # Specific class ID to generate (None = all classes)
            "out_file": "generated.csv",  # Output CSV filename
            "gen_batch_size": 256,  # Generation batch size
            "feature_dim": None,  # Feature dimensionality (None = auto-detect)
            "small_class_threshold": 100,  # Threshold for small class detection
            "small_class_min_samples": 10,  # Minimum samples for small classes
        },
        "dataloader": {  # DataLoader configuration
            "num_workers": 8,  # Number of workers for data loading
            "pin_memory": True,  # Use pinned memory for faster GPU transfer
            "persistent_workers": True,  # Keep workers alive between epochs
            "prefetch_factor": 2,  # Number of batches to prefetch
        },
        "plotting": {  # Visualization configuration
            "enabled": True,  # Enable plot generation
            "filename": "training_metrics.png",  # Plot filename
            "subdir": "plots",  # Subdirectory under data augmentation outputs for plots
            "figsize": [18, 10],  # Figure size [width, height]
            "dpi": 300,  # Image resolution
            "subplot_rows": 2,  # Number of subplot rows
            "subplot_cols": 3,  # Number of subplot columns
            "linewidth": 1.5,  # Line width for plots
            "alpha": 0.7,  # Transparency for plot lines
            "grid_alpha": 0.3,  # Grid transparency
        },
        "paths": {  # File paths configuration
            "out_dir": "outputs",  # Output directory for models/logs
            "logs_dir": "./Logs",  # Directory for log files
            "checkpoint_subdir": "Checkpoints",  # Checkpoints subdirectory name
            "data_augmentation_subdir": "Data_Augmentation",  # Data augmentation subdirectory name
        },
        "logging": {  # Logging configuration
            "enabled": True,  # Enable file logging
            "clean": True,  # Clear log file on start
            "tqdm_bar_format": "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",  # Progress bar format
        },
        "telegram": {  # Telegram bot configuration
            "enabled": True,  # Enable Telegram notifications
            "verify_env": True,  # Verify .env file existence
        },
        "sound": {  # Sound notification configuration
            "enabled": True,  # Enable sound notifications
            "commands": {  # Commands to play sound for each operating system
                "Darwin": "afplay",
                "Linux": "aplay",
                "Windows": "start",
            },
            "file": "./.assets/Sounds/NotificationSound.wav",  # Path to the sound file
        },
    }  # End configuration dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def load_configuration(config_path: Optional[str] = None, cli_overrides: Optional[Dict] = None):
    """
    Load configuration from config.yaml with fallback to defaults.

    Priority order (highest to lowest):
    1. CLI arguments (cli_overrides)
    2. config.yaml (if exists)
    3. config.yaml.example (if exists)
    4. Hard-coded defaults (from get_default_config)

    :param config_path: Optional path to config.yaml file (default: search in current dir)
    :param cli_overrides: Optional dictionary of CLI argument overrides
    :return: Merged configuration dictionary
    """

    try:
        global CONFIG  # Declare global CONFIG variable
        
        config = get_default_config()  # Start with default configuration

        if config_path is None:  # If no config path provided
            path = Path("config.yaml")  # Try config.yaml first
        else:  # If config path provided
            path = Path(config_path)  # Convert to Path object

        if not path.exists():  # If config.yaml not found
            path = Path("config.yaml.example")  # Fallback to example config

        if path.exists():  # If a config file exists
            try:  # Try to load configuration from file
                with open(path, "r") as f:  # Open config file for reading
                    file_config = yaml.safe_load(f)  # Load YAML configuration
                if file_config:  # If config loaded successfully
                    config = deep_merge(config, file_config)  # Merge with defaults
                    print(f"{BackgroundColors.GREEN}Loaded configuration from: {BackgroundColors.CYAN}{path}{Style.RESET_ALL}")  # Print success message
            except Exception as e:  # If loading fails
                print(f"{BackgroundColors.YELLOW}Warning: Failed to load {path}: {e}{Style.RESET_ALL}")  # Print warning
                print(f"{BackgroundColors.YELLOW}Using default configuration{Style.RESET_ALL}")  # Fallback message

        if cli_overrides:  # If CLI overrides provided
            config = deep_merge(config, cli_overrides)  # Merge CLI args (highest priority)

        CONFIG = config  # Store in global variable
        return config  # Return merged configuration
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries, with override taking precedence.

    Recursively merges nested dictionaries. Non-dict values from override
    completely replace corresponding values in base.

    :param base: Base dictionary (lower priority)
    :param override: Override dictionary (higher priority)
    :return: Merged dictionary
    """

    try:
        result = base.copy()  # Create copy of base dictionary
        for key, value in override.items():  # Iterate override keys
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):  # If both are dicts
                result[key] = deep_merge(result[key], value)  # Recursively merge
            else:  # Direct replacement
                result[key] = value  # Override value
        return result  # Return merged dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def initialize_logger(config: Dict):
    """
    Initialize the logger for output redirection.

    :param config: Configuration dictionary containing logging settings
    :return: None
    """

    try:
        global logger  # Declare global logger variable
        
        if not config.get("logging", {}).get("enabled", True):  # If logging disabled
            logger = None  # Don't initialize logger
            return  # Exit early
        
        logs_dir = config.get("paths", {}).get("logs_dir", "./Logs")  # Get logs directory
        log_file = f"{logs_dir}/{Path(__file__).stem}.log"  # Construct log file path
        clean = config.get("logging", {}).get("clean", True)  # Get clean flag
        
        logger = Logger(log_file, clean=clean)  # Create a Logger instance
        sys.stdout = logger  # Redirect stdout to the logger
        sys.stderr = logger  # Redirect stderr to the logger
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def detect_label_column(columns, config: Optional[Dict] = None):
    """
    Try to guess the label column based on common naming conventions.

    :param columns: List of column names
    :param config: Optional configuration dictionary containing label candidates
    :return: The name of the label column if found, else None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG or get_default_config()  # Use global or default config
        
        candidates = config.get("dataset", {}).get("label_candidates", ["label", "class", "target"])  # Get label column candidates from config

        for col in columns:  # First search for exact matches
            if col.lower() in candidates:  # Verify if the column name matches any candidate exactly
                return col  # Return the column name if found

        for col in columns:  # Second search for partial matches
            if "target" in col.lower() or "label" in col.lower():  # Verify if the column name contains any candidate
                return col  # Return the column name if found

        return None  # Return None if no label column is found
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def preprocess_dataframe(df, label_col, remove_zero_variance=None, config: Optional[Dict] = None):
    """
    Preprocess a DataFrame by:
    1. Selecting only numeric feature columns (excluding label)
    2. Removing rows with NaN or infinite values
    3. Optionally dropping zero-variance numeric features

    :param df: pandas DataFrame to preprocess
    :param label_col: name of the label column to preserve
    :param remove_zero_variance: whether to drop numeric columns with zero variance (None = use config)
    :param config: Optional configuration dictionary
    :return: cleaned DataFrame with only numeric features and the label column
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG or get_default_config()  # Use global or default config
        
        if remove_zero_variance is None:  # If not specified
            remove_zero_variance = config.get("dataset", {}).get("remove_zero_variance", True)  # Get from config
        
        verbose_output(
            f"{BackgroundColors.GREEN}Preprocessing DataFrame: selecting numeric features, removing NaN/inf, handling zero-variance.{Style.RESET_ALL}",
            config=config
        )  # Output verbose message

        if df is None:  # If the DataFrame is None
            return df  # Return None

        df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names

        labels = df[label_col].copy()  # Save labels

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()  # Get numeric column names
        if label_col in numeric_cols:  # If label is numeric, remove it from features
            numeric_cols.remove(label_col)  # Remove label from feature list

        verbose_output(
            f"{BackgroundColors.GREEN}Found {len(numeric_cols)} numeric feature columns out of {len(df.columns)-1} total features.{Style.RESET_ALL}"
        )  # Output count

        df_numeric = df[numeric_cols].copy()  # Select numeric features

        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
        valid_mask = ~df_numeric.isna().any(axis=1)  # Mask for rows without NaN

        df_clean = df_numeric[valid_mask].copy()  # Keep only valid rows
        labels_clean = labels[valid_mask].copy()  # Keep corresponding labels

        rows_dropped = len(df) - len(df_clean)  # Calculate dropped rows
        if rows_dropped > 0:  # If rows were dropped
            verbose_output(
                f"{BackgroundColors.YELLOW}Dropped {rows_dropped} rows with NaN/inf values ({rows_dropped/len(df)*100:.2f}%).{Style.RESET_ALL}"
            )  # Output warning

        if remove_zero_variance and len(df_clean) > 0:  # If removal enabled and data remains
            variances = df_clean.var(axis=0, ddof=0)  # Calculate column variances
            zero_var_cols = variances[variances == 0].index.tolist()  # Get zero-variance columns
            if zero_var_cols:  # If zero-variance columns exist
                verbose_output(
                    f"{BackgroundColors.YELLOW}Dropping {len(zero_var_cols)} zero-variance columns.{Style.RESET_ALL}"
                )  # Output warning
                df_clean = df_clean.drop(columns=zero_var_cols)  # Drop zero-variance columns

        df_clean[label_col] = labels_clean.values  # Restore labels

        return df_clean  # Return cleaned DataFrame
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


# Classes Definitions:


class CSVFlowDataset(Dataset):
    """
    Dataset class for loading and preprocessing CSV flow data with automatic scaling and label encoding."""

    def __init__(
        self,
        csv_path: str,
        label_col: str,
        feature_cols: Optional[List[str]] = None,
        scaler: Optional[StandardScaler] = None,
        label_encoder: Optional[LabelEncoder] = None,
        fit_scaler: bool = True,
    ):
        """
        Initialize the CSVFlowDataset for loading flow data from CSV with automatic preprocessing.

        :param csv_path: Path to CSV file containing flows and labels
        :param label_col: Column name that contains the class labels
        :param feature_cols: Optional list of feature column names (None = use all numeric columns)
        :param scaler: Optional pre-fitted StandardScaler to use for features
        :param label_encoder: Optional pre-fitted LabelEncoder to transform labels
        :param fit_scaler: If True and scaler is None, fit a new StandardScaler on the data
        :return: None
        """
        
        df = pd.read_csv(csv_path, low_memory=False)  # Load CSV file into a DataFrame with low_memory=False to avoid DtypeWarning

        df.columns = df.columns.str.strip()  # Remove leading/trailing whitespace from column names

        if label_col not in df.columns:  # If the specified label column is not found
            detected_col = detect_label_column(df.columns)  # Try to detect the label column
            if detected_col is not None:  # If a label column was detected
                print(f"{BackgroundColors.YELLOW}Warning: Label column '{label_col}' not found. Using detected column: '{detected_col}'{Style.RESET_ALL}")  # Warn user
                label_col = detected_col  # Use the detected column
            else:  # If no label column was detected
                raise ValueError(f"Label column '{label_col}' not found in CSV. Available columns: {list(df.columns)}")  # Raise error

        df = preprocess_dataframe(df, label_col, remove_zero_variance=True)  # Clean and filter DataFrame

        if len(df) == 0:  # If all rows were dropped
            raise ValueError(f"No valid data remaining after preprocessing {csv_path}")  # Raise error

        available_features = [c for c in df.columns if c != label_col]  # List numeric features

        if feature_cols is None:  # When user does not specify features
            feature_cols = available_features  # Use all available numeric features
        else:  # User specified features
            feature_cols = [c for c in feature_cols if c in available_features]  # Keep valid features
            if not feature_cols:  # If no valid features remain
                raise ValueError(f"None of the specified feature columns are numeric or available in {csv_path}")  # Raise error

        self.label_col = label_col  # Save label column name
        self.feature_cols = feature_cols  # Save list of feature columns

        self.labels_raw: np.ndarray = np.asarray(df[label_col].values, dtype=str)  # Extract raw labels as string array for consistent encoding, store as instance variable

        self.labels: Any  # Must be Any or Pylance will error

        labels_arr = np.asarray(self.labels_raw, dtype=str)  # Convert labels to string array for consistent encoding

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
        """
        Return the number of samples in the dataset.

        :return: Total number of feature vectors in the dataset
        """

        return len(self.X)  # Return number of feature vectors

    def __getitem__(self, idx):  # Fetch one item by index
        """
        Fetch a single sample by index.

        :param idx: Index of the sample to retrieve
        :return: Tuple of (features, label) where features is a numpy array and label is an integer
        """

        x = self.X[idx]  # Get feature row
        y = int(self.labels[idx])  # Get encoded label
        return x, y  # Return (features, label)


class ResidualBlockFC(nn.Module):
    """
    Simple fully-connected residual block with skip connection for generator network."""

    def __init__(self, dim, leaky_relu_alpha=0.2):
        """
        Initialize a residual fully-connected block for the generator.

        :param dim: Input and output dimensionality of the block
        :param leaky_relu_alpha: Negative slope for LeakyReLU activation (default: 0.2)
        :return: None
        """

        super().__init__()  # Initialize the parent nn.Module class

        self.net = nn.Sequential(  # Define the residual transformation path
            nn.Linear(dim, dim),  # First linear projection
            nn.BatchNorm1d(dim),  # Normalize activations
            nn.LeakyReLU(leaky_relu_alpha, inplace=True),  # Apply nonlinearity
            nn.Linear(dim, dim),  # Second linear projection
            nn.BatchNorm1d(dim),  # Second batch normalization
        )  # End of sequential block

        self.act = nn.LeakyReLU(leaky_relu_alpha, inplace=True)  # Activation after merging residual shortcut

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
    Conditional generator network that maps noise and class labels to synthetic feature vectors using residual blocks.
    """

    def __init__(
        self,
        latent_dim: int,
        feature_dim: int,
        n_classes: int,
        hidden_dims: Optional[List[int]] = None,
        embed_dim: int = 32,
        n_resblocks: int = 3,
        leaky_relu_alpha: float = 0.2,
    ):
        """
        Initialize conditional generator that maps (z, y) -> feature vector.

        :param latent_dim: Dimensionality of noise vector z
        :param feature_dim: Dimensionality of output feature vector
        :param n_classes: Number of conditioning classes
        :param hidden_dims: List of hidden layer sizes for initial MLP (default: [256, 512])
        :param embed_dim: Size of label embedding (default: 32)
        :param n_resblocks: Number of residual blocks to apply (default: 3)
        :param leaky_relu_alpha: Negative slope for LeakyReLU activation (default: 0.2)
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
            layers.append(nn.LeakyReLU(leaky_relu_alpha, inplace=True))  # Apply activation
            prev = h  # Update width tracker

        res_dim = prev  # Width entering residual blocks
        self.pre = nn.Sequential(*layers)  # Store assembled MLP

        self.resblocks = nn.ModuleList(  # Build list of residual blocks
            [ResidualBlockFC(res_dim, leaky_relu_alpha) for _ in range(n_resblocks)]  # Create required count of blocks
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
    Conditional discriminator network (Wasserstein critic) that scores feature vectors conditioned on class labels.
    """

    def __init__(
        self,
        feature_dim: int,
        n_classes: int,
        hidden_dims: Optional[List[int]] = None,
        embed_dim: int = 32,
        leaky_relu_alpha: float = 0.2,
    ):
        """
        Initialize conditional critic/discriminator network that scores (x, y).

        :param feature_dim: Dimensionality of input feature vector
        :param n_classes: Number of classes for conditioning
        :param hidden_dims: List of hidden layer sizes (default: [512, 256, 128])
        :param embed_dim: Dimensionality of label embedding (default: 32)
        :param leaky_relu_alpha: Negative slope for LeakyReLU activation (default: 0.2)
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
            layers.append(nn.LeakyReLU(leaky_relu_alpha, inplace=True))  # Activation function
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


def verbose_output(true_string="", false_string="", config: Optional[Dict] = None):
    """
    Outputs a message if the verbose flag is enabled in configuration.

    :param true_string: The string to be outputted if verbose is enabled.
    :param false_string: The string to be outputted if verbose is disabled.
    :param config: Optional configuration dictionary containing verbose setting
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG or get_default_config()  # Use global or default config
        
        verbose = config.get("execution", {}).get("verbose", False)  # Get verbose setting from config
        
        if verbose and true_string != "":  # If verbose is True and a true_string was provided
            print(true_string)  # Output the true statement string
        elif false_string != "":  # If a false_string was provided
            print(false_string)  # Output the false statement string
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def verify_dot_env_file(config: Optional[Dict] = None):
    """
    Verifies if the .env file exists in the current directory.

    :param config: Optional configuration dictionary containing telegram settings
    :return: True if the .env file exists, False otherwise
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG or get_default_config()  # Use global or default config
        
        if not config.get("telegram", {}).get("verify_env", True):  # If verification disabled
            return True  # Skip verification
        
        env_path = Path(__file__).parent / ".env"  # Path to the .env file
        if not env_path.exists():  # If the .env file does not exist
            print(f"{BackgroundColors.CYAN}.env{BackgroundColors.YELLOW} file not found at {BackgroundColors.CYAN}{env_path}{BackgroundColors.YELLOW}. Telegram messages may not be sent.{Style.RESET_ALL}")
            return False  # Return False

        return True  # Return True if the .env file exists
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def setup_telegram_bot(config: Optional[Dict] = None):
    """
    Sets up the Telegram bot for progress messages.

    :param config: Optional configuration dictionary containing telegram settings
    :return: None
    """
    
    try:
        if config is None:  # If no config provided
            config = CONFIG or get_default_config()  # Use global or default config
        
        if not config.get("telegram", {}).get("enabled", True):  # If Telegram disabled
            return  # Exit early
        
        verbose_output(
            f"{BackgroundColors.GREEN}Setting up Telegram bot for messages...{Style.RESET_ALL}",
            config=config
        )  # Output the verbose message

        verify_dot_env_file(config)  # Verify if the .env file exists

        global TELEGRAM_BOT  # Declare the module-global telegram_bot variable

        try:  # Try to initialize the Telegram bot
            TELEGRAM_BOT = TelegramBot()  # Initialize Telegram bot for progress messages
            telegram_module.TELEGRAM_DEVICE_INFO = f"{telegram_module.get_local_ip()} - {platform.system()}"
            telegram_module.RUNNING_CODE = os.path.basename(__file__)
            telegram_module.TELEGRAM_BOT = TELEGRAM_BOT  # Register the created bot instance in telegram_bot module
        except Exception as e:
            print(f"{BackgroundColors.RED}Failed to initialize Telegram bot: {e}{Style.RESET_ALL}")
            TELEGRAM_BOT = None  # Set to None if initialization fails
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def parse_args():
    """
    Parse command-line arguments and return namespace.

    :return: parsed arguments namespace
    """

    try:
        p = argparse.ArgumentParser(description="DRCGAN-like WGAN-GP for CICDDoS2019 features")  # Create argument parser
    
        p.add_argument("--config", type=str, default=None, help="Path to configuration YAML file (default: config.yaml)")  # Add config file argument
        
        p.add_argument("--verbose", action="store_true", help="Enable verbose output messages")  # Add verbose argument
        p.add_argument("--no_sound", action="store_true", help="Disable sound notification")  # Add no sound argument
        
        p.add_argument("--mode", choices=["train", "gen", "both"], default=None, help="Mode: train, gen, or both")  # Add mode argument
        p.add_argument("--csv_path", type=str, default=None, help="Path to CSV training data")  # Add CSV path argument
        p.add_argument("--label_col", type=str, default=None, help="Column name for class label")  # Add label column argument
        p.add_argument("--feature_cols", nargs="+", default=None, help="List of feature column names")  # Add feature columns argument
        
        p.add_argument("--out_dir", type=str, default=None, help="Output directory for models/logs")  # Add output directory argument
        p.add_argument("--logs_dir", type=str, default=None, help="Directory for log files")  # Add logs directory argument
        
        p.add_argument("--epochs", type=int, default=None, help="Number of training epochs")  # Add epochs argument
        p.add_argument("--batch_size", type=int, default=None, help="Training batch size")  # Add batch size argument
        p.add_argument("--critic_steps", type=int, default=None, help="Discriminator updates per generator update")  # Add critic steps argument
        p.add_argument("--lr", type=float, default=None, help="Learning rate")  # Add learning rate argument
        p.add_argument("--beta1", type=float, default=None, help="Adam optimizer beta1")  # Add beta1 argument
        p.add_argument("--beta2", type=float, default=None, help="Adam optimizer beta2")  # Add beta2 argument
        p.add_argument("--lambda_gp", type=float, default=None, dest="lambda_gp", help="Gradient penalty coefficient")  # Add lambda_gp argument
        p.add_argument("--seed", type=int, default=None, help="Random seed")  # Add seed argument
        p.add_argument("--save_every", type=int, default=None, help="Save checkpoint every N epochs")  # Add save_every argument
        p.add_argument("--log_interval", type=int, default=None, help="Log metrics every N steps")  # Add log_interval argument
        p.add_argument("--sample_batch", type=int, default=None, help="Number of samples for fixed noise")  # Add sample_batch argument
        p.add_argument("--force_cpu", action="store_true", help="Force CPU usage")  # Add force_cpu argument
        p.add_argument("--use_amp", action="store_true", help="Use automatic mixed precision")  # Add use_amp argument
        p.add_argument("--compile", action="store_true", help="Use torch.compile()")  # Add compile argument
        p.add_argument("--from_scratch", action="store_true", help="Train from scratch")  # Add from_scratch argument
        
        p.add_argument("--latent_dim", type=int, default=None, help="Noise vector dimensionality")  # Add latent_dim argument
        p.add_argument("--g_hidden", nargs="+", type=int, default=None, help="Generator hidden layer sizes")  # Add g_hidden argument
        p.add_argument("--g_embed_dim", type=int, default=None, help="Generator label embedding dim")  # Add g_embed_dim argument
        p.add_argument("--n_resblocks", type=int, default=None, help="Number of residual blocks")  # Add n_resblocks argument
        p.add_argument("--g_leaky_relu_alpha", type=float, default=None, help="Generator LeakyReLU alpha")  # Add g_leaky_relu_alpha argument
        
        p.add_argument("--d_hidden", nargs="+", type=int, default=None, help="Discriminator hidden layer sizes")  # Add d_hidden argument
        p.add_argument("--d_embed_dim", type=int, default=None, help="Discriminator label embedding dim")  # Add d_embed_dim argument
        p.add_argument("--d_leaky_relu_alpha", type=float, default=None, help="Discriminator LeakyReLU alpha")  # Add d_leaky_relu_alpha argument
        
        p.add_argument("--checkpoint", type=str, default=None, help="Generator checkpoint path")  # Add checkpoint argument
        p.add_argument("--n_samples", type=float, default=None, help="Number/percentage of samples to generate")  # Add n_samples argument
        p.add_argument("--force_new_samples", action="store_true", help="Force generation of new samples even if output file exists")  # Add force_new_samples argument
        p.add_argument("--gen_label", type=int, default=None, help="Specific class ID to generate")  # Add gen_label argument
        p.add_argument("--out_file", type=str, default=None, help="Output CSV filename")  # Add out_file argument
        p.add_argument("--gen_batch_size", type=int, default=None, help="Generation batch size")  # Add gen_batch_size argument
        p.add_argument("--feature_dim", type=int, default=None, help="Feature dimensionality")  # Add feature_dim argument
        
        p.add_argument("--num_workers", type=int, default=None, help="Number of DataLoader workers")  # Add num_workers argument
        
        p.add_argument("--remove_zero_variance", action="store_true", help="Remove zero-variance features")  # Add remove_zero_variance argument
        p.add_argument("--no_remove_zero_variance", action="store_true", help="Don't remove zero-variance features")  # Add no_remove_zero_variance argument
        
        return p.parse_args()  # Parse arguments and return namespace
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def args_to_config_overrides(args):
    """
    Convert parsed CLI arguments to configuration overrides dictionary.

    :param args: argparse.Namespace containing CLI arguments
    :return: Dictionary with configuration overrides
    """

    try:
        overrides = {}  # Initialize overrides dictionary
    
        if args.verbose:  # If verbose flag set
            overrides.setdefault("execution", {})["verbose"] = True  # Enable verbose
        if args.no_sound:  # If no_sound flag set
            overrides.setdefault("sound", {})["enabled"] = False  # Disable sound
        
        if args.mode is not None:  # If mode specified
            overrides.setdefault("wgangp", {})["mode"] = args.mode  # Set mode
        if args.csv_path is not None:  # If csv_path specified
            overrides.setdefault("wgangp", {})["csv_path"] = args.csv_path  # Set csv_path
        if args.label_col is not None:  # If label_col specified
            overrides.setdefault("wgangp", {})["label_col"] = args.label_col  # Set label_col
        if args.feature_cols is not None:  # If feature_cols specified
            overrides.setdefault("wgangp", {})["feature_cols"] = args.feature_cols  # Set feature_cols
        if args.seed is not None:  # If seed specified
            overrides.setdefault("wgangp", {})["seed"] = args.seed  # Set seed
        if args.force_cpu:  # If force_cpu flag set
            overrides.setdefault("wgangp", {})["force_cpu"] = True  # Enable force_cpu
        if args.from_scratch:  # If from_scratch flag set
            overrides.setdefault("wgangp", {})["from_scratch"] = True  # Enable from_scratch
        
        if args.out_dir is not None:  # If out_dir specified
            overrides.setdefault("paths", {})["out_dir"] = args.out_dir  # Set out_dir
        if args.logs_dir is not None:  # If logs_dir specified
            overrides.setdefault("paths", {})["logs_dir"] = args.logs_dir  # Set logs_dir
        
        if args.epochs is not None:  # If epochs specified
            overrides.setdefault("training", {})["epochs"] = args.epochs  # Set epochs
        if args.batch_size is not None:  # If batch_size specified
            overrides.setdefault("training", {})["batch_size"] = args.batch_size  # Set batch_size
        if args.critic_steps is not None:  # If critic_steps specified
            overrides.setdefault("training", {})["critic_steps"] = args.critic_steps  # Set critic_steps
        if args.lr is not None:  # If lr specified
            overrides.setdefault("training", {})["lr"] = args.lr  # Set lr
        if args.beta1 is not None:  # If beta1 specified
            overrides.setdefault("training", {})["beta1"] = args.beta1  # Set beta1
        if args.beta2 is not None:  # If beta2 specified
            overrides.setdefault("training", {})["beta2"] = args.beta2  # Set beta2
        if args.lambda_gp is not None:  # If lambda_gp specified
            overrides.setdefault("training", {})["lambda_gp"] = args.lambda_gp  # Set lambda_gp
        if args.save_every is not None:  # If save_every specified
            overrides.setdefault("training", {})["save_every"] = args.save_every  # Set save_every
        if args.log_interval is not None:  # If log_interval specified
            overrides.setdefault("training", {})["log_interval"] = args.log_interval  # Set log_interval
        if args.sample_batch is not None:  # If sample_batch specified
            overrides.setdefault("training", {})["sample_batch"] = args.sample_batch  # Set sample_batch
        if args.use_amp:  # If use_amp flag set
            overrides.setdefault("training", {})["use_amp"] = True  # Enable use_amp
        if args.compile:  # If compile flag set
            overrides.setdefault("training", {})["compile"] = True  # Enable compile
        
        if args.latent_dim is not None:  # If latent_dim specified
            overrides.setdefault("generator", {})["latent_dim"] = args.latent_dim  # Set latent_dim
        if args.g_hidden is not None:  # If g_hidden specified
            overrides.setdefault("generator", {})["hidden_dims"] = args.g_hidden  # Set hidden_dims
        if args.g_embed_dim is not None:  # If g_embed_dim specified
            overrides.setdefault("generator", {})["embed_dim"] = args.g_embed_dim  # Set embed_dim
        if args.n_resblocks is not None:  # If n_resblocks specified
            overrides.setdefault("generator", {})["n_resblocks"] = args.n_resblocks  # Set n_resblocks
        if args.g_leaky_relu_alpha is not None:  # If g_leaky_relu_alpha specified
            overrides.setdefault("generator", {})["leaky_relu_alpha"] = args.g_leaky_relu_alpha  # Set leaky_relu_alpha
        
        if args.d_hidden is not None:  # If d_hidden specified
            overrides.setdefault("discriminator", {})["hidden_dims"] = args.d_hidden  # Set hidden_dims
        if args.d_embed_dim is not None:  # If d_embed_dim specified
            overrides.setdefault("discriminator", {})["embed_dim"] = args.d_embed_dim  # Set embed_dim
        if args.d_leaky_relu_alpha is not None:  # If d_leaky_relu_alpha specified
            overrides.setdefault("discriminator", {})["leaky_relu_alpha"] = args.d_leaky_relu_alpha  # Set leaky_relu_alpha
        
        if args.checkpoint is not None:  # If checkpoint specified
            overrides.setdefault("generation", {})["checkpoint"] = args.checkpoint  # Set checkpoint
        if args.n_samples is not None:  # If n_samples specified
            overrides.setdefault("generation", {})["n_samples"] = args.n_samples  # Set n_samples
        if getattr(args, "force_new_samples", False):  # If force_new_samples flag set
            overrides.setdefault("generation", {})["force_new_samples"] = True  # Set force_new_samples
        if args.gen_label is not None:  # If gen_label specified
            overrides.setdefault("generation", {})["label"] = args.gen_label  # Set label
        if args.out_file is not None:  # If out_file specified
            overrides.setdefault("generation", {})["out_file"] = args.out_file  # Set out_file
        if args.gen_batch_size is not None:  # If gen_batch_size specified
            overrides.setdefault("generation", {})["gen_batch_size"] = args.gen_batch_size  # Set gen_batch_size
        if args.feature_dim is not None:  # If feature_dim specified
            overrides.setdefault("generation", {})["feature_dim"] = args.feature_dim  # Set feature_dim
        
        if args.num_workers is not None:  # If num_workers specified
            overrides.setdefault("dataloader", {})["num_workers"] = args.num_workers  # Set num_workers
        
        if args.remove_zero_variance:  # If remove_zero_variance flag set
            overrides.setdefault("dataset", {})["remove_zero_variance"] = True  # Enable remove_zero_variance
        if args.no_remove_zero_variance:  # If no_remove_zero_variance flag set
            overrides.setdefault("dataset", {})["remove_zero_variance"] = False  # Disable remove_zero_variance
        
        return overrides  # Return overrides dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def set_seed(seed: int):
    """
    Sets random seeds for reproducibility across all libraries.

    :param seed: The seed value to use for all random number generators
    :return: None
    """

    try:
        random.seed(seed)  # Set Python random seed for reproducibility
        np.random.seed(seed)  # Set NumPy random seed for reproducibility
        torch.manual_seed(seed)  # Set PyTorch CPU seed for reproducibility
        torch.cuda.manual_seed_all(seed)  # Set CUDA seed for all devices
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def verify_filepath_exists(filepath):
    """
    Verify if a file or folder exists at the specified path.

    :param filepath: Path to the file or folder
    :return: True if the file or folder exists, False otherwise
    """

    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Verifying if the file or folder exists at the path: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}"
        )  # Output the verbose message

        return os.path.exists(filepath)  # Return True if the file or folder exists, False otherwise
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_files_to_process(directory_path, file_extension=".csv", config: Optional[Dict] = None):
    """
    Collect all files with a given extension inside a directory (non-recursive).

    Performs validation, respects ignore_files configuration, and optionally filters by
    match_filenames_to_process when defined.

    :param directory_path: Path to the directory to scan
    :param file_extension: File extension to include (default: ".csv")
    :param config: Optional configuration dictionary containing execution settings
    :return: Sorted list of matching file paths
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG or get_default_config()  # Use global or default config
        
        verbose_output(
            f"{BackgroundColors.GREEN}Getting all {BackgroundColors.CYAN}{file_extension}{BackgroundColors.GREEN} files in: {BackgroundColors.CYAN}{directory_path}{Style.RESET_ALL}",
            config=config
        )  # Verbose: starting file collection
        verify_filepath_exists(directory_path)  # Validate directory path exists

        if not os.path.isdir(directory_path):  # Check if path is a valid directory
            verbose_output(
                f"{BackgroundColors.RED}Not a directory: {BackgroundColors.CYAN}{directory_path}{Style.RESET_ALL}",
                config=config
            )  # Verbose: invalid directory
            return []  # Return empty list for invalid paths

        ignore_files = config.get("execution", {}).get("ignore_files", [])  # Get ignore files list from config
        match_filenames = config.get("execution", {}).get("match_filenames_to_process", [""])  # Get match filenames list from config
        
        match_names = (
            set(match_filenames) if match_filenames not in ([], [""], [" "]) else None
        )  # Load match list or None
        if match_names:  # If filtering is to be applied
            verbose_output(
                f"{BackgroundColors.GREEN}Filtering to filenames: {BackgroundColors.CYAN}{match_names}{Style.RESET_ALL}",
                config=config
            )  # Verbose: applying filename filter

        files = []  # Accumulator for valid files

        for item in os.listdir(directory_path):  # Iterate directory entries
            item_path = os.path.join(directory_path, item)  # Absolute path
            filename = os.path.basename(item_path)  # Extract just the filename

            if any(ignore == filename or ignore == item_path for ignore in ignore_files):  # Check if file is in ignore list
                verbose_output(
                    f"{BackgroundColors.YELLOW}Ignoring {BackgroundColors.CYAN}{filename}{BackgroundColors.YELLOW} (listed in ignore_files){Style.RESET_ALL}",
                    config=config
                )  # Verbose: ignoring file
                continue  # Skip ignored file

            if os.path.isfile(item_path) and item.lower().endswith(file_extension):  # File matches extension requirement
                if (
                    match_names is not None and filename not in match_names
                ):  # Filename not included in match_filenames_to_process
                    verbose_output(
                        f"{BackgroundColors.YELLOW}Skipping {BackgroundColors.CYAN}{filename}{BackgroundColors.YELLOW} (not in match_filenames_to_process){Style.RESET_ALL}",
                        config=config
                    )  # Verbose: skipping non-matching file
                    continue  # Skip this file
                files.append(item_path)  # Add file to result list

        return sorted(files)  # Return sorted list for deterministic output
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def gradient_penalty(critic, real_samples, fake_samples, labels, device, config: Optional[Dict] = None):
    """
    Compute the WGAN-GP gradient penalty.

    :param critic: critic network callable that accepts (samples, labels)
    :param real_samples: tensor of real samples (B, feature_dim)
    :param fake_samples: tensor of fake samples (B, feature_dim)
    :param labels: tensor of integer labels (B,)
    :param device: torch device to run computations on
    :param config: Optional configuration dictionary containing gradient penalty epsilon
    :return: scalar gradient penalty term
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG or get_default_config()  # Use global or default config
        
        epsilon = float(config.get("gradient_penalty", {}).get("epsilon", 1e-12))  # Get epsilon from config and cast to float
        
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
        grad_norm = torch.sqrt(torch.sum(grads**2, dim=1) + epsilon)  # Compute L2 norm of gradients with epsilon
        gp = ((grad_norm - 1) ** 2).mean()  # Calculate gradient penalty term
        return gp  # Return scalar gradient penalty
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def plot_training_metrics(metrics_history, out_dir, filename=None, config: Optional[Dict] = None):
    """
    Plot training metrics and save to output directory.

    :param metrics_history: dictionary containing lists of metrics over training
    :param out_dir: directory to save plots
    :param filename: name of the plot file (default: from config or "training_metrics.png")
    :param config: Optional configuration dictionary containing plotting settings
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG or get_default_config()  # Use global or default config
        
        if not config.get("plotting", {}).get("enabled", True):  # If plotting disabled
            return  # Skip plotting
        
        if filename is None:  # If no filename provided
            filename = config.get("plotting", {}).get("filename", "training_metrics.png")  # Get filename from config
        
        figsize = config.get("plotting", {}).get("figsize", [18, 10])  # Get figure size
        dpi = int(config.get("plotting", {}).get("dpi", 300))  # Get DPI and cast to int
        subplot_rows = int(config.get("plotting", {}).get("subplot_rows", 2))  # Get subplot rows and cast to int
        subplot_cols = int(config.get("plotting", {}).get("subplot_cols", 3))  # Get subplot columns and cast to int
        linewidth = float(config.get("plotting", {}).get("linewidth", 1.5))  # Get line width and cast to float
        alpha = float(config.get("plotting", {}).get("alpha", 0.7))  # Get alpha and cast to float
        grid_alpha = float(config.get("plotting", {}).get("grid_alpha", 0.3))  # Get grid alpha and cast to float
    
        fig, axes = plt.subplots(subplot_rows, subplot_cols, figsize=figsize)  # Create subplot grid
        fig.suptitle("WGAN-GP Training Metrics", fontsize=16, fontweight="bold")  # Add main title

        steps = metrics_history["steps"]  # Get step numbers

        axes[0, 0].plot(steps, metrics_history["loss_D"], color="blue", linewidth=linewidth, alpha=alpha)  # Plot loss_D
        axes[0, 0].set_title("Discriminator Loss (WGAN)", fontweight="bold")  # Set subplot title
        axes[0, 0].set_xlabel("Training Step")  # Set x-axis label
        axes[0, 0].set_ylabel("Loss D")  # Set y-axis label
        axes[0, 0].grid(True, alpha=grid_alpha)  # Add grid

        axes[0, 1].plot(steps, metrics_history["loss_G"], color="red", linewidth=linewidth, alpha=alpha)  # Plot loss_G
        axes[0, 1].set_title("Generator Loss (WGAN)", fontweight="bold")  # Set subplot title
        axes[0, 1].set_xlabel("Training Step")  # Set x-axis label
        axes[0, 1].set_ylabel("Loss G")  # Set y-axis label
        axes[0, 1].grid(True, alpha=grid_alpha)  # Add grid

        axes[0, 2].plot(steps, metrics_history["gp"], color="green", linewidth=linewidth, alpha=alpha)  # Plot gradient penalty
        axes[0, 2].set_title("Gradient Penalty", fontweight="bold")  # Set subplot title
        axes[0, 2].set_xlabel("Training Step")  # Set x-axis label
        axes[0, 2].set_ylabel("GP")  # Set y-axis label
        axes[0, 2].grid(True, alpha=grid_alpha)  # Add grid

        axes[1, 0].plot(steps, metrics_history["D_real"], label="E[D(real)]", color="darkblue", linewidth=linewidth, alpha=alpha)  # Plot real scores
        axes[1, 0].plot(steps, metrics_history["D_fake"], label="E[D(fake)]", color="darkred", linewidth=linewidth, alpha=alpha)  # Plot fake scores
        axes[1, 0].set_title("Critic Scores (Real vs Fake)", fontweight="bold")  # Set subplot title
        axes[1, 0].set_xlabel("Training Step")  # Set x-axis label
        axes[1, 0].set_ylabel("Critic Score")  # Set y-axis label
        axes[1, 0].legend(loc="best")  # Add legend
        axes[1, 0].grid(True, alpha=grid_alpha)  # Add grid

        axes[1, 1].plot(steps, metrics_history["wasserstein"], color="purple", linewidth=linewidth, alpha=alpha)  # Plot Wasserstein distance
        axes[1, 1].set_title("Wasserstein Distance Estimate", fontweight="bold")  # Set subplot title
        axes[1, 1].set_xlabel("Training Step")  # Set x-axis label
        axes[1, 1].set_ylabel("W-Distance")  # Set y-axis label
        axes[1, 1].grid(True, alpha=grid_alpha)  # Add grid

        axes[1, 2].plot(steps, metrics_history["loss_D"], label="Loss D", color="blue", linewidth=linewidth, alpha=0.6)  # Plot loss_D
        axes[1, 2].plot(steps, metrics_history["loss_G"], label="Loss G", color="red", linewidth=linewidth, alpha=0.6)  # Plot loss_G
        axes[1, 2].set_title("Combined Loss View", fontweight="bold")  # Set subplot title
        axes[1, 2].set_xlabel("Training Step")  # Set x-axis label
        axes[1, 2].set_ylabel("Loss")  # Set y-axis label
        axes[1, 2].legend(loc="best")  # Add legend
        axes[1, 2].grid(True, alpha=grid_alpha)  # Add grid

        plt.tight_layout()  # Adjust spacing between subplots

        data_aug_subdir = config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation")  # Get data augmentation subdir from config
        plotting_subdir = config.get("plotting", {}).get("subdir", "plots")  # Get plotting subdir from config
        out_path = Path(out_dir)  # Convert provided out_dir to a Path object for safe operations
        if out_path.name == data_aug_subdir:  # If out_dir already ends with the data_augmentation subdir
            save_dir = out_path / plotting_subdir  # Use out_dir plus plotting subdir (avoid duplicating Data_Augmentation)
        else:  # Otherwise, out_dir does not include data_augmentation yet
            save_dir = out_path / data_aug_subdir / plotting_subdir  # Append Data_Augmentation then plotting subdir
        try:  # Try to create the save directory, but catch exceptions (e.g., permission issues, invalid path, etc.)
            save_dir.mkdir(parents=True, exist_ok=True)  # Create save directory with parents if it doesn't exist, but don't raise error if it already exists
        except Exception as e:  # Catch any exception during directory creation
            print(str(e))  # Print directory creation error to terminal for visibility
            try:  # Attempt to notify about directory creation error via Telegram
                send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full directory creation error via Telegram
            except Exception:  # If notification fails, ignore to avoid cascading errors
                pass  # Ignore Telegram send errors during directory creation fallback
            save_dir = out_path  # Fallback to out_dir if subdirectories cannot be created

        plot_path = str(save_dir / filename)  # Construct full path for the plot file
        plt.savefig(plot_path, dpi=dpi, bbox_inches="tight")  # Save figure to file
        print(f"{BackgroundColors.GREEN}Training metrics plot saved to: {BackgroundColors.CYAN}{plot_path}{Style.RESET_ALL}")  # Print save message
        plt.close()  # Close figure to free memory
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def autocast(device_type: str, enabled: bool = True):
    """
    Return an autocast context manager when enabled on CUDA, else a nullcontext.

    This avoids referencing `torch.amp.autocast` directly (Pylance warning) and
    supports environments without CUDA.

    :param device_type: The device type ("cuda" or "cpu") to create autocast context for
    :param enabled: Whether to enable autocast context (default: True)
    :return: Autocast context manager if enabled on CUDA, otherwise nullcontext
    """

    try:
        if enabled and device_type == "cuda" and _torch_autocast is not None:  # If enabled and CUDA available and autocast exists
            return _torch_autocast(device_type)  # Return CUDA autocast context
        return nullcontext()  # Return null context for CPU or when disabled
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def open_results_csv(results_csv_path, results_cols_cfg):
    """
    Open results CSV in append mode and return (file_obj, writer); write header if absent.

    This function memoizes file handles in RESULTS_CSV_HANDLES to avoid
    repeated open/close operations and to ensure header is written once.

    :param results_csv_path: Path object to results CSV
    :param results_cols_cfg: List of column names in desired order
    :return: (file_obj, csv.writer)
    """
    
    try:
        key = str(results_csv_path)  # Use string path as registry key
        if key in RESULTS_CSV_HANDLES:  # If already opened, reuse handle
            return RESULTS_CSV_HANDLES[key]  # Return cached (file_obj, writer)

        existed = results_csv_path.exists()  # Check whether file exists already
        os.makedirs(results_csv_path.parent, exist_ok=True)  # Ensure parent dir exists
        f = open(results_csv_path, "a", newline="", encoding="utf-8")  # Open file in append mode once
        writer = csv.writer(f)  # Create CSV writer for append operations
        if not existed:  # If file did not exist previously
            writer.writerow(results_cols_cfg)  # Write header row in configured order
            f.flush()  # Flush header to disk immediately
        RESULTS_CSV_HANDLES[key] = (f, writer)  # Cache file handle and writer for reuse
        return (f, writer)  # Return created handle and writer
    except Exception as _e:  # On failure, print warning and return None tuple
        print(f"{BackgroundColors.YELLOW}Warning: could not open results CSV {results_csv_path}: {_e}{Style.RESET_ALL}")  # Warn about inability to open
        return (None, None)  # Return sentinel values so callers can continue


def find_config_value(cfg, key):
    """
    Search `cfg` recursively for `key` and return first found value or None.
    
    :param cfg: Configuration dictionary or list to search through
    :param key: The key to search for in the configuration
    :return: The value associated with the key if found, otherwise None
    """
    
    try:
        if cfg is None:  # If no configuration provided
            return None  # Nothing to search
        if isinstance(cfg, dict):  # If current node is a mapping
            if key in cfg:  # If key directly exists here
                return cfg[key]  # Return direct match
            for v in cfg.values():  # Iterate values to search deeper
                try:  # Guard recursive call
                    found = find_config_value(v, key)  # Recurse into nested value
                except Exception:
                    found = None  # Ignore recursion errors
                if found is not None:  # If found in nested value
                    return found  # Return the first match
            return None  # Not found in this mapping
        if isinstance(cfg, (list, tuple)):  # If current node is a sequence
            for item in cfg:  # Iterate sequence items
                try:  # Guard recursive call
                    found = find_config_value(item, key)  # Recurse into item
                except Exception:
                    found = None  # Ignore recursion errors
                if found is not None:  # If a match is found
                    return found  # Return it
            return None  # Not found in sequence
        return None  # Base case: not a container type
    except Exception:
        return None  # On unexpected errors return None


def compose_training_start_message(args, file_progress_prefix) -> str:
    """
    Compose the training start Telegram message including CSV file statistics.

    :param args: Parsed CLI arguments; must contain `csv_path` and `epochs` attributes.
    :param file_progress_prefix: Progress prefix string to include at message start.
    :return: Single formatted f-string message including file name, sample count, file size in GB and epochs.
    """
    try:
        file_name = Path(args.csv_path).name  # Extract file name from provided CSV path
        try:  # Attempt to read CSV and count rows for total samples, but catch exceptions to avoid crashing on problematic files
            num_samples = len(pd.read_csv(args.csv_path, low_memory=False))  # Read CSV and count rows for total samples
        except Exception:  # On failure (e.g., file too large, malformed CSV, etc.), fallback to unknown sample count
            num_samples = "?"  # Use "?" to indicate unknown sample count when reading fails
        file_size_bytes = Path(args.csv_path).stat().st_size  # Get file size in bytes from filesystem
        file_size_gb = float(file_size_bytes) / (1024.0 ** 3)  # Convert bytes to gigabytes (GB)
        return f"{file_progress_prefix} Startining on {file_name} ({num_samples} samples, {file_size_gb:.2f} GB) for {args.epochs} epochs"  # Single formatted f-string as requested
    except Exception as e:
        print(str(e))  # Print exception for visibility
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception information via Telegram
        raise  # Re-raise exception to allow outer handler to manage it


def train(args, config: Optional[Dict] = None):
    """
    Train the WGAN-GP model using the provided arguments and configuration.

    :param args: parsed arguments namespace containing training configuration
    :param config: Optional configuration dictionary (will use global CONFIG if not provided)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG or get_default_config()  # Use global or default config
    
        args.lr = float(args.lr)  # Ensure learning rate is float
        args.beta1 = float(args.beta1)  # Ensure beta1 is float
        args.beta2 = float(args.beta2)  # Ensure beta2 is float
        args.lambda_gp = float(args.lambda_gp)  # Ensure lambda_gp is float
        args.n_samples = float(args.n_samples)  # Ensure n_samples is float
        args.seed = int(args.seed)  # Ensure seed is int
        args.epochs = int(args.epochs)  # Ensure epochs is int
        args.batch_size = int(args.batch_size)  # Ensure batch_size is int
        args.critic_steps = int(args.critic_steps)  # Ensure critic_steps is int
        args.save_every = int(args.save_every)  # Ensure save_every is int
        args.log_interval = int(args.log_interval)  # Ensure log_interval is int
        args.sample_batch = int(args.sample_batch)  # Ensure sample_batch is int
        args.latent_dim = int(args.latent_dim)  # Ensure latent_dim is int
        args.embed_dim = int(args.embed_dim)  # Ensure embed_dim is int
        args.n_resblocks = int(args.n_resblocks)  # Ensure n_resblocks is int
        args.gen_batch_size = int(args.gen_batch_size)  # Ensure gen_batch_size is int
        args.num_workers = int(args.num_workers)  # Ensure num_workers is int
        
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
        )  # Select device for training
        training_start_time = time.time()  # Record training session start timestamp
        file_start_time = training_start_time  # Record this file processing start timestamp
        set_seed(args.seed)  # Set random seed for reproducibility

        gpu_count = torch.cuda.device_count() if (torch.cuda.is_available() and not args.force_cpu) else 0  # Detect number of CUDA devices available
        use_dataparallel = gpu_count > 1  # Whether to use DataParallel when multiple GPUs present
        if gpu_count > 0:  # If at least one GPU is available
            torch.backends.cudnn.benchmark = True  # Enable cuDNN autotuner for potential speedups
        # Batch-size scaling heuristic: scale by 2x per GPU but cap multiplier to 8x to avoid OOM
        batch_multiplier = min(8, max(1, 2 * gpu_count)) if gpu_count > 0 else 1  # Compute safe multiplier
        scaled_batch = int(args.batch_size) * batch_multiplier  # Compute scaled batch size
        args.batch_size = int(scaled_batch)  # Apply scaled batch size to args
        # Auto-enable AMP on CUDA if available (safe, falls back on CPU since _torch_autocast may be None)
        args.use_amp = bool(args.use_amp or (gpu_count > 0 and _torch_autocast is not None))  # Enable AMP when CUDA + autocast available
        suggested_workers = min(max(1, (os.cpu_count() or 1) // 2), 32)  # Suggest a conservative default for num_workers
        file_progress_prefix = getattr(args, "file_progress_prefix", f"{BackgroundColors.CYAN}[1/1]{Style.RESET_ALL}")  # Build colored prefix (default single-file)
        # Print startup hardware and runtime configuration summary for visibility
        print(f"{BackgroundColors.GREEN}Detected {gpu_count} GPUs.{Style.RESET_ALL}")  # Print GPU count
        print(f"{BackgroundColors.GREEN}Using DataParallel: {use_dataparallel}{Style.RESET_ALL}")  # Print whether DataParallel will be used
        print(f"{BackgroundColors.GREEN}Batch size: {BackgroundColors.CYAN}{args.batch_size}{Style.RESET_ALL}")  # Print effective batch size after scaling
        print(f"{BackgroundColors.GREEN}Suggested num_workers: {BackgroundColors.CYAN}{suggested_workers}{Style.RESET_ALL}")  # Print suggested workers value
        print(f"{BackgroundColors.GREEN}AMP enabled: {BackgroundColors.CYAN}{args.use_amp}{Style.RESET_ALL}")  # Print AMP usage
        print(f"{BackgroundColors.GREEN}cuDNN benchmark: {BackgroundColors.CYAN}{torch.backends.cudnn.benchmark}{Style.RESET_ALL}")  # Print cuDNN benchmark status
        send_telegram_message(TELEGRAM_BOT, compose_training_start_message(args, file_progress_prefix))  # Telegram start with colored prefix and file statistics

        print(f"{BackgroundColors.GREEN}Device: {BackgroundColors.CYAN}{device.type.upper()}{Style.RESET_ALL}")
        if args.use_amp and device.type == "cuda":
            print(f"{BackgroundColors.GREEN}Using Automatic Mixed Precision (AMP) for faster training{Style.RESET_ALL}")
        if args.compile:
            print(f"{BackgroundColors.GREEN}Using torch.compile() for optimized execution{Style.RESET_ALL}")

        dataset = CSVFlowDataset(
            args.csv_path, label_col=args.label_col, feature_cols=args.feature_cols
        )  # Load dataset from CSV
        
        num_workers = int(config.get("dataloader", {}).get("num_workers", 8))  # Get num_workers from config and cast to int
        if device.type == "cuda" and num_workers == 0:  # If CUDA available but user set 0 workers
            num_workers = max(1, (os.cpu_count() or 1))  # Ensure at least one worker for CUDA to improve throughput
        pin_memory = True if device.type == "cuda" else False  # Always enable pin_memory on CUDA for faster host->device transfers
        persistent_workers = config.get("dataloader", {}).get("persistent_workers", True) if num_workers > 0 else False  # Get persistent_workers from config
        prefetch_factor = int(config.get("dataloader", {}).get("prefetch_factor", 2)) if num_workers > 0 else None  # Get prefetch_factor from config and cast to int
        
        dataloader = DataLoader(
            dataset,  # Dataset object to load data from
            batch_size=args.batch_size,  # Batch size for training
            shuffle=True,  # Shuffle data each epoch for better training
            drop_last=True,  # Drop last incomplete batch for consistent batch sizes
            num_workers=num_workers,  # Number of subprocesses for data loading
            pin_memory=pin_memory,  # Faster CPU->GPU transfer
            persistent_workers=persistent_workers,  # Keep workers alive between epochs
            prefetch_factor=prefetch_factor,  # Prefetch batches for better GPU utilization
        )  # Create dataloader for batching

        results_csv_file = None  # Placeholder for per-dataset open results CSV file object
        results_csv_writer = None  # Placeholder for per-dataset CSV writer
        results_cols_cfg = config.get("wgangp", {}).get("results_csv_columns", [])  # Read configured results columns list
        if not isinstance(results_cols_cfg, list) or len(results_cols_cfg) == 0:  # Validate list exists and is non-empty
            print(f"{BackgroundColors.RED}Configuration error: 'results_csv_columns' missing, empty, or not a list under 'wgangp' section in configuration.{Style.RESET_ALL}")  # Clear error message
            raise ValueError("'results_csv_columns' missing, empty, or not a list under 'wgangp' section in configuration")  # Stop safely
        if getattr(args, "csv_path", None):  # If csv_path provided, prepare persistent results CSV handle
            try:  # Attempt to open results CSV once with header written if needed
                csv_path_obj = Path(args.csv_path)  # Create Path object from csv_path
                data_aug_subdir = config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation")  # Read Data_Augmentation subdir from config
                data_aug_dir = csv_path_obj.parent / data_aug_subdir  # Construct Data_Augmentation directory under dataset folder
                os.makedirs(data_aug_dir, exist_ok=True)  # Ensure Data_Augmentation directory exists before writing
                results_csv_path = data_aug_dir / "data_augmentation_results.csv"  # Place results CSV inside Data_Augmentation dir
                results_csv_file, results_csv_writer = open_results_csv(results_csv_path, results_cols_cfg)  # Open and cache writer
            except Exception as _rw:  # On failure, warn and continue without persistent csv
                print(f"{BackgroundColors.YELLOW}Warning: could not initialize results CSV writer: {_rw}{Style.RESET_ALL}")  # Warn and continue

        feature_dim = dataset.feature_dim  # Get feature dimensionality from dataset
        n_classes = dataset.n_classes  # Get number of label classes from dataset

        g_leaky_relu_alpha = float(config.get("generator", {}).get("leaky_relu_alpha", 0.2))  # Get generator LeakyReLU alpha and cast to float
        d_leaky_relu_alpha = float(config.get("discriminator", {}).get("leaky_relu_alpha", 0.2))  # Get discriminator LeakyReLU alpha and cast to float
        
        G = Generator(
            latent_dim=args.latent_dim,  # Noise vector dimensionality for generator input
            feature_dim=feature_dim,  # Dimensionality of the output features (matches dataset)
            n_classes=n_classes,  # Number of classes for conditional generation (matches dataset)
            hidden_dims=args.g_hidden,  # List of hidden layer sizes for generator MLP
            embed_dim=args.embed_dim,  # Dimensionality of label embedding in generator
            n_resblocks=args.n_resblocks,  # Number of residual blocks in generator architecture
            leaky_relu_alpha=g_leaky_relu_alpha,  # Use config value
        ).to(
            device
        )  # Initialize generator model
        D = Discriminator(
            feature_dim=feature_dim, n_classes=n_classes, hidden_dims=args.d_hidden, embed_dim=args.embed_dim,
            leaky_relu_alpha=d_leaky_relu_alpha,  # Use config value
        ).to(
            device
        )  # Initialize discriminator model

        if torch.cuda.is_available() and not args.force_cpu and torch.cuda.device_count() > 1:  # Check multi-GPU condition
            G = torch.nn.DataParallel(G)  # Wrap generator in DataParallel to utilize multiple GPUs
            D = torch.nn.DataParallel(D)  # Wrap discriminator in DataParallel to utilize multiple GPUs
            print(f"{BackgroundColors.GREEN}Wrapped models using DataParallel across {torch.cuda.device_count()} GPUs{Style.RESET_ALL}")  # Notify wrapping

        if args.compile and not isinstance(G, torch.nn.DataParallel):  # Only compile when not using DataParallel (compile may not be compatible)
            try:  # Try compiling models, but catch exceptions if torch.compile() is not available or fails
                G = torch.compile(G, mode="reduce-overhead")  # Compile generator for performance
                D = torch.compile(D, mode="reduce-overhead")  # Compile discriminator for performance
                print(f"{BackgroundColors.GREEN}Models compiled successfully{Style.RESET_ALL}")  # Notify successful compilation
            except Exception as e:  # Catch any exception during compilation (e.g., torch.compile not available)
                print(f"{BackgroundColors.YELLOW}torch.compile() not available or failed: {e}{Style.RESET_ALL}")  # Warn but continue

        scaler = torch.cuda.amp.GradScaler() if args.use_amp and device.type == "cuda" else None  # Initialize gradient scaler for AMP if enabled and on CUDA

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

        metrics_history = {
            "steps": [],  # Training step numbers
            "loss_D": [],  # Discriminator loss values
            "loss_G": [],  # Generator loss values
            "gp": [],  # Gradient penalty values
            "D_real": [],  # Average critic score for real samples
            "D_fake": [],  # Average critic score for fake samples
            "wasserstein": [],  # Estimated Wasserstein distance (D_real - D_fake)
        }  # Dictionary to store training metrics

        if not args.from_scratch and args.csv_path:  # If not forcing from scratch and CSV path provided
            csv_path_obj = Path(args.csv_path)  # Create Path object from csv_path
            checkpoint_dir = csv_path_obj.parent / "Data_Augmentation" / "Checkpoints"  # Expected checkpoint directory
            checkpoint_prefix = csv_path_obj.stem  # Expected filename prefix
            
            if checkpoint_dir.exists():  # If checkpoint directory exists
                checkpoint_files = sorted(checkpoint_dir.glob(f"{checkpoint_prefix}_generator_epoch*.pt"))  # Find matching checkpoints
                
                if checkpoint_files:  # If checkpoints found for this file
                    g_checkpoint_path = checkpoint_files[-1]  # Get latest checkpoint
                    epoch_num = g_checkpoint_path.stem.split("epoch")[-1]  # Extract epoch number
                    d_checkpoint_path = checkpoint_dir / f"{checkpoint_prefix}_discriminator_epoch{epoch_num}.pt"  # Build discriminator path
                    
                    print(f"{BackgroundColors.CYAN}Found existing checkpoints for {csv_path_obj.name}{Style.RESET_ALL}")
                    print(f"{BackgroundColors.CYAN}Attempting to resume from epoch {epoch_num}...{Style.RESET_ALL}")
                    
                    if g_checkpoint_path.exists():  # If generator checkpoint exists
                        try:  # Try to load checkpoint
                            print(f"{BackgroundColors.GREEN}Loading generator checkpoint: {g_checkpoint_path.name}{Style.RESET_ALL}")
                            g_checkpoint = torch.load(g_checkpoint_path, map_location=device, weights_only=False)  # Load generator checkpoint with sklearn objects
                            if hasattr(cast(Any, G), "module"):  # If model wrapped by DataParallel
                                cast(Any, G).module.load_state_dict(g_checkpoint["state_dict"])  # Restore generator weights into module
                            else:  # Not DataParallel
                                cast(Any, G).load_state_dict(g_checkpoint["state_dict"])  # Restore generator weights
                            start_epoch = g_checkpoint["epoch"]  # Set starting epoch
                            
                            if "opt_G_state" in g_checkpoint:  # If optimizer state saved
                                opt_G.load_state_dict(g_checkpoint["opt_G_state"])  # Restore generator optimizer
                                print(f"{BackgroundColors.GREEN}✓ Restored generator optimizer state{Style.RESET_ALL}")
                            
                            metrics_loaded = False  # Flag to track if metrics were loaded
                            if "metrics_history" in g_checkpoint:  # If metrics history saved in checkpoint
                                metrics_history = g_checkpoint["metrics_history"]  # Restore metrics from checkpoint
                                step = metrics_history["steps"][-1] if metrics_history["steps"] else 0  # Restore step counter
                                metrics_loaded = True  # Mark as loaded
                                print(f"{BackgroundColors.GREEN}✓ Restored metrics history from checkpoint ({len(metrics_history["steps"])} steps){Style.RESET_ALL}")
                            else:  # Try loading from separate JSON file
                                metrics_json_path = checkpoint_dir / f"{checkpoint_prefix}_metrics_history.json"  # Path to metrics JSON
                                if metrics_json_path.exists():  # If JSON file exists
                                    try:  # Try to load metrics
                                        with open(metrics_json_path, "r") as f:  # Open file for reading
                                            metrics_history = json.load(f)  # Load metrics from JSON
                                        step = metrics_history["steps"][-1] if metrics_history["steps"] else 0  # Restore step counter
                                        metrics_loaded = True  # Mark as loaded
                                        print(f"{BackgroundColors.GREEN}✓ Restored metrics history from JSON file ({len(metrics_history['steps'])} steps){Style.RESET_ALL}")
                                    except Exception as e:  # If loading fails
                                        print(f"{BackgroundColors.YELLOW}⚠ Warning: Failed to load metrics from JSON: {e}{Style.RESET_ALL}")
                            
                            if scaler is not None and "scaler_state" in g_checkpoint:  # If using AMP and scaler state saved
                                scaler.load_state_dict(g_checkpoint["scaler_state"])  # Restore scaler state
                                print(f"{BackgroundColors.GREEN}✓ Restored AMP scaler state{Style.RESET_ALL}")
                            
                            if d_checkpoint_path.exists():  # If discriminator checkpoint exists
                                print(f"{BackgroundColors.GREEN}Loading discriminator checkpoint: {d_checkpoint_path.name}{Style.RESET_ALL}")
                                d_checkpoint = torch.load(d_checkpoint_path, map_location=device, weights_only=False)  # Load discriminator checkpoint
                                if hasattr(cast(Any, D), "module"):  # If discriminator wrapped by DataParallel
                                    cast(Any, D).module.load_state_dict(d_checkpoint["state_dict"])  # Restore discriminator weights into module
                                else:  # Not DataParallel
                                    cast(Any, D).load_state_dict(d_checkpoint["state_dict"])  # Restore discriminator weights
                                
                                if "opt_D_state" in d_checkpoint:  # If optimizer state saved
                                    opt_D.load_state_dict(d_checkpoint["opt_D_state"])  # Restore discriminator optimizer
                                    print(f"{BackgroundColors.GREEN}✓ Restored discriminator optimizer state{Style.RESET_ALL}")
                            else:  # Discriminator checkpoint not found
                                print(f"{BackgroundColors.YELLOW}⚠ Warning: Discriminator checkpoint not found{Style.RESET_ALL}")
                            
                            plot_dir = csv_path_obj.parent / "Data_Augmentation"  # Plot directory
                            plot_filename = csv_path_obj.stem + "_training_metrics.png"  # Plot filename
                            plot_path = plot_dir / plot_filename  # Full plot path
                            
                            if not plot_path.exists():  # If plot doesn't exist
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

        telegram_cfg = config.get("telegram", {}) if isinstance(config, dict) else {}  # Retrieve telegram config dict from merged config
        telegram_enabled = bool(telegram_cfg.get("enabled", True))  # Whether telegram notifications are enabled
        try:  # Parse progress_pct from config, ensuring it's a valid integer percentage between 1 and 100, with a default of 10%
            progress_pct = int(telegram_cfg.get("progress_pct", 10) or 10)  # Percentage step for notifications (default 10)
        except Exception:  # Catch any exception during parsing (e.g., invalid type, out of range, etc.)
            progress_pct = 10  # Fallback to 10% on parse error
        if progress_pct <= 0 or progress_pct > 100:
            progress_pct = 10  # Sanitize invalid percentage values

        next_notify = progress_pct  # Initialize next notification threshold
        try:
            percent_done = int((start_epoch / float(max(1, args.epochs))) * 100)  # Percent complete at resume point
        except Exception:
            percent_done = 0  # If anything fails, assume 0% done
        while percent_done >= next_notify and next_notify <= 100:  # Advance next_notify to the next threshold above the current progress percentage
            next_notify += progress_pct  # Skip thresholds already passed when resuming

        for epoch in range(start_epoch, args.epochs):  # Loop over epochs starting from resume point
            epoch_start_time = time.time()  # Record epoch start timestamp
            pbar = tqdm(
                dataloader, 
                desc=f"{BackgroundColors.CYAN}Epoch {epoch+1}/{args.epochs}{Style.RESET_ALL}", 
                unit="batch",
                file=sys.stdout,  # Use stdout before Logger redirection
                ncols=None,  # Auto-detect terminal width
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"  # Custom format
            )
            
            for real_x_np, labels_np in pbar:  # Loop over batches in dataloader with progress bar
                real_x = real_x_np.to(device, non_blocking=True)  # Move real features to device with non_blocking when pinned
                labels = labels_np.to(device, dtype=torch.long, non_blocking=True)  # Move labels to device with non_blocking when pinned

                loss_D = torch.tensor(0.0, device=device)  # Initialize discriminator loss
                gp = torch.tensor(0.0, device=device)  # Initialize gradient penalty
                d_real_score = torch.tensor(0.0, device=device)  # Initialize real score tracker
                d_fake_score = torch.tensor(0.0, device=device)  # Initialize fake score tracker
                
                for _ in range(args.critic_steps):  # Train discriminator multiple steps
                    with autocast(device.type, enabled=(scaler is not None)):  # Enable AMP if available
                        z = torch.randn(args.batch_size, args.latent_dim, device=device)  # Sample noise for discriminator step
                        fake_x = G(z, labels).detach()  # Generate fake samples and detach for discriminator
                        d_real = D(real_x, labels)  # Get discriminator score for real samples
                        d_fake = D(fake_x, labels)  # Get discriminator score for fake samples
                        gp = gradient_penalty(D, real_x, fake_x, labels, device, config)  # Compute gradient penalty with config
                        loss_D = d_fake.mean() - d_real.mean() + args.lambda_gp * gp  # Calculate WGAN-GP discriminator loss

                    opt_D.zero_grad()  # Zero discriminator gradients
                    if scaler is not None:  # If using mixed precision
                        scaler.scale(loss_D).backward()  # Scale loss and backpropagate
                        scaler.step(opt_D)  # Update discriminator parameters with scaled gradients
                        scaler.update()  # Update scaler for next iteration
                    else:  # Standard precision
                        loss_D.backward()  # Backpropagate discriminator loss
                        opt_D.step()  # Update discriminator parameters

                    d_real_score = d_real.mean()  # Store average real score
                    d_fake_score = d_fake.mean()  # Store average fake score

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

                pbar.set_description(
                    (
                        f"{getattr(args, 'file_progress_prefix', '')} "
                        f"{BackgroundColors.CYAN}{(Path(getattr(args, 'csv_path', '')).name if getattr(args, 'csv_path', None) else '')}{Style.RESET_ALL} | "
                    )
                    + f"{BackgroundColors.CYAN}Epoch {epoch+1}/{args.epochs}{Style.RESET_ALL} | "
                    + f"{BackgroundColors.YELLOW}step {step}{Style.RESET_ALL} | "
                    + f"{BackgroundColors.RED}loss_D: {loss_D.item():.4f}{Style.RESET_ALL} | "
                    + f"{BackgroundColors.GREEN}loss_G: {g_loss.item():.4f}{Style.RESET_ALL} | "
                    + f"gp: {gp.item():.4f} | "
                    + f"D(real): {d_real_score.item():.4f} | "
                    + f"D(fake): {d_fake_score.item():.4f}"
                )
                
                if step % args.log_interval == 0:  # Log training progress periodically
                    wasserstein_dist = (d_real_score - d_fake_score).item()  # Compute W-distance
                    
                    metrics_history["steps"].append(step)  # Record step number
                    metrics_history["loss_D"].append(loss_D.item())  # Record discriminator loss
                    metrics_history["loss_G"].append(g_loss.item())  # Record generator loss
                    metrics_history["gp"].append(gp.item())  # Record gradient penalty
                    metrics_history["D_real"].append(d_real_score.item())  # Record real score
                    metrics_history["D_fake"].append(d_fake_score.item())  # Record fake score
                    metrics_history["wasserstein"].append(wasserstein_dist)  # Record Wasserstein distance
                step += 1  # Increment global step counter

            try:  # Safely compute and print epoch elapsed time without interrupting training
                epoch_elapsed = time.time() - epoch_start_time  # Calculate epoch elapsed seconds
                print(f"{BackgroundColors.GREEN}Epoch {epoch+1} elapsed: {BackgroundColors.CYAN}{epoch_elapsed:.2f}s{Style.RESET_ALL}")  # Print epoch elapsed time
                args._last_epoch_time = float(epoch_elapsed)  # Store last epoch elapsed on args for external use
            except Exception as _te:  # If timing calculation fails
                print(f"{BackgroundColors.YELLOW}Warning: failed to measure epoch time: {_te}{Style.RESET_ALL}")  # Warn but continue

            try:  # Wrap CSV write to avoid crashing on I/O errors
                if results_csv_writer and results_cols_cfg:  # Only write if we have a valid writer and columns
                    row_runtime = {}  # Collect runtime-derived metrics into a dedicated mapping
                    row_runtime["original_file"] = Path(args.csv_path).name if getattr(args, "csv_path", None) else ""  # Original file name
                    row_runtime["epoch"] = epoch + 1  # Current epoch number
                    row_runtime["epoch_time_s"] = getattr(args, "_last_epoch_time", "")  # Epoch elapsed seconds
                    row_runtime["training_time_s"] = getattr(args, "_last_training_time", "")  # Total training elapsed seconds
                    row_runtime["file_time_s"] = getattr(args, "_last_file_time", "")  # File processing elapsed seconds
                    # Fill recognized metric placeholders safely into runtime mapping
                    row_runtime["loss_D"] = metrics_history.get("loss_D", [])[-1] if metrics_history.get("loss_D") else ""  # Last discriminator loss
                    row_runtime["loss_G"] = metrics_history.get("loss_G", [])[-1] if metrics_history.get("loss_G") else ""  # Last generator loss
                    ordered = []  # Prepare ordered list following config order
                    for c in results_cols_cfg:  # For each configured column name
                        if c in row_runtime:  # If runtime metric provides this column
                            ordered.append(row_runtime.get(c))  # Use runtime value
                        else:  # Otherwise attempt to find value in configuration
                            cfg_val = None  # Default when not found
                            try:  # Guard config lookup
                                cfg_val = find_config_value(config, c)  # Search config recursively for key
                            except Exception:
                                cfg_val = None  # On failure, treat as missing
                            if cfg_val is not None:  # If config provided a value
                                ordered.append(cfg_val)  # Use configured hyperparameter value
                            else:  # Neither runtime nor config provided the column value
                                print(f"{BackgroundColors.YELLOW}Warning: results CSV column '{c}' not found in runtime metrics or config; writing None{Style.RESET_ALL}")  # Warn about missing column
                                ordered.append(None)  # Use None to indicate missing value explicitly
                    # Inject hardware string into ordered row when hardware tracking is enabled in config
                    if config.get("hardware_tracking", False):  # If hardware tracking requested
                        try:  # Guard hardware detection to avoid breaking training
                            hw_specs = get_hardware_specifications(device_used=device)  # Query hardware specs dict
                            hw_part = hw_specs.get("gpu", "None") if hw_specs.get("gpu", None) is not None else "None"  # GPU part
                            hardware_str = (  # Build human-readable hardware string
                                f"{hw_specs.get('cpu_model','Unknown')} | Cores: {hw_specs.get('cores','N/A')}"
                                f" | RAM: {hw_specs.get('ram_gb','N/A')} GB | OS: {hw_specs.get('os','Unknown')}"
                                f" | GPU: {hw_part} | CUDA: {hw_specs.get('cuda','No')} | Device Used: {hw_specs.get('device_used','Unknown')}"
                            )  # End hardware string
                            if "hardware" in results_cols_cfg:  # If a hardware column exists in configured schema
                                try:  # Protect index operations
                                    idx_hw = results_cols_cfg.index("hardware")  # Find hardware column index
                                    if idx_hw < len(ordered):  # Ensure index is within the ordered list
                                        ordered[idx_hw] = hardware_str  # Place hardware string in row
                                except Exception:
                                    pass  # Ignore hardware insertion errors
                        except Exception:
                            pass  # Ignore any hardware detection errors to keep training running
                    results_csv_writer.writerow(ordered)  # Write ordered row following configured schema
                    try:  # Flush to disk right away to persist progressively
                        if results_csv_file is not None:  # Only call flush when file handle exists
                            try:  # Guard flush call to avoid raising
                                results_csv_file.flush()  # Flush OS buffer for safety
                            except Exception:
                                pass  # Ignore flush errors
                    except Exception:
                        pass  # Ignore outer errors as well
            except Exception as _cw:  # If writing fails, warn and continue training
                print(f"{BackgroundColors.YELLOW}Warning: failed to write epoch row to results CSV: {_cw}{Style.RESET_ALL}")  # Warn but do not abort

            if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:  # Save checkpoints periodically
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
                
                unique_labels, label_counts = np.unique(dataset.labels, return_counts=True)  # Get class distribution
                class_distribution = dict(zip(unique_labels.tolist(), label_counts.tolist()))  # Create label:count mapping
                
                g_checkpoint = {
                    "epoch": epoch + 1,  # Save current epoch number
                    "state_dict": (cast(Any, G).module.state_dict() if hasattr(cast(Any, G), "module") else cast(Any, G).state_dict()),  # Save generator state dict (unwrap DataParallel.module if present)
                    "opt_G_state": cast(Any, opt_G).state_dict(),  # Save generator optimizer state
                    "scaler": dataset.scaler,  # Save scaler for inverse transform
                    "label_encoder": dataset.label_encoder,  # Save label encoder for mapping
                    "feature_cols": dataset.feature_cols,  # Save feature column names for generation
                    "class_distribution": class_distribution,  # Save class distribution for percentage-based generation
                    "metrics_history": metrics_history,  # Save metrics history for resume
                    "args": vars(args),  # Save training arguments
                }

                if scaler is not None:  # If using AMP
                    g_checkpoint["scaler_state"] = scaler.state_dict()  # Save scaler state
                
                try:  # Time model saving to measure save phase duration
                    model_save_start_time = time.time()  # Record model save start timestamp
                    torch.save(g_checkpoint, str(g_path))  # Save generator checkpoint to disk
                    
                    d_checkpoint = {
                    "epoch": epoch + 1,  # Save current epoch number
                    "state_dict": (cast(Any, D).module.state_dict() if hasattr(cast(Any, D), "module") else cast(Any, D).state_dict()),  # Save discriminator state dict (unwrap DataParallel.module if present)
                    "opt_D_state": cast(Any, opt_D).state_dict(),  # Save discriminator optimizer state
                    "args": vars(args),  # Save training arguments
                }
                    torch.save(d_checkpoint, str(d_path))  # Save discriminator checkpoint to disk
                    latest_path = checkpoint_dir / f"{checkpoint_prefix}_generator_latest.pt"  # Path for latest generator
                    torch.save(cast(Any, G).state_dict(), str(latest_path))  # Save latest generator weights
                    model_save_elapsed = time.time() - model_save_start_time  # Compute model save elapsed seconds
                    args._last_model_save_time = float(model_save_elapsed)  # Store last model save elapsed on args
                    print(f"{BackgroundColors.GREEN}Model save elapsed: {BackgroundColors.CYAN}{model_save_elapsed:.2f}s{Style.RESET_ALL}")  # Print model save elapsed
                except Exception as _ms:  # If saving failed, warn but continue
                    print(f"{BackgroundColors.YELLOW}Warning: model save failed: {_ms}{Style.RESET_ALL}")  # Warn about save failure
                    args._last_model_save_time = ""  # Ensure attribute exists even on failure
                
                metrics_path = checkpoint_dir / f"{checkpoint_prefix}_metrics_history.json"  # Path for metrics JSON
                with open(metrics_path, "w") as f:  # Open file for writing
                    json.dump(metrics_history, f, indent=2)  # Save metrics as JSON
                print(f"{BackgroundColors.GREEN}Saved metrics history to {BackgroundColors.CYAN}{metrics_path}{Style.RESET_ALL}")  # Print metrics save message
                print(f"{BackgroundColors.GREEN}Saved generator to {BackgroundColors.CYAN}{g_path}{Style.RESET_ALL}")  # Print checkpoint save message

            try:
                if telegram_enabled and args.epochs > 0:  # Only notify when enabled and epochs is positive
                    percent = int(((epoch + 1) / float(args.epochs)) * 100)  # Compute percent completed after this epoch
                    # Send one or more notifications if multiple thresholds were crossed
                    while percent >= next_notify and next_notify <= 100:  # While we've passed the next threshold
                        msg = (
                            f"WGAN-GP training progress: {next_notify}% "  # Short progress message text
                            f"({epoch+1}/{args.epochs} epochs) on {Path(args.csv_path).name if args.csv_path else 'unknown file'}"  # Include filename and epoch info
                        )
                        send_telegram_message(TELEGRAM_BOT, msg)  # Send message via shared helper
                        next_notify += progress_pct  # Advance to next threshold to avoid duplicate sends
            except Exception as _err:  # Catch any notification errors and continue training
                pass  # Intentionally ignore notification failures to not interrupt training
        
        try:  # Safely compute total training and file elapsed times
            training_elapsed = time.time() - training_start_time  # Calculate total training elapsed seconds
            args._last_training_time = float(training_elapsed)  # Store total training elapsed on args for downstream use
            file_elapsed = time.time() - file_start_time  # Calculate file processing elapsed seconds
            args._last_file_time = float(file_elapsed)  # Store file elapsed on args for downstream use
            print(f"{BackgroundColors.GREEN}Training finished! Total training elapsed: {BackgroundColors.CYAN}{training_elapsed:.2f}s{Style.RESET_ALL}")  # Print total training elapsed message
            print(f"{BackgroundColors.GREEN}File processing elapsed: {BackgroundColors.CYAN}{file_elapsed:.2f}s{Style.RESET_ALL}")  # Print per-file elapsed message
        except Exception as _tt:  # If timing calculation fails, warn but do not interrupt
            print(f"{BackgroundColors.YELLOW}Warning: failed to compute final training/file elapsed times: {_tt}{Style.RESET_ALL}")  # Warn on failure
            args._last_training_time = ""  # Ensure attribute exists even on failure
            args._last_file_time = ""  # Ensure attribute exists even on failure

        try:  # Wrap writes to avoid crashing on I/O errors
            if results_csv_writer and results_cols_cfg:  # Only write if writer and schema are available
                final_runtime = {}  # Build runtime-only values for final per-file row
                final_runtime["original_file"] = Path(args.csv_path).name if getattr(args, "csv_path", None) else ""  # Original filename
                final_runtime["total_generated_samples"] = ""  # Placeholder, generation may fill this later
                final_runtime["training_time_s"] = getattr(args, "_last_training_time", "")  # Total training elapsed
                final_runtime["file_time_s"] = getattr(args, "_last_file_time", "")  # Per-file processing elapsed
                ordered_final = []  # Prepare ordered list according to configured schema
                for c in results_cols_cfg:  # For each configured column name
                    if c in final_runtime:  # If runtime mapping contains key
                        ordered_final.append(final_runtime.get(c))  # Use runtime value
                    else:  # Attempt to obtain from configuration when not a runtime metric
                        cfg_val = None  # Default missing
                        try:  # Guard recursive lookup
                            cfg_val = find_config_value(config, c)  # Search config for matching key
                        except Exception:
                            cfg_val = None  # On error treat as missing
                        if cfg_val is not None:  # If found in config
                            ordered_final.append(cfg_val)  # Use configured hyperparameter value
                        else:  # Not found anywhere; warn and use None
                            print(f"{BackgroundColors.YELLOW}Warning: results CSV column '{c}' not found in runtime metrics or config; writing None{Style.RESET_ALL}")  # Warn about missing column
                            ordered_final.append(None)  # Explicit None for missing value
                # Inject hardware string into final ordered row when hardware tracking is enabled in config
                if config.get("hardware_tracking", False):  # If hardware tracking requested
                    try:  # Guard hardware detection to avoid breaking flow
                        hw_specs = get_hardware_specifications(device_used=device)  # Query hardware specs dict
                        hw_part = hw_specs.get("gpu", "None") if hw_specs.get("gpu", None) is not None else "None"  # GPU part
                        hardware_str = (  # Build human-readable hardware string
                            f"{hw_specs.get('cpu_model','Unknown')} | Cores: {hw_specs.get('cores','N/A')}"
                            f" | RAM: {hw_specs.get('ram_gb','N/A')} GB | OS: {hw_specs.get('os','Unknown')}"
                            f" | GPU: {hw_part} | CUDA: {hw_specs.get('cuda','No')} | Device Used: {hw_specs.get('device_used','Unknown')}"
                        )  # End hardware string
                        if "hardware" in results_cols_cfg:  # If a hardware column exists in configured schema
                            try:  # Protect index operations
                                idx_hw = results_cols_cfg.index("hardware")  # Find hardware column index
                                if idx_hw < len(ordered_final):  # Ensure index is within the ordered_final list
                                    ordered_final[idx_hw] = hardware_str  # Place hardware string in final row
                            except Exception:
                                pass  # Ignore hardware insertion errors
                    except Exception:
                        pass  # Ignore any hardware detection errors to keep flow
                results_csv_writer.writerow(ordered_final)  # Write final per-file ordered row to CSV
                try:  # Flush buffer to persist row immediately
                    if results_csv_file is not None:  # Only call flush when file handle exists
                        try:  # Guard flush call to avoid raising
                            results_csv_file.flush()  # Flush file buffer to disk
                        except Exception:
                            pass  # Ignore flush errors
                except Exception:
                    pass  # Ignore outer errors as well
        except Exception as _fw:  # If writing fails, warn and continue
            print(f"{BackgroundColors.YELLOW}Warning: failed to write final file row to results CSV: {_fw}{Style.RESET_ALL}")  # Warn about failure
        
        if len(metrics_history["steps"]) > 0:  # If metrics were collected
            print(f"{file_progress_prefix} {BackgroundColors.GREEN}Generating training metrics plots...{Style.RESET_ALL}")  # Print plotting message with prefix
            if args.csv_path:  # If CSV path is provided
                csv_path_obj = Path(args.csv_path)  # Create Path object from csv_path
                plot_dir = csv_path_obj.parent / "Data_Augmentation"  # Create Data_Augmentation subdirectory
                os.makedirs(plot_dir, exist_ok=True)  # Ensure directory exists
                plot_filename = csv_path_obj.stem + "_training_metrics.png"  # Use input filename for plot
                original_out_dir = args.out_dir  # Save original out_dir
                args.out_dir = str(plot_dir)  # Set out_dir to Data_Augmentation
                temp_metrics = metrics_history.copy()  # Copy metrics
                plot_training_metrics(temp_metrics, str(plot_dir), plot_filename, config)  # Create and save plots with config
                args.out_dir = original_out_dir  # Restore original out_dir
            else:  # No CSV path, use default out_dir
                plot_training_metrics(metrics_history, args.out_dir, "training_metrics.png", config)  # Create and save plots with config

        send_telegram_message(TELEGRAM_BOT, f"{file_progress_prefix} Finished WGAN-GP training on {Path(args.csv_path).name} after {args.epochs} epochs")  # Telegram finish with prefix
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def apply_zebra_style(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Apply zebra-striping style to a DataFrame using pandas Styler.

    :param df: Input DataFrame to style
    :return: pandas Styler with zebra row background colors applied
    """
    try:
        # Define row styling function that returns a list of styles for each cell in the row
        def _row_style(row):  # Define helper used by Styler.apply
            # Compute background color based on row index parity
            bg = "white" if (row.name % 2) == 0 else "#f2f2f2"  # white and light gray
            # Return style for every column in the row preserving column order
            return [f"background-color: {bg};" for _ in row.index]  # Preserve columns

        styled = df.style.apply(_row_style, axis=1)  # Apply zebra function row-wise
        styled = styled.set_table_attributes('style="border-collapse:collapse; width:100%;"')  # Tight table style
        styled = styled.set_properties(**{"border": "1px solid #ddd", "padding": "6px"})  # Cell padding/border
        return styled  # Return the styled object
    except Exception as e:
        print(str(e))  # Print error for visibility
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Notify via Telegram
        raise  # Propagate error to caller


def export_dataframe_image(styled_df: pd.io.formats.style.Styler, output_path: Union[str, Path]):
    """
    Export a pandas Styler to a PNG image using dataframe_image.

    :param styled_df: pandas Styler object to export
    :param output_path: Path to output PNG file
    :raises: Any exception raised by dataframe_image.export will be propagated
    """
    try:
        out_p = Path(output_path)  # Ensure Path object for output
        # Ensure parent directory exists before writing
        out_p.parent.mkdir(parents=True, exist_ok=True)  # Create directories as needed
        # Use dataframe_image to export the styled dataframe to PNG
        dfi.export(styled_df, str(out_p))  # Export to PNG using dataframe_image
    except Exception as e:
        print(str(e))  # Print export error for visibility
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Notify via Telegram
        raise  # Do not swallow errors; propagate to caller


def generate_table_image_from_dataframe(df: pd.DataFrame, output_path: Union[str, Path]):
    """
    Generate a zebra-striped table image (.png) from an in-memory DataFrame.

    :param df: DataFrame in memory (must not be re-read from disk)
    :param output_path: Target PNG path (will be overwritten if exists)
    :raises: PermissionError if directory not writable, or any dataframe_image error
    """
    try:
        out_p = Path(output_path)  # Convert to Path for manipulation
        parent = out_p.parent  # Parent directory
        if not parent.exists():  # If parent directory does not exist
            parent.mkdir(parents=True, exist_ok=True)  # Try to create it
        # Verify directory is writable
        if not os.access(str(parent), os.W_OK):  # Check write permission
            raise PermissionError(f"Directory not writable: {parent}")  # Raise on non-writable
        styled = apply_zebra_style(df)  # Create a zebra-styled Styler from df
        export_dataframe_image(styled, out_p)  # Export styled DataFrame to PNG
    except Exception:
        raise  # Propagate all exceptions to caller (no silent failures)


def generate_csv_and_image(df: pd.DataFrame, csv_path: Union[str, Path], is_visualizable: bool = True):
    """
    Save DataFrame to CSV and optionally generate a zebra-striped PNG table image.

    This function centralizes CSV saving and image generation to avoid duplicating
    CSV writing logic across the codebase.

    :param df: DataFrame to save
    :param csv_path: Destination CSV file path
    :param is_visualizable: Flag indicating whether to generate PNG image
    :raises: Propagates IO and dataframe_image exceptions
    """
    try:
        csv_p = Path(csv_path)  # Convert csv_path to Path
        parent = csv_p.parent  # Parent directory for CSV
        parent.mkdir(parents=True, exist_ok=True)  # Ensure parent exists
        # Verify parent directory is writable before attempting to write CSV
        if not os.access(str(parent), os.W_OK):  # If not writable
            raise PermissionError(f"Directory not writable: {parent}")  # Raise permission error
        df.to_csv(str(csv_p), index=False)  # Save CSV to disk preserving DataFrame content/order
        # Only generate image when explicitly requested
        if is_visualizable:  # If a visual representation is desired
            png_path = csv_p.with_suffix('.png')  # Replace CSV extension with PNG
            generate_table_image_from_dataframe(df, png_path)  # Generate PNG from in-memory DataFrame
    except Exception:
        raise  # Propagate exceptions to caller (do not swallow)


def compose_generation_start_message(n: int, args, generated_file_name: str, original_num: Optional[int] = None):
    """
    Compose the generation start message including the ratio relative to the original dataset.

    :param n: Number of samples that will be generated
    :param args: Parsed CLI arguments (may contain n_samples and csv_path)
    :param generated_file_name: Output file name where samples will be saved
    :param original_num: Optional original dataset size (if already known)
    :return: Formatted Telegram message string
    """

    try:
        orig = original_num  # Start with provided original dataset size
        
        if orig is None:  # If original size was not explicitly provided
            csv_path = getattr(args, "csv_path", None)  # Try to get CSV path from args
            
            if csv_path:  # If CSV path exists
                try:
                    try:  # Guard CSV reading to avoid crashing if file is large or malformed
                        df = pd.read_csv(csv_path, low_memory=False)  # Load dataset safely
                    except Exception:  # If reading with low_memory fails, try again without it (may use more memory but can handle some files)
                        df = None  # Initialize df to None before second attempt
                    orig = len(df)  # Determine original number of samples
                except Exception:
                    orig = None  # If reading fails, keep as None
        
        ratio_info = ""  # Initialize ratio info
        
        if orig is not None and orig > 0:  # If original dataset size is known and valid
            ratio = float(n) / float(orig)  # Compute generation ratio
            percentage = ratio * 100.0  # Convert ratio to percentage
            ratio_info = f"{percentage:.2f}% ({n}/{orig})"  # Format ratio info
        
        elif args.n_samples is not None:  # If n_samples was explicitly provided
            try:
                requested = float(args.n_samples)  # Convert to float
                
                if requested <= 1.0:  # If decimal (percentage mode)
                    ratio_info = f"{requested * 100:.2f}%"  # Percentage requested
                else:  # If integer (absolute mode)
                    ratio_info = f"{int(requested)} samples"  # Absolute requested
            except Exception:
                ratio_info = ""  # Fallback if conversion fails
        
        if ratio_info != "":  # If ratio information is available
            return f"Starting generation: Producing {n} samples ({ratio_info}) to {generated_file_name}"  # Final formatted message
        
        return f"Starting generation: Producing {n} samples to {generated_file_name}"  # Fallback message
    
    except Exception as e:
        print(str(e))  # Print exception message
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception to Telegram
        raise

def compute_expected_samples_for_percentage(percent: float, args, config: Optional[Dict] = None) -> Optional[int]:
    """
    Compute the expected number of generated samples when n_samples is given as a percentage.

    This reproduces the same small-class logic used during generation to ensure
    consistency between verification and generation steps.

    :param percent: Percentage value in decimal form (e.g., 1.0 for 100%)
    :param args: Runtime arguments (may include checkpoint, csv_path, label_col, feature_cols)
    :param config: Optional configuration dictionary
    :return: Expected total number of samples or None if undeterminable
    """

    try:
        if config is None:  # Use global config if not provided
            config = CONFIG or get_default_config()  # Load default configuration if necessary

        class_dist = None  # Initialize class distribution container

        if getattr(args, "checkpoint", None):  # If checkpoint path exists in args
            try:
                ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)  # Load checkpoint
                class_dist = ck.get("class_distribution", None)  # Extract class distribution
            except Exception:
                class_dist = None  # Fallback if checkpoint loading fails

        if class_dist is None and getattr(args, "csv_path", None):  # If no checkpoint distribution available
            try:
                ds = CSVFlowDataset(
                    args.csv_path,  # CSV dataset path
                    label_col=getattr(
                        args,
                        "label_col",
                        config.get("wgangp", {}).get("label_col", "Label"),
                    ),  # Resolve label column
                    feature_cols=getattr(args, "feature_cols", None),  # Optional feature columns
                )
                unique, counts = np.unique(ds.labels, return_counts=True)  # Compute label counts
                class_dist = dict(zip([int(u) for u in unique.tolist()], counts.tolist()))  # Build distribution dict
            except Exception:
                class_dist = None  # Fallback if dataset loading fails

        if class_dist is None:  # If still undetermined
            return None  # Cannot compute expected value safely

        threshold = int(config.get("generation", {}).get("small_class_threshold", 100))  # Small class threshold
        small_min = int(config.get("generation", {}).get("small_class_min_samples", 10))  # Minimum small class samples

        n_per_class = {}  # Container for computed samples per class

        for label, original_count in class_dist.items():  # Iterate over classes
            calculated = int(original_count * percent)  # Percentage-based computation
            final_count = max(
                small_min if original_count < threshold else 1,
                calculated,
            )  # Apply small-class logic
            n_per_class[int(label)] = final_count  # Store per-class result

        return sum(n_per_class.values())  # Return total expected samples

    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def verify_data_augmentation_file(args, config: Optional[Dict] = None) -> bool:
    """
    Verify whether the data-augmentation output file exists and matches the
    configured number of samples. Returns True when generation should proceed
    (file missing, mismatched, or forced), or False when generation can be
    skipped because an existing file already matches the requested size.

    Behavior:
      - If output file does not exist: return True (proceed)
      - If output file exists and equals expected sample count: return False
        unless `force_new_samples` is True (in that case delete and return True)
      - If output file exists but count mismatches: delete file and return True

    :param args: Runtime args (must include `out_file`, `n_samples`, `checkpoint`, `csv_path`)
    :param config: Optional configuration dictionary
    :return: bool, True => proceed with generation, False => skip generation
    """

    try:
        if config is None:  # Use global if not provided
            config = CONFIG or get_default_config()  # Load default config when needed

        out_path = Path(getattr(args, "out_file", ""))  # Resolve output path
        file_prefix = getattr(args, "file_progress_prefix", "")  # Telegram prefix

        if not out_path.exists():  # If file does not exist
            return True  # Proceed with generation

        try:
            requested = float(
                getattr(args, "n_samples", None)
                if getattr(args, "n_samples", None) is not None
                else config.get("generation", {}).get("n_samples", 1.0)
            )  # Resolve requested n_samples
        except Exception:
            requested = config.get("generation", {}).get("n_samples", 1.0)  # Fallback

        expected_n: Optional[int] = None  # Initialize expected sample count

        if requested <= 1.0:  # Percentage mode
            expected_n = compute_expected_samples_for_percentage(requested, args, config)  # Compute expected
        else:
            try:
                expected_n = int(requested)  # Absolute mode
            except Exception:
                expected_n = None  # Fallback if conversion fails

        if expected_n is None:  # If expected size cannot be determined
            send_telegram_message(
                TELEGRAM_BOT,
                f"{file_prefix} Unable to verify existing augmented file {out_path.name}: expected count undeterminable — regenerating.",
            )
            try:
                out_path.unlink()  # Attempt to remove problematic file
            except Exception:
                pass
            return True  # Proceed safely

        try:
            existing_count = len(pd.read_csv(out_path, low_memory=False))  # Count rows safely
        except Exception:
            existing_count = None  # Fallback if unreadable

        if existing_count is None:  # If file unreadable
            send_telegram_message(
                TELEGRAM_BOT,
                f"{file_prefix} Existing augmented file {out_path.name} unreadable — regenerating.",
            )
            try:
                out_path.unlink()  # Attempt deletion
            except Exception:
                pass
            return True  # Proceed

        if existing_count == expected_n and not getattr(args, "force_new_samples", False):  # Matching and no force
            send_telegram_message(
                TELEGRAM_BOT,
                f"{file_prefix} Skipping Generation: {out_path.name} already exists with {existing_count} samples (expected {expected_n}).",
            )
            return False  # Skip generation

        if getattr(args, "force_new_samples", False):  # Forced regeneration
            send_telegram_message(
                TELEGRAM_BOT,
                f"{file_prefix} Force Regeneration Requested: Removing existing {out_path.name} ({existing_count} samples) and regenerating to {expected_n}.",
            )
        else:
            send_telegram_message(
                TELEGRAM_BOT,
                f"{file_prefix} Existing {out_path.name} has {existing_count} samples but expected {expected_n}; removing and regenerating.",
            )

        try:
            out_path.unlink()  # Delete existing file
        except Exception:
            pass

        return True  # Proceed with generation

    except Exception as e:
        print(str(e))
        try:
            send_exception_via_telegram(type(e), e, e.__traceback__)
        except Exception:
            pass
        return True


def generate(args, config: Optional[Dict] = None):
    """
    Generate synthetic samples from a saved generator checkpoint.

    :param args: parsed arguments namespace containing generation options
    :param config: Optional configuration dictionary (will use global CONFIG if not provided)
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG or get_default_config()  # Use global or default config
    
        args.latent_dim = int(args.latent_dim)  # Ensure latent_dim is int
        args.embed_dim = int(args.embed_dim)  # Ensure embed_dim is int
        args.n_resblocks = int(args.n_resblocks)  # Ensure n_resblocks is int
        args.gen_batch_size = int(args.gen_batch_size)  # Ensure gen_batch_size is int
        if args.feature_dim is not None:  # If feature_dim provided
            args.feature_dim = int(args.feature_dim)  # Ensure feature_dim is int
        if args.n_samples is not None:  # If n_samples provided
            args.n_samples = float(args.n_samples)  # Ensure n_samples is float
        if args.label is not None:  # If label provided
            args.label = int(args.label)  # Ensure label is int
        
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
        )  # Select device for generation

        file_progress_prefix = getattr(args, "file_progress_prefix", f"{BackgroundColors.CYAN}[1/1]{Style.RESET_ALL}")  # Build colored prefix (default single-file)
        send_telegram_message(TELEGRAM_BOT, f"{file_progress_prefix} Starting WGAN-GP generation from {Path(args.checkpoint).name}")  # Telegram start with colored prefix
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
        else:  # Try to infer feature dimension from scaler's mean_ attribute if available
            mean_attr = getattr(scaler, "mean_", None) if scaler is not None else None  # Get mean_ attribute from scaler if available
            if mean_attr is not None:  # If mean_ attribute exists, infer feature dimension from it
                mean_arr = np.asarray(mean_attr)  # Convert mean_ to numpy array
                if mean_arr.ndim == 0:  # If mean_ is scalar, cannot infer feature dimension
                    raise RuntimeError(
                        "Scaler.mean_ is scalar; unable to infer feature dimension. Provide --feature_dim."
                    )  # Raise error if mean_ is not an array
                feature_dim = int(mean_arr.shape[0])  # Infer feature dimension from scaler
            else:  # If no way to infer feature dimension, raise error
                raise RuntimeError(
                    "Unable to determine feature dimension; provide --feature_dim or a checkpoint with scaler."
                )  # Raise error if not available
        n_classes = len(label_encoder.classes_)  # Get number of classes from label encoder

        g_leaky_relu_alpha = float(config.get("generator", {}).get("leaky_relu_alpha", 0.2))  # Get generator LeakyReLU alpha and cast to float
        
        G = Generator(
            latent_dim=args.latent_dim,  # Noise vector dimensionality for generator input
            feature_dim=feature_dim,  # Dimensionality of the output features (matches dataset)
            n_classes=n_classes,  # Number of classes for conditional generation (matches dataset)
            hidden_dims=args.g_hidden,  # List of hidden layer sizes for generator MLP
            embed_dim=args.embed_dim,  # Dimensionality of label embedding in generator
            n_resblocks=args.n_resblocks,  # Number of residual blocks in generator architecture
            leaky_relu_alpha=g_leaky_relu_alpha,  # Use config value
        ).to(
            device
        )  # Initialize generator model
        if hasattr(G, "module"):  # If generator wrapped by DataParallel
            G.module.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)  # Load into underlying module
        else:  # Not DataParallel
            G.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)  # Load generator weights from checkpoint
        G.eval()  # Set generator to evaluation mode

        small_class_threshold = int(config.get("generation", {}).get("small_class_threshold", 100))  # Get small class threshold and cast to int
        small_class_min_samples = int(config.get("generation", {}).get("small_class_min_samples", 10))  # Get min samples for small classes and cast to int
        
        if args.n_samples <= 1.0:  # Percentage mode: generate percentage of training data per class (1.0 == 100%)
            if class_distribution is None:  # If class distribution not available
                raise RuntimeError(
                    "Percentage-based generation requires class_distribution in checkpoint or --csv_path to calculate it."
                )  # Raise error
            print(f"{BackgroundColors.CYAN}Generating {args.n_samples*100:.1f}% of training data per class (min {small_class_min_samples} samples for small classes){Style.RESET_ALL}")
            if args.label is not None:  # If specific label requested
                if args.label not in class_distribution:  # Verify label exists
                    raise ValueError(f"Label {args.label} not found in training data class distribution")  # Raise error
                original_count = class_distribution[args.label]  # Get original class count
                calculated = int(original_count * args.n_samples)  # Calculate percentage-based count
                final_count = max(small_class_min_samples if original_count < small_class_threshold else 1, calculated)  # Apply minimum threshold
                n_per_class = {args.label: final_count}  # Store final count
            else:  # Generate for all classes
                n_per_class = {}  # Initialize dictionary
                for label, original_count in class_distribution.items():  # For each class
                    calculated = int(original_count * args.n_samples)  # Calculate percentage-based count
                    final_count = max(small_class_min_samples if original_count < small_class_threshold else 1, calculated)  # Apply minimum threshold
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
            else:  # No specific label, sample uniformly across all classes
                labels = np.random.randint(0, n_classes, size=(n,), dtype=np.int64)  # Sample labels uniformly

        batch_size = args.gen_batch_size  # Set generation batch size
        try:
            start_msg = compose_generation_start_message(n, args, Path(args.out_file).name, original_num=None, class_distribution=class_distribution, args_ck=args_ck)
            send_telegram_message(TELEGRAM_BOT, start_msg)  # Notify start of generation
        except Exception as e:  # Failed to notify start of generation
            print(str(e))  # Print send error to terminal for visibility
            try:  # Attempt to send full error via Telegram using exception sender
                send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full send error via Telegram
            except Exception:  # If notification fails, continue without raising to allow generation
                pass  # Ignore Telegram send errors and continue generation
        all_fake = []  # List to store generated feature batches
        all_labels = []  # List to store corresponding labels
        sample_generation_start_time = time.time()  # Record sample generation start timestamp
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

        df = pd.DataFrame(X_orig, columns=feature_cols)  # Create DataFrame with original feature names
        df[args.label_col] = label_encoder.inverse_transform(Y_fake)  # Map integer labels back to original strings
        # Optionally populate hardware column on generated DataFrame when hardware tracking is enabled
        if config.get("hardware_tracking", False):  # If enabled in config
            try:  # Guard hardware population to avoid breaking generation
                df = populate_hardware_column(df, column_name="hardware", device_used=device)  # Populate hardware column
            except Exception:
                pass  # Ignore hardware population errors and continue
        generate_csv_and_image(df, args.out_file, is_visualizable=True)  # Save CSV and generate PNG image when appropriate
        print(f"{file_progress_prefix} {BackgroundColors.GREEN}Saved {BackgroundColors.CYAN}{n}{BackgroundColors.GREEN} generated samples to {BackgroundColors.CYAN}{args.out_file}{Style.RESET_ALL}")  # Print completion message with prefix

        try:  # Safely compute and print sample generation elapsed time
            sample_generation_elapsed = time.time() - sample_generation_start_time  # Calculate sample generation elapsed seconds
            args._last_sample_generation_time = float(sample_generation_elapsed)  # Store sample generation elapsed on args for downstream use
            print(f"{BackgroundColors.GREEN}Sample generation elapsed: {BackgroundColors.CYAN}{sample_generation_elapsed:.2f}s{Style.RESET_ALL}")  # Print sample generation elapsed
        except Exception as _sg:  # If timing calculation fails
            print(f"{BackgroundColors.YELLOW}Warning: failed to measure sample generation time: {_sg}{Style.RESET_ALL}")  # Warn but continue
            args._last_sample_generation_time = ""  # Ensure attribute exists even on failure

        send_telegram_message(TELEGRAM_BOT, f"{file_progress_prefix} Finished WGAN-GP generation: Saved {n} samples ({(f'{n/original_num:.2%}, ' if 'original_num' in locals() and original_num else '')}{(Path(args.out_file).stat().st_size >= 1024**3 and f'{Path(args.out_file).stat().st_size/1024**3:.2f} GB' or f'{Path(args.out_file).stat().st_size/1024**2:.2f} MB')}) to {Path(args.out_file).name}")  # Telegram finish with prefix
        
        try:  # Wrap result writing in try/except to avoid breaking generation on failures
            results_cols_cfg = config.get("wgangp", {}).get("results_csv_columns", [])  # Read configured results columns list
            if not isinstance(results_cols_cfg, list) or len(results_cols_cfg) == 0:  # Validate list exists and is non-empty
                print(f"{BackgroundColors.RED}Configuration error: 'results_csv_columns' missing, empty, or not a list under 'wgangp' section in configuration.{Style.RESET_ALL}")  # Clear error message
                raise ValueError("'results_csv_columns' missing, empty, or not a list under 'wgangp' section in configuration")  # Stop safely
            results_csv_path = None  # Initialize results CSV path variable
            if getattr(args, "csv_path", None):  # If csv_path available on args
                csv_path_obj = Path(args.csv_path)  # Create Path object from args.csv_path
                data_aug_subdir = config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation")  # Read configured subdir name
                data_aug_dir = csv_path_obj.parent / data_aug_subdir  # Construct Data_Augmentation directory under dataset folder
                os.makedirs(data_aug_dir, exist_ok=True)  # Ensure Data_Augmentation directory exists before writing
                results_csv_path = data_aug_dir / "data_augmentation_results.csv"  # Use results CSV inside Data_Augmentation dir
            else:  # Try to recover original csv_path from checkpoint args saved in checkpoint
                try:
                    if args_ck and args_ck.get("csv_path"):  # Use saved args from checkpoint (args_ck defined earlier)
                        ck_csv_path = Path(args_ck.get("csv_path"))  # Path object for saved csv_path from checkpoint
                        ck_data_aug_subdir = config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation")  # Read subdir name from config
                        ck_data_aug_dir = ck_csv_path.parent / ck_data_aug_subdir  # Construct Data_Augmentation directory for checkpoint csv_path
                        os.makedirs(ck_data_aug_dir, exist_ok=True)  # Ensure Data_Augmentation directory exists for checkpoint recover
                        results_csv_path = ck_data_aug_dir / "data_augmentation_results.csv"  # Use results CSV inside Data_Augmentation dir for checkpoint
                except Exception:
                    results_csv_path = None  # Leave as None if recovery fails

            if results_csv_path is not None:  # If we have a place to record results
                # Compute common metric values used in CSV columns
                original_file_name = Path(args.csv_path).name if getattr(args, "csv_path", None) else ((args_ck and args_ck.get("csv_path") and Path(args_ck.get("csv_path")).name) or "")  # Original file name
                generated_file_name = Path(args.out_file).name if getattr(args, "out_file", None) else ""  # Generated file name
                try:
                    original_num = None  # Default original count
                    if getattr(args, "csv_path", None):  # If csv_path provided, try reading length
                        original_num = len(pd.read_csv(args.csv_path, low_memory=False))  # Count original CSV rows
                except Exception:
                    original_num = None  # Leave as None if reading fails

                total_generated = int(n) if n is not None else ""  # Total generated samples
                generated_ratio = ""  # Default generated ratio
                try:
                    if original_num and original_num > 0:  # If original count available
                        generated_ratio = float(total_generated) / float(original_num)  # Compute ratio
                except Exception:
                    generated_ratio = ""  # Leave blank on failure

                timing_values = {  # Map common column names to stored timing attributes
                    "training_time_s": getattr(args, "_last_training_time", ""),  # Total training elapsed seconds
                    "file_time_s": getattr(args, "_last_file_time", ""),  # Per-file processing elapsed seconds
                    "epoch_time_s": getattr(args, "_last_epoch_time", ""),  # Last epoch elapsed seconds
                    "sample_generation_time_s": getattr(args, "_last_sample_generation_time", ""),  # Sample generation elapsed seconds
                    "model_save_time_s": getattr(args, "_last_model_save_time", ""),  # Model save phase elapsed seconds
                }  # End timing values map
                training_time_val = timing_values.get("training_time_s", "")  # Pull training time in case caller expects that key
                testing_time_val = ""  # No testing time available by default

                row_runtime_defaults = {
                    "critic_iterations": getattr(args, "critic_steps", ""),
                    "learning_rate_generator": getattr(args, "lr", ""),
                    "learning_rate_critic": getattr(args, "lr", ""),
                    "testing_time_s": testing_time_val,
                    "hardware": None,
                }

                critic_loss_val = ""  # Default critic loss
                generator_loss_val = ""  # Default generator loss
                try:
                    metrics_history = ckpt.get("metrics_history")  # Try to get metrics history from checkpoint (may be None)
                    if isinstance(metrics_history, dict):  # If metrics_history is a dict
                        ld = metrics_history.get("loss_D") or []  # Safe list for discriminator losses
                        lg = metrics_history.get("loss_G") or []  # Safe list for generator losses
                        if isinstance(ld, (list, tuple)) and len(ld) > 0:  # If list-like and non-empty
                            critic_loss_val = ld[-1]  # Use last recorded discriminator loss
                        if isinstance(lg, (list, tuple)) and len(lg) > 0:  # If list-like and non-empty
                            generator_loss_val = lg[-1]  # Use last recorded generator loss
                except Exception:
                    critic_loss_val = ""  # Ignore failures and leave blank
                    generator_loss_val = ""  # Ignore failures and leave blank

                row_runtime = {}  # Dictionary to hold runtime values for configured columns
                row_runtime["original_file"] = original_file_name  # Original CSV filename
                row_runtime["generated_file"] = generated_file_name  # Generated output filename
                row_runtime["original_num_samples"] = original_num if original_num is not None else ""  # Original sample count
                row_runtime["total_generated_samples"] = total_generated  # Total generated count
                row_runtime["generated_ratio"] = generated_ratio  # Generated/original ratio
                row_runtime["critic_loss"] = critic_loss_val  # Last critic loss from checkpoint metrics
                row_runtime["generator_loss"] = generator_loss_val  # Last generator loss from checkpoint metrics
                for k, v in timing_values.items():  # For each timing key known
                    row_runtime[k] = v  # Store timing value under the timing key

                try:
                    for kk, vv in row_runtime_defaults.items():
                        if kk not in row_runtime or row_runtime.get(kk) in (None, ""):
                            row_runtime[kk] = vv
                except Exception:
                    pass

                try:
                    if (row_runtime.get("hardware") is None) or row_runtime.get("hardware") == "":
                        hw_specs = get_hardware_specifications(device_used=device) if 'get_hardware_specifications' in globals() else None
                        if isinstance(hw_specs, dict):
                            hw_part = hw_specs.get("gpu", "None") if hw_specs.get("gpu", None) is not None else "None"
                            hardware_str = (
                                f"{hw_specs.get('cpu_model','Unknown')} | Cores: {hw_specs.get('cores','N/A')}"
                                f" | RAM: {hw_specs.get('ram_gb','N/A')} GB | OS: {hw_specs.get('os','Unknown')}"
                                f" | GPU: {hw_part} | CUDA: {hw_specs.get('cuda','No')} | Device Used: {hw_specs.get('device_used','Unknown')}"
                            )
                            row_runtime["hardware"] = hardware_str
                except Exception:
                    pass

                try:  # Wrap open/write in try/except to avoid crashing on I/O issues
                    f_handle, writer = open_results_csv(results_csv_path, results_cols_cfg)  # Get persistent handle and writer
                    if f_handle and writer:  # If we successfully opened or reused a writer
                        ordered = []  # Build ordered list following configured schema
                        for c in results_cols_cfg:  # For each configured column name
                            if c in row_runtime:  # If runtime has this value
                                ordered.append(row_runtime.get(c))  # Use runtime value
                            else:  # Otherwise attempt to fetch from configuration
                                cfg_val = None  # Default to missing
                                try:  # Guard recursive lookup
                                    cfg_val = find_config_value(config, c)  # Search config for matching key
                                except Exception:
                                    cfg_val = None  # On error treat as missing
                                if cfg_val is not None:  # If found in config
                                    ordered.append(cfg_val)  # Use configured hyperparameter value
                                else:  # Not found anywhere; warn and use None
                                    print(f"{BackgroundColors.YELLOW}Warning: results CSV column '{c}' not found in runtime metrics or config; writing None{Style.RESET_ALL}")  # Warn about missing column
                                    ordered.append(None)  # Explicit None for missing value
                        writer.writerow(ordered)  # Write the ordered row to CSV
                        try:  # Flush to persist immediately
                            f_handle.flush()  # Flush file buffer to disk
                        except Exception:
                            pass  # Ignore flush errors
                except Exception as _we:  # On any write/open failure, warn and continue
                    print(f"{BackgroundColors.YELLOW}Warning: failed to persist generation row: {_we}{Style.RESET_ALL}")  # Warn but do not abort
        except Exception as e:
            print(f"{BackgroundColors.YELLOW}Warning: could not prepare results CSV entry: {e}{Style.RESET_ALL}")  # Warn on top-level failures
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def to_seconds(obj):
    """
    Converts various time-like objects to seconds.
    
    :param obj: The object to convert (can be int, float, timedelta, datetime, etc.)
    :return: The equivalent time in seconds as a float, or None if conversion fails
    """
    
    try:
        if obj is None:  # None can't be converted
            return None  # Signal failure to convert
        if isinstance(obj, (int, float)):  # Already numeric (seconds or timestamp)
            return float(obj)  # Return as float seconds
        if hasattr(obj, "total_seconds"):  # Timedelta-like objects
            try:  # Attempt to call total_seconds()
                return float(obj.total_seconds())  # Use the total_seconds() method
            except Exception as e:  # total_seconds() failed
                print(str(e))  # Print error to terminal for visibility
                try:  # Attempt to notify about total_seconds failure via Telegram
                    send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full error via Telegram
                except Exception:  # If notification fails, ignore to avoid recursion
                    pass  # Ignore Telegram send errors and fall through
        if hasattr(obj, "timestamp"):  # Datetime-like objects
            try:  # Attempt to call timestamp()
                return float(obj.timestamp())  # Use timestamp() to get seconds since epoch
            except Exception as e:  # timestamp() failed
                print(str(e))  # Print error to terminal for visibility
                try:  # Attempt to notify about timestamp failure via Telegram
                    send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full error via Telegram
                except Exception:  # If notification fails, ignore to avoid recursion
                    pass  # Ignore Telegram send errors and fall through
        return None  # Couldn't convert
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def calculate_execution_time(start_time, finish_time=None):
    """
    Calculates the execution time and returns a human-readable string.

    Accepts either:
    - Two datetimes/timedeltas: `calculate_execution_time(start, finish)`
    - A single timedelta or numeric seconds: `calculate_execution_time(delta)`
    - Two numeric timestamps (seconds): `calculate_execution_time(start_s, finish_s)`

    Returns a string like "1h 2m 3s".
    """

    try:
        if finish_time is None:  # Single-argument mode: start_time already represents duration or seconds
            total_seconds = to_seconds(start_time)  # Try to convert provided value to seconds
            if total_seconds is None:  # Conversion failed
                try:  # Attempt numeric coercion
                    total_seconds = float(start_time)  # Attempt numeric coercion
                except Exception as e:  # Coercion failed
                    print(str(e))  # Print coercion error to terminal for visibility
                    try:  # Attempt to notify about coercion failure via Telegram
                        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full coercion error via Telegram
                    except Exception:  # If notification fails, ignore to avoid cascading errors
                        pass  # Ignore Telegram send errors and fallback
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
                except Exception as e:  # Subtraction failed
                    print(str(e))  # Print subtraction error to terminal for visibility
                    try:  # Attempt to notify about subtraction failure via Telegram
                        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full subtraction error via Telegram
                    except Exception:  # If notification fails, ignore to avoid cascading errors
                        pass  # Ignore Telegram send errors and continue to numeric coercion
                    try:  # Final attempt: Numeric coercion
                        total_seconds = float(finish_time) - float(start_time)  # Final numeric coercion attempt
                    except Exception as e:  # Numeric coercion failed
                        print(str(e))  # Print numeric coercion error to terminal for visibility
                        try:  # Attempt to notify about numeric coercion failure via Telegram
                            send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full coercion error via Telegram
                        except Exception:  # If notification fails, ignore to avoid cascading errors
                            pass  # Ignore Telegram send errors and fallback
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
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def get_hardware_specifications(device_used=None):
    """
    Returns system specs: real CPU model (Windows/Linux/macOS), physical cores,
    RAM in GB, OS name/version, GPU summary, CUDA availability and device used.

    :param device_used: Optional `torch.device` indicating which device the program is using
    :return: Dictionary with keys: cpu_model, cores, ram_gb, os, gpu, cuda, device_used
    """

    try:
        verbose_output(
            f"{BackgroundColors.GREEN}Fetching system specifications...{Style.RESET_ALL}"
        )  # Output the verbose message

        system = platform.system()  # Identify OS type

        try:  # Try to fetch real CPU model using OS-specific methods
            if system == "Windows":  # Windows: use WMIC
                out = subprocess.check_output("wmic cpu get Name", shell=True).decode(errors="ignore")  # Run WMIC
                cpu_model = out.strip().split("\n")[1].strip()  # Extract model line

            elif system == "Linux":  # Linux: read from /proc/cpuinfo
                cpu_model = "Unknown"  # Default
                with open("/proc/cpuinfo") as f:  # Open cpuinfo
                    for line in f:  # Iterate lines
                        if "model name" in line:  # Model name entry
                            cpu_model = line.split(":", 1)[1].strip()  # Extract name
                            break  # Stop after first match

            elif system == "Darwin":  # MacOS: use sysctl
                out = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"])  # Run sysctl
                cpu_model = out.decode().strip()  # Extract model string

            else:  # Unsupported OS
                cpu_model = "Unknown"  # Fallback

        except Exception:  # If any method fails
            cpu_model = "Unknown"  # Fallback on failure

        cores = psutil.cpu_count(logical=False) if psutil else None  # Physical core count
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1) if psutil else None  # Total RAM in GB
        os_name = f"{platform.system()} {platform.release()}"  # OS name + version

        # GPU detection and summary
        try:
            cuda_available = torch.cuda.is_available() if hasattr(torch, "cuda") else False  # Whether CUDA is available
        except Exception:
            cuda_available = False  # Fallback

        gpu_summary = None  # Default GPU summary
        gpu_count = 0  # Default GPU count
        try:  # Try to collect GPU information when CUDA is present
            if cuda_available:  # If CUDA available
                gpu_count = torch.cuda.device_count()  # Number of CUDA devices
                gpu_info_map = {}  # Map of (name, mem_gb) -> count
                for i in range(gpu_count):  # Iterate devices
                    try:  # Query device name and memory
                        name = torch.cuda.get_device_name(i)  # Device name
                        props = torch.cuda.get_device_properties(i)  # Device properties
                        mem_gb = int(round(props.total_memory / (1024**3)))  # Memory in GB
                        key = (name, mem_gb)  # Grouping key
                        gpu_info_map[key] = gpu_info_map.get(key, 0) + 1  # Increment count for this GPU spec
                    except Exception:
                        continue  # Skip device on failure
                parts = []  # Parts for GPU summary
                for (name, mem_gb), cnt in gpu_info_map.items():  # Build human-readable parts
                    if cnt > 1:  # Multiple identical devices
                        parts.append(f"{name} ({mem_gb}GB) x{cnt}")  # Include multiplicity
                    else:
                        parts.append(f"{name} ({mem_gb}GB)")  # Single device
                gpu_summary = ", ".join(parts) if parts else None  # Join parts if present
        except Exception:
            gpu_summary = None  # On any failure, set to None

        device_used_str = "CPU"  # Default device used label
        try:
            if isinstance(device_used, torch.device):  # If a torch.device was provided
                device_used_str = "CUDA" if device_used.type == "cuda" else "CPU"  # Map to human label
            else:  # If not a torch.device, try string representation
                device_used_str = str(device_used)  # Use provided value as string
        except Exception:
            device_used_str = "Unknown"  # Fallback on failure

        return {  # Build final dictionary including GPU and CUDA info
            "cpu_model": cpu_model,  # CPU model string
            "cores": cores,  # Physical cores
            "ram_gb": ram_gb,  # RAM in gigabytes
            "os": os_name,  # Operating system
            "gpu": gpu_summary if gpu_summary else None,  # GPU summary string or None
            "cuda": "Yes" if cuda_available else "No",  # CUDA availability
            "device_used": device_used_str,  # Device used label
            "gpu_count": gpu_count,  # Number of GPUs detected
        }
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def populate_hardware_column(df, column_name="hardware", device_used=None):
    """
    Populate "df[column_name]" with a readable hardware description built from
    "get_hardware_specifications()". On failure the column will be set to None.

    :param df: pandas.DataFrame to modify in-place
    :param column_name: Name of the column to set (default: "hardware")
    :param device_used: Optional `torch.device` indicating which device the program is using
    :return: The modified DataFrame
    """

    try:
        try:  # Try to fetch hardware specifications
            hardware_specs = get_hardware_specifications(device_used=device_used)  # Get system specs with device info
            # Build readable hardware string including GPU, CUDA and device used
            gpu_part = hardware_specs.get('gpu', 'None') if hardware_specs.get('gpu', None) is not None else 'None'  # GPU part
            hardware_str = (  # Build readable hardware string
                f"{hardware_specs.get('cpu_model','Unknown')} | Cores: {hardware_specs.get('cores', 'N/A')}"
                f" | RAM: {hardware_specs.get('ram_gb', 'N/A')} GB | OS: {hardware_specs.get('os','Unknown')}"
                f" | GPU: {gpu_part} | CUDA: {hardware_specs.get('cuda','No')} | Device Used: {hardware_specs.get('device_used','Unknown')}"
            )
            df[column_name] = hardware_str  # Set the hardware column
        except Exception:  # On any failure
            df[column_name] = None  # Set hardware column to None

        return df  # Return the modified DataFrame
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


def play_sound(config: Optional[Dict] = None):
    """
    Plays a sound when the program finishes and skips if the operating system is Windows.

    :param config: Optional configuration dictionary containing sound settings
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG or get_default_config()  # Use global or default config
        
        if not config.get("sound", {}).get("enabled", True):  # If sound disabled
            return  # Exit early
        
        current_os = platform.system()  # Get the current operating system
        if current_os == "Windows":  # If the current operating system is Windows
            return  # Do nothing

        sound_file = config.get("sound", {}).get("file", "./.assets/Sounds/NotificationSound.wav")  # Get sound file path
        sound_commands = config.get("sound", {}).get("commands", {})  # Get sound commands dictionary
        
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
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_wgangp(config: Optional[Union [Dict, str]] = None, **kwargs):
    """
    Programmatic entry point for WGAN-GP execution from external orchestrators.

    This function allows running WGAN-GP training/generation from Python code
    without command-line interface. Supports both config dict and config file path.

    Usage examples:
        # From config dictionary:
        run_wgangp(config={"wgangp": {"csv_path": "data.csv", "mode": "train"}})

        # From config file:
        run_wgangp(config="custom_config.yaml")

        # With direct keyword overrides:
        run_wgangp(csv_path="data.csv", mode="train", epochs=100)

        # Mixed approach:
        run_wgangp(config="config.yaml", epochs=100, batch_size=128)

    :param config: Configuration dictionary or path to YAML config file (optional)
    :param kwargs: Direct keyword argument overrides for configuration
    :return: None
    """

    try:
        global CONFIG  # Declare global CONFIG variable

        if config is None:  # No config provided
            final_config = load_configuration()  # Load from default locations
        elif isinstance(config, str):  # Config is a file path
            final_config = load_configuration(config_path=config)  # Load from specified file
        elif isinstance(config, dict):  # Config is a dictionary
            final_config = load_configuration()  # Load defaults first
            final_config = deep_merge(final_config, config)  # Merge with provided dict
        else:  # Invalid config type
            raise TypeError(f"config must be dict, str, or None, not {type(config)}")

        if kwargs:  # If keyword arguments provided
            cli_style_overrides = {}  # Build config-style dict from kwargs
            for key, value in kwargs.items():  # For each kwarg
                if key in ["csv_path", "mode", "label_col", "feature_cols", "seed", "force_cpu", "from_scratch"]:  # WGAN-GP params
                    cli_style_overrides.setdefault("wgangp", {})[key] = value
                elif key in ["out_dir", "logs_dir"]:  # Path params
                    cli_style_overrides.setdefault("paths", {})[key] = value
                elif key in ["epochs", "batch_size", "critic_steps", "lr", "beta1", "beta2", "lambda_gp", "save_every", "log_interval", "sample_batch", "use_amp", "compile"]:  # Training params
                    cli_style_overrides.setdefault("training", {})[key] = value
                elif key in ["latent_dim", "n_resblocks", "leaky_relu_alpha"]:  # Generator params
                    cli_style_overrides.setdefault("generator", {})[key] = value
                    if key == "leaky_relu_alpha":  # Also set discriminator alpha
                        cli_style_overrides.setdefault("discriminator", {})[key] = value
                elif key in ["g_hidden"]:  # Generator hidden layers
                    cli_style_overrides.setdefault("generator", {})["hidden_dims"] = value
                elif key in ["d_hidden"]:  # Discriminator hidden layers
                    cli_style_overrides.setdefault("discriminator", {})["hidden_dims"] = value
                elif key in ["g_embed_dim"]:  # Generator embedding dim
                    cli_style_overrides.setdefault("generator", {})["embed_dim"] = value
                elif key in ["d_embed_dim"]:  # Discriminator embedding dim
                    cli_style_overrides.setdefault("discriminator", {})["embed_dim"] = value
                elif key in ["checkpoint", "n_samples", "label", "out_file", "gen_batch_size", "feature_dim"]:  # Generation params
                    cli_style_overrides.setdefault("generation", {})[key] = value
                elif key in ["num_workers"]:  # DataLoader params
                    cli_style_overrides.setdefault("dataloader", {})[key] = value
                elif key in ["remove_zero_variance"]:  # Dataset params
                    cli_style_overrides.setdefault("dataset", {})[key] = value
                elif key in ["verbose"]:  # Execution params
                    cli_style_overrides.setdefault("execution", {})[key] = value
                elif key in ["play_sound", "enabled"] and key == "play_sound":  # Sound params
                    cli_style_overrides.setdefault("sound", {})["enabled"] = value
                else:  # Unknown parameter
                    print(f"{BackgroundColors.YELLOW}Warning: Unknown parameter '{key}' will be ignored{Style.RESET_ALL}")
            final_config = deep_merge(final_config, cli_style_overrides)  # Apply kwargs overrides

        CONFIG = final_config  # Update global config

        # Validate results_csv_columns existence and type in config for wgangp module
        results_cols_chk = final_config.get("wgangp", {}).get("results_csv_columns")  # Read configured results columns list
        if not isinstance(results_cols_chk, list) or len(results_cols_chk) == 0:  # Ensure the value exists, is a list, and is non-empty
            print(f"{BackgroundColors.RED}Configuration error: 'results_csv_columns' missing, empty, or not a list under 'wgangp' section in configuration.{Style.RESET_ALL}")  # Print clear error message
            raise ValueError("'results_csv_columns' missing, empty, or not a list under 'wgangp' section in configuration")  # Stop execution safely with clear error
        initialize_logger(final_config)  # Initialize logger with final configuration

        setup_telegram_bot(final_config)  # Setup Telegram bot with final configuration

        class ConfigNamespace:
            """
            Namespace object that wraps configuration dictionary.
            """
            
            def __init__(self, config_dict):
                """
                Initialize the ConfigNamespace with a configuration dictionary.
                
                :param self: The instance of the ConfigNamespace.
                :param config_dict: The configuration dictionary to wrap.
                """
                
                self.config = config_dict  # Store the original config dictionary
                self.mode = config_dict.get("wgangp", {}).get("mode", "both")
                self.csv_path = config_dict.get("wgangp", {}).get("csv_path")
                self.label_col = config_dict.get("wgangp", {}).get("label_col", "Label")
                self.feature_cols = config_dict.get("wgangp", {}).get("feature_cols")
                self.seed = int(config_dict.get("wgangp", {}).get("seed", 42))  # Cast to int
                self.force_cpu = config_dict.get("wgangp", {}).get("force_cpu", False)
                self.from_scratch = config_dict.get("wgangp", {}).get("from_scratch", False)
                self.out_dir = config_dict.get("paths", {}).get("out_dir", "outputs")
                self.epochs = int(config_dict.get("training", {}).get("epochs", 60))  # Cast to int
                self.batch_size = int(config_dict.get("training", {}).get("batch_size", 64))  # Cast to int
                self.critic_steps = int(config_dict.get("training", {}).get("critic_steps", 5))  # Cast to int
                self.lr = float(config_dict.get("training", {}).get("lr", 1e-4))  # Cast to float
                self.beta1 = float(config_dict.get("training", {}).get("beta1", 0.5))  # Cast to float
                self.beta2 = float(config_dict.get("training", {}).get("beta2", 0.9))  # Cast to float
                self.lambda_gp = float(config_dict.get("training", {}).get("lambda_gp", 10.0))  # Cast to float
                self.save_every = int(config_dict.get("training", {}).get("save_every", 5))  # Cast to int
                self.log_interval = int(config_dict.get("training", {}).get("log_interval", 50))  # Cast to int
                self.sample_batch = int(config_dict.get("training", {}).get("sample_batch", 16))  # Cast to int
                self.use_amp = config_dict.get("training", {}).get("use_amp", False)
                self.compile = config_dict.get("training", {}).get("compile", False)
                self.latent_dim = int(config_dict.get("generator", {}).get("latent_dim", 100))  # Cast to int
                self.g_hidden = config_dict.get("generator", {}).get("hidden_dims", [256, 512])
                self.embed_dim = int(config_dict.get("generator", {}).get("embed_dim", 32))  # Cast to int
                self.n_resblocks = int(config_dict.get("generator", {}).get("n_resblocks", 3))  # Cast to int
                self.d_hidden = config_dict.get("discriminator", {}).get("hidden_dims", [512, 256, 128])
                self.checkpoint = config_dict.get("generation", {}).get("checkpoint")
                self.n_samples = float(config_dict.get("generation", {}).get("n_samples", 1.0))  # Cast to float
                self.label = config_dict.get("generation", {}).get("label")
                if self.label is not None:  # If label is not None
                    self.label = int(self.label)  # Cast to int
                self.out_file = config_dict.get("generation", {}).get("out_file", "generated.csv")
                self.gen_batch_size = int(config_dict.get("generation", {}).get("gen_batch_size", 256))  # Cast to int
                self.feature_dim = config_dict.get("generation", {}).get("feature_dim")
                if self.feature_dim is not None:  # If feature_dim is not None
                    self.feature_dim = int(self.feature_dim)  # Cast to int
                self.num_workers = int(config_dict.get("dataloader", {}).get("num_workers", 8))  # Cast to int
                self._last_training_time = 0.0  # Placeholder for last training elapsed time (set after train)

        args = ConfigNamespace(final_config)  # Create namespace from config

        start_time = datetime.datetime.now()  # Record start time
        send_telegram_message(TELEGRAM_BOT, f"Starting WGAN-GP (programmatic) at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:  # Execute with error handling
            if args.mode == "train":  # Training mode
                training_start_time = time.time()  # Record training start time using time.time()
                train(args, final_config)  # Train model
                args._last_training_time = time.time() - training_start_time  # Store training elapsed time on args
            elif args.mode == "gen":  # Generation mode
                if args.checkpoint is None:  # Verify checkpoint provided
                    raise ValueError("Generation mode requires checkpoint path")
                generate(args, final_config)  # Generate samples
            elif args.mode == "both":  # Combined mode
                training_start_time = time.time()  # Record training start time using time.time()
                train(args, final_config)  # Train first
                args._last_training_time = time.time() - training_start_time  # Store training elapsed time on args
                if args.csv_path:  # If CSV provided
                    csv_path_obj = Path(args.csv_path)
                    checkpoint_dir = csv_path_obj.parent / final_config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation") / final_config.get("paths", {}).get("checkpoint_subdir", "Checkpoints")
                    checkpoint_path = checkpoint_dir / f"{csv_path_obj.stem}_generator_epoch{args.epochs}.pt"
                    if not checkpoint_path.exists():  # Find latest if specific epoch not found
                        checkpoints = sorted(checkpoint_dir.glob(f"{csv_path_obj.stem}_generator_epoch*.pt"))
                        if checkpoints:
                            checkpoint_path = checkpoints[-1]
                    args.checkpoint = str(checkpoint_path)
                generate(args, final_config)  # Generate samples
            else:  # Invalid mode
                raise ValueError(f"Invalid mode: {args.mode}")
        finally:  # Always show execution time
            finish_time = datetime.datetime.now()
            execution_time = calculate_execution_time(start_time, finish_time)
            print(f"{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{execution_time}{Style.RESET_ALL}")
            send_telegram_message(TELEGRAM_BOT, f"WGAN-GP execution finished. Time: {execution_time}")
            if final_config.get("sound", {}).get("enabled", True):  # If sound enabled
                play_sound(final_config)  # Play completion sound
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def close_all_results_csv_handles():
    """
    Close all opened results CSV file handles at process exit.
    
    This function is registered with atexit to ensure that all file handles in RESULTS_CSV_HANDLES are properly closed when the program terminates, preventing resource leaks.
    
    :param None
    :return: None
    """
    
    for key, (f, _) in list(RESULTS_CSV_HANDLES.items()):  # Iterate over cached handles
        try:
            if f and not f.closed:  # If file object exists and is open
                f.close()  # Close the file
        except Exception:
            pass  # Ignore close errors


def main():
    """
    Main CLI entry point.

    Handles command-line argument parsing, configuration loading, and execution routing.

    :param: None
    :return: None
    """

    try:
        global CONFIG  # Declare global CONFIG variable

        print(
            f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}WGAN-GP Data Augmentation{BackgroundColors.GREEN} program!{Style.RESET_ALL}"
        )  # Output the welcome message

        args = parse_args()  # Get CLI arguments
        cli_overrides = args_to_config_overrides(args)  # Convert to config overrides

        config = load_configuration(config_path=args.config, cli_overrides=cli_overrides)  # Load merged config
        CONFIG = config  # Store in global

        initialize_logger(config)  # Initialize logging system with configuration

        setup_telegram_bot(config)  # Setup Telegram bot with configuration

        start_time = datetime.datetime.now()  # Get the start time of the program
        send_telegram_message(TELEGRAM_BOT, [f"Starting WGAN-GP Data Augmentation at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"])  # Send Telegram notification

        mode = config.get("wgangp", {}).get("mode", "both")  # Get mode
        csv_path = config.get("wgangp", {}).get("csv_path")  # Get CSV path
        results_suffix = config.get("execution", {}).get("results_suffix", "_data_augmented")  # Get results suffix
        datasets = config.get("dataset", {}).get("datasets", {})  # Get datasets dictionary

        class ConfigNamespace:
            """
            Namespace wrapper for config dict.
            """
            
            def __init__(self, cfg):
                """
                Initialize the ConfigNamespace with a configuration dictionary.
                
                :param self: The instance of the ConfigNamespace.
                :param cfg: The configuration dictionary to wrap.
                :return: None
                """
                
                self.mode = cfg.get("wgangp", {}).get("mode", "both")
                self.csv_path = cfg.get("wgangp", {}).get("csv_path")
                self.label_col = cfg.get("wgangp", {}).get("label_col", "Label")
                self.feature_cols = cfg.get("wgangp", {}).get("feature_cols")
                self.seed = int(cfg.get("wgangp", {}).get("seed", 42))  # Cast to int
                self.force_cpu = cfg.get("wgangp", {}).get("force_cpu", False)
                self.from_scratch = cfg.get("wgangp", {}).get("from_scratch", False)
                self.out_dir = cfg.get("paths", {}).get("out_dir", "outputs")
                self.epochs = int(cfg.get("training", {}).get("epochs", 60))  # Cast to int
                self.batch_size = int(cfg.get("training", {}).get("batch_size", 64))  # Cast to int
                self.critic_steps = int(cfg.get("training", {}).get("critic_steps", 5))  # Cast to int
                self.lr = float(cfg.get("training", {}).get("lr", 1e-4))  # Cast to float
                self.beta1 = float(cfg.get("training", {}).get("beta1", 0.5))  # Cast to float
                self.beta2 = float(cfg.get("training", {}).get("beta2", 0.9))  # Cast to float
                self.lambda_gp = float(cfg.get("training", {}).get("lambda_gp", 10.0))  # Cast to float
                self.save_every = int(cfg.get("training", {}).get("save_every", 5))  # Cast to int
                self.log_interval = int(cfg.get("training", {}).get("log_interval", 50))  # Cast to int
                self.sample_batch = int(cfg.get("training", {}).get("sample_batch", 16))  # Cast to int
                self.use_amp = cfg.get("training", {}).get("use_amp", False)
                self.compile = cfg.get("training", {}).get("compile", False)
                self.latent_dim = int(cfg.get("generator", {}).get("latent_dim", 100))  # Cast to int
                self.g_hidden = cfg.get("generator", {}).get("hidden_dims", [256, 512])
                self.embed_dim = int(cfg.get("generator", {}).get("embed_dim", 32))  # Cast to int
                self.n_resblocks = int(cfg.get("generator", {}).get("n_resblocks", 3))  # Cast to int
                self.d_hidden = cfg.get("discriminator", {}).get("hidden_dims", [512, 256, 128])
                self.checkpoint = cfg.get("generation", {}).get("checkpoint")
                self.n_samples = float(cfg.get("generation", {}).get("n_samples", 1.0))  # Cast to float
                self.label = cfg.get("generation", {}).get("label")
                if self.label is not None:  # If label is not None
                    self.label = int(self.label)  # Cast to int
                self.out_file = cfg.get("generation", {}).get("out_file", "generated.csv")
                self.gen_batch_size = int(cfg.get("generation", {}).get("gen_batch_size", 256))  # Cast to int
                self.feature_dim = cfg.get("generation", {}).get("feature_dim")
                if self.feature_dim is not None:  # If feature_dim is not None
                    self.feature_dim = int(self.feature_dim)  # Cast to int
                self.force_new_samples = cfg.get("generation", {}).get("force_new_samples", False)
                self.num_workers = int(cfg.get("dataloader", {}).get("num_workers", 8))  # Cast to int
                self._last_training_time = 0.0  # Placeholder for last training elapsed time (set after train)

        args = ConfigNamespace(config)  # Create args namespace
        # Validate results_csv_columns existence and type for CLI runs
        results_cols = config.get("wgangp", {}).get("results_csv_columns")  # Read configured results columns list
        if not isinstance(results_cols, list) or len(results_cols) == 0:  # Ensure the value exists, is a list, and is non-empty
            print(f"{BackgroundColors.RED}Configuration error: 'results_csv_columns' missing, empty, or not a list under 'wgangp' section in configuration.{Style.RESET_ALL}")  # Print clear error message
            raise ValueError("'results_csv_columns' missing, empty, or not a list under 'wgangp' section in configuration")  # Stop execution safely with clear error
        
        if csv_path is not None:  # Single file mode (csv_path provided):
            csv_path_obj = Path(csv_path)  # Create Path object from csv_path (always available in this branch)
            # Ensure results CSV header columns configuration is available
            if args.out_file == "generated.csv" and mode in ["gen", "both"]:  # If using default output file
                data_aug_subdir = config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation")  # Get subdir name
                data_aug_dir = csv_path_obj.parent / data_aug_subdir  # Create Data_Augmentation subdirectory path
                os.makedirs(data_aug_dir, exist_ok=True)  # Ensure Data_Augmentation directory exists
                output_filename = f"{csv_path_obj.stem}{results_suffix}{csv_path_obj.suffix}"  # Use input name with suffix
                args.out_file = str(data_aug_dir / output_filename)  # Set output file path to Data_Augmentation subdirectory
            # Ensure per-dataset results CSV exists at dataset directory root
            data_aug_subdir = config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation")  # Get subdir name from config
            data_aug_dir = csv_path_obj.parent / data_aug_subdir  # Build Data_Augmentation directory path for this dataset
            os.makedirs(data_aug_dir, exist_ok=True)  # Ensure Data_Augmentation directory exists before creating results CSV
            results_csv_path = data_aug_dir / "data_augmentation_results.csv"  # Path for per-directory results CSV inside Data_Augmentation
            if not results_csv_path.exists():  # Only write header if file does not exist
                with open(results_csv_path, "w", newline="", encoding="utf-8") as _f:  # Open file for writing header
                    writer = csv.writer(_f)  # Create CSV writer
                    writer.writerow(results_cols)  # Write header exactly in configured order
            
            if mode == "train":  # Training mode
                train(args, config)  # Train the model
            elif mode == "gen":  # Generation mode
                assert args.checkpoint is not None, "Generation requires --checkpoint"  # Ensure checkpoint is provided
                if verify_data_augmentation_file(args, config):  # Verify if generation is necessary according to existing output and configuration
                    generate(args, config)  # Generate synthetic samples
                else:
                    print(f"{BackgroundColors.GREEN}Skipping generation: output file already satisfies configured n_samples.{Style.RESET_ALL}")
            elif mode == "both":  # Combined mode
                print(f"{BackgroundColors.GREEN}[1/2] Training model...{Style.RESET_ALL}")
                training_start_time = time.time()  # Record training start time using time.time()
                train(args, config)  # Train the model
                args._last_training_time = time.time() - training_start_time  # Store training elapsed time on args
                
                csv_path_obj = Path(csv_path)  # Create Path object from csv_path
                checkpoint_prefix = csv_path_obj.stem  # Use CSV filename as prefix
                checkpoint_subdir = config.get("paths", {}).get("checkpoint_subdir", "Checkpoints")  # Get checkpoint subdir
                data_aug_subdir = config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation")  # Get data aug subdir
                checkpoint_dir = csv_path_obj.parent / data_aug_subdir / checkpoint_subdir  # Checkpoints in Data_Augmentation/Checkpoints
                checkpoint_path = checkpoint_dir / f"{checkpoint_prefix}_generator_epoch{args.epochs}.pt"  # Expected checkpoint path
                if not checkpoint_path.exists():  # If specific epoch checkpoint not found, find the latest one
                    checkpoints = sorted(checkpoint_dir.glob(f"{checkpoint_prefix}_generator_epoch*.pt"))  # Find all checkpoints for this dataset
                    if checkpoints:  # If checkpoints found
                        checkpoint_path = checkpoints[-1]  # Use the latest checkpoint
                    else:  # No checkpoints found
                        raise FileNotFoundError(f"No generator checkpoint found for {csv_path_obj.name} in {checkpoint_dir}")
                
                args.checkpoint = str(checkpoint_path)  # Set checkpoint path for generation
                print(f"\n{BackgroundColors.CYAN}[2/2] Generating samples from {checkpoint_path.name}...{Style.RESET_ALL}")
                print(f"{BackgroundColors.GREEN}Output will be saved to: {BackgroundColors.CYAN}{args.out_file}{Style.RESET_ALL}")
                
                if verify_data_augmentation_file(args, config):  # Verify if generation is necessary according to existing output and configuration
                    generate(args, config)  # Generate synthetic samples
                else:
                    print(f"{BackgroundColors.GREEN}Skipping generation: output file already satisfies configured n_samples.{Style.RESET_ALL}")
        
        else:
            print(
                f"{BackgroundColors.GREEN}No CSV path provided. Processing datasets in batch mode...{Style.RESET_ALL}"
            )  # Notify batch mode
            
            for dataset_name, paths in datasets.items():  # For each dataset in the datasets dictionary
                print(
                    f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing dataset: {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}"
                )
                for input_path in paths:  # For each path in the dataset's paths list
                    if not verify_filepath_exists(input_path):  # If the input path does not exist
                        verbose_output(
                            f"{BackgroundColors.YELLOW}Skipping missing path: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}",
                            config=config
                        )
                        continue  # Skip to the next path if the current one doesn't exist

                    data_aug_subdir = config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation")  # Get data augmentation subdir name from config
                    data_aug_dir = Path(input_path) / data_aug_subdir  # Construct Data_Augmentation directory path for this input_path
                    per_dir_results_csv = data_aug_dir / "data_augmentation_results.csv"  # Results CSV path for this dataset inside Data_Augmentation
                    if not per_dir_results_csv.exists():  # Only write header if file does not exist
                        os.makedirs(data_aug_dir, exist_ok=True)  # Ensure Data_Augmentation directory exists before header write
                        with open(per_dir_results_csv, "w", newline="", encoding="utf-8") as _f:  # Open file for header writing
                            writer = csv.writer(_f)  # Create CSV writer
                            writer.writerow(results_cols)  # Write header exactly in configured order

                    files_to_process = get_files_to_process(
                        input_path, file_extension=".csv", config=config
                    )  # Get list of CSV files to process
                    total_files = len(files_to_process)  # Compute total files once per input_path
                    for index, file in enumerate(files_to_process, start=1):  # Iterate with index and file
                        try:  # Wrap per-file processing so exceptions are caught by the outer handler
                            file_progress_prefix = f"{BackgroundColors.CYAN}[{index}/{total_files}]{Style.RESET_ALL}"  # Build colored per-file prefix once per file
                            args.file_progress_prefix = file_progress_prefix  # Attach colored prefix to args for train/gen functions
                            try:  # Guard path resolution and membership check
                                resolved_path = str(Path(file).resolve())  # Resolve to absolute path string
                            except Exception:
                                resolved_path = str(Path(file))  # Fallback to given path string on failure
                            if resolved_path in PROCESSED_FILES:  # If file already processed
                                print(f"{BackgroundColors.YELLOW}{file_progress_prefix} Skipping already-processed file: {resolved_path}{Style.RESET_ALL}")  # Warn and skip duplicate with prefix
                                continue  # Skip to next file
                            print(
                                f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*80}{Style.RESET_ALL}"
                            )  # Decorative separator
                            print(
                                f"{file_progress_prefix} {BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing file: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}"
                            )  # High-level processing message with prefix
                            print(
                                f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*80}{Style.RESET_ALL}\n"
                            )  # Decorative separator
                            
                            csv_path_obj = Path(file)  # Create Path object from file path
                            data_aug_subdir = config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation")  # Get subdir name
                            data_aug_dir = csv_path_obj.parent / data_aug_subdir  # Create Data_Augmentation subdirectory path
                            os.makedirs(data_aug_dir, exist_ok=True)  # Ensure Data_Augmentation directory exists
                            output_filename = f"{csv_path_obj.stem}{results_suffix}{csv_path_obj.suffix}"  # Use input name with RESULTS_SUFFIX
                            args.out_file = str(data_aug_dir / output_filename)  # Set output file path to Data_Augmentation subdirectory
                            args.csv_path = file  # Set CSV path to current file
                            
                            try:  # Try to execute the specified mode for the current file
                                if mode == "train":  # Training mode
                                    training_start_time = time.time()  # Record training start time using time.time()
                                    train(args, config)  # Train the model only
                                    args._last_training_time = time.time() - training_start_time  # Store training elapsed time on args
                                elif mode == "gen":  # Generation mode
                                    assert args.checkpoint is not None, "Generation requires --checkpoint"
                                    if verify_data_augmentation_file(args, config):  # Verify whether generation should proceed for this file according to existing output and configuration
                                        generate(args, config)  # Generate synthetic samples only
                                    else:
                                        print(f"{BackgroundColors.GREEN}Skipping generation: output file already satisfies configured n_samples.{Style.RESET_ALL}")
                                elif mode == "both":  # Combined mode
                                    print(f"{BackgroundColors.GREEN}[1/2] Training model on {BackgroundColors.CYAN}{csv_path_obj.name}{BackgroundColors.GREEN}...{Style.RESET_ALL}")
                                    training_start_time = time.time()  # Record training start time using time.time()
                                    train(args, config)  # Train the model
                                    args._last_training_time = time.time() - training_start_time  # Store training elapsed time on args
                                    
                                    checkpoint_prefix = csv_path_obj.stem  # Use CSV filename as prefix
                                    checkpoint_subdir = config.get("paths", {}).get("checkpoint_subdir", "Checkpoints")  # Get checkpoint subdir
                                    checkpoint_dir = data_aug_dir / checkpoint_subdir  # Checkpoints in Data_Augmentation/Checkpoints
                                    checkpoint_path = checkpoint_dir / f"{checkpoint_prefix}_generator_epoch{args.epochs}.pt"
                                    if not checkpoint_path.exists():  # If specific epoch checkpoint not found, find the latest one
                                        checkpoints = sorted(checkpoint_dir.glob(f"{checkpoint_prefix}_generator_epoch*.pt"))  # Find all checkpoints for this dataset
                                        if checkpoints:  # If checkpoints found
                                            checkpoint_path = checkpoints[-1]  # Use the latest checkpoint
                                        else:  # No checkpoints found
                                            raise FileNotFoundError(f"No generator checkpoint found for {csv_path_obj.name} in {checkpoint_dir}")
                                    
                                    args.checkpoint = str(checkpoint_path)  # Set checkpoint path for generation
                                    print(f"\n{BackgroundColors.CYAN}[2/2] Generating samples from {checkpoint_path.name}...{Style.RESET_ALL}")
                                    print(f"{BackgroundColors.GREEN}Output will be saved to: {BackgroundColors.CYAN}{args.out_file}{Style.RESET_ALL}")
                                    
                                    if verify_data_augmentation_file(args, config):  # Verify whether generation should proceed for this file according to existing output and configuration
                                        generate(args, config)  # Generate synthetic samples
                                    else:
                                        print(f"{BackgroundColors.GREEN}Skipping generation: output file already satisfies configured n_samples.{Style.RESET_ALL}")
                            finally:  # Always mark file as processed even if generation/training raised (prevents re-entry)
                                try:  # Guard adding to registry
                                    PROCESSED_FILES.add(resolved_path)  # Remember that this file was processed in this run
                                except Exception:  # Ignore failures when updating registry
                                    pass  # Continue despite registry update failure
                                try:  # Attempt to flush the logger to persist recent outputs
                                    if logger is not None:  # Only flush if logger initialized
                                        logger.flush()  # Flush log file buffer to disk
                                except Exception:  # Ignore any logger flush errors to avoid masking primary failures
                                    pass  # No-op on flush failure
                        except Exception as e:  #   
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

        if config.get("sound", {}).get("enabled", True):  # If sound enabled
            atexit.register(lambda: play_sound(config))  # Register the play_sound function to be called when the program finishes
        atexit.register(close_all_results_csv_handles)  # Ensure handles are closed at program exit
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


if __name__ == "__main__":  # Standard Python entrypoint check
    """  # Module executed as script: call main() with robust error handling
    This is the standard boilerplate that calls the main() function.

    :return: None
    """

    try:  # Protect main execution to ensure errors are reported and notified
        main()  # Call the main function
    except KeyboardInterrupt:  # User-initiated interrupt
        try:  # Attempt friendly shutdown notification on interrupt
            print("Execution interrupted by user (KeyboardInterrupt)")  # Inform terminal about user interrupt
            send_telegram_message(TELEGRAM_BOT, ["WGAN-GP execution interrupted by user (KeyboardInterrupt)"])  # Notify via Telegram
        except Exception:  # Ignore notification failures during interrupt handling
            pass  # Continue to cleanup even if notification fails
        try:  # Attempt to flush and close logger for clean logs on interrupt
            if logger is not None:  # Only interact with logger if it exists
                logger.flush()  # Flush pending log writes to disk
                logger.close()  # Close log file handle cleanly
        except Exception:  # Ignore logger errors during cleanup
            pass  # Proceed with raising the interrupt
        raise  # Re-raise KeyboardInterrupt to allow upstream handling
    except BaseException as e:  # Catch everything (including SystemExit) and report
        try:  # Try to log and notify about the fatal error
            print(f"Fatal error: {e}")  # Print the exception message to terminal for logs
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback and message via Telegram
        except Exception:  # If notification fails, attempt to print traceback to stderr
            try:  # Try printing a full traceback as fallback
                traceback.print_exc()  # Print traceback to stderr
            except Exception:  # If even traceback printing fails, ignore silently
                pass  # No further fallback available
        try:  # Attempt to flush and close logger to preserve logs on fatal errors
            if logger is not None:  # Only flush/close if logger initialized
                logger.flush()  # Flush pending log writes
                logger.close()  # Close the log file handle
        except Exception:  # Ignore any logger cleanup failures
            pass  # Continue to re-raise after best-effort cleanup
        raise  # Re-raise exception to preserve exit semantics
