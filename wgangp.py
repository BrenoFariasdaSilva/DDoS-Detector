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
import threading  # For background resource watcher thread
import shutil  # For disk space queries when psutil is unavailable
import signal  # For sending signals to this process when critical
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

RESOURCE_MONITOR_STOP_EVENT = None  # Event to signal watcher to stop
RESOURCE_MONITOR_THREAD = None  # Background thread running the watcher


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
        self.original_num_samples = len(df)  # Store number of samples after preprocessing (NaN/inf removed)

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

        self.X = torch.from_numpy(np.ascontiguousarray(self.X)).float()  # Pre-convert features to float tensor to avoid per-batch numpy-to-tensor conversion overhead
        self._labels_tensor = torch.from_numpy(np.asarray(self.labels, dtype=np.int64))  # Pre-convert encoded labels to long tensor for efficient DataLoader collation

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
        :return: Tuple of (features_tensor, label_tensor) as pre-converted tensors for efficient DataLoader collation
        """

        return self.X[idx], self._labels_tensor[idx]  # Return pre-converted tensor views directly without numpy-to-tensor overhead


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
        
        self.config = cfg  # Store the original config dictionary
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
        self.file_progress_prefix = ""  # Default per-file progress prefix (set at runtime when batch processing)


# Functions Definitions:


setup_global_exception_hook()  # Set global exception hook to shared Telegram handler


def safe_log(level: str, message: str):
    """
    Safe logging helper that works with the minimal `Logger` (write/flush)
    or with richer logger objects exposing level methods.

    :param level: Logging level name (e.g. "debug", "warning", "critical")
    :param message: Message to log
    :return: None
    """

    try:
        if logger and hasattr(logger, level): getattr(logger, level)(message)  # Use level method if present
        elif logger and hasattr(logger, "write"): logger.write(f"[{level.upper()}] {message}")  # Fallback to write()
        else: 
            print(f"{BackgroundColors.YELLOW}Logger not available; fallback to print: [{level.upper()}] {message}{Style.RESET_ALL}")  # Warn about logger unavailability
    except Exception:
        try:
            print(str(message))  # Best-effort fallback to print
        except Exception:
            pass  # Suppress any further errors


def safe_debug(message: str):
    """
    Debug-level safe logger wrapper.

    :param message: Message to log at debug level
    :return: None
    """

    try:
        safe_log("debug", message)  # Delegate to safe_log
    except Exception:
        pass  # Ignore logging failures


def safe_warning(message: str):
    """
    Warning-level safe logger wrapper.

    :param message: Message to log at warning level
    :return: None
    """

    try:
        safe_log("warning", message)  # Delegate to safe_log
    except Exception:
        pass  # Ignore logging failures


def safe_critical(message: str):
    """
    Critical-level safe logger wrapper.

    :param message: Message to log at critical level
    :return: None
    """

    try:
        safe_log("critical", message)  # Delegate to safe_log
    except Exception:
        pass  # Ignore logging failures

def resource_monitor_loop(stop_event: threading.Event, config: Optional[Dict] = None):
    """
    Background loop that periodically inspects system resource usage (RAM/CPU)
    and logs / reports when thresholds are approached or exceeded.

    :param stop_event: Event used to signal the loop to stop
    :param config: Optional configuration dictionary containing watchdog settings
    :return: None
    """

    try:
        if psutil is None: return  # If psutil missing, nothing to monitor

        cfg = config or CONFIG or get_default_config()  # Prefer explicit config, fallback to global/default
        wd = cfg.get("watchdog", {}) if isinstance(cfg, dict) else {}  # Get watchdog config dict
        max_ram = wd.get("max_ram_percent", 90)  # Percent RAM usage considered critical
        max_cpu = wd.get("max_cpu_percent", 95)  # Percent CPU usage considered critical
        check_interval = wd.get("check_interval_s", 5)  # Seconds between checks
        sustained_checks = max(1, int((wd.get("critical_duration_s", 30) // max(1, check_interval))))  # Number of checks to consider sustained

        consecutive = 0  # Counter for consecutive threshold breaches

        while not stop_event.is_set():
            mem = psutil.virtual_memory().percent  # Current RAM usage percent
            cpu = psutil.cpu_percent(interval=None)  # Current CPU usage percent (non-blocking)

            if logger:
                safe_debug(f"Resources Monitor: Memory={mem:.1f}%, CPU={cpu:.1f}%")  # Debug log

            if mem >= max_ram or cpu >= max_cpu:
                consecutive += 1  # Count this breach
                if logger:
                    safe_warning(f"Resources Monitor: Threshold Breach (Memory={mem:.1f}%, CPU={cpu:.1f}%)")  # Warn
                try:
                    send_telegram_message(TELEGRAM_BOT, f"Resource Watcher: High Resource Usage Detected: Memory={mem:.1f}%, CPU={cpu:.1f}%")  # Notify
                except Exception:
                    pass  # Ignore Telegram failures to avoid cascading errors
            else:
                consecutive = 0  # Reset counter when metrics return to normal

            if consecutive >= sustained_checks:
                if logger:
                    safe_critical(f"Resources Monitor: Sustained High Resource Usage For ~{sustained_checks * check_interval}s; Requesting Termination")  # Critical log
                try:
                    send_telegram_message(TELEGRAM_BOT, f"Resource Watcher: Sustained High Resource Usage; Requesting Process Termination to Avoid Abrupt OOM Kill")  # Notify
                except Exception:
                    pass  # Ignore failures in notification

                try:
                    os.kill(os.getpid(), signal.SIGTERM)  # Send SIGTERM to self for graceful shutdown
                except Exception:
                    try:
                        os._exit(1)  # Force exit if signal fails
                    except Exception:
                        pass  # Best effort: if even exit fails, continue

            stop_event.wait(timeout=check_interval)  # Wait with interruptability

    except Exception as e:
        try:
            print(str(e))  # Print to stdout as fallback
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Notify about monitor failure
        except Exception:
            pass  # Ignore notification failures from within monitor


def start_resource_monitor(config: Optional[Dict] = None):
    """
    Start the background resource monitor thread and return (stop_event, thread).

    :param config: Optional configuration dict to pass through to the loop
    :return: (stop_event, thread)
    """

    stop_event = threading.Event()  # Event used to stop the thread
    thread = threading.Thread(target=resource_monitor_loop, args=(stop_event, config), daemon=True)  # Daemon thread so it won't block process exit
    thread.start()  # Start background monitoring thread
    return stop_event, thread  # Return handles to caller


def stop_resource_monitor():
    """
    Signal the resource monitor to stop and join the thread if running.

    :return: None
    """

    global RESOURCE_MONITOR_STOP_EVENT, RESOURCE_MONITOR_THREAD  # Access global handles
    try:
        if RESOURCE_MONITOR_STOP_EVENT is not None: RESOURCE_MONITOR_STOP_EVENT.set()  # Signal the monitor to stop
        if RESOURCE_MONITOR_THREAD is not None and RESOURCE_MONITOR_THREAD.is_alive(): RESOURCE_MONITOR_THREAD.join(timeout=5)  # Join briefly to allow cleanup
    except Exception:
        pass  # Ignore errors during shutdown


def reconstruct_missing_preprocessing_objects(args, scaler: Optional[Any], label_encoder: Optional[Any], feature_cols: Optional[List[str]], class_distribution: Optional[Dict]) -> tuple:
    """
    Reconstruct missing preprocessing objects from CSV when checkpoint is incomplete.

    :param args: Parsed arguments namespace with csv_path and label_col.
    :param scaler: Scaler from checkpoint or None if missing.
    :param label_encoder: Label encoder from checkpoint or None if missing.
    :param feature_cols: Feature column names from checkpoint or None if missing.
    :param class_distribution: Class distribution from checkpoint or None if missing.
    :return: Tuple of (scaler, label_encoder, feature_cols, class_distribution).
    """

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
        if args.n_samples <= 1.0:  # If percentage mode or 100% requested, calculate class distribution
            unique_labels, label_counts = np.unique(tmp_ds.labels, return_counts=True)  # Get class distribution
            class_distribution = dict(zip(unique_labels.tolist(), label_counts.tolist()))  # Create label:count mapping
    return (scaler, label_encoder, feature_cols, class_distribution)  # Return reconstructed or original preprocessing objects


def determine_feature_dim_and_n_classes(args, scaler: Optional[Any], label_encoder: Any) -> tuple:
    """
    Determine feature dimensionality and number of classes from args or scaler.

    :param args: Parsed arguments namespace with optional feature_dim.
    :param scaler: Fitted StandardScaler with mean_ attribute.
    :param label_encoder: Fitted LabelEncoder with classes_ attribute.
    :return: Tuple of (feature_dim, n_classes).
    """

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
    return (feature_dim, n_classes)  # Return feature dimensionality and class count


def build_and_load_generator(args, config: Dict, ckpt: Dict, device: torch.device, feature_dim: int, n_classes: int) -> nn.Module:
    """
    Build the generator model and load weights from checkpoint.

    :param args: Parsed arguments namespace with architecture hyperparameters.
    :param config: Configuration dictionary with generator settings.
    :param ckpt: Loaded checkpoint dictionary containing state_dict.
    :param device: Torch device for model placement.
    :param feature_dim: Dimensionality of the output features.
    :param n_classes: Number of label classes for conditional generation.
    :return: Generator model in evaluation mode.
    """

    g_leaky_relu_alpha = safe_float(config.get("generator", {}).get("leaky_relu_alpha", 0.2), 0.2)  # Get generator LeakyReLU alpha safely from config
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
    if isinstance(G, torch.nn.DataParallel):  # If generator wrapped by DataParallel
        G.module.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)  # Load into underlying module
    else:  # Not DataParallel
        G.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)  # Load generator weights from checkpoint
    G.eval()  # Set generator to evaluation mode
    return G  # Return generator model ready for inference


def compute_generation_counts_and_labels(args, config: Dict, class_distribution: Optional[Dict], label_encoder: Any, n_classes: int) -> tuple:
    """
    Compute per-class counts and build the label array for generation.

    :param args: Parsed arguments namespace with n_samples and label.
    :param config: Configuration dictionary with generation thresholds.
    :param class_distribution: Dictionary mapping label indices to counts.
    :param label_encoder: Fitted LabelEncoder for inverse transforms.
    :param n_classes: Number of label classes for uniform sampling.
    :return: Tuple of (n_per_class, labels, n).
    """

    small_class_threshold = int(config.get("generation", {}).get("small_class_threshold", 100))  # Get small class threshold and cast to int
    small_class_min_samples = int(config.get("generation", {}).get("small_class_min_samples", 10))  # Get min samples for small classes and cast to int
    n_per_class = None  # Initialize per-class count dictionary
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
    return (n_per_class, labels, n)  # Return per-class counts, label array, and total count


def notify_start_of_generation(args, n: int) -> None:
    """
    Send Telegram notification about the start of sample generation.

    :param args: Parsed arguments namespace with out_file.
    :param n: Total number of samples to generate.
    :return: None.
    """

    try:
        start_msg = compose_generation_start_message(n, args, Path(args.out_file).name, original_num=None)
        send_telegram_message(TELEGRAM_BOT, start_msg)  # Notify start of generation
    except Exception as e:  # Failed to notify start of generation
        print(str(e))  # Print send error to terminal for visibility
        try:  # Attempt to send full error via Telegram using exception sender
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full send error via Telegram
        except Exception:  # If notification fails, continue without raising to allow generation
            pass  # Ignore Telegram send errors and continue generation


def generate_batches_and_collect_results(args, G: nn.Module, device: torch.device, labels: np.ndarray, n: int) -> tuple:
    """
    Generate synthetic samples in batches and collect all results.

    :param args: Parsed arguments namespace with gen_batch_size and latent_dim.
    :param G: Generator model in evaluation mode.
    :param device: Torch device for tensor allocation.
    :param labels: Numpy array of integer labels for generation.
    :param n: Total number of samples to generate.
    :return: Tuple of (all_fake, all_labels, sample_generation_start_time).
    """

    batch_size = args.gen_batch_size  # Set generation batch size
    _gen_model = getattr(G, "module", G)  # Unwrap DataParallel if present to access feature_dim attribute
    feat_dim = cast(Any, _gen_model.feature_dim)  # Get output feature dimensionality cast to Any to suppress type-checker attribute errors
    feat_dim_int = int(feat_dim)  # Convert feature dimension to plain int for numpy shape allocation
    all_fake = np.empty((int(n), feat_dim_int), dtype=np.float32)  # Pre-allocate contiguous output array to avoid list-append-then-vstack overhead
    offset = 0  # Track write position in pre-allocated array
    sample_generation_start_time = time.time()  # Record sample generation start timestamp
    
    with torch.no_grad():  # Disable gradient computation for generation
        for i in range(0, n, batch_size):  # Loop over batches for generation
            b = min(batch_size, n - i)  # Calculate current batch size
            z = torch.randn(b, args.latent_dim, device=device)  # Sample noise for batch
            y = torch.from_numpy(labels[i : i + b]).to(device, dtype=torch.long)  # Convert labels to tensor
            fake = G(z, y).cpu().numpy()  # Generate fake samples and move to CPU
            all_fake[offset : offset + b] = fake  # Write batch directly into pre-allocated array slice
            offset += b  # Advance write offset by current batch size
    
    return (all_fake, labels, sample_generation_start_time)  # Return pre-allocated array and original labels array


def postprocess_generated_arrays_to_dataframe(args, config: Dict, all_fake: List, all_labels: List, scaler: Any, label_encoder: Any, feature_cols: List[str], device: torch.device, n: int, file_progress_prefix: str) -> pd.DataFrame:
    """
    Stack generated arrays, inverse-transform, and build output DataFrame.

    :param args: Parsed arguments namespace with label_col and out_file.
    :param config: Configuration dictionary with hardware_tracking setting.
    :param all_fake: List of generated feature batch arrays.
    :param all_labels: List of corresponding label arrays.
    :param scaler: Fitted StandardScaler for inverse transformation.
    :param label_encoder: Fitted LabelEncoder for label decoding.
    :param feature_cols: List of feature column names.
    :param device: Torch device used for hardware column population.
    :param n: Total number of generated samples.
    :param file_progress_prefix: Colored prefix string for progress display.
    :return: DataFrame with inverse-transformed features and decoded labels.
    """

    X_fake = np.vstack(all_fake) if isinstance(all_fake, list) else all_fake  # Stack batches if list or use pre-allocated array directly
    Y_fake = np.concatenate(all_labels) if isinstance(all_labels, list) else all_labels  # Concatenate if list or use array directly
    X_orig = scaler.inverse_transform(X_fake)  # Inverse transform features to original scale
    df = pd.DataFrame(X_orig, columns=feature_cols)  # Create DataFrame with original feature names
    df[args.label_col] = label_encoder.inverse_transform(Y_fake)  # Map integer labels back to original strings
    
    if config.get("hardware_tracking", False):  # If enabled in config
        try:  # Guard hardware population to avoid breaking generation
            df = populate_hardware_column(df, column_name="hardware", device_used=device)  # Populate hardware column
        except Exception:
            pass  # Ignore hardware population errors and continue
    
    generate_csv_and_image(df, args.out_file, is_visualizable=True)  # Save CSV and generate PNG image when appropriate
    print(f"{file_progress_prefix} {BackgroundColors.GREEN}Saved {BackgroundColors.CYAN}{n}{BackgroundColors.GREEN} generated samples to {BackgroundColors.CYAN}{args.out_file}{Style.RESET_ALL}")  # Print completion message with prefix
    
    if getattr(args, "csv_path", None):  # Verify csv_path is available to derive augmented output path
        try:  # Guard augmented file save to avoid breaking generation on path derivation failure
            csv_src = Path(args.csv_path)  # Create Path object from original training dataset path
            suffix = config.get("wgangp", {}).get("results_suffix", "_data_augmented")  # Read results suffix from wgangp config with fallback
            augmented_path = csv_src.parent / f"{csv_src.stem}{suffix}.csv"  # Derive augmented path in same directory as original dataset
            df.to_csv(str(augmented_path), index=False)  # Save generated samples to augmented file without row index
            print(f"{file_progress_prefix} {BackgroundColors.GREEN}Saved augmented dataset to {BackgroundColors.CYAN}{augmented_path}{Style.RESET_ALL}")  # Print augmented file save confirmation
            try:  # Guard telegram notification to avoid breaking generation on notify failure
                size_bytes = augmented_path.stat().st_size if augmented_path.exists() else 0  # Get augmented file size in bytes safely
                if size_bytes >= 1024 ** 3:  # If size is in gigabytes
                    size_str = f"{size_bytes / (1024 ** 3):.2f} GB"  # Format size in GB for readability
                else:  # Otherwise show size in megabytes
                    size_str = f"{size_bytes / (1024 ** 2):.2f} MB"  # Format size in MB for readability
                msg = f"{file_progress_prefix} Augmented dataset saved: {augmented_path.name} ({int(n)} samples, {size_str}) at {str(augmented_path)}"  # Compose detailed telegram message
                send_telegram_message(TELEGRAM_BOT, msg)  # Send Telegram notification with file, count and size details
            except Exception as _tg_e:  # If notification fails, warn to console but continue
                print(f"{BackgroundColors.YELLOW}Warning: failed to send Telegram notification about augmented file: {_tg_e}{Style.RESET_ALL}")  # Warn about telegram failure
        except Exception as _aug_e:  # If augmented file save fails, warn and continue without aborting generation
            print(f"{BackgroundColors.YELLOW}[WARNING] Failed to save augmented dataset to derived path: {_aug_e}{Style.RESET_ALL}")  # Warn about augmented save failure
    
    return df  # Return constructed DataFrame


def record_sample_generation_timing(args, sample_generation_start_time: float) -> None:
    """
    Compute and record the elapsed time for sample generation.

    :param args: Parsed arguments namespace to store timing on.
    :param sample_generation_start_time: Timestamp when sample generation started.
    :return: None.
    """

    try:  # Safely compute and print sample generation elapsed time
        sample_generation_elapsed = time.time() - sample_generation_start_time  # Calculate sample generation elapsed seconds
        args._last_sample_generation_time = float(sample_generation_elapsed)  # Store sample generation elapsed on args for downstream use
        print(f"{BackgroundColors.GREEN}Sample generation elapsed: {BackgroundColors.CYAN}{sample_generation_elapsed:.2f}s{Style.RESET_ALL}")  # Print sample generation elapsed
    except Exception as _sg:  # If timing calculation fails
        print(f"{BackgroundColors.YELLOW}Warning: failed to measure sample generation time: {_sg}{Style.RESET_ALL}")  # Warn but continue
        args._last_sample_generation_time = ""  # Ensure attribute exists even on failure


def notify_generation_finish_via_telegram(args, n: int, file_progress_prefix: str) -> None:
    """
    Build and send a Telegram notification about generation completion.

    :param args: Parsed arguments namespace with out_file.
    :param n: Total number of generated samples.
    :param file_progress_prefix: Colored prefix string for progress display.
    :return: None.
    """

    try:  # Build a safe, human-readable finish message and notify via Telegram
        gen_path = Path(args.out_file)  # Path object for generated file
        try:  # Try to determine original dataset size for ratio calculation
            original_num = None  # Default original count
            if getattr(args, "csv_path", None):  # If csv_path is available on args
                original_num = len(pd.read_csv(args.csv_path, low_memory=False))  # Count original CSV rows for ratio
        except Exception:
            original_num = None  # Fallback to None on any error
        try:  # Try to compute ratio string safely
            ratio_str = (
                f"{(safe_float(n,0.0)/safe_float(original_num,1.0))*100:.2f}% ({int(safe_float(n,0.0))}/{int(safe_float(original_num,0.0))})"
                if original_num and safe_float(original_num, 0.0) > 0.0
                else ""
            )  # Ratio string if original_num available and >0
        except Exception:
            ratio_str = ""  # Fallback empty ratio on error
        try:  # Try to compute generated file size string safely
            size_bytes = gen_path.stat().st_size if gen_path.exists() else 0  # File size in bytes when exists
            if size_bytes >= 1024 ** 3:  # Size in GB
                size_str = f"{size_bytes / (1024 ** 3):.2f} GB"  # Human-readable GB
            else:  # Size in MB
                size_str = f"{size_bytes / (1024 ** 2):.2f} MB"  # Human-readable MB
        except Exception:
            size_str = "Unknown size"  # Fallback when unable to determine size
        msg = f"{file_progress_prefix} Finished WGAN-GP generation: Saved {int(safe_float(n,0.0))} samples{(f' ({ratio_str}, {size_str})' if ratio_str or size_str else '')} to {gen_path.name}"  # Compose final message
        send_telegram_message(TELEGRAM_BOT, msg)  # Send message via Telegram
    except Exception:
        pass  # Ignore notification failures to avoid breaking flow


def resolve_generation_results_csv_path(args, config: Dict, args_ck: Dict) -> tuple:
    """
    Resolve the results CSV path for generation from args or checkpoint saved args.

    :param args: Parsed arguments namespace with csv_path attribute.
    :param config: Configuration dictionary with paths settings.
    :param args_ck: Saved arguments dictionary from checkpoint.
    :return: Tuple of (results_csv_path, ck_csv_path) where either may be None.
    """

    results_csv_path = None  # Initialize results CSV path variable
    ck_csv_path = None  # Initialize checkpoint csv Path placeholder to satisfy static analysis
    if getattr(args, "csv_path", None):  # If csv_path available on args
        csv_path_obj = Path(args.csv_path)  # Create Path object from args.csv_path
        data_aug_subdir = config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation")  # Read configured subdir name
        data_aug_dir = csv_path_obj.parent / data_aug_subdir  # Construct Data_Augmentation directory under dataset folder
        os.makedirs(data_aug_dir, exist_ok=True)  # Ensure Data_Augmentation directory exists before writing
        results_csv_path = data_aug_dir / "data_augmentation_results.csv"  # Use results CSV inside Data_Augmentation dir
    else:  # Try to recover original csv_path from checkpoint args saved in checkpoint
        ck_csv_path = None  # Reset checkpoint csv path
        try:  # Attempt recovery from checkpoint args
            if args_ck and args_ck.get("csv_path"):  # Use saved args from checkpoint (args_ck defined earlier)
                ck_csv_value = args_ck.get("csv_path")  # Get csv_path value from checkpoint args
                if ck_csv_value:  # If value is non-empty
                    ck_csv_path = Path(ck_csv_value)  # Path object for saved csv_path from checkpoint
                    ck_data_aug_subdir = config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation")  # Read subdir name from config
                    ck_data_aug_dir = ck_csv_path.parent / ck_data_aug_subdir  # Construct Data_Augmentation directory for checkpoint csv_path
                    os.makedirs(ck_data_aug_dir, exist_ok=True)  # Ensure Data_Augmentation directory exists for checkpoint recover
                    results_csv_path = ck_data_aug_dir / "data_augmentation_results.csv"  # Use results CSV inside Data_Augmentation dir for checkpoint
                else:  # Empty csv_path value in checkpoint
                    ck_csv_path = None  # Keep as None when value is empty
        except Exception:  # Recovery failed
            results_csv_path = None  # Leave as None if recovery fails
    return results_csv_path, ck_csv_path  # Return resolved results CSV path and checkpoint csv path


def collect_generation_file_metadata(args, ckpt: Dict, n: int, ck_csv_path) -> Dict:
    """
    Collect file metadata for generation results including filenames, counts, and timing.

    :param args: Parsed arguments namespace with csv_path, out_file, and timing attributes.
    :param ckpt: Loaded checkpoint dictionary containing metrics_history.
    :param n: Total number of generated samples.
    :param ck_csv_path: Checkpoint-recovered csv Path object (may be None).
    :return: Dictionary with generation metadata values.
    """

    metadata = {}  # Initialize metadata dictionary
    if getattr(args, "csv_path", None):  # Verify args.csv_path exists and is not None
        metadata["original_file_name"] = Path(args.csv_path).name  # Use basename from args.csv_path when available
    else:  # Fallback to checkpoint saved path when args.csv_path missing
        if ck_csv_path:  # Use guarded Path object from checkpoint when available
            try:  # Attempt to get name from checkpoint path
                metadata["original_file_name"] = ck_csv_path.name  # Use basename from guarded checkpoint Path
            except Exception:  # If name extraction fails
                metadata["original_file_name"] = ""  # Use empty string on any failure
        else:  # No path available
            metadata["original_file_name"] = ""  # Default to empty when no path available
    metadata["generated_file_name"] = Path(args.out_file).name if getattr(args, "out_file", None) else ""  # Generated file name
    try:  # Attempt to count original CSV rows
        metadata["original_num"] = None  # Default original count
        if getattr(args, "csv_path", None):  # If csv_path provided, try reading length
            metadata["original_num"] = len(pd.read_csv(args.csv_path, low_memory=False))  # Count original CSV rows
    except Exception:  # Reading failed
        metadata["original_num"] = None  # Leave as None if reading fails
    metadata["total_generated"] = int(n) if n is not None else ""  # Total generated samples
    metadata["generated_ratio"] = ""  # Default generated ratio
    try:  # Attempt to compute generated ratio
        if metadata["original_num"] and metadata["original_num"] > 0:  # If original count available
            metadata["generated_ratio"] = float(metadata["total_generated"]) / float(metadata["original_num"])  # Compute ratio
    except Exception:  # Computation failed
        metadata["generated_ratio"] = ""  # Leave blank on failure
    metadata["timing_values"] = {  # Map common column names to stored timing attributes
        "training_time_s": getattr(args, "_last_training_time", ""),  # Total training elapsed seconds
        "file_time_s": getattr(args, "_last_file_time", ""),  # Per-file processing elapsed seconds
        "epoch_time_s": getattr(args, "_last_epoch_time", ""),  # Last epoch elapsed seconds
        "sample_generation_time_s": getattr(args, "_last_sample_generation_time", ""),  # Sample generation elapsed seconds
        "model_save_time_s": getattr(args, "_last_model_save_time", ""),  # Model save phase elapsed seconds
    }  # End timing values map
    metadata["critic_loss"] = ""  # Default critic loss
    metadata["generator_loss"] = ""  # Default generator loss
    try:  # Attempt to extract losses from checkpoint metrics
        metrics_history = ckpt.get("metrics_history")  # Try to get metrics history from checkpoint (may be None)
        if isinstance(metrics_history, dict):  # If metrics_history is a dict
            ld = metrics_history.get("loss_D") or []  # Safe list for discriminator losses
            lg = metrics_history.get("loss_G") or []  # Safe list for generator losses
            if isinstance(ld, (list, tuple)) and len(ld) > 0:  # If list-like and non-empty
                metadata["critic_loss"] = ld[-1]  # Use last recorded discriminator loss
            if isinstance(lg, (list, tuple)) and len(lg) > 0:  # If list-like and non-empty
                metadata["generator_loss"] = lg[-1]  # Use last recorded generator loss
    except Exception:  # Extraction failed
        metadata["critic_loss"] = ""  # Ignore failures and leave blank
        metadata["generator_loss"] = ""  # Ignore failures and leave blank
    return metadata  # Return populated metadata dictionary


def build_generation_runtime_row(args, config: Dict, n: int, device: torch.device, metadata: Dict) -> Dict:
    """
    Build runtime metrics dictionary for generation results CSV row.

    :param args: Parsed arguments namespace with generation settings and timing.
    :param config: Configuration dictionary for hyperparameter lookup.
    :param n: Total number of generated samples.
    :param device: Torch device used for hardware specification lookup.
    :param metadata: Dictionary with generation file metadata from collect_generation_file_metadata.
    :return: Dictionary mapping column names to runtime-computed values for CSV row.
    """

    row_runtime_defaults = {  # Default values for common runtime columns
        "critic_iterations": getattr(args, "critic_steps", ""),  # Critic iterations default
        "learning_rate_generator": getattr(args, "lr", ""),  # Generator LR default
        "learning_rate_critic": getattr(args, "lr", ""),  # Critic LR default
        "testing_time_s": "",  # No testing time available by default
        "hardware": None,  # Hardware placeholder
    }  # End runtime defaults
    row_runtime = {}  # Dictionary to hold runtime values for configured columns
    row_runtime["original_file"] = metadata["original_file_name"]  # Original CSV filename
    row_runtime["generated_file"] = metadata["generated_file_name"]  # Generated output filename
    row_runtime["original_num_samples"] = metadata["original_num"] if metadata["original_num"] is not None else ""  # Original sample count
    row_runtime["total_generated_samples"] = metadata["total_generated"]  # Total generated count
    row_runtime["generated_ratio"] = metadata["generated_ratio"]  # Generated/original ratio
    row_runtime["critic_loss"] = metadata["critic_loss"]  # Last critic loss from checkpoint metrics
    row_runtime["generator_loss"] = metadata["generator_loss"]  # Last generator loss from checkpoint metrics
    for k, v in metadata["timing_values"].items():  # For each timing key known
        row_runtime[k] = v  # Store timing value under the timing key
    try:  # Apply runtime defaults for missing columns
        for kk, vv in row_runtime_defaults.items():  # For each default key
            if kk not in row_runtime or row_runtime.get(kk) in (None, ""):  # If column missing or empty
                row_runtime[kk] = vv  # Apply default value
    except Exception:  # Ignore default application errors
        pass  # Continue despite errors
    try:  # Attempt to inject hardware specification string
        if (row_runtime.get("hardware") is None) or row_runtime.get("hardware") == "":  # If hardware not yet set
            hw_specs = get_hardware_specifications(device_used=device) if 'get_hardware_specifications' in globals() else None  # Query hardware specs
            if isinstance(hw_specs, dict):  # If specs returned a valid dict
                hw_part = hw_specs.get("gpu", "None") if hw_specs.get("gpu", None) is not None else "None"  # GPU part
                hardware_str = (  # Build human-readable hardware string
                    f"{hw_specs.get('cpu_model','Unknown')} | Cores: {hw_specs.get('cores','N/A')}"
                    f" | RAM: {hw_specs.get('ram_gb','N/A')} GB | OS: {hw_specs.get('os','Unknown')}"
                    f" | GPU: {hw_part} | CUDA: {hw_specs.get('cuda','No')} | Device Used: {hw_specs.get('device_used','Unknown')}"
                )  # End hardware string
                row_runtime["hardware"] = hardware_str  # Store hardware specification in runtime dict
    except Exception:  # Ignore hardware detection errors
        pass  # Continue despite hardware detection failure
    return row_runtime  # Return populated generation runtime dictionary


def prepare_and_write_results_csv_row(args, config: Dict, args_ck: Dict, ckpt: Dict, n: int, device: torch.device) -> None:
    """
    Prepare and write a results CSV row with generation metrics and hyperparameters.

    :param args: Parsed arguments namespace with csv_path and out_file.
    :param config: Configuration dictionary with wgangp and paths settings.
    :param args_ck: Saved arguments dictionary from checkpoint.
    :param ckpt: Loaded checkpoint dictionary containing metrics_history.
    :param n: Total number of generated samples.
    :param device: Torch device used for hardware specification lookup.
    :return: None.
    """

    results_cols_cfg = config.get("wgangp", {}).get("results_csv_columns", [])  # Read configured results columns list
    if not isinstance(results_cols_cfg, list) or len(results_cols_cfg) == 0:  # Validate list exists and is non-empty
        print(f"{BackgroundColors.RED}Configuration error: 'results_csv_columns' missing, empty, or not a list under 'wgangp' section in configuration.{Style.RESET_ALL}")  # Clear error message
        raise ValueError("'results_csv_columns' missing, empty, or not a list under 'wgangp' section in configuration")  # Stop safely
    results_csv_path, ck_csv_path = resolve_generation_results_csv_path(args, config, args_ck)  # Resolve results CSV path from args or checkpoint
    if results_csv_path is not None:  # If we have a place to record results
        metadata = collect_generation_file_metadata(args, ckpt, n, ck_csv_path)  # Collect file metadata for generation results
        row_runtime = build_generation_runtime_row(args, config, n, device, metadata)  # Build runtime metrics dictionary for CSV row
        try:  # Wrap open/write in try/except to avoid crashing on I/O issues
            f_handle, writer = open_results_csv(results_csv_path, results_cols_cfg)  # Get persistent handle and writer
            if f_handle and writer:  # If we successfully opened or reused a writer
                ordered = build_ordered_csv_row_from_runtime(row_runtime, results_cols_cfg, config)  # Build ordered CSV row from runtime values using shared helper
                writer.writerow(ordered)  # Write the ordered row to CSV
                flush_csv_file_safely(f_handle)  # Flush CSV file to disk safely
        except Exception as _we:  # On any write/open failure, warn and continue
            print(f"{BackgroundColors.YELLOW}Warning: failed to persist generation row: {_we}{Style.RESET_ALL}")  # Warn but do not abort


def warn_on_results_csv_failure(e: Exception) -> None:
    """
    Print a warning when results CSV preparation fails.

    :param e: Exception that caused the failure.
    :return: None.
    """

    print(f"{BackgroundColors.YELLOW}Warning: could not prepare results CSV entry: {e}{Style.RESET_ALL}")  # Warn on top-level failures


def handle_generate_top_level_exception(e: Exception) -> None:
    """
    Handle top-level exceptions in the generate function by logging and re-raising.

    :param e: Exception that was caught at the top level.
    :return: None.
    """

    print(str(e))  # Print exception message to terminal
    send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full exception via Telegram
    raise e  # Re-raise the exception to propagate


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
        ckpt, args_ck, scaler, label_encoder, feature_cols, class_distribution, device, file_progress_prefix = normalize_args_and_load_checkpoint(args, config)  # Normalize args, select device, and load checkpoint
        scaler, label_encoder, feature_cols, class_distribution = reconstruct_missing_preprocessing_objects(args, scaler, label_encoder, feature_cols, class_distribution)  # Reconstruct missing preprocessing objects from CSV if needed
        feature_dim, n_classes = determine_feature_dim_and_n_classes(args, scaler, label_encoder)  # Determine feature dimensionality and class count
        G = build_and_load_generator(args, config, ckpt, device, feature_dim, n_classes)  # Build and load generator model from checkpoint
        n_per_class, labels, n = compute_generation_counts_and_labels(args, config, class_distribution, label_encoder, n_classes)  # Compute per-class counts and label array
        notify_start_of_generation(args, n)  # Send Telegram notification about generation start
        all_fake, all_labels, sample_generation_start_time = generate_batches_and_collect_results(args, G, device, labels, n)  # Generate synthetic samples in batches
        df = postprocess_generated_arrays_to_dataframe(args, config, all_fake, all_labels, scaler, label_encoder, feature_cols, device, n, file_progress_prefix)  # Postprocess arrays into DataFrame and save
        record_sample_generation_timing(args, sample_generation_start_time)  # Record sample generation elapsed time
        notify_generation_finish_via_telegram(args, n, file_progress_prefix)  # Send Telegram finish notification
        try:  # Wrap result writing in try/except to avoid breaking generation on failures
            prepare_and_write_results_csv_row(args, config, args_ck, ckpt, n, device)  # Prepare and write results CSV row
        except Exception as e:
            warn_on_results_csv_failure(e)  # Warn on results CSV preparation failure
    except Exception as e:
        handle_generate_top_level_exception(e)  # Handle top-level exception by logging and re-raising


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

        try:  # Detect GPU availability and build GPU summary
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
            gpu_part = hardware_specs.get('gpu', 'None') if hardware_specs.get('gpu', None) is not None else 'None'  # GPU part for readable hardware string
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


def row_style_for_zebra(row):
    """
    Top-level helper to produce zebra row styles for pandas Styler.

    :param row: pandas Series representing a row
    :return: List[str] of CSS style strings for each cell
    """
    bg = "white" if (row.name % 2) == 0 else "#f2f2f2"  # white for even rows, light gray for odd rows
    return [f"background-color: {bg};" for _ in row.index]  # Return style for every column in the row


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


def build_config_overrides_from_kwargs(kwargs: Dict) -> Dict:
    """
    Convert keyword arguments into a nested configuration overrides dictionary.

    Maps recognized keyword argument names to their corresponding configuration
    sections (wgangp, paths, training, generator, discriminator, generation,
    dataloader, dataset, execution, sound). Warns on unrecognized keys.

    :param kwargs: Flat keyword arguments to convert
    :return: Nested configuration overrides dictionary
    """

    try:
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
        return cli_style_overrides  # Return the assembled overrides dictionary
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def dispatch_wgangp_execution_mode(args, final_config: Dict) -> None:
    """
    Dispatch WGAN-GP execution based on the selected mode (train, gen, or both).

    Handles mode routing, training time tracking, checkpoint resolution for
    combined mode, and generation invocation.

    :param args: ConfigNamespace holding resolved execution arguments
    :param final_config: Merged configuration dictionary
    :return: None
    """

    try:
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
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_wgangp(config: Optional[Union[Dict, str]] = None, **kwargs) -> None:  # Entry point wrapper accepting optional config dict or path
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
            cli_style_overrides = build_config_overrides_from_kwargs(kwargs)  # Convert kwargs to config overrides using extracted helper
            final_config = deep_merge(final_config, cli_style_overrides)  # Apply kwargs overrides

        CONFIG = final_config  # Update global config

        results_cols_chk = final_config.get("wgangp", {}).get("results_csv_columns")  # Validate and read configured results columns list from wgangp section
        if not isinstance(results_cols_chk, list) or len(results_cols_chk) == 0:  # Ensure the value exists, is a list, and is non-empty
            print(f"{BackgroundColors.RED}Configuration error: 'results_csv_columns' missing, empty, or not a list under 'wgangp' section in configuration.{Style.RESET_ALL}")  # Print clear error message
            raise ValueError("'results_csv_columns' missing, empty, or not a list under 'wgangp' section in configuration")  # Stop execution safely with clear error
        initialize_logger(final_config)  # Initialize logger with final configuration

        setup_telegram_bot(final_config)  # Setup Telegram bot with final configuration

        args = ConfigNamespace(final_config)  # Create namespace from config

        start_time = datetime.datetime.now()  # Record start time
        send_telegram_message(TELEGRAM_BOT, f"Starting WGAN-GP (programmatic) at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        try:  # Execute with error handling
            dispatch_wgangp_execution_mode(args, final_config)  # Dispatch execution based on mode using extracted helper
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
    
    try:
        stop_resource_monitor()  # Signal and join resource monitor thread if running
    except Exception:
        pass  # Ignore any errors stopping the monitor during exit

    for key, (f, _) in list(RESULTS_CSV_HANDLES.items()):  # Iterate over cached handles
        try:
            if f and not f.closed:  # If file object exists and is open
                f.close()  # Close the file
        except Exception:
            pass  # Ignore close errors


def initialize_cli_and_config() -> Dict:
    """
    Parse CLI arguments, load configuration, and set the global CONFIG variable.

    :return: Merged configuration dictionary.
    """

    global CONFIG  # Declare global CONFIG for mutation
    args = parse_args()  # Parse command-line arguments from CLI
    cli_overrides = args_to_config_overrides(args)  # Convert parsed CLI args to configuration overrides dict
    config = load_configuration(config_path=args.config, cli_overrides=cli_overrides)  # Load and merge configuration from file and CLI
    CONFIG = config  # Persist loaded configuration in global registry
    return config  # Return merged configuration dictionary to caller


def extract_runtime_parameters(config: Dict) -> tuple:
    """
    Extract runtime parameters from configuration and create the argument namespace.

    :param config: Merged configuration dictionary.
    :return: Tuple of (mode, csv_path, results_suffix, datasets, args).
    """

    mode = config.get("wgangp", {}).get("mode", "both")  # Read execution mode from wgangp config section
    csv_path = config.get("wgangp", {}).get("csv_path")  # Read optional single-file CSV path from config
    results_suffix = config.get("wgangp", {}).get("results_suffix", "_data_augmented")  # Read output filename suffix from config
    datasets = config.get("dataset", {}).get("datasets", {})  # Read batch dataset path registry from config
    args = ConfigNamespace(config)  # Build unified argument namespace wrapping the config dict
    return mode, csv_path, results_suffix, datasets, args  # Return all extracted runtime parameters as a tuple


def validate_results_csv_columns(config: Dict) -> list:
    """
    Validate that results_csv_columns is present, non-empty, and is a list.

    :param config: Configuration dictionary containing wgangp settings.
    :return: Validated list of results CSV column names.
    """

    results_cols = config.get("wgangp", {}).get("results_csv_columns")  # Read configured results columns list from wgangp section
    if not isinstance(results_cols, list) or len(results_cols) == 0:  # Verify value exists, is a list, and is non-empty
        print(f"{BackgroundColors.RED}Configuration error: 'results_csv_columns' missing, empty, or not a list under 'wgangp' section in configuration.{Style.RESET_ALL}")  # Print descriptive configuration error
        raise ValueError("'results_csv_columns' missing, empty, or not a list under 'wgangp' section in configuration")  # Halt execution with descriptive error
    return results_cols  # Return validated column list to caller


def create_results_csv_if_absent(results_csv_path: Path, results_cols: list) -> None:
    """
    Create the results CSV file with a header row if it does not already exist.

    :param results_csv_path: Path object pointing to the desired results CSV location.
    :param results_cols: List of column names to write as the CSV header.
    :return: None
    """

    if not results_csv_path.exists():  # Only write header when file does not already exist
        os.makedirs(results_csv_path.parent, exist_ok=True)  # Ensure parent directory exists before writing
        with open(results_csv_path, "w", newline="", encoding="utf-8") as _f:  # Open results CSV for header writing
            writer = csv.writer(_f)  # Instantiate CSV writer for the new file
            writer.writerow(results_cols)  # Write header row in configured column order


def setup_single_file_output_path(args: Any, config: Dict, csv_path_obj: Path, mode: str, results_suffix: str) -> Path:
    """
    Configure the output file path for single-file mode and return the data augmentation directory.

    :param args: Argument namespace containing out_file attribute to update.
    :param config: Configuration dictionary with paths settings.
    :param csv_path_obj: Path object for the input CSV file.
    :param mode: Execution mode string (train, gen, or both).
    :param results_suffix: Suffix string appended to the input filename for the output file.
    :return: Path object for the canonical data augmentation output directory.
    """

    if args.out_file == "generated.csv" and mode in ["gen", "both"]:  # Update default output path when in generation or combined mode
        data_aug_subdir = config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation")  # Get configured subdir name for this branch
        data_aug_dir = csv_path_obj.parent / data_aug_subdir  # Construct Data_Augmentation subdirectory path alongside input file
        os.makedirs(data_aug_dir, exist_ok=True)  # Ensure Data_Augmentation directory exists before output path assignment
        output_filename = f"{csv_path_obj.stem}{results_suffix}{csv_path_obj.suffix}"  # Build output filename by appending suffix to input stem
        args.out_file = str(data_aug_dir / output_filename)  # Assign resolved output file path to args for downstream use
    data_aug_subdir = config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation")  # Re-read subdir name for authoritative path computation
    data_aug_dir = csv_path_obj.parent / data_aug_subdir  # Build canonical Data_Augmentation directory path for results CSV
    os.makedirs(data_aug_dir, exist_ok=True)  # Ensure canonical Data_Augmentation directory exists before results CSV creation
    return data_aug_dir  # Return canonical data augmentation directory path for downstream use


def execute_training_with_timing(args: Any, config: Dict) -> None:
    """
    Execute model training and store elapsed time on args.

    :param args: Argument namespace with all training configuration settings.
    :param config: Configuration dictionary passed to the train function.
    :return: None
    """

    training_start_time = time.time()  # Record training start timestamp for elapsed time calculation
    train(args, config)  # Execute WGAN-GP training loop with provided configuration
    args._last_training_time = time.time() - training_start_time  # Store elapsed training seconds on args for reporting


def execute_generation_with_verification(args: Any, config: Dict) -> None:
    """
    Verify whether generation is needed and execute it if so.

    :param args: Argument namespace containing generation settings and output path.
    :param config: Configuration dictionary passed to the generate function.
    :return: None
    """

    if verify_data_augmentation_file(args, config):  # Verify if generation is necessary according to existing output and configuration
        generate(args, config)  # Execute synthetic sample generation when output is absent or insufficient
    else:  # Skip generation when configured n_samples already satisfied by existing output
        print(f"{BackgroundColors.GREEN}Skipping generation: output file already satisfies configured n_samples.{Style.RESET_ALL}")  # Inform user that generation is skipped


def resolve_checkpoint_after_training(args: Any, config: Dict, csv_path_obj: Path, data_aug_dir: Path) -> Path:
    """
    Resolve the generator checkpoint path produced after training and assign it to args.

    :param args: Argument namespace containing epochs count and checkpoint attribute.
    :param config: Configuration dictionary with paths settings.
    :param csv_path_obj: Path object for the trained CSV source file.
    :param data_aug_dir: Path object for the data augmentation output directory.
    :return: Resolved checkpoint Path object pointing to the latest available checkpoint.
    """

    checkpoint_prefix = csv_path_obj.stem  # Derive checkpoint filename prefix from input CSV stem
    checkpoint_subdir = config.get("paths", {}).get("checkpoint_subdir", "Checkpoints")  # Read configured checkpoint subdirectory name
    checkpoint_dir = data_aug_dir / checkpoint_subdir  # Build checkpoint directory path under Data_Augmentation
    checkpoint_path = checkpoint_dir / f"{checkpoint_prefix}_generator_epoch{args.epochs}.pt"  # Construct expected final-epoch checkpoint filename
    
    if not checkpoint_path.exists():  # Fall back to latest checkpoint when specific epoch file is absent
        checkpoints = sorted(checkpoint_dir.glob(f"{checkpoint_prefix}_generator_epoch*.pt"))  # Discover all matching checkpoints in directory
        if checkpoints:  # Assign latest checkpoint when at least one file exists
            checkpoint_path = checkpoints[-1]  # Select the last entry by sorted order as the latest checkpoint
        else:  # Raise a descriptive error when no checkpoints exist at all
            raise FileNotFoundError(f"No generator checkpoint found for {csv_path_obj.name} in {checkpoint_dir}")  # Abort with informative message
        
    args.checkpoint = str(checkpoint_path)  # Assign resolved checkpoint path string to args for generation
    
    return checkpoint_path  # Return resolved Path object for caller printing use


def run_both_mode_for_csv(args: Any, config: Dict, csv_path_obj: Path, data_aug_dir: Path, training_label: str) -> None:
    """
    Execute combined train-then-generate workflow for a single CSV file.

    :param args: Argument namespace with training and generation settings.
    :param config: Configuration dictionary with paths and model settings.
    :param csv_path_obj: Path object for the input CSV file being processed.
    :param data_aug_dir: Path object for the data augmentation output directory.
    :param training_label: Label suffix string inserted into the [1/2] training progress message.
    :return: None
    """

    print(f"{BackgroundColors.GREEN}[1/2] {training_label}{Style.RESET_ALL}")  # Print training phase progress header with label
    execute_training_with_timing(args, config)  # Train model and record elapsed time on args
    checkpoint_path = resolve_checkpoint_after_training(args, config, csv_path_obj, data_aug_dir)  # Resolve generator checkpoint produced by training
    print(f"\n{BackgroundColors.CYAN}[2/2] Generating samples from {checkpoint_path.name}...{Style.RESET_ALL}")  # Print generation phase progress header with checkpoint filename
    print(f"{BackgroundColors.GREEN}Output will be saved to: {BackgroundColors.CYAN}{args.out_file}{Style.RESET_ALL}")  # Print resolved output file destination path
    execute_generation_with_verification(args, config)  # Verify necessity and execute synthetic sample generation


def dispatch_single_file_mode(args: Any, config: Dict, mode: str, csv_path_obj: Path, data_aug_dir: Path) -> None:
    """
    Dispatch the configured execution mode for a single-file run.

    :param args: Argument namespace with mode-specific settings.
    :param config: Configuration dictionary passed to train and generate functions.
    :param mode: Execution mode string (train, gen, or both).
    :param csv_path_obj: Path object for the input CSV file.
    :param data_aug_dir: Path object for the data augmentation output directory.
    :return: None
    """

    if mode == "train":  # Training-only mode for single-file run
        train(args, config)  # Execute training without elapsed time capture
    elif mode == "gen":  # Generation-only mode for single-file run
        assert args.checkpoint is not None, "Generation requires --checkpoint"  # Verify checkpoint is provided before generation
        execute_generation_with_verification(args, config)  # Verify necessity and run generation
    elif mode == "both":  # Combined train-then-generate mode for single-file run
        run_both_mode_for_csv(args, config, csv_path_obj, data_aug_dir, "Training model...")  # Execute full train-generate pipeline


def handle_single_file_mode(args: Any, config: Dict, mode: str, csv_path: str, results_cols: list, results_suffix: str) -> None:
    """
    Orchestrate all processing steps for a single input CSV file.

    :param args: Argument namespace updated with output file path during execution.
    :param config: Configuration dictionary with paths and execution settings.
    :param mode: Execution mode string (train, gen, or both).
    :param csv_path: String path to the single input CSV file.
    :param results_cols: List of column names for the results CSV header.
    :param results_suffix: Suffix appended to the input filename for the output file.
    :return: None
    """

    csv_path_obj = Path(csv_path)  # Create Path object from csv_path string for downstream path operations
    data_aug_dir = setup_single_file_output_path(args, config, csv_path_obj, mode, results_suffix)  # Configure output path and retrieve canonical data augmentation directory
    results_csv_path = data_aug_dir / "data_augmentation_results.csv"  # Construct results CSV path inside data augmentation directory
    create_results_csv_if_absent(results_csv_path, results_cols)  # Write header to results CSV when file does not exist
    dispatch_single_file_mode(args, config, mode, csv_path_obj, data_aug_dir)  # Dispatch configured mode to appropriate handler


def apply_dataset_ordering(files: list, generating_order: str) -> list:
    """
    Sort the list of files according to the configured dataset ordering strategy.

    :param files: List of file path strings to sort.
    :param generating_order: Ordering strategy string (off, Ascending, or Descending).
    :return: Ordered list of file path strings.
    """

    if generating_order not in ("off", "Ascending", "Descending"):  # Validate ordering value against allowed set
        generating_order = "off"  # Fall back to off when value is unrecognized
    if generating_order == "Ascending":  # Sort files from smallest to largest by file size
        files = sorted(files, key=lambda f: os.path.getsize(f))  # Apply ascending size sort to file list
        safe_debug("WGANGP dataset generation order: Ascending")  # Log chosen ascending ordering strategy
    elif generating_order == "Descending":  # Sort files from largest to smallest by file size
        files = sorted(files, key=lambda f: os.path.getsize(f), reverse=True)  # Apply descending size sort to file list
        safe_debug("WGANGP dataset generation order: Descending")  # Log chosen descending ordering strategy
    else:  # Preserve original discovery order when ordering is disabled
        safe_debug("WGANGP dataset generation order: Off")  # Log that no ordering is applied to the file list
    return files  # Return ordered file list for iteration


def build_file_progress_prefix(index: int, total: int) -> str:
    """
    Build a colored [index/total] progress prefix string for file processing messages.

    :param index: Current file index (1-based).
    :param total: Total number of files to process.
    :return: Colored progress prefix string enclosed in cyan brackets.
    """

    return f"{BackgroundColors.CYAN}[{index}/{total}]{Style.RESET_ALL}"  # Return formatted cyan-colored progress prefix


def setup_per_file_output(args: Any, config: Dict, file: str, results_suffix: str) -> tuple:
    """
    Configure the per-file output path attributes on args and return path objects.

    :param args: Argument namespace to update with csv_path and out_file attributes.
    :param config: Configuration dictionary with paths settings.
    :param file: String path to the CSV file currently being processed.
    :param results_suffix: Suffix appended to the input filename for the output file.
    :return: Tuple of (csv_path_obj, data_aug_dir) Path objects.
    """

    csv_path_obj = Path(file)  # Create Path object from file string for path operations
    data_aug_subdir = config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation")  # Get configured data augmentation subdirectory name
    data_aug_dir = csv_path_obj.parent / data_aug_subdir  # Construct Data_Augmentation directory path relative to input file
    os.makedirs(data_aug_dir, exist_ok=True)  # Ensure Data_Augmentation directory exists before output file creation
    output_filename = f"{csv_path_obj.stem}{results_suffix}{csv_path_obj.suffix}"  # Build output filename by appending suffix to input stem
    args.out_file = str(data_aug_dir / output_filename)  # Assign computed output file path to args
    args.csv_path = file  # Assign current CSV file path to args for train and generate functions
    return csv_path_obj, data_aug_dir  # Return Path objects for mode dispatch use


def dispatch_mode_for_file(args: Any, config: Dict, mode: str, csv_path_obj: Path, data_aug_dir: Path) -> None:
    """
    Dispatch the configured execution mode for a single batched CSV file.

    :param args: Argument namespace with mode-specific training and generation settings.
    :param config: Configuration dictionary passed to train and generate functions.
    :param mode: Execution mode string (train, gen, or both).
    :param csv_path_obj: Path object for the current CSV file being processed.
    :param data_aug_dir: Path object for the data augmentation output directory.
    :return: None
    """

    if mode == "train":  # Training-only mode for batched file
        execute_training_with_timing(args, config)  # Train model and capture elapsed time on args
    elif mode == "gen":  # Generation-only mode for batched file
        assert args.checkpoint is not None, "Generation requires --checkpoint"  # Verify checkpoint is provided before generation
        execute_generation_with_verification(args, config)  # Verify necessity and execute generation
    elif mode == "both":  # Combined train-then-generate mode for batched file
        training_label = f"Training model on {BackgroundColors.CYAN}{csv_path_obj.name}{BackgroundColors.GREEN}..."  # Build label including filename for progress message
        run_both_mode_for_csv(args, config, csv_path_obj, data_aug_dir, training_label)  # Execute full train-generate pipeline with filename-specific label


def mark_file_as_processed(resolved_path: str) -> None:
    """
    Register the resolved file path in PROCESSED_FILES and flush the logger.

    :param resolved_path: Absolute string path to register as processed in this run.
    :return: None
    """

    global PROCESSED_FILES  # Access global processed-files registry for mutation
    try:  # Guard registry update against unexpected errors
        PROCESSED_FILES.add(resolved_path)  # Register resolved path to prevent duplicate processing
    except Exception:  # Silently ignore registry update failures
        pass  # Continue despite registry update failure
    try:  # Attempt logger flush to persist buffered output to disk
        if logger is not None:  # Only flush when logger is initialized
            logger.flush()  # Flush log file buffer to disk
    except Exception:  # Silently ignore logger flush failures
        pass  # No-op on flush failure


def process_single_dataset_file(args: Any, config: Dict, mode: str, file: str, index: int, total_files: int, results_suffix: str) -> None:
    """
    Process a single CSV file within a batch dataset loop.

    :param args: Argument namespace updated in-place with per-file output paths.
    :param config: Configuration dictionary with paths and execution settings.
    :param mode: Execution mode string (train, gen, or both).
    :param file: String path to the CSV file to process.
    :param index: 1-based index of this file among all files in the current dataset path.
    :param total_files: Total number of files being processed in this dataset path.
    :param results_suffix: Suffix appended to each input filename for its output file.
    :return: None
    """

    file_progress_prefix = build_file_progress_prefix(index, total_files)  # Build colored [index/total] progress prefix once per file
    args.file_progress_prefix = file_progress_prefix  # Attach colored prefix to args for use in train and generate functions
    
    try:  # Guard path resolution against OS-level failures
        resolved_path = str(Path(file).resolve())  # Resolve file to absolute canonical path string
    except Exception:  # Handle path resolution failure gracefully
        resolved_path = str(Path(file))  # Fall back to non-resolved path string on resolution failure
    
    if resolved_path in PROCESSED_FILES:  # Skip files already processed in this run
        print(f"{BackgroundColors.YELLOW}{file_progress_prefix} Skipping already-processed file: {resolved_path}{Style.RESET_ALL}")  # Warn user about duplicate detection with prefix
        return  # Exit early to prevent duplicate processing
    
    print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*80}{Style.RESET_ALL}")  # Print decorative separator before file processing header
    print(f"{file_progress_prefix} {BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing file: {BackgroundColors.CYAN}{file}{Style.RESET_ALL}")  # Announce current file with progress prefix
    print(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}{'='*80}{Style.RESET_ALL}\n")  # Print decorative separator after file processing header
    csv_path_obj, data_aug_dir = setup_per_file_output(args, config, file, results_suffix)  # Configure output paths and retrieve Path objects for dispatch
    
    try:  # Execute configured mode with guaranteed cleanup in finally block
        dispatch_mode_for_file(args, config, mode, csv_path_obj, data_aug_dir)  # Dispatch train, gen, or both to handler
    finally:  # Always mark file as processed even when mode execution raises an error
        mark_file_as_processed(resolved_path)  # Register path in PROCESSED_FILES and flush logger


def process_dataset_path(args: Any, config: Dict, mode: str, input_path: str, results_cols: list, results_suffix: str) -> None:
    """
    Process all CSV files discovered within a single dataset directory path.

    :param args: Argument namespace updated per-file during batch processing.
    :param config: Configuration dictionary with paths and execution settings.
    :param mode: Execution mode string (train, gen, or both).
    :param input_path: String path to the dataset directory to scan for CSV files.
    :param results_cols: List of column names for the per-directory results CSV header.
    :param results_suffix: Suffix appended to each input filename for its output file.
    :return: None
    """

    data_aug_subdir = config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation")  # Read configured data augmentation subdirectory name
    data_aug_dir = Path(input_path) / data_aug_subdir  # Construct Data_Augmentation directory path inside dataset root
    per_dir_results_csv = data_aug_dir / "data_augmentation_results.csv"  # Build path for per-directory results CSV
    
    create_results_csv_if_absent(per_dir_results_csv, results_cols)  # Write header to results CSV when file does not yet exist
    
    files_to_process = get_files_to_process(input_path, file_extension=".csv", config=config)  # Collect all CSV files in dataset directory
    generating_order = config.get("wgangp", {}).get("generating_order", "off")  # Read configured generation ordering strategy
    files_to_process = apply_dataset_ordering(files_to_process, generating_order)  # Apply ordering strategy to collected file list
    total_files = len(files_to_process)  # Count total files for progress display
    
    for index, file in enumerate(files_to_process, start=1):  # Iterate files with 1-based index for progress tracking
        try:  # Catch per-file errors to allow batch to continue after individual failures
            process_single_dataset_file(args, config, mode, file, index, total_files, results_suffix)  # Process file with progress tracking
        except Exception as e:  # Handle any unrecovered error from file processing
            print(f"{BackgroundColors.RED}Error processing {BackgroundColors.CYAN}{file}{BackgroundColors.RED}: {e}{Style.RESET_ALL}")  # Print descriptive error with file path
            traceback.print_exc()  # Print full exception traceback for debugging investigation
            continue  # Advance to next file despite error in current file


def run_batch_mode(args: Any, config: Dict, datasets: Dict, mode: str, results_suffix: str, results_cols: list) -> None:
    """
    Iterate over all configured datasets and process each discovered CSV file.

    :param args: Argument namespace updated per-file during batch processing.
    :param config: Configuration dictionary with paths and execution settings.
    :param datasets: Dictionary mapping dataset names to lists of directory path strings.
    :param mode: Execution mode string (train, gen, or both).
    :param results_suffix: Suffix appended to each input filename for its output file.
    :param results_cols: List of column names for each per-directory results CSV header.
    :return: None
    """

    print(f"{BackgroundColors.GREEN}No CSV path provided. Processing datasets in batch mode...{Style.RESET_ALL}")  # Announce batch mode entry to user
    
    for dataset_name, paths in datasets.items():  # Iterate over each named dataset entry in registry
        print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing dataset: {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}")  # Announce current dataset name
        for input_path in paths:  # Iterate over each directory path within the dataset
            if not verify_filepath_exists(input_path):  # Skip paths that do not exist on disk
                verbose_output(f"{BackgroundColors.YELLOW}Skipping missing path: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}", config=config)  # Log skipped path at verbose level
                continue  # Advance to next path in dataset
            process_dataset_path(args, config, mode, input_path, results_cols, results_suffix)  # Process all CSV files in this dataset directory
    
    print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Batch processing completed!{Style.RESET_ALL}")  # Announce batch processing completion


def print_execution_summary(start_time: datetime.datetime, finish_time: datetime.datetime) -> None:
    """
    Print the program start time, finish time, and total execution duration.

    :param start_time: Datetime representing when the program started.
    :param finish_time: Datetime representing when the program finished.
    :return: None
    """

    print(f"\n{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}")  # Print formatted timing summary with start, finish, and elapsed duration
    print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}")  # Print program completion message


def register_exit_handlers(config: Dict) -> None:
    """
    Register atexit cleanup handlers for sound playback and results CSV flushing.

    :param config: Configuration dictionary containing sound settings.
    :return: None
    """

    if config.get("sound", {}).get("enabled", True):  # Register sound playback only when sound is enabled in config
        atexit.register(lambda: play_sound(config))  # Schedule sound playback callback at program exit
    atexit.register(close_all_results_csv_handles)  # Schedule results CSV handle closure at program exit


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

        config = initialize_cli_and_config()  # Parse CLI args, load configuration, and set global CONFIG
        initialize_logger(config)  # Initialize logging system with provided configuration
        setup_telegram_bot(config)  # Initialize global Telegram bot with provided configuration
        start_time = datetime.datetime.now()  # Get the start time of the program
        send_telegram_message(TELEGRAM_BOT, [f"Starting WGAN-GP Data Augmentation at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"])  # Send formatted start notification via Telegram
        
        mode, csv_path, results_suffix, datasets, args = extract_runtime_parameters(config)  # Extract execution mode, paths, and arg namespace from config
        results_cols = validate_results_csv_columns(config)  # Validate and retrieve results CSV column configuration

        if csv_path is not None:  # Single file mode (csv_path provided)
            handle_single_file_mode(args, config, mode, csv_path, results_cols, results_suffix)  # Execute all single-file output setup and mode dispatch steps
        else:  # Batch dataset mode (no csv_path provided)
            run_batch_mode(args, config, datasets, mode, results_suffix, results_cols)  # Iterate and process all configured dataset directory files

        finish_time = datetime.datetime.now()  # Get the finish time of the program
        print_execution_summary(start_time, finish_time)  # Print start time, finish time, and total elapsed duration
        send_telegram_message(TELEGRAM_BOT, [f"WGAN-GP Data Augmentation finished. Execution time: {calculate_execution_time(start_time, finish_time)}"])  # Send execution time summary notification via Telegram
        register_exit_handlers(config)  # Register atexit sound playback and CSV handle cleanup handlers
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


if __name__ == "__main__":
    """
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
