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


def ensure_figure_min_4k_and_save(fig=None, path=None, dpi=None, **kwargs):
    """
    Ensure a Matplotlib figure meets 4k minimum pixel dimensions and save it.

    :param fig: Matplotlib Figure instance or None to use current figure.
    :param path: Path where the image will be saved.
    :param dpi: DPI to use for saving; preserved if provided.
    :return: None
    """

    fig = fig or plt.gcf()  # Use provided figure or get current figure if None
    effective_dpi = dpi if dpi is not None else fig.get_dpi()  # Determine effective DPI for size calculations
    width_inch, height_inch = fig.get_size_inches()  # Get current figure size in inches

    if width_inch * effective_dpi < 3840 or height_inch * effective_dpi < 2160:  # Verify if current size is below 4k dimensions
        new_w = max(width_inch, 3840.0 / effective_dpi)  # Calculate new width in inches to meet 4k width requirement
        new_h = max(height_inch, 2160.0 / effective_dpi)  # Calculate new height in inches to meet 4k height requirement
        fig.set_size_inches(new_w, new_h)  # Update figure size to ensure minimum 4k dimensions

    save_kwargs = dict(kwargs)  # Copy additional save parameters

    if dpi is not None:  # If a specific DPI was provided, ensure it is included in save parameters
        save_kwargs["dpi"] = dpi  # Set DPI for saving

    resolved_fig = fig  # Ensure we close the exact figure used for saving
    if path is None:  # Verify that a valid path argument was provided
        raise ValueError("path must be provided to save the figure")  # Raise explicit error when path is missing to avoid passing None to savefig
    try:  # Save the figure to the specified path with the given parameters
        resolved_fig.savefig(path, **save_kwargs)  # Save the figure to the specified path with the given parameters
    finally:  # Ensure the figure is closed to free resources
        plt.close(resolved_fig)  # Close the figure to free memory


def resolve_plot_save_directory(out_dir: str, config: Dict) -> Path:
    """
    Resolve the save directory for training metrics plots based on configuration.

    :param out_dir: Base output directory string.
    :param config: Configuration dictionary with paths and plotting settings.
    :return: Path object for the resolved plot save directory.
    """

    data_aug_subdir = config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation")  # Get data augmentation subdir from config
    plotting_subdir = config.get("plotting", {}).get("subdir", "plots")  # Get plotting subdir from config
    out_path = Path(out_dir)  # Convert provided out_dir to a Path object for safe operations
    if out_path.name == data_aug_subdir:  # If out_dir already ends with the data_augmentation subdir
        save_dir = out_path / plotting_subdir  # Use out_dir plus plotting subdir (avoid duplicating Data_Augmentation)
    else:  # Otherwise, out_dir does not include data_augmentation yet
        save_dir = out_path / data_aug_subdir / plotting_subdir  # Append Data_Augmentation then plotting subdir
    try:  # Try to create the save directory, but catch exceptions
        save_dir.mkdir(parents=True, exist_ok=True)  # Create save directory with parents if it doesn't exist
    except Exception as e:  # Catch any exception during directory creation
        print(str(e))  # Print directory creation error to terminal for visibility
        try:  # Attempt to notify about directory creation error via Telegram
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full directory creation error via Telegram
        except Exception:  # If notification fails, ignore to avoid cascading errors
            pass  # Ignore Telegram send errors during directory creation fallback
        save_dir = out_path  # Fallback to out_dir if subdirectories cannot be created
    return save_dir  # Return resolved plot save directory path


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
        linewidth = safe_float(config.get("plotting", {}).get("linewidth", 1.5), 1.5)  # Get line width safely from config
        alpha = safe_float(config.get("plotting", {}).get("alpha", 0.7), 0.7)  # Get alpha safely from config
        grid_alpha = safe_float(config.get("plotting", {}).get("grid_alpha", 0.3), 0.3)  # Get grid alpha safely from config
    
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

        save_dir = resolve_plot_save_directory(out_dir, config)  # Resolve save directory for training metrics plots

        plot_path = str(save_dir / filename)  # Construct full path for the plot file
        ensure_figure_min_4k_and_save(fig=plt.gcf(), path=plot_path, dpi=dpi, bbox_inches="tight")  # Save figure ensuring >=4k pixels without changing DPI
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

        existed = results_csv_path.exists()  # Verify whether file exists already
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
        file_size_gb = safe_float(file_size_bytes, 0.0) / (1024.0 ** 3)  # Convert bytes to gigabytes (GB) safely
        return f"{file_progress_prefix} Starting on {file_name} ({num_samples} samples, {file_size_gb:.2f} GB) for {args.epochs} epochs"  # Single formatted f-string as requested
    except Exception as e:
        print(str(e))  # Print exception for visibility
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception information via Telegram
        raise  # Re-raise exception to allow outer handler to manage it


def send_file_saved_and_timing_messages(args: Any, config: Dict) -> None:  # Create helper to send saved-file and timing messages
    """
    Compose and send messages about saved files, sizes, and timing.

    :param args: Runtime arguments namespace containing timing and path attributes.
    :param config: Configuration dictionary for path and formatting values.
    :return: None
    """

    file_progress_prefix = getattr(args, "file_progress_prefix", f"{BackgroundColors.CYAN}[1/1]{Style.RESET_ALL}")  # Build colored prefix from args or default
    gen_file = getattr(args, "out_file", None) or ""  # Retrieve explicit out_file from args if present
    if not gen_file and getattr(args, "csv_path", None):  # Derive augmented path when out_file not provided
        try:  # Attempt derivation of default generated file path from csv_path and config
            csv_obj = Path(args.csv_path)  # Construct Path from provided CSV path
            suffix = config.get("wgangp", {}).get("results_suffix", "_data_augmented")  # Read results suffix from wgangp config
            derived = csv_obj.parent / config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation") / f"{csv_obj.stem}{suffix}.csv"  # Build derived path
            gen_file = str(derived)  # Use derived path string as generated file path
        except Exception:  # If derivation fails, fall back to empty string silently
            gen_file = ""  # Leave generated file string empty on failure

    gen_path = Path(gen_file) if gen_file else None  # Create Path object when gen_file is available
    try:  # Obtain file size bytes if file exists
        size_bytes = gen_path.stat().st_size if gen_path and gen_path.exists() else 0  # Get file size in bytes or zero
    except Exception:  # On error, default to zero bytes
        size_bytes = 0  # Default size when stat fails

    file_exists = gen_path is not None and gen_path.exists() and size_bytes > 0  # Determine whether the generated file exists with content on disk

    if size_bytes >= 1024 ** 3:  # If file is at least 1 GiB
        size_str = f"{size_bytes / (1024 ** 3):.2f} GB"  # Format as gigabytes
    else:  # Otherwise present in megabytes for readability
        size_str = f"{size_bytes / (1024 ** 2):.2f} MB"  # Format as megabytes

    training_time = getattr(args, "_last_training_time", "")  # Read stored training elapsed seconds from args
    file_time = getattr(args, "_last_file_time", "")  # Read stored per-file elapsed seconds from args
    model_save_time = getattr(args, "_last_model_save_time", "")  # Read last model save elapsed seconds from args
    generation_time = getattr(args, "_last_sample_generation_time", "")  # Read sample generation elapsed seconds from args

    if file_exists:  # Only print generated file path and size when the file exists with content
        print(f"{file_progress_prefix} {BackgroundColors.GREEN}Saved generated file to {BackgroundColors.CYAN}{gen_file} {BackgroundColors.GREEN}({size_str}){Style.RESET_ALL}")  # Print saved file path and human-readable size
    print(f"{file_progress_prefix} {BackgroundColors.GREEN}Training time: {BackgroundColors.CYAN}{training_time}s{Style.RESET_ALL}")  # Print total training elapsed seconds
    print(f"{file_progress_prefix} {BackgroundColors.GREEN}File processing time: {BackgroundColors.CYAN}{file_time}s{Style.RESET_ALL}")  # Print per-file processing elapsed seconds
    if model_save_time != "":  # Only print model save time when available
        print(f"{file_progress_prefix} {BackgroundColors.GREEN}Model save time: {BackgroundColors.CYAN}{model_save_time}s{Style.RESET_ALL}")  # Print model save elapsed seconds
    if generation_time != "":  # Only print generation time when available
        print(f"{file_progress_prefix} {BackgroundColors.GREEN}Generation time: {BackgroundColors.CYAN}{generation_time}s{Style.RESET_ALL}")  # Print sample generation elapsed seconds

    try:  # Attempt to notify via Telegram with a compact summary message
        training_time_str = calculate_execution_time(training_time)  # Convert training_time to human-readable duration string
        generation_time_str = calculate_execution_time(generation_time) if generation_time != "" else ""  # Convert generation_time when present otherwise empty
        if file_exists:  # Only include file info in Telegram when the generated file exists with content
            display_path = os.path.relpath(gen_file) if gen_file else "unknown"  # Compute relative path for generated file or 'unknown'
            samples_written = ""  # Initialize samples written placeholder
            try:  # Attempt to count data rows in the generated CSV when file exists
                if gen_file and Path(gen_file).exists():  # Verify generated file path exists before counting
                    with open(gen_file, "r", encoding="utf-8") as _f:  # Open generated file for reading rows
                        _reader = csv.reader(_f)  # Create CSV reader to iterate rows
                        _total_rows = sum(1 for _ in _reader)  # Count total rows including header
                    if _total_rows > 0:  # If at least one row present deduct header to estimate data rows
                        samples_written = str(max(0, _total_rows - 1))  # Compute data rows as total minus header
                    else:  # File exists but has no rows
                        samples_written = "0"  # Zero rows when file empty
            except Exception:  # On counting failure, leave samples_written empty to avoid breaking flow
                samples_written = ""  # Preserve empty when counting fails
            samples_display = samples_written if samples_written != "" else "unknown"  # Derive display value for samples or unknown
            msg = (  # Compose compact message using relative path, sample count and formatted durations
                f"{file_progress_prefix} Saved file {Path(display_path).name if gen_file else 'unknown'} ({size_str}) | Path: {display_path} | "
                f"Samples: {samples_display} | Training: {training_time_str} | Generation: {generation_time_str}"
            )  # End message composition with file info included
        else:  # File does not exist yet (training phase before generation)
            msg = (  # Compose timing-only message without file info for training phase
                f"{file_progress_prefix} Training completed | Training: {training_time_str}"
            )  # End training-only message composition
        send_telegram_message(TELEGRAM_BOT, msg)  # Send composed summary by Telegram using shared helper
    except Exception as e:
        print(str(e))  # Print exception for visibility
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception information via Telegram
        raise  # Re-raise exception to allow outer handler to manage it


def get_available_disk_space_gb(config: Optional[Dict] = None) -> float:
    """
    Return the available disk space in GB for the current working directory.

    Uses psutil.disk_usage when available and falls back to shutil.disk_usage
    when psutil is not installed. Returns a safe fallback of 0.0 on failure.

    :param config: Optional configuration dictionary.
    :return: Available disk space in GB as a float.
    """

    try:
        cwd_str = str(Path(".").resolve())  # Resolve current working directory for disk usage query
        if psutil is not None:  # Prefer psutil for disk usage query when available
            disk_stat = psutil.disk_usage(cwd_str)  # Query disk usage via psutil
            free_bytes = disk_stat.free  # Available disk space in bytes from psutil
        else:  # Fall back to stdlib shutil.disk_usage when psutil is unavailable
            print(f"{BackgroundColors.YELLOW}[WARNING] psutil not available; falling back to shutil.disk_usage for available disk space query{Style.RESET_ALL}")  # Warn about psutil unavailability and fallback
            shutil_stat = shutil.disk_usage(cwd_str)  # Query disk usage via shutil stdlib fallback
            free_bytes = shutil_stat.free  # Available disk space in bytes from shutil
        return safe_float(free_bytes, 0.0) / (1024.0 ** 3)  # Convert bytes to GB and return safely
    except Exception as e:  # On unexpected errors, warn and return safe fallback
        print(f"{BackgroundColors.YELLOW}[WARNING] Failed to query available disk space: {e}; returning 0.0 GB{Style.RESET_ALL}")  # Warn about query failure
        return 0.0  # Return safe fallback of zero GB on failure


def get_used_disk_percentage(config: Optional[Dict] = None) -> float:
    """
    Return the percentage of disk space currently used for the current working directory.

    Uses psutil.disk_usage when available and falls back to shutil.disk_usage
    when psutil is not installed. Returns a safe fallback of 0.0 on failure.

    :param config: Optional configuration dictionary.
    :return: Used disk space percentage as a float.
    """

    try:
        cwd_str = str(Path(".").resolve())  # Resolve current working directory for disk usage query
        if psutil is not None:  # Prefer psutil for disk usage query when available
            disk_stat = psutil.disk_usage(cwd_str)  # Query disk usage via psutil
            used_bytes = disk_stat.used  # Used disk space in bytes from psutil
            total_bytes = disk_stat.total  # Total disk space in bytes from psutil
        else:  # Fall back to stdlib shutil.disk_usage when psutil is unavailable
            print(f"{BackgroundColors.YELLOW}[WARNING] psutil not available; falling back to shutil.disk_usage for used disk percentage query{Style.RESET_ALL}")  # Warn about psutil unavailability and fallback
            shutil_stat = shutil.disk_usage(cwd_str)  # Query disk usage via shutil stdlib fallback
            used_bytes = shutil_stat.used  # Used disk space in bytes from shutil
            total_bytes = shutil_stat.total  # Total disk space in bytes from shutil
        if total_bytes <= 0:  # Guard against division by zero when total is zero or negative
            return 0.0  # Return safe fallback when total disk space is unavailable
        used_percent = (safe_float(used_bytes, 0.0) / safe_float(total_bytes, 1.0)) * 100.0  # Compute used percentage from used and total bytes
        return used_percent  # Return computed usage percentage
    except Exception as e:  # On unexpected errors, warn and return safe fallback
        print(f"{BackgroundColors.YELLOW}[WARNING] Failed to query used disk percentage: {e}; returning 0.0%{Style.RESET_ALL}")  # Warn about query failure
        return 0.0  # Return safe fallback of zero percent on failure


def get_total_disk_space_gb(config: Optional[Dict] = None) -> float:
    """
    Return the total disk space in GB for the current working directory.

    Uses psutil.disk_usage when available and falls back to shutil.disk_usage
    when psutil is not installed. Returns a safe fallback of 0.0 on failure.

    :param config: Optional configuration dictionary.
    :return: Total disk space in GB as a float.
    """

    try:
        cwd_str = str(Path(".").resolve())  # Resolve current working directory for disk usage query
        if psutil is not None:  # Prefer psutil for disk usage query when available
            disk_stat = psutil.disk_usage(cwd_str)  # Query disk usage via psutil
            total_bytes = disk_stat.total  # Total disk space in bytes from psutil
        else:  # Fall back to stdlib shutil.disk_usage when psutil is unavailable
            print(f"{BackgroundColors.YELLOW}[WARNING] psutil not available; falling back to shutil.disk_usage for total disk space query{Style.RESET_ALL}")  # Warn about psutil unavailability and fallback
            shutil_stat = shutil.disk_usage(cwd_str)  # Query disk usage via shutil stdlib fallback
            total_bytes = shutil_stat.total  # Total disk space in bytes from shutil
        return safe_float(total_bytes, 0.0) / (1024.0 ** 3)  # Convert bytes to GB and return safely
    except Exception as e:  # On unexpected errors, warn and return safe fallback
        print(f"{BackgroundColors.YELLOW}[WARNING] Failed to query total disk space: {e}; returning 0.0 GB{Style.RESET_ALL}")  # Warn about query failure
        return 0.0  # Return safe fallback of zero GB on failure


def adjust_num_workers_for_file(csv_path: str, suggested_workers: int, config: Optional[Dict] = None) -> int:
    """
    Adjust DataLoader `num_workers` based on CSV file size and total FREE system RAM.

    New rule:
        num_workers = (file_size_gb * 3) / free_ram_gb

    :param csv_path: Path to the CSV file to inspect
    :param suggested_workers: Initial suggested num_workers value from config
    :param config: Optional configuration dictionary
    :return: Adjusted num_workers integer
    """

    try:  # Guard helper with try/except to follow project style
        if config is None: config = CONFIG or get_default_config()  # Use provided config or fallback to global/default

        file_size_bytes = Path(csv_path).stat().st_size  # Get file size in bytes from filesystem
        file_size_gb = safe_float(file_size_bytes, 0.0) / (1024.0 ** 3)  # Convert bytes to gigabytes (GB) safely

        if psutil is None:  # If psutil is unavailable
            print(f"{BackgroundColors.YELLOW}Warning: psutil not available, cannot detect free RAM; using suggested num_workers: {suggested_workers}{Style.RESET_ALL}")  # Warn about fallback to suggested_workers
            final = max(0, int(suggested_workers))  # Fallback to suggested
            try:
                send_telegram_message(TELEGRAM_BOT, f"[INFO] num_workers for {Path(csv_path).name}: {final} (psutil unavailable)")  # Notify via Telegram
            except Exception:
                pass  # Do not break execution if Telegram fails
            return final  # Return suggested_workers when RAM detection is impossible

        vm = psutil.virtual_memory()  # Capture virtual memory snapshot
        free_ram_gb = safe_float(vm.available, 0.0) / (1024.0 ** 3)  # Use AVAILABLE RAM (not total) in GB

        print(f"{BackgroundColors.GREEN}Detected file size: {BackgroundColors.CYAN}{file_size_gb:.4f} GB{Style.RESET_ALL}")  # Log detected file size
        print(f"{BackgroundColors.GREEN}Detected free RAM: {BackgroundColors.CYAN}{free_ram_gb:.4f} GB{Style.RESET_ALL}")  # Log detected available RAM
        print(f"{BackgroundColors.GREEN}Original suggested num_workers: {BackgroundColors.CYAN}{suggested_workers}{Style.RESET_ALL}")  # Log original suggestion

        computed = None
        if file_size_gb <= 0.0:  # If file size cannot be determined or is zero
            print("[WARNING] File size detected as 0 GB; keeping suggested num_workers")
            final = max(0, int(suggested_workers))
        else:  # Otherwise, compute num_workers based on file size and free RAM
            computed = (free_ram_gb * 3.0) / file_size_gb  # Updated formula: free_ram numerator
            try:  # Attempt to convert computed value to float for proper scaling
                computed_val = float(computed)
            except Exception:
                computed_val = float(int(suggested_workers))
            computed = computed_val
            final = int(max(0, computed_val))  # Ensure non-negative integer

        cpu_count = os.cpu_count() or 1  # Detect CPU count safely
        final = min(final, cpu_count)  # Do not exceed logical CPUs

        print(f"{BackgroundColors.GREEN}Computed num_workers based on formula: {BackgroundColors.CYAN}{computed:.4f}{Style.RESET_ALL}")  # Log computed value before final adjustment

        available_disk_gb = get_available_disk_space_gb()  # Retrieve available disk space in GB for Telegram notification
        total_disk_gb = get_total_disk_space_gb()  # Retrieve total disk space in GB for Telegram notification
        used_disk_percent = get_used_disk_percentage()  # Retrieve used disk percentage for Telegram notification
        total_cores = os.cpu_count() or 1  # Detect total logical CPU cores with safe fallback for reporting
        try:  # Notify via Telegram (non-blocking)
            send_telegram_message(
                TELEGRAM_BOT,
                f"[INFO] num_workers adjusted for {Path(csv_path).name} | "
                f"file={file_size_gb:.2f}GB | free_ram={free_ram_gb:.2f}GB | "
                f"storage={available_disk_gb:.2f}/{total_disk_gb:.2f}GB ({used_disk_percent:.2f}%) | "
                f"final_workers={final} (total cores: {total_cores})",
            )
        except Exception:
            pass  # Never allow Telegram failure to affect training

        return final  # Return computed value

    except Exception as e:  # On error, report and re-raise following project conventions
        print(str(e))  # Print exception for visibility
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception via Telegram
        raise  # Re-raise to allow outer handler to manage it


def compute_epoch_milestones(total_epochs: int) -> List[int]:
    """
    Compute milestone epoch indices for progress-based CSV writes.

    Always includes epoch 1 and epoch `total_epochs`.  Additionally includes
    the 25%, 50% and 75% epochs computed with ceiling rounding and deduped.

    :param total_epochs: Total number of training epochs
    :return: Sorted list of unique integer epoch indices (1-based)
    """

    try:  # Preserve repository try/except pattern for robustness
        total = int(total_epochs)  # Cast provided total to integer type
        if total < 1:  # If no positive epochs requested
            return []  # Return empty list when there are no valid epochs
        
        milestones = set()  # Create a set to deduplicate milestone indices efficiently
        milestones.add(1)  # Ensure the first epoch is always present
        milestones.add(total)  # Ensure the final epoch is always present
        m25 = (total + 3) // 4  # Compute ceil(total * 1/4) using integer arithmetic
        m50 = (total + 1) // 2  # Compute ceil(total * 1/2) using integer arithmetic
        m75 = (3 * total + 3) // 4  # Compute ceil(total * 3/4) using integer arithmetic
        
        for m in (m25, m50, m75):  # Iterate candidate fractional milestones
            if 1 <= m <= total:  # Only add candidates within valid epoch bounds
                milestones.add(int(m))  # Add integer milestone to set to dedupe
        
        return sorted(milestones)  # Return a sorted list for predictable ordering
    except Exception as e:  # Follow repo convention: report then re-raise
        print(str(e))  # Print exception for visibility
        
        try:  # Attempt to notify via Telegram when available
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception via Telegram
        except Exception:
            pass  # Ignore notification errors to avoid cascading failures
        raise  # Re-raise exception to be handled by outer logic


def is_checkpoint_space_available(dataset_dirs: List[str], config: Optional[Dict] = None) -> bool:
    """
    Determine whether sufficient disk space exists before saving checkpoints.

    Recursively computes the total size of all dataset directories and compares it
    against available free disk space. Checkpoint saving is permitted only when
    available free space is at least twice the total dataset size.

    :param dataset_dirs: List of dataset directory paths to measure total size from.
    :param config: Optional configuration dictionary.
    :return: True if available disk space is at least twice the dataset size, False otherwise.
    """

    try:
        total_dataset_size = 0  # Initialize total dataset size accumulator in bytes
        for dir_path in dataset_dirs:  # Iterate all provided dataset directory paths
            try:  # Guard per-directory size calculation to avoid aborting on individual failures
                p = Path(dir_path)  # Create Path object for the directory
                if not p.exists():  # Verify directory exists before recursing
                    continue  # Skip non-existent directories silently
                for file_entry in p.rglob("*"):  # Recursively walk all entries in the directory
                    try:  # Guard per-file stat to avoid crashing on permission errors
                        if file_entry.is_file():  # Only accumulate size for regular files
                            total_dataset_size += file_entry.stat().st_size  # Add file size in bytes to accumulator
                    except Exception:  # If stat fails due to permission error or similar issue
                        pass  # Continue without accumulating this file's size
            except Exception:  # If directory traversal fails unexpectedly
                pass  # Continue with remaining directories
        free_gb = get_available_disk_space_gb(config)  # Retrieve available disk space in GB via helper
        free_bytes = int(safe_float(free_gb, 0.0) * (1024.0 ** 3))  # Convert GB back to bytes for threshold comparison
        required_bytes = total_dataset_size * 2  # Minimum required free space is twice the total dataset size
        dataset_size_gb = safe_float(total_dataset_size, 0.0) / (1024.0 ** 3)  # Convert dataset size to GB for logging
        required_gb = safe_float(required_bytes, 0.0) / (1024.0 ** 3)  # Convert required space to GB for logging
        print(f"{BackgroundColors.GREEN}Disk space verification: dataset_size={BackgroundColors.CYAN}{dataset_size_gb:.2f} GB{BackgroundColors.GREEN}, free={BackgroundColors.CYAN}{free_gb:.2f} GB{BackgroundColors.GREEN}, required={BackgroundColors.CYAN}{required_gb:.2f} GB{Style.RESET_ALL}")  # Print disk space verification summary
        if free_bytes < required_bytes:  # If free space is below the required threshold
            print(f"{BackgroundColors.YELLOW}[WARNING] Insufficient disk space for checkpoint saving. Skipping checkpoint creation.{Style.RESET_ALL}")  # Warn about insufficient space
            return False  # Deny checkpoint saving when space is insufficient
        return True  # Permit checkpoint saving when sufficient space is available
    except Exception as e:  # On unexpected errors, warn but allow checkpoint saving
        print(f"{BackgroundColors.YELLOW}[WARNING] Disk space verification failed: {e}; assuming checkpoint space is available{Style.RESET_ALL}")  # Warn about verification failure
        return True  # Default to allowing checkpoint saving when verification fails


def normalize_args_and_setup_hardware(args, config: Dict) -> tuple:
    """
    Normalize argument types and configure hardware settings for training.

    :param args: parsed arguments namespace to normalize in-place
    :param config: configuration dictionary with training settings
    :return: Tuple of (device, training_start_time, file_start_time, epoch_milestones, file_progress_prefix)
    """

    args.lr = safe_float(getattr(args, "lr", None), config.get("training", {}).get("lr", 1e-4))  # Ensure learning rate is float safely
    args.beta1 = safe_float(getattr(args, "beta1", None), config.get("training", {}).get("beta1", 0.5))  # Ensure beta1 is float safely
    args.beta2 = safe_float(getattr(args, "beta2", None), config.get("training", {}).get("beta2", 0.9))  # Ensure beta2 is float safely
    args.lambda_gp = safe_float(getattr(args, "lambda_gp", None), config.get("training", {}).get("lambda_gp", 10.0))  # Ensure lambda_gp is float safely
    args.n_samples = safe_float(getattr(args, "n_samples", None), config.get("generation", {}).get("n_samples", 1.0))  # Ensure n_samples is float safely
    args.seed = int(args.seed)  # Ensure seed is int
    args.epochs = int(args.epochs)  # Ensure epochs is int
    epoch_milestones = compute_epoch_milestones(int(args.epochs))  # Precompute milestone epochs after epochs finalized
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
        
    batch_multiplier = min(8, max(1, 2 * gpu_count)) if gpu_count > 0 else 1  # Scale by 2x per GPU but cap to 8x to avoid OOM
    scaled_batch = int(args.batch_size) * batch_multiplier  # Compute scaled batch size
    args.batch_size = int(scaled_batch)  # Apply scaled batch size to args
    args.use_amp = bool(args.use_amp or (gpu_count > 0 and _torch_autocast is not None))  # Enable AMP when CUDA and autocast available
    suggested_workers = min(max(1, (os.cpu_count() or 1) // 2), 32)  # Suggest a conservative default for num_workers
    file_progress_prefix = getattr(args, "file_progress_prefix", f"{BackgroundColors.CYAN}[1/1]{Style.RESET_ALL}")  # Build colored prefix (default single-file)
    
    print(f"{BackgroundColors.GREEN}Detected {gpu_count} GPUs.{Style.RESET_ALL}")  # Print GPU count
    print(f"{BackgroundColors.GREEN}Using DataParallel: {use_dataparallel}{Style.RESET_ALL}")  # Print whether DataParallel will be used
    print(f"{BackgroundColors.GREEN}Batch size: {BackgroundColors.CYAN}{args.batch_size}{Style.RESET_ALL}")  # Print effective batch size after scaling
    print(f"{BackgroundColors.GREEN}Suggested num_workers: {BackgroundColors.CYAN}{suggested_workers}{Style.RESET_ALL}")  # Print suggested workers value
    print(f"{BackgroundColors.GREEN}AMP enabled: {BackgroundColors.CYAN}{args.use_amp}{Style.RESET_ALL}")  # Print AMP usage
    print(f"{BackgroundColors.GREEN}cuDNN benchmark: {BackgroundColors.CYAN}{torch.backends.cudnn.benchmark}{Style.RESET_ALL}")  # Print cuDNN benchmark status
    send_telegram_message(TELEGRAM_BOT, compose_training_start_message(args, file_progress_prefix))  # Telegram start with colored prefix and file statistics

    print(f"{BackgroundColors.GREEN}Device: {BackgroundColors.CYAN}{device.type.upper()}{Style.RESET_ALL}")  # Print device type
    if args.use_amp and device.type == "cuda":  # If AMP enabled and using CUDA
        print(f"{BackgroundColors.GREEN}Using Automatic Mixed Precision (AMP) for faster training{Style.RESET_ALL}")  # Print AMP detail
    if args.compile:  # If torch.compile requested
        print(f"{BackgroundColors.GREEN}Using torch.compile() for optimized execution{Style.RESET_ALL}")  # Print compile detail

    return device, training_start_time, file_start_time, epoch_milestones, file_progress_prefix  # Return hardware setup results


def create_dataset_and_dataloader(args, config: Dict, device: torch.device) -> tuple:
    """
    Create the CSV flow dataset and configure the training data loader.

    :param args: parsed arguments namespace with csv_path, feature_cols, label_col, and batch_size
    :param config: configuration dictionary with dataloader settings
    :param device: torch device for pin_memory optimization
    :return: Tuple of (dataset, dataloader)
    """

    dataset = CSVFlowDataset(
        args.csv_path, label_col=args.label_col, feature_cols=args.feature_cols
    )  # Load dataset from CSV

    num_workers = int(config.get("dataloader", {}).get("num_workers", 8))  # Get base num_workers from config and cast to int
    num_workers = adjust_num_workers_for_file(args.csv_path, num_workers, config)  # Adjust num_workers based on file size and system RAM
    
    if device.type == "cuda" and num_workers == 0:  # If CUDA available but adjusted workers is 0
        try:  # Attempt to fetch total RAM to decide whether to raise workers for CUDA
            total_ram_gb = (safe_float(psutil.virtual_memory().total, 0.0) / (1024.0 ** 3)) if psutil is not None else None  # Detect total RAM if psutil available safely
        except Exception:  # If detection fails
            total_ram_gb = None  # Unknown RAM when detection fails
        if total_ram_gb is None or total_ram_gb >= 8.0:  # If RAM unknown or sufficient
            num_workers = max(1, (os.cpu_count() or 1))  # Ensure at least one worker for CUDA when RAM allows
        else:  # Low RAM and CUDA present: keep zero workers to avoid memory pressure
            print(f"{BackgroundColors.YELLOW}Warning: num_workers set to 0 due to low RAM and CUDA presence{Style.RESET_ALL}")  # Warn about zero workers for CUDA
    
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

    return dataset, dataloader  # Return dataset and configured dataloader


def init_results_csv_and_feature_dims(args, config: Dict, dataset) -> tuple:
    """
    Initialize the results CSV writer and extract feature dimensions from the dataset.

    :param args: parsed arguments namespace with csv_path
    :param config: configuration dictionary with wgangp results_csv_columns and paths
    :param dataset: loaded CSVFlowDataset instance
    :return: Tuple of (results_csv_file, results_csv_writer, results_cols_cfg, feature_dim, n_classes)
    """

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

    return results_csv_file, results_csv_writer, results_cols_cfg, feature_dim, n_classes  # Return CSV handles and feature dimensions


def create_models_and_optimizers(args, config: Dict, device: torch.device, feature_dim: int, n_classes: int) -> tuple:
    """
    Create generator and discriminator models, optimizers, and initialize training state.

    :param args: parsed arguments namespace with model architecture and optimizer settings
    :param config: configuration dictionary with generator and discriminator settings
    :param device: torch device for model placement
    :param feature_dim: dimensionality of the output features
    :param n_classes: number of label classes for conditional generation
    :return: Tuple of (G, D, opt_D, opt_G, scaler, fixed_noise, fixed_labels, step, start_epoch, metrics_history)
    """

    g_leaky_relu_alpha = safe_float(config.get("generator", {}).get("leaky_relu_alpha", 0.2), 0.2)  # Get generator LeakyReLU alpha safely from config
    d_leaky_relu_alpha = safe_float(config.get("discriminator", {}).get("leaky_relu_alpha", 0.2), 0.2)  # Get discriminator LeakyReLU alpha safely from config

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

    if torch.cuda.is_available() and not args.force_cpu and torch.cuda.device_count() > 1:  # If multi-GPU condition met
        G = torch.nn.DataParallel(G)  # Wrap generator in DataParallel to utilize multiple GPUs
        D = torch.nn.DataParallel(D)  # Wrap discriminator in DataParallel to utilize multiple GPUs
        print(f"{BackgroundColors.GREEN}Wrapped models using DataParallel across {torch.cuda.device_count()} GPUs{Style.RESET_ALL}")  # Notify wrapping

    if args.compile and not isinstance(G, torch.nn.DataParallel):  # Only compile when not using DataParallel
        try:  # Try compiling models, but catch exceptions if torch.compile() is not available or fails
            G = torch.compile(G, mode="reduce-overhead")  # Compile generator for performance
            D = torch.compile(D, mode="reduce-overhead")  # Compile discriminator for performance
            print(f"{BackgroundColors.GREEN}Models compiled successfully{Style.RESET_ALL}")  # Notify successful compilation
        except Exception as e:  # Catch any exception during compilation
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

    return G, D, opt_D, opt_G, scaler, fixed_noise, fixed_labels, step, start_epoch, metrics_history  # Return models and training state


def load_and_restore_generator_state(g_checkpoint_path: Path, device: torch.device, G, opt_G, scaler) -> tuple:
    """
    Load generator checkpoint and restore model weights, optimizer, and scaler state.

    :param g_checkpoint_path: Path to the generator checkpoint file.
    :param device: Torch device for checkpoint loading via map_location.
    :param G: Generator model to restore weights into.
    :param opt_G: Generator optimizer to restore state into.
    :param scaler: AMP gradient scaler to restore state into (may be None).
    :return: Tuple of (g_checkpoint_dict, start_epoch).
    """

    print(f"{BackgroundColors.GREEN}Loading generator checkpoint: {g_checkpoint_path.name}{Style.RESET_ALL}")  # Print loading message
    g_checkpoint = torch.load(g_checkpoint_path, map_location=device, weights_only=False)  # Load generator checkpoint with sklearn objects
    if isinstance(cast(Any, G), torch.nn.DataParallel):  # If model wrapped by DataParallel
        cast(Any, G).module.load_state_dict(g_checkpoint["state_dict"])  # Restore generator weights into module
    else:  # Not DataParallel
        cast(Any, G).load_state_dict(g_checkpoint["state_dict"])  # Restore generator weights
    start_epoch = g_checkpoint["epoch"]  # Set starting epoch
    if "opt_G_state" in g_checkpoint:  # If optimizer state saved
        opt_G.load_state_dict(g_checkpoint["opt_G_state"])  # Restore generator optimizer
        print(f"{BackgroundColors.GREEN}✓ Restored generator optimizer state{Style.RESET_ALL}")  # Confirm optimizer restoration
    if scaler is not None and "scaler_state" in g_checkpoint:  # If using AMP and scaler state saved
        scaler.load_state_dict(g_checkpoint["scaler_state"])  # Restore scaler state
        print(f"{BackgroundColors.GREEN}✓ Restored AMP scaler state{Style.RESET_ALL}")  # Confirm scaler restoration
    return g_checkpoint, start_epoch  # Return loaded checkpoint dict and starting epoch


def restore_metrics_from_checkpoint_or_json(g_checkpoint: Dict, checkpoint_dir: Path, checkpoint_prefix: str, metrics_history: Dict, step: int) -> tuple:
    """
    Restore metrics history from checkpoint dict or fallback JSON file.

    :param g_checkpoint: Loaded generator checkpoint dictionary potentially containing metrics_history.
    :param checkpoint_dir: Directory containing checkpoint and metrics files.
    :param checkpoint_prefix: Filename prefix used for the metrics JSON file.
    :param metrics_history: Current metrics_history dict to potentially replace.
    :param step: Current global step counter to potentially update.
    :return: Tuple of (metrics_history, step, metrics_loaded).
    """

    metrics_loaded = False  # Flag to track if metrics were loaded
    if "metrics_history" in g_checkpoint:  # If metrics history saved in checkpoint
        metrics_history = g_checkpoint["metrics_history"]  # Restore metrics from checkpoint
        step = metrics_history["steps"][-1] if metrics_history["steps"] else 0  # Restore step counter
        metrics_loaded = True  # Mark as loaded
        print(f"{BackgroundColors.GREEN}✓ Restored metrics history from checkpoint ({len(metrics_history['steps'])} steps){Style.RESET_ALL}")  # Confirm metrics restoration
    else:  # Try loading from separate JSON file
        metrics_json_path = checkpoint_dir / f"{checkpoint_prefix}_metrics_history.json"  # Path to metrics JSON
        if metrics_json_path.exists():  # If JSON file exists
            try:  # Try to load metrics
                with open(metrics_json_path, "r") as f:  # Open file for reading
                    metrics_history = json.load(f)  # Load metrics from JSON
                step = metrics_history["steps"][-1] if metrics_history["steps"] else 0  # Restore step counter
                metrics_loaded = True  # Mark as loaded
                print(f"{BackgroundColors.GREEN}✓ Restored metrics history from JSON file ({len(metrics_history['steps'])} steps){Style.RESET_ALL}")  # Confirm JSON metrics restoration
            except Exception as e:  # If loading fails
                print(f"{BackgroundColors.YELLOW}⚠ Warning: Failed to load metrics from JSON: {e}{Style.RESET_ALL}")  # Warn about JSON load failure
    return metrics_history, step, metrics_loaded  # Return updated metrics history, step counter, and loaded flag


def load_and_restore_discriminator_state(d_checkpoint_path: Path, device: torch.device, D, opt_D) -> None:
    """
    Load discriminator checkpoint and restore model weights and optimizer state.

    :param d_checkpoint_path: Path to the discriminator checkpoint file.
    :param device: Torch device for checkpoint loading via map_location.
    :param D: Discriminator model to restore weights into.
    :param opt_D: Discriminator optimizer to restore state into.
    :return: None
    """

    if d_checkpoint_path.exists():  # If discriminator checkpoint exists
        print(f"{BackgroundColors.GREEN}Loading discriminator checkpoint: {d_checkpoint_path.name}{Style.RESET_ALL}")  # Print loading message
        d_checkpoint = torch.load(d_checkpoint_path, map_location=device, weights_only=False)  # Load discriminator checkpoint
        if isinstance(cast(Any, D), torch.nn.DataParallel):  # If discriminator wrapped by DataParallel
            cast(Any, D).module.load_state_dict(d_checkpoint["state_dict"])  # Restore discriminator weights into module
        else:  # Not DataParallel
            cast(Any, D).load_state_dict(d_checkpoint["state_dict"])  # Restore discriminator weights
        if "opt_D_state" in d_checkpoint:  # If optimizer state saved
            opt_D.load_state_dict(d_checkpoint["opt_D_state"])  # Restore discriminator optimizer
            print(f"{BackgroundColors.GREEN}✓ Restored discriminator optimizer state{Style.RESET_ALL}")  # Confirm optimizer restoration
    else:  # Discriminator checkpoint not found
        print(f"{BackgroundColors.YELLOW}⚠ Warning: Discriminator checkpoint not found{Style.RESET_ALL}")  # Warn about missing discriminator


def regenerate_missing_training_plot(csv_path_obj: Path, metrics_loaded: bool, metrics_history: Dict) -> None:
    """
    Regenerate training metrics plot from metrics history when the plot file is missing.

    :param csv_path_obj: Path object for the input CSV file used to derive plot filename.
    :param metrics_loaded: Whether metrics history was successfully loaded from checkpoint.
    :param metrics_history: Dictionary of tracked training metrics for plot generation.
    :return: None
    """

    plot_dir = csv_path_obj.parent / "Data_Augmentation"  # Plot directory
    plot_filename = csv_path_obj.stem + "_training_metrics.png"  # Plot filename
    plot_path = plot_dir / plot_filename  # Full plot path
    if not plot_path.exists():  # If plot doesn't exist
        if metrics_loaded and len(metrics_history.get("steps", [])) > 0:  # If metrics available
            print(f"{BackgroundColors.YELLOW}Training metrics plot not found, generating from metrics history...{Style.RESET_ALL}")  # Notify plot generation
            os.makedirs(plot_dir, exist_ok=True)  # Ensure directory exists
            plot_training_metrics(metrics_history, str(plot_dir), plot_filename)  # Generate plot
            print(f"{BackgroundColors.GREEN}✓ Generated training metrics plot: {plot_filename}{Style.RESET_ALL}")  # Confirm plot generation
        else:  # No metrics available
            print(f"{BackgroundColors.YELLOW}⚠ Warning: Training metrics plot not found and no metrics history available to generate it{Style.RESET_ALL}")  # Warn about missing plot and metrics
    else:  # Plot already exists
        print(f"{BackgroundColors.GREEN}✓ Training metrics plot already exists{Style.RESET_ALL}")  # Confirm existing plot


def resume_from_checkpoint(args, config: Dict, device: torch.device, G, D, opt_G, opt_D, scaler, metrics_history: Dict, start_epoch: int, step: int) -> tuple:
    """
    Attempt to resume training from the latest checkpoint if available.

    :param args: parsed arguments namespace with csv_path and from_scratch flag
    :param config: configuration dictionary
    :param device: torch device for checkpoint loading
    :param G: generator model to restore weights into
    :param D: discriminator model to restore weights into
    :param opt_G: generator optimizer to restore state into
    :param opt_D: discriminator optimizer to restore state into
    :param scaler: AMP gradient scaler to restore state into (may be None)
    :param metrics_history: metrics dictionary to potentially replace from checkpoint
    :param start_epoch: current starting epoch to potentially update
    :param step: current global step counter to potentially update
    :return: Tuple of (metrics_history, start_epoch, step)
    """

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

                print(f"{BackgroundColors.CYAN}Found existing checkpoints for {csv_path_obj.name}{Style.RESET_ALL}")  # Notify checkpoint discovery
                print(f"{BackgroundColors.CYAN}Attempting to resume from epoch {epoch_num}...{Style.RESET_ALL}")  # Notify resume attempt

                if g_checkpoint_path.exists():  # If generator checkpoint exists
                    try:  # Try to load checkpoint
                        g_checkpoint, start_epoch = load_and_restore_generator_state(g_checkpoint_path, device, G, opt_G, scaler)  # Load and restore generator state from checkpoint
                        metrics_history, step, metrics_loaded = restore_metrics_from_checkpoint_or_json(g_checkpoint, checkpoint_dir, checkpoint_prefix, metrics_history, step)  # Restore metrics from checkpoint or JSON fallback
                        load_and_restore_discriminator_state(d_checkpoint_path, device, D, opt_D)  # Load and restore discriminator state from checkpoint
                        regenerate_missing_training_plot(csv_path_obj, metrics_loaded, metrics_history)  # Regenerate training plot if missing
                        print(f"{BackgroundColors.GREEN}✓ Resuming training from epoch {start_epoch} (step {step}){Style.RESET_ALL}")  # Confirm resume point
                    except Exception as e:  # If loading fails
                        print(f"{BackgroundColors.YELLOW}⚠ Failed to load checkpoint: {e}{Style.RESET_ALL}")  # Warn about load failure
                        print(f"{BackgroundColors.YELLOW}⚠ Starting training from scratch{Style.RESET_ALL}")  # Notify scratch start
                        start_epoch = 0  # Reset to start from beginning
                        step = 0  # Reset step counter
            else:  # No checkpoints found for this file
                print(f"{BackgroundColors.CYAN}No existing checkpoints found for {csv_path_obj.name}{Style.RESET_ALL}")  # Notify no checkpoints
                print(f"{BackgroundColors.CYAN}Starting training from scratch{Style.RESET_ALL}")  # Notify scratch start
        else:  # Checkpoint directory doesn't exist
            print(f"{BackgroundColors.CYAN}No checkpoint directory found{Style.RESET_ALL}")  # Notify no directory
            print(f"{BackgroundColors.CYAN}Starting training from scratch{Style.RESET_ALL}")  # Notify scratch start
    elif args.from_scratch:  # If user explicitly requested from scratch
        print(f"{BackgroundColors.CYAN}--from_scratch flag set, ignoring existing checkpoints{Style.RESET_ALL}")  # Notify from_scratch flag
        print(f"{BackgroundColors.CYAN}Starting training from scratch{Style.RESET_ALL}")  # Notify scratch start

    return metrics_history, start_epoch, step  # Return potentially updated training state


def init_telegram_progress(args, config: Dict, start_epoch: int) -> tuple:
    """
    Initialize telegram notification progress tracking for epoch-based notifications.

    :param args: parsed arguments namespace with epochs count
    :param config: configuration dictionary with telegram settings
    :param start_epoch: starting epoch for progress calculation when resuming
    :return: Tuple of (telegram_enabled, next_notify, progress_pct)
    """

    telegram_cfg = config.get("telegram", {}) if isinstance(config, dict) else {}  # Retrieve telegram config dict from merged config
    telegram_enabled = bool(telegram_cfg.get("enabled", True))  # Whether telegram notifications are enabled
    try:  # Parse progress_pct from config, ensuring valid integer percentage between 1 and 100
        progress_pct = int(telegram_cfg.get("progress_pct", 10) or 10)  # Percentage step for notifications (default 10)
    except Exception:  # Catch any exception during parsing
        progress_pct = 10  # Fallback to 10% on parse error
    if progress_pct <= 0 or progress_pct > 100:  # If percentage is invalid
        progress_pct = 10  # Sanitize invalid percentage values

    next_notify = progress_pct  # Initialize next notification threshold
    try:  # Calculate percent complete at resume point
        percent_done = int((start_epoch / float(max(1, args.epochs))) * 100)  # Percent complete at resume point
    except Exception:  # If calculation fails
        percent_done = 0  # Assume 0% done on failure
    while percent_done >= next_notify and next_notify <= 100:  # Advance next_notify past already-completed thresholds
        next_notify += progress_pct  # Skip thresholds already passed when resuming

    return telegram_enabled, next_notify, progress_pct  # Return telegram progress state


def create_epoch_progress_bar(dataloader, args, epoch: int) -> tuple:
    """
    Create a tqdm progress bar for the current training epoch.

    :param dataloader: training data loader to iterate over
    :param args: parsed arguments namespace with epochs count
    :param epoch: current epoch index (zero-based)
    :return: Tuple of (pbar, total_steps)
    """

    pbar = tqdm(
        dataloader,
        desc=f"{BackgroundColors.CYAN}Epoch {epoch+1}/{args.epochs}{Style.RESET_ALL}",
        unit="batch",
        file=sys.stdout,  # Use stdout before Logger redirection
        ncols=None,  # Auto-detect terminal width
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"  # Custom format
    )  # Create tqdm progress bar for epoch batches
    total_steps = len(dataloader)  # Number of batches per epoch (explicit total for logging)

    return pbar, total_steps  # Return progress bar and batch count


def execute_discriminator_training_steps(G, D, opt_D, scaler, real_x, labels, device: torch.device, args, config: Dict, n_classes: int) -> tuple:
    """
    Execute multiple discriminator training steps with gradient penalty.

    :param G: Generator model for producing fake samples.
    :param D: Discriminator model to train.
    :param opt_D: Discriminator optimizer.
    :param scaler: AMP gradient scaler (may be None).
    :param real_x: Tensor of real features on device.
    :param labels: Tensor of integer labels on device.
    :param device: Torch device for tensor allocation.
    :param args: Parsed arguments namespace with critic_steps, batch_size, latent_dim, lambda_gp.
    :param config: Configuration dictionary for gradient penalty computation.
    :param n_classes: Number of label classes for conditional generation.
    :return: Tuple of (loss_D, gp, d_real_score, d_fake_score) as tensors.
    """

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
        opt_D.zero_grad(set_to_none=True)  # Reset discriminator gradients to None for reduced memory overhead
        if scaler is not None:  # If using mixed precision
            scaler.scale(loss_D).backward()  # Scale loss and backpropagate
            scaler.step(opt_D)  # Update discriminator parameters with scaled gradients
            scaler.update()  # Update scaler for next iteration
        else:  # Standard precision
            loss_D.backward()  # Backpropagate discriminator loss
            opt_D.step()  # Update discriminator parameters
        d_real_score = d_real.mean()  # Store average real score
        d_fake_score = d_fake.mean()  # Store average fake score
    return loss_D, gp, d_real_score, d_fake_score  # Return discriminator training results


def execute_generator_training_step(G, D, opt_G, scaler, device: torch.device, args, n_classes: int):
    """
    Execute a single generator training step.

    :param G: Generator model to train.
    :param D: Discriminator model for scoring fake samples.
    :param opt_G: Generator optimizer.
    :param scaler: AMP gradient scaler (may be None).
    :param device: Torch device for tensor allocation.
    :param args: Parsed arguments namespace with batch_size and latent_dim.
    :param n_classes: Number of label classes for conditional generation.
    :return: Generator loss tensor.
    """

    with autocast(device.type, enabled=(scaler is not None)):  # Enable AMP if available
        z = torch.randn(args.batch_size, args.latent_dim, device=device)  # Sample noise for generator step
        gen_labels = torch.randint(0, n_classes, (args.batch_size,), device=device)  # Sample labels for generator
        fake_x = G(z, gen_labels)  # Generate fake samples with generator
        g_loss = -D(fake_x, gen_labels).mean()  # Calculate generator loss
    opt_G.zero_grad(set_to_none=True)  # Reset generator gradients to None for reduced memory overhead
    if scaler is not None:  # If using mixed precision
        scaler.scale(g_loss).backward()  # Scale loss and backpropagate
        scaler.step(opt_G)  # Update generator parameters with scaled gradients
        scaler.update()  # Update scaler for next iteration
    else:  # Standard precision
        g_loss.backward()  # Backpropagate generator loss
        opt_G.step()  # Update generator parameters
    return g_loss  # Return generator loss tensor


def record_training_step_metrics(step: int, args, metrics_history: Dict, loss_D, g_loss, gp, d_real_score, d_fake_score) -> tuple:
    """
    Extract scalar metrics at log intervals and append to metrics history.

    :param step: Current global step counter.
    :param args: Parsed arguments namespace with log_interval setting.
    :param metrics_history: Dictionary of tracked training metrics to append to in-place.
    :param loss_D: Discriminator loss tensor.
    :param g_loss: Generator loss tensor.
    :param gp: Gradient penalty tensor.
    :param d_real_score: Average real score tensor.
    :param d_fake_score: Average fake score tensor.
    :return: Tuple of cached scalar values (cached_loss_D, cached_g_loss, cached_gp, cached_d_real, cached_d_fake).
    """

    cached_loss_D = loss_D.item()  # Cache discriminator loss as Python float
    cached_g_loss = g_loss.item()  # Cache generator loss as Python float
    cached_gp = gp.item()  # Cache gradient penalty as Python float
    cached_d_real = d_real_score.item()  # Cache average real score as Python float
    cached_d_fake = d_fake_score.item()  # Cache average fake score as Python float
    wasserstein_dist = cached_d_real - cached_d_fake  # Compute Wasserstein distance from cached values
    
    metrics_history["steps"].append(step)  # Record step number
    metrics_history["loss_D"].append(cached_loss_D)  # Record discriminator loss
    metrics_history["loss_G"].append(cached_g_loss)  # Record generator loss
    metrics_history["gp"].append(cached_gp)  # Record gradient penalty
    metrics_history["D_real"].append(cached_d_real)  # Record real score
    metrics_history["D_fake"].append(cached_d_fake)  # Record fake score
    metrics_history["wasserstein"].append(wasserstein_dist)  # Record Wasserstein distance
    
    return cached_loss_D, cached_g_loss, cached_gp, cached_d_real, cached_d_fake  # Return cached scalar values


def run_batch_training_loop(G, D, opt_G, opt_D, scaler, device: torch.device, args, config: Dict, n_classes: int, pbar, step: int, metrics_history: Dict, total_steps: int, epoch: int, file_progress_prefix: str) -> int:
    """
    Execute one full epoch of batch training steps for generator and discriminator.

    :param G: generator model
    :param D: discriminator model
    :param opt_G: generator optimizer
    :param opt_D: discriminator optimizer
    :param scaler: AMP gradient scaler (may be None)
    :param device: torch device for tensor allocation
    :param args: parsed arguments namespace with training hyperparameters
    :param config: configuration dictionary for gradient penalty computation
    :param n_classes: number of label classes for conditional generation
    :param pbar: tqdm progress bar wrapping the dataloader
    :param step: current global step counter
    :param metrics_history: metrics dictionary to append to in-place
    :param total_steps: number of batches per epoch for display
    :param epoch: current epoch index (zero-based)
    :param file_progress_prefix: colored prefix string for progress display
    :return: Updated global step counter
    """

    _cached_loss_D = 0.0  # Cached discriminator loss scalar for progress display without CUDA sync
    _cached_g_loss = 0.0  # Cached generator loss scalar for progress display without CUDA sync
    _cached_gp = 0.0  # Cached gradient penalty scalar for progress display without CUDA sync
    _cached_d_real = 0.0  # Cached real score scalar for progress display without CUDA sync
    _cached_d_fake = 0.0  # Cached fake score scalar for progress display without CUDA sync

    for batch_idx, (real_x_batch, labels_batch) in enumerate(pbar):  # Enumerate batches to obtain current batch index
        real_x = real_x_batch.to(device, non_blocking=True)  # Move real features to device with non_blocking when pinned
        labels = labels_batch.to(device, dtype=torch.long, non_blocking=True)  # Move labels to device with non_blocking when pinned

        loss_D, gp, d_real_score, d_fake_score = execute_discriminator_training_steps(G, D, opt_D, scaler, real_x, labels, device, args, config, n_classes)  # Execute discriminator training steps with gradient penalty
        g_loss = execute_generator_training_step(G, D, opt_G, scaler, device, args, n_classes)  # Execute single generator training step

        if step % args.log_interval == 0:  # Extract scalar metrics only at log interval to avoid per-batch CUDA synchronization
            _cached_loss_D, _cached_g_loss, _cached_gp, _cached_d_real, _cached_d_fake = record_training_step_metrics(step, args, metrics_history, loss_D, g_loss, gp, d_real_score, d_fake_score)  # Record metrics and cache scalar values

        pbar.set_description(  # Update tqdm description using cached scalars to avoid CUDA synchronization
            (
                f"{getattr(args, 'file_progress_prefix', '')} "  # File progress prefix (may include colored index)
                f"{BackgroundColors.CYAN}{(Path(getattr(args, 'csv_path', '')).name if getattr(args, 'csv_path', None) else '')}{Style.RESET_ALL} | "  # Current filename
            )
            + f"{BackgroundColors.CYAN}Epoch {epoch+1}/{args.epochs}{Style.RESET_ALL} | "  # Current epoch and total epochs
            + f"{BackgroundColors.YELLOW}step {batch_idx+1}/{total_steps}{Style.RESET_ALL} | "  # Current batch index and total batches per epoch
            + f"{BackgroundColors.RED}loss_D: {_cached_loss_D:.4f}{Style.RESET_ALL} | "  # Discriminator loss from cached value
            + f"{BackgroundColors.GREEN}loss_G: {_cached_g_loss:.4f}{Style.RESET_ALL} | "  # Generator loss from cached value
            + f"gp: {_cached_gp:.4f} | "  # Gradient penalty from cached value
            + f"D(real): {_cached_d_real:.4f} | "  # Average critic score on real from cached value
            + f"D(fake): {_cached_d_fake:.4f}"  # Average critic score on fake from cached value
        )

        step += 1  # Increment global step counter

    return step  # Return updated step counter


def build_ordered_csv_row_from_runtime(row_runtime: Dict, results_cols_cfg: list, config: Dict) -> list:
    """
    Build an ordered CSV row list from runtime metrics dictionary following configured column schema.

    :param row_runtime: Dictionary mapping column names to runtime-computed values.
    :param results_cols_cfg: List of configured column names defining output order.
    :param config: Configuration dictionary for recursive value lookup on missing columns.
    :return: Ordered list of values aligned to results_cols_cfg schema.
    """

    ordered = []  # Prepare ordered list following config column order
    for c in results_cols_cfg:  # For each configured column name
        if c in row_runtime:  # If runtime metric provides this column
            ordered.append(row_runtime.get(c))  # Use runtime value
        else:  # Otherwise attempt to find value in configuration
            cfg_val = None  # Default when not found
            try:  # Guard config lookup
                cfg_val = find_config_value(config, c)  # Search config recursively for key
            except Exception:  # If lookup fails
                cfg_val = None  # Treat as missing on failure
            if cfg_val is not None:  # If config provided a value
                ordered.append(cfg_val)  # Use configured hyperparameter value
            else:  # Neither runtime nor config provided the column value
                ordered.append(None)  # Use None to indicate missing value explicitly
    return ordered  # Return ordered row list aligned to configured schema


def inject_hardware_into_csv_row(ordered: list, results_cols_cfg: list, config: Dict, device: torch.device) -> list:
    """
    Insert hardware specification string into a CSV row when hardware_tracking is enabled.

    :param ordered: Ordered list of CSV row values to potentially modify in-place.
    :param results_cols_cfg: List of configured column names for index lookup.
    :param config: Configuration dictionary with hardware_tracking flag.
    :param device: Torch device used for hardware specification lookup.
    :return: The same ordered list with hardware string potentially injected.
    """

    if config.get("hardware_tracking", False):  # If hardware tracking requested in config
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
                    if idx_hw < len(ordered):  # Ensure index is within the ordered list
                        ordered[idx_hw] = hardware_str  # Place hardware string in row
                except Exception:  # If index operation fails
                    pass  # Ignore hardware insertion errors
        except Exception:  # If hardware detection fails
            pass  # Ignore any hardware detection errors to keep flow
    return ordered  # Return ordered list with hardware potentially injected


def flush_csv_file_safely(file_handle) -> None:
    """
    Flush a CSV file handle to disk safely, ignoring any errors.

    :param file_handle: Open file handle to flush (may be None).
    :return: None
    """

    try:  # Flush buffer to persist row immediately
        if file_handle is not None:  # Only call flush when file handle exists
            try:  # Guard flush call to avoid raising
                file_handle.flush()  # Flush file buffer to disk
            except Exception:  # If flush fails
                pass  # Continue when flush fails
    except Exception:  # If outer flush logic fails
        pass  # Ignore outer errors as well


def extract_optimizer_learning_rates(opt_G, opt_D, args) -> tuple:
    """
    Safely extract current learning rates from generator and discriminator optimizers.

    :param opt_G: Generator optimizer with param_groups attribute.
    :param opt_D: Discriminator optimizer with param_groups attribute.
    :param args: Parsed arguments namespace with lr fallback attribute.
    :return: Tuple of (learning_rate_generator, learning_rate_critic).
    """

    try:  # Attempt to read current generator optimizer learning rate
        lr_gen = safe_float(opt_G.param_groups[0].get("lr", None), getattr(args, "lr", 0.0))  # Generator LR safely
    except Exception:  # If reading fails
        lr_gen = getattr(args, "lr", "")  # Fallback to args.lr
    try:  # Attempt to read current discriminator optimizer learning rate
        lr_crit = safe_float(opt_D.param_groups[0].get("lr", None), getattr(args, "lr", 0.0))  # Critic LR safely
    except Exception:  # If reading fails
        lr_crit = getattr(args, "lr", "")  # Fallback to args.lr
    return lr_gen, lr_crit  # Return both learning rates as a tuple


def build_epoch_runtime_row(args, dataset, epoch: int, metrics_history: Dict, opt_G, opt_D) -> Dict:
    """
    Build runtime metrics dictionary for a single training epoch CSV row.

    :param args: Parsed arguments namespace with training settings and timing attributes.
    :param dataset: Loaded CSVFlowDataset instance for original sample count.
    :param epoch: Current epoch index (zero-based).
    :param metrics_history: Dictionary of tracked training metrics.
    :param opt_G: Generator optimizer for learning rate extraction.
    :param opt_D: Discriminator optimizer for learning rate extraction.
    :return: Dictionary mapping column names to runtime-computed values.
    """

    row_runtime = {}  # Collect runtime-derived metrics into a dedicated mapping
    row_runtime["original_file"] = Path(args.csv_path).name if getattr(args, "csv_path", None) else ""  # Original file name
    row_runtime["epoch"] = epoch + 1  # Current epoch number (1-based)
    row_runtime["epochs"] = getattr(args, "epochs", "")  # Total epochs configured
    row_runtime["epoch_time_s"] = getattr(args, "_last_epoch_time", "")  # Epoch elapsed seconds
    row_runtime["training_time_s"] = getattr(args, "_last_training_time", "")  # Total training elapsed seconds
    row_runtime["file_time_s"] = getattr(args, "_last_file_time", "")  # File processing elapsed seconds
    row_runtime["batch_size"] = getattr(args, "batch_size", "")  # Effective batch size used
    row_runtime["lambda_gp"] = getattr(args, "lambda_gp", "")  # Gradient penalty coefficient
    row_runtime["latent_dim"] = getattr(args, "latent_dim", "")  # Latent noise dimensionality
    row_runtime["critic_loss"] = metrics_history.get("loss_D", [])[-1] if metrics_history.get("loss_D") else ""  # Last discriminator/critic loss
    row_runtime["generator_loss"] = metrics_history.get("loss_G", [])[-1] if metrics_history.get("loss_G") else ""  # Last generator loss
    row_runtime["gp"] = metrics_history.get("gp", [])[-1] if metrics_history.get("gp") else ""  # Last gradient penalty value
    row_runtime["D_real"] = metrics_history.get("D_real", [])[-1] if metrics_history.get("D_real") else ""  # Avg critic score for real samples
    row_runtime["D_fake"] = metrics_history.get("D_fake", [])[-1] if metrics_history.get("D_fake") else ""  # Avg critic score for fake samples
    row_runtime["wasserstein"] = metrics_history.get("wasserstein", [])[-1] if metrics_history.get("wasserstein") else ""  # Estimated wasserstein distance
    row_runtime["original_num_samples"] = getattr(dataset, "original_num_samples", "")  # Original sample count after preprocessing
    row_runtime["critic_iterations"] = getattr(args, "critic_steps", "")  # Critic iterations per generator update
    lr_gen, lr_crit = extract_optimizer_learning_rates(opt_G, opt_D, args)  # Extract learning rates safely from both optimizers
    row_runtime["learning_rate_generator"] = lr_gen  # Generator learning rate
    row_runtime["learning_rate_critic"] = lr_crit  # Critic learning rate
    return row_runtime  # Return populated epoch runtime metrics dictionary


def write_epoch_csv_row(args, config: Dict, device: torch.device, dataset, epoch: int, epoch_start_time: float, epoch_milestones, results_csv_writer, results_csv_file, results_cols_cfg, metrics_history: Dict, opt_G, opt_D) -> None:
    """
    Compute epoch elapsed time and write a milestone CSV row for the current epoch.

    :param args: parsed arguments namespace with training settings
    :param config: configuration dictionary with hardware_tracking settings
    :param device: torch device used for training
    :param dataset: loaded CSVFlowDataset instance for sample counts
    :param epoch: current epoch index (zero-based)
    :param epoch_start_time: timestamp when the current epoch started
    :param epoch_milestones: set of milestone epoch numbers for CSV writes
    :param results_csv_writer: CSV writer object (may be None)
    :param results_csv_file: open CSV file handle (may be None)
    :param results_cols_cfg: list of configured CSV column names
    :param metrics_history: dictionary of tracked training metrics
    :param opt_G: generator optimizer for learning rate extraction
    :param opt_D: discriminator optimizer for learning rate extraction
    :return: None
    """

    try:  # Safely compute and print epoch elapsed time without interrupting training
        epoch_elapsed = time.time() - epoch_start_time  # Calculate epoch elapsed seconds
        print(f"{BackgroundColors.GREEN}Epoch {epoch+1} elapsed: {BackgroundColors.CYAN}{epoch_elapsed:.2f}s{Style.RESET_ALL}")  # Print epoch elapsed time
        args._last_epoch_time = safe_float(epoch_elapsed, 0.0)  # Store last epoch elapsed on args for external use safely
    except Exception as _te:  # If timing calculation fails
        print(f"{BackgroundColors.YELLOW}Warning: failed to measure epoch time: {_te}{Style.RESET_ALL}")  # Warn but continue

    try:  # Wrap CSV write to avoid crashing on I/O errors
        if results_csv_writer and results_cols_cfg:  # Only write if we have a valid writer and columns
            row_runtime = build_epoch_runtime_row(args, dataset, epoch, metrics_history, opt_G, opt_D)  # Build epoch runtime metrics dict from training state
            ordered = build_ordered_csv_row_from_runtime(row_runtime, results_cols_cfg, config)  # Build ordered CSV row from runtime values
            ordered = inject_hardware_into_csv_row(ordered, results_cols_cfg, config, device)  # Inject hardware spec into row if tracking enabled
            if (epoch + 1) in epoch_milestones:  # Only write per-epoch row when this epoch is a milestone
                results_csv_writer.writerow(ordered)  # Write ordered row following configured schema for milestone epoch
                flush_csv_file_safely(results_csv_file)  # Flush CSV file to disk safely after milestone write
    except Exception as _cw:  # If writing fails, warn and continue training
        print(f"{BackgroundColors.YELLOW}Warning: failed to write epoch row to results CSV: {_cw}{Style.RESET_ALL}")  # Warn but do not abort


def resolve_checkpoint_dir_and_prefix(args, config: Dict, epoch: int) -> tuple:
    """
    Resolve checkpoint directory path, filename prefix, and verify disk space availability.

    :param args: Parsed arguments namespace with csv_path, out_dir, and save_every.
    :param config: Configuration dictionary with dataset directories for disk space verification.
    :param epoch: Current epoch index (zero-based) for periodic save check.
    :return: Tuple of (checkpoint_dir, checkpoint_prefix, skip_checkpoint).
    """

    checkpoint_dir = Path(args.out_dir) / "Checkpoints"  # Default checkpoint directory
    checkpoint_prefix = "model"  # Default prefix for checkpoint files
    skip_checkpoint = False  # Initialize skip flag to False
    if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:  # Save checkpoints periodically
        dataset_dir_entries = config.get("dataset", {}).get("datasets", {}) if isinstance(config, dict) else {}  # Get dataset section from config
        dataset_dirs_list = []  # Accumulate all dataset directory paths for disk space verification
        for ck_v in dataset_dir_entries.values():  # Iterate dataset group values from config
            if isinstance(ck_v, list):  # If value is a list of directory paths
                dataset_dirs_list.extend(ck_v)  # Add all paths from this group to accumulator
        if not dataset_dirs_list and getattr(args, "csv_path", None):  # If no dirs from config, fallback to csv_path parent
            dataset_dirs_list.append(str(Path(args.csv_path).parent))  # Use CSV file's parent directory as fallback
        skip_checkpoint = not is_checkpoint_space_available(dataset_dirs_list, config)  # Verify sufficient disk space before creating checkpoint directories
        if not skip_checkpoint:  # Only proceed when disk space is sufficient for checkpoint saving
            if args.csv_path:  # If CSV path is provided
                csv_path_obj = Path(args.csv_path)  # Create Path object from csv_path
                checkpoint_dir = csv_path_obj.parent / "Data_Augmentation" / "Checkpoints"  # Create Checkpoints subdirectory
                try:  # Guard directory creation against disk-full and other I/O errors
                    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure directory exists
                except OSError as _ose_dir:  # Catch filesystem errors including disk full during makedirs
                    if _ose_dir.errno == 28:  # If error is errno 28 (ENOSPC - no space left on device)
                        print(f"{BackgroundColors.YELLOW}[WARNING] Checkpoint could not be saved due to disk space exhaustion.{Style.RESET_ALL}")  # Warn about disk exhaustion
                    else:  # If error is unrelated to disk space
                        print(f"{BackgroundColors.YELLOW}Warning: Failed to create checkpoint directory: {_ose_dir}{Style.RESET_ALL}")  # Warn about directory creation failure
                    skip_checkpoint = True  # Mark checkpoint as skipped due to directory creation failure
                checkpoint_prefix = csv_path_obj.stem  # Use input filename as prefix
            else:  # No CSV path, use default out_dir
                checkpoint_dir = Path(args.out_dir) / "Checkpoints"  # Create Checkpoints subdirectory in out_dir
                try:  # Guard directory creation against disk-full and other I/O errors
                    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure directory exists
                except OSError as _ose_dir:  # Catch filesystem errors including disk full during makedirs
                    if _ose_dir.errno == 28:  # If error is errno 28 (ENOSPC - no space left on device)
                        print(f"{BackgroundColors.YELLOW}[WARNING] Checkpoint could not be saved due to disk space exhaustion.{Style.RESET_ALL}")  # Warn about disk exhaustion
                    else:  # If error is unrelated to disk space
                        print(f"{BackgroundColors.YELLOW}Warning: Failed to create checkpoint directory: {_ose_dir}{Style.RESET_ALL}")  # Warn about directory creation failure
                    skip_checkpoint = True  # Mark checkpoint as skipped due to directory creation failure
                checkpoint_prefix = "model"  # Default prefix
    else:  # Not a checkpoint epoch
        skip_checkpoint = True  # Mark as skipped because this is not a save epoch
    return checkpoint_dir, checkpoint_prefix, skip_checkpoint  # Return resolved directory, prefix, and skip flag


def build_generator_checkpoint_payload(G, opt_G, scaler, dataset, epoch: int, metrics_history: Dict, args) -> Dict:
    """
    Build the generator checkpoint dictionary with all training state for serialization.

    :param G: Generator model to extract state_dict from.
    :param opt_G: Generator optimizer to extract state_dict from.
    :param scaler: AMP gradient scaler (may be None).
    :param dataset: Loaded CSVFlowDataset instance for scaler, label_encoder, and feature_cols.
    :param epoch: Current epoch index (zero-based).
    :param metrics_history: Dictionary of tracked training metrics.
    :param args: Parsed arguments namespace to serialize with checkpoint.
    :return: Dictionary containing all generator checkpoint data.
    """

    unique_labels, label_counts = np.unique(dataset.labels, return_counts=True)  # Get class distribution
    class_distribution = dict(zip(unique_labels.tolist(), label_counts.tolist()))  # Create label:count mapping
    g_checkpoint = {
        "epoch": epoch + 1,  # Save current epoch number
        "state_dict": (cast(Any, G).module.state_dict() if isinstance(cast(Any, G), torch.nn.DataParallel) else cast(Any, G).state_dict()),  # Save generator state dict (unwrap DataParallel.module if present)
        "opt_G_state": cast(Any, opt_G).state_dict(),  # Save generator optimizer state
        "scaler": dataset.scaler,  # Save scaler for inverse transform
        "label_encoder": dataset.label_encoder,  # Save label encoder for mapping
        "feature_cols": dataset.feature_cols,  # Save feature column names for generation
        "class_distribution": class_distribution,  # Save class distribution for percentage-based generation
        "metrics_history": metrics_history,  # Save metrics history for resume
        "args": vars(args),  # Save training arguments
    }  # End generator checkpoint dictionary
    if scaler is not None:  # If using AMP
        g_checkpoint["scaler_state"] = scaler.state_dict()  # Save scaler state
    return g_checkpoint  # Return assembled generator checkpoint payload


def save_model_checkpoints_to_disk(g_checkpoint: Dict, G, D, opt_D, checkpoint_dir: Path, checkpoint_prefix: str, epoch: int, args) -> None:
    """
    Save generator and discriminator checkpoint files and latest generator weights to disk.

    :param g_checkpoint: Assembled generator checkpoint dictionary to save.
    :param G: Generator model for extracting latest state_dict.
    :param D: Discriminator model to checkpoint.
    :param opt_D: Discriminator optimizer to checkpoint.
    :param checkpoint_dir: Directory path for saving checkpoint files.
    :param checkpoint_prefix: Filename prefix for checkpoint files.
    :param epoch: Current epoch index (zero-based).
    :param args: Parsed arguments namespace to serialize with discriminator checkpoint.
    :return: None
    """

    g_path = checkpoint_dir / f"{checkpoint_prefix}_generator_epoch{epoch+1}.pt"  # Path for generator checkpoint
    d_path = checkpoint_dir / f"{checkpoint_prefix}_discriminator_epoch{epoch+1}.pt"  # Path for discriminator checkpoint
    try:  # Time model saving to measure save phase duration
        model_save_start_time = time.time()  # Record model save start timestamp
        torch.save(g_checkpoint, str(g_path))  # Save generator checkpoint to disk
        d_checkpoint = {
            "epoch": epoch + 1,  # Save current epoch number
            "state_dict": (cast(Any, D).module.state_dict() if isinstance(cast(Any, D), torch.nn.DataParallel) else cast(Any, D).state_dict()),  # Save discriminator state dict (unwrap DataParallel.module if present)
            "opt_D_state": cast(Any, opt_D).state_dict(),  # Save discriminator optimizer state
            "args": vars(args),  # Save training arguments
        }  # End discriminator checkpoint dictionary
        torch.save(d_checkpoint, str(d_path))  # Save discriminator checkpoint to disk
        latest_path = checkpoint_dir / f"{checkpoint_prefix}_generator_latest.pt"  # Path for latest generator
        torch.save((cast(Any, G).module.state_dict() if isinstance(cast(Any, G), torch.nn.DataParallel) else cast(Any, G).state_dict()), str(latest_path))  # Save latest generator weights safely handling DataParallel
        model_save_elapsed = time.time() - model_save_start_time  # Compute model save elapsed seconds
        args._last_model_save_time = safe_float(model_save_elapsed, 0.0)  # Store last model save elapsed on args safely
        print(f"{BackgroundColors.GREEN}Model save elapsed: {BackgroundColors.CYAN}{model_save_elapsed:.2f}s{Style.RESET_ALL}")  # Print model save elapsed
    except OSError as _ose_ms:  # Catch filesystem errors during model checkpoint save
        if _ose_ms.errno == 28:  # If error is errno 28 (ENOSPC - no space left on device)
            print(f"{BackgroundColors.YELLOW}[WARNING] Checkpoint could not be saved due to disk space exhaustion.{Style.RESET_ALL}")  # Warn about disk exhaustion
        else:  # If error is unrelated to disk space
            print(f"{BackgroundColors.YELLOW}Warning: model save failed: {_ose_ms}{Style.RESET_ALL}")  # Warn about save failure
        args._last_model_save_time = ""  # Ensure attribute exists even on failure
    except Exception as _ms:  # If saving failed for any other reason, warn but continue
        print(f"{BackgroundColors.YELLOW}Warning: model save failed: {_ms}{Style.RESET_ALL}")  # Warn about save failure
        args._last_model_save_time = ""  # Ensure attribute exists even on failure
    print(f"{BackgroundColors.GREEN}Saved generator to {BackgroundColors.CYAN}{g_path}{Style.RESET_ALL}")  # Print checkpoint save message


def save_metrics_history_to_json(checkpoint_dir: Path, checkpoint_prefix: str, metrics_history: Dict) -> None:
    """
    Save training metrics history to a JSON file in the checkpoint directory.

    :param checkpoint_dir: Directory path for saving the metrics JSON file.
    :param checkpoint_prefix: Filename prefix for the metrics JSON file.
    :param metrics_history: Dictionary of tracked training metrics to serialize.
    :return: None
    """

    try:  # Guard metrics JSON write against disk-full and other I/O errors
        metrics_path = checkpoint_dir / f"{checkpoint_prefix}_metrics_history.json"  # Path for metrics JSON
        with open(metrics_path, "w") as f:  # Open file for writing
            json.dump(metrics_history, f, indent=2)  # Save metrics as JSON
        print(f"{BackgroundColors.GREEN}Saved metrics history to {BackgroundColors.CYAN}{metrics_path}{Style.RESET_ALL}")  # Print metrics save message
    except OSError as _ose_mj:  # Catch filesystem errors during metrics JSON write
        if _ose_mj.errno == 28:  # If error is errno 28 (ENOSPC - no space left on device)
            print(f"{BackgroundColors.YELLOW}[WARNING] Checkpoint could not be saved due to disk space exhaustion.{Style.RESET_ALL}")  # Warn about disk exhaustion
        else:  # If error is unrelated to disk space
            print(f"{BackgroundColors.YELLOW}Warning: failed to write metrics history: {_ose_mj}{Style.RESET_ALL}")  # Warn about metrics write failure
    except Exception as _mj:  # If metrics write failed for any other reason
        print(f"{BackgroundColors.YELLOW}Warning: failed to write metrics history: {_mj}{Style.RESET_ALL}")  # Warn about failure


def save_training_checkpoint(args, config: Dict, device: torch.device, G, D, opt_G, opt_D, scaler, dataset, epoch: int, metrics_history: Dict) -> None:
    """
    Save generator and discriminator checkpoints with metrics at periodic intervals.

    :param args: parsed arguments namespace with save_every, csv_path, out_dir
    :param config: configuration dictionary with dataset directories for disk space verification
    :param device: torch device used for training
    :param G: generator model to checkpoint
    :param D: discriminator model to checkpoint
    :param opt_G: generator optimizer to checkpoint
    :param opt_D: discriminator optimizer to checkpoint
    :param scaler: AMP gradient scaler to checkpoint (may be None)
    :param dataset: loaded CSVFlowDataset instance for scaler and label encoder
    :param epoch: current epoch index (zero-based)
    :param metrics_history: dictionary of tracked training metrics
    :return: None
    """

    checkpoint_dir, checkpoint_prefix, skip_checkpoint = resolve_checkpoint_dir_and_prefix(args, config, epoch)  # Resolve checkpoint directory, prefix, and disk space availability
    if not skip_checkpoint:  # Only save files when directory creation succeeded and space is available
        g_checkpoint = build_generator_checkpoint_payload(G, opt_G, scaler, dataset, epoch, metrics_history, args)  # Build generator checkpoint dictionary with all training state
        save_model_checkpoints_to_disk(g_checkpoint, G, D, opt_D, checkpoint_dir, checkpoint_prefix, epoch, args)  # Save generator and discriminator checkpoint files to disk
        save_metrics_history_to_json(checkpoint_dir, checkpoint_prefix, metrics_history)  # Save metrics history JSON to checkpoint directory


def send_epoch_telegram_notifications(args, telegram_enabled: bool, epoch: int, next_notify: int, progress_pct: int) -> int:
    """
    Send telegram progress notifications when epoch completion crosses percentage thresholds.

    :param args: parsed arguments namespace with epochs count and csv_path
    :param telegram_enabled: whether telegram notifications are active
    :param epoch: current epoch index (zero-based)
    :param next_notify: next percentage threshold to trigger notification
    :param progress_pct: percentage increment between notifications
    :return: Updated next_notify threshold
    """

    try:  # Guard notification logic to avoid interrupting training
        if telegram_enabled and args.epochs > 0:  # Only notify when enabled and epochs is positive
            percent = int(((epoch + 1) / float(args.epochs)) * 100)  # Compute percent completed after this epoch
            while percent >= next_notify and next_notify <= 100:  # Send notifications for each crossed threshold
                msg = (
                    f"WGAN-GP training progress: {next_notify}% "  # Short progress message text
                    f"({epoch+1}/{args.epochs} epochs) on {Path(args.csv_path).name if args.csv_path else 'unknown file'}"  # Include filename and epoch info
                )  # Compose progress message
                send_telegram_message(TELEGRAM_BOT, msg)  # Send message via shared helper
                next_notify += progress_pct  # Advance to next threshold to avoid duplicate sends
    except Exception as _err:  # Catch any notification errors and continue training
        pass  # Intentionally ignore notification failures to not interrupt training

    return next_notify  # Return updated notification threshold


def compute_and_store_final_timing(args, training_start_time: float, file_start_time: float) -> None:
    """
    Compute total training and file processing elapsed times and store on args.

    :param args: Parsed arguments namespace to store timing attributes on.
    :param training_start_time: Timestamp when training session started.
    :param file_start_time: Timestamp when file processing started.
    :return: None
    """

    try:  # Safely compute total training and file elapsed times
        training_elapsed = time.time() - training_start_time  # Calculate total training elapsed seconds
        args._last_training_time = safe_float(training_elapsed, 0.0)  # Store total training elapsed on args for downstream use safely
        file_elapsed = time.time() - file_start_time  # Calculate file processing elapsed seconds
        args._last_file_time = safe_float(file_elapsed, 0.0)  # Store file elapsed on args for downstream use safely
        print(f"{BackgroundColors.GREEN}Training finished! Total training elapsed: {BackgroundColors.CYAN}{training_elapsed:.2f}s{Style.RESET_ALL}")  # Print total training elapsed message
        print(f"{BackgroundColors.GREEN}File processing elapsed: {BackgroundColors.CYAN}{file_elapsed:.2f}s{Style.RESET_ALL}")  # Print per-file elapsed message
    except Exception as _tt:  # If timing calculation fails, warn but do not interrupt
        print(f"{BackgroundColors.YELLOW}Warning: failed to compute final training/file elapsed times: {_tt}{Style.RESET_ALL}")  # Warn on failure
        args._last_training_time = ""  # Ensure attribute exists even on failure
        args._last_file_time = ""  # Ensure attribute exists even on failure


def build_final_training_runtime_row(args, config: Dict, dataset, metrics_history: Dict, opt_G, opt_D) -> Dict:
    """
    Build runtime metrics dictionary for the final per-file training summary CSV row.

    :param args: Parsed arguments namespace with training settings and timing attributes.
    :param config: Configuration dictionary with wgangp and paths settings.
    :param dataset: Loaded CSVFlowDataset instance for original sample count.
    :param metrics_history: Dictionary of tracked training metrics.
    :param opt_G: Generator optimizer for learning rate extraction.
    :param opt_D: Discriminator optimizer for learning rate extraction.
    :return: Dictionary mapping column names to runtime-computed values for final row.
    """

    final_runtime: Dict[str, Any] = {}  # Build runtime-only values for final per-file row with explicit typing
    final_runtime["original_file"] = Path(args.csv_path).name if getattr(args, "csv_path", None) else ""  # Original filename
    final_runtime["original_num_samples"] = getattr(dataset, "original_num_samples", "")  # Original sample count after preprocessing
    gen_file = getattr(args, "out_file", None) or ""  # Prefer explicit out_file if set
    if not gen_file and getattr(args, "csv_path", None):  # Derive augmented filename when not explicitly set
        try:  # Attempt to construct derived augmented file path
            csv_obj = Path(args.csv_path)  # Path object for csv
            suffix = config.get("wgangp", {}).get("results_suffix", "_data_augmented")  # Suffix from wgangp config
            derived = csv_obj.parent / config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation") / f"{csv_obj.stem}{suffix}.csv"  # Construct path
            gen_file = str(derived)  # Use derived path string
        except Exception:  # If derivation fails
            gen_file = ""  # Leave blank on failure
    final_runtime["generated_file"] = gen_file  # Store generated file path (may be empty)
    final_runtime["total_generated_samples"] = ""  # Placeholder, generation may fill this later
    final_runtime["generated_ratio"] = ""  # Placeholder ratio
    final_runtime["training_time_s"] = getattr(args, "_last_training_time", "")  # Total training elapsed
    final_runtime["file_time_s"] = getattr(args, "_last_file_time", "")  # Per-file processing elapsed
    final_runtime["testing_time_s"] = 0.0  # Default testing/generation time is zero unless generation runs
    final_runtime["epoch"] = ""  # Final summary must not include a per-epoch value
    final_runtime["epoch_time_s"] = ""  # Final summary must not include last epoch duration
    final_runtime["epochs"] = getattr(args, "epochs", "")  # Total epochs configured
    final_runtime["batch_size"] = getattr(args, "batch_size", "")  # Effective batch size used
    final_runtime["lambda_gp"] = getattr(args, "lambda_gp", "")  # Gradient penalty coefficient
    final_runtime["latent_dim"] = getattr(args, "latent_dim", "")  # Latent noise dimensionality
    final_runtime["critic_iterations"] = getattr(args, "critic_steps", "")  # Critic iterations per generator update
    lr_gen, lr_crit = extract_optimizer_learning_rates(opt_G, opt_D, args)  # Extract learning rates safely from both optimizers
    final_runtime["learning_rate_generator"] = lr_gen  # Generator learning rate
    final_runtime["learning_rate_critic"] = lr_crit  # Critic learning rate
    final_runtime["critic_loss"] = metrics_history.get("loss_D", [])[-1] if metrics_history.get("loss_D") else ""  # Final critic loss
    final_runtime["generator_loss"] = metrics_history.get("loss_G", [])[-1] if metrics_history.get("loss_G") else ""  # Final generator loss
    final_runtime["gp"] = metrics_history.get("gp", [])[-1] if metrics_history.get("gp") else ""  # Final gradient penalty
    final_runtime["D_real"] = metrics_history.get("D_real", [])[-1] if metrics_history.get("D_real") else ""  # Final avg real score
    final_runtime["D_fake"] = metrics_history.get("D_fake", [])[-1] if metrics_history.get("D_fake") else ""  # Final avg fake score
    final_runtime["wasserstein"] = metrics_history.get("wasserstein", [])[-1] if metrics_history.get("wasserstein") else ""  # Final wasserstein estimate
    try:  # Safely compute generated_ratio using safe conversions
        total_generated = safe_float(final_runtime.get("total_generated_samples"), 0.0)  # Total generated safely
        original_samples = safe_float(final_runtime.get("original_num_samples"), 0.0)  # Original samples safely
        final_runtime["generated_ratio"] = (total_generated / original_samples) if original_samples > 0.0 else 0.0  # Guard division and avoid ZeroDivisionError
    except Exception:  # If computation fails
        final_runtime["generated_ratio"] = ""  # Leave blank on failure
    return final_runtime  # Return populated final runtime metrics dictionary


def write_final_timing_and_csv_row(args, config: Dict, device: torch.device, dataset, training_start_time: float, file_start_time: float, results_csv_writer, results_csv_file, results_cols_cfg, metrics_history: Dict, opt_G, opt_D) -> None:
    """
    Compute final training elapsed times and write the summary CSV row.

    :param args: parsed arguments namespace with training settings
    :param config: configuration dictionary with hardware_tracking and paths settings
    :param device: torch device used for training
    :param dataset: loaded CSVFlowDataset instance for sample counts
    :param training_start_time: timestamp when training session started
    :param file_start_time: timestamp when file processing started
    :param results_csv_writer: CSV writer object (may be None)
    :param results_csv_file: open CSV file handle (may be None)
    :param results_cols_cfg: list of configured CSV column names
    :param metrics_history: dictionary of tracked training metrics
    :param opt_G: generator optimizer for learning rate extraction
    :param opt_D: discriminator optimizer for learning rate extraction
    :return: None
    """

    compute_and_store_final_timing(args, training_start_time, file_start_time)  # Compute and store final timing on args

    try:  # Wrap writes to avoid crashing on I/O errors
        if results_csv_writer and results_cols_cfg:  # Only write if writer and schema are available
            final_runtime = build_final_training_runtime_row(args, config, dataset, metrics_history, opt_G, opt_D)  # Build final training runtime metrics dict
            ordered_final = build_ordered_csv_row_from_runtime(final_runtime, results_cols_cfg, config)  # Build ordered CSV row from runtime values
            ordered_final = inject_hardware_into_csv_row(ordered_final, results_cols_cfg, config, device)  # Inject hardware spec into final row if tracking enabled
            results_csv_writer.writerow(ordered_final)  # Write final per-file ordered row to CSV
            flush_csv_file_safely(results_csv_file)  # Flush CSV file to disk safely after final row
    except Exception as _fw:  # If writing fails, warn and continue
        print(f"{BackgroundColors.YELLOW}Warning: failed to write final file row to results CSV: {_fw}{Style.RESET_ALL}")  # Warn about failure


def generate_training_plots(args, config: Dict, metrics_history: Dict, file_progress_prefix: str) -> None:
    """
    Generate and save training metrics plots when metrics data is available.

    :param args: parsed arguments namespace with csv_path and out_dir
    :param config: configuration dictionary passed to plot_training_metrics
    :param metrics_history: dictionary of tracked training metrics with steps list
    :param file_progress_prefix: colored prefix string for console messages
    :return: None
    """

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


def send_final_telegram_messages(args, config: Dict, file_progress_prefix: str) -> None:
    """
    Send final training completion messages via telegram and saved-file helper.

    :param args: parsed arguments namespace with csv_path and epochs
    :param config: configuration dictionary passed to send_file_saved_and_timing_messages
    :param file_progress_prefix: colored prefix string for telegram message
    :return: None
    """

    send_file_saved_and_timing_messages(args, config)  # Send saved-file, size and timing messages via helper
    send_telegram_message(TELEGRAM_BOT, f"{file_progress_prefix} Finished WGAN-GP training on {Path(args.csv_path).name} after {args.epochs} epochs")  # Telegram finish with prefix


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
    
        device, training_start_time, file_start_time, epoch_milestones, file_progress_prefix = normalize_args_and_setup_hardware(args, config)  # Normalize arguments and setup hardware

        dataset, dataloader = create_dataset_and_dataloader(args, config, device)  # Create dataset and dataloader

        results_csv_file, results_csv_writer, results_cols_cfg, feature_dim, n_classes = init_results_csv_and_feature_dims(args, config, dataset)  # Initialize results CSV and feature dimensions

        G, D, opt_D, opt_G, scaler, fixed_noise, fixed_labels, step, start_epoch, metrics_history = create_models_and_optimizers(args, config, device, feature_dim, n_classes)  # Create models and optimizers

        metrics_history, start_epoch, step = resume_from_checkpoint(args, config, device, G, D, opt_G, opt_D, scaler, metrics_history, start_epoch, step)  # Resume from checkpoint if available
        telegram_enabled, next_notify, progress_pct = init_telegram_progress(args, config, start_epoch)  # Initialize telegram progress tracking

        for epoch in range(start_epoch, args.epochs):  # Loop over epochs starting from resume point
            epoch_start_time = time.time()  # Record epoch start timestamp
            pbar, total_steps = create_epoch_progress_bar(dataloader, args, epoch)  # Create progress bar for epoch
            
            step = run_batch_training_loop(G, D, opt_G, opt_D, scaler, device, args, config, n_classes, pbar, step, metrics_history, total_steps, epoch, file_progress_prefix)  # Execute batch training loop

            write_epoch_csv_row(args, config, device, dataset, epoch, epoch_start_time, epoch_milestones, results_csv_writer, results_csv_file, results_cols_cfg, metrics_history, opt_G, opt_D)  # Write epoch CSV row
            save_training_checkpoint(args, config, device, G, D, opt_G, opt_D, scaler, dataset, epoch, metrics_history)  # Save checkpoint if due
            next_notify = send_epoch_telegram_notifications(args, telegram_enabled, epoch, next_notify, progress_pct)  # Send epoch telegram notifications
        
        write_final_timing_and_csv_row(args, config, device, dataset, training_start_time, file_start_time, results_csv_writer, results_csv_file, results_cols_cfg, metrics_history, opt_G, opt_D)  # Write final timing and CSV row
        generate_training_plots(args, config, metrics_history, file_progress_prefix)  # Generate training plots
        send_final_telegram_messages(args, config, file_progress_prefix)  # Send final telegram messages
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
        styled = df.style.apply(row_style_for_zebra, axis=1)  # Apply zebra function row-wise using top-level helper
        styled = styled.set_table_attributes('style="border-collapse:collapse; width:100%;"')  # Tight table style
        styled = cast(pd.io.formats.style.Styler, cast(Any, styled).set_properties(**{"border": "1px solid #ddd", "padding": "6px"}))  # Cell padding/border (cast to Any to satisfy typing)
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
        out_p.parent.mkdir(parents=True, exist_ok=True)  # Ensure parent directory exists before writing
        dfi.export(cast(Any, styled_df), str(out_p))  # Export styled dataframe to PNG using dataframe_image with cast to satisfy static typing
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
        if not os.access(str(parent), os.W_OK):  # Verify directory is writable before proceeding
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
        if not os.access(str(parent), os.W_OK):  # Verify parent directory is writable before writing CSV
            raise PermissionError(f"Directory not writable: {parent}")  # Raise permission error
        df.to_csv(str(csv_p), index=False)  # Save CSV to disk preserving DataFrame content/order
        
        if is_visualizable:  # If a visual representation is desired
            try:  # Guard PNG rendering to prevent aborting the CSV pipeline on image export failures
                png_path = csv_p.with_suffix(".png")  # Replace CSV extension with PNG
                generate_table_image_from_dataframe(df, png_path)  # Generate PNG from in-memory DataFrame
            except Exception as _png_e:  # If PNG generation fails, warn but preserve the already-written CSV
                print(f"{BackgroundColors.YELLOW}Warning: PNG generation failed for {csv_p.name}: {_png_e}{Style.RESET_ALL}")  # Warn about PNG failure without aborting
    except Exception:
        raise  # Propagate CSV write exceptions to caller (do not swallow)


def compose_generation_start_message(
    n: int,
    args,
    generated_file_name: str,
    original_num: Optional[int] = None,
) -> str:
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
                    if df is not None:  # If reading succeeded, determine original number of samples
                        orig = len(df)  # Determine original number of samples
                    else:  # If both attempts to read failed, keep orig as None
                        orig = None
                except Exception:
                    orig = None  # If reading fails, keep as None
        
        ratio_info = ""  # Initialize ratio info
        
        if orig is not None and safe_float(orig, 0.0) > 0.0:  # If original dataset size is known and valid
            ratio = safe_float(n, 0.0) / safe_float(orig, 1.0)  # Compute generation ratio safely
            percentage = ratio * 100.0  # Convert ratio to percentage
            ratio_info = f"{percentage:.2f}% ({int(safe_float(n,0.0))}/{int(safe_float(orig,0.0))})"  # Format ratio info
        
        elif getattr(args, "n_samples", None) is not None:  # If n_samples was explicitly provided
            try:
                requested = safe_float(getattr(args, "n_samples", None), 0.0)  # Convert to float safely
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


def resolve_expected_sample_count(args, config: Dict) -> Optional[int]:
    """
    Resolve the expected number of generated samples from configuration and args.

    :param args: Parsed arguments namespace with n_samples attribute.
    :param config: Configuration dictionary with generation settings.
    :return: Expected sample count as integer, or None if undeterminable.
    """

    try:  # Safely resolve requested n_samples
        requested = safe_float(
            getattr(args, "n_samples", None)
            if getattr(args, "n_samples", None) is not None
            else config.get("generation", {}).get("n_samples", 1.0),
            config.get("generation", {}).get("n_samples", 1.0),
        )  # Resolve requested n_samples safely
    except Exception:  # Fallback on resolution failure
        requested = config.get("generation", {}).get("n_samples", 1.0)  # Fallback
    expected_n: Optional[int] = None  # Initialize expected sample count
    if requested <= 1.0:  # Percentage mode
        expected_n = compute_expected_samples_for_percentage(requested, args, config)  # Compute expected
    else:  # Absolute mode
        try:  # Attempt integer conversion
            expected_n = int(requested)  # Absolute mode
        except Exception:  # Fallback if conversion fails
            expected_n = None  # Fallback if conversion fails
    return expected_n  # Return resolved expected sample count


def evaluate_existing_augmentation_file(out_path: Path, file_prefix: str, existing_count: int, expected_n: int, args) -> bool:
    """
    Evaluate whether to regenerate or skip based on existing file count versus expected count.

    :param out_path: Path to the existing augmentation output file.
    :param file_prefix: Telegram prefix string for progress display.
    :param existing_count: Number of rows in the existing output file.
    :param expected_n: Expected number of samples based on configuration.
    :param args: Parsed arguments namespace with force_new_samples flag.
    :return: True if generation should proceed, False if it should be skipped.
    """

    if existing_count == expected_n and not getattr(args, "force_new_samples", False):  # Matching and no force
        send_telegram_message(
            TELEGRAM_BOT,
            f"{file_prefix} Skipping Generation: {out_path.name} already exists with {existing_count} samples (expected {expected_n}).",
        )  # Notify skip via Telegram
        return False  # Skip generation
    if getattr(args, "force_new_samples", False):  # Forced regeneration
        send_telegram_message(
            TELEGRAM_BOT,
            f"{file_prefix} Force Regeneration Requested: Removing existing {out_path.name} ({existing_count} samples) and regenerating to {expected_n}.",
        )  # Notify forced regeneration via Telegram
    else:  # Count mismatch requires regeneration
        send_telegram_message(
            TELEGRAM_BOT,
            f"{file_prefix} Existing {out_path.name} has {existing_count} samples but expected {expected_n}; removing and regenerating.",
        )  # Notify mismatch via Telegram
    try:  # Attempt to delete existing file before regeneration
        out_path.unlink()  # Delete existing file
    except Exception:  # Ignore deletion errors
        pass  # Continue regardless of deletion failure
    return True  # Proceed with generation


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

        expected_n = resolve_expected_sample_count(args, config)  # Resolve expected sample count from configuration

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

        return evaluate_existing_augmentation_file(out_path, file_prefix, existing_count, expected_n, args)  # Evaluate whether to regenerate or skip based on count comparison

    except Exception as e:
        print(str(e))
        try:
            send_exception_via_telegram(type(e), e, e.__traceback__)
        except Exception:
            pass
        return True


def normalize_args_and_load_checkpoint(args, config: Dict) -> tuple:
    """
    Normalize argument types, select device, and load generator checkpoint.

    :param args: Parsed arguments namespace containing generation options.
    :param config: Configuration dictionary with generation settings.
    :return: Tuple of (ckpt, args_ck, scaler, label_encoder, feature_cols, class_distribution, device, file_progress_prefix).
    """

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
    return (ckpt, args_ck, scaler, label_encoder, feature_cols, class_distribution, device, file_progress_prefix)  # Return checkpoint data and device info


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
