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
import yaml  # For loading configuration files
from colorama import Style  # For coloring the terminal
from contextlib import nullcontext  # For null context manager
from Logger import Logger  # For logging output to both terminal and file
from pathlib import Path  # For handling file paths
from sklearn.preprocessing import LabelEncoder, StandardScaler  # For data preprocessing
from telegram_bot import TelegramBot, send_telegram_message  # For sending progress messages to Telegram
from torch import autograd  # For gradient penalty
from torch.utils.data import DataLoader, Dataset  # Dataset and DataLoader
from tqdm import tqdm  # For progress bar visualization
from typing import Any, Dict, List, Optional, Union, cast  # For Any type hint and cast

# Prefer CUDA autocast when available; provide a safe fallback context manager
try:  # Attempt to import torch.amp.autocast for mixed precision support
    from torch.amp.autocast_mode import autocast as _torch_autocast  # Import autocast for mixed precision
except Exception:  # If import fails (e.g., CUDA not available), define a fallback
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

# Telegram Bot Setup:
TELEGRAM_BOT = None  # Global Telegram bot instance (initialized in setup_telegram_bot)

# Logger Setup:
logger = None  # Will be initialized in initialize_logger()


# Functions Definitions:


def get_default_config():
    """
    Return the default configuration dictionary for WGAN-GP.

    This defines all configurable parameters with their default values.
    These defaults match config.yaml.example structure and can be overridden
    by config.yaml file or CLI arguments.

    :return: Dictionary containing default configuration values
    """

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

    global CONFIG  # Declare global CONFIG variable
    
    config = get_default_config()  # Start with default configuration

    if config_path is None:  # If no config path provided
        config_path = Path("config.yaml")  # Try config.yaml first
    else:  # If config path provided
        config_path = Path(config_path)  # Convert to Path object

    if not config_path.exists():  # If config.yaml not found
        config_path = Path("config.yaml.example")  # Fallback to example config

    if config_path.exists():  # If a config file exists
        try:  # Try to load configuration from file
            with open(config_path, "r") as f:  # Open config file for reading
                file_config = yaml.safe_load(f)  # Load YAML configuration
            if file_config:  # If config loaded successfully
                config = deep_merge(config, file_config)  # Merge with defaults
                print(f"{BackgroundColors.GREEN}Loaded configuration from: {BackgroundColors.CYAN}{config_path}{Style.RESET_ALL}")  # Print success message
        except Exception as e:  # If loading fails
            print(f"{BackgroundColors.YELLOW}Warning: Failed to load {config_path}: {e}{Style.RESET_ALL}")  # Print warning
            print(f"{BackgroundColors.YELLOW}Using default configuration{Style.RESET_ALL}")  # Fallback message

    if cli_overrides:  # If CLI overrides provided
        config = deep_merge(config, cli_overrides)  # Merge CLI args (highest priority)

    CONFIG = config  # Store in global variable
    return config  # Return merged configuration


def deep_merge(base: Dict, override: Dict) -> Dict:
    """
    Deep merge two dictionaries, with override taking precedence.

    Recursively merges nested dictionaries. Non-dict values from override
    completely replace corresponding values in base.

    :param base: Base dictionary (lower priority)
    :param override: Override dictionary (higher priority)
    :return: Merged dictionary
    """

    result = base.copy()  # Create copy of base dictionary
    for key, value in override.items():  # Iterate override keys
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):  # If both are dicts
            result[key] = deep_merge(result[key], value)  # Recursively merge
        else:  # Direct replacement
            result[key] = value  # Override value
    return result  # Return merged dictionary


def initialize_logger(config: Dict):
    """
    Initialize the logger for output redirection.

    :param config: Configuration dictionary containing logging settings
    :return: None
    """

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


def detect_label_column(columns, config: Optional[Dict] = None):
    """
    Try to guess the label column based on common naming conventions.

    :param columns: List of column names
    :param config: Optional configuration dictionary containing label candidates
    :return: The name of the label column if found, else None
    """

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
            self.seed = config_dict.get("wgangp", {}).get("seed", 42)
            self.force_cpu = config_dict.get("wgangp", {}).get("force_cpu", False)
            self.from_scratch = config_dict.get("wgangp", {}).get("from_scratch", False)
            self.out_dir = config_dict.get("paths", {}).get("out_dir", "outputs")
            self.epochs = config_dict.get("training", {}).get("epochs", 60)
            self.batch_size = config_dict.get("training", {}).get("batch_size", 64)
            self.critic_steps = config_dict.get("training", {}).get("critic_steps", 5)
            self.lr = config_dict.get("training", {}).get("lr", 1e-4)
            self.beta1 = config_dict.get("training", {}).get("beta1", 0.5)
            self.beta2 = config_dict.get("training", {}).get("beta2", 0.9)
            self.lambda_gp = config_dict.get("training", {}).get("lambda_gp", 10.0)
            self.save_every = config_dict.get("training", {}).get("save_every", 5)
            self.log_interval = config_dict.get("training", {}).get("log_interval", 50)
            self.sample_batch = config_dict.get("training", {}).get("sample_batch", 16)
            self.use_amp = config_dict.get("training", {}).get("use_amp", False)
            self.compile = config_dict.get("training", {}).get("compile", False)
            self.latent_dim = config_dict.get("generator", {}).get("latent_dim", 100)
            self.g_hidden = config_dict.get("generator", {}).get("hidden_dims", [256, 512])
            self.embed_dim = config_dict.get("generator", {}).get("embed_dim", 32)
            self.n_resblocks = config_dict.get("generator", {}).get("n_resblocks", 3)
            self.d_hidden = config_dict.get("discriminator", {}).get("hidden_dims", [512, 256, 128])
            self.checkpoint = config_dict.get("generation", {}).get("checkpoint")
            self.n_samples = config_dict.get("generation", {}).get("n_samples", 1.0)
            self.label = config_dict.get("generation", {}).get("label")
            self.out_file = config_dict.get("generation", {}).get("out_file", "generated.csv")
            self.gen_batch_size = config_dict.get("generation", {}).get("gen_batch_size", 256)
            self.feature_dim = config_dict.get("generation", {}).get("feature_dim")
            self.num_workers = config_dict.get("dataloader", {}).get("num_workers", 8)

    args = ConfigNamespace(final_config)  # Create namespace from config

    start_time = datetime.datetime.now()  # Record start time
    send_telegram_message(TELEGRAM_BOT, f"Starting WGAN-GP (programmatic) at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:  # Execute with error handling
        if args.mode == "train":  # Training mode
            train(args, final_config)  # Train model
        elif args.mode == "gen":  # Generation mode
            if args.checkpoint is None:  # Verify checkpoint provided
                raise ValueError("Generation mode requires checkpoint path")
            generate(args, final_config)  # Generate samples
        elif args.mode == "both":  # Combined mode
            train(args, final_config)  # Train first
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


def main():
    """
    Main CLI entry point.

    Handles command-line argument parsing, configuration loading, and execution routing.

    :param: None
    :return: None
    """

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
            self.seed = cfg.get("wgangp", {}).get("seed", 42)
            self.force_cpu = cfg.get("wgangp", {}).get("force_cpu", False)
            self.from_scratch = cfg.get("wgangp", {}).get("from_scratch", False)
            self.out_dir = cfg.get("paths", {}).get("out_dir", "outputs")
            self.epochs = cfg.get("training", {}).get("epochs", 60)
            self.batch_size = cfg.get("training", {}).get("batch_size", 64)
            self.critic_steps = cfg.get("training", {}).get("critic_steps", 5)
            self.lr = cfg.get("training", {}).get("lr", 1e-4)
            self.beta1 = cfg.get("training", {}).get("beta1", 0.5)
            self.beta2 = cfg.get("training", {}).get("beta2", 0.9)
            self.lambda_gp = cfg.get("training", {}).get("lambda_gp", 10.0)
            self.save_every = cfg.get("training", {}).get("save_every", 5)
            self.log_interval = cfg.get("training", {}).get("log_interval", 50)
            self.sample_batch = cfg.get("training", {}).get("sample_batch", 16)
            self.use_amp = cfg.get("training", {}).get("use_amp", False)
            self.compile = cfg.get("training", {}).get("compile", False)
            self.latent_dim = cfg.get("generator", {}).get("latent_dim", 100)
            self.g_hidden = cfg.get("generator", {}).get("hidden_dims", [256, 512])
            self.embed_dim = cfg.get("generator", {}).get("embed_dim", 32)
            self.n_resblocks = cfg.get("generator", {}).get("n_resblocks", 3)
            self.d_hidden = cfg.get("discriminator", {}).get("hidden_dims", [512, 256, 128])
            self.checkpoint = cfg.get("generation", {}).get("checkpoint")
            self.n_samples = cfg.get("generation", {}).get("n_samples", 1.0)
            self.label = cfg.get("generation", {}).get("label")
            self.out_file = cfg.get("generation", {}).get("out_file", "generated.csv")
            self.gen_batch_size = cfg.get("generation", {}).get("gen_batch_size", 256)
            self.feature_dim = cfg.get("generation", {}).get("feature_dim")
            self.num_workers = cfg.get("dataloader", {}).get("num_workers", 8)

    args = ConfigNamespace(config)  # Create args namespace
    
    if csv_path is not None:  # Single file mode (csv_path provided):
        if args.out_file == "generated.csv" and mode in ["gen", "both"]:  # If using default output file
            csv_path_obj = Path(csv_path)  # Create Path object from csv_path
            data_aug_subdir = config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation")  # Get subdir name
            data_aug_dir = csv_path_obj.parent / data_aug_subdir  # Create Data_Augmentation subdirectory path
            os.makedirs(data_aug_dir, exist_ok=True)  # Ensure Data_Augmentation directory exists
            output_filename = f"{csv_path_obj.stem}{results_suffix}{csv_path_obj.suffix}"  # Use input name with suffix
            args.out_file = str(data_aug_dir / output_filename)  # Set output file path to Data_Augmentation subdirectory
        
        if mode == "train":  # Training mode
            train(args, config)  # Train the model
        elif mode == "gen":  # Generation mode
            assert args.checkpoint is not None, "Generation requires --checkpoint"  # Ensure checkpoint is provided
            generate(args, config)  # Generate synthetic samples
        elif mode == "both":  # Combined mode
            print(f"{BackgroundColors.GREEN}[1/2] Training model...{Style.RESET_ALL}")
            train(args, config)  # Train the model
            
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
            generate(args, config)  # Generate synthetic samples
    
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

                files_to_process = get_files_to_process(
                    input_path, file_extension=".csv", config=config
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
                    
                    csv_path_obj = Path(file)  # Create Path object from file path
                    data_aug_subdir = config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation")  # Get subdir name
                    data_aug_dir = csv_path_obj.parent / data_aug_subdir  # Create Data_Augmentation subdirectory path
                    os.makedirs(data_aug_dir, exist_ok=True)  # Ensure Data_Augmentation directory exists
                    output_filename = f"{csv_path_obj.stem}{results_suffix}{csv_path_obj.suffix}"  # Use input name with RESULTS_SUFFIX
                    args.out_file = str(data_aug_dir / output_filename)  # Set output file path to Data_Augmentation subdirectory
                    args.csv_path = file  # Set CSV path to current file
                    
                    try:  # Try to execute the specified mode for the current file
                        if mode == "train":  # Training mode
                            train(args, config)  # Train the model only
                        elif mode == "gen":  # Generation mode
                            assert args.checkpoint is not None, "Generation requires --checkpoint"
                            generate(args, config)  # Generate synthetic samples only
                        elif mode == "both":  # Combined mode
                            print(f"{BackgroundColors.GREEN}[1/2] Training model on {BackgroundColors.CYAN}{csv_path_obj.name}{BackgroundColors.GREEN}...{Style.RESET_ALL}")
                            train(args, config)  # Train the model
                            
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
                            generate(args, config)  # Generate synthetic samples
                            
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


if __name__ == "__main__":
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """

    main()  # Call the main function
