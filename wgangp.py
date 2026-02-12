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
try:
    from torch.amp.autocast_mode import autocast as _torch_autocast
except Exception:
    _torch_autocast = None


def autocast(device_type: str, enabled: bool = True):
    """
    Return an autocast context manager when enabled on CUDA, else a nullcontext.

    This avoids referencing `torch.amp.autocast` directly (Pylance warning) and
    supports environments without CUDA.

    :param device_type: The device type ('cuda' or 'cpu') to create autocast context for
    :param enabled: Whether to enable autocast context (default: True)
    :return: Autocast context manager if enabled on CUDA, otherwise nullcontext
    """

    if enabled and device_type == "cuda" and _torch_autocast is not None:  # If enabled and CUDA available and autocast exists
        return _torch_autocast(device_type)  # Return CUDA autocast context
    return nullcontext()  # Return null context for CPU or when disabled


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

    # Parse CLI arguments
    args = parse_args()  # Get CLI arguments
    cli_overrides = args_to_config_overrides(args)  # Convert to config overrides

    # Load configuration (CLI > config.yaml > config.yaml.example > defaults)
    config = load_configuration(config_path=args.config, cli_overrides=cli_overrides)  # Load merged config
    CONFIG = config  # Store in global

    # Initialize logger after config loaded
    initialize_logger(config)

    # Setup Telegram bot
    setup_telegram_bot(config)

    start_time = datetime.datetime.now()  # Get the start time of the program
    send_telegram_message(TELEGRAM_BOT, [f"Starting WGAN-GP Data Augmentation at {start_time.strftime('%Y-%m-%d %H:%M:%S')}"])  # Send Telegram notification

    # Extract execution parameters from config
    mode = config.get("wgangp", {}).get("mode", "both")  # Get mode
    csv_path = config.get("wgangp", {}).get("csv_path")  # Get CSV path
    results_suffix = config.get("execution", {}).get("results_suffix", "_data_augmented")  # Get results suffix
    datasets = config.get("dataset", {}).get("datasets", {})  # Get datasets dictionary

    # Build args-like namespace for compatibility
    class ConfigNamespace:
        """Namespace wrapper for config dict."""
    class ConfigNamespace:
        """Namespace wrapper for config dict."""
        def __init__(self, cfg):
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
        # Set output file path if using default
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
            
            # Set checkpoint path to the last saved model (dataset-specific)
            csv_path_obj = Path(csv_path)
            checkpoint_prefix = csv_path_obj.stem  # Use CSV filename as prefix
            checkpoint_subdir = config.get("paths", {}).get("checkpoint_subdir", "Checkpoints")  # Get checkpoint subdir
            data_aug_subdir = config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation")  # Get data aug subdir
            checkpoint_dir = csv_path_obj.parent / data_aug_subdir / checkpoint_subdir
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
            generate(args, config)  # Generate synthetic samples
    
    # Batch processing mode (no csv_path provided):
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
                    
                    # Set output file path: Data_Augmentation subdirectory with same filename
                    csv_path_obj = Path(file)
                    data_aug_subdir = config.get("paths", {}).get("data_augmentation_subdir", "Data_Augmentation")  # Get subdir name
                    data_aug_dir = csv_path_obj.parent / data_aug_subdir  # Create Data_Augmentation subdirectory path
                    os.makedirs(data_aug_dir, exist_ok=True)  # Ensure Data_Augmentation directory exists
                    output_filename = f"{csv_path_obj.stem}{results_suffix}{csv_path_obj.suffix}"  # Use input name with RESULTS_SUFFIX
                    args.out_file = str(data_aug_dir / output_filename)  # Set output file path to Data_Augmentation subdirectory
                    args.csv_path = file  # Set CSV path to current file
                    
                    try:
                        if mode == "train":  # Training mode
                            train(args, config)  # Train the model only
                        elif mode == "gen":  # Generation mode
                            assert args.checkpoint is not None, "Generation requires --checkpoint"
                            generate(args, config)  # Generate synthetic samples only
                        elif mode == "both":  # Combined mode
                            print(f"{BackgroundColors.GREEN}[1/2] Training model on {BackgroundColors.CYAN}{csv_path_obj.name}{BackgroundColors.GREEN}...{Style.RESET_ALL}")
                            train(args, config)  # Train the model
                            
                            # Set checkpoint path to the last saved model (dataset-specific)
                            checkpoint_prefix = csv_path_obj.stem  # Use CSV filename as prefix
                            checkpoint_subdir = config.get("paths", {}).get("checkpoint_subdir", "Checkpoints")  # Get checkpoint subdir
                            checkpoint_dir = data_aug_dir / checkpoint_subdir  # Checkpoints in Data_Augmentation/Checkpoints
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
                            generate(args, config)  # Generate synthetic samples
                            
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

    if config.get("sound", {}).get("enabled", True):  # If sound enabled
        atexit.register(lambda: play_sound(config))  # Register the play_sound function to be called when the program finishes


if __name__ == "__main__":
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """

    main()  # Call the main function
