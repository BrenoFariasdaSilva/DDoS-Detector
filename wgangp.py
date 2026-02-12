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


def train(args, config: Optional[Dict] = None):
    """
    Train the WGAN-GP model using the provided arguments and configuration.

    :param args: parsed arguments namespace containing training configuration
    :param config: Optional configuration dictionary (will use global CONFIG if not provided)
    :return: None
    """

    if config is None:  # If no config provided
        config = CONFIG or get_default_config()  # Use global or default config
    
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
    
    # Get dataloader settings from config
    num_workers = config.get("dataloader", {}).get("num_workers", 8)  # Get num_workers from config
    pin_memory = config.get("dataloader", {}).get("pin_memory", True) if device.type == 'cuda' else False  # Get pin_memory from config
    persistent_workers = config.get("dataloader", {}).get("persistent_workers", True) if num_workers > 0 else False  # Get persistent_workers from config
    prefetch_factor = config.get("dataloader", {}).get("prefetch_factor", 2) if num_workers > 0 else None  # Get prefetch_factor from config
    
    # Optimized DataLoader settings for better performance
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        drop_last=True, 
        num_workers=num_workers,  # Configurable number of workers
        pin_memory=pin_memory,  # Faster CPU->GPU transfer
        persistent_workers=persistent_workers,  # Keep workers alive between epochs
        prefetch_factor=prefetch_factor,  # Prefetch batches for better GPU utilization
    )  # Create dataloader for batching

    feature_dim = dataset.feature_dim  # Get feature dimensionality from dataset
    n_classes = dataset.n_classes  # Get number of label classes from dataset

    # Get leaky_relu_alpha from config for generator and discriminator
    g_leaky_relu_alpha = config.get("generator", {}).get("leaky_relu_alpha", 0.2)  # Get generator LeakyReLU alpha
    d_leaky_relu_alpha = config.get("discriminator", {}).get("leaky_relu_alpha", 0.2)  # Get discriminator LeakyReLU alpha
    
    G = Generator(
        latent_dim=args.latent_dim,
        feature_dim=feature_dim,
        n_classes=n_classes,
        hidden_dims=args.g_hidden,
        embed_dim=args.embed_dim,
        n_resblocks=args.n_resblocks,
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
            plot_training_metrics(temp_metrics, str(plot_dir), plot_filename, config)  # Create and save plots with config
            args.out_dir = original_out_dir  # Restore original out_dir
        else:  # No CSV path, use default out_dir
            plot_training_metrics(metrics_history, args.out_dir, "training_metrics.png", config)  # Create and save plots with config

    send_telegram_message(TELEGRAM_BOT, f"Finished WGAN-GP training on {Path(args.csv_path).name} after {args.epochs} epochs")


def generate(args, config: Optional[Dict] = None):
    """
    Generate synthetic samples from a saved generator checkpoint.

    :param args: parsed arguments namespace containing generation options
    :param config: Optional configuration dictionary (will use global CONFIG if not provided)
    :return: None
    """

    if config is None:  # If no config provided
        config = CONFIG or get_default_config()  # Use global or default config
    
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

    # Get generator config from checkpoint or use defaults
    g_leaky_relu_alpha = config.get("generator", {}).get("leaky_relu_alpha", 0.2)  # Get generator LeakyReLU alpha
    
    G = Generator(
        latent_dim=args.latent_dim,
        feature_dim=feature_dim,
        n_classes=n_classes,
        hidden_dims=args.g_hidden,
        embed_dim=args.embed_dim,
        n_resblocks=args.n_resblocks,
        leaky_relu_alpha=g_leaky_relu_alpha,  # Use config value
    ).to(
        device
    )  # Initialize generator model
    G.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)  # Load generator weights from checkpoint
    G.eval()  # Set generator to evaluation mode

    # Get generation config
    small_class_threshold = config.get("generation", {}).get("small_class_threshold", 100)  # Get small class threshold
    small_class_min_samples = config.get("generation", {}).get("small_class_min_samples", 10)  # Get min samples for small classes
    
    # Determine number of samples to generate (supports both absolute count and percentage)
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
            # For small classes, ensure minimum samples are generated
            final_count = max(small_class_min_samples if original_count < small_class_threshold else 1, calculated)  # Apply minimum threshold
            n_per_class = {args.label: final_count}  # Store final count
        else:  # Generate for all classes
            n_per_class = {}  # Initialize dictionary
            for label, original_count in class_distribution.items():  # For each class
                calculated = int(original_count * args.n_samples)  # Calculate percentage-based count
                # For small classes, ensure minimum samples are generated
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


def play_sound(config: Optional[Dict] = None):
    """
    Plays a sound when the program finishes and skips if the operating system is Windows.

    :param config: Optional configuration dictionary containing sound settings
    :return: None
    """

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

    # Load base configuration
    if config is None:  # No config provided
        final_config = load_configuration()  # Load from default locations
    elif isinstance(config, str):  # Config is a file path
        final_config = load_configuration(config_path=config)  # Load from specified file
    elif isinstance(config, dict):  # Config is a dictionary
        final_config = load_configuration()  # Load defaults first
        final_config = deep_merge(final_config, config)  # Merge with provided dict
    else:  # Invalid config type
        raise TypeError(f"config must be dict, str, or None, not {type(config)}")

    # Apply kwargs as highest-priority overrides
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

    # Initialize logger
    initialize_logger(final_config)

    # Setup Telegram bot
    setup_telegram_bot(final_config)

    # Create args-like namespace for compatibility with existing train/generate functions
    class ConfigNamespace:
        """Namespace object that wraps configuration dictionary."""
        def __init__(self, config_dict):
            self.config = config_dict
            # Extract commonly used values for direct access
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

    # Execute based on mode
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
            # Auto-set checkpoint for generation
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
