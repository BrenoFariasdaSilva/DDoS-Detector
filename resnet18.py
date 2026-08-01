"""
Provide a sklearn-compatible one-dimensional ResNet18 classifier.
"""

import math  # Provide validation-loss infinity
import multiprocessing  # Detect automatic device selection inside worker processes
from typing import Any, Optional  # Type public estimator inputs and optional controls

import numpy as np  # Process CPU-resident tabular arrays and memmaps
import torch  # Reuse the repository deep-learning backend
from sklearn.base import BaseEstimator, ClassifierMixin  # Provide sklearn estimator behavior
from sklearn.model_selection import train_test_split  # Derive validation rows only from training input
from torch import nn  # Build residual one-dimensional neural layers


class ResNet18BasicBlock1D(nn.Module):  # Implement one one-dimensional residual block
    """
    Transform numeric feature-axis activations with a residual Conv1d block.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param stride: Convolution stride for optional downsampling.
    :param kernel_size: Odd padding-safe convolution kernel size.
    :param dropout: Dropout probability used inside the block.
    :param activation: Activation function name.
    :param normalization: Normalization layer name.
    :return: Initialized residual block.
    """

    expansion = 1  # Preserve basic-block output width

    def __init__(self, in_channels: int, out_channels: int, stride: int, kernel_size: int, dropout: float, activation: str, normalization: str) -> None:  # Initialize one residual block
        super().__init__()  # Initialize native PyTorch module state
        padding = int(kernel_size) // 2  # Preserve short feature widths through padded convolutions
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=int(kernel_size), stride=int(stride), padding=padding, bias=False)  # Build first residual convolution
        self.norm1 = self.build_normalization(out_channels, normalization)  # Build first normalization layer
        self.activation = self.build_activation(activation)  # Build configured nonlinearity
        self.dropout = nn.Dropout(float(dropout))  # Build configured dropout
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=int(kernel_size), stride=1, padding=padding, bias=False)  # Build second residual convolution
        self.norm2 = self.build_normalization(out_channels, normalization)  # Build second normalization layer
        self.shortcut = nn.Identity() if int(stride) == 1 and int(in_channels) == int(out_channels) else nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=int(stride), bias=False), self.build_normalization(out_channels, normalization))  # Build identity or projection shortcut

    def build_activation(self, activation: str) -> nn.Module:  # Build one supported activation module
        if activation == "relu":  # Select ReLU activation
            return nn.ReLU(inplace=False)  # Return ReLU module
        if activation == "gelu":  # Select GELU activation
            return nn.GELU()  # Return GELU module
        return nn.SiLU(inplace=False)  # Return SiLU module

    def build_normalization(self, channels: int, normalization: str) -> nn.Module:  # Build one supported normalization module
        if normalization == "batch_norm":  # Select batch normalization
            return nn.BatchNorm1d(channels)  # Return batch normalization over channels
        if normalization == "group_norm":  # Select group normalization
            groups = 8 if int(channels) % 8 == 0 else 4 if int(channels) % 4 == 0 else 1  # Pick a divisor that supports every configured width
            return nn.GroupNorm(groups, channels)  # Return group normalization over channels
        return nn.Identity()  # Return identity when normalization is disabled

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # Compute one residual block output
        residual = self.shortcut(features)  # Compute residual branch
        hidden = self.conv1(features)  # Apply first convolution
        hidden = self.norm1(hidden)  # Normalize first convolution output
        hidden = self.activation(hidden)  # Apply configured activation
        hidden = self.dropout(hidden)  # Apply dropout during training
        hidden = self.conv2(hidden)  # Apply second convolution
        hidden = self.norm2(hidden)  # Normalize second convolution output
        return self.activation(hidden + residual)  # Return activated residual sum


class ResNet18Network1D(nn.Module):  # Implement dynamic one-dimensional ResNet18 architecture
    """
    Classify row-wise numeric tabular records with a ResNet18-style Conv1d network.

    :param n_features: Number of numeric input features.
    :param n_classes: Number of output classes.
    :param initial_channels: Stem channel count.
    :param stage_widths: Four residual-stage channel widths.
    :param kernel_size: Odd padding-safe convolution kernel size.
    :param dropout: Dropout probability.
    :param activation: Activation function name.
    :param normalization: Normalization layer name.
    :return: Initialized ResNet18 network.
    """

    def __init__(self, n_features: int, n_classes: int, initial_channels: int, stage_widths: tuple[int, int, int, int], kernel_size: int, dropout: float, activation: str, normalization: str) -> None:  # Initialize dynamic network dimensions
        super().__init__()  # Initialize native PyTorch module state
        self.architecture_variant = "resnet18_1d"  # Record factual architecture variant
        self.stage_blocks = [2, 2, 2, 2]  # Preserve canonical ResNet18 block counts
        self.input_shape_ = (1, int(n_features))  # Record batch-excluded Conv1d input shape
        self.stem = nn.Sequential(nn.Conv1d(1, int(initial_channels), kernel_size=int(kernel_size), stride=1, padding=int(kernel_size) // 2, bias=False), self.build_normalization(int(initial_channels), normalization), self.build_activation(activation))  # Build short-feature-safe stem
        self.in_channels = int(initial_channels)  # Track current channel width while stages are built
        self.layer1 = self.build_stage(int(stage_widths[0]), 2, 1, int(kernel_size), float(dropout), str(activation), str(normalization))  # Build first residual stage
        self.layer2 = self.build_stage(int(stage_widths[1]), 2, 2, int(kernel_size), float(dropout), str(activation), str(normalization))  # Build second residual stage
        self.layer3 = self.build_stage(int(stage_widths[2]), 2, 2, int(kernel_size), float(dropout), str(activation), str(normalization))  # Build third residual stage
        self.layer4 = self.build_stage(int(stage_widths[3]), 2, 2, int(kernel_size), float(dropout), str(activation), str(normalization))  # Build fourth residual stage
        self.pool = nn.AdaptiveAvgPool1d(1)  # Pool any feature-axis width to one value per channel
        self.dropout = nn.Dropout(float(dropout))  # Apply configured head dropout
        self.head = nn.Linear(int(stage_widths[3]), int(n_classes))  # Project pooled channels to dynamic class logits

    def build_activation(self, activation: str) -> nn.Module:  # Build one supported activation module
        if activation == "relu":  # Select ReLU activation
            return nn.ReLU(inplace=False)  # Return ReLU module
        if activation == "gelu":  # Select GELU activation
            return nn.GELU()  # Return GELU module
        return nn.SiLU(inplace=False)  # Return SiLU module

    def build_normalization(self, channels: int, normalization: str) -> nn.Module:  # Build one supported normalization module
        if normalization == "batch_norm":  # Select batch normalization
            return nn.BatchNorm1d(channels)  # Return batch normalization over channels
        if normalization == "group_norm":  # Select group normalization
            groups = 8 if int(channels) % 8 == 0 else 4 if int(channels) % 4 == 0 else 1  # Pick a divisor that supports every configured width
            return nn.GroupNorm(groups, channels)  # Return group normalization over channels
        return nn.Identity()  # Return identity when normalization is disabled

    def build_stage(self, out_channels: int, blocks: int, stride: int, kernel_size: int, dropout: float, activation: str, normalization: str) -> nn.Sequential:  # Build one residual stage
        layers = [ResNet18BasicBlock1D(self.in_channels, out_channels, stride, kernel_size, dropout, activation, normalization)]  # Build first block with optional downsampling
        self.in_channels = int(out_channels)  # Advance channel tracker after first block
        for _ in range(1, int(blocks)):  # Add remaining identity-width blocks
            layers.append(ResNet18BasicBlock1D(self.in_channels, out_channels, 1, kernel_size, dropout, activation, normalization))  # Append same-width residual block
        return nn.Sequential(*layers)  # Return sequential residual stage

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # Compute class logits for one bounded batch
        hidden = features.unsqueeze(1)  # Convert batch by features to batch by channel by features
        hidden = self.stem(hidden)  # Apply one-dimensional stem
        hidden = self.layer1(hidden)  # Apply first residual stage
        hidden = self.layer2(hidden)  # Apply second residual stage
        hidden = self.layer3(hidden)  # Apply third residual stage
        hidden = self.layer4(hidden)  # Apply fourth residual stage
        hidden = self.pool(hidden).flatten(1)  # Adaptively pool feature axis
        hidden = self.dropout(hidden)  # Apply head dropout during training
        return self.head(hidden)  # Return class logits


class ResNet18Classifier(ClassifierMixin, BaseEstimator):  # Expose ResNet18 through sklearn classifier interfaces
    """
    Train and serve a mini-batch one-dimensional ResNet18 classifier.

    :param architecture_variant: Factual architecture variant identifier.
    :param epochs: Maximum training epochs.
    :param batch_size: Training and validation mini-batch size.
    :param initial_channels: Stem channel count.
    :param stage_widths: Four residual-stage channel widths.
    :param kernel_size: Odd padding-safe convolution kernel size.
    :param normalization: Normalization layer name.
    :param activation: Activation function name.
    :param dropout: Dropout probability.
    :param learning_rate: Optimizer learning rate.
    :param weight_decay: Optimizer weight decay.
    :param optimizer: Optimizer name.
    :param class_weight: Class-weighting mode.
    :param patience: Validation epochs without improvement before stopping.
    :param validation_fraction: Fraction split from input training rows for validation.
    :param min_delta: Minimum validation-loss improvement.
    :param device: Requested device name or auto selection.
    :param allow_device_fallback: Whether explicit unavailable accelerators may fall back to CPU.
    :param prediction_batch_size: Inference mini-batch size.
    :param data_loader_workers: Batch worker policy, currently zero for CPU-backed slice streaming.
    :param mixed_precision: Whether to use CUDA mixed precision.
    :param gradient_clip_norm: Optional gradient clipping norm.
    :param torch_threads: Native PyTorch CPU thread count.
    :param random_state: Deterministic random seed.
    :return: Initialized sklearn-compatible classifier.
    """

    def __init__(self, architecture_variant: str = "resnet18_1d", epochs: int = 20, batch_size: int = 1024, initial_channels: int = 32, stage_widths: tuple[int, int, int, int] = (32, 64, 128, 256), kernel_size: int = 3, normalization: str = "batch_norm", activation: str = "relu", dropout: float = 0.1, learning_rate: float = 0.001, weight_decay: float = 0.00001, optimizer: str = "adamw", class_weight: str = "none", patience: int = 3, validation_fraction: float = 0.1, min_delta: float = 0.0001, device: str = "auto", allow_device_fallback: bool = True, prediction_batch_size: int = 4096, data_loader_workers: int = 0, mixed_precision: bool = False, gradient_clip_norm: Optional[float] = None, torch_threads: int = 1, random_state: int = 42) -> None:  # Store cloneable parameters without allocating model state
        self.architecture_variant = architecture_variant  # Store factual architecture variant
        self.epochs = epochs  # Store maximum epoch count
        self.batch_size = batch_size  # Store bounded training batch size
        self.initial_channels = initial_channels  # Store stem channel count
        self.stage_widths = stage_widths  # Store four residual-stage widths
        self.kernel_size = kernel_size  # Store one-dimensional convolution kernel size
        self.normalization = normalization  # Store normalization selection
        self.activation = activation  # Store activation selection
        self.dropout = dropout  # Store dropout probability
        self.learning_rate = learning_rate  # Store optimizer learning rate
        self.weight_decay = weight_decay  # Store optimizer weight decay
        self.optimizer = optimizer  # Store optimizer selection
        self.class_weight = class_weight  # Store class-weighting mode
        self.patience = patience  # Store early-stopping patience
        self.validation_fraction = validation_fraction  # Store internal validation fraction
        self.min_delta = min_delta  # Store minimum validation-loss improvement
        self.device = device  # Store requested compute device
        self.allow_device_fallback = allow_device_fallback  # Store accelerator fallback policy
        self.prediction_batch_size = prediction_batch_size  # Store bounded inference batch size
        self.data_loader_workers = data_loader_workers  # Store batch worker policy
        self.mixed_precision = mixed_precision  # Store CUDA mixed-precision selection
        self.gradient_clip_norm = gradient_clip_norm  # Store optional gradient clipping norm
        self.torch_threads = torch_threads  # Store bounded native CPU thread count
        self.random_state = random_state  # Store deterministic seed

    def resolve_device(self) -> torch.device:  # Resolve accelerator selection with explicit fallback policy
        requested = str(self.device).lower()  # Normalize requested device name
        if requested == "auto" and multiprocessing.current_process().name != "MainProcess":  # Avoid uncontrolled GPU use in feature-set workers
            requested = "cpu"  # Select CPU for automatic worker processes
        if requested == "auto":  # Select best available PyTorch device
            requested = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"  # Prefer CUDA, then Apple GPU, then CPU
        if requested.startswith("cuda") and not torch.cuda.is_available():  # Handle unavailable CUDA
            if not bool(self.allow_device_fallback):  # Honor strict accelerator policy
                raise RuntimeError("CUDA device was requested for ResNet18 but CUDA is unavailable")  # Raise clear accelerator error
            requested = "cpu"  # Preserve functional CPU execution
        if requested == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):  # Handle unavailable Apple GPU
            if not bool(self.allow_device_fallback):  # Honor strict accelerator policy
                raise RuntimeError("MPS device was requested for ResNet18 but MPS is unavailable")  # Raise clear accelerator error
            requested = "cpu"  # Preserve functional CPU execution
        return torch.device(requested)  # Return effective device

    def validate_parameters(self) -> None:  # Validate architecture and training controls
        if str(self.architecture_variant) != "resnet18_1d":  # Require truthful variant metadata
            raise ValueError("architecture_variant must be resnet18_1d")  # Reject unsupported architecture variants
        if int(self.epochs) < 1 or int(self.batch_size) < 1 or int(self.prediction_batch_size) < 1 or int(self.torch_threads) < 1:  # Require usable iteration, batch, and thread values
            raise ValueError("epochs, batch_size, prediction_batch_size, and torch_threads must be positive")  # Reject unusable controls
        if int(self.data_loader_workers) != 0:  # Preserve CPU-backed array and memmap streaming policy
            raise ValueError("data_loader_workers must be 0 for ResNet18 CPU-backed slice streaming")  # Reject worker-based duplication risk
        if int(self.initial_channels) < 1 or len(tuple(self.stage_widths)) != 4 or any(int(width) < 1 for width in tuple(self.stage_widths)):  # Require valid channel dimensions
            raise ValueError("initial_channels must be positive and stage_widths must contain four positive integers")  # Reject invalid channel configuration
        if int(self.kernel_size) < 1 or int(self.kernel_size) % 2 != 1:  # Require odd convolution kernel size
            raise ValueError("kernel_size must be a positive odd integer")  # Reject padding-unsafe kernel size
        if not 0.0 <= float(self.dropout) < 1.0 or not 0.0 < float(self.validation_fraction) < 1.0:  # Require valid probability controls
            raise ValueError("dropout must be in [0, 1) and validation_fraction must be in (0, 1)")  # Reject invalid probability controls
        if float(self.learning_rate) <= 0.0 or float(self.weight_decay) < 0.0 or int(self.patience) < 1:  # Require valid optimizer controls
            raise ValueError("learning_rate must be positive; weight_decay must be non-negative; patience must be positive")  # Reject invalid optimizer controls
        if self.gradient_clip_norm is not None and float(self.gradient_clip_norm) <= 0.0:  # Require valid optional clipping norm
            raise ValueError("gradient_clip_norm must be positive when provided")  # Reject invalid clipping norm
        if str(self.activation) not in ("relu", "gelu", "silu"):  # Restrict activation to supported modules
            raise ValueError("activation must be relu, gelu, or silu")  # Reject unsupported activation names
        if str(self.normalization) not in ("batch_norm", "group_norm", "none"):  # Restrict normalization to supported modules
            raise ValueError("normalization must be batch_norm, group_norm, or none")  # Reject unsupported normalization names
        if str(self.optimizer) not in ("adamw", "adam", "sgd"):  # Restrict optimizer to implemented options
            raise ValueError("optimizer must be adamw, adam, or sgd")  # Reject unsupported optimizer names
        if str(self.class_weight) not in ("none", "balanced"):  # Restrict class weighting to implemented options
            raise ValueError("class_weight must be none or balanced")  # Reject unsupported class-weighting mode

    def build_network(self, n_features: int, n_classes: int) -> ResNet18Network1D:  # Build network for fitted dimensions
        return ResNet18Network1D(n_features, n_classes, int(self.initial_channels), tuple(int(width) for width in tuple(self.stage_widths)), int(self.kernel_size), float(self.dropout), str(self.activation), str(self.normalization))  # Return dynamically sized network

    def build_optimizer(self, network: nn.Module) -> torch.optim.Optimizer:  # Build configured optimizer
        if str(self.optimizer) == "adam":  # Select Adam optimizer
            return torch.optim.Adam(network.parameters(), lr=float(self.learning_rate), weight_decay=float(self.weight_decay))  # Return Adam optimizer
        if str(self.optimizer) == "sgd":  # Select SGD optimizer
            return torch.optim.SGD(network.parameters(), lr=float(self.learning_rate), weight_decay=float(self.weight_decay), momentum=0.9)  # Return SGD optimizer
        return torch.optim.AdamW(network.parameters(), lr=float(self.learning_rate), weight_decay=float(self.weight_decay))  # Return AdamW optimizer

    def build_loss_function(self, encoded_labels: np.ndarray, n_classes: int, device: torch.device) -> nn.Module:  # Build configured loss function
        if str(self.class_weight) != "balanced":  # Use unweighted cross entropy by default
            return nn.CrossEntropyLoss()  # Return multiclass cross-entropy
        class_counts = np.bincount(encoded_labels, minlength=int(n_classes)).astype(np.float64)  # Count classes in current training input
        class_counts[class_counts == 0.0] = 1.0  # Avoid division by zero for absent labels
        weights = class_counts.sum() / (float(n_classes) * class_counts)  # Compute balanced class weights
        return nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32, device=device))  # Return weighted multiclass cross-entropy

    def format_features(self, features: Any, device: torch.device) -> torch.Tensor:  # Convert one CPU batch to device tensor
        return torch.as_tensor(np.asarray(features, dtype=np.float32), device=device)  # Transfer current feature batch only

    def fit(self, X: Any, y: Any) -> "ResNet18Classifier":  # Fit from CPU arrays using bounded device batches
        self.validate_parameters()  # Validate public controls before allocation
        features = X.to_numpy(copy=False) if hasattr(X, "to_numpy") else np.asarray(X)  # Preserve arrays and memmaps without unconditional duplication
        labels = np.asarray(y).reshape(-1)  # Normalize labels to one vector
        if features.ndim != 2 or features.shape[0] != labels.shape[0] or features.shape[0] < 2 or features.shape[1] < 1:  # Require aligned tabular inputs
            raise ValueError("X must be a non-empty two-dimensional array aligned with y")  # Reject invalid input
        self.classes_ = np.unique(labels)  # Record dynamic class labels
        if self.classes_.shape[0] < 2:  # Require classification targets
            raise ValueError("ResNet18Classifier requires at least two classes")  # Reject single-class input
        contiguous_classes = np.array_equal(self.classes_, np.arange(self.classes_.shape[0]))  # Detect pipeline integer encoding
        encoded_labels = np.asarray(labels, dtype=np.int64) if contiguous_classes else np.searchsorted(self.classes_, labels).astype(np.int64, copy=False)  # Reuse or map labels once on CPU
        class_counts = np.bincount(encoded_labels, minlength=int(self.classes_.shape[0]))  # Count encoded labels for stratified validation safety
        stratify_labels = encoded_labels if int(class_counts.min()) > 1 else None  # Use stratification only when every class can appear in both splits
        all_indices = np.arange(features.shape[0], dtype=np.int64)  # Build row indices without copying features
        train_indices, validation_indices = train_test_split(all_indices, test_size=float(self.validation_fraction), random_state=int(self.random_state), stratify=stratify_labels)  # Derive validation only from training input
        self.n_features_in_ = int(features.shape[1])  # Record dynamic input width
        effective_device = self.resolve_device()  # Resolve compute device
        torch.set_num_threads(int(self.torch_threads))  # Bound native CPU threads
        torch.manual_seed(int(self.random_state))  # Seed CPU and active backend initialization
        if torch.cuda.is_available():  # Seed available CUDA devices
            torch.cuda.manual_seed_all(int(self.random_state))  # Make CUDA initialization reproducible
        torch.use_deterministic_algorithms(False)  # Allow nondeterministic CUDA kernels without reproducibility warnings
        network = self.build_network(self.n_features_in_, int(self.classes_.shape[0])).to(effective_device)  # Allocate model parameters on selected device
        optimizer = self.build_optimizer(network)  # Build configured optimizer
        loss_function = self.build_loss_function(encoded_labels[train_indices], int(self.classes_.shape[0]), effective_device)  # Build current-training loss without test rows
        scaler = torch.amp.GradScaler("cuda", enabled=bool(self.mixed_precision) and effective_device.type == "cuda")  # Enable mixed precision only on CUDA
        random_generator = np.random.default_rng(int(self.random_state))  # Build deterministic batch-order generator
        best_loss = math.inf  # Initialize validation target
        best_state = None  # Retain only best model state on CPU
        best_epoch = 0  # Record best validation epoch
        stale_epochs = 0  # Count epochs without improvement
        epochs_completed = 0  # Record effective completed epochs
        for epoch_index in range(int(self.epochs)):  # Train until limit or early stopping
            network.train()  # Enable training-time dropout and normalization behavior
            shuffled_indices = random_generator.permutation(train_indices)  # Shuffle row indices without copying features
            for batch_start in range(0, shuffled_indices.shape[0], int(self.batch_size)):  # Stream bounded training batches
                batch_indices = shuffled_indices[batch_start:batch_start + int(self.batch_size)]  # Select current training rows
                batch_features = self.format_features(features[batch_indices], effective_device)  # Transfer current feature batch only
                batch_labels = torch.as_tensor(encoded_labels[batch_indices], dtype=torch.long, device=effective_device)  # Transfer current label batch only
                optimizer.zero_grad(set_to_none=True)  # Release prior gradients
                with torch.autocast(device_type="cuda", enabled=bool(self.mixed_precision) and effective_device.type == "cuda"):  # Use CUDA mixed precision only when safe
                    loss = loss_function(network(batch_features), batch_labels)  # Compute current batch loss
                scaler.scale(loss).backward()  # Backpropagate current batch
                if self.gradient_clip_norm is not None:  # Apply configured gradient clipping
                    scaler.unscale_(optimizer)  # Unscale gradients before clipping
                    torch.nn.utils.clip_grad_norm_(network.parameters(), float(self.gradient_clip_norm))  # Clip gradient norm
                scaler.step(optimizer)  # Update model parameters
                scaler.update()  # Update mixed-precision scaler state
            network.eval()  # Disable dropout and freeze normalization for validation
            validation_loss_sum = 0.0  # Accumulate weighted validation loss
            with torch.inference_mode():  # Disable autograd during validation
                for batch_start in range(0, validation_indices.shape[0], int(self.batch_size)):  # Stream bounded validation batches
                    batch_indices = validation_indices[batch_start:batch_start + int(self.batch_size)]  # Select current validation rows
                    batch_features = self.format_features(features[batch_indices], effective_device)  # Transfer current validation features only
                    batch_labels = torch.as_tensor(encoded_labels[batch_indices], dtype=torch.long, device=effective_device)  # Transfer current validation labels only
                    validation_loss_sum += float(loss_function(network(batch_features), batch_labels).item()) * int(batch_indices.shape[0])  # Accumulate sample-weighted loss
            validation_loss = validation_loss_sum / float(validation_indices.shape[0])  # Compute mean validation loss
            epochs_completed = epoch_index + 1  # Record completed epoch count
            progress_callback = getattr(self, "progress_callback", None)  # Read optional stacking progress callback
            if callable(progress_callback):  # Report real completed epochs when stacking attaches a callback
                progress_callback(epochs_completed)  # Emit epoch-based progress without changing training semantics
            if validation_loss < best_loss - float(self.min_delta):  # Accept meaningful improvement
                best_loss = validation_loss  # Record improved loss
                best_epoch = epochs_completed  # Record improved epoch
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in network.state_dict().items()}  # Snapshot model parameters on CPU
                stale_epochs = 0  # Reset stopping counter
            else:  # Handle no meaningful improvement
                stale_epochs += 1  # Advance stopping counter
                if stale_epochs >= int(self.patience):  # Apply configured patience
                    break  # Stop without using experiment test rows
        if best_state is not None:  # Restore best validated weights
            network.load_state_dict(best_state)  # Load best model state
        self.model_ = network.to("cpu")  # Keep fitted model CPU-safe for joblib
        self.device_ = str(effective_device)  # Record selected training device
        self.training_config_ = {"architecture_variant": str(self.architecture_variant), "device": self.device_, "input_dimension": int(self.n_features_in_), "input_shape": [1, int(self.n_features_in_)], "class_count": int(self.classes_.shape[0]), "class_order": self.classes_.tolist(), "epochs_requested": int(self.epochs), "epochs_completed": epochs_completed, "best_epoch": best_epoch, "best_validation_loss": float(best_loss), "batch_size": int(self.batch_size), "prediction_batch_size": int(self.prediction_batch_size), "data_loader_workers": int(self.data_loader_workers), "torch_threads": int(self.torch_threads), "initial_channels": int(self.initial_channels), "stage_widths": [int(width) for width in tuple(self.stage_widths)], "stage_blocks": [2, 2, 2, 2], "kernel_size": int(self.kernel_size), "normalization": str(self.normalization), "activation": str(self.activation), "dropout": float(self.dropout), "learning_rate": float(self.learning_rate), "weight_decay": float(self.weight_decay), "optimizer": str(self.optimizer), "class_weight": str(self.class_weight), "patience": int(self.patience), "validation_fraction": float(self.validation_fraction), "mixed_precision": bool(self.mixed_precision) and effective_device.type == "cuda", "gradient_clip_norm": None if self.gradient_clip_norm is None else float(self.gradient_clip_norm), "random_state": int(self.random_state)}  # Record effective training configuration
        print(f"[RESNET18] Variant: {self.architecture_variant} | Device: {self.device_} | Input shape: (batch, 1, {self.n_features_in_}) | Classes: {len(self.classes_)} | Epochs: {epochs_completed}/{self.epochs} | Best epoch: {best_epoch}")  # Log training outcome
        del optimizer, loss_function, scaler, best_state, train_indices, validation_indices, all_indices  # Release training-only state
        if effective_device.type == "cuda":  # Release unused cached CUDA allocations
            torch.cuda.empty_cache()  # Return cached CUDA blocks
        return self  # Return fitted estimator

    def predict_logits(self, X: Any) -> np.ndarray:  # Produce CPU logits through bounded batches
        if not hasattr(self, "model_") or not hasattr(self, "classes_"):  # Require fitted state
            raise RuntimeError("ResNet18Classifier is not fitted")  # Reject inference before fit
        features = X.to_numpy(copy=False) if hasattr(X, "to_numpy") else np.asarray(X)  # Preserve arrays and memmaps without unconditional duplication
        if features.ndim != 2 or features.shape[1] != int(self.n_features_in_):  # Require fitted feature width
            raise ValueError(f"X must contain {self.n_features_in_} features")  # Reject incompatible input schema
        effective_device = self.resolve_device()  # Resolve available inference device
        torch.set_num_threads(int(self.torch_threads))  # Preserve bounded CPU execution
        network = self.model_.to(effective_device)  # Move model parameters to inference device
        network.eval()  # Disable dropout during inference
        outputs = []  # Accumulate CPU batch logits
        with torch.inference_mode():  # Disable autograd during inference
            for batch_start in range(0, features.shape[0], int(self.prediction_batch_size)):  # Stream bounded inference batches
                batch_features = self.format_features(features[batch_start:batch_start + int(self.prediction_batch_size)], effective_device)  # Transfer current batch only
                outputs.append(network(batch_features).cpu().numpy())  # Return current logits to CPU
        self.model_ = network.to("cpu")  # Restore CPU-safe fitted state
        self.inference_device_ = str(effective_device)  # Record effective inference device
        if effective_device.type == "cuda":  # Release unused cached CUDA allocations
            torch.cuda.empty_cache()  # Return cached CUDA blocks
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, len(self.classes_)), dtype=np.float32)  # Return CPU logits in row order

    def predict_proba(self, X: Any) -> np.ndarray:  # Return probabilities for stacking and explainability
        logits = self.predict_logits(X)  # Compute bounded CPU logits
        logits -= logits.max(axis=1, keepdims=True)  # Stabilize exponentiation
        probabilities = np.exp(logits)  # Convert logits to positive scores
        probabilities /= probabilities.sum(axis=1, keepdims=True)  # Normalize probabilities row-wise
        return probabilities  # Return sklearn-compatible probability matrix

    def predict(self, X: Any) -> np.ndarray:  # Return labels in fitted class order
        return self.classes_[np.argmax(self.predict_logits(X), axis=1)]  # Map maximum logits to original labels
