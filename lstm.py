"""
Provide a sklearn-compatible supervised LSTM sequence classifier.
"""
if __name__ in {"__main__", "__mp_main__"}:
    try:
        from setproctitle import setproctitle
        setproctitle(f"DDoS-{__file__.rsplit('/', 1)[-1].rsplit('.', 1)[0]}")
    except ImportError:
        pass


import math  # Provide validation-loss infinity
import multiprocessing  # Detect automatic device selection inside worker processes
from typing import Any, Optional  # Type public estimator inputs and optional controls

import numpy as np  # Process CPU-resident sequence arrays and memmaps
import torch  # Reuse the repository deep-learning backend
from sklearn.base import BaseEstimator, ClassifierMixin  # Provide sklearn estimator behavior
from sklearn.model_selection import train_test_split  # Derive validation sequences only from training input
from torch import nn  # Build recurrent neural layers


class LSTMSequenceIndexDataset:  # Store CPU-backed sequence data with bounded batch transfer
    """
    Stream pre-windowed sequence arrays through index slices.

    :param features: Sequence features shaped samples by timesteps by features.
    :param labels: Optional labels aligned to sequence samples.
    :return: Initialized CPU-backed sequence view.
    """

    def __init__(self, features: Any, labels: Optional[np.ndarray] = None) -> None:  # Store source arrays without device transfer
        self.features = features  # Preserve original array, memmap, or view
        self.labels = labels  # Preserve optional encoded labels

    def batch_features(self, indices: np.ndarray, device: torch.device) -> torch.Tensor:  # Materialize one feature batch on target device
        return torch.as_tensor(np.asarray(self.features[indices], dtype=np.float32), device=device)  # Transfer only current sequence batch

    def batch_labels(self, indices: np.ndarray, device: torch.device) -> torch.Tensor:  # Materialize one label batch on target device
        if self.labels is None:  # Reject label access when labels were not provided
            raise RuntimeError("Labels are unavailable for this LSTM sequence dataset")  # Raise clear dataset-state error
        return torch.as_tensor(self.labels[indices], dtype=torch.long, device=device)  # Transfer only current label batch


class LSTMClassifierNetwork(nn.Module):  # Implement supervised LSTM sequence classifier
    """
    Classify validated sequence windows with a recurrent encoder and head.

    :param n_features: Number of numeric features per timestep.
    :param n_classes: Number of output classes.
    :param hidden_size: LSTM hidden size.
    :param num_layers: LSTM recurrent layer count.
    :param bidirectional: Whether to use bidirectional recurrent context.
    :param dropout: Dropout probability.
    :param activation: Classification-head activation.
    :param normalization: Head normalization mode.
    :param classifier_hidden_dims: Classification-head hidden dimensions.
    :return: Initialized LSTM classifier network.
    """

    def __init__(self, n_features: int, n_classes: int, hidden_size: int, num_layers: int, bidirectional: bool, dropout: float, activation: str, normalization: str, classifier_hidden_dims: tuple[int, ...]) -> None:  # Initialize dynamic recurrent network
        super().__init__()  # Initialize native PyTorch module state
        self.architecture_variant = "lstm_sequence_classifier"  # Record factual architecture variant
        self.input_shape_ = (int(num_layers), int(n_features))  # Record batch-excluded recurrent input metadata
        recurrent_dropout = float(dropout) if int(num_layers) > 1 else 0.0  # Use PyTorch recurrent dropout only when valid
        self.lstm = nn.LSTM(input_size=int(n_features), hidden_size=int(hidden_size), num_layers=int(num_layers), dropout=recurrent_dropout, bidirectional=bool(bidirectional), batch_first=True)  # Build recurrent encoder
        head_input = int(hidden_size) * (2 if bool(bidirectional) else 1)  # Resolve classifier input width
        head_layers = []  # Accumulate classification-head layers
        for hidden_dim in tuple(classifier_hidden_dims):  # Build configured head hidden stack
            head_layers.append(nn.Linear(head_input, int(hidden_dim)))  # Add hidden projection
            head_layers.append(self.build_normalization(int(hidden_dim), normalization))  # Add configured normalization
            head_layers.append(self.build_activation(activation))  # Add configured activation
            head_layers.append(nn.Dropout(float(dropout)))  # Add configured dropout
            head_input = int(hidden_dim)  # Advance head width
        head_layers.append(nn.Linear(head_input, int(n_classes)))  # Add final class projection
        self.classifier_head = nn.Sequential(*head_layers)  # Store classification head

    def build_activation(self, activation: str) -> nn.Module:  # Build one supported activation module
        if activation == "relu":  # Select ReLU activation
            return nn.ReLU(inplace=False)  # Return ReLU module
        if activation == "gelu":  # Select GELU activation
            return nn.GELU()  # Return GELU module
        return nn.SiLU(inplace=False)  # Return SiLU module

    def build_normalization(self, width: int, normalization: str) -> nn.Module:  # Build one supported head normalization module
        if normalization == "layer_norm":  # Select layer normalization
            return nn.LayerNorm(int(width))  # Return layer normalization over hidden width
        return nn.Identity()  # Return identity when normalization is disabled

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # Compute class logits for one bounded sequence batch
        recurrent_output, _ = self.lstm(features)  # Encode batch by timesteps by features
        final_state = recurrent_output[:, -1, :]  # Select final timestep representation for causal label target
        return self.classifier_head(final_state)  # Return class logits


class LSTMClassifier(ClassifierMixin, BaseEstimator):  # Expose LSTM through sklearn classifier interfaces
    """
    Train and serve a mini-batch supervised LSTM sequence classifier.

    :param sequence_length: Required timestep count for each sequence window.
    :param sequence_stride: Sequence-window stride used by the upstream sequence builder.
    :param minimum_group_length: Minimum upstream group length required before windowing.
    :param chronological_field: Upstream chronological metadata field name.
    :param group_fields: Upstream grouping metadata field names.
    :param label_strategy: Sequence label strategy.
    :param mixed_label_window_policy: Mixed-label window policy used upstream.
    :param incomplete_window_policy: Incomplete-window policy used upstream.
    :param trusted_sequence_input: Whether supplied X is verified partition-local sequence data.
    :param hidden_size: LSTM hidden size.
    :param num_layers: LSTM recurrent layer count.
    :param bidirectional: Whether to use bidirectional recurrent context.
    :param dropout: Dropout probability.
    :param normalization: Classification-head normalization mode.
    :param classifier_hidden_dims: Classification-head hidden dimensions.
    :param activation: Classification-head activation.
    :param epochs: Maximum training epochs.
    :param batch_size: Training and validation mini-batch size.
    :param learning_rate: Optimizer learning rate.
    :param weight_decay: Optimizer weight decay.
    :param optimizer: Optimizer name.
    :param class_weight: Class-weighting mode.
    :param patience: Validation epochs without improvement before stopping.
    :param validation_fraction: Fraction split from input training sequences for validation.
    :param min_delta: Minimum validation-loss improvement.
    :param device: Requested device name or auto selection.
    :param allow_device_fallback: Whether explicit unavailable accelerators may fall back to CPU.
    :param prediction_batch_size: Inference mini-batch size.
    :param data_loader_workers: Batch worker policy, currently zero for CPU-backed slice streaming.
    :param mixed_precision: Whether to use CUDA mixed precision.
    :param gradient_clip_norm: Optional gradient clipping norm.
    :param torch_threads: Native PyTorch CPU thread count.
    :param random_state: Deterministic random seed.
    :param sequence_source_row_count: Optional upstream source-row count metadata.
    :param discarded_prefix_rows: Optional upstream discarded-prefix metadata.
    :param mixed_label_windows: Optional upstream mixed-label-window metadata.
    :param skipped_groups: Optional upstream skipped-group metadata.
    :return: Initialized sklearn-compatible classifier.
    """

    def __init__(self, sequence_length: int = 8, sequence_stride: int = 1, minimum_group_length: int = 8, chronological_field: Optional[str] = None, group_fields: tuple[str, ...] = (), label_strategy: str = "final_timestep", mixed_label_window_policy: str = "keep", incomplete_window_policy: str = "drop", trusted_sequence_input: bool = False, hidden_size: int = 128, num_layers: int = 1, bidirectional: bool = False, dropout: float = 0.1, normalization: str = "layer_norm", classifier_hidden_dims: tuple[int, ...] = (64,), activation: str = "relu", epochs: int = 20, batch_size: int = 1024, learning_rate: float = 0.001, weight_decay: float = 0.00001, optimizer: str = "adamw", class_weight: str = "none", patience: int = 3, validation_fraction: float = 0.1, min_delta: float = 0.0001, device: str = "auto", allow_device_fallback: bool = True, prediction_batch_size: int = 4096, data_loader_workers: int = 0, mixed_precision: bool = False, gradient_clip_norm: Optional[float] = None, torch_threads: int = 1, random_state: int = 42, sequence_source_row_count: Optional[int] = None, discarded_prefix_rows: Optional[int] = None, mixed_label_windows: Optional[int] = None, skipped_groups: Optional[int] = None) -> None:  # Store cloneable parameters without allocating model state
        self.sequence_length = sequence_length  # Store required timestep count
        self.sequence_stride = sequence_stride  # Store upstream sequence-window stride
        self.minimum_group_length = minimum_group_length  # Store upstream minimum group length
        self.chronological_field = chronological_field  # Store upstream chronological metadata field
        self.group_fields = group_fields  # Store upstream grouping metadata fields
        self.label_strategy = label_strategy  # Store sequence label strategy
        self.mixed_label_window_policy = mixed_label_window_policy  # Store mixed-label window policy
        self.incomplete_window_policy = incomplete_window_policy  # Store incomplete-window policy
        self.trusted_sequence_input = trusted_sequence_input  # Store external sequence validation flag
        self.hidden_size = hidden_size  # Store LSTM hidden size
        self.num_layers = num_layers  # Store recurrent layer count
        self.bidirectional = bidirectional  # Store recurrent direction mode
        self.dropout = dropout  # Store dropout probability
        self.normalization = normalization  # Store head normalization mode
        self.classifier_hidden_dims = classifier_hidden_dims  # Store classification head dimensions
        self.activation = activation  # Store head activation
        self.epochs = epochs  # Store maximum epoch count
        self.batch_size = batch_size  # Store bounded training batch size
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
        self.sequence_source_row_count = sequence_source_row_count  # Store optional source-row metadata
        self.discarded_prefix_rows = discarded_prefix_rows  # Store optional discarded-prefix metadata
        self.mixed_label_windows = mixed_label_windows  # Store optional mixed-label metadata
        self.skipped_groups = skipped_groups  # Store optional skipped-group metadata

    def resolve_device(self) -> torch.device:  # Resolve accelerator selection with explicit fallback policy
        requested = str(self.device).lower()  # Normalize requested device name
        if requested == "auto" and multiprocessing.current_process().name != "MainProcess":  # Avoid uncontrolled GPU use in feature-set workers
            requested = "cpu"  # Select CPU for automatic worker processes
        if requested == "auto":  # Select best available PyTorch device
            requested = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"  # Prefer CUDA, then Apple GPU, then CPU
        if requested.startswith("cuda") and not torch.cuda.is_available():  # Handle unavailable CUDA
            if not bool(self.allow_device_fallback):  # Honor strict accelerator policy
                raise RuntimeError("CUDA device was requested for LSTM but CUDA is unavailable")  # Raise clear accelerator error
            requested = "cpu"  # Preserve functional CPU execution
        if requested == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):  # Handle unavailable Apple GPU
            if not bool(self.allow_device_fallback):  # Honor strict accelerator policy
                raise RuntimeError("MPS device was requested for LSTM but MPS is unavailable")  # Raise clear accelerator error
            requested = "cpu"  # Preserve functional CPU execution
        return torch.device(requested)  # Return effective device

    def normalize_dimensions(self, values: tuple[int, ...]) -> tuple[int, ...]:  # Normalize configured head dimensions
        dimensions = tuple(int(value) for value in tuple(values))  # Convert YAML lists and tuples to integers
        if any(value < 1 for value in dimensions):  # Require positive hidden dimensions
            raise ValueError("classifier_hidden_dims must contain positive integers")  # Reject invalid hidden dimensions
        return dimensions  # Return normalized dimensions

    def validate_parameters(self) -> None:  # Validate architecture, sequence metadata, and training controls
        if int(self.sequence_length) < 2 or int(self.sequence_stride) < 1 or int(self.minimum_group_length) < int(self.sequence_length):  # Require meaningful sequence-window metadata
            raise ValueError("sequence_length must be at least 2, sequence_stride must be positive, and minimum_group_length must cover sequence_length")  # Reject invalid sequence metadata
        if str(self.label_strategy) != "final_timestep":  # Restrict to implemented target strategy
            raise ValueError("label_strategy must be final_timestep")  # Reject unsupported label strategy
        if str(self.mixed_label_window_policy) not in ("keep", "drop", "reject"):  # Restrict upstream mixed-label policies
            raise ValueError("mixed_label_window_policy must be keep, drop, or reject")  # Reject unsupported mixed-label policy
        if str(self.incomplete_window_policy) != "drop":  # Restrict incomplete-window behavior
            raise ValueError("incomplete_window_policy must be drop")  # Reject unsupported incomplete-window behavior
        if int(self.epochs) < 1 or int(self.batch_size) < 1 or int(self.prediction_batch_size) < 1 or int(self.torch_threads) < 1:  # Require usable iteration, batch, and thread values
            raise ValueError("epochs, batch_size, prediction_batch_size, and torch_threads must be positive")  # Reject unusable controls
        if int(self.data_loader_workers) != 0:  # Preserve CPU-backed array and memmap streaming policy
            raise ValueError("data_loader_workers must be 0 for LSTM CPU-backed slice streaming")  # Reject worker-based duplication risk
        if int(self.hidden_size) < 1 or int(self.num_layers) < 1:  # Require valid recurrent dimensions
            raise ValueError("hidden_size and num_layers must be positive")  # Reject invalid recurrent dimensions
        if not 0.0 <= float(self.dropout) < 1.0 or not 0.0 < float(self.validation_fraction) < 1.0:  # Require valid probability controls
            raise ValueError("dropout must be in [0, 1) and validation_fraction must be in (0, 1)")  # Reject invalid probability controls
        if float(self.learning_rate) <= 0.0 or float(self.weight_decay) < 0.0 or int(self.patience) < 1:  # Require valid optimizer controls
            raise ValueError("learning_rate must be positive; weight_decay must be non-negative; patience must be positive")  # Reject invalid optimizer controls
        if self.gradient_clip_norm is not None and float(self.gradient_clip_norm) <= 0.0:  # Require valid optional clipping norm
            raise ValueError("gradient_clip_norm must be positive when provided")  # Reject invalid clipping norm
        if str(self.activation) not in ("relu", "gelu", "silu"):  # Restrict activation to supported modules
            raise ValueError("activation must be relu, gelu, or silu")  # Reject unsupported activation names
        if str(self.normalization) not in ("layer_norm", "none"):  # Restrict normalization to batch-size-safe modules
            raise ValueError("normalization must be layer_norm or none")  # Reject unsupported normalization names
        if str(self.optimizer) not in ("adamw", "adam", "sgd"):  # Restrict optimizer to implemented options
            raise ValueError("optimizer must be adamw, adam, or sgd")  # Reject unsupported optimizer names
        if str(self.class_weight) not in ("none", "balanced"):  # Restrict class weighting to implemented options
            raise ValueError("class_weight must be none or balanced")  # Reject unsupported class-weighting mode

    def validate_sequence_input(self, features: Any, labels: Optional[np.ndarray] = None) -> np.ndarray:  # Validate pre-windowed sequence input without fabricating chronology
        sequence_features = features.to_numpy(copy=False) if hasattr(features, "to_numpy") else np.asarray(features)  # Preserve arrays and memmaps without unconditional duplication
        if not bool(self.trusted_sequence_input):  # Require explicit upstream sequence-validation signal
            raise ValueError("LSTM requires trusted_sequence_input=true after partition-local windows are built from verified chronology and grouping metadata")  # Reject unverified chronology
        if sequence_features.ndim != 3:  # Require true sequence tensors
            raise ValueError("LSTM requires X shaped (samples, timesteps, features); row-wise tabular matrices from stacking.py are not valid sequence input")  # Reject feature-as-time reshaping
        if sequence_features.shape[1] != int(self.sequence_length):  # Require configured timestep identity to match data
            raise ValueError(f"LSTM sequence_length={self.sequence_length} but X contains {sequence_features.shape[1]} timesteps")  # Reject incompatible cache/artifact identity
        if sequence_features.shape[0] < 2 or sequence_features.shape[2] < 1:  # Require usable sequence samples and per-step features
            raise ValueError("LSTM requires at least two sequence samples and one feature per timestep")  # Reject empty or degenerate inputs
        if labels is not None and sequence_features.shape[0] != labels.shape[0]:  # Require sample alignment
            raise ValueError("X sequence sample count must match y label count")  # Reject misaligned labels
        return sequence_features  # Return validated CPU-backed sequence features

    def build_network(self, n_features: int, n_classes: int) -> LSTMClassifierNetwork:  # Build network for fitted dimensions
        head_dims = self.normalize_dimensions(tuple(self.classifier_hidden_dims))  # Normalize classification-head dimensions
        return LSTMClassifierNetwork(n_features, n_classes, int(self.hidden_size), int(self.num_layers), bool(self.bidirectional), float(self.dropout), str(self.activation), str(self.normalization), head_dims)  # Return dynamically sized network

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

    def fit(self, X: Any, y: Any) -> "LSTMClassifier":  # Fit from CPU sequence arrays using bounded device batches
        self.validate_parameters()  # Validate public controls before allocation
        labels = np.asarray(y).reshape(-1)  # Normalize labels to one vector
        features = self.validate_sequence_input(X, labels)  # Validate genuine pre-windowed sequence input
        self.classes_ = np.unique(labels)  # Record dynamic class labels
        if self.classes_.shape[0] < 2:  # Require classification targets
            raise ValueError("LSTMClassifier requires at least two classes")  # Reject single-class input
        contiguous_classes = np.array_equal(self.classes_, np.arange(self.classes_.shape[0]))  # Detect pipeline integer encoding
        encoded_labels = np.asarray(labels, dtype=np.int64) if contiguous_classes else np.searchsorted(self.classes_, labels).astype(np.int64, copy=False)  # Reuse or map labels once on CPU
        class_counts = np.bincount(encoded_labels, minlength=int(self.classes_.shape[0]))  # Count encoded labels for stratified validation safety
        stratify_labels = encoded_labels if int(class_counts.min()) > 1 else None  # Use stratification only when every class can appear in both splits
        all_indices = np.arange(features.shape[0], dtype=np.int64)  # Build sequence indices without copying features
        train_indices, validation_indices = train_test_split(all_indices, test_size=float(self.validation_fraction), random_state=int(self.random_state), stratify=stratify_labels)  # Derive validation only from training input
        self.n_features_in_ = int(features.shape[2])  # Record per-timestep input width
        self.sequence_length_ = int(features.shape[1])  # Record fitted timestep count
        effective_device = self.resolve_device()  # Resolve compute device
        torch.set_num_threads(int(self.torch_threads))  # Bound native CPU threads
        torch.manual_seed(int(self.random_state))  # Seed CPU and active backend initialization
        if torch.cuda.is_available():  # Seed available CUDA devices
            torch.cuda.manual_seed_all(int(self.random_state))  # Make CUDA initialization reproducible
        torch.use_deterministic_algorithms(False)  # Allow nondeterministic CUDA kernels without reproducibility warnings
        dataset = LSTMSequenceIndexDataset(features, encoded_labels)  # Wrap CPU-backed sequence storage
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
            shuffled_indices = random_generator.permutation(train_indices)  # Shuffle sequence indices without copying features
            for batch_start in range(0, shuffled_indices.shape[0], int(self.batch_size)):  # Stream bounded training batches
                batch_indices = shuffled_indices[batch_start:batch_start + int(self.batch_size)]  # Select current training sequences
                batch_features = dataset.batch_features(batch_indices, effective_device)  # Transfer current feature batch only
                batch_labels = dataset.batch_labels(batch_indices, effective_device)  # Transfer current label batch only
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
                    batch_indices = validation_indices[batch_start:batch_start + int(self.batch_size)]  # Select current validation sequences
                    batch_features = dataset.batch_features(batch_indices, effective_device)  # Transfer current validation features only
                    batch_labels = dataset.batch_labels(batch_indices, effective_device)  # Transfer current validation labels only
                    validation_loss_sum += float(loss_function(network(batch_features), batch_labels).item()) * int(batch_indices.shape[0])  # Accumulate sample-weighted loss
            validation_loss = validation_loss_sum / float(validation_indices.shape[0])  # Compute mean validation loss
            epochs_completed = epoch_index + 1  # Record completed epoch count
            progress_callback = getattr(self, "progress_callback", None)  # Read optional stacking progress callback
            if callable(progress_callback):  # Report completed epochs only when stacking attached a callback
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
        head_dims = self.normalize_dimensions(tuple(self.classifier_hidden_dims))  # Normalize fitted head dimensions for metadata
        self.model_ = network.to("cpu")  # Keep fitted model CPU-safe for joblib
        self.device_ = str(effective_device)  # Record selected training device
        self.training_config_ = {"model_family": "lstm_sequence_classifier", "architecture_variant": "lstm_sequence_classifier", "device": self.device_, "input_shape": [int(self.sequence_length_), int(self.n_features_in_)], "input_dimension": int(self.n_features_in_), "sequence_length": int(self.sequence_length_), "sequence_stride": int(self.sequence_stride), "minimum_group_length": int(self.minimum_group_length), "chronological_field": self.chronological_field, "group_fields": list(tuple(self.group_fields)), "label_strategy": str(self.label_strategy), "mixed_label_window_policy": str(self.mixed_label_window_policy), "incomplete_window_policy": str(self.incomplete_window_policy), "trusted_sequence_input": bool(self.trusted_sequence_input), "sequence_source_row_count": self.sequence_source_row_count, "generated_sequences": int(features.shape[0]), "discarded_prefix_rows": self.discarded_prefix_rows, "mixed_label_windows": self.mixed_label_windows, "skipped_groups": self.skipped_groups, "class_count": int(self.classes_.shape[0]), "class_order": self.classes_.tolist(), "epochs_requested": int(self.epochs), "epochs_completed": epochs_completed, "best_epoch": best_epoch, "best_validation_loss": float(best_loss), "batch_size": int(self.batch_size), "prediction_batch_size": int(self.prediction_batch_size), "data_loader_workers": int(self.data_loader_workers), "torch_threads": int(self.torch_threads), "hidden_size": int(self.hidden_size), "num_layers": int(self.num_layers), "bidirectional": bool(self.bidirectional), "direction": "bidirectional" if bool(self.bidirectional) else "unidirectional", "classifier_hidden_dims": [int(value) for value in head_dims], "activation": str(self.activation), "normalization": str(self.normalization), "dropout": float(self.dropout), "learning_rate": float(self.learning_rate), "weight_decay": float(self.weight_decay), "optimizer": str(self.optimizer), "class_weight": str(self.class_weight), "patience": int(self.patience), "validation_fraction": float(self.validation_fraction), "mixed_precision": bool(self.mixed_precision) and effective_device.type == "cuda", "gradient_clip_norm": None if self.gradient_clip_norm is None else float(self.gradient_clip_norm), "random_state": int(self.random_state)}  # Record effective training configuration
        print(f"[LSTM] Variant: lstm_sequence_classifier | Device: {self.device_} | Input shape: (batch, {self.sequence_length_}, {self.n_features_in_}) | Classes: {len(self.classes_)} | Direction: {self.training_config_['direction']} | Epochs: {epochs_completed}/{self.epochs} | Best epoch: {best_epoch}")  # Log training outcome
        del optimizer, loss_function, scaler, best_state, train_indices, validation_indices, all_indices, dataset  # Release training-only state
        if effective_device.type == "cuda":  # Release unused cached CUDA allocations
            torch.cuda.empty_cache()  # Return cached CUDA blocks
        return self  # Return fitted estimator

    def predict_logits(self, X: Any) -> np.ndarray:  # Produce CPU logits through bounded batches
        if not hasattr(self, "model_") or not hasattr(self, "classes_"):  # Require fitted state
            raise RuntimeError("LSTMClassifier is not fitted")  # Reject inference before fit
        features = self.validate_sequence_input(X, None)  # Validate sequence input without labels
        if features.shape[1] != int(self.sequence_length_) or features.shape[2] != int(self.n_features_in_):  # Require fitted sequence shape
            raise ValueError(f"X must contain sequences shaped ({self.sequence_length_}, {self.n_features_in_})")  # Reject incompatible input schema
        effective_device = self.resolve_device()  # Resolve available inference device
        torch.set_num_threads(int(self.torch_threads))  # Preserve bounded CPU execution
        dataset = LSTMSequenceIndexDataset(features)  # Wrap CPU-backed sequence storage
        network = self.model_.to(effective_device)  # Move model parameters to inference device
        network.eval()  # Disable dropout during inference
        outputs = []  # Accumulate CPU batch logits
        with torch.inference_mode():  # Disable autograd during inference
            for batch_start in range(0, features.shape[0], int(self.prediction_batch_size)):  # Stream bounded inference batches
                batch_indices = np.arange(batch_start, min(batch_start + int(self.prediction_batch_size), features.shape[0]), dtype=np.int64)  # Select current sequence range
                batch_features = dataset.batch_features(batch_indices, effective_device)  # Transfer current batch only
                outputs.append(network(batch_features).cpu().numpy())  # Return current logits to CPU
        self.model_ = network.to("cpu")  # Restore CPU-safe fitted state
        self.inference_device_ = str(effective_device)  # Record effective inference device
        if effective_device.type == "cuda":  # Release unused cached CUDA allocations
            torch.cuda.empty_cache()  # Return cached CUDA blocks
        return np.concatenate(outputs, axis=0) if outputs else np.empty((0, len(self.classes_)), dtype=np.float32)  # Return CPU logits in sequence order

    def predict_proba(self, X: Any) -> np.ndarray:  # Return probabilities for stacking and explainability
        logits = self.predict_logits(X)  # Compute bounded CPU logits
        logits -= logits.max(axis=1, keepdims=True)  # Stabilize exponentiation
        probabilities = np.exp(logits)  # Convert logits to positive scores
        probabilities /= probabilities.sum(axis=1, keepdims=True)  # Normalize probabilities row-wise
        return probabilities  # Return sklearn-compatible probability matrix

    def predict(self, X: Any) -> np.ndarray:  # Return labels in fitted class order
        return self.classes_[np.argmax(self.predict_logits(X), axis=1)]  # Map maximum logits to original labels
