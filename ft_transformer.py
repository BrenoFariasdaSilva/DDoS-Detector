"""
Provide a sklearn-compatible FT-Transformer classifier.
"""
if __name__ in {"__main__", "__mp_main__"}:
    try:
        from setproctitle import setproctitle
        setproctitle(f"DDoS-{__file__.rsplit('/', 1)[-1].rsplit('.', 1)[0]}")
    except ImportError:
        pass


import math  # Provide validation-loss infinity
from typing import Any  # Type public estimator inputs

import numpy as np  # Process CPU-resident tabular arrays and memmaps
import torch  # Reuse the repository deep-learning backend
from sklearn.base import BaseEstimator, ClassifierMixin  # Provide sklearn estimator behavior
from sklearn.model_selection import train_test_split  # Derive validation rows only from training input
from torch import nn  # Build the FT-Transformer network with native layers


class FTTransformerNetwork(nn.Module):  # Implement numeric feature tokenization and Transformer encoding
    """
    Encode row-wise numeric tabular features with an FT-Transformer network.

    :param n_features: Number of numeric input features.
    :param n_classes: Number of output classes.
    :param token_dim: Embedding width for every feature token.
    :param n_blocks: Number of Transformer encoder blocks.
    :param n_heads: Number of attention heads per block.
    :param dropout: Dropout probability used by Transformer blocks.
    :param ffn_multiplier: Feed-forward width multiplier relative to token width.
    :param activation: Transformer feed-forward activation name.
    :return: Initialized FT-Transformer network.
    """

    def __init__(self, n_features: int, n_classes: int, token_dim: int, n_blocks: int, n_heads: int, dropout: float, ffn_multiplier: float, activation: str) -> None:  # Initialize dynamic network dimensions
        super().__init__()  # Initialize native PyTorch module state
        self.feature_weight = nn.Parameter(torch.empty(n_features, token_dim))  # Learn one projection vector per feature
        self.feature_bias = nn.Parameter(torch.empty(n_features, token_dim))  # Learn one additive bias per feature
        self.cls_token = nn.Parameter(torch.empty(1, 1, token_dim))  # Learn one shared classification token
        encoder_layer = nn.TransformerEncoderLayer(d_model=token_dim, nhead=n_heads, dim_feedforward=max(token_dim, int(token_dim * ffn_multiplier)), dropout=dropout, activation=activation, batch_first=True, norm_first=True)  # Build one pre-normalized attention block
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_blocks, enable_nested_tensor=False)  # Stack configured Transformer blocks
        self.output_norm = nn.LayerNorm(token_dim)  # Normalize final classification token
        self.head = nn.Linear(token_dim, n_classes)  # Project classification token to dynamic class logits
        nn.init.normal_(self.feature_weight, std=0.02)  # Initialize numeric token weights after seeding
        nn.init.normal_(self.feature_bias, std=0.02)  # Initialize numeric token biases after seeding
        nn.init.normal_(self.cls_token, std=0.02)  # Initialize classification token after seeding

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # Compute class logits for one bounded batch
        feature_tokens = features.unsqueeze(-1) * self.feature_weight.unsqueeze(0) + self.feature_bias.unsqueeze(0)  # Convert each scalar feature into its own token
        cls_tokens = self.cls_token.expand(features.shape[0], -1, -1)  # Expand classification token without dataset-sized state
        encoded = self.encoder(torch.cat((cls_tokens, feature_tokens), dim=1))  # Apply attention across tokens within each row
        return self.head(self.output_norm(encoded[:, 0]))  # Return classification-token logits


class FTTransformerClassifier(ClassifierMixin, BaseEstimator):  # Expose FT-Transformer through sklearn classifier interfaces
    """
    Train and serve a mini-batch FT-Transformer classifier.

    :param epochs: Maximum training epochs.
    :param batch_size: Training and validation mini-batch size.
    :param token_dim: Embedding width for every numeric feature token.
    :param n_blocks: Number of Transformer encoder blocks.
    :param n_heads: Number of attention heads per block.
    :param dropout: Dropout probability used by Transformer blocks.
    :param ffn_multiplier: Feed-forward width multiplier relative to token width.
    :param learning_rate: AdamW learning rate.
    :param weight_decay: AdamW weight decay.
    :param patience: Validation epochs without improvement before stopping.
    :param validation_fraction: Fraction split from input training rows for validation.
    :param min_delta: Minimum validation-loss improvement.
    :param activation: Transformer feed-forward activation name.
    :param device: Requested device name or auto selection.
    :param prediction_batch_size: Inference mini-batch size.
    :param torch_threads: Native PyTorch CPU thread count.
    :param random_state: Deterministic random seed.
    :return: Initialized sklearn-compatible classifier.
    """

    def __init__(self, epochs: int = 20, batch_size: int = 1024, token_dim: int = 32, n_blocks: int = 2, n_heads: int = 4, dropout: float = 0.1, ffn_multiplier: float = 2.0, learning_rate: float = 0.001, weight_decay: float = 0.00001, patience: int = 3, validation_fraction: float = 0.1, min_delta: float = 0.0001, activation: str = "gelu", device: str = "auto", prediction_batch_size: int = 4096, torch_threads: int = 1, random_state: int = 42) -> None:  # Store cloneable parameters without allocating model state
        self.epochs = epochs  # Store maximum epoch count
        self.batch_size = batch_size  # Store bounded training batch size
        self.token_dim = token_dim  # Store token embedding width
        self.n_blocks = n_blocks  # Store Transformer block count
        self.n_heads = n_heads  # Store attention head count
        self.dropout = dropout  # Store Transformer dropout probability
        self.ffn_multiplier = ffn_multiplier  # Store feed-forward width multiplier
        self.learning_rate = learning_rate  # Store optimizer learning rate
        self.weight_decay = weight_decay  # Store optimizer weight decay
        self.patience = patience  # Store early-stopping patience
        self.validation_fraction = validation_fraction  # Store internal validation fraction
        self.min_delta = min_delta  # Store minimum validation-loss improvement
        self.activation = activation  # Store Transformer activation name
        self.device = device  # Store requested compute device
        self.prediction_batch_size = prediction_batch_size  # Store bounded inference batch size
        self.torch_threads = torch_threads  # Store bounded native CPU thread count
        self.random_state = random_state  # Store deterministic seed

    def resolve_device(self) -> torch.device:  # Resolve GPU acceleration with CPU fallback
        requested = str(self.device).lower()  # Normalize requested device name
        if requested == "auto":  # Select best available PyTorch device
            requested = "cuda" if torch.cuda.is_available() else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"  # Prefer CUDA, then Apple GPU, then CPU
        if requested.startswith("cuda") and not torch.cuda.is_available():  # Handle unavailable CUDA
            requested = "cpu"  # Preserve functional CPU execution
        if requested == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):  # Handle unavailable Apple GPU
            requested = "cpu"  # Preserve functional CPU execution
        return torch.device(requested)  # Return effective device

    def validate_parameters(self) -> None:  # Validate architecture and training controls
        if int(self.epochs) < 1 or int(self.batch_size) < 1 or int(self.prediction_batch_size) < 1 or int(self.torch_threads) < 1:  # Require positive iteration, batch, and thread values
            raise ValueError("epochs, batch_size, prediction_batch_size, and torch_threads must be positive")  # Reject unusable controls
        if int(self.token_dim) < 1 or int(self.n_blocks) < 1 or int(self.n_heads) < 1 or int(self.token_dim) % int(self.n_heads) != 0:  # Require valid attention dimensions
            raise ValueError("token_dim must be positive and divisible by n_heads; n_blocks must be positive")  # Reject invalid Transformer dimensions
        if not 0.0 <= float(self.dropout) < 1.0 or not 0.0 < float(self.validation_fraction) < 1.0:  # Require valid probability controls
            raise ValueError("dropout must be in [0, 1) and validation_fraction must be in (0, 1)")  # Reject invalid probability controls
        if float(self.learning_rate) <= 0.0 or float(self.weight_decay) < 0.0 or int(self.patience) < 1 or float(self.ffn_multiplier) <= 0.0:  # Require valid optimizer controls
            raise ValueError("learning_rate and ffn_multiplier must be positive; weight_decay must be non-negative; patience must be positive")  # Reject invalid optimizer controls
        if self.activation not in ("relu", "gelu"):  # Restrict activation to native Transformer options
            raise ValueError("activation must be relu or gelu")  # Reject unsupported activation names

    def build_network(self, n_features: int, n_classes: int) -> FTTransformerNetwork:  # Build network for fitted dimensions
        return FTTransformerNetwork(n_features, n_classes, int(self.token_dim), int(self.n_blocks), int(self.n_heads), float(self.dropout), float(self.ffn_multiplier), str(self.activation))  # Return dynamically sized network

    def fit(self, X: Any, y: Any) -> "FTTransformerClassifier":  # Fit from CPU arrays using bounded device batches
        self.validate_parameters()  # Validate public controls before allocation
        features = np.asarray(X)  # Preserve arrays and memmaps without unconditional duplication
        labels = np.asarray(y).reshape(-1)  # Normalize labels to one vector
        if features.ndim != 2 or features.shape[0] != labels.shape[0] or features.shape[0] < 2 or features.shape[1] < 1:  # Require aligned tabular inputs
            raise ValueError("X must be a non-empty two-dimensional array aligned with y")  # Reject invalid input
        self.classes_ = np.unique(labels)  # Record dynamic class labels
        if self.classes_.shape[0] < 2:  # Require classification targets
            raise ValueError("FTTransformerClassifier requires at least two classes")  # Reject single-class input
        contiguous_classes = np.array_equal(self.classes_, np.arange(self.classes_.shape[0]))  # Detect pipeline integer encoding
        encoded_labels = np.asarray(labels, dtype=np.int64) if contiguous_classes else np.searchsorted(self.classes_, labels).astype(np.int64, copy=False)  # Reuse or map labels once on CPU
        all_indices = np.arange(features.shape[0], dtype=np.int64)  # Build row indices without copying features
        train_indices, validation_indices = train_test_split(all_indices, test_size=float(self.validation_fraction), random_state=int(self.random_state), stratify=encoded_labels)  # Derive validation only from training input
        self.n_features_in_ = int(features.shape[1])  # Record dynamic input width
        effective_device = self.resolve_device()  # Resolve compute device
        torch.set_num_threads(int(self.torch_threads))  # Bound native CPU threads
        torch.manual_seed(int(self.random_state))  # Seed CPU and active backend initialization
        if torch.cuda.is_available():  # Seed available CUDA devices
            torch.cuda.manual_seed_all(int(self.random_state))  # Make CUDA initialization reproducible
        torch.use_deterministic_algorithms(False)  # Allow nondeterministic CUDA kernels without reproducibility warnings
        network = self.build_network(self.n_features_in_, int(self.classes_.shape[0])).to(effective_device)  # Allocate model parameters on selected device
        optimizer = torch.optim.AdamW(network.parameters(), lr=float(self.learning_rate), weight_decay=float(self.weight_decay))  # Build configured optimizer
        loss_function = nn.CrossEntropyLoss()  # Use multiclass cross-entropy
        random_generator = np.random.default_rng(int(self.random_state))  # Build deterministic batch-order generator
        best_loss = math.inf  # Initialize validation target
        best_state = None  # Retain only best model state on CPU
        best_epoch = 0  # Record best validation epoch
        stale_epochs = 0  # Count epochs without improvement
        epochs_completed = 0  # Record effective completed epochs
        for epoch_index in range(int(self.epochs)):  # Train until limit or early stopping
            network.train()  # Enable training-time dropout
            shuffled_indices = random_generator.permutation(train_indices)  # Shuffle row indices without copying features
            for batch_start in range(0, shuffled_indices.shape[0], int(self.batch_size)):  # Stream bounded training batches
                batch_indices = shuffled_indices[batch_start:batch_start + int(self.batch_size)]  # Select current training rows
                batch_features = torch.as_tensor(np.asarray(features[batch_indices], dtype=np.float32), device=effective_device)  # Transfer current feature batch only
                batch_labels = torch.as_tensor(encoded_labels[batch_indices], dtype=torch.long, device=effective_device)  # Transfer current label batch only
                optimizer.zero_grad(set_to_none=True)  # Release prior gradients
                loss = loss_function(network(batch_features), batch_labels)  # Compute current batch loss
                loss.backward()  # Backpropagate current batch
                optimizer.step()  # Update model parameters
            network.eval()  # Disable dropout for validation
            validation_loss_sum = 0.0  # Accumulate weighted validation loss
            with torch.inference_mode():  # Disable autograd during validation
                for batch_start in range(0, validation_indices.shape[0], int(self.batch_size)):  # Stream bounded validation batches
                    batch_indices = validation_indices[batch_start:batch_start + int(self.batch_size)]  # Select current validation rows
                    batch_features = torch.as_tensor(np.asarray(features[batch_indices], dtype=np.float32), device=effective_device)  # Transfer current validation features only
                    batch_labels = torch.as_tensor(encoded_labels[batch_indices], dtype=torch.long, device=effective_device)  # Transfer current validation labels only
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
        self.model_ = network.to("cpu")  # Keep fitted model CPU-safe for joblib
        self.device_ = str(effective_device)  # Record selected training device
        self.training_config_ = {"device": self.device_, "epochs_requested": int(self.epochs), "epochs_completed": epochs_completed, "best_epoch": best_epoch, "best_validation_loss": float(best_loss), "batch_size": int(self.batch_size), "prediction_batch_size": int(self.prediction_batch_size), "torch_threads": int(self.torch_threads), "token_dim": int(self.token_dim), "n_blocks": int(self.n_blocks), "n_heads": int(self.n_heads), "dropout": float(self.dropout), "ffn_multiplier": float(self.ffn_multiplier), "learning_rate": float(self.learning_rate), "weight_decay": float(self.weight_decay), "patience": int(self.patience), "validation_fraction": float(self.validation_fraction), "random_state": int(self.random_state)}  # Record effective training configuration
        print(f"[FT-TRANSFORMER] Device: {self.device_} | Features: {self.n_features_in_} | Classes: {len(self.classes_)} | Epochs: {epochs_completed}/{self.epochs} | Best epoch: {best_epoch}")  # Log training outcome
        del optimizer, loss_function, best_state, train_indices, validation_indices, all_indices  # Release training-only state
        if effective_device.type == "cuda":  # Release unused cached CUDA allocations
            torch.cuda.empty_cache()  # Return cached CUDA blocks
        return self  # Return fitted estimator

    def predict_logits(self, X: Any) -> np.ndarray:  # Produce CPU logits through bounded batches
        if not hasattr(self, "model_") or not hasattr(self, "classes_"):  # Require fitted state
            raise RuntimeError("FTTransformerClassifier is not fitted")  # Reject inference before fit
        features = np.asarray(X)  # Preserve arrays and memmaps without unconditional duplication
        if features.ndim != 2 or features.shape[1] != int(self.n_features_in_):  # Require fitted feature width
            raise ValueError(f"X must contain {self.n_features_in_} features")  # Reject incompatible input schema
        effective_device = self.resolve_device()  # Resolve available inference device
        torch.set_num_threads(int(self.torch_threads))  # Preserve bounded CPU execution
        network = self.model_.to(effective_device)  # Move model parameters to inference device
        network.eval()  # Disable dropout during inference
        outputs = []  # Accumulate CPU batch logits
        with torch.inference_mode():  # Disable autograd during inference
            for batch_start in range(0, features.shape[0], int(self.prediction_batch_size)):  # Stream bounded inference batches
                batch_features = torch.as_tensor(np.asarray(features[batch_start:batch_start + int(self.prediction_batch_size)], dtype=np.float32), device=effective_device)  # Transfer current batch only
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
