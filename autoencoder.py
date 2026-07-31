"""
Provide a sklearn-compatible supervised Autoencoder classifier.
"""

import math  # Provide validation-loss infinity
import multiprocessing  # Detect automatic device selection inside worker processes
from typing import Any, Optional  # Type public estimator inputs and optional controls

import numpy as np  # Process CPU-resident tabular arrays and memmaps
import torch  # Reuse the repository deep-learning backend
from sklearn.base import BaseEstimator, ClassifierMixin  # Provide sklearn estimator behavior
from sklearn.model_selection import train_test_split  # Derive validation rows only from training input
from torch import nn  # Build encoder, decoder, and classifier head layers


class AutoencoderEncoder(nn.Module):  # Implement numeric encoder
    """
    Encode numeric tabular rows into a latent bottleneck.

    :param n_features: Number of numeric input features.
    :param hidden_dims: Encoder hidden dimensions.
    :param latent_dim: Latent bottleneck dimension.
    :param activation: Activation function name.
    :param normalization: Normalization layer name.
    :param dropout: Dropout probability.
    :return: Initialized encoder module.
    """

    def __init__(self, n_features: int, hidden_dims: tuple[int, ...], latent_dim: int, activation: str, normalization: str, dropout: float) -> None:  # Initialize encoder layers
        super().__init__()  # Initialize native PyTorch module state
        layers = []  # Accumulate encoder layers
        input_dim = int(n_features)  # Track current layer input width
        for hidden_dim in tuple(hidden_dims):  # Build configured hidden projection stack
            layers.append(nn.Linear(input_dim, int(hidden_dim)))  # Add hidden projection
            layers.append(self.build_normalization(int(hidden_dim), normalization))  # Add configured normalization
            layers.append(self.build_activation(activation))  # Add configured activation
            layers.append(nn.Dropout(float(dropout)))  # Add configured dropout
            input_dim = int(hidden_dim)  # Advance current width
        layers.append(nn.Linear(input_dim, int(latent_dim)))  # Add latent bottleneck projection
        self.network = nn.Sequential(*layers)  # Store encoder sequence

    def build_activation(self, activation: str) -> nn.Module:  # Build one supported activation module
        if activation == "relu":  # Select ReLU activation
            return nn.ReLU(inplace=False)  # Return ReLU module
        if activation == "gelu":  # Select GELU activation
            return nn.GELU()  # Return GELU module
        return nn.SiLU(inplace=False)  # Return SiLU module

    def build_normalization(self, width: int, normalization: str) -> nn.Module:  # Build one supported normalization module
        if normalization == "batch_norm":  # Select batch normalization
            return nn.BatchNorm1d(width)  # Return batch normalization over hidden width
        if normalization == "layer_norm":  # Select layer normalization
            return nn.LayerNorm(width)  # Return layer normalization over hidden width
        return nn.Identity()  # Return identity when normalization is disabled

    def forward(self, features: torch.Tensor) -> torch.Tensor:  # Compute latent representation
        return self.network(features)  # Return encoded bottleneck activations


class AutoencoderDecoder(nn.Module):  # Implement numeric decoder
    """
    Decode latent bottleneck activations back to input feature space.

    :param latent_dim: Latent bottleneck dimension.
    :param hidden_dims: Decoder hidden dimensions.
    :param n_features: Number of numeric output features.
    :param activation: Activation function name.
    :param normalization: Normalization layer name.
    :param dropout: Dropout probability.
    :return: Initialized decoder module.
    """

    def __init__(self, latent_dim: int, hidden_dims: tuple[int, ...], n_features: int, activation: str, normalization: str, dropout: float) -> None:  # Initialize decoder layers
        super().__init__()  # Initialize native PyTorch module state
        layers = []  # Accumulate decoder layers
        input_dim = int(latent_dim)  # Track current layer input width
        for hidden_dim in tuple(hidden_dims):  # Build configured decoder stack
            layers.append(nn.Linear(input_dim, int(hidden_dim)))  # Add hidden projection
            layers.append(self.build_normalization(int(hidden_dim), normalization))  # Add configured normalization
            layers.append(self.build_activation(activation))  # Add configured activation
            layers.append(nn.Dropout(float(dropout)))  # Add configured dropout
            input_dim = int(hidden_dim)  # Advance current width
        layers.append(nn.Linear(input_dim, int(n_features)))  # Add reconstruction projection
        self.network = nn.Sequential(*layers)  # Store decoder sequence

    def build_activation(self, activation: str) -> nn.Module:  # Build one supported activation module
        if activation == "relu":  # Select ReLU activation
            return nn.ReLU(inplace=False)  # Return ReLU module
        if activation == "gelu":  # Select GELU activation
            return nn.GELU()  # Return GELU module
        return nn.SiLU(inplace=False)  # Return SiLU module

    def build_normalization(self, width: int, normalization: str) -> nn.Module:  # Build one supported normalization module
        if normalization == "batch_norm":  # Select batch normalization
            return nn.BatchNorm1d(width)  # Return batch normalization over hidden width
        if normalization == "layer_norm":  # Select layer normalization
            return nn.LayerNorm(width)  # Return layer normalization over hidden width
        return nn.Identity()  # Return identity when normalization is disabled

    def forward(self, latent: torch.Tensor) -> torch.Tensor:  # Compute reconstruction output
        return self.network(latent)  # Return reconstructed feature rows


class AutoencoderClassifierHead(nn.Module):  # Implement multiclass classifier head
    """
    Classify latent bottleneck activations into multiclass logits.

    :param latent_dim: Latent bottleneck dimension.
    :param hidden_dims: Classification-head hidden dimensions.
    :param n_classes: Number of output classes.
    :param activation: Activation function name.
    :param normalization: Normalization layer name.
    :param dropout: Dropout probability.
    :return: Initialized classification head.
    """

    def __init__(self, latent_dim: int, hidden_dims: tuple[int, ...], n_classes: int, activation: str, normalization: str, dropout: float) -> None:  # Initialize classification head layers
        super().__init__()  # Initialize native PyTorch module state
        layers = []  # Accumulate classification layers
        input_dim = int(latent_dim)  # Track current layer input width
        for hidden_dim in tuple(hidden_dims):  # Build configured head stack
            layers.append(nn.Linear(input_dim, int(hidden_dim)))  # Add hidden projection
            layers.append(self.build_normalization(int(hidden_dim), normalization))  # Add configured normalization
            layers.append(self.build_activation(activation))  # Add configured activation
            layers.append(nn.Dropout(float(dropout)))  # Add configured dropout
            input_dim = int(hidden_dim)  # Advance current width
        layers.append(nn.Linear(input_dim, int(n_classes)))  # Add multiclass logits projection
        self.network = nn.Sequential(*layers)  # Store classification sequence

    def build_activation(self, activation: str) -> nn.Module:  # Build one supported activation module
        if activation == "relu":  # Select ReLU activation
            return nn.ReLU(inplace=False)  # Return ReLU module
        if activation == "gelu":  # Select GELU activation
            return nn.GELU()  # Return GELU module
        return nn.SiLU(inplace=False)  # Return SiLU module

    def build_normalization(self, width: int, normalization: str) -> nn.Module:  # Build one supported normalization module
        if normalization == "batch_norm":  # Select batch normalization
            return nn.BatchNorm1d(width)  # Return batch normalization over hidden width
        if normalization == "layer_norm":  # Select layer normalization
            return nn.LayerNorm(width)  # Return layer normalization over hidden width
        return nn.Identity()  # Return identity when normalization is disabled

    def forward(self, latent: torch.Tensor) -> torch.Tensor:  # Compute class logits
        return self.network(latent)  # Return logits in class-column order


class AutoencoderClassifierNetwork(nn.Module):  # Implement complete supervised autoencoder classifier
    """
    Reconstruct numeric inputs and classify labels from the same latent representation.

    :param n_features: Number of numeric input features.
    :param n_classes: Number of output classes.
    :param encoder_hidden_dims: Encoder hidden dimensions.
    :param decoder_hidden_dims: Decoder hidden dimensions.
    :param latent_dim: Latent bottleneck dimension.
    :param classifier_hidden_dims: Classification-head hidden dimensions.
    :param activation: Activation function name.
    :param normalization: Normalization layer name.
    :param dropout: Dropout probability.
    :return: Initialized supervised autoencoder classifier.
    """

    def __init__(self, n_features: int, n_classes: int, encoder_hidden_dims: tuple[int, ...], decoder_hidden_dims: tuple[int, ...], latent_dim: int, classifier_hidden_dims: tuple[int, ...], activation: str, normalization: str, dropout: float) -> None:  # Initialize complete network
        super().__init__()  # Initialize native PyTorch module state
        self.encoder = AutoencoderEncoder(n_features, encoder_hidden_dims, latent_dim, activation, normalization, dropout)  # Build encoder
        self.decoder = AutoencoderDecoder(latent_dim, decoder_hidden_dims, n_features, activation, normalization, dropout)  # Build decoder
        self.classifier_head = AutoencoderClassifierHead(latent_dim, classifier_hidden_dims, n_classes, activation, normalization, dropout)  # Build supervised classification head

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # Compute reconstruction, logits, and latent activations
        latent = self.encoder(features)  # Encode current feature batch
        reconstruction = self.decoder(latent)  # Reconstruct current feature batch
        logits = self.classifier_head(latent)  # Classify current latent batch
        return reconstruction, logits, latent  # Return all supervised autoencoder outputs


class AutoencoderClassifier(ClassifierMixin, BaseEstimator):  # Expose Autoencoder through sklearn classifier interfaces
    """
    Train and serve a supervised mini-batch Autoencoder classifier.

    :param training_strategy: Training strategy name.
    :param epochs: Maximum joint-training epochs.
    :param batch_size: Training and validation mini-batch size.
    :param encoder_hidden_dims: Encoder hidden dimensions.
    :param decoder_hidden_dims: Decoder hidden dimensions or empty tuple for mirrored encoder.
    :param latent_dim: Explicit latent bottleneck dimension, or None to use latent_ratio.
    :param latent_ratio: Ratio used to derive latent dimension when latent_dim is None.
    :param classifier_hidden_dims: Classification-head hidden dimensions.
    :param activation: Activation function name.
    :param normalization: Normalization layer name.
    :param dropout: Dropout probability.
    :param denoising_noise: Training-only Gaussian input noise level.
    :param reconstruction_loss: Reconstruction loss type.
    :param reconstruction_loss_weight: Reconstruction objective weight.
    :param classification_loss_weight: Classification objective weight.
    :param learning_rate: AdamW learning rate.
    :param weight_decay: AdamW weight decay.
    :param patience: Validation epochs without improvement before stopping.
    :param validation_fraction: Fraction split from input training rows for validation.
    :param min_delta: Minimum validation-loss improvement.
    :param device: Requested device name or auto selection.
    :param allow_device_fallback: Whether explicit unavailable accelerators may fall back to CPU.
    :param prediction_batch_size: Inference mini-batch size.
    :param data_loader_workers: Batch worker policy, currently zero for CPU-backed slice streaming.
    :param torch_threads: Native PyTorch CPU thread count.
    :param random_state: Deterministic random seed.
    :return: Initialized sklearn-compatible classifier.
    """

    def __init__(self, training_strategy: str = "joint", epochs: int = 20, batch_size: int = 1024, encoder_hidden_dims: tuple[int, ...] = (128, 64), decoder_hidden_dims: tuple[int, ...] = (), latent_dim: Optional[int] = None, latent_ratio: float = 0.25, classifier_hidden_dims: tuple[int, ...] = (64,), activation: str = "relu", normalization: str = "batch_norm", dropout: float = 0.1, denoising_noise: float = 0.0, reconstruction_loss: str = "mse", reconstruction_loss_weight: float = 0.5, classification_loss_weight: float = 1.0, learning_rate: float = 0.001, weight_decay: float = 0.00001, patience: int = 3, validation_fraction: float = 0.1, min_delta: float = 0.0001, device: str = "auto", allow_device_fallback: bool = True, prediction_batch_size: int = 4096, data_loader_workers: int = 0, torch_threads: int = 1, random_state: int = 42) -> None:  # Store cloneable parameters without allocating model state
        self.training_strategy = training_strategy  # Store selected training strategy
        self.epochs = epochs  # Store maximum epoch count
        self.batch_size = batch_size  # Store bounded training batch size
        self.encoder_hidden_dims = encoder_hidden_dims  # Store encoder hidden dimensions
        self.decoder_hidden_dims = decoder_hidden_dims  # Store decoder hidden dimensions
        self.latent_dim = latent_dim  # Store optional explicit latent dimension
        self.latent_ratio = latent_ratio  # Store latent ratio fallback
        self.classifier_hidden_dims = classifier_hidden_dims  # Store classification head dimensions
        self.activation = activation  # Store activation selection
        self.normalization = normalization  # Store normalization selection
        self.dropout = dropout  # Store dropout probability
        self.denoising_noise = denoising_noise  # Store training-only denoising noise level
        self.reconstruction_loss = reconstruction_loss  # Store reconstruction loss type
        self.reconstruction_loss_weight = reconstruction_loss_weight  # Store reconstruction objective weight
        self.classification_loss_weight = classification_loss_weight  # Store classification objective weight
        self.learning_rate = learning_rate  # Store optimizer learning rate
        self.weight_decay = weight_decay  # Store optimizer weight decay
        self.patience = patience  # Store early-stopping patience
        self.validation_fraction = validation_fraction  # Store internal validation fraction
        self.min_delta = min_delta  # Store minimum validation-loss improvement
        self.device = device  # Store requested compute device
        self.allow_device_fallback = allow_device_fallback  # Store accelerator fallback policy
        self.prediction_batch_size = prediction_batch_size  # Store bounded inference batch size
        self.data_loader_workers = data_loader_workers  # Store batch worker policy
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
                raise RuntimeError("CUDA device was requested for Autoencoder (AE) but CUDA is unavailable")  # Raise clear accelerator error
            requested = "cpu"  # Preserve functional CPU execution
        if requested == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):  # Handle unavailable Apple GPU
            if not bool(self.allow_device_fallback):  # Honor strict accelerator policy
                raise RuntimeError("MPS device was requested for Autoencoder (AE) but MPS is unavailable")  # Raise clear accelerator error
            requested = "cpu"  # Preserve functional CPU execution
        return torch.device(requested)  # Return effective device

    def normalize_dimensions(self, values: tuple[int, ...]) -> tuple[int, ...]:  # Normalize configured hidden dimensions
        dimensions = tuple(int(value) for value in tuple(values))  # Convert YAML lists and tuples to integers
        if any(value < 1 for value in dimensions):  # Require positive hidden dimensions
            raise ValueError("hidden dimensions must be positive integers")  # Reject invalid hidden dimensions
        return dimensions  # Return normalized dimensions

    def validate_parameters(self) -> None:  # Validate architecture and training controls
        if str(self.training_strategy) != "joint":  # Restrict to implemented supervised joint strategy
            raise ValueError("training_strategy must be joint")  # Reject unsupported strategy names
        if int(self.epochs) < 1 or int(self.batch_size) < 1 or int(self.prediction_batch_size) < 1 or int(self.torch_threads) < 1:  # Require usable iteration, batch, and thread values
            raise ValueError("epochs, batch_size, prediction_batch_size, and torch_threads must be positive")  # Reject unusable controls
        if int(self.data_loader_workers) != 0:  # Preserve CPU-backed array and memmap streaming policy
            raise ValueError("data_loader_workers must be 0 for Autoencoder (AE) CPU-backed slice streaming")  # Reject worker-based duplication risk
        if not 0.0 <= float(self.dropout) < 1.0 or not 0.0 <= float(self.denoising_noise) < 1.0 or not 0.0 < float(self.validation_fraction) < 1.0:  # Require valid probability controls
            raise ValueError("dropout and denoising_noise must be in [0, 1); validation_fraction must be in (0, 1)")  # Reject invalid probability controls
        if float(self.learning_rate) <= 0.0 or float(self.weight_decay) < 0.0 or int(self.patience) < 1:  # Require valid optimizer controls
            raise ValueError("learning_rate must be positive; weight_decay must be non-negative; patience must be positive")  # Reject invalid optimizer controls
        if float(self.reconstruction_loss_weight) < 0.0 or float(self.classification_loss_weight) <= 0.0:  # Require meaningful supervised objective weights
            raise ValueError("reconstruction_loss_weight must be non-negative and classification_loss_weight must be positive")  # Reject invalid objective weights
        if str(self.activation) not in ("relu", "gelu", "silu"):  # Restrict activation to supported modules
            raise ValueError("activation must be relu, gelu, or silu")  # Reject unsupported activation names
        if str(self.normalization) not in ("batch_norm", "layer_norm", "none"):  # Restrict normalization to supported modules
            raise ValueError("normalization must be batch_norm, layer_norm, or none")  # Reject unsupported normalization names
        if str(self.reconstruction_loss) not in ("mse", "mae"):  # Restrict reconstruction loss to implemented options
            raise ValueError("reconstruction_loss must be mse or mae")  # Reject unsupported reconstruction loss

    def resolve_latent_dim(self, n_features: int) -> int:  # Resolve effective latent bottleneck dimension
        if int(n_features) < 2:  # Require room for a bottleneck
            raise ValueError("Autoencoder (AE) requires at least two input features for a latent bottleneck")  # Reject non-bottleneck input
        if self.latent_dim is not None:  # Use explicit latent dimension when provided
            resolved = int(self.latent_dim)  # Convert configured latent dimension
            if resolved < 1 or resolved >= int(n_features):  # Require strict bottleneck
                raise ValueError(f"latent_dim must be between 1 and {int(n_features) - 1} for {int(n_features)} input features")  # Reject invalid explicit latent dimension
            return resolved  # Return explicit bottleneck dimension
        if not 0.0 < float(self.latent_ratio) < 1.0:  # Require ratio that can form a bottleneck
            raise ValueError("latent_ratio must be in (0, 1)")  # Reject invalid latent ratio
        resolved = int(math.floor(int(n_features) * float(self.latent_ratio)))  # Resolve ratio deterministically
        if resolved < 1 or resolved >= int(n_features):  # Require valid ratio-derived bottleneck
            raise ValueError(f"latent_ratio resolves to invalid latent dimension {resolved} for {int(n_features)} input features")  # Reject invalid derived latent dimension
        return resolved  # Return derived bottleneck dimension

    def resolve_decoder_hidden_dims(self, encoder_dims: tuple[int, ...]) -> tuple[int, ...]:  # Resolve decoder hidden dimensions
        decoder_dims = self.normalize_dimensions(tuple(self.decoder_hidden_dims))  # Normalize configured decoder dimensions
        return tuple(reversed(encoder_dims)) if not decoder_dims else decoder_dims  # Mirror encoder when no decoder dimensions are configured

    def build_network(self, n_features: int, n_classes: int, latent_dim: int) -> AutoencoderClassifierNetwork:  # Build network for fitted dimensions
        encoder_dims = self.normalize_dimensions(tuple(self.encoder_hidden_dims))  # Normalize encoder dimensions
        decoder_dims = self.resolve_decoder_hidden_dims(encoder_dims)  # Resolve decoder dimensions
        classifier_dims = self.normalize_dimensions(tuple(self.classifier_hidden_dims))  # Normalize classification-head dimensions
        return AutoencoderClassifierNetwork(n_features, n_classes, encoder_dims, decoder_dims, latent_dim, classifier_dims, str(self.activation), str(self.normalization), float(self.dropout))  # Return dynamically sized network

    def build_reconstruction_loss(self) -> nn.Module:  # Build configured reconstruction loss
        return nn.L1Loss() if str(self.reconstruction_loss) == "mae" else nn.MSELoss()  # Return selected reconstruction criterion

    def format_features(self, features: Any, device: torch.device) -> torch.Tensor:  # Convert one CPU batch to device tensor
        return torch.as_tensor(np.asarray(features, dtype=np.float32), device=device)  # Transfer current feature batch only

    def prepare_training_batch(self, clean_features: torch.Tensor) -> torch.Tensor:  # Apply optional training-only denoising
        if float(self.denoising_noise) <= 0.0:  # Use clean inputs when denoising is disabled
            return clean_features  # Return unmodified batch features
        return clean_features + torch.randn_like(clean_features) * float(self.denoising_noise)  # Return noise-augmented training inputs

    def fit(self, X: Any, y: Any) -> "AutoencoderClassifier":  # Fit from CPU arrays using bounded device batches
        self.validate_parameters()  # Validate public controls before allocation
        features = X.to_numpy(copy=False) if hasattr(X, "to_numpy") else np.asarray(X)  # Preserve arrays and memmaps without unconditional duplication
        labels = np.asarray(y).reshape(-1)  # Normalize labels to one vector
        if features.ndim != 2 or features.shape[0] != labels.shape[0] or features.shape[0] < 2 or features.shape[1] < 2:  # Require aligned tabular inputs
            raise ValueError("X must be a two-dimensional array with at least two features aligned with y")  # Reject invalid input
        self.classes_ = np.unique(labels)  # Record dynamic class labels
        if self.classes_.shape[0] < 2:  # Require classification targets
            raise ValueError("AutoencoderClassifier requires at least two classes")  # Reject single-class input
        contiguous_classes = np.array_equal(self.classes_, np.arange(self.classes_.shape[0]))  # Detect pipeline integer encoding
        encoded_labels = np.asarray(labels, dtype=np.int64) if contiguous_classes else np.searchsorted(self.classes_, labels).astype(np.int64, copy=False)  # Reuse or map labels once on CPU
        class_counts = np.bincount(encoded_labels, minlength=int(self.classes_.shape[0]))  # Count encoded labels for stratified validation safety
        stratify_labels = encoded_labels if int(class_counts.min()) > 1 else None  # Use stratification only when every class can appear in both splits
        all_indices = np.arange(features.shape[0], dtype=np.int64)  # Build row indices without copying features
        train_indices, validation_indices = train_test_split(all_indices, test_size=float(self.validation_fraction), random_state=int(self.random_state), stratify=stratify_labels)  # Derive validation only from training input
        self.n_features_in_ = int(features.shape[1])  # Record dynamic input width
        self.latent_dim_ = self.resolve_latent_dim(self.n_features_in_)  # Resolve fitted latent bottleneck dimension
        effective_device = self.resolve_device()  # Resolve compute device
        torch.set_num_threads(int(self.torch_threads))  # Bound native CPU threads
        torch.manual_seed(int(self.random_state))  # Seed CPU and active backend initialization
        if torch.cuda.is_available():  # Seed available CUDA devices
            torch.cuda.manual_seed_all(int(self.random_state))  # Make CUDA initialization reproducible
        torch.use_deterministic_algorithms(False)  # Allow nondeterministic CUDA kernels without reproducibility warnings
        network = self.build_network(self.n_features_in_, int(self.classes_.shape[0]), int(self.latent_dim_)).to(effective_device)  # Allocate model parameters on selected device
        optimizer = torch.optim.AdamW(network.parameters(), lr=float(self.learning_rate), weight_decay=float(self.weight_decay))  # Build configured optimizer
        reconstruction_loss_function = self.build_reconstruction_loss()  # Build reconstruction criterion
        classification_loss_function = nn.CrossEntropyLoss()  # Build multiclass classification criterion
        random_generator = np.random.default_rng(int(self.random_state))  # Build deterministic batch-order generator
        best_loss = math.inf  # Initialize validation target
        best_classification_loss = math.inf  # Initialize validation classification target
        best_reconstruction_loss = math.inf  # Initialize validation reconstruction target
        best_state = None  # Retain only best model state on CPU
        best_epoch = 0  # Record best validation epoch
        stale_epochs = 0  # Count epochs without improvement
        epochs_completed = 0  # Record effective completed epochs
        for epoch_index in range(int(self.epochs)):  # Train until limit or early stopping
            network.train()  # Enable training-time dropout and normalization behavior
            shuffled_indices = random_generator.permutation(train_indices)  # Shuffle row indices without copying features
            for batch_start in range(0, shuffled_indices.shape[0], int(self.batch_size)):  # Stream bounded training batches
                batch_indices = shuffled_indices[batch_start:batch_start + int(self.batch_size)]  # Select current training rows
                clean_features = self.format_features(features[batch_indices], effective_device)  # Transfer current clean feature batch only
                input_features = self.prepare_training_batch(clean_features)  # Apply optional training-only denoising
                batch_labels = torch.as_tensor(encoded_labels[batch_indices], dtype=torch.long, device=effective_device)  # Transfer current label batch only
                optimizer.zero_grad(set_to_none=True)  # Release prior gradients
                reconstruction, logits, _ = network(input_features)  # Compute joint autoencoder and classifier outputs
                reconstruction_loss_value = reconstruction_loss_function(reconstruction, clean_features)  # Compute reconstruction loss against clean inputs
                classification_loss_value = classification_loss_function(logits, batch_labels)  # Compute supervised multiclass loss
                loss = float(self.reconstruction_loss_weight) * reconstruction_loss_value + float(self.classification_loss_weight) * classification_loss_value  # Combine weighted objectives
                loss.backward()  # Backpropagate current batch
                optimizer.step()  # Update model parameters
            network.eval()  # Disable dropout and freeze normalization for validation
            validation_loss_sum = 0.0  # Accumulate weighted validation loss
            validation_classification_sum = 0.0  # Accumulate validation classification loss
            validation_reconstruction_sum = 0.0  # Accumulate validation reconstruction loss
            with torch.inference_mode():  # Disable autograd during validation
                for batch_start in range(0, validation_indices.shape[0], int(self.batch_size)):  # Stream bounded validation batches
                    batch_indices = validation_indices[batch_start:batch_start + int(self.batch_size)]  # Select current validation rows
                    batch_features = self.format_features(features[batch_indices], effective_device)  # Transfer current validation features only
                    batch_labels = torch.as_tensor(encoded_labels[batch_indices], dtype=torch.long, device=effective_device)  # Transfer current validation labels only
                    reconstruction, logits, _ = network(batch_features)  # Compute validation outputs without denoising
                    reconstruction_loss_value = reconstruction_loss_function(reconstruction, batch_features)  # Compute validation reconstruction loss
                    classification_loss_value = classification_loss_function(logits, batch_labels)  # Compute validation classification loss
                    combined_loss_value = float(self.reconstruction_loss_weight) * reconstruction_loss_value + float(self.classification_loss_weight) * classification_loss_value  # Compute validation weighted objective
                    validation_loss_sum += float(combined_loss_value.item()) * int(batch_indices.shape[0])  # Accumulate sample-weighted combined loss
                    validation_classification_sum += float(classification_loss_value.item()) * int(batch_indices.shape[0])  # Accumulate sample-weighted classification loss
                    validation_reconstruction_sum += float(reconstruction_loss_value.item()) * int(batch_indices.shape[0])  # Accumulate sample-weighted reconstruction loss
            validation_loss = validation_loss_sum / float(validation_indices.shape[0])  # Compute mean validation combined loss
            epochs_completed = epoch_index + 1  # Record completed epoch count
            if validation_loss < best_loss - float(self.min_delta):  # Accept meaningful improvement
                best_loss = validation_loss  # Record improved combined loss
                best_classification_loss = validation_classification_sum / float(validation_indices.shape[0])  # Record matching classification loss
                best_reconstruction_loss = validation_reconstruction_sum / float(validation_indices.shape[0])  # Record matching reconstruction loss
                best_epoch = epochs_completed  # Record improved epoch
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in network.state_dict().items()}  # Snapshot model parameters on CPU
                stale_epochs = 0  # Reset stopping counter
            else:  # Handle no meaningful improvement
                stale_epochs += 1  # Advance stopping counter
                if stale_epochs >= int(self.patience):  # Apply configured patience
                    break  # Stop without using experiment test rows
        if best_state is not None:  # Restore best validated weights
            network.load_state_dict(best_state)  # Load best model state
        encoder_dims = self.normalize_dimensions(tuple(self.encoder_hidden_dims))  # Normalize fitted encoder dimensions for metadata
        decoder_dims = self.resolve_decoder_hidden_dims(encoder_dims)  # Resolve fitted decoder dimensions for metadata
        classifier_dims = self.normalize_dimensions(tuple(self.classifier_hidden_dims))  # Normalize fitted head dimensions for metadata
        self.model_ = network.to("cpu")  # Keep fitted model CPU-safe for joblib
        self.device_ = str(effective_device)  # Record selected training device
        self.training_config_ = {"model_family": "supervised_autoencoder", "training_strategy": str(self.training_strategy), "device": self.device_, "input_dimension": int(self.n_features_in_), "latent_dimension": int(self.latent_dim_), "latent_ratio": None if self.latent_dim is not None else float(self.latent_ratio), "class_count": int(self.classes_.shape[0]), "class_order": self.classes_.tolist(), "epochs_requested": int(self.epochs), "epochs_completed": epochs_completed, "best_epoch": best_epoch, "best_validation_loss": float(best_loss), "best_validation_classification_loss": float(best_classification_loss), "best_validation_reconstruction_loss": float(best_reconstruction_loss), "batch_size": int(self.batch_size), "prediction_batch_size": int(self.prediction_batch_size), "data_loader_workers": int(self.data_loader_workers), "torch_threads": int(self.torch_threads), "encoder_hidden_dims": [int(value) for value in encoder_dims], "decoder_hidden_dims": [int(value) for value in decoder_dims], "classifier_hidden_dims": [int(value) for value in classifier_dims], "activation": str(self.activation), "normalization": str(self.normalization), "dropout": float(self.dropout), "denoising_noise": float(self.denoising_noise), "reconstruction_loss": str(self.reconstruction_loss), "reconstruction_loss_weight": float(self.reconstruction_loss_weight), "classification_loss_weight": float(self.classification_loss_weight), "learning_rate": float(self.learning_rate), "weight_decay": float(self.weight_decay), "patience": int(self.patience), "validation_fraction": float(self.validation_fraction), "random_state": int(self.random_state)}  # Record effective training configuration
        print(f"[AUTOENCODER (AE)] Strategy: {self.training_strategy} | Device: {self.device_} | Features: {self.n_features_in_} | Latent: {self.latent_dim_} | Classes: {len(self.classes_)} | Epochs: {epochs_completed}/{self.epochs} | Best epoch: {best_epoch}")  # Log training outcome
        del optimizer, reconstruction_loss_function, classification_loss_function, best_state, train_indices, validation_indices, all_indices  # Release training-only state
        if effective_device.type == "cuda":  # Release unused cached CUDA allocations
            torch.cuda.empty_cache()  # Return cached CUDA blocks
        return self  # Return fitted estimator

    def predict_logits(self, X: Any) -> np.ndarray:  # Produce CPU logits through bounded batches
        if not hasattr(self, "model_") or not hasattr(self, "classes_"):  # Require fitted state
            raise RuntimeError("AutoencoderClassifier is not fitted")  # Reject inference before fit
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
                _, logits, _ = network(batch_features)  # Compute current batch logits
                outputs.append(logits.cpu().numpy())  # Return current logits to CPU
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
