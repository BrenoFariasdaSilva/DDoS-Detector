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
    - Implement multi-GPU training support

Dependencies:
    - Python >= 3.9
    - torch >= 1.9.0
    - numpy
    - pandas
    - scikit-learn

Assumptions & Notes:
    - CSV should contain feature columns and a label column
    - Features use class-conditional quantiles and empirically calibrated flow relationships
    - Labels are encoded via LabelEncoder (categorical to integer)
    - Output features are inverse-transformed per class, with original discrete domains preserved
    - CUDA is used if available; use --force_cpu to disable
"""
if __name__ in {"__main__", "__mp_main__"}:
    try:
        from setproctitle import setproctitle
        setproctitle(f"DDoS-{__file__.rsplit('/', 1)[-1].rsplit('.', 1)[0]}")
    except ImportError:
        pass


import argparse  # For CLI argument parsing
import atexit  # For playing a sound when the program finishes
import csv  # For writing per-directory results CSV
import hashlib  # Compute stable experiment fingerprints and sample hashes
import dataframe_image as dfi  # For exporting DataFrame to PNG images
import datetime  # For tracking execution time
import json  # For saving/loading metrics history
import math  # Use ceiling rounding for persisted duration values
import matplotlib.pyplot as plt  # For plotting training metrics
import numpy as np  # Numerical operations
import os  # For running a command in the terminal
import pandas as pd  # For CSV handling
import platform  # For getting the operating system name
import random  # For reproducibility
import re  # Identify CSV-generated index features during preprocessing
import shutil  # For disk space queries when psutil is unavailable
import signal  # For sending signals to this process when critical
import subprocess  # For running small system commands to query hardware
import sys  # For system-specific parameters and functions
import telegram_bot as telegram_module  # For setting Telegram prefix and device info
import threading  # For background resource watcher thread
import tempfile  # For uniquely named, atomically committed results CSV updates
import time  # For elapsed time tracking
import torch  # PyTorch core
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Straight-through Gumbel-Softmax for categorical flow attributes
import traceback  # For printing tracebacks on exceptions
import yaml  # For loading configuration files
from colorama import Style  # For coloring the terminal
from contextlib import nullcontext  # For null context manager
try:
    import fcntl  # Serialize same-directory results updates across Linux processes.
except ImportError:
    fcntl = None  # Atomic replacement still works when POSIX advisory locks are unavailable.
from Logger import Logger  # For logging output to both terminal and file
from mpl_toolkits.mplot3d import Axes3D  # For 3D projection type annotation and axis method resolution
from pathlib import Path  # For handling file paths
from sklearn.manifold import TSNE  # For 3D t-SNE separability visualization
from sklearn.preprocessing import LabelEncoder, StandardScaler, QuantileTransformer, RobustScaler  # For mixed-type class-conditional preprocessing and reference-fitted t-SNE scaling
from telegram_bot import TelegramBot, send_exception_via_telegram, send_telegram_message, setup_global_exception_hook  # For Telegram utilities and global exception hook
from torch import autograd  # For gradient penalty
from torch.utils.data import DataLoader, Dataset, Sampler  # Dataset, dataloader and bounded class-aware sampling
from tqdm import tqdm  # For progress bar visualization
from typing import Any, Dict, List, Optional, Union, cast  # For Any type hint and cast
from utils.process_name import set_runtime_process_name  # Apply optional htop-visible process identities.
from zoneinfo import ZoneInfo  # Format ETA finish timestamps in Sao Paulo local time

psutil = (
    __import__("psutil") if __import__("importlib").util.find_spec("psutil") else None
)  # Import psutil if available, otherwise set to None

SAO_PAULO_TIMEZONE_NAME = "America/Sao_Paulo"  # Use Sao Paulo timezone for ETA finish timestamps
FLOW_PREPROCESSING_CHECKPOINT_TAG = "ConditionalMixedStructuralV62MixedDiscrete"  # V6.2 expands discrete output groups; never restore checkpoints trained with the old feature width.

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
RESOURCE_MONITOR_LATEST_SUFFIX = " | System RAM: N/A | Process RAM: N/A | CPU: N/A | GPU Usage: N/A | GPU VRAM: N/A"  # Immutable cached telemetry read without I/O by the training progress loop.
V6_REFERENCE_CORRELATIONS = {}  # Runtime-only per-file correlation targets.
V61_REFERENCE_MOMENTS = {}  # Runtime-only per-file mean/standard-deviation targets; never checkpoint these GPU tensors.


# Classes Definitions:


class ClassConditionalFlowScaler:
    """Map each class's skewed flow features into a bounded training representation.

    :param feature_cols: Ordered model feature names.
    :param seed: Reproducible bounded fitting seed.
    :param n_quantiles: Maximum empirical quantile resolution.
    :param fit_sample_rows: Maximum fit samples for each class.
    :return: A class-conditional invertible preprocessing object.
    """

    def __init__(self, feature_cols: List[str], seed: int = 42, n_quantiles: int = 512, fit_sample_rows: int = 50000):  # Configure bounded quantile fitting.
        self.feature_cols = list(feature_cols)  # Preserve the exact source feature schema.
        self.seed = int(seed)  # Preserve a deterministic sample seed.
        self.n_quantiles = max(16, int(n_quantiles))  # Bound the empirical distribution resolution.
        self.fit_sample_rows = max(self.n_quantiles, int(fit_sample_rows))  # Keep fitting memory independent of total dataset size.
        self.transformers = {}  # Hold independent class-specific marginal transformations.
        self.discrete_support = {}  # Retain observed discrete support rather than hallucinating fractional network fields.
        self.derived_rules = {}  # Record source-calibrated flow relationships by label.
        self.structural_rules = []  # Source-supported identities valid in every class; never predict these columns independently.
        self.structural_targets = set()  # Full-CSV column indexes rebuilt after generation.
        self.numeric_indices = []  # Model output continuous feature indexes in the original schema.
        self.categorical_indices = []  # Small-domain original feature indexes modeled with one-hot groups.
        self.categorical_values = {}  # Observed legal values for every categorical output group.
        self.categorical_class_masks = []  # Which categories have actually appeared in each original class.
        self.categorical_groups = []  # Model-output [start, end) offsets consumed by Gumbel-Softmax.
        self.model_feature_dim = None  # Encoded, structurally reduced mixed-type model width.
        self.mean_ = np.zeros(len(self.feature_cols), dtype=np.float64)  # Retain the legacy dimension-discovery interface.

    def _build_model_schema(self, X: np.ndarray, labels: np.ndarray) -> None:
        """Select globally supported identities and small categorical domains from training data."""
        class_ids = sorted(int(value) for value in np.unique(labels))
        # A target can be structurally removed only when all source classes support the same identity.
        per_class = [{rule[-1]: rule for rule in self.derived_rules[class_id] if not str(rule[-1]).startswith("bounds:")}
                     for class_id in class_ids]
        common = set.intersection(*(set(rules) for rules in per_class)) if per_class else set()
        self.structural_rules = [rule for rule in per_class[0].values()
                                 if rule[-1] in common and (rule[2] < 0 or np.all(X[:, rule[2]] > 0))] if per_class else []
        # For zero-duration/zero-denominator flows, keep the derived target trainable instead of filling its value with a fabricated median.
        self.structural_targets = {int(rule[0]) for rule in self.structural_rules}
        # The class-aware decoder still checks and applies every source-supported relationship.
        categorical_candidates = {"protocol", "inbound", "fwd psh flags", "bwd psh flags", "fwd urg flags", "bwd urg flags", "down/up ratio"}
        self.categorical_values = {}
        self.categorical_indices = []
        for index, column in enumerate(self.feature_cols):
            # Small observed count/flag domains deserve the same exact categorical
            # treatment as protocol and inbound. Large cardinality ports/counts stay
            # numeric to avoid an unbounded model width or discarded rare categories.
            if index in self.structural_targets or not (column.strip().lower() in categorical_candidates or is_discrete_flow_feature(column)):
                continue
            # Accumulate exact support in bounded chunks; never remap rare *real* categories to another category.
            seen = set()
            for start in range(0, len(X), 65536):
                values = np.unique(X[start:start + 65536, index])
                seen.update(float(value) for value in values if np.isfinite(value))
                if len(seen) > 64:
                    break
            if not 2 <= len(seen) <= 64:
                continue  # Constant fields or high-cardinality integers retain normal numeric representation.
            support = np.asarray(sorted(seen), dtype=np.float64)
            if np.any(flow_feature_invalid_mask(column, support)):
                raise ValueError(f"Source categorical feature {column!r} has invalid support after preprocessing")
            self.categorical_indices.append(index)
            self.categorical_values[index] = support
        self.numeric_indices = [i for i in range(len(self.feature_cols))
                                if i not in self.structural_targets and i not in self.categorical_values]
        offset = len(self.numeric_indices)
        self.categorical_groups = []
        for index in self.categorical_indices:
            count = len(self.categorical_values[index])
            self.categorical_groups.append((offset, offset + count))
            offset += count
        self.model_feature_dim = offset
        # Record per-class categorical support so the generator cannot invent unseen protocol/flag combinations.
        self.categorical_class_masks = []
        for index in self.categorical_indices:
            support = self.categorical_values[index]
            mask = np.zeros((max(class_ids) + 1, len(support)), dtype=bool)
            for class_id in class_ids:
                indexes = np.flatnonzero(labels == class_id)
                observed = set()
                for start in range(0, len(indexes), 65536):
                    observed.update(float(v) for v in np.unique(X[indexes[start:start+65536], index]))
                mask[class_id] = np.isin(support, list(observed))
                if not mask[class_id].any():
                    raise ValueError(f"No observed {self.feature_cols[index]!r} category for class {class_id}")
            self.categorical_class_masks.append(mask)
        if not offset:
            raise ValueError("No trainable features remain after structural feature selection")
        print(f"[V6.2] Original features={len(self.feature_cols)} | modeled numeric={len(self.numeric_indices)} | "
              f"categorical groups={len(self.categorical_groups)} | structural derived={len(self.structural_targets)} | model width={offset}")

    def _encode_block(self, normalized: np.ndarray, original: np.ndarray, destination: np.ndarray) -> None:
        """Pack model numeric values and exact real categorical one-hot groups."""
        if self.numeric_indices:
            destination[:, :len(self.numeric_indices)] = normalized[:, self.numeric_indices]
        for index, (start, stop) in zip(self.categorical_indices, self.categorical_groups):
            support = self.categorical_values[index]
            positions = np.searchsorted(support, np.asarray(original[:, index], dtype=np.float64))
            bounded = np.minimum(positions, len(support) - 1)
            if np.any((positions >= len(support)) | (support[bounded] != original[:, index])):
                raise ValueError(f"Unknown training category in {self.feature_cols[index]!r}; refusing to alter real records")
            destination[np.arange(len(destination)), start + bounded] = 1.0

    def fit_transform(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Fit source-class marginal scalers and encode only independent mixed-type model features."""
        labels = np.asarray(labels).reshape(-1)
        if len(labels) != len(X):
            raise ValueError("Feature and label arrays have different lengths")
        rng = np.random.default_rng(self.seed)
        self.transformers, self.discrete_support, self.derived_rules = {}, {}, {}
        for class_label in np.unique(labels):
            indexes = np.flatnonzero(labels == class_label)
            picked = np.sort(rng.choice(indexes, size=min(len(indexes), self.fit_sample_rows), replace=False))
            minimum, maximum = np.full(X.shape[1], np.inf), np.full(X.shape[1], -np.inf)
            minimum_rows = np.full(X.shape[1], int(indexes[0]), dtype=np.int64)
            maximum_rows = np.full(X.shape[1], int(indexes[0]), dtype=np.int64)
            columns = np.arange(X.shape[1])
            for pos in range(0, len(indexes), 65536):
                chunk_indices = indexes[pos:pos + 65536]
                chunk = np.asarray(X[chunk_indices], dtype=np.float64)
                local_min, local_max = np.argmin(chunk, axis=0), np.argmax(chunk, axis=0)
                selected_min, selected_max = chunk[local_min, columns], chunk[local_max, columns]
                new_min, new_max = selected_min < minimum, selected_max > maximum
                minimum[new_min], maximum[new_max] = selected_min[new_min], selected_max[new_max]
                minimum_rows[new_min] = chunk_indices[local_min[new_min]]
                maximum_rows[new_max] = chunk_indices[local_max[new_max]]
            picked = np.unique(np.concatenate((picked, minimum_rows, maximum_rows)))
            observed = np.asarray(X[picked], dtype=np.float64)
            if not np.isfinite(observed).all():
                raise ValueError(f"Nonfinite source features for label {class_label}")
            transformer = QuantileTransformer(n_quantiles=min(self.n_quantiles, len(picked)),
                                              output_distribution="normal", subsample=None,
                                              random_state=self.seed, copy=True)
            transformer.fit(observed)
            class_id = int(class_label)
            self.transformers[class_id] = transformer
            support = {}
            for idx, column in enumerate(self.feature_cols):
                if is_discrete_flow_feature(column):
                    values = np.unique(observed[:, idx])
                    legal = values[~flow_feature_invalid_mask(column, values)]
                    if len(legal) == 0:
                        raise ValueError(f"No domain-valid source support for {column!r} class {class_id}")
                    support[idx] = legal
            self.discrete_support[class_id] = support
            self.derived_rules[class_id] = calibrate_flow_relationships(observed, self.feature_cols)
        self._build_model_schema(X, labels)
        return self.transform(X, labels)

    def transform(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Encode each class in bounded chunks using the fixed mixed/structural model schema."""
        labels = np.asarray(labels).reshape(-1)
        if len(labels) != len(X) or self.model_feature_dim is None:
            raise ValueError("Unfitted mixed model representation or mismatched labels")
        encoded = np.zeros((len(X), self.model_feature_dim), dtype=np.float32)
        for class_label in np.unique(labels):
            indices = np.flatnonzero(labels == class_label)
            transformer = self.transformers[int(class_label)]
            for pos in range(0, len(indices), 65536):
                block = indices[pos:pos + 65536]
                raw = np.asarray(X[block], dtype=np.float64)
                normalized = transformer.transform(raw).astype(np.float32)
                packed = np.zeros((len(block), self.model_feature_dim), dtype=np.float32)
                self._encode_block(normalized, raw, packed)
                encoded[block] = packed
        return encoded

    def inverse_transform(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Recover original CSV schema, exact categorical categories and derived relationships."""
        labels = np.asarray(labels).reshape(-1)
        if len(labels) != len(X) or X.shape[1] != self.model_feature_dim:
            raise ValueError("Generated representation does not match the V6.2 model schema")
        result = np.empty((len(X), len(self.feature_cols)), dtype=np.float32)
        for class_label in np.unique(labels):
            class_id = int(class_label)
            if class_id not in self.transformers:
                raise ValueError(f"Unknown generated class {class_id}")
            indices = np.flatnonzero(labels == class_label)
            transformer = self.transformers[class_id]
            for pos in range(0, len(indices), 65536):
                block = indices[pos:pos + 65536]
                mixed = np.asarray(X[block], dtype=np.float64)
                if not np.isfinite(mixed).all():
                    raise ValueError(f"Nonfinite generated features for class {class_id}")
                # Unmodeled structural/categorical dimensions are placeholders, replaced below.
                full_normalized = np.zeros((len(block), len(self.feature_cols)), dtype=np.float64)
                full_normalized[:, self.numeric_indices] = np.clip(mixed[:, :len(self.numeric_indices)], -5.2, 5.2)
                restored = transformer.inverse_transform(full_normalized)
                for idx, (start, stop) in zip(self.categorical_indices, self.categorical_groups):
                    categories = self.categorical_values[idx]
                    restored[:, idx] = categories[np.argmax(mixed[:, start:stop], axis=1)]
                for idx, observed_values in self.discrete_support[class_id].items():
                    if idx in self.structural_targets or idx in self.categorical_values:
                        continue
                    column = self.feature_cols[idx]
                    values = np.asarray(observed_values, dtype=np.float64)
                    values = np.unique(values[~flow_feature_invalid_mask(column, values)])
                    if len(values) == 0:
                        raise ValueError(f"No domain-valid source support for {column!r} in class {class_id}")
                    locations = np.searchsorted(values, restored[:, idx])
                    high = np.minimum(locations, len(values) - 1)
                    low = np.maximum(high - 1, 0)
                    restored[:, idx] = np.where(np.abs(restored[:, idx] - values[low]) <=
                                                np.abs(restored[:, idx] - values[high]), values[low], values[high])
                # Globally supported identities are reconstituted from the sampled parent attributes.
                restored = apply_calibrated_flow_relationships(restored, self.feature_cols, self.derived_rules.get(class_id, []))
                result[block] = restored.astype(np.float32)
        return result


def is_discrete_flow_feature(name: Any) -> bool:
    """Identify CIC flow fields that must be selected from observed discrete domains.

    :param name: Original flow feature column name.
    :return: Whether the column is a discrete-valued flow field.
    """

    column = str(name).strip().lower()  # Normalize original CIC feature naming.
    if column in {"source port", "destination port", "protocol", "inbound", "act_data_pkt_fwd", "min_seg_size_forward", "down/up ratio", "fwd psh flags", "bwd psh flags", "fwd urg flags", "bwd urg flags", "init_win_bytes_forward", "init_win_bytes_backward"}:  # Identify known integer and categorical fields.
        return True  # Preserve empirical discrete support.
    return "flag count" in column or column in {"total fwd packets", "total backward packets", "subflow fwd packets", "subflow bwd packets", "total length of fwd packets", "total length of bwd packets", "fwd header length", "fwd header length.1", "bwd header length"}  # Preserve count and packet-length integer semantics.


def flow_relationship_candidates(feature_columns: List[str]) -> List[tuple]:
    """Return mathematical identities that must be calibrated against the source before use.

    :param feature_columns: Original ordered CIC flow feature names.
    :return: Candidate (target, numerator, denominator, multiplier) relations.
    """

    names = {str(column).lower().strip(): index for index, column in enumerate(feature_columns)}
    candidates = [
        ("fwd packet length mean", "total length of fwd packets", "total fwd packets", 1.0),
        ("bwd packet length mean", "total length of bwd packets", "total backward packets", 1.0),
        ("flow bytes/s", ("total length of fwd packets", "total length of bwd packets"), "flow duration", 1e6),
        ("flow packets/s", ("total fwd packets", "total backward packets"), "flow duration", 1e6),
        ("fwd packets/s", "total fwd packets", "flow duration", 1e6),
        ("bwd packets/s", "total backward packets", "flow duration", 1e6),
        ("avg fwd segment size", "fwd packet length mean", None, 1.0),
        ("avg bwd segment size", "bwd packet length mean", None, 1.0),
        ("packet length variance", "packet length std", "__square__", 1.0),
        ("fwd header length.1", "fwd header length", None, 1.0),
    ]
    results = []
    for target, numerators, denominator, multiplier in candidates:
        numerator_keys = numerators if isinstance(numerators, tuple) else (numerators,)
        if target in names and all(key in names for key in numerator_keys) and (denominator is None or denominator == "__square__" or denominator in names):
            den_idx = -2 if denominator is None else (-3 if denominator == "__square__" else names[denominator])
            results.append((names[target], tuple(names[key] for key in numerator_keys), den_idx, multiplier, target))
    return results


def predict_flow_relationship(features: np.ndarray, rule: tuple) -> tuple:
    """Evaluate a candidate relationship only for positive denominators and finite inputs.

    :param features: Unscaled source or synthetic numerical feature matrix.
    :param rule: Calibrated relationship definition.
    :return: Predicted derived values and applicability mask.
    """

    target, numerators, denominator, multiplier, _ = rule
    if denominator in (-2, -3):
        source = np.asarray(features[:, numerators[0]], dtype=np.float64)
        predicted = source if denominator == -2 else source ** 2
        predicted = np.asarray(predicted, dtype=np.float64) * multiplier
        valid = np.isfinite(predicted) & (predicted >= 0)
        return predicted, valid
    denominator_values = features[:, denominator]
    numerator_values = np.zeros(len(features), dtype=np.float64)
    for index in numerators:
        numerator_values += features[:, index]
    valid = (np.isfinite(denominator_values) & (denominator_values > 0) & np.isfinite(numerator_values))
    predicted = np.zeros(len(features), dtype=np.float64)
    np.divide(numerator_values * multiplier, denominator_values, out=predicted, where=valid)
    valid &= np.isfinite(predicted) & (predicted >= 0)
    return predicted, valid


def calibrate_flow_relationships(original: np.ndarray, feature_columns: List[str]) -> List[tuple]:
    """Learn which identities hold for the actual CICDDoS extraction schema.

    :param original: Unmodified original class rows selected for fitting.
    :param feature_columns: Source feature names.
    :return: Source-supported relations, without arbitrary cross-feature repair.
    """

    accepted = []
    for rule in flow_relationship_candidates(feature_columns):
        predicted, valid = predict_flow_relationship(original, rule)
        if int(valid.sum()) < max(20, int(len(original) * 0.01)):
            continue
        observed = original[valid, rule[0]]
        error = np.abs(predicted[valid] - observed) / np.maximum(np.maximum(np.abs(predicted[valid]), np.abs(observed)), 1e-6)
        if np.mean(error <= 0.05) >= 0.98:  # Require empirical 98% support before reconstructing derived values.
            accepted.append(rule)
    names = {str(column).strip().lower(): index for index, column in enumerate(feature_columns)}
    for prefix in ("fwd packet length", "bwd packet length", "packet length", "flow iat", "fwd iat", "bwd iat", "active", "idle"):
        keys = [f"{prefix} {suffix}" for suffix in ("min", "mean", "max")]
        if all(key in names for key in keys):
            low, middle, high = (original[:, names[key]] for key in keys)
            valid = np.isfinite(low) & np.isfinite(middle) & np.isfinite(high)
            if int(valid.sum()) >= max(20, int(len(original) * 0.01)) and np.mean((low[valid] <= middle[valid] + 1e-5) & (middle[valid] <= high[valid] + 1e-5)) >= 0.98:
                accepted.append((names[keys[1]], (names[keys[0]], names[keys[2]]), -1, 0.0, f"bounds:{prefix}"))
    return accepted


def apply_calibrated_flow_relationships(generated: np.ndarray, feature_columns: List[str], rules: List[tuple]) -> np.ndarray:
    """Reconstruct verified derived flow fields from their joint parent features.

    :param generated: Source-scale synthesized flow matrix.
    :param feature_columns: Column names retained for consistent interfaces.
    :param rules: Relations validated on original training-only class rows.
    :return: Synthetics with calibrated derived quantities made coherent.
    """

    for rule in rules:
        if str(rule[-1]).startswith("bounds:"):
            mean = generated[:, rule[0]]
            low_index, high_index = rule[1]
            generated[:, low_index] = np.minimum(generated[:, low_index], mean)
            generated[:, high_index] = np.maximum(generated[:, high_index], mean)
        else:
            predicted, valid = predict_flow_relationship(generated, rule)
            generated[valid, rule[0]] = predicted[valid]
    return generated


def flow_relationship_violation_counts(generated: np.ndarray, columns: List[str], rules: List[tuple]) -> Dict[str, int]:
    """Identify inter-feature relationships violated by generated rows.

    :param generated: Source-scale class records.
    :param columns: Ordered CIC flow feature names.
    :param rules: Empirically supported flow relations for this class.
    :return: Per-relationship violation counts.
    """

    violations = {}
    for rule in rules:
        if str(rule[-1]).startswith("bounds:"):
            mean = generated[:, rule[0]]
            low, high = (generated[:, index] for index in rule[1])
            count = int(np.count_nonzero((low > mean + 1e-5) | (mean > high + 1e-5)))
            if count:
                violations[rule[-1]] = count
            continue
        predicted, valid = predict_flow_relationship(generated, rule)
        actual = generated[:, rule[0]]
        errors = np.zeros(len(generated), dtype=np.float64)
        errors[valid] = np.abs(predicted[valid] - actual[valid]) / np.maximum(np.maximum(np.abs(predicted[valid]), np.abs(actual[valid])), 1e-6)
        count = int(np.count_nonzero(valid & (errors > 0.05)))
        if count:
            violations[rule[-1]] = count
    return violations


def flow_feature_invalid_mask(name: Any, values: Any) -> np.ndarray:
    """Return the exact CIC flow-domain violations used by both decoding and quality validation.

    The header-length fields must be nonnegative even when a source CSV or old
    V3 checkpoint contains negative CICFlowMeter artifacts. Discrete decoding may
    select only domain-valid values actually observed for the corresponding class.
    Signed init-window byte sentinels remain intentionally permitted.
    """

    feature = str(name).strip().lower()
    numbers = np.asarray(values, dtype=np.float64)
    invalid = ~np.isfinite(numbers)
    port = feature in {"source port", "destination port"}
    protocol = feature == "protocol"
    binary = feature in {"inbound", "fwd psh flags", "bwd psh flags", "fwd urg flags", "bwd urg flags"}
    integral = is_discrete_flow_feature(feature) and feature not in {"init_win_bytes_forward", "init_win_bytes_backward"}
    nonnegative = ("count" in feature or "packets" in feature or "bytes" in feature or "length" in feature or "duration" in feature or "iat" in feature or "flag" in feature or "size" in feature or "/s" in feature) and not feature.startswith("init_win_bytes") and feature != "min_seg_size_forward"
    if nonnegative or port or protocol or binary:
        invalid |= numbers < 0
    if feature == "down/up ratio":
        invalid |= numbers < 0
    if port:
        invalid |= numbers > 65535
    if protocol:
        invalid |= numbers > 255
    if integral:
        invalid |= np.abs(numbers - np.rint(numbers)) > 1e-5
    if binary:
        invalid |= (numbers != 0) & (numbers != 1)
    return invalid


class ClassAwareFlowSampler(Sampler[int]):
    """Sample real flow classes less unevenly without allocating replacement index lists.

    :param labels: Encoded real source labels.
    :param seed: Reproducible class sampler seed.
    :param exponent: Count weighting exponent, with one equal to original prevalence.
    :return: Bounded-memory sampler exposing one epoch of source-length examples.
    """

    def __init__(self, labels: np.ndarray, seed: int = 42, exponent: float = 0.5):  # Initialize class-balanced real critic sampling.
        self.class_indexes = {}  # Store compact source positions for every attack class.
        self.seed = int(seed)  # Retain reproducible sampling seed.
        self.epoch = 0  # Increment the deterministic epoch stream on every iteration.
        self.total = len(labels)  # Preserve original epoch length for progress reporting.
        exponent = float(exponent)  # Normalize configured class prevalence exponent.
        if not 0.0 <= exponent <= 1.0:  # Reject invalid class-balancing policies.
            raise ValueError("class_sampling_exponent must be between 0.0 and 1.0")  # Require an interpretable policy.
        labels = np.asarray(labels).reshape(-1)  # Convert source labels to one-dimensional IDs.
        for class_id in np.unique(labels):  # Preserve all original classes.
            self.class_indexes[int(class_id)] = np.flatnonzero(labels == class_id)  # Retain original source indices for class-specific sampling.
        self.class_ids = np.asarray(sorted(self.class_indexes), dtype=np.int64)  # Stabilize class choice ordering.
        weights = np.asarray([len(self.class_indexes[int(class_id)]) ** exponent for class_id in self.class_ids], dtype=np.float64)  # Temper original class imbalance without forcing uniformly replicated rare classes.
        self.class_probabilities = weights / weights.sum()  # Normalize real-class sampling probabilities.

    def __len__(self) -> int:
        """Return the original source population size.

        :return: Number of sampling draws per epoch.
        """

        return self.total  # Preserve existing epoch length.

    def __iter__(self):
        """Yield source indices in bounded random batches with class-aware weighting.

        :return: Iterator of valid original source-row positions.
        """

        rng = np.random.default_rng(self.seed + self.epoch)  # Use an independent reproducible RNG per epoch.
        self.epoch += 1  # Advance the sampling stream for the next epoch.
        for offset in range(0, self.total, 4096):  # Avoid materializing a full epoch of replacement indices.
            count = min(4096, self.total - offset)  # Restrict temporary work to one small batch.
            classes = rng.choice(self.class_ids, size=count, p=self.class_probabilities)  # Draw per-class real samples under the configured weighting policy.
            positions = np.empty(count, dtype=np.int64)  # Allocate one bounded batch of source positions.
            for class_id in np.unique(classes):  # Sample individual source rows from each selected class.
                mask = classes == class_id  # Locate the draws for this class within the batch.
                class_rows = self.class_indexes[int(class_id)]  # Resolve original rows for the chosen class.
                positions[mask] = class_rows[rng.integers(0, len(class_rows), size=int(mask.sum()))]  # Draw real rows with replacement within class.
            yield from positions.tolist()  # Stream only valid original indices to the DataLoader.


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
        if df is None or df.empty:  # Ensure preprocessing returned usable tabular data
            raise ValueError(f"No valid data remaining after preprocessing {csv_path}")  # Raise error
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

        label_series = cast(pd.Series, df[label_col])  # Narrow dynamic pandas column access to the expected label Series
        self.labels_raw: np.ndarray = np.asarray(label_series.values, dtype=str)  # Extract raw labels as string array for consistent encoding, store as instance variable

        self.labels: Any  # Must be Any or Pylance will error

        labels_arr = np.asarray(self.labels_raw, dtype=str)  # Convert labels to string array for consistent encoding

        if label_encoder is None:  # If no label encoder is given
            self.label_encoder = LabelEncoder()  # Create a fresh label encoder
            self.labels = self.label_encoder.fit_transform(labels_arr)  # Fit encoder and encode labels
        else:  # If encoder is provided
            self.label_encoder = label_encoder  # Store provided encoder
            self.labels = self.label_encoder.transform(labels_arr)  # Encode labels with given encoder

        X = df.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=False)  # Extract features and cast to float32

        if scaler is None:  # Fit class-specific marginals when the caller supplies no preprocessor.
            if not fit_scaler:  # An unfitted scaler cannot transform source data.
                raise ValueError("Class-conditional feature preprocessing requires fitted source distributions")  # Avoid using an identity substitute.
            quality_cfg = (CONFIG or {}).get("quality", {})  # Resolve empirical preprocessing limits from active configuration.
            self.scaler = ClassConditionalFlowScaler(feature_cols, seed=int((CONFIG or {}).get("wgangp", {}).get("seed", 42)), n_quantiles=int(quality_cfg.get("n_quantiles", 512)), fit_sample_rows=int(quality_cfg.get("fit_sample_rows", 50000)))  # Fit class-aware empirical feature distributions.
            self.X = self.scaler.fit_transform(X, self.labels)  # Encode mixed continuous and categorical network features per class.
        else:  # A compatible fitted class-conditional scaler was provided.
            self.scaler = scaler  # Preserve supplied preprocessing provenance.
            if not isinstance(scaler, ClassConditionalFlowScaler):  # Refuse legacy raw StandardScaler preprocessing during new training.
                raise ValueError("Legacy StandardScaler checkpoint is incompatible with class-conditional flow preprocessing")  # Avoid mixing feature spaces.
            self.X = np.asarray(scaler.fit_transform(X, self.labels) if fit_scaler else scaler.transform(X, self.labels), dtype=np.float32)  # Preserve source-to-normal-space transformation semantics.

        label_classes = getattr(self.label_encoder, "classes_", None)  # Retrieve fitted label classes from the encoder
        if label_classes is None:  # Verify the label encoder is fitted before counting classes
            raise ValueError("Label encoder has no fitted classes_ attribute")  # Raise clear error when encoder state is invalid
        self.n_classes = len(label_classes)  # Count number of unique classes
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
        categorical_groups: Optional[List[tuple]] = None,
        categorical_class_masks: Optional[List[np.ndarray]] = None,
        categorical_temperature: float = 1.25,
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
        self.categorical_temperature = float(categorical_temperature)
        if not np.isfinite(self.categorical_temperature) or self.categorical_temperature <= 0:
            raise ValueError("categorical_temperature must be a positive finite number")
        self.categorical_groups = tuple((int(start), int(end)) for start, end in (categorical_groups or []))
        for start, end in self.categorical_groups:
            if not 0 <= start < end <= feature_dim:
                raise ValueError("Categorical group outside the generator output feature dimension")
        for group_id, ((start, stop), allowed) in enumerate(zip(self.categorical_groups, categorical_class_masks or [])):
            mask = torch.as_tensor(allowed, dtype=torch.bool)
            if mask.shape != (n_classes, stop-start) or not bool(mask.any(dim=1).all()):
                raise ValueError("Invalid per-class categorical support for generator")
            self.register_buffer(f"categorical_allowed_{group_id}", mask)
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

        with autocast(z.device.type, enabled=torch.is_autocast_enabled()):  # Select BF16 inside each DataParallel worker thread
            y_e = self.embed(y)  # Convert class ID to embedding
            x = torch.cat([z, y_e], dim=1)  # Concatenate noise and embedding
            x = self.pre(x)  # Process through MLP
            for b in self.resblocks:  # Loop through residual blocks
                x = b(x)  # Apply block
            out = self.out(x)  # Produce final feature vector and categorical logits
            if self.categorical_groups:
                # Straight-through categorical choices match real one-hot critic inputs while retaining gradients.
                chunks = []
                offset = 0
                for group_id, (start, stop) in enumerate(self.categorical_groups):
                    if offset < start:
                        chunks.append(out[:, offset:start])
                    logits = out[:, start:stop]
                    class_mask = getattr(self, f"categorical_allowed_{group_id}", None)
                    if class_mask is not None:
                        logits = logits.masked_fill(~class_mask[y], -1e4)  # Never draw an unseen category for that class.
                    if self.training:
                        choice = F.gumbel_softmax(logits.float(), tau=1.0, hard=True, dim=1).to(logits.dtype)
                    else:
                        # Sample rather than take argmax: deterministic argmax can collapse minority categories during inference.
                        probabilities = torch.softmax(logits.float() / self.categorical_temperature, dim=1)
                        sampled = torch.multinomial(probabilities, num_samples=1).squeeze(1)
                        choice = F.one_hot(sampled, num_classes=stop-start).to(logits.dtype)
                    chunks.append(choice)
                    offset = stop
                if offset < out.shape[1]:
                    chunks.append(out[:, offset:])
                out = torch.cat(chunks, dim=1)
        return out  # Return mixed continuous values and exact categorical one-hot groups


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
        
        self.force_regenerate_samples = cfg.get("generation", {}).get("force_regenerate_samples", False)  # Resolve forced regeneration with a false fallback
        self.num_workers = int(cfg.get("dataloader", {}).get("num_workers", 8))  # Cast to int
        self._last_training_time = 0.0  # Placeholder for last training elapsed time (set after train)
        self.file_progress_prefix = ""  # Default per-file progress prefix (set at runtime when batch processing)


# Functions Definitions:


setup_global_exception_hook()  # Set global exception hook to shared Telegram handler


def safe_log(level: str, message: str):
    """
    Safe logging function that works with the minimal `Logger` (write/flush)
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

def get_gpu_telemetry() -> List[tuple]:
    """Read per-device NVIDIA utilization and VRAM from one bounded nvidia-smi call.

    :return: GPU index, utilization percent, used MiB and total MiB tuples; empty if unavailable.
    """

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            check=False,
            text=True,
            timeout=2,
        )  # Run only from the background resource monitor, never per training batch.
        if result.returncode != 0:
            return []
        devices = []
        for row in result.stdout.splitlines():
            fields = [field.strip() for field in row.split(",")]
            if len(fields) != 4:
                continue  # Ignore incomplete driver rows without affecting model training.
            try:
                index, utilization, used_mib, total_mib = (int(field) for field in fields)
            except ValueError:
                continue  # Ignore N/A or unsupported NVIDIA telemetry fields.
            if index >= 0 and 0 <= utilization <= 100 and 0 <= used_mib <= total_mib and total_mib > 0:
                devices.append((index, utilization, used_mib, total_mib))
        return sorted(devices)
    except (OSError, subprocess.SubprocessError):
        return []  # No NVIDIA driver or executable on this host.


def get_gpu_usage_summary() -> Optional[str]:
    """Preserve the existing human-readable GPU monitor summary interface."""

    devices = get_gpu_telemetry()
    if not devices:
        return None
    return ", ".join(f"GPU {index}={utilization}% ({used}/{total} MiB)" for index, utilization, used, total in devices)


def format_training_resource_suffix(memory: Any, cpu_percent: Any, process_rss: Optional[int], devices: List[tuple], process_cpu_percent: Optional[float] = None) -> str:
    """Format one sampled host/process/GPU snapshot for the tqdm description.

    Telemetry describes whole-system CPU/RAM, this process's RSS, and each NVIDIA GPU.
    It does not attribute device utilization to a particular WGAN-GP process.
    """

    if memory is None:
        ram_label = "N/A"
    else:
        ram_label = f"{(memory.total - memory.available) / (1024 ** 3):.1f}/{memory.total / (1024 ** 3):.1f} GiB ({memory.percent:.1f}%)"
    process_label = f"{process_rss / (1024 ** 3):.2f} GiB" if process_rss is not None else "N/A"
    cpu_label = f"{float(cpu_percent):.1f}%" if cpu_percent is not None else "N/A"
    process_cpu_label = f"{float(process_cpu_percent):.1f}%" if process_cpu_percent is not None else "N/A"
    if devices:
        gpu_label = ", ".join(f"GPU {index}: {utilization}%" for index, utilization, _, _ in devices)
        vram_label = ", ".join(f"GPU {index}: {used}/{total} MiB ({100 * used / total:.1f}%)" for index, _, used, total in devices)
    else:
        gpu_label, vram_label = "N/A", "N/A"
    return f" | System RAM: {ram_label} | Process RAM: {process_label} | CPU: {cpu_label} | Process CPU: {process_cpu_label} | GPU Usage: {gpu_label} | GPU VRAM: {vram_label}"


def get_cached_training_resource_suffix() -> str:
    """Return the monitor's latest immutable preformatted snapshot without blocking training."""

    return RESOURCE_MONITOR_LATEST_SUFFIX


def resource_monitor_loop(stop_event: threading.Event, config: Optional[Dict] = None):
    """
    Background loop that periodically inspects system resource usage (RAM/CPU/GPU)
    and logs / reports when thresholds are approached or exceeded.

    :param stop_event: Event used to signal the loop to stop
    :param config: Optional configuration dictionary containing watchdog settings
    :return: None
    """

    try:
        if psutil is None: return  # Verify psutil is available before monitoring

        cfg = config or CONFIG or get_default_config()  # Prefer explicit config, fallback to global/default
        wd = cfg.get("watchdog", {}) if isinstance(cfg, dict) else {}  # Get watchdog config dict
        max_ram = wd.get("max_ram_percent", 100)  # Percent RAM usage considered critical (default 100%)
        max_cpu = wd.get("max_cpu_percent", 100)  # Percent CPU usage considered critical (default 100%)
        check_interval = wd.get("check_interval_s", 5)  # Seconds between verifying resource usage (default 5 seconds)
        memory_threshold_duration_seconds = wd.get("memory_threshold_duration_seconds", wd.get("critical_duration_s", 3600))  # Seconds memory usage must remain above threshold before termination (default 30 minutes)

        high_usage_start_timestamp = None  # Timestamp when sustained high memory usage begins
        current_sustained_duration_seconds = 0.0  # Current sustained high-memory duration in seconds

        process = psutil.Process(os.getpid())  # Observe only this coordinator's resident memory; do not sum shared child memory.
        process.cpu_percent(interval=None)  # Prime psutil's non-blocking per-process CPU counter.
        while not stop_event.is_set():
            global RESOURCE_MONITOR_LATEST_SUFFIX  # Publish one immutable formatted resource snapshot for tqdm.
            virtual_memory = psutil.virtual_memory()  # Read available and total host memory together.
            mem = virtual_memory.percent  # Preserve the existing watchdog threshold semantics.
            cpu = psutil.cpu_percent(interval=None)  # Current whole-system CPU usage; no sampling delay.
            devices = get_gpu_telemetry()  # Query all GPU utilizations and VRAM once per monitor interval.
            gpu_usage = ", ".join(f"GPU {i}={util}% ({used}/{total} MiB)" for i, util, used, total in devices) or None  # Preserve debug reporting.
            try:
                process_rss = process.memory_info().rss  # Record the coordinator's physical resident memory.
                process_cpu = process.cpu_percent(interval=None)  # Non-blocking utilization since the previous monitor sample.
            except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
                process_rss, process_cpu = None, None  # Resource reporting must not interrupt training.
            RESOURCE_MONITOR_LATEST_SUFFIX = format_training_resource_suffix(virtual_memory, cpu, process_rss, devices, process_cpu)  # Atomic string replacement; no per-batch polling.
            current_timestamp = time.time()  # Capture the current timestamp for sustained duration tracking

            if logger:
                gpu_suffix = f", {gpu_usage}" if gpu_usage else ""  # Omit GPU output when no NVIDIA telemetry is available
                safe_debug(f"Resources Monitor: Memory={mem:.1f}%, CPU={cpu:.1f}%{gpu_suffix}")  # Debug log

            if mem >= max_ram:  # Verify memory is above the configured threshold
                if high_usage_start_timestamp is None:  # Verify this is the first sustained breach
                    high_usage_start_timestamp = current_timestamp  # Record when high memory usage started
                    current_sustained_duration_seconds = 0.0  # Reset sustained duration at the start of the breach
                    if logger:  # Verify logging is enabled before reporting high memory usage
                        safe_warning(f"Resource Watcher: High memory usage started (Memory={mem:.1f}%, Threshold={max_ram:.1f}%)")  # Warn when sustained high usage begins
                else:
                    current_sustained_duration_seconds = current_timestamp - high_usage_start_timestamp  # Update sustained high-memory duration
                    if logger:  # Verify logging is enabled before reporting sustained high memory usage
                        safe_warning(f"Resource Watcher: High memory usage sustained for {current_sustained_duration_seconds:.1f}s (Memory={mem:.1f}%, Threshold={max_ram:.1f}%)")  # Warn while memory remains elevated
                if current_sustained_duration_seconds >= memory_threshold_duration_seconds:  # Verify sustained duration meets the configured threshold
                    if logger:  # Verify logging is enabled before reporting termination
                        safe_critical(f"Resource Watcher: High memory usage sustained for {current_sustained_duration_seconds:.1f}s; Requesting termination")  # Critical log for real termination
                    try:  # Verify graceful termination signaling is available
                        send_telegram_message(TELEGRAM_BOT, f"Resource Watcher: High memory usage sustained for {current_sustained_duration_seconds:.1f}s; Requesting process termination to avoid abrupt OOM kill")  # Notify only on real termination
                    except Exception:
                        pass  # Ignore failures in notification

                    try:  # Verify SIGTERM delivery succeeds
                        os.kill(os.getpid(), signal.SIGTERM)  # Send SIGTERM to self for graceful shutdown
                    except Exception:
                        try:  # Verify forced exit remains available if SIGTERM fails
                            os._exit(1)  # Force exit if signal fails
                        except Exception:
                            pass  # Best effort: if even exit fails, continue
            else:  # Verify memory has returned below the configured threshold
                if high_usage_start_timestamp is not None:  # Verify a sustained high-memory interval was active
                    if logger:  # Verify logging is enabled before reporting the reset
                        safe_debug(f"Resource Watcher: High memory usage reset after {current_sustained_duration_seconds:.1f}s (Memory={mem:.1f}%, Threshold={max_ram:.1f}%)")  # Debug reset when memory recovers
                high_usage_start_timestamp = None  # Clear the stored start timestamp after recovery
                current_sustained_duration_seconds = 0.0  # Reset sustained duration after recovery

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


def build_default_runtime_config() -> Dict:
    """
    Build the default runtime configuration sections.

    Returns a dictionary containing the execution, wgangp, and dataset sections
    with their default values.

    :return: Dictionary with execution, wgangp, and dataset defaults
    """

    return {  # Begin runtime configuration dictionary
        "execution": {  # Execution control parameters
            "verbose": False,  # Enable verbose output messages
            "play_sound": True,  # Play sound notification when complete
            "results_suffix": "result",  # Suffix to add to generated filenames
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
            "generating_order": "off",  # Dataset file processing order by file size: "off", "Ascending", or "Descending"
            "execution_mode": "both",  # Execution mode: train, gen, or both for WGANGP
            "results_suffix": "_data_augmented",  # Suffix to add to generated filenames for WGANGP
            "results_csv_columns": [  # Define the canonical WGAN-GP augmentation result schema
                "original_file",  # Store the source dataset filename
                "generated_file",  # Store the persisted generated dataset path
                "original_num_samples",  # Store the source dataset row count
                "generated_num_samples",  # Store only newly generated synthetic rows
                "total_num_samples_after_augmentation",  # Store conceptual source plus generated population
                "generated_ratio",  # Store generated rows divided by source rows
                "epochs",  # Store configured total training epochs
                "batch_size",  # Store effective training batch size
                "critic_iterations",  # Store critic updates per generator update
                "lambda_gp",  # Store gradient penalty coefficient
                "latent_dim",  # Store generator latent vector size
                "learning_rate_generator",  # Store generator learning rate
                "learning_rate_critic",  # Store critic learning rate
                "epoch",  # Store last completed training epoch used for generation
                "epoch_time_s",  # Store last epoch duration in integer seconds
                "critic_loss",  # Store final logged critic loss
                "generator_loss",  # Store final logged generator loss
                "gp",  # Store final logged gradient penalty
                "D_real",  # Store final logged real-sample critic score
                "D_fake",  # Store final logged synthetic-sample critic score
                "wasserstein",  # Store final logged Wasserstein estimate
                "training_time_s",  # Store training-loop duration in integer seconds
                "generated_time_s",  # Store generation and output-write duration in integer seconds
                "testing_time_s",  # Store output verification and quality-plot duration in integer seconds
                "hardware",  # Store serialized hardware information
            ],  # End canonical WGAN-GP augmentation result schema
        },
        "watchdog": {  # Resource watcher configuration
            "max_ram_percent": 100,  # Percent RAM usage considered critical (default 100%)
            "max_cpu_percent": 100,  # Percent CPU usage considered critical (default 100%)
            "check_interval_s": 5,  # Seconds between verifying resource usage (default 5 seconds)
            "memory_threshold_duration_seconds": 1800,  # Seconds memory usage must remain above threshold before termination (default 30 minutes)
        },
        "dataset": {  # Dataset configuration
            "remove_zero_variance": True,  # Remove zero-variance features during preprocessing
            "label_candidates": ["label", "class", "target", "y", "category"],  # Common label column names for auto-detection
            "datasets": {  # Dictionary containing dataset paths
                "CICDDoS2019-Dataset": [  # List of paths to the CICDDoS2019 dataset
                    "./Datasets/CICDDoS2019/01-12/",
                    "./Datasets/CICDDoS2019/03-11/",
                ],
            },
        },
    }  # End runtime configuration dictionary


def build_wgangp_provenance(args: Any) -> Dict:
    """
    Describe source identity and generator training settings without reading large source files.

    :param args: Effective generator and training arguments.
    :return: Canonical source and model configuration plus its SHA-256 fingerprint.
    """

    source = Path(args.csv_path).resolve() if getattr(args, "csv_path", None) else None  # Resolve source without hashing tens of gigabytes.
    source_stat = source.stat() if source is not None else None  # Require an existing source for provenance.
    fields = ("seed", "label_col", "feature_cols", "batch_size", "critic_steps", "lr", "beta1", "beta2", "lambda_gp", "latent_dim", "g_hidden", "d_hidden", "embed_dim", "n_resblocks", "g_leaky_relu_alpha", "d_leaky_relu_alpha", "use_amp")  # Identify settings that affect learned weights.
    settings = {name: getattr(args, name, None) for name in fields}  # Snapshot resolved model and optimization settings.
    settings["moment_fidelity_weight"] = float((CONFIG or {}).get("training", {}).get("moment_fidelity_weight", 0.10))  # Ensure changed loss settings invalidate checkpoints.
    settings["correlation_fidelity_weight"] = float((CONFIG or {}).get("training", {}).get("correlation_fidelity_weight", 2.0))  # Track joint-fidelity training policy.
    settings["correlation_reference_rows_per_class"] = int((CONFIG or {}).get("training", {}).get("correlation_reference_rows_per_class", 4096))
    settings["moment_reference_rows_per_class"] = int((CONFIG or {}).get("training", {}).get("moment_reference_rows_per_class", 4096))
    identity = {"source_path": str(source) if source else None, "source_size": source_stat.st_size if source_stat else None, "source_mtime_ns": source_stat.st_mtime_ns if source_stat else None, "preprocessing": "class_conditional_mixed_structural_v62_mixed_discrete", "preprocessing_parameters": {key: (CONFIG or {}).get("quality", {}).get(key, default) for key, default in (("n_quantiles", 512), ("fit_sample_rows", 50000), ("class_sampling_exponent", 0.5))}, "settings": settings}  # Record source metadata and effective hyperparameters.
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode("utf-8")).hexdigest()  # Produce a stable comparable identity.

    return {"identity": identity, "sha256": digest}  # Return provenance for checkpoints and output sidecars.


def verify_wgangp_provenance(saved: Any, args: Any, context: str) -> None:
    """
    Reject missing or incompatible generator provenance before restoring model state.

    :param saved: Provenance record stored with the checkpoint.
    :param args: Effective execution arguments.
    :param context: Human-readable artifact description.
    :return: None.
    """

    expected = build_wgangp_provenance(args)  # Resolve current source and configuration fingerprint.
    if not isinstance(saved, dict) or saved.get("sha256") != expected["sha256"]:  # Reject legacy or mismatched checkpoint configurations.
        raise ValueError(f"{context} provenance does not match the current source/configuration; use --from_scratch with a fresh output or restore the original configuration")  # Prevent invalid model reuse.


def build_quality_policy_fingerprint(config: Dict) -> str:
    """Fingerprint quality thresholds independently of generator training hyperparameters.

    :param config: Runtime configuration with quality acceptance settings.
    :return: Stable SHA-256 identity for quality acceptance and cached-output reuse.
    """

    policy = {key: value for key, value in config.get("quality", {}).items() if key not in {"n_quantiles", "fit_sample_rows", "class_sampling_exponent", "plot_samples_per_class"}}
    return hashlib.sha256(json.dumps(policy, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def assess_generated_flow_quality(args: Any, config: Dict, generated: pd.DataFrame, scaler: Optional[Any] = None, label_encoder: Optional[Any] = None) -> None:
    """Audit per-class joint fidelity, diversity and network constraints before output acceptance.

    :param args: Generation arguments and exact original source path.
    :param config: Active sampling and reject-policy configuration.
    :param generated: Original-scale generated flows to validate.
    :param scaler: Fitted class-specific flow transformer containing source-supported joint rules.
    :param label_encoder: Fitted label encoder aligning class strings with the transformer.
    :return: None; reject unacceptable results before saving synthetic CSV files.
    """

    settings = config.get("quality", {})
    if not settings.get("enabled", True):
        raise ValueError("Quality auditing cannot be disabled in the strict flow generation format")
    label = args.label_col
    numeric_columns = [c for c in generated.select_dtypes(include=[np.number]).columns if c != label]
    if not numeric_columns or label not in generated or generated.empty:
        raise ValueError("Synthetic quality evaluation requires nonempty numeric features and a class label")
    sample_limit = max(128, int(settings.get("sample_rows_per_class", 4096)))
    near_limit = min(sample_limit, max(128, int(settings.get("nearest_reference_rows", 1024))))
    rng = np.random.default_rng(int(getattr(args, "seed", 42)))
    source_samples, source_priorities = {}, {}
    if getattr(args, "csv_path", None):
        required_columns = numeric_columns + [label]
        for chunk in pd.read_csv(args.csv_path, usecols=lambda col: col.strip() in required_columns, chunksize=8192, low_memory=False):
            chunk.columns = chunk.columns.str.strip()
            if label not in chunk.columns:
                raise ValueError(f"Missing original label column {label} for quality evaluation")
            for class_label, group in chunk.groupby(label, sort=False):
                class_key = str(class_label)
                candidates = group[numeric_columns].to_numpy(dtype=np.float32, copy=True)
                reference_valid = np.isfinite(candidates).all(axis=1)
                for column_index, feature_name in enumerate(numeric_columns):
                    reference_valid &= ~flow_feature_invalid_mask(feature_name, candidates[:, column_index])
                candidates = candidates[reference_valid]
                if not len(candidates):
                    continue
                priorities = rng.random(len(candidates))
                if class_key in source_samples:
                    candidates = np.concatenate([source_samples[class_key], candidates])
                    priorities = np.concatenate([source_priorities[class_key], priorities])
                count = min(sample_limit, len(priorities))
                selected = np.argpartition(priorities, count - 1)[:count]
                source_samples[class_key] = candidates[selected]
                source_priorities[class_key] = priorities[selected]
    rules_by_label = {}
    if isinstance(scaler, ClassConditionalFlowScaler) and label_encoder is not None:
        for class_id, rules in scaler.derived_rules.items():
            rules_by_label[str(label_encoder.inverse_transform([int(class_id)])[0])] = rules
    reports = []
    invalid_total = 0
    rejecting_classes = []
    for class_label, group in generated.groupby(label, sort=False):
        class_name = str(class_label)
        sample = group.sample(n=min(sample_limit, len(group)), random_state=int(getattr(args, "seed", 42)))
        sample_values = sample[numeric_columns].to_numpy(dtype=np.float64, copy=True)
        valid_generated = sample_values[np.isfinite(sample_values).all(axis=1)]
        reference = np.asarray(source_samples.get(class_name, []), dtype=np.float64)
        report = {"class": class_name, "generated_rows": int(len(group)), "sampled_rows": int(len(sample)), "violations": {}, "relationship_violations": {}, "quality_flags": []}
        for column in numeric_columns:
            values = pd.to_numeric(group[column], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            invalid = flow_feature_invalid_mask(column, values)  # Enforce the identical domain used by class-conditional decoding.
            count = int(np.count_nonzero(invalid))
            if count:
                report["violations"][str(column)] = count
            invalid_total += count
        # Inspect up to 250k complete original-scale rows per class, instead of relying on a 4k
        # sample that can conceal the true rate of identical network-flow records.
        audit_limit = max(4096, int(settings.get("duplicate_audit_rows_per_class", 250000)))
        audit = group if len(group) <= audit_limit else group.sample(n=audit_limit, random_state=int(getattr(args, "seed", 42)) + 17)
        audit_features = audit[numeric_columns]
        audited_duplicate_fraction = float(audit_features.duplicated().mean())
        report["audited_duplicate_fraction"] = audited_duplicate_fraction
        report["duplicate_audit_rows"] = int(len(audit))
        report["duplicate_audit_is_complete"] = bool(len(audit) == len(group))
        if (audited_duplicate_fraction > float(settings.get("max_audited_duplicate_fraction", 0.10))
                and audited_duplicate_fraction > 0.0):
            report["quality_flags"].append("excessive_audited_duplicate_rows")
        relation_rules = rules_by_label.get(class_name, [])
        report["enforced_derived_relationships"] = [rule[-1] for rule in relation_rules]
        report["relationship_violations"] = flow_relationship_violation_counts(sample_values, numeric_columns, relation_rules)
        if report["relationship_violations"]:
            report["quality_flags"].append("invalid_joint_feature_relationships")
        if len(reference) >= 32 and len(valid_generated) >= 32:
            original_mean = np.mean(reference, axis=0)
            original_std = np.std(reference, axis=0)
            scale = np.maximum(original_std, 1e-5)
            standardized_original = (reference - original_mean) / scale
            standardized_synthetic = (valid_generated - original_mean) / scale
            source_variable = np.std(standardized_original, axis=0) > 1e-4
            observed_variation = np.std(standardized_synthetic[:, source_variable], axis=0)
            real_variation = np.std(standardized_original[:, source_variable], axis=0)
            mean_shift = float(np.mean(np.abs(np.mean(standardized_synthetic, axis=0))))
            low_variance = float(np.mean(observed_variation < 0.1 * real_variation)) if len(real_variation) else 0.0
            variable_names = np.asarray(numeric_columns)[source_variable]
            std_ratios = np.divide(observed_variation, real_variation,
                                   out=np.full_like(observed_variation, np.nan), where=real_variation > 1e-8)
            worst = np.argsort(np.nan_to_num(std_ratios, nan=np.inf))[:15]
            report["lowest_generated_to_real_std_ratios"] = [
                {"feature": str(variable_names[i]), "generated_to_real_std_ratio": float(std_ratios[i])}
                for i in worst if np.isfinite(std_ratios[i])
            ]
            # Preserve the real class's natural duplicate baseline to avoid penalizing unavoidable identical flows.
            duplicate_fraction = float(pd.DataFrame(np.round(standardized_synthetic, 4)).duplicated().mean())
            real_duplicate_fraction = float(pd.DataFrame(np.round(standardized_original, 4)).duplicated().mean())
            differences = []
            valid_features = np.flatnonzero(source_variable & (np.std(standardized_synthetic, axis=0) > 1e-4))
            if len(valid_features) >= 2:
                real_corr = np.corrcoef(standardized_original[:, valid_features], rowvar=False)
                fake_corr = np.corrcoef(standardized_synthetic[:, valid_features], rowvar=False)
                triangle = np.triu_indices(len(valid_features), k=1)
                differences = np.abs(real_corr[triangle] - fake_corr[triangle])
            corr_shift = float(np.mean(differences)) if len(differences) else None
            from sklearn.neighbors import NearestNeighbors
            nearest_original = standardized_original[:near_limit]
            model = NearestNeighbors(n_neighbors=1).fit(nearest_original)
            distances = model.kneighbors(standardized_synthetic, return_distance=True)[0].reshape(-1)
            near_fraction = float(np.mean(distances < 0.05))
            # Baseline: compare held-out real reference rows against a disjoint reference subset.
            # A high synthetic near-match fraction is not evidence of copying if real flows naturally repeat.
            reference_holdout = standardized_original[near_limit:]
            reference_near_baseline = (float(np.mean(model.kneighbors(reference_holdout, return_distance=True)[0].reshape(-1) < 0.05))
                                       if len(reference_holdout) >= 32 else None)
            report.update({"mean_shift_standardized": mean_shift, "low_variance_feature_fraction": low_variance, "synthetic_near_duplicate_fraction": duplicate_fraction, "original_near_duplicate_fraction": real_duplicate_fraction, "joint_correlation_absolute_difference_mean": corr_shift, "reference_nearest_distance_median": float(np.median(distances)), "reference_near_match_fraction": near_fraction, "reference_real_to_real_near_match_fraction": reference_near_baseline})
            if mean_shift > float(settings.get("max_mean_shift", 2.0)):
                report["quality_flags"].append("large_marginal_mean_shift")
            if low_variance > float(settings.get("max_low_variance_fraction", 0.5)):
                report["quality_flags"].append("low_feature_diversity")
            if duplicate_fraction > max(float(settings.get("max_near_duplicate_fraction", 0.3)), real_duplicate_fraction + 0.2):
                report["quality_flags"].append("synthetic_mode_collapse")
            # A single 4k synthetic-vs-real correlation comparison is not a reliable
            # universal hard limit for skewed CICFlowMeter traffic. Calibrate sampling
            # variability from two disjoint REAL subsets using the same feature mask.
            real_baseline = None
            if len(reference) >= 128 and len(valid_features) >= 2:
                permuted = rng.permutation(len(reference))
                half = len(permuted) // 2
                real_a = standardized_original[permuted[:half]][:, valid_features]
                real_b = standardized_original[permuted[half:2 * half]][:, valid_features]
                eligible = ((np.std(real_a, axis=0) > 1e-5) & (np.std(real_b, axis=0) > 1e-5))
                if int(eligible.sum()) >= 2:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ca = np.corrcoef(real_a[:, eligible], rowvar=False)
                        cb = np.corrcoef(real_b[:, eligible], rowvar=False)
                    pairs = np.triu_indices(int(eligible.sum()), k=1)
                    deltas = np.abs(ca[pairs] - cb[pairs])
                    finite_deltas = deltas[np.isfinite(deltas)]
                    if len(finite_deltas):
                        real_baseline = float(np.mean(finite_deltas))
            original_limit = float(settings.get("max_correlation_difference", 0.25))
            margin = float(settings.get("correlation_real_baseline_margin", 0.05))
            ceiling = float(settings.get("max_calibrated_correlation_difference", 0.40))
            if not (np.isfinite(original_limit) and np.isfinite(margin) and np.isfinite(ceiling) and
                    original_limit > 0 and margin >= 0 and ceiling >= original_limit):
                raise ValueError("Invalid V6 correlation acceptance limits")
            correlation_limit = min(ceiling, max(original_limit, (real_baseline + margin) if real_baseline is not None else original_limit))
            report["real_to_real_correlation_difference_mean"] = real_baseline
            report["effective_correlation_difference_limit"] = correlation_limit
            report["raw_correlation_threshold_exceeded"] = bool(corr_shift is not None and corr_shift > original_limit)
            if corr_shift is not None and corr_shift > correlation_limit:
                report["quality_flags"].append("joint_correlation_mismatch")
            if (reference_near_baseline is not None and
                near_fraction > float(settings.get("max_reference_near_match_fraction", 0.95)) and
                near_fraction - reference_near_baseline > float(settings.get("max_reference_near_match_excess", 0.25))):
                report["quality_flags"].append("possible_training_data_memorization")
        else:
            report["quality_flags"].append("insufficient_reference_or_synthetic_sample")
        if report["violations"]:
            report["quality_flags"].append("invalid_individual_feature_domains")
        report["accepted"] = not bool(report["quality_flags"])
        if not report["accepted"]:
            rejecting_classes.append(class_name)
        reports.append(report)
    report_path = Path(args.out_file).with_suffix(".quality.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    accepted = invalid_total == 0 and not rejecting_classes
    result = {"schema_version": 3, "accepted": accepted, "quality_policy_sha256": build_quality_policy_fingerprint(config), "sampling_note": "Bounded screening statistics do not prove synthetic fidelity or downstream classifier utility", "sample_rows_per_class": sample_limit, "nearest_reference_rows": near_limit, "invalid_feature_values": invalid_total, "rejected_classes": rejecting_classes, "classes": reports}
    with report_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
    print(f"[QUALITY] Report: {report_path} | accepted={accepted} | invalid values={invalid_total} | flagged classes={rejecting_classes}")
    must_reject = ((bool(invalid_total) and settings.get("reject_invalid", True)) or
                   (bool(rejecting_classes) and settings.get("reject_quality_flags", True)))
    if must_reject and settings.get("save_rejected_candidate", True):
        # Save under a quarantine name. NEVER write rejected samples to the accepted
        # output path or a valid-looking results/provenance record.
        quarantine_path = Path(args.out_file).with_suffix(".rejected.csv")
        temporary_path = quarantine_path.with_name(quarantine_path.name + ".tmp")
        try:
            candidate_limit = max(1, int(settings.get("rejected_candidate_max_rows", 250000)))
            candidate = (generated if len(generated) <= candidate_limit else
                         generated.sample(n=candidate_limit, random_state=int(getattr(args, "seed", 42)) + 619))
            candidate.to_csv(temporary_path, index=False)
            os.replace(temporary_path, quarantine_path)
            print(f"[QUALITY] Rejected candidate retained for inspection ONLY: {quarantine_path} ({len(candidate)}/{len(generated)} rows)")
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    if invalid_total and settings.get("reject_invalid", True):
        raise ValueError(f"Rejected {invalid_total} invalid network-flow feature values; see {report_path}")
    if rejecting_classes and settings.get("reject_quality_flags", True):
        raise ValueError(f"Rejected synthetic output with class-specific fidelity/diversity/relationship failures: {rejecting_classes}; see {report_path}")


def build_default_training_and_model_config() -> Dict:
    """
    Build the default training and model architecture configuration sections.

    Returns a dictionary containing the training, generator, discriminator,
    gradient_penalty, generation, and dataloader sections with their default values.

    :return: Dictionary with training and model architecture defaults
    """

    return {  # Begin training and model configuration dictionary
        "training": {  # Training hyperparameters
            "epochs": 60,  # Number of training epochs
            "batch_size": 64,  # Training batch size
            "critic_steps": 5,  # Number of discriminator updates per generator update
            "lr": 1e-4,  # Learning rate for both networks
            "beta1": 0.5,  # Adam optimizer beta1 parameter
            "beta2": 0.9,  # Adam optimizer beta2 parameter
            "lambda_gp": 10.0,  # Gradient penalty coefficient
            "moment_fidelity_weight": 0.10,  # V6.2 sparse-mode-aware reference moments; tune against independent validation data.
            "moment_reference_rows_per_class": 4096,  # Stable real source samples per class for mean and variance targets.
            "correlation_reference_rows_per_class": 4096,  # Stable per-class real reference for correlation loss.
            "correlation_fidelity_weight": 2.0,  # V6 stable class-reference correlation penalty; 0 disables it.
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
            "force_regenerate_samples": False,  # Preserve missing-only generation by default
            "out_file": "generated.csv",  # Output CSV filename
            "gen_batch_size": 256,  # Generation batch size
            "categorical_temperature": 1.25,  # V5 stochastic categorical sampling temperature (>0).
            "feature_dim": None,  # Feature dimensionality (None = auto-detect)
            "small_class_threshold": 100,  # Threshold for small class detection
            "small_class_min_samples": 10,  # Minimum samples for small classes
        },
        "quality": {"enabled": True, "sample_rows_per_class": 4096, "nearest_reference_rows": 1024, "reject_invalid": True, "reject_quality_flags": True, "n_quantiles": 512, "fit_sample_rows": 50000, "class_sampling_exponent": 0.5, "plot_samples_per_class": 512, "max_mean_shift": 2.0, "max_low_variance_fraction": 0.5, "max_correlation_difference": 0.25, "correlation_real_baseline_margin": 0.05, "max_calibrated_correlation_difference": 0.40, "save_rejected_candidate": True, "rejected_candidate_max_rows": 250000, "max_near_duplicate_fraction": 0.30, "max_reference_near_match_fraction": 0.95, "max_reference_near_match_excess": 0.25, "duplicate_audit_rows_per_class": 250000, "max_audited_duplicate_fraction": 0.10},  # Configure bounded quality diagnostics and fail-closed constraint enforcement.
        "dataloader": {  # DataLoader configuration
            "num_workers": 8,  # Number of workers for data loading
            "pin_memory": True,  # Use pinned memory for faster GPU transfer
            "persistent_workers": True,  # Keep workers alive between epochs
            "prefetch_factor": 2,  # Number of batches to prefetch
        },
    }  # End training and model configuration dictionary


def build_default_output_and_notification_config() -> Dict:
    """
    Build the default output and notification configuration sections.

    Returns a dictionary containing the plotting, paths, logging, telegram,
    and sound sections with their default values.

    :return: Dictionary with output and notification defaults
    """

    return {  # Begin output and notification configuration dictionary
        "plotting": {  # Visualization configuration
            "enabled": True,  # Enable plot generation
            "filename": "training_metrics.png",  # Plot filename
            "subdir": "Plots/metrics",  # Subdirectory for training metrics plots
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
            "data_augmentation_dir": "Data_Augmentation",  # Base directory for all augmentation artifacts
            "data_augmentation_sample_dir": "Samples",  # Subdirectory containing generated augmented dataset CSV files
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
    }  # End output and notification configuration dictionary


def get_default_config():
    """
    Return the default configuration dictionary for WGAN-GP.

    This defines all configurable parameters with their default values.
    These defaults match config.yaml.example structure and can be overridden
    by config.yaml file or CLI arguments.

    :return: Dictionary containing default configuration values
    """

    try:
        runtime = build_default_runtime_config()  # Build execution, wgangp, and dataset defaults
        training_model = build_default_training_and_model_config()  # Build training and model architecture defaults
        output_notif = build_default_output_and_notification_config()  # Build output and notification defaults
        merged = {}  # Initialize merged configuration dictionary
        merged.update(runtime)  # Merge runtime section into final config
        merged.update(training_model)  # Merge training and model section into final config
        merged.update(output_notif)  # Merge output and notification section into final config
        return merged  # Return complete default configuration dictionary
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
                    generation_config = file_config.get("generation", {}) if isinstance(file_config, dict) else {}  # Read generation settings for legacy-key migration
                    if "force_regenerate_samples" not in generation_config and "force_new_samples" in generation_config:  # Preserve old custom configurations only when new key is absent
                        generation_config["force_regenerate_samples"] = generation_config["force_new_samples"]  # Migrate legacy forced-generation value into current setting
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
        
        globals()["safe_debug"] = safe_debug  # Make available globally
        globals()["safe_warning"] = safe_warning  # Make available globally
        globals()["safe_critical"] = safe_critical  # Make available globally

        global RESOURCE_MONITOR_STOP_EVENT, RESOURCE_MONITOR_THREAD  # Access globals for monitor
        if RESOURCE_MONITOR_THREAD is None or not (RESOURCE_MONITOR_THREAD and RESOURCE_MONITOR_THREAD.is_alive()):  # If monitor not running
            RESOURCE_MONITOR_STOP_EVENT, RESOURCE_MONITOR_THREAD = start_resource_monitor(config)  # Start monitor with current config
        else:  # Monitor already running
            safe_debug("Resource monitor already active; skipping start")  # Debug message
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
        
        candidates = config.get("dataset", {}).get("label_candidates", ["label", "class", "target", "y", "category"])  # Get label column candidates from config

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
        # CSV serialization artifacts and synthetic row IDs are metadata, not network-flow features.
        index_columns = [column for column in numeric_cols if re.fullmatch(r"Unnamed:\s*\d+(?:\.\d+)*", str(column), re.IGNORECASE)
                         or str(column).strip().lower() in {"index", "row_index", "row_id"}]
        if index_columns:
            print(f"[V5] Excluding CSV index/row metadata from GAN training: {index_columns}")
            numeric_cols = [column for column in numeric_cols if column not in index_columns]
        if label_col in numeric_cols:  # If label is numeric, remove it from features
            numeric_cols.remove(label_col)  # Remove label from feature list

        verbose_output(
            f"{BackgroundColors.GREEN}Found {len(numeric_cols)} numeric feature columns out of {len(df.columns)-1} total features.{Style.RESET_ALL}"
        )  # Output count

        df_numeric = df[numeric_cols].copy()  # Select numeric features

        df_numeric = df_numeric.replace([np.inf, -np.inf], np.nan)  # Replace inf with NaN
        valid_mask = ~df_numeric.isna().any(axis=1)  # Mask for rows without NaN

        # Exclude physically invalid source records before quantile fitting; do not teach the GAN corrupt flows.
        domain_invalid = np.zeros(len(df_numeric), dtype=bool)
        for column in numeric_cols:
            values = pd.to_numeric(df_numeric[column], errors="coerce").to_numpy(dtype=np.float64, copy=False)
            invalid = flow_feature_invalid_mask(column, values)
            if np.any(invalid & valid_mask.to_numpy()):
                print(f"[V5] Source domain violations in {column!r}: {int(np.count_nonzero(invalid & valid_mask.to_numpy()))} rows")
            domain_invalid |= invalid
        valid_mask &= ~domain_invalid
        df_clean = df_numeric[valid_mask].copy()  # Keep only finite, domain-valid rows
        labels_clean = labels[valid_mask].copy()  # Keep corresponding labels

        rows_dropped = len(df) - len(df_clean)  # Calculate dropped rows
        if rows_dropped > 0:  # If rows were dropped
            verbose_output(
                f"{BackgroundColors.YELLOW}Dropped {rows_dropped} rows with NaN/inf or invalid network-flow feature values ({rows_dropped/len(df)*100:.2f}%).{Style.RESET_ALL}"
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


def verbose_output(true_string="", false_string="", config: Optional[Dict] = None):
    """
    Output a message if the verbose flag is enabled in configuration.

    :param true_string: The string to be outputted if verbose is enabled.
    :param false_string: The string to be outputted if verbose is disabled.
    :param config: Optional configuration dictionary containing verbose setting
    :return: None.
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
    Verify if the .env file exists in the current directory.

    :param config: Optional configuration dictionary containing telegram settings
    :return: True if the .env file exists, False otherwise.
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
    Set up the Telegram bot for progress messages.

    :param config: Optional configuration dictionary containing telegram settings
    :return: None.
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
            telegram_module.TELEGRAM_DEVICE_INFO = f"{telegram_module.get_local_ip()} - {platform.system()}"  # Set device info string with IP and OS
            telegram_module.RUNNING_CODE = os.path.basename(__file__)  # Set currently running script name
            telegram_module.TELEGRAM_BOT = TELEGRAM_BOT  # Register the created bot instance in telegram_bot module
        except Exception as e:
            print(f"{BackgroundColors.RED}Failed to initialize Telegram bot: {e}{Style.RESET_ALL}")  # Report initialization failure to terminal
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
        p.add_argument("--process-name", type=str, default=None, help="Process title displayed by htop and similar tools")  # Allow concurrent runs to have distinct operating-system identities.
        
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
        p.add_argument("--force-regenerate-samples", "--force_regenerate_samples", "--force_new_samples", dest="force_regenerate_samples", action="store_true", default=None, help="Force regeneration even if generated samples already exist")  # Add explicit forced-regeneration override
        p.add_argument("--no-force-regenerate-samples", "--no_force_regenerate_samples", dest="force_regenerate_samples", action="store_false", help="Disable forced regeneration and use existing generated samples")  # Add explicit forced-regeneration disable override
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


def apply_core_cli_overrides(args, overrides: Dict) -> None:
    """
    Apply core CLI argument overrides for execution, wgangp, and paths sections.

    Mutates the overrides dictionary in-place with values from verbose, no_sound,
    mode, csv_path, label_col, feature_cols, seed, force_cpu, from_scratch,
    out_dir, and logs_dir arguments.

    :param args: argparse.Namespace containing CLI arguments
    :param overrides: Overrides dictionary to mutate in-place
    :return: None
    """

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


def apply_training_and_model_cli_overrides(args, overrides: Dict) -> None:
    """
    Apply training, generator, and discriminator CLI argument overrides.

    Mutates the overrides dictionary in-place with values for epochs, batch_size,
    critic_steps, lr, beta1, beta2, lambda_gp, save_every, log_interval,
    sample_batch, use_amp, compile, latent_dim, hidden_dims, embed_dim,
    n_resblocks, and leaky_relu_alpha arguments.

    :param args: argparse.Namespace containing CLI arguments
    :param overrides: Overrides dictionary to mutate in-place
    :return: None
    """

    if args.epochs is not None:  # If epochs specified
        overrides.setdefault("training", {})["epochs"] = args.epochs  # Set epochs
    if args.batch_size is not None:  # If batch_size specified
        overrides.setdefault("training", {})["batch_size"] = args.batch_size  # Set batch_size
        overrides.setdefault("training", {})["batch_size_cli_override"] = True  # Preserve explicit CLI precedence
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


def apply_generation_and_data_cli_overrides(args, overrides: Dict) -> None:
    """
    Apply generation, dataloader, and dataset CLI argument overrides.

    Mutates the overrides dictionary in-place with values for checkpoint,
    n_samples, force_regenerate_samples, gen_label, out_file, gen_batch_size,
    feature_dim, num_workers, remove_zero_variance, and
    no_remove_zero_variance arguments.

    :param args: argparse.Namespace containing CLI arguments
    :param overrides: Overrides dictionary to mutate in-place
    :return: None
    """

    if args.checkpoint is not None:  # If checkpoint specified
        overrides.setdefault("generation", {})["checkpoint"] = args.checkpoint  # Set checkpoint
    if args.n_samples is not None:  # If n_samples specified
        overrides.setdefault("generation", {})["n_samples"] = args.n_samples  # Set n_samples
    if getattr(args, "force_regenerate_samples", None) is not None:  # Apply only an explicit forced-regeneration CLI value
        overrides.setdefault("generation", {})["force_regenerate_samples"] = args.force_regenerate_samples  # Override configured forced regeneration
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
        overrides.setdefault("dataloader", {})["num_workers_cli_override"] = True  # Preserve explicit CLI precedence

    if args.remove_zero_variance:  # If remove_zero_variance flag set
        overrides.setdefault("dataset", {})["remove_zero_variance"] = True  # Enable remove_zero_variance
    if args.no_remove_zero_variance:  # If no_remove_zero_variance flag set
        overrides.setdefault("dataset", {})["remove_zero_variance"] = False  # Disable remove_zero_variance


def args_to_config_overrides(args):
    """
    Convert parsed CLI arguments to configuration overrides dictionary.

    :param args: argparse.Namespace containing CLI arguments
    :return: Dictionary with configuration overrides
    """

    try:
        overrides = {}  # Initialize overrides dictionary
    
        apply_core_cli_overrides(args, overrides)  # Apply execution, wgangp, and paths overrides
        apply_training_and_model_cli_overrides(args, overrides)  # Apply training, generator, and discriminator overrides
        apply_generation_and_data_cli_overrides(args, overrides)  # Apply generation, dataloader, and dataset overrides
        
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


def resolve_entry_with_trailing_space(current_path: str, entry: str, stripped_part: str) -> str:
    """
    Resolve and optionally rename a directory entry with trailing spaces.

    :param current_path: Current directory path.
    :param entry: Directory entry name.
    :param stripped_part: Normalized target name without surrounding spaces.
    :return: Resolved path after optional rename.
    """

    try:  # Wrap full function logic to ensure safe execution
        resolved = os.path.join(current_path, entry)  # Build resolved path

        if entry != stripped_part:  # Verify trailing spaces exist
            corrected = os.path.join(current_path, stripped_part)  # Build corrected path
            try:  # Attempt to rename entry
                os.rename(resolved, corrected)  # Rename entry to stripped version
                verbose_output(true_string=f"{BackgroundColors.GREEN}Renamed: {BackgroundColors.CYAN}{resolved}{BackgroundColors.GREEN} -> {BackgroundColors.CYAN}{corrected}{Style.RESET_ALL}")  # Log rename
                resolved = corrected  # Update resolved path after rename
            except Exception:  # Handle rename failure
                verbose_output(true_string=f"{BackgroundColors.RED}Failed to rename: {BackgroundColors.CYAN}{resolved}{Style.RESET_ALL}")  # Log failure

        return resolved  # Return resolved path
    except Exception:  # Catch unexpected errors
        return os.path.join(current_path, entry)  # Return fallback resolved path


def resolve_full_trailing_space_path(filepath: str) -> str:
    """
    Resolve trailing space issues across all path components.

    :param filepath: Path to resolve potential trailing space mismatches.
    :return: Corrected full path if matches are found, otherwise original filepath.
    """

    try:  # Wrap full function logic to ensure safe execution
        verbose_output(true_string=f"{BackgroundColors.GREEN}Resolving full trailing space path for: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}")  # Log start

        if not isinstance(filepath, str) or not filepath:  # Verify filepath validity
            verbose_output(true_string=f"{BackgroundColors.YELLOW}Invalid filepath provided, skipping resolution.{Style.RESET_ALL}")  # Log invalid input
            return filepath  # Return original

        filepath = os.path.expanduser(filepath)  # Expand ~ to user directory
        parts = filepath.split(os.sep)  # Split path into components

        if not parts:  # Verify path parts exist
            return filepath  # Return original

        if filepath.startswith(os.sep):  # Handle absolute paths
            current_path = os.sep  # Start from root
            parts = parts[1:]  # Remove empty root part
        else:
            current_path = parts[0] if parts[0] else os.getcwd()  # Initialize base
            parts = parts[1:] if parts[0] else parts  # Adjust parts

        for part in parts:  # Iterate over each path component
            if part == "":  # Skip empty parts
                continue  # Continue iteration

            try:  # Attempt to list current directory
                entries = os.listdir(current_path) if os.path.isdir(current_path) else []  # List current directory entries
            except Exception:  # Handle failure to list directory contents
                verbose_output(true_string=f"{BackgroundColors.RED}Failed to list directory: {BackgroundColors.CYAN}{current_path}{Style.RESET_ALL}")  # Log failure
                return filepath  # Return original

            stripped_part = part.strip()  # Normalize current part
            match_found = False  # Initialize match flag

            for entry in entries:  # Iterate directory entries
                try:  # Attempt safe comparison for each entry
                    if entry.strip() == stripped_part:  # Compare stripped names
                        current_path = resolve_entry_with_trailing_space(current_path, entry, stripped_part)  # Resolve entry and update current path
                        match_found = True  # Mark match
                        break  # Stop searching
                except Exception:  # Handle any unexpected error during comparison
                    continue  # Continue on error

            if not match_found:  # If no match found for this segment
                verbose_output(true_string=f"{BackgroundColors.YELLOW}No match for segment: {BackgroundColors.CYAN}{part}{Style.RESET_ALL}")  # Log miss
                return filepath  # Return original

        return current_path  # Return fully resolved path

    except Exception:  # Catch unexpected errors to maintain stability
        verbose_output(true_string=f"{BackgroundColors.RED}Error resolving full path: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}")  # Log error
        return filepath  # Return original


def verify_filepath_exists(filepath):
    """
    Verify if a file or folder exists at the specified path.

    :param filepath: Path to the file or folder
    :return: True if the file or folder exists, False otherwise
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Verifying if the file or folder exists at the path: {BackgroundColors.CYAN}{filepath}{Style.RESET_ALL}"
        )  # Output the verbose message
        
        if not isinstance(filepath, str) or not filepath.strip():  # Verify for non-string or empty/whitespace-only input   
            verbose_output(true_string=f"{BackgroundColors.YELLOW}Invalid filepath provided, skipping existence verification.{Style.RESET_ALL}")  # Log invalid input
            return False  # Return False for invalid input

        if os.path.exists(filepath):  # Fast path: original input exists
            return True  # Return True immediately

        candidate = str(filepath).strip()  # Normalize input to string and strip surrounding whitespace

        if (candidate.startswith("'") and candidate.endswith("'")) or (
            candidate.startswith('"') and candidate.endswith('"')
        ):  # Handle quoted paths from config files
            candidate = candidate[1:-1].strip()  # Remove wrapping quotes and trim again

        candidate = os.path.expanduser(candidate)  # Expand ~ to user home directory
        candidate = os.path.normpath(candidate)  # Normalize path separators and structure

        if os.path.exists(candidate):  # Verify normalized candidate directly
            return True  # Return True if normalized path exists

        repo_dir = os.path.dirname(os.path.abspath(__file__))  # Resolve repository directory
        cwd = os.getcwd()  # Capture current working directory

        alt = candidate.lstrip(os.sep) if candidate.startswith(os.sep) else candidate  # Prepare relative-safe path

        repo_candidate = os.path.join(repo_dir, alt)  # Build repo-relative candidate
        cwd_candidate = os.path.join(cwd, alt)  # Build cwd-relative candidate

        for path_variant in (repo_candidate, cwd_candidate):  # Iterate alternative base paths
            try:
                normalized_variant = os.path.normpath(path_variant)  # Normalize variant
                if os.path.exists(normalized_variant):  # Verify existence
                    return True  # Return True if found
            except Exception:
                continue  # Continue safely on error

        try:  # Attempt absolute path resolution as fallback
            abs_candidate = os.path.abspath(candidate)  # Build absolute path
            if os.path.exists(abs_candidate):  # Verify existence
                return True  # Return True if found
        except Exception:
            pass  # Ignore resolution errors

        for path_variant in (candidate, repo_candidate, cwd_candidate):  # Attempt trailing-space resolution on all variants
            try:  # Attempt to resolve trailing space issues across path components for this variant
                resolved = resolve_full_trailing_space_path(path_variant)  # Resolve trailing space issues across path components
                if resolved != path_variant and os.path.exists(resolved):  # Verify resolved path exists
                    verbose_output(
                        f"{BackgroundColors.YELLOW}Resolved trailing space mismatch: {BackgroundColors.CYAN}{path_variant}{BackgroundColors.YELLOW} -> {BackgroundColors.CYAN}{resolved}{Style.RESET_ALL}"
                    )  # Log successful resolution
                    return True  # Return True if corrected path exists
            except Exception:  # Catch any exception during trailing space resolution   
                continue  # Continue safely on error

        return False  # Not found after all resolution strategies
    except Exception as e:  # Catch any exception to ensure logging and Telegram alert
        print(str(e))  # Print error to terminal for server logs
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full traceback via Telegram
        raise  # Re-raise to preserve original failure semantics


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

        if not os.path.isdir(directory_path):  # Verify path is a valid directory
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

            if any(ignore == filename or ignore == item_path for ignore in ignore_files):  # Verify if file is in ignore list
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


def safe_float(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float, returning default if conversion fails.

    :param value: Value to convert to float
    :param default: Value to return if conversion fails
    :return: float representation or default
    """
    
    try:
        return float(value)  # Attempt conversion to float
    except (TypeError, ValueError):
        return float(default)  # Return fallback default on conversion failure


def ceil_seconds(value: Any) -> Union[int, str]:
    """
    Return elapsed seconds rounded upward when a value is available.

    :param value: Raw elapsed seconds value.
    :return: Integer seconds rounded upward, or an empty string when unavailable.
    """

    if value in (None, ""):  # Preserve unavailable duration values
        return ""  # Return an empty value for absent timing data

    numeric_value = safe_float(value, 0.0)  # Convert the duration to float seconds

    if numeric_value < 0.0:  # Normalize negative duration values
        numeric_value = abs(numeric_value)  # Store the positive duration magnitude

    return int(math.ceil(numeric_value))  # Return upward-rounded integer seconds


def calculate_generated_count_from_ratio(original_count: int, ratio: float) -> int:
    """
    Calculate the synthetic sample count requested by a ratio.

    :param original_count: Source dataset row count.
    :param ratio: Requested generated-to-original ratio.
    :return: Deterministic synthetic sample count.
    """

    original_count_int = max(0, int(original_count))  # Normalize the source row count to a non-negative integer
    ratio_float = safe_float(ratio, 0.0)  # Normalize the requested ratio to a float

    if original_count_int == 0:  # Return zero when no source rows exist
        return 0  # Preserve the empty-source invariant

    if not 0.0 <= ratio_float <= 1.0:  # Preserve percentage-mode semantics.
        raise ValueError("Percentage generation ratios must be within [0.0, 1.0]")  # Refuse unsupported ratios.
    return int(math.floor(original_count_int * ratio_float))  # Allocate the requested share without inventing samples for zero ratios.


def distribute_generated_counts_by_ratio(class_distribution: Dict, total_preprocessed: int, target_n: int, small_class_threshold: int, small_class_min_samples: int) -> Dict:  # Allocate exact synthetic counts by class
    """
    Allocate deterministic per-class synthetic counts that sum to the target.

    :param class_distribution: Dictionary mapping class labels to source counts.
    :param total_preprocessed: Total number of preprocessed source rows.
    :param target_n: Requested total number of synthetic rows.
    :param small_class_threshold: Source-count threshold for small-class minimums.
    :param small_class_min_samples: Minimum synthetic rows for small classes when feasible.
    :return: Dictionary mapping class labels to synthetic counts.
    """

    target_n_int = max(0, int(target_n))  # Normalize target count to a non-negative integer
    total_preprocessed_int = max(1, int(total_preprocessed))  # Normalize denominator for proportional allocation
    class_items = [(label, max(0, int(count))) for label, count in class_distribution.items()]  # Normalize class counts for deterministic allocation

    if target_n_int == 0 or not class_items:  # Return zero allocation when no samples can be produced
        return {label: 0 for label, _count in class_items}  # Preserve labels while assigning zero synthetic rows

    base_counts = {label: 0 for label, _count in class_items}  # Initialize exact per-class allocation
    small_labels = [label for label, count in class_items if count < small_class_threshold]  # Identify classes eligible for a small-class floor
    minimum_total = len(small_labels) * max(0, int(small_class_min_samples))  # Compute total floor demand

    if small_labels and 0 < minimum_total <= target_n_int:  # Apply small-class floors only when the requested target can support them
        for label in small_labels:  # Iterate over eligible small classes
            base_counts[label] = max(0, int(small_class_min_samples))  # Assign the configured small-class floor

    remaining = target_n_int - sum(base_counts.values())  # Compute remaining rows after feasible floors

    if remaining <= 0:  # Return floor allocation when it exactly consumes the target
        return base_counts  # Preserve exact target total after feasible floors

    raw_targets = {label: (count / total_preprocessed_int) * target_n_int for label, count in class_items}  # Compute proportional target shares
    residual_weights = {label: max(0.0, raw_targets[label] - base_counts[label]) for label, _count in class_items}  # Compute allocation weights after floors
    residual_weight_total = sum(residual_weights.values())  # Sum residual weights for proportional allocation

    if residual_weight_total <= 0.0:  # Fallback when floors exceed proportional raw shares
        residual_weights = {label: float(count) for label, count in class_items}  # Use source class counts as fallback weights
        residual_weight_total = sum(residual_weights.values())  # Recompute fallback weight total

    if residual_weight_total <= 0.0:  # Handle degenerate all-zero class counts
        first_label = class_items[0][0]  # Select the first stable class label
        base_counts[first_label] += remaining  # Assign all remaining rows deterministically
        return base_counts  # Return exact allocation for degenerate input

    fractional_parts = []  # Initialize fractional remainder storage
    allocated_remaining = 0  # Track rows allocated by integer floors

    for label, _count in class_items:  # Allocate integer floors for each class
        raw_remaining = (residual_weights[label] / residual_weight_total) * remaining  # Compute raw residual allocation
        floor_remaining = int(math.floor(raw_remaining))  # Round residual allocation down deterministically
        base_counts[label] += floor_remaining  # Add integer residual allocation to class total
        allocated_remaining += floor_remaining  # Accumulate allocated residual rows
        fractional_parts.append((raw_remaining - floor_remaining, str(label), label))  # Store fractional part with stable tie-breaker

    rows_left = remaining - allocated_remaining  # Compute rows left after floor allocation
    fractional_parts.sort(key=lambda item: (-item[0], item[1]))  # Sort by largest remainder and stable label representation

    for index in range(rows_left):  # Distribute remaining rows by largest fractional remainder
        label = fractional_parts[index % len(fractional_parts)][2]  # Select class label for the remaining row
        base_counts[label] += 1  # Add one deterministic remainder row to the selected class

    return base_counts  # Return exact per-class synthetic row allocation


def gradient_penalty(critic, real_samples, fake_samples, labels, device, config: Optional[Dict] = None, args: Optional[Any] = None, epoch: int = 0, step: int = 0, component: str = "gradient_penalty"):
    """
    Compute the WGAN-GP gradient penalty.

    :param critic: critic network callable that accepts (samples, labels)
    :param real_samples: tensor of real samples (B, feature_dim)
    :param fake_samples: tensor of fake samples (B, feature_dim)
    :param labels: tensor of integer labels (B,)
    :param device: torch device to run computations on
    :param config: Optional configuration dictionary containing gradient penalty epsilon
    :param args: Optional runtime arguments for finite-state diagnostics
    :param epoch: One-based epoch number for finite-state diagnostics
    :param step: One-based training step for finite-state diagnostics
    :param component: Training component name for finite-state diagnostics
    :return: scalar gradient penalty term
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG or get_default_config()  # Use global or default config
        
        epsilon = max(0.0, safe_float(config.get("gradient_penalty", {}).get("epsilon", 1e-12), 1e-12))  # Resolve non-negative gradient penalty epsilon
        
        batch_size = real_samples.size(0)  # Get batch size from real samples
        alpha = torch.rand(batch_size, 1, device=device)  # Sample random interpolation factors
        alpha = alpha.expand_as(real_samples)  # Expand alpha to match feature shape
        interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)  # Create interpolated samples
        interpolates.requires_grad_(True)  # Enable gradients for interpolated samples
        if args is not None:  # Emit context-rich failure at interpolation boundary
            require_finite_tensor(interpolates, "interpolated_samples", args, epoch, step, component)  # Reject invalid interpolation before critic evaluation
        d_interpolates = critic(interpolates, labels)  # Get critic scores for interpolated samples
        if args is not None:  # Emit context-rich failure at critic-score boundary
            require_finite_tensor(d_interpolates, "interpolated_scores", args, epoch, step, component)  # Reject invalid interpolation critic scores
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
        if args is not None:  # Emit context-rich failure at autograd boundary
            require_finite_tensor(grads, "interpolated_gradients", args, epoch, step, component)  # Reject invalid interpolation gradients
        grad_norm = torch.hypot(torch.linalg.vector_norm(grads, ord=2, dim=1), torch.full((batch_size,), math.sqrt(epsilon), device=device, dtype=grads.dtype))  # Compute stable L2 norm with existing epsilon semantics
        if args is not None:  # Emit context-rich failure at norm boundary
            require_finite_tensor(grad_norm, "interpolated_gradient_norms", args, epoch, step, component)  # Reject invalid interpolation gradient norms
        gp = ((grad_norm - 1) ** 2).mean()  # Calculate gradient penalty term
        if args is not None:  # Emit context-rich failure at penalty boundary
            require_finite_tensor(gp, "gradient_penalty", args, epoch, step, component)  # Reject invalid gradient penalty scalar
        return gp  # Return scalar gradient penalty
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


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

    data_aug_subdir = config.get("paths", {}).get("data_augmentation_dir", "Data_Augmentation")  # Get data augmentation base directory name from config
    plotting_subdir = config.get("plotting", {}).get("subdir", "Plots/metrics")  # Get metrics subdir from config
    out_path = Path(out_dir)  # Convert provided out_dir to a Path object for safe operations
    if out_path.name == data_aug_subdir:  # If out_dir already ends with the data_augmentation base directory
        save_dir = out_path / plotting_subdir  # Use out_dir plus plotting subdir (avoid duplicating Data_Augmentation)
    else:  # Otherwise, out_dir does not include data_augmentation yet
        save_dir = out_path / data_aug_subdir / plotting_subdir  # Append data augmentation base directory then plotting subdir
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
    Return a CUDA BF16 autocast context with explicit precision boundaries.

    This avoids referencing `torch.amp.autocast` directly (Pylance warning) and
    supports environments without CUDA.

    :param device_type: The device type ("cuda" or "cpu") to create autocast context for
    :param enabled: Whether to enable autocast context (default: True)
    :return: Explicit CUDA autocast context or a CPU nullcontext
    """

    try:
        if device_type == "cuda" and _torch_autocast is not None:  # Preserve disabled regions inside an outer autocast context
            return _torch_autocast(device_type, enabled=enabled, dtype=torch.bfloat16)  # Use BF16 exponent range without FP16 loss scaling
        return nullcontext()  # Return null context for CPU or when disabled
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def require_finite_tensor(tensor: torch.Tensor, tensor_name: str, args: Any, epoch: int, step: int, component: str) -> None:
    """
    Require a training tensor to contain only finite values.

    :param tensor: Tensor whose values must be finite.
    :param tensor_name: Diagnostic name for the tensor.
    :param args: Runtime arguments containing the current dataset path.
    :param epoch: One-based epoch number associated with the tensor.
    :param step: One-based training or generation step associated with the tensor.
    :param component: Training component that produced the tensor.
    :return: None.
    """

    detached_tensor = tensor.detach()  # Detach tensor before diagnostic inspection
    finite_mask = torch.isfinite(detached_tensor)  # Identify finite tensor elements
    if bool(finite_mask.all().item()):  # Return immediately when every value is finite
        return  # Preserve valid training behavior
    invalid_values = detached_tensor[~finite_mask]  # Select invalid values for diagnostics
    first_invalid = invalid_values.reshape(-1)[0].item()  # Capture first invalid value exactly
    nan_count = int(torch.isnan(detached_tensor).sum().item())  # Count NaN values
    positive_infinity_count = int(torch.isposinf(detached_tensor).sum().item())  # Count positive infinity values
    negative_infinity_count = int(torch.isneginf(detached_tensor).sum().item())  # Count negative infinity values
    finite_values = detached_tensor[finite_mask]  # Select finite values for bounded range diagnostics
    if finite_values.numel() > 0:  # Include nearby finite magnitude when available
        finite_stats = f", finite_min={finite_values.min().item()}, finite_max={finite_values.max().item()}, finite_mean={finite_values.float().mean().item()}"  # Summarize finite values without materializing tensors
    else:  # Handle tensors containing no finite values
        finite_stats = ", finite_values=0"  # Record absence of finite values
    dataset_name = Path(getattr(args, "csv_path", "unknown")).name  # Resolve current dataset filename
    message = f"Non-finite tensor detected: dataset={dataset_name}, epoch={epoch}, step={step}, component={component}, value={tensor_name}, first_invalid={first_invalid}, nan={nan_count}, posinf={positive_infinity_count}, neginf={negative_infinity_count}, shape={tuple(detached_tensor.shape)}, dtype={detached_tensor.dtype}, device={detached_tensor.device}{finite_stats}"  # Build precise failure diagnostic
    safe_critical(message)  # Persist diagnostic through existing logging
    raise FloatingPointError(message)  # Stop before invalid state propagates


def require_finite_model_state(model: nn.Module, state_name: str, args: Any, epoch: int, step: int, component: str, gradients: bool = False) -> None:
    """
    Require model parameters or gradients to contain only finite values.

    :param model: Model whose parameters or gradients must be finite.
    :param state_name: Diagnostic name for the inspected model state.
    :param args: Runtime arguments containing the current dataset path.
    :param epoch: One-based epoch number associated with the model state.
    :param step: One-based training or generation step associated with the model state.
    :param component: Training component associated with the model state.
    :param gradients: Inspect gradients when true, otherwise inspect parameters.
    :return: None.
    """

    named_tensors = [(parameter_name, parameter.grad if gradients else parameter) for parameter_name, parameter in model.named_parameters()]  # Collect model tensors for one-pass validation
    named_tensors = [(parameter_name, tensor) for parameter_name, tensor in named_tensors if tensor is not None]  # Remove unavailable gradients
    if not named_tensors:  # Return when no model tensors are available
        return  # Preserve valid empty-state behavior
    finite_states = torch.stack([torch.isfinite(tensor.detach()).all() for _, tensor in named_tensors])  # Combine tensor validity flags before device synchronization
    if bool(finite_states.all().item()):  # Return after one synchronization when all model tensors are finite
        return  # Preserve valid model behavior
    for parameter_name, tensor in named_tensors:  # Identify exact invalid model tensor only after aggregate failure
        require_finite_tensor(tensor, f"{state_name}.{parameter_name}", args, epoch, step, component)  # Raise precise diagnostic for invalid model state


def require_finite_optimizer_state(optimizer: torch.optim.Optimizer, optimizer_name: str, args: Any, epoch: int, step: int) -> None:
    """
    Require optimizer tensor state to contain only finite values.

    :param optimizer: Optimizer whose tensor state must be finite.
    :param optimizer_name: Diagnostic name for the optimizer.
    :param args: Runtime arguments containing the current dataset path.
    :param epoch: One-based epoch number associated with the optimizer state.
    :param step: One-based training step associated with the optimizer state.
    :return: None.
    """

    for parameter_index, parameter_state in enumerate(optimizer.state.values()):  # Iterate restored optimizer parameter states
        for state_name, state_value in parameter_state.items():  # Iterate each optimizer state value
            if torch.is_tensor(state_value):  # Inspect only tensor-backed optimizer state
                require_finite_tensor(state_value, f"{optimizer_name}[{parameter_index}].{state_name}", args, epoch, step, optimizer_name)  # Reject invalid optimizer state before use


def require_finite_saved_state(saved_state: Any, state_name: str, args: Any, epoch: int, component: str) -> None:
    """
    Require every numeric value in a saved training state to be finite.

    :param saved_state: Nested saved state containing tensors or scalar values.
    :param state_name: Diagnostic path for the saved state.
    :param args: Runtime arguments containing the current dataset path.
    :param epoch: Saved epoch associated with the state.
    :param component: Training component associated with the state.
    :return: None.
    """

    if torch.is_tensor(saved_state):  # Inspect tensor values directly
        require_finite_tensor(saved_state, state_name, args, epoch, 0, component)  # Reject invalid saved tensor values
    elif isinstance(saved_state, float):  # Inspect floating-point metadata
        require_finite_tensor(torch.tensor(saved_state), state_name, args, epoch, 0, component)  # Reject invalid saved scalar values
    elif isinstance(saved_state, dict):  # Traverse mapping values
        for nested_name, nested_state in saved_state.items():  # Visit every nested mapping entry
            require_finite_saved_state(nested_state, f"{state_name}.{nested_name}", args, epoch, component)  # Validate nested mapping state
    elif isinstance(saved_state, (list, tuple)):  # Traverse sequence values
        for nested_index, nested_state in enumerate(saved_state):  # Visit every nested sequence entry
            require_finite_saved_state(nested_state, f"{state_name}[{nested_index}]", args, epoch, component)  # Validate nested sequence state


def extract_saved_epoch(checkpoint_path: Path) -> int:
    """
    Extract numeric epoch from a generator checkpoint path.

    :param checkpoint_path: Generator checkpoint path containing an epoch suffix.
    :return: Numeric checkpoint epoch.
    """

    return int(checkpoint_path.stem.rsplit("epoch", 1)[-1])  # Parse numeric epoch suffix for chronological selection


def select_finite_saved_pair(checkpoint_files: List[Path], checkpoint_dir: Path, checkpoint_prefix: str, csv_path_obj: Path, args: Any, scaler: Any) -> tuple:
    """
    Select the newest complete saved epoch whose training state is finite.

    :param checkpoint_files: Generator artifact paths available for the dataset.
    :param checkpoint_dir: Directory containing saved model artifacts.
    :param checkpoint_prefix: Dataset-specific saved artifact prefix.
    :param csv_path_obj: Dataset path used in diagnostics.
    :param args: Runtime arguments containing the current dataset path.
    :param scaler: AMP gradient scaler when mixed precision is active.
    :return: Generator and discriminator paths for the newest usable epoch.
    """

    candidate_errors = []  # Preserve rejected saved-pair diagnostics
    for g_candidate_path in sorted(checkpoint_files, key=extract_saved_epoch, reverse=True):  # Visit saved epochs from newest to oldest
        epoch_num = extract_saved_epoch(g_candidate_path)  # Resolve numeric saved epoch
        d_candidate_path = checkpoint_dir / f"{checkpoint_prefix}_discriminator_epoch{epoch_num}.pt"  # Resolve paired discriminator path
        print(f"{BackgroundColors.CYAN}Attempting to resume from epoch {epoch_num}...{Style.RESET_ALL}")  # Notify resume attempt
        try:  # Validate the pair before mutating live training state
            if not d_candidate_path.exists():  # Reject an unpaired generator artifact
                raise FileNotFoundError(f"Discriminator checkpoint not found: {d_candidate_path}")  # Preserve precise missing-pair diagnostic
            g_candidate = torch.load(g_candidate_path, map_location="cpu", weights_only=False)  # Load generator candidate on CPU for validation
            d_candidate = torch.load(d_candidate_path, map_location="cpu", weights_only=False)  # Load discriminator candidate on CPU for validation
            verify_wgangp_provenance(g_candidate.get("provenance"), args, str(g_candidate_path))  # Reject mismatched source or training hyperparameters before restore.
            require_finite_saved_state(g_candidate["state_dict"], "parameters", args, epoch_num, "saved_generator")  # Reject invalid generator parameters
            require_finite_saved_state(d_candidate["state_dict"], "parameters", args, epoch_num, "saved_discriminator")  # Reject invalid discriminator parameters
            if "opt_G_state" in g_candidate:  # Validate available generator optimizer state
                require_finite_saved_state(g_candidate["opt_G_state"], "optimizer", args, epoch_num, "saved_generator")  # Reject invalid generator optimizer values
            if "opt_D_state" in d_candidate:  # Validate available discriminator optimizer state
                require_finite_saved_state(d_candidate["opt_D_state"], "optimizer", args, epoch_num, "saved_discriminator")  # Reject invalid discriminator optimizer values
            if scaler is not None and "scaler_state" in g_candidate:  # Validate available AMP scaler state
                require_finite_saved_state(g_candidate["scaler_state"], "scaler", args, epoch_num, "saved_generator")  # Reject invalid AMP scaler values
            return g_candidate_path, d_candidate_path  # Return the newest finite saved pair
        except (OSError, RuntimeError, ValueError, TypeError, KeyError, EOFError, FloatingPointError) as candidate_error:  # Handle expected saved-state failures
            candidate_message = f"Saved epoch {epoch_num} is unusable for {csv_path_obj.name}: {candidate_error}"  # Build precise recovery diagnostic
            candidate_errors.append(candidate_message)  # Preserve failure for terminal reporting
            safe_warning(candidate_message)  # Persist rejected-pair diagnostic through existing logging
            print(f"{BackgroundColors.YELLOW}⚠ {candidate_message}{Style.RESET_ALL}")  # Display rejected-pair diagnostic

    failure_summary = "; ".join(candidate_errors)  # Combine all rejected-pair diagnostics
    raise RuntimeError(f"Checkpoint resume failed for {csv_path_obj.name}: {failure_summary}")  # Prevent training from an invalid or partial state


def release_cuda_memory() -> None:
    """
    Release unused CUDA cache after a failed model operation.

    :return: None.
    """

    try:  # Attempt CUDA cache cleanup without masking original failures
        if torch.cuda.is_available():  # Run cleanup only when CUDA is available
            torch.cuda.empty_cache()  # Release unused cached CUDA allocations
    except Exception as cleanup_error:  # Preserve original exception when CUDA cleanup fails
        safe_warning(f"CUDA cleanup failed: {cleanup_error}")  # Report cleanup failure through existing logging


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


def resolve_results_csv_path(args, config: Dict) -> Optional[Path]:
    """
    Resolve the absolute path to the results CSV from runtime arguments and configuration.

    :param args: Parsed arguments namespace containing csv_path attribute.
    :param config: Configuration dictionary with paths section for directory names.
    :return: Absolute Path to the results CSV file, or None when resolution fails.
    """

    try:
        if not getattr(args, "csv_path", None):  # Verify csv_path is set on args before proceeding
            return None  # Cannot resolve path without source CSV
        csv_path_obj = Path(args.csv_path)  # Create Path object from csv_path
        data_aug_subdir = config.get("paths", {}).get("data_augmentation_dir", "Data_Augmentation")  # Read data augmentation base directory name from config
        data_aug_dir = csv_path_obj.parent / data_aug_subdir  # Construct data augmentation directory under dataset folder
        return data_aug_dir / "data_augmentation_results.csv"  # Return absolute path to results CSV
    except Exception:  # If resolution fails for any reason
        return None  # Return None to indicate path resolution failure


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


def send_file_saved_and_timing_messages(args: Any, config: Dict) -> None:
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
            derived = csv_obj.parent / config.get("paths", {}).get("data_augmentation_dir", "Data_Augmentation") / config.get("paths", {}).get("data_augmentation_sample_dir", "Samples") / f"{csv_obj.stem}{suffix}.csv"  # Build derived path in data augmentation samples directory
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
        send_telegram_message(TELEGRAM_BOT, msg)  # Send composed summary by Telegram using shared function
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

    try:  # Guard function with try/except to follow project style
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
        free_gb = get_available_disk_space_gb(config)  # Retrieve available disk space in GB via function
        free_bytes = int(safe_float(free_gb, 0.0) * (1024.0 ** 3))  # Convert GB back to bytes for threshold comparison
        required_bytes = total_dataset_size * 1.1  # Minimum required free space is twice the total dataset size
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


def normalize_args_and_setup_hardware(args: Any, config: Dict) -> tuple:
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
    configured_batch_size = int(  # Resolve unscaled batch size from merged configuration
        config.get("training", {}).get("batch_size", args.batch_size)
    )
    args.batch_size = configured_batch_size  # Reset batch size before hardware scaling
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
        
    batch_size_cli_override = bool(config.get("training", {}).get("batch_size_cli_override", False))  # Detect explicit CLI batch-size precedence
    batch_multiplier = 1 if batch_size_cli_override else (min(8, max(1, 2 * gpu_count)) if gpu_count > 0 else 1)  # Scale configured batch size only when CLI did not override it
    scaled_batch = configured_batch_size * batch_multiplier  # Compute scaled batch size once
    args.batch_size = int(scaled_batch)  # Apply scaled batch size to args
    args.use_amp = bool(args.use_amp and gpu_count > 0 and _torch_autocast is not None)  # Honor explicit AMP configuration only when CUDA autocast is available
    if args.use_amp:  # Resolve BF16 support across every DataParallel device
        for device_index in range(gpu_count):  # Inspect each participating CUDA device
            with torch.cuda.device(device_index):  # Query support on the selected device
                if not hasattr(torch.cuda, "is_bf16_supported") or not torch.cuda.is_bf16_supported():  # Require supported BF16 arithmetic
                    args.use_amp = False  # Select FP32 when BF16 is unavailable
                    print(f"{BackgroundColors.YELLOW}BF16 AMP unavailable on CUDA device {device_index}; using FP32 training{Style.RESET_ALL}")  # Report precision fallback explicitly
                    break  # Stop after discovering an unsupported device
    suggested_workers = min(max(1, (os.cpu_count() or 1) // 2), 32)  # Suggest a conservative default for num_workers
    file_progress_prefix = getattr(args, "file_progress_prefix", f"{BackgroundColors.CYAN}[1/1]{Style.RESET_ALL}")  # Build colored prefix (default single-file)
    
    print(f"{BackgroundColors.GREEN}Detected {gpu_count} GPUs.{Style.RESET_ALL}")  # Print GPU count
    print(f"{BackgroundColors.GREEN}Using DataParallel: {use_dataparallel}{Style.RESET_ALL}")  # Print whether DataParallel will be used
    print(f"{BackgroundColors.GREEN}Batch size: {BackgroundColors.CYAN}{args.batch_size}{Style.RESET_ALL}")  # Print effective batch size after scaling
    print(f"{BackgroundColors.GREEN}Hardware suggested num_workers: {BackgroundColors.CYAN}{suggested_workers}{Style.RESET_ALL}")  # Print hardware-derived worker suggestion
    print(f"{BackgroundColors.GREEN}AMP enabled: {BackgroundColors.CYAN}{args.use_amp}{Style.RESET_ALL}")  # Print AMP usage
    print(f"{BackgroundColors.GREEN}cuDNN benchmark: {BackgroundColors.CYAN}{torch.backends.cudnn.benchmark}{Style.RESET_ALL}")  # Print cuDNN benchmark status
    send_telegram_message(TELEGRAM_BOT, compose_training_start_message(args, file_progress_prefix))  # Telegram start with colored prefix and file statistics

    print(f"{BackgroundColors.GREEN}Device: {BackgroundColors.CYAN}{device.type.upper()}{Style.RESET_ALL}")  # Print device type
    if args.use_amp and device.type == "cuda":  # If AMP enabled and using CUDA
        print(f"{BackgroundColors.GREEN}Using BF16 AMP for generator training without loss scaling; WGAN-GP critic remains FP32{Style.RESET_ALL}")  # Print resolved precision boundary
    if args.compile:  # If torch.compile requested
        print(f"{BackgroundColors.GREEN}torch.compile() requested for optimized execution{Style.RESET_ALL}")  # Print compile request detail

    return device, training_start_time, file_start_time, epoch_milestones, file_progress_prefix  # Return hardware setup results


def build_class_reference_correlations(features: torch.Tensor, labels: np.ndarray, sample_limit: int = 4096, seed: int = 42) -> Dict[int, tuple]:
    """Fit stable class-conditional correlation targets from *real* encoded training rows.

    Compared with V5's volatile 100-200-row minibatch target, this fixed reservoir
    substantially reduces noise. Sampling does not modify the original dataset.
    Targets are computed once per file and transferred to the selected device once.
    """
    if sample_limit < 128:
        raise ValueError("correlation_reference_rows_per_class must be at least 128")
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) != len(features):
        raise ValueError("Correlation reference labels do not match the encoded features")
    rng = np.random.default_rng(int(seed))
    targets = {}
    for class_id in np.unique(labels):
        rows = np.flatnonzero(labels == class_id)
        if len(rows) < 32:
            continue
        chosen = rng.choice(rows, size=min(sample_limit, len(rows)), replace=False)
        data = features[torch.as_tensor(chosen, dtype=torch.long)].detach().cpu().numpy().astype(np.float64)
        active = np.flatnonzero(np.std(data, axis=0) > 0.05)
        if len(active) < 2:
            continue
        centered = data[:, active] - np.mean(data[:, active], axis=0)
        scale = np.sqrt(np.mean(centered ** 2, axis=0) + 1e-4)
        standardized = centered / scale
        correlation = (standardized.T @ standardized) / len(data)
        if np.isfinite(correlation).all():
            targets[int(class_id)] = (active.astype(np.int64), np.asarray(correlation, dtype=np.float32))
    return targets


def build_class_reference_moments(features: torch.Tensor, labels: np.ndarray, sample_limit: int = 4096, seed: int = 42) -> Dict[int, tuple]:
    """Build reproducible, class-specific mean/std targets from real encoded training rows.

    The reference is held fixed throughout training, rather than changing with each
    minibatch and randomly oversampling rare flows. Target statistics are detached;
    only generated samples receive gradients. No additional data partition is made.
    """
    if not 128 <= int(sample_limit) <= 100000:
        raise ValueError("moment_reference_rows_per_class must be between 128 and 100000")
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) != len(features):
        raise ValueError("Moment reference labels do not match the encoded features")
    rng = np.random.default_rng(int(seed) + 173)
    targets = {}
    for class_id in np.unique(labels):
        rows = np.flatnonzero(labels == class_id)
        if len(rows) < 32:
            continue
        chosen = rng.choice(rows, size=min(int(sample_limit), len(rows)), replace=False)
        values = features[torch.as_tensor(chosen, dtype=torch.long)].detach().cpu().numpy().astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite real moment reference for class {class_id}")
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        # A rare source mode cannot reliably appear in every 64-256-row generator batch.
        # Skip *standard-deviation* pressure for near-deterministic columns rather than
        # forcing the generator to invent rare traffic patterns in each minibatch.
        reliable_std = np.empty(values.shape[1], dtype=bool)
        for column_index in range(values.shape[1]):
            _, counts = np.unique(values[:, column_index], return_counts=True)
            reliable_std[column_index] = (counts.max() / len(values) <= 0.985)
        targets[int(class_id)] = (mean.astype(np.float32), std.astype(np.float32), reliable_std)
    return targets


def class_reference_moment_loss(fake_x: torch.Tensor, labels: torch.Tensor, targets: Dict[int, tuple]) -> torch.Tensor:
    """Match per-class means and reliably measurable standard deviations.

    A reference feature with >98.5% identical values may contain rare valid modes;
    demanding its full-reference variance in *every* small generator batch creates
    noisy, unrealistic gradients. Match its mean, but do not penalize the batch
    standard deviation. Constant or sparse columns remain controlled by WGAN-GP.
    """
    loss = fake_x.float().new_zeros(())
    groups = 0
    for class_id, target in targets.items():
        target_mean, target_std = target[:2]
        std_reliable = target[2] if len(target) > 2 else None
        selected = labels == int(class_id)
        if int(selected.sum()) < 8:
            continue
        fake = fake_x[selected].float()
        mean = fake.mean(dim=0)
        std = torch.sqrt(fake.var(dim=0, unbiased=False) + 1e-6)
        scale = target_std.clamp(min=0.5)
        mean_loss = (((mean - target_mean) / scale).square()).mean()
        std_errors = ((std - target_std) / scale).square()
        if std_reliable is not None:
            reliable = std_reliable.to(device=fake_x.device, dtype=torch.bool)
            if bool(reliable.any()):
                std_loss = std_errors[reliable].mean()
            else:
                std_loss = std_errors.new_zeros(())
        else:
            std_loss = std_errors.mean()
        loss = loss + mean_loss + std_loss
        groups += 1
    return loss / groups if groups else loss


def class_reference_correlation_loss(fake_x: torch.Tensor, labels: torch.Tensor, reference_targets: Dict[int, tuple]) -> torch.Tensor:
    """Match synthetic per-class correlations to stable, independent real-data reservoirs."""
    loss = fake_x.float().new_zeros(())
    groups = 0
    for class_id, (feature_indices, correlation) in reference_targets.items():
        selected = labels == int(class_id)
        if int(selected.sum()) < 16:
            continue
        x = fake_x[selected].float().index_select(1, feature_indices)
        centered = x - x.mean(dim=0, keepdim=True)
        normalized = centered / (centered.square().mean(dim=0, keepdim=True) + 1e-4).sqrt()
        fake_corr = normalized.T @ normalized / len(x)
        upper = torch.triu_indices(fake_corr.shape[0], fake_corr.shape[1], offset=1, device=x.device)
        loss = loss + (fake_corr[upper[0], upper[1]] - correlation[upper[0], upper[1]]).square().mean()
        groups += 1
    return loss / groups if groups else loss


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

    args._mixed_categorical_groups = list(dataset.scaler.categorical_groups)  # Define the exact V5 output groups before model initialization.
    args._mixed_categorical_class_masks = dataset.scaler.categorical_class_masks
    if float(config.get("training", {}).get("correlation_fidelity_weight", 2.0)) > 0:
        reference_rows = int(config.get("training", {}).get("correlation_reference_rows_per_class", 4096))
        reference_targets = build_class_reference_correlations(dataset.X, dataset.labels, sample_limit=reference_rows, seed=int(getattr(args, "seed", 42)))
        global V6_REFERENCE_CORRELATIONS
        V6_REFERENCE_CORRELATIONS = {
            class_id: (torch.as_tensor(indices, dtype=torch.long, device=device), torch.as_tensor(corr, dtype=torch.float32, device=device))
            for class_id, (indices, corr) in reference_targets.items()
        }
        print(f"[V6] Stable real-data correlation targets: {len(reference_targets)} classes, <= {reference_rows} rows/class; "
              f"correlation_fidelity_weight={config.get('training', {}).get('correlation_fidelity_weight', 2.0)}")
    else:
        V6_REFERENCE_CORRELATIONS = {}
    # V6.1: fixed per-class moment targets complement stable V6 correlations.
    # Keep source statistics in runtime only, never in args or checkpoint payloads.
    global V61_REFERENCE_MOMENTS
    moment_weight = float(config.get("training", {}).get("moment_fidelity_weight", 0.10))
    if not np.isfinite(moment_weight) or moment_weight < 0:
        raise ValueError("moment_fidelity_weight must be finite and nonnegative")
    V61_REFERENCE_MOMENTS = {}
    if moment_weight > 0:
        moment_rows = int(config.get("training", {}).get("moment_reference_rows_per_class", 4096))
        moments = build_class_reference_moments(dataset.X, dataset.labels, sample_limit=moment_rows, seed=int(getattr(args, "seed", 42)))
        V61_REFERENCE_MOMENTS = {
            class_id: (torch.as_tensor(mean, dtype=torch.float32, device=device),
                       torch.as_tensor(std, dtype=torch.float32, device=device),
                       torch.as_tensor(reliable, dtype=torch.bool, device=device))
            for class_id, (mean, std, reliable) in moments.items()
        }
        print(f"[V6.2] Sparse-mode-aware real moment targets: {len(moments)} classes, <= {moment_rows} rows/class; moment_fidelity_weight={moment_weight}")
    num_workers = int(config.get("dataloader", {}).get("num_workers", 8))  # Get base num_workers from config and cast to int
    num_workers_cli_override = bool(config.get("dataloader", {}).get("num_workers_cli_override", False))  # Detect explicit CLI worker precedence
    if not num_workers_cli_override:  # Auto-adjust workers only when CLI did not override them
        num_workers = adjust_num_workers_for_file(args.csv_path, num_workers, config)  # Adjust num_workers based on file size and system RAM
    else:  # Keep explicit CLI worker value unchanged
        print(f"{BackgroundColors.GREEN}Using explicit num_workers: {BackgroundColors.CYAN}{num_workers}{Style.RESET_ALL}")  # Print deterministic CLI worker value
    
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

    class_sampler = ClassAwareFlowSampler(dataset.labels, seed=int(getattr(args, "seed", 42)), exponent=float(config.get("quality", {}).get("class_sampling_exponent", 0.5)))  # Improve real-class exposure for imbalanced attack data.
    print(f"[TRAINING] Real class sampler: prevalence exponent={config.get('quality', {}).get('class_sampling_exponent', 0.5)} | Classes={len(class_sampler.class_ids)}")  # Record sampling strategy for reproducibility.
    dataloader = DataLoader(
        dataset,  # Dataset object to load data from
        batch_size=args.batch_size,  # Batch size for training
        sampler=class_sampler,  # Sample real rows by tempered class frequencies without materializing complete epoch indexes.
        shuffle=False,  # Avoid combining a shuffle sampler with the explicit class sampler.
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
        n_resblocks=args.n_resblocks,  # Number of residual blocks in generator
        categorical_groups=getattr(args, "_mixed_categorical_groups", []),  # Preserve structural mixed-type real/fake one-hot locations
        categorical_class_masks=getattr(args, "_mixed_categorical_class_masks", []),
        categorical_temperature=float(config.get("generation", {}).get("categorical_temperature", 1.25)),
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
    elif args.compile:  # If compile requested with DataParallel
        print(f"{BackgroundColors.YELLOW}torch.compile() skipped because DataParallel wraps the models{Style.RESET_ALL}")  # Report existing DataParallel compile limitation

    scaler = None  # BF16 and FP32 training do not require dynamic loss scaling

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
        if not math.isfinite(scaler.get_scale()) or scaler.get_scale() <= 0.0:  # Verify restored AMP scale is finite and positive
            raise ValueError(f"Invalid AMP scaler state in {g_checkpoint_path}")  # Stop before using an invalid restored scale
        print(f"{BackgroundColors.GREEN}✓ Restored AMP scaler state{Style.RESET_ALL}")  # Confirm scaler restoration
    elif "scaler_state" in g_checkpoint:  # Preserve model and optimizer restore when migrating from scaled FP16
        print(f"{BackgroundColors.GREEN}Legacy FP16 loss scale not restored: current training uses unscaled BF16 or FP32{Style.RESET_ALL}")  # Report precision migration without reviving scale growth
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
        raise FileNotFoundError(f"Discriminator checkpoint not found: {d_checkpoint_path}")  # Prevent resume with an unpaired discriminator state


def regenerate_missing_training_plot(
    csv_path_obj: Path,
    metrics_loaded: bool,
    metrics_history: Dict,
    config: Dict,
) -> None:
    """
    Regenerate training metrics plot from metrics history when the plot file is missing.

    :param csv_path_obj: Path object for the input CSV file used to derive plot filename.
    :param metrics_loaded: Whether metrics history was successfully loaded from checkpoint.
    :param metrics_history: Dictionary of tracked training metrics for plot generation.
    :param config: Configuration dictionary used to resolve the plot directory.
    :return: None
    """

    config_paths = config.get("paths", {})  # Read paths
    data_aug_subdir = config_paths.get("data_augmentation_dir", "Data_Augmentation")  # Read data augmentation directory
    plot_dir = csv_path_obj.parent / data_aug_subdir  # Build plot base directory
    plot_filename = csv_path_obj.stem + "_training_metrics.png"  # Plot filename
    plot_path = resolve_plot_save_directory(str(plot_dir), config) / plot_filename  # Build metrics plot path
    if not plot_path.exists():  # If plot doesn't exist
        if metrics_loaded and len(metrics_history.get("steps", [])) > 0:  # If metrics available
            print(f"{BackgroundColors.YELLOW}Training metrics plot not found, generating from metrics history...{Style.RESET_ALL}")  # Notify plot generation
            os.makedirs(plot_dir, exist_ok=True)  # Ensure directory exists
            plot_training_metrics(metrics_history, str(plot_dir), plot_filename, config)  # Generate plot
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
        checkpoint_dir = csv_path_obj.parent / "Data_Augmentation" / "Checkpoints" / FLOW_PREPROCESSING_CHECKPOINT_TAG  # Resume only compatible class-conditional preprocessing checkpoints.
        checkpoint_prefix = csv_path_obj.stem  # Expected filename prefix

        if checkpoint_dir.exists():  # If checkpoint directory exists
            checkpoint_files = list(checkpoint_dir.glob(f"{checkpoint_prefix}_generator_epoch*.pt"))  # Find matching checkpoints

            if checkpoint_files:  # If checkpoints found for this file
                print(f"{BackgroundColors.CYAN}Found existing checkpoints for {csv_path_obj.name}{Style.RESET_ALL}")  # Notify checkpoint discovery
                g_checkpoint_path, d_checkpoint_path = select_finite_saved_pair(checkpoint_files, checkpoint_dir, checkpoint_prefix, csv_path_obj, args, scaler)  # Select newest usable saved pair
                try:  # Restore the prevalidated pair into live training state
                    g_checkpoint, start_epoch = load_and_restore_generator_state(g_checkpoint_path, device, G, opt_G, scaler)  # Load and restore generator state from saved pair
                    metrics_history, step, metrics_loaded = restore_metrics_from_checkpoint_or_json(g_checkpoint, checkpoint_dir, checkpoint_prefix, metrics_history, step)  # Restore metrics from saved state or JSON fallback
                    load_and_restore_discriminator_state(d_checkpoint_path, device, D, opt_D)  # Load and restore discriminator state from saved pair
                    regenerate_missing_training_plot(  # Regenerate plot
                        csv_path_obj,  # Pass source path
                        metrics_loaded,  # Pass metrics state
                        metrics_history,  # Pass metrics data
                        config,  # Pass configuration
                    )  # Finish regeneration
                    print(f"{BackgroundColors.GREEN}✓ Resuming training from epoch {start_epoch} (step {step}){Style.RESET_ALL}")  # Confirm resume point
                except Exception as e:  # If restoration fails after validation
                    print(f"{BackgroundColors.YELLOW}⚠ Failed to load checkpoint: {e}{Style.RESET_ALL}")  # Warn about load failure
                    raise RuntimeError(f"Checkpoint resume failed for {csv_path_obj.name}") from e  # Stop instead of training from partially restored state
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


def execute_discriminator_training_steps(G, D, opt_D, scaler, real_x, labels, device: torch.device, args, config: Dict, n_classes: int, epoch: int, step: int) -> tuple:
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
    :param epoch: Current zero-based epoch index.
    :param step: Current zero-based global training step.
    :return: Tuple of (loss_D, gp, d_real_score, d_fake_score) as tensors.
    """

    loss_D = torch.tensor(0.0, device=device)  # Initialize discriminator loss
    gp = torch.tensor(0.0, device=device)  # Initialize gradient penalty
    d_real_score = torch.tensor(0.0, device=device)  # Initialize real score tracker
    d_fake_score = torch.tensor(0.0, device=device)  # Initialize fake score tracker
    for critic_step in range(args.critic_steps):  # Train discriminator multiple steps
        component = f"discriminator[{critic_step + 1}]"  # Name current critic update for diagnostics
        require_finite_tensor(real_x, "real_samples", args, epoch + 1, step + 1, component)  # Reject invalid real samples before critic evaluation
        require_finite_tensor(labels, "labels", args, epoch + 1, step + 1, component)  # Reject invalid conditioning labels before critic evaluation
        require_finite_model_state(D, "parameters", args, epoch + 1, step + 1, component)  # Reject invalid critic parameters before backward execution
        require_finite_optimizer_state(opt_D, f"discriminator_optimizer[{critic_step + 1}]", args, epoch + 1, step + 1)  # Reject invalid critic optimizer state before backward execution
        with autocast(device.type, enabled=bool(getattr(args, "use_amp", False))):  # Enable resolved BF16 AMP for sample generation
            z = torch.randn(args.batch_size, args.latent_dim, device=device)  # Sample noise for discriminator step
            fake_x = G(z, labels).detach()  # Generate fake samples and detach for discriminator
        require_finite_tensor(fake_x, "generated_samples", args, epoch + 1, step + 1, component)  # Reject invalid generated samples before critic reuse
        with autocast(device.type, enabled=False):  # Keep complete critic objective and backward graph in full precision
            d_real = D(real_x.float(), labels)  # Compute real critic scores in full precision
            d_fake = D(fake_x.float(), labels)  # Compute fake critic scores in full precision
            require_finite_tensor(d_real, "real_scores", args, epoch + 1, step + 1, component)  # Reject invalid real critic scores
            require_finite_tensor(d_fake, "fake_scores", args, epoch + 1, step + 1, component)  # Reject invalid fake critic scores
            gp = gradient_penalty(D, real_x.float(), fake_x.float(), labels, device, config, args, epoch + 1, step + 1, component)  # Compute complete gradient penalty in full precision
            loss_D = d_fake.float().mean() - d_real.float().mean() + args.lambda_gp * gp  # Calculate full-precision WGAN-GP discriminator loss
        require_finite_tensor(loss_D, "loss", args, epoch + 1, step + 1, component)  # Reject invalid discriminator loss before backward execution
        opt_D.zero_grad(set_to_none=True)  # Reset discriminator gradients to None for reduced memory overhead
        loss_D.backward()  # Backpropagate complete full-precision critic objective
        require_finite_model_state(D, "gradients", args, epoch + 1, step + 1, component, gradients=True)  # Reject invalid discriminator gradients before optimizer execution
        opt_D.step()  # Update discriminator parameters with unscaled full-precision gradients
        require_finite_model_state(D, "parameters", args, epoch + 1, step + 1, component)  # Reject invalid discriminator parameters immediately after update
        require_finite_optimizer_state(opt_D, f"discriminator_optimizer[{critic_step + 1}]", args, epoch + 1, step + 1)  # Reject invalid critic optimizer state immediately after update
        d_real_score = d_real.mean()  # Store average real score
        d_fake_score = d_fake.mean()  # Store average fake score
    return loss_D, gp, d_real_score, d_fake_score  # Return discriminator training results


def class_conditional_moment_loss(real_x: torch.Tensor, fake_x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Differentiable per-class moment matching on mixed numerical/one-hot feature representations.

    Compute separate moments for each class to avoid rewarding an incorrect class mixture.
    Small batches are skipped rather than comparing unreliable moments.
    """
    objective = fake_x.new_zeros((), dtype=torch.float32)
    groups = 0
    real = real_x.detach().float()
    fake = fake_x.float()
    for class_id in torch.unique(labels):
        selected = labels == class_id
        if int(selected.sum()) < 8:
            continue
        reference = real[selected]
        synthesized = fake[selected]
        reference_mean = reference.mean(dim=0)
        reference_std = torch.sqrt(reference.var(dim=0, unbiased=False) + 1e-6)
        synthetic_mean = synthesized.mean(dim=0)
        synthetic_std = torch.sqrt(synthesized.var(dim=0, unbiased=False) + 1e-6)
        # Avoid exploding gradients for genuinely rare binary features and constant real columns.
        scale = reference_std.clamp(min=0.5).detach()
        objective = objective + (((synthetic_mean - reference_mean) / scale).square().mean()
                                 + ((synthetic_std - reference_std) / scale).square().mean())
        groups += 1
    return objective / groups if groups else objective


def class_conditional_correlation_loss(real_x: torch.Tensor, fake_x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Match off-diagonal feature correlations within each sufficiently sampled class.

    Excludes constant real features; the independent moment loss handles variance collapse.
    Uses detached real reference statistics to keep the critic's real batch out of gradients.
    """
    objective = fake_x.new_zeros((), dtype=torch.float32)
    groups = 0
    real, fake = real_x.detach().float(), fake_x.float()
    for class_id in torch.unique(labels):
        selected = labels == class_id
        if int(selected.sum()) < 16:
            continue
        reference = real[selected]
        synthesized = fake[selected]
        reference_std = reference.std(dim=0, unbiased=False)
        features = reference_std > 0.05
        if int(features.sum()) < 2:
            continue
        reference = reference[:, features]
        synthesized = synthesized[:, features]
        r_centered = reference - reference.mean(0, keepdim=True)
        f_centered = synthesized - synthesized.mean(0, keepdim=True)
        r_normalized = r_centered / (r_centered.square().mean(0, keepdim=True) + 1e-4).sqrt()
        f_normalized = f_centered / (f_centered.square().mean(0, keepdim=True) + 1e-4).sqrt()
        r_corr = r_normalized.T @ r_normalized / len(reference)
        f_corr = f_normalized.T @ f_normalized / len(synthesized)
        upper = torch.triu_indices(r_corr.size(0), r_corr.size(1), offset=1, device=r_corr.device)
        objective = objective + (f_corr[upper[0], upper[1]] - r_corr[upper[0], upper[1]]).square().mean()
        groups += 1
    return objective / groups if groups else objective


def execute_generator_training_step(G, D, opt_G, scaler, device: torch.device, args, n_classes: int, epoch: int, step: int, real_labels: Optional[torch.Tensor] = None, real_x: Optional[torch.Tensor] = None, config: Optional[Dict] = None) -> torch.Tensor:
    """
    Execute a single generator training step.

    :param G: Generator model to train.
    :param D: Discriminator model for scoring fake samples.
    :param opt_G: Generator optimizer.
    :param scaler: AMP gradient scaler (may be None).
    :param device: Torch device for tensor allocation.
    :param args: Parsed arguments namespace with batch_size and latent_dim.
    :param n_classes: Number of label classes for conditional generation.
    :param epoch: Current zero-based epoch index.
    :param step: Current zero-based global training step.
    :return: Generator loss tensor.
    """

    require_finite_model_state(G, "parameters", args, epoch + 1, step + 1, "generator")  # Reject invalid generator parameters before backward execution
    require_finite_optimizer_state(opt_G, "generator_optimizer", args, epoch + 1, step + 1)  # Reject invalid generator optimizer state before backward execution
    with autocast(device.type, enabled=bool(getattr(args, "use_amp", False))):  # Enable resolved BF16 AMP for generator training
        z = torch.randn(args.batch_size, args.latent_dim, device=device)  # Sample noise for generator step
        gen_labels = real_labels if real_labels is not None else torch.randint(0, n_classes, (args.batch_size,), device=device)  # Align conditioning with real critic classes
        fake_x = G(z, gen_labels)  # Generate fake samples with generator
    require_finite_tensor(fake_x, "generated_samples", args, epoch + 1, step + 1, "generator")  # Reject invalid generated samples before loss calculation
    with autocast(device.type, enabled=False):  # Keep critic scoring in FP32 during generator training
        fake_scores = D(fake_x.float(), gen_labels)  # Preserve differentiability through the precision conversion
    require_finite_tensor(fake_scores, "fake_scores", args, epoch + 1, step + 1, "generator")  # Reject invalid fake critic scores
    g_loss = -fake_scores.float().mean()  # Calculate generator loss with full-precision reduction
    moment_weight = float((config or {}).get("training", {}).get("moment_fidelity_weight", 0.10))
    if not np.isfinite(moment_weight) or moment_weight < 0:
        raise ValueError("moment_fidelity_weight must be finite and nonnegative")
    if moment_weight and real_x is not None:
        if real_labels is None or real_x.shape != fake_x.shape:
            raise ValueError("Moment matching requires aligned real features and labels")
        if V61_REFERENCE_MOMENTS:
            g_loss = g_loss + moment_weight * class_reference_moment_loss(fake_x, gen_labels, V61_REFERENCE_MOMENTS)
        else:
            g_loss = g_loss + moment_weight * class_conditional_moment_loss(real_x, fake_x, gen_labels)
    corr_weight = float((config or {}).get("training", {}).get("correlation_fidelity_weight", 2.0))
    if not np.isfinite(corr_weight) or corr_weight < 0:
        raise ValueError("correlation_fidelity_weight must be finite and nonnegative")
    if corr_weight and real_x is not None:
        if real_labels is None or real_x.shape != fake_x.shape:
            raise ValueError("Correlation matching requires aligned real features and labels")
        stable_targets = V6_REFERENCE_CORRELATIONS
        if stable_targets:
            g_loss = g_loss + corr_weight * class_reference_correlation_loss(fake_x, gen_labels, stable_targets)
        else:
            g_loss = g_loss + corr_weight * class_conditional_correlation_loss(real_x, fake_x, gen_labels)
    require_finite_tensor(g_loss, "loss", args, epoch + 1, step + 1, "generator")  # Reject invalid generator loss before backward execution
    opt_G.zero_grad(set_to_none=True)  # Reset generator gradients to None for reduced memory overhead
    if scaler is not None:  # If using mixed precision
        scaler.scale(g_loss).backward()  # Scale loss and backpropagate
        scaler.unscale_(opt_G)  # Unscale generator gradients before finite validation
        require_finite_model_state(G, "gradients", args, epoch + 1, step + 1, "generator", gradients=True)  # Reject invalid generator gradients before optimizer execution
        scaler.step(opt_G)  # Update generator parameters with scaled gradients
        scaler.update()  # Update scaler for next iteration
    else:  # Standard precision
        g_loss.backward()  # Backpropagate generator loss
        require_finite_model_state(G, "gradients", args, epoch + 1, step + 1, "generator", gradients=True)  # Reject invalid generator gradients before optimizer execution
        opt_G.step()  # Update generator parameters
    require_finite_model_state(G, "parameters", args, epoch + 1, step + 1, "generator")  # Reject invalid generator parameters immediately after update
    require_finite_optimizer_state(opt_G, "generator_optimizer", args, epoch + 1, step + 1)  # Reject invalid generator optimizer state immediately after update
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


def run_batch_training_loop(G, D, opt_G, opt_D, scaler, device: torch.device, args, config: Dict, n_classes: int, pbar, step: int, metrics_history: Dict, total_steps: int, epoch: int, training_start_time: float, file_progress_prefix: str) -> int:
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
    :param training_start_time: wall-clock timestamp marking start of training loop
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
        require_finite_tensor(real_x, "real_samples", args, epoch + 1, step + 1, "input")  # Reject invalid input samples before model execution

        loss_D, gp, d_real_score, d_fake_score = execute_discriminator_training_steps(G, D, opt_D, scaler, real_x, labels, device, args, config, n_classes, epoch, step)  # Execute discriminator training steps with gradient penalty
        g_loss = execute_generator_training_step(G, D, opt_G, scaler, device, args, n_classes, epoch, step, real_labels=labels, real_x=real_x, config=config)  # Match class labels and moments to the actual real batch

        if step % args.log_interval == 0:  # Extract scalar metrics only at log interval to avoid per-batch CUDA synchronization
            _cached_loss_D, _cached_g_loss, _cached_gp, _cached_d_real, _cached_d_fake = record_training_step_metrics(step, args, metrics_history, loss_D, g_loss, gp, d_real_score, d_fake_score)  # Record metrics and cache scalar values

        completed_global_steps = (epoch * total_steps) + (batch_idx + 1)  # Count completed training batches across all epochs
        total_global_steps = max(1, int(args.epochs) * int(total_steps))  # Resolve total training batches across all epochs
        elapsed_seconds = max(0.0, time.time() - float(training_start_time))  # Measure total elapsed wall-clock training time
        remaining_seconds = ((elapsed_seconds / float(completed_global_steps)) * float(total_global_steps - completed_global_steps)) if completed_global_steps > 0 and total_global_steps > completed_global_steps else 0.0  # Estimate remaining wall-clock time from completed global batch progress
        finish_suffix = format_estimated_finish_time_suffix(remaining_seconds)  # Format overall estimated finish timestamp once per displayed update
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
            + f"D(fake): {_cached_d_fake:.4f}{get_cached_training_resource_suffix()}{finish_suffix}"  # Append cached CPU/RAM/GPU telemetry before the unchanged finish timestamp.
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
            hardware_str = build_hardware_csv_value(device)  # Build GA-style JSON hardware value for one CSV cell
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


def build_hardware_csv_value(device: torch.device) -> str:
    """
    Serialize hardware specifications for a CSV cell.

    :param device: Torch device used by the WGAN-GP run.
    :return: JSON string containing hardware specifications.
    """

    hardware_specs = get_hardware_specifications(device_used=device)  # Collect hardware specifications with WGAN-GP device context
    return json.dumps(hardware_specs, default=str)  # Serialize hardware specifications as one CSV cell


def count_persisted_output_rows(out_file: Union[str, Path]) -> int:
    """
    Count persisted data rows in an augmented CSV file.

    :param out_file: Path to the augmented CSV file.
    :return: Number of data rows persisted on disk.
    """

    out_path = Path(out_file)  # Normalize output path before opening
    with open(out_path, "r", newline="", encoding="utf-8") as file_obj:  # Open augmented CSV using the project CSV parsing mode
        reader = csv.reader(file_obj)  # Create standard CSV reader for quoted cell safety
        row_total = sum(1 for _ in reader)  # Count CSV records including the header
    persisted_rows = max(0, row_total - 1)  # Convert total records into data-row count
    if persisted_rows <= 0:  # Verify persisted augmented data rows exist
        raise ValueError(f"Augmented CSV {out_path} does not contain persisted data rows")  # Raise before writing a successful result row
    return persisted_rows  # Return actual persisted output row count


def validate_augmentation_sample_totals(args: Any, persisted_total_rows: int) -> None:
    """
    Validate generated output rows before result persistence.

    :param args: Runtime arguments namespace carrying augmentation count attributes.
    :param persisted_total_rows: Data-row count read from the persisted augmented CSV.
    :return: None.
    """

    original_rows = int(safe_float(getattr(args, "_last_original_num_samples", 0), 0.0))  # Read source row count from args
    generated_rows = int(safe_float(getattr(args, "_last_generated_num_samples", 0), 0.0))  # Read synthetic row count from args
    conceptual_total_rows = original_rows + generated_rows  # Calculate conceptual source plus generated population

    if original_rows < 0 or generated_rows < 0:  # Verify row counts are non-negative
        raise ValueError("Augmentation row counts must be non-negative")  # Raise before writing an invalid result row

    if persisted_total_rows != generated_rows:  # Verify persisted output count matches generated rows
        raise ValueError(  # Raise error
            f"Generated CSV row count {persisted_total_rows} does not equal generated {generated_rows}"  # Format error
        )  # Finish error

    args._last_total_num_samples_after_augmentation = int(conceptual_total_rows)  # Store total


def validate_completed_generation_row(row_runtime: Dict, results_cols_cfg: list) -> None:
    """
    Validate required completed generation result fields.

    :param row_runtime: Runtime row values keyed by schema column.
    :param results_cols_cfg: Ordered CSV schema columns.
    :return: None.
    """

    # Resuming a completed checkpoint performs no new epoch; epoch_time_s is then
    # legitimately unknown, and should be blank rather than blocking persistence.
    required_fields = ["original_file", "generated_file", "original_num_samples", "generated_num_samples", "total_num_samples_after_augmentation", "generated_ratio", "epoch", "generated_time_s", "testing_time_s"]
    missing_schema = [field for field in required_fields if field not in results_cols_cfg]  # Find required fields absent from schema
    if missing_schema:  # Verify required fields are present in configured schema
        raise ValueError(f"Results CSV schema is missing required fields: {missing_schema}")  # Raise before any row write
    missing_values = [field for field in required_fields if row_runtime.get(field) in (None, "")]  # Find required fields without values
    if missing_values:  # Verify successful result fields have real values
        raise ValueError(f"Completed generation row is missing required values: {missing_values}")  # Raise before any row write

    original_value = int(safe_float(row_runtime.get("original_num_samples"), 0.0))  # Convert source row count to integer
    generated_value = int(safe_float(row_runtime.get("generated_num_samples"), 0.0))  # Convert synthetic row count to integer
    total_value = int(safe_float(row_runtime.get("total_num_samples_after_augmentation"), 0.0))  # Convert total
    ratio_value = safe_float(row_runtime.get("generated_ratio"), 0.0)  # Convert generated ratio to float
    completed_epoch_value = safe_float(row_runtime.get("epoch"), 0.0)  # Convert completed epoch to numeric value
    epoch_time_value = safe_float(row_runtime.get("epoch_time_s"), 0.0)  # Convert completed epoch duration to numeric seconds
    metric_fields = ["critic_loss", "generator_loss", "gp", "D_real", "D_fake", "wasserstein"]  # Define WGAN-GP metric fields
    metric_present = any(row_runtime.get(field) not in (None, "") for field in metric_fields)  # Detect whether any final metric value is available
    missing_metrics = [field for field in metric_fields if row_runtime.get(field) in (None, "")] if metric_present else []  # Find missing metrics only when metric history exists

    if original_value < 0 or generated_value < 0:  # Verify row counts are non-negative
        raise ValueError("Completed generation row has negative sample counts")  # Raise before any row write
    if total_value != original_value + generated_value:  # Verify total
        raise ValueError("Completed generation row violates original plus generated sample total")  # Raise before any row write
    if original_value > 0 and abs(ratio_value - (generated_value / original_value)) > 1e-12:  # Verify generated ratio matches generated rows over source rows
        raise ValueError("Completed generation row has an inconsistent generated_ratio")  # Raise before any row write
    if completed_epoch_value <= 0.0:  # Verify completed epoch count is positive
        raise ValueError("Completed generation row has no completed epoch")  # Raise before any row write
    if row_runtime.get("epoch_time_s") not in (None, "") and epoch_time_value < 0.0:
        raise ValueError("Completed generation row has a negative epoch duration")
    if missing_metrics:  # Verify all WGAN-GP metrics are present when any final metric is present
        raise ValueError(f"Completed generation row is missing WGAN-GP metric values: {missing_metrics}")  # Raise before any row write


def update_results_csv_row(results_csv_path: Path, ordered_row: list, results_cols_cfg: list, original_file: str) -> None:
    """Atomically insert or replace exactly one successful generation result per source CSV.

    Existing records for other sources are preserved. Duplicate entries for the
    current source are collapsed, including historical entries containing the
    source's path instead of its basename. Results are committed after every
    successful file rather than at the end of a multi-file execution.

    An incompatible existing header is an error, never grounds for silently
    discarding historical results. A temporary file and an optional POSIX lock
    protect the destination from interrupted and concurrent writes.
    """
    results_csv_path = Path(results_csv_path)
    if not isinstance(results_cols_cfg, list) or not results_cols_cfg or len(set(results_cols_cfg)) != len(results_cols_cfg):
        raise ValueError("Results CSV schema must have unique, nonempty columns")
    if "original_file" not in results_cols_cfg:
        raise ValueError("Results CSV schema must include original_file for deduplication")
    if len(ordered_row) != len(results_cols_cfg):
        raise ValueError(f"Results row has {len(ordered_row)} fields but schema has {len(results_cols_cfg)}")

    def source_key(value: Any) -> str:
        return str(value or "").strip().replace("\\", "/").rstrip("/").split("/")[-1]

    key = source_key(original_file)
    if not key or key.lower() == "original_file":
        raise ValueError("Cannot upsert results without a valid original source filename")
    row_mapping = dict(zip(results_cols_cfg, ordered_row))
    if source_key(row_mapping.get("original_file")) != key:
        raise ValueError("Results row's original_file differs from the requested source")
    if not row_mapping.get("generated_file"):
        raise ValueError("Results row has no generated output filename")

    results_csv_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = results_csv_path.with_name(results_csv_path.name + ".lock")
    temp_name = None
    # Persistent separate lock inode avoids invalidating the lock after os.replace.
    with open(lock_path, "a+b") as lock_file:
        if fcntl is not None:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            retained: list = []
            if results_csv_path.exists() and results_csv_path.stat().st_size:
                with open(results_csv_path, "r", newline="", encoding="utf-8-sig") as source_file:
                    reader = csv.DictReader(source_file)
                    existing_columns = reader.fieldnames
                    if (not existing_columns or len(set(existing_columns)) != len(existing_columns)
                            or set(existing_columns) != set(results_cols_cfg)):
                        raise ValueError(
                            f"Existing results CSV has a different schema: {existing_columns!r}; "
                            "preserving the file rather than overwriting historical entries"
                        )
                    for old in reader:
                        if None in old:  # Overflow columns indicate malformed rows.
                            raise ValueError("Malformed historical results row has unexpected extra fields")
                        if not any(str(value or "").strip() for value in old.values()):
                            continue
                        if all(old.get(column) == column for column in results_cols_cfg):
                            continue  # Ignore historical repeated header rows.
                        if source_key(old.get("original_file")) != key:
                            retained.append([old[column] for column in results_cols_cfg])
            # Write and fsync a unique same-directory temporary file before replacing.
            with tempfile.NamedTemporaryFile(
                mode="w", newline="", encoding="utf-8", dir=results_csv_path.parent,
                prefix=results_csv_path.name + ".", suffix=".tmp", delete=False
            ) as destination:
                temp_name = destination.name
                writer = csv.writer(destination)
                writer.writerow(results_cols_cfg)
                writer.writerows(retained)
                writer.writerow(ordered_row)
                destination.flush()
                os.fsync(destination.fileno())
            if results_csv_path.exists():
                # Preserve an existing results file's permissions across replacement.
                import stat
                os.chmod(temp_name, stat.S_IMODE(results_csv_path.stat().st_mode))
            os.replace(temp_name, results_csv_path)
            temp_name = None
        finally:
            if temp_name is not None:
                try:
                    os.unlink(temp_name)
                except FileNotFoundError:
                    pass
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

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
        args._last_epoch_time = ceil_seconds(epoch_elapsed)  # Store last epoch elapsed as upward-rounded integer seconds
        args._last_completed_epoch = epoch + 1  # Store the completed epoch number for final generation results
    except Exception as _te:  # If timing calculation fails
        print(f"{BackgroundColors.YELLOW}Warning: failed to measure epoch time: {_te}{Style.RESET_ALL}")  # Warn but continue


def resolve_checkpoint_dir_and_prefix(args, config: Dict, epoch: int) -> tuple:
    """
    Resolve checkpoint directory path, filename prefix, and verify disk space availability.

    :param args: Parsed arguments namespace with csv_path, out_dir, and save_every.
    :param config: Configuration dictionary with dataset directories for disk space verification.
    :param epoch: Current epoch index (zero-based) for periodic save verification.
    :return: Tuple of (checkpoint_dir, checkpoint_prefix, skip_checkpoint).
    """

    checkpoint_dir = Path(args.out_dir) / "Checkpoints" / FLOW_PREPROCESSING_CHECKPOINT_TAG  # Default versioned checkpoint directory
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
                checkpoint_dir = csv_path_obj.parent / "Data_Augmentation" / "Checkpoints" / FLOW_PREPROCESSING_CHECKPOINT_TAG  # Persist weights separately from historical preprocessing checkpoints.
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
                checkpoint_dir = Path(args.out_dir) / "Checkpoints" / FLOW_PREPROCESSING_CHECKPOINT_TAG  # Create a versioned checkpoint directory in out_dir
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
        "model_feature_dim": dataset.scaler.model_feature_dim,  # Verify mixed feature width at generation.
        "categorical_groups": list(dataset.scaler.categorical_groups),  # Preserve categorical output layout.
        "categorical_class_masks": dataset.scaler.categorical_class_masks,  # Preserve exact class-conditional categorical domains.
        "label_encoder": dataset.label_encoder,  # Save label encoder for mapping
        "feature_cols": dataset.feature_cols,  # Save feature column names for generation
        "class_distribution": class_distribution,  # Save class distribution for percentage-based generation
        "metrics_history": metrics_history,  # Save metrics history for resume
        "args": vars(args),  # Save training arguments
        "provenance": build_wgangp_provenance(args),  # Bind saved weights to source identity and training configuration.
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


def send_epoch_telegram_notifications(args, telegram_enabled: bool, epoch: int, next_notify: int, progress_pct: int, training_start_time: float, file_progress_prefix: Optional[str] = None) -> int:
    """
    Send telegram progress notifications when epoch completion crosses percentage thresholds.

    :param args: parsed arguments namespace with epochs count and csv_path.
    :param telegram_enabled: whether telegram notifications are active.
    :param epoch: current epoch index (zero-based).
    :param next_notify: next percentage threshold to trigger notification.
    :param progress_pct: percentage increment between notifications.
    :param training_start_time: wall-clock timestamp marking start of training loop.
    :param file_progress_prefix: Optional colored prefix string like "[3/11]" to prepend to messages.
    :return: Updated next_notify threshold.
    """

    try:  # Guard notification logic to avoid interrupting training
        if telegram_enabled and args.epochs > 0:  # Only notify when enabled and epochs is positive
            percent = int(((epoch + 1) / float(args.epochs)) * 100)  # Compute percent completed after this epoch
            completed_epochs = max(1, epoch + 1)  # Keep epoch-based elapsed and ETA math valid after first completed epoch
            elapsed_seconds = max(0.0, time.time() - float(training_start_time))  # Measure total elapsed training time at notification boundary
            remaining_epochs = max(0, int(args.epochs) - completed_epochs)  # Compute remaining epochs for linear ETA estimate
            eta_seconds = (elapsed_seconds / float(completed_epochs)) * float(remaining_epochs) if remaining_epochs > 0 else 0.0  # Estimate remaining wall-clock time from completed epochs
            elapsed_label = calculate_execution_time(elapsed_seconds)  # Reuse existing human-readable duration formatter
            eta_label = calculate_execution_time(eta_seconds)  # Reuse existing human-readable duration formatter for ETA
            prefix = file_progress_prefix if file_progress_prefix is not None else getattr(args, "file_progress_prefix", f"{BackgroundColors.CYAN}[1/1]{Style.RESET_ALL}")  # Determine prefix from parameter or args fallback
            while percent >= next_notify and next_notify <= 100:  # Send notifications for each crossed threshold
                msg = (
                    f"{prefix} WGAN-GP training progress: {next_notify}% "  # Prepend prefix then short progress message text
                    f"({epoch+1}/{args.epochs} epochs) on {Path(args.csv_path).name if args.csv_path else 'unknown file'}"  # Include filename and epoch info
                    f" | Elapsed: {elapsed_label} | ETA: {eta_label}{format_estimated_finish_time_suffix(eta_seconds)}"  # Add overall elapsed time, current ETA estimate, and finish timestamp
                )  # Compose progress message
                send_telegram_message(TELEGRAM_BOT, msg)  # Send message via shared function
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
        args._last_training_time = ceil_seconds(training_elapsed)  # Store training elapsed as upward-rounded integer seconds
        file_elapsed = time.time() - file_start_time  # Calculate file processing elapsed seconds
        args._last_file_time = ceil_seconds(file_elapsed)  # Store file elapsed as upward-rounded integer seconds for console reporting
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
            derived = csv_obj.parent / config.get("paths", {}).get("data_augmentation_dir", "Data_Augmentation") / config.get("paths", {}).get("data_augmentation_sample_dir", "Samples") / f"{csv_obj.stem}{suffix}.csv"  # Construct path in data augmentation samples directory
            gen_file = str(derived)  # Use derived path string
        except Exception:  # If derivation fails
            gen_file = ""  # Leave blank on failure
    final_runtime["generated_file"] = gen_file  # Store generated file path (may be empty)
    final_runtime["generated_num_samples"] = ""  # Placeholder, generation may fill this later
    final_runtime["total_num_samples_after_augmentation"] = ""  # Placeholder, generation may fill this later
    final_runtime["generated_ratio"] = ""  # Placeholder ratio
    final_runtime["training_time_s"] = getattr(args, "_last_training_time", "")  # Total training elapsed
    final_runtime["generated_time_s"] = ""  # Placeholder, generation may fill this later
    final_runtime["testing_time_s"] = ""  # Placeholder, quality verification may fill this later
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
        total_generated = safe_float(final_runtime.get("generated_num_samples"), 0.0)  # Total generated safely
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
            config_paths = config.get("paths", {})  # Read paths
            data_aug_subdir = config_paths.get("data_augmentation_dir", "Data_Augmentation")  # Read directory
            plot_dir = csv_path_obj.parent / data_aug_subdir  # Create data augmentation directory
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
    Send final training completion messages via telegram and saved-file function.

    :param args: parsed arguments namespace with csv_path and epochs
    :param config: configuration dictionary passed to send_file_saved_and_timing_messages
    :param file_progress_prefix: colored prefix string for telegram message
    :return: None
    """

    send_file_saved_and_timing_messages(args, config)  # Send saved-file, size and timing messages via function
    send_telegram_message(TELEGRAM_BOT, f"{file_progress_prefix} Finished WGAN-GP training on {Path(args.csv_path).name} after {args.epochs} epochs")  # Telegram finish with prefix


def train(args, config: Optional[Dict] = None):
    """
    Train the WGAN-GP model using the provided arguments and configuration.

    :param args: parsed arguments namespace containing training configuration
    :param config: Optional configuration dictionary (will use global CONFIG if not provided)
    :return: Tuple of (Generator, CSVFlowDataset, device) for in-memory generation fallback, or None on failure
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG or get_default_config()  # Use global or default config
    
        device, training_start_time, file_start_time, epoch_milestones, file_progress_prefix = normalize_args_and_setup_hardware(args, config)  # Normalize arguments and setup hardware

        dataset, dataloader = create_dataset_and_dataloader(args, config, device)  # Create dataset and dataloader

        results_csv_file, results_csv_writer, results_cols_cfg, feature_dim, n_classes = init_results_csv_and_feature_dims(args, config, dataset)  # Initialize results CSV and feature dimensions

        G, D, opt_D, opt_G, scaler, fixed_noise, fixed_labels, step, start_epoch, metrics_history = create_models_and_optimizers(args, config, device, feature_dim, n_classes)  # Create models and optimizers

        metrics_history, start_epoch, step = resume_from_checkpoint(args, config, device, G, D, opt_G, opt_D, scaler, metrics_history, start_epoch, step)  # Resume from checkpoint if available
        require_finite_model_state(G, "parameters", args, start_epoch, step, "generator")  # Reject invalid generator checkpoint or initialization parameters
        require_finite_model_state(D, "parameters", args, start_epoch, step, "discriminator")  # Reject invalid discriminator checkpoint or initialization parameters
        require_finite_optimizer_state(opt_G, "generator_optimizer", args, start_epoch, step)  # Reject invalid generator optimizer state before training
        require_finite_optimizer_state(opt_D, "discriminator_optimizer", args, start_epoch, step)  # Reject invalid discriminator optimizer state before training
        telegram_enabled, next_notify, progress_pct = init_telegram_progress(args, config, start_epoch)  # Initialize telegram progress tracking

        training_start_time = time.time()  # Start persisted training duration immediately before the epoch loop
        for epoch in range(start_epoch, args.epochs):  # Loop over epochs starting from resume point
            epoch_start_time = time.time()  # Record epoch start timestamp
            pbar, total_steps = create_epoch_progress_bar(dataloader, args, epoch)  # Create progress bar for epoch
            
            step = run_batch_training_loop(G, D, opt_G, opt_D, scaler, device, args, config, n_classes, pbar, step, metrics_history, total_steps, epoch, training_start_time, file_progress_prefix)  # Execute batch training loop

            write_epoch_csv_row(args, config, device, dataset, epoch, epoch_start_time, epoch_milestones, results_csv_writer, results_csv_file, results_cols_cfg, metrics_history, opt_G, opt_D)  # Write epoch CSV row
            save_training_checkpoint(args, config, device, G, D, opt_G, opt_D, scaler, dataset, epoch, metrics_history)  # Save checkpoint if due
            next_notify = send_epoch_telegram_notifications(args, telegram_enabled, epoch, next_notify, progress_pct, training_start_time, file_progress_prefix)  # Send epoch telegram notifications with file progress prefix and timing

        write_final_timing_and_csv_row(args, config, device, dataset, training_start_time, file_start_time, results_csv_writer, results_csv_file, results_cols_cfg, metrics_history, opt_G, opt_D)  # Compute final training timing state
        args._last_metrics_history = metrics_history  # Store metrics history for in-memory generation result rows
        generate_training_plots(args, config, metrics_history, file_progress_prefix)  # Generate training plots
        send_final_telegram_messages(args, config, file_progress_prefix)  # Send final telegram messages
        return (G, dataset, device)  # Return trained generator, dataset, and device for in-memory generation fallback
    except Exception as e:
        print(str(e))
        release_cuda_memory()  # Release unused CUDA cache before propagating training failure
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def apply_zebra_style(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """
    Apply zebra-striping style to a DataFrame using pandas Styler.

    :param df: Input DataFrame to style
    :return: pandas Styler with zebra row background colors applied
    """
    
    try:
        styled = df.style.apply(row_style_for_zebra, axis=1)  # Apply zebra function row-wise using top-level function
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
        
        if is_visualizable and len(df) <= 100:  # Generate PNG only for visualizable DataFrames within the safe row limit
            try:  # Guard PNG rendering to prevent aborting the CSV pipeline on image export failures
                png_path = csv_p.with_suffix(".png")  # Replace CSV extension with PNG
                generate_table_image_from_dataframe(df, png_path)  # Generate PNG from in-memory DataFrame
            except Exception as _png_e:  # If PNG generation fails, warn but preserve the already-written CSV
                print(f"{BackgroundColors.YELLOW}Warning: PNG generation failed for {csv_p.name}: {_png_e}{Style.RESET_ALL}")  # Warn about PNG failure without aborting
    except Exception:
        raise  # Propagate CSV write exceptions to caller (do not swallow)


def generate_tsne_3d_separability_plot(
    original_features: np.ndarray,
    generated_features: np.ndarray,
    out_dir: Union[str, Path],
    dataset_name: Optional[str] = None,
    config: Optional[Dict] = None,
    original_labels: Optional[np.ndarray] = None,  # Retain source class labels for detailed grouping.
    generated_labels: Optional[np.ndarray] = None,  # Retain synthetic class labels for detailed grouping.
) -> None:
    """
    Generate and save a 3D t-SNE scatter plot comparing original and synthetic samples.

    :param original_features: Numpy array of original dataset feature values (rows x features).
    :param generated_features: Numpy array of generated synthetic feature values (rows x features).
    :param out_dir: Data augmentation directory under which the separability directory is created.
    :param dataset_name: Optional dataset identifier used as the filename prefix.
    :param config: Optional configuration dictionary reserved for future extension.
    :param original_labels: Optional source class labels aligned with original_features.
    :param generated_labels: Optional synthetic class labels aligned with generated_features.
    :return: None.
    """

    try:
        max_samples = 5000  # Maximum samples per group to keep t-SNE tractable
        total_combined = len(original_features) + len(generated_features)  # Compute combined dataset row count
        original_label_values = (
            np.asarray(original_labels, dtype=str) if original_labels is not None else None
        )  # Convert source labels for indexed grouping.
        generated_label_values = (
            np.asarray(generated_labels, dtype=str) if generated_labels is not None else None
        )  # Convert synthetic labels for indexed grouping.

        if original_label_values is not None and (
            len(original_label_values) != len(original_features)
        ):  # Require aligned source labels for valid coordinates.
            raise ValueError(
                "Original t-SNE label count does not match feature count"
            )  # Reject misaligned source labels.
        if generated_label_values is not None and (
            len(generated_label_values) != len(generated_features)
        ):  # Require aligned synthetic labels for valid coordinates.
            raise ValueError(
                "Generated t-SNE label count does not match feature count"
            )  # Reject misaligned synthetic labels.

        if total_combined > 10000:  # Subsample when combined size exceeds tractability threshold
            rng = np.random.default_rng(42)  # Seed RNG for reproducible subsampling

            n_orig_sample = min(max_samples, len(original_features))  # Clamp original sample count to max_samples
            orig_idx = rng.choice(len(original_features), size=n_orig_sample, replace=False)  # Draw random indices from original pool
            orig_subset = original_features[orig_idx]  # Extract sampled original feature rows
            original_labels_subset = (
                original_label_values[orig_idx] if original_label_values is not None else None
            )  # Keep source labels aligned with sampled features.

            n_gen_sample = min(max_samples, len(generated_features))  # Clamp generated sample count to max_samples
            gen_idx = rng.choice(len(generated_features), size=n_gen_sample, replace=False)  # Draw random indices from generated pool
            gen_subset = generated_features[gen_idx]  # Extract sampled generated feature rows
            generated_labels_subset = (
                generated_label_values[gen_idx] if generated_label_values is not None else None
            )  # Keep synthetic labels aligned with sampled features.

        else:  # Use full arrays when combined size is within threshold
            orig_subset = original_features  # Use all original feature rows
            gen_subset = generated_features  # Use all generated feature rows
            original_labels_subset = original_label_values  # Retain all source labels with full feature set.
            generated_labels_subset = generated_label_values  # Retain all synthetic labels with full feature set.

        n_orig = len(orig_subset)  # Count original samples used for t-SNE index boundary
        combined = np.vstack([orig_subset, gen_subset]).astype(np.float32)  # Vertically stack both groups for joint t-SNE fit
        combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)  # Replace non-finite values to prevent t-SNE numerical failure
        # Fit the scaling only on original flows; synthetic records cannot set the
        # reference axes. Clip extreme transformed values so rate outliers do not
        # overwhelm all other packet-flow attributes in the t-SNE embedding.
        reference_scaler = RobustScaler(quantile_range=(10.0, 90.0))
        reference_scaler.fit(combined[:n_orig].astype(np.float64))
        combined = np.clip(reference_scaler.transform(combined.astype(np.float64)), -10.0, 10.0).astype(np.float32)
        if not np.isfinite(combined).all():
            raise ValueError("Nonfinite robust-scaled features for t-SNE")

        tsne = TSNE(n_components=3, random_state=42, perplexity=30)  # Instantiate t-SNE for 3D reduction with fixed seed and default perplexity
        embedded = tsne.fit_transform(combined)  # Fit and reduce combined array to 3-dimensional embedding

        orig_emb = embedded[:n_orig]  # Slice embedding to isolate original sample coordinates
        gen_emb = embedded[n_orig:]  # Slice embedding to isolate generated sample coordinates

        fig = plt.figure(figsize=(12, 8))  # Create figure with generous size for 3D scatter legibility
        ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))  # Cast to Axes3D so pylance resolves 3D scatter signature and set_zlabel method

        ax.scatter(
            orig_emb[:, 0], orig_emb[:, 1], zs=cast(Any, orig_emb[:, 2]),
            c="#1f77b4", alpha=0.5, s=8, label="Original Samples",
        )  # Plot original sample embeddings in blue; cast(Any) bypasses the incorrect int stub for zs
        ax.scatter(
            gen_emb[:, 0], gen_emb[:, 1], zs=cast(Any, gen_emb[:, 2]),
            c="#ff7f0e", alpha=0.5, s=8, label="Generated Samples",
        )  # Plot generated sample embeddings in orange; cast(Any) bypasses the incorrect int stub for zs

        ax.set_xlabel("TSNE Dimension 1")  # Label x-axis as first t-SNE dimension
        ax.set_ylabel("TSNE Dimension 2")  # Label y-axis as second t-SNE dimension
        ax.set_zlabel("TSNE Dimension 3")  # Label z-axis as third t-SNE dimension
        ax.set_title("WGAN-GP Data Distribution (Original vs Generated) - t-SNE 3D")  # Set descriptive plot title
        ax.legend(loc="upper left")  # Render legend in upper-left corner of 3D axes

        plots_dir = Path(out_dir) / "Plots" / "data separability"  # Build separability plot directory
        os.makedirs(str(plots_dir), exist_ok=True)  # Create separability plot directory

        prefix = f"{dataset_name}_" if dataset_name else ""  # Build filename prefix from dataset name when available
        plot_path = plots_dir / f"{prefix}tsne_3d_original_vs_generated.png"  # Construct deterministic plot file path

        ensure_figure_min_4k_and_save(fig=fig, path=str(plot_path), dpi=300)  # Save plot enforcing minimum 4K pixel dimensions
        print(f"{BackgroundColors.GREEN}Saved t-SNE 3D separability plot to {BackgroundColors.CYAN}{plot_path}{Style.RESET_ALL}")  # Confirm plot saved successfully

        if original_labels_subset is not None and (
            generated_labels_subset is not None
        ):  # Create class-aware plot when both label arrays are available.
            detailed_fig = plt.figure(figsize=(12, 8))  # Create separate figure for provenance and class groups.
            detailed_ax = cast(
                Axes3D, detailed_fig.add_subplot(111, projection="3d")
            )  # Create typed 3D axis for detailed scatter groups.
            original_classes = np.unique(original_labels_subset)  # Identify source classes present in plotted rows.
            generated_classes = np.unique(
                generated_labels_subset
            )  # Identify synthetic classes present in plotted rows.
            detailed_group_count = (
                len(original_classes) + len(generated_classes)
            )  # Count provenance and class combinations for colors.
            detailed_colors = plt.get_cmap(
                "hsv", detailed_group_count
            )  # Allocate distinct hues for every detailed group.

            for color_index, class_label in enumerate(
                original_classes
            ):  # Plot each source class as an independent group.
                class_mask = original_labels_subset == class_label  # Select embedded source rows for current class.
                detailed_ax.scatter(
                    orig_emb[class_mask, 0], orig_emb[class_mask, 1], zs=cast(Any, orig_emb[class_mask, 2]),
                    c=[detailed_colors(color_index)], alpha=0.5, s=8, label=class_label,
                )  # Plot source class coordinates with dedicated color and label.

            for color_index, class_label in enumerate(
                generated_classes, start=len(original_classes)
            ):  # Plot each synthetic class as an independent group.
                class_mask = generated_labels_subset == class_label  # Select embedded synthetic rows for current class.
                detailed_ax.scatter(
                    gen_emb[class_mask, 0], gen_emb[class_mask, 1], zs=cast(Any, gen_emb[class_mask, 2]),
                    c=[detailed_colors(color_index)], alpha=0.5, s=8, label=f"{class_label} augmented",
                )  # Plot synthetic class coordinates with dedicated color and provenance suffix.

            detailed_ax.set_xlabel("TSNE Dimension 1")  # Label x-axis as first t-SNE dimension.
            detailed_ax.set_ylabel("TSNE Dimension 2")  # Label y-axis as second t-SNE dimension.
            detailed_ax.set_zlabel("TSNE Dimension 3")  # Label z-axis as third t-SNE dimension.
            detailed_ax.set_title(
                "WGAN-GP Data Distribution (Original vs Generated by Class) - t-SNE 3D"
            )  # Set descriptive detailed plot title.
            detailed_ax.legend(loc="upper left", fontsize="small", ncol=2)  # Render readable detailed group legend.

            detailed_plot_path = (
                plots_dir / f"{prefix}tsne_3d_original_vs_generated_detailed.png"
            )  # Construct detailed plot file path.
            ensure_figure_min_4k_and_save(
                fig=detailed_fig, path=str(detailed_plot_path), dpi=300
            )  # Save detailed plot enforcing minimum 4K pixel dimensions.
            print(
                f"{BackgroundColors.GREEN}Saved detailed t-SNE 3D separability plot to "
                f"{BackgroundColors.CYAN}{detailed_plot_path}{Style.RESET_ALL}"
            )  # Confirm detailed plot saved successfully.

    except Exception as e:
        print(f"{BackgroundColors.YELLOW}[WARNING] t-SNE 3D separability plot generation failed: {e}{Style.RESET_ALL}")  # Warn on failure and continue without aborting
        try:
            send_exception_via_telegram(type(e), e, e.__traceback__)  # Forward t-SNE exception details via Telegram
        except Exception:
            pass  # Ignore notification failures to avoid breaking downstream flow


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

        if getattr(args, "label", None) is not None:  # Respect an explicitly requested class during expected-count planning.
            requested_label = int(args.label)  # Normalize selected class identity.
            if requested_label not in class_dist:  # Reject a missing selected class.
                return None  # Avoid accepting an unrelated output as complete.
            return calculate_generated_count_from_ratio(int(class_dist[requested_label]), percent)  # Use only the selected preprocessed class.
        return sum(calculate_generated_count_from_ratio(int(count), percent) for count in class_dist.values())  # Match the production per-class allocation exactly.

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


def resolve_expected_output_row_count(args, config: Dict) -> Optional[int]:
    """
    Resolve the expected persisted generated dataset row count.

    :param args: Parsed arguments namespace with source and generation settings.
    :param config: Configuration dictionary with generation settings.
    :return: Expected generated output row count, or None if unavailable.
    """

    expected_generated = resolve_expected_sample_count(args, config)  # Resolve the expected synthetic row count

    if expected_generated is None:  # Propagate unavailable generated count
        return None  # Return unavailable output count

    return int(expected_generated)  # Return generated artifact row count


def evaluate_existing_augmentation_file(out_path: Path, file_prefix: str, existing_count: int, expected_n: int, args) -> bool:
    """
    Evaluate whether to regenerate or skip based on existing file count versus expected count.

    :param out_path: Path to the existing augmentation output file.
    :param file_prefix: Telegram prefix string for progress display.
    :param existing_count: Number of rows in the existing output file.
    :param expected_n: Expected number of samples based on configuration.
    :param args: Parsed arguments namespace with force_regenerate_samples flag.
    :return: True if generation should proceed, False if it should be skipped.
    """

    if existing_count >= expected_n and not getattr(args, "force_regenerate_samples", False):  # Preserve sufficient-output skip unless regeneration is forced
        send_telegram_message(
            TELEGRAM_BOT,
            f"{file_prefix} Skipping Generation: {out_path.name} already exists with {existing_count} output rows (required {expected_n}).",
        )  # Notify skip via Telegram
        return False  # Skip generation
    if getattr(args, "force_regenerate_samples", False):  # Bypass every existing-output skip when regeneration is explicitly forced
        source_name = Path(getattr(args, "csv_path", out_path.name)).name  # Identify selected input file for factual logging
        forced_message = f"{file_prefix} Force Regeneration Requested: Regenerating existing result for {source_name} ({existing_count} output rows) because regeneration was explicitly forced."  # Build concise forced-regeneration context
        print(f"{BackgroundColors.YELLOW}{forced_message}{Style.RESET_ALL}")  # Log forced regeneration to terminal
        safe_log("info", forced_message)  # Persist forced-regeneration context through existing logging
        send_telegram_message(TELEGRAM_BOT, forced_message)  # Notify forced regeneration through existing Telegram flow
        return True  # Preserve existing output until replacement generation is ready
    send_telegram_message(TELEGRAM_BOT, f"{file_prefix} Existing {out_path.name} has {existing_count} output rows but requires {expected_n}; removing and regenerating.")  # Notify mismatch via Telegram
    return True  # Regenerate without deleting the prior output before validation succeeds.


def verify_data_augmentation_file(args, config: Optional[Dict] = None) -> bool:
    """
    Verify whether the data-augmentation output file exists and matches the
    configured number of samples. Returns True when generation should proceed
    (file missing, mismatched, or forced), or False when generation can be
    skipped because an existing file already matches the requested size.

    Behavior:
      - If output file does not exist: return True (proceed)
      - If output file exists and equals expected sample count: return False
        unless `force_regenerate_samples` is True (in that case return True)
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

        existing_manifest_path = out_path.with_suffix(".provenance.json")  # Resolve the exact source and preprocessing identity beside the existing CSV.
        existing_quality_path = out_path.with_suffix(".quality.json")  # Resolve the quality diagnostics produced by the matching generation run.
        try:  # Require source, preprocessing and quality provenance before trusting an existing output.
            existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))  # Read the persisted output's training identity.
            existing_quality = json.loads(existing_quality_path.read_text(encoding="utf-8"))  # Read class-specific output validation evidence.
            verified_output = existing_manifest.get("sha256") == build_wgangp_provenance(args)["sha256"] and existing_quality.get("schema_version") == 3 and existing_quality.get("accepted") is True and existing_quality.get("quality_policy_sha256") == build_quality_policy_fingerprint(config)  # Require a passed class-quality acceptance report.
        except (OSError, ValueError, TypeError, AttributeError):  # Treat absent or malformed historical artifacts as unverified.
            verified_output = False  # Never skip regeneration based on count alone.
        if not verified_output:  # Reject incompatible historical artifacts while preserving their bytes until a successful replacement.
            print(f"[QUALITY] Existing augmented CSV is unverified for current preprocessing; regenerating {out_path.name}")  # Explain the safe regeneration decision.
            return True  # Regenerate from a matching trained generator rather than accepting old output.

        expected_n = resolve_expected_output_row_count(args, config)  # Resolve expected final output row count from configuration

        if expected_n is None:  # If expected size cannot be determined
            send_telegram_message(
                TELEGRAM_BOT,
                f"{file_prefix} Unable to verify existing augmented file {out_path.name}: expected output rows undeterminable — regenerating.",
            )
            return True  # Preserve existing bytes until validated replacement is ready.

        try:
            existing_count = len(pd.read_csv(out_path, low_memory=False))  # Count rows safely
        except Exception:
            existing_count = None  # Fallback if unreadable

        if existing_count is None:  # If file unreadable
            send_telegram_message(
                TELEGRAM_BOT,
                f"{file_prefix} Existing augmented file {out_path.name} unreadable — regenerating.",
            )
            return True  # Preserve prior output until a replacement passes quality validation.

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
    args._generation_epoch = int(ckpt.get("epoch", 0))  # Store checkpoint epoch for generation diagnostics
    if not getattr(args, "csv_path", None) and isinstance(args_ck, dict) and args_ck.get("csv_path"):  # Recover source CSV path from saved training arguments
        args.csv_path = args_ck.get("csv_path")  # Store recovered source CSV path on live args
    verify_wgangp_provenance(ckpt.get("provenance"), args, str(args.checkpoint))  # Reject mismatched checkpoint source and configuration before generation.
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

    if isinstance(scaler, ClassConditionalFlowScaler) and scaler.model_feature_dim is not None:
        dimension = int(scaler.model_feature_dim)
        if args.feature_dim is not None and int(args.feature_dim) != dimension:
            raise ValueError("Configured feature_dim conflicts with the V5 checkpoint's mixed feature width")
        return dimension, len(label_encoder.classes_)
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
        n_resblocks=args.n_resblocks,  # Number of residual blocks in generator
        categorical_groups=getattr(args, "_mixed_categorical_groups", []),  # Preserve structural mixed-type real/fake one-hot locations
        categorical_class_masks=getattr(args, "_mixed_categorical_class_masks", []),
        categorical_temperature=float(config.get("generation", {}).get("categorical_temperature", 1.25)),
        leaky_relu_alpha=g_leaky_relu_alpha,  # Use config value
    ).to(
        device
    )  # Initialize generator model
    if isinstance(G, torch.nn.DataParallel):  # If generator wrapped by DataParallel
        G.module.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)  # Load into underlying module
    else:  # Not DataParallel
        G.load_state_dict(ckpt["state_dict"] if "state_dict" in ckpt else ckpt)  # Load generator weights from checkpoint
    G.eval()  # Set generator to evaluation mode
    require_finite_model_state(G, "parameters", args, int(ckpt.get("epoch", 0)), 0, "generation")  # Reject invalid checkpoint parameters before generation
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
    if args.n_samples <= 1.0:  # Percentage mode: generate percentage of original dataset size (1.0 == 100%)
        if class_distribution is None:  # If class distribution not available
            raise RuntimeError(
                "Percentage-based generation requires class_distribution in checkpoint or --csv_path to calculate it."
            )  # Raise error when class distribution missing

        total_preprocessed = int(sum(class_distribution.values()))  # Define the actual usable source population after GAN preprocessing.
        args._last_original_preprocessed_num_samples = total_preprocessed  # Persist the correct source denominator for generated_ratio and output metadata.
        if args.label is not None:  # Keep percentage generation class-specific when one class is selected.
            if args.label not in class_distribution:  # Verify the requested encoded class exists.
                raise ValueError(f"Label {args.label} not found in training data class distribution")  # Refuse missing conditioning labels.
            selected_counts = {args.label: class_distribution[args.label]}  # Restrict the source denominator to the chosen class.
        else:  # Generate the requested ratio independently for every available class.
            selected_counts = class_distribution  # Preserve the generator checkpoint's class distribution.
        n_per_class = {label: calculate_generated_count_from_ratio(int(count), args.n_samples) for label, count in selected_counts.items()}  # Apply the requested ratio to each preprocessed class independently.
        target_n = sum(n_per_class.values())  # Resolve exact total after deterministic per-class rounding.
        print(f"{BackgroundColors.CYAN}Generating {args.n_samples*100:.1f}% per original preprocessed class ({target_n} of {sum(int(count) for count in selected_counts.values())} usable source rows){Style.RESET_ALL}")  # Report per-class rather than raw-file ratio semantics.

        labels = []  # List to build label array
        for label, count in n_per_class.items():  # For each class
            labels.extend([label] * count)  # Repeat label by count
        labels = np.array(labels, dtype=np.int64)  # Convert to numpy int64 array
        n = len(labels)  # Total number of samples equals target_n

        print(f"{BackgroundColors.GREEN}Total samples to generate: {BackgroundColors.CYAN}{n}{Style.RESET_ALL}")  # Print total sample count
        for label, count in n_per_class.items():  # Print per-class breakdown
            class_name = label_encoder.inverse_transform([label])[0]  # Get human-readable class name
            print(f"{BackgroundColors.GREEN}  - Class '{class_name}': {BackgroundColors.CYAN}{count}{BackgroundColors.GREEN} samples{Style.RESET_ALL}")  # Print per-class count
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
    """Generate a fixed batch population in inference mode, restoring the previous training mode."""
    batch_size = int(args.gen_batch_size)
    if batch_size <= 0:
        raise ValueError("Generation batch size must be positive")
    base_model = getattr(G, "module", G)
    feature_dim = int(base_model.feature_dim)
    all_fake = np.empty((int(n), feature_dim), dtype=np.float32)
    started = time.time()
    previous_mode = G.training
    try:
        G.eval()  # Essential for stable BatchNorm inference and final one-row batches.
        with torch.no_grad():
            for offset in range(0, n, batch_size):
                count = min(batch_size, n-offset)
                z = torch.randn(count, args.latent_dim, device=device)
                y = torch.from_numpy(labels[offset:offset+count]).to(device, dtype=torch.long)
                fake_tensor = G(z, y)
                require_finite_tensor(fake_tensor, "generated_samples", args, int(getattr(args, "_generation_epoch", getattr(args, "_last_completed_epoch", 0))), offset // batch_size + 1, "generation")
                all_fake[offset:offset+count] = fake_tensor.cpu().numpy()
    finally:
        G.train(previous_mode)
    return all_fake, labels, started


def build_augmented_output_dataframe(args: Any, df_generated: pd.DataFrame) -> tuple:
    """
    Build the persisted generated dataset and retain source row metadata.

    :param args: Parsed arguments namespace containing csv_path.
    :param df_generated: DataFrame containing newly generated synthetic rows.
    :return: Tuple of generated output DataFrame and original source row count.
    """

    original_count = getattr(args, "_last_original_preprocessed_num_samples", None)  # Prefer the exact usable original population supporting per-class ratios.
    if original_count is None and getattr(args, "csv_path", None):  # Retain the legacy raw-count fallback for absolute-count generation.
        original_count = len(pd.read_csv(args.csv_path, low_memory=False))  # Count raw source rows only when preprocessed counts are unavailable.
    original_count = int(original_count or 0)  # Normalize sample count for result metadata.

    return df_generated, original_count  # Return generated rows and source row count


def generate_generation_quality_outputs(args: Any, config: Dict, df_generated: pd.DataFrame, feature_cols: List[str]) -> None:
    """
    Generate post-generation quality visualization outputs.

    :param args: Parsed arguments namespace containing csv_path and out_file.
    :param config: Configuration dictionary with path settings.
    :param df_generated: DataFrame containing newly generated synthetic rows.
    :param feature_cols: Feature column names used by the generator.
    :return: None.
    """

    if not getattr(args, "csv_path", None):  # Skip quality output when no source CSV path is available
        return  # Return without creating file-based quality plots

    try:  # Bound source and synthetic plot inputs independently of the full CSV population.
        sample_limit = max(32, int(config.get("quality", {}).get("plot_samples_per_class", 512)))  # Configure a representative bounded plotting sample per class.
        label = args.label_col  # Resolve the class label used by this generator.
        rng = np.random.default_rng(int(getattr(args, "seed", 42)) + 1)  # Keep plot sampling reproducible without reusing quality reservoirs.
        sample_values = {}  # Accumulate sampled original features by class.
        sample_priorities = {}  # Retain uniform sampling priorities across source chunks.
        columns = list(feature_cols) + [label]  # Read only original labels and model features.
        for chunk in pd.read_csv(args.csv_path, usecols=lambda value: value.strip() in columns, chunksize=8192, low_memory=False):  # Stream original source data without loading the entire CSV.
            chunk.columns = chunk.columns.str.strip()  # Normalize column names like the training preprocessor.
            if label not in chunk.columns:  # Require aligned original attack labels.
                raise ValueError(f"t-SNE source is missing label column: {label}")  # Stop mislabelled visualizations.
            for class_name, group in chunk.groupby(label, sort=False):  # Uniformly sample each real class over the complete source file.
                key = str(class_name)  # Preserve the exact displayed class identity.
                values = group[feature_cols].to_numpy(dtype=np.float32, copy=True)  # Materialize only one bounded source chunk.
                ranks = rng.random(len(values))  # Assign uniformly distributed sampling priorities.
                if key in sample_values:  # Combine with previously selected real examples.
                    values = np.concatenate([sample_values[key], values])  # Keep at most one chunk plus the reservoir in memory.
                    ranks = np.concatenate([sample_priorities[key], ranks])  # Preserve aligned sampling keys.
                selected = np.argpartition(ranks, min(sample_limit, len(ranks)) - 1)[:sample_limit]  # Select a uniform bounded reservoir.
                sample_values[key] = values[selected]  # Retain one bounded original sample per class.
                sample_priorities[key] = ranks[selected]  # Retain matching reservoir priorities.
        original_matrices = []  # Collect bounded source class matrices for plotting.
        original_classes = []  # Collect aligned original class identities.
        for class_name, values in sample_values.items():  # Assemble sampled original data with class labels.
            finite = np.isfinite(values).all(axis=1)  # Match generation preprocessing's finite-row requirement.
            original_matrices.append(values[finite])  # Exclude invalid source values without changing generated CSVs.
            original_classes.extend([class_name] * int(finite.sum()))  # Keep labels aligned with the selected original rows.
        X_original_tsne = np.concatenate(original_matrices).astype(np.float32)  # Materialize only bounded original plot rows.
        generated_matrices = []  # Collect bounded synthetic samples independently by class.
        generated_classes = []  # Track aligned synthetic attack labels.
        for class_name, group in df_generated.groupby(label, sort=False):  # Keep minority attacks represented in t-SNE.
            sample = group.sample(n=min(sample_limit, len(group)), random_state=int(getattr(args, "seed", 42)))  # Sample generated rows reproducibly by class.
            generated_matrices.append(sample[feature_cols].to_numpy(dtype=np.float32, copy=True))  # Retain bounded generated plot features.
            generated_classes.extend([str(class_name)] * len(sample))  # Preserve attack labels aligned with synthetic rows.
        X_generated_tsne = np.concatenate(generated_matrices).astype(np.float32)  # Materialize bounded synthetic plot rows.
        config_paths = config.get("paths", {})  # Resolve existing output-directory configuration.
        data_aug_subdir = config_paths.get("data_augmentation_dir", "Data_Augmentation")  # Preserve existing plot destination structure.
        tsne_out_dir = Path(args.csv_path).parent / data_aug_subdir  # Store plots beside the dataset's augmentation artifacts.
        generate_tsne_3d_separability_plot(X_original_tsne, X_generated_tsne, tsne_out_dir, dataset_name=Path(args.csv_path).stem, config=config, original_labels=np.asarray(original_classes), generated_labels=np.asarray(generated_classes))  # Render original-versus-synthetic 3D plots using bounded aligned class samples.

    except Exception as _tsne_e:  # Continue when quality visualization fails
        print(f"{BackgroundColors.YELLOW}[WARNING] t-SNE separability plot failed: {_tsne_e}{Style.RESET_ALL}")  # Warn about t-SNE failure without aborting generation


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
    if not isinstance(scaler, ClassConditionalFlowScaler):  # Require a checkpoint created for the validated mixed-type representation.
        raise ValueError("Generator checkpoint uses an incompatible legacy preprocessing representation; retrain with --from_scratch")  # Never use a raw StandardScaler to decode quantile-space generator output.
    X_orig = scaler.inverse_transform(X_fake, Y_fake)  # Rebuild full flow schema from independent mixed-type model features and verified identities.
    df_generated = pd.DataFrame(X_orig, columns=pd.Index(feature_cols))  # Create DataFrame with original feature names
    df_generated[args.label_col] = label_encoder.inverse_transform(Y_fake)  # Map integer labels back to original strings
    assess_generated_flow_quality(args, config, df_generated, scaler=scaler, label_encoder=label_encoder)  # Enforce class-specific quality acceptance before saving.
    output_df, original_count = build_augmented_output_dataframe(args, df_generated)  # Build output

    if config.get("hardware_tracking", False):  # If enabled in config
        try:  # Guard hardware population to avoid breaking generation
            output_df = populate_hardware_column(output_df, column_name="hardware", device_used=device)  # Populate hardware column on persisted output
        except Exception:
            pass  # Ignore hardware population errors and continue

    args._last_original_num_samples = int(original_count)  # Store source row count for result persistence
    args._last_generated_num_samples = int(len(df_generated))  # Store synthetic row count for result persistence
    args._last_total_num_samples_after_augmentation = int(original_count + len(df_generated))  # Store total

    safe_log("info", f"Saving {len(output_df)} generated rows to {args.out_file}")  # Log saving
    generate_csv_and_image(output_df, args.out_file, is_visualizable=True)  # Save CSV
    with Path(args.out_file).with_suffix(".provenance.json").open("w", encoding="utf-8") as provenance_file:  # Record which training source and hyperparameters produced the output.
        json.dump(build_wgangp_provenance(args), provenance_file, indent=2)  # Write the canonical generator provenance alongside the CSV.
    safe_log("info", f"Successfully saved {len(output_df)} generated rows to {args.out_file}")  # Log saved rows
    print(  # Print saved rows
        f"{file_progress_prefix} {BackgroundColors.GREEN}Saved {BackgroundColors.CYAN}"  # Format prefix
        f"{len(output_df)}{BackgroundColors.GREEN} generated rows to "  # Format count
        f"{BackgroundColors.CYAN}{args.out_file}{Style.RESET_ALL}"  # Format path
    )  # Finish output

    return df_generated  # Return generated-only DataFrame for quality visualization


def record_sample_generation_timing(args, sample_generation_start_time: float) -> None:
    """
    Compute and record the elapsed time for sample generation.

    :param args: Parsed arguments namespace to store timing on.
    :param sample_generation_start_time: Timestamp when sample generation started.
    :return: None.
    """

    try:  # Safely compute and print sample generation elapsed time
        sample_generation_elapsed = time.time() - sample_generation_start_time  # Calculate sample generation elapsed seconds
        args._last_sample_generation_time = ceil_seconds(sample_generation_elapsed)  # Store generation elapsed as upward-rounded integer seconds
        print(f"{BackgroundColors.GREEN}Sample generation elapsed: {BackgroundColors.CYAN}{sample_generation_elapsed:.2f}s{Style.RESET_ALL}")  # Print sample generation elapsed
    except Exception as _sg:  # If timing calculation fails
        print(f"{BackgroundColors.YELLOW}Warning: failed to measure sample generation time: {_sg}{Style.RESET_ALL}")  # Warn but continue
        args._last_sample_generation_time = ""  # Ensure attribute exists even on failure


def record_testing_timing(args, testing_start_time: float) -> None:
    """
    Record post-generation verification and quality-output duration.

    :param args: Parsed arguments namespace to store timing on.
    :param testing_start_time: Timestamp when post-generation verification started.
    :return: None.
    """

    try:  # Safely compute and print verification elapsed time
        testing_elapsed = time.time() - testing_start_time  # Calculate verification elapsed seconds
        args._last_testing_time = ceil_seconds(testing_elapsed)  # Store verification elapsed as upward-rounded integer seconds
        print(f"{BackgroundColors.GREEN}Post-generation verification elapsed: {BackgroundColors.CYAN}{testing_elapsed:.2f}s{Style.RESET_ALL}")  # Print verification elapsed
    except Exception as _tg:  # If timing calculation fails
        print(f"{BackgroundColors.YELLOW}Warning: failed to measure post-generation verification time: {_tg}{Style.RESET_ALL}")  # Warn but continue
        args._last_testing_time = ""  # Ensure attribute exists even on failure


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
        data_aug_subdir = config.get("paths", {}).get("data_augmentation_dir", "Data_Augmentation")  # Read configured data augmentation base directory name
        data_aug_dir = csv_path_obj.parent / data_aug_subdir  # Construct data augmentation directory under dataset folder
        os.makedirs(data_aug_dir, exist_ok=True)  # Ensure data augmentation directory exists before writing
        results_csv_path = data_aug_dir / "data_augmentation_results.csv"  # Use results CSV inside Data_Augmentation dir
    else:  # Try to recover original csv_path from checkpoint args saved in checkpoint
        ck_csv_path = None  # Reset checkpoint csv path
        try:  # Attempt recovery from checkpoint args
            if args_ck and args_ck.get("csv_path"):  # Use saved args from checkpoint (args_ck defined earlier)
                ck_csv_value = args_ck.get("csv_path")  # Get csv_path value from checkpoint args
                if ck_csv_value:  # If value is non-empty
                    ck_csv_path = Path(ck_csv_value)  # Path object for saved csv_path from checkpoint
                    ck_data_aug_subdir = config.get("paths", {}).get("data_augmentation_dir", "Data_Augmentation")  # Read data augmentation base directory name from config
                    ck_data_aug_dir = ck_csv_path.parent / ck_data_aug_subdir  # Construct data augmentation directory for checkpoint csv_path
                    os.makedirs(ck_data_aug_dir, exist_ok=True)  # Ensure Data_Augmentation directory exists for checkpoint recover
                    results_csv_path = ck_data_aug_dir / "data_augmentation_results.csv"  # Use results CSV inside Data_Augmentation dir for checkpoint
                else:  # Empty csv_path value in checkpoint
                    ck_csv_path = None  # Keep as None when value is empty
        except Exception:  # Recovery failed
            results_csv_path = None  # Leave as None if recovery fails
    return results_csv_path, ck_csv_path  # Return resolved results CSV path and checkpoint csv path


def collect_generation_file_metadata(args, args_ck: Dict, ckpt: Dict, n: int, ck_csv_path) -> Dict:
    """
    Collect file metadata for generation results including filenames, counts, and timing.

    :param args: Parsed arguments namespace with csv_path, out_file, and timing attributes.
    :param args_ck: Saved arguments dictionary from the loaded state.
    :param ckpt: Loaded checkpoint dictionary containing metrics_history.
    :param n: Total number of generated samples.
    :param ck_csv_path: Checkpoint-recovered csv Path object (may be None).
    :return: Dictionary with generation metadata values.
    """

    metadata: Dict[str, Any] = {}  # Initialize metadata dictionary with mixed value types
    saved_args = args_ck if isinstance(args_ck, dict) else {}  # Normalize saved arguments
    source_csv_path = Path(args.csv_path) if getattr(args, "csv_path", None) else ck_csv_path  # Resolve source CSV path from live args or saved path
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
    metadata["generated_file_name"] = str(Path(args.out_file)) if getattr(args, "out_file", None) else ""  # Generated file path
    # Use the exact usable population saved during generation. The source CSV's
    # raw row count may contain invalid/filtered rows and MUST NOT overwrite it.
    metadata["original_num"] = getattr(args, "_last_original_num_samples", None)
    if metadata["original_num"] in (None, ""):
        metadata["original_num"] = getattr(args, "_last_original_preprocessed_num_samples", None)
    if metadata["original_num"] in (None, ""):
        distribution = ckpt.get("class_distribution") if isinstance(ckpt, dict) else None
        if isinstance(distribution, dict) and distribution:
            metadata["original_num"] = sum(int(count) for count in distribution.values())
    if metadata["original_num"] in (None, ""):
        raise ValueError("Usable source sample count is unavailable; refusing to record an inaccurate generation result")
    metadata["original_num"] = int(metadata["original_num"])
    metadata["generated_num"] = int(n) if n is not None else ""  # Store newly generated synthetic rows
    # The dataset saved on disk contains generated rows only; this results field
    # describes the conceptual total of valid original plus generated samples.
    metadata["total_after_augmentation"] = int(metadata["original_num"]) + int(metadata["generated_num"])
    metadata["generated_ratio"] = ""  # Default generated ratio
    try:  # Attempt to compute generated ratio
        if metadata["original_num"] and metadata["original_num"] > 0:  # If original count available
            metadata["generated_ratio"] = float(metadata["generated_num"]) / float(metadata["original_num"])  # Compute synthetic-to-source ratio
    except Exception:  # Computation failed
        metadata["generated_ratio"] = ""  # Leave blank on failure
    metadata["completed_epoch"] = ckpt.get("epoch", saved_args.get("_last_completed_epoch", getattr(args, "_last_completed_epoch", "")))  # Resolve completed epoch from saved state or live args
    metadata["timing_values"] = {  # Map common column names to stored timing attributes
        "training_time_s": ceil_seconds(getattr(args, "_last_training_time", saved_args.get("_last_training_time", ""))),  # Total training elapsed seconds
        "generated_time_s": ceil_seconds(getattr(args, "_last_sample_generation_time", saved_args.get("_last_sample_generation_time", ""))),  # Generation and output-write elapsed seconds
        "testing_time_s": ceil_seconds(getattr(args, "_last_testing_time", saved_args.get("_last_testing_time", ""))),  # Post-generation verification elapsed seconds
        "epoch_time_s": ceil_seconds(getattr(args, "_last_epoch_time", saved_args.get("_last_epoch_time", ""))),  # Last completed epoch elapsed seconds
        "sample_generation_time_s": ceil_seconds(getattr(args, "_last_sample_generation_time", saved_args.get("_last_sample_generation_time", ""))),  # Sample generation elapsed seconds
        "model_save_time_s": ceil_seconds(getattr(args, "_last_model_save_time", saved_args.get("_last_model_save_time", ""))),  # Model save phase elapsed seconds
    }  # End timing values map
    metric_key_map = {  # Map CSV result keys to metrics history list keys
        "critic_loss": "loss_D",  # Critic loss history key
        "generator_loss": "loss_G",  # Generator loss history key
        "gp": "gp",  # Gradient penalty history key
        "D_real": "D_real",  # Average real critic score history key
        "D_fake": "D_fake",  # Average fake critic score history key
        "wasserstein": "wasserstein",  # Estimated Wasserstein distance history key
    }  # End metric key map
    for result_key in metric_key_map:  # Initialize metric result fields as unavailable
        metadata[result_key] = ""  # Use existing empty-string convention for unavailable metrics
    try:  # Attempt to extract final training metrics from live and saved histories
        metrics_candidates = [getattr(args, "_last_metrics_history", None), ckpt.get("metrics_history"), saved_args.get("_last_metrics_history")]  # Prefer live metrics before saved metrics
        for metrics_history in metrics_candidates:  # Iterate metric history candidates in priority order
            if isinstance(metrics_history, dict):  # Use dictionaries that match the training history shape
                for result_key, history_key in metric_key_map.items():  # Iterate each supported CSV metric
                    history_values = metrics_history.get(history_key) or []  # Read metric history list safely
                    if metadata.get(result_key) in (None, "") and isinstance(history_values, (list, tuple)) and len(history_values) > 0:  # Fill only missing metric values
                        metadata[result_key] = history_values[-1]  # Store latest metric value
        if metadata.get("wasserstein") in (None, "") and metadata.get("D_real") not in (None, "") and metadata.get("D_fake") not in (None, ""):  # Derive Wasserstein only from logged critic scores
            metadata["wasserstein"] = safe_float(metadata.get("D_real"), 0.0) - safe_float(metadata.get("D_fake"), 0.0)  # Use existing D_real minus D_fake convention
    except Exception:  # Extraction failed
        for result_key in metric_key_map:  # Reset metric fields on extraction failure
            metadata[result_key] = ""  # Preserve empty-string convention on extraction failure
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
        "generated_time_s": "",  # Fallback when generation timing is unavailable
        "testing_time_s": "",  # Fallback when post-generation verification timing is unavailable
        "hardware": None,  # Hardware placeholder
    }  # End runtime defaults
    row_runtime = {}  # Dictionary to hold runtime values for configured columns
    row_runtime["original_file"] = metadata["original_file_name"]  # Original CSV filename
    row_runtime["generated_file"] = metadata["generated_file_name"]  # Generated output filename
    row_runtime["original_num_samples"] = metadata["original_num"] if metadata["original_num"] is not None else ""  # Original sample count
    row_runtime["generated_num_samples"] = metadata["generated_num"]  # Newly generated synthetic row count
    row_runtime["total_num_samples_after_augmentation"] = metadata["total_after_augmentation"]  # Store total
    row_runtime["generated_ratio"] = metadata["generated_ratio"]  # Generated/original ratio
    row_runtime["epoch"] = metadata["completed_epoch"]  # Completed training epoch used for generation
    row_runtime["critic_loss"] = metadata["critic_loss"]  # Last critic loss from checkpoint metrics
    row_runtime["generator_loss"] = metadata["generator_loss"]  # Last generator loss from checkpoint metrics
    row_runtime["gp"] = metadata.get("gp", "")  # Last gradient penalty from training metrics
    row_runtime["D_real"] = metadata.get("D_real", "")  # Last average real critic score from training metrics
    row_runtime["D_fake"] = metadata.get("D_fake", "")  # Last average fake critic score from training metrics
    row_runtime["wasserstein"] = metadata.get("wasserstein", "")  # Last Wasserstein estimate from training metrics
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
            row_runtime["hardware"] = build_hardware_csv_value(device)  # Store GA-style JSON hardware specification in runtime dict
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
        metadata = collect_generation_file_metadata(args, args_ck, ckpt, n, ck_csv_path)  # Collect file metadata for generation results
        row_runtime = build_generation_runtime_row(args, config, n, device, metadata)  # Build runtime metrics dictionary for CSV row
        validate_completed_generation_row(row_runtime, results_cols_cfg)  # Verify completed row fields before writing
        ordered = build_ordered_csv_row_from_runtime(row_runtime, results_cols_cfg, config)
        original_file = row_runtime.get("original_file", "")
        update_results_csv_row(results_csv_path, ordered, results_cols_cfg, original_file)
        print(f"[RESULTS] Updated {results_csv_path} for {original_file} ({n} generated samples)")


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
    release_cuda_memory()  # Release unused CUDA cache before propagating generation failure
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
        args._mixed_categorical_groups = list(scaler.categorical_groups)  # Use checkpoint-saved group positions during reconstruction.
        args._mixed_categorical_class_masks = scaler.categorical_class_masks
        G = build_and_load_generator(args, config, ckpt, device, feature_dim, n_classes)  # Build and load generator model from checkpoint
        n_per_class, labels, n = compute_generation_counts_and_labels(args, config, class_distribution, label_encoder, n_classes)  # Compute per-class counts and label array
        notify_start_of_generation(args, n)  # Send Telegram notification about generation start
        all_fake, all_labels, sample_generation_start_time = generate_batches_and_collect_results(args, G, device, labels, n)  # Generate synthetic samples in batches
        df = postprocess_generated_arrays_to_dataframe(args, config, all_fake, all_labels, scaler, label_encoder, feature_cols, device, n, file_progress_prefix)  # Postprocess arrays into DataFrame and save
        record_sample_generation_timing(args, sample_generation_start_time)  # Record sample generation elapsed time
        testing_start_time = time.time()  # Start post-generation verification and quality-output timing
        persisted_total_rows = count_persisted_output_rows(args.out_file)  # Count actual rows persisted to augmented CSV
        validate_augmentation_sample_totals(args, persisted_total_rows)  # Verify rows
        generate_generation_quality_outputs(args, config, df, feature_cols)  # Generate post-generation quality visualization outputs
        record_testing_timing(args, testing_start_time)  # Record post-generation verification and quality-output elapsed time
        generated_n = int(safe_float(getattr(args, "_last_generated_num_samples", n), n))  # Resolve actual synthetic row count for result persistence
        prepare_and_write_results_csv_row(args, config, args_ck, ckpt, generated_n, device)  # Persist each generated file immediately; surface errors.
        notify_generation_finish_via_telegram(args, generated_n, file_progress_prefix)  # Announce completion only after the results row is committed.
    except Exception as e:
        handle_generate_top_level_exception(e)  # Handle top-level exception by logging and re-raising


def generate_from_in_memory_generator(args, config: Dict, G: nn.Module, dataset: Any, device: torch.device) -> None:
    """
    Generate synthetic samples using an in-memory trained generator when no checkpoint is available on disk.

    :param args: Parsed arguments namespace containing generation options and output path.
    :param config: Configuration dictionary with generation and paths settings.
    :param G: Trained generator model already in memory from the training phase.
    :param dataset: CSVFlowDataset instance with scaler, label_encoder, feature_cols, and labels.
    :param device: Torch device for tensor allocation during generation.
    :return: None
    """

    try:
        if config is None:  # If no config provided
            config = CONFIG or get_default_config()  # Use global or default config

        file_progress_prefix = getattr(args, "file_progress_prefix", f"{BackgroundColors.CYAN}[1/1]{Style.RESET_ALL}")  # Retrieve colored progress prefix from args or use default

        scaler = dataset.scaler  # Extract fitted scaler from dataset for inverse transform
        label_encoder = dataset.label_encoder  # Extract fitted label encoder from dataset for label decoding
        feature_cols = dataset.feature_cols  # Extract feature column names from dataset for DataFrame reconstruction

        unique_labels, label_counts = np.unique(dataset.labels, return_counts=True)  # Compute class distribution from training dataset labels
        class_distribution = dict(zip(unique_labels.tolist(), label_counts.tolist()))  # Build label-to-count mapping for percentage-based generation

        n_classes = dataset.n_classes  # Get number of classes from dataset

        G.eval()  # Set generator to evaluation mode for deterministic inference
        args._generation_epoch = int(getattr(args, "_last_completed_epoch", 0))  # Store completed training epoch for generation diagnostics
        require_finite_model_state(G, "parameters", args, args._generation_epoch, 0, "generation")  # Reject invalid in-memory generator parameters before generation

        n_per_class, labels, n = compute_generation_counts_and_labels(args, config, class_distribution, label_encoder, n_classes)  # Compute per-class counts and label array
        notify_start_of_generation(args, n)  # Send Telegram notification about generation start
        all_fake, all_labels, sample_generation_start_time = generate_batches_and_collect_results(args, G, device, labels, n)  # Generate synthetic samples in batches
        df = postprocess_generated_arrays_to_dataframe(args, config, all_fake, all_labels, scaler, label_encoder, feature_cols, device, n, file_progress_prefix)  # Postprocess arrays into DataFrame and save
        record_sample_generation_timing(args, sample_generation_start_time)  # Record sample generation elapsed time
        testing_start_time = time.time()  # Start post-generation verification and quality-output timing
        persisted_total_rows = count_persisted_output_rows(args.out_file)  # Count actual rows persisted to augmented CSV
        validate_augmentation_sample_totals(args, persisted_total_rows)  # Verify rows
        generate_generation_quality_outputs(args, config, df, feature_cols)  # Generate post-generation quality visualization outputs
        record_testing_timing(args, testing_start_time)  # Record post-generation verification and quality-output elapsed time
        generated_n = int(safe_float(getattr(args, "_last_generated_num_samples", n), n))  # Resolve actual synthetic row count for result persistence

        in_memory_ckpt = {  # Metadata for generation without a disk checkpoint.
            "epoch": getattr(args, "_last_completed_epoch", ""),
            "metrics_history": getattr(args, "_last_metrics_history", {}),
            "args": vars(args),
        }
        prepare_and_write_results_csv_row(args, config, vars(args), in_memory_ckpt, generated_n, device)
        notify_generation_finish_via_telegram(args, generated_n, file_progress_prefix)  # Announce completion only after the results row is committed.

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
            except Exception as e:  # Total_seconds() failed
                print(str(e))  # Print error to terminal for visibility
                try:  # Attempt to notify about total_seconds failure via Telegram
                    send_exception_via_telegram(type(e), e, e.__traceback__)  # Send full error via Telegram
                except Exception:  # If notification fails, ignore to avoid recursion
                    pass  # Ignore Telegram send errors and fall through
        if hasattr(obj, "timestamp"):  # Datetime-like objects
            try:  # Attempt to call timestamp()
                return float(obj.timestamp())  # Use timestamp() to get seconds since epoch
            except Exception as e:  # Timestamp() failed
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
    Calculate the execution time and return a human-readable string.

    :param start_time: The start time or duration value (datetime, timedelta, or numeric seconds).
    :param finish_time: Optional finish time; if None, start_time is treated as the total duration.
    :return: Human-readable execution time string formatted as days, hours, minutes, and seconds.
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


def format_estimated_finish_time_suffix(eta_seconds: Optional[float]) -> str:
    """Format ETA-derived finish timestamp in Sao Paulo time."""

    try:
        if eta_seconds is None or not math.isfinite(float(eta_seconds)) or float(eta_seconds) < 0:
            return ""
        finish_time = datetime.datetime.now(ZoneInfo(SAO_PAULO_TIMEZONE_NAME)) + datetime.timedelta(seconds=float(eta_seconds))
        return f" | Estimated Finish Time: {finish_time.strftime('%Hh:%Mm-%d/%m/%Y')}"
    except Exception:
        return ""


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
            if isinstance(device_used, torch.device) and device_used is not None:  # If a valid torch.device was provided
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
    Top-level function to produce zebra row styles for pandas Styler.

    :param row: pandas Series representing a row
    :return: List[str] of CSS style strings for each cell
    """
    bg = "white" if (row.name % 2) == 0 else "#f2f2f2"  # White for even rows, light gray for odd rows
    return [f"background-color: {bg};" for _ in row.index]  # Return style for every column in the row


def play_sound(config: Optional[Dict] = None):
    """
    Play a sound when the program finishes and skip if the operating system is Windows.

    :param config: Optional configuration dictionary containing sound settings
    :return: None.
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
                if key == "batch_size":  # If direct call sets batch size
                    cli_style_overrides.setdefault("training", {})["batch_size_cli_override"] = True  # Preserve direct-call batch-size precedence
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
            elif key in ["checkpoint", "n_samples", "force_regenerate_samples", "label", "out_file", "gen_batch_size", "feature_dim"]:  # Generation params
                cli_style_overrides.setdefault("generation", {})[key] = value
            elif key in ["num_workers"]:  # DataLoader params
                cli_style_overrides.setdefault("dataloader", {})[key] = value
                cli_style_overrides.setdefault("dataloader", {})["num_workers_cli_override"] = True  # Preserve direct-call worker precedence
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
            if getattr(args, "_last_training_time", "") in ("", None):  # Preserve train-loop timing when train recorded it
                args._last_training_time = ceil_seconds(time.time() - training_start_time)  # Store fallback elapsed seconds with upward rounding
        elif args.mode == "gen":  # Generation mode
            if args.checkpoint is None:  # Verify checkpoint provided
                raise ValueError("Generation mode requires checkpoint path")
            generate(args, final_config)  # Generate samples
        elif args.mode == "both":  # Combined mode
            training_start_time = time.time()  # Record training start time using time.time()
            train(args, final_config)  # Train first
            if getattr(args, "_last_training_time", "") in ("", None):  # Preserve train-loop timing when train recorded it
                args._last_training_time = ceil_seconds(time.time() - training_start_time)  # Store fallback elapsed seconds with upward rounding
            if args.csv_path:  # If CSV provided
                csv_path_obj = Path(args.csv_path)  # Create Path object from csv_path for path operations
                data_aug_subdir = final_config.get("paths", {}).get("data_augmentation_dir", "Data_Augmentation")  # Read configured data augmentation base directory name
                data_aug_dir = csv_path_obj.parent / data_aug_subdir  # Construct data augmentation directory alongside input file
                os.makedirs(data_aug_dir, exist_ok=True)  # Ensure data augmentation directory exists before output file assignment
                results_suffix = final_config.get("wgangp", {}).get("results_suffix", "_data_augmented")  # Read output filename suffix from config
                output_filename = f"{csv_path_obj.stem}{results_suffix}{csv_path_obj.suffix}"  # Build output filename by appending suffix to input stem
                if args.out_file == "generated.csv":  # Update default output path when still at default value
                    samples_dir = data_aug_dir / final_config.get("paths", {}).get("data_augmentation_sample_dir", "Samples")  # Build augmented samples subdirectory path inside data augmentation directory
                    os.makedirs(samples_dir, exist_ok=True)  # Ensure augmented samples subdirectory exists before output file assignment
                    args.out_file = str(samples_dir / output_filename)  # Assign resolved output file path to args for generation
                checkpoint_dir = data_aug_dir / final_config.get("paths", {}).get("checkpoint_subdir", "Checkpoints") / FLOW_PREPROCESSING_CHECKPOINT_TAG  # Discover only class-conditional generator checkpoints.
                checkpoint_path = checkpoint_dir / f"{csv_path_obj.stem}_generator_epoch{args.epochs}.pt"  # Construct expected final-epoch checkpoint filename
                if not checkpoint_path.exists():  # Find latest if specific epoch not found
                    checkpoints = list(checkpoint_dir.glob(f"{csv_path_obj.stem}_generator_epoch*.pt"))  # Discover all matching generator checkpoints in directory
                    if checkpoints:  # Assign latest checkpoint when at least one file exists
                        checkpoint_path = max(checkpoints, key=extract_saved_epoch)  # Select latest checkpoint by numeric epoch
                    else:  # Raise a descriptive error when no checkpoints exist at all
                        raise FileNotFoundError(f"No generator checkpoint found for {csv_path_obj.name} in {checkpoint_dir}")  # Abort with informative message
                args.checkpoint = str(checkpoint_path)  # Assign resolved checkpoint path string to args for generation
            file_progress_prefix = getattr(args, "file_progress_prefix", f"{BackgroundColors.CYAN}[1/1]{Style.RESET_ALL}")  # Retrieve or default progress prefix for logging
            safe_log("info", f"Starting generation after training for {Path(args.csv_path).name if args.csv_path else 'unknown'}")  # Log generation start at INFO level
            generate(args, final_config)  # Generate samples
            if args.csv_path:  # If CSV path available for post-generation validation
                validate_augmentation_output(args, Path(args.csv_path), file_progress_prefix)  # Validate augmented output file exists, is non-empty, and has rows
        else:  # Invalid mode
            raise ValueError(f"Invalid mode: {args.mode}")
    except Exception as e:
        print(str(e))
        send_exception_via_telegram(type(e), e, e.__traceback__)
        raise


def run_wgangp(config: Optional[Union[Dict, str]] = None, **kwargs) -> None:
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
            cli_style_overrides = build_config_overrides_from_kwargs(kwargs)  # Convert kwargs to config overrides using extracted function
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
            dispatch_wgangp_execution_mode(args, final_config)  # Dispatch execution based on mode using extracted function
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

    :return: None.
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


def resolve_force_regenerate_samples_source(args: Any) -> str:
    """
    Resolve where the effective force-regenerate-samples value came from.

    :param args: Parsed CLI arguments namespace.
    :return: Source label for logging.
    """

    if getattr(args, "force_regenerate_samples", None) is not None:  # Prefer explicit CLI force-regeneration values.
        return "CLI"  # Return CLI as the effective source.
    config_path = Path(args.config) if getattr(args, "config", None) else Path("config.yaml")  # Resolve the same preferred configuration path used by load_configuration.
    if not config_path.exists():  # Use example configuration path only when primary configuration is absent.
        config_path = Path("config.yaml.example")  # Resolve fallback example configuration path.
    try:  # Inspect only the loaded generation section for source logging.
        if config_path.exists():  # Read source YAML only when an existing configuration file is present.
            with open(config_path, "r") as config_file:  # Open selected YAML configuration file.
                file_config = yaml.safe_load(config_file)  # Parse YAML for source detection.
            generation_config = file_config.get("generation", {}) if isinstance(file_config, dict) else {}  # Read generation settings.
            if "force_regenerate_samples" in generation_config or "force_new_samples" in generation_config:  # Detect configured current or legacy force-regeneration key.
                return f"configuration ({config_path})"  # Return configuration source path.
    except Exception:  # Fall back to default source when source detection fails.
        pass  # Preserve configuration loading behavior.
    return "fallback"  # Return fallback source when neither CLI nor YAML supplies the value.


def initialize_cli_and_config() -> Dict:
    """
    Parse CLI arguments, load configuration, and set the global CONFIG variable.

    :return: Merged configuration dictionary.
    """

    global CONFIG  # Declare global CONFIG for mutation
    args = parse_args()  # Parse command-line arguments from CLI
    set_runtime_process_name(args.process_name, script_path=__file__)  # Apply immediate CLI or generated htop identity before configuration and logging initialization.
    cli_overrides = args_to_config_overrides(args)  # Convert parsed CLI args to configuration overrides dict
    config = load_configuration(config_path=args.config, cli_overrides=cli_overrides)  # Load and merge configuration from file and CLI
    set_runtime_process_name(getattr(args, "process_name", None), script_path=__file__, config=config)  # Re-apply with merged config so config.yaml can override only when CLI omitted the process name.
    force_source = resolve_force_regenerate_samples_source(args)  # Resolve effective force-regeneration source for logging.
    print(f"{BackgroundColors.GREEN}[CONFIG] force_regenerate_samples={BackgroundColors.CYAN}{bool(config.get('generation', {}).get('force_regenerate_samples', False))}{BackgroundColors.GREEN} | Source: {BackgroundColors.CYAN}{force_source}{Style.RESET_ALL}")  # Log effective force-regeneration value and source.
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
        data_aug_subdir = config.get("paths", {}).get("data_augmentation_dir", "Data_Augmentation")  # Get configured data augmentation base directory name for this branch
        data_aug_dir = csv_path_obj.parent / data_aug_subdir  # Construct data augmentation subdirectory path alongside input file
        os.makedirs(data_aug_dir, exist_ok=True)  # Ensure data augmentation directory exists before output path assignment
        output_filename = f"{csv_path_obj.stem}{results_suffix}{csv_path_obj.suffix}"  # Build output filename by appending suffix to input stem
        samples_dir = data_aug_dir / config.get("paths", {}).get("data_augmentation_sample_dir", "Samples")  # Build augmented samples subdirectory path inside data augmentation directory
        os.makedirs(samples_dir, exist_ok=True)  # Ensure augmented samples subdirectory exists before output file assignment
        args.out_file = str(samples_dir / output_filename)  # Assign resolved output file path to args for downstream use
    data_aug_subdir = config.get("paths", {}).get("data_augmentation_dir", "Data_Augmentation")  # Re-read data augmentation base directory name for authoritative path computation
    data_aug_dir = csv_path_obj.parent / data_aug_subdir  # Build canonical data augmentation directory path for results CSV
    os.makedirs(data_aug_dir, exist_ok=True)  # Ensure canonical data augmentation directory exists before results CSV creation
    return data_aug_dir  # Return canonical data augmentation directory path for downstream use


def execute_training_with_timing(args: Any, config: Dict) -> Optional[tuple]:
    """
    Execute model training and store elapsed time on args.

    :param args: Argument namespace with all training configuration settings.
    :param config: Configuration dictionary passed to the train function.
    :return: Tuple of (Generator, CSVFlowDataset, device) from train, or None on failure.
    """

    training_start_time = time.time()  # Record training start timestamp for elapsed time calculation
    result = train(args, config)  # Execute WGAN-GP training loop with provided configuration
    if getattr(args, "_last_training_time", "") in ("", None):  # Preserve train-recorded elapsed seconds when available
        args._last_training_time = ceil_seconds(time.time() - training_start_time)  # Store fallback training elapsed seconds with upward rounding
    return result  # Return training artifacts for in-memory generation fallback


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


def skip_both_mode_when_augmented_output_sufficient(args: Any, config: Dict, csv_path_obj: Path, file_progress_prefix: str) -> bool:
    """
    Return whether both-mode training and generation can be skipped for an existing augmented output.

    :param args: Argument namespace containing generation settings and output path.
    :param config: Configuration dictionary passed to generation sizing functions.
    :param csv_path_obj: Path object for the input CSV file being processed.
    :param file_progress_prefix: Colored prefix string for progress display.
    :return: True when the existing output already satisfies the requested augmented row count.
    """

    out_path = Path(getattr(args, "out_file", ""))  # Resolve the existing output path using configured per-file output logic.
    force_regenerate = bool(getattr(args, "force_regenerate_samples", False))  # Resolve the effective force-regeneration value once for this file.
    requested_ratio = safe_float(getattr(args, "n_samples", config.get("generation", {}).get("n_samples", 1.0)), config.get("generation", {}).get("n_samples", 1.0))  # Resolve the effective requested generation ratio or absolute count.
    required_rows = None if not out_path.exists() or force_regenerate else resolve_expected_output_row_count(args, config)  # Avoid loading large source datasets for nonexistent or forcibly replaced outputs.
    original_rows = "unavailable"  # Initialize source row count for logging when source read fails.

    try:  # Read source row count for skip evidence only.
        original_rows = len(pd.read_csv(str(csv_path_obj), low_memory=False))  # Count original source rows without preprocessing side effects.
    except Exception:  # Keep missing source count from masking normal training.
        original_rows = "unavailable"  # Preserve unavailable source count in the skip decision log.

    if force_regenerate:  # Bypass skip when regeneration is explicitly enabled.
        message = f"{file_progress_prefix} [INFO] Force regeneration enabled; training will run for {csv_path_obj} | Output={out_path} | Existing rows=not evaluated | Required rows={required_rows} | Requested n_samples={requested_ratio} | Force regeneration={force_regenerate}"  # Build force-bypass evidence.
        print(f"{BackgroundColors.YELLOW}{message}{Style.RESET_ALL}")  # Log force-bypass evidence before training.
        safe_log("info", message)  # Persist force-bypass evidence through existing logger.
        send_telegram_message(TELEGRAM_BOT, message)  # Notify force-bypass evidence through existing Telegram path.
        return False  # Continue normal train and generate flow.

    if required_rows is None or not out_path.exists():  # Continue when output is absent or required count is unavailable.
        return False  # Proceed through normal training and generation.

    try:  # Validate existing output readability and count rows.
        existing_rows = len(pd.read_csv(out_path, low_memory=False))  # Count existing augmented output rows with the existing CSV reader behavior.
    except Exception as read_error:  # Continue normal flow for corrupt or unreadable output.
        message = f"{file_progress_prefix} [INFO] Existing augmented output is unreadable; training will run for {csv_path_obj} | Output={out_path} | Error={read_error} | Required rows={required_rows} | Requested n_samples={requested_ratio} | Force regeneration={force_regenerate}"  # Build unreadable-output evidence.
        print(f"{BackgroundColors.YELLOW}{message}{Style.RESET_ALL}")  # Log unreadable-output evidence before training.
        safe_log("info", message)  # Persist unreadable-output evidence through existing logger.
        return False  # Proceed through normal training and generation.

    message = f"{file_progress_prefix} [INFO] Existing augmented output evaluated for {csv_path_obj} | Output={out_path} | Original rows={original_rows} | Existing rows={existing_rows} | Required rows={required_rows} | Requested n_samples={requested_ratio} | Force regeneration={force_regenerate}"  # Build complete sufficiency evidence.
    print(f"{BackgroundColors.GREEN if existing_rows >= int(required_rows) else BackgroundColors.YELLOW}{message}{Style.RESET_ALL}")  # Log sufficiency evidence before any training starts.
    safe_log("info", message)  # Persist sufficiency evidence through existing logger.
    send_telegram_message(TELEGRAM_BOT, message)  # Notify sufficiency evidence through existing Telegram path.

    provenance_file = out_path.with_suffix(".provenance.json")  # Locate the persisted output provenance manifest.
    try:  # Verify the source identity and hyperparameters associated with the existing synthetic CSV.
        existing_provenance = json.loads(provenance_file.read_text(encoding="utf-8"))  # Read the output manifest when present.
        quality_path = out_path.with_suffix(".quality.json")
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        provenance_valid = existing_provenance.get("sha256") == build_wgangp_provenance(args)["sha256"] and quality.get("schema_version") == 3 and quality.get("accepted") is True and quality.get("quality_policy_sha256") == build_quality_policy_fingerprint(config)  # Reuse only validated prior output.
    except (OSError, ValueError, TypeError, AttributeError):  # Treat missing or malformed output manifests as unverified.
        provenance_valid = False  # Require regeneration for unverifiable existing outputs.
    if existing_rows == int(required_rows) and provenance_valid:  # Skip only matching output size and verified source/model configuration.
        skip_message = f"{file_progress_prefix} [INFO] Skipping WGAN-GP training and generation for {csv_path_obj.name}: existing augmented output already satisfies requested size."  # Build skip notification.
        print(f"{BackgroundColors.GREEN}{skip_message}{Style.RESET_ALL}")  # Log skip before returning to directory loop.
        safe_log("info", skip_message)  # Persist skip through existing logger.
        send_telegram_message(TELEGRAM_BOT, skip_message)  # Notify skip through existing Telegram path.
        return True  # Skip both training and generation for this file only.

    return False  # Proceed through normal training and generation for missing capacity or incompatible provenance.


def resolve_checkpoint_after_training(args: Any, config: Dict, csv_path_obj: Path, data_aug_dir: Path) -> Optional[Path]:
    """
    Resolve the generator checkpoint path produced after training and assign it to args.

    :param args: Argument namespace containing epochs count and checkpoint attribute.
    :param config: Configuration dictionary with paths settings.
    :param csv_path_obj: Path object for the trained CSV source file.
    :param data_aug_dir: Path object for the data augmentation output directory.
    :return: Resolved checkpoint Path object pointing to the latest available checkpoint, or None when no checkpoint exists.
    """

    checkpoint_prefix = csv_path_obj.stem  # Derive checkpoint filename prefix from input CSV stem
    checkpoint_subdir = config.get("paths", {}).get("checkpoint_subdir", "Checkpoints")  # Read configured checkpoint subdirectory name
    checkpoint_dir = data_aug_dir / checkpoint_subdir / FLOW_PREPROCESSING_CHECKPOINT_TAG  # Keep checkpoint discovery in the versioned preprocessing namespace.
    checkpoint_path = checkpoint_dir / f"{checkpoint_prefix}_generator_epoch{args.epochs}.pt"  # Construct expected final-epoch checkpoint filename
    
    if not checkpoint_path.exists():  # Fall back to latest checkpoint when specific epoch file is absent
        checkpoints = list(checkpoint_dir.glob(f"{checkpoint_prefix}_generator_epoch*.pt"))  # Discover all matching checkpoints in directory
        if checkpoints:  # Assign latest checkpoint when at least one file exists
            checkpoint_path = max(checkpoints, key=extract_saved_epoch)  # Select latest checkpoint by numeric epoch
        else:  # No checkpoint files found on disk; signal in-memory fallback to caller
            print(f"{BackgroundColors.YELLOW}[WARNING] No generator checkpoint found for {csv_path_obj.name} in {checkpoint_dir}. Will use in-memory generator.{Style.RESET_ALL}")  # Warn about missing checkpoint and inform fallback strategy
            return None  # Return None to signal in-memory generation fallback
        
    args.checkpoint = str(checkpoint_path)  # Assign resolved checkpoint path string to args for generation
    
    return checkpoint_path  # Return resolved Path object for caller printing use


def validate_augmentation_output(args: Any, csv_path_obj: Path, file_progress_prefix: str) -> None:
    """
    Validate that the augmented CSV file was written correctly after generation.

    :param args: Argument namespace containing out_file path to validate.
    :param csv_path_obj: Path object for the original input CSV file.
    :param file_progress_prefix: Colored prefix string for progress display.
    :return: None.
    """

    out_path = Path(args.out_file)  # Resolve output file path from args
    original_name = csv_path_obj.name  # Extract original CSV filename for error messages
    if not out_path.exists():  # Verify augmented file exists on disk
        error_msg = f"[ERROR] WGAN-GP augmentation failed on {original_name} — generated dataset was not written at {out_path}"  # Build error message with file path
        print(f"{BackgroundColors.RED}{error_msg}{Style.RESET_ALL}")  # Print error to console in red
        send_telegram_message(TELEGRAM_BOT, error_msg)  # Send failure notification via Telegram
        sys.exit(1)  # Terminate the script immediately to prevent silent continuation
    file_size = out_path.stat().st_size  # Read file size in bytes from disk
    if file_size == 0:  # Verify file is not empty
        error_msg = f"[ERROR] WGAN-GP augmentation failed on {original_name} — augmented file {out_path.name} exists but is empty (0 bytes)"  # Build zero-size error message
        print(f"{BackgroundColors.RED}{error_msg}{Style.RESET_ALL}")  # Print error to console in red
        send_telegram_message(TELEGRAM_BOT, error_msg)  # Send zero-size failure notification via Telegram
        sys.exit(1)  # Terminate the script immediately due to empty output
    try:  # Attempt to read the augmented file to verify row count
        row_count = len(pd.read_csv(str(out_path), low_memory=False))  # Count rows in the written augmented CSV
    except Exception as read_err:  # Handle unreadable file gracefully
        error_msg = f"[ERROR] WGAN-GP augmentation failed on {original_name} — augmented file {out_path.name} is unreadable: {read_err}"  # Build read-failure error message
        print(f"{BackgroundColors.RED}{error_msg}{Style.RESET_ALL}")  # Print error to console in red
        send_telegram_message(TELEGRAM_BOT, error_msg)  # Send read-failure notification via Telegram
        sys.exit(1)  # Terminate the script due to unreadable output
    if row_count == 0:  # Verify the augmented file contains at least one data row
        error_msg = f"[ERROR] WGAN-GP augmentation failed on {original_name} — augmented file {out_path.name} has 0 data rows"  # Build zero-row error message
        print(f"{BackgroundColors.RED}{error_msg}{Style.RESET_ALL}")  # Print error to console in red
        send_telegram_message(TELEGRAM_BOT, error_msg)  # Send zero-row failure notification via Telegram
        sys.exit(1)  # Terminate the script due to zero-row output
    expected_generated_rows = getattr(args, "_last_generated_num_samples", "")  # Read expected rows
    expected_generated_count = int(safe_float(expected_generated_rows, 0.0))  # Convert expected rows
    if expected_generated_rows not in (None, "") and row_count != expected_generated_count:  # Verify rows
        error_msg = (  # Build error
            f"[ERROR] WGAN-GP augmentation failed on {original_name} — generated file "  # Format file
            f"{out_path.name} has {row_count} rows but expected {expected_generated_rows}"  # Format counts
        )  # Finish error
        print(f"{BackgroundColors.RED}{error_msg}{Style.RESET_ALL}")  # Print error to console in red
        send_telegram_message(TELEGRAM_BOT, error_msg)  # Send row-mismatch failure notification via Telegram
        sys.exit(1)  # Terminate the script due to mismatched output rows
    print(  # Print saved dataset
        f"{file_progress_prefix} {BackgroundColors.GREEN}[INFO] Generated dataset successfully "  # Format status
        f"saved ({BackgroundColors.CYAN}{row_count}{BackgroundColors.GREEN} rows){Style.RESET_ALL}"  # Format count
    )  # Finish output
    send_telegram_message(  # Send saved dataset
        TELEGRAM_BOT,  # Pass bot
        f"{file_progress_prefix} [INFO] Generated dataset successfully saved ({row_count} rows) "  # Format status
        f"at {out_path}",  # Format path
    )  # Finish message


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

    file_progress_prefix = getattr(args, "file_progress_prefix", f"{BackgroundColors.CYAN}[1/1]{Style.RESET_ALL}")  # Retrieve colored progress prefix from args or use default
    args._file_processing_start_time = time.time()  # Record combined per-file processing start before training begins
    if skip_both_mode_when_augmented_output_sufficient(args, config, csv_path_obj, file_progress_prefix):  # Skip expensive work when the existing augmented output is already large enough.
        return  # Continue with the next file in directory or batch mode.
    print(f"{BackgroundColors.GREEN}[1/2] {training_label}{Style.RESET_ALL}")  # Print training phase progress header with label
    training_result = execute_training_with_timing(args, config)  # Train model and capture in-memory artifacts for fallback generation
    checkpoint_path = resolve_checkpoint_after_training(args, config, csv_path_obj, data_aug_dir)  # Resolve generator checkpoint produced by training

    if checkpoint_path is not None:  # Checkpoint file available on disk for standard generation path
        print(f"\n{BackgroundColors.CYAN}[2/2] Generating samples from {checkpoint_path.name}...{Style.RESET_ALL}")  # Print generation phase progress header with checkpoint filename
        print(f"{BackgroundColors.GREEN}Output will be saved to: {BackgroundColors.CYAN}{args.out_file}{Style.RESET_ALL}")  # Print resolved output file destination path
        print(f"{file_progress_prefix} {BackgroundColors.GREEN}[INFO] Generating synthetic samples for {BackgroundColors.CYAN}{csv_path_obj.name}{Style.RESET_ALL}")  # Log generation start for this CSV file
        send_telegram_message(TELEGRAM_BOT, f"{file_progress_prefix} [INFO] Generating synthetic samples for {csv_path_obj.name}")  # Send generation start notification via Telegram
        execute_generation_with_verification(args, config)  # Verify necessity and execute synthetic sample generation from checkpoint
    else:  # No checkpoint on disk; fall back to in-memory trained generator
        print(f"\n{BackgroundColors.CYAN}[2/2] Generating samples from in-memory trained generator...{Style.RESET_ALL}")  # Print generation phase header for in-memory fallback
        print(f"{BackgroundColors.GREEN}Output will be saved to: {BackgroundColors.CYAN}{args.out_file}{Style.RESET_ALL}")  # Print resolved output file destination path
        print(f"{file_progress_prefix} {BackgroundColors.YELLOW}[INFO] Checkpoint saving was skipped; using in-memory generator for {csv_path_obj.name}{Style.RESET_ALL}")  # Log in-memory fallback for this CSV file
        send_telegram_message(TELEGRAM_BOT, f"{file_progress_prefix} [INFO] Using in-memory generator for {csv_path_obj.name} (checkpoint was skipped)")  # Send in-memory fallback notification via Telegram
        if training_result is not None:  # Verify training returned generator artifacts successfully
            G, dataset, device = training_result  # Unpack trained generator, dataset, and device from training result
            if verify_data_augmentation_file(args, config):  # Verify if generation is necessary according to existing output and configuration
                generate_from_in_memory_generator(args, config, G, dataset, device)  # Generate synthetic samples using in-memory trained generator
            else:  # Skip generation when configured n_samples already satisfied by existing output
                print(f"{BackgroundColors.GREEN}Skipping generation: output file already satisfies configured n_samples.{Style.RESET_ALL}")  # Inform user that generation is skipped
        else:  # Training failed to return artifacts; cannot generate without checkpoint or in-memory generator
            error_msg = f"[ERROR] Training did not return generator artifacts and no checkpoint exists for {csv_path_obj.name}"  # Build descriptive error message
            print(f"{BackgroundColors.RED}{error_msg}{Style.RESET_ALL}")  # Print error to console in red
            send_telegram_message(TELEGRAM_BOT, error_msg)  # Send failure notification via Telegram
            raise RuntimeError(error_msg)  # Raise to prevent silent failure

    print(f"{file_progress_prefix} {BackgroundColors.GREEN}[INFO] Saving augmented dataset: {BackgroundColors.CYAN}{Path(args.out_file).name}{Style.RESET_ALL}")  # Log augmented dataset file name being validated
    validate_augmentation_output(args, csv_path_obj, file_progress_prefix)  # Validate augmented output file exists, is non-empty, and has rows


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
    data_aug_subdir = config.get("paths", {}).get("data_augmentation_dir", "Data_Augmentation")  # Get configured data augmentation base directory name
    data_aug_dir = csv_path_obj.parent / data_aug_subdir  # Construct data augmentation directory path relative to input file
    os.makedirs(data_aug_dir, exist_ok=True)  # Ensure data augmentation directory exists before output file creation
    output_filename = f"{csv_path_obj.stem}{results_suffix}{csv_path_obj.suffix}"  # Build output filename by appending suffix to input stem
    samples_dir = data_aug_dir / config.get("paths", {}).get("data_augmentation_sample_dir", "Samples")  # Build augmented samples subdirectory path inside data augmentation directory
    os.makedirs(samples_dir, exist_ok=True)  # Ensure augmented samples subdirectory exists before output file creation
    args.out_file = str(samples_dir / output_filename)  # Assign computed output file path to args
    args.csv_path = file  # Assign current CSV file path to args for train and generate functions
    # One namespace is reused for every source in batch mode: never carry a
    # previous file's sample counts, epoch duration, or metrics into this row.
    for previous_field in ("_last_original_num_samples", "_last_original_preprocessed_num_samples",
                           "_last_generated_num_samples", "_last_total_num_samples_after_augmentation",
                           "_last_sample_generation_time", "_last_testing_time", "_last_epoch_time",
                           "_last_completed_epoch", "_last_metrics_history", "_last_file_time"):
        if hasattr(args, previous_field):
            delattr(args, previous_field)
    args._last_training_time = 0.0
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

    data_aug_subdir = config.get("paths", {}).get("data_augmentation_dir", "Data_Augmentation")  # Read configured data augmentation base directory name
    data_aug_dir = Path(input_path) / data_aug_subdir  # Construct data augmentation directory path inside dataset root
    per_dir_results_csv = data_aug_dir / "data_augmentation_results.csv"  # Build path for per-directory results CSV
    
    create_results_csv_if_absent(per_dir_results_csv, results_cols)  # Write header to results CSV when file does not yet exist
    
    files_to_process = get_files_to_process(input_path, file_extension=".csv", config=config)  # Collect all CSV files in dataset directory
    generating_order = config.get("wgangp", {}).get("generating_order", "off")  # Read configured generation ordering strategy
    files_to_process = apply_dataset_ordering(files_to_process, generating_order)  # Apply ordering strategy to collected file list
    total_files = len(files_to_process)  # Count total files for progress display
    failed_files = []  # Collect per-file failures for final error propagation
    verbose_output(f"Found {total_files} CSV files in {input_path}")  # Log total discovered CSV file count for this directory
    if total_files == 0:  # Warn and return early when no CSV files are present in directory
        print(f"{BackgroundColors.YELLOW}Warning: no CSV files found in directory: {BackgroundColors.CYAN}{input_path}{Style.RESET_ALL}")  # Print actionable warning about empty directory
        return  # Exit without processing when directory yields no CSV files
    
    for index, file in enumerate(files_to_process, start=1):  # Iterate files with 1-based index for progress tracking
        try:  # Catch per-file errors to allow batch to continue after individual failures
            process_single_dataset_file(args, config, mode, file, index, total_files, results_suffix)  # Process file with progress tracking
        except Exception as e:  # Handle any unrecovered error from file processing
            print(f"{BackgroundColors.RED}Error processing {BackgroundColors.CYAN}{file}{BackgroundColors.RED}: {e}{Style.RESET_ALL}")  # Print descriptive error with file path
            traceback.print_exc()  # Print full exception traceback for debugging investigation
            try:  # Attempt to send error notification via Telegram so monitoring users see failures
                send_telegram_message(TELEGRAM_BOT, f"[ERROR] Failed to process {Path(file).name}: {e}")  # Send per-file error to Telegram for remote visibility
            except Exception:  # Ignore Telegram notification failures to avoid masking the original error
                pass  # Continue to next file even if notification fails
            failed_files.append(f"{Path(file).name}: {e}")  # Record failure while preserving next-file processing
            if isinstance(e, getattr(torch, "AcceleratorError", ())):  # Stop immediately when CUDA reports an unrecoverable accelerator failure
                raise  # Propagate poisoned CUDA context instead of processing another file
            continue  # Advance to next file despite error in current file
    if failed_files:  # Propagate aggregate failure after all discovered files were attempted
        raise RuntimeError(f"Failed to process {len(failed_files)} of {total_files} files in {input_path}: {'; '.join(failed_files)}")  # Prevent failed batch from reporting success


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
        dataset_name = str(dataset_name).strip()  # Normalize dataset name by removing leading and trailing spaces
        paths = [p.strip() if isinstance(p, str) else p for p in paths] if isinstance(paths, list) else paths  # Normalize paths list by stripping each string entry
        print(f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing dataset: {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}")  # Announce current dataset name
        for input_path in paths:  # Iterate over each directory path within the dataset
            input_path = input_path.strip() if isinstance(input_path, str) else input_path  # Normalize input path by removing leading and trailing spaces
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
    Run the main CLI entry point for command-line argument parsing, configuration loading, and execution routing.

    :return: None.
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

        if csv_path is not None and os.path.isdir(csv_path):  # Directory mode (csv_path is an existing directory)
            safe_debug(f"Running on directory: {csv_path}")  # Log directory mode entry with resolved path
            process_dataset_path(args, config, mode, csv_path, results_cols, results_suffix)  # Process all CSV files discovered inside the provided directory
        elif csv_path is not None:  # Single file mode (csv_path is a file path)
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
