"""
Build verified partition-local sequence windows for LSTM inputs.
"""
if __name__ in {"__main__", "__mp_main__"}:
    try:
        from setproctitle import setproctitle
        setproctitle(f"DDoS-{__file__.rsplit('/', 1)[-1].rsplit('.', 1)[0]}")
    except ImportError:
        pass


from typing import Any, Optional

import numpy as np
import pandas as pd


class LSTMSequenceMetadataError(ValueError):
    """Raised when row metadata is insufficient for safe LSTM windowing."""


def build_lstm_sequence_windows(features: Any, labels: Any, row_metadata: pd.DataFrame, sequence_length: int, sequence_stride: int = 1, minimum_group_length: Optional[int] = None, label_strategy: str = "final_timestep", mixed_label_window_policy: str = "keep") -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Convert one already-split partition into source-local chronological windows.

    :param features: Partition feature matrix shaped rows by features.
    :param labels: Partition labels aligned to rows.
    :param row_metadata: DataFrame with source_file and row_order columns aligned to rows.
    :param sequence_length: Number of timesteps per generated window.
    :param sequence_stride: Row stride between window starts inside a group.
    :param minimum_group_length: Minimum rows required per group; defaults to sequence_length.
    :param label_strategy: Supported strategy, final_timestep.
    :param mixed_label_window_policy: keep, drop, or reject.
    :return: Tuple of sequence features, sequence labels, and generation metadata.
    """

    X = np.asarray(features)
    y = np.asarray(labels).reshape(-1)
    if X.ndim != 2:
        raise LSTMSequenceMetadataError(f"LSTM sequence builder requires 2D partition features, got {X.shape}")
    if X.shape[0] != y.shape[0]:
        raise LSTMSequenceMetadataError("LSTM sequence builder requires row-aligned features and labels")
    if row_metadata is None or len(row_metadata) != X.shape[0]:
        raise LSTMSequenceMetadataError("LSTM sequence builder requires row metadata aligned to the active partition")
    if "source_file" not in row_metadata.columns or "row_order" not in row_metadata.columns:
        missing = [name for name in ("source_file", "row_order") if name not in row_metadata.columns]
        raise LSTMSequenceMetadataError(f"LSTM sequence metadata is missing required field(s): {missing}")
    if label_strategy != "final_timestep":
        raise LSTMSequenceMetadataError("Only final_timestep LSTM label alignment is supported")
    if mixed_label_window_policy not in {"keep", "drop", "reject"}:
        raise LSTMSequenceMetadataError("mixed_label_window_policy must be keep, drop, or reject")

    seq_len = int(sequence_length)
    stride = int(sequence_stride)
    min_group = int(minimum_group_length or seq_len)
    if seq_len < 2 or stride < 1 or min_group < seq_len:
        raise LSTMSequenceMetadataError("Invalid LSTM sequence length, stride, or minimum group length")

    metadata = row_metadata.reset_index(drop=True).copy()
    metadata["partition_position"] = np.arange(len(metadata), dtype=np.int64)
    metadata["row_order"] = pd.to_numeric(metadata["row_order"], errors="raise")
    sequences = []
    sequence_labels = []
    mixed_windows = 0
    skipped_groups = 0
    discarded_prefix_rows = 0

    for _, group in metadata.groupby("source_file", sort=False):
        ordered_positions = group.sort_values("row_order", kind="mergesort")["partition_position"].to_numpy(dtype=np.int64)
        if ordered_positions.shape[0] < min_group:
            skipped_groups += 1
            discarded_prefix_rows += int(ordered_positions.shape[0])
            continue
        for start in range(0, ordered_positions.shape[0] - seq_len + 1, stride):
            window_positions = ordered_positions[start:start + seq_len]
            window_labels = y[window_positions]
            if not np.all(window_labels == window_labels[-1]):
                mixed_windows += 1
                if mixed_label_window_policy == "reject":
                    raise LSTMSequenceMetadataError("Mixed-label LSTM window encountered with reject policy")
                if mixed_label_window_policy == "drop":
                    continue
            sequences.append(X[window_positions])
            sequence_labels.append(window_labels[-1])

    if not sequences:
        raise LSTMSequenceMetadataError("LSTM sequence builder produced no windows; check source grouping and sequence_length")

    sequence_features = np.asarray(sequences, dtype=X.dtype)
    sequence_targets = np.asarray(sequence_labels, dtype=y.dtype)
    generation_metadata = {
        "chronological_field": "row_order",
        "group_fields": ("source_file",),
        "sequence_source_row_count": int(X.shape[0]),
        "discarded_prefix_rows": int(discarded_prefix_rows),
        "mixed_label_windows": int(mixed_windows),
        "skipped_groups": int(skipped_groups),
        "generated_sequences": int(sequence_features.shape[0]),
    }
    return sequence_features, sequence_targets, generation_metadata
