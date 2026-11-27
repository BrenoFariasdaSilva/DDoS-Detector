"""Bounded, lossless storage of large numeric evaluation matrices."""
import errno
import os
from pathlib import Path
import shutil

import numpy as np


def allocate_matrix(path, shape, dtype):
    """Reserve storage before mapping, avoiding sparse-file ENOSPC/SIGBUS."""
    path = Path(path)
    dtype = np.dtype(dtype)
    if dtype.hasobject or len(shape) != 2 or min(shape) <= 0:
        raise ValueError(f"Cannot map matrix {shape}, {dtype}")
    size = int(shape[0]) * int(shape[1]) * dtype.itemsize
    if shutil.disk_usage(path.parent).free < size:
        raise OSError(errno.ENOSPC, f"Insufficient space for {size} bytes", str(path))
    created = False
    try:
        with path.open('xb') as stream:
            created = True
            if hasattr(os, 'posix_fallocate'):
                # On Linux fail explicitly if the filesystem cannot reserve the bytes.
                os.posix_fallocate(stream.fileno(), 0, size)
            else:
                stream.truncate(size)
        return np.memmap(path, dtype=dtype, mode='r+', shape=shape, order='C')
    except BaseException:
        if created:
            path.unlink(missing_ok=True)
        raise


def close_matrix(matrix):
    if isinstance(matrix, np.memmap) and not matrix._mmap.closed:
        try:
            matrix.flush()
        finally:
            matrix._mmap.close()


def copy_matrix(source, path, dtype=None, chunk_rows=100000):
    """Copy/cast in row chunks, retaining the original order and values."""
    result = allocate_matrix(path, source.shape, source.dtype if dtype is None else dtype)
    try:
        for start in range(0, len(source), chunk_rows):
            result[start:start + chunk_rows] = source[start:start + chunk_rows]
        result.flush()
        return result
    except BaseException:
        close_matrix(result)
        raise


class DiskSyntheticSource:
    """Same cleaned candidate rows, ordered original feature schema, and labels."""
    def __init__(self, frame, columns, directory, chunk_rows=100000):
        self.directory = Path(directory)
        self.matrix = None
        self.labels = None
        self.columns = tuple(columns)
        try:
            numeric = frame.iloc[:0, :-1].select_dtypes(include=np.number)
            if not set(columns).issubset(numeric.columns):
                raise ValueError('Synthetic candidate schema is missing original numeric columns')
            dtype = numeric[list(columns)].to_numpy(copy=False).dtype
            self.matrix = allocate_matrix(self.directory / 'synthetic.dat', (len(frame), len(columns)), dtype)
            for start in range(0, len(frame), chunk_rows):
                self.matrix[start:start + chunk_rows] = frame.iloc[start:start + chunk_rows][list(columns)].to_numpy(copy=False)
            self.matrix.flush()
            self.labels = frame.iloc[:, -1].to_numpy(copy=True)
        except BaseException:
            self.close()
            raise

    @property
    def empty(self):
        return self.labels is None or not len(self.labels)

    def close(self):
        close_matrix(self.matrix)
        self.matrix = None
        self.labels = None
        shutil.rmtree(self.directory, ignore_errors=True)


def merge_scaled_training(original, synthetic, positions, scaler, path, chunk_rows=100000):
    """Write the exact original-then-synthetic matrix without a dense concat."""
    # Transform a real row to match sklearn's output dtype, including integer inputs.
    transformed_dtype = scaler.transform(synthetic[positions[:1]]).dtype if len(positions) else original.dtype
    result = allocate_matrix(path, (len(original) + len(positions), original.shape[1]), np.result_type(original.dtype, transformed_dtype))
    try:
        for start in range(0, len(original), chunk_rows):
            end = min(start + chunk_rows, len(original))
            result[start:end] = original[start:end]
        for start in range(0, len(positions), chunk_rows):
            selected = positions[start:start + chunk_rows]
            result[len(original) + start:len(original) + start + len(selected)] = scaler.transform(synthetic[selected])
        result.flush()
        return result
    except BaseException:
        close_matrix(result)
        raise
