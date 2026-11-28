"""Deterministic GA evaluation scheduling and durable, identity-scoped storage."""
from collections import defaultdict
from concurrent.futures import as_completed
import hashlib
import json
import os
from pathlib import Path
import pickle
import sqlite3
import tempfile

import numpy as np


def file_digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _digest_stream(stream)


def _digest_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def identity(*parts):
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()


def array_digest(array):
    array = np.asarray(array)
    digest = hashlib.sha256(str((array.shape, array.dtype.str)).encode())
    if array.dtype.hasobject:
        raise ValueError('GA cache requires numeric or fixed-width label arrays')
    for start in range(0, len(array), 65536):
        digest.update(np.ascontiguousarray(array[start:start + 65536]).tobytes())
    return digest.hexdigest()


def atomic_pickle(path, value, protocol=pickle.HIGHEST_PROTOCOL):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            pickle.dump(value, stream, protocol=protocol)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if os.name == 'posix':
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class FitnessStore:
    """Only the coordinator writes; every completed result is committed separately."""
    def __init__(self, path):
        self.connection = sqlite3.connect(path, timeout=60)
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.execute('CREATE TABLE IF NOT EXISTS fitness (mask TEXT PRIMARY KEY, metrics TEXT NOT NULL)')
        self.connection.commit()

    @staticmethod
    def key(mask):
        return ''.join(str(int(bit)) for bit in mask)

    def get(self, mask):
        row = self.connection.execute('SELECT metrics FROM fitness WHERE mask=?', (self.key(mask),)).fetchone()
        if row is None:
            return None
        result = tuple(json.loads(row[0]))
        if len(result) != 13 or result[-1] != sum(mask) or not all(np.isfinite(result)):
            raise ValueError('Invalid cached GA fitness; inspect cache before continuing')
        return result

    def put(self, mask, metrics):
        values = tuple(float(x) for x in metrics)
        if len(values) != 13 or values[-1] != sum(mask) or not all(np.isfinite(values)):
            raise ValueError('Invalid GA fitness result')
        with self.connection:
            self.connection.execute('INSERT OR REPLACE INTO fitness VALUES (?, ?)', (self.key(mask), json.dumps(values, allow_nan=False)))

    def close(self):
        self.connection.close()


def evaluate_batch(masks, evaluate, cached, save, completed, executor=None):
    """Deduplicate before submission; report completion order, return original order."""
    positions = defaultdict(list)
    for index, mask in enumerate(masks):
        positions[tuple(int(bit) for bit in mask)].append(index)
    results = [None] * len(masks)
    pending = []

    def finish(mask, metrics, recovered):
        if not recovered:
            save(mask, metrics)
        fit = (metrics[3], -int(metrics[12]))
        for index in positions[mask]:
            results[index] = fit
        completed(mask, fit, len(positions[mask]), recovered)

    for mask in positions:
        result = cached(mask)
        if result is None:
            pending.append(mask)
        else:
            finish(mask, result, True)
    if executor is None:
        for mask in pending:
            finish(mask, evaluate(mask), False)
    else:
        futures = {executor.submit(evaluate, mask): mask for mask in pending}
        try:
            for future in as_completed(futures):
                finish(futures[future], future.result(), False)
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    return results
