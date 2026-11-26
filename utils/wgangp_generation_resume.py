"""Crash-safe, private first-unique generation state (not a published CSV).

Only trusted local checkpoint/workspace files may be read. Each committed block
contains its rows, canonical hashes, counters and post-block RNG state. Atomic
rename makes a block either entirely committed or entirely absent.
"""
import hashlib
import json
import math
import os
import pickle
import tempfile
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import numpy as np
import pandas as pd


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_sha256(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()


def adaptive_refill_limit(base_rounds, batch_rows, missing, draws, duplicates, rounds, max_draws):
    """Minimum configured rounds plus a size/yield-dependent finite allowance.

    The draw budget is always the hard ceiling; each round consumes >=1 draw.
    A 25% safety margin and four tail rounds avoid a small final shortfall.
    """
    yield_rate = max(0.01, (draws - duplicates) / max(draws, 1))
    needed_draws = math.ceil(missing / yield_rate * 1.25)
    projected = rounds + math.ceil(needed_draws / batch_rows) + 4
    return min(rounds + max(0, max_draws - draws), max(base_rounds, projected))


def recovery_draw_budget(quota, settings):
    """The configured absolute value is a scalable floor, not a hard cap.

    Preserve the existing per-class multiplier as the actual finite budget.
    """
    scaled = math.ceil(quota * settings["draw_multiplier"])
    cap = max(settings["max_draw_rows"], scaled) if settings.get("dynamic_draw_budget", True) else settings["max_draw_rows"]
    return min(cap, scaled)


@contextmanager
def output_lock(path):
    """Nonblocking process lock, automatically released on crash/exit."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.parent / ('.' + path.name + '.generation.lock'), 'a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            if stream.tell() == 0:
                stream.write(b'\0')
                stream.flush()
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError(f'Generation already active for {path}') from exc
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f'Generation already active for {path}') from exc
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def locked_generation(function):
    @wraps(function)
    def wrapped(args, *rest, **kwargs):
        with output_lock(args.out_file):
            if getattr(args, 'only_missing_augmented', False) and Path(args.out_file).exists():
                print(f'[GENERATION] Existing output preserved: {args.out_file}')
                return None
            return function(args, *rest, **kwargs)
    return wrapped


def _sync_directory(path):
    if os.name != 'nt':
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_pickle(path, *objects):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.pending-', delete=False) as stream:
            temporary = Path(stream.name)
            for obj in objects:
                pickle.dump(obj, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class GenerationJournal:
    """Append-only blocks; rebuild only the compact uniqueness index on resume."""
    def __init__(self, output, identity, enabled=True):
        self.enabled = enabled
        self.identity = identity_sha256(identity)
        output = Path(output).resolve()
        self.directory = output.parent / ('.' + output.name + '.generation-resume') / self.identity
        self.seen = set()
        self.blocks = []
        self.state = None
        if not enabled:
            return
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        _sync_directory(self.directory.parent)
        for pending in self.directory.glob('.pending-*'):
            pending.unlink()
        for index, path in enumerate(sorted(self.directory.glob('block-*.pkl'))):
            if path.name != f'block-{index:012d}.pkl':
                raise RuntimeError(f'Incomplete generation journal sequence: {path}')
            with path.open('rb') as stream:
                header = pickle.load(stream)
            if header['identity'] != self.identity or header['index'] != index:
                raise RuntimeError(f'Incompatible generation journal block: {path}')
            signatures = header['signatures']
            previous = len(self.seen)
            self.seen.update(map(int, signatures))
            if len(self.seen) != previous + len(signatures):
                raise RuntimeError(f'Duplicate identities in generation journal: {path}')
            self.state = header['state']
            if sum(self.state['counts'].values()) != len(self.seen):
                raise RuntimeError(f'Invalid generation journal counters: {path}')
            self.blocks.append(path)

    def select(self, frame, signatures):
        selected = []
        for position, signature in enumerate(signatures):
            token = int(signature)
            if token not in self.seen:
                self.seen.add(token)
                selected.append(position)
        return frame.iloc[selected].copy(), signatures[selected]

    def commit(self, frame, signatures, state):
        if self.enabled:
            index = len(self.blocks)
            path = self.directory / f'block-{index:012d}.pkl'
            payload = pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)
            header = {'identity': self.identity, 'index': index, 'signatures': signatures, 'state': state,
                      'payload_sha256': hashlib.sha256(payload).hexdigest()}
            atomic_pickle(path, header, payload)
            self.blocks.append(path)
        elif len(frame):
            self.blocks.append(frame)
        self.state = state

    def materialize(self, columns):
        if not self.enabled:
            return pd.concat(self.blocks, ignore_index=True, copy=False)
        # Allocate the final float32 matrix just once; never hold every batch plus
        # a second concatenated copy in memory. Decoded production data is float32.
        total = sum(self.state['counts'].values())
        values = np.empty((total, len(columns) - 1), dtype=np.float32)
        labels = np.empty(total, dtype=object)
        offset = 0
        for path in self.blocks:
            with path.open('rb') as stream:
                header = pickle.load(stream)
                payload = pickle.load(stream)
            if hashlib.sha256(payload).hexdigest() != header['payload_sha256']:
                raise RuntimeError(f'Corrupt generation journal frame: {path}')
            frame = pickle.loads(payload)
            if list(frame.columns) != columns or len(frame) != len(header['signatures']):
                raise RuntimeError(f'Invalid generation journal frame: {path}')
            if any(dtype != np.dtype(np.float32) for dtype in frame.dtypes.iloc[:-1]):
                raise RuntimeError(f'Unexpected decoded numeric dtype in journal: {path}')
            end = offset + len(frame)
            values[offset:end] = frame.iloc[:, :-1].to_numpy(copy=False)
            labels[offset:end] = frame.iloc[:, -1].to_numpy(copy=False)
            offset = end
        if offset != total:
            raise RuntimeError('Generation journal row count mismatch')
        result = pd.DataFrame(values, columns=columns[:-1])
        result[columns[-1]] = labels
        return result
