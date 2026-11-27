"""Private, execution-scoped snapshots for sequential combined evaluations.

Only files created by the current execution are opened. These are not persistent
result caches and must never be used to authorize recovery of classifier metrics.
"""
from pathlib import Path
import shutil

from joblib import dump, load

from utils.stacking.disk_arrays import DiskSyntheticSource


def source_signatures(paths):
    """Record ordered local source identity without rereading dataset contents."""
    signatures = []
    for path in paths:
        resolved = Path(path).resolve(strict=True)
        stat = resolved.stat()
        signatures.append((str(resolved), stat.st_dev, stat.st_ino, stat.st_size,
                           stat.st_mtime_ns, stat.st_ctime_ns))
    return tuple(signatures)


def check_source_signatures(paths, expected):
    if source_signatures(paths) != expected:
        raise RuntimeError("Combined dataset sources changed during this execution; restart with stable inputs")


class CombinedDatasetSession:
    """Own one exact original population and one lazy synthetic candidate pool.

    Numeric blocks in the uncompressed joblib snapshot are reopened with private
    copy-on-write mappings. Pandas dtypes, index, labels and row order are retained.
    The directory is created/tracked/removed by stacking's existing spill owner.
    """
    def __init__(self, frame, paths, directory, expected_signatures=None, distribution=None):
        self.directory = Path(directory)
        self.sample_count = len(frame)
        self.columns = tuple(frame.columns)
        self.numeric_columns = tuple(frame.iloc[:0, :-1].select_dtypes(include='number').columns)
        self.target_column = str(frame.columns[-1])
        self.distribution = distribution
        self.original_path = self.directory / 'original.joblib'
        self.paths = tuple(paths)
        self.signatures = (source_signatures(self.paths) if expected_signatures is None
                           else expected_signatures)
        self.synthetic = None
        self.synthetic_paths = None
        self.synthetic_signatures = None
        self.assert_original_sources()
        # An uncompressed private snapshot preserves the exact pandas representation.
        # Unlike sparse mmap allocation, dump performs ordinary checked file writes.
        if shutil.disk_usage(self.directory).free < int(frame.memory_usage(deep=False).sum()):
            raise OSError('Insufficient disk space for combined original snapshot')
        dump(frame, self.original_path, compress=0)
        self.assert_original_sources()

    def assert_original_sources(self):
        check_source_signatures(self.paths, self.signatures)

    def load_original(self):
        self.assert_original_sources()
        return load(self.original_path, mmap_mode='c')

    def assert_synthetic_sources(self, paths):
        if self.synthetic is not None:
            if tuple(paths) != self.synthetic_paths:
                raise RuntimeError('Synthetic source order changed during combined evaluation')
            check_source_signatures(paths, self.synthetic_signatures)

    def get_synthetic(self, paths, builder, columns, min_bytes, chunk_rows):
        """Build once, including across independent experiment runs."""
        paths = tuple(paths)
        if self.synthetic is not None:
            self.assert_synthetic_sources(paths)
            return self.synthetic
        signatures = source_signatures(paths)
        frame = builder()
        if frame is None:
            return None
        check_source_signatures(paths, signatures)
        if int(frame.memory_usage(deep=False).sum()) >= min_bytes:
            directory = self.directory / 'synthetic'
            directory.mkdir(mode=0o700)
            self.synthetic = DiskSyntheticSource(frame, columns, directory, chunk_rows)
        else:
            self.synthetic = frame
        self.synthetic_paths = paths
        self.synthetic_signatures = signatures
        return self.synthetic

    def close(self):
        if isinstance(self.synthetic, DiskSyntheticSource):
            self.synthetic.close()
        self.synthetic = None


class CombinedRunSplits:
    """Private, run-local split snapshots; never share writable estimator inputs.

    The caller supplies the established split/scale function. Its exact output is
    serialized once. Numeric arrays reopen as private copy-on-write mappings,
    so closing or modifying one group's arrays cannot affect a later group.
    """
    def __init__(self, session, run, seed, numeric_columns, include_metadata):
        self.session = session
        self.run = run
        self.seed = seed
        self.numeric_columns = tuple(numeric_columns)
        self.include_metadata = include_metadata
        self.sample_count = session.sample_count
        self.target_column = session.target_column
        self.directory = session.directory / f'run-{run}'
        self.directory.mkdir(mode=0o700)
        self.base_path = self.directory / 'split.joblib'
        self.metadata_path = self.directory / 'metadata.joblib'
        self.prepared = False
        self.invalid = False

    def get(self, prepare, include_metadata):
        if include_metadata and not self.include_metadata:
            raise RuntimeError('LSTM metadata was not requested by the run plan')
        if self.invalid:
            return None
        if not self.prepared:
            result = prepare([self.session.load_original()], self.include_metadata)
            if result is None:
                self.invalid = True
                return None
            if len(result) != (8 if self.include_metadata else 6):
                raise ValueError('Unexpected original split tuple')
            required = sum(value.nbytes for value in result[:4])
            if self.include_metadata:
                required += sum(int(value.memory_usage(deep=False).sum()) for value in result[6:])
            if shutil.disk_usage(self.directory).free < required:
                raise OSError('Insufficient disk space for run-local original split')
            dump(result[:6], self.base_path, compress=0)
            if self.include_metadata:
                dump(result[6:], self.metadata_path, compress=0)
            del result  # No resident train/test arrays retained by the cache owner.
            self.prepared = True
            print(f'[MEMORY] Run {self.run}: saved original split and fitted scaler (seed={self.seed})')
        else:
            print(f'[MEMORY] Run {self.run}: reusing original split and fitted scaler (seed={self.seed})')
        result = load(self.base_path, mmap_mode='c')
        if include_metadata:
            result += load(self.metadata_path, mmap_mode='c')
        return result

    def close(self):
        shutil.rmtree(self.directory)
