"""Lossless disk-backed storage for large sequential RandomForestClassifier fits.

The estimator is still sklearn's RandomForestClassifier. Only the storage of
completed trees changes. Each tree is serialized independently, and prediction
uses sklearn's ordinary ordered probability averaging with n_jobs=1.
"""
from collections.abc import Sequence
import copy
import errno
from pathlib import Path
import shutil
import zipfile

from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier


_BUNDLE = "forest-bundle.joblib"


def _tree_name(index):
    return f"trees/{index:08d}.joblib"


class DiskForestTrees(Sequence):
    """Read a single fitted tree at a time from a ZIP64 archive.

    A temporary training archive is owned by this sequence; a published
    archive is not. Hold an open file handle so atomic replacement of the
    pathname cannot change the trees of an already loaded model on Linux.
    """

    def __init__(self, path, count, owned_directory=None):
        self.path = Path(path)
        self.count = int(count)
        self.owned_directory = owned_directory
        self._archive = zipfile.ZipFile(self.path, "r")
        try:
            expected = {_tree_name(i) for i in range(self.count)}
            members = self._archive.namelist()
            if len(members) != len(set(members)) or set(members) not in (expected, expected | {_BUNDLE}):
                raise ValueError("Incomplete or invalid disk forest archive")
        except BaseException:
            self.close()
            raise

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(self.count))]
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        with self._archive.open(_tree_name(index)) as stream:
            return load(stream)

    def close(self):
        archive = getattr(self, "_archive", None)
        if archive is not None:
            archive.close()
            self._archive = None
        directory = getattr(self, "owned_directory", None)
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)
            self.owned_directory = None

    def __del__(self):
        self.close()


def append_tree(archive, index, tree):
    """Write a lossless fitted tree without serializing the whole forest."""
    # Reserve conservative headroom before writing another tree. Compression may
    # use less space; never silently truncate or drop a tree on ENOSPC.
    state = tree.tree_.__getstate__()
    raw_size = state["nodes"].nbytes + state["values"].nbytes
    if shutil.disk_usage(Path(archive.filename).parent).free < raw_size + 1048576:
        raise OSError(errno.ENOSPC, "Insufficient storage for another forest tree")
    with archive.open(_tree_name(index), "w", force_zip64=True) as stream:
        dump(tree, stream, compress=0)


def is_disk_forest(model):
    return isinstance(getattr(model, "estimators_", None), DiskForestTrees)


def write_disk_forest_bundle(bundle, destination):
    """Publish a self-contained ZIP64 bundle with the original estimator metadata.

    The caller owns atomic publication and SHA-256 verification. No temporary
    path is serialized. The existing joblib loader is used for legacy bundles.
    """
    forest = bundle["model"]
    trees = forest.estimators_
    # Copy the archive as a stream; never deserialize its completed trees.
    # For a fitted temporary forest the source cannot be replaced concurrently.
    if shutil.disk_usage(Path(destination).parent).free < trees.path.stat().st_size + 1048576:
        raise OSError(errno.ENOSPC, "Insufficient storage for atomic forest publication")
    shutil.copyfile(trees.path, destination)
    header = dict(bundle)
    # Do not invoke sklearn's pickle/deepcopy machinery on a lazy sequence.
    model = copy.copy(forest)
    model.estimators_ = []
    header["model"] = model
    with zipfile.ZipFile(destination, "a", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
        if _BUNDLE in archive.namelist():
            raise ValueError("Disk forest export requires an unpublished tree archive")
        with archive.open(_BUNDLE, "w", force_zip64=True) as stream:
            dump(header, stream, compress=0)


def load_classifier_bundle(path, legacy_loader=load):
    """Load legacy joblib artifacts or the self-contained disk forest format."""
    if not zipfile.is_zipfile(path):
        return legacy_loader(path)
    with zipfile.ZipFile(path, "r") as archive:
        with archive.open(_BUNDLE) as stream:
            bundle = load(stream)
    model = bundle.get("model") if isinstance(bundle, dict) else None
    if type(model) is not RandomForestClassifier or model.estimators_ != [] or model.n_jobs != 1:
        raise ValueError("Invalid disk forest estimator header")
    model.estimators_ = DiskForestTrees(path, model.n_estimators)
    return bundle
