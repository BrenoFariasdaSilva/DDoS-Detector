"""Regression checks for bounded, lossless disk-backed training matrices."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils.stacking.disk_arrays import (
    DiskSyntheticSource,
    allocate_matrix,
    close_matrix,
    copy_matrix,
    merge_scaled_training,
)


class DiskArrayTests(unittest.TestCase):
    def test_candidate_pool_selection_order_and_chunked_scaling_match_eager(self):
        rng = np.random.RandomState(42)
        frame = pd.DataFrame(
            {
                "b": rng.normal(size=203),
                "a": rng.randint(0, 100, 203),
                "label": ["a", "b", "c"] * 67 + ["a", "b"],
            }
        )
        columns = ["a", "b"]
        original = rng.normal(size=(151, 2))
        scaler = StandardScaler().fit(original)
        scaled = scaler.transform(original)
        positions = rng.choice(len(frame), 77, replace=False)
        expected = np.concatenate(
            (scaled, scaler.transform(frame.iloc[positions][columns].to_numpy(copy=False)))
        )

        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "pool"
            directory.mkdir()
            source = DiskSyntheticSource(frame, columns, directory, chunk_rows=13)
            try:
                np.testing.assert_array_equal(source.labels, frame.label.to_numpy())
                np.testing.assert_array_equal(source.matrix, frame[columns].to_numpy())
                merged = merge_scaled_training(
                    scaled,
                    source.matrix,
                    positions,
                    scaler,
                    Path(tmp) / "merged.dat",
                    chunk_rows=17,
                )
                try:
                    np.testing.assert_array_equal(merged, expected)
                    self.assertEqual(merged.dtype, expected.dtype)
                    cast = copy_matrix(
                        merged, Path(tmp) / "cast.dat", dtype=np.float32, chunk_rows=11
                    )
                    try:
                        np.testing.assert_array_equal(cast, expected.astype(np.float32))
                        self.assertTrue(cast.flags.c_contiguous)
                    finally:
                        close_matrix(cast)
                finally:
                    close_matrix(merged)
            finally:
                source.close()
            self.assertFalse(directory.exists())

    def test_existing_file_is_not_removed_on_allocation_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "existing.dat"
            path.write_bytes(b"preserve")
            with self.assertRaises(FileExistsError):
                allocate_matrix(path, (2, 3), np.float64)
            self.assertEqual(path.read_bytes(), b"preserve")

    def test_partial_owned_file_is_removed_when_preallocation_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "owned.dat"
            with mock.patch(
                "utils.stacking.disk_arrays.os.posix_fallocate", side_effect=OSError("full"), create=True
            ):
                with self.assertRaises(OSError):
                    allocate_matrix(path, (2, 3), np.float64)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
