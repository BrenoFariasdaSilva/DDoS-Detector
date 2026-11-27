"""Regression checks for lossless dataframe and split memory lifecycle changes."""

import unittest
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import stacking


class StackingMemoryLifecycleTests(unittest.TestCase):
    def test_split_scaler_labels_and_metadata_match_eager_operations(self):
        rng = np.random.RandomState(21)
        labels = pd.Series(["a", "b", "c"] * 100)
        train, test = train_test_split(
            np.arange(len(labels)),
            test_size=0.2,
            random_state=42,
            stratify=labels.to_numpy(),
        )
        metadata = pd.DataFrame(
            {"source_file": ["fixture"] * len(labels), "row_order": np.arange(len(labels))}
        )

        for dtype in (np.float32, np.float64, np.int64):
            values = (rng.normal(size=(len(labels), 5)) * 100).astype(dtype)
            reference_scaler = StandardScaler()
            expected_train = reference_scaler.fit_transform(values[train])
            expected_test = reference_scaler.transform(values[test])
            encoder = LabelEncoder().fit(labels.to_numpy()[train])

            for row_metadata in (None, metadata):
                with self.subTest(dtype=dtype, metadata=row_metadata is not None), mock.patch.object(
                    stacking, "write_memory_phase_event"
                ), mock.patch.object(stacking, "verbose_output"):
                    result = stacking.scale_and_split(
                        pd.DataFrame(values),
                        labels,
                        config={"evaluation": {"random_state": 42}},
                        row_metadata=row_metadata,
                    )

                np.testing.assert_array_equal(result[0], expected_train)
                np.testing.assert_array_equal(result[1], expected_test)
                np.testing.assert_array_equal(result[2], encoder.transform(labels.to_numpy()[train]))
                np.testing.assert_array_equal(result[3], encoder.transform(labels.to_numpy()[test]))
                for attribute in ("mean_", "var_", "scale_"):
                    np.testing.assert_array_equal(
                        getattr(result[4], attribute), getattr(reference_scaler, attribute)
                    )
                if row_metadata is not None:
                    pd.testing.assert_frame_equal(result[6], metadata.iloc[train].reset_index(drop=True))
                    pd.testing.assert_frame_equal(result[7], metadata.iloc[test].reset_index(drop=True))

    def test_prepare_splits_preserves_feature_order_and_optional_metadata(self):
        frame = pd.DataFrame(
            {
                "z": np.arange(30, dtype=np.int64),
                stacking.LSTM_SOURCE_FILE_COLUMN: ["fixture"] * 30,
                "ignored": ["text"] * 30,
                "a": np.linspace(0, 1, 30),
                stacking.LSTM_ROW_ORDER_COLUMN: np.arange(30, dtype=np.int64),
                "label": ["a", "b", "c"] * 10,
            }
        )
        with mock.patch.object(stacking, "write_memory_phase_event"), mock.patch.object(
            stacking, "verbose_output"
        ):
            without_metadata = stacking.prepare_evaluation_data_splits(
                frame.copy(), config={"evaluation": {"random_state": 42}}, include_lstm_metadata=False
            )
            with_metadata = stacking.prepare_evaluation_data_splits(
                frame.copy(), config={"evaluation": {"random_state": 42}}, include_lstm_metadata=True
            )

        self.assertEqual(len(without_metadata), 6)
        self.assertEqual(len(with_metadata), 8)
        self.assertEqual(without_metadata[0].shape[1], 2)
        np.testing.assert_array_equal(without_metadata[0], with_metadata[0])
        np.testing.assert_array_equal(without_metadata[1], with_metadata[1])
        np.testing.assert_array_equal(without_metadata[2], with_metadata[2])
        np.testing.assert_array_equal(without_metadata[3], with_metadata[3])


if __name__ == "__main__":
    unittest.main()
