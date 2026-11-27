"""Regression checks for lossless disk-backed training storage (no large dataset)."""
import tempfile
import unittest
from pathlib import Path
import zipfile
from unittest import mock
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from utils.stacking.disk_arrays import DiskSyntheticSource, allocate_matrix, copy_matrix, close_matrix, merge_scaled_training
from utils.stacking.disk_forest import DiskForestTrees, append_tree, load_classifier_bundle, write_disk_forest_bundle


class DiskMemoryTests(unittest.TestCase):
    def test_candidate_pool_and_chunked_scaling_match_eager(self):
        rng = np.random.RandomState(42)
        frame = pd.DataFrame({'b': rng.normal(size=203), 'a': rng.randint(0, 100, 203), 'label': ['a', 'b', 'c'] * 67 + ['a', 'b']})
        columns = ['a', 'b']
        original = rng.normal(size=(151, 2))
        scaler = StandardScaler().fit(original)
        scaled = scaler.transform(original)
        positions = rng.choice(len(frame), 77, replace=False)
        expected = np.concatenate((scaled, scaler.transform(frame.iloc[positions][columns].to_numpy(copy=False))))
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / 'pool'; directory.mkdir()
            source = DiskSyntheticSource(frame, columns, directory, chunk_rows=13)
            try:
                np.testing.assert_array_equal(source.labels, frame.label.to_numpy())
                np.testing.assert_array_equal(source.matrix, frame[columns].to_numpy())
                merged = merge_scaled_training(scaled, source.matrix, positions, scaler, Path(tmp) / 'merged.dat', chunk_rows=17)
                try:
                    np.testing.assert_array_equal(merged, expected)
                    self.assertEqual(merged.dtype, expected.dtype)
                    cast = copy_matrix(merged, Path(tmp) / 'cast.dat', dtype=np.float32, chunk_rows=11)
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
            path = Path(tmp) / 'existing.dat'; path.write_bytes(b'preserve')
            with self.assertRaises(FileExistsError):
                allocate_matrix(path, (2, 3), np.float64)
            self.assertEqual(path.read_bytes(), b'preserve')

    def test_forest_seeds_predictions_and_artifact_roundtrip(self):
        rng = np.random.RandomState(8)
        X = rng.normal(size=(600, 7)).astype(np.float32)
        y = rng.randint(0, 3, size=len(X)); weights = rng.uniform(.2, 2, len(X))
        for seed, weight_mode in [(42, None), (43, 'balanced_subsample')]:
            with self.subTest(seed=seed, weight_mode=weight_mode), tempfile.TemporaryDirectory() as tmp:
                params = dict(n_estimators=9, n_jobs=1, random_state=seed, class_weight=weight_mode)
                reference = RandomForestClassifier(**params).fit(X, y, sample_weight=weights)
                actual = RandomForestClassifier(**params)
                owner = Path(tmp) / 'training'; owner.mkdir(); path = owner / 'trees.zip'
                tree_rng = np.random.RandomState(seed)
                with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as archive:
                    for i in range(params['n_estimators']):
                        actual.set_params(n_estimators=1, warm_start=False, random_state=tree_rng)
                        actual.fit(X, y, sample_weight=weights)
                        self.assertEqual(actual.estimators_[0].random_state, reference.estimators_[i].random_state)
                        append_tree(archive, i, actual.estimators_[0]); actual.estimators_.clear()
                actual.set_params(**params)
                actual.estimators_ = DiskForestTrees(path, params['n_estimators'], owner)
                try:
                    self.assertEqual(actual.get_params(), reference.get_params())
                    for a,b in zip(actual.estimators_, reference.estimators_):
                        np.testing.assert_array_equal(a.tree_.value, b.tree_.value)
                        for field in a.tree_.__getstate__()['nodes'].dtype.names:
                            np.testing.assert_array_equal(a.tree_.__getstate__()['nodes'][field], b.tree_.__getstate__()['nodes'][field])
                    np.testing.assert_array_equal(actual.predict_proba(X), reference.predict_proba(X))
                    np.testing.assert_array_equal(actual.predict(X), reference.predict(X))
                    artifact = Path(tmp) / 'new.joblib'
                    write_disk_forest_bundle({'model': actual, 'identity': 'test'}, artifact)
                finally:
                    actual.estimators_.close()
                self.assertFalse(owner.exists())
                with self.assertRaises(Exception):
                    load(artifact)
                restored = load_classifier_bundle(artifact)
                try:
                    np.testing.assert_array_equal(restored['model'].predict_proba(X), reference.predict_proba(X))
                    self.assertEqual(restored['identity'], 'test')
                finally:
                    restored['model'].estimators_.close()
                self.assertTrue(artifact.is_file())
                legacy = Path(tmp) / 'legacy.joblib'; dump({'model': reference}, legacy, compress=3)
                np.testing.assert_array_equal(load_classifier_bundle(legacy)['model'].predict(X), reference.predict(X))

    def test_missing_tree_archive_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'empty.zip'
            with zipfile.ZipFile(path, 'w'): pass
            with self.assertRaises(ValueError): DiskForestTrees(path, 1)


class ProductionIntegrationTests(unittest.TestCase):
    def setUp(self):
        try:
            import stacking
        except ModuleNotFoundError as exc:
            # Run the independent helper tests in lightweight environments; the
            # full dependency environment must execute these integration tests.
            if exc.name in {'arff', 'dataframe_image', 'lime', 'colorama', 'tensorflow', 'shap', 'pytorch_tabnet', 'optuna', 'lightgbm', 'xgboost', 'torch', 'torchvision', 'imblearn', 'telegram', 'flaml', 'pyarrow'}:
                self.skipTest(f'Optional project dependency unavailable: {exc.name}')
            raise
        self.stacking = stacking

    def test_real_progress_fit_and_cancellation(self):
        s = self.stacking
        rng = np.random.RandomState(7)
        X = rng.normal(size=(301, 5)).astype(np.float32)
        y = rng.randint(0, 3, len(X))
        params = dict(n_estimators=7, n_jobs=1, random_state=42)
        config = {'stacking': {'memory_management': {'disk_forest_min_train_nbytes': 0}}}
        reference = RandomForestClassifier(**params).fit(X, y)
        with tempfile.TemporaryDirectory() as root, mock.patch.object(s, 'create_feature_process_temp_directory', side_effect=lambda *a, **k: tempfile.mkdtemp(dir=root)), mock.patch.object(s, 'write_memory_phase_event'), mock.patch.object(s, 'build_training_progress'):
            actual = RandomForestClassifier(**params)
            s.fit_classifier_with_progress(actual, X, y, 'Full Features', 'Random Forest', config=config, augmentation_ratio=.25, forest_dataset_file='fixture')
            try:
                np.testing.assert_array_equal(actual.predict_proba(X), reference.predict_proba(X))
                np.testing.assert_array_equal(actual.predict(X), reference.predict(X))
                self.assertEqual(actual.get_params(), reference.get_params())
            finally:
                actual.estimators_.close()
            self.assertEqual(list(Path(root).iterdir()), [])
            failed = RandomForestClassifier(**params)
            calls = iter([False, True])
            with self.assertRaises(s.RuntimeSkipRequested):
                s.fit_classifier_with_progress(failed, X, y, 'Full Features', 'Random Forest', config=config, augmentation_ratio=.25, forest_dataset_file='fixture', cancellation_checker=lambda: next(calls))
            self.assertEqual(list(Path(root).iterdir()), [])
            self.assertEqual(failed.get_params(), reference.get_params())

    def test_disk_forest_is_not_forced_for_incompatible_estimators(self):
        s = self.stacking
        rng = np.random.RandomState(17)
        X = rng.normal(size=(120, 4)).astype(np.float32)
        y = rng.randint(0, 2, len(X))
        config = {'stacking': {'memory_management': {'disk_forest_min_train_nbytes': 0}}}
        incompatible = (
            RandomForestClassifier(n_estimators=3, n_jobs=2, random_state=42),
            RandomForestClassifier(n_estimators=3, n_jobs=1, random_state=None),
            RandomForestClassifier(n_estimators=3, n_jobs=1, random_state=42, oob_score=True),
            RandomForestClassifier(n_estimators=3, n_jobs=1, random_state=42, warm_start=True),
        )
        for model in incompatible:
            with self.subTest(params=model.get_params()), mock.patch.object(
                s, 'create_feature_process_temp_directory'
            ) as create_directory, mock.patch.object(s, 'write_memory_phase_event'), mock.patch.object(
                s, 'build_training_progress'
            ):
                s.fit_classifier_with_progress(
                    model,
                    X,
                    y,
                    'Full Features',
                    'Random Forest',
                    config=config,
                    augmentation_ratio=.25,
                    forest_dataset_file='fixture',
                )
                create_directory.assert_not_called()
                self.assertNotIsInstance(model.estimators_, DiskForestTrees)

    def test_real_split_scaler_matches_original_eager_operations(self):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        s = self.stacking
        rng = np.random.RandomState(21)
        labels = pd.Series(['a', 'b', 'c'] * 100)
        train, test = train_test_split(np.arange(len(labels)), test_size=.2, random_state=42, stratify=labels.to_numpy())
        metadata = pd.DataFrame({'source_file': ['fixture'] * len(labels), 'row_order': np.arange(len(labels))})
        for dtype in (np.float32, np.float64, np.int64):
            values = (rng.normal(size=(len(labels), 5)) * 100).astype(dtype)
            scaler = StandardScaler()
            expected_train = scaler.fit_transform(values[train])
            expected_test = scaler.transform(values[test])
            encoder = LabelEncoder().fit(labels.to_numpy()[train])
            for meta in (None, metadata):
                with mock.patch.object(s, 'write_memory_phase_event'), mock.patch.object(s, 'verbose_output'):
                    result = s.scale_and_split(pd.DataFrame(values), labels, config={'evaluation': {'random_state': 42}}, row_metadata=meta)
                np.testing.assert_array_equal(result[0], expected_train)
                np.testing.assert_array_equal(result[1], expected_test)
                np.testing.assert_array_equal(result[2], encoder.transform(labels.to_numpy()[train]))
                np.testing.assert_array_equal(result[3], encoder.transform(labels.to_numpy()[test]))
                for attr in ('mean_', 'var_', 'scale_'):
                    np.testing.assert_array_equal(getattr(result[4], attr), getattr(scaler, attr))
                if meta is not None:
                    pd.testing.assert_frame_equal(result[6], metadata.iloc[train].reset_index(drop=True))
                    pd.testing.assert_frame_equal(result[7], metadata.iloc[test].reset_index(drop=True))


if __name__ == '__main__':
    unittest.main()
