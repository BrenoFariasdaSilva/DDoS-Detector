"""Regression coverage for completed-model retention and offline pruning.

Run from the repository root using its pinned Python environment:
    python -m unittest discover -s tests -p test_stacking_model_retention.py -v
"""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

import stacking


class RetentionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.config = stacking.get_default_config()
        self.config['explainability']['enabled'] = False
        self.row = {key: 0.75 for key in ('accuracy', 'weighted_precision', 'weighted_recall', 'weighted_f1_score', 'macro_f1_score', 'fpr', 'fnr')}
        self.row.update(experiment_mode='original_training_augmented_training', experiment_run=1, model_name='Random Forest', feature_set='Full Features', hyperparameter_mode='Default Hyperparameters', per_class_f1_scores='[{"class_label":"A","f1_score":0.75}]')

    def test_complete_augmented_model_is_disposable(self):
        self.assertFalse(stacking.should_export_completed_classifier(self.config, self.row, 'complete-prefix'))

    def test_original_model_is_disposable_only_without_consumers(self):
        self.row['experiment_mode'] = 'original_only'
        self.assertFalse(stacking.should_export_completed_classifier(self.config, self.row, 'complete-prefix'))
        self.config['explainability']['enabled'] = True
        self.assertTrue(stacking.should_export_completed_classifier(self.config, self.row, 'complete-prefix'))

    def test_augmented_explainability_does_not_create_a_consumer(self):
        self.config['explainability']['enabled'] = True
        self.assertFalse(stacking.should_export_completed_classifier(self.config, self.row, 'complete-prefix'))

    def test_each_explicit_consumer_overrides_discard(self):
        for section, field in [('stacking', 'keep_trained_models'), ('stacking', 'model_recovery'), ('execution', 'skip_train_if_model_exists')]:
            with self.subTest(field=field):
                config = copy.deepcopy(self.config)
                config[section][field] = True
                self.assertTrue(stacking.should_export_completed_classifier(config, self.row, 'complete-prefix'))
        self.assertTrue(stacking.should_export_completed_classifier(self.config, self.row, 'complete-prefix', requires_model_artifact=True))

    def test_unknown_and_legacy_modes_fail_closed(self):
        for mode in ('legacy_original_training_augmented_testing', 'original_training_augmented_testing', '', 'automl'):
            with self.subTest(mode=mode):
                self.row['experiment_mode'] = mode
                self.assertTrue(stacking.should_export_completed_classifier(self.config, self.row, 'complete-prefix'))

    def test_incomplete_evidence_fails_closed(self):
        self.assertTrue(stacking.should_export_completed_classifier(self.config, self.row, None))
        for field, value in [('macro_f1_score', None), ('accuracy', float('nan')), ('fnr', 1.1), ('per_class_f1_scores', '[]'), ('per_class_f1_scores', 'bad'), ('per_class_f1_scores', '[{"class_label":"A","f1_score":NaN}]')]:
            with self.subTest(field=field, value=value):
                row = dict(self.row, **{field: value})
                self.assertTrue(stacking.should_export_completed_classifier(self.config, row, 'complete-prefix'))

    def test_cli_requires_explicit_dataset_for_prune(self):
        with mock.patch('sys.argv', ['stacking.py', '--prune-completed-models', 'preview']):
            with self.assertRaises(SystemExit) as result:
                stacking.parse_cli_args()
            self.assertEqual(result.exception.code, 2)

    def test_cli_retention_flags_and_config(self):
        for flag, expected in [('--keep-trained-models', True), ('--discard-completed-models', False)]:
            with self.subTest(flag=flag), mock.patch('sys.argv', ['stacking.py', flag]):
                args = stacking.parse_cli_args()
                config = stacking.merge_configs(stacking.get_default_config(), {}, args)
                self.assertIs(config['stacking']['keep_trained_models'], expected)

    def test_session_lock_blocks_cleanup_with_live_reader(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(stacking, '__file__', str(Path(tmp) / 'stacking.py')):
            with stacking.acquire_model_retention_session_lock(exclusive=False):
                with self.assertRaises(BlockingIOError):
                    stacking.acquire_model_retention_session_lock(exclusive=True)
            with stacking.acquire_model_retention_session_lock(exclusive=True):
                with self.assertRaises(BlockingIOError):
                    stacking.acquire_model_retention_session_lock(exclusive=False)


class HistoricalPruningTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'original.csv'
        self.source.write_text('feature_a,feature_b,Label\n0,0,A\n1,1,B\n', encoding='utf-8')
        self.synthetic_source = self.root / 'synthetic.csv'
        self.synthetic_source.write_text('feature_a,feature_b,Label\n0,1,A\n', encoding='utf-8')
        self.config = stacking.get_default_config()
        self.config['explainability']['enabled'] = False
        self.config['telegram']['enabled'] = False
        self.config['logging']['enabled'] = False
        self.config['stacking']['model_recovery'] = False
        self.config['stacking']['keep_trained_models'] = False
        self.X_train = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.], [0.1, 0.1], [0.9, 0.9]])
        self.y_train = np.array([0, 0, 1, 1, 0, 1])
        self.X_test = np.array([[0., 0.], [1., 1.], [0.1, 0.], [1., 0.9]])
        self.y_test = np.array([0, 1, 0, 1])
        self.encoder = LabelEncoder().fit(['A', 'B'])
        self.scaler = StandardScaler().fit(self.X_train)
        self.model = RandomForestClassifier(n_estimators=3, n_jobs=1, random_state=42).fit(self.scaler.transform(self.X_train), self.y_train)
        self.pred = self.model.predict(self.scaler.transform(self.X_test))
        self.feature_names = ['feature_a', 'feature_b']

    def fixture(self, mode='original_only', *, with_matrix=True, worker_format=False):
        ratio = None if mode == 'original_only' else 0.25
        source_label = 'Original' if ratio is None else 'TrainingAugmented@25%'
        slot = 'Full Features - Default Hyperparameters' + ('' if ratio is None else ' - TrainAugmented@25%')
        acc, precision, recall, f1, fpr, fnr, macro, per_class = stacking.compute_classification_metrics(self.y_test, self.pred, self.encoder.classes_)
        metrics = (acc, precision, recall, f1, fpr, fnr, 1.0, self.pred, macro, per_class)
        row = stacking.build_classifier_result_entry(self.model.__class__.__name__, str(self.source), 'separate_files', None, 'Full Features', 'Individual', 'Random Forest', source_label, 'retention-fixture', mode, ratio, 2, len(self.y_train), len(self.y_test), metrics, self.feature_names, effective_hyperparameters=stacking.serialize_effective_estimator_parameters(self.model), experiment_run=1)
        ctx = stacking.build_stacking_model_artifact_context(str(self.source), [str(self.source)], 'separate_files', None, 'Label', 'Random Forest', self.model, slot, self.feature_names, self.feature_names, list(self.encoder.classes_), None, False)
        if worker_format:
            self.assertIsNotNone(ratio)
            stat = self.synthetic_source.stat()
            ctx.update(artifact_type='stacking_augmented_training_classifier', training_data='original_training_plus_synthetic', feature_set='Full Features - Default Hyperparameters', experiment_mode=mode, augmentation_ratio=ratio, synthetic_source_files=[{'path':str(self.synthetic_source), 'size':stat.st_size, 'mtime_ns':stat.st_mtime_ns}], original_split_identity='a'*64)
        cache = {}
        stacking.persist_cache_result_entry(str(self.source), row, cache, config=self.config)
        path = Path(stacking.export_model_and_scaler(self.model, self.scaler, self.root.name, 'Random Forest', slot, str(self.source), self.config, ctx, self.encoder))
        stacking.persist_exported_model_filename(str(self.source), row, cache, str(path), config=self.config)
        if with_matrix:
            # Use the production writer and validation path, including both PNGs.
            stacking.persist_per_run_confusion_matrix_artifacts(row, self.y_test, self.pred, self.encoder.classes_, str(self.source), self.config)
        meta = path.with_name(path.name[:-len('_model.joblib')] + '_meta.json')
        cache_path = Path(stacking.get_cache_file_path(str(self.source), config=self.config))
        return row, path, meta, cache_path

    def prune(self, apply=False):
        return stacking.prune_completed_classifier_models(str(self.source), self.config, apply=apply)

    def test_preview_and_apply_preserve_scientific_results_and_metadata(self):
        row, model, meta, cache = self.fixture()
        backup = Path(str(cache) + '.bak')
        artifacts = [cache, backup] + list(cache.parent.glob('Run-1-Confusion-Matrix/*'))
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts}
        other = model.parent / 'unrelated.joblib'; other.write_bytes(b'keep')
        original_metadata = json.loads(meta.read_text())
        size = model.stat().st_size
        preview = self.prune()
        self.assertEqual(preview['eligible'], 1)
        self.assertEqual(preview['bytes'], size)
        self.assertTrue(model.is_file()); self.assertTrue(meta.is_file())
        self.assertFalse(meta.with_suffix('.retired.json').exists())
        result = self.prune(apply=True)
        self.assertEqual(result['deleted'], 1); self.assertEqual(result['bytes'], size)
        self.assertFalse(model.exists()); self.assertFalse(meta.exists()); self.assertTrue(other.is_file())
        self.assertEqual(json.loads(meta.with_suffix('.retired.json').read_text()), original_metadata)
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts})
        self.assertEqual(self.prune(apply=True)['deleted'], 0)

    def test_sequential_augmented_artifact_is_eligible(self):
        _, model, _, _ = self.fixture(mode='original_training_augmented_training')
        self.assertEqual(self.prune(apply=True)['deleted'], 1)
        self.assertFalse(model.exists())

    def test_worker_augmented_artifact_is_eligible(self):
        _, model, _, _ = self.fixture(mode='original_training_augmented_training', worker_format=True)
        self.assertEqual(self.prune(apply=True)['deleted'], 1)
        self.assertFalse(model.exists())

    def test_duplicate_raw_cache_rows_fail_closed(self):
        _, model, _, cache = self.fixture()
        frame = pd.read_csv(cache)
        pd.concat([frame, frame], ignore_index=True).to_csv(cache, index=False)
        with self.assertRaisesRegex(ValueError, 'duplicate cache rows'):
            self.prune(apply=True)
        self.assertTrue(model.is_file())

    def run_sequential(self, cache, *, keep=False):
        self.config['stacking']['keep_trained_models'] = keep
        self.config['stacking']['memory_management']['disk_forest_min_train_nbytes'] = 0
        self.config['stacking']['memory_management']['spill_min_array_nbytes'] = 0
        train = pd.DataFrame(self.scaler.transform(self.X_train), columns=self.feature_names)
        test = pd.DataFrame(self.scaler.transform(self.X_test), columns=self.feature_names)
        with mock.patch.object(stacking, 'notify_new_best_result_if_applicable'), mock.patch.object(stacking, 'CONFIG', self.config):
            return stacking.run_individual_classifiers_for_feature_set(
                'Full Features', {'Random Forest': RandomForestClassifier(n_estimators=3, n_jobs=1, random_state=42)},
                train, self.y_train, test, self.y_test, test.to_numpy(), 2, str(self.source), 'separate_files', None,
                'TrainingAugmented@25%', 'retention-integration', 'original_training_augmented_training', 0.25, {},
                self.scaler, self.encoder, None, self.feature_names, 'Label', [str(self.source)], self.feature_names,
                1, 1, mock.Mock(), config=self.config, cache_dict=cache, cache_ref_file=str(self.source))

    def test_sequential_disk_forest_discard_and_cache_resume(self):
        cache = {}
        results, _ = self.run_sequential(cache)
        row = results[('Full Features', 'Random Forest')]
        self.assertFalse(row.get('exported_model_filename'))
        self.assertFalse(list(self.root.rglob('*_model.joblib')))
        self.assertFalse(list(self.root.rglob('trees.zip')))
        self.assertTrue(list(self.root.rglob('*raw*.csv')))
        restored = stacking.load_cache_results(str(self.source), config=self.config)
        with mock.patch.object(stacking, 'fit_classifier_with_progress', side_effect=AssertionError('must resume without fitting')):
            recovered, _ = self.run_sequential(restored)
        self.assertEqual(row['weighted_f1_score'], recovered[('Full Features', 'Random Forest')]['weighted_f1_score'])

    def test_sequential_disk_forest_explicit_keep(self):
        results, _ = self.run_sequential({}, keep=True)
        row = results[('Full Features', 'Random Forest')]
        self.assertTrue(row.get('exported_model_filename'))
        self.assertEqual(len(list(self.root.rglob('*_model.joblib'))), 1)
        self.assertFalse(list(self.root.rglob('trees.zip')))

    def test_missing_matrix_retains_model(self):
        _, model, meta, _ = self.fixture(with_matrix=False)
        self.assertEqual(self.prune(apply=True)['deleted'], 0)
        self.assertTrue(model.is_file()); self.assertTrue(meta.is_file())

    def test_missing_backup_retains_model(self):
        _, model, _, cache = self.fixture()
        Path(str(cache) + '.bak').unlink()
        self.assertEqual(self.prune(apply=True)['deleted'], 0)
        self.assertTrue(model.is_file())

    def test_corrupt_model_retains_model(self):
        _, model, meta, _ = self.fixture()
        with model.open('ab') as stream: stream.write(b'corrupt')
        self.assertEqual(self.prune(apply=True)['deleted'], 0)
        self.assertTrue(model.is_file()); self.assertTrue(meta.is_file())

    def test_shared_legacy_reference_retains_model(self):
        row, model, _, _ = self.fixture()
        legacy = dict(row, experiment_mode='original_training_augmented_testing', augmentation_ratio=0.25, data_source='Augmented@25%', experiment_id='legacy-shared', data_augmentation_enabled=True)
        stacking.persist_cache_result_entry(str(self.source), legacy, {}, config=self.config)
        self.assertEqual(self.prune(apply=True)['deleted'], 0)
        self.assertTrue(model.is_file())

    def test_pending_transaction_retains_model(self):
        _, model, _, _ = self.fixture()
        (model.parent / f'.{model.name}.pending.json').write_text('{}')
        self.assertEqual(self.prune(apply=True)['deleted'], 0)
        self.assertTrue(model.is_file())

    def test_existing_consumer_refuses_prune(self):
        _, model, _, _ = self.fixture()
        self.config['stacking']['model_recovery'] = True
        with self.assertRaises(ValueError): self.prune(apply=True)
        self.assertTrue(model.is_file())

    def test_interrupted_retirement_receipt_can_be_retried(self):
        _, model, meta, _ = self.fixture()
        receipt = meta.with_suffix('.retired.json')
        receipt.write_bytes(meta.read_bytes())
        self.assertEqual(self.prune(apply=True)['deleted'], 1)
        self.assertFalse(model.exists()); self.assertTrue(receipt.is_file())

    def test_changed_source_provenance_retains_model(self):
        _, model, _, _ = self.fixture(mode='original_training_augmented_training', worker_format=True)
        self.synthetic_source.write_text('changed')
        self.assertEqual(self.prune(apply=True)['deleted'], 0)
        self.assertTrue(model.is_file())


if __name__ == '__main__':
    unittest.main()
