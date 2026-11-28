"""Regression coverage for automatically detected source labels and legacy journals."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

import wgangp as wg
from utils.wgangp_generation_resume import GenerationJournal, identity_sha256, file_sha256


class GenerationLabelSchemaTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / 'latam.csv'
        self.features = ['x', 'y_value']
        rng = np.random.default_rng(17)
        self.real = pd.DataFrame({'x': rng.normal(20, 3, 2048),
                                  'y_value': rng.normal(10, 2, 2048),
                                  'Category': np.repeat([0, 1], 1024)})
        self.real.to_csv(self.source, index=False)
        self.config = wg.get_default_config()
        self.config['wgangp'].update(csv_path=str(self.source), mode='gen', force_cpu=True)
        self.config['quality'].update(sample_rows_per_class=128, nearest_reference_rows=128)
        old = wg.CONFIG
        wg.CONFIG = self.config
        self.addCleanup(setattr, wg, 'CONFIG', old)
        self.args = wg.ConfigNamespace(self.config)
        self.args.out_file = str(self.root / 'augmented.csv')
        self.args._generation_checkpoint_digest = 'legacy-checkpoint'
        self.device = torch.device('cpu')
        self.model = torch.nn.Identity()
        self.encoder = LabelEncoder().fit(['0', '1'])
        self.scaler = wg.ClassConditionalFlowScaler(self.features)
        self.synthetic = self.real.rename(columns={'Category': 'Label'}).copy()
        self.synthetic['Label'] = self.synthetic['Label'].astype(str)
        self.synthetic['x'] += rng.normal(0, .01, len(self.synthetic))
        self.synthetic['y_value'] += rng.normal(0, .01, len(self.synthetic))

    def test_training_detected_label_and_generation_schema(self):
        dataset = wg.CSVFlowDataset(str(self.source), label_col='Label')
        self.assertEqual(dataset.label_col, 'Category')
        self.assertEqual(dataset.feature_cols, self.features)
        self.assertEqual(wg.resolve_generation_source_label(self.args, self.features), 'Category')
        self.assertEqual(self.args.label_col, 'Label')

    def test_selection_reservoirs_match_canonical_label_exactly(self):
        actual = wg.build_quality_selection_references(self.args, self.features, ['0', '1'], 512)
        self.assertEqual(set(actual), {'0', '1'})
        canonical = self.root / 'canonical.csv'
        self.real.rename(columns={'Category': 'Label'}).to_csv(canonical, index=False)
        other = copy.copy(self.args)
        other.csv_path = str(canonical)
        expected = wg.build_quality_selection_references(other, self.features, ['0', '1'], 512)
        for name in actual:
            for a, b in zip(actual[name], expected[name]):
                np.testing.assert_array_equal(a, b)

    def test_strict_audit_metrics_and_decisions_match_canonical_label(self):
        actual = wg.assess_generated_flow_quality(self.args, self.config, self.synthetic,
                                                  persist_report=False, enforce=False)
        self.assertEqual({r['class'] for r in actual['classes']}, {'0', '1'})
        canonical = self.root / 'canonical.csv'
        self.real.rename(columns={'Category': 'Label'}).to_csv(canonical, index=False)
        other = copy.copy(self.args)
        other.csv_path = str(canonical)
        expected = wg.assess_generated_flow_quality(other, self.config, self.synthetic,
                                                    persist_report=False, enforce=False)
        self.assertEqual(actual, expected)
        self.assertEqual([r['mean_shift_reference_rows'] for r in actual['classes']], [1024, 1024])

    def test_missing_feature_or_label_fails_before_any_generation(self):
        with self.assertRaisesRegex(ValueError, 'missing features'):
            wg.resolve_generation_source_label(self.args, ['missing'])
        bad = self.root / 'bad.csv'
        self.real.rename(columns={'Category': 'unknown'}).to_csv(bad, index=False)
        self.args.csv_path = str(bad)
        with patch.object(wg, 'decode_candidate_frame') as draw:
            with self.assertRaisesRegex(ValueError, 'detected label=None'):
                wg.generate_unique_decoded_samples(self.args, self.config, self.model, self.device,
                    np.array([0, 1], dtype=np.int64), self.scaler, self.encoder, self.features)
            draw.assert_not_called()
        self.assertFalse(Path(self.args.out_file).exists())

    def test_whitespace_and_explicit_label_precedence(self):
        self.real.rename(columns={'Category': ' Category '}).to_csv(self.source, index=False)
        self.assertEqual(wg.resolve_generation_source_label(self.args, self.features), 'Category')
        self.real.assign(Label=1).to_csv(self.source, index=False)
        self.assertEqual(wg.resolve_generation_source_label(self.args, self.features), 'Label')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            wg.resolve_generation_source_label(self.args, self.features + ['Label'])

    def test_legacy_first_unique_journal_resumes_with_real_selection(self):
        requested = np.repeat([0, 1], 64).astype(np.int64)
        initial = pd.concat([self.synthetic.iloc[:64], self.synthetic.iloc[1024:1088]], ignore_index=True)
        initial = initial.astype({feature: np.float32 for feature in self.features})
        identity = wg.generation_journal_identity(self.args, self.model, self.scaler, requested, self.features, self.device)
        self.assertEqual(identity['release'], 'V6.5.31')
        self.assertEqual(identity['label_col'], 'Label')
        journal = GenerationJournal(self.args.out_file, identity)
        columns = self.features + ['Label']
        rows, hashes = journal.select(initial, wg.hash_canonical_decoded_rows(initial, columns))
        state = {'cursor': len(requested), 'counts': {0: 64, 1: 64}, 'draws': len(requested),
                 'duplicates': 0, 'rounds': 0, 'elapsed': 1.0, 'rng': wg.generation_rng_state(self.device)}
        journal.commit(rows, hashes, state)
        self.config['generation'].update(quality_selection_extra_fraction=.25, quality_selection_reference_rows_per_class=512)
        def extra(_args, _model, _device, labels, _scaler, encoder, _features):
            # Extra synthetic candidates only; the completed first population must not be redrawn.
            self.assertLess(len(labels), len(requested))
            return pd.DataFrame({'x': np.arange(len(labels), dtype=np.float32) + 100,
                                 'y_value': np.arange(len(labels), dtype=np.float32) + 200,
                                 'Label': encoder.inverse_transform(labels)})
        with patch.object(wg, 'decode_candidate_frame', side_effect=extra) as draw:
            result, _, _ = wg.generate_unique_decoded_samples(self.args, self.config, self.model, self.device,
                requested, self.scaler, self.encoder, self.features)
        self.assertGreater(draw.call_count, 0)
        self.assertEqual(result.Label.value_counts().to_dict(), {'0': 64, '1': 64})
        self.assertFalse(result.duplicated().any())
        self.assertEqual(self.args._unique_generation_journal_dir, journal.directory)
        self.assertEqual(journal.identity, identity_sha256(identity))
        self.assertEqual(set(self.args._last_generation_selection['classes']), {'0', '1'})

    def test_legacy_checkpoint_load_preserves_weights_and_provenance(self):
        dataset = wg.CSVFlowDataset(str(self.source), label_col='Label')
        self.args.batch_size = 256
        self.args.latent_dim = 8
        self.args.g_hidden = [16]
        self.args.embed_dim = 4
        self.args.n_resblocks = 1
        wg.set_seed(42)
        width = dataset.scaler.model_feature_dim
        assert width is not None
        model = wg.Generator(8, int(width), 2, [16], 4, 1)
        optimizer = torch.optim.Adam(model.parameters())
        payload = wg.build_generator_checkpoint_payload(model, optimizer, None, dataset, 59, {}, self.args)
        self.assertEqual(payload['source_label_col'], 'Category')
        payload.pop('source_label_col')  # The V6.5.49 checkpoint did not save this field.
        checkpoint = self.root / 'latam_generator_epoch60.pt'
        torch.save(payload, checkpoint)
        digest = file_sha256(checkpoint)
        fresh = wg.ConfigNamespace(copy.deepcopy(self.config))
        fresh.checkpoint, fresh.out_file = str(checkpoint), self.args.out_file
        config = copy.deepcopy(self.config)
        loaded, _, scaler, encoder, features, distribution, device, _ = wg.normalize_args_and_load_checkpoint(fresh, config)
        self.assertEqual(loaded['epoch'], 60)
        self.assertEqual(fresh.label_col, 'Label')
        self.assertEqual(wg.resolve_generation_source_label(fresh, features), 'Category')
        self.assertEqual(wg.build_wgangp_provenance(fresh), payload['provenance'])
        self.assertEqual(distribution, {0: 1024, 1: 1024})
        fresh._mixed_categorical_groups = list(scaler.categorical_groups)
        fresh._mixed_categorical_class_masks = scaler.categorical_class_masks
        restored = wg.build_and_load_generator(fresh, config, loaded, device, int(scaler.model_feature_dim), len(encoder.classes_))
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
        self.assertEqual(file_sha256(checkpoint), digest)

    def test_published_csv_uses_source_label_and_preserves_internal_population(self):
        # Real strict audit (no mocked acceptance) and atomic CSV publication.
        before = self.synthetic.copy(deep=True)
        source_digest = file_sha256(self.source)
        provenance = wg.build_wgangp_provenance(self.args)
        self.args._last_original_preprocessed_num_samples = len(self.real)
        config = self.config
        config['hardware_tracking'] = False
        output = wg.postprocess_generated_arrays_to_dataframe(self.args, config, self.synthetic,
            [], self.scaler, self.encoder, self.features, self.device, len(self.synthetic), '')
        pd.testing.assert_frame_equal(output, before)
        published = pd.read_csv(self.args.out_file)
        self.assertEqual(list(published.columns), self.features + ['Category'])
        self.assertEqual(published.Category.value_counts().to_dict(), {0: 1024, 1: 1024})
        np.testing.assert_allclose(published[self.features], before[self.features], rtol=1e-14, atol=1e-14)
        manifest = json.loads(Path(self.args.out_file).with_suffix('.provenance.json').read_text())
        self.assertEqual(manifest['sha256'], provenance['sha256'])
        self.assertEqual(manifest['generation_output_label_col'], 'Category')
        quality = json.loads(Path(self.args.out_file).with_suffix('.quality.json').read_text())
        self.assertTrue(quality['accepted'])
        self.assertEqual(file_sha256(self.source), source_digest)

    def test_current_quality_report_and_source_label_required_for_output_reuse(self):
        self.args.n_samples = float(len(self.synthetic))
        path = Path(self.args.out_file)
        self.synthetic.rename(columns={'Label': 'Category'}).to_csv(path, index=False)
        manifest = wg.build_wgangp_provenance(self.args)
        manifest['generation_uniqueness'] = {'policy': 'exact_decoded_numeric_rows_v1', 'verified': True}
        manifest['generation_selection'] = {'settings_sha256': wg.build_generation_selection_fingerprint(self.config)}
        path.with_suffix('.provenance.json').write_text(json.dumps(manifest), encoding='utf-8')
        report = {'schema_version': wg.QUALITY_REPORT_SCHEMA_VERSION, 'accepted': True,
                  'quality_policy_sha256': wg.build_quality_policy_fingerprint(self.config)}
        report_path = path.with_suffix('.quality.json')
        report_path.write_text(json.dumps(report), encoding='utf-8')
        self.assertFalse(wg.verify_data_augmentation_file(self.args, self.config))
        self.synthetic.to_csv(path, index=False)  # Incorrect output label must never be reused.
        self.assertTrue(wg.verify_data_augmentation_file(self.args, self.config))
        self.synthetic.rename(columns={'Label': 'Category'}).to_csv(path, index=False)
        report['accepted'] = False
        report_path.write_text(json.dumps(report), encoding='utf-8')
        self.assertTrue(wg.verify_data_augmentation_file(self.args, self.config))
        report['accepted'], report['schema_version'] = True, 4
        report_path.write_text(json.dumps(report), encoding='utf-8')
        self.assertTrue(wg.verify_data_augmentation_file(self.args, self.config))

    def test_tsne_receives_actual_classes(self):
        with patch.object(wg, 'generate_tsne_3d_separability_plot') as plot:
            wg.generate_generation_quality_outputs(self.args, self.config, self.synthetic, self.features)
        plot.assert_called_once()
        call = plot.call_args
        assert call is not None
        self.assertEqual(set(call.kwargs['original_labels']), {'0', '1'})
        self.assertEqual(set(call.kwargs['generated_labels']), {'0', '1'})


if __name__ == '__main__':
    unittest.main()
