"""Regression tests for the V6.5.31 checkpoint-compatible generation recovery."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pandas.io.formats.style  # Load pandas' optional style namespace for wgangp's annotations.
import torch
import yaml
from sklearn.preprocessing import LabelEncoder

import wgangp as wg
from utils.wgangp_generation_resume import GenerationJournal, adaptive_refill_limit, file_sha256, output_lock, recovery_draw_budget


class GenerationRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source.csv'
        self.source.write_text('x,Label\n1,A\n2,B\n')
        self.config = wg.get_default_config()
        self.config['wgangp'].update(mode='gen', csv_path=str(self.source), force_cpu=True)
        self.config['generation'].update(quality_selection_enabled=False, duplicate_refill_batch_rows=7)
        self.old_config = wg.CONFIG
        wg.CONFIG = self.config
        self.addCleanup(setattr, wg, 'CONFIG', self.old_config)
        self.args = wg.ConfigNamespace(self.config)
        self.args.out_file = str(self.root / 'samples.csv')
        self.args._generation_checkpoint_digest = 'test-checkpoint'
        self.encoder = LabelEncoder().fit(['A', 'B'])
        self.scaler = wg.ClassConditionalFlowScaler(['x'])
        self.device = torch.device('cpu')

    def generate(self, labels, decoder, args=None, config=None):
        with patch.object(wg, 'decode_candidate_frame', side_effect=decoder):
            return wg.generate_unique_decoded_samples(args or self.args, config or self.config, None,
                self.device, np.asarray(labels, dtype=np.int64), self.scaler, self.encoder, ['x'])[0]

    def decoder(self, args, model, device, labels, scaler, encoder, features):
        values = torch.randint(0, 500, (len(labels),)).numpy().astype(np.float32)
        return pd.DataFrame({'x': values, 'Label': encoder.inverse_transform(labels)})

    def test_tftp_and_larger_dynamic_budgets(self):
        self.assertGreaterEqual(adaptive_refill_limit(64, 32768, 754592, 17269478, 754592, 0, 86347390), 64)
        self.assertGreater(adaptive_refill_limit(64, 32768, 10_000_000, 100_000_000, 10_000_000, 0, 500_000_000), 64)
        settings = wg.resolve_quality_recovery_settings(self.config)
        self.assertEqual(recovery_draw_budget(17244468, settings), 51733404)
        self.assertEqual(recovery_draw_budget(40_000_000, settings), 120_000_000)
        self.assertEqual(recovery_draw_budget(40_000_000, dict(settings, dynamic_draw_budget=False)), 52_000_000)
        self.assertEqual(adaptive_refill_limit(64, 7, 10, 100, 20, 9, 100), 9)

    def test_generation_budget_does_not_invalidate_checkpoint(self):
        before = wg.build_wgangp_provenance(self.args)
        self.config['generation'].update(max_duplicate_refill_rounds=200, max_duplicate_draw_multiplier=10,
             duplicate_refill_batch_rows=1000, quality_recovery_max_draw_rows_per_class=100_000_000)
        self.assertEqual(before, wg.build_wgangp_provenance(self.args))

    def test_checkpoint_auto_restores_effective_batch_and_rejects_mismatch(self):
        self.args.batch_size = 256
        self.args.use_amp = True
        saved = wg.build_wgangp_provenance(self.args)
        ckpt = {'epoch': 60, 'provenance': saved, 'args': {'epochs': 60, '_last_training_time': 335535}}
        live = wg.ConfigNamespace(self.config)
        wg.restore_generation_training_settings(live, self.config, ckpt)
        wg.verify_wgangp_provenance(saved, live, 'test')
        self.assertEqual(live.batch_size, 256)
        self.assertTrue(live.use_amp)
        wg.restore_generation_training_time(live, self.config, ckpt)
        self.assertEqual(live._last_training_time, 335535)
        live.seed += 1
        with self.assertRaisesRegex(ValueError, 'provenance'):
            wg.verify_wgangp_provenance(saved, live, 'test')
        live.batch_size = 64
        self.config['training']['batch_size_cli_override'] = True
        with self.assertRaisesRegex(ValueError, 'Explicit'):
            wg.restore_generation_training_settings(live, self.config, ckpt)

    def test_custom_model_settings_restore_in_private_config(self):
        self.args.seed = 99
        self.args.latent_dim = 13
        self.args.g_hidden = [12, 24]
        self.config['training']['moment_fidelity_weight'] = 0.25
        self.config['generation']['categorical_temperature'] = 2.0
        provenance = wg.build_wgangp_provenance(self.args)
        ckpt = {'epoch': 60, 'provenance': provenance, 'args': {'epochs': 60}}
        fresh = wg.get_default_config()
        fresh['wgangp'].update(mode='gen', csv_path=str(self.source))
        effective = copy.deepcopy(fresh)
        args = wg.ConfigNamespace(effective)
        wg.restore_generation_training_settings(args, effective, ckpt)
        wg.verify_wgangp_provenance(provenance, args, 'test')
        self.assertEqual(args.seed, 99)
        self.assertEqual(args.latent_dim, 13)
        self.assertEqual(args.g_hidden, [12, 24])
        self.assertEqual(effective['training']['moment_fidelity_weight'], 0.25)
        self.assertEqual(effective['generation']['categorical_temperature'], 2.0)
        self.assertEqual(effective['quality']['max_mean_shift'], fresh['quality']['max_mean_shift'])
        self.assertEqual(fresh['wgangp']['seed'], 42)
        args.csv_path = str(self.root / 'different.csv')
        Path(args.csv_path).write_text('x,Label\n1,A\n2,B\n')
        with self.assertRaisesRegex(ValueError, 'provenance'):
            wg.verify_wgangp_provenance(provenance, args, 'test')
        fresh['wgangp']['generation_explicit_training_fields'] = ['seed']
        with self.assertRaisesRegex(ValueError, 'Explicit generation seed'):
            wg.restore_generation_training_settings(wg.ConfigNamespace(fresh), fresh, ckpt)

    def test_legacy_timing_is_unknown_or_explicitly_supplied(self):
        ckpt = {'epoch': 60, 'args': {'_last_training_time': 0}}
        wg.restore_generation_training_time(self.args, self.config, ckpt)
        self.assertEqual(self.args._last_training_time, '')
        self.config['generation']['historical_training_time_s'] = 335534.02
        wg.restore_generation_training_time(self.args, self.config, ckpt)
        self.assertEqual(self.args._last_training_time, 335535)
        self.config['generation']['historical_training_time_s'] = float('nan')
        with self.assertRaises(ValueError):
            wg.restore_generation_training_time(self.args, self.config, ckpt)

    def test_final_timing_sidecar_binds_exact_checkpoint(self):
        path = self.root / 'source_generator_epoch60.pt'
        path.write_bytes(b'checkpoint')
        self.args._last_saved_generator_path = str(path)
        self.args._last_completed_epoch = 60
        self.args._last_training_time = 335535
        wg.persist_checkpoint_training_duration(self.args)
        ckpt = {'epoch': 60, 'args': {'_last_training_time': 335534}}
        self.assertEqual(wg.checkpoint_training_duration(ckpt, path), 335535)
        self.assertEqual(json.loads(path.with_suffix('.timing.json').read_text())['checkpoint_sha256'], file_sha256(path))
        path.write_bytes(b'other checkpoint')
        self.assertEqual(wg.checkpoint_training_duration(ckpt, path), 335534)

    def test_resume_after_interrupted_refill_matches_uninterrupted_rng(self):
        labels = np.repeat([0, 1], [150, 200])
        wg.set_seed(42)
        reference_args = copy.copy(self.args)
        reference_args.out_file = str(self.root / 'reference.csv')
        reference = self.generate(labels, self.decoder, args=reference_args)
        wg.set_seed(42)
        calls = 0
        def fail(*args):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise RuntimeError('simulated interruption')
            return self.decoder(*args)
        with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
            self.generate(labels, fail)
        journal = self.args._unique_generation_journal_dir
        self.assertEqual(len(list(journal.glob('block-*.pkl'))), 2)
        self.assertFalse(Path(self.args.out_file).exists())
        wg.set_seed(42)
        recovered = self.generate(labels, self.decoder)
        pd.testing.assert_frame_equal(reference, recovered)
        self.assertEqual(recovered.Label.value_counts().to_dict(), {'B': 200, 'A': 150})
        self.assertFalse(recovered.duplicated().any())
        wg.cleanup_unique_generation_journal(self.args)
        self.assertFalse(journal.exists())

    def test_initial_population_is_committed_before_next_large_block(self):
        labels = np.zeros(70000, dtype=np.int64)
        def decode(args, model, device, labels, scaler, encoder, features):
            # Use a continuous deterministic stream to avoid artificial support exhaustion.
            values = torch.arange(len(labels), dtype=torch.float32).numpy() + float(torch.randint(0, 100, (1,))) * 100000
            return pd.DataFrame({'x': values, 'Label': encoder.inverse_transform(labels)})
        wg.set_seed(42)
        calls = 0
        def fail(*args):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('interrupted initial population')
            return decode(*args)
        with self.assertRaisesRegex(RuntimeError, 'interrupted initial'):
            self.generate(labels, fail)
        resumed = GenerationJournal(self.args.out_file, wg.generation_journal_identity(self.args, None, self.scaler, labels, ['x'], self.device))
        self.assertEqual(resumed.state['cursor'], 65536)
        self.assertEqual(resumed.state['counts'][0], 65536)
        del resumed
        wg.set_seed(42)
        result = self.generate(labels, decode)
        self.assertEqual(len(result), 70000)
        self.assertFalse(result.duplicated().any())

    def test_real_torch_checkpoint_and_decoder_smoke(self):
        self.args.batch_size = 256
        self.args.gen_batch_size = 16
        self.args.latent_dim = 8
        self.args.g_hidden = [16]
        self.args.embed_dim = 4
        self.args.n_resblocks = 1
        labels = np.repeat([0, 1], 100)
        source = np.linspace(1, 200, len(labels), dtype=np.float32).reshape(-1, 1)
        self.scaler.fit_transform(source, labels)
        wg.set_seed(42)
        generator = wg.Generator(latent_dim=8, feature_dim=self.scaler.model_feature_dim, n_classes=2,
                                 hidden_dims=[16], embed_dim=4, n_resblocks=1)
        path = self.root / 'source_generator_epoch60.pt'
        torch.save({'epoch': 60, 'state_dict': generator.state_dict(), 'scaler': self.scaler,
                    'label_encoder': self.encoder, 'feature_cols': ['x'], 'class_distribution': {0: 100, 1: 100},
                    'args': {'epochs': 60, '_last_training_time': 335535},
                    'provenance': wg.build_wgangp_provenance(self.args)}, path)
        self.args = wg.ConfigNamespace(self.config)
        self.args.checkpoint = str(path)
        self.args.out_file = str(self.root / 'samples.csv')
        self.args.gen_batch_size = 16
        self.assertEqual(self.args.batch_size, 64)
        ckpt, _, scaler, encoder, features, distribution, device, _ = wg.normalize_args_and_load_checkpoint(self.args, self.config)
        self.assertEqual(self.args.latent_dim, 8)
        self.assertEqual(self.args.g_hidden, [16])
        self.assertEqual(self.args.batch_size, 256)
        self.assertEqual(self.args._last_training_time, 335535)
        self.assertEqual(distribution, {0: 100, 1: 100})
        model = wg.build_and_load_generator(self.args, self.config, ckpt, device, scaler.model_feature_dim, 2)
        self.args.n_samples = 0.2
        wg.set_seed(42)
        _, requested, total = wg.compute_generation_counts_and_labels(self.args, self.config, distribution, encoder, 2)
        result, _, _ = wg.generate_unique_decoded_samples(self.args, self.config, model, device, requested, scaler, encoder, features)
        self.assertEqual(len(result), total)
        self.assertEqual(result.Label.value_counts().to_dict(), {'A': 20, 'B': 20})
        self.assertFalse(result.duplicated().any())
        self.assertTrue(np.isfinite(result.x).all())

    def test_completed_first_unique_journal_reuses_all_rows(self):
        labels = np.repeat([0, 1], [100, 150])
        wg.set_seed(8)
        expected = self.generate(labels, self.decoder)
        wg.set_seed(8)
        def unexpected(*args):
            self.fail('Completed journal should not draw the first population again')
        pd.testing.assert_frame_equal(expected, self.generate(labels, unexpected))

    def test_adaptive_refill_can_exceed_configured_floor(self):
        self.config['generation']['max_duplicate_refill_rounds'] = 1
        labels = np.repeat([0, 1], [100, 200])
        wg.set_seed(42)
        result = self.generate(labels, self.decoder)
        self.assertEqual(len(result), 300)
        self.assertGreater(self.args._last_generation_uniqueness['refill_rounds'], 1)
        self.assertLessEqual(self.args._last_generation_uniqueness['candidate_draws'], 1500)

    def test_hard_draw_budget_and_retry_with_increased_budget(self):
        self.config['generation']['max_duplicate_draw_multiplier'] = 1
        labels = np.repeat([0, 1], [100, 200])
        wg.set_seed(42)
        with self.assertRaisesRegex(RuntimeError, 'exhausted'):
            self.generate(labels, self.decoder)
        self.assertFalse(Path(self.args.out_file).exists())
        old = self.args._unique_generation_journal_dir
        self.config['generation']['max_duplicate_draw_multiplier'] = 5
        wg.set_seed(42)
        result = self.generate(labels, self.decoder)
        self.assertEqual(len(result), 300)
        self.assertEqual(old, self.args._unique_generation_journal_dir)

    def test_stagnation_and_existing_output_protection(self):
        Path(self.args.out_file).write_text('existing output')
        def identical(args, model, device, labels, scaler, encoder, features):
            return pd.DataFrame({'x': np.zeros(len(labels), dtype=np.float32), 'Label': encoder.inverse_transform(labels)})
        with self.assertRaisesRegex(RuntimeError, 'three successive'):
            self.generate(np.repeat([0, 1], [10, 10]), identical)
        self.assertEqual(Path(self.args.out_file).read_text(), 'existing output')
        self.args.only_missing_augmented = True
        with patch.object(wg, 'normalize_args_and_load_checkpoint', side_effect=AssertionError('must skip')):
            wg.generate(self.args, self.config)
        self.assertEqual(Path(self.args.out_file).read_text(), 'existing output')

    def test_signed_zero_and_class_identity(self):
        frame = pd.DataFrame({'x': np.array([0, -0.0, 0], dtype=np.float32), 'Label': ['A', 'A', 'B']})
        signatures = wg.hash_canonical_decoded_rows(frame, ['x', 'Label'])
        journal = GenerationJournal(self.args.out_file, {'test': True})
        selected, hashes = journal.select(frame, signatures)
        self.assertEqual(len(selected), 2)
        journal.commit(selected, hashes, {'counts': {0: 1, 1: 1}})
        reopened = GenerationJournal(self.args.out_file, {'test': True})
        self.assertEqual(len(reopened.seen), 2)
        self.assertEqual(len(reopened.materialize(['x', 'Label'])), 2)
        self.assertEqual(len(GenerationJournal(self.args.out_file, {'test': False}).seen), 0)

    def test_corrupted_payload_is_not_materialized(self):
        frame = pd.DataFrame({'x': np.array([1], dtype=np.float32), 'Label': ['A']})
        journal = GenerationJournal(self.args.out_file, {'test': True})
        selected, hashes = journal.select(frame, wg.hash_canonical_decoded_rows(frame, ['x', 'Label']))
        journal.commit(selected, hashes, {'counts': {0: 1}})
        import pickle
        path = journal.blocks[0]
        with path.open('rb') as stream:
            header, payload = pickle.load(stream), pickle.load(stream)
        with path.open('wb') as stream:
            pickle.dump(header, stream)
            pickle.dump(payload + b'corruption', stream)
        reopened = GenerationJournal(self.args.out_file, {'test': True})
        with self.assertRaisesRegex(RuntimeError, 'Corrupt generation journal frame'):
            reopened.materialize(['x', 'Label'])

    def test_output_lock_is_exclusive(self):
        with output_lock(self.args.out_file):
            with self.assertRaisesRegex(RuntimeError, 'already active'):
                with output_lock(self.args.out_file):
                    self.fail('must not acquire duplicate lock')
        with output_lock(self.args.out_file):
            pass

    def test_cli_and_yaml_defaults(self):
        argv = ['wgangp.py', '--max-duplicate-refill-rounds', '90', '--max_duplicate_draw_multiplier', '6',
            '--duplicate-refill-batch-rows', '123', '--quality-recovery-max-draw-rows-per-class', '60000000',
            '--quality-recovery-max-draw-multiplier', '4', '--historical-training-time-s', '335534.02',
            '--no-resume-unique-generation', '--no-quality-recovery-dynamic-draw-budget']
        with patch.object(sys, 'argv', argv):
            generation = wg.args_to_config_overrides(wg.parse_args())['generation']
        self.assertEqual(generation, {'max_duplicate_refill_rounds': 90, 'max_duplicate_draw_multiplier': 6.0,
            'duplicate_refill_batch_rows': 123, 'quality_recovery_max_draw_rows_per_class': 60000000,
            'quality_recovery_max_draw_multiplier': 4.0, 'historical_training_time_s': 335534.02,
            'resume_unique_generation': False, 'quality_recovery_dynamic_draw_budget': False})
        root = Path(__file__).resolve().parents[1]
        for file in ('config.yaml', 'config.yaml.example'):
            loaded = yaml.safe_load((root / file).read_text())['generation']
            self.assertEqual(loaded['max_duplicate_refill_rounds'], 64)
            self.assertEqual(loaded['max_duplicate_draw_multiplier'], 5.0)
            self.assertEqual(loaded['duplicate_refill_batch_rows'], 32768)
            self.assertEqual(loaded['quality_recovery_max_draw_rows_per_class'], 52000000)


if __name__ == '__main__':
    unittest.main()
