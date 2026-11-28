"""GA scheduling and checkpoint regression tests (no expensive LATAM training)."""
import ast
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import os
from pathlib import Path
import pickle
import random
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from utils.ga_execution import FitnessStore, array_digest, atomic_pickle, evaluate_batch, identity


def metrics(mask):
    return (0.9, 0.9, 0.9, sum(mask) / len(mask), 0., 0., 0., 0., 0., 0., 0., 0., sum(mask))


def delayed_metrics(mask):
    time.sleep(.15 if mask[0] else .005)
    return metrics(mask)


def real_cv_mask(mask, source=None):
    # Run the actual RF/CV definitions even in a minimal test environment with
    # no Telegram or DEAP. The full-module tests below exercise spawn wiring.
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from threading import Lock
    from typing import Any, cast
    from types import SimpleNamespace
    source = Path(source) if source is not None else Path(__file__).resolve().parents[1] / 'genetic_algorithm.py'
    names = {'resolve_n_jobs', 'instantiate_estimator', 'compute_fpr_fnr_from_labels', 'compute_fold_metrics',
             'safe_cv_predict', 'run_cv_fold_loop', 'evaluate_feature_mask_holdout', 'evaluate_feature_mask'}
    tree = ast.parse(source.read_text(encoding='utf-8'))
    definitions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(locals(), np=np, CONFIG={'genetic_algorithm': {'multiprocessing': {'n_jobs': 1},
                     'cross_validation': {'n_folds': 10}, 'early_stop': {'folds': 3, 'acc_threshold': .75}}},
                     GA_ACTIVE_WORKERS=1, CPU_PROCESSES=1, GA_FITNESS_ID='test', GA_WORKER_DATA=None,
                     GA_CV_SPLITS=None, fitness_cache={}, fitness_cache_lock=Lock(),
                     BackgroundColors=SimpleNamespace(YELLOW=''), Style=SimpleNamespace(RESET_ALL=''),
                     send_exception_via_telegram=lambda *args: None)
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), 'exec'), namespace)
    rng = np.random.default_rng(42)
    x = rng.normal(size=(120, 4))
    y = (x[:, 0] > 0).astype(int)
    return namespace['evaluate_feature_mask'](mask, x, y)


class SchedulingTests(unittest.TestCase):
    def test_completion_order_and_duplicate_mask_assignments(self):
        first, second = (1, 1, 0), (0, 1, 0)
        completed, saved = [], {}
        with ThreadPoolExecutor(2) as pool:
            result = evaluate_batch([first, second, first], delayed_metrics, saved.get,
                                    saved.__setitem__, lambda m, f, n, r: completed.append((m, n, r)), pool)
        self.assertEqual(completed, [(second, 1, False), (first, 2, False)])
        self.assertEqual(result, [(2/3, -2), (1/3, -1), (2/3, -2)])
        self.assertEqual(len(saved), 2)

    def test_restart_reuses_each_durable_mask(self):
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder) / 'fitness.sqlite3'
            a, b = (1, 0), (0, 1)
            store = FitnessStore(db)
            def fail(mask):
                if mask == b:
                    raise RuntimeError('simulated interruption')
                return metrics(mask)
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                evaluate_batch([a, b], fail, store.get, store.put, lambda *args: None)
            store.close()
            store = FitnessStore(db)
            calls = []
            try:
                result = evaluate_batch([b, a, a], lambda m: calls.append(m) or metrics(m), store.get, store.put, lambda *args: None)
                self.assertEqual(calls, [b])
                self.assertEqual(result, [(0.5, -1)] * 3)
            finally:
                store.close()

    def test_failed_atomic_checkpoint_preserves_previous(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'checkpoint.pkl'
            atomic_pickle(path, {'gen': 1})
            with patch('utils.ga_execution.os.replace', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):
                    atomic_pickle(path, {'gen': 2})
            self.assertEqual(pickle.loads(path.read_bytes()), {'gen': 1})
            self.assertEqual(list(Path(folder).iterdir()), [path])

    def test_identity_changes_when_values_order_or_config_changes(self):
        x = np.array([[1., 2.], [3., 4.]])
        self.assertEqual(array_digest(x), array_digest(np.asfortranarray(x)))
        self.assertNotEqual(array_digest(x), array_digest(x[::-1]))
        self.assertNotEqual(array_digest(x), array_digest(x[:, ::-1]))
        self.assertNotEqual(identity(array_digest(x), 10), identity(array_digest(x), 5))

    def test_spawn_executor_with_real_processes(self):
        masks = [(1, 0), (1, 1), (1, 0)]
        with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
            result = evaluate_batch(masks, metrics, lambda _: None, lambda *_: None, lambda *_: None, pool)
        self.assertEqual(result, [(0.5, -1), (1., -2), (0.5, -1)])

    def test_real_cv_metrics_preserved_across_spawn_workers(self):
        masks = [(1, 1, 0, 0), (1, 1, 1, 0), (1, 1, 0, 0)]
        baseline = {mask: real_cv_mask(mask) for mask in set(masks)}
        observed = {}
        with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
            result = evaluate_batch(masks, real_cv_mask, observed.get, observed.__setitem__, lambda *_: None, pool)
        self.assertEqual(result, [(baseline[m][3], -baseline[m][12]) for m in masks])
        for mask, value in baseline.items():
            np.testing.assert_array_equal(observed[mask], value)

    def test_no_verbose_branch_around_batch_evaluation(self):
        tree = ast.parse(Path('genetic_algorithm.py').read_text(encoding='utf-8'))
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'create_and_evaluate_offspring')
        self.assertEqual(sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'evaluate_feature_masks_verbose' for n in ast.walk(function)), 1)
        self.assertFalse(any(isinstance(n, ast.If) and isinstance(n.test, ast.Name) and n.test.id == 'verbose' for n in ast.walk(function)))


# Full integration needs the same dependencies as the application. CI installs them
# and explicitly imports genetic_algorithm before invoking this test file.
try:
    import genetic_algorithm as ga
except ModuleNotFoundError as exc:
    ga = None
    MISSING = str(exc)
else:
    MISSING = ''


@unittest.skipIf(ga is None, f'GA dependencies unavailable: {MISSING}')
class GAIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        for name in ('CONFIG', 'CPU_PROCESSES', 'GA_ACTIVE_WORKERS', 'GA_RUN_ID', 'GA_FITNESS_ID', 'GA_CV_SPLITS', 'GA_STORE'):
            self.addCleanup(setattr, ga, name, getattr(ga, name))
        ga.CONFIG = ga.get_default_config()
        ga.CONFIG['execution']['verbose'] = False
        ga.CONFIG['telegram']['enabled'] = False
        ga.CONFIG['genetic_algorithm']['export']['results_dir'] = str(self.folder)
        ga.CONFIG['genetic_algorithm']['multiprocessing']['n_jobs'] = 1
        ga.GA_ACTIVE_WORKERS = 1
        ga.GA_RUN_ID = 'test-run'
        ga.GA_FITNESS_ID = 'test-fitness'
        ga.GA_STORE = None
        ga.GA_CV_SPLITS = None
        ga.fitness_cache.clear()
        self.addCleanup(ga.fitness_cache.clear)
        self.addCleanup(random.setstate, random.getstate())
        self.addCleanup(np.random.set_state, np.random.get_state())
        for name in ('send_telegram_message', 'send_exception_via_telegram'):
            mock = patch.object(ga, name, return_value=None)
            mock.start()
            self.addCleanup(mock.stop)

    def test_real_rf_cv_equal_in_parallel_and_serial_with_verbose(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(120, 4))
        y = (X[:, 0] > 0).astype(int)
        masks = [(1, 0, 1, 0), (1, 1, 0, 0), (1, 0, 1, 0)]
        # Real 100-tree RF and real ten-fold CV, unchanged evaluation code.
        serial = [ga.evaluate_feature_mask(m, X, y) for m in masks]
        ga.fitness_cache.clear()
        ga.GA_CV_SPLITS = None
        ga.CPU_PROCESSES = 2
        _, executor = ga.create_sweep_results_and_pool(4, 4, (X, X, y, y, list('abcd')), self.folder)
        try:
            toolbox, _, _ = ga.setup_genetic_algorithm(4, 4, pool=executor)
            ga.CONFIG['execution']['verbose'] = True
            fits = ga.evaluate_feature_masks_verbose(masks, toolbox, 1, 1, 1, 1, 4)
            self.assertEqual(fits, [(m[3], -m[12]) for m in serial])
            for mask, result in zip(masks, serial):
                np.testing.assert_array_equal(ga.fitness_cache[mask], result)
            ga.CONFIG['execution']['verbose'] = False
            self.assertEqual(ga.evaluate_feature_masks_verbose(masks, toolbox), fits)
        finally:
            ga.close_shared_pool_safely(executor)

    def test_resume_trajectory_after_interruption(self):
        X = np.zeros((10, 4)); y = np.arange(10) % 2
        def run():
            toolbox, population, hof = ga.setup_genetic_algorithm(4, 6)
            return ga.run_genetic_algorithm_loop(toolbox, population, hof, X, y,
                    n_generations=5, csv_path=str(self.folder / 'test.csv'), pop_size=6,
                    cxpb=.5, mutpb=.01, run=1, runs=1)
        with patch.object(ga, 'evaluate_feature_mask', side_effect=lambda mask, **_: metrics(mask)):
            random.seed(123); np.random.seed(456)
            ga.CONFIG['execution']['resume_progress'] = False
            baseline = run()
            baseline_random = random.getstate()
            baseline_numpy = np.random.get_state()
            ga.fitness_cache.clear()
            random.seed(123); np.random.seed(456)
            ga.CONFIG['execution']['resume_progress'] = True
            evolve = ga.evolve_population_one_generation
            calls = 0
            def interrupt(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError('interrupt')
                return evolve(*args, **kwargs)
            with patch.object(ga, 'evolve_population_one_generation', side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, 'interrupt'):
                    run()
            ga.fitness_cache.clear()
            random.seed(999); np.random.seed(999)
            resumed = run()
            self.assertEqual(list(resumed[0]), list(baseline[0]))
            self.assertEqual(resumed[0].fitness.values, baseline[0].fitness.values)
            self.assertEqual(resumed[1:], baseline[1:])
            self.assertEqual(random.getstate(), baseline_random)
            self.assertEqual(np.random.get_state()[0], baseline_numpy[0])
            np.testing.assert_array_equal(np.random.get_state()[1], baseline_numpy[1])

    def test_completed_run_file_retained_and_rng_restored(self):
        args = (str(self.folder / 'dataset.csv'), 4, 5, .5, .01, 1, 10)
        state_id = ga.compute_state_id(*args)
        ga.save_run_result(self.folder, state_id, {'best_ind': [1, 0], 'metrics': (1,)})
        ga.cleanup_state_for_id(self.folder, state_id)
        saved_rng = random.getstate()
        random.random()
        result, recovered_id = ga.load_cached_run_if_any(self.folder, *args)
        self.assertEqual(recovered_id, state_id)
        self.assertEqual(result['best_ind'], [1, 0])
        self.assertEqual(random.getstate(), saved_rng)


if __name__ == '__main__':
    unittest.main()
