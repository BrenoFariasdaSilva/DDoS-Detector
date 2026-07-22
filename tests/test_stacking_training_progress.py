import argparse  # Build minimal CLI namespaces for precedence tests.
import contextlib  # Capture deterministic progress output without changing production logging.
import io  # Provide an in-memory text stream for progress assertions.
import os  # Verify progress records contain the active process identity.
import pickle  # Verify fitted estimator serialization remains byte-identical.
from pathlib import Path  # Read isolated detached log artifacts.
import tempfile  # Create isolated detached log destinations.
import threading  # Verify heartbeat threads stop after every fit outcome.
import time  # Simulate blocking estimators for heartbeat coverage.
import unittest  # Provide the repository-standard focused test runner.
from unittest import mock  # Inject deterministic clocks and lightweight AutoML objectives.
import warnings  # Suppress expected convergence warnings in tiny deterministic fixtures.

import lightgbm as lgb  # Exercise the exact installed LightGBM estimator callback.
import numpy as np  # Compare predictions, probabilities, metrics, and feature importance exactly.
from sklearn.base import clone  # Create independent baseline and progress-enabled estimators.
from sklearn.datasets import make_classification  # Build small deterministic classification data.
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier  # Exercise boosting, forest, and stacking paths.
from sklearn.linear_model import LogisticRegression  # Exercise heartbeat-only logistic training.
from sklearn.metrics import accuracy_score, f1_score  # Compare unchanged evaluation metrics.
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid  # Exercise phase-only neighbor training.
from sklearn.neural_network import MLPClassifier  # Exercise heartbeat-only neural-network training.
from sklearn.svm import SVC  # Exercise heartbeat-only SVM training.
from xgboost import XGBClassifier  # Exercise the exact installed XGBoost estimator callback.
from xgboost.callback import TrainingCallback  # Build one existing public callback for preservation coverage.

import stacking  # Exercise the real production progress integration.
import training_progress  # Exercise reusable progress state, ETA, heartbeat, and terminal behavior directly.


class ExistingXGBoostCallback(TrainingCallback):  # Record existing public XGBoost callback execution
    """Record completed XGBoost rounds through the public callback API."""

    def __init__(self):  # Initialize existing callback state
        """
        Initialize one existing XGBoost callback fixture.

        :param self: Instance of the ExistingXGBoostCallback class.
        :return: None.
        """

        self.units = []  # Store completed public boosting rounds.

    def after_iteration(self, model, epoch, evals_log):  # Record one completed public boosting round
        """
        Record one completed XGBoost boosting round.

        :param self: Instance of the ExistingXGBoostCallback class.
        :param model: Active public XGBoost model handle.
        :param epoch: Zero-based completed boosting-round index.
        :param evals_log: Public XGBoost evaluation history mapping.
        :return: False so training continues unchanged.
        """

        self.units.append(epoch + 1)  # Record the real completed round.
        return False  # Preserve ordinary training continuation.


class SleepingEstimator:  # Simulate one unsupported blocking estimator
    """Provide a deterministic blocking fit without public training units."""

    def __init__(self, delay_seconds=0.04):  # Initialize blocking estimator state
        """
        Initialize one sleeping estimator fixture.

        :param self: Instance of the SleepingEstimator class.
        :param delay_seconds: Blocking fit duration in seconds.
        :return: None.
        """

        self.delay_seconds = delay_seconds  # Store the deterministic blocking duration.

    def fit(self, X_train, y_train):  # Execute one blocking fit
        """
        Sleep once and return the fitted estimator.

        :param self: Instance of the SleepingEstimator class.
        :param X_train: Training feature matrix.
        :param y_train: Training labels.
        :return: The fitted estimator instance.
        """

        time.sleep(self.delay_seconds)  # Block long enough for multiple heartbeat intervals.
        return self  # Preserve standard estimator fit return semantics.

    def predict(self, X_test):  # Produce deterministic predictions for lifecycle integration tests
        """Return one deterministic class for every supplied row."""

        return np.zeros(len(X_test), dtype=np.int64)  # Keep the fixture independent from estimator internals.


class FailingEstimator(SleepingEstimator):  # Simulate one unsupported failing estimator
    """Raise one original fit exception after a deterministic heartbeat window."""

    def __init__(self, failure, delay_seconds=0.04):  # Initialize failing estimator state
        """
        Initialize one failing estimator fixture.

        :param self: Instance of the FailingEstimator class.
        :param failure: Exception instance raised by fit.
        :param delay_seconds: Blocking fit duration in seconds.
        :return: None.
        """

        super().__init__(delay_seconds=delay_seconds)  # Initialize the inherited blocking duration.
        self.failure = failure  # Store the exact exception instance to preserve.

    def fit(self, X_train, y_train):  # Execute one failing blocking fit
        """
        Sleep once and raise the configured exception instance.

        :param self: Instance of the FailingEstimator class.
        :param X_train: Training feature matrix.
        :param y_train: Training labels.
        :return: Never returns because the configured exception is raised.
        """

        time.sleep(self.delay_seconds)  # Block long enough for heartbeat output before failure.
        raise self.failure  # Preserve the original exception instance and traceback path.


class TrainingProgressTests(unittest.TestCase):  # Group deterministic training progress behavior
    """Verify callbacks, heartbeats, phases, results, and serialization."""

    @classmethod
    def setUpClass(cls):  # Build one small deterministic dataset for the suite
        """
        Build one deterministic binary classification fixture.

        :param cls: TrainingProgressTests class.
        :return: None.
        """

        cls.X_train, cls.y_train = make_classification(n_samples=80, n_features=6, n_informative=4, n_redundant=0, random_state=42)  # Build a compact deterministic binary dataset.
        cls.fast_config = {"evaluation": {"training_progress_interval_minutes": 0.01 / 60.0}}  # Use a ten-millisecond progress interval only inside focused tests.

    def fit_with_output(self, model, classifier_name, fit_kwargs=None):  # Fit one estimator and capture progress output
        """
        Fit one estimator through the production progress integration.

        :param self: Instance of the TrainingProgressTests class.
        :param model: Estimator instance to fit.
        :param classifier_name: Classifier identity for progress output.
        :param fit_kwargs: Optional public fit keyword arguments.
        :return: Captured progress output string.
        """

        output = io.StringIO()  # Allocate one isolated progress output stream.
        with contextlib.redirect_stdout(output):  # Capture only output from this estimator fit.
            stacking.fit_classifier_with_progress(model, self.X_train, self.y_train, "PCA Components", classifier_name, config=self.fast_config, fit_kwargs=fit_kwargs)  # Execute the production fit path on small deterministic data.
        return output.getvalue()  # Return captured progress text for assertions.

    def assert_results_identical(self, baseline, observed):  # Compare scientific outputs from two fitted estimators
        """
        Require identical predictions, probabilities when exposed, metrics, and serialization.

        :param self: Instance of the TrainingProgressTests class.
        :param baseline: Estimator fitted without progress integration.
        :param observed: Estimator fitted with progress integration.
        :return: None.
        """

        baseline_predictions = baseline.predict(self.X_train)  # Generate baseline predictions on the deterministic fixture.
        observed_predictions = observed.predict(self.X_train)  # Generate progress-enabled predictions on the same fixture.
        np.testing.assert_array_equal(baseline_predictions, observed_predictions)  # Require exact prediction equality.
        self.assertEqual(accuracy_score(self.y_train, baseline_predictions), accuracy_score(self.y_train, observed_predictions))  # Require unchanged accuracy.
        self.assertEqual(f1_score(self.y_train, baseline_predictions), f1_score(self.y_train, observed_predictions))  # Require unchanged F1 score.
        if hasattr(baseline, "predict_proba") and hasattr(observed, "predict_proba"):  # Compare probabilities only when both estimators expose them.
            np.testing.assert_array_equal(baseline.predict_proba(self.X_train), observed.predict_proba(self.X_train))  # Require exact probability equality.
        self.assertEqual(pickle.dumps(baseline), pickle.dumps(observed))  # Require byte-identical fitted model serialization.

    def test_default_yaml_cli_precedence_and_fractional_conversion(self):  # Verify the single minutes-based configuration source
        """Verify default, YAML, CLI precedence, parsing, logging, and seconds conversion."""

        self.assertEqual(training_progress.DEFAULT_TRAINING_PROGRESS_INTERVAL_MINUTES, 15.0)  # Require the built-in fifteen-minute default.
        defaults = stacking.get_default_config()  # Build the production defaults without file or CLI overrides.
        self.assertEqual(defaults["evaluation"]["training_progress_interval_minutes"], 15.0)  # Require the runtime default to use the shared constant.
        repository_root = Path(stacking.__file__).resolve().parent  # Resolve both committed configuration files beside stacking.py.
        for config_name in ("config.yaml", "config.yaml.example"):  # Verify the operator and example configurations together.
            with self.subTest(config=config_name):  # Isolate configuration-file failures.
                file_config = stacking.load_config_file(str(repository_root / config_name))  # Parse through the production YAML loader.
                self.assertEqual(file_config["evaluation"]["training_progress_interval_minutes"], 15)  # Require the documented fifteen-minute YAML default.

        yaml_config = stacking.merge_configs(stacking.get_default_config(), {"evaluation": {"training_progress_interval_minutes": 0.5}}, None)  # Merge a fractional YAML override without CLI arguments.
        self.assertEqual(yaml_config["evaluation"]["training_progress_interval_minutes"], 0.5)  # Require YAML to override the built-in default.
        cli_args = argparse.Namespace(training_progress_interval_minutes=2.5)  # Build one explicit CLI override only.
        cli_config = stacking.merge_configs(stacking.get_default_config(), {"evaluation": {"training_progress_interval_minutes": 0.5}}, cli_args)  # Apply CLI over YAML and defaults.
        self.assertEqual(cli_config["evaluation"]["training_progress_interval_minutes"], 2.5)  # Require CLI to win over YAML.
        with mock.patch.object(stacking.sys, "argv", ["stacking.py", "--training-progress-interval-minutes", "0.5"]):  # Parse the public CLI spelling directly.
            parsed = stacking.parse_cli_args()  # Exercise the production argparse path.
        self.assertEqual(parsed.training_progress_interval_minutes, 0.5)  # Require fractional CLI minutes to survive parsing.
        progress = stacking.build_training_progress("Full Features", "SVM", config={"evaluation": {"training_progress_interval_minutes": 0.5}})  # Cross the single minutes-to-seconds runtime boundary.
        self.assertEqual(progress.interval_seconds, 30.0)  # Require 0.5 minutes to become exactly thirty seconds once.

        output = io.StringIO()  # Capture resolved runtime configuration logging.
        with contextlib.redirect_stdout(output):  # Isolate the one configuration log call.
            stacking.log_resolved_configuration(cli_config)  # Log the effective CLI-over-YAML interval.
        self.assertEqual(output.getvalue().count("Training progress interval:"), 1)  # Require one resolved interval record.
        self.assertIn("2.5 minutes", output.getvalue())  # Preserve readable fractional formatting.

    def test_invalid_progress_intervals_are_rejected_clearly(self):  # Verify finite positive validation
        """Reject zero, negatives, non-finite values, empty values, text, and booleans."""

        invalid_values = (0, -1, float("nan"), float("inf"), float("-inf"), "", "invalid", None, True)  # Enumerate every prohibited configuration class.
        for value in invalid_values:  # Validate every invalid YAML/programmatic value.
            with self.subTest(value=value):  # Isolate the rejected value.
                with self.assertRaisesRegex(ValueError, "evaluation.training_progress_interval_minutes.*positive finite"):  # Require the configuration key in the error.
                    stacking.validate_training_progress_interval_minutes(value)  # Exercise production validation directly.

        with mock.patch.object(stacking, "send_exception_via_telegram"):  # Keep the merge failure local to this deterministic test.
            with self.assertRaisesRegex(ValueError, "--training-progress-interval-minutes.*positive finite"):  # Require the invalid CLI option in the error.
                stacking.merge_configs(stacking.get_default_config(), {}, argparse.Namespace(training_progress_interval_minutes=0.0))  # Reject an explicit zero CLI override.
        for invalid_text in ("", "not-a-number"):  # Verify argparse rejects empty and non-numeric option values.
            with self.subTest(cli_value=invalid_text):  # Isolate parser failures.
                stderr = io.StringIO()  # Capture argparse's clear option error.
                with mock.patch.object(stacking.sys, "argv", ["stacking.py", "--training-progress-interval-minutes", invalid_text]):  # Supply the invalid public option.
                    with contextlib.redirect_stderr(stderr), self.assertRaises(SystemExit):  # Require normal argparse rejection.
                        stacking.parse_cli_args()  # Exercise production CLI parsing.
                self.assertIn("--training-progress-interval-minutes", stderr.getvalue())  # Identify the invalid option clearly.

    def test_unit_callbacks_are_rate_limited_and_final_is_immediate(self):  # Verify callback throttling and latest-state output
        """Emit only the latest due callback state and one immediate final state."""

        output = io.StringIO()  # Capture deterministic callback-backed progress.
        progress = training_progress.TrainingProgress("GA Features", "XGBoost", stacking.calculate_execution_time, output_stream=output, total_units=10, unit_label="Round", report_interval_seconds=60.0)  # Build one minute-spaced reporter.
        with mock.patch.object(training_progress.time, "monotonic", side_effect=[0.0, 5.0, 20.0, 59.0, 60.0, 61.0, 62.0]):  # Control start and frequent callback times.
            with progress:  # Start this task's independent timer at zero.
                progress.report_unit(1)  # Retain without logging at five seconds.
                progress.report_unit(2)  # Retain without logging at twenty seconds.
                progress.report_unit(3)  # Retain without logging just before one minute.
                progress.report_unit(4)  # Emit the latest state at the interval boundary.
                progress.report_unit(10)  # Emit final 100% immediately one second later.
                progress.report_unit(10)  # Suppress a duplicate final callback.
        records = [line for line in output.getvalue().splitlines() if line.startswith("[TRAINING]")]  # Read only progress records.
        self.assertEqual(len(records), 2)  # Require one recurring record and one immediate final record.
        self.assertIn("Round: 4/10 | Progress: 40.00%", records[0])  # Require the most recent callback state at the interval.
        self.assertIn("Round: 10/10 | Progress: 100.00%", records[1])  # Require immediate final progress.
        self.assertNotIn("Round: 1/10", output.getvalue())  # Forbid per-round output before the interval.
        self.assertEqual(progress.latest_completed_units, 10)  # Retain the latest genuine public state even after rate limiting.

    def test_heartbeat_interval_and_task_timers_are_independent(self):  # Verify heartbeat timing, worker independence, and reset
        """Keep heartbeat and combination timers local to each progress scope."""

        heartbeat_output = io.StringIO()  # Capture one deterministic heartbeat scope.
        heartbeat_progress = training_progress.TrainingProgress("RFE Features", "SVM", stacking.calculate_execution_time, output_stream=heartbeat_output, report_interval_seconds=60.0)  # Build one minute-spaced heartbeat reporter.
        with mock.patch.object(training_progress.time, "monotonic", side_effect=[100.0, 159.0, 160.0]):  # Control start, pre-interval, and due times.
            with heartbeat_progress:  # Start this task at one hundred seconds.
                self.assertFalse(heartbeat_progress.report_heartbeat())  # Require silence before the configured interval.
                self.assertTrue(heartbeat_progress.report_heartbeat())  # Require one heartbeat at the interval.
        self.assertEqual(heartbeat_output.getvalue().count("Status: Active"), 1)  # Require exactly one due heartbeat.

        first_output, second_output = io.StringIO(), io.StringIO()  # Capture two concurrent-worker-equivalent task scopes separately.
        first = training_progress.TrainingProgress("Full Features", "SVM", stacking.calculate_execution_time, output_stream=first_output, report_interval_seconds=60.0)  # Build the first worker-local reporter.
        second = training_progress.TrainingProgress("PCA Components", "MLP", stacking.calculate_execution_time, output_stream=second_output, report_interval_seconds=60.0)  # Build the second worker-local reporter.
        with mock.patch.object(training_progress.time, "monotonic", side_effect=[100.0, 200.0, 160.0, 260.0]):  # Give each scope its own start and due timestamp.
            with first, second:  # Activate both independent scopes together.
                self.assertTrue(first.report_heartbeat())  # Emit the Full task's heartbeat at its own boundary.
                self.assertTrue(second.report_heartbeat())  # Emit the PCA task's heartbeat at its separate boundary.
        self.assertEqual(first_output.getvalue().count("Status: Active"), 1)  # Require the first timer to emit independently.
        self.assertEqual(second_output.getvalue().count("Status: Active"), 1)  # Require the second timer to emit independently.

        new_combination = training_progress.TrainingProgress("Full Features", "Random Forest", stacking.calculate_execution_time, output_stream=io.StringIO(), report_interval_seconds=60.0)  # Build the next classifier combination.
        with mock.patch.object(training_progress.time, "monotonic", side_effect=[300.0, 300.0]):  # Start and poll the new scope at the same timestamp.
            with new_combination:  # Reset timing through a fresh training scope.
                self.assertFalse(new_combination.report_heartbeat())  # Require no inherited due state from the previous Full task.

    def test_short_fit_start_completion_and_failure_remain_immediate(self):  # Verify lifecycle events bypass recurring throttling
        """Keep start, completion, and failure behavior immediate for short fits."""

        slow_interval_config = {"evaluation": {"training_progress_interval_minutes": 1.0}}  # Configure an interval far longer than the fixture fits.
        common_args = (self.X_train, self.y_train, self.X_train, self.y_train)  # Reuse deterministic training and testing arrays.
        success_output = io.StringIO()  # Capture production lifecycle records for a short successful fit.
        with mock.patch.object(stacking, "send_telegram_message"), mock.patch.object(stacking, "write_memory_phase_event"):  # Isolate external notifications and watcher artifacts.
            with contextlib.redirect_stdout(success_output):  # Capture immediate lifecycle output.
                stacking.evaluate_individual_classifier(SleepingEstimator(delay_seconds=0.005), "SVM", *common_args, dataset_file="fixture.csv", feature_names=[f"f{index}" for index in range(self.X_train.shape[1])], feature_set="Full Features", config=slow_interval_config, training_ram_stats={})  # Execute the real production lifecycle around one short fit.
        self.assertIn("Phase: Training | Status: Started", success_output.getvalue())  # Require immediate training start.
        self.assertIn("Phase: Training | Status: Completed", success_output.getvalue())  # Require immediate fit completion.
        self.assertNotIn("Status: Active", success_output.getvalue())  # Avoid an unnecessary intermediate heartbeat for a short fit.

        failure = RuntimeError("Immediate fixture failure")  # Create the exact failure expected from the production fit path.
        failure_output = io.StringIO()  # Capture the production failure output.
        with mock.patch.object(stacking, "send_exception_via_telegram"), mock.patch.object(stacking, "write_memory_phase_event"):  # Isolate external failure reporting and watcher artifacts.
            with self.assertRaises(RuntimeError) as raised:  # Require immediate original failure propagation.
                with contextlib.redirect_stdout(failure_output):  # Capture the existing immediate failure record.
                    stacking.evaluate_individual_classifier(FailingEstimator(failure, delay_seconds=0.005), "SVM", *common_args, dataset_file="fixture.csv", feature_names=[f"f{index}" for index in range(self.X_train.shape[1])], feature_set="Full Features", config=slow_interval_config, training_ram_stats={})  # Execute the real failing lifecycle without waiting for the recurring interval.
        self.assertIs(raised.exception, failure)  # Preserve the exact estimator failure object.
        self.assertIn("Immediate fixture failure", failure_output.getvalue())  # Require the existing failure log immediately.
        self.assertNotIn("Status: Active", failure_output.getvalue())  # Avoid an intermediate heartbeat before a short failure.

    def test_evaluation_combination_metadata_is_exact_in_phase_and_progress_records(self):  # Verify authoritative metadata formatting across all planned modes
        """Verify phase and progress records identify every required evaluation-combination mode."""

        combinations = [(False, None, "Default Hyperparameters", "Off"), (True, None, "Optimized Hyperparameters", "Off"), (False, 0.25, "Default Hyperparameters", "25%"), (True, 0.50, "Optimized Hyperparameters", "50%"), (False, 0.75, "Default Hyperparameters", "75%"), (True, 1.00, "Optimized Hyperparameters", "100%")]  # Cover every required hyperparameter and augmentation label.
        for hyperparameters_enabled, augmentation_ratio, hyperparameter_label, augmentation_label in combinations:  # Verify each authoritative combination independently.
            with self.subTest(hyperparameters_enabled=hyperparameters_enabled, augmentation_ratio=augmentation_ratio):  # Isolate exact combination-format failures.
                expected_fields = f"Hyperparameters: {hyperparameter_label} | Data Augmentation: {augmentation_label}"  # Build exact required field sequence.
                phase_output = io.StringIO()  # Capture one lifecycle record for current combination.
                with contextlib.redirect_stdout(phase_output):  # Route phase output into isolated stream.
                    stacking.log_training_phase("GA Features", "XGBoost", "Training", "Started", hyperparameters_enabled, augmentation_ratio)  # Emit phase record from authoritative combination values.
                self.assertIn(expected_fields, phase_output.getvalue())  # Require exact lifecycle metadata fields.
                progress_output = io.StringIO()  # Capture one callback progress record for current combination.
                progress = training_progress.TrainingProgress("GA Features", "XGBoost", stacking.calculate_execution_time, output_stream=progress_output, total_units=1, unit_label="Round", report_interval_seconds=60.0, hyperparameters_enabled=hyperparameters_enabled, augmentation_ratio=augmentation_ratio)  # Build callback reporter from same authoritative values.
                with mock.patch.object(training_progress.time, "monotonic", side_effect=[0.0, 1.0]):  # Control start and final callback timestamps.
                    with progress:  # Activate current combination progress scope.
                        progress.report_unit(1)  # Emit immediate final callback record.
                self.assertIn(expected_fields, progress_output.getvalue())  # Require exact callback metadata fields.
        self.assertEqual(training_progress.format_training_combination_fields(None, None), "")  # Keep non-combination AutoML and standalone records unlabeled.

    def test_xgboost_public_rounds_preserve_results_callbacks_and_identity(self):  # Verify XGBoost genuine progress
        """
        Verify XGBoost rounds, results, serialization, callback preservation, and identity restoration.

        :param self: Instance of the TrainingProgressTests class.
        :return: None.
        """

        baseline = XGBClassifier(n_estimators=7, max_depth=2, learning_rate=0.1, random_state=42, n_jobs=1)  # Build the deterministic no-progress baseline.
        observed = clone(baseline)  # Build the independent progress-enabled estimator.
        baseline.fit(self.X_train, self.y_train)  # Fit the baseline once without progress integration.
        output = self.fit_with_output(observed, "XGBoost")  # Fit through the public XGBoost callback path.
        self.assertIn("Round: 7/7 | Progress: 100.00%", output)  # Require the exact configured boosting-round total.
        self.assertNotIn("Status: Active", output)  # Require genuine callbacks instead of heartbeat-only reporting.
        self.assertIsNone(observed.get_params(deep=False).get("callbacks"))  # Require temporary callback removal from estimator identity.
        np.testing.assert_array_equal(baseline.feature_importances_, observed.feature_importances_)  # Require unchanged XGBoost feature importance.
        self.assert_results_identical(baseline, observed)  # Require unchanged predictions, metrics, probabilities, and serialization.

        existing_callback = ExistingXGBoostCallback()  # Build one pre-existing public XGBoost callback.
        existing_callbacks = [existing_callback]  # Preserve an identity-bearing callback list.
        callback_model = XGBClassifier(n_estimators=4, max_depth=2, learning_rate=0.1, random_state=42, n_jobs=1, callbacks=existing_callbacks)  # Build an estimator with an existing callback.
        callback_output = self.fit_with_output(callback_model, "XGBoost")  # Fit while composing the temporary progress callback.
        self.assertEqual(existing_callback.units, [1, 2, 3, 4])  # Require every existing callback invocation to remain intact.
        self.assertIs(callback_model.get_params(deep=False).get("callbacks"), existing_callbacks)  # Require exact callback-list identity restoration.
        self.assertIn("Round: 4/4 | Progress: 100.00%", callback_output)  # Require progress alongside the existing callback.

    def test_lightgbm_public_iterations_preserve_results_and_callbacks(self):  # Verify LightGBM genuine progress
        """
        Verify LightGBM iterations, results, serialization, and callback preservation.

        :param self: Instance of the TrainingProgressTests class.
        :return: None.
        """

        baseline = lgb.LGBMClassifier(n_estimators=7, max_depth=2, learning_rate=0.1, random_state=42, n_jobs=1, verbosity=-1)  # Build the deterministic no-progress baseline.
        observed = clone(baseline)  # Build the independent progress-enabled estimator.
        baseline.fit(self.X_train, self.y_train)  # Fit the baseline once without progress integration.
        output = self.fit_with_output(observed, "LightGBM")  # Fit through the public LightGBM callback path.
        self.assertIn("Iteration: 7/7 | Progress: 100.00%", output)  # Require the exact configured boosting-iteration total.
        np.testing.assert_array_equal(baseline.feature_importances_, observed.feature_importances_)  # Require unchanged LightGBM feature importance.
        self.assertEqual(baseline.booster_.model_to_string(), observed.booster_.model_to_string())  # Require unchanged serialized booster content.
        self.assert_results_identical(baseline, observed)  # Require unchanged predictions, metrics, probabilities, and serialization.

        existing_units = []  # Record one caller-supplied LightGBM callback's invocations.

        def existing_callback(environment):  # Record one existing public LightGBM callback invocation
            """
            Record one completed LightGBM iteration.

            :param environment: Public LightGBM callback environment.
            :return: None.
            """

            existing_units.append(environment.iteration + 1)  # Record the real completed iteration.

        callback_model = clone(baseline)  # Build an independent LightGBM estimator for callback composition.
        callback_output = self.fit_with_output(callback_model, "LightGBM", fit_kwargs={"callbacks": [existing_callback]})  # Fit with the caller's existing public callback preserved.
        self.assertEqual(existing_units, list(range(1, 8)))  # Require every existing LightGBM callback invocation.
        self.assertIn("Iteration: 7/7 | Progress: 100.00%", callback_output)  # Require genuine progress beside the existing callback.

    def test_gradient_boosting_public_stages_preserve_results_and_monitor(self):  # Verify sklearn Gradient Boosting genuine progress
        """
        Verify Gradient Boosting stages, results, serialization, and monitor preservation.

        :param self: Instance of the TrainingProgressTests class.
        :return: None.
        """

        baseline = GradientBoostingClassifier(n_estimators=7, max_depth=2, learning_rate=0.1, random_state=42)  # Build the deterministic no-progress baseline.
        observed = clone(baseline)  # Build the independent progress-enabled estimator.
        baseline.fit(self.X_train, self.y_train)  # Fit the baseline once without progress integration.
        existing_stages = []  # Record an existing public monitor's stage invocations.

        def existing_monitor(stage_index, estimator, local_variables):  # Record one existing public sklearn monitor invocation
            """
            Record one completed Gradient Boosting stage.

            :param stage_index: Zero-based completed stage index.
            :param estimator: Active GradientBoostingClassifier instance.
            :param local_variables: Public monitor local-variable mapping.
            :return: False so training continues unchanged.
            """

            existing_stages.append(stage_index + 1)  # Record the real completed stage.
            return False  # Preserve ordinary training continuation.

        output = self.fit_with_output(observed, "Gradient Boosting", fit_kwargs={"monitor": existing_monitor})  # Fit through the composed public monitor path.
        self.assertEqual(existing_stages, list(range(1, 8)))  # Require every existing monitor invocation.
        self.assertIn("Stage: 7/7 | Progress: 100.00%", output)  # Require the exact configured boosting-stage total.
        np.testing.assert_array_equal(baseline.feature_importances_, observed.feature_importances_)  # Require unchanged sklearn feature importance.
        self.assert_results_identical(baseline, observed)  # Require unchanged predictions, metrics, probabilities, and serialization.

    def test_unsupported_estimators_use_single_fit_without_percentages(self):  # Verify heartbeat-only model semantics
        """
        Verify unsupported estimators retain single-fit semantics without fabricated percentages.

        :param self: Instance of the TrainingProgressTests class.
        :return: None.
        """

        estimators = [  # Build every requested estimator lacking a safe public unit callback.
            ("Random Forest", RandomForestClassifier(n_estimators=7, random_state=42, n_jobs=1)),  # Exercise Random Forest without warm-start batching.
            ("SVM", SVC(kernel="rbf", probability=True, random_state=42)),  # Exercise SVM without solver changes.
            ("Logistic Regression", LogisticRegression(max_iter=200, random_state=42, n_jobs=1)),  # Exercise Logistic Regression without solver changes.
            ("KNN", KNeighborsClassifier(n_neighbors=3, n_jobs=1)),  # Exercise KNN without fabricated training units.
            ("Nearest Centroid", NearestCentroid()),  # Exercise Nearest Centroid without fabricated training units.
            ("MLP (Neural Net)", MLPClassifier(hidden_layer_sizes=(8,), max_iter=40, random_state=42)),  # Exercise MLP without repeated partial fits.
            ("StackingClassifier", StackingClassifier(estimators=[("lr", LogisticRegression(max_iter=200, random_state=42)), ("knn", KNeighborsClassifier(n_neighbors=3))], final_estimator=LogisticRegression(max_iter=200, random_state=42), cv=2, n_jobs=1)),  # Exercise unchanged sklearn stacking CV behavior.
        ]  # Complete the heartbeat-only estimator list.
        with warnings.catch_warnings():  # Suppress expected tiny-fixture convergence warnings.
            warnings.simplefilter("ignore")  # Keep focused output limited to progress assertions.
            for classifier_name, prototype in estimators:  # Compare every heartbeat-only estimator against a direct-fit baseline.
                with self.subTest(classifier=classifier_name):  # Isolate failures by classifier identity.
                    baseline = clone(prototype)  # Build the direct-fit baseline estimator.
                    observed = clone(prototype)  # Build the progress-enabled estimator.
                    baseline.fit(self.X_train, self.y_train)  # Execute one baseline blocking fit.
                    output = self.fit_with_output(observed, classifier_name)  # Execute one progress-enabled blocking fit.
                    self.assertNotIn("Progress:", output)  # Forbid fabricated percentages for unsupported estimators.
                    self.assert_results_identical(baseline, observed)  # Require unchanged predictions, metrics, probabilities, and serialization.
                    if classifier_name == "Random Forest":  # Compare Random Forest importance separately.
                        np.testing.assert_array_equal(baseline.feature_importances_, observed.feature_importances_)  # Require unchanged forest feature importance.
                        self.assertFalse(observed.warm_start)  # Require the original single-fit forest behavior without batching.
                    if classifier_name == "MLP (Neural Net)":  # Inspect the neural-network continuation setting separately.
                        self.assertFalse(observed.warm_start)  # Require the original single-fit MLP behavior without epoch batching.

    def test_heartbeat_stops_after_success_and_preserves_failure(self):  # Verify heartbeat lifecycle and exceptions
        """
        Verify heartbeat cleanup after success and exact exception preservation after failure.

        :param self: Instance of the TrainingProgressTests class.
        :return: None.
        """

        success_output = self.fit_with_output(SleepingEstimator(), "SVM")  # Run one successful blocking heartbeat-only fit.
        self.assertIn("Status: Active", success_output)  # Require a low-frequency active heartbeat.
        self.assertIn("ETA: unavailable", success_output)  # Require unavailable ETA without public units.
        self.assertIn(f"PID: {os.getpid()}", success_output)  # Require the active process identity.
        self.assertNotIn("Progress:", success_output)  # Forbid elapsed-time-derived percentages.
        self.assertFalse(any(thread.name.startswith("training-heartbeat-") for thread in threading.enumerate()))  # Require no heartbeat thread after successful fit.

        failure = RuntimeError("Original fit failure")  # Create the exact exception instance expected from fit.
        output = io.StringIO()  # Allocate isolated failure progress output.
        with self.assertRaises(RuntimeError) as raised:  # Require the original fit exception to propagate.
            with contextlib.redirect_stdout(output):  # Capture heartbeat output from the failing fit.
                stacking.fit_classifier_with_progress(FailingEstimator(failure), self.X_train, self.y_train, "PCA Components", "SVM", config=self.fast_config)  # Execute one failing heartbeat-only fit.
        self.assertIs(raised.exception, failure)  # Require exact exception-instance preservation.
        self.assertIn("Status: Active", output.getvalue())  # Require heartbeat output before the failure.
        self.assertFalse(any(thread.name.startswith("training-heartbeat-") for thread in threading.enumerate()))  # Require no heartbeat thread after exceptional fit.

    def test_eta_uses_completed_public_units(self):  # Verify unit-derived ETA calculation
        """
        Verify ETA derives only from completed public training units.

        :param self: Instance of the TrainingProgressTests class.
        :return: None.
        """

        output = io.StringIO()  # Allocate isolated deterministic ETA output.
        progress = training_progress.TrainingProgress("PCA Components", "XGBoost", stacking.calculate_execution_time, output_stream=output, total_units=4, unit_label="Round", heartbeat=False, report_interval_seconds=0.01)  # Build a four-round genuine progress scope directly from the focused module.
        with mock.patch.object(training_progress.time, "monotonic", side_effect=[100.0, 110.0]):  # Fix start and first-completed-unit timestamps in the reusable module.
            with progress:  # Start timing at the fixed initial timestamp.
                progress.report_unit(1)  # Report one real completed unit after ten seconds.
        self.assertIn("Round: 1/4 | Progress: 25.00% | Elapsed: 10s | ETA: 30s", output.getvalue())  # Require ETA derived from one completed unit only.

    def test_automl_uses_public_completed_trial_callback(self):  # Verify Optuna trial progress
        """
        Verify AutoML model search reports genuine completed Optuna trials.

        :param self: Instance of the TrainingProgressTests class.
        :return: None.
        """

        def objective(trial, X_train, y_train, cv_folds, config=None):  # Provide one lightweight deterministic Optuna objective
            """
            Return a deterministic value after selecting one model identity.

            :param trial: Active public Optuna trial.
            :param X_train: Training feature matrix.
            :param y_train: Training labels.
            :param cv_folds: Configured cross-validation fold count.
            :param config: Runtime configuration dictionary.
            :return: Deterministic trial score.
            """

            trial.suggest_categorical("model_name", ["KNN"])  # Populate the production best-model parameter.
            return float(trial.number)  # Return a deterministic increasing trial score.

        config = {"automl": {"n_trials": 3, "timeout": 30, "cv_folds": 2, "random_state": 42}, "evaluation": {"training_progress_interval_minutes": 0.01 / 60.0}}  # Configure three lightweight public trials.
        output = io.StringIO()  # Allocate isolated AutoML progress output.
        with mock.patch.object(stacking, "automl_objective", side_effect=objective):  # Replace only expensive model evaluation inside the real study flow.
            with contextlib.redirect_stdout(output):  # Capture public trial callback records.
                best_model_name, best_params, study = stacking.run_automl_model_search(self.X_train, self.y_train, "fixture.csv", config=config)  # Run the production Optuna orchestration.
        self.assertEqual(best_model_name, "KNN")  # Require the deterministic selected model identity.
        self.assertEqual(len(study.trials), 3)  # Require the configured real trial count.
        self.assertIn("Trial: 3/3 | Progress: 100.00%", output.getvalue())  # Require genuine callback completion at the exact total.
        self.assertFalse(any(thread.name.startswith("training-heartbeat-") for thread in threading.enumerate()))  # Require AutoML heartbeat cleanup after study completion.

    def test_automl_stacking_uses_public_completed_trial_callback(self):  # Verify Optuna stacking-trial progress
        """
        Verify AutoML stacking search reports genuine completed Optuna trials.

        :param self: Instance of the TrainingProgressTests class.
        :return: None.
        """

        def objective(trial, X_train, y_train, cv_folds, candidate_models, config=None):  # Provide one lightweight deterministic stacking objective
            """
            Return a deterministic stacking score after populating required public parameters.

            :param trial: Active public Optuna trial.
            :param X_train: Training feature matrix.
            :param y_train: Training labels.
            :param cv_folds: Configured cross-validation fold count.
            :param candidate_models: Candidate base-learner parameter mapping.
            :param config: Runtime configuration dictionary.
            :return: Deterministic stacking trial score.
            """

            trial.suggest_categorical("meta_learner", ["Logistic Regression"])  # Populate the production meta-learner parameter.
            trial.suggest_int("stacking_cv_splits", 2, 2)  # Populate the production stacking fold parameter.
            for candidate_name in candidate_models:  # Populate every production base-learner selection parameter.
                parameter_name = f"use_{candidate_name.replace(' ', '_').replace('(', '').replace(')', '')}"  # Mirror the production public parameter identity.
                trial.suggest_categorical(parameter_name, [True])  # Select every deterministic candidate base learner.
            return float(trial.number)  # Return a deterministic increasing stacking score.

        candidates = {"KNN": {"n_neighbors": 3}, "Decision Tree": {"max_depth": 2}}  # Provide two deterministic candidate model configurations.
        config = {"automl": {"stacking_trials": 3, "stacking_top_n": 2, "timeout": 30, "cv_folds": 2, "random_state": 42}, "evaluation": {"training_progress_interval_minutes": 0.01 / 60.0}}  # Configure three lightweight public stacking trials.
        output = io.StringIO()  # Allocate isolated stacking-search progress output.
        with mock.patch.object(stacking, "extract_top_automl_models", return_value=candidates):  # Replace model extraction with two deterministic candidates.
            with mock.patch.object(stacking, "automl_stacking_objective", side_effect=objective):  # Replace only expensive stacking evaluation inside the real study flow.
                with contextlib.redirect_stdout(output):  # Capture public stacking-trial callback records.
                    best_config, study = stacking.run_automl_stacking_search(self.X_train, self.y_train, object(), "fixture.csv", config=config)  # Run the production Optuna stacking orchestration.
        self.assertEqual(len(study.trials), 3)  # Require the configured real stacking-trial count.
        self.assertEqual(best_config["meta_learner"], "Logistic Regression")  # Require the deterministic selected meta-learner.
        self.assertIn("Trial: 3/3 | Progress: 100.00%", output.getvalue())  # Require genuine stacking callback completion at the exact total.
        self.assertFalse(any(thread.name.startswith("training-heartbeat-") for thread in threading.enumerate()))  # Require stacking-search heartbeat cleanup after study completion.

    def test_phase_records_are_newline_delimited_in_detached_log(self):  # Verify detached log readability
        """
        Verify every required phase remains newline-delimited in detached stacking.log output.

        :param self: Instance of the TrainingProgressTests class.
        :return: None.
        """

        phases = ["Model preparation", "Training", "Prediction", "Metrics", "Explainability", "Model export", "Cache persistence"]  # List every required lifecycle phase.
        with tempfile.TemporaryDirectory() as temporary_directory:  # Isolate the detached log artifact.
            log_path = Path(temporary_directory) / "stacking.log"  # Resolve the isolated production-style log path.
            detached_logger = stacking.Logger(str(log_path), clean=True)  # Create the repository's real dual-channel logger.
            original_stdout = stacking.sys.stdout  # Preserve the active test output stream.
            try:  # Restore stdout and close the logger after phase emission.
                stacking.sys.stdout = detached_logger  # Route production phase records through stacking.log.
                detached_interactive = training_progress.interactive_terminal_attached(detached_logger)  # Resolve reusable terminal behavior through the detached repository logger.
                for phase in phases:  # Emit every required lifecycle phase once.
                    stacking.log_training_phase("PCA Components", "SVM", phase, "Completed")  # Write one production phase record.
            finally:  # Restore output even if phase logging fails.
                stacking.sys.stdout = original_stdout  # Restore the active test output stream.
                detached_logger.close()  # Close the isolated log file.
            log_text = log_path.read_text(encoding="utf-8")  # Read the durable detached log output.
        for phase in phases:  # Verify every required lifecycle phase is durable.
            self.assertIn(f"Phase: {phase}", log_text)  # Require a distinct record for this phase.
        self.assertNotIn("\r", log_text)  # Forbid interactive carriage-return rendering in detached logs.
        self.assertIn("Feature Set: PCA | Classifier: SVM", log_text)  # Require concise feature-set and classifier identities.
        self.assertFalse(detached_interactive)  # Require interactive progress rendering to remain disabled for detached logging.


if __name__ == "__main__":  # Support direct focused test execution.
    unittest.main()  # Run the training progress test suite.
