import datetime as datetime_module  # Build deterministic timezone-aware wall-clock fixtures
import multiprocessing as mp  # Exercise the production spawned feature-worker logger initialization
import re  # Validate the exact timestamp prefix and duplicate count
import sys  # Restore process-global streams after production logger initialization
import tempfile  # Isolate every focused log artifact
import traceback  # Emit a real multiline failure record through the runtime logger
import unittest  # Provide repository-standard focused test execution
import warnings  # Emit one repository-path warning through redirected standard error
from pathlib import Path  # Resolve isolated stacking.log paths
from unittest import mock  # Isolate the original terminal and Telegram transport

import stacking  # Exercise coordinator, feature-worker, lifecycle, and Telegram runtime paths
import training_progress  # Control the monotonic clock used by recurring training output
from Logger import Logger, SAO_PAULO_TIMEZONE_NAME  # Exercise the centralized process-safe timestamp boundary
from training_progress import TrainingProgress  # Emit real heartbeat and unit-based training records


FIXED_UTC_TIME = datetime_module.datetime(2026, 7, 12, 2, 11, 32, tzinfo=datetime_module.timezone.utc)  # Convert deterministically to 11/07/2026 23:11:32 in São Paulo
EXPECTED_PREFIX = "11/07/2026 - 23h11m32s: "  # Define the exact expected Brazilian timestamp fixture
TIMESTAMP_PREFIX_PATTERN = re.compile(r"^\d{2}/\d{2}/\d{4} - \d{2}h\d{2}m\d{2}s: ")  # Match one exact zero-padded prefix


def fixed_sao_paulo_time(zone):  # Return the deterministic fixture converted through the supplied explicit timezone
    return FIXED_UTC_TIME.astimezone(zone)  # Convert from UTC without consulting the host timezone


def run_timestamp_feature_worker(logs_dir, feature_set, release_event):  # Emit concurrent records through production worker initialization
    stacking.initialize_feature_process_logger({"paths": {"logs_dir": logs_dir}})  # Install the real append-only feature-worker logger
    stacking.logger.timestamp_now = fixed_sao_paulo_time  # Inject a deterministic emission clock after production initialization
    release_event.wait(10.0)  # Align worker writes closely enough to exercise shared flock serialization
    print(f"[WORKER] Feature Set: {feature_set} | Status: Started")  # Emit one worker startup record
    print(f"[TRAINING] Feature Set: {feature_set} | Classifier: Probe | Status: Active | PID: {stacking.os.getpid()}")  # Emit one heartbeat-shaped worker record
    print(f"[WORKER] Feature Set: {feature_set} | Result: Complete")  # Emit one result-shaped worker record
    print(f"[WORKER] Feature Set: {feature_set} | Status: Shutdown")  # Emit one worker shutdown record
    stacking.logger.flush()  # Flush every child record before process exit
    stacking.logger.close()  # Close the child file descriptor after its complete records are durable


class StackingLogTimestampTests(unittest.TestCase):  # Verify centralized Brazilian timestamps without scientific changes
    """Verify formatting, runtime coverage, concurrency, multiline output, and Telegram isolation."""

    def test_clean_coordinator_cannot_overwrite_appended_worker_record(self):  # Reproduce stale coordinator offsets against one shared log
        with tempfile.TemporaryDirectory() as temporary_directory:  # Isolate the shared append destination
            log_path = Path(temporary_directory) / "shared.log"  # Resolve one coordinator and worker log path
            coordinator = Logger(str(log_path), clean=True)  # Open the long-lived coordinator descriptor first
            coordinator.write("coordinator-start")  # Advance the coordinator file position before worker output
            worker = Logger(str(log_path), clean=False)  # Open the worker descriptor against the same file
            worker.write("worker-start")  # Append one record while the coordinator remains open
            worker.close()  # Close the simulated worker descriptor before coordinator resumes
            coordinator.write("coordinator-finish")  # Append after worker output without overwriting it
            coordinator.close()  # Flush and close the coordinator descriptor before reading
            lines = log_path.read_text(encoding="utf-8").splitlines()  # Read exact durable record order
        self.assertEqual(lines, ["coordinator-start", "worker-start", "coordinator-finish"])  # Preserve every shared record exactly once

    def test_exact_timezone_conversion_padding_multiline_and_duplicate_prevention(self):  # Verify direct centralized formatting semantics
        observed_zones = []  # Capture the explicit zone passed at each emission

        def recording_clock(zone):  # Record the production-selected timezone while returning a deterministic instant
            observed_zones.append(zone.key)  # Preserve the exact IANA identity for assertion
            return FIXED_UTC_TIME.astimezone(zone)  # Convert UTC through the explicit zone independently of host settings

        with tempfile.TemporaryDirectory() as temporary_directory:  # Isolate the durable log destination
            log_path = Path(temporary_directory) / "timestamp.log"  # Build the focused log path
            logger = Logger(str(log_path), clean=True, timestamp_timezone=SAO_PAULO_TIMEZONE_NAME, timestamp_now=recording_clock)  # Enable deterministic São Paulo formatting
            with mock.patch.object(sys, "__stdout__", None):  # Suppress terminal mirroring while preserving file output
                logger.write("Normal stacking record")  # Emit one ordinary single-line record
                logger.write("\033[H\033[J\033[92mColored startup\033[0m")  # Preserve leading terminal controls before the visible prefix
                logger.write(f"{EXPECTED_PREFIX}Already timestamped")  # Verify exact existing prefixes are never duplicated
                logger.write("Traceback (most recent call last):\n  File \"stacking.py\", line 1\n\nRuntimeError: failure")  # Emit one readable multiline traceback with a blank separator
            logger.close()  # Close the isolated log before reading it
            logged_text = log_path.read_text(encoding="utf-8")  # Read the ANSI-clean durable representation
        expected_text = f"{EXPECTED_PREFIX}Normal stacking record\n{EXPECTED_PREFIX}Colored startup\n{EXPECTED_PREFIX}Already timestamped\n{EXPECTED_PREFIX}Traceback (most recent call last):\n{EXPECTED_PREFIX}  File \"stacking.py\", line 1\n\n{EXPECTED_PREFIX}RuntimeError: failure\n"  # Build exact non-empty-line prefix expectations
        self.assertEqual(logged_text, expected_text)  # Require exact prefix placement and readable blank traceback separation
        self.assertEqual(observed_zones, [SAO_PAULO_TIMEZONE_NAME] * 4)  # Use America/Sao_Paulo independently for every emitted record
        self.assertEqual(logged_text.count(f"{EXPECTED_PREFIX}Already timestamped"), 1)  # Preserve exactly one prefix on preformatted input

        padded_utc_time = datetime_module.datetime(2026, 1, 2, 4, 5, 9, tzinfo=datetime_module.timezone.utc)  # Build a fixture requiring zero padding after conversion
        with tempfile.TemporaryDirectory() as temporary_directory:  # Isolate the zero-padding artifact
            log_path = Path(temporary_directory) / "padding.log"  # Build a second focused path
            logger = Logger(str(log_path), clean=True, timestamp_timezone=SAO_PAULO_TIMEZONE_NAME, timestamp_now=lambda zone: padded_utc_time.astimezone(zone))  # Convert the zero-padding fixture explicitly
            with mock.patch.object(sys, "__stdout__", None):  # Suppress focused terminal mirroring
                logger.write("Padded")  # Emit the zero-padding fixture record
            logger.close()  # Close the focused logger before reading
            padded_text = log_path.read_text(encoding="utf-8")  # Read the exact formatted output
        self.assertEqual(padded_text, "02/01/2026 - 01h05m09s: Padded\n")  # Require two digits for day, month, hour, minute, and second

    def test_coordinator_and_all_feature_workers_share_atomic_timestamped_output(self):  # Verify parent and four spawned worker coverage
        with tempfile.TemporaryDirectory() as temporary_directory:  # Share one isolated stacking.log across parent and children
            config = {"paths": {"logs_dir": temporary_directory}, "logging": {"clean": True}}  # Build the minimal production logger configuration
            original_stdout = sys.stdout  # Preserve the test runner output stream
            original_stderr = sys.stderr  # Preserve the test runner error stream
            original_logger = stacking.logger  # Preserve the imported module logger state
            try:  # Restore process-global state even if an assertion fails
                stacking.initialize_logger(config=config)  # Install the real coordinator logger
                self.assertEqual(stacking.logger.timestamp_zone.key, SAO_PAULO_TIMEZONE_NAME)  # Require explicit São Paulo configuration in the coordinator
                stacking.logger.timestamp_now = fixed_sao_paulo_time  # Inject the deterministic coordinator emission clock
                print("[COORDINATOR] Status: Started")  # Emit one coordinator startup record through redirected stdout
                stacking.logger.flush()  # Flush the coordinator record before child startup
                stacking.logger.close()  # Release the coordinator descriptor before restoring test streams
            finally:  # Restore test runner streams after production initialization
                sys.stdout = original_stdout  # Restore standard output
                sys.stderr = original_stderr  # Restore standard error
                stacking.logger = original_logger  # Restore imported module logger state
            process_context = mp.get_context("spawn")  # Use the production persistent-worker start method
            release_event = process_context.Event()  # Coordinate concurrent worker emission without a timing thread
            feature_sets = ("Full Features", "GA Features", "PCA Components", "RFE Features")  # Cover every persistent feature worker identity
            processes = [process_context.Process(target=run_timestamp_feature_worker, args=(temporary_directory, feature_set, release_event)) for feature_set in feature_sets]  # Build exactly four focused workers
            for process in processes:  # Start every feature worker before releasing writes
                process.start()  # Spawn one clean interpreter using production initialization
            release_event.set()  # Release all feature workers to write concurrently
            for process in processes:  # Reap every feature worker deterministically
                process.join(60.0)  # Allow dependency imports and synchronized output to complete
                self.assertEqual(process.exitcode, 0)  # Require successful worker initialization, output, and shutdown
            log_path = Path(temporary_directory) / "stacking.log"  # Resolve the production log filename
            lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line]  # Read non-empty physical records only
        self.assertEqual(len(lines), 17)  # Preserve one coordinator and four complete records from each of four workers
        self.assertTrue(all(TIMESTAMP_PREFIX_PATTERN.match(line) for line in lines))  # Timestamp every parent and child record
        self.assertTrue(all(len(TIMESTAMP_PREFIX_PATTERN.findall(line)) == 1 for line in lines))  # Attach exactly one prefix to every complete record
        self.assertIn(f"{EXPECTED_PREFIX}[COORDINATOR] Status: Started", lines)  # Cover coordinator startup logging
        for feature_set in feature_sets:  # Verify every feature-set process identity independently
            self.assertTrue(any(f"Feature Set: {feature_set} | Status: Started" in line for line in lines))  # Cover feature-worker startup
            self.assertTrue(any(f"Feature Set: {feature_set} | Status: Shutdown" in line for line in lines))  # Cover feature-worker shutdown

    def test_progress_result_cache_persistence_and_failure_records_use_one_boundary(self):  # Verify representative runtime categories converge on Logger
        with tempfile.TemporaryDirectory() as temporary_directory:  # Isolate runtime-category output
            log_path = Path(temporary_directory) / "runtime.log"  # Build the focused runtime log path
            logger = Logger(str(log_path), clean=True, timestamp_timezone=SAO_PAULO_TIMEZONE_NAME, timestamp_now=fixed_sao_paulo_time)  # Install the centralized deterministic boundary
            heartbeat = TrainingProgress("Full Features", "Random Forest", lambda seconds: f"{int(seconds)}s", output_stream=logger, report_interval_seconds=1.0)  # Build one heartbeat-capable task timer
            with mock.patch.object(training_progress.time, "monotonic", side_effect=[100.0, 101.0]):  # Advance exactly one independent heartbeat interval
                with heartbeat:  # Start and reset one classifier task timer
                    self.assertTrue(heartbeat.report_heartbeat())  # Emit the actual recurring active record
            round_progress = TrainingProgress("GA Features", "XGBoost", lambda seconds: f"{int(seconds)}s", output_stream=logger, total_units=2, unit_label="Round", report_interval_seconds=1.0)  # Build one round-based reporter
            with mock.patch.object(training_progress.time, "monotonic", side_effect=[200.0, 201.0]):  # Advance exactly one round-report interval
                with round_progress:  # Start one independent round task timer
                    round_progress.report_unit(1)  # Emit the latest actual round state
            iteration_progress = TrainingProgress("PCA Components", "LightGBM", lambda seconds: f"{int(seconds)}s", output_stream=logger, total_units=2, unit_label="Iteration", report_interval_seconds=1.0)  # Build one iteration-based reporter
            with mock.patch.object(training_progress.time, "monotonic", side_effect=[300.0, 301.0]):  # Advance exactly one iteration-report interval
                with iteration_progress:  # Start one independent iteration task timer
                    iteration_progress.report_unit(1)  # Emit the latest actual iteration state
            original_stdout = sys.stdout  # Preserve the test runner stream before runtime phase calls
            original_stderr = sys.stderr  # Preserve the test runner error stream
            try:  # Restore process-global streams after representative records
                sys.stdout = logger  # Route stacking lifecycle records through the production boundary
                sys.stderr = logger  # Route traceback output through the same boundary
                stacking.log_training_phase("RFE Features", "KNN", "Metrics", "Completed")  # Emit a real training lifecycle record
                print("KNN: Mode CombinedFiles | F1-Score 0.81 | Total Time: 4s")  # Emit an evaluation result summary
                print("Loaded cached results from: /tmp/cache.csv")  # Emit a cache record
                print("Persistence completed: /tmp/model.joblib")  # Emit a persistence and model-export record
                with warnings.catch_warnings():  # Isolate warning filter changes inside this focused record
                    warnings.simplefilter("always")  # Ensure the deterministic warning is emitted
                    warnings.warn("Injected runtime warning", RuntimeWarning)  # Route one warning through redirected standard error
                try:  # Build a real traceback without affecting test outcome
                    raise RuntimeError("Injected runtime failure")  # Raise one deterministic failure
                except RuntimeError:  # Route the full multiline failure through redirected stderr
                    traceback.print_exc()  # Emit the actual traceback record
            finally:  # Restore test streams before assertions
                sys.stdout = original_stdout  # Restore standard output
                sys.stderr = original_stderr  # Restore standard error
                logger.close()  # Close the focused runtime log
            lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line]  # Read every non-empty runtime physical line
        self.assertTrue(all(line.startswith(EXPECTED_PREFIX) for line in lines))  # Prefix heartbeats, units, phases, summaries, cache, persistence, and failures uniformly
        self.assertTrue(any("Status: Active" in line for line in lines))  # Cover recurring heartbeat output
        self.assertTrue(any("Round: 1/2" in line for line in lines))  # Cover XGBoost-style round progress
        self.assertTrue(any("Iteration: 1/2" in line for line in lines))  # Cover LightGBM-style iteration progress
        self.assertTrue(any("F1-Score 0.81" in line for line in lines))  # Cover evaluation result summaries
        self.assertTrue(any("Loaded cached results" in line for line in lines))  # Cover cache output
        self.assertTrue(any("Persistence completed" in line for line in lines))  # Cover persistence and model output
        self.assertTrue(any("Injected runtime warning" in line for line in lines))  # Cover warnings emitted through the repository stream
        self.assertTrue(any("Traceback (most recent call last)" in line for line in lines))  # Preserve readable multiline failures

    def test_telegram_payload_remains_unmodified(self):  # Verify logger timestamps never enter Telegram message construction
        task = {"global_id": 4, "hyperparameters_enabled": False, "augmentation_ratio": None, "experiment_mode": "original", "execution_mode": "combined_files", "feature_set": "GA Features", "classifier_name": "Random Forest"}  # Build one authoritative completion identity
        result_entry = {"feature_set": "GA Features", "hyperparameter_mode": "Default Hyperparameters", "model_name": "Random Forest", "execution_mode": "combined_files", "experiment_mode": "original", "augmentation_ratio": None, "f1_score": 0.75, "elapsed_time_s": 65}  # Build one persisted scalar result
        with mock.patch.object(stacking, "send_telegram_message") as telegram_send:  # Capture the established Telegram body before transport formatting
            self.assertTrue(stacking.send_feature_process_result_notification(task, result_entry, "computed", 240, set()))  # Send one application-level completion attempt
        telegram_body = telegram_send.call_args.args[1]  # Read the untouched Telegram message content
        self.assertEqual(telegram_body, "Finished combination 4/240: GA Features - Default Hyperparameters - Original Test Data - Random Forest with F1: 0.75 in 1m 5s")  # Preserve exact preexisting Telegram wording
        self.assertIsNone(TIMESTAMP_PREFIX_PATTERN.match(telegram_body))  # Exclude the new runtime-log prefix from Telegram payloads


if __name__ == "__main__":  # Support direct focused execution
    unittest.main()  # Run the timestamp-focused suite
