if __name__ in {"__main__", "__mp_main__"}:
    try:
        from setproctitle import setproctitle
        setproctitle(f"DDoS-{__file__.rsplit('/', 1)[-1].rsplit('.', 1)[0]}")
    except ImportError:
        pass

import multiprocessing as mp  # Provide independent-process cache writer coverage.
import os  # Provide explicit process exit handling.
from pathlib import Path  # Provide deterministic temporary artifact inspection.
import tempfile  # Provide isolated cache directories for every test.
from typing import Any  # Provide explicit annotations for injected interfaces.
import unittest  # Provide the repository-independent standard test runner.
from unittest import mock  # Provide deterministic failure injection.

import numpy as np  # Provide compact evaluation-flow fixtures.
import pandas as pd  # Provide legacy cache fixtures and serialization failure injection.

import stacking  # Exercise the real production cache persistence and recovery paths.


def make_cache_result(index: int) -> dict:
    """
    Build one complete production-compatible cache result.

    :param index: Unique result index used in resume identity fields.
    :return: Complete cache result dictionary.
    """

    return {  # Return every field required by the production cache schema.
        "experiment_id": f"experiment-{index}",  # Provide stable experiment identity metadata.
        "experiment_mode": "original_only",  # Use baseline evaluation semantics.
        "execution_mode": "separate_files",  # Use single-file evaluation semantics.
        "data_source": "Original",  # Use the baseline data-source label.
        "dataset": "dataset.csv",  # Provide stable dataset identity metadata.
        "attack_types_combined": None,  # Preserve separate-files attack scope semantics.
        "augmentation_ratio": 0.0,  # Persist the explicit baseline ratio.
        "feature_selection_enabled": True,  # Mark the selected feature-set context.
        "hyperparameters_enabled": False,  # Mark default hyperparameter mode.
        "data_augmentation_enabled": False,  # Mark baseline data usage.
        "hyperparameter_mode": "Default Hyperparameters",  # Provide the resume identity label.
        "feature_set": f"Feature Set {index}",  # Provide a unique feature-set identity.
        "classifier_type": "Individual",  # Mark the classifier grouping.
        "model_name": f"Model {index}",  # Provide a unique classifier identity.
        "model": "DeterministicClassifier",  # Provide concrete estimator metadata.
        "n_features": 2,  # Provide the evaluated feature count.
        "n_samples_train": 8,  # Provide the training sample count.
        "n_samples_test": 2,  # Provide the test sample count.
        "accuracy": 0.9,  # Provide a completed accuracy metric.
        "precision": 0.8,  # Provide a completed precision metric.
        "recall": 0.7,  # Provide a completed recall metric.
        "f1_score": 0.75,  # Provide a completed F1 metric.
        "fpr": 0.1,  # Provide a completed false-positive rate.
        "fnr": 0.2,  # Provide a completed false-negative rate.
        "elapsed_time_s": float(index + 1),  # Provide deterministic elapsed-time metadata.
        "cv_method": "StratifiedKFold",  # Provide evaluation-method metadata.
        "rfe_ranking": None,  # Preserve optional RFE ranking semantics.
        "hyperparameters": {"index": index},  # Provide meaningful effective estimator parameters.
        "features_list": ["feature_a", "feature_b"],  # Provide complete ordered feature metadata.
    }  # Complete the production-compatible cache result.


def run_process_cache_writer(cache_path: str, index: int, start_event: Any) -> None:
    """
    Persist one cache result from an independent process.

    :param cache_path: Shared primary cache destination.
    :param index: Unique result index.
    :param start_event: Multiprocessing event coordinating simultaneous starts.
    :return: None.
    """

    try:  # Convert child failures into a nonzero process exit status.
        start_event.wait(10.0)  # Wait until every process has been started by the parent.
        with mock.patch.object(stacking, "get_cache_file_path", return_value=cache_path):  # Route the production writer to the shared test cache.
            with mock.patch.object(stacking, "send_exception_via_telegram"):  # Prevent external notifications during deterministic tests.
                stacking.save_cache_result_entry("dataset.csv", make_cache_result(index), config={})  # Execute the complete locked production write path.
    except Exception:  # Report independent-process persistence failure through the process exit code.
        os._exit(1)  # Exit immediately with failure for deterministic parent assertions.


class AtomicCacheRecoveryTests(unittest.TestCase):  # Group cache transaction and recovery behavior.
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()  # Allocate one isolated cache directory.
        self.addCleanup(self.temporary_directory.cleanup)  # Remove isolated test files after every test.
        self.cache_path = str(Path(self.temporary_directory.name) / "Cache_dataset-Stacking_Classifiers_Results.csv")  # Resolve the primary cache path.
        self.backup_path = stacking.get_cache_backup_path(self.cache_path)  # Resolve the sibling backup path.
        self.path_patch = mock.patch.object(stacking, "get_cache_file_path", return_value=self.cache_path)  # Route production path construction to the isolated cache.
        self.path_patch.start()  # Activate isolated cache path routing.
        self.addCleanup(self.path_patch.stop)  # Restore production path construction after every test.
        self.telegram_patch = mock.patch.object(stacking, "send_exception_via_telegram")  # Prevent external notifications during failure tests.
        self.telegram_patch.start()  # Activate notification isolation.
        self.addCleanup(self.telegram_patch.stop)  # Restore notification behavior after every test.

    def read_cache(self, path: str, config: Any = None) -> dict:
        """
        Read one validated cache artifact through production deserialization.

        :param path: Primary or backup cache path.
        :param config: Optional runtime configuration used for run-specific validation.
        :return: Production resume dictionary.
        """

        runtime_config = {} if config is None else config  # Resolve optional test config.
        _, cache_dict = stacking.read_validated_cache_file(path, config=runtime_config)
        return cache_dict  # Return recovered cache entries.

    def temporary_artifacts(self) -> list:
        """
        List incomplete cache transaction files in the isolated directory.

        :return: Paths matching the production transaction suffix.
        """

        directory = Path(self.temporary_directory.name)  # Resolve the isolated cache directory.
        return list(directory.glob(".*.tmp"))  # Return every unconsumed transaction file.

    def test_first_write_creates_valid_primary_and_backup(self) -> None:
        stacking.save_cache_result_entry("dataset.csv", make_cache_result(0), config={})  # Persist the first completed result.
        self.assertTrue(Path(self.cache_path).is_file())  # Verify primary creation.
        self.assertTrue(Path(self.backup_path).is_file())  # Verify first-write backup creation.
        primary_cache = self.read_cache(self.cache_path)  # Deserialize the primary cache.
        backup_cache = self.read_cache(self.backup_path)  # Deserialize the backup cache.
        self.assertEqual(len(primary_cache), 1)  # Verify primary deserialization.
        self.assertEqual(len(backup_cache), 1)  # Verify backup deserialization.
        self.assertEqual(set(primary_cache), set(backup_cache))  # Verify first-write logical synchronization.
        self.assertEqual(self.temporary_artifacts(), [])  # Verify successful transaction cleanup.

    def test_subsequent_write_keeps_primary_and_backup_synchronized(self) -> None:
        stacking.save_cache_result_entry("dataset.csv", make_cache_result(0), config={})  # Create the initial primary and backup.
        stacking.save_cache_result_entry("dataset.csv", make_cache_result(1), config={})  # Persist a second distinct result.
        primary_cache = self.read_cache(self.cache_path)  # Deserialize the authoritative primary.
        backup_cache = self.read_cache(self.backup_path)  # Deserialize the synchronized backup.
        self.assertEqual(len(primary_cache), 2)  # Verify the authoritative primary includes both results.
        self.assertEqual(set(primary_cache), set(backup_cache))  # Verify the backup mirrors the committed primary.
        self.assertEqual({entry["model_name"] for entry in backup_cache.values()}, {"Model 0", "Model 1"})

    def test_primary_updates_incrementally_after_each_completed_result(self) -> None:
        expected_names = set()  # Track identities expected after each atomic persistence call.
        for index in range(4):  # Persist several completed results sequentially.
            stacking.persist_cache_result_entry("dataset.csv", make_cache_result(index), {}, config={})
            expected_names.add(f"Model {index}")  # Record the newly committed identity.
            primary_cache = self.read_cache(self.cache_path)
            backup_cache = self.read_cache(self.backup_path)  # Read the backup immediately after this completed result.
            self.assertEqual({entry["model_name"] for entry in primary_cache.values()}, expected_names)
            self.assertEqual(set(primary_cache), set(backup_cache))  # Verify backup synchronization after each result.

    def test_missing_cache_reference_fails_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Cache reference file is required"):
            stacking.persist_cache_result_entry(None, make_cache_result(0), {}, config={})
        self.assertFalse(Path(self.cache_path).exists())  # Verify no cache file was created.

    def test_corrupt_primary_falls_back_to_backup(self) -> None:
        stacking.save_cache_result_entry("dataset.csv", make_cache_result(0), config={})  # Create recoverable primary and backup files.
        Path(self.cache_path).write_text("corrupt-cache", encoding="utf-8")  # Replace the primary with invalid CSV content.
        recovered = stacking.load_cache_results("dataset.csv", config={})  # Execute production backup recovery.
        self.assertEqual(len(recovered), 1)  # Verify one known-good entry was recovered.
        self.assertIn("Model 0", {entry["model_name"] for entry in recovered.values()})  # Verify recovered identity preservation.
        self.assertEqual(set(self.read_cache(self.cache_path)), set(self.read_cache(self.backup_path)))

    def test_truncated_primary_falls_back_to_backup(self) -> None:
        stacking.save_cache_result_entry("dataset.csv", make_cache_result(0), config={})  # Create recoverable primary and backup files.
        primary_bytes = Path(self.cache_path).read_bytes()  # Read the complete primary bytes.
        Path(self.cache_path).write_bytes(primary_bytes[: max(1, len(primary_bytes) // 3)])  # Truncate the primary deterministically.
        recovered = stacking.load_cache_results("dataset.csv", config={})  # Execute production backup recovery.
        self.assertEqual(len(recovered), 1)  # Verify the valid backup remains recoverable.
        self.assertEqual(set(self.read_cache(self.cache_path)), set(self.read_cache(self.backup_path)))

    def test_startup_recovery_promotes_newer_valid_backup_to_primary(self) -> None:
        primary_df = stacking.prepare_cache_dataframe(pd.DataFrame([make_cache_result(0)]), config={})
        backup_df = stacking.prepare_cache_dataframe(
            pd.DataFrame([make_cache_result(0), make_cache_result(1)]),
            config={},
        )
        primary_df.to_csv(self.cache_path, index=False)  # Publish the older primary fixture.
        backup_df.to_csv(self.backup_path, index=False)  # Publish the newer backup fixture.
        recovered = stacking.load_cache_results("dataset.csv", config={})  # Run the real startup cache loading path.
        primary_cache = self.read_cache(self.cache_path)  # Read the repaired primary.
        backup_cache = self.read_cache(self.backup_path)  # Read the synchronized backup.
        self.assertEqual({entry["model_name"] for entry in recovered.values()}, {"Model 0", "Model 1"})
        self.assertEqual(set(primary_cache), set(backup_cache))  # Verify recovery synchronized cache artifacts.
        self.assertEqual({entry["model_name"] for entry in primary_cache.values()}, {"Model 0", "Model 1"})
        recovered_again = stacking.load_cache_results("dataset.csv", config={})
        self.assertEqual(set(recovered), set(recovered_again))

    def test_startup_recovery_keeps_more_complete_primary(self) -> None:
        primary_df = stacking.prepare_cache_dataframe(
            pd.DataFrame([make_cache_result(0), make_cache_result(1)]),
            config={},
        )
        backup_df = stacking.prepare_cache_dataframe(pd.DataFrame([make_cache_result(0)]), config={})
        primary_df.to_csv(self.cache_path, index=False)  # Publish the newer primary fixture.
        backup_df.to_csv(self.backup_path, index=False)  # Publish the older backup fixture.
        recovered = stacking.load_cache_results("dataset.csv", config={})  # Run the real startup cache loading path.
        primary_cache = self.read_cache(self.cache_path)  # Read the primary after loading.
        backup_cache = self.read_cache(self.backup_path)  # Read the backup after loading.
        self.assertEqual({entry["model_name"] for entry in recovered.values()}, {"Model 0", "Model 1"})
        self.assertEqual({entry["model_name"] for entry in primary_cache.values()}, {"Model 0", "Model 1"})
        self.assertEqual({entry["model_name"] for entry in backup_cache.values()}, {"Model 0"})

    def test_startup_recovery_leaves_equivalent_files_stable(self) -> None:
        cache_df = stacking.prepare_cache_dataframe(
            pd.DataFrame([make_cache_result(0), make_cache_result(1)]),
            config={},
        )
        cache_df.to_csv(self.cache_path, index=False)  # Publish primary fixture.
        cache_df.to_csv(self.backup_path, index=False)  # Publish equivalent backup fixture.
        recovered = stacking.load_cache_results("dataset.csv", config={})  # Run the real startup cache loading path.
        self.assertEqual({entry["model_name"] for entry in recovered.values()}, {"Model 0", "Model 1"})
        self.assertEqual(set(self.read_cache(self.cache_path)), set(self.read_cache(self.backup_path)))

    def test_startup_recovery_ignores_invalid_backup(self) -> None:
        primary_df = stacking.prepare_cache_dataframe(
            pd.DataFrame([make_cache_result(0), make_cache_result(1)]),
            config={},
        )
        primary_df.to_csv(self.cache_path, index=False)  # Publish the valid primary fixture.
        Path(self.backup_path).write_text("invalid-backup", encoding="utf-8")  # Publish an invalid backup fixture.
        recovered = stacking.load_cache_results("dataset.csv", config={})  # Run the real startup cache loading path.
        primary_cache = self.read_cache(self.cache_path)  # Read the authoritative primary after loading.
        self.assertEqual({entry["model_name"] for entry in recovered.values()}, {"Model 0", "Model 1"})
        self.assertEqual({entry["model_name"] for entry in primary_cache.values()}, {"Model 0", "Model 1"})

    def test_startup_recovery_merges_divergent_valid_content(self) -> None:
        primary_df = stacking.prepare_cache_dataframe(
            pd.DataFrame([make_cache_result(0), make_cache_result(2)]),
            config={},
        )
        backup_df = stacking.prepare_cache_dataframe(
            pd.DataFrame([make_cache_result(0), make_cache_result(1)]),
            config={},
        )
        primary_df.to_csv(self.cache_path, index=False)  # Publish divergent primary fixture.
        backup_df.to_csv(self.backup_path, index=False)  # Publish divergent backup fixture.
        recovered = stacking.load_cache_results("dataset.csv", config={})  # Run the real startup cache loading path.
        primary_cache = self.read_cache(self.cache_path)  # Read the merged primary.
        backup_cache = self.read_cache(self.backup_path)  # Read the merged backup.
        self.assertEqual({entry["model_name"] for entry in recovered.values()}, {"Model 0", "Model 1", "Model 2"})
        self.assertEqual(set(primary_cache), set(backup_cache))  # Verify divergent recovery synchronized artifacts.

    def test_both_invalid_files_follow_cache_miss_behavior(self) -> None:
        Path(self.cache_path).write_text("invalid-primary", encoding="utf-8")  # Create an invalid primary cache artifact.
        Path(self.backup_path).write_text("invalid-backup", encoding="utf-8")  # Create an invalid backup cache artifact.
        with mock.patch("builtins.print") as print_mock:  # Capture recovery diagnostics without relying on terminal text output.
            recovered = stacking.load_cache_results("dataset.csv", config={})  # Attempt production recovery from both invalid sources.
        self.assertEqual(recovered, {})  # Verify established cache-miss behavior.
        printed_text = " ".join(str(call) for call in print_mock.call_args_list)  # Flatten diagnostics for one factual assertion.
        self.assertIn("No valid cache data could be recovered", printed_text)  # Verify recovery is not claimed silently.

    def test_unreadable_primary_does_not_overwrite_valid_backup(self) -> None:
        stacking.save_cache_result_entry("dataset.csv", make_cache_result(0), config={})  # Create recoverable primary and backup files.
        original_reader = stacking.read_validated_cache_file  # Preserve the real validator for non-primary paths.

        def deny_primary(path: str, config: Any = None, **kwargs: Any) -> Any:
            if path == self.cache_path:  # Inject deterministic primary unreadability.
                raise PermissionError("Injected primary read failure")  # Simulate an unreadable primary cache.
            return original_reader(path, config=config, **kwargs)  # Preserve backup and temporary validation behavior.

        with mock.patch.object(stacking, "read_validated_cache_file", side_effect=deny_primary):  # Activate primary read failure during one update.
            stacking.save_cache_result_entry("dataset.csv", make_cache_result(1), config={})  # Merge the new result from the valid backup.
        primary_cache = self.read_cache(self.cache_path)  # Read the recovered primary.
        backup_cache = self.read_cache(self.backup_path)  # Read the synchronized backup.
        self.assertEqual(len(primary_cache), 2)  # Verify recovered entries were preserved in the new primary.
        self.assertEqual(set(primary_cache), set(backup_cache))  # Verify backup synchronization after primary recovery.

    def test_failed_temporary_write_preserves_primary_and_backup(self) -> None:
        stacking.save_cache_result_entry("dataset.csv", make_cache_result(0), config={})  # Create the known-good baseline files.
        primary_before = Path(self.cache_path).read_bytes()  # Snapshot the primary before injected failure.
        backup_before = Path(self.backup_path).read_bytes()  # Snapshot the backup before injected failure.
        with mock.patch.object(pd.DataFrame, "to_csv", side_effect=OSError("Injected serialization failure")):  # Fail the first temporary serialization deterministically.
            with self.assertRaises(OSError):  # Require the original serialization error to surface.
                stacking.save_cache_result_entry("dataset.csv", make_cache_result(1), config={})  # Attempt the failed transaction.
        self.assertEqual(Path(self.cache_path).read_bytes(), primary_before)  # Verify primary preservation.
        self.assertEqual(Path(self.backup_path).read_bytes(), backup_before)  # Verify backup preservation.
        self.assertEqual(self.temporary_artifacts(), [])  # Verify failed transaction cleanup.

    def test_failed_validation_preserves_primary_and_backup(self) -> None:
        stacking.save_cache_result_entry("dataset.csv", make_cache_result(0), config={})  # Create the known-good baseline files.
        primary_before = Path(self.cache_path).read_bytes()  # Snapshot the primary before injected failure.
        backup_before = Path(self.backup_path).read_bytes()  # Snapshot the backup before injected failure.
        original_reader = stacking.read_validated_cache_file  # Preserve validation for existing final paths.

        def reject_temporary(path: str, config: Any = None, **kwargs: Any) -> Any:
            if str(path).endswith(".tmp"):  # Inject failure only after temporary serialization.
                raise ValueError("Injected temporary validation failure")  # Reject the staged cache deterministically.
            return original_reader(path, config=config, **kwargs)  # Preserve existing primary and backup validation.

        with mock.patch.object(stacking, "read_validated_cache_file", side_effect=reject_temporary):  # Activate temporary validation failure.
            with self.assertRaises(ValueError):  # Require the original validation error to surface.
                stacking.save_cache_result_entry("dataset.csv", make_cache_result(1), config={})  # Attempt the failed transaction.
        self.assertEqual(Path(self.cache_path).read_bytes(), primary_before)  # Verify primary preservation.
        self.assertEqual(Path(self.backup_path).read_bytes(), backup_before)  # Verify backup preservation.
        self.assertEqual(self.temporary_artifacts(), [])  # Verify rejected temporary cleanup.

    def test_failed_primary_replacement_does_not_advance_backup(self) -> None:
        stacking.save_cache_result_entry("dataset.csv", make_cache_result(0), config={})  # Create the known-good baseline files.
        stacking.save_cache_result_entry("dataset.csv", make_cache_result(1), config={})  # Make the primary newer than its known-good backup.
        primary_before = Path(self.cache_path).read_bytes()  # Snapshot the primary before injected failure.
        backup_before = Path(self.backup_path).read_bytes()  # Snapshot the backup before injected failure.
        original_replace = stacking.replace_cache_file_atomically

        def fail_primary_replace(source: str, destination: str) -> None:
            if destination == self.cache_path:  # Inject failure only when publishing the new primary.
                raise OSError("Injected primary replacement failure")  # Simulate an atomic primary replacement failure.
            original_replace(source, destination)  # Preserve non-primary publication behavior.

        with mock.patch.object(stacking, "replace_cache_file_atomically", side_effect=fail_primary_replace):  # Activate primary replacement failure.
            with self.assertRaises(OSError):  # Require the original replacement error to surface.
                stacking.save_cache_result_entry("dataset.csv", make_cache_result(2), config={})  # Attempt the failed transaction.
        self.assertEqual(Path(self.cache_path).read_bytes(), primary_before)  # Verify primary preservation.
        self.assertEqual(Path(self.backup_path).read_bytes(), backup_before)
        self.assertEqual(self.temporary_artifacts(), [])  # Verify replacement failure cleanup.

    def test_failed_backup_replacement_preserves_recoverable_files(self) -> None:
        stacking.save_cache_result_entry("dataset.csv", make_cache_result(0), config={})  # Create the known-good baseline files.
        backup_before = Path(self.backup_path).read_bytes()  # Snapshot the backup before injected failure.
        original_replace = stacking.replace_cache_file_atomically  # Preserve real atomic replacement for non-backup destinations.

        def fail_backup_replace(source: str, destination: str) -> None:
            if destination == self.backup_path:  # Inject failure only when rotating the known-good backup.
                raise OSError("Injected backup replacement failure")  # Simulate an atomic backup replacement failure.
            original_replace(source, destination)  # Preserve other atomic replacement behavior.

        with mock.patch.object(stacking, "replace_cache_file_atomically", side_effect=fail_backup_replace):  # Activate backup replacement failure.
            with self.assertRaises(OSError):  # Require the original replacement error to surface.
                stacking.save_cache_result_entry("dataset.csv", make_cache_result(1), config={})  # Attempt the failed transaction.
        self.assertEqual(
            {entry["model_name"] for entry in self.read_cache(self.cache_path).values()},
            {"Model 0", "Model 1"},
        )
        self.assertEqual(Path(self.backup_path).read_bytes(), backup_before)  # Verify known-good backup preservation.
        self.assertEqual(self.temporary_artifacts(), [])  # Verify backup replacement failure cleanup.

    def test_existing_legacy_cache_remains_compatible(self) -> None:
        legacy_result = make_cache_result(0)  # Build one complete row before removing newer context fields.
        legacy_result.pop("feature_selection_enabled")  # Remove a field supplied by schema migration.
        legacy_result.pop("hyperparameters_enabled")  # Remove a field supplied by hyperparameter-mode migration.
        legacy_result.pop("data_augmentation_enabled")  # Remove a field supplied by augmentation migration.
        pd.DataFrame([legacy_result]).to_csv(self.cache_path, index=False)  # Write the legacy-compatible CSV directly.
        recovered = stacking.load_cache_results("dataset.csv", config={})  # Load through production schema migration.
        self.assertEqual(len(recovered), 1)  # Verify legacy cache compatibility.
        self.assertIn("Model 0", {entry["model_name"] for entry in recovered.values()})  # Verify legacy identity preservation.

    def test_recovered_entries_are_not_lost_during_next_write(self) -> None:
        stacking.save_cache_result_entry("dataset.csv", make_cache_result(0), config={})  # Create recoverable primary and backup files.
        Path(self.cache_path).write_text("invalid-primary", encoding="utf-8")  # Corrupt only the primary before the next write.
        stacking.save_cache_result_entry("dataset.csv", make_cache_result(1), config={})  # Merge from backup and publish a new primary.
        primary_cache = self.read_cache(self.cache_path)  # Deserialize the recovered authoritative primary.
        backup_cache = self.read_cache(self.backup_path)  # Deserialize the synchronized backup.
        self.assertEqual(len(primary_cache), 2)  # Verify both recovered and new entries are present.
        self.assertEqual({entry["model_name"] for entry in primary_cache.values()}, {"Model 0", "Model 1"})  # Verify no logical update was lost.
        self.assertEqual(set(primary_cache), set(backup_cache))  # Verify backup synchronized after recovery write.

    def test_backup_only_valid_rows_are_recovered_with_valid_primary(self) -> None:
        primary_df = stacking.prepare_cache_dataframe(pd.DataFrame([make_cache_result(0)]), config={})  # Build valid primary rows.
        backup_df = stacking.prepare_cache_dataframe(pd.DataFrame([make_cache_result(0), make_cache_result(1)]), config={})  # Build valid backup rows with one extra identity.
        primary_df.to_csv(self.cache_path, index=False)  # Publish the valid but older primary fixture.
        backup_df.to_csv(self.backup_path, index=False)  # Publish the valid newer backup fixture.
        recovered = stacking.load_cache_results("dataset.csv", config={})  # Load through production primary and backup recovery.
        self.assertEqual({entry["model_name"] for entry in recovered.values()}, {"Model 0", "Model 1"})  # Verify backup-only identity recovery.
        stacking.save_cache_result_entry("dataset.csv", make_cache_result(2), config={})  # Merge a new result after recovery.
        primary_cache = self.read_cache(self.cache_path)  # Read the final authoritative primary.
        backup_cache = self.read_cache(self.backup_path)  # Read the final synchronized backup.
        self.assertEqual({entry["model_name"] for entry in primary_cache.values()}, {"Model 0", "Model 1", "Model 2"})  # Verify writer union preservation.
        self.assertEqual(set(primary_cache), set(backup_cache))  # Verify backup mirrors the recovered primary.

    def test_stale_in_memory_cache_does_not_skip_disk_persistence(self) -> None:
        stale_entry = make_cache_result(0)  # Build one completed result entry.
        stale_key = stacking.build_cache_identity_from_row(stale_entry)  # Build the production resume identity.
        stale_cache = {stale_key: stale_entry}  # Simulate worker memory that believes the row already exists.
        stacking.persist_cache_result_entry("dataset.csv", stale_entry, stale_cache, config={})  # Persist through the public atomic result path.
        recovered = stacking.load_cache_results("dataset.csv", config={})  # Reload from disk through production resume.
        self.assertIn(stale_key, recovered)  # Verify stale memory did not suppress persistence.

    def test_duplicate_completed_result_does_not_create_duplicate_rows(self) -> None:
        result_entry = make_cache_result(0)  # Build one completed result entry.
        stacking.persist_cache_result_entry("dataset.csv", result_entry, {}, config={})  # Persist the result once.
        stacking.persist_cache_result_entry("dataset.csv", dict(result_entry), {}, config={})
        primary_cache = self.read_cache(self.cache_path)  # Read the primary after duplicate persistence.
        backup_cache = self.read_cache(self.backup_path)  # Read the backup after duplicate persistence.
        self.assertEqual(len(primary_cache), 1)  # Verify duplicate identity collapsed to one row.
        self.assertEqual(set(primary_cache), set(backup_cache))  # Verify synchronized deduplicated cache artifacts.

    def test_run_specific_cache_files_remain_isolated(self) -> None:
        run_paths = {
            run_index: str(
                Path(self.temporary_directory.name) / f"Cache_dataset-Stacking_Classifiers_Results_Run_{run_index}.csv"
            )
            for run_index in (1, 2)
        }

        def run_cache_path(csv_path: str, config: Any = None, experiment_run: Any = None, legacy: bool = False) -> str:
            run_index = int(experiment_run or stacking.get_current_experiment_run(config))
            return run_paths[run_index]  # Return the destination for the requested run.

        with mock.patch.object(stacking, "get_cache_file_path", side_effect=run_cache_path):
            run_one = {"stacking": {"experiment_run": 1}}  # Build run 1 configuration.
            run_two = {"stacking": {"experiment_run": 2}}  # Build run 2 configuration.
            run_one_result = make_cache_result(0)  # Build run 1 result data.
            run_two_result = make_cache_result(1)  # Build run 2 result data.
            run_one_result["experiment_run"] = 1  # Match the run 1 destination.
            run_two_result["experiment_run"] = 2  # Match the run 2 destination.
            stacking.persist_cache_result_entry("dataset.csv", run_one_result, {}, config=run_one)
            stacking.persist_cache_result_entry("dataset.csv", run_two_result, {}, config=run_two)
            run_one_cache = self.read_cache(run_paths[1], config=run_one)  # Read run 1 primary.
            run_two_cache = self.read_cache(run_paths[2], config=run_two)  # Read run 2 primary.
            self.assertEqual({entry["model_name"] for entry in run_one_cache.values()}, {"Model 0"})
            self.assertEqual({entry["model_name"] for entry in run_two_cache.values()}, {"Model 1"})
            self.assertEqual(
                set(run_one_cache),
                set(self.read_cache(stacking.get_cache_backup_path(run_paths[1]), config=run_one)),
            )
            self.assertEqual(
                set(run_two_cache),
                set(self.read_cache(stacking.get_cache_backup_path(run_paths[2]), config=run_two)),
            )

    def test_artifact_recovery_target_persists_to_primary_and_backup(self) -> None:
        config = {
            "stacking": {"methods": {"feature_selection": False, "stacking": False}},
            "evaluation": {"random_state": 42},
            "explainability": {"enabled": False},
        }
        dataframe = pd.DataFrame(
            {
                "feature_a": [0.0, 1.0, 2.0, 3.0],
                "feature_b": [1.0, 2.0, 3.0, 4.0],
                "attack_type": ["benign", "attack", "benign", "attack"],
            }
        )
        progress_bar = mock.Mock()  # Provide the minimal progress API used by evaluate_on_dataset.
        splits = (
            np.zeros((2, 2)),
            np.ones((2, 2)),
            np.array([0, 1]),
            np.array([0, 1]),
            mock.Mock(),
            mock.Mock(classes_=np.array(["attack", "benign"])),
            None,
            None,
        )

        def feature_sets(*args: Any, **kwargs: Any) -> Any:
            yield ("Feature Set 0", np.zeros((2, 2)), np.ones((2, 2)), ["feature_a", "feature_b"], None)

        def persist_recovered_classifier(*args: Any, **kwargs: Any) -> Any:
            result_entry = make_cache_result(0)  # Build the recovered completed result.
            stacking.persist_cache_result_entry(
                kwargs["cache_ref_file"],
                result_entry,
                kwargs["cache_dict"],
                config=kwargs["config"],
            )
            return ({("Feature Set 0", "Model 0"): result_entry}, 2)

        with mock.patch.object(stacking, "sanitize_and_verify_feature_selections", return_value=(None, None, None)):
            with mock.patch.object(stacking, "prepare_evaluation_data_splits", return_value=splits):
                with mock.patch.object(stacking, "iterate_feature_sets_sequentially", side_effect=feature_sets):
                    with mock.patch.object(stacking, "tqdm", return_value=progress_bar):
                        with mock.patch.object(
                            stacking,
                            "run_individual_classifiers_for_feature_set",
                            side_effect=persist_recovered_classifier,
                        ):
                            stacking.evaluate_on_dataset(
                                "combined_files_combined",
                                dataframe,
                                ["feature_a", "feature_b"],
                                None,
                                None,
                                None,
                                {"Model 0": object()},
                                data_source_label="Original",
                                hyperparams_map={},
                                experiment_id="artifact-recovery",
                                experiment_mode="original_only",
                                augmentation_ratio=None,
                                execution_mode_str="combined_files",
                                attack_types_combined=["DDoS"],
                                config=config,
                                cache_ref_file="dataset.csv",
                                hyperparameters_enabled=False,
                                artifact_recovery_target=("Feature Set 0", "Model 0"),
                            )
        primary_cache = self.read_cache(self.cache_path)  # Read the authoritative primary after artifact recovery.
        backup_cache = self.read_cache(self.backup_path)  # Read the synchronized backup after artifact recovery.
        self.assertEqual({entry["model_name"] for entry in primary_cache.values()}, {"Model 0"})
        self.assertEqual(set(primary_cache), set(backup_cache))  # Verify recovered result reached backup too.

    @unittest.skipUnless("fork" in mp.get_all_start_methods(), "POSIX fork context is required by the production fcntl backend")  # Limit independent-process coverage to the supported production platform.
    def test_independent_process_updates_preserve_all_entries(self) -> None:
        process_context = mp.get_context("fork")  # Use independent processes sharing only the filesystem cache state.
        start_event = process_context.Event()  # Coordinate simultaneous writer starts deterministically.
        processes = [process_context.Process(target=run_process_cache_writer, args=(self.cache_path, index, start_event)) for index in range(6)]  # Create distinct cache writers.
        for process in processes:  # Start every writer before releasing the synchronization event.
            process.start()  # Launch one independent cache writer.
        start_event.set()  # Release every writer into the locked read-modify-write path.
        for process in processes:  # Await deterministic completion of every writer.
            process.join(30.0)  # Bound child completion time.
            self.assertEqual(process.exitcode, 0)  # Require every independent writer to succeed.
        primary_cache = self.read_cache(self.cache_path)  # Validate and deserialize the final primary.
        backup_cache = self.read_cache(self.backup_path)  # Validate and deserialize the final backup.
        self.assertEqual(len(primary_cache), 6)  # Verify no unrelated concurrent update was lost.
        self.assertEqual(len(backup_cache), 6)  # Verify the backup mirrors the final committed primary snapshot.
        self.assertEqual(set(primary_cache), set(backup_cache))
        self.assertEqual({entry["model_name"] for entry in primary_cache.values()}, {f"Model {index}" for index in range(6)})  # Verify every concurrent identity survived.
        self.assertEqual(self.temporary_artifacts(), [])  # Verify concurrent transaction cleanup.


if __name__ == "__main__":  # Support direct focused test execution.
    unittest.main()  # Run the cache recovery test suite.
