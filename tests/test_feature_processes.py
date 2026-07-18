import multiprocessing as mp  # Inspect child cleanup after process scheduling.
from pathlib import Path  # Verify coordinator-owned memmap deletion timing.
import tempfile  # Isolate disk-backed matrix resources.
import unittest  # Run focused deterministic scheduling coverage.

import numpy as np  # Build exact small feature and label matrices.

from feature_processes import create_resource_directory, partition_evaluation_plan, remove_resource_directory, run_feature_set_processes, save_array_resource, validate_feature_set_workers  # Exercise production process infrastructure.


FEATURE_SETS = ("GA Features", "PCA Components", "RFE Features")  # Define the production-equivalent selected feature identities.


def build_plan(classifier_count: int = 8) -> list:
    """
    Build one production-equivalent feature evaluation plan.

    :param classifier_count: Number of classifiers in each HP/testing slice.
    :return: Ordered plan tuples matching the production nesting.
    """

    classifiers = [f"Classifier {index}" for index in range(classifier_count)]  # Build deterministic classifier identities.
    return [(feature_set, hp_enabled, ratio, classifier) for hp_enabled in (False, True) for ratio in (None, 0.25, 0.50, 0.75, 1.00) for feature_set in FEATURE_SETS for classifier in classifiers]  # Preserve HP, testing, feature, and classifier order.


def build_payloads(partitions: dict, array_metadata: dict = None, label_metadata: dict = None, log_path: str = None) -> list:
    """
    Build metadata-only fixture payloads from production plan partitions.

    :param partitions: Feature-set partitions returned by production code.
    :param array_metadata: Optional shared read-only memmap metadata.
    :param label_metadata: Optional shared read-only label memmap metadata.
    :param log_path: Optional shared process-safe fixture log path.
    :return: One payload per feature set.
    """

    payloads = []  # Accumulate one persistent worker payload per feature set.
    for feature_set, tasks in partitions.items():  # Preserve authoritative feature order.
        payloads.append({"feature_set": feature_set, "worker_index": 1, "tasks": tasks, "array": array_metadata, "labels": label_metadata, "log_path": log_path, "block": feature_set == "PCA Components"})  # Pass no full matrix through the process payload.
    return payloads  # Return the metadata-only payload list.


def payload_contains_arrays(value) -> bool:
    """
    Determine whether a nested process payload contains an in-memory NumPy array.

    :param value: Arbitrarily nested process payload value.
    :return: True when an in-memory NumPy array appears anywhere in the payload.
    """

    if isinstance(value, np.ndarray):  # Detect arrays before traversing container types.
        return True  # Report forbidden in-memory matrix transport.
    if isinstance(value, dict):  # Traverse every mapping value.
        return any(payload_contains_arrays(item) for item in value.values())  # Report any nested in-memory array.
    if isinstance(value, (list, tuple)):  # Traverse ordered payload containers.
        return any(payload_contains_arrays(item) for item in value)  # Report any nested in-memory array.
    return False  # Accept scalar and metadata-only values.


class FeatureProcessTests(unittest.TestCase):  # Cover plan partitioning and persistent process lifecycle.
    def test_plan_partitions_into_exact_80_80_80_without_identity_changes(self) -> None:
        """
        Verify production-equivalent dynamic partition identity and ordering.

        :param self: Instance of the FeatureProcessTests class.
        :return: None.
        """

        plan = build_plan()  # Build the production-equivalent 240-combination plan.
        partitions = partition_evaluation_plan(plan)  # Partition actual plan objects dynamically.
        self.assertEqual(len(plan), 240)  # Require the complete authoritative denominator.
        self.assertEqual({name: len(tasks) for name, tasks in partitions.items()}, {"GA Features": 80, "PCA Components": 80, "RFE Features": 80})  # Require exact feature queue sizes.
        flattened_ids = [task["global_id"] for tasks in partitions.values() for task in tasks]  # Collect every preserved global ID.
        self.assertEqual(sorted(flattened_ids), list(range(1, 241)))  # Require no lost or duplicated combination.
        self.assertEqual([tasks[0]["global_id"] for tasks in partitions.values()], [1, 9, 17])  # Require original first global IDs.
        for feature_set, tasks in partitions.items():  # Verify every local queue independently.
            self.assertEqual([task["local_id"] for task in tasks], list(range(1, 81)))  # Require stable feature-local ordering.
            self.assertTrue(all(task["combination"] is plan[task["global_id"] - 1] for task in tasks))  # Require preservation of the actual plan tuple objects.
            self.assertTrue(all(task["combination"][0] == feature_set for task in tasks))  # Require strict feature isolation.

    def test_worker_configuration_parses_and_rejects_unproven_counts(self) -> None:
        """
        Verify supported process mappings and explicit multiworker rejection.

        :param self: Instance of the FeatureProcessTests class.
        :return: None.
        """

        self.assertEqual(validate_feature_set_workers("ga=1,pca=1,rfe=1"), {"ga": 1, "pca": 1, "rfe": 1})  # Require CLI parsing.
        self.assertEqual(validate_feature_set_workers({}), {})  # Require sequential execution when multiprocessing is disabled.
        with self.assertRaisesRegex(ValueError, "must be exactly 1"):  # Require an explicit unsupported-count error.
            validate_feature_set_workers("ga=2,pca=1,rfe=1")  # Reject misleading multiworker queue behavior.

    def test_three_persistent_processes_advance_independently_and_reap(self) -> None:
        """
        Verify three persistent workers advance independently and exit cleanly.

        :param self: Instance of the FeatureProcessTests class.
        :return: None.
        """

        small_plan = [(feature_set, False, None, f"Classifier {index}") for feature_set in FEATURE_SETS for index in range(3)]  # Build three complete independent queues.
        partitions = partition_evaluation_plan(small_plan)  # Partition the focused nine-combination plan.
        with tempfile.TemporaryDirectory() as temporary_directory:  # Isolate the concurrent log artifact.
            log_path = str(Path(temporary_directory) / "workers.log")  # Build one shared append-only fixture log path.
            payloads = build_payloads(partitions, log_path=log_path)  # Give every persistent child its complete multi-task queue.
            self.assertFalse(payload_contains_arrays(payloads))  # Prove process arguments contain no in-memory full matrices.
            results = run_feature_set_processes(payloads, 9, "tests.feature_process_fixture", "run_fixture_worker")  # Execute exactly three real child processes.
            log_lines = Path(log_path).read_text(encoding="utf-8").splitlines()  # Read every complete concurrent log record.
            self.assertEqual(len(log_lines), 9)  # Require one readable record for every completed combination.
            self.assertTrue(all(line.count("|") == 2 for line in log_lines))  # Require no interleaved or partial records.
        pids_by_feature = {feature_set: {result["pid"] for result in results if result["feature_set"] == feature_set} for feature_set in FEATURE_SETS}  # Group observed PIDs by feature.
        self.assertEqual(len({next(iter(pids)) for pids in pids_by_feature.values()}), 3)  # Require exactly three distinct child PIDs.
        self.assertTrue(all(len(pids) == 1 for pids in pids_by_feature.values()))  # Require one persistent PID across each complete queue.
        self.assertTrue(all(len([result for result in results if result["feature_set"] == feature_set]) == 3 for feature_set in FEATURE_SETS))  # Require multiple combinations per child.
        pca_first_finish = min(result["finished_at"] for result in results if result["feature_set"] == "PCA Components")  # Resolve the blocked PCA first completion.
        self.assertLess(max(result["finished_at"] for result in results if result["feature_set"] != "PCA Components"), pca_first_finish)  # Require GA and RFE to finish while PCA remains blocked.
        self.assertFalse(any(process.name.startswith("stacking-") for process in mp.active_children()))  # Require no orphan or zombie worker handles.

    def test_cache_hits_continue_without_fitting(self) -> None:
        """
        Verify a cache hit advances and the same process continues its queue.

        :param self: Instance of the FeatureProcessTests class.
        :return: None.
        """

        plan = [("GA Features", False, None, "A"), ("GA Features", False, None, "B"), ("GA Features", False, None, "C")]  # Build one single-feature queue.
        tasks = partition_evaluation_plan(plan)["GA Features"]  # Resolve the complete local queue.
        tasks[0]["cached"] = True  # Mark the first task as a deterministic cache hit.
        payload = {"feature_set": "GA Features", "worker_index": 1, "tasks": tasks, "array": None, "block": False}  # Build one metadata-only persistent worker payload.
        results = run_feature_set_processes([payload], 3, "tests.feature_process_fixture", "run_fixture_worker")  # Execute single-feature process mode.
        self.assertEqual([result["fitted"] for result in results], [False, True, True])  # Require immediate continuation after the cache hit.
        self.assertEqual(len({result["pid"] for result in results}), 1)  # Require one persistent child for the complete queue.

    def test_memmap_values_are_identical_and_coordinator_owns_cleanup(self) -> None:
        """
        Verify exact memmap values and coordinator-only deletion ownership.

        :param self: Instance of the FeatureProcessTests class.
        :return: None.
        """

        matrix = np.arange(24, dtype=np.float64).reshape(8, 3)  # Build one exact deterministic feature matrix.
        labels = np.asarray([0, 1, 1, 0, 1, 0, 0, 1], dtype=np.int64)  # Build exact deterministic estimator labels.
        with tempfile.TemporaryDirectory() as temporary_directory:  # Isolate coordinator resources.
            resource_directory = create_resource_directory(temporary_directory)  # Create one owned resource directory.
            metadata = save_array_resource(matrix, resource_directory, "features")  # Persist the matrix without process pickling.
            label_metadata = save_array_resource(labels, resource_directory, "labels")  # Persist labels without process pickling.
            self.assertNotIn("array", metadata)  # Require path, shape, and dtype metadata only.
            plan = [("GA Features", False, None, "A")]  # Build one single-feature combination.
            payloads = build_payloads(partition_evaluation_plan(plan), array_metadata=metadata, label_metadata=label_metadata)  # Pass only reopen metadata to the child.
            self.assertFalse(payload_contains_arrays(payloads))  # Prove full features and labels are absent from process arguments.
            results = run_feature_set_processes(payloads, 1, "tests.feature_process_fixture", "run_fixture_worker")  # Reopen and consume the matrix in one child.
            np.testing.assert_array_equal(results[0]["array_values"], matrix)  # Require numerically identical estimator feature values.
            np.testing.assert_array_equal(results[0]["label_values"], labels)  # Require numerically identical estimator label values.
            self.assertTrue(Path(metadata["path"]).is_file())  # Require resources to remain after the worker exits until coordinator cleanup.
            remove_resource_directory(resource_directory)  # Execute explicit coordinator-owned deletion.
            self.assertFalse(Path(resource_directory).exists())  # Require complete cleanup after child join.

    def test_child_exception_reaches_coordinator_after_sibling_completion(self) -> None:
        """
        Verify child exceptions surface after independent siblings finish.

        :param self: Instance of the FeatureProcessTests class.
        :return: None.
        """

        plan = [(feature_set, False, None, "A") for feature_set in FEATURE_SETS]  # Build one task per independent feature set.
        payloads = build_payloads(partition_evaluation_plan(plan))  # Build three process payloads.
        payloads[1]["fail"] = True  # Fail only the PCA worker.
        with self.assertRaisesRegex(RuntimeError, "fixture failure for PCA Components"):  # Require child exception surfacing.
            run_feature_set_processes(payloads, 3, "tests.feature_process_fixture", "run_fixture_worker")  # Allow sibling workers to complete before coordinator failure.
        self.assertFalse(any(process.name.startswith("stacking-") for process in mp.active_children()))  # Require deterministic reaping after failure.

    def test_duplicate_result_identity_is_rejected(self) -> None:
        """
        Verify exact-once validation rejects duplicate global identities.

        :param self: Instance of the FeatureProcessTests class.
        :return: None.
        """

        tasks = partition_evaluation_plan([("GA Features", False, None, "A"), ("GA Features", False, None, "B")])["GA Features"]  # Build two authoritative tasks.
        tasks[1]["global_id"] = tasks[0]["global_id"]  # Inject one duplicate delivery identity.
        payload = {"feature_set": "GA Features", "worker_index": 1, "tasks": tasks, "array": None, "labels": None, "log_path": None, "block": False}  # Build one deterministic invalid queue.
        with self.assertRaisesRegex(RuntimeError, "returned 2 results"):  # Require exact-once result validation.
            run_feature_set_processes([payload], 2, "tests.feature_process_fixture", "run_fixture_worker")  # Reject duplicate global identity delivery.


if __name__ == "__main__":  # Support direct focused test execution.
    unittest.main()  # Run the feature-process test suite.
