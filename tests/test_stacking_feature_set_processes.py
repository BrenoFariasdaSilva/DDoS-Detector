import json  # Serialize small spawned-process evidence records
import inspect  # Read production scheduler source for forbidden fixed-denominator regression coverage
import multiprocessing as mp  # Exercise real spawned OS process lifecycle
import os  # Record child and parent process identities
from pathlib import Path  # Build isolated configuration, memmap, and log paths
import queue  # Assert cross-process combination reservation blocking
import sys  # Inject deterministic CLI arguments
import tempfile  # Isolate every focused process and backing-file artifact
import time  # Prove independent feature-set completion timing
import traceback  # Surface complete probe child failures through the coordinator
import unittest  # Provide the repository-standard focused test runner
from unittest import mock  # Observe lazy augmentation loading and CLI behavior

import numpy as np  # Verify exact memmap features and reject matrix task payloads
import pandas as pd  # Build small deterministic production preprocessing input
import yaml  # Validate production and example configuration fallbacks

import stacking  # Exercise the production feature-set process architecture


FEATURE_SET_NAMES = ["GA Features", "PCA Components", "RFE Features"]  # Preserve the stated server feature-set process order
FOUR_FEATURE_SET_NAMES = ["Full Features", *FEATURE_SET_NAMES]  # Extend the committed three-worker order with the Full Features baseline
SERVER_CLASSIFIER_NAMES = ["Random Forest", "XGBoost", "Logistic Regression", "KNN", "Nearest Centroid", "Gradient Boosting", "LightGBM", "MLP (Neural Net)"]  # Mirror one explicit server execution grid


def run_feature_process_probe(process_payload, status_queue, status_state):  # Exercise the production coordinator with a small spawned lifecycle target
    """
    Run one deterministic spawned feature-set lifecycle probe.

    :param process_payload: Small feature-set process payload.
    :param status_queue: Multiprocessing lifecycle status queue.
    :param status_state: Shared process-safe global and feature-local status counters.
    :return: None.
    """

    feature_set = process_payload["feature_set"]  # Resolve this child's sole feature-set assignment
    status_queue.put({"status": "started", "feature_set": feature_set, "worker_index": process_payload["worker_index"], "pid": os.getpid(), "ppid": os.getppid(), "queue_size": len(process_payload["tasks"])})  # Report production-compatible startup metadata
    log_writer = stacking.Logger(process_payload["probe_log"], clean=False)  # Open the shared append-only process-safe logger
    try:  # Preserve complete child evidence and deterministic failure status
        for line_index in range(10):  # Emit multiple concurrent records from every child
            log_writer.write(f"PROBE|{feature_set}|{os.getpid()}|{line_index}")  # Write one complete process-locked record
        time.sleep(float(process_payload["probe_delays"].get(feature_set, 0.0)))  # Simulate independent feature-set duration without threads
        for task in process_payload["tasks"]:  # Process each matrix-free descriptor sequentially in its sole feature worker
            stacking.transition_feature_process_status(status_state, task, "started")  # Move the dequeued probe task from pending to running atomically
            status_queue.put({"status": "running", "feature_set": feature_set, "global_id": task["global_id"], "pid": os.getpid()})  # Report the active task identity before terminal work
            if process_payload.get("probe_failure") == feature_set:  # Inject one requested task failure after its running transition
                stacking.transition_feature_process_status(status_state, task, "failed")  # Count failure without claiming successful completion
                raise RuntimeError(f"Injected {feature_set} failure")  # Preserve an exact child failure identity
            stacking.transition_feature_process_status(status_state, task, "computed")  # Count one durably simulated successful result exactly once
            status_queue.put({"status": "progress", "feature_set": feature_set, "global_id": task["global_id"], "event": "computed", "pid": os.getpid()})  # Publish small terminal progress metadata
        completed = stacking.read_feature_process_status(status_state)["global"]["completed"]  # Read cache-inclusive completion after this queue finishes
        evidence_path = Path(process_payload["probe_directory"]) / f"{process_payload['worker_key']}.json"  # Resolve one feature-local completion evidence file
        evidence_path.write_text(json.dumps({"feature_set": feature_set, "pid": os.getpid(), "ppid": os.getppid(), "tasks": [task["feature_set"] for task in process_payload["tasks"]], "finished": time.time()}), encoding="utf-8")  # Persist small child identity and timing evidence
        status_queue.put({"status": "done", "feature_set": feature_set, "worker_index": process_payload["worker_index"], "pid": os.getpid(), "completed": completed})  # Report production-compatible successful terminal metadata
    except BaseException as error:  # Surface the injected child exception through the production coordinator channel
        status_queue.put({"status": "error", "feature_set": feature_set, "worker_index": process_payload["worker_index"], "pid": os.getpid(), "global_id": task.get("global_id") if "task" in locals() else None, "error": str(error), "traceback": traceback.format_exc()})  # Send complete textual failure evidence without matrices
        raise  # Preserve a nonzero child exit code
    finally:  # Flush and close the isolated process logger under every outcome
        log_writer.flush()  # Flush final child records
        log_writer.close()  # Close only this child process's file handle
def run_combination_lock_probe(task, process_payload, status_queue):  # Report only after acquiring one production combination reservation
    """
    Acquire and report one production combination reservation.

    :param task: Small feature-process task descriptor.
    :param process_payload: Small cache and combination identity payload.
    :param status_queue: Multiprocessing status queue for acquisition evidence.
    :return: None.
    """

    combination_lock = stacking.acquire_feature_process_combination_lock(task, process_payload)  # Wait for the exact production combination reservation
    try:  # Hold the reservation until acquisition evidence is transported
        status_queue.put({"pid": os.getpid(), "acquired_at": time.time()})  # Report only small successful acquisition metadata
    finally:  # Release the production reservation after evidence publication
        combination_lock.close()  # Close the descriptor and release the process-safe lock


def make_probe_payload(temporary_directory, delays=None, failure=None, feature_names=None):  # Build one matrix-free coordinator payload for spawned lifecycle validation
    """
    Build a matrix-free persistent-process probe payload.

    :param temporary_directory: Isolated directory for child evidence and logging.
    :param delays: Optional feature-set delay mapping.
    :param failure: Optional feature set that raises in its child.
    :param feature_names: Optional active feature-set names.
    :return: Small coordinator process payload.
    """

    active_features = list(feature_names or FEATURE_SET_NAMES)  # Resolve this probe's actual configured feature-set processes
    worker_counts = {key: int(any(stacking.resolve_feature_set_worker_key(name) == key for name in active_features)) for key in stacking.FEATURE_SET_WORKER_KEYS}  # Derive worker configuration from active feature roles
    return {"feature_mode_names": active_features, "feature_metadata_by_name": {name: {"feature_names": [name], "indices": [0], "feature_count": 1} for name in active_features}, "config": {"evaluation": {"feature_set_workers": worker_counts}}, "probe_directory": str(temporary_directory), "probe_log": str(Path(temporary_directory) / "concurrent.log"), "probe_delays": dict(delays or {}), "probe_failure": failure}  # Return only small JSON-compatible process metadata


def make_probe_tasks(feature_names=None, tasks_per_feature=1):  # Build dynamic matrix-free tasks and feature-local pending queues
    """
    Build small probe tasks from the active feature-set list.

    :param feature_names: Optional active feature-set names.
    :param tasks_per_feature: Number of sequential tasks assigned to each persistent worker.
    :return: Tuple containing authoritative tasks and pending queues.
    """

    active_features = list(feature_names or FEATURE_SET_NAMES)  # Resolve the current configured feature-set plan
    planned_features = [feature_name for feature_name in active_features for _ in range(tasks_per_feature)]  # Expand each feature queue without creating more workers
    tasks = [{"feature_set": feature_name, "global_id": global_id} for global_id, feature_name in enumerate(planned_features, start=1)]  # Preserve one dynamic original global identity per task
    pending = {feature_name: [task for task in tasks if task["feature_set"] == feature_name] for feature_name in active_features}  # Partition only active tasks by their exact feature set
    return tasks, pending  # Return authoritative plan order and isolated feature-local queues


def make_process_cache_row(task):  # Build the fields used by process cache compatibility validation
    """
    Build one process-compatible cache row.

    :param task: Small feature-process task descriptor.
    :return: Cache result fields required by process preflight validation.
    """

    expected_test_count = task["expected_n_samples_test"] if task["expected_n_samples_test"] is not None else 25  # Supply persisted augmented cardinality without opening source rows
    return {"n_features": task["expected_n_features"], "features_list": list(task["expected_feature_names"]), "n_samples_train": task["expected_n_samples_train"], "n_samples_test": expected_test_count, "feature_set": task["feature_set"], "model_name": task["classifier_name"]}  # Preserve exact shape and ordered feature identity


class FeatureSetProcessTests(unittest.TestCase):  # Group persistent feature-set process architecture coverage
    """Verify plan identity, cache-first scheduling, spawned lifecycle, memory, and logging."""

    def build_server_plan(self):  # Build one explicit server-specific 240-combination fixture
        """
        Build one server-shaped dynamic evaluation plan and task descriptors.

        :param self: FeatureSetProcessTests instance.
        :return: Tuple of evaluation plan, feature metadata, and enriched tasks.
        """

        default_models = {name: object() for name in SERVER_CLASSIFIER_NAMES}  # Provide this server's ordered default classifier identities
        optimized_models = {name: object() for name in SERVER_CLASSIFIER_NAMES}  # Provide this server's ordered optimized classifier identities
        hp_runs = [(False, default_models, {}), (True, optimized_models, {})]  # Preserve default-first hyperparameter mode order
        ratios = [None, 0.25, 0.50, 0.75, 1.00]  # Preserve original-first testing configurations
        evaluation_plan = stacking.build_evaluation_plan(hp_runs, ratios, FEATURE_SET_NAMES, False)  # Build the dynamic authoritative grid without stacking classifier
        feature_metadata = stacking.build_feature_process_metadata(["a", "b", "c", "d"], ["a", "c"], 2, ["b", "d"])  # Build small ordered feature descriptors
        tasks = stacking.build_feature_process_plan(evaluation_plan, feature_metadata, 100, "dataset.csv", "combined_files")  # Enrich global and feature-local identities without matrices or augmentation reads
        return evaluation_plan, feature_metadata, tasks  # Return the complete deterministic fixture

    def test_server_fixture_partitions_80_80_80(self):  # Verify worker fallback and one explicit server plan partition
        """
        Verify YAML worker fallback and the stated server 80/80/80 plan.

        :param self: FeatureSetProcessTests instance.
        :return: None.
        """

        repository_root = Path(stacking.__file__).resolve().parent  # Resolve the repository root
        for config_name in ("config.yaml", "config.yaml.example"):  # Validate runtime and example fallbacks together
            config_data = yaml.safe_load((repository_root / config_name).read_text(encoding="utf-8"))  # Parse the complete repository YAML file
            self.assertEqual(config_data["evaluation"]["feature_set_workers"], {"full": 0, "ga": 0, "pca": 0, "rfe": 0})  # Require sequential fallback until CLI override
            self.assertIn("SVM", config_data["stacking"]["enabled_classifiers"])  # Prevent scheduler fixtures from altering scientific classifier configuration
        evaluation_plan, _, tasks = self.build_server_plan()  # Build this explicit server plan
        self.assertEqual(len(evaluation_plan), 240)  # Preserve this server-specific expected scenario only
        counts = {name: sum(task["feature_set"] == name for task in tasks) for name in FEATURE_SET_NAMES}  # Count every feature-local queue dynamically
        self.assertEqual(counts, {"GA Features": 80, "PCA Components": 80, "RFE Features": 80})  # Preserve this server-specific expected partition only

    def test_evaluation_plan_totals_follow_enabled_dimensions(self):  # Verify every plan dimension changes totals without fixed production counts
        """
        Verify plan totals follow enabled classifiers, modes, tests, ratios, and features.

        :param self: FeatureSetProcessTests instance.
        :return: None.
        """

        feature_sets = list(FOUR_FEATURE_SET_NAMES)  # Build the stated four-feature scenario from enabled identities
        augmentation_modes = [None, 0.25, 0.50, 0.75, 1.00]  # Build original plus four enabled augmentation ratios
        full_models = {name: object() for name in SERVER_CLASSIFIER_NAMES}  # Build one enabled-classifier mapping
        reduced_models = dict(list(full_models.items())[:-1])  # Build a scenario with one classifier disabled
        full_hp_runs = [(False, full_models, {}), (True, full_models, {})]  # Build default and optimized modes
        reduced_hp_runs = [(enabled, reduced_models, params) for enabled, _, params in full_hp_runs]  # Disable one classifier across active HP modes
        three_feature_plan = stacking.build_evaluation_plan(full_hp_runs, augmentation_modes, FEATURE_SET_NAMES, False)  # Build the explicit server-disabled-feature scenario
        four_feature_plan = stacking.build_evaluation_plan(full_hp_runs, augmentation_modes, feature_sets, False)  # Build the corresponding Full Features scenario
        self.assertEqual(len(three_feature_plan), 240)  # Prove the explicit three-feature fixture only
        self.assertEqual(len(four_feature_plan), 320)  # Prove Full Features changes the same active dimensions dynamically
        four_feature_metadata = stacking.build_feature_process_metadata(["a", "b", "c", "d"], ["a", "c"], 2, ["b", "d"])  # Build all four matrix-free feature identities
        four_feature_tasks = stacking.build_feature_process_plan(four_feature_plan, four_feature_metadata, 100, "dataset.csv", "combined_files")  # Preserve global and feature-local identities for the explicit fixture
        four_feature_counts = {name: sum(task["feature_set"] == name for task in four_feature_tasks) for name in feature_sets}  # Derive every feature-local total from generated descriptors
        self.assertEqual(four_feature_counts, {name: 80 for name in feature_sets})  # Require the explicit 320 fixture to partition dynamically as 80 per feature
        _, four_feature_pending = stacking.partition_feature_process_tasks(four_feature_tasks, {}, {"attack_types_combined": ["BENIGN", "ATTACK"], "execution_mode": "combined_files", "feature_mode_names": feature_sets}, {})  # Build cache-first feature-local queues without opening data
        self.assertTrue(all(all(task["feature_set"] == name for task in queue_tasks) for name, queue_tasks in four_feature_pending.items()))  # Keep every Full, GA, PCA, and RFE combination in its sole queue
        self.assertEqual({task["global_id"] for queue_tasks in four_feature_pending.values() for task in queue_tasks}, {task["global_id"] for task in four_feature_tasks})  # Preserve every original global ID exactly once across all four queues
        four_feature_status = stacking.create_feature_process_status(mp.get_context("spawn"), four_feature_tasks, four_feature_pending)  # Build dynamic global and per-feature Full status
        full_progress = stacking.format_feature_process_progress(four_feature_pending["Full Features"][0], four_feature_status, pid=1234)  # Format one Full combination from actual runtime state
        self.assertIn("[FULL 0/80", full_progress)  # Use dynamic Full local total
        self.assertIn("Global ID 1/320", full_progress)  # Preserve original global identity and dynamic global total
        self.assertIn("Completed 0/320", full_progress)  # Report process-safe cache-inclusive completion
        self.assertIn("Pending 320 | Running 0", full_progress)  # Report exact global pending and running state
        self.assertEqual(len(stacking.build_evaluation_plan(reduced_hp_runs, augmentation_modes, feature_sets, False)), 280)  # Remove one classifier across both hyperparameter modes dynamically
        self.assertEqual(len(stacking.build_evaluation_plan(full_hp_runs, augmentation_modes[:-1], feature_sets, False)), 256)  # Remove one augmentation ratio across every classifier, mode, and feature set dynamically
        self.assertEqual(len(stacking.build_evaluation_plan(full_hp_runs, augmentation_modes, feature_sets[:-1], False)), 240)  # Disable one feature set without changing any other active dimension
        scenarios = [(full_hp_runs, augmentation_modes, feature_sets), (full_hp_runs, augmentation_modes, feature_sets[:-1]), (full_hp_runs, augmentation_modes[:-1], feature_sets), (reduced_hp_runs, augmentation_modes, feature_sets), (full_hp_runs[:1], augmentation_modes, feature_sets)]  # Vary one enabled plan dimension per scenario
        for active_hp_runs, active_augmentation_modes, active_feature_sets in scenarios:  # Evaluate every dynamic dimension variant
            evaluation_plan = stacking.build_evaluation_plan(active_hp_runs, active_augmentation_modes, active_feature_sets, False)  # Build authoritative plan from current enabled values
            expected_per_feature = sum(len(models) for _, models, _ in active_hp_runs) * len(active_augmentation_modes)  # Derive one feature-local total from active modes and classifiers
            expected_total = expected_per_feature * len(active_feature_sets)  # Derive global total from active feature sets
            self.assertEqual(len(evaluation_plan), expected_total)  # Require plan length to follow active dimensions
            feature_totals = {name: sum(item[0] == name for item in evaluation_plan) for name in active_feature_sets}  # Derive every local total from actual plan
            self.assertEqual(feature_totals, {name: expected_per_feature for name in active_feature_sets})  # Require plan-derived feature-local totals
        scheduler_source = "\n".join(inspect.getsource(function) for function in (stacking.build_feature_process_plan, stacking.partition_feature_process_tasks, stacking.create_feature_process_status, stacking.format_feature_process_progress, stacking.execute_feature_set_processes))  # Read only production scheduling and progress implementations
        self.assertNotIn(" 320", scheduler_source)  # Forbid the explicit four-feature fixture total in production scheduling code
        self.assertNotIn(" 240", scheduler_source)  # Forbid the explicit fixture total in production scheduling code
        self.assertNotIn(" 80", scheduler_source)  # Forbid the explicit fixture local total in production scheduling code

    def test_plan_preserves_global_ids_local_order_and_no_duplicates(self):  # Verify combination identity preservation
        """
        Verify global IDs, feature-local order, and complete unique partitioning.

        :param self: FeatureSetProcessTests instance.
        :return: None.
        """

        evaluation_plan, _, tasks = self.build_server_plan()  # Build the explicit server-shaped task list
        total_combinations = len(evaluation_plan)  # Derive global total from authoritative plan
        self.assertEqual([task["global_id"] for task in tasks], list(range(1, total_combinations + 1)))  # Preserve every original one-based global ID
        reconstructed = [(task["feature_set"], task["hyperparameters_enabled"], task["augmentation_ratio"], task["classifier_name"]) for task in tasks]  # Reconstruct authoritative tuple identities
        self.assertEqual(reconstructed, evaluation_plan)  # Require unchanged HP, testing, ratio, feature, and classifier ordering
        partitioned_ids = []  # Accumulate feature-local partition identities
        for feature_set in FEATURE_SET_NAMES:  # Inspect each feature-local queue independently
            local_tasks = [task for task in tasks if task["feature_set"] == feature_set]  # Preserve global relative order within this feature
            self.assertEqual([task["feature_local_position"] for task in local_tasks], list(range(1, len(local_tasks) + 1)))  # Require stable dynamic one-based local order
            self.assertTrue(all(task["feature_local_total"] == len(local_tasks) for task in local_tasks))  # Require each task's local denominator from its actual partition
            partitioned_ids.extend(task["global_id"] for task in local_tasks)  # Accumulate every partitioned global identity
        self.assertEqual(len(partitioned_ids), len(set(partitioned_ids)))  # Forbid duplicated combinations across partitions
        self.assertEqual(set(partitioned_ids), set(range(1, total_combinations + 1)))  # Forbid lost combinations against dynamic plan size

    def test_status_counters_and_progress_use_dynamic_runtime_state(self):  # Verify complete cache, lifecycle, failure, and denominator accounting
        """
        Verify synchronized counters and progress use only actual task-plan values.

        :param self: FeatureSetProcessTests instance.
        :return: None.
        """

        tasks = [{"feature_set": "GA Features", "worker_key": "ga", "global_id": 1, "feature_local_position": 1, "feature_local_total": 2}, {"feature_set": "GA Features", "worker_key": "ga", "global_id": 2, "feature_local_position": 2, "feature_local_total": 2, "pending_queue_position": 1, "pending_queue_total": 1}, {"feature_set": "PCA Components", "worker_key": "pca", "global_id": 3, "feature_local_position": 1, "feature_local_total": 1}, {"feature_set": "RFE Features", "worker_key": "rfe", "global_id": 4, "feature_local_position": 1, "feature_local_total": 1, "pending_queue_position": 1, "pending_queue_total": 1}]  # Build a non-fixture task plan with mixed cache state
        pending = {"GA Features": [tasks[1]], "PCA Components": [], "RFE Features": [tasks[3]]}  # Mark GA and RFE pending while PCA and one GA task are cached
        status_state = stacking.create_feature_process_status(mp.get_context("spawn"), tasks, pending)  # Initialize all status from the actual cache partition
        initial = stacking.read_feature_process_status(status_state)  # Read one atomic initial snapshot
        self.assertEqual(initial["global"], {"total": 4, "cached": 2, "pending": 2, "running": 0, "computed": 0, "failed": 0, "completed": 2})  # Require exact cache-first global accounting
        self.assertEqual(initial["features"]["PCA Components"], {"total": 1, "cached": 1, "pending": 0, "running": 0, "computed": 0, "failed": 0, "completed": 1})  # Require correct all-cached feature state
        stacking.transition_feature_process_status(status_state, tasks[1], "started")  # Move the pending GA task into running state
        running_prefix = stacking.format_feature_process_progress(tasks[1], status_state, pid=1234)  # Format progress from synchronized dynamic denominators
        self.assertIn("[GA 1/2", running_prefix)  # Use the actual feature-local completion and total
        self.assertIn("Global ID 2/4", running_prefix)  # Use the task's original global identity and actual plan total
        self.assertIn("Completed 2/4", running_prefix)  # Include cached work in completion immediately
        self.assertIn("Pending 1 | Running 1", running_prefix)  # Distinguish remaining pending and active work
        stacking.transition_feature_process_status(status_state, tasks[1], "computed")  # Complete GA through successful persistence
        stacking.transition_feature_process_status(status_state, tasks[3], "started")  # Move RFE into running state
        final = stacking.transition_feature_process_status(status_state, tasks[3], "failed")  # Fail RFE without claiming successful completion
        self.assertEqual(final["global"], {"total": 4, "cached": 2, "pending": 0, "running": 0, "computed": 1, "failed": 1, "completed": 3})  # Require every task in one exact terminal state
        self.assertEqual(final["features"]["GA Features"]["completed"], 2)  # Combine one cached and one computed GA task
        self.assertEqual(final["features"]["RFE Features"]["completed"], 0)  # Exclude failed RFE work from completion

    def test_planning_and_cache_recovery_do_not_open_augmentation_rows(self):  # Verify metadata-only planning and cache-first recovery
        """
        Verify planning and complete cache recovery never open augmentation content.

        :param self: FeatureSetProcessTests instance.
        :return: None.
        """

        with mock.patch.object(stacking, "load_augmented_dataset") as augmented_open_mock, mock.patch.object(stacking, "load_feature_process_ratio_data") as ratio_open_mock:  # Instrument every production augmentation row-opening boundary
            _, _, tasks = self.build_server_plan()  # Build tasks using only located-path-independent plan metadata
            cache_dict = {}  # Build compatible cached results for every original and augmented combination
            for task in tasks:  # Populate the unchanged production identity mapping
                cache_key = stacking.build_resume_cache_key("combined_files", task["data_source_label"], task["experiment_mode"], task["augmentation_ratio"], ["BENIGN", "ATTACK"], task["feature_set"], task["classifier_name"], task["hyperparameters_enabled"])  # Resolve one exact cache key without opening data
                cache_dict[cache_key] = make_process_cache_row(task)  # Persist only small compatible result metadata
            process_payload = {"attack_types_combined": ["BENIGN", "ATTACK"], "execution_mode": "combined_files", "feature_mode_names": list(FEATURE_SET_NAMES)}  # Provide only cache-partition metadata
            cached_results, pending = stacking.partition_feature_process_tasks(tasks, cache_dict, process_payload, {})  # Recover all cache entries before any child or data resource
        self.assertEqual(len(cached_results), len(tasks))  # Require complete cache recovery from metadata alone
        self.assertTrue(all(not queue_tasks for queue_tasks in pending.values()))  # Never enqueue cached augmented combinations
        augmented_open_mock.assert_not_called()  # Never parse augmentation CSV rows during planning or cache recovery
        ratio_open_mock.assert_not_called()  # Never build augmentation arrays or memmaps during planning or cache recovery

    def test_cached_augmented_and_pending_original_tasks_do_not_load_augmentation(self):  # Verify task-level cache and original-data lazy boundaries
        """
        Verify cached augmented and pending original combinations skip augmentation loading.

        :param self: FeatureSetProcessTests instance.
        :return: None.
        """

        _, _, tasks = self.build_server_plan()  # Build exact production-shaped identities
        augmented_task = next(task for task in tasks if task["augmentation_ratio"] is not None)  # Select one cached augmented combination
        augmented_status = stacking.create_feature_process_status(mp.get_context("spawn"), [augmented_task], {augmented_task["feature_set"]: [augmented_task]})  # Initialize one pending augmented lifecycle
        augmented_payload = {"feature_set": augmented_task["feature_set"], "attack_types_combined": ["BENIGN", "ATTACK"]}  # Provide only task-owned metadata outside patched boundaries
        augmented_models = {augmented_task["hyperparameters_enabled"]: {augmented_task["classifier_name"]: object()}}  # Provide one process-local estimator identity
        augmented_resources = {"original_resources": None, "ratio_data": None, "active_ratio": None, "cache_dict": {}}  # Start without any loaded data
        augmented_queue = queue.Queue()  # Capture only small lifecycle messages
        augmented_lock = mock.Mock()  # Observe exact reservation release
        with mock.patch.object(stacking, "acquire_feature_process_combination_lock", return_value=augmented_lock), mock.patch.object(stacking, "reload_feature_process_task_cache", return_value=({"cached": True}, {})), mock.patch.object(stacking, "load_feature_process_ratio_data") as ratio_open_mock, mock.patch.object(stacking, "log_feature_process_combination"):  # Isolate the final-cache-hit execution path
            stacking.process_feature_process_task(augmented_task, augmented_payload, augmented_models, augmented_resources, augmented_queue, augmented_status)  # Recover the augmented task without opening its ratio
        ratio_open_mock.assert_not_called()  # Never load data for a cached augmented task
        self.assertEqual(stacking.read_feature_process_status(augmented_status)["global"]["cached"], 1)  # Count the final cache recovery exactly once
        original_task = next(task for task in tasks if task["augmentation_ratio"] is None)  # Select one pending original-data combination
        original_status = stacking.create_feature_process_status(mp.get_context("spawn"), [original_task], {original_task["feature_set"]: [original_task]})  # Initialize one pending original lifecycle
        original_payload = {"feature_set": original_task["feature_set"], "worker_index": 1, "attack_types_combined": ["BENIGN", "ATTACK"]}  # Provide small worker identity metadata
        original_models = {original_task["hyperparameters_enabled"]: {original_task["classifier_name"]: object()}}  # Provide one process-local estimator identity
        original_resources = {"original_resources": None, "ratio_data": None, "active_ratio": None, "cache_dict": {}}  # Start without any data resource
        original_queue = queue.Queue()  # Capture small original-task lifecycle messages
        original_lock = mock.Mock()  # Observe original combination reservation release
        prepared_original = {"X_train": np.ones((2, 1), dtype=np.float64)}  # Provide minimal matrix metadata at the patched preparation boundary
        with mock.patch.object(stacking, "acquire_feature_process_combination_lock", return_value=original_lock), mock.patch.object(stacking, "reload_feature_process_task_cache", return_value=(None, {})), mock.patch.object(stacking, "prepare_feature_process_original_resources", return_value=prepared_original), mock.patch.object(stacking, "evaluate_feature_process_original_task", return_value={"persisted": True}), mock.patch.object(stacking, "load_feature_process_ratio_data") as original_ratio_open_mock, mock.patch.object(stacking, "log_feature_process_combination"):  # Isolate the original-data computation path
            stacking.process_feature_process_task(original_task, original_payload, original_models, original_resources, original_queue, original_status)  # Compute the original task without augmentation
        original_ratio_open_mock.assert_not_called()  # Never load augmentation for an original-data combination
        self.assertEqual(stacking.read_feature_process_status(original_status)["global"]["computed"], 1)  # Count successful original persistence exactly once

    def test_full_cache_and_task_data_loading_boundaries(self):  # Verify Full uses the generic cache-first and lazy-data task path
        """Verify cached, original, and augmented Full tasks load only required resources."""

        metadata = stacking.build_feature_process_metadata(["a", "b"], ["a"], 1, ["b"])  # Build the Full feature identity without matrix data
        plan = stacking.build_evaluation_plan([(False, {"Random Forest": object()}, {})], [None, 0.5], ["Full Features"], False)  # Build one original and one augmented Full combination
        tasks = stacking.build_feature_process_plan(plan, metadata, 10, "dataset.csv", "combined_files")  # Preserve generic task identities and dynamic totals
        original_task, augmented_task = tasks  # Resolve both Full lifecycle boundaries
        model_maps = {False: {"Random Forest": object()}}  # Provide one process-local prototype identity behind patched evaluation boundaries
        cached_state = stacking.create_feature_process_status(mp.get_context("spawn"), [original_task], {"Full Features": [original_task]})  # Initialize final-cache recovery as pending work
        cached_resources = {"original_resources": None, "ratio_data": None, "active_ratio": None, "cache_dict": {}}  # Start with no data loaded
        with mock.patch.object(stacking, "acquire_feature_process_combination_lock", return_value=mock.Mock()), mock.patch.object(stacking, "reload_feature_process_task_cache", return_value=({"cached": True}, {})), mock.patch.object(stacking, "prepare_feature_process_original_resources") as original_load_mock, mock.patch.object(stacking, "load_feature_process_ratio_data") as ratio_load_mock, mock.patch.object(stacking, "evaluate_feature_process_original_task") as original_fit_mock, mock.patch.object(stacking, "evaluate_feature_process_augmented_task") as augmented_fit_mock, mock.patch.object(stacking, "log_feature_process_combination"):  # Exercise a concurrent final cache hit
            stacking.process_feature_process_task(original_task, {"feature_set": "Full Features", "attack_types_combined": None}, model_maps, cached_resources, queue.Queue(), cached_state)  # Recover Full without fit or data loading
        original_load_mock.assert_not_called()  # Skip Full original memmap opening on cache recovery
        ratio_load_mock.assert_not_called()  # Skip augmentation on cache recovery
        original_fit_mock.assert_not_called()  # Never fit a cached Full combination
        augmented_fit_mock.assert_not_called()  # Never enter augmented evaluation for the cached original task
        self.assertEqual(stacking.read_feature_process_status(cached_state)["features"]["Full Features"]["cached"], 1)  # Count Full cache recovery in feature-local status
        original_state = stacking.create_feature_process_status(mp.get_context("spawn"), [original_task], {"Full Features": [original_task]})  # Initialize one pending original Full task
        original_resources = {"original_resources": None, "ratio_data": None, "active_ratio": None, "cache_dict": {}}  # Start original Full without augmentation resources
        prepared_full = {"X_train": np.ones((8, 2), dtype=np.float64)}  # Provide minimal prepared Full matrix metadata behind the patched scientific boundary
        with mock.patch.object(stacking, "acquire_feature_process_combination_lock", return_value=mock.Mock()), mock.patch.object(stacking, "reload_feature_process_task_cache", return_value=(None, {})), mock.patch.object(stacking, "prepare_feature_process_original_resources", return_value=prepared_full), mock.patch.object(stacking, "load_feature_process_ratio_data") as original_ratio_load_mock, mock.patch.object(stacking, "evaluate_feature_process_original_task", return_value={"persisted": True}), mock.patch.object(stacking, "log_feature_process_combination"):  # Exercise original Full through the generic computation path
            stacking.process_feature_process_task(original_task, {"feature_set": "Full Features", "worker_index": 1, "attack_types_combined": None}, model_maps, original_resources, queue.Queue(), original_state)  # Compute original Full without touching augmentation
        original_ratio_load_mock.assert_not_called()  # Original Full loads no augmentation data
        stacking.cleanup_feature_process_original_resources(original_resources["original_resources"])  # Release focused original resources
        augmented_state = stacking.create_feature_process_status(mp.get_context("spawn"), [augmented_task], {"Full Features": [augmented_task]})  # Initialize one pending augmented Full task
        augmented_resources = {"original_resources": None, "ratio_data": None, "active_ratio": None, "cache_dict": {}}  # Start augmented Full without original matrices
        ratio_data = {"ratio": 0.5, "X_raw": np.ones((5, 2), dtype=np.float64), "y_encoded": np.array([0, 1, 0, 1, 0], dtype=np.int64)}  # Provide only requested ratio data
        with mock.patch.object(stacking, "acquire_feature_process_combination_lock", return_value=mock.Mock()), mock.patch.object(stacking, "reload_feature_process_task_cache", return_value=(None, {})), mock.patch.object(stacking, "prepare_feature_process_original_resources") as augmented_original_load_mock, mock.patch.object(stacking, "load_feature_process_ratio_data", return_value=ratio_data) as augmented_ratio_load_mock, mock.patch.object(stacking, "evaluate_feature_process_augmented_task", return_value={"persisted": True}), mock.patch.object(stacking, "log_feature_process_combination"):  # Exercise lazy augmented Full loading
            stacking.process_feature_process_task(augmented_task, {"feature_set": "Full Features", "attack_types_combined": None}, model_maps, augmented_resources, queue.Queue(), augmented_state)  # Load exactly the pending Full ratio
        augmented_original_load_mock.assert_not_called()  # Augmented Full does not reopen original matrices
        augmented_ratio_load_mock.assert_called_once_with(mock.ANY, 0.5)  # Load only the required augmentation ratio
        stacking.cleanup_feature_process_ratio_data(augmented_resources["ratio_data"])  # Release focused current-ratio resources

    def test_ratio_change_releases_previous_resource_before_loading_next(self):  # Verify bounded one-ratio worker ownership
        """
        Verify a worker closes its previous ratio before opening a different ratio.

        :param self: FeatureSetProcessTests instance.
        :return: None.
        """

        _, _, tasks = self.build_server_plan()  # Build authoritative task metadata
        ratio_tasks = [next(task for task in tasks if task["augmentation_ratio"] == ratio) for ratio in (0.25, 0.50)]  # Select two consecutive ratio identities for one feature worker
        pending = {ratio_tasks[0]["feature_set"]: ratio_tasks}  # Build one feature-local sequential pending queue
        status_state = stacking.create_feature_process_status(mp.get_context("spawn"), ratio_tasks, pending)  # Initialize dynamic shared status for both ratios
        process_payload = {"feature_set": ratio_tasks[0]["feature_set"], "attack_types_combined": ["BENIGN", "ATTACK"]}  # Provide small worker metadata
        model_maps = {task["hyperparameters_enabled"]: {task["classifier_name"]: object()} for task in ratio_tasks}  # Provide process-local prototypes for both tasks
        resource_state = {"original_resources": None, "ratio_data": None, "active_ratio": None, "cache_dict": {}}  # Start with no active ratio
        status_queue = queue.Queue()  # Capture only small lifecycle messages
        first_ratio = {"ratio": 0.25, "X_raw": np.ones((1, 1)), "y_encoded": np.array([0])}  # Represent the first required ratio resource
        second_ratio = {"ratio": 0.50, "X_raw": np.ones((2, 1)), "y_encoded": np.array([0, 1])}  # Represent the next required ratio resource
        cleanup_calls = []  # Record resource identities at every production cleanup boundary
        cleanup_implementation = stacking.cleanup_feature_process_ratio_data  # Preserve the real bounded release behavior before instrumentation
        def record_cleanup(ratio_data):  # Record and perform one production ratio release
            """
            Record one ratio cleanup boundary.

            :param ratio_data: Active ratio mapping or None.
            :return: None.
            """

            cleanup_calls.append(None if ratio_data is None else ratio_data.get("ratio"))  # Preserve the ratio identity before the mapping is cleared
            cleanup_implementation(ratio_data)  # Perform the real memmap and mapping release
        with mock.patch.object(stacking, "acquire_feature_process_combination_lock", side_effect=[mock.Mock(), mock.Mock()]), mock.patch.object(stacking, "reload_feature_process_task_cache", return_value=(None, {})), mock.patch.object(stacking, "load_feature_process_ratio_data", side_effect=[first_ratio, second_ratio]) as ratio_open_mock, mock.patch.object(stacking, "cleanup_feature_process_ratio_data", side_effect=record_cleanup), mock.patch.object(stacking, "evaluate_feature_process_augmented_task", return_value={"persisted": True}), mock.patch.object(stacking, "log_feature_process_combination"):  # Instrument exact ratio open and close ordering
            for task in ratio_tasks:  # Process both ratios sequentially in one persistent worker state
                stacking.process_feature_process_task(task, process_payload, model_maps, resource_state, status_queue, status_state)  # Load only the current task's required ratio
        self.assertEqual([call.args[1] for call in ratio_open_mock.call_args_list], [0.25, 0.50])  # Open only the required ratios in task order
        self.assertEqual(cleanup_calls, [None, 0.25])  # Release the first ratio before opening the second without retaining both
        self.assertEqual(first_ratio, {})  # Drop every first-ratio reference after switching
        self.assertEqual(resource_state["active_ratio"], 0.50)  # Retain only the currently useful ratio
        cleanup_implementation(resource_state["ratio_data"])  # Release the final ratio at the worker terminal boundary

    def test_cli_config_validation_and_sequential_fallback(self):  # Verify CLI parsing, config precedence, rejection, and disabled behavior
        """
        Verify feature-set worker CLI parsing and sequential fallback behavior.

        :param self: FeatureSetProcessTests instance.
        :return: None.
        """

        production_arguments = ["stacking.py", "--verbose", "--combined-files", "--disable-stacking", "--enable-augmentation", "--test-augmentation", "--enable-feature-selection", "--enable-hyperparameters", "--disable-automl", "--dataset-file-format=csv", "--augmentation-file-format=csv", "--enable-memory-watcher", "--dataset-path", "./Datasets/CICDDoS2019/01-12/", "--feature-sets", "ga,pca,rfe", "--n-jobs", "1", "--feature-extraction-n-jobs", "1", "--feature-set-workers", "ga=1,pca=1,rfe=1", "--disable-explainability"]  # Mirror the stacking-full target and stated production arguments
        with mock.patch.object(sys, "argv", production_arguments):  # Parse the exact effective production command
            cli_args = stacking.parse_cli_args()  # Exercise production CLI parsing
        repository_config = yaml.safe_load((Path(stacking.__file__).resolve().parent / "config.yaml").read_text(encoding="utf-8"))  # Load the actual production configuration fallback
        merged = stacking.merge_configs(stacking.get_default_config(), repository_config, cli_args)  # Exercise CLI-over-config precedence
        self.assertEqual(merged["evaluation"]["feature_set_workers"], {"full": 0, "ga": 1, "pca": 1, "rfe": 1})  # Preserve exact committed three-worker runtime configuration with Full disabled
        self.assertEqual(merged["evaluation"]["n_jobs"], 1)  # Preserve estimator-level parallelism independently
        self.assertEqual(merged["evaluation"]["feature_extraction_n_jobs"], 1)  # Preserve feature-extraction parallelism independently
        self.assertEqual(merged["stacking"]["feature_sets_config"], {"use_full": False, "use_pca": True, "use_rfe": True, "use_ga": True, "explicit_features": []})  # Resolve only the stated GA, PCA, and RFE queues
        self.assertFalse(merged["stacking"]["methods"]["stacking"])  # Preserve individual-classifier-only execution for the stated command
        watcher_stub = mock.Mock(pid=9876)  # Represent the independently monitored watcher sidecar
        with tempfile.TemporaryDirectory() as temporary_directory, mock.patch.object(stacking, "MEMORY_WATCHER_PROCESS", None), mock.patch.object(stacking, "MEMORY_WATCHER_RUN_DIR", None), mock.patch.object(stacking, "MEMORY_WATCHER_PHASE_STATE_PATH", None), mock.patch.object(stacking, "create_memory_watcher_run_directory", return_value=temporary_directory), mock.patch.object(stacking, "write_memory_phase_event"), mock.patch.object(stacking.subprocess, "Popen", return_value=watcher_stub) as watcher_process_mock:  # Observe watcher preservation without starting a real diagnostics process
            self.assertIs(stacking.start_memory_watcher(config=merged), watcher_stub)  # Preserve enabled memory monitoring alongside feature workers
        watcher_process_mock.assert_called_once()  # Require one watcher sidecar independent of feature-worker count
        self.assertEqual(stacking.FEATURE_PROCESS_START_METHOD, "spawn")  # Require the explicitly selected clean-interpreter start method
        self.assertFalse(stacking.persistent_feature_set_processes_enabled(FEATURE_SET_NAMES, config=stacking.get_default_config()))  # Preserve established sequential execution when disabled
        self.assertFalse(stacking.persistent_feature_set_processes_enabled(["Full Features"], config=stacking.get_default_config()))  # Preserve sequential Full Features execution when persistent workers are disabled
        selected_worker_scenarios = [("full=1", ["Full Features"]), ("full=1,ga=1", ["Full Features", "GA Features"]), ("full=1,pca=1,rfe=1", ["Full Features", "PCA Components", "RFE Features"]), ("ga=1,pca=1,rfe=1", FEATURE_SET_NAMES)]  # Cover supported selected-feature worker subsets
        for worker_specification, active_features in selected_worker_scenarios:  # Validate only enabled and selected feature workers
            worker_config = {"evaluation": {"feature_set_workers": stacking.validate_feature_set_workers(worker_specification, "--feature-set-workers")}}  # Normalize one CLI-equivalent mapping with disabled fallbacks
            self.assertTrue(stacking.persistent_feature_set_processes_enabled(active_features, config=worker_config))  # Enable one persistent process for every active feature only
        full_arguments = ["stacking.py", "--combined-files", "--disable-stacking", "--feature-sets", "full,ga,pca,rfe", "--feature-set-workers", "full=1,ga=1,pca=1,rfe=1"]  # Build the requested four-worker CLI selection
        with mock.patch.object(sys, "argv", full_arguments):  # Parse Full through the same production CLI boundary
            full_cli_args = stacking.parse_cli_args()  # Exercise the complete Full worker override
        full_merged = stacking.merge_configs(stacking.get_default_config(), repository_config, full_cli_args)  # Apply CLI precedence over disabled YAML fallbacks
        self.assertEqual(full_merged["evaluation"]["feature_set_workers"], {"full": 1, "ga": 1, "pca": 1, "rfe": 1})  # Require four normalized persistent worker identities
        self.assertEqual(full_merged["stacking"]["feature_sets_config"], {"use_full": True, "use_pca": True, "use_rfe": True, "use_ga": True, "explicit_features": []})  # Select all four runtime feature identities
        with self.assertRaisesRegex(ValueError, "greater than 1"):  # Require explicit rejection instead of false multi-worker behavior
            stacking.validate_feature_set_workers("ga=2,pca=1,rfe=1", "--feature-set-workers")  # Reject unsafe multiple workers per feature set
        with self.assertRaisesRegex(ValueError, "greater than 1"):  # Apply the same one-worker safety limit to Full Features
            stacking.validate_feature_set_workers("full=2", "--feature-set-workers")  # Reject unsafe multiple Full workers

    def test_valid_cached_combinations_never_enter_worker_queues(self):  # Verify complete cache-first exclusion
        """
        Verify every valid cached combination is excluded before child startup.

        :param self: FeatureSetProcessTests instance.
        :return: None.
        """

        _, _, tasks = self.build_server_plan()  # Build all explicit server-shaped tasks
        total_combinations = len(tasks)  # Derive complete count from generated task plan
        cache_dict = {}  # Build a complete compatible cache mapping
        for task in tasks:  # Populate every authoritative cache identity
            cache_key = stacking.build_resume_cache_key("combined_files", task["data_source_label"], task["experiment_mode"], task["augmentation_ratio"], ["BENIGN", "ATTACK"], task["feature_set"], task["classifier_name"], task["hyperparameters_enabled"])  # Build the unchanged production cache key
            cache_dict[cache_key] = make_process_cache_row(task)  # Store exact compatible feature and sample metadata
        process_payload = {"attack_types_combined": ["BENIGN", "ATTACK"], "execution_mode": "combined_files", "feature_mode_names": list(FEATURE_SET_NAMES)}  # Provide only cache partition context
        cached_results, pending = stacking.partition_feature_process_tasks(tasks, cache_dict, process_payload, {})  # Classify all combinations before any data or child startup
        self.assertEqual(len(cached_results), total_combinations)  # Require every valid cache row to recover
        self.assertTrue(all(not queue_tasks for queue_tasks in pending.values()))  # Require no cached task to be fitted or enqueued
        with tempfile.TemporaryDirectory() as temporary_directory:  # Isolate cache-only persistent process evidence
            probe_payload = make_probe_payload(temporary_directory)  # Build a matrix-free cache-only lifecycle payload
            status_snapshot = stacking.execute_feature_set_processes(pending, probe_payload, tasks, cached_results, process_context=mp.get_context("spawn"), process_target=run_feature_process_probe)  # Start configured persistent children with plan-derived cache-only state and no fitted tasks
            self.assertEqual(status_snapshot["global"]["completed"], total_combinations)  # Preserve dynamic cache-inclusive completed total without computation
            self.assertEqual(status_snapshot["global"]["cached"], total_combinations)  # Classify every cache hit authoritatively
            self.assertEqual(status_snapshot["global"]["computed"], 0)  # Prove cache-only startup performs no fits
            evidence = [json.loads((Path(temporary_directory) / f"{key}.json").read_text(encoding="utf-8")) for key in ("ga", "pca", "rfe")]  # Load cache-only child queue evidence
            self.assertTrue(all(not row["tasks"] for row in evidence))  # Require every valid cached combination to bypass fitting

    def test_payload_rejects_full_matrices_and_memmaps_preserve_arrays_and_ownership(self):  # Verify matrix-free tasks and exact disk-backed arrays and cleanup ownership
        """
        Verify full matrices cannot cross the process boundary and memmap subsets are exact.

        :param self: FeatureSetProcessTests instance.
        :return: None.
        """

        with self.assertRaisesRegex(TypeError, "forbidden matrix data"):  # Require explicit task payload rejection
            stacking.validate_feature_process_payload({"task": {"X": np.ones((2, 2))}})  # Attempt to pickle a forbidden full matrix
        with tempfile.TemporaryDirectory() as temporary_directory:  # Isolate source and worker-owned memmaps
            source_values = np.arange(30, dtype=np.float64).reshape(6, 5)  # Build deterministic source features
            source_descriptor = stacking.persist_feature_process_array(source_values, temporary_directory, "source", chunk_rows=2)  # Persist exact source values without dtype conversion
            label_values = np.array([1, 0, 1, 1, 0, 0], dtype=np.int64)  # Build deterministic estimator labels
            label_descriptor = stacking.persist_feature_process_array(label_values, temporary_directory, "labels", chunk_rows=2)  # Persist exact labels without dtype conversion
            source_memmap = stacking.open_feature_process_array(source_descriptor)  # Reopen source bytes read-only
            label_memmap = stacking.open_feature_process_array(label_descriptor)  # Reopen label bytes read-only
            subset_memmap = stacking.materialize_feature_process_subset(source_memmap, [4, 1], temporary_directory, "subset", 2)  # Materialize exact ordered columns in bounded slices
            np.testing.assert_array_equal(subset_memmap, source_values[:, [4, 1]])  # Require numerically identical estimator features
            np.testing.assert_array_equal(label_memmap, label_values)  # Require numerically identical estimator labels
            self.assertIsInstance(subset_memmap, np.memmap)  # Require real disk-backed storage
            stacking.close_feature_source_memmap(subset_memmap, "subset")  # Release the worker-owned subset mapping
            stacking.close_feature_source_memmap(source_memmap, "source")  # Release the coordinator-owned source mapping
            stacking.close_feature_source_memmap(label_memmap, "labels")  # Release the coordinator-owned label mapping
            owned_directory = Path(temporary_directory) / "owned"  # Resolve one exact process-owned cleanup target
            owned_directory.mkdir()  # Create only the isolated ownership directory
            owned_descriptor = stacking.persist_feature_process_array(source_values, str(owned_directory), "owned", chunk_rows=2)  # Persist one ownership-bound backing file
            owned_memmap = stacking.open_feature_process_array(owned_descriptor)  # Keep the exact owned mapping open until owner cleanup
            stacking.cleanup_feature_process_directory(str(owned_directory), arrays=[owned_memmap])  # Close the owned mapping before removing its exact backing directory
            self.assertFalse(owned_directory.exists())  # Require deterministic owner-controlled backing cleanup
            self.assertTrue(Path(source_descriptor["path"]).is_file())  # Require unrelated coordinator backing to remain intact

    def test_augmentation_loader_retains_only_one_required_ratio(self):  # Verify exact lazy current-ratio loading and release
        """
        Verify augmentation loading occurs only for one requested ratio at a time.

        :param self: FeatureSetProcessTests instance.
        :return: None.
        """

        with tempfile.TemporaryDirectory() as temporary_directory:  # Isolate one small augmented CSV
            augmented_path = Path(temporary_directory) / "augmented.csv"  # Resolve the isolated augmentation source
            augmented_path.write_text("feature_a,feature_b,label\n1,10,A\n2,20,B\n3,30,A\n4,40,B\n", encoding="utf-8")  # Persist deterministic valid augmented rows
            payload = {"execution_mode": "separate_files", "augmentation_file_paths": [str(augmented_path)], "config": {"stacking": {"augmentation_file_format": "csv"}, "execution": {"low_memory": False}}, "original_sample_count": 4, "input_feature_names": ["feature_a", "feature_b"]}  # Build small ratio-loading metadata
            with mock.patch.object(stacking, "load_augmented_dataset", wraps=stacking.load_augmented_dataset) as load_mock:  # Observe production augmented source loading
                ratio_data = stacking.load_feature_process_ratio_data(payload, 0.5)  # Load only the exact requested 50 percent ratio
            self.assertEqual(load_mock.call_count, 1)  # Require one source load for one required ratio
            self.assertEqual(ratio_data["ratio"], 0.5)  # Require only the current ratio identity
            self.assertEqual(len(ratio_data["y_raw"]), 2)  # Require exact deterministic sampled cardinality
            stacking.cleanup_feature_process_ratio_data(ratio_data)  # Release current-ratio DataFrame and array views
            self.assertEqual(ratio_data, {})  # Require no retained augmentation ratio cache
            shared_directory = Path(temporary_directory) / "shared_ratios"  # Resolve one coordinator-owned shared ratio directory
            shared_directory.mkdir()  # Create only the isolated shared ratio directory
            payload["augmentation_shared_directory"] = str(shared_directory)  # Enable lazy cross-feature ratio reuse through a small path
            payload["label_classes"] = ["A", "B"]  # Preserve the original-training label identity for shared encoding
            with mock.patch.object(stacking, "load_augmented_dataset", wraps=stacking.load_augmented_dataset) as shared_load_mock:  # Observe complete augmented-source reconstruction across repeated consumers
                first_ratio_data = stacking.load_feature_process_ratio_data(payload, 0.5)  # Let the first feature consumer publish the required ratio
                first_features = np.array(first_ratio_data["X_raw"], copy=True)  # Preserve small deterministic evidence before mapping release
                first_labels = np.array(first_ratio_data["y_encoded"], copy=True)  # Preserve exact encoded-label evidence before mapping release
                stacking.cleanup_feature_process_ratio_data(first_ratio_data)  # Release the first consumer's current-ratio mappings
                second_ratio_data = stacking.load_feature_process_ratio_data(payload, 0.5)  # Reopen the same shared ratio for another feature consumer
            self.assertEqual(shared_load_mock.call_count, 1)  # Reconstruct the complete augmented source only once for the shared ratio
            self.assertIsInstance(second_ratio_data["X_raw"], np.memmap)  # Require shared raw feature bytes to remain disk-backed
            self.assertIsInstance(second_ratio_data["y_encoded"], np.memmap)  # Require shared encoded labels to remain disk-backed
            np.testing.assert_array_equal(second_ratio_data["X_raw"], first_features)  # Preserve exact sampled feature values and ordering across consumers
            np.testing.assert_array_equal(second_ratio_data["y_encoded"], first_labels)  # Preserve exact sampled label values and ordering across consumers
            self.assertEqual(len(list(shared_directory.glob("ratio_*.json"))), 1)  # Publish only the one genuinely required ratio descriptor
            stacking.cleanup_feature_process_ratio_data(second_ratio_data)  # Release the second consumer's current-ratio mappings
            stacking.cleanup_feature_process_directory(str(shared_directory))  # Delete coordinator-owned ratio backing after every consumer releases it
            self.assertFalse(shared_directory.exists())  # Require deterministic post-consumer ratio cleanup

    def test_shared_preprocessing_memmaps_match_sequential_features_and_labels(self):  # Verify exact scientific split and coordinator ownership
        """
        Verify shared memmaps preserve the sequential preprocessing outputs exactly.

        :param self: FeatureSetProcessTests instance.
        :return: None.
        """

        original_df = pd.DataFrame({"feature_a": np.arange(20, dtype=np.float64), "feature_b": np.arange(20, dtype=np.float64) * 3.0, "attack_type": ["A", "B"] * 10})  # Build deterministic balanced original data
        with tempfile.TemporaryDirectory() as temporary_directory:  # Isolate coordinator-owned backing files
            config = {"stacking": {"memory_management": {"spill_directory": temporary_directory}}}  # Route only this focused memmap lifecycle to the isolated directory
            sequential = stacking.prepare_evaluation_data_splits(original_df, config=config)  # Compute the established scientific preprocessing outputs
            resources = stacking.create_feature_process_shared_resources(original_df, str(Path(temporary_directory) / "dataset.csv"), config)  # Persist the same deterministic preprocessing outputs once
            shared_memmaps = [stacking.open_feature_process_array(resources[name]) for name in ("X_train_scaled", "X_test_scaled", "y_train", "y_test")]  # Reopen only read-only descriptors as a worker would
            for shared_array, sequential_array in zip(shared_memmaps, sequential[:4]):  # Compare every estimator feature and label array
                np.testing.assert_array_equal(shared_array, sequential_array)  # Require byte-equivalent scientific inputs
            preprocessing_bundle = stacking.load(resources["preprocessing_path"])  # Load only the small fitted preprocessing bundle
            np.testing.assert_array_equal(preprocessing_bundle["scaler"].mean_, sequential[4].mean_)  # Require identical fitted scaling state
            np.testing.assert_array_equal(preprocessing_bundle["label_encoder"].classes_, sequential[5].classes_)  # Require identical label identity and ordering
            shared_directory = Path(resources["temp_dir"])  # Resolve the coordinator's exact ownership-bearing directory
            self.assertTrue(shared_directory.is_dir())  # Keep backing files alive while worker mappings may exist
            stacking.cleanup_feature_process_directory(str(shared_directory), arrays=shared_memmaps)  # Close all worker mappings before coordinator-owned deletion
            self.assertFalse(shared_directory.exists())  # Require complete post-use backing cleanup

    def test_full_original_resources_reuse_shared_memmaps_and_match_sequential(self):  # Verify exact Full inputs and coordinator ownership
        """Verify Full reopens sequential-equivalent shared matrices without another complete copy."""

        original_df = pd.DataFrame({"feature_a": np.arange(20, dtype=np.float64), "feature_b": np.arange(20, dtype=np.float64) * 3.0, "attack_type": ["A", "B"] * 10})  # Build deterministic balanced original data
        with tempfile.TemporaryDirectory() as temporary_directory:  # Isolate coordinator-owned Full backing files
            config = {"stacking": {"memory_management": {"spill_directory": temporary_directory}}}  # Route focused memmaps to the isolated directory
            sequential = stacking.prepare_evaluation_data_splits(original_df, config=config)  # Compute established sequential Full inputs
            shared = stacking.create_feature_process_shared_resources(original_df, str(Path(temporary_directory) / "dataset.csv"), config)  # Persist one coordinator-owned copy
            metadata = stacking.build_feature_process_metadata(["feature_a", "feature_b"], ["feature_a"], 1, ["feature_b"])["Full Features"]  # Build ordered Full identity metadata
            payload = {"shared_resources": shared, "feature_set": "Full Features", "feature_metadata": metadata, "config": config, "file": str(Path(temporary_directory) / "dataset.csv"), "input_feature_names": ["feature_a", "feature_b"], "source_files": [], "pca_cache_context": {}}  # Pass only small descriptors to Full resource preparation
            with mock.patch.object(stacking, "create_feature_process_temp_directory") as temp_directory_mock, mock.patch.object(stacking, "materialize_feature_process_subset") as subset_mock, mock.patch.object(stacking, "load_feature_process_ratio_data") as ratio_load_mock:  # Reject any Full copy or augmentation access
                resources = stacking.prepare_feature_process_original_resources(payload)  # Reopen coordinator backing read-only
            temp_directory_mock.assert_not_called()  # Full owns no duplicate worker matrix directory
            subset_mock.assert_not_called()  # Full never materializes an all-column subset copy
            ratio_load_mock.assert_not_called()  # Original Full preparation loads no augmentation data
            self.assertEqual(str(resources["X_train"].filename), shared["X_train_scaled"]["path"])  # Reuse exact coordinator training backing identity
            self.assertEqual(str(resources["X_test"].filename), shared["X_test_scaled"]["path"])  # Reuse exact coordinator testing backing identity
            self.assertIsNone(resources["temp_dir"])  # Record no worker-owned Full backing directory
            np.testing.assert_array_equal(resources["X_train"], sequential[0])  # Preserve exact Full training values and order
            np.testing.assert_array_equal(resources["X_test"], sequential[1])  # Preserve exact Full testing values and order
            np.testing.assert_array_equal(resources["y_train"], sequential[2])  # Preserve exact training labels
            np.testing.assert_array_equal(resources["y_test"], sequential[3])  # Preserve exact testing labels
            shared_directory = Path(shared["temp_dir"])  # Resolve coordinator ownership boundary
            stacking.cleanup_feature_process_original_resources(resources)  # Close only Full worker mappings
            self.assertTrue(shared_directory.is_dir())  # Keep shared backing alive until all dependent workers exit
            stacking.cleanup_feature_process_directory(str(shared_directory))  # Delete coordinator backing after worker release
            self.assertFalse(shared_directory.exists())  # Require deterministic final ownership cleanup

    def test_augmented_full_reuses_scaled_matrix_without_all_column_copy(self):  # Verify Full augmented memory conservation
        """Verify augmented Full passes scaler output directly to existing evaluator."""

        scaled = np.arange(12, dtype=np.float64).reshape(6, 2)  # Build deterministic complete scaled Full matrix
        scaler = mock.Mock()  # Represent persisted original-training scaler
        scaler.transform.return_value = scaled  # Return the exact matrix whose identity must be preserved
        artifact_bundle = {"model": object(), "scaler": scaler, "transformer": None, "input_feature_names": ["a", "b"], "model_feature_names": ["a", "b"], "label_encoder": mock.Mock()}  # Build complete Full artifact metadata
        task = {"feature_set": "Full Features", "hyperparameters_enabled": False, "classifier_name": "Random Forest", "augmentation_ratio": 0.5, "expected_feature_names": ["a", "b"], "expected_n_features": 2, "expected_n_samples_train": 8, "data_source_label": "Augmented@50%_CombinedFiles", "experiment_id": "experiment", "experiment_mode": "original_training_augmented_testing"}  # Build one augmented Full task
        payload = {"file": "dataset.csv", "cache_ref_file": "dataset.csv", "execution_mode": "combined_files", "attack_types_combined": None, "config": {}, "optimized_params": {}}  # Provide only small evaluation metadata
        ratio_data = {"X_raw": np.ones((6, 2), dtype=np.float64), "y_encoded": np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)}  # Provide one required ratio resource
        with mock.patch.object(stacking, "build_feature_process_artifact_context", return_value={}), mock.patch.object(stacking, "build_filename_safe_dataset_identity", return_value="dataset"), mock.patch.object(stacking, "load_existing_model_if_available", return_value=(artifact_bundle, None)), mock.patch.object(stacking, "evaluate_individual_classifier", return_value={"Accuracy": 1.0}) as evaluate_mock, mock.patch.object(stacking, "build_classifier_result_entry", return_value={"persisted": True}), mock.patch.object(stacking, "persist_cache_result_entry"), mock.patch.object(stacking, "log_feature_process_combination"):  # Isolate scientific evaluator input identity
            stacking.evaluate_feature_process_augmented_task(task, payload, ratio_data, object(), {}, {})  # Reuse existing augmented evaluator for Full
        self.assertIs(evaluate_mock.call_args.args[4], scaled)  # Pass scaler output directly without an identical all-column matrix copy

    def test_augmented_task_accepts_shared_encoded_labels_without_local_frame(self):  # Verify shared ratio resources flow through the generic task path
        """
        Verify augmented tasks consume shared encoded labels without local frame data.

        :param self: FeatureSetProcessTests instance.
        :return: None.
        """

        _, _, tasks = self.build_server_plan()  # Build exact server-shaped task identities
        task = next(task for task in tasks if task["augmentation_ratio"] is not None)  # Select the first augmented-testing combination
        process_payload = {"feature_set": task["feature_set"], "attack_types_combined": ["BENIGN", "ATTACK"]}  # Build only small metadata consumed outside patched production boundaries
        model_prototype = object()  # Represent one process-local estimator prototype without fitting
        model_maps = {False: {task["classifier_name"]: model_prototype}, True: {}}  # Preserve the task's hyperparameter and classifier lookup identity
        resource_state = {"original_resources": None, "ratio_data": None, "active_ratio": None, "cache_dict": {}}  # Initialize an empty bounded worker resource state
        pending = {task["feature_set"]: [task]}  # Build one dynamic pending queue for shared status initialization
        status_state = stacking.create_feature_process_status(mp.get_context("spawn"), [task], pending)  # Build production synchronized lifecycle counters
        status_queue = queue.Queue()  # Capture the small progress record locally
        combination_lock = mock.Mock()  # Observe complete reservation release after persistence
        shared_ratio = {"ratio": task["augmentation_ratio"], "X_raw": np.ones((2, 2), dtype=np.float64), "y_encoded": np.array([0, 1], dtype=np.int64), "shared": True}  # Provide shared ratio resources without a y_raw frame view
        with mock.patch.object(stacking, "acquire_feature_process_combination_lock", return_value=combination_lock), mock.patch.object(stacking, "reload_feature_process_task_cache", return_value=(None, {})), mock.patch.object(stacking, "load_feature_process_ratio_data", return_value=shared_ratio), mock.patch.object(stacking, "evaluate_feature_process_augmented_task", return_value={"persisted": True}) as evaluation_mock, mock.patch.object(stacking, "log_feature_process_combination"):  # Isolate scheduling while retaining the generic augmented task flow
            stacking.process_feature_process_task(task, process_payload, model_maps, resource_state, status_queue, status_state)  # Execute one shared-label augmented combination
        evaluation_mock.assert_called_once()  # Require the existing augmented evaluator to receive the shared ratio resource
        final_status = stacking.read_feature_process_status(status_state)  # Read exact cache, pending, running, computed, failed, and completed state
        self.assertEqual(final_status["global"]["computed"], 1)  # Count the durably persisted combination once
        self.assertEqual(final_status["global"]["completed"], 1)  # Include successful computation in completed status exactly once
        lifecycle_events = [status_queue.get_nowait()["status"], status_queue.get_nowait()["event"]]  # Read running followed by computed lifecycle transport
        self.assertEqual(lifecycle_events, ["running", "computed"])  # Report exact state order without false cache recovery
        combination_lock.close.assert_called_once()  # Release the complete combination reservation after persistence
        stacking.cleanup_feature_process_ratio_data(resource_state["ratio_data"])  # Release the focused shared ratio resources

    def test_three_spawned_processes_progress_independently_and_logs_are_readable(self):  # Verify exact process count, feature isolation, timing independence, and log locking
        """
        Verify three real spawned processes run feature-local queues independently.

        :param self: FeatureSetProcessTests instance.
        :return: None.
        """

        with tempfile.TemporaryDirectory() as temporary_directory:  # Isolate spawned completion and log evidence
            payload = make_probe_payload(temporary_directory, delays={"GA Features": 0.05, "PCA Components": 0.8, "RFE Features": 0.05})  # Make PCA deliberately slower than GA and RFE
            tasks, pending = make_probe_tasks(tasks_per_feature=2)  # Give every persistent child two sequential matrix-free descriptors
            baseline_children = {child.pid for child in mp.active_children()}  # Snapshot unrelated active children before the focused run
            pending_total = sum(len(queue_tasks) for queue_tasks in pending.values())  # Derive global pending total from active feature queues
            final_status = stacking.execute_feature_set_processes(pending, payload, tasks, {}, process_context=mp.get_context("spawn"), process_target=run_feature_process_probe)  # Exercise real start, monitor, join, and close coordinator with dynamic total
            self.assertEqual(final_status["global"]["completed"], pending_total)  # Require one completed task from every active persistent child
            self.assertEqual(final_status["global"]["computed"], pending_total)  # Require every uncached probe task to be computed
            evidence = [json.loads((Path(temporary_directory) / f"{key}.json").read_text(encoding="utf-8")) for key in ("ga", "pca", "rfe")]  # Load exact spawned process evidence
            self.assertEqual(len({row["pid"] for row in evidence}), len(FEATURE_SET_NAMES))  # Require one distinct real child PID per active feature set
            self.assertTrue(all(row["ppid"] == os.getpid() for row in evidence))  # Require every child to belong to the coordinator
            self.assertTrue(all(row["tasks"] == [row["feature_set"], row["feature_set"]] for row in evidence))  # Require strict feature-set queue isolation within one persistent PID
            finished_by_feature = {row["feature_set"]: row["finished"] for row in evidence}  # Resolve independent terminal timestamps
            self.assertLess(finished_by_feature["GA Features"], finished_by_feature["PCA Components"])  # Require slow PCA not to block GA
            self.assertLess(finished_by_feature["RFE Features"], finished_by_feature["PCA Components"])  # Require slow PCA not to block RFE
            remaining_children = {child.pid for child in mp.active_children()} - baseline_children  # Identify any process left by the focused coordinator
            self.assertEqual(remaining_children, set())  # Require no orphan or zombie child remains
            log_lines = [line for line in (Path(temporary_directory) / "concurrent.log").read_text(encoding="utf-8").splitlines() if line]  # Read complete nonblank process-locked records
            expected_log_lines = len(FEATURE_SET_NAMES) * 10  # Derive expected records from active processes and per-probe writes
            self.assertEqual(len(log_lines), expected_log_lines)  # Require every concurrent log record exactly once
            self.assertEqual(len(set(log_lines)), expected_log_lines)  # Require no interleaved, duplicated, or partial records

    def test_four_spawned_workers_are_isolated_and_finish_independently(self):  # Verify Full extends the generic real-process coordinator
        """Verify one Full, GA, PCA, and RFE worker with isolated queues and independent completion."""

        delays = {"Full Features": 0.7, "GA Features": 0.0, "PCA Components": 1.4, "RFE Features": 0.0}  # Make Full slower than GA/RFE but faster than PCA
        with tempfile.TemporaryDirectory() as temporary_directory:  # Isolate four-worker completion and logging evidence
            payload = make_probe_payload(temporary_directory, delays=delays, feature_names=FOUR_FEATURE_SET_NAMES)  # Configure exactly four feature-worker roles
            tasks, pending = make_probe_tasks(FOUR_FEATURE_SET_NAMES, tasks_per_feature=2)  # Assign two matrix-free combinations to each sole worker
            stacking.validate_feature_process_payload({"tasks": tasks})  # Prove Full task payloads contain no matrix data
            baseline_children = {child.pid for child in mp.active_children()}  # Snapshot unrelated active children before startup
            captured_tree = {}  # Capture production role classification without changing it
            original_tree_logger = stacking.log_feature_process_tree  # Preserve real process-tree classification
            def capture_tree(process_records):  # Record and return production process-tree output
                result = original_tree_logger(process_records)  # Classify feature workers separately from auxiliaries
                captured_tree.update(result)  # Preserve plain role evidence for assertions
                return result  # Keep coordinator behavior unchanged
            with mock.patch.object(stacking, "log_feature_process_tree", side_effect=capture_tree):  # Observe exact configured process roles
                final_status = stacking.execute_feature_set_processes(pending, payload, tasks, {}, process_context=mp.get_context("spawn"), process_target=run_feature_process_probe)  # Start, monitor, join, and close four real workers
            evidence = [json.loads((Path(temporary_directory) / f"{key}.json").read_text(encoding="utf-8")) for key in ("full", "ga", "pca", "rfe")]  # Load one completion record per configured feature worker
            self.assertEqual(len({row["pid"] for row in evidence}), 4)  # Create exactly four distinct persistent feature processes
            self.assertTrue(all(row["tasks"] == [row["feature_set"], row["feature_set"]] for row in evidence))  # Keep every worker queue feature-local
            self.assertEqual({record["feature_set"] for record in captured_tree["feature_workers"]}, set(FOUR_FEATURE_SET_NAMES))  # Classify Full and three committed workers as feature workers
            self.assertTrue(all(record["role"] == "Feature Worker" for record in captured_tree["feature_workers"]))  # Keep auxiliaries outside feature-worker count
            finished = {row["feature_set"]: row["finished"] for row in evidence}  # Resolve independent terminal timestamps
            self.assertLess(finished["GA Features"], finished["Full Features"])  # Slow Full does not block GA
            self.assertLess(finished["RFE Features"], finished["Full Features"])  # Slow Full does not block RFE
            self.assertLess(finished["Full Features"], finished["PCA Components"])  # Full can finish while another worker remains active
            self.assertEqual(final_status["global"], {"total": 8, "cached": 0, "pending": 0, "running": 0, "computed": 8, "failed": 0, "completed": 8})  # Preserve exact dynamic global status
            self.assertTrue(all(status["completed"] == 2 for status in final_status["features"].values()))  # Complete each feature-local queue independently
            remaining_children = {child.pid for child in mp.active_children()} - baseline_children  # Identify leaked focused children
            self.assertEqual(remaining_children, set())  # Reap all four workers after success

    def test_single_enabled_feature_creates_one_persistent_worker(self):  # Verify process creation follows the actual enabled feature plan
        """
        Verify one enabled feature set creates one persistent feature worker.

        :param self: FeatureSetProcessTests instance.
        :return: None.
        """

        active_features = ["GA Features"]  # Enable only one feature-set worker role
        with tempfile.TemporaryDirectory() as temporary_directory:  # Isolate single-worker evidence and logging
            payload = make_probe_payload(temporary_directory, feature_names=active_features)  # Build one configured GA process payload
            tasks, pending = make_probe_tasks(active_features, tasks_per_feature=2)  # Assign two combinations to the same persistent GA queue
            baseline_children = {child.pid for child in mp.active_children()}  # Snapshot unrelated children before the focused run
            final_status = stacking.execute_feature_set_processes(pending, payload, tasks, {}, process_context=mp.get_context("spawn"), process_target=run_feature_process_probe)  # Start, monitor, join, and close the one configured feature worker
            evidence = json.loads((Path(temporary_directory) / "ga.json").read_text(encoding="utf-8"))  # Load exact single-worker process evidence
            self.assertEqual(evidence["tasks"], active_features * 2)  # Process both GA combinations sequentially in one PID
            self.assertEqual(final_status["global"]["total"], len(tasks))  # Derive total from the two-task authoritative plan
            self.assertEqual(final_status["global"]["completed"], len(tasks))  # Complete both tasks exactly once
            remaining_children = {child.pid for child in mp.active_children()} - baseline_children  # Identify any leaked focused child
            self.assertEqual(remaining_children, set())  # Leave no orphan or zombie after single-feature completion

    def test_process_tree_separates_feature_workers_from_auxiliaries(self):  # Verify role-aware process-tree reporting
        """
        Verify watcher and resource tracker never inflate feature-worker counts.

        :param self: FeatureSetProcessTests instance.
        :return: None.
        """

        process_records = []  # Build authoritative coordinator-owned feature worker handles
        direct_children = []  # Build matching OS child metadata plus auxiliaries
        for index, feature_name in enumerate(FEATURE_SET_NAMES, start=1):  # Represent exactly one configured worker per feature set
            process_handle = mock.Mock()  # Build one multiprocessing-like feature handle
            process_handle.pid = 1000 + index  # Assign a distinct feature-worker PID
            process_handle.name = f"Stacking-{stacking.resolve_feature_set_worker_key(feature_name).upper()}-1"  # Preserve explicit role-bearing process name
            process_records.append({"feature_set": feature_name, "worker_key": stacking.resolve_feature_set_worker_key(feature_name), "process": process_handle})  # Register authoritative feature ownership
            child_handle = mock.Mock()  # Represent the same worker in OS child discovery
            child_handle.pid = process_handle.pid  # Match the coordinator-owned worker PID
            direct_children.append(child_handle)  # Include the worker among direct OS children
        watcher_child = mock.Mock()  # Represent the existing repository memory watcher
        watcher_child.pid = 2001  # Assign a distinct watcher PID
        watcher_child.cmdline.return_value = [sys.executable, "Scripts/memory_watcher.py"]  # Preserve role-bearing watcher command metadata
        tracker_child = mock.Mock()  # Represent Python's spawn resource tracker
        tracker_child.pid = 2002  # Assign a distinct runtime auxiliary PID
        tracker_child.cmdline.return_value = [sys.executable, "-c", "from multiprocessing.resource_tracker import main"]  # Preserve standard resource-tracker command metadata
        direct_children.extend([watcher_child, tracker_child])  # Include both auxiliaries outside feature-worker ownership
        parent_process = mock.Mock()  # Represent coordinator process inspection
        parent_process.children.return_value = direct_children  # Return exact direct children without recursive descendants
        watcher_process = mock.Mock(pid=watcher_child.pid)  # Represent the globally owned watcher handle
        with mock.patch.object(stacking.psutil, "Process", return_value=parent_process), mock.patch.object(stacking, "MEMORY_WATCHER_PROCESS", watcher_process), mock.patch("builtins.print"):  # Isolate deterministic role classification output
            process_tree = stacking.log_feature_process_tree(process_records)  # Classify configured workers and auxiliary runtime processes
        self.assertEqual(len(process_tree["feature_workers"]), len(FEATURE_SET_NAMES))  # Report exactly three feature workers from authoritative handles
        self.assertEqual({record["role"] for record in process_tree["auxiliary_processes"]}, {"Memory Watcher", "Python Resource Tracker"})  # Report known auxiliaries separately
        self.assertTrue(all(record["pid"] not in {2001, 2002} for record in process_tree["feature_workers"]))  # Never misreport an auxiliary as a feature evaluator

    def test_child_exception_reaches_coordinator_and_siblings_are_reaped(self):  # Verify complete child traceback propagation and deterministic cleanup
        """
        Verify child exceptions reach the coordinator and leave no live siblings.

        :param self: FeatureSetProcessTests instance.
        :return: None.
        """

        with tempfile.TemporaryDirectory() as temporary_directory:  # Isolate partial completion evidence
            payload = make_probe_payload(temporary_directory, delays={"GA Features": 0.01, "PCA Components": 0.2, "RFE Features": 2.0}, failure="PCA Components")  # Complete GA before failing PCA while RFE remains active
            tasks, pending = make_probe_tasks()  # Assign one matrix-free task per feature process
            baseline_children = {child.pid for child in mp.active_children()}  # Snapshot unrelated active children
            with self.assertRaisesRegex(RuntimeError, "Injected PCA Components failure") as raised:  # Require the exact child exception at the coordinator
                stacking.execute_feature_set_processes(pending, payload, tasks, {}, process_context=mp.get_context("spawn"), process_target=run_feature_process_probe)  # Exercise production failure termination and joining with dynamic total
            self.assertIn("Traceback", str(raised.exception))  # Require complete child traceback text
            failure_status = raised.exception.status_snapshot  # Read reconciled counters attached to the surfaced child exception
            self.assertEqual(failure_status["global"]["failed"], 1)  # Count the PCA combination as failed without completion
            self.assertEqual(failure_status["global"]["completed"], 1)  # Preserve only the GA result completed before failure
            self.assertEqual(failure_status["global"]["pending"], 1)  # Return terminated RFE work to pending state for safe resume
            self.assertEqual(failure_status["global"]["running"], 0)  # Leave no task falsely running after every child is reaped
            self.assertTrue((Path(temporary_directory) / "ga.json").is_file())  # Preserve work completed before another child failed
            remaining_children = {child.pid for child in mp.active_children()} - baseline_children  # Identify leaked focused children
            self.assertEqual(remaining_children, set())  # Require no orphan or zombie process after failure

    def test_full_worker_failure_reaches_coordinator_and_all_children_are_reaped(self):  # Verify Full uses generic failure propagation
        """Verify Full failure preserves completed sibling work and reaps every child."""

        delays = {"GA Features": 0.0, "Full Features": 0.7, "PCA Components": 2.0, "RFE Features": 2.0}  # Complete GA before failing Full while PCA/RFE remain active
        with tempfile.TemporaryDirectory() as temporary_directory:  # Isolate partial four-worker completion evidence
            payload = make_probe_payload(temporary_directory, delays=delays, failure="Full Features", feature_names=FOUR_FEATURE_SET_NAMES)  # Inject failure through the Full worker role
            tasks, pending = make_probe_tasks(FOUR_FEATURE_SET_NAMES)  # Assign one task per feature worker
            baseline_children = {child.pid for child in mp.active_children()}  # Snapshot unrelated active children
            with self.assertRaisesRegex(RuntimeError, "Injected Full Features failure") as raised:  # Require exact Full child exception at coordinator
                stacking.execute_feature_set_processes(pending, payload, tasks, {}, process_context=mp.get_context("spawn"), process_target=run_feature_process_probe)  # Exercise generic failure termination and joining
            failure_status = raised.exception.status_snapshot  # Read reconciled process-safe status
            self.assertEqual(failure_status["features"]["Full Features"]["failed"], 1)  # Count failed Full task without completion
            self.assertEqual(failure_status["features"]["GA Features"]["completed"], 1)  # Preserve sibling result completed before Full failed
            self.assertEqual(failure_status["global"]["running"], 0)  # Leave no task falsely active after termination
            self.assertTrue((Path(temporary_directory) / "ga.json").is_file())  # Preserve completed sibling evidence
            remaining_children = {child.pid for child in mp.active_children()} - baseline_children  # Identify leaked feature children
            self.assertEqual(remaining_children, set())  # Reap Full and every sibling after failure

    def test_combination_reservation_serializes_final_cache_and_computation_window(self):  # Verify independent coordinators cannot compute one combination together
        """
        Verify one combination reservation spans independent spawned processes.

        :param self: FeatureSetProcessTests instance.
        :return: None.
        """

        _, _, tasks = self.build_server_plan()  # Build one exact server-scenario combination identity
        with tempfile.TemporaryDirectory() as temporary_directory:  # Isolate combination synchronization artifacts
            dataset_path = Path(temporary_directory) / "dataset.csv"  # Resolve a small dataset identity for cache paths
            dataset_path.write_text("feature,label\n1,A\n2,B\n", encoding="utf-8")  # Persist only path provenance required by cache resolution
            process_payload = {"cache_ref_file": str(dataset_path), "config": stacking.get_default_config(), "attack_types_combined": ["BENIGN", "ATTACK"]}  # Build the small production lock identity payload
            parent_lock = stacking.acquire_feature_process_combination_lock(tasks[0], process_payload)  # Reserve the first combination in the coordinator process
            context = mp.get_context("spawn")  # Use the same clean-interpreter start method as production
            status_queue = context.Queue()  # Carry only child acquisition evidence
            child = context.Process(target=run_combination_lock_probe, args=(tasks[0], process_payload, status_queue), name="Stacking-Combination-Lock-Probe")  # Start one independent contender for the same combination
            child.start()  # Begin the child lock acquisition attempt
            try:  # Prove the child remains blocked until the coordinator releases the reservation
                with self.assertRaises(queue.Empty):  # Require no premature acquisition evidence
                    status_queue.get(timeout=0.3)  # Wait briefly while the parent still owns the reservation
            finally:  # Release the parent reservation under every focused test outcome
                parent_lock.close()  # Allow the blocked child to acquire the exact combination
            evidence = status_queue.get(timeout=10.0)  # Receive child acquisition only after parent release
            child.join(10.0)  # Reap the spawned contender deterministically
            self.assertEqual(child.exitcode, 0)  # Require successful post-release child completion
            self.assertNotEqual(evidence["pid"], os.getpid())  # Require a real independent OS process acquisition
            child.close()  # Release the reaped process handle
            status_queue.close()  # Stop accepting focused status records
            status_queue.join_thread()  # Join the focused queue feeder deterministically


if __name__ == "__main__":  # Support direct focused test execution
    unittest.main()  # Run the persistent feature-set process suite
