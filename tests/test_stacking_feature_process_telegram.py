import multiprocessing as mp  # Exercise coordinator-owned notifications across real spawned feature workers
import os  # Record spawned worker identities in lifecycle events
import queue  # Capture direct task lifecycle events without another process
import time  # Coordinate deterministic failure-drain ordering
import unittest  # Provide repository-standard focused test execution
from unittest import mock  # Isolate Telegram transport and scientific evaluation boundaries

import numpy as np  # Prove notification payloads exclude scientific arrays

import stacking  # Exercise production persistent-process notification behavior


FEATURE_NAMES = ["Full Features", "GA Features", "PCA Components", "RFE Features"]  # Preserve supported persistent feature-worker order


def make_notification_task(feature_set, global_id, dynamic_total, augmentation_ratio=None):  # Build one authoritative matrix-free notification task
    """Build one small feature-process task for deterministic notification tests."""

    return {"feature_set": feature_set, "global_id": global_id, "total_combinations": dynamic_total, "feature_local_position": 1, "feature_local_total": 1, "hyperparameters_enabled": False, "augmentation_ratio": augmentation_ratio, "classifier_name": "Random Forest", "experiment_mode": "original_only" if augmentation_ratio is None else "original_training_augmented_testing", "execution_mode": "combined_files"}  # Preserve every identity used by production message construction


def make_notification_result(task, f1_score=0.75, elapsed_time_s=63):  # Build one small persisted-result notification mapping
    """Build scalar persisted result fields matching one task."""

    return {"feature_set": task["feature_set"], "hyperparameter_mode": "Default Hyperparameters", "model_name": task["classifier_name"], "execution_mode": task["execution_mode"], "experiment_mode": task["experiment_mode"], "augmentation_ratio": 0.0 if task["augmentation_ratio"] is None else task["augmentation_ratio"], "accuracy": 0.8, "precision": 0.77, "recall": 0.76, "f1_score": f1_score, "fpr": 0.1, "fnr": 0.2, "elapsed_time_s": elapsed_time_s}  # Mirror persisted cache fields without matrices or estimators


def make_notification_payload(process_context, feature_names, duplicate=False):  # Build one matrix-free coordinator payload for spawned notification tests
    """Build small persistent-process metadata and a shared computation counter."""

    worker_counts = {key: int(any(stacking.resolve_feature_set_worker_key(name) == key for name in feature_names)) for key in stacking.FEATURE_SET_WORKER_KEYS}  # Enable exactly the selected feature workers
    return {"feature_mode_names": list(feature_names), "feature_metadata_by_name": {name: {"feature_names": [name], "indices": [0], "feature_count": 1} for name in feature_names}, "config": {"evaluation": {"feature_set_workers": worker_counts}}, "notification_duplicate": bool(duplicate), "notification_computations": process_context.Value("q", 0)}  # Return only small metadata and synchronized test evidence


def run_notification_process_probe(process_payload, status_queue, status_state):  # Publish persisted-result events through one spawned feature worker
    """Run a deterministic production-compatible notification lifecycle."""

    feature_set = process_payload["feature_set"]  # Resolve this child worker's sole feature identity
    status_queue.put({"status": "started", "feature_set": feature_set, "worker_index": process_payload["worker_index"], "pid": os.getpid(), "ppid": os.getppid(), "queue_size": len(process_payload["tasks"])})  # Report production-compatible startup
    for task in process_payload["tasks"]:  # Process this feature-local task queue sequentially
        stacking.transition_feature_process_status(status_state, task, "started")  # Move the exact task from pending to running
        status_queue.put({"status": "running", "feature_set": feature_set, "global_id": task["global_id"], "pid": os.getpid()})  # Report active task identity
        with process_payload["notification_computations"].get_lock():  # Serialize shared computation evidence
            process_payload["notification_computations"].value += 1  # Count one simulated durable computation
        result_entry = make_notification_result(task, f1_score=0.7 + task["global_id"] / 100.0, elapsed_time_s=60 + task["global_id"])  # Build deterministic persisted values
        stacking.transition_feature_process_status(status_state, task, "computed")  # Clear running and count successful persisted computation
        stacking.publish_feature_process_result_event(task, process_payload, result_entry, "computed", status_queue)  # Wait until coordinator consumes this exact completion event
        if process_payload.get("notification_duplicate", False):  # Inject one stale duplicate event when requested
            stacking.publish_feature_process_result_event(task, process_payload, result_entry, "computed", status_queue)  # Re-publish identical small metadata after acknowledgement
    status_queue.put({"status": "done", "feature_set": feature_set, "worker_index": process_payload["worker_index"], "pid": os.getpid()})  # Report successful worker shutdown only after notification handling


def run_notification_shutdown_probe(process_payload, status_queue, status_state):  # Exercise notification draining after a sibling worker failure
    """Publish one completion beside one deterministic sibling failure."""

    feature_set = process_payload["feature_set"]  # Resolve this child worker's assigned feature identity
    task = process_payload["tasks"][0]  # Resolve this probe's sole task
    status_queue.put({"status": "started", "feature_set": feature_set, "worker_index": process_payload["worker_index"], "pid": os.getpid(), "ppid": os.getppid(), "queue_size": 1})  # Report production-compatible startup
    if feature_set == "Full Features":  # Publish one successfully persisted result before sibling failure
        stacking.transition_feature_process_status(status_state, task, "started")  # Move Full task to running
        status_queue.put({"status": "running", "feature_set": feature_set, "global_id": task["global_id"], "pid": os.getpid()})  # Report active Full task
        stacking.transition_feature_process_status(status_state, task, "computed")  # Count Full result as durably computed
        notification_result = stacking.build_feature_process_notification_result(make_notification_result(task))  # Build only small persisted fields
        status_queue.put({"status": "progress", "feature_set": feature_set, "global_id": task["global_id"], "event": "computed", "notification_result": notification_result, "pid": os.getpid()})  # Queue completion before releasing failure sibling
        process_payload["notification_barrier"].set()  # Allow sibling failure only after completion event publication
        acknowledgement = process_payload["notification_acknowledgement"]  # Resolve Full worker acknowledgement state
        while int(acknowledgement.value) < int(task["global_id"]):  # Wait for coordinator send or deterministic failure drain
            time.sleep(0.01)  # Poll bounded shared state without another thread
        status_queue.put({"status": "done", "feature_set": feature_set, "worker_index": process_payload["worker_index"], "pid": os.getpid()})  # Report terminal Full worker after notification handling
        return  # End successful Full worker lifecycle
    process_payload["notification_barrier"].wait(5.0)  # Wait until Full completion event is already queued
    stacking.transition_feature_process_status(status_state, task, "started")  # Move GA failure task to running
    stacking.transition_feature_process_status(status_state, task, "failed")  # Count one failure without completion
    error = RuntimeError("Injected notification shutdown failure")  # Build deterministic sibling failure
    status_queue.put({"status": "error", "feature_set": feature_set, "worker_index": process_payload["worker_index"], "pid": os.getpid(), "global_id": task["global_id"], "error": str(error), "traceback": "Traceback: injected"})  # Surface production-compatible failure metadata
    raise error  # Preserve nonzero child exit behavior


class RecordingStatusQueue:  # Capture direct lifecycle order and status snapshots
    """Record local queue events alongside process-safe status snapshots."""

    def __init__(self, status_state, order):  # Store direct lifecycle evidence destinations
        self.status_state = status_state  # Retain shared status state for event-time snapshots
        self.order = order  # Retain mutable lifecycle ordering evidence
        self.records = []  # Accumulate queue payloads for terminal assertions

    def put(self, record):  # Record one production-compatible lifecycle payload
        self.records.append(record)  # Preserve exact queue payload order
        if record.get("notification_result") is not None:  # Capture status only at notification publication
            self.order.append("notification")  # Record notification boundary before task cleanup
            self.notification_status = stacking.read_feature_process_status(self.status_state)  # Read authoritative counters at send handoff


class FeatureProcessTelegramTests(unittest.TestCase):  # Verify persistent per-result Telegram restoration
    """Verify timing, identity, de-duplication, cache semantics, failure isolation, and cleanup."""

    def test_four_spawned_workers_send_one_persisted_completion_each(self):  # Verify Full, GA, PCA, and RFE coordinator delivery
        process_context = mp.get_context("spawn")  # Use the production process start method
        tasks = [make_notification_task(feature_name, index, len(FEATURE_NAMES)) for index, feature_name in enumerate(FEATURE_NAMES, start=1)]  # Build one dynamic combination per feature worker
        pending = {feature_name: [task for task in tasks if task["feature_set"] == feature_name] for feature_name in FEATURE_NAMES}  # Partition exact feature-local queues
        payload = make_notification_payload(process_context, FEATURE_NAMES)  # Build four-worker matrix-free coordinator metadata
        baseline_children = {child.pid for child in mp.active_children()}  # Snapshot unrelated live children
        with mock.patch.object(stacking, "send_telegram_message") as telegram_send, mock.patch.object(stacking, "log_feature_process_tree", return_value={"feature_workers": [], "auxiliary_processes": []}):  # Isolate external delivery and process-tree probing
            final_status = stacking.execute_feature_set_processes(pending, payload, tasks, {}, process_context=process_context, process_target=run_notification_process_probe)  # Run four real spawned persistent workers
        messages = [call.args[1] for call in telegram_send.call_args_list]  # Read coordinator-owned Telegram bodies
        self.assertEqual(telegram_send.call_count, 4)  # Send exactly one completion for every supported feature worker
        self.assertEqual({message.split(":", 1)[0] for message in messages}, {f"Finished combination {index}/4" for index in range(1, 5)})  # Use every original dynamic global ID and total regardless of completion order
        self.assertTrue(any("Full Features" in message for message in messages))  # Notify one Full result
        self.assertTrue(any("GA Features" in message for message in messages))  # Notify one GA result
        self.assertTrue(any("PCA - Default Hyperparameters" in message for message in messages))  # Notify one PCA result with established label
        self.assertTrue(any("RFE Features" in message for message in messages))  # Notify one RFE result
        self.assertEqual(final_status["global"], {"total": 4, "cached": 0, "pending": 0, "running": 0, "computed": 4, "failed": 0, "completed": 4})  # Preserve exact scientific completion status
        self.assertEqual(payload["notification_computations"].value, 4)  # Compute each combination exactly once
        self.assertEqual({child.pid for child in mp.active_children()} - baseline_children, set())  # Reap every feature worker after acknowledged delivery

    def test_duplicate_delivery_failure_does_not_recompute_or_fail_result(self):  # Verify at-most-once attempt and Telegram failure isolation
        process_context = mp.get_context("spawn")  # Use a real spawned worker and coordinator
        task = make_notification_task("Full Features", 1, 1)  # Build one Full completion identity
        payload = make_notification_payload(process_context, ["Full Features"], duplicate=True)  # Request one duplicate completion event
        with mock.patch.object(stacking, "send_telegram_message", side_effect=RuntimeError("Telegram unavailable")) as telegram_send, mock.patch.object(stacking, "log_feature_process_tree", return_value={"feature_workers": [], "auxiliary_processes": []}):  # Inject external delivery failure only
            final_status = stacking.execute_feature_set_processes({"Full Features": [task]}, payload, [task], {}, process_context=process_context, process_target=run_notification_process_probe)  # Complete scientific work despite Telegram failure
        self.assertEqual(telegram_send.call_count, 1)  # Consume only one delivery attempt despite duplicate event
        self.assertEqual(payload["notification_computations"].value, 1)  # Never recompute classifier work after delivery failure
        self.assertEqual(final_status["global"]["computed"], 1)  # Retain successfully computed status
        self.assertEqual(final_status["global"]["failed"], 0)  # Keep Telegram failure outside scientific failure counters

    def test_message_uses_dynamic_plan_and_persisted_values(self):  # Verify authoritative ID, total, F1, and elapsed duration
        task = make_notification_task("GA Features", 4, 240)  # Build a nonlocal global identity in a dynamic 240-task plan
        result_entry = make_notification_result(task, f1_score=0.769859173861937, elapsed_time_s=1563)  # Preserve authoritative persisted result values
        notified = set()  # Create one coordinator-local de-duplication set
        with mock.patch.object(stacking, "send_telegram_message") as telegram_send:  # Capture message before established host and script prefixing
            sent = stacking.send_feature_process_result_notification(task, result_entry, "computed", 240, notified)  # Send one restored completion body
        self.assertTrue(sent)  # Consume the one delivery attempt
        self.assertEqual(telegram_send.call_args.args[1], "Finished combination 4/240: GA Features - Default Hyperparameters - Original Test Data - Random Forest with F1: 0.769859173861937 in 26m 3s")  # Preserve old wording with dynamic plan and persisted values
        self.assertEqual(notified, {4})  # Reserve exact original global identity

    def test_cached_notifications_match_sequential_semantics(self):  # Verify cache hits remain CACHE notifications without fresh completion wording
        task = make_notification_task("RFE Features", 7, 32)  # Build one cached persistent result identity
        result_entry = make_notification_result(task, f1_score=0.66, elapsed_time_s=125)  # Build persisted cache metrics
        resume_key = stacking.build_resume_cache_key(task["execution_mode"], "Original Combined Files", task["experiment_mode"], task["augmentation_ratio"], ["BENIGN", "ATTACK"], task["feature_set"], task["classifier_name"], task["hyperparameters_enabled"])  # Build exact sequential recovery key
        progress_bar = mock.Mock()  # Capture sequential cache progress advancement
        with mock.patch.object(stacking, "send_telegram_message") as telegram_send:  # Capture both established sequential and persistent cache messages
            recovered, _ = stacking.recover_cached_individual_classifier_result({resume_key: result_entry}, task["execution_mode"], "Original Combined Files", task["experiment_mode"], task["augmentation_ratio"], ["BENIGN", "ATTACK"], task["feature_set"], task["classifier_name"], {}, 7, 32, progress_bar, task["hyperparameters_enabled"])  # Exercise current sequential cache path
            stacking.send_feature_process_result_notification(task, result_entry, "cached", 32, set())  # Exercise persistent coordinator cache path
        sequential_message, persistent_message = [call.args[1] for call in telegram_send.call_args_list]  # Read both cache notification bodies
        self.assertTrue(recovered)  # Preserve sequential cache recovery
        self.assertEqual(sequential_message, persistent_message)  # Reuse exact established cache wording and metrics
        self.assertTrue(persistent_message.startswith("[CACHE] Recovered saved result"))  # Preserve cache provenance
        self.assertNotIn("Finished combination", persistent_message)  # Never misreport cache recovery as fresh computation

    def test_notification_follows_status_and_persistence_before_cleanup(self):  # Verify required successful lifecycle order
        process_context = mp.get_context("spawn")  # Build production-compatible shared counters
        task = make_notification_task("Full Features", 1, 1)  # Build one Full original-data task
        task.update({"expected_n_features": 2, "expected_feature_names": ["a", "b"], "expected_n_samples_train": 8, "expected_n_samples_test": 2})  # Supply existing evaluator metadata
        status_state = stacking.create_feature_process_status(process_context, [task], {"Full Features": [task]})  # Initialize exact process-safe task status
        order = []  # Capture persistence, notification, and cleanup boundaries
        status_queue = RecordingStatusQueue(status_state, order)  # Capture notification-time status
        result_entry = make_notification_result(task)  # Build one simulated durably persisted result
        resource_state = {"original_resources": {"loaded": True}, "ratio_data": None, "active_ratio": None, "cache_dict": {}}  # Avoid unrelated data loading in focused lifecycle test
        combination_lock = mock.Mock()  # Observe combination reservation release
        def persisted_evaluation(*args, **kwargs):  # Return only after simulated cache and model persistence
            order.extend(["cache persistence", "model persistence"])  # Record both required durable writes
            return result_entry  # Return the small persisted result row
        def lifecycle_log(task_value, state_value, message):  # Record only combination cleanup boundaries
            if message.startswith("Combination cleanup"):  # Ignore unrelated lifecycle text
                order.append(message)  # Preserve cleanup order around notification handoff
        with mock.patch.object(stacking, "acquire_feature_process_combination_lock", return_value=combination_lock), mock.patch.object(stacking, "reload_feature_process_task_cache", return_value=(None, {})), mock.patch.object(stacking, "evaluate_feature_process_original_task", side_effect=persisted_evaluation), mock.patch.object(stacking, "log_feature_process_combination", side_effect=lifecycle_log):  # Isolate scientific work while retaining production task lifecycle
            stacking.process_feature_process_task(task, {"feature_set": "Full Features"}, {False: {"Random Forest": object()}}, resource_state, status_queue, status_state)  # Execute one complete persistent task path
        self.assertLess(order.index("cache persistence"), order.index("model persistence"))  # Persist cache before required model artifact in existing evaluator order
        self.assertLess(order.index("model persistence"), order.index("notification"))  # Publish notification only after required persistence
        self.assertLess(order.index("notification"), order.index("Combination cleanup started"))  # Keep cleanup after coordinator handoff
        self.assertEqual(status_queue.notification_status["global"]["running"], 0)  # Never notify while completed combination remains running
        self.assertEqual(status_queue.notification_status["global"]["computed"], 1)  # Count successfully computed result before notification
        self.assertEqual(status_queue.notification_status["global"]["completed"], 1)  # Count completed result before notification

    def test_fit_prediction_metrics_and_persistence_failures_publish_no_completion(self):  # Verify every scientific failure boundary suppresses fresh completion
        for failure_stage in ("fit", "prediction", "metrics", "cache persistence", "model persistence"):  # Exercise every required failure stage
            with self.subTest(stage=failure_stage):  # Isolate terminal evidence by failure stage
                process_context = mp.get_context("spawn")  # Build fresh production-compatible status state
                task = make_notification_task("GA Features", 1, 1)  # Build one pending GA task
                status_state = stacking.create_feature_process_status(process_context, [task], {"GA Features": [task]})  # Initialize exact counters
                status_queue = queue.Queue()  # Capture lifecycle queue records locally
                resource_state = {"original_resources": {"loaded": True}, "ratio_data": None, "active_ratio": None, "cache_dict": {}}  # Avoid unrelated data loading
                combination_lock = mock.Mock()  # Observe reservation cleanup after failure
                with mock.patch.object(stacking, "acquire_feature_process_combination_lock", return_value=combination_lock), mock.patch.object(stacking, "reload_feature_process_task_cache", return_value=(None, {})), mock.patch.object(stacking, "evaluate_feature_process_original_task", side_effect=RuntimeError(failure_stage)), mock.patch.object(stacking, "log_feature_process_combination"):  # Inject one scientific-stage failure before persisted result return
                    with self.assertRaisesRegex(RuntimeError, failure_stage):  # Preserve exact scientific exception
                        stacking.process_feature_process_task(task, {"feature_set": "GA Features"}, {False: {"Random Forest": object()}}, resource_state, status_queue, status_state)  # Exercise production failure transition
                records = []  # Accumulate all available lifecycle events
                while not status_queue.empty():  # Drain deterministic local queue records
                    records.append(status_queue.get_nowait())  # Preserve event payloads for notification assertions
                final_status = stacking.read_feature_process_status(status_state)["global"]  # Read terminal failure counters
                self.assertFalse(any(record.get("notification_result") is not None for record in records))  # Publish no successful completion notification
                self.assertEqual(final_status["computed"], 0)  # Count no computed result
                self.assertEqual(final_status["completed"], 0)  # Count no successful completion
                self.assertEqual(final_status["failed"], 1)  # Count exact scientific failure once

    def test_notification_payload_excludes_large_scientific_objects(self):  # Verify queue transport remains scalar-only
        task = make_notification_task("PCA Components", 1, 1)  # Build one PCA result identity
        result_entry = make_notification_result(task)  # Build persisted scalar fields
        result_entry.update({"matrix": np.ones((3, 3)), "estimator": object(), "predictions": np.array([0, 1]), "probabilities": np.ones((2, 2)), "features_list": ["PC1"]})  # Add forbidden scientific objects outside notification fields
        notification_result = stacking.build_feature_process_notification_result(result_entry)  # Reduce result through production event builder
        self.assertEqual(set(notification_result), {"feature_set", "hyperparameter_mode", "model_name", "execution_mode", "experiment_mode", "augmentation_ratio", "accuracy", "precision", "recall", "f1_score", "fpr", "fnr", "elapsed_time_s"})  # Carry only required scalar fields
        self.assertFalse(any(isinstance(value, np.ndarray) for value in notification_result.values()))  # Exclude matrices, predictions, and probabilities

    def test_failure_shutdown_drains_already_queued_completion(self):  # Verify pending completion survives coordinator failure shutdown
        process_context = mp.get_context("spawn")  # Use real spawned sibling workers
        feature_names = ["Full Features", "GA Features"]  # Pair one success with one failure
        tasks = [make_notification_task(feature_name, index, 2) for index, feature_name in enumerate(feature_names, start=1)]  # Build one task per sibling
        pending = {feature_name: [task for task in tasks if task["feature_set"] == feature_name] for feature_name in feature_names}  # Partition exact feature-local queues
        payload = make_notification_payload(process_context, feature_names)  # Build production-compatible coordinator payload
        payload["notification_barrier"] = process_context.Event()  # Coordinate failure only after completion publication
        baseline_children = {child.pid for child in mp.active_children()}  # Snapshot unrelated live children
        with mock.patch.object(stacking, "send_telegram_message") as telegram_send, mock.patch.object(stacking, "log_feature_process_tree", return_value={"feature_workers": [], "auxiliary_processes": []}):  # Capture coordinator send and isolate process inspection
            with self.assertRaisesRegex(RuntimeError, "Injected notification shutdown failure"):  # Require existing worker failure propagation
                stacking.execute_feature_set_processes(pending, payload, tasks, {}, process_context=process_context, process_target=run_notification_shutdown_probe)  # Exercise bounded failure drain and deterministic reaping
        self.assertEqual(telegram_send.call_count, 1)  # Deliver already-persisted Full completion exactly once
        self.assertIn("Finished combination 1/2", telegram_send.call_args.args[1])  # Preserve original global identity and dynamic total during failure
        self.assertEqual({child.pid for child in mp.active_children()} - baseline_children, set())  # Reap both success and failure workers


if __name__ == "__main__":  # Support direct focused test execution
    unittest.main()  # Run persistent Telegram notification tests
