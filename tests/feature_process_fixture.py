"""Spawn-safe deterministic worker fixtures for feature-process scheduling tests."""

import os  # Report real child process identities.
import time  # Simulate one blocked feature queue deterministically.
from typing import Any  # Describe multiprocessing event queues.

from feature_processes import close_array_resource, open_array_resource  # Exercise production memmap reopening and ownership.
from Logger import Logger  # Exercise complete process-safe log record writes.


def run_fixture_worker(payload: dict, event_queue: Any) -> None:
    """
    Process a deterministic test queue inside one persistent child.

    :param payload: Metadata-only fixture worker payload.
    :param event_queue: Coordinator event queue.
    :return: None.
    """

    feature_set = payload["feature_set"]  # Read the assigned feature identity.
    worker_index = payload.get("worker_index", 1)  # Read the configured worker index.
    event_queue.put({"type": "startup", "feature_set": feature_set, "worker_index": worker_index, "pid": os.getpid(), "message": f"fixture startup {feature_set}"})  # Report one startup event.
    mapped = open_array_resource(payload["array"]) if payload.get("array") else None  # Open the optional production memmap resource.
    labels = open_array_resource(payload["labels"]) if payload.get("labels") else None  # Open optional exact label memmap metadata.
    process_logger = Logger(payload["log_path"], clean=False) if payload.get("log_path") else None  # Open one append descriptor per concurrent fixture worker.
    try:  # Close worker-owned mappings and logging after every outcome.
        for task in payload["tasks"]:  # Process multiple combinations in the same persistent process.
            if payload.get("fail") and task["local_id"] == 1:  # Inject one deterministic child failure.
                raise RuntimeError(f"fixture failure for {feature_set}")  # Surface the exact worker identity.
            if payload.get("block") and task["local_id"] == 1:  # Block only the first PCA-style task.
                time.sleep(0.5)  # Simulate a slow valid fit without blocking sibling processes.
            fitted = not bool(task.get("cached", False))  # Simulate cache recovery without fitting.
            if process_logger is not None:  # Write one complete concurrent lifecycle record.
                process_logger.write(f"{feature_set}|{task['local_id']}|{os.getpid()}")  # Persist one newline-delimited fixture record.
            result = {"feature_set": feature_set, "pid": os.getpid(), "local_id": task["local_id"], "global_id": task["global_id"], "fitted": fitted, "finished_at": time.monotonic(), "array_values": mapped.tolist() if mapped is not None else None, "label_values": labels.tolist() if labels is not None else None}  # Build one small deterministic result.
            event_queue.put({"type": "progress", "feature_set": feature_set, "worker_index": worker_index, "local_id": task["local_id"], "local_total": task["local_total"], "global_id": task["global_id"]})  # Report this completed task.
            event_queue.put({"type": "result", "feature_set": feature_set, "worker_index": worker_index, "global_id": task["global_id"], "result": result})  # Return only the small result payload.
        event_queue.put({"type": "done", "feature_set": feature_set, "worker_index": worker_index, "pid": os.getpid(), "message": f"fixture done {feature_set}"})  # Report independent queue completion.
    finally:  # Release only the worker-owned mapping handle.
        close_array_resource(mapped)  # Close the mapping without deleting coordinator-owned files.
        close_array_resource(labels)  # Close the label mapping without deleting coordinator-owned files.
        if process_logger is not None:  # Close this worker's append descriptor.
            process_logger.close()  # Release the worker-owned log file descriptor.
