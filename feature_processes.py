"""Process and memmap primitives for persistent feature-set evaluation workers."""

import hashlib  # Build host-scoped temporary directory identities.
import importlib  # Resolve the repository worker entry point inside spawned children.
import ctypes  # Configure Linux parent-death process behavior without dependencies.
import multiprocessing as mp  # Create real operating-system feature-set workers.
import os  # Read process identities and manage resource paths.
from pathlib import Path  # Build explicit disk-backed resource paths.
import queue  # Handle bounded coordinator event waits.
import shutil  # Remove coordinator-owned resource directories.
import signal  # Terminate orphaned Linux workers when their coordinator dies.
import sys  # Detect the Linux runtime before using prctl.
import tempfile  # Create unique coordinator-owned resource directories.
import time  # Bound deterministic coordinator shutdown polling.
import traceback  # Surface complete child-process failures.
from typing import Any, Dict, List, Optional, Tuple  # Describe process payloads and plan partitions.

import numpy as np  # Store and reopen exact NumPy arrays as disk-backed mappings.
import psutil  # Prove stale resource-directory ownership before deletion.


FEATURE_SET_KEYS = {"full": "Full Features", "explicit": "Explicit Features", "ga": "GA Features", "pca": "PCA Components", "rfe": "RFE Features"}  # Map CLI configuration keys to runtime feature identities.
FEATURE_SET_LABELS = {"Full Features": "FULL", "Explicit Features": "EXPLICIT", "GA Features": "GA", "PCA Components": "PCA", "RFE Features": "RFE"}  # Define concise progress labels.


def validate_feature_set_workers(value: Any, source: str = "evaluation.feature_set_workers") -> Dict[str, int]:
    """
    Validate persistent feature-set worker configuration.

    :param value: Mapping or comma-separated key=value text.
    :param source: User-facing setting name used in validation errors.
    :return: Normalized feature-set worker mapping.
    """

    if value in (None, "", {}):  # Treat an absent configuration as sequential execution.
        return {}  # Return the explicit multiprocessing-disabled state.
    if isinstance(value, str):  # Parse the CLI key=value representation.
        entries = [entry.strip() for entry in value.split(",") if entry.strip()]  # Split non-empty comma-separated assignments.
        parsed: Dict[str, int] = {}  # Accumulate normalized CLI assignments.
        for entry in entries:  # Validate every assignment independently.
            if entry.count("=") != 1:  # Require one unambiguous key/value delimiter.
                raise ValueError(f"{source} must use comma-separated key=1 entries")  # Reject malformed CLI syntax.
            key, raw_count = [part.strip().lower() for part in entry.split("=", 1)]  # Normalize one assignment.
            if key in parsed:  # Reject duplicate feature identities.
                raise ValueError(f"{source} contains duplicate feature set '{key}'")  # Prevent silent last-value wins.
            try:  # Convert the worker count without accepting floats.
                parsed[key] = int(raw_count)  # Store the normalized integer count.
            except ValueError as exc:  # Report the exact invalid assignment.
                raise ValueError(f"{source} worker count for '{key}' must be 1") from exc  # Reject non-integer counts.
        value = parsed  # Continue through shared mapping validation.
    if not isinstance(value, dict):  # Require YAML mappings or CLI assignment text only.
        raise ValueError(f"{source} must be a mapping or comma-separated key=1 text")  # Reject ambiguous configuration types.
    normalized: Dict[str, int] = {}  # Accumulate validated worker assignments.
    for raw_key, raw_count in value.items():  # Validate every configured feature set.
        key = str(raw_key).strip().lower()  # Normalize the public feature-set key.
        if key not in FEATURE_SET_KEYS:  # Reject unsupported feature identities.
            raise ValueError(f"{source} contains unsupported feature set '{raw_key}'")  # List the invalid public key.
        if isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count != 1:  # Support only the proven one-persistent-worker design.
            raise ValueError(f"{source}.{key} must be exactly 1; multiple workers per feature set are not supported")  # Reject misleading shared-queue settings.
        normalized[key] = raw_count  # Preserve the validated insertion order.
    return normalized  # Return the normalized configuration.


def partition_evaluation_plan(evaluation_plan: List[Tuple[str, bool, Optional[float], str]]) -> Dict[str, List[dict]]:
    """
    Partition one authoritative evaluation plan by feature-set identity.

    :param evaluation_plan: Ordered runtime combination tuples.
    :return: Feature-set queues preserving original objects, global IDs, and local order.
    """

    partitions: Dict[str, List[dict]] = {}  # Accumulate queues in first-occurrence feature order.
    total_combinations = len(evaluation_plan)  # Preserve the authoritative global denominator.
    for global_id, combination in enumerate(evaluation_plan, start=1):  # Assign immutable one-based global identities.
        feature_set = combination[0]  # Read the actual feature-set identity from the plan object.
        feature_queue = partitions.setdefault(feature_set, [])  # Create one persistent queue per feature set.
        feature_queue.append({"global_id": global_id, "global_total": total_combinations, "local_id": len(feature_queue) + 1, "combination": combination})  # Preserve the original tuple and relative order.
    for feature_queue in partitions.values():  # Add the final local denominator after partitioning is complete.
        for task in feature_queue:  # Update every small task metadata object.
            task["local_total"] = len(feature_queue)  # Preserve the exact feature-local queue size.
    return partitions  # Return every dynamic feature partition.


def cleanup_stale_resource_directories(base_directory: Path) -> None:
    """
    Remove feature-process resources whose exact coordinator process no longer exists.

    :param base_directory: Directory containing owned feature-process resource folders.
    :return: None.
    """

    host_identity = hashlib.sha256(os.uname().nodename.encode("utf-8")).hexdigest()[:12]  # Identify resources created on this host.
    for candidate in base_directory.glob(f"FeatureProcesses_{host_identity}_*"):  # Inspect only directories using the owned prefix.
        parts = candidate.name.split("_", 4)  # Parse the host, PID, and process-birth identity.
        if len(parts) < 5 or candidate.is_symlink() or not candidate.is_dir():  # Preserve malformed or unsafe candidates.
            continue  # Skip resources without exact ownership proof.
        try:  # Prove whether the creating coordinator remains active.
            owner_pid = int(parts[2])  # Parse the coordinator PID.
            owner_create_time = int(parts[3])  # Parse the microsecond process birth time.
            active = int(psutil.Process(owner_pid).create_time() * 1000000) == owner_create_time  # Reject PID reuse as active ownership.
        except psutil.NoSuchProcess:  # Treat a missing exact owner as stale.
            active = False  # Permit cleanup of abandoned resources.
        except (ValueError, psutil.AccessDenied, psutil.ZombieProcess):  # Preserve candidates whose ownership cannot be proven safely.
            active = True  # Disable cleanup under ambiguous ownership.
        if not active:  # Remove only resources from a proven-dead coordinator.
            shutil.rmtree(candidate)  # Delete the complete abandoned resource directory.


def create_resource_directory(base_directory: str) -> str:
    """
    Create one coordinator-owned directory for disk-backed worker resources.

    :param base_directory: Existing or creatable array-cache base directory.
    :return: Absolute coordinator-owned resource directory path.
    """

    base_path = Path(base_directory).resolve()  # Resolve the configured resource root.
    base_path.mkdir(parents=True, exist_ok=True)  # Create the resource root when absent.
    cleanup_stale_resource_directories(base_path)  # Recover resources abandoned by killed coordinators.
    host_identity = hashlib.sha256(os.uname().nodename.encode("utf-8")).hexdigest()[:12]  # Build the local host ownership identity.
    create_time = int(psutil.Process(os.getpid()).create_time() * 1000000)  # Pair the PID with its process birth time.
    prefix = f"FeatureProcesses_{host_identity}_{os.getpid()}_{create_time}_"  # Build a collision-resistant ownership prefix.
    return str(Path(tempfile.mkdtemp(prefix=prefix, dir=str(base_path))).resolve())  # Create and return the unique owned directory.


def save_array_resource(array: Any, resource_directory: str, name: str) -> dict:
    """
    Persist one array as an exact coordinator-owned NumPy memmap resource.

    :param array: NumPy-compatible source array.
    :param resource_directory: Coordinator-owned resource directory.
    :param name: Filename-safe logical resource name.
    :return: Small metadata dictionary used to reopen the array.
    """

    source = np.asarray(array)  # Normalize the array without changing its values or dtype.
    resource_path = Path(resource_directory) / f"{name}.npy"  # Build the explicit backing-file path.
    mapped = np.lib.format.open_memmap(resource_path, mode="w+", dtype=source.dtype, shape=source.shape)  # Allocate a standard NumPy disk-backed array.
    mapped[...] = source  # Copy exact values without dtype conversion.
    mapped.flush()  # Publish every backing byte before child startup.
    del mapped  # Close the coordinator's writable mapping before workers reopen it.
    return {"path": str(resource_path.resolve()), "shape": tuple(int(value) for value in source.shape), "dtype": str(source.dtype)}  # Return small reopen metadata only.


def open_array_resource(metadata: dict) -> np.memmap:
    """
    Reopen one coordinator-owned NumPy resource read-only.

    :param metadata: Metadata returned by save_array_resource.
    :return: Read-only NumPy memmap with validated shape and dtype.
    """

    mapped = np.load(metadata["path"], mmap_mode="r", allow_pickle=False)  # Reopen the standard NumPy resource without object deserialization.
    if tuple(mapped.shape) != tuple(metadata["shape"]) or str(mapped.dtype) != str(metadata["dtype"]):  # Validate the complete reopen contract.
        raise ValueError(f"Memmap metadata mismatch for {metadata['path']}")  # Reject replaced or incompatible resources.
    return mapped  # Return the validated read-only mapping.


def close_array_resource(array: Any) -> None:
    """
    Close one worker-owned read-only memmap handle.

    :param array: NumPy memmap or ordinary array reference.
    :return: None.
    """

    mmap_object = getattr(array, "_mmap", None)  # Resolve the NumPy-owned operating-system mapping.
    if mmap_object is not None and not mmap_object.closed:  # Close only an active mapping.
        mmap_object.close()  # Release the worker-owned mapping without deleting coordinator files.


def remove_resource_directory(resource_directory: Optional[str]) -> None:
    """
    Remove one coordinator-owned resource directory after every child has exited.

    :param resource_directory: Exact directory returned by create_resource_directory.
    :return: None.
    """

    if resource_directory and Path(resource_directory).is_dir():  # Delete only the explicit coordinator-owned target.
        shutil.rmtree(resource_directory)  # Remove memmaps after worker joins release every handle.


def configure_parent_death_signal(expected_parent_pid: int) -> None:
    """
    Configure Linux to terminate a worker when its exact coordinator dies.

    :param expected_parent_pid: Coordinator PID recorded before process startup.
    :return: None.
    """

    if not sys.platform.startswith("linux"):  # Keep non-Linux development and test environments portable.
        return  # Skip the Linux-only kernel process contract.
    libc = ctypes.CDLL(None, use_errno=True)  # Load the current process C runtime.
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # Request PR_SET_PDEATHSIG with SIGTERM.
        error_number = ctypes.get_errno()  # Capture the kernel failure before another C call.
        raise OSError(error_number, os.strerror(error_number))  # Refuse an unprotected persistent Linux worker.
    if os.getppid() != expected_parent_pid:  # Close the race where the coordinator died before prctl completed.
        os.kill(os.getpid(), signal.SIGTERM)  # Terminate this already-orphaned worker immediately.


def dispatch_feature_set_worker(worker_module: str, worker_function: str, payload: dict, event_queue: Any) -> None:
    """
    Import and execute one repository feature-set worker inside a spawned process.

    :param worker_module: Importable module containing the worker function.
    :param worker_function: Public worker function name.
    :param payload: Small worker metadata payload without full matrices.
    :param event_queue: Coordinator event queue receiving small status payloads.
    :return: None.
    """

    try:  # Convert every child exception into a coordinator-visible event.
        configure_parent_death_signal(payload["coordinator_pid"])  # Prevent Linux workers from surviving forced coordinator death.
        module = importlib.import_module(worker_module)  # Import the worker implementation after spawn initialization.
        function = getattr(module, worker_function)  # Resolve the public worker entry point.
        function(payload, event_queue)  # Run the persistent feature-set queue in this process.
    except BaseException as exc:  # Surface ordinary failures and interrupts from the child.
        event_queue.put({"type": "error", "feature_set": payload.get("feature_set"), "worker_index": payload.get("worker_index", 1), "pid": os.getpid(), "error": str(exc), "traceback": traceback.format_exc()})  # Send complete child failure context.
        raise  # Preserve a nonzero child exit status.


def run_feature_set_processes(worker_payloads: List[dict], total_combinations: int, worker_module: str, worker_function: str) -> List[dict]:
    """
    Run persistent spawned feature-set processes and aggregate their small results.

    :param worker_payloads: One metadata-only payload per persistent worker.
    :param total_combinations: Authoritative global combination count.
    :param worker_module: Importable module containing the worker function.
    :param worker_function: Public worker function name.
    :return: Result entries ordered by original global combination ID.
    """

    context = mp.get_context("spawn")  # Avoid inheriting pandas, NumPy, logger, and active-thread ownership.
    event_queue = context.Queue()  # Carry only startup, progress, result, completion, and error metadata.
    processes = []  # Track every child for deterministic joining.
    terminal_workers = set()  # Track workers that emitted completion or failure.
    failures = []  # Accumulate child failures while independent siblings finish.
    results = []  # Accumulate small result entries with their original global IDs.
    completed_count = 0  # Count out-of-order completed or cache-recovered combinations.
    try:  # Ensure every startup path reaches deterministic shutdown.
        for payload in worker_payloads:  # Start exactly one process for every validated feature queue.
            payload["coordinator_pid"] = os.getpid()  # Record exact parent ownership before spawning this worker.
            process = context.Process(target=dispatch_feature_set_worker, args=(worker_module, worker_function, payload, event_queue), name=f"stacking-{FEATURE_SET_LABELS.get(payload['feature_set'], payload['feature_set']).lower()}-1")  # Create one persistent named child.
            process.start()  # Start the real operating-system process.
            processes.append(process)  # Retain the process for monitoring and joining.
        while len(terminal_workers) < len(processes):  # Continue until every child has reported or exited.
            try:  # Wait briefly so dead children without events can be detected.
                event = event_queue.get(timeout=0.2)  # Receive one small worker event.
            except queue.Empty:  # Inspect process exits during an idle event interval.
                for process, payload in zip(processes, worker_payloads):  # Match every process with its feature identity.
                    worker_key = (payload["feature_set"], payload.get("worker_index", 1))  # Build the terminal identity.
                    if process.exitcode is not None and worker_key not in terminal_workers:  # Detect an exit that produced no terminal event.
                        terminal_workers.add(worker_key)  # Prevent duplicate exit reporting.
                        if process.exitcode != 0:  # Treat unexplained nonzero exits as failures.
                            failures.append({"feature_set": payload["feature_set"], "pid": process.pid, "error": f"worker exited with code {process.exitcode}"})  # Preserve the unexplained exit.
                continue  # Resume event monitoring for remaining workers.
            event_type = event.get("type")  # Resolve the event category.
            worker_key = (event.get("feature_set"), event.get("worker_index", 1))  # Resolve the emitting worker identity.
            if event_type == "startup":  # Report process and matrix ownership at startup.
                pass  # The child already wrote the complete startup record through the shared process-safe logger.
            elif event_type == "progress":  # Advance global completion after persistence or cache recovery.
                completed_count += 1  # Count this out-of-order completion exactly once.
                label = FEATURE_SET_LABELS.get(event["feature_set"], event["feature_set"])  # Resolve the concise feature label.
                print(f"[{label} {event['local_id']}/{event['local_total']} | Global ID {event['global_id']}/{total_combinations} | Completed {completed_count}/{total_combinations}]", flush=True)  # Emit independent local and global progress.
            elif event_type == "result":  # Retain one small completed result for final export.
                results.append({"global_id": event["global_id"], "result": event["result"]})  # Preserve original ordering metadata beside the result.
            elif event_type == "done":  # Mark one independently completed feature queue.
                terminal_workers.add(worker_key)  # Record normal terminal status.
                pass  # The child already wrote the durable completion record through the shared process-safe logger.
            elif event_type == "error":  # Preserve a failure without cancelling independent siblings.
                terminal_workers.add(worker_key)  # Record the failed worker terminal status.
                failures.append(event)  # Retain complete child exception context.
                print(f"[FEATURE WORKER ERROR] Feature Set: {event.get('feature_set')} | Worker: {event.get('worker_index')} | PID: {event.get('pid')} | Error: {event.get('error')}", flush=True)  # Emit a concise durable failure record.
        for process in processes:  # Reap every normally completed or failed child.
            process.join()  # Wait without timeout after terminal monitoring proves every child exited or reported.
        if failures:  # Surface child exceptions only after independent siblings finish.
            failure_text = "\n".join(f"{failure.get('feature_set')}: {failure.get('error')}\n{failure.get('traceback', '')}" for failure in failures)  # Format every preserved failure.
            raise RuntimeError(f"Feature-set worker failure(s):\n{failure_text}")  # Fail the coordinator with complete child context.
        result_global_ids = [item["global_id"] for item in results]  # Collect delivered authoritative identities for exact-once validation.
        expected_global_ids = set(range(1, total_combinations + 1))  # Build the complete expected global identity set.
        if completed_count != total_combinations or len(results) != total_combinations or set(result_global_ids) != expected_global_ids or len(set(result_global_ids)) != len(result_global_ids):  # Reject lost or duplicated result delivery.
            raise RuntimeError(f"Feature-set workers completed {completed_count}/{total_combinations} combinations and returned {len(results)} results")  # Surface incomplete scheduling.
        return [item["result"] for item in sorted(results, key=lambda item: item["global_id"])]  # Restore authoritative global result order.
    except BaseException:  # Shut down every live child on coordinator interruption or internal failure.
        for process in processes:  # Inspect all started children.
            if process.is_alive():  # Terminate only processes that have not exited.
                process.terminate()  # Request deterministic child termination.
        deadline = time.monotonic() + 10.0  # Bound graceful termination joins.
        for process in processes:  # Reap every child after termination.
            process.join(timeout=max(0.0, deadline - time.monotonic()))  # Wait within the shared shutdown deadline.
            if process.is_alive():  # Escalate only a child that ignored termination.
                process.kill()  # Force the remaining child to exit.
                process.join()  # Reap the killed child to prevent zombies.
        raise  # Preserve the coordinator failure or interrupt.
    finally:  # Release coordinator queue resources after all joins.
        event_queue.close()  # Close the coordinator queue endpoint.
        event_queue.join_thread()  # Reap the queue feeder thread.
