"""Runtime registry for active stacking combinations."""

from copy import deepcopy
from threading import RLock
from typing import Any, Optional


PENDING = "pending"
STARTING = "starting"
ACTIVE = "active"
COMPLETED = "completed"
FAILED = "failed"
SKIPPED = "skipped"
CANCELLED = "cancelled"
TERMINAL_STATES = {COMPLETED, FAILED, SKIPPED, CANCELLED}


class RuntimeCombinationRegistry:
    """Thread-safe parent-process registry keyed by execution ID and global ID."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._combinations: dict[tuple[str, int], dict[str, Any]] = {}
        self._active_by_worker: dict[tuple[str, int], tuple[str, int]] = {}

    def register_tasks(self, execution_id: str, tasks: list[dict[str, Any]]) -> None:
        """Register one authoritative task plan as pending/completed runtime state."""

        if not execution_id:
            return
        with self._lock:
            for task in tasks:
                global_id = task.get("global_id")
                if global_id is None:
                    continue
                key = (execution_id, int(global_id))
                self._combinations[key] = {
                    **build_runtime_combination_metadata(execution_id, task),
                    "worker_pid": None,
                    "lifecycle_state": PENDING,
                    "last_event": None,
                    "error": None,
                }

    def update_state(self, execution_id: str, global_id: Optional[int], state: str, worker_pid: Optional[int] = None, **updates: Any) -> None:
        """Update one combination state without exposing worker-owned mutable state."""

        if not execution_id or global_id is None:
            return
        key = (execution_id, int(global_id))
        with self._lock:
            current = self._combinations.setdefault(key, {"execution_id": execution_id, "global_combination_id": int(global_id)})
            current.update({name: value for name, value in updates.items() if value is not None})
            current["lifecycle_state"] = state
            if worker_pid is not None:
                worker_key = (execution_id, int(worker_pid))
                previous_key = self._active_by_worker.get(worker_key)
                if previous_key is not None and previous_key != key and previous_key in self._combinations:
                    self._combinations[previous_key]["worker_pid"] = None
                if state in TERMINAL_STATES:
                    if self._active_by_worker.get(worker_key) == key:
                        self._active_by_worker.pop(worker_key, None)
                    current["worker_pid"] = None
                else:
                    self._active_by_worker[worker_key] = key
                    current["worker_pid"] = int(worker_pid)
            elif state in TERMINAL_STATES:
                stale_workers = [worker_key for worker_key, active_key in self._active_by_worker.items() if active_key == key]
                for worker_key in stale_workers:
                    self._active_by_worker.pop(worker_key, None)
                current["worker_pid"] = None

    def get(self, execution_id: str, global_id: int) -> Optional[dict[str, Any]]:
        """Return a defensive copy for one registered combination."""

        with self._lock:
            value = self._combinations.get((execution_id, int(global_id)))
            return deepcopy(value) if value is not None else None

    def snapshot(self, execution_id: Optional[str] = None) -> dict[tuple[str, int], dict[str, Any]]:
        """Return a defensive copy of registered combinations."""

        with self._lock:
            items = self._combinations.items()
            if execution_id:
                items = ((key, value) for key, value in items if key[0] == execution_id)
            return {key: deepcopy(value) for key, value in items}


def build_runtime_combination_metadata(execution_id: str, task: dict[str, Any]) -> dict[str, Any]:
    """Build registry metadata from the existing feature-process task descriptor."""

    augmentation_ratio = task.get("augmentation_ratio")
    return {
        "execution_id": execution_id,
        "global_combination_id": task.get("global_id"),
        "global_combination_total": task.get("total_combinations"),
        "canonical_total": task.get("canonical_total"),
        "local_combination_id": task.get("feature_local_position"),
        "local_combination_total": task.get("feature_local_total"),
        "classifier_model": task.get("classifier_name"),
        "experiment_run": task.get("experiment_run"),
        "feature_set": task.get("feature_set"),
        "hyperparameter_mode": "Optimized Hyperparameters" if task.get("hyperparameters_enabled") else "Default Hyperparameters",
        "experiment_mode": task.get("experiment_mode"),
        "data_source": task.get("data_source_label"),
        "data_augmentation_mode": "original" if augmentation_ratio is None else "augmented",
        "augmentation_ratio": augmentation_ratio,
        "dataset": task.get("dataset") or task.get("file"),
        "execution_mode": task.get("execution_mode"),
    }


RUNTIME_COMBINATION_REGISTRY = RuntimeCombinationRegistry()
