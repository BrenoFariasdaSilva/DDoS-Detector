"""Strict Telegram runtime-control command parsing."""

import shlex
from dataclasses import dataclass
from typing import Any, Optional


GLOBAL_ID_FIELDS = {"execution", "combination"}
METADATA_FIELDS = {"execution", "classifier", "run", "feature_set", "hyperparameters", "data"}
ALL_FIELDS = GLOBAL_ID_FIELDS | METADATA_FIELDS


@dataclass(frozen=True)
class SkipCommand:
    execution_id: str
    combination_id: Optional[int] = None
    metadata: Optional[dict[str, Any]] = None


def parse_skip_command(text: str) -> SkipCommand:
    """Parse one strict skip command."""

    parts = shlex.split(str(text or ""))
    if not parts or parts[0] != "skip":
        raise ValueError("Invalid command. Use: skip execution=<id> combination=<global_id>")
    values = {}
    for part in parts[1:]:
        if "=" not in part:
            raise ValueError(f"Invalid command token: {part}")
        key, value = part.split("=", 1)
        if key not in ALL_FIELDS:
            raise ValueError(f"Unsupported command field: {key}")
        if key in values:
            raise ValueError(f"Duplicate command field: {key}")
        values[key] = value
    execution_id = str(values.get("execution", "")).strip().upper()
    if not execution_id:
        raise ValueError("Missing required field: execution")
    if set(values) == GLOBAL_ID_FIELDS:
        try:
            combination_id = int(values["combination"])
        except ValueError as error:
            raise ValueError("Combination must be an integer") from error
        if combination_id <= 0:
            raise ValueError("Combination must be a positive integer")
        return SkipCommand(execution_id=execution_id, combination_id=combination_id)
    if set(values) == METADATA_FIELDS:
        try:
            run_index = int(values["run"])
        except ValueError as error:
            raise ValueError("Run must be an integer") from error
        return SkipCommand(
            execution_id=execution_id,
            metadata={
                "classifier_model": values["classifier"],
                "experiment_run": run_index,
                "feature_set": values["feature_set"],
                "hyperparameter_mode": values["hyperparameters"],
                "data_source": values["data"],
            },
        )
    raise ValueError("Use either execution+combination or execution+classifier+run+feature_set+hyperparameters+data")


def format_skip_ack(status: str, execution_id: str, global_id: Optional[int] = None, detail: str = "") -> str:
    """Build one deterministic Telegram control acknowledgement."""

    combination = f" | Combination {global_id}" if global_id is not None else ""
    suffix = f" | {detail}" if detail else ""
    return f"[CONTROL] Execution {execution_id}{combination} | {status}{suffix}"
