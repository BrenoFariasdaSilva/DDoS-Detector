"""
Small runtime execution identity helpers.
"""

import uuid  # For collision-resistant execution identifiers
from typing import Optional  # For optional type hints


EXECUTION_ID_LENGTH = 10  # 40 bits from UUID4, short enough to type and large enough for concurrent runs.


def create_execution_id(length: int = EXECUTION_ID_LENGTH) -> str:
    """
    Create a short uppercase execution identifier from UUID4 randomness.

    :param length: Number of hexadecimal characters to keep.
    :return: Short uppercase execution identifier.
    """

    safe_length = max(6, min(32, int(length)))  # Keep IDs readable without allowing an empty identifier.
    return uuid.uuid4().hex[:safe_length].upper()  # Return a compact UUID4-derived identifier.


def assign_execution_id(config: dict, execution_id: Optional[str] = None) -> str:
    """
    Assign one execution identifier to a runtime configuration.

    :param config: Runtime configuration dictionary.
    :param execution_id: Optional pre-created execution identifier.
    :return: Assigned execution identifier.
    """

    runtime_config = config.setdefault("runtime", {})  # Keep transient runtime identity outside user-facing experiment sections.
    assigned_execution_id = str(execution_id or create_execution_id()).strip().upper()  # Normalize display form.
    runtime_config["execution_id"] = assigned_execution_id  # Store one picklable scalar for child processes.
    return assigned_execution_id  # Return assigned value for logging and Telegram setup.


def ensure_execution_id(config: dict) -> str:
    """
    Return existing runtime execution identifier, creating one only when absent.

    :param config: Runtime configuration dictionary.
    :return: Stable execution identifier for this runtime config.
    """

    runtime_config = config.setdefault("runtime", {})  # Resolve transient runtime section.
    execution_id = str(runtime_config.get("execution_id", "")).strip().upper()  # Read normalized existing value.
    if execution_id:  # Preserve existing top-level execution identity.
        runtime_config["execution_id"] = execution_id  # Normalize case in-place.
        return execution_id  # Return stable value.
    return assign_execution_id(config)  # Create an identifier only for config paths that skipped top-level initialization.
