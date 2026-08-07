"""Operating-system process-title utilities."""

from typing import Optional  # Type optional process-name overrides.


def set_runtime_process_name(process_name: Optional[str]) -> None:
    """
    Set the operating-system process title shown by tools such as htop.

    :param process_name: Requested process title or None to preserve the default title.
    :return: None.
    """

    if process_name is None:  # Preserve the existing script title when no override is supplied.
        return  # Leave the process title unchanged.
    normalized_name = str(process_name).strip()  # Normalize surrounding shell or configuration whitespace.
    if not normalized_name or "\0" in normalized_name:  # Reject names that cannot form a valid process title.
        raise ValueError("--process-name must be a non-empty string without null bytes")  # Surface an actionable CLI error.
    try:  # Load the existing optional process-title dependency only when requested.
        from setproctitle import setproctitle  # Import the platform process-title setter.
    except ImportError as error:  # Fail explicitly because the requested htop identity cannot be applied.
        raise RuntimeError("--process-name requires the setproctitle package") from error  # Preserve the missing dependency cause.
    setproctitle(normalized_name)  # Replace the script default with the requested identity.
