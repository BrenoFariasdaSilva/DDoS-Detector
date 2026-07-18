import os  # Provide process identities for detached progress records.
import sys  # Resolve active and original output streams.
import threading  # Run low-overhead heartbeat reporting beside blocking fits.
import time  # Measure elapsed training time with a monotonic clock.
from typing import Any, Callable, Optional, cast  # Define progress context and callback type hints.
from xgboost.callback import TrainingCallback  # Use XGBoost's public boosting-round callback API.


TRAINING_HEARTBEAT_INTERVAL_SECONDS = 60.0  # Define the detached classifier training heartbeat fallback.


def format_training_feature_set(feature_set: Optional[str]) -> str:  # Normalize progress feature-set labels
    """
    Format one feature-set identity for classifier progress output.

    :param feature_set: Feature-set name or artifact label.
    :return: Concise feature-set label for progress output.
    """

    feature_label = str(feature_set or "Unknown").split(" - ", 1)[0]  # Remove artifact hyperparameter suffixes from the display label.
    return "PCA" if feature_label == "PCA Components" else feature_label  # Use the concise PCA label while preserving every other identity.


def interactive_terminal_attached(output_stream: Optional[Any] = None) -> bool:  # Resolve whether interactive progress rendering is safe
    """
    Return whether the selected output stream is attached to an interactive terminal.

    :param output_stream: Output stream using the active standard output when None.
    :return: True when interactive terminal rendering is available.
    """

    active_stream = output_stream if output_stream is not None else sys.stdout  # Resolve the caller's output stream without retaining global state.
    logger_tty = getattr(active_stream, "is_tty", None)  # Read the repository logger's captured terminal state when present.
    if isinstance(logger_tty, bool):  # Use the logger's terminal state without requiring an isatty method.
        return logger_tty  # Return the logger's captured interactive state.
    stream_isatty = getattr(active_stream, "isatty", None)  # Read the selected stream terminal probe when available.
    if callable(stream_isatty):  # Use the standard stream terminal probe when callable.
        try:  # Keep display detection from affecting model execution.
            return bool(stream_isatty())  # Return the selected stream's interactive state.
        except Exception:  # Fall back to the original stdout stream on probe failure.
            pass  # Preserve output behavior after a failed terminal probe.
    return bool(sys.__stdout__ is not None and sys.__stdout__.isatty())  # Fall back to the interpreter's original stdout terminal state.


class TrainingProgress:  # Report genuine public units or heartbeat-only activity
    """Report genuine training units or low-frequency active heartbeats."""

    def __init__(self, feature_set: Optional[str], classifier_name: str, duration_formatter: Callable[[float], str], output_stream: Optional[Any] = None, total_units: Optional[int] = None, unit_label: Optional[str] = None, heartbeat: bool = False, heartbeat_interval_seconds: float = TRAINING_HEARTBEAT_INTERVAL_SECONDS):  # Initialize one training progress scope
        """
        Initialize one classifier training progress scope.

        :param self: Instance of the TrainingProgress class.
        :param feature_set: Feature-set name or artifact label.
        :param classifier_name: Classifier identity shown in progress output.
        :param duration_formatter: Callable that formats elapsed seconds for logging.
        :param output_stream: Output stream receiving progress records.
        :param total_units: Exact public training-unit total when available.
        :param unit_label: Public training-unit label when available.
        :param heartbeat: Whether to emit low-frequency active heartbeats.
        :param heartbeat_interval_seconds: Configured heartbeat interval in seconds.
        :return: None.
        """

        try:  # Keep malformed progress configuration from affecting estimator training.
            resolved_interval = float(heartbeat_interval_seconds)  # Normalize the configured heartbeat interval.
        except (TypeError, ValueError):  # Fall back when the optional reporting value is invalid.
            resolved_interval = TRAINING_HEARTBEAT_INTERVAL_SECONDS  # Use the stable low-frequency heartbeat fallback.
        self.feature_set = format_training_feature_set(feature_set)  # Store the normalized feature-set label.
        self.classifier_name = str(classifier_name)  # Store the classifier identity as log-safe text.
        self.duration_formatter = duration_formatter  # Store the caller's established duration formatter.
        self.output_stream = output_stream if output_stream is not None else sys.stdout  # Store the caller's active output stream.
        self.total_units = int(total_units) if total_units is not None else None  # Store the exact public unit total when available.
        self.unit_label = str(unit_label) if unit_label is not None else None  # Store the public unit label when available.
        self.heartbeat = bool(heartbeat)  # Store whether this scope emits active heartbeats.
        self.interval_seconds = resolved_interval if resolved_interval > 0 else TRAINING_HEARTBEAT_INTERVAL_SECONDS  # Use the configured positive interval or stable fallback.
        self.start_time: Optional[float] = None  # Initialize the monotonic training start timestamp.
        self.stop_event = threading.Event()  # Create a scope-owned heartbeat stop event.
        self.thread: Optional[threading.Thread] = None  # Initialize the optional heartbeat thread reference.

    def __enter__(self):  # Start one training progress scope
        """
        Start timing and optional heartbeat reporting for one classifier fit.

        :param self: Instance of the TrainingProgress class.
        :return: Active TrainingProgress instance.
        """

        self.start_time = time.monotonic()  # Record a monotonic timestamp immediately before blocking training.
        if self.heartbeat:  # Start a heartbeat only when no reliable internal percentage is available.
            try:  # Keep heartbeat startup failures from affecting estimator training.
                thread_name = f"training-heartbeat-{os.getpid()}-{id(self)}"  # Build a process-specific thread identity for future multiprocessing compatibility.
                self.thread = threading.Thread(target=self.emit_heartbeats, name=thread_name, daemon=True)  # Create one low-overhead daemon heartbeat thread.
                self.thread.start()  # Start heartbeat waiting immediately before the blocking fit.
            except Exception:  # Continue with the original blocking fit if thread startup is unavailable.
                self.thread = None  # Clear incomplete heartbeat state before training proceeds.
        return self  # Return the active progress scope to callback adapters.

    def __exit__(self, exception_type, exception_value, exception_traceback):  # Stop one training progress scope
        """
        Stop heartbeat reporting without suppressing a training exception.

        :param self: Instance of the TrainingProgress class.
        :param exception_type: Exception type raised by the classifier fit, if any.
        :param exception_value: Exception instance raised by the classifier fit, if any.
        :param exception_traceback: Traceback raised by the classifier fit, if any.
        :return: False so the original training exception remains unchanged.
        """

        try:  # Keep cleanup reporting from masking the original fit result or exception.
            self.stop_event.set()  # Signal the heartbeat wait to stop in every exit path.
            if self.thread is not None:  # Join only when this scope started a heartbeat thread.
                self.thread.join(timeout=1.0)  # Wait briefly for event-driven heartbeat shutdown.
                if self.thread.is_alive():  # Report a cleanup anomaly without masking the classifier result.
                    print(f"[TRAINING] Feature Set: {self.feature_set} | Classifier: {self.classifier_name} | Status: Heartbeat shutdown pending | PID: {os.getpid()}", file=self.output_stream)  # Write a durable cleanup warning.
                    self.output_stream.flush()  # Flush the cleanup warning immediately.
        except Exception:  # Ignore cleanup output failures after signaling thread shutdown.
            pass  # Preserve the original fit outcome unchanged.
        return False  # Preserve the original fit return or exception semantics.

    def emit_heartbeats(self) -> None:  # Emit active records until training exits
        """
        Emit low-frequency active records until the scope stop event is set.

        :param self: Instance of the TrainingProgress class.
        :return: None.
        """

        while not self.stop_event.wait(self.interval_seconds):  # Wait efficiently between heartbeat records.
            try:  # Keep reporting failures isolated from estimator training.
                elapsed_seconds = max(time.monotonic() - cast(float, self.start_time), 0.0)  # Calculate elapsed time from the monotonic training start.
                elapsed_label = self.duration_formatter(elapsed_seconds)  # Format elapsed time through the caller's established formatter.
                print(f"[TRAINING] Feature Set: {self.feature_set} | Classifier: {self.classifier_name} | Status: Active | Elapsed: {elapsed_label} | ETA: unavailable | PID: {os.getpid()}", file=self.output_stream)  # Write one newline-delimited heartbeat without a fabricated percentage.
                self.output_stream.flush()  # Flush every heartbeat immediately to detached logs.
            except Exception:  # Stop only progress output if the stream becomes unavailable.
                return  # Leave estimator training untouched after a reporting failure.

    def report_unit(self, completed_units: int) -> None:  # Report one completed public training unit
        """
        Report progress and ETA from completed public estimator units.

        :param self: Instance of the TrainingProgress class.
        :param completed_units: Number of real public training units completed.
        :return: None.
        """

        try:  # Keep progress output isolated from estimator training semantics.
            completed = int(completed_units)  # Normalize the callback's completed-unit count.
            total = int(self.total_units) if self.total_units is not None else 0  # Resolve the exact public unit total.
            if completed < 1 or total < 1 or completed > total or self.unit_label is None or self.start_time is None:  # Reject incomplete or inconsistent callback metadata.
                return  # Avoid emitting an unreliable percentage or ETA.
            elapsed_seconds = max(time.monotonic() - self.start_time, 0.0)  # Calculate elapsed time after the completed unit.
            remaining_seconds = (elapsed_seconds / completed) * (total - completed)  # Estimate remaining time only from completed real units.
            progress_percent = (completed / total) * 100.0  # Calculate genuine percentage from the public unit denominator.
            elapsed_label = self.duration_formatter(elapsed_seconds)  # Format elapsed time through the caller's established formatter.
            eta_label = self.duration_formatter(remaining_seconds)  # Format the unit-based ETA through the caller's established formatter.
            print(f"[TRAINING] Feature Set: {self.feature_set} | Classifier: {self.classifier_name} | {self.unit_label}: {completed}/{total} | Progress: {progress_percent:.2f}% | Elapsed: {elapsed_label} | ETA: {eta_label} | PID: {os.getpid()}", file=self.output_stream)  # Write one genuine unit-completion record.
            self.output_stream.flush()  # Flush every genuine progress record immediately to detached logs.
        except Exception:  # Ignore reporting failures so callbacks cannot alter fitted results.
            return  # Preserve estimator training after a reporting failure.


class XGBoostProgressCallback(TrainingCallback):  # Adapt XGBoost public rounds to the shared reporter
    """Adapt XGBoost's public callback API to TrainingProgress."""

    def __init__(self, progress: TrainingProgress):  # Initialize the public XGBoost callback adapter
        """
        Initialize the XGBoost progress callback adapter.

        :param self: Instance of the XGBoostProgressCallback class.
        :param progress: Active TrainingProgress instance.
        :return: None.
        """

        self.progress = progress  # Retain only the lightweight progress scope.

    def after_iteration(self, model: Any, epoch: int, evals_log: dict) -> bool:  # Report one completed XGBoost round
        """
        Report one completed XGBoost boosting round.

        :param self: Instance of the XGBoostProgressCallback class.
        :param model: Active public XGBoost model handle.
        :param epoch: Zero-based completed boosting-round index.
        :param evals_log: Public XGBoost evaluation history mapping.
        :return: False so training continuation behavior remains unchanged.
        """

        self.progress.report_unit(epoch + 1)  # Convert the zero-based callback index into completed rounds.
        return False  # Never request early stopping from progress reporting.
