if __name__ in {"__main__", "__mp_main__"}:
    try:
        from setproctitle import setproctitle
        setproctitle(f"DDoS-{__file__.rsplit('/', 1)[-1].rsplit('.', 1)[0]}")
    except ImportError:
        pass

import os  # Provide process identities for detached progress records.
import math  # Validate finite progress intervals without affecting training.
import sys  # Resolve active and original output streams.
import threading  # Run low-overhead heartbeat reporting beside blocking fits.
import time  # Measure elapsed training time with a monotonic clock.
from typing import Any, Callable, Optional  # Define progress context and callback type hints.
from xgboost.callback import TrainingCallback  # Use XGBoost's public boosting-round callback API.


DEFAULT_TRAINING_PROGRESS_INTERVAL_MINUTES = 15.0  # Define the single built-in recurring training-progress default.
DEFAULT_TRAINING_PROGRESS_INTERVAL_SECONDS = DEFAULT_TRAINING_PROGRESS_INTERVAL_MINUTES * 60.0  # Convert the built-in fallback once for the seconds-based reporter boundary.


def format_training_feature_set(feature_set: Optional[str]) -> str:  # Normalize progress feature-set labels
    """
    Format one feature-set identity for classifier progress output.

    :param feature_set: Feature-set name or artifact label.
    :return: Concise feature-set label for progress output.
    """

    feature_label = str(feature_set or "Unknown").split(" - ", 1)[0]  # Remove artifact hyperparameter suffixes from the display label.
    return "PCA" if feature_label == "PCA Components" else feature_label  # Use the concise PCA label while preserving every other identity.


def format_training_combination_fields(hyperparameters_enabled: Optional[bool], augmentation_ratio: Optional[float]) -> str:  # Format authoritative evaluation-combination fields
    """
    Format hyperparameter and data-augmentation fields for one evaluation combination.

    :param hyperparameters_enabled: Whether optimized hyperparameters are active, or None outside an evaluation combination.
    :param augmentation_ratio: Authoritative augmentation ratio, or None for original data.
    :return: Delimiter-prefixed fields, or an empty string outside an evaluation combination.
    """

    if hyperparameters_enabled is None:  # Omit combination fields when no evaluation-combination identity exists.
        return ""  # Preserve truthful AutoML and standalone progress records.
    hyperparameter_label = "Optimized Hyperparameters" if hyperparameters_enabled else "Default Hyperparameters"  # Resolve label from authoritative combination mode.
    augmentation_label = "Off" if augmentation_ratio is None else f"{int(float(augmentation_ratio) * 100)}%"  # Format authoritative original or ratio mode.
    return f" | Hyperparameters: {hyperparameter_label} | Data Augmentation: {augmentation_label}"  # Return fields in required stable order.


def format_training_combination_prefix(local_combination_index: Optional[int], local_combination_total: Optional[int]) -> str:  # Format feature-local combination prefix
    """
    Format one feature-local combination prefix for progress output.

    :param local_combination_index: Active feature-local combination index.
    :param local_combination_total: Active feature-local combination total.
    :return: Bracketed combination prefix or an empty string when unavailable.
    """

    if local_combination_index is None or local_combination_total is None:  # Omit prefix when either authoritative value is unavailable.
        return ""  # Preserve unchanged standalone and AutoML progress logs.
    return f"[{int(local_combination_index)}/{int(local_combination_total)}]"  # Return the stable local-combination prefix.


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

    def __init__(self, feature_set: Optional[str], classifier_name: str, duration_formatter: Callable[[float], str], output_stream: Optional[Any] = None, total_units: Optional[int] = None, unit_label: Optional[str] = None, heartbeat: bool = False, report_interval_seconds: float = DEFAULT_TRAINING_PROGRESS_INTERVAL_SECONDS, hyperparameters_enabled: Optional[bool] = None, augmentation_ratio: Optional[float] = None, eta_callback: Optional[Callable[[str, Optional[float]], None]] = None, resource_suffix_callback: Optional[Callable[[], str]] = None, estimated_finish_suffix_callback: Optional[Callable[[float], str]] = None, estimated_total_seconds: Optional[float] = None, local_combination_index: Optional[int] = None, local_combination_total: Optional[int] = None):  # Initialize one training progress scope
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
        :param report_interval_seconds: Configured recurring progress interval in seconds.
        :param hyperparameters_enabled: Whether optimized hyperparameters are active, or None outside an evaluation combination.
        :param augmentation_ratio: Authoritative augmentation ratio, or None for original data.
        :param eta_callback: Optional callback receiving the first emitted nonfinal ETA label.
        :param resource_suffix_callback: Optional callback returning already-collected resource text.
        :param estimated_total_seconds: Optional historical total duration for heartbeat-only ETA.
        :return: None.
        """

        try:  # Keep malformed progress configuration from affecting estimator training.
            resolved_interval = float(report_interval_seconds)  # Normalize the configured recurring progress interval.
        except (TypeError, ValueError):  # Fall back when the optional reporting value is invalid.
            resolved_interval = DEFAULT_TRAINING_PROGRESS_INTERVAL_SECONDS  # Use the stable low-frequency progress fallback.
        self.feature_set = format_training_feature_set(feature_set)  # Store the normalized feature-set label.
        self.classifier_name = str(classifier_name)  # Store the classifier identity as log-safe text.
        self.combination_prefix = format_training_combination_prefix(local_combination_index, local_combination_total)  # Store immutable feature-local combination prefix when available.
        self.combination_fields = format_training_combination_fields(hyperparameters_enabled, augmentation_ratio)  # Store immutable authoritative combination fields.
        self.duration_formatter = duration_formatter  # Store the caller's established duration formatter.
        self.output_stream = output_stream if output_stream is not None else sys.stdout  # Store the caller's active output stream.
        self.total_units = int(total_units) if total_units is not None else None  # Store the exact public unit total when available.
        self.unit_label = str(unit_label) if unit_label is not None else None  # Store the public unit label when available.
        self.heartbeat = bool(heartbeat)  # Store whether this scope emits active heartbeats.
        self.interval_seconds = resolved_interval if math.isfinite(resolved_interval) and resolved_interval > 0 else DEFAULT_TRAINING_PROGRESS_INTERVAL_SECONDS  # Use the configured positive finite interval or stable fallback.
        self.start_time: Optional[float] = None  # Initialize the monotonic training start timestamp.
        self.last_report_time: Optional[float] = None  # Initialize this task's independent recurring-report timestamp.
        self.latest_completed_units: Optional[int] = None  # Retain the newest public callback state even when it is rate-limited.
        self.final_unit_reported = False  # Prevent duplicate immediate 100% progress records.
        self.report_lock = threading.Lock()  # Serialize heartbeat and callback timing inside this classifier task only.
        self.stop_event = threading.Event()  # Create a scope-owned heartbeat stop event.
        self.thread: Optional[threading.Thread] = None  # Initialize the optional heartbeat thread reference.
        self.eta_callback = eta_callback  # Store optional ETA notification callback.
        self.eta_callback_reported = False  # Track one ETA callback emission per training scope.
        self.resource_suffix_callback = resource_suffix_callback  # Store optional resource suffix callback.
        self.estimated_finish_suffix_callback = estimated_finish_suffix_callback  # Store optional ETA-derived finish-time callback.
        try:
            resolved_estimate = float(estimated_total_seconds) if estimated_total_seconds is not None else None  # Normalize historical runtime estimates.
        except (TypeError, ValueError):
            resolved_estimate = None  # Ignore malformed estimates.
        self.estimated_total_seconds = resolved_estimate if resolved_estimate is not None and math.isfinite(resolved_estimate) and resolved_estimate > 0 else None  # Store only positive finite estimates.

    def resource_suffix(self) -> str:  # Resolve optional resource suffix text.
        """
        Resolve the current resource suffix without affecting training.

        :param self: Instance of the TrainingProgress class.
        :return: Delimiter-prefixed resource suffix or an empty string.
        """

        try:  # Keep resource text failures from affecting training output.
            return str(self.resource_suffix_callback()) if self.resource_suffix_callback is not None else ""  # Return caller-provided resource suffix.
        except Exception:  # Preserve the original training row when resource text is unavailable.
            return ""  # Return no suffix on callback failure.

    def estimated_finish_suffix(self, eta_seconds: Optional[float]) -> str:  # Resolve optional estimated finish suffix text.
        """
        Resolve the current estimated finish suffix without affecting training.

        :param self: Instance of the TrainingProgress class.
        :param eta_seconds: Remaining seconds for the active ETA.
        :return: Delimiter-prefixed estimated finish suffix or an empty string.
        """

        try:  # Keep finish-time text failures from affecting training output.
            return str(self.estimated_finish_suffix_callback(float(eta_seconds))) if self.estimated_finish_suffix_callback is not None and eta_seconds is not None and math.isfinite(float(eta_seconds)) and float(eta_seconds) >= 0 else ""  # Return caller-provided finish-time suffix.
        except Exception:  # Preserve the original training row when finish-time text is unavailable.
            return ""  # Return no suffix on callback failure.

    def __enter__(self):  # Start one training progress scope
        """
        Start timing and optional heartbeat reporting for one classifier fit.

        :param self: Instance of the TrainingProgress class.
        :return: Active TrainingProgress instance.
        """

        self.start_time = time.monotonic()  # Record a monotonic timestamp immediately before blocking training.
        self.last_report_time = self.start_time  # Schedule the first recurring record one full interval after training starts.
        self.latest_completed_units = None  # Reset retained callback state for this training scope.
        self.final_unit_reported = False  # Reset final-report state for this training scope.
        self.eta_callback_reported = False  # Reset ETA callback state for this training scope.
        self.stop_event.clear()  # Reset the scope-owned event before an optional heartbeat thread starts.
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
                    print(f"{self.combination_prefix}[TRAINING] Feature Set: {self.feature_set} | Classifier: {self.classifier_name}{self.combination_fields} | Status: Heartbeat shutdown pending | PID: {os.getpid()}{self.resource_suffix()}", file=self.output_stream)  # Write a durable contextual cleanup warning.
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
            if not self.report_heartbeat():  # Emit only when this task's shared recurring interval is due.
                continue  # Keep waiting when a recent unit callback already consumed this interval.

    def report_heartbeat(self) -> bool:  # Emit one active record when the task interval is due
        """
        Emit one rate-limited heartbeat for this classifier task.

        :param self: Instance of the TrainingProgress class.
        :return: True when a heartbeat was emitted, otherwise False.
        """

        try:  # Keep reporting failures isolated from estimator training.
            now = time.monotonic()  # Read the monotonic clock for this interval decision.
            with self.report_lock:  # Coordinate heartbeat timing with public unit callbacks for this task only.
                if self.start_time is None or self.last_report_time is None or self.final_unit_reported:  # Require an active unfinished scope.
                    return False  # Skip inactive or already-finalized progress scopes.
                if now - self.last_report_time < self.interval_seconds:  # Enforce the configured recurring-report interval.
                    return False  # Retain silence until this task's next interval boundary.
                elapsed_seconds = max(now - self.start_time, 0.0)  # Calculate elapsed time from the monotonic training start.
                elapsed_label = self.duration_formatter(elapsed_seconds)  # Format elapsed time through the caller's established formatter.
                completed = int(self.latest_completed_units) if self.latest_completed_units is not None else 0  # Read the latest genuine public unit count.
                total = int(self.total_units) if self.total_units is not None else 0  # Read the configured public unit total.
                if completed > 0 and total > completed:  # Prefer genuine public estimator units when available.
                    remaining_seconds = (elapsed_seconds / completed) * (total - completed)  # Estimate remaining time from completed units.
                    eta_label = self.duration_formatter(remaining_seconds)  # Format ETA from completed units.
                elif self.estimated_total_seconds is not None and elapsed_seconds < self.estimated_total_seconds:  # Fall back to comparable cached runtime for heartbeat-only estimators.
                    remaining_seconds = self.estimated_total_seconds - elapsed_seconds  # Report remaining time from historical total duration.
                    eta_label = self.duration_formatter(remaining_seconds)  # Format historical-runtime ETA.
                else:
                    remaining_seconds = None  # Keep unavailable when no factual estimate exists.
                    eta_label = "unavailable"  # Keep unavailable when no factual estimate exists.
                print(f"{self.combination_prefix}[TRAINING] Feature Set: {self.feature_set} | Classifier: {self.classifier_name}{self.combination_fields} | Status: Active | Elapsed: {elapsed_label} | ETA: {eta_label}{self.estimated_finish_suffix(remaining_seconds)} | PID: {os.getpid()}{self.resource_suffix()}", file=self.output_stream)  # Write contextual heartbeat with factual ETA when units exist.
                self.output_stream.flush()  # Flush every heartbeat immediately to detached logs.
                if self.eta_callback is not None and not self.eta_callback_reported and eta_label != "unavailable" and eta_label != "0s":  # Notify once when heartbeat-only ETA becomes available.
                    self.eta_callback_reported = True  # Reserve the one ETA callback before external notification code.
                    self.eta_callback(eta_label, remaining_seconds)  # Send the exact formatted ETA label and seconds to the caller.
                self.last_report_time = now  # Advance only this classifier task's recurring-report timer.
            return True  # Report successful heartbeat emission.
        except Exception:  # Stop only progress output if the stream becomes unavailable.
            return False  # Leave estimator training untouched after a reporting failure.

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
            now = time.monotonic()  # Read the monotonic clock once for this callback and interval decision.
            with self.report_lock:  # Coordinate callback timing with an optional heartbeat for this task only.
                self.latest_completed_units = completed  # Retain the most recent genuine round, iteration, stage, or trial state.
                is_final = completed == total  # Detect the configured final public unit.
                if is_final and self.final_unit_reported:  # Suppress duplicate 100% callbacks.
                    return  # Preserve one immediate final progress record only.
                if is_final:  # Leave final progress to existing completion records instead of another active row.
                    self.final_unit_reported = True  # Prevent duplicate final callbacks from printing later.
                    return  # Keep normal in-progress output limited to active ETA rows.
                if not is_final and self.last_report_time is not None and now - self.last_report_time < self.interval_seconds:  # Rate-limit recurring non-final callback records.
                    return  # Retain the latest state without emitting before the interval.
                elapsed_seconds = max(now - self.start_time, 0.0)  # Calculate elapsed time after the latest completed unit.
                remaining_seconds = (elapsed_seconds / completed) * (total - completed)  # Estimate remaining time only from completed real units.
                elapsed_label = self.duration_formatter(elapsed_seconds)  # Format elapsed time through the caller's established formatter.
                eta_label = self.duration_formatter(remaining_seconds)  # Format the unit-based ETA through the caller's established formatter.
                print(f"{self.combination_prefix}[TRAINING] Feature Set: {self.feature_set} | Classifier: {self.classifier_name}{self.combination_fields} | Status: Active | Elapsed: {elapsed_label} | ETA: {eta_label}{self.estimated_finish_suffix(remaining_seconds)} | PID: {os.getpid()}{self.resource_suffix()}", file=self.output_stream)  # Write one active ETA row for normal in-progress output.
                self.output_stream.flush()  # Flush every emitted genuine progress record immediately to detached logs.
                if self.eta_callback is not None and not self.eta_callback_reported and not is_final and eta_label != "0s":  # Notify only on the first meaningful emitted ETA.
                    self.eta_callback_reported = True  # Reserve the one ETA callback before external notification code.
                    self.eta_callback(eta_label, remaining_seconds)  # Send the exact formatted ETA label and seconds to the caller.
                self.last_report_time = now  # Advance only this classifier task's recurring-report timer.
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
