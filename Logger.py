r"""
================================================================================
Logger Utility Module
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-12-11
Description :
    Dual-channel logger that mirrors console output to both the terminal
    (preserving ANSI color sequences when the terminal is a TTY) and a
    sanitized log file (ANSI sequences removed). Designed for use in
    interactive sessions, background jobs, CI pipelines and Makefile runs.

    Behavior:
        - When attached to `sys.stdout`/`sys.stderr` the logger writes colored
            output to the controlling terminal (when available) and a color-free
            record to the specified log file.
        - Optional IANA-timezone timestamps are applied once at the shared
            terminal/file emission boundary.
        - ANSI escape sequences are removed from the file output using a
            conservative regex; lines are flushed immediately to keep logs live.
        - Provides minimal API: `write()`, `flush()` and `close()` so it can be
            used as a drop-in replacement for `sys.stdout`.

Usage:
    from Logger import Logger
    logger = Logger("./Logs/myrun.log", clean=True)
    sys.stdout = logger # optional: redirect all prints to logger

Notes & TODOs:
    - Consider adding log rotation and JSON output format.
    - The ANSI regex is intentionally simple; adjust if you need broader support.

Dependencies:
    - Python >= 3.9 (no external runtime dependencies required)

Assumptions:
    - The log file will contain cleaned, human-readable text (no ANSI codes).
    - The logger is safe for short-lived scripts and long-running processes.
"""

import fcntl  # Provide process-safe serialization for shared detached log writes
import os  # For interacting with the filesystem
import re  # For stripping ANSI escape sequences
import sys  # For replacing stdout/stderr
from datetime import datetime  # Obtain timezone-aware wall-clock timestamps at record emission
from zoneinfo import ZoneInfo  # Resolve the standard-library São Paulo timezone

# Regex Constants:
ANSI_ESCAPE_REGEX = re.compile(r"\x1B\[[0-9;]*[a-zA-Z]")  # Pattern to remove ANSI colors
LEADING_ANSI_ESCAPE_REGEX = re.compile(r"^(?:\x1B\[[0-9;]*[a-zA-Z])+")  # Preserve leading terminal controls before the visible timestamp
LOG_TIMESTAMP_PREFIX_REGEX = re.compile(r"^\d{2}/\d{2}/\d{4} - \d{2}h\d{2}m\d{2}s: ")  # Recognize the exact durable timestamp prefix
SAO_PAULO_TIMEZONE_NAME = "America/Sao_Paulo"  # Define the explicit Brazilian runtime log timezone

# Classes Definitions:


class Logger:
    """
    Simple logger class that prints colored messages to the terminal and
    writes a cleaned (ANSI-stripped) version to a log file.

    Usage:
       logger = Logger("./Logs/output.log", clean=True)
       logger.info("\x1b[92mHello world\x1b[0m")

    :param logfile_path: Path to the log file.
    :param clean: If True, truncate the log file on init; otherwise append.
    """

    def __init__(self, logfile_path, clean=False, timestamp_timezone=None, timestamp_now=None):  # Initialize optional per-record timezone formatting
        """
        Initialize the Logger.

        :param self: Instance of the Logger class.
        :param logfile_path: Path to the log file.
        :param clean: If True, truncate the log file on init; otherwise append.
        :param timestamp_timezone: Optional IANA timezone applied to every non-empty physical line.
        :param timestamp_now: Optional timezone-aware clock callable used for deterministic testing.
        """

        self.logfile_path = logfile_path  # Store log file path

        parent = os.path.dirname(logfile_path)  # Ensure log directory exists
        if parent and not os.path.exists(parent):  # Create parent directories if needed
            os.makedirs(parent, exist_ok=True)  # Safe creation

        self.logfile = open(logfile_path, "a", encoding="utf-8")  # Keep every process descriptor append-only so stale offsets cannot overwrite records
        if clean:  # Reset prior run contents before concurrent writers exist
            self.logfile.seek(0)  # Position the initial coordinator descriptor for truncation
            self.logfile.truncate()  # Clear prior run contents while retaining append-only writes
        self.is_tty = sys.stdout.isatty()  # Verify if stdout is a TTY
        self.timestamp_zone = ZoneInfo(str(timestamp_timezone)) if timestamp_timezone is not None else None  # Resolve the explicit timezone once per logger instance
        self.timestamp_now = timestamp_now if timestamp_now is not None else datetime.now  # Use an injectable timezone-aware emission clock

    def format_message(self, message):  # Prefix every non-empty physical line at emission time
        """
        Format one complete log record with optional timezone-aware timestamps.

        :param self: Instance of the Logger class.
        :param message: Newline-terminated record text.
        :return: Record text with exactly one timestamp on every non-empty physical line.
        """

        if self.timestamp_zone is None:  # Preserve callers that do not opt into timestamp formatting
            return message  # Return the original record unchanged
        emitted_at = self.timestamp_now(self.timestamp_zone)  # Obtain the current explicit-zone time when the record is emitted
        timestamp_prefix = emitted_at.strftime("%d/%m/%Y - %Hh%Mm%Ss: ")  # Format the exact zero-padded Brazilian timestamp
        formatted_lines = []  # Collect physical lines while preserving original line endings
        for line in message.splitlines(keepends=True):  # Format every physical line in one locked logical record
            leading_match = LEADING_ANSI_ESCAPE_REGEX.match(line)  # Locate leading terminal controls such as screen clearing and colors
            leading_ansi = leading_match.group(0) if leading_match is not None else ""  # Preserve leading terminal controls before visible output
            visible_line = line[len(leading_ansi):]  # Read record content after leading terminal controls
            clean_visible_line = ANSI_ESCAPE_REGEX.sub("", visible_line)  # Normalize content before duplicate-prefix detection
            if not clean_visible_line.strip() or LOG_TIMESTAMP_PREFIX_REGEX.match(clean_visible_line):  # Preserve blank lines and already timestamped records
                formatted_lines.append(line)  # Keep the physical line unchanged
            else:  # Prefix one previously untimestamped non-empty physical line
                formatted_lines.append(f"{leading_ansi}{timestamp_prefix}{visible_line}")  # Place terminal controls before the visible timestamp without duplicating content
        return "".join(formatted_lines)  # Return one complete formatted record for locked output

    def write(self, message):
        """
        Internal method to write messages to both terminal and log file.

        :param self: Instance of the Logger class.
        :param message: The message to log.
        """

        if message is None:  # Ignore None messages
            return  # Early exit

        out = str(message)  # Convert message to string
        if not out.endswith("\n"):  # Ensure newline termination
            out += "\n"  # Append newline if missing

        lock_acquired = False  # Track the process-safe log lock for exception-safe release
        try:  # Serialize file and terminal writes from every stacking process
            fcntl.flock(self.logfile.fileno(), fcntl.LOCK_EX)  # Acquire an exclusive advisory lock for one complete record
            lock_acquired = True  # Record successful lock acquisition before writing
            out = self.format_message(out)  # Sample and apply the optional timestamp at synchronized emission
            clean_out = ANSI_ESCAPE_REGEX.sub("", out)  # Strip ANSI sequences for log file
            self.logfile.write(clean_out)  # Write the complete cleaned record while holding the process lock
            self.logfile.flush()  # Flush every locked record immediately
            if sys.__stdout__ is not None:  # Mirror the same complete record to the original terminal when available
                if self.is_tty:  # Terminal supports colors
                    sys.__stdout__.write(out)  # Write colored message
                    sys.__stdout__.flush()  # Flush immediately
                else:  # Terminal does not support colors
                    sys.__stdout__.write(clean_out)  # Write cleaned message
                    sys.__stdout__.flush()  # Flush immediately
        except Exception:  # Fail silently to avoid breaking user code
            pass  # Silent fail
        finally:  # Release the process-safe record lock after every outcome
            if lock_acquired:  # Unlock only when this write acquired the advisory lock
                try:  # Keep unlock failures from affecting user code
                    fcntl.flock(self.logfile.fileno(), fcntl.LOCK_UN)  # Release the shared log file for the next process
                except Exception:  # Fail silently during lock release
                    pass  # Preserve logger compatibility after an unlock failure

    def flush(self):
        """
        Flush the log file.

        :param self: Instance of the Logger class.
        """

        try:  # Flush log file buffer
            self.logfile.flush()  # Flush log file
        except Exception:  # Fail silently
            pass  # Silent fail

    def close(self):
        """
        Close the log file.

        :param self: Instance of the Logger class.
        """

        try:  # Close log file
            self.logfile.close()  # Close log file
        except Exception:  # Fail silently
            pass  # Silent fail
