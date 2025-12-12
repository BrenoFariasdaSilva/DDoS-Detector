"""
================================================================================
Logger Utility Module
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-12-11
Description :
   This module implements a dual-channel logging system designed to cleanly
   capture all console output produced by Python scripts.

   Its purpose is to provide consistent, color-safe logging across interactive
   terminals, background executions, CI pipelines, Makefile pipelines, and
   nohup/systemd environments.

   Key features include:
      - Automatic ANSI color detection and stripping for log files
      - Full compatibility with interactive and non-interactive terminals
      - Mirrored output: terminal + clean log file
      - Seamless integration with print() and exceptions
      - Drop-in replacement for sys.stdout and sys.stderr

Usage:
   1. Import the module in any script requiring safe logging.

         from logger import DualLogger
         sys.stdout = DualLogger("./Logs/output.log")
         sys.stderr = sys.stdout

   2. Run your script normally or via Makefile.  
         $ make dataset_converter   or   $ python script.py

   3. All printed output appears colored in the terminal (when interactive)
      and clean (color-free) in the log file.

Outputs:
   - <path>.log file containing fully sanitized log output
   - Real-time console output with correct ANSI handling

TODOs:
   - Add timestamp prefixing to all log lines
   - Add optional file rotation or max-size log splitting
   - Add a CLI flag to force-enable or disable color output
   - Add support for JSON-structured logs

Dependencies:
   - Python >= 3.8
   - colorama (optional but recommended)

Assumptions & Notes:
   - ANSI escape sequences follow the regex pattern \x1B\[[0-9;]*[a-zA-Z]
   - Log file is always written without ANSI sequences
   - If the output is redirected (not a TTY), color output is disabled
"""

import os # For interacting with the filesystem
import re # For stripping ANSI escape sequences
import sys # For replacing stdout/stderr

# Regex Constants:
ANSI_ESCAPE_REGEX = re.compile(r"\x1B\[[0-9;]*[a-zA-Z]") # Pattern to remove ANSI colors

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

   def __init__(self, logfile_path, clean=False):
      self.logfile_path = logfile_path

      # Ensure parent directory exists
      parent = os.path.dirname(logfile_path)
      if parent and not os.path.exists(parent):
         os.makedirs(parent, exist_ok=True)

      mode = "w" if clean else "a"
      self.logfile = open(logfile_path, mode, encoding="utf-8")
      self.is_tty = sys.stdout.isatty()

   def _write(self, message):
      # Accept None/empty strings gracefully
      if message is None:
         return

      out = str(message)
      if not out.endswith("\n"):
         out += "\n"

      clean_out = ANSI_ESCAPE_REGEX.sub("", out)

      # Write to file (cleaned)
      try:
         self.logfile.write(clean_out)
         self.logfile.flush()
      except Exception:
         # Fail silently to avoid breaking user code
         pass

      # Write to terminal: colored when TTY, cleaned otherwise
      try:
         if self.is_tty:
            sys.__stdout__.write(out)
            sys.__stdout__.flush()
         else:
            sys.__stdout__.write(clean_out)
            sys.__stdout__.flush()
      except Exception:
         pass

   def info(self, message):
      """Write an informational message to terminal and log file."""
      self._write(message)

   def warn(self, message):
      """Alias for warnings; kept for API familiarity."""
      self._write(message)

   def error(self, message):
      """Alias for errors; kept for API familiarity."""
      self._write(message)

   def flush(self):
      try:
         self.logfile.flush()
      except Exception:
         pass

   def close(self):
      try:
         self.logfile.close()
      except Exception:
         pass
