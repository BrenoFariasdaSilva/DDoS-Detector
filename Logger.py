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

class DualLogger:
   """
   A logging class that mirrors all console output to both the terminal and
   a log file, while stripping ANSI escape codes from the log file.

   This ensures that the log file remains clean for archival, debugging,
   or machine parsing, while the console retains colored output when
   running interactively.

   :param logfile_path: Path to the output log file.
   :return: None
   """

   def __init__(self, logfile_path):
      """
      Initializes the DualLogger instance.

      :param logfile_path: The path where logs will be written.
      :return: None
      """

      self.logfile_path = logfile_path # Store the log file path
      self.logfile = open(logfile_path, "w", encoding="utf-8") # Open the log file

      self.is_tty = sys.stdout.isatty() # Verify if running in a real terminal

   def write(self, data):
      """
      Writes output to both the terminal (optionally with color) and
      the log file (without ANSI codes).

      :param data: The raw string to be written.
      :return: None
      """

      if not data: # If there is no output, do nothing
         return # Early exit

      clean_data = ANSI_ESCAPE_REGEX.sub("", data) # Remove ANSI codes for log file

      self.logfile.write(clean_data) # Write clean data to the log file
      self.logfile.flush() # Ensure it is written immediately

      if self.is_tty: # If running in a real terminal
         sys.__stdout__.write(data) # Write colored output
         sys.__stdout__.flush() # Ensure it is written immediately
      else: # Non-TTY: strip colors to avoid garbled characters
         sys.__stdout__.write(clean_data) # Write clean output
         sys.__stdout__.flush() # Ensure it is written immediately

   def flush(self):
      """
      Flushes both the log file and terminal output streams.

      :return: None
      """

      self.logfile.flush() # Flush the log file
      sys.__stdout__.flush() # Flush terminal output
