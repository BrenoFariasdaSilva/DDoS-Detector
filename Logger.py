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
