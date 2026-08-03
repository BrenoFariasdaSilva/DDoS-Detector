"""Plan exact OOM restarts for persistent stacking workers."""

import os  # Read process, environment, history, and cgroup paths.
import re  # Locate shell assignments without evaluating them.
import shlex  # Tokenize commands for safe comparison and launching.
import subprocess  # Launch the detached restart process.
from pathlib import Path  # Normalize repository and cgroup paths.
from typing import Any, Callable, Optional  # Type small restart planning hooks.

from utils.skip_combinations import normalize_plan_augmentation_ratio  # Reuse skip-rule augmentation normalization.


AUTO_RESTART_COMMAND_ENV = "DDOS_DETECTOR_AUTO_RESTART_COMMAND"  # Private env var carrying the exact transformed command.
AUTO_RESTART_ATTEMPT_ENV = "DDOS_DETECTOR_AUTO_RESTART_ATTEMPT"  # Private env var carrying restart attempt count.
ARGS_ASSIGNMENT_PATTERN = re.compile(r'ARGS=(["\'])(.*?)\1')  # Locate the Make ARGS assignment without shell execution.


def canonical_path_text(path: str) -> str:  # Normalize a path for command validation.
    """
    Normalize a path for command validation.

    :param path: Raw path text.
    :return: Resolved path text.
    """

    return str(Path(path).resolve())  # Return canonical filesystem text.


def command_targets_repository(command: str, repository_root: str) -> bool:  # Validate that a command targets this repository.
    """
    Validate that a command targets this repository and stacking execution.

    :param command: Candidate shell command.
    :param repository_root: Current repository root.
    :return: True when the command safely targets this repository.
    """

    root_text = canonical_path_text(repository_root)  # Resolve current repository root.
    normalized_command = command.replace("\\\\ ", " ")  # Normalize escaped spaces enough for path matching.
    targets_repo = root_text in normalized_command or str(Path(root_text).name) in normalized_command and "DDoS-Detector" in normalized_command  # Require visible repository identity.
    targets_stacking = "stacking.py" in normalized_command or "stacking-full" in normalized_command  # Require stacking execution identity.
    return bool(targets_repo and targets_stacking)  # Return strict validation result.


def read_proc_cmdline(pid: int) -> str:  # Read one Linux process command line.
    """
    Read one Linux process command line.

    :param pid: Process ID.
    :return: Space-joined command line, or empty text.
    """

    try:  # Keep process disappearance harmless.
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()  # Read null-delimited cmdline.
        return raw.replace(b"\0", b" ").decode("utf-8", errors="replace").strip()  # Return readable cmdline.
    except Exception:  # Treat inaccessible process metadata as absent.
        return ""  # Return empty command.


def read_proc_parent_pid(pid: int) -> Optional[int]:  # Read one Linux parent process ID.
    """
    Read one Linux parent process ID.

    :param pid: Process ID.
    :return: Parent PID, or None.
    """

    try:  # Keep process disappearance harmless.
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8", errors="replace").splitlines():  # Scan small status file.
            if line.startswith("PPid:"):  # Locate parent PID field.
                return int(line.split(":", 1)[1].strip())  # Return parent PID.
    except Exception:  # Treat inaccessible process metadata as absent.
        return None  # Return missing parent.
    return None  # Return missing parent.


def recover_command_from_environment(repository_root: str) -> Optional[dict]:  # Recover command from private env var.
    """
    Recover command from the private restart environment variable.

    :param repository_root: Current repository root.
    :return: Recovery mapping, or None.
    """

    command = os.environ.get(AUTO_RESTART_COMMAND_ENV, "").strip()  # Read private restart command.
    if command and command_targets_repository(command, repository_root):  # Accept only validated commands.
        return {"source": "environment", "command": command}  # Return recovered command.
    return None  # Report no valid env command.


def recover_command_from_proc(repository_root: str) -> Optional[dict]:  # Recover command from Linux ancestors.
    """
    Recover command from Linux ancestor process metadata.

    :param repository_root: Current repository root.
    :return: Recovery mapping, or None.
    """

    if not Path("/proc").is_dir():  # Limit proc recovery to Linux procfs.
        return None  # Report no proc recovery.
    pid = os.getpid()  # Start at current process.
    seen = set()  # Avoid parent loops from corrupt proc metadata.
    for _ in range(24):  # Bound ancestor traversal.
        if pid in seen or pid <= 1:  # Stop at root or cycles.
            break  # Leave traversal.
        seen.add(pid)  # Record visited PID.
        command = read_proc_cmdline(pid)  # Read this process command line.
        if command_targets_repository(command, repository_root):  # Accept newest matching ancestor.
            return {"source": "proc", "command": command}  # Return proc command.
        parent_pid = read_proc_parent_pid(pid)  # Read parent PID.
        if parent_pid is None:  # Stop when parent is unavailable.
            break  # Leave traversal.
        pid = parent_pid  # Move to parent process.
    return None  # Report no proc command.


def history_paths() -> list[Path]:  # Resolve shell history paths.
    """
    Resolve shell history paths.

    :return: Candidate history paths.
    """

    paths = []  # Accumulate candidate history files.
    histfile = os.environ.get("HISTFILE")  # Read explicit history path.
    if histfile:  # Prefer configured history file.
        paths.append(Path(histfile).expanduser())  # Add configured history file.
    paths.append(Path("~/.bash_history").expanduser())  # Add default Bash history.
    return list(dict.fromkeys(paths))  # Return unique paths in priority order.


def recover_command_from_history(repository_root: str) -> Optional[dict]:  # Recover command from shell history files.
    """
    Recover command from shell history files.

    :param repository_root: Current repository root.
    :return: Recovery mapping, or None.
    """

    for path in history_paths():  # Scan configured history files.
        try:  # Keep missing history harmless.
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()  # Read history lines.
        except Exception:  # Ignore inaccessible history files.
            continue  # Try next history path.
        for line in reversed(lines):  # Use newest valid matching entry.
            command = line.strip()  # Normalize history line.
            if command_targets_repository(command, repository_root):  # Accept only exact repository stacking commands.
                return {"source": f"history:{path}", "command": command}  # Return recovered history command.
    return None  # Report no history command.


def recover_launch_command(repository_root: str) -> dict:  # Recover exact launch command by priority.
    """
    Recover exact launch command by configured priority.

    :param repository_root: Current repository root.
    :return: Recovery result mapping.
    """

    for recover in (recover_command_from_environment, recover_command_from_proc, recover_command_from_history):  # Try sources in required order.
        result = recover(repository_root)  # Attempt one recovery source.
        if result is not None:  # Use first safe match.
            return {**result, "ok": True, "reason": None}  # Return successful recovery.
    return {"ok": False, "source": None, "command": None, "reason": "exact launch command unavailable"}  # Return failure reason.


def read_cgroup_paths() -> list[Path]:  # Resolve current cgroup v2 paths.
    """
    Resolve current cgroup v2 paths.

    :return: Candidate cgroup directories.
    """

    paths = []  # Accumulate cgroup directories.
    try:  # Read current process cgroup membership.
        for line in Path("/proc/self/cgroup").read_text(encoding="utf-8", errors="replace").splitlines():  # Parse membership rows.
            relative_path = line.strip().split(":")[-1].lstrip("/")  # Extract cgroup-relative path.
            paths.append(Path("/sys/fs/cgroup") / relative_path if relative_path else Path("/sys/fs/cgroup"))  # Store resolved path.
    except Exception:  # Keep missing proc data harmless.
        pass  # Continue with root fallback.
    if Path("/sys/fs/cgroup").is_dir():  # Include root cgroup when visible.
        paths.append(Path("/sys/fs/cgroup"))  # Add root cgroup.
    return list(dict.fromkeys(paths))  # Return unique cgroup paths.


def read_oom_kill_count(path: Path) -> Optional[int]:  # Read one cgroup oom_kill counter.
    """
    Read one cgroup oom_kill counter.

    :param path: Cgroup directory.
    :return: oom_kill counter, or None.
    """

    events_path = path / "memory.events"  # Resolve cgroup v2 memory events file.
    try:  # Keep absent cgroups harmless.
        for line in events_path.read_text(encoding="utf-8", errors="replace").splitlines():  # Parse memory event rows.
            name, value = line.split(maxsplit=1)  # Split event name and counter.
            if name == "oom_kill":  # Locate OOM kill counter.
                return int(value)  # Return numeric counter.
    except Exception:  # Treat unreadable counters as absent.
        return None  # Return missing counter.
    return None  # Return missing counter.


def capture_oom_baseline() -> dict:  # Capture current cgroup oom_kill counters.
    """
    Capture current cgroup oom_kill counters.

    :return: Mapping of cgroup path to oom_kill count.
    """

    return {str(path): value for path in read_cgroup_paths() for value in [read_oom_kill_count(path)] if value is not None}  # Return visible counters.


def oom_kill_delta(baseline: Optional[dict], current: Optional[dict] = None) -> dict:  # Compare cgroup OOM counters.
    """
    Compare cgroup OOM counters.

    :param baseline: Baseline counter mapping.
    :param current: Current counter mapping.
    :return: Delta summary.
    """

    baseline_counts = baseline if isinstance(baseline, dict) else {}  # Normalize baseline.
    current_counts = current if isinstance(current, dict) else capture_oom_baseline()  # Read current counters only when not supplied.
    increases = {path: int(value) - int(baseline_counts.get(path, value)) for path, value in current_counts.items() if int(value) > int(baseline_counts.get(path, value))}  # Keep positive deltas only.
    return {"confirmed": bool(increases), "baseline": baseline_counts, "current": current_counts, "increases": increases}  # Return confirmation summary.


def first_alias(canonical_value: str, aliases_by_value: dict) -> Optional[str]:  # Resolve the preferred short alias for one canonical value.
    """
    Resolve the preferred short alias for one canonical value.

    :param canonical_value: Runtime canonical value.
    :param aliases_by_value: Canonical-to-alias mapping.
    :return: Preferred alias text, or None.
    """

    aliases = aliases_by_value.get(canonical_value)  # Read aliases for canonical value.
    if not aliases:  # Reject missing alias mapping.
        return None  # Report missing alias.
    return str(tuple(aliases)[0])  # Return first configured alias.


def build_exact_oom_skip_rule(task: Optional[dict], feature_aliases: dict, classifier_aliases: dict, hyperparameter_aliases: dict) -> dict:  # Build exact four-dimensional OOM skip rule.
    """
    Build exact four-dimensional OOM skip rule.

    :param task: Authoritative task metadata.
    :param feature_aliases: Runtime feature aliases.
    :param classifier_aliases: Runtime classifier aliases.
    :param hyperparameter_aliases: Runtime hyperparameter aliases.
    :return: Rule result mapping.
    """

    if not isinstance(task, dict):  # Require authoritative task metadata.
        return {"ok": False, "rule": None, "missing": ["task"]}  # Report missing task.
    hyperparameter_label = "Optimized Hyperparameters" if task.get("hyperparameters_enabled") else "Default Hyperparameters" if task.get("hyperparameters_enabled") is not None else None  # Resolve exact hyperparameter label.
    values = {"feature_set": task.get("feature_set"), "classifier": task.get("classifier_name"), "hyperparameters": hyperparameter_label, "augmentation_ratio": normalize_plan_augmentation_ratio(task.get("augmentation_ratio")) if "augmentation_ratio" in task else None}  # Build required dimensions.
    missing = [field for field, value in values.items() if value is None]  # Identify unresolved dimensions.
    if missing:  # Abort when any required dimension is missing.
        return {"ok": False, "rule": None, "missing": missing}  # Return missing field list.
    feature_alias = first_alias(str(values["feature_set"]), feature_aliases)  # Resolve feature alias.
    classifier_alias = first_alias(str(values["classifier"]), classifier_aliases)  # Resolve classifier alias.
    hyperparameter_alias = first_alias(str(values["hyperparameters"]), hyperparameter_aliases)  # Resolve hyperparameter alias.
    alias_missing = [name for name, value in (("feature_set", feature_alias), ("classifier", classifier_alias), ("hyperparameters", hyperparameter_alias)) if value is None]  # Identify missing aliases.
    if alias_missing:  # Abort when aliases are unavailable.
        return {"ok": False, "rule": None, "missing": alias_missing}  # Return missing alias list.
    rule = f"{feature_alias}&{classifier_alias}&{hyperparameter_alias}&{values['augmentation_ratio']}"  # Build exact four-dimensional rule.
    return {"ok": True, "rule": rule, "missing": [], "details": values}  # Return generated rule.


def split_command_tokens(command: str) -> list[str]:  # Split a command for comparison.
    """
    Split a command for comparison.

    :param command: Shell command text.
    :return: Shell tokens, or whitespace tokens on malformed input.
    """

    try:  # Prefer shell-compatible parsing.
        return shlex.split(command)  # Return parsed tokens.
    except ValueError:  # Fall back for incomplete history lines.
        return command.split()  # Return coarse tokens.


def remove_skip_tokens(tokens: list[str]) -> list[str]:  # Remove repeated skip arguments from tokens.
    """
    Remove repeated skip-combination arguments from tokens.

    :param tokens: Command tokens.
    :return: Tokens without skip-combination arguments.
    """

    result = []  # Accumulate non-skip tokens.
    index = 0  # Track current token index.
    while index < len(tokens):  # Walk every token.
        token = tokens[index]  # Read current token.
        if token == "--skip-combination":  # Remove split skip option and value.
            index += 2  # Skip option and value.
            continue  # Continue scanning.
        if token.startswith("--skip-combination="):  # Remove equals-form skip option.
            index += 1  # Skip this token.
            continue  # Continue scanning.
        result.append(token)  # Preserve all other tokens.
        index += 1  # Advance normally.
    return result  # Return filtered tokens.


def extract_skip_values(tokens: list[str]) -> list[str]:  # Extract skip-combination values from tokens.
    """
    Extract skip-combination values from tokens.

    :param tokens: Command tokens.
    :return: Ordered skip values.
    """

    values = []  # Accumulate skip values.
    index = 0  # Track current token index.
    while index < len(tokens):  # Walk every token.
        token = tokens[index]  # Read current token.
        if token == "--skip-combination" and index + 1 < len(tokens):  # Read split skip option.
            values.append(tokens[index + 1])  # Store following value.
            index += 2  # Skip option and value.
            continue  # Continue scanning.
        if token.startswith("--skip-combination="):  # Read equals-form skip option.
            values.append(token.split("=", 1)[1])  # Store attached value.
        index += 1  # Advance token index.
    return values  # Return ordered values.


def append_skip_to_args_text(args_text: str, skip_values: list[str]) -> str:  # Append skip arguments to Make ARGS text.
    """
    Append skip arguments to Make ARGS text.

    :param args_text: Existing ARGS value.
    :param skip_values: Skip values to append.
    :return: Updated ARGS value.
    """

    suffix = "".join(f" --skip-combination {shlex.quote(value)}" for value in skip_values)  # Build safely quoted skip suffix.
    return f"{args_text}{suffix}"  # Return updated ARGS text.


def transform_command_with_skip_rule(command: str, new_rule: str, effective_rule_texts: list[str], new_rule_canonical: str, existing_canonical_rules: set[str], canonicalize_rule: Callable[[str], str]) -> dict:  # Append exact skip rule to a launch command.
    """
    Append exact skip rule to a launch command.

    :param command: Original shell command.
    :param new_rule: Exact generated skip rule.
    :param effective_rule_texts: Effective YAML or CLI rule texts.
    :param new_rule_canonical: Canonical generated rule.
    :param existing_canonical_rules: Existing canonical rule set.
    :param canonicalize_rule: Callable returning canonical rule text.
    :return: Transformation result.
    """

    if new_rule_canonical in existing_canonical_rules:  # Prevent duplicate exact restarts.
        return {"ok": False, "reason": "generated exact skip rule already effective", "updated_command": None, "previous_skip_count": len(existing_canonical_rules), "updated_skip_count": len(existing_canonical_rules), "only_skip_args_changed": True}  # Return duplicate reason.
    match = ARGS_ASSIGNMENT_PATTERN.search(command)  # Locate Make ARGS value when present.
    if match:  # Update only ARGS skip arguments.
        args_text = match.group(2)  # Read existing ARGS text.
        args_tokens = split_command_tokens(args_text)  # Parse ARGS value.
        existing_cli_values = extract_skip_values(args_tokens)  # Read existing CLI skip values.
        prepend_values = [] if existing_cli_values else list(effective_rule_texts)  # Preserve YAML effective rules when CLI would replace them.
        append_values = [value for value in prepend_values + [new_rule]]  # Build values to append.
        updated_args_text = append_skip_to_args_text(args_text, append_values)  # Append skip args.
        updated_command = f"{command[:match.start(2)]}{updated_args_text}{command[match.end(2):]}"  # Splice updated ARGS text into original command.
        original_compare_tokens = remove_skip_tokens(split_command_tokens(args_text))  # Remove skip args from original ARGS tokens.
        updated_compare_tokens = remove_skip_tokens(split_command_tokens(updated_args_text))  # Remove skip args from updated ARGS tokens.
    else:  # Append skip arguments to the command itself.
        existing_cli_values = extract_skip_values(split_command_tokens(command))  # Read existing CLI skip values.
        prepend_values = [] if existing_cli_values else list(effective_rule_texts)  # Preserve YAML effective rules when CLI would replace them.
        append_values = prepend_values + [new_rule]  # Build skip values to append.
        updated_command = f"{command}{''.join(f' --skip-combination {shlex.quote(value)}' for value in append_values)}"  # Append skip args to command.
        original_compare_tokens = remove_skip_tokens(split_command_tokens(command))  # Remove skip args from original command tokens.
        updated_compare_tokens = remove_skip_tokens(split_command_tokens(updated_command))  # Remove skip args from updated command tokens.
    updated_values = existing_cli_values + append_values  # Combine visible old and new CLI rule values.
    updated_canonical_rules = {canonicalize_rule(value) for value in updated_values}  # Canonicalize updated visible skip values.
    only_skip_args_changed = original_compare_tokens == updated_compare_tokens  # Verify non-skip command tokens are unchanged.
    if not only_skip_args_changed:  # Abort unsafe transformation.
        return {"ok": False, "reason": "command transformation changed non-skip arguments", "updated_command": updated_command, "previous_skip_count": len(existing_canonical_rules), "updated_skip_count": len(updated_canonical_rules), "only_skip_args_changed": False}  # Return mismatch reason.
    return {"ok": True, "reason": None, "updated_command": updated_command, "previous_skip_count": len(existing_canonical_rules), "updated_skip_count": len(updated_canonical_rules), "only_skip_args_changed": True, "added_skip_values": append_values}  # Return successful transformation.


def schedule_detached_restart(updated_command: str, repository_root: str, attempt: int) -> dict:  # Schedule detached restart after current process exits.
    """
    Schedule detached restart after current process exits.

    :param updated_command: Updated shell command.
    :param repository_root: Repository root for restart cwd.
    :param attempt: Restart attempt number.
    :return: Scheduling result.
    """

    env = os.environ.copy()  # Copy current environment for restart.
    env[AUTO_RESTART_COMMAND_ENV] = updated_command  # Store exact command for future recovery.
    env[AUTO_RESTART_ATTEMPT_ENV] = str(attempt)  # Store attempt count.
    parent_pid = os.getpid()  # Capture current coordinator PID.
    launcher = f"while kill -0 {parent_pid} 2>/dev/null; do sleep 1; done; exec /bin/bash -lc \"$DDOS_DETECTOR_AUTO_RESTART_COMMAND\""  # Wait for failed coordinator before relaunch.
    process = subprocess.Popen(["/bin/bash", "-lc", launcher], cwd=repository_root, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)  # Launch detached waiter without shell=True.
    return {"scheduled": True, "pid": process.pid, "attempt": attempt}  # Return detached launcher metadata.
