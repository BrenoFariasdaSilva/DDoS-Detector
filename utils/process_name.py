"""Operating-system process-title utilities."""

from pathlib import Path  # Normalize script and path-like CLI values.
import re  # Sanitize process-title fragments consistently.
import sys  # Read current CLI arguments when callers do not pass them explicitly.
from typing import Any, Optional, Sequence  # Type optional process-name overrides, configs, and argv fragments.


PROCESS_TITLE_PREFIX = "DDoSDetector"  # Shared prefix for all user-facing process titles.
PATH_NOISE_SEGMENTS = {"dataset", "datasets"}  # Drop redundant path wrappers already implied by option names.
DISPLAY_WORDS = {"api": "API", "arff": "ARFF", "automl": "AutoML", "cpu": "CPU", "csv": "CSV", "ga": "GA", "gpu": "GPU", "json": "JSON", "lstm": "LSTM", "n": "N", "oom": "OOM", "os": "OS", "pca": "PCA", "pid": "PID", "ram": "RAM", "rfe": "RFE", "shap": "SHAP", "svm": "SVM", "txt": "TXT", "wgan": "WGAN", "xml": "XML", "yaml": "YAML"}  # Preserve common acronym casing in generated titles.


def _format_display_word(word: str) -> str:
    """Return one process-title word using repository-specific acronym casing."""

    normalized = re.sub(r"[^A-Za-z0-9]+", "", str(word or "").strip())
    if not normalized:
        return ""
    return DISPLAY_WORDS.get(normalized.lower(), normalized[:1].upper() + normalized[1:])


def _format_option_name(option_name: str) -> str:
    """Convert one CLI option name into a compact display token."""

    parts = [_format_display_word(part) for part in re.split(r"[^A-Za-z0-9]+", str(option_name or "").strip()) if part]
    return "".join(part for part in parts if part)


SCRIPT_SECTION_ALIASES = {"extratrees": ("extra_trees", "extratrees"), "genetic_algorithm": ("genetic_algorithm",), "hyperparameters_optimization": ("hyperparameters_optimization",), "dataset_converter": ("dataset_converter",), "dataset_descriptor": ("dataset_descriptor",), "stacking": ("stacking",), "pca": ("pca",), "rfe": ("rfe",), "wgangp": ("wgangp",), "main": ()}  # Match known config section names for process-name-enabled entry points.


def _normalize_requested_process_name(process_name: Optional[str]) -> Optional[str]:
    """Return one exact user-requested process title after validation."""

    if process_name is None:
        return None
    normalized_name = str(process_name).strip()
    if not normalized_name or "\0" in normalized_name:
        raise ValueError("--process-name must be a non-empty string without null bytes")
    return normalized_name


def resolve_configured_process_name(config: Optional[dict[str, Any]], script_path: Optional[str] = None) -> Optional[str]:
    """Resolve one exact process-name override from merged configuration."""

    if not isinstance(config, dict):
        return None
    for direct_name in (config.get("process_name"), config.get("runtime", {}).get("process_name"), config.get("execution", {}).get("process_name")):
        normalized_name = _normalize_requested_process_name(direct_name)
        if normalized_name is not None:
            return normalized_name
    script_stem = Path(str(script_path or sys.argv[0] or "python")).stem
    for section_name in SCRIPT_SECTION_ALIASES.get(script_stem, (script_stem,)):
        section = config.get(section_name)
        if isinstance(section, dict):
            normalized_name = _normalize_requested_process_name(section.get("process_name"))
            if normalized_name is not None:
                return normalized_name
    return None


def _normalize_path_value(value: str) -> list[str]:
    """Return meaningful trailing path fragments for one CLI path-like value."""

    raw_segments = [segment for segment in re.split(r"[\\/]+", str(value or "").strip()) if segment not in {"", ".", ".."}]
    meaningful_segments = [segment for segment in raw_segments if segment.lower() not in PATH_NOISE_SEGMENTS]
    selected_segments = meaningful_segments[-2:] if len(meaningful_segments) > 2 else meaningful_segments
    return [re.sub(r"[^A-Za-z0-9.]+", "-", segment).strip("-") for segment in selected_segments if re.sub(r"[^A-Za-z0-9.]+", "-", segment).strip("-")]


def _normalize_option_value(value: str) -> list[str]:
    """Normalize one CLI value into process-title fragments."""

    normalized_value = str(value or "").strip().strip("\"'")
    if not normalized_value:
        return []
    if re.fullmatch(r"-\d+(?:\.\d+)?", normalized_value):
        return [f"Minus{normalized_value[1:]}"]
    if "/" in normalized_value or "\\" in normalized_value:
        path_tokens = _normalize_path_value(normalized_value)
        if path_tokens:
            return path_tokens
    compact_value = re.sub(r"[^A-Za-z0-9.]+", "-", normalized_value).strip("-")
    return [compact_value] if compact_value else []


def build_runtime_process_name(script_path: Optional[str], argv: Optional[Sequence[str]] = None) -> str:
    """
    Build one default process title from script identity and CLI arguments.

    :param script_path: Script path used to derive the filename fragment.
    :param argv: Optional CLI argument sequence excluding the executable.
    :return: Generated process title.
    """

    script_label = Path(str(script_path or sys.argv[0] or "python")).name
    title_parts = [PROCESS_TITLE_PREFIX, script_label]
    seen_parts = {PROCESS_TITLE_PREFIX.lower(), script_label.lower()}

    def append_part(part: str) -> None:
        normalized_part = str(part or "").strip()
        dedupe_key = normalized_part.lower()
        dedupe_allowed = any(character.isalpha() for character in normalized_part)
        if not normalized_part or dedupe_allowed and dedupe_key in seen_parts:
            return
        title_parts.append(normalized_part)
        if dedupe_allowed:
            seen_parts.add(dedupe_key)

    cli_tokens = list(sys.argv[1:] if argv is None else argv)
    index = 0
    while index < len(cli_tokens):
        token = str(cli_tokens[index] or "").strip()
        if not token:
            index += 1
            continue
        if token == "--":
            break
        if token.startswith("--"):
            option_text = token[2:]
            inline_value = None
            if "=" in option_text:
                option_text, inline_value = option_text.split("=", 1)
            if option_text == "process-name":
                if inline_value is None and index + 1 < len(cli_tokens):
                    index += 2
                else:
                    index += 1
                continue
            option_label = _format_option_name(option_text)
            if option_label:
                append_part(option_label)
            if inline_value is not None:
                for part in _normalize_option_value(inline_value):
                    append_part(part)
                index += 1
                continue
            if index + 1 < len(cli_tokens) and str(cli_tokens[index + 1]) != "--" and not str(cli_tokens[index + 1]).startswith("--"):
                for part in _normalize_option_value(str(cli_tokens[index + 1])):
                    append_part(part)
                index += 2
                continue
            index += 1
            continue
        if token.startswith("-") and len(token) > 1:
            option_label = _format_option_name(token.lstrip("-"))
            if option_label:
                append_part(option_label)
            if index + 1 < len(cli_tokens) and str(cli_tokens[index + 1]) != "--" and not str(cli_tokens[index + 1]).startswith("--"):
                for part in _normalize_option_value(str(cli_tokens[index + 1])):
                    append_part(part)
                index += 2
                continue
        index += 1
    return "-".join(title_parts)


def resolve_runtime_process_name(process_name: Optional[str], script_path: Optional[str] = None, argv: Optional[Sequence[str]] = None, config: Optional[dict[str, Any]] = None) -> str:
    """
    Resolve one exact runtime process title from CLI, config, or generated fallback.

    :param process_name: Requested exact process title from CLI, when provided.
    :param script_path: Optional script path used to include the current filename in the generated title.
    :param argv: Optional CLI argument sequence excluding the executable.
    :param config: Optional merged configuration used for exact process-name overrides when CLI did not specify one.
    :return: Resolved process title.
    """

    normalized_name = _normalize_requested_process_name(process_name)  # Prefer exact CLI override when present.
    configured_name = resolve_configured_process_name(config, script_path=script_path) if normalized_name is None else None  # Fall back to exact config override only when CLI omitted one.
    return normalized_name or configured_name or build_runtime_process_name(script_path=script_path, argv=argv)  # Resolve one stable runtime title for parent and child processes.


def set_runtime_process_name(process_name: Optional[str], script_path: Optional[str] = None, argv: Optional[Sequence[str]] = None, config: Optional[dict[str, Any]] = None) -> None:
    """
    Set the operating-system process title shown by tools such as htop.

    :param process_name: Requested exact process title from CLI, when provided.
    :param script_path: Optional script path used to include the current filename in the generated title.
    :param argv: Optional CLI argument sequence excluding the executable.
    :param config: Optional merged configuration used for exact process-name overrides when CLI did not specify one.
    :return: None.
    """

    if script_path is None and process_name is None and config is None:  # Preserve the legacy exact-title behavior for old direct calls.
        return  # Leave the process title unchanged.
    try:  # Load the existing optional process-title dependency only when requested.
        from setproctitle import setproctitle  # Import the platform process-title setter.
    except ImportError as error:  # Fail explicitly because the requested htop identity cannot be applied.
        raise RuntimeError("--process-name requires the setproctitle package") from error  # Preserve the missing dependency cause.
    setproctitle(resolve_runtime_process_name(process_name, script_path=script_path, argv=argv, config=config))  # Apply exact CLI, then exact config, then generated default.
