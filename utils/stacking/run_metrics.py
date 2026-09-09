"""Cross-run stacking cache metrics summary generation."""

import csv  # Read and write deterministic CSV rows.
import math  # Calculate standard errors.
import os  # Publish files atomically.
from pathlib import Path  # Resolve cache paths.
import tempfile  # Stage same-directory summary files.
from typing import Any, Callable, Dict, List, Optional, Tuple  # Preserve explicit callback annotations.

RUN_METRICS_FILENAME = "Run_Metrics.csv"  # Store the derived cross-run summary filename.
EXPERIMENT_CONFIGURATION_COLUMNS = [  # Preserve non-result columns useful for identifying one experiment configuration.
    "experiment_mode",  # Include original versus augmented testing mode.
    "execution_mode",  # Include separate-files versus combined-files mode.
    "data_source",  # Include original or augmented source label.
    "dataset",  # Include dataset identity because one Cache_Results directory can contain multiple cache families.
    "attack_types_combined",  # Include combined-files attack scope.
    "augmentation_ratio",  # Include exact augmented-test ratio.
    "feature_selection_enabled",  # Include effective feature-selection mode.
    "hyperparameters_enabled",  # Include default versus optimized boolean mode.
    "data_augmentation_enabled",  # Include effective data-augmentation mode.
    "hyperparameter_mode",  # Include explicit hyperparameter mode label.
    "feature_set",  # Include selected feature representation.
    "classifier_type",  # Include individual versus stacking classifier group.
    "model_name",  # Include configured classifier name.
    "model",  # Include concrete estimator class.
    "n_features",  # Include evaluated feature count as configuration context.
    "n_samples_train",  # Include training sample count as execution context.
    "n_samples_test",  # Include testing sample count as execution context.
    "cv_method",  # Include cross-validation method metadata.
    "hyperparameters",  # Include persisted estimator parameter metadata for auditability.
    "features_list",  # Include canonical feature payload for auditability.
]  # Finish deterministic configuration column order.
AGGREGATED_METRICS = [  # Limit statistics to result metrics and code-backed runtime fields.
    ("f1_score", "F1-Score", "higher"),  # Treat higher F1 as better.
    ("accuracy", "Accuracy", "higher"),  # Treat higher accuracy as better.
    ("precision", "Precision", "higher"),  # Treat higher precision as better.
    ("recall", "Recall", "higher"),  # Treat higher recall as better.
    ("fpr", "FPR", "lower"),  # Treat lower false-positive rate as better.
    ("fnr", "FNR", "lower"),  # Treat lower false-negative rate as better.
    ("elapsed_time_s", "elapsed_time_s", "lower"),  # Treat lower elapsed runtime as better.
    ("preprocessing_time_s", "preprocessing_time_s", "lower"),  # Treat lower preprocessing runtime as better.
    ("feature_selection_time_s", "feature_selection_time_s", "lower"),  # Treat lower selector runtime as better.
    (  # Treat lower search runtime as better.
        "hyperparameter_optimization_time_s",  # Use persisted optimization runtime column.
        "hyperparameter_optimization_time_s",  # Use existing column name as output label.
        "lower",  # Mark lower values as better.
    ),
    ("training_time_s", "training_time_s", "lower"),  # Treat lower training runtime as better.
    ("inference_time_s", "inference_time_s", "lower"),  # Treat lower inference runtime as better.
]  # Finish deterministic metric order.


def student_t_critical(degrees_freedom: int, probability: float = 0.975) -> float:
    """
    Resolve Student t critical value.

    :param degrees_freedom: Positive degrees of freedom.
    :param probability: Target cumulative probability.
    :return: Student t critical value.
    """

    from scipy.stats import t as scipy_student_t  # type: ignore[import-untyped]  # Import project SciPy lazily.
    return float(scipy_student_t.ppf(probability, degrees_freedom))  # Return Student t critical value.


def strip_known_cache_suffix(filename: str) -> str:
    """
    Strip known cache artifact suffixes from one filename.

    :param filename: Cache artifact filename.
    :return: Primary CSV filename.
    """

    primary_name = str(filename)  # Normalize filename text.
    for suffix in (".bak", ".lock"):  # Traverse supported sibling artifact suffixes.
        if primary_name.endswith(suffix):  # Detect one sibling artifact suffix.
            primary_name = primary_name[: -len(suffix)]  # Remove only the recognized suffix.
    return primary_name  # Return primary-style filename.


def parse_logical_run_filename(filename: str) -> Optional[Tuple[str, int, str]]:
    """
    Parse one logical cache family and run number from a primary CSV filename.

    :param filename: Primary CSV filename.
    :return: Tuple of family filename, run number, and primary filename, or None.
    """

    primary_name = strip_known_cache_suffix(filename)  # Normalize backup and lock names to primary names.
    if primary_name == RUN_METRICS_FILENAME:  # Exclude the derived summary artifact.
        return None  # Return no logical cache run.
    if not primary_name.endswith(".csv"):  # Exclude non-CSV artifacts.
        return None  # Return no logical cache run.
    stem = primary_name[:-4]  # Remove the CSV suffix for run parsing.
    marker = "_Run_"  # Define the existing run-number delimiter.
    if marker not in stem:  # Treat legacy unsuffixed cache files as run 1.
        return primary_name, 1, primary_name  # Return legacy run identity.
    family_stem, run_text = stem.rsplit(marker, 1)  # Split only the final run marker.
    if not run_text.isdigit() or int(run_text) < 1:  # Reject malformed run suffixes.
        return None  # Return no logical cache run.
    return f"{family_stem}.csv", int(run_text), primary_name  # Return run-specific cache identity.


def discover_logical_cache_runs(cache_directory: Path) -> List[dict]:
    """
    Discover logical cache runs from primary, backup, and lock artifacts.

    :param cache_directory: Cache_Results directory to scan.
    :return: Deterministically ordered logical cache run records.
    """

    logical_runs: Dict[Tuple[str, int], dict] = {}  # Accumulate one record per family and run.
    if not cache_directory.is_dir():  # Handle absent cache directory as empty source state.
        return []  # Return no logical runs.
    candidates = sorted(cache_directory.iterdir(), key=lambda path: path.name)  # Traverse entries deterministically.
    for candidate in candidates:  # Process each candidate artifact.
        if not candidate.is_file():  # Ignore directories and special files.
            continue  # Move to next cache entry.
        parsed = parse_logical_run_filename(candidate.name)  # Parse existing cache naming semantics.
        if parsed is None:  # Ignore unrelated artifacts.
            continue  # Move to next cache entry.
        family_name, run_number, primary_name = parsed  # Unpack logical cache identity.
        record_key = (family_name, run_number)  # Build stable logical run key.
        logical_runs[record_key] = {  # Store one logical run record.
            "family": family_name,  # Store cache family name.
            "run": run_number,  # Store logical run number.
            "primary": cache_directory / primary_name,  # Store primary cache path.
        }
    ordered_keys = sorted(logical_runs, key=lambda item: (item[0], item[1]))  # Sort by family and run.
    return [logical_runs[key] for key in ordered_keys]  # Return family-then-run order.


def finite_number(value: Any) -> Optional[float]:
    """
    Convert one scalar to a finite float when possible.

    :param value: Raw scalar value.
    :return: Finite float value, or None.
    """

    if value is None:  # Reject absent values.
        return None  # Return no numeric observation.
    if isinstance(value, str) and not value.strip():  # Reject empty strings.
        return None  # Return no numeric observation.
    try:  # Convert numeric strings and scalar values.
        numeric_value = float(value)  # Normalize to a Python float.
    except (TypeError, ValueError):  # Reject malformed values.
        return None  # Return no numeric observation.
    if not math.isfinite(numeric_value):  # Reject NaN and infinity.
        return None  # Return no numeric observation.
    return numeric_value  # Return valid finite value.


def calculate_metric_statistics(observations: List[Tuple[int, float]], direction: str) -> dict:
    """
    Calculate repeated-run statistics for one metric.

    :param observations: Pairs of run number and finite metric value.
    :param direction: Metric direction, either higher or lower.
    :return: Statistic values for one metric.
    """

    sorted_observations = sorted(observations, key=lambda item: item[0])  # Preserve run-number tie handling.
    values = [value for _, value in sorted_observations]  # Extract finite metric values.
    sample_count = len(values)  # Count valid finite observations.
    result = {  # Initialize empty statistical output.
        "Sample Count": sample_count,  # Store finite observation count.
        "Mean": "",  # Initialize mean.
        "Standard Deviation": "",  # Initialize sample standard deviation.
        "95% CI Lower": "",  # Initialize lower confidence bound.
        "95% CI Upper": "",  # Initialize upper confidence bound.
        "Best": "",  # Initialize best value.
        "Best Run": "",  # Initialize best run.
        "Worst": "",  # Initialize worst value.
        "Worst Run": "",  # Initialize worst run.
    }
    if sample_count == 0:  # Leave every statistic empty when no valid values exist.
        return result  # Return empty statistic result.
    mean_value = sum(values) / sample_count  # Calculate arithmetic mean.
    result["Mean"] = mean_value  # Store mean for any valid sample count.
    if direction == "higher":  # Select higher-better extrema.
        best_observation = max(sorted_observations, key=lambda item: (item[1], -item[0]))  # Use lowest-run tie.
        worst_observation = min(sorted_observations, key=lambda item: (item[1], item[0]))  # Use lowest-run tie.
    else:  # Select lower-better extrema.
        best_observation = min(sorted_observations, key=lambda item: (item[1], item[0]))  # Use lowest-run tie.
        worst_observation = max(sorted_observations, key=lambda item: (item[1], -item[0]))  # Use lowest-run tie.
    result["Best"] = best_observation[1]  # Store best metric value.
    result["Best Run"] = best_observation[0]  # Store best logical run number.
    result["Worst"] = worst_observation[1]  # Store worst metric value.
    result["Worst Run"] = worst_observation[0]  # Store worst logical run number.
    if sample_count < 2:  # Avoid fabricated sample spread or confidence intervals.
        return result  # Return single-sample statistics.
    squared_error_sum = sum((value - mean_value) ** 2 for value in values)  # Sum deviations from the mean.
    variance = squared_error_sum / (sample_count - 1)  # Calculate sample variance with one degree of freedom.
    standard_deviation = math.sqrt(variance)  # Calculate sample standard deviation.
    standard_error = standard_deviation / math.sqrt(sample_count)  # Calculate standard error of the mean.
    t_critical = student_t_critical(sample_count - 1, 0.975)  # Resolve two-sided Student t critical value.
    interval_margin = t_critical * standard_error  # Calculate confidence interval margin for the mean.
    result["Standard Deviation"] = standard_deviation  # Store sample standard deviation.
    result["95% CI Lower"] = mean_value - interval_margin  # Store lower confidence bound for the mean.
    result["95% CI Upper"] = mean_value + interval_margin  # Store upper confidence bound for the mean.
    return result  # Return complete metric statistics.


def build_run_metrics_header() -> List[str]:
    """
    Build deterministic Run_Metrics.csv header columns.

    :return: Ordered summary CSV header columns.
    """

    header = list(EXPERIMENT_CONFIGURATION_COLUMNS) + ["Run Count", "Runs"]  # Start with identity and run fields.
    for _, metric_label, _ in AGGREGATED_METRICS:  # Append one stable statistics block per metric.
        statistic_names = (  # Preserve deterministic statistic order.
            "Sample Count",  # Include valid finite value count.
            "Mean",  # Include arithmetic mean.
            "Standard Deviation",  # Include sample standard deviation.
            "95% CI Lower",  # Include lower confidence bound.
            "95% CI Upper",  # Include upper confidence bound.
            "Best",  # Include best value.
            "Best Run",  # Include best run.
            "Worst",  # Include worst value.
            "Worst Run",  # Include worst run.
        )
        for statistic_name in statistic_names:  # Add statistic columns in fixed order.
            header.append(f"{metric_label} {statistic_name}")  # Add the metric statistic column.
    return header  # Return complete header.


def format_summary_value(value: Any) -> Any:
    """
    Format one summary field for deterministic CSV serialization.

    :param value: Raw value from cache rows or calculated statistics.
    :return: CSV-safe scalar value.
    """

    if value is None:  # Preserve project-style blank cells for missing values.
        return ""  # Return empty CSV value.
    if isinstance(value, float) and not math.isfinite(value):  # Reject non-finite calculated output.
        return ""  # Return empty CSV value.
    return value  # Return original scalar value.


def build_summary_rows(observations_by_identity: Dict[tuple, dict]) -> List[dict]:
    """
    Build deterministic summary rows from grouped cache observations.

    :param observations_by_identity: Grouped cache observations by exact production identity.
    :return: Ordered Run_Metrics.csv rows.
    """

    rows = []  # Accumulate summary rows.
    for group in observations_by_identity.values():  # Traverse grouped experiment configurations.
        run_numbers = sorted(group["runs"])  # Sort contributing runs numerically.
        if len(run_numbers) < 2:  # Exclude configurations that appear in only one logical run.
            continue  # Move to next configuration.
        representative = group["representative"]  # Use the earliest deterministic row for configuration fields.
        row = {  # Copy configuration columns only.
            column: format_summary_value(representative.get(column, ""))  # Format one configuration field.
            for column in EXPERIMENT_CONFIGURATION_COLUMNS  # Preserve configured identity column order.
        }
        row["Run Count"] = len(run_numbers)  # Store distinct logical run count.
        row["Runs"] = ",".join(str(run_number) for run_number in run_numbers)  # Store sorted logical run numbers.
        for metric_column, metric_label, direction in AGGREGATED_METRICS:  # Calculate every configured metric block.
            metric_observations = [  # Keep valid finite values only.
                (run_number, value)  # Store run number and finite metric value.
                for run_number, result_row in group["run_rows"].items()  # Traverse one row per logical run.
                if (value := finite_number(result_row.get(metric_column, None))) is not None  # Reject invalid values.
            ]
            metric_statistics = calculate_metric_statistics(metric_observations, direction)  # Calculate statistics.
            for statistic_name, statistic_value in metric_statistics.items():  # Append statistic values.
                statistic_column = f"{metric_label} {statistic_name}"  # Build the output statistic column name.
                row[statistic_column] = format_summary_value(statistic_value)  # Store CSV-safe statistic value.
        rows.append(row)  # Store completed summary row.
    return sorted(  # Return deterministic row order.
        rows,  # Sort completed summary rows.
        key=lambda row: tuple(  # Sort by identity fields.
            str(row.get(column, "")) for column in EXPERIMENT_CONFIGURATION_COLUMNS  # Preserve identity order.
        ),
    )


def read_logical_cache_dataframe(
    record: dict,  # Receive logical cache run metadata.
    config: dict,  # Receive runtime configuration.
    read_validated_cache_file: Callable[..., Tuple[Any, dict]],  # Receive production cache reader.
    merge_valid_cache_snapshots: Callable[..., Tuple[Any, dict]],  # Receive production cache merger.
    cache_file_lock: Callable[..., Any],  # Receive production cache lock.
) -> Optional[Any]:
    """
    Read one logical cache run without modifying Run_N artifacts.

    :param record: Logical cache run record.
    :param config: Runtime configuration dictionary.
    :param read_validated_cache_file: Production cache CSV reader.
    :param merge_valid_cache_snapshots: Production in-memory cache merge function.
    :param cache_file_lock: Production cache lock context manager.
    :return: Prepared cache DataFrame, or None.
    """

    primary_path = record["primary"]  # Resolve primary run cache path.
    backup_path = Path(f"{primary_path}.bak")  # Resolve sibling backup path.
    run_number = int(record["run"])  # Resolve expected logical run number.
    primary_df = None  # Track valid primary rows.
    backup_df = None  # Track valid backup rows.
    with cache_file_lock(str(primary_path), exclusive=False):  # Coordinate with active cache writers read-only.
        if primary_path.is_file():  # Read primary cache when present.
            try:  # Isolate primary damage from backup recovery.
                primary_df, _ = read_validated_cache_file(  # Reuse production cache validation.
                    str(primary_path),  # Pass primary cache path.
                    config=config,  # Pass runtime configuration.
                    expected_experiment_run=run_number,  # Pass logical run number.
                )
            except Exception:  # Ignore invalid primary when backup can still recover rows.
                primary_df = None  # Preserve no valid primary rows.
        if backup_path.is_file():  # Read backup cache when present.
            try:  # Isolate backup damage from primary recovery.
                backup_df, _ = read_validated_cache_file(  # Reuse production backup validation.
                    str(backup_path),  # Pass backup cache path.
                    config=config,  # Pass runtime configuration.
                    expected_experiment_run=run_number,  # Pass logical run number.
                )
            except Exception:  # Ignore invalid backup when primary can still recover rows.
                backup_df = None  # Preserve no valid backup rows.
    if primary_df is not None and backup_df is not None:  # Merge divergent valid sources in memory only.
        merged_df, _ = merge_valid_cache_snapshots(  # Merge valid sources in memory only.
            primary_df,  # Pass valid primary rows.
            backup_df,  # Pass valid backup rows.
            config=config,  # Pass runtime configuration.
            expected_experiment_run=run_number,  # Pass logical run number.
            source_path=str(primary_path),  # Preserve source label for validation errors.
        )
        return merged_df  # Return production-deduplicated union without publishing.
    return primary_df if primary_df is not None else backup_df  # Return the valid source, if any.


def group_logical_cache_runs(
    cache_directory: Path,  # Receive Cache_Results directory.
    config: dict,  # Receive runtime configuration.
    build_cache_identity_from_row: Callable[[Any], tuple],  # Receive production identity function.
    read_validated_cache_file: Callable[..., Tuple[Any, dict]],  # Receive production cache reader.
    merge_valid_cache_snapshots: Callable[..., Tuple[Any, dict]],  # Receive production cache merger.
    cache_file_lock: Callable[..., Any],  # Receive production cache lock.
) -> Dict[tuple, dict]:
    """
    Group authoritative logical cache rows by production experiment identity.

    :param cache_directory: Cache_Results directory to scan.
    :param config: Runtime configuration dictionary.
    :param build_cache_identity_from_row: Production cache identity function.
    :param read_validated_cache_file: Production cache CSV reader.
    :param merge_valid_cache_snapshots: Production in-memory cache merge function.
    :param cache_file_lock: Production cache lock context manager.
    :return: Grouped observations by exact experiment identity.
    """

    grouped: Dict[tuple, dict] = {}  # Accumulate grouped observations.
    for record in discover_logical_cache_runs(cache_directory):  # Traverse one logical cache artifact per family.
        cache_df = read_logical_cache_dataframe(  # Load authoritative in-memory rows for this run.
            record,  # Pass logical run record.
            config,  # Pass runtime configuration.
            read_validated_cache_file,  # Pass production cache reader.
            merge_valid_cache_snapshots,  # Pass production cache merger.
            cache_file_lock,  # Pass production cache lock.
        )
        if cache_df is None:  # Ignore unrecoverable logical runs.
            continue  # Move to next logical run.
        for _, row in cache_df.iterrows():  # Traverse production-prepared rows in stored order.
            result_row = row.to_dict()  # Convert pandas row to normal mapping for identity and output.
            identity = (record["family"], build_cache_identity_from_row(result_row))  # Reuse production identity.
            group = grouped.setdefault(  # Create grouped observation record.
                identity,  # Use cache-family plus production identity.
                {"representative": result_row, "runs": set(), "run_rows": {}},  # Store group state.
            )
            run_number = int(record["run"])  # Resolve the current logical run number.
            if run_number < min(group["runs"], default=run_number):  # Prefer lowest-run representative.
                group["representative"] = result_row  # Store earlier representative row.
            group["runs"].add(run_number)  # Count each logical run once per identity.
            group["run_rows"][run_number] = result_row  # Store one authoritative observation per logical run.
    return grouped  # Return grouped authoritative observations.


def write_rows_atomically(output_path: Path, header: List[str], rows: List[dict]) -> None:
    """
    Write summary rows through a same-directory atomic replacement.

    :param output_path: Final Run_Metrics.csv path.
    :param header: Ordered CSV header.
    :param rows: Ordered summary rows.
    :return: None.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists.
    temporary_descriptor, temporary_path = tempfile.mkstemp(  # Stage summary beside destination.
        dir=str(output_path.parent),  # Use destination directory for atomic replacement.
        prefix=f".{output_path.name}.",  # Use hidden destination-specific temporary prefix.
        suffix=".tmp",  # Use existing temporary suffix style.
    )
    try:  # Serialize the complete summary before replacement.
        with os.fdopen(temporary_descriptor, "w", encoding="utf-8", newline="") as temporary_file:  # Own descriptor.
            writer = csv.DictWriter(temporary_file, fieldnames=header, extrasaction="ignore")  # Create CSV writer.
            writer.writeheader()  # Write header even when no rows qualify.
            for row in rows:  # Write rows in deterministic order.
                output_row = {  # Serialize known columns.
                    column: format_summary_value(row.get(column, "")) for column in header  # Format one field.
                }
                writer.writerow(output_row)  # Write one summary row.
            temporary_file.flush()  # Flush Python buffers before replacement.
            os.fsync(temporary_file.fileno())  # Synchronize staged bytes before replacement.
        os.replace(temporary_path, output_path)  # Atomically publish the completed summary.
        if os.name != "nt":  # Synchronize directory metadata on POSIX platforms.
            directory_descriptor = os.open(str(output_path.parent), os.O_RDONLY)  # Open destination directory.
            try:  # Synchronize directory entry.
                os.fsync(directory_descriptor)  # Flush rename metadata.
            finally:  # Close directory descriptor.
                os.close(directory_descriptor)  # Release directory descriptor.
    except Exception:  # Preserve original serialization or replacement failure.
        if os.path.exists(temporary_path):  # Remove incomplete staged summary.
            os.unlink(temporary_path)  # Delete the temporary summary file.
        raise  # Re-raise original failure.


def write_run_metrics_summary(
    cache_path: str,  # Receive any run-specific cache path in target directory.
    config: dict,  # Receive runtime configuration.
    build_cache_identity_from_row: Callable[[Any], tuple],  # Receive production identity function.
    read_validated_cache_file: Callable[..., Tuple[Any, dict]],  # Receive production cache reader.
    merge_valid_cache_snapshots: Callable[..., Tuple[Any, dict]],  # Receive production cache merger.
    cache_file_lock: Callable[..., Any],  # Receive production cache lock.
) -> dict:
    """
    Rebuild Run_Metrics.csv from authoritative logical cache runs.

    :param cache_path: Any run-specific cache path inside the target Cache_Results directory.
    :param config: Runtime configuration dictionary.
    :param build_cache_identity_from_row: Production cache identity function.
    :param read_validated_cache_file: Production cache CSV reader.
    :param merge_valid_cache_snapshots: Production in-memory cache merge function.
    :param cache_file_lock: Production cache lock context manager.
    :return: Summary metadata including path and row count.
    """

    cache_directory = Path(cache_path).resolve().parent  # Resolve Cache_Results directory from cache path.
    output_path = cache_directory / RUN_METRICS_FILENAME  # Resolve derived summary output path.
    grouped = group_logical_cache_runs(  # Group all recoverable logical runs.
        cache_directory,  # Pass Cache_Results directory.
        config,  # Pass runtime configuration.
        build_cache_identity_from_row,  # Pass production identity function.
        read_validated_cache_file,  # Pass production cache reader.
        merge_valid_cache_snapshots,  # Pass production cache merger.
        cache_file_lock,  # Pass production cache lock.
    )
    header = build_run_metrics_header()  # Build deterministic output schema.
    rows = build_summary_rows(grouped)  # Build summary rows from grouped observations.
    with cache_file_lock(str(output_path), exclusive=True):  # Serialize concurrent summary writers only.
        write_rows_atomically(output_path, header, rows)  # Publish derived summary atomically.
    logical_run_count = len(discover_logical_cache_runs(cache_directory))  # Count logical cache runs discovered.
    return {"path": str(output_path), "rows": len(rows), "logical_runs": logical_run_count}  # Return metadata.
