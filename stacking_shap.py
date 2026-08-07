"""SHAP shape normalization, explainer selection, and progress utilities."""

import sys  # Route SHAP progress through the active process output stream.
from typing import Any, Callable, Optional, Tuple  # Preserve public progress-target annotations.

import numpy as np  # Preserve SHAP array normalization and deterministic sampling.
import shap  # Build the same SHAP explainers used by the orchestration layer.

from telegram_bot import send_exception_via_telegram  # Preserve explainability failure notifications.
from training_progress import interactive_terminal_attached  # Preserve interactive progress rendering behavior.


def sample_shap_test_data(X_test, y_test, max_samples, random_state):
    """
    Samples test data for SHAP computation when the test set exceeds the maximum sample size.

    :param X_test: Test features array to sample from
    :param y_test: Test labels array or Series to sample from
    :param max_samples: Maximum number of samples to use for SHAP computation
    :param random_state: Random seed for reproducible sampling
    :return: Tuple (X_test_sampled, y_test_sampled) with sampled or full test data
    """

    try:
        if len(X_test) > max_samples:  # If test set exceeds the sample limit
            rng = np.random.default_rng(random_state)  # Create explicit RNG to avoid using global RNG
            sample_indices = rng.choice(len(X_test), size=max_samples, replace=False)  # Draw reproducible sample indices from RNG
            if hasattr(X_test, 'iloc') and hasattr(X_test, 'iloc'):  # Verify pandas DataFrame/Series slicing is applicable
                X_test_sampled = X_test.iloc[sample_indices]  # Slice test features via iloc for DataFrame/Series
            else:  # Fallback to numpy-style indexing for arrays
                X_test_sampled = X_test[sample_indices]  # Slice test features via numpy indexing
            if hasattr(y_test, 'iloc'):  # If y_test is a pandas Series
                y_test_sampled = y_test.iloc[sample_indices]  # Slice labels via iloc for pandas Series
            else:  # If y_test is numpy array-like
                y_test_sampled = y_test[sample_indices]  # Slice labels via numpy indexing for arrays
        else:  # Test set is within the sample limit
            X_test_sampled = X_test  # Use full test features without sampling
            y_test_sampled = y_test  # Use full test labels without sampling
        return (X_test_sampled, y_test_sampled)  # Return the sampled or full test data tuple
    except Exception as e:  # Handle unexpected errors
        print(str(e))  # Print the exception string for diagnostics
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception details via Telegram if configured
        raise  # Re-raise the exception to preserve original behavior


def describe_raw_shap_result(shap_values):
    """
    Describe a raw SHAP result without changing its values.

    :param shap_values: Raw value returned by a SHAP explainer.
    :return: Tuple containing the qualified result type and shape metadata.
    """

    raw_type = f"{type(shap_values).__module__}.{type(shap_values).__name__}"  # Preserve the concrete SHAP return representation in diagnostics.
    if isinstance(shap_values, list):  # Historical SHAP versions return one samples-by-features array per output.
        item_shapes = []  # Preserve every per-output shape instead of coercing a heterogeneous list to object dtype.
        for item in shap_values:  # Inspect each output independently.
            item_value = item.values if isinstance(item, shap.Explanation) else item  # Extract numeric values from Explanation list items when present.
            item_shapes.append(tuple(np.asarray(item_value).shape))  # Record this output's exact array shape.
        return raw_type, item_shapes  # Return list-aware shape metadata.
    raw_value = shap_values.values if isinstance(shap_values, shap.Explanation) else shap_values  # Extract the numeric payload from modern Explanation results.
    return raw_type, tuple(np.asarray(raw_value).shape)  # Return the raw numeric shape for ndarray-like results.


def resolve_model_class_count(model):
    """
    Resolve the fitted classifier class count when the estimator exposes one.

    :param model: Fitted classifier being explained.
    :return: Number of fitted classes, or None when the estimator does not expose a single class axis.
    """

    classes = getattr(model, "classes_", None)  # Prefer the fitted class ordering used by classifier output columns.
    if classes is not None:  # Resolve ordinary single-target classifiers first.
        classes_array = np.asarray(classes)  # Normalize sklearn and third-party class containers.
        if classes_array.ndim == 1:  # A one-dimensional classes_ value maps directly to classifier outputs.
            return int(classes_array.shape[0])  # Return the exact fitted class count.
    n_classes = getattr(model, "n_classes_", None)  # Fall back to estimators such as XGBoost that expose n_classes_.
    if n_classes is not None and np.isscalar(n_classes):  # Accept only a single unambiguous class count.
        return int(n_classes)  # Return the estimator-provided class count.
    return None  # Leave multi-target or opaque estimator output counts unresolved.


def normalize_shap_output(shap_values, n_samples, n_features, n_model_outputs=None):
    """
    Normalize supported SHAP return representations for scientifically consistent plotting.

    The lossless internal representation is ``(outputs, samples, features)``. A
    single output keeps its signed SHAP values for plotting. Multiple outputs are
    reduced to ``(samples, features)`` with an equal-weight mean of absolute SHAP
    magnitudes across every output. This macro aggregation preserves every class
    or output without mixing sample and feature axes or privileging class support.

    :param shap_values: ndarray, list of arrays, or shap.Explanation returned by SHAP.
    :param n_samples: Expected number of sampled rows.
    :param n_features: Expected number of model input features.
    :param n_model_outputs: Optional fitted class/output count used only to disambiguate equal-sized axes.
    :return: Dictionary containing lossless output values, normalized plot values, and shape metadata.
    """

    expected_samples = int(n_samples)  # Normalize expected dimensions for exact comparisons.
    expected_features = int(n_features)  # Normalize expected dimensions for exact comparisons.
    if expected_samples <= 0 or expected_features <= 0:  # Reject empty matrices before any SHAP plotting path.
        raise ValueError(f"SHAP normalization requires positive sample and feature counts; received samples={expected_samples}, features={expected_features}")

    raw_type, raw_shape = describe_raw_shap_result(shap_values)  # Capture untouched representation metadata for mandatory diagnostics.
    if isinstance(shap_values, list):  # Normalize legacy list-of-output arrays.
        if not shap_values:  # An empty output list cannot be aligned scientifically.
            raise ValueError("SHAP normalization received an empty list of output arrays")
        output_arrays = []  # Accumulate one exact samples-by-features matrix per output.
        for output_index, item in enumerate(shap_values):  # Validate every class/output independently.
            item_value = item.values if isinstance(item, shap.Explanation) else item  # Support Explanation objects inside legacy-style lists.
            item_array = np.asarray(item_value)  # Normalize numeric array access without flattening axes.
            if item_array.ndim != 2 or item_array.shape != (expected_samples, expected_features):  # Require documented list item layout.
                raise ValueError(
                    f"SHAP output list item {output_index} has shape {item_array.shape}; expected ({expected_samples}, {expected_features}) as (samples, features)"
                )
            output_arrays.append(item_array)  # Preserve output order exactly as returned by the explainer.
        values_by_output = np.stack(output_arrays, axis=0)  # Build canonical (outputs, samples, features) representation.
    else:  # Normalize ndarray and modern Explanation payloads.
        raw_value = shap_values.values if isinstance(shap_values, shap.Explanation) else shap_values  # Read Explanation values without discarding metadata from the caller's raw result.
        raw_array = np.asarray(raw_value)  # Normalize array access while preserving dimensionality.
        if raw_array.ndim == 2:  # Single-output SHAP representation.
            if raw_array.shape != (expected_samples, expected_features):  # Never guess or transpose an undocumented two-dimensional layout.
                raise ValueError(
                    f"Two-dimensional SHAP values have shape {raw_array.shape}; expected ({expected_samples}, {expected_features}) as (samples, features)"
                )
            values_by_output = raw_array[np.newaxis, :, :]  # Add a lossless one-output axis.
        elif raw_array.ndim == 3:  # Multi-output SHAP representations vary by SHAP version and explainer.
            axis_candidates = []  # Accumulate every axis assignment consistent with sampled rows and features.
            for sample_axis in range(3):  # Consider every possible sample axis.
                if raw_array.shape[sample_axis] != expected_samples:  # Require exact sampled-row alignment.
                    continue
                for feature_axis in range(3):  # Consider every remaining feature axis.
                    if feature_axis == sample_axis or raw_array.shape[feature_axis] != expected_features:  # Require a distinct exact feature axis.
                        continue
                    output_axis = next(axis for axis in range(3) if axis not in (sample_axis, feature_axis))  # The remaining axis is the output axis.
                    axis_candidates.append((output_axis, sample_axis, feature_axis))  # Store canonical transpose order.
            axis_candidates = list(dict.fromkeys(axis_candidates))  # Remove duplicate assignments without changing deterministic order.
            if n_model_outputs is not None:  # Use fitted class count only when it can resolve an otherwise ambiguous layout.
                class_matched_candidates = [candidate for candidate in axis_candidates if raw_array.shape[candidate[0]] == int(n_model_outputs)]  # Find candidates whose output axis matches model classes.
                if class_matched_candidates:  # Prefer class-consistent candidates but allow one-output model APIs when no class-sized axis exists.
                    axis_candidates = class_matched_candidates  # Narrow candidates using estimator evidence.
            if len(axis_candidates) != 1:  # Refuse ambiguous or incompatible layouts instead of silently selecting an axis.
                raise ValueError(
                    f"Cannot unambiguously align three-dimensional SHAP shape {raw_array.shape} to samples={expected_samples}, features={expected_features}, model_outputs={n_model_outputs}; candidates={axis_candidates}"
                )
            values_by_output = np.transpose(raw_array, axis_candidates[0])  # Reorder axes losslessly to (outputs, samples, features).
        else:  # Current repository explainers do not produce scientifically plottable values with other ranks.
            raise ValueError(
                f"Unsupported SHAP value rank {raw_array.ndim} for shape {raw_array.shape}; expected a 2D single-output or 3D multi-output result"
            )

    if values_by_output.ndim != 3 or values_by_output.shape[1:] != (expected_samples, expected_features):  # Assert the canonical contract before aggregation.
        raise ValueError(
            f"Canonical SHAP alignment failed: shape={values_by_output.shape}, expected=(outputs, {expected_samples}, {expected_features})"
        )
    if values_by_output.shape[0] == 1:  # Preserve signed direction for genuine single-output explanations.
        normalized_values = values_by_output[0]  # Remove only the singleton output axis.
        aggregation = "single_output_signed"  # Document the exact plotting behavior.
    else:  # Build one class-agnostic global view without dropping any output.
        normalized_values = np.mean(np.abs(values_by_output), axis=0)  # Macro-average absolute magnitude over every class/output.
        aggregation = "macro_mean_absolute_across_all_outputs"  # Document equal-weight scientific aggregation.
    if normalized_values.shape != (expected_samples, expected_features):  # Enforce the public plot invariant after aggregation.
        raise ValueError(
            f"Normalized SHAP values have shape {normalized_values.shape}; expected ({expected_samples}, {expected_features})"
        )
    return {  # Return both lossless and plot-specific representations with auditable metadata.
        "raw_type": raw_type,
        "raw_shape": raw_shape,
        "values_by_output": values_by_output,
        "normalized_values": normalized_values,
        "normalized_shape": tuple(normalized_values.shape),
        "output_count": int(values_by_output.shape[0]),
        "aggregation": aggregation,
    }


def aggregate_mean_shap_importance(shap_values_summary, feature_names):
    """
    Computes mean absolute SHAP values and maps them to feature names.

    :param shap_values_summary: SHAP values array (2D: samples x features)
    :param feature_names: List of feature names corresponding to the SHAP value columns
    :return: Dictionary mapping each feature name to its mean absolute SHAP importance
    """

    try:
        shap_array = np.asarray(shap_values_summary)  # Convert normalized SHAP values to numpy array for consistent operations
        if shap_array.ndim != 2:  # Downstream importance requires the same samples-by-features contract as plotting.
            raise ValueError(f"SHAP importance requires a 2D (samples, features) array; received shape {shap_array.shape}")
        if not isinstance(feature_names, (list, tuple)) or shap_array.shape[1] != len(feature_names):  # Refuse index-based fallback that would lose feature identity.
            names_count = len(feature_names) if hasattr(feature_names, "__len__") else None  # Preserve mismatch evidence in the technical error.
            raise ValueError(f"SHAP importance feature mismatch: values have {shap_array.shape[1]} columns but feature names have {names_count} entries")
        mean_shap_values = np.mean(np.abs(shap_array), axis=0)  # Compute mean absolute SHAP value per feature across samples
        mean_shap_list = mean_shap_values.tolist() if hasattr(mean_shap_values, 'tolist') else list(mean_shap_values)  # Convert numpy array to plain Python list
        shap_importance = dict(zip(feature_names, mean_shap_list))  # Map the already-proven feature axis to its exact ordered names.
        return shap_importance  # Return importance dictionary for downstream use
    except Exception as e:  # Handle unexpected errors
        print(str(e))  # Print the exception string for diagnostics
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception details via Telegram if configured
        raise  # Re-raise the exception to preserve original behavior


def supports_predict_proba(model):  # Verify model supports predict_proba at module level
    """
    Verify if model supports predict_proba.

    :param model: Model instance.
    :return: Boolean indicating support.
    """

    return hasattr(model, "predict_proba")  # Verify presence of predict_proba method


def get_shap_prediction_function(model):  # Resolve prediction function for SHAP/LIME at module level
    """
    Resolve prediction function for SHAP based on model capabilities.

    :param model: Model instance.
    :return: Callable prediction function.
    """

    if supports_predict_proba(model):  # Verify if model supports predict_proba
        return model.predict_proba  # Use probability predictions when available

    return model.predict  # Fallback to class prediction when probabilities unavailable


def build_kernel_explainer(model, X_test_sampled, random_state):
    """
    Build a SHAP KernelExplainer with a bounded, reproducible background sample.

    :param model: Trained model object.
    :param X_test_sampled: Sampled test features used to derive background data.
    :param random_state: Random seed used for deterministic background sampling.
    :return: Instantiated shap.KernelExplainer.
    """

    rng = np.random.default_rng(random_state)  # Create explicit RNG for deterministic background sampling
    bkg_size = min(50, len(X_test_sampled)) if hasattr(X_test_sampled, "__len__") else 50  # Determine background sample size defensively
    indices = rng.choice(len(X_test_sampled), size=bkg_size, replace=False)  # Draw reproducible background indices
    if hasattr(X_test_sampled, "iloc"):  # If sampled data is a pandas object
        background = X_test_sampled.iloc[indices]  # Slice background via iloc
    else:  # Otherwise assume numpy-like array
        background = X_test_sampled[indices]  # Slice background via numpy indexing
    prediction_fn = get_shap_prediction_function(model)  # Resolve SHAP-compatible prediction callable
    return shap.KernelExplainer(prediction_fn, background)  # Build and return KernelExplainer


def select_shap_explainer(model, X_test_sampled, random_state):
    """
    Selects and instantiates the appropriate SHAP explainer based on model type.

    :param model: Trained model object for which to build the explainer
    :param X_test_sampled: Sampled test features used for KernelExplainer background data
    :param random_state: Random seed used for KernelExplainer background sampling
    :return: Instantiated SHAP explainer object
    """

    try:
        model_type = model.__class__.__name__  # Get model class name for branch selection
        n_classes = getattr(model, "n_classes_", None)  # Resolve fitted class count when available
        if model_type in ["RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier", "LightGBMClassifier", "ExtraTreesClassifier"]:  # Tree-based models
            if model_type == "GradientBoostingClassifier" and n_classes is not None and int(n_classes) > 2:  # SHAP TreeExplainer does not support multiclass GradientBoostingClassifier
                return build_kernel_explainer(model, X_test_sampled, random_state)  # Fallback to model-agnostic KernelExplainer for multiclass GB
            try:  # Try fast tree explainer first for supported tree models
                return shap.TreeExplainer(model)  # Use TreeExplainer for supported tree-based models
            except Exception as e:  # If SHAP tree path fails for a known unsupported case
                err = str(e).lower()  # Normalize exception string for safe matching
                if model_type == "GradientBoostingClassifier" and "only supported for binary classification" in err:  # Explicit SHAP multiclass GB limitation
                    return build_kernel_explainer(model, X_test_sampled, random_state)  # Fallback to KernelExplainer when SHAP rejects multiclass GB
                raise  # Re-raise unknown errors to preserve failure visibility
        elif model_type in ["LogisticRegression", "LinearSVC", "SGDClassifier"]:  # Linear models
            return shap.LinearExplainer(model, X_test_sampled)  # Use LinearExplainer for linear models
        else:  # Other models that require a fallback explainer
            return build_kernel_explainer(model, X_test_sampled, random_state)  # Use KernelExplainer with bounded deterministic background sampling
    except Exception as e:  # Handle unexpected errors
        print(str(e))  # Print the exception string for diagnostics
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception details via Telegram if configured
        raise  # Re-raise the exception to preserve original behavior


def build_shap_progress_description(model_name, dataset_name, explainer_name):
    """
    Build a concise contextual SHAP progress-bar description.

    :param model_name: Name of the model being explained.
    :param dataset_name: Name of the dataset being explained.
    :param explainer_name: Name of the SHAP explainer in use.
    :return: Context-rich progress-bar description string.
    """

    try:
        model_label = str(model_name) if model_name else "UnknownModel"  # Normalize model label for progress output.
        dataset_label = str(dataset_name) if dataset_name else "UnknownDataset"  # Normalize dataset label for progress output.
        explainer_label = str(explainer_name) if explainer_name else "SHAP"  # Normalize explainer label for progress output.
        return f"SHAP {explainer_label} | {model_label} | {dataset_label}"  # Return concise contextual description for SHAP progress output.
    except Exception as e:
        print(str(e))  # Print the exception string for diagnostics.
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception details via Telegram if configured.
        raise  # Re-raise the exception to preserve original behavior.

def resolve_shap_progress_target(shap_callable) -> Tuple[Optional[str], Optional[Any], Optional[str], Optional[Callable[..., Any]]]:
    """
    Resolve the runtime tqdm symbol used by a SHAP callable when exposed.

    :param shap_callable: Bound SHAP callable used to compute SHAP values.
    :return: Tuple describing the patch target and original tqdm callable, or empty values when unavailable.
    """

    try:
        method_globals = getattr(shap_callable, "__globals__", {})  # Access callable globals for runtime tqdm resolution.
        direct_tqdm = method_globals.get("tqdm", None)  # Resolve direct tqdm symbol from callable globals when present.
        if callable(direct_tqdm):  # Verify direct tqdm symbol is callable before using it.
            return ("globals_dict", method_globals, "tqdm", direct_tqdm)  # Return direct globals-based patch target and original tqdm callable.
        for value in method_globals.values():  # Iterate global values to locate module-style tqdm exposure when used by SHAP.
            module_name = getattr(value, "__name__", "") if value is not None else ""  # Resolve module-like name defensively for tqdm filtering.
            tqdm_attr = getattr(value, "tqdm", None) if value is not None else None  # Resolve nested tqdm attribute when a module wrapper is used.
            if callable(tqdm_attr) and "tqdm" in str(module_name).lower():  # Verify nested tqdm attribute belongs to a tqdm-related module.
                return ("module_attr", value, "tqdm", tqdm_attr)  # Return module-attribute patch target and original tqdm callable.
        return (None, None, None, None)  # Return empty patch metadata when SHAP does not expose a runtime tqdm hook.
    except Exception as e:
        print(str(e))  # Print the exception string for diagnostics.
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception details via Telegram if configured.
        raise  # Re-raise the exception to preserve original behavior.


def create_shap_progress_wrapper(tqdm_callable, progress_desc, progress_phase):
    """
    Create a tqdm wrapper that injects SHAP-specific context without changing iteration semantics.

    :param tqdm_callable: Original tqdm callable resolved from SHAP runtime globals.
    :param progress_desc: Description string to inject into the progress bar.
    :param progress_phase: Postfix string describing the current SHAP phase.
    :return: Wrapped tqdm callable.
    """

    def wrapped_tqdm(*args, **kwargs):
        kwargs.setdefault("desc", progress_desc)  # Inject contextual description only when SHAP did not already provide one.
        kwargs.setdefault("file", sys.stdout)  # Route progress output through the configured stdout logger.
        kwargs.setdefault("disable", not interactive_terminal_attached(sys.stdout))  # Render SHAP's interactive bar only on an attached terminal.
        progress_bar = tqdm_callable(*args, **kwargs)  # Delegate progress-bar construction to the original tqdm callable.
        if hasattr(progress_bar, "set_postfix_str") and progress_phase:  # Verify postfix support before appending SHAP phase metadata.
            progress_bar.set_postfix_str(progress_phase, refresh=False)  # Append concise SHAP phase metadata without forcing a redraw.
        return progress_bar  # Return the original tqdm instance with injected contextual metadata.

    return wrapped_tqdm  # Return wrapped tqdm callable for temporary SHAP runtime patching.


def compute_shap_values_with_context(explainer, X_test_for_shap, progress_desc, progress_phase):
    """
    Compute SHAP values while temporarily injecting contextual progress metadata when SHAP exposes tqdm.

    :param explainer: Instantiated SHAP explainer object.
    :param X_test_for_shap: Test feature matrix passed to SHAP.
    :param progress_desc: Description string for the SHAP progress bar.
    :param progress_phase: Postfix string describing the current SHAP phase.
    :return: SHAP values returned by the explainer.
    """

    try:
        patch_kind, patch_owner, patch_attr, original_tqdm = resolve_shap_progress_target(explainer.shap_values)  # Resolve SHAP runtime tqdm target when exposed by the installed version.
        if original_tqdm is None or patch_owner is None or patch_attr is None:  # Verify whether SHAP exposes a complete patchable tqdm hook before attempting runtime injection.
            return explainer.shap_values(X_test_for_shap)  # Compute SHAP values directly when no patchable tqdm hook exists.
        wrapped_tqdm = create_shap_progress_wrapper(original_tqdm, progress_desc, progress_phase)  # Build contextual tqdm wrapper around the original SHAP progress constructor.
        try:
            if patch_kind == "globals_dict":  # Verify whether SHAP uses a direct globals-based tqdm symbol.
                patch_owner[patch_attr] = wrapped_tqdm  # Replace SHAP globals-based tqdm symbol temporarily.
            else:
                setattr(patch_owner, patch_attr, wrapped_tqdm)  # Replace SHAP module-attribute tqdm symbol temporarily.
            return explainer.shap_values(X_test_for_shap)  # Compute SHAP values while the contextual tqdm wrapper is active.
        finally:
            if patch_kind == "globals_dict":  # Verify whether the patched tqdm symbol lives in SHAP callable globals.
                patch_owner[patch_attr] = original_tqdm  # Restore the original SHAP globals-based tqdm symbol after computation.
            else:
                setattr(patch_owner, patch_attr, original_tqdm)  # Restore the original SHAP module-attribute tqdm symbol after computation.
    except Exception as e:
        print(str(e))  # Print the exception string for diagnostics.
        send_exception_via_telegram(type(e), e, e.__traceback__)  # Send exception details via Telegram if configured.
        raise  # Re-raise the exception to preserve original behavior.
