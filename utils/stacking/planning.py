"""Pure evaluation-grid and feature-worker planning utilities."""

from typing import Any, List, Optional, Tuple  # Preserve public planning annotations.


FEATURE_SET_WORKER_KEYS = ("full", "ga", "pca", "rfe", "extra_trees")  # Define supported persistent worker identities.


def build_evaluation_plan(hp_runs: List[Tuple[bool, dict, dict]], augmentation_modes: List[Optional[float]], feature_mode_names: List[str], stacking_enabled: bool) -> List[Tuple[str, bool, Optional[float], str]]:
    """
    Build the exact ordered combinations represented by one evaluation progress bar.

    :param hp_runs: Ordered runnable hyperparameter modes and their model mappings.
    :param augmentation_modes: Ordered augmentation ratios with None representing original-only data.
    :param feature_mode_names: Ordered feature modes produced by the evaluation iterator.
    :param stacking_enabled: Whether the stacking classifier runs after individual classifiers.
    :return: Ordered tuples of feature set, hyperparameter mode, augmentation ratio, and classifier.
    """

    evaluation_plan = []  # Accumulate combinations in canonical global phase order.
    original_modes = [ratio for ratio in augmentation_modes if ratio is None]  # Preserve every configured original-testing mode before augmentation.
    augmented_ratios = [ratio for ratio in augmentation_modes if ratio is not None]  # Preserve configured augmented-ratio order.
    for feature_mode_name in feature_mode_names:  # Complete original experiments for each feature set in configured order.
        for hyperparameters_enabled, models_map, _ in hp_runs:  # Preserve default-first hyperparameter order for current feature set.
            classifier_names = list(models_map.keys()) + (["StackingClassifier"] if stacking_enabled else [])  # Preserve classifier order followed by optional stacking.
            for augmentation_ratio in original_modes:  # Keep original testing isolated in first global phase.
                for classifier_name in classifier_names:  # Preserve classifier order inside current original hyperparameter mode.
                    evaluation_plan.append((feature_mode_name, hyperparameters_enabled, augmentation_ratio, classifier_name))  # Store one original-data combination.
    for augmentation_ratio in augmented_ratios:  # Complete every combination for current ratio before next ratio.
        for feature_mode_name in feature_mode_names:  # Preserve configured feature-set order inside current ratio.
            for hyperparameters_enabled, models_map, _ in hp_runs:  # Preserve default-first hyperparameter order inside current ratio and feature set.
                classifier_names = list(models_map.keys()) + (["StackingClassifier"] if stacking_enabled else [])  # Preserve classifier order followed by optional stacking.
                for classifier_name in classifier_names:  # Preserve classifier order inside current augmented hyperparameter mode.
                    evaluation_plan.append((feature_mode_name, hyperparameters_enabled, augmentation_ratio, classifier_name))  # Store one ratio-grouped augmented combination.

    return evaluation_plan  # Return the authoritative ordered progress plan


def retain_stacking_classifier_plan(evaluation_plan: List[Tuple[str, bool, Optional[float], str]], stacking_only: bool) -> List[Tuple[str, bool, Optional[float], str]]:
    """
    Retain only stacking classifier combinations when stacking-only mode is active.

    :param evaluation_plan: Ordered runtime evaluation combinations.
    :param stacking_only: Whether only StackingClassifier combinations may run.
    :return: Original plan or its stacking-classifier-only subset.
    """

    if stacking_only:  # Restrict the runtime plan to the requested ensemble classifier.
        return [combination for combination in evaluation_plan if combination[3] == "StackingClassifier"]  # Preserve canonical order while removing every individual classifier.
    return evaluation_plan  # Preserve the normal full evaluation plan unchanged.


def resolve_feature_set_worker_key(feature_set_name: str) -> str:  # Resolve one runtime feature-set name to its configured process key
    """
    Resolve a runtime feature-set name to its configured process key.

    :param feature_set_name: Runtime feature-set display name.
    :return: Configured worker key for Full, GA, PCA, RFE, or Extra Trees.
    """

    key_by_name = {"Full Features": "full", "GA Features": "ga", "PCA Components": "pca", "RFE Features": "rfe", "Extra Trees Features": "extra_trees"}  # Map supported runtime identities to configuration keys
    if feature_set_name not in key_by_name:  # Reject unsupported persistent feature-set identities
        raise ValueError(f"Persistent feature-set processes support only Full Features, GA Features, PCA Components, RFE Features, and Extra Trees Features, not {feature_set_name}")  # Report the unsupported runtime identity
    return key_by_name[feature_set_name]  # Return the configured process key


def build_feature_process_metadata(feature_names: List[Any], ga_selected_features: Any, pca_n_components: Any, rfe_selected_features: Any, extra_trees_selected_features: Any = None) -> dict:  # Build ordered feature identities and indices without materializing matrices
    """
    Build feature-set metadata without materializing feature matrices.

    :param feature_names: Ordered numeric input feature names.
    :param ga_selected_features: Ordered GA-selected feature names.
    :param pca_n_components: Selected PCA component count.
    :param rfe_selected_features: Ordered RFE-selected feature names.
    :param extra_trees_selected_features: Ordered Extra Trees-selected feature names.
    :return: Mapping of runtime feature-set names to small metadata descriptors.
    """

    normalized_feature_names = [str(feature) for feature in feature_names]  # Normalize the input schema without changing order
    feature_index = {feature: index for index, feature in enumerate(normalized_feature_names)}  # Build one deterministic positional lookup
    ga_names = [str(feature) for feature in (ga_selected_features or []) if str(feature) in feature_index]  # Preserve valid GA feature order
    rfe_names = [str(feature) for feature in (rfe_selected_features or []) if str(feature) in feature_index]  # Preserve valid RFE feature order
    extra_trees_names = [str(feature) for feature in (extra_trees_selected_features or []) if str(feature) in feature_index]  # Preserve valid Extra Trees feature order
    pca_count = min(int(pca_n_components or 0), len(normalized_feature_names))  # Resolve the exact effective PCA component count
    return {  # Return small descriptors for every supported persistent feature set
        "Full Features": {"feature_names": normalized_feature_names, "indices": list(range(len(normalized_feature_names))), "feature_count": len(normalized_feature_names)},  # Describe the unchanged full input columns
        "GA Features": {"feature_names": ga_names, "indices": [feature_index[name] for name in ga_names], "feature_count": len(ga_names)},  # Describe GA input columns
        "PCA Components": {"feature_names": [f"PC{index + 1}" for index in range(pca_count)], "indices": None, "feature_count": pca_count},  # Describe PCA output components
        "RFE Features": {"feature_names": rfe_names, "indices": [feature_index[name] for name in rfe_names], "feature_count": len(rfe_names)},  # Describe RFE input columns
        "Extra Trees Features": {"feature_names": extra_trees_names, "indices": [feature_index[name] for name in extra_trees_names], "feature_count": len(extra_trees_names)},  # Describe Extra Trees input columns
    }  # Complete the supported descriptor mapping
