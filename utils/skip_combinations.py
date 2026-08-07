"""Parse, normalize, match, and report stacking skip-combination rules."""
if __name__ in {"__main__", "__mp_main__"}:
    try:
        from setproctitle import setproctitle
        setproctitle(f"DDoS-{__file__.rsplit('/', 1)[-1].rsplit('.', 1)[0]}")
    except ImportError:
        pass


import math  # Use finite-safe augmentation ratio normalization.
from typing import Any, List, NamedTuple, Optional, Tuple  # Define compact typed rule structures.


SKIP_RULE_FIELDS = ("feature_set", "classifier", "hyperparameters", "augmentation_ratio")  # Define predicate field order.
SKIP_RULE_FIELD_ALIASES = {"feature": "feature_set", "featureset": "feature_set", "classifier": "classifier", "model": "classifier", "hyperparameters": "hyperparameters", "hyperparametermode": "hyperparameters", "augmentation": "augmentation_ratio", "augmentationratio": "augmentation_ratio"}  # Map normalized field aliases.


class SkipCombinationRule(NamedTuple):  # Store one compiled skip expression immutably.
    canonical: str  # Store normalized user-readable expression.
    clauses: Tuple[Tuple[Tuple[str, Any], ...], ...]  # Store OR clauses containing AND predicates.


def normalize_skip_token(value: Any) -> str:  # Normalize user tokens for alias lookup.
    """
    Normalize a skip-rule token for exact alias matching.

    :param value: Raw token value.
    :return: Lowercase alphanumeric token.
    """

    return "".join(character.lower() for character in str(value) if character.isalnum())  # Strip separators and normalize case.


def build_alias_lookup(feature_aliases: dict, classifier_aliases: dict, hyperparameter_aliases: dict) -> dict:  # Build runtime alias lookup.
    """
    Build exact normalized skip-rule alias lookup.

    :param feature_aliases: Mapping of canonical feature names to aliases.
    :param classifier_aliases: Mapping of canonical classifier names to aliases.
    :param hyperparameter_aliases: Mapping of canonical hyperparameter labels to aliases.
    :return: Mapping of normalized tokens to candidate field and value pairs.
    """

    lookup = {}  # Accumulate aliases across dimensions.
    alias_sources = (("feature_set", feature_aliases), ("classifier", classifier_aliases), ("hyperparameters", hyperparameter_aliases))  # Define supported alias groups.
    for field, aliases_by_value in alias_sources:  # Register aliases dimension by dimension.
        for canonical_value, aliases in aliases_by_value.items():  # Register aliases for one canonical value.
            for alias in tuple(aliases) + (canonical_value,):  # Include canonical display text as an alias.
                lookup.setdefault(normalize_skip_token(alias), set()).add((field, canonical_value))  # Store exact normalized alias.
    return lookup  # Return complete lookup.


def normalize_skip_augmentation_ratio(value: Any, source: str) -> int:  # Normalize user-facing augmentation ratio.
    """
    Normalize a user-facing augmentation percentage.

    :param value: Raw augmentation token.
    :param source: User-facing rule location.
    :return: Integer percentage from 0 through 100.
    """

    token = str(value).strip()  # Normalize textual whitespace.
    if not token.isdigit():  # Require integer percentages only.
        raise ValueError(f"{source} augmentation ratio must be an integer from 0 to 100")  # Reject floats, signs, and text.
    ratio = int(token)  # Convert validated integer text.
    if ratio < 0 or ratio > 100:  # Enforce inclusive user-facing bounds.
        raise ValueError(f"{source} augmentation ratio must be from 0 to 100")  # Reject out-of-range values.
    return ratio  # Return canonical percentage.


def normalize_skip_rule_term(term: str, source: str, alias_lookup: dict) -> Tuple[str, Any]:  # Normalize one rule term.
    """
    Normalize one skip-rule term into a field predicate.

    :param term: Raw rule term.
    :param source: User-facing rule location.
    :param alias_lookup: Normalized alias lookup produced from runtime registries.
    :return: Canonical field and value pair.
    """

    if "=" in term:  # Parse explicit field-qualified syntax.
        field_text, value_text = (part.strip() for part in term.split("=", 1))  # Split one field assignment.
        field = SKIP_RULE_FIELD_ALIASES.get(normalize_skip_token(field_text))  # Resolve explicit field alias.
        if field is None:  # Reject unknown explicit fields.
            raise ValueError(f"{source} unknown skip-combination field: {field_text}")  # Report unsupported field name.
        if value_text == "":  # Reject missing assignment value.
            raise ValueError(f"{source} empty value for skip-combination field: {field_text}")  # Report empty explicit value.
        if field == "augmentation_ratio":  # Normalize augmentation percentages separately.
            return field, normalize_skip_augmentation_ratio(value_text, source)  # Return explicit ratio predicate.
        candidates = [candidate for candidate in alias_lookup.get(normalize_skip_token(value_text), set()) if candidate[0] == field]  # Resolve exact aliases for requested field.
        if not candidates:  # Reject unknown value for requested dimension.
            raise ValueError(f"{source} unknown {field} value: {value_text}")  # Report unsupported explicit value.
        return candidates[0]  # Return field-bound predicate.
    if term.strip().isdigit():  # Treat bare integer tokens as augmentation percentages.
        return "augmentation_ratio", normalize_skip_augmentation_ratio(term, source)  # Return shorthand ratio predicate.
    candidates = sorted(alias_lookup.get(normalize_skip_token(term), set()))  # Resolve exact shorthand aliases across dimensions.
    if not candidates:  # Reject unknown shorthand names.
        raise ValueError(f"{source} unknown skip-combination term: {term}")  # Report unsupported shorthand token.
    fields = {field for field, _ in candidates}  # Identify dimensions matched by shorthand token.
    if len(fields) > 1:  # Reject ambiguous shorthand tokens.
        raise ValueError(f"{source} ambiguous skip-combination term: {term}; use explicit field syntax")  # Ask for field-qualified syntax.
    return candidates[0]  # Return unambiguous shorthand predicate.


def parse_skip_combination_rule(rule_text: str, source: str, alias_lookup: dict) -> SkipCombinationRule:  # Parse one skip expression.
    """
    Parse one skip-combination expression.

    :param rule_text: Raw expression text.
    :param source: User-facing rule location.
    :param alias_lookup: Normalized alias lookup produced from runtime registries.
    :return: Compiled immutable skip rule.
    """

    if not isinstance(rule_text, str) or not rule_text.strip():  # Require non-empty strings from YAML or CLI.
        raise ValueError(f"{source} skip-combination rule must be a non-empty string")  # Reject invalid rule entries.
    expression = rule_text.strip()  # Trim outer whitespace without changing internal aliases.
    if any(symbol in expression for symbol in ("(", ")", "!")):  # Reject unsupported expression syntax explicitly.
        raise ValueError(f"{source} skip-combination rule does not support parentheses or negation: {expression}")  # Prevent silent misinterpretation.
    if "&&" in expression or "|" in expression.replace("||", ""):  # Reject malformed AND and single-pipe operators.
        raise ValueError(f"{source} malformed skip-combination operators: {expression}")  # Require only '&' and '||'.
    clauses = []  # Accumulate OR clauses.
    for clause_index, raw_clause in enumerate(expression.split("||"), start=1):  # Parse OR before AND by required precedence.
        clause_text = raw_clause.strip()  # Normalize clause whitespace.
        if not clause_text:  # Reject leading, trailing, or repeated OR operators.
            raise ValueError(f"{source} empty skip-combination clause in: {expression}")  # Report malformed clause.
        predicates = {}  # Store one value per dimension inside AND clause.
        for raw_term in clause_text.split("&"):  # Parse AND terms inside current clause.
            term = raw_term.strip()  # Normalize term whitespace.
            if not term:  # Reject leading, trailing, repeated, or empty AND terms.
                raise ValueError(f"{source} empty skip-combination term in: {expression}")  # Report malformed term.
            field, value = normalize_skip_rule_term(term, f"{source} clause {clause_index}", alias_lookup)  # Normalize term through exact aliases.
            if field in predicates and predicates[field] != value:  # Reject contradictory values in one AND clause.
                raise ValueError(f"{source} contradictory {field} values in: {expression}")  # Report impossible clause.
            predicates[field] = value  # Store or deduplicate same field-value predicate.
        clauses.append(tuple((field, predicates[field]) for field in SKIP_RULE_FIELDS if field in predicates))  # Store canonical AND clause.
    canonical = "||".join("&".join(str(value) for _, value in clause) for clause in clauses)  # Build stable display expression.
    return SkipCombinationRule(canonical, tuple(clauses))  # Return compiled immutable rule.


def compile_skip_combination_rules(raw_rules: Any, source: str, alias_lookup: dict) -> Tuple[SkipCombinationRule, ...]:  # Compile configured skip expressions.
    """
    Validate and compile skip-combination rules.

    :param raw_rules: Raw YAML, CLI, or default rule list.
    :param source: User-facing configuration source.
    :param alias_lookup: Normalized alias lookup produced from runtime registries.
    :return: Tuple of compiled skip rules.
    """

    if not isinstance(raw_rules, list):  # Require YAML and CLI lists.
        raise ValueError(f"{source} skip_combinations must be a list of strings")  # Reject scalar and mapping configuration.
    compiled_rules = []  # Accumulate parsed rules in configured order.
    for index, rule_text in enumerate(raw_rules, start=1):  # Validate every configured rule independently.
        if not isinstance(rule_text, str) or not rule_text.strip():  # Reject null, numbers, booleans, mappings, and empty strings.
            raise ValueError(f"{source} skip_combinations[{index}] must be a non-empty string")  # Report invalid list item.
        compiled_rules.append(parse_skip_combination_rule(rule_text, f"{source} skip_combinations[{index}]", alias_lookup))  # Parse this rule once.
    return tuple(compiled_rules)  # Return immutable compiled rules.


def compile_only_combination_rules(raw_rules: Any, source: str, alias_lookup: dict) -> Tuple[SkipCombinationRule, ...]:
    """
    Validate and compile only-combination rules using skip-rule syntax.

    :param raw_rules: Raw YAML, CLI, or default rule list.
    :param source: User-facing configuration source.
    :param alias_lookup: Normalized alias lookup produced from runtime registries.
    :return: Tuple of compiled allowlist rules.
    """

    if not isinstance(raw_rules, list):  # Require YAML and CLI lists.
        raise ValueError(f"{source} only_combinations must be a list of strings")  # Reject scalar and mapping configuration.
    compiled_rules = []  # Accumulate parsed allowlist rules in configured order.
    for index, rule_text in enumerate(raw_rules, start=1):  # Validate every configured rule independently.
        if not isinstance(rule_text, str) or not rule_text.strip():  # Reject invalid or empty entries.
            raise ValueError(f"{source} only_combinations[{index}] must be a non-empty string")  # Report the invalid list item.
        compiled_rules.append(parse_skip_combination_rule(rule_text, f"{source} only_combinations[{index}]", alias_lookup))  # Reuse the established exact combination-rule grammar.
    return tuple(compiled_rules)  # Return immutable compiled allowlist rules.


def normalize_plan_augmentation_ratio(value: Any) -> int:  # Normalize internal plan ratio.
    """
    Normalize an internal plan augmentation value to a user-facing percent.

    :param value: Internal augmentation ratio or original-data sentinel.
    :return: Integer percentage where 0 means augmentation off.
    """

    if value is None:  # Treat original-only combinations as augmentation off.
        return 0  # Return off percentage.
    ratio = float(value)  # Convert internal numeric ratio.
    if math.isclose(ratio, 0.0, rel_tol=0.0, abs_tol=1e-12):  # Treat internal zero as augmentation off.
        return 0  # Return off percentage.
    return int(round(ratio * 100.0))  # Convert internal fraction to nearest user-facing percent.


def build_skip_combination_metadata(feature_set_name: str, hyperparameters_enabled: bool, augmentation_ratio: Optional[float], classifier_name: str) -> dict:  # Build matcher metadata.
    """
    Build normalized metadata for skip-rule matching.

    :param feature_set_name: Runtime feature-set display name.
    :param hyperparameters_enabled: Whether optimized hyperparameters are active.
    :param augmentation_ratio: Internal augmentation ratio or None for original data.
    :param classifier_name: Runtime classifier display name.
    :return: Normalized combination metadata.
    """

    return {"feature_set": feature_set_name, "classifier": classifier_name, "hyperparameters": "Optimized Hyperparameters" if hyperparameters_enabled else "Default Hyperparameters", "augmentation_ratio": normalize_plan_augmentation_ratio(augmentation_ratio)}  # Return canonical matcher fields.


def skip_rule_matches_combination(rule: SkipCombinationRule, combination: dict) -> bool:  # Match one rule against one combination.
    """
    Match one compiled skip rule against one combination.

    :param rule: Compiled skip rule.
    :param combination: Canonical combination metadata.
    :return: True when any OR clause matches.
    """

    for clause in rule.clauses:  # Evaluate independent OR clauses.
        if all(combination[field] == value for field, value in clause):  # Require every AND predicate in clause.
            return True  # Report matched expression.
    return False  # Report no matching clause.


def describe_skip_combination(feature_set_name: str, hyperparameters_enabled: bool, augmentation_ratio: Optional[float], classifier_name: str, global_id: int, canonical_total: int, index: int, skipped_total: int, matched_rule_indexes: Tuple[int, ...], experiment_run: int = 1) -> str:  # Format one skipped-combination line.
    """
    Format one skipped-combination log line.

    :param feature_set_name: Runtime feature-set display name.
    :param hyperparameters_enabled: Whether optimized hyperparameters are active.
    :param augmentation_ratio: Internal augmentation ratio or None for original data.
    :param classifier_name: Runtime classifier display name.
    :param global_id: Stable global combination ID.
    :param canonical_total: Total canonical combinations before filtering.
    :param index: One-based skipped-combination display index.
    :param skipped_total: Unique skipped-combination count.
    :param matched_rule_indexes: One-based matching rule indexes.
    :param experiment_run: One-based experiment run index.
    :return: User-readable skipped-combination line.
    """

    hyperparameter_label = "Optimized Hyperparameters" if hyperparameters_enabled else "Default Hyperparameters"  # Resolve canonical hyperparameter label.
    testing_data_label = "Augmented Data" if augmentation_ratio is not None else "Original Data"  # Resolve testing source label.
    augmentation_state = "On" if augmentation_ratio is not None else "Off"  # Resolve augmentation state label.
    augmentation_percent = normalize_plan_augmentation_ratio(augmentation_ratio)  # Normalize ratio for display.
    matched_rules = ",".join(str(rule_index) for rule_index in matched_rule_indexes)  # Format overlapping rule indexes deterministically.
    return f"[SKIPPED {index}/{skipped_total} | Global ID {global_id}/{canonical_total}] Experiment Run: {experiment_run} | Feature Set: {feature_set_name} | Classifier: {classifier_name} | Hyperparameters: {hyperparameter_label} | Training Data: Original Data | Testing Data: {testing_data_label} | Data Augmentation: {augmentation_state} | Augmentation Ratio: {augmentation_percent} | Matched Rules: {matched_rules}"  # Return complete skipped combination line.


def apply_skip_combination_rules(evaluation_plan: List[Tuple[str, bool, Optional[float], str]], rules: Tuple[SkipCombinationRule, ...], source: str, experiment_run: int = 1) -> Tuple[List[Tuple[str, bool, Optional[float], str]], dict]:  # Filter one canonical plan.
    """
    Filter an evaluation plan with compiled skip-combination rules.

    :param evaluation_plan: Canonical unfiltered evaluation plan.
    :param rules: Compiled skip rules.
    :param source: Resolved skip-rule source label.
    :param experiment_run: One-based experiment run index.
    :return: Eligible plan and skip summary metadata.
    """

    if not rules:  # Preserve original plan when no rules are configured.
        return list(evaluation_plan), {"canonical_total": len(evaluation_plan), "skipped": 0, "eligible": len(evaluation_plan), "global_ids": {combo: index for index, combo in enumerate(evaluation_plan, start=1)}, "rule_match_counts": [], "skipped_combinations": [], "source": source, "rules": rules}  # Return no-op summary.
    skipped_records = []  # Track unique skipped combinations with matching rule references.
    rule_match_counts = [0 for _ in rules]  # Track per-rule generated-plan matches.
    eligible_plan = []  # Accumulate unskipped combinations in canonical order.
    global_ids = {}  # Preserve original global IDs for eligible combinations.
    for global_index, combination in enumerate(evaluation_plan, start=1):  # Visit canonical combinations in original order.
        metadata = build_skip_combination_metadata(combination[0], combination[1], combination[2], combination[3])  # Build normalized matcher metadata.
        matched_rule_indexes = tuple(index + 1 for index, rule in enumerate(rules) if skip_rule_matches_combination(rule, metadata))  # Evaluate every rule without reparsing text.
        if matched_rule_indexes:  # Skip combinations matched by one or more rules.
            for rule_index in matched_rule_indexes:  # Count per-rule matches independently.
                rule_match_counts[rule_index - 1] += 1  # Increment matching rule count.
            skipped_records.append({"global_id": global_index, "combination": combination, "matched_rule_indexes": matched_rule_indexes, "matched_rules": tuple(rules[index - 1].canonical for index in matched_rule_indexes), "line": None})  # Store one unique skipped combination.
            continue  # Do not add skipped combination to eligible work.
        eligible_plan.append(combination)  # Preserve eligible canonical order.
        global_ids[combination] = global_index  # Store original stable global identity.
    skipped_total = len(skipped_records)  # Count unique skipped combinations once.
    for record_index, record in enumerate(skipped_records, start=1):  # Add deterministic display line after total is known.
        feature_set_name, hyperparameters_enabled, augmentation_ratio, classifier_name = record["combination"]  # Unpack canonical plan tuple.
        record["line"] = describe_skip_combination(feature_set_name, hyperparameters_enabled, augmentation_ratio, classifier_name, record["global_id"], len(evaluation_plan), record_index, skipped_total, record["matched_rule_indexes"], experiment_run)  # Store final skipped line.
    return eligible_plan, {"canonical_total": len(evaluation_plan), "skipped": skipped_total, "eligible": len(eligible_plan), "global_ids": global_ids, "rule_match_counts": rule_match_counts, "skipped_combinations": skipped_records, "source": source, "rules": rules}  # Return filtered plan and diagnostics.


def apply_only_combination_rules(evaluation_plan: List[Tuple[str, bool, Optional[float], str]], rules: Tuple[SkipCombinationRule, ...], source: str) -> Tuple[List[Tuple[str, bool, Optional[float], str]], dict]:
    """
    Retain combinations matched by at least one compiled allowlist rule.

    :param evaluation_plan: Canonical unfiltered evaluation plan.
    :param rules: Compiled only-combination rules.
    :param source: Resolved only-rule source label.
    :return: Selected plan and allowlist summary metadata.
    """

    if not rules:  # Preserve the complete plan when no allowlist is configured.
        return list(evaluation_plan), {"generated": len(evaluation_plan), "selected": len(evaluation_plan), "excluded": 0, "rule_match_counts": [], "source": source, "rules": rules}  # Return a no-op summary.
    selected_plan = []  # Accumulate allowlisted combinations in canonical order.
    rule_match_counts = [0 for _ in rules]  # Track generated-plan matches per rule.
    for combination in evaluation_plan:  # Visit every generated combination once.
        metadata = build_skip_combination_metadata(combination[0], combination[1], combination[2], combination[3])  # Build normalized matcher metadata.
        matched_rule_indexes = tuple(index for index, rule in enumerate(rules) if skip_rule_matches_combination(rule, metadata))  # Match every allowlist rule without reparsing.
        for rule_index in matched_rule_indexes:  # Count overlapping rule matches independently.
            rule_match_counts[rule_index] += 1  # Increment the matching allowlist rule count.
        if matched_rule_indexes:  # Retain a combination matched by any rule.
            selected_plan.append(combination)  # Preserve canonical execution order and avoid duplicates.
    return selected_plan, {"generated": len(evaluation_plan), "selected": len(selected_plan), "excluded": len(evaluation_plan) - len(selected_plan), "rule_match_counts": rule_match_counts, "source": source, "rules": rules}  # Return selected combinations and diagnostics.


def format_skip_rules_for_info(rules: Tuple[SkipCombinationRule, ...], source: str) -> List[str]:  # Format startup INFO skip lines.
    """
    Format resolved skip rules for startup INFO logs.

    :param rules: Compiled skip rules.
    :param source: Resolved skip-rule source label.
    :return: User-readable INFO message bodies.
    """

    if not rules:  # Report disabled state concisely.
        return ["Skip-combination rules: Disabled"]  # Return disabled line.
    lines = [f"Skip-combination rules source: {source}", f"Skip-combination rules: {len(rules)}"]  # Start with source and count.
    lines.extend(f"Skip rule {index}: {rule.canonical}" for index, rule in enumerate(rules, start=1))  # Append canonical rules.
    return lines  # Return complete startup INFO lines.


def format_only_rules_for_info(rules: Tuple[SkipCombinationRule, ...], source: str) -> List[str]:
    """
    Format resolved only-combination rules for startup logs.

    :param rules: Compiled only-combination rules.
    :param source: Resolved only-rule source label.
    :return: User-readable startup message bodies.
    """

    if not rules:  # Report disabled state concisely.
        return ["Only-combination rules: Disabled"]  # Return the disabled line.
    lines = [f"Only-combination rules source: {source}", f"Only-combination rules: {len(rules)}"]  # Start with source and count.
    lines.extend(f"Only rule {index}: {rule.canonical}" for index, rule in enumerate(rules, start=1))  # Append canonical allowlist rules.
    return lines  # Return complete startup lines.


def format_only_summary_line(only_summary: dict) -> str:
    """
    Format aggregate only-combination selection totals.

    :param only_summary: Summary returned by apply_only_combination_rules.
    :return: User-readable aggregate allowlist summary.
    """

    return f"[ONLY SUMMARY] Generated={only_summary['generated']} | Selected={only_summary['selected']} | Excluded={only_summary['excluded']}"  # Return complete allowlist totals.


def format_skip_rule_match_lines(skip_summary: dict) -> List[str]:  # Format per-rule match counts.
    """
    Format per-rule match-count lines.

    :param skip_summary: Skip summary returned by apply_skip_combination_rules.
    :return: User-readable per-rule match count lines.
    """

    rules = tuple(skip_summary.get("rules", ()))  # Read compiled rules from summary.
    counts = list(skip_summary.get("rule_match_counts", []))  # Read per-rule match counts from summary.
    return [f"[SKIP RULE {index}] {rule.canonical} | Matched={counts[index - 1] if index - 1 < len(counts) else 0}" for index, rule in enumerate(rules, start=1)]  # Return deterministic count lines.


def format_skip_summary_line(skip_summary: dict, cached_count: Optional[int] = None, pending_count: Optional[int] = None) -> str:  # Format aggregate skip totals.
    """
    Format aggregate skip summary line.

    :param skip_summary: Skip summary returned by apply_skip_combination_rules.
    :param cached_count: Optional recovered eligible count.
    :param pending_count: Optional pending eligible count.
    :return: User-readable aggregate skip summary line.
    """

    base = f"[SKIP SUMMARY] Canonical={skip_summary['canonical_total']} | Skipped={skip_summary['skipped']} | Eligible={skip_summary['eligible']}"  # Build canonical aggregate fields.
    if cached_count is None or pending_count is None:  # Return pre-cache summary when cache data is absent.
        return base  # Return canonical, skipped, and eligible totals only.
    return f"{base} | Cached={cached_count} | Pending={pending_count}"  # Return cache-aware aggregate totals.


def format_skip_rules_for_telegram(skip_summary: dict, cached_count: int, pending_count: int) -> List[str]:  # Format Telegram skip fields.
    """
    Format skip-rule fields for Telegram evaluation-plan summary.

    :param skip_summary: Skip summary returned by apply_skip_combination_rules.
    :param cached_count: Recovered eligible combination count.
    :param pending_count: Pending eligible combination count.
    :return: User-readable Telegram lines.
    """

    rules = tuple(skip_summary.get("rules", ()))  # Read compiled rules from summary.
    if not rules:  # Report disabled skip rules in Telegram.
        rule_lines = ["Skip rules: Disabled"]  # Build disabled line.
    else:  # Report source, count, and canonical rules.
        rule_lines = [f"Skip-rule source: {skip_summary.get('source', 'Default')}", f"Skip rules: {len(rules)}"]  # Build source and count lines.
        rule_lines.extend(f"Skip rule {index}: {rule.canonical}" for index, rule in enumerate(rules, start=1))  # Add canonical rule lines.
    totals = [f"Canonical combinations: {skip_summary['canonical_total']}", f"Skipped by rules: {skip_summary['skipped']}", f"Eligible combinations: {skip_summary['eligible']}", f"Recovered eligible combinations: {cached_count}", f"Pending eligible combinations: {pending_count}"]  # Build explicit totals.
    return rule_lines + totals  # Return complete Telegram skip section.


def filter_models_for_plan_group(models_map: dict, evaluation_plan: List[Tuple[str, bool, Optional[float], str]], feature_mode_name: str, hyperparameters_enabled: bool, augmentation_ratio: Optional[float]) -> Tuple[dict, bool, List[str]]:  # Select eligible models for one group.
    """
    Select classifiers eligible for one feature, hyperparameter, and augmentation group.

    :param models_map: Runtime model mapping for active hyperparameter mode.
    :param evaluation_plan: Skip-filtered eligible plan.
    :param feature_mode_name: Current feature-set name.
    :param hyperparameters_enabled: Current hyperparameter mode.
    :param augmentation_ratio: Current augmentation ratio.
    :return: Filtered model map, stacking flag, and planned classifier names.
    """

    planned_names = [classifier for feature_set, hp_enabled, ratio, classifier in evaluation_plan if feature_set == feature_mode_name and hp_enabled == hyperparameters_enabled and ratio == augmentation_ratio]  # Preserve planned classifier order for group.
    filtered_models = {name: models_map[name] for name in planned_names if name in models_map}  # Keep only individual classifiers in eligible group.
    stacking_planned = "StackingClassifier" in planned_names  # Resolve whether stacking remains eligible for group.
    return filtered_models, stacking_planned, planned_names  # Return model subset and exact planned names.
