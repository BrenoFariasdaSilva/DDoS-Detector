"""Focused only-combination allowlist coverage."""

import unittest  # Provide the repository-standard focused test runner.
from unittest import mock  # Supply deterministic CLI arguments.

import stacking  # Exercise production CLI and merged configuration behavior.
from utils.skip_combinations import apply_only_combination_rules, apply_skip_combination_rules, build_alias_lookup, compile_only_combination_rules, compile_skip_combination_rules  # Exercise production rule parsing and filtering.


class OnlyCombinationTests(unittest.TestCase):
    """Verify allowlist selection and its ordering relative to skip rules."""

    def test_only_rules_select_union_before_skip_rules(self):
        """
        Verify repeated allowlist rules select a union before exclusions run.

        :return: None.
        """

        aliases = build_alias_lookup({"Full Features": ("full",), "PCA Components": ("pca",)}, {"Random Forest": ("random_forest",), "StackingClassifier": ("stacking",)}, {"Default Hyperparameters": ("default",), "Optimized Hyperparameters": ("optimized",)})  # Build the same dimension-aware aliases used by production.
        plan = [("Full Features", False, None, "Random Forest"), ("Full Features", False, None, "StackingClassifier"), ("PCA Components", True, 0.5, "StackingClassifier"), ("PCA Components", False, None, "Random Forest")]  # Build canonical combinations containing selected and excluded work.
        only_rules = compile_only_combination_rules(["full&random_forest&default&0", "pca&stacking&optimized&50"], "CLI", aliases)  # Compile two repeated allowlist rules as a union.
        selected, only_summary = apply_only_combination_rules(plan, only_rules, "CLI")  # Retain only combinations matching either allowlist rule.
        self.assertEqual(selected, [plan[0], plan[2]])  # Preserve canonical order without duplicate combinations.
        self.assertEqual((only_summary["generated"], only_summary["selected"], only_summary["excluded"]), (4, 2, 2))  # Report exact selection totals.
        skip_rules = compile_skip_combination_rules(["stacking"], "CLI", aliases)  # Compile one exclusion that overlaps the selected subset.
        eligible, _ = apply_skip_combination_rules(selected, skip_rules, "CLI")  # Apply skips after allowlist selection.
        self.assertEqual(eligible, [plan[0]])  # Require the exclusion to win inside the selected subset.

    def test_only_combination_cli_replaces_yaml_rules(self):
        """
        Verify repeated CLI allowlist rules replace configured YAML rules.

        :return: None.
        """

        with mock.patch("sys.argv", ["stacking.py", "--only-combination", "full&random_forest", "--only-combination", "pca&stacking"]):  # Supply two repeatable production CLI rules.
            cli_args = stacking.parse_cli_args()  # Parse the allowlist options through the public CLI.
        file_config = {"stacking": {"only_combinations": ["rfe&svm"]}}  # Provide a YAML rule that CLI must replace.
        config = stacking.merge_configs(stacking.get_default_config(), file_config, cli_args)  # Apply normal CLI-over-YAML precedence and compilation.
        self.assertEqual(config["stacking"]["only_combinations"], ["full&random_forest", "pca&stacking"])  # Preserve the exact repeated CLI rule order.
        self.assertEqual(config["stacking"]["only_combinations_source"], "CLI")  # Report CLI as the effective source.
        self.assertEqual([rule.canonical for rule in config["stacking"]["compiled_only_combinations"]], ["Full Features&Random Forest", "PCA Components&StackingClassifier"])  # Compile aliases to canonical runtime identities.


if __name__ == "__main__":
    unittest.main()  # Run the focused allowlist contract directly.
