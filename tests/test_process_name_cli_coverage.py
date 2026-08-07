"""Process-name CLI coverage for executable pipeline scripts."""

import ast  # Inspect parser wiring without importing expensive pipeline dependencies.
from pathlib import Path  # Resolve repository script paths.
import unittest  # Provide the repository-standard focused test runner.


SCRIPT_NAMES = ("main.py", "genetic_algorithm.py", "pca.py", "dataset_converter.py", "hyperparameters_optimization.py", "dataset_descriptor.py", "rfe.py", "extratrees.py", "stacking.py", "wgangp.py")  # Define every process-name-enabled entry point.


class ProcessNameCliCoverageTests(unittest.TestCase):
    """Verify every executable entry point exposes and applies process naming."""

    def test_all_entry_points_expose_and_apply_process_name(self):
        """
        Verify every supported script parses and applies the process-name option.

        :return: None.
        """

        repository_root = Path(__file__).resolve().parents[1]  # Resolve the shared repository root from this focused test.
        for script_name in SCRIPT_NAMES:  # Validate every requested executable consistently.
            source = (repository_root / script_name).read_text(encoding="utf-8")  # Load source without triggering pipeline imports or global side effects.
            syntax_tree = ast.parse(source, filename=script_name)  # Require syntactically valid Python before inspecting calls.
            process_option_found = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument" and any(isinstance(argument, ast.Constant) and argument.value == "--process-name" for argument in node.args) for node in ast.walk(syntax_tree))  # Locate the exact CLI option declaration.
            process_setter_found = any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "set_runtime_process_name" for node in ast.walk(syntax_tree))  # Locate the shared runtime naming call.
            self.assertTrue(process_option_found, f"{script_name} does not expose --process-name")  # Require the public CLI option.
            self.assertTrue(process_setter_found, f"{script_name} does not apply --process-name")  # Require the parsed value to affect the process title.


if __name__ == "__main__":
    unittest.main()  # Run the focused cross-script contract directly.
