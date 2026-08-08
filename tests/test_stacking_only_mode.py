"""Focused stacking-only pipeline mode coverage."""

import unittest  # Provide the repository-standard focused test runner.
from unittest import mock  # Observe CLI parsing and process-title application.

import stacking  # Exercise production configuration and planning behavior.


class StackingOnlyModeTests(unittest.TestCase):
    """Verify isolated stacking mode without fitting production classifiers."""

    def test_stacking_only_restricts_configuration_and_plan(self):
        """
        Verify stacking-only mode disables optional stages and individual classifiers.

        :return: None.
        """

        config = stacking.get_default_config()  # Build an independent production-default configuration.
        config["stacking"]["stacking_only"] = True  # Select isolated stacking through the config contract.
        config = stacking.merge_configs(config, {}, None)  # Apply the same normalization used by config.yaml execution.
        methods = config["stacking"]["methods"]  # Resolve normalized method toggles.
        self.assertEqual(methods, {"augmentation": False, "feature_selection": False, "hyperparameter_optimization": False, "automl": False, "stacking": True})  # Require exactly one active classifier method.
        self.assertFalse(config["execution"]["test_data_augmentation"])  # Require augmented testing to remain inactive.
        self.assertFalse(config["explainability"]["enabled"])  # Require explainability to remain inactive.
        self.assertFalse(config["memory_watcher"]["enabled"])  # Require no watcher sidecar.
        plan = stacking.build_evaluation_plan([(False, {"Random Forest": object()}, {})], [None], ["Full Features"], True)  # Build one individual plus one stacking combination.
        self.assertEqual(stacking.retain_stacking_classifier_plan(plan, True), [("Full Features", False, None, "StackingClassifier")])  # Retain only the requested stacking classifier combination.

    def test_process_name_cli_overrides_generated_default(self):
        """
        Verify the CLI process name is honored exactly when explicitly provided.

        :return: None.
        """

        with mock.patch("sys.argv", ["stacking.py", "--process-name", "DDoSDetector-AutoMLOnly", "--automl-only", "--combined-files", "--dataset-path", "./Datasets/CICDDoS2019/01-12/", "--n-jobs", "1", "--experiment-runs", "1"]):  # Parse the same process-title form used by detached server commands.
            cli_args = stacking.parse_cli_args()  # Resolve the requested operating-system identity.
        process_title_module = mock.Mock()  # Provide the optional production dependency without renaming this test process.
        with mock.patch.dict("sys.modules", {"setproctitle": process_title_module}):  # Route the local import through the isolated module.
            stacking.set_runtime_process_name(cli_args.process_name, script_path=stacking.__file__)  # Apply the production process-title path.
        process_title_module.setproctitle.assert_called_once_with("DDoSDetector-AutoMLOnly")  # Require the explicit user-supplied title without rewriting.


if __name__ == "__main__":
    unittest.main()  # Run the focused mode contract directly.
