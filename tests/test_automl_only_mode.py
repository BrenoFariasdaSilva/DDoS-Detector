"""Focused AutoML-only pipeline mode coverage."""

import unittest  # Provide the repository-standard focused test runner.
from unittest import mock  # Observe the isolated production route without training models.

import pandas as pd  # Build a minimal combined dataset for routing coverage.

import stacking  # Exercise production configuration and AutoML routing behavior.


class AutoMLOnlyModeTests(unittest.TestCase):
    """Verify isolated AutoML mode without running optimization trials."""

    def test_automl_only_restricts_configuration_and_bypasses_grid(self):
        """
        Verify AutoML-only mode skips regular planning and artifact loading.

        :return: None.
        """

        config = stacking.get_default_config()  # Build an independent production-default configuration.
        config["stacking"]["automl_only"] = True  # Select isolated AutoML through the config contract.
        config = stacking.merge_configs(config, {}, None)  # Apply the same normalization used by config.yaml execution.
        config["stacking"]["experiment_run"] = 1  # Enter the per-run combined evaluation boundary directly.
        self.assertEqual(config["stacking"]["methods"], {"augmentation": False, "feature_selection": False, "hyperparameter_optimization": False, "automl": True, "stacking": False})  # Require exactly the AutoML pipeline.
        combined_df = pd.DataFrame({"feature": [0.0, 1.0], "attack_type": ["a", "b"]})  # Provide one numeric feature and a two-class target.
        with mock.patch.object(stacking, "resolve_combined_files_dataset_identity", return_value="/tmp/dataset"), mock.patch.object(stacking, "compute_class_distribution", return_value=[]), mock.patch.object(stacking, "send_telegram_message"), mock.patch.object(stacking, "run_automl_pipeline") as automl_run, mock.patch.object(stacking, "load_feature_selection_results") as artifact_load, mock.patch.object(stacking, "get_models") as model_load:  # Isolate routing from filesystem, notifications, and training.
            stacking.process_combined_files_evaluation(["/tmp/dataset/input.csv"], combined_df, ["a", "b"], "dataset", config=config)  # Execute the production combined-mode boundary.
        automl_run.assert_called_once_with("/tmp/dataset", combined_df, ["feature"], data_source_label="Original Combined Files", config=config)  # Require exactly one AutoML pipeline call.
        artifact_load.assert_not_called()  # Require no feature-selection artifact work.
        model_load.assert_not_called()  # Require no regular classifier construction.


if __name__ == "__main__":
    unittest.main()  # Run the focused mode contract directly.
