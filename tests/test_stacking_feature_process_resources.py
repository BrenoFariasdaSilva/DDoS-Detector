import argparse  # Build minimal invalid CLI namespaces.
import copy  # Isolate runtime configuration mutations.
from pathlib import Path  # Build temporary source and resource paths.
import sys  # Supply deterministic CLI arguments.
import tempfile  # Isolate real preprocessing artifacts.
import unittest  # Run deterministic resource-equivalence coverage.
from unittest import mock  # Isolate CLI and sequential orchestration boundaries.

import numpy as np  # Build and compare exact numeric estimator inputs.
import pandas as pd  # Build small original and augmented datasets.
from sklearn.linear_model import LogisticRegression  # Exercise the existing real classifier workflow.
from sklearn.preprocessing import LabelEncoder, StandardScaler  # Build exact fitted preprocessing artifacts.
import yaml  # Validate both repository configuration files.

from feature_processes import close_array_resource, create_resource_directory, open_array_resource, partition_evaluation_plan, remove_resource_directory, run_feature_set_processes, save_array_resource  # Inspect production process, memmap, and ownership behavior.
import stacking  # Exercise the real split, feature-selection, PCA, and augmentation resource path.


class StackingFeatureProcessResourceTests(unittest.TestCase):  # Prove real process resources preserve scientific inputs.
    def test_cli_yaml_fallback_and_sequential_mode_validation(self) -> None:
        """
        Verify CLI precedence, YAML fallback, and explicit scheduling boundaries.

        :param self: Instance of the StackingFeatureProcessResourceTests class.
        :return: None.
        """

        repository_root = Path(__file__).resolve().parents[1]  # Resolve the repository configuration location.
        for config_name in ("config.yaml", "config.yaml.example"):  # Validate both operator configuration surfaces.
            loaded_config = yaml.safe_load((repository_root / config_name).read_text(encoding="utf-8"))  # Parse the complete YAML document.
            self.assertEqual(loaded_config["evaluation"]["feature_set_workers"], {})  # Require the documented sequential fallback.
        with mock.patch.object(sys, "argv", ["stacking.py", "--feature-set-workers", "ga=1,pca=1,rfe=1"]):  # Supply the production CLI mapping.
            cli_args = stacking.parse_cli_args()  # Parse through the actual command-line surface.
        merged = stacking.merge_configs(stacking.get_default_config(), {}, cli_args)  # Apply the actual CLI precedence path.
        self.assertEqual(merged["evaluation"]["feature_set_workers"], {"ga": 1, "pca": 1, "rfe": 1})  # Require normalized runtime configuration.
        invalid_args = argparse.Namespace(feature_set_workers="ga=2,pca=1,rfe=1")  # Build one explicitly unsupported worker count.
        with self.assertRaisesRegex(ValueError, "must be exactly 1"):  # Require validation instead of misleading shared-queue behavior.
            stacking.merge_configs(stacking.get_default_config(), {}, invalid_args)  # Reject the invalid CLI override.
        sequential_config = stacking.get_default_config()  # Build the empty-worker sequential fallback.
        with mock.patch.object(stacking, "determine_files_to_process", return_value=[]):  # Avoid dataset discovery in the focused mode test.
            stacking.orchestrate_all_combinations("unused", config=sequential_config)  # Preserve existing sequential execution when workers are disabled.
        process_config = stacking.get_default_config()  # Build an invalid separate-files process request.
        process_config["evaluation"]["feature_set_workers"] = {"ga": 1}  # Enable one persistent process outside combined-files mode.
        with self.assertRaisesRegex(ValueError, "supported only"):  # Require an explicit mode boundary.
            stacking.orchestrate_all_combinations("unused", config=process_config)  # Reject silently ignored separate-files process configuration.

    def test_real_feature_and_label_resources_match_existing_preprocessing(self) -> None:
        """
        Verify process memmaps match established feature and label preprocessing.

        :param self: Instance of the StackingFeatureProcessResourceTests class.
        :return: None.
        """

        with tempfile.TemporaryDirectory() as temporary_directory:  # Isolate PCA and memmap artifacts.
            dataset_directory = Path(temporary_directory)  # Resolve the combined dataset identity.
            source_file = dataset_directory / "source.csv"  # Build one real provenance file.
            values = np.arange(240, dtype=np.float64).reshape(60, 4)  # Build deterministic nonconstant numeric features.
            labels = np.asarray(["Benign", "Attack"] * 30)  # Build balanced deterministic labels.
            original_df = pd.DataFrame(values, columns=["f0", "f1", "f2", "f3"])  # Build the original feature frame.
            original_df["attack_type"] = labels  # Append the positional target column.
            original_df.to_csv(source_file, index=False)  # Persist provenance required by PCA identity validation.
            augmented_df = pd.DataFrame(values[::-1] + 0.5, columns=["f0", "f1", "f2", "f3"])  # Build deterministic augmented feature values.
            augmented_df["attack_type"] = labels[::-1]  # Append compatible augmented labels.
            config = copy.deepcopy(stacking.get_default_config())  # Start from established runtime defaults.
            config["execution"]["execution_mode"] = "combined_files"  # Preserve combined-files preprocessing identity.
            config["stacking"]["feature_sets_config"] = {"use_full": False, "use_pca": True, "use_rfe": True, "use_ga": True, "explicit_features": []}  # Select only the three requested feature sets.
            config["memory_watcher"]["enabled"] = False  # Avoid unrelated watcher artifacts in the focused test.
            stacking.CONFIG = config  # Preserve global fallback behavior used by existing nested functions.
            direct_splits = stacking.prepare_evaluation_data_splits(original_df.copy(), config=config)  # Produce the established exact split and scaling outputs.
            X_train_scaled, X_test_scaled, y_train, y_test, direct_scaler, direct_encoder = direct_splits  # Unpack authoritative estimator inputs.
            resource_directory = None  # Track coordinator cleanup ownership across assertions.
            opened_arrays = []  # Track test-owned read-only mappings.
            try:  # Release mappings before coordinator resource deletion.
                resource_directory, resources, scaler, label_encoder, input_names = stacking.build_feature_process_resources(str(dataset_directory), original_df.copy(), ["f0", "f1", "f2", "f3"], ["f0", "f1"], 2, ["f2", "f3"], [str(source_file)], [0.5], augmented_df.copy(), ["Attack", "Benign"], config)  # Build real production process resources.
                self.assertEqual(input_names, ["f0", "f1", "f2", "f3"])  # Require unchanged numeric input ordering.
                self.assertEqual(list(label_encoder.classes_), list(direct_encoder.classes_))  # Require unchanged label encoding identity.
                mapped_y_train = open_array_resource(resources["labels"]["train"])  # Open coordinator-persisted training labels.
                mapped_y_test = open_array_resource(resources["labels"]["test"])  # Open coordinator-persisted testing labels.
                opened_arrays.extend([mapped_y_train, mapped_y_test])  # Register label mapping ownership.
                np.testing.assert_array_equal(mapped_y_train, y_train)  # Require exact training-label values and order.
                np.testing.assert_array_equal(mapped_y_test, y_test)  # Require exact testing-label values and order.
                expected_by_feature = {"GA Features": (X_train_scaled[:, [0, 1]], X_test_scaled[:, [0, 1]]), "PCA Components": (resources["features"]["PCA Components"]["transformer"].transform(X_train_scaled), resources["features"]["PCA Components"]["transformer"].transform(X_test_scaled)), "RFE Features": (X_train_scaled[:, [2, 3]], X_test_scaled[:, [2, 3]])}  # Build exact established feature-set expectations.
                for feature_set, (expected_train, expected_test) in expected_by_feature.items():  # Compare every selected process matrix.
                    mapped_train = open_array_resource(resources["features"][feature_set]["train"])  # Open this feature set's training resource.
                    mapped_test = open_array_resource(resources["features"][feature_set]["test"])  # Open this feature set's testing resource.
                    opened_arrays.extend([mapped_train, mapped_test])  # Register feature mapping ownership.
                    np.testing.assert_array_equal(mapped_train, expected_train)  # Require bitwise-identical training inputs.
                    np.testing.assert_array_equal(mapped_test, expected_test)  # Require bitwise-identical testing inputs.
                sampled_augmented = stacking.sample_augmented_by_ratio(augmented_df, len(original_df), 0.5)  # Reuse established deterministic ratio sampling.
                augmented_scaled = direct_scaler.transform(sampled_augmented[["f0", "f1", "f2", "f3"]].to_numpy(copy=False))  # Build the established augmented scaled matrix.
                augmented_labels = direct_encoder.transform(sampled_augmented["attack_type"].to_numpy(copy=False)).astype(np.int64)  # Build established augmented labels.
                expected_augmented = {"GA Features": augmented_scaled[:, [0, 1]], "PCA Components": resources["features"]["PCA Components"]["transformer"].transform(augmented_scaled), "RFE Features": augmented_scaled[:, [2, 3]]}  # Build exact augmented feature expectations.
                for feature_set, expected_test in expected_augmented.items():  # Compare every augmented process matrix.
                    augmented_metadata = resources["features"][feature_set]["augmented"][0.5]  # Resolve ratio-specific metadata.
                    mapped_augmented = open_array_resource(augmented_metadata["test"])  # Open this feature set's augmented test resource.
                    mapped_augmented_labels = open_array_resource(augmented_metadata["labels"])  # Open shared augmented labels.
                    opened_arrays.extend([mapped_augmented, mapped_augmented_labels])  # Register augmented mapping ownership.
                    np.testing.assert_array_equal(mapped_augmented, expected_test)  # Require bitwise-identical augmented estimator features.
                    np.testing.assert_array_equal(mapped_augmented_labels, augmented_labels)  # Require exact augmented label values and order.
            finally:  # Enforce worker-before-coordinator cleanup ordering in the test.
                for array in reversed(opened_arrays):  # Close every test-owned mapping first.
                    close_array_resource(array)  # Release the read-only mapping without deleting its file.
                remove_resource_directory(resource_directory)  # Delete resources only after all mappings are closed.
                if resource_directory is not None:  # Verify coordinator deletion after known ownership release.
                    self.assertFalse(Path(resource_directory).exists())  # Require no process resource leak.

    def test_real_stacking_worker_fits_then_recovers_cache_without_model_rewrite(self) -> None:
        """
        Verify the real persistent worker reuses evaluation, artifacts, and cache recovery.

        :param self: Instance of the StackingFeatureProcessResourceTests class.
        :return: None.
        """

        with tempfile.TemporaryDirectory() as temporary_directory:  # Isolate cache, models, logs, and memmaps.
            dataset_directory = Path(temporary_directory)  # Resolve the combined dataset identity.
            source_file = dataset_directory / "source.csv"  # Build one real model-provenance source.
            raw_features = np.arange(80, dtype=np.float64).reshape(40, 2)  # Build deterministic nonconstant source values.
            raw_labels = np.asarray(["Attack", "Benign"] * 20)  # Build balanced source labels.
            source_df = pd.DataFrame(raw_features, columns=["f0", "f1"])  # Build the provenance feature frame.
            source_df["attack_type"] = raw_labels  # Append the positional source target.
            source_df.to_csv(source_file, index=False)  # Persist exact source provenance.
            scaler = StandardScaler().fit(raw_features[:32])  # Fit preprocessing only on deterministic training rows.
            label_encoder = LabelEncoder().fit(raw_labels)  # Fit the established sorted label identity.
            X_train = scaler.transform(raw_features[:32])  # Build exact deterministic estimator training features.
            X_test = scaler.transform(raw_features[32:])  # Build exact deterministic estimator testing features.
            y_train = label_encoder.transform(raw_labels[:32]).astype(np.int64)  # Build exact encoded training labels.
            y_test = label_encoder.transform(raw_labels[32:]).astype(np.int64)  # Build exact encoded testing labels.
            resource_directory = create_resource_directory(str(dataset_directory))  # Create one coordinator-owned memmap directory.
            try:  # Delete resources only after both real worker runs exit.
                feature_resources = {"train": save_array_resource(X_train, resource_directory, "ga_train"), "test": save_array_resource(X_test, resource_directory, "ga_test"), "feature_names": ["f0", "f1"], "transformer": None, "augmented": {}}  # Persist only the assigned feature matrices.
                label_resources = {"train": save_array_resource(y_train, resource_directory, "y_train"), "test": save_array_resource(y_test, resource_directory, "y_test")}  # Persist shared labels once.
                plan = [("GA Features", False, None, "Logistic Regression")]  # Build one real classifier combination.
                tasks = partition_evaluation_plan(plan)["GA Features"]  # Preserve authoritative combination metadata.
                config = copy.deepcopy(stacking.get_default_config())  # Start from established runtime defaults.
                config["execution"]["execution_mode"] = "combined_files"  # Preserve combined-files cache and artifact identity.
                config["evaluation"]["feature_set_workers"] = {"ga": 1}  # Enable one real persistent feature worker.
                config["evaluation"]["n_jobs"] = 1  # Preserve estimator-level sequential execution.
                config["stacking"]["methods"]["stacking"] = False  # Keep this focused integration workload to one classifier.
                config["explainability"]["enabled"] = False  # Avoid unrelated explainability work.
                config["memory_watcher"]["enabled"] = False  # Avoid unrelated watcher work.
                config["telegram"]["enabled"] = False  # Avoid external notifications.
                config["paths"]["logs_dir"] = str(dataset_directory / "Logs")  # Isolate the shared process log.
                model = LogisticRegression(random_state=42, max_iter=200)  # Build one deterministic production-compatible estimator.
                payload = {"feature_set": "GA Features", "worker_index": 1, "tasks": tasks, "feature_resources": feature_resources, "label_resources": label_resources, "scaler": scaler, "label_encoder": label_encoder, "models_by_hp": {False: {"Logistic Regression": model}}, "params_by_hp": {False: {}}, "experiment_ids": {(False, None): "integration-original"}, "file": str(dataset_directory), "execution_mode": "combined_files", "attack_types": ["Attack", "Benign"], "original_data_source": "Original Combined Files", "input_feature_names": ["f0", "f1"], "target_column": "attack_type", "source_files": [str(source_file)], "config": config}  # Pass only small metadata, preprocessing, model prototypes, and memmap paths.
                first_results = run_feature_set_processes([payload], 1, "stacking", "run_feature_set_process")  # Fit, predict, persist, export, and exit through the real child worker.
                self.assertEqual(len(first_results), 1)  # Require the complete real result delivery.
                model_artifacts = sorted((dataset_directory / "Stacking" / "Models").rglob("*.joblib"))  # Locate every atomically persisted fitted artifact.
                self.assertTrue(model_artifacts)  # Require real model persistence before cache recovery.
                artifact_mtimes = {path: path.stat().st_mtime_ns for path in model_artifacts}  # Record exact artifact modification times.
                second_results = run_feature_set_processes([payload], 1, "stacking", "run_feature_set_process")  # Recover the same combination from cache without fitting.
                self.assertEqual(second_results[0]["accuracy"], first_results[0]["accuracy"])  # Require unchanged recovered metrics.
                self.assertEqual({path: path.stat().st_mtime_ns for path in model_artifacts}, artifact_mtimes)  # Require no model rewrite on the cache hit.
            finally:  # Enforce coordinator-only resource deletion after both child joins.
                remove_resource_directory(resource_directory)  # Delete exact memmaps after every real worker exits.
                self.assertFalse(Path(resource_directory).exists())  # Require no memmap resource leak.


if __name__ == "__main__":  # Support direct focused execution.
    unittest.main()  # Run real resource-equivalence coverage.
