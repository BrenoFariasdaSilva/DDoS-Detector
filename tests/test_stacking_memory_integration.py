"""Plan, cache-policy, and recovery regressions for memory-managed stacking runs."""

import math
import unittest
from unittest import mock

import stacking


class StackingMemoryIntegrationTests(unittest.TestCase):
    def test_augmented_plan_covers_all_ratios_features_hp_and_model_types(self):
        feature_metadata = {
            "Full Features": {"feature_count": 4, "feature_names": ["a", "b", "c", "d"]},
            "GA Features": {"feature_count": 2, "feature_names": ["a", "c"]},
            "PCA Components": {"feature_count": 2, "feature_names": ["PC1", "PC2"]},
            "RFE Features": {"feature_count": 2, "feature_names": ["b", "d"]},
            "Extra Trees Features": {"feature_count": 2, "feature_names": ["a", "d"]},
        }
        ratios = (0.25, 0.50, 0.75, 1.00)
        classifiers = ("Random Forest", "LSTM")
        plan = [
            (feature_set, hp_enabled, ratio, classifier)
            for feature_set in feature_metadata
            for hp_enabled in (False, True)
            for ratio in ratios
            for classifier in classifiers
        ]
        global_ids = {combination: index * 2 + 1 for index, combination in enumerate(plan)}
        tasks = stacking.build_feature_process_plan(
            plan,
            feature_metadata,
            101,
            "/tmp/fixture",
            "combined_files",
            1,
            global_ids,
            200,
        )

        self.assertEqual(len(tasks), 5 * 2 * 4 * 2)
        self.assertEqual({task["augmentation_ratio"] for task in tasks}, set(ratios))
        self.assertEqual({task["feature_set"] for task in tasks}, set(feature_metadata))
        self.assertEqual({task["hyperparameters_enabled"] for task in tasks}, {False, True})
        self.assertEqual({task["classifier_name"] for task in tasks}, set(classifiers))
        self.assertTrue(all(task["expected_n_samples_train"] is None for task in tasks))
        self.assertTrue(all(task["expected_n_samples_test"] == math.ceil(101 * 0.2) for task in tasks))
        self.assertEqual([task["global_id"] for task in tasks], [global_ids[item] for item in plan])
        self.assertTrue(all(task["canonical_total"] == 200 for task in tasks))

    def test_fully_cached_ratio_requires_every_task_to_pass_normal_cache_policy(self):
        tasks = [
            {"augmentation_ratio": 0.25, "global_id": 1},
            {"augmentation_ratio": 0.25, "global_id": 2},
            {"augmentation_ratio": 0.50, "global_id": 3},
            {"augmentation_ratio": 0.50, "global_id": 4},
            {"augmentation_ratio": None, "global_id": 5},
        ]
        with mock.patch.object(
            stacking,
            "feature_process_cache_result",
            side_effect=lambda task, cache, attacks: {} if task["global_id"] in {1, 3, 4} else None,
        ):
            complete = stacking.resolve_fully_cached_ratios(tasks, {"cache": object()}, ["attack"])
        self.assertEqual(complete, {0.50})

    def test_identity_only_augmented_cache_cannot_skip_source_provenance(self):
        metadata = {"Full Features": {"feature_count": 2, "feature_names": ["a", "b"]}}
        plan = [("Full Features", False, 0.25, "Random Forest")]
        task = stacking.build_feature_process_plan(
            plan, metadata, 100, "/tmp/fixture", "combined_files", 1
        )[0]
        key = stacking.build_resume_cache_key(
            task["execution_mode"],
            task["data_source_label"],
            task["experiment_mode"],
            task["augmentation_ratio"],
            ["attack"],
            task["feature_set"],
            task["classifier_name"],
            task["hyperparameters_enabled"],
        )
        plausible_but_unproven = {
            "n_features": 2,
            "features_list": ["a", "b"],
            "n_samples_train": 100,
            "n_samples_test": 20,
        }
        self.assertEqual(
            stacking.resolve_fully_cached_ratios([task], {key: plausible_but_unproven}, ["attack"]),
            set(),
        )

    def test_disabling_model_recovery_does_not_disable_metrics_cache_reruns(self):
        with mock.patch(
            "sys.argv",
            ["stacking.py", "--disable-model-recovery", "--rerun-cached-experiments"],
        ):
            args = stacking.parse_cli_args()
        self.assertFalse(args.enable_model_recovery)
        self.assertEqual(args.rerun_cached_experiments, 1)


if __name__ == "__main__":
    unittest.main()
