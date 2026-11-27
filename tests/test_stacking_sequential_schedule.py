"""Focused coverage for the sequential combined-grid runtime scheduler."""

import unittest
from unittest import mock

import stacking


def task(feature_set, hyperparameters_enabled, ratio, classifier, global_id):
    return {
        "feature_set": feature_set,
        "hyperparameters_enabled": hyperparameters_enabled,
        "augmentation_ratio": ratio,
        "classifier_name": classifier,
        "global_id": global_id,
    }


class SequentialCombinedScheduleTests(unittest.TestCase):
    def test_runtime_order_stays_inside_groups_and_stacking_remains_last(self):
        tasks = [
            task("Full Features", False, None, "Slow", 2),
            task("Full Features", False, None, "StackingClassifier", 3),
            task("Full Features", False, None, "Fast", 5),
            task("Full Features", True, 0.25, "Second", 11),
            task("Full Features", True, 0.25, "First", 14),
            task("PCA Components", False, 1.0, "Only", 27),
        ]
        original_tuples = {
            (item["feature_set"], item["hyperparameters_enabled"], item["augmentation_ratio"], item["classifier_name"])
            for item in tasks
        }
        original_ids = {item["classifier_name"]: item["global_id"] for item in tasks}
        observed_groups = []

        def fake_runtime_sort(groups, cache_dict, payload):
            observed_groups.extend(groups)
            groups[("Full Features", False, None)][:] = [
                groups[("Full Features", False, None)][2],
                groups[("Full Features", False, None)][1],
                groups[("Full Features", False, None)][0],
            ]
            groups[("Full Features", True, 0.25)][:] = list(
                reversed(groups[("Full Features", True, 0.25)])
            )

        with mock.patch.object(
            stacking, "sort_pending_feature_process_tasks_by_elapsed_time", side_effect=fake_runtime_sort
        ):
            ordered = stacking.order_sequential_combined_tasks_by_runtime(
                tasks, {"cache": object()}, {"config": {"stacking": {"sort_pending_by_elapsed_time": True}}}
            )

        self.assertEqual(
            observed_groups,
            [
                ("Full Features", False, None),
                ("Full Features", True, 0.25),
                ("PCA Components", False, 1.0),
            ],
        )
        self.assertEqual(
            ordered,
            [
                ("Full Features", False, None, "Fast"),
                ("Full Features", False, None, "Slow"),
                ("Full Features", False, None, "StackingClassifier"),
                ("Full Features", True, 0.25, "First"),
                ("Full Features", True, 0.25, "Second"),
                ("PCA Components", False, 1.0, "Only"),
            ],
        )
        self.assertEqual(set(ordered), original_tuples)
        self.assertEqual(
            {item["classifier_name"]: item["global_id"] for item in tasks}, original_ids
        )

    def test_disabled_runtime_sort_keeps_stable_pending_order(self):
        pending = {
            "Full Features": [
                task("Full Features", False, None, "Second", 1),
                task("Full Features", False, None, "First", 2),
            ]
        }
        stacking.sort_pending_feature_process_tasks_by_elapsed_time(
            pending, {}, {"config": {"stacking": {"sort_pending_by_elapsed_time": False}}}
        )
        self.assertEqual(
            [item["classifier_name"] for item in pending["Full Features"]], ["Second", "First"]
        )


if __name__ == "__main__":
    unittest.main()
