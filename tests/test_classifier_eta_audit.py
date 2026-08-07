import contextlib
import io
import unittest

import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification

import classifier_eta
import stacking
import training_progress


def all_enabled_config():
    config = stacking.get_default_config()
    config["stacking"]["enabled_classifiers"] = list(stacking.load_config_file("config.yaml")["stacking"]["enabled_classifiers"])
    config["evaluation"]["n_jobs"] = 1
    return config


class ClassifierETAAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = all_enabled_config()
        cls.X, cls.y = make_classification(n_samples=40, n_features=5, n_informative=3, n_redundant=0, random_state=7)

    def discover_classifiers(self):
        models = dict(stacking.get_models(self.config))
        for name in stacking.get_automl_search_spaces(self.config):
            if name not in models:
                models[name] = stacking.create_model_from_params(name, {}, self.config)
        models["StackingClassifier"] = stacking.build_evaluation_stacking_model({"KNN": models["KNN"], "Decision Tree": models["Decision Tree"]}, self.config)
        return models

    def test_every_supported_classifier_has_eta_classification(self):
        models = self.discover_classifiers()
        self.assertEqual(classifier_eta.missing_eta_classifications(models), [])
        rows = classifier_eta.audit_classifier_eta_support(models)
        self.assertEqual({row["classifier"] for row in rows}, set(models))
        for row in rows:
            self.assertTrue(row["library_class"])
            self.assertIn(row["final_support"], {"Derived live ETA", "ETA unavailable"})
            if row["final_support"] == "ETA unavailable":
                self.assertTrue(row["unavailable_reason"])
            else:
                self.assertTrue(row["measurable_progress"])

    def test_eta_calculator_edges(self):
        output = io.StringIO()
        progress = training_progress.TrainingProgress("PCA Components", "XGBoost", lambda seconds: f"{int(seconds)}s", output_stream=output, total_units=4, unit_label="Round", report_interval_seconds=10.0)
        with unittest.mock.patch.object(training_progress.time, "monotonic", side_effect=[100.0, 110.0, 120.0, 130.0]):
            with progress:
                progress.report_unit(0)
                progress.report_unit(1)
                progress.report_unit(2)
                progress.report_unit(4)
        text = output.getvalue()
        self.assertNotIn("inf", text)
        self.assertNotIn("nan", text)
        self.assertNotIn("ETA: -", text)
        self.assertIn("ETA: 30s", text)
        self.assertIn("ETA: 20s", text)

    def test_unknown_classifier_is_flagged(self):
        with self.assertRaisesRegex(KeyError, "Missing ETA classification"):
            classifier_eta.audit_classifier_eta_support({"New Model": object()})

    def test_no_metric_change_for_supported_eta_estimators(self):
        supported = ["XGBoost", "LightGBM", "Gradient Boosting"]
        config = dict(self.config)
        config["evaluation"] = dict(self.config["evaluation"], training_progress_interval_minutes=0.0 + 0.001 / 60.0)
        models = self.discover_classifiers()
        for name in supported:
            with self.subTest(name=name):
                baseline = clone(models[name])
                observed = clone(models[name])
                baseline.set_params(**({"n_estimators": 3} if "n_estimators" in baseline.get_params() else {}))
                observed.set_params(**({"n_estimators": 3} if "n_estimators" in observed.get_params() else {}))
                baseline.fit(self.X, self.y)
                with contextlib.redirect_stdout(io.StringIO()):
                    stacking.fit_classifier_with_progress(observed, self.X, self.y, "Full Features", name, config=config)
                np.testing.assert_array_equal(baseline.predict(self.X), observed.predict(self.X))


if __name__ == "__main__":
    unittest.main()
