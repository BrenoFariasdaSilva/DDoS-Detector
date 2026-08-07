from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class ClassifierETASupport:
    mechanism: str
    native_eta: bool
    derived_eta: bool
    prediction_eta: str
    final_support: str
    unavailable_reason: str = ""

    @property
    def measurable_progress(self) -> bool:
        return self.native_eta or self.derived_eta


CLASSIFIER_ETA_SUPPORT: dict[str, ClassifierETASupport] = {
    "Random Forest": ClassifierETASupport("n_estimators is configured, but sklearn exposes no fit callback; warm_start batching would change training semantics", False, False, "Unavailable: predict is one blocking sklearn call", "ETA unavailable", "No public per-tree training progress callback"),
    "SVM": ClassifierETASupport("No public libsvm progress callback; verbose output is text-only and not structured for ETA", False, False, "Unavailable: predict is one blocking sklearn call", "ETA unavailable", "SVC.fit is one blocking solver call"),
    "XGBoost": ClassifierETASupport("xgboost.callback.TrainingCallback.after_iteration plus get_num_boosting_rounds()", False, True, "Unavailable: predict is one blocking XGBoost call", "Derived live ETA"),
    "Logistic Regression": ClassifierETASupport("n_iter_ exists only after fit completes", False, False, "Unavailable: predict is one blocking sklearn call", "ETA unavailable", "No public live iteration callback"),
    "KNN": ClassifierETASupport("Training stores neighbors in one blocking fit", False, False, "Unavailable: predict is one blocking sklearn call", "ETA unavailable", "No measurable training loop"),
    "Nearest Centroid": ClassifierETASupport("Centroids computed in one blocking fit", False, False, "Unavailable: predict is one blocking sklearn call", "ETA unavailable", "No measurable training loop"),
    "Gradient Boosting": ClassifierETASupport("sklearn GradientBoostingClassifier.fit(monitor=...) stage callback; staged_predict exists post-fit only", False, True, "Unavailable: predict is one blocking sklearn call", "Derived live ETA"),
    "LightGBM": ClassifierETASupport("LightGBM fit callbacks with CallbackEnv.iteration and n_estimators", False, True, "Unavailable: predict is one blocking LightGBM call", "Derived live ETA"),
    "MLP (Neural Net)": ClassifierETASupport("n_iter_ exists during/after fit but sklearn exposes no public per-iteration callback", False, False, "Unavailable: predict is one blocking sklearn call", "ETA unavailable", "Live ETA would require private sklearn monkeypatching or changing to partial_fit/warm_start"),
    "FT-Transformer": ClassifierETASupport("Repository progress_callback called after each completed epoch", False, True, "Unavailable: predict batches internally but exposes no prediction progress callback", "Derived live ETA"),
    "Tabular ResNet": ClassifierETASupport("Repository progress_callback called after each completed epoch", False, True, "Unavailable: predict batches internally but exposes no prediction progress callback", "Derived live ETA"),
    "ResNet18": ClassifierETASupport("Repository progress_callback called after each completed epoch", False, True, "Unavailable: predict batches internally but exposes no prediction progress callback", "Derived live ETA"),
    "AutoEncoder": ClassifierETASupport("Repository progress_callback called after each completed epoch", False, True, "Unavailable: predict batches internally but exposes no prediction progress callback", "Derived live ETA"),
    "LSTM": ClassifierETASupport("Repository progress_callback called after each completed epoch", False, True, "Unavailable: predict batches internally but exposes no prediction progress callback", "Derived live ETA"),
    "Extra Trees": ClassifierETASupport("n_estimators is configured, but sklearn exposes no fit callback; warm_start batching would change training semantics", False, False, "Unavailable: predict is one blocking sklearn call", "ETA unavailable", "No public per-tree training progress callback"),
    "Decision Tree": ClassifierETASupport("Tree built in one blocking fit", False, False, "Unavailable: predict is one blocking sklearn call", "ETA unavailable", "No measurable training loop"),
    "StackingClassifier": ClassifierETASupport("sklearn StackingClassifier hides nested CV/base-estimator fit progress", False, False, "Unavailable: predict is one blocking sklearn call", "ETA unavailable", "No public live progress across folds/base learners"),
}


def audit_classifier_eta_support(classifiers: Mapping[str, Any]) -> list[dict[str, Any]]:
    unknown = [name for name in classifiers if name not in CLASSIFIER_ETA_SUPPORT]
    if unknown:
        raise KeyError(f"Missing ETA classification for: {', '.join(sorted(unknown))}")
    rows = []
    for name, model in classifiers.items():
        support = CLASSIFIER_ETA_SUPPORT[name]
        model_type = type(model)
        rows.append({
            "classifier": name,
            "library_class": f"{model_type.__module__}.{model_type.__qualname__}",
            "native_eta": support.native_eta,
            "measurable_progress": support.measurable_progress,
            "mechanism": support.mechanism,
            "derived_eta": support.derived_eta,
            "prediction_eta": support.prediction_eta,
            "final_support": support.final_support,
            "unavailable_reason": support.unavailable_reason,
        })
    return rows


def missing_eta_classifications(names: Iterable[str]) -> list[str]:
    return sorted(name for name in names if name not in CLASSIFIER_ETA_SUPPORT)
