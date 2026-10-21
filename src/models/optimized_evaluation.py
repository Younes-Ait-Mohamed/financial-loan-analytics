"""Metrics, calibration, thresholds, and deployment bundle utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def extract_model_scores(model: Any, X: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Return continuous positive-class scores and their source type."""
    if hasattr(model, "predict_proba"):
        scores = np.asarray(model.predict_proba(X))[:, 1]
        return scores.astype("float64"), "probability"
    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(X)).reshape(-1)
        return scores.astype("float64"), "decision"
    raise TypeError(
        "The fitted model exposes neither predict_proba nor decision_function."
    )


def normalized_pr_auc(pr_auc: float, prevalence: float) -> float:
    """Map prevalence-level PR-AUC to 0 and perfect PR-AUC to 1."""
    denominator = 1.0 - prevalence
    if denominator <= 0:
        return 0.0
    return float(np.clip((pr_auc - prevalence) / denominator, 0.0, 1.0))


def ranking_metrics(
    y_true: pd.Series | np.ndarray,
    scores: np.ndarray,
) -> dict[str, float]:
    """Calculate threshold-free ranking metrics."""
    y_array = np.asarray(y_true, dtype="int8")
    prevalence = float(y_array.mean())
    roc_auc = float(roc_auc_score(y_array, scores))
    pr_auc = float(average_precision_score(y_array, scores))
    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "prevalence": prevalence,
        "pr_auc_lift": pr_auc / prevalence if prevalence > 0 else np.nan,
        "normalized_pr_auc": normalized_pr_auc(pr_auc, prevalence),
        "ranking_score": (
            0.60 * normalized_pr_auc(pr_auc, prevalence) + 0.40 * roc_auc
        ),
    }


def operational_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate probability and threshold-dependent evaluation metrics."""
    y_array = np.asarray(y_true, dtype="int8")
    probabilities = np.clip(
        np.asarray(probabilities, dtype="float64"),
        1e-8,
        1.0 - 1e-8,
    )
    predictions = (probabilities >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(
        y_array,
        predictions,
        labels=[0, 1],
    ).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    false_positive_rate = fp / (tn + fp) if (tn + fp) else 0.0
    mcc = float(matthews_corrcoef(y_array, predictions))
    prevalence = float(y_array.mean())
    pr_auc = float(average_precision_score(y_array, probabilities))

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_array, predictions)),
        "precision": float(precision_score(y_array, predictions, zero_division=0)),
        "recall": float(recall_score(y_array, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_array, predictions, zero_division=0)),
        "f2_score": float(fbeta_score(y_array, predictions, beta=2, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_array, predictions)),
        "mcc": mcc,
        "specificity": float(specificity),
        "false_positive_rate": float(false_positive_rate),
        "roc_auc": float(roc_auc_score(y_array, probabilities)),
        "pr_auc": pr_auc,
        "pr_auc_lift": pr_auc / prevalence if prevalence > 0 else np.nan,
        "brier_score": float(brier_score_loss(y_array, probabilities)),
        "log_loss": float(log_loss(y_array, probabilities, labels=[0, 1])),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "support": int(len(y_array)),
        "prevalence": prevalence,
    }


def threshold_candidates(probabilities: np.ndarray) -> np.ndarray:
    """Create a dense threshold grid over the observed score distribution."""
    probabilities = np.asarray(probabilities, dtype="float64")
    fixed = np.linspace(0.01, 0.99, 197)
    quantile_grid = np.quantile(
        probabilities,
        np.linspace(0.02, 0.98, 97),
    )
    candidates = np.unique(
        np.clip(np.concatenate([fixed, quantile_grid]), 1e-4, 1.0 - 1e-4)
    )
    return candidates


def threshold_analysis(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """Evaluate transparent threshold policies on validation data."""
    rows = []
    for threshold in threshold_candidates(probabilities):
        metrics = operational_metrics(y_true, probabilities, threshold)
        mcc_scaled = (float(metrics["mcc"]) + 1.0) / 2.0
        metrics["balanced_composite"] = (
            0.30 * float(metrics["f1_score"])
            + 0.25 * float(metrics["balanced_accuracy"])
            + 0.25 * mcc_scaled
            + 0.10 * float(metrics["precision"])
            + 0.10 * float(metrics["recall"])
        )
        rows.append(metrics)
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def select_threshold_policies(thresholds: pd.DataFrame) -> pd.DataFrame:
    """Return several valid threshold choices without hiding trade-offs."""
    if thresholds.empty:
        raise ValueError("The threshold table is empty.")

    policies = {
        "balanced_composite": thresholds["balanced_composite"].idxmax(),
        "maximum_f1": thresholds["f1_score"].idxmax(),
        "maximum_f2": thresholds["f2_score"].idxmax(),
        "maximum_mcc": thresholds["mcc"].idxmax(),
        "maximum_balanced_accuracy": thresholds["balanced_accuracy"].idxmax(),
    }
    rows = []
    for policy, index in policies.items():
        row = thresholds.loc[index].to_dict()
        row["policy"] = policy
        rows.append(row)
    return pd.DataFrame(rows).set_index("policy")


class ScoreCalibrator:
    """Calibrate probability or decision scores without refitting the base model."""

    VALID_METHODS = {"identity", "sigmoid", "isotonic"}

    def __init__(self, method: str = "identity") -> None:
        if method not in self.VALID_METHODS:
            valid = ", ".join(sorted(self.VALID_METHODS))
            raise ValueError(f"Calibration method must be one of: {valid}")
        self.method = method

    def fit(
        self,
        scores: np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "ScoreCalibrator":
        scores = np.asarray(scores, dtype="float64").reshape(-1)
        target = np.asarray(y, dtype="int8")
        if self.method == "identity":
            if scores.min() < 0 or scores.max() > 1:
                raise ValueError("Identity calibration requires probability scores.")
            self.model_ = None
        elif self.method == "sigmoid":
            self.model_ = LogisticRegression(
                C=1e6,
                max_iter=2000,
                random_state=42,
            )
            self.model_.fit(scores.reshape(-1, 1), target)
        else:
            self.model_ = IsotonicRegression(
                out_of_bounds="clip",
                y_min=0.0,
                y_max=1.0,
            )
            self.model_.fit(scores, target)
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise AttributeError("The calibrator must be fitted first.")
        scores = np.asarray(scores, dtype="float64").reshape(-1)
        if self.method == "identity":
            probabilities = scores
        elif self.method == "sigmoid":
            probabilities = self.model_.predict_proba(scores.reshape(-1, 1))[:, 1]
        else:
            probabilities = self.model_.predict(scores)
        return np.clip(probabilities, 1e-8, 1.0 - 1e-8)


def compare_calibrators(
    score_type: str,
    calibration_scores: np.ndarray,
    calibration_target: pd.Series,
    selection_scores: np.ndarray,
    selection_target: pd.Series,
) -> tuple[ScoreCalibrator, pd.DataFrame]:
    """Fit calibrators on one period and select them on a later period."""
    methods = ["sigmoid", "isotonic"]
    if score_type == "probability":
        methods.insert(0, "identity")

    rows = []
    fitted: dict[str, ScoreCalibrator] = {}
    target = np.asarray(selection_target, dtype="int8")
    for method in methods:
        calibrator = ScoreCalibrator(method=method).fit(
            calibration_scores,
            calibration_target,
        )
        probabilities = calibrator.transform(selection_scores)
        fitted[method] = calibrator
        rows.append(
            {
                "method": method,
                "brier_score": brier_score_loss(target, probabilities),
                "log_loss": log_loss(target, probabilities, labels=[0, 1]),
            }
        )

    comparison = pd.DataFrame(rows).sort_values(
        ["brier_score", "log_loss"],
        ascending=True,
    )
    best_method = str(comparison.iloc[0]["method"])
    return fitted[best_method], comparison.reset_index(drop=True)


@dataclass
class OptimizedRiskModel:
    """Serializable pipeline, calibrator, threshold, and inference metadata."""

    pipeline: Any
    calibrator: ScoreCalibrator
    threshold: float
    model_name: str
    raw_features: list[str]
    feature_configuration: dict[str, Any]
    metadata: dict[str, Any]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._validate_input(X)
        scores, _ = extract_model_scores(self.pipeline, X)
        probabilities = self.calibrator.transform(scores)
        return np.column_stack([1.0 - probabilities, probabilities])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= self.threshold).astype("int8")

    def predict_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        start = perf_counter()
        probabilities = self.predict_proba(X)[:, 1]
        predictions = (probabilities >= self.threshold).astype("int8")
        elapsed = perf_counter() - start
        return pd.DataFrame(
            {
                "predicted_class": predictions,
                "probability_of_default": probabilities,
                "decision_threshold": self.threshold,
                "prediction_seconds": elapsed,
            },
            index=X.index,
        )

    def save(self, path: Path | str) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)

    @classmethod
    def load(cls, path: Path | str) -> "OptimizedRiskModel":
        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise TypeError("Saved artifact is not an OptimizedRiskModel.")
        return model

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Prediction input must be a pandas DataFrame.")
        if X.empty:
            raise ValueError("Prediction input must contain at least one row.")
        missing = sorted(set(self.raw_features) - set(X.columns))
        if missing:
            raise ValueError(
                "Missing required origination columns: " + ", ".join(missing)
            )
