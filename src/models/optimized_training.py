"""Time-aware data preparation, model construction, screening, and tuning."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .optimized_evaluation import extract_model_scores, ranking_metrics
from .optimized_features import (
    FEATURE_CONFIGURATIONS,
    SAFE_RAW_FEATURES,
    TARGET_COLUMN,
    FinancialFeatureEngineer,
    build_preprocessor,
)

RANDOM_STATE = 42

TRAIN_START = pd.Timestamp("2024-01-01")
TRAIN_END = pd.Timestamp("2024-10-01")
VALIDATION_START = pd.Timestamp("2024-10-01")
CALIBRATION_END = pd.Timestamp("2024-12-01")
THRESHOLD_START = pd.Timestamp("2024-12-01")
VALIDATION_END = pd.Timestamp("2025-01-01")
TEST_START = pd.Timestamp("2025-01-01")
TEST_END = pd.Timestamp("2025-04-28")

MODEL_COMPLEXITY = {
    "Logistic Regression": 1,
    "Decision Tree": 2,
    "Linear SVM": 2,
    "Gradient Boosting": 3,
    "LightGBM": 3,
    "XGBoost": 3,
    "Random Forest": 4,
    "Extra Trees": 4,
}


@dataclass(frozen=True)
class TemporalPartitions:
    """Chronological development, calibration, threshold, and test data."""

    train: pd.DataFrame
    validation: pd.DataFrame
    calibration: pd.DataFrame
    threshold_selection: pd.DataFrame
    test: pd.DataFrame
    split_summary: pd.DataFrame


@dataclass(frozen=True)
class CandidateConfiguration:
    """One model, preprocessing, feature, and hyperparameter configuration."""

    model_name: str
    feature_configuration: str
    model_parameters: dict[str, Any]
    scaling: str = "robust"


def load_temporal_partitions(data_path: Path | str) -> TemporalPartitions:
    """Load labelled loans and create an untouched recent out-of-time test."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {path.resolve()}")

    columns = [*SAFE_RAW_FEATURES, TARGET_COLUMN]
    data = pd.read_csv(path, usecols=columns, low_memory=False)
    data["loan_issued_at"] = pd.to_datetime(
        data["loan_issued_at"],
        errors="coerce",
        format="mixed",
    )
    data = data[data[TARGET_COLUMN].notna()].copy()
    target_values = set(data[TARGET_COLUMN].unique())
    if not target_values.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(
            f"{TARGET_COLUMN} must contain only 0, 1, or missing values."
        )
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype("int8")
    data = data[data["loan_issued_at"].notna()].sort_values("loan_issued_at")

    train = data[
        (data["loan_issued_at"] >= TRAIN_START) & (data["loan_issued_at"] < TRAIN_END)
    ].copy()
    validation = data[
        (data["loan_issued_at"] >= VALIDATION_START)
        & (data["loan_issued_at"] < VALIDATION_END)
    ].copy()
    calibration = validation[validation["loan_issued_at"] < CALIBRATION_END].copy()
    threshold_selection = validation[
        validation["loan_issued_at"] >= THRESHOLD_START
    ].copy()
    test = data[
        (data["loan_issued_at"] >= TEST_START) & (data["loan_issued_at"] < TEST_END)
    ].copy()

    partitions = {
        "training": train,
        "validation": validation,
        "calibration_oct_nov": calibration,
        "threshold_december": threshold_selection,
        "untouched_test": test,
    }
    for name, partition in partitions.items():
        if partition.empty:
            raise ValueError(f"Temporal partition is empty: {name}")
        if partition[TARGET_COLUMN].nunique() != 2:
            raise ValueError(f"Both target classes are required in: {name}")

    if train["loan_issued_at"].max() >= validation["loan_issued_at"].min():
        raise ValueError("Training and validation dates overlap.")
    if validation["loan_issued_at"].max() >= test["loan_issued_at"].min():
        raise ValueError("Validation and test dates overlap.")
    if set(train.index) & set(validation.index):
        raise ValueError("Training and validation rows overlap.")
    if set(validation.index) & set(test.index):
        raise ValueError("Validation and test rows overlap.")

    summary_rows = []
    for name, partition in partitions.items():
        summary_rows.append(
            {
                "partition": name,
                "start_date": partition["loan_issued_at"].min(),
                "end_date": partition["loan_issued_at"].max(),
                "rows": len(partition),
                "defaults": int(partition[TARGET_COLUMN].sum()),
                "default_rate": float(partition[TARGET_COLUMN].mean()),
            }
        )

    return TemporalPartitions(
        train=train,
        validation=validation,
        calibration=calibration,
        threshold_selection=threshold_selection,
        test=test,
        split_summary=pd.DataFrame(summary_rows),
    )


def rolling_monthly_folds(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create three expanding chronological folds within Jan–Sep 2024."""
    dates = train["loan_issued_at"]
    boundaries = [
        (pd.Timestamp("2024-04-01"), pd.Timestamp("2024-06-01")),
        (pd.Timestamp("2024-06-01"), pd.Timestamp("2024-08-01")),
        (pd.Timestamp("2024-08-01"), pd.Timestamp("2024-10-01")),
    ]
    folds = []
    for validation_start, validation_end in boundaries:
        train_positions = np.flatnonzero(
            ((dates >= TRAIN_START) & (dates < validation_start)).to_numpy()
        )
        validation_positions = np.flatnonzero(
            ((dates >= validation_start) & (dates < validation_end)).to_numpy()
        )
        if not len(train_positions) or not len(validation_positions):
            raise ValueError("A rolling temporal fold is empty.")
        folds.append((train_positions, validation_positions))
    return folds


def positive_class_ratio(y: pd.Series) -> float:
    """Return negative-to-positive class ratio."""
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    if positives == 0:
        raise ValueError("Positive class is absent.")
    return negatives / positives


def model_family(model_name: str) -> str:
    """Map a model name to its preprocessing family."""
    if model_name == "Logistic Regression":
        return "linear"
    if model_name == "Linear SVM":
        return "svm"
    return "tree"


def _classifier(
    model_name: str,
    parameters: dict[str, Any],
    class_ratio: float,
) -> Any:
    params = dict(parameters)
    if "scale_pos_weight" in params:
        ratio_policy = params["scale_pos_weight"]
        ratio_values = {
            "none": 1.0,
            "sqrt_ratio": float(np.sqrt(class_ratio)),
            "full_ratio": float(class_ratio),
        }
        params["scale_pos_weight"] = ratio_values.get(
            ratio_policy,
            ratio_policy,
        )
    if model_name == "Logistic Regression":
        defaults = {
            "C": 1.0,
            "class_weight": "balanced",
            "solver": "liblinear",
            "max_iter": 2000,
            "random_state": RANDOM_STATE,
        }
        return LogisticRegression(**(defaults | params))
    if model_name == "Decision Tree":
        defaults = {
            "max_depth": 8,
            "min_samples_leaf": 100,
            "class_weight": "balanced",
            "random_state": RANDOM_STATE,
        }
        return DecisionTreeClassifier(**(defaults | params))
    if model_name == "Random Forest":
        defaults = {
            "n_estimators": 300,
            "max_depth": 18,
            "min_samples_leaf": 20,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
        }
        return RandomForestClassifier(**(defaults | params))
    if model_name == "Gradient Boosting":
        defaults = {
            "n_estimators": 250,
            "learning_rate": 0.05,
            "max_depth": 3,
            "min_samples_leaf": 50,
            "subsample": 0.8,
            "random_state": RANDOM_STATE,
        }
        return GradientBoostingClassifier(**(defaults | params))
    if model_name == "Linear SVM":
        defaults = {
            "C": 0.1,
            "class_weight": "balanced",
            "dual": "auto",
            "max_iter": 5000,
            "random_state": RANDOM_STATE,
        }
        return LinearSVC(**(defaults | params))
    if model_name == "Extra Trees":
        defaults = {
            "n_estimators": 300,
            "max_depth": 20,
            "min_samples_leaf": 10,
            "max_features": "sqrt",
            "class_weight": "balanced",
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
        }
        return ExtraTreesClassifier(**(defaults | params))
    if model_name == "XGBoost":
        defaults = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 4,
            "min_child_weight": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "scale_pos_weight": np.sqrt(class_ratio),
            "tree_method": "hist",
            "eval_metric": "logloss",
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
            "verbosity": 0,
        }
        return XGBClassifier(**(defaults | params))
    if model_name == "LightGBM":
        defaults = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "min_child_samples": 50,
            "subsample": 0.8,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "scale_pos_weight": np.sqrt(class_ratio),
            "n_jobs": -1,
            "random_state": RANDOM_STATE,
            "verbosity": -1,
        }
        return LGBMClassifier(**(defaults | params))
    raise KeyError(f"Unsupported model: {model_name}")


def build_optimized_pipeline(
    candidate: CandidateConfiguration,
    class_ratio: float,
) -> Pipeline:
    """Build an end-to-end pipeline for one controlled experiment."""
    feature_config = FEATURE_CONFIGURATIONS[candidate.feature_configuration]
    feature_engineer = FinancialFeatureEngineer(
        feature_set=feature_config.feature_set,
        engineered=feature_config.engineered,
        seasonality=feature_config.seasonality,
        include_country=feature_config.include_country,
    )
    preprocessor = build_preprocessor(
        feature_engineer=feature_engineer,
        model_family=model_family(candidate.model_name),
        scaling=candidate.scaling,
    )
    classifier = _classifier(
        model_name=candidate.model_name,
        parameters=candidate.model_parameters,
        class_ratio=class_ratio,
    )
    return Pipeline(
        steps=[
            ("feature_engineering", feature_engineer),
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def evaluate_candidate_cv(
    candidate: CandidateConfiguration,
    train: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate one configuration on identical expanding temporal folds."""
    X = train[SAFE_RAW_FEATURES]
    y = train[TARGET_COLUMN]
    fold_rows = []

    for fold_number, (train_positions, validation_positions) in enumerate(
        folds,
        start=1,
    ):
        fold_target = y.iloc[train_positions]
        pipeline = build_optimized_pipeline(
            candidate,
            positive_class_ratio(fold_target),
        )
        start = perf_counter()
        pipeline.fit(
            X.iloc[train_positions],
            fold_target,
        )
        fit_seconds = perf_counter() - start
        scores, score_type = extract_model_scores(
            pipeline,
            X.iloc[validation_positions],
        )
        metrics = ranking_metrics(y.iloc[validation_positions], scores)
        fold_rows.append(
            {
                "fold": fold_number,
                "train_rows": len(train_positions),
                "validation_rows": len(validation_positions),
                "score_type": score_type,
                "fit_seconds": fit_seconds,
                **metrics,
            }
        )

    fold_table = pd.DataFrame(fold_rows)
    summary = {
        "model": candidate.model_name,
        "feature_configuration": candidate.feature_configuration,
        "scaling": candidate.scaling,
        "parameters": json.dumps(
            candidate.model_parameters,
            sort_keys=True,
            default=str,
        ),
        "cv_roc_auc": float(fold_table["roc_auc"].mean()),
        "cv_roc_auc_std": float(fold_table["roc_auc"].std(ddof=0)),
        "cv_pr_auc": float(fold_table["pr_auc"].mean()),
        "cv_pr_auc_std": float(fold_table["pr_auc"].std(ddof=0)),
        "cv_pr_auc_lift": float(fold_table["pr_auc_lift"].mean()),
        "cv_ranking_score": float(fold_table["ranking_score"].mean()),
        "cv_ranking_std": float(fold_table["ranking_score"].std(ddof=0)),
        "stability_adjusted_score": float(
            fold_table["ranking_score"].mean()
            - 0.10 * fold_table["ranking_score"].std(ddof=0)
        ),
        "mean_fit_seconds": float(fold_table["fit_seconds"].mean()),
    }
    return summary, fold_table


def screening_candidates(
    feature_configuration: str,
) -> list[CandidateConfiguration]:
    """Return sensible fixed settings for broad algorithm screening."""
    model_names = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Gradient Boosting",
        "Linear SVM",
        "Extra Trees",
        "XGBoost",
        "LightGBM",
    ]
    return [
        CandidateConfiguration(
            model_name=name,
            feature_configuration=feature_configuration,
            model_parameters={},
            scaling="robust",
        )
        for name in model_names
    ]


def feature_screening_candidates() -> list[CandidateConfiguration]:
    """Compare raw, engineered, pricing, rating, country, and seasonality sets."""
    return [
        CandidateConfiguration(
            model_name="LightGBM",
            feature_configuration=config_name,
            model_parameters={},
        )
        for config_name in FEATURE_CONFIGURATIONS
    ]


def parameter_space(
    model_name: str,
    class_ratio: float,
) -> dict[str, list[Any]]:
    """Return bounded, deployment-conscious hyperparameter options."""
    # Resolve these policies from each fold's training labels during CV.
    ratio_options = ["none", "sqrt_ratio", "full_ratio"]
    spaces = {
        "Logistic Regression": {
            "C": [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            "class_weight": [None, "balanced"],
            # Scikit-learn 1.8+ expresses L1/L2 through l1_ratio.
            "l1_ratio": [0.0, 1.0],
        },
        "Decision Tree": {
            "max_depth": [4, 6, 8, 12, 16, None],
            "min_samples_leaf": [20, 50, 100, 250, 500],
            "min_samples_split": [50, 100, 250, 500],
            "class_weight": [None, "balanced"],
            "criterion": ["gini", "entropy", "log_loss"],
        },
        "Random Forest": {
            "n_estimators": [300, 500, 700],
            "max_depth": [10, 14, 18, 24, None],
            "min_samples_leaf": [5, 10, 20, 50],
            "max_features": ["sqrt", 0.5, 0.8],
            "class_weight": [None, "balanced", "balanced_subsample"],
        },
        "Gradient Boosting": {
            "n_estimators": [150, 250, 400, 600],
            "learning_rate": [0.02, 0.04, 0.07, 0.1],
            "max_depth": [2, 3, 4, 5],
            "min_samples_leaf": [20, 50, 100, 200],
            "subsample": [0.65, 0.8, 1.0],
            "max_features": [None, "sqrt", 0.7],
        },
        "Linear SVM": {
            "C": [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            "class_weight": [None, "balanced"],
            "loss": ["hinge", "squared_hinge"],
        },
        "Extra Trees": {
            "n_estimators": [300, 500, 700],
            "max_depth": [10, 14, 18, 24, None],
            "min_samples_leaf": [2, 5, 10, 20, 50],
            "max_features": ["sqrt", 0.5, 0.8, 1.0],
            "class_weight": [None, "balanced"],
        },
        "XGBoost": {
            "n_estimators": [300, 500, 800, 1100],
            "learning_rate": [0.02, 0.04, 0.07, 0.1],
            "max_depth": [2, 3, 4, 5, 6],
            "min_child_weight": [5, 10, 20, 50],
            "subsample": [0.65, 0.8, 1.0],
            "colsample_bytree": [0.65, 0.8, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5, 1.0],
            "reg_lambda": [0.5, 1.0, 3.0, 10.0],
            "scale_pos_weight": ratio_options,
        },
        "LightGBM": {
            "n_estimators": [300, 500, 800, 1100],
            "learning_rate": [0.02, 0.04, 0.07, 0.1],
            "num_leaves": [15, 31, 47, 63],
            "max_depth": [-1, 4, 6, 8, 10],
            "min_child_samples": [20, 50, 100, 200],
            "subsample": [0.65, 0.8, 1.0],
            "colsample_bytree": [0.65, 0.8, 1.0],
            "reg_alpha": [0.0, 0.1, 0.5, 1.0],
            "reg_lambda": [0.0, 1.0, 3.0, 10.0],
            "scale_pos_weight": ratio_options,
        },
    }
    return spaces[model_name]


def tuning_candidates(
    model_name: str,
    feature_configurations: list[str],
    class_ratio: float,
    iterations: int,
) -> list[CandidateConfiguration]:
    """Sample deterministic hyperparameters across the strongest feature sets."""
    sampled = list(
        ParameterSampler(
            parameter_space(model_name, class_ratio),
            n_iter=iterations,
            random_state=RANDOM_STATE,
        )
    )
    candidates = []
    for index, parameters in enumerate(sampled):
        feature_name = feature_configurations[index % len(feature_configurations)]
        scaling = (
            "standard"
            if (model_name in {"Logistic Regression", "Linear SVM"} and index % 2)
            else "robust"
        )
        candidates.append(
            CandidateConfiguration(
                model_name=model_name,
                feature_configuration=feature_name,
                model_parameters=parameters,
                scaling=scaling,
            )
        )
    return candidates


def candidate_to_dict(candidate: CandidateConfiguration) -> dict[str, Any]:
    """Return JSON-compatible configuration metadata."""
    return asdict(candidate)
