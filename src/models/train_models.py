"""Training utilities for the three default-risk baseline models."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .data_preparation import OriginationFeatureTransformer, RANDOM_STATE

MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression_pipeline.joblib",
    "Random Forest": "random_forest_pipeline.joblib",
    "Gradient Boosting": "gradient_boosting_pipeline.joblib",
}


def create_classifiers() -> dict[str, object]:
    """Return the three configured baseline classifiers."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE,
        ),
    }


def build_model_pipeline(
    preprocessor: ColumnTransformer,
    classifier: object,
) -> Pipeline:
    """Combine feature creation, preprocessing, and one classifier."""
    return Pipeline(
        steps=[
            ("feature_builder", OriginationFeatureTransformer()),
            ("preprocessor", clone(preprocessor)),
            ("classifier", classifier),
        ]
    )


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor: ColumnTransformer,
    models_dir: Path | str,
) -> dict[str, Pipeline]:
    """Fit and persist every baseline pipeline on the same training data."""
    output_dir = Path(models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fitted_models: dict[str, Pipeline] = {}
    for model_name, classifier in create_classifiers().items():
        print(f"Training {model_name}...")
        pipeline = build_model_pipeline(preprocessor, classifier)
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, output_dir / MODEL_FILENAMES[model_name])
        fitted_models[model_name] = pipeline
        print(f"Completed {model_name}.")

    return fitted_models
