"""Data loading and leakage-safe preparation for default-risk models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "has_default_within_12_months"
RANDOM_STATE = 42
TEST_SIZE = 0.20

RAW_MODEL_FEATURES = [
    "country",
    "loan_issued_at",
    "issued_amount",
    "initial_interest_rate",
    "initial_loan_duration",
    "combined_income",
    "customer_risk_rating",
]

FINAL_MODEL_FEATURES = [
    "country",
    "issued_amount",
    "initial_interest_rate",
    "initial_loan_duration",
    "combined_income",
    "customer_risk_rating",
    "loan_issue_year",
    "loan_issue_month",
]

LEAKAGE_AND_IDENTIFIER_COLUMNS = [
    "loan_id",
    "loan_status",
    "is_default",
    TARGET_COLUMN,
    "early_repaid_at",
    "is_early_repaid_within_14_days",
    "loan_last_recorded_action_date_local",
    "nr_of_payments",
    "principal_balance",
    "principal_debt",
    "principal_paid_total",
    "interest_paid_total",
    "extra_interest_paid_total",
    "late_fee_paid_total",
    "maintenance_fee_paid_total",
    "next_payment_nr",
    "next_payment_date_local",
    "debt_occured_date_local",
    "days_past_due_principal",
    "months_in_default",
    "months_on_book",
    "loan_status_risk",
    "repaid_amount_total",
]

ADDITIONAL_EXCLUDED_COLUMNS = [
    "projected_npv_return",
]


@dataclass(frozen=True)
class PreparedData:
    """Train/test data and feature metadata used by all classifiers."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessor: ColumnTransformer
    numerical_features: list[str]
    categorical_features: list[str]
    target_distribution: pd.DataFrame
    excluded_columns: list[str]


class OriginationFeatureTransformer(BaseEstimator, TransformerMixin):
    """Select origination fields and derive issuance calendar features."""

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "OriginationFeatureTransformer":
        self._validate_input(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(X)
        transformed = X.loc[:, RAW_MODEL_FEATURES].copy()
        issue_dates = pd.to_datetime(
            transformed.pop("loan_issued_at"),
            errors="coerce",
            format="mixed",
        )
        transformed["loan_issue_year"] = issue_dates.dt.year.astype("float64")
        transformed["loan_issue_month"] = issue_dates.dt.month.astype("float64")
        return transformed.loc[:, FINAL_MODEL_FEATURES]

    def get_feature_names_out(
        self,
        input_features: list[str] | None = None,
    ) -> np.ndarray:
        return np.asarray(FINAL_MODEL_FEATURES, dtype=object)

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Model input must be a pandas DataFrame.")
        missing = sorted(set(RAW_MODEL_FEATURES) - set(X.columns))
        if missing:
            raise ValueError(
                "Missing required origination columns: " + ", ".join(missing)
            )


def load_cleaned_data(data_path: Path | str) -> pd.DataFrame:
    """Load the cleaned CSV without modifying it."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {path.resolve()}")
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, received: {path.suffix}")

    data = pd.read_csv(path, low_memory=False)
    if data.empty:
        raise ValueError("The cleaned dataset is empty.")
    return data


def validate_target(data: pd.DataFrame) -> None:
    """Confirm that the target exists and its observed values are binary."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' was not found.")

    observed_values = set(data[TARGET_COLUMN].dropna().unique().tolist())
    if not observed_values:
        raise ValueError(f"Target column '{TARGET_COLUMN}' has no observed values.")
    if not observed_values.issubset({0, 1, 0.0, 1.0}):
        raise ValueError(
            f"Target '{TARGET_COLUMN}' must be binary; found {observed_values}."
        )


def build_preprocessor(
    numerical_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Create train-fitted numerical and categorical preprocessing."""
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numerical_pipeline, numerical_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def prepare_data(
    data_path: Path | str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> PreparedData:
    """Load, validate, select origination features, and split the data."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    data = load_cleaned_data(data_path)
    validate_target(data)

    modelling_data = data.loc[data[TARGET_COLUMN].notna()].copy()
    if modelling_data.empty:
        raise ValueError("No rows remain after removing missing targets.")

    missing_features = sorted(set(RAW_MODEL_FEATURES) - set(data.columns))
    if missing_features:
        raise ValueError(
            "Required model features are missing: " + ", ".join(missing_features)
        )

    X = modelling_data.loc[:, RAW_MODEL_FEATURES].copy()
    y = modelling_data[TARGET_COLUMN].astype("int8")

    class_counts = y.value_counts().sort_index()
    if len(class_counts) != 2:
        raise ValueError("Both target classes must be present for training.")

    distribution = pd.DataFrame(
        {
            "class": class_counts.index.astype(int),
            "count": class_counts.values,
            "proportion": (class_counts.values / len(y)),
        }
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    feature_builder = OriginationFeatureTransformer()
    derived_train = feature_builder.fit_transform(X_train)
    numerical_features = derived_train.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    categorical_features = derived_train.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()

    identified = set(numerical_features + categorical_features)
    unidentified = sorted(set(derived_train.columns) - identified)
    if unidentified:
        raise TypeError(
            "Unsupported feature dtypes for columns: " + ", ".join(unidentified)
        )

    excluded_columns = sorted(
        set(data.columns) - set(RAW_MODEL_FEATURES) - {TARGET_COLUMN}
    )
    preprocessor = build_preprocessor(
        numerical_features=numerical_features,
        categorical_features=categorical_features,
    )

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        target_distribution=distribution,
        excluded_columns=excluded_columns,
    )


def format_target_distribution(distribution: pd.DataFrame) -> str:
    """Return a concise printable target-distribution table."""
    printable = distribution.copy()
    printable["proportion"] = printable["proportion"].map(lambda value: f"{value:.2%}")
    return printable.to_string(index=False)
