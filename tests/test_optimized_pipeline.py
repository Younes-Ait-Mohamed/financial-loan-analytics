"""Tests for leakage guards, engineered features, thresholds, and inference."""

from __future__ import annotations

import numpy as np
import pandas as pd
from src.models.optimized_evaluation import (
    ScoreCalibrator,
    operational_metrics,
    select_threshold_policies,
    threshold_analysis,
)
from src.models.optimized_features import (
    POST_ORIGINATION_EXCLUSIONS,
    SAFE_RAW_FEATURES,
    FinancialFeatureEngineer,
)


def sample_loans() -> pd.DataFrame:
    """Return two valid origination-time records."""
    return pd.DataFrame(
        {
            "country": ["Finland", "Netherlands"],
            "loan_issued_at": ["2024-03-15", "2024-08-01"],
            "issued_amount": [2_000.0, 5_000.0],
            "initial_interest_rate": [0.18, 0.10],
            "initial_loan_duration": [36, 60],
            "combined_income": [2_500.0, 4_000.0],
            "customer_risk_rating": ["C", "A"],
        }
    )


def test_post_origination_payments_are_excluded() -> None:
    """The known leakage field must never enter the optimized schema."""
    assert set(SAFE_RAW_FEATURES).isdisjoint(POST_ORIGINATION_EXCLUSIONS)
    assert "nr_of_payments" in POST_ORIGINATION_EXCLUSIONS
    assert "nr_of_payments" not in SAFE_RAW_FEATURES
    assert "has_default_within_12_months" not in SAFE_RAW_FEATURES


def test_financial_features_are_created_without_target() -> None:
    """Financial ratios must use only source-verified origination values."""
    transformer = FinancialFeatureEngineer(
        feature_set="stacked",
        engineered=True,
        seasonality=True,
    )
    transformed = transformer.fit_transform(sample_loans())

    assert "loan_to_income_ratio" in transformed.columns
    assert "estimated_monthly_payment" in transformed.columns
    assert "payment_to_income_ratio" in transformed.columns
    assert "has_default_within_12_months" not in transformed.columns
    assert np.isfinite(transformed["estimated_monthly_payment"]).all()
    assert (transformed["estimated_monthly_payment"] > 0).all()


def test_missing_income_does_not_become_zero() -> None:
    """Missing values remain missing until train-fitted preprocessing."""
    loans = sample_loans()
    loans.loc[0, "combined_income"] = np.nan
    transformer = FinancialFeatureEngineer(feature_set="offer")
    transformed = transformer.fit_transform(loans)

    assert pd.isna(transformed.loc[0, "combined_income"])
    assert pd.isna(transformed.loc[0, "payment_to_income_ratio"])


def test_threshold_policies_return_valid_tradeoffs() -> None:
    """Every reported operating threshold must be bounded and reproducible."""
    target = np.array([0, 0, 0, 1, 1, 1], dtype="int8")
    probabilities = np.array([0.05, 0.20, 0.40, 0.35, 0.70, 0.90])
    table = threshold_analysis(target, probabilities)
    policies = select_threshold_policies(table)

    assert "balanced_composite" in policies.index
    assert policies["threshold"].between(0, 1).all()
    assert policies["recall"].between(0, 1).all()


def test_sigmoid_calibrator_returns_probabilities() -> None:
    """Decision scores must be transformable into bounded probabilities."""
    scores = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    target = np.array([0, 0, 0, 1, 1])
    calibrator = ScoreCalibrator(method="sigmoid").fit(scores, target)
    probabilities = calibrator.transform(scores)

    assert np.all(probabilities > 0)
    assert np.all(probabilities < 1)
    assert np.all(np.diff(probabilities) >= 0)
    assert probabilities[-1] > probabilities[0]


def test_operational_metrics_include_all_confusion_counts() -> None:
    """The confusion matrix must reconcile exactly with the sample size."""
    target = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.2, 0.9])
    metrics = operational_metrics(target, probabilities, threshold=0.5)
    total = (
        metrics["true_negatives"]
        + metrics["false_positives"]
        + metrics["false_negatives"]
        + metrics["true_positives"]
    )

    assert total == len(target)
    assert metrics["support"] == len(target)
