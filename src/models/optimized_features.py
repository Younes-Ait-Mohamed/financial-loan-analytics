"""Leakage-safe financial feature engineering for optimized default models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

TARGET_COLUMN = "has_default_within_12_months"

# Every field below is explicitly described by the source dictionary as known
# when the loan is issued. Lifecycle fields such as nr_of_payments are excluded.
SAFE_RAW_FEATURES = [
    "country",
    "loan_issued_at",
    "issued_amount",
    "initial_interest_rate",
    "initial_loan_duration",
    "combined_income",
    "customer_risk_rating",
]

POST_ORIGINATION_EXCLUSIONS = [
    "loan_id",
    "early_repaid_at",
    "is_early_repaid_within_14_days",
    "loan_status",
    "loan_last_recorded_action_date_local",
    "nr_of_payments",
    "principal_balance",
    "principal_debt",
    "principal_paid_total",
    "interest_paid_total",
    "extra_interest_paid_total",
    "late_fee_paid_total",
    "maintenance_fee_paid_total",
    "is_default",
    "next_payment_nr",
    "next_payment_date_local",
    "debt_occured_date_local",
    "days_past_due_principal",
    "months_in_default",
    "months_on_book",
    "loan_status_risk",
    "repaid_amount_total",
    TARGET_COLUMN,
]

CONSERVATIVE_EXCLUSIONS = ["projected_npv_return"]


@dataclass(frozen=True)
class FeatureConfiguration:
    """Controlled feature-set experiment configuration."""

    name: str
    feature_set: str
    engineered: bool
    seasonality: bool
    include_country: bool = True


FEATURE_CONFIGURATIONS = {
    "application_raw": FeatureConfiguration(
        name="application_raw",
        feature_set="application",
        engineered=False,
        seasonality=False,
    ),
    "application_engineered": FeatureConfiguration(
        name="application_engineered",
        feature_set="application",
        engineered=True,
        seasonality=True,
    ),
    "offer_engineered": FeatureConfiguration(
        name="offer_engineered",
        feature_set="offer",
        engineered=True,
        seasonality=True,
    ),
    "stacked_engineered": FeatureConfiguration(
        name="stacked_engineered",
        feature_set="stacked",
        engineered=True,
        seasonality=True,
    ),
    "stacked_no_country": FeatureConfiguration(
        name="stacked_no_country",
        feature_set="stacked",
        engineered=True,
        seasonality=True,
        include_country=False,
    ),
    "stacked_no_seasonality": FeatureConfiguration(
        name="stacked_no_seasonality",
        feature_set="stacked",
        engineered=True,
        seasonality=False,
    ),
}


def required_raw_features(
    configuration: FeatureConfiguration,
) -> list[str]:
    """Return the minimum input schema for one feature configuration."""
    columns = [
        "loan_issued_at",
        "issued_amount",
        "initial_loan_duration",
        "combined_income",
    ]
    if configuration.include_country:
        columns.insert(0, "country")
    if configuration.feature_set in {"offer", "stacked"}:
        columns.append("initial_interest_rate")
    if configuration.feature_set == "stacked":
        columns.append("customer_risk_rating")
    return columns


def _safe_ratio(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Divide only by finite positive denominators."""
    valid_denominator = denominator.where(
        denominator.notna() & np.isfinite(denominator) & (denominator > 0)
    )
    result = numerator / valid_denominator
    return result.replace([np.inf, -np.inf], np.nan)


def _estimated_annuity_payment(
    principal: pd.Series,
    annual_rate: pd.Series,
    duration_months: pd.Series,
) -> pd.Series:
    """Estimate a fixed monthly annuity payment from origination terms."""
    principal_values = principal.astype("float64").to_numpy()
    annual_values = annual_rate.astype("float64").to_numpy()
    duration_values = duration_months.astype("float64").to_numpy()

    monthly_rate = annual_values / 12.0
    valid = (
        np.isfinite(principal_values)
        & np.isfinite(monthly_rate)
        & np.isfinite(duration_values)
        & (principal_values >= 0)
        & (monthly_rate >= 0)
        & (duration_values > 0)
    )
    payment = np.full(len(principal_values), np.nan, dtype="float64")

    zero_rate = valid & np.isclose(monthly_rate, 0.0)
    payment[zero_rate] = principal_values[zero_rate] / duration_values[zero_rate]

    positive_rate = valid & (monthly_rate > 0)
    growth = np.power(
        1.0 + monthly_rate[positive_rate],
        duration_values[positive_rate],
    )
    payment[positive_rate] = (
        principal_values[positive_rate]
        * monthly_rate[positive_rate]
        * growth
        / (growth - 1.0)
    )
    return pd.Series(payment, index=principal.index)


class FinancialFeatureEngineer(BaseEstimator, TransformerMixin):
    """Create source-verified features without learning from future outcomes."""

    VALID_FEATURE_SETS = {"application", "offer", "stacked"}

    def __init__(
        self,
        feature_set: str = "stacked",
        engineered: bool = True,
        seasonality: bool = True,
        include_country: bool = True,
    ) -> None:
        self.feature_set = feature_set
        self.engineered = engineered
        self.seasonality = seasonality
        self.include_country = include_country

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
    ) -> "FinancialFeatureEngineer":
        self._validate_input(X)
        self.feature_names_out_ = self._transform_frame(X.head(1)).columns.to_numpy()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(X)
        return self._transform_frame(X)

    def get_feature_names_out(
        self,
        input_features: list[str] | None = None,
    ) -> np.ndarray:
        if not hasattr(self, "feature_names_out_"):
            raise AttributeError("The feature engineer must be fitted first.")
        return np.asarray(self.feature_names_out_, dtype=object)

    def categorical_features(self) -> list[str]:
        """Return the engineered categorical column names."""
        columns = []
        if self.include_country:
            columns.append("country")
        if self.feature_set == "stacked":
            columns.append("customer_risk_rating")
        if self.engineered:
            columns.append("duration_band")
        return columns

    def numerical_features(self) -> list[str]:
        """Return the engineered numerical column names."""
        columns = [
            "issued_amount",
            "initial_loan_duration",
            "combined_income",
        ]
        if self.feature_set in {"offer", "stacked"}:
            columns.append("initial_interest_rate")

        if self.engineered:
            columns.extend(
                [
                    "log_issued_amount",
                    "log_combined_income",
                    "loan_to_income_ratio",
                    "amount_per_contract_month",
                    "income_nonpositive",
                ]
            )
            if self.feature_set in {"offer", "stacked"}:
                columns.extend(
                    [
                        "log_interest_rate",
                        "estimated_monthly_payment",
                        "payment_to_income_ratio",
                        "estimated_total_interest",
                        "total_repayment_to_annual_income",
                    ]
                )
        if self.seasonality:
            columns.extend(["issue_month_sin", "issue_month_cos"])
        return columns

    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Model input must be a pandas DataFrame.")
        if self.feature_set not in self.VALID_FEATURE_SETS:
            valid = ", ".join(sorted(self.VALID_FEATURE_SETS))
            raise ValueError(f"feature_set must be one of: {valid}")

        required = {
            "loan_issued_at",
            "issued_amount",
            "initial_loan_duration",
            "combined_income",
        }
        if self.include_country:
            required.add("country")
        if self.feature_set in {"offer", "stacked"}:
            required.add("initial_interest_rate")
        if self.feature_set == "stacked":
            required.add("customer_risk_rating")

        missing = sorted(required - set(X.columns))
        if missing:
            raise ValueError(
                "Missing required origination columns: " + ", ".join(missing)
            )

    def _transform_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(index=X.index)

        if self.include_country:
            output["country"] = X["country"].astype("object")
        if self.feature_set == "stacked":
            output["customer_risk_rating"] = X["customer_risk_rating"].astype("object")

        amount = pd.to_numeric(X["issued_amount"], errors="coerce")
        duration = pd.to_numeric(X["initial_loan_duration"], errors="coerce").astype(
            "float64"
        )
        income = pd.to_numeric(X["combined_income"], errors="coerce")

        output["issued_amount"] = amount
        output["initial_loan_duration"] = duration
        output["combined_income"] = income

        interest = None
        if self.feature_set in {"offer", "stacked"}:
            interest = pd.to_numeric(X["initial_interest_rate"], errors="coerce")
            output["initial_interest_rate"] = interest

        if self.engineered:
            output["duration_band"] = pd.cut(
                duration,
                bins=[-np.inf, 12, 36, 60, np.inf],
                labels=["up_to_12m", "13_to_36m", "37_to_60m", "over_60m"],
            ).astype("object")
            output["log_issued_amount"] = np.log1p(amount.clip(lower=0))
            output["log_combined_income"] = np.log1p(income.where(income > 0))
            output["loan_to_income_ratio"] = _safe_ratio(amount, income)
            output["amount_per_contract_month"] = _safe_ratio(amount, duration)
            output["income_nonpositive"] = (income.notna() & (income <= 0)).astype(
                "int8"
            )

            if interest is not None:
                payment = _estimated_annuity_payment(
                    principal=amount,
                    annual_rate=interest,
                    duration_months=duration,
                )
                total_repayment = payment * duration
                output["log_interest_rate"] = np.log1p(interest.clip(lower=0))
                output["estimated_monthly_payment"] = payment
                output["payment_to_income_ratio"] = _safe_ratio(payment, income)
                output["estimated_total_interest"] = (total_repayment - amount).clip(
                    lower=0
                )
                output["total_repayment_to_annual_income"] = _safe_ratio(
                    total_repayment, income * 12.0
                )

        if self.seasonality:
            issue_dates = pd.to_datetime(
                X["loan_issued_at"],
                errors="coerce",
                format="mixed",
            )
            issue_month = issue_dates.dt.month.astype("float64")
            output["issue_month_sin"] = np.sin(2.0 * np.pi * issue_month / 12.0)
            output["issue_month_cos"] = np.cos(2.0 * np.pi * issue_month / 12.0)

        expected = self.categorical_features() + self.numerical_features()
        return output.loc[:, expected]


def build_preprocessor(
    feature_engineer: FinancialFeatureEngineer,
    model_family: str,
    scaling: str = "robust",
) -> ColumnTransformer:
    """Build model-specific preprocessing fitted only on training folds."""
    numerical_features = feature_engineer.numerical_features()
    categorical_features = feature_engineer.categorical_features()

    numerical_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ]
    is_linear = model_family in {"linear", "svm"}
    if is_linear:
        if scaling == "standard":
            numerical_steps.append(("scaler", StandardScaler()))
        elif scaling == "robust":
            numerical_steps.append(("scaler", RobustScaler()))
        else:
            raise ValueError("Linear scaling must be 'standard' or 'robust'.")

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Missing",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    min_frequency=20,
                    sparse_output=is_linear,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(steps=numerical_steps),
                numerical_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0 if is_linear else 0.0,
        verbose_feature_names_out=False,
    )
