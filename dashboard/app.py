"""Interactive Streamlit dashboard for the financial loan analytics project."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.optimized_evaluation import OptimizedRiskModel  # noqa: E402

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "data_cleaned.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "optimized" / "best_optimized_default_model_f1_policy.joblib"
FALLBACK_MODEL_PATH = PROJECT_ROOT / "models" / "optimized" / "best_optimized_default_model.joblib"
MODEL_RESULTS_PATH = (
    PROJECT_ROOT / "reports" / "optimized_model_results" / "weighted_finalist_results.csv"
)
IMPORTANCE_PATH = (
    PROJECT_ROOT
    / "reports"
    / "optimized_model_results"
    / "selected_model_permutation_importance.csv"
)

TARGET = "has_default_within_12_months"
CORE_COLUMNS = [
    "loan_id",
    "country",
    "loan_issued_at",
    "issued_amount",
    "loan_status",
    "initial_interest_rate",
    "initial_loan_duration",
    "combined_income",
    TARGET,
    "customer_risk_rating",
]
RATING_ORDER = ["AA", "A", "B", "C", "D", "E", "F", "G", "HR", "Missing"]
PAGE_OPTIONS = [
    "Portfolio Overview",
    "Risk Analysis",
    "Model Performance",
    "Loan Risk Scoring",
]
COLORS = {
    "navy": "#17324D",
    "blue": "#2563EB",
    "teal": "#0F9D8A",
    "amber": "#F59E0B",
    "red": "#DC2626",
    "slate": "#64748B",
    "light": "#F4F7FB",
}


st.set_page_config(
    page_title="AI Loan Analytics",
    page_icon="€",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #F4F7FB;
    }
    .block-container {
        max-width: 1500px;
        padding-top: 1.6rem;
        padding-bottom: 2rem;
    }
    [data-testid="stSidebar"] {
        background: #102A43;
    }
    [data-testid="stSidebar"] * {
        color: #F8FAFC;
    }
    [data-testid="stSidebar"] .stMultiSelect span,
    [data-testid="stSidebar"] .stDateInput input {
        color: #0F172A;
    }
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }
    .dashboard-title {
        color: #17324D;
        font-size: 2.15rem;
        font-weight: 750;
        letter-spacing: -0.03em;
        margin-bottom: 0.15rem;
    }
    .dashboard-subtitle {
        color: #64748B;
        font-size: 1rem;
        margin-bottom: 1.4rem;
    }
    .section-label {
        color: #17324D;
        font-size: 1.15rem;
        font-weight: 700;
        margin: 0.4rem 0 0.7rem 0;
    }
    .risk-card {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 1.2rem;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }
    .footer {
        color: #94A3B8;
        font-size: 0.82rem;
        text-align: center;
        padding-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Loading the loan portfolio...")
def load_portfolio(path: Path) -> pd.DataFrame:
    """Load only the columns required by the dashboard."""
    frame = pd.read_csv(
        path,
        usecols=CORE_COLUMNS,
        low_memory=False,
    )
    frame["loan_issued_at"] = pd.to_datetime(
        frame["loan_issued_at"],
        errors="coerce",
        format="mixed",
    )
    for column in ["issued_amount", "initial_interest_rate", TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["combined_income"] = pd.to_numeric(
        frame["combined_income"],
        errors="coerce",
    )
    frame["customer_risk_rating"] = frame["customer_risk_rating"].astype("string").fillna("Missing")
    frame["country"] = frame["country"].astype("string").fillna("Missing")
    frame["loan_status"] = frame["loan_status"].astype("string").fillna("Missing")
    frame["issue_month"] = frame["loan_issued_at"].dt.to_period("M").dt.to_timestamp()
    return frame


@st.cache_data(show_spinner=False)
def load_model_results(path: Path) -> pd.DataFrame:
    """Load the persisted model comparison."""
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_feature_importance(path: Path) -> pd.DataFrame:
    """Load the selected model's raw-feature permutation importance."""
    return pd.read_csv(path)


@st.cache_resource(show_spinner="Loading the risk model...")
def load_risk_model(path: Path) -> OptimizedRiskModel:
    """Load and cache the serialized end-to-end risk model."""
    return OptimizedRiskModel.load(path)


def page_header(title: str, subtitle: str) -> None:
    """Render a consistent page heading."""
    st.markdown(f'<div class="dashboard-title">{title}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="dashboard-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def euro(value: float, decimals: int = 0) -> str:
    """Format one value as euros."""
    if pd.isna(value):
        return "—"
    return f"€{value:,.{decimals}f}"


def percentage(value: float, decimals: int = 1) -> str:
    """Format one decimal fraction as a percentage."""
    if pd.isna(value):
        return "—"
    return f"{value:.{decimals}%}"


def apply_filters(
    frame: pd.DataFrame,
    date_range: tuple[date, date],
    countries: list[str],
    ratings: list[str],
    statuses: list[str],
) -> pd.DataFrame:
    """Apply sidebar filters without modifying the cached source frame."""
    start_date, end_date = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    mask = frame["loan_issued_at"].between(
        start_date,
        end_date + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1),
    )
    if countries:
        mask &= frame["country"].isin(countries)
    if ratings:
        mask &= frame["customer_risk_rating"].isin(ratings)
    if statuses:
        mask &= frame["loan_status"].isin(statuses)
    return frame.loc[mask].copy()


def render_portfolio_overview(frame: pd.DataFrame) -> None:
    """Render portfolio KPIs and composition charts."""
    page_header(
        "Portfolio Overview",
        "Monitor lending volume, exposure, portfolio composition, and observed outcomes.",
    )

    labelled = frame[frame[TARGET].notna()]
    total_loans = len(frame)
    total_issued = frame["issued_amount"].sum()
    average_amount = frame["issued_amount"].mean()
    average_rate = frame["initial_interest_rate"].mean()
    default_rate = labelled[TARGET].mean() if len(labelled) else np.nan

    columns = st.columns(5)
    columns[0].metric("Loans", f"{total_loans:,.0f}")
    columns[1].metric("Issued amount", euro(total_issued))
    columns[2].metric("Average loan", euro(average_amount))
    columns[3].metric("Average interest", percentage(average_rate, 2))
    columns[4].metric("12-month default rate", percentage(default_rate, 2))

    st.caption(
        f"Default rate uses {len(labelled):,} loans with a known 12-month outcome; "
        f"{frame[TARGET].isna().sum():,} filtered loans remain unlabelled."
    )

    monthly = (
        frame.dropna(subset=["issue_month"])
        .groupby("issue_month", as_index=False)
        .agg(
            loans=("loan_id", "size"),
            issued_amount=("issued_amount", "sum"),
        )
        .sort_values("issue_month")
    )

    st.markdown(
        '<div class="section-label">Portfolio development</div>',
        unsafe_allow_html=True,
    )
    volume_tab, exposure_tab = st.tabs(["Loan volume", "Issued amount"])
    with volume_tab:
        figure = px.area(
            monthly,
            x="issue_month",
            y="loans",
            labels={"issue_month": "Issuance month", "loans": "Loans"},
            color_discrete_sequence=[COLORS["blue"]],
        )
        figure.update_traces(line_width=2)
        figure.update_layout(
            template="plotly_white",
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(figure, width="stretch")
    with exposure_tab:
        figure = px.area(
            monthly,
            x="issue_month",
            y="issued_amount",
            labels={
                "issue_month": "Issuance month",
                "issued_amount": "Issued amount (€)",
            },
            color_discrete_sequence=[COLORS["teal"]],
        )
        figure.update_traces(line_width=2)
        figure.update_layout(
            template="plotly_white",
            height=360,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        figure.update_yaxes(tickprefix="€", separatethousands=True)
        st.plotly_chart(figure, width="stretch")

    left, right = st.columns(2)
    with left:
        country_summary = (
            frame.groupby("country", as_index=False, observed=True)
            .agg(
                loans=("loan_id", "size"),
                issued_amount=("issued_amount", "sum"),
            )
            .sort_values("issued_amount", ascending=True)
        )
        figure = px.bar(
            country_summary,
            x="issued_amount",
            y="country",
            orientation="h",
            title="Issued amount by country",
            labels={"issued_amount": "Issued amount (€)", "country": ""},
            color_discrete_sequence=[COLORS["blue"]],
            hover_data={"loans": ":,.0f"},
        )
        figure.update_layout(
            template="plotly_white",
            height=390,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        figure.update_xaxes(tickprefix="€", separatethousands=True)
        st.plotly_chart(figure, width="stretch")

    with right:
        status_summary = (
            frame["loan_status"]
            .value_counts(dropna=False)
            .rename_axis("loan_status")
            .reset_index(name="loans")
        )
        figure = px.pie(
            status_summary,
            names="loan_status",
            values="loans",
            title="Current loan-status mix",
            hole=0.55,
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        figure.update_traces(textposition="inside", textinfo="percent")
        figure.update_layout(
            template="plotly_white",
            height=390,
            margin=dict(l=10, r=10, t=50, b=10),
            legend_title_text="Status",
        )
        st.plotly_chart(figure, width="stretch")

    with st.expander("View a recent loan sample"):
        sample_columns = [
            "loan_id",
            "country",
            "loan_issued_at",
            "issued_amount",
            "initial_interest_rate",
            "initial_loan_duration",
            "customer_risk_rating",
            "loan_status",
        ]
        sample = frame.sort_values("loan_issued_at", ascending=False).head(250)
        st.dataframe(
            sample[sample_columns],
            width="stretch",
            hide_index=True,
            column_config={
                "loan_issued_at": st.column_config.DatetimeColumn(
                    "Issued at",
                    format="DD MMM YYYY",
                ),
                "issued_amount": st.column_config.NumberColumn(
                    "Issued amount",
                    format="€ %.2f",
                ),
                "initial_interest_rate": st.column_config.NumberColumn(
                    "Interest rate",
                    format="percent",
                ),
            },
        )


def render_risk_analysis(frame: pd.DataFrame) -> None:
    """Render outcome analysis on loans with a measurable target."""
    page_header(
        "Risk Analysis",
        "Explore observed 12-month default behaviour across portfolio segments.",
    )
    labelled = frame[frame[TARGET].notna()].copy()
    if labelled.empty:
        st.warning("No labelled loans remain under the selected filters.")
        return

    defaults = int(labelled[TARGET].sum())
    columns = st.columns(4)
    columns[0].metric("Labelled loans", f"{len(labelled):,.0f}")
    columns[1].metric("Observed defaults", f"{defaults:,.0f}")
    columns[2].metric("Default rate", percentage(labelled[TARGET].mean(), 2))
    columns[3].metric(
        "Average default exposure",
        euro(labelled.loc[labelled[TARGET].eq(1), "issued_amount"].mean()),
    )

    left, right = st.columns(2)
    with left:
        rating_summary = labelled.groupby(
            "customer_risk_rating",
            as_index=False,
            observed=True,
        ).agg(
            loans=(TARGET, "size"),
            defaults=(TARGET, "sum"),
            default_rate=(TARGET, "mean"),
        )
        rating_summary["customer_risk_rating"] = pd.Categorical(
            rating_summary["customer_risk_rating"],
            categories=RATING_ORDER,
            ordered=True,
        )
        rating_summary = rating_summary.sort_values("customer_risk_rating")
        figure = px.bar(
            rating_summary,
            x="customer_risk_rating",
            y="default_rate",
            title="Default rate by customer risk rating",
            labels={
                "customer_risk_rating": "Risk rating",
                "default_rate": "Default rate",
            },
            color="default_rate",
            color_continuous_scale="OrRd",
            hover_data={"loans": ":,.0f", "defaults": ":,.0f"},
        )
        figure.update_layout(
            template="plotly_white",
            height=390,
            margin=dict(l=10, r=10, t=50, b=10),
            coloraxis_showscale=False,
        )
        figure.update_yaxes(tickformat=".0%")
        st.plotly_chart(figure, width="stretch")

    with right:
        country_summary = (
            labelled.groupby("country", as_index=False, observed=True)
            .agg(
                loans=(TARGET, "size"),
                defaults=(TARGET, "sum"),
                default_rate=(TARGET, "mean"),
            )
            .sort_values("default_rate", ascending=True)
        )
        figure = px.bar(
            country_summary,
            x="default_rate",
            y="country",
            orientation="h",
            title="Default rate by country",
            labels={"default_rate": "Default rate", "country": ""},
            color="default_rate",
            color_continuous_scale="OrRd",
            hover_data={"loans": ":,.0f", "defaults": ":,.0f"},
        )
        figure.update_layout(
            template="plotly_white",
            height=390,
            margin=dict(l=10, r=10, t=50, b=10),
            coloraxis_showscale=False,
        )
        figure.update_xaxes(tickformat=".0%")
        st.plotly_chart(figure, width="stretch")

    labelled["amount_band"] = pd.cut(
        labelled["issued_amount"],
        bins=[0, 500, 1_500, 3_000, 6_000, np.inf],
        labels=["≤ €500", "€501–€1,500", "€1,501–€3,000", "€3,001–€6,000", "> €6,000"],
        include_lowest=True,
    )
    amount_summary = labelled.groupby("amount_band", as_index=False, observed=True).agg(
        loans=(TARGET, "size"),
        default_rate=(TARGET, "mean"),
    )
    figure = px.bar(
        amount_summary,
        x="amount_band",
        y="default_rate",
        title="Default rate by issued-amount band",
        labels={"amount_band": "Issued amount", "default_rate": "Default rate"},
        color_discrete_sequence=[COLORS["amber"]],
        text_auto=".1%",
        hover_data={"loans": ":,.0f"},
    )
    figure.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    figure.update_yaxes(tickformat=".0%")
    st.plotly_chart(figure, width="stretch")

    st.info(
        "Only loans with a known 12-month outcome are included. Missing targets "
        "represent loans whose outcome window is not yet measurable—not non-defaults."
    )


def render_model_performance() -> None:
    """Render the weighted and default-class evaluation results."""
    page_header(
        "Model Performance",
        "Compare the three tuned finalists on the untouched 2025 evaluation period.",
    )
    if not MODEL_RESULTS_PATH.exists():
        st.error(f"Model results were not found: {MODEL_RESULTS_PATH}")
        return

    results = load_model_results(MODEL_RESULTS_PATH)
    selected = results.loc[results["model"].eq("Logistic Regression")].iloc[0]
    columns = st.columns(4)
    columns[0].metric("Selected accuracy", percentage(selected["accuracy"], 2))
    columns[1].metric(
        "Weighted precision",
        percentage(selected["weighted_precision"], 2),
    )
    columns[2].metric("Weighted F1", percentage(selected["weighted_f1"], 2))
    columns[3].metric("ROC-AUC", percentage(selected["roc_auc"], 2))

    overall_tab, default_tab, importance_tab = st.tabs(
        ["Overall weighted metrics", "Default-class metrics", "Feature importance"]
    )

    with overall_tab:
        display_columns = [
            "model",
            "maximum_f1_threshold",
            "accuracy",
            "weighted_precision",
            "weighted_f1",
            "roc_auc",
        ]
        st.dataframe(
            results[display_columns],
            width="stretch",
            hide_index=True,
            column_config={
                "model": "Model",
                "maximum_f1_threshold": st.column_config.NumberColumn(
                    "Threshold",
                    format="percent",
                ),
                "accuracy": st.column_config.NumberColumn(
                    "Accuracy",
                    format="percent",
                ),
                "weighted_precision": st.column_config.NumberColumn(
                    "Weighted precision",
                    format="percent",
                ),
                "weighted_f1": st.column_config.NumberColumn(
                    "Weighted F1",
                    format="percent",
                ),
                "roc_auc": st.column_config.NumberColumn(
                    "ROC-AUC",
                    format="percent",
                ),
            },
        )
        melted = results.melt(
            id_vars=["model"],
            value_vars=[
                "accuracy",
                "weighted_precision",
                "weighted_f1",
                "roc_auc",
            ],
            var_name="metric",
            value_name="score",
        )
        melted["metric"] = (
            melted["metric"]
            .str.replace("_", " ", regex=False)
            .str.title()
            .str.replace("Roc Auc", "ROC-AUC", regex=False)
        )
        figure = px.bar(
            melted,
            x="model",
            y="score",
            color="metric",
            barmode="group",
            title="Overall model comparison",
            labels={"model": "", "score": "Score", "metric": "Metric"},
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        figure.update_layout(
            template="plotly_white",
            height=420,
            margin=dict(l=10, r=10, t=50, b=10),
            legend_orientation="h",
            legend_y=-0.2,
        )
        figure.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(figure, width="stretch")
        st.caption(
            "Weighted metrics are dominated by the non-default majority class. "
            "Use the next tab to inspect performance specifically on defaults."
        )

    with default_tab:
        default_columns = [
            "model",
            "default_precision",
            "default_recall",
            "default_f1",
        ]
        st.dataframe(
            results[default_columns],
            width="stretch",
            hide_index=True,
            column_config={
                "model": "Model",
                "default_precision": st.column_config.NumberColumn(
                    "Default precision",
                    format="percent",
                ),
                "default_recall": st.column_config.NumberColumn(
                    "Default recall",
                    format="percent",
                ),
                "default_f1": st.column_config.NumberColumn(
                    "Default F1",
                    format="percent",
                ),
            },
        )
        st.warning(
            "Default-class results are lower because only 7.49% of the recent "
            "evaluation loans default. These metrics are essential for credit-risk "
            "interpretation and should not be replaced by weighted results."
        )

    with importance_tab:
        if not IMPORTANCE_PATH.exists():
            st.info("Feature-importance results are not available.")
        else:
            importance = load_feature_importance(IMPORTANCE_PATH).sort_values(
                "importance_mean",
                ascending=True,
            )
            figure = px.bar(
                importance,
                x="importance_mean",
                y="raw_feature",
                orientation="h",
                error_x="importance_std",
                title="Selected Logistic Regression — permutation importance",
                labels={
                    "importance_mean": "Mean decrease in PR-AUC",
                    "raw_feature": "",
                },
                color_discrete_sequence=[COLORS["teal"]],
            )
            figure.update_layout(
                template="plotly_white",
                height=420,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(figure, width="stretch")
            st.caption(
                "Importance measures predictive contribution, not causality. "
                "Country and the existing risk rating require governance review."
            )

    with st.expander("Evaluation design"):
        st.markdown("""
            - Training: January–September 2024
            - Probability calibration: October–November 2024
            - Threshold selection: December 2024
            - Untouched test: January–27 April 2025
            - All lifecycle, repayment, arrears, status, and target fields were
              excluded from model inputs.
            """)


def render_risk_scoring(frame: pd.DataFrame) -> None:
    """Render an individual-loan risk scoring form."""
    page_header(
        "Loan Risk Scoring",
        "Estimate 12-month default risk using the saved at-origination model.",
    )
    active_model_path = MODEL_PATH if MODEL_PATH.exists() else FALLBACK_MODEL_PATH
    if not active_model_path.exists():
        st.error(
            "No optimized model artifact was found. Run the model-training pipeline "
            "before using this page."
        )
        return

    try:
        model = load_risk_model(active_model_path)
    except Exception as error:
        st.error(f"The saved model could not be loaded: {error}")
        return

    countries = sorted(value for value in frame["country"].dropna().unique() if value != "Missing")
    ratings = [
        value
        for value in RATING_ORDER
        if value == "Missing" or value in set(frame["customer_risk_rating"].dropna().unique())
    ]
    default_country_index = countries.index("Finland") if "Finland" in countries else 0
    default_rating_index = ratings.index("C") if "C" in ratings else 0

    form_column, output_column = st.columns([1.05, 0.95], gap="large")
    with form_column:
        st.markdown(
            '<div class="section-label">Loan application inputs</div>',
            unsafe_allow_html=True,
        )
        with st.form("loan-risk-form"):
            country = st.selectbox(
                "Country",
                countries,
                index=default_country_index,
            )
            issue_date = st.date_input(
                "Issue date",
                value=date.today(),
            )
            amount = st.number_input(
                "Issued amount (€)",
                min_value=1.0,
                max_value=100_000.0,
                value=3_000.0,
                step=100.0,
            )
            interest_percent = st.number_input(
                "Initial annual interest rate (%)",
                min_value=0.0,
                max_value=300.0,
                value=17.8,
                step=0.1,
                help="Enter 17.8 for an interest rate stored as 0.178.",
            )
            duration = st.slider(
                "Initial duration (months)",
                min_value=1,
                max_value=120,
                value=60,
            )
            income_missing = st.checkbox("Combined income is unavailable")
            income = st.number_input(
                "Combined monthly income (€)",
                min_value=0.0,
                max_value=250_000.0,
                value=2_500.0,
                step=100.0,
                disabled=income_missing,
            )
            rating = st.selectbox(
                "Customer risk rating",
                ratings,
                index=default_rating_index,
                help="Select Missing when no at-issue rating is available.",
            )
            submitted = st.form_submit_button(
                "Estimate default risk",
                type="primary",
                width="stretch",
            )

    with output_column:
        st.markdown(
            '<div class="section-label">Risk result</div>',
            unsafe_allow_html=True,
        )
        if not submitted:
            st.markdown(
                """
                <div class="risk-card">
                    Complete the application inputs and select
                    <strong>Estimate default risk</strong>.
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            application = pd.DataFrame(
                {
                    "country": [country],
                    "loan_issued_at": [pd.Timestamp(issue_date)],
                    "issued_amount": [amount],
                    "initial_interest_rate": [interest_percent / 100.0],
                    "initial_loan_duration": [duration],
                    "combined_income": [np.nan if income_missing else income],
                    "customer_risk_rating": [np.nan if rating == "Missing" else rating],
                }
            )
            prediction = model.predict_frame(application).iloc[0]
            probability = float(prediction["probability_of_default"])
            threshold = float(prediction["decision_threshold"])
            is_flagged = bool(prediction["predicted_class"])

            if probability >= threshold:
                category, color = "Review recommended", COLORS["red"]
            elif probability >= threshold * 0.65:
                category, color = "Medium risk", COLORS["amber"]
            else:
                category, color = "Lower risk", COLORS["teal"]

            gauge_limit = max(0.30, threshold * 2.0, probability * 1.35)
            figure = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    number={"suffix": "%", "valueformat": ".2f"},
                    title={"text": "Estimated 12-month default probability"},
                    gauge={
                        "axis": {
                            "range": [0, gauge_limit * 100],
                            "ticksuffix": "%",
                        },
                        "bar": {"color": color},
                        "threshold": {
                            "line": {"color": COLORS["navy"], "width": 4},
                            "thickness": 0.8,
                            "value": threshold * 100,
                        },
                        "steps": [
                            {
                                "range": [0, threshold * 65],
                                "color": "#DCFCE7",
                            },
                            {
                                "range": [threshold * 65, threshold * 100],
                                "color": "#FEF3C7",
                            },
                            {
                                "range": [threshold * 100, gauge_limit * 100],
                                "color": "#FEE2E2",
                            },
                        ],
                    },
                )
            )
            figure.update_layout(
                height=330,
                margin=dict(l=20, r=20, t=65, b=20),
                paper_bgcolor="white",
            )
            st.plotly_chart(figure, width="stretch")

            first, second = st.columns(2)
            first.metric("Decision", "Flagged" if is_flagged else "Not flagged")
            second.metric("Policy threshold", percentage(threshold, 2))

            if is_flagged:
                st.error(f"**{category}.** The estimated risk exceeds the threshold.")
            elif category == "Medium risk":
                st.warning(f"**{category}.** The estimate is below but close to the threshold.")
            else:
                st.success(f"**{category}.** The estimate is below the threshold.")

            st.caption(
                "This score supports manual review. It must not be used as an "
                "automatic approval or rejection decision."
            )

    st.info(
        "The model was developed on issued Bondora loans and uses the assigned "
        "interest rate and existing customer risk rating. It is not validated on "
        "rejected loan applications."
    )


def render_footer() -> None:
    """Render the source and project footer."""
    st.markdown(
        """
        <div class="footer">
        AI-Driven Loan Analytics Platform · Data source:
        <a href="https://goandgrow.eu/en/public-statistics/" target="_blank">
        Bondora Go &amp; Grow public statistics</a>
        </div>
        """,
        unsafe_allow_html=True,
    )


if not DATA_PATH.exists():
    st.error(f"Cleaned dataset not found: {DATA_PATH}")
    st.code(
        "Expected file: data/processed/data_cleaned.csv",
        language="text",
    )
    st.stop()

portfolio = load_portfolio(DATA_PATH)
valid_dates = portfolio["loan_issued_at"].dropna()
if valid_dates.empty:
    st.error("The portfolio does not contain usable issuance dates.")
    st.stop()

with st.sidebar:
    st.markdown("## AI Loan Analytics")
    st.caption("Portfolio intelligence and default-risk monitoring")
    page = st.radio("Navigation", PAGE_OPTIONS, label_visibility="collapsed")
    st.divider()

    minimum_date = valid_dates.min().date()
    maximum_date = valid_dates.max().date()
    selected_dates = st.date_input(
        "Issuance date range",
        value=(minimum_date, maximum_date),
        min_value=minimum_date,
        max_value=maximum_date,
    )
    if not isinstance(selected_dates, tuple) or len(selected_dates) != 2:
        selected_dates = (minimum_date, maximum_date)

    country_options = sorted(portfolio["country"].dropna().unique())
    selected_countries = st.multiselect(
        "Countries",
        country_options,
        placeholder="All countries",
    )
    rating_options = [
        rating
        for rating in RATING_ORDER
        if rating in set(portfolio["customer_risk_rating"].dropna().unique())
    ]
    selected_ratings = st.multiselect(
        "Risk ratings",
        rating_options,
        placeholder="All ratings",
    )
    status_options = sorted(portfolio["loan_status"].dropna().unique())
    selected_statuses = st.multiselect(
        "Loan statuses",
        status_options,
        placeholder="All statuses",
    )
    st.divider()
    st.caption(f"Source portfolio: {len(portfolio):,} loans")

filtered_portfolio = apply_filters(
    portfolio,
    selected_dates,
    selected_countries,
    selected_ratings,
    selected_statuses,
)

if filtered_portfolio.empty and page != "Model Performance":
    st.warning("No loans match the selected filters. Broaden the sidebar filters.")
elif page == "Portfolio Overview":
    render_portfolio_overview(filtered_portfolio)
elif page == "Risk Analysis":
    render_risk_analysis(filtered_portfolio)
elif page == "Model Performance":
    render_model_performance()
elif page == "Loan Risk Scoring":
    scoring_frame = filtered_portfolio if not filtered_portfolio.empty else portfolio
    render_risk_scoring(scoring_frame)

render_footer()
