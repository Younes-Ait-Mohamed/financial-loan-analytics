# Streamlit Loan Analytics Dashboard

This initial dashboard connects the cleaned Bondora portfolio, optimized model
results, and saved 12-month default-risk model.

## Pages

- **Portfolio Overview:** lending KPIs, issuance trends, country exposure, and
  loan-status composition.
- **Risk Analysis:** observed default rates by rating, country, and loan size.
- **Model Performance:** weighted and default-class metrics for the three tuned
  finalists, plus feature importance.
- **Loan Risk Scoring:** manual scoring form using the saved 11% F1-policy model.

## Run the dashboard

From the repository root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run dashboard\app.py
```

Streamlit will print a local address, normally
`http://localhost:8501`.

## Required local artifacts

The following files must exist:

```text
data/processed/data_cleaned.csv
models/optimized/best_optimized_default_model_f1_policy.joblib
reports/optimized_model_results/weighted_finalist_results.csv
```

The cleaned dataset and model artifacts may be ignored by Git, so a fresh clone
must receive or regenerate them before the dashboard can run.

The first dashboard load reads approximately 154 MB of portfolio data and may
take several seconds. Streamlit caches the result for later interactions.


