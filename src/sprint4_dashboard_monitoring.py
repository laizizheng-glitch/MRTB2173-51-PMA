from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]

SPRINT3_OUTPUTS = BASE / "outputs" / "sprint3"
OUTPUTS = BASE / "outputs" / "sprint4"
CHARTS = BASE / "charts" / "sprint4"

OUTPUTS.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

def save_table(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def population_stability_index(reference, current, bins=10):
    reference = reference.dropna().astype(float)
    current = current.dropna().astype(float)

    if reference.nunique() <= 1 or current.nunique() <= 1:
        return 0.0

    breakpoints = np.quantile(reference, np.linspace(0, 1, bins + 1))
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) <= 2:
        return 0.0

    reference_counts, _ = np.histogram(reference, bins=breakpoints)
    current_counts, _ = np.histogram(current, bins=breakpoints)

    reference_pct = np.maximum(reference_counts / max(reference_counts.sum(), 1), 0.0001)
    current_pct = np.maximum(current_counts / max(current_counts.sum(), 1), 0.0001)

    psi = np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct))

    return float(round(psi, 4))

def main():
    prediction_path = SPRINT3_OUTPUTS / "dashboard_prediction_output.csv"
    metrics_path = SPRINT3_OUTPUTS / "model_metrics_long.csv"

    if not prediction_path.exists():
        raise FileNotFoundError("Sprint 3 prediction output not found.")

    if not metrics_path.exists():
        raise FileNotFoundError("Sprint 3 model metrics not found.")

    predictions = pd.read_csv(prediction_path)
    model_metrics = pd.read_csv(metrics_path)

    best_f1 = float(model_metrics["F1"].max())

    monitoring = pd.DataFrame(
        [
            {
                "Metric": "Missing value rate",
                "What is Monitored": "Missing values in dashboard prediction output.",
                "Current Result": float(predictions.isna().mean().mean()),
                "Interpretation": "High missingness may reduce dashboard reliability.",
            },
            {
                "Metric": "Duplicate customer ID count",
                "What is Monitored": "Repeated customer IDs in prediction output.",
                "Current Result": int(predictions["customerID"].duplicated().sum()),
                "Interpretation": "Duplicate IDs may distort customer-level decisions.",
            },
            {
                "Metric": "Best model F1 score",
                "What is Monitored": "Balance of precision and recall from model evaluation.",
                "Current Result": best_f1,
                "Interpretation": "Lower future F1 may indicate model degradation.",
            },
            {
                "Metric": "High-risk customer share",
                "What is Monitored": "Percentage of customers classified as high risk.",
                "Current Result": float((predictions["RiskLevel"] == "High").mean()),
                "Interpretation": "Large changes may indicate data drift or business changes.",
            },
        ]
    )

    save_table(monitoring, OUTPUTS / "monitoring_metrics.csv")

    reference = predictions.iloc[: int(len(predictions) * 0.70)]
    current = predictions.iloc[int(len(predictions) * 0.70):]

    drift_rows = []

    for feature in [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "ServiceCount",
        "ChurnProbability",
    ]:
        psi = population_stability_index(reference[feature], current[feature])

        if psi < 0.10:
            status = "Low drift"
        elif psi < 0.25:
            status = "Moderate drift"
        else:
            status = "High drift"

        drift_rows.append(
            {
                "Feature": feature,
                "PSI": psi,
                "Drift Status": status,
                "Possible Effect": "Distribution changes may reduce future model reliability.",
            }
        )

    drift = pd.DataFrame(drift_rows)

    save_table(drift, OUTPUTS / "data_drift_analysis.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(drift["Feature"], drift["PSI"])
    ax.set_title("Data Drift Analysis Using PSI")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Population Stability Index")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(CHARTS / "data_drift_psi.png", dpi=160)
    plt.close(fig)

    retrospective = pd.DataFrame(
        [
            {
                "What went well": "The project produced a complete churn-risk pipeline from raw data to dashboard output.",
                "What did not go well": "The data cleaning and feature engineering steps required careful validation.",
                "Improvement for next sprint": "Add stronger automated checks before every dashboard refresh.",
            },
            {
                "What went well": "The dashboard output made the model results easier for stakeholders to understand.",
                "What did not go well": "The first model output needed clearer customer-level risk explanations.",
                "Improvement for next sprint": "Improve explanation quality and collect actual user feedback.",
            },
        ]
    )

    save_table(retrospective, OUTPUTS / "agile_retrospective.csv")

    sprint4_backlog = pd.DataFrame(
        [
            {
                "Priority": "High",
                "Backlog Item": "Add automated schema and drift checks before dashboard refresh",
                "Expected Benefit": "Prevents unreliable outputs when future data changes.",
            },
            {
                "Priority": "High",
                "Backlog Item": "Add monitoring alert for sudden increase in high-risk customer share",
                "Expected Benefit": "Helps stakeholders identify possible market or data-quality changes early.",
            },
            {
                "Priority": "Medium",
                "Backlog Item": "Collect retention action feedback after campaigns",
                "Expected Benefit": "Connects model predictions with real business outcomes.",
            },
        ]
    )

    save_table(sprint4_backlog, OUTPUTS / "sprint4_backlog.csv")

    print("=" * 72)
    print("SPRINT 4 MONITORING COMPLETED")
    print("=" * 72)
    print("Monitoring metrics saved.")
    print("Data drift analysis saved.")
    print("Sprint 4 outputs:", OUTPUTS)
    print("=" * 72)

if __name__ == "__main__":
    main()
