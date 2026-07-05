from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Telco Churn Risk Dashboard",
    layout="wide",
)

BASE = Path(__file__).resolve().parents[1]

PREDICTION_PATH = BASE / "outputs" / "sprint3" / "dashboard_prediction_output.csv"
MONITORING_PATH = BASE / "outputs" / "sprint4" / "monitoring_metrics.csv"
DRIFT_PATH = BASE / "outputs" / "sprint4" / "data_drift_analysis.csv"
METRICS_PATH = BASE / "outputs" / "sprint3" / "model_metrics_long.csv"

st.title("Telco Customer Churn Risk Dashboard")

st.write(
    "This dashboard supports stakeholder decision-making by displaying churn-risk predictions, "
    "customer segments, monitoring metrics and drift analysis."
)

if not PREDICTION_PATH.exists():
    st.error("Prediction output not found. Run Sprint 3 first.")
    st.stop()

df = pd.read_csv(PREDICTION_PATH)

risk_options = ["All"] + sorted(df["RiskLevel"].dropna().unique().tolist())
selected_risk = st.sidebar.selectbox("Filter by risk level", risk_options)

contract_options = ["All"] + sorted(df["Contract"].dropna().unique().tolist())
selected_contract = st.sidebar.selectbox("Filter by contract type", contract_options)

minimum_probability = st.sidebar.slider(
    "Minimum churn probability",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
)

customer_search = st.sidebar.text_input("Search customer ID")

view = df.copy()

if selected_risk != "All":
    view = view[view["RiskLevel"] == selected_risk]

if selected_contract != "All":
    view = view[view["Contract"] == selected_contract]

view = view[view["ChurnProbability"] >= minimum_probability]

if customer_search:
    view = view[
        view["customerID"].astype(str).str.contains(
            customer_search,
            case=False,
            na=False,
        )
    ]

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric("Customers shown", f"{len(view):,}")
metric2.metric("High-risk customers", f"{int((view['RiskLevel'] == 'High').sum()):,}")
metric3.metric(
    "Average churn probability",
    f"{view['ChurnProbability'].mean():.1%}" if len(view) else "N/A",
)
metric4.metric(
    "Historical churn rate",
    f"{(view['Churn'] == 'Yes').mean():.1%}" if len(view) else "N/A",
)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Visualizations",
        "Predictive Output",
        "Monitoring",
        "Model Results",
    ]
)

with tab1:
    st.subheader("Visualization 1: Risk Level Distribution")
    st.bar_chart(view["RiskLevel"].value_counts())

    st.subheader("Visualization 2: Risk Level by Contract Type")
    st.bar_chart(pd.crosstab(view["Contract"], view["RiskLevel"]))

    st.subheader("Visualization 3: Average Churn Probability by Tenure Group")
    tenure_summary = view.groupby("TenureGroup")["ChurnProbability"].mean().sort_index()
    st.line_chart(tenure_summary)

with tab2:
    st.subheader("Predictive Output: Ranked Customer Churn Risk")

    columns = [
        "customerID",
        "ChurnProbability",
        "RiskLevel",
        "RiskFactors",
        "Contract",
        "TenureGroup",
        "MonthlyCharges",
        "ServiceCount",
        "Churn",
    ]

    st.dataframe(
        view[columns].sort_values("ChurnProbability", ascending=False),
        use_container_width=True,
    )

    st.download_button(
        "Download filtered customer list",
        view[columns].to_csv(index=False),
        "filtered_customer_risk.csv",
        "text/csv",
    )

with tab3:
    st.subheader("Monitoring Metrics")

    if MONITORING_PATH.exists():
        monitoring = pd.read_csv(MONITORING_PATH)
        st.dataframe(monitoring, use_container_width=True)
        numeric_monitoring = monitoring.copy()
        numeric_monitoring["Current Result"] = pd.to_numeric(
            numeric_monitoring["Current Result"],
            errors="coerce",
        )
        st.bar_chart(numeric_monitoring.set_index("Metric")["Current Result"])
    else:
        st.warning("Monitoring output not found.")

    st.subheader("Data Drift Analysis")

    if DRIFT_PATH.exists():
        drift = pd.read_csv(DRIFT_PATH)
        st.dataframe(drift, use_container_width=True)
        st.bar_chart(drift.set_index("Feature")["PSI"])
    else:
        st.warning("Drift output not found.")

with tab4:
    st.subheader("Model Evaluation Results")

    if METRICS_PATH.exists():
        metrics = pd.read_csv(METRICS_PATH)
        st.dataframe(metrics, use_container_width=True)
        chart_data = metrics.set_index("Model")[
            ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
        ]
        st.bar_chart(chart_data)
    else:
        st.warning("Model metrics not found.")

st.caption(
    "This dashboard is a decision-support tool. It does not automatically decide customer actions."
)
