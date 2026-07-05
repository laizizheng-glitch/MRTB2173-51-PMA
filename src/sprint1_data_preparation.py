from pathlib import Path
import io
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]

RAW_PATH = BASE / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUTS = BASE / "outputs" / "sprint1"
CHARTS = BASE / "charts" / "sprint1"

OUTPUTS.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)

DATASET_SOURCE_URL = "https://www.kaggle.com/datasets/blastchar/telco-customer-churn"

REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

def save_table(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def capture_info(df):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        df.info()
    return buffer.getvalue()

def load_dataset():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df

def clean_text_columns(df):
    result = df.copy()

    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].astype(str).str.strip()
        result[column] = result[column].replace({"": np.nan, "nan": np.nan})

    return result

def identify_data_quality_issues(df):
    total_charges_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce")

    blank_strings = int(
        (df.select_dtypes(include=["object", "string"]) == " ")
        .sum()
        .sum()
    )

    missing_values = int(df.isna().sum().sum())

    issues = pd.DataFrame(
        [
            {
                "Issue": "Missing values and blank strings",
                "How Identified": "df.isna().sum() and blank-string check",
                "Count": missing_values + blank_strings,
                "Potential Impact": "May cause unreliable dashboard summaries and model training errors.",
                "Action": "Strip text values and handle blank TotalCharges.",
            },
            {
                "Issue": "Non-numeric TotalCharges",
                "How Identified": "pd.to_numeric(TotalCharges, errors='coerce')",
                "Count": int(total_charges_numeric.isna().sum()),
                "Potential Impact": "Prevents proper scaling and modelling of billing values.",
                "Action": "Convert TotalCharges to numeric and repair valid blanks.",
            },
            {
                "Issue": "Duplicate customer IDs",
                "How Identified": "customerID.duplicated().sum()",
                "Count": int(df["customerID"].duplicated().sum()),
                "Potential Impact": "May distort customer-level predictions and dashboard counts.",
                "Action": "Check uniqueness and remove exact duplicate rows.",
            },
            {
                "Issue": "Exact duplicate rows",
                "How Identified": "df.duplicated().sum()",
                "Count": int(df.duplicated().sum()),
                "Potential Impact": "May inflate sample size and bias model evaluation.",
                "Action": "Remove exact duplicate records.",
            },
            {
                "Issue": "Invalid target categories",
                "How Identified": "Check Churn values outside Yes/No",
                "Count": int((~df["Churn"].isin(["Yes", "No"])).sum()),
                "Potential Impact": "Invalid target values prevent supervised classification.",
                "Action": "Validate target labels before modelling.",
            },
        ]
    )

    return issues

def clean_dataset(df):
    result = clean_text_columns(df)
    result = result.drop_duplicates().copy()

    result["TotalCharges"] = pd.to_numeric(result["TotalCharges"], errors="coerce")
    result["MonthlyCharges"] = pd.to_numeric(result["MonthlyCharges"], errors="coerce")
    result["tenure"] = pd.to_numeric(result["tenure"], errors="coerce")

    result["SeniorCitizen"] = pd.to_numeric(
        result["SeniorCitizen"],
        errors="coerce",
    ).fillna(0).astype(int)

    result.loc[
        result["TotalCharges"].isna() & result["tenure"].eq(0),
        "TotalCharges",
    ] = 0

    for column in ["TotalCharges", "MonthlyCharges", "tenure"]:
        if result[column].isna().sum() > 0:
            result[column] = result[column].fillna(result[column].median())

    for column in SERVICE_COLUMNS:
        result[column] = result[column].replace(
            {
                "No internet service": "No",
                "No phone service": "No",
            }
        )

    if not set(result["Churn"].dropna().unique()).issubset({"Yes", "No"}):
        raise ValueError("Invalid Churn values found.")

    return result.reset_index(drop=True)

def build_data_dictionary(df):
    rows = []

    for column in df.columns:
        rows.append(
            {
                "Column": column,
                "Data Type": str(df[column].dtype),
                "Non-null Count": int(df[column].notna().sum()),
                "Missing Count": int(df[column].isna().sum()),
                "Unique Values": int(df[column].nunique(dropna=True)),
                "Example Values": ", ".join(map(str, df[column].dropna().unique()[:3])),
            }
        )

    return pd.DataFrame(rows)

def add_basic_churn_flag(df):
    result = df.copy()
    result["ChurnFlag"] = result["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    return result

def segment_summary(df, group_column):
    summary = (
        df.groupby(group_column)["ChurnFlag"]
        .agg(
            Customer_Count="count",
            Churned_Customers="sum",
            Churn_Rate="mean",
        )
        .reset_index()
        .sort_values("Churn_Rate", ascending=False)
    )

    summary["Churn_Rate_Percent"] = (summary["Churn_Rate"] * 100).round(2)

    return summary

def create_eda_charts(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["MonthlyCharges"], bins=30)
    ax.set_title("Distribution Analysis: Monthly Charges")
    ax.set_xlabel("Monthly Charges")
    ax.set_ylabel("Number of Customers")
    fig.tight_layout()
    fig.savefig(CHARTS / "distribution_monthly_charges.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(df["tenure"], df["TotalCharges"], alpha=0.35)
    ax.set_title("Relationship Analysis: Tenure vs Total Charges")
    ax.set_xlabel("Tenure")
    ax.set_ylabel("Total Charges")
    fig.tight_layout()
    fig.savefig(CHARTS / "relationship_tenure_totalcharges.png", dpi=160)
    plt.close(fig)

    contract_summary = segment_summary(df, "Contract")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(contract_summary["Contract"], contract_summary["Churn_Rate_Percent"])
    ax.set_title("Categorical Analysis: Churn Rate by Contract Type")
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Churn Rate (%)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(CHARTS / "categorical_churn_by_contract.png", dpi=160)
    plt.close(fig)

def main():
    raw_df = load_dataset()

    raw_df.head().to_csv(OUTPUTS / "dataset_preview_head.csv", index=False)

    (OUTPUTS / "dataset_info.txt").write_text(
        capture_info(raw_df),
        encoding="utf-8",
    )

    dataset_description = pd.DataFrame(
        [
            {
                "Dataset Source": DATASET_SOURCE_URL,
                "Organizational Problem": "Identify customers likely to churn so retention action can be prioritized.",
                "Stakeholders": "Retention manager, customer service supervisor, business manager, data science team.",
                "Number of Records": len(raw_df),
                "Number of Variables": raw_df.shape[1],
                "Target Variable": "Churn",
                "Analysis Type": "Binary classification and dashboard analytics.",
            }
        ]
    )

    save_table(dataset_description, OUTPUTS / "dataset_description.csv")

    quality_issues = identify_data_quality_issues(raw_df)
    save_table(quality_issues, OUTPUTS / "data_quality_issues.csv")

    cleaned_df = clean_dataset(raw_df)
    cleaned_with_flag = add_basic_churn_flag(cleaned_df)

    save_table(cleaned_df, OUTPUTS / "cleaned_telco_churn.csv")
    save_table(build_data_dictionary(cleaned_df), OUTPUTS / "data_dictionary.csv")

    for group_column in ["Contract", "PaymentMethod", "InternetService"]:
        save_table(
            segment_summary(cleaned_with_flag, group_column),
            OUTPUTS / f"churn_summary_by_{group_column}.csv",
        )

    create_eda_charts(cleaned_with_flag)

    sprint1_backlog = pd.DataFrame(
        [
            {
                "Priority": "High",
                "Backlog Item": "Convert TotalCharges to numeric and repair blank values",
                "Reason for Priority": "TotalCharges is required for modelling and dashboard analysis.",
                "Expected Deliverable": "Cleaned TotalCharges column with documented rule.",
            },
            {
                "Priority": "High",
                "Backlog Item": "Remove exact duplicate rows and validate customer ID uniqueness",
                "Reason for Priority": "Duplicates may distort churn rate and prediction reliability.",
                "Expected Deliverable": "Duplicate check output and cleaned dataset.",
            },
            {
                "Priority": "Medium",
                "Backlog Item": "Standardize inconsistent service category labels",
                "Reason for Priority": "Consistent categories improve dashboard readability.",
                "Expected Deliverable": "Standardized Yes/No service columns.",
            },
        ]
    )

    save_table(sprint1_backlog, OUTPUTS / "sprint1_backlog.csv")

    print("=" * 72)
    print("SPRINT 1 COMPLETED")
    print("=" * 72)
    print("Records:", len(raw_df))
    print("Variables:", raw_df.shape[1])
    print("Cleaned dataset saved.")
    print("EDA charts saved.")
    print("Sprint 1 outputs:", OUTPUTS)
    print("=" * 72)

if __name__ == "__main__":
    main()
