from pathlib import Path
import sys

import pandas as pd

BASE = Path(__file__).resolve().parents[1]

RAW = BASE / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
CLEANED = BASE / "outputs" / "sprint1" / "cleaned_telco_churn.csv"
ENGINEERED = BASE / "outputs" / "sprint2" / "engineered_telco_churn.csv"
PREDICTIONS = BASE / "outputs" / "sprint3" / "dashboard_prediction_output.csv"

required_columns = [
    "customerID",
    "tenure",
    "Contract",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

errors = []

if not RAW.exists():
    errors.append("Raw dataset is missing.")

if not CLEANED.exists():
    errors.append("Cleaned dataset is missing.")
else:
    cleaned = pd.read_csv(CLEANED)

    missing_columns = [
        column
        for column in required_columns
        if column not in cleaned.columns
    ]

    if missing_columns:
        errors.append(f"Cleaned dataset missing columns: {missing_columns}")

    if cleaned.empty:
        errors.append("Cleaned dataset is empty.")

    if int(cleaned.isna().sum().sum()) != 0:
        errors.append("Cleaned dataset still contains missing values.")

if not ENGINEERED.exists():
    errors.append("Engineered dataset is missing.")
else:
    engineered = pd.read_csv(ENGINEERED)

    required_engineered = [
        "TenureGroup",
        "ServiceCount",
        "LongTermContract",
        "AutomaticPayment",
        "HasSupport",
        "ChurnFlag",
    ]

    missing_engineered = [
        column
        for column in required_engineered
        if column not in engineered.columns
    ]

    if missing_engineered:
        errors.append(f"Engineered features missing: {missing_engineered}")

if not PREDICTIONS.exists():
    errors.append("Prediction output is missing.")
else:
    predictions = pd.read_csv(PREDICTIONS)

    if not predictions["ChurnProbability"].between(0, 1).all():
        errors.append("ChurnProbability contains invalid values.")

    invalid_risk = set(predictions["RiskLevel"].unique()) - {"Low", "Medium", "High"}

    if invalid_risk:
        errors.append(f"Invalid RiskLevel values: {invalid_risk}")

print("=" * 72)
print("AUTOMATED VALIDATION RESULT")
print("=" * 72)

if errors:
    print("Validation failed.")

    for error in errors:
        print("-", error)

    sys.exit(1)

print("Validation passed.")
print("Dataset, engineered features and prediction outputs are valid.")
print("=" * 72)
