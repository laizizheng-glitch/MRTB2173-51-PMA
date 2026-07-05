from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parents[1]

def test_cleaned_dataset_exists_and_has_no_missing_values():
    path = BASE / "outputs" / "sprint1" / "cleaned_telco_churn.csv"
    assert path.exists()

    df = pd.read_csv(path)
    assert not df.empty
    assert int(df.isna().sum().sum()) == 0

def test_engineered_features_exist():
    path = BASE / "outputs" / "sprint2" / "engineered_telco_churn.csv"
    assert path.exists()

    df = pd.read_csv(path)

    required_features = [
        "TenureGroup",
        "ServiceCount",
        "LongTermContract",
        "AutomaticPayment",
        "HasSupport",
        "ChurnFlag",
    ]

    for feature in required_features:
        assert feature in df.columns

def test_prediction_output_is_valid():
    path = BASE / "outputs" / "sprint3" / "dashboard_prediction_output.csv"
    assert path.exists()

    df = pd.read_csv(path)

    assert "ChurnProbability" in df.columns
    assert "RiskLevel" in df.columns
    assert df["ChurnProbability"].between(0, 1).all()
    assert set(df["RiskLevel"].unique()).issubset({"Low", "Medium", "High"})

def test_dashboard_and_monitoring_outputs_exist():
    required_files = [
        BASE / "dashboard" / "app.py",
        BASE / "outputs" / "sprint4" / "monitoring_metrics.csv",
        BASE / "outputs" / "sprint4" / "data_drift_analysis.csv",
    ]

    for path in required_files:
        assert path.exists(), f"Missing file: {path}"
