from pathlib import Path

import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parents[1]

SPRINT2_OUTPUTS = BASE / "outputs" / "sprint2"
OUTPUTS = BASE / "outputs" / "sprint3"
CHARTS = BASE / "charts" / "sprint3"
MODELS = BASE / "models"

OUTPUTS.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

def save_table(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def prepare_model_data(df):
    exclude_columns = [
        "customerID",
        "Churn",
        "ChurnFlag",
        "gender",
        "RiskFactors",
        "MonthlyCharges_Median",
    ]

    existing_exclude = [
        column
        for column in exclude_columns
        if column in df.columns
    ]

    X = df.drop(columns=existing_exclude)
    y = df["ChurnFlag"]

    numeric_features = X.select_dtypes(
        include=["int64", "float64", "int32", "float32"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return X, y, preprocessor

def evaluate_model(model, X_test, y_test, model_name):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = pd.DataFrame(
        [
            {
                "Model": model_name,
                "Accuracy": accuracy_score(y_test, predictions),
                "Precision": precision_score(y_test, predictions, zero_division=0),
                "Recall": recall_score(y_test, predictions, zero_division=0),
                "F1": f1_score(y_test, predictions, zero_division=0),
                "ROC_AUC": roc_auc_score(y_test, probabilities),
            }
        ]
    ).round(4)

    matrix = confusion_matrix(y_test, predictions)

    return metrics, matrix, probabilities

def save_confusion_matrix(matrix, title, path):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["No Churn", "Churn"])
    ax.set_yticks([0, 1], labels=["No Churn", "Churn"])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def assign_risk(probabilities):
    return pd.cut(
        probabilities,
        bins=[-0.001, 0.30, 0.60, 1.001],
        labels=["Low", "Medium", "High"],
    ).astype(str)

def create_risk_factor(row):
    reasons = []

    if row["Contract"] == "Month-to-month":
        reasons.append("month-to-month contract")

    if row["tenure"] <= 12:
        reasons.append("short tenure")

    if row["TechSupport"] == "No":
        reasons.append("no technical support")

    if row["MonthlyCharges"] >= row["MonthlyCharges_Median"]:
        reasons.append("above-median monthly charges")

    if row["AutomaticPayment"] == 0:
        reasons.append("non-automatic payment method")

    return "; ".join(reasons[:3]) if reasons else "lower-risk profile"

def save_dashboard_charts(df):
    risk_order = ["Low", "Medium", "High"]

    fig, ax = plt.subplots(figsize=(8, 5))
    df["RiskLevel"].value_counts().reindex(risk_order).fillna(0).plot(
        kind="bar",
        ax=ax,
    )
    ax.set_title("Risk Level Distribution")
    ax.set_xlabel("Risk Level")
    ax.set_ylabel("Number of Customers")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    fig.savefig(CHARTS / "risk_level_distribution.png", dpi=160)
    plt.close(fig)

    risk_contract = pd.crosstab(df["Contract"], df["RiskLevel"])

    fig, ax = plt.subplots(figsize=(8, 5))
    risk_contract.plot(kind="bar", ax=ax)
    ax.set_title("Risk Level by Contract Type")
    ax.set_xlabel("Contract Type")
    ax.set_ylabel("Number of Customers")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(CHARTS / "risk_by_contract.png", dpi=160)
    plt.close(fig)

    tenure_risk = (
        df.groupby("TenureGroup")["ChurnProbability"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        tenure_risk["TenureGroup"].astype(str),
        tenure_risk["ChurnProbability"],
        marker="o",
    )
    ax.set_title("Average Churn Probability by Tenure Group")
    ax.set_xlabel("Tenure Group")
    ax.set_ylabel("Average Churn Probability")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(CHARTS / "average_risk_by_tenure.png", dpi=160)
    plt.close(fig)

def main():
    engineered_path = SPRINT2_OUTPUTS / "engineered_telco_churn.csv"

    if not engineered_path.exists():
        raise FileNotFoundError("Sprint 2 engineered dataset not found.")

    df = pd.read_csv(engineered_path)

    df["MonthlyCharges_Median"] = df["MonthlyCharges"].median()
    df["RiskFactors"] = df.apply(create_risk_factor, axis=1)

    X, y, preprocessor = prepare_model_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    baseline_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    improved_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=10,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    baseline_model.fit(X_train, y_train)
    improved_model.fit(X_train, y_train)

    baseline_metrics, baseline_matrix, baseline_probabilities = evaluate_model(
        baseline_model,
        X_test,
        y_test,
        "Sprint 2 Baseline - Logistic Regression",
    )

    improved_metrics, improved_matrix, improved_probabilities = evaluate_model(
        improved_model,
        X_test,
        y_test,
        "Sprint 3 Improved - Random Forest",
    )

    save_table(baseline_metrics, OUTPUTS / "baseline_metrics_recalculated.csv")
    save_table(improved_metrics, OUTPUTS / "improved_metrics.csv")

    save_confusion_matrix(
        improved_matrix,
        "Improved Random Forest Confusion Matrix",
        CHARTS / "improved_confusion_matrix.png",
    )

    comparison = pd.DataFrame(
        [
            {
                "Aspect": "Model Used",
                "Sprint 2 Baseline": "Logistic Regression",
                "Sprint 3 Improved": "Random Forest",
            },
            {
                "Aspect": "Evaluation Metric",
                "Sprint 2 Baseline": "Accuracy, Precision, Recall, F1, ROC-AUC",
                "Sprint 3 Improved": "Accuracy, Precision, Recall, F1, ROC-AUC",
            },
            {
                "Aspect": "Evaluation Result",
                "Sprint 2 Baseline": (
                    f"F1 = {baseline_metrics.loc[0, 'F1']:.3f}; "
                    f"Accuracy = {baseline_metrics.loc[0, 'Accuracy']:.3f}"
                ),
                "Sprint 3 Improved": (
                    f"F1 = {improved_metrics.loc[0, 'F1']:.3f}; "
                    f"Accuracy = {improved_metrics.loc[0, 'Accuracy']:.3f}"
                ),
            },
            {
                "Aspect": "Improvement Introduced",
                "Sprint 2 Baseline": "Simple baseline classification model",
                "Sprint 3 Improved": "Alternative ensemble model with class weighting and tuned depth",
            },
        ]
    )

    save_table(comparison, OUTPUTS / "model_comparison_table.csv")

    all_metrics = pd.concat(
        [
            baseline_metrics,
            improved_metrics,
        ],
        ignore_index=True,
    )

    save_table(all_metrics, OUTPUTS / "model_metrics_long.csv")

    if improved_metrics.loc[0, "F1"] >= baseline_metrics.loc[0, "F1"]:
        best_model = improved_model
        best_model_name = "Random Forest"
    else:
        best_model = baseline_model
        best_model_name = "Logistic Regression"

    all_probabilities = best_model.predict_proba(X)[:, 1]

    prediction_output = df.copy()
    prediction_output["ChurnProbability"] = all_probabilities
    prediction_output["RiskLevel"] = assign_risk(all_probabilities)

    prediction_output = prediction_output.sort_values(
        "ChurnProbability",
        ascending=False,
    ).reset_index(drop=True)

    save_table(prediction_output, OUTPUTS / "dashboard_prediction_output.csv")
    save_table(prediction_output.head(25), OUTPUTS / "top_25_high_risk_customers.csv")

    save_dashboard_charts(prediction_output)

    joblib.dump(best_model, MODELS / "sprint3_best_churn_model.joblib")

    metadata = {
        "best_model": best_model_name,
        "baseline_f1": float(baseline_metrics.loc[0, "F1"]),
        "improved_f1": float(improved_metrics.loc[0, "F1"]),
        "target": "ChurnFlag",
    }

    (MODELS / "sprint3_model_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("=" * 72)
    print("SPRINT 3 COMPLETED")
    print("=" * 72)
    print(all_metrics)
    print("Best model:", best_model_name)
    print("Prediction output saved.")
    print("Sprint 3 outputs:", OUTPUTS)
    print("=" * 72)

if __name__ == "__main__":
    main()
