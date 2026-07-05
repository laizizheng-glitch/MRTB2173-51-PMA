from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
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

SPRINT1_OUTPUTS = BASE / "outputs" / "sprint1"
OUTPUTS = BASE / "outputs" / "sprint2"
CHARTS = BASE / "charts" / "sprint2"
MODELS = BASE / "models"

OUTPUTS.mkdir(parents=True, exist_ok=True)
CHARTS.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

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

def add_engineered_features(df):
    result = df.copy()

    result["ChurnFlag"] = result["Churn"].map({"No": 0, "Yes": 1}).astype(int)

    result["TenureGroup"] = pd.cut(
        result["tenure"],
        bins=[-1, 12, 24, 48, 72],
        labels=[
            "0-12 months",
            "13-24 months",
            "25-48 months",
            "49-72 months",
        ],
    ).astype(str)

    result["ServiceCount"] = result[SERVICE_COLUMNS].eq("Yes").sum(axis=1)

    result["LongTermContract"] = result["Contract"].isin(
        ["One year", "Two year"]
    ).astype(int)

    result["AutomaticPayment"] = result["PaymentMethod"].isin(
        [
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ]
    ).astype(int)

    result["HasSupport"] = result[
        [
            "OnlineSecurity",
            "TechSupport",
        ]
    ].eq("Yes").any(axis=1).astype(int)

    result["AverageChargePerMonth"] = (
        result["TotalCharges"] / result["tenure"].replace(0, np.nan)
    )

    result["AverageChargePerMonth"] = result["AverageChargePerMonth"].fillna(
        result["MonthlyCharges"]
    )

    return result

def prepare_model_data(df):
    exclude_columns = [
        "customerID",
        "Churn",
        "ChurnFlag",
        "gender",
    ]

    X = df.drop(columns=exclude_columns)
    y = df["ChurnFlag"]

    numeric_features = X.select_dtypes(
        include=[
            "int64",
            "float64",
            "int32",
            "float32",
        ]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=[
            "object",
            "category",
        ]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return X, y, preprocessor

def evaluate_model(model, X_test, y_test):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    metrics = pd.DataFrame(
        [
            {
                "Model": "Sprint 2 Baseline - Logistic Regression",
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

def save_confusion_matrix(matrix):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    image = ax.imshow(matrix)

    ax.set_title("Baseline Logistic Regression Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], labels=["No Churn", "Churn"])
    ax.set_yticks([0, 1], labels=["No Churn", "Churn"])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(CHARTS / "baseline_confusion_matrix.png", dpi=160)
    plt.close(fig)

def assign_risk(probabilities):
    return pd.cut(
        probabilities,
        bins=[-0.001, 0.30, 0.60, 1.001],
        labels=["Low", "Medium", "High"],
    ).astype(str)

def main():
    cleaned_path = SPRINT1_OUTPUTS / "cleaned_telco_churn.csv"

    if not cleaned_path.exists():
        raise FileNotFoundError("Sprint 1 cleaned dataset not found.")

    cleaned_df = pd.read_csv(cleaned_path)
    engineered_df = add_engineered_features(cleaned_df)

    save_table(engineered_df, OUTPUTS / "engineered_telco_churn.csv")

    feature_steps = pd.DataFrame(
        [
            {
                "Feature Engineering Step": "Categorical encoding",
                "Reason": "Categorical customer attributes must be converted into numeric model inputs.",
            },
            {
                "Feature Engineering Step": "Numerical scaling",
                "Reason": "Logistic Regression is sensitive to numerical scale differences.",
            },
            {
                "Feature Engineering Step": "Feature creation",
                "Reason": "TenureGroup, ServiceCount, LongTermContract, AutomaticPayment and HasSupport improve business meaning.",
            },
        ]
    )

    save_table(feature_steps, OUTPUTS / "feature_engineering_steps.csv")

    X, y, preprocessor = prepare_model_data(engineered_df)

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

    baseline_model.fit(X_train, y_train)

    metrics, matrix, test_probabilities = evaluate_model(
        baseline_model,
        X_test,
        y_test,
    )

    save_table(metrics, OUTPUTS / "baseline_metrics.csv")
    save_confusion_matrix(matrix)

    all_probabilities = baseline_model.predict_proba(X)[:, 1]

    baseline_output = engineered_df.copy()
    baseline_output["BaselineChurnProbability"] = all_probabilities
    baseline_output["BaselineRiskLevel"] = assign_risk(all_probabilities)

    baseline_output = baseline_output.sort_values(
        "BaselineChurnProbability",
        ascending=False,
    ).reset_index(drop=True)

    save_table(baseline_output, OUTPUTS / "baseline_prediction_output.csv")
    save_table(baseline_output.head(25), OUTPUTS / "baseline_top_25_high_risk_customers.csv")

    joblib.dump(baseline_model, MODELS / "sprint2_baseline_logistic_regression.joblib")

    print("=" * 72)
    print("SPRINT 2 COMPLETED")
    print("=" * 72)
    print(metrics)
    print("Engineered dataset saved.")
    print("Baseline model saved.")
    print("Sprint 2 outputs:", OUTPUTS)
    print("=" * 72)

if __name__ == "__main__":
    main()
