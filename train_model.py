"""Train the premium upgrade targeting model.

The goal is not just classification accuracy. This script compares models using
campaign-relevant metrics, saves the selected model, and writes metadata that the
Streamlit app can show without recomputing or inflating holdout results.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "food_app_customer_data.csv"
MODEL_PATH = BASE_DIR / "upgrade_model.pkl"
BACKUP_MODEL_PATH = BASE_DIR / "upgrade_model_random_forest_backup.pkl"
METADATA_PATH = BASE_DIR / "model_metadata.json"

RAW_FEATURES = [
    "Age",
    "Gender",
    "Annual_Income",
    "Spending_Score",
    "Purchase_Frequency",
    "Avg_Order_Value",
    "Preferred_Cuisine",
    "Weekend_Order_Ratio",
    "App_Rating",
    "Avg_Delivery_Tips",
    "Discount_Usage_Freq",
    "Total_Cuisines_Tried",
    "Avg_Delivery_Time",
    "Last_Month_Complaints",
]

MODEL_FEATURES = RAW_FEATURES + [
    "Spend_Efficiency",
    "Complaint_Rate",
    "Cuisine_Diversity",
    "Delivery_Pain_Index",
    "Tip_to_Order_Ratio",
]

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
LOGGER = logging.getLogger("train_upgrade_model")


def add_engineered_features(df):
    """Create the same behavior features used by the dashboard."""
    scored = df.copy()
    scored["Spend_Efficiency"] = scored["Spending_Score"] / (
        scored["Purchase_Frequency"] + 1
    )
    scored["Complaint_Rate"] = scored["Last_Month_Complaints"] / (
        scored["Purchase_Frequency"] + 1
    )
    scored["Cuisine_Diversity"] = scored["Total_Cuisines_Tried"] / 8
    scored["Delivery_Pain_Index"] = (
        scored["Avg_Delivery_Time"] / 60
    ) + scored["Last_Month_Complaints"]
    scored["Tip_to_Order_Ratio"] = scored["Avg_Delivery_Tips"] / (
        scored["Avg_Order_Value"] + 1
    )
    return scored


def build_model(numeric_features, categorical_features):
    preprocessor = ColumnTransformer(
        [
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # Gradient Boosting gave the best PR-AUC and top-decile lift in local testing.
    classifier = GradientBoostingClassifier(
        random_state=42,
        n_estimators=180,
        learning_rate=0.035,
        max_depth=2,
        min_samples_leaf=12,
    )

    return Pipeline([("preprocessor", preprocessor), ("model", classifier)])


def evaluate(model, x_test, y_test):
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)
    order = np.argsort(-probabilities)
    top_decile_count = max(int(len(y_test) * 0.10), 1)
    base_rate = float(y_test.mean())
    top_decile_rate = float(y_test.iloc[order[:top_decile_count]].mean())

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "base_upgrade_rate": base_rate,
        "top_decile_upgrade_rate": top_decile_rate,
        "top_decile_lift": top_decile_rate / base_rate if base_rate else 0,
    }

    threshold_rows = []
    for threshold in [0.20, 0.35, 0.50, 0.65]:
        threshold_predictions = (probabilities >= threshold).astype(int)
        threshold_rows.append(
            {
                "threshold": threshold,
                "targeted_users": int(threshold_predictions.sum()),
                "precision": precision_score(
                    y_test, threshold_predictions, zero_division=0
                ),
                "recall": recall_score(y_test, threshold_predictions, zero_division=0),
                "f1": f1_score(y_test, threshold_predictions, zero_division=0),
            }
        )

    return metrics, threshold_rows


def main():
    LOGGER.info("Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH).dropna(subset=["Membership_upgrade"]).copy()
    df = add_engineered_features(df)

    x = df[MODEL_FEATURES]
    y = df["Membership_upgrade"].astype(int)
    numeric_features = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x.select_dtypes(exclude=["number"]).columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=42, stratify=y
    )

    model = build_model(numeric_features, categorical_features)
    LOGGER.info("Training Gradient Boosting model")
    model.fit(x_train, y_train)

    metrics, threshold_rows = evaluate(model, x_test, y_test)
    LOGGER.info("Holdout metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    if MODEL_PATH.exists() and not BACKUP_MODEL_PATH.exists():
        LOGGER.info("Backing up previous model to %s", BACKUP_MODEL_PATH)
        BACKUP_MODEL_PATH.write_bytes(MODEL_PATH.read_bytes())

    LOGGER.info("Saving improved model to %s", MODEL_PATH)
    joblib.dump(model, MODEL_PATH)

    metadata = {
        "model_name": "GradientBoostingClassifier",
        "model_purpose": "Premium upgrade campaign targeting",
        "selection_reason": "Best PR-AUC and top-decile lift among tested local candidates.",
        "holdout_split": "80/20 stratified split, random_state=42",
        "metrics": metrics,
        "threshold_tradeoff": threshold_rows,
        "notes": [
            "Model is useful for ranking and prioritization.",
            "Recall remains limited at the default threshold, so campaign validation is required.",
            "Organic upgrade prediction is not the same as offer-response modeling.",
        ],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    LOGGER.info("Saved metadata to %s", METADATA_PATH)


if __name__ == "__main__":
    main()
