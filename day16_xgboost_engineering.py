import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

TARGET = "satisfaction"
ID_COLUMN = "id"
RANDOM_STATE = 42

CATEGORICAL_COLUMNS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]

SERVICE_COLUMNS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]


def prepare_features(df):
    X = df.drop(columns=[TARGET, ID_COLUMN], errors="ignore").copy()

    # Handle missing values
    X["Arrival Delay in Minutes"] = X[
        "Arrival Delay in Minutes"
    ].fillna(X["Arrival Delay in Minutes"].median())

    # Service rating aggregates
    ratings = X[SERVICE_COLUMNS]

    X["service_rating_mean"] = ratings.mean(axis=1)
    X["service_rating_min"] = ratings.min(axis=1)
    X["service_rating_max"] = ratings.max(axis=1)
    X["service_rating_std"] = ratings.std(axis=1)
    X["service_rating_sum"] = ratings.sum(axis=1)

    X["low_rating_count"] = (ratings <= 2).sum(axis=1)
    X["high_rating_count"] = (ratings >= 4).sum(axis=1)

    # Targeted interactions
    X["wifi_boarding_interaction"] = (
        X["Inflight wifi service"] * X["Online boarding"]
    )

    X["entertainment_seat_interaction"] = (
        X["Inflight entertainment"] * X["Seat comfort"]
    )

    X["booking_wifi_interaction"] = (
        X["Ease of Online booking"] * X["Inflight wifi service"]
    )

    # Delay features
    departure = X["Departure Delay in Minutes"]
    arrival = X["Arrival Delay in Minutes"]

    X["total_delay"] = departure + arrival
    X["delay_difference"] = arrival - departure
    X["log_total_delay"] = np.log1p(X["total_delay"].clip(lower=0))

    # Passenger and flight features
    X["age_group"] = pd.cut(
        X["Age"],
        bins=[0, 18, 25, 35, 45, 55, 65, 100],
        labels=False,
        include_lowest=True,
    ).fillna(-1).astype(int)

    X["log_flight_distance"] = np.log1p(
        X["Flight Distance"].clip(lower=0)
    )

    # One-hot encode categorical features
    X = pd.get_dummies(
        X,
        columns=CATEGORICAL_COLUMNS,
        dtype=np.uint8,
    )

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    return X


def main():
    start = time.time()

    print("Loading training data...")
    df = pd.read_csv(DATA_DIR / "train.csv")

    y = df[TARGET].astype(int)
    X = prepare_features(df)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_valid):,}")
    print(f"Engineered features: {X.shape[1]}")

    print("\nTraining engineered XGBoost...")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=2.0,
        tree_method="hist",
        early_stopping_rounds=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=100,
    )

    probabilities = model.predict_proba(
        X_valid,
        iteration_range=(0, model.best_iteration + 1),
    )[:, 1]

    auc = roc_auc_score(y_valid, probabilities)

    model_path = MODEL_DIR / "day16_xgboost_engineered.json"
    model.save_model(str(model_path))

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_path = REPORT_DIR / "day16_feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    result = {
        "experiment": "Day 16 - XGBoost feature engineering",
        "validation_roc_auc": float(auc),
        "best_iteration": int(model.best_iteration),
        "training_rows": len(X_train),
        "validation_rows": len(X_valid),
        "feature_count": X.shape[1],
        "elapsed_seconds": round(time.time() - start, 2),
    }

    report_path = REPORT_DIR / "day16_feature_engineering_metrics.json"

    report_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\n========== DAY 16 RESULTS ==========")
    print(f"Validation ROC AUC: {auc:.6f}")
    print(f"Day 12 baseline: 0.958002")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Elapsed: {result['elapsed_seconds']} seconds")

    print("\nTop 15 features:")
    print(importance.head(15).to_string(index=False))

    print(f"\nModel saved: {model_path}")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
