
import json
import time
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")

TARGET = "satisfaction"
ID_COLUMN = "id"
RANDOM_STATE = 42
BASELINE_AUC = 0.9577167082

CATEGORICAL_COLUMNS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]

SERVICE_RATING_COLUMNS = [
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
    "Inflight service",
    "Cleanliness",
]


def add_features(X):
    X = X.copy()

    # Aggregate the passenger's service ratings.
    rating_columns = [
        col for col in SERVICE_RATING_COLUMNS if col in X.columns
    ]

    if rating_columns:
        ratings = X[rating_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        X["service_rating_mean"] = ratings.mean(axis=1)
        X["service_rating_sum"] = ratings.sum(axis=1, min_count=1)
        X["service_rating_count"] = ratings.notna().sum(axis=1)

    # Group age into broad categories.
    if "Age" in X.columns:
        age = pd.to_numeric(X["Age"], errors="coerce")
        X["age_group"] = pd.cut(
            age,
            bins=[0, 18, 30, 45, 60, float("inf")],
            labels=["under_18", "18_29", "30_44", "45_59", "60_plus"],
            right=False,
        ).astype("string").fillna("Missing").astype(str)

    # Capture overall delay and the difference between arrival
    # and departure delays.
    departure_col = "Departure Delay in Minutes"
    arrival_col = "Arrival Delay in Minutes"

    if departure_col in X.columns:
        X[departure_col] = pd.to_numeric(
            X[departure_col], errors="coerce"
        )

    if arrival_col in X.columns:
        X[arrival_col] = pd.to_numeric(
            X[arrival_col], errors="coerce"
        )

    if departure_col in X.columns and arrival_col in X.columns:
        X["total_delay_minutes"] = X[
            [departure_col, arrival_col]
        ].sum(axis=1, min_count=1)

        X["arrival_departure_delay_difference"] = (
            X[arrival_col] - X[departure_col]
        )

        X["arrival_delay_missing"] = (
            X[arrival_col].isna().astype(int)
        )

    return X


def main():
    start_time = time.time()
    REPORT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    print("Loading training data...")
    df = pd.read_csv(DATA_DIR / "train.csv")

    X = df.drop(columns=[TARGET, ID_COLUMN]).copy()
    y = df[TARGET].astype(int)

    print("Creating engineered features...")
    original_columns = X.columns.tolist()
    X = add_features(X)

    categorical_columns = [
        col for col in CATEGORICAL_COLUMNS if col in X.columns
    ]
    if "age_group" in X.columns:
        categorical_columns.append("age_group")

    for col in categorical_columns:
        X[col] = X[col].fillna("Missing").astype(str)

    numeric_columns = [
        col for col in X.columns if col not in categorical_columns
    ]
    for col in numeric_columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_valid):,}")
    print(f"Original features: {len(original_columns)}")
    print(f"Total features after engineering: {X.shape[1]}")
    print("Training CatBoost...")

    model = CatBoostClassifier(
        iterations=1000,
        depth=8,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=RANDOM_STATE,
        verbose=100,
        early_stopping_rounds=100,
        allow_writing_files=False,
        thread_count=-1,
    )

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_columns,
        eval_set=(X_valid, y_valid),
        verbose=100,
    )

    probabilities = model.predict_proba(X_valid)[:, 1]
    validation_auc = roc_auc_score(y_valid, probabilities)

    model_path = MODEL_DIR / "day6_catboost_feature_engineering.cbm"
    report_path = REPORT_DIR / "day6_feature_engineering_metrics.json"
    importance_path = REPORT_DIR / "day6_feature_importance.csv"

    model.save_model(str(model_path))

    importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(importance_path, index=False)

    results = {
        "experiment": "CatBoost with feature engineering",
        "baseline_validation_roc_auc": BASELINE_AUC,
        "validation_roc_auc": float(validation_auc),
        "auc_difference_from_baseline": float(
            validation_auc - BASELINE_AUC
        ),
        "baseline_improved": bool(validation_auc > BASELINE_AUC),
        "best_iteration": int(model.get_best_iteration()),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "original_feature_count": len(original_columns),
        "engineered_feature_count": int(X.shape[1]),
        "categorical_features": categorical_columns,
        "random_state": RANDOM_STATE,
        "iterations": 1000,
        "depth": 8,
        "learning_rate": 0.05,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }

    report_path.write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print("\n--- Day 6 Results ---")
    print(f"Baseline validation ROC AUC: {BASELINE_AUC:.6f}")
    print(f"New validation ROC AUC:      {validation_auc:.6f}")
    print(
        "Difference:                  "
        f"{validation_auc - BASELINE_AUC:+.6f}"
    )
    print(f"Baseline improved: {validation_auc > BASELINE_AUC}")
    print(f"Best iteration: {model.get_best_iteration()}")
    print(f"Elapsed time: {results['elapsed_seconds']:.1f} seconds")
    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {report_path}")
    print(f"Feature importance saved: {importance_path}")
    print("\nTop 10 important features:")
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
