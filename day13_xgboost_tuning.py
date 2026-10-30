import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
REPORT_DIR = Path("reports")
MODEL_DIR = Path("models")

REPORT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "satisfaction"
ID_COLUMN = "id"
RANDOM_STATE = 42

CATEGORICAL_COLUMNS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]


def prepare_data():
    print("Loading training data...")
    df = pd.read_csv(DATA_DIR / "train.csv")

    X = df.drop(columns=[TARGET, ID_COLUMN])
    y = df[TARGET].astype(int)

    categorical_columns = [
        col for col in CATEGORICAL_COLUMNS if col in X.columns
    ]

    for col in categorical_columns:
        X[col] = X[col].fillna("Missing").astype(str)

    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        dtype=np.uint8,
    )

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"Training rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_valid):,}")
    print(f"Features: {X.shape[1]}")

    return X_train, X_valid, y_train, y_valid


def train_model(depth, X_train, X_valid, y_train, y_valid):
    print(f"\n========== XGBOOST DEPTH {depth} ==========")
    start = time.time()

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=depth,
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

    model_path = MODEL_DIR / f"day13_xgboost_depth{depth}.json"
    importance_path = (
        REPORT_DIR / f"day13_xgboost_depth{depth}_importance.csv"
    )

    model.save_model(str(model_path))

    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance.to_csv(importance_path, index=False)

    result = {
        "model": "XGBoost",
        "max_depth": depth,
        "validation_roc_auc": float(auc),
        "best_iteration": int(model.best_iteration),
        "elapsed_seconds": round(time.time() - start, 2),
        "parameters": {
            "learning_rate": 0.03,
            "min_child_weight": 5,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
        },
    }

    print(f"\nDepth: {depth}")
    print(f"Validation ROC AUC: {auc:.6f}")
    print(f"Best iteration: {model.best_iteration}")
    print(f"Elapsed: {result['elapsed_seconds']} seconds")
    print(f"Model saved: {model_path}")

    return result


def main():
    X_train, X_valid, y_train, y_valid = prepare_data()

    results = []

    for depth in [7, 9]:
        results.append(
            train_model(
                depth,
                X_train,
                X_valid,
                y_train,
                y_valid,
            )
        )

    report = {
        "experiment": "Day 13 - XGBoost depth tuning",
        "random_state": RANDOM_STATE,
        "training_rows": len(X_train),
        "validation_rows": len(X_valid),
        "results": results,
    }

    report_path = REPORT_DIR / "day13_xgboost_tuning_metrics.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n========== DAY 13 RESULTS ==========")

    for result in sorted(
        results,
        key=lambda item: item["validation_roc_auc"],
        reverse=True,
    ):
        print(
            f"Depth {result['max_depth']}: "
            f"{result['validation_roc_auc']:.6f} "
            f"(iteration {result['best_iteration']})"
        )

    print("\nDay 12 depth-8 reference: 0.958002")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
