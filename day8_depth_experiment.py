import json
import time
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
MODEL_PATH = ROOT / "models" / "day8_catboost_depth7.cbm"
METRICS_PATH = ROOT / "reports" / "day8_depth_experiment_metrics.json"
IMPORTANCE_PATH = ROOT / "reports" / "day8_feature_importance.csv"

TARGET = "satisfaction"
ID_COL = "id"

def main():
    start_time = time.time()
    df = pd.read_csv(DATA_PATH)

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET, ID_COL])

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    for col in categorical_cols:
        X[col] = X[col].fillna("Missing").astype(str)

    for col in X.columns:
        if col not in categorical_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = CatBoostClassifier(
        iterations=1000,
        depth=7,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        early_stopping_rounds=100,
        verbose=100,
        thread_count=-1
    )

    print("Training Day 8 CatBoost model...")
    print("Configuration: depth=7, iterations=1000, learning_rate=0.05")

    model.fit(
        X_train,
        y_train,
        cat_features=categorical_cols,
        eval_set=(X_val, y_val),
        use_best_model=True
    )

    val_probabilities = model.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, val_probabilities)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(str(MODEL_PATH))

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.get_feature_importance()
    }).sort_values("importance", ascending=False)

    feature_importance.to_csv(IMPORTANCE_PATH, index=False)

    elapsed = time.time() - start_time

    metrics = {
        "experiment": "Day 8 - CatBoost Depth 7",
        "iterations": 1000,
        "depth": 7,
        "learning_rate": 0.05,
        "random_seed": 42,
        "best_iteration": int(model.best_iteration_),
        "validation_roc_auc": float(val_auc),
        "elapsed_seconds": round(elapsed, 2),
        "reference_day4_validation_roc_auc": 0.9577167082,
        "difference_from_day4": round(val_auc - 0.9577167082, 10)
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\nDay 8 Results")
    print(json.dumps(metrics, indent=4))

    print("\nTop 10 Features")
    print(feature_importance.head(10).to_string(index=False))

    print(f"\nModel saved: {MODEL_PATH}")
    print(f"Metrics saved: {METRICS_PATH}")
    print(f"Feature importance saved: {IMPORTANCE_PATH}")

if __name__ == "__main__":
    main()
