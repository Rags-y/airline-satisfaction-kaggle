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

REPORT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

TARGET = "satisfaction"
ID_COLUMN = "id"
RANDOM_STATE = 42
RANDOM_STRENGTH = 2

CATEGORICAL_COLUMNS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]


def main():
    start_time = time.time()

    print("Loading training data...")
    df = pd.read_csv(DATA_DIR / "train.csv")

    X = df.drop(columns=[TARGET, ID_COLUMN])
    y = df[TARGET].astype(int)

    categorical_columns = [
        col for col in CATEGORICAL_COLUMNS if col in X.columns
    ]

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
    print(f"Categorical features: {categorical_columns}")
    print(f"Random strength: {RANDOM_STRENGTH}")
    print("Training CatBoost...")

    model = CatBoostClassifier(
        iterations=1000,
        depth=8,
        learning_rate=0.05,
        random_strength=RANDOM_STRENGTH,
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

    model_path = MODEL_DIR / "day11_catboost_randomstrength2.cbm"
    report_path = REPORT_DIR / "day11_random_strength_metrics.json"
    importance_path = REPORT_DIR / "day11_feature_importance.csv"

    model.save_model(str(model_path))

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    feature_importance.to_csv(importance_path, index=False)

    results = {
        "experiment": "CatBoost random strength",
        "model": "CatBoostClassifier",
        "validation_roc_auc": float(validation_auc),
        "best_iteration": int(model.get_best_iteration()),
        "training_rows": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "random_state": RANDOM_STATE,
        "iterations": 1000,
        "depth": 8,
        "learning_rate": 0.05,
        "random_strength": RANDOM_STRENGTH,
        "categorical_features": categorical_columns,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }

    report_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n--- Day 11 Results ---")
    print(f"Validation ROC AUC: {validation_auc:.6f}")
    print(f"Best iteration: {model.get_best_iteration()}")
    print(f"Random strength: {RANDOM_STRENGTH}")
    print(f"Elapsed time: {results['elapsed_seconds']:.1f} seconds")
    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {report_path}")
    print(f"Feature importance saved: {importance_path}")
    print("\nTop 10 important features:")
    print(feature_importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
