import json
import time
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier
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

CONFIGS = [
    {
        "name": "depth6_lr005",
        "iterations": 1200,
        "depth": 6,
        "learning_rate": 0.05,
    },
    {
        "name": "depth8_lr003",
        "iterations": 1500,
        "depth": 8,
        "learning_rate": 0.03,
    },
]


def main():
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(DATA_DIR / "train.csv")

    X = df.drop(columns=[TARGET, ID_COLUMN]).copy()
    y = df[TARGET].astype(int)

    categorical = [
        col for col in CATEGORICAL_COLUMNS if col in X.columns
    ]

    for col in categorical:
        X[col] = X[col].fillna("Missing").astype(str)

    for col in X.columns:
        if col not in categorical:
            X[col] = pd.to_numeric(X[col], errors="coerce")

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results = []

    for config in CONFIGS:
        name = config["name"]
        print(f"\n{'=' * 55}")
        print(f"Training configuration: {name}")
        print(config)

        start = time.time()

        model = CatBoostClassifier(
            iterations=config["iterations"],
            depth=config["depth"],
            learning_rate=config["learning_rate"],
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=RANDOM_STATE,
            early_stopping_rounds=100,
            verbose=100,
            allow_writing_files=False,
            thread_count=-1,
        )

        model.fit(
            X_train,
            y_train,
            cat_features=categorical,
            eval_set=(X_valid, y_valid),
            verbose=100,
        )

        probabilities = model.predict_proba(X_valid)[:, 1]
        auc = roc_auc_score(y_valid, probabilities)
        elapsed = round(time.time() - start, 2)

        model_path = MODEL_DIR / f"day5_catboost_{name}.cbm"
        model.save_model(str(model_path))

        result = {
            **config,
            "validation_roc_auc": float(auc),
            "best_iteration": int(model.get_best_iteration()),
            "elapsed_seconds": elapsed,
            "model_path": str(model_path),
        }
        results.append(result)

        print(f"Validation ROC AUC: {auc:.6f}")
        print(f"Best iteration: {model.get_best_iteration()}")
        print(f"Elapsed time: {elapsed:.1f} seconds")
        print(f"Model saved: {model_path}")

        pd.DataFrame(results).to_csv(
            REPORT_DIR / "day5_catboost_tuning.csv",
            index=False,
        )
        (REPORT_DIR / "day5_catboost_tuning.json").write_text(
            json.dumps(results, indent=2),
            encoding="utf-8",
        )

    print("\n" + "=" * 55)
    print("DAY 5 MODEL COMPARISON")
    print(f"Day 4 reference ROC AUC: {0.9577167082:.6f}")

    comparison = pd.DataFrame(results).sort_values(
        "validation_roc_auc", ascending=False
    )
    print(comparison[
        [
            "name",
            "depth",
            "learning_rate",
            "iterations",
            "best_iteration",
            "validation_roc_auc",
            "elapsed_seconds",
        ]
    ].to_string(index=False))

    best = comparison.iloc[0]
    if best["validation_roc_auc"] > 0.9577167082:
        print(f"\nBest candidate: {best['name']}")
    else:
        print("\nNeither candidate beat the Day 4 validation score.")
        print("Keep the Day 4 model as the current reference.")

    print("\nResults saved in reports/day5_catboost_tuning.csv and .json")


if __name__ == "__main__":
    main()
