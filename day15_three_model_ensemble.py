import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
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


def prepare_data():
    print("Loading training data...")
    df = pd.read_csv(DATA_DIR / "train.csv")

    X = df.drop(columns=[TARGET, ID_COLUMN])
    y = df[TARGET].astype(int)

    for col in CATEGORICAL_COLUMNS:
        if col in X.columns:
            X[col] = X[col].fillna("Missing").astype(str)

    X_cat = X.copy()

    _, X_valid_cat, _, y_valid = train_test_split(
        X_cat,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_encoded = pd.get_dummies(
        X,
        columns=CATEGORICAL_COLUMNS,
        dtype=np.uint8,
    )

    X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce")
    X_encoded = X_encoded.replace([np.inf, -np.inf], np.nan)

    _, X_valid_xgb, _, _ = train_test_split(
        X_encoded,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return X_valid_cat, X_valid_xgb, y_valid


def main():
    X_valid_cat, X_valid_xgb, y_valid = prepare_data()

    print("\nLoading Day 4 CatBoost...")
    cat4 = CatBoostClassifier()
    cat4.load_model(str(MODEL_DIR / "day4_catboost.cbm"))

    print("Loading Day 11 CatBoost...")
    cat11 = CatBoostClassifier()
    cat11.load_model(str(MODEL_DIR / "day11_catboost_randomstrength2.cbm"))

    print("Loading Day 12 XGBoost...")
    xgb12 = XGBClassifier()
    xgb12.load_model(str(MODEL_DIR / "day12_xgboost.json"))

    print("\nGenerating predictions...")

    pred4 = cat4.predict_proba(X_valid_cat)[:, 1]
    pred11 = cat11.predict_proba(X_valid_cat)[:, 1]

    pred12 = xgb12.predict_proba(
        X_valid_xgb,
        iteration_range=(0, xgb12.best_iteration + 1),
    )[:, 1]

    individual = {
        "day4_catboost": float(roc_auc_score(y_valid, pred4)),
        "day11_catboost": float(roc_auc_score(y_valid, pred11)),
        "day12_xgboost": float(roc_auc_score(y_valid, pred12)),
    }

    print("\nIndividual model scores:")
    for name, score in individual.items():
        print(f"{name}: {score:.6f}")

    results = []

    # Two-model combinations
    for weight in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        blend = weight * pred11 + (1 - weight) * pred12

        results.append({
            "models": "day11_catboost + day12_xgboost",
            "weights": {
                "day11_catboost": weight,
                "day12_xgboost": 1 - weight,
            },
            "validation_roc_auc": float(
                roc_auc_score(y_valid, blend)
            ),
        })

    # Three-model combinations, weights in increments of 10%
    for w4, w11, w12 in product(
        range(0, 11),
        repeat=3,
    ):
        if w4 + w11 + w12 != 10:
            continue

        if min(w4, w11, w12) < 0:
            continue

        weights = np.array([w4, w11, w12], dtype=float) / 10

        blend = (
            weights[0] * pred4
            + weights[1] * pred11
            + weights[2] * pred12
        )

        results.append({
            "models": "three_model_ensemble",
            "weights": {
                "day4_catboost": float(weights[0]),
                "day11_catboost": float(weights[1]),
                "day12_xgboost": float(weights[2]),
            },
            "validation_roc_auc": float(
                roc_auc_score(y_valid, blend)
            ),
        })

    results.sort(
        key=lambda item: item["validation_roc_auc"],
        reverse=True,
    )

    print("\n========== TOP 10 ENSEMBLES ==========")

    for index, result in enumerate(results[:10], start=1):
        print(
            f"{index}. {result['models']} | "
            f"{result['weights']} | "
            f"AUC: {result['validation_roc_auc']:.6f}"
        )

    report = {
        "experiment": "Day 15 - Three-model ensemble",
        "validation_rows": len(y_valid),
        "individual_scores": individual,
        "best_ensemble": results[0],
        "top_10_ensembles": results[:10],
    }

    REPORT_DIR.mkdir(exist_ok=True)

    report_path = REPORT_DIR / "day15_ensemble_metrics.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nBest ensemble:")
    print(results[0])
    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
