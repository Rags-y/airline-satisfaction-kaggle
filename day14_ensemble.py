import json
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

    X_train_cat, X_valid_cat, y_train, y_valid = train_test_split(
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

    X_train_xgb, X_valid_xgb, _, _ = train_test_split(
        X_encoded,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    return (
        X_valid_cat,
        X_valid_xgb,
        y_valid,
    )


def main():
    X_valid_cat, X_valid_xgb, y_valid = prepare_data()

    print("\nLoading CatBoost Day 4 model...")
    cat_model = CatBoostClassifier()
    cat_model.load_model(str(MODEL_DIR / "day4_catboost.cbm"))

    print("Loading XGBoost Day 12 model...")
    xgb_model = XGBClassifier()
    xgb_model.load_model(str(MODEL_DIR / "day12_xgboost.json"))

    print("\nGenerating validation predictions...")

    cat_predictions = cat_model.predict_proba(
        X_valid_cat
    )[:, 1]

    xgb_predictions = xgb_model.predict_proba(
        X_valid_xgb,
        iteration_range=(0, xgb_model.best_iteration + 1),
    )[:, 1]

    cat_auc = roc_auc_score(y_valid, cat_predictions)
    xgb_auc = roc_auc_score(y_valid, xgb_predictions)

    print("\nIndividual model scores:")
    print(f"CatBoost Day 4: {cat_auc:.6f}")
    print(f"XGBoost Day 12: {xgb_auc:.6f}")

    weights = [0.3, 0.4, 0.5, 0.6, 0.7]

    results = []

    for cat_weight in weights:
        xgb_weight = 1 - cat_weight

        blended = (
            cat_weight * cat_predictions
            + xgb_weight * xgb_predictions
        )

        auc = roc_auc_score(y_valid, blended)

        results.append({
            "catboost_weight": cat_weight,
            "xgboost_weight": xgb_weight,
            "validation_roc_auc": float(auc),
        })

        print(
            f"CatBoost {cat_weight:.0%} + "
            f"XGBoost {xgb_weight:.0%}: {auc:.6f}"
        )

    best = max(
        results,
        key=lambda item: item["validation_roc_auc"],
    )

    report = {
        "experiment": "Day 14 CatBoost XGBoost ensemble",
        "individual_scores": {
            "catboost_day4": float(cat_auc),
            "xgboost_day12": float(xgb_auc),
        },
        "ensemble_results": results,
        "best_ensemble": best,
    }

    REPORT_DIR.mkdir(exist_ok=True)

    report_path = REPORT_DIR / "day14_ensemble_metrics.json"

    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n========== BEST ENSEMBLE ==========")
    print(f"CatBoost weight: {best['catboost_weight']:.0%}")
    print(f"XGBoost weight: {best['xgboost_weight']:.0%}")
    print(f"Validation ROC AUC: {best['validation_roc_auc']:.6f}")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
