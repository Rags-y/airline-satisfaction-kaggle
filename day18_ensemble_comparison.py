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


def main():
    print("Loading training data...")
    df = pd.read_csv(DATA_DIR / "train.csv")

    X = df.drop(columns=[TARGET, ID_COLUMN]).copy()
    y = df[TARGET].astype(int)

    for col in CATEGORICAL_COLUMNS:
        X[col] = X[col].fillna("Missing").astype(str)

    # CatBoost validation data
    _, X_valid_cat, _, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # XGBoost validation data
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

    print(f"Validation rows: {len(y_valid):,}")

    print("\nLoading Day 9 CatBoost...")
    cat9 = CatBoostClassifier()
    cat9.load_model(str(MODEL_DIR / "day9_catboost_lr003.cbm"))

    print("Loading Day 11 CatBoost...")
    cat11 = CatBoostClassifier()
    cat11.load_model(
        str(MODEL_DIR / "day11_catboost_randomstrength2.cbm")
    )

    print("Loading Day 12 XGBoost...")
    xgb12 = XGBClassifier()
    xgb12.load_model(str(MODEL_DIR / "day12_xgboost.json"))

    print("\nGenerating validation predictions...")

    pred9 = cat9.predict_proba(X_valid_cat)[:, 1]
    pred11 = cat11.predict_proba(X_valid_cat)[:, 1]

    pred12 = xgb12.predict_proba(
        X_valid_xgb,
        iteration_range=(0, xgb12.best_iteration + 1),
    )[:, 1]

    individual_scores = {
        "day9_catboost": float(roc_auc_score(y_valid, pred9)),
        "day11_catboost": float(roc_auc_score(y_valid, pred11)),
        "day12_xgboost": float(roc_auc_score(y_valid, pred12)),
    }

    print("\n========== INDIVIDUAL MODEL SCORES ==========")
    for name, score in individual_scores.items():
        print(f"{name}: {score:.6f}")

    results = []

    # Day 9 CatBoost + Day 12 XGBoost
    for cat_weight in np.arange(0.1, 1.0, 0.1):
        xgb_weight = 1.0 - cat_weight

        blend = cat_weight * pred9 + xgb_weight * pred12

        results.append({
            "ensemble": "day9_catboost + day12_xgboost",
            "catboost_model": "day9",
            "catboost_weight": round(float(cat_weight), 2),
            "xgboost_weight": round(float(xgb_weight), 2),
            "validation_roc_auc": float(
                roc_auc_score(y_valid, blend)
            ),
        })

    # Day 11 CatBoost + Day 12 XGBoost (Day 17 reference)
    for cat_weight in np.arange(0.1, 1.0, 0.1):
        xgb_weight = 1.0 - cat_weight

        blend = cat_weight * pred11 + xgb_weight * pred12

        results.append({
            "ensemble": "day11_catboost + day12_xgboost",
            "catboost_model": "day11",
            "catboost_weight": round(float(cat_weight), 2),
            "xgboost_weight": round(float(xgb_weight), 2),
            "validation_roc_auc": float(
                roc_auc_score(y_valid, blend)
            ),
        })

    results.sort(
        key=lambda item: item["validation_roc_auc"],
        reverse=True,
    )

    print("\n========== TOP 10 ENSEMBLES ==========")

    for rank, result in enumerate(results[:10], start=1):
        print(
            f"{rank}. {result['ensemble']} | "
            f"CatBoost: {result['catboost_weight']:.0%} | "
            f"XGBoost: {result['xgboost_weight']:.0%} | "
            f"AUC: {result['validation_roc_auc']:.6f}"
        )

    report = {
        "experiment": "Day 18 - Day 9 vs Day 11 ensemble",
        "validation_rows": len(y_valid),
        "individual_scores": individual_scores,
        "best_ensemble": results[0],
        "top_10_ensembles": results[:10],
        "day17_reference_validation_auc": 0.9580802998645265,
    }

    REPORT_DIR.mkdir(exist_ok=True)

    report_path = REPORT_DIR / "day18_ensemble_comparison.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\n========== DAY 18 BEST RESULT ==========")
    print(json.dumps(results[0], indent=2))
    print(f"\nDay 17 validation reference: 0.958080")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
