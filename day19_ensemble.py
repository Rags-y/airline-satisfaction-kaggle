import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

CATEGORICAL_COLUMNS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]


def main():
    print("Loading training data...")
    df = pd.read_csv(DATA_DIR / "train.csv")

    X = df.drop(columns=["satisfaction", "id"]).copy()
    y = df["satisfaction"].astype(int)

    for col in CATEGORICAL_COLUMNS:
        X[col] = X[col].fillna("Missing").astype(str)

    X = pd.get_dummies(
        X,
        columns=CATEGORICAL_COLUMNS,
        dtype=np.uint8,
    )

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    _, X_valid, _, y_valid = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Validation rows: {len(y_valid):,}")

    print("\nLoading LightGBM...")
    lgb_model = lgb.Booster(
        model_file=str(MODEL_DIR / "day12_lightgbm.txt")
    )

    print("Loading XGBoost...")
    xgb_model = XGBClassifier()
    xgb_model.load_model(
        str(MODEL_DIR / "day12_xgboost.json")
    )

    print("\nGenerating predictions...")

    pred_lgb = lgb_model.predict(
        X_valid,
        num_iteration=1400,
    )

    pred_xgb = xgb_model.predict_proba(
        X_valid,
        iteration_range=(0, xgb_model.best_iteration + 1),
    )[:, 1]

    lgb_auc = roc_auc_score(y_valid, pred_lgb)
    xgb_auc = roc_auc_score(y_valid, pred_xgb)

    print("\nIndividual scores:")
    print(f"LightGBM: {lgb_auc:.6f}")
    print(f"XGBoost:  {xgb_auc:.6f}")

    results = []

    for lgb_weight in np.arange(0.0, 1.0, 0.05):
        xgb_weight = 1.0 - lgb_weight

        blend = (
            lgb_weight * pred_lgb
            + xgb_weight * pred_xgb
        )

        results.append({
            "lightgbm_weight": round(float(lgb_weight), 2),
            "xgboost_weight": round(float(xgb_weight), 2),
            "validation_roc_auc": float(
                roc_auc_score(y_valid, blend)
            ),
        })

    results.sort(
        key=lambda item: item["validation_roc_auc"],
        reverse=True,
    )

    print("\n========== TOP 10 BLENDS ==========")

    for rank, result in enumerate(results[:10], start=1):
        print(
            f"{rank}. LightGBM {result['lightgbm_weight']:.0%} + "
            f"XGBoost {result['xgboost_weight']:.0%} | "
            f"AUC: {result['validation_roc_auc']:.6f}"
        )

    report = {
        "experiment": "Day 19 - LightGBM XGBoost ensemble",
        "individual_scores": {
            "lightgbm": float(lgb_auc),
            "xgboost": float(xgb_auc),
        },
        "best_ensemble": results[0],
        "top_10_ensembles": results[:10],
    }

    REPORT_DIR.mkdir(exist_ok=True)

    report_path = REPORT_DIR / "day19_ensemble_metrics.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print("\nBest ensemble:")
    print(json.dumps(results[0], indent=2))
    print("\nDay 17 reference: 0.958080")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
