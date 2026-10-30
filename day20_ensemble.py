import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
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

REFERENCE_AUC = 0.9580802998645265


def main():
    print("Loading training data...")
    df = pd.read_csv(DATA_DIR / "train.csv")

    y = df["satisfaction"].astype(int)
    X_raw = df.drop(columns=["satisfaction", "id"]).copy()

    # Prepare original categorical features for CatBoost.
    X_cat = X_raw.copy()

    for col in CATEGORICAL_COLUMNS:
        X_cat[col] = X_cat[col].fillna("Missing").astype(str)

    # Prepare one-hot encoded features for LightGBM and XGBoost.
    X_encoded = X_raw.copy()

    for col in CATEGORICAL_COLUMNS:
        X_encoded[col] = X_encoded[col].fillna("Missing").astype(str)

    X_encoded = pd.get_dummies(
        X_encoded,
        columns=CATEGORICAL_COLUMNS,
        dtype=np.uint8,
    )

    X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce")
    X_encoded = X_encoded.replace([np.inf, -np.inf], np.nan)

    # Reproduce the same validation split.
    train_idx, valid_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_valid_cat = X_cat.iloc[valid_idx]
    X_valid_encoded = X_encoded.iloc[valid_idx]
    y_valid = y.iloc[valid_idx]

    print(f"Validation rows: {len(y_valid):,}")

    print("\nLoading CatBoost Day 11...")
    cat_model = CatBoostClassifier()
    cat_model.load_model(
        str(MODEL_DIR / "day11_catboost_randomstrength2.cbm")
    )

    print("Loading XGBoost Day 12...")
    xgb_model = XGBClassifier()
    xgb_model.load_model(
        str(MODEL_DIR / "day12_xgboost.json")
    )

    print("Loading LightGBM Day 12...")
    lgb_model = lgb.Booster(
        model_file=str(MODEL_DIR / "day12_lightgbm.txt")
    )

    print("\nGenerating validation predictions...")

    pred_cat = cat_model.predict_proba(X_valid_cat)[:, 1]

    pred_xgb = xgb_model.predict_proba(
        X_valid_encoded,
        iteration_range=(0, xgb_model.best_iteration + 1),
    )[:, 1]

    pred_lgb = lgb_model.predict(
        X_valid_encoded,
        num_iteration=1400,
    )

    individual_scores = {
        "catboost_day11": float(roc_auc_score(y_valid, pred_cat)),
        "xgboost_day12": float(roc_auc_score(y_valid, pred_xgb)),
        "lightgbm_day12": float(roc_auc_score(y_valid, pred_lgb)),
    }

    print("\nIndividual scores:")
    for name, score in individual_scores.items():
        print(f"{name}: {score:.6f}")

    print("\nTesting three-model weight combinations...")

    results = []

    weights = np.arange(0, 1.0001, 0.05)

    for cat_weight in weights:
        for xgb_weight in weights:
            lgb_weight = round(1.0 - cat_weight - xgb_weight, 2)

            if lgb_weight < -0.000001:
                continue

            lgb_weight = max(0.0, lgb_weight)

            blend = (
                cat_weight * pred_cat
                + xgb_weight * pred_xgb
                + lgb_weight * pred_lgb
            )

            score = roc_auc_score(y_valid, blend)

            results.append({
                "catboost_weight": round(float(cat_weight), 2),
                "xgboost_weight": round(float(xgb_weight), 2),
                "lightgbm_weight": round(float(lgb_weight), 2),
                "validation_roc_auc": float(score),
            })

    results.sort(
        key=lambda item: item["validation_roc_auc"],
        reverse=True,
    )

    print("\n========== TOP 20 THREE-MODEL BLENDS ==========")

    for rank, result in enumerate(results[:20], start=1):
        print(
            f"{rank}. Cat {result['catboost_weight']:.0%} + "
            f"XGB {result['xgboost_weight']:.0%} + "
            f"LGB {result['lightgbm_weight']:.0%} | "
            f"AUC: {result['validation_roc_auc']:.6f}"
        )

    best = results[0]

    print("\n========== FINAL COMPARISON ==========")
    print(f"Day 17 reference: {REFERENCE_AUC:.6f}")
    print(f"Day 20 best:      {best['validation_roc_auc']:.6f}")
    print(
        f"Difference:      "
        f"{best['validation_roc_auc'] - REFERENCE_AUC:+.6f}"
    )

    report = {
        "experiment": "Day 20 - CatBoost XGBoost LightGBM ensemble",
        "individual_scores": individual_scores,
        "reference_day17_validation_auc": REFERENCE_AUC,
        "best_ensemble": best,
        "top_20_ensembles": results[:20],
    }

    REPORT_DIR.mkdir(exist_ok=True)

    report_path = REPORT_DIR / "day20_ensemble_metrics.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"\nReport saved: {report_path}")


if __name__ == "__main__":
    main()
