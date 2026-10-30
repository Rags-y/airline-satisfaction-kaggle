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


def main():
    print("Loading training data...")
    df = pd.read_csv(DATA_DIR / "train.csv")

    y = df["satisfaction"].astype(int)
    X_raw = df.drop(columns=["satisfaction", "id"]).copy()

    categorical = [
        "Gender",
        "Customer Type",
        "Type of Travel",
        "Class",
    ]

    X_cat = X_raw.copy()
    X_encoded = X_raw.copy()

    for col in categorical:
        X_cat[col] = X_cat[col].fillna("Missing").astype(str)
        X_encoded[col] = X_encoded[col].fillna("Missing").astype(str)

    X_encoded = pd.get_dummies(
        X_encoded,
        columns=categorical,
        dtype=np.uint8,
    )

    X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce")
    X_encoded = X_encoded.replace([np.inf, -np.inf], np.nan)

    print("Loading models...")
    cat_model = CatBoostClassifier()
    cat_model.load_model(str(MODEL_DIR / "day11_catboost_randomstrength2.cbm"))

    xgb_model = XGBClassifier()
    xgb_model.load_model(str(MODEL_DIR / "day12_xgboost.json"))

    lgb_model = lgb.Booster(
        model_file=str(MODEL_DIR / "day12_lightgbm.txt")
    )

    xgb_features = xgb_model.get_booster().feature_names
    lgb_features = lgb_model.feature_name()

    if [x.replace(" ", "_") for x in xgb_features] != lgb_features:
        raise ValueError("Model feature mapping mismatch.")

    X_xgb = X_encoded.reindex(columns=xgb_features, fill_value=0)

    X_lgb = X_encoded.copy()
    X_lgb.columns = [x.replace(" ", "_") for x in X_lgb.columns]
    X_lgb = X_lgb.reindex(columns=lgb_features, fill_value=0)

    results = []

    # Test two independent stratified validation splits.
    for seed in [42, 123]:
        print(f"\n========== VALIDATION SEED {seed} ==========")

        _, valid_idx = train_test_split(
            np.arange(len(df)),
            test_size=0.20,
            random_state=seed,
            stratify=y,
        )

        y_valid = y.iloc[valid_idx]

        pred_cat = cat_model.predict_proba(
            X_cat.iloc[valid_idx]
        )[:, 1]

        pred_xgb = xgb_model.predict_proba(
            X_xgb.iloc[valid_idx],
            iteration_range=(0, xgb_model.best_iteration + 1),
        )[:, 1]

        pred_lgb = lgb_model.predict(
            X_lgb.iloc[valid_idx],
            num_iteration=1400,
        )

        print("Individual AUC:")
        print("CatBoost:", roc_auc_score(y_valid, pred_cat))
        print("XGBoost:", roc_auc_score(y_valid, pred_xgb))
        print("LightGBM:", roc_auc_score(y_valid, pred_lgb))

        for cat_weight in [0.20, 0.225, 0.25, 0.275, 0.30]:
            for xgb_weight in np.arange(0.55, 0.651, 0.025):
                lgb_weight = round(1.0 - cat_weight - xgb_weight, 3)

                if lgb_weight < 0:
                    continue

                blend = (
                    cat_weight * pred_cat
                    + xgb_weight * pred_xgb
                    + lgb_weight * pred_lgb
                )

                results.append({
                    "seed": seed,
                    "catboost_weight": round(cat_weight, 3),
                    "xgboost_weight": round(float(xgb_weight), 3),
                    "lightgbm_weight": lgb_weight,
                    "validation_roc_auc": float(
                        roc_auc_score(y_valid, blend)
                    ),
                })

    print("\n========== TOP 15 RESULTS ==========")

    results.sort(
        key=lambda x: x["validation_roc_auc"],
        reverse=True,
    )

    for i, result in enumerate(results[:15], 1):
        print(
            f"{i}. Seed {result['seed']} | "
            f"Cat {result['catboost_weight']:.1%} + "
            f"XGB {result['xgboost_weight']:.1%} + "
            f"LGB {result['lightgbm_weight']:.1%} | "
            f"AUC {result['validation_roc_auc']:.6f}"
        )

    REPORT_DIR.mkdir(exist_ok=True)

    report = {
        "experiment": "Day 21 - Three-model fine weight search",
        "validation_seeds": [42, 123],
        "results": results,
    }

    path = REPORT_DIR / "day21_ensemble_metrics.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nReport saved: {path}")


if __name__ == "__main__":
    main()
