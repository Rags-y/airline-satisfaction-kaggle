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

CATEGORICAL = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]

CANDIDATES = {
    "Day20": (0.25, 0.60, 0.15),
    "Day21_best": (0.20, 0.65, 0.15),
    "Middle": (0.225, 0.625, 0.15),
}


def main():
    df = pd.read_csv(DATA_DIR / "train.csv")
    y = df["satisfaction"].astype(int)
    X = df.drop(columns=["satisfaction", "id"]).copy()

    X_cat = X.copy()
    X_encoded = X.copy()

    for col in CATEGORICAL:
        X_cat[col] = X_cat[col].fillna("Missing").astype(str)
        X_encoded[col] = X_encoded[col].fillna("Missing").astype(str)

    X_encoded = pd.get_dummies(
        X_encoded,
        columns=CATEGORICAL,
        dtype=np.uint8,
    )
    X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce")
    X_encoded = X_encoded.replace([np.inf, -np.inf], np.nan)

    cat = CatBoostClassifier()
    cat.load_model(str(MODEL_DIR / "day11_catboost_randomstrength2.cbm"))

    xgb = XGBClassifier()
    xgb.load_model(str(MODEL_DIR / "day12_xgboost.json"))

    lgbm = lgb.Booster(model_file=str(MODEL_DIR / "day12_lightgbm.txt"))

    xgb_features = xgb.get_booster().feature_names
    lgb_features = lgbm.feature_name()

    if [c.replace(" ", "_") for c in xgb_features] != lgb_features:
        raise ValueError("Feature mismatch.")

    X_xgb = X_encoded.reindex(columns=xgb_features, fill_value=0)

    X_lgb = X_encoded.copy()
    X_lgb.columns = [c.replace(" ", "_") for c in X_lgb.columns]
    X_lgb = X_lgb.reindex(columns=lgb_features, fill_value=0)

    all_results = []

    for seed in [42, 123]:
        _, valid_idx = train_test_split(
            np.arange(len(df)),
            test_size=0.20,
            random_state=seed,
            stratify=y,
        )

        y_valid = y.iloc[valid_idx]

        p_cat = cat.predict_proba(X_cat.iloc[valid_idx])[:, 1]
        p_xgb = xgb.predict_proba(
            X_xgb.iloc[valid_idx],
            iteration_range=(0, xgb.best_iteration + 1),
        )[:, 1]
        p_lgb = lgbm.predict(
            X_lgb.iloc[valid_idx],
            num_iteration=1400,
        )

        print(f"\n========== SEED {seed} ==========")

        for name, weights in CANDIDATES.items():
            wc, wx, wl = weights
            blend = wc * p_cat + wx * p_xgb + wl * p_lgb
            score = roc_auc_score(y_valid, blend)

            all_results.append({
                "seed": seed,
                "candidate": name,
                "catboost_weight": wc,
                "xgboost_weight": wx,
                "lightgbm_weight": wl,
                "validation_roc_auc": float(score),
            })

            print(f"{name}: {score:.6f}")

    print("\n========== COMPARISON ==========")

    for name in CANDIDATES:
        scores = [
            r["validation_roc_auc"]
            for r in all_results
            if r["candidate"] == name
        ]
        print(
            f"{name}: seed42={scores[0]:.6f}, "
            f"seed123={scores[1]:.6f}, "
            f"mean={np.mean(scores):.6f}"
        )

    Path("reports").mkdir(exist_ok=True)
    Path("reports/day21_candidate_comparison.json").write_text(
        json.dumps(all_results, indent=2),
        encoding="utf-8",
    )

    print("\nReport saved: reports/day21_candidate_comparison.json")


if __name__ == "__main__":
    main()
