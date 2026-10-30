from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

CATEGORICAL_COLUMNS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]


def main():
    print("Loading test data...")
    test = pd.read_csv(DATA_DIR / "test.csv")

    test_ids = test["id"].copy()
    X_raw = test.drop(columns=["id"]).copy()

    # CatBoost preprocessing
    X_cat = X_raw.copy()

    for col in CATEGORICAL_COLUMNS:
        X_cat[col] = X_cat[col].fillna("Missing").astype(str)

    # One-hot encoding for XGBoost and LightGBM
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

    print("Loading trained models...")

    cat_model = CatBoostClassifier()
    cat_model.load_model(
        str(MODEL_DIR / "day11_catboost_randomstrength2.cbm")
    )

    xgb_model = XGBClassifier()
    xgb_model.load_model(
        str(MODEL_DIR / "day12_xgboost.json")
    )

    lgb_model = lgb.Booster(
        model_file=str(MODEL_DIR / "day12_lightgbm.txt")
    )

    # Each model gets its own exact training feature names and order.
    xgb_features = xgb_model.get_booster().feature_names
    lgb_features = lgb_model.feature_name()

    if xgb_features is None:
        raise ValueError("XGBoost feature names are unavailable.")

    # Verify the naming convention maps to the same underlying columns.
    normalized_xgb = [name.replace(" ", "_") for name in xgb_features]

    if normalized_xgb != lgb_features:
        raise ValueError(
            "Feature mapping differs between XGBoost and LightGBM."
        )

    X_xgb = X_encoded.reindex(
        columns=xgb_features,
        fill_value=0,
    )

    X_lgb = X_encoded.copy()
    X_lgb.columns = [
        column.replace(" ", "_") for column in X_lgb.columns
    ]

    X_lgb = X_lgb.reindex(
        columns=lgb_features,
        fill_value=0,
    )

    print(f"XGBoost features: {X_xgb.shape[1]}")
    print(f"LightGBM features: {X_lgb.shape[1]}")

    print("\nGenerating predictions...")

    pred_cat = cat_model.predict_proba(X_cat)[:, 1]

    pred_xgb = xgb_model.predict_proba(
        X_xgb,
        iteration_range=(0, xgb_model.best_iteration + 1),
    )[:, 1]

    pred_lgb = lgb_model.predict(
        X_lgb,
        num_iteration=1400,
    )

    # Day 20 best validation blend
    predictions = (
        0.20 * pred_cat
        + 0.65 * pred_xgb
        + 0.15 * pred_lgb
    )

    submission = pd.DataFrame({
        "id": test_ids,
        "satisfaction": predictions,
    })

    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    if list(submission.columns) != list(sample.columns):
        raise ValueError("Submission columns do not match sample.")

    if len(submission) != len(sample):
        raise ValueError("Submission row count does not match sample.")

    if not submission["id"].equals(sample["id"]):
        raise ValueError("Submission IDs do not match sample.")

    if submission["satisfaction"].isna().any():
        raise ValueError("Missing predictions found.")

    if not submission["satisfaction"].between(0, 1).all():
        raise ValueError("Predictions outside [0, 1].")

    OUTPUT_DIR.mkdir(exist_ok=True)

    output_path = OUTPUT_DIR / "submission_day21_final_ensemble.csv"
    submission.to_csv(output_path, index=False)

    print("\n========== VALIDATION ==========")
    print(f"Rows: {len(submission):,}")
    print(f"Columns: {list(submission.columns)}")
    print(f"Missing predictions: {submission['satisfaction'].isna().sum()}")
    print(f"Minimum probability: {predictions.min():.6f}")
    print(f"Maximum probability: {predictions.max():.6f}")
    print(f"Output: {output_path}")

    print("\nFirst five rows:")
    print(submission.head().to_string(index=False))

    print("\nSubmission created and validated successfully.")


if __name__ == "__main__":
    main()

