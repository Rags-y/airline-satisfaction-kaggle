from pathlib import Path

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from xgboost import XGBClassifier

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
OUTPUT_DIR = Path("outputs")

ID_COLUMN = "id"
CATEGORICAL_COLUMNS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]

CATBOOST_WEIGHT = 0.30
XGBOOST_WEIGHT = 0.70


def main():
    print("Loading competition data...")

    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    test_ids = test[ID_COLUMN].copy()

    # Match Day 15 CatBoost preprocessing.
    X_train_cat = train.drop(
        columns=["satisfaction", ID_COLUMN]
    ).copy()

    X_test_cat = test.drop(
        columns=[ID_COLUMN]
    ).copy()

    for col in CATEGORICAL_COLUMNS:
        X_train_cat[col] = X_train_cat[col].fillna("Missing").astype(str)
        X_test_cat[col] = X_test_cat[col].fillna("Missing").astype(str)

    # Match Day 15 XGBoost preprocessing.
    X_train_xgb = pd.get_dummies(
        X_train_cat,
        columns=CATEGORICAL_COLUMNS,
        dtype=np.uint8,
    )

    X_test_xgb = pd.get_dummies(
        X_test_cat,
        columns=CATEGORICAL_COLUMNS,
        dtype=np.uint8,
    )

    # Keep exactly the training feature columns and their order.
    X_test_xgb = X_test_xgb.reindex(
        columns=X_train_xgb.columns,
        fill_value=0,
    )

    X_test_xgb = X_test_xgb.apply(pd.to_numeric, errors="coerce")
    X_test_xgb = X_test_xgb.replace([np.inf, -np.inf], np.nan)

    print(f"Test rows: {len(test):,}")
    print(f"CatBoost features: {X_test_cat.shape[1]}")
    print(f"XGBoost features: {X_test_xgb.shape[1]}")

    print("\nLoading Day 11 CatBoost model...")
    cat_model = CatBoostClassifier()
    cat_model.load_model(
        str(MODEL_DIR / "day11_catboost_randomstrength2.cbm")
    )

    print("Loading Day 12 XGBoost model...")
    xgb_model = XGBClassifier()
    xgb_model.load_model(
        str(MODEL_DIR / "day12_xgboost.json")
    )

    print("\nGenerating CatBoost predictions...")
    cat_predictions = cat_model.predict_proba(
        X_test_cat
    )[:, 1]

    print("Generating XGBoost predictions...")
    xgb_predictions = xgb_model.predict_proba(
        X_test_xgb,
        iteration_range=(0, xgb_model.best_iteration + 1),
    )[:, 1]

    print("\nCombining predictions...")
    final_predictions = (
        CATBOOST_WEIGHT * cat_predictions
        + XGBOOST_WEIGHT * xgb_predictions
    )

    submission = pd.DataFrame({
        ID_COLUMN: test_ids,
        "satisfaction": final_predictions,
    })

    # Validate submission.
    assert len(submission) == len(test)
    assert submission[ID_COLUMN].equals(test_ids)
    assert submission.columns.tolist() == sample.columns.tolist()
    assert submission["satisfaction"].notna().all()
    assert submission["satisfaction"].between(0, 1).all()

    OUTPUT_DIR.mkdir(exist_ok=True)

    output_path = OUTPUT_DIR / "submission_day17_ensemble.csv"
    submission.to_csv(output_path, index=False)

    print("\n========== DAY 17 SUBMISSION ==========")
    print(f"CatBoost weight: {CATBOOST_WEIGHT:.0%}")
    print(f"XGBoost weight: {XGBOOST_WEIGHT:.0%}")
    print(f"Rows: {len(submission):,}")
    print(f"Prediction minimum: {submission['satisfaction'].min():.6f}")
    print(f"Prediction maximum: {submission['satisfaction'].max():.6f}")
    print(f"Missing predictions: {submission['satisfaction'].isna().sum()}")
    print(f"Output: {output_path}")
    print("\nSubmission validation passed.")


if __name__ == "__main__":
    main()
