from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_PATH = ROOT / "models" / "day7_catboost_seed123.cbm"
OUTPUT_PATH = ROOT / "outputs" / "submission_day7_seed123.csv"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"

TARGET = "satisfaction"
ID_COL = "id"

def main():
    for path in [TRAIN_PATH, TEST_PATH, SAMPLE_PATH, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    sample = pd.read_csv(SAMPLE_PATH)

    if TARGET not in train.columns:
        raise ValueError(f"Target column '{TARGET}' not found in train.csv")

    if ID_COL not in test.columns or ID_COL not in sample.columns:
        raise ValueError(f"'{ID_COL}' column must exist in test.csv and sample_submission.csv")

    categorical_cols = train.drop(columns=[TARGET]).select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # Match the original Day 4/Day 7 preprocessing.
    for col in categorical_cols:
        if col in test.columns:
            train[col] = train[col].fillna("Missing").astype(str)
            test[col] = test[col].fillna("Missing").astype(str)

    feature_cols = [col for col in train.columns if col not in [TARGET, ID_COL]]

    missing_cols = [col for col in feature_cols if col not in test.columns]
    if missing_cols:
        raise ValueError(f"Test data is missing feature columns: {missing_cols}")

    X_test = test[feature_cols].copy()

    for col in X_test.columns:
        if col not in categorical_cols:
            X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    probabilities = model.predict_proba(X_test)[:, 1]

    if len(probabilities) != len(sample):
        raise ValueError(
            f"Prediction count ({len(probabilities)}) does not match "
            f"sample submission ({len(sample)})"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions contain NaN or infinite values")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Predictions must be between 0 and 1")

    if not np.array_equal(test[ID_COL].to_numpy(), sample[ID_COL].to_numpy()):
        raise ValueError("Test IDs do not match sample submission IDs in order")

    submission = pd.DataFrame({
        ID_COL: sample[ID_COL],
        TARGET: probabilities,
    })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)

    print(f"Model: {MODEL_PATH.name}")
    print(f"Rows: {len(submission):,}")
    print(f"Columns: {submission.columns.tolist()}")
    print(f"Probability range: {probabilities.min():.6f} to {probabilities.max():.6f}")
    print(f"Missing predictions: {submission[TARGET].isna().sum()}")
    print(f"Saved submission: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
