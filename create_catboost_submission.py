from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

DATA_DIR = Path("data")
MODEL_PATH = Path("models/day4_catboost.cbm")
OUTPUT_DIR = Path("outputs")
OUTPUT_PATH = OUTPUT_DIR / "submission_day4_catboost.csv"

TARGET = "satisfaction"
ID_COLUMN = "id"

CATEGORICAL_COLUMNS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]


def main():
    print("Loading test data and CatBoost model...")

    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if list(sample.columns) != [ID_COLUMN, TARGET]:
        raise ValueError(f"Unexpected sample submission columns: {sample.columns.tolist()}")

    if len(test) != len(sample):
        raise ValueError("Test and sample submission row counts do not match.")

    if not test[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("Test IDs do not match sample submission IDs.")

    X_test = test.drop(columns=[ID_COLUMN]).copy()

    categorical_columns = [
        col for col in CATEGORICAL_COLUMNS if col in X_test.columns
    ]

    for col in categorical_columns:
        X_test[col] = X_test[col].fillna("Missing").astype(str)

    numeric_columns = [
        col for col in X_test.columns if col not in categorical_columns
    ]

    for col in numeric_columns:
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))

    probabilities = model.predict_proba(X_test)[:, 1]

    if not np.isfinite(probabilities).all():
        raise ValueError("Predictions contain non-finite values.")

    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Predictions must be between 0 and 1.")

    submission = pd.DataFrame({
        ID_COLUMN: test[ID_COLUMN],
        TARGET: probabilities,
    })

    if submission.isnull().any().any():
        raise ValueError("Submission contains missing values.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)

    # Reload and validate the saved file.
    saved = pd.read_csv(OUTPUT_PATH)

    assert list(saved.columns) == [ID_COLUMN, TARGET]
    assert len(saved) == len(sample)
    assert saved[ID_COLUMN].equals(sample[ID_COLUMN])
    assert saved[TARGET].between(0, 1).all()

    print("\nSubmission created and validated successfully.")
    print(f"File: {OUTPUT_PATH}")
    print(f"Rows: {len(saved):,}")
    print(f"Columns: {saved.columns.tolist()}")
    print(f"Minimum probability: {saved[TARGET].min():.6f}")
    print(f"Maximum probability: {saved[TARGET].max():.6f}")
    print(f"Mean probability: {saved[TARGET].mean():.6f}")
    print("\nFirst five predictions:")
    print(saved.head().to_string(index=False))


if __name__ == "__main__":
    main()
