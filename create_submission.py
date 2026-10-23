from pathlib import Path

import joblib
import numpy as np
import pandas as pd

DATA_DIR = Path("data")
MODEL_PATH = Path("models/baseline_logistic_regression.joblib")
OUTPUT_DIR = Path("outputs")
OUTPUT_PATH = OUTPUT_DIR / "submission_day3.csv"

ID_COLUMN = "id"
TARGET = "satisfaction"

print("Loading saved baseline model...")
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. Run baseline_model.py first."
    )

model = joblib.load(MODEL_PATH)

print("Loading test and sample submission files...")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

if ID_COLUMN not in test.columns:
    raise ValueError(f"Test data is missing '{ID_COLUMN}'.")

if list(sample.columns) != [ID_COLUMN, TARGET]:
    raise ValueError(
        f"Unexpected sample submission columns: {list(sample.columns)}"
    )

if len(test) != len(sample):
    raise ValueError(
        f"Row count mismatch: test={len(test)}, sample={len(sample)}"
    )

if not test[ID_COLUMN].reset_index(drop=True).equals(
    sample[ID_COLUMN].reset_index(drop=True)
):
    raise ValueError("Test IDs do not match sample submission IDs.")

print(f"Test rows: {len(test):,}")
print("Generating satisfaction probabilities...")

X_test = test.drop(columns=[ID_COLUMN])
probabilities = model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    ID_COLUMN: test[ID_COLUMN],
    TARGET: probabilities,
})

# Validate the final file before saving.
if list(submission.columns) != list(sample.columns):
    raise ValueError("Submission column order does not match sample.")

if len(submission) != len(sample):
    raise ValueError("Submission row count does not match sample.")

if submission[ID_COLUMN].duplicated().any():
    raise ValueError("Duplicate IDs found in submission.")

if not np.isfinite(probabilities).all():
    raise ValueError("Predictions contain NaN or infinite values.")

if ((probabilities < 0) | (probabilities > 1)).any():
    raise ValueError("Predictions must be probabilities between 0 and 1.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
submission.to_csv(OUTPUT_PATH, index=False)

# Read it back to verify the actual saved CSV.
saved = pd.read_csv(OUTPUT_PATH)

if list(saved.columns) != list(sample.columns):
    raise ValueError("Saved CSV has incorrect columns.")

if len(saved) != len(sample):
    raise ValueError("Saved CSV has incorrect row count.")

if not saved[ID_COLUMN].reset_index(drop=True).equals(
    sample[ID_COLUMN].reset_index(drop=True)
):
    raise ValueError("Saved CSV IDs do not match the sample.")

print("\n" + "=" * 45)
print("DAY 3 SUBMISSION VALIDATION")
print("=" * 45)
print(f"File: {OUTPUT_PATH}")
print(f"Rows: {len(saved):,}")
print(f"Columns: {list(saved.columns)}")
print(f"Minimum probability: {saved[TARGET].min():.6f}")
print(f"Maximum probability: {saved[TARGET].max():.6f}")
print(f"Mean probability: {saved[TARGET].mean():.6f}")
print("\nFirst five rows:")
print(saved.head().to_string(index=False))
print("\nSubmission CSV created and validated successfully.")
