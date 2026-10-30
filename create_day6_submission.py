
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

DATA_DIR = Path("data")
MODEL_PATH = Path("models/day6_catboost_feature_engineering.cbm")
OUTPUT_DIR = Path("outputs")
OUTPUT_PATH = OUTPUT_DIR / "submission_day6_feature_engineering.csv"

ID_COLUMN = "id"
TARGET = "satisfaction"

CATEGORICAL_COLUMNS = [
    "Gender",
    "Customer Type",
    "Type of Travel",
    "Class",
]

SERVICE_RATING_COLUMNS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
]


def add_features(X):
    X = X.copy()

    rating_columns = [
        col for col in SERVICE_RATING_COLUMNS if col in X.columns
    ]

    if rating_columns:
        ratings = X[rating_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        X["service_rating_mean"] = ratings.mean(axis=1)
        X["service_rating_sum"] = ratings.sum(axis=1, min_count=1)
        X["service_rating_count"] = ratings.notna().sum(axis=1)

    if "Age" in X.columns:
        age = pd.to_numeric(X["Age"], errors="coerce")
        X["age_group"] = pd.cut(
            age,
            bins=[0, 18, 30, 45, 60, float("inf")],
            labels=["under_18", "18_29", "30_44", "45_59", "60_plus"],
            right=False,
        ).astype("string").fillna("Missing").astype(str)

    departure_col = "Departure Delay in Minutes"
    arrival_col = "Arrival Delay in Minutes"

    if departure_col in X.columns:
        X[departure_col] = pd.to_numeric(
            X[departure_col], errors="coerce"
        )

    if arrival_col in X.columns:
        X[arrival_col] = pd.to_numeric(
            X[arrival_col], errors="coerce"
        )

    if departure_col in X.columns and arrival_col in X.columns:
        X["total_delay_minutes"] = X[
            [departure_col, arrival_col]
        ].sum(axis=1, min_count=1)

        X["arrival_departure_delay_difference"] = (
            X[arrival_col] - X[departure_col]
        )

        X["arrival_delay_missing"] = X[arrival_col].isna().astype(int)

    return X


def main():
    print("Loading test data and Day 6 model...")

    test = pd.read_csv(DATA_DIR / "test.csv")
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if list(sample.columns) != [ID_COLUMN, TARGET]:
        raise ValueError(
            f"Unexpected sample submission columns: {sample.columns.tolist()}"
        )

    if len(test) != len(sample):
        raise ValueError("Test and sample submission row counts do not match.")

    if not test[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("Test IDs do not match sample submission IDs.")

    X_test = test.drop(columns=[ID_COLUMN]).copy()
    X_test = add_features(X_test)

    categorical_columns = [
        col for col in CATEGORICAL_COLUMNS if col in X_test.columns
    ]
    if "age_group" in X_test.columns:
        categorical_columns.append("age_group")

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

    OUTPUT_DIR.mkdir(exist_ok=True)
    submission.to_csv(OUTPUT_PATH, index=False)

    saved = pd.read_csv(OUTPUT_PATH)

    if list(saved.columns) != [ID_COLUMN, TARGET]:
        raise ValueError("Saved submission has incorrect columns.")
    if len(saved) != len(sample):
        raise ValueError("Saved submission has an incorrect row count.")
    if not saved[ID_COLUMN].equals(sample[ID_COLUMN]):
        raise ValueError("Saved submission IDs do not match.")
    if not saved[TARGET].between(0, 1).all():
        raise ValueError("Saved predictions fall outside [0, 1].")
    if saved.isnull().any().any():
        raise ValueError("Saved submission contains missing values.")

    print("\nSubmission created and validated successfully.")
    print(f"File: {OUTPUT_PATH}")
    print(f"Rows: {len(saved):,}")
    print(f"Minimum probability: {saved[TARGET].min():.6f}")
    print(f"Maximum probability: {saved[TARGET].max():.6f}")
    print(f"Mean probability: {saved[TARGET].mean():.6f}")
    print(saved.head().to_string(index=False))


if __name__ == "__main__":
    main()
