
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
train = pd.read_csv(DATA_DIR / "train.csv")
test = pd.read_csv(DATA_DIR / "test.csv")
sample = pd.read_csv(DATA_DIR / "sample_submission.csv")

print("=" * 60)
print("DATASET SHAPES")
print("=" * 60)
print(f"Train: {train.shape}")
print(f"Test: {test.shape}")
print(f"Sample submission: {sample.shape}")

target = "satisfaction"

print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)
print(train[target].value_counts(dropna=False))
print("\nTarget proportions:")
print(train[target].value_counts(normalize=True, dropna=False))

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
print("Training:")
print(train.isnull().sum()[train.isnull().sum() > 0])
print("\nTest:")
print(test.isnull().sum()[test.isnull().sum() > 0])

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)
print(train.dtypes.to_string())

print("\n" + "=" * 60)
print("DUPLICATE CHECKS")
print("=" * 60)
print("Duplicate training rows:", train.duplicated().sum())
print("Duplicate test rows:", test.duplicated().sum())
print("Duplicate training IDs:", train["id"].duplicated().sum())
print("Duplicate test IDs:", test["id"].duplicated().sum())

print("\n" + "=" * 60)
print("NUMERICAL FEATURE SUMMARY")
print("=" * 60)
print(train.describe().round(2).to_string())

print("\n" + "=" * 60)
print("CATEGORICAL FEATURE VALUES")
print("=" * 60)

categorical = train.select_dtypes(
    include=["object", "category", "bool"]
).columns

for column in categorical:
    print(f"\n{column}:")
    print(train[column].value_counts(dropna=False).to_string())

print("\n" + "=" * 60)
print("ID AND SUBMISSION FORMAT")
print("=" * 60)
print("Training ID range:", train["id"].min(), "to", train["id"].max())
print("Test ID range:", test["id"].min(), "to", test["id"].max())
print("Sample submission columns:", sample.columns.tolist())
print("Sample submission first rows:")
print(sample.head().to_string(index=False))

assert target in train.columns
assert target not in test.columns
assert train["id"].is_unique
assert test["id"].is_unique
assert sample.columns.tolist() == ["id", target]
assert len(test) == len(sample)

print("\nDay 1 exploration completed successfully.")
