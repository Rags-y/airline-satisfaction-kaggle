import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

TARGET = "satisfaction"
ID_COLUMN = "id"
RANDOM_STATE = 42

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

print("Loading training data...")
df = pd.read_csv(DATA_DIR / "train.csv")

print(f"Dataset shape: {df.shape}")

y = df[TARGET].astype(int)
X = df.drop(columns=[TARGET, ID_COLUMN], errors="ignore")

categorical_features = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

numerical_features = [
    column for column in X.columns
    if column not in categorical_features
]

print(f"Numerical features: {len(numerical_features)}")
print(f"Categorical features: {len(categorical_features)}")

X_train, X_valid, y_train, y_valid = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y,
)

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numerical_pipeline, numerical_features),
        ("categorical", categorical_pipeline, categorical_features),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=RANDOM_STATE,
            ),
        ),
    ]
)

print(f"Training rows: {len(X_train):,}")
print(f"Validation rows: {len(X_valid):,}")
print("Training Logistic Regression baseline...")

model.fit(X_train, y_train)

print("Generating validation probabilities...")
validation_probabilities = model.predict_proba(X_valid)[:, 1]

roc_auc = roc_auc_score(y_valid, validation_probabilities)

metrics = {
    "model": "LogisticRegression",
    "validation_roc_auc": float(roc_auc),
    "random_state": RANDOM_STATE,
    "validation_size": 0.20,
    "training_rows": int(len(X_train)),
    "validation_rows": int(len(X_valid)),
    "feature_count_before_encoding": int(X.shape[1]),
    "numerical_feature_count": len(numerical_features),
    "categorical_feature_count": len(categorical_features),
    "target": TARGET,
    "scikit_learn_version": sklearn.__version__,
}

report_path = REPORT_DIR / "day2_baseline_metrics.json"
with report_path.open("w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

model_path = MODEL_DIR / "baseline_logistic_regression.joblib"
joblib.dump(model, model_path)

print("\n" + "=" * 45)
print("DAY 2 BASELINE RESULTS")
print("=" * 45)
print(f"Validation ROC AUC: {roc_auc:.6f}")
print(f"Metrics saved to: {report_path}")
print(f"Model saved to: {model_path}")
print("Baseline training completed successfully.")
