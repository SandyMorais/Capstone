import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error,
    explained_variance_score
)

from features import build_dataset


# =========================================================
# LOAD
# =========================================================

df = build_dataset("data/data.csv")
df = df.sort_values("received_dttm")

print(df.columns)
# =========================================================
# FEATURE SET (SINGLE SOURCE OF TRUTH)
# =========================================================

FEATURES = [
    "call_type",
    "call_type_group",
    "original_priority",
    "unit_type",
    #"unit_id",
    "station_area",
    "battalion",
    "zipcode_of_incident",
    "neighborhoods_analysis_boundaries",
    "hour",
    "month",
    "dow",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
    "load_1h",
    "load_1d",
    "station_queue_pressure",
    "station_ewm_load",
    "station_speed_index"
]

TARGET = "response_time"

# =========================================================
# SAFE SELECTION (NO KEYERROR EVER)
# =========================================================

X = df.reindex(columns=FEATURES).copy()
y = df[TARGET]


# =========================================================
# AUTO DETECT TYPES (NO external lists)
# =========================================================

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()


# =========================================================
# TYPE SAFETY
# =========================================================

for col in categorical_features:
    X[col] = X[col].astype(str).fillna("missing")

for col in numeric_features:
    X[col] = pd.to_numeric(X[col], errors="coerce")


# =========================================================
# SPLIT (time-safe)
# =========================================================

split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]


# =========================================================
# PREPROCESSOR
# =========================================================

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ]
)


# =========================================================
# MODEL
# =========================================================

model = LinearRegression()

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])


# =========================================================
# TRAIN
# =========================================================

pipeline.fit(X_train, y_train)


# =========================================================
# PREDICT
# =========================================================

pred = pipeline.predict(X_test)


# =========================================================
# METRICS
# =========================================================

errors = y_test - pred
abs_errors = np.abs(errors)

print("\n================ METRICS ================")
print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))
print("R2:", r2_score(y_test, pred))
print("Median AE:", median_absolute_error(y_test, pred))
print("Explained Variance:", explained_variance_score(y_test, pred))

print("\nTAIL ERRORS")
print("P90:", np.percentile(abs_errors, 90))
print("P95:", np.percentile(abs_errors, 95))