import pickle
import joblib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
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

df = df.sort_values("received_dttm").reset_index(drop=True)

# =========================================================
# SAME FEATURE SET AS LINEAR REGRESSION
# =========================================================

FEATURES = [
    "original_priority",
    "unit_type",
    "call_type_group",
    "call_type",
    "station_speed_index",
    "zipcode_of_incident",
    "battalion",
    "station_area",
    "neighborhoods_analysis_boundaries",
    "hour",
    "hour_cos"
]

TARGET = "response_time"


# =========================================================
# SAFE DATA SELECTION
# =========================================================

X = df.reindex(columns=FEATURES).copy()
y = df[TARGET]

# =========================================================
# TYPE SAFETY
# =========================================================

cat_features = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
num_features = X.select_dtypes(include=[np.number]).columns.tolist()

for c in cat_features:
    X[c] = X[c].astype(str).fillna("missing")

for c in num_features:
    X[c] = pd.to_numeric(X[c], errors="coerce")
    X[c] = X[c].fillna(X[c].median())


# =========================================================
# TIME SPLIT (IDENTICAL LOGIC)
# =========================================================

split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]


# =========================================================
# MODEL
# =========================================================

model = CatBoostRegressor(

    iterations=1200,
    depth=8,
    learning_rate=0.05,

    loss_function="RMSE",
    eval_metric="MAE",

    l2_leaf_reg=5,
    random_strength=1,
    bagging_temperature=0.5,

    grow_policy="SymmetricTree",

    od_type="Iter",
    od_wait=100,

    random_seed=42,
    verbose=200
)


# =========================================================
# TRAIN
# =========================================================

model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    use_best_model=True
)


# =========================================================
# PREDICT
# =========================================================
pred = model.predict(X_test)


# =========================================================
# METRICS (SAME AS LR)
# =========================================================

errors = y_test - pred
abs_errors = np.abs(errors)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)
median_ae = median_absolute_error(y_test, pred)
evs = explained_variance_score(y_test, pred)

p90 = np.percentile(abs_errors, 90)
p95 = np.percentile(abs_errors, 95)


# =========================================================
# Save to use in API
# =========================================================
# Save the model
joblib.dump(model, 'model.pickle')

# Save the column names
with open('columns.json', 'w') as fh:
    json.dump(X_train.columns.tolist(), fh)

# Save the dtypes
with open('dtypes.pickle', 'wb') as fh:
    pickle.dump(X_train.dtypes.to_dict(), fh)  # convert Series to dict for stability

# =========================================================
# RESULTS
# =========================================================
print("\n================ CATBOOST RESULTS ================")

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)
print("Median AE:", median_ae)
print("Explained Variance:", evs)

print("\nTAIL ERRORS")
print("P90:", p90)
print("P95:", p95)

'''

# =========================================================
# FEATURE IMPORTANCE
# =========================================================
importances = model.get_feature_importance()

fi = pd.DataFrame({
    "feature": X_train.columns,
    "importance": importances
})

fi = fi.sort_values("importance", ascending=False)


print("\n================ FEATURE IMPORTANCE ================")
print(fi)


# =========================================================
# TOP 20 PLOT
# =========================================================

top = fi.head(20)

plt.figure(figsize=(10, 6))

plt.barh(
    top["feature"][::-1],
    top["importance"][::-1]
)

plt.title("CatBoost Feature Importance")
plt.xlabel("Importance")

plt.tight_layout()
plt.show()

'''
