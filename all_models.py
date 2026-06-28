
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from catboost import CatBoostRegressor

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from features import build_dataset


# =========================================================
# LOAD SATA
# =========================================================
df = build_dataset("data/data.csv")

df = df.sort_values("received_dttm").reset_index(drop=True)

# =========================================================
# FEATURES
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
    "neighborhood_district",
    "hour",
    "hour_cos"
]

TARGET = "response_time"

# =========================================================
# SPLIT (same as before)
# =========================================================
split = int(len(df) * 0.8)

X = df[FEATURES]
y = df["response_time"]

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]


# =========================================================
# IDENTIFY TYPES
# =========================================================
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()


# =========================================================
# PREPROCESSOR (for non-CatBoost models)
# =========================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ("num", "passthrough", num_cols)
    ]
)


# =========================================================
# MODELS
# =========================================================
models = {

    "LinearRegression": Pipeline([
        ("prep", preprocessor),
        ("model", LinearRegression())
    ]),

    "RandomForest": Pipeline([
        ("prep", preprocessor),
        ("model", RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "GradientBoosting": Pipeline([
        ("prep", preprocessor),
        ("model", GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        ))
    ]),

    "HistGradientBoosting": Pipeline([
        ("prep", preprocessor),
        ("model", HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.05,
            max_depth=8,
            random_state=42
        ))
    ]),

    "LightGBM": Pipeline([
        ("prep", preprocessor),
        ("model", LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            reg_alpha=1,
            reg_lambda=1,
            random_state=42
        ))
    ]),

    "XGBoost": Pipeline([
        ("prep", preprocessor),
        ("model", XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1,
            random_state=42,
            n_jobs=-1,
            tree_method="hist"   # faster + recommended
        ))
    ])


}

# =========================================================
# TRAIN + EVALUATE
# =========================================================
results = []

for name, model in models.items():
    print(f"\n==== {name} ====")

    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)

    abs_err = np.abs(y_test - pred)
    p90 = np.percentile(abs_err, 90)
    p95 = np.percentile(abs_err, 95)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")
    print(f"P90: {p90:.2f}")
    print(f"P95: {p95:.2f}")

    results.append([name, mae, rmse, r2, p90, p95])

# =========================================================
# CATBOOST (separately, no encoding)
# =========================================================
cat_features = cat_cols

cat_model = CatBoostRegressor(
    iterations=1200,
    depth=8,
    learning_rate=0.055,
    l2_leaf_reg=5,
    loss_function="RMSE",
    random_strength=2,
    bagging_temperature=0.5,
    random_seed=42,
    verbose=400
)

cat_model.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_test, y_test),
    use_best_model=True
)

pred = cat_model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

abs_err = np.abs(y_test - pred)
p90 = np.percentile(abs_err, 90)
p95 = np.percentile(abs_err, 95)

print("\n==== CatBoost ====")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2: {r2:.4f}")
print(f"P90: {p90:.2f}")
print(f"P95: {p95:.2f}")

#results.append(["CatBoost", mae, rmse, r2, p90, p95])


# =========================================================
# FINAL TABLE
# =========================================================
results_df = pd.DataFrame(
    results,
    columns=["Model", "MAE", "RMSE", "R2", "P90", "P95"]
)

print("\n================ MODEL COMPARISON ================")
print(results_df.sort_values("MAE")) 

# Save to .csv
results_df.to_csv('outputs/model_metrics_results.csv')


'''
results_df = pd.read_csv('outputs/model_results.csv')

models = results_df["Model"]
mae = results_df["MAE"]
rmse = results_df["RMSE"]

x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(x - width/2, mae, width, color='steelblue', label="MAE")
plt.bar(x + width/2, rmse, width, color='darkred', label="RMSE")

plt.xticks(x, models, rotation=45, size=13)
plt.ylabel("Error", size=14, labelpad=10)
plt.yticks(fontsize=12)
plt.ylim(0, 350)

plt.title("Model Comparison: MAE vs RMSE", size=16)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig(
        f'outputs/model_metrics_results.pdf',
        bbox_inches='tight')
plt.show()
'''
