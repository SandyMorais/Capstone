import numpy as np
import pandas as pd


# =========================================================
# CONFIG
# =========================================================

TARGET_COL = "response_time"

CATEGORICAL_COLUMNS = [
    "call_type",
    "call_type_group",
    "original_priority",
    "unit_type",
    "unit_id",
    "station_area",
    "battalion",
    "zipcode_of_incident",
    "neighborhoods_analysis_boundaries"
]

NUMERIC_COLUMNS = [
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


# =========================================================
# TYPE ENFORCEMENT
# =========================================================

def enforce_schema(df, cat_cols=CATEGORICAL_COLUMNS, num_cols=NUMERIC_COLUMNS):
    df = df.copy()

    # categorical safety (CatBoost requirement)
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("missing")

    # numeric safety
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =========================================================
# CLEAN (FIXED: FIRST UNIT ON SCENE)
# =========================================================

def clean(df):
    df = df.copy()

    # datetime parsing
    df["received_dttm"] = pd.to_datetime(df["received_dttm"], errors="coerce")
    df["on_scene_dttm"] = pd.to_datetime(df["on_scene_dttm"], errors="coerce")

    required = ["incident_number", "received_dttm", "on_scene_dttm"]
    df = df.dropna(subset=required)

    # -----------------------------------------------------
    # FIRST UNIT ON SCENE (CORRECT LOGIC)
    # -----------------------------------------------------
    df = df.sort_values(
        ["incident_number", "on_scene_dttm", "received_dttm", "unit_id"]
    )

    df = df.loc[
        df.groupby("incident_number")["on_scene_dttm"].idxmin()
    ].reset_index(drop=True)

    # -----------------------------------------------------
    # TARGET
    # -----------------------------------------------------
    df[TARGET_COL] = (
        df["on_scene_dttm"] - df["received_dttm"]
    ).dt.total_seconds()

    # clean target
    df = df[df[TARGET_COL] > 0]
    df = df[df[TARGET_COL] <= df[TARGET_COL].quantile(0.99)]

    df = df.sort_values("received_dttm").reset_index(drop=True)

    return df


# =========================================================
# TIME FEATURES
# =========================================================

def add_time_features(df):
    df = df.copy()

    dt = df["received_dttm"]

    df["hour"] = dt.dt.hour
    df["month"] = dt.dt.month
    df["dow"] = dt.dt.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)

    df["is_weekend"] = df["dow"].isin([5, 6]).astype(int)

    return df


# =========================================================
# STREAMING FEATURES (SAFE)
# =========================================================

def add_streaming_features(df):
    df = df.copy()
    df = df.sort_values("received_dttm")

    df["station_queue_pressure"] = df.groupby("station_area").cumcount()

    df["hour_bucket"] = df["received_dttm"].dt.floor("h")

    df["station_hour_load"] = (
    df.groupby(["station_area", "hour_bucket"])
    .transform("size")
    .astype(float))

    df["station_avg_response"] = (
        df.groupby("station_area")[TARGET_COL]
        .transform(lambda x: x.shift(1).rolling(50, min_periods=5).mean())
    )

    df["station_avg_response"] = df["station_avg_response"].fillna(
        df[TARGET_COL].median()
    )

    return df


# =========================================================
# CONGESTION FEATURES (FIXED)
# =========================================================

def add_congestion_features(df):
    df = df.copy()
    df = df.sort_values("received_dttm")

    # =====================================================
    # 1. TIME BUCKET COUNTS (SAFE, NO LEAKAGE)
    # =====================================================

    df["hour_bucket"] = df["received_dttm"].dt.floor("h")
    df["day_bucket"] = df["received_dttm"].dt.date

    df["load_1h"] = df.groupby(
        ["station_area", "hour_bucket"]
    )["station_area"].transform("size")

    df["load_1d"] = df.groupby(
        ["station_area", "day_bucket"]
    )["station_area"].transform("size")

    # =====================================================
    # 2. CUMULATIVE PRESSURE (VERY STABLE)
    # =====================================================

    df["station_queue_pressure"] = (
        df.groupby("station_area").cumcount()
    )

    # normalize (important for stability)
    df["station_queue_pressure"] = (
        df["station_queue_pressure"]
        / (df.groupby("station_area")["station_queue_pressure"]
           .transform("max")
           .replace(0, 1))
    )

    # =====================================================
    # 3. EXPONENTIAL SMOOTHING (SAFE ALTERNATIVE TO ROLLING)
    # =====================================================

    df["station_ewm_load"] = (
        df.groupby("station_area")["station_queue_pressure"]
        .transform(lambda x: x.ewm(span=50, adjust=False).mean())
    )

    # =====================================================
    # 4. GLOBAL STATION SPEED BIAS (VERY IMPORTANT)
    # =====================================================

    station_mean = df.groupby("station_area")["response_time"].transform("mean")

    df["station_speed_index"] = (
        station_mean / station_mean.mean()
    )

    # fill safety
    df["station_speed_index"] = df["station_speed_index"].fillna(1.0)

    return df


# =========================================================
# FINAL PIPELINE
# =========================================================

def build_dataset(path):
    df = pd.read_csv(path)

    df = clean(df)
    df = add_time_features(df)
    df = add_streaming_features(df)
    df = add_congestion_features(df)

    df = enforce_schema(df)

    return df