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
    "neighborhood_district",
]

NUMERIC_COLUMNS = [
    "hour",
    "month",
    "day_of_week",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend",
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
            df[col] = df[col].fillna("missing").astype(str)

    # numeric safety
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# =========================================================
# CLEAN
# =========================================================

def clean(df):
    df = df.copy()

    # datetime parsing
    df["received_dttm"] = pd.to_datetime(df["received_dttm"], errors="coerce")
    df["on_scene_dttm"] = pd.to_datetime(df["on_scene_dttm"], errors="coerce")

    df = df.rename(columns={"neighborhoods_analysis_boundaries": "neighborhood_district"})

    # -----------------------------------------------------
    # valid_ battalions
    # -----------------------------------------------------
    valid_battalions = [f'B{i:02d}' for i in range(1, 11)]

    df['battalion'] = df['battalion'].where(
    df['battalion'].isin(valid_battalions), 'unknown')

    # -----------------------------------------------------
    # Drop missing values
    # -----------------------------------------------------
    df = df.dropna(subset=["incident_number", "received_dttm", "on_scene_dttm"])

    # -----------------------------------------------------
    # FIRST UNIT ON SCENE 
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
    df["day_of_week"] = dt.dt.weekday

    df["day_of_week_name"] = dt.dt.strftime('%A')

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    return df


# =========================================================
# CONGESTION FEATURES (FIXED)
# =========================================================

def add_features(df):
    df = df.copy()
    df = df.sort_values("received_dttm")

    # =====================================================
    # GLOBAL STATION SPEED BIAS 
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
    df = add_features(df)

    df = enforce_schema(df)

    return df

