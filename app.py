import os
import json
import pickle
import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from peewee import Model, SqliteDatabase, CharField, FloatField

# =========================
# DATABASE
# =========================
DB = SqliteDatabase("predictions.db")

class Prediction(Model):
    unit_id = CharField()
    received_dttm = CharField()
    predicted_response_time_seconds = FloatField()
    actual_response_time_seconds = FloatField(null=True)

    class Meta:
        database = DB

DB.connect()
DB.create_tables([Prediction], safe=True)

# =========================
# LOAD MODEL AND SCHEMA
# =========================
COLUMNS_PATH = "columns.json"
DTYPES_PATH = "dtypes.pickle"
MODEL_PATH = "model.pickle"

with open(COLUMNS_PATH) as f:
    COLUMNS = json.load(f)

with open(DTYPES_PATH, "rb") as f:
    DTYPES = pickle.load(f)

model = joblib.load(MODEL_PATH)

# =========================
# REQUIRED FIELDS
# =========================
PREDICT_REQUIRED = [
    "call_type","call_type_group","original_priority",
    "unit_id","unit_type","station_area","battalion",
    "neighborhood_district","zipcode_of_incident","received_dttm"
]

ACTUAL_REQUIRED = ["unit_id","received_dttm","on_scene_dttm"]

# =========================
# HELPERS
# =========================
def validate_fields(data, required_fields):
    return isinstance(data, dict) and all(f in data for f in required_fields)

def make_features(payload: dict):
    # only keep model features (exclude received_dttm)
    X_dict = {col: payload.get(col, np.nan) for col in COLUMNS}
    df = pd.DataFrame([X_dict])

    # fill missing columns and enforce dtypes
    for col in COLUMNS:
        if col not in df.columns:
            if col in DTYPES and np.issubdtype(DTYPES[col], np.number):
                df[col] = 0
            else:
                df[col] = "missing"
        if col in DTYPES:
            if np.issubdtype(DTYPES[col], np.integer):
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).replace([np.inf, -np.inf], 0).astype(int)
            elif np.issubdtype(DTYPES[col], np.floating):
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).replace([np.inf, -np.inf], 0).astype(float)
            else:
                df[col] = df[col].astype(str).fillna("missing")

    return df[COLUMNS]

# =========================
# FLASK APP
# =========================
app = Flask(__name__)

@app.route("/predict_response/", methods=["POST"])
def predict_response():
    payload = request.get_json()
    if not validate_fields(payload, PREDICT_REQUIRED):
        return jsonify({"error": "invalid input"}), 422

    try:
        X = make_features(payload)
        pred = float(model.predict(X)[0])

        # store prediction (unique by unit_id + received_dttm)
        Prediction.get_or_create(
            unit_id=payload["unit_id"],
            received_dttm=payload["received_dttm"],
            defaults={"predicted_response_time_seconds": pred}
        )

        return jsonify({
            "unit_id": payload["unit_id"],
            "received_dttm": payload["received_dttm"],
            "predicted_response_time_seconds": pred
        })

    except Exception as e:
        return jsonify({"error": "prediction failed", "detail": str(e)}), 422

@app.route("/actual_response/", methods=["POST"])
def actual_response():
    payload = request.get_json()
    if not validate_fields(payload, ACTUAL_REQUIRED):
        return jsonify({"error": "invalid input"}), 422

    record = Prediction.get_or_none(
        (Prediction.unit_id == payload["unit_id"]) &
        (Prediction.received_dttm == payload["received_dttm"])
    )
    if record is None:
        return jsonify({"error": "record not found"}), 422

    try:
        actual_seconds = (
            pd.to_datetime(payload["on_scene_dttm"]) -
            pd.to_datetime(payload["received_dttm"])
        ).total_seconds()
        record.actual_response_time_seconds = actual_seconds
        record.save()

        return jsonify({
            "unit_id": record.unit_id,
            "received_dttm": record.received_dttm,
            "on_scene_dttm": payload["on_scene_dttm"],
            "predicted_response_time_seconds": record.predicted_response_time_seconds,
            "actual_response_time_seconds": actual_seconds
        })
    except Exception as e:
        return jsonify({"error": "update failed", "detail": str(e)}), 422

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)