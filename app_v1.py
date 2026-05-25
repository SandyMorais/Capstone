import os
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from playhouse.db_url import connect

from flask import Flask, request, jsonify

from peewee import (
    Model,
    SqliteDatabase,
    CharField,
    FloatField,
    CompositeKey,
    IntegrityError
)

# =========================================================
# DATABASE
# =========================================================

DB = connect(os.environ.get('DATABASE_URL') or 'sqlite:///predictions.db')

class Prediction(Model):

    unit_id = CharField()
    received_dttm = CharField()

    predicted_response_time_seconds = FloatField()

    actual_response_time_seconds = FloatField(
        null=True
    )

    class Meta:

        database = DB

        # unique(unit_id, received_dttm)
        primary_key = CompositeKey(
            "unit_id",
            "received_dttm"
        )


DB.create_tables(
    [Prediction],
    safe=True
)

# =========================================================
# LOAD MODEL + SCHEMA
# =========================================================

with open("columns.json") as f:
    COLUMNS = json.load(f)

with open("dtypes.pickle", "rb") as f:
    DTYPES = pickle.load(f)

model = joblib.load("model.pickle")

# =========================================================
# VALID VALUES
# =========================================================
allowed_unit_type = {'MEDIC', 'ENGINE', 'PRIVATE', 'CHIEF', 'TRUCK', 'RESCUE CAPTAIN', 'SUPPORT', 
'RESCUE SQUAD', 'INVESTIGATION', 'CP', 'BLS'}

allowed_battalions = { "B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09"}

allowed_call_type_group = ["potentially life-threatening", "non life-threatening", "alarm", "fire" ]

allowed_priority = ['1','2','3','E','I','A','B','T']

SF_zipcodes = { "94102", "94103", "94104", "94105", "94107", "94108", "94109", "94110", "94111", 
    "94112", "94114", "94115", "94116", "94117", "94118", "94121", "94122", "94123", "94124", "94127", 
    "94129", "94130", "94131", "94132", "94133", "94134", "94158" }

# =========================================================
# REQUIRED FIELDS
# =========================================================

PREDICT_REQUIRED = [
    "call_type",
    "call_type_group",
    "original_priority",
    "unit_id",
    "unit_type",
    "station_area",
    "battalion",
    "neighborhood_district",
    "zipcode_of_incident",
    "received_dttm"
]

ACTUAL_REQUIRED = [
    "unit_id",
    "received_dttm",
    "on_scene_dttm"
]

# =========================================================
# VALIDATION
# =========================================================

def validate_predict(data):

    if not isinstance(data, dict):
        return False, "payload must be valid json"

    for field in PREDICT_REQUIRED:

        if field not in data:
            return False, f"missing field: {field}"

    # validate datetime
    try:

        pd.to_datetime(data["received_dttm"])

    except:

        return False, (
            "received_dttm must be valid ISO 8601 datetime "
            "(example: 2026-05-22T14:30:00)"
        )

    # validate zipcode
    zipcode = str(data["zipcode_of_incident"])

    if zipcode not in SF_zipcodes:

        return False, (
            f"zipcode_of_incident '{zipcode}' is invalid. "
            f"Allowed San Francisco zipcodes: "
            f"{sorted(SF_zipcodes)}"
        )

    # validate priority
    priority = str(data["original_priority"])

    if priority not in allowed_priority:

        return False, (
            f"original_priority '{priority}' is invalid. "
            f"Allowed values: {allowed_priority}"
        )

    # validate call type group
    data["call_type_group"] = (str(data["call_type_group"]).strip().lower())
    group = str(data["call_type_group"])

    if group not in allowed_call_type_group:

        return False, (
            f"call_type_group '{group}' is invalid. "
            f"Allowed values: {allowed_call_type_group}"
        )

    # normalize unit_type
    data["unit_type"] = (str(data["unit_type"]).strip().upper())
    unit_type = data["unit_type"]

    if unit_type not in allowed_unit_type:

        return False, (
            f"unit_type '{unit_type}' is invalid. "
            f"Allowed values: {sorted(allowed_unit_type)}"
        )

    # normalize battalion
    battalion = str(data["battalion"]).strip().upper()
    # convert B1 -> B01
    if battalion.startswith("B"):

        num = battalion[1:]

        if num.isdigit():

            battalion = f"B{int(num):02d}"

    data["battalion"] = battalion
    if battalion not in allowed_battalions:
        return False, (
            f"battalion '{battalion}' is invalid. "
            f"Allowed values: {sorted(allowed_battalions)}"
        )

    return True, None


def validate_actual(data):

    if not isinstance(data, dict):
        return False, "payload must be valid json"

    for field in ACTUAL_REQUIRED:

        if field not in data:
            return False, f"missing field: {field}"

    try:

        received = pd.to_datetime(
            data["received_dttm"]
        )

    except:

        return False, (
            "received_dttm must be valid ISO 8601 datetime "
            "(example: 2026-05-22T14:30:00)"
        )

    try:

        on_scene = pd.to_datetime(
            data["on_scene_dttm"]
        )

    except:

        return False, (
            "on_scene_dttm must be valid ISO 8601 datetime "
            "(example: 2026-05-22T14:45:00)"
        )

    if on_scene < received:

        return False, (
            "on_scene_dttm cannot be earlier "
            "than received_dttm"
        )

    return True, None

# =========================================================
# FEATURE BUILDER
# =========================================================

def make_features(payload):

    row = {}

    for col in COLUMNS:

        if col == "neighborhoods_analysis_boundaries":

            row[col] = payload.get(
                "neighborhood_district",
                "missing"
            )

        else:

            row[col] = payload.get(
                col,
                np.nan
            )

    X = pd.DataFrame([row])

    # enforce dtypes
    for col in COLUMNS:

        dtype = DTYPES[col]

        if np.issubdtype(dtype, np.number):

            X[col] = pd.to_numeric(
                X[col],
                errors="coerce"
            )

            X[col] = (
                X[col]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0)
            )

        else:

            X[col] = (
                X[col]
                .astype(str)
                .replace("nan", "missing")
                .fillna("missing")
            )

    return X[COLUMNS]

# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

# =========================================================
# PREDICT RESPONSE
# =========================================================

@app.route(
    "/predict_response/",
    methods=["POST"]
)
def predict_response():

    payload = request.get_json()

    valid, error = validate_predict(
        payload
    )

    if not valid:

        return jsonify({
            "error": error
        }), 422

    try:

        X = make_features(payload)

        pred = float(
            model.predict(X)[0]
        )

        Prediction.create(
            unit_id=payload["unit_id"],
            received_dttm=payload["received_dttm"],
            predicted_response_time_seconds=pred
        )

        return jsonify({
            "unit_id": payload["unit_id"],
            "received_dttm": payload["received_dttm"],
            "predicted_response_time_seconds": pred
        })

    except IntegrityError:

        return jsonify({
            "error": (
                "prediction already exists for "
                "this unit_id and received_dttm pair"
            )
        }), 422

    except Exception as e:

        return jsonify({
            "error": "prediction failed",
            "detail": str(e)
        }), 422

# =========================================================
# ACTUAL RESPONSE
# =========================================================

@app.route(
    "/actual_response/",
    methods=["POST"]
)
def actual_response():

    payload = request.get_json()

    valid, error = validate_actual(
        payload
    )

    if not valid:

        return jsonify({
            "error": error
        }), 422

    record = Prediction.get_or_none(
        (Prediction.unit_id == payload["unit_id"]) &
        (
            Prediction.received_dttm ==
            payload["received_dttm"]
        )
    )

    if record is None:

        return jsonify({
            "error": (
                "record not found for "
                "provided unit_id and received_dttm"
            )
        }), 422

    try:

        actual_seconds = (
            pd.to_datetime(
                payload["on_scene_dttm"]
            )
            -
            pd.to_datetime(
                payload["received_dttm"]
            )
        ).total_seconds()

        record.actual_response_time_seconds = (
            actual_seconds
        )

        record.save()

        return jsonify({

            "unit_id": record.unit_id,

            "received_dttm":
                record.received_dttm,

            "on_scene_dttm":
                payload["on_scene_dttm"],

            "actual_response_time_seconds":
                actual_seconds,

            "predicted_response_time_seconds":
                record.predicted_response_time_seconds
        })

    except Exception as e:

        return jsonify({
            "error": "update failed",
            "detail": str(e)
        }), 422

# =========================================================
# HEALTH CHECK
# =========================================================

@app.route("/", methods=["GET"])
def health():

    return jsonify({
        "status": "ok"
    })

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
