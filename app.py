import os
import re
import json
import pickle
import joblib
import numpy as np
import pandas as pd

from flask import Flask, request, jsonify
from peewee import (
    Model,
    CharField,
    FloatField,
    CompositeKey,
    IntegrityError,
)
from playhouse.db_url import connect
from playhouse.sqlite_ext import JSONField

# =========================================================
# DB
# =========================================================

DB = connect(
    os.environ.get("DATABASE_URL") or "sqlite:///predictions.db"
)

class Prediction(Model):
    unit_id = CharField()
    received_dttm = CharField()

    predicted_response_time_seconds = FloatField()
    actual_response_time_seconds = FloatField(null=True)

    predict_payload = JSONField(null=True)
    actual_payload = JSONField(null=True)

    class Meta:
        database = DB
        primary_key = CompositeKey("unit_id", "received_dttm")


DB.create_tables([Prediction], safe=True)

# =========================================================
# LOAD MODEL
# =========================================================

with open("columns.json") as f:
    COLUMNS = json.load(f)

with open("dtypes.pickle", "rb") as f:
    DTYPES = pickle.load(f)

model = joblib.load("model.pickle")

# =========================================================
# VALID VALUES
# =========================================================

UNIT_TYPE_MAP = {
    "MEDIC": "MEDIC",
    "EMS": "MEDIC",
    "AMBULANCE": "MEDIC",
    "AMB": "MEDIC",
    "PARAMEDIC": "MEDIC",
    "ENGINE": "ENGINE",
    "ENG": "ENGINE",
    "ENGINE COMPANY": "ENGINE",
    "TRUCK": "TRUCK",
    "TRUCK COMPANY": "TRUCK",
    "CHIEF": "CHIEF",
    "PRIVATE": "PRIVATE",
    "SUPPORT": "SUPPORT",
    "RESCUE CAPTAIN": "RESCUE CAPTAIN",
    "RESCUE SQUAD": "RESCUE SQUAD",
    "INVESTIGATION": "INVESTIGATION",
    "CP": "CP",
    "BLS": "BLS"
}

allowed_unit_type = set(UNIT_TYPE_MAP.values())

allowed_battalions = {
    "B01","B02","B03","B04","B05",
    "B06","B07","B08","B09"
}

# =========================================================
# FIX: CALL TYPE GROUP MAP
# =========================================================

CALL_TYPE_GROUP_MAP = {
    "POTENTIALLY LIFE THREATENING": "Potentially Life-Threatening",
    "NON LIFE THREATENING": "Non Life-threatening",
    "ALARM": "Alarm",
    "FIRE": "Fire"
}

allowed_call_type_group = {
    "Potentially Life-Threatening",
    "Non Life-threatening",
    "Alarm",
    "Fire"
}

allowed_priority = ['1','2','3','E','I','A','B','T']

SF_zipcodes = {
    "94102","94103","94104","94105","94107","94108",
    "94109","94110","94111","94112","94114","94115",
    "94116","94117","94118","94121","94122","94123",
    "94124","94127","94129","94130","94131","94132",
    "94133","94134","94158"
}

# =========================================================
# ERROR HANDLER
# =========================================================

def validation_error(field, value, allowed=None, message=None):
    if message:
        return False, message

    if allowed is not None:
        allowed_str = ", ".join(sorted(map(str, allowed)))
        return False, (
            f"{field} '{value}' is invalid. "
            f"Allowed values: {allowed_str}"
        )

    return False, f"{field} '{value}' is invalid"

# =========================================================
# NORMALIZATION
# =========================================================

def normalize_text(value: str):
    if value is None:
        return None
    v = str(value).upper().strip()
    v = re.sub(r"[^A-Z0-9]", " ", v)
    v = re.sub(r"\s+", " ", v)
    return v.strip()


def normalize_battalion(value: str):
    if value is None:
        return None

    v = str(value).upper().strip()
    v = re.sub(r"[^A-Z0-9]", "", v)

    match = re.search(r"(\d+)", v)
    if not match:
        return None

    return f"B{int(match.group(1)):02d}"


def normalize_unit_type(value: str):
    v = normalize_text(value)

    if v in UNIT_TYPE_MAP:
        return UNIT_TYPE_MAP[v]

    if v and "AMBUL" in v:
        return "MEDIC"
    if v and "ENGINE" in v:
        return "ENGINE"
    if v and "TRUCK" in v:
        return "TRUCK"

    return None

# =========================================================
# VALIDATION
# =========================================================

def validate_predict(data):

    if not isinstance(data, dict):
        return False, "payload must be valid json"

    for f in [
        "call_type","call_type_group","original_priority",
        "unit_id","unit_type","station_area","battalion",
        "neighborhood_district","zipcode_of_incident",
        "received_dttm"
    ]:
        if f not in data:
            return False, f"missing field: {f}"

    # -------------------------
    # call_type (required check only)
    # -------------------------
    if not isinstance(data.get("call_type"), str) or not data["call_type"].strip():
        return validation_error("call_type", data.get("call_type"))

    # datetime
    try:
        pd.to_datetime(data["received_dttm"])
    except:
        return False, "received_dttm must be valid ISO datetime"

    # zipcode
    zipcode = str(data["zipcode_of_incident"])
    if zipcode not in SF_zipcodes:
        return validation_error("zipcode_of_incident", zipcode, SF_zipcodes)

    # priority
    priority = str(data["original_priority"])
    if priority not in allowed_priority:
        return validation_error("original_priority", priority, allowed_priority)

    # =====================================================
    # FIXED call_type_group (normalization + mapping)
    # =====================================================

    raw_ctg = data["call_type_group"]
    normalized_ctg = normalize_text(raw_ctg)

    if not normalized_ctg:
        return validation_error("call_type_group", raw_ctg, allowed_call_type_group)

    mapped_ctg = CALL_TYPE_GROUP_MAP.get(normalized_ctg)

    if mapped_ctg is None:
        return validation_error("call_type_group", raw_ctg, allowed_call_type_group)

    data["call_type_group"] = mapped_ctg

    # unit_type
    unit_type = normalize_unit_type(data["unit_type"])
    if unit_type not in allowed_unit_type:
        return validation_error("unit_type", data["unit_type"], allowed_unit_type)

    data["unit_type"] = unit_type

    # battalion
    battalion = normalize_battalion(data["battalion"])
    if battalion not in allowed_battalions:
        return validation_error("battalion", data["battalion"], allowed_battalions)

    data["battalion"] = battalion

    return True, None


def validate_actual(data):

    for f in ["unit_id","received_dttm","on_scene_dttm"]:
        if f not in data:
            return False, f"missing field: {f}"

    try:
        r = pd.to_datetime(data["received_dttm"])
        o = pd.to_datetime(data["on_scene_dttm"])
    except:
        return False, "invalid datetime format"

    if o < r:
        return False, "on_scene_dttm cannot be earlier than received_dttm"

    return True, None

# =========================================================
# FEATURE BUILDER
# =========================================================

def make_features(payload):

    row = {}

    for col in COLUMNS:
        if col == "neighborhoods_analysis_boundaries":
            row[col] = payload.get("neighborhood_district", "missing")
        else:
            row[col] = payload.get(col, np.nan)

    X = pd.DataFrame([row])

    for col in COLUMNS:
        dtype = DTYPES[col]

        if np.issubdtype(dtype, np.number):
            X[col] = pd.to_numeric(X[col], errors="coerce")
            X[col] = X[col].replace([np.inf,-np.inf], np.nan).fillna(0)
        else:
            X[col] = (
                X[col].astype(str)
                .replace("nan","missing")
                .fillna("missing")
            )

    return X[COLUMNS]

# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

@app.route("/predict_response/", methods=["POST"])
def predict_response():

    payload = request.get_json()
    valid, error = validate_predict(payload)

    if not valid:
        return jsonify({"error": error}), 422

    try:
        X = make_features(payload)
        pred = float(model.predict(X)[0])

        Prediction.create(
            unit_id=payload["unit_id"],
            received_dttm=payload["received_dttm"],
            predicted_response_time_seconds=pred,
            predict_payload=payload
        )

        return jsonify({
            "unit_id": payload["unit_id"],
            "received_dttm": payload["received_dttm"],
            "predicted_response_time_seconds": pred
        })

    except IntegrityError:
        return jsonify({
            "error": "prediction already exists for this unit_id and received_dttm pair"
        }), 422

    except Exception as e:
        return jsonify({"error": str(e)}), 422


@app.route("/actual_response/", methods=["POST"])
def actual_response():

    payload = request.get_json()
    valid, error = validate_actual(payload)

    if not valid:
        return jsonify({"error": error}), 422

    record = Prediction.get_or_none(
        (Prediction.unit_id == payload["unit_id"]) &
        (Prediction.received_dttm == payload["received_dttm"])
    )

    if record is None:
        return jsonify({"error": "record not found"}), 422

    actual_seconds = (
        pd.to_datetime(payload["on_scene_dttm"]) -
        pd.to_datetime(payload["received_dttm"])
    ).total_seconds()

    record.actual_response_time_seconds = actual_seconds
    record.actual_payload = payload
    record.save()

    return jsonify({
        "unit_id": record.unit_id,
        "received_dttm": record.received_dttm,
        "actual_response_time_seconds": actual_seconds,
        "predicted_response_time_seconds": record.predicted_response_time_seconds
    })


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)