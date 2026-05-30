import os
import re
import json
import pickle
import joblib
import logging
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
# LOGGING (IMPORTANT)
# =========================================================

logging.basicConfig(level=logging.INFO)

# =========================================================
# DATABASE
# =========================================================

DB = connect(
    os.environ.get("DATABASE_URL")
    or "sqlite:///predictions.db"
)

# =========================================================
# MODEL
# =========================================================

class Prediction(Model):

    unit_id = CharField()
    received_dttm = CharField()

    predicted_response_time_seconds = FloatField()
    actual_response_time_seconds = FloatField(null=True)

    call_type = CharField(null=True)
    call_type_group = CharField(null=True)
    original_priority = CharField(null=True)
    unit_type = CharField(null=True)
    station_area = CharField(null=True)
    battalion = CharField(null=True)
    neighborhood_district = CharField(null=True)
    zipcode_of_incident = CharField(null=True)
    on_scene_dttm = CharField(null=True)

    raw_predict_payload = JSONField(null=True)
    normalized_predict_payload = JSONField(null=True)
    raw_actual_payload = JSONField(null=True)
    normalized_actual_payload = JSONField(null=True)

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
# NORMALIZER
# =========================================================

class CategoricalNormalizer:

    def normalize_text(self, value):
        if value is None:
            return None
        v = str(value).upper().strip()
        v = re.sub(r"[^A-Z0-9]+", " ", v)
        return re.sub(r"\s+", " ", v).strip()

    def normalize(self, field, value):
        if field == "battalion":
            match = re.search(r"(\d+)", str(value))
            if not match:
                return None
            return f"B{int(match.group(1)):02d}"

        if field == "unit_type":
            v = self.normalize_text(value)
            return v.split()[0] if v else None

        if field == "call_type_group":
            v = self.normalize_text(value)
            return v

        return value

# =========================================================
# VALIDATOR
# =========================================================

class SchemaValidator:

    def validate(self, section, data):

        if not isinstance(data, dict):
            return False, "payload must be valid json"

        required = {
            "predict": [
                "call_type", "call_type_group", "original_priority",
                "unit_id", "unit_type", "station_area",
                "battalion", "neighborhood_district",
                "zipcode_of_incident", "received_dttm"
            ],
            "actual": [
                "unit_id", "received_dttm", "on_scene_dttm"
            ]
        }

        for f in required[section]:
            if f not in data:
                return False, f"missing field: {f}"

        # datetime validation
        try:
            data["received_dttm"] = pd.to_datetime(data["received_dttm"]).isoformat()
            if section == "actual":
                data["on_scene_dttm"] = pd.to_datetime(data["on_scene_dttm"]).isoformat()
        except:
            return False, "invalid datetime format"

        if section == "actual":
            if data["on_scene_dttm"] <= data["received_dttm"]:
                return False, "on_scene_dttm must be later than received_dttm"

        return True, None

normalizer = CategoricalNormalizer()
validator = SchemaValidator()

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
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)
        else:
            X[col] = X[col].astype(str).fillna("missing")

    return X[COLUMNS]

# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)

# ---------------- PREDICT ----------------

@app.route("/predict_response/", methods=["POST"])
def predict_response():

    payload = request.get_json(silent=True)

    logging.info(f"NEW /predict_response REQUEST: {payload}")

    if payload is None:
        return jsonify({"error": "invalid json"}), 400

    raw_payload = payload.copy()

    valid, error = validator.validate("predict", payload)
    if not valid:
        logging.error(f"VALIDATION ERROR: {error}")
        return jsonify({"error": error}), 422

    try:
        X = make_features(payload)
        pred = float(model.predict(X)[0])

        Prediction.create(
            unit_id=payload["unit_id"],
            received_dttm=payload["received_dttm"],
            predicted_response_time_seconds=pred,
            raw_predict_payload=raw_payload,
            normalized_predict_payload=payload
        )

        return jsonify({
            "unit_id": payload["unit_id"],
            "received_dttm": payload["received_dttm"],
            "predicted_response_time_seconds": pred
        })

    except IntegrityError:
        logging.error("DUPLICATE RECORD")
        return jsonify({"error": "duplicate record"}), 422

    except Exception as e:
        logging.exception("PREDICTION FAILED")
        return jsonify({"error": str(e)}), 500


# ---------------- ACTUAL ----------------

@app.route("/actual_response/", methods=["POST"])
def actual_response():

    payload = request.get_json(silent=True)

    logging.info(f"NEW /actual_response REQUEST: {payload}")

    if payload is None:
        return jsonify({"error": "invalid json"}), 400

    raw_payload = payload.copy()

    valid, error = validator.validate("actual", payload)
    if not valid:
        logging.error(f"VALIDATION ERROR: {error}")
        return jsonify({"error": error}), 422

    record = Prediction.get_or_none(
        (Prediction.unit_id == payload["unit_id"]) &
        (Prediction.received_dttm == payload["received_dttm"])
    )

    if record is None:
        logging.error("RECORD NOT FOUND")
        return jsonify({"error": "record not found"}), 422

    try:
        actual_seconds = (
            pd.to_datetime(payload["on_scene_dttm"]) -
            pd.to_datetime(payload["received_dttm"])
        ).total_seconds()

        record.on_scene_dttm = payload["on_scene_dttm"]
        record.actual_response_time_seconds = actual_seconds
        record.raw_actual_payload = raw_payload
        record.normalized_actual_payload = payload
        record.save()

        return jsonify({
            "actual_response_time_seconds": actual_seconds,
            "predicted_response_time_seconds": record.predicted_response_time_seconds
        })

    except Exception as e:
        logging.exception("ACTUAL UPDATE FAILED")
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
