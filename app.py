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
# LOAD MODEL + SCHEMA
# =========================================================

with open("columns.json") as f:
    COLUMNS = json.load(f)

with open("dtypes.pickle", "rb") as f:
    DTYPES = pickle.load(f)

model = joblib.load("model.pickle")

# =========================================================
# CATEGORICAL CONFIG
# =========================================================

CATEGORICAL_CONFIG = {
    "unit_type": {
        "type": "map",
        "allow_token_fallback": True,
        "map": {
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
    },

    "call_type_group": {
        "type": "map",
        "map": {
            "POTENTIALLY LIFE THREATENING": "Potentially Life-Threatening",
            "NON LIFE THREATENING": "Non Life-threatening",
            "ALARM": "Alarm",
            "FIRE": "Fire"
        }
    },

    "battalion": {
        "type": "regex_extract",
        "pattern": r"(\d+)",
        "format": "B{:02d}"
    }
}

# =========================================================
# VALIDATION SCHEMA
# =========================================================

VALIDATION_SCHEMA = {
    "predict": {
        "call_type": {"type": "string", "required": True},
        "call_type_group": {"type": "categorical", "required": True, "normalize": "call_type_group"},
        "original_priority": {"type": "enum", "required": True, "allowed": ['1','2','3','E','I','A','B','T']},
        "unit_id": {"type": "string", "required": True},
        "unit_type": {"type": "categorical", "required": True, "normalize": "unit_type"},
        "station_area": {"type": "string", "required": True},
        "battalion": {"type": "categorical", "required": True, "normalize": "battalion",
                      "allowed": {"B01","B02","B03","B04","B05","B06","B07","B08","B09"}},
        "neighborhood_district": {"type": "string", "required": True},
        "zipcode_of_incident": {"type": "enum", "required": True,
                                "allowed": {"94102","94103","94104","94105","94107","94108","94109","94110",
                                            "94111","94112","94114","94115","94116","94117","94118","94121",
                                            "94122","94123","94124","94127","94129","94130","94131","94132",
                                            "94133","94134","94158"}},
        "received_dttm": {"type": "datetime", "required": True}
    },

    "actual": {
        "unit_id": {"type": "string", "required": True},
        "received_dttm": {"type": "datetime", "required": True},
        "on_scene_dttm": {"type": "datetime", "required": True}
    }
}

# =========================================================
# NORMALIZER
# =========================================================

class CategoricalNormalizer:

    def __init__(self, config):
        self.config = config

    def normalize_text(self, value):
        if value is None:
            return None

        v = str(value).upper().strip()
        v = re.sub(r"[^A-Z0-9]+", " ", v)
        v = re.sub(r"\s+", " ", v)
        return v.strip()

    def normalize(self, field, value):
        if field not in self.config:
            return value

        rule = self.config[field]

        if rule["type"] == "map":
            v = self.normalize_text(value)
            if not v:
                return None

            mapping = rule["map"]

            if v in mapping:
                return mapping[v]

            if rule.get("allow_token_fallback", False):
                base = re.split(r"\s+", v)[0]
                if base in mapping:
                    return mapping[base]

            return None

        if rule["type"] == "regex_extract":
            match = re.search(rule["pattern"], str(value).upper())
            if not match:
                return None
            num = int(match.group(1))
            return rule["format"].format(num)

        return None

# =========================================================
# VALIDATOR
# =========================================================

class SchemaValidator:

    def __init__(self, schema, normalizer):
        self.schema = schema
        self.normalizer = normalizer

    def validation_error(self, field, value, allowed=None):
        if allowed:
            allowed_str = ", ".join(sorted(map(str, allowed)))
            return f"{field} '{value}' is invalid. Allowed values: {allowed_str}"
        return f"{field} '{value}' is invalid"

    def validate(self, section, data):

        if not isinstance(data, dict):
            return False, "payload must be valid json"

        rules = self.schema[section]

        for field, rule in rules.items():
            if rule.get("required") and field not in data:
                return False, f"missing field: {field}"

        for field, rule in rules.items():

            value = data.get(field)

            if value is None:
                continue

            field_type = rule["type"]

            if field_type == "string":
                if not isinstance(value, str) or not value.strip():
                    return False, f"{field} must be a non-empty string"

            elif field_type == "enum":
                if str(value) not in rule["allowed"]:
                    return False, self.validation_error(field, value, rule["allowed"])

            elif field_type == "datetime":
                try:
                    pd.to_datetime(value)
                except:
                    return False, f"{field} must be valid ISO 8601 datetime"

            elif field_type == "categorical":
                normalized = self.normalizer.normalize(rule["normalize"], value)

                if normalized is None:
                    config = CATEGORICAL_CONFIG[rule["normalize"]]
                    allowed = set(config["map"].values()) if config["type"] == "map" else None
                    return False, self.validation_error(field, value, allowed)

                if "allowed" in rule and normalized not in rule["allowed"]:
                    return False, self.validation_error(field, value, rule["allowed"])

                data[field] = normalized

        # =====================================================
        # FIX: CROSS-FIELD VALIDATION (MISSING IN ORIGINAL)
        # =====================================================

        if section == "actual":

            received = pd.to_datetime(data["received_dttm"])
            on_scene = pd.to_datetime(data["on_scene_dttm"])

            if on_scene <= received:
                return False, (
                    "on_scene_dttm must be later than received_dttm"
                )

        return True, None

# =========================================================
# INIT
# =========================================================

normalizer = CategoricalNormalizer(CATEGORICAL_CONFIG)
validator = SchemaValidator(VALIDATION_SCHEMA, normalizer)

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
            X[col] = X[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        else:
            X[col] = X[col].astype(str).replace("nan", "missing").fillna("missing")

    return X[COLUMNS]

# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)

@app.route("/predict_response/", methods=["POST"])
def predict_response():

    payload = request.get_json()

    raw_payload = payload.copy() if isinstance(payload, dict) else payload

    valid, error = validator.validate("predict", payload)
    if not valid:
        return jsonify({"error": error}), 422

    try:
        X = make_features(payload)
        pred = float(model.predict(X)[0])

        Prediction.create(
            unit_id=payload["unit_id"],
            received_dttm=payload["received_dttm"],
            predicted_response_time_seconds=pred,

            call_type=payload["call_type"],
            call_type_group=payload["call_type_group"],
            original_priority=payload["original_priority"],
            unit_type=payload["unit_type"],
            station_area=payload["station_area"],
            battalion=payload["battalion"],
            neighborhood_district=payload["neighborhood_district"],
            zipcode_of_incident=payload["zipcode_of_incident"],

            raw_predict_payload=raw_payload,
            normalized_predict_payload=payload
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
        return jsonify({
            "error": "prediction failed",
            "detail": str(e)
        }), 422


@app.route("/actual_response/", methods=["POST"])
def actual_response():

    payload = request.get_json()

    raw_payload = payload.copy() if isinstance(payload, dict) else payload

    valid, error = validator.validate("actual", payload)
    if not valid:
        return jsonify({"error": error}), 422

    record = Prediction.get_or_none(
        (Prediction.unit_id == payload["unit_id"]) &
        (Prediction.received_dttm == payload["received_dttm"])
    )

    if record is None:
        return jsonify({
            "error": "record not found for provided unit_id and received_dttm"
        }), 422

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
            "unit_id": record.unit_id,
            "received_dttm": record.received_dttm,
            "on_scene_dttm": payload["on_scene_dttm"],
            "actual_response_time_seconds": actual_seconds,
            "predicted_response_time_seconds": record.predicted_response_time_seconds
        })

    except Exception as e:
        return jsonify({
            "error": "update failed",
            "detail": str(e)
        }), 422


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)