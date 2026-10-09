
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

MODEL_PATH = Path("seoul_bike_model.pkl")

FEATURES = [
    "Hour",
    "Temperature(°C)",
    "Humidity(%)",
    "Wind speed (m/s)",
    "Visibility (10m)",
    "Dew point temperature(°C)",
    "Solar Radiation (MJ/m2)",
    "Rainfall(mm)",
    "Snowfall (cm)",
    "Seasons",
    "Holiday",
    "Functioning Day",
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "seoul-bike-prediction"
    }), 200


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({"error": "A JSON object is required"}), 400

    missing = [name for name in FEATURES if name not in data]
    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing
        }), 400

    try:
        sample = pd.DataFrame(
            [{name: data[name] for name in FEATURES}],
            columns=FEATURES
        )

        model = load_model()
        prediction = float(model.predict(sample)[0])

        return jsonify({
            "predicted_rented_bike_count": round(max(0.0, prediction), 2)
        }), 200

    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 500
    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed",
            "details": str(error)
        }), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
