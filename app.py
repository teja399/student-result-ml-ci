
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

@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "seoul-bike-prediction"
    })

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
        # Build the original raw input columns.
        sample = pd.DataFrame([{
            name: data[name] for name in FEATURES
        }])

        # The training pipeline expects date-derived features
        # and one-hot-encoded categorical columns.
        sample["Day"] = int(data.get("Day", 9))
        sample["Month"] = int(data.get("Month", 10))
        sample["Year"] = int(data.get("Year", 2018))
        sample["DayOfWeek"] = int(data.get("DayOfWeek", 1))

        sample["Seasons_Spring"] = int(
            data["Seasons"] == "Spring"
        )
        sample["Seasons_Summer"] = int(
            data["Seasons"] == "Summer"
        )
        sample["Seasons_Winter"] = int(
            data["Seasons"] == "Winter"
        )
        sample["Holiday_No Holiday"] = int(
            data["Holiday"] == "No Holiday"
        )
        sample["Functioning Day_Yes"] = int(
            data["Functioning Day"] == "Yes"
        )

        model = joblib.load(MODEL_PATH)

        # Keep only the features actually expected by the
        # fitted model, in the order stored during training.
        expected = list(model.feature_names_in_)
        sample = sample.reindex(columns=expected, fill_value=0)

        prediction = float(model.predict(sample)[0])

        return jsonify({
            "predicted_rented_bike_count": round(
                max(0.0, prediction), 2
            )
        }), 200

    except Exception as error:
        app.logger.exception("Prediction failed")
        return jsonify({
            "error": "Prediction failed",
            "details": str(error)
        }), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
