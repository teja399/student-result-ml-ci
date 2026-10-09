
import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_model_created(self):
        self.assertTrue(os.path.exists("seoul_bike_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_metrics_are_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        self.assertIn("r2_score", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)
        self.assertGreaterEqual(metrics["mae"], 0)
        self.assertGreaterEqual(metrics["rmse"], 0)

    def test_model_can_predict(self):
        model = joblib.load("seoul_bike_model.pkl")
        df = pd.read_csv("seoul_bike_processed.csv")

        X = df.drop(columns=["Rented Bike Count"])
        X["Date"] = pd.to_datetime(
            X["Date"], dayfirst=True, errors="coerce"
        )
        X["Month"] = X["Date"].dt.month
        X["DayOfWeek"] = X["Date"].dt.dayofweek
        X = X.drop(columns=["Date"])

        prediction = model.predict(X.head(5))

        self.assertEqual(len(prediction), 5)
        self.assertTrue(all(value >= 0 for value in prediction))


if __name__ == "__main__":
    unittest.main()
