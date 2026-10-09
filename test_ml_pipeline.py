
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
        with open("metrics.json", "r", encoding="utf-8") as file:
            metrics = json.load(file)

        self.assertIn("r2_score", metrics)
        self.assertIn("mae", metrics)
        self.assertIn("rmse", metrics)

        self.assertGreaterEqual(metrics["mae"], 0)
        self.assertGreaterEqual(metrics["rmse"], 0)

    def test_model_can_predict(self):
        model = joblib.load("seoul_bike_model.pkl")
        df = pd.read_csv("seoul_bike_processed.csv")
        df.columns = df.columns.str.strip()

        target = "Rented Bike Count"
        self.assertIn(target, df.columns)

        # Remove the target column to obtain input features
        X = df.drop(columns=[target]).copy()

        # Match the date processing used during training,
        # but only when the Date column exists.
        if "Date" in X.columns:
            date_values = pd.to_datetime(
                X["Date"], dayfirst=True, errors="coerce"
            )
            X["Month"] = date_values.dt.month
            X["DayOfWeek"] = date_values.dt.dayofweek
            X = X.drop(columns=["Date"])

        # Match Boolean preprocessing used during training
        bool_columns = X.select_dtypes(include=["bool"]).columns
        for column in bool_columns:
            X[column] = X[column].astype(str)

        predictions = model.predict(X.head(5))

        self.assertEqual(len(predictions), min(5, len(X)))
        self.assertTrue(
            all(pd.notna(value) for value in predictions)
        )


if __name__ == "__main__":
    unittest.main()
