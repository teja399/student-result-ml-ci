
import unittest

from app import app


class TestPredictionApplication(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_bike_count_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "Hour": 10,
                "Temperature(°C)": 18.0,
                "Humidity(%)": 45,
                "Wind speed (m/s)": 2.0,
                "Visibility (10m)": 1500,
                "Dew point temperature(°C)": 6.0,
                "Solar Radiation (MJ/m2)": 0.8,
                "Rainfall(mm)": 0.0,
                "Snowfall (cm)": 0.0,
                "Seasons": "Spring",
                "Holiday": "No Holiday",
                "Functioning Day": "Yes"
            }
        )

        self.assertEqual(response.status_code, 200)

        result = response.get_json()
        self.assertIn("predicted_rented_bike_count", result)
        self.assertIsInstance(
            result["predicted_rented_bike_count"], (int, float)
        )

    def test_missing_field_validation(self):
        response = self.client.post(
            "/predict",
            json={
                "Hour": 10,
                "Temperature(°C)": 18.0
            }
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())

    def test_json_body_required(self):
        response = self.client.post("/predict")

        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.get_json())


if __name__ == "__main__":
    unittest.main()
