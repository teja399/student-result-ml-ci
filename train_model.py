
import json
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("seoul_bike_processed.csv")

# Remove rows where the target is missing
df = df.dropna(subset=["Rented Bike Count"])

# Extract date features
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df["Month"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek
df = df.drop(columns=["Date"])

# Separate features and target
X = df.drop(columns=["Rented Bike Count"])
y = df["Rented Bike Count"]

# Identify column types
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

# Preprocessing
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build and train model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)

metrics = {
    "model": "RandomForestRegressor",
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "mae": float(mean_absolute_error(y_test, predictions)),
    "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
    "r2_score": float(r2_score(y_test, predictions))
}

# Save trained model
joblib.dump(model, "seoul_bike_model.pkl")

# Save metrics
with open("metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)

print("Model training completed.")
print(json.dumps(metrics, indent=4))
