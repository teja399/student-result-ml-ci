
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

# 1. Load dataset
df = pd.read_csv("seoul_bike_processed.csv")
df.columns = df.columns.str.strip()

# 2. Define target
target = "Rented Bike Count"

if target not in df.columns:
    raise ValueError(
        f"Target '{target}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

# 3. Clean target values
df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target]).copy()

if len(df) < 10:
    raise ValueError("Dataset contains too few valid rows.")

# 4. Extract date features if Date exists
if "Date" in df.columns:
    date_values = pd.to_datetime(
        df["Date"], dayfirst=True, errors="coerce"
    )
    df["Month"] = date_values.dt.month
    df["DayOfWeek"] = date_values.dt.dayofweek
    df = df.drop(columns=["Date"])

# 5. Separate features and target
X = df.drop(columns=[target]).copy()
y = df[target]

if X.empty:
    raise ValueError("No input features found.")

# 6. Convert Boolean columns to categorical strings
bool_columns = X.select_dtypes(include=["bool"]).columns

for column in bool_columns:
    X[column] = X[column].astype(str)

# Identify numeric and categorical columns
numeric_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category", "string"]
).columns.tolist()

if not numeric_features and not categorical_features:
    raise ValueError("No usable input features were detected.")

print("Dataset rows:", len(df))
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

# 7. Build preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []

if numeric_features:
    transformers.append(
        ("num", numeric_pipeline, numeric_features)
    )

if categorical_features:
    transformers.append(
        ("cat", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(transformers=transformers)

# 8. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 9. Build and train model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

# 10. Evaluate model
predictions = model.predict(X_test)

metrics = {
    "model": "RandomForestRegressor",
    "training_samples": int(len(X_train)),
    "testing_samples": int(len(X_test)),
    "mae": float(mean_absolute_error(y_test, predictions)),
    "rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
    "r2_score": float(r2_score(y_test, predictions))
}

# 11. Save trained model
joblib.dump(model, "seoul_bike_model.pkl")

# 12. Save evaluation metrics
with open("metrics.json", "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

# 13. Display results
print("\nMODEL TRAINING COMPLETED")
print(json.dumps(metrics, indent=4))
print("\nGenerated files:")
print("- seoul_bike_model.pkl")
print("- metrics.json")
