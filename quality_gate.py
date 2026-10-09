
import json
import os
import sys

METRICS_FILE = "metrics.json"

if not os.path.exists(METRICS_FILE):
    print("QUALITY GATE FAILED: metrics.json not found.")
    sys.exit(1)

with open(METRICS_FILE, "r") as file:
    metrics = json.load(file)

r2_score = metrics.get("r2_score")

if r2_score is None:
    print("QUALITY GATE FAILED: r2_score is missing.")
    sys.exit(1)

print(f"R2 Score: {r2_score:.4f}")

if r2_score < 0.70:
    print("QUALITY GATE FAILED: R2 score is below 0.70.")
    sys.exit(1)

print("QUALITY GATE PASSED: R2 score is at least 0.70.")
