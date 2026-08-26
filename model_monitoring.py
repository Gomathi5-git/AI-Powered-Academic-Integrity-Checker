import time
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.metrics import accuracy_score


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "best_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
DATA_PATH = "training_data_balanced.csv"

LATENCY_THRESHOLD_MS = 100.0
DRIFT_THRESHOLD = 0.20


# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

print("Model and vectorizer loaded successfully.")
print("Model:", type(model).__name__)


# ============================================================
# LOAD MONITORING DATA
# ============================================================

df = pd.read_csv(DATA_PATH)

texts = df["text"].fillna("").astype(str)
actual_labels = df["label"]


print("Monitoring dataset loaded:", len(df), "samples")


# ============================================================
# PREDICTION + LATENCY MONITORING
# ============================================================

predictions = []
latencies = []

for text in texts:

    start_time = time.perf_counter()

    features = vectorizer.transform([text])

    prediction = model.predict(features)[0]

    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    predictions.append(prediction)
    latencies.append(latency_ms)


# ============================================================
# PREDICTION DISTRIBUTION
# ============================================================

prediction_series = pd.Series(predictions)

prediction_counts = prediction_series.value_counts()

total_predictions = len(predictions)

original_count = int(
    prediction_counts.get("original", 0)
)

plagiarized_count = int(
    prediction_counts.get("plagiarized", 0)
)

original_percentage = original_count / total_predictions

plagiarized_percentage = plagiarized_count / total_predictions


# ============================================================
# LATENCY METRICS
# ============================================================

average_latency = float(np.mean(latencies))

maximum_latency = float(np.max(latencies))

minimum_latency = float(np.min(latencies))


latency_alert = average_latency > LATENCY_THRESHOLD_MS


# ============================================================
# BASELINE DISTRIBUTION
# ============================================================

baseline_distribution = {
    "original": 0.50,
    "plagiarized": 0.50
}


# ============================================================
# DRIFT CALCULATION
# ============================================================

original_drift = abs(
    original_percentage -
    baseline_distribution["original"]
)

plagiarized_drift = abs(
    plagiarized_percentage -
    baseline_distribution["plagiarized"]
)

maximum_drift = max(
    original_drift,
    plagiarized_drift
)

drift_detected = maximum_drift > DRIFT_THRESHOLD


# ============================================================
# MODEL PERFORMANCE
# ============================================================

accuracy = accuracy_score(
    actual_labels,
    predictions
)


# ============================================================
# ALERTS
# ============================================================

alerts = []

if latency_alert:
    alerts.append(
        "High latency detected"
    )

if drift_detected:
    alerts.append(
        "Prediction distribution drift detected"
    )


if not alerts:
    alerts.append(
        "No monitoring alerts detected"
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 60)
print("MODEL MONITORING RESULTS")
print("=" * 60)

print()
print("Prediction Distribution")
print("-" * 60)

print(
    f"Original      : {original_count} "
    f"({original_percentage * 100:.2f}%)"
)

print(
    f"Plagiarized   : {plagiarized_count} "
    f"({plagiarized_percentage * 100:.2f}%)"
)

print()
print("Latency")
print("-" * 60)

print(
    f"Average latency : {average_latency:.4f} ms"
)

print(
    f"Minimum latency : {minimum_latency:.4f} ms"
)

print(
    f"Maximum latency : {maximum_latency:.4f} ms"
)

print(
    f"Latency threshold: {LATENCY_THRESHOLD_MS:.2f} ms"
)

print()
print("Model Performance")
print("-" * 60)

print(
    f"Accuracy: {accuracy:.4f}"
)

print()
print("Drift Detection")
print("-" * 60)

print(
    f"Maximum distribution drift: "
    f"{maximum_drift:.4f}"
)

print(
    f"Drift threshold: "
    f"{DRIFT_THRESHOLD:.4f}"
)

print(
    f"Drift detected: "
    f"{drift_detected}"
)

print()
print("Alerts")
print("-" * 60)

for alert in alerts:
    print("ALERT:", alert)

print("=" * 60)


# ============================================================
# SAVE MONITORING LOG
# ============================================================

monitoring_record = {
    "timestamp": datetime.now().isoformat(),

    "model": type(model).__name__,

    "total_predictions": total_predictions,

    "prediction_distribution": {
        "original": original_percentage,
        "plagiarized": plagiarized_percentage
    },

    "latency": {
        "average_ms": average_latency,
        "minimum_ms": minimum_latency,
        "maximum_ms": maximum_latency
    },

    "accuracy": accuracy,

    "drift": {
        "maximum_distribution_drift": maximum_drift,
        "threshold": DRIFT_THRESHOLD,
        "detected": drift_detected
    },

    "alerts": alerts
}


with open(
    "monitoring_log.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        monitoring_record,
        file,
        indent=4
    )


# ============================================================
# SAVE MONITORING CSV
# ============================================================

monitoring_table = pd.DataFrame(
    [{
        "timestamp": monitoring_record["timestamp"],
        "model": monitoring_record["model"],
        "total_predictions": total_predictions,
        "original_percentage": original_percentage,
        "plagiarized_percentage": plagiarized_percentage,
        "average_latency_ms": average_latency,
        "minimum_latency_ms": minimum_latency,
        "maximum_latency_ms": maximum_latency,
        "accuracy": accuracy,
        "maximum_drift": maximum_drift,
        "drift_detected": drift_detected,
        "latency_alert": latency_alert
    }]
)

monitoring_table.to_csv(
    "monitoring_results.csv",
    index=False
)


# ============================================================
# CREATE PREDICTION DISTRIBUTION CHART
# ============================================================

labels = [
    "Original",
    "Plagiarized"
]

values = [
    original_count,
    plagiarized_count
]

plt.figure(figsize=(8, 5))

plt.bar(
    labels,
    values
)

plt.title(
    "Prediction Distribution"
)

plt.ylabel(
    "Number of Predictions"
)

plt.tight_layout()

plt.savefig(
    "prediction_distribution.png",
    dpi=150
)

plt.close()


# ============================================================
# CREATE LATENCY CHART
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(latencies) + 1),
    latencies,
    marker="o"
)

plt.axhline(
    LATENCY_THRESHOLD_MS,
    linestyle="--",
    label="Latency Threshold"
)

plt.title(
    "Prediction Latency"
)

plt.xlabel(
    "Prediction Number"
)

plt.ylabel(
    "Latency (ms)"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "prediction_latency.png",
    dpi=150
)

plt.close()


print()
print("Monitoring files created:")
print("  monitoring_log.json")
print("  monitoring_results.csv")
print("  prediction_distribution.png")
print("  prediction_latency.png")