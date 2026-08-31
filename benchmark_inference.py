import time
import joblib
import pandas as pd
import numpy as np

# Load model and vectorizer
model = joblib.load("best_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load test data
df = pd.read_csv("training_data_balanced.csv")

texts = df["text"].tolist()

print("=" * 60)
print("INFERENCE PERFORMANCE BENCHMARK")
print("=" * 60)
print(f"Model: {type(model).__name__}")
print(f"Test samples: {len(texts)}")

latencies = []
predictions = []

# Run inference for every sample
for text in texts:
    start_time = time.perf_counter()

    # Convert text into TF-IDF features
    features = vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(features)[0]

    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    latencies.append(latency_ms)
    predictions.append(prediction)

# Calculate statistics
average_latency = np.mean(latencies)
minimum_latency = np.min(latencies)
maximum_latency = np.max(latencies)
median_latency = np.median(latencies)

print("\nLatency Results")
print("-" * 60)
print(f"Average latency : {average_latency:.4f} ms")
print(f"Minimum latency : {minimum_latency:.4f} ms")
print(f"Maximum latency : {maximum_latency:.4f} ms")
print(f"Median latency  : {median_latency:.4f} ms")

print("\nPrediction Distribution")
print("-" * 60)

unique, counts = np.unique(predictions, return_counts=True)

for label, count in zip(unique, counts):
    percentage = (count / len(predictions)) * 100
    print(f"{label:<15}: {count} ({percentage:.2f}%)")

print("\n" + "=" * 60)
print("BASELINE BENCHMARK COMPLETED")
print("=" * 60)