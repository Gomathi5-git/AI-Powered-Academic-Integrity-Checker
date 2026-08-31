import time
import hashlib
import json
import joblib
import redis
import pandas as pd
import numpy as np


# ---------------------------------------------------------
# Load model and vectorizer
# ---------------------------------------------------------

model = joblib.load("best_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("Model and vectorizer loaded successfully.")


# ---------------------------------------------------------
# Connect to Redis
# ---------------------------------------------------------

cache = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

cache.ping()

print("Redis connection successful.")


# ---------------------------------------------------------
# Normal inference
# ---------------------------------------------------------

def normal_prediction(text):
    start = time.perf_counter()

    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]

    latency = (time.perf_counter() - start) * 1000

    return prediction, latency


# ---------------------------------------------------------
# Cached inference
# ---------------------------------------------------------

def cached_prediction(text):
    text_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    cache_key = f"plagiarism_prediction:{text_hash}"

    start = time.perf_counter()

    cached_result = cache.get(cache_key)

    if cached_result is not None:
        result = json.loads(cached_result)

        latency = (time.perf_counter() - start) * 1000

        return result["prediction"], latency, True

    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]

    result = {
        "prediction": prediction
    }

    # Store result for one hour
    cache.set(
        cache_key,
        json.dumps(result),
        ex=3600
    )

    latency = (time.perf_counter() - start) * 1000

    return prediction, latency, False


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------

df = pd.read_csv("training_data_balanced.csv")

texts = df["text"].tolist()

print(f"Benchmark samples: {len(texts)}")


# ---------------------------------------------------------
# Baseline benchmark
# ---------------------------------------------------------

baseline_latencies = []

for text in texts:

    prediction, latency = normal_prediction(text)

    baseline_latencies.append(latency)


# ---------------------------------------------------------
# Cached benchmark
# ---------------------------------------------------------

cached_latencies = []
cache_hits = 0

# First populate the cache
for text in texts:
    cached_prediction(text)

# Measure cached requests
for text in texts:

    prediction, latency, cache_hit = cached_prediction(text)

    cached_latencies.append(latency)

    if cache_hit:
        cache_hits += 1


# ---------------------------------------------------------
# Calculate statistics
# ---------------------------------------------------------

baseline_average = np.mean(baseline_latencies)
baseline_min = np.min(baseline_latencies)
baseline_max = np.max(baseline_latencies)

cached_average = np.mean(cached_latencies)
cached_min = np.min(cached_latencies)
cached_max = np.max(cached_latencies)

latency_reduction = (
    (baseline_average - cached_average)
    / baseline_average
) * 100

cache_hit_rate = (
    cache_hits / len(texts)
) * 100


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\n" + "=" * 65)
print("INFERENCE OPTIMIZATION RESULTS")
print("=" * 65)

print("\nBaseline Inference")
print("-" * 65)

print(f"Average latency : {baseline_average:.4f} ms")
print(f"Minimum latency : {baseline_min:.4f} ms")
print(f"Maximum latency : {baseline_max:.4f} ms")

print("\nRedis Cached Inference")
print("-" * 65)

print(f"Average latency : {cached_average:.4f} ms")
print(f"Minimum latency : {cached_min:.4f} ms")
print(f"Maximum latency : {cached_max:.4f} ms")

print("\nCaching Performance")
print("-" * 65)

print(f"Cache hits      : {cache_hits}/{len(texts)}")
print(f"Cache hit rate  : {cache_hit_rate:.2f}%")
print(f"Latency reduction: {latency_reduction:.2f}%")

print("\n" + "=" * 65)
print("INFERENCE OPTIMIZATION COMPLETED")
print("=" * 65)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results = pd.DataFrame({
    "Metric": [
        "Baseline Average Latency (ms)",
        "Cached Average Latency (ms)",
        "Baseline Minimum Latency (ms)",
        "Cached Minimum Latency (ms)",
        "Baseline Maximum Latency (ms)",
        "Cached Maximum Latency (ms)",
        "Cache Hit Rate (%)",
        "Latency Reduction (%)"
    ],
    "Value": [
        baseline_average,
        cached_average,
        baseline_min,
        cached_min,
        baseline_max,
        cached_max,
        cache_hit_rate,
        latency_reduction
    ]
})

results.to_csv(
    "inference_optimization_results.csv",
    index=False
)

print("\nResults saved to:")
print("  inference_optimization_results.csv")