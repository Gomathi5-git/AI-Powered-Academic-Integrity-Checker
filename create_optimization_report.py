import pandas as pd


# Load benchmark results
df = pd.read_csv("inference_optimization_results.csv")

metrics = dict(zip(df["Metric"], df["Value"]))


baseline = metrics["Baseline Average Latency (ms)"]
cached = metrics["Cached Average Latency (ms)"]
reduction = metrics["Latency Reduction (%)"]
hit_rate = metrics["Cache Hit Rate (%)"]


report = f"""
============================================================
INFERENCE OPTIMIZATION REPORT
AI-POWERED ACADEMIC INTEGRITY CHECKER
============================================================

1. OBJECTIVE
------------------------------------------------------------

The inference process was optimized to reduce prediction
latency for repeated plagiarism-checking requests.

Redis caching was implemented so that previously processed
text can return its stored prediction without running the
machine learning model again.


2. MODEL
------------------------------------------------------------

Model:
    RandomForestClassifier

Feature processing:
    TF-IDF Vectorizer


3. BASELINE INFERENCE
------------------------------------------------------------

Without caching, every request performs:

    Text
      |
      v
    TF-IDF Transformation
      |
      v
    Random Forest Prediction
      |
      v
    Result


Average latency:
    {baseline:.4f} ms


4. REDIS CACHING
------------------------------------------------------------

Redis/Memurai was integrated as an inference cache.

For every input text, a SHA-256 hash is generated and used
as the Redis cache key.

If the result already exists:

    Text
      |
      v
    Redis Cache
      |
      v
    Cached Prediction

The machine learning model does not need to run again.


5. OPTIMIZED INFERENCE
------------------------------------------------------------

Average cached latency:
    {cached:.4f} ms

Cache hit rate:
    {hit_rate:.2f}%


6. PERFORMANCE IMPROVEMENT
------------------------------------------------------------

Average latency reduction:
    {reduction:.2f}%

The cached inference path is significantly faster for
repeated requests because it avoids TF-IDF transformation
and model prediction.


7. CACHING STRATEGY
------------------------------------------------------------

Cache system:
    Redis / Memurai

Cache key:
    SHA-256 hash of input text

Cache expiration:
    3600 seconds (1 hour)

The expiration prevents cached results from remaining
indefinitely in the system.


8. VALIDATION
------------------------------------------------------------

The same text was submitted multiple times.

First request:
    Cache miss
    Model performs prediction
    Result stored in Redis

Repeated request:
    Cache hit
    Stored prediction returned directly

The prediction remains consistent between the cached and
non-cached requests.


9. LIMITATIONS
------------------------------------------------------------

Caching primarily improves repeated or identical requests.

A new text that is not already present in Redis still
requires normal model inference.

The benchmark was performed on a small local dataset and
local development environment. Production latency may
differ depending on server hardware, network conditions,
request volume and Redis configuration.


10. FUTURE IMPROVEMENTS
------------------------------------------------------------

Possible future improvements include:

- Integrating Redis directly into the FastAPI application
- Adding cache hit/miss monitoring
- Testing with larger request volumes
- Load testing with concurrent users
- Configuring automatic cache invalidation
- Adding production monitoring dashboards
- Optimizing model loading and application startup
- Containerizing the API and Redis service


11. CONCLUSION
------------------------------------------------------------

The inference optimization phase successfully implemented
Redis-based caching for repeated plagiarism detection
requests.

The benchmark demonstrates a measurable reduction in
prediction latency when cached results are available.

Redis caching therefore provides an effective optimization
for improving the responsiveness of the Academic Integrity
Checker for repeated requests.


============================================================
END OF INFERENCE OPTIMIZATION REPORT
============================================================
"""


with open("inference_optimization_report.txt", "w", encoding="utf-8") as file:
    file.write(report)

print("Optimization report created:")
print("  inference_optimization_report.txt")