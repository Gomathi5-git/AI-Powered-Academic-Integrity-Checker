import time
import hashlib
import json
import joblib
import redis


# ---------------------------------------------------------
# Load model and vectorizer
# ---------------------------------------------------------

model = joblib.load("best_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

print("Model and vectorizer loaded successfully.")


# ---------------------------------------------------------
# Connect to Redis / Memurai
# ---------------------------------------------------------

cache = redis.Redis(
    host="localhost",
    port=6379,
    decode_responses=True
)

if not cache.ping():
    raise ConnectionError("Redis connection failed.")

print("Redis connection successful.")


# ---------------------------------------------------------
# Prediction function with caching
# ---------------------------------------------------------

def predict_with_cache(text):

    # Create a unique key for the input text
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    cache_key = f"plagiarism_prediction:{text_hash}"

    # Check Redis first
    cached_result = cache.get(cache_key)

    if cached_result is not None:
        return json.loads(cached_result), True

    # Cache miss → run the ML model
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]

    # Store result in Redis
    result = {
        "prediction": prediction
    }

    cache.setex(
        cache_key,
        3600,
        json.dumps(result)
    )

    return result, False


# ---------------------------------------------------------
# Test text
# ---------------------------------------------------------

test_text = (
    "Artificial intelligence helps computers analyze "
    "large amounts of information."
)


# ---------------------------------------------------------
# First request - Cache MISS
# ---------------------------------------------------------

start = time.perf_counter()

result_1, cache_hit_1 = predict_with_cache(test_text)

latency_1 = (time.perf_counter() - start) * 1000


# ---------------------------------------------------------
# Second request - Cache HIT
# ---------------------------------------------------------

start = time.perf_counter()

result_2, cache_hit_2 = predict_with_cache(test_text)

latency_2 = (time.perf_counter() - start) * 1000


# ---------------------------------------------------------
# Display results
# ---------------------------------------------------------

print("\n" + "=" * 60)
print("REDIS INFERENCE CACHE TEST")
print("=" * 60)

print("\nFirst Request")
print("-" * 60)
print(f"Prediction : {result_1['prediction']}")
print(f"Cache hit  : {cache_hit_1}")
print(f"Latency    : {latency_1:.4f} ms")

print("\nSecond Request")
print("-" * 60)
print(f"Prediction : {result_2['prediction']}")
print(f"Cache hit  : {cache_hit_2}")
print(f"Latency    : {latency_2:.4f} ms")

if latency_2 > 0:
    improvement = ((latency_1 - latency_2) / latency_1) * 100

    print("\nPerformance Improvement")
    print("-" * 60)
    print(f"Latency reduction: {improvement:.2f}%")

print("\n" + "=" * 60)
print("REDIS CACHE TEST COMPLETED")
print("=" * 60)