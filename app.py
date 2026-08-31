from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import redis
import hashlib
import json
import time


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="AI-Powered Academic Integrity Checker",
    description="Plagiarism detection API with Redis caching",
    version="2.0"
)


# ============================================================
# LOAD MODEL AND VECTORIZER
# ============================================================

try:
    model = joblib.load("best_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    print("Model and vectorizer loaded successfully.")
    print(f"Model: {type(model).__name__}")

except Exception as e:
    print(f"Error loading model/vectorizer: {e}")
    raise


# ============================================================
# REDIS / MEMURAI CONNECTION
# ============================================================

try:
    redis_client = redis.Redis(
        host="localhost",
        port=6379,
        decode_responses=True
    )

    redis_client.ping()

    print("Redis cache connected successfully.")

except Exception as e:
    print(f"Redis connection failed: {e}")
    redis_client = None


# ============================================================
# REQUEST MODEL
# ============================================================

class TextRequest(BaseModel):
    text: str


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def home():
    return {
        "message": "AI-Powered Academic Integrity Checker API",
        "status": "running",
        "model": type(model).__name__,
        "redis_cache": redis_client is not None
    }


@app.get("/health")
def health():
    redis_status = False

    if redis_client is not None:
        try:
            redis_status = redis_client.ping()
        except Exception:
            redis_status = False

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "redis_connected": redis_status
    }


# ============================================================
# PREDICTION WITH REDIS CACHE
# ============================================================

def predict_with_cache(text: str):

    # --------------------------------------------------------
    # Create unique cache key
    # --------------------------------------------------------

    text_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    cache_key = f"plagiarism_prediction:{text_hash}"


    # --------------------------------------------------------
    # Check Redis cache
    # --------------------------------------------------------

    if redis_client is not None:

        try:

            cached_result = redis_client.get(cache_key)

            if cached_result is not None:

                result = json.loads(cached_result)

                return {
                    "prediction": result["prediction"],
                    "cache_hit": True
                }

        except Exception as e:

            print(f"Redis read error: {e}")


    # --------------------------------------------------------
    # Cache MISS → Run ML model
    # --------------------------------------------------------

    features = vectorizer.transform([text])

    prediction = model.predict(features)[0]


    # --------------------------------------------------------
    # Store prediction in Redis
    # --------------------------------------------------------

    result = {
        "prediction": prediction
    }

    if redis_client is not None:

        try:

            redis_client.set(
                cache_key,
                json.dumps(result),
                ex=3600
            )

        except Exception as e:

            print(f"Redis write error: {e}")


    return {
        "prediction": prediction,
        "cache_hit": False
    }


# ============================================================
# PLAGIARISM PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(request: TextRequest):

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not request.text.strip():

        raise HTTPException(
            status_code=400,
            detail="Text cannot be empty."
        )


    # --------------------------------------------------------
    # Measure total inference latency
    # --------------------------------------------------------

    start_time = time.perf_counter()


    # --------------------------------------------------------
    # Prediction with Redis cache
    # --------------------------------------------------------

    result = predict_with_cache(request.text)


    # --------------------------------------------------------
    # Calculate latency
    # --------------------------------------------------------

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000


    # --------------------------------------------------------
    # Return API response
    # --------------------------------------------------------

    return {
        "text": request.text,
        "prediction": result["prediction"],
        "cache_hit": result["cache_hit"],
        "latency_ms": round(latency_ms, 4)
    }


# ============================================================
# CACHE MANAGEMENT
# ============================================================

@app.delete("/cache")
def clear_cache():

    if redis_client is None:

        raise HTTPException(
            status_code=503,
            detail="Redis cache is not available."
        )

    try:

        keys = redis_client.keys(
            "plagiarism_prediction:*"
        )

        if keys:
            redis_client.delete(*keys)

        return {
            "message": "Prediction cache cleared successfully.",
            "keys_deleted": len(keys)
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Unable to clear cache: {e}"
        )


# ============================================================
# APPLICATION STARTUP
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )