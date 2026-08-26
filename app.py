"""
app.py
======
Phase 3 - Task 1: Model Serving Layer API

A FastAPI-based web service that serves the trained Random Forest plagiarism detection
model for real-time inferences.

Endpoints:
----------
- GET  /health   : Health check endpoint returning service status and model metadata.
- POST /predict  : Plagiarism prediction endpoint accepting submission text and returning
                   classification label, plagiarism boolean flag, and confidence score.
"""

import os
from contextlib import asynccontextmanager
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, field_validator

# Paths to saved model and vectorizer artifacts
MODEL_PATH = "random_forest_model.pkl"
ALT_MODEL_PATH = "best_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

# Global artifact storage
model = None
vectorizer = None


def load_artifacts():
    """Load the trained model and TF-IDF vectorizer artifacts."""
    global model, vectorizer

    # Load model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
    elif os.path.exists(ALT_MODEL_PATH):
        model = joblib.load(ALT_MODEL_PATH)
    else:
        model = None

    # Load vectorizer
    if os.path.exists(VECTORIZER_PATH):
        vectorizer = joblib.load(VECTORIZER_PATH)
    else:
        vectorizer = None


# Load artifacts at module load time so TestClient and Uvicorn have immediate access
load_artifacts()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to handle application startup and shutdown events."""
    load_artifacts()
    yield


app = FastAPI(
    title="AI-Powered Academic Integrity Checker API",
    description=(
        "Model serving API for detecting potential plagiarism in academic submissions "
        "using a trained Random Forest classifier and TF-IDF feature extraction."
    ),
    version="1.0.0",
    lifespan=lifespan
)


class HealthResponse(BaseModel):
    """Schema for service health check response."""
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    model: str = Field(..., json_schema_extra={"example": "Random Forest"})


class PredictionRequest(BaseModel):
    """Schema for incoming plagiarism prediction requests."""
    text: str = Field(
        ...,
        description="The student submission text to analyze for plagiarism.",
        json_schema_extra={"example": "Artificial intelligence helps computers analyze large amounts of information."}
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("Input text must be a string.")
        if not value:
            raise ValueError("Input text must not be empty.")
        if not value.strip():
            raise ValueError("Input text must not contain only whitespace.")
        if len(value) > 10000:
            raise ValueError("Input text exceeds maximum allowed length of 10000 characters.")
        return value


class PredictionResponse(BaseModel):
    """Schema for structured plagiarism prediction responses."""
    prediction: str = Field(..., description="Predicted class label ('original' or 'plagiarized').", json_schema_extra={"example": "original"})
    is_plagiarized: bool = Field(..., description="Boolean flag indicating plagiarism status.", json_schema_extra={"example": False})
    confidence: float = Field(..., description="Probability confidence score of the predicted label.", json_schema_extra={"example": 0.82})


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Service Health Check",
    tags=["System"]
)
def health_check():
    """Return health status and model metadata."""
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or vectorizer artifacts are not loaded."
        )
    return HealthResponse(
        status="healthy",
        model="Random Forest"
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict Academic Text Plagiarism",
    tags=["Inference"]
)
def predict_plagiarism(request: PredictionRequest):
    """
    Analyze student submission text and predict plagiarism probability.

    - Preprocesses text using fitted TF-IDF vectorizer.
    - Predicts class label using trained Random Forest model.
    - Calculates confidence probability score.
    """
    if model is None or vectorizer is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model or vectorizer artifacts are not loaded."
        )

    try:
        # Preprocess text with TF-IDF vectorizer
        features = vectorizer.transform([request.text])

        # Predict label
        predicted_label = str(model.predict(features)[0])

        # Calculate confidence / probability score
        confidence = 1.0
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]
            classes = list(model.classes_)
            if predicted_label in classes:
                label_idx = classes.index(predicted_label)
                confidence = float(probabilities[label_idx])
            else:
                confidence = float(np.max(probabilities))

        is_plagiarized = bool(predicted_label.lower() == "plagiarized")

        return PredictionResponse(
            prediction=predicted_label,
            is_plagiarized=is_plagiarized,
            confidence=round(confidence, 4)
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}"
        )
