"""
test_api.py
===========
Test suite for the FastAPI model serving layer.
Uses FastAPI TestClient to validate API endpoints, request validation,
and inference responses against actual saved model artifacts.
"""

import pytest
from fastapi.testclient import TestClient
from app import app, model

client = TestClient(app)


def test_health_check():
    """Test GET /health returns HTTP 200 and expected health metadata."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model"] == "Random Forest"


def test_predict_valid_original_text():
    """Test POST /predict with valid text returns HTTP 200 and valid schema."""
    payload = {
        "text": "Machine learning is a method of teaching computers to learn patterns from data."
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "is_plagiarized" in data
    assert "confidence" in data
    assert data["prediction"] in ["original", "plagiarized"]
    assert isinstance(data["is_plagiarized"], bool)
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_valid_plagiarized_text():
    """Test POST /predict with plagiarized sample text returns HTTP 200."""
    payload = {
        "text": "Natural language processing helps computers understand human language."
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["original", "plagiarized"]


def test_predict_empty_text():
    """Test POST /predict with empty text is rejected with HTTP 422."""
    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_whitespace_text():
    """Test POST /predict with whitespace-only text is rejected with HTTP 422."""
    payload = {"text": "   \n\t   "}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_missing_text():
    """Test POST /predict with missing text field is rejected with HTTP 422."""
    payload = {}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_data_type():
    """Test POST /predict with integer text field is rejected with HTTP 422."""
    payload = {"text": 12345}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_exceeds_max_length():
    """Test POST /predict with text exceeding 10000 characters is rejected with HTTP 422."""
    payload = {"text": "a" * 10001}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
