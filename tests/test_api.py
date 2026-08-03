"""
============================================================
File : test_api.py

Description
-----------
Integration unit tests for FastAPI endpoints.

Author:
    Varsha Shukla
============================================================
"""

from io import BytesIO
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from api.app import app
from configs.settings import settings


@pytest.fixture(scope="module")
def client():
    """
    TestClient fixture with app lifespan execution.
    """
    with TestClient(app) as test_client:
        yield test_client


def create_test_image(
    format_name: str = "JPEG", size: tuple[int, int] = (128, 128)
) -> bytes:
    """
    Helper function to generate an in-memory test image payload.
    """
    img = Image.new("RGB", size, color=(255, 0, 0))
    buffer = BytesIO()
    img.save(buffer, format=format_name)
    return buffer.getvalue()


# ============================================================
# Root Endpoint Tests
# ============================================================


def test_root_endpoint(client: TestClient):
    """
    Test GET / endpoint returns 200 and application details.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["application"] == settings.APP_NAME
    assert data["status"] == "running"
    assert "version" in data


# ============================================================
# Health Endpoint Tests
# ============================================================


def test_health_endpoint(client: TestClient):
    """
    Test GET /health endpoint returns 200, healthy status, and uptime.
    """
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime" in data


# ============================================================
# Version Endpoint Tests
# ============================================================


def test_version_endpoint(client: TestClient):
    """
    Test GET /version endpoint returns 200 and model metadata.
    """
    response = client.get("/version")
    assert response.status_code == 200
    data = response.json()
    assert data["api_version"] == settings.APP_VERSION
    assert data["model_version"] == settings.MODEL_VERSION
    assert data["model_name"] == settings.MODEL_NAME


# ============================================================
# Middleware Tests
# ============================================================


def test_request_id_middleware(client: TestClient):
    """
    Test that X-Request-ID header is present in response.
    """
    response = client.get("/")
    assert "x-request-id" in response.headers


# ============================================================
# Prediction Endpoint Tests
# ============================================================


def test_predict_valid_image(client: TestClient):
    """
    Test POST /predict with a valid JPEG image payload.
    """
    image_bytes = create_test_image(format_name="JPEG")
    files = {"file": ("test_dog.jpg", image_bytes, "image/jpeg")}

    response = client.post("/predict", files=files)
    assert response.status_code == 200

    data = response.json()
    assert "predicted_class" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "inference_time_ms" in data
    assert data["inference_time_ms"] >= 0.0
    assert "model_version" in data


def test_predict_invalid_extension(client: TestClient):
    """
    Test POST /predict with unsupported file extension.
    """
    files = {"file": ("test_file.txt", b"hello world", "text/plain")}

    response = client.post("/predict", files=files)
    assert response.status_code == 400
    data = response.json()
    assert "Unsupported image format" in data["detail"]


def test_predict_empty_file(client: TestClient):
    """
    Test POST /predict with 0-byte file payload.
    """
    files = {"file": ("empty.jpg", b"", "image/jpeg")}

    response = client.post("/predict", files=files)
    assert response.status_code == 400
    data = response.json()
    assert "empty" in data["detail"].lower()


def test_predict_corrupted_image(client: TestClient):
    """
    Test POST /predict with corrupted non-image binary data masquerading as .jpg.
    """
    files = {"file": ("corrupt.jpg", b"not a real image payload", "image/jpeg")}

    response = client.post("/predict", files=files)
    assert response.status_code == 400
    data = response.json()
    assert (
        "Invalid image format" in data["detail"] or "invalid" in data["detail"].lower()
    )


def test_predict_file_size_exceeded(client: TestClient):
    """
    Test POST /predict with payload exceeding size limit.
    """
    large_payload = b"0" * (settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024 + 1024)
    files = {"file": ("large.jpg", large_payload, "image/jpeg")}

    response = client.post("/predict", files=files)
    assert response.status_code == 413
    data = response.json()
    assert "exceeds maximum allowed limit" in data["detail"]
