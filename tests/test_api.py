from unittest.mock import patch
from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_happy_path_with_mock(client: TestClient, sample_features):
    with patch("prodml.api.main.predictor.predict_one", return_value=12.5):
        response = client.post("/predict", json=sample_features)
        assert response.status_code == 200
        
        data = response.json()
        assert data["prediction"] == 12.5
        assert "correlation_id" in data
        assert "latency_ms" in data


def test_predict_invalid_payload_returns_422(client: TestClient):
    bad_payload = {
        "PULocationID": 100,
        "DOLocationID": 200,
        "trip_distance": -5.0   
    }
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422
    assert "detail" in response.json()

def test_metadata_endpoint(client: TestClient):
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "framework" in data
    assert "features" in data


def test_predict_batch_endpoint(client: TestClient, sample_features):
    with patch("prodml.api.main.predictor.predict_batch", return_value=[12.5, 15.0]):
        batch_payload = {"items": [sample_features, sample_features]}
        response = client.post("/predict/batch", json=batch_payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 2
        