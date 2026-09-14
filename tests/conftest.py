import pytest
from fastapi.testclient import TestClient
from prodml.predict import DurationPredictor
from prodml.api.main import app


@pytest.fixture
def sample_features():
    return {
        "PULocationID": 100,
        "DOLocationID": 200,
        "trip_distance": 2.5
    }


@pytest.fixture(scope="session")
def trained_model():
    predictor = DurationPredictor()
    predictor.load()
    return predictor


@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client