import pytest
from prodml.predict import DurationPredictor


@pytest.mark.parametrize("invalid_features, expected_behavior", [
    ({"PULocationID": 9999, "DOLocationID": 8888, "trip_distance": 1.5}, "unseen_pair"),
    ({"PULocationID": 100, "DOLocationID": 200, "trip_distance": 0.0}, "zero_distance"),
    ({"PULocationID": 100, "DOLocationID": 200, "trip_distance": 150.0}, "extreme_distance"),
])
def test_feature_edge_cases(trained_model: DurationPredictor, invalid_features, expected_behavior):
    pred = trained_model.predict_one(invalid_features)
    assert isinstance(pred, float)
    assert pred >= 0.0, "Prediction duration should not be negative"