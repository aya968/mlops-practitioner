import pytest
from prodml.predict import DurationPredictor


def test_prediction_output_type_and_range(trained_model: DurationPredictor, sample_features):
    prediction = trained_model.predict_one(sample_features)
    assert isinstance(prediction, float)
    assert 0.0 < prediction < 300.0, "Trip duration falls outside sane limits"


def test_prediction_determinism(trained_model: DurationPredictor, sample_features):
    pred_1 = trained_model.predict_one(sample_features)
    pred_2 = trained_model.predict_one(sample_features)
    assert pred_1 == pytest.approx(pred_2, rel=1e-6)