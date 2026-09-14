from pathlib import Path
import numpy as np
import onnxruntime as rt
import pytest
from prodml.config import settings
from prodml.predict import DurationPredictor


def test_pickle_onnx_parity(trained_model: DurationPredictor):
    onnx_path = Path(settings.model_path).parent / "model.onnx"
    if not onnx_path.exists():
        pytest.skip("model.onnx not found, skipping parity test.")

    sample_dict = {"PULocationID": 100, "DOLocationID": 200, "trip_distance": 5.0}
    
    # Pickle Prediction
    X_mat = trained_model.dv.transform([sample_dict])
    pkl_pred = np.expm1(trained_model.model.predict(X_mat))[0]

    # ONNX Prediction
    sess = rt.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    X_dense = X_mat.toarray().astype(np.float32)
    onnx_raw = sess.run(None, {input_name: X_dense})[0].flatten()
    onnx_pred = np.expm1(onnx_raw)[0]

    np.testing.assert_allclose(pkl_pred, onnx_pred, atol=1e-4)