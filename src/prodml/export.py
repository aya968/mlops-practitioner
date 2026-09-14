import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from prodml.config import settings
from prodml.predict import DurationPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prodml.export")

def export_and_benchmark():
    predictor = DurationPredictor().load()
    model = predictor.model
    dv = predictor.dv
    
    num_features = len(dv.feature_names_)
    initial_type = [('float_input', FloatTensorType([None, num_features]))]

    logger.info(f"Exporting model with {num_features} features to ONNX...")
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    
    onnx_path = Path(settings.model_path).parent / "model.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    logger.info(f"ONNX model successfully saved to: {onnx_path}")

    dummy_dicts = [
        {
            "PULocationID": int(np.random.randint(1, 263)),
            "DOLocationID": int(np.random.randint(1, 263)),
            "trip_distance": float(np.random.uniform(0.5, 30.0)),
        }
        for _ in range(500)
    ]

    # --- Pickle Inference Benchmark ---
    start_time = time.perf_counter()
    X_matrix = dv.transform(dummy_dicts)
    pkl_preds = np.expm1(model.predict(X_matrix))
    pkl_duration = (time.perf_counter() - start_time) * 1000  # ms

    # --- ONNX Inference Benchmark ---
    sess = rt.InferenceSession(str(onnx_path))
    input_name = sess.get_inputs()[0].name
    X_matrix_dense = X_matrix.toarray().astype(np.float32)

    start_time = time.perf_counter()
    onnx_raw_preds = sess.run(None, {input_name: X_matrix_dense})[0].flatten()
    onnx_preds = np.expm1(onnx_raw_preds)
    onnx_duration = (time.perf_counter() - start_time) * 1000  # ms

    np.testing.assert_allclose(pkl_preds, onnx_preds, atol=1e-4)
    logger.info(" Parity test PASSED! Pickle and ONNX predictions match within 1e-4 tolerance.")

    print("\n" + "="*50)
    print("         BENCHMARK RESULTS (500 Validation Rows)      ")
    print("="*50)
    print(f"Pickle Total Latency: {pkl_duration:.4f} ms")
    print(f"ONNX Total Latency:   {onnx_duration:.4f} ms")
    print(f"Mean Latency (Pickle): {pkl_duration / 500:.6f} ms/row")
    print(f"Mean Latency (ONNX):   {onnx_duration / 500:.6f} ms/row")
    print("="*50 + "\n")

if __name__ == "__main__":
    export_and_benchmark()