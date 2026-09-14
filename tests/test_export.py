from pathlib import Path
import pytest
from prodml.config import settings
from prodml.export import export_and_benchmark


def test_export_and_benchmark_execution():
    if not Path(settings.model_path).exists():
        pytest.skip("Base pickle model does not exist, skipping export test.")

    export_and_benchmark()

    onnx_path = Path(settings.model_path).parent / "model.onnx"
    assert onnx_path.exists(), "ONNX export failed to create model.onnx"