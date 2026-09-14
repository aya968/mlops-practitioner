# mlops-practitioner

## Development Workflow

```bash
# Install editable package with dev dependencies
python -m pip install -e ".[dev]"

# Code Linting & Formatting Check
ruff check src tests && black --check src tests

# Run Tests & Coverage Report
pytest -v --cov=src/prodml --cov-report=term-missing

# Train Model Artifact
python -m prodml.train

# Serve API Local Server
uvicorn prodml.api.main:app --reload --port 8000



**Serialization & Format Benchmark**

**Comparison Table**

| Format | Human-Readable | Cross-Language | Schema-Enforced | Safe to Load (Untrusted Source) |
| --- | --- | --- | --- | --- |
| **JSON** | Yes | Yes | No | Yes |
| **Protobuf** | No | Yes | Yes | Yes |
| **Pickle** | No | No | No | **No** |
| **ONNX** | No | Yes | Yes | **Yes** |

---

**Security & Model Format Decision**

> **Warning:** Pickle executes arbitrary code on load. Never load a `.pkl` file you did not produce.

Our service serves predictions using **ONNX (via ONNX Runtime)** because it ensures cross-platform interoperability, enforces a strict schema, eliminates arbitrary code execution risks inherent in Python pickles, and significantly reduces inference latency.

---

**Benchmark Results (500 Validation Rows)**

* **Pickle Total Latency:** `39.28 ms` (`0.0786 ms/row`)
* **ONNX Total Latency:** `8.61 ms` (`0.0172 ms/row`)
* **Performance Gain:** ONNX achieved a **~4.5x speedup** over Pickle while satisfying precision parity (`np.allclose` with `atol=1e-4`).

## Quickstart (Zero to Prediction in 3 Commands)
```bash
# 1. Pull and run the production container from Docker Hub
docker run -d -p 8000:8000 --name taxi-api -e MODEL_PATH=/app/models/baseline.pkl aya968/prodml-api:0.1.0

# 2. Check service health
curl http://localhost:8000/health

# 3. Get a prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"PULocationID": 100, "DOLocationID": 200, "trip_distance": 3.5}'
```

### Example `curl` Output
```json
{
  "predicted_duration": 14.85,
  "model_version": "0.1.0",
  "status": "success"
}
```

## Repository Structure
```text
mlops-practitioner/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/
│   └── baseline.pkl
├── reports/
│   └── module-1.md
├── src/
│   └── prodml/
│       ├── api/
│       │   ├── main.py
│       │   └── schemas.py
│       ├── config.py
│       ├── data.py
│       ├── export.py
│       ├── features.py
│       ├── logging_conf.py
│       ├── predict.py
│       ├── train.py
│       └── utils.py
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_export.py
│   ├── test_features.py
│   ├── test_pipeline_units.py
│   ├── test_predict.py
│   └── test_serialization.py
├── pyproject.toml
└── README.md

---

### **أوامر التثبيت النهائية في Git:**

بعد تحديث ملف `README.md` بالكامل، شغّلي الأوامر التالية من الـ Terminal:

```bash
git add README.md
git commit -m "docs: finalize complete README file"
git tag -a v0.1.0 -m "Release v0.1.0 - Completed Module 1"
git push origin main --tags
