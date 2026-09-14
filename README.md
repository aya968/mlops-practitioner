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