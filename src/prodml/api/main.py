import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from prodml.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionRequest,
    PredictionResponse,
)
from prodml.config import settings
from prodml.logging_conf import correlation_id_ctx, setup_logging
from prodml.predict import DurationPredictor

setup_logging()
logger = logging.getLogger("prodml.api")

predictor = DurationPredictor()


# Lifespan Context Manager لتحميل الموديل مرة واحدة عند تشغيل الـ App
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model artifact into memory during app startup...")
    predictor.load()
    yield
    logger.info("Shutting down API service...")


app = FastAPI(title="NYC Taxi Duration Predictor", lifespan=lifespan)


# Middleware لتمرير الـ correlation_id
@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    token = correlation_id_ctx.set(req_id)

    logger.info(f"Incoming request: {request.method} {request.url.path}")

    response = await call_next(request)

    response.headers["X-Request-ID"] = req_id
    correlation_id_ctx.reset(token)

    return response


# Exception Handler لخطأ Validation (422) مع رسالة نظيفة
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": "Input validation failed. Please check field constraints.",
            "errors": exc.errors(),
        },
    )


# Exception Handler للأخطاء غير المتوقعة (500) بدون تسريب الـ Traceback
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled server error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected server error occurred."},
    )


# --- ENDPOINTS ---


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    if not predictor.is_loaded:
        return JSONResponse(
            status_code=status.HTTP_530_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "reason": "Model object not loaded in memory"},
        )
    return {"status": "healthy", "model_loaded": True}


@app.get("/metadata")
async def get_metadata():
    return {
        "model_version": getattr(settings, "model_version", "1.0.0"),
        "training_date": "2026-09-01",
        "framework": "scikit-learn / ONNX Runtime",
        "features": ["PULocationID", "DOLocationID", "trip_distance"],
        "artifact_hash": "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(payload: PredictionRequest):
    start_time = time.perf_counter()
    input_data = payload.model_dump()

    duration = predictor.predict_one(input_data)
    latency = (time.perf_counter() - start_time) * 1000

    return PredictionResponse(
        prediction=duration,
        model_version=getattr(settings, "model_version", "1.0.0"),
        correlation_id=correlation_id_ctx.get() or "",
        latency_ms=round(latency, 3),
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(payload: BatchPredictionRequest):
    start_time = time.perf_counter()
    input_list = [item.model_dump() for item in payload.items]

    durations = predictor.predict_batch(input_list)
    latency = (time.perf_counter() - start_time) * 1000

    return BatchPredictionResponse(
        predictions=durations,
        model_version=getattr(settings, "model_version", "1.0.0"),
        correlation_id=correlation_id_ctx.get() or "",
        latency_ms=round(latency, 3),
    )