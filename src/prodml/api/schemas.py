from typing import List
from pydantic import BaseModel, Field, ConfigDict


class PredictionRequest(BaseModel):
    PULocationID: int = Field(..., gt=0, description="Pickup Location ID")
    DOLocationID: int = Field(..., gt=0, description="Dropoff Location ID")
    trip_distance: float = Field(..., gt=0, lt=200, description="Trip distance in miles")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "PULocationID": 100,
                "DOLocationID": 200,
                "trip_distance": 2.5,
            }
        }
    )


class BatchPredictionRequest(BaseModel):
    items: List[PredictionRequest]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "items": [
                    {"PULocationID": 100, "DOLocationID": 200, "trip_distance": 2.5},
                    {"PULocationID": 132, "DOLocationID": 138, "trip_distance": 12.4},
                ]
            }
        }
    )


class PredictionResponse(BaseModel):
    prediction: float
    model_version: str
    correlation_id: str
    latency_ms: float


class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    model_version: str
    correlation_id: str
    latency_ms: float