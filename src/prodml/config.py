from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = BASE_DIR / ".env", env_file_encoding = "utf-8", extra = "ignore"
    )

    # Paths
    data_dir: Path = BASE_DIR / "data"/ "green_tripdata_2023-01.parquet"
    model_dir: Path = BASE_DIR / "models"
    model_path: Path = model_dir / "baseline.pkl"

    # Data Splitting
    test_size: float = 0.2  
    random_state: int = 42

    # Feature Engineering
    categorical_features: list[str] = ["PU_DO"]
    numerical_features: list[str] = ["log_trip_distance"]

    # Model Parameters
    rf_n_estimators: int = 100
    rf_max_depth: int = 10

settings = Settings()
