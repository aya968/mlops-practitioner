from pathlib import Path
from typing import Tuple, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from prodml.config import settings
from prodml.features import engineer_features


def load_and_clean_data(file_path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_parquet(file_path)
    df_cleaned = df.copy()

    if "ehail_fee" in df_cleaned.columns:
        df_cleaned.drop(columns=["ehail_fee"], inplace=True)

    # Filtering trip distance and total amount
    df_cleaned = df_cleaned[
        (df_cleaned["trip_distance"] > 0) & (df_cleaned["trip_distance"] < 100)
    ].copy()
    df_cleaned = df_cleaned[df_cleaned["total_amount"] > 0].copy()

    # Passenger count filter
    df_cleaned = df_cleaned[
        (df_cleaned["passenger_count"] >= 1)
        & (df_cleaned["passenger_count"] <= 6)
    ].copy()

    # Duration calculation and filter
    df_cleaned["duration"] = (
        df_cleaned["lpep_dropoff_datetime"] - df_cleaned["lpep_pickup_datetime"]
    ).dt.total_seconds() / 60
    df_cleaned = df_cleaned[
        (df_cleaned["duration"] >= 1) & (df_cleaned["duration"] <= 60)
    ].copy()

    return df_cleaned


def get_train_val_data(
    file_path: Union[str, Path] = settings.data_dir,
) -> Tuple[list, list, pd.Series, pd.Series]:
    df_cleaned = load_and_clean_data(file_path)
    df_features = engineer_features(df_cleaned)

    feature_cols = settings.categorical_features + settings.numerical_features
    dicts = df_features[feature_cols].to_dict(orient="records")
    y = df_features["log_duration"].values

    return train_test_split(
        dicts,
        y,
        test_size=settings.test_size,
        random_state=settings.random_state,
    )
    