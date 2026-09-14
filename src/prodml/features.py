import numpy as np
import pandas as pd


def get_time_of_day(hour: int) -> str:
    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 22:
        return "evening"
    else:
        return "night"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df_features = df.copy()

    # Datetime features
    df_features["pickup_hour"] = df_features["lpep_pickup_datetime"].dt.hour
    df_features["day_of_week"] = df_features["lpep_pickup_datetime"].dt.dayofweek
    df_features["is_weekend"] = df_features["day_of_week"].isin([5, 6]).astype(int)

    # Peak Hours Flag (Morning: 7-9 AM, Evening: 3-6 PM)
    df_features["is_peak_hour"] = df_features["pickup_hour"].apply(
        lambda h: 1 if (7 <= h <= 9) or (15 <= h <= 18) else 0
    )

    # Categorical Route & Location Features
    df_features["PU_DO"] = (
        df_features["PULocationID"].astype(str)
        + "_"
        + df_features["DOLocationID"].astype(str)
    )
    df_features["is_same_location"] = (
        df_features["PULocationID"] == df_features["DOLocationID"]
    ).astype(int)

    df_features["PULocationID"] = df_features["PULocationID"].astype(str)
    df_features["DOLocationID"] = df_features["DOLocationID"].astype(str)

    # Time of Day Feature
    df_features["time_of_day"] = df_features["pickup_hour"].apply(get_time_of_day)

    # Log Transformation
    df_features["log_trip_distance"] = np.log1p(df_features["trip_distance"])
    df_features["log_duration"] = np.log1p(df_features["duration"])

    return df_features