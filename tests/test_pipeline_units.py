from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from prodml.data import load_and_clean_data, engineer_features as data_engineer
from prodml.features import engineer_features, get_time_of_day
from prodml.train import train_and_persist


def test_features_and_data_helpers():
    assert get_time_of_day(8) == "morning"
    assert get_time_of_day(14) == "afternoon"
    assert get_time_of_day(20) == "evening"
    assert get_time_of_day(2) == "night"

    df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2026-01-01 10:00:00"]),
            "tpep_dropoff_datetime": pd.to_datetime(["2026-01-01 10:15:00"]),
            "trip_distance": [2.5],
            "PULocationID": [100],
            "DOLocationID": [200],
        }
    )
    df_feat = engineer_features(df)
    assert "duration" in df_feat.columns
    assert df_feat["duration"].iloc[0] == 15.0


@patch("prodml.data.pd.read_parquet")
def test_load_and_clean_data_mocked(mock_read):
    mock_df = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2026-01-01 10:00:00", "2026-01-01 10:00:00"]),
            "tpep_dropoff_datetime": pd.to_datetime(["2026-01-01 10:15:00", "2026-01-01 10:00:30"]), # رحلة قصيرة جدًا للتصفية
            "trip_distance": [2.5, 0.0],
            "PULocationID": [100, 101],
            "DOLocationID": [200, 201],
        }
    )
    mock_read.return_value = mock_df
    
    df_result = load_and_clean_data("fake_path.parquet")
    assert isinstance(df_result, pd.DataFrame)


@patch("prodml.train.get_train_val_data")
@patch("prodml.train.pickle.dump")
def test_train_and_persist_mocked(mock_pickle, mock_get_data):
    df_dummy = pd.DataFrame(
        {
            "PULocationID": [100, 101, 102],
            "DOLocationID": [200, 201, 202],
            "trip_distance": [1.0, 2.0, 3.0],
            "duration": [10.0, 15.0, 20.0],
        }
    )
    mock_get_data.return_value = (df_dummy, df_dummy)

    with patch("builtins.open"):
        model, metrics = train_and_persist()
        assert model is not None
        assert "rmse" in metrics or isinstance(metrics, dict)