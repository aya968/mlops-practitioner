import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

from prodml.config import settings
from prodml.data import get_train_val_data


def train_and_persist() -> None:
    X_train_dicts, X_val_dicts, y_train_log, y_val_log = get_train_val_data()

    dv = DictVectorizer(sparse=True)
    X_train = dv.fit_transform(X_train_dicts)
    X_val = dv.transform(X_val_dicts)

    # Model 1: LinearRegression
    lr = LinearRegression()
    lr.fit(X_train, y_train_log)

    y_pred_lr_log = lr.predict(X_val)
    y_pred_lr_real = np.expm1(y_pred_lr_log)
    y_val_real = np.expm1(y_val_log)

    rmse_lr = np.sqrt(mean_squared_error(y_val_real, y_pred_lr_real))
    mae_lr = mean_absolute_error(y_val_real, y_pred_lr_real)
    print(f"LinearRegression Validation RMSE: {rmse_lr:.2f} | MAE: {mae_lr:.2f}")

    # Model 2: RandomForestRegressor
    rf = RandomForestRegressor(
        n_estimators=settings.rf_n_estimators,
        max_depth=settings.rf_max_depth,
        random_state=settings.random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train_log)

    y_pred_rf_log = rf.predict(X_val)
    y_pred_rf_real = np.expm1(y_pred_rf_log)

    rmse_rf = np.sqrt(mean_squared_error(y_val_real, y_pred_rf_real))
    mae_rf = mean_absolute_error(y_val_real, y_pred_rf_real)
    print(f"RandomForestRegressor Validation RMSE: {rmse_rf:.2f} | MAE: {mae_rf:.2f}")

    # Save baseline artifact (DictVectorizer + LinearRegression) to settings.model_path
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    with open(settings.model_path, "wb") as f:
        pickle.dump((dv, lr), f)

    print(f"Model saved successfully to {settings.model_path}")


if __name__ == "__main__":
    train_and_persist()