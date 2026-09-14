# Module 1 Report: Baseline Model Performance

## Validation Metrics
* **Validation RMSE:** {rmse:.2f} minutes
* **Validation MAE:** {mae:.2f} minutes

---

## Model Overview
* **Model Architecture:** Linear Regression (Baseline)
* **Features Used:**
  * `PU_DO` (Categorical interaction: Pickup Location + Dropoff Location)
  * `log_trip_distance` (Numerical: Log-transformed trip distance)
* **Preprocessing:** `DictVectorizer(sparse=True)`
* **Target Variable:** `log_duration` (Evaluated on `np.expm1` inverse transformation in real minutes)

## Benchmark Comparison
* **Baseline LinearRegression RMSE:** 6.54 minutes
* **RandomForestRegressor (Iterative Experiment) RMSE:** 4.79 minutes
