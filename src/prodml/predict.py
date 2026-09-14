import pickle
from pathlib import Path
from typing import Any, Union

import numpy as np

from prodml.config import settings
from prodml.utils import timed


class DurationPredictor:
    def __init__(self, model_path: Union[str, Path] = settings.model_path) -> None:
        self.model_path = Path(model_path)
        self.dv: Any = None
        self.model: Any = None
        self.is_loaded: bool = False

    def load(self) -> "DurationPredictor":
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {self.model_path}. Run training first."
            )

        with open(self.model_path, "rb") as f:
            self.dv, self.model = pickle.load(f)

        self.is_loaded = True
        return self

    @timed
    def predict_one(self, features: dict[str, Any]) -> float:
        if not self.is_loaded:
            self.load()

        X = self.dv.transform([features])
        y_pred_log = self.model.predict(X)
        y_pred_real = np.expm1(y_pred_log)

        return float(y_pred_real[0])

    def predict_batch(self, features_list: list[dict[str, Any]]) -> list[float]:
        if not self.is_loaded:
            self.load()

        X = self.dv.transform(features_list)
        y_pred_log = self.model.predict(X)
        y_pred_real = np.expm1(y_pred_log)

        return y_pred_real.tolist()