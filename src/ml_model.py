from __future__ import annotations

import time
import numpy as np
import pandas as pd

from dataclasses import dataclass
from typing import Dict, Tuple, Any

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class MLReport:
    mae: float
    rmse: float
    r2: float
    train_seconds: float


def train_surrogate_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 400,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Tuple[RandomForestRegressor, float]:
    """
    Train a RandomForest surrogate model and return (model, training_time_seconds).
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    t0 = time.time()
    model.fit(X_train, y_train)
    train_seconds = time.time() - t0
    return model, train_seconds


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Predict and compute MAE/RMSE/R2.
    """
    pred = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = float(r2_score(y_test, pred))
    return pred, {"MAE": mae, "RMSE": rmse, "R2": r2}


def benchmark_inference(
    model: Any,
    X: pd.DataFrame,
    n_samples: int = 10000,
    random_state: int = 0,
) -> float:
    """
    Return inference time in seconds for n_samples.
    """
    Xb = X.sample(min(n_samples, len(X)), random_state=random_state)
    t0 = time.time()
    _ = model.predict(Xb)
    return time.time() - t0