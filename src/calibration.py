import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def make_models(
    ridge_alpha=1.0,
    lasso_alpha=1e-3,
    enet_alpha=1e-3,
    enet_l1_ratio=0.5,
):
    models = {
        "OLS": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),
        f"Ridge({ridge_alpha})": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Ridge(alpha=ridge_alpha))
        ]),
        f"Lasso({lasso_alpha})": Pipeline([
            ("scaler", StandardScaler()),
            ("model", Lasso(alpha=lasso_alpha, max_iter=5000))
        ]),
        f"ElasticNet({enet_alpha},{enet_l1_ratio})": Pipeline([
            ("scaler", StandardScaler()),
            ("model", ElasticNet(alpha=enet_alpha, l1_ratio=enet_l1_ratio, max_iter=5000))
        ]),
        "Huber": Pipeline([
            ("scaler", StandardScaler()),
            ("model", HuberRegressor())
        ]),
    }
    return models


def compare_models(X_train, y_train, X_test, y_test,
                   ridge_alpha=1.0,
                   lasso_alpha=1e-3,
                   enet_alpha=1e-3,
                   enet_l1_ratio=0.5):

    models = make_models(ridge_alpha, lasso_alpha, enet_alpha, enet_l1_ratio)

    rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, pred)
        rmse = np.sqrt(mean_squared_error(y_test, pred))

        rows.append({
            "model": name,
            "MAE": mae,
            "RMSE": rmse
        })

    return pd.DataFrame(rows).sort_values("RMSE")