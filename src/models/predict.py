# src/models/predict.py

import os
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
import hydra
from datetime import timedelta

# Import feature builders
from src.data.make_features import (
    preprocess,
    create_time_features,
    create_lag_features,
)


# =====================================================
# FUNÇÕES DE APOIO
# =====================================================

def load_xgb_model(path):
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model


def build_features(df):
    df = preprocess(df)
    df = create_time_features(df)
    df = create_lag_features(df, target_col="PJME_MW", lags=[364, 728, 1092])
    return df


# =====================================================
# 1. PREDICT TEST
# =====================================================

def predict_test(df, model, split_date, features, target, output_path):
    df_train = df[df.index < split_date].copy()
    df_test = df[df.index >= split_date].copy()

    df_train = df_train.dropna(subset=features + [target])
    df_test = df_test.dropna(subset=features + [target])

    X_test = df_test[features]
    y_test = df_test[target]

    print("\n Running TEST prediction...")
    pred = model.predict(X_test)

    df_test["prediction"] = pred
    rmse = np.sqrt(np.mean((y_test - pred) ** 2))

    print(f" Test RMSE: {rmse:,.4f}")

    os.makedirs(output_path.parent, exist_ok=True)
    df_test[[target, "prediction"]].to_csv(output_path)

    print(f" Test predictions saved to: {output_path}")

    return df_test, rmse


# =====================================================
# 2. PREDICT FUTURE
# =====================================================

def predict_future(df, model, horizon_hours, features, output_path):
    last_date = df.index.max()

    future_index = pd.date_range(
        start=last_date + timedelta(hours=1),
        periods=horizon_hours,
        freq="1H"
    )

    future_df = pd.DataFrame(index=future_index)
    df_all = pd.concat([df, future_df])

    df_all = create_time_features(df_all)
    df_all = create_lag_features(df_all, target_col="PJME_MW", lags=[364, 728, 1092])

    df_future = df_all.loc[future_index].copy()
    df_future["prediction"] = model.predict(df_future[features])

    os.makedirs(output_path.parent, exist_ok=True)
    df_future[["prediction"]].to_csv(output_path)

    print(f" Future predictions saved to: {output_path}")

    return df_future


# =====================================================
# HYDRA MAIN
# =====================================================

@hydra.main(config_path="../../configs/predict", config_name="default", version_base=None)
def main(cfg: DictConfig):

    ROOT = Path(get_original_cwd())

    model_path = ROOT / cfg.predict.model_path
    features_path = ROOT / cfg.predict.features_path
    test_output_path = ROOT / cfg.predict.test_output_path
    future_output_path = ROOT / cfg.predict.future_output_path

    print(f"\n Loading processed dataset from: {features_path}")
    df = pd.read_csv(
        features_path,
        parse_dates=[cfg.predict.datetime_column],
        index_col=cfg.predict.datetime_column
    )
    df = df.sort_index()

    features = [
        "dayofyear", "hour", "dayofweek", "quarter", "month", "year",
        "lag_364d", "lag_728d", "lag_1092d"
    ]

    print(f" Loading model from: {model_path}")
    model = load_xgb_model(model_path)

    print("\n Running TEST prediction")
    predict_test(
        df=df,
        model=model,
        split_date="2014-01-01",
        features=features,
        target=cfg.predict.target_column,
        output_path=test_output_path
    )

    print("\n Running FUTURE prediction")
    predict_future(
        df=df,
        model=model,
        horizon_hours=cfg.predict.forecast_horizon,
        features=features,
        output_path=future_output_path
    )


if __name__ == "__main__":
    main()
