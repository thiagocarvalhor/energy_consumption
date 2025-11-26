# src/models/predict.py

import os
from pathlib import Path
from datetime import timedelta

import hydra
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error

from src.utils.logger import get_logger
from src.data.make_features import create_time_features, create_lag_features


# ============================================================
# Helper: Load XGBoost model
# ============================================================
def load_xgb_model(path: Path) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(path)
    return model


# ============================================================
# Helper: Future prediction
# ============================================================
def run_future_predictions(
    df: pd.DataFrame,
    model: xgb.XGBRegressor,
    features: list[str],
    horizon_hours: int,
    target_col: str,
) -> pd.DataFrame:

    last_date = df.index.max()

    future_index = pd.date_range(
        start=last_date + timedelta(hours=1),
        periods=horizon_hours,
        freq="1H"
    )

    future_df = pd.DataFrame(index=future_index)
    df_all = pd.concat([df, future_df])

    # Features
    df_all = create_time_features(df_all)
    df_all = create_lag_features(df_all, target_col=target_col, lags=[364, 728, 1092])

    df_future = df_all.loc[future_index].copy()
    df_future["prediction"] = model.predict(df_future[features])

    return df_future


# ============================================================
# MAIN FUNCTION FOR PIPELINE (NO MLflow start here)
# ============================================================
def run_predict(cfg: DictConfig, logger, model_path: Path, parent_run_id: str):

    ROOT = Path(get_original_cwd())

    predict_cfg = cfg.predict
    mlflow_cfg = cfg.mlflow

    # Verify that we're inside an active MLflow run
    if mlflow.active_run() is None:
        raise RuntimeError(
            "run_predict() was called without an active MLflow run. "
            "This function must run inside an MLflow.start_run() block."
        )

    # Load data
    features_path = ROOT / predict_cfg.features_path
    df = pd.read_csv(
        features_path,
        parse_dates=[predict_cfg.datetime_column],
        index_col=predict_cfg.datetime_column,
    ).sort_index()

    features = [
        "dayofyear", "hour", "dayofweek", "quarter",
        "month", "year", "lag_364d", "lag_728d", "lag_1092d"
    ]

    # Log artifacts and parameters
    mlflow.log_param("model_used", str(model_path))
    mlflow.set_tag("parent_train_run", parent_run_id)

    # Load model
    logger.info(f"Loading model: {model_path}")
    model = load_xgb_model(model_path)

    # ================================
    # 1 — Test Prediction
    # ================================
    split_date = predict_cfg.split_date

    df_train = df[df.index < split_date].copy()
    df_test = df[df.index >= split_date].copy()

    df_test = df_test.dropna(subset=features + [predict_cfg.target_column])

    X_test = df_test[features]
    y_test = df_test[predict_cfg.target_column]

    preds = model.predict(X_test)
    df_test["prediction"] = preds

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mlflow.log_metric("test_rmse", rmse)

    logger.info(f"Test RMSE: {rmse:,.4f}")

    test_output_path = ROOT / predict_cfg.test_output_path
    os.makedirs(test_output_path.parent, exist_ok=True)
    df_test[[predict_cfg.target_column, "prediction"]].to_csv(test_output_path)

    mlflow.log_artifact(str(test_output_path))

    # ================================
    # 2 — Future Forecast
    # ================================
    df_future = run_future_predictions(
        df=df,
        model=model,
        features=features,
        horizon_hours=predict_cfg.forecast_horizon,
        target_col=predict_cfg.target_column
    )

    future_output_path = ROOT / predict_cfg.future_output_path
    os.makedirs(future_output_path.parent, exist_ok=True)

    df_future.to_csv(future_output_path)
    mlflow.log_artifact(str(future_output_path))

    # ↓ Return paths for evaluate()
    return test_output_path, future_output_path


# ============================================================
# CLI MODE — when running standalone
# ============================================================
@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    ROOT = Path(get_original_cwd())

    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)

    logger = get_logger(
        name="energy_logger_predict",
        log_file=logs_dir / "predict.log",
        level=cfg.logger.level
    )

    logger.info("Running prediction step via CLI...")

    # Recover run_id and model path
    run_file = ROOT / "last_run" / "run_id.txt"
    if not run_file.exists():
        raise FileNotFoundError("Training run_id not found. Run training first.")

    train_run_id = run_file.read_text().strip()
    model_path = ROOT / cfg.predict.model_path

    # For CLI: start a dedicated run
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="predict_cli"):
        run_predict(cfg, logger, model_path=model_path, parent_run_id=train_run_id)


if __name__ == "__main__":
    main()
