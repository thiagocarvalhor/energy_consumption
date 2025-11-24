# src/models/train.py

import os
from pathlib import Path

import hydra
import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

from src.utils.logger import get_logger


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Train an XGBoost model using:
      - Hydra for configuration
      - Custom logger
      - MLflow for experiment tracking
    """

    # ────────────────────────────────────────────────
    # 0. LOAD .ENV (DagsHub or future credentials) + BASIC PATHS
    # ────────────────────────────────────────────────
    load_dotenv()  # loads MLFLOW_TRACKING_USERNAME / PASSWORD from .env

    ROOT = Path(get_original_cwd())

    train_cfg = cfg.train
    mlflow_cfg = cfg.mlflow
    logger_cfg = cfg.logger
    seed = cfg.seed

    # Logger
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(exist_ok=True)
    logger = get_logger(
        name=logger_cfg.name,
        log_file=ROOT / logger_cfg.file_path,
        level=logger_cfg.level,
    )

    logger.info("Starting training pipeline...")

    # ────────────────────────────────────────────────
    # 1. PATHS AND LOADING DATA
    # ────────────────────────────────────────────────
    data_path = ROOT / train_cfg.path.processed_path
    model_output_dir = ROOT / train_cfg.path.output_dir
    os.makedirs(model_output_dir, exist_ok=True)

    logger.info(f"Loading processed dataset from: {data_path}")

    df = pd.read_csv(
        data_path,
        parse_dates=[train_cfg.datetime_column],
        index_col=train_cfg.datetime_column,
    ).sort_index()

    # ────────────────────────────────────────────────
    # 2. FEATURE CONFIGURATION AND TRAINING WINDOW
    # ────────────────────────────────────────────────
    FEATURES = list(train_cfg.features.columns)
    TARGET = train_cfg.target
    train_end = train_cfg.train_end

    logger.info(f"Using features: {FEATURES}")
    logger.info(f"Target: {TARGET}")
    logger.info(f"Training window until: {train_end}")

    df_train = df[df.index < train_end].copy()
    df_train = df_train.dropna(subset=FEATURES + [TARGET])

    logger.info(f"Training samples after filtering/NaN removal: {len(df_train):,}")

    # ────────────────────────────────────────────────
    # 3. MLFLOW SETUP
    # ────────────────────────────────────────────────
    mlflow.set_tracking_uri(mlflow_cfg.tracking_uri)
    mlflow.set_experiment(mlflow_cfg.experiment_name)

    logger.info(f"MLflow tracking URI: {mlflow_cfg.tracking_uri}")
    logger.info(f"MLflow experiment: {mlflow_cfg.experiment_name}")

    # ────────────────────────────────────────────────
    # 4. TIME SERIES SPLIT
    # ────────────────────────────────────────────────
    logger.info(
        f"Configuring TimeSeriesSplit: "
        f"n_splits={train_cfg.cv.n_splits}, "
        f"test_size={train_cfg.cv.test_size}, "
        f"gap={train_cfg.cv.gap}"
    )

    tss = TimeSeriesSplit(
        n_splits=train_cfg.cv.n_splits,
        test_size=train_cfg.cv.test_size,
        gap=train_cfg.cv.gap,
    )

    scores: list[float] = []
    model = None  # last trained model

    # ────────────────────────────────────────────────
    # 5. MLFLOW RUN
    # ────────────────────────────────────────────────
    with mlflow.start_run(run_name=mlflow_cfg.run_name):
        logger.info(f"Starting MLflow run: {mlflow_cfg.run_name}")

        # General parameters
        mlflow.log_param("train_end", train_end)
        mlflow.log_param("seed", seed)
        mlflow.log_param("features", ",".join(FEATURES))

        # Model hyperparameters
        mlflow.log_params({k: v for k, v in train_cfg.model.items()})

        # ────────────────────────────────────────────
        # 6. TRAINING LOOP WITH CROSS-VALIDATION
        # ────────────────────────────────────────────
        for fold, (train_idx, val_idx) in enumerate(tss.split(df_train), start=1):
            logger.info(f"===== FOLD {fold} =====")

            train_fold = df_train.iloc[train_idx]
            val_fold = df_train.iloc[val_idx]

            X_train = train_fold[FEATURES]
            y_train = train_fold[TARGET]

            X_val = val_fold[FEATURES]
            y_val = val_fold[TARGET]

            model = xgb.XGBRegressor(
                **train_cfg.model,
                random_state=seed,
            )

            logger.info(
                f"Training XGBoost model on fold {fold} "
                f"({len(X_train):,} train samples, {len(X_val):,} val samples)..."
            )

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False,
            )

            y_pred = model.predict(X_val)
            rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
            scores.append(rmse)

            logger.info(f"RMSE fold {fold}: {rmse:,.4f}")
            mlflow.log_metric(f"rmse_fold_{fold}", rmse)

        # ────────────────────────────────────────────
        # 7. FINAL METRICS (MEAN OF FOLDS)
        # ────────────────────────────────────────────
        mean_rmse = float(np.mean(scores))
        logger.info("===== FINAL CV RESULTS =====")
        logger.info(f"RMSE by fold: {scores}")
        logger.info(f"Mean RMSE: {mean_rmse:,.4f}")

        mlflow.log_metric("rmse_mean", mean_rmse)

        # ────────────────────────────────────────────
        # 8. SAVE MODEL TO DISK + MLflow ARTIFACT
        # ────────────────────────────────────────────
        model_name = f"{train_cfg.model.name}.json"
        model_path = model_output_dir / model_name

        logger.info(f"Saving trained model to: {model_path}")
        model.save_model(model_path)

        mlflow.log_artifact(str(model_path))

        # Optional: Log train.yaml used
        train_config_path = ROOT / "configs" / "train" / "train.yaml"
        if train_config_path.exists():
            mlflow.log_artifact(str(train_config_path))

        logger.info("Training finished and logged to MLflow.")


if __name__ == "__main__":
    main()
