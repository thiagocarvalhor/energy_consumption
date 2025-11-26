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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from src.utils.logger import get_logger


# ============================================================
#  FUNCTION: run_train  (used by full_pipeline.py)
# ============================================================
def run_train(cfg: DictConfig, logger, ROOT: Path = None):

    # ROOT only exists when called from CLI; pipeline passes manually
    if ROOT is None:
        ROOT = Path(get_original_cwd())

    load_dotenv()

    train_cfg = cfg.train
    mlflow_cfg = cfg.mlflow
    seed = cfg.seed

    # Paths
    data_path = ROOT / train_cfg.path.processed_path
    model_output_dir = ROOT / train_cfg.path.output_dir
    os.makedirs(model_output_dir, exist_ok=True)

    logger.info(f"Loading processed dataset: {data_path}")

    df = pd.read_csv(
        data_path,
        parse_dates=[train_cfg.datetime_column],
        index_col=train_cfg.datetime_column,
    ).sort_index()

    # Feature selection
    FEATURES = list(train_cfg.features.columns)
    TARGET = train_cfg.target
    train_end = train_cfg.train_end

    df_train = df[df.index < train_end].copy()
    df_train = df_train.dropna(subset=FEATURES + [TARGET])

    logger.info(f"Training samples: {len(df_train):,}")

    # --------- MLflow: use the ACTIVE run ---------
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError(
            "ERROR: run_train() was called without an active MLflow run. "
            "This function must be executed inside a MLflow.start_run() block."
        )

    run_id = active_run.info.run_id
    logger.info(f"Logging into active MLflow run_id = {run_id}")

    # --------- Log parameters ---------
    mlflow.log_param("train_end", train_end)
    mlflow.log_param("seed", seed)
    mlflow.log_param("features", ",".join(FEATURES))
    mlflow.log_params(train_cfg.model)

    # TimeSeries CV
    tss = TimeSeriesSplit(
        n_splits=train_cfg.cv.n_splits,
        test_size=train_cfg.cv.test_size,
        gap=train_cfg.cv.gap,
    )

    scores = []
    model = None

    # --------- Training Loop ---------
    for fold, (train_idx, val_idx) in enumerate(tss.split(df_train), start=1):

        logger.info(f"===== FOLD {fold} =====")

        train_fold = df_train.iloc[train_idx]
        val_fold = df_train.iloc[val_idx]

        X_train, y_train = train_fold[FEATURES], train_fold[TARGET]
        X_val, y_val = val_fold[FEATURES], val_fold[TARGET]

        model = xgb.XGBRegressor(**train_cfg.model, random_state=seed)

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        preds = model.predict(X_val)
        rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        scores.append(rmse)

        mlflow.log_metric(f"rmse_fold_{fold}", rmse)
        logger.info(f"Fold {fold} RMSE: {rmse:,.4f}")

    # --------- Final Metrics ---------
    mean_rmse = float(np.mean(scores))
    mlflow.log_metric("rmse_mean", mean_rmse)
    logger.info(f"Mean RMSE: {mean_rmse:,.4f}")

    # --------- Save Model ---------
    model_name = f"{train_cfg.model.name}.json"
    model_path = model_output_dir / model_name

    model.save_model(model_path)
    mlflow.log_artifact(str(model_path))

    logger.info(f"Model saved at: {model_path}")

    # Return model path for prediction step
    return run_id, model_path


# ============================================================
#  HYDRA WRAPPER — execution via CLI
# ============================================================
@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    ROOT = Path(get_original_cwd())

    logger = get_logger(
        name="energy_logger",
        log_file=ROOT / "logs/train.log",
        level="INFO",
    )

    #  Running stand-alone → here YES we open MLflow run
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    with mlflow.start_run(run_name="train_cli"):

        run_train(cfg, logger, ROOT)


if __name__ == "__main__":
    main()
