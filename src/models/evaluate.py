# src/models/evaluate.py

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow

from hydra.utils import get_original_cwd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =======================================
#  METRICS
# =======================================
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"mae": mae, "rmse": rmse}


# =======================================
#  MERGE FILES
# =======================================
def merge_actual_and_test_pred(actual_path, pred_path, datetime_col, target_col, pred_col):

    df_actual = pd.read_csv(
        actual_path, parse_dates=[datetime_col], index_col=datetime_col
    ).sort_index()

    df_pred = pd.read_csv(
        pred_path, parse_dates=[datetime_col], index_col=datetime_col
    ).sort_index()

    df_merged = df_actual[[target_col]].join(df_pred[[pred_col]], how="inner").dropna()

    return df_actual, df_pred, df_merged


# =======================================
# LOAD FUTURE PREDICTIONS
# =======================================
def load_future_predictions(path, datetime_col, pred_col):

    if not os.path.exists(path):
        return None

    df_future = pd.read_csv(
        path, parse_dates=[datetime_col], index_col=datetime_col
    ).sort_index()

    return df_future


# =======================================
# PLOTS
# =======================================
def plot_actual_vs_test_pred(df, target_col, pred_col, path):
    plt.figure(figsize=(18, 6))
    plt.plot(df.index, df[target_col], label="Actual")
    plt.plot(df.index, df[pred_col], label="Test Prediction")
    plt.title("Actual vs Test Predictions")
    plt.xlabel("Datetime")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_full_forecast(df_actual, df_test, df_future, target_col, pred_col, path):
    plt.figure(figsize=(20, 7))

    if df_actual is not None:
        plt.plot(df_actual.index, df_actual[target_col], label="Actual")

    if df_test is not None:
        plt.plot(df_test.index, df_test[pred_col], label="Test Prediction")

    if df_future is not None:
        plt.plot(df_future.index, df_future[pred_col], label="Future Forecast")

    plt.title("Full Forecast")
    plt.xlabel("Datetime")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# =======================================
# PIPELINE-FRIENDLY FUNCTION
# =======================================
def run_evaluate(cfg, logger, test_predictions_path, parent_run_id=None):
    """
    Evaluation step used inside full_pipeline.py
    (NO Hydra, NO MLflow.start_run inside here)
    """

    # Ensure active MLflow run exists
    if mlflow.active_run() is None:
        raise RuntimeError(
            "run_evaluate() called without an active MLflow run. "
            "This function must be executed inside an MLflow.start_run() block."
        )

    # Recover original project root
    ROOT = Path(get_original_cwd())

    eval_cfg = cfg.eval

    # Paths (relative to project root)
    actual_path = ROOT / eval_cfg.actual_path
    future_pred_path = ROOT / eval_cfg.future_pred_path

    output_dir = ROOT / eval_cfg.output_dir
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    target_col = eval_cfg.target_col
    pred_col = eval_cfg.pred_col
    datetime_col = eval_cfg.datetime_col

    logger.info("Loading evaluation datasets...")

    df_actual, df_test_pred, df_merged = merge_actual_and_test_pred(
        actual_path=actual_path,
        pred_path=test_predictions_path,
        datetime_col=datetime_col,
        target_col=target_col,
        pred_col=pred_col,
    )

    logger.info("Computing metrics...")
    metrics = compute_metrics(df_merged[target_col], df_merged[pred_col])

    logger.info(f"Test MAE  = {metrics['mae']:.2f}")
    logger.info(f"Test RMSE = {metrics['rmse']:.2f}")

    # MLflow — log metrics
    mlflow.log_metric("eval_mae", metrics["mae"])
    mlflow.log_metric("eval_rmse", metrics["rmse"])

    # Save metrics
    pd.DataFrame([metrics]).to_csv(output_dir / "test_metrics.csv", index=False)
    mlflow.log_artifact(str(output_dir / "test_metrics.csv"))

    df_merged.to_csv(tables_dir / "test_evaluation.csv")
    mlflow.log_artifact(str(tables_dir / "test_evaluation.csv"))

    # Load future predictions
    df_future = load_future_predictions(
        path=future_pred_path,
        datetime_col=datetime_col,
        pred_col=pred_col,
    )

    # Plots
    plot_actual_vs_test_pred(
        df=df_merged,
        target_col=target_col,
        pred_col=pred_col,
        path=plots_dir / "actual_vs_test.png",
    )

    mlflow.log_artifact(str(plots_dir / "actual_vs_test.png"))

    plot_full_forecast(
        df_actual=df_actual,
        df_test=df_test_pred,
        df_future=df_future,
        target_col=target_col,
        pred_col=pred_col,
        path=plots_dir / "full_forecast.png",
    )

    mlflow.log_artifact(str(plots_dir / "full_forecast.png"))

    logger.info("Evaluation complete.")
    return metrics
