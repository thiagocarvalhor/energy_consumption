# src/models/evaluate.py

import os
from pathlib import Path
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ===============================================
# 1. METRICS
# ===============================================
def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"mae": mae, "rmse": rmse}


# ===============================================
# 2. MERGE HISTORICAL + TEST
# ===============================================
def merge_actual_and_test_pred(actual_path, pred_path, datetime_col, target_col, pred_col):

    df_actual = pd.read_csv(
        actual_path, parse_dates=[datetime_col], index_col=datetime_col
    ).sort_index()

    df_pred = pd.read_csv(
        pred_path, parse_dates=[datetime_col], index_col=datetime_col
    ).sort_index()

    df_merged = df_actual[[target_col]].join(df_pred[[pred_col]], how="inner")
    df_merged = df_merged.dropna()

    return df_actual, df_pred, df_merged


# ===============================================
# 3. LOAD FUTURE
# ===============================================
def load_future_predictions(path, datetime_col, pred_col):

    if not os.path.exists(path):
        print(" Future prediction file not found. Skipping future forecast plot.")
        return None

    df_future = pd.read_csv(
        path, parse_dates=[datetime_col], index_col=datetime_col
    ).sort_index()

    if pred_col not in df_future.columns:
        raise ValueError(f"Future prediction file missing column '{pred_col}'")

    return df_future


# ===============================================
# 4. PLOTS
# ===============================================
def plot_actual_vs_test_pred(df_merged, target_col, pred_col, path):
    plt.figure(figsize=(18, 6))
    plt.plot(df_merged.index, df_merged[target_col], label="Actual")
    plt.plot(df_merged.index, df_merged[pred_col], label="Test Prediction")

    plt.title("Actual vs Test Predictions")
    plt.xlabel("Datetime")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_full_forecast(df_actual, df_test_pred, df_future, target_col, pred_col, path):
    plt.figure(figsize=(20, 7))

    if df_actual is not None:
        plt.plot(df_actual.index, df_actual[target_col], label="Actual")

    if df_test_pred is not None:
        plt.plot(df_test_pred.index, df_test_pred[pred_col], label="Test Prediction")

    if df_future is not None:
        plt.plot(df_future.index, df_future[pred_col], label="Future Forecast")

    plt.title("Full Forecast")
    plt.xlabel("Datetime")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# ===============================================
# 5. HYDRA MAIN
# ===============================================
@hydra.main(config_path="../../configs/eval", config_name="default", version_base=None)
def main(cfg: DictConfig):

    ROOT = Path(get_original_cwd())  # always original project root

    actual_path = ROOT / cfg.eval.actual_path
    test_pred_path = ROOT / cfg.eval.test_pred_path
    future_pred_path = ROOT / cfg.eval.future_pred_path

    output_dir = ROOT / cfg.eval.output_dir
    plots_dir = output_dir / "plots"
    tables_dir = output_dir / "tables"

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    target_col = cfg.eval.target_col
    pred_col = cfg.eval.pred_col
    datetime_col = cfg.eval.datetime_col

    print("\n Loading actual + test prediction files...")
    df_actual, df_test, df_test_merged = merge_actual_and_test_pred(
        actual_path=actual_path,
        pred_path=test_pred_path,
        datetime_col=datetime_col,
        target_col=target_col,
        pred_col=pred_col,
    )

    print("\n Computing metrics...")
    metrics = compute_metrics(
        df_test_merged[target_col], df_test_merged[pred_col]
    )

    print(f" Test MAE : {metrics['mae']:.2f}")
    print(f" Test RMSE: {metrics['rmse']:.2f}")

    pd.DataFrame([metrics]).to_csv(output_dir / "test_metrics.csv", index=False)
    df_test_merged.to_csv(tables_dir / "test_evaluation_table.csv")

    print("\n Loading future predictions...")
    df_future = load_future_predictions(
        path=future_pred_path,
        datetime_col=datetime_col,
        pred_col=pred_col,
    )

    print(" Generating plots...")

    plot_actual_vs_test_pred(
        df_merged=df_test_merged,
        target_col=target_col,
        pred_col=pred_col,
        path=plots_dir / "actual_vs_test.png"
    )

    plot_full_forecast(
        df_actual=df_actual,
        df_test_pred=df_test,
        df_future=df_future,
        target_col=target_col,
        pred_col=pred_col,
        path=plots_dir / "full_forecast.png",
    )

    print("\n Evaluation completed successfully!")


if __name__ == "__main__":
    main()
