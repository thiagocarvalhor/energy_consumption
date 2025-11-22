# src/models/evaluate.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ──────────────────────────────────────────────
# 1. METRICS
# ──────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    """
    Compute standard regression metrics: MAE and RMSE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"mae": mae, "rmse": rmse}


# ──────────────────────────────────────────────
# 2. MERGE ACTUAL + TEST PREDICTIONS
# ──────────────────────────────────────────────

def merge_actual_and_test_pred(
    actual_path="data/processed/energy_features.csv",
    pred_path="data/processed/test_predictions.csv",
    datetime_col="Datetime",
    target_col="PJME_MW",
    pred_col="prediction",
):
    """
    Merge actual historical data with test predictions using Datetime index.
    """

    print("\n Loading actual (historical) dataframe...")
    df_actual = pd.read_csv(
        actual_path,
        parse_dates=[datetime_col],
        index_col=datetime_col,
    )
    df_actual = df_actual.sort_index()

    print(" Loading test prediction dataframe...")
    df_pred = pd.read_csv(
        pred_path,
        parse_dates=[datetime_col],
        index_col=datetime_col,
    )
    df_pred = df_pred.sort_index()

    print(" Merging actual + test predictions...")
    df_merged = df_actual[[target_col]].join(
        df_pred[[pred_col]],
        how="inner",
    )

    df_merged = df_merged.dropna(subset=[target_col, pred_col])
    return df_actual, df_pred, df_merged


# ──────────────────────────────────────────────
# 3. LOAD FUTURE PREDICTIONS
# ──────────────────────────────────────────────

def load_future_predictions(
    future_pred_path="data/processed/future_predictions.csv",
    datetime_col="Datetime",
    pred_col="prediction",
):
    """
    Load future predictions (Datetime must be parsed and set as index).
    """

    if not os.path.exists(future_pred_path):
        print(" No future prediction file found. Skipping future forecast plot.")
        return None

    print(" Loading future predictions...")

    df_future = pd.read_csv(
        future_pred_path,
        parse_dates=[datetime_col],
        index_col=datetime_col,
    )

    # ensure correct ordering
    df_future = df_future.sort_index()

    # ensure prediction column exists
    if pred_col not in df_future.columns:
        raise ValueError(f"Future prediction file missing column '{pred_col}'")

    return df_future



# ──────────────────────────────────────────────
# 4. PLOTTING FUNCTIONS
# ──────────────────────────────────────────────

def plot_actual_vs_test_pred(df_merged, target_col, pred_col, fig_path):
    """
    Plot actual vs test predictions only.
    """
    plt.figure(figsize=(18, 6))

    plt.plot(df_merged.index, df_merged[target_col], label="Actual", linewidth=2)
    plt.plot(df_merged.index, df_merged[pred_col], label="Test Prediction", linewidth=2)

    plt.title("Actual vs Test Predictions")
    plt.xlabel("Datetime")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path)
    plt.close()


def plot_full_forecast(
    df_actual,
    df_test_pred,
    df_future,
    target_col,
    pred_col,
    fig_path,
):
    """
    Plot actual data, test predictions, and future predictions together.
    """

    plt.figure(figsize=(20, 7))

    # actual
    if df_actual is not None and len(df_actual) > 0:
        plt.plot(df_actual.index, df_actual[target_col], label="Actual", linewidth=2)

    # test predictions
    if df_test_pred is not None and len(df_test_pred) > 0:
        plt.plot(df_test_pred.index, df_test_pred[pred_col],
                 label="Test Prediction", linewidth=2)

    # future predictions
    if df_future is not None and len(df_future) > 0:
        plt.plot(df_future.index, df_future[pred_col],
                 label="Future Prediction", linewidth=2)

    plt.title("Full Forecast: Actual vs Test Predictions vs Future Forecast")
    plt.xlabel("Datetime")
    plt.ylabel(target_col)
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path)
    plt.close()


# ──────────────────────────────────────────────
# 5. FULL EVALUATION PIPELINE
# ──────────────────────────────────────────────

def evaluate_predictions(
    actual_path="data/processed/energy_features.csv",
    test_pred_path="data/processed/test_predictions.csv",
    future_pred_path="data/processed/future_predictions.csv",
    output_dir="data/evaluation",
    target_col="PJME_MW",
    pred_col="prediction",
    datetime_col="Datetime",
):
    """
    Full evaluation pipeline:
      • Load actual + test predictions
      • Compute test metrics
      • Load future predictions
      • Plot actual/test/future
      • Save tables and metrics
    """

    print("\n Starting evaluation...")

    # Prepare output directories
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    tables_dir = os.path.join(output_dir, "tables")

    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # ------------------------------------------------------------------------------
    # Load and merge actual + test predictions
    df_actual, df_test, df_test_merged = merge_actual_and_test_pred(
        actual_path=actual_path,
        pred_path=test_pred_path,
        datetime_col=datetime_col,
        target_col=target_col,
        pred_col=pred_col,
    )

    # ------------------------------------------------------------------------------
    # Compute metrics on test predictions
    metrics = compute_metrics(
        df_test_merged[target_col],
        df_test_merged[pred_col]
    )

    print(f" Test MAE : {metrics['mae']:.2f}")
    print(f" Test RMSE: {metrics['rmse']:.2f}")

    # Save merged evaluation table
    table_path = os.path.join(tables_dir, "test_evaluation_table.csv")
    df_test_merged.to_csv(table_path)
    print(f" Test evaluation table saved → {table_path}")

    # Save metrics file
    metrics_path = os.path.join(output_dir, "test_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f" Test metrics saved → {metrics_path}")

    # ------------------------------------------------------------------------------
    # Load future predictions
    df_future = load_future_predictions(
        future_pred_path=future_pred_path,
        datetime_col=datetime_col,
    )

    # ------------------------------------------------------------------------------
    # Plot: actual vs test
    plot_actual_vs_test_pred_path = os.path.join(plots_dir, "actual_vs_test.png")
    plot_actual_vs_test_pred(
        df_test_merged,
        target_col,
        pred_col,
        plot_actual_vs_test_pred_path
    )
    print(f" Plot saved → {plot_actual_vs_test_pred_path}")

    # ------------------------------------------------------------------------------
    # Plot: full forecast (actual + test + future)
    full_forecast_path = os.path.join(plots_dir, "full_forecast.png")

    plot_full_forecast(
        df_actual=df_actual,
        df_test_pred=df_test,
        df_future=df_future,
        target_col=target_col,
        pred_col=pred_col,
        fig_path=full_forecast_path
    )

    print(f" Full forecast plot saved → {full_forecast_path}")

    print("\n Evaluation completed successfully!")
    return df_test_merged, df_future, metrics


# ──────────────────────────────────────────────
# 6. MAIN
# ──────────────────────────────────────────────

def main():
    evaluate_predictions()


if __name__ == "__main__":
    main()
