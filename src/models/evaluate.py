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
# 2. MERGE ACTUAL + PREDICTED
# ──────────────────────────────────────────────

def merge_actual_and_pred(
    actual_path="data/processed/energy_features.csv",
    pred_path="data/processed/test_predictions.csv",
    datetime_col="Datetime",
    target_col="PJME_MW",
    pred_col="prediction",
):
    """
    Merge historical actual data with prediction results using the Datetime index.
    """

    print("\n Loading actual (historical) dataframe...")
    df_actual = pd.read_csv(
        actual_path,
        parse_dates=[datetime_col],
        index_col=datetime_col
    )
    df_actual = df_actual.sort_index()

    print(" Loading predictions dataframe...")
    df_pred = pd.read_csv(
        pred_path,
        parse_dates=[datetime_col],
        index_col=datetime_col
    )
    df_pred = df_pred.sort_index()

    print(" Merging actual + predicted...")
    df = df_actual[[target_col]].join(
        df_pred[[pred_col]],
        how="inner"
    )

    df = df.dropna(subset=[target_col, pred_col])
    return df


# ──────────────────────────────────────────────
# 3. PLOTS
# ──────────────────────────────────────────────

def plot_actual_vs_pred(df, target_col, pred_col, fig_path):
    """
    Plot actual vs predicted time series and save as PNG.
    """
    plt.figure(figsize=(18, 6))
    plt.plot(df.index, df[target_col], label="Actual", linewidth=2)
    plt.plot(df.index, df[pred_col], label="Prediction", linewidth=2)

    plt.title("Actual vs Predicted Energy Consumption")
    plt.xlabel("Datetime")
    plt.ylabel("PJME_MW")
    plt.legend()
    plt.tight_layout()

    plt.savefig(fig_path)
    plt.close()


# ──────────────────────────────────────────────
# 4. FULL EVALUATION PIPELINE
# ──────────────────────────────────────────────

def evaluate_predictions(
    actual_path="data/processed/energy_features.csv",
    pred_path="data/processed/test_predictions.csv",
    output_dir="data/evaluation",
    target_col="PJME_MW",
    pred_col="prediction",
    datetime_col="Datetime",
):
    """
    Complete evaluation pipeline:
      • Load real data + predictions
      • Merge
      • Compute MAE and RMSE
      • Save evaluation CSV
      • Save metrics CSV
      • Save plot PNG
    """

    print("\n Starting evaluation...")

    # Create output structure
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Merge real + predicted
    df_eval = merge_actual_and_pred(
        actual_path=actual_path,
        pred_path=pred_path,
        datetime_col=datetime_col,
        target_col=target_col,
        pred_col=pred_col,
    )

    # Compute metrics
    print(" Computing metrics...")
    metrics = compute_metrics(
        df_eval[target_col],
        df_eval[pred_col],
    )

    print(f" MAE : {metrics['mae']:.2f}")
    print(f" RMSE: {metrics['rmse']:.2f}")

    # Save merged evaluation table
    table_path = os.path.join(tables_dir, "evaluation_table.csv")
    df_eval.to_csv(table_path)
    print(f" Evaluation table saved → {table_path}")

    # Save metrics file
    metrics_path = os.path.join(output_dir, "metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
    print(f" Metrics saved → {metrics_path}")

    # Save plot
    fig_path = os.path.join(plots_dir, "actual_vs_pred.png")
    plot_actual_vs_pred(df_eval, target_col, pred_col, fig_path)
    print(f" Plot saved → {fig_path}")

    print("\n Evaluation completed!")
    return df_eval, metrics


# ──────────────────────────────────────────────
# 5. MAIN
# ──────────────────────────────────────────────

def main():
    evaluate_predictions()


if __name__ == "__main__":
    main()
