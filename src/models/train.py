import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


def train_model(
    processed_path="data/processed/energy_features.csv",
    model_output_path="models/xgb_model_2014.pkl",
    train_end="2014-01-01"
):
    """
    Train an XGBoost model using time-series cross-validation (TimeSeriesSplit)
    """

    print("\n Loading processed dataset...")
    df = pd.read_csv(processed_path, parse_dates=["Datetime"], index_col="Datetime")
    df = df.sort_index()

    # ──────────────────────────────────────────────
    # Define features
    # ──────────────────────────────────────────────
    FEATURES = [
        "dayofyear",
        "hour",
        "dayofweek",
        "quarter",
        "month",
        "year",
        "lag_364d",
        "lag_728d",
        "lag_1092d",
    ]
    TARGET = "PJME_MW"

    # ──────────────────────────────────────────────
    # CUT TRAINING WINDOW (important)
    # ──────────────────────────────────────────────
    df_train = df[df.index < train_end].copy()
    print(f"📌 Training window end: {train_end}")
    print(f"📌 Training samples: {len(df_train):,}")

    # Remove NaNs created by lag features
    df_train = df_train.dropna(subset=FEATURES + [TARGET])

    # ──────────────────────────────────────────────
    # TimeSeriesSplit — only inside the training period
    # ──────────────────────────────────────────────
    tss = TimeSeriesSplit(
        n_splits=5,
        test_size=24 * 365,  # 1 year
        gap=24               # 24-hour gap
    )

    scores = []
    fold = 0
    model = None  # will store last trained model

    for train_idx, val_idx in tss.split(df_train):
        fold += 1
        print(f"\n===== FOLD {fold} =====")

        train = df_train.iloc[train_idx]
        val = df_train.iloc[val_idx]

        X_train = train[FEATURES]
        y_train = train[TARGET]

        X_val = val[FEATURES]
        y_val = val[TARGET]

        # Kaggle-like XGBoost model
        model = xgb.XGBRegressor(
            base_score=0.5,
            booster="gbtree",
            n_estimators=1000,
            early_stopping_rounds=50,
            objective="reg:linear",
            max_depth=3,
            learning_rate=0.01,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=100,
        )

        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        scores.append(rmse)

        print(f"Fold {fold} RMSE: {rmse:.4f}")

    # ──────────────────────────────────────────────
    # FINAL RESULTS
    # ──────────────────────────────────────────────
    print("\n===== RESULTS =====")
    print(f"RMSE per fold: {scores}")
    print(f"Mean RMSE: {np.mean(scores):.4f}")

    # ──────────────────────────────────────────────
    # Save final model (from last fold)
    # ──────────────────────────────────────────────
    print(f"\n Saving model to: {model_output_path}")
    model.save_model(model_output_path)
    print(" Model saved successfully!")

    return model


def main():
    train_model()


if __name__ == "__main__":
    main()
