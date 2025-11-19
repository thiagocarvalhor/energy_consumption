import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


def train_model(
    processed_path="data/processed/energy_features.csv",
    model_output_path="models/xgb_model.pkl",
):
    """
    Train an XGBoost model using time-series cross-validation (TimeSeriesSplit)
    """

    # Load processed data
    df = pd.read_csv(processed_path, parse_dates=["Datetime"], index_col="Datetime")
    df = df.sort_index()

    # Features used in Kaggle
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

    # Drop rows with NaN caused by lag creation
    df = df.dropna(subset=FEATURES + [TARGET])

    # TimeSeriesSplit config (same as Kaggle)
    tss = TimeSeriesSplit(
        n_splits=5,
        test_size=24 * 365 * 1,  # 1 year test window per fold
        gap=24,                  # 24h gap
    )

    scores = []
    fold = 0

    for train_idx, val_idx in tss.split(df):
        fold += 1
        print(f"\n===== FOLD {fold} =====")

        train = df.iloc[train_idx]
        val = df.iloc[val_idx]

        X_train = train[FEATURES]
        y_train = train[TARGET]

        X_val = val[FEATURES]
        y_val = val[TARGET]

        # XGBoost model parameters from Kaggle
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

        # Validation predictions
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        scores.append(rmse)

        print(f"Fold {fold} RMSE: {rmse:.4f}")

    print("\n===== RESULTS =====")
    print(f"RMSE per fold: {scores}")
    print(f"Mean RMSE: {np.mean(scores):.4f}")

    # Save final model (the last trained one)
    model.save_model(model_output_path)
    print(f"\nModel saved to {model_output_path}")


def main():
    train_model()


if __name__ == "__main__":
    main()
