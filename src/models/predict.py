# src/models/predict.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# ──────────────────────────────────────────────
# 1. LOAD MODEL
# ──────────────────────────────────────────────

def load_model(model_path="models/xgb_model.pkl"):
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    return model


# ──────────────────────────────────────────────
# 2. PREDICT TEST (com y verdadeiro)
# ──────────────────────────────────────────────

def predict_test(
    processed_path="data/processed/energy_features.csv",
    model_path="models/xgb_model.pkl",
    output_path="data/processed/test_predictions.csv",
    split_date="2014-01-01"
):
    """
    Run prediction on historical test data (real known data).
    Train until split_date, predict after split_date.
    """

    print("\n Loading processed dataset...")
    df = pd.read_csv(processed_path, parse_dates=["Datetime"], index_col="Datetime")
    df = df.sort_index()

    FEATURES = [
        "dayofyear", "hour", "dayofweek", "quarter", "month", "year",
        "lag_364d", "lag_728d", "lag_1092d"
    ]
    TARGET = "PJME_MW"

    print(" Splitting train/test based on date...")

    train = df[df.index < split_date].copy()
    test = df[df.index >= split_date].copy()

    print(f"Train size: {len(train)}, Test size: {len(test)}")

    print(" Dropping NaN rows caused by lags...")
    train = train.dropna(subset=FEATURES + [TARGET])
    test = test.dropna(subset=FEATURES + [TARGET])

    X_train = train[FEATURES]
    y_train = train[TARGET]

    X_test = test[FEATURES]
    y_test = test[TARGET]

    print(" Loading model...")
    model = load_model(model_path)

    print(" Predicting on test set...")
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"\n✅ Test RMSE: {rmse:,.2f}")

    print(f" Saving predictions to {output_path}")

    out_df = test.copy()
    out_df["prediction"] = y_pred
    out_df[["PJME_MW", "prediction"]].to_csv(output_path)

    return out_df, rmse


# ──────────────────────────────────────────────
# 3. PREDICT FUTURE (sem y verdadeiro)
# ──────────────────────────────────────────────

def build_full_feature_set(df):
    """
    Apply preprocessing, time-based features and lag features
    to historical data.
    """
    df = preprocess(df)
    df = create_time_features(df)

    lags = [364, 728, 1092]
    df = create_lag_features(df, target_col="PJME_MW", lags=lags)

    return df


def create_future_dataframe(last_date, periods=24 * 365):
    """
    Create future hourly timestamps for prediction.
    """

    future_index = pd.date_range(
        start=last_date + timedelta(hours=1),
        periods=periods,
        freq="1H"
    )

    future_df = pd.DataFrame(index=future_index)
    future_df["isFuture"] = True
    return future_df


def build_features_for_future(df_full):
    """
    After concatenating historical + future,
    apply the SAME feature functions to the future data.
    """

    df_full = create_time_features(df_full)

    lags = [364, 728, 1092]
    df_full = create_lag_features(df_full, target_col="PJME_MW", lags=lags)

    return df_full


def predict_future(
    processed_path="data/processed/energy_features.csv",
    model_path="models/xgb_model.pkl",
    output_path="data/processed/future_predictions.csv",
    future_hours=24 * 365,  # 1 ano
):
    """
    Generate future predictions using the trained XGBoost model.
    """

    print("\n Loading historical processed data...")
    df = pd.read_csv(processed_path, parse_dates=["Datetime"], index_col="Datetime")
    df = df.sort_index()

    print(" Loading model...")
    model = load_model(model_path)

    print(" Rebuilding full historical feature set...")
    df_full = build_full_feature_set(df)
    df_full["isFuture"] = False

    print(" Creating future timestamps...")
    last_date = df_full.index.max()
    future_df = create_future_dataframe(last_date, periods=future_hours)

    print(" Concatenating historical + future...")
    df_future_all = pd.concat([df_full, future_df], axis=0)

    print(" Applying feature engineering to future data...")
    df_future_all = build_features_for_future(df_future_all)

    # Select only future rows
    df_future = df_future_all[df_future_all["isFuture"] == True].copy()

    FEATURES = [
        "dayofyear", "hour", "dayofweek", "quarter", "month", "year",
        "lag_364d", "lag_728d", "lag_1092d"
    ]

    print(" Running predictions...")
    df_future["prediction"] = model.predict(df_future[FEATURES])

    print(f" Saving predictions to {output_path}")
    df_future[["prediction"]].to_csv(output_path)

    print("\n Prediction completed!")
    return df_future


# ──────────────────────────────────────────────
# 4. MAIN 
# ──────────────────────────────────────────────

def main():
    # Você escolhe o que rodar
    predict_test()
    # predict_future()


if __name__ == "__main__":
    main()