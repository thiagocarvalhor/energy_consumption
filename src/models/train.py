import os
import hydra
from omegaconf import DictConfig
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):

    # ────────────────────────────────────────────────
    # 1. LOAD DATA
    # ────────────────────────────────────────────────
    print("\n Loading processed dataset...")
    df = pd.read_csv(
        cfg.data.path,
        parse_dates=[cfg.data.datetime_column],
        index_col=cfg.data.datetime_column
    )
    df = df.sort_index()

    TARGET = cfg.data.target_column
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

    # ────────────────────────────────────────────────
    # 2. FILTER TRAIN WINDOW
    # ────────────────────────────────────────────────
    train_end = cfg.data.train_end
    df_train = df[df.index < train_end].copy()

    print(f"\n Training end date: {train_end}")
    print(f" Training samples: {len(df_train):,}")

    # Drop NaN (caused by lag features)
    df_train = df_train.dropna(subset=FEATURES + [TARGET])

    # ────────────────────────────────────────────────
    # 3. TIME SERIES SPLIT
    # ────────────────────────────────────────────────
    print("\n⏳ Running TimeSeriesSplit CV...")

    tss = TimeSeriesSplit(
        n_splits=5,
        test_size=24 * 365,  # 1 year in hours
        gap=24               # 24-hour gap
    )

    scores = []
    model = None

    for fold, (train_idx, val_idx) in enumerate(tss.split(df_train), start=1):
        print(f"\n===== FOLD {fold} =====")

        train = df_train.iloc[train_idx]
        val = df_train.iloc[val_idx]

        X_train = train[FEATURES]
        y_train = train[TARGET]

        X_val = val[FEATURES]
        y_val = val[TARGET]

        # ────────────────────────────────────────────────
        # 4. MODEL FROM HYDRA CONFIG
        # ────────────────────────────────────────────────
        model = xgb.XGBRegressor(
            booster=cfg.model.booster,
            n_estimators=cfg.model.n_estimators,
            max_depth=cfg.model.max_depth,
            learning_rate=cfg.model.learning_rate,
            base_score=cfg.model.base_score,
            objective=cfg.model.objective,
            random_state=cfg.seed,
        )

        # Train with early stopping
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            early_stopping_rounds=cfg.model.early_stopping_rounds,
            verbose=100,
        )

        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        scores.append(rmse)

        print(f"Fold {fold} RMSE: {rmse:.4f}")

    # ────────────────────────────────────────────────
    # 5. FINAL RESULTS
    # ────────────────────────────────────────────────
    print("\n===== FINAL CV RESULTS =====")
    print(f"RMSE per fold: {scores}")
    print(f"Mean RMSE: {np.mean(scores):.4f}")

    # ────────────────────────────────────────────────
    # 6. SAVE MODEL IN HYDRA OUTPUT DIR
    # ────────────────────────────────────────────────
    output_dir = os.getcwd()
    model_path = os.path.join(output_dir, f"{cfg.model.name}.json")

    print(f"\n Saving model to: {model_path}")
    model.save_model(model_path)
    print(" Model saved successfully!")

    return model


if __name__ == "__main__":
    main()
