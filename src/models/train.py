# src/train/train.py

import os
import hydra
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error


@hydra.main(config_path="../../configs/train", config_name="train", version_base=None)
def main(cfg: DictConfig):

    # ────────────────────────────────────────────────
    # PATHS
    # ────────────────────────────────────────────────
    ROOT = Path(get_original_cwd())

    data_path = ROOT / cfg.path.processed_path
    model_output_dir = ROOT / cfg.path.output_dir
    os.makedirs(model_output_dir, exist_ok=True)

    print(f"\n📥 Loading dataset from: {data_path}")
    df = pd.read_csv(
        data_path,
        parse_dates=[cfg.datetime_column],
        index_col=cfg.datetime_column
    )
    df = df.sort_index()

    # ────────────────────────────────────────────────
    # CONFIG
    # ────────────────────────────────────────────────
    FEATURES = cfg.features.columns
    TARGET = cfg.target
    train_end = cfg.train_end

    # ────────────────────────────────────────────────
    # FILTER TRAIN WINDOW
    # ────────────────────────────────────────────────
    df_train = df[df.index < train_end].copy()
    df_train = df_train.dropna(subset=FEATURES + [TARGET])

    print(f"\n Training window ends at: {train_end}")
    print(f" Train samples: {len(df_train):,}")

    # ────────────────────────────────────────────────
    # TIME SERIES SPLIT
    # ────────────────────────────────────────────────
    print("\n Running TimeSeriesSplit...")

    tss = TimeSeriesSplit(
        n_splits=cfg.cv.n_splits,
        test_size=cfg.cv.test_size,
        gap=cfg.cv.gap,
    )

    scores = []
    model = None

    # ────────────────────────────────────────────────
    # TRAIN LOOP
    # ────────────────────────────────────────────────
    for fold, (train_idx, val_idx) in enumerate(tss.split(df_train), start=1):
        print(f"\n===== FOLD {fold} =====")

        train = df_train.iloc[train_idx]
        val = df_train.iloc[val_idx]

        X_train = train[FEATURES]
        y_train = train[TARGET]

        X_val = val[FEATURES]
        y_val = val[TARGET]

        # MODEL FROM YAML
        model = xgb.XGBRegressor(
            **cfg.model,
            random_state=cfg.seed,
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

        print(f" Fold {fold} RMSE: {rmse:,.4f}")

    # ────────────────────────────────────────────────
    # FINAL RESULTS
    # ────────────────────────────────────────────────
    print("\n===== FINAL CV RESULTS =====")
    print(f"RMSE per fold: {scores}")
    print(f"Mean RMSE: {np.mean(scores):,.4f}")

    # ────────────────────────────────────────────────
    # SAVE MODEL
    # ────────────────────────────────────────────────
    model_name = f"{cfg.model.name}.json"
    model_path = model_output_dir / model_name

    print(f"\n Saving trained model to: {model_path}")
    model.save_model(model_path)

    print(" Training completed successfully!")


if __name__ == "__main__":
    main()
