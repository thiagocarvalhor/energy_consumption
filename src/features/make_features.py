# src/data/make_features.py

import os
import pandas as pd
import hydra
from omegaconf import DictConfig
from pathlib import Path
from hydra.utils import get_original_cwd


# ───────────────────────────────────────────────
# 1. PREPROCESS
# ───────────────────────────────────────────────
def preprocess(df):
    """Remove outliers (Kaggle style)."""
    return df.query("PJME_MW > 19000").copy()


# ───────────────────────────────────────────────
# 2. LAG FEATURES
# ───────────────────────────────────────────────
def create_lag_features(df, target_col, lags):
    df = df.copy()
    target_map = df[target_col].to_dict()

    for lag in lags:
        df[f"lag_{lag}d"] = (df.index - pd.Timedelta(days=lag)).map(target_map)

    return df


# ───────────────────────────────────────────────
# 3. TIME FEATURES
# ───────────────────────────────────────────────
def create_time_features(df):
    df = df.copy()
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["quarter"] = df.index.quarter
    df["month"] = df.index.month
    df["year"] = df.index.year
    df["dayofyear"] = df.index.dayofyear
    df["dayofmonth"] = df.index.day
    df["weekofyear"] = df.index.isocalendar().week.astype(int)
    return df


# ───────────────────────────────────────────────
# 4. BUILD FULL FEATURE SET
# ───────────────────────────────────────────────
def build_features(df, target_col, lags):
    df = preprocess(df)
    df = create_time_features(df)
    df = create_lag_features(df, target_col=target_col, lags=lags)
    return df


# ───────────────────────────────────────────────
# 5. HYDRA MAIN
# ───────────────────────────────────────────────
@hydra.main(config_path=str(Path(__file__).resolve().parents[2] / "configs" / "features"),
            config_name="make", version_base=None)
def main(cfg: DictConfig):

    # Always use original project root (not hydra working dir)
    ROOT = Path(get_original_cwd())

    input_path = ROOT / cfg.features.input_path
    output_path = ROOT / cfg.features.output_path

    print(f"\n Loading INTERIM data from: {input_path}")
    df = pd.read_csv(
        input_path,
        parse_dates=[cfg.features.datetime_column],
        index_col=cfg.features.datetime_column
    )

    # Build features
    print(" Building feature set...")
    df_feat = build_features(
        df,
        target_col=cfg.features.target_column,
        lags=cfg.features.lags
    )

    # Save output
    os.makedirs(output_path.parent, exist_ok=True)
    df_feat.to_csv(output_path)

    print(f" Saved processed dataset to: {output_path}")
    print(" make_features completed successfully.")


if __name__ == "__main__":
    main()
