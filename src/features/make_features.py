import pandas as pd

def preprocess(df):
    """
    Remove outliers and perform basic cleaning.
    """
    df = df.query("PJME_MW > 19000").copy()
    return df


def create_lag_features(df, target_col, lags):
    """
    Create lag features based on timedelta offsets.
    """
    df = df.copy()

    # Pre-map target values for fast lookup
    target_map = df[target_col].to_dict()

    # Create each lag
    for lag in lags:
        df[f"lag_{lag}d"] = (df.index - pd.Timedelta(days=lag)).map(target_map)

    return df


def create_time_features(df):
    """
    Create time-based features from the Datetime index.
    """
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


def build_features(df):
    """
    Apply preprocessing, time-based features, and seasonal lag features.
    """
    df = preprocess(df)
    df = create_time_features(df)

    # Kaggle lag values (1y, 2y, 3y)
    lags = [364, 728, 1092]

    df = create_lag_features(df, target_col="PJME_MW", lags=lags)
    return df


def save_processed(df, path):
    df.to_csv(path)
    print(f"Saved processed dataset to {path}")


def main(
    interim_path="data/interim/energy_interim.csv",
    processed_path="data/processed/energy_features.csv"
):
    # read
    df = pd.read_csv(interim_path, parse_dates=["Datetime"], index_col="Datetime")

    # build features
    df_features = build_features(df)

    # save
    save_processed(df_features, processed_path)


if __name__ == "__main__":
    main()
