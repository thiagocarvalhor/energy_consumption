# src/data/load_data.py

import os
import pandas as pd


def load_raw_energy_data(raw_path: str) -> pd.DataFrame:
    """
    Load the PJME_hourly.csv dataset and prepare the datetime index.

    Parameters
    ----------
    raw_path : str
        Path to the raw CSV downloaded from Kaggle.

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by Datetime (parsed as datetime64).
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at: {raw_path}")

    # Load CSV exactly like Kaggle
    df = pd.read_csv(raw_path)

    datetime_col = "Datetime"
    if datetime_col not in df.columns:
        raise ValueError(f"Column '{datetime_col}' not found in dataset.")

    # Set Datetime as index
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.set_index(datetime_col)

    # Sort index (always safe for time series)
    df = df.sort_index()

    return df


def save_interim(df: pd.DataFrame, output_path: str) -> None:
    """
    Save cleaned dataset to interim directory.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)


def main( raw_path = "data/raw/PJME_hourly.csv",
    interim_path = "data/interim/energy_interim.csv"):


    print(f"Loading raw dataset from: {raw_path}")
    df = load_raw_energy_data(raw_path)

    print(f"Saving interim dataset to: {interim_path}")
    save_interim(df, interim_path)

    print("Done.")


if __name__ == "__main__":
    main()
