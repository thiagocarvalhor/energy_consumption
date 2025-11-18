# src/data/load_data.py

import os
import pandas as pd


def load_raw_energy_data(raw_path: str) -> pd.DataFrame:
    """
    Load the raw energy dataset stored locally.

    Parameters
    ----------
    raw_path : str
        Path to the raw CSV downloaded from Kaggle.

    Returns
    -------
    pd.DataFrame
        DataFrame with datetime column parsed and sorted.
    """
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at: {raw_path}")

    df = pd.read_csv(raw_path)

    # Standard Kaggle notebook step:
    # Parse timestamp column and sort
    datetime_col = "timestamp"
    if datetime_col not in df.columns:
        raise ValueError(f"Column '{datetime_col}' not found in dataset.")

    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(datetime_col).reset_index(drop=True)

    return df


def save_interim(df: pd.DataFrame, output_path: str) -> None:
    """
    Save cleaned raw dataset into interim directory.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    output_path : str
        Path to save the interim CSV.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    """
    Executes the raw → interim data transformation.

    Adjust paths below according to your cookiecutter structure.
    """
    raw_path = "data/raw/kaggle_energy.csv"
    interim_path = "data/interim/energy_interim.csv"

    print(f"Loading raw dataset from: {raw_path}")
    df = load_raw_energy_data(raw_path)

    print(f"Saving interim dataset to: {interim_path}")
    save_interim(df, interim_path)

    print("Done.")


if __name__ == "__main__":
    main()
