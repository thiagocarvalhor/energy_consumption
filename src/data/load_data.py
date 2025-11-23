# src/data/load_data.py

import os
import pandas as pd
import hydra
from omegaconf import DictConfig
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

def load_raw_energy_data(raw_path: str, datetime_column: str) -> pd.DataFrame:
    """
    Load the Kaggle PJME dataset and parse the datetime column.
    """

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw dataset not found at: {raw_path}")

    df = pd.read_csv(raw_path)

    if datetime_column not in df.columns:
        raise ValueError(f"Column '{datetime_column}' not found in dataset.")

    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df = df.set_index(datetime_column)
    df = df.sort_index()

    return df


def save_interim(df: pd.DataFrame, output_path: str) -> None:
    """
    Save cleaned dataset to interim directory.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path)


@hydra.main(config_path=str(CONFIG_DIR / "data_load"), config_name="load", version_base=None)
def main(cfg: DictConfig):

    from hydra.utils import get_original_cwd
    project_root = Path(get_original_cwd())

    raw_path = project_root / cfg.data_load.raw_path
    output_path = project_root / cfg.data_load.output_path
    datetime_column = cfg.data_load.datetime_column

    print(f"\n Loading RAW dataset from: {raw_path}")
    df = load_raw_energy_data(raw_path, datetime_column)

    print(f" Saving INTERIM dataset to: {output_path}")
    save_interim(df, output_path)

    print(" load_data completed successfully.")

if __name__ == "__main__":
    main()
