"""
Data loading utilities.

Implement the logic to load datasets based on Hydra configs.
"""


def load_data(cfg):
    """
    Load data according to cfg.data.source.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config object.

    Returns
    -------
    DataFrame
        Loaded data.
    """
    raise NotImplementedError("Implement data loading cfg.data.source")
