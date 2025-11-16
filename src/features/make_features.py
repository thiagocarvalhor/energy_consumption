"""
Feature engineering utilities for time series forecasting.

Implement lag features, rolling windows, scaling, etc.
"""


def make_features(df, cfg):
    """
    Build features from raw dataframe.

    Parameters
    ----------
    df : DataFrame
        Input dataset.

    cfg : DictConfig
        Hydra config object.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """
    raise NotImplementedError("Implement feature engineering here.")
