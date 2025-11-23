# src/utils/logger.py

import logging
from pathlib import Path


def get_logger(name: str = __name__, log_file: Path = None, level: int = logging.INFO):
    """
    Creates and configures a logger instance.

    Args:
        name (str): Name of the logger.
        log_file (Path, optional): If provided, logs will also be written to this file.
        level (int): Logging level (default INFO).

    Returns:
        logging.Logger: Configured logger.
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers in case get_logger() is called multiple times
    if logger.hasHandlers():
        return logger

    # ───────────────────────────────────────────────
    # Console handler
    # ───────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_format = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # ───────────────────────────────────────────────
    # File handler (optional)
    # ───────────────────────────────────────────────
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        file_format = logging.Formatter(
            "%(asctime)s — %(levelname)s — %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger
