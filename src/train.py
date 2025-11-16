"""
Main training script using Hydra.

This file is intentionally minimal.
Implement the full training logic here following your project's needs.
"""

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="main", version_base=None)
def main(cfg: DictConfig):
    print("Training pipeline started.")
    print("Config loaded:\n", cfg)

    # TODO: Implement training logic
    pass


if __name__ == "__main__":
    main()
