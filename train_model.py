from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.climate_pipeline.training import train_from_config
from src.climate_pipeline.utils import read_config


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAINING_CONFIG_PATH = "configs/training_config.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the crop guidance model from a JSON config.")
    parser.add_argument(
        "--config",
        default=DEFAULT_TRAINING_CONFIG_PATH,
        help="Path to the training config JSON.",
    )
    args = parser.parse_args()

    config_path = (ROOT_DIR / args.config).resolve()
    config = read_config(config_path)
    result = train_from_config(ROOT_DIR, config, config_path=config_path)
    print(json.dumps(result["artifact_paths"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
