from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.climate_pipeline.pipeline import run_pipeline_from_path


ROOT_DIR = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the processed dataset from raw configured sources.")
    parser.add_argument(
        "--config",
        default="configs/data_new_config.json",
        help="Path to the pipeline config JSON.",
    )
    args = parser.parse_args()

    config_path = (ROOT_DIR / args.config).resolve()
    result = run_pipeline_from_path(ROOT_DIR, config_path)
    print(json.dumps(result["outputs"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
