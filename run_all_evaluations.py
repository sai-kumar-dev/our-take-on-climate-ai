from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from src.climate_pipeline.evaluation import run_all_evaluations


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_VENV_PYTHON = ROOT_DIR / ".venv" / "Scripts" / "python.exe"


def ensure_project_python() -> int | None:
    if os.getenv("CLIMATE_EVAL_USING_PROJECT_VENV") == "1":
        return None
    if not PROJECT_VENV_PYTHON.exists():
        return None

    current_python = Path(sys.executable).resolve()
    project_python = PROJECT_VENV_PYTHON.resolve()
    if current_python == project_python:
        return None

    env = os.environ.copy()
    env["CLIMATE_EVAL_USING_PROJECT_VENV"] = "1"
    command = [str(project_python), str(ROOT_DIR / "run_all_evaluations.py"), *sys.argv[1:]]
    completed = subprocess.run(command, cwd=ROOT_DIR, env=env, check=False)
    return int(completed.returncode)


def main() -> int:
    delegated_code = ensure_project_python()
    if delegated_code is not None:
        return delegated_code

    parser = argparse.ArgumentParser(description="Run the full evaluation pipeline for a saved artifact bundle.")
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Optional artifact directory override. Defaults to the latest supported artifact resolved from the registry.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/evaluation",
        help="Directory where evaluation outputs will be written.",
    )
    args = parser.parse_args()

    run_all_evaluations(
        root_dir=ROOT_DIR,
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
