from __future__ import annotations

import os
from pathlib import Path


def load_project_env(
    *,
    root_dir: str | Path | None = None,
    env_filename: str = ".env",
    override: bool = False,
) -> Path | None:
    resolved_root = Path(root_dir).resolve() if root_dir else Path(__file__).resolve().parents[1]
    env_path = resolved_root / env_filename
    if not env_path.exists():
        return None

    for key, value in parse_env_file(env_path).items():
        if override or key not in os.environ:
            os.environ[key] = value
    return env_path


def parse_env_file(env_path: str | Path) -> dict[str, str]:
    path = Path(env_path)
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        clean_key = key.strip()
        if not clean_key:
            continue
        values[clean_key] = _normalize_env_value(raw_value.strip())
    return values


def _normalize_env_value(raw_value: str) -> str:
    if not raw_value:
        return ""
    if len(raw_value) >= 2 and raw_value[0] == raw_value[-1] and raw_value[0] in {'"', "'"}:
        return raw_value[1:-1]
    return raw_value
