from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch a background process and store its PID.")
    parser.add_argument("--pid-file", required=True)
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--stderr-log", required=True)
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.command or args.command[0] != "--" or len(args.command) < 2:
        parser.error("Pass the target command after --")
    args.command = args.command[1:]
    return args


def build_env(overrides: list[str]) -> dict[str, str]:
    env = dict(os.environ)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid env override: {item}")
        key, value = item.split("=", 1)
        env[key] = value
    return env


def main() -> int:
    args = parse_args()
    pid_path = Path(args.pid_file)
    stdout_path = Path(args.stdout_log)
    stderr_path = Path(args.stderr_log)
    cwd = Path(args.cwd)
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    creationflags = 0
    if os.name == "nt":
        creationflags = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )

    env = build_env(args.env)
    with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr_handle:
        process = subprocess.Popen(
            args.command,
            cwd=str(cwd),
            stdout=stdout_handle,
            stderr=stderr_handle,
            env=env,
            creationflags=creationflags,
        )

    pid_path.write_text(str(process.pid), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
