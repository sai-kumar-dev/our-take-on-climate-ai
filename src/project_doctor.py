from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass
class CheckResult:
    label: str
    status: str
    detail: str


def parse_requirements(path: Path) -> dict[str, str]:
    requirements: dict[str, str] = {}
    if not path.exists():
        return requirements
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "==" not in line:
            continue
        name, pinned = line.split("==", 1)
        requirements[name.strip()] = pinned.strip()
    return requirements


def check_paths() -> tuple[list[CheckResult], bool, bool]:
    runtime_files = [
        ("API entrypoint", ROOT_DIR / "src" / "app_api_entry.py"),
        ("UI entrypoint", ROOT_DIR / "src" / "ui_app_source.py"),
        ("Launcher", ROOT_DIR / "run_all.bat"),
        ("Production model", ROOT_DIR / "artifacts" / "data_new_training" / "trained_model.pkl"),
        ("Feature config", ROOT_DIR / "artifacts" / "data_new_training" / "feature_config.json"),
        ("Scaler", ROOT_DIR / "artifacts" / "data_new_training" / "scaler.pkl"),
    ]
    training_files = [
        ("Training config", ROOT_DIR / "configs" / "training_data_new.json"),
        ("Test dataset", ROOT_DIR / "data" / "processed" / "final_ml_dataset.csv"),
        ("Training dataset", ROOT_DIR / "data" / "processed" / "data_new_final_ml_dataset.csv"),
    ]

    results: list[CheckResult] = []
    ready_to_run = True
    ready_to_retrain = True

    for label, path in runtime_files:
        exists = path.exists()
        results.append(CheckResult(label, "ok" if exists else "missing", str(path)))
        ready_to_run = ready_to_run and exists

    for label, path in training_files:
        exists = path.exists()
        results.append(CheckResult(label, "ok" if exists else "missing", str(path)))
        ready_to_retrain = ready_to_retrain and exists

    return results, ready_to_run, ready_to_retrain


def check_requirements(path: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    requirements = parse_requirements(path)
    if not requirements:
        return [CheckResult("Dependencies", "missing", str(path))]

    for package_name, pinned_version in requirements.items():
        try:
            installed_version = version(package_name)
        except PackageNotFoundError:
            results.append(CheckResult(package_name, "missing", f"expected {pinned_version}"))
            continue
        if installed_version == pinned_version:
            results.append(CheckResult(package_name, "ok", installed_version))
        else:
            results.append(
                CheckResult(
                    package_name,
                    "mismatch",
                    f"installed {installed_version}, expected {pinned_version}",
                )
            )
    return results


def main() -> int:
    print("[doctor] Climate Crop Guidance project check")
    print(f"[doctor] Root: {ROOT_DIR}")
    print("[doctor] Note: large raw data and model artifacts are kept local and are not committed to git.")

    path_results, ready_to_run, ready_to_retrain = check_paths()
    print("[doctor] File checks:")
    for item in path_results:
        print(f"  - [{item.status}] {item.label}: {item.detail}")

    dependency_results = check_requirements(ROOT_DIR / "requirements.txt")
    print("[doctor] Dependency checks:")
    mismatches = 0
    for item in dependency_results:
        print(f"  - [{item.status}] {item.label}: {item.detail}")
        if item.status != "ok":
            mismatches += 1

    print(f"[doctor] Ready to run: {'yes' if ready_to_run else 'no'}")
    print(f"[doctor] Ready to retrain: {'yes' if ready_to_retrain else 'no'}")
    return 0 if ready_to_run and mismatches == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
