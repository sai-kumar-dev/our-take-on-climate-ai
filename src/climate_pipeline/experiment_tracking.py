from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .utils import ensure_parent_dir, write_json


def resolve_git_commit(root_dir: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str))
        handle.write("\n")


def config_fingerprint(config: dict[str, Any]) -> str:
    raw = json.dumps(config, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def build_model_card(
    evaluation_report: dict[str, Any],
    artifact_paths: dict[str, Any],
    config: dict[str, Any],
    git_commit: str | None,
) -> str:
    metadata = artifact_paths.get("model_metadata", {})
    test_metrics = evaluation_report.get("metrics", {}).get("test", {})
    validation_metrics = evaluation_report.get("metrics", {}).get("validation", {})
    slice_metrics = evaluation_report.get("slice_metrics", {})
    generalization = evaluation_report.get("generalization_checks", {})

    lines = [
        "# Model Card",
        "",
        "## Summary",
        f"- Model version: `{metadata.get('model_version', 'unknown')}`",
        f"- Data version: `{metadata.get('data_version', 'unknown')}`",
        f"- Backend: `{metadata.get('model_type', 'unknown')}`",
        f"- Training mode: `{metadata.get('mode', 'unknown')}`",
        f"- Sanity mode: `{metadata.get('sanity_mode', 'unknown')}`",
        f"- Git commit: `{git_commit or 'unknown'}`",
        "",
        "## Intended Use",
        "- Use this model as district-season crop pattern guidance plus field input ranking.",
        "- Do not treat it as a guaranteed agronomic outcome or economic return predictor.",
        "",
        "## Label Definition",
        "- Current target: normalized historical crop-share distribution by district and time step.",
        "- Limitation: this reflects historical cultivation patterns, not direct yield, profit, or failure labels.",
        "",
        "## Primary Metrics",
        f"- Test top-1 accuracy: `{test_metrics.get('top_1_accuracy', 0.0):.4f}`",
        f"- Test top-k accuracy: `{test_metrics.get('top_k_accuracy', 0.0):.4f}`",
        f"- Test cross-entropy: `{test_metrics.get('cross_entropy', 0.0):.4f}`",
    ]
    if validation_metrics:
        lines.extend(
            [
                f"- Validation top-1 accuracy: `{validation_metrics.get('top_1_accuracy', 0.0):.4f}`",
                f"- Validation cross-entropy: `{validation_metrics.get('cross_entropy', 0.0):.4f}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Research Protocol",
            f"- Split strategy: `{evaluation_report.get('split', {}).get('strategy', 'unknown')}`",
            "- Forward-chaining temporal evaluation is used as the primary validation protocol.",
            "- Slice metrics and geography-overlap checks should be reviewed before promotion.",
            "",
            "## Slice Metrics Available",
            f"- Slice groups: `{', '.join(slice_metrics.keys()) if slice_metrics else 'none'}`",
            f"- Geography checks: `{', '.join(generalization.keys()) if generalization else 'none'}`",
            "",
            "## Scenario Guidance",
            f"- Scenario stress uses rule blend weight `{config.get('inference', {}).get('scenario_rule_blend_weight', 'default')}` at inference time.",
            "- Scenario outputs should be interpreted as stress-tested shortlist movement, not causal intervention proof.",
            "",
            "## Operational Notes",
            f"- Artifact directory: `{artifact_paths.get('output_dir', 'unknown')}`",
            f"- Feature config: `{artifact_paths.get('feature_config', 'unknown')}`",
            f"- Evaluation report: `{artifact_paths.get('evaluation_report', 'unknown')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def record_training_run(
    root_dir: Path,
    config: dict[str, Any],
    artifact_paths: dict[str, Any],
    evaluation_report: dict[str, Any],
    training_summary: dict[str, Any],
) -> dict[str, Any]:
    governance_cfg = config.get("governance", {})
    registry_dir = (root_dir / governance_cfg.get("registry_dir", "artifacts/registry")).resolve()
    run_manifest_path = Path(artifact_paths["output_dir"]) / governance_cfg.get("run_manifest_name", "run_manifest.json")
    model_card_path = Path(artifact_paths["output_dir"]) / governance_cfg.get("model_card_name", "model_card.md")
    experiment_registry_path = registry_dir / governance_cfg.get("experiment_registry_name", "training_runs.jsonl")
    model_registry_path = registry_dir / governance_cfg.get("model_registry_name", "model_registry.jsonl")

    git_commit = resolve_git_commit(root_dir)
    timestamp = datetime.now(timezone.utc).isoformat()
    run_record = {
        "recorded_at": timestamp,
        "git_commit": git_commit,
        "config_fingerprint": config_fingerprint(config),
        "artifacts": artifact_paths,
        "model_metadata": artifact_paths.get("model_metadata", {}),
        "metrics": evaluation_report.get("metrics", {}),
        "slice_metrics": evaluation_report.get("slice_metrics", {}),
        "generalization_checks": evaluation_report.get("generalization_checks", {}),
        "training_summary": training_summary,
        "target_definition": {
            "type": "historical_crop_share_distribution",
            "promotion_guardrail": "human_review_required_for_farmer-facing use",
        },
    }
    write_json(run_manifest_path, run_record)
    model_card = build_model_card(
        evaluation_report=evaluation_report,
        artifact_paths=artifact_paths,
        config=config,
        git_commit=git_commit,
    )
    ensure_parent_dir(model_card_path)
    model_card_path.write_text(model_card, encoding="utf-8")

    append_jsonl(experiment_registry_path, run_record)
    append_jsonl(
        model_registry_path,
        {
            "recorded_at": timestamp,
            "model_version": artifact_paths.get("model_metadata", {}).get("model_version", "unknown"),
            "data_version": artifact_paths.get("model_metadata", {}).get("data_version", "unknown"),
            "artifact_dir": artifact_paths.get("output_dir"),
            "feature_config": artifact_paths.get("feature_config"),
            "evaluation_report": artifact_paths.get("evaluation_report"),
            "model_card": str(model_card_path),
            "git_commit": git_commit,
            "promotion_status": "local_candidate",
        },
    )
    return {
        "run_manifest": str(run_manifest_path),
        "model_card": str(model_card_path),
        "experiment_registry": str(experiment_registry_path),
        "model_registry": str(model_registry_path),
        "git_commit": git_commit,
    }
