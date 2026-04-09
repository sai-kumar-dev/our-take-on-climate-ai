from __future__ import annotations

from pathlib import Path
from typing import Any

from .inspection import inspect_dataset
from .transforms import (
    merge_datasets,
    prepare_climate_features,
    prepare_crop_labels,
    prepare_soil_features,
    validate_final_dataset,
)
from .utils import ensure_parent_dir, read_config, read_table, resolve_path, write_json


def run_pipeline_from_config(
    root_dir: Path,
    raw_config: dict[str, Any],
    config_path: Path | None = None,
) -> dict[str, Any]:
    pipeline_cfg = dict(raw_config)
    datasets_cfg = pipeline_cfg.get("datasets", {})
    outputs_cfg = pipeline_cfg.get("outputs", {})

    raw_frames: dict[str, Any] = {}
    inspection_report: dict[str, Any] = {
        "config_path": str(config_path) if config_path else None,
        "datasets": {},
    }
    for dataset_name, dataset_cfg in datasets_cfg.items():
        dataset_path = resolve_path(root_dir, dataset_cfg["path"])
        frame = read_table(dataset_path)
        raw_frames[dataset_name] = frame
        inspection_payload = inspect_dataset(dataset_name, frame, dataset_cfg)
        inspection_payload["path"] = str(dataset_path)
        inspection_report["datasets"][dataset_name] = inspection_payload

    climate_frame = prepare_climate_features(raw_frames["climate"], datasets_cfg["climate"], pipeline_cfg)
    soil_frame, soil_time_mode = prepare_soil_features(raw_frames["soil"], datasets_cfg["soil"], pipeline_cfg)
    crop_frame, crop_probability_columns = prepare_crop_labels(raw_frames["crop"], datasets_cfg["crop"], pipeline_cfg)
    final_frame, merge_report = merge_datasets(
        climate_frame=climate_frame,
        soil_frame=soil_frame,
        crop_frame=crop_frame,
        crop_probability_columns=crop_probability_columns,
        soil_time_mode=soil_time_mode,
        pipeline_cfg=pipeline_cfg,
    )
    validation_report, summary_stats = validate_final_dataset(final_frame, crop_probability_columns)
    validation_report["merge_report"] = merge_report

    final_dataset_path = resolve_path(root_dir, outputs_cfg["final_dataset"])
    summary_stats_path = resolve_path(root_dir, outputs_cfg["summary_stats"])
    inspection_report_path = resolve_path(root_dir, outputs_cfg["inspection_report"])
    validation_report_path = resolve_path(root_dir, outputs_cfg["validation_report"])

    ensure_parent_dir(final_dataset_path)
    ensure_parent_dir(summary_stats_path)
    ensure_parent_dir(inspection_report_path)
    ensure_parent_dir(validation_report_path)
    final_frame.to_csv(final_dataset_path, index=False)
    summary_stats.to_csv(summary_stats_path)
    write_json(inspection_report_path, inspection_report)
    write_json(validation_report_path, validation_report)

    return {
        "config_path": str(config_path) if config_path else None,
        "outputs": {
            "final_dataset": str(final_dataset_path),
            "summary_stats": str(summary_stats_path),
            "inspection_report": str(inspection_report_path),
            "validation_report": str(validation_report_path),
        },
        "inspection_report": inspection_report,
        "validation_report": validation_report,
        "row_count": int(len(final_frame)),
        "crop_probability_columns": crop_probability_columns,
    }


def run_pipeline_from_path(root_dir: Path, config_path: Path) -> dict[str, Any]:
    config = read_config(config_path)
    return run_pipeline_from_config(root_dir=root_dir, raw_config=config, config_path=config_path)
