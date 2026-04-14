from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score, precision_recall_fscore_support

from .scenario_explainer import (
    format_for_ui,
    generate_scenario_explanation,
    render_explanations_markdown,
)
from .training import (
    CropSuitabilityModelBundle,
    apply_scenario,
    crop_name_from_label,
    jensen_shannon_divergence,
    normalize_probability_matrix,
)
from .utils import ensure_parent_dir, read_table, resolve_path, write_json

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


OUTPUT_SUBDIRS = [
    "performance",
    "stability",
    "scenario",
    "shap",
    "tables",
    "summaries",
]
ALLOWED_REGISTRY_ARTIFACTS = ("data_new_training", "test_inference_model")
INTEGER_LIKE_FEATURES = {"dry_spell_days", "time_step_missing", "climate_gap_filled", "soil_imputed"}
BOUNDED_ZERO_ONE_FEATURES = {
    "irrigation_index",
    "rotation_score",
    "time_step_missing",
    "climate_gap_filled",
    "soil_imputed",
    "geo_confidence",
    "data_confidence",
}
NON_NEGATIVE_FEATURES = {"rain_total", "rain_variance", "max_rain_1d", "dry_spell_days", "N", "P", "K", "soil_health_index"}
TOP_K = 3
SHAP_MAX_ROWS = 256
SHAP_MAX_DISPLAY = 20

SCENARIO_CONFIGS: dict[str, dict[str, dict[str, float]]] = {
    "low_rainfall": {
        "feature_multipliers": {
            "rain_total": 0.70,
            "rain_lag_14": 0.70,
            "humidity_avg": 0.95,
        }
    },
    "high_temperature": {
        "feature_additions": {
            "temp_avg": 4.0,
            "max_temp": 5.0,
            "max_temp_3d": 5.0,
        }
    },
    "increased_irrigation": {
        "feature_additions": {
            "irrigation_index": 0.25,
        }
    },
}


@dataclass
class EvaluationContext:
    root_dir: Path
    artifact_dir: Path
    output_dir: Path
    output_dirs: dict[str, Path]
    model_bundle: CropSuitabilityModelBundle
    feature_config: dict[str, Any]
    evaluation_report: dict[str, Any]
    dataset_path: Path
    dataset_frame: pd.DataFrame
    train_frame: pd.DataFrame
    validation_frame: pd.DataFrame
    test_frame: pd.DataFrame
    label_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    model_version: str
    backend: str


def run_all_evaluations(
    root_dir: Path,
    artifact_dir: str | Path | None = None,
    output_dir: str | Path = "artifacts/evaluation",
) -> dict[str, Any]:
    context = load_evaluation_context(root_dir=root_dir, artifact_dir=artifact_dir, output_dir=output_dir)

    performance = run_performance(context)
    stability = run_stability(context, base_predictions=performance["predictions"])
    scenario = run_scenario(context, base_predictions=performance["predictions"])
    shap_summary = run_shap(context, base_predictions=performance["predictions"])
    summary = write_summary_artifacts(
        context=context,
        performance=performance,
        stability=stability,
        scenario=scenario,
        shap_summary=shap_summary,
    )
    print(f"Saved evaluation summary to {context.output_dirs['summaries']}")
    return {
        "artifact_dir": str(context.artifact_dir),
        "output_dir": str(context.output_dir),
        "model_version": context.model_version,
        "performance": performance["metrics"],
        "stability": stability["summary"],
        "scenario": scenario["summary_rows"],
        "shap": shap_summary,
        "summary": summary,
    }


def load_evaluation_context(
    root_dir: Path,
    artifact_dir: str | Path | None = None,
    output_dir: str | Path = "artifacts/evaluation",
) -> EvaluationContext:
    ensure_runtime_import_path(root_dir)
    resolved_artifact_dir = resolve_artifact_dir(root_dir, artifact_dir)
    resolved_output_dir = resolve_path(root_dir, str(output_dir))
    output_dirs = ensure_output_dirs(resolved_output_dir)

    model_path = resolved_artifact_dir / "trained_model.pkl"
    if not model_path.exists():
        versioned_models = sorted(resolved_artifact_dir.glob("model_v*.pkl"))
        if not versioned_models:
            raise FileNotFoundError(f"No trained model found in {resolved_artifact_dir}")
        model_path = versioned_models[-1]

    feature_config_path = resolved_artifact_dir / "feature_config.json"
    evaluation_report_path = resolved_artifact_dir / "evaluation_report.json"
    if not feature_config_path.exists():
        raise FileNotFoundError(f"Missing feature config: {feature_config_path}")
    if not evaluation_report_path.exists():
        raise FileNotFoundError(f"Missing evaluation report: {evaluation_report_path}")

    model_bundle = joblib.load(model_path)
    with feature_config_path.open("r", encoding="utf-8") as handle:
        feature_config = json.load(handle)
    with evaluation_report_path.open("r", encoding="utf-8") as handle:
        evaluation_report = json.load(handle)

    dataset_path = resolve_dataset_path(root_dir, feature_config, evaluation_report)
    dataset_frame = read_table(dataset_path)
    train_frame, validation_frame, test_frame = split_dataset_frame(dataset_frame, feature_config, evaluation_report)

    preprocessor_cfg = feature_config.get("preprocessor", {})
    label_columns = list(feature_config.get("labels", []))
    if not label_columns:
        raise ValueError(f"No label columns found in {feature_config_path}")
    if test_frame.empty:
        raise ValueError("Resolved test split is empty; cannot run evaluation.")

    model_version = str(feature_config.get("model_metadata", {}).get("model_version", "unknown"))
    backend = str(feature_config.get("backend", getattr(model_bundle, "backend", "unknown")))

    return EvaluationContext(
        root_dir=root_dir,
        artifact_dir=resolved_artifact_dir,
        output_dir=resolved_output_dir,
        output_dirs=output_dirs,
        model_bundle=model_bundle,
        feature_config=feature_config,
        evaluation_report=evaluation_report,
        dataset_path=dataset_path,
        dataset_frame=dataset_frame,
        train_frame=train_frame,
        validation_frame=validation_frame,
        test_frame=test_frame,
        label_columns=label_columns,
        numeric_features=list(preprocessor_cfg.get("numeric_features", [])),
        categorical_features=list(preprocessor_cfg.get("categorical_features", [])),
        model_version=model_version,
        backend=backend,
    )


def run_performance(context: EvaluationContext) -> dict[str, Any]:
    y_true = normalize_probability_matrix(context.test_frame[context.label_columns].to_numpy(dtype=float))
    predictions = normalize_probability_matrix(context.model_bundle.predict(context.test_frame))
    true_top1 = np.argmax(y_true, axis=1)
    pred_top1 = np.argmax(predictions, axis=1)
    top3_predictions = top_k_indices(predictions, TOP_K)
    top3_accuracy = float(np.mean([int(true_top1[i] in top3_predictions[i]) for i in range(len(true_top1))]))

    precision, recall, f1, support = precision_recall_fscore_support(
        true_top1,
        pred_top1,
        labels=list(range(len(context.label_columns))),
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_top1,
        pred_top1,
        average="macro",
        zero_division=0,
    )
    ndcg = float(ndcg_score(y_true, predictions))

    metric_records = [
        {"metric": "Top-1 Accuracy", "value": float(np.mean(true_top1 == pred_top1))},
        {"metric": "Top-3 Accuracy", "value": top3_accuracy},
        {"metric": "Macro Precision", "value": float(macro_precision)},
        {"metric": "Macro Recall", "value": float(macro_recall)},
        {"metric": "Macro F1", "value": float(macro_f1)},
        {"metric": "NDCG", "value": ndcg},
    ]
    metrics_df = pd.DataFrame(metric_records)

    classwise_rows = []
    for class_index, label in enumerate(context.label_columns):
        class_mask = true_top1 == class_index
        accuracy = float(np.mean(pred_top1[class_mask] == class_index)) if class_mask.any() else 0.0
        classwise_rows.append(
            {
                "class_name": display_crop_name(label),
                "label_column": label,
                "accuracy": accuracy,
                "precision": float(precision[class_index]),
                "recall": float(recall[class_index]),
                "f1": float(f1[class_index]),
                "support": int(support[class_index]),
            }
        )
    classwise_df = pd.DataFrame(classwise_rows).sort_values(
        ["accuracy", "support", "class_name"],
        ascending=[False, False, True],
    )

    metrics_json = {
        "model_version": context.model_version,
        "artifact_dir": str(context.artifact_dir),
        "dataset_path": str(context.dataset_path),
        "backend": context.backend,
        "test_rows": int(len(context.test_frame)),
        "metrics": {row["metric"]: round(float(row["value"]), 6) for row in metric_records},
    }
    performance_dir = context.output_dirs["performance"]
    tables_dir = context.output_dirs["tables"]
    write_dataframe(performance_dir / "metrics.csv", metrics_df)
    write_json(performance_dir / "metrics.json", metrics_json)
    write_dataframe(tables_dir / "classwise_performance.csv", classwise_df)

    create_metric_bar_chart(metrics_df, performance_dir / "performance_bar_chart.png")
    create_classwise_chart(classwise_df, performance_dir / "classwise_accuracy.png")

    print(f"Saved performance metrics to {performance_dir}")
    return {
        "metrics": metrics_json["metrics"],
        "metrics_df": metrics_df,
        "classwise_df": classwise_df,
        "predictions": predictions,
        "y_true": y_true,
    }


def run_stability(context: EvaluationContext, base_predictions: np.ndarray | None = None) -> dict[str, Any]:
    reference_frame = context.test_frame.copy().reset_index(drop=True)
    perturbed_frame = perturb_numeric_inputs(reference_frame, context.numeric_features, noise_fraction=0.03, seed=42)

    baseline = normalize_probability_matrix(
        base_predictions if base_predictions is not None else context.model_bundle.predict(reference_frame)
    )
    perturbed = normalize_probability_matrix(context.model_bundle.predict(perturbed_frame))

    baseline_top1 = np.argmax(baseline, axis=1)
    perturbed_top1 = np.argmax(perturbed, axis=1)
    baseline_top3 = [set(row.tolist()) for row in top_k_indices(baseline, TOP_K)]
    perturbed_top3 = [set(row.tolist()) for row in top_k_indices(perturbed, TOP_K)]
    js_values = jensen_shannon_divergence(baseline, perturbed)
    probability_delta = np.mean(np.abs(baseline - perturbed), axis=1)

    stability_df = pd.DataFrame(
        {
            "row_index": np.arange(len(reference_frame)),
            "region_key": reference_frame.get("region_key", pd.Series("", index=reference_frame.index)).astype(str).tolist(),
            "time": reference_frame.get("time", pd.Series("", index=reference_frame.index)).astype(str).tolist(),
            "top1_consistent": (baseline_top1 == perturbed_top1).astype(int),
            "top3_consistent": np.asarray(
                [int(baseline_top3[i] == perturbed_top3[i]) for i in range(len(reference_frame))],
                dtype=int,
            ),
            "js_divergence": js_values,
            "mean_probability_delta": probability_delta,
        }
    )

    summary = {
        "rows_evaluated": int(len(reference_frame)),
        "noise_fraction": 0.03,
        "top1_consistency": round(float(stability_df["top1_consistent"].mean()), 6),
        "top3_consistency": round(float(stability_df["top3_consistent"].mean()), 6),
        "mean_js_divergence": round(float(stability_df["js_divergence"].mean()), 6),
        "max_js_divergence": round(float(stability_df["js_divergence"].max()), 6),
        "mean_probability_delta": round(float(stability_df["mean_probability_delta"].mean()), 6),
    }

    stability_dir = context.output_dirs["stability"]
    write_dataframe(stability_dir / "stability_metrics.csv", stability_df)
    write_json(stability_dir / "stability_summary.json", summary)
    create_stability_histogram(stability_df["js_divergence"], stability_dir / "stability_histogram.png")

    print(f"Saved stability metrics to {stability_dir}")
    return {
        "summary": summary,
        "details": stability_df,
    }


def run_scenario(context: EvaluationContext, base_predictions: np.ndarray | None = None) -> dict[str, Any]:
    reference_frame = context.test_frame.copy().reset_index(drop=True)
    baseline = normalize_probability_matrix(
        base_predictions if base_predictions is not None else context.model_bundle.predict(reference_frame)
    )
    baseline_top1 = np.argmax(baseline, axis=1)
    mean_baseline = baseline.mean(axis=0)
    baseline_payload = build_ranking_payload(mean_baseline, context.label_columns, top_n=5)

    summary_rows: list[dict[str, Any]] = []
    scenario_rows: list[dict[str, Any]] = []
    explanation_records: list[dict[str, Any]] = []

    for scenario_name, scenario_cfg in SCENARIO_CONFIGS.items():
        scenario_frame = apply_scenario(reference_frame, scenario_cfg)
        scenario_frame = clip_numeric_features(scenario_frame, context.numeric_features)
        scenario_prediction = normalize_probability_matrix(context.model_bundle.predict(scenario_frame))

        mean_scenario = scenario_prediction.mean(axis=0)
        mean_delta = mean_scenario - mean_baseline
        js_values = jensen_shannon_divergence(baseline, scenario_prediction)
        top1_change_rate = float(np.mean(np.argmax(scenario_prediction, axis=1) != baseline_top1))
        mean_probability_delta = float(np.mean(np.abs(baseline - scenario_prediction)))

        top_delta_index = int(np.argmax(np.abs(mean_delta)))
        summary_rows.append(
            {
                "scenario": scenario_name,
                "rows_evaluated": int(len(reference_frame)),
                "mean_js_divergence": round(float(np.mean(js_values)), 6),
                "top1_change_rate": round(top1_change_rate, 6),
                "mean_probability_delta": round(mean_probability_delta, 6),
                "baseline_top_crop": display_crop_name(context.label_columns[int(np.argmax(mean_baseline))]),
                "scenario_top_crop": display_crop_name(context.label_columns[int(np.argmax(mean_scenario))]),
                "largest_shift_crop": display_crop_name(context.label_columns[top_delta_index]),
                "largest_mean_delta": round(float(mean_delta[top_delta_index]), 6),
            }
        )

        for class_index, label in enumerate(context.label_columns):
            scenario_rows.append(
                {
                    "scenario": scenario_name,
                    "crop": display_crop_name(label),
                    "label_column": label,
                    "mean_baseline_probability": float(mean_baseline[class_index]),
                    "mean_scenario_probability": float(mean_scenario[class_index]),
                    "mean_delta": float(mean_delta[class_index]),
                    "abs_mean_delta": float(abs(mean_delta[class_index])),
                    "mean_js_divergence": float(np.mean(js_values)),
                    "top1_change_rate": top1_change_rate,
                    "rows_evaluated": int(len(reference_frame)),
                }
            )

        scenario_payload = build_ranking_payload(mean_scenario, context.label_columns, top_n=5)
        feature_changes = build_feature_change_payload(reference_frame, scenario_frame, scenario_cfg)
        explanation = generate_scenario_explanation(
            baseline=baseline_payload,
            scenario=scenario_payload,
            scenario_name=scenario_name,
            feature_changes=feature_changes,
        )
        ui_payload = format_for_ui({"scenario_name": scenario_name, **explanation})
        explanation_records.append(
            {
                "scenario_name": scenario_name,
                "baseline": baseline_payload,
                "scenario": scenario_payload,
                "feature_changes": feature_changes,
                "metrics": summary_rows[-1],
                "explanation": explanation,
                "ui": ui_payload,
            }
        )

    scenario_df = pd.DataFrame(scenario_rows).sort_values(
        ["scenario", "abs_mean_delta", "crop"],
        ascending=[True, False, True],
    )
    scenario_summary_df = pd.DataFrame(summary_rows)

    scenario_dir = context.output_dirs["scenario"]
    tables_dir = context.output_dirs["tables"]
    write_dataframe(scenario_dir / "scenario_results.csv", scenario_df)
    write_dataframe(tables_dir / "scenario_summary.csv", scenario_summary_df)
    write_json(
        scenario_dir / "scenario_explanations.json",
        {
            "model_version": context.model_version,
            "artifact_dir": str(context.artifact_dir),
            "scenario_count": len(explanation_records),
            "scenarios": explanation_records,
        },
    )
    markdown_path = scenario_dir / "scenario_explanations.md"
    ensure_parent_dir(markdown_path)
    markdown_path.write_text(render_explanations_markdown(explanation_records), encoding="utf-8")
    create_scenario_comparison_plot(scenario_df, scenario_dir / "scenario_comparison.png")

    print(f"Saved scenario analysis to {scenario_dir}")
    return {
        "summary_rows": summary_rows,
        "scenario_df": scenario_df,
        "summary_df": scenario_summary_df,
        "explanations": explanation_records,
    }


def run_shap(context: EvaluationContext, base_predictions: np.ndarray | None = None) -> dict[str, Any]:
    if str(context.backend).casefold() != "xgboost":
        raise ValueError(f"SHAP evaluation requires an XGBoost model, found backend={context.backend}")

    try:
        import shap
    except ImportError as exc:
        raise ImportError("SHAP is required to run explainability evaluation.") from exc

    sample_frame = context.test_frame.head(min(SHAP_MAX_ROWS, len(context.test_frame))).copy().reset_index(drop=True)
    y_true = normalize_probability_matrix(sample_frame[context.label_columns].to_numpy(dtype=float))
    predictions = normalize_probability_matrix(
        base_predictions[: len(sample_frame)] if base_predictions is not None else context.model_bundle.predict(sample_frame)
    )
    dominant_label_index = int(np.bincount(np.argmax(y_true, axis=1), minlength=len(context.label_columns)).argmax())
    dominant_label = context.label_columns[dominant_label_index]
    estimator = estimator_for_label(context.model_bundle, dominant_label_index)

    transformed = context.model_bundle.preprocessor.transform(sample_frame)
    feature_names = list(
        getattr(context.model_bundle.preprocessor, "feature_names_out", None)
        or context.feature_config.get("preprocessor", {}).get("transformed_feature_names", [])
    )
    transformed_df = pd.DataFrame(transformed, columns=feature_names)

    explainer = shap.TreeExplainer(estimator)
    shap_values = np.asarray(explainer.shap_values(transformed_df), dtype=float)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)
    if shap_values.shape[0] != len(transformed_df):
        shap_values = np.asarray(shap_values).reshape(len(transformed_df), -1)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs_shap,
            }
        )
        .sort_values(["mean_abs_shap", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )

    shap_dir = context.output_dirs["shap"]
    tables_dir = context.output_dirs["tables"]
    write_dataframe(tables_dir / "shap_feature_importance.csv", shap_importance_df)

    plt.close("all")
    shap.summary_plot(shap_values, transformed_df, show=False, max_display=SHAP_MAX_DISPLAY)
    summary_figure = plt.gcf()
    summary_figure.set_size_inches(10, 6)
    plt.title(f"SHAP Summary Plot for {display_crop_name(dominant_label)}")
    plt.tight_layout()
    summary_figure.savefig(shap_dir / "shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close(summary_figure)

    plt.close("all")
    shap.summary_plot(shap_values, transformed_df, plot_type="bar", show=False, max_display=SHAP_MAX_DISPLAY)
    bar_figure = plt.gcf()
    bar_figure.set_size_inches(10, 6)
    plt.title(f"Global SHAP Importance for {display_crop_name(dominant_label)}")
    plt.tight_layout()
    bar_figure.savefig(shap_dir / "shap_bar.png", dpi=300, bbox_inches="tight")
    plt.close(bar_figure)

    force_row_index = int(np.argmax(predictions[:, dominant_label_index]))
    save_shap_force_plot(
        shap_module=shap,
        explainer=explainer,
        shap_values=shap_values,
        transformed_df=transformed_df,
        feature_names=feature_names,
        row_index=force_row_index,
        title=f"SHAP Force Plot for {display_crop_name(dominant_label)}",
        output_path=shap_dir / "shap_force.png",
    )

    summary = {
        "explained_label": dominant_label,
        "explained_crop": display_crop_name(dominant_label),
        "rows_used": int(len(sample_frame)),
        "feature_count": int(len(feature_names)),
        "top_features": shap_importance_df.head(10).to_dict(orient="records"),
    }
    print(f"Saved SHAP artifacts to {shap_dir}")
    return summary


def write_summary_artifacts(
    context: EvaluationContext,
    performance: dict[str, Any],
    stability: dict[str, Any],
    scenario: dict[str, Any],
    shap_summary: dict[str, Any],
) -> dict[str, Any]:
    summaries_dir = context.output_dirs["summaries"]

    final_metric_rows = []
    for metric_name, metric_value in performance["metrics"].items():
        final_metric_rows.append({"section": "performance", "metric": metric_name, "value": metric_value})
    for metric_name, metric_value in stability["summary"].items():
        final_metric_rows.append({"section": "stability", "metric": metric_name, "value": metric_value})
    for row in scenario["summary_rows"]:
        scenario_name = row["scenario"]
        for key, value in row.items():
            if key == "scenario":
                continue
            final_metric_rows.append({"section": f"scenario:{scenario_name}", "metric": key, "value": value})
    final_metric_rows.append({"section": "shap", "metric": "explained_crop", "value": shap_summary["explained_crop"]})
    final_metric_rows.append({"section": "shap", "metric": "rows_used", "value": shap_summary["rows_used"]})

    final_metrics_df = pd.DataFrame(final_metric_rows)
    final_metrics_path = summaries_dir / "final_metrics_table.csv"
    write_dataframe(final_metrics_path, final_metrics_df)

    report_lines = [
        "# Evaluation Summary",
        "",
        f"- Model version: `{context.model_version}`",
        f"- Artifact directory: `{context.artifact_dir}`",
        f"- Dataset path: `{context.dataset_path}`",
        f"- Test rows evaluated: `{len(context.test_frame)}`",
        f"- Backend: `{context.backend}`",
        "",
        "## Performance",
    ]
    for metric_name, metric_value in performance["metrics"].items():
        report_lines.append(f"- {metric_name}: `{metric_value:.6f}`")
    report_lines.extend(["", "## Stability"])
    for metric_name, metric_value in stability["summary"].items():
        if metric_name in {"rows_evaluated", "noise_fraction"}:
            report_lines.append(f"- {metric_name.replace('_', ' ').title()}: `{metric_value}`")
        else:
            report_lines.append(f"- {metric_name.replace('_', ' ').title()}: `{float(metric_value):.6f}`")
    report_lines.extend(["", "## Scenario Analysis"])
    for row in scenario["summary_rows"]:
        report_lines.append(
            "- "
            + f"{row['scenario']}: mean_js_divergence=`{row['mean_js_divergence']:.6f}`, "
            + f"top1_change_rate=`{row['top1_change_rate']:.6f}`, "
            + f"largest_shift_crop=`{row['largest_shift_crop']}`, "
            + f"largest_mean_delta=`{row['largest_mean_delta']:.6f}`"
        )
    report_lines.extend(
        [
            "",
            "## SHAP",
            f"- Explained crop: `{shap_summary['explained_crop']}`",
            f"- Rows used: `{shap_summary['rows_used']}`",
            "- Top features:",
        ]
    )
    for item in shap_summary["top_features"][:5]:
        report_lines.append(f"  - {item['feature']}: `{float(item['mean_abs_shap']):.6f}`")
    report_lines.extend(
        [
            "",
            "## Generated Files",
            "- `artifacts/evaluation/performance/metrics.json`",
            "- `artifacts/evaluation/performance/metrics.csv`",
            "- `artifacts/evaluation/performance/performance_bar_chart.png`",
            "- `artifacts/evaluation/performance/classwise_accuracy.png`",
            "- `artifacts/evaluation/stability/stability_metrics.csv`",
            "- `artifacts/evaluation/stability/stability_summary.json`",
            "- `artifacts/evaluation/stability/stability_histogram.png`",
            "- `artifacts/evaluation/scenario/scenario_results.csv`",
            "- `artifacts/evaluation/scenario/scenario_comparison.png`",
            "- `artifacts/evaluation/scenario/scenario_explanations.json`",
            "- `artifacts/evaluation/scenario/scenario_explanations.md`",
            "- `artifacts/evaluation/shap/shap_summary.png`",
            "- `artifacts/evaluation/shap/shap_bar.png`",
            "- `artifacts/evaluation/shap/shap_force.png`",
            "- `artifacts/evaluation/summaries/report_summary.md`",
            "- `artifacts/evaluation/summaries/final_metrics_table.csv`",
        ]
    )

    report_path = summaries_dir / "report_summary.md"
    ensure_parent_dir(report_path)
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    return {
        "report_summary": str(report_path),
        "final_metrics_table": str(final_metrics_path),
    }


def resolve_artifact_dir(root_dir: Path, artifact_dir: str | Path | None) -> Path:
    if artifact_dir is not None:
        return resolve_path(root_dir, str(artifact_dir))

    registry_records = load_jsonl(root_dir / "artifacts" / "registry" / "model_registry.jsonl")
    for record in reversed(registry_records):
        candidate_dir = Path(str(record.get("artifact_dir", "")))
        if candidate_dir.name in ALLOWED_REGISTRY_ARTIFACTS and candidate_dir.exists():
            return candidate_dir

    for artifact_name in ALLOWED_REGISTRY_ARTIFACTS:
        candidate_dir = root_dir / "artifacts" / artifact_name
        if candidate_dir.exists():
            return candidate_dir.resolve()

    for record in reversed(registry_records):
        candidate_dir = Path(str(record.get("artifact_dir", "")))
        if candidate_dir.exists():
            return candidate_dir

    raise FileNotFoundError("Unable to resolve an artifact directory from artifacts/registry or local fallback directories.")


def resolve_dataset_path(root_dir: Path, feature_config: dict[str, Any], evaluation_report: dict[str, Any]) -> Path:
    candidate_paths = [
        feature_config.get("training_metadata", {}).get("dataset_profile", {}).get("dataset_path"),
        evaluation_report.get("dataset_profile", {}).get("dataset_path"),
    ]
    for raw_path in candidate_paths:
        if raw_path:
            candidate = resolve_path(root_dir, str(raw_path))
            if candidate.exists():
                return candidate
    raise FileNotFoundError("Unable to locate the dataset path from the saved artifact metadata.")


def split_dataset_frame(
    frame: pd.DataFrame,
    feature_config: dict[str, Any],
    evaluation_report: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training_metadata = feature_config.get("training_metadata", {})
    split_cfg = evaluation_report.get("split", {})
    train_times = [str(value) for value in training_metadata.get("train_times", split_cfg.get("train_times", []))]
    validation_times = [str(value) for value in training_metadata.get("validation_times", split_cfg.get("validation_times", []))]
    test_times = [str(value) for value in training_metadata.get("test_times", split_cfg.get("test_times", []))]

    if "time" not in frame.columns:
        raise ValueError("Dataset is missing the required 'time' column needed to rebuild the test split.")

    working = frame.copy()
    working["time"] = working["time"].astype(str)
    working["_evaluation_time"] = pd.to_datetime(working["time"].astype(str) + "-01", errors="coerce")

    if not test_times:
        test_frame = sort_evaluation_frame(working).reset_index(drop=True)
        return (
            pd.DataFrame(columns=frame.columns),
            pd.DataFrame(columns=frame.columns),
            test_frame.drop(columns=["_evaluation_time"], errors="ignore"),
        )

    train_frame = working[working["time"].isin(train_times)].copy()
    validation_frame = working[working["time"].isin(validation_times)].copy()
    test_frame = working[working["time"].isin(test_times)].copy()

    return (
        sort_evaluation_frame(train_frame).reset_index(drop=True).drop(columns=["_evaluation_time"], errors="ignore"),
        sort_evaluation_frame(validation_frame).reset_index(drop=True).drop(columns=["_evaluation_time"], errors="ignore"),
        sort_evaluation_frame(test_frame).reset_index(drop=True).drop(columns=["_evaluation_time"], errors="ignore"),
    )


def build_ranking_payload(probabilities: np.ndarray, label_columns: list[str], top_n: int = 5) -> dict[str, Any]:
    vector = np.asarray(probabilities, dtype=float).reshape(-1)
    ranked_indices = np.argsort(vector)[::-1]
    top_indices = ranked_indices[: min(top_n, len(ranked_indices))]
    ranking = [display_crop_name(label_columns[int(index)]) for index in top_indices]
    scores = {
        display_crop_name(label_columns[int(index)]): round(float(vector[int(index)]), 6)
        for index in top_indices
    }
    return {
        "top_crop": ranking[0] if ranking else "",
        "ranking": ranking,
        "scores": scores,
    }


def build_feature_change_payload(
    baseline_frame: pd.DataFrame,
    scenario_frame: pd.DataFrame,
    scenario_cfg: dict[str, dict[str, float]],
) -> dict[str, str]:
    changed_features = []
    changed_features.extend(list(scenario_cfg.get("feature_multipliers", {}).keys()))
    changed_features.extend(list(scenario_cfg.get("feature_additions", {}).keys()))
    feature_changes: dict[str, str] = {}

    for feature in dict.fromkeys(changed_features):
        baseline_mean = float(pd.to_numeric(baseline_frame.get(feature), errors="coerce").mean())
        scenario_mean = float(pd.to_numeric(scenario_frame.get(feature), errors="coerce").mean())
        if feature in scenario_cfg.get("feature_multipliers", {}):
            multiplier = float(scenario_cfg["feature_multipliers"][feature])
            pct_change = (multiplier - 1.0) * 100.0
            feature_changes[feature] = (
                f"{pct_change:+.1f}% ({baseline_mean:.3f} -> {scenario_mean:.3f})"
            )
        elif feature in scenario_cfg.get("feature_additions", {}):
            addition = float(scenario_cfg["feature_additions"][feature])
            unit = " C" if "temp" in feature or "heat" in feature else ""
            feature_changes[feature] = (
                f"{addition:+.2f}{unit} ({baseline_mean:.3f} -> {scenario_mean:.3f})"
            )
        else:
            delta = scenario_mean - baseline_mean
            feature_changes[feature] = f"{delta:+.3f} ({baseline_mean:.3f} -> {scenario_mean:.3f})"
    return feature_changes


def ensure_output_dirs(base_dir: Path) -> dict[str, Path]:
    directories = {name: base_dir / name for name in OUTPUT_SUBDIRS}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    return directories


def ensure_runtime_import_path(root_dir: Path) -> None:
    src_dir = (root_dir / "src").resolve()
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def sort_evaluation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    sort_columns = []
    if "_evaluation_time" in frame.columns:
        sort_columns.append("_evaluation_time")
    if "region_key" in frame.columns:
        sort_columns.append("region_key")
    elif "region" in frame.columns:
        sort_columns.append("region")
    if sort_columns:
        return frame.sort_values(sort_columns)
    return frame.copy()


def perturb_numeric_inputs(
    frame: pd.DataFrame,
    numeric_features: list[str],
    noise_fraction: float,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    perturbed = frame.copy()
    for feature in numeric_features:
        if feature not in perturbed.columns:
            continue
        base_values = pd.to_numeric(perturbed[feature], errors="coerce").fillna(0.0).astype(float)
        noise = rng.uniform(-noise_fraction, noise_fraction, size=len(base_values))
        perturbed[feature] = base_values * (1.0 + noise)
    return clip_numeric_features(perturbed, numeric_features)


def clip_numeric_features(frame: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    clipped = frame.copy()
    for feature in numeric_features:
        if feature not in clipped.columns:
            continue
        values = pd.to_numeric(clipped[feature], errors="coerce").astype(float)
        if feature in BOUNDED_ZERO_ONE_FEATURES:
            values = values.clip(lower=0.0, upper=1.0)
        if feature in NON_NEGATIVE_FEATURES:
            values = values.clip(lower=0.0)
        if feature == "humidity_avg":
            values = values.clip(lower=0.0, upper=100.0)
        if feature == "pH":
            values = values.clip(lower=0.0, upper=14.0)
        if feature in INTEGER_LIKE_FEATURES:
            values = values.round()
        clipped[feature] = values
    return clipped


def create_metric_bar_chart(metrics_df: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(metrics_df["metric"], metrics_df["value"])
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Score")
    axis.set_title("Overall Predictive Performance on the Held-Out Test Set")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def create_classwise_chart(classwise_df: pd.DataFrame, output_path: Path) -> None:
    display_df = classwise_df[classwise_df["support"] > 0].sort_values(
        ["accuracy", "support", "class_name"],
        ascending=[True, True, False],
    )
    height = max(5, 0.3 * len(display_df))
    figure, axis = plt.subplots(figsize=(10, height))
    axis.barh(display_df["class_name"], display_df["accuracy"])
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Accuracy")
    axis.set_ylabel("Crop Class")
    axis.set_title("Class-Wise Top-1 Accuracy on the Held-Out Test Set")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def create_stability_histogram(js_series: pd.Series, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(js_series.to_numpy(dtype=float), bins=30)
    axis.set_xlabel("Jensen-Shannon Divergence")
    axis.set_ylabel("Row Count")
    axis.set_title("Prediction Stability Under +/-3% Input Perturbations")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def create_scenario_comparison_plot(scenario_df: pd.DataFrame, output_path: Path) -> None:
    scenarios = scenario_df["scenario"].drop_duplicates().tolist()
    figure, axes = plt.subplots(len(scenarios), 1, figsize=(11, max(4, 3.5 * len(scenarios))), squeeze=False)

    for index, scenario_name in enumerate(scenarios):
        axis = axes[index, 0]
        subset = (
            scenario_df[scenario_df["scenario"] == scenario_name]
            .sort_values(["abs_mean_delta", "crop"], ascending=[False, True])
            .head(10)
            .sort_values("mean_delta", ascending=True)
        )
        axis.barh(subset["crop"], subset["mean_delta"])
        axis.axvline(0.0, linewidth=0.8)
        axis.set_xlabel("Mean Probability Shift")
        axis.set_ylabel("Crop")
        axis.set_title(f"Prediction Shift Under the {scenario_name.replace('_', ' ').title()} Scenario")

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def save_shap_force_plot(
    shap_module: Any,
    explainer: Any,
    shap_values: np.ndarray,
    transformed_df: pd.DataFrame,
    feature_names: list[str],
    row_index: int,
    title: str,
    output_path: Path,
) -> None:
    base_value = getattr(explainer, "expected_value", 0.0)
    base_value = float(np.asarray(base_value).reshape(-1)[0])
    feature_row = transformed_df.iloc[row_index]

    plt.close("all")
    try:
        shap_module.force_plot(
            base_value,
            shap_values[row_index],
            feature_row,
            feature_names=feature_names,
            matplotlib=True,
            show=False,
        )
    except Exception:
        explanation = shap_module.Explanation(
            values=shap_values[row_index],
            base_values=base_value,
            data=feature_row.to_numpy(dtype=float),
            feature_names=feature_names,
        )
        shap_module.plots.force(explanation, matplotlib=True, show=False)

    figure = plt.gcf()
    figure.set_size_inches(12, 3)
    plt.title(title)
    plt.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def estimator_for_label(model_bundle: CropSuitabilityModelBundle, label_index: int) -> Any:
    label_name = model_bundle.label_columns[label_index]
    if model_bundle.models and label_name in model_bundle.models:
        return model_bundle.models[label_name]
    if model_bundle.model is not None and hasattr(model_bundle.model, "estimators_"):
        return model_bundle.model.estimators_[label_index]
    raise ValueError(f"Unable to resolve the estimator for label index {label_index}")


def top_k_indices(probabilities: np.ndarray, top_k: int) -> np.ndarray:
    k = min(int(top_k), probabilities.shape[1])
    return np.argsort(probabilities, axis=1)[:, -k:][:, ::-1]


def display_crop_name(label: str) -> str:
    return crop_name_from_label(label).replace("_", " ").title()


def write_dataframe(path: Path, frame: pd.DataFrame) -> None:
    ensure_parent_dir(path)
    frame.to_csv(path, index=False)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records
