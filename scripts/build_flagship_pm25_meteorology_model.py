#!/usr/bin/env python3
"""Build the flagship PM2.5 meteorology attribution model entry point.

The flagship model is a unified routing estimator over the best existing
period-specific v2 weather-only attribution models. It does not retrain models;
it wraps the selected submodels into one documented model object for downstream
research use.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.flagship_pm25_meteorology_model import FlagshipPM25MeteorologyModel

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
DATA_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_DATA = DATA_DIR / "bth_pm25_meteorology_attribution_v2_features.parquet"
DEFAULT_DATA_METADATA = DATA_DIR / "bth_pm25_meteorology_attribution_v2_features.metadata.json"
DEFAULT_FLAGSHIP_MODEL = MODEL_DIR / "flagship_pm25_meteorology_attribution.joblib"
DEFAULT_METRICS_CSV = REPORT_DIR / "flagship_pm25_meteorology_model_metrics.csv"
DEFAULT_MANIFEST_JSON = REPORT_DIR / "flagship_pm25_meteorology_model_manifest.json"
DEFAULT_SUMMARY_MD = REPORT_DIR / "flagship_pm25_meteorology_model_summary.md"
DEFAULT_PREDICTION_OUTPUT = REPORT_DIR / "flagship_pm25_meteorology_predictions.csv"
DEFAULT_CHEMICAL_MODEL_DIR = MODEL_DIR / "chemical_composition"
DEFAULT_CHEMICAL_REPORT_DIR = REPORT_DIR / "chemical_composition"
DEFAULT_CHEMICAL_RUN_PREFIX = "chemical_composition_eac4_main"
DEFAULT_CHEMICAL_RESULTS_JSON = PROJECT_ROOT / "app_assets" / "chemical_composition_results.json"

PERIOD_LABELS = {
    "pre_covid_2018_2019": "疫情前",
    "covid_2020_2022": "疫情期",
    "post_covid_2023_plus": "疫情后",
}

TARGET_LABELS = {
    "raw": "PM2.5 原值",
    "log1p": "log1p(PM2.5)",
    "anomaly": "同城同月同小时气候态异常",
}

CHEMICAL_TARGETS = (
    "sulfate",
    "nitrate",
    "ammonium",
    "sna",
    "sna_fraction",
    "black_carbon",
    "organic_matter",
    "secondary_fraction",
    "nitrate_sulfate_ratio",
)

CHEMICAL_TARGET_LABELS = {
    "sulfate": "硫酸盐 sulfate",
    "nitrate": "硝酸盐 nitrate",
    "ammonium": "铵盐 ammonium",
    "sna": "SNA 二次无机组分",
    "sna_fraction": "SNA 占比",
    "black_carbon": "黑碳 black carbon",
    "organic_matter": "有机质 organic matter",
    "secondary_fraction": "二次组分占比",
    "nitrate_sulfate_ratio": "硝酸盐/硫酸盐比值",
}

CHEMICAL_TARGET_KINDS = {
    "sulfate": "concentration",
    "nitrate": "concentration",
    "ammonium": "concentration",
    "sna": "concentration",
    "black_carbon": "concentration",
    "organic_matter": "concentration",
    "sna_fraction": "fraction",
    "secondary_fraction": "fraction",
    "nitrate_sulfate_ratio": "ratio",
}

CHEMICAL_TARGET_UNITS = {
    "concentration": "ug/m3",
    "fraction": "fraction",
    "ratio": "ratio",
}


@dataclass(frozen=True)
class PeriodSelection:
    period: str
    target_kind: str
    model_path: Path
    metrics_path: Path
    reason: str

    @property
    def period_label(self) -> str:
        return PERIOD_LABELS.get(self.period, self.period)

    @property
    def target_label(self) -> str:
        return TARGET_LABELS.get(self.target_kind, self.target_kind)


@dataclass(frozen=True)
class ChemicalSelection:
    target: str
    model_path: Path
    metrics_path: Path
    reason: str

    @property
    def target_label(self) -> str:
        return CHEMICAL_TARGET_LABELS.get(self.target, self.target)

    @property
    def target_kind(self) -> str:
        return CHEMICAL_TARGET_KINDS.get(self.target, "diagnostic")

    @property
    def value_unit(self) -> str:
        return CHEMICAL_TARGET_UNITS.get(self.target_kind, "")


DEFAULT_SELECTIONS = (
    PeriodSelection(
        period="pre_covid_2018_2019",
        target_kind="log1p",
        model_path=MODEL_DIR / "meteorology_attribution_v2_core_pre_covid_2018_2019_log1p.joblib",
        metrics_path=REPORT_DIR / "meteorology_attribution_v2_core_pre_covid_2018_2019_log1p_metrics.json",
        reason="疫情前污染浓度分布偏右尾，log1p 目标在稳定误差和保留机制解释之间更均衡。",
    ),
    PeriodSelection(
        period="covid_2020_2022",
        target_kind="raw",
        model_path=MODEL_DIR / "meteorology_attribution_v2_core_covid_2020_2022_raw.joblib",
        metrics_path=REPORT_DIR / "meteorology_attribution_v2_core_covid_2020_2022_raw_metrics.json",
        reason="疫情期排放扰动较强，原值目标保留绝对浓度尺度，测试集误差最低且偏差较小。",
    ),
    PeriodSelection(
        period="post_covid_2023_plus",
        target_kind="anomaly",
        model_path=MODEL_DIR / "meteorology_attribution_v2_core_post_covid_2023_plus_anomaly.joblib",
        metrics_path=REPORT_DIR / "meteorology_attribution_v2_core_post_covid_2023_plus_anomaly_metrics.json",
        reason="疫情后长期趋势更明显，异常目标更适合剥离同城同月同小时气候态并解释气象扰动。",
    ),
)

PERIOD_ORDER = {selection.period: index for index, selection in enumerate(DEFAULT_SELECTIONS)}


def make_chemical_selections(
    model_dir: Path,
    report_dir: Path,
    run_prefix: str,
    targets: tuple[str, ...] = CHEMICAL_TARGETS,
) -> tuple[ChemicalSelection, ...]:
    return tuple(
        ChemicalSelection(
            target=target,
            model_path=model_dir / f"{run_prefix}_{target}_precursor_all.joblib",
            metrics_path=report_dir / f"{run_prefix}_{target}_precursor_all_metrics.json",
            reason="作为旗舰主模型的化学机制诊断头，使用气象、城市/时间控制和 O3/NO2/SO2/CO 前体物辅助信息。",
        )
        for target in targets
    )


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def file_sha256(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(block_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-model", default=str(DEFAULT_FLAGSHIP_MODEL))
    parser.add_argument("--metrics-output", default=str(DEFAULT_METRICS_CSV))
    parser.add_argument("--manifest-output", default=str(DEFAULT_MANIFEST_JSON))
    parser.add_argument("--summary-output", default=str(DEFAULT_SUMMARY_MD))
    parser.add_argument("--source-data", default=str(DEFAULT_DATA))
    parser.add_argument("--source-data-metadata", default=str(DEFAULT_DATA_METADATA))
    parser.add_argument("--chemical-model-dir", default=str(DEFAULT_CHEMICAL_MODEL_DIR))
    parser.add_argument("--chemical-report-dir", default=str(DEFAULT_CHEMICAL_REPORT_DIR))
    parser.add_argument("--chemical-run-prefix", default=DEFAULT_CHEMICAL_RUN_PREFIX)
    parser.add_argument("--chemical-results-json", default=str(DEFAULT_CHEMICAL_RESULTS_JSON))
    parser.add_argument(
        "--no-chemical-models",
        action="store_true",
        help="Do not package the chemical-composition diagnostic head into the flagship model.",
    )
    parser.add_argument("--predict-data", help="Optional CSV/Parquet feature table to score with the flagship model.")
    parser.add_argument("--prediction-output", default=str(DEFAULT_PREDICTION_OUTPUT))
    parser.add_argument(
        "--allow-anomaly-target-only",
        action="store_true",
        help="For anomaly rows without baseline_pm2_5, keep predicted_pm2_5 empty instead of failing.",
    )
    return parser.parse_args()


def validate_payload(selection: PeriodSelection, payload: dict[str, Any], metrics: dict[str, Any]) -> None:
    required_keys = {"preprocessor", "model", "features", "target_kind", "period", "target_meta"}
    missing = required_keys.difference(payload)
    if missing:
        raise ValueError(f"{rel(selection.model_path)} is missing keys: {sorted(missing)}")
    if payload["period"] != selection.period:
        raise ValueError(f"{rel(selection.model_path)} period mismatch: {payload['period']} != {selection.period}")
    if payload["target_kind"] != selection.target_kind:
        raise ValueError(
            f"{rel(selection.model_path)} target mismatch: {payload['target_kind']} != {selection.target_kind}"
        )
    if metrics.get("period") != selection.period:
        raise ValueError(f"{rel(selection.metrics_path)} period mismatch: {metrics.get('period')} != {selection.period}")
    if metrics.get("target_kind") != selection.target_kind:
        raise ValueError(
            f"{rel(selection.metrics_path)} target mismatch: {metrics.get('target_kind')} != {selection.target_kind}"
        )
    if len(payload.get("features", [])) != int(metrics.get("feature_count", len(payload.get("features", [])))):
        raise ValueError(f"{rel(selection.model_path)} feature count does not match its metrics JSON.")


def load_selection(selection: PeriodSelection) -> tuple[dict[str, Any], dict[str, Any]]:
    if not selection.model_path.exists():
        raise FileNotFoundError(f"Missing model: {selection.model_path}")
    if not selection.metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics: {selection.metrics_path}")
    payload = joblib.load(selection.model_path)
    metrics = read_json(selection.metrics_path)
    validate_payload(selection, payload, metrics)
    return payload, metrics


def validate_chemical_payload(selection: ChemicalSelection, payload: dict[str, Any], metrics: dict[str, Any]) -> None:
    required_keys = {"preprocessor", "model", "features"}
    missing = required_keys.difference(payload)
    if missing:
        raise ValueError(f"{rel(selection.model_path)} is missing keys: {sorted(missing)}")
    if metrics.get("target") != selection.target:
        raise ValueError(
            f"{rel(selection.metrics_path)} target mismatch: {metrics.get('target')} != {selection.target}"
        )
    if metrics.get("feature_set") != "precursor":
        raise ValueError(f"{rel(selection.metrics_path)} is not a precursor-assisted chemical model.")
    if metrics.get("period") != "all":
        raise ValueError(f"{rel(selection.metrics_path)} chemical period mismatch: {metrics.get('period')} != all")
    metrics_features = metrics.get("features", len(payload.get("features", [])))
    metrics_feature_count = len(metrics_features) if isinstance(metrics_features, list) else int(metrics_features)
    if len(payload.get("features", [])) != metrics_feature_count:
        raise ValueError(f"{rel(selection.model_path)} feature count does not match its metrics JSON.")


def load_chemical_selection(selection: ChemicalSelection) -> tuple[dict[str, Any], dict[str, Any]]:
    if not selection.model_path.exists():
        raise FileNotFoundError(f"Missing chemical model: {selection.model_path}")
    if not selection.metrics_path.exists():
        raise FileNotFoundError(f"Missing chemical metrics: {selection.metrics_path}")
    payload = joblib.load(selection.model_path)
    metrics = read_json(selection.metrics_path)
    validate_chemical_payload(selection, payload, metrics)
    return payload, metrics


def load_chemical_results_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    data = read_json(path)
    return {
        "title": data.get("title"),
        "dataset": data.get("dataset"),
        "time_min": data.get("time_min"),
        "time_max": data.get("time_max"),
        "valid_start": data.get("valid_start"),
        "test_start": data.get("test_start"),
        "target_count": data.get("target_count"),
        "model_count": data.get("model_count"),
        "feature_sets": data.get("feature_sets", {}),
        "headline": data.get("headline", {}),
        "source_path": rel(path),
        "source_sha256": file_sha256(path),
    }


def load_chemical_models(args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    if args.no_chemical_models:
        return {}, pd.DataFrame(), {}

    selections = make_chemical_selections(
        Path(args.chemical_model_dir),
        Path(args.chemical_report_dir),
        str(args.chemical_run_prefix),
    )
    loaded: dict[str, dict[str, Any]] = {}
    metric_rows = []
    for selection in selections:
        payload, metrics = load_chemical_selection(selection)
        test = metrics.get("metrics", {})
        blocked_cv = metrics.get("blocked_time_cv", {})
        package = {
            "target": selection.target,
            "target_label": selection.target_label,
            "target_kind": selection.target_kind,
            "value_unit": selection.value_unit,
            "feature_set": metrics.get("feature_set", "precursor"),
            "target_transform": metrics.get("target_transform", "identity"),
            "reason": selection.reason,
            "source_model_path": rel(selection.model_path),
            "source_model_sha256": file_sha256(selection.model_path),
            "source_metrics_path": rel(selection.metrics_path),
            "source_metrics_sha256": file_sha256(selection.metrics_path),
            "metrics": metrics,
            "model_payload": payload,
        }
        loaded[selection.target] = package
        metric_rows.append(
            {
                "target": selection.target,
                "target_label": selection.target_label,
                "target_kind": selection.target_kind,
                "value_unit": selection.value_unit,
                "feature_set": metrics.get("feature_set", "precursor"),
                "target_transform": metrics.get("target_transform", "identity"),
                "feature_count": int(len(payload["features"])),
                "rows": metrics.get("rows"),
                "train_rows": metrics.get("train_rows"),
                "valid_rows": metrics.get("valid_rows"),
                "test_rows": metrics.get("test_rows"),
                "test_mae": test.get("mae"),
                "test_rmse": test.get("rmse"),
                "test_r2": test.get("r2"),
                "blocked_cv_mean_r2": blocked_cv.get("mean_r2"),
                "model_path": rel(selection.model_path),
                "metrics_path": rel(selection.metrics_path),
                "selection_reason": selection.reason,
            }
        )

    summary = load_chemical_results_summary(Path(args.chemical_results_json))
    summary.update(
        {
            "packaged_targets": list(loaded),
            "model_dir": rel(Path(args.chemical_model_dir)),
            "report_dir": rel(Path(args.chemical_report_dir)),
            "run_prefix": str(args.chemical_run_prefix),
        }
    )
    return loaded, pd.DataFrame(metric_rows), summary


def compact_metrics_row(selection: PeriodSelection, payload: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    test = metrics.get("test_pm25", {})
    validation = metrics.get("validation_pm25", {})
    blocked_cv = metrics.get("blocked_time_cv", {})
    return {
        "period": selection.period,
        "period_label": selection.period_label,
        "selected_target_kind": selection.target_kind,
        "selected_target_label": selection.target_label,
        "feature_count": int(len(payload["features"])),
        "transformed_feature_count": metrics.get("transformed_feature_count"),
        "train_rows": metrics.get("train_rows"),
        "valid_rows": metrics.get("valid_rows"),
        "train_valid_rows": metrics.get("train_valid_rows"),
        "test_rows": metrics.get("test_rows"),
        "valid_start": metrics.get("valid_start"),
        "test_start": metrics.get("test_start"),
        "test_mae": test.get("mae"),
        "test_rmse": test.get("rmse"),
        "test_r2": test.get("r2"),
        "test_bias": test.get("bias"),
        "validation_mae": validation.get("mae"),
        "validation_rmse": validation.get("rmse"),
        "validation_r2": validation.get("r2"),
        "blocked_cv_folds_completed": blocked_cv.get("folds_completed"),
        "blocked_cv_mean_mae": blocked_cv.get("mean_mae"),
        "blocked_cv_mean_rmse": blocked_cv.get("mean_rmse"),
        "blocked_cv_mean_r2": blocked_cv.get("mean_r2"),
        "model_path": rel(selection.model_path),
        "metrics_path": rel(selection.metrics_path),
        "selection_reason": selection.reason,
    }


def load_data_metadata(data_path: Path, metadata_path: Path) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = read_json(metadata_path)
    metadata.update(
        {
            "path": rel(data_path),
            "exists": data_path.exists(),
            "sha256": file_sha256(data_path) if data_path.exists() else None,
            "metadata_path": rel(metadata_path) if metadata_path.exists() else None,
        }
    )
    return metadata


def build_flagship_artifact(args: argparse.Namespace) -> tuple[dict[str, Any], pd.DataFrame]:
    loaded: dict[str, dict[str, Any]] = {}
    metric_rows = []
    for selection in DEFAULT_SELECTIONS:
        payload, metrics = load_selection(selection)
        model_package = {
            "selection": {
                "period": selection.period,
                "period_label": selection.period_label,
                "target_kind": selection.target_kind,
                "target_label": selection.target_label,
                "reason": selection.reason,
            },
            "source_model_path": rel(selection.model_path),
            "source_model_sha256": file_sha256(selection.model_path),
            "source_metrics_path": rel(selection.metrics_path),
            "source_metrics_sha256": file_sha256(selection.metrics_path),
            "metrics": metrics,
            "model_payload": payload,
        }
        loaded[selection.period] = model_package
        metric_rows.append(compact_metrics_row(selection, payload, metrics))

    feature_sets = [set(package["model_payload"]["features"]) for package in loaded.values()]
    common_features = sorted(set.intersection(*feature_sets))
    required_features = sorted(set.union(*feature_sets))
    source_data = load_data_metadata(Path(args.source_data), Path(args.source_data_metadata))
    chemical_models, chemical_metrics, chemical_summary = load_chemical_models(args)

    artifact = {
        "model_name": "flagship_pm25_meteorology_attribution",
        "model_family": "Flagship weather-only PM2.5 meteorological attribution model",
        "version": "v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "source_data": source_data,
        "period_selector": {
            period: {
                "period_label": package["selection"]["period_label"],
                "target_kind": package["selection"]["target_kind"],
                "target_label": package["selection"]["target_label"],
                "selection_reason": package["selection"]["reason"],
            }
            for period, package in loaded.items()
        },
        "period_models": loaded,
        "required_columns": sorted(set(required_features + ["period"])),
        "common_feature_count": int(len(common_features)),
        "required_feature_count": int(len(required_features)),
        "feature_policy": {
            "scope": "weather-only meteorological attribution",
            "uses_pm25_lag_or_rolling_features": False,
            "uses_co_pollutants": False,
            "uses_target_leakage_columns": False,
            "allowed_inputs": "meteorology, temporal encodings, geography, city/province, weather types, weather lags/rolling features.",
            "scientific_role": "主模型用于解释气象条件对 PM2.5 波动的贡献，不作为依赖污染物自相关的最高分预测模型。",
        },
        "prediction_contract": {
            "period_column": "Required unless the input has a time column that can be mapped to 2018-2019, 2020-2022, and 2023+ periods.",
            "output_columns": ["predicted_model_target", "predicted_pm2_5", "prediction_target_kind"],
            "anomaly_target_note": "Rows routed to the post-covid anomaly model require baseline_pm2_5 to convert anomaly predictions back to PM2.5.",
        },
        "chemical_models": chemical_models,
        "chemical_model_metadata": {
            "scientific_role": "化学组分机制诊断头，用于解释 O3/NO2/SO2/CO 前体物信息对二次组分和碳质组分的辅助刻画，不替代 PM2.5 主预测。",
            "summary": chemical_summary,
            "target_metrics": chemical_metrics.to_dict(orient="records") if not chemical_metrics.empty else [],
        },
        "chemical_prediction_contract": {
            "method": "predict_chemical_frame",
            "role": "Returns chemical-composition diagnostics from the same flagship object; predict_frame remains the PM2.5 main prediction interface.",
            "required_default_inputs": [
                "city",
                "province",
                "latitude",
                "longitude",
                "time/date or period plus time encodings",
                "meteorological variables",
                "carbon_monoxide",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "ozone",
            ],
            "custom_input_policy": "Users may override O3/NO2/SO2/CO and other feature columns in the prediction frame. Missing numeric columns are left to the packaged preprocessor imputation when available.",
            "output_targets": list(chemical_models),
            "output_units": {
                "concentration_targets": "ug/m3",
                "fraction_targets": "0-1 fraction",
                "ratio_targets": "unitless ratio",
            },
        },
        "usage_notes": [
            "Run this script to rebuild the flagship package from the selected v2 core models.",
            "Use scripts/train_meteorology_attribution_v2.py only when you need to retrain submodels.",
            "For new prediction data, build the same v2 feature table first, including weather_type_k6 and derived weather lag/rolling fields.",
            "Use predict_chemical_frame only as a chemical-composition mechanism diagnostic head; custom O3/NO2/SO2/CO fields can be supplied by user data.",
            "For scientific reporting, cite the flagship summary and the period-specific SHAP/condition/weather-type diagnostics.",
        ],
    }
    metrics = pd.DataFrame(metric_rows)
    metrics["period_order"] = metrics["period"].map(PERIOD_ORDER)
    metrics = metrics.sort_values("period_order").drop(columns=["period_order"])
    return artifact, metrics


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input table format: {path}")


def write_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(path, index=False)
    elif suffix in {".csv", ".txt"}:
        frame.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output table format: {path}")


def write_summary(path: Path, metrics: pd.DataFrame, artifact: dict[str, Any], output_model: Path) -> None:
    lines = [
        "# 旗舰 PM2.5 气象归因模型",
        "",
        "## 定位",
        "",
        "该旗舰模型不是重新训练的一套新模型，而是把现有 v2 core 中最适合支撑科研结论的分时期模型封装成一个可直接加载调用的主模型对象。",
        "主模型坚持 weather-only 归因口径：不使用 PM2.5 滞后/滚动特征、不使用共污染物、不使用目标泄漏统计量，重点解释气象条件对 PM2.5 波动的贡献。",
        "",
        "## 分期选择",
        "",
        "| 时期 | 目标 | 测试 R2 | 测试 RMSE | 测试 MAE | Bias | 阻塞CV R2 | 选择理由 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in metrics.to_dict(orient="records"):
        lines.append(
            f"| {row['period_label']} | {row['selected_target_label']} | "
            f"{row['test_r2']:.3f} | {row['test_rmse']:.2f} | {row['test_mae']:.2f} | "
            f"{row['test_bias']:.2f} | {row['blocked_cv_mean_r2']:.3f} | {row['selection_reason']} |"
        )

    chemical_rows = artifact.get("chemical_model_metadata", {}).get("target_metrics", [])
    if chemical_rows:
        lines.extend(
            [
                "",
                "## 化学组分机制诊断头",
                "",
                "同一个旗舰模型对象额外封装前体物辅助化学组分模型。该层用于机制诊断，不替代 PM2.5 主预测；预测台可用城市历史画像自动补齐 O3、NO2、SO2、CO，也可由用户输入自定义前体物数据。",
                "",
                "| 组分目标 | 类型 | 单位 | 测试 R2 | 测试 RMSE | 测试 MAE | 变换 |",
                "|---|---|---|---:|---:|---:|---|",
            ]
        )
        for row in chemical_rows:
            lines.append(
                f"| {row['target_label']} | {row['target_kind']} | {row['value_unit']} | "
                f"{row['test_r2']:.3f} | {row['test_rmse']:.3f} | {row['test_mae']:.3f} | "
                f"{row['target_transform']} |"
            )

    lines.extend(
        [
            "",
            "## 入口产物",
            "",
            f"- 旗舰模型：`{rel(output_model)}`",
            "- 模型类：`scripts.flagship_pm25_meteorology_model.FlagshipPM25MeteorologyModel`",
            f"- 指标表：`{rel(Path(artifact['reports']['metrics_csv']))}`",
            f"- Manifest：`{rel(Path(artifact['reports']['manifest_json']))}`",
            "",
            "## 调用方式",
            "",
            "```python",
            "import joblib",
            "",
            "model = joblib.load('models/flagship_pm25_meteorology_attribution.joblib')",
            "predictions = model.predict_frame(feature_table)",
            "chemical_diagnostics = model.predict_chemical_frame(feature_table)",
            "pm25_values = model.predict(feature_table)",
            "summary = model.submodel_summary()",
            "```",
            "",
            "## 使用建议",
            "",
            "- 论文或汇报中，将该模型作为主模型；高精度预测模型只作为对照基线。",
            "- 分期解释时，优先引用本 summary 中的主指标，再结合各子模型已有 SHAP、天气型、典型气象条件和事件识别指标。",
            "- 疫情后子模型使用 anomaly 目标；若要把 anomaly 预测转换为 PM2.5 原值，预测输入必须提供 `baseline_pm2_5`。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_model = Path(args.output_model)
    metrics_output = Path(args.metrics_output)
    manifest_output = Path(args.manifest_output)
    summary_output = Path(args.summary_output)

    artifact, metrics = build_flagship_artifact(args)
    artifact["reports"] = {
        "metrics_csv": str(metrics_output),
        "manifest_json": str(manifest_output),
        "summary_md": str(summary_output),
    }
    flagship_model = FlagshipPM25MeteorologyModel(**artifact)

    output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(flagship_model, output_model, compress=3)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_output, index=False)

    write_json(manifest_output, flagship_model.to_manifest())
    write_summary(summary_output, metrics, artifact, output_model)

    print(f"Saved flagship model: {rel(output_model)}")
    print(f"Saved metrics: {rel(metrics_output)}")
    print(f"Saved manifest: {rel(manifest_output)}")
    print(f"Saved summary: {rel(summary_output)}")

    if args.predict_data:
        prediction_input = Path(args.predict_data)
        prediction_output = Path(args.prediction_output)
        frame = read_table(prediction_input)
        predictions = flagship_model.predict_frame(
            frame,
            allow_anomaly_target_only=bool(args.allow_anomaly_target_only),
        )
        write_table(predictions, prediction_output)
        print(f"Saved predictions: {rel(prediction_output)}")


if __name__ == "__main__":
    main()
