"""Unified flagship PM2.5 meteorology attribution model wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


IDENTIFIER_COLUMNS = [
    "time",
    "date",
    "city",
    "province",
    "period",
    "weather_type_k6",
    "pm2_5",
    "baseline_pm2_5",
]

CHEMICAL_IDENTIFIER_COLUMNS = [
    "time",
    "date",
    "city",
    "province",
    "period",
]

CHEMICAL_CATEGORICAL_FEATURES = {"city", "province", "period"}
CHEMICAL_FRACTION_TARGETS = {"sna_fraction", "secondary_fraction"}
CHEMICAL_RATIO_TARGETS = {"nitrate_sulfate_ratio"}


@dataclass
class FlagshipPM25MeteorologyModel:
    """A single prediction interface over the selected period-specific models."""

    model_name: str
    model_family: str
    version: str
    created_at_utc: str
    project_root: str
    source_data: dict[str, Any]
    period_selector: dict[str, Any]
    period_models: dict[str, Any]
    required_columns: list[str]
    common_feature_count: int
    required_feature_count: int
    feature_policy: dict[str, Any]
    prediction_contract: dict[str, Any]
    usage_notes: list[str]
    reports: dict[str, str] = field(default_factory=dict)
    chemical_models: dict[str, Any] = field(default_factory=dict)
    chemical_model_metadata: dict[str, Any] = field(default_factory=dict)
    chemical_prediction_contract: dict[str, Any] = field(default_factory=dict)
    period_model_paths: dict[str, str] = field(default_factory=dict)
    chemical_model_paths: dict[str, str] = field(default_factory=dict)
    external_model_dir: str = ""
    _period_model_cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    _chemical_model_cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    main_model_class: str = "scripts.flagship_pm25_meteorology_model.FlagshipPM25MeteorologyModel"

    @property
    def periods(self) -> tuple[str, ...]:
        preferred_order = (
            "pre_covid_2018_2019",
            "covid_2020_2022",
            "post_covid_2023_plus",
        )
        period_keys = set(getattr(self, "period_models", {}) or {})
        period_keys.update(getattr(self, "period_model_paths", {}) or {})
        ordered = [period for period in preferred_order if period in period_keys]
        ordered.extend(sorted(period_keys - set(ordered)))
        return tuple(ordered)

    def resolve_external_path(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate

        module_root = Path(__file__).resolve().parents[1]
        search_roots = [Path.cwd(), module_root]
        if getattr(self, "project_root", ""):
            search_roots.append(Path(self.project_root))

        for root in search_roots:
            resolved = root / candidate
            if resolved.exists():
                return resolved

        external_model_dir = getattr(self, "external_model_dir", "")
        if external_model_dir:
            external_dir = Path(external_model_dir)
            external_roots = [external_dir] if external_dir.is_absolute() else [root / external_dir for root in search_roots]
            for root in external_roots:
                resolved = root / candidate
                if resolved.exists():
                    return resolved
        return module_root / candidate

    def get_period_package(self, period: str) -> dict[str, Any]:
        cache = getattr(self, "_period_model_cache", None)
        if cache is None:
            cache = {}
            self._period_model_cache = cache
        if period in cache:
            return cache[period]

        package = (getattr(self, "period_models", {}) or {}).get(period)
        if package and "model_payload" in package:
            cache[period] = package
            return package

        path = (getattr(self, "period_model_paths", {}) or {}).get(period)
        if not path:
            raise KeyError(f"No model package available for period: {period}")
        package = joblib.load(self.resolve_external_path(path))
        cache[period] = package
        return package

    def iter_chemical_packages(self) -> list[tuple[str, dict[str, Any]]]:
        cache = getattr(self, "_chemical_model_cache", None)
        if cache is None:
            cache = {}
            self._chemical_model_cache = cache

        items: list[tuple[str, dict[str, Any]]] = []
        chemical_models = getattr(self, "chemical_models", {}) or {}
        for target, package in chemical_models.items():
            if package and "model_payload" in package:
                cache[target] = package
                items.append((target, package))

        for target, path in (getattr(self, "chemical_model_paths", {}) or {}).items():
            if target not in cache:
                cache[target] = joblib.load(self.resolve_external_path(path))
            if target not in chemical_models:
                items.append((target, cache[target]))
        return items

    def infer_period_from_time(self, frame: pd.DataFrame) -> pd.Series:
        if "time" not in frame.columns:
            raise ValueError("Input must contain a period column, or a time column from which period can be inferred.")
        time = pd.to_datetime(frame["time"])
        period = pd.Series(index=frame.index, dtype="object")
        period.loc[time < pd.Timestamp("2020-01-01")] = "pre_covid_2018_2019"
        period.loc[(time >= pd.Timestamp("2020-01-01")) & (time < pd.Timestamp("2023-01-01"))] = "covid_2020_2022"
        period.loc[time >= pd.Timestamp("2023-01-01")] = "post_covid_2023_plus"
        if period.isna().any():
            raise ValueError("Could not infer period for all rows.")
        return period

    def prepare_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        if "period" not in prepared.columns:
            prepared["period"] = self.infer_period_from_time(prepared)
        else:
            prepared["period"] = prepared["period"].astype(str)
        for column in ["city", "province"]:
            if column in prepared.columns:
                prepared[column] = prepared[column].astype(str)
        if "weather_type_k6" in prepared.columns:
            converted = pd.to_numeric(prepared["weather_type_k6"], errors="coerce")
            if converted.notna().all():
                prepared["weather_type_k6"] = converted.astype("int16")
        return prepared

    def prediction_to_pm25(self, prediction: np.ndarray, frame: pd.DataFrame, target_kind: str) -> np.ndarray:
        if target_kind == "raw":
            return np.clip(prediction, 0, None)
        if target_kind == "log1p":
            return np.clip(np.expm1(prediction), 0, None)
        if target_kind == "anomaly":
            if "baseline_pm2_5" not in frame.columns:
                raise ValueError("anomaly prediction requires a baseline_pm2_5 column.")
            return np.clip(frame["baseline_pm2_5"].to_numpy(dtype="float64") + prediction, 0, None)
        raise ValueError(f"Unknown target kind: {target_kind}")

    def transform_features(self, payload: dict[str, Any], frame: pd.DataFrame) -> Any:
        transformed = payload["preprocessor"].transform(frame)
        if hasattr(transformed, "tocoo"):
            return transformed

        columns = getattr(payload["model"], "feature_name_", None)
        if columns is None and hasattr(payload["preprocessor"], "get_feature_names_out"):
            columns = payload["preprocessor"].get_feature_names_out()
        if columns is not None and len(columns) == transformed.shape[1]:
            return pd.DataFrame(transformed, columns=list(columns), index=frame.index)
        return transformed

    def predict_frame(
        self,
        frame: pd.DataFrame,
        allow_anomaly_target_only: bool = False,
    ) -> pd.DataFrame:
        """Return row-level predictions routed through the correct period submodel."""

        data = self.prepare_frame(frame)
        known_periods = set(self.periods)
        unknown_periods = sorted(set(data["period"].dropna().astype(str)) - known_periods)
        if unknown_periods:
            raise ValueError(f"Unknown periods in prediction data: {unknown_periods}")

        output_columns = [column for column in IDENTIFIER_COLUMNS if column in data.columns]
        predictions = data[output_columns].copy()
        predictions.insert(0, "row_id", data.index.to_numpy())
        predictions["predicted_model_target"] = np.nan
        predictions["predicted_pm2_5"] = np.nan
        predictions["prediction_target_kind"] = ""
        predictions["routed_period"] = data["period"].to_numpy()

        for period in self.periods:
            mask = data["period"] == period
            if not mask.any():
                continue
            package = self.get_period_package(period)
            payload = package["model_payload"]
            features = payload["features"]
            missing_features = [feature for feature in features if feature not in data.columns]
            if missing_features:
                sample = missing_features[:20]
                raise ValueError(f"Prediction data is missing {len(missing_features)} features for {period}: {sample}")

            part = data.loc[mask].copy()
            transformed = self.transform_features(payload, part[features])
            target_prediction = payload["model"].predict(transformed)
            target_kind = payload["target_kind"]
            predictions.loc[mask, "predicted_model_target"] = target_prediction
            predictions.loc[mask, "prediction_target_kind"] = target_kind

            if target_kind == "anomaly" and "baseline_pm2_5" not in part.columns and allow_anomaly_target_only:
                continue
            predictions.loc[mask, "predicted_pm2_5"] = self.prediction_to_pm25(target_prediction, part, target_kind)

        if "pm2_5" in predictions.columns:
            predictions["residual_pm2_5"] = predictions["pm2_5"] - predictions["predicted_pm2_5"]
        return predictions

    def timestamp_from_frame(self, frame: pd.DataFrame) -> pd.Series | None:
        if "time" in frame.columns:
            return pd.to_datetime(frame["time"])
        if "date" in frame.columns:
            timestamp = pd.to_datetime(frame["date"])
            if "hour" in frame.columns:
                hours = pd.to_numeric(frame["hour"], errors="coerce").fillna(0)
                timestamp = timestamp + pd.to_timedelta(hours, unit="h")
            return timestamp
        return None

    def add_time_columns(self, frame: pd.DataFrame, timestamp: pd.Series) -> pd.DataFrame:
        prepared = frame.copy()

        def fill_column(column: str, values: Any) -> None:
            if column not in prepared.columns:
                prepared[column] = values
                return
            mask = prepared[column].isna()
            if mask.any():
                prepared.loc[mask, column] = pd.Series(values, index=prepared.index).loc[mask]

        fill_column("hour", timestamp.dt.hour)
        fill_column("month", timestamp.dt.month)
        fill_column("year", timestamp.dt.year)
        fill_column("dayofyear", timestamp.dt.dayofyear)
        fill_column("weekday", timestamp.dt.weekday)
        fill_column("is_weekend", timestamp.dt.weekday.isin([5, 6]).astype("int8"))

        if "period" not in prepared.columns or prepared["period"].isna().any():
            inferred = pd.Series(index=prepared.index, dtype="object")
            inferred.loc[timestamp < pd.Timestamp("2020-01-01")] = "pre_covid_2018_2019"
            inferred.loc[(timestamp >= pd.Timestamp("2020-01-01")) & (timestamp < pd.Timestamp("2023-01-01"))] = (
                "covid_2020_2022"
            )
            inferred.loc[timestamp >= pd.Timestamp("2023-01-01")] = "post_covid_2023_plus"
            fill_column("period", inferred)

        fill_column("is_covid_period", (prepared["period"].astype(str) == "covid_2020_2022").astype("int8"))
        fill_column("hour_sin", np.sin(2 * np.pi * pd.to_numeric(prepared["hour"], errors="coerce") / 24))
        fill_column("hour_cos", np.cos(2 * np.pi * pd.to_numeric(prepared["hour"], errors="coerce") / 24))
        fill_column("dayofyear_sin", np.sin(2 * np.pi * pd.to_numeric(prepared["dayofyear"], errors="coerce") / 366))
        fill_column("dayofyear_cos", np.cos(2 * np.pi * pd.to_numeric(prepared["dayofyear"], errors="coerce") / 366))
        fill_column("month_sin", np.sin(2 * np.pi * pd.to_numeric(prepared["month"], errors="coerce") / 12))
        fill_column("month_cos", np.cos(2 * np.pi * pd.to_numeric(prepared["month"], errors="coerce") / 12))
        return prepared

    def prepare_chemical_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        timestamp = self.timestamp_from_frame(prepared)
        if timestamp is not None:
            prepared = self.add_time_columns(prepared, timestamp)
        elif "period" not in prepared.columns:
            raise ValueError("Chemical diagnostics require a period column, or a time/date column.")

        for column in CHEMICAL_CATEGORICAL_FEATURES:
            if column in prepared.columns:
                prepared[column] = prepared[column].astype(str)
        return prepared

    def complete_chemical_features(self, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
        completed = frame.copy()
        for feature in features:
            if feature in completed.columns:
                continue
            completed[feature] = "unknown" if feature in CHEMICAL_CATEGORICAL_FEATURES else np.nan
        return completed[features]

    def inverse_chemical_prediction(self, prediction: np.ndarray, package: dict[str, Any]) -> np.ndarray:
        metadata = package.get("metrics", {})
        target = package.get("target") or metadata.get("target", "")
        transform = package.get("target_transform") or metadata.get("target_transform", "identity")

        values = np.asarray(prediction, dtype="float64")
        if transform == "log1p":
            values = np.expm1(values)
        elif transform not in {"identity", None, ""}:
            raise ValueError(f"Unknown chemical target transform for {target}: {transform}")

        if target in CHEMICAL_FRACTION_TARGETS:
            return np.clip(values, 0.0, 1.0)
        if target in CHEMICAL_RATIO_TARGETS:
            return np.clip(values, 0.0, None)
        return np.clip(values, 0.0, None)

    def has_chemical_models(self) -> bool:
        return bool(getattr(self, "chemical_models", {}) or getattr(self, "chemical_model_paths", {}))

    def predict_chemical_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return row-level chemical-composition mechanism diagnostics."""

        chemical_packages = self.iter_chemical_packages()
        if not chemical_packages:
            raise ValueError("This flagship model does not contain chemical-composition diagnostic models.")

        data = self.prepare_chemical_frame(frame)
        output_columns = [column for column in CHEMICAL_IDENTIFIER_COLUMNS if column in data.columns]
        predictions = data[output_columns].copy()
        predictions.insert(0, "row_id", data.index.to_numpy())

        for target, package in chemical_packages:
            payload = package["model_payload"]
            features = list(payload["features"])
            feature_frame = self.complete_chemical_features(data, features)
            transformed = self.transform_features(payload, feature_frame)
            target_prediction = payload["model"].predict(transformed)
            values = self.inverse_chemical_prediction(target_prediction, package)
            predictions[f"predicted_{target}"] = values
            predictions[f"{target}_model_target"] = target_prediction

        return predictions

    def chemical_target_summary(self) -> pd.DataFrame:
        rows = []
        for target, package in self.iter_chemical_packages():
            metrics = package.get("metrics", {})
            test = metrics.get("metrics", {})
            blocked_cv = metrics.get("blocked_time_cv", {})
            rows.append(
                {
                    "target": target,
                    "target_label": package.get("target_label", target),
                    "target_kind": package.get("target_kind"),
                    "value_unit": package.get("value_unit"),
                    "feature_set": package.get("feature_set"),
                    "target_transform": package.get("target_transform") or metrics.get("target_transform"),
                    "feature_count": len(package["model_payload"].get("features", [])),
                    "test_r2": test.get("r2"),
                    "test_rmse": test.get("rmse"),
                    "test_mae": test.get("mae"),
                    "blocked_cv_mean_r2": blocked_cv.get("mean_r2"),
                    "source_model_path": package.get("source_model_path"),
                }
            )
        return pd.DataFrame(rows)

    def predict(self, frame: pd.DataFrame, allow_anomaly_target_only: bool = False) -> np.ndarray:
        """Return predicted PM2.5 values, matching a normal estimator-style API."""

        predictions = self.predict_frame(frame, allow_anomaly_target_only=allow_anomaly_target_only)
        return predictions["predicted_pm2_5"].to_numpy(dtype="float64")

    def predict_model_target(self, frame: pd.DataFrame) -> np.ndarray:
        """Return predictions on each routed submodel's native target scale."""

        predictions = self.predict_frame(frame, allow_anomaly_target_only=True)
        return predictions["predicted_model_target"].to_numpy(dtype="float64")

    def submodel_summary(self) -> pd.DataFrame:
        rows = []
        for period in self.periods:
            package = self.get_period_package(period)
            metrics = package.get("metrics", {})
            selection = package.get("selection", {})
            test = metrics.get("test_pm25", {})
            blocked_cv = metrics.get("blocked_time_cv", {})
            rows.append(
                {
                    "period": period,
                    "period_label": selection.get("period_label"),
                    "target_kind": selection.get("target_kind"),
                    "target_label": selection.get("target_label"),
                    "feature_count": len(package["model_payload"].get("features", [])),
                    "test_r2": test.get("r2"),
                    "test_rmse": test.get("rmse"),
                    "test_mae": test.get("mae"),
                    "test_bias": test.get("bias"),
                    "blocked_cv_mean_r2": blocked_cv.get("mean_r2"),
                    "source_model_path": package.get("source_model_path"),
                }
            )
        return pd.DataFrame(rows)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_family": self.model_family,
            "version": self.version,
            "main_model_class": self.main_model_class,
            "created_at_utc": self.created_at_utc,
            "project_root": self.project_root,
            "source_data": self.source_data,
            "period_selector": self.period_selector,
            "required_columns": self.required_columns,
            "common_feature_count": self.common_feature_count,
            "required_feature_count": self.required_feature_count,
            "feature_policy": self.feature_policy,
            "prediction_contract": self.prediction_contract,
            "chemical_prediction_contract": self.chemical_prediction_contract,
            "chemical_model_metadata": self.chemical_model_metadata,
            "usage_notes": self.usage_notes,
            "reports": self.reports,
            "period_models": {
                period: {
                    "selection": package["selection"],
                    "source_model_path": package["source_model_path"],
                    "source_model_sha256": package["source_model_sha256"],
                    "source_metrics_path": package["source_metrics_path"],
                    "source_metrics_sha256": package["source_metrics_sha256"],
                    "test_pm25": package["metrics"].get("test_pm25", {}),
                    "blocked_time_cv": package["metrics"].get("blocked_time_cv", {}),
                    "feature_count": len(package["model_payload"]["features"]),
                }
                for period, package in ((period, self.get_period_package(period)) for period in self.periods)
            },
            "chemical_models": {
                target: {
                    "target": package.get("target", target),
                    "target_label": package.get("target_label", target),
                    "target_kind": package.get("target_kind"),
                    "value_unit": package.get("value_unit"),
                    "feature_set": package.get("feature_set"),
                    "target_transform": package.get("target_transform"),
                    "source_model_path": package.get("source_model_path"),
                    "source_model_sha256": package.get("source_model_sha256"),
                    "source_metrics_path": package.get("source_metrics_path"),
                    "source_metrics_sha256": package.get("source_metrics_sha256"),
                    "test_metrics": package.get("metrics", {}).get("metrics", {}),
                    "blocked_time_cv": package.get("metrics", {}).get("blocked_time_cv", {}),
                    "feature_count": len(package["model_payload"]["features"]),
                }
                for target, package in self.iter_chemical_packages()
            },
        }
