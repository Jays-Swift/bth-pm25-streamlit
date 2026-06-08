from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_DIR = PROJECT_ROOT / "models"
ASSET_DIR = PROJECT_ROOT / "app_assets"

Assets = dict[str, Any]
Metrics = dict[str, Any]

CURRENT_MODEL_PATH = MODEL_DIR / "high_accuracy_lightgbm_extended_target_pm2_5_full_2018_plus_cnemc.joblib"
NEXT24_MODEL_PATH = MODEL_DIR / "high_accuracy_lightgbm_core_target_pm2_5_next_24h.joblib"

PREDICTION_MODEL_SPECS = {
    "full_high_accuracy": {
        "label": "全时期高精度模型",
        "path": "high_accuracy_lightgbm_extended_target_pm2_5_full_2018_plus_cnemc.joblib",
        "type": "高精度预测",
        "description": "默认主模型，适合展示整体预测能力。",
    },
    "pre_high_accuracy": {
        "label": "疫情前高精度模型",
        "path": "high_accuracy_lightgbm_extended_target_pm2_5_pre_covid_2018_2019_high_accuracy.joblib",
        "type": "高精度预测",
        "description": "2018-2019 单独训练，含 PM2.5 时滞、滚动均值和共污染物。",
    },
    "covid_high_accuracy": {
        "label": "疫情期高精度模型",
        "path": "high_accuracy_lightgbm_extended_target_pm2_5_covid_2020_2022_high_accuracy.joblib",
        "type": "高精度预测",
        "description": "2020-2022 单独训练，适合疫情期预测对照。",
    },
    "post_high_accuracy": {
        "label": "疫情后高精度模型",
        "path": "high_accuracy_lightgbm_extended_target_pm2_5_post_covid_2023_plus_high_accuracy.joblib",
        "type": "高精度预测",
        "description": "2023+ 单独训练，需注意 PM2.5 数据源与 2018-2022 不完全一致。",
    },
    "pre_meteorology": {
        "label": "疫情前气象归因模型",
        "path": "high_accuracy_lightgbm_meteorology_target_pm2_5_pre_covid_2018_2019_meteorology_only.joblib",
        "type": "气象归因",
        "description": "排除 PM2.5 持续性和共污染物，用于解释气象贡献。",
    },
    "covid_meteorology": {
        "label": "疫情期气象归因模型",
        "path": "high_accuracy_lightgbm_meteorology_target_pm2_5_covid_2020_2022_meteorology_only.joblib",
        "type": "气象归因",
        "description": "排除 PM2.5 持续性和共污染物，用于疫情期气象贡献分析。",
    },
    "post_meteorology": {
        "label": "疫情后气象归因模型",
        "path": "high_accuracy_lightgbm_meteorology_target_pm2_5_post_covid_2023_plus_meteorology_only.joblib",
        "type": "气象归因",
        "description": "排除 PM2.5 持续性和共污染物，用于疫情后气象贡献分析。",
    },
    "pre_meteorology_v2": {
        "label": "疫情前气象贡献 v2 模型",
        "path": "meteorology_attribution_v2_core_pre_covid_2018_2019_log1p.joblib",
        "type": "气象贡献 v2",
        "description": "本轮最新气象-only 归因模型，目标为 log1p(PM2.5)，用于疫情前气象贡献情景估计。",
    },
    "covid_meteorology_v2": {
        "label": "疫情期气象贡献 v2 模型",
        "path": "meteorology_attribution_v2_core_covid_2020_2022_raw.joblib",
        "type": "气象贡献 v2",
        "description": "本轮最新气象-only 归因模型，目标为 PM2.5 原值，用于疫情期气象贡献情景估计。",
    },
    "post_meteorology_v2": {
        "label": "疫情后气象贡献 v2 模型",
        "path": "meteorology_attribution_v2_core_post_covid_2023_plus_anomaly.joblib",
        "type": "气象贡献 v2",
        "description": "本轮最新气象-only 归因模型，目标为同城同月同小时气候态异常，用于疫情后气象贡献情景估计。",
    },
}

MODEL_SELECT_OPTIONS = [
    "全时期高精度模型",
    "按日期自动选择分时期高精度模型",
    "按日期自动选择 v2 气象贡献模型（推荐归因）",
    "按日期自动选择旧版气象归因模型（对照）",
    "疫情前高精度模型",
    "疫情期高精度模型",
    "疫情后高精度模型",
    "疫情前气象贡献 v2 模型",
    "疫情期气象贡献 v2 模型",
    "疫情后气象贡献 v2 模型",
    "疫情前旧版气象归因模型",
    "疫情期旧版气象归因模型",
    "疫情后旧版气象归因模型",
]

MODEL_LABEL_TO_KEY = {
    "全时期高精度模型": "full_high_accuracy",
    "疫情前高精度模型": "pre_high_accuracy",
    "疫情期高精度模型": "covid_high_accuracy",
    "疫情后高精度模型": "post_high_accuracy",
    "疫情前气象贡献 v2 模型": "pre_meteorology_v2",
    "疫情期气象贡献 v2 模型": "covid_meteorology_v2",
    "疫情后气象贡献 v2 模型": "post_meteorology_v2",
    "疫情前旧版气象归因模型": "pre_meteorology",
    "疫情期旧版气象归因模型": "covid_meteorology",
    "疫情后旧版气象归因模型": "post_meteorology",
    "疫情前气象归因模型": "pre_meteorology",
    "疫情期气象归因模型": "covid_meteorology",
    "疫情后气象归因模型": "post_meteorology",
}

PERIOD_TO_HIGH_ACCURACY = {
    "pre_covid_2018_2019": "pre_high_accuracy",
    "covid_2020_2022": "covid_high_accuracy",
    "post_covid_2023_plus": "post_high_accuracy",
}

PERIOD_TO_METEOROLOGY = {
    "pre_covid_2018_2019": "pre_meteorology",
    "covid_2020_2022": "covid_meteorology",
    "post_covid_2023_plus": "post_meteorology",
}

PERIOD_TO_METEOROLOGY_V2 = {
    "pre_covid_2018_2019": "pre_meteorology_v2",
    "covid_2020_2022": "covid_meteorology_v2",
    "post_covid_2023_plus": "post_meteorology_v2",
}

MODEL_KEY_TO_PERIOD = {
    **{key: period for period, key in PERIOD_TO_HIGH_ACCURACY.items()},
    **{key: period for period, key in PERIOD_TO_METEOROLOGY.items()},
    **{key: period for period, key in PERIOD_TO_METEOROLOGY_V2.items()},
}

PERIOD_LABELS = {
    "pre_covid_2018_2019": "疫情前 2018-2019",
    "covid_2020_2022": "疫情期 2020-2022",
    "post_covid_2023_plus": "疫情后 2023+",
}


AQI_BANDS = [
    (35, "优", "#16a34a"),
    (75, "良好", "#22c55e"),
    (115, "轻度污染", "#f59e0b"),
    (150, "中度污染", "#dc2626"),
    (250, "重度污染", "#991b1b"),
    (10_000, "严重污染", "#7f1d1d"),
]

FEATURE_LABELS = {
    "pm2_5_roll_mean_3h": "3小时 PM2.5 滚动均值",
    "pm2_5_roll_mean_6h": "6小时 PM2.5 滚动均值",
    "pm2_5_lag_1h": "前1小时 PM2.5",
    "pm2_5_lag_3h": "前3小时 PM2.5",
    "pm2_5_lag_6h": "前6小时 PM2.5",
    "pm10": "PM10",
    "carbon_monoxide": "CO",
    "nitrogen_dioxide": "NO2",
    "sulphur_dioxide": "SO2",
    "ozone": "O3",
    "dust": "dust",
    "aerosol_optical_depth": "AOD",
    "boundary_layer_height": "边界层高度 PBLH",
    "boundary_layer_height_lag_1h": "前1小时 PBLH",
    "boundary_layer_height_lag_3h": "前3小时 PBLH",
    "boundary_layer_height_roll_mean_3h": "3小时平均 PBLH",
    "boundary_layer_height_roll_mean_24h": "24小时平均 PBLH",
    "boundary_layer_height_roll_mean_48h": "48小时平均 PBLH",
    "boundary_layer_height_change_24h": "24小时 PBLH 变化量",
    "pblh_q25_city_period": "同城同阶段 PBLH 低分位阈值",
    "low_pblh_flag": "低边界层标记",
    "t_inverse_850_1000": "逆温指数 T850-T1000",
    "temperature_850hPa": "850hPa 温度",
    "temperature_2m": "2m 气温",
    "temperature_2m_change_24h": "24小时气温变化量",
    "temperature_2m_roll_mean_72h": "72小时平均气温",
    "dew_point_2m": "露点温度",
    "dew_point_2m_lag_3h": "前3小时露点",
    "dew_point_2m_lag_72h": "前72小时露点",
    "dew_point_2m_roll_mean_3h": "3小时平均露点",
    "dew_point_2m_roll_mean_6h": "6小时平均露点",
    "dew_point_2m_roll_mean_72h": "72小时平均露点",
    "relative_humidity_2m": "相对湿度",
    "relative_humidity_2m_roll_mean_3h": "3小时平均相对湿度",
    "relative_humidity_2m_roll_mean_12h": "12小时平均相对湿度",
    "relative_humidity_2m_roll_mean_24h": "24小时平均相对湿度",
    "rh_q75_city_period": "同城同阶段湿度高分位阈值",
    "humidity_pblh_interaction": "湿度-PBLH 复合项",
    "pressure_msl": "海平面气压",
    "pressure_msl_lag_72h": "前72小时海平面气压",
    "pressure_msl_change_24h": "24小时海平面气压变化",
    "surface_pressure": "地面气压",
    "surface_pressure_lag_48h": "前48小时地面气压",
    "surface_pressure_lag_72h": "前72小时地面气压",
    "wind_speed_10m": "10m 风速",
    "wind_speed_10m_lag_3h": "前3小时风速",
    "wind_speed_10m_lag_24h": "前24小时风速",
    "wind_speed_10m_roll_mean_12h": "12小时平均风速",
    "wind_direction_10m": "10m 风向",
    "wind_u_10m": "东西向风 U",
    "wind_v_10m": "南北向风 V",
    "wind_u_10m_lag_3h": "前3小时东西向风 U",
    "wind_u_10m_lag_24h": "前24小时东西向风 U",
    "wind_v_10m_lag_3h": "前3小时南北风 V",
    "wind_v_10m_lag_24h": "前24小时南北风 V",
    "wind_v_10m_roll_mean_12h": "12小时平均南北风 V",
    "wind_v_10m_roll_mean_24h": "24小时平均南北风 V",
    "wind_v_10m_roll_mean_48h": "48小时平均南北风 V",
    "wind_gusts_10m": "10m 阵风",
    "northerly_cleaning_10m": "北风清洁输送",
    "northerly_cleaning_10m_roll_mean_24h": "24小时北风清洁输送",
    "northerly_cleaning_10m_roll_mean_48h": "48小时北风清洁输送",
    "northerly_cleaning_10m_roll_mean_72h": "72小时北风清洁输送",
    "southerly_transport_10m": "南风污染输送",
    "ventilation_coefficient_lag_1h": "前1小时通风系数",
    "ventilation_coefficient_lag_3h": "前3小时通风系数",
    "ventilation_coefficient_roll_mean_3h": "3小时平均通风系数",
    "ventilation_coefficient_roll_mean_24h": "24小时平均通风系数",
    "precipitation": "降水量",
    "precipitation_roll_sum_72h": "72小时累计降水",
    "rain": "降雨量",
    "cloud_cover": "云量",
    "dayofyear_cos": "年内周期 cos",
    "dayofyear_sin": "年内周期 sin",
    "dayofyear": "年内日序",
    "month_cos": "月份周期 cos",
    "month_sin": "月份周期 sin",
    "month": "月份",
    "hour_sin": "小时周期 sin",
    "hour_cos": "小时周期 cos",
    "hour": "小时",
    "weekday": "星期",
    "province_Hebei": "河北省类别",
    "latitude": "纬度",
    "longitude": "经度",
    "year": "年份",
}

METEOROLOGY_FEATURE_HINTS = (
    "temperature",
    "dew_point",
    "relative_humidity",
    "pressure",
    "boundary_layer_height",
    "pblh",
    "low_pblh",
    "t_inverse",
    "wind_",
    "northerly",
    "southerly",
    "ventilation",
    "humidity_pblh",
    "rh_q",
    "precipitation",
    "rain",
    "cloud_cover",
)


def resolve_path(*parts: str) -> Path:
    direct = PROJECT_ROOT.joinpath(*parts)
    if direct.exists():
        return direct
    packaged = APP_DIR.joinpath(*parts)
    if packaged.exists():
        return packaged
    return direct


@st.cache_resource(show_spinner=False)
def load_model(path: str) -> dict:
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_assets() -> Assets:
    asset_dir = resolve_path("app_assets")
    meteorology_v2_path = asset_dir / "meteorology_attribution_v2_core_results.json"
    with (asset_dir / "app_metadata.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    return {
        "metadata": metadata,
        "city_info": pd.read_csv(asset_dir / "city_info.csv"),
        "profiles": pd.read_csv(asset_dir / "city_month_hour_profiles.csv"),
        "daily": pd.read_csv(asset_dir / "city_daily_history.csv", parse_dates=["date"]),
        "seasonal": pd.read_csv(asset_dir / "seasonal_reference.csv"),
        "current_metrics": json.loads((asset_dir / "current_metrics.json").read_text(encoding="utf-8")),
        "next24_metrics": json.loads((asset_dir / "next24_metrics.json").read_text(encoding="utf-8")),
        "extended_current_metrics": json.loads((asset_dir / "extended_current_metrics.json").read_text(encoding="utf-8")),
        "extended_next24_metrics": json.loads((asset_dir / "extended_next24_metrics.json").read_text(encoding="utf-8")),
        "current_shap": pd.read_csv(asset_dir / "current_shap_importance.csv"),
        "next24_shap": pd.read_csv(asset_dir / "next24_shap_importance.csv"),
        "pre_covid_meteorology_metrics": json.loads((asset_dir / "pre_covid_meteorology_metrics.json").read_text(encoding="utf-8")),
        "covid_meteorology_metrics": json.loads((asset_dir / "covid_meteorology_metrics.json").read_text(encoding="utf-8")),
        "post_covid_meteorology_metrics": json.loads((asset_dir / "post_covid_meteorology_metrics.json").read_text(encoding="utf-8")),
        "pre_covid_high_accuracy_metrics": json.loads((asset_dir / "pre_covid_high_accuracy_metrics.json").read_text(encoding="utf-8")),
        "covid_high_accuracy_metrics": json.loads((asset_dir / "covid_high_accuracy_metrics.json").read_text(encoding="utf-8")),
        "post_covid_high_accuracy_metrics": json.loads((asset_dir / "post_covid_high_accuracy_metrics.json").read_text(encoding="utf-8")),
        "pre_covid_meteorology_shap": pd.read_csv(asset_dir / "pre_covid_meteorology_shap_importance.csv"),
        "covid_meteorology_shap": pd.read_csv(asset_dir / "covid_meteorology_shap_importance.csv"),
        "post_covid_meteorology_shap": pd.read_csv(asset_dir / "post_covid_meteorology_shap_importance.csv"),
        "pre_covid_high_accuracy_shap": pd.read_csv(asset_dir / "pre_covid_high_accuracy_shap_importance.csv"),
        "covid_high_accuracy_shap": pd.read_csv(asset_dir / "covid_high_accuracy_shap_importance.csv"),
        "post_covid_high_accuracy_shap": pd.read_csv(asset_dir / "post_covid_high_accuracy_shap_importance.csv"),
        "model_metrics_summary": pd.read_csv(asset_dir / "pm25_model_metrics_summary.csv"),
        "model_top_shap_summary": pd.read_csv(asset_dir / "pm25_model_top_shap_summary.csv"),
        "period_residual_analysis": pd.read_csv(asset_dir / "period_residual_analysis.csv"),
        "meteorology_v2": (
            json.loads(meteorology_v2_path.read_text(encoding="utf-8"))
            if meteorology_v2_path.exists()
            else None
        ),
    }


def pm25_category(value: float) -> tuple[str, str]:
    for upper, label, color in AQI_BANDS:
        if value <= upper:
            return label, color
    return "严重污染", "#4e342e"


def metric_text(metrics: Metrics) -> str:
    test = metrics["test"]
    return f"MAE {test['mae']:.2f} | RMSE {test['rmse']:.2f} | R2 {test['r2']:.3f}"


def metric_badges(metrics: Metrics) -> str:
    test = metrics["test"]
    return (
        f'<span class="score-pill"><b>MAE</b>{test["mae"]:.2f}</span>'
        f'<span class="score-pill"><b>RMSE</b>{test["rmse"]:.2f}</span>'
        f'<span class="score-pill"><b>R2</b>{test["r2"]:.3f}</span>'
    )


def metric_badges_from_test(test: dict[str, float]) -> str:
    return (
        f'<span class="score-pill"><b>MAE</b>{float(test["mae"]):.2f}</span>'
        f'<span class="score-pill"><b>RMSE</b>{float(test["rmse"]):.2f}</span>'
        f'<span class="score-pill"><b>R2</b>{float(test["r2"]):.3f}</span>'
    )


def section_bridge_html(kicker: str, title: str, body: str, theme: str = "blue") -> str:
    return f"""
    <section class="section-bridge {theme}">
      <div class="section-bridge-title">{title}</div>
    </section>
    """


def model_intro_html(assets: Assets) -> str:
    current = assets["current_metrics"]
    next24 = assets["next24_metrics"]
    return f"""
    <div class="model-intro">
      <div class="model-hero-grid">
        <div class="model-hero-card primary">
          <div class="model-kicker">预测台默认</div>
          <h3>全时期高精度模型</h3>
          <p>覆盖 2018-01-01 至 2026-05-31，用于当前小时 PM2.5 高精度预测。它综合气象、ERA5 PBLH、逆温、风输送、污染时滞、滚动均值和共污染物。</p>
          <div class="score-row">{metric_badges(current)}</div>
        </div>
        <div class="model-hero-card muted">
          <div class="model-kicker">辅助展示</div>
          <h3>24 小时辅助模型</h3>
          <p>用于给出 24 小时后趋势参考。它仍沿用较早训练口径，因此不作为本轮疫情分时期气象归因的主要证据。</p>
          <div class="score-row">{metric_badges(next24)}</div>
        </div>
      </div>
      <div class="model-route-grid">
        <div class="model-route-card accent-blue">
          <span class="route-badge">预测主线</span>
          <h4>全时期高精度模型</h4>
          <p>用于刻画模型在多源环境信息约束下的应用预测能力，适合预测台展示和整体性能说明。</p>
          <div class="route-meta">2018-2026 | 气象 + PBLH + 污染持续性 + 共污染物</div>
        </div>
        <div class="model-route-card accent-green">
          <span class="route-badge">疫情前</span>
          <h4>疫情前气象贡献模型</h4>
          <p>只保留气象、PBLH、稳定度、风和时空控制变量，用于正常排放时期气象贡献分析。</p>
          <div class="route-meta">2018-2019 | 正常时期对照基线</div>
        </div>
        <div class="model-route-card accent-amber">
          <span class="route-badge">疫情期</span>
          <h4>疫情期气象贡献模型</h4>
          <p>排除 PM2.5 持续性和共污染物，专门观察人为活动减弱背景下气象因子权重变化。</p>
          <div class="route-meta">2020-2022 | 气象归因核心阶段</div>
        </div>
        <div class="model-route-card accent-purple">
          <span class="route-badge">疫情后</span>
          <h4>疫情后气象贡献模型</h4>
          <p>用于和疫情前、疫情期对照，分析 PBLH、湿度、风速和逆温等贡献在恢复期的再强化特征。</p>
          <div class="route-meta">2023+ | 恢复期贡献对照</div>
        </div>
      </div>
    </div>
    """


def public_training_update_html(assets: Assets) -> str:
    v2 = assets.get("meteorology_v2") or {}
    best = v2.get("best_summary", [])
    best_by_period = {row["period"]: row for row in best}
    pre = best_by_period.get("pre_covid_2018_2019", {})
    covid = best_by_period.get("covid_2020_2022", {})
    post = best_by_period.get("post_covid_2023_plus", {})

    def card(period: str, label: str, row: dict[str, Any], theme: str, fallback: Metrics) -> str:
        if row:
            target = row["target_label"]
            metrics = metric_badges_from_test({"mae": row["mae"], "rmse": row["rmse"], "r2": row["r2"]})
            feature_count = int(row["feature_count"])
            train_rows = int(row["train_rows"])
            meta = f"{target} | {feature_count} 个气象特征 | 训练样本 {train_rows:,}"
        else:
            metrics = metric_badges(fallback)
            meta = "旧版气象贡献模型 | 当前小时气象 baseline"
        return (
            f'<article class="intro-model-card {theme}">'
            f'<div class="model-card-tag">{period}</div>'
            f"<h4>{label}</h4>"
            f"<p>{meta}</p>"
            f'<div class="score-row compact">{metrics}</div>'
            "</article>"
        )

    return f"""
    <div class="intro-page public-update">
      <section class="intro-hero attribution">
        <div class="intro-hero-main">
          <div class="intro-kicker">Public update · 2026-06-08</div>
          <h3>本轮新训练内容已更新至公网版本</h3>
          <p>网页已纳入今天新训练的 v2 气象贡献模型，并同步到预测台、模型介绍和训练策略。v2 模型严格保持气象-only 变量口径，排除 PM2.5 时滞、滚动均值和共污染物，重点使用 PBLH、多时滞气象变量、通风系数、连续低 PBLH、连续弱风、高湿、南风输送和北风清洁输送等 248 个过程型气象特征。</p>
          <div class="intro-chip-row">
            <span>v2 气象贡献模型</span>
            <span>三时期单独训练</span>
            <span>raw / log1p / anomaly 目标对照</span>
            <span>预测台可直接选择</span>
          </div>
        </div>
        <div class="intro-score-panel">
          <div class="intro-score-label">网页当前公开版</div>
          <h4>高精度预测 + v2 气象贡献</h4>
          <p>高精度模型用于预测展示；v2 气象贡献模型用于分时期气象解释和疫情前后对比。</p>
        </div>
      </section>
      <div class="intro-model-grid">
        {card("疫情前 2018-2019", "疫情前气象贡献 v2 模型", pre, "green", assets["pre_covid_meteorology_metrics"])}
        {card("疫情期 2020-2022", "疫情期气象贡献 v2 模型", covid, "amber", assets["covid_meteorology_metrics"])}
        {card("疫情后 2023+", "疫情后气象贡献 v2 模型", post, "teal", assets["post_covid_meteorology_metrics"])}
      </div>
    </div>
    """


def high_accuracy_intro_html(assets: Assets) -> str:
    current = assets["current_metrics"]
    pre = assets["pre_covid_high_accuracy_metrics"]
    covid = assets["covid_high_accuracy_metrics"]
    post = assets["post_covid_high_accuracy_metrics"]
    return f"""
    <div class="intro-page">
      <section class="intro-hero prediction">
        <div class="intro-hero-main">
          <div class="intro-kicker">High-accuracy prediction models</div>
          <h3>高精度预测模型：面向应用预测的 PM2.5 估计体系</h3>
          <p>这一组模型的目标是尽可能准确地给出京津冀城市小时尺度 PM2.5 浓度。它并不只依赖气象变量，而是把气象、ERA5 边界层高度、逆温指数、风输送、PM2.5 时滞、滚动均值、共污染物和时空控制变量共同纳入 LightGBM。这样做的优势是预测稳定、误差较低，适合预测台交互展示；需要注意的是，它的 SHAP 结果会同时反映污染持续性、共污染物和气象条件，不应被直接等同为“纯气象贡献”。</p>
          <div class="intro-chip-row">
            <span>2018-2026 全时期主模型</span>
            <span>疫情前/疫情期/疫情后分时期对照</span>
            <span>PM2.5 时滞与滚动特征</span>
            <span>共污染物协同约束</span>
          </div>
        </div>
        <div class="intro-score-panel">
          <div class="intro-score-label">预测台默认主模型</div>
          <h4>全时期高精度模型</h4>
          <div class="score-row">{metric_badges(current)}</div>
          <p>覆盖 2018-01-01 至 2026-05-31，是当前预测台默认使用的模型，适合展示项目的工程化预测能力。</p>
        </div>
      </section>
      {section_bridge_html(
        "模型谱系",
        "四类高精度模型分别承担不同展示任务",
        "下面这组卡片不是重复罗列指标，而是把预测模型按使用场景拆开：全时期模型负责预测台默认输出，三套分时期模型负责疫情前、疫情期、疫情后的预测能力对照。读这组卡片时，应重点关注数据时期、特征口径和研究定位。",
        "blue",
      )}
      <div class="intro-model-grid">
        <article class="intro-model-card blue">
          <div class="model-card-tag">全时期</div>
          <h4>统一主模型</h4>
          <p>在完整 2018+ 数据上训练，学习跨年份、跨城市的总体非线性规律，主要用于评估当前气象与污染背景约束下的 PM2.5 应用预测能力。</p>
          <div class="score-row compact">{metric_badges(current)}</div>
        </article>
        <article class="intro-model-card green">
          <div class="model-card-tag">疫情前</div>
          <h4>正常时期预测上限</h4>
          <p>2018-2019 单独训练，用作正常排放背景下的高精度预测基准。其特征口径与主模型一致，便于和疫情期、疫情后模型横向比较。</p>
          <div class="score-row compact">{metric_badges(pre)}</div>
        </article>
        <article class="intro-model-card amber">
          <div class="model-card-tag">疫情期</div>
          <h4>人为活动减弱时期预测</h4>
          <p>2020-2022 单独训练，用于检验疫情时期污染水平和污染持续性改变后的短时预测稳定性。</p>
          <div class="score-row compact">{metric_badges(covid)}</div>
        </article>
        <article class="intro-model-card teal">
          <div class="model-card-tag">疫情后</div>
          <h4>恢复期预测对照</h4>
          <p>2023+ 单独训练，用于观察恢复期预测精度和特征贡献结构。跨时期解释时需标注 2023+ PM2.5 数据源与 2018-2022 不完全一致。</p>
          <div class="score-row compact">{metric_badges(post)}</div>
        </article>
      </div>
      {section_bridge_html(
        "解读边界",
        "高精度模型展示预测上限，但不直接等同于纯气象归因",
        "这一部分说明高精度模型的性能来源及学术使用边界。模型加入污染时滞和共污染物后，预测误差会显著下降，但这些变量也会吸收排放背景和污染持续性的影响，因此需要和气象贡献模型分开解读。",
        "slate",
      )}
      <div class="method-band">
        <div>
          <span class="method-band-kicker">性能来源</span>
          <h4>高精度模型把污染持续性作为关键可预测信息</h4>
          <p>PM2.5 具有明显的小时级持续性：前 1 小时、前 3 小时、前 24 小时浓度以及滚动均值通常能携带边界层内污染累积、区域输送和排放背景的综合信号。模型同时使用 PM10、NO2、CO、SO2、O3、AOD、dust 等共污染物，进一步约束同一气团和同一污染过程中的化学与输送状态。因此高精度模型 R2 往往显著高于气象贡献模型，这是合理的，也正是它适合交互预测的原因。</p>
        </div>
        <div>
          <span class="method-band-kicker">归因边界</span>
          <h4>不能把高精度 SHAP 直接写成气象归因结论</h4>
          <p>当 PM2.5 滞后、滚动均值和共污染物进入模型后，SHAP 前列经常由污染持续性变量占据。它们提高了预测精度，但会吸收一部分排放、人为活动和二次生成过程的影响。因此该模型适合证明“机器学习框架能预测得准”，而关于 PBLH、逆温、湿度、气压和风输送的独立贡献，应优先引用气象贡献模型。</p>
        </div>
      </div>
    </div>
    """


def meteorology_contribution_intro_html(assets: Assets) -> str:
    v2 = assets.get("meteorology_v2") or {}
    best = v2.get("best_summary", [])
    best_by_period = {row["period"]: row for row in best}
    pre = best_by_period.get("pre_covid_2018_2019", {})
    covid = best_by_period.get("covid_2020_2022", {})
    post = best_by_period.get("post_covid_2023_plus", {})

    def score(row: dict[str, Any], fallback: Metrics) -> str:
        if row:
            return metric_badges_from_test({"mae": row["mae"], "rmse": row["rmse"], "r2": row["r2"]})
        return metric_badges(fallback)

    return f"""
    <div class="intro-page">
      <section class="intro-hero attribution">
        <div class="intro-hero-main">
          <div class="intro-kicker">Meteorology contribution models</div>
          <h3>气象贡献模型：面向学术归因的天气驱动解释体系</h3>
          <p>这一组模型的核心不是追求最高预测分数，而是尽量隔离气象背景对 PM2.5 变化的独立解释力。最新 v2 气象贡献模型排除了 PM2.5 时滞、PM2.5 滚动均值和共污染物，只保留气象、PBLH、逆温、风输送、降水、复合扩散指数、气象时滞/累积特征、城市空间和时间周期变量。这样得到的精度通常低于高精度预测模型，但更适合评估大气稳定度、边界层高度、湿度、气压和风输送在不同时期的相对贡献。</p>
          <div class="intro-chip-row">
            <span>气象-only 归因口径</span>
            <span>248 个气象与时空特征</span>
            <span>raw / log1p / anomaly 三目标对照</span>
            <span>SHAP bootstrap 置信区间</span>
          </div>
        </div>
        <div class="intro-score-panel">
          <div class="intro-score-label">本轮最新归因训练</div>
          <h4>v2 分时期最佳模型</h4>
          <p>每个时期从 PM2.5 原值、log1p(PM2.5)、同城同月同小时气候态异常三种目标中选择测试 R2 表现最优者，用于阶段性结论。</p>
        </div>
      </section>
      {section_bridge_html(
        "分时期归因模型",
        "疫情前、疫情期、疫情后分别训练，避免时期差异被平均掉",
        "下面三张卡片对应本课题最核心的气象贡献比较。每个时期都单独训练，并从不同目标形式中选择最适合该时期的解释尺度，用于刻画气象因子贡献在疫情期和非疫情期的阶段性差异。",
        "green",
      )}
      <div class="intro-model-grid">
        <article class="intro-model-card green">
          <div class="model-card-tag">疫情前 2018-2019</div>
          <h4>{pre.get("target_label", "最佳目标")} 归因模型</h4>
          <p>疫情前模型用于刻画正常排放背景下气象扩散条件的解释力。最新结果中，季节周期、空间纬度、24小时平均 PBLH、北风清洁输送和气压滞后共同构成主要解释信号。</p>
          <div class="score-row compact">{score(pre, assets["pre_covid_meteorology_metrics"])}</div>
        </article>
        <article class="intro-model-card amber">
          <div class="model-card-tag">疫情期 2020-2022</div>
          <h4>{covid.get("target_label", "最佳目标")} 归因模型</h4>
          <p>疫情期模型单独训练，避免人为活动减弱时期被正常时期样本稀释。PBLH 滚动均值、南北风 V 分量、气压滞后和露点/湿度变量是解释 PM2.5 变化的重要气象信号。</p>
          <div class="score-row compact">{score(covid, assets["covid_meteorology_metrics"])}</div>
        </article>
        <article class="intro-model-card teal">
          <div class="model-card-tag">疫情后 2023+</div>
          <h4>{post.get("target_label", "最佳目标")} 归因模型</h4>
          <p>疫情后模型用于分析恢复期扩散约束的再强化过程。结果显示 24小时平均 PBLH、露点、通风系数、风速滚动均值和北风清洁输送在归因结果中更加突出。</p>
          <div class="score-row compact">{score(post, assets["post_covid_meteorology_metrics"])}</div>
        </article>
      </div>
      {section_bridge_html(
        "方法升级",
        "v2 气象贡献模型从当前小时气象升级为过程型气象解释",
        "这一组方法卡片说明 v2 相比旧版 baseline 的改进：不仅看当前小时，还把持续低 PBLH、弱风、高湿、通风系数、南北输送和降水持续过程纳入气象-only 特征矩阵，并用 SHAP bootstrap 检查贡献排序稳定性。",
        "green",
      )}
      <div class="method-band three">
        <div>
          <span class="method-band-kicker">特征工程</span>
          <h4>从单时刻气象扩展到过程型气象</h4>
          <p>v2 模型加入 PBLH、风速、U/V 风、相对湿度、气压、露点、降水和通风系数的 1/3/6/12/24/48/72 小时时滞、滚动均值或累计量，还构建了低 PBLH、弱风、高湿、南风输送、北风清洁输送和湿度-PBLH 交互等复合特征。这些特征仍属于气象变量，不会破坏气象归因属性。</p>
        </div>
        <div>
          <span class="method-band-kicker">目标函数</span>
          <h4>用多目标形式降低单一 PM2.5 原值的偏差</h4>
          <p>直接预测 PM2.5 原值容易被极端污染和排放背景影响。v2 同时比较原值、log1p(PM2.5) 和同城同月同小时气候态异常：log 目标降低重污染极端值影响，异常目标更聚焦气象扰动造成的偏离，原值目标便于解释实际浓度尺度。</p>
        </div>
        <div>
          <span class="method-band-kicker">稳健性</span>
          <h4>用分时期训练和 SHAP 置信区间支撑结论</h4>
          <p>疫情前、疫情期、疫情后分别训练，避免一个总模型把时期差异平均掉。SHAP 重要性使用 bootstrap 给出置信区间，使“哪个气象因子更重要”的说法不仅是单次排序，而有稳定性证据。</p>
        </div>
      </div>
    </div>
    """


def prediction_metric_html(label: str, value: str, status: str, accent: str, detail: str = "") -> str:
    detail_html = f'<div class="forecast-metric-detail">{detail}</div>' if detail else ""
    return f"""
    <div class="forecast-metric" style="--metric-color:{accent};">
      <div class="forecast-metric-label">{label}</div>
      <div class="forecast-metric-value">{value}</div>
      <div class="forecast-status"><span></span>{status}</div>
      {detail_html}
    </div>
    """


def scenario_summary_html(
    city: str,
    selected_date: date,
    hour: int,
    selected_model: dict,
    current_prediction: float,
    next24_prediction: float,
    category: str,
    next_category: str,
    color: str,
    next_color: str,
    t_inverse: float,
    pblh: float,
    row: dict,
    overrides: dict,
    wind_speed: float,
    wind_direction: float,
) -> str:
    inversion_status = "存在逆温" if t_inverse > 0 else "无逆温"
    pblh_status = "低边界层" if row["low_pblh_flag"] else "扩散较好"
    return f"""
    <div class="forecast-overview">
      {prediction_metric_html("当前小时 PM2.5", f"{current_prediction:.1f} ug/m3", category, color, "模型当前输出")}
      {prediction_metric_html("24 小时后 PM2.5", f"{next24_prediction:.1f} ug/m3", next_category, next_color, "趋势参考")}
      {prediction_metric_html("逆温指数", f"{t_inverse:.1f} C", inversion_status, "#d97706" if t_inverse > 0 else "#059669", "T850 - T1000")}
      {prediction_metric_html("边界层高度 PBLH", f"{pblh:.0f} m", pblh_status, "#dc2626" if row["low_pblh_flag"] else "#059669", "垂直扩散空间")}
    </div>
    <div class="forecast-panel">
      <div class="forecast-panel-main">
        <div class="forecast-place">{city} | {selected_date} {hour:02d}:00</div>
        <div class="forecast-number" style="color:{color};">{current_prediction:.1f}<span>ug/m3</span></div>
        <div class="forecast-model-line">当前模型：<b>{selected_model["label"]}</b><span>{selected_model["type"]}</span></div>
        <div class="forecast-main-note">{selected_model["description"]}</div>
        <div class="forecast-chip-row">
          <span>温湿度 {overrides['temperature_2m']:.1f} C / {overrides['relative_humidity_2m']:.0f}%</span>
          <span>风 {wind_speed:.1f} m/s @ {wind_direction:.0f} deg</span>
          <span>PBLH {pblh:.0f} m</span>
          <span>{inversion_status}</span>
        </div>
      </div>
      <div class="forecast-panel-side">
        <div class="forecast-side-item"><div>空气质量等级</div><strong style="color:{color};">{category}</strong></div>
        <div class="forecast-side-item"><div>24 小时后等级</div><strong style="color:{next_color};">{next_category}</strong></div>
        <div class="forecast-side-item"><div>关键扩散条件</div><strong>{pblh_status} / {inversion_status}</strong></div>
      </div>
    </div>
    <div class="scenario-strip">
      <div class="scenario-item"><div class="scenario-label">已确认日期</div><div class="scenario-value">{selected_date}</div></div>
      <div class="scenario-item"><div class="scenario-label">预测小时</div><div class="scenario-value">{hour:02d}:00</div></div>
      <div class="scenario-item wide"><div class="scenario-label">当前模型</div><div class="scenario-value">{selected_model["label"]}</div></div>
      <div class="scenario-item"><div class="scenario-label">温湿度</div><div class="scenario-value">{overrides['temperature_2m']:.1f} C / {overrides['relative_humidity_2m']:.0f}%</div></div>
      <div class="scenario-item"><div class="scenario-label">风</div><div class="scenario-value">{wind_speed:.1f} m/s @ {wind_direction:.0f} deg</div></div>
      <div class="scenario-item"><div class="scenario-label">PBLH</div><div class="scenario-value">{pblh:.0f} m</div></div>
    </div>
    """


def training_strategy_html(assets: Assets) -> str:
    metadata = assets["metadata"]
    current = assets["current_metrics"]
    covid_met = assets["covid_meteorology_metrics"]
    pre_met = assets["pre_covid_meteorology_metrics"]
    post_met = assets["post_covid_meteorology_metrics"]
    return f"""
    <div class="training-intro">
      <section class="training-hero">
        <div class="training-kicker">训练设计</div>
        <h3>围绕时间外推和变量控制设计训练流程</h3>
        <p>训练策略的核心不是简单堆模型，而是先固定时间切分，再用不同特征口径做对照实验：高精度模型检验应用预测能力，气象归因模型控制污染持续性和共污染物，专门考察气象、PBLH、逆温和风输送的独立解释力。</p>
        <div class="training-chip-row">
          <span>{metadata.get("start_time", "2018+")} 至 {metadata.get("end_time", "2026")}</span>
          <span>13 个城市</span>
          <span>PM2.5 缺失率 0%</span>
          <span>ERA5 PBLH 缺失率 {float(metadata.get("pblh_missing_rate", 0)):.2%}</span>
        </div>
      </section>
      <section class="training-score-card">
        <div class="training-score-label">主模型测试表现</div>
        <h4>全时期高精度模型</h4>
        <div class="score-row">{metric_badges(current)}</div>
        <p>最终测试集完全后置于训练和验证时段，用于检验模型在未来时段上的泛化能力，而不是随机抽样下的拟合能力。</p>
      </section>
    </div>

    {section_bridge_html(
      "特征口径",
      "高精度模型允许使用全部有助于预测的环境信息",
      "下面三张卡片先说明高精度模型的变量边界：它既使用气象和 PBLH，也使用污染持续性和共污染物。这样做是为了服务预测台和精度展示，因此读者需要把它和后面的气象-only 归因模型区分开。",
      "blue",
    )}
    <div class="training-family-grid">
      <section class="training-family-card blue">
        <div class="family-index">01</div>
        <h4>目标变量定义</h4>
        <p>当前小时模型预测 target_pm2_5；24 小时辅助模型预测 target_pm2_5_next_24h。所有样本按城市和时间形成 city-hour 监督学习表。</p>
        <div class="family-foot">避免把当前预测和提前量预测混作同一个任务</div>
      </section>
      <section class="training-family-card green">
        <div class="family-index">02</div>
        <h4>特征口径控制</h4>
        <p>extended 特征集包含共污染物和 PM2.5 持续性；meteorology 特征集主动剔除这些强预测变量，只保留气象、PBLH、稳定度、风输送和时空控制变量。</p>
        <div class="family-foot">把“预测增益”和“气象解释”分离</div>
      </section>
      <section class="training-family-card amber">
        <div class="family-index">03</div>
        <h4>分时期验证</h4>
        <p>疫情前、疫情期、疫情后分别训练和测试，避免疫情期人为活动变化被全时期模型平均掉，再比较不同阶段 SHAP 和残差结构。</p>
        <div class="family-foot">服务于 2020-2022 与非疫情期对比</div>
      </section>
    </div>

    <div class="training-flow">
      <div class="flow-step"><b>1</b><h4>数据整合</h4><p>CNEMC/quotsoft PM2.5、ERA5/CDS 气象、ERA5 PBLH 合并为 city-hour 表。</p></div>
      <div class="flow-step"><b>2</b><h4>特征构建</h4><p>PBLH、逆温指数、U/V 风分量、南北输送、时滞和时间周期统一进入特征矩阵。</p></div>
      <div class="flow-step"><b>3</b><h4>时间切分</h4><p>全时期和分时期模型都按时间后置验证、测试，避免随机切分造成信息泄漏。</p></div>
      <div class="flow-step"><b>4</b><h4>Optuna 调参</h4><p>以验证集 RMSE 为目标搜索 LightGBM 参数，使用 early stopping 控制过拟合。</p></div>
      <div class="flow-step"><b>5</b><h4>解释输出</h4><p>在测试集报告 MAE、RMSE、R2，并用 SHAP 与残差分析支撑气象贡献讨论。</p></div>
    </div>

    <div class="training-period-grid">
      <section class="training-period-card">
        <div class="period-tag">疫情前 2018-2019</div>
        <h4>正常时期基线</h4>
        <p>气象归因模型 R2 {pre_met["test"]["r2"]:.3f}，用于观察常规排放背景下气象条件的解释能力。</p>
      </section>
      <section class="training-period-card">
        <div class="period-tag">疫情期 2020-2022</div>
        <h4>人为活动减弱阶段</h4>
        <p>气象归因模型 R2 {covid_met["test"]["r2"]:.3f}，重点比较湿度、露点、风输送和 PBLH 的权重变化。</p>
      </section>
      <section class="training-period-card">
        <div class="period-tag">疫情后 2023+</div>
        <h4>恢复期对照</h4>
        <p>气象归因模型 R2 {post_met["test"]["r2"]:.3f}，用于分析边界层高度和低 PBLH 标记在恢复期的贡献变化。</p>
      </section>
    </div>
    """


def tuning_method_html(assets: Assets) -> str:
    current = assets["current_metrics"]
    covid_met = assets["covid_meteorology_metrics"]
    current_params = current.get("best_params", {})
    covid_params = covid_met.get("best_params", {})
    return f"""
    <div class="method-deep-dive">
      <section class="method-panel">
        <div class="method-kicker">关键方法 01</div>
        <h4>预处理与特征矩阵</h4>
        <p>数值特征进入 median imputer，并额外保留缺失指示列；类别特征先用众数补齐，再做 one-hot 编码，避免城市、时期等类别变量被错误当成连续数值。</p>
        <p>时间变量不只用 hour、month、dayofyear，也构建 sin/cos 周期特征，避免 23 点和 0 点、12 月和 1 月在模型中被误认为相距很远。</p>
      </section>
      <section class="method-panel">
        <div class="method-kicker">关键方法 02</div>
        <h4>时间切分原则</h4>
        <p>全时期模型使用固定时间后置验证：验证集从 2024-09-01 开始，测试集从 2024-10-01 开始。</p>
        <p>分时期模型在各自时期内部按时间顺序自动切为 70% 训练、15% 验证、15% 测试。没有使用随机切分，因为 PM2.5 强时间连续性会造成相邻小时泄漏。</p>
      </section>
    </div>

    <div class="tuning-board">
      <section class="tuning-main">
        <div class="method-kicker">关键方法 03</div>
        <h4>Optuna + LightGBM 调参方式</h4>
        <p>每个 trial 都在训练集拟合 LightGBM，并只用验证集 RMSE 作为优化目标。搜索过程不接触测试集，测试集只在最终模型确定后使用一次。</p>
        <div class="param-grid">
          <div class="param-row"><b>learning_rate</b><span>0.01 - 0.08，log 搜索</span></div>
          <div class="param-row"><b>num_leaves</b><span>31 - 255，控制树的复杂度</span></div>
          <div class="param-row"><b>max_depth</b><span>5 - 14，限制单棵树深度</span></div>
          <div class="param-row"><b>min_child_samples</b><span>10 - 160，控制叶节点最小样本</span></div>
          <div class="param-row"><b>subsample</b><span>0.65 - 1.00，行采样抑制过拟合</span></div>
          <div class="param-row"><b>colsample_bytree</b><span>0.65 - 1.00，列采样提升稳健性</span></div>
          <div class="param-row"><b>reg_alpha / reg_lambda</b><span>L1: 1e-4 - 10；L2: 1e-4 - 30</span></div>
          <div class="param-row"><b>min_split_gain</b><span>0 - 0.25，控制分裂收益门槛</span></div>
        </div>
      </section>
      <section class="tuning-side">
        <h4>本轮搜索轮数</h4>
        <div class="tuning-stat"><span>全时期高精度</span><b>{current.get("trials", "NA")} 轮</b></div>
        <div class="tuning-stat"><span>分时期高精度</span><b>25 轮/时期</b></div>
        <div class="tuning-stat"><span>分时期气象归因</span><b>12 轮/时期</b></div>
        <div class="tuning-stat"><span>early stopping</span><b>120 轮</b></div>
      </section>
    </div>

    <div class="training-detail-grid">
      <section class="training-detail-card">
        <h4>最终重训逻辑</h4>
        <p>候选模型先在训练集上搜索参数，并在验证集上 early stopping。确定最佳参数后，将训练集和验证集合并，重新拟合最终模型。</p>
        <p>最终 n_estimators 不是无限增大，而是取验证阶段 best_iteration 的约 1.08 倍，给合并训练留出少量余量。</p>
      </section>
      <section class="training-detail-card">
        <h4>最佳参数示例</h4>
        <p>全时期高精度模型：learning_rate {float(current_params.get("learning_rate", 0)):.4f}，num_leaves {current_params.get("num_leaves", "NA")}，max_depth {current_params.get("max_depth", "NA")}，best_iteration {current.get("best_iteration", "NA")}。</p>
        <p>疫情期气象归因模型：learning_rate {float(covid_params.get("learning_rate", 0)):.4f}，num_leaves {covid_params.get("num_leaves", "NA")}，max_depth {covid_params.get("max_depth", "NA")}，best_iteration {covid_met.get("best_iteration", "NA")}。</p>
      </section>
      <section class="training-detail-card">
        <h4>解释与稳健性输出</h4>
        <p>每个最终模型在测试集上报告 MAE、RMSE、R2；SHAP 使用测试集抽样计算平均绝对贡献，默认最多抽取 5000 行。</p>
        <p>气象残差只表示气象归因模型未解释部分，可辅助讨论非气象因素，不能直接等同于排放量变化。</p>
      </section>
    </div>
    """


def high_accuracy_training_html(assets: Assets) -> str:
    metadata = assets["metadata"]
    current = assets["current_metrics"]
    pre = assets["pre_covid_high_accuracy_metrics"]
    covid = assets["covid_high_accuracy_metrics"]
    post = assets["post_covid_high_accuracy_metrics"]
    return f"""
    <div class="training-intro">
      <section class="training-hero">
        <div class="training-kicker">高精度训练主线</div>
        <h3>用完整污染过程信息追求 PM2.5 应用预测精度</h3>
        <p>高精度模型的训练目标是尽可能准确地复原当前小时 PM2.5 浓度，因此特征矩阵采用 extended 口径：既包含气象和 ERA5 PBLH，也纳入 PM2.5 时滞、滚动均值、PM10、NO2、CO、SO2、O3、AOD、dust 等共污染物。这个口径不是为了做纯气象归因，而是为了给预测台提供稳定、低误差的应用预测结果，并作为气象贡献模型的精度上限参照。</p>
        <div class="training-chip-row">
          <span>{metadata.get("start_time", "2018+")} 至 {metadata.get("end_time", "2026")}</span>
          <span>{int(metadata.get("rows", 0)):,} 条 city-hour 样本</span>
          <span>{metadata.get("cities", 13)} 个城市</span>
          <span>PM2.5 缺失率 {float(metadata.get("pm25_missing_rate", 0)):.2%}</span>
        </div>
      </section>
      <section class="training-score-card">
        <div class="training-score-label">全时期主模型</div>
        <h4>测试集表现</h4>
        <div class="score-row">{metric_badges(current)}</div>
        <p>全时期主模型使用 71 个特征，Optuna 搜索 {current.get("trials", "NA")} 轮，测试集从 {current.get("test_start", "NA")} 开始，适合作为预测台默认模型。</p>
      </section>
    </div>

    {section_bridge_html(
      "特征口径",
      "高精度模型允许使用全部有助于预测的环境信息",
      "下面三张卡片先说明高精度模型的变量边界：它既使用气象和 PBLH，也使用污染持续性和共污染物。这样做是为了服务预测台和精度展示，因此读者需要把它和后面的气象-only 归因模型区分开。",
      "blue",
    )}
    <div class="training-family-grid">
      <section class="training-family-card blue">
        <div class="family-index">01</div>
        <h4>多源特征矩阵构建</h4>
        <p>基础气象变量包括气温、露点、相对湿度、气压、降水、云量、风速、风向、阵风和 ERA5 PBLH。风向被拆解为 U/V 分量，进一步构造南风污染输送和北风清洁输送，使模型能理解京津冀区域输送方向。</p>
        <div class="family-foot">气象 + PBLH + 稳定度 + 风输送</div>
      </section>
      <section class="training-family-card green">
        <div class="family-index">02</div>
        <h4>污染持续性信息的预测增益</h4>
        <p>PM2.5 的短时变化具有明显记忆性，前 1/3/24 小时 PM2.5 和滚动均值能表达污染团累积、滞留和消散过程。共污染物则提供同一污染过程中的化学和排放背景信息，因此能显著提高预测精度。</p>
        <div class="family-foot">这是预测能力来源，不是纯气象解释</div>
      </section>
      <section class="training-family-card amber">
        <div class="family-index">03</div>
        <h4>分时期建模的对照意义</h4>
        <p>全时期模型学习总体规律，分时期高精度模型分别学习疫情前、疫情期、疫情后的污染持续性和共污染物结构，从而比较疫情期人为活动变化对预测难度和特征贡献结构的影响。</p>
        <div class="family-foot">全时期用于应用，分时期用于对照</div>
      </section>
    </div>

    {section_bridge_html(
      "训练流水线",
      "从数据补齐到最终评估的五个步骤",
      "这一排流程卡片用于交代模型指标生成机制。重点是时间顺序切分、验证集调参、测试集最终评估三个环节，避免把相邻小时样本随机打散后造成过高估计。",
      "blue",
    )}
    <div class="training-flow">
      <div class="flow-step"><b>1</b><h4>数据补齐</h4><p>2018-2022 CNEMC/quotsoft PM2.5 与 ERA5 气象、ERA5 PBLH 合并；缺失原始日按城市和污染物线性插补。</p></div>
      <div class="flow-step"><b>2</b><h4>特征生成</h4><p>生成时间周期、PM2.5 时滞/滚动、共污染物、PBLH、逆温、U/V 风和南北输送等特征。</p></div>
      <div class="flow-step"><b>3</b><h4>时间切分</h4><p>全时期模型固定后置验证和测试；分时期模型在每个时期内按时间顺序 70/15/15 切分。</p></div>
      <div class="flow-step"><b>4</b><h4>调参重训</h4><p>Optuna 在验证集 RMSE 上搜索 LightGBM 参数；最佳参数确定后合并训练集和验证集重训最终模型。</p></div>
      <div class="flow-step"><b>5</b><h4>测试解释</h4><p>测试集只在最终阶段使用，报告 MAE、RMSE、R2，并用 SHAP 判断预测信息主要来自哪些变量。</p></div>
    </div>

    {section_bridge_html(
      "时期对照",
      "全时期模型用于应用预测，分时期模型用于比较特殊阶段",
      "下面的时期对照把全时期、疫情期和非疫情期放在同一阅读层级中。它强调的是预测上限和时期差异，不直接解释人为活动强度，也不替代气象归因模型的结论。",
      "slate",
    )}
    <div class="method-band three">
      <div>
        <span class="method-band-kicker">全时期</span>
        <h4>统一主模型</h4>
        <p>训练 {int(current.get("train_rows", 0)):,} 行，验证 {int(current.get("valid_rows", 0)):,} 行，测试 {int(current.get("test_rows", 0)):,} 行；R2 {current["test"]["r2"]:.3f}，RMSE {current["test"]["rmse"]:.2f}。</p>
      </div>
      <div>
        <span class="method-band-kicker">疫情期</span>
        <h4>特殊时期预测对照</h4>
        <p>2020-2022 单独训练，测试 R2 {covid["test"]["r2"]:.3f}，RMSE {covid["test"]["rmse"]:.2f}。它用于检验疫情期污染过程的短时可预测性和模型稳定性。</p>
      </div>
      <div>
        <span class="method-band-kicker">非疫情期</span>
        <h4>疫情前/后参照</h4>
        <p>疫情前 R2 {pre["test"]["r2"]:.3f}，疫情后 R2 {post["test"]["r2"]:.3f}。两个时期用于和疫情期比较预测上限，而不是直接解释排放强度。</p>
      </div>
    </div>
    """


def high_accuracy_tuning_html(assets: Assets) -> str:
    current = assets["current_metrics"]
    current_params = current.get("best_params", {})
    pre = assets["pre_covid_high_accuracy_metrics"]
    covid = assets["covid_high_accuracy_metrics"]
    post = assets["post_covid_high_accuracy_metrics"]
    return f"""
    {section_bridge_html(
      "调参框架",
      "Optuna 搜索与高精度预测优化",
      "这个模块解释模型不是手工随意设参，而是围绕验证集 RMSE 系统搜索 LightGBM 的树复杂度、学习率、采样比例和正则强度。它对应论文方法章节中的模型优化部分。",
      "blue",
    )}
    <div class="tuning-board">
      <section class="tuning-main">
        <div class="method-kicker">调参方式</div>
        <h4>高精度模型的 Optuna 搜索逻辑</h4>
        <p>高精度模型以验证集 RMSE 为唯一优化目标。每个 trial 训练一套 LightGBM 参数，并通过 early stopping 监测验证集误差的收敛趋势。测试集不参与参数选择，避免“看着测试集调模型”。</p>
        <p>搜索空间覆盖树复杂度、学习率、采样比例和正则强度。这样的设计比手工固定参数更稳，因为 PM2.5 数据既有强短时持续性，也有跨季节、跨城市的非线性差异。</p>
        <div class="param-grid">
          <div class="param-row"><b>learning_rate</b><span>控制每棵树对最终预测的贡献，较小学习率配合更多迭代提升稳定性。</span></div>
          <div class="param-row"><b>num_leaves / max_depth</b><span>控制非线性表达能力，避免模型只拟合简单线性关系，也避免树过深记住噪声。</span></div>
          <div class="param-row"><b>min_child_samples</b><span>限制叶节点最小样本量，减少小样本时段或城市造成的局部过拟合。</span></div>
          <div class="param-row"><b>subsample / colsample</b><span>通过行采样和列采样提升泛化能力，减少某几个强特征完全支配模型。</span></div>
          <div class="param-row"><b>reg_alpha / reg_lambda</b><span>L1/L2 正则项抑制过大的叶节点权重，让模型在未来时段更稳。</span></div>
          <div class="param-row"><b>best_iteration</b><span>由验证集 early stopping 决定，最终重训时按最佳迭代略放宽。</span></div>
        </div>
      </section>
      <section class="tuning-side">
        <h4>高精度模型训练规模</h4>
        <div class="tuning-stat"><span>全时期搜索</span><b>{current.get("trials", "NA")} 轮</b></div>
        <div class="tuning-stat"><span>疫情前搜索</span><b>{pre.get("trials", "NA")} 轮</b></div>
        <div class="tuning-stat"><span>疫情期搜索</span><b>{covid.get("trials", "NA")} 轮</b></div>
        <div class="tuning-stat"><span>疫情后搜索</span><b>{post.get("trials", "NA")} 轮</b></div>
        <div class="tuning-stat"><span>全时期最佳迭代</span><b>{current.get("best_iteration", "NA")}</b></div>
      </section>
    </div>

    {section_bridge_html(
      "迭代收尾",
      "最佳参数到最终模型的转化流程",
      "这组三张卡片说明训练结束后的处理方式：先合并训练集和验证集重训，再固定最终模型输出测试集指标和 SHAP 解释。这样读者能追溯最终分数从哪里来，也能理解高精度 SHAP 的解释边界。",
      "blue",
    )}
    <div class="training-detail-grid">
      <section class="training-detail-card">
        <h4>最终重训</h4>
        <p>调参完成后，不直接保存验证阶段模型，而是把训练集和验证集合并，用最佳参数重新训练最终模型。这样既保留了时间后置验证，又尽量利用可用于训练的历史样本。</p>
      </section>
      <section class="training-detail-card">
        <h4>最佳参数示例</h4>
        <p>全时期高精度模型的 learning_rate 为 {float(current_params.get("learning_rate", 0)):.4f}，num_leaves 为 {current_params.get("num_leaves", "NA")}，max_depth 为 {current_params.get("max_depth", "NA")}，best_iteration 为 {current.get("best_iteration", "NA")}。</p>
      </section>
      <section class="training-detail-card">
        <h4>结果解释口径</h4>
        <p>高精度模型的 SHAP 前列通常包含 PM2.5 滞后、滚动均值和共污染物。这说明模型预测得准，但归因解释不能简单写成“气象变量贡献最大”。</p>
      </section>
    </div>
    """


def meteorology_training_html(assets: Assets) -> str:
    metadata = assets["metadata"]
    v2 = assets.get("meteorology_v2") or {}
    summary_rows = v2.get("summary", [])
    best_rows = v2.get("best_summary", [])
    first_row = summary_rows[0] if summary_rows else {}
    feature_count = int(first_row.get("feature_count", 248))
    candidate_count = len(summary_rows) or 9
    best_count = len(best_rows) or 3
    return f"""
    <div class="training-intro">
      <section class="training-hero">
        <div class="training-kicker">气象归因训练策略</div>
        <h3>从 city-hour 样本到分时期气象贡献结论的训练流程</h3>
        <p>这一页只说明训练方法，不重复模型介绍。气象归因训练先把每条城市小时样本归入疫情前、疫情期、疫情后三个时期；随后执行变量准入控制，禁止污染历史和共污染物进入特征矩阵；再生成气象时滞、滚动、累计和复合扩散特征；最后在每个时期分别训练三种目标形式，并用统一评估和 SHAP bootstrap 输出气象贡献证据。</p>
        <div class="training-chip-row">
          <span>{metadata.get("start_time", "2018+")} 至 {metadata.get("end_time", "2026")}</span>
          <span>3 个时期独立训练</span>
          <span>{candidate_count} 套 v2 候选实验</span>
          <span>{feature_count} 个气象与时空特征</span>
        </div>
      </section>
      <section class="training-score-card">
        <div class="training-score-label">实验矩阵</div>
        <h4>3 个时期 x 3 种目标形式</h4>
        <div class="tuning-stat"><span>时期分组</span><b>疫情前 / 疫情期 / 疫情后</b></div>
        <div class="tuning-stat"><span>目标形式</span><b>raw / log1p / anomaly</b></div>
        <div class="tuning-stat"><span>最终报告</span><b>{best_count} 套时期最佳</b></div>
        <p>参数搜索和模型比较在同一训练规范下完成，使三个时期的气象贡献具有可比性。</p>
      </section>
    </div>

    {section_bridge_html(
      "",
      "训练数据与时期切分",
      "",
      "green",
    )}
    <div class="training-family-grid">
      <section class="training-family-card green">
        <div class="family-index">01</div>
        <h4>样本单元固定为 city-hour</h4>
        <p>每一行代表一个城市在一个小时的 PM2.5 与同步气象状态。训练目标不在站点随机抽样层面定义，而是在城市小时序列上构建监督学习表。</p>
        <div class="family-foot">保证污染过程具有时间顺序</div>
      </section>
      <section class="training-family-card amber">
        <div class="family-index">02</div>
        <h4>时期掩码先于模型训练</h4>
        <p>2018-2019、2020-2022、2023+ 三段先拆开，再分别训练。疫情期样本不会和非疫情期混合拟合，避免特殊时期被全时期平均效应掩盖。</p>
        <div class="family-foot">服务疫情期与非疫情期对比</div>
      </section>
      <section class="training-family-card blue">
        <div class="family-index">03</div>
        <h4>时间后置训练验证测试</h4>
        <p>每个时期内部按时间顺序划分训练、验证、测试，不使用随机切分。验证集用于调参和 early stopping，测试集用于最终报告。</p>
        <div class="family-foot">降低相邻小时泄漏风险</div>
      </section>
    </div>

    {section_bridge_html(
      "",
      "变量准入与泄漏控制",
      "",
      "green",
    )}
    <div class="training-detail-grid">
      <section class="training-detail-card">
        <h4>禁止进入特征矩阵</h4>
        <p>PM2.5 时滞、PM2.5 滚动均值、PM10、NO2、CO、SO2、O3、AOD、dust 和目标派生列全部剔除，避免污染持续性和共污染物替代气象解释。</p>
      </section>
      <section class="training-detail-card">
        <h4>允许进入特征矩阵</h4>
        <p>保留气温、露点、湿度、气压、降水、云量、风速、U/V 风、PBLH、稳定度、南北输送、城市空间和时间周期变量。</p>
      </section>
      <section class="training-detail-card">
        <h4>预处理执行规则</h4>
        <p>数值特征采用缺失补齐并保留可追踪的缺失状态；城市、时期、天气型等类别变量以类别编码进入模型，避免被当作连续数值。</p>
      </section>
    </div>

    {section_bridge_html(
      "",
      "气象特征生成顺序",
      "",
      "green",
    )}
    <div class="training-flow">
      <div class="flow-step"><b>1</b><h4>原始气象入表</h4><p>近地面温湿压、降水、云量、风速风向、PBLH 和压力层温度先与 PM2.5 按城市小时对齐。</p></div>
      <div class="flow-step"><b>2</b><h4>风向矢量化</h4><p>将角度风向转化为 U/V 分量，并生成南风输送、北风清洁输送等方向性传输特征。</p></div>
      <div class="flow-step"><b>3</b><h4>稳定度量化</h4><p>构建逆温指数、低 PBLH 标记、弱风标记、高湿标记和通风系数，表达垂直扩散能力。</p></div>
      <div class="flow-step"><b>4</b><h4>滞后累计生成</h4><p>为 PBLH、风速、U/V 风、湿度、气压、露点、降水和通风系数生成时滞、滚动均值或累计量。</p></div>
      <div class="flow-step"><b>5</b><h4>天气型辅助分层</h4><p>使用气象变量聚类形成 weather_type_k6，用于后续条件误差分析和 SHAP 解释分层。</p></div>
    </div>

    {section_bridge_html(
      "",
      "目标变量构造与候选训练矩阵",
      "",
      "green",
    )}
    <div class="method-band three">
      <div>
        <span class="method-band-kicker">raw</span>
        <h4>PM2.5 原值目标</h4>
        <p>直接拟合实际浓度尺度，便于报告 RMSE、MAE 和偏差；该目标对重污染极端值更敏感。</p>
      </div>
      <div>
        <span class="method-band-kicker">log1p</span>
        <h4>对数变换目标</h4>
        <p>训练阶段拟合 log1p(PM2.5)，预测后还原到浓度尺度，用于减弱极端污染小时对参数学习的牵引。</p>
      </div>
      <div>
        <span class="method-band-kicker">anomaly</span>
        <h4>气候态异常目标</h4>
        <p>拟合同城同月同小时气候态偏差，突出气象扰动造成的浓度偏离，降低城市季节基线差异影响。</p>
      </div>
    </div>
    """


def meteorology_tuning_html(assets: Assets) -> str:
    pre_old = assets["pre_covid_meteorology_metrics"]
    covid_old = assets["covid_meteorology_metrics"]
    post_old = assets["post_covid_meteorology_metrics"]
    v2 = assets.get("meteorology_v2") or {}
    summary_rows = v2.get("summary", [])
    candidate_count = len(summary_rows) or 9
    return f"""
    {section_bridge_html(
      "",
      "参数搜索与候选模型选择",
      "",
      "green",
    )}
    <div class="tuning-board">
      <section class="tuning-main">
        <div class="method-kicker">LightGBM 训练执行</div>
        <h4>同一搜索空间约束下训练全部候选模型</h4>
        <p>每个时期分别训练 raw、log1p、anomaly 三类目标，形成 {candidate_count} 套 v2 候选模型。每套候选都只在训练集拟合，在验证集上进行 early stopping 和参数选择，测试集保留到最终比较阶段。</p>
        <p>候选模型最终统一还原到 PM2.5 浓度尺度，并在同一后置测试集上报告 R2、RMSE、MAE、Bias。这样比较的对象不是模型类型差异，而是目标尺度和时期背景差异。</p>
        <div class="param-grid">
          <div class="param-row"><b>objective</b><span>以验证集 RMSE 作为主优化指标，保证不同目标形式可比较。</span></div>
          <div class="param-row"><b>early stopping</b><span>验证集误差不再改善时停止迭代，避免树模型继续拟合噪声。</span></div>
          <div class="param-row"><b>search space</b><span>调节学习率、叶节点数、树深度、叶节点样本数、采样比例和正则强度。</span></div>
          <div class="param-row"><b>final report</b><span>最终报告统一使用测试集浓度尺度指标和归因解释输出。</span></div>
        </div>
      </section>
      <section class="tuning-side">
        <h4>训练对照信息</h4>
        <div class="tuning-stat"><span>疫情前旧版 R2</span><b>{pre_old["test"]["r2"]:.3f}</b></div>
        <div class="tuning-stat"><span>疫情期旧版 R2</span><b>{covid_old["test"]["r2"]:.3f}</b></div>
        <div class="tuning-stat"><span>疫情后旧版 R2</span><b>{post_old["test"]["r2"]:.3f}</b></div>
        <div class="tuning-stat"><span>旧版调参</span><b>{covid_old.get("trials", "NA")} 轮/时期</b></div>
        <div class="tuning-stat"><span>v2 候选模型</span><b>{candidate_count} 套</b></div>
      </section>
    </div>

    {section_bridge_html(
      "",
      "评估分层与归因产出",
      "",
      "green",
    )}
    <div class="training-detail-grid">
      <section class="training-detail-card">
        <h4>总体误差评估</h4>
        <p>每套候选模型报告 R2、RMSE、MAE 和 Bias。该层评估用于判断气象-only 特征对 PM2.5 浓度变化的总体解释能力。</p>
      </section>
      <section class="training-detail-card">
        <h4>条件误差分层</h4>
        <p>测试集按低 PBLH、弱风、高湿、低通风、南北输送和天气型分层，检验模型在关键污染气象条件下的误差结构。</p>
      </section>
      <section class="training-detail-card">
        <h4>SHAP bootstrap 输出</h4>
        <p>在测试集抽样计算 SHAP，并对贡献排序重复抽样生成均值、标准差和 95% 置信区间，用于比较不同时期气象因子贡献稳定性。</p>
      </section>
    </div>
    """


def high_accuracy_training_rows(assets: Assets) -> pd.DataFrame:
    rows = training_strategy_rows(assets)
    return rows[rows["类型"].isin(["预测主模型", "分时期预测", "提前量预测"])].reset_index(drop=True)


def meteorology_legacy_training_rows(assets: Assets) -> pd.DataFrame:
    rows = training_strategy_rows(assets)
    return rows[rows["类型"] == "气象归因"].reset_index(drop=True)


def meteorology_v2_training_rows(assets: Assets) -> pd.DataFrame:
    table = meteorology_v2_summary_table(assets)
    if table.empty:
        return table
    table = table.copy()
    table.insert(0, "模型代际", "v2 气象贡献")
    return table


def metric_row(name: str, metrics: Metrics, role: str) -> dict[str, object]:
    test = metrics["test"]
    return {
        "模型": name,
        "定位": role,
        "MAE": round(float(test["mae"]), 3),
        "RMSE": round(float(test["rmse"]), 3),
        "R2": round(float(test["r2"]), 3),
    }


def feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def is_meteorology_feature(feature: str) -> bool:
    lower = feature.lower()
    return any(hint in lower for hint in METEOROLOGY_FEATURE_HINTS)


def top_shap_table(shap_df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    table = shap_df.head(n)[["feature", "mean_abs_shap"]].copy()
    table["特征"] = table["feature"].map(feature_label)
    table["平均绝对 SHAP"] = table["mean_abs_shap"].round(3)
    return table[["特征", "平均绝对 SHAP"]]


def meteorology_shap_table(shap_df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    weather_df = shap_df[shap_df["feature"].map(is_meteorology_feature)]
    return top_shap_table(weather_df, n)


def high_accuracy_performance_table(assets: Assets) -> pd.DataFrame:
    return pd.DataFrame(
        [
            metric_row("全时期高精度", assets["current_metrics"], "预测台主模型"),
            metric_row("疫情前高精度", assets["pre_covid_high_accuracy_metrics"], "2018-2019 预测对照"),
            metric_row("疫情期高精度", assets["covid_high_accuracy_metrics"], "2020-2022 预测对照"),
            metric_row("疫情后高精度", assets["post_covid_high_accuracy_metrics"], "2023+ 预测对照"),
            metric_row("24 小时辅助", assets["next24_metrics"], "旧口径辅助预测"),
        ]
    )


def legacy_meteorology_performance_table(assets: Assets) -> pd.DataFrame:
    return pd.DataFrame(
        [
            metric_row("疫情前旧版气象归因", assets["pre_covid_meteorology_metrics"], "2018-2019 气象-only 原值模型"),
            metric_row("疫情期旧版气象归因", assets["covid_meteorology_metrics"], "2020-2022 气象-only 原值模型"),
            metric_row("疫情后旧版气象归因", assets["post_covid_meteorology_metrics"], "2023+ 气象-only 原值模型"),
        ]
    )


def meteorology_v2_summary_table(assets: Assets) -> pd.DataFrame:
    v2 = assets.get("meteorology_v2")
    if not v2:
        return pd.DataFrame()
    rows = []
    for row in v2.get("summary", []):
        rows.append(
            {
                "时期": row["period_label"],
                "目标形式": row["target_label"],
                "R2": round(float(row["r2"]), 3),
                "RMSE": round(float(row["rmse"]), 3),
                "MAE": round(float(row["mae"]), 3),
                "Bias": round(float(row["bias"]), 3),
                "训练样本": f"{int(row['train_rows']):,}",
                "验证样本": f"{int(row['valid_rows']):,}",
                "测试样本": f"{int(row['test_rows']):,}",
                "特征数": int(row["feature_count"]),
                "最佳迭代": int(row["best_iteration"]),
            }
        )
    return pd.DataFrame(rows)


def meteorology_v2_best_table(assets: Assets) -> pd.DataFrame:
    v2 = assets.get("meteorology_v2")
    if not v2:
        return pd.DataFrame()
    rows = []
    for row in v2.get("best_summary", []):
        rows.append(
            {
                "时期": row["period_label"],
                "最佳目标": row["target_label"],
                "R2": round(float(row["r2"]), 3),
                "RMSE": round(float(row["rmse"]), 3),
                "MAE": round(float(row["mae"]), 3),
                "Bias": round(float(row["bias"]), 3),
                "训练/验证/测试": f"{int(row['train_rows']):,} / {int(row['valid_rows']):,} / {int(row['test_rows']):,}",
                "特征数": int(row["feature_count"]),
                "最佳迭代": int(row["best_iteration"]),
            }
        )
    return pd.DataFrame(rows)


def meteorology_v2_old_new_compare_table(assets: Assets) -> pd.DataFrame:
    v2_best = meteorology_v2_best_table(assets)
    legacy = {
        "疫情前": assets["pre_covid_meteorology_metrics"]["test"],
        "疫情期": assets["covid_meteorology_metrics"]["test"],
        "疫情后": assets["post_covid_meteorology_metrics"]["test"],
    }
    rows = []
    for _, row in v2_best.iterrows():
        old = legacy.get(row["时期"], {})
        rows.append(
            {
                "时期": row["时期"],
                "旧版气象 R2": round(float(old.get("r2", 0)), 3),
                "v2 最佳 R2": row["R2"],
                "R2 提升": round(float(row["R2"]) - float(old.get("r2", 0)), 3),
                "旧版 RMSE": round(float(old.get("rmse", 0)), 2),
                "v2 最佳 RMSE": round(float(row["RMSE"]), 2),
                "v2 最佳目标": row["最佳目标"],
            }
        )
    return pd.DataFrame(rows)


def meteorology_v2_shap_table(model: dict[str, Any], n: int = 10) -> pd.DataFrame:
    rows = []
    for item in model.get("top_shap", [])[:n]:
        rows.append(
            {
                "特征": feature_label(item["feature"]),
                "平均绝对 SHAP": round(float(item["mean_abs_shap"]), 3),
                "Bootstrap 均值": round(float(item.get("bootstrap_mean_abs_shap", item["mean_abs_shap"])), 3),
                "95% CI 下限": round(float(item.get("bootstrap_ci_low", 0)), 3),
                "95% CI 上限": round(float(item.get("bootstrap_ci_high", 0)), 3),
            }
        )
    return pd.DataFrame(rows)


def meteorology_v2_shap_chart(model: dict[str, Any]) -> go.Figure:
    data = meteorology_v2_shap_table(model, 12).copy()
    data = data.sort_values("平均绝对 SHAP")
    fig = px.bar(
        data,
        x="平均绝对 SHAP",
        y="特征",
        orientation="h",
        title=f"{model['period_label']} v2 最佳气象贡献 Top SHAP",
    )
    fig.update_traces(marker_color="#059669")
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=20), xaxis_title="平均绝对 SHAP", yaxis_title="")
    return fig


def meteorology_v2_r2_chart(summary: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        summary,
        x="时期",
        y="R2",
        color="目标形式",
        barmode="group",
        text="R2",
        title="v2 气象贡献模型三目标 R2 对比",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=20), yaxis_range=[0, 1.0], xaxis_title="")
    return fig


def v2_period_narrative(model: dict[str, Any]) -> str:
    period = model["period"]
    if period == "pre_covid_2018_2019":
        return (
            "疫情前最佳模型选择 log1p(PM2.5)，说明在正常排放背景下，降低重污染极端值对训练的干扰后，"
            "气象变量能够更稳定地解释浓度变化。SHAP 前列包含年内周期、纬度、24小时平均 PBLH、"
            "北风清洁输送和72小时气压滞后，反映出季节背景、城市南北空间差异、边界层扩散和冷空气过程共同控制污染累积。"
        )
    if period == "covid_2020_2022":
        return (
            "疫情期最佳模型选择 PM2.5 原值，R2 在三目标中最高，说明人为活动减弱并不意味着气象解释力消失。"
            "24小时平均 PBLH、24/48小时南北风 V 分量、72小时气压滞后与露点/湿度相关变量共同进入前列，"
            "表明静稳扩散条件、区域输送和湿度过程仍然是疫情期 PM2.5 波动的重要气象驱动。"
        )
    return (
        "疫情后最佳模型选择同城同月同小时气候态异常，说明恢复期更适合解释 PM2.5 相对本地气候背景的偏离。"
        "24小时平均 PBLH、露点、24小时通风系数、12小时平均风速和北风清洁输送位于前列，"
        "体现边界层约束、湿度过程和通风扩散能力对污染异常的控制作用增强。"
    )


def metrics_overview(metrics: Metrics) -> pd.DataFrame:
    rows = [
        ("训练样本", f"{metrics.get('train_rows', 0):,}"),
        ("验证样本", f"{metrics.get('valid_rows', 0):,}"),
        ("测试样本", f"{metrics.get('test_rows', 0):,}"),
        ("特征数", f"{metrics.get('feature_count', 0):,}"),
        ("调参轮数", f"{metrics.get('trials', 'NA')}"),
        ("最佳迭代", f"{metrics.get('best_iteration', 'NA')}"),
        ("验证起点", str(metrics.get("valid_start", "NA"))),
        ("测试起点", str(metrics.get("test_start", "NA"))),
    ]
    return pd.DataFrame(rows, columns=["项目", "值"])


def render_model_card(spec: dict, assets: Assets) -> None:
    metrics = assets[spec["metrics_key"]]
    shap_df = assets.get(spec.get("shap_key"))
    test = metrics["test"]
    title = (
        f"{spec['title']} | {spec['family']} | "
        f"R2 {test['r2']:.3f} / RMSE {test['rmse']:.2f}"
    )
    with st.expander(title, expanded=spec.get("expanded", False)):
        metric_cols = st.columns(3)
        metric_cols[0].metric("测试 MAE", f"{test['mae']:.2f}")
        metric_cols[1].metric("测试 RMSE", f"{test['rmse']:.2f}")
        metric_cols[2].metric("测试 R2", f"{test['r2']:.3f}")
        st.markdown(f"**定位**：{spec['role']}")
        st.markdown(f"**研究任务定位**：{spec['question']}")
        st.markdown(f"**特征口径**：{spec['features']}")
        st.markdown(f"**学术使用建议**：{spec['academic_use']}")
        if spec.get("result_reading"):
            st.markdown(f"**结果解读**：{spec['result_reading']}")
        if spec.get("limitation"):
            st.markdown(f"**使用边界**：{spec['limitation']}")
        st.caption(spec["note"])
        detail_a, detail_b = st.columns([0.42, 0.58])
        with detail_a:
            st.dataframe(metrics_overview(metrics), use_container_width=True, hide_index=True)
        with detail_b:
            if shap_df is not None:
                st.markdown("**总体 Top SHAP**")
                st.dataframe(top_shap_table(shap_df, 8), use_container_width=True, hide_index=True)
                st.markdown("**气象因子 Top SHAP**")
                st.dataframe(meteorology_shap_table(shap_df, 8), use_container_width=True, hide_index=True)
                st.caption("总体表会包含时间周期、城市/经纬度、污染滞后等控制变量；气象因子表只保留温度、湿度、气压、PBLH、逆温、风输送、降水和云量等气象变量。")


def model_card_specs() -> list[dict]:
    return [
        {
            "title": "全时期高精度模型",
            "family": "预测主模型",
            "metrics_key": "current_metrics",
            "shap_key": "current_shap",
            "role": "预测台默认模型，用于 2018+ 全时期当前小时 PM2.5 估计。",
            "question": "评估当前气象、PM2.5 历史和共污染物背景约束下的 PM2.5 当前小时预测能力。",
            "features": "气象、ERA5 PBLH、逆温、风输送、PM2.5 时滞、滚动均值、PM10、CO、NO2、SO2、O3、AOD、dust。",
            "academic_use": "用于证明机器学习框架具备较强预测能力，不直接作为气象因子独立贡献的唯一依据。",
            "result_reading": "该模型精度最高，说明在跨年份、跨城市场景下，LightGBM 能有效捕捉污染持续性、共污染物协同变化和气象扩散条件之间的非线性关系。它是网页预测台最稳妥的默认选择。",
            "limitation": "SHAP 前列会包含 PM2.5 时滞、滚动均值和共污染物，因此它更适合说明预测能力，而不是单独说明气象因子贡献。",
            "note": "该模型跨越 2018-2026，2023+ PM2.5 数据源与 2018-2022 不完全一致，跨期解释需谨慎。",
            "expanded": True,
        },
        {
            "title": "疫情前高精度模型",
            "family": "分时期预测",
            "metrics_key": "pre_covid_high_accuracy_metrics",
            "shap_key": "pre_covid_high_accuracy_shap",
            "role": "2018-2019 单独训练的高精度预测模型。",
            "question": "量化正常排放时期污染持续性和共污染物信息对 PM2.5 短时预测上限的提升作用。",
            "features": "与全时期高精度模型同类，但只在疫情前数据内训练和测试。",
            "academic_use": "作为疫情前预测精度上限和气象归因模型的对照。",
            "result_reading": "它提供了疫情前正常排放背景下的预测上限。若该模型显著优于疫情前气象模型，说明 PM2.5 自身持续性和共污染物背景对短时预测有很强贡献。",
            "limitation": "该模型仍包含污染历史和共污染物，不能直接用于判断 PBLH、湿度、风和逆温的独立贡献强弱。",
            "note": "该模型 SHAP 前列主要为 PM2.5 滚动均值、时滞和 PM10，说明预测能力主要来自污染持续性。",
        },
        {
            "title": "疫情前气象归因模型",
            "family": "气象归因",
            "metrics_key": "pre_covid_meteorology_metrics",
            "shap_key": "pre_covid_meteorology_shap",
            "role": "2018-2019 单独训练，只保留气象、PBLH、风输送、稳定度、时间和城市特征。",
            "question": "正常排放时期，气象背景本身能解释多少 PM2.5 变化。",
            "features": "排除 PM2.5 时滞、滚动均值和共污染物，突出 PBLH、湿度、风、气压、逆温等气象贡献。",
            "academic_use": "论文中解释疫情前气象贡献的核心模型。",
            "result_reading": "旧版模型已经能显示气象-only 口径与高精度口径之间的差距，说明气象对 PM2.5 有解释力，但不足以替代污染持续性和排放背景。",
            "limitation": "旧版模型主要使用当前小时气象和少量基础派生变量，缺少本轮 v2 中的长时滞、累积和复合扩散指数，因此最新论文结论应优先引用 v2 结果。",
            "note": "精度显著低于高精度模型是预期现象，因为它故意不使用强污染持续性特征。",
        },
        {
            "title": "疫情期高精度模型",
            "family": "分时期预测",
            "metrics_key": "covid_high_accuracy_metrics",
            "shap_key": "covid_high_accuracy_shap",
            "role": "2020-2022 单独训练的高精度预测模型。",
            "question": "评估疫情期污染持续性和共污染物结构变化背景下的 PM2.5 短时预测稳定性。",
            "features": "气象 + PBLH + 稳定度 + PM2.5 时滞/滚动均值 + 共污染物。",
            "academic_use": "用于展示疫情期预测能力，并和疫情期气象归因模型形成对照。",
            "result_reading": "疫情期高精度模型可用于检验特殊时期污染过程的可学习短时持续性。它的表现越好，越说明即便人为活动下降，污染过程仍保留连续演变特征。",
            "limitation": "该模型不适合直接判定疫情期气象因子贡献的增强或减弱，因为污染历史变量会吸收一部分非气象因素。",
            "note": "高 R2 主要说明污染短时持续性强，不表示气象变量单独解释了全部污染变化。",
        },
        {
            "title": "疫情期气象归因模型",
            "family": "气象归因",
            "metrics_key": "covid_meteorology_metrics",
            "shap_key": "covid_meteorology_shap",
            "role": "2020-2022 单独训练，专门观察人为活动减弱背景下气象贡献结构。",
            "question": "比较疫情期气象条件对 PM2.5 的解释权重相对于疫情前的阶段性变化。",
            "features": "排除 PM2.5 持续性和共污染物，只看气象、PBLH、逆温、风输送、时间和城市。",
            "academic_use": "论文中疫情期气象贡献分析的核心模型。",
            "result_reading": "旧版疫情期气象模型为疫情三年单独建模，能初步显示人为活动减弱背景下气象变量解释结构的变化。",
            "limitation": "由于旧版特征主要集中在当前小时，无法充分表达持续低边界层、连续高湿、长时间弱风和通风条件累积效应；相关方法局限已在 v2 模型中改进。",
            "note": "适合解释露点、湿度、季节周期、风输送等在疫情期的贡献变化。",
        },
        {
            "title": "疫情后高精度模型",
            "family": "分时期预测",
            "metrics_key": "post_covid_high_accuracy_metrics",
            "shap_key": "post_covid_high_accuracy_shap",
            "role": "2023+ 单独训练的高精度预测模型。",
            "question": "疫情后阶段加入共污染物和污染持续性后，预测精度上限是多少。",
            "features": "气象、PBLH、PM2.5 时滞/滚动均值、共污染物和气溶胶变量。",
            "academic_use": "用于疫情后预测展示和与气象归因模型对照。",
            "result_reading": "疫情后高精度模型用于观察恢复期污染短时预测能力。它与疫情前、疫情期高精度模型一起构成分时期预测性能对照。",
            "limitation": "2023+ 数据源差异会影响跨阶段对比，模型表现应更多看作恢复期内部预测能力，而不是直接代表排放恢复强度。",
            "note": "2023+ PM2.5 数据源与 2018-2022 不完全一致，跨时期比较要标注这一点。",
        },
        {
            "title": "疫情后气象归因模型",
            "family": "气象归因",
            "metrics_key": "post_covid_meteorology_metrics",
            "shap_key": "post_covid_meteorology_shap",
            "role": "2023+ 单独训练，用于疫情后气象贡献解释。",
            "question": "刻画疫情后 PBLH、低边界层、湿度、风速和逆温对污染累积解释力的恢复特征。",
            "features": "只保留气象、PBLH、稳定度、风输送、时间和城市特征。",
            "academic_use": "论文中疫情后气象贡献分析的核心模型。",
            "result_reading": "旧版疫情后气象模型已经显示 PBLH、低边界层和湿度相关特征重新突出，支持边界层扩散约束在恢复期重新变强的判断。",
            "limitation": "最新 v2 模型加入异常目标和通风系数等过程型特征后，更适合作为疫情后归因结论的主证据。",
            "note": "该模型中 PBLH 和低边界层标记重新突出，是边界层约束污染累积的重要证据。",
        },
        {
            "title": "24 小时辅助模型",
            "family": "提前量预测",
            "metrics_key": "next24_metrics",
            "shap_key": "next24_shap",
            "role": "用于展示 24 小时后 PM2.5 辅助预测。",
            "question": "评估较长提前量条件下模型提供 PM2.5 趋势参考的可用性。",
            "features": "旧训练口径的核心特征集，未完全纳入本轮 2018+ 分时期训练框架。",
            "academic_use": "只作为网页辅助展示，不作为疫情分时期贡献分析的主要证据。",
            "result_reading": "24 小时辅助模型提供趋势参考，适合在预测台中帮助用户理解一天后的大致变化方向。",
            "limitation": "该模型未纳入本轮 2018+ 分时期训练和 v2 气象归因框架，不能作为论文核心模型。",
            "note": "当前课题核心结论应优先引用当前小时全时期模型和三套分时期气象贡献模型。",
        },
    ]


def training_strategy_rows(assets: Assets) -> pd.DataFrame:
    rows = []
    for spec in model_card_specs():
        metrics = assets[spec["metrics_key"]]
        rows.append(
            {
                "模型": spec["title"],
                "类型": spec["family"],
                "特征集": metrics.get("feature_set", "NA"),
                "训练样本": int(metrics.get("train_rows", 0)),
                "验证样本": int(metrics.get("valid_rows", 0)),
                "测试样本": int(metrics.get("test_rows", 0)),
                "调参轮数": metrics.get("trials", "NA"),
                "最佳迭代": metrics.get("best_iteration", "NA"),
                "测试 R2": round(float(metrics["test"]["r2"]), 3),
                "测试 RMSE": round(float(metrics["test"]["rmse"]), 3),
                "测试 MAE": round(float(metrics["test"]["mae"]), 3),
            }
        )
    return pd.DataFrame(rows)


def get_profile(profiles: pd.DataFrame, city: str, month: int, hour: int) -> dict[str, float]:
    match = profiles[(profiles["city"] == city) & (profiles["month"] == month) & (profiles["hour"] == hour)]
    if match.empty:
        match = profiles[profiles["city"] == city]
    if match.empty:
        match = profiles
    return match.median(numeric_only=True).to_dict()


def add_time_features(row: dict, selected_date: date, hour: int) -> None:
    day = pd.Timestamp(selected_date)
    timestamp = pd.Timestamp(selected_date) + pd.Timedelta(hours=hour)
    row["hour"] = hour
    row["month"] = day.month
    row["year"] = day.year
    row["dayofyear"] = day.dayofyear
    row["weekday"] = day.weekday()
    row["is_weekend"] = int(row["weekday"] in [5, 6])
    row["period"] = "post_covid_2023_plus"
    if timestamp < pd.Timestamp("2020-01-01 00:00:00"):
        row["period"] = "pre_covid_2018_2019"
    if pd.Timestamp("2020-01-01 00:00:00") <= timestamp <= pd.Timestamp("2022-12-31 23:00:00"):
        row["period"] = "covid_2020_2022"
    row["is_covid_period"] = int(row["period"] == "covid_2020_2022")
    row["hour_sin"] = math.sin(2 * math.pi * hour / 24)
    row["hour_cos"] = math.cos(2 * math.pi * hour / 24)
    row["dayofyear_sin"] = math.sin(2 * math.pi * row["dayofyear"] / 366)
    row["dayofyear_cos"] = math.cos(2 * math.pi * row["dayofyear"] / 366)
    row["month_sin"] = math.sin(2 * math.pi * row["month"] / 12)
    row["month_cos"] = math.cos(2 * math.pi * row["month"] / 12)


def period_for_datetime(selected_date: date, hour: int) -> str:
    timestamp = pd.Timestamp(selected_date) + pd.Timedelta(hours=hour)
    if timestamp < pd.Timestamp("2020-01-01 00:00:00"):
        return "pre_covid_2018_2019"
    if timestamp <= pd.Timestamp("2022-12-31 23:00:00"):
        return "covid_2020_2022"
    return "post_covid_2023_plus"


def resolve_prediction_model_key(choice: str, selected_date: date, hour: int) -> str:
    period = period_for_datetime(selected_date, hour)
    if choice == "按日期自动选择分时期高精度模型":
        return PERIOD_TO_HIGH_ACCURACY[period]
    if choice in {"按日期自动选择 v2 气象贡献模型（推荐归因）", "按日期自动选择气象归因模型"}:
        return PERIOD_TO_METEOROLOGY_V2[period]
    if choice == "按日期自动选择旧版气象归因模型（对照）":
        return PERIOD_TO_METEOROLOGY[period]
    return MODEL_LABEL_TO_KEY[choice]


@st.cache_resource(show_spinner=False)
def load_prediction_models() -> dict[str, dict]:
    return {
        key: load_model(str(resolve_path("models", spec["path"])))
        for key, spec in PREDICTION_MODEL_SPECS.items()
    }


def add_derived_features(row: dict, pblh_reference: float) -> None:
    direction_rad = math.radians(float(row["wind_direction_10m"]))
    speed = float(row["wind_speed_10m"])
    row["wind_u_10m"] = -speed * math.sin(direction_rad)
    row["wind_v_10m"] = -speed * math.cos(direction_rad)
    row["southerly_transport_10m"] = max(row["wind_v_10m"], 0.0)
    row["northerly_cleaning_10m"] = max(-row["wind_v_10m"], 0.0)
    row["t_inverse_850_1000"] = float(row["temperature_850hPa"]) - float(row["temperature_1000hPa"])
    row["has_inversion_850_1000"] = int(row["t_inverse_850_1000"] > 0)
    row["low_pblh_flag"] = int(float(row["boundary_layer_height"]) < pblh_reference)


def add_v2_meteorology_features(row: dict, pblh_reference: float) -> None:
    lag_hours = [1, 3, 6, 12, 24, 48, 72]
    rolling_windows = [3, 6, 12, 24, 48, 72]
    row["era5_boundary_layer_height"] = float(row.get("boundary_layer_height", 0.0))
    wind_speed = float(row.get("wind_speed_10m", 0.0))
    pblh = float(row.get("boundary_layer_height", 0.0))
    humidity = float(row.get("relative_humidity_2m", 0.0))
    precipitation = float(row.get("precipitation", 0.0))
    row["rain"] = float(row.get("rain", precipitation))
    row["ventilation_coefficient"] = pblh * wind_speed
    row["inverse_ventilation"] = 1.0 / max(float(row["ventilation_coefficient"]), 1.0)
    row["transport_balance_10m"] = float(row["southerly_transport_10m"]) - float(row["northerly_cleaning_10m"])
    row["southerly_transport_intensity"] = float(row["southerly_transport_10m"]) * wind_speed
    row["northerly_cleaning_intensity"] = float(row["northerly_cleaning_10m"]) * wind_speed
    row["humidity_pblh_interaction"] = humidity / 100.0 * float(row["inverse_ventilation"])
    row["cloud_humidity_interaction"] = float(row.get("cloud_cover", 0.0)) * humidity / 100.0
    row["precip_present"] = int(precipitation > 0.1)

    row["pblh_q25_city_period"] = float(row.get("pblh_q25_city_period", pblh_reference))
    row["wind_q25_city_period"] = float(row.get("wind_q25_city_period", max(wind_speed * 0.65, 0.8)))
    row["rh_q75_city_period"] = float(row.get("rh_q75_city_period", min(max(humidity * 1.15, 70.0), 95.0)))
    row["ventilation_q25_city_period"] = float(
        row.get("ventilation_q25_city_period", row["pblh_q25_city_period"] * row["wind_q25_city_period"])
    )
    row["low_pblh_v2"] = int(pblh <= row["pblh_q25_city_period"])
    row["weak_wind_v2"] = int(wind_speed <= row["wind_q25_city_period"])
    row["high_humidity_v2"] = int(humidity >= row["rh_q75_city_period"])
    row["low_ventilation_v2"] = int(row["ventilation_coefficient"] <= row["ventilation_q25_city_period"])
    row["no_precip_v2"] = int(precipitation <= 0.1)
    row["stagnant_weather_flag"] = int(
        row["low_pblh_v2"]
        and row["weak_wind_v2"]
        and row["high_humidity_v2"]
        and row["no_precip_v2"]
    )
    row["stagnation_index"] = int(
        row["low_pblh_v2"]
        + row["weak_wind_v2"]
        + row["high_humidity_v2"]
        + row["low_ventilation_v2"]
        + row["no_precip_v2"]
    )
    for flag in ["low_pblh_v2", "weak_wind_v2", "high_humidity_v2", "low_ventilation_v2", "stagnant_weather_flag"]:
        row[f"{flag}_streak_h"] = int(row[flag])

    lag_sources = [
        "boundary_layer_height",
        "wind_speed_10m",
        "wind_u_10m",
        "wind_v_10m",
        "southerly_transport_10m",
        "northerly_cleaning_10m",
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "pressure_msl",
        "surface_pressure",
        "cloud_cover",
        "precipitation",
        "ventilation_coefficient",
    ]
    for source in lag_sources:
        current = float(row.get(source, 0.0))
        for lag in lag_hours:
            row.setdefault(f"{source}_lag_{lag}h", current)

    rolling_mean_sources = [
        "boundary_layer_height",
        "wind_speed_10m",
        "wind_u_10m",
        "wind_v_10m",
        "southerly_transport_10m",
        "northerly_cleaning_10m",
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "pressure_msl",
        "cloud_cover",
        "ventilation_coefficient",
    ]
    for source in rolling_mean_sources:
        current = float(row.get(source, 0.0))
        for window in rolling_windows:
            row.setdefault(f"{source}_roll_mean_{window}h", current)

    for source in ["precipitation", "rain"]:
        current = max(float(row.get(source, 0.0)), 0.0)
        for window in rolling_windows:
            row.setdefault(f"{source}_roll_sum_{window}h", current * window)

    for source in ["pressure_msl", "temperature_2m", "boundary_layer_height", "ventilation_coefficient"]:
        current = float(row.get(source, 0.0))
        row[f"{source}_change_3h"] = current - float(row.get(f"{source}_lag_3h", current))
        row[f"{source}_change_24h"] = current - float(row.get(f"{source}_lag_24h", current))

    if precipitation > 0.1:
        weather_type = 4
    elif row["stagnant_weather_flag"]:
        weather_type = 0
    elif row["northerly_cleaning_10m"] > 1:
        weather_type = 1
    elif row["southerly_transport_10m"] > 1:
        weather_type = 2
    elif row["ventilation_coefficient"] > 1200:
        weather_type = 3
    else:
        weather_type = 5
    row["weather_type_k6"] = str(int(row.get("weather_type_k6", weather_type)))
    row["baseline_pm2_5"] = float(row.get("pm2_5", row.get("baseline_pm2_5", 35.0)))


def complete_row(features: list[str], row: dict) -> pd.DataFrame:
    completed = {feature: row.get(feature, 0) for feature in features}
    return pd.DataFrame([completed])


def predict(bundle: dict, row: dict) -> float:
    features = bundle["features"]
    frame = complete_row(features, row)
    matrix = bundle["preprocessor"].transform(frame[features])
    prediction = float(bundle["model"].predict(matrix)[0])
    target_kind = bundle.get("target_kind", bundle.get("target_meta", {}).get("target_kind", "raw"))
    if target_kind == "log1p":
        return max(math.expm1(prediction), 0.0)
    if target_kind == "anomaly":
        target_meta = bundle.get("target_meta", {})
        baseline = float(
            row.get(
                "baseline_pm2_5",
                target_meta.get("climatology_global_train_valid", target_meta.get("climatology_global_train", 0.0)),
            )
        )
        return max(baseline + prediction, 0.0)
    return max(prediction, 0.0)


def scenario_row(
    city_info: pd.DataFrame,
    profiles: pd.DataFrame,
    city: str,
    selected_date: date,
    hour: int,
    overrides: dict,
) -> dict:
    city_row = city_info[city_info["city"] == city].iloc[0].to_dict()
    row = get_profile(profiles, city, selected_date.month, hour)
    row.update(city_row)
    row.update(overrides)
    add_time_features(row, selected_date, hour)
    pblh_reference = get_profile(profiles, city, selected_date.month, hour).get("boundary_layer_height", 600) * 0.75
    add_derived_features(row, pblh_reference)
    add_v2_meteorology_features(row, pblh_reference)

    if "pm2_5_lag_1h" in overrides or "pm2_5_lag_3h" in overrides:
        lag1 = float(row.get("pm2_5_lag_1h", row.get("pm2_5", 35)))
        lag3 = float(row.get("pm2_5_lag_3h", lag1))
        lag24 = float(row.get("pm2_5_lag_24h", lag3))
        row["pm2_5_roll_mean_3h"] = (lag1 + lag3 + float(row.get("pm2_5_roll_mean_3h", lag3))) / 3
        row["pm2_5_roll_mean_24h"] = (lag1 + lag3 + lag24 + float(row.get("pm2_5_roll_mean_24h", lag24))) / 4

    return row


def shap_chart(shap_df: pd.DataFrame, title: str) -> go.Figure:
    top = shap_df.head(15).copy()
    top["特征"] = top["feature"].map(feature_label)
    top = top.sort_values("mean_abs_shap")
    fig = px.bar(top, x="mean_abs_shap", y="特征", orientation="h", title=title)
    fig.update_traces(marker_color="#2563eb")
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=50, b=20), xaxis_title="平均绝对 SHAP", yaxis_title="")
    return fig


def add_pm25_bands(fig: go.Figure, y_max: float) -> None:
    bands = [
        (0, 35, "优", "rgba(22, 163, 74, 0.10)"),
        (35, 75, "良好", "rgba(34, 197, 94, 0.10)"),
        (75, 115, "轻度", "rgba(245, 158, 11, 0.13)"),
        (115, 150, "中度", "rgba(220, 38, 38, 0.11)"),
        (150, max(250, y_max), "重度+", "rgba(153, 27, 27, 0.10)"),
    ]
    for y0, y1, label, color in bands:
        fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0, annotation_text=label, annotation_position="left")


def risk_items(row: dict, current_prediction: float) -> list[dict[str, str]]:
    items = []
    if row["low_pblh_flag"]:
        items.append({"title": "垂直扩散", "value": "偏弱", "detail": "PBLH 低，污染物更容易堆积", "color": "#c62828"})
    else:
        items.append({"title": "垂直扩散", "value": "较好", "detail": "PBLH 不低，垂直混合空间较充足", "color": "#2e7d32"})
    if row["t_inverse_850_1000"] > 0:
        items.append({"title": "热力稳定度", "value": "逆温", "detail": "上暖下冷，扩散受抑制", "color": "#ef6c00"})
    else:
        items.append({"title": "热力稳定度", "value": "无逆温", "detail": "热力层结相对有利扩散", "color": "#2e7d32"})
    if row["southerly_transport_10m"] > 1:
        items.append({"title": "区域输送", "value": "南风输送", "detail": "京津冀南向输送贡献可能增强", "color": "#ef6c00"})
    elif row["northerly_cleaning_10m"] > 1:
        items.append({"title": "区域输送", "value": "北风清除", "detail": "北风条件通常更利于清洁空气输入", "color": "#2e7d32"})
    else:
        items.append({"title": "区域输送", "value": "弱风", "detail": "水平输送弱，局地累积更重要", "color": "#b7791f"})
    if current_prediction > 75:
        items.append({"title": "污染水平", "value": "需关注", "detail": "预测值超过良级上限", "color": "#c62828"})
    else:
        items.append({"title": "污染水平", "value": "可接受", "detail": "预测值处于优良范围", "color": "#2e7d32"})
    return items


def risk_cards_html(items: list[dict[str, str]]) -> str:
    cards = [
        (
            f'<div class="factor-card" style="border-top-color:{item["color"]}">'
            f'<div class="factor-title">{item["title"]}</div>'
            f'<div class="factor-value" style="color:{item["color"]}">{item["value"]}</div>'
            f'<div class="factor-detail">{item["detail"]}</div>'
            "</div>"
        )
        for item in items
    ]
    return f'<div class="factor-grid">{"".join(cards)}</div>'


def performance_chart(performance: pd.DataFrame) -> go.Figure:
    long = performance.melt(id_vars=["模型", "定位"], value_vars=["MAE", "RMSE"], var_name="指标", value_name="数值")
    fig = px.bar(long, x="模型", y="数值", color="指标", barmode="group", text_auto=".1f", title="模型误差对比")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=20), yaxis_title="ug/m3", xaxis_title="")
    return fig


def r2_chart(performance: pd.DataFrame) -> go.Figure:
    fig = px.bar(performance, x="模型", y="R2", color="定位", text="R2", title="模型解释度 R2")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=20), yaxis_range=[0, 1.05], xaxis_title="")
    return fig


def seasonal_chart(seasonal: pd.DataFrame, city: str, month: int) -> go.Figure:
    data = seasonal[seasonal["city"] == city].sort_values("month")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data["month"],
            y=data["pm2_5_p75"],
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data["month"],
            y=data["pm2_5_p25"],
            fill="tonexty",
            line=dict(width=0),
            name="P25-P75",
            fillcolor="rgba(37, 99, 235, 0.16)",
        )
    )
    fig.add_trace(go.Scatter(x=data["month"], y=data["pm2_5_median"], mode="lines+markers", name="PM2.5 中位数"))
    fig.add_vline(x=month, line_width=2, line_dash="dash", line_color="#c62828")
    fig.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=20), title=f"{city} 月尺度 PM2.5 季节参考", xaxis_title="月份", yaxis_title="ug/m3")
    return fig


def wind_polar(speed: float, direction: float) -> go.Figure:
    fig = go.Figure(
        go.Barpolar(
            r=[speed],
            theta=[direction],
            width=[24],
            marker_color=["#1976d2"],
            marker_line_color="#0d47a1",
            marker_line_width=1,
            opacity=0.82,
        )
    )
    fig.update_layout(
        title="输入风向风速",
        height=330,
        margin=dict(l=10, r=10, t=50, b=10),
        polar=dict(radialaxis=dict(title="m/s"), angularaxis=dict(direction="clockwise", rotation=90)),
        showlegend=False,
    )
    return fig


def build_daily_prediction(
    current_bundle: dict,
    city_info: pd.DataFrame,
    profiles: pd.DataFrame,
    city: str,
    selected_date: date,
    overrides: dict,
) -> pd.DataFrame:
    rows = []
    for hour in range(24):
        row = scenario_row(city_info, profiles, city, selected_date, hour, overrides)
        rows.append(
            {
                "hour": hour,
                "predicted_pm2_5": predict(current_bundle, row),
                "temperature_2m": row["temperature_2m"],
                "wind_speed_10m": row["wind_speed_10m"],
                "boundary_layer_height": row["boundary_layer_height"],
            }
        )
    return pd.DataFrame(rows)


def render_weather_context(
    seasonal: pd.DataFrame,
    daily: pd.DataFrame,
    current_bundle: dict,
    city_info: pd.DataFrame,
    profiles: pd.DataFrame,
    city: str,
    selected_date: date,
    overrides: dict,
    wind_speed: float,
    wind_direction: float,
) -> None:
    st.markdown("### 气象背景与扩散条件")
    st.caption("这些图直接服务于当前预测：左侧看季节背景，右侧看历史 PM2.5 与 PBLH，下面展示输入风场和当天气象剖面。")
    seasonal_col, hist_col = st.columns([0.42, 0.58])
    with seasonal_col:
        st.plotly_chart(seasonal_chart(seasonal, city, selected_date.month), use_container_width=True)

    hist = daily[daily["city"] == city].sort_values("date").tail(240)
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Scatter(x=hist["date"], y=hist["pm2_5"], name="PM2.5", yaxis="y1"))
    fig_hist.add_trace(go.Scatter(x=hist["date"], y=hist["boundary_layer_height"], name="PBLH", yaxis="y2"))
    fig_hist.update_layout(
        title=f"{city} 历史 PM2.5 与边界层高度",
        height=330,
        margin=dict(l=10, r=10, t=50, b=20),
        yaxis=dict(title="PM2.5 ug/m3"),
        yaxis2=dict(title="PBLH m", overlaying="y", side="right"),
    )
    with hist_col:
        st.plotly_chart(fig_hist, use_container_width=True)

    w1, w2 = st.columns([0.38, 0.62])
    with w1:
        st.plotly_chart(wind_polar(wind_speed, wind_direction), use_container_width=True)
    weather_df = build_daily_prediction(current_bundle, city_info, profiles, city, selected_date, overrides)
    fig_weather = px.line(
        weather_df,
        x="hour",
        y=["temperature_2m", "wind_speed_10m", "boundary_layer_height"],
        title="当天气象变量剖面",
    )
    fig_weather.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=20), xaxis_title="小时", yaxis_title="")
    fig_weather.update_traces(line=dict(width=3))
    with w2:
        st.plotly_chart(fig_weather, use_container_width=True)


def input_defaults(profile: dict) -> dict[str, float]:
    wind_speed = float(profile.get("wind_speed_10m", 2.0))
    temperature = float(profile.get("temperature_2m", 20.0))
    return {
        "temperature_2m": temperature,
        "relative_humidity_2m": float(profile.get("relative_humidity_2m", 55.0)),
        "pressure_msl": float(profile.get("pressure_msl", 1010.0)),
        "surface_pressure": float(profile.get("surface_pressure", 990.0)),
        "cloud_cover": float(profile.get("cloud_cover", 30.0)),
        "precipitation": float(profile.get("precipitation", 0.0)),
        "wind_speed_10m": wind_speed,
        "wind_direction_10m": float(profile.get("wind_direction_10m", 180.0)),
        "wind_gusts_10m": float(profile.get("wind_gusts_10m", wind_speed * 1.8)),
        "boundary_layer_height": float(profile.get("boundary_layer_height", 650.0)),
        "temperature_850hPa": float(profile.get("temperature_850hPa", temperature - 6)),
        "temperature_1000hPa": float(profile.get("temperature_1000hPa", temperature)),
        "pm2_5_lag_1h": float(profile.get("pm2_5_lag_1h", 35.0)),
        "pm2_5_lag_3h": float(profile.get("pm2_5_lag_3h", 35.0)),
        "pm2_5_lag_24h": float(profile.get("pm2_5_lag_24h", 35.0)),
    }


def write_input_state(defaults: dict[str, float]) -> None:
    for key, value in defaults.items():
        st.session_state[f"input_{key}"] = value


def make_overrides(values: dict[str, float]) -> dict[str, float]:
    temperature = float(values["temperature_2m"])
    humidity = float(values["relative_humidity_2m"])
    precipitation = float(values["precipitation"])
    wind_speed = float(values["wind_speed_10m"])
    dew_point = temperature - (100 - humidity) / 5
    return {
        "temperature_2m": temperature,
        "relative_humidity_2m": humidity,
        "dew_point_2m": dew_point,
        "apparent_temperature": temperature,
        "precipitation": precipitation,
        "rain": precipitation,
        "snowfall": 0.0 if temperature > 1 else precipitation,
        "pressure_msl": float(values["pressure_msl"]),
        "surface_pressure": float(values["surface_pressure"]),
        "cloud_cover": float(values["cloud_cover"]),
        "wind_speed_10m": wind_speed,
        "wind_direction_10m": float(values["wind_direction_10m"]),
        "wind_gusts_10m": float(values["wind_gusts_10m"]),
        "boundary_layer_height": float(values["boundary_layer_height"]),
        "temperature_850hPa": float(values["temperature_850hPa"]),
        "temperature_1000hPa": float(values["temperature_1000hPa"]),
        "pm2_5_lag_1h": float(values["pm2_5_lag_1h"]),
        "pm2_5_lag_3h": float(values["pm2_5_lag_3h"]),
        "pm2_5_lag_24h": float(values["pm2_5_lag_24h"]),
        "wind_speed_10m_lag_1h": wind_speed,
        "wind_speed_10m_lag_3h": wind_speed,
        "wind_speed_10m_lag_24h": wind_speed,
    }


def style_page() -> None:
    st.set_page_config(page_title="京津冀 PM2.5 预测模型", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {background:#f5f7fb;}
        .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
        div[data-testid="stMetricValue"] {font-size: 1.85rem;}
        div[data-testid="stMetric"] {
            background:#ffffff;
            border:1px solid #d8dee6;
            border-radius:8px;
            padding:14px 14px 10px 14px;
            box-shadow:0 8px 24px rgba(24,39,75,0.06);
        }
        .stTabs [data-baseweb="tab-list"] {gap: 8px;}
        .stTabs [data-baseweb="tab"] {
            background:#ffffff;
            border:1px solid #d8dee6;
            border-radius:8px;
            padding:8px 14px;
        }
        .prediction-card {
            border: 1px solid #d8dee6;
            border-radius: 8px;
            padding: 18px 18px 12px 18px;
            background: #ffffff;
            box-shadow:0 8px 24px rgba(24,39,75,0.06);
        }
        .forecast-overview {
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:12px;
            margin:6px 0 14px 0;
        }
        .forecast-metric {
            position:relative;
            overflow:hidden;
            background:#ffffff;
            border:1px solid #d8dee6;
            border-radius:10px;
            padding:15px 16px 14px 16px;
            min-height:132px;
            box-shadow:0 10px 28px rgba(24,39,75,0.055);
        }
        .forecast-metric::before {
            content:"";
            position:absolute;
            left:0;
            top:0;
            width:100%;
            height:4px;
            background:var(--metric-color);
        }
        .forecast-metric-label {
            color:#64748b;
            font-size:0.84rem;
            font-weight:800;
            margin-bottom:8px;
        }
        .forecast-metric-value {
            color:#132033;
            font-size:1.72rem;
            font-weight:900;
            line-height:1.15;
            margin-bottom:11px;
        }
        .forecast-status {
            display:inline-flex;
            align-items:center;
            gap:6px;
            border-radius:999px;
            padding:5px 9px;
            background:color-mix(in srgb, var(--metric-color) 12%, white);
            color:var(--metric-color);
            font-size:0.86rem;
            font-weight:850;
        }
        .forecast-status span {
            width:7px;
            height:7px;
            border-radius:999px;
            background:var(--metric-color);
        }
        .forecast-metric-detail {
            margin-top:8px;
            color:#94a3b8;
            font-size:0.78rem;
            font-weight:700;
        }
        .forecast-panel {
            display:grid;
            grid-template-columns:minmax(0,1.45fr) minmax(280px,0.75fr);
            gap:18px;
            align-items:stretch;
            border:1px solid #d8dee6;
            border-radius:12px;
            background:#ffffff;
            padding:18px;
            box-shadow:0 14px 34px rgba(24,39,75,0.065);
            margin:6px 0 14px 0;
        }
        .forecast-panel-main {
            border-radius:10px;
            background:#f8fafc;
            padding:18px 20px;
            border:1px solid #e5eaf0;
            display:flex;
            flex-direction:column;
        }
        .forecast-place {
            color:#475569;
            font-size:0.96rem;
            font-weight:800;
            margin-bottom:8px;
        }
        .forecast-number {
            font-size:3.2rem;
            font-weight:950;
            line-height:1.05;
            margin-bottom:10px;
        }
        .forecast-number span {
            color:#64748b;
            font-size:1.15rem;
            font-weight:800;
            margin-left:8px;
        }
        .forecast-model-line {
            color:#475569;
            font-size:0.95rem;
            line-height:1.55;
        }
        .forecast-model-line b {
            color:#132033;
        }
        .forecast-model-line span {
            display:inline-flex;
            margin-left:8px;
            border-radius:999px;
            padding:3px 8px;
            color:#1d4ed8;
            background:#dbeafe;
            font-size:0.78rem;
            font-weight:800;
        }
        .forecast-main-note {
            color:#64748b;
            font-size:0.9rem;
            line-height:1.55;
            margin:10px 0 14px 0;
            max-width:820px;
        }
        .forecast-chip-row {
            display:flex;
            flex-wrap:wrap;
            gap:8px;
            margin-top:auto;
            padding-top:4px;
        }
        .forecast-chip-row span {
            display:inline-flex;
            border:1px solid #d8dee6;
            border-radius:999px;
            padding:6px 9px;
            background:#ffffff;
            color:#334155;
            font-size:0.8rem;
            font-weight:800;
        }
        .forecast-panel-side {
            display:grid;
            gap:10px;
        }
        .forecast-side-item {
            border:1px solid #e5eaf0;
            border-radius:10px;
            padding:12px 13px;
            background:#ffffff;
        }
        .forecast-side-item div {
            color:#64748b;
            font-size:0.8rem;
            font-weight:800;
            margin-bottom:4px;
        }
        .forecast-side-item strong {
            color:#132033;
            font-size:1.02rem;
            line-height:1.35;
        }
        .small-note {color: #5f6b7a; font-size: 0.9rem;}
        .scenario-strip {
            display:grid;
            grid-template-columns:repeat(6,minmax(0,1fr));
            gap:8px;
            margin:12px 0 16px 0;
        }
        .scenario-item {
            background:#f8fafc;
            border:1px solid #e5eaf0;
            border-radius:10px;
            padding:11px 12px;
        }
        .scenario-item.wide {
            background:#eef6ff;
            border-color:#bfdbfe;
        }
        .scenario-label {
            color:#64748b;
            font-size:0.82rem;
            margin-bottom:4px;
        }
        .scenario-value {
            color:#132033;
            font-weight:700;
            font-size:1rem;
        }
        .factor-grid {
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:10px;
            margin:14px 0;
        }
        .factor-card {
            background:#ffffff;
            border:1px solid #d8dee6;
            border-top:4px solid #2563eb;
            border-radius:8px;
            padding:12px;
            min-height:112px;
            box-shadow:0 8px 24px rgba(24,39,75,0.05);
        }
        .factor-title {color:#64748b;font-size:0.86rem;margin-bottom:6px;}
        .factor-value {font-size:1.15rem;font-weight:800;margin-bottom:6px;}
        .factor-detail {color:#475569;font-size:0.88rem;line-height:1.45;}
        .model-card {
            background:#ffffff;
            border:1px solid #d8dee6;
            border-radius:8px;
            padding:14px;
            min-height:130px;
        }
        .model-card b {display:block;margin-bottom:6px;}
        .model-card p {color:#475569;margin:0;line-height:1.55;}
        .model-intro {
            margin:4px 0 20px 0;
        }
        .model-hero-grid {
            display:grid;
            grid-template-columns:1.15fr 0.85fr;
            gap:16px;
            margin:10px 0 18px 0;
        }
        .model-hero-card {
            position:relative;
            overflow:hidden;
            border:1px solid #d8dee6;
            border-radius:12px;
            padding:20px 22px 18px 22px;
            min-height:190px;
            box-shadow:0 14px 34px rgba(24,39,75,0.07);
        }
        .model-hero-card.primary {
            background:linear-gradient(135deg,#ffffff 0%,#eef6ff 100%);
            border-color:#bfdbfe;
        }
        .model-hero-card.muted {
            background:linear-gradient(135deg,#ffffff 0%,#f8fafc 100%);
        }
        .model-hero-card::after {
            content:"";
            position:absolute;
            right:-42px;
            top:-48px;
            width:150px;
            height:150px;
            border-radius:999px;
            background:rgba(37,99,235,0.08);
        }
        .model-hero-card.muted::after {background:rgba(100,116,139,0.08);}
        .model-kicker {
            display:inline-flex;
            align-items:center;
            border-radius:999px;
            padding:4px 10px;
            background:#dbeafe;
            color:#1d4ed8;
            font-size:0.78rem;
            font-weight:800;
            margin-bottom:10px;
        }
        .model-hero-card.muted .model-kicker {
            background:#e2e8f0;
            color:#475569;
        }
        .model-hero-card h3 {
            margin:0 0 10px 0;
            color:#132033;
            font-size:1.25rem;
            line-height:1.35;
        }
        .model-hero-card p {
            margin:0;
            color:#475569;
            line-height:1.72;
            font-size:0.96rem;
            max-width:760px;
        }
        .score-row {
            display:flex;
            flex-wrap:wrap;
            gap:8px;
            margin-top:16px;
        }
        .score-pill {
            display:inline-flex;
            align-items:baseline;
            gap:6px;
            border:1px solid #d8dee6;
            border-radius:999px;
            padding:7px 11px;
            background:rgba(255,255,255,0.78);
            color:#132033;
            font-weight:800;
            box-shadow:0 6px 18px rgba(24,39,75,0.05);
        }
        .score-pill b {
            color:#64748b;
            font-size:0.72rem;
            letter-spacing:0;
        }
        .model-route-grid {
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:12px;
            margin:12px 0 4px 0;
        }
        .model-route-card {
            position:relative;
            background:#ffffff;
            border:1px solid #d8dee6;
            border-radius:10px;
            padding:15px 15px 14px 15px;
            min-height:210px;
            box-shadow:0 10px 26px rgba(24,39,75,0.055);
        }
        .model-route-card::before {
            content:"";
            position:absolute;
            left:0;
            top:0;
            width:100%;
            height:4px;
            background:var(--route-color);
        }
        .model-route-card.accent-blue {--route-color:#2563eb;--route-soft:#eff6ff;--route-text:#1d4ed8;}
        .model-route-card.accent-green {--route-color:#059669;--route-soft:#ecfdf5;--route-text:#047857;}
        .model-route-card.accent-amber {--route-color:#d97706;--route-soft:#fffbeb;--route-text:#b45309;}
        .model-route-card.accent-purple {--route-color:#7c3aed;--route-soft:#f5f3ff;--route-text:#6d28d9;}
        .route-badge {
            display:inline-flex;
            border-radius:999px;
            padding:4px 9px;
            background:var(--route-soft);
            color:var(--route-text);
            font-size:0.76rem;
            font-weight:800;
            margin-bottom:10px;
        }
        .model-route-card h4 {
            margin:0 0 9px 0;
            color:#132033;
            font-size:1.02rem;
            line-height:1.38;
        }
        .model-route-card p {
            margin:0;
            color:#475569;
            line-height:1.62;
            font-size:0.92rem;
        }
        .route-meta {
            margin-top:12px;
            padding-top:10px;
            border-top:1px solid #e5eaf0;
            color:#64748b;
            font-size:0.82rem;
            line-height:1.45;
            font-weight:700;
        }
        .intro-page {
            margin:6px 0 20px 0;
        }
        .intro-hero {
            display:grid;
            grid-template-columns:minmax(0,1.25fr) minmax(310px,0.75fr);
            gap:16px;
            align-items:stretch;
            margin:10px 0 16px 0;
        }
        .intro-hero-main,
        .intro-score-panel {
            border:1px solid #d8dee6;
            border-radius:8px;
            background:#ffffff;
            box-shadow:0 12px 30px rgba(24,39,75,0.06);
        }
        .intro-hero-main {
            padding:22px 24px 20px 24px;
            border-top:4px solid var(--intro-accent);
        }
        .intro-score-panel {
            padding:20px;
            background:#f8fafc;
            border-top:4px solid var(--intro-accent);
        }
        .intro-hero.prediction {--intro-accent:#2563eb;--intro-soft:#eff6ff;--intro-text:#1d4ed8;}
        .intro-hero.attribution {--intro-accent:#059669;--intro-soft:#ecfdf5;--intro-text:#047857;}
        .intro-kicker,
        .intro-score-label,
        .model-card-tag,
        .method-band-kicker {
            display:inline-flex;
            border-radius:999px;
            padding:5px 10px;
            font-size:0.76rem;
            font-weight:900;
            letter-spacing:0;
        }
        .intro-kicker,
        .intro-score-label {
            background:var(--intro-soft);
            color:var(--intro-text);
            margin-bottom:10px;
        }
        .intro-hero h3 {
            margin:0 0 10px 0;
            color:#132033;
            font-size:1.36rem;
            line-height:1.35;
        }
        .intro-hero h4 {
            margin:0 0 10px 0;
            color:#132033;
            font-size:1.06rem;
            line-height:1.35;
        }
        .intro-hero p {
            margin:0;
            color:#475569;
            line-height:1.76;
            font-size:0.96rem;
        }
        .intro-chip-row {
            display:flex;
            flex-wrap:wrap;
            gap:8px;
            margin-top:16px;
        }
        .intro-chip-row span {
            border:1px solid #d8dee6;
            border-radius:999px;
            background:#ffffff;
            color:#334155;
            padding:6px 10px;
            font-size:0.8rem;
            font-weight:850;
        }
        .section-bridge {
            border-left:4px solid var(--bridge-accent);
            padding:0 0 0 14px;
            margin:26px 0 12px 0;
        }
        .section-bridge.blue {--bridge-accent:#2563eb;}
        .section-bridge.green {--bridge-accent:#059669;}
        .section-bridge.slate {--bridge-accent:#64748b;}
        .section-bridge-title {
            color:#132033;
            font-size:1.22rem;
            line-height:1.35;
            font-weight:900;
        }
        .intro-model-grid {
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:12px;
            margin:12px 0 16px 0;
        }
        .intro-model-card {
            position:relative;
            border:1px solid #d8dee6;
            border-radius:8px;
            background:#ffffff;
            padding:16px;
            min-height:238px;
            box-shadow:0 9px 24px rgba(24,39,75,0.05);
            border-top:4px solid var(--card-accent);
        }
        .intro-model-card.blue {--card-accent:#2563eb;--card-soft:#eff6ff;--card-text:#1d4ed8;}
        .intro-model-card.green {--card-accent:#059669;--card-soft:#ecfdf5;--card-text:#047857;}
        .intro-model-card.amber {--card-accent:#d97706;--card-soft:#fffbeb;--card-text:#b45309;}
        .intro-model-card.teal {--card-accent:#0f766e;--card-soft:#ecfdf5;--card-text:#0f766e;}
        .model-card-tag {
            background:var(--card-soft);
            color:var(--card-text);
            margin-bottom:10px;
        }
        .intro-model-card h4 {
            margin:0 0 9px 0;
            color:#132033;
            font-size:1rem;
            line-height:1.35;
        }
        .intro-model-card p {
            margin:0;
            color:#475569;
            line-height:1.62;
            font-size:0.9rem;
        }
        .score-row.compact {
            gap:6px;
            margin-top:12px;
        }
        .score-row.compact .score-pill {
            padding:5px 8px;
            font-size:0.82rem;
            box-shadow:none;
        }
        .method-band {
            display:grid;
            grid-template-columns:repeat(2,minmax(0,1fr));
            gap:12px;
            margin:12px 0 18px 0;
        }
        .method-band.three {
            grid-template-columns:repeat(3,minmax(0,1fr));
        }
        .method-band > div {
            border:1px solid #d8dee6;
            border-radius:8px;
            background:#ffffff;
            padding:17px;
            box-shadow:0 8px 22px rgba(24,39,75,0.045);
        }
        .method-band-kicker {
            background:#f1f5f9;
            color:#475569;
            margin-bottom:10px;
        }
        .method-band h4 {
            margin:0 0 9px 0;
            color:#132033;
            font-size:1rem;
            line-height:1.38;
        }
        .method-band p {
            margin:0;
            color:#475569;
            line-height:1.7;
            font-size:0.91rem;
        }
        .explain-band {
            border:1px solid #d8dee6;
            border-left:4px solid var(--explain-accent);
            border-radius:8px;
            background:#ffffff;
            padding:17px 18px;
            margin:12px 0 16px 0;
            box-shadow:0 8px 22px rgba(24,39,75,0.045);
        }
        .explain-band.blue {--explain-accent:#2563eb;}
        .explain-band.green {--explain-accent:#059669;}
        .explain-band.amber {--explain-accent:#d97706;}
        .explain-band h4 {
            margin:0 0 8px 0;
            color:#132033;
            font-size:1.04rem;
        }
        .explain-band p {
            margin:0 0 8px 0;
            color:#475569;
            line-height:1.7;
            font-size:0.92rem;
        }
        .explain-band p:last-child {margin-bottom:0;}
        .v2-card-note {
            border:1px solid #d8dee6;
            border-radius:8px;
            background:#f8fafc;
            padding:13px 14px;
            color:#475569;
            line-height:1.65;
            font-size:0.91rem;
            margin:8px 0 12px 0;
        }
        .training-intro {
            display:grid;
            grid-template-columns:minmax(0,1.35fr) minmax(300px,0.65fr);
            gap:16px;
            margin:8px 0 18px 0;
        }
        .training-hero,
        .training-score-card {
            position:relative;
            overflow:hidden;
            border:1px solid #d8dee6;
            border-radius:12px;
            background:#ffffff;
            box-shadow:0 14px 34px rgba(24,39,75,0.065);
        }
        .training-hero {
            padding:22px 24px 20px 24px;
            background:linear-gradient(135deg,#ffffff 0%,#eef6ff 100%);
            border-color:#bfdbfe;
        }
        .training-hero::after {
            content:"";
            position:absolute;
            right:-56px;
            top:-62px;
            width:190px;
            height:190px;
            border-radius:999px;
            background:rgba(37,99,235,0.08);
        }
        .training-kicker {
            display:inline-flex;
            border-radius:999px;
            padding:5px 10px;
            background:#dbeafe;
            color:#1d4ed8;
            font-size:0.78rem;
            font-weight:900;
            margin-bottom:10px;
        }
        .training-hero h3 {
            margin:0 0 10px 0;
            color:#132033;
            font-size:1.28rem;
            line-height:1.35;
        }
        .training-hero p,
        .training-score-card p {
            margin:0;
            color:#475569;
            line-height:1.72;
            font-size:0.96rem;
        }
        .training-chip-row {
            display:flex;
            flex-wrap:wrap;
            gap:8px;
            margin-top:16px;
        }
        .training-chip-row span {
            display:inline-flex;
            border:1px solid #c7d2fe;
            border-radius:999px;
            padding:6px 10px;
            background:#ffffff;
            color:#334155;
            font-size:0.8rem;
            font-weight:850;
        }
        .training-score-card {
            padding:20px;
            background:linear-gradient(135deg,#ffffff 0%,#f8fafc 100%);
        }
        .training-score-label {
            color:#64748b;
            font-size:0.82rem;
            font-weight:900;
            margin-bottom:10px;
        }
        .training-score-card h4 {
            margin:0 0 12px 0;
            color:#132033;
            font-size:1.05rem;
        }
        .training-family-grid {
            display:grid;
            grid-template-columns:repeat(3,minmax(0,1fr));
            gap:12px;
            margin:12px 0 18px 0;
        }
        .training-family-card {
            position:relative;
            border:1px solid #d8dee6;
            border-radius:10px;
            background:#ffffff;
            padding:16px 16px 14px 16px;
            min-height:188px;
            box-shadow:0 10px 26px rgba(24,39,75,0.055);
        }
        .training-family-card::before {
            content:"";
            position:absolute;
            left:0;
            top:0;
            width:100%;
            height:4px;
            background:var(--family-color);
        }
        .training-family-card.blue {--family-color:#2563eb;--family-soft:#eff6ff;--family-text:#1d4ed8;}
        .training-family-card.green {--family-color:#059669;--family-soft:#ecfdf5;--family-text:#047857;}
        .training-family-card.amber {--family-color:#d97706;--family-soft:#fffbeb;--family-text:#b45309;}
        .family-index {
            display:inline-flex;
            border-radius:999px;
            padding:4px 9px;
            background:var(--family-soft);
            color:var(--family-text);
            font-size:0.76rem;
            font-weight:900;
            margin-bottom:10px;
        }
        .training-family-card h4 {
            margin:0 0 9px 0;
            color:#132033;
            font-size:1.02rem;
        }
        .training-family-card p {
            margin:0;
            color:#475569;
            line-height:1.62;
            font-size:0.92rem;
        }
        .family-foot {
            margin-top:12px;
            padding-top:10px;
            border-top:1px solid #e5eaf0;
            color:#64748b;
            font-size:0.82rem;
            font-weight:850;
        }
        .training-flow {
            display:grid;
            grid-template-columns:repeat(5,minmax(0,1fr));
            gap:10px;
            margin:12px 0 18px 0;
        }
        .flow-step {
            border:1px solid #d8dee6;
            border-radius:10px;
            background:#ffffff;
            padding:14px;
            min-height:170px;
            box-shadow:0 8px 22px rgba(24,39,75,0.045);
        }
        .flow-step b {
            display:inline-flex;
            width:28px;
            height:28px;
            align-items:center;
            justify-content:center;
            border-radius:999px;
            background:#eef6ff;
            color:#2563eb;
            font-size:0.84rem;
            margin-bottom:10px;
        }
        .flow-step h4 {
            margin:0 0 8px 0;
            color:#132033;
            font-size:0.98rem;
        }
        .flow-step p {
            margin:0;
            color:#475569;
            font-size:0.86rem;
            line-height:1.58;
        }
        .training-period-grid {
            display:grid;
            grid-template-columns:repeat(3,minmax(0,1fr));
            gap:12px;
            margin:12px 0 16px 0;
        }
        .training-period-card {
            border:1px solid #d8dee6;
            border-radius:10px;
            background:#ffffff;
            padding:15px;
            box-shadow:0 8px 22px rgba(24,39,75,0.045);
        }
        .training-period-card h4 {
            margin:0 0 8px 0;
            color:#132033;
            font-size:1rem;
        }
        .training-period-card p {
            margin:0;
            color:#475569;
            line-height:1.6;
            font-size:0.9rem;
        }
        .method-deep-dive {
            display:grid;
            grid-template-columns:repeat(2,minmax(0,1fr));
            gap:12px;
            margin:12px 0 16px 0;
        }
        .method-panel {
            border:1px solid #d8dee6;
            border-radius:10px;
            background:#ffffff;
            padding:17px;
            box-shadow:0 8px 22px rgba(24,39,75,0.045);
        }
        .method-kicker {
            display:inline-flex;
            border-radius:999px;
            padding:4px 9px;
            background:#eef6ff;
            color:#2563eb;
            font-size:0.76rem;
            font-weight:900;
            margin-bottom:10px;
        }
        .method-panel h4,
        .tuning-main h4,
        .tuning-side h4 {
            margin:0 0 9px 0;
            color:#132033;
            font-size:1.04rem;
            line-height:1.38;
        }
        .method-panel p,
        .tuning-main p {
            margin:0 0 9px 0;
            color:#475569;
            line-height:1.68;
            font-size:0.92rem;
        }
        .tuning-board {
            display:grid;
            grid-template-columns:minmax(0,1.35fr) minmax(280px,0.65fr);
            gap:12px;
            margin:12px 0 16px 0;
        }
        .tuning-main,
        .tuning-side {
            border:1px solid #d8dee6;
            border-radius:10px;
            background:#ffffff;
            padding:17px;
            box-shadow:0 10px 26px rgba(24,39,75,0.055);
        }
        .tuning-main {
            border-top:4px solid #2563eb;
        }
        .tuning-side {
            border-top:4px solid #059669;
        }
        .param-grid {
            display:grid;
            grid-template-columns:repeat(2,minmax(0,1fr));
            gap:8px;
            margin-top:12px;
        }
        .param-row {
            border:1px solid #e5eaf0;
            border-radius:9px;
            background:#f8fafc;
            padding:10px 11px;
            min-height:74px;
        }
        .param-row b {
            display:block;
            color:#132033;
            font-size:0.86rem;
            margin-bottom:4px;
        }
        .param-row span {
            color:#64748b;
            font-size:0.82rem;
            line-height:1.45;
        }
        .tuning-stat {
            display:flex;
            align-items:center;
            justify-content:space-between;
            gap:12px;
            border:1px solid #e5eaf0;
            border-radius:9px;
            background:#f8fafc;
            padding:10px 11px;
            margin-top:8px;
        }
        .tuning-stat span {
            color:#64748b;
            font-size:0.84rem;
            font-weight:800;
        }
        .tuning-stat b {
            color:#132033;
            font-size:0.95rem;
            white-space:nowrap;
        }
        .training-detail-grid {
            display:grid;
            grid-template-columns:repeat(3,minmax(0,1fr));
            gap:12px;
            margin:12px 0 16px 0;
        }
        .training-detail-card {
            border:1px solid #d8dee6;
            border-radius:10px;
            background:#ffffff;
            padding:15px;
            box-shadow:0 8px 22px rgba(24,39,75,0.045);
        }
        .training-detail-card h4 {
            margin:0 0 8px 0;
            color:#132033;
            font-size:1rem;
        }
        .training-detail-card p {
            margin:0 0 8px 0;
            color:#475569;
            line-height:1.6;
            font-size:0.9rem;
        }
        .training-artifacts {
            display:grid;
            grid-template-columns:repeat(3,minmax(0,1fr));
            gap:10px;
            margin:12px 0 4px 0;
        }
        .artifact-card {
            border:1px solid #d8dee6;
            border-radius:10px;
            background:#f8fafc;
            padding:13px;
        }
        .artifact-card b {
            display:block;
            color:#132033;
            margin-bottom:5px;
        }
        .artifact-card span {
            color:#64748b;
            font-size:0.86rem;
            line-height:1.5;
        }
        .period-conclusion-grid {
            display:grid;
            grid-template-columns:repeat(3,minmax(0,1fr));
            gap:14px;
            margin:10px 0 18px 0;
        }
        .period-conclusion-card {
            position:relative;
            overflow:hidden;
            background:#ffffff;
            border:1px solid #d8dee6;
            border-radius:8px;
            padding:16px 16px 15px 16px;
            min-height:188px;
            box-shadow:0 10px 28px rgba(24,39,75,0.06);
        }
        .period-conclusion-card::before {
            content:"";
            position:absolute;
            left:0;
            top:0;
            width:100%;
            height:4px;
            background:var(--accent);
        }
        .period-tag {
            display:inline-flex;
            align-items:center;
            border-radius:999px;
            padding:4px 9px;
            background:var(--soft);
            color:var(--accent);
            font-size:0.78rem;
            font-weight:800;
            margin-bottom:10px;
        }
        .period-conclusion-card h4 {
            margin:0 0 10px 0;
            color:#132033;
            font-size:1.05rem;
            line-height:1.35;
        }
        .period-factor-row {
            display:flex;
            flex-wrap:wrap;
            gap:6px;
            margin:0 0 12px 0;
        }
        .period-factor {
            border:1px solid #d8dee6;
            border-radius:999px;
            padding:3px 8px;
            color:#334155;
            background:#f8fafc;
            font-size:0.78rem;
            font-weight:700;
        }
        .period-conclusion-card p {
            margin:0;
            color:#475569;
            line-height:1.68;
            font-size:0.93rem;
        }
        .period-warning {
            margin-top:10px;
            padding:8px 10px;
            border-radius:8px;
            background:#fff7ed;
            color:#9a3412;
            font-size:0.82rem;
            line-height:1.5;
        }
        @media (max-width: 900px) {
            .forecast-overview {grid-template-columns:repeat(2,minmax(0,1fr));}
            .forecast-panel {grid-template-columns:1fr;}
            .scenario-strip {grid-template-columns:repeat(2,minmax(0,1fr));}
            .factor-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
            .model-hero-grid {grid-template-columns:1fr;}
            .model-route-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
            .intro-hero {grid-template-columns:1fr;}
            .intro-model-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
            .method-band,
            .method-band.three {grid-template-columns:1fr;}
            .training-intro {grid-template-columns:1fr;}
            .training-family-grid {grid-template-columns:1fr;}
            .training-flow {grid-template-columns:repeat(2,minmax(0,1fr));}
            .training-period-grid {grid-template-columns:1fr;}
            .method-deep-dive {grid-template-columns:1fr;}
            .tuning-board {grid-template-columns:1fr;}
            .param-grid {grid-template-columns:1fr;}
            .training-detail-grid {grid-template-columns:1fr;}
            .training-artifacts {grid-template-columns:1fr;}
            .period-conclusion-grid {grid-template-columns:1fr;}
        }
        @media (max-width: 620px) {
            .forecast-overview {grid-template-columns:1fr;}
            .forecast-number {font-size:2.5rem;}
            .model-route-grid {grid-template-columns:1fr;}
            .model-hero-card {padding:18px 16px;}
            .intro-model-grid {grid-template-columns:1fr;}
            .intro-hero-main,
            .intro-score-panel {padding:17px 16px;}
            .training-flow {grid-template-columns:1fr;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    style_page()
    assets = load_assets()
    prediction_models = load_prediction_models()
    next24_bundle = load_model(str(resolve_path("models", "high_accuracy_lightgbm_core_target_pm2_5_next_24h.joblib")))

    city_info = assets["city_info"]
    profiles = assets["profiles"]
    daily = assets["daily"]
    seasonal = assets["seasonal"]
    metadata = assets["metadata"]

    st.title("京津冀 PM2.5 浓度预测与气象贡献度展示")

    left, right = st.columns([0.78, 0.22])
    with right:
        st.caption(f"训练数据: {metadata['start_time']} 至 {metadata['end_time']}")
        st.caption(f"城市数: {metadata['cities']} | ERA5 PBLH 缺失率: {metadata['pblh_missing_rate']:.2%}")

    with st.sidebar:
        st.header("输入")
        city = st.selectbox(
            "城市",
            city_info["city"].tolist(),
            index=city_info["city"].tolist().index("Beijing"),
            key="selected_city",
        )
        selected_date = st.date_input("日期", value=date.today(), key="selected_date")
        hour = st.slider("小时", 0, 23, 12, key="selected_hour")
        st.caption("日期会影响月份、日序、周末特征，并决定载入哪个季节/小时的历史默认气象画像。")

        profile = get_profile(profiles, city, selected_date.month, hour)
        defaults = input_defaults(profile)
        if "input_temperature_2m" not in st.session_state:
            write_input_state(defaults)

        if st.button("载入该城市历史默认值", use_container_width=True):
            write_input_state(defaults)
            st.session_state["input_confirmed"] = False
            st.rerun()

        with st.form("prediction_input_form", border=True):
            st.caption("修改后点击底部按钮，预测结果才会更新。")
            if st.session_state.get("selected_prediction_model") not in MODEL_SELECT_OPTIONS:
                st.session_state["selected_prediction_model"] = MODEL_SELECT_OPTIONS[0]
            model_choice = st.selectbox(
                "当前小时预测模型",
                MODEL_SELECT_OPTIONS,
                index=0,
                key="selected_prediction_model",
            )
            st.caption("高精度模型适合实际预测；v2 气象贡献模型适合观察气象-only 条件下的贡献情景；旧版气象归因模型保留为对照。")
            st.subheader("近地面气象")
            c1, c2 = st.columns(2)
            with c1:
                temperature = st.number_input("2m 气温 (C)", step=0.5, key="input_temperature_2m")
                pressure = st.number_input("海平面气压 (hPa)", step=0.5, key="input_pressure_msl")
                cloud = st.slider("云量 (%)", 0.0, 100.0, key="input_cloud_cover")
            with c2:
                humidity = st.slider("相对湿度 (%)", 0.0, 100.0, key="input_relative_humidity_2m")
                surface_pressure = st.number_input("地面气压 (hPa)", step=0.5, key="input_surface_pressure")
                precipitation = st.number_input("降水量 (mm)", min_value=0.0, step=0.1, key="input_precipitation")

            st.subheader("风与稳定度")
            c3, c4 = st.columns(2)
            with c3:
                wind_speed = st.number_input("10m 风速 (m/s)", min_value=0.0, step=0.2, key="input_wind_speed_10m")
                wind_gust = st.number_input("阵风 (m/s)", min_value=0.0, step=0.2, key="input_wind_gusts_10m")
                t850 = st.number_input("850hPa 温度 (C)", step=0.5, key="input_temperature_850hPa")
            with c4:
                wind_direction = st.slider("风向 (度)", 0.0, 360.0, key="input_wind_direction_10m")
                pblh = st.number_input("ERA5 PBLH (m)", min_value=1.0, step=20.0, key="input_boundary_layer_height")
                t1000 = st.number_input("1000hPa 温度 (C)", step=0.5, key="input_temperature_1000hPa")

            st.subheader("污染持续性")
            lag1 = st.number_input("前 1 小时 PM2.5 (ug/m3)", min_value=0.0, step=1.0, key="input_pm2_5_lag_1h")
            lag3 = st.number_input("前 3 小时 PM2.5 (ug/m3)", min_value=0.0, step=1.0, key="input_pm2_5_lag_3h")
            lag24 = st.number_input("前 24 小时 PM2.5 (ug/m3)", min_value=0.0, step=1.0, key="input_pm2_5_lag_24h")
            submitted = st.form_submit_button("确认输入并预测", type="primary", use_container_width=True)

        form_values = {
            "temperature_2m": temperature,
            "relative_humidity_2m": humidity,
            "pressure_msl": pressure,
            "surface_pressure": surface_pressure,
            "cloud_cover": cloud,
            "precipitation": precipitation,
            "wind_speed_10m": wind_speed,
            "wind_direction_10m": wind_direction,
            "wind_gusts_10m": wind_gust,
            "boundary_layer_height": pblh,
            "temperature_850hPa": t850,
            "temperature_1000hPa": t1000,
            "pm2_5_lag_1h": lag1,
            "pm2_5_lag_3h": lag3,
            "pm2_5_lag_24h": lag24,
        }
        if submitted or "confirmed_inputs" not in st.session_state:
            st.session_state["confirmed_inputs"] = {
                "city": city,
                "date": selected_date,
                "hour": hour,
                "model_choice": model_choice,
                "values": form_values,
            }
            st.session_state["input_confirmed"] = True

        confirmed = st.session_state["confirmed_inputs"]
        changed = (
            confirmed["city"] != city
            or confirmed["date"] != selected_date
            or confirmed["hour"] != hour
            or confirmed.get("model_choice", MODEL_SELECT_OPTIONS[0]) != model_choice
            or any(float(confirmed["values"][key]) != float(form_values[key]) for key in form_values)
        )
        if changed:
            st.info("当前表单已有修改，点击“确认输入并预测”后主面板才会更新。")
        elif st.session_state.get("input_confirmed", False):
            st.success("输入已确认，结果来自当前表单。")

    city = confirmed["city"]
    selected_date = confirmed["date"]
    hour = confirmed["hour"]
    model_choice = confirmed.get("model_choice", MODEL_SELECT_OPTIONS[0])
    overrides = make_overrides(confirmed["values"])
    pblh = overrides["boundary_layer_height"]
    wind_speed = overrides["wind_speed_10m"]
    wind_direction = overrides["wind_direction_10m"]
    row = scenario_row(city_info, profiles, city, selected_date, hour, overrides)
    selected_model_key = resolve_prediction_model_key(model_choice, selected_date, hour)
    selected_model = PREDICTION_MODEL_SPECS[selected_model_key]
    current_bundle = prediction_models[selected_model_key]
    current_prediction = predict(current_bundle, row)
    next24_prediction = predict(next24_bundle, row)
    category, color = pm25_category(current_prediction)
    next_category, next_color = pm25_category(next24_prediction)
    t_inverse = row["t_inverse_850_1000"]

    tab_predict, tab_results, tab_training = st.tabs(["预测台", "模型介绍", "训练策略"])

    with tab_predict:
        st.markdown(
            scenario_summary_html(
                city,
                selected_date,
                hour,
                selected_model,
                current_prediction,
                next24_prediction,
                category,
                next_category,
                color,
                next_color,
                t_inverse,
                pblh,
                row,
                overrides,
                wind_speed,
                wind_direction,
            ),
            unsafe_allow_html=True,
        )
        st.caption(selected_model["description"])
        selected_period = period_for_datetime(selected_date, hour)
        expected_period = MODEL_KEY_TO_PERIOD.get(selected_model_key)
        if expected_period and expected_period != selected_period:
            st.warning(
                f"当前日期属于{PERIOD_LABELS[selected_period]}，但你选择的是{PERIOD_LABELS[expected_period]}模型。"
                "这属于跨时期外推，结果更适合做敏感性对照，不建议作为主预测。"
            )
        if "气象" in selected_model["type"]:
            st.info("当前使用的是气象贡献/归因模型：它会弱化或排除污染持续性和共污染物信息，适合解释气象贡献，预测精度通常低于高精度模型。")
        if selected_model["type"] == "气象贡献 v2":
            st.caption("v2 模型需要长时滞、滚动和复合扩散特征；预测台会用当前输入叠加该城市月小时历史画像自动补齐这些气象过程变量。")
        st.markdown(risk_cards_html(risk_items(row, current_prediction)), unsafe_allow_html=True)

        chart_col, map_col = st.columns([0.62, 0.38])
        with chart_col:
            day_pred = build_daily_prediction(current_bundle, city_info, profiles, city, selected_date, overrides)
            fig = px.line(day_pred, x="hour", y="predicted_pm2_5", markers=True, title="当天逐小时 PM2.5 预测曲线")
            add_pm25_bands(fig, float(day_pred["predicted_pm2_5"].max()))
            fig.add_vline(x=hour, line_width=2, line_dash="dash", line_color="#c62828")
            fig.update_traces(line=dict(width=3, color="#2563eb"), marker=dict(size=7, color="#2563eb"))
            fig.update_layout(height=390, margin=dict(l=10, r=10, t=50, b=20), yaxis_title="ug/m3", xaxis_title="小时")
            st.plotly_chart(fig, use_container_width=True)
        with map_col:
            city_month = seasonal[seasonal["month"] == selected_date.month][["city", "pm2_5_median"]]
            map_data = city_info.merge(city_month, on="city", how="left")
            fig = px.scatter(
                map_data,
                x="longitude",
                y="latitude",
                size="pm2_5_median",
                color="pm2_5_median",
                hover_name="city",
                color_continuous_scale="RdYlGn_r",
                title="京津冀城市历史 PM2.5 中位数",
            )
            selected = map_data[map_data["city"] == city]
            fig.add_trace(
                go.Scatter(
                    x=selected["longitude"],
                    y=selected["latitude"],
                    mode="markers+text",
                    text=selected["city"],
                    textposition="top center",
                    marker=dict(size=18, color="#111827", symbol="x"),
                    name="当前城市",
                )
            )
            fig.update_traces(marker=dict(line=dict(width=1, color="#263238")))
            fig.update_layout(height=390, margin=dict(l=10, r=10, t=50, b=20), xaxis_title="经度", yaxis_title="纬度")
            st.plotly_chart(fig, use_container_width=True)

        render_weather_context(
            seasonal,
            daily,
            current_bundle,
            city_info,
            profiles,
            city,
            selected_date,
            overrides,
            wind_speed,
            wind_direction,
        )

    with tab_results:
        st.markdown(public_training_update_html(assets), unsafe_allow_html=True)

        high_accuracy_tab, attribution_tab = st.tabs(["高精度预测模型", "气象贡献模型"])

        with high_accuracy_tab:
            st.markdown(high_accuracy_intro_html(assets), unsafe_allow_html=True)

            st.subheader("模型表现总览")
            st.caption("这一页只展示追求预测精度的模型。它们适合预测台交互、性能展示和预测误差讨论。")
            performance = high_accuracy_performance_table(assets)
            chart_a, chart_b = st.columns(2)
            chart_a.plotly_chart(performance_chart(performance), use_container_width=True)
            chart_b.plotly_chart(r2_chart(performance), use_container_width=True)
            st.dataframe(performance, use_container_width=True, hide_index=True)

            with st.expander("数据补齐与高精度训练口径", expanded=True):
                st.write(
                    "2018-2022 新增 PM2.5 来自 CNEMC/quotsoft 城市小时空气质量数据，"
                    "与 ERA5/CDS 城市小时气象、ERA5 PBLH 合并后形成 2018+ 全时期训练表。"
                )
                st.write(
                    "全时期训练表共 958,711 条 city-hour 记录，13 个城市，PM2.5 缺测率为 0%。"
                    "源站缺失的 2018-12-22 至 2018-12-26、2019-08-24 共 6 个原始日文件，"
                    "已按城市和污染物做线性插补，并在报告中记录。"
                )
                st.write(
                    "高精度模型使用 PM2.5 前 1/3/24 小时时滞、滚动均值、共污染物和气象变量，"
                    "目的是尽量复原当前小时 PM2.5 的实际浓度。因此它是应用预测模型，不是纯气象归因模型。"
                )

            st.markdown(
                """
                <div class="explain-band blue">
                  <h4>结果解读口径</h4>
                  <p>全时期高精度模型是网页预测台的默认主模型，用于展示项目在实际输入场景下的预测能力。分时期高精度模型则把疫情前、疫情期、疫情后分别训练，用于比较特殊时期污染持续性的阶段性变化。</p>
                  <p>如果高精度模型 R2 明显高于气象贡献模型，并不说明气象不重要，而是说明 PM2.5 历史浓度和共污染物对短时预测非常有信息量。论文中可把这部分写成“预测能力上限”，再用气象贡献模型讨论独立气象影响。</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("高精度模型卡片")
            st.caption("点击卡片展开后，可查看模型定位、研究任务、特征口径、训练样本、迭代次数和 SHAP 前列特征。")
            for spec in model_card_specs():
                if spec["family"] in {"预测主模型", "分时期预测", "提前量预测"}:
                    render_model_card(spec, assets)

            st.subheader("高精度模型 SHAP 解释")
            shap_a, shap_b = st.columns(2)
            shap_a.plotly_chart(shap_chart(assets["current_shap"], "全时期高精度模型 SHAP 贡献"), use_container_width=True)
            shap_b.plotly_chart(shap_chart(assets["pre_covid_high_accuracy_shap"], "疫情前高精度模型 SHAP 贡献"), use_container_width=True)

            h1, h2, h3 = st.columns(3)
            with h1:
                st.markdown("**疫情前高精度**")
                st.caption(metric_text(assets["pre_covid_high_accuracy_metrics"]))
                st.dataframe(top_shap_table(assets["pre_covid_high_accuracy_shap"], 8), use_container_width=True, hide_index=True)
            with h2:
                st.markdown("**疫情期高精度**")
                st.caption(metric_text(assets["covid_high_accuracy_metrics"]))
                st.dataframe(top_shap_table(assets["covid_high_accuracy_shap"], 8), use_container_width=True, hide_index=True)
            with h3:
                st.markdown("**疫情后高精度**")
                st.caption(metric_text(assets["post_covid_high_accuracy_metrics"]))
                st.dataframe(top_shap_table(assets["post_covid_high_accuracy_shap"], 8), use_container_width=True, hide_index=True)

            st.markdown(
                """
                <div class="explain-band amber">
                  <h4>高精度 SHAP 的论文写法</h4>
                  <p>高精度模型的 SHAP 前列通常包含 PM2.5 滞后、滚动均值和共污染物。它们代表污染过程的短时记忆和共变结构，能显著提升预测精度，但会混合排放、人为活动和二次生成信息。</p>
                  <p>因此，高精度模型的学术价值在于证明“PM2.5 浓度可被机器学习稳定预测”，并为气象贡献模型提供性能参照；真正讨论 PBLH、逆温、湿度、气压、风输送贡献时，应切换到右侧“气象贡献模型”标签页。</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with attribution_tab:
            st.markdown(meteorology_contribution_intro_html(assets), unsafe_allow_html=True)
            v2 = assets.get("meteorology_v2")

            if v2:
                st.subheader("v2 最新气象贡献模型总览")
                st.caption("本轮训练把疫情前、疫情期、疫情后三个时期分别建模，并对 PM2.5 原值、log1p(PM2.5)、同城同月同小时气候态异常三种目标逐一比较。")
                v2_summary = meteorology_v2_summary_table(assets)
                v2_best = meteorology_v2_best_table(assets)
                v2_chart_col, v2_table_col = st.columns([0.55, 0.45])
                v2_chart_col.plotly_chart(meteorology_v2_r2_chart(v2_summary), use_container_width=True)
                with v2_table_col:
                    st.markdown("**每个时期的最佳归因模型**")
                    st.dataframe(v2_best, use_container_width=True, hide_index=True)

                with st.expander("查看 9 套 v2 候选模型完整指标", expanded=False):
                    st.dataframe(v2_summary, use_container_width=True, hide_index=True)
                    st.caption("同一时期内比较三种目标形式，可以避免只用 PM2.5 原值时被极端污染和排放背景过度牵引。")

                st.subheader("旧版气象模型与 v2 精进效果")
                old_new = meteorology_v2_old_new_compare_table(assets)
                st.dataframe(old_new, use_container_width=True, hide_index=True)
                st.markdown(
                    """
                    <div class="explain-band green">
                      <h4>v2 模型的学术价值</h4>
                      <p>旧版气象贡献模型主要验证“只用气象变量也能解释一部分 PM2.5”。v2 则进一步把气象过程做成时滞、滚动、累计和复合扩散指数，并用三种目标形式筛选每个时期最合适的解释尺度。</p>
                      <p>这使得模型不只是给出一个 R2，而是能讨论连续低 PBLH、长时间弱风、高湿、通风系数、北风清洁输送、南北风 V 分量和气压滞后等具有大气环境意义的过程变量。</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.subheader("v2 最佳模型卡片")
                st.caption("点击展开后可查看每个时期最佳目标、样本量、特征数、最佳迭代和带 bootstrap 置信区间的 Top SHAP。")
                for index, model in enumerate(v2.get("best_models", [])):
                    test = model["test_pm25"]
                    title = (
                        f"{model['period_label']} | {model['target_label']} | "
                        f"R2 {test['r2']:.3f} / RMSE {test['rmse']:.2f}"
                    )
                    with st.expander(title, expanded=index == 1):
                        metric_cols = st.columns(4)
                        metric_cols[0].metric("测试 MAE", f"{test['mae']:.2f}")
                        metric_cols[1].metric("测试 RMSE", f"{test['rmse']:.2f}")
                        metric_cols[2].metric("测试 R2", f"{test['r2']:.3f}")
                        metric_cols[3].metric("Bias", f"{test['bias']:.2f}")
                        st.markdown(f'<div class="v2-card-note">{v2_period_narrative(model)}</div>', unsafe_allow_html=True)
                        info_col, shap_col = st.columns([0.38, 0.62])
                        with info_col:
                            st.markdown("**训练信息**")
                            st.dataframe(
                                pd.DataFrame(
                                    [
                                        ("目标形式", model["target_label"]),
                                        ("训练样本", f"{int(model['train_rows']):,}"),
                                        ("验证样本", f"{int(model['valid_rows']):,}"),
                                        ("测试样本", f"{int(model['test_rows']):,}"),
                                        ("特征数", f"{int(model['feature_count']):,}"),
                                        ("最佳迭代", str(model["best_iteration"])),
                                        ("模型文件", model["model_path"]),
                                    ],
                                    columns=["项目", "值"],
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )
                        with shap_col:
                            st.markdown("**Top SHAP 与置信区间**")
                            st.dataframe(meteorology_v2_shap_table(model, 10), use_container_width=True, hide_index=True)
                        st.plotly_chart(meteorology_v2_shap_chart(model), use_container_width=True)
                        st.caption("Bootstrap 置信区间用于评估 SHAP 排名稳定性；特征贡献仍是模型解释结果，需要结合大气物理机制讨论。")
            else:
                st.warning("当前 app_assets 中未找到 meteorology_attribution_v2_core_results.json，暂时只能展示旧版气象归因模型。")
                v2_best = pd.DataFrame()

            st.subheader("气象贡献模型与高精度模型的分工")
            if v2 and not v2_best.empty:
                best_map = {row["时期"]: row for _, row in v2_best.iterrows()}

                def best_metric(period_label: str, column: str, fallback: float) -> float:
                    row = best_map.get(period_label)
                    return float(row[column]) if row is not None else float(fallback)
            else:
                def best_metric(period_label: str, column: str, fallback: float) -> float:
                    return float(fallback)

            compare = pd.DataFrame(
                [
                    {
                        "时期": "疫情前",
                        "高精度 R2": assets["pre_covid_high_accuracy_metrics"]["test"]["r2"],
                        "高精度 RMSE": assets["pre_covid_high_accuracy_metrics"]["test"]["rmse"],
                        "气象贡献 R2": best_metric("疫情前", "R2", assets["pre_covid_meteorology_metrics"]["test"]["r2"]),
                        "气象贡献 RMSE": best_metric("疫情前", "RMSE", assets["pre_covid_meteorology_metrics"]["test"]["rmse"]),
                        "解释重点": "正常排放背景下扩散条件、季节周期和区域输送对污染差异的放大效应",
                    },
                    {
                        "时期": "疫情期",
                        "高精度 R2": assets["covid_high_accuracy_metrics"]["test"]["r2"],
                        "高精度 RMSE": assets["covid_high_accuracy_metrics"]["test"]["rmse"],
                        "气象贡献 R2": best_metric("疫情期", "R2", assets["covid_meteorology_metrics"]["test"]["r2"]),
                        "气象贡献 RMSE": best_metric("疫情期", "RMSE", assets["covid_meteorology_metrics"]["test"]["rmse"]),
                        "解释重点": "人为活动减弱期间 PBLH、湿度、气压和风输送对污染波动的持续解释力",
                    },
                    {
                        "时期": "疫情后",
                        "高精度 R2": assets["post_covid_high_accuracy_metrics"]["test"]["r2"],
                        "高精度 RMSE": assets["post_covid_high_accuracy_metrics"]["test"]["rmse"],
                        "气象贡献 R2": best_metric("疫情后", "R2", assets["post_covid_meteorology_metrics"]["test"]["r2"]),
                        "气象贡献 RMSE": best_metric("疫情后", "RMSE", assets["post_covid_meteorology_metrics"]["test"]["rmse"]),
                        "解释重点": "恢复期边界层约束、湿度过程和通风扩散能力的重新强化",
                    },
                ]
            )
            for column in ["高精度 R2", "高精度 RMSE", "气象贡献 R2", "气象贡献 RMSE"]:
                compare[column] = compare[column].astype(float).round(3)
            st.dataframe(compare, use_container_width=True, hide_index=True)
            st.caption("高精度模型用于评估应用预测能力；气象贡献模型用于评估气象背景对污染变化的独立解释力。论文的归因结论应以气象贡献模型为主。")

            st.subheader("旧版气象归因模型卡片")
            st.caption("这些是预测台当前可直接选择的气象归因模型。它们保留为交互展示和旧版对照；本轮论文式结论优先引用 v2 最新模型。")
            for spec in model_card_specs():
                if spec["family"] == "气象归因":
                    render_model_card(spec, assets)

            st.subheader("旧版气象 SHAP 与残差辅助")
            p1, p2, p3 = st.columns(3)
            with p1:
                st.markdown("**疫情前 2018-2019**")
                st.caption(metric_text(assets["pre_covid_meteorology_metrics"]))
                st.dataframe(top_shap_table(assets["pre_covid_meteorology_shap"], 8), use_container_width=True, hide_index=True)
            with p2:
                st.markdown("**疫情期 2020-2022**")
                st.caption(metric_text(assets["covid_meteorology_metrics"]))
                st.dataframe(top_shap_table(assets["covid_meteorology_shap"], 8), use_container_width=True, hide_index=True)
            with p3:
                st.markdown("**疫情后 2023+**")
                st.caption(metric_text(assets["post_covid_meteorology_metrics"]))
                st.dataframe(top_shap_table(assets["post_covid_meteorology_shap"], 8), use_container_width=True, hide_index=True)

            residual = assets["period_residual_analysis"][
                ["时期", "气象模型_R2", "高精度模型_R2", "气象残差均值", "负残差占比"]
            ].copy()
            for column in ["气象模型_R2", "高精度模型_R2", "气象残差均值", "负残差占比"]:
                residual[column] = residual[column].astype(float).round(3)
            st.dataframe(residual, use_container_width=True, hide_index=True)
            st.caption("气象残差 = 实测 PM2.5 - 气象模型预测 PM2.5。残差只作为非气象因素讨论的辅助证据，不能直接等同于排放变化量。")

            st.subheader("阶段性归因结论")
            st.markdown(
                """
                <div class="period-conclusion-grid">
                  <section class="period-conclusion-card" style="--accent:#059669;--soft:#ecfdf5;">
                    <div class="period-tag">疫情前 2018-2019</div>
                    <h4>季节背景、空间差异与边界层扩散共同控制</h4>
                    <div class="period-factor-row">
                      <span class="period-factor">年内周期</span>
                      <span class="period-factor">纬度</span>
                      <span class="period-factor">24小时平均 PBLH</span>
                      <span class="period-factor">北风清洁输送</span>
                    </div>
                    <p>疫情前最佳 v2 模型显示，PM2.5 变化并非只由单时刻气象决定，而是由季节周期、城市南北空间差异、连续边界层扩散能力和冷空气输送共同约束。</p>
                  </section>
                  <section class="period-conclusion-card" style="--accent:#d97706;--soft:#fffbeb;">
                    <div class="period-tag">疫情期 2020-2022</div>
                    <h4>气象解释力仍然存在，区域输送和气压过程突出</h4>
                    <div class="period-factor-row">
                      <span class="period-factor">24小时平均 PBLH</span>
                      <span class="period-factor">南北风 V 分量</span>
                      <span class="period-factor">气压滞后</span>
                      <span class="period-factor">露点/湿度</span>
                    </div>
                    <p>疫情期人为活动减弱，但气象归因模型仍能获得较高解释度，说明静稳扩散、风向输送和湿度过程仍会显著影响 PM2.5 波动。</p>
                  </section>
                  <section class="period-conclusion-card" style="--accent:#0f766e;--soft:#ecfdf5;">
                    <div class="period-tag">疫情后 2023+</div>
                    <h4>边界层约束和通风扩散能力重新成为核心证据</h4>
                    <div class="period-factor-row">
                      <span class="period-factor">24小时平均 PBLH</span>
                      <span class="period-factor">通风系数</span>
                      <span class="period-factor">平均风速</span>
                      <span class="period-factor">北风清洁输送</span>
                    </div>
                    <p>疫情后最佳模型使用气候态异常目标，说明气象贡献更适合解释相对本地季节背景的偏离。PBLH、通风系数和风速共同指向扩散条件对污染异常的控制作用。</p>
                    <div class="period-warning">跨时期比较需注明：2023+ PM2.5 数据源与 2018-2022 不完全一致。</div>
                  </section>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_training:
        training_prediction_tab, training_attribution_tab = st.tabs(["高精度预测模型", "气象归因模型"])

        with training_prediction_tab:
            st.markdown(high_accuracy_training_html(assets), unsafe_allow_html=True)

            st.subheader("高精度模型训练明细")
            st.caption(
                "这里保留追求预测准确率的模型：全时期主模型、疫情前/疫情期/疫情后分时期高精度模型，"
                "以及 24 小时辅助预测模型。它们可以使用 PM2.5 历史浓度和共污染物，因此适合讨论预测能力上限。"
            )
            st.dataframe(high_accuracy_training_rows(assets), use_container_width=True, hide_index=True)

            st.subheader("高精度模型调参和迭代")
            st.caption(
                "这一部分说明高精度模型的 Optuna + LightGBM 参数搜索、时间后置验证设计，"
                "以及测试集仅用于最终评估的实验规范。"
            )
            st.markdown(high_accuracy_tuning_html(assets), unsafe_allow_html=True)

            st.subheader("高精度模型训练产物")
            st.markdown(
                """
                <div class="training-artifacts">
                  <div class="artifact-card"><b>models/</b><span>保存全时期主模型、分时期高精度模型和 24 小时辅助预测模型。</span></div>
                  <div class="artifact-card"><b>reports/</b><span>保存高精度模型指标、测试集预测、SHAP 排名、训练摘要和模型对比结果。</span></div>
                  <div class="artifact-card"><b>app_assets/</b><span>保存预测台和模型介绍页直接读取的高精度模型展示数据。</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("网页预测台中的默认主模型和分时期高精度模型，均来自这一套训练流程。")

        with training_attribution_tab:
            st.markdown(meteorology_training_html(assets), unsafe_allow_html=True)

            st.subheader("训练明细：旧版 baseline")
            st.caption(
                "这张表记录旧版气象-only baseline 的样本量、特征集、调参轮数和测试表现，"
                "用于和 v2 过程型气象训练策略形成方法对照。"
            )
            st.dataframe(meteorology_legacy_training_rows(assets), use_container_width=True, hide_index=True)

            v2_rows = meteorology_v2_training_rows(assets)
            if not v2_rows.empty:
                st.subheader("训练明细：v2 候选实验矩阵")
                st.caption(
                    "这张表对应 3 个时期 x 3 种目标形式的候选训练矩阵，展示每套候选的样本量、特征数、最佳迭代和测试集指标。"
                )
                st.dataframe(v2_rows, use_container_width=True, hide_index=True)

            st.subheader("参数搜索、评估分层与解释输出")
            st.caption(
                "这一部分对应训练策略的执行细节：参数搜索以验证集为依据，测试集用于最终比较，"
                "条件误差和 SHAP bootstrap 用于支撑归因结论。"
            )
            st.markdown(meteorology_tuning_html(assets), unsafe_allow_html=True)

            st.subheader("气象归因训练产物")
            st.markdown(
                """
                <div class="training-artifacts">
                  <div class="artifact-card"><b>models/</b><span>保存旧版气象 baseline、v2 时期最佳气象贡献模型，以及可在预测台选择的时期模型。</span></div>
                  <div class="artifact-card"><b>reports/</b><span>保存气象-only 指标、v2 三目标对比、条件误差、天气型结果和 SHAP bootstrap 置信区间。</span></div>
                  <div class="artifact-card"><b>app_assets/</b><span>保存模型介绍页、训练策略页和预测台读取的气象贡献模型结果。</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("论文中的气象因子贡献、疫情期与非疫情期对比，建议优先引用 v2 气象贡献模型。")

if __name__ == "__main__":
    main()
