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
        "description": "全时期综合预测基准，用于评估多源环境信息约束下的短时浓度估计能力。",
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
        "label": "疫情前基础气象归因模型",
        "path": "high_accuracy_lightgbm_meteorology_target_pm2_5_pre_covid_2018_2019_meteorology_only.joblib",
        "type": "基础气象归因",
        "description": "基础气象-only 口径，用于与过程型气象贡献模型进行方法对照。",
    },
    "covid_meteorology": {
        "label": "疫情期基础气象归因模型",
        "path": "high_accuracy_lightgbm_meteorology_target_pm2_5_covid_2020_2022_meteorology_only.joblib",
        "type": "基础气象归因",
        "description": "基础气象-only 口径，用于疫情期气象贡献结构的基准对照。",
    },
    "post_meteorology": {
        "label": "疫情后基础气象归因模型",
        "path": "high_accuracy_lightgbm_meteorology_target_pm2_5_post_covid_2023_plus_meteorology_only.joblib",
        "type": "基础气象归因",
        "description": "基础气象-only 口径，用于恢复期气象贡献结构的基准对照。",
    },
    "pre_meteorology_v2": {
        "label": "疫情前过程型气象贡献模型",
        "path": "meteorology_attribution_v2_core_pre_covid_2018_2019_log1p.joblib",
        "type": "过程型气象贡献",
        "description": "过程型气象-only 口径，目标为 log1p(PM2.5)，用于疫情前气象贡献估计。",
    },
    "covid_meteorology_v2": {
        "label": "疫情期过程型气象贡献模型",
        "path": "meteorology_attribution_v2_core_covid_2020_2022_raw.joblib",
        "type": "过程型气象贡献",
        "description": "过程型气象-only 口径，目标为 PM2.5 原值，用于疫情期气象贡献估计。",
    },
    "post_meteorology_v2": {
        "label": "疫情后过程型气象贡献模型",
        "path": "meteorology_attribution_v2_core_post_covid_2023_plus_anomaly.joblib",
        "type": "过程型气象贡献",
        "description": "过程型气象-only 口径，目标为同城同月同小时气候态异常，用于疫情后气象贡献估计。",
    },
}

MODEL_SELECT_OPTIONS = [
    "全时期高精度模型",
    "按日期自动选择分时期高精度模型",
    "按日期自动选择过程型气象贡献模型",
    "按日期自动选择基础气象归因模型（对照口径）",
    "疫情前高精度模型",
    "疫情期高精度模型",
    "疫情后高精度模型",
    "疫情前过程型气象贡献模型",
    "疫情期过程型气象贡献模型",
    "疫情后过程型气象贡献模型",
    "疫情前基础气象归因模型",
    "疫情期基础气象归因模型",
    "疫情后基础气象归因模型",
]

MODEL_LABEL_TO_KEY = {
    "全时期高精度模型": "full_high_accuracy",
    "疫情前高精度模型": "pre_high_accuracy",
    "疫情期高精度模型": "covid_high_accuracy",
    "疫情后高精度模型": "post_high_accuracy",
    "疫情前过程型气象贡献模型": "pre_meteorology_v2",
    "疫情期过程型气象贡献模型": "covid_meteorology_v2",
    "疫情后过程型气象贡献模型": "post_meteorology_v2",
    "疫情前基础气象归因模型": "pre_meteorology",
    "疫情期基础气象归因模型": "covid_meteorology",
    "疫情后基础气象归因模型": "post_meteorology",
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
    "relative_humidity_2m_roll_mean_6h": "6小时平均相对湿度",
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
    "northerly_cleaning_intensity": "北风清洁输送强度",
    "northerly_cleaning_10m_roll_mean_24h": "24小时北风清洁输送",
    "northerly_cleaning_10m_roll_mean_48h": "48小时北风清洁输送",
    "northerly_cleaning_10m_roll_mean_72h": "72小时北风清洁输送",
    "southerly_transport_10m": "南风污染输送",
    "southerly_transport_intensity": "南风输送强度",
    "ventilation_coefficient": "通风系数",
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
    research_upgrade_path = asset_dir / "atmospheric_research_upgrade_results.json"
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
        "combined_extrapolation": (
            pd.read_csv(asset_dir / "meteorology_v2_combined_space_time_extrapolation.csv")
            if (asset_dir / "meteorology_v2_combined_space_time_extrapolation.csv").exists()
            else pd.DataFrame()
        ),
        "combined_extrapolation_summary": (
            pd.read_csv(asset_dir / "meteorology_v2_combined_space_time_extrapolation_summary.csv")
            if (asset_dir / "meteorology_v2_combined_space_time_extrapolation_summary.csv").exists()
            else pd.DataFrame()
        ),
        "feature_group_ablation": (
            pd.read_csv(asset_dir / "meteorology_v2_feature_group_ablation.csv")
            if (asset_dir / "meteorology_v2_feature_group_ablation.csv").exists()
            else pd.DataFrame()
        ),
        "validation_extension_metadata": (
            json.loads((asset_dir / "meteorology_validation_extension_run_metadata.json").read_text(encoding="utf-8"))
            if (asset_dir / "meteorology_validation_extension_run_metadata.json").exists()
            else {}
        ),
        "meteorology_v2": (
            json.loads(meteorology_v2_path.read_text(encoding="utf-8"))
            if meteorology_v2_path.exists()
            else None
        ),
        "research_upgrade": (
            json.loads(research_upgrade_path.read_text(encoding="utf-8"))
            if research_upgrade_path.exists()
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
      <div class="explain-band green">
        <h4>项目定位</h4>
        <p>本项目面向京津冀城市小时尺度 PM2.5 浓度预测与气象贡献度分析，围绕多源空气质量与气象资料整合、LightGBM 建模、分时期重训和模型解释验证展开。当前网页把工作分成两条主线：一条是面向应用预测的高精度模型，另一条是用于讨论气象背景解释贡献的 weather-only 模型。SHAP、残差和天气型分析均按模型解释证据处理，不直接表述为严格因果效应。</p>
      </div>
      <div class="model-hero-grid">
        <div class="model-hero-card primary">
          <div class="model-kicker">综合预测基准</div>
          <h3>全时期高精度模型</h3>
          <p>覆盖 2018-01-01 至 2026-05-31，综合气象、PBLH、逆温、风输送、污染时滞、滚动均值和共污染物，作为当前小时浓度预测的主模型。</p>
          <div class="score-row">{metric_badges(current)}</div>
        </div>
        <div class="model-hero-card muted">
          <div class="model-kicker">提前量参照</div>
          <h3>24 小时辅助模型</h3>
          <p>给出 24 小时后趋势参考，主要用于观察提前量变化下的预测难度，不作为气象贡献分析的核心依据。</p>
          <div class="score-row">{metric_badges(next24)}</div>
        </div>
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
          <div class="intro-kicker">预测模型体系</div>
          <h3>高精度预测模型：短时浓度估计与性能参照</h3>
          <p>高精度模型的任务是尽量准确地估计当前小时 PM2.5 浓度，因此特征矩阵同时纳入气象、ERA5 PBLH、逆温、风输送、PM2.5 时滞、滚动均值、共污染物和时空控制变量。它反映的是完整信息条件下的预测能力上限，分时期模型沿用这一预测口径作为阶段对照，并不承担纯气象贡献解释任务。</p>
        </div>
        <div class="intro-score-panel">
          <div class="intro-score-label">当前主预测口径</div>
          <h4>全时期高精度模型</h4>
          <div class="score-row">{metric_badges(current)}</div>
          <p>全时期模型负责统一预测口径；分时期高精度模型是既有预测基准，调参轮数为 25 轮/时期。</p>
        </div>
      </section>
      <div class="intro-model-grid">
        <article class="intro-model-card blue">
          <div class="model-card-tag">全时期</div>
          <h4>统一主模型</h4>
          <p>在完整 2018+ 数据上训练，学习跨年份、跨城市的总体非线性规律。</p>
          <div class="score-row compact">{metric_badges(current)}</div>
        </article>
        <article class="intro-model-card green">
          <div class="model-card-tag">疫情前</div>
          <h4>正常时期预测参照</h4>
          <p>2018-2019 单独训练，用作常规排放背景下的预测参照。</p>
          <div class="score-row compact">{metric_badges(pre)}</div>
        </article>
        <article class="intro-model-card amber">
          <div class="model-card-tag">疫情期</div>
          <h4>疫情时期预测参照</h4>
          <p>2020-2022 单独训练，用于观察特殊时期污染持续性改变后的预测稳定性。</p>
          <div class="score-row compact">{metric_badges(covid)}</div>
        </article>
        <article class="intro-model-card teal">
          <div class="model-card-tag">疫情后</div>
          <h4>恢复期预测对照</h4>
          <p>2023+ 单独训练。跨时期解释时需注明 2023+ PM2.5 数据源与 2018-2022 不完全一致。</p>
          <div class="score-row compact">{metric_badges(post)}</div>
        </article>
      </div>
      <p class="v2-card-note">本轮重训没有重新训练高精度模型；网页保留其指标，是为了给过程型气象贡献模型提供预测性能参照。高精度模型的较高 R2 主要来自污染持续性与共污染物的信息增益，讨论 PBLH、逆温、湿度、气压和风输送的相对独立贡献时，应转向气象-only 模型。</p>
    </div>
    """


def meteorology_contribution_intro_html(assets: Assets) -> str:
    v2 = assets.get("meteorology_v2") or {}
    scope = v2.get("retraining_scope", {})
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
          <div class="intro-kicker">过程型气象贡献模型体系</div>
          <h3>过程型气象贡献模型：气象-only 变量约束下的解释模型</h3>
          <p>过程型气象贡献模型是本轮重训的核心。模型主动剔除 PM2.5 时滞、滚动均值和共污染物，只保留气象、PBLH、逆温、风输送、降水、复合扩散指数、气象时滞/累积特征以及城市和时间变量。它的精度低于高精度预测模型是预期结果，研究价值在于更清楚地讨论边界层、湿度、气压、通风条件和区域输送的模型解释贡献。</p>
        </div>
        <div class="intro-score-panel">
          <div class="intro-score-label">过程型气象-only 口径</div>
          <h4>分时期代表模型</h4>
          <p>本轮重训只针对 v2-core 过程型气象贡献模型：3 个时期 x 3 种目标形式，共 9 套候选；每套均完成 {int(scope.get("optuna_trials_per_model", 60))} 轮 Optuna trial。旧版基础气象归因模型为 12 轮/时期，高精度分时期模型为 25 轮/时期，二者均作为既有基准保留。</p>
        </div>
      </section>
      <div class="intro-model-grid">
        <article class="intro-model-card green">
          <div class="model-card-tag">疫情前 2018-2019</div>
          <h4>{pre.get("target_label", "代表目标")} 贡献模型</h4>
          <p>季节周期、空间纬度、24 小时平均 PBLH、北风清洁输送和气压滞后共同构成主要解释信号。</p>
          <div class="score-row compact">{score(pre, assets["pre_covid_meteorology_metrics"])}</div>
        </article>
        <article class="intro-model-card amber">
          <div class="model-card-tag">疫情期 2020-2022</div>
          <h4>{covid.get("target_label", "代表目标")} 贡献模型</h4>
          <p>PBLH 滚动均值、南北风 V 分量、气压滞后和露点/湿度变量是该时期的重要气象信号。</p>
          <div class="score-row compact">{score(covid, assets["covid_meteorology_metrics"])}</div>
        </article>
        <article class="intro-model-card teal">
          <div class="model-card-tag">疫情后 2023+</div>
          <h4>{post.get("target_label", "代表目标")} 贡献模型</h4>
          <p>24 小时平均 PBLH、露点、通风系数、风速滚动均值和北风清洁输送在结果中更突出。</p>
          <div class="score-row compact">{score(post, assets["post_covid_meteorology_metrics"])}</div>
        </article>
      </div>
      <p class="v2-card-note">代表模型不是单纯按最高 R2 机械筛选，而是同时考虑目标形式与解释任务。SHAP bootstrap 用于检查贡献排序稳定性，相关结论仍应表述为模型解释贡献。</p>
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
        <h3>围绕时间外推和变量控制建立训练规范</h3>
        <p>训练策略以时间阻塞验证和变量准入控制为基础。项目中保留三类模型：高精度模型用于给出 PM2.5 短时预测能力上限；旧版基础气象归因模型采用较简化的 weather-only 特征，是早期对照口径；本轮 v2 过程型气象贡献模型在气象过程特征、目标形式和解释验证上做了系统重训，是当前气象贡献讨论的主要依据。</p>
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
      "高精度模型的变量边界需要先行说明：该口径既使用气象和 PBLH，也使用污染持续性和共污染物。该设计服务于预测精度评估，因此应与后续气象-only 贡献模型区分解读。",
      "blue",
    )}
    <div class="training-family-grid">
      <section class="training-family-card blue">
        <div class="family-index">目标设定</div>
        <h4>目标变量定义</h4>
        <p>当前小时模型预测 target_pm2_5，24 小时辅助模型预测 target_pm2_5_next_24h。所有样本均按城市和时间组织为 city-hour 监督学习表，保证预测任务和提前量任务在目标定义上相互区分。</p>
        <div class="family-foot">避免把当前预测和提前量预测混作同一个任务</div>
      </section>
      <section class="training-family-card green">
        <div class="family-index">变量口径</div>
        <h4>特征口径控制</h4>
        <p>extended 特征集允许使用共污染物和 PM2.5 持续性信息，服务于预测精度；meteorology 特征集主动剔除这些强预测变量，只保留气象、PBLH、稳定度、风输送和时空控制变量，服务于气象解释。</p>
        <div class="family-foot">把“预测增益”和“气象解释”分离</div>
      </section>
      <section class="training-family-card amber">
        <div class="family-index">时期设计</div>
        <h4>分时期验证</h4>
        <p>疫情前、疫情期、疫情后分别训练和测试，目的是避免疫情期人为活动变化被全时期模型平均掉，并在一致的数据切分原则下比较不同阶段的 SHAP 和残差结构。</p>
        <div class="family-foot">服务于 2020-2022 与非疫情期对比</div>
      </section>
    </div>

    <div class="training-flow">
      <div class="flow-step"><b>数据层</b><h4>数据整合</h4><p>CNEMC/quotsoft PM2.5、ERA5/CDS 气象、ERA5 PBLH 合并为 city-hour 表。</p></div>
      <div class="flow-step"><b>特征层</b><h4>特征构建</h4><p>PBLH、逆温指数、U/V 风分量、南北输送、时滞和时间周期统一进入特征矩阵。</p></div>
      <div class="flow-step"><b>验证层</b><h4>时间切分</h4><p>全时期和分时期模型都按时间后置验证、测试，避免随机切分造成信息泄漏。</p></div>
      <div class="flow-step"><b>优化层</b><h4>Optuna 调参</h4><p>以验证集 RMSE 为目标搜索 LightGBM 参数，使用 early stopping 控制过拟合。</p></div>
      <div class="flow-step"><b>解释层</b><h4>解释输出</h4><p>在测试集报告 MAE、RMSE、R2，并用 SHAP 与残差分析支撑气象贡献讨论。</p></div>
    </div>

    <div class="training-period-grid">
      <section class="training-period-card">
        <div class="period-tag">疫情前 2018-2019</div>
        <h4>正常时期基线</h4>
        <p>基础气象归因模型 R2 {pre_met["test"]["r2"]:.3f}，用于观察常规排放背景下气象条件的解释能力。</p>
      </section>
      <section class="training-period-card">
        <div class="period-tag">疫情期 2020-2022</div>
        <h4>人为活动减弱阶段</h4>
        <p>基础气象归因模型 R2 {covid_met["test"]["r2"]:.3f}，重点比较湿度、露点、风输送和 PBLH 的权重变化。</p>
      </section>
      <section class="training-period-card">
        <div class="period-tag">疫情后 2023+</div>
        <h4>恢复期对照</h4>
        <p>基础气象归因模型 R2 {post_met["test"]["r2"]:.3f}，用于分析边界层高度和低 PBLH 标记在恢复期的贡献变化。</p>
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
        <div class="method-kicker">预处理规范</div>
        <h4>预处理与特征矩阵</h4>
        <p>数值特征使用 median imputer，并保留缺失指示列；类别特征先用众数补齐，再做 one-hot 编码。时间变量除 hour、month、dayofyear 外，还构建 sin/cos 周期特征，使 23 点与 0 点、12 月与 1 月这类相邻周期在模型空间中保持连续。</p>
      </section>
      <section class="method-panel">
        <div class="method-kicker">验证规范</div>
        <h4>时间切分原则</h4>
        <p>全时期模型使用固定时间后置验证：验证集从 2024-09-01 开始，测试集从 2024-10-01 开始。分时期模型在各自时期内部按时间顺序切为 70% 训练、15% 验证、15% 测试。这里没有使用随机切分，因为 PM2.5 具有强时间连续性，随机抽样容易造成相邻小时信息泄漏。</p>
      </section>
    </div>

    <div class="tuning-board">
      <section class="tuning-main">
        <div class="method-kicker">参数搜索</div>
        <h4>Optuna + LightGBM 调参方式</h4>
        <p>每个 trial 都在训练集拟合 LightGBM，并只用验证集 RMSE 作为优化目标。搜索过程不接触测试集，测试集只在最终模型确定后使用一次。旧版基础气象归因模型为 12 轮/时期，分时期高精度模型为 25 轮/时期；本轮新增的 v2-core 过程型气象贡献模型则统一扩展到 60 轮/候选。</p>
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
        <div class="tuning-stat"><span>基础气象归因</span><b>12 轮/时期</b></div>
        <div class="tuning-stat"><span>early stopping</span><b>120 轮</b></div>
      </section>
    </div>

    <div class="training-detail-grid">
      <section class="training-detail-card">
        <h4>最终重训逻辑</h4>
        <p>候选模型先在训练集上搜索参数，并在验证集上 early stopping。确定最佳参数后，将训练集和验证集合并，重新拟合最终模型。</p>
        <p>最终 n_estimators 依据验证阶段 best_iteration 设定，并保留约 1.08 倍余量，用于合并训练集和验证集后的最终拟合。</p>
      </section>
      <section class="training-detail-card">
        <h4>最佳参数示例</h4>
        <p>全时期高精度模型：learning_rate {float(current_params.get("learning_rate", 0)):.4f}，num_leaves {current_params.get("num_leaves", "NA")}，max_depth {current_params.get("max_depth", "NA")}，best_iteration {current.get("best_iteration", "NA")}。</p>
        <p>疫情期基础气象归因模型：learning_rate {float(covid_params.get("learning_rate", 0)):.4f}，num_leaves {covid_params.get("num_leaves", "NA")}，max_depth {covid_params.get("max_depth", "NA")}，best_iteration {covid_met.get("best_iteration", "NA")}。</p>
      </section>
      <section class="training-detail-card">
        <h4>解释与稳健性输出</h4>
        <p>每个最终模型在测试集上报告 MAE、RMSE、R2；SHAP 使用测试集抽样计算平均绝对贡献，通常最多抽取 5000 行。</p>
        <p>气象残差只表示基础气象归因模型或过程型气象贡献模型未解释部分，可辅助讨论非气象因素，不能直接等同于排放量变化。</p>
      </section>
    </div>
    """


def high_accuracy_training_html(assets: Assets) -> str:
    metadata = assets["metadata"]
    current = assets["current_metrics"]
    return f"""
    <div class="training-intro">
      <section class="training-hero">
        <div class="training-kicker">高精度训练主线</div>
        <h3>用完整污染过程信息追求 PM2.5 应用预测精度</h3>
        <p>高精度模型采用 extended 特征口径：气象与 ERA5 PBLH 之外，还加入 PM2.5 时滞、滚动均值、PM10、NO2、CO、SO2、O3、AOD、dust 等污染过程变量。这个设计服务于应用预测和性能基准，不承担纯气象贡献解释任务。</p>
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
          <p>全时期主模型使用 71 个特征，Optuna 搜索 {current.get("trials", "NA")} 轮，测试集从 {current.get("test_start", "NA")} 开始。</p>
        </section>
      </div>

    <p class="v2-card-note">训练过程保持时间顺序：验证集用于调参与 early stopping，测试集只在最终模型确定后报告一次。分时期高精度模型沿用同一特征口径，用来比较不同阶段的预测难度和特征结构。</p>
    """


def high_accuracy_tuning_html(assets: Assets) -> str:
    current = assets["current_metrics"]
    current_params = current.get("best_params", {})
    pre = assets["pre_covid_high_accuracy_metrics"]
    covid = assets["covid_high_accuracy_metrics"]
    post = assets["post_covid_high_accuracy_metrics"]
    return f"""
    <div class="tuning-board">
      <section class="tuning-main">
        <div class="method-kicker">调参方案</div>
        <h4>以验证集 RMSE 为目标搜索预测模型参数</h4>
        <p>高精度模型采用 Optuna 搜索 LightGBM 参数，每个 trial 只依据验证集 RMSE 评价，并用 early stopping 监测收敛。该组模型是既有预测基准，不属于本轮 9 套 v2-core 气象贡献模型重训对象；这里保留调参方案，是为了说明测试集指标如何产生。</p>
        <div class="param-grid">
          <div class="param-row"><b>learning_rate</b><span>0.01 - 0.08，log 搜索，控制每棵树对最终预测的贡献。</span></div>
          <div class="param-row"><b>num_leaves</b><span>31 - 255，控制叶节点数量和非线性表达能力。</span></div>
          <div class="param-row"><b>max_depth</b><span>5 - 14，限制单棵树深度，降低过拟合风险。</span></div>
          <div class="param-row"><b>min_child_samples</b><span>10 - 160，限制叶节点最小样本量，减少小样本局部拟合。</span></div>
          <div class="param-row"><b>subsample / colsample</b><span>subsample 0.65 - 1.00；colsample_bytree 0.65 - 1.00，用行列采样提升泛化能力。</span></div>
          <div class="param-row"><b>reg_alpha / reg_lambda</b><span>L1: 1e-4 - 10；L2: 1e-4 - 30，抑制过大的叶节点权重。</span></div>
          <div class="param-row"><b>min_split_gain</b><span>0 - 0.25，控制节点继续分裂所需的最小收益。</span></div>
          <div class="param-row"><b>early stopping</b><span>验证集 120 轮无改善即停止，并记录 best_iteration。</span></div>
        </div>
      </section>
      <section class="tuning-side">
        <h4>搜索轮数与收尾规则</h4>
        <div class="tuning-stat"><span>全时期搜索</span><b>{current.get("trials", "NA")} 轮</b></div>
        <div class="tuning-stat"><span>疫情前搜索</span><b>{pre.get("trials", "NA")} 轮</b></div>
        <div class="tuning-stat"><span>疫情期搜索</span><b>{covid.get("trials", "NA")} 轮</b></div>
        <div class="tuning-stat"><span>疫情后搜索</span><b>{post.get("trials", "NA")} 轮</b></div>
        <div class="tuning-stat"><span>early stopping</span><b>120 轮</b></div>
        <div class="tuning-stat"><span>全时期最佳迭代</span><b>{current.get("best_iteration", "NA")}</b></div>
      </section>
    </div>

    <p class="v2-card-note">调参结束后，将训练集和验证集合并，用最佳参数重新训练最终模型。全时期高精度模型的 learning_rate={float(current_params.get("learning_rate", 0)):.4f}，num_leaves={current_params.get("num_leaves", "NA")}，max_depth={current_params.get("max_depth", "NA")}，best_iteration={current.get("best_iteration", "NA")}。</p>
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
        <div class="training-kicker">过程型气象贡献模型训练策略</div>
        <h3>从 city-hour 样本到分时期气象贡献判断的研究路径</h3>
        <p>本轮 v2 过程型气象贡献模型先按疫情前、疫情期、疫情后三个时期拆分样本，再排除污染历史和共污染物，生成气象时滞、滚动、累计和复合扩散特征。每个时期分别训练 raw、log1p、anomaly 三种目标形式，并用统一测试集评价与 SHAP bootstrap 检查解释稳定性。旧版基础气象归因模型仅作为方法对照保留。</p>
        <div class="training-chip-row">
          <span>{metadata.get("start_time", "2018+")} 至 {metadata.get("end_time", "2026")}</span>
          <span>3 个时期独立训练</span>
          <span>{candidate_count} 套候选实验</span>
          <span>每套候选 60 轮 Optuna trial</span>
          <span>{feature_count} 个气象与时空特征</span>
        </div>
      </section>
      <section class="training-score-card">
        <div class="training-score-label">实验矩阵</div>
        <h4>3 个时期 x 3 种目标形式</h4>
        <div class="tuning-stat"><span>时期分组</span><b>疫情前 / 疫情期 / 疫情后</b></div>
        <div class="tuning-stat"><span>目标形式</span><b>raw / log1p / anomaly</b></div>
        <div class="tuning-stat"><span>本轮重训</span><b>9 套过程型候选</b></div>
        <div class="tuning-stat"><span>最终报告</span><b>{best_count} 套时期代表</b></div>
        <p>每套候选模型均使用 60 轮 Optuna trial。高精度预测模型、基础气象归因模型和 24 小时辅助模型不属于本轮重训对象。</p>
      </section>
    </div>

    <div class="training-detail-grid">
      <section class="training-detail-card">
        <h4>变量准入</h4>
        <p>PM2.5 时滞、PM2.5 滚动均值、PM10、NO2、CO、SO2、O3、AOD、dust 和目标派生列全部剔除，避免污染持续性和共污染物替代气象解释。</p>
      </section>
      <section class="training-detail-card">
        <h4>气象过程表达</h4>
        <p>保留气温、露点、湿度、气压、降水、云量、风速、U/V 风、PBLH、稳定度、南北输送、城市空间和时间周期变量，并构造低 PBLH、弱风、高湿、通风系数等过程特征。</p>
      </section>
      <section class="training-detail-card">
        <h4>时间验证</h4>
        <p>每个时期内部按时间顺序划分训练、验证、测试，不使用随机切分。验证集用于调参和 early stopping，测试集只用于最终报告。</p>
      </section>
    </div>

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
    <p class="v2-card-note">上述设置的目的，是在统一 weather-only 约束下比较时期差异和目标尺度差异，而不是追求由污染自相关带来的最高预测分数。</p>
    """


def meteorology_tuning_html(assets: Assets) -> str:
    pre_old = assets["pre_covid_meteorology_metrics"]
    covid_old = assets["covid_meteorology_metrics"]
    post_old = assets["post_covid_meteorology_metrics"]
    v2 = assets.get("meteorology_v2") or {}
    summary_rows = v2.get("summary", [])
    candidate_count = len(summary_rows) or 9
    trials = int((summary_rows[0].get("trials") if summary_rows else 60) or 60)
    return f"""
    <div class="tuning-board">
      <section class="tuning-main">
        <div class="method-kicker">调参方案</div>
        <h4>同一搜索空间下训练 9 套 weather-only 候选模型</h4>
        <p>每个时期分别训练 raw、log1p、anomaly 三类目标，形成 {candidate_count} 套候选模型。每套候选均使用 {trials} 轮 Optuna trial，只在训练集拟合，并以验证集 RMSE 选择参数；测试集保留到最终比较阶段。候选模型统一还原到 PM2.5 浓度尺度后报告 R2、RMSE、MAE 和 Bias。</p>
        <div class="param-grid">
          <div class="param-row"><b>objective</b><span>以验证集 RMSE 作为主优化指标，保证 raw、log1p、anomaly 最终可在浓度尺度比较。</span></div>
          <div class="param-row"><b>learning_rate</b><span>0.015 - 0.07，log 搜索，用较稳的学习率约束气象-only 模型。</span></div>
          <div class="param-row"><b>num_leaves</b><span>31 - 191，控制树模型非线性复杂度，低于高精度模型上限以降低过拟合。</span></div>
          <div class="param-row"><b>max_depth</b><span>5 - 13，限制单棵树深度，使气象解释更稳健。</span></div>
          <div class="param-row"><b>min_child_samples</b><span>20 - 220，提高叶节点样本门槛，减少特殊天气小时的噪声拟合。</span></div>
          <div class="param-row"><b>subsample / colsample</b><span>subsample 0.72 - 1.00；colsample_bytree 0.68 - 1.00，用采样增强泛化。</span></div>
          <div class="param-row"><b>reg_alpha / reg_lambda</b><span>L1: 1e-4 - 8；L2: 1e-3 - 35，用正则项约束叶节点权重。</span></div>
          <div class="param-row"><b>min_split_gain</b><span>0 - 0.18，限制低收益分裂，降低过拟合。</span></div>
          <div class="param-row"><b>early stopping</b><span>验证集 100 轮无改善即停止，best_iteration 进入最终重训。</span></div>
        </div>
      </section>
      <section class="tuning-side">
        <h4>候选矩阵与对照边界</h4>
        <div class="tuning-stat"><span>本轮重训</span><b>{candidate_count} 套候选</b></div>
        <div class="tuning-stat"><span>每套 trial</span><b>{trials} 轮</b></div>
        <div class="tuning-stat"><span>early stopping</span><b>100 轮</b></div>
        <div class="tuning-stat"><span>疫情前基础模型 R2</span><b>{pre_old["test"]["r2"]:.3f}</b></div>
        <div class="tuning-stat"><span>疫情期基础模型 R2</span><b>{covid_old["test"]["r2"]:.3f}</b></div>
        <div class="tuning-stat"><span>疫情后基础模型 R2</span><b>{post_old["test"]["r2"]:.3f}</b></div>
      </section>
    </div>

    <div class="training-detail-grid">
      <section class="training-detail-card">
        <h4>代表模型选择</h4>
        <p>每个时期在 raw、log1p 和 anomaly 三种目标中选择一套研究代表模型。选择时不仅看测试 R2，也考虑目标形式是否符合该时期的解释任务，例如疫情后异常目标更适合解释相对本地气候态基准的气象扰动。</p>
      </section>
      <section class="training-detail-card">
        <h4>最终评估口径</h4>
        <p>每套候选模型最终统一回到 PM2.5 浓度尺度报告 R2、RMSE、MAE 和 Bias。测试集只用于最终报告，不参与 trial 选择、early stopping 或代表模型调参。</p>
      </section>
      <section class="training-detail-card">
        <h4>解释与稳健性输出</h4>
        <p>最终模型输出 SHAP bootstrap、条件误差、天气型误差、分城市指标、PDP-like 响应和 ALE 曲线。相关结果用于支撑气象贡献讨论，不表述为严格因果效应。</p>
      </section>
    </div>
    """


def high_accuracy_training_rows(assets: Assets) -> pd.DataFrame:
    rows = training_strategy_rows(assets)
    return rows[rows["类型"].isin(["预测主模型", "分时期预测", "提前量预测"])].reset_index(drop=True)


def meteorology_legacy_training_rows(assets: Assets) -> pd.DataFrame:
    rows = training_strategy_rows(assets)
    return rows[rows["类型"] == "基础气象归因"].reset_index(drop=True)


def meteorology_v2_training_rows(assets: Assets) -> pd.DataFrame:
    table = meteorology_v2_summary_table(assets)
    if table.empty:
        return table
    table = table.copy()
    table.insert(0, "模型口径", "过程型气象贡献")
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
            metric_row("疫情前基础气象归因", assets["pre_covid_meteorology_metrics"], "2018-2019 气象-only 原值模型"),
            metric_row("疫情期基础气象归因", assets["covid_meteorology_metrics"], "2020-2022 气象-only 原值模型"),
            metric_row("疫情后基础气象归因", assets["post_covid_meteorology_metrics"], "2023+ 气象-only 原值模型"),
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
                "调参轮数": int(row.get("trials") or 60),
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
                "代表目标": row["target_label"],
                "R2": round(float(row["r2"]), 3),
                "RMSE": round(float(row["rmse"]), 3),
                "MAE": round(float(row["mae"]), 3),
                "Bias": round(float(row["bias"]), 3),
                "训练/验证/测试": f"{int(row['train_rows']):,} / {int(row['valid_rows']):,} / {int(row['test_rows']):,}",
                "特征数": int(row["feature_count"]),
                "最佳迭代": int(row["best_iteration"]),
                "调参轮数": int(row.get("trials") or 60),
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
                "基础气象归因 R2": round(float(old.get("r2", 0)), 3),
                "过程型气象贡献 R2": row["R2"],
                "R2 差值": round(float(row["R2"]) - float(old.get("r2", 0)), 3),
                "基础气象归因 RMSE": round(float(old.get("rmse", 0)), 2),
                "过程型气象贡献 RMSE": round(float(row["RMSE"]), 2),
                "过程型代表目标": row["代表目标"],
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
        title=f"{model['period_label']}过程型气象贡献 Top SHAP",
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
        title="过程型气象贡献模型三目标 R2 对比",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=20), yaxis_range=[0, 1.0], xaxis_title="")
    return fig


def research_frame(assets: Assets, key: str) -> pd.DataFrame:
    upgrade = assets.get("research_upgrade") or {}
    data = upgrade.get(key, [])
    return pd.DataFrame(data)


def research_upgrade_status_html(assets: Assets) -> str:
    upgrade = assets.get("research_upgrade") or {}
    findings = upgrade.get("key_findings", [])
    evidence_items = "".join(
        (
            '<section class="research-evidence-item">'
            '<div class="research-evidence-mark"></div>'
            f"<p>{finding}</p>"
            "</section>"
        )
        for finding in findings
    )
    return (
        '<div class="research-upgrade-hero">'
        '<section class="research-upgrade-main">'
        '<div class="research-kicker">研究验证体系</div>'
        "<h3>从预测模型评估扩展为大气环境机器学习研究框架</h3>"
        "<p>研究验证不再仅以总体 R2 作为评价依据，而是在过程型气象贡献模型基础上引入留城市验证、透明样条对照、天气型机制分层、典型气象条件误差和 SHAP 稳定性分析。该体系用于检验模型是否能够外推到未参与训练的城市，气象解释是否具有稳定性，以及关键污染天气型是否与边界层、通风、湿度和区域输送等大气环境机制相一致。</p>"
        '<div class="intro-chip-row">'
        "<span>留城市验证</span>"
        "<span>透明样条对照</span>"
        "<span>天气型机制</span>"
        "<span>SHAP bootstrap 稳定性</span>"
        "</div>"
        "</section>"
        '<section class="research-upgrade-side">'
        '<div class="research-kicker muted">解释口径</div>'
        "<h4>模型解释贡献，不等同于严格因果效应</h4>"
        "<p>SHAP、偏依赖响应和透明样条响应曲线用于说明模型在给定特征空间下如何使用气象信息。相关结果应表述为模型解释贡献，并与边界层、通风、湿度、输送和天气型机制共同论证，避免将其直接等同于严格因果效应。</p>"
        "</section>"
        "</div>"
        '<section class="research-evidence-panel">'
        "<h4>验证结果概览</h4>"
        f'<div class="research-evidence-list">{evidence_items}</div>'
        "</section>"
    )


def _first_matching(rows: list[dict[str, Any]], **criteria: Any) -> dict[str, Any]:
    for row in rows:
        if all(row.get(key) == value for key, value in criteria.items()):
            return row
    return {}


def _fmt_number(value: Any, digits: int = 3, fallback: str = "NA") -> str:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return fallback
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return fallback


def _fmt_pm25(value: Any, fallback: str = "NA") -> str:
    try:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return fallback
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return fallback


def _top_feature_labels(model: dict[str, Any], n: int = 5) -> str:
    top = model.get("top_shap", [])[:n]
    labels = [feature_label(row.get("feature", "")) for row in top]
    return "、".join(labels) if labels else "NA"


def research_conclusion_html(assets: Assets) -> str:
    v2 = assets.get("meteorology_v2") or {}
    upgrade = assets.get("research_upgrade") or {}
    best_summary = v2.get("best_summary", [])
    best_models = v2.get("best_models", [])
    leave_city = upgrade.get("leave_city_summary", [])
    weather_types = upgrade.get("weather_type_mechanism", [])
    weather_highlights = upgrade.get("weather_type_highlights", [])
    shap_similarity = upgrade.get("shap_rank_similarity", [])
    gam_like = upgrade.get("gam_like_metrics", [])

    pre = _first_matching(best_summary, period="pre_covid_2018_2019")
    covid = _first_matching(best_summary, period="covid_2020_2022")
    post = _first_matching(best_summary, period="post_covid_2023_plus")
    pre_model = _first_matching(best_models, period="pre_covid_2018_2019")
    covid_model = _first_matching(best_models, period="covid_2020_2022")
    post_model = _first_matching(best_models, period="post_covid_2023_plus")
    pre_leave = _first_matching(leave_city, period_label="疫情前")
    covid_leave = _first_matching(leave_city, period_label="疫情期")
    post_leave = _first_matching(leave_city, period_label="疫情后")
    pre_highlight = _first_matching(weather_highlights, period_label="疫情前")
    covid_highlight = _first_matching(weather_highlights, period_label="疫情期")
    post_highlight = _first_matching(weather_highlights, period_label="疫情后")
    pre_gam = _first_matching(gam_like, period_label="疫情前")
    covid_gam = _first_matching(gam_like, period_label="疫情期")
    post_gam = _first_matching(gam_like, period_label="疫情后")
    pre_covid_sim = _first_matching(shap_similarity, left_period="疫情前", right_period="疫情期")
    pre_post_sim = _first_matching(shap_similarity, left_period="疫情前", right_period="疫情后")
    covid_post_sim = _first_matching(shap_similarity, left_period="疫情期", right_period="疫情后")

    north_pre = _first_matching(weather_types, period_label="疫情前", weather_type_label="强北风清洁输送型")
    north_covid = _first_matching(weather_types, period_label="疫情期", weather_type_label="强北风清洁输送型")
    north_post = _first_matching(weather_types, period_label="疫情后", weather_type_label="强北风清洁输送型")
    south_post = _first_matching(weather_types, period_label="疫情后", weather_type_label="强南风输送型")

    headline_stats = [
        ("气象-only 解释力", f"R2 {_fmt_number(pre.get('r2'))} / {_fmt_number(covid.get('r2'))} / {_fmt_number(post.get('r2'))}", "疫情前、疫情期、疫情后代表模型"),
        ("留城市泛化", f"平均 R2 {_fmt_number(pre_leave.get('mean_r2'))} / {_fmt_number(covid_leave.get('mean_r2'))}", "疫情前与疫情期具有较稳定空间外推能力"),
        ("静稳高湿污染型", f"中位数 {_fmt_pm25(pre_highlight.get('highest_pm25_median'))} / {_fmt_pm25(covid_highlight.get('highest_pm25_median'))} / {_fmt_pm25(post_highlight.get('highest_pm25_median'))}", "三个时期最高 PM2.5 天气型均为极低边界层静稳高湿型"),
        ("SHAP 结构差异", f"Jaccard {_fmt_number(covid_post_sim.get('top20_jaccard'))}", "疫情期与疫情后 Top20 贡献结构差异明显"),
    ]
    stat_html = "".join(
        (
            '<div class="research-stat-tile">'
            f"<span>{label}</span>"
            f"<b>{value}</b>"
            f"<p>{note}</p>"
            "</div>"
        )
        for label, value, note in headline_stats
    )

    return (
        '<div class="research-brief">'
        '<section class="research-brief-head">'
        '<div class="research-kicker">阶段性研究判断</div>'
        "<h3>京津冀 PM2.5 的气象影响主要体现为边界层通风、湿度静稳和南北向输送的共同调制</h3>"
        "<p>当前结果可概括为一组阶段性研究证据：高精度模型给出污染浓度短时预测的性能上限；气象-only 模型在剔除 PM2.5 滞后、滚动均值和共污染物后，用于估计气象背景对浓度波动的相对独立解释力。研究重点并非单一 R2 数值，而是不同时期的气象贡献结构及其大气环境机制。</p>"
        '<div class="research-stat-grid">'
        f"{stat_html}"
        "</div>"
        "</section>"
        '<div class="research-brief-body">'
        '<section class="research-note">'
        "<h4>气象背景的独立解释力</h4>"
        f"<p>过程型气象-only 模型在疫情前、疫情期和疫情后测试集上的 R2 分别为 {_fmt_number(pre.get('r2'))}、{_fmt_number(covid.get('r2'))} 和 {_fmt_number(post.get('r2'))}；透明样条对照模型仍保留 {_fmt_number(pre_gam.get('r2'))}、{_fmt_number(covid_gam.get('r2'))} 和 {_fmt_number(post_gam.get('r2'))} 的解释度。由于这组模型不使用污染时滞和共污染物，上述结果说明气象、PBLH、风输送和时空背景本身能够解释相当比例的 PM2.5 波动。</p>"
        "</section>"
        '<section class="research-note">'
        "<h4>分时期贡献结构</h4>"
        f"<p>疫情前的前列变量集中于 {_top_feature_labels(pre_model, 5)}；疫情期集中于 {_top_feature_labels(covid_model, 5)}；疫情后集中于 {_top_feature_labels(post_model, 5)}。这一变化表明，疫情前后气象贡献结构并非简单平移，恢复期的 PBLH、露点、通风系数和风速过程更突出，提示扩散条件约束有所强化。</p>"
        "</section>"
        '<section class="research-note">'
        "<h4>典型天气背景</h4>"
        f"<p>三个时期 PM2.5 中位数最高的天气型均为极低边界层静稳高湿型，对应中位浓度约 {_fmt_pm25(pre_highlight.get('highest_pm25_median'))}、{_fmt_pm25(covid_highlight.get('highest_pm25_median'))} 和 {_fmt_pm25(post_highlight.get('highest_pm25_median'))} ug/m3。该天气型同时具有低 PBLH、弱风、高湿和低通风特征，符合污染累积天气背景的物理认识。</p>"
        "</section>"
        '<section class="research-note">'
        "<h4>输送方向与空间泛化</h4>"
        f"<p>强北风清洁输送型在三个时期均对应较低 PM2.5 中位数，约 {_fmt_pm25(north_pre.get('pm25_median'))}、{_fmt_pm25(north_covid.get('pm25_median'))} 和 {_fmt_pm25(north_post.get('pm25_median'))} ug/m3；疫情后强南风输送型频率约 {_fmt_number(south_post.get('frequency_pct'), 1)}%。留城市验证中，疫情前和疫情期平均 R2 分别为 {_fmt_number(pre_leave.get('mean_r2'))} 和 {_fmt_number(covid_leave.get('mean_r2'))}，说明模型具有一定区域泛化能力；疫情后异常量目标的留城市表现下降，提示同城气候态基准存在空间外推边界。</p>"
        "</section>"
        "</div>"
        '<div class="research-boundary-panel">'
        "<h4>解释边界</h4>"
        f"<p>SHAP 排名稳定性支持分时期差异判断：疫情前与疫情期 Top20 Jaccard 为 {_fmt_number(pre_covid_sim.get('top20_jaccard'))}，疫情期与疫情后为 {_fmt_number(covid_post_sim.get('top20_jaccard'))}。但 SHAP 反映的是模型在给定特征空间下对变量的使用方式，不等同于严格因果效应。跨时期比较还应同时考虑 2023+ PM2.5 数据源与 2018-2022 不完全一致、排放活动缺少直接清单约束，以及异常量目标对城市历史基准的依赖。</p>"
        "</div>"
        "</div>"
    )


def leave_city_summary_table(assets: Assets) -> pd.DataFrame:
    frame = research_frame(assets, "leave_city_summary")
    if frame.empty:
        return frame
    table = frame[
        [
            "period_label",
            "target_label",
            "cities",
            "mean_r2",
            "median_r2",
            "min_r2",
            "max_r2",
            "mean_rmse",
            "mean_mae",
            "mean_bias",
        ]
    ].copy()
    table.columns = ["时期", "目标形式", "城市数", "平均 R2", "中位 R2", "最低 R2", "最高 R2", "平均 RMSE", "平均 MAE", "平均 Bias"]
    for column in ["平均 R2", "中位 R2", "最低 R2", "最高 R2", "平均 RMSE", "平均 MAE", "平均 Bias"]:
        table[column] = table[column].astype(float).round(3)
    return table


def leave_city_r2_chart(assets: Assets) -> go.Figure:
    table = leave_city_summary_table(assets)
    long = table.melt(id_vars=["时期"], value_vars=["平均 R2", "中位 R2"], var_name="指标", value_name="R2")
    fig = px.bar(long, x="时期", y="R2", color="指标", barmode="group", text="R2", title="留城市验证：平均与中位 R2")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=20), yaxis_range=[0, 0.95], xaxis_title="")
    return fig


def leave_city_detail_chart(assets: Assets) -> go.Figure:
    frame = research_frame(assets, "leave_city_metrics")
    if frame.empty:
        return go.Figure()
    frame = frame.copy()
    frame["R2"] = frame["r2"].astype(float)
    fig = px.bar(
        frame,
        x="holdout_city",
        y="R2",
        color="period_label",
        barmode="group",
        hover_data={"rmse": ":.2f", "mae": ":.2f", "bias": ":.2f"},
        title="13 城市逐一留出验证 R2",
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#64748b")
    fig.update_layout(height=410, margin=dict(l=10, r=10, t=50, b=20), yaxis_title="R2", xaxis_title="留出城市")
    return fig


def combined_extrapolation_summary_table(assets: Assets) -> pd.DataFrame:
    frame = assets.get("combined_extrapolation_summary", pd.DataFrame())
    if frame.empty:
        return frame
    table = frame[
        [
            "period_label",
            "target_label",
            "cities",
            "mean_r2",
            "median_r2",
            "min_r2",
            "max_r2",
            "mean_rmse",
            "mean_mae",
            "mean_bias",
        ]
    ].copy()
    table.columns = ["时期", "代表目标", "城市数", "平均 R2", "中位 R2", "最低 R2", "最高 R2", "平均 RMSE", "平均 MAE", "平均 Bias"]
    for column in ["平均 R2", "中位 R2", "最低 R2", "最高 R2", "平均 RMSE", "平均 MAE", "平均 Bias"]:
        table[column] = pd.to_numeric(table[column], errors="coerce").round(3)
    return table


def combined_extrapolation_summary_chart(assets: Assets) -> go.Figure:
    table = combined_extrapolation_summary_table(assets)
    if table.empty:
        return go.Figure()
    long = table.melt(id_vars=["时期"], value_vars=["平均 R2", "中位 R2"], var_name="指标", value_name="R2")
    fig = px.bar(long, x="时期", y="R2", color="指标", barmode="group", text="R2", title="组合外推验证：平均与中位 R2")
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#64748b")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=20), yaxis_range=[-0.1, 0.8], xaxis_title="")
    return fig


def combined_extrapolation_city_chart(assets: Assets) -> go.Figure:
    frame = assets.get("combined_extrapolation", pd.DataFrame())
    if frame.empty:
        return go.Figure()
    frame = frame.copy()
    frame["R2"] = pd.to_numeric(frame["r2"], errors="coerce")
    fig = px.bar(
        frame,
        x="holdout_city",
        y="R2",
        color="period_label",
        barmode="group",
        hover_data={"rmse": ":.2f", "mae": ":.2f", "bias": ":.2f", "test_rows": True},
        title="组合外推逐城市 R2",
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#64748b")
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=20), yaxis_title="R2", xaxis_title="留出城市")
    return fig


def feature_group_ablation_table(assets: Assets) -> pd.DataFrame:
    frame = assets.get("feature_group_ablation", pd.DataFrame())
    if frame.empty:
        return frame
    table = frame[
        [
            "period_label",
            "feature_group_label",
            "feature_count",
            "r2",
            "rmse",
            "delta_r2_from_previous",
            "rmse_reduction_from_previous",
        ]
    ].copy()
    table.columns = ["时期", "特征组", "特征数", "测试 R2", "测试 RMSE", "R2 增量", "RMSE 降低"]
    for column in ["测试 R2", "测试 RMSE", "R2 增量", "RMSE 降低"]:
        table[column] = pd.to_numeric(table[column], errors="coerce").round(3)
    return table


def feature_group_ablation_chart(assets: Assets) -> go.Figure:
    frame = assets.get("feature_group_ablation", pd.DataFrame())
    if frame.empty:
        return go.Figure()
    frame = frame.copy()
    frame["测试 R2"] = pd.to_numeric(frame["r2"], errors="coerce")
    frame["特征组"] = frame["feature_group_label"]
    group_order = frame["特征组"].drop_duplicates().tolist()
    fig = px.line(
        frame,
        x="特征组",
        y="测试 R2",
        color="period_label",
        markers=True,
        category_orders={"特征组": group_order},
        title="特征组消融：逐步加入气象过程信息后的 R2",
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#64748b")
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=105), xaxis_title="", yaxis_title="测试 R2")
    fig.update_xaxes(tickangle=-22)
    return fig


def feature_group_delta_chart(assets: Assets) -> go.Figure:
    frame = assets.get("feature_group_ablation", pd.DataFrame())
    if frame.empty:
        return go.Figure()
    frame = frame.copy()
    frame["R2 增量"] = pd.to_numeric(frame["delta_r2_from_previous"], errors="coerce")
    frame = frame.dropna(subset=["R2 增量"])
    if frame.empty:
        return go.Figure()
    fig = px.bar(
        frame,
        x="feature_group_label",
        y="R2 增量",
        color="period_label",
        barmode="group",
        text="R2 增量",
        title="相对上一特征组的 R2 增量",
    )
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#64748b")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=50, b=105), xaxis_title="", yaxis_title="R2 增量")
    fig.update_xaxes(tickangle=-22)
    return fig


def gam_like_compare_table(assets: Assets) -> pd.DataFrame:
    gam = research_frame(assets, "gam_like_metrics")
    v2 = meteorology_v2_best_table(assets)
    if gam.empty or v2.empty:
        return pd.DataFrame()
    v2_lookup = {row["时期"]: row for _, row in v2.iterrows()}
    rows = []
    for _, row in gam.iterrows():
        period = row["period_label"]
        v2_row = v2_lookup.get(period, {})
        rows.append(
            {
                "时期": period,
                "目标形式": row["target_label"],
                "透明对照 R2": round(float(row["r2"]), 3),
                "透明对照 RMSE": round(float(row["rmse"]), 2),
                "过程型 LightGBM R2": round(float(v2_row.get("R2", 0)), 3),
                "过程型 LightGBM RMSE": round(float(v2_row.get("RMSE", 0)), 2),
                "透明模型特征数": int(row["feature_count"]),
                "样条气象变量数": int(row["spline_feature_count"]),
            }
        )
    return pd.DataFrame(rows)


def gam_like_compare_chart(assets: Assets) -> go.Figure:
    table = gam_like_compare_table(assets)
    if table.empty:
        return go.Figure()
    long = table.melt(id_vars=["时期"], value_vars=["透明对照 R2", "过程型 LightGBM R2"], var_name="模型", value_name="R2")
    fig = px.bar(long, x="时期", y="R2", color="模型", barmode="group", text="R2", title="透明对照模型与过程型 LightGBM 性能比较")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=20), yaxis_range=[0, 0.85], xaxis_title="")
    return fig


def gam_response_chart(assets: Assets, feature: str) -> go.Figure:
    frame = research_frame(assets, "gam_like_response_profiles")
    if frame.empty:
        return go.Figure()
    part = frame[frame["feature"] == feature].copy()
    if part.empty:
        return go.Figure()
    part["气象变量"] = part["feature"].map(feature_label)
    fig = px.line(
        part,
        x="value",
        y="mean_prediction_pm25",
        color="period_label",
        markers=True,
        title=f"透明对照响应曲线：{feature_label(feature)}",
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=5))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=50, b=20), xaxis_title=feature_label(feature), yaxis_title="平均预测 PM2.5")
    return fig


def weather_type_table(assets: Assets) -> pd.DataFrame:
    frame = research_frame(assets, "weather_type_mechanism")
    if frame.empty:
        return frame
    table = frame[
        [
            "period_label",
            "weather_type_k6",
            "weather_type_label",
            "frequency_pct",
            "pm25_median",
            "pm25_p90",
            "pblh_mean",
            "wind_speed_mean",
            "relative_humidity_mean",
            "mae",
            "rmse",
            "r2",
        ]
    ].copy()
    table.columns = ["时期", "天气型", "机制名称", "频率(%)", "PM2.5 中位数", "PM2.5 P90", "平均 PBLH", "平均风速", "平均相对湿度", "MAE", "RMSE", "R2"]
    for column in ["频率(%)", "PM2.5 中位数", "PM2.5 P90", "平均 PBLH", "平均风速", "平均相对湿度", "MAE", "RMSE", "R2"]:
        table[column] = pd.to_numeric(table[column], errors="coerce").round(2)
    return table


def weather_type_frequency_chart(assets: Assets) -> go.Figure:
    frame = research_frame(assets, "weather_type_mechanism")
    if frame.empty:
        return go.Figure()
    fig = px.bar(
        frame,
        x="weather_type_label",
        y="frequency_pct",
        color="period_label",
        barmode="group",
        title="分时期天气型出现频率",
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=120), xaxis_title="", yaxis_title="频率(%)")
    fig.update_xaxes(tickangle=-28)
    return fig


def weather_type_pm25_chart(assets: Assets) -> go.Figure:
    frame = research_frame(assets, "weather_type_mechanism")
    if frame.empty:
        return go.Figure()
    fig = px.bar(
        frame,
        x="weather_type_label",
        y="pm25_median",
        color="period_label",
        barmode="group",
        title="不同天气型 PM2.5 中位数",
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=120), xaxis_title="", yaxis_title="PM2.5 ug/m3")
    fig.update_xaxes(tickangle=-28)
    return fig


def condition_metrics_table(assets: Assets) -> pd.DataFrame:
    frame = research_frame(assets, "critical_condition_metrics")
    if frame.empty:
        return frame
    table = frame[["period_label", "condition_label", "rows", "observed_mean", "predicted_mean", "mae", "rmse", "r2", "bias"]].copy()
    table.columns = ["时期", "气象条件", "样本数", "实测均值", "预测均值", "MAE", "RMSE", "R2", "Bias"]
    for column in ["实测均值", "预测均值", "MAE", "RMSE", "R2", "Bias"]:
        table[column] = pd.to_numeric(table[column], errors="coerce").round(2)
    return table


def condition_rmse_chart(assets: Assets) -> go.Figure:
    table = condition_metrics_table(assets)
    if table.empty:
        return go.Figure()
    core = table[table["气象条件"].isin(["低边界层条件", "弱风条件", "高湿条件", "低通风条件", "南风输送条件", "北风清洁输送条件"])].copy()
    if core.empty:
        core = table
    fig = px.bar(core, x="气象条件", y="RMSE", color="时期", barmode="group", text="RMSE", title="关键气象条件下模型误差")
    fig.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=50, b=90), xaxis_title="", yaxis_title="RMSE ug/m3")
    fig.update_xaxes(tickangle=-22)
    return fig


def shap_similarity_table(assets: Assets) -> pd.DataFrame:
    frame = research_frame(assets, "shap_rank_similarity")
    if frame.empty:
        return frame
    table = frame.copy()
    table["时期对比"] = table["left_period"] + " vs " + table["right_period"]
    table = table[["时期对比", "common_features", "top20_overlap", "top20_jaccard", "spearman_rank_corr_all_features"]]
    table.columns = ["时期对比", "共同特征数", "Top20 交集", "Top20 Jaccard", "全特征 Spearman"]
    table["Top20 Jaccard"] = table["Top20 Jaccard"].astype(float).round(3)
    table["全特征 Spearman"] = table["全特征 Spearman"].astype(float).round(3)
    return table


def shap_similarity_chart(assets: Assets) -> go.Figure:
    table = shap_similarity_table(assets)
    if table.empty:
        return go.Figure()
    long = table.melt(id_vars=["时期对比"], value_vars=["Top20 Jaccard", "全特征 Spearman"], var_name="稳定性指标", value_name="数值")
    fig = px.bar(long, x="时期对比", y="数值", color="稳定性指标", barmode="group", text="数值", title="分时期 SHAP 排名相似度")
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=20), yaxis_range=[0, 1.0], xaxis_title="")
    return fig


def shap_stability_period_table(assets: Assets, period_label: str, n: int = 12) -> pd.DataFrame:
    frame = research_frame(assets, "shap_stability_top20")
    if frame.empty:
        return frame
    part = frame[frame["period_label"] == period_label].head(n).copy()
    table = part[["feature_label", "mean_abs_shap", "bootstrap_ci_low", "bootstrap_ci_high", "relative_ci_width", "rank"]].copy()
    table.columns = ["特征", "平均绝对 SHAP", "95% CI 下限", "95% CI 上限", "相对 CI 宽度", "排名"]
    for column in ["平均绝对 SHAP", "95% CI 下限", "95% CI 上限", "相对 CI 宽度"]:
        table[column] = table[column].astype(float).round(4)
    return table


def shap_stability_period_chart(assets: Assets, period_label: str) -> go.Figure:
    table = shap_stability_period_table(assets, period_label, 12)
    if table.empty:
        return go.Figure()
    table = table.sort_values("平均绝对 SHAP")
    error_plus = table["95% CI 上限"] - table["平均绝对 SHAP"]
    error_minus = table["平均绝对 SHAP"] - table["95% CI 下限"]
    fig = go.Figure(
        go.Bar(
            x=table["平均绝对 SHAP"],
            y=table["特征"],
            orientation="h",
            marker_color="#0f766e",
            error_x=dict(type="data", symmetric=False, array=error_plus, arrayminus=error_minus, color="#334155"),
        )
    )
    fig.update_layout(
        title=f"{period_label} SHAP Top 特征与 bootstrap 置信区间",
        height=430,
        margin=dict(l=10, r=10, t=50, b=20),
        xaxis_title="平均绝对 SHAP",
        yaxis_title="",
    )
    return fig


def render_research_validation_section(assets: Assets) -> None:
    if not assets.get("research_upgrade"):
        st.warning("尚未检索到研究验证结果文件，当前仅保留模型训练与预测结果。")
        return

    st.markdown(research_conclusion_html(assets), unsafe_allow_html=True)
    st.markdown(research_upgrade_status_html(assets), unsafe_allow_html=True)
    validation_tab, extension_tab, transparent_tab, weather_type_tab, shap_tab, condition_tab = st.tabs(
        ["空间泛化验证", "组合外推与消融", "透明解释对照", "天气型机制", "SHAP 稳定性", "条件误差"]
    )

    with validation_tab:
        st.markdown(
            """
            <div class="explain-band green">
              <h4>留城市验证的研究含义</h4>
              <p>留城市验证每次完整移除一个城市，只用其余 12 个城市训练，再预测被留出城市。它检验的是模型能否学习区域气象规律，而不是记住某个城市的固定浓度水平。</p>
              <p>疫情前与疫情期平均 R2 均接近 0.69，说明过程型气象-only 模型具有一定空间泛化能力；疫情后气候态异常目标明显较弱，提示该目标对城市历史基准依赖较强，空间外推时需要单独说明边界条件。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        lv_col_a, lv_col_b = st.columns([0.45, 0.55])
        lv_col_a.plotly_chart(leave_city_r2_chart(assets), use_container_width=True)
        lv_col_b.plotly_chart(leave_city_detail_chart(assets), use_container_width=True)
        st.dataframe(leave_city_summary_table(assets), use_container_width=True, hide_index=True)

    with extension_tab:
        st.markdown(
            """
            <div class="explain-band amber">
              <h4>补充验证的研究含义</h4>
              <p>组合外推验证同时留出城市和后段时间，用于检查模型在更严格条件下是否仍保留区域气象规律。该结果不替代主模型的时间阻塞测试，而是用于说明空间外推边界。</p>
              <p>特征组消融按时间/城市基线、基础气象、PBLH、风场输送、滞后滚动与静稳过程逐步加入特征，用于回答模型解释力是否只是来自季节和城市背景。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        ext_col_a, ext_col_b = st.columns([0.45, 0.55])
        ext_col_a.plotly_chart(combined_extrapolation_summary_chart(assets), use_container_width=True)
        ext_col_b.plotly_chart(combined_extrapolation_city_chart(assets), use_container_width=True)
        st.dataframe(combined_extrapolation_summary_table(assets), use_container_width=True, hide_index=True)
        st.caption("组合外推结果显示，疫情前和疫情期仍保留中等解释力；疫情后异常目标的空间外推边界更明显。")

        ab_col_a, ab_col_b = st.columns([0.52, 0.48])
        ab_col_a.plotly_chart(feature_group_ablation_chart(assets), use_container_width=True)
        ab_col_b.plotly_chart(feature_group_delta_chart(assets), use_container_width=True)
        st.dataframe(feature_group_ablation_table(assets), use_container_width=True, hide_index=True)
        st.caption(
            "消融结果显示，滞后滚动与静稳过程在三个时期均提供明确增量；天气型标签在完整过程特征已经存在时增量较小，更适合作为机制解释和分层讨论工具。"
        )

    with transparent_tab:
        st.markdown(
            """
            <div class="explain-band blue">
              <h4>透明模型对照的研究含义</h4>
              <p>透明样条对照模型使用关键气象变量的平滑基函数与 Ridge 回归，牺牲一部分精度来换取更清晰的响应曲线。该模型用于检验主要气象变量是否能在较透明的模型结构中保留一致的响应方向。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        gam_col_a, gam_col_b = st.columns([0.48, 0.52])
        gam_col_a.plotly_chart(gam_like_compare_chart(assets), use_container_width=True)
        with gam_col_b:
            response_feature = st.selectbox(
                "响应曲线变量",
                [
                    "boundary_layer_height",
                    "ventilation_coefficient",
                    "relative_humidity_2m",
                    "wind_speed_10m",
                    "pressure_msl",
                    "precipitation",
                ],
                format_func=feature_label,
            )
            st.plotly_chart(gam_response_chart(assets, response_feature), use_container_width=True)
        st.dataframe(gam_like_compare_table(assets), use_container_width=True, hide_index=True)
        st.caption("透明对照模型的精度通常低于过程型 LightGBM；其作用在于提供可解释响应形态，而不是追求最高 R2。")

    with weather_type_tab:
        st.markdown(
            """
            <div class="explain-band green">
              <h4>天气型机制解释</h4>
              <p>天气型聚类只使用气象变量，不使用 PM2.5。它把低边界层、高湿、南北风输送、强扩散和降水清除等过程组织成可解释的天气背景，使模型结果能落回大气环境机制。</p>
              <p>极低边界层静稳高湿型在三个时期中均对应较高 PM2.5，是低 PBLH、弱风、高湿和低通风共同造成污染累积的主要证据；疫情后强南风输送型频率上升，可作为恢复期区域输送增强的机制线索。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        wt_col_a, wt_col_b = st.columns(2)
        wt_col_a.plotly_chart(weather_type_frequency_chart(assets), use_container_width=True)
        wt_col_b.plotly_chart(weather_type_pm25_chart(assets), use_container_width=True)
        with st.expander("天气型机制明细", expanded=False):
            st.dataframe(weather_type_table(assets), use_container_width=True, hide_index=True)

    with shap_tab:
        st.markdown(
            """
            <div class="explain-band amber">
              <h4>SHAP 稳定性解释</h4>
              <p>单次 SHAP 排名容易受抽样扰动影响。本项目对测试样本重复抽样，计算 Top 特征均值、标准差和 95% 置信区间，并比较三时期特征排序相似度。相关结果应表述为模型解释贡献，而不是严格因果效应。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        shap_col_a, shap_col_b = st.columns([0.46, 0.54])
        shap_col_a.plotly_chart(shap_similarity_chart(assets), use_container_width=True)
        with shap_col_b:
            shap_period = st.selectbox("SHAP 时期", ["疫情前", "疫情期", "疫情后"], index=1)
            st.plotly_chart(shap_stability_period_chart(assets, shap_period), use_container_width=True)
        st.dataframe(shap_similarity_table(assets), use_container_width=True, hide_index=True)
        with st.expander("所选时期 Top SHAP 置信区间", expanded=False):
            st.dataframe(shap_stability_period_table(assets, shap_period, 20), use_container_width=True, hide_index=True)

    with condition_tab:
        st.markdown(
            """
            <div class="explain-band blue">
              <h4>典型气象条件误差分析</h4>
              <p>气象贡献模型的评估不应只依赖总体 R2。低边界层、弱风、高湿、低通风、南风输送和北风清洁输送等条件是 PM2.5 污染过程的关键背景，在这些条件下单独报告误差可以检验模型是否覆盖主要污染气象机制。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(condition_rmse_chart(assets), use_container_width=True)
        with st.expander("关键气象条件误差明细", expanded=False):
            st.dataframe(condition_metrics_table(assets), use_container_width=True, hide_index=True)


def v2_period_narrative(model: dict[str, Any]) -> str:
    period = model["period"]
    if period == "pre_covid_2018_2019":
        return (
            "疫情前代表模型采用 log1p(PM2.5) 目标，说明在正常排放背景下，降低重污染极端值对训练的干扰后，"
            "气象变量能够更稳定地解释浓度变化。SHAP 前列包含年内周期、纬度、24小时平均 PBLH、"
            "北风清洁输送和72小时气压滞后，反映出季节背景、城市南北空间差异、边界层扩散和冷空气过程共同控制污染累积。"
        )
    if period == "covid_2020_2022":
        return (
            "疫情期代表模型采用 PM2.5 原值目标，R2 在三种目标形式中最高，说明人为活动减弱并不意味着气象解释力消失。"
            "24小时平均 PBLH、24/48小时南北风 V 分量、72小时气压滞后与露点/湿度相关变量共同进入前列，"
            "表明静稳扩散条件、区域输送和湿度过程仍然是疫情期 PM2.5 波动的重要气象驱动。"
        )
    return (
        "疫情后代表模型采用同城同月同小时气候态异常目标，说明恢复期更适合解释 PM2.5 相对本地气候背景的偏离。"
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
        paragraphs = [
            f"{spec['role']} {spec['question']}",
            f"该模型的特征口径为：{spec['features']}。{spec['academic_use']}",
        ]
        if spec.get("result_reading"):
            paragraphs.append(spec["result_reading"])
        if spec.get("limitation"):
            paragraphs.append(spec["limitation"])
        st.markdown(
            '<div class="model-card-reading">'
            + "".join(f"<p>{paragraph}</p>" for paragraph in paragraphs)
            + "</div>",
            unsafe_allow_html=True,
        )
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
            "role": "综合预测基准模型，用于 2018+ 全时期当前小时 PM2.5 估计。",
            "question": "评估当前气象、PM2.5 历史和共污染物背景约束下的 PM2.5 当前小时预测能力。",
            "features": "气象、ERA5 PBLH、逆温、风输送、PM2.5 时滞、滚动均值、PM10、CO、NO2、SO2、O3、AOD、dust。",
            "academic_use": "用于评估机器学习框架的预测能力，不直接作为气象因子独立贡献的唯一依据。",
            "result_reading": "该模型精度最高，说明在跨年份、跨城市场景下，LightGBM 能有效捕捉污染持续性、共污染物协同变化和气象扩散条件之间的非线性关系。它构成当前综合预测口径的主模型。",
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
            "academic_use": "作为疫情前预测精度上限和气象贡献模型的对照。",
            "result_reading": "它提供了疫情前正常排放背景下的预测上限。若该模型显著优于疫情前气象模型，说明 PM2.5 自身持续性和共污染物背景对短时预测有很强贡献。",
            "limitation": "该模型仍包含污染历史和共污染物，不宜直接用于判断 PBLH、湿度、风和逆温的独立贡献强弱。",
            "note": "该模型 SHAP 前列主要为 PM2.5 滚动均值、时滞和 PM10，说明预测能力主要来自污染持续性。",
        },
        {
            "title": "疫情前基础气象归因模型",
            "family": "基础气象归因",
            "metrics_key": "pre_covid_meteorology_metrics",
            "shap_key": "pre_covid_meteorology_shap",
            "role": "2018-2019 单独训练，只保留气象、PBLH、风输送、稳定度、时间和城市特征。",
            "question": "正常排放时期，气象背景本身能解释多少 PM2.5 变化。",
            "features": "排除 PM2.5 时滞、滚动均值和共污染物，突出 PBLH、湿度、风、气压、逆温等气象贡献。",
            "academic_use": "用于解释疫情前气象贡献结构的主要对照模型。",
            "result_reading": "基础气象模型能够显示气象-only 口径与高精度口径之间的差距，说明气象对 PM2.5 有解释力，但不足以替代污染持续性和排放背景。",
            "limitation": "基础气象模型主要使用当前小时气象和少量基础派生变量，长时滞、累积和复合扩散指数表达不足，因此阶段性研究结论应优先依据过程型气象贡献模型。",
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
            "academic_use": "用于评估疫情期预测能力，并和疫情期气象贡献模型形成对照。",
            "result_reading": "疫情期高精度模型可用于检验特殊时期污染过程的可学习短时持续性。它的表现越好，越说明即便人为活动下降，污染过程仍保留连续演变特征。",
            "limitation": "该模型不适合直接判定疫情期气象因子贡献的增强或减弱，因为污染历史变量会吸收一部分非气象因素。",
            "note": "高 R2 主要说明污染短时持续性强，不表示气象变量单独解释了全部污染变化。",
        },
        {
            "title": "疫情期基础气象归因模型",
            "family": "基础气象归因",
            "metrics_key": "covid_meteorology_metrics",
            "shap_key": "covid_meteorology_shap",
            "role": "2020-2022 单独训练，专门观察人为活动减弱背景下气象贡献结构。",
            "question": "比较疫情期气象条件对 PM2.5 的解释权重相对于疫情前的阶段性变化。",
            "features": "排除 PM2.5 持续性和共污染物，只看气象、PBLH、逆温、风输送、时间和城市。",
            "academic_use": "用于疫情期气象贡献分析的主要对照模型。",
            "result_reading": "基础气象模型为疫情三年单独建模，能够初步显示人为活动减弱背景下气象变量解释结构的变化。",
            "limitation": "由于基础特征主要集中在当前小时，无法充分表达持续低边界层、连续高湿、长时间弱风和通风条件累积效应；相关方法局限已在过程型模型中改进。",
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
            "academic_use": "用于疫情后预测评估和与气象贡献模型对照。",
            "result_reading": "疫情后高精度模型用于观察恢复期污染短时预测能力。它与疫情前、疫情期高精度模型一起构成分时期预测性能对照。",
            "limitation": "2023+ 数据源差异会影响跨阶段对比，模型表现应更多看作恢复期内部预测能力，而不是直接代表排放恢复强度。",
            "note": "2023+ PM2.5 数据源与 2018-2022 不完全一致，跨时期比较要标注这一点。",
        },
        {
            "title": "疫情后基础气象归因模型",
            "family": "基础气象归因",
            "metrics_key": "post_covid_meteorology_metrics",
            "shap_key": "post_covid_meteorology_shap",
            "role": "2023+ 单独训练，用于疫情后气象贡献解释。",
            "question": "刻画疫情后 PBLH、低边界层、湿度、风速和逆温对污染累积解释力的恢复特征。",
            "features": "只保留气象、PBLH、稳定度、风输送、时间和城市特征。",
            "academic_use": "用于疫情后气象贡献分析的主要对照模型。",
            "result_reading": "基础气象模型已经显示 PBLH、低边界层和湿度相关特征重新突出，支持边界层扩散约束在恢复期重新变强的判断。",
            "limitation": "过程型模型加入异常目标和通风系数等特征后，更适合作为疫情后归因结论的主证据。",
            "note": "该模型中 PBLH 和低边界层标记重新突出，是边界层约束污染累积的重要证据。",
        },
        {
            "title": "24 小时辅助模型",
            "family": "提前量预测",
            "metrics_key": "next24_metrics",
            "shap_key": "next24_shap",
            "role": "用于提供 24 小时后 PM2.5 辅助预测。",
            "question": "评估较长提前量条件下模型提供 PM2.5 趋势参考的可用性。",
            "features": "早期预测口径的主要特征集，未完全纳入本轮 2018+ 分时期训练框架。",
            "academic_use": "只作为趋势辅助参考，不作为疫情分时期贡献分析的主要证据。",
            "result_reading": "24 小时辅助模型提供较长提前量下的趋势参照，用于辅助判断次日 PM2.5 浓度变化方向。",
            "limitation": "该模型未纳入本轮 2018+ 分时期训练和过程型气象贡献框架，不作为主要研究证据。",
            "note": "当前课题主要结论应优先引用当前小时全时期模型和三套过程型气象贡献代表模型。",
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
    if choice == "按日期自动选择过程型气象贡献模型":
        return PERIOD_TO_METEOROLOGY_V2[period]
    if choice in {
        "按日期自动选择基础气象归因模型",
        "按日期自动选择基础气象归因模型（对照口径）",
    }:
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
    st.caption("图组用于刻画当前预测情景的气象背景：左侧呈现季节参考，右侧呈现历史 PM2.5 与 PBLH，下方给出输入风场和当天气象剖面。")
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
    st.set_page_config(page_title="京津冀 PM2.5 浓度预测与气象贡献度分析", layout="wide")
    st.markdown(
        """
        <style>
        .stApp {background:#f5f7fb;}
        header {visibility:hidden;}
        #MainMenu {visibility:hidden;}
        footer {visibility:hidden;}
        [data-testid="stToolbar"] {visibility:hidden; height:0;}
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
            box-shadow:0 4px 14px rgba(24,39,75,0.03);
        }
        .model-hero-card.primary {
            background:#ffffff;
            border-color:#bfdbfe;
        }
        .model-hero-card.muted {
            background:#ffffff;
        }
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
            box-shadow:0 4px 14px rgba(24,39,75,0.03);
        }
        .training-hero {
            padding:22px 24px 20px 24px;
            background:#ffffff;
            border-color:#bfdbfe;
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
            background:#ffffff;
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
            align-items:center;
            justify-content:center;
            border-radius:999px;
            background:#eef6ff;
            color:#2563eb;
            font-size:0.78rem;
            line-height:1.2;
            padding:5px 9px;
            margin-bottom:10px;
            white-space:nowrap;
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
        .research-upgrade-hero {
            display:grid;
            grid-template-columns:minmax(0,1.25fr) minmax(300px,0.75fr);
            gap:14px;
            margin:10px 0 14px 0;
        }
        .research-upgrade-main,
        .research-upgrade-side {
            border:1px solid #d8dee6;
            border-radius:10px;
            background:#ffffff;
            box-shadow:0 10px 26px rgba(24,39,75,0.055);
            padding:18px 20px;
            border-top:4px solid #0f766e;
        }
        .research-upgrade-side {
            background:#f8fafc;
            border-top-color:#64748b;
        }
        .research-kicker {
            display:inline-flex;
            border-radius:999px;
            padding:5px 10px;
            background:#ecfdf5;
            color:#0f766e;
            font-size:0.76rem;
            font-weight:900;
            margin-bottom:10px;
        }
        .research-kicker.muted {
            background:#e2e8f0;
            color:#475569;
        }
        .research-upgrade-main h3,
        .research-upgrade-side h4 {
            margin:0 0 10px 0;
            color:#132033;
            line-height:1.35;
        }
        .research-upgrade-main h3 {font-size:1.22rem;}
        .research-upgrade-side h4 {font-size:1.04rem;}
        .research-upgrade-main p,
        .research-upgrade-side p {
            margin:0;
            color:#475569;
            line-height:1.7;
            font-size:0.93rem;
        }
        .research-evidence-panel {
            border:1px solid #d8dee6;
            border-radius:8px;
            background:#ffffff;
            padding:17px 18px 15px 18px;
            margin:12px 0 16px 0;
            box-shadow:0 4px 14px rgba(24,39,75,0.03);
        }
        .research-evidence-panel h4 {
            margin:0 0 12px 0;
            color:#132033;
            font-size:1.04rem;
            line-height:1.35;
        }
        .research-evidence-list {
            display:grid;
            grid-template-columns:repeat(2,minmax(0,1fr));
            gap:0 20px;
        }
        .research-evidence-item {
            position:relative;
            padding:0 0 12px 16px;
            border-bottom:1px solid #e5eaf0;
        }
        .research-evidence-item:nth-last-child(-n+2) {
            border-bottom:0;
            padding-bottom:0;
        }
        .research-evidence-mark {
            position:absolute;
            left:0;
            top:0.58em;
            width:5px;
            height:5px;
            border-radius:999px;
            background:#0f766e;
        }
        .research-evidence-item p {
            margin:0;
            color:#475569;
            line-height:1.68;
            font-size:0.9rem;
        }
        .research-brief {
            border:1px solid #d8dee6;
            border-radius:10px;
            background:#ffffff;
            padding:20px 22px;
            margin:10px 0 18px 0;
            box-shadow:0 6px 18px rgba(24,39,75,0.035);
        }
        .research-brief-head {
            border-bottom:1px solid #e5eaf0;
            padding-bottom:16px;
            margin-bottom:16px;
        }
        .research-brief-head h3 {
            margin:0 0 10px 0;
            color:#132033;
            font-size:1.26rem;
            line-height:1.38;
        }
        .research-brief-head p,
        .research-note p,
        .research-boundary-panel p {
            margin:0;
            color:#475569;
            line-height:1.76;
            font-size:0.94rem;
        }
        .research-brief-body {
            display:grid;
            grid-template-columns:repeat(2,minmax(0,1fr));
            gap:0;
            border:1px solid #e5eaf0;
            border-radius:8px;
            overflow:hidden;
            margin-top:14px;
        }
        .research-note {
            padding:15px 16px;
            border-right:1px solid #e5eaf0;
            border-bottom:1px solid #e5eaf0;
            background:#ffffff;
        }
        .research-note:nth-child(2n) {
            border-right:0;
        }
        .research-note:nth-last-child(-n+2) {
            border-bottom:0;
        }
        .research-note h4,
        .research-boundary-panel h4 {
            margin:0 0 8px 0;
            color:#132033;
            font-size:1.02rem;
            line-height:1.36;
        }
        .research-boundary-panel {
            margin-top:14px;
            padding:14px 16px;
            border-left:3px solid #64748b;
            background:#f8fafc;
            border-radius:8px;
        }
        .research-stat-grid {
            display:grid;
            grid-template-columns:repeat(4,minmax(0,1fr));
            gap:10px;
            margin-top:16px;
        }
        .research-stat-tile {
            border:1px solid #d8dee6;
            border-radius:9px;
            background:#f8fafc;
            padding:11px 12px;
            min-height:108px;
        }
        .research-stat-tile span {
            display:block;
            color:#64748b;
            font-size:0.78rem;
            font-weight:900;
            margin-bottom:5px;
        }
        .research-stat-tile b {
            display:block;
            color:#0f766e;
            font-size:1.02rem;
            line-height:1.32;
            margin-bottom:6px;
        }
        .research-stat-tile p {
            color:#64748b;
            font-size:0.78rem;
            line-height:1.45;
        }
        .model-hero-card,
        .intro-hero-main,
        .intro-score-panel,
        .training-hero,
        .training-score-card,
        .training-family-card,
        .flow-step,
        .method-panel,
        .tuning-main,
        .tuning-side,
        .training-detail-card,
        .period-conclusion-card,
        .research-upgrade-main,
        .research-upgrade-side,
        .research-evidence-panel {
            box-shadow:0 4px 14px rgba(24,39,75,0.03);
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
            .research-upgrade-hero {grid-template-columns:1fr;}
            .research-evidence-list {grid-template-columns:1fr;}
            .research-evidence-item:nth-last-child(-n+2) {
                border-bottom:1px solid #e5eaf0;
                padding-bottom:12px;
            }
            .research-evidence-item:last-child {
                border-bottom:0;
                padding-bottom:0;
            }
            .research-brief-body {grid-template-columns:1fr;}
            .research-note,
            .research-note:nth-child(2n),
            .research-note:nth-last-child(-n+2) {
                border-right:0;
                border-bottom:1px solid #e5eaf0;
            }
            .research-note:last-child {border-bottom:0;}
            .research-stat-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
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
            .research-stat-grid {grid-template-columns:1fr;}
            .research-brief {padding:16px;}
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

    st.title("京津冀 PM2.5 浓度预测与气象贡献度分析")

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
        st.caption("日期用于确定月份、日序、周末标记，并匹配相应季节与小时的城市历史气象画像。")

        profile = get_profile(profiles, city, selected_date.month, hour)
        defaults = input_defaults(profile)
        if "input_temperature_2m" not in st.session_state:
            write_input_state(defaults)

        if st.button("载入城市历史气象画像", use_container_width=True):
            write_input_state(defaults)
            st.session_state["input_confirmed"] = False
            st.rerun()

        with st.form("prediction_input_form", border=True):
            st.caption("表单修改后需提交确认，主面板结果随已确认输入更新。")
            if st.session_state.get("selected_prediction_model") not in MODEL_SELECT_OPTIONS:
                st.session_state["selected_prediction_model"] = MODEL_SELECT_OPTIONS[0]
            model_choice = st.selectbox(
                "当前小时预测模型",
                MODEL_SELECT_OPTIONS,
                index=0,
                key="selected_prediction_model",
            )
            st.caption("高精度模型用于浓度估计；过程型气象贡献模型用于气象-only 解释；基础气象归因模型作为方法对照保留。")
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
            st.info("当前表单包含尚未确认的修改；提交后主面板将按新输入重新计算。")
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

    tab_predict, tab_results, tab_training, tab_validation = st.tabs(["预测台", "模型介绍", "训练策略", "研究验证"])

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
                "这属于跨时期外推，结果更适合作为敏感性对照，不作为主预测依据。"
            )
        if "气象" in selected_model["type"]:
            st.info("当前选择气象侧模型：该类模型弱化或排除污染持续性和共污染物信息，适合解释气象贡献，预测精度通常低于高精度模型。")
        if selected_model["type"] == "过程型气象贡献":
            st.caption("过程型气象贡献模型需要长时滞、滚动和复合扩散特征；预测台根据当前输入和城市月小时历史画像补齐气象过程变量。")
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
        high_accuracy_tab, attribution_tab = st.tabs(["高精度预测模型", "气象模型体系"])

        with high_accuracy_tab:
            st.markdown(high_accuracy_intro_html(assets), unsafe_allow_html=True)

            st.subheader("模型表现总览")
            st.caption("高精度模型以短时浓度估计为目标，表中同时保留全时期与分时期结果。")
            performance = high_accuracy_performance_table(assets)
            chart_a, chart_b = st.columns(2)
            chart_a.plotly_chart(performance_chart(performance), use_container_width=True)
            chart_b.plotly_chart(r2_chart(performance), use_container_width=True)
            st.dataframe(performance, use_container_width=True, hide_index=True)

            with st.expander("数据补齐与高精度训练口径", expanded=False):
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
                    "目的是尽量复原当前小时 PM2.5 的实际浓度。因此它是应用预测模型，不是基础气象归因模型，也不是过程型气象贡献模型；本轮重训没有改变这一预测基准口径。"
                )

            st.markdown(
                """
                <div class="explain-band blue">
                  <h4>结果解读口径</h4>
                  <p>高精度模型 R2 明显高于基础气象归因模型和过程型气象贡献模型，并不意味着气象因子不重要，而是说明 PM2.5 历史浓度和共污染物对短时预测提供了强信息量。该结果适合作为预测能力上限，再由 weather-only 模型讨论独立气象影响。</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("高精度模型档案")
            st.caption("模型档案包含模型定位、研究任务、特征口径、训练样本、迭代次数和 SHAP 前列特征。")
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
                  <h4>高精度 SHAP 的解释边界</h4>
                  <p>高精度模型的 SHAP 前列通常包含 PM2.5 滞后、滚动均值和共污染物。它们代表污染过程的短时记忆和共变结构，能显著提升预测精度，但会混合排放、人为活动和二次生成信息。</p>
                  <p>因此，高精度模型的研究价值在于评估 PM2.5 浓度的可预测性，并为气象侧模型提供性能参照；讨论 PBLH、逆温、湿度、气压、风输送贡献时，应以右侧“气象模型体系”标签页中的过程型气象贡献模型为主要依据。</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with attribution_tab:
            st.markdown(meteorology_contribution_intro_html(assets), unsafe_allow_html=True)
            v2 = assets.get("meteorology_v2")

            if v2:
                st.subheader("过程型气象贡献模型总览")
                st.caption("本轮重训仅针对 v2-core 过程型气象贡献模型：3 个时期 x 3 种目标形式，共 9 套候选，每套 60 轮 Optuna trial。")
                v2_summary = meteorology_v2_summary_table(assets)
                v2_best = meteorology_v2_best_table(assets)
                v2_chart_col, v2_table_col = st.columns([0.55, 0.45])
                v2_chart_col.plotly_chart(meteorology_v2_r2_chart(v2_summary), use_container_width=True)
                with v2_table_col:
                    st.markdown("**各时期研究代表模型**")
                    st.dataframe(v2_best, use_container_width=True, hide_index=True)

                with st.expander("9 套过程型候选模型完整指标", expanded=False):
                    st.dataframe(v2_summary, use_container_width=True, hide_index=True)
                    st.caption("代表模型兼顾测试指标和解释口径，不一定等同于同一时期最高 R2 候选。")

                st.subheader("基础气象归因模型与过程型气象贡献模型对照")
                old_new = meteorology_v2_old_new_compare_table(assets)
                st.dataframe(old_new, use_container_width=True, hide_index=True)
                st.markdown(
                    """
                    <div class="explain-band green">
                      <h4>过程型气象贡献模型的研究意义</h4>
                      <p>基础气象归因模型主要验证气象变量对 PM2.5 变化具有一定解释力。过程型气象贡献模型进一步把气象过程表达为时滞、滚动、累计和复合扩散指数，并用三种目标形式筛选每个时期更合适的解释尺度。</p>
                      <p>因此，过程型结果不局限于单一 R2 数值，还能够讨论连续低 PBLH、长时间弱风、高湿、通风系数、北风清洁输送、南北风 V 分量和气压滞后等具有大气环境意义的过程变量。</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.subheader("过程型代表模型档案")
                st.caption("各时期档案汇总代表目标、样本量、特征数、60 轮调参信息和带 bootstrap 置信区间的 Top SHAP。")
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
                                        ("调参轮数", f"{int(model.get('trials') or 60)}"),
                                        ("迭代轮数", str(model["best_iteration"])),
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
                st.warning("尚未检索到过程型气象贡献模型结果，当前仅呈现基础气象归因模型。")
                v2_best = pd.DataFrame()

            st.subheader("过程型气象贡献模型与高精度模型的分工")
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
                        "过程型气象贡献 R2": best_metric("疫情前", "R2", assets["pre_covid_meteorology_metrics"]["test"]["r2"]),
                        "过程型气象贡献 RMSE": best_metric("疫情前", "RMSE", assets["pre_covid_meteorology_metrics"]["test"]["rmse"]),
                        "解释重点": "正常排放背景下扩散条件、季节周期和区域输送对污染差异的放大效应",
                    },
                    {
                        "时期": "疫情期",
                        "高精度 R2": assets["covid_high_accuracy_metrics"]["test"]["r2"],
                        "高精度 RMSE": assets["covid_high_accuracy_metrics"]["test"]["rmse"],
                        "过程型气象贡献 R2": best_metric("疫情期", "R2", assets["covid_meteorology_metrics"]["test"]["r2"]),
                        "过程型气象贡献 RMSE": best_metric("疫情期", "RMSE", assets["covid_meteorology_metrics"]["test"]["rmse"]),
                        "解释重点": "人为活动减弱期间 PBLH、湿度、气压和风输送对污染波动的持续解释力",
                    },
                    {
                        "时期": "疫情后",
                        "高精度 R2": assets["post_covid_high_accuracy_metrics"]["test"]["r2"],
                        "高精度 RMSE": assets["post_covid_high_accuracy_metrics"]["test"]["rmse"],
                        "过程型气象贡献 R2": best_metric("疫情后", "R2", assets["post_covid_meteorology_metrics"]["test"]["r2"]),
                        "过程型气象贡献 RMSE": best_metric("疫情后", "RMSE", assets["post_covid_meteorology_metrics"]["test"]["rmse"]),
                        "解释重点": "恢复期边界层约束、湿度过程和通风扩散能力的重新强化",
                    },
                ]
            )
            for column in ["高精度 R2", "高精度 RMSE", "过程型气象贡献 R2", "过程型气象贡献 RMSE"]:
                compare[column] = compare[column].astype(float).round(3)
            st.dataframe(compare, use_container_width=True, hide_index=True)
            st.caption("预测性能比较看高精度模型；气象贡献判断以 weather-only 过程型模型为主。")

            st.subheader("基础气象归因模型档案")
            st.caption("基础气象归因模型保留为方法对照；阶段性研究结论优先依据过程型气象贡献模型。")
            for spec in model_card_specs():
                if spec["family"] == "基础气象归因":
                    render_model_card(spec, assets)

            st.subheader("基础气象 SHAP 与残差辅助")
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

            st.subheader("阶段性气象贡献判断")
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
                    <p>疫情前代表过程型模型显示，PM2.5 变化并非只由单时刻气象决定，而是由季节周期、城市南北空间差异、连续边界层扩散能力和冷空气输送共同约束。</p>
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
                    <p>疫情期人为活动减弱，但过程型气象贡献模型仍能获得较高解释度，说明静稳扩散、风向输送和湿度过程仍会显著影响 PM2.5 波动。</p>
                  </section>
                  <section class="period-conclusion-card" style="--accent:#0f766e;--soft:#ecfdf5;">
                    <div class="period-tag">疫情后 2023+</div>
                    <h4>边界层约束和通风扩散能力重新成为主要证据</h4>
                    <div class="period-factor-row">
                      <span class="period-factor">24小时平均 PBLH</span>
                      <span class="period-factor">通风系数</span>
                      <span class="period-factor">平均风速</span>
                      <span class="period-factor">北风清洁输送</span>
                    </div>
                    <p>疫情后代表模型使用气候态异常目标，说明气象贡献更适合解释相对本地季节背景的偏离。PBLH、通风系数和风速共同指向扩散条件对污染异常的控制作用。</p>
                    <div class="period-warning">跨时期比较需注明：2023+ PM2.5 数据源与 2018-2022 不完全一致。</div>
                  </section>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_training:
        training_prediction_tab, training_attribution_tab = st.tabs(["高精度预测模型", "气象模型体系"])

        with training_prediction_tab:
            st.markdown(high_accuracy_training_html(assets), unsafe_allow_html=True)

            st.subheader("高精度模型训练明细")
            st.caption(
                "全时期主模型、分时期高精度模型和 24 小时辅助模型均属于预测口径，可使用污染历史和共污染物。"
            )
            st.dataframe(high_accuracy_training_rows(assets), use_container_width=True, hide_index=True)

            st.subheader("调参方案")
            st.caption(
                "参数搜索以验证集 RMSE 为目标，测试集仅用于最终评估。"
            )
            st.markdown(high_accuracy_tuning_html(assets), unsafe_allow_html=True)

        with training_attribution_tab:
            st.markdown(meteorology_training_html(assets), unsafe_allow_html=True)

            st.subheader("训练明细：基础气象归因模型")
            st.caption(
                "基础气象归因模型保留为方法对照，不属于本轮 v2-core 重训。"
            )
            st.dataframe(meteorology_legacy_training_rows(assets), use_container_width=True, hide_index=True)

            v2_rows = meteorology_v2_training_rows(assets)
            if not v2_rows.empty:
                st.subheader("训练明细：过程型气象贡献候选矩阵")
                st.caption(
                    "本轮重训矩阵为 3 个时期 x 3 种目标形式；每套候选均完成 60 轮调参。"
                )
                st.dataframe(v2_rows, use_container_width=True, hide_index=True)

            st.subheader("调参方案")
            st.caption(
                "过程型气象贡献模型统一使用 weather-only 特征、验证集 RMSE 和后置测试集。"
            )
            st.markdown(meteorology_tuning_html(assets), unsafe_allow_html=True)

    with tab_validation:
        render_research_validation_section(assets)

if __name__ == "__main__":
    main()
