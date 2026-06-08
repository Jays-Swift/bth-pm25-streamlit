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
}

MODEL_SELECT_OPTIONS = [
    "全时期高精度模型",
    "按日期自动选择分时期高精度模型",
    "按日期自动选择气象归因模型",
    "疫情前高精度模型",
    "疫情期高精度模型",
    "疫情后高精度模型",
    "疫情前气象归因模型",
    "疫情期气象归因模型",
    "疫情后气象归因模型",
]

MODEL_LABEL_TO_KEY = {
    "全时期高精度模型": "full_high_accuracy",
    "疫情前高精度模型": "pre_high_accuracy",
    "疫情期高精度模型": "covid_high_accuracy",
    "疫情后高精度模型": "post_high_accuracy",
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

MODEL_KEY_TO_PERIOD = {
    **{key: period for period, key in PERIOD_TO_HIGH_ACCURACY.items()},
    **{key: period for period, key in PERIOD_TO_METEOROLOGY.items()},
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
    "low_pblh_flag": "低边界层标记",
    "t_inverse_850_1000": "逆温指数 T850-T1000",
    "temperature_850hPa": "850hPa 温度",
    "temperature_2m": "2m 气温",
    "dew_point_2m": "露点温度",
    "relative_humidity_2m": "相对湿度",
    "pressure_msl": "海平面气压",
    "surface_pressure": "地面气压",
    "wind_speed_10m": "10m 风速",
    "wind_speed_10m_lag_3h": "前3小时风速",
    "wind_speed_10m_lag_24h": "前24小时风速",
    "wind_direction_10m": "10m 风向",
    "wind_u_10m": "东西向风 U",
    "wind_v_10m": "南北向风 V",
    "wind_u_10m_lag_3h": "前3小时东西向风 U",
    "wind_u_10m_lag_24h": "前24小时东西向风 U",
    "wind_v_10m_lag_3h": "前3小时南北风 V",
    "wind_v_10m_lag_24h": "前24小时南北风 V",
    "wind_gusts_10m": "10m 阵风",
    "northerly_cleaning_10m": "北风清洁输送",
    "southerly_transport_10m": "南风污染输送",
    "precipitation": "降水量",
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
          <p>回答“能不能预测得准”。适合预测台展示和整体性能说明。</p>
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
          <p>用于和疫情前、疫情期对照，观察 PBLH、湿度、风速和逆温等贡献是否重新突出。</p>
          <div class="route-meta">2023+ | 恢复期贡献对照</div>
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
        <p>气象归因模型 R2 {post_met["test"]["r2"]:.3f}，用于观察边界层高度和低 PBLH 标记是否重新突出。</p>
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
        st.markdown(f"**适合回答的问题**：{spec['question']}")
        st.markdown(f"**特征口径**：{spec['features']}")
        st.markdown(f"**学术使用建议**：{spec['academic_use']}")
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
            "question": "在给定当前气象、PM2.5 历史和共污染物背景时，PM2.5 能否预测得准。",
            "features": "气象、ERA5 PBLH、逆温、风输送、PM2.5 时滞、滚动均值、PM10、CO、NO2、SO2、O3、AOD、dust。",
            "academic_use": "用于证明机器学习框架具备较强预测能力，不直接作为气象因子独立贡献的唯一依据。",
            "note": "该模型跨越 2018-2026，2023+ PM2.5 数据源与 2018-2022 不完全一致，跨期解释需谨慎。",
            "expanded": True,
        },
        {
            "title": "疫情前高精度模型",
            "family": "分时期预测",
            "metrics_key": "pre_covid_high_accuracy_metrics",
            "shap_key": "pre_covid_high_accuracy_shap",
            "role": "2018-2019 单独训练的高精度预测模型。",
            "question": "正常排放时期，加入污染持续性和共污染物后 PM2.5 可预测到什么程度。",
            "features": "与全时期高精度模型同类，但只在疫情前数据内训练和测试。",
            "academic_use": "作为疫情前预测精度上限和气象归因模型的对照。",
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
            "note": "精度显著低于高精度模型是预期现象，因为它故意不使用强污染持续性特征。",
        },
        {
            "title": "疫情期高精度模型",
            "family": "分时期预测",
            "metrics_key": "covid_high_accuracy_metrics",
            "shap_key": "covid_high_accuracy_shap",
            "role": "2020-2022 单独训练的高精度预测模型。",
            "question": "疫情期加入污染持续性和共污染物后，PM2.5 是否仍可高精度预测。",
            "features": "气象 + PBLH + 稳定度 + PM2.5 时滞/滚动均值 + 共污染物。",
            "academic_use": "用于展示疫情期预测能力，并和疫情期气象归因模型形成对照。",
            "note": "高 R2 主要说明污染短时持续性强，不表示气象变量单独解释了全部污染变化。",
        },
        {
            "title": "疫情期气象归因模型",
            "family": "气象归因",
            "metrics_key": "covid_meteorology_metrics",
            "shap_key": "covid_meteorology_shap",
            "role": "2020-2022 单独训练，专门观察人为活动减弱背景下气象贡献结构。",
            "question": "疫情期气象条件对 PM2.5 的解释权重与疫情前是否不同。",
            "features": "排除 PM2.5 持续性和共污染物，只看气象、PBLH、逆温、风输送、时间和城市。",
            "academic_use": "论文中疫情期气象贡献分析的核心模型。",
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
            "note": "2023+ PM2.5 数据源与 2018-2022 不完全一致，跨时期比较要标注这一点。",
        },
        {
            "title": "疫情后气象归因模型",
            "family": "气象归因",
            "metrics_key": "post_covid_meteorology_metrics",
            "shap_key": "post_covid_meteorology_shap",
            "role": "2023+ 单独训练，用于疫情后气象贡献解释。",
            "question": "疫情后 PBLH、低边界层、湿度、风速和逆温对污染累积的解释是否增强。",
            "features": "只保留气象、PBLH、稳定度、风输送、时间和城市特征。",
            "academic_use": "论文中疫情后气象贡献分析的核心模型。",
            "note": "该模型中 PBLH 和低边界层标记重新突出，是边界层约束污染累积的重要证据。",
        },
        {
            "title": "24 小时辅助模型",
            "family": "提前量预测",
            "metrics_key": "next24_metrics",
            "shap_key": "next24_shap",
            "role": "用于展示 24 小时后 PM2.5 辅助预测。",
            "question": "在较长提前量下，模型能否给出趋势参考。",
            "features": "旧训练口径的核心特征集，未完全纳入本轮 2018+ 分时期训练框架。",
            "academic_use": "只作为网页辅助展示，不作为疫情分时期贡献分析的主要证据。",
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
    if choice == "按日期自动选择气象归因模型":
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


def complete_row(features: list[str], row: dict) -> pd.DataFrame:
    completed = {feature: row.get(feature, 0) for feature in features}
    return pd.DataFrame([completed])


def predict(bundle: dict, row: dict) -> float:
    features = bundle["features"]
    frame = complete_row(features, row)
    matrix = bundle["preprocessor"].transform(frame[features])
    return float(bundle["model"].predict(matrix)[0])


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
            model_choice = st.selectbox(
                "当前小时预测模型",
                MODEL_SELECT_OPTIONS,
                index=0,
                key="selected_prediction_model",
            )
            st.caption("高精度模型适合预测展示；气象归因模型适合观察单靠气象背景得到的估计。")
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
        if selected_model["type"] == "气象归因":
            st.info("当前使用的是气象归因模型：它会弱化或排除污染持续性和共污染物信息，适合解释气象贡献，预测精度通常低于高精度模型。")
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
        st.markdown(model_intro_html(assets), unsafe_allow_html=True)

        st.subheader("模型档案")
        st.caption("点击任一模型卡片可查看模型定位、适合回答的问题、特征口径、训练样本、迭代轮数和 SHAP 前列特征。")
        for spec in model_card_specs():
            render_model_card(spec, assets)

        performance = pd.DataFrame(
            [
                metric_row("全时期高精度", assets["current_metrics"], "预测台主模型"),
                metric_row("疫情前高精度", assets["pre_covid_high_accuracy_metrics"], "2018-2019 预测对照"),
                metric_row("疫情前气象", assets["pre_covid_meteorology_metrics"], "2018-2019 气象贡献"),
                metric_row("疫情期高精度", assets["covid_high_accuracy_metrics"], "2020-2022 预测对照"),
                metric_row("疫情期气象", assets["covid_meteorology_metrics"], "2020-2022 气象贡献"),
                metric_row("疫情后高精度", assets["post_covid_high_accuracy_metrics"], "2023+ 预测对照"),
                metric_row("疫情后气象", assets["post_covid_meteorology_metrics"], "2023+ 气象贡献"),
                metric_row("24 小时辅助", assets["next24_metrics"], "旧口径辅助预测"),
            ]
        )
        chart_a, chart_b = st.columns(2)
        chart_a.plotly_chart(performance_chart(performance), use_container_width=True)
        chart_b.plotly_chart(r2_chart(performance), use_container_width=True)
        st.dataframe(performance, use_container_width=True, hide_index=True)
        st.info(
            "当前小时预测已经换成新训练的全时期高精度模型。"
            "24 小时后 PM2.5 仍是此前 2022-2026 旧口径模型，因此只作为辅助展示；"
            "课题正文中的时期对比和气象贡献结论应引用三个分时期气象贡献模型。"
        )

        with st.expander("数据补齐与训练口径", expanded=True):
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
                "高精度模型追求预测精度，会使用 PM2.5 滞后、滚动均值和共污染物；"
                "气象贡献模型则故意排除这些污染持续性和共污染物特征，以便比较 PBLH、逆温、湿度、气压、风输送等气象因子的独立贡献。"
            )

        st.subheader("预测模型与气象归因模型对照")
        compare = pd.DataFrame(
            [
                {
                    "时期": "疫情前 2018-2019",
                    "高精度 R2": assets["pre_covid_high_accuracy_metrics"]["test"]["r2"],
                    "高精度 RMSE": assets["pre_covid_high_accuracy_metrics"]["test"]["rmse"],
                    "气象归因 R2": assets["pre_covid_meteorology_metrics"]["test"]["r2"],
                    "气象归因 RMSE": assets["pre_covid_meteorology_metrics"]["test"]["rmse"],
                },
                {
                    "时期": "疫情期 2020-2022",
                    "高精度 R2": assets["covid_high_accuracy_metrics"]["test"]["r2"],
                    "高精度 RMSE": assets["covid_high_accuracy_metrics"]["test"]["rmse"],
                    "气象归因 R2": assets["covid_meteorology_metrics"]["test"]["r2"],
                    "气象归因 RMSE": assets["covid_meteorology_metrics"]["test"]["rmse"],
                },
                {
                    "时期": "疫情后 2023+",
                    "高精度 R2": assets["post_covid_high_accuracy_metrics"]["test"]["r2"],
                    "高精度 RMSE": assets["post_covid_high_accuracy_metrics"]["test"]["rmse"],
                    "气象归因 R2": assets["post_covid_meteorology_metrics"]["test"]["r2"],
                    "气象归因 RMSE": assets["post_covid_meteorology_metrics"]["test"]["rmse"],
                },
            ]
        )
        for column in ["高精度 R2", "高精度 RMSE", "气象归因 R2", "气象归因 RMSE"]:
            compare[column] = compare[column].astype(float).round(3)
        st.dataframe(compare, use_container_width=True, hide_index=True)
        st.caption("高精度模型回答“能否预测得准”；气象归因模型回答“气象背景能独立解释多少污染变化”。两类模型都需要保留，但学术解释应以气象归因模型为主。")

        shap_a, shap_b = st.columns(2)
        shap_a.plotly_chart(shap_chart(assets["current_shap"], "全时期高精度模型 SHAP 贡献"), use_container_width=True)
        shap_b.plotly_chart(shap_chart(assets["covid_meteorology_shap"], "疫情期气象贡献模型 SHAP 贡献"), use_container_width=True)

        st.subheader("分时期气象贡献对比")
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

        st.subheader("分时期高精度模型 SHAP 对照")
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
        st.caption("高精度模型的 SHAP 前列通常是 PM2.5 滞后、滚动均值和共污染物，说明其主要价值是提升预测能力，而不是替代气象因子归因。")

        st.subheader("气象模型残差分析")
        residual = assets["period_residual_analysis"][
            ["时期", "气象模型_R2", "高精度模型_R2", "气象残差均值", "负残差占比"]
        ].copy()
        for column in ["气象模型_R2", "高精度模型_R2", "气象残差均值", "负残差占比"]:
            residual[column] = residual[column].astype(float).round(3)
        st.dataframe(residual, use_container_width=True, hide_index=True)
        st.caption("气象残差 = 实测 PM2.5 - 气象模型预测 PM2.5。由于三套气象模型分别校准、测试时段不同，残差只作为非气象因素讨论的辅助证据，不直接等同于排放变化量。")

        st.subheader("阶段性结论")
        st.markdown(
            """
            <div class="period-conclusion-grid">
              <section class="period-conclusion-card" style="--accent:#2563eb;--soft:#eff6ff;">
                <div class="period-tag">疫情前 2018-2019</div>
                <h4>扩散条件与区域输送主导差异</h4>
                <div class="period-factor-row">
                  <span class="period-factor">PBLH</span>
                  <span class="period-factor">南北风 V 分量</span>
                  <span class="period-factor">湿度</span>
                  <span class="period-factor">气压</span>
                </div>
                <p>正常排放时期，边界层高度、南北风输送、湿度和气压贡献更明显，说明扩散条件与区域输送更容易放大城市间和时段间的污染差异。</p>
              </section>
              <section class="period-conclusion-card" style="--accent:#7c3aed;--soft:#f5f3ff;">
                <div class="period-tag">疫情期 2020-2022</div>
                <h4>气象仍重要，但 PBLH 权重下降</h4>
                <div class="period-factor-row">
                  <span class="period-factor">露点</span>
                  <span class="period-factor">湿度</span>
                  <span class="period-factor">季节周期</span>
                  <span class="period-factor">风输送</span>
                </div>
                <p>疫情期露点、湿度、季节周期和风输送仍保持较强解释力；但 PBLH 的 SHAP 强度低于疫情前，符合人为活动减弱后气象背景解释权重变化的预期。</p>
              </section>
              <section class="period-conclusion-card" style="--accent:#0f766e;--soft:#ecfdf5;">
                <div class="period-tag">疫情后 2023+</div>
                <h4>边界层高度对污染累积解释增强</h4>
                <div class="period-factor-row">
                  <span class="period-factor">PBLH</span>
                  <span class="period-factor">低边界层标记</span>
                  <span class="period-factor">湿度</span>
                  <span class="period-factor">风速</span>
                </div>
                <p>疫情后 PBLH、低边界层标记、湿度和风速重新突出，边界层高度对污染累积的解释更强。</p>
                <div class="period-warning">跨时期比较需注明：2023+ PM2.5 数据源与 2018-2022 不完全一致。</div>
              </section>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_training:
        st.markdown(training_strategy_html(assets), unsafe_allow_html=True)

        st.subheader("训练明细")
        st.caption("下表保留每个模型的训练样本、验证样本、测试样本、调参轮数、最佳迭代和测试表现，用作方法部分的可追溯依据。")
        st.dataframe(training_strategy_rows(assets), use_container_width=True, hide_index=True)

        st.subheader("关键方法说明")
        st.caption("这一部分对应论文方法章节，重点说明模型如何调参、如何避免时间泄漏、以及最终指标如何产生。")
        st.markdown(tuning_method_html(assets), unsafe_allow_html=True)

        st.subheader("训练产物")
        st.markdown(
            """
            <div class="training-artifacts">
              <div class="artifact-card"><b>models/</b><span>保存 LightGBM 预测模型、气象归因模型和 24 小时辅助模型。</span></div>
              <div class="artifact-card"><b>reports/</b><span>保存指标 JSON、测试集预测、SHAP 排名、训练摘要和残差分析。</span></div>
              <div class="artifact-card"><b>app_assets/</b><span>保存网页直接读取的指标、图表数据、城市画像和展示资源。</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("当前预测台已支持选择全时期高精度、按日期自动分时期高精度、按日期自动气象归因，以及各时期单独模型。")

if __name__ == "__main__":
    main()
