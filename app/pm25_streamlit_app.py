from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
MODEL_DIR = PROJECT_ROOT / "models"
ASSET_DIR = PROJECT_ROOT / "app_assets"

CURRENT_MODEL_PATH = MODEL_DIR / "high_accuracy_lightgbm_core_target_pm2_5.joblib"
NEXT24_MODEL_PATH = MODEL_DIR / "high_accuracy_lightgbm_core_target_pm2_5_next_24h.joblib"


AQI_BANDS = [
    (35, "优", "#2e7d32"),
    (75, "良", "#f9a825"),
    (115, "轻度污染", "#ef6c00"),
    (150, "中度污染", "#c62828"),
    (250, "重度污染", "#6a1b9a"),
    (10_000, "严重污染", "#4e342e"),
]


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
def load_assets() -> dict[str, pd.DataFrame | dict]:
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
    }


def pm25_category(value: float) -> tuple[str, str]:
    for upper, label, color in AQI_BANDS:
        if value <= upper:
            return label, color
    return "严重污染", "#4e342e"


def metric_text(metrics: dict) -> str:
    test = metrics["test"]
    return f"MAE {test['mae']:.2f} | RMSE {test['rmse']:.2f} | R2 {test['r2']:.3f}"


def metric_row(name: str, metrics: dict, role: str) -> dict[str, object]:
    test = metrics["test"]
    return {
        "模型": name,
        "定位": role,
        "MAE": round(float(test["mae"]), 3),
        "RMSE": round(float(test["rmse"]), 3),
        "R2": round(float(test["r2"]), 3),
    }


def get_profile(profiles: pd.DataFrame, city: str, month: int, hour: int) -> dict[str, float]:
    match = profiles[(profiles["city"] == city) & (profiles["month"] == month) & (profiles["hour"] == hour)]
    if match.empty:
        match = profiles[profiles["city"] == city]
    if match.empty:
        match = profiles
    return match.median(numeric_only=True).to_dict()


def add_time_features(row: dict, selected_date: date, hour: int) -> None:
    day = pd.Timestamp(selected_date)
    row["hour"] = hour
    row["month"] = day.month
    row["dayofyear"] = day.dayofyear
    row["weekday"] = day.weekday()
    row["is_weekend"] = int(row["weekday"] in [5, 6])
    row["hour_sin"] = math.sin(2 * math.pi * hour / 24)
    row["hour_cos"] = math.cos(2 * math.pi * hour / 24)
    row["dayofyear_sin"] = math.sin(2 * math.pi * row["dayofyear"] / 366)
    row["dayofyear_cos"] = math.cos(2 * math.pi * row["dayofyear"] / 366)
    row["month_sin"] = math.sin(2 * math.pi * row["month"] / 12)
    row["month_cos"] = math.cos(2 * math.pi * row["month"] / 12)


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
    top = shap_df.head(15).sort_values("mean_abs_shap")
    fig = px.bar(top, x="mean_abs_shap", y="feature", orientation="h", title=title)
    fig.update_traces(marker_color="#2563eb")
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=50, b=20), xaxis_title="平均绝对 SHAP", yaxis_title="")
    return fig


def add_pm25_bands(fig: go.Figure, y_max: float) -> None:
    bands = [
        (0, 35, "优", "rgba(46, 125, 50, 0.10)"),
        (35, 75, "良", "rgba(249, 168, 37, 0.12)"),
        (75, 115, "轻度", "rgba(239, 108, 0, 0.12)"),
        (115, 150, "中度", "rgba(198, 40, 40, 0.10)"),
        (150, max(250, y_max), "重度+", "rgba(106, 27, 154, 0.08)"),
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
    cards = []
    for item in items:
        cards.append(
            f"""
            <div class="factor-card" style="border-top-color:{item['color']}">
              <div class="factor-title">{item['title']}</div>
              <div class="factor-value" style="color:{item['color']}">{item['value']}</div>
              <div class="factor-detail">{item['detail']}</div>
            </div>
            """
        )
    return f"<div class='factor-grid'>{''.join(cards)}</div>"


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
        .small-note {color: #5f6b7a; font-size: 0.9rem;}
        .scenario-strip {
            display:grid;
            grid-template-columns:repeat(6,minmax(0,1fr));
            gap:10px;
            margin:12px 0 16px 0;
        }
        .scenario-item {
            background:#ffffff;
            border:1px solid #d8dee6;
            border-radius:8px;
            padding:10px 12px;
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
        @media (max-width: 900px) {
            .scenario-strip {grid-template-columns:repeat(2,minmax(0,1fr));}
            .factor-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    style_page()
    assets = load_assets()
    current_bundle = load_model(str(resolve_path("models", "high_accuracy_lightgbm_core_target_pm2_5.joblib")))
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
                "values": form_values,
            }
            st.session_state["input_confirmed"] = True

        confirmed = st.session_state["confirmed_inputs"]
        changed = (
            confirmed["city"] != city
            or confirmed["date"] != selected_date
            or confirmed["hour"] != hour
            or any(float(confirmed["values"][key]) != float(form_values[key]) for key in form_values)
        )
        if changed:
            st.info("当前表单已有修改，点击“确认输入并预测”后主面板才会更新。")
        elif st.session_state.get("input_confirmed", False):
            st.success("输入已确认，结果来自当前表单。")

    city = confirmed["city"]
    selected_date = confirmed["date"]
    hour = confirmed["hour"]
    overrides = make_overrides(confirmed["values"])
    pblh = overrides["boundary_layer_height"]
    wind_speed = overrides["wind_speed_10m"]
    wind_direction = overrides["wind_direction_10m"]
    row = scenario_row(city_info, profiles, city, selected_date, hour, overrides)
    current_prediction = predict(current_bundle, row)
    next24_prediction = predict(next24_bundle, row)
    category, color = pm25_category(current_prediction)
    next_category, next_color = pm25_category(next24_prediction)
    t_inverse = row["t_inverse_850_1000"]

    tab_predict, tab_results, tab_weather = st.tabs(["预测台", "模型成果", "气象图"])

    with tab_predict:
        col_a, col_b, col_c, col_d = st.columns(4)
        col_a.metric("当前小时 PM2.5", f"{current_prediction:.1f} ug/m3", category)
        col_b.metric("24 小时后 PM2.5", f"{next24_prediction:.1f} ug/m3", next_category)
        col_c.metric("逆温指数", f"{t_inverse:.1f} C", "存在逆温" if t_inverse > 0 else "无逆温")
        col_d.metric("PBLH", f"{pblh:.0f} m", "低边界层" if row["low_pblh_flag"] else "扩散较好")

        st.markdown(
            f"""
            <div class="prediction-card">
              <div style="font-size:1.05rem;color:#263238;">{city} | {selected_date} {hour:02d}:00</div>
              <div style="font-size:2.6rem;font-weight:700;color:{color};line-height:1.15;">{current_prediction:.1f} ug/m3</div>
              <div style="font-size:1rem;color:#5f6b7a;">空气质量等级: {category}；24 小时后预测等级: <span style="color:{next_color};font-weight:600;">{next_category}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="scenario-strip">
              <div class="scenario-item"><div class="scenario-label">已确认日期</div><div class="scenario-value">{selected_date}</div></div>
              <div class="scenario-item"><div class="scenario-label">预测小时</div><div class="scenario-value">{hour:02d}:00</div></div>
              <div class="scenario-item"><div class="scenario-label">温湿度</div><div class="scenario-value">{overrides['temperature_2m']:.1f} C / {overrides['relative_humidity_2m']:.0f}%</div></div>
              <div class="scenario-item"><div class="scenario-label">风</div><div class="scenario-value">{wind_speed:.1f} m/s @ {wind_direction:.0f} deg</div></div>
              <div class="scenario-item"><div class="scenario-label">PBLH</div><div class="scenario-value">{pblh:.0f} m</div></div>
              <div class="scenario-item"><div class="scenario-label">逆温指数</div><div class="scenario-value">{t_inverse:.1f} C</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
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

    with tab_results:
        m1, m2 = st.columns(2)
        m1.metric("当前小时核心模型", metric_text(assets["current_metrics"]))
        m2.metric("24 小时核心模型", metric_text(assets["next24_metrics"]))
        st.markdown(
            """
            <div class="factor-grid">
              <div class="model-card"><b>核心当前小时模型</b><p>气象 + ERA5 PBLH + 逆温 + 风输送 + 时滞特征。用于当前 PM2.5 估计和气象贡献度主分析。</p></div>
              <div class="model-card"><b>核心 24 小时模型</b><p>同样只使用核心气象特征，用来检验提前 24 小时预测能力。</p></div>
              <div class="model-card"><b>扩展当前小时模型</b><p>加入 PM10、CO、NO2、SO2、O3、AOD、dust，作为精度上限对照。</p></div>
              <div class="model-card"><b>扩展 24 小时模型</b><p>用于检验共污染物对 24 小时预测是否有补充价值。</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        performance = pd.DataFrame(
            [
                metric_row("核心当前小时", assets["current_metrics"], "主气象归因模型"),
                metric_row("扩展当前小时", assets["extended_current_metrics"], "精度上限对照"),
                metric_row("核心 24 小时", assets["next24_metrics"], "主预测对照"),
                metric_row("扩展 24 小时", assets["extended_next24_metrics"], "共污染物对照"),
            ]
        )
        chart_a, chart_b = st.columns(2)
        chart_a.plotly_chart(performance_chart(performance), use_container_width=True)
        chart_b.plotly_chart(r2_chart(performance), use_container_width=True)
        st.dataframe(performance, use_container_width=True, hide_index=True)
        with st.expander("为什么气象贡献度以核心模型为准", expanded=False):
            st.write(
                "扩展模型中的 PM10、CO、SO2、NO2、O3、AOD 和 dust 与 PM2.5 存在同源排放、二次转化或再分析同化关系。"
                "这些变量能帮助预测，但会把排放和化学过程的相关性带入解释结果。"
                "因此论文中定量分析边界层高度、逆温、风速风向、气压、湿度等气象贡献时，应优先引用核心模型的 SHAP 排名。"
            )
        shap_a, shap_b = st.columns(2)
        shap_a.plotly_chart(shap_chart(assets["current_shap"], "当前小时模型 SHAP 贡献"), use_container_width=True)
        shap_b.plotly_chart(shap_chart(assets["next24_shap"], "24 小时模型 SHAP 贡献"), use_container_width=True)

    with tab_weather:
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
        w1.plotly_chart(wind_polar(wind_speed, wind_direction), use_container_width=True)
        weather_df = build_daily_prediction(current_bundle, city_info, profiles, city, selected_date, overrides)
        fig_weather = px.line(
            weather_df,
            x="hour",
            y=["temperature_2m", "wind_speed_10m", "boundary_layer_height"],
            title="当天气象变量剖面",
        )
        fig_weather.update_layout(height=330, margin=dict(l=10, r=10, t=50, b=20), xaxis_title="小时", yaxis_title="")
        fig_weather.update_traces(line=dict(width=3))
        w2.plotly_chart(fig_weather, use_container_width=True)


if __name__ == "__main__":
    main()
