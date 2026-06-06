# 京津冀 PM2.5 预测网站部署包

这个目录是公网部署用的最小 Streamlit 网站包，已包含：

- `streamlit_app.py`: 云平台入口。
- `app/`: 网页应用代码。
- `models/`: 已训练模型。
- `app_assets/`: 城市画像、指标和 SHAP 表。
- `requirements.txt`: 运行依赖。

当前版本已经更新为 2018-2026 京津冀 PM2.5 与气象特征数据，包含全时期高精度预测模型、疫情前/疫情期/疫情后分时期高精度模型，以及分时期气象贡献模型。网页中空气质量等级配色为：优/良好绿色，轻度污染黄色，中度污染及更差红色至深红色。

## Hugging Face Spaces

1. 新建 Space，SDK 选择 `Streamlit`。
2. 上传本目录全部文件。
3. 等待构建完成，得到公网 URL。

## Streamlit Community Cloud

1. 把本目录推送到一个 GitHub 仓库。
2. 在 Streamlit Cloud 选择该仓库。
3. Main file path 填 `streamlit_app.py`。
4. 部署完成后即可得到公网 URL。

## Render / Railway 等服务器

启动命令：

```bash
streamlit run streamlit_app.py --server.address 0.0.0.0 --server.port $PORT
```

## 说明

完整原始数据没有放入部署包，网站只使用模型和轻量资产，因此包体积较小，适合公网部署。
