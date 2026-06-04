# 京津冀 PM2.5 预测网站部署包

这个目录是公网部署用的最小 Streamlit 网站包，已包含：

- `streamlit_app.py`: 云平台入口。
- `app/`: 网页应用代码。
- `models/`: 已训练模型。
- `app_assets/`: 城市画像、指标和 SHAP 表。
- `requirements.txt`: 运行依赖。

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
