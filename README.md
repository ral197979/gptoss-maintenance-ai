
# 🤖 GPT-OSS Predictive Maintenance AI

This app uses a fine-tuned open-weight GPT-OSS-20B model to provide predictive maintenance suggestions based on sensor input data. It supports:

- ✅ API deployment with FastAPI
- ✅ Interactive UI with Streamlit
- ✅ CLI inference
- ✅ Docker and Render one-click deployment

---

## 🚀 One-Click Deploy

### 🟢 Deploy FastAPI (REST API):
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/ral197979/gptoss-maintenance-ai&env=MODE=api)

### 💬 Deploy Streamlit (Chat UI):
[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/ral197979/gptoss-maintenance-ai&env=MODE=ui)

---

## 🧪 Local Development

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Run API:
```bash
./run_app.sh api
```

### Run UI:
```bash
./run_app.sh ui
```

---

## 📦 Docker

### Build and run (API):
```bash
docker build -t gptoss .
docker run -p 8000:8000 gptoss
```

### Switch to UI:
Edit the last line in `Dockerfile`:
```dockerfile
CMD ["streamlit", "run", "streamlit_gptoss.py"]
```

---

## 🔧 Example Prompt:
```
Sensor report: Temp=90°C, Vibration=1.2g

What is the likely issue?
```

---

> Created by [ral197979](https://github.com/ral197979)
