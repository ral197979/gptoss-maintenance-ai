
# Dockerfile for GPT-OSS API or UI
FROM python:3.10-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir torch transformers peft accelerate bitsandbytes fastapi uvicorn streamlit

# Default to inference API
CMD ["uvicorn", "api_gptoss:app", "--host", "0.0.0.0", "--port", "8000"]
