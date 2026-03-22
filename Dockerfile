FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AIRLLM_HOST=0.0.0.0 \
    AIRLLM_PORT=8000 \
    AIRLLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY air_llm /app/air_llm

RUN pip install --upgrade pip && \
    pip install ./air_llm

EXPOSE 8000

CMD ["airllm", "serve", "--host", "0.0.0.0", "--port", "8000"]
