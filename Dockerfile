FROM nvidia/cuda:13.0.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    AIRLLM_HOST=0.0.0.0 \
    AIRLLM_PORT=8000 \
    AIRLLM_DEVICE=cuda:0 \
    AIRLLM_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    AIRLLM_MAX_SEQ_LEN= \
    AIRLLM_MAX_NEW_TOKENS=256 \
    AIRLLM_TEMPERATURE=0.2 \
    AIRLLM_TOP_P=0.95 \
    AIRLLM_PREFETCHING=true \
    AIRLLM_LAYERS_PER_BATCH=auto \
    AIRLLM_LAZY_LOAD_MODEL=true \
    AIRLLM_ENFORCE_AUTH=false \
    AIRLLM_API_KEY=

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY air_llm /app/air_llm

RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip setuptools wheel && \
    /opt/venv/bin/pip install ./air_llm

EXPOSE 8000

CMD ["python3", "-m", "airllm.server", "serve", "--host", "0.0.0.0", "--port", "8000"]
