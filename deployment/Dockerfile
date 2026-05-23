# Hugging Face Spaces / Docker deployment for the Streamlit comparison app.
# Uses Ollama for the OSS model (no torch/transformers in the image).
#
# Build from repo root:
#   docker build -f deployment/Dockerfile -t ai-assistant-comparison .
#
# Hugging Face Spaces: root Dockerfile is a symlink to this file.

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    OLLAMA_HOST=http://127.0.0.1:11434 \
    OSS_BACKEND=ollama \
    OSS_MODEL=qwen2.5:1.5b \
    FRONTIER_BACKEND=openai \
    FRONTIER_MODEL=llama-3.1-8b-instant \
    MAX_EXCHANGES=8 \
    STREAMLIT_SERVER_PORT=7860 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    procps \
    zstd \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama. The model is pulled at container startup so the image stays small.
RUN curl -fsSL https://ollama.com/install.sh | sh

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
RUN chmod +x deployment/start_space.sh \
    && mkdir -p logs evals/results screenshots

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=5 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

CMD ["deployment/start_space.sh"]
