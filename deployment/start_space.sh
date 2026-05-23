#!/usr/bin/env bash
set -euo pipefail

export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
export OSS_BACKEND="${OSS_BACKEND:-ollama}"
export OSS_MODEL="${OSS_MODEL:-qwen2.5:1.5b}"
export FRONTIER_BACKEND="${FRONTIER_BACKEND:-openai}"
export FRONTIER_MODEL="${FRONTIER_MODEL:-llama-3.1-8b-instant}"

echo "Starting Ollama on ${OLLAMA_HOST}..."
ollama serve &
OLLAMA_PID="$!"

echo "Waiting for Ollama..."
for _ in $(seq 1 60); do
    if ollama list >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

echo "Pulling OSS model: ${OSS_MODEL}"
ollama pull "${OSS_MODEL}"

echo "Starting Streamlit app..."
streamlit run apps/comparison_app.py \
    --server.port="${STREAMLIT_SERVER_PORT:-7860}" \
    --server.address="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}" \
    --server.headless=true &
STREAMLIT_PID="$!"

trap 'kill "${STREAMLIT_PID}" "${OLLAMA_PID}" 2>/dev/null || true' INT TERM
wait "${STREAMLIT_PID}"
