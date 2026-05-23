# Deployment

Docker deployment for Hugging Face Spaces and local containers.

| File | Purpose |
|---|---|
| `Dockerfile` | Python 3.11 image with Ollama + Streamlit on port 7860 |
| `start_space.sh` | Starts `ollama serve`, pulls the OSS model, runs the comparison app |

**Hugging Face Spaces:** the root `Dockerfile` symlinks here so HF can find it at the repo root.

**Local build:**

```bash
docker build -f deployment/Dockerfile -t ai-assistant-comparison .
docker run -p 7860:7860 --env-file .env ai-assistant-comparison
```

Set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, and model env vars as described in the main `README.md`.
