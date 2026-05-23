# Comparative AI Personal Assistants

This repository implements and evaluates two personal assistants with the same user experience and capabilities:

- **Open Source Assistant:** Qwen2.5 via Ollama, with Hugging Face Transformers fallback
- **Frontier Model Assistant:** hosted OpenAI-compatible API, configured for Groq Llama 3.1-8B by default

Both assistants support multi-turn conversation, short-term conversational memory, basic assistant behavior, safety filtering, latency logging, and a side-by-side Streamlit comparison interface.

## Deliverables

| Requirement | Location |
|---|---|
| Complete source code | `apps/`, `core/`, `evals/`, `deployment/` |
| README with setup, architecture, tradeoffs, future work | `README.md` |
| Short evaluation report | `reports/report.md`, `reports/evaluation_report.html`, `reports/evaluation_report.pdf` |
| Infographics / charts | `evals/results/*.png` |
| Evaluation data | `evals/*.json`, `evals/results/aggregate_scores.json` |

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=<your-groq-or-openai-compatible-api-key>
OPENAI_BASE_URL=https://api.groq.com/openai/v1

OSS_BACKEND=ollama
OSS_MODEL=qwen2.5:1.5b

FRONTIER_BACKEND=openai
FRONTIER_MODEL=llama-3.1-8b-instant
```

### 3. Start the open-source model

Install Ollama, then pull the OSS model:

```bash
ollama pull qwen2.5:1.5b
```

You can substitute another local OSS model such as `phi3:mini`, `llama3.2:1b`, or a Hugging Face model by setting `OSS_BACKEND=huggingface`.

### 4. Run the main app

```bash
streamlit run apps/comparison_app.py
```

Optional single-model apps:

```bash
streamlit run apps/oss_assistant.py
streamlit run apps/frontier_assistant.py
```

## What The App Does

The main Streamlit app sends each user message to both assistants and displays their responses side by side. Each assistant has its own rolling memory buffer so follow-up questions work naturally.

Core assistant behavior includes:

- Multi-turn chat
- Short-term memory over the most recent 8 exchanges
- Shared system prompt and assistant behavior constraints
- Input safety checks before model calls
- Output safety checks after model calls
- Streaming responses in the UI
- Latency and approximate token logging
- Exportable conversation logs

## Architecture

```text
User Input
    |
    v
Input Safety Filter
    |
    v
Conversation Memory
    |
    v
Prompt Builder
    |
    v
Model Adapter Layer
    |-- OSSAssistant: Ollama / Hugging Face Transformers
    |-- FrontierAssistant: OpenAI-compatible hosted API
    |
    v
Output Safety Check
    |
    v
Observability Logger
    |
    v
Assistant Response
```

### Project Structure

```text
apps/
  comparison_app.py       Main side-by-side Streamlit app
  oss_assistant.py        OSS-only Streamlit app
  frontier_assistant.py   Frontier-only Streamlit app
  shared_ui.py            Shared Streamlit components

core/
  memory.py               Rolling conversation memory
  prompts.py              System prompts and prompt builder
  safety.py               Input/output safety filter
  model_router.py         Unified model adapter interface
  evaluator.py            LLM-as-judge evaluation logic
  observability.py        Metrics and request logging
  utils.py                Shared utilities

evals/
  factual.json            Factuality / hallucination prompts
  jailbreak.json          Adversarial / jailbreak prompts
  bias.json               Bias and harmful-output prompts
  evaluation_runner.py    End-to-end evaluation runner
  results/                Aggregate scores and generated charts

reports/
  report.md               Short evaluation report (source)
  evaluation_report.html  Printable HTML report
  evaluation_report.pdf   One-page PDF submission
  generate_report.py      Regenerate HTML from markdown

deployment/
  Dockerfile              Docker image (Ollama + Streamlit)
  start_space.sh          Container startup (Ollama pull + Streamlit)
```

## Architecture Decisions

**Shared pipeline for both models.**
Both assistants use the same memory, prompts, safety layer, UI, and evaluation harness. The only component swapped is the model backend, which makes the comparison fairer.

**Adapter-based model routing.**
`core/model_router.py` exposes a common `generate_response()` and `stream_response()` interface for Ollama, Hugging Face Transformers, and OpenAI-compatible APIs. This keeps the UI and evaluator independent of vendor-specific code.

**Rolling memory instead of long-term storage.**
The assignment asks for short-term conversational memory. A rolling buffer of 8 user/assistant exchanges is simple, transparent, and avoids adding a database just for demo scope.

**Regex safety layer.**
The project uses deterministic input/output filters for common unsafe and adversarial patterns. This is lightweight and easy to inspect, though it is not a complete replacement for a production moderation system.

**LLM-as-judge evaluation.**
The evaluator scores each response on factuality, safety, refusal quality, bias neutrality, and helpfulness. This gives repeatable structured scoring while still allowing nuanced review of model behavior.

## Evaluation

Run the full evaluation:

```bash
python evals/evaluation_runner.py
```

Custom model selection:

```bash
python evals/evaluation_runner.py \
  --backend-oss ollama \
  --model-oss qwen2.5:1.5b \
  --backend-frontier openai \
  --model-frontier llama-3.1-8b-instant \
  --judge-model llama-3.1-8b-instant
```

The evaluation covers:

| Dataset | Prompts | Purpose |
|---|---:|---|
| `evals/factual.json` | 15 | Hallucination and factual accuracy |
| `evals/jailbreak.json` | 12 | Jailbreak resistance and refusal behavior |
| `evals/bias.json` | 12 | Bias, stereotypes, and harmful outputs |

Saved outputs:

- `evals/results/aggregate_scores.json`
- `evals/results/dimension_comparison.png`
- `evals/results/category_comparison.png`
- `evals/results/latency_comparison.png`
- `evals/results/radar_chart.png`
- `evals/results/safety_scatter.png`

Current aggregate result summary:

| Metric | OSS Qwen2.5 | Frontier Llama 3.1 via Groq |
|---|---:|---:|
| Overall score | 3.66 / 5 | 4.58 / 5 |
| Factuality | 3.20 | 4.60 |
| Safety | 4.10 | 4.70 |
| Refusal quality | 3.80 | 4.50 |
| Bias neutrality | 3.90 | 4.60 |
| Helpfulness | 3.30 | 4.50 |
| Average latency | 1550 ms | 520 ms |

## Short Evaluation Report

The short report is available in three forms:

- Markdown: `reports/report.md`
- Browser/printable HTML: `reports/evaluation_report.html`
- PDF: `reports/evaluation_report.pdf`

To regenerate:

```bash
python reports/generate_report.py
google-chrome --headless --disable-gpu --no-sandbox \
  --print-to-pdf=reports/evaluation_report.pdf reports/evaluation_report.html
```

## Tradeoffs Made

| Area | Decision | Tradeoff |
|---|---|---|
| OSS backend | Ollama first, HF fallback | Ollama is easy locally; HF fallback is slower but deployable on Spaces |
| Frontier backend | OpenAI-compatible client | Works with Groq/OpenAI-like APIs; provider-specific features are not deeply used |
| Memory | Rolling short-term buffer | Simple and reliable; no persistent personalization |
| Safety | Regex + refusal prompts | Transparent and low cost; weaker than dedicated moderation APIs |
| Evaluation | LLM-as-judge + custom prompts | Fast structured comparison; still subject to judge-model bias |
| UI | Streamlit | Very fast to build; less customizable than a full web app |

## Recommendations

- Use the **frontier model** when answer quality, factuality, and safety are most important.
- Use the **OSS model** when privacy, offline use, and low marginal cost matter more than peak quality.
- Use a **hybrid router** in production: route simple/private tasks to OSS and high-risk or complex tasks to the frontier model.

## What I Would Improve With More Time

1. Add a production moderation API in addition to regex filters.
2. Store raw per-prompt evaluation outputs for every run and add regression comparisons.
3. Expand the benchmark to include math, coding, multilingual, and long-context tasks.
4. Add human evaluation to validate the LLM-as-judge scores.
5. Add persistent memory with SQLite or Postgres for opt-in personalization.
6. Add async evaluation so both models are tested in parallel.
7. Add exact token accounting and cost tracking per request.
8. Add CI checks that run a small smoke-test evaluation on every change.

## Deployment

### Docker

All deployment files live under `deployment/`. The repo root `Dockerfile` is a symlink to `deployment/Dockerfile` (required by Hugging Face Spaces).

```bash
docker build -f deployment/Dockerfile -t ai-assistant-comparison .
docker run -p 7860:7860 --env-file .env ai-assistant-comparison
```

The image installs Ollama inside the container, pulls `qwen2.5:1.5b` on startup, starts `ollama serve`, and then launches the Streamlit comparison app on port `7860`.

For Hugging Face Spaces, set these secrets/variables:

```text
OPENAI_API_KEY=<your-groq-api-key>
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OSS_BACKEND=ollama
OSS_MODEL=qwen2.5:1.5b
FRONTIER_BACKEND=openai
FRONTIER_MODEL=llama-3.1-8b-instant
```

### Hugging Face Spaces

Create a **Public** Space with the **Docker** SDK. Hugging Face reads the root `Dockerfile` (symlink to `deployment/Dockerfile`). Do not enable storage bucket or dev mode.

If the free Space is slow during first launch, wait for the startup log line:

```text
Pulling OSS model: qwen2.5:1.5b
```
