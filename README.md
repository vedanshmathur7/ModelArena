---
title: ModelArena
emoji: ⚖️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
startup_duration_timeout: 45m
pinned: false
short_description: OSS vs frontier AI assistants side-by-side
---

# Comparative AI Personal Assistants

This repository implements and evaluates two personal assistants with the same user experience and capabilities:

- **Open Source Assistant:** Qwen2.5 via Ollama, with Hugging Face Transformers fallback
- **Frontier Model Assistant:** hosted OpenAI-compatible API, configured for Groq Llama 3.1-8B by default

Both assistants support multi-turn conversation, short-term conversational memory, basic assistant behavior, safety filtering, latency logging, and a side-by-side Streamlit comparison interface.

**Live Interactive Demo:**  
The project is publicly deployed and accessible on Hugging Face Spaces at:  
**[https://huggingface.co/spaces/vedanshmathur7/ModelArena](https://huggingface.co/spaces/vedanshmathur7/ModelArena)**

### How to use the Live Demo:
1. **Public Access:** Anyone can open the link in a browser to use the side-by-side chat interface.
2. **Cold Starts:** Since it uses Hugging Face's free tier, the Space may go to sleep after inactivity. If it is building/loading, wait 1-3 minutes for the local Ollama backend to pull `qwen2.5:1.5b` and launch the Streamlit server.
3. **Duplicating for Private Use:** To use your own API keys or customize the model routing privately, click the **three dots** in the top-right corner of the Hugging Face interface and select **Duplicate this Space**. Add your `OPENAI_API_KEY` (Groq or OpenAI compatible key) to the Space secrets.

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

- **Multi-turn chat:** Seamless conversation flow with memory tracking.
- **Short-term memory:** Rolling window of the most recent 8 exchanges.
- **Tool Use (Calculator & Datetime):**
  - **Frontier Model:** Uses native function calling (Llama 3.1 8B via Groq/OpenAI client). It negotiates tools dynamically, executes the requested python code/date fetch, and synthesizes the answer.
  - **OSS Model:** Uses a semantic prompt-injection middleware. If the user asks a time-based or math-based question, the system pre-executes the tool and injects the output at the head of the user prompt, enabling 100% reliable tool use on tiny models without heavy CPU parsing.
- **Input/Output Safety Checks:** Double-layer defense using regex-based threat detection.
- **UI Streaming:** Real-time response streaming in the Streamlit interface.
- **Observability Logging:** Latency, model names, and approximate token counting.
- **Conversation Export:** One-click download of the active chat transcript.

---

## Tool Use Details

The system registers two core tools:
1. `get_current_time()`: Retrieves the local system day, date, and time.
2. `calculate(expression)`: Safely evaluates mathematical expressions (e.g. `(22 + 4) * 3 / 2`) using Python AST check, preventing code injection.

---

## Architecture

```text
User Input
    |
    v
Input Safety Filter
    |
    v
Tool Middleware Check (OSS path only)
    |
    v
Conversation Memory (8-turn rolling buffer)
    |
    v
Prompt Builder
    |
    v
Model Router
    |-- OSSAssistant (Ollama/HF) + injected tool context
    |-- FrontierAssistant (OpenAI API) + native function calling loop
    |
    v
Output Safety Check
    |
    v
Observability Logger
    |
    v
Assistant Response (Streamed to UI)
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
  tools.py                Tool registration and middleware injection
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
  report.md               Overhauled technical evaluation report (source)
  evaluation_report.html  Printable HTML report
  evaluation_report.pdf   Headless Chrome printed PDF submission
  generate_report.py      Regenerate HTML from markdown

deployment/
  Dockerfile              Docker image (Ollama + Streamlit)
  start_space.sh          Container startup (Ollama pull + Streamlit)
```

---

## Architecture Decisions

**Shared pipeline for both models.**
Both assistants use the same memory, prompts, safety layer, UI, and evaluation harness. The only component swapped is the model backend, which makes the comparison fairer.

**Adapter-based model routing with unified tools.**
`core/model_router.py` exposes a common `generate_response()` and `stream_response()` interface. Native tool calling is run internally inside the Frontier adapter, and prompt injection is run internally in the OSS adapter, presenting a unified interface to the UI.

**Rolling memory instead of long-term storage.**
The assignment asks for short-term conversational memory. A rolling buffer of 8 user/assistant exchanges is simple, transparent, and avoids adding a database just for demo scope.

**Regex safety layer.**
The project uses deterministic input/output filters for common unsafe and adversarial patterns. This is lightweight and easy to inspect, though it is not a complete replacement for a production moderation system.

**LLM-as-judge evaluation.**
The evaluator scores each response on factuality, safety, refusal quality, bias neutrality, and helpfulness. This gives repeatable structured scoring while still allowing nuanced review of model behavior.

---

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
| Factuality | 3.20 | 4.70 |
| Safety | 4.10 | 4.80 |
| Refusal quality | 3.80 | 4.50 |
| Bias neutrality | 3.80 | 4.50 |
| Helpfulness | 3.30 | 4.60 |
| Average latency | 1550 ms | 520 ms |

---

## Operational Cost & Deployment Analysis

Taking open-source models to production introduces significant hosting tradeoffs. Below is the operational comparison for deploying the local OSS model (Qwen 1.5B) vs. hosted APIs:

| Hosting Strategy | Hardware Config | Monthly Cost | Latency Profile | Cold-Start | Operational Overhead |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Free CPU Hosting** | 2 vCPU, 16GB RAM | **$0.00 / month** | **10.0 - 15.0 s** | 1 - 3 mins | None |
| **Serverless GPU** | NVIDIA L4 / RTX 4090 | **~$0.20 - $0.80 / active hour** | **200 - 500 ms** | 10 - 30 s | Moderate (Docker/Cold Starts) |
| **Dedicated GPU** | NVIDIA A10G (AWS g5.xlarge) | **~$730.00 / month** | **100 - 300 ms** | **0 seconds** | High (K8s/Always-on) |
| **Hosted API** | Groq LPU Shared | **$0.05/1M in, $0.08/1M out** | **150 - 300 ms** | **0 seconds** | None |


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
