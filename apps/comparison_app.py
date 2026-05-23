"""
Side-by-side comparison app — runs both assistants simultaneously.

Run with:
    streamlit run apps/comparison_app.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

from core.memory import ConversationMemory
from core.model_router import create_assistant
from core.observability import ObservabilityManager, Timer, estimate_tokens
from core.prompts import FRONTIER_SYSTEM_PROMPT, OSS_SYSTEM_PROMPT, PromptBuilder
from core.safety import SafetyFilter
from core.utils import setup_logging

setup_logging()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OSS_BACKEND = os.getenv("OSS_BACKEND", "ollama")
OSS_MODEL = os.getenv("OSS_MODEL", "qwen2.5:1.5b")
FRONTIER_BACKEND = os.getenv("FRONTIER_BACKEND", "openai")
FRONTIER_MODEL = os.getenv("FRONTIER_MODEL", "llama-3.1-8b-instant")
MAX_EXCHANGES = int(os.getenv("MAX_EXCHANGES", "8"))

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Assistant Comparison",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Comparative AI Assistant")
st.caption("Side-by-side evaluation of OSS vs Frontier models")

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state():
    if "oss_memory" not in st.session_state:
        st.session_state.oss_memory = ConversationMemory(MAX_EXCHANGES, OSS_SYSTEM_PROMPT)
    if "frontier_memory" not in st.session_state:
        st.session_state.frontier_memory = ConversationMemory(MAX_EXCHANGES, FRONTIER_SYSTEM_PROMPT)
    if "obs" not in st.session_state:
        st.session_state.obs = ObservabilityManager()
    if "safety" not in st.session_state:
        st.session_state.safety = SafetyFilter(strict_mode=True)
    if "chat_log" not in st.session_state:
        st.session_state.chat_log = []  # List of {user, oss, frontier, metrics}
    if "oss_assistant" not in st.session_state:
        try:
            st.session_state.oss_assistant = create_assistant(OSS_BACKEND, OSS_MODEL)
        except Exception as e:
            st.session_state.oss_assistant = None
            st.session_state.oss_error = str(e)
    if "frontier_assistant" not in st.session_state:
        try:
            st.session_state.frontier_assistant = create_assistant(FRONTIER_BACKEND, FRONTIER_MODEL)
        except Exception as e:
            st.session_state.frontier_assistant = None
            st.session_state.frontier_error = str(e)


_init_state()

oss_memory: ConversationMemory = st.session_state.oss_memory
frontier_memory: ConversationMemory = st.session_state.frontier_memory
obs: ObservabilityManager = st.session_state.obs
safety: SafetyFilter = st.session_state.safety
chat_log: list = st.session_state.chat_log

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Settings")
    strict_mode = st.toggle("Strict safety mode", value=True)
    safety.strict_mode = strict_mode

    st.divider()
    st.markdown("### Models")
    st.markdown(f"**OSS:** `{OSS_MODEL}`")
    st.markdown(f"**Frontier:** `{FRONTIER_MODEL}`")

    st.divider()
    st.markdown("### Session")
    col1, col2 = st.columns(2)
    if col1.button("Clear", use_container_width=True):
        oss_memory.clear()
        frontier_memory.clear()
        st.session_state.chat_log = []
        st.rerun()

    if col2.button("Export", use_container_width=True):
        export_data = json.dumps(chat_log, indent=2)
        st.download_button(
            "Download JSON",
            data=export_data,
            file_name="comparison_log.json",
            mime="application/json",
        )

    st.divider()
    summary = obs.get_summary()
    if summary:
        st.markdown("### Stats")
        by_b = summary.get("by_backend", {})
        for b, data in by_b.items():
            st.markdown(f"**{b}**")
            st.markdown(f"  Avg latency: {data['avg_latency_ms']:.0f}ms")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_chat, tab_eval, tab_obs = st.tabs(["Chat", "Eval Results", "Observability"])

# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------

with tab_chat:
    # Render existing conversation
    for turn in chat_log:
        with st.chat_message("user"):
            st.markdown(turn["user"])

        col_oss, col_frontier = st.columns(2)
        with col_oss:
            st.markdown(f"**OSS — {OSS_MODEL}**")
            st.markdown(turn.get("oss", ""))
            if "oss_latency" in turn:
                st.caption(f"{turn['oss_latency']:.0f}ms | ~{turn.get('oss_tokens', 0)} tokens")

        with col_frontier:
            st.markdown(f"**Frontier — {FRONTIER_MODEL}**")
            st.markdown(turn.get("frontier", ""))
            if "frontier_latency" in turn:
                st.caption(f"{turn['frontier_latency']:.0f}ms | ~{turn.get('frontier_tokens', 0)} tokens")

    # Chat input
    if user_input := st.chat_input("Ask both assistants..."):
        with st.chat_message("user"):
            st.markdown(user_input)

        # Safety check
        input_check = safety.check_input(user_input)
        if not input_check.is_safe:
            st.warning(f"Safety filter: {input_check.violation_type.value}")
            st.markdown(input_check.refusal_message)
        else:
            col_oss, col_frontier = st.columns(2)

            oss_response = ""
            frontier_response = ""
            oss_latency = 0.0
            frontier_latency = 0.0

            oss_builder = PromptBuilder(OSS_SYSTEM_PROMPT)
            frontier_builder = PromptBuilder(FRONTIER_SYSTEM_PROMPT)

            oss_messages = oss_builder.build(oss_memory, user_input)
            frontier_messages = frontier_builder.build(frontier_memory, user_input)

            # OSS response
            with col_oss:
                st.markdown(f"**OSS — {OSS_MODEL}**")
                oss_placeholder = st.empty()
                oss_assistant = st.session_state.oss_assistant

                if oss_assistant is None:
                    oss_response = f"[OSS unavailable: {st.session_state.get('oss_error', 'unknown error')}]"
                    oss_placeholder.markdown(oss_response)
                else:
                    with Timer() as t:
                        for chunk in oss_assistant.stream_response(oss_messages):
                            oss_response += chunk
                            oss_placeholder.markdown(oss_response + "▌")
                    oss_placeholder.markdown(oss_response)
                    oss_latency = t.elapsed_ms
                    st.caption(f"{oss_latency:.0f}ms | ~{estimate_tokens(oss_response)} tokens")

            # Frontier response
            with col_frontier:
                st.markdown(f"**Frontier — {FRONTIER_MODEL}**")
                frontier_placeholder = st.empty()
                frontier_assistant = st.session_state.frontier_assistant

                if frontier_assistant is None:
                    frontier_response = f"[Frontier unavailable: {st.session_state.get('frontier_error', 'unknown error')}]"
                    frontier_placeholder.markdown(frontier_response)
                else:
                    with Timer() as t:
                        for chunk in frontier_assistant.stream_response(frontier_messages):
                            frontier_response += chunk
                            frontier_placeholder.markdown(frontier_response + "▌")
                    frontier_placeholder.markdown(frontier_response)
                    frontier_latency = t.elapsed_ms
                    st.caption(f"{frontier_latency:.0f}ms | ~{estimate_tokens(frontier_response)} tokens")

            # Update memories
            oss_memory.add_user_message(user_input)
            oss_memory.add_assistant_message(oss_response)
            frontier_memory.add_user_message(user_input)
            frontier_memory.add_assistant_message(frontier_response)

            # Log
            obs.log_request(
                model_used=OSS_MODEL, backend=OSS_BACKEND,
                prompt=user_input, response=oss_response,
                latency_ms=oss_latency,
            )
            obs.log_request(
                model_used=FRONTIER_MODEL, backend=FRONTIER_BACKEND,
                prompt=user_input, response=frontier_response,
                latency_ms=frontier_latency,
            )

            # Append to chat log
            chat_log.append({
                "user": user_input,
                "oss": oss_response,
                "frontier": frontier_response,
                "oss_latency": oss_latency,
                "frontier_latency": frontier_latency,
                "oss_tokens": estimate_tokens(oss_response),
                "frontier_tokens": estimate_tokens(frontier_response),
            })

# ---------------------------------------------------------------------------
# Eval Results tab
# ---------------------------------------------------------------------------

with tab_eval:
    st.markdown("## Evaluation Results")

    import pandas as pd

    results_dir = Path("evals/results")

    # --- Aggregate scores (always show if available) ---
    agg_file = results_dir / "aggregate_scores.json"
    if agg_file.exists():
        with open(agg_file) as f:
            agg = json.load(f)

        st.markdown("### Aggregate Scores")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("OSS Overall", f"{agg.get('oss_overall', 0):.2f} / 5.0")
        col2.metric("Frontier Overall", f"{agg.get('frontier_overall', 0):.2f} / 5.0")
        col3.metric("OSS Avg Latency", f"{agg.get('avg_oss_latency_ms', 0):.0f}ms")
        col4.metric("Frontier Avg Latency", f"{agg.get('avg_frontier_latency_ms', 0):.0f}ms")

        # Per-dimension breakdown table
        oss_avg = agg.get("oss_avg", {})
        frontier_avg = agg.get("frontier_avg", {})
        if oss_avg and frontier_avg:
            st.markdown("### Scores by Dimension")
            dim_rows = []
            for dim in oss_avg:
                dim_rows.append({
                    "Dimension": dim.replace("_", " ").title(),
                    "OSS": round(oss_avg[dim], 2),
                    "Frontier": round(frontier_avg.get(dim, 0), 2),
                    "Delta": round(frontier_avg.get(dim, 0) - oss_avg[dim], 2),
                })
            st.dataframe(pd.DataFrame(dim_rows), use_container_width=True, hide_index=True)

        # Per-category breakdown
        by_cat = agg.get("by_category", {})
        if by_cat:
            st.markdown("### Scores by Category")
            cat_rows = []
            for cat, data in by_cat.items():
                cat_rows.append({
                    "Category": cat.title(),
                    "Prompts": data.get("count", 0),
                    "OSS Overall": round(sum(data["oss_avg"].values()) / len(data["oss_avg"]), 2),
                    "Frontier Overall": round(sum(data["frontier_avg"].values()) / len(data["frontier_avg"]), 2),
                    "OSS Latency (ms)": data.get("avg_oss_latency_ms", 0),
                    "Frontier Latency (ms)": data.get("avg_frontier_latency_ms", 0),
                })
            st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)
    else:
        st.info("aggregate_scores.json not found in evals/results/.")

    # --- Charts ---
    chart_files = sorted(results_dir.glob("*.png"))
    if chart_files:
        st.markdown("### Charts")
        cols = st.columns(2)
        for i, chart in enumerate(chart_files):
            with cols[i % 2]:
                st.image(str(chart), caption=chart.stem.replace("_", " ").title())

    # --- Per-prompt results (only if the detailed JSON exists) ---
    result_files = sorted(results_dir.glob("eval_results_*.json"), reverse=True)
    if result_files:
        st.markdown("### Per-Prompt Results")
        selected_file = st.selectbox(
            "Select results file",
            options=result_files,
            format_func=lambda p: p.name,
        )
        with open(selected_file) as f:
            results = json.load(f)
        rows = []
        for r in results:
            rows.append({
                "ID": r["prompt_id"],
                "Category": r["category"],
                "Prompt": r["prompt"][:60] + "...",
                "OSS Overall": round(r["oss_score"]["overall"], 2),
                "Frontier Overall": round(r["frontier_score"]["overall"], 2),
                "OSS Latency (ms)": r["oss_latency_ms"],
                "Frontier Latency (ms)": r["frontier_latency_ms"],
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # --- Operational Cost & Latency comparison ---
    st.markdown("### Deployment Costs & Latency Analysis")
    st.markdown(
        "When taking these assistants to production, you have several deployment paths for the Open Source Model. "
        "Here is an operational cost-latency comparison table for running **Qwen2.5 (1.5B)** vs. the **Frontier API**:"
    )
    cost_data = [
        {
            "Deployment Tier": "Hugging Face Spaces (CPU Basic)",
            "Hardware Resources": "2 vCPU, 16GB RAM (Shared)",
            "Running Cost ($/mo)": "$0.00 (Free Tier)",
            "Avg. Latency (Qwen 1.5B)": "10.0 - 15.0 seconds",
            "Cold Start": "1 - 3 minutes",
            "Recommendation / Use Case": "Hobbyist demos, low-concurrency testing. Unsuitable for production."
        },
        {
            "Deployment Tier": "Serverless GPU (RunPod / Modal)",
            "Hardware Resources": "NVIDIA L4 / RTX 4090 (24GB VRAM)",
            "Running Cost ($/mo)": "~$0.20 - $0.80 per active hour",
            "Avg. Latency (Qwen 1.5B)": "200 - 500 ms",
            "Cold Start": "10 - 30 seconds",
            "Recommendation / Use Case": "Variable production traffic. Extremely cost-effective for medium workloads."
        },
        {
            "Deployment Tier": "Dedicated Cloud GPU (AWS/GCP)",
            "Hardware Resources": "NVIDIA A10G (g5.xlarge, 24GB VRAM)",
            "Running Cost ($/mo)": "~$730.00 / month (Flat-rate)",
            "Avg. Latency (Qwen 1.5B)": "100 - 300 ms",
            "Cold Start": "0 seconds (Always-on)",
            "Recommendation / Use Case": "High, steady-state production traffic where serverless cold starts are unacceptable."
        },
        {
            "Deployment Tier": "Hosted Frontier API (Groq LPU)",
            "Hardware Resources": "Shared LPU Infrastructure",
            "Running Cost ($/mo)": "$0.05/1M prompt, $0.08/1M completion tokens",
            "Avg. Latency (Llama 3.1 8B)": "150 - 300 ms",
            "Cold Start": "0 seconds",
            "Recommendation / Use Case": "Most cost-effective starting point. Scale-to-millions without server management."
        }
    ]
    st.dataframe(pd.DataFrame(cost_data), use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Observability tab
# ---------------------------------------------------------------------------

with tab_obs:
    from apps.shared_ui import render_observability_dashboard
    render_observability_dashboard(obs)
