"""
Shared UI components used by both assistant apps.
Keeps the individual app files thin and DRY.
"""

from __future__ import annotations

import json
import time
from typing import Callable, Generator, List, Optional

import streamlit as st

from core.memory import ConversationMemory
from core.observability import ObservabilityManager
from core.safety import SafetyFilter, SafetyResult


# ---------------------------------------------------------------------------
# Page config helper
# ---------------------------------------------------------------------------

def configure_page(title: str, icon: str = "🤖") -> None:
    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

# ---------------------------------------------------------------------------
# Sidebar components
# ---------------------------------------------------------------------------

def render_sidebar_info(
    model_name: str,
    backend: str,
    memory: ConversationMemory,
    obs_manager: ObservabilityManager,
    safety_filter: SafetyFilter,
) -> dict:
    """
    Render the sidebar with model info, memory controls, and observability stats.
    Returns a dict of user-configured settings.
    """
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        st.markdown(f"**Model:** `{model_name}`")
        st.markdown(f"**Backend:** `{backend}`")
        st.divider()

        # Memory controls
        st.markdown("### 🧠 Memory")
        st.markdown(f"Conversation turns: **{memory.turn_count}**")
        if st.button("🗑️ Clear Memory", use_container_width=True):
            memory.clear()
            st.success("Memory cleared.")
            st.rerun()

        st.divider()

        # Safety mode
        st.markdown("### 🛡️ Safety")
        strict_mode = st.toggle("Strict safety mode", value=True)
        st.markdown(f"Refusals this session: **{safety_filter.refusal_count}**")

        st.divider()

        # Observability summary
        st.markdown("### 📊 Session Stats")
        summary = obs_manager.get_summary()
        if summary:
            st.metric("Total Requests", summary.get("total_requests", 0))
            st.metric("Avg Latency", f"{summary.get('avg_latency_ms', 0):.0f}ms")
            st.metric("Refusal Rate", f"{summary.get('refusal_rate', 0)*100:.1f}%")
        else:
            st.caption("No requests yet.")

        st.divider()

        # Export
        st.markdown("### 💾 Export")
        if st.button("📥 Download Chat Log", use_container_width=True):
            logs = obs_manager.get_all_logs()
            st.download_button(
                label="Save logs.json",
                data=json.dumps(logs, indent=2),
                file_name="chat_logs.json",
                mime="application/json",
            )

        # Chat history export
        history = memory.history
        if history:
            chat_export = "\n\n".join(
                f"**{m.role.upper()}**: {m.content}" for m in history
            )
            st.download_button(
                label="Save Chat History",
                data=chat_export,
                file_name="chat_history.txt",
                mime="text/plain",
                use_container_width=True,
            )

    return {"strict_mode": strict_mode}


# ---------------------------------------------------------------------------
# Chat message rendering
# ---------------------------------------------------------------------------

def render_chat_history(memory: ConversationMemory) -> None:
    """Render all messages in the conversation history."""
    for msg in memory.history:
        with st.chat_message(msg.role):
            st.markdown(msg.content)


def render_metrics_row(
    latency_ms: float,
    prompt_tokens: int,
    response_tokens: int,
    was_refused: bool,
) -> None:
    """Render a compact metrics row below a response."""
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⏱️ Latency", f"{latency_ms:.0f}ms")
    col2.metric("📥 Prompt Tokens", f"~{prompt_tokens}")
    col3.metric("📤 Response Tokens", f"~{response_tokens}")
    col4.metric("🚫 Refused", "Yes" if was_refused else "No")


# ---------------------------------------------------------------------------
# Core chat handler
# ---------------------------------------------------------------------------

def handle_chat_turn(
    user_input: str,
    memory: ConversationMemory,
    prompt_builder,
    assistant,
    safety_filter: SafetyFilter,
    obs_manager: ObservabilityManager,
    backend: str,
    stream: bool = True,
) -> None:
    """
    Execute one full chat turn:
      1. Input safety check
      2. Build prompt
      3. Generate response (streaming or not)
      4. Output safety check
      5. Update memory
      6. Log to observability
    """
    from core.observability import estimate_tokens, Timer

    # --- Input safety check ---
    input_check: SafetyResult = safety_filter.check_input(user_input)
    if not input_check.is_safe:
        with st.chat_message("assistant"):
            st.warning(f"🛡️ **Safety Filter Triggered** ({input_check.violation_type.value})")
            st.markdown(input_check.refusal_message)
        obs_manager.log_request(
            model_used=assistant.model_name,
            backend=backend,
            prompt=user_input,
            response=input_check.refusal_message,
            latency_ms=0.0,
            was_refused=True,
            violation_type=input_check.violation_type.value,
        )
        return

    # --- Build prompt ---
    messages = prompt_builder.build(memory, user_input)
    prompt_str = " ".join(m["content"] for m in messages)

    # --- Generate response ---
    response_text = ""
    with st.chat_message("assistant"):
        if stream:
            response_placeholder = st.empty()
            with Timer() as timer:
                for chunk in assistant.stream_response(messages):
                    response_text += chunk
                    response_placeholder.markdown(response_text + "▌")
            response_placeholder.markdown(response_text)
        else:
            with Timer() as timer:
                response_text = assistant.generate_response(messages)
            st.markdown(response_text)

    latency_ms = timer.elapsed_ms

    # --- Output safety check ---
    output_check: SafetyResult = safety_filter.check_output(response_text)
    if not output_check.is_safe:
        response_text = output_check.refusal_message
        with st.chat_message("assistant"):
            st.warning(f"🛡️ **Output filtered** ({output_check.violation_type.value})")
            st.markdown(response_text)

    # --- Update memory ---
    memory.add_user_message(user_input)
    memory.add_assistant_message(response_text)

    # --- Observability ---
    log = obs_manager.log_request(
        model_used=assistant.model_name,
        backend=backend,
        prompt=prompt_str,
        response=response_text,
        latency_ms=latency_ms,
        was_refused=not output_check.is_safe,
        violation_type=output_check.violation_type.value if not output_check.is_safe else None,
    )

    # Show metrics
    with st.expander("📊 Request metrics", expanded=False):
        render_metrics_row(
            latency_ms=log.latency_ms,
            prompt_tokens=log.estimated_prompt_tokens,
            response_tokens=log.estimated_response_tokens,
            was_refused=log.was_refused,
        )


# ---------------------------------------------------------------------------
# Observability dashboard
# ---------------------------------------------------------------------------

def render_observability_dashboard(obs_manager: ObservabilityManager) -> None:
    """Render a full observability dashboard in a Streamlit expander."""
    import pandas as pd

    summary = obs_manager.get_summary()
    logs = obs_manager.get_all_logs()

    st.markdown("## 📊 Observability Dashboard")

    if not logs:
        st.info("No requests logged yet. Start chatting to see metrics.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Requests", summary.get("total_requests", 0))
    col2.metric("Avg Latency", f"{summary.get('avg_latency_ms', 0):.0f}ms")
    col3.metric("Refusal Rate", f"{summary.get('refusal_rate', 0)*100:.1f}%")
    col4.metric("Total Tokens (est.)", f"{summary.get('total_estimated_tokens', 0):,}")

    # Per-backend breakdown
    by_backend = summary.get("by_backend", {})
    if by_backend:
        st.markdown("### By Backend")
        df_backend = pd.DataFrame(by_backend).T.reset_index()
        df_backend.columns = ["Backend", "Requests", "Avg Latency (ms)", "Refusals"]
        st.dataframe(df_backend, use_container_width=True)

    # Recent logs table
    st.markdown("### Recent Requests")
    df = pd.DataFrame(logs[-20:])  # Last 20
    if not df.empty:
        display_cols = [
            "timestamp", "backend", "model_used",
            "latency_ms", "estimated_prompt_tokens",
            "estimated_response_tokens", "was_refused",
        ]
        display_cols = [c for c in display_cols if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True)
