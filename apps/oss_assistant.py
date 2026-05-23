"""
OSS Assistant — Streamlit app backed by Ollama (Qwen2.5 / Phi-3 / Llama3.2).

Run with:
    streamlit run apps/oss_assistant.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st

from apps.shared_ui import (
    configure_page,
    handle_chat_turn,
    render_chat_history,
    render_observability_dashboard,
    render_sidebar_info,
)
from core.memory import ConversationMemory
from core.model_router import create_assistant
from core.observability import ObservabilityManager
from core.prompts import OSS_SYSTEM_PROMPT, PromptBuilder
from core.safety import SafetyFilter
from core.utils import setup_logging

setup_logging()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OSS_BACKEND = os.getenv("OSS_BACKEND", "ollama")
OSS_MODEL = os.getenv("OSS_MODEL", "qwen2.5:1.5b")
MAX_EXCHANGES = int(os.getenv("MAX_EXCHANGES", "8"))

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

configure_page("OSS Assistant — Qwen2.5", icon="🦙")

st.title("🦙 OSS Assistant")
st.caption(f"Powered by **{OSS_MODEL}** via **{OSS_BACKEND}**")

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "oss_memory" not in st.session_state:
    st.session_state.oss_memory = ConversationMemory(
        max_exchanges=MAX_EXCHANGES,
        system_prompt=OSS_SYSTEM_PROMPT,
    )

if "oss_obs" not in st.session_state:
    st.session_state.oss_obs = ObservabilityManager()

if "oss_safety" not in st.session_state:
    st.session_state.oss_safety = SafetyFilter(strict_mode=True)

if "oss_assistant" not in st.session_state:
    with st.spinner(f"Loading {OSS_MODEL}..."):
        try:
            st.session_state.oss_assistant = create_assistant(
                backend=OSS_BACKEND,
                model_name=OSS_MODEL,
            )
        except Exception as e:
            st.error(f"Failed to initialize OSS assistant: {e}")
            st.stop()

memory: ConversationMemory = st.session_state.oss_memory
obs_manager: ObservabilityManager = st.session_state.oss_obs
safety_filter: SafetyFilter = st.session_state.oss_safety
assistant = st.session_state.oss_assistant

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

settings = render_sidebar_info(
    model_name=OSS_MODEL,
    backend=OSS_BACKEND,
    memory=memory,
    obs_manager=obs_manager,
    safety_filter=safety_filter,
)

# Apply safety mode setting
safety_filter.strict_mode = settings["strict_mode"]

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

tab_chat, tab_obs = st.tabs(["💬 Chat", "📊 Observability"])

with tab_chat:
    # Render existing conversation
    render_chat_history(memory)

    # Chat input
    if user_input := st.chat_input("Ask me anything..."):
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)

        prompt_builder = PromptBuilder(system_prompt=OSS_SYSTEM_PROMPT)

        handle_chat_turn(
            user_input=user_input,
            memory=memory,
            prompt_builder=prompt_builder,
            assistant=assistant,
            safety_filter=safety_filter,
            obs_manager=obs_manager,
            backend=OSS_BACKEND,
            stream=True,
        )

with tab_obs:
    render_observability_dashboard(obs_manager)
