"""
Frontier Assistant — Streamlit app backed by Groq API (Llama 3.1-8B-Instant).

Run with:
    streamlit run apps/frontier_assistant.py
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
from core.prompts import FRONTIER_SYSTEM_PROMPT, PromptBuilder
from core.safety import SafetyFilter
from core.utils import setup_logging

setup_logging()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FRONTIER_BACKEND = os.getenv("FRONTIER_BACKEND", "openai")
FRONTIER_MODEL = os.getenv("FRONTIER_MODEL", "llama-3.1-8b-instant")
MAX_EXCHANGES = int(os.getenv("MAX_EXCHANGES", "8"))

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

configure_page("Frontier Assistant — Llama 3.1-8B (Groq)", icon="🚀")

st.title("🚀 Frontier Assistant")
st.caption(f"Powered by **{FRONTIER_MODEL}** via **{FRONTIER_BACKEND} API**")

# ---------------------------------------------------------------------------
# API key check
# ---------------------------------------------------------------------------

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    st.warning(
        "⚠️ **OPENAI_API_KEY not set.** "
        "Add it to your `.env` file or set it as an environment variable. "
        "The assistant will not work without it."
    )

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "frontier_memory" not in st.session_state:
    st.session_state.frontier_memory = ConversationMemory(
        max_exchanges=MAX_EXCHANGES,
        system_prompt=FRONTIER_SYSTEM_PROMPT,
    )

if "frontier_obs" not in st.session_state:
    st.session_state.frontier_obs = ObservabilityManager()

if "frontier_safety" not in st.session_state:
    st.session_state.frontier_safety = SafetyFilter(strict_mode=True)

if "frontier_assistant" not in st.session_state:
    try:
        st.session_state.frontier_assistant = create_assistant(
            backend=FRONTIER_BACKEND,
            model_name=FRONTIER_MODEL,
        )
    except Exception as e:
        st.error(f"Failed to initialize Frontier assistant: {e}")
        st.stop()

memory: ConversationMemory = st.session_state.frontier_memory
obs_manager: ObservabilityManager = st.session_state.frontier_obs
safety_filter: SafetyFilter = st.session_state.frontier_safety
assistant = st.session_state.frontier_assistant

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

settings = render_sidebar_info(
    model_name=FRONTIER_MODEL,
    backend=FRONTIER_BACKEND,
    memory=memory,
    obs_manager=obs_manager,
    safety_filter=safety_filter,
)

safety_filter.strict_mode = settings["strict_mode"]

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

tab_chat, tab_obs = st.tabs(["💬 Chat", "📊 Observability"])

with tab_chat:
    render_chat_history(memory)

    if user_input := st.chat_input("Ask me anything..."):
        with st.chat_message("user"):
            st.markdown(user_input)

        prompt_builder = PromptBuilder(system_prompt=FRONTIER_SYSTEM_PROMPT)

        handle_chat_turn(
            user_input=user_input,
            memory=memory,
            prompt_builder=prompt_builder,
            assistant=assistant,
            safety_filter=safety_filter,
            obs_manager=obs_manager,
            backend=FRONTIER_BACKEND,
            stream=True,
        )

with tab_obs:
    render_observability_dashboard(obs_manager)
