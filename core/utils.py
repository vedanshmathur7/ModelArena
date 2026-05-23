"""
Shared utility functions used across the project.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a clean format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def get_env(key: str, default: Optional[str] = None, required: bool = False) -> Optional[str]:
    """
    Retrieve an environment variable with optional validation.

    Args:
        key: Environment variable name
        default: Default value if not set
        required: Raise ValueError if not set and no default
    """
    value = os.getenv(key, default)
    if required and not value:
        raise ValueError(
            f"Required environment variable '{key}' is not set. "
            f"Check your .env file."
        )
    return value


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def load_json(path: Path | str) -> Any:
    """Load and return JSON from a file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, path: Path | str, indent: int = 2) -> None:
    """Save data as JSON to a file, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def truncate(text: str, max_chars: int = 200, suffix: str = "...") -> str:
    """Truncate text to max_chars, appending suffix if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)] + suffix


def count_words(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: Path | str = ".env") -> Dict[str, str]:
    """
    Load key=value pairs from a .env file into a dict.
    Does NOT set environment variables — use python-dotenv for that.
    """
    config: Dict[str, str] = {}
    path = Path(config_path)
    if not path.exists():
        return config
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                config[key.strip()] = value.strip().strip('"').strip("'")
    return config


# ---------------------------------------------------------------------------
# Eval dataset helpers
# ---------------------------------------------------------------------------

def load_eval_dataset(paths: List[Path | str]) -> List[dict]:
    """Load and merge multiple eval JSON files into a single list."""
    combined: List[dict] = []
    for path in paths:
        try:
            data = load_json(path)
            if isinstance(data, list):
                combined.extend(data)
            elif isinstance(data, dict) and "prompts" in data:
                combined.extend(data["prompts"])
        except Exception as e:
            logging.getLogger(__name__).warning("Could not load eval file %s: %s", path, e)
    return combined


def format_cost_table(
    oss_tokens: int,
    frontier_tokens: int,
    oss_cost_per_1k: float = 0.0,
    frontier_cost_per_1k: float = 0.0006,
) -> str:
    """
    Format a simple cost comparison table as a markdown string.

    Default costs:
      - OSS (self-hosted): $0.00 per 1k tokens
      - Llama 3.1-8B via Groq: $0.00 (free tier) / ~$0.00005 per 1k tokens (paid)
    """
    oss_cost = (oss_tokens / 1000) * oss_cost_per_1k
    frontier_cost = (frontier_tokens / 1000) * frontier_cost_per_1k

    return (
        f"| Model       | Tokens | Cost (USD) |\n"
        f"|-------------|--------|------------|\n"
        f"| OSS (local) | {oss_tokens:,} | ${oss_cost:.4f} |\n"
        f"| Frontier    | {frontier_tokens:,} | ${frontier_cost:.4f} |\n"
    )
