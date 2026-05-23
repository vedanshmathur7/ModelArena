"""
Lightweight observability layer.

Tracks per-request metrics and persists them to logs/logs.json.
Metrics collected:
  - timestamp
  - model_used
  - backend
  - prompt_length (chars)
  - response_length (chars)
  - estimated_prompt_tokens
  - estimated_response_tokens
  - latency_ms
  - was_refused (safety refusal)
  - violation_type (if refused)
  - session_id
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

LOGS_PATH = Path("logs/logs.json")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RequestLog:
    session_id: str
    model_used: str
    backend: str
    prompt_length: int
    response_length: int
    estimated_prompt_tokens: int
    estimated_response_tokens: int
    latency_ms: float
    was_refused: bool
    violation_type: Optional[str]
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """
    Rough token estimate: ~4 characters per token (standard approximation).
    Good enough for observability; not a replacement for a proper tokenizer.
    """
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Observability manager
# ---------------------------------------------------------------------------

class ObservabilityManager:
    """
    Collects and persists request logs.
    Thread-safe for single-process Streamlit apps.
    """

    def __init__(self, log_path: Path = LOGS_PATH):
        self.log_path = log_path
        self._logs: List[RequestLog] = []
        self._session_id: str = str(uuid.uuid4())[:8]
        self._ensure_log_dir()
        self._load_existing_logs()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_request(
        self,
        model_used: str,
        backend: str,
        prompt: str,
        response: str,
        latency_ms: float,
        was_refused: bool = False,
        violation_type: Optional[str] = None,
    ) -> RequestLog:
        """Record a single request/response pair."""
        log = RequestLog(
            session_id=self._session_id,
            model_used=model_used,
            backend=backend,
            prompt_length=len(prompt),
            response_length=len(response),
            estimated_prompt_tokens=estimate_tokens(prompt),
            estimated_response_tokens=estimate_tokens(response),
            latency_ms=round(latency_ms, 2),
            was_refused=was_refused,
            violation_type=violation_type,
        )
        self._logs.append(log)
        self._persist(log)
        return log

    def get_all_logs(self) -> List[dict]:
        return [log.to_dict() for log in self._logs]

    def get_summary(self) -> dict:
        """Aggregate statistics across all logged requests."""
        if not self._logs:
            return {}

        total = len(self._logs)
        refused = sum(1 for l in self._logs if l.was_refused)
        latencies = [l.latency_ms for l in self._logs if not l.was_refused]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        total_tokens = sum(
            l.estimated_prompt_tokens + l.estimated_response_tokens
            for l in self._logs
        )

        by_backend: dict = {}
        for log in self._logs:
            b = log.backend
            if b not in by_backend:
                by_backend[b] = {"count": 0, "latencies": [], "refused": 0}
            by_backend[b]["count"] += 1
            by_backend[b]["latencies"].append(log.latency_ms)
            if log.was_refused:
                by_backend[b]["refused"] += 1

        backend_summary = {}
        for b, data in by_backend.items():
            lats = data["latencies"]
            backend_summary[b] = {
                "requests": data["count"],
                "avg_latency_ms": round(sum(lats) / len(lats), 2) if lats else 0,
                "refusals": data["refused"],
            }

        return {
            "total_requests": total,
            "total_refusals": refused,
            "refusal_rate": round(refused / total, 3) if total else 0,
            "avg_latency_ms": round(avg_latency, 2),
            "total_estimated_tokens": total_tokens,
            "by_backend": backend_summary,
        }

    def new_session(self) -> None:
        """Start a new session (new session_id)."""
        self._session_id = str(uuid.uuid4())[:8]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_log_dir(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _load_existing_logs(self) -> None:
        """Load existing logs from disk on startup."""
        if not self.log_path.exists():
            return
        try:
            with open(self.log_path, "r") as f:
                raw = json.load(f)
            for entry in raw:
                try:
                    self._logs.append(RequestLog(**entry))
                except Exception:
                    pass  # Skip malformed entries
        except Exception as exc:
            logger.warning("Could not load existing logs: %s", exc)

    def _persist(self, log: RequestLog) -> None:
        """Append a single log entry to the JSON file."""
        try:
            existing: List[dict] = []
            if self.log_path.exists():
                with open(self.log_path, "r") as f:
                    existing = json.load(f)
            existing.append(log.to_dict())
            with open(self.log_path, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as exc:
            logger.error("Failed to persist log: %s", exc)


# ---------------------------------------------------------------------------
# Context manager for timing
# ---------------------------------------------------------------------------

class Timer:
    """Simple context manager for measuring elapsed time in milliseconds."""

    def __init__(self):
        self.elapsed_ms: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
