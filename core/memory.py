"""
Short-term conversational memory manager.
Maintains a rolling buffer of the last N exchanges for both assistants.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional


Role = Literal["user", "assistant", "system"]


@dataclass
class Message:
    role: Role
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ConversationMemory:
    """
    Rolling buffer memory that keeps the last `max_exchanges` user/assistant pairs.
    A system prompt is always prepended when building the message list.
    """

    def __init__(self, max_exchanges: int = 8, system_prompt: Optional[str] = None):
        self.max_exchanges = max_exchanges
        self.system_prompt: Optional[str] = system_prompt
        self._history: List[Message] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_user_message(self, content: str) -> None:
        self._history.append(Message(role="user", content=content))
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        self._history.append(Message(role="assistant", content=content))
        self._trim()

    def get_messages(self) -> List[dict]:
        """Return the full message list ready for an LLM API call."""
        messages: List[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(msg.to_dict() for msg in self._history)
        return messages

    def clear(self) -> None:
        self._history.clear()

    def set_system_prompt(self, prompt: str) -> None:
        self.system_prompt = prompt

    @property
    def history(self) -> List[Message]:
        return list(self._history)

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self._history if m.role == "user")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trim(self) -> None:
        """Keep only the last max_exchanges * 2 messages (user + assistant pairs)."""
        max_messages = self.max_exchanges * 2
        if len(self._history) > max_messages:
            self._history = self._history[-max_messages:]

    def __repr__(self) -> str:
        return (
            f"ConversationMemory(turns={self.turn_count}, "
            f"max_exchanges={self.max_exchanges})"
        )
