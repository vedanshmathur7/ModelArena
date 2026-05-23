"""
Model adapter layer — unified interface for OSS and Frontier backends.

Architecture:
    BaseAssistant (abstract)
        ├── OSSAssistant   — Ollama or HuggingFace Transformers
        └── FrontierAssistant — OpenAI-compatible API (Llama 3.1 via Groq, etc.)

Both expose the same generate_response() interface so the rest of the
pipeline never needs to know which backend is active.
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Generator, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseAssistant(ABC):
    """
    Abstract base for all model backends.
    Subclasses must implement `generate_response` and `stream_response`.
    """

    def __init__(self, model_name: str, max_new_tokens: int = 512):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    @abstractmethod
    def generate_response(self, messages: List[dict]) -> str:
        """Generate a complete response given a message list."""
        ...

    @abstractmethod
    def stream_response(self, messages: List[dict]) -> Generator[str, None, None]:
        """Yield response tokens/chunks as they are generated."""
        ...

    def health_check(self) -> bool:
        """Return True if the backend is reachable."""
        try:
            result = self.generate_response(
                [{"role": "user", "content": "ping"}]
            )
            return bool(result)
        except Exception as exc:
            logger.warning("Health check failed for %s: %s", self.model_name, exc)
            return False

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model_name})"


# ---------------------------------------------------------------------------
# OSS Assistant — Ollama backend
# ---------------------------------------------------------------------------

class OSSAssistant(BaseAssistant):
    """
    Open-source model assistant backed by Ollama.

    Requires Ollama to be running locally (default: http://localhost:11434).
    Recommended models: qwen2.5:1.5b, phi3:mini, llama3.2:1b
    """

    def __init__(
        self,
        model_name: str = "qwen2.5:1.5b",
        base_url: str = "http://localhost:11434",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ):
        super().__init__(model_name=model_name, max_new_tokens=max_new_tokens)
        self.base_url = base_url
        self.temperature = temperature
        self._client = self._init_client()

    def _init_client(self):
        try:
            import ollama  # type: ignore
            return ollama
        except ImportError:
            logger.warning(
                "ollama package not installed. "
                "Install with: pip install ollama"
            )
            return None

    def generate_response(self, messages: List[dict]) -> str:
        if self._client is None:
            return self._fallback_response()

        # Inject tool output if applicable
        from core.tools import check_and_inject_tools
        messages = check_and_inject_tools(messages)

        try:
            response = self._client.chat(
                model=self.model_name,
                messages=messages,
                options={
                    "num_predict": self.max_new_tokens,
                    "temperature": self.temperature,
                },
            )
            return response["message"]["content"]
        except Exception as exc:
            logger.error("Ollama generation error: %s", exc)
            return f"[OSS model error: {exc}]"

    def stream_response(self, messages: List[dict]) -> Generator[str, None, None]:
        if self._client is None:
            yield self._fallback_response()
            return

        # Inject tool output if applicable
        from core.tools import check_and_inject_tools
        messages = check_and_inject_tools(messages)

        try:
            stream = self._client.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                options={
                    "num_predict": self.max_new_tokens,
                    "temperature": self.temperature,
                },
            )
            for chunk in stream:
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
        except Exception as exc:
            logger.error("Ollama streaming error: %s", exc)
            yield f"[OSS model streaming error: {exc}]"

    def _fallback_response(self) -> str:
        return (
            "OSS model is unavailable. "
            "Please ensure Ollama is running: `ollama serve`"
        )


# ---------------------------------------------------------------------------
# OSS Assistant — HuggingFace Transformers backend (fallback)
# ---------------------------------------------------------------------------

class HFAssistant(BaseAssistant):
    """
    Open-source model assistant backed by HuggingFace Transformers.
    Useful when Ollama is not available (e.g., Hugging Face Spaces).

    Recommended: Qwen/Qwen2.5-0.5B-Instruct (CPU-friendly)
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        device: str = "cpu",
    ):
        super().__init__(model_name=model_name, max_new_tokens=max_new_tokens)
        self.temperature = temperature
        self.device = device
        self._pipeline = None
        self._tokenizer = None
        self._model = None

    def _load_model(self):
        """Lazy-load the model on first use."""
        if self._pipeline is not None:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch

            logger.info("Loading HF model: %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map=self.device,
            )
            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            logger.info("HF model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load HF model: %s", exc)
            raise

    def _messages_to_text(self, messages: List[dict]) -> str:
        """Convert message list to a chat template string."""
        try:
            self._load_model()
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            return text
        except Exception:
            # Fallback: simple concatenation
            parts = []
            for msg in messages:
                role = msg["role"].upper()
                parts.append(f"{role}: {msg['content']}")
            parts.append("ASSISTANT:")
            return "\n".join(parts)

    def generate_response(self, messages: List[dict]) -> str:
        # Inject tool output if applicable
        from core.tools import check_and_inject_tools
        messages = check_and_inject_tools(messages)

        try:
            self._load_model()
            prompt = self._messages_to_text(messages)
            outputs = self._pipeline(prompt)
            generated = outputs[0]["generated_text"]
            # Strip the prompt from the output
            if generated.startswith(prompt):
                generated = generated[len(prompt):]
            return generated.strip()
        except Exception as exc:
            logger.error("HF generation error: %s", exc)
            return f"[HF model error: {exc}]"

    def stream_response(self, messages: List[dict]) -> Generator[str, None, None]:
        """HF streaming via TextIteratorStreamer."""
        # Inject tool output if applicable
        from core.tools import check_and_inject_tools
        messages = check_and_inject_tools(messages)

        try:
            from transformers import TextIteratorStreamer
            import threading

            self._load_model()
            prompt = self._messages_to_text(messages)
            inputs = self._tokenizer(prompt, return_tensors="pt")

            streamer = TextIteratorStreamer(
                self._tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )

            generation_kwargs = {
                **inputs,
                "streamer": streamer,
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature,
                "do_sample": True,
            }

            thread = threading.Thread(
                target=self._model.generate, kwargs=generation_kwargs
            )
            thread.start()

            for token in streamer:
                yield token

            thread.join()
        except Exception as exc:
            logger.error("HF streaming error: %s", exc)
            yield f"[HF streaming error: {exc}]"


# ---------------------------------------------------------------------------
# Frontier Assistant — OpenAI-compatible API
# ---------------------------------------------------------------------------

class FrontierAssistant(BaseAssistant):
    """
    Frontier model assistant backed by OpenAI-compatible API.
    Works with: Llama 3.1 via Groq, Mixtral via Groq, or any OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        model_name: str = "llama-3.1-8b-instant",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        super().__init__(model_name=model_name, max_new_tokens=max_new_tokens)
        self.temperature = temperature
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._client = self._init_client()

    def _init_client(self):
        try:
            from openai import OpenAI

            kwargs = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url

            return OpenAI(**kwargs)
        except ImportError:
            logger.warning("openai package not installed. pip install openai")
            return None
        except Exception as exc:
            logger.error("Failed to init OpenAI client: %s", exc)
            return None

    def generate_response(self, messages: List[dict]) -> str:
        if self._client is None:
            return "[Frontier model unavailable — check API key and openai package]"

        try:
            # Check for native function/tool calling support
            from core.tools import get_tools_definition, execute_tool
            tools = get_tools_definition()
            import json

            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                tools=tools,
                tool_choice="auto",
            )
            
            message = response.choices[0].message
            if message.tool_calls:
                # Copy messages to avoid modifying caller's history list in-place
                messages_copy = list(messages)
                messages_copy.append({
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in message.tool_calls
                    ]
                })
                
                for tool_call in message.tool_calls:
                    func_name = tool_call.function.name
                    try:
                        func_args = json.loads(tool_call.function.arguments)
                    except Exception:
                        func_args = {}
                    tool_output = execute_tool(func_name, func_args)
                    
                    messages_copy.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": tool_output
                    })
                
                final_response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages_copy,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                return final_response.choices[0].message.content or ""
            
            return message.content or ""
        except Exception as exc:
            logger.error("OpenAI generation error: %s", exc)
            return f"[Frontier model error: {exc}]"

    def stream_response(self, messages: List[dict]) -> Generator[str, None, None]:
        if self._client is None:
            yield "[Frontier model unavailable — check API key and openai package]"
            return

        try:
            from core.tools import get_tools_definition, execute_tool
            tools = get_tools_definition()
            import json

            stream = self._client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature,
                tools=tools,
                tool_choice="auto",
                stream=True,
            )
            
            tool_calls_data = []
            tool_calls_detected = False
            
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.tool_calls:
                    tool_calls_detected = True
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if len(tool_calls_data) <= idx:
                            tool_calls_data.append({
                                "id": tc_delta.id or "",
                                "name": tc_delta.function.name or "",
                                "arguments": tc_delta.function.arguments or "",
                            })
                        else:
                            if tc_delta.id:
                                tool_calls_data[idx]["id"] = tc_delta.id
                            if tc_delta.function.name:
                                tool_calls_data[idx]["name"] += tc_delta.function.name
                            if tc_delta.function.arguments:
                                tool_calls_data[idx]["arguments"] += tc_delta.function.arguments
                
                if delta.content:
                    yield delta.content
            
            if tool_calls_detected:
                messages_copy = list(messages)
                
                assistant_tool_calls = []
                for tc in tool_calls_data:
                    assistant_tool_calls.append({
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"]
                        }
                    })
                
                messages_copy.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": assistant_tool_calls
                })
                
                for tc in tool_calls_data:
                    func_name = tc["name"]
                    try:
                        func_args = json.loads(tc["arguments"])
                    except Exception:
                        func_args = {}
                    tool_output = execute_tool(func_name, func_args)
                    
                    messages_copy.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": func_name,
                        "content": tool_output
                    })
                
                final_stream = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=messages_copy,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    stream=True,
                )
                for chunk in final_stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
        except Exception as exc:
            logger.error("OpenAI streaming error: %s", exc)
            yield f"[Frontier streaming error: {exc}]"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_assistant(
    backend: str,
    model_name: Optional[str] = None,
    **kwargs,
) -> BaseAssistant:
    """
    Factory function to create the appropriate assistant.

    Args:
        backend: One of "ollama", "huggingface", "openai"
        model_name: Override the default model name
        **kwargs: Additional kwargs passed to the assistant constructor
    """
    backend = backend.lower()

    if backend == "ollama":
        return OSSAssistant(
            model_name=model_name or "qwen2.5:1.5b",
            **kwargs,
        )
    elif backend in ("huggingface", "hf", "transformers"):
        return HFAssistant(
            model_name=model_name or "Qwen/Qwen2.5-0.5B-Instruct",
            **kwargs,
        )
    elif backend in ("openai", "frontier", "gpt"):
        return FrontierAssistant(
            model_name=model_name or "llama-3.1-8b-instant",
            **kwargs,
        )
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. "
            "Choose from: 'ollama', 'huggingface', 'openai'"
        )
