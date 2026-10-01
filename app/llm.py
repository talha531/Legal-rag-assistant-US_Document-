"""Groq LLM client (light, free-tier friendly) with retry/backoff and robust JSON parsing."""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Optional, Protocol

from app.config import ConfigError, Settings

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str, json_mode: bool = False, max_tokens: Optional[int] = None) -> str: ...


def parse_json(text: str) -> dict:
    """Parse a JSON object from model output (tolerates code fences / surrounding prose)."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise LLMError("Model did not return JSON")
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise LLMError(f"Invalid JSON from model: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMError("Expected a JSON object")
    return data


class GroqLLM:
    def __init__(self, settings: Settings, model: Optional[str] = None, max_attempts: int = 4):
        if not settings.has_groq_key:
            raise ConfigError(
                "GROQ_API_KEY is missing. Copy .env.example to .env and paste your key "
                "(free at https://console.groq.com/keys)."
            )
        from langchain_groq import ChatGroq

        self.model = model or settings.groq_model
        self.max_attempts = max_attempts
        # gpt-oss models "think" before answering; keep that short and leave room for the answer
        self.is_reasoning = "gpt-oss" in self.model

        params = dict(
            model=self.model,
            api_key=settings.groq_api_key,
            temperature=settings.groq_temperature,
            max_tokens=settings.groq_max_tokens,
            timeout=60,
            max_retries=1,
        )
        if self.is_reasoning:
            params["reasoning_effort"] = "low"
        try:
            self._llm = ChatGroq(**params)
        except Exception:  # older langchain-groq versions do not know reasoning_effort
            params.pop("reasoning_effort", None)
            self._llm = ChatGroq(**params)

    def complete(self, system: str, user: str, json_mode: bool = False, max_tokens: Optional[int] = None) -> str:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        use_json = json_mode
        last_exc: Exception | None = None
        for attempt in range(self.max_attempts):
            llm = self._llm
            if use_json:
                llm = llm.bind(response_format={"type": "json_object"})
            if max_tokens:
                cap = max(max_tokens, 1024) if self.is_reasoning else max_tokens
                llm = llm.bind(max_tokens=cap)
            try:
                return str(llm.invoke(messages).content or "")
            except Exception as exc:  # groq.* errors differ between versions; classify by message
                last_exc = exc
                msg = str(exc).lower()
                if any(k in msg for k in ("401", "invalid api key", "authentication")):
                    raise LLMError("Groq rejected the API key. Check GROQ_API_KEY in your .env file.") from exc
                if "model_not_found" in msg or "does not exist" in msg or "decommissioned" in msg:
                    raise LLMError(
                        f"Groq model '{self.model}' is not available for your key. "
                        "Set GROQ_MODEL=openai/gpt-oss-20b in .env (or pick another model in Settings)."
                    ) from exc
                if use_json and "response_format" in msg:
                    use_json = False  # model does not support JSON mode -> retry as plain text
                    continue
                if any(k in msg for k in ("rate limit", "429", "overloaded", "503", "timeout", "timed out")):
                    wait = self._retry_after(msg) or min(2 ** attempt * 2, 20)
                    logger.warning("Groq transient error (attempt %s): waiting %.1fs", attempt + 1, wait)
                    time.sleep(wait)
                    continue
                raise LLMError(f"Groq request failed: {exc}") from exc
        raise LLMError(
            "Groq is rate-limiting this key (free-tier limit). Wait a moment and try again, "
            f"or lower TOP_K / use a smaller model. Last error: {last_exc}"
        )

    @staticmethod
    def _retry_after(msg: str) -> float:
        m = re.search(r"try again in (\d+(?:\.\d+)?)\s*(ms|s|m)\b", msg)
        if not m:
            return 0.0
        val, unit = float(m.group(1)), m.group(2)
        secs = val / 1000 if unit == "ms" else val * 60 if unit == "m" else val
        return min(secs + 0.5, 25.0)