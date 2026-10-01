"""
llm_client.py
-------------
LLM backends for the generation step of RAG (synthesizing a research-gap
analysis from retrieved papers).

Providers (pick with --provider / LLM_PROVIDER / the UI dropdown):

  anthropic   Claude (paid API)                       key: ANTHROPIC_API_KEY
  gemini      Google Gemini via AI Studio (free tier) key: GEMINI_API_KEY
  groq        Groq (free tier, open models)           key: GROQ_API_KEY
  openrouter  OpenRouter (has ':free' models)         key: OPENROUTER_API_KEY
  ollama      Local models on your own PC (100% free, no key, no internet)

gemini / groq / openrouter / ollama all speak the OpenAI-compatible
/chat/completions protocol, so one small client (plain `requests`) covers them.

Free-tier model names and limits change often. Every default below can be
overridden with --model, the UI field, or the LLM_MODEL env var.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import requests
from anthropic import Anthropic

DEFAULT_PROVIDER = "anthropic"
DEFAULT_MODEL = "claude-sonnet-5"  # kept for backwards compatibility

PROVIDERS: Dict[str, dict] = {
    "anthropic": {
        "model": DEFAULT_MODEL,
        "key_env": "ANTHROPIC_API_KEY",
        "key_url": "https://console.anthropic.com/",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.5-flash",
        "key_env": "GEMINI_API_KEY",
        "key_url": "https://aistudio.google.com/apikey",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
        "key_env": "GROQ_API_KEY",
        "key_url": "https://console.groq.com/keys",
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "key_env": "OPENROUTER_API_KEY",
        "key_url": "https://openrouter.ai/keys",
    },
    "ollama": {
        "base_url": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "model": "llama3.1",
        "key_env": None,  # local, no key needed
        "key_url": "https://ollama.com/download",
    },
}


def default_model(provider: str) -> str:
    return PROVIDERS[provider]["model"]


def _resolve_model(provider: str, model: Optional[str]) -> str:
    env_model = os.environ.get("LLM_MODEL")
    if provider == "anthropic":
        env_model = env_model or os.environ.get("ANTHROPIC_MODEL")
    return model or env_model or PROVIDERS[provider]["model"]


def _missing_key_error(provider: str) -> ValueError:
    cfg = PROVIDERS[provider]
    return ValueError(
        f"No API key found for provider '{provider}'. Set {cfg['key_env']} in your .env file "
        f"(get one at {cfg['key_url']}) or pick another provider, e.g. --provider ollama for a local free model."
    )


class LLMClient:
    """Anthropic (Claude) backend."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise _missing_key_error("anthropic")
        # The SDK automatically retries rate-limit / transient errors.
        self.client = Anthropic(api_key=self.api_key, max_retries=3, timeout=120.0)
        self.model = _resolve_model("anthropic", model)

    def complete(self, system: str, user_prompt: str, max_tokens: int = 4096) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        parts: List[str] = [b.text for b in response.content if getattr(b, "type", None) == "text"]
        if getattr(response, "stop_reason", None) == "max_tokens":
            raise RuntimeError("LLM output was truncated (max_tokens reached). Lower --top-k or raise max_tokens.")
        return "\n".join(parts).strip()


class OpenAICompatClient:
    """Any OpenAI-compatible /chat/completions endpoint (Gemini, Groq, OpenRouter, Ollama...)."""

    def __init__(
        self,
        provider: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        cfg = PROVIDERS[provider]
        self.provider = provider
        self.base_url = (base_url or cfg["base_url"]).rstrip("/")
        self.model = _resolve_model(provider, model)
        key_env = cfg["key_env"]
        self.api_key = api_key or (os.environ.get(key_env) if key_env else "")
        if key_env and not self.api_key:
            raise _missing_key_error(provider)

    def complete(self, system: str, user_prompt: str, max_tokens: int = 4096) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
        }

        resp = None
        for attempt in range(3):
            try:
                resp = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=payload, timeout=180)
            except requests.ConnectionError as e:
                hint = " Is Ollama running? Start it with `ollama serve`." if self.provider == "ollama" else ""
                raise RuntimeError(f"Could not reach {self.provider} at {self.base_url}.{hint}") from e
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2 * (attempt + 1))  # free tiers rate-limit; back off and retry
                continue
            break

        if resp.status_code != 200:
            raise RuntimeError(
                f"{self.provider} API error {resp.status_code}: {resp.text[:300]} "
                f"(model='{self.model}' - check the model name / your free-tier quota)"
            )
        data = resp.json()
        try:
            choice = data["choices"][0]
            text = (choice["message"].get("content") or "").strip()
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Unexpected response from {self.provider}: {str(data)[:300]}") from e
        if choice.get("finish_reason") == "length":
            raise RuntimeError("LLM output was truncated (max_tokens reached). Lower --top-k or raise max_tokens.")
        if not text:
            raise RuntimeError(f"{self.provider} returned an empty reply (model='{self.model}').")
        return text


def get_llm_client(provider: Optional[str] = None, model: Optional[str] = None, api_key: Optional[str] = None):
    """Factory used by the pipeline. Provider defaults to $LLM_PROVIDER, else 'anthropic'."""
    provider = (provider or os.environ.get("LLM_PROVIDER") or DEFAULT_PROVIDER).lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}")
    if provider == "anthropic":
        return LLMClient(api_key=api_key, model=model)
    return OpenAICompatClient(provider, api_key=api_key, model=model)
