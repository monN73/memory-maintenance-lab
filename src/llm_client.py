"""Minimal OpenAI-compatible HTTP client for DeepSeek / local endpoints using stdlib."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def load_env_file(env_path: Path | None = None) -> dict[str, str]:
    """Load key=value pairs from .env without mutating global secrets unnecessarily."""
    candidates = []
    if env_path is not None:
        candidates.append(env_path)
    candidates.append(Path.home() / ".hermes" / ".env")
    loaded: dict[str, str] = {}
    for p in candidates:
        if not p.exists():
            continue
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and k not in loaded:
                loaded[k] = v
        break
    return loaded


DEFAULT_MODELS = {"deepseek": "deepseek-flash", "ecnu": "ecnu-plus"}


class LLMClient:
    """Zero-dependency OpenAI-compatible chat completions client."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_s: int = 120,
        max_retries: int = 3,
        provider: str = "deepseek",
    ) -> None:
        if provider not in DEFAULT_MODELS:
            raise ValueError(f"Unsupported provider: {provider}")
        self.provider = provider
        env_vars = load_env_file()
        key_name = f"{provider.upper()}_API_KEY"
        base_name = f"{provider.upper()}_BASE_URL"
        self.api_key = api_key or os.getenv(key_name) or env_vars.get(key_name, "")
        raw_base = base_url or os.getenv(base_name) or env_vars.get(base_name, "")
        if not self.api_key:
            raise RuntimeError(f"{key_name} not found in environment or ~/.hermes/.env")
        if not raw_base:
            if provider == "deepseek":
                raw_base = "https://api.deepseek.com/v1"
            else:
                raise RuntimeError(f"{base_name} not found in environment or ~/.hermes/.env")
        self.base_url = raw_base.rstrip("/")
        if provider == "deepseek" and "deepseek.com" in self.base_url and not self.base_url.endswith("/v1"):
            self.base_url += "/v1"
        self.endpoint_hash = hashlib.sha256(self.base_url.encode("utf-8")).hexdigest()
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 512,
        seed: int | None = None,
        thinking: str | None = None,
    ) -> dict[str, Any]:
        """Send a chat completion request and return text, usage, latency_ms, and metadata."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        if thinking is not None:
            if self.provider != "deepseek" or thinking not in ("enabled", "disabled"):
                raise ValueError("thinking is supported only for DeepSeek and must be enabled or disabled")
            payload["thinking"] = {"type": thinking}

        body = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/chat/completions"

        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            req = urllib.request.Request(
                url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            t0 = time.perf_counter()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                latency_ms = int((time.perf_counter() - t0) * 1000)
                choice = raw["choices"][0]
                text = (choice["message"].get("content") or "").strip()
                usage = raw.get("usage") or {}
                return {
                    "provider": self.provider,
                    "model_requested": model,
                    "endpoint_hash": self.endpoint_hash,
                    "text": text,
                    "usage": {
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                        "total_tokens": usage.get("total_tokens"),
                        "completion_tokens_details": usage.get("completion_tokens_details"),
                    },
                    "latency_ms": latency_ms,
                    "model_returned": raw.get("model"),
                    "system_fingerprint": raw.get("system_fingerprint"),
                    "finish_reason": choice.get("finish_reason"),
                    "attempts": attempt,
                }
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="replace")
                last_err = RuntimeError(f"HTTP {e.code}: {err_body}")
                if e.code in (400, 401, 403, 404):
                    raise last_err
            except Exception as e:
                last_err = e

            if attempt < self.max_retries:
                time.sleep(min(2 ** attempt, 8))

        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {last_err}")
