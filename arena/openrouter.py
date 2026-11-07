"""Minimal OpenRouter chat client (stdlib only) plus a model-listing helper."""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

BASE_URL = "https://openrouter.ai/api/v1"


class ApiError(Exception):
    """A request failed in a way that is specific to one call/model."""


class FatalApiError(ApiError):
    """Bad key / no credits: abort the whole run instead of limping on."""


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    latency: float = 0.0


def load_dotenv(path: str = ".env") -> None:
    """Tiny .env reader so you don't need python-dotenv."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, base_url: str = BASE_URL,
                 timeout: float = 90.0, max_retries: int = 4, json_mode: bool = False,
                 title: str = "LLM Among Us Arena", referer: str | None = None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise FatalApiError("OPENROUTER_API_KEY is not set (export it or put it in a .env file).")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.json_mode = json_mode
        self.title = title
        self.referer = referer

    def chat(self, model: str, messages: list[dict], **params) -> ChatResult:
        body: dict = {"model": model, "messages": messages, "usage": {"include": True}}
        body.update({k: v for k, v in params.items() if v is not None})
        if self.json_mode:
            body.setdefault("response_format", {"type": "json_object"})
        data = json.dumps(body).encode()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": self.title,
        }
        if self.referer:
            headers["HTTP-Referer"] = self.referer

        last: object = None
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(f"{self.base_url}/chat/completions", data=data,
                                         headers=headers, method="POST")
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="replace")[:400]
                if e.code in (401, 402):
                    raise FatalApiError(f"HTTP {e.code} from OpenRouter for {model}: {text}") from None
                if e.code in (408, 409, 425, 429) or e.code >= 500:
                    last = f"HTTP {e.code}: {text}"
                    self._sleep(attempt, e.headers.get("Retry-After"))
                    continue
                raise ApiError(f"HTTP {e.code} for {model}: {text}") from None
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                last = repr(e)
                self._sleep(attempt)
                continue

            if not payload.get("choices"):
                err = payload.get("error") or {}
                code = err.get("code")
                last = f"error payload: {err}"
                if isinstance(code, int) and (code == 429 or code >= 500):
                    self._sleep(attempt)
                    continue
                raise ApiError(f"{model}: {err or payload}")

            msg = payload["choices"][0].get("message") or {}
            usage = payload.get("usage") or {}
            return ChatResult(
                content=msg.get("content") or "",
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                cost=float(usage.get("cost") or 0.0),
                latency=time.time() - t0,
            )
        raise ApiError(f"{model}: gave up after {self.max_retries + 1} attempts ({last})")

    @staticmethod
    def _sleep(attempt: int, retry_after: str | None = None) -> None:
        try:
            delay = float(retry_after) if retry_after else 0.0
        except ValueError:
            delay = 0.0
        delay = max(delay, min(30.0, 1.5 * 2 ** attempt)) + random.random()
        time.sleep(delay)


def list_models(name_filter: str | None = None, limit: int = 60) -> list[dict]:
    """Public endpoint, no key needed."""
    req = urllib.request.Request(f"{BASE_URL}/models", headers={"X-Title": "LLM Among Us Arena"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode())["data"]
    rows = []
    for m in data:
        if name_filter and name_filter.lower() not in m["id"].lower():
            continue
        pricing = m.get("pricing") or {}
        rows.append({
            "id": m["id"],
            "context": m.get("context_length"),
            "prompt_per_M": float(pricing.get("prompt") or 0) * 1e6,
            "completion_per_M": float(pricing.get("completion") or 0) * 1e6,
        })
    rows.sort(key=lambda r: r["id"])
    return rows[:limit]
