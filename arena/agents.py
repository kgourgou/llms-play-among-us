"""Agents: the thing that turns a prompt into a decision.

LLMAgent  -> calls an OpenRouter model, parses JSON, retries once on bad output.
RandomAgent / MockClient -> free, offline stand-ins for testing the harness.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass

from . import prompts
from .openrouter import ApiError, ChatResult, FatalApiError


@dataclass
class Turn:
    kind: str            # "action" | "speak" | "vote"
    system: str
    user: str
    n_options: int = 0
    max_chars: int = 500


@dataclass
class Reply:
    choice: int | None = None
    say: str | None = None
    thought: str = ""
    notes: str | None = None
    raw: str = ""
    error: str | None = None      # set when we had to fall back (unparseable / API failure)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    latency: float = 0.0


class Agent:
    label = "agent"

    def act(self, turn: Turn) -> Reply:  # pragma: no cover - interface
        raise NotImplementedError


# --- parsing -----------------------------------------------------------------

def extract_json(text: str) -> dict | None:
    """Find a JSON object in free-form model output (handles ``` fences and chatter)."""
    dec = json.JSONDecoder()
    found: list[dict] = []
    for m in re.finditer(r"\{", text):
        try:
            obj, _ = dec.raw_decode(text[m.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            found.append(obj)
    for obj in found:                       # prefer the object that has the keys we need
        if "choice" in obj or "say" in obj:
            return obj
    return found[0] if found else None


def parse_reply(text: str, turn: Turn) -> tuple[Reply | None, str]:
    obj = extract_json(text)
    if obj is None and turn.kind != "speak":
        m = re.search(r'"?choice"?\s*[:=]\s*"?(\d+)', text)
        if m:
            obj = {"choice": m.group(1)}
    if obj is None:
        return None, "no JSON object found"

    r = Reply(raw=text)
    r.thought = str(obj.get("thought", ""))[:600]
    notes = obj.get("notes")
    if isinstance(notes, str) and notes.strip():
        r.notes = notes.strip()[:400]

    if turn.kind == "speak":
        say = obj.get("say")
        if not isinstance(say, str) or not say.strip():
            return None, 'missing "say"'
        r.say = say.strip()[: turn.max_chars]
        return r, ""

    m = re.search(r"\d+", str(obj.get("choice", "")))
    if not m:
        return None, 'missing "choice"'
    c = int(m.group(0))
    if not 1 <= c <= turn.n_options:
        return None, f"choice {c} is out of range 1-{turn.n_options}"
    r.choice = c
    return r, ""


# --- agents --------------------------------------------------------------------

class LLMAgent(Agent):
    def __init__(self, client, model: str, params: dict | None = None, label: str | None = None):
        self.client = client
        self.model = model
        self.params = params or {}
        self.label = label or model

    def act(self, turn: Turn) -> Reply:
        messages = [{"role": "system", "content": turn.system},
                    {"role": "user", "content": turn.user}]
        pt = ct = 0
        cost = lat = 0.0
        problem = "no attempt made"
        raw = ""
        for attempt in range(2):
            try:
                res: ChatResult = self.client.chat(self.model, messages, **self.params)
            except FatalApiError:
                raise
            except ApiError as e:
                problem = f"API error: {e}"
                continue
            pt += res.prompt_tokens
            ct += res.completion_tokens
            cost += res.cost
            lat += res.latency
            raw = res.content
            reply, problem = parse_reply(raw, turn)
            if reply is not None:
                reply.prompt_tokens, reply.completion_tokens, reply.cost, reply.latency = pt, ct, cost, lat
                if attempt == 1:
                    reply.error = None   # recovered on retry; still counted via tokens
                return reply
            messages = messages + [{"role": "assistant", "content": raw or "(empty)"},
                                   {"role": "user", "content": prompts.retry_reminder(turn.kind, problem)}]
        return Reply(raw=raw, error=problem, prompt_tokens=pt, completion_tokens=ct, cost=cost, latency=lat)


_RANDOM_LINES = [
    "I was doing tasks, nothing suspicious on my end.",
    "Who was near the body? I think we should look at that.",
    "I saw someone acting weird earlier, but I'm not sure who.",
    "Let's not vote randomly. Skip unless someone has evidence.",
    "I was in a different room the whole time.",
]


class RandomAgent(Agent):
    """Uniformly random legal choices. Free; useful for smoke tests and as a baseline."""

    def __init__(self, label: str = "random", seed: int | None = None):
        self.label = label
        self.rng = random.Random(seed)

    def act(self, turn: Turn) -> Reply:
        if turn.kind == "speak":
            return Reply(say=self.rng.choice(_RANDOM_LINES), thought="(random)")
        return Reply(choice=self.rng.randint(1, turn.n_options), thought="(random)")


class MockClient:
    """Offline stand-in for OpenRouterClient. Reads the real prompt, so it exercises
    the whole prompt -> JSON -> parse -> fallback path without spending a cent."""

    def __init__(self, seed: int = 0, garbage_rate: float = 0.06, latency: float = 0.0):
        self.rng = random.Random(seed)
        self.garbage_rate = garbage_rate
        self.latency = latency          # seconds; simulates a slow model so you can watch the live view

    def chat(self, model: str, messages: list[dict], **params) -> ChatResult:
        if self.latency:
            time.sleep(self.latency * (0.5 + random.random()))
        users = [m["content"] for m in messages if m["role"] == "user"]
        kind = re.findall(r"\[\[TURN: (\w+)\]\]", users[-1])[-1]
        first = users[0]
        rng = self.rng
        if rng.random() < self.garbage_rate:
            return ChatResult("Hmm, I think the third option sounds best, honestly.", 800, 20, 0.0, 0.01)

        if kind == "speak":
            m = re.search(r"ALIVE \(\d+\): (.*)", first)
            names = [n.strip() for n in m.group(1).split(",")] if m else ["Red"]
            sus = rng.choice(names)
            say = rng.choice([
                f"I was doing tasks. {sus} seems suspicious to me.",
                f"I saw {sus} near the body. Vote {sus}?",
                "I have no info. I'd skip.",
                f"Not me! Check {sus}, they were alone earlier.",
            ])
            body = {"thought": "mock thought", "say": say}
        else:
            opts = re.findall(r"^\s*(\d+)\) (.*)$", first.split("OPTIONS:")[-1], re.M)
            pick = None
            for want, p in (("REPORT", .9), ("KILL", .6), ("Do task", .7), ("Pretend", .35)):
                cand = [int(n) for n, label in opts if label.startswith(want)]
                if cand and rng.random() < p:
                    pick = rng.choice(cand)
                    break
            if pick is None:
                pick = int(rng.choice(opts)[0])
            if kind == "vote" and rng.random() < 0.3:
                pick = int(opts[-1][0])
            body = {"thought": "mock thought", "choice": pick}
        if rng.random() < 0.3:
            return ChatResult("```json\n" + json.dumps(body) + "\n```", 800, 40, 0.0, 0.01)
        return ChatResult(json.dumps(body), 800, 40, 0.0, 0.01)
