"""Browser-driven setup: pick the role and the model of every seat, then start the game.

    python -m arena serve        # opens http://127.0.0.1:8765 with a setup screen

The Lobby owns the "which game is running" state and exposes a tiny JSON API that LiveServer routes to:

    GET  /api/state    phase, roster, defaults, whether an API key is set
    GET  /api/models   every OpenRouter model id (for the custom-model autocomplete)
    POST /api/start    {"seats": [{"model": "...", "role": "random|crew|impostor"}, ...], "impostors": 2}
    POST /api/stop     abort the running game
"""
from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .agents import MockClient
from .game import COLORS, GameAborted, GameConfig, validate_roles
from .openrouter import FatalApiError, OpenRouterClient, list_models
from .tournament import play_one, preflight_errors

MIN_PLAYERS = 4
MAX_PLAYERS = len(COLORS)
MOCK_ID = "mock/bot"
RANDOM_ID = "random"
_MODEL_RE = re.compile(r"^[A-Za-z0-9._:@+/\-]{1,100}$")


class BadRequest(Exception):
    """The setup the browser sent can't be played; the message is shown to the user."""


class RoutingClient:
    """One client for a mixed table: `mock/...` ids go to the offline mock, everything else to OpenRouter."""

    def __init__(self, real, mock):
        self.real, self.mock = real, mock

    def chat(self, model: str, messages: list[dict], **params):
        client = self.mock if model.startswith("mock/") else self.real
        return client.chat(model, messages, **params)


class Lobby:
    def __init__(self, server, cfg: GameConfig, roster: list[dict], orc: dict, out_dir: str | Path = "runs",
                 mock_latency: float = 0.5):
        self.server, self.cfg, self.roster, self.orc = server, cfg, roster, orc
        self.out_dir = Path(out_dir)
        self.mock_latency = mock_latency
        self.base_params = {"temperature": orc.get("temperature", 0.7), "max_tokens": orc.get("max_tokens", 800)}
        self.phase = "idle"             # idle | running | finished | error
        self.error: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._over = False              # the running game has emitted its "end" event (only file-writing is left)
        self._thread: threading.Thread | None = None
        self._models: list[str] | None = None

    # ------------------------------------------------------------------ API entry point
    def handle(self, method: str, path: str, body: dict | None) -> tuple[int, dict]:
        try:
            if method == "GET" and path == "/api/state":
                return 200, self.state()
            if method == "GET" and path == "/api/models":
                return 200, {"models": self.models()}
            if method == "POST" and path == "/api/start":
                return 200, self.start(body or {})
            if method == "POST" and path == "/api/stop":
                self._stop.set()
                return 200, {"ok": True}
        except BadRequest as e:
            return 400, {"error": str(e)}
        return 404, {"error": "not found"}

    def state(self) -> dict:
        return {
            "phase": self.phase, "error": self.error,
            "has_key": bool(os.environ.get("OPENROUTER_API_KEY")),
            "defaults": {"players": self.cfg.players, "impostors": self.cfg.impostors},
            "limits": {"min_players": MIN_PLAYERS, "max_players": MAX_PLAYERS},
            "roster": [{"id": m["id"], "label": m.get("label") or m["id"]} for m in self.roster],
            "colors": COLORS, "mock_id": MOCK_ID, "random_id": RANDOM_ID,
        }

    def models(self) -> list[str]:
        if self._models is None:
            try:
                self._models = [m["id"] for m in list_models(limit=100000)]
            except Exception:           # offline or OpenRouter down: the box still accepts typed ids
                return []
        return self._models

    # ------------------------------------------------------------------ start / stop
    def _entry(self, model_id: str) -> dict:
        """Roster entry (with its per-model params) if the id is in config.toml, else a bare entry."""
        for m in self.roster:
            if m["id"] == model_id:
                return {"id": model_id, "label": m.get("label"), "params": m.get("params") or dict(self.base_params)}
        label = {MOCK_ID: "mock-bot", RANDOM_ID: "random-bot"}.get(model_id)
        return {"id": model_id, "label": label, "params": dict(self.base_params)}

    def start(self, spec: dict) -> dict:
        seats_in = spec.get("seats")
        if not isinstance(seats_in, list):
            raise BadRequest("missing seats")
        n = len(seats_in)
        if not MIN_PLAYERS <= n <= MAX_PLAYERS:
            raise BadRequest(f"a game needs {MIN_PLAYERS}-{MAX_PLAYERS} players, got {n}")
        seats, roles = [], []
        for i, s in enumerate(seats_in):
            model = s.get("model") if isinstance(s, dict) else None
            role = (s.get("role") if isinstance(s, dict) else None) or "random"
            if not isinstance(model, str) or not (model in (RANDOM_ID,) or _MODEL_RE.match(model)):
                raise BadRequest(f"{COLORS[i]}: pick a model (got {model!r})")
            seats.append(self._entry(model))
            roles.append(role)
        try:
            impostors = int(spec.get("impostors", self.cfg.impostors))
            cfg = replace(self.cfg, players=n, impostors=impostors)
            validate_roles(n, impostors, roles)
        except (TypeError, ValueError) as e:
            raise BadRequest(str(e)) from None

        if not self._lock.acquire(blocking=False):
            raise BadRequest("another game is being started, one moment")
        try:
            if self._thread and self._thread.is_alive():
                if not self._over:
                    raise BadRequest("a game is already running (stop it first)")
                self._thread.join(timeout=5)        # it just ended and is still writing its files
                if self._thread.is_alive():
                    raise BadRequest("the previous game is still saving, try again in a moment")
            client = self._make_client(seats)
            self._preflight(client, seats)
            seed = int(time.time()) % 100000
            out = self.out_dir / f"{datetime.now():%Y%m%d-%H%M%S}-play"
            self._stop.clear()
            self.server.reset()
            self._over = False
            self.phase, self.error = "running", None
            self._thread = threading.Thread(target=self._run, args=(seats, roles, cfg, client, seed, out), daemon=True)
            self._thread.start()
        finally:
            self._lock.release()
        return {"ok": True, "out": str(out)}

    def _make_client(self, seats: list[dict]):
        mock = MockClient(seed=int(time.time()) % 1000, latency=self.mock_latency)
        needs_real = any(s["id"] != RANDOM_ID and not s["id"].startswith("mock/") for s in seats)
        if not needs_real:
            return RoutingClient(None, mock)
        try:
            real = OpenRouterClient(timeout=self.orc.get("timeout", 90), json_mode=self.orc.get("json_mode", False),
                                    max_retries=self.orc.get("max_retries", 4), referer=self.orc.get("referer"))
        except FatalApiError as e:
            raise BadRequest(f"{e} Pick Mock bot or Random bot for every seat to play without a key.") from None
        return RoutingClient(real, mock)

    def _preflight(self, client, seats: list[dict]) -> None:
        real = list({s["id"]: s for s in seats if s["id"] != RANDOM_ID and not s["id"].startswith("mock/")}.values())
        bad = preflight_errors(client, real)
        if bad:
            raise BadRequest("Model check failed:\n" + "\n".join(f"- {m}: {err}" for m, err in bad))

    def _run(self, seats, roles, cfg, client, seed, out) -> None:
        try:
            out.mkdir(parents=True, exist_ok=True)
            def sink(event: dict) -> None:
                if event.get("type") == "end":
                    self._over = True
                self.server.push(event)

            play_one(0, [], cfg, client, "llm", seed, out, sink=sink, seats=seats, roles=roles,
                     should_stop=self._stop.is_set)
            self.phase = "finished"
        except GameAborted:
            self.phase = "finished"
            self.server.push({"type": "error", "message": "Game stopped."})
        except Exception as e:                  # a crash must reach the browser, not die in this thread
            self.phase, self.error = "error", f"{type(e).__name__}: {e}"
            self.server.push({"type": "error", "message": self.error})
