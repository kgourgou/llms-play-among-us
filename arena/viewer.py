"""Turn a game's replay.json into one self-contained HTML replay, or render the live page.

    python -m arena replay runs/demo          # writes runs/demo/replay.html
"""
from __future__ import annotations

import json
from pathlib import Path

TEMPLATE = Path(__file__).with_name("viewer.html").read_text(encoding="utf-8")


def render_html(replay: dict | None, standalone: bool = True, live: bool = False, setup: bool = False) -> str:
    """replay=None + live=True gives the live page that streams events from /events.
    setup=True adds the new-game screen (needs the /api/ routes from lobby.py)."""
    data = "null" if replay is None else json.dumps(replay, separators=(",", ":")).replace("</", "<\\/")
    body = (TEMPLATE.replace("__LIVE__", "true" if live else "false")
            .replace("__SETUP__", "true" if setup else "false").replace("__DATA__", data))
    if not standalone:
        return body
    return ("<!doctype html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
            "<style>:root{color-scheme:light}body{margin:0;font:14px system-ui,sans-serif}</style></head>"
            f"<body>\n{body}\n</body></html>\n")


def build_replay(run_dir: str | Path, standalone: bool = True) -> Path:
    run_dir = Path(run_dir)
    replay = json.loads((run_dir / "replay.json").read_text())
    out = run_dir / "replay.html"
    out.write_text(render_html(replay, standalone=standalone))
    return out
