"""CLI:  python -m arena {play,tournament,serve,replay,models} ..."""
from __future__ import annotations

import argparse
import sys
import time
import tomllib
from datetime import datetime
from pathlib import Path

from .agents import MockClient
from .game import GameConfig
from .openrouter import FatalApiError, OpenRouterClient, list_models, load_dotenv
from .tournament import preflight, run_games, write_leaderboard
from .viewer import build_replay

DEFAULT_MODELS = [{"id": "mock/a"}, {"id": "mock/b"}, {"id": "mock/c"}]


def load_config(path: str | None) -> tuple[GameConfig, list[dict], dict]:
    data: dict = {}
    if path and Path(path).exists():
        data = tomllib.loads(Path(path).read_text())
    cfg = GameConfig.from_dict(data.get("game", {}))
    orc = data.get("openrouter", {})
    base_params = {"temperature": orc.get("temperature", 0.7), "max_tokens": orc.get("max_tokens", 800)}
    entries = []
    for m in data.get("models", []):
        entries.append({"id": m["id"], "label": m.get("label"), "params": {**base_params, **m.get("params", {})}})
    return cfg, entries, orc


def pick_models(args, entries: list[dict], orc: dict) -> list[dict]:
    if args.models:
        base = {"temperature": orc.get("temperature", 0.7), "max_tokens": orc.get("max_tokens", 800)}
        return [{"id": m.strip(), "label": None, "params": base} for m in args.models.split(",") if m.strip()]
    return entries


def make_client(args, orc: dict):
    if args.mock:
        lat = args.mock_latency if args.mock_latency is not None else (0.5 if args.live else 0.0)
        return MockClient(seed=args.seed or 0, latency=lat)
    return OpenRouterClient(timeout=orc.get("timeout", 90), json_mode=orc.get("json_mode", False),
                            max_retries=orc.get("max_retries", 4), referer=orc.get("referer"))


def cmd_run(args, tournament: bool) -> int:
    load_dotenv()
    cfg, entries, orc = load_config(args.config)
    if args.concurrency:
        cfg.concurrency = args.concurrency
    if args.players:
        cfg.players = args.players
    if args.impostors:
        cfg.impostors = args.impostors
    entries = pick_models(args, entries, orc)
    mode = "random" if args.random else ("mock" if args.mock else "llm")
    if mode != "llm" and not entries:
        entries = DEFAULT_MODELS
    if not entries:
        print("No models configured. Add [[models]] to config.toml or pass --models a/b,c/d", file=sys.stderr)
        return 2
    client = None if mode == "random" else make_client(args, orc)
    if mode == "llm" and not args.no_preflight:
        print(f"Preflight: checking {len(entries)} model(s)...")
        preflight(client, entries)
    n_games = args.games if tournament else 1
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(args.out) / f"{stamp}-{'tournament' if tournament else 'play'}"
    seed = args.seed if args.seed is not None else int(datetime.now().timestamp()) % 100000
    print(f"Models: {', '.join(e.get('label') or e['id'] for e in entries)}")
    print(f"{n_games} game(s), {cfg.players} players ({cfg.impostors} impostors), seed {seed}, output {out}")
    server = None
    if args.live:
        from .live import LiveServer
        server = LiveServer(port=args.port)
        server.start()
        print(f"\n>>> LIVE VIEW: {server.url}  (open it in your browser; it plays along as the game runs)\n")
        if not args.no_browser:
            try:
                import webbrowser
                webbrowser.open(server.url)
            except Exception:
                pass
        time.sleep(1.5)          # give the browser a moment to connect so it catches the first event
    results = run_games(n_games, entries, cfg, client, mode, seed, out,
                        parallel=args.parallel if tournament else 1,
                        echo=args.echo or (not tournament and not args.live), html=not args.no_html,
                        sink=server.push if server else None)
    if tournament or len(entries) > 1:
        print("\n" + write_leaderboard(results, out))
    if not args.no_html:
        print("Replay(s):", ", ".join(f"{r['dir']}/replay.html" for r in results[:5]) + (" ..." if len(results) > 5 else ""))
    total = sum(p["cost"] for r in results for p in r["players"])
    if total:
        print(f"Total cost reported by OpenRouter: ${total:.4f}")
    if server:
        print(f"\nGame finished. The live view stays up at {server.url} so you can scrub back through it. Ctrl-C to quit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    return 0


def cmd_serve(args) -> int:
    """Web UI with a setup screen: pick each seat's model and role in the browser, then watch it play."""
    from .live import LiveServer
    from .lobby import Lobby
    load_dotenv()
    cfg, entries, orc = load_config(args.config)
    if args.concurrency:
        cfg.concurrency = args.concurrency
    server = LiveServer(port=args.port, setup=True)
    server.api = Lobby(server, cfg, entries, orc, out_dir=args.out,
                       mock_latency=args.mock_latency if args.mock_latency is not None else 0.5)
    server.start()
    print(f"\n>>> ARENA: {server.url}  (set up and start games in your browser; Ctrl-C to quit)\n")
    if not args.no_browser:
        try:
            import webbrowser
            webbrowser.open(server.url)
        except Exception:
            pass
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="arena", description="LLM Among Us arena (OpenRouter)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", default="config.toml")
        p.add_argument("--models", help="comma-separated OpenRouter model ids (overrides config)")
        p.add_argument("--mock", action="store_true", help="offline mock LLM (no network, no cost)")
        p.add_argument("--random", action="store_true", help="uniformly random agents (baseline)")
        p.add_argument("--seed", type=int)
        p.add_argument("--out", default="runs")
        p.add_argument("--players", type=int)
        p.add_argument("--impostors", type=int)
        p.add_argument("--concurrency", type=int, help="parallel LLM calls inside one game")
        p.add_argument("--echo", action="store_true", help="print the god-view transcript live")
        p.add_argument("--no-html", action="store_true", help="skip generating replay.html")
        p.add_argument("--no-preflight", action="store_true")
        p.add_argument("--live", action="store_true", help="watch the game live in your browser (god mode)")
        p.add_argument("--port", type=int, default=8765, help="port for --live")
        p.add_argument("--no-browser", action="store_true", help="with --live, don't auto-open the browser")
        p.add_argument("--mock-latency", type=float, help="seconds of fake delay per mock LLM call (default 0.5 with --live)")

    p = sub.add_parser("play", help="run a single game")
    common(p)
    p = sub.add_parser("tournament", help="run many games and print a leaderboard")
    common(p)
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--parallel", type=int, default=2, help="games running at once")
    p = sub.add_parser("serve", help="open the web UI and set up games (roles and models per seat) in the browser")
    p.add_argument("--config", default="config.toml")
    p.add_argument("--out", default="runs")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--no-browser", action="store_true")
    p.add_argument("--concurrency", type=int, help="parallel LLM calls inside one game")
    p.add_argument("--mock-latency", type=float, help="seconds of fake delay per Mock bot call (default 0.5)")
    p = sub.add_parser("replay", help="(re)build replay.html for a finished game folder")
    p.add_argument("run_dir")
    p = sub.add_parser("models", help="list OpenRouter model ids")
    p.add_argument("--filter")
    p.add_argument("--limit", type=int, default=60)

    args = ap.parse_args(argv)
    try:
        if args.cmd == "play":
            return cmd_run(args, tournament=False)
        if args.cmd == "tournament":
            return cmd_run(args, tournament=True)
        if args.cmd == "serve":
            return cmd_serve(args)
        if args.cmd == "replay":
            print(build_replay(args.run_dir))
            return 0
        if args.cmd == "models":
            for r in list_models(args.filter, args.limit):
                print(f"{r['id']:<55} ctx={r['context']}  ${r['prompt_per_M']:.2f}/${r['completion_per_M']:.2f} per M tok")
            return 0
    except FatalApiError as e:
        print(f"\nFatal API error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
