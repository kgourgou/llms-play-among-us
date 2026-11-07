"""Run one or many games, rotate seats/roles, and aggregate per-model stats."""
from __future__ import annotations

import concurrent.futures as cf
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from .agents import Agent, LLMAgent, MockClient, RandomAgent
from .game import Game, GameConfig
from .viewer import build_replay


def assign_seats(entries: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Fill n seats cycling through shuffled model entries so each model plays about equally often."""
    seats: list[dict] = []
    while len(seats) < n:
        batch = entries[:]
        rng.shuffle(batch)
        seats.extend(batch)
    seats = seats[:n]
    rng.shuffle(seats)
    return seats


def build_agents(seats: list[dict], client, mode: str, seed: int) -> list[Agent]:
    agents: list[Agent] = []
    for i, e in enumerate(seats):
        label = e.get("label") or e["id"]
        if mode == "random" or e["id"] == "random":      # "random" as a model id = per-seat random bot
            agents.append(RandomAgent(label=label, seed=seed * 100 + i))
        else:
            agents.append(LLMAgent(client, e["id"], e.get("params"), label=label))
    return agents


def play_one(idx: int, entries: list[dict], cfg: GameConfig, client, mode: str, seed: int,
             out_dir: Path, echo: bool = False, html: bool = True, sink=None, label: str = "",
             seats: list[dict] | None = None, roles: list[str | None] | None = None,
             should_stop=None) -> dict:
    """seats/roles pin the model and role of each seat (the lobby does this); otherwise they're drawn."""
    if seats is None:
        seats = assign_seats(entries, cfg.players, random.Random(seed))
    agents = build_agents(seats, client, mode, seed)
    gdir = out_dir / f"game_{idx:03d}"
    summary = Game(agents, cfg, seed=seed, out_dir=gdir, echo=echo, sink=sink, label=label,
                   roles=roles, should_stop=should_stop).run()
    if html:
        build_replay(gdir)
    summary["dir"] = str(gdir)
    return summary


def run_games(n_games: int, entries: list[dict], cfg: GameConfig, client, mode: str, seed: int,
              out_dir: Path, parallel: int = 1, echo: bool = False, html: bool = True,
              progress=print, sink=None) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    if sink is not None:
        parallel = 1          # the live viewer follows one game at a time

    def job(i: int) -> dict:
        return play_one(i, entries, cfg, client, mode, seed + i, out_dir, echo=echo and parallel == 1,
                        html=html, sink=sink, label=f"Game {i + 1}/{n_games}" if n_games > 1 else "")

    t0 = time.time()
    if parallel <= 1:
        for i in range(n_games):
            s = job(i)
            results.append(s)
            progress(_line(i, n_games, s))
    else:
        with cf.ThreadPoolExecutor(max_workers=parallel) as ex:
            futs = {ex.submit(job, i): i for i in range(n_games)}
            for k, f in enumerate(cf.as_completed(futs), 1):
                s = f.result()
                results.append(s)
                progress(_line(futs[f], n_games, s, done=k))
    results.sort(key=lambda s: s["dir"])
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    progress(f"{n_games} game(s) in {time.time() - t0:.0f}s -> {out_dir}")
    return results


def _line(i: int, n: int, s: dict, done: int | None = None) -> str:
    cost = sum(p["cost"] for p in s["players"])
    tag = f"[{done}/{n}]" if done else f"[{i + 1}/{n}]"
    return (f"{tag} game {i:03d}: {s['winner'].upper():<9} ({s['reason']}) ticks={s['ticks']} "
            f"kills={s['kills']} meetings={s['meetings']} cost=${cost:.4f}")


# --------------------------------------------------------------------------- aggregation

def aggregate(results: list[dict]) -> list[dict]:
    m: dict[str, dict] = defaultdict(lambda: defaultdict(float))
    for s in results:
        for p in s["players"]:
            a = m[p["model"]]
            won = (s["winner"] == "crew") == (p["role"] == "crew")
            role = "crew" if p["role"] == "crew" else "imp"
            a[f"{role}_games"] += 1
            a[f"{role}_wins"] += won
            a[f"{role}_alive"] += p["alive"]
            a["calls"] += p["calls"]
            a["invalid"] += p["invalid"]
            a["tokens_in"] += p["tokens_in"]
            a["tokens_out"] += p["tokens_out"]
            a["cost"] += p["cost"]
            a["games"] += 1
            if role == "crew":
                a["tasks_done"] += p["tasks_done"]
                a["crew_votes"] += p["votes_for_impostor"] + p["votes_for_crew"]
                a["crew_votes_right"] += p["votes_for_impostor"]
            else:
                a["kills"] += p["kills"]
    rows = []
    for model, a in m.items():
        cg, ig = a["crew_games"], a["imp_games"]
        cw = a["crew_wins"] / cg if cg else None
        iw = a["imp_wins"] / ig if ig else None
        parts = [x for x in (cw, iw) if x is not None]
        rows.append({
            "model": model, "games": int(a["games"]),
            "crew_games": int(cg), "crew_win": cw,
            "imp_games": int(ig), "imp_win": iw,
            "overall": sum(parts) / len(parts) if parts else 0.0,
            "vote_accuracy": (a["crew_votes_right"] / a["crew_votes"]) if a["crew_votes"] else None,
            "crew_survival": (a["crew_alive"] / cg) if cg else None,
            "imp_survival": (a["imp_alive"] / ig) if ig else None,
            "kills_per_imp_game": (a["kills"] / ig) if ig else None,
            "invalid_rate": (a["invalid"] / a["calls"]) if a["calls"] else 0.0,
            "cost_per_game": a["cost"] / a["games"] if a["games"] else 0.0,
            "tokens_per_game": (a["tokens_in"] + a["tokens_out"]) / a["games"] if a["games"] else 0.0,
        })
    rows.sort(key=lambda r: r["overall"], reverse=True)
    return rows


def _pct(x) -> str:
    return "-" if x is None else f"{x * 100:.0f}%"


def leaderboard_md(rows: list[dict], n_games: int) -> str:
    out = [f"# Leaderboard ({n_games} games)", "",
           "Wins are team outcomes: every Crewmate shares a Crew win, every Impostor shares an Impostor win. "
           "With few games the numbers are noisy.", "",
           "| Model | Games | Crew win | Imp win | Overall | Crew vote accuracy | Imp survival | Kills / imp game | Invalid replies | $ / game |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        kills = "-" if r["kills_per_imp_game"] is None else f"{r['kills_per_imp_game']:.1f}"
        out.append(f"| {r['model']} | {r['games']} | {_pct(r['crew_win'])} ({r['crew_games']}) | "
                   f"{_pct(r['imp_win'])} ({r['imp_games']}) | {_pct(r['overall'])} | {_pct(r['vote_accuracy'])} | "
                   f"{_pct(r['imp_survival'])} | {kills} | {_pct(r['invalid_rate'])} | ${r['cost_per_game']:.4f} |")
    return "\n".join(out) + "\n"


def write_leaderboard(results: list[dict], out_dir: Path) -> str:
    rows = aggregate(results)
    md = leaderboard_md(rows, len(results))
    (out_dir / "leaderboard.md").write_text(md)
    (out_dir / "leaderboard.json").write_text(json.dumps(rows, indent=2))
    return md


def preflight_errors(client, entries: list[dict]) -> list[tuple[str, str]]:
    """One tiny call per model so a typo'd model id or bad key fails in seconds, not mid-game."""
    from .openrouter import ApiError
    bad = []

    def check(e: dict):
        try:
            client.chat(e["id"], [{"role": "user", "content": 'Reply with exactly: {"ok": true}'}],
                        **{**(e.get("params") or {}), "max_tokens": max(60, int((e.get("params") or {}).get("max_tokens", 60)))})
            return None
        except ApiError as ex:
            return (e["id"], str(ex))

    if not entries:
        return bad
    with cf.ThreadPoolExecutor(max_workers=min(8, len(entries))) as ex:
        for r in ex.map(check, entries):
            if r:
                bad.append(r)
    return bad


def preflight(client, entries: list[dict]) -> None:
    bad = preflight_errors(client, entries)
    if bad:
        msg = "\n".join(f"  - {m}: {err}" for m, err in bad)
        raise SystemExit(f"Preflight failed for:\n{msg}\n(Use `uv run python -m arena models --filter <name>` to look up valid ids.)")
