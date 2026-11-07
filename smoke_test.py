"""Offline smoke tests. No network, no cost.

1. Many games with the mock LLM client + random agents (crash / invariant check).
2. A scripted pacifist run that must end in a Crew win by tasks (checks task bar + win logic).
3. A scripted killer run that must produce kills, a meeting and a vote (checks the meeting flow).
"""
import collections
import re
import sys
import traceback

from arena.agents import Agent, LLMAgent, MockClient, RandomAgent, Reply, Turn
from arena.game import Game, GameConfig


class Scripted(Agent):
    """Does its tasks, repairs the reactor, reports bodies. Optionally kills when it can."""

    def __init__(self, kill: bool = False, label: str = "scripted"):
        self.kill, self.label = kill, label

    def act(self, turn: Turn) -> Reply:
        if turn.kind == "speak":
            return Reply(say="I was doing tasks.")
        opts = re.findall(r"^\s*(\d+)\) (.*)$", turn.user.split("OPTIONS:")[-1], re.M)
        def first(prefix):
            return next((int(n) for n, t in opts if t.startswith(prefix)), None)
        if turn.kind == "vote":
            return Reply(choice=int(opts[-1][0]))          # skip
        for prefix in (("KILL",) if self.kill else ()) + ("REPORT", "Repair", "Do task"):
            c = first(prefix)
            if c:
                return Reply(choice=c)
        m = re.search(r"\[ \] .*? @ ([A-Za-z0-9 ]+?)(?: \(|\s*\[|$)", turn.user, re.M)
        if m:
            c = next((int(n) for n, t in opts if t.startswith(f"Go to {m.group(1).strip()} ")), None)
            if c:
                return Reply(choice=c)
        c = first("Pretend")
        return Reply(choice=c or int(opts[-1][0]))


def test_random(n_games: int = 40) -> int:
    cfg = GameConfig()
    outcomes, bad = collections.Counter(), 0
    for seed in range(n_games):
        client = MockClient(seed=seed)
        agents = [LLMAgent(client, f"mock/model-{i % 3}") if i % 4 else RandomAgent(f"random-{i}", seed=seed + i)
                  for i in range(cfg.players)]
        try:
            s = Game(agents, cfg, seed=seed).run()
            assert s["ticks"] <= cfg.max_ticks
        except Exception:
            traceback.print_exc()
            bad += 1
            continue
        outcomes[(s["winner"], s["reason"])] += 1
    print(f"[random/mock] games={n_games} crashes={bad}")
    for k, v in outcomes.most_common():
        print(f"    {k[0]:<9} {k[1]:<28} {v}")
    return bad


def test_crew_win() -> int:
    cfg = GameConfig(max_ticks=80)
    wins = 0
    for seed in range(5):
        s = Game([Scripted() for _ in range(cfg.players)], cfg, seed=seed).run()
        ok = s["winner"] == "crew" and s["reason"] == "all tasks completed" and s["kills"] == 0
        wins += ok
        print(f"[pacifist seed={seed}] {s['winner']} ({s['reason']}) ticks={s['ticks']} bar={s['task_bar']}")
    return 0 if wins == 5 else 1


def test_killers() -> int:
    cfg = GameConfig()
    meetings = kills = 0
    for seed in range(8):
        s = Game([Scripted(kill=True) for _ in range(cfg.players)], cfg, seed=seed).run()
        meetings += s["meetings"]
        kills += s["kills"]
    print(f"[killers] total kills={kills} meetings={meetings}")
    return 0 if kills > 0 and meetings > 0 else 1


def test_http_client() -> int:
    """OpenRouterClient against a local fake server: auth header, 429 retry, usage/cost parsing, bad key."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from arena.openrouter import FatalApiError, OpenRouterClient

    seen = {"n": 0, "auth": None, "body": None}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen["auth"], seen["body"] = self.headers.get("Authorization"), body
            seen["n"] += 1
            if body["model"] == "bad/key":
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"No auth"}}')
                return
            if seen["n"] == 1:
                self.send_response(429)
                self.send_header("Retry-After", "0")
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"slow down"}}')
                return
            out = {"choices": [{"message": {"content": '```json\n{"thought":"ok","choice":2}\n```'}}],
                   "usage": {"prompt_tokens": 123, "completion_tokens": 45, "cost": 0.00042}}
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(out).encode())

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    fails = 0
    client = OpenRouterClient(api_key="sk-test", base_url=base, max_retries=2)
    agent = LLMAgent(client, "x/y", {"temperature": 0.5, "max_tokens": 99})
    r = agent.act(Turn("action", "sys", "user [[TURN: action]]", n_options=3))
    ok = (r.choice == 2 and r.prompt_tokens == 123 and abs(r.cost - 0.00042) < 1e-9 and r.error is None
          and seen["auth"] == "Bearer sk-test" and seen["body"]["max_tokens"] == 99 and seen["n"] == 2)
    print(f"[http] retry+parse+usage: {'ok' if ok else 'FAILED'} (requests={seen['n']}, reply={r})")
    fails += not ok
    try:
        client.chat("bad/key", [{"role": "user", "content": "hi"}])
        print("[http] bad key: FAILED (no exception)")
        fails += 1
    except FatalApiError:
        print("[http] bad key -> FatalApiError: ok")
    srv.shutdown()
    return fails


def test_live_stream() -> int:
    """Run a game with the live server as the sink and read the SSE stream like the browser does."""
    import json
    import socket
    import urllib.request

    from arena.live import LiveServer

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = LiveServer(port=port)
    srv.start()
    cfg = GameConfig()
    client = MockClient(seed=46)
    agents = [LLMAgent(client, f"mock/m{i}") for i in range(cfg.players)]
    Game(agents, cfg, seed=46, sink=srv.push).run()

    types, last_id = [], None
    with urllib.request.urlopen(f"{srv.url}/events", timeout=10) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if line.startswith("id: "):
                last_id = int(line[4:])
            elif line.startswith("data: "):
                ev = json.loads(line[6:])
                types.append(ev["type"])
                if ev["type"] == "end":
                    break
    with urllib.request.urlopen(srv.url, timeout=10) as resp:
        page = resp.read().decode()
    srv.stop()
    need = {"meta", "thinking", "tick", "meeting_open", "meeting_say", "meeting_end", "end"}
    ok = need <= set(types) and types[0] == "meta" and types[-1] == "end" and "var LIVE = true" in page
    print(f"[live] {len(types)} events, types ok={need <= set(types)}, last id={last_id}: {'ok' if ok else 'FAILED'}")
    return 0 if ok else 1


def test_pinned_roles() -> int:
    """Roles chosen per seat are honoured; unpinned seats fill the remaining Impostor slots; bad setups are refused."""
    from arena.game import validate_roles

    fails = 0
    cfg = GameConfig(players=6, impostors=2)
    cases = [  # roles per seat, expected impostor seats (None = any valid draw)
        (["impostor", "crew", "crew", "impostor", "crew", "crew"], {0, 3}),
        (["impostor", "crew", "random", "random", "crew", "crew"], None),
        ([None] * 6, None),
    ]
    for roles, expect in cases:
        for seed in range(20):
            g = Game([RandomAgent(f"r{i}", seed=i) for i in range(6)], cfg, seed=seed, roles=roles)
            imps = {i for i, p in enumerate(g.players) if p.role == "impostor"}
            ok = len(imps) == 2 and all(roles[i] != "crew" for i in imps)
            ok = ok and (expect is None or imps == expect) and (roles[0] != "impostor" or 0 in imps)
            if not ok:
                print(f"[roles] FAILED roles={roles} seed={seed} -> impostors at {sorted(imps)}")
                fails += 1
                break
    bad = [
        (6, 2, ["impostor"] * 3 + ["crew"] * 3),          # more pinned Impostors than the game has
        (6, 2, ["crew"] * 5 + ["impostor"]),              # not enough seats left to be the second Impostor
        (6, 3, None),                                     # Impostors must be fewer than half
        (6, 2, ["crew"] * 5),                             # wrong length
        (6, 2, ["boss"] + ["crew"] * 5),                  # unknown role
    ]
    for players, imps, roles in bad:
        try:
            validate_roles(players, imps, roles)
            print(f"[roles] FAILED: accepted {players=} {imps=} {roles=}")
            fails += 1
        except ValueError:
            pass
    print(f"[roles] pinned seats honoured, {len(bad)} bad setups refused: {'ok' if not fails else 'FAILED'}")
    return fails


def test_lobby() -> int:
    """The browser setup API: start a mixed mock/random table with pinned roles, reject bad/foreign requests,
    stop mid-game, then start again on the same server (events reset)."""
    import json
    import socket
    import tempfile
    import time
    import urllib.error
    import urllib.request

    from arena.live import LiveServer
    from arena.lobby import Lobby

    with socket.socket() as sk:
        sk.bind(("127.0.0.1", 0))
        port = sk.getsockname()[1]
    srv = LiveServer(port=port, setup=True)
    tmp = tempfile.mkdtemp()
    srv.api = Lobby(srv, GameConfig(), [{"id": "x/never-called", "label": None, "params": {}}], {}, out_dir=tmp,
                    mock_latency=0.2)
    srv.start()

    def call(path, body=None, headers=None):
        h = {"Content-Type": "application/json", **(headers or {})}
        req = urllib.request.Request(srv.url + path, None if body is None else json.dumps(body).encode(),
                                     h, method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.load(r)
        except urllib.error.HTTPError as e:
            return e.code, json.load(e)

    def first_events(n=400, secs=3):
        out, t0 = [], time.time()
        with urllib.request.urlopen(srv.url + "/events", timeout=secs) as r:
            try:
                for raw in r:
                    if raw.startswith(b"data: "):
                        out.append(json.loads(raw[6:]))
                    if len(out) >= n or time.time() - t0 > secs:
                        break
            except OSError:
                pass
        return out

    seats = [{"model": m, "role": r} for m, r in [("mock/bot", "impostor"), ("mock/bot", "crew"), ("random", "crew"),
                                                  ("mock/bot", "impostor"), ("random", "random"), ("mock/bot", "random")]]
    checks = {}
    checks["state"] = call("/api/state")[1]["phase"] == "idle"
    checks["foreign origin refused"] = call("/api/start", {"seats": seats}, {"Origin": "http://evil.example"})[0] == 403
    checks["non-json refused"] = call("/api/start", {"seats": seats}, {"Content-Type": "text/plain"})[0] == 415
    two_imps = [{"model": "mock/bot", "role": "impostor"}] * 2 + [{"model": "mock/bot", "role": "crew"}] * 2
    checks["too many pinned Impostors -> 400"] = call("/api/start", {"impostors": 1, "seats": two_imps})[0] == 400
    checks["bad model id -> 400"] = call("/api/start", {"impostors": 1, "seats": [{"model": "bad id!"}] * 4})[0] == 400
    st, r = call("/api/start", {"impostors": 2, "seats": seats})
    checks["start ok"] = st == 200
    time.sleep(0.8)
    checks["second start refused while running"] = call("/api/start", {"impostors": 1, "seats": [{"model": "mock/bot"}] * 4})[0] == 400
    ev = first_events(3)
    meta = ev[0] if ev else {}
    checks["meta first"] = meta.get("type") == "meta"
    checks["models+roles in meta"] = [(p["model"], p["role"]) for p in meta.get("players", [])][:4] == [
        ("mock-bot", "impostor"), ("mock-bot", "crew"), ("random-bot", "crew"), ("mock-bot", "impostor")]
    call("/api/stop", {})
    deadline = time.time() + 10
    while time.time() < deadline and call("/api/state")[1]["phase"] == "running":
        time.sleep(0.1)
    checks["stopped"] = call("/api/state")[1]["phase"] == "finished"
    checks["error event pushed"] = any(e["type"] == "error" for e in first_events(2000, 2))
    st, _ = call("/api/start", {"impostors": 1, "seats": [{"model": "mock/bot", "role": "impostor"}] + [{"model": "mock/bot"}] * 3})
    time.sleep(0.5)
    ev = first_events(2)
    checks["new game resets the stream"] = st == 200 and ev and ev[0]["type"] == "meta" and sum(e["type"] == "meta" for e in ev) == 1 \
        and len(ev[0]["players"]) == 4
    call("/api/stop", {})
    time.sleep(1)
    srv.stop()
    bad = [k for k, v in checks.items() if not v]
    print(f"[lobby] {len(checks) - len(bad)}/{len(checks)} checks ok" + (f"; FAILED: {bad}" if bad else ""))
    return len(bad)


class Pick(Agent):
    """Always takes the first option starting with `prefix` (else waits); skips every vote."""

    def __init__(self, prefix: str):
        self.prefix, self.label = prefix, f"pick-{prefix}"

    def act(self, turn: Turn) -> Reply:
        if turn.kind == "speak":
            return Reply(say="...")
        opts = re.findall(r"^\s*(\d+)\) (.*)$", turn.user.split("OPTIONS:")[-1], re.M)
        if turn.kind == "vote":
            return Reply(choice=int(opts[-1][0]))
        for n, t in opts:
            if t.startswith(self.prefix):
                return Reply(choice=int(n))
        return Reply(choice=next(int(n) for n, t in opts if t.startswith("Wait")))


def test_report_vs_kill() -> int:
    """Pink reports a body in the same tick Red kills Black (alone together). No official tie-break, so it is a
    coin flip: kill first (parity -> Impostors win, no meeting) or report first (kill cancelled, labelled honestly)."""
    from arena.game import Body

    seen, bad = collections.Counter(), []
    for seed in range(60):
        cfg = GameConfig()
        roles = ["impostor", "crew", "impostor", "crew", "crew", "crew", "crew"]   # Red, Blue, Green, Pink, Orange, Yellow, Black
        agents = [Pick("KILL") if i == 0 else Pick("REPORT") if i == 3 else Pick("Wait") for i in range(cfg.players)]
        g = Game(agents, cfg, seed=seed, roles=roles)
        for p in g.players:
            p.alive = p.name in ("Red", "Pink", "Black")
            p.room = "Cafeteria"
        g.by_name["Pink"].room = "MedBay"
        g.by_name["Red"].kill_cd = 0
        g.bodies.append(Body("Orange", "MedBay", 0))
        g._tick()
        frame = g.frames[0]
        red_label, black = frame["acts"]["Red"], g.by_name["Black"]
        if black.alive:                          # report first
            seen["report-first"] += 1
            ok = (g.meetings == 1 and g.kills == 0 and red_label.startswith("(cancelled")
                  and any("did not happen" in k for k in g.by_name["Red"].key))
        else:                                    # kill first: parity, so the game ends before any meeting
            seen["kill-first"] += 1
            ok = (g.kills == 1 and g.meetings == 0 and red_label == "KILL Black"
                  and g.result == ("impostors", "Impostors reached parity"))
        if not ok:
            bad.append((seed, red_label, g.kills, g.meetings, g.result))
    both = seen["report-first"] > 5 and seen["kill-first"] > 5
    print(f"[tie] report-first={seen['report-first']} kill-first={seen['kill-first']}"
          f"{' (inconsistent: ' + str(bad[:2]) + ')' if bad else ''}: {'ok' if both and not bad else 'FAILED'}")
    return 0 if both and not bad else 1


if __name__ == "__main__":
    fails = (test_random(int(sys.argv[1]) if len(sys.argv) > 1 else 40) + test_crew_win()
             + test_killers() + test_http_client() + test_live_stream() + test_pinned_roles() + test_report_vs_kill() + test_lobby())
    print("ALL OK" if not fails else f"FAILURES: {fails}")
    sys.exit(1 if fails else 0)
