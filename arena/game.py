"""The Among Us engine.

The harness does the boring parts (pathfinding, who can see what, cooldowns, vote counting)
and hands every model a numbered list of legal actions each tick.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import random
import time
from collections import Counter
from dataclasses import dataclass, field, fields
from pathlib import Path

from . import prompts, world
from .agents import Agent, Reply, Turn
from .openrouter import FatalApiError


class GameAborted(Exception):
    """Raised inside a running game when its should_stop() callback returns True."""


COLORS = ["Red", "Blue", "Green", "Pink", "Orange", "Yellow", "Black", "White", "Purple", "Brown", "Cyan", "Lime"]


@dataclass
class GameConfig:
    players: int = 7
    impostors: int = 2
    max_ticks: int = 60
    tasks_per_crew: int = 4
    fake_tasks: int = 3
    kill_cooldown: int = 3
    sabotage_cooldown: int = 8
    sabotage_timer: int = 8
    discussion_rounds: int = 2
    confirm_ejects: bool = True
    anonymous_votes: bool = False
    timeout_winner: str = "impostors"      # who wins if the tick limit is reached
    concurrency: int = 8                   # parallel LLM calls inside one game
    max_memory_lines: int = 40             # recent-log lines shown to a model
    max_statement_chars: int = 500

    @classmethod
    def from_dict(cls, d: dict) -> "GameConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def validate_roles(players: int, impostors: int, roles: list[str | None] | None = None) -> None:
    """Raise ValueError if this player/impostor/seat-role combination can't be played.

    roles has one entry per seat: "crew", "impostor", or None/"random" (the engine decides).
    """
    if impostors < 1 or impostors * 2 >= players:
        raise ValueError("impostors must be >= 1 and fewer than half the players")
    if roles is None:
        return
    if len(roles) != players:
        raise ValueError(f"need a role for each of the {players} seats, got {len(roles)}")
    bad = [r for r in roles if r not in (None, "random", "crew", "impostor")]
    if bad:
        raise ValueError(f"unknown role {bad[0]!r}")
    picked = sum(r == "impostor" for r in roles)
    free = sum(r in (None, "random") for r in roles)
    if picked > impostors:
        raise ValueError(f"{picked} seats are Impostors but the game only has {impostors}")
    if impostors - picked > free:
        raise ValueError(f"{impostors} Impostors needed but only {picked + free} seats can be one "
                         "(too many seats are locked to Crew)")


@dataclass
class TaskState:
    d: world.TaskDef
    done: int = 0

    @property
    def complete(self) -> bool:
        return self.done >= self.d.steps


@dataclass
class Body:
    name: str
    room: str
    tick: int


@dataclass
class Option:
    kind: str
    label: str
    data: dict = field(default_factory=dict)


class Player:
    def __init__(self, idx: int, name: str, role: str, agent: Agent):
        self.idx, self.name, self.role, self.agent = idx, name, role, agent
        self.room = world.HUB
        self.alive = True
        self.tasks: list[TaskState] = []
        self.fake: list[world.TaskDef] = []
        self.route: list[str] = []
        self.log: list[str] = []        # per-tick observations (rolling window shown to the model)
        self.key: list[str] = []        # important facts (kept much longer)
        self.chats: list[str] = []      # digests of past meetings
        self.notes = ""
        self.kill_cd = 0
        self.buttons = 1
        self.done_at_death = 0
        self.system = ""
        self.votes: list[str | None] = []
        self.stats = dict(calls=0, invalid=0, tokens_in=0, tokens_out=0, cost=0.0, kills=0, tasks_done=0)

    def remember(self, text: str, key: bool = False) -> None:
        (self.key if key else self.log).append(text)

    def steps_done(self) -> int:
        return sum(t.done for t in self.tasks)


class Game:
    def __init__(self, agents: list[Agent], cfg: GameConfig, seed: int | None = None,
                 out_dir: str | Path | None = None, echo: bool = False,
                 sink=None, label: str = "", roles: list[str | None] | None = None,
                 should_stop=None):
        if len(agents) != cfg.players:
            raise ValueError(f"need exactly {cfg.players} agents, got {len(agents)}")
        validate_roles(cfg.players, cfg.impostors, roles)
        self.cfg, self.seed, self.echo = cfg, seed, echo
        self.sink = sink            # callable(event: dict) for the live viewer
        self.should_stop = should_stop   # callable() -> bool; True aborts the game with GameAborted
        self.label = label
        self.rng = random.Random(seed)
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._tx = open(self.out_dir / "transcript.txt", "w", buffering=1)
        else:
            self._tx = None

        n = cfg.players
        if roles is None:
            imps = set(self.rng.sample(range(n), cfg.impostors))
        else:                       # seats the caller pinned keep their role; the rest are drawn at random
            imps = {i for i, r in enumerate(roles) if r == "impostor"}
            free = [i for i, r in enumerate(roles) if r in (None, "random")]
            imps |= set(self.rng.sample(free, cfg.impostors - len(imps)))
        self.players = [Player(i, COLORS[i], "impostor" if i in imps else "crew", agents[i]) for i in range(n)]
        self.by_name = {p.name: p for p in self.players}
        non_visual = [t for t in world.TASKS if not t.visual]
        for p in self.players:
            if p.role == "crew":
                p.tasks = [TaskState(t) for t in self.rng.sample(world.TASKS, cfg.tasks_per_crew)]
            else:
                p.fake = self.rng.sample(non_visual, cfg.fake_tasks)
            p.kill_cd = cfg.kill_cooldown
        for p in self.players:
            allies = [q.name for q in self.players if q.role == "impostor" and q is not p] if p.role == "impostor" else []
            p.system = prompts.system_prompt(p.name, p.role, allies, cfg, n, cfg.impostors)

        self.tick = 0
        self.bodies: list[Body] = []
        self.sab_active = False
        self.sab_timer = 0
        self.sab_progress = 0
        self.sab_cd = cfg.sabotage_cooldown
        self.meetings = 0
        self.kills = 0
        self.ejections: list[dict] = []
        self.result: tuple[str, str] | None = None
        self.frames: list[dict] = []
        self._evs: dict[int, list[str]] = {}
        self._intr: set[int] = set()

    # ------------------------------------------------------------------ helpers
    def alive(self) -> list[Player]:
        return [p for p in self.players if p.alive]

    def alive_in(self, room: str) -> list[Player]:
        return [p for p in self.players if p.alive and p.room == room]

    def bar(self) -> float:
        done = total = 0
        for p in self.players:
            if p.role != "crew":
                continue
            if p.alive:
                done += p.steps_done()
                total += sum(t.d.steps for t in p.tasks)
            else:
                done += p.done_at_death
                total += p.done_at_death
        return done / total if total else 1.0

    def _say(self, line: str) -> None:
        if self._tx:
            self._tx.write(line + "\n")
        if self.echo:
            print(line, flush=True)

    def _emit(self, event: dict) -> None:
        if self.sink:
            try:
                self.sink(event)
            except Exception:  # a broken viewer must never break a game
                self.sink = None

    def meta(self, winner: str | None = None, reason: str | None = None) -> dict:
        return {
            "winner": winner, "reason": reason, "seed": self.seed, "label": self.label,
            "players": [{"name": p.name, "model": p.agent.label, "role": p.role} for p in self.players],
            "rooms": world.ROOMS,
            "edges": sorted({tuple(sorted((a, b))) for a in world.ADJ for b in world.ADJ[a]}),
            "vents": dict(world.VENTS), "max_ticks": self.cfg.max_ticks,
        }

    def _ev(self, q: Player, text: str, key: bool = False) -> None:
        if not q.alive:
            return
        self._evs[q.idx].append(text)
        if key:
            q.key.append(f"t{self.tick}: {text}")

    def _interrupt(self, q: Player) -> None:
        q.route = []
        self._intr.add(q.idx)

    # ------------------------------------------------------------------ prompts / options
    def _options(self, p: Player) -> list[Option]:
        opts: list[Option] = []

        def add(kind: str, label: str, **data) -> None:
            opts.append(Option(kind, label, data))

        room = p.room
        for b in self.bodies:
            if b.room == room:
                add("report", f"REPORT the body of {b.name} (starts a meeting)", body=b.name)
        if room == world.HUB and p.buttons > 0 and not self.sab_active:
            add("button", "Press the emergency button (starts a meeting)")
        if p.role == "crew":
            for ts in p.tasks:
                if ts.d.room == room and not ts.complete:
                    step = f" (step {ts.done + 1} of {ts.d.steps})" if ts.d.steps > 1 else ""
                    vis = " [VISUAL]" if ts.d.visual else ""
                    add("task", f"Do task: {ts.d.name}{step}{vis}", ts=ts)
            if self.sab_active and room == world.REACTOR:
                add("repair", f"Repair the reactor (progress {self.sab_progress}/2)")
        else:
            for fd in p.fake:
                if fd.room == room:
                    add("fake", f"Pretend to do task: {fd.name} (fake, no effect)", name=fd.name)
            if p.kill_cd == 0:
                for q in self.alive_in(room):
                    if q.role == "crew":
                        add("kill", f"KILL {q.name}", target=q.name)
            for dest in world.VENTS.get(room, []):
                add("vent", f"Use the vent to {dest}", dest=dest)
            if not self.sab_active and self.sab_cd == 0:
                add("sabotage", "Sabotage the reactor (starts a meltdown timer)")
        if room == world.SECURITY:
            add("cameras", f"Watch the security cameras ({', '.join(world.CAMERA_ROOMS)})")
        if room == world.ADMIN:
            add("admin", "Check the Admin map (player count per room)")
        add("wait", "Wait here for one tick")
        for r in sorted(world.ROOMS, key=lambda r: (world.distance(room, r), r)):
            if r != room:
                d = world.distance(room, r)
                add("go", f"Go to {r} ({d} step{'s' if d > 1 else ''}, walked automatically)", dest=r)
        return opts

    def _common(self, p: Player) -> dict:
        if p.role == "crew":
            title = "YOUR TASKS:"
            tasks = [f"[{'x' if t.complete else ' '}] {t.d.name} @ {t.d.room}"
                     + (f" ({t.done}/{t.d.steps})" if t.d.steps > 1 else "")
                     + (" [VISUAL]" if t.d.visual else "") for t in p.tasks]
        else:
            allies = ", ".join(f"{q.name} ({'alive' if q.alive else 'dead'})"
                               for q in self.players if q.role == "impostor" and q is not p) or "none"
            title = f"YOUR FAKE TASKS (cover story only). Fellow Impostors: {allies}."
            tasks = [f"- {t.name} @ {t.room}" for t in p.fake]
        sab = []
        if self.sab_active:
            sab.append(f"!! REACTOR MELTDOWN in {self.sab_timer} tick(s) - repair progress {self.sab_progress}/2. "
                       "Crewmates must repair it in the Reactor room !!")
        if p.role == "impostor":
            sab.append(f"Your kill cooldown: {p.kill_cd} tick(s). Sabotage cooldown: {self.sab_cd} tick(s).")
        return dict(
            alive=[q.name for q in self.alive()],
            dead=[q.name for q in self.players if not q.alive],
            tasks_title=title, tasks=tasks, notes=p.notes,
            key=p.key[-60:], log=p.log[-self.cfg.max_memory_lines:], prev_chats=p.chats[-2:],
            sab_line="\n".join(sab),
        )

    # ------------------------------------------------------------------ LLM plumbing
    def _ask_one(self, p: Player, turn: Turn) -> Reply:
        if self.should_stop and self.should_stop():
            raise GameAborted()
        try:
            return p.agent.act(turn)
        except FatalApiError:
            raise
        except Exception as e:  # a buggy agent must not kill the game
            return Reply(error=f"{type(e).__name__}: {e}")

    def _ask_many(self, turns: dict[int, Turn]) -> dict[int, Reply]:
        if len(turns) <= 1 or self.cfg.concurrency <= 1:
            return {i: self._ask_one(self.players[i], t) for i, t in turns.items()}
        with cf.ThreadPoolExecutor(max_workers=min(self.cfg.concurrency, len(turns))) as ex:
            futs = {i: ex.submit(self._ask_one, self.players[i], t) for i, t in turns.items()}
            return {i: f.result() for i, f in futs.items()}

    def _account(self, p: Player, r: Reply) -> None:
        s = p.stats
        s["calls"] += 1
        s["tokens_in"] += r.prompt_tokens
        s["tokens_out"] += r.completion_tokens
        s["cost"] += r.cost
        if r.error:
            s["invalid"] += 1
        if r.notes:
            p.notes = r.notes

    # ------------------------------------------------------------------ main loop
    def run(self) -> dict:
        t0 = time.time()
        self._say(f"=== GAME START (seed={self.seed}) ===")
        self._emit({"type": "meta", **self.meta()})
        for p in self.players:
            extra = f" fake tasks: {[t.name for t in p.fake]}" if p.role == "impostor" else \
                f" tasks: {[t.d.name for t in p.tasks]}"
            self._say(f"  {p.name:<7} {p.role.upper():<9} model={p.agent.label}{extra}")
        while self.result is None:
            self._tick()
            if self.result is None:
                self._check_win()
            if self.result is None and self.tick >= self.cfg.max_ticks:
                self.result = (self.cfg.timeout_winner, "time ran out")
        winner, reason = self.result
        self._say(f"=== GAME OVER: {winner.upper()} WIN ({reason}) at tick {self.tick} ===")
        end = {"type": "end", "t": self.tick, "winner": winner, "reason": reason}
        self.frames.append(end)
        self._emit(end)
        summary = self._summary(winner, reason, time.time() - t0)
        self._write_outputs(summary)
        return summary

    def _tick(self) -> None:
        cfg = self.cfg
        self.tick += 1
        for p in self.alive():
            p.kill_cd = max(0, p.kill_cd - 1)
        self.sab_cd = max(0, self.sab_cd - 1)
        if self.sab_active:
            self.sab_timer -= 1
            if self.sab_timer <= 0:
                self.result = ("impostors", "reactor meltdown")
                return
        self._evs = {p.idx: [] for p in self.players}
        self._intr = set()
        alive = self.alive()
        self._say(f"\n--- tick {self.tick} | bar {self.bar():.0%} | "
                  + ", ".join(f"{p.name}@{p.room}" for p in alive))

        # 1. ask everyone who is not mid-walk
        deciders = [p for p in alive if not p.route]
        opts_by, turns = {}, {}
        for p in deciders:
            opts = self._options(p)
            opts_by[p.idx] = opts
            s = self._common(p)
            s.update(tick=self.tick, max_ticks=cfg.max_ticks, bar=round(self.bar() * 100), room=p.room,
                     adj=world.ADJ[p.room], here=[q.name for q in self.alive_in(p.room) if q is not p],
                     bodies=[b.name for b in self.bodies if b.room == p.room], options=[o.label for o in opts])
            turns[p.idx] = Turn("action", p.system, prompts.action_user(s), len(opts))
        self._emit({"type": "thinking", "names": [p.name for p in deciders]})
        replies = self._ask_many(turns)

        act: dict[int, Option] = {}
        acts_label: dict[str, str] = {}
        thoughts: dict[str, str] = {}
        for p in deciders:
            r = replies[p.idx]
            self._account(p, r)
            opts = opts_by[p.idx]
            if r.choice is not None and 1 <= r.choice <= len(opts):
                o = opts[r.choice - 1]
            else:
                o = next(x for x in opts if x.kind == "wait")
            act[p.idx] = o
            acts_label[p.name] = o.label
            thoughts[p.name] = r.thought
            flag = "  [INVALID->wait]" if r.error else ""
            self._say(f"  {p.name} ({p.role[0].upper()}): {o.label}{flag}"
                      + (f"\n      thought: {r.thought[:240]}" if r.thought else ""))
        for p in alive:
            if p.idx not in act:
                hop = p.route.pop(0)
                act[p.idx] = Option("auto", f"(walking to {hop})", {"dest": hop})
                acts_label[p.name] = f"(walking via {hop})"

        # 2. a report / emergency button pre-empts the rest of the tick. If an Impostor tried to kill in the
        #    same tick there is no official tie-break (the real game's host handles whichever input arrives
        #    first), so a coin flip decides: kills land first, or the meeting wins and the kills never happen.
        done_kills: set[int] = set()
        trig = [p for p in alive if act[p.idx].kind in ("report", "button")]
        if trig:
            killers = [p for p in alive if act[p.idx].kind == "kill"]
            if killers and self.rng.random() < 0.5:
                self.rng.shuffle(killers)
                for p in killers:
                    self._do_kill(p, act[p.idx])
                    done_kills.add(p.idx)
                for p in alive:         # a victim's own action (even a report) never happened
                    if not p.alive:
                        acts_label[p.name] = f"(killed before they could) {acts_label[p.name]}"
                trig = [p for p in trig if p.alive]
                self._check_win()
            if self.result is not None or trig:         # the tick is cut short here
                for p in alive:         # say honestly what did not happen, in the viewer and to the model
                    o = act[p.idx]
                    if not p.alive or o.kind in ("report", "button", "auto") or p.idx in done_kills:
                        continue
                    if o.kind not in ("go", "wait"):
                        p.remember(f"t{self.tick}: Your action '{o.label}' did not happen: "
                                   "a meeting was called first.", key=True)
                    acts_label[p.name] = f"(cancelled, meeting called first) {acts_label[p.name]}"
                god = [f"{n}: {t}" for n, t in acts_label.items() if t.startswith(("KILL", "Use the vent", "Sabotage"))]
                self._frame(acts_label, thoughts, god)
                if self.result is None:
                    caller = self.rng.choice(trig)
                    self._meeting(caller, act[caller.idx])
                return
            # every would-be caller was killed: no meeting, the tick carries on below

        start = {p.idx: p.room for p in alive}
        order = list(alive)
        self.rng.shuffle(order)

        # 3. stationary actions (kills first), based on start-of-tick positions
        for p in order:
            if act[p.idx].kind == "kill" and p.idx not in done_kills:
                self._do_kill(p, act[p.idx])
        for p in order:
            if not p.alive:
                continue
            o = act[p.idx]
            if o.kind == "task":
                self._do_task(p, o)
            elif o.kind == "fake":
                for q in self.alive_in(p.room):
                    if q is not p:
                        self._ev(q, f"{p.name} is working at the {o.data['name']} station.")
            elif o.kind == "sabotage":
                self._do_sabotage(p)
            elif o.kind == "repair":
                self._do_repair(p)
            elif o.kind == "cameras":
                snap = "; ".join(f"{r}: {', '.join(q.name for q in self.alive_in(r)) or 'nobody'}"
                                 for r in world.CAMERA_ROOMS)
                self._ev(p, f"CAMERAS show -> {snap}", key=True)
            elif o.kind == "admin":
                counts = "; ".join(f"{r}: {len(self.alive_in(r))}" for r in world.ROOMS if self.alive_in(r))
                self._ev(p, f"ADMIN MAP counts -> {counts}", key=True)

        # 4. movement
        moves: dict[int, tuple[str, bool]] = {}
        for p in order:
            if not p.alive:
                continue
            o = act[p.idx]
            if o.kind == "go":
                route = world.path(p.room, o.data["dest"])
                moves[p.idx] = (route[0], False)
                p.route = route[1:]
            elif o.kind == "auto":
                moves[p.idx] = (o.data["dest"], False)
            elif o.kind == "vent":
                moves[p.idx] = (o.data["dest"], True)
                p.route = []
        for p in order:
            if p.alive and p.idx in moves:
                p.room = moves[p.idx][0]
        for p in order:
            if p.alive and p.idx in moves:
                dest, vent = moves[p.idx]
                old = start[p.idx]
                for q in self.alive():
                    if q is p:
                        continue
                    qs = start.get(q.idx)
                    if vent:
                        if qs == old:
                            self._ev(q, f"{p.name} VANISHED INTO A VENT in {old}!", key=True)
                            self._interrupt(q)
                        if q.room == dest and qs != dest:
                            self._ev(q, f"{p.name} POPPED OUT OF A VENT in {dest}!", key=True)
                            self._interrupt(q)
                    else:
                        if qs == old and q.room != dest:
                            self._ev(q, f"{p.name} left {old} heading toward {dest}.")
                        if q.room == dest and qs != old:
                            self._ev(q, f"{p.name} entered {dest} from {old}.")
                found = [b.name for b in self.bodies if b.room == dest]
                if found:
                    self._ev(p, f"You found the body of {', '.join(found)} in {dest}!", key=True)
                    self._interrupt(p)
        for idx in self._intr:
            self.players[idx].route = []

        # 5. everyone writes down what they observed this tick
        alive_now = self.alive()
        for q in alive_now:
            here = [x.name for x in alive_now if x.room == q.room and x is not q]
            began = start.get(q.idx, q.room)
            line = f"t{self.tick}: " + (f"moved {began} -> {q.room}" if began != q.room else f"stayed in {q.room}")
            if self._evs[q.idx]:
                line += ". During the tick: " + " ".join(e if e.endswith((".", "!")) else e + "." for e in self._evs[q.idx])
            else:
                line += "."
            line += f" Now in {q.room} with: {', '.join(here) or 'nobody'}."
            bodies = [b.name for b in self.bodies if b.room == q.room]
            if bodies:
                line += f" BODY of {', '.join(bodies)} is here."
            q.log.append(line)
        god = [f"{n}: {t}" for n, t in acts_label.items() if t.startswith(("KILL", "Use the vent", "Sabotage"))]
        self._frame(acts_label, thoughts, god)

    # ------------------------------------------------------------------ action handlers
    def _do_kill(self, p: Player, o: Option) -> None:
        target = self.by_name[o.data["target"]]
        if not (p.alive and target.alive and target.room == p.room and p.kill_cd == 0):
            self._ev(p, f"Your attack on {target.name} failed (they were already gone).")
            return
        target.alive = False
        target.done_at_death = target.steps_done()
        target.route = []
        self.bodies.append(Body(target.name, p.room, self.tick))
        p.kill_cd = self.cfg.kill_cooldown
        p.stats["kills"] += 1
        self.kills += 1
        self._say(f"  !! {p.name} KILLED {target.name} in {p.room} (witnesses: "
                  f"{', '.join(q.name for q in self.alive_in(p.room) if q is not p) or 'none'})")
        for q in self.alive_in(p.room):
            if q is not p:
                self._ev(q, f"You SAW {p.name} KILL {target.name} in {p.room}!", key=True)
                self._interrupt(q)

    def _do_task(self, p: Player, o: Option) -> None:
        ts: TaskState = o.data["ts"]
        if ts.complete:
            return
        ts.done += 1
        if ts.complete:
            p.stats["tasks_done"] += 1
            self._ev(p, f"You completed '{ts.d.name}'.")
        else:
            self._ev(p, f"You made progress on '{ts.d.name}' ({ts.done}/{ts.d.steps}).")
        for q in self.alive_in(p.room):
            if q is p:
                continue
            if ts.d.visual:
                self._ev(q, f"{p.name} completed the VISUAL task '{ts.d.name}' "
                            f"(visible effect, so {p.name} is a real Crewmate)", key=True)
            else:
                self._ev(q, f"{p.name} is working at the {ts.d.name} station.")

    def _do_sabotage(self, p: Player) -> None:
        if self.sab_active or self.sab_cd > 0:
            return
        self.sab_active = True
        self.sab_timer = self.cfg.sabotage_timer
        self.sab_progress = 0
        self.sab_cd = self.cfg.sabotage_cooldown
        self._say(f"  !! {p.name} SABOTAGED the reactor")
        for q in self.alive():
            self._ev(q, f"ALERT: the REACTOR is melting down! {self.sab_timer} ticks left. "
                        "Crewmates must go to the Reactor and repair it.", key=True)
            self._interrupt(q)

    def _do_repair(self, p: Player) -> None:
        if not self.sab_active or p.room != world.REACTOR:
            return
        self.sab_progress += 1
        self._ev(p, f"You worked on the reactor repair ({self.sab_progress}/2).")
        for q in self.alive_in(p.room):
            if q is not p:
                self._ev(q, f"{p.name} is working at the Reactor console.")
        if self.sab_progress >= 2:
            self.sab_active = False
            self._say("  -- reactor stabilized")
            for q in self.alive():
                self._ev(q, "The reactor has been stabilized.", key=True)

    # ------------------------------------------------------------------ meetings
    def _meeting(self, caller: Player, opt: Option) -> None:
        cfg = self.cfg
        self.meetings += 1
        n = self.meetings
        if opt.kind == "report":
            b = next(b for b in self.bodies if b.name == opt.data["body"])
            header = f"{caller.name} reported the body of {b.name}, found in {b.room}."
        else:
            caller.buttons -= 1
            header = f"{caller.name} pressed the emergency button."
        self._say(f"\n##### MEETING #{n} (tick {self.tick}): {header}")
        self._emit({"type": "meeting_open", "no": n, "header": header, "caller": caller.name, "t": self.tick})
        self.bodies = []
        self.sab_active = False
        for p in self.alive():
            p.room = world.HUB
            p.route = []
        alive = self.alive()

        chat: list[tuple[str, str]] = []
        chat_frames: list[dict] = []
        for rnd in range(1, cfg.discussion_rounds + 1):
            order = list(alive)
            self.rng.shuffle(order)
            if rnd == 1:
                order.remove(caller)
                order.insert(0, caller)
            for p in order:
                s = self._common(p)
                s.update(meeting_no=n, header=header, chat=chat, round=rnd, rounds=cfg.discussion_rounds)
                self._emit({"type": "thinking", "names": [p.name]})
                r = self._ask_one(p, Turn("speak", p.system, prompts.speak_user(s), 0, cfg.max_statement_chars))
                self._account(p, r)
                text = r.say or "..."
                chat.append((p.name, text))
                chat_frames.append({"name": p.name, "say": text, "thought": r.thought, "round": rnd})
                self._emit({"type": "meeting_say", **chat_frames[-1]})
                self._say(f"  [{p.name} ({p.role[0].upper()})]: {text}"
                          + (f"\n      thought: {r.thought[:240]}" if r.thought else ""))

        turns, others_by = {}, {}
        for p in alive:
            others = [q for q in alive if q is not p]
            others_by[p.idx] = others
            labels = [f"Vote to eject {q.name}" for q in others] + ["Skip (vote for nobody)"]
            s = self._common(p)
            s.update(meeting_no=n, header=header, chat=chat, options=labels)
            turns[p.idx] = Turn("vote", p.system, prompts.vote_user(s), len(labels))
        self._emit({"type": "thinking", "names": [p.name for p in alive]})
        replies = self._ask_many(turns)
        votes: dict[str, str | None] = {}
        for p in alive:
            r = replies[p.idx]
            self._account(p, r)
            others = others_by[p.idx]
            target = others[r.choice - 1].name if r.choice and 1 <= r.choice <= len(others) else None
            votes[p.name] = target
            p.votes.append(target)
            self._say(f"  vote: {p.name} -> {target or 'skip'}" + (f"   ({r.thought[:160]})" if r.thought else ""))

        tally = Counter(v for v in votes.values() if v)
        skips = sum(1 for v in votes.values() if v is None)
        ejected = None
        if tally:
            top = max(tally.values())
            leaders = [k for k, c in tally.items() if c == top]
            if len(leaders) == 1 and top > skips:
                ejected = leaders[0]
        if ejected:
            ej = self.by_name[ejected]
            ej.alive = False
            ej.done_at_death = ej.steps_done()
            ej.route = []
            if cfg.confirm_ejects:
                result = f"{ejected} was ejected. {ejected} was {'AN IMPOSTOR' if ej.role == 'impostor' else 'NOT an Impostor'}."
            else:
                result = f"{ejected} was ejected."
            self.ejections.append({"meeting": n, "tick": self.tick, "name": ejected, "role": ej.role})
        else:
            result = "No one was ejected (tie or skip)."
        self._say(f"  RESULT: {result}  [true role: {self.by_name[ejected].role if ejected else '-'}]")

        if cfg.anonymous_votes:
            vote_str = ", ".join(f"{k}: {c}" for k, c in tally.most_common()) + f", skip: {skips}"
        else:
            vote_str = "; ".join(f"{k}->{v or 'skip'}" for k, v in votes.items())
        digest = (f"--- Meeting #{n} (tick {self.tick}) ---\n{header}\n"
                  + "\n".join(f'{a}: "{t}"' for a, t in chat)
                  + f"\nVotes: {vote_str}\nResult: {result}")
        for p in self.alive():
            p.chats.append(digest)
            p.remember(f"t{self.tick}: Meeting #{n}: {header} Result: {result}", key=True)
            p.room = world.HUB
            p.kill_cd = cfg.kill_cooldown
        self.sab_cd = cfg.sabotage_cooldown

        frame = {
            "type": "meeting", "t": self.tick, "no": n, "header": header, "caller": caller.name,
            "chat": chat_frames, "votes": votes, "result": result, "ejected": ejected,
            "ejected_role": self.by_name[ejected].role if ejected else None,
            "alive": {p.name: p.alive for p in self.players},
            "bar": self.bar(),
        }
        self.frames.append(frame)
        self._emit({**frame, "type": "meeting_end"})
        self._check_win()

    # ------------------------------------------------------------------ win logic / output
    def _check_win(self) -> None:
        crew = sum(1 for p in self.alive() if p.role == "crew")
        imps = sum(1 for p in self.alive() if p.role == "impostor")
        if imps == 0:
            self.result = ("crew", "all Impostors ejected")
        elif imps >= crew:
            self.result = ("impostors", "Impostors reached parity")
        elif self.bar() >= 1.0:
            self.result = ("crew", "all tasks completed")

    def _frame(self, acts: dict, thoughts: dict, god: list[str]) -> None:
        frame = {
            "type": "tick", "t": self.tick,
            "pos": {p.name: p.room for p in self.players},
            "alive": {p.name: p.alive for p in self.players},
            "bodies": [{"name": b.name, "room": b.room} for b in self.bodies],
            "bar": round(self.bar(), 3),
            "sab": self.sab_timer if self.sab_active else None,
            "acts": acts, "thoughts": thoughts, "events": god,
        }
        self.frames.append(frame)
        self._emit(frame)

    def _summary(self, winner: str, reason: str, secs: float) -> dict:
        players = []
        for p in self.players:
            real_imps = {q.name for q in self.players if q.role == "impostor"}
            casts = [v for v in p.votes if v]
            players.append({
                "name": p.name, "model": p.agent.label, "role": p.role, "alive": p.alive,
                "tasks_done": p.stats["tasks_done"], "kills": p.stats["kills"],
                "votes_cast": len(p.votes), "votes_for_impostor": sum(1 for v in casts if v in real_imps),
                "votes_for_crew": sum(1 for v in casts if v not in real_imps),
                "calls": p.stats["calls"], "invalid": p.stats["invalid"],
                "tokens_in": p.stats["tokens_in"], "tokens_out": p.stats["tokens_out"],
                "cost": round(p.stats["cost"], 6),
            })
        return {"seed": self.seed, "winner": winner, "reason": reason, "ticks": self.tick,
                "meetings": self.meetings, "kills": self.kills, "ejections": self.ejections,
                "task_bar": round(self.bar(), 3), "seconds": round(secs, 1), "players": players}

    def _write_outputs(self, summary: dict) -> None:
        if self._tx:
            self._tx.close()
        if not self.out_dir:
            return
        (self.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        replay = {
            "meta": self.meta(summary["winner"], summary["reason"]),
            "frames": self.frames,
        }
        (self.out_dir / "replay.json").write_text(json.dumps(replay))
