"""Ship map, task catalogue and path-finding helpers (pure data, no game state)."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

# --- Map ---------------------------------------------------------------------
# A simplified Skeld-like ship. Edit freely: everything else derives from these lists.

_EDGES = [
    ("Cafeteria", "Weapons"), ("Cafeteria", "Admin"), ("Cafeteria", "MedBay"),
    ("Cafeteria", "Upper Engine"), ("Cafeteria", "Storage"),
    ("Weapons", "O2"), ("Weapons", "Navigation"),
    ("O2", "Navigation"),
    ("Navigation", "Shields"),
    ("Shields", "Communications"), ("Shields", "Storage"),
    ("Communications", "Storage"),
    ("Storage", "Admin"), ("Storage", "Electrical"), ("Storage", "Lower Engine"),
    ("Electrical", "Lower Engine"),
    ("Lower Engine", "Security"), ("Lower Engine", "Reactor"),
    ("Security", "Reactor"), ("Security", "Upper Engine"),
    ("Reactor", "Upper Engine"),
    ("Upper Engine", "MedBay"),
]

# Vent groups: an Impostor in a room that appears in a group can hop to the others.
_VENT_GROUPS = [
    ["Weapons", "Navigation", "Shields"],
    ["Electrical", "MedBay"],
    ["Admin", "Communications"],
]

# Rooms covered by the security cameras (watched from the Security room).
CAMERA_ROOMS = ["Cafeteria", "Storage", "Shields", "Upper Engine"]

HUB = "Cafeteria"          # meeting room + emergency button
REACTOR = "Reactor"        # where the reactor sabotage is repaired
SECURITY = "Security"
ADMIN = "Admin"

ADJ: dict[str, list[str]] = {}
for a, b in _EDGES:
    ADJ.setdefault(a, []).append(b)
    ADJ.setdefault(b, []).append(a)
for _r in ADJ:
    ADJ[_r].sort()
ROOMS: list[str] = sorted(ADJ)

VENTS: dict[str, list[str]] = {}
for _g in _VENT_GROUPS:
    for _r in _g:
        VENTS.setdefault(_r, [])
        VENTS[_r].extend(x for x in _g if x != _r and x not in VENTS[_r])


def _bfs_paths(src: str) -> dict[str, list[str]]:
    """Shortest path (excluding src) from src to every room."""
    prev: dict[str, str | None] = {src: None}
    q = deque([src])
    while q:
        cur = q.popleft()
        for nxt in ADJ[cur]:
            if nxt not in prev:
                prev[nxt] = cur
                q.append(nxt)
    out: dict[str, list[str]] = {}
    for dst in prev:
        path, node = [], dst
        while node != src:
            path.append(node)
            node = prev[node]  # type: ignore[assignment]
        out[dst] = list(reversed(path))
    return out


PATHS = {r: _bfs_paths(r) for r in ROOMS}


def path(src: str, dst: str) -> list[str]:
    """Rooms to walk through to get from src to dst (excludes src, includes dst)."""
    return PATHS[src][dst]


def distance(src: str, dst: str) -> int:
    return len(PATHS[src][dst])


# --- Tasks -------------------------------------------------------------------

@dataclass(frozen=True)
class TaskDef:
    name: str
    room: str
    visual: bool = False   # completing it produces a visible effect that proves innocence
    steps: int = 1         # number of separate "work" actions needed


TASKS = [
    TaskDef("Clear Asteroids", "Weapons", visual=True),
    TaskDef("Prime Shields", "Shields", visual=True),
    TaskDef("Empty Garbage", "Storage", visual=True),
    TaskDef("Submit Scan", "MedBay", visual=True),
    TaskDef("Fix Wiring", "Electrical", steps=2),
    TaskDef("Divert Power", "Electrical"),
    TaskDef("Fuel Engines", "Storage", steps=2),
    TaskDef("Swipe Card", "Admin"),
    TaskDef("Upload Data", "Communications", steps=2),
    TaskDef("Chart Course", "Navigation"),
    TaskDef("Stabilize Steering", "Navigation"),
    TaskDef("Clean O2 Filter", "O2"),
    TaskDef("Align Engine Output", "Upper Engine"),
    TaskDef("Start Reactor", "Reactor", steps=2),
    TaskDef("Unlock Manifolds", "Reactor"),
    TaskDef("Download Data", "Cafeteria", steps=2),
    TaskDef("Calibrate Distributor", "Electrical"),
    TaskDef("Inspect Sample", "MedBay"),
    TaskDef("Accept Diverted Power", "Lower Engine"),
    TaskDef("Monitor Tree", "Security"),
]
