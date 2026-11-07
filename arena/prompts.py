"""All text shown to the LLMs. Kept in one place so the prompts are easy to tweak."""
from __future__ import annotations

from . import world

TURN_TAG = "[[TURN: {kind}]]"   # harmless marker; also lets the mock client recognise the turn type


def map_text() -> str:
    return "\n".join(f"- {r}: {', '.join(world.ADJ[r])}" for r in world.ROOMS)


def system_prompt(name: str, role: str, allies: list[str], cfg, n_players: int, n_impostors: int) -> str:
    if role == "impostor":
        ally_txt = ", ".join(allies) if allies else "none (you are alone)"
        role_txt = (
            f"YOUR ROLE: IMPOSTOR. Fellow Impostor(s): {ally_txt}.\n"
            "Kill Crewmates when you can avoid witnesses, blend in by pretending to do tasks, "
            "sabotage the reactor to pull Crewmates away, and deflect suspicion in meetings. "
            "You may lie. Anyone in the room when you kill or vent will see you do it and can report you."
        )
    else:
        role_txt = (
            "YOUR ROLE: CREWMATE. You do not know who the Impostors are.\n"
            "Finish your tasks, pay attention to who you see where, report bodies, and use meetings to "
            "work out who is lying. Voting out a fellow Crewmate helps the Impostors."
        )
    win_timeout = "Impostors win" if cfg.timeout_winner == "impostors" else "Crewmates win"
    return f"""You are {name}, playing a text-based game of Among Us (social deduction) against other AI players.
There are {n_players} players ({n_impostors} Impostor(s), the rest Crewmates), identified only by color names.

SHIP MAP (room: adjacent rooms):
{map_text()}

RULES
- The game runs in ticks (limit {cfg.max_ticks}). Each tick every living player picks ONE action from a numbered list that I give you.
- Walking is automated. Choosing "Go to <room>" walks you along the shortest path, one room per tick. You are only asked again when you arrive or when something important happens (you see a body, a kill, a vent, a sabotage alert).
- You only see players who are in the same room as you. Anything that happens in your room, you see.
- Crewmates win by completing all tasks (the task bar shows progress) or by ejecting every Impostor.
- Impostors win when living Impostors >= living Crewmates, when a reactor meltdown finishes, or if time runs out ({win_timeout} at the tick limit).
- Bodies stay where they fall until someone reports them. Dead players are out of the game.
- Some tasks are VISUAL (they produce a visible effect). Seeing someone finish a visual task proves they are a real Crewmate. Impostors cannot do tasks; their "fake tasks" look the same as ordinary non-visual tasks from the outside.
- REPORTING a body, or pressing the emergency button (Cafeteria only, once per player, not during a sabotage), starts a meeting: everyone is moved to the Cafeteria, talks for {cfg.discussion_rounds} round(s), then votes. The player with the most votes is ejected; a tie or a Skip majority ejects nobody.
- Reactor sabotage (Impostors only): the reactor melts down in {cfg.sabotage_timer} ticks unless Crewmates make 2 repair actions in the Reactor room. Meetings cancel it.
- Security cameras (Security room) show who is in {', '.join(world.CAMERA_ROOMS)}; the Admin table shows how many players are in each room (no names).

{role_txt}

HOW TO ANSWER
Reply with exactly ONE JSON object and nothing else.
- For an action or a vote: {{"thought": "<private reasoning, max 3 sentences>", "choice": <option number>, "notes": "<optional: notes to your future self, max 300 chars; replaces your previous notes>"}}
- For speaking in a meeting: {{"thought": "<private reasoning, max 3 sentences>", "say": "<what you say out loud, max 3 sentences>", "notes": "<optional>"}}
Your "thought" and "notes" are private. Only "say" is heard by others."""


def _block(title: str, lines: list[str], empty: str = "(nothing yet)") -> str:
    body = "\n".join(lines) if lines else empty
    return f"{title}\n{body}"


def action_user(s: dict) -> str:
    parts = [
        f"=== TICK {s['tick']} of {s['max_ticks']} | Task bar: {s['bar']}% ===",
        f"ALIVE ({len(s['alive'])}): {', '.join(s['alive'])}",
        f"DEAD: {', '.join(s['dead']) if s['dead'] else 'nobody yet'}",
    ]
    if s["sab_line"]:
        parts.append(s["sab_line"])
    parts.append(f"LOCATION: {s['room']} (adjacent: {', '.join(s['adj'])})")
    parts.append(f"Players here with you: {', '.join(s['here']) if s['here'] else 'nobody'}")
    if s["bodies"]:
        parts.append(f"BODIES HERE: {', '.join(s['bodies'])}")
    parts.append(_block(s["tasks_title"], s["tasks"]))
    parts.append(f"YOUR NOTES: {s['notes'] or '(none)'}")
    parts.append(_block("KEY FACTS (oldest first):", s["key"]))
    parts.append(_block("RECENT LOG (oldest first):", s["log"]))
    if s["prev_chats"]:
        parts.append("PREVIOUS MEETINGS:\n" + "\n\n".join(s["prev_chats"]))
    opts = "\n".join(f"{i}) {o}" for i, o in enumerate(s["options"], 1))
    parts.append(f"OPTIONS:\n{opts}")
    parts.append("Pick one option and reply with the JSON object. " + TURN_TAG.format(kind="action"))
    return "\n\n".join(parts)


def _meeting_common(s: dict) -> list[str]:
    parts = [
        f"=== MEETING #{s['meeting_no']} === {s['header']}",
        f"ALIVE ({len(s['alive'])}): {', '.join(s['alive'])}",
        f"DEAD: {', '.join(s['dead']) if s['dead'] else 'nobody'}",
        _block(s["tasks_title"], s["tasks"]),
        f"YOUR NOTES: {s['notes'] or '(none)'}",
        _block("KEY FACTS (oldest first):", s["key"]),
        _block("RECENT LOG (oldest first):", s["log"]),
    ]
    if s["prev_chats"]:
        parts.append("EARLIER MEETINGS:\n" + "\n\n".join(s["prev_chats"]))
    chat = [f'{n}: "{t}"' for n, t in s["chat"]]
    parts.append(_block("DISCUSSION SO FAR IN THIS MEETING:", chat, empty="(no one has spoken yet)"))
    return parts


def speak_user(s: dict) -> str:
    parts = _meeting_common(s)
    parts.append(f"It is round {s['round']} of {s['rounds']} and your turn to speak. "
                 'Reply with the JSON object containing "say". ' + TURN_TAG.format(kind="speak"))
    return "\n\n".join(parts)


def vote_user(s: dict) -> str:
    parts = _meeting_common(s)
    opts = "\n".join(f"{i}) {o}" for i, o in enumerate(s["options"], 1))
    parts.append(f"Discussion is over. Time to VOTE.\nOPTIONS:\n{opts}")
    parts.append("Pick one option and reply with the JSON object. " + TURN_TAG.format(kind="vote"))
    return "\n\n".join(parts)


def retry_reminder(kind: str, problem: str) -> str:
    what = '"say"' if kind == "speak" else '"choice" (an option number)'
    return (f"Your last reply could not be used ({problem}). Reply again with ONLY one JSON object "
            f"containing {what}, plus optional \"thought\" and \"notes\". " + TURN_TAG.format(kind=kind))
