# LLM Among Us Arena

Built with Claude Sonnet 5.5. 

A text-based Among Us where different [OpenRouter](https://openrouter.ai) models play each other. The harness handles movement, visibility, cooldowns and vote counting. Each model only has to pick from a numbered list of legal actions and, in meetings, say a few sentences.

Pure Python 3.11+, standard library only, no dependencies. Run everything with [uv](https://docs.astral.sh/uv/) (`uv run ...` sets up the environment on first use).

```
uv run python -m arena serve                           # set up games in your browser: pick every seat's role and model
uv run python -m arena play --mock --live               # watch a fake game live in your browser, free, no API key
uv run python -m arena play --live --models openai/gpt-4o-mini,google/gemini-2.0-flash-001   # real models, live
uv run python -m arena play --mock                      # offline run, no browser
uv run python -m arena tournament --games 20 --parallel 4
uv run python -m arena models --filter claude           # look up valid OpenRouter ids
uv run python -m arena replay runs/<run>/game_000       # rebuild the visual replay
uv run smoke_test.py                           # offline tests of the engine and HTTP client
```

Put your key in `OPENROUTER_API_KEY` (or copy `.env.example` to `.env`). Edit the roster in `config.toml`.

## Setting up a game in the browser

`uv run python -m arena serve` opens the arena with a **New game** screen instead of starting a game right away:

- **Players** (4-12) and **Impostors**: the number of seats and Impostors.
- Each seat (Red, Blue, ...) has a **role** (Random / Crew / Impostor) and a **model**. The model list comes from `[[models]]` in `config.toml`, plus **Mock bot** (free, offline), **Random bot** (no LLM) and **Custom model id...**, which autocompletes from OpenRouter's full model list.
- Pin the roles you care about and leave the rest on Random: the remaining Impostor slots are drawn from the Random seats. If no seat is Random, the Impostor count follows your picks.
- **Set every seat to** fills all seats with one model. Mix real models, Mock bots and Random bots freely at the same table.

**Start game** checks every real model with a tiny request first (so a typo shows up on this screen, not mid-game), then switches to the live view. **Stop game** aborts a running game; **New game** comes back to this screen with your seats kept. Each game is saved under `runs/` like `play` does. Seats using real models need `OPENROUTER_API_KEY`; Mock and Random seats work without one.

The server only accepts requests from its own page (`Origin`/`Host` are checked), so other websites you have open can't start games on your key.

## Watching live (god mode)

`--live` starts a small local web server (default http://127.0.0.1:8765, `--port` to change it) and opens your browser. The page plays along as the game runs:

- the ship map with crewmates sliding between rooms, bodies where they fall, and the reactor flashing red during a sabotage
- a pulsing `...` over every player whose model is still deciding, plus "Waiting for Red, Blue ..." in the side panel
- toasts for kills, vents, sabotage and reports, and speech bubbles over the Cafeteria while models argue in a meeting
- the side panel shows each action with the model's private reasoning, then the meeting chat, votes and ejection
- the **God view** checkbox reveals roles (red rings), vent links and thoughts; untick it to watch like a crewmate would
- pause, step, scrub the timeline (kills and meetings are marked), or press Latest to jump back to the live edge

Without a key, `--mock --live` simulates LLM latency (`--mock-latency` seconds, default 0.5) so you can see it working. With `tournament --live`, games play one at a time and the page resets for each. After the game ends the server stays up so you can scrub back; press Ctrl-C to quit. If your browser doesn't open, use `--no-browser` and paste the URL yourself.

Every game also saves a standalone `replay.html` you can open later or share.

## What the harness does for the models

Every tick, each living player gets a prompt with their location, who is in the room, their tasks, a rolling log of what they saw, key facts, their own notes, and a **numbered action menu**:

```
OPTIONS:
1) Do task: Fix Wiring (step 1 of 2)
2) Wait here for one tick
3) Go to Storage (1 step, walked automatically)
4) Go to Reactor (2 steps, walked automatically)
...
```

"Go to X" walks the shortest path automatically, one room per tick. The model is only asked again on arrival or when something notable happens (it sees a body, a kill, a vent, or a sabotage alert), so a cross-ship trip costs one LLM call instead of five. That is the "skip the movement" part.

Models reply with one JSON object: `{"thought": "...", "choice": 3, "notes": "..."}`. `thought` is private (it shows up in the God view of the replay), `notes` is a scratchpad that carries over between calls. In meetings they reply with `{"say": "..."}` and later vote from a menu. Bad output gets one retry, then a safe fallback (wait / skip), and it is counted in the leaderboard as an invalid reply.

Players are just colors (Red, Blue, ...), so models cannot see who they are playing against. Seats and roles are shuffled per game.

## Rules implemented

- Crew win by finishing all tasks or ejecting every Impostor. Impostors win at parity (alive Impostors >= alive Crew), on a reactor meltdown, or when the tick limit is hit (configurable).
- Impostors: kill (same room, with cooldown), vent between linked rooms, sabotage the reactor, fake non-visual tasks. They know each other.
- Everyone in a room sees everything there, including kills and vents. Bodies stay until reported.
- **Visual tasks** (Clear Asteroids, Prime Shields, Empty Garbage, Submit Scan) prove the player is a Crewmate to anyone who watches them finish one.
- Security cameras (4 rooms, names) and the Admin table (counts per room, no names).
- Meetings: report a body or press the button in the Cafeteria (once per player, not during a sabotage). N rounds of discussion, then a vote. Plurality ejects; ties and skip-majority eject nobody.

Simplifications: dead players are out (no ghost tasks; their unfinished tasks leave the task bar), no lights or door sabotage, one sabotage type, one map. The map, tasks and vents live in `arena/world.py`.

## Output

Each game gets a folder under `runs/`:

| File | What |
|---|---|
| `replay.html` | Visual replay: ship map, moving crewmates, task bar, meeting chat and votes, timeline with kills and meetings marked. God view shows roles, vents and each model's private thoughts. |
| `transcript.txt` | God-view text log with every action, thought, kill, statement and vote. Follow live with `--echo`. |
| `summary.json` | Winner, roles, per-player tokens, cost, invalid replies, votes. |
| `replay.json` | Raw frames the viewer reads. |

A tournament also writes `leaderboard.md` and `results.json`. The leaderboard has crew win rate, impostor win rate, how often each model's crew votes hit a real Impostor, impostor survival, kills, invalid-reply rate and cost per game.

Wins are team outcomes, so a weak Crewmate shares in a Crew win. You need dozens of games per model before the numbers mean much.

## Cost

A 7-player game is roughly 250-400 LLM calls at about 2-3k input tokens each, so on the order of 0.5-1M input tokens and 50-100k output tokens. Start with one `play` run on cheap models and read the per-game cost that OpenRouter reports in `summary.json`. Reasoning models cost more and need `max_tokens` raised via per-model `params` in `config.toml`.

`--parallel` runs several games at once and `concurrency` sets parallel calls inside a game. Lower both if you hit rate limits (the client retries 429s and 5xx with backoff, and stops the whole run on a bad key or no credits).

## Tuning

- `max_ticks`, `tasks_per_crew`, `kill_cooldown`, `impostors` in `config.toml` shift the balance. If Crew win too easily, add Impostors or cut `max_ticks`.
- `anonymous_votes = true` hides who voted for whom. `confirm_ejects = false` hides the ejected role.
- All prompt text is in `arena/prompts.py`.
- Preflight sends one tiny request per model before a run, so a mistyped model id fails in seconds. Skip it with `--no-preflight`.
- `--random` runs uniformly random agents as a baseline.
