# vizdoom_clone_project_vibecoded — explained for newcomers

## What this project is

A neural network that learns to play DOOM. Specifically: a CNN (convolutional neural network) trained with **PPO** (Proximal Policy Optimization, a reinforcement-learning algorithm) to look at raw game-screen pixels and decide what action to take — move, turn, shoot — the same way a human would, just from what's on screen. It's built on top of [ViZDoom](https://github.com/Farama-Foundation/ViZDoom), a research platform that wraps the original Doom engine and exposes it as a Gymnasium (the standard Python RL-environment API) environment.

## Why RL and not an LLM

This came up early in the project and was deliberately ruled out: ViZDoom needs a decision roughly every 30ms. No local LLM can generate a response that fast, and a text-only model can't even see the screen without adding a vision step on top — more latency on an already-too-slow path. So the "brain" here is a small, fast, purpose-built CNN policy instead. (There's a possible future phase — see below — where an LLM sits *above* this as a slow strategic layer, but that's explicitly deferred.)

## Food for thought
    Can I build a custom LLM for this? Can I build a bunch of LLMs, models/layers/nueral pathways talking to each other, planning actions? I say LLM as LLMS 4 me is the easiest measure of intelligence. 

## The core training loop, conceptually

1. ViZDoom renders a frame → it's converted to grayscale, resized to 84×84 pixels, and stacked with the previous 3 frames (so the network can perceive motion, not just a static image).
2. The CNN looks at that stack and outputs an action.
3. The game advances 4 engine ticks per action (`frame_skip=4` — the network doesn't need to react every single tick).
4. The scenario's built-in reward (plus optional bonus shaping — see below) tells the network whether that was good or bad.
5. PPO uses thousands of these steps, run across parallel game instances, to gradually improve the policy.

## Food for thought
    I want to build a prettier UI so that this can look better. Think a window application, that can visualize the run, or even give the network a test level to run/play in. 
    When I want the take the latest model to play a level. I want it to view the complete game(full res), something I can stream ideally. 
    
## The levels

Every single-player ViZDoom training scenario is implemented, each with its own train/watch script and its own model:

- **Trained and working:** `basic` (one room, one monster — proved the pipeline) and `deadly_corridor` (corridor full of enemies at max difficulty; its sparse built-in reward is augmented with **reward shaping** — bonus reward for hits, kills, exploring new ground, picking up weapons).
- **Implemented, still to be trained:** `defend_the_center` / `defend_the_line` (stand your ground, turn and shoot), `health_gathering` (+ a harder `supreme` maze variant — grab medkits, survive an acid floor), `my_way_home` (find your way out of a maze), `predict_position` (lead a moving target with one rocket), `take_cover` (pure dodging), and the basic variants `simpler_basic`, `rocket_basic`, and `basic_audio` (the network gets *sound* as well as pixels).
- **Actual DOOM levels:** `train_doom_level.py --map E1M1` (or any map up to `E4M9` / `MAP32`) trains on real game maps. Out of the box it uses Freedoom (free, bundled with vizdoom); if you own DOOM/DOOM II, drop `doom.wad`/`doom2.wad` into `wads/` and they're used automatically.

Reward shaping is per-level: each scenario's env module turns on only the bonuses that match its objective (e.g. exploration bonus for the maze levels, health-change bonus for the medkit levels, nothing at all for `basic`). Every knob is editable in the UI or via CLI flags.

Every scenario **auto-resumes**: each training script checks for a single model file under `models/latest/` on startup and continues from it if present, otherwise starts fresh. Training periodically overwrites that same file (~every 10k steps) rather than keeping numbered checkpoints — so there's always exactly one "current" model per scenario.

## What the network actually looks like

A stack of convolution layers that turns the 4 most recent stacked game frames into a decision. Click **Visualize Model** in the desktop UI (below) to render the real, currently-trained network for the selected scenario — no static picture is checked into the repo, it's regenerated from the actual saved model each time.

## How you'd actually use it

Easiest path — a small desktop UI:
```powershell
.venv\Scripts\python.exe train_ui.py
```
Pick any of the 14 levels, tweak reward-bonus values if you want, hit Start Training / Watch Agent. Three more buttons:

- **Visualize Model** — draws a picture of the neural network itself (what layers it has, how big) from that level's saved model, right there next to the log.
- **Export Model** — saves the current model to a single file you can back up or share.
- **Import Model** — loads such a file back in as that level's active model (the old one is backed up automatically, and it warns you if the file came from a different level).

Or from the command line:
```powershell
.venv\Scripts\Activate.ps1
python scenarios/train_basic.py            # one scenarios/train_*.py per level — never two at once
python scenarios/train_doom_level.py --map E1M1   # a real DOOM level
python scenarios/watch_agent.py            # separate terminal, see it actually play, live
tensorboard --logdir logs/tensorboard     # reward/loss curves over time
python export_model.py basic              # / import_model.py <file> --scenario basic
```

## Where things live

- `envs/` — the Gymnasium environment setup: one `*_env.py` per level holding its reward-shaping defaults, plus shared preprocessing/wrappers in `common.py`
- `scenarios/` — one short PPO training entry point (`train_*.py`) and one live viewer (`watch_agent_*.py`) per level; run them from the repo root. The shared machinery lives in `train_common.py`
- `scenarios/_bootstrap.py` — tiny helper each scenario script imports first so it can still find `train_common` / `envs` from the subfolder
- `train_ui.py` — the GUI wrapper around all of the above
- `model_io.py` + `export_model.py` / `import_model.py` — save a trained model to one file / load one back in (what the UI's Export/Import buttons run)
- `train_ui.py`'s "Visualize Model" button draws a picture of the network's architecture (layers/shapes) straight from the trained model, not part of training itself
- `models/latest/` — the one live model file per level; `models/backups/` — what Import replaced; `exports/` — default Export destination
- `wads/` — where to put `doom.wad` / `doom2.wad` if you own them (Freedoom is the built-in fallback)
- `CLAUDE.md` — much deeper technical writeup (gotchas already solved, performance tuning already applied, exact file responsibilities) if you want to go further

## What's next on the roadmap

Everything is implemented; most of it hasn't *trained* yet. The plan: run each new scenario end-to-end (roughly easiest-first: `defend_the_line` → `health_gathering` → its `supreme` variant → `my_way_home` → `take_cover` → `predict_position` → the basic variants) and confirm the reward curve trends upward, tuning each level's reward-shaping defaults as results come in. Then the big one: full DOOM levels (`E1M1` onward), which is also where the deferred LLM-as-strategic-planner idea would slot in if revisited.

## Future enhancements (ideas on the shelf)

Things that could make the agent learn better or make experimenting easier, in plain terms. None of these are built yet.

1. **A fair scorecard.** Right now the reward curve mixes the game's own score with the bonus points we add for hits, kills, exploring, etc. Change a bonus and the curve moves even if the agent plays exactly the same. The fix is to periodically play a few test episodes with all bonuses switched off and record just the game's own score — like a standardised exam next to the homework marks.
2. **A "try the knobs" script.** Instead of guessing bonus values one run at a time, run the same level several times with different bonus settings and print a table comparing them on the fair scorecard above.
3. **Better default settings for PPO.** The training library's defaults were designed for robot-simulation tasks, not pixel games. The well-known "Atari recipe" (collect fewer frames per update, learn in bigger batches, slowly lower the learning rate) is probably the biggest free improvement available.
4. **A memory for the maze levels.** The network sees only the last 4 frames — about a tenth of a second. In `my_way_home` or `health_gathering_supreme` it needs to remember "I already checked that corridor." Adding a small memory unit (an LSTM) to the network fixes that.
5. **Start easy, get harder (curriculum).** A brand-new agent will never reach the exit of a full DOOM level, so it never learns that reaching the exit is the goal. Start at the lowest skill with short episodes (maybe near the exit), and raise the difficulty as it starts succeeding.
6. **Show the network the HUD numbers directly.** Health, ammo and kill count are already read off the game for reward shaping; feeding them into the network as numbers means it doesn't have to learn to read the on-screen digits from pixels.
7. **Groundwork for the LLM planner idea.** Before putting an LLM anywhere near the controls, just log a short text description of the game state (health, ammo, enemies in view, how much has been explored) plus a screenshot every few seconds while training runs. That gives real data to test planner prompts against offline, with no risk to training.
8. **Let a human show it the ropes first.** Record 10–15 minutes of a person playing a level (ViZDoom has a spectator mode for this), teach the network to imitate those moves, *then* hand over to PPO. Skips the long phase where the agent wanders into walls.
9. **Housekeeping for experiments.** Short video clips of test episodes, a `--seed` option so runs are repeatable, and a simple results table built from `logs/training_history.jsonl`.

Items 6, 7 and 8 are the ones flagged as most interesting to dig into next. Items 1 and 3 together are the sensible first step.
