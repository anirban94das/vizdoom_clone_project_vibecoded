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
    
## Two scenarios currently in play

- **`basic.wad`** — one room, one monster. The simplest built-in scenario, used to prove the pipeline works end-to-end. It already performs well.
- **`deadly_corridor.wad`** — harder: a corridor full of enemies, `doom_skill=5`, and a death penalty. Its built-in reward is sparse (mostly just "did you reach the end / did you die"), so this project adds **reward shaping**: extra bonus reward for landing hits, getting kills, exploring new areas, and picking up weapons. That's implemented as opt-in wrapper classes in `envs/common.py`, off by default, turned on for this scenario specifically.

Both scenarios **auto-resume**: each training script checks for a single model file under `models/latest/` on startup and continues from it if present, otherwise starts fresh. Training periodically overwrites that same file (~every 10k steps) rather than keeping numbered checkpoints — so there's always exactly one "current" model per scenario.

## How you'd actually use it

Easiest path — a small desktop UI:
```powershell
.venv\Scripts\python.exe train_ui.py
```
Pick a scenario, tweak reward-bonus values if you want, hit Start Training / Watch Agent. There's also a **Visualize Model** button — it draws a picture of the neural network itself (what layers it has, how big) using whichever scenario's saved model you currently have, and shows it right there in the window next to the log.

Or from the command line:
```powershell
.venv\Scripts\Activate.ps1
python train_basic.py               # or train_deadly_corridor.py — not both at once
python watch_agent.py               # separate terminal, see it actually play, live
tensorboard --logdir logs/tensorboard   # reward/loss curves over time
```

## Where things live

- `envs/` — the Gymnasium environment setup (screen preprocessing, reward shaping wrappers)
- `train_basic.py` / `train_deadly_corridor.py` — the actual PPO training entry points
- `watch_agent*.py` — loads the current model and shows it playing live
- `train_ui.py` — the GUI wrapper around all of the above
- `visualize_PPO_model.py` — draws a picture of the network's architecture (layers/shapes), not part of training itself
- `models/latest/` — the one live model file per scenario
- `CLAUDE.md` — much deeper technical writeup (gotchas already solved, performance tuning already applied, exact file responsibilities) if you want to go further

## What's next on the roadmap

Once `deadly_corridor`'s shaped-reward run shows a clean upward trend, the plan is to move on to `defend_the_center.wad` (a "stand your ground" scenario) using the same pipeline.
