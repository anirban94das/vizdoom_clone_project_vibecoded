# vizdoom_clone_project_vibecoded

A neural network that plays DOOM, trained with reinforcement learning on top of [ViZDoom](https://github.com/Farama-Foundation/ViZDoom).

## Approach

The agent is a CNN policy trained with **PPO** (via `stable-baselines3`) directly on preprocessed screen pixels — there is no LLM in the control loop. That was a deliberate choice: ViZDoom makes a decision roughly every 30ms, which is far faster than any local LLM can generate, and a text-only model can't see the screen without adding even more latency. A future phase may add an LLM as a *high-level* planner (issuing directives like `EXPLORE` / `RETREAT` every second or two) sitting on top of this policy, but that only makes sense once the low-level controller already works — so this repo focuses on that first.

## Status

- **`basic.wad`** (single room, one monster — the simplest built-in ViZDoom scenario): training pipeline built, tuned for throughput, and performing well.
- **`deadly_corridor.wad`** (harder, sparse-reward, `doom_skill=5`, must advance under fire): baseline run completed (`ep_rew_mean` climbed from -88 to +151 over 950k steps). A reward-shaped version is now the default — see below — warm-started from that baseline.
- Next up: `defend_the_center.wad`, once the shaped `deadly_corridor` run shows a clean upward trend.

### Reward shaping

`deadly_corridor.cfg` only scores distance-to-goal and a death penalty out of the box — it doesn't reward kills, hits, exploration, or item pickups directly. Four opt-in wrappers (`envs/common.py`) add denser signal on top of any scenario's built-in reward:

| Bonus | What it rewards | Default (`basic`) | Default (`deadly_corridor`) |
|---|---|---|---|
| `kill_reward_bonus` | Each `KILLCOUNT` increment | 0.0 (off) | 20.0 |
| `hit_reward_bonus` | Each `HITCOUNT` increment (landing a shot, not just a kill) | 0.0 (off) | 5.0 |
| `exploration_bonus_per_cell` | First visit to each discretized position cell per episode | 0.0 (off) | 1.0 |
| `weapon_pickup_bonus` | First time each episode a `WEAPON0`–`WEAPON9` slot is acquired (e.g. the shotgun a dead `ShotgunGuy` drops) | 0.0 (off) | 15.0 |

`basic.wad`'s built-in reward is already sufficient, so all four default off there. Both scripts accept `--kill-reward-bonus`, `--hit-reward-bonus`, `--exploration-bonus-per-cell`, `--exploration-cell-size`, and `--weapon-pickup-bonus` flags to override any of these per run.

### Policy architecture

Rendered via `visualize_PPO_model.py` from each scenario's actual saved `CnnPolicy` (not a diagram of the code) — NatureCNN feature extractor feeding into PPO's action head. `basic.wad`'s head outputs 4 actions, `deadly_corridor`'s outputs 8; both share the same CNN trunk.

**`basic.wad`:**

![PPO actor architecture — basic.wad](ppo_actor_render_basic.png)

**`deadly_corridor.wad` (shaped reward):**

![PPO actor architecture — deadly_corridor.wad](ppo_actor_render_deadly_corridor.png)

## Setup

Requires a project-local virtual environment (do not use a global Python install for this).

```powershell
# Windows cmd
setup_env.bat
```
```bash
# Git Bash / WSL / any POSIX shell
./setup_env.sh
```

Both scripts resolve a Python 3.14 interpreter via the `py` launcher, create `.venv`, and `pip install -r requirements.txt` (which pins `torch==2.12.1+cu130` via `--extra-index-url`, plus `vizdoom`, `gymnasium`, `stable-baselines3`, `tensorboard`, `matplotlib`, and `visualtorch`) — re-runs skip the reinstall if `requirements.txt` hasn't changed since the last successful install. If both scripts break, fall back to manual setup:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Installed and verified on this machine: Python 3.14.6, `vizdoom` 1.3.0, `gymnasium` 1.3.0, `torch` 2.12.1+cu130, `stable-baselines3` 2.9.0, on an AMD Ryzen 7 5800H (8 cores / 16 threads) with an NVIDIA RTX 3060 (6GB).

## Usage

### Desktop launcher (recommended)

A small Tkinter UI wraps everything below — pick a scenario, tweak reward-shaping values, and start/stop training or live-watching without touching the terminal:

```powershell
.venv\Scripts\python.exe train_ui.py
```

A third button, **Visualize Model**, renders the selected scenario's saved `CnnPolicy` architecture as a PNG (via `visualize_PPO_model.py`) and shows it inline in a panel next to the log — useful for sanity-checking the network shape, or just seeing what's actually training. Disabled with an error dialog if that scenario doesn't have a saved model yet.

### Training recap

Every run of `train_basic.py` / `train_deadly_corridor.py` (CLI or via `train_ui.py`) ends with a printed recap comparing the first ~20 episodes of that run against the last ~20 — reward, kills, hits, cells explored, and weapons picked up — e.g.:

```
[recap] ppo_basic - 94 episodes this run (first 20 vs last 20, 4096 cumulative timesteps):
  reward            :  -151.85 ->  -156.55
  kills             :     0.55 ->     0.60
  hits              :     0.55 ->     0.60
  cells_explored    :     5.50 ->     5.15
  weapons_picked_up :     0.00 ->     0.00
```

The same line is appended as JSON to `logs/training_history.jsonl`, so past runs' recaps accumulate over time instead of only being visible in that run's console output. Kills/hits/exploration/weapons are tracked for every scenario regardless of whether that scenario's reward-shaping bonuses are turned on (`envs/common.py`'s `EpisodeStatsWrapper`), so `basic.wad` gets a behavior recap too, not just `deadly_corridor`.

### Command line

Train the agent (auto-resumes from `models/latest/` if a model already exists — see below):

```powershell
.venv\Scripts\Activate.ps1
python train_basic.py
python train_deadly_corridor.py   # don't run alongside train_basic.py — see Performance notes
```

Watch training metrics live:

```powershell
.venv\Scripts\Activate.ps1
tensorboard --logdir logs/tensorboard
# open http://localhost:6006
```

Watch the agent actually play, live, in a second terminal (reloads the latest saved model between episodes, so behavior updates as training progresses):

```powershell
.venv\Scripts\Activate.ps1
python watch_agent.py                    # basic.wad
python watch_agent_deadly_corridor.py    # deadly_corridor.wad
```

### Auto-resume

Each `train_*.py` checks for a single fixed model file under `models/latest/` on startup (`ppo_basic.zip` / `ppo_deadly_corridor_shaped.zip`) and resumes from it with `PPO.load` if present, otherwise starts a fresh `CnnPolicy`. There are no step-numbered checkpoints to manage — `training_utils.OverwriteCheckpointCallback` saves to that same fixed path roughly every 10k timesteps, overwriting it in place, so exactly one file per scenario exists at any time and it's always current. To force a from-scratch run, delete that scenario's file under `models/latest/` first.

`train_deadly_corridor.py` trains under a `ppo_deadly_corridor_shaped` identity (not the older `ppo_deadly_corridor` baseline) since it enables reward shaping by default — a different reward function than the baseline run it evolved from. If no shaped model exists yet, it warm-starts weights from `models/ppo_deadly_corridor.zip` (the baseline's final save) instead, carrying over learned visual features/aiming/movement, but resets the timestep/TensorBoard counter — expect a visible jump/dip in the reward curve right at that handoff.

## Project structure

```
envs/common.py                    Shared Gymnasium env factory — screen-only
                                   observation, grayscale, resized to 84x84
                                   (84,84,1), frame_skip=4, native render at
                                   RES_160X120, unique per-process ZDoom
                                   config path, four opt-in reward-shaping
                                   wrappers (kill/hit/exploration/weapon)
envs/basic_env.py                 Thin wrapper registering VizdoomBasic-v1
envs/deadly_corridor_env.py       Thin wrapper registering
                                   VizdoomDeadlyCorridor-v1, reward shaping
                                   on by default
training_utils.py                 Shared OverwriteCheckpointCallback — saves
                                   to one fixed path on a schedule instead of
                                   a new file per checkpoint. Also
                                   EpisodeRecapCallback — prints a start-vs-
                                   end-of-run behavior comparison and appends
                                   it to logs/training_history.jsonl
train_basic.py                    PPO training entry point for basic.wad
                                   (CnnPolicy, 12 parallel envs via
                                   SubprocVecEnv, frame stack at the vec-env
                                   level, 100k timesteps/run)
train_deadly_corridor.py          Same structure, for deadly_corridor.wad
                                   (300k timesteps/run, reward shaping on)
train_ui.py                       Tkinter launcher — start/stop training and
                                   live-watching for either scenario, edit
                                   reward-shaping values, render the model
                                   architecture inline, without the CLI
visualize_PPO_model.py            One-shot script, renders a saved (or
                                   untrained) CnnPolicy's architecture as a
                                   PNG via visualtorch — not part of training
setup_env.sh / setup_env.bat       Bootstrap .venv + pip install -r
                                   requirements.txt from scratch on a new
                                   machine
watch_agent.py                    Reloads models/latest/ppo_basic.zip before
                                   each episode, renders live gameplay in a
                                   window, independent of training
watch_agent_deadly_corridor.py    Same, for
                                   models/latest/ppo_deadly_corridor_shaped.zip
models/latest/                    The single actively-trained model per
                                   scenario, overwritten in place — what
                                   auto-resume and watch_agent_*.py read
models/checkpoints/               Leftover step-numbered checkpoints from
                                   before the single-file scheme; no longer
                                   written to
models/                           (top level) older one-off final saves;
                                   ppo_deadly_corridor.zip is still read as
                                   the shaped run's warm-start source
logs/tensorboard/                 TensorBoard logs
logs/training_history.jsonl       One JSON line per completed training run
                                   (reward/kills/hits/cells-explored/weapons,
                                   first-~20 vs last-~20 episodes), appended
                                   by EpisodeRecapCallback
configs/                          Auto-generated, per-process ZDoom ini files
                                   (gitignored; safe to delete when nothing's
                                   running)
```

## Performance notes

- Envs run via `SubprocVecEnv` (one ViZDoom instance per OS process) rather than the sequential `DummyVecEnv` default — ViZDoom's engine step is CPU-bound (software rendering), so this is what actually parallelizes across cores. `N_ENVS = 12` on this machine's 8-core/16-thread CPU — throughput is capped more by physical cores than logical ones, and going much higher had diminishing returns (and hit a startup race at 14).
- **Don't run `train_basic.py` and `train_deadly_corridor.py` at the same time** — each spawns its own `N_ENVS` `SubprocVecEnv` workers, and running both oversubscribes this machine's 8 physical cores.
- `frame_skip=4` — the policy acts once every 4 engine ticks rather than every tick, matching standard Atari/ViZDoom RL practice.
- Native render resolution is forced down to `RES_160X120` (from ViZDoom's default `RES_320X240`) since the pipeline resizes to 84x84 anyway — no reason to make the software rasterizer draw 4x more pixels per step across every parallel worker.
- Frame-stacking happens at the vec-env level (`VecFrameStack` wrapping the already-parallel `SubprocVecEnv`), not per-env. Each worker ships one new `(84,84,1)` frame across its process pipe per step instead of a full `(84,84,4)` stack — a 4x cut in inter-process payload.
- Training envs render headless (`render_mode=None`) for speed; use `watch_agent*.py` (or the "Watch Agent" button in `train_ui.py`) separately to see gameplay without slowing training down — it's a single-process `DummyVecEnv`, so it can run alongside a training run without oversubscription concerns.

## Known gotchas (already handled, documented so they don't get "fixed" twice)

- **`viz_instance_id is write protected`** — happens when multiple `SubprocVecEnv` workers launch simultaneously and all default to the same `_vizdoom.ini` in the working directory; they race on a ZDoom cvar used for shared-memory IPC naming. Fixed by giving each worker process a unique `doom_config_path` (`configs/vizdoom_<pid>.ini`) in `envs/common.py`.
- **`gymnasium.wrappers.ResizeObservation` silently drops a size-1 channel dim.** It *declares* an output shape of `(84, 84, 1)` but internally calls `cv2.resize`, which returns `(84, 84)` for single-channel input — the declared and actual shapes disagree, and this breaks `VecFrameStack` downstream. Fixed by resizing the plain 2D grayscale image (`keep_dim=False`), then explicitly restoring the channel dim with `ReshapeObservation(env, (84, 84, 1))`.

## Related workspace projects

- [`ViZDoom`](../ViZDoom) — the game environment this project depends on.
- [`Claude_Cowork/gemma-web-agent`](../Claude_Cowork/gemma-web-agent) — existing local LLM (LM Studio) integration pattern, earmarked for a future high-level planner layer on top of this policy.

See `CLAUDE.md` for the fuller architecture writeup and session-to-session context.
