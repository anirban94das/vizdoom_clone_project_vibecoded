# vizdoom_clone_project_vibecoded

A neural network that plays DOOM, trained with reinforcement learning on top of [ViZDoom](https://github.com/Farama-Foundation/ViZDoom).

## Approach

The agent is a CNN policy trained with **PPO** (via `stable-baselines3`) directly on preprocessed screen pixels — there is no LLM in the control loop. That was a deliberate choice: ViZDoom makes a decision roughly every 30ms, which is far faster than any local LLM can generate, and a text-only model can't see the screen without adding even more latency. A future phase may add an LLM as a *high-level* planner (issuing directives like `EXPLORE` / `RETREAT` every second or two) sitting on top of this policy, but that only makes sense once the low-level controller already works — so this repo focuses on that first.

## Status

Training pipeline for the `basic.wad` scenario (single room, one monster — the simplest built-in ViZDoom scenario) is built, tuned for throughput, and runnable. No full training run has completed yet. Harder scenarios (`deadly_corridor.wad`, `defend_the_center.wad`) come after `basic.wad` shows a clean upward reward trend.

## Setup

Requires a project-local virtual environment (do not use a global Python install for this).

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install vizdoom gymnasium stable-baselines3 torch tensorboard
```

Installed and verified on this machine: Python 3.14.6, `vizdoom` 1.3.0, `gymnasium` 1.3.0, `torch` 2.12.1+cu130, `stable-baselines3` 2.9.0, on an AMD Ryzen 7 5800H (8 cores / 16 threads) with an NVIDIA RTX 3060 (6GB).

## Usage

Train the agent on `basic.wad`:

```powershell
.venv\Scripts\Activate.ps1
python train_basic.py
```

Watch training metrics live:

```powershell
.venv\Scripts\Activate.ps1
tensorboard --logdir logs/tensorboard
# open http://localhost:6006
```

Watch the agent actually play, live, in a second terminal (updates as new checkpoints land):

```powershell
.venv\Scripts\Activate.ps1
python watch_agent.py
```

Resume training from a checkpoint instead of starting over:

```python
model = PPO.load("models/checkpoints/ppo_basic_<N>_steps", env=vec_env)
model.learn(total_timesteps=..., reset_num_timesteps=False, tb_log_name="ppo_basic")
```

## Project structure

```
envs/basic_env.py      Gymnasium env factory for basic.wad — screen-only
                        observation, grayscale, resized to 84x84 (84,84,1),
                        frame_skip=4, native render at RES_160X120, unique
                        per-process ZDoom config path
train_basic.py          PPO training entry point (CnnPolicy, 14 parallel envs
                         via SubprocVecEnv, frame stack applied at the
                         vec-env level, checkpoints every ~10k steps)
watch_agent.py           Loads the newest checkpoint and renders live
                          gameplay in a window, independent of training
models/checkpoints/       Periodic checkpoints saved during training
models/                   Final saved model weights (model.save at the end)
logs/tensorboard/          TensorBoard logs
configs/                    Auto-generated, per-process ZDoom ini files
                             (gitignored; safe to delete when nothing's running)
```

## Performance notes

- Envs run via `SubprocVecEnv` (one ViZDoom instance per OS process) rather than the sequential `DummyVecEnv` default — ViZDoom's engine step is CPU-bound (software rendering), so this is what actually parallelizes across cores. `N_ENVS = 14` on this machine's 8-core/16-thread CPU; going much higher has diminishing returns since ViZDoom stepping is single-threaded work and SMT isn't a 2x multiplier.
- `frame_skip=4` — the policy acts once every 4 engine ticks rather than every tick, matching standard Atari/ViZDoom RL practice.
- Native render resolution is forced down to `RES_160X120` (from ViZDoom's `basic.wad` default of `RES_320X240`) since the pipeline resizes to 84x84 anyway — no reason to make the software rasterizer draw 4x more pixels per step across 14 parallel workers.
- Frame-stacking happens at the vec-env level (`VecFrameStack` wrapping the already-parallel `SubprocVecEnv`), not per-env. Each worker ships one new `(84,84,1)` frame across its process pipe per step instead of a full `(84,84,4)` stack — a 4x cut in inter-process payload.
- Training envs render headless (`render_mode=None`) for speed; use `watch_agent.py` separately to see gameplay without slowing training down.
- Checkpoints save every ~10k timesteps (`CheckpointCallback`) so a killed run doesn't lose all progress — `model.save()` alone only fires once, at the very end of `model.learn()`.

## Known gotchas (already handled, documented so they don't get "fixed" twice)

- **`viz_instance_id is write protected`** — happens when multiple `SubprocVecEnv` workers launch simultaneously and all default to the same `_vizdoom.ini` in the working directory; they race on a ZDoom cvar used for shared-memory IPC naming. Fixed by giving each worker process a unique `doom_config_path` (`configs/vizdoom_<pid>.ini`) in `envs/basic_env.py`.
- **`gymnasium.wrappers.ResizeObservation` silently drops a size-1 channel dim.** It *declares* an output shape of `(84, 84, 1)` but internally calls `cv2.resize`, which returns `(84, 84)` for single-channel input — the declared and actual shapes disagree, and this breaks `VecFrameStack` downstream. Fixed by resizing the plain 2D grayscale image (`keep_dim=False`), then explicitly restoring the channel dim with `ReshapeObservation(env, (84, 84, 1))`.

## Related workspace projects

- [`ViZDoom`](../ViZDoom) — the game environment this project depends on.
- [`Claude_Cowork/gemma-web-agent`](../Claude_Cowork/gemma-web-agent) — existing local LLM (LM Studio) integration pattern, earmarked for a future high-level planner layer on top of this policy.

See `CLAUDE.md` for the fuller architecture writeup and session-to-session context.
