# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A neural network that plays DOOM, using `E:\Code_Base\ViZDoom` as the game environment. The agent is a CNN policy trained with PPO (`stable-baselines3`) on ViZDoom's Gymnasium-compatible API — not an LLM. An LLM controller was ruled out early: ViZDoom needs a decision every ~30ms, which is far faster than any local LLM can generate, and a text-only model can't see the screen without added latency. See "Architecture decision" below for the full reasoning and the deferred Option B.

## Commands

```powershell
# Activate the project venv (Python 3.14.6, not the global install)
.venv\Scripts\Activate.ps1

# Train PPO on the basic.wad scenario
python train_basic.py

# Watch training progress (reward/loss curves)
tensorboard --logdir logs/tensorboard

# Watch the agent actually play, live, in a second terminal — reloads the
# newest checkpoint between episodes so behavior updates as training runs
python watch_agent.py
```

No install step needed — `.venv` already has `torch`, `stable_baselines3`, `vizdoom`, and `gymnasium` installed (see Environment below).

To resume training from a checkpoint instead of starting over:
```python
model = PPO.load("models/checkpoints/ppo_basic_<N>_steps", env=vec_env)
model.learn(total_timesteps=..., reset_num_timesteps=False, tb_log_name="ppo_basic")
```

## Key files

- `envs/basic_env.py` — Gymnasium env factory (`make_basic_env`) for ViZDoom's `basic.wad` scenario. Registers `VizdoomBasic-v1` via `vizdoom.gymnasium_wrapper`, strips the dict observation down to just the screen buffer (`ScreenOnlyObservation`), forces native render resolution down to `RES_160X120`, then applies grayscale → resize to 84×84 → explicit reshape back to `(84, 84, 1)` (see "Known gotchas"). `frame_skip=4` by default. Each call generates a unique per-process `doom_config_path` under `configs/` (see "Known gotchas").
- `train_basic.py` — PPO training entry point. `CnnPolicy`, 14 parallel envs via `make_vec_env(..., vec_env_cls=SubprocVecEnv)`, frame-stacking applied afterward at the vec-env level via `VecFrameStack(n_stack=4)` (not per-env — see Performance below), 100k timesteps, `CheckpointCallback` saving to `models/checkpoints/` every ~10k timesteps, final save to `models/ppo_basic`, `device="cuda"`.
- `watch_agent.py` — loads the newest file in `models/checkpoints/`, plays one episode with a visible window (`DummyVecEnv` + `VecFrameStack`, matching training's observation shape), reloads before the next episode. Run alongside `train_basic.py` to watch behavior evolve live without slowing training down.
- `models/checkpoints/` — periodic checkpoints (currently populated once a training run has been started since the last update; check freshness before assuming a checkpoint is current).
- `models/` — final saved model (`ppo_basic.zip`) — only written once, at the very end of a full `model.learn()` call.
- `logs/` — TensorBoard logs.
- `configs/` — auto-generated per-process ZDoom ini files (one per `SubprocVecEnv` worker, named by PID). Gitignored; safe to delete when nothing is running.
- `_vizdoom.ini` — leftover from before per-process config paths were added; no longer written to by the current code, safe to ignore/delete.

## Environment

Project-local `.venv` (not the global Python install):

| Package | Version |
|---|---|
| Python | 3.14.6 |
| `vizdoom` | 1.3.0 |
| `gymnasium` | 1.3.0 |
| `torch` | 2.12.1+cu130 |
| `stable_baselines3` | 2.9.0 |

CPU: AMD Ryzen 7 5800H, 8 physical cores / 16 logical (SMT) — relevant because ViZDoom's engine step is CPU-bound, single-threaded work, so `SubprocVecEnv` parallelism is capped more by physical cores than logical ones (diminishing returns past ~8-10 workers; `N_ENVS = 14` leaves headroom for the main process and OS).

GPU: NVIDIA RTX 3060 (6GB), CUDA build of `torch` matches it — `nvidia-smi` confirmed working. Training scripts default to `device="cuda"`. Note: for `basic.wad`'s tiny CNN, the GPU is not the bottleneck — env stepping (CPU) is.

## Performance tuning already applied (don't redo, do reconsider if scaling to harder scenarios)

- `SubprocVecEnv` instead of `DummyVecEnv` (the `make_vec_env` default) — `DummyVecEnv` steps all envs sequentially in one process, leaving most cores idle.
- `frame_skip=4` instead of the ViZDoom default of 1 — act once every 4 engine ticks.
- Native render resolution forced to `RES_160X120` (down from `basic.wad`'s default `RES_320X240`) since the pipeline resizes to 84×84 anyway.
- Frame-stacking moved from per-env (`gymnasium.wrappers.FrameStackObservation`) to vec-env level (`stable_baselines3...VecFrameStack`) — each `SubprocVecEnv` worker now ships one `(84,84,1)` frame across its process pipe per step instead of a full `(84,84,4)` stack, a 4x cut in inter-process payload.

## Known gotchas (already fixed — read before re-debugging these from scratch)

- **`viz_instance_id is write protected`**: multiple `SubprocVecEnv` workers launching at once all defaulted to the same `_vizdoom.ini` in the working directory and raced on a ZDoom cvar used for shared-memory IPC naming. Fixed via `doom_config_path` set to a unique, PID-named file per worker in `envs/basic_env.py`.
- **`gymnasium.wrappers.ResizeObservation` silently drops a size-1 channel dimension.** It declares output shape `(84, 84, 1)` but internally calls `cv2.resize`, which returns `(84, 84)` for single-channel input — the declared and actual shapes disagree, and `VecFrameStack` breaks on the mismatch (`ValueError: could not broadcast ... into shape (3,84,84,4)` from a 3-env smoke test). Fixed by resizing with `keep_dim=False` (plain 2D array, no declared/actual mismatch), then explicitly restoring the channel dim with `gymnasium.wrappers.ReshapeObservation(env, (84, 84, 1))`.

## Next steps (in order)

1. Run `train_basic.py` end-to-end and confirm episode reward trends upward in TensorBoard — check `models/` and `logs/tensorboard/` for freshness before assuming this is done.
2. Once `basic.wad` works, move to harder built-in scenarios: `deadly_corridor.wad`, then `defend_the_center.wad`.
3. Consider reward shaping (damage dealt, kills, pickups, death penalty, discourage standing still) as scenarios get harder — this matters more than architecture size.
4. After a working low-level controller exists, optionally revisit Option B (LLM high-level planner) — see below.

## Architecture decision

Two options were weighed:

- **Option A — RL policy network (chosen, in progress):** small CNN trained with PPO on ViZDoom's pixel observations via `stable-baselines3`. Proven, standard approach; runs fine on a single consumer GPU or CPU for the basic scenarios.
- **Option B — LLM as high-level planner + separate fast controller (deferred):** an LLM (e.g. Gemma 4B via the user's existing LM Studio setup — see `E:\Code_Base\Claude_Cowork\gemma-web-agent\`, which already has an OpenAI-compatible LM Studio client and tool-calling agentic loop pattern) would issue strategic directives (`EXPLORE`, `RETREAT`, `ENGAGE`, `FIND_HEALTH`) every 1-2 seconds from a text summary of game state, while the Option A policy handles frame-by-frame control. Mirrors how "LLM plays a game" research (DeepMind SIMA, Voyager) is architected — the LLM never sits in the twitch-reaction path.

Option A was built first because Option B needs a working low-level controller before a planner layered on top is useful, and Option A alone is already a complete project.

## Related workspace projects (reference only, not modified as part of this work)

- `E:\Code_Base\ViZDoom\` — the ViZDoom platform itself, has its own `CLAUDE.md`. This project depends on it as the game environment but lives in its own directory.
- `E:\Code_Base\Open-ai-gym\` — vendored unmodified clone of `openai/gym`, kept as a Gymnasium-API reference for ViZDoom. Do not edit.
- `E:\Code_Base\Claude_Cowork\gemma-web-agent\` — working LM Studio + Gemma integration pattern that Option B would reuse for the LLM planner layer.
- `E:\Code_Base\nn-zero-to-hero\` — Karpathy's from-scratch NN course notebooks; useful tone/level reference if explaining the RL/CNN architecture from first principles is helpful.
