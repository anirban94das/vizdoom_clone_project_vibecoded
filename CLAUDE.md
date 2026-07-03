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

# Train PPO on the harder deadly_corridor.wad scenario (don't run
# simultaneously with train_basic.py — see Key files below)
python train_deadly_corridor.py

# Watch training progress (reward/loss curves) — both scenarios' runs show
# up side by side since they share logs/tensorboard
tensorboard --logdir logs/tensorboard

# Watch the agent actually play, live, in a second terminal — reloads the
# newest checkpoint between episodes so behavior updates as training runs
python watch_agent.py                    # basic.wad
python watch_agent_deadly_corridor.py    # deadly_corridor.wad
```

No install step needed — `.venv` already has `torch`, `stable_baselines3`, `vizdoom`, and `gymnasium` installed (see Environment below).

Both `train_basic.py` and `train_deadly_corridor.py` auto-resume: on startup each globs `models/checkpoints/<prefix>_*_steps.zip` for its own prefix, picks the highest step count, and `PPO.load`s it if one exists (falling back to a fresh `CnnPolicy` otherwise). Each run then trains for `TOTAL_TIMESTEPS` *additional* steps on top of wherever the checkpoint left off — `reset_num_timesteps` is set accordingly. To force a from-scratch run, delete or move that scenario's checkpoints first.

`train_deadly_corridor.py` checkpoints/logs under `ppo_deadly_corridor_shaped` (not the older `ppo_deadly_corridor` prefix) since it enables reward shaping by default (see `envs/deadly_corridor_env.py` below) — a different reward function than the `ppo_deadly_corridor` baseline run it evolved from. If no `ppo_deadly_corridor_shaped` checkpoint exists yet, it warm-starts weights from the latest `ppo_deadly_corridor` baseline checkpoint instead (visual features/aiming/movement carry over) but resets the timestep/TensorBoard counter, since the reward scale underneath changed — expect a visible jump/dip in the reward curve right at that handoff.

**Don't run both training scripts at once** — each spawns a full set of `SubprocVecEnv` workers (`N_ENVS`), and this machine has 8 physical cores, so running both simultaneously oversubscribes and slows both down.

## Key files

- `envs/common.py` — shared Gymnasium env factory (`make_vizdoom_env(env_id, ...)`) used by every per-scenario env module. Strips the dict observation down to just the screen buffer (`ScreenOnlyObservation`), forces native render resolution down to `RES_160X120`, then applies grayscale → resize to 84×84 → explicit reshape back to `(84, 84, 1)` (see "Known gotchas"). `frame_skip=4` by default. Each call generates a unique per-process `doom_config_path` under `configs/` (see "Known gotchas"). Also defines three opt-in reward-shaping wrappers, all disabled by default (0.0) so scenarios that don't ask for them are unaffected: `KillRewardBonus` (adds `bonus_per_kill` reward per `GameVariable.KILLCOUNT` increment), `ExplorationBonus` (adds `bonus_per_cell` reward the first time each episode a discretized `(POSITION_X, POSITION_Y)` grid cell is visited — grid-cell novelty rather than raw per-step distance, so oscillating in place doesn't farm reward), and `WeaponPickupBonus` (adds `bonus_per_weapon` reward the first time each episode a `WEAPON0`..`WEAPON9` ownership flag flips to owned). All three read game variables directly off `env.unwrapped.game`, which works even for variables not listed in the scenario's `.cfg` `available_game_variables` (confirmed empirically for `POSITION_X`/`POSITION_Y`/`WEAPON3` against `deadly_corridor.cfg`, which only declares `HEALTH`).
- `envs/basic_env.py` / `envs/deadly_corridor_env.py` — thin per-scenario wrappers around `envs.common.make_vizdoom_env`, registering `VizdoomBasic-v1` / `VizdoomDeadlyCorridor-v1` respectively via `vizdoom.gymnasium_wrapper`. `deadly_corridor.cfg` defines `death_penalty=100` and `doom_skill=5` but doesn't score kills, exploration, or pickups directly, so `make_deadly_corridor_env` enables all three reward-shaping wrappers by default (`kill_reward_bonus=20.0`, `exploration_bonus_per_cell=1.0`, `exploration_cell_size=32.0` map units ≈ 1 Doom grid tile, `weapon_pickup_bonus=15.0`). The scenario's monsters are `Zombieman` and `ShotgunGuy` (confirmed via ViZDoom's labels buffer) — the latter drops a pickupable shotgun on death (vanilla Doom behavior), which is what `weapon_pickup_bonus` rewards picking up. `exploration_bonus_per_cell` is not a good default for `defend_the_center.wad` (next on the roadmap) — standing still is the actual objective there, so it should stay at 0 for that scenario; `kill_reward_bonus` and `weapon_pickup_bonus` likely still apply.
- `train_basic.py` / `train_deadly_corridor.py` — PPO training entry points, one per scenario, same structure (`CnnPolicy`, parallel envs via `make_vec_env(..., vec_env_cls=SubprocVecEnv)`, frame-stacking applied afterward at the vec-env level via `VecFrameStack(n_stack=4)` — not per-env, see Performance below — `CheckpointCallback` saving to `models/checkpoints/` every ~10k timesteps under a scenario-specific prefix, `device="cuda"`). `train_basic.py` uses `N_ENVS=8` and runs 100k timesteps per invocation, final save `models/ppo_basic`. `train_deadly_corridor.py` uses `N_ENVS=12` and runs 300k timesteps per invocation (harder, sparser-reward scenario), final save `models/ppo_deadly_corridor_shaped`.
- `watch_agent.py` / `watch_agent_deadly_corridor.py` — loads the newest checkpoint for that scenario's prefix in `models/checkpoints/`, plays one episode with a visible window (`DummyVecEnv` + `VecFrameStack`, matching training's observation shape), reloads before the next episode. Run alongside the matching `train_*.py` to watch behavior evolve live without slowing training down. `watch_agent_deadly_corridor.py` currently still points at the `ppo_deadly_corridor` baseline prefix, not `ppo_deadly_corridor_shaped` — flip its `CHECKPOINT_PREFIX` constant once shaped checkpoints exist and you want to watch that agent instead.
- `models/checkpoints/` — periodic checkpoints for all scenarios, distinguished by filename prefix (`ppo_basic_*`, `ppo_deadly_corridor_*` baseline, `ppo_deadly_corridor_shaped_*` reward-shaped); check freshness before assuming a checkpoint is current.
- `models/` — final saved models (`ppo_basic.zip`, `ppo_deadly_corridor_shaped.zip`) — only written once, at the very end of a full `model.learn()` call.
- `logs/` — TensorBoard logs, one run subdirectory per scenario/invocation.
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

1. ~~Run `train_basic.py` end-to-end and confirm episode reward trends upward in TensorBoard~~ — done, agent performs well on `basic.wad`.
2. ~~Run `train_deadly_corridor.py` end-to-end~~ — done, baseline (`ppo_deadly_corridor` prefix) reached step 950,000 with `ep_rew_mean` climbing from -88 to +151. Now run the reward-shaped version (`kill_reward_bonus` + `exploration_bonus_per_cell`, `ppo_deadly_corridor_shaped` prefix, warm-started from that baseline) and confirm reward keeps climbing after the expected reward-scale jump/dip at the handoff.
3. Once `deadly_corridor.wad`'s shaped-reward run looks good, move to `defend_the_center.wad` (already confirmed registered as `VizdoomDefendCenter-v1`, same `Dict(screen, gamevariables)` observation shape — same `envs/common.py` factory pattern applies, `kill_reward_bonus` likely useful there too but leave `exploration_bonus_per_cell=0` since standing still is the objective).
4. Consider reward shaping (damage dealt, kills, pickups, death penalty, discourage standing still) if a scenario's built-in reward isn't sufficient — this matters more than architecture size.
5. After a working low-level controller exists, optionally revisit Option B (LLM high-level planner) — see below.

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
