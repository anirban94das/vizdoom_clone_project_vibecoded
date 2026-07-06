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

# Train PPO on the defend_the_center.wad scenario (fixed position, only
# turning + attacking — don't run simultaneously with the other two either)
python train_defend_the_center.py

# All three accept reward-shaping override flags (defaults differ per
# scenario, see envs/common.py in Key files below):
#   --kill-reward-bonus --hit-reward-bonus --exploration-bonus-per-cell
#   --exploration-cell-size --weapon-pickup-bonus
python train_deadly_corridor.py --kill-reward-bonus 30.0

# Watch training progress (reward/loss curves) — all scenarios' runs show
# up side by side since they share logs/tensorboard
tensorboard --logdir logs/tensorboard

# Watch the agent actually play, live, in a second terminal — reloads the
# newest checkpoint between episodes so behavior updates as training runs
python watch_agent.py                       # basic.wad
python watch_agent_deadly_corridor.py       # deadly_corridor.wad
python watch_agent_defend_the_center.py     # defend_the_center.wad

# Desktop launcher: pick a scenario, edit reward-shaping values, and
# start/stop training or live-watching without the CLI at all
.venv\Scripts\python.exe train_ui.py
```

No install step needed on this machine — `.venv` already has `torch`, `stable_baselines3`, `vizdoom`, `gymnasium`, `matplotlib`, and `visualtorch` installed (see Environment below). To bootstrap `.venv` from scratch elsewhere, `setup_env.sh` (Git Bash/WSL/POSIX) and `setup_env.bat` (cmd) both resolve a Python 3.14 interpreter via the `py` launcher, create `.venv`, and `pip install -r requirements.txt`, skipping the (slow) reinstall if `requirements.txt`'s hash hasn't changed since the last successful run.

`setup_env.bat` previously had a real bug (now fixed): it shells out to `powershell -NoProfile -Command "(Get-FileHash ...).Hash"` to compute that hash, but `Get-FileHash` isn't always autoloaded in a `-NoProfile` session — when that silently fails, the hash comparison sees two empty strings as "unchanged" and skips installing dependencies entirely, which is what caused the "failed partway through" behavior the old commit message referenced. Fixed by explicitly `Import-Module Microsoft.PowerShell.Utility` before calling `Get-FileHash`, plus a fallback that forces a reinstall (rather than silently skipping one) if the hash still can't be computed. `setup_env.sh` never had this bug (it uses `sha256sum` directly, no PowerShell involved) so it remains the more battle-tested option, but `setup_env.bat` should now work standalone too.

All three of `train_basic.py`, `train_deadly_corridor.py`, and `train_defend_the_center.py` auto-resume: on startup each checks for a single fixed model file (`models/latest/ppo_basic.zip` / `models/latest/ppo_deadly_corridor_shaped.zip` / `models/latest/ppo_defend_the_center.zip`) and `PPO.load`s it if present (falling back to a fresh `CnnPolicy` otherwise). There are no step-numbered checkpoints — a shared `OverwriteCheckpointCallback` (in `training_utils.py`) saves to that same fixed path roughly every 10k timesteps, overwriting it in place, so exactly one file per scenario exists at any time. Each run then trains for `TOTAL_TIMESTEPS` *additional* steps on top of wherever that file left off — `reset_num_timesteps` is set accordingly. To force a from-scratch run, delete that scenario's file under `models/latest/` first.

`train_deadly_corridor.py` trains/logs under `ppo_deadly_corridor_shaped` (not the older `ppo_deadly_corridor` prefix) since it enables reward shaping by default (see `envs/deadly_corridor_env.py` below) — a different reward function than the `ppo_deadly_corridor` baseline run it evolved from. If `models/latest/ppo_deadly_corridor_shaped.zip` doesn't exist yet, it warm-starts weights from `models/ppo_deadly_corridor.zip` (the baseline run's final saved model) instead (visual features/aiming/movement carry over) but resets the timestep/TensorBoard counter, since the reward scale underneath changed — expect a visible jump/dip in the reward curve right at that handoff.

**Don't run more than one training script at once** — each spawns a full set of `SubprocVecEnv` workers (`N_ENVS`), and this machine has 8 physical cores, so running more than one simultaneously oversubscribes and slows all of them down.

## Key files

- `envs/common.py` — shared Gymnasium env factory (`make_vizdoom_env(env_id, ...)`) used by every per-scenario env module. Strips the dict observation down to just the screen buffer (`ScreenOnlyObservation`), forces native render resolution down to `RES_160X120`, then applies grayscale → resize to 84×84 → explicit reshape back to `(84, 84, 1)` (see "Known gotchas"). `frame_skip=4` by default. Each call generates a unique per-process `doom_config_path` under `configs/` (see "Known gotchas"). Also defines four opt-in reward-shaping wrappers, all disabled by default (0.0) so scenarios that don't ask for them are unaffected: `KillRewardBonus` (adds `bonus_per_kill` reward per `GameVariable.KILLCOUNT` increment), `HitRewardBonus` (adds `bonus_per_hit` reward per `GameVariable.HITCOUNT` increment — fires on every successful hit landed on an enemy, not just kills, so it's denser signal leading up to a kill), `ExplorationBonus` (adds `bonus_per_cell` reward the first time each episode a discretized `(POSITION_X, POSITION_Y)` grid cell is visited — grid-cell novelty rather than raw per-step distance, so oscillating in place doesn't farm reward), and `WeaponPickupBonus` (adds `bonus_per_weapon` reward the first time each episode a `WEAPON0`..`WEAPON9` ownership flag flips to owned). All four read game variables directly off `env.unwrapped.game`, which works even for variables not listed in the scenario's `.cfg` `available_game_variables` (confirmed empirically for `POSITION_X`/`POSITION_Y`/`WEAPON3`/`HITCOUNT` against `deadly_corridor.cfg`, which only declares `HEALTH`). A fifth wrapper, `EpisodeStatsWrapper`, is always applied (not opt-in like the four above) — it tracks the same kills/hits/explored-cells/weapons-picked-up per episode purely for reporting (`info["episode_stats"]` on the terminal step, consumed by `training_utils.EpisodeRecapCallback`), so every scenario gets a behavior recap even when its reward bonuses are off (e.g. `basic.wad`).
- `envs/basic_env.py` / `envs/deadly_corridor_env.py` / `envs/defend_the_center_env.py` — thin per-scenario wrappers around `envs.common.make_vizdoom_env`, registering `VizdoomBasic-v1` / `VizdoomDeadlyCorridor-v1` / `VizdoomDefendCenter-v1` respectively via `vizdoom.gymnasium_wrapper`. `deadly_corridor.cfg` defines `death_penalty=100` and `doom_skill=5` but doesn't score kills, hits, exploration, or pickups directly, so `make_deadly_corridor_env` enables all four reward-shaping wrappers by default (`kill_reward_bonus=20.0`, `hit_reward_bonus=5.0`, `exploration_bonus_per_cell=1.0`, `exploration_cell_size=32.0` map units ≈ 1 Doom grid tile, `weapon_pickup_bonus=15.0`). The scenario's monsters are `Zombieman` and `ShotgunGuy` (confirmed via ViZDoom's labels buffer) — the latter drops a pickupable shotgun on death (vanilla Doom behavior), which is what `weapon_pickup_bonus` rewards picking up. `defend_the_center.cfg` fixes the player at the center of the room (only `TURN_LEFT`/`TURN_RIGHT`/`ATTACK` are available buttons — no movement) and already scores +1 per kill / a small `death_penalty=1` internally, so `make_defend_the_center_env` enables `kill_reward_bonus=20.0` and `hit_reward_bonus=5.0` on top of that but leaves `exploration_bonus_per_cell=0.0` (standing still is the actual objective here, unlike deadly_corridor) and `weapon_pickup_bonus=0.0` (no pickupable weapons in this scenario).
- `training_utils.py` — shared `OverwriteCheckpointCallback`, used by `train_basic.py`, `train_deadly_corridor.py`, and `train_defend_the_center.py`. Behaves like `stable_baselines3`'s `CheckpointCallback` but saves to one fixed path every time instead of appending the step count to the filename, so periodic saves and the final `model.save()` all target the same file. Also defines `EpisodeRecapCallback` — collects each finished episode's reward (from SB3's `Monitor` wrapper, `info["episode"]`) alongside `envs.common.EpisodeStatsWrapper`'s stats (`info["episode_stats"]`) for the duration of one `model.learn()` call; at `_on_training_end` it compares the first ~20 episodes seen this run against the last ~20, prints the comparison, and appends one JSON line to `logs/training_history.jsonl` (via `history_path`) — so past runs stay visible instead of being overwritten like the model file. Each `train_*.py` script passes its own instance alongside `checkpoint_callback` as a list to `model.learn(callback=[...])`.
- `train_basic.py` / `train_deadly_corridor.py` / `train_defend_the_center.py` — PPO training entry points, one per scenario, same structure (`CnnPolicy`, parallel envs via `make_vec_env(..., vec_env_cls=SubprocVecEnv)`, frame-stacking applied afterward at the vec-env level via `VecFrameStack(n_stack=4)` — not per-env, see Performance below — `OverwriteCheckpointCallback` saving to a single fixed file under `models/latest/` every ~10k timesteps, `device="cuda"`). All three take `argparse` flags (`--kill-reward-bonus`, `--hit-reward-bonus`, `--exploration-bonus-per-cell`, `--exploration-cell-size`, `--weapon-pickup-bonus`) that override that scenario's reward-shaping defaults for one run — overriding them doesn't invalidate an existing checkpoint, it just changes what reward the resumed agent trains against going forward. `train_basic.py` uses `N_ENVS=12` and runs 100k timesteps per invocation, model at `models/latest/ppo_basic.zip`, all bonuses default to 0.0 (off). `train_deadly_corridor.py` also uses `N_ENVS=12` and runs 300k timesteps per invocation (harder, sparser-reward scenario), model at `models/latest/ppo_deadly_corridor_shaped.zip`, bonuses default to this scenario's shaped values (`kill_reward_bonus=20.0`, `hit_reward_bonus=5.0`, `exploration_bonus_per_cell=1.0`, `weapon_pickup_bonus=15.0`). `train_defend_the_center.py` also uses `N_ENVS=12` and runs 300k timesteps per invocation, model at `models/latest/ppo_defend_the_center.zip`, bonuses default to `kill_reward_bonus=20.0`, `hit_reward_bonus=5.0`, `exploration_bonus_per_cell=0.0`, `weapon_pickup_bonus=0.0` — unlike deadly_corridor there's no earlier unshaped baseline to warm-start from, so the first run on this scenario always starts from a fresh `CnnPolicy`.
- `watch_agent.py` / `watch_agent_deadly_corridor.py` / `watch_agent_defend_the_center.py` — reloads that scenario's fixed file in `models/latest/` before every episode, plays one episode with a visible window (`DummyVecEnv` + `VecFrameStack`, matching training's observation shape). Run alongside the matching `train_*.py` to watch behavior evolve live without slowing training down — since training overwrites the same file in place, no extra bookkeeping is needed to find "the latest" one.
- `train_ui.py` — Tkinter desktop launcher wrapping the `train_*.py`/`watch_agent_*.py` scripts above as subprocesses (unmodified — it's just a launcher); always resolves and runs the project's own `.venv` interpreter regardless of which Python started the UI. Lets you pick a scenario (`LEVELS` dict — currently Basic, Deadly Corridor (shaped), Defend the Center), edit the same reward-shaping fields as the CLI flags (pre-filled with that scenario's defaults from `REWARD_DEFAULTS`), and start/stop training. Training and watching are tracked as separate subprocesses: stopping training uses `taskkill /F /T` (not a plain terminate) since `train_*.py` itself spawns `SubprocVecEnv` worker processes that would otherwise be orphaned; watching is a single-process `DummyVecEnv` with its own visible window, so it can run alongside training without the oversubscription concern below. A third button, "Visualize Model," runs `visualize_PPO_model.py` as a one-shot subprocess against the selected level's `models/latest/*.zip` and renders the resulting PNG inline in a right-hand panel (Tk's built-in PNG support, no Pillow needed for display) next to the log — so a running Watch Agent's output and a model-architecture render are visible in the same window at once. Guards against clicking it before that level has a saved model. Adding a new scenario here means adding one entry each to `LEVELS`, `WATCH_SCRIPTS`, `MODEL_PATHS`, `VIZ_OUTPUT_NAMES`, and `REWARD_DEFAULTS`.
- `visualize_PPO_model.py` — one-shot diagnostic script (no ViZDoom/gymnasium import, not part of the training loop) that renders a PPO `CnnPolicy`'s architecture as a PNG via `visualtorch`. `--model models/latest/ppo_basic.zip` loads the real trained model and pulls out its actual `pi_features_extractor` → `mlp_extractor.policy_net` → `action_net` branch, so the render reflects the real action count (4 or 8 depending on scenario — defend_the_center is also 4, same button count as basic); with no `--model` it renders a hardcoded untrained stand-in sized for `basic.wad`. Note: `policy.observation_space.shape` is channel-first `(C, H, W)` (SB3's `VecTransposeImage` already transposes it before the policy sees it) — the script used to assume `(H, W, C)` and crashed with a channel-mismatch error on any real `--model`; fixed to read it as `(n_stack, height, width)` directly. Requires `matplotlib` + `visualtorch` (both in `requirements.txt`).
- `models/latest/` — the single actively-trained model per scenario (`ppo_basic.zip`, `ppo_deadly_corridor_shaped.zip`, `ppo_defend_the_center.zip`), overwritten in place roughly every 10k timesteps and again at the end of `model.learn()`. This is what auto-resume and the `watch_agent_*.py` scripts read.
- `models/` (top level) — one-off final saves from before the single-file scheme (`ppo_basic.zip`, `ppo_deadly_corridor.zip`, `ppo_deadly_corridor_shaped.zip`). `ppo_deadly_corridor.zip` is still read as the warm-start source for a from-scratch `ppo_deadly_corridor_shaped` run; the others are stale and safe to ignore.
- `models/checkpoints/` — leftover step-numbered checkpoints from before the single-file scheme (hundreds of files, `ppo_basic_*_steps.zip` etc.). No longer written to; safe to delete if disk space matters, kept for now as history.
- `logs/` — TensorBoard logs, one run subdirectory per scenario/invocation, plus `logs/training_history.jsonl` — one JSON line appended per completed `model.learn()` call by `EpisodeRecapCallback` (reward/kills/hits/cells-explored/weapons-picked-up, first-~20-episodes vs last-~20-episodes of that run), so past runs' behavior recaps accumulate across invocations rather than only showing in that run's console output.
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

CPU: AMD Ryzen 7 5800H, 8 physical cores / 16 logical (SMT) — relevant because ViZDoom's engine step is CPU-bound, single-threaded work, so `SubprocVecEnv` parallelism is capped more by physical cores than logical ones. Both training scripts use `N_ENVS = 12`; `N_ENVS = 14` was tried and hit a startup race (all 14 game engines booting simultaneously left one worker half-initialized), so 12 is the current throughput/stability sweet spot on this machine.

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
2. ~~Run `train_deadly_corridor.py` end-to-end~~ — done, baseline (`ppo_deadly_corridor` prefix) reached step 950,000 with `ep_rew_mean` climbing from -88 to +151. Reward-shaped version (`kill_reward_bonus` + `exploration_bonus_per_cell`, `ppo_deadly_corridor_shaped` prefix, warm-started from that baseline) is up and running.
3. ~~Implement `defend_the_center.wad`~~ — done (`envs/defend_the_center_env.py`, `train_defend_the_center.py`, `watch_agent_defend_the_center.py`, wired into `train_ui.py`). No earlier baseline to warm-start from, so its first run trains a fresh `CnnPolicy`. Next: run it end-to-end and confirm `ep_rew_mean` trends upward.
4. Move to the next scenario in the roadmap below, `defend_the_line.wad` — same `Dict(screen, gamevariables)` observation shape and fixed-position button set as `defend_the_center`, so it should follow the exact same implementation pattern (copy `envs/defend_the_center_env.py` → `envs/defend_the_line_env.py`, swap `ENV_ID` to `VizdoomDefendLine-v1`, etc.). Confirm its `.cfg` button/reward setup with a quick read before assuming it's identical to `defend_the_center.cfg`.
5. Consider reward shaping (damage dealt, kills, pickups, death penalty, discourage standing still) if a scenario's built-in reward isn't sufficient — this matters more than architecture size.
6. After a working low-level controller exists, optionally revisit Option B (LLM high-level planner) — see below.

## Scenario roadmap

Implemented so far, in `envs/`, one `train_*.py`/`watch_agent_*.py` pair each: `basic`, `deadly_corridor`, `defend_the_center`.

Remaining scenarios from [ViZDoom's `scenarios/` directory](https://github.com/Farama-Foundation/ViZDoom/tree/main/scenarios), in the order they're planned to be implemented (single-player only — this project trains one PPO agent, not multiplayer bots):

1. `defend_the_line` — like `defend_the_center` but enemies approach from a line rather than surrounding the player; likely same button set and reward shape.
2. `health_gathering` — survival scenario, collect health kits and avoid the acid floor; no combat, so `kill_reward_bonus`/`hit_reward_bonus`/`weapon_pickup_bonus` won't apply, but `health_change_bonus` (already in `envs/common.py`) is a natural fit.
3. `health_gathering_supreme` — harder map layout for the same objective as `health_gathering`; implement after that one's working, likely reusing most of its env module.
4. `my_way_home` — navigation/exploration scenario with sparse reward; `exploration_bonus_per_cell` (already built for `deadly_corridor`) is directly applicable here, unlike `defend_the_center`/`defend_the_line`.
5. `predict_position` — aim-leading a moving target with a rocket launcher; single shot per episode, so kill/hit bonuses may need much larger magnitudes than the hitscan scenarios above to matter.
6. `take_cover` — survive incoming fireballs by dodging; no player weapon, so this is a pure survival/movement scenario like `health_gathering`.

Variants of `basic.wad`, lower priority since they exercise the same underlying task with a different observation/output surface rather than new agent behavior: `basic_audio` (adds an audio buffer to the observation — would need a new observation wrapper, not just a new `.cfg`), `basic_notifications` (adds in-game text notifications), `rocket_basic`/`simpler_basic` (rocket launcher / simplified basic — **not registered** in this project's installed `vizdoom` package's `gymnasium_wrapper` `__init__.py`, so using them would require a manual `gymnasium.register(...)` call rather than just pointing at an existing `Vizdoom*-v1` id).

Out of scope for this project (multiplayer, needs a bot/opponent process rather than a single trained agent): `deathmatch`, `cig`, `cig_with_unknown`, `multi_duel`, `multi_deathmatch`.

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
