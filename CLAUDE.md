# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

A neural network that plays DOOM, using `E:\Code_Base\ViZDoom` as the game environment. The agent is a CNN policy trained with PPO (`stable-baselines3`) on ViZDoom's Gymnasium-compatible API — not an LLM. An LLM controller was ruled out early: ViZDoom needs a decision every ~30ms, which is far faster than any local LLM can generate, and a text-only model can't see the screen without added latency. See "Architecture decision" below for the full reasoning and the deferred Option B.

## Commands

```powershell
# Activate the project venv (Python 3.14.6, not the global install)
.venv\Scripts\Activate.ps1

# Train PPO on a scenario — one train_*.py per scenario, all thin wrappers
# around train_common.run_training (never run two at once — 8 physical cores):
python train_basic.py
python train_simpler_basic.py
python train_rocket_basic.py
python train_basic_audio.py           # screen+audio Dict obs; needs OpenAL (untested)
python train_deadly_corridor.py
python train_defend_the_center.py
python train_defend_the_line.py
python train_health_gathering.py
python train_health_gathering_supreme.py   # warm-starts from health_gathering's model
python train_my_way_home.py
python train_predict_position.py
python train_take_cover.py

# Train on a FULL DOOM/DOOM II level (E1M1..E4M9 / MAP01..MAP32). Uses the
# Freedoom WADs bundled with vizdoom unless a real doom.wad/doom2.wad is in
# wads/ (see wads/README.md). One model per map (models/latest/ppo_doom_<MAP>.zip).
python train_doom_level.py --map E1M1 --skill 3
python train_doom_level.py --map MAP01

# Every train script accepts the same reward-shaping override flags
# (defaults differ per scenario — see that scenario's envs/*_env.py):
#   --kill-reward-bonus --hit-reward-bonus --exploration-bonus-per-cell
#   --exploration-cell-size --weapon-pickup-bonus --damage-dealt-bonus
#   --damage-taken-penalty --health-change-bonus --armor-change-bonus
# plus PPO stability guards --ent-coef / --target-kl (defaults are SB3's,
# except deadly_corridor's 0.01/0.03 — see its docstring)
python train_deadly_corridor.py --kill-reward-bonus 30.0

# Watch training progress (reward/loss curves) — all scenarios' runs show
# up side by side since they share logs/tensorboard
tensorboard --logdir logs/tensorboard

# Watch the agent actually play, live, in a second terminal — reloads the
# newest checkpoint between episodes so behavior updates as training runs.
# One watch_agent_*.py per scenario, same names as the train scripts:
python watch_agent.py                            # basic.wad
python watch_agent_deadly_corridor.py            # (etc.)
python watch_agent_doom_level.py --map E1M1      # full levels take --map/--skill

# Export a trained model to a single shareable file / import one back
# (scenario tag embedded; import backs up the old model to models/backups/)
python export_model.py deadly_corridor --out D:\backups\corridor_v1.zip
python import_model.py D:\backups\corridor_v1.zip --scenario deadly_corridor
python export_model.py doom_E1M1                 # full levels: doom_<MAP>

# Desktop launcher: pick any of the 14 levels, edit reward-shaping values,
# start/stop training or live-watching, visualize the model, and
# export/import models — all without the CLI
.venv\Scripts\python.exe train_ui.py
```

No install step needed on this machine — `.venv` already has `torch`, `stable_baselines3`, `vizdoom`, `gymnasium`, `matplotlib`, and `visualtorch` installed (see Environment below). To bootstrap `.venv` from scratch elsewhere, `setup_env.sh` (Git Bash/WSL/POSIX) and `setup_env.bat` (cmd) both resolve a Python 3.14 interpreter via the `py` launcher, create `.venv`, and `pip install -r requirements.txt`, skipping the (slow) reinstall if `requirements.txt`'s hash hasn't changed since the last successful run.

`setup_env.bat` previously had a real bug (now fixed): it shells out to `powershell -NoProfile -Command "(Get-FileHash ...).Hash"` to compute that hash, but `Get-FileHash` isn't always autoloaded in a `-NoProfile` session — when that silently fails, the hash comparison sees two empty strings as "unchanged" and skips installing dependencies entirely, which is what caused the "failed partway through" behavior the old commit message referenced. Fixed by explicitly `Import-Module Microsoft.PowerShell.Utility` before calling `Get-FileHash`, plus a fallback that forces a reinstall (rather than silently skipping one) if the hash still can't be computed. `setup_env.sh` never had this bug (it uses `sha256sum` directly, no PowerShell involved) so it remains the more battle-tested option, but `setup_env.bat` should now work standalone too.

Every `train_*.py` auto-resumes (the logic lives in `train_common.run_training`): on startup it checks for that scenario's single fixed model file under `models/latest/` and `PPO.load`s it if present (falling back to a warm-start model where one is configured, else a fresh policy). There are no step-numbered checkpoints — a shared `OverwriteCheckpointCallback` (in `training_utils.py`) saves to that same fixed path roughly every 10k timesteps, overwriting it in place, so exactly one file per scenario exists at any time. Each run then trains for `TOTAL_TIMESTEPS` *additional* steps on top of wherever that file left off — `reset_num_timesteps` is set accordingly. To force a from-scratch run, delete that scenario's file under `models/latest/` first. Importing a model via `import_model.py` (or the UI's Import button) replaces that same file, so auto-resume and watching pick it up with no further steps.

`train_deadly_corridor.py` trains/logs under `ppo_deadly_corridor_shaped` (not the older `ppo_deadly_corridor` prefix) since it enables reward shaping by default (see `envs/deadly_corridor_env.py` below) — a different reward function than the `ppo_deadly_corridor` baseline run it evolved from. If `models/latest/ppo_deadly_corridor_shaped.zip` doesn't exist yet, it warm-starts weights from `models/ppo_deadly_corridor.zip` (the baseline run's final saved model) instead (visual features/aiming/movement carry over) but resets the timestep/TensorBoard counter, since the reward scale underneath changed — expect a visible jump/dip in the reward curve right at that handoff.

**Don't run more than one training script at once** — each spawns a full set of `SubprocVecEnv` workers (`N_ENVS`), and this machine has 8 physical cores, so running more than one simultaneously oversubscribes and slows all of them down.

## Key files

- `envs/common.py` — shared Gymnasium env plumbing used by every per-scenario env module, now split into three composable pieces: `make_raw_vizdoom_env(env_id, ..., **extra_game_kwargs)` (the bare `gym.make` with the per-process `doom_config_path` under `configs/`, startup jitter, `RES_160X120`, and a forced `screen_format=RGB24` — see "Known gotchas" for why; `extra_game_kwargs` pass any `.cfg` key or constructor arg through to the env, which is how the full-level envs override `doom_map`/`doom_game_path`/`episode_timeout`/etc.), `apply_stats_and_reward_shaping(env, ...)` (the always-on `EpisodeStatsWrapper` plus whichever opt-in bonuses are non-zero), and `make_vizdoom_env(...)` (both of the above plus the standard screen pipeline: `ScreenOnlyObservation` → grayscale → resize 84×84 → reshape `(84, 84, 1)`, see "Known gotchas"). `frame_skip=4` by default. Nine opt-in reward-shaping wrappers, all off (0.0) unless a scenario's env module enables them: `KillRewardBonus` (per `KILLCOUNT` increment), `HitRewardBonus` (per `HITCOUNT` increment — every landed hit, denser than kills), `ExplorationBonus` (first visit to each discretized `(POSITION_X, POSITION_Y)` grid cell per episode — grid-cell novelty, so oscillating in place doesn't farm reward), `WeaponPickupBonus` (first time a `WEAPON0`..`WEAPON9` ownership flag flips per episode), `DamageDealtBonus` / `DamageTakenPenalty` (per `DAMAGECOUNT` / `DAMAGE_TAKEN` point), `HealthChangeBonus` / `ArmorChangeBonus` (per net `HEALTH`/`ARMOR` point delta, symmetric), all reading game variables directly off `env.unwrapped.game` (works even for variables not in the `.cfg`'s `available_game_variables` — confirmed empirically against `deadly_corridor.cfg`). `EpisodeStatsWrapper` is always applied — kills/hits/damage/explored-cells/weapons per episode via `info["episode_stats"]`, consumed by `training_utils.EpisodeRecapCallback`.
- `envs/*_env.py` (one per scenario) — thin per-scenario wrappers around `envs.common.make_vizdoom_env` whose only real content is that scenario's shaping defaults, documented in each module's docstring: `basic_env` / `simpler_basic_env` / `rocket_basic_env` (all bonuses off — built-in reward suffices; the latter two also self-register `VizdoomSimplerBasic-v1`/`VizdoomRocketBasic-v1` since the vizdoom package doesn't), `deadly_corridor_env` (kill 20 / hit 5 / exploration 1.0 per 32-unit cell / weapon-pickup 15 — the scenario scores distance+death only; its `ShotgunGuy`s drop a pickupable shotgun), `defend_the_center_env` and `defend_the_line_env` (kill 20 / hit 5; fixed-position scenarios, exploration stays 0), `health_gathering_env` and `health_gathering_supreme_env` (health_change 1.0; no combat), `my_way_home_env` (exploration 1.0 — the built-in reward is one +1 at the goal, about as sparse as it gets), `predict_position_env` (kill 100 / hit 25 — single slow rocket per episode needs the success event to dominate), `take_cover_env` (damage_taken_penalty 0.5 — distinguishes near-miss from hit while +1/tic survival accrues).
- `envs/basic_audio_env.py` — the one non-screen-only observation: Dict `{screen (84,84,1), audio (flattened normalized waveform over the frame_skip window)}` for SB3's `MultiInputPolicy`; built from `make_raw_vizdoom_env` + `apply_stats_and_reward_shaping` + its own cv2 preprocessing since gymnasium's Grayscale/Resize wrappers don't reach inside Dicts. **Needs a working OpenAL device — untested on this machine**; if `DoomGame.init()` raises at startup, that's the first suspect.
- `envs/doom_level_env.py` — full DOOM/DOOM II game levels. vizdoom 1.3.0 pre-registers `Vizdoom{Doom,Doom2,Freedoom1,Freedoom2}{MAP}-S{1..5}-v0` for every map/skill and bundles the Freedoom WADs, so this works with zero downloads; `make_doom_level_env(map_id, skill)` picks the env id from the map name, auto-detecting a commercial `wads/doom.wad`/`wads/doom2.wad` (passed as `doom_game_path`) and falling back to Freedoom. Overrides vs. the stock full-game cfgs: `max_buttons_pressed=1` (Discrete 20-action space consistent with the scenarios, instead of the registered MultiBinary), `episode_timeout=21000` (10 min, down from 60), `map_exit_reward=1000` + `death_penalty=100`, audio/automap buffers off. Shaping defaults: the full deadly_corridor combat set + exploration + health/armor deltas (big maps, pickups matter).
- `train_common.py` — the shared runner every `train_*.py` / `watch_agent_*.py` delegates to. `build_parser(reward_defaults, ent_coef, target_kl)` gives every scenario the same nine reward flags + `--ent-coef`/`--target-kl`; `run_training(...)` holds the whole formerly-copy-pasted body (SubprocVecEnv construction, optional `VecFrameStack` via `n_stack` — 1 skips it for basic_audio — callbacks, auto-resume → warm-start → fresh decision, learn, save); `run_watch(...)` is the matching watch loop. Behavior is intentionally identical to the pre-refactor per-scenario scripts. Scenario scripts declare constants (`TOTAL_TIMESTEPS`, `MODEL_PATH`, `REWARD_DEFAULTS`, optional `WARM_START_PATH`, optional `policy=`) and call these.
- `model_io.py` + `export_model.py` / `import_model.py` — one-click model portability (also behind `train_ui.py`'s Export/Import buttons). Export copies the scenario's `models/latest/*.zip` and appends an `export_metadata.json` *inside* the zip (SB3 model files are ordinary zips; `PPO.load` ignores unknown entries, so the export stays directly loadable) recording scenario/timestamp/package versions. Import validates the zip (must contain SB3's `data` entry), refuses a scenario-tag mismatch without `--force` (exit code 3 — the UI uses it to offer an override), backs up the existing model to `models/backups/<name>_<timestamp>.zip`, then installs over `models/latest/` so auto-resume and watching pick it up unchanged. Scenario keys are `model_io.SCENARIO_MODELS`'s keys plus dynamic `doom_<MAP>`.
- `training_utils.py` — shared `OverwriteCheckpointCallback`, used via `train_common.run_training` by every `train_*.py`. Behaves like `stable_baselines3`'s `CheckpointCallback` but saves to one fixed path every time instead of appending the step count to the filename, so periodic saves and the final `model.save()` all target the same file. Also defines `EpisodeRecapCallback` — collects each finished episode's reward (from SB3's `Monitor` wrapper, `info["episode"]`) alongside `envs.common.EpisodeStatsWrapper`'s stats (`info["episode_stats"]`) for the duration of one `model.learn()` call; at `_on_training_end` it compares the first ~20 episodes seen this run against the last ~20, prints the comparison, and appends one JSON line to `logs/training_history.jsonl` (via `history_path`) — so past runs stay visible instead of being overwritten like the model file. Each `train_*.py` script passes its own instance alongside `checkpoint_callback` as a list to `model.learn(callback=[...])`.
- `train_*.py` (one per scenario, plus `train_doom_level.py --map <MAP> --skill <1-5>` for full levels) — thin declarative entry points over `train_common.run_training`: `CnnPolicy` (except `train_basic_audio.py`'s `MultiInputPolicy`), `N_ENVS=12`, `VecFrameStack(n_stack=4)` (except basic_audio's `n_stack=1`), `device="cuda"`, `OverwriteCheckpointCallback` to a single fixed file under `models/latest/` every ~10k timesteps. Timesteps per invocation: 100k for the basic-family, 300k for the scenario roadmap, 500k for full levels. Reward-flag overrides don't invalidate an existing checkpoint — they just change what reward the resumed agent trains against going forward. Warm starts: `train_deadly_corridor.py` from `models/ppo_deadly_corridor.zip` (pre-shaping baseline), `train_health_gathering_supreme.py` from `models/latest/ppo_health_gathering.zip` (same task, harder map); everything else starts fresh when no checkpoint exists.
- `watch_agent*.py` (one per scenario; `watch_agent_doom_level.py` takes `--map`/`--skill`) — via `train_common.run_watch`: reloads that scenario's fixed file in `models/latest/` before every episode, plays with a visible window (`DummyVecEnv` + `VecFrameStack` matching training's observation shape). Run alongside the matching `train_*.py` to watch behavior evolve live without slowing training down.
- `train_ui.py` — Tkinter desktop launcher wrapping the `train_*.py`/`watch_agent_*.py`/`export_model.py`/`import_model.py` scripts as subprocesses (unmodified — it's just a launcher); always resolves and runs the project's own `.venv` interpreter regardless of which Python started the UI. All 14 levels are in the `LEVELS` dict (values are now argument *lists*, since the two full-level entries share `train_doom_level.py` parameterized by `--map`); reward-shaping fields pre-fill from `REWARD_DEFAULTS` (built via the `_shaping()` helper to mirror each script's own defaults). Stopping training uses `taskkill /F /T` (not a plain terminate) since `train_*.py` spawns `SubprocVecEnv` workers that would otherwise be orphaned; watching is single-process with its own window so it can run alongside training. "Visualize Model" renders the selected level's model architecture PNG inline via `visualize_PPO_model.py`. "Export Model" / "Import Model" run the CLI scripts synchronously (file copies, sub-second): export opens a save dialog; import opens a file picker, warns if training is running (the next checkpoint save would overwrite the import), and on exit code 3 (scenario-tag mismatch) offers a `--force` retry. Adding a new scenario here means one entry each in `LEVELS`, `WATCH_SCRIPTS`, `SCENARIO_KEYS`, `MODEL_PATHS`, and `REWARD_DEFAULTS` (`VIZ_OUTPUT_NAMES` derives from `SCENARIO_KEYS`), plus its key in `model_io.SCENARIO_MODELS`.
- `visualize_PPO_model.py` — one-shot diagnostic script (no ViZDoom/gymnasium import, not part of the training loop) that renders a PPO `CnnPolicy`'s architecture as a PNG via `visualtorch`. `--model models/latest/ppo_basic.zip` loads the real trained model and pulls out its actual `pi_features_extractor` → `mlp_extractor.policy_net` → `action_net` branch, so the render reflects the real action count (4 or 8 depending on scenario — defend_the_center is also 4, same button count as basic); with no `--model` it renders a hardcoded untrained stand-in sized for `basic.wad`. Note: `policy.observation_space.shape` is channel-first `(C, H, W)` (SB3's `VecTransposeImage` already transposes it before the policy sees it) — the script used to assume `(H, W, C)` and crashed with a channel-mismatch error on any real `--model`; fixed to read it as `(n_stack, height, width)` directly. Requires `matplotlib` + `visualtorch` (both in `requirements.txt`).
- `visualize_cnn_diagnostics.py` — Zeiler & Fergus (2014) style diagnostics for the same `CnnPolicy`, sibling to `visualize_PPO_model.py` (same `--model` convention, lazy-imports SB3/gymnasium/vizdoom so the zero-argument path only needs torch+matplotlib): `DiagnosticPolicy` wraps a model's real `pi_features_extractor.cnn`/`.linear`, `action_net`, `value_net` layers directly (verified via `stable_baselines3` 2.9.0 against `models/latest/ppo_basic.zip`: `share_features_extractor=True` and `net_arch=[]` are both SB3's `CnnPolicy` defaults here, so `pi_features_extractor is vf_features_extractor` and `mlp_extractor.policy_net`/`.value_net` are empty `nn.Sequential()`s — both heads read the same 512-d features with no hidden MLP). Three techniques: deconvnet reconstruction (`deconv_reconstruct`/`deconv_topk` — rectify + `ConvTranspose2d` sharing each conv's own weight per layer, no unpooling since this net has no max-pooling to invert; `--channel` auto-picks the strongest-activation channel via `strongest_channel` since a fixed index is often dead/all-zero for a given frame), occlusion sensitivity (`occlusion_sensitivity` — one batched forward pass per slid gray patch, probability/value *drop* per position), and saliency/guided backprop (`saliency_map` — guided backprop's ReLU hook is `grad_input[0] * (grad_output[0] > 0)`, not `relu(grad_output)`, since the latter drops the forward-positivity mask autograd already encodes in `grad_input`). `--env {basic,deadly_corridor}` grabs live frames through the project's own `DummyVecEnv`+`VecFrameStack` pipeline rather than reimplementing frame-stacking order by hand. Outputs go to `--out-dir` (default `cnn_diagnostics_out/`, gitignored); four representative PNGs from a run against the real `ppo_basic.zip` are checked into the repo root (`cnn_diagnostics_{occlusion,deconv,saliency,guided}_basic.png`) and embedded in README's "CNN diagnostics" section.
- `models/latest/` — the single actively-trained model per scenario (`ppo_basic.zip`, `ppo_deadly_corridor_shaped.zip`, `ppo_defend_the_center.zip`, `ppo_defend_the_line.zip`, ..., `ppo_doom_E1M1.zip` — one per scenario/map, named after its scenario key), overwritten in place roughly every 10k timesteps and again at the end of `model.learn()`. This is what auto-resume, the `watch_agent_*.py` scripts, and `export_model.py` read, and what `import_model.py` writes.
- `models/backups/` — timestamped copies of whatever `import_model.py` replaced, one per import. Restore by copying back over the `models/latest/` file.
- `exports/` — default destination for `export_model.py` when `--out` isn't given. Gitignored-in-spirit like models; exported files are self-contained and safe to move/share.
- `wads/` — optional commercial `doom.wad`/`doom2.wad` for full-level training (see `wads/README.md`); auto-detected by `envs/doom_level_env.py`, Freedoom bundled with the vizdoom package is the fallback. Don't commit WADs.
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

CPU: AMD Ryzen 7 5800H, 8 physical cores / 16 logical (SMT) — relevant because ViZDoom's engine step is CPU-bound, single-threaded work, so `SubprocVecEnv` parallelism is capped more by physical cores than logical ones. All training runs use `train_common.N_ENVS = 12`; `N_ENVS = 14` was tried and hit a startup race (all 14 game engines booting simultaneously left one worker half-initialized), so 12 is the current throughput/stability sweet spot on this machine.

GPU: NVIDIA RTX 3060 (6GB), CUDA build of `torch` matches it — `nvidia-smi` confirmed working. Training scripts default to `device="cuda"`. Note: for `basic.wad`'s tiny CNN, the GPU is not the bottleneck — env stepping (CPU) is.

## Performance tuning already applied (don't redo, do reconsider if scaling to harder scenarios)

- `SubprocVecEnv` instead of `DummyVecEnv` (the `make_vec_env` default) — `DummyVecEnv` steps all envs sequentially in one process, leaving most cores idle.
- `frame_skip=4` instead of the ViZDoom default of 1 — act once every 4 engine ticks.
- Native render resolution forced to `RES_160X120` (down from `basic.wad`'s default `RES_320X240`) since the pipeline resizes to 84×84 anyway.
- Frame-stacking moved from per-env (`gymnasium.wrappers.FrameStackObservation`) to vec-env level (`stable_baselines3...VecFrameStack`) — each `SubprocVecEnv` worker now ships one `(84,84,1)` frame across its process pipe per step instead of a full `(84,84,4)` stack, a 4x cut in inter-process payload.

## Known gotchas (already fixed — read before re-debugging these from scratch)

- **`viz_instance_id is write protected`**: multiple `SubprocVecEnv` workers launching at once all defaulted to the same `_vizdoom.ini` in the working directory and raced on a ZDoom cvar used for shared-memory IPC naming. Fixed via `doom_config_path` set to a unique, PID-named file per worker in `envs/basic_env.py`.
- **`gymnasium.wrappers.ResizeObservation` silently drops a size-1 channel dimension.** It declares output shape `(84, 84, 1)` but internally calls `cv2.resize`, which returns `(84, 84)` for single-channel input — the declared and actual shapes disagree, and `VecFrameStack` breaks on the mismatch (`ValueError: could not broadcast ... into shape (3,84,84,4)` from a 3-env smoke test). Fixed by resizing with `keep_dim=False` (plain 2D array, no declared/actual mismatch), then explicitly restoring the channel dim with `gymnasium.wrappers.ReshapeObservation(env, (84, 84, 1))`.
- **Some scenario cfgs declare `screen_format = GRAY8`** (`rocket_basic.cfg`, `simpler_basic.cfg`), which the gymnasium wrapper honors — the screen buffer arrives single-channel and `GrayscaleObservation` (which expects 3 channels) would crash. `make_raw_vizdoom_env` forces `screen_format=RGB24` at `gym.make` time for every scenario; this is a no-op for the others (their `CRCGCB`/default formats were already being coerced to RGB24 by the wrapper).
- **The full-game cfgs (`doom.cfg` etc.) enable the audio buffer**, and ViZDoom's audio needs a working OpenAL device — a missing/broken one fails at `DoomGame.init()`. `envs/doom_level_env.py` disables audio (and automap) buffers explicitly; only `basic_audio` intentionally keeps audio on.

## Next steps (in order)

1. ~~Run `train_basic.py` end-to-end and confirm episode reward trends upward in TensorBoard~~ — done, agent performs well on `basic.wad`.
2. ~~Run `train_deadly_corridor.py` end-to-end~~ — done, baseline (`ppo_deadly_corridor` prefix) reached step 950,000 with `ep_rew_mean` climbing from -88 to +151. Reward-shaped version (`kill_reward_bonus` + `exploration_bonus_per_cell`, `ppo_deadly_corridor_shaped` prefix, warm-started from that baseline) is up and running.
3. ~~Implement `defend_the_center.wad`~~ — done (`envs/defend_the_center_env.py`, `train_defend_the_center.py`, `watch_agent_defend_the_center.py`, wired into `train_ui.py`). No earlier baseline to warm-start from, so its first run trains a fresh `CnnPolicy`. Next: run it end-to-end and confirm `ep_rew_mean` trends upward.
4. ~~Implement the remaining single-player scenarios~~ — done, all of them (see roadmap below): `defend_the_line`, `health_gathering`, `health_gathering_supreme`, `my_way_home`, `predict_position`, `take_cover`, plus the basic variants `simpler_basic`, `rocket_basic`, `basic_audio`, and full DOOM/DOOM II levels via `train_doom_level.py`. **None of the new scenarios has completed a training run yet** — next action is running them end-to-end (suggested order: `defend_the_line` → `health_gathering` → `health_gathering_supreme` → `my_way_home` → `take_cover` → `predict_position` → basic variants → `doom_E1M1`) and confirming `ep_rew_mean` trends upward per scenario. `basic_audio` additionally needs its OpenAL dependency confirmed on this machine at all.
5. Tune reward shaping per scenario as those runs come back (the defaults in each `envs/*_env.py` are reasoned starting points, not validated values — `predict_position`'s 100/25 and the full-level defaults especially). This matters more than architecture size.
6. After a working low-level controller exists, optionally revisit Option B (LLM high-level planner) — see below. Full-level training (`doom_E1M1`+) is the natural place for it: long horizons and navigation goals are exactly what a planner adds over a reactive CNN.

## Scenario roadmap

**All single-player scenarios are now implemented**, one `envs/*_env.py` + `train_*.py` + `watch_agent_*.py` triple each: `basic`, `simpler_basic`, `rocket_basic`, `basic_audio`, `deadly_corridor`, `defend_the_center`, `defend_the_line`, `health_gathering`, `health_gathering_supreme`, `my_way_home`, `predict_position`, `take_cover` — plus full DOOM/DOOM II levels (any `E?M?`/`MAP??`) through the shared `envs/doom_level_env.py` + `train_doom_level.py --map`.

Training status: `basic` and `deadly_corridor` have successful completed runs; `defend_the_center` is implemented and running; everything newer is implemented but untrained (see Next steps).

Not implemented: `basic_notifications` (adds an in-game text notification buffer to the observation — no RL value over basic without an encoder for the text), `learning.cfg`/`oblige.cfg` (config templates, not scenarios). Out of scope (multiplayer, needs a bot/opponent process rather than a single trained agent): `deathmatch`, `cig`, `cig_with_unknown`, `multi_duel`, `multi_deathmatch`.

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
