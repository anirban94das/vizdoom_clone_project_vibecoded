"""Shared training/watching runner behind every train_*.py / watch_agent_*.py.

Extracted once the scenario count went from 3 to 14: train_basic.py,
train_deadly_corridor.py, and train_defend_the_center.py had near-identical
bodies (vec-env construction, callbacks, auto-resume/warm-start/fresh-start
decision, learn + save), so each train_*.py now just declares its scenario's
constants — env factory, model path, timesteps, reward-shaping defaults —
and calls run_training(). Behavior is intentionally identical to the
pre-refactor scripts:

- auto-resume from model_path if it exists (reset_num_timesteps=False),
- else a one-time warm start from warm_start_path if given and present
  (weights carry over, timestep/TensorBoard counters reset since the reward
  scale changed — expect a jump/dip in the reward curve at the handoff),
- else a fresh policy,
- OverwriteCheckpointCallback saving to one fixed file every ~10k timesteps,
- EpisodeRecapCallback appending one JSON line per run to
  logs/training_history.jsonl.

--ent-coef / --target-kl (the PPO policy-collapse guards introduced after the
deadly_corridor incident where reward crashed from +340 to -46,000) are now
flags on every scenario; defaults are per-scenario (SB3's own defaults of
0.0/None everywhere except deadly_corridor, which keeps its 0.01/0.03).
"""

import argparse
import time
from pathlib import Path
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

from training_utils import EpisodeRecapCallback, OverwriteCheckpointCallback

# Same physical-core reasoning as always: 8 physical cores on this machine,
# N_ENVS=14 hit a startup race (one worker half-initialized), 12 is the
# throughput/stability sweet spot.
N_ENVS = 12

# The nine reward-shaping knobs every scenario exposes, as
# (env-factory kwarg, CLI flag) pairs. train_ui.py passes all of them to
# every train_*.py, so every scenario's parser must accept all nine even if
# its defaults leave most at 0.0 (off).
REWARD_KNOB_FLAGS = [
    ("kill_reward_bonus", "--kill-reward-bonus"),
    ("hit_reward_bonus", "--hit-reward-bonus"),
    ("exploration_bonus_per_cell", "--exploration-bonus-per-cell"),
    ("exploration_cell_size", "--exploration-cell-size"),
    ("weapon_pickup_bonus", "--weapon-pickup-bonus"),
    ("damage_dealt_bonus", "--damage-dealt-bonus"),
    ("damage_taken_penalty", "--damage-taken-penalty"),
    ("health_change_bonus", "--health-change-bonus"),
    ("armor_change_bonus", "--armor-change-bonus"),
]


def build_parser(
    reward_defaults: dict[str, float],
    ent_coef: float = 0.0,
    target_kl: float | None = None,
) -> argparse.ArgumentParser:
    """Parser with the nine reward-shaping flags (defaults per scenario;
    anything not named in reward_defaults is off) plus the PPO stability
    guards. Scenario scripts that need extra flags (e.g. train_doom_level.py's
    --map/--skill) add them onto the returned parser."""
    parser = argparse.ArgumentParser()
    for key, flag in REWARD_KNOB_FLAGS:
        fallback = 32.0 if key == "exploration_cell_size" else 0.0
        parser.add_argument(flag, type=float, default=reward_defaults.get(key, fallback))
    parser.add_argument("--ent-coef", type=float, default=ent_coef)
    parser.add_argument("--target-kl", type=float, default=target_kl)
    return parser


def reward_kwargs_from_args(args: argparse.Namespace) -> dict[str, float]:
    """The nine knob values as env-factory kwargs."""
    return {key: getattr(args, key) for key, _flag in REWARD_KNOB_FLAGS}


def run_training(
    make_env_fn: Callable,
    env_kwargs: dict,
    scenario: str,
    model_path: str | Path,
    total_timesteps: int,
    args: argparse.Namespace,
    warm_start_path: str | Path | None = None,
    policy: str = "CnnPolicy",
    n_stack: int = 4,
) -> None:
    """The whole training run: build the vec env, resume/warm-start/create the
    model, learn for total_timesteps ADDITIONAL steps, save. `scenario` names
    the TensorBoard run, the recap history line, and nothing else."""
    model_path = Path(model_path)
    print(f"Reward shaping: {env_kwargs}")
    print(f"PPO stability guards: ent_coef={args.ent_coef}, target_kl={args.target_kl}")

    # SubprocVecEnv runs each ViZDoom instance in its own process. ViZDoom's
    # engine step is CPU-bound (software rendering), so DummyVecEnv's
    # single-process/sequential stepping left most cores idle.
    vec_env = make_vec_env(
        make_env_fn, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs
    )
    # Frame-stacking at the vec-env level (not per-env) so each worker ships
    # one (84,84,1) frame across its process pipe per step, not a full stack.
    # n_stack=1 skips stacking entirely (basic_audio: the audio buffer already
    # carries the temporal signal, and its dict observation doesn't stack).
    if n_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=n_stack)

    # save_freq is per-env steps; the callback fires every N_ENVS actual
    # timesteps, so this saves roughly every 10_000 real timesteps.
    checkpoint_callback = OverwriteCheckpointCallback(
        save_freq=max(10_000 // N_ENVS, 1),
        save_path=model_path,
        verbose=1,
    )
    recap_callback = EpisodeRecapCallback(
        scenario=scenario,
        history_path=Path("logs/training_history.jsonl"),
    )

    ppo_overrides = dict(ent_coef=args.ent_coef, target_kl=args.target_kl)
    if model_path.exists():
        print(f"Resuming from: {model_path}")
        model = PPO.load(
            model_path,
            env=vec_env,
            device="cuda",
            tensorboard_log="logs/tensorboard",
            **ppo_overrides,
        )
        reset_num_timesteps = False
    elif warm_start_path is not None and Path(warm_start_path).exists():
        # Weights carry over (visual features/aiming/movement), but the
        # reward function underneath differs, so timesteps/TensorBoard
        # logging start fresh.
        print(f"Warm-starting from: {warm_start_path}")
        model = PPO.load(
            warm_start_path,
            env=vec_env,
            device="cuda",
            tensorboard_log="logs/tensorboard",
            **ppo_overrides,
        )
        reset_num_timesteps = True
    else:
        model = PPO(
            policy,
            vec_env,
            verbose=1,
            tensorboard_log="logs/tensorboard",
            device="cuda",
            **ppo_overrides,
        )
        reset_num_timesteps = True

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=scenario,
        callback=[checkpoint_callback, recap_callback],
        reset_num_timesteps=reset_num_timesteps,
    )
    model.save(model_path)


def run_watch(
    make_env_fn: Callable,
    model_path: str | Path,
    env_kwargs: dict | None = None,
    n_stack: int = 4,
) -> None:
    """The whole watch loop: single visible env (DummyVecEnv, its own window),
    reload the scenario's fixed model file before every episode so behavior
    updates live while training overwrites that file in place."""
    model_path = Path(model_path)
    vec_env = DummyVecEnv([lambda: make_env_fn(render_mode="human", **(env_kwargs or {}))])
    if n_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=n_stack)

    while True:
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found yet — wait for training to hit its "
                "first save_freq interval."
            )
        print(f"Loading {model_path}")
        model = PPO.load(model_path, env=vec_env, device="cuda")

        obs = vec_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = vec_env.step(action)
            done = dones[0]
            time.sleep(1 / 35)  # ViZDoom's native tic rate, for human-watchable speed
