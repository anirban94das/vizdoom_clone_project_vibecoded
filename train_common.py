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

--ent-coef / --target-kl (the PPO policy-collapse guards) are flags on every
scenario. --target-kl now defaults to 0.03 everywhere (build_parser), not
just on deadly_corridor: after switching to PPO_HYPERPARAMS's smaller
n_steps=128 rollout below, basic.wad hit the same collapse deadly_corridor
did originally (approx_kl spiked past 1.0, ep_rew_mean crashed from healthy
positive values to a pinned -300 and never recovered, reproduced twice under
a fixed seed) - the smaller, noisier rollout makes a single runaway update
more likely across every scenario, not just deadly_corridor, so every
scenario now gets the same 0.03 KL circuit breaker by default. --ent-coef
stays 0.0 by default (deadly_corridor's 0.01 floor-of-exploration override
is scenario-specific, still set explicitly in its own script).

PPO_HYPERPARAMS below (future-enhancements item 3, "Atari-style PPO
hyperparameters") applies to every scenario the same way: SB3's own
defaults (n_steps=2048, batch_size=64, n_epochs=10, clip_range=0.2, constant
learning_rate=3e-4) were tuned for MuJoCo-style low-dimensional
continuous-control tasks. Every scenario here is pixel input -> CNN ->
discrete actions instead - architecturally the same as Atari, not MuJoCo -
so this borrows the hyperparameter recipe from the original PPO paper /
OpenAI Baselines' Atari config (the same one stable-baselines3-zoo's
atari.yml uses) rather than SB3's generic defaults.
"""

import argparse
import time
from pathlib import Path
from typing import Callable

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

from training_utils import EpisodeRecapCallback, OverwriteCheckpointCallback, UnshapedEvalCallback

# Same physical-core reasoning as always: 8 physical cores on this machine,
# N_ENVS=14 hit a startup race (one worker half-initialized), 12 is the
# throughput/stability sweet spot.
N_ENVS = 12


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linearly anneals from initial_value at the start of training to 0.0 at
    the end. SB3 accepts a callable for `learning_rate` and calls it with
    progress_remaining going from 1.0 (start) to 0.0 (end) - the Atari-recipe
    learning-rate schedule below uses this instead of a constant rate."""
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return schedule


# Atari-style PPO hyperparameters (see the module docstring for why "Atari"
# is the right recipe to borrow here, not SB3's MuJoCo-tuned defaults). With
# N_ENVS=12 fixed, SB3's own n_steps=2048 default meant a 24,576-step
# rollout collected under a frozen policy before a single gradient update -
# these values refresh the policy roughly 16x more often (128*12=1,536 steps
# per rollout) with each update trained on less stale (closer-to-on-policy)
# data, and the tighter clip_range adds a second brake on top of that.
# Applies to every scenario via ppo_overrides below - no per-scenario change
# needed. Safe to apply on top of an existing checkpoint: these are training
# hyperparameters, not part of the saved network's architecture.
PPO_HYPERPARAMS: dict[str, object] = dict(
    n_steps=128,
    batch_size=256,
    n_epochs=4,
    clip_range=0.1,
    learning_rate=linear_schedule(2.5e-4),
)

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
    target_kl: float | None = 0.03,
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


# All nine reward knobs forced to 0.0, except exploration_cell_size (a
# bucket-size scale, not a bonus - zeroing it would divide-by-zero in
# EpisodeStatsWrapper._cell()). Overlaying this on any scenario's env_kwargs
# yields the scenario's built-in, unshaped reward - used by
# UnshapedEvalCallback below and reused as-is by ablation.py.
ZERO_SHAPING_KWARGS: dict[str, float] = {
    key: 0.0 for key, _flag in REWARD_KNOB_FLAGS if key != "exploration_cell_size"
}


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
    eval_freq: int = 50_000,
    n_eval_episodes: int = 5,
    seed: int | None = None,
    history_path: str | Path = "logs/training_history.jsonl",
) -> None:
    """The whole training run: build the vec env, resume/warm-start/create the
    model, learn for total_timesteps ADDITIONAL steps, save. `scenario` names
    the TensorBoard run, the recap history line, and nothing else.

    eval_freq/n_eval_episodes control UnshapedEvalCallback (see
    training_utils.py): every eval_freq real timesteps it plays
    n_eval_episodes deterministic episodes with every reward-shaping knob at
    0.0, so there's always a shaping-invariant score to compare runs by.
    seed and history_path exist mainly for ablation.py, which needs
    reproducible short runs and a history file separate from real training's.
    """
    model_path = Path(model_path)
    history_path = Path(history_path)
    print(f"Reward shaping: {env_kwargs}")
    print(f"PPO stability guards: ent_coef={args.ent_coef}, target_kl={args.target_kl}")
    print(f"PPO hyperparameters (Atari-style): n_steps={PPO_HYPERPARAMS['n_steps']}, "
          f"batch_size={PPO_HYPERPARAMS['batch_size']}, n_epochs={PPO_HYPERPARAMS['n_epochs']}, "
          f"clip_range={PPO_HYPERPARAMS['clip_range']}")

    # SubprocVecEnv runs each ViZDoom instance in its own process. ViZDoom's
    # engine step is CPU-bound (software rendering), so DummyVecEnv's
    # single-process/sequential stepping left most cores idle.
    vec_env = make_vec_env(
        make_env_fn, n_envs=N_ENVS, seed=seed, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs
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
        history_path=history_path,
    )
    eval_callback = UnshapedEvalCallback(
        scenario=scenario,
        make_env_fn=make_env_fn,
        unshaped_env_kwargs={**env_kwargs, **ZERO_SHAPING_KWARGS},
        history_path=history_path,
        eval_freq=max(eval_freq // N_ENVS, 1),
        n_eval_episodes=n_eval_episodes,
        n_stack=n_stack,
        verbose=1,
    )

    ppo_overrides = dict(
        ent_coef=args.ent_coef, target_kl=args.target_kl, seed=seed, **PPO_HYPERPARAMS
    )
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
        callback=[checkpoint_callback, recap_callback, eval_callback],
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
