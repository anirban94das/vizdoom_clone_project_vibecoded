"""Train a PPO agent on ViZDoom's basic.wad scenario.

First end-to-end training run for this project. Uses stable-baselines3's
CnnPolicy on a (84, 84, 4) frame stack: envs.basic_env.make_basic_env
produces single (84, 84, 1) grayscale frames, and VecFrameStack stacks 4 of
them together after they cross the SubprocVecEnv process pipe (cheaper than
stacking before, which would ship 4x the bytes per step). Progress is logged
to TensorBoard under logs/tensorboard (`tensorboard --logdir logs/tensorboard`).
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

from envs.basic_env import make_basic_env
from training_utils import EpisodeRecapCallback, OverwriteCheckpointCallback

TOTAL_TIMESTEPS = 100_000
# This machine has 8 physical cores / 16 logical (SMT). ViZDoom's engine step
# is single-threaded CPU work, so throughput scales with physical cores more
# than logical ones. N_ENVS=14 hit a startup race (all 14 game engines
# booting simultaneously left one worker half-initialized) - 8, matching
# physical cores, is both the throughput sweet spot and safer to boot.
N_ENVS = 12
# Single overwritten file, not a growing set of step-numbered checkpoints —
# this is both the resume point and the only saved copy of this scenario's model.
MODEL_PATH = Path("models/latest/ppo_basic.zip")


def parse_args() -> argparse.Namespace:
    """Reward-shaping bonuses, all off by default (matching basic.wad's
    original unshaped reward) — pass flags to opt in for experimentation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kill-reward-bonus", type=float, default=0.0)
    parser.add_argument("--hit-reward-bonus", type=float, default=0.0)
    parser.add_argument("--exploration-bonus-per-cell", type=float, default=0.0)
    parser.add_argument("--exploration-cell-size", type=float, default=32.0)
    parser.add_argument("--weapon-pickup-bonus", type=float, default=0.0)
    parser.add_argument("--damage-dealt-bonus", type=float, default=0.0)
    parser.add_argument("--damage-taken-penalty", type=float, default=0.0)
    parser.add_argument("--health-change-bonus", type=float, default=0.0)
    parser.add_argument("--armor-change-bonus", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_kwargs = dict(
        kill_reward_bonus=args.kill_reward_bonus,
        hit_reward_bonus=args.hit_reward_bonus,
        exploration_bonus_per_cell=args.exploration_bonus_per_cell,
        exploration_cell_size=args.exploration_cell_size,
        weapon_pickup_bonus=args.weapon_pickup_bonus,
        damage_dealt_bonus=args.damage_dealt_bonus,
        damage_taken_penalty=args.damage_taken_penalty,
        health_change_bonus=args.health_change_bonus,
        armor_change_bonus=args.armor_change_bonus,
    )
    print(f"Reward shaping: {env_kwargs}")

    # SubprocVecEnv runs each ViZDoom instance in its own process. ViZDoom's
    # engine step is CPU-bound (software rendering), so DummyVecEnv's
    # single-process/sequential stepping left most cores idle.
    vec_env = make_vec_env(
        make_basic_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs
    )
    vec_env = VecFrameStack(vec_env, n_stack=4)

    # save_freq is per-env steps; the callback fires every N_ENVS actual
    # timesteps, so this saves roughly every 10_000 real timesteps.
    checkpoint_callback = OverwriteCheckpointCallback(
        save_freq=max(10_000 // N_ENVS, 1),
        save_path=MODEL_PATH,
        verbose=1,
    )
    recap_callback = EpisodeRecapCallback(
        scenario="ppo_basic",
        history_path=Path("logs/training_history.jsonl"),
    )

    resuming = MODEL_PATH.exists()
    if resuming:
        print(f"Resuming from: {MODEL_PATH}")
        model = PPO.load(
            MODEL_PATH,
            env=vec_env,
            device="cuda",
            tensorboard_log="logs/tensorboard",
        )
    else:
        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=1,
            tensorboard_log="logs/tensorboard",
            device="cuda",
        )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name="ppo_basic",
        callback=[checkpoint_callback, recap_callback],
        reset_num_timesteps=not resuming,
    )
    model.save(MODEL_PATH)


if __name__ == "__main__":
    main()
