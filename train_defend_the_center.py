"""Train a PPO agent on ViZDoom's defend_the_center.wad scenario.

Mirrors train_basic.py's / train_deadly_corridor.py's structure and
auto-resume behavior. Unlike deadly_corridor, this scenario has no earlier
unshaped baseline to warm-start from — it's a from-scratch run the first time
models/latest/ppo_defend_the_center.zip doesn't exist.

The player is fixed at the center of the room (only TURN_LEFT/TURN_RIGHT/
ATTACK are available — see defend_the_center.cfg), so exploration_bonus_per_cell
defaults to 0.0 and should stay off: standing still is the objective here, not
a failure mode to discourage like in deadly_corridor.

Do not run this alongside train_basic.py or train_deadly_corridor.py — each
spawns N_ENVS SubprocVecEnv workers, and this machine has 8 physical cores,
so running more than one at once oversubscribes and slows all of them down.
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

from envs.defend_the_center_env import make_defend_the_center_env
from training_utils import EpisodeRecapCallback, OverwriteCheckpointCallback

# Shorter, more constrained scenario than deadly_corridor (episode_timeout is
# 2100 tics / ~60s of real time at frame_skip=4) but still sparser-reward than
# basic.wad's single point-blank shot, so starting at the same order of
# magnitude as deadly_corridor and relying on auto-resume for more if needed.
TOTAL_TIMESTEPS = 300_000
# Same physical-core reasoning as the other train_*.py scripts (unchanged hardware).
N_ENVS = 12
# Single overwritten file, not a growing set of step-numbered checkpoints —
# this is both the resume point and the only saved copy of this scenario's model.
MODEL_PATH = Path("models/latest/ppo_defend_the_center.zip")


def parse_args() -> argparse.Namespace:
    """Reward-shaping bonuses, defaulting to this scenario's existing values —
    pass flags to override for experimentation. Overriding these doesn't
    invalidate an existing checkpoint; it just changes the reward the resumed
    agent trains against going forward."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kill-reward-bonus", type=float, default=20.0)
    parser.add_argument("--hit-reward-bonus", type=float, default=5.0)
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
        make_defend_the_center_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs
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
        scenario="ppo_defend_the_center",
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
        tb_log_name="ppo_defend_the_center",
        callback=[checkpoint_callback, recap_callback],
        reset_num_timesteps=not resuming,
    )
    model.save(MODEL_PATH)


if __name__ == "__main__":
    main()
