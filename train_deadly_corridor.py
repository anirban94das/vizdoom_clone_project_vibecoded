"""Train a PPO agent on ViZDoom's deadly_corridor.wad scenario.

Mirrors train_basic.py's structure and auto-resume behavior, pointed at the
harder deadly_corridor scenario (envs.deadly_corridor_env), which now
enables reward shaping by default (kill_reward_bonus + exploration bonus —
see envs/deadly_corridor_env.py). Because the reward function changed from
the earlier "ppo_deadly_corridor" baseline run, this trains/logs under a new
"ppo_deadly_corridor_shaped" identity rather than silently overwriting that
run's history, but still warm-starts from its final weights (below) so the
already-learned visual features/aiming/movement aren't thrown away.

Do not run this alongside train_basic.py — each spawns N_ENVS SubprocVecEnv
workers, and this machine has 8 physical cores, so running both at once
oversubscribes and slows both down.
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

from envs.deadly_corridor_env import make_deadly_corridor_env
from training_utils import EpisodeRecapCallback, OverwriteCheckpointCallback

# deadly_corridor is a harder, sparser-reward scenario (doom_skill=5, must
# navigate under fire) than basic.wad's ~100k-step convergence. Starting
# higher and relying on auto-resume (below) to add more later if needed.
TOTAL_TIMESTEPS = 300_000
# Same physical-core reasoning as train_basic.py (unchanged hardware).
N_ENVS = 12
# Single overwritten file, not a growing set of step-numbered checkpoints —
# this is both the resume point and the only saved copy of this scenario's model.
MODEL_PATH = Path("models/latest/ppo_deadly_corridor_shaped.zip")
# Pre-reward-shaping baseline run's final saved model — used only as a
# one-time warm start if MODEL_PATH doesn't exist yet.
WARM_START_PATH = Path("models/ppo_deadly_corridor.zip")


def parse_args() -> argparse.Namespace:
    """Reward-shaping bonuses, defaulting to this scenario's existing values —
    pass flags to override for experimentation. Note: overriding these doesn't
    invalidate an existing checkpoint (see module docstring on warm-starting);
    it just changes the reward the resumed agent trains against going forward."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--kill-reward-bonus", type=float, default=20.0)
    parser.add_argument("--hit-reward-bonus", type=float, default=5.0)
    parser.add_argument("--exploration-bonus-per-cell", type=float, default=1.0)
    parser.add_argument("--exploration-cell-size", type=float, default=32.0)
    parser.add_argument("--weapon-pickup-bonus", type=float, default=15.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_kwargs = dict(
        kill_reward_bonus=args.kill_reward_bonus,
        hit_reward_bonus=args.hit_reward_bonus,
        exploration_bonus_per_cell=args.exploration_bonus_per_cell,
        exploration_cell_size=args.exploration_cell_size,
        weapon_pickup_bonus=args.weapon_pickup_bonus,
    )
    print(f"Reward shaping: {env_kwargs}")

    # SubprocVecEnv runs each ViZDoom instance in its own process. ViZDoom's
    # engine step is CPU-bound (software rendering), so DummyVecEnv's
    # single-process/sequential stepping left most cores idle.
    vec_env = make_vec_env(
        make_deadly_corridor_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs
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
        scenario="ppo_deadly_corridor_shaped",
        history_path=Path("logs/training_history.jsonl"),
    )

    if MODEL_PATH.exists():
        print(f"Resuming shaped-reward run from: {MODEL_PATH}")
        model = PPO.load(
            MODEL_PATH,
            env=vec_env,
            device="cuda",
            tensorboard_log="logs/tensorboard",
        )
        reset_num_timesteps = False
    elif WARM_START_PATH.exists():
        # Weights carry over (visual features/aiming/movement), but the
        # reward function underneath has changed, so timesteps/TensorBoard
        # logging start fresh — expect a visible jump/dip in the reward
        # curve right at this handoff.
        print(f"Warm-starting from pre-shaping model: {WARM_START_PATH}")
        model = PPO.load(
            WARM_START_PATH,
            env=vec_env,
            device="cuda",
            tensorboard_log="logs/tensorboard",
        )
        reset_num_timesteps = True
    else:
        model = PPO(
            "CnnPolicy",
            vec_env,
            verbose=1,
            tensorboard_log="logs/tensorboard",
            device="cuda",
        )
        reset_num_timesteps = True

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name="ppo_deadly_corridor_shaped",
        callback=[checkpoint_callback, recap_callback],
        reset_num_timesteps=reset_num_timesteps,
    )
    model.save(MODEL_PATH)


if __name__ == "__main__":
    main()
