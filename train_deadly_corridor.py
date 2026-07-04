"""Train a PPO agent on ViZDoom's deadly_corridor.wad scenario.

Mirrors train_basic.py's structure and auto-resume behavior, pointed at the
harder deadly_corridor scenario (envs.deadly_corridor_env), which now
enables reward shaping by default (kill_reward_bonus + exploration bonus —
see envs/deadly_corridor_env.py). Because the reward function changed from
the earlier "ppo_deadly_corridor" baseline run, this checkpoints/logs under
a new "ppo_deadly_corridor_shaped" identity rather than silently overwriting
that run's history, but still warm-starts from its final weights (below) so
the already-learned visual features/aiming/movement aren't thrown away.

Do not run this alongside train_basic.py — each spawns N_ENVS SubprocVecEnv
workers, and this machine has 8 physical cores, so running both at once
oversubscribes and slows both down.
"""

import argparse
import re
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

from envs.deadly_corridor_env import make_deadly_corridor_env

# deadly_corridor is a harder, sparser-reward scenario (doom_skill=5, must
# navigate under fire) than basic.wad's ~100k-step convergence. Starting
# higher and relying on auto-resume (below) to add more later if needed.
TOTAL_TIMESTEPS = 300_000
# Same physical-core reasoning as train_basic.py (unchanged hardware).
N_ENVS = 12
CHECKPOINT_DIR = Path("models/checkpoints")
CHECKPOINT_PREFIX = "ppo_deadly_corridor_shaped"
# Pre-reward-shaping baseline run (step 950,000 as of this change) — used
# only as a one-time warm start if no shaped-reward checkpoint exists yet.
WARM_START_PREFIX = "ppo_deadly_corridor"


def find_latest_checkpoint(prefix: str) -> Path | None:
    """Return the checkpoint with the highest step count for prefix, or None."""
    checkpoints = list(CHECKPOINT_DIR.glob(f"{prefix}_*_steps.zip"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(re.search(r"_(\d+)_steps", p.stem).group(1)))


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
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // N_ENVS, 1),
        save_path=str(CHECKPOINT_DIR),
        name_prefix=CHECKPOINT_PREFIX,
    )

    latest_shaped = find_latest_checkpoint(CHECKPOINT_PREFIX)
    if latest_shaped is not None:
        print(f"Resuming shaped-reward run from checkpoint: {latest_shaped}")
        model = PPO.load(
            latest_shaped,
            env=vec_env,
            device="cuda",
            tensorboard_log="logs/tensorboard",
        )
        reset_num_timesteps = False
    else:
        warm_start = find_latest_checkpoint(WARM_START_PREFIX)
        if warm_start is not None:
            # Weights carry over (visual features/aiming/movement), but the
            # reward function underneath has changed, so timesteps/TensorBoard
            # logging start fresh under CHECKPOINT_PREFIX — expect a visible
            # jump/dip in the reward curve right at this handoff.
            print(f"Warm-starting from pre-shaping checkpoint: {warm_start}")
            model = PPO.load(
                warm_start,
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
        reset_num_timesteps = True

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        tb_log_name=CHECKPOINT_PREFIX,
        callback=checkpoint_callback,
        reset_num_timesteps=reset_num_timesteps,
    )
    model.save("models/ppo_deadly_corridor_shaped")


if __name__ == "__main__":
    main()
