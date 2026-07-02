"""Train a PPO agent on ViZDoom's deadly_corridor.wad scenario.

Mirrors train_basic.py's structure and auto-resume behavior, pointed at the
harder deadly_corridor scenario (envs.deadly_corridor_env). Checkpoints share
models/checkpoints/ with the basic.wad run, distinguished by the
"ppo_deadly_corridor" filename prefix; TensorBoard logs share logs/tensorboard
under a separate run name so both scenarios are comparable side by side.

Do not run this alongside train_basic.py — each spawns N_ENVS=8
SubprocVecEnv workers, and this machine has 8 physical cores, so running both
at once oversubscribes and slows both down.
"""

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
N_ENVS = 8
CHECKPOINT_DIR = Path("models/checkpoints")
CHECKPOINT_PREFIX = "ppo_deadly_corridor"


def find_latest_checkpoint() -> Path | None:
    """Return the checkpoint with the highest step count, or None if empty."""
    checkpoints = list(CHECKPOINT_DIR.glob(f"{CHECKPOINT_PREFIX}_*_steps.zip"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(re.search(r"_(\d+)_steps", p.stem).group(1)))


def main() -> None:
    # SubprocVecEnv runs each ViZDoom instance in its own process. ViZDoom's
    # engine step is CPU-bound (software rendering), so DummyVecEnv's
    # single-process/sequential stepping left most cores idle.
    vec_env = make_vec_env(make_deadly_corridor_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    vec_env = VecFrameStack(vec_env, n_stack=4)

    # save_freq is per-env steps; the callback fires every N_ENVS actual
    # timesteps, so this saves roughly every 10_000 real timesteps.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // N_ENVS, 1),
        save_path=str(CHECKPOINT_DIR),
        name_prefix=CHECKPOINT_PREFIX,
    )

    latest_checkpoint = find_latest_checkpoint()
    if latest_checkpoint is not None:
        print(f"Resuming from checkpoint: {latest_checkpoint}")
        model = PPO.load(
            latest_checkpoint,
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
        tb_log_name=CHECKPOINT_PREFIX,
        callback=checkpoint_callback,
        reset_num_timesteps=latest_checkpoint is None,
    )
    model.save("models/ppo_deadly_corridor")


if __name__ == "__main__":
    main()
