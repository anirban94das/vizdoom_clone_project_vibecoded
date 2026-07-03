"""Watch the agent play deadly_corridor.wad live using the most recent checkpoint.

Mirrors watch_agent.py, pointed at the "ppo_deadly_corridor" checkpoint
prefix written by train_deadly_corridor.py. Run this alongside
train_deadly_corridor.py (separate terminal, same venv) to see behavior
update as training saves new checkpoints.

Uses the same DummyVecEnv + VecFrameStack stacking as train_deadly_corridor.py
so the observation shape matches what each checkpoint was trained on.
"""

import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from envs.deadly_corridor_env import make_deadly_corridor_env

CHECKPOINT_DIR = Path("models/checkpoints")
CHECKPOINT_PREFIX = "ppo_deadly_corridor"


def latest_checkpoint() -> Path:
    checkpoints = sorted(
        CHECKPOINT_DIR.glob(f"{CHECKPOINT_PREFIX}_*_steps.zip"),
        key=lambda p: p.stat().st_mtime,
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {CHECKPOINT_DIR} yet — wait for training "
            "to hit its first save_freq interval."
        )
    return checkpoints[-1]


def main() -> None:
    vec_env = DummyVecEnv([lambda: make_deadly_corridor_env(render_mode="human")])
    vec_env = VecFrameStack(vec_env, n_stack=4)

    while True:
        checkpoint = latest_checkpoint()
        print(f"Loading {checkpoint.name}")
        model = PPO.load(checkpoint, env=vec_env, device="cuda")

        obs = vec_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = vec_env.step(action)
            done = dones[0]
            time.sleep(1 / 35)  # ViZDoom's native tic rate, for human-watchable speed


if __name__ == "__main__":
    main()
