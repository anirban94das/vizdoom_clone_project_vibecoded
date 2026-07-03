"""Watch the agent play basic.wad live using the most recent checkpoint.

Run this alongside train_basic.py (separate terminal, same venv). It loads
whichever checkpoint in models/checkpoints/ is newest, plays one episode in a
visible window, then reloads before the next episode — so as training saves
new checkpoints, you'll see the agent's behavior update every few episodes.

Uses the same DummyVecEnv + VecFrameStack stacking as train_basic.py so the
observation shape matches what each checkpoint was trained on.
"""

import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from envs.basic_env import make_basic_env

CHECKPOINT_DIR = Path("models/checkpoints")


def latest_checkpoint() -> Path:
    checkpoints = sorted(
        CHECKPOINT_DIR.glob("ppo_basic_*_steps.zip"),
        key=lambda p: p.stat().st_mtime,
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoints found in {CHECKPOINT_DIR} yet — wait for training "
            "to hit its first save_freq interval."
        )
    return checkpoints[-1]


def main() -> None:
    vec_env = DummyVecEnv([lambda: make_basic_env(render_mode="human")])
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
