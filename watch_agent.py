"""Watch the agent play basic.wad live using the latest saved model.

Run this alongside train_basic.py (separate terminal, same venv). It reloads
models/latest/ppo_basic.zip before every episode — since train_basic.py
overwrites that same file on a schedule, you'll see the agent's behavior
update every few episodes as training progresses.

Uses the same DummyVecEnv + VecFrameStack stacking as train_basic.py so the
observation shape matches what the model was trained on.
"""

import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from envs.basic_env import make_basic_env

MODEL_PATH = Path("models/latest/ppo_basic.zip")


def main() -> None:
    vec_env = DummyVecEnv([lambda: make_basic_env(render_mode="human")])
    vec_env = VecFrameStack(vec_env, n_stack=4)

    while True:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"{MODEL_PATH} not found yet — wait for training to hit its "
                "first save_freq interval."
            )
        print(f"Loading {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=vec_env, device="cuda")

        obs = vec_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = vec_env.step(action)
            done = dones[0]
            time.sleep(1 / 35)  # ViZDoom's native tic rate, for human-watchable speed


if __name__ == "__main__":
    main()
