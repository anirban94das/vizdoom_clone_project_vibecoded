"""Watch the agent play defend_the_center.wad live using the latest saved model.

Mirrors watch_agent.py / watch_agent_deadly_corridor.py, pointed at
models/latest/ppo_defend_the_center.zip — the file train_defend_the_center.py
overwrites on a schedule. Run this alongside train_defend_the_center.py
(separate terminal, same venv) to see behavior update as training progresses.

Uses the same DummyVecEnv + VecFrameStack stacking as
train_defend_the_center.py so the observation shape matches what the model
was trained on.
"""

import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from envs.defend_the_center_env import make_defend_the_center_env

MODEL_PATH = Path("models/latest/ppo_defend_the_center.zip")


def main() -> None:
    vec_env = DummyVecEnv([lambda: make_defend_the_center_env(render_mode="human")])
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
