"""Watch the agent play deadly_corridor.wad live using the latest saved model.

Mirrors watch_agent.py, pointed at models/latest/ppo_deadly_corridor_shaped.zip
— the file train_deadly_corridor.py's reward-shaped run overwrites on a
schedule. Run this alongside train_deadly_corridor.py (separate terminal,
same venv) to see behavior update as training progresses.

Uses the same DummyVecEnv + VecFrameStack stacking as train_deadly_corridor.py
so the observation shape matches what the model was trained on.
"""

import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from envs.deadly_corridor_env import make_deadly_corridor_env

MODEL_PATH = Path("models/latest/ppo_deadly_corridor_shaped.zip")


def main() -> None:
    vec_env = DummyVecEnv([lambda: make_deadly_corridor_env(render_mode="human")])
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
