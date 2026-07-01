"""Train a PPO agent on ViZDoom's basic.wad scenario.

First end-to-end training run for this project. Uses stable-baselines3's
CnnPolicy directly on the preprocessed (4, 84, 84) frame stack produced by
envs.basic_env.make_basic_env. Progress is logged to TensorBoard under
logs/tensorboard (`tensorboard --logdir logs/tensorboard`).
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from envs.basic_env import make_basic_env

TOTAL_TIMESTEPS = 100_000
N_ENVS = 12


def main() -> None:
    # SubprocVecEnv runs each ViZDoom instance in its own process. ViZDoom's
    # engine step is CPU-bound (software rendering), so DummyVecEnv's
    # single-process/sequential stepping left most cores idle.
    vec_env = make_vec_env(make_basic_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

    # save_freq is per-env steps; the callback fires every N_ENVS actual
    # timesteps, so this saves roughly every 10_000 real timesteps.
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // N_ENVS, 1),
        save_path="models/checkpoints",
        name_prefix="ppo_basic",
    )

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
        callback=checkpoint_callback,
    )
    model.save("models/ppo_basic")


if __name__ == "__main__":
    main()
