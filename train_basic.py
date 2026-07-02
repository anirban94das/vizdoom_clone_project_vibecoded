"""Train a PPO agent on ViZDoom's basic.wad scenario.

First end-to-end training run for this project. Uses stable-baselines3's
CnnPolicy on a (84, 84, 4) frame stack: envs.basic_env.make_basic_env
produces single (84, 84, 1) grayscale frames, and VecFrameStack stacks 4 of
them together after they cross the SubprocVecEnv process pipe (cheaper than
stacking before, which would ship 4x the bytes per step). Progress is logged
to TensorBoard under logs/tensorboard (`tensorboard --logdir logs/tensorboard`).
"""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack

from envs.basic_env import make_basic_env

TOTAL_TIMESTEPS = 100_000
# This machine has 8 physical cores / 16 logical (SMT). ViZDoom's engine step
# is single-threaded CPU work, so throughput scales with physical cores more
# than logical ones. N_ENVS=14 hit a startup race (all 14 game engines
# booting simultaneously left one worker half-initialized) - 8, matching
# physical cores, is both the throughput sweet spot and safer to boot.
N_ENVS = 8


def main() -> None:
    # SubprocVecEnv runs each ViZDoom instance in its own process. ViZDoom's
    # engine step is CPU-bound (software rendering), so DummyVecEnv's
    # single-process/sequential stepping left most cores idle.
    vec_env = make_vec_env(make_basic_env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
    vec_env = VecFrameStack(vec_env, n_stack=4)

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
