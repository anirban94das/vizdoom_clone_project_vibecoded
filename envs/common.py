"""Shared Gymnasium env factory for ViZDoom scenarios.

Preprocessing pipeline used by every per-scenario env module (envs/basic_env.py,
envs/deadly_corridor_env.py, ...): strips the Dict observation down to the raw
screen buffer, then grayscale -> resize to 84x84 -> explicit reshape back to
(84, 84, 1) so stable-baselines3's CnnPolicy gets a consistent channel-last
uint8 frame regardless of scenario. Frame-stacking is applied at the vec-env
level (see each train_*.py's VecFrameStack) rather than here, so each
SubprocVecEnv worker only ships one new frame across the process pipe per
step instead of a full 4-frame stack.
"""

import os
import random
import time
from pathlib import Path

import gymnasium as gym
import vizdoom as vzd
import vizdoom.gymnasium_wrapper  # noqa: F401  (registers Vizdoom* env ids)
from gymnasium.wrappers import GrayscaleObservation, ReshapeObservation, ResizeObservation

CONFIG_DIR = Path("configs")


class ScreenOnlyObservation(gym.ObservationWrapper):
    """Discards gamevariables and exposes only the raw screen buffer."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = env.observation_space["screen"]

    def observation(self, obs):
        return obs["screen"]


def make_vizdoom_env(
    env_id: str,
    render_mode: str | None = None,
    frame_skip: int = 4,
    screen_resolution: vzd.ScreenResolution = vzd.ScreenResolution.RES_160X120,
) -> gym.Env:
    # Each SubprocVecEnv worker is a separate OS process; give it its own
    # ZDoom config file so concurrent instances don't race on the shared
    # default _vizdoom.ini (which manifests as "viz_instance_id is write
    # protected" when multiple instances start at once).
    CONFIG_DIR.mkdir(exist_ok=True)
    doom_config_path = str(CONFIG_DIR / f"vizdoom_{os.getpid()}.ini")

    # SubprocVecEnv starts all workers at once, so every worker's DoomGame
    # boots in the same instant. Under heavy concurrency that race left one
    # worker's engine half-initialized (state stuck at None, "Call reset
    # before using step" on the first step). Jittering startup spreads the
    # engine boots out instead of hitting them all simultaneously.
    time.sleep(random.uniform(0, 2.0))

    env = gym.make(
        env_id,
        render_mode=render_mode,
        frame_skip=frame_skip,
        doom_config_path=doom_config_path,
        # Native render resolution: default is usually higher than 84x84, but
        # we resize down to 84x84 anyway, so render at the smallest ViZDoom
        # supports (still comfortably >84px) to cut the software rasterizer's
        # per-step cost in every parallel worker.
        screen_resolution=screen_resolution,
    )
    env = ScreenOnlyObservation(env)
    env = GrayscaleObservation(env, keep_dim=False)
    # ResizeObservation calls cv2.resize, which silently drops a size-1
    # channel dim (bug: it declares shape (84, 84, 1) but returns (84, 84)).
    # Resize on the plain 2D grayscale image, then add the channel dim back.
    env = ResizeObservation(env, shape=(84, 84))
    env = ReshapeObservation(env, (84, 84, 1))
    return env
