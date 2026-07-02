"""Gymnasium env factory for ViZDoom's basic.wad scenario.

Wraps the Dict observation ViZDoom's gymnasium_wrapper returns (screen +
gamevariables) down to a single preprocessed image stream suitable for
stable-baselines3's CnnPolicy: grayscale, resized to 84x84, channel-last
(84, 84, 1). Frame-stacking is applied at the vec-env level (see
train_basic.py's VecFrameStack) rather than per-env, so each SubprocVecEnv
worker only has to ship one new frame across the process pipe per step
instead of a full 4-frame stack.
"""

import os
import random
import time
from pathlib import Path

import gymnasium as gym
import vizdoom as vzd
import vizdoom.gymnasium_wrapper  # noqa: F401  (registers Vizdoom* env ids)
from gymnasium.wrappers import GrayscaleObservation, ReshapeObservation, ResizeObservation

ENV_ID = "VizdoomBasic-v1"
CONFIG_DIR = Path("configs")


class ScreenOnlyObservation(gym.ObservationWrapper):
    """Discards gamevariables and exposes only the raw screen buffer."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = env.observation_space["screen"]

    def observation(self, obs):
        return obs["screen"]


def make_basic_env(render_mode: str | None = None, frame_skip: int = 4) -> gym.Env:
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
        ENV_ID,
        render_mode=render_mode,
        frame_skip=frame_skip,
        doom_config_path=doom_config_path,
        # Native render resolution: default is RES_320X240, but we resize
        # down to 84x84 anyway, so render at the smallest ViZDoom supports
        # (still comfortably >84px) to cut the software rasterizer's
        # per-step cost in every parallel worker.
        screen_resolution=vzd.ScreenResolution.RES_160X120,
    )
    env = ScreenOnlyObservation(env)
    env = GrayscaleObservation(env, keep_dim=False)
    # ResizeObservation calls cv2.resize, which silently drops a size-1
    # channel dim (bug: it declares shape (84, 84, 1) but returns (84, 84)).
    # Resize on the plain 2D grayscale image, then add the channel dim back.
    env = ResizeObservation(env, shape=(84, 84))
    env = ReshapeObservation(env, (84, 84, 1))
    return env
