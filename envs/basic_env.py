"""Gymnasium env factory for ViZDoom's basic.wad scenario.

Wraps the Dict observation ViZDoom's gymnasium_wrapper returns (screen +
gamevariables) down to a single preprocessed image stream suitable for
stable-baselines3's CnnPolicy: grayscale, resized to 84x84, 4-frame stack,
channel-first (4, 84, 84).
"""

import gymnasium as gym
import vizdoom.gymnasium_wrapper  # noqa: F401  (registers Vizdoom* env ids)
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation, ResizeObservation

ENV_ID = "VizdoomBasic-v1"


class ScreenOnlyObservation(gym.ObservationWrapper):
    """Discards gamevariables and exposes only the raw screen buffer."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = env.observation_space["screen"]

    def observation(self, obs):
        return obs["screen"]


def make_basic_env(render_mode: str | None = None, frame_skip: int = 4) -> gym.Env:
    env = gym.make(ENV_ID, render_mode=render_mode, frame_skip=frame_skip)
    env = ScreenOnlyObservation(env)
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, shape=(84, 84))
    env = FrameStackObservation(env, stack_size=4)
    return env
