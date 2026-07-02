"""Gymnasium env factory for ViZDoom's basic.wad scenario.

Thin wrapper around envs.common.make_vizdoom_env — see that module for the
shared preprocessing pipeline (grayscale, resize, reshape, per-process ZDoom
config). Kept as its own module so train_basic.py's import and call
signature don't need to change as more scenarios are added.
"""

import gymnasium as gym

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomBasic-v1"


def make_basic_env(render_mode: str | None = None, frame_skip: int = 4) -> gym.Env:
    return make_vizdoom_env(ENV_ID, render_mode=render_mode, frame_skip=frame_skip)
