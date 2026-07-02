"""Gymnasium env factory for ViZDoom's deadly_corridor.wad scenario.

Thin wrapper around envs.common.make_vizdoom_env — see that module for the
shared preprocessing pipeline (grayscale, resize, reshape, per-process ZDoom
config). deadly_corridor.cfg already defines death_penalty=100 and
doom_skill=5, so no extra reward shaping is needed here.
"""

import gymnasium as gym

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomDeadlyCorridor-v1"


def make_deadly_corridor_env(render_mode: str | None = None, frame_skip: int = 4) -> gym.Env:
    return make_vizdoom_env(ENV_ID, render_mode=render_mode, frame_skip=frame_skip)
