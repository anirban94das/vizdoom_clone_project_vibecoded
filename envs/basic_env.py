"""Gymnasium env factory for ViZDoom's basic.wad scenario.

Thin wrapper around envs.common.make_vizdoom_env — see that module for the
shared preprocessing pipeline (grayscale, resize, reshape, per-process ZDoom
config) and the full list of reward-shaping knobs. Kept as its own module so
train_basic.py's import and call signature don't change as scenarios are
added.

basic.wad's built-in reward is already sufficient, so no shaping bonus is
turned on by default (SHAPING_DEFAULTS is empty). train_basic.py can still
opt into any of them via CLI flags, which arrive here as shaping_overrides.
"""

import gymnasium as gym

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomBasic-v1"

# Knobs this scenario turns on by default; every knob not named here defaults
# off in make_vizdoom_env. A train_*.py --flag overrides any of these.
SHAPING_DEFAULTS: dict[str, float] = {}


def make_basic_env(
    render_mode: str | None = None, frame_skip: int = 4, **shaping_overrides: float
) -> gym.Env:
    return make_vizdoom_env(
        ENV_ID,
        render_mode=render_mode,
        frame_skip=frame_skip,
        **{**SHAPING_DEFAULTS, **shaping_overrides},
    )
