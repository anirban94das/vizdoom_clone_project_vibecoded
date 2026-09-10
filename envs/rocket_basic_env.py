"""Gymnasium env factory for ViZDoom's rocket_basic.wad scenario.

Same task as basic.wad (one monster on the far wall, MOVE_LEFT/MOVE_RIGHT/
ATTACK) but with a rocket launcher instead of a hitscan pistol and
sv_noautoaim on — the projectile is slow, so the shot has to be aimed where
the reward will be, not where the monster is. Built-in reward mirrors basic
(hit bonus, living_reward=-1, 300-tic timeout), so no shaping bonus is on by
default.

This cfg is NOT registered by vizdoom.gymnasium_wrapper's __init__ (unlike
the main scenarios), so this module registers VizdoomRocketBasic-v1 itself
using the same entry point/kwargs pattern the package uses, guarded so that
SubprocVecEnv workers re-importing this module don't double-register.

Note: rocket_basic.cfg declares screen_format=GRAY8; envs.common forces
RGB24 at gym.make time so the shared grayscale->84x84 pipeline sees the same
3-channel input here as everywhere else.
"""

import gymnasium as gym
from gymnasium.envs.registration import register

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomRocketBasic-v1"

if ENV_ID not in gym.registry:
    register(
        id=ENV_ID,
        entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
        kwargs={"scenario_config_file": "rocket_basic.cfg", "max_buttons_pressed": 1},
    )

# Knobs this scenario turns on by default; every knob not named here defaults
# off in make_vizdoom_env. A train_*.py --flag overrides any of these.
SHAPING_DEFAULTS: dict[str, float] = {}


def make_rocket_basic_env(
    render_mode: str | None = None, frame_skip: int = 4, **shaping_overrides: float
) -> gym.Env:
    return make_vizdoom_env(
        ENV_ID,
        render_mode=render_mode,
        frame_skip=frame_skip,
        **{**SHAPING_DEFAULTS, **shaping_overrides},
    )
