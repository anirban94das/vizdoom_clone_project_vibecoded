"""Gymnasium env factory for ViZDoom's simpler_basic.wad scenario.

A gentler variant of basic.wad (same MOVE_LEFT/MOVE_RIGHT/ATTACK buttons,
living_reward=-1, 300-tic timeout) — useful as a smoke-test scenario since
anything that trains on basic should breeze through this. No shaping bonus on
by default, same as basic.

Like rocket_basic, this cfg is NOT registered by
vizdoom.gymnasium_wrapper's __init__, so this module registers
VizdoomSimplerBasic-v1 itself (guarded against SubprocVecEnv workers
double-registering on re-import). Its cfg also declares screen_format=GRAY8;
envs.common forces RGB24 so the shared pipeline is unaffected.
"""

import gymnasium as gym
from gymnasium.envs.registration import register

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomSimplerBasic-v1"

if ENV_ID not in gym.registry:
    register(
        id=ENV_ID,
        entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
        kwargs={"scenario_config_file": "simpler_basic.cfg", "max_buttons_pressed": 1},
    )

# Knobs this scenario turns on by default; every knob not named here defaults
# off in make_vizdoom_env. A train_*.py --flag overrides any of these.
SHAPING_DEFAULTS: dict[str, float] = {}


def make_simpler_basic_env(
    render_mode: str | None = None, frame_skip: int = 4, **shaping_overrides: float
) -> gym.Env:
    return make_vizdoom_env(
        ENV_ID,
        render_mode=render_mode,
        frame_skip=frame_skip,
        **{**SHAPING_DEFAULTS, **shaping_overrides},
    )
