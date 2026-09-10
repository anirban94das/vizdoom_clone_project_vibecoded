"""Gymnasium env factory for ViZDoom's defend_the_center.wad scenario.

Thin wrapper around envs.common.make_vizdoom_env — see that module for the
shared preprocessing pipeline (grayscale, resize, reshape, per-process ZDoom
config) and the full list of reward-shaping knobs. defend_the_center.cfg only
exposes TURN_LEFT/TURN_RIGHT/ATTACK (the player is fixed at the center of the
room and can't move) and already scores +1 per kill / -1 on death via
death_penalty=1 internally, but doesn't score hits directly, so the kill and
hit bonuses are turned on by default here on top of that built-in signal.
Exploration stays off — standing still is the actual objective in this
scenario, unlike deadly_corridor — and there are no pickupable weapons on the
floor.
"""

import gymnasium as gym

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomDefendCenter-v1"

# Knobs this scenario turns on by default; every knob not named here defaults
# off in make_vizdoom_env. A train_*.py --flag overrides any of these.
SHAPING_DEFAULTS = {
    "kill_reward_bonus": 20.0,
    "hit_reward_bonus": 5.0,
}


def make_defend_the_center_env(
    render_mode: str | None = None, frame_skip: int = 4, **shaping_overrides: float
) -> gym.Env:
    return make_vizdoom_env(
        ENV_ID,
        render_mode=render_mode,
        frame_skip=frame_skip,
        **{**SHAPING_DEFAULTS, **shaping_overrides},
    )
