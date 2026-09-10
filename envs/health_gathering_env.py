"""Gymnasium env factory for ViZDoom's health_gathering.wad scenario.

Thin wrapper around envs.common.make_vizdoom_env — see that module for the
shared preprocessing pipeline and the full list of reward-shaping knobs.
Survival scenario: the floor is acid and constantly drains health, medkits
spawn around the room, and the built-in reward is living_reward=+1 per tic
with death_penalty=100 (episode_timeout 2100 tics ≈ 60s). No combat —
TURN_LEFT/TURN_RIGHT/MOVE_FORWARD are the only buttons — so the kill/hit/
weapon bonuses don't apply. health_change_bonus is the natural shaping here:
+1 reward per HEALTH point gained (medkit pickups) and -1 per point lost
(acid damage) makes seeking medkits an explicit, denser signal than the
survival-time reward alone.
"""

import gymnasium as gym

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomHealthGathering-v1"

# Knobs this scenario turns on by default; every knob not named here defaults
# off in make_vizdoom_env. A train_*.py --flag overrides any of these.
SHAPING_DEFAULTS = {
    "health_change_bonus": 1.0,
}


def make_health_gathering_env(
    render_mode: str | None = None, frame_skip: int = 4, **shaping_overrides: float
) -> gym.Env:
    return make_vizdoom_env(
        ENV_ID,
        render_mode=render_mode,
        frame_skip=frame_skip,
        **{**SHAPING_DEFAULTS, **shaping_overrides},
    )
