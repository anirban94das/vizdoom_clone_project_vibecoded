"""Gymnasium env factory for ViZDoom's deadly_corridor.wad scenario.

Thin wrapper around envs.common.make_vizdoom_env — see that module for the
shared preprocessing pipeline (grayscale, resize, reshape, per-process ZDoom
config) and the full list of reward-shaping knobs. deadly_corridor.cfg
already defines death_penalty=100 and doom_skill=5, but doesn't score kills,
hits, exploration, or item pickups directly, so those four bonuses are turned
on by default here. weapon_pickup_bonus specifically rewards picking up the
shotgun that ShotgunGuy enemies drop on death (confirmed via the labels
buffer — this scenario's monsters are Zombieman and ShotgunGuy).
hit_reward_bonus rewards landing a shot on an enemy (HITCOUNT) even before it
dies, denser signal than the kill bonus alone.
"""

import gymnasium as gym

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomDeadlyCorridor-v1"

# Knobs this scenario turns on by default; every knob not named here defaults
# off in make_vizdoom_env. A train_*.py --flag overrides any of these.
SHAPING_DEFAULTS = {
    "kill_reward_bonus": 20.0,
    "hit_reward_bonus": 5.0,
    "exploration_bonus_per_cell": 1.0,
    "weapon_pickup_bonus": 15.0,
}


def make_deadly_corridor_env(
    render_mode: str | None = None, frame_skip: int = 4, **shaping_overrides: float
) -> gym.Env:
    return make_vizdoom_env(
        ENV_ID,
        render_mode=render_mode,
        frame_skip=frame_skip,
        **{**SHAPING_DEFAULTS, **shaping_overrides},
    )
