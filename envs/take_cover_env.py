"""Gymnasium env factory for ViZDoom's take_cover.wad scenario.

Thin wrapper around envs.common.make_vizdoom_env — see that module for the
shared preprocessing pipeline and the full list of reward-shaping knobs. Pure
dodging: monsters at the far wall lob fireballs at the player, who can only
MOVE_LEFT/MOVE_RIGHT (no weapon, no turning). Built-in reward is
living_reward=+1 per tic with no episode_timeout — the episode ends only on
death (doom_skill=4), so reward equals survival time. damage_taken_penalty
adds a dense "that fireball you almost dodged still cost you" signal on top:
HEALTH only drops in chunks when a fireball connects, so penalizing each
DAMAGE_TAKEN point makes near misses vs. direct hits distinguishable to the
agent earlier in training.
"""

import gymnasium as gym

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomTakeCover-v1"

# Knobs this scenario turns on by default; every knob not named here defaults
# off in make_vizdoom_env. A train_*.py --flag overrides any of these.
SHAPING_DEFAULTS = {
    "damage_taken_penalty": 0.5,
}


def make_take_cover_env(
    render_mode: str | None = None, frame_skip: int = 4, **shaping_overrides: float
) -> gym.Env:
    return make_vizdoom_env(
        ENV_ID,
        render_mode=render_mode,
        frame_skip=frame_skip,
        **{**SHAPING_DEFAULTS, **shaping_overrides},
    )
