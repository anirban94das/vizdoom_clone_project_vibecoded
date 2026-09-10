"""Gymnasium env factory for ViZDoom's health_gathering_supreme.wad scenario.

Same objective, buttons, and built-in reward as health_gathering (see
envs/health_gathering_env.py) on a harder map: a maze layout instead of one
open room, so medkits must be actively found rather than just steered toward.
Identical shaping defaults (health_change_bonus=1.0). train_*.py warm-starts
this scenario from the plain health_gathering model when available — same
task, same action space, transferable visual features.

Exploration stays off by default even though this is a maze: the
health_change_bonus + living_reward already reward finding medkits, and
rewarding exploration for its own sake risks the agent wandering
acid-damaged corridors instead of surviving. Turn it on via flag if the
agent gets stuck circling the starting area.
"""

import gymnasium as gym

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomHealthGatheringSupreme-v1"

# Knobs this scenario turns on by default; every knob not named here defaults
# off in make_vizdoom_env. A train_*.py --flag overrides any of these.
SHAPING_DEFAULTS = {
    "health_change_bonus": 1.0,
}


def make_health_gathering_supreme_env(
    render_mode: str | None = None, frame_skip: int = 4, **shaping_overrides: float
) -> gym.Env:
    return make_vizdoom_env(
        ENV_ID,
        render_mode=render_mode,
        frame_skip=frame_skip,
        **{**SHAPING_DEFAULTS, **shaping_overrides},
    )
