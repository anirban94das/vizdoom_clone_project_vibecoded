"""Gymnasium env factory for ViZDoom's defend_the_line.wad scenario.

Thin wrapper around envs.common.make_vizdoom_env — see that module for the
shared preprocessing pipeline and the full list of reward-shaping knobs. Like
defend_the_center, the player can't move: defend_the_line.cfg only exposes
TURN_LEFT/TURN_RIGHT/ATTACK, but enemies approach from a line in front rather
than surrounding the player, and there's no episode_timeout — the episode
ends when the player dies (doom_skill=3, death_penalty=1, +1 per kill scored
by the scenario itself). Same shaping defaults as defend_the_center: kill and
hit bonuses on top of the built-in +1/kill, everything else off (standing
still is the objective; no pickupable weapons on the floor).
"""

import gymnasium as gym

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomDefendLine-v1"

# Knobs this scenario turns on by default; every knob not named here defaults
# off in make_vizdoom_env. A train_*.py --flag overrides any of these.
SHAPING_DEFAULTS = {
    "kill_reward_bonus": 20.0,
    "hit_reward_bonus": 5.0,
}


def make_defend_the_line_env(
    render_mode: str | None = None, frame_skip: int = 4, **shaping_overrides: float
) -> gym.Env:
    return make_vizdoom_env(
        ENV_ID,
        render_mode=render_mode,
        frame_skip=frame_skip,
        **{**SHAPING_DEFAULTS, **shaping_overrides},
    )
