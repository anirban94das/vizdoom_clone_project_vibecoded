"""Gymnasium env factory for ViZDoom's predict_position.wad scenario.

Thin wrapper around envs.common.make_vizdoom_env — see that module for the
shared preprocessing pipeline and the full list of reward-shaping knobs.
Aim-leading task: a monster walks across the far end of the room, the player
has a rocket launcher and effectively one shot per episode (episode_timeout
is only 300 tics, and rockets travel slowly, so a missed rocket usually IS
the episode). Buttons: TURN_LEFT/TURN_RIGHT/ATTACK, with sv_noautoaim so the
engine won't help. Built-in reward: living_reward=-0.001/tic, +1 for the kill
(doom_skill=1).

Kill/hit bonuses default an order of magnitude larger than the hitscan
scenarios' 20/5, per the roadmap note that single-shot scenarios likely need
much larger magnitudes to matter: success happens at most once per episode,
so that one event has to dominate the episode's return. Untested guess —
tune via flags if the agent doesn't converge.
"""

import gymnasium as gym

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomPredictPosition-v1"

# Knobs this scenario turns on by default; every knob not named here defaults
# off in make_vizdoom_env. A train_*.py --flag overrides any of these.
SHAPING_DEFAULTS = {
    "kill_reward_bonus": 100.0,
    "hit_reward_bonus": 25.0,
}


def make_predict_position_env(
    render_mode: str | None = None, frame_skip: int = 4, **shaping_overrides: float
) -> gym.Env:
    return make_vizdoom_env(
        ENV_ID,
        render_mode=render_mode,
        frame_skip=frame_skip,
        **{**SHAPING_DEFAULTS, **shaping_overrides},
    )
