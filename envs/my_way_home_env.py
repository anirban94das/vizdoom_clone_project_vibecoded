"""Gymnasium env factory for ViZDoom's my_way_home.wad scenario.

Thin wrapper around envs.common.make_vizdoom_env — see that module for the
shared preprocessing pipeline and the full list of reward-shaping knobs. Pure
navigation: the player spawns in a random room of a small maze and must find
a green vest (+1 on reaching it, living reward -0.0001/tic, episode_timeout
2100 tics). Buttons: TURN_LEFT/TURN_RIGHT/MOVE_FORWARD/MOVE_LEFT/MOVE_RIGHT.
The built-in reward is about as sparse as it gets — one +1 at the very end of
a successful episode — which is exactly what ExplorationBonus was built for
in deadly_corridor: +1 per newly visited 32-unit grid cell per episode gives
dense signal for covering new ground until the vest is stumbled into, after
which the +1 goal reward can take over. No combat, so all combat bonuses
default off.
"""

import gymnasium as gym

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomMyWayHome-v1"

# Knobs this scenario turns on by default; every knob not named here defaults
# off in make_vizdoom_env. A train_*.py --flag overrides any of these.
SHAPING_DEFAULTS = {
    "exploration_bonus_per_cell": 1.0,
}


def make_my_way_home_env(
    render_mode: str | None = None, frame_skip: int = 4, **shaping_overrides: float
) -> gym.Env:
    return make_vizdoom_env(
        ENV_ID,
        render_mode=render_mode,
        frame_skip=frame_skip,
        **{**SHAPING_DEFAULTS, **shaping_overrides},
    )
