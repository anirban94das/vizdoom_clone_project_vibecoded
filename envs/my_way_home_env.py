"""Gymnasium env factory for ViZDoom's my_way_home.wad scenario.

Thin wrapper around envs.common.make_vizdoom_env — see that module for the
shared preprocessing pipeline. Pure navigation: the player spawns in a random
room of a small maze and must find a green vest (+1 on reaching it, living
reward -0.0001/tic, episode_timeout 2100 tics). Buttons: TURN_LEFT/TURN_RIGHT/
MOVE_FORWARD/MOVE_LEFT/MOVE_RIGHT. The built-in reward is about as sparse as
it gets — one +1 at the very end of a successful episode — which is exactly
what ExplorationBonus was built for in deadly_corridor: +1 per newly visited
32-unit grid cell per episode gives dense signal for covering new ground
until the vest is stumbled into, after which the +1 goal reward can take over.
No combat, so all combat bonuses default off.
"""

import gymnasium as gym

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomMyWayHome-v1"


def make_my_way_home_env(
    render_mode: str | None = None,
    frame_skip: int = 4,
    kill_reward_bonus: float = 0.0,
    exploration_bonus_per_cell: float = 1.0,
    exploration_cell_size: float = 32.0,
    weapon_pickup_bonus: float = 0.0,
    hit_reward_bonus: float = 0.0,
    damage_dealt_bonus: float = 0.0,
    damage_taken_penalty: float = 0.0,
    health_change_bonus: float = 0.0,
    armor_change_bonus: float = 0.0,
) -> gym.Env:
    return make_vizdoom_env(
        ENV_ID,
        render_mode=render_mode,
        frame_skip=frame_skip,
        kill_reward_bonus=kill_reward_bonus,
        exploration_bonus_per_cell=exploration_bonus_per_cell,
        exploration_cell_size=exploration_cell_size,
        weapon_pickup_bonus=weapon_pickup_bonus,
        hit_reward_bonus=hit_reward_bonus,
        damage_dealt_bonus=damage_dealt_bonus,
        damage_taken_penalty=damage_taken_penalty,
        health_change_bonus=health_change_bonus,
        armor_change_bonus=armor_change_bonus,
    )
