"""Gymnasium env factory for ViZDoom's health_gathering_supreme.wad scenario.

Same objective, buttons, and built-in reward as health_gathering (see
envs/health_gathering_env.py) on a harder map: a maze layout instead of one
open room, so medkits must be actively found rather than just steered toward.
Identical shaping defaults (health_change_bonus=1.0). train_*.py warm-starts
this scenario from the plain health_gathering model when available — same
task, same action space, transferable visual features.

exploration_bonus_per_cell stays 0.0 by default even though this is a maze:
the health_change_bonus + living_reward already reward finding medkits, and
rewarding exploration for its own sake risks the agent wandering acid-damaged
corridors instead of surviving. Turn it on via flag if the agent gets stuck
circling the starting area.
"""

import gymnasium as gym

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomHealthGatheringSupreme-v1"


def make_health_gathering_supreme_env(
    render_mode: str | None = None,
    frame_skip: int = 4,
    kill_reward_bonus: float = 0.0,
    exploration_bonus_per_cell: float = 0.0,
    exploration_cell_size: float = 32.0,
    weapon_pickup_bonus: float = 0.0,
    hit_reward_bonus: float = 0.0,
    damage_dealt_bonus: float = 0.0,
    damage_taken_penalty: float = 0.0,
    health_change_bonus: float = 1.0,
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
