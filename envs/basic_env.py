"""Gymnasium env factory for ViZDoom's basic.wad scenario.

Thin wrapper around envs.common.make_vizdoom_env — see that module for the
shared preprocessing pipeline (grayscale, resize, reshape, per-process ZDoom
config). Kept as its own module so train_basic.py's import and call
signature don't need to change as more scenarios are added.

Reward-shaping bonuses default to 0.0 (off) since basic.wad's built-in reward
is already sufficient, but are exposed here (rather than hardcoded at the
make_vizdoom_env call) so train_basic.py can opt into them via CLI flags.
"""

import gymnasium as gym

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomBasic-v1"


def make_basic_env(
    render_mode: str | None = None,
    frame_skip: int = 4,
    kill_reward_bonus: float = 0.0,
    exploration_bonus_per_cell: float = 0.0,
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
