"""Gymnasium env factory for ViZDoom's take_cover.wad scenario.

Thin wrapper around envs.common.make_vizdoom_env — see that module for the
shared preprocessing pipeline. Pure dodging: monsters at the far wall lob
fireballs at the player, who can only MOVE_LEFT/MOVE_RIGHT (no weapon, no
turning). Built-in reward is living_reward=+1 per tic with no
episode_timeout — the episode ends only on death (doom_skill=4), so reward
equals survival time. damage_taken_penalty adds a dense "that fireball you
almost dodged still cost you" signal on top: HEALTH only drops in chunks
when a fireball connects, so penalizing each DAMAGE_TAKEN point makes near
misses vs. direct hits distinguishable to the agent earlier in training.
"""

import gymnasium as gym

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomTakeCover-v1"


def make_take_cover_env(
    render_mode: str | None = None,
    frame_skip: int = 4,
    kill_reward_bonus: float = 0.0,
    exploration_bonus_per_cell: float = 0.0,
    exploration_cell_size: float = 32.0,
    weapon_pickup_bonus: float = 0.0,
    hit_reward_bonus: float = 0.0,
    damage_dealt_bonus: float = 0.0,
    damage_taken_penalty: float = 0.5,
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
