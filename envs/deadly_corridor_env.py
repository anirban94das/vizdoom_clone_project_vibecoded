"""Gymnasium env factory for ViZDoom's deadly_corridor.wad scenario.

Thin wrapper around envs.common.make_vizdoom_env — see that module for the
shared preprocessing pipeline (grayscale, resize, reshape, per-process ZDoom
config). deadly_corridor.cfg already defines death_penalty=100 and
doom_skill=5, but doesn't score kills, hits, exploration, or item pickups
directly, so kill_reward_bonus, hit_reward_bonus, exploration_bonus_per_cell,
and weapon_pickup_bonus are all enabled by default here. weapon_pickup_bonus
specifically rewards picking up the shotgun that ShotgunGuy enemies drop on
death (confirmed via the labels buffer — this scenario's monsters are
Zombieman and ShotgunGuy). hit_reward_bonus rewards landing a shot on an
enemy (HITCOUNT) even before it dies, denser signal than the kill bonus alone.
"""

import gymnasium as gym

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomDeadlyCorridor-v1"


def make_deadly_corridor_env(
    render_mode: str | None = None,
    frame_skip: int = 4,
    kill_reward_bonus: float = 20.0,
    exploration_bonus_per_cell: float = 1.0,
    exploration_cell_size: float = 32.0,
    weapon_pickup_bonus: float = 15.0,
    hit_reward_bonus: float = 5.0,
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
