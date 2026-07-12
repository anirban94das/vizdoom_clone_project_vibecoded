"""Gymnasium env factory for ViZDoom's health_gathering.wad scenario.

Thin wrapper around envs.common.make_vizdoom_env — see that module for the
shared preprocessing pipeline. Survival scenario: the floor is acid and
constantly drains health, medkits spawn around the room, and the built-in
reward is living_reward=+1 per tic with death_penalty=100 (episode_timeout
2100 tics ≈ 60s). No combat — TURN_LEFT/TURN_RIGHT/MOVE_FORWARD are the only
buttons — so the kill/hit/weapon bonuses don't apply. health_change_bonus is
the natural shaping here: +1 reward per HEALTH point gained (medkit pickups)
and -1 per point lost (acid damage) makes seeking medkits an explicit,
denser signal than the survival-time reward alone.
"""

import gymnasium as gym

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomHealthGathering-v1"


def make_health_gathering_env(
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
