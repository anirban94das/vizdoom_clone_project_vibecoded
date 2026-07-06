"""Gymnasium env factory for ViZDoom's defend_the_center.wad scenario.

Thin wrapper around envs.common.make_vizdoom_env — see that module for the
shared preprocessing pipeline (grayscale, resize, reshape, per-process ZDoom
config). defend_the_center.cfg only exposes TURN_LEFT/TURN_RIGHT/ATTACK (the
player is fixed at the center of the room and can't move) and already scores
+1 per kill / -1 on death via death_penalty=1 internally, but doesn't score
hits directly, so kill_reward_bonus and hit_reward_bonus are enabled by
default here on top of that built-in signal. exploration_bonus_per_cell
defaults to 0.0 (off) and stays off — standing still is the actual objective
in this scenario, unlike deadly_corridor. weapon_pickup_bonus also defaults
to 0.0 since this scenario doesn't have pickupable weapons on the floor.
"""

import gymnasium as gym

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomDefendCenter-v1"


def make_defend_the_center_env(
    render_mode: str | None = None,
    frame_skip: int = 4,
    kill_reward_bonus: float = 20.0,
    exploration_bonus_per_cell: float = 0.0,
    exploration_cell_size: float = 32.0,
    weapon_pickup_bonus: float = 0.0,
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
