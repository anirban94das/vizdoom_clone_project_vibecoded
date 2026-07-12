"""Gymnasium env factory for ViZDoom's defend_the_line.wad scenario.

Thin wrapper around envs.common.make_vizdoom_env — see that module for the
shared preprocessing pipeline. Like defend_the_center, the player can't move:
defend_the_line.cfg only exposes TURN_LEFT/TURN_RIGHT/ATTACK, but enemies
approach from a line in front rather than surrounding the player, and there's
no episode_timeout — the episode ends when the player dies (doom_skill=3,
death_penalty=1, +1 per kill scored by the scenario itself). Same shaping
defaults as defend_the_center: kill and hit bonuses on top of the built-in
+1/kill, exploration and weapon-pickup bonuses off (standing still is the
objective; no pickupable weapons on the floor).
"""

import gymnasium as gym

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomDefendLine-v1"


def make_defend_the_line_env(
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
