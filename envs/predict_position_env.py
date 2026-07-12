"""Gymnasium env factory for ViZDoom's predict_position.wad scenario.

Thin wrapper around envs.common.make_vizdoom_env — see that module for the
shared preprocessing pipeline. Aim-leading task: a monster walks across the
far end of the room, the player has a rocket launcher and effectively one
shot per episode (episode_timeout is only 300 tics, and rockets travel
slowly, so a missed rocket usually IS the episode). Buttons: TURN_LEFT/
TURN_RIGHT/ATTACK, with sv_noautoaim so the engine won't help. Built-in
reward: living_reward=-0.001/tic, +1 for the kill (doom_skill=1).

Kill/hit bonuses default an order of magnitude larger than the hitscan
scenarios' 20/5, per the roadmap note that single-shot scenarios likely need
much larger magnitudes to matter: success happens at most once per episode,
so that one event has to dominate the episode's return. Untested guess —
tune via flags if the agent doesn't converge.
"""

import gymnasium as gym

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomPredictPosition-v1"


def make_predict_position_env(
    render_mode: str | None = None,
    frame_skip: int = 4,
    kill_reward_bonus: float = 100.0,
    exploration_bonus_per_cell: float = 0.0,
    exploration_cell_size: float = 32.0,
    weapon_pickup_bonus: float = 0.0,
    hit_reward_bonus: float = 25.0,
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
