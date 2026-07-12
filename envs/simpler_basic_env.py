"""Gymnasium env factory for ViZDoom's simpler_basic.wad scenario.

A gentler variant of basic.wad (same MOVE_LEFT/MOVE_RIGHT/ATTACK buttons,
living_reward=-1, 300-tic timeout) — useful as a smoke-test scenario since
anything that trains on basic should breeze through this. Shaping defaults
all off, same as basic.

Like rocket_basic, this cfg is NOT registered by
vizdoom.gymnasium_wrapper's __init__, so this module registers
VizdoomSimplerBasic-v1 itself (guarded against SubprocVecEnv workers
double-registering on re-import). Its cfg also declares screen_format=GRAY8;
envs.common forces RGB24 so the shared pipeline is unaffected.
"""

import gymnasium as gym
from gymnasium.envs.registration import register

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomSimplerBasic-v1"

if ENV_ID not in gym.registry:
    register(
        id=ENV_ID,
        entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
        kwargs={"scenario_config_file": "simpler_basic.cfg", "max_buttons_pressed": 1},
    )


def make_simpler_basic_env(
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
