"""Gymnasium env factory for ViZDoom's rocket_basic.wad scenario.

Same task as basic.wad (one monster on the far wall, MOVE_LEFT/MOVE_RIGHT/
ATTACK) but with a rocket launcher instead of a hitscan pistol and
sv_noautoaim on — the projectile is slow, so the shot has to be aimed where
the reward will be, not where the monster is. Built-in reward mirrors basic
(hit bonus, living_reward=-1, 300-tic timeout), so shaping defaults are all
off.

This cfg is NOT registered by vizdoom.gymnasium_wrapper's __init__ (unlike
the main scenarios), so this module registers VizdoomRocketBasic-v1 itself
using the same entry point/kwargs pattern the package uses, guarded so that
SubprocVecEnv workers re-importing this module don't double-register.

Note: rocket_basic.cfg declares screen_format=GRAY8; envs.common forces
RGB24 at gym.make time so the shared grayscale->84x84 pipeline sees the same
3-channel input here as everywhere else.
"""

import gymnasium as gym
from gymnasium.envs.registration import register

from envs.common import make_vizdoom_env

ENV_ID = "VizdoomRocketBasic-v1"

if ENV_ID not in gym.registry:
    register(
        id=ENV_ID,
        entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
        kwargs={"scenario_config_file": "rocket_basic.cfg", "max_buttons_pressed": 1},
    )


def make_rocket_basic_env(
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
