"""Shared Gymnasium env factory for ViZDoom scenarios.

Preprocessing pipeline used by every per-scenario env module (envs/basic_env.py,
envs/deadly_corridor_env.py, ...): strips the Dict observation down to the raw
screen buffer, then grayscale -> resize to 84x84 -> explicit reshape back to
(84, 84, 1) so stable-baselines3's CnnPolicy gets a consistent channel-last
uint8 frame regardless of scenario. Frame-stacking is applied at the vec-env
level (see each train_*.py's VecFrameStack) rather than here, so each
SubprocVecEnv worker only ships one new frame across the process pipe per
step instead of a full 4-frame stack.
"""

import os
import random
import time
from pathlib import Path

import gymnasium as gym
import vizdoom as vzd
import vizdoom.gymnasium_wrapper  # noqa: F401  (registers Vizdoom* env ids)
from gymnasium.wrappers import GrayscaleObservation, ReshapeObservation, ResizeObservation

CONFIG_DIR = Path("configs")


class ScreenOnlyObservation(gym.ObservationWrapper):
    """Discards gamevariables and exposes only the raw screen buffer."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = env.observation_space["screen"]

    def observation(self, obs):
        return obs["screen"]


class KillRewardBonus(gym.Wrapper):
    """Adds bonus_per_kill on top of the scenario's built-in reward for each
    KILLCOUNT increment. Scenarios like deadly_corridor score distance-to-goal
    and death_penalty but not kills directly, so this makes killing enemies an
    explicit incentive rather than a side-effect of survival."""

    def __init__(self, env: gym.Env, bonus_per_kill: float):
        super().__init__(env)
        self.bonus_per_kill = bonus_per_kill
        self._last_kills = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_kills = self.unwrapped.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        kills = self.unwrapped.game.get_game_variable(vzd.GameVariable.KILLCOUNT)
        reward += (kills - self._last_kills) * self.bonus_per_kill
        self._last_kills = kills
        return obs, reward, terminated, truncated, info


class ExplorationBonus(gym.Wrapper):
    """Rewards visiting new (discretized) positions, once per episode per
    cell. Grid-cell novelty (not raw per-step distance) specifically avoids
    rewarding back-and-forth oscillation in place — only genuinely new ground
    pays out."""

    def __init__(self, env: gym.Env, bonus_per_cell: float, cell_size: float):
        super().__init__(env)
        self.bonus_per_cell = bonus_per_cell
        self.cell_size = cell_size
        self._visited: set[tuple[int, int]] = set()

    def _cell(self) -> tuple[int, int]:
        game = self.unwrapped.game
        x = game.get_game_variable(vzd.GameVariable.POSITION_X)
        y = game.get_game_variable(vzd.GameVariable.POSITION_Y)
        return (int(x // self.cell_size), int(y // self.cell_size))

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._visited = {self._cell()}
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cell = self._cell()
        if cell not in self._visited:
            self._visited.add(cell)
            reward += self.bonus_per_cell
        return obs, reward, terminated, truncated, info


class WeaponPickupBonus(gym.Wrapper):
    """Adds bonus_per_weapon reward the first time each episode a new weapon
    slot (WEAPON0..WEAPON9) is acquired. In deadly_corridor, ShotgunGuy
    enemies drop a pickupable shotgun on death (vanilla Doom behavior) — this
    rewards walking over and picking it up, distinct from the kill itself."""

    WEAPON_VARS = [getattr(vzd.GameVariable, f"WEAPON{i}") for i in range(10)]

    def __init__(self, env: gym.Env, bonus_per_weapon: float):
        super().__init__(env)
        self.bonus_per_weapon = bonus_per_weapon
        self._owned: set[int] = set()

    def _owned_weapons(self) -> set[int]:
        game = self.unwrapped.game
        return {i for i, var in enumerate(self.WEAPON_VARS) if game.get_game_variable(var) > 0}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._owned = self._owned_weapons()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        owned = self._owned_weapons()
        reward += len(owned - self._owned) * self.bonus_per_weapon
        self._owned = owned
        return obs, reward, terminated, truncated, info


class EpisodeStatsWrapper(gym.Wrapper):
    """Tracks kills/hits/explored-cells/weapons-picked-up per episode purely
    for reporting (via info["episode_stats"] on the terminal step), separate
    from the opt-in reward-shaping wrappers above. Always applied regardless
    of whether those bonuses are turned on, so a scenario like basic.wad
    (all bonuses off by default) still gets a behavior recap, not just a
    reward number."""

    def __init__(self, env: gym.Env, exploration_cell_size: float = 32.0):
        super().__init__(env)
        self.exploration_cell_size = exploration_cell_size
        self._visited: set[tuple[int, int]] = set()
        self._owned_at_reset: set[int] = set()
        self._damage_dealt_at_reset = 0.0
        self._damage_taken_at_reset = 0.0

    def _cell(self) -> tuple[int, int]:
        game = self.unwrapped.game
        x = game.get_game_variable(vzd.GameVariable.POSITION_X)
        y = game.get_game_variable(vzd.GameVariable.POSITION_Y)
        return (int(x // self.exploration_cell_size), int(y // self.exploration_cell_size))

    def _owned_weapons(self) -> set[int]:
        game = self.unwrapped.game
        return {i for i, var in enumerate(WeaponPickupBonus.WEAPON_VARS) if game.get_game_variable(var) > 0}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        game = self.unwrapped.game
        self._visited = {self._cell()}
        self._owned_at_reset = self._owned_weapons()
        self._damage_dealt_at_reset = game.get_game_variable(vzd.GameVariable.DAMAGECOUNT)
        self._damage_taken_at_reset = game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._visited.add(self._cell())
        if terminated or truncated:
            game = self.unwrapped.game
            info["episode_stats"] = {
                "kills": game.get_game_variable(vzd.GameVariable.KILLCOUNT),
                "hits": game.get_game_variable(vzd.GameVariable.HITCOUNT),
                "damage_dealt": game.get_game_variable(vzd.GameVariable.DAMAGECOUNT) - self._damage_dealt_at_reset,
                "damage_taken": game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN) - self._damage_taken_at_reset,
                "cells_explored": len(self._visited),
                "weapons_picked_up": len(self._owned_weapons() - self._owned_at_reset),
            }
        return obs, reward, terminated, truncated, info


class HitRewardBonus(gym.Wrapper):
    """Adds bonus_per_hit on top of the scenario's built-in reward for each
    HITCOUNT increment. HITCOUNT fires on every successful hit landed on an
    enemy, not just kills — this rewards shooting at (and connecting with) an
    enemy as a denser signal leading up to KillRewardBonus's kill credit."""

    def __init__(self, env: gym.Env, bonus_per_hit: float):
        super().__init__(env)
        self.bonus_per_hit = bonus_per_hit
        self._last_hits = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_hits = self.unwrapped.game.get_game_variable(vzd.GameVariable.HITCOUNT)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        hits = self.unwrapped.game.get_game_variable(vzd.GameVariable.HITCOUNT)
        reward += (hits - self._last_hits) * self.bonus_per_hit
        self._last_hits = hits
        return obs, reward, terminated, truncated, info


class DamageDealtBonus(gym.Wrapper):
    """Adds bonus_per_damage reward per DAMAGECOUNT point dealt to enemies.
    Denser/more informative than HitRewardBonus - a solid hit and a grazing
    hit both count as "1 hit" under HITCOUNT, but deal very different
    DAMAGECOUNT, so this rewards landing damaging hits specifically."""

    def __init__(self, env: gym.Env, bonus_per_damage: float):
        super().__init__(env)
        self.bonus_per_damage = bonus_per_damage
        self._last_damage = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_damage = self.unwrapped.game.get_game_variable(vzd.GameVariable.DAMAGECOUNT)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        damage = self.unwrapped.game.get_game_variable(vzd.GameVariable.DAMAGECOUNT)
        reward += (damage - self._last_damage) * self.bonus_per_damage
        self._last_damage = damage
        return obs, reward, terminated, truncated, info


class DamageTakenPenalty(gym.Wrapper):
    """Subtracts penalty_per_damage reward per DAMAGE_TAKEN point received.
    penalty_per_damage is a non-negative magnitude that gets subtracted (like
    ViZDoom's own death_penalty config value) rather than added, so a
    positive value here always discourages getting hurt instead of
    accidentally rewarding it."""

    def __init__(self, env: gym.Env, penalty_per_damage: float):
        super().__init__(env)
        self.penalty_per_damage = penalty_per_damage
        self._last_damage_taken = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_damage_taken = self.unwrapped.game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        damage_taken = self.unwrapped.game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN)
        reward -= (damage_taken - self._last_damage_taken) * self.penalty_per_damage
        self._last_damage_taken = damage_taken
        return obs, reward, terminated, truncated, info


class HealthChangeBonus(gym.Wrapper):
    """Adds bonus_per_point reward per net HEALTH point gained, and removes
    it per point lost, each step. A positive delta means a medkit/health
    pickup, a negative delta means damage taken - one coefficient naturally
    rewards healing and penalizes health loss symmetrically."""

    def __init__(self, env: gym.Env, bonus_per_point: float):
        super().__init__(env)
        self.bonus_per_point = bonus_per_point
        self._last_health = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_health = self.unwrapped.game.get_game_variable(vzd.GameVariable.HEALTH)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        health = self.unwrapped.game.get_game_variable(vzd.GameVariable.HEALTH)
        reward += (health - self._last_health) * self.bonus_per_point
        self._last_health = health
        return obs, reward, terminated, truncated, info


class ArmorChangeBonus(gym.Wrapper):
    """Adds bonus_per_point reward per net ARMOR point gained, and removes it
    per point lost, each step. Armor absorbs part of incoming damage before
    HEALTH drops, and is topped up by armor bonus/armor pickups - same
    symmetric-delta approach as HealthChangeBonus."""

    def __init__(self, env: gym.Env, bonus_per_point: float):
        super().__init__(env)
        self.bonus_per_point = bonus_per_point
        self._last_armor = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._last_armor = self.unwrapped.game.get_game_variable(vzd.GameVariable.ARMOR)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        armor = self.unwrapped.game.get_game_variable(vzd.GameVariable.ARMOR)
        reward += (armor - self._last_armor) * self.bonus_per_point
        self._last_armor = armor
        return obs, reward, terminated, truncated, info


def make_vizdoom_env(
    env_id: str,
    render_mode: str | None = None,
    frame_skip: int = 4,
    screen_resolution: vzd.ScreenResolution = vzd.ScreenResolution.RES_160X120,
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
    # Each SubprocVecEnv worker is a separate OS process; give it its own
    # ZDoom config file so concurrent instances don't race on the shared
    # default _vizdoom.ini (which manifests as "viz_instance_id is write
    # protected" when multiple instances start at once).
    CONFIG_DIR.mkdir(exist_ok=True)
    doom_config_path = str(CONFIG_DIR / f"vizdoom_{os.getpid()}.ini")

    # SubprocVecEnv starts all workers at once, so every worker's DoomGame
    # boots in the same instant. Under heavy concurrency that race left one
    # worker's engine half-initialized (state stuck at None, "Call reset
    # before using step" on the first step). Jittering startup spreads the
    # engine boots out instead of hitting them all simultaneously.
    time.sleep(random.uniform(0, 2.0))

    env = gym.make(
        env_id,
        render_mode=render_mode,
        frame_skip=frame_skip,
        doom_config_path=doom_config_path,
        # Native render resolution: default is usually higher than 84x84, but
        # we resize down to 84x84 anyway, so render at the smallest ViZDoom
        # supports (still comfortably >84px) to cut the software rasterizer's
        # per-step cost in every parallel worker.
        screen_resolution=screen_resolution,
    )
    # Always on (unlike the opt-in bonuses below) so every scenario gets a
    # behavior recap - kills/hits/exploration/weapons - even when none of
    # the reward bonuses are enabled for it.
    env = EpisodeStatsWrapper(env, exploration_cell_size=exploration_cell_size)
    if kill_reward_bonus:
        env = KillRewardBonus(env, kill_reward_bonus)
    if exploration_bonus_per_cell:
        env = ExplorationBonus(env, exploration_bonus_per_cell, exploration_cell_size)
    if weapon_pickup_bonus:
        env = WeaponPickupBonus(env, weapon_pickup_bonus)
    if hit_reward_bonus:
        env = HitRewardBonus(env, hit_reward_bonus)
    if damage_dealt_bonus:
        env = DamageDealtBonus(env, damage_dealt_bonus)
    if damage_taken_penalty:
        env = DamageTakenPenalty(env, damage_taken_penalty)
    if health_change_bonus:
        env = HealthChangeBonus(env, health_change_bonus)
    if armor_change_bonus:
        env = ArmorChangeBonus(env, armor_change_bonus)
    env = ScreenOnlyObservation(env)
    env = GrayscaleObservation(env, keep_dim=False)
    # ResizeObservation calls cv2.resize, which silently drops a size-1
    # channel dim (bug: it declares shape (84, 84, 1) but returns (84, 84)).
    # Resize on the plain 2D grayscale image, then add the channel dim back.
    env = ResizeObservation(env, shape=(84, 84))
    env = ReshapeObservation(env, (84, 84, 1))
    return env
