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


def make_vizdoom_env(
    env_id: str,
    render_mode: str | None = None,
    frame_skip: int = 4,
    screen_resolution: vzd.ScreenResolution = vzd.ScreenResolution.RES_160X120,
    kill_reward_bonus: float = 0.0,
    exploration_bonus_per_cell: float = 0.0,
    exploration_cell_size: float = 32.0,
    weapon_pickup_bonus: float = 0.0,
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
    if kill_reward_bonus:
        env = KillRewardBonus(env, kill_reward_bonus)
    if exploration_bonus_per_cell:
        env = ExplorationBonus(env, exploration_bonus_per_cell, exploration_cell_size)
    if weapon_pickup_bonus:
        env = WeaponPickupBonus(env, weapon_pickup_bonus)
    env = ScreenOnlyObservation(env)
    env = GrayscaleObservation(env, keep_dim=False)
    # ResizeObservation calls cv2.resize, which silently drops a size-1
    # channel dim (bug: it declares shape (84, 84, 1) but returns (84, 84)).
    # Resize on the plain 2D grayscale image, then add the channel dim back.
    env = ResizeObservation(env, shape=(84, 84))
    env = ReshapeObservation(env, (84, 84, 1))
    return env
