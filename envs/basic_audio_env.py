"""Gymnasium env factory for ViZDoom's basic_audio.wad scenario.

Same task as basic.wad, but the scenario enables ViZDoom's audio buffer and
the whole point is exercising sound as an observation channel — so unlike
every other env module, this one does NOT go through
envs.common.make_vizdoom_env's screen-only pipeline. Instead it keeps a Dict
observation {screen, audio} for stable-baselines3's MultiInputPolicy (see
train_basic_audio.py):

- screen: grayscaled + resized to (84, 84, 1) manually via cv2 here, since
  gymnasium's Grayscale/Resize wrappers only operate on top-level array
  observations, not entries inside a Dict.
- audio: the raw int16 waveform ViZDoom captures over the frame_skip window
  (shape (samples, 2) at 44.1kHz — the gymnasium wrapper sizes the buffer to
  frame_skip tics automatically), normalized to float32 in [-1, 1] and
  flattened. Because the buffer spans the whole frame_skip window, it already
  carries temporal signal — which is why train_basic_audio.py skips
  VecFrameStack (n_stack=1) rather than stacking dicts.

Reward shaping: all off by default, same as basic.wad (built-in reward
suffices). The launch logic (per-process config, startup jitter) and the
stats/bonus wrappers are reused from envs.common — they only touch
reward/info, so they're observation-agnostic.

CAVEAT: ViZDoom's audio buffer needs a working OpenAL device; on machines
where audio init fails, DoomGame.init() raises. This module is untested on
this project's machine — if it fails at startup, that's the first suspect.
"""

import cv2
import gymnasium as gym
import numpy as np

from envs.common import apply_stats_and_reward_shaping, make_raw_vizdoom_env

ENV_ID = "VizdoomBasicAudio-v1"


class ScreenAudioPreprocess(gym.ObservationWrapper):
    """Dict-in, Dict-out: grayscale+resize the screen, normalize+flatten the
    audio, drop everything else (gamevariables)."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        audio_space = env.observation_space["audio"]
        n_audio = int(np.prod(audio_space.shape))
        self.observation_space = gym.spaces.Dict(
            {
                "screen": gym.spaces.Box(0, 255, shape=(84, 84, 1), dtype=np.uint8),
                "audio": gym.spaces.Box(-1.0, 1.0, shape=(n_audio,), dtype=np.float32),
            }
        )

    def observation(self, obs):
        screen = cv2.cvtColor(obs["screen"], cv2.COLOR_RGB2GRAY)
        screen = cv2.resize(screen, (84, 84), interpolation=cv2.INTER_AREA)
        audio = (obs["audio"].astype(np.float32) / 32768.0).ravel()
        return {"screen": screen[..., None], "audio": audio}


def make_basic_audio_env(
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
    env = make_raw_vizdoom_env(
        ENV_ID,
        render_mode=render_mode,
        frame_skip=frame_skip,
    )
    env = apply_stats_and_reward_shaping(
        env,
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
    return ScreenAudioPreprocess(env)
