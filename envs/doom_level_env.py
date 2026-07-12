"""Gymnasium env factory for full DOOM / DOOM II game levels (E1M1, MAP01, ...).

vizdoom 1.3.0 pre-registers every original map at every skill as
Vizdoom{Doom,Doom2,Freedoom1,Freedoom2}{MAP}-S{1..5}-v0 (via doom.cfg /
doom2.cfg / freedoom1.cfg / freedoom2.cfg + a doom_map kwarg), and it ships
the Freedoom WADs — so Freedoom "just works" with no downloads. The original
doom.wad / doom2.wad are commercial (the engine is GPL; the game data isn't):
if you own them (e.g. Steam), drop them into this project's wads/ directory
and this factory auto-detects and uses them instead, passing doom_game_path
through gym.make. Same model file either way — swapping the WAD between runs
just changes the art/level layouts the agent trains on from then on.

Differences from the scenario envs, all handled here:

- Which env id: picked from the map name (E?M? -> Doom/Freedoom1,
  MAP?? -> Doom2/Freedoom2) + skill (1-5, default 3 "Hurt me plenty" —
  the scenarios' doom_skill=5 equivalent is brutal on full maps).
- Action space: the full-game cfgs expose 19 buttons and are only registered
  as MultiBinary; max_buttons_pressed=1 is overridden at gym.make time so we
  get the same Discrete action space (one button per step, 20 actions incl.
  noop) as every other scenario — CnnPolicy-compatible and consistent.
- Episode length: the cfgs' episode_timeout is 126000 tics (60 min);
  default here is 21000 (10 min) so early training gets more resets.
- Reward: the cfgs only score map_exit_reward=1 and nothing else. Overrides:
  map_exit_reward=1000 (finishing the level should dominate everything) and
  death_penalty=100, plus this factory's shaping defaults — the full
  deadly_corridor combat set (kill/hit/weapon) + exploration (these are big
  maps to navigate) + health/armor deltas (pickups matter over a 10-minute
  episode).
- Buffers: the cfgs enable audio + automap buffers; both are disabled here
  (audio needs OpenAL and can fail at init; neither is in our observation).
"""

import re
from pathlib import Path

import gymnasium as gym

from envs.common import make_vizdoom_env

WADS_DIR = Path(__file__).resolve().parent.parent / "wads"

_EPISODE_MAP = re.compile(r"^E[1-4]M[1-9]$")  # doom.wad / freedoom1.wad slots
_LEVEL_MAP = re.compile(r"^MAP(0[1-9]|[12][0-9]|3[0-2])$")  # doom2 / freedoom2 slots


def resolve_game(map_id: str) -> tuple[str, Path | None]:
    """Maps e.g. "E1M1" -> ("Doom", wads/doom.wad) if the commercial WAD is
    present, else ("Freedoom1", None) — None meaning "use the freedoom WAD
    bundled with the vizdoom package" (its cfg's default doom_game_path)."""
    map_id = map_id.upper()
    if _EPISODE_MAP.match(map_id):
        wad = WADS_DIR / "doom.wad"
        return ("Doom", wad) if wad.exists() else ("Freedoom1", None)
    if _LEVEL_MAP.match(map_id):
        wad = WADS_DIR / "doom2.wad"
        return ("Doom2", wad) if wad.exists() else ("Freedoom2", None)
    raise ValueError(
        f"Unrecognized map {map_id!r}: expected E1M1..E4M9 (DOOM) or MAP01..MAP32 (DOOM II)."
    )


def make_doom_level_env(
    map_id: str = "E1M1",
    skill: int = 3,
    render_mode: str | None = None,
    frame_skip: int = 4,
    kill_reward_bonus: float = 20.0,
    exploration_bonus_per_cell: float = 1.0,
    exploration_cell_size: float = 32.0,
    weapon_pickup_bonus: float = 15.0,
    hit_reward_bonus: float = 5.0,
    damage_dealt_bonus: float = 0.0,
    damage_taken_penalty: float = 0.0,
    health_change_bonus: float = 1.0,
    armor_change_bonus: float = 0.5,
    episode_timeout: int = 21000,
    map_exit_reward: float = 1000.0,
    death_penalty: float = 100.0,
) -> gym.Env:
    map_id = map_id.upper()
    game, wad = resolve_game(map_id)
    if not 1 <= skill <= 5:
        raise ValueError(f"skill must be 1..5, got {skill}")

    env_id = f"Vizdoom{game}{map_id}-S{skill}-v0"
    extra = dict(
        max_buttons_pressed=1,  # Discrete actions, consistent with every other scenario
        episode_timeout=episode_timeout,
        map_exit_reward=map_exit_reward,
        death_penalty=death_penalty,
        audio_buffer_enabled=False,
        automap_buffer_enabled=False,
    )
    if wad is not None:
        extra["doom_game_path"] = str(wad)
        print(f"[doom_level] {map_id}: using commercial WAD {wad}")
    else:
        print(f"[doom_level] {map_id}: no commercial WAD in wads/ — using bundled {game}")

    return make_vizdoom_env(
        env_id,
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
        **extra,
    )
