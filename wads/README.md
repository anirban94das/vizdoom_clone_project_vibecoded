# wads/ — optional commercial game WADs

Full-level training (`train_doom_level.py`, `envs/doom_level_env.py`) plays
real DOOM maps. By default it uses the **Freedoom** WADs bundled with the
installed `vizdoom` package — free, BSD-licensed replacements with the same
map slots and monster/weapon behavior but different level layouts and art.
Nothing to download; it works out of the box.

If you own the original games (e.g. on Steam), drop the IWADs here to train
on the real levels instead:

| File (exact name)  | Game            | Used for maps    |
|--------------------|-----------------|------------------|
| `wads/doom.wad`    | DOOM (1993)     | `E1M1`..`E4M9`   |
| `wads/doom2.wad`   | DOOM II (1994)  | `MAP01`..`MAP32` |

Steam locations, typically:
- `steamapps/common/Ultimate Doom/base/DOOM.WAD` → rename to `doom.wad`
- `steamapps/common/Doom 2/base/DOOM2.WAD` → rename to `doom2.wad`

Detection happens per run: if the file exists it's used, otherwise Freedoom.
The model file per map is the same either way (`models/latest/ppo_doom_<MAP>.zip`),
so swapping WADs mid-training just changes the art/level layout the agent
sees from then on — for DOOM vs Freedoom the map slot names match but the
actual level geometry differs, so expect a reward dip after a swap.

These files are commercial game data — don't commit them to the repo.
