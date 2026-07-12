"""Watch the agent play a full DOOM / DOOM II level using the latest saved model.

Takes the same --map/--skill flags as train_doom_level.py and reloads that
map's models/latest/ppo_doom_<MAP>.zip before every episode (see
train_common.run_watch).
"""

import argparse

from train_common import run_watch
from envs.doom_level_env import make_doom_level_env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map", default="E1M1", help="E1M1..E4M9 or MAP01..MAP32")
    parser.add_argument("--skill", type=int, default=3, help="Doom skill 1-5")
    args = parser.parse_args()

    map_id = args.map.upper()
    run_watch(
        make_doom_level_env,
        f"models/latest/ppo_doom_{map_id}.zip",
        env_kwargs={"map_id": map_id, "skill": args.skill},
    )


if __name__ == "__main__":
    main()
