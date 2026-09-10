"""Watch the agent play deadly_corridor.wad live using the latest saved model.

Points at models/latest/ppo_deadly_corridor_shaped.zip — the file
train_deadly_corridor.py's reward-shaped run overwrites on a schedule — and
reloads it before every episode (see train_common.run_watch), so it can run
alongside training to show behavior updating live.
"""

import _bootstrap  # noqa: F401  -- prepends the repo root to sys.path

from train_common import run_watch
from envs.deadly_corridor_env import make_deadly_corridor_env

if __name__ == "__main__":
    run_watch(
        make_deadly_corridor_env, "models/latest/ppo_deadly_corridor_shaped.zip"
    )
