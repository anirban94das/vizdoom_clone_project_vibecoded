"""Watch the agent play defend_the_center.wad live using the latest saved model.

Reloads models/latest/ppo_defend_the_center.zip before every episode (see
train_common.run_watch), so it can run alongside train_defend_the_center.py
to show behavior updating live.
"""

import _bootstrap  # noqa: F401  -- prepends the repo root to sys.path

from train_common import run_watch
from envs.defend_the_center_env import make_defend_the_center_env

if __name__ == "__main__":
    run_watch(
        make_defend_the_center_env, "models/latest/ppo_defend_the_center.zip"
    )
