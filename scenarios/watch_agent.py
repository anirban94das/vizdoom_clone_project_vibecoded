"""Watch the agent play basic.wad live using the latest saved model.

Reloads models/latest/ppo_basic.zip before every episode (see
train_common.run_watch), so it can run alongside train_basic.py (separate
terminal, same venv) to show behavior updating as training progresses.
"""

import _bootstrap  # noqa: F401  -- prepends the repo root to sys.path

from train_common import run_watch
from envs.basic_env import make_basic_env

if __name__ == "__main__":
    run_watch(make_basic_env, "models/latest/ppo_basic.zip")
