"""Watch the agent play health_gathering_supreme.wad live using the latest saved model.

Reloads models/latest/ppo_health_gathering_supreme.zip before every episode
(see train_common.run_watch), so it can run alongside
train_health_gathering_supreme.py to show behavior updating live.
"""

import _bootstrap  # noqa: F401  -- prepends the repo root to sys.path

from train_common import run_watch
from envs.health_gathering_supreme_env import make_health_gathering_supreme_env

if __name__ == "__main__":
    run_watch(
        make_health_gathering_supreme_env,
        "models/latest/ppo_health_gathering_supreme.zip",
    )
