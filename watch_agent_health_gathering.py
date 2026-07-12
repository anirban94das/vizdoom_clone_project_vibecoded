"""Watch the agent play health_gathering.wad live using the latest saved model.

Reloads models/latest/ppo_health_gathering.zip before every episode (see
train_common.run_watch), so it can run alongside train_health_gathering.py to
show behavior updating live.
"""

from train_common import run_watch
from envs.health_gathering_env import make_health_gathering_env

if __name__ == "__main__":
    run_watch(make_health_gathering_env, "models/latest/ppo_health_gathering.zip")
