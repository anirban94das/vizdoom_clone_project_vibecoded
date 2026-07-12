"""Watch the agent play rocket_basic.wad live using the latest saved model.

Reloads models/latest/ppo_rocket_basic.zip before every episode (see
train_common.run_watch).
"""

from train_common import run_watch
from envs.rocket_basic_env import make_rocket_basic_env

if __name__ == "__main__":
    run_watch(make_rocket_basic_env, "models/latest/ppo_rocket_basic.zip")
