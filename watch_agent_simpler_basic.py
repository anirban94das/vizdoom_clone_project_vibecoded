"""Watch the agent play simpler_basic.wad live using the latest saved model.

Reloads models/latest/ppo_simpler_basic.zip before every episode (see
train_common.run_watch).
"""

from train_common import run_watch
from envs.simpler_basic_env import make_simpler_basic_env

if __name__ == "__main__":
    run_watch(make_simpler_basic_env, "models/latest/ppo_simpler_basic.zip")
