"""Watch the agent play my_way_home.wad live using the latest saved model.

Reloads models/latest/ppo_my_way_home.zip before every episode (see
train_common.run_watch), so it can run alongside train_my_way_home.py to
show behavior updating live.
"""

from train_common import run_watch
from envs.my_way_home_env import make_my_way_home_env

if __name__ == "__main__":
    run_watch(make_my_way_home_env, "models/latest/ppo_my_way_home.zip")
