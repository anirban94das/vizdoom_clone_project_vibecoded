"""Watch the agent play take_cover.wad live using the latest saved model.

Reloads models/latest/ppo_take_cover.zip before every episode (see
train_common.run_watch), so it can run alongside train_take_cover.py to show
behavior updating live.
"""

from train_common import run_watch
from envs.take_cover_env import make_take_cover_env

if __name__ == "__main__":
    run_watch(make_take_cover_env, "models/latest/ppo_take_cover.zip")
