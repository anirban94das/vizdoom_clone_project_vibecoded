"""Watch the agent play defend_the_line.wad live using the latest saved model.

Reloads models/latest/ppo_defend_the_line.zip before every episode (see
train_common.run_watch), so it can run alongside train_defend_the_line.py to
show behavior updating live.
"""

from train_common import run_watch
from envs.defend_the_line_env import make_defend_the_line_env

if __name__ == "__main__":
    run_watch(make_defend_the_line_env, "models/latest/ppo_defend_the_line.zip")
