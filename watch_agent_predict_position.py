"""Watch the agent play predict_position.wad live using the latest saved model.

Reloads models/latest/ppo_predict_position.zip before every episode (see
train_common.run_watch), so it can run alongside train_predict_position.py to
show behavior updating live.
"""

from train_common import run_watch
from envs.predict_position_env import make_predict_position_env

if __name__ == "__main__":
    run_watch(make_predict_position_env, "models/latest/ppo_predict_position.zip")
