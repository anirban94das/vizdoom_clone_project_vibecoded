"""Watch the agent play basic_audio.wad live using the latest saved model.

Reloads models/latest/ppo_basic_audio.zip before every episode. n_stack=1 to
match train_basic_audio.py's dict observation (no frame stacking — the audio
buffer carries the temporal signal; see envs/basic_audio_env.py).
"""

from train_common import run_watch
from envs.basic_audio_env import make_basic_audio_env

if __name__ == "__main__":
    run_watch(make_basic_audio_env, "models/latest/ppo_basic_audio.zip", n_stack=1)
