"""Train a PPO agent on ViZDoom's basic_audio.wad scenario (screen + audio).

The only scenario using a Dict observation: {screen (84,84,1), audio
(flattened waveform)} — see envs/basic_audio_env.py. Two deviations from
every other train_*.py, both passed through train_common.run_training's
params rather than a separate code path:

- policy="MultiInputPolicy": SB3's CombinedExtractor runs its NatureCNN on
  the screen entry and a flatten+MLP on the audio entry, then concatenates.
- n_stack=1 (no VecFrameStack): the audio buffer already spans the whole
  frame_skip window, so it carries the temporal signal that frame-stacking
  provides elsewhere; SB3's VecFrameStack over a Dict with a 1-D audio
  entry would stack it into a shape the extractor doesn't expect anyway.

CAVEAT: needs a working OpenAL audio device — DoomGame.init() raises
otherwise. Untested on this machine; see envs/basic_audio_env.py.
"""

from train_common import build_parser, reward_kwargs_from_args, run_training
from envs.basic_audio_env import make_basic_audio_env

TOTAL_TIMESTEPS = 100_000
MODEL_PATH = "models/latest/ppo_basic_audio.zip"

REWARD_DEFAULTS: dict[str, float] = {}


def main() -> None:
    args = build_parser(REWARD_DEFAULTS).parse_args()
    run_training(
        make_env_fn=make_basic_audio_env,
        env_kwargs=reward_kwargs_from_args(args),
        scenario="ppo_basic_audio",
        model_path=MODEL_PATH,
        total_timesteps=TOTAL_TIMESTEPS,
        args=args,
        policy="MultiInputPolicy",
        n_stack=1,
    )


if __name__ == "__main__":
    main()
