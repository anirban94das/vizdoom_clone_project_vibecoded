"""Train a PPO agent on ViZDoom's simpler_basic.wad scenario.

Gentler basic.wad variant; useful as a pipeline smoke test. Built-in reward
suffices, shaping defaults all off. See train_common.run_training for the
shared auto-resume/checkpoint/recap behavior.
"""

from train_common import build_parser, reward_kwargs_from_args, run_training
from envs.simpler_basic_env import make_simpler_basic_env

TOTAL_TIMESTEPS = 100_000
MODEL_PATH = "models/latest/ppo_simpler_basic.zip"

REWARD_DEFAULTS: dict[str, float] = {}


def main() -> None:
    args = build_parser(REWARD_DEFAULTS).parse_args()
    run_training(
        make_env_fn=make_simpler_basic_env,
        env_kwargs=reward_kwargs_from_args(args),
        scenario="ppo_simpler_basic",
        model_path=MODEL_PATH,
        total_timesteps=TOTAL_TIMESTEPS,
        args=args,
    )


if __name__ == "__main__":
    main()
