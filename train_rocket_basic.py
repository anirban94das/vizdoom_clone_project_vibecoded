"""Train a PPO agent on ViZDoom's rocket_basic.wad scenario.

basic.wad with a rocket launcher and no autoaim — the slow projectile makes
timing/leading matter even though the monster spawns on one wall. Built-in
reward suffices, so shaping defaults are all off, same as basic. See
train_common.run_training for the shared auto-resume/checkpoint/recap
behavior.
"""

from train_common import build_parser, reward_kwargs_from_args, run_training
from envs.rocket_basic_env import make_rocket_basic_env

TOTAL_TIMESTEPS = 100_000
MODEL_PATH = "models/latest/ppo_rocket_basic.zip"

REWARD_DEFAULTS: dict[str, float] = {}


def main() -> None:
    args = build_parser(REWARD_DEFAULTS).parse_args()
    run_training(
        make_env_fn=make_rocket_basic_env,
        env_kwargs=reward_kwargs_from_args(args),
        scenario="ppo_rocket_basic",
        model_path=MODEL_PATH,
        total_timesteps=TOTAL_TIMESTEPS,
        args=args,
    )


if __name__ == "__main__":
    main()
