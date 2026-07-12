"""Train a PPO agent on ViZDoom's my_way_home.wad scenario.

Sparse-reward maze navigation: +1 for finding the vest, -0.0001/tic
otherwise, random starting room. exploration_bonus_per_cell=1.0 (this
scenario's shaping default, see envs/my_way_home_env.py) is doing the heavy
lifting early on — without it the built-in reward gives PPO essentially
nothing to climb until a rollout reaches the vest by chance. See
train_common.run_training for the shared auto-resume/checkpoint/recap
behavior.
"""

from train_common import build_parser, reward_kwargs_from_args, run_training
from envs.my_way_home_env import make_my_way_home_env

TOTAL_TIMESTEPS = 300_000
MODEL_PATH = "models/latest/ppo_my_way_home.zip"

REWARD_DEFAULTS = {
    "exploration_bonus_per_cell": 1.0,
    "exploration_cell_size": 32.0,
}


def main() -> None:
    args = build_parser(REWARD_DEFAULTS).parse_args()
    run_training(
        make_env_fn=make_my_way_home_env,
        env_kwargs=reward_kwargs_from_args(args),
        scenario="ppo_my_way_home",
        model_path=MODEL_PATH,
        total_timesteps=TOTAL_TIMESTEPS,
        args=args,
    )


if __name__ == "__main__":
    main()
