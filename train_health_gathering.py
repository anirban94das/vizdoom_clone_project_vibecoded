"""Train a PPO agent on ViZDoom's health_gathering.wad scenario.

Survival on an acid floor: collect medkits, live as long as possible.
Built-in reward is +1/tic alive; health_change_bonus=1.0 (this scenario's
shaping default, see envs/health_gathering_env.py) makes medkit pickups an
explicit signal. First combat-free scenario in the roadmap — the combat
bonuses are accepted as flags but default off. See train_common.run_training
for the shared auto-resume/checkpoint/recap behavior.
"""

from train_common import build_parser, reward_kwargs_from_args, run_training
from envs.health_gathering_env import make_health_gathering_env

TOTAL_TIMESTEPS = 300_000
MODEL_PATH = "models/latest/ppo_health_gathering.zip"

REWARD_DEFAULTS = {
    "health_change_bonus": 1.0,
}


def main() -> None:
    args = build_parser(REWARD_DEFAULTS).parse_args()
    run_training(
        make_env_fn=make_health_gathering_env,
        env_kwargs=reward_kwargs_from_args(args),
        scenario="ppo_health_gathering",
        model_path=MODEL_PATH,
        total_timesteps=TOTAL_TIMESTEPS,
        args=args,
    )


if __name__ == "__main__":
    main()
