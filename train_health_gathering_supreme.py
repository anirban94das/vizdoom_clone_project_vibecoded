"""Train a PPO agent on ViZDoom's health_gathering_supreme.wad scenario.

Same task as health_gathering on a harder maze map. Warm-starts from the
plain health_gathering model if this scenario has no checkpoint yet — same
action space, same objective, transferable visual features — following the
same warm-start pattern deadly_corridor used from its unshaped baseline
(weights carry over, timestep/TensorBoard counters reset). Train
health_gathering first for that transfer to kick in; otherwise this starts
from a fresh CnnPolicy.
"""

from train_common import build_parser, reward_kwargs_from_args, run_training
from envs.health_gathering_supreme_env import make_health_gathering_supreme_env

TOTAL_TIMESTEPS = 300_000
MODEL_PATH = "models/latest/ppo_health_gathering_supreme.zip"
WARM_START_PATH = "models/latest/ppo_health_gathering.zip"

REWARD_DEFAULTS = {
    "health_change_bonus": 1.0,
}


def main() -> None:
    args = build_parser(REWARD_DEFAULTS).parse_args()
    run_training(
        make_env_fn=make_health_gathering_supreme_env,
        env_kwargs=reward_kwargs_from_args(args),
        scenario="ppo_health_gathering_supreme",
        model_path=MODEL_PATH,
        total_timesteps=TOTAL_TIMESTEPS,
        args=args,
        warm_start_path=WARM_START_PATH,
    )


if __name__ == "__main__":
    main()
