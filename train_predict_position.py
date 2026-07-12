"""Train a PPO agent on ViZDoom's predict_position.wad scenario.

Aim-leading with a rocket launcher: one slow projectile, a moving target,
300-tic episodes, no autoaim. Kill/hit bonuses default to 100/25 — an order
of magnitude above the hitscan scenarios — because success happens at most
once per episode and has to dominate that episode's return (untested
starting point, tune via flags). See train_common.run_training for the
shared auto-resume/checkpoint/recap behavior.
"""

from train_common import build_parser, reward_kwargs_from_args, run_training
from envs.predict_position_env import make_predict_position_env

TOTAL_TIMESTEPS = 300_000
MODEL_PATH = "models/latest/ppo_predict_position.zip"

REWARD_DEFAULTS = {
    "kill_reward_bonus": 100.0,
    "hit_reward_bonus": 25.0,
}


def main() -> None:
    args = build_parser(REWARD_DEFAULTS).parse_args()
    run_training(
        make_env_fn=make_predict_position_env,
        env_kwargs=reward_kwargs_from_args(args),
        scenario="ppo_predict_position",
        model_path=MODEL_PATH,
        total_timesteps=TOTAL_TIMESTEPS,
        args=args,
    )


if __name__ == "__main__":
    main()
