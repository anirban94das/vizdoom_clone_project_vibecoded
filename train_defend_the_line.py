"""Train a PPO agent on ViZDoom's defend_the_line.wad scenario.

Same fixed-position turret setup as defend_the_center (TURN_LEFT/TURN_RIGHT/
ATTACK only) but enemies approach from a line in front, and there's no
episode_timeout — episodes end on death. Fresh CnnPolicy on first run; no
earlier baseline to warm-start from. See train_common.run_training for the
shared auto-resume/checkpoint/recap behavior, and CLAUDE.md for the
don't-run-two-training-scripts-at-once warning (8 physical cores).
"""

from train_common import build_parser, reward_kwargs_from_args, run_training
from envs.defend_the_line_env import make_defend_the_line_env

TOTAL_TIMESTEPS = 300_000
MODEL_PATH = "models/latest/ppo_defend_the_line.zip"

REWARD_DEFAULTS = {
    "kill_reward_bonus": 20.0,
    "hit_reward_bonus": 5.0,
}


def main() -> None:
    args = build_parser(REWARD_DEFAULTS).parse_args()
    run_training(
        make_env_fn=make_defend_the_line_env,
        env_kwargs=reward_kwargs_from_args(args),
        scenario="ppo_defend_the_line",
        model_path=MODEL_PATH,
        total_timesteps=TOTAL_TIMESTEPS,
        args=args,
    )


if __name__ == "__main__":
    main()
