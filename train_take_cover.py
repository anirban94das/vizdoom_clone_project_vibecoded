"""Train a PPO agent on ViZDoom's take_cover.wad scenario.

Pure dodging: MOVE_LEFT/MOVE_RIGHT only, fireballs incoming, +1/tic alive,
episode ends on death (no timeout). damage_taken_penalty=0.5 (this
scenario's shaping default, see envs/take_cover_env.py) distinguishes
near-misses from direct hits earlier than survival time alone can. See
train_common.run_training for the shared auto-resume/checkpoint/recap
behavior.
"""

from train_common import build_parser, reward_kwargs_from_args, run_training
from envs.take_cover_env import make_take_cover_env

TOTAL_TIMESTEPS = 300_000
MODEL_PATH = "models/latest/ppo_take_cover.zip"

REWARD_DEFAULTS = {
    "damage_taken_penalty": 0.5,
}


def main() -> None:
    args = build_parser(REWARD_DEFAULTS).parse_args()
    run_training(
        make_env_fn=make_take_cover_env,
        env_kwargs=reward_kwargs_from_args(args),
        scenario="ppo_take_cover",
        model_path=MODEL_PATH,
        total_timesteps=TOTAL_TIMESTEPS,
        args=args,
    )


if __name__ == "__main__":
    main()
