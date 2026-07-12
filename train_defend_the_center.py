"""Train a PPO agent on ViZDoom's defend_the_center.wad scenario.

The player is fixed at the center of the room (only TURN_LEFT/TURN_RIGHT/
ATTACK — see defend_the_center.cfg), so exploration_bonus_per_cell defaults
to 0.0 and should stay off: standing still is the objective here, not a
failure mode to discourage like in deadly_corridor. No earlier unshaped
baseline to warm-start from — first run trains a fresh CnnPolicy. See
train_common.run_training for the shared auto-resume/checkpoint/recap
behavior.
"""

from train_common import build_parser, reward_kwargs_from_args, run_training
from envs.defend_the_center_env import make_defend_the_center_env

TOTAL_TIMESTEPS = 300_000
MODEL_PATH = "models/latest/ppo_defend_the_center.zip"

REWARD_DEFAULTS = {
    "kill_reward_bonus": 20.0,
    "hit_reward_bonus": 5.0,
}


def main() -> None:
    args = build_parser(REWARD_DEFAULTS).parse_args()
    run_training(
        make_env_fn=make_defend_the_center_env,
        env_kwargs=reward_kwargs_from_args(args),
        scenario="ppo_defend_the_center",
        model_path=MODEL_PATH,
        total_timesteps=TOTAL_TIMESTEPS,
        args=args,
    )


if __name__ == "__main__":
    main()
