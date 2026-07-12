"""Train a PPO agent on ViZDoom's basic.wad scenario.

basic.wad's built-in reward (shoot the monster: ~+101, miss: -5, living: -1)
is already sufficient, so every reward-shaping bonus defaults to 0.0 (off) —
the flags exist for experimentation. See train_common.run_training for the
shared auto-resume/checkpoint/recap behavior; this scenario's body used to
live here until the scenario count made the copy-paste untenable.

Do not run alongside another train_*.py — each spawns N_ENVS SubprocVecEnv
workers, and this machine has 8 physical cores.
"""

from train_common import build_parser, reward_kwargs_from_args, run_training
from envs.basic_env import make_basic_env

# basic.wad converges in ~100k steps; harder scenarios use 300k.
TOTAL_TIMESTEPS = 100_000
MODEL_PATH = "models/latest/ppo_basic.zip"

REWARD_DEFAULTS: dict[str, float] = {}  # all bonuses off — built-in reward suffices


def main() -> None:
    args = build_parser(REWARD_DEFAULTS).parse_args()
    run_training(
        make_env_fn=make_basic_env,
        env_kwargs=reward_kwargs_from_args(args),
        scenario="ppo_basic",
        model_path=MODEL_PATH,
        total_timesteps=TOTAL_TIMESTEPS,
        args=args,
    )


if __name__ == "__main__":
    main()
