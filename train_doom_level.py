"""Train a PPO agent on a full DOOM / DOOM II game level.

Unlike the scenario scripts, the level is a flag: --map E1M1 (DOOM episode
maps E1M1..E4M9) or --map MAP01 (DOOM II maps MAP01..MAP32), plus --skill
1-5 (default 3). Each map gets its own model file, TensorBoard run, and
recap identity (ppo_doom_<MAP>), so training E1M1 doesn't touch MAP01's
checkpoint. WAD selection (commercial from wads/ vs. bundled Freedoom) and
the full-game reward setup live in envs/doom_level_env.py.

Fair warning: a full DOOM map is a much harder RL problem than any scenario
— long horizon, keys/doors, mixed enemies. 500k steps per invocation is a
starting point; expect to lean on auto-resume repeatedly, and treat the
exploration/exit-reward defaults as the first thing to tune.
"""

from train_common import build_parser, reward_kwargs_from_args, run_training
from envs.doom_level_env import make_doom_level_env

TOTAL_TIMESTEPS = 500_000

REWARD_DEFAULTS = {
    "kill_reward_bonus": 20.0,
    "hit_reward_bonus": 5.0,
    "exploration_bonus_per_cell": 1.0,
    "exploration_cell_size": 32.0,
    "weapon_pickup_bonus": 15.0,
    "health_change_bonus": 1.0,
    "armor_change_bonus": 0.5,
}


def main() -> None:
    parser = build_parser(REWARD_DEFAULTS)
    parser.add_argument("--map", default="E1M1", help="E1M1..E4M9 or MAP01..MAP32")
    parser.add_argument("--skill", type=int, default=3, help="Doom skill 1-5 (3 = Hurt me plenty)")
    args = parser.parse_args()

    map_id = args.map.upper()
    env_kwargs = {**reward_kwargs_from_args(args), "map_id": map_id, "skill": args.skill}
    run_training(
        make_env_fn=make_doom_level_env,
        env_kwargs=env_kwargs,
        scenario=f"ppo_doom_{map_id}",
        model_path=f"models/latest/ppo_doom_{map_id}.zip",
        total_timesteps=TOTAL_TIMESTEPS,
        args=args,
    )


if __name__ == "__main__":
    main()
