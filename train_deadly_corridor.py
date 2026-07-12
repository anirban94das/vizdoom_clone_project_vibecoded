"""Train a PPO agent on ViZDoom's deadly_corridor.wad scenario.

Trains and logs under the "ppo_deadly_corridor_shaped" identity (not the older
"ppo_deadly_corridor" baseline prefix) since reward shaping is on by default
— a different reward function than the baseline run it evolved from. If
models/latest/ppo_deadly_corridor_shaped.zip doesn't exist yet, it
warm-starts weights from models/ppo_deadly_corridor.zip (the baseline run's
final saved model): visual features/aiming/movement carry over, but the
timestep/TensorBoard counter resets since the reward scale underneath
changed — expect a visible jump/dip in the reward curve at that handoff.

--ent-coef / --target-kl default to 0.01 / 0.03 here (unlike the other
scenarios' SB3 defaults of 0.0/None): guards against PPO's policy-collapse
failure mode seen on this scenario (reward crashed from +340 to -46,000
around step 1.6M and never recovered). ent_coef keeps a floor of
exploration; target_kl aborts an update that would change the policy too
much in one step, the suspected collapse mechanism.

See train_common.run_training for the shared auto-resume/checkpoint/recap
behavior. Do not run alongside another train_*.py (8 physical cores).
"""

from train_common import build_parser, reward_kwargs_from_args, run_training
from envs.deadly_corridor_env import make_deadly_corridor_env

# Harder, sparser-reward scenario (doom_skill=5, must navigate under fire)
# than basic.wad's ~100k-step convergence.
TOTAL_TIMESTEPS = 300_000
MODEL_PATH = "models/latest/ppo_deadly_corridor_shaped.zip"
# Pre-reward-shaping baseline run's final saved model — used only as a
# one-time warm start if MODEL_PATH doesn't exist yet.
WARM_START_PATH = "models/ppo_deadly_corridor.zip"

REWARD_DEFAULTS = {
    "kill_reward_bonus": 20.0,
    "hit_reward_bonus": 5.0,
    "exploration_bonus_per_cell": 1.0,
    "exploration_cell_size": 32.0,
    "weapon_pickup_bonus": 15.0,
}


def main() -> None:
    args = build_parser(REWARD_DEFAULTS, ent_coef=0.01, target_kl=0.03).parse_args()
    run_training(
        make_env_fn=make_deadly_corridor_env,
        env_kwargs=reward_kwargs_from_args(args),
        scenario="ppo_deadly_corridor_shaped",
        model_path=MODEL_PATH,
        total_timesteps=TOTAL_TIMESTEPS,
        args=args,
        warm_start_path=WARM_START_PATH,
    )


if __name__ == "__main__":
    main()
