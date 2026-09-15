"""Reward-shaping ablation harness (future-enhancements item 2).

Runs one scenario from scratch, for a short fixed budget and a fixed seed,
once per named reward-shaping configuration ("knob set"), then compares the
resulting policies on the *unshaped* eval score from
training_utils.evaluate_unshaped (item 1) - not the shaped training reward,
which by construction moves whenever a knob does, whether or not behavior
actually improved.

Ablation runs never touch real training state: models are written under
models/ablation/<scenario>/<knob-set>.zip (always deleted and retrained from
scratch, never auto-resumed) and history goes to logs/ablation_history.jsonl
- models/latest/ and logs/training_history.jsonl are untouched.

Usage:
    python ablation.py --scenario deadly_corridor --timesteps 20000
    python ablation.py --scenario deadly_corridor --timesteps 20000 --ent-coef 0.01
    python ablation.py --scenario doom_E1M1 --timesteps 30000
    python ablation.py --scenario health_gathering --knobs my_knobsets.json

--target-kl defaults to 0.03 (matching train_common.build_parser's project-
wide default - a PPO policy-collapse guard, see train_common.py's module
docstring); --ent-coef defaults to 0.0 except deadly_corridor's real
train_*.py script, which uses 0.01.

Without --knobs, compares two configurations: "no_shaping" (every one of the
nine reward knobs at 0.0) against "scenario_defaults" (that scenario's
envs/*_env.py SHAPING_DEFAULTS) - the most basic and most useful ablation
question ("is shaping actually helping on this scenario at all?").

--knobs points to a JSON file of {name: {knob: value, ...}}, e.g.:
    {"baseline": {}, "high_kill": {"kill_reward_bonus": 50.0}}
Any of the nine knobs left unspecified in a set defaults to 0.0 (not to the
scenario's own defaults) - knob sets are absolute reward configurations, not
deltas on top of anything.

Do not run alongside a real train_*.py - like any train_common.run_training
call, this spawns N_ENVS SubprocVecEnv workers per knob-set (sequentially,
one knob-set at a time, so only one set of workers exists at once).
"""

import argparse
import importlib
import json
from datetime import datetime, timezone
from pathlib import Path

from stable_baselines3 import PPO

import model_io
import train_common
from training_utils import evaluate_unshaped


def _resolve_scenario(scenario: str, map_id: str | None, skill: int):
    """Returns (make_env_fn, shaping_defaults, policy, n_stack, static_env_kwargs).
    static_env_kwargs are non-shaping env-factory kwargs (only doom_<MAP>'s
    map_id/skill today) that every knob-set's run must also receive."""
    if scenario.startswith("doom_"):
        from envs.doom_level_env import SHAPING_DEFAULTS, make_doom_level_env

        resolved_map = (map_id or scenario[len("doom_") :]).upper()
        return (
            make_doom_level_env,
            dict(SHAPING_DEFAULTS),
            "CnnPolicy",
            4,
            {"map_id": resolved_map, "skill": skill},
        )

    if scenario not in model_io.SCENARIO_MODELS:
        known = ", ".join(sorted(model_io.SCENARIO_MODELS))
        raise ValueError(f"Unknown scenario {scenario!r}. Known: {known}, or doom_<MAP> (e.g. doom_E1M1).")

    # Every envs/<key>_env.py follows the same make_<key>_env / SHAPING_DEFAULTS
    # naming, so this avoids a third hand-maintained copy of what train_ui.py
    # and each train_*.py already declare.
    module = importlib.import_module(f"envs.{scenario}_env")
    make_env_fn = getattr(module, f"make_{scenario}_env")
    shaping_defaults = dict(getattr(module, "SHAPING_DEFAULTS", {}))
    policy = "MultiInputPolicy" if scenario == "basic_audio" else "CnnPolicy"
    n_stack = 1 if scenario == "basic_audio" else 4
    return make_env_fn, shaping_defaults, policy, n_stack, {}


def _load_knob_sets(knobs_path: Path | None, shaping_defaults: dict) -> dict[str, dict[str, float]]:
    zero = train_common.ZERO_SHAPING_KWARGS
    if knobs_path is None:
        return {
            "no_shaping": dict(zero),
            "scenario_defaults": {**zero, **shaping_defaults},
        }
    raw = json.loads(knobs_path.read_text())
    return {name: {**zero, **overrides} for name, overrides in raw.items()}


def _print_table(results: dict[str, dict[str, float]]) -> None:
    if not results:
        return
    columns = list(next(iter(results.values())).keys())
    name_width = max(len(name) for name in results) + 2
    header = f"{'knob set':<{name_width}}" + "".join(f"{c:>16}" for c in columns)
    print(header)
    print("-" * len(header))
    for name, values in sorted(results.items(), key=lambda kv: kv[1]["mean_reward"], reverse=True):
        row = f"{name:<{name_width}}" + "".join(f"{values[c]:>16.2f}" for c in columns)
        print(row)


def run_ablation(
    scenario: str,
    timesteps: int,
    eval_episodes: int,
    eval_freq: int,
    seed: int,
    ent_coef: float,
    target_kl: float | None,
    knobs_path: Path | None,
    map_id: str | None,
    skill: int,
) -> dict[str, dict[str, float]]:
    make_env_fn, shaping_defaults, policy, n_stack, static_env_kwargs = _resolve_scenario(
        scenario, map_id, skill
    )
    knob_sets = _load_knob_sets(knobs_path, shaping_defaults)
    unshaped_env_kwargs = {**static_env_kwargs, **train_common.ZERO_SHAPING_KWARGS}

    ablation_root = Path("models/ablation") / scenario
    ablation_root.mkdir(parents=True, exist_ok=True)
    history_path = Path("logs/ablation_history.jsonl")
    args_ns = argparse.Namespace(ent_coef=ent_coef, target_kl=target_kl)

    results: dict[str, dict[str, float]] = {}
    for name, shaping in knob_sets.items():
        model_path = ablation_root / f"{name}.zip"
        if model_path.exists():
            model_path.unlink()  # always train from scratch - no auto-resume for ablation runs

        print(f"\n=== ablation: {scenario} / {name} ({timesteps} steps, seed={seed}) ===")
        print(f"    shaping: {shaping}")
        train_common.run_training(
            make_env_fn=make_env_fn,
            env_kwargs={**static_env_kwargs, **shaping},
            scenario=f"ablation_{scenario}_{name}",
            model_path=model_path,
            total_timesteps=timesteps,
            args=args_ns,
            policy=policy,
            n_stack=n_stack,
            eval_freq=eval_freq,
            n_eval_episodes=min(3, eval_episodes),
            seed=seed,
            history_path=history_path,
        )

        model = PPO.load(model_path, device="cuda")
        final = evaluate_unshaped(
            model, make_env_fn, unshaped_env_kwargs, n_stack=n_stack, n_episodes=eval_episodes
        )
        results[name] = final
        print(f"[ablation] {name}: unshaped mean_reward={final['mean_reward']:.2f} "
              f"over {eval_episodes} final eval episodes")

    summary_dir = Path("logs/ablation")
    summary_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = summary_dir / f"{scenario}_{stamp}.json"
    summary_path.write_text(json.dumps({
        "scenario": scenario,
        "timesteps": timesteps,
        "seed": seed,
        "eval_episodes": eval_episodes,
        "knob_sets": knob_sets,
        "results": results,
    }, indent=2))

    print(f"\n=== {scenario}: unshaped-eval comparison ({eval_episodes} episodes each) ===")
    _print_table(results)
    print(f"\nSummary written to {summary_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenario", required=True,
                         help=f"one of {', '.join(sorted(model_io.SCENARIO_MODELS))}, or doom_<MAP>")
    parser.add_argument("--timesteps", type=int, default=20_000,
                         help="short from-scratch training budget per knob-set (default 20000)")
    parser.add_argument("--eval-episodes", type=int, default=10,
                         help="deterministic episodes for the final comparison (default 10)")
    parser.add_argument("--eval-freq", type=int, default=5_000,
                         help="periodic unshaped-eval interval during each ablation run (default 5000)")
    parser.add_argument("--seed", type=int, default=0,
                         help="fixed seed so knob-sets are compared under identical env/policy randomness")
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--target-kl", type=float, default=0.03,
                         help="PPO policy-collapse guard, aborts an update that drifts too far "
                              "from the rollout policy in one step (default 0.03, matching "
                              "train_common.build_parser's project-wide default)")
    parser.add_argument("--knobs", type=Path, default=None,
                         help="JSON file of {name: {knob: value, ...}}; default compares "
                              "no_shaping vs scenario_defaults")
    parser.add_argument("--map", default=None, help="doom_<MAP> scenarios only: overrides the map")
    parser.add_argument("--skill", type=int, default=3, help="doom_<MAP> scenarios only")
    args = parser.parse_args()

    run_ablation(
        scenario=args.scenario,
        timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        eval_freq=args.eval_freq,
        seed=args.seed,
        ent_coef=args.ent_coef,
        target_kl=args.target_kl,
        knobs_path=args.knobs,
        map_id=args.map,
        skill=args.skill,
    )


if __name__ == "__main__":
    main()
