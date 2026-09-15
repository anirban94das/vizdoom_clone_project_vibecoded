"""Shared training helpers used by every scenarios/train_*.py, via train_common.run_training."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


class OverwriteCheckpointCallback(BaseCallback):
    """Periodically saves the model to a single fixed path, overwriting it each time.

    Unlike stable_baselines3's CheckpointCallback, this never appends a step
    count to the filename, so exactly one file exists at save_path instead of
    accumulating one per save interval.
    """

    def __init__(self, save_freq: int, save_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)

    def _init_callback(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.model.save(self.save_path)
            if self.verbose >= 1:
                print(f"Saved model to {self.save_path} at {self.num_timesteps} timesteps")
        return True


class EpisodeRecapCallback(BaseCallback):
    """Collects per-episode reward (from SB3's own Monitor wrapper, via
    info["episode"]) alongside kills/hits/explored-cells/weapons-picked-up
    (from envs.common.EpisodeStatsWrapper, via info["episode_stats"]) for
    every episode finished during this run. At the end of training, compares
    the first few episodes seen against the last few - a plain rolling
    ep_rew_mean only shows a window of the last 100 episodes overall and
    says nothing about *this run's* trend specifically - then prints the
    comparison and appends one line to a persistent JSONL history file so
    past runs stay visible instead of being overwritten like the model file.
    """

    STAT_KEYS = (
        "reward",
        "kills",
        "hits",
        "damage_dealt",
        "damage_taken",
        "cells_explored",
        "weapons_picked_up",
    )

    def __init__(self, scenario: str, history_path: Path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.scenario = scenario
        self.history_path = Path(history_path)
        self._episodes: list[dict[str, float]] = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            stats = info.get("episode_stats")
            episode = info.get("episode")
            if stats is not None and episode is not None:
                self._episodes.append(
                    {
                        "reward": episode["r"],
                        "kills": stats["kills"],
                        "hits": stats["hits"],
                        "damage_dealt": stats["damage_dealt"],
                        "damage_taken": stats["damage_taken"],
                        "cells_explored": stats["cells_explored"],
                        "weapons_picked_up": stats["weapons_picked_up"],
                    }
                )
                # record_mean (not record) so this accumulates like SB3's own
                # rollout/ep_rew_mean and resets after each logger dump,
                # instead of only showing the last episode's value.
                self.logger.record_mean("rollout/ep_kills_mean", stats["kills"])
                self.logger.record_mean("rollout/ep_hits_mean", stats["hits"])
        return True

    def _on_training_end(self) -> None:
        n = len(self._episodes)
        if n == 0:
            print(f"\n[recap] {self.scenario}: no episodes finished this run - nothing to recap.")
            return

        window = min(20, max(1, n // 2))
        early, late = self._episodes[:window], self._episodes[-window:]

        def _mean(rows: list[dict[str, float]], key: str) -> float:
            return sum(r[key] for r in rows) / len(rows)

        summary = {
            "kind": "recap",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenario": self.scenario,
            "cumulative_timesteps": self.model.num_timesteps,
            "episodes_this_run": n,
        }
        for key in self.STAT_KEYS:
            summary[f"{key}_overall"] = _mean(self._episodes, key)
            summary[f"{key}_start"] = _mean(early, key)
            summary[f"{key}_end"] = _mean(late, key)

        print(f"\n[recap] {self.scenario} - {n} episodes this run "
              f"({self.model.num_timesteps} cumulative timesteps):")
        print("  overall averages (this run):")
        for key in self.STAT_KEYS:
            print(f"    {key:18s}: {summary[f'{key}_overall']:8.2f}")
        print(f"  trend (first {len(early)} vs last {len(late)} episodes):")
        for key in self.STAT_KEYS:
            print(f"    {key:18s}: {summary[f'{key}_start']:8.2f} -> {summary[f'{key}_end']:8.2f}")

        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "a") as f:
            f.write(json.dumps(summary) + "\n")
        print(f"[recap] appended to {self.history_path}")


def evaluate_unshaped(
    model,
    make_env_fn: Callable,
    env_kwargs: dict,
    n_stack: int = 4,
    n_episodes: int = 5,
) -> dict[str, float]:
    """Plays n_episodes deterministic episodes on a fresh single env built via
    make_env_fn(**env_kwargs) and returns the mean reward plus mean
    EpisodeStatsWrapper stats. Callers are expected to pass env_kwargs with
    every reward-shaping bonus forced to 0.0 (see train_common.ZERO_SHAPING_KWARGS),
    so the returned reward is the scenario's built-in score rather than the
    shaped training signal - the number that stays comparable across runs
    with different shaping, unlike ep_rew_mean.

    Builds and tears down its own env per call rather than caching one, since
    this only runs every eval_freq (tens of thousands of steps) - not worth
    holding a live DoomGame instance open between calls for. Shared by
    UnshapedEvalCallback (periodic, during a normal training run) and
    ablation.py (one final, larger-n_episodes call per knob-set).
    """
    vec_env = DummyVecEnv([lambda: make_env_fn(**env_kwargs)])
    if n_stack > 1:
        vec_env = VecFrameStack(vec_env, n_stack=n_stack)

    stat_keys = [k for k in EpisodeRecapCallback.STAT_KEYS if k != "reward"]
    rewards: list[float] = []
    stat_totals: dict[str, list[float]] = {k: [] for k in stat_keys}
    try:
        for _ in range(n_episodes):
            obs = vec_env.reset()
            done = False
            ep_reward = 0.0
            ep_stats: dict = {}
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, dones, infos = vec_env.step(action)
                ep_reward += float(reward[0])
                done = bool(dones[0])
                if done:
                    ep_stats = infos[0].get("episode_stats", {})
            rewards.append(ep_reward)
            for key in stat_keys:
                stat_totals[key].append(float(ep_stats.get(key, 0.0)))
    finally:
        vec_env.close()

    summary = {"mean_reward": sum(rewards) / len(rewards)}
    for key, values in stat_totals.items():
        summary[key] = sum(values) / len(values)
    return summary


class UnshapedEvalCallback(BaseCallback):
    """Every eval_freq real timesteps, plays n_eval_episodes deterministic
    episodes on a separate env with every reward-shaping knob forced to 0.0
    (via evaluate_unshaped), and logs the built-in score to TensorBoard
    (eval/*) plus one JSON line per evaluation to history_path.

    Exists because ep_rew_mean and EpisodeRecapCallback both measure the
    *shaped* reward (built-in + bonuses) - turning a knob moves that number
    even when the agent's actual in-game performance hasn't changed. Holding
    shaping at zero here makes eval/mean_reward_unshaped the number that only
    moves when behavior does, which is what ablation.py's final comparison
    also reads (via the same evaluate_unshaped helper).
    """

    def __init__(
        self,
        scenario: str,
        make_env_fn: Callable,
        unshaped_env_kwargs: dict,
        history_path: Path,
        eval_freq: int,
        n_eval_episodes: int = 5,
        n_stack: int = 4,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.scenario = scenario
        self.make_env_fn = make_env_fn
        self.unshaped_env_kwargs = unshaped_env_kwargs
        self.history_path = Path(history_path)
        # eval_freq is expected pre-divided by n_envs (n_calls units), same
        # convention as OverwriteCheckpointCallback.save_freq.
        self.eval_freq = max(eval_freq, 1)
        self.n_eval_episodes = n_eval_episodes
        self.n_stack = n_stack

    def _init_callback(self) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate()
        return True

    def _evaluate(self) -> None:
        result = evaluate_unshaped(
            self.model,
            self.make_env_fn,
            self.unshaped_env_kwargs,
            n_stack=self.n_stack,
            n_episodes=self.n_eval_episodes,
        )

        self.logger.record("eval/mean_reward_unshaped", result["mean_reward"])
        for key, value in result.items():
            if key != "mean_reward":
                self.logger.record(f"eval/{key}_unshaped", value)
        self.logger.dump(self.num_timesteps)

        print(
            f"[eval] {self.scenario} @ {self.num_timesteps} timesteps: "
            f"unshaped mean_reward={result['mean_reward']:.2f} "
            f"over {self.n_eval_episodes} episodes"
        )

        summary = {
            "kind": "unshaped_eval",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "scenario": self.scenario,
            "cumulative_timesteps": int(self.num_timesteps),
            "n_eval_episodes": self.n_eval_episodes,
        }
        summary["mean_reward_unshaped"] = result["mean_reward"]
        for key, value in result.items():
            if key != "mean_reward":
                summary[f"{key}_unshaped"] = value

        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_path, "a") as f:
            f.write(json.dumps(summary) + "\n")
