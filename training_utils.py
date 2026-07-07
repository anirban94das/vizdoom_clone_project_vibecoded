"""Shared training helpers used by both train_basic.py and train_deadly_corridor.py."""

import json
from datetime import datetime, timezone
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


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
