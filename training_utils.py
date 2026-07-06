"""Shared training helpers used by both train_basic.py and train_deadly_corridor.py."""

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
