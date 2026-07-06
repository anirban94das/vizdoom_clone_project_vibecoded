"""Small desktop UI to launch training and live-watching for a chosen ViZDoom scenario.

Wraps the existing train_basic.py / train_deadly_corridor.py and
watch_agent.py / watch_agent_deadly_corridor.py entry points as subprocesses
(they're unmodified — this is just a launcher). Runs the project's own
.venv interpreter so it doesn't matter which Python started this UI. Only
one training run is allowed at a time from this window, since
train_deadly_corridor.py's docstring already warns against running both
scripts simultaneously (each spawns N_ENVS SubprocVecEnv worker processes and
this machine has 8 physical cores). Watching runs as its own independent
subprocess (single-process DummyVecEnv, not SubprocVecEnv) and opens its own
foreground ViZDoom window, so it can run alongside training without that
concern.

Run with: .venv\\Scripts\\python.exe train_ui.py
"""

import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parent

LEVELS = {
    "Basic": "train_basic.py",
    "Deadly Corridor (shaped)": "train_deadly_corridor.py",
}

# watch_agent_*.py opens its own visible ViZDoom window (render_mode="human")
# and reloads that scenario's models/latest/*.zip before every episode, so it
# can run alongside training to show behavior updating live.
WATCH_SCRIPTS = {
    "Basic": "watch_agent.py",
    "Deadly Corridor (shaped)": "watch_agent_deadly_corridor.py",
}

# Reward-shaping knobs, one row per train_*.py --flag. Defaults mirror each
# script's argparse defaults (envs/basic_env.py's are all 0.0/off;
# train_deadly_corridor.py's match its existing hardcoded values) so leaving
# every field untouched reproduces today's behavior exactly.
REWARD_KNOBS = [
    ("kill_reward_bonus", "--kill-reward-bonus", "Kill bonus"),
    ("hit_reward_bonus", "--hit-reward-bonus", "Hit bonus"),
    ("exploration_bonus_per_cell", "--exploration-bonus-per-cell", "Exploration bonus / cell"),
    ("exploration_cell_size", "--exploration-cell-size", "Exploration cell size"),
    ("weapon_pickup_bonus", "--weapon-pickup-bonus", "Weapon pickup bonus"),
]

REWARD_DEFAULTS = {
    "Basic": {
        "kill_reward_bonus": 0.0,
        "hit_reward_bonus": 0.0,
        "exploration_bonus_per_cell": 0.0,
        "exploration_cell_size": 32.0,
        "weapon_pickup_bonus": 0.0,
    },
    "Deadly Corridor (shaped)": {
        "kill_reward_bonus": 20.0,
        "hit_reward_bonus": 5.0,
        "exploration_bonus_per_cell": 1.0,
        "exploration_cell_size": 32.0,
        "weapon_pickup_bonus": 15.0,
    },
}


def resolve_python() -> str:
    """Prefer this project's .venv interpreter over whatever launched the UI."""
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return str(venv_python) if venv_python.exists() else sys.executable


class TrainingLauncher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ViZDoom Training Launcher")
        self.geometry("720x480")

        self.python_exe = resolve_python()
        self.process: subprocess.Popen | None = None
        self.watch_process: subprocess.Popen | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()

        self._build_widgets()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._drain_output_queue)

    def _build_widgets(self) -> None:
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Level:").pack(side="left")
        self.level_var = tk.StringVar(value=next(iter(LEVELS)))
        level_dropdown = ttk.Combobox(
            top, textvariable=self.level_var, values=list(LEVELS), state="readonly", width=30
        )
        level_dropdown.pack(side="left", padx=8)
        level_dropdown.bind("<<ComboboxSelected>>", self._on_level_changed)

        self.start_button = ttk.Button(top, text="Start Training", command=self._start_training)
        self.start_button.pack(side="left", padx=4)

        self.stop_button = ttk.Button(
            top, text="Stop", command=self._stop_training, state="disabled"
        )
        self.stop_button.pack(side="left", padx=4)

        self.watch_button = ttk.Button(top, text="Watch Agent", command=self._start_watching)
        self.watch_button.pack(side="left", padx=(16, 4))

        self.stop_watch_button = ttk.Button(
            top, text="Stop Watching", command=self._stop_watching, state="disabled"
        )
        self.stop_watch_button.pack(side="left", padx=4)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(top, textvariable=self.status_var).pack(side="right")

        self.watch_status_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.watch_status_var).pack(side="right", padx=(0, 12))

        rewards_frame = ttk.LabelFrame(self, text="Reward shaping", padding=10)
        rewards_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.reward_vars: dict[str, tk.StringVar] = {}
        for col, (key, _flag, label) in enumerate(REWARD_KNOBS):
            ttk.Label(rewards_frame, text=label).grid(row=0, column=col, padx=6, sticky="w")
            var = tk.StringVar()
            ttk.Entry(rewards_frame, textvariable=var, width=10).grid(
                row=1, column=col, padx=6
            )
            self.reward_vars[key] = var
        self._on_level_changed()

        log_frame = ttk.Frame(self, padding=(10, 0, 10, 10))
        log_frame.pack(fill="both", expand=True)

        self.log_text = tk.Text(log_frame, state="disabled", wrap="none", bg="black", fg="lightgreen")
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

    def _on_level_changed(self, _event=None) -> None:
        """Reset reward fields to the newly selected level's defaults."""
        defaults = REWARD_DEFAULTS[self.level_var.get()]
        for key, var in self.reward_vars.items():
            var.set(str(defaults[key]))

    def _reward_args(self) -> list[str]:
        """Read the reward fields and turn them into train_*.py CLI flags.
        Raises ValueError (with a field name in the message) on bad input."""
        args = []
        for key, flag, label in REWARD_KNOBS:
            raw = self.reward_vars[key].get()
            try:
                value = float(raw)
            except ValueError:
                raise ValueError(f"{label!r} must be a number, got {raw!r}") from None
            args += [flag, str(value)]
        return args

    def _start_training(self) -> None:
        if self.process is not None:
            return

        try:
            reward_args = self._reward_args()
        except ValueError as exc:
            messagebox.showerror("Invalid reward value", str(exc))
            return

        script = LEVELS[self.level_var.get()]
        command = [self.python_exe, script, *reward_args]
        self._append_log(f"$ {' '.join(command)}\n")

        self.process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        threading.Thread(target=self._read_process_output, daemon=True).start()

        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.status_var.set(f"Training: {self.level_var.get()}")

    def _read_process_output(self) -> None:
        assert self.process is not None and self.process.stdout is not None
        for line in self.process.stdout:
            self.output_queue.put(line)
        self.process.wait()
        self.output_queue.put("__PROCESS_DONE__")

    def _drain_output_queue(self) -> None:
        try:
            while True:
                line = self.output_queue.get_nowait()
                if line == "__PROCESS_DONE__":
                    self._on_process_done()
                elif line == "__WATCH_DONE__":
                    self._on_watch_process_done()
                else:
                    self._append_log(line)
        except queue.Empty:
            pass
        self.after(100, self._drain_output_queue)

    def _on_process_done(self) -> None:
        self._append_log("\n[process exited]\n")
        self.process = None
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status_var.set("Idle")

    def _stop_training(self) -> None:
        if self.process is None:
            return
        # taskkill /T kills the whole process tree — needed because
        # train_*.py itself spawns SubprocVecEnv worker processes that a
        # plain terminate()/Ctrl+C on just the parent PID would orphan.
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
            capture_output=True,
        )
        self._append_log("\n[stop requested]\n")

    def _start_watching(self) -> None:
        """Launch watch_agent_*.py for the selected level as a foreground
        subprocess — it opens its own visible ViZDoom window and reloads
        that scenario's models/latest/*.zip before every episode, so it can
        run alongside (or independently of) training to show live behavior."""
        if self.watch_process is not None:
            return

        level = self.level_var.get()
        script = WATCH_SCRIPTS[level]
        command = [self.python_exe, script]
        self._append_log(f"$ {' '.join(command)}\n")

        self.watch_process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        threading.Thread(target=self._read_watch_process_output, daemon=True).start()

        self.stop_watch_button.configure(state="normal")
        self.watch_status_var.set(f"Watching: {level}")

    def _read_watch_process_output(self) -> None:
        assert self.watch_process is not None and self.watch_process.stdout is not None
        for line in self.watch_process.stdout:
            self.output_queue.put(f"[watch] {line}")
        self.watch_process.wait()
        self.output_queue.put("__WATCH_DONE__")

    def _on_watch_process_done(self) -> None:
        self._append_log("\n[watch process exited]\n")
        self.watch_process = None
        self.stop_watch_button.configure(state="disabled")
        self.watch_status_var.set("")

    def _stop_watching(self) -> None:
        if self.watch_process is None:
            return
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(self.watch_process.pid)],
            capture_output=True,
        )
        self._append_log("\n[stop watching requested]\n")

    def _append_log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _on_close(self) -> None:
        if self.process is not None:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(self.process.pid)],
                capture_output=True,
            )
        if self.watch_process is not None:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(self.watch_process.pid)],
                capture_output=True,
            )
        self.destroy()


if __name__ == "__main__":
    TrainingLauncher().mainloop()
