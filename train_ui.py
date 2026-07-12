"""Small desktop UI to launch training and live-watching for a chosen ViZDoom scenario.

Wraps the train_*.py / watch_agent_*.py entry points (all 14 levels, incl.
the full Doom E1M1 / Doom II MAP01 levels via train_doom_level.py --map) and
export_model.py / import_model.py as subprocesses (they're unmodified — this
is just a launcher). Runs the project's own .venv interpreter so it doesn't
matter which Python started this UI. Only one training run is allowed at a
time from this window, since the train scripts warn against running two
simultaneously (each spawns N_ENVS SubprocVecEnv worker processes and this
machine has 8 physical cores). Watching runs as its own independent
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
from tkinter import filedialog, messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parent

# Level -> the command (script + any extra args) to launch training. Values
# are argument lists (not bare script names) because the full-level entries
# share one script parameterized by --map.
LEVELS = {
    "Basic": ["train_basic.py"],
    "Simpler Basic": ["train_simpler_basic.py"],
    "Rocket Basic": ["train_rocket_basic.py"],
    "Basic Audio (screen+sound)": ["train_basic_audio.py"],
    "Deadly Corridor (shaped)": ["train_deadly_corridor.py"],
    "Defend the Center": ["train_defend_the_center.py"],
    "Defend the Line": ["train_defend_the_line.py"],
    "Health Gathering": ["train_health_gathering.py"],
    "Health Gathering Supreme": ["train_health_gathering_supreme.py"],
    "My Way Home": ["train_my_way_home.py"],
    "Predict Position": ["train_predict_position.py"],
    "Take Cover": ["train_take_cover.py"],
    "Doom E1M1 (full level)": ["train_doom_level.py", "--map", "E1M1"],
    "Doom II MAP01 (full level)": ["train_doom_level.py", "--map", "MAP01"],
}

# watch_agent_*.py opens its own visible ViZDoom window (render_mode="human")
# and reloads that scenario's models/latest/*.zip before every episode, so it
# can run alongside training to show behavior updating live.
WATCH_SCRIPTS = {
    "Basic": ["watch_agent.py"],
    "Simpler Basic": ["watch_agent_simpler_basic.py"],
    "Rocket Basic": ["watch_agent_rocket_basic.py"],
    "Basic Audio (screen+sound)": ["watch_agent_basic_audio.py"],
    "Deadly Corridor (shaped)": ["watch_agent_deadly_corridor.py"],
    "Defend the Center": ["watch_agent_defend_the_center.py"],
    "Defend the Line": ["watch_agent_defend_the_line.py"],
    "Health Gathering": ["watch_agent_health_gathering.py"],
    "Health Gathering Supreme": ["watch_agent_health_gathering_supreme.py"],
    "My Way Home": ["watch_agent_my_way_home.py"],
    "Predict Position": ["watch_agent_predict_position.py"],
    "Take Cover": ["watch_agent_take_cover.py"],
    "Doom E1M1 (full level)": ["watch_agent_doom_level.py", "--map", "E1M1"],
    "Doom II MAP01 (full level)": ["watch_agent_doom_level.py", "--map", "MAP01"],
}

# Level -> the scenario key export_model.py / import_model.py take (see
# model_io.SCENARIO_MODELS; doom_<MAP> keys resolve dynamically there).
SCENARIO_KEYS = {
    "Basic": "basic",
    "Simpler Basic": "simpler_basic",
    "Rocket Basic": "rocket_basic",
    "Basic Audio (screen+sound)": "basic_audio",
    "Deadly Corridor (shaped)": "deadly_corridor",
    "Defend the Center": "defend_the_center",
    "Defend the Line": "defend_the_line",
    "Health Gathering": "health_gathering",
    "Health Gathering Supreme": "health_gathering_supreme",
    "My Way Home": "my_way_home",
    "Predict Position": "predict_position",
    "Take Cover": "take_cover",
    "Doom E1M1 (full level)": "doom_E1M1",
    "Doom II MAP01 (full level)": "doom_MAP01",
}

# Mirrors each train_*.py's MODEL_PATH constant - the file visualize_PPO_model.py
# is pointed at for the currently selected level. Kept in sync with
# model_io.SCENARIO_MODELS via the scenario key.
MODEL_PATHS = {
    "Basic": "models/latest/ppo_basic.zip",
    "Simpler Basic": "models/latest/ppo_simpler_basic.zip",
    "Rocket Basic": "models/latest/ppo_rocket_basic.zip",
    "Basic Audio (screen+sound)": "models/latest/ppo_basic_audio.zip",
    "Deadly Corridor (shaped)": "models/latest/ppo_deadly_corridor_shaped.zip",
    "Defend the Center": "models/latest/ppo_defend_the_center.zip",
    "Defend the Line": "models/latest/ppo_defend_the_line.zip",
    "Health Gathering": "models/latest/ppo_health_gathering.zip",
    "Health Gathering Supreme": "models/latest/ppo_health_gathering_supreme.zip",
    "My Way Home": "models/latest/ppo_my_way_home.zip",
    "Predict Position": "models/latest/ppo_predict_position.zip",
    "Take Cover": "models/latest/ppo_take_cover.zip",
    "Doom E1M1 (full level)": "models/latest/ppo_doom_E1M1.zip",
    "Doom II MAP01 (full level)": "models/latest/ppo_doom_MAP01.zip",
}

# One render output per level so switching levels doesn't clobber the other's
# image, named after the scenario key.
VIZ_OUTPUT_NAMES = {
    level: f"ppo_actor_render_{key}.png" for level, key in SCENARIO_KEYS.items()
}

# Reward-shaping knobs, one entry per train_*.py --flag (wrapped across
# multiple UI rows - see KNOBS_PER_ROW below). Defaults mirror each script's
# argparse defaults (envs/basic_env.py's are all 0.0/off; train_deadly_corridor.py's
# match its existing hardcoded values) so leaving every field untouched
# reproduces today's behavior exactly.
REWARD_KNOBS = [
    ("kill_reward_bonus", "--kill-reward-bonus", "Kill bonus"),
    ("hit_reward_bonus", "--hit-reward-bonus", "Hit bonus"),
    ("exploration_bonus_per_cell", "--exploration-bonus-per-cell", "Exploration bonus / cell"),
    ("exploration_cell_size", "--exploration-cell-size", "Exploration cell size"),
    ("weapon_pickup_bonus", "--weapon-pickup-bonus", "Weapon pickup bonus"),
    ("damage_dealt_bonus", "--damage-dealt-bonus", "Damage dealt bonus"),
    ("damage_taken_penalty", "--damage-taken-penalty", "Damage taken penalty"),
    ("health_change_bonus", "--health-change-bonus", "Health change bonus"),
    ("armor_change_bonus", "--armor-change-bonus", "Armor change bonus"),
]
KNOBS_PER_ROW = 5

# Hover text for each knob, one sentence pulled from that wrapper's docstring
# in envs/common.py so this stays a faithful summary rather than a guess.
KNOB_DESCRIPTIONS = {
    "kill_reward_bonus": "Reward added each time KILLCOUNT increases (per kill). "
    "Makes killing enemies an explicit incentive on scenarios that don't score it directly.",
    "hit_reward_bonus": "Reward added each time HITCOUNT increases - fires on every "
    "successful hit landed on an enemy, not just kills, so it's denser signal leading up to a kill.",
    "exploration_bonus_per_cell": "Reward for the first visit to each discretized "
    "position cell per episode (grid-cell novelty, not raw distance) - oscillating in place doesn't farm reward.",
    "exploration_cell_size": "Size, in map units, of each exploration grid cell. "
    "~32 units is about one Doom grid tile.",
    "weapon_pickup_bonus": "Reward the first time each episode a WEAPON0-9 "
    "ownership flag flips to owned (e.g. picking up the shotgun a dead ShotgunGuy drops).",
    "damage_dealt_bonus": "Reward per DAMAGECOUNT point dealt to enemies - denser "
    "than the hit bonus since it distinguishes a grazing hit from a solid one.",
    "damage_taken_penalty": "Penalty subtracted per DAMAGE_TAKEN point received. "
    "Enter as a positive magnitude - it's always subtracted, never added.",
    "health_change_bonus": "Reward per net HEALTH point change this step - "
    "positive for pickups/healing, negative for damage taken.",
    "armor_change_bonus": "Reward per net ARMOR point change this step - "
    "positive for armor pickups, negative as armor absorbs damage.",
}

def _shaping(**overrides: float) -> dict[str, float]:
    """Full nine-knob dict: everything off except the given overrides —
    mirrors train_common.build_parser's fallback behavior so each entry
    below only states what that scenario's train_*.py actually defaults on."""
    base = {key: 0.0 for key, _flag, _label in REWARD_KNOBS}
    base["exploration_cell_size"] = 32.0
    base.update(overrides)
    return base


# Mirrors each train_*.py's REWARD_DEFAULTS so leaving every field untouched
# reproduces that script's own defaults exactly.
REWARD_DEFAULTS = {
    "Basic": _shaping(),
    "Simpler Basic": _shaping(),
    "Rocket Basic": _shaping(),
    "Basic Audio (screen+sound)": _shaping(),
    "Deadly Corridor (shaped)": _shaping(
        kill_reward_bonus=20.0,
        hit_reward_bonus=5.0,
        exploration_bonus_per_cell=1.0,
        weapon_pickup_bonus=15.0,
    ),
    "Defend the Center": _shaping(kill_reward_bonus=20.0, hit_reward_bonus=5.0),
    "Defend the Line": _shaping(kill_reward_bonus=20.0, hit_reward_bonus=5.0),
    "Health Gathering": _shaping(health_change_bonus=1.0),
    "Health Gathering Supreme": _shaping(health_change_bonus=1.0),
    "My Way Home": _shaping(exploration_bonus_per_cell=1.0),
    "Predict Position": _shaping(kill_reward_bonus=100.0, hit_reward_bonus=25.0),
    "Take Cover": _shaping(damage_taken_penalty=0.5),
    "Doom E1M1 (full level)": _shaping(
        kill_reward_bonus=20.0,
        hit_reward_bonus=5.0,
        exploration_bonus_per_cell=1.0,
        weapon_pickup_bonus=15.0,
        health_change_bonus=1.0,
        armor_change_bonus=0.5,
    ),
    "Doom II MAP01 (full level)": _shaping(
        kill_reward_bonus=20.0,
        hit_reward_bonus=5.0,
        exploration_bonus_per_cell=1.0,
        weapon_pickup_bonus=15.0,
        health_change_bonus=1.0,
        armor_change_bonus=0.5,
    ),
}


class Tooltip:
    """Small hover pop-up for one widget - no built-in equivalent in Tk.
    Shown ~½s after the pointer enters the widget, positioned just below it,
    dismissed on mouse-leave."""

    DELAY_MS = 500

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self._after_id: str | None = None
        self._popup: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._after_id = self.widget.after(self.DELAY_MS, self._show)

    def _show(self) -> None:
        if self._popup is not None:
            return
        x = self.widget.winfo_rootx()
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._popup = tk.Toplevel(self.widget)
        self._popup.wm_overrideredirect(True)
        self._popup.wm_geometry(f"+{x}+{y}")
        ttk.Label(
            self._popup,
            text=self.text,
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            wraplength=260,
            padding=4,
        ).pack()

    def _hide(self, _event=None) -> None:
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None
        if self._popup is not None:
            self._popup.destroy()
            self._popup = None


def resolve_python() -> str:
    """Prefer this project's .venv interpreter over whatever launched the UI."""
    venv_python = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"
    return str(venv_python) if venv_python.exists() else sys.executable


class TrainingLauncher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ViZDoom Training Launcher")
        # Wide enough for the full top-row button set (incl. Export/Import).
        self.geometry("1280x600")

        self.python_exe = resolve_python()
        self.process: subprocess.Popen | None = None
        self.watch_process: subprocess.Popen | None = None
        self.visualize_process: subprocess.Popen | None = None
        self.output_queue: queue.Queue[str] = queue.Queue()
        self.viz_image: tk.PhotoImage | None = None  # kept alive; Tk drops GC'd images
        self._last_viz_result: tuple[int, Path] | None = None

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

        self.visualize_button = ttk.Button(
            top, text="Visualize Model", command=self._start_visualize
        )
        self.visualize_button.pack(side="left", padx=(16, 4))

        # One-click model file management for the selected level — thin
        # wrappers around export_model.py / import_model.py (see model_io.py),
        # consistent with this UI's launcher-only role.
        self.export_button = ttk.Button(top, text="Export Model", command=self._export_model)
        self.export_button.pack(side="left", padx=(16, 4))

        self.import_button = ttk.Button(top, text="Import Model", command=self._import_model)
        self.import_button.pack(side="left", padx=4)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(top, textvariable=self.status_var).pack(side="right")

        self.watch_status_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.watch_status_var).pack(side="right", padx=(0, 12))

        rewards_frame = ttk.LabelFrame(self, text="Reward shaping", padding=10)
        rewards_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.reward_vars: dict[str, tk.StringVar] = {}
        for i, (key, _flag, label) in enumerate(REWARD_KNOBS):
            row, col = divmod(i, KNOBS_PER_ROW)
            label_widget = ttk.Label(rewards_frame, text=f"{label}  ⓘ")
            label_widget.grid(
                row=row * 2, column=col, padx=6, pady=(4 if row else 0, 0), sticky="w"
            )
            var = tk.StringVar()
            entry_widget = ttk.Entry(rewards_frame, textvariable=var, width=10)
            entry_widget.grid(row=row * 2 + 1, column=col, padx=6)
            self.reward_vars[key] = var

            description = KNOB_DESCRIPTIONS[key]
            Tooltip(label_widget, description)
            Tooltip(entry_widget, description)
        self._on_level_changed()

        bottom_frame = ttk.Frame(self, padding=(10, 0, 10, 10))
        bottom_frame.pack(fill="both", expand=True)

        # Log on the left (shows train/watch/visualize subprocess output, all
        # interleaved with a prefix per source) so a running Watch Agent's
        # output stays visible while a model render happens alongside it.
        log_frame = ttk.Frame(bottom_frame)
        log_frame.pack(side="left", fill="both", expand=True)

        self.log_text = tk.Text(log_frame, state="disabled", wrap="none", bg="black", fg="lightgreen")
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        # Model architecture render, in the same window rather than a popup,
        # so it sits next to whatever Watch Agent is doing.
        viz_frame = ttk.LabelFrame(bottom_frame, text="Model architecture", padding=10)
        viz_frame.pack(side="right", fill="y", padx=(10, 0))

        self.viz_status_var = tk.StringVar(value="No render yet")
        ttk.Label(viz_frame, textvariable=self.viz_status_var).pack(anchor="w")

        self.viz_image_label = ttk.Label(viz_frame, text="(click Visualize Model)")
        self.viz_image_label.pack(fill="both", expand=True, pady=(6, 0))

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

        command = [self.python_exe, *LEVELS[self.level_var.get()], *reward_args]
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
                elif line == "__VISUALIZE_DONE__":
                    self._on_visualize_done()
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
        command = [self.python_exe, *WATCH_SCRIPTS[level]]
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

    def _start_visualize(self) -> None:
        """Render the selected level's saved model architecture via
        visualize_PPO_model.py and display the resulting PNG inline (right
        panel), without disturbing whatever's in the log from a running
        train/watch subprocess."""
        if self.visualize_process is not None:
            return

        level = self.level_var.get()
        model_path = MODEL_PATHS[level]
        if not (PROJECT_ROOT / model_path).exists():
            messagebox.showerror(
                "Model not found", f"{model_path} doesn't exist yet - train this level first."
            )
            return

        out_name = VIZ_OUTPUT_NAMES[level]
        command = [self.python_exe, "visualize_PPO_model.py", "--model", model_path, "--out", out_name]
        self._append_log(f"$ {' '.join(command)}\n")

        self.visualize_process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )
        threading.Thread(
            target=self._read_visualize_output, args=(PROJECT_ROOT / out_name,), daemon=True
        ).start()

        self.visualize_button.configure(state="disabled")
        self.viz_status_var.set(f"Rendering: {level}...")

    def _read_visualize_output(self, out_path: Path) -> None:
        """Runs in a worker thread - visualize_PPO_model.py is a one-shot
        script (unlike train/watch's loops), so this just waits for it to
        exit once and reports the result back via the same queue/prefix
        pattern as the other subprocesses."""
        assert self.visualize_process is not None and self.visualize_process.stdout is not None
        for line in self.visualize_process.stdout:
            self.output_queue.put(f"[visualize] {line}")
        returncode = self.visualize_process.wait()
        self._last_viz_result = (returncode, out_path)
        self.output_queue.put("__VISUALIZE_DONE__")

    def _on_visualize_done(self) -> None:
        assert self._last_viz_result is not None
        returncode, out_path = self._last_viz_result
        self.visualize_process = None
        self.visualize_button.configure(state="normal")

        if returncode == 0 and out_path.exists():
            self._load_render_image(out_path)
            self.viz_status_var.set(f"Rendered: {out_path.name}")
        else:
            self.viz_status_var.set("Render failed - see log")
            self._append_log(f"\n[visualize] process exited with code {returncode}\n")

    def _load_render_image(self, path: Path) -> None:
        """Loads the PNG via Tk's built-in PNG support (no Pillow dependency
        needed here) and downscales it to fit the side panel using subsample,
        since PhotoImage has no smooth resize of its own."""
        img = tk.PhotoImage(file=str(path))
        max_w, max_h = 420, 420
        factor = max(1, -(-img.width() // max_w), -(-img.height() // max_h))
        if factor > 1:
            img = img.subsample(factor, factor)
        self.viz_image = img  # keep a reference - Tk drops images with no live ref
        self.viz_image_label.configure(image=img, text="")

    def _run_one_shot(self, command: list[str], prefix: str) -> subprocess.CompletedProcess:
        """Runs a quick command (export/import are file copies, well under a
        second) synchronously, logging its output with a [prefix] like the
        long-running subprocesses get."""
        self._append_log(f"$ {' '.join(command)}\n")
        result = subprocess.run(
            command, cwd=PROJECT_ROOT, capture_output=True, text=True
        )
        for stream in (result.stdout, result.stderr):
            if stream:
                for line in stream.splitlines():
                    self._append_log(f"[{prefix}] {line}\n")
        return result

    def _export_model(self) -> None:
        """Save the selected level's current model to a file of the user's
        choosing (with scenario metadata embedded — see model_io.py)."""
        level = self.level_var.get()
        key = SCENARIO_KEYS[level]
        if not (PROJECT_ROOT / MODEL_PATHS[level]).exists():
            messagebox.showerror(
                "Model not found",
                f"{MODEL_PATHS[level]} doesn't exist yet - train this level first.",
            )
            return
        dest = filedialog.asksaveasfilename(
            title=f"Export {level} model",
            defaultextension=".zip",
            initialfile=f"ppo_{key}_export.zip",
            filetypes=[("SB3 model", "*.zip")],
        )
        if not dest:
            return
        result = self._run_one_shot(
            [self.python_exe, "export_model.py", key, "--out", dest], "export"
        )
        if result.returncode == 0:
            messagebox.showinfo("Export complete", f"Exported to:\n{dest}")
        else:
            messagebox.showerror("Export failed", result.stderr or "See log for details.")

    def _import_model(self) -> None:
        """Install an exported model file as the selected level's active
        model (models/latest/), backing up the existing one. Retries with
        --force after confirmation if the file's scenario tag mismatches
        (import_model.py signals that with exit code 3)."""
        level = self.level_var.get()
        key = SCENARIO_KEYS[level]
        if self.process is not None:
            if not messagebox.askyesno(
                "Training is running",
                "This level may be training right now - its next checkpoint "
                "save would overwrite the imported model. Import anyway?",
            ):
                return
        src = filedialog.askopenfilename(
            title=f"Import model for {level}",
            filetypes=[("SB3 model", "*.zip"), ("All files", "*.*")],
        )
        if not src:
            return
        command = [self.python_exe, "import_model.py", src, "--scenario", key]
        result = self._run_one_shot(command, "import")
        if result.returncode == 3:  # scenario-tag mismatch; offer override
            if messagebox.askyesno(
                "Scenario mismatch",
                f"{result.stderr.strip()}\n\nImport anyway?",
            ):
                result = self._run_one_shot([*command, "--force"], "import")
            else:
                return
        if result.returncode == 0:
            messagebox.showinfo(
                "Import complete",
                f"{level} will now train/watch from the imported model.\n"
                "(The previous model, if any, was backed up to models/backups/.)",
            )
        else:
            messagebox.showerror("Import failed", result.stderr or "See log for details.")

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
        if self.visualize_process is not None:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(self.visualize_process.pid)],
                capture_output=True,
            )
        self.destroy()


if __name__ == "__main__":
    TrainingLauncher().mainloop()
