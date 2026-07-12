"""Export/import of trained scenario models — shared logic behind
export_model.py / import_model.py (CLI) and train_ui.py's Export/Import
buttons.

An exported model is just the scenario's models/latest/*.zip copied to the
destination with one extra file, export_metadata.json, appended INTO the
zip (SB3 model files are ordinary zip archives and PPO.load only reads the
entries it knows, so the extra member is harmless — the exported file stays
directly loadable). The metadata records which scenario the model came from,
when, and under which package versions, so an import can refuse to clobber
e.g. the Deadly Corridor model with a Basic export (different action spaces
— such a model would crash at PPO.load time anyway, but with a much less
helpful error).

Importing copies the file over the scenario's models/latest/*.zip — the
exact file auto-resume and watch_agent_*.py read — after backing up the
existing one to models/backups/<name>_<timestamp>.zip. Nothing else needs
updating: training resumes from the imported weights on the next run.
"""

import json
import shutil
import zipfile
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LATEST_DIR = PROJECT_ROOT / "models" / "latest"
BACKUP_DIR = PROJECT_ROOT / "models" / "backups"
EXPORTS_DIR = PROJECT_ROOT / "exports"
METADATA_NAME = "export_metadata.json"

# Scenario key -> model filename under models/latest/. Keys are what the CLIs
# take and what gets stamped into export metadata. Full-game levels aren't
# listed: any "doom_<MAP>" key resolves dynamically (see resolve_model_path).
SCENARIO_MODELS = {
    "basic": "ppo_basic.zip",
    "deadly_corridor": "ppo_deadly_corridor_shaped.zip",
    "defend_the_center": "ppo_defend_the_center.zip",
    "defend_the_line": "ppo_defend_the_line.zip",
    "health_gathering": "ppo_health_gathering.zip",
    "health_gathering_supreme": "ppo_health_gathering_supreme.zip",
    "my_way_home": "ppo_my_way_home.zip",
    "predict_position": "ppo_predict_position.zip",
    "take_cover": "ppo_take_cover.zip",
    "rocket_basic": "ppo_rocket_basic.zip",
    "simpler_basic": "ppo_simpler_basic.zip",
    "basic_audio": "ppo_basic_audio.zip",
}


class ScenarioMismatchError(Exception):
    """Import target scenario differs from the scenario stamped in the file's
    export metadata (use force=True / --force to override)."""


def resolve_model_path(scenario: str) -> Path:
    if scenario in SCENARIO_MODELS:
        return LATEST_DIR / SCENARIO_MODELS[scenario]
    if scenario.startswith("doom_"):
        return LATEST_DIR / f"ppo_{scenario}.zip"  # e.g. doom_E1M1 -> ppo_doom_E1M1.zip
    known = ", ".join(sorted(SCENARIO_MODELS))
    raise ValueError(
        f"Unknown scenario {scenario!r}. Known: {known}, or doom_<MAP> (e.g. doom_E1M1)."
    )


def _package_versions() -> dict[str, str]:
    versions = {}
    for pkg in ("stable_baselines3", "torch", "vizdoom", "gymnasium"):
        try:
            versions[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            versions[pkg] = "unknown"
    return versions


def read_export_metadata(path: str | Path) -> dict | None:
    """The export_metadata.json embedded in an exported model, or None if the
    zip predates export / came straight from models/latest/."""
    with zipfile.ZipFile(path) as zf:
        if METADATA_NAME in zf.namelist():
            return json.loads(zf.read(METADATA_NAME))
    return None


def export_model(scenario: str, dest: str | Path | None = None) -> Path:
    """Copies the scenario's current model to dest (default: exports/ with a
    timestamped name) and embeds export metadata. Returns the written path."""
    src = resolve_model_path(scenario)
    if not src.exists():
        raise FileNotFoundError(f"{src} doesn't exist — train the {scenario!r} scenario first.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if dest is None:
        EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
        dest = EXPORTS_DIR / f"ppo_{scenario}_{stamp}.zip"
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src, dest)
    meta = {
        "scenario": scenario,
        "model_file": src.name,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "package_versions": _package_versions(),
    }
    with zipfile.ZipFile(dest, "a") as zf:
        zf.writestr(METADATA_NAME, json.dumps(meta, indent=2))
    return dest


def import_model(src: str | Path, scenario: str, force: bool = False) -> tuple[Path, Path | None]:
    """Installs src as the scenario's models/latest/*.zip. Validates that src
    is an SB3 model archive; if it carries export metadata naming a different
    scenario, refuses unless force. Returns (installed_path, backup_path) —
    backup_path is None if there was nothing to back up."""
    src = Path(src)
    if not src.exists():
        raise FileNotFoundError(f"{src} not found.")

    try:
        with zipfile.ZipFile(src) as zf:
            names = zf.namelist()
            if "data" not in names:
                raise ValueError(
                    f"{src} doesn't look like a stable-baselines3 model "
                    "(no 'data' entry in the zip)."
                )
            meta = json.loads(zf.read(METADATA_NAME)) if METADATA_NAME in names else None
    except zipfile.BadZipFile:
        raise ValueError(f"{src} is not a zip archive (expected an SB3 model .zip).") from None

    if meta is not None and meta.get("scenario") not in (None, scenario) and not force:
        raise ScenarioMismatchError(
            f"{src.name} was exported from scenario {meta['scenario']!r}, but the import "
            f"target is {scenario!r}. Scenarios can differ in action/observation spaces, "
            "in which case the model won't load. Use --force to import anyway."
        )

    target = resolve_model_path(scenario)
    backup = None
    if target.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = BACKUP_DIR / f"{target.stem}_{stamp}.zip"
        shutil.copy2(target, backup)

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, target)
    return target, backup
