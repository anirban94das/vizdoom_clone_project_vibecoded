"""Import an exported model file as a scenario's active model.

Usage:
    python import_model.py path/to/ppo_basic_20260712_101500.zip --scenario basic
    python import_model.py corridor_v1.zip --scenario deadly_corridor --force

Installs the file as models/latest/<that scenario's model>.zip — the file
auto-resume and watch_agent_*.py read — after backing up the existing model
to models/backups/. Refuses a file whose embedded export metadata names a
different scenario unless --force (exit code 3, which train_ui.py's Import
button uses to offer an override). Don't import while that scenario is
training: the next checkpoint save would overwrite the import.
"""

import argparse
import sys

from model_io import SCENARIO_MODELS, ScenarioMismatchError, import_model

EXIT_MISMATCH = 3  # distinct code so train_ui.py can offer a force-retry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="exported model .zip to import")
    parser.add_argument(
        "--scenario",
        required=True,
        help=f"one of: {', '.join(sorted(SCENARIO_MODELS))}, or doom_<MAP> (e.g. doom_E1M1)",
    )
    parser.add_argument("--force", action="store_true", help="ignore a scenario-tag mismatch")
    args = parser.parse_args()

    try:
        target, backup = import_model(args.file, args.scenario, force=args.force)
    except ScenarioMismatchError as exc:
        print(f"Import refused: {exc}", file=sys.stderr)
        return EXIT_MISMATCH
    except (ValueError, FileNotFoundError) as exc:
        print(f"Import failed: {exc}", file=sys.stderr)
        return 1

    print(f"Imported into: {target}")
    if backup is not None:
        print(f"Previous model backed up to: {backup}")
    print("Training auto-resume and watch_agent_*.py will now use the imported model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
