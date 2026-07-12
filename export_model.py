"""Export a trained scenario model to a shareable, re-importable single file.

Usage:
    python export_model.py basic
    python export_model.py deadly_corridor --out D:/backups/corridor_v1.zip
    python export_model.py doom_E1M1

The output is the scenario's models/latest/*.zip with export metadata
(scenario, timestamp, package versions) embedded — still directly loadable
with PPO.load, and importable on any copy of this project via
import_model.py. Default destination: exports/ppo_<scenario>_<timestamp>.zip.
See model_io.py for the mechanics; train_ui.py's Export button runs this
script.
"""

import argparse
import sys

from model_io import SCENARIO_MODELS, export_model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario",
        help=f"one of: {', '.join(sorted(SCENARIO_MODELS))}, or doom_<MAP> (e.g. doom_E1M1)",
    )
    parser.add_argument("--out", default=None, help="destination file (default: exports/...)")
    args = parser.parse_args()

    try:
        dest = export_model(args.scenario, args.out)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Export failed: {exc}", file=sys.stderr)
        return 1
    print(f"Exported {args.scenario!r} model to: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
