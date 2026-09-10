"""Put the repo root on sys.path for the scenario entry points.

Every train_*.py / watch_agent_*.py in this folder imports shared modules that
live at the repo root (`train_common`, `training_utils`, the `envs` package)
and is meant to be run as `python scenarios/train_basic.py` from the repo
root. When Python runs a script it prepends *that script's* directory
(scenarios/) to sys.path, not the repo root, so those imports would fail.

Each script does `import _bootstrap` as its first project import; importing
this module prepends the parent directory (the repo root) to sys.path. It
works because scenarios/ is already on sys.path[0] when the script runs.

Relative paths used downstream (models/, logs/, configs/) stay relative to
the process CWD, so keep running these from the repo root.
"""

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
