#!/usr/bin/env bash
# Sets up the project-local .venv and installs dependencies.
# Mirrors setup_env.bat — use whichever shell you prefer (this one is for
# Git Bash / WSL / any POSIX shell on this machine).
set -euo pipefail

# Directory this script lives in, so it works regardless of the caller's
# current directory.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"

# Interpreter used to CREATE the venv. Resolved dynamically so nothing here
# is tied to one machine/user profile. Override by exporting PYTHON_EXE
# yourself before running this script if you want to force a specific one.
if [ -n "${PYTHON_EXE:-}" ]; then
    PY_CMD=("$PYTHON_EXE")
elif command -v py >/dev/null 2>&1 && py -3.14 -c "" >/dev/null 2>&1; then
    # Windows py launcher: resolves the registered 3.14 install regardless
    # of where it's actually installed on this machine.
    PY_CMD=(py -3.14)
elif command -v py >/dev/null 2>&1 && py -3 -c "" >/dev/null 2>&1; then
    echo "[setup] Python 3.14 not registered with 'py', falling back to the latest Python 3 it finds."
    PY_CMD=(py -3)
elif command -v python >/dev/null 2>&1; then
    echo "[setup] 'py' launcher not found, falling back to 'python' on PATH."
    PY_CMD=(python)
else
    echo "[setup] FAILED: no Python interpreter found (checked 'py' launcher and PATH)."
    exit 1
fi
echo "[setup] Using interpreter: $("${PY_CMD[@]}" -c 'import sys; print(sys.executable, sys.version.split()[0])')"

if [ -x "$VENV_DIR/Scripts/python.exe" ]; then
    echo "[setup] .venv already exists at $VENV_DIR - skipping creation."
else
    echo "[setup] Creating virtual environment at $VENV_DIR ..."
    "${PY_CMD[@]}" -m venv "$VENV_DIR"
fi

VENV_PY="$VENV_DIR/Scripts/python.exe"

echo "[setup] Upgrading pip ..."
"$VENV_PY" -m pip install --upgrade pip

# Skip the (slow) pip resolve/install entirely if requirements.txt hasn't
# changed since the last successful install here - re-running pip install
# with no version pins is what caused packages to get silently
# uninstalled/reinstalled on every run (the resolver re-picks versions for
# unpinned transitive deps like numpy/opencv each time).
REQ_FILE="$PROJECT_ROOT/requirements.txt"
HASH_FILE="$VENV_DIR/.requirements.sha256"

NEW_HASH="$(sha256sum "$REQ_FILE" | cut -d' ' -f1)"
OLD_HASH=""
if [ -f "$HASH_FILE" ]; then
    OLD_HASH="$(cat "$HASH_FILE")"
fi

if [ "$NEW_HASH" = "$OLD_HASH" ]; then
    echo "[setup] requirements.txt unchanged since last install - skipping pip install."
else
    echo "[setup] Installing project dependencies from requirements.txt ..."
    "$VENV_PY" -m pip install -r "$REQ_FILE"
    echo "$NEW_HASH" > "$HASH_FILE"
fi

echo "[setup] Verifying the install ..."
"$VENV_PY" -c "import vizdoom, gymnasium, stable_baselines3, torch, cv2; print('vizdoom', vizdoom.__version__); print('gymnasium', gymnasium.__version__); print('stable_baselines3', stable_baselines3.__version__); print('torch', torch.__version__); print('cv2', cv2.__version__); print('CUDA available:', torch.cuda.is_available())"

echo
echo "[setup] Done. Launch the desktop UI with:"
echo "  .venv/Scripts/python.exe train_ui.py"
