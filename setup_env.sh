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

echo "[setup] Installing project dependencies (vizdoom, gymnasium, stable-baselines3, torch, tensorboard) ..."
"$VENV_PY" -m pip install vizdoom gymnasium stable-baselines3 torch tensorboard

echo "[setup] Verifying the install ..."
"$VENV_PY" -c "import vizdoom, gymnasium, stable_baselines3, torch; print('vizdoom', vizdoom.__version__); print('gymnasium', gymnasium.__version__); print('stable_baselines3', stable_baselines3.__version__); print('torch', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

echo
echo "[setup] Done. Launch the desktop UI with:"
echo "  .venv/Scripts/python.exe train_ui.py"
