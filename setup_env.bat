@echo off
setlocal enabledelayedexpansion

rem Directory this script lives in (trailing backslash included) so it works
rem regardless of the caller's current directory.
set "PROJECT_ROOT=%~dp0"
set "VENV_DIR=%PROJECT_ROOT%.venv"

rem Interpreter used to CREATE the venv. Resolved dynamically so nothing
rem here is tied to one machine/user profile. Override by setting PYTHON_EXE
rem yourself before running this script if you want to force a specific one.
if defined PYTHON_EXE (
    set "PY_CMD=%PYTHON_EXE%"
) else (
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3.14 -c "" >nul 2>&1
        if not errorlevel 1 set "PY_CMD=py -3.14"
    )
    if not defined PY_CMD (
        where py >nul 2>&1
        if not errorlevel 1 (
            py -3 -c "" >nul 2>&1
            if not errorlevel 1 (
                echo [setup] Python 3.14 not registered with 'py', falling back to the latest Python 3 it finds.
                set "PY_CMD=py -3"
            )
        )
    )
    if not defined PY_CMD (
        where python >nul 2>&1
        if not errorlevel 1 (
            echo [setup] 'py' launcher not found, falling back to 'python' on PATH.
            set "PY_CMD=python"
        )
    )
)

if not defined PY_CMD (
    echo [setup] FAILED: no Python interpreter found ^(checked 'py' launcher and PATH^).
    exit /b 1
)

echo [setup] Using interpreter:
%PY_CMD% -c "import sys; print(sys.executable, sys.version.split()[0])"

if exist "%VENV_DIR%\Scripts\python.exe" (
    echo [setup] .venv already exists at %VENV_DIR% - skipping creation.
) else (
    echo [setup] Creating virtual environment at %VENV_DIR% ...
    %PY_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [setup] FAILED to create the virtual environment.
        exit /b 1
    )
)

set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

echo [setup] Upgrading pip ...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [setup] FAILED to upgrade pip.
    exit /b 1
)

rem Skip the (slow) pip resolve/install entirely if requirements.txt hasn't
rem changed since the last successful install here - re-running pip install
rem with no version pins is what caused packages to get silently
rem uninstalled/reinstalled on every run (the resolver re-picks versions for
rem unpinned transitive deps like numpy/opencv each time).
set "REQ_FILE=%PROJECT_ROOT%requirements.txt"
set "HASH_FILE=%VENV_DIR%\.requirements.sha256"

for /f "usebackq delims=" %%H in (`powershell -NoProfile -Command "(Get-FileHash '%REQ_FILE%' -Algorithm SHA256).Hash"`) do set "NEW_HASH=%%H"

set "OLD_HASH="
if exist "%HASH_FILE%" set /p OLD_HASH=<"%HASH_FILE%"

if "%NEW_HASH%"=="%OLD_HASH%" (
    echo [setup] requirements.txt unchanged since last install - skipping pip install.
) else (
    echo [setup] Installing project dependencies from requirements.txt ...
    "%VENV_PY%" -m pip install -r "%REQ_FILE%"
    if errorlevel 1 (
        echo [setup] FAILED to install dependencies.
        exit /b 1
    )
    >"%HASH_FILE%" echo %NEW_HASH%
)

echo [setup] Verifying the install ...
"%VENV_PY%" -c "import vizdoom, gymnasium, stable_baselines3, torch, cv2; print('vizdoom', vizdoom.__version__); print('gymnasium', gymnasium.__version__); print('stable_baselines3', stable_baselines3.__version__); print('torch', torch.__version__); print('cv2', cv2.__version__); print('CUDA available:', torch.cuda.is_available())"
if errorlevel 1 (
    echo [setup] Import check FAILED - see error above.
    exit /b 1
)

echo.
echo [setup] Done. Launch the desktop UI with:
echo   .venv\Scripts\python.exe train_ui.py
endlocal
