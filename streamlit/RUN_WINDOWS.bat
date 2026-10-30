@echo off
REM ===================================================================
REM  Industrial Equipment Failure Prediction - one-step launcher
REM  Creates a virtual environment, installs dependencies,
REM  verifies the setup, then opens the dashboard.
REM ===================================================================
setlocal

echo.
echo ============================================================
echo   Industrial Equipment Failure Prediction
echo ============================================================
echo.

REM --- 1. find a usable python -------------------------------------
set PY=
python --version >nul 2>&1 && set PY=python
if not defined PY (
    py --version >nul 2>&1 && set PY=py
)
if not defined PY (
    echo [ERROR] Python was not found.
    echo.
    echo   Install Python 3.11 or 3.12 from https://www.python.org/downloads/
    echo   IMPORTANT: tick "Add python.exe to PATH" on the first installer screen.
    echo   Then close this window, open a NEW one, and run this script again.
    echo.
    pause
    exit /b 1
)

echo [1/4] Using interpreter:
%PY% --version
echo.

REM --- 2. create the virtual environment ---------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [2/4] Creating virtual environment in .venv ...
    %PY% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        echo         On some systems you may need:  %PY% -m pip install --upgrade virtualenv
        pause
        exit /b 1
    )
) else (
    echo [2/4] Virtual environment already exists - reusing it.
)
echo.

REM --- 3. install dependencies ------------------------------------
echo [3/4] Installing / checking dependencies ...
echo       (first run downloads ~150 MB and takes a few minutes)
".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
".venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Run these to see the details:
    echo           .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)
echo       Dependencies installed.
echo.

REM --- 4. verify --------------------------------------------------
echo [4/4] Verifying the environment ...
".venv\Scripts\python.exe" scripts\check_env.py
if errorlevel 1 (
    echo.
    echo [WARNING] The environment check reported problems (see FAIL lines above).
    echo           The dashboard may still work. Press any key to try launching it.
    pause
)

echo.
echo ============================================================
echo   Launching the dashboard...
echo   Open http://localhost:8501  in your browser
echo   Press Ctrl+C in this window to stop.
echo ============================================================
echo.

".venv\Scripts\python.exe" -m streamlit run app\streamlit_app.py

echo.
echo Dashboard stopped.
pause
