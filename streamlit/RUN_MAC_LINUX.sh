#!/usr/bin/env bash
# ===================================================================
#  Industrial Equipment Failure Prediction - one-step launcher
#  Creates a virtual environment, installs dependencies,
#  verifies the setup, then opens the dashboard.
# ===================================================================
set -e

echo
echo "============================================================"
echo "  Industrial Equipment Failure Prediction"
echo "============================================================"
echo

# --- 1. find a usable python ---------------------------------------
PY=""
for cand in python3.12 python3.11 python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then
        ver=$("$cand" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || echo "0.0")
        major=${ver%%.*}
        minor=${ver##*.}
        if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ]; then
            PY="$cand"
            break
        fi
    fi
done

if [ -z "$PY" ]; then
    echo "[ERROR] No Python 3.10+ found."
    echo
    echo "  macOS : brew install python@3.12"
    echo "          (or download from https://www.python.org/downloads/)"
    echo "  Linux : sudo apt install python3 python3-pip python3-venv"
    echo
    echo "  Then run this script again."
    exit 1
fi

echo "[1/4] Using interpreter:"
"$PY" --version
echo

# --- 2. create the virtual environment -----------------------------
if [ ! -f ".venv/bin/python" ]; then
    echo "[2/4] Creating virtual environment in .venv ..."
    "$PY" -m venv .venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Could not create the virtual environment."
        echo "        On Debian/Ubuntu you may need:  sudo apt install python3-venv"
        exit 1
    fi
else
    echo "[2/4] Virtual environment already exists - reusing it."
fi
echo

# --- 3. install dependencies ---------------------------------------
echo "[3/4] Installing / checking dependencies ..."
echo "      (first run downloads ~150 MB and takes a few minutes)"
.venv/bin/python -m pip install --upgrade pip --quiet
if ! .venv/bin/python -m pip install -r requirements.txt --quiet; then
    echo "[ERROR] Dependency installation failed. Run this to see details:"
    echo "          .venv/bin/python -m pip install -r requirements.txt"
    exit 1
fi
echo "      Dependencies installed."
echo

# --- 4. verify -----------------------------------------------------
echo "[4/4] Verifying the environment ..."
if ! .venv/bin/python scripts/check_env.py; then
    echo
    echo "[WARNING] The environment check reported problems (see FAIL lines above)."
    echo "          The dashboard may still work. Launching anyway..."
fi

echo
echo "============================================================"
echo "  Launching the dashboard..."
echo "  Open http://localhost:8501 in your browser"
echo "  Press Ctrl+C to stop."
echo "============================================================"
echo

exec .venv/bin/python -m streamlit run app/streamlit_app.py
