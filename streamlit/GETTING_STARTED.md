# 🚀 Getting Started — run this on your laptop, step by step

A complete walkthrough from **nothing installed** to **dashboard running in your browser**.

Total time: **about 10 minutes** (most of it is downloading Python and packages).
You will not need to edit any code.

---

## What you are about to run

| Piece | What it is | How it opens |
|---|---|---|
| **Dashboard** | Interactive web app (the main thing) | `http://localhost:8501` in your browser |
| **Notebook** | The full analysis, with all charts already saved | Jupyter in your browser |
| **Report** | Written report, Markdown + Word | Double-click, or your editor |

The project ships **pre-trained models**, so the dashboard works immediately. Retraining is
optional and takes ~60 seconds.

> 💡 `requirements.txt` intentionally uses **minimum** versions, so your laptop may install
> slightly newer packages than the ones used to build the models. That is fine — the full
> pipeline was verified on pandas 3.1 and scikit-learn 1.9 with identical results. You may see
> an `InconsistentVersionWarning` on first load; it is harmless, and the dashboard explains it
> in the sidebar. Run `python -m src.train` if you want an exact version match.

---

# STEP 0 — Check what you already have

Open a terminal:

- **Windows** → press `Win`, type `cmd`, press Enter
- **macOS** → press `Cmd+Space`, type `terminal`, press Enter
- **Linux** → `Ctrl+Alt+T`

Type this and press Enter:

```bash
python --version
```

Then try this one:

```bash
python3 --version
```

**What you should see:** something like `Python 3.11.5` or `Python 3.12.1`.

**Which command do you use?** Whichever one worked:

- `python --version` works → use **`python`** everywhere below
- only `python3` works → use **`python3`** everywhere below
- both say *"not recognised"* / *"command not found"* → do **Step 1**
- it says `Python 3.9.x` or older → do **Step 1** (you need 3.10+)

> ⚠️ **macOS users:** the `python3` that ships with macOS is often 3.9 and is locked by the
> system. Do Step 1 anyway and you'll get a modern version.

---

# STEP 1 — Install Python (skip if Step 0 gave 3.10 or newer)

Go to **<https://www.python.org/downloads/>** and download the latest Python 3.12 (or 3.11).
Both are rock-solid for this project. Avoid 3.14+ for now — some packages have no prebuilt
installer yet.

### Windows

1. Run the installer you downloaded.
2. **⚠️ On the very first screen, tick the box "Add python.exe to PATH".** This single checkbox
   prevents 90% of setup problems. It is easy to miss at the bottom of the window.
3. Click **Install Now**.
4. **Close your terminal and open a new one** (the PATH change only applies to new terminals).
5. Verify:

```bash
python --version
```

### macOS

Two options — either is fine.

**Option A (recommended, no admin needed):** install Homebrew, then Python.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
brew install python@3.12
```

**Option B:** download the macOS installer from python.org, run it, then verify:

```bash
python3 --version
```

### Linux (Ubuntu / Debian / WSL)

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv
python3 --version
```

> If `python3 --version` is below 3.10 on an old distro, use a PPA or pyenv. Ubuntu 22.04+ and
> Debian 12+ are fine.

✅ **Checkpoint:** `python --version` (or `python3`) shows **3.10 or higher**.

---

# STEP 2 — Unzip the project and open a terminal inside it

1. Unzip `industrial-failure-prediction.zip` to a folder you can find again, e.g.
   - Windows: `C:\Users\YourName\Projects\`
   - macOS/Linux: `~/Projects/`

2. Open a terminal **inside that folder** — this is the step people most often get wrong.

**Easiest way to be sure:** open the folder in your file explorer, then:

| OS | How |
|---|---|
| Windows | Click the address bar, type `cmd`, press Enter |
| macOS | Right-click the folder → *Services* → **New Terminal at Folder** |
| Linux | Right-click inside the folder → **Open in Terminal** |
| VS Code | *File → Open Folder* → then *Terminal → New Terminal* |

3. Verify you are in the right place:

```bash
dir          # Windows
ls           # macOS / Linux
```

**You should see:** `README.md`, `requirements.txt`, `app`, `src`, `data`, `notebooks`,
`reports`, `Makefile`, `Dockerfile`.

> ❌ If you see `Python` or `Downloads` or nothing familiar, you are in the wrong folder.
> `cd` into the project folder and try again. Everything below **must** be run from this folder.

✅ **Checkpoint:** `ls` (or `dir`) lists `requirements.txt`.

---

# STEP 3 — Create a virtual environment

A virtual environment keeps this project's packages separate from everything else on your
computer. It prevents version conflicts and makes "delete and start over" trivial.

```bash
python -m venv .venv
```

*(use `python3` instead of `python` if that's what worked in Step 0)*

This creates a `.venv` folder. It takes a few seconds and prints nothing on success.

### Activate it

**Windows (Command Prompt):**
```bat
.venv\Scripts\activate
```

**Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```
> If PowerShell blocks you with *"running scripts is disabled on this system"*, run this once,
> then try again:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

✅ **Checkpoint:** your prompt now starts with `(.venv)`, e.g.

```
(.venv) C:\Users\You\Projects\industrial-failure-prediction>
```

> 🔁 **You must re-activate every time you open a new terminal** — just re-run the activate
> command above. If you forget, you'll get "module not found" errors.

---

# STEP 4 — Install the dependencies

Make sure you see `(.venv)` in your prompt, then:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

This downloads ~150 MB and takes **2–5 minutes**. You'll see a wall of progress bars — normal.

**On macOS with an Apple Silicon chip (M1/M2/M3/M4):** everything is prebuilt, nothing special
needed.

✅ **Checkpoint:** the last lines say something like
`Successfully installed streamlit-1.66.1 pandas-2.2.3 ...` with no red `ERROR`.

---

# STEP 5 — Verify everything before running anything ⭐

**Do not skip this.** It catches every common setup problem in one shot:

```bash
python scripts/check_env.py
```

**You should see** a report ending in:

```
  ✓ Your environment is ready.

  Next steps:
    1) streamlit run app/streamlit_app.py     → open the dashboard
    2) python -m src.train                    → rebuild everything (~60s)
    3) jupyter notebook notebooks/            → read the walkthrough
```

It also proves the saved model actually works by running a live prediction end-to-end. If
anything is wrong it tells you exactly what, in plain language — fix that and run it again.

> `WARN` lines are fine. Only `FAIL` lines block you.

✅ **Checkpoint:** the script prints `✓ Your environment is ready.` and exits.

---

# STEP 6 — Launch the dashboard 🎉

```bash
streamlit run app/streamlit_app.py
```

Windows alternative if `streamlit` isn't found:

```bat
python -m streamlit run app/streamlit_app.py
```

You'll see:

```
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8501
  Network URL: http://192.168.1.42:8501
```

Your browser should **open automatically**. If it doesn't, open
**<http://localhost:8501>** yourself.

### What you should see

A dark industrial-themed dashboard with four views (switch them with the pill buttons at the
top: **Machine Risk Check · Batch Scoring · Model Evaluation · Patterns & Drivers**):

1. **🔧 Machine Risk Check** — drag the sliders; the risk gauge and the "why" panel update live.
   Try the dropdown → **"High-power overstrain"** to see a high-risk reading.
2. **📁 Batch Scoring** — click **"▶ Use the 2,000-row holdout sample instead"** to score a
   whole file and download the results.
3. **📈 Model Evaluation** — all three models compared, confusion matrices, thresholds.
4. **🔍 Patterns & Drivers** — the charts and the feature-importance explanations.

**To stop the dashboard:** click the terminal and press `Ctrl+C`.

✅ **Checkpoint:** the page loads with no red error box.

---

# STEP 7 (optional) — Read the analysis notebook

```bash
jupyter notebook notebooks/01_failure_prediction_walkthrough.ipynb
```

or, if you prefer the newer interface:

```bash
jupyter lab notebooks/01_failure_prediction_walkthrough.ipynb
```

A browser tab opens. Click **Kernel → Restart & Run All** to execute it yourself, or just
scroll — **all outputs are already saved**, so you can read every chart and table without
running a single cell.

**To stop Jupyter:** `Ctrl+C` twice in the terminal.

> The notebook takes ~60 seconds to run fully because it retrains all three models.

---

# STEP 8 (optional) — Retrain from scratch

The models are already trained, but you can rebuild everything yourself:

```bash
python -m src.train
```

This runs **dataset → cleaning → feature engineering → training → evaluation → explanation**
and overwrites everything in `artifacts/`. Takes **~60 seconds**. You'll see:

```
STEP 1/5  Import + clean
  rows=10,000  cols=25  failures=339  removed=0
STEP 3/5  Train + evaluate (LR / Decision Tree / Random Forest)
  Logistic Regression  thr=0.910 P=0.537 R=0.426 F1=0.475 PR-AUC=0.446
  Decision Tree        thr=0.950 P=0.911 R=0.750 F1=0.823 PR-AUC=0.835
  Random Forest        thr=0.440 P=0.946 R=0.779 F1=0.855 PR-AUC=0.883
  best model -> random_forest
STEP 4/5  Explanations + figures
  14 figures written
Done in 62.5s
```

Refresh the dashboard in your browser to see the new artifacts.

---

# STEP 9 (optional) — Regenerate the report

```bash
python scripts/build_report.py
```

Rewrites `reports/failure_prediction_report.md` **and** `.docx` from the actual measured
results, embedding all 14 figures and 18 tables. The numbers can therefore never drift away
from what the models produced.

---

# 🍎🪟 One-command shortcuts

Instead of Steps 3–6 you can use the included helper scripts.

**Windows** — double-click or run:

```bat
RUN_WINDOWS.bat
```

**macOS / Linux:**

```bash
chmod +x RUN_MAC_LINUX.sh
./RUN_MAC_LINUX.sh
```

They create the virtual environment, install dependencies, run the environment check, and
launch the dashboard.

**If you have `make`** (macOS/Linux, or Windows with Git Bash):

| Command | Does |
|---|---|
| `make install` | install dependencies |
| `make check` | run the environment verification |
| `make app` | launch the dashboard |
| `make train` | rebuild models, metrics and figures |
| `make report` | regenerate the Word/Markdown report |
| `make notebook` | rebuild **and execute** the notebook |
| `make all` | data → train → report → notebook |
| `make clean` | delete generated artifacts (keeps the raw data) |

---

# 🐳 Alternative: run it in Docker (no Python setup at all)

If you have Docker Desktop installed, this skips Steps 1–5 entirely.

```bash
cd industrial-failure-prediction
docker build -t failure-prediction .
docker run -p 8501:8501 failure-prediction
```

Then open <http://localhost:8501>.

---

# 🔧 Troubleshooting — real error messages and their fixes

| What you see | Cause | Fix |
|---|---|---|
| `'python' is not recognized as an internal or external command` | Python not on PATH (Windows) | Reinstall Python and **tick "Add python.exe to PATH"**, or use `py` instead of `python`. Reopen the terminal. |
| `command not found: python` (macOS) | Only `python3` exists | Use `python3` everywhere instead of `python`. |
| `bash: .venv/bin/activate: No such file or directory` | venv was never created, or you're in the wrong folder | Go back to Step 2 (`ls` should show `requirements.txt`), then redo `python -m venv .venv`. |
| `running scripts is disabled on this system` (PowerShell) | Windows execution policy | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, then re-activate. |
| `ERROR: Could not find a version that satisfies the requirement ...` | Usually an outdated `pip` | `python -m pip install --upgrade pip` then retry. |
| `ModuleNotFoundError: No module named 'streamlit'` | Virtual environment not activated, or packages not installed | Check for `(.venv)` in your prompt. If absent, re-activate (Step 3). |
| `ModuleNotFoundError: No module named 'src'` | Ran a file directly instead of as a module | Use `python -m src.train` (with `-m`), and run it **from the project root**. |
| `streamlit: command not found` | Scripts folder not on PATH inside the venv | Use `python -m streamlit run app/streamlit_app.py`. |
| Browser shows **"This site can't be reached"** | Dashboard not running, or wrong port | Make sure the terminal still shows the app running. Try `http://localhost:8501`. |
| Page is **stuck on the loading skeleton** (grey placeholders, never loads) | The WebSocket session was rejected — usually a reverse proxy, VPN, or corporate firewall | Hard-refresh (`Ctrl+Shift+R`). If you're behind a proxy, confirm `.streamlit/config.toml` has `enableCORS = false` and `enableXsrfProtection = false`. Try a different browser or network. |
| `OSError: [Errno 98] Address already in use` (or `10048` on Windows) | Port 8501 already used | `streamlit run app/streamlit_app.py --server.port 8502` then open `http://localhost:8502`. |
| Dashboard opens but says **"No trained models found"** | `artifacts/models.joblib` missing | Run `python -m src.train`, then refresh the page. |
| Charts missing / `findfont` warnings | Harmless matplotlib font warnings | Ignore them. If a figure file is genuinely absent, run `python -m src.train`. |
| `InconsistentVersionWarning: Trying to unpickle estimator ... from version 1.6.1` | Your scikit-learn differs from the one that saved the models | **Harmless — ignore it.** Predictions are valid. The dashboard shows a friendly note. For an exact match run `python -m src.train` (~60 s). |
| `No module named 'docx'` when building the report | Only affects report generation | `pip install python-docx` (or just read the shipped `.docx`). |
| Notebook won't open / `jupyter: command not found` | Jupyter missing | `pip install jupyter` then `jupyter notebook notebooks/`. |
| `jupyter nbconvert` fails on `make notebook` | nbconvert not installed | `pip install nbconvert nbclient`. |
| Everything worked yesterday, nothing works now | New terminal, venv not active | Re-activate: `.venv\Scripts\activate` (Win) or `source .venv/bin/activate` (mac/Linux). |

**Still stuck?** Run `python scripts/check_env.py` and read the `FAIL` lines — it diagnoses
missing files, wrong Python versions, uninstalled packages and broken model artifacts.

---

# ✅ Cheat sheet — the whole thing in 6 lines

```bash
cd industrial-failure-prediction      # 1. go to the project folder
python -m venv .venv                  # 2. create the environment
.venv\Scripts\activate                #    Windows  (mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt       # 3. install packages
python scripts/check_env.py           # 4. verify everything
streamlit run app/streamlit_app.py    # 5. launch → http://localhost:8501
```

---

# ⚠️ One last reminder about what you're running

The model was trained on the **simulated** AI4I 2020 dataset. Every number it produces is a
*probability*, not a guaranteed failure:

- A **high** risk score means the sensor pattern resembles historical failures → send someone
  to look at the machine.
- A **low** score does **not** certify that a machine is healthy.

It is a tool for prioritising inspections. It is not a safety system, and it must never
trigger an automatic shutdown.

See `reports/failure_prediction_report.md` §9 for the full list of limitations.
