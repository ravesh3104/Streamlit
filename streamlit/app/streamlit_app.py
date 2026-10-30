"""Dashboard application (Streamlit).

Scope (user-approved):
  1. Machine Risk Check   - single prediction with a risk gauge + "why" panel
  2. Batch Scoring        - upload a CSV, score every row, download results
  3. Model Evaluation     - precision / recall / F1 / confusion matrices / CIs
  4. Patterns & Drivers   - EDA charts + feature influence explanations

Every prediction is framed as a probability, never a certainty. A persistent
amber banner repeats this on every tab.

Run:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import io
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import cleaning, config as C, modeling  # noqa: E402

st.set_page_config(
    page_title="Industrial Failure Prediction",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Theme
# --------------------------------------------------------------------------
CSS = """
<style>
:root{
  --bg:#0b1220; --panel:#111d31; --panel2:#16243b; --line:#22344f;
  --text:#e8eff9; --muted:#93a7c4; --teal:#2dd4bf; --amber:#fbbf24;
  --red:#ef4444; --blue:#60a5fa;
}
.stApp{background:radial-gradient(1200px 600px at 15% -10%,#16243b 0%,#0b1220 55%),#0b1220;}
section[data-testid="stSidebar"]{background:#0d1728;border-right:1px solid var(--line);}
html,body,[class*="css"]{color:var(--text);font-family:'Inter','Segoe UI',system-ui,sans-serif;}
h1,h2,h3,h4{color:var(--text)!important;letter-spacing:-.02em;}
.block-container{padding-top:1.6rem;max-width:1420px;}
.hero{background:linear-gradient(120deg,#12243c 0%,#0f1c2e 60%,#132a44 100%);
  border:1px solid var(--line);border-radius:16px;padding:22px 26px;margin-bottom:14px;
  box-shadow:0 10px 30px rgba(0,0,0,.35);}
.hero h1{margin:0;font-size:1.65rem;font-weight:800;}
.hero p{margin:.35rem 0 0;color:var(--muted);font-size:.92rem;}
.pill{display:inline-block;padding:3px 11px;border-radius:999px;font-size:.72rem;
  font-weight:700;letter-spacing:.04em;text-transform:uppercase;}
.pill-teal{background:rgba(45,212,191,.14);color:var(--teal);border:1px solid rgba(45,212,191,.4);}
.pill-amber{background:rgba(251,191,36,.13);color:var(--amber);border:1px solid rgba(251,191,36,.4);}
.pill-red{background:rgba(239,68,68,.14);color:#fca5a5;border:1px solid rgba(239,68,68,.45);}
.banner{background:rgba(251,191,36,.09);border:1px solid rgba(251,191,36,.38);
  border-left:4px solid var(--amber);border-radius:10px;padding:11px 16px;
  color:#fde68a;font-size:.84rem;margin:8px 0 16px;line-height:1.5;}
.card{background:var(--panel);border:1px solid var(--line);border-radius:14px;
  padding:18px 20px;box-shadow:0 6px 18px rgba(0,0,0,.25);}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:13px;
  padding:14px 16px;text-align:left;height:100%;}
.kpi .lbl{color:var(--muted);font-size:.71rem;text-transform:uppercase;letter-spacing:.08em;}
.kpi .val{font-size:1.5rem;font-weight:800;margin-top:2px;}
.kpi .sub{color:var(--muted);font-size:.73rem;margin-top:2px;}
.risk{padding:20px 22px;border-radius:14px;border:1px solid;margin-bottom:12px;}
.risk-low{background:rgba(45,212,191,.08);border-color:rgba(45,212,191,.45);}
.risk-med{background:rgba(251,191,36,.08);border-color:rgba(251,191,36,.45);}
.risk-high{background:rgba(239,68,68,.09);border-color:rgba(239,68,68,.5);}
.risk .head{font-size:1.35rem;font-weight:800;margin-bottom:2px;}
.risk .body{color:var(--muted);font-size:.86rem;}
.why{background:var(--panel2);border:1px solid var(--line);border-radius:10px;
  padding:10px 14px;margin-bottom:8px;font-size:.86rem;color:var(--text);}
.why b{color:var(--teal);}
.sev-up{color:var(--red);font-weight:700;}
.sev-down{color:var(--teal);font-weight:700;}
.sev-neutral{color:var(--muted);font-weight:700;}
.small{color:var(--muted);font-size:.78rem;line-height:1.55;}
.stTabs [data-baseweb="tab-list"]{gap:6px;background:transparent;border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{background:var(--panel);border:1px solid var(--line);
  border-bottom:none;border-radius:10px 10px 0 0;padding:9px 18px;color:var(--muted);
  font-weight:600;font-size:.86rem;}
.stTabs [aria-selected="true"]{background:var(--panel2)!important;color:var(--text)!important;
  border-color:rgba(45,212,191,.5)!important;box-shadow:inset 0 2px 0 var(--teal);}
div[data-testid="stMetricValue"]{color:var(--text);}
.stButton>button{background:linear-gradient(90deg,#0f766e,#0d9488);color:#eafffb;
  border:1px solid #14b8a6;border-radius:10px;font-weight:700;padding:.5rem 1.1rem;}
.stButton>button:hover{background:linear-gradient(90deg,#0d9488,#14b8a6);color:#021a17;}
.stDownloadButton>button{background:var(--panel2);color:var(--teal);
  border:1px solid rgba(45,212,191,.45);border-radius:10px;font-weight:600;}
hr{border-color:var(--line);}
code{background:#0d1728;color:#7dd3fc;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Load artifacts
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_bundle():
    if not C.MODELS_PATH.exists():
        return None
    # A model pickled by a different scikit-learn version triggers a noisy
    # InconsistentVersionWarning. It is almost always harmless (and fixing it is
    # a 60-second retrain), so it is captured and reported politely below
    # rather than printed as a wall of red text.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bundle = joblib.load(C.MODELS_PATH)
        mismatches = [
            str(w.message).strip() for w in caught
            if "InconsistentVersionWarning" in type(w.message).__name__
        ]
    bundle["_version_warnings"] = mismatches
    return bundle


@st.cache_data(show_spinner=False)
def load_results():
    if not C.RESULTS_PATH.exists():
        return {}
    return json.loads(C.RESULTS_PATH.read_text())


@st.cache_data(show_spinner=False)
def load_data():
    if not C.CLEAN_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(C.CLEAN_CSV)


def fig(name: str):
    path = C.FIG_DIR / name
    if path.exists():
        st.image(str(path), width='stretch')
    return path.exists()


BUNDLE = load_bundle()
RESULTS = load_results()
DF = load_data()

if BUNDLE is None:
    st.error(
        "No trained models found.\n\nRun `python -m src.train` (or `make train`) "
        "to build them, then reload this page."
    )
    st.stop()

MODELS = BUNDLE["models"]
BEST_KEY = BUNDLE["best_model"]
THRESHOLDS = BUNDLE["tuned_thresholds"]
REF = BUNDLE["training_reference"]
METRICS = BUNDLE["metrics_summary"]
MODEL_LABELS = modeling.MODEL_LABELS


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def risk_band(p: float, thr: float):
    """Three bands anchored on the model's own tuned alarm threshold."""
    if p >= thr:
        return "HIGH", "risk-high", "Above the tuned alarm threshold"
    if p >= thr * 0.5:
        return "ELEVATED", "risk-med", "Below threshold, but worth watching"
    return "LOW", "risk-low", "Well below the tuned alarm threshold"


def banner():
    st.markdown(
        f'<div class="banner"><b>⚠️ Decision support, not certainty.</b> {C.DISCLAIMER}</div>',
        unsafe_allow_html=True,
    )


def kpi(label, value, sub="", colour="var(--text)"):
    st.markdown(
        f'<div class="kpi"><div class="lbl">{label}</div>'
        f'<div class="val" style="color:{colour}">{value}</div>'
        f'<div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def make_row(air, proc, rpm, torque, wear, mtype, udi=1, serial=1) -> pd.DataFrame:
    """One raw row in the same schema as the raw CSV.

    The row is pushed through the SAME cleaning + feature + design-matrix code
    used in training, which removes any train/serve skew.
    """
    base = {
        "UDI": udi,
        "Product ID": f"{mtype}{int(serial):05d}",
        "Type": mtype,
        "Air temperature [K]": air,
        "Process temperature [K]": proc,
        "Rotational speed [rpm]": rpm,
        "Torque [Nm]": torque,
        "Tool wear [min]": wear,
        C.TARGET: 0,
        "TWF": 0, "HDF": 0, "PWF": 0, "OSF": 0, "RNF": 0,
    }
    raw = pd.DataFrame([base])
    cleaned_df, _, _ = cleaning.clean(raw)
    featured = cleaning.build_features(cleaned_df)
    X, _, _ = modeling.build_design_matrix(featured)
    # A single-row frame only contains the dummy column for the type present
    # (e.g. only Type_M), so align explicitly to the training schema. This is
    # what prevents train/serve skew.
    return X.reindex(columns=BUNDLE["feature_names"], fill_value=0.0)


def score_row(X: pd.DataFrame):
    out = {}
    for key, model in MODELS.items():
        p = float(model.predict_proba(X)[:, 1][0])
        out[key] = p
    return out


def explain_features(X: pd.DataFrame, model_key: str, top_n: int = 6):
    """Native, per-prediction attribution.

    Logistic Regression: exact additive log-odds contribution.
    Tree models: (feature value - training median) x global importance, signed
    by the direction of the partial-dependence slope. This is an approximation
    and is labelled as such in the UI.
    """
    feats = BUNDLE["feature_names"]
    exp = RESULTS.get("models", {}).get(model_key, {}).get("explanation", {})
    row = X.iloc[0]

    if model_key == "logistic_regression" and "lr_coefficients" in exp:
        scaler = MODELS[model_key].named_steps["scaler"]
        clf = MODELS[model_key].named_steps["clf"]
        z = scaler.transform(X)[0]
        contrib = z * clf.coef_[0]
        ranked = sorted(zip(feats, contrib), key=lambda kv: -abs(kv[1]))[:top_n]
        return [
            {
                "feature": f,
                "value": float(row[f]),
                "effect": float(v),
                "direction": "raises risk" if v > 0 else "lowers risk",
                "method": "exact log-odds contribution",
            }
            for f, v in ranked
        ]

    imp = exp.get("permutation_importance", {}).get("ranked", {})
    pdp = exp.get("partial_dependence", {})
    contribs = []
    for f in feats:
        w = imp.get(f, 0.0)
        med = REF["sensor_medians"].get(f, float(DF[f].median()) if f in DF else 0.0)
        delta = float(row[f]) - float(med)
        # direction: use PDP slope where available, else sign of delta
        sign = np.sign(delta) if delta != 0 else 0.0
        if f in pdp and len(pdp[f].get("x", [])) > 2:
            ys = np.array(pdp[f]["y"])
            sign = np.sign(ys[-1] - ys[0]) * sign if sign != 0 else np.sign(ys[-1] - ys[0])
        contribs.append((f, w * abs(delta) * (sign if sign != 0 else 1.0), delta, w))
    contribs.sort(key=lambda t: -abs(t[1]))
    return [
        {
            "feature": f,
            "value": float(row[f]),
            "effect": float(c),
            "direction": "raises risk" if c > 0 else "lowers risk",
            "method": f"deviation x global importance (weight {w:.4f})",
        }
        for f, c, d, w in contribs[:top_n]
    ]


def risk_gauge_svg(p: float, thr: float) -> str:
    """Hand-drawn SVG gauge - no external assets, works in a sandboxed iframe."""
    pct = max(0.0, min(1.0, p))
    angle = -90 + 180 * pct
    thr_angle = -90 + 180 * max(0.0, min(1.0, thr))
    colour = "#ef4444" if p >= thr else ("#fbbf24" if p >= thr * 0.5 else "#2dd4bf")
    import math

    def pt(a, r):
        rad = math.radians(a)
        return 100 + r * math.sin(rad), 100 - r * math.cos(rad)

    x1, y1 = pt(-90, 78)
    x2, y2 = pt(angle, 78)
    large = 1 if angle > 0 else 0
    tx1, ty1 = pt(thr_angle, 62)
    tx2, ty2 = pt(thr_angle, 92)
    return f"""
<svg viewBox="0 0 200 172" width="100%" style="max-width:360px">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#2dd4bf"/><stop offset="55%" stop-color="#fbbf24"/>
      <stop offset="100%" stop-color="#ef4444"/>
    </linearGradient>
  </defs>
  <path d="M {pt(-90,78)[0]} {pt(-90,78)[1]} A 78 78 0 0 1 {pt(90,78)[0]} {pt(90,78)[1]}"
        fill="none" stroke="#1c2c45" stroke-width="15" stroke-linecap="round"/>
  <path d="M {x1} {y1} A 78 78 0 {large} 1 {x2} {y2}"
        fill="none" stroke="url(#g)" stroke-width="15" stroke-linecap="round"/>
  <line x1="{tx1}" y1="{ty1}" x2="{tx2}" y2="{ty2}" stroke="#e8eff9"
        stroke-width="2.2" stroke-dasharray="3,3" opacity="0.85"/>
  <text x="22" y="126" fill="#64748b" font-size="9"
        font-family="Inter,system-ui">0%</text>
  <text x="178" y="126" fill="#64748b" font-size="9" text-anchor="end"
        font-family="Inter,system-ui">100%</text>
  <text x="100" y="118" text-anchor="middle" fill="{colour}" font-size="30"
        font-weight="800" font-family="Inter,system-ui">{p*100:.1f}%</text>
  <text x="100" y="140" text-anchor="middle" fill="#93a7c4" font-size="9.5"
        letter-spacing="1.2" font-family="Inter,system-ui">FAILURE PROBABILITY</text>
  <text x="100" y="160" text-anchor="middle" fill="#64748b" font-size="9"
        font-family="Inter,system-ui">alarm threshold {thr*100:.0f}%</text>
</svg>"""


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
best_label = MODEL_LABELS[BEST_KEY]
st.markdown(
    f"""<div class="hero">
  <span class="pill pill-teal">Predictive maintenance</span>
  <span class="pill pill-amber">AI4I 2020 · simulated</span>
  <h1>⚙️ Industrial Equipment Failure Prediction</h1>
  <p>Logistic Regression vs Decision Tree vs Random Forest · {REF['n_rows']:,} machine records ·
     {REF['failure_rate']:.2%} historical failure rate · default predictor:
     <b>{best_label}</b> (selected on held-out F1)</p>
</div>""",
    unsafe_allow_html=True,
)
banner()

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Controls")
    chosen = st.selectbox(
        "Prediction model",
        options=list(MODELS),
        index=list(MODELS).index(BEST_KEY),
        format_func=lambda k: f"{MODEL_LABELS[k]}"
        + (" ⭐ best" if k == BEST_KEY else ""),
        help="⭐ marks the model with the highest tuned F1 on the held-out test set.",
    )
    thr = st.slider(
        "Alarm threshold",
        min_value=0.05, max_value=0.95,
        value=float(THRESHOLDS[chosen]), step=0.01,
        help="Probability above which a unit is flagged for inspection. "
             "Default is the F1-optimal threshold learned from training folds only.",
    )
    st.caption(
        f"Tuned default for {MODEL_LABELS[chosen]}: **{THRESHOLDS[chosen]:.3f}**. "
        "Lower it to catch more failures (more false alarms); raise it to alarm less."
    )
    st.markdown("---")
    st.markdown("### 📊 Model at a glance")
    for k in MODELS:
        m = METRICS[k]["test_at_tuned"]
        star = " ⭐" if k == BEST_KEY else ""
        st.markdown(
            f"**{MODEL_LABELS[k]}**{star}  \n"
            f"<span class='small'>F1 {m['f1']:.3f} · P {m['precision']:.3f} · "
            f"R {m['recall']:.3f}</span>",
            unsafe_allow_html=True,
        )
    st.markdown("---")
    if BUNDLE.get("_version_warnings"):
        import sklearn

        trained_with = BUNDLE.get("library_versions", {}).get("scikit-learn", "an older version")
        st.info(
            f"**Library version note** — the saved models were built with "
            f"scikit-learn {trained_with}; your environment has {sklearn.__version__}. "
            "This is harmless and the predictions are still valid. For an exact match, "
            "run `python -m src.train` (about 60 seconds) to rebuild the models."
        )
    st.markdown(
        "<span class='small'>Threshold and model choices only affect this view. "
        "Saved metrics were computed on the untouched test set.</span>",
        unsafe_allow_html=True,
    )

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------
# Each view is a function and ONLY the selected one is executed.
# This matters because a Streamlit page is pushed to the browser in full on
# first paint: rendering all four views eagerly meant downloading 14 charts
# (~1.2 MB) plus several dataframes before anything became interactive, which
# made the app appear stuck on its loading skeleton in a proxied preview.


def render_risk_check() -> None:
    """Single-machine risk assessment."""
    left, right = st.columns([1.05, 1.0], gap="large")

    with left:
        st.markdown("#### Sensor readings")
        st.caption("Defaults are the dataset medians. Move the sliders to simulate a machine.")
        c1, c2 = st.columns(2)
        air = c1.slider("Air temperature [K]", 295.0, 305.0, float(REF["sensor_medians"]["Air temperature [K]"]), 0.1)
        proc = c2.slider("Process temperature [K]", 305.0, 314.0, float(REF["sensor_medians"]["Process temperature [K]"]), 0.1)
        rpm = c1.slider("Rotational speed [rpm]", 1168, 2886, int(REF["sensor_medians"]["Rotational speed [rpm]"]), 1)
        torque = c2.slider("Torque [Nm]", 3.8, 76.6, float(REF["sensor_medians"]["Torque [Nm]"]), 0.1)
        wear = st.slider("Tool wear [min]", 0, 253, int(REF["sensor_medians"]["Tool wear [min]"]), 1)
        mtype = st.radio("Machine type", ["L", "M", "H"], horizontal=True, index=1,
                         help="L = low grade (most common), H = high grade (rarest, most failures per unit).")

        preset = st.selectbox(
            "Or load a scenario",
            ["Custom", "High-power overstrain", "Worn tool under load",
             "Heat-dissipation risk", "Healthy reference"],
        )
        if preset == "High-power overstrain":
            air, proc, rpm, torque, wear, mtype = 299.0, 309.5, 1330, 68.0, 60, "L"
            st.info("Torque is very high at low speed → the classic overstrain signature.")
        elif preset == "Worn tool under load":
            air, proc, rpm, torque, wear, mtype = 300.0, 310.0, 1450, 55.0, 235, "M"
            st.info("Tool wear near its limit while still under load.")
        elif preset == "Heat-dissipation risk":
            air, proc, rpm, torque, wear, mtype = 303.8, 313.1, 1300, 62.0, 180, "H"
            st.info("Large air/process temperature gap with high torque at low speed.")
        elif preset == "Healthy reference":
            air, proc, rpm, torque, wear, mtype = 298.2, 308.7, 1500, 40.0, 20, "M"
            st.info("Typical healthy operating point.")

        X_row = make_row(air, proc, rpm, torque, wear, mtype)

    probs = score_row(X_row)
    p = probs[chosen]
    band, band_class, band_note = risk_band(p, thr)
    colour = {"LOW": "#2dd4bf", "ELEVATED": "#fbbf24", "HIGH": "#ef4444"}[band]
    verdict = (
        "Flag for inspection" if p >= thr else "No immediate action"
    )

    with right:
        st.markdown("#### Prediction")
        st.markdown(
            f'<div class="risk {band_class}">'
            f'<div class="head" style="color:{colour}">{verdict} · {band} risk</div>'
            f'<div class="body">Estimated failure probability '
            f'<b style="color:{colour}">{p:.1%}</b> from {MODEL_LABELS[chosen]} at a '
            f'{thr:.0%} alarm threshold. {band_note}.</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(risk_gauge_svg(p, thr), unsafe_allow_html=True)
        st.caption(
            "Dashed line = current alarm threshold. The needle is a probability, "
            "not a diagnosis."
        )

        st.markdown("##### All three models on this reading")
        cmp_df = pd.DataFrame(
            {
                "Model": [MODEL_LABELS[k] for k in MODELS],
                "Failure probability": [probs[k] for k in MODELS],
                "Flagged": ["🔴 Yes" if probs[k] >= THRESHOLDS[k] else "🟢 No" for k in MODELS],
            }
        )
        st.dataframe(
            cmp_df.style.format({"Failure probability": "{:.1%}"}),
            hide_index=True, width='stretch',
        )
        spread = max(probs.values()) - min(probs.values())
        if spread > 0.3:
            st.warning(
                f"The models disagree by {spread:.0%} on this reading. "
                "That is a signal to inspect manually rather than to trust any single model."
            )
        else:
            st.info(f"Models agree within {spread:.0%} — a relatively stable reading.")

    st.markdown("---")
    st.markdown(f"#### Why did {MODEL_LABELS[chosen]} produce this number?")
    st.caption(
        "These are the inputs that moved this specific prediction the most. "
        "Directions are indicative, not causal."
    )
    why = explain_features(X_row, chosen)
    maxw = max([abs(w["effect"]) for w in why] + [1e-9])
    cols = st.columns(len(why))
    for col, w in zip(cols, why):
        cls = "sev-up" if w["direction"] == "raises risk" else "sev-down"
        arrow = "▲" if w["direction"] == "raises risk" else "▼"
        bar_pct = int(60 * abs(w["effect"]) / maxw) + 6
        col.markdown(
            f'<div class="why"><div class="small">{w["feature"]}</div>'
            f'<div style="font-weight:800;font-size:1.05rem">{w["value"]:.2f}</div>'
            f'<div class="{cls}">{arrow} {w["direction"]}</div>'
            f'<div style="height:4px;border-radius:2px;margin-top:6px;'
            f'background:{"#ef4444" if w["direction"]=="raises risk" else "#2dd4bf"};'
            f'width:{bar_pct}%"></div></div>',
            unsafe_allow_html=True,
        )
    st.caption(
        f"Method: {why[0]['method'] if why else 'n/a'}. "
        "Engineered features (power, wear-torque, temperature gap) are derived from the sensors above."
    )

    with st.expander("🔬 Cross-check: compare this reading against the training data"):
        a, b, c = st.columns(3)
        a.metric("Mechanical power", f"{X_row['Mechanical power [kW]'].iloc[0]:.2f} kW",
                 f"median {DF['Mechanical power [kW]'].median():.2f} kW")
        b.metric("Temp difference", f"{X_row['Temp difference [K]'].iloc[0]:.1f} K",
                 f"median {DF['Temp difference [K]'].median():.1f} K")
        c.metric("Wear-torque", f"{X_row['Wear-torque interaction'].iloc[0]:.0f}",
                 f"median {DF['Wear-torque interaction'].median():.0f}")
        pct = (DF["Mechanical power [kW]"] < X_row["Mechanical power [kW]"].iloc[0]).mean()
        st.caption(
            f"This reading's mechanical power is higher than **{pct:.0%}** of the "
            "10,000 historical records."
        )

# ==========================================================================

def render_batch_scoring() -> None:
    """Batch CSV scoring."""
    st.markdown("#### Score a whole file of machines at once")
    st.caption(
        "Upload a CSV with the raw sensor columns. The same cleaning pipeline "
        "used in training is applied, so messy files are handled and every fix is reported."
    )
    required = ["Type"] + C.SENSOR_COLS
    with st.expander("Expected columns (minimum required)"):
        st.code("\n".join(required), language="text")
        st.caption(
            "`UDI` and `Product ID` are optional. Any extra columns (including a "
            "ground-truth `Machine failure`) are carried through untouched."
        )

    up = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")
    use_sample = st.button("▶ Use the 2,000-row holdout sample instead")

    batch = None
    if up is not None:
        try:
            batch = pd.read_csv(up)
            st.success(f"Loaded {len(batch):,} rows × {batch.shape[1]} columns from {up.name}")
        except Exception as exc:
            st.error(f"Could not read that file: {exc}")
    elif use_sample and len(DF):
        batch = DF.sample(min(2000, len(DF)), random_state=C.RANDOM_STATE).drop(
            columns=[C.TARGET], errors="ignore"
        )
        st.success(f"Loaded a {len(batch):,}-row random sample of the cleaned dataset.")

    if batch is not None:
        missing = [c for c in required if c not in batch.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            truth = batch[C.TARGET] if C.TARGET in batch.columns else None
            to_clean = batch.drop(columns=[C.TARGET], errors="ignore").copy()
            cleaned_b, csum, _ = cleaning.clean(to_clean)
            feat_b = cleaning.build_features(cleaned_b)
            Xb, _, _ = modeling.build_design_matrix(feat_b)
            Xb = Xb.reindex(columns=BUNDLE["feature_names"], fill_value=0)

            pb = MODELS[chosen].predict_proba(Xb)[:, 1]
            out_b = batch.copy()
            out_b["failure_probability"] = pb
            out_b["flagged_for_inspection"] = (pb >= thr).astype(int)
            out_b["risk_band"] = np.where(
                pb >= thr, "HIGH", np.where(pb >= thr * 0.5, "ELEVATED", "LOW")
            )

            n_flag = int(out_b["flagged_for_inspection"].sum())
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                kpi("Rows scored", f"{len(out_b):,}", f"model: {MODEL_LABELS[chosen]}")
            with k2:
                kpi("Flagged for inspection", f"{n_flag:,}",
                    f"{n_flag/len(out_b):.1%} of rows", "#ef4444" if n_flag else "#2dd4bf")
            with k3:
                kpi("Mean probability", f"{pb.mean():.1%}", f"threshold {thr:.0%}")
            with k4:
                if truth is not None and len(truth) == len(out_b):
                    tp = int(((pb >= thr) & (truth == 1)).sum())
                    fn = int(((pb < thr) & (truth == 1)).sum())
                    rec = tp / (tp + fn) if tp + fn else 0.0
                    kpi("Recall on this file", f"{rec:.1%}", f"{tp}/{tp+fn} real failures caught")
                else:
                    kpi("Ranking", "—", "add a 'Machine failure' column to score accuracy")

            if truth is not None and len(truth) == len(out_b):
                from sklearn.metrics import confusion_matrix, f1_score, precision_score
                pred = out_b["flagged_for_inspection"]
                cm = confusion_matrix(truth, pred, labels=[0, 1])
                st.markdown("##### How this file actually scored")
                m1, m2, m3 = st.columns(3)
                m1.metric("Precision", f"{precision_score(truth, pred, zero_division=0):.3f}")
                m2.metric("Recall", f"{f1_score(truth, pred, zero_division=0):.3f}" if False else
                          f"{tp/(tp+fn) if (tp+fn) else 0:.3f}")
                m3.metric("F1", f"{f1_score(truth, pred, zero_division=0):.3f}")
                st.caption(
                    f"Confusion matrix on uploaded file — TN {cm[0,0]}, FP {cm[0,1]}, "
                    f"FN {cm[1,0]}, TP {cm[1,1]}."
                )

            st.markdown("##### Cleaning actions applied to your file")
            acts = [a for a in csum["log"] if a["rows_affected"] > 0]
            if acts:
                st.dataframe(
                    pd.DataFrame([{"Step": a["step"], "What happened": a["detail"],
                                   "Rows": a["rows_affected"]} for a in acts]),
                    hide_index=True, width='stretch',
                )
            else:
                st.caption("No repairs were needed — the file was already clean.")

            st.markdown("##### Scored results (highest risk first)")
            show_cols = [c for c in out_b.columns if c not in ("failure_probability",)]
            st.dataframe(
                out_b.sort_values("failure_probability", ascending=False)[show_cols]
                .style.format({"failure_probability": "{:.3f}"})
                .background_gradient(subset=["failure_probability"], cmap="YlOrRd"),
                hide_index=True, width='stretch', height=420,
            )

            buf = io.StringIO()
            out_b.to_csv(buf, index=False)
            st.download_button(
                "⬇ Download scored CSV", buf.getvalue(),
                file_name="scored_machines.csv", mime="text/csv",
            )
            st.caption(
                "Only the rows above the threshold need human attention. "
                "A flag is a request for inspection — not proof of a fault."
            )

# ==========================================================================

def render_evaluation() -> None:
    """Model evaluation metrics."""
    st.markdown("#### How well do the three models actually work?")
    st.caption(
        "Everything here is measured on the held-out 20% test set that was never "
        "used for training or threshold selection."
    )
    rows = []
    for k in MODELS:
        m05 = METRICS[k]["test_at_0.5"]
        mt = METRICS[k]["test_at_tuned"]
        rows.append(
            {
                "Model": MODEL_LABELS[k] + (" ⭐" if k == BEST_KEY else ""),
                "Threshold": f"{mt['threshold']:.2f}",
                "Precision": mt["precision"], "Recall": mt["recall"], "F1": mt["f1"],
                "PR-AUC": mt["pr_auc"], "ROC-AUC": mt["roc_auc"],
                "MCC": mt["mcc"],
                "F1 @0.50": m05["f1"],
            }
        )
    st.dataframe(
        pd.DataFrame(rows).style.format(
            {c: "{:.3f}" for c in ["Precision", "Recall", "F1", "PR-AUC", "ROC-AUC", "MCC", "F1 @0.50"]}
        ).highlight_max(subset=["F1", "PR-AUC"], color="rgba(45,212,191,.25)"),
        hide_index=True, width='stretch',
    )
    st.caption(
        "Accuracy is deliberately absent: predicting 'healthy' for every row scores "
        "96.6% accuracy while catching zero failures."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Confusion matrices")
        fig("07_confusion_matrices.png")
        st.caption(
            "Bottom-right is a caught failure (TP). Bottom-left is a missed failure (FN) — "
            "the expensive error in maintenance."
        )
    with c2:
        st.markdown("##### Precision-Recall & ROC")
        fig("08_pr_roc.png")
        st.caption("PR-AUC is the fairer summary when only 3.4% of rows are failures.")

    st.markdown("##### Where the alarm threshold should sit")
    fig("09_thresholds.png")
    st.caption(
        "Thresholds were tuned on training folds only; this chart shows how the "
        "test-set trade-off behaves, not how the threshold was chosen."
    )

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### Model comparison")
        fig("10_model_comparison.png")
    with c4:
        st.markdown("##### Random vs time-ordered split")
        fig("14_split_comparison.png")
        st.caption(
            "Training on the past and testing on the future is stricter. If the bars "
            "dropped a lot, the random split was flattering the model."
        )

    st.markdown("##### Uncertainty: bootstrap confidence intervals (best model)")
    ci = RESULTS.get("models", {}).get(BEST_KEY, {}).get("test_ci_at_tuned", {})
    if ci:
        cols = st.columns(4)
        for col, key in zip(cols, ["f1", "precision", "recall", "pr_auc"]):
            if key in ci:
                col.metric(
                    key.upper().replace("_", "-"),
                    f"{ci[key]['mean']:.3f}",
                    f"95% CI {ci[key]['low']:.3f}–{ci[key]['high']:.3f}",
                    delta_color="off",
                )
        st.caption(
            f"From {C.N_BOOTSTRAP} bootstrap resamples of the test set. Wide intervals "
            "are the honest consequence of having only 68 failures in the test set."
        )

# ==========================================================================

def render_patterns() -> None:
    """EDA and explainability charts."""
    st.markdown("#### What patterns exist, and which inputs drive the model?")
    st.caption("Left: what the data looks like. Right: what the models keyed on.")

    st.markdown("##### The dataset at a glance")
    a, b, c, d = st.columns(4)
    with a:
        kpi("Records", f"{REF['n_rows']:,}", "one row = one machine run")
    with b:
        kpi("Historical failures", f"{REF['failure_rate']:.2%}", "339 of 10,000", "#ef4444")
    with c:
        kpi("Features used", f"{len(BUNDLE['feature_names'])}",
            "sensors + engineered + type")
    with d:
        kpi("Best held-out F1", f"{METRICS[BEST_KEY]['test_at_tuned']['f1']:.3f}",
            MODEL_LABELS[BEST_KEY], "#2dd4bf")

    st.markdown("##### Class imbalance")
    fig("01_class_imbalance.png")
    st.caption(
        "Only 3.4% of records are failures. This is why precision/recall are reported "
        "instead of accuracy, and why class weights are used during training."
    )

    st.markdown("##### Sensor distributions, split by outcome")
    fig("02_sensor_distributions.png")
    st.caption(
        "Failures cluster at higher torque and lower rotational speed. Temperature "
        "shifts are subtle — a single sensor rarely tells the story."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Torque vs speed")
        fig("05_torque_speed.png")
        st.caption("The iso-power curves show why high torque at low speed is dangerous.")
    with c2:
        st.markdown("##### Correlations")
        fig("03_correlation.png")

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("##### Wear and machine type")
        fig("06_wear_type.png")
    with c4:
        st.markdown("##### Failure modes (descriptive only)")
        fig("04_failure_modes.png")
        st.caption(
            "⚠️ These five columns are excluded from every model — they would leak the answer."
        )

    st.markdown("---")
    st.markdown("##### Which inputs influence the predictions?")
    fig("11_feature_importance.png")
    st.caption(
        "Permutation importance (left) is model-agnostic: it measures how much PR-AUC "
        "drops when a feature is shuffled. Gini importance (right) is faster but biased "
        "toward high-cardinality features — where the two disagree, trust the left chart."
    )

    c5, c6 = st.columns([1, 1])
    with c5:
        st.markdown("##### Logistic Regression coefficients")
        fig("12_lr_coefficients.png")
        st.caption(
            "Standardised, so comparable in size. Red increases failure odds, teal decreases them."
        )
    with c6:
        st.markdown("##### Partial dependence")
        fig("13_partial_dependence.png")
        st.caption(
            "How predicted risk moves as one sensor changes with everything else held fixed."
        )

    st.markdown("##### Do the ID columns matter? (you chose to keep them as inputs)")
    imp = RESULTS.get("models", {}).get(BEST_KEY, {}).get("explanation", {}) \
        .get("permutation_importance", {}).get("ranked", {})
    if imp:
        id_rows = [{"Feature": k, "Permutation importance": v}
                   for k, v in imp.items() if k in ("UDI", "Product ID serial")]
        if id_rows:
            st.dataframe(
                pd.DataFrame(id_rows).style.format({"Permutation importance": "{:.5f}"}),
                hide_index=True, width='stretch',
            )
            total = sum(abs(v) for v in imp.values()) or 1
            id_share = sum(abs(r["Permutation importance"]) for r in id_rows) / total
            st.caption(
                f"The two identifier features account for **{id_share:.2%}** of total "
                "permutation importance. Their near-zero contribution is the evidence that "
                "row IDs carry no predictive signal — set "
                "`USE_ID_COLUMNS_AS_FEATURES = False` in `src/config.py` to drop them."
            )

    st.markdown("##### Data cleaning audit")
    audit = RESULTS.get("dataset", {})
    cl = RESULTS.get("cleaning", {})
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Missing cells found", f"{audit.get('missing_total', 0):,}")
    cc2.metric("Duplicate rows found", f"{audit.get('duplicate_rows', 0):,}")
    cc3.metric("Rows removed", f"{cl.get('rows_removed', 0):,}")
    with st.expander("Show every cleaning action (on the real data)"):
        log = cl.get("log", [])
        if log:
            st.dataframe(
                pd.DataFrame([{"Step": a["step"], "What happened": a["detail"],
                               "Rows affected": a["rows_affected"]} for a in log]),
                hide_index=True, width='stretch',
            )
    with st.expander("Show the corrupted-copy pipeline test"):
        ct = cl.get("corruption_test", {})
        if ct:
            st.markdown(
                f"The pipeline was validated on a deliberately damaged copy of the dataset "
                f"({ct['injected_missing']} injected missing cells, "
                f"{ct['injected_duplicates']} injected duplicate rows, plus impossible "
                f"readings and bad labels). Rows went from {ct['rows_before']:,} to "
                f"{ct['rows_after']:,}, and all injected duplicates were recovered: "
                f"**{ct['pipeline_recovered']}**."
            )
            st.dataframe(
                pd.DataFrame([{"Step": a["step"], "What happened": a["detail"],
                               "Rows": a["rows_affected"]} for a in ct["actions"]]),
                hide_index=True, width='stretch',
            )

    st.markdown("---")
    st.markdown("##### Known limitations (read before trusting any number on this page)")
    st.markdown(
        """
<div class="card small">
<b>1. The data is simulated.</b> AI4I 2020 was generated with a physics-inspired model, not
collected from a real factory. Real sensor streams have drift, correlated sensors, dropouts
and operator interventions that this dataset does not.<br><br>
<b>2. Failures are rare and few.</b> 339 failures total, 68 in the test set. Every metric here
carries a wide confidence interval — a single rerun can move F1 several points.<br><br>
<b>3. Random failure (RNF) is unlearnable by construction.</b> 19 records fail at random with no
sensor signature. No model can predict them, and they cap achievable recall.<br><br>
<b>4. Labels are imperfect.</b> 18 rows flag a failure mode while labelled healthy, and 9
failures have no mode flagged. The published target was preserved unchanged rather than
"fixed", so some label noise remains.<br><br>
<b>5. This is not a causal model.</b> Feature importance shows what the model used, not what
physically causes failure. Do not use it to justify a setpoint change without engineering review.<br><br>
<b>6. Thresholds are policy, not truth.</b> The tuned cutoff balances precision and recall on
one particular test set. A real deployment must re-tune it against the actual cost of a missed
failure versus a wasted inspection.<br><br>
<b>7. A prediction is a probability, never a guarantee.</b> A "HIGH risk" reading means the
model saw patterns resembling historical failures — the machine may be perfectly fine, and a
"LOW risk" reading does not certify that it is safe.
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.caption(
        f"Dataset: AI4I 2020 Predictive Maintenance (UCI ML Repository, ID 601). "
        f"Generated {RESULTS.get('generated_at', 'n/a')} · "
        f"random_state={C.RANDOM_STATE} · {C.CV_FOLDS}-fold CV · "
        f"class_weight='balanced'."
    )



# --------------------------------------------------------------------------
# Navigation - only the active view is rendered
# --------------------------------------------------------------------------
_VIEWS = [
    ("risk", "🔧 Machine Risk Check", render_risk_check),
    ("batch", "📁 Batch Scoring", render_batch_scoring),
    ("eval", "📈 Model Evaluation", render_evaluation),
    ("patterns", "🔍 Patterns & Drivers", render_patterns),
]

_labels = [v[1] for v in _VIEWS]
if hasattr(st, "pills"):
    choice = st.pills("View", _labels, default=_labels[0],
                      label_visibility="collapsed", key="view")
else:  # streamlit < 1.40
    choice = st.radio("View", _labels, horizontal=True,
                      label_visibility="collapsed", key="view")
if choice is None:
    choice = _labels[0]

for _key, _label, _fn in _VIEWS:
    if choice == _label:
        _fn()
        break
