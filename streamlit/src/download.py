"""Download (or refresh) the raw AI4I 2020 dataset from UCI.

The repository already bundles `data/raw_ai4i2020.csv`, so this script is only
needed to restore or refresh that file. It falls back to a clear error message
if the machine has no network access.

Run:  python -m src.download
"""
from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

from . import config as C

URL = "https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip"
FALLBACK = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"


def download(force: bool = False) -> Path:
    if C.RAW_CSV.exists() and not force:
        print(f"already present: {C.RAW_CSV} ({C.RAW_CSV.stat().st_size:,} bytes)")
        print("use force=True to re-download")
        return C.RAW_CSV

    print(f"downloading {URL} ...")
    try:
        with urllib.request.urlopen(URL, timeout=60) as resp:
            payload = resp.read()
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            name = [n for n in zf.namelist() if n.endswith(".csv")][0]
            with zf.open(name) as fh:
                df = pd.read_csv(fh)
        print(f"extracted {name} from the archive")
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"  archive download failed ({exc}); trying direct CSV ...")
        try:
            df = pd.read_csv(FALLBACK)
        except Exception as exc2:
            print(f"ERROR: could not download the dataset: {exc2}", file=sys.stderr)
            print("The bundled copy at data/raw_ai4i2020.csv is still usable.", file=sys.stderr)
            raise

    C.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.RAW_CSV, index=False)
    print(f"saved {C.RAW_CSV} ({len(df):,} rows x {df.shape[1]} columns)")
    return C.RAW_CSV


if __name__ == "__main__":
    download(force="--force" in sys.argv)
