"""Download and load the UCI datasets for the real-data experiments.

The seven regression datasets (UCI Machine Learning Repository, CC BY 4.0)
used in the paper. Run `prepare_all()` (or execute this module directly)
once to download the archives and cache cleaned CSVs (feature columns +
last column = response) under data/realdata/. Subsequent loads need no
network access.

Usage:
  python experiments/realdata/datasets.py        # prepare all datasets
"""

import io
import os
import sys
import zipfile
import urllib.request

import numpy as np
import pandas as pd

# Cache location relative to the supplementary root
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(_ROOT, "data", "realdata")

# Static UCI archives
_BASE = "https://archive.ics.uci.edu/static/public"
_SOURCES = {
    "airfoil": f"{_BASE}/291/airfoil+self+noise.zip",
    "concrete": f"{_BASE}/165/concrete+compressive+strength.zip",
    "energy": f"{_BASE}/242/energy+efficiency.zip",
    "real_estate": f"{_BASE}/477/real+estate+valuation+data+set.zip",
    "wine": f"{_BASE}/186/wine+quality.zip",
}

# Dataset names used in the experiments (Table 2 of the paper)
DATASET_NAMES = [
    "airfoil",         # Airfoil Self-Noise
    "concrete",        # Concrete Compressive Strength
    "energy_heating",  # Energy Efficiency (heating load)
    "energy_cooling",  # Energy Efficiency (cooling load)
    "real_estate",     # Real Estate Valuation
    "wine_red",        # Wine Quality (red)
    "wine_white",      # Wine Quality (white)
]


def _fetch_zip(url: str) -> zipfile.ZipFile:
    """Fetch a zip archive from UCI and open it in memory."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return zipfile.ZipFile(io.BytesIO(resp.read()))


def _find_member(zf: zipfile.ZipFile, suffix: str) -> list[str]:
    """List the members of the archive with the given extension."""
    return [n for n in zf.namelist() if n.lower().endswith(suffix)]


def _save(df: pd.DataFrame, name: str) -> None:
    """Save a cleaned CSV (rows with missing values dropped; last column =
    response)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    df = df.dropna()
    df.to_csv(os.path.join(CACHE_DIR, f"{name}.csv"), index=False)
    print(f"  saved {name}.csv  shape={df.shape}")


def _prepare_airfoil() -> None:
    zf = _fetch_zip(_SOURCES["airfoil"])
    member = _find_member(zf, ".dat")[0]
    df = pd.read_csv(zf.open(member), sep=r"\s+", header=None)
    df.columns = ["frequency", "angle", "chord", "velocity",
                  "thickness", "sound_pressure"]
    _save(df, "airfoil")


def _prepare_concrete() -> None:
    zf = _fetch_zip(_SOURCES["concrete"])
    member = _find_member(zf, ".xls")[0]
    df = pd.read_excel(zf.open(member))  # last column = strength (response)
    _save(df, "concrete")


def _prepare_energy() -> None:
    zf = _fetch_zip(_SOURCES["energy"])
    member = _find_member(zf, ".xlsx")[0]
    df = pd.read_excel(zf.open(member)).dropna(how="all", axis=1)
    features = [c for c in df.columns if str(c).startswith("X")]
    # Y1 = heating load, Y2 = cooling load -> two datasets
    _save(df[features + ["Y1"]], "energy_heating")
    _save(df[features + ["Y2"]], "energy_cooling")


def _prepare_real_estate() -> None:
    zf = _fetch_zip(_SOURCES["real_estate"])
    member = _find_member(zf, ".xlsx")[0]
    df = pd.read_excel(zf.open(member))
    df = df.drop(columns=[c for c in df.columns if str(c).strip() == "No"])
    _save(df, "real_estate")  # last column = house price per unit area


def _prepare_wine() -> None:
    zf = _fetch_zip(_SOURCES["wine"])
    for member in _find_member(zf, ".csv"):
        df = pd.read_csv(zf.open(member), sep=";")  # last column = quality
        color = "red" if "red" in member.lower() else "white"
        _save(df, f"wine_{color}")


_PREPARERS = {
    "airfoil": _prepare_airfoil,
    "concrete": _prepare_concrete,
    "energy_heating": _prepare_energy,
    "energy_cooling": _prepare_energy,
    "real_estate": _prepare_real_estate,
    "wine_red": _prepare_wine,
    "wine_white": _prepare_wine,
}


def prepare_all(force: bool = False) -> None:
    """Download and clean all datasets (cached files are skipped)."""
    done = set()
    for name in DATASET_NAMES:
        path = os.path.join(CACHE_DIR, f"{name}.csv")
        if os.path.exists(path) and not force:
            print(f"  skip {name}.csv (cached)")
            continue
        fn = _PREPARERS[name]
        if fn in done:  # energy / wine produce two files per download
            continue
        print(f"preparing {name} ...")
        fn()
        done.add(fn)


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) from a cached CSV; raise if not prepared yet.

    No automatic download here so that compute nodes without network
    access are safe; run prepare_all() beforehand.
    """
    path = os.path.join(CACHE_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found; run "
            "`python experiments/realdata/datasets.py` first."
        )
    df = pd.read_csv(path)
    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df.iloc[:, -1].to_numpy(dtype=float)
    return X, y


if __name__ == "__main__":
    force = "--force" in sys.argv
    prepare_all(force=force)
    for name in DATASET_NAMES:
        X, y = load_dataset(name)
        print(f"{name:16s} X={X.shape} y={y.shape}")
