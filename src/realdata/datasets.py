"""実データ実験用の UCI データセットのダウンロードと読み込み。

si4pipeline 論文 (Shiraishi et al., 2024) の実データ実験と同じ
8 データセット (UCI Machine Learning Repository, CC BY 4.0) を扱う。

初回に `prepare_all()` (または本モジュールを直接実行) で zip を取得し、
「特徴量列 + 最終列 = 目的変数」の整形済み CSV を data/realdata/ に
キャッシュする。以降の読み込みはネットワーク不要 (計算ノードで安全)。

使い方:
  uv run python src/realdata/datasets.py        # 全データセットを準備
"""

import io
import os
import sys
import zipfile
import urllib.request

import numpy as np
import pandas as pd

# プロジェクトルートからの相対でキャッシュ先を固定する
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE_DIR = os.path.join(_ROOT, "data", "realdata")

# UCI の静的配布 zip。値は (URL, 変換関数名)。
_BASE = "https://archive.ics.uci.edu/static/public"
_SOURCES = {
    "airfoil": f"{_BASE}/291/airfoil+self+noise.zip",
    "concrete": f"{_BASE}/165/concrete+compressive+strength.zip",
    "energy": f"{_BASE}/242/energy+efficiency.zip",
    "gas_turbine": f"{_BASE}/551/gas+turbine+co+and+nox+emission+data+set.zip",
    "real_estate": f"{_BASE}/477/real+estate+valuation+data+set.zip",
    "wine": f"{_BASE}/186/wine+quality.zip",
}

# 実験で使うデータセット名 (論文の Data1..Data8 に対応)
DATASET_NAMES = [
    "airfoil",         # Data1: Airfoil Self-Noise
    "concrete",        # Data2: Concrete Compressive Strength
    "energy_heating",  # Data3: Energy Efficiency (heating load)
    "energy_cooling",  # Data4: Energy Efficiency (cooling load)
    "gas_turbine",     # Data5: Gas Turbine CO and NOx (目的変数 NOX)
    "real_estate",     # Data6: Real Estate Valuation
    "wine_red",        # Data7: Wine Quality (red)
    "wine_white",      # Data8: Wine Quality (white)
]


def _fetch_zip(url: str) -> zipfile.ZipFile:
    """UCI から zip を取得してメモリ上で開く。"""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return zipfile.ZipFile(io.BytesIO(resp.read()))


def _find_member(zf: zipfile.ZipFile, suffix: str) -> list[str]:
    """zip 内から指定拡張子のファイル名を列挙する。"""
    return [n for n in zf.namelist() if n.lower().endswith(suffix)]


def _save(df: pd.DataFrame, name: str) -> None:
    """欠損行を除いた整形済み CSV (最終列 = 目的変数) を保存する。"""
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
    df = pd.read_excel(zf.open(member))  # 最終列が強度 (目的変数)
    _save(df, "concrete")


def _prepare_energy() -> None:
    zf = _fetch_zip(_SOURCES["energy"])
    member = _find_member(zf, ".xlsx")[0]
    df = pd.read_excel(zf.open(member)).dropna(how="all", axis=1)
    features = [c for c in df.columns if str(c).startswith("X")]
    # Y1 = heating load, Y2 = cooling load で 2 データセットに分ける
    _save(df[features + ["Y1"]], "energy_heating")
    _save(df[features + ["Y2"]], "energy_cooling")


def _prepare_gas_turbine() -> None:
    zf = _fetch_zip(_SOURCES["gas_turbine"])
    members = sorted(_find_member(zf, ".csv"))  # gt_2011.csv .. gt_2015.csv
    df = pd.concat([pd.read_csv(zf.open(m)) for m in members], ignore_index=True)
    # 目的変数は NOX。CO はもう一方の排出量指標なので特徴量からは除く。
    features = [c for c in df.columns if c not in ("CO", "NOX")]
    _save(df[features + ["NOX"]], "gas_turbine")


def _prepare_real_estate() -> None:
    zf = _fetch_zip(_SOURCES["real_estate"])
    member = _find_member(zf, ".xlsx")[0]
    df = pd.read_excel(zf.open(member))
    df = df.drop(columns=[c for c in df.columns if str(c).strip() == "No"])
    _save(df, "real_estate")  # 最終列が単位面積あたり住宅価格


def _prepare_wine() -> None:
    zf = _fetch_zip(_SOURCES["wine"])
    for member in _find_member(zf, ".csv"):
        df = pd.read_csv(zf.open(member), sep=";")  # 最終列が quality
        color = "red" if "red" in member.lower() else "white"
        _save(df, f"wine_{color}")


_PREPARERS = {
    "airfoil": _prepare_airfoil,
    "concrete": _prepare_concrete,
    "energy_heating": _prepare_energy,
    "energy_cooling": _prepare_energy,
    "gas_turbine": _prepare_gas_turbine,
    "real_estate": _prepare_real_estate,
    "wine_red": _prepare_wine,
    "wine_white": _prepare_wine,
}


def prepare_all(force: bool = False) -> None:
    """全データセットをダウンロードして整形済み CSV を作る (キャッシュ優先)。"""
    done = set()
    for name in DATASET_NAMES:
        path = os.path.join(CACHE_DIR, f"{name}.csv")
        if os.path.exists(path) and not force:
            print(f"  skip {name}.csv (cached)")
            continue
        fn = _PREPARERS[name]
        if fn in done:  # energy / wine は 1 回の準備で 2 ファイル作られる
            continue
        print(f"preparing {name} ...")
        fn()
        done.add(fn)


def load_dataset(name: str) -> tuple[np.ndarray, np.ndarray]:
    """整形済み CSV から (X, y) を返す。未取得ならエラーにする。

    計算ノードにネットワークが無い場合を考慮し、ここでは自動ダウンロード
    しない。事前に prepare_all() を実行しておくこと。
    """
    path = os.path.join(CACHE_DIR, f"{name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} が見つかりません。先に "
            "`uv run python src/realdata/datasets.py` を実行してください。"
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
