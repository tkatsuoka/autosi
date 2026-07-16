"""実験用データ生成モジュール。

src/ref/dataset.py の generate_images_corr に倣い、相関ノイズは
共分散 Sigma[i, j] = 0.5^|i-j| (AR(1) 型) を multivariate_normal で生成する。

- iid:        ノイズ共分散が単位行列 (各観測が独立同分布)
- corr: ノイズ共分散が AR(1) 型 (隣接観測ほど強く相関)

signal=0 のとき null 仮説 (FPR / p 値の一様性の検証)、
signal>0 のとき対立仮説 (検出力 power の検証) になる。

selective inference では prob_vec (= y) の共分散行列を `var` として
渡す必要があるため、生成した共分散もあわせて返す。
"""

import numpy as np


def _ar1_cov(n: int, rho: float = 0.5) -> np.ndarray:
    """AR(1) 型共分散行列 Sigma[i, j] = rho^|i-j| を生成する。"""
    idx = np.arange(n)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def generate_data(
    data_type: str,
    n_samples: int,
    n_features: int,
    seed: int,
    signal: float = 0.0,
    n_signal: int = 1,
    rho: float = 0.5,
):
    """実験用の (X, y, Sigma) を生成する。

    Parameters
    ----------
    data_type : "iid" または "corr"
    n_samples : サンプル数
    n_features : 特徴量数
    seed : 乱数シード
    signal : 真の係数の大きさ Δ (0 なら null 仮説)
    n_signal : 信号を載せる先頭特徴量の数 (既定 1、run_experiment の既定と揃える)
    rho : corr 時の AR(1) 相関係数

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    Sigma : np.ndarray | float
        y の共分散。iid なら 1.0 (スカラー)、corr なら共分散行列。
    """
    rng = np.random.default_rng(seed)

    # signal=0 なら beta=0 (null)。signal>0 なら先頭 n_signal 個に Δ を載せる
    # (先頭 n_signal 要素を Δ、残り 0)。
    beta = np.zeros(n_features)
    if signal != 0.0:
        k = min(n_signal, n_features)
        beta[:k] = signal

    X = rng.standard_normal((n_samples, n_features))

    if data_type == "iid":
        # 各観測は独立同分布。共分散はスカラー 1.0 で表現できる。
        noise = rng.standard_normal(n_samples)
        Sigma = 1.0

    elif data_type == "corr":
        # AR(1) 型共分散をもつ相関ノイズを生成 (ref/dataset.py と同方式)。
        Sigma = _ar1_cov(n_samples, rho)
        noise = rng.multivariate_normal(np.zeros(n_samples), Sigma)

    else:
        raise ValueError(f"unknown data_type: {data_type}")

    y = X @ beta + noise
    return X, y, Sigma
