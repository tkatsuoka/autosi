"""実験用データ生成モジュール。

相関構造は共分散 Sigma[i, j] = 0.5^|i-j| (AR(1) 型) を使う。

- iid:  ノイズ共分散が単位行列、X も iid (各観測が独立同分布)
- corr: ノイズと X の各列の両方が行方向に AR(1) で相関する時系列的な
        依存データ (隣接観測ほど強く相関)

signal=0 のとき null 仮説 (FPR / p 値の一様性の検証)、
signal>0 のとき対立仮説 (検出力 power の検証) になる。

ロバスト性実験用に、iid ではノイズを非ガウス分布に差し替えられる
(noise / distance)。分散を同じデータから推定する設定には estimate_sigma2 を使う。

selective inference では prob_vec (= y) の共分散行列を `var` として
渡す必要があるため、生成した共分散もあわせて返す。
"""

from functools import lru_cache

import numpy as np
from sicore import generate_non_gaussian_rv

# ノイズ分布の選択肢。gaussian 以外は sicore の非ガウス分布族で、いずれも
# 平均 0・分散 1 に標準化されており、N(0, 1) からの 1-Wasserstein 距離
# (distance) で非ガウス性の強さを指定する
NOISES = ["gaussian", "skewnorm", "exponnorm", "gennormsteep", "gennormflat", "t"]


@lru_cache(maxsize=None)
def _non_gaussian_rv(noise: str, distance: float):
    """標準化済みの非ガウス分布を返す (同じ設定の再計算を避けるためキャッシュ)。"""
    return generate_non_gaussian_rv(noise, distance)


def estimate_sigma2(X: np.ndarray, y: np.ndarray) -> float:
    """OLS 残差からノイズ分散を推定する: ||y - X(X^T X)^{-1} X^T y||^2 / (n - d)。

    合成データは切片を持たないため自由度は n - d
    (実データ版 run_realdata._estimate_sigma2 は中心化して切片分も控除する)。
    """
    n, d = X.shape
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta
    return float(resid @ resid / (n - d))


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
    n_signal: int = 3,
    rho: float = 0.5,
    noise: str = "gaussian",
    distance: float = 0.0,
):
    """実験用の (X, y, Sigma) を生成する。

    Parameters
    ----------
    data_type : "iid" または "corr"
    n_samples : サンプル数
    n_features : 特徴量数
    seed : 乱数シード
    signal : 真の係数の大きさ Δ (0 なら null 仮説)
    n_signal : 信号を載せる先頭特徴量の数 (既定 3、run_experiment の既定と揃える)
    rho : corr 時の AR(1) 相関係数
    noise : ノイズ分布 (NOISES のいずれか)。gaussian 以外は iid のみ対応
    distance : 非ガウスノイズの N(0, 1) からの 1-Wasserstein 距離

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    Sigma : np.ndarray | float
        y の共分散。iid なら 1.0 (スカラー)、corr なら共分散行列。
    """
    if noise not in NOISES:
        raise ValueError(f"unknown noise: {noise} (choices: {NOISES})")
    if noise != "gaussian" and data_type != "iid":
        raise ValueError("非ガウスノイズは iid のみ対応")

    rng = np.random.default_rng(seed)

    # signal=0 なら beta=0 (null)。signal>0 なら先頭 n_signal 個に Δ を載せる
    # (先頭 n_signal 要素を Δ、残り 0)。
    beta = np.zeros(n_features)
    if signal != 0.0:
        k = min(n_signal, n_features)
        beta[:k] = signal

    if data_type == "iid":
        # 各観測は独立同分布。共分散はスカラー 1.0 で表現できる。
        # X を先に生成するので、同じ seed なら noise によらず X は共通になる。
        X = rng.standard_normal((n_samples, n_features))
        if noise == "gaussian":
            eps = rng.standard_normal(n_samples)
        else:
            # 非ガウスノイズ。分散 1 に標準化済みなので Sigma は 1.0 のまま
            eps = _non_gaussian_rv(noise, distance).rvs(
                size=n_samples, random_state=rng
            )
        Sigma = 1.0

    elif data_type == "corr":
        # 時系列的な依存データ: ノイズだけでなく X の各列も同じ AR(1) 構造で
        # 行方向に相関させる (各列の周辺分散は 1 のまま)。X も相関させるのは、
        # 分割された半分どうしが共変量経由でも結び付き、データ分割法の
        # 「選択と検定の独立性」が実際に破れるようにするため。proposed は
        # X を固定計画として扱い y の共分散 Sigma を厳密に使うので影響しない。
        Sigma = _ar1_cov(n_samples, rho)
        L = np.linalg.cholesky(Sigma)
        X = L @ rng.standard_normal((n_samples, n_features))
        eps = rng.multivariate_normal(np.zeros(n_samples), Sigma)

    else:
        raise ValueError(f"unknown data_type: {data_type}")

    y = X @ beta + eps
    return X, y, Sigma
