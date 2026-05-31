from __future__ import annotations

import builtins

import numpy as np
from numpy.polynomial.polynomial import polydiv, polytrim
from sicore import RealSubset, polynomial_iso_sign_interval


def poly_add(p1: list[np.ndarray], p2: list[np.ndarray]) -> list[np.ndarray]:
    """多項式の加算（長い方に短い方を加算してコピーを最小化）"""
    d1, d2 = len(p1), len(p2)
    if d1 >= d2:
        result = [c.copy() for c in p1]
        for i in range(d2):
            result[i] = result[i] + p2[i]
    else:
        result = [c.copy() for c in p2]
        for i in range(d1):
            result[i] = result[i] + p1[i]
    return result


def poly_sub(p1: list[np.ndarray], p2: list[np.ndarray]) -> list[np.ndarray]:
    """多項式の減算"""
    d1, d2 = len(p1), len(p2)
    n = builtins.max(d1, d2)
    result = []
    for i in range(n):
        c1 = p1[i] if i < d1 else np.zeros_like(p1[0])
        c2 = p2[i] if i < d2 else np.zeros_like(p2[0])
        result.append(c1 - c2)
    return result


def poly_mul(p1: list[np.ndarray], p2: list[np.ndarray]) -> list[np.ndarray]:
    """多項式の乗算（NumPyベクトル化版）

    - 次数0同士: 直接乗算
    - スカラー係数: np.convolve で高速計算
    - 配列係数: 内側ループをスライス加算でベクトル化（O(d1) Pythonループ）
    """
    d1, d2 = len(p1), len(p2)

    # 定数×定数の最速パス
    if d1 == 1 and d2 == 1:
        return [p1[0] * p2[0]]

    n = d1 + d2 - 1
    shape1 = np.asarray(p1[0]).shape
    shape2 = np.asarray(p2[0]).shape

    # 両方スカラー係数: np.convolve（FFTベース）で高速計算
    if shape1 == () and shape2 == ():
        a = np.array([float(c) for c in p1])
        b = np.array([float(c) for c in p2])
        conv = np.convolve(a, b)
        return [np.array(conv[k]) for k in range(n)]

    # 配列係数: 内側ループをスライス加算でベクトル化
    # result[i+j] += p1[i] * p2[j] → result[i:i+d2] += p1[i] * stacked_p2
    shape = np.broadcast_shapes(shape1, shape2)
    a = np.asarray(p1)  # (d1, *shape1)
    b = np.asarray(p2)  # (d2, *shape2)
    dtype = np.result_type(a.dtype, b.dtype)
    result = np.zeros((n, *shape), dtype=dtype)

    # a[i] * b が result のスライスとブロードキャストできるよう次元を揃える
    # 例: shape1=(n,), shape2=() → b は (d2,) → (d2, 1) に reshape
    ndim_diff = len(shape1) - len(shape2)
    if ndim_diff > 0:
        b = b.reshape((d2,) + shape2 + (1,) * ndim_diff)

    for i in range(d1):
        result[i : i + d2] += a[i] * b  # inner loop を排除
    return [result[k] for k in range(n)]


def _denoms_equal(d1: list[np.ndarray], d2: list[np.ndarray]) -> bool:
    """2つの分母多項式が等しいかを判定"""
    if len(d1) != len(d2):
        return False
    return all(np.array_equal(a, b) for a, b in zip(d1, d2))


def _denom_is_one(coef_den: list[np.ndarray]) -> bool:
    """分母が定数1（すべての要素が1.0の次数0多項式）かを判定"""
    return len(coef_den) == 1 and np.allclose(coef_den[0], 1.0)


def _batch_iso_sign_intervals(coefs: np.ndarray, z: float) -> np.ndarray:
    """複数の多項式の iso-sign 区間をバッチ計算する。

    polynomial_iso_sign_interval を n 回呼ぶ代わりに、コンパニオン行列の
    固有値計算を np.linalg.eigvals のバッチ処理でまとめて行う。

    Args:
        coefs: shape (n, d+1)、各行が1つの多項式の昇順係数
        z: 基準点
    Returns:
        shape (n, 2) の配列 [[lower, upper], ...]
    """
    n, d1 = coefs.shape
    d = d1 - 1
    result = np.empty((n, 2))
    result[:, 0] = -np.inf
    result[:, 1] = np.inf

    if d == 0:
        return result  # 定数多項式: 根なし → (-inf, inf)

    if d == 1:
        # 線形: root = -a/b（解析解で最速）
        a, b = coefs[:, 0], coefs[:, 1]
        valid = np.abs(b) > 1e-15
        roots = np.where(valid, -a / np.where(valid, b, 1.0), np.nan)
        result[:, 0] = np.where(roots < z, roots, -np.inf)
        result[:, 1] = np.where(roots > z, roots, np.inf)
        return result

    # 高次: コンパニオン行列の固有値をバッチ計算
    # companion[k, 1:, :-1] = I(d-1)、companion[k, :, -1] = -coef[k, :-1] / leading[k]
    leading = coefs[:, -1]
    C = np.zeros((n, d, d))
    if d > 1:
        C[:, 1:, :-1] = np.eye(d - 1)  # サブ対角線の1（全ポリノミアル共通）
    safe_leading = np.where(np.abs(leading) > 1e-15, leading, 1.0)
    C[:, :, -1] = -coefs[:, :-1] / safe_leading[:, np.newaxis]

    eigenvalues = np.linalg.eigvals(C)  # (n, d) complex、バッチ固有値計算

    # polynomial_iso_sign_interval に合わせて複素根の実部もそのまま使用
    real_roots = eigenvalues.real  # (n, d)

    # z 未満の最大根 → lower bound
    below = np.where(real_roots < z - 1e-10, real_roots, np.nan)
    has_below = ~np.all(np.isnan(below), axis=1)
    if has_below.any():
        result[has_below, 0] = np.nanmax(below[has_below], axis=1)

    # z 超過の最小根 → upper bound
    above = np.where(real_roots > z + 1e-10, real_roots, np.nan)
    has_above = ~np.all(np.isnan(above), axis=1)
    if has_above.any():
        result[has_above, 1] = np.nanmin(above[has_above], axis=1)

    return result


def calc_polynominal_intervals_batch(
    numerators: np.ndarray,    # (n, deg_num+1) 昇順係数
    denominators: np.ndarray,  # (n, deg_den+1) 昇順係数
    z: float,
    threshold: float = 0.0,
) -> list[RealSubset]:
    """n 個の有理関数 num_i/den_i > threshold となる区間をバッチ計算する。

    polynomial_iso_sign_interval を n 回呼ぶ代わりに、バッチ固有値計算で処理する。
    calc_polynominal_interval の数学的等価バッチ版:
      iso_sign(den, z) & iso_sign(num - den*threshold, z)
    """
    d_num, d_den = numerators.shape[1], denominators.shape[1]
    d_max = max(d_num, d_den)
    if d_num < d_max:
        numerators = np.pad(numerators, ((0, 0), (0, d_max - d_num)))
    if d_den < d_max:
        denominators = np.pad(denominators, ((0, 0), (0, d_max - d_den)))

    adj_num = numerators - denominators * threshold

    # 2回のバッチ計算で n 要素分をまとめて処理
    cond_intervals = _batch_iso_sign_intervals(denominators, z)  # (n, 2)
    adj_intervals  = _batch_iso_sign_intervals(adj_num, z)        # (n, 2)

    return [
        RealSubset([cond_intervals[i].tolist()]) & RealSubset([adj_intervals[i].tolist()])
        for i in range(len(numerators))
    ]


def _poly_gcd_1d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """1次元多項式のGCDをユークリッドアルゴリズムで計算（昇順係数、モニック正規化）"""
    a = np.atleast_1d(polytrim(np.asarray(a, dtype=float)))
    b = np.atleast_1d(polytrim(np.asarray(b, dtype=float)))

    while len(b) > 0 and not np.allclose(b, 0, atol=1e-10):
        _, r = polydiv(a, b)
        r = np.atleast_1d(polytrim(r))
        a, b = b, r

    # モニック正規化（先頭係数を1に）
    if len(a) > 0 and not np.isclose(a[-1], 0, atol=1e-10):
        a = a / a[-1]
    return a if len(a) > 0 else np.array([1.0])


def poly_reduce(
    num: list[np.ndarray], den: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """スカラー有理多項式を既約分数に簡約化する。

    以下の場合は即座にそのまま返す（オーバーヘッドを最小化）:
    - 配列係数（shape != ()）
    - num か den のどちらかが定数（degree 0）→ GCDは必ず定数
    """
    # 配列係数は非対応
    if np.asarray(num[0]).shape != ():
        return num, den

    # どちらかが定数なら多項式GCDは常に定数 → 約分不要
    if len(num) <= 1 or len(den) <= 1:
        return num, den

    a = np.array([float(c) for c in num])
    b = np.array([float(c) for c in den])
    gcd = _poly_gcd_1d(a, b)

    # GCDが定数なら約分不要
    if len(gcd) <= 1:
        return num, den

    q_num, r_num = polydiv(a, gcd)
    q_den, r_den = polydiv(b, gcd)

    # 浮動小数点誤差により割り切れない場合はスキップ
    if not (np.allclose(r_num, 0, atol=1e-10) and np.allclose(r_den, 0, atol=1e-10)):
        return num, den

    q_num = np.atleast_1d(polytrim(q_num))
    q_den = np.atleast_1d(polytrim(q_den))

    if len(q_num) == 0:
        q_num = np.array([0.0])
    if len(q_den) == 0:
        q_den = np.array([1.0])

    return [np.array(c) for c in q_num], [np.array(c) for c in q_den]


def calc_polynominal_interval(
    numerator: np.ndarray,
    denominator: np.ndarray,
    z: float,
    threshold: float = 0.0,
) -> RealSubset:
    """有理関数 > threshold となるzの範囲を計算"""
    if len(numerator) < len(denominator):
        numerator = np.concatenate(
            [numerator, np.zeros(len(denominator) - len(numerator))]
        )
    elif len(denominator) < len(numerator):
        denominator = np.concatenate(
            [denominator, np.zeros(len(numerator) - len(denominator))]
        )

    # denominator > 0 の場合: numerator - denominator * threshold > 0 を解く
    pos_cond = RealSubset(polynomial_iso_sign_interval(denominator, z))
    pos_int = RealSubset(
        polynomial_iso_sign_interval(numerator - denominator * threshold, z)
    )
    intervals = pos_cond & pos_int

    # denominator < 0 の場合: numerator - denominator * threshold < 0 を解く
    neg_cond = RealSubset(polynomial_iso_sign_interval(-denominator, z))
    neg_int = RealSubset(
        polynomial_iso_sign_interval(denominator * threshold - numerator, z)
    )
    intervals = intervals | (neg_cond & neg_int)

    return RealSubset(intervals)
