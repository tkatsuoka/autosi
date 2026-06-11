from __future__ import annotations

import numpy as np
from sicore import RealSubset


class PolyMemo:
    """forward_si の z 反復に対する多項式区間計算のメモ化。

    1回の inference()（=1つのp-value）の中では選択直線 a, b が固定で z のみが
    変化するため、各計算地点で生成される多項式の係数は呼び出し間で不変になる。
    これを利用して2層のキャッシュを行う:

    - レベルA（根キャッシュ）: 多項式の根（固有値）は z に依存しないため、係数
      ハッシュをキーに根をキャッシュし、重い eigvals の再計算を省く。
    - レベルB（区間キャッシュ）: iso-sign 区間は「z の周りで結果が一定の最大区間」
      なので、結果区間 [lo, hi] に次の z が含まれれば計算ごとスキップできる。
      z 探索は正・負を反復するため、区間キャッシュは符号別（True/False）に保持する。

    p-value 計算ごとに a, b が変わるので、係数が変わればキー自体が変化し誤った
    再利用は起きない。ただしメモリ膨張を防ぐため inference() 開始時に reset() し、
    各計算地点につき符号別に最新1件のみ保持する。
    """

    def __init__(self) -> None:
        self._enabled = True
        self._roots: dict = {}   # key -> (roots_den, roots_adj)  レベルA
        self._region: dict = {}  # key -> {sign(bool): (lo, hi, result)}  レベルB
        # レベルC（実行全体スキップ）: 最深(最小)の累積区間に z が含まれれば
        # forward_si 全体をスキップ。符号別に最終区間と出力をキャッシュする。
        self._run: dict = {}     # sign(bool) -> (lo, hi, output, intervals)

    # ------------------------------------------------------------------
    # 状態管理
    # ------------------------------------------------------------------

    def set_enabled(self, flag: bool) -> None:
        self._enabled = bool(flag)

    def is_enabled(self) -> bool:
        return self._enabled

    def reset(self) -> None:
        """キャッシュを全消去（p-value 計算の切り替え時に呼ぶ）"""
        self._roots.clear()
        self._region.clear()
        self._run.clear()

    # ------------------------------------------------------------------
    # レベルC: 実行全体のスキップ（最深=最小累積区間ファストパス）
    # ------------------------------------------------------------------

    def run_lookup(self, z: float):
        """新しい z が前回の最終(最小)累積区間に含まれれば (output, intervals) を返す。

        含まれる場合、全制約区間が z を含むため出力・区間は前回と完全一致し、
        forward_si を丸ごとスキップできる。含まれなければ None。
        """
        entry = self._run.get(z >= 0)
        if entry is not None and entry[0] <= z <= entry[1]:
            return entry[2], entry[3]
        return None

    def run_store(self, z: float, output, intervals) -> None:
        """forward_si の結果を符号別にキャッシュする（z を含む連結成分を保持）。"""
        try:
            lo, hi = intervals.find_interval_containing(z)
        except Exception:
            return  # z が区間外（通常起きない）は安全側でキャッシュしない
        self._run[z >= 0] = (lo, hi, output, intervals)

    # ------------------------------------------------------------------
    # メモ化本体
    # ------------------------------------------------------------------

    @staticmethod
    def _key(*arrays: np.ndarray) -> tuple:
        """係数配列群から決定的なハッシュキーを作る。"""
        parts: list = []
        for a in arrays:
            a = np.ascontiguousarray(a, dtype=float)
            parts.append((a.shape, a.tobytes()))
        return tuple(parts)

    def batch_intervals(
        self, denominators: np.ndarray, adj_num: np.ndarray, z: float
    ) -> list[RealSubset]:
        """バッチ有理関数区間のメモ化計算。"""
        from .poly_utils import _poly_roots, _iso_from_roots, _combine_intervals

        key = self._key(denominators, adj_num)
        sign = z >= 0

        # レベルB: 前回区間に z が含まれれば結果をそのまま返す
        region = self._region.get(key)
        if region is not None and sign in region:
            lo, hi, result = region[sign]
            if lo <= z <= hi:
                return result

        # レベルA: 根は z 不変なのでキャッシュから取得（無ければ計算）
        roots = self._roots.get(key)
        if roots is None:
            roots = (_poly_roots(denominators), _poly_roots(adj_num))
            self._roots[key] = roots

        cond_intervals = _iso_from_roots(roots[0], z)  # (n, 2)
        adj_intervals = _iso_from_roots(roots[1], z)   # (n, 2)
        result = _combine_intervals(cond_intervals, adj_intervals)

        # 結果が一定となる区間 = 全多項式の iso 区間の積集合（z を含む）
        lo = max(cond_intervals[:, 0].max(), adj_intervals[:, 0].max())
        hi = min(cond_intervals[:, 1].min(), adj_intervals[:, 1].min())
        self._region.setdefault(key, {})[sign] = (lo, hi, result)
        return result

    def scalar_interval(
        self,
        numerator: np.ndarray,
        denominator: np.ndarray,
        z: float,
        threshold: float,
    ) -> RealSubset:
        """スカラー有理関数区間のメモ化計算（レベルB）。"""
        from .poly_utils import _calc_polynominal_interval

        key = self._key(numerator, denominator) + (float(threshold),)
        sign = z >= 0

        region = self._region.get(key)
        if region is not None and sign in region:
            lo, hi, result = region[sign]
            if lo <= z <= hi:
                return result

        result = _calc_polynominal_interval(numerator, denominator, z, threshold)

        # 結果が一定となる区間を求める: z が解集合内ならその区間、
        # 解集合外なら補集合側の区間（いずれも隣接根に挟まれた一定領域）
        try:
            lo, hi = result.find_interval_containing(z)
        except Exception:
            try:
                lo, hi = (~result).find_interval_containing(z)
            except Exception:
                lo, hi = z, z  # 安全側: キャッシュ再利用しない
        self._region.setdefault(key, {})[sign] = (lo, hi, result)
        return result


# モジュールレベルのシングルトン
_MEMO = PolyMemo()


def is_enabled() -> bool:
    return _MEMO.is_enabled()


def set_enabled(flag: bool) -> None:
    _MEMO.set_enabled(flag)


def reset() -> None:
    _MEMO.reset()


def batch_intervals(denominators, adj_num, z):
    return _MEMO.batch_intervals(denominators, adj_num, z)


def scalar_interval(numerator, denominator, z, threshold):
    return _MEMO.scalar_interval(numerator, denominator, z, threshold)


def run_lookup(z):
    return _MEMO.run_lookup(z)


def run_store(z, output, intervals):
    _MEMO.run_store(z, output, intervals)
