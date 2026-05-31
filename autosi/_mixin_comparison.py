from __future__ import annotations

import numpy as np

from .interval_tracker import IntervalTracker
from .poly_utils import calc_polynominal_interval, calc_polynominal_intervals_batch


class ComparisonMixin:
    """siarray の比較演算子"""

    def _track_diff_constraint(self, other, negate: bool) -> np.ndarray:
        """diff = self - other の区間制約を tracker に追加し、diff.data を返す。

        スカラーは単一処理、配列はバッチ処理で全要素の制約を一括追跡する。
        negate=True のとき -diff/den > 0（__lt__, __le__ 用）を追跡する。
        """
        other = self._to_siarray(other)
        diff = self - other
        z = IntervalTracker.get_instance().get_z()
        tracker = IntervalTracker.get_instance()
        n_elems = diff.data.size

        if n_elems == 1:
            # スカラー: 単一処理（軽量パス）
            num = np.array([c.flatten()[0] if c.size > 0 else 0 for c in diff.coef_num])
            den = np.array([c.flatten()[0] if c.size > 0 else 1 for c in diff.coef_den])
            if negate:
                num = -num
            tracker.update(calc_polynominal_interval(num, den, z))
        else:
            # 配列: 全要素をバッチ処理
            d_num, d_den = len(diff.coef_num), len(diff.coef_den)
            nums = np.array([c.flatten() for c in diff.coef_num]).T  # (n_elems, d_num)
            dens = np.array([c.flatten() for c in diff.coef_den]).T  # (n_elems, d_den)
            if negate:
                nums = -nums
            for interval in calc_polynominal_intervals_batch(nums, dens, z):
                tracker.update(interval)

        return diff.data

    def _scalar_or_array(self, result: np.ndarray):
        """スカラーなら bool、配列なら numpy 配列を返す"""
        if result.size == 1:
            return bool(result.item())
        return result

    def __lt__(self, other):
        # diff < 0  ⟺  -diff / den > 0
        data = self._track_diff_constraint(other, negate=True)
        return self._scalar_or_array(data < 0)

    def __le__(self, other):
        data = self._track_diff_constraint(other, negate=True)
        return self._scalar_or_array(data <= 0)

    def __gt__(self, other):
        # diff > 0  ⟺  diff / den > 0
        data = self._track_diff_constraint(other, negate=False)
        return self._scalar_or_array(data > 0)

    def __ge__(self, other):
        data = self._track_diff_constraint(other, negate=False)
        return self._scalar_or_array(data >= 0)

    def __eq__(self, other):
        other = self._to_siarray(other)
        return self._scalar_or_array(np.equal(self.data, other.data))

    def __ne__(self, other):
        other = self._to_siarray(other)
        return self._scalar_or_array(np.not_equal(self.data, other.data))
