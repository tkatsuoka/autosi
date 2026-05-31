from __future__ import annotations

import numpy as np

from .interval_tracker import IntervalTracker
from .poly_utils import calc_polynominal_intervals_batch


class ArrayMixin:
    """siarray の配列操作メソッド"""

    @property
    def T(self):
        return type(self).from_rational(
            [c.T for c in self.coef_num],
            [c.T for c in self.coef_den],
        )

    def transpose(self, axes=None):
        return type(self).from_rational(
            [np.transpose(c, axes) for c in self.coef_num],
            [np.transpose(c, axes) for c in self.coef_den],
        )

    def inv(self):
        return type(self).from_rational(
            [np.linalg.inv(c) for c in self.coef_num],
            [c.copy() for c in self.coef_den],
        )

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        new_num = [c.reshape(shape) for c in self.coef_num]
        new_den = [c.reshape(shape) for c in self.coef_den]
        result = type(self).from_rational(new_num, new_den)
        result._shape = shape
        return result

    def argsort(self, axis=-1, kind=None, order=None):
        sorted_idx = np.argsort(self.data, axis=axis, kind=kind, order=order)
        tracker = IntervalTracker.get_instance()
        n = len(sorted_idx) - 1
        if n > 0:
            # 隣接要素の差分をバッチ処理
            diffs = [self[sorted_idx[i]] - self[sorted_idx[i + 1]] for i in range(n)]
            d_num = max(len(d.coef_num) for d in diffs)
            d_den = max(len(d.coef_den) for d in diffs)
            nums = np.zeros((n, d_num))
            dens = np.zeros((n, d_den))
            for i, diff in enumerate(diffs):
                for k, c in enumerate(diff.coef_num):
                    nums[i, k] = float(c)
                for k, c in enumerate(diff.coef_den):
                    dens[i, k] = float(c)
            for interval in calc_polynominal_intervals_batch(nums, dens, tracker.get_z()):
                tracker.update(interval)
        return type(self).from_rational(
            [np.array(sorted_idx)],
            [np.ones_like(sorted_idx)],
        )

    def copy(self):
        return type(self).from_rational(
            [c.copy() for c in self.coef_num],
            [c.copy() for c in self.coef_den],
        )
