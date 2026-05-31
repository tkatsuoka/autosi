from __future__ import annotations

import numpy as np

from .interval_tracker import IntervalTracker
from .poly_utils import (
    calc_polynominal_interval,
    calc_polynominal_intervals_batch,
    poly_mul,
    poly_add,
    poly_sub,
    poly_reduce,
    _denoms_equal,
)


class ArithmeticMixin:
    """siarray の算術演算子"""

    def _common_denominator(self):
        """分母を共通化（通分）する。ユニークな分母のみの積を使用。"""
        # 高速チェック: 各次数の係数配列が全要素同じか
        d0 = [c.flat[0] for c in self.coef_den]
        if all(np.allclose(c, d0[deg]) for deg, c in enumerate(self.coef_den)):
            return self

        flat_size = self.coef_num[0].size

        # 各要素がどのユニーク分母に属するかをマッピング
        unique_dens: list = []
        elem_to_unique: list[int] = []

        for i in range(flat_size):
            den = [c.flat[i] for c in self.coef_den]
            found_idx = -1
            for j, u in enumerate(unique_dens):
                if len(u) == len(den) and all(
                    np.isclose(u[k], den[k]) for k in range(len(den))
                ):
                    found_idx = j
                    break
            if found_idx == -1:
                found_idx = len(unique_dens)
                unique_dens.append(den)
            elem_to_unique.append(found_idx)

        # 各ユニーク分母に対して「それ以外の積」を事前計算
        n_unique = len(unique_dens)
        multipliers = []
        for i in range(n_unique):
            mult = [np.array(1.0)]
            for j in range(n_unique):
                if i != j:
                    mult = poly_mul(mult, unique_dens[j])
            multipliers.append(mult)

        common_den = poly_mul(multipliers[0], unique_dens[0])

        # 各要素の分子を調整
        max_deg = len(common_den)
        new_num = [np.zeros(self._shape) for _ in range(max_deg)]

        for i in range(flat_size):
            elem_num = [c.flat[i] for c in self.coef_num]
            adjusted_num = poly_mul(elem_num, multipliers[elem_to_unique[i]])
            idx = np.unravel_index(i, self._shape)
            for deg, coef in enumerate(adjusted_num):
                if deg < max_deg:
                    new_num[deg][idx] = coef

        new_den = [np.full(self._shape, c) for c in common_den]
        return type(self).from_rational(new_num, new_den)

    def __add__(self, other):
        other = self._to_siarray(other)
        # 分母が同じなら分子のみ加算（分母の乗算を回避）
        if _denoms_equal(self.coef_den, other.coef_den):
            return type(self).from_rational(
                poly_add(self.coef_num, other.coef_num),
                list(self.coef_den),
            )
        new_num = poly_add(
            poly_mul(self.coef_num, other.coef_den),
            poly_mul(other.coef_num, self.coef_den),
        )
        new_den = poly_mul(self.coef_den, other.coef_den)
        new_num, new_den = poly_reduce(new_num, new_den)
        return type(self).from_rational(new_num, new_den)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._to_siarray(other)
        # 分母が同じなら分子のみ減算（分母の乗算を回避）
        if _denoms_equal(self.coef_den, other.coef_den):
            return type(self).from_rational(
                poly_sub(self.coef_num, other.coef_num),
                list(self.coef_den),
            )
        new_num = poly_sub(
            poly_mul(self.coef_num, other.coef_den),
            poly_mul(other.coef_num, self.coef_den),
        )
        new_den = poly_mul(self.coef_den, other.coef_den)
        new_num, new_den = poly_reduce(new_num, new_den)
        return type(self).from_rational(new_num, new_den)

    def __rsub__(self, other):
        return self._to_siarray(other).__sub__(self)

    def __mul__(self, other):
        other = self._to_siarray(other)
        return type(self).from_rational(
            poly_mul(self.coef_num, other.coef_num),
            poly_mul(self.coef_den, other.coef_den),
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = self._to_siarray(other)
        if len(other.coef_num) == 1 and len(other.coef_den) == 1:
            # 定数除算: 次数が変わらないため poly_reduce 不要
            new_num = [c / other.coef_num[0] for c in self.coef_num]
            new_den = [c / other.coef_den[0] for c in self.coef_den]
            return type(self).from_rational(new_num, new_den)
        new_num = poly_mul(self.coef_num, other.coef_den)
        new_den = poly_mul(self.coef_den, other.coef_num)
        new_num, new_den = poly_reduce(new_num, new_den)
        return type(self).from_rational(new_num, new_den)

    def __rtruediv__(self, other):
        return self._to_siarray(other).__truediv__(self)

    def __matmul__(self, other):
        other = self._to_siarray(other)

        # 通分する
        self_cd = self._common_denominator()
        other_cd = other._common_denominator()

        n = len(self_cd.coef_num) + len(other_cd.coef_num) - 1
        new_num = [
            np.zeros_like(self_cd.coef_num[0] @ other_cd.coef_num[0])
            for _ in range(n)
        ]
        for i, c1 in enumerate(self_cd.coef_num):
            for j, c2 in enumerate(other_cd.coef_num):
                new_num[i + j] = new_num[i + j] + c1 @ c2

        result_shape = new_num[0].shape
        den_val = self_cd.coef_den[0].flat[0] * other_cd.coef_den[0].flat[0]
        new_den = [np.full(result_shape, den_val)]
        return type(self).from_rational(new_num, new_den)

    def __rmatmul__(self, other):
        return self._to_siarray(other).__matmul__(self)

    def __neg__(self):
        return type(self).from_rational(
            [-c.copy() for c in self.coef_num], self.coef_den
        )

    def __pow__(self, power):
        """累乗演算"""
        power_val = int(power.data.item()) if isinstance(power, type(self)) else power
        result = self
        for _ in range(power_val - 1):
            result = result * self
        return result

    def __abs__(self):
        z = IntervalTracker.get_instance().get_z()
        tracker = IntervalTracker.get_instance()
        positive_idx = (self.data >= 0).view(bool)
        negative_idx = (self.data < 0).view(bool)

        new_num = [
            np.where(positive_idx, self.coef_num[i].copy(), -self.coef_num[i].copy())
            for i in range(len(self.coef_num))
        ]
        new_den = [c.copy() for c in self.coef_den]

        def _update_tracker_batch(coef_num_list, coef_den_list):
            """係数リストをまとめてバッチ処理し tracker を更新"""
            # shape: (n_terms, n_elems) → transpose → (n_elems, n_terms)
            nums = np.array(coef_num_list).T   # (n_elems, n_num_terms)
            dens = np.array(coef_den_list).T   # (n_elems, n_den_terms)
            for interval in calc_polynominal_intervals_batch(nums, dens, z):
                tracker.update(interval)

        if positive_idx.any():
            pos_num = [self.coef_num[i][positive_idx].flatten() for i in range(len(self.coef_num))]
            pos_den = [self.coef_den[i][positive_idx].flatten() for i in range(len(self.coef_den))]
            _update_tracker_batch(pos_num, pos_den)

        if negative_idx.any():
            neg_num = [-self.coef_num[i][negative_idx].flatten() for i in range(len(self.coef_num))]
            neg_den = [self.coef_den[i][negative_idx].flatten() for i in range(len(self.coef_den))]
            _update_tracker_batch(neg_num, neg_den)

        return type(self).from_rational(new_num, new_den)
