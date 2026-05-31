from __future__ import annotations

import numpy as np

from .poly_utils import poly_add, poly_mul, _denom_is_one


class ReductionMixin:
    """siarray の集約メソッド（sum, mean）"""

    def sum(self, axis=None, keepdims=False):
        """配列要素の合計（有理関数版）"""
        if axis is None:
            return self._sum_all()
        return self._sum_axis(axis, keepdims)

    def _sum_all(self):
        """全要素の合計"""
        # 分母がすべて定数1の高速パス: 各次数の係数配列を直接sum
        if _denom_is_one(self.coef_den):
            result_num = [np.sum(c) for c in self.coef_num]
            return type(self).from_rational(result_num, [np.ones(())])

        acc_num = [np.zeros(())]
        acc_den = [np.ones(())]
        for idx in np.ndindex(self._shape):
            elem_num = [c[idx] if c.size > 1 else c.item() for c in self.coef_num]
            elem_den = [c[idx] if c.size > 1 else c.item() for c in self.coef_den]
            new_num = poly_add(
                poly_mul(acc_num, elem_den), poly_mul(elem_num, acc_den)
            )
            acc_num = new_num
            acc_den = poly_mul(acc_den, elem_den)
        return type(self).from_rational(acc_num, acc_den)

    def _sum_axis(self, axis: int, keepdims: bool):
        """特定の軸に沿った合計"""
        if axis < 0:
            axis = len(self._shape) + axis
        if axis < 0 or axis >= len(self._shape):
            raise ValueError(
                f"axis {axis} is out of bounds for array of dimension {len(self._shape)}"
            )

        if keepdims:
            result_shape = tuple(
                1 if i == axis else s for i, s in enumerate(self._shape)
            )
        else:
            result_shape = tuple(s for i, s in enumerate(self._shape) if i != axis)

        result_shape_for_init = result_shape if result_shape else (1,)
        axis_size = self._shape[axis]

        # 分母がすべて定数1の高速パス: axis方向にnp.sumを適用するだけ
        if _denom_is_one(self.coef_den):
            result_num = [np.sum(c, axis=axis, keepdims=keepdims) for c in self.coef_num]
            if result_shape == ():
                result_num = [c.item() * np.ones(()) for c in result_num]
            return type(self).from_rational(result_num, [np.ones(result_shape_for_init)])

        # 最悪ケースの次数を見積もって結果バッファを初期化
        max_num_deg = (len(self.coef_num) - 1) * axis_size
        max_den_deg = (len(self.coef_den) - 1) * axis_size
        result_num = [np.zeros(result_shape_for_init) for _ in range(max_num_deg + 1)]
        result_den = [np.zeros(result_shape_for_init) for _ in range(max_den_deg + 1)]

        for result_idx in np.ndindex(result_shape_for_init):
            acc_num = [np.zeros(())]
            acc_den = [np.ones(())]

            for i in range(axis_size):
                if keepdims:
                    src_idx = list(result_idx)
                    src_idx[axis] = i
                else:
                    src_idx = list(result_idx)
                    src_idx.insert(axis, i)
                src_idx = tuple(src_idx)

                elem_num = [c[src_idx] if c.size > 1 else c.item() for c in self.coef_num]
                elem_den = [c[src_idx] if c.size > 1 else c.item() for c in self.coef_den]
                new_num = poly_add(
                    poly_mul(acc_num, elem_den), poly_mul(elem_num, acc_den)
                )
                acc_num = new_num
                acc_den = poly_mul(acc_den, elem_den)

            for deg, coef in enumerate(acc_num):
                result_num[deg][result_idx] = coef
            for deg, coef in enumerate(acc_den):
                result_den[deg][result_idx] = coef

        # 末尾のゼロ係数を削除
        while len(result_num) > 1 and np.allclose(result_num[-1], 0):
            result_num.pop()
        while len(result_den) > 1 and np.allclose(result_den[-1], 0):
            result_den.pop()

        # 形状を調整
        if result_shape == ():
            result_num = [c.item() * np.ones(()) for c in result_num]
            result_den = [c.item() * np.ones(()) for c in result_den]
        elif not keepdims:
            result_num = [c.reshape(result_shape) for c in result_num]
            result_den = [c.reshape(result_shape) for c in result_den]

        return type(self).from_rational(result_num, result_den)

    def mean(self, axis=None, keepdims=False):
        """配列要素の平均（有理関数版）"""
        if axis is None:
            return self.sum(axis=None, keepdims=False) / np.prod(self._shape)

        if axis < 0:
            axis = len(self._shape) + axis
        if axis < 0 or axis >= len(self._shape):
            raise ValueError(
                f"axis {axis} is out of bounds for array of dimension {len(self._shape)}"
            )
        return self.sum(axis=axis, keepdims=keepdims) / self._shape[axis]
