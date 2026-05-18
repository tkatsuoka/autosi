from __future__ import annotations

import builtins
import numpy as np

from sicore import RealSubset, polynomial_iso_sign_interval


class IntervalTracker:
    """グローバルなintervals管理用のシングルトンクラス"""

    _instance = None
    _intervals = None
    _z = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self, z: float | None = None):
        """intervalsとzをリセット"""
        self._intervals = RealSubset([[-float("inf"), float("inf")]])
        self._z = z

    def get(self):
        """現在のintervalsを取得"""
        if self._intervals is None:
            self.reset()
        return self._intervals

    def get_z(self):
        """現在のzを取得"""
        return self._z

    def set_z(self, z: float):
        """zを設定"""
        self._z = z

    def update(self, new_intervals):
        """intervalsを更新（積集合）"""
        if self._intervals is None:
            self._intervals = new_intervals
        else:
            self._intervals = self._intervals & new_intervals
        return self._intervals

    def get_state(self):
        """現在の状態（intervalsとz）を取得"""
        return {"intervals": self._intervals, "z": self._z}


class siarray:
    def __init__(
        self,
        X: np.ndarray | float | int | "siarray" | None = None,
        numerator: list[np.ndarray] | None = None,
        denominator: list[np.ndarray] | None = None,
    ):
        """
        Parameters
        ----------
        X : np.ndarray, float, int, or siarray, optional
            ラップする配列またはスカラー値
        numerator : list[np.ndarray], optional
            有理関数の分子係数
        denominator : list[np.ndarray], optional
            有理関数の分母係数
        """
        if isinstance(X, siarray):
            # siarrayからコピー
            self.coef_num = [c.copy() for c in X.coef_num]
            self.coef_den = [c.copy() for c in X.coef_den]
            self._shape = X._shape
        elif numerator is not None:
            # 係数を直接指定
            self.coef_num = list(numerator)
            self.coef_den = (
                list(denominator)
                if denominator is not None
                else [np.ones_like(self.coef_num[0])]
            )
            self._shape = numerator[0].shape
        else:
            # 通常の値: 0次の多項式として扱う
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            self.coef_num = [X]
            self.coef_den = [np.ones_like(X)]
            self._shape = X.shape

    # ------------------------------------------------------------------------
    # ファクトリメソッド
    # ------------------------------------------------------------------------

    @classmethod
    def constant(cls, value: np.ndarray | float | int) -> "siarray":
        """定数ノードを作成"""
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        obj = cls.__new__(cls)
        obj.coef_num = [value]
        obj.coef_den = [np.ones_like(value)]
        obj._shape = value.shape
        return obj

    @classmethod
    def from_rational(
        cls, numerator: list[np.ndarray], denominator: list[np.ndarray] | None = None
    ) -> "siarray":
        """RationalFunctionとして作成"""
        obj = cls.__new__(cls)
        obj.coef_num = list(numerator)
        obj.coef_den = (
            list(denominator)
            if denominator is not None
            else [np.ones_like(numerator[0])]
        )
        obj._shape = numerator[0].shape
        return obj

    # ------------------------------------------------------------------------
    # ヘルパーメソッド
    # ------------------------------------------------------------------------

    def _to_siarray(self, other) -> "siarray":
        """他の値をsiarrayオブジェクトに変換"""
        if isinstance(other, siarray):
            return other
        return siarray.constant(other)

    @property
    def data(self) -> np.ndarray:
        """内部のNumPy配列を返す"""
        z = IntervalTracker.get_instance().get_z()
        if z is None:
            z = 0.0  # z が設定されていない場合はデフォルト値
        coef_num_terms = [self.coef_num[i] * (z**i) for i in range(len(self.coef_num))]
        coef_den_terms = [self.coef_den[i] * (z**i) for i in range(len(self.coef_den))]
        coef_num_val = np.sum(coef_num_terms, axis=0)
        coef_den_val = np.sum(coef_den_terms, axis=0)

        return (coef_num_val / coef_den_val).reshape(self._shape)

    def __int__(self):
        """整数への変換（range等で使用）"""
        if self.size != 1:
            raise TypeError("only size-1 arrays can be converted to Python scalars")
        return int(self.data.item())

    def __bool__(self):
        """真偽値への変換（if文で使用）"""
        if self.size != 1:
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous"
            )
        return bool(self.data.item())

    def __index__(self):
        """インデックスとして使用（range等で使用）"""
        if self.size != 1:
            raise TypeError(
                "only integer scalar arrays can be converted to a scalar index"
            )
        return int(self.data.item())

    # ------------------------------------------------------------------------
    # 多項式演算ヘルパー
    # ------------------------------------------------------------------------

    @staticmethod
    def _poly_add(p1: list[np.ndarray], p2: list[np.ndarray]) -> list[np.ndarray]:
        n = builtins.max(len(p1), len(p2))
        result = []
        for i in range(n):
            c1 = p1[i] if i < len(p1) else np.zeros_like(p1[0])
            c2 = p2[i] if i < len(p2) else np.zeros_like(p2[0])
            result.append(c1 + c2)
        return result

    @staticmethod
    def _poly_sub(p1: list[np.ndarray], p2: list[np.ndarray]) -> list[np.ndarray]:
        n = builtins.max(len(p1), len(p2))
        result = []
        for i in range(n):
            c1 = p1[i] if i < len(p1) else np.zeros_like(p1[0])
            c2 = p2[i] if i < len(p2) else np.zeros_like(p2[0])
            result.append(c1 - c2)
        return result

    @staticmethod
    def _poly_mul(p1: list[np.ndarray], p2: list[np.ndarray]) -> list[np.ndarray]:
        n = len(p1) + len(p2) - 1
        result = [np.zeros_like(p1[0]) for _ in range(n)]
        for i, c1 in enumerate(p1):
            for j, c2 in enumerate(p2):
                result[i + j] = result[i + j] + c1 * c2
        return result

    # ------------------------------------------------------------------------
    # 算術演算子
    # ------------------------------------------------------------------------

    def __add__(self, other):
        other = self._to_siarray(other)
        new_coef_num = self._poly_add(
            self._poly_mul(self.coef_num, other.coef_den),
            self._poly_mul(other.coef_num, self.coef_den),
        )
        new_coef_den = self._poly_mul(self.coef_den, other.coef_den)
        return siarray.from_rational(new_coef_num, new_coef_den)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._to_siarray(other)
        new_coef_num = self._poly_sub(
            self._poly_mul(self.coef_num, other.coef_den),
            self._poly_mul(other.coef_num, self.coef_den),
        )
        new_coef_den = self._poly_mul(self.coef_den, other.coef_den)
        return siarray.from_rational(new_coef_num, new_coef_den)

    def __rsub__(self, other):
        return self._to_siarray(other).__sub__(self)

    def __mul__(self, other):
        other = self._to_siarray(other)
        new_coef_num = self._poly_mul(self.coef_num, other.coef_num)
        new_coef_den = self._poly_mul(self.coef_den, other.coef_den)
        return siarray.from_rational(new_coef_num, new_coef_den)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = self._to_siarray(other)
        if len(other.coef_num) == 1 and len(other.coef_den) == 1:
            # オーバーフローを防ぐため，定数による割り算は直接計算
            new_coef_num = [c / other.coef_num[0] for c in self.coef_num]
            new_coef_den = [c / other.coef_den[0] for c in self.coef_den]
        else:
            new_coef_num = self._poly_mul(self.coef_num, other.coef_den)
            new_coef_den = self._poly_mul(self.coef_den, other.coef_num)
        return siarray.from_rational(new_coef_num, new_coef_den)

    def __rtruediv__(self, other):
        return self._to_siarray(other).__truediv__(self)

    def _common_denominator(self) -> "siarray":
        """分母を共通化（通分）する。ユニークな分母のみの積を使用。"""
        # 高速チェック: 各次数の係数配列が全要素同じか
        d0 = [c.flat[0] for c in self.coef_den]
        all_same = all(np.allclose(c, d0[deg]) for deg, c in enumerate(self.coef_den))
        if all_same:
            return self

        flat_size = self.coef_num[0].size

        # 各要素がどのユニーク分母に属するかをマッピング
        unique_dens = []
        elem_to_unique = []

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
                    mult = self._poly_mul(mult, unique_dens[j])
            multipliers.append(mult)

        # 共通分母 = 全ユニーク分母の積
        common_den = self._poly_mul(multipliers[0], unique_dens[0])

        # 各要素の分子を調整
        max_deg = len(common_den)
        new_num = [np.zeros(self._shape) for _ in range(max_deg)]

        for i in range(flat_size):
            elem_num = [c.flat[i] for c in self.coef_num]
            mult = multipliers[elem_to_unique[i]]
            adjusted_num = self._poly_mul(elem_num, mult)

            idx = np.unravel_index(i, self._shape)
            for deg, coef in enumerate(adjusted_num):
                if deg < max_deg:
                    new_num[deg][idx] = coef

        new_den = [np.full(self._shape, c) for c in common_den]

        return siarray.from_rational(new_num, new_den)

    def __matmul__(self, other):
        other = self._to_siarray(other)

        # 通分する
        self = self._common_denominator()
        other = other._common_denominator()

        n = len(self.coef_num) + len(other.coef_num) - 1
        new_coef_num = [
            np.zeros_like(self.coef_num[0] @ other.coef_num[0]) for _ in range(n)
        ]
        for i, c1 in enumerate(self.coef_num):
            for j, c2 in enumerate(other.coef_num):
                new_coef_num[i + j] = new_coef_num[i + j] + c1 @ c2

        result_shape = new_coef_num[0].shape
        den_val = self.coef_den[0].flat[0] * other.coef_den[0].flat[0]
        new_coef_den = [np.full(result_shape, den_val)]

        return siarray.from_rational(new_coef_num, new_coef_den)

    def __rmatmul__(self, other):
        """右側からの行列積"""
        return self._to_siarray(other).__matmul__(self)

    def __neg__(self):
        new_coef_num = [-c.copy() for c in self.coef_num]
        return siarray.from_rational(new_coef_num, self.coef_den)

    def __pow__(self, power):
        """累乗演算"""
        if isinstance(power, siarray):
            power_val = int(power.data.item())
        else:
            power_val = power

        if isinstance(self, siarray):
            result = self
            for _ in range(power_val - 1):
                result = result * self
            return result
        else:
            result_data = self.data**power_val
            return siarray.constant(result_data)

    def __abs__(self):
        z = IntervalTracker.get_instance().get_z()
        tracker = IntervalTracker.get_instance()
        positive_idx = (self.data >= 0).view(bool)
        negative_idx = (self.data < 0).view(bool)

        new_coef_num = [
            np.where(positive_idx, self.coef_num[i].copy(), -self.coef_num[i].copy())
            for i in range(len(self.coef_num))
        ]
        new_coef_den = [c.copy() for c in self.coef_den]

        if positive_idx.any():
            pos_coef_num = np.array(
                [
                    self.coef_num[i][positive_idx].flatten()
                    for i in range(len(self.coef_num))
                ]
            )
            pos_coef_den = np.array(
                [
                    self.coef_den[i][positive_idx].flatten()
                    for i in range(len(self.coef_den))
                ]
            )

            for j in range(pos_coef_num.shape[1] if pos_coef_num.ndim > 1 else 1):
                _num = pos_coef_num[:, j] if pos_coef_num.ndim > 1 else pos_coef_num
                _den = pos_coef_den[:, j] if pos_coef_den.ndim > 1 else pos_coef_den
                if len(_num) < len(_den):
                    _num = np.concatenate([_num, np.zeros(len(_den) - len(_num))])
                elif len(_den) < len(_num):
                    _den = np.concatenate([_den, np.zeros(len(_num) - len(_den))])
                pos_intervals = calc_polynominal_interval(_num, _den, z)
                tracker.update(pos_intervals)

        if negative_idx.any():
            neg_coef_num = np.array(
                [
                    -self.coef_num[i][negative_idx].flatten()
                    for i in range(len(self.coef_num))
                ]
            )
            neg_coef_den = np.array(
                [
                    self.coef_den[i][negative_idx].flatten()
                    for i in range(len(self.coef_den))
                ]
            )
            for j in range(neg_coef_num.shape[1] if neg_coef_num.ndim > 1 else 1):
                _num = neg_coef_num[:, j] if neg_coef_num.ndim > 1 else neg_coef_num
                _den = neg_coef_den[:, j] if neg_coef_den.ndim > 1 else neg_coef_den
                if len(_num) < len(_den):
                    _num = np.concatenate([_num, np.zeros(len(_den) - len(_num))])
                elif len(_den) < len(_num):
                    _den = np.concatenate([_den, np.zeros(len(_num) - len(_den))])
                neg_intervals = calc_polynominal_interval(_num, _den, z)
                tracker.update(neg_intervals)

        return siarray.from_rational(new_coef_num, new_coef_den)

    # ------------------------------------------------------------------------
    # 比較演算子
    # ------------------------------------------------------------------------

    def __lt__(self, other):
        other = self._to_siarray(other)
        diff = self - other
        neg_num = np.array(
            [-c.flatten()[0] if c.size > 0 else 0 for c in diff.coef_num]
        )
        den = np.array([c.flatten()[0] if c.size > 0 else 1 for c in diff.coef_den])
        z = IntervalTracker.get_instance().get_z()
        new_intervals = calc_polynominal_interval(neg_num, den, z, threshold=0.0)
        IntervalTracker.get_instance().update(new_intervals)

        result_value = diff.data < 0

        # スカラーの場合はPythonのboolを返す
        if result_value.size == 1:
            return bool(result_value.item())
        # 配列の場合はnumpy配列を返す
        return result_value

    def __le__(self, other):
        other = self._to_siarray(other)
        diff = self - other
        neg_num = np.array(
            [-c.flatten()[0] if c.size > 0 else 0 for c in diff.coef_num]
        )
        den = np.array([c.flatten()[0] if c.size > 0 else 1 for c in diff.coef_den])
        z = IntervalTracker.get_instance().get_z()
        new_intervals = calc_polynominal_interval(neg_num, den, z, threshold=0.0)
        IntervalTracker.get_instance().update(new_intervals)

        result_value = diff.data <= 0

        if result_value.size == 1:
            return bool(result_value.item())
        return result_value

    def __gt__(self, other):
        other = self._to_siarray(other)
        diff = self - other
        num = np.array([c.flatten()[0] if c.size > 0 else 0 for c in diff.coef_num])
        den = np.array([c.flatten()[0] if c.size > 0 else 1 for c in diff.coef_den])
        z = IntervalTracker.get_instance().get_z()
        new_intervals = calc_polynominal_interval(num, den, z, threshold=0.0)
        IntervalTracker.get_instance().update(new_intervals)

        result_value = diff.data > 0

        if result_value.size == 1:
            return bool(result_value.item())
        return result_value

    def __ge__(self, other):
        other = self._to_siarray(other)
        diff = self - other
        num = np.array([c.flatten()[0] if c.size > 0 else 0 for c in diff.coef_num])
        den = np.array([c.flatten()[0] if c.size > 0 else 1 for c in diff.coef_den])
        z = IntervalTracker.get_instance().get_z()
        new_intervals = calc_polynominal_interval(num, den, z, threshold=0.0)
        IntervalTracker.get_instance().update(new_intervals)

        result_value = diff.data >= 0

        if result_value.size == 1:
            return bool(result_value.item())
        return result_value

    def __eq__(self, other):
        other = self._to_siarray(other)
        result_data = np.equal(self.data, other.data)

        if result_data.size == 1:
            return bool(result_data.item())
        return result_data

    def __ne__(self, other):
        other = self._to_siarray(other)
        result_data = np.not_equal(self.data, other.data)

        if result_data.size == 1:
            return bool(result_data.item())
        return result_data

    # ------------------------------------------------------------------------
    # インデックス演算
    # ------------------------------------------------------------------------

    def __getitem__(self, key):
        """要素の取得"""
        if isinstance(key, siarray):
            key = key.coef_num[0]

        return siarray.from_rational(
            [c[key].copy() for c in self.coef_num],
            [c[key].copy() for c in self.coef_den],
        )

    def __setitem__(self, key, value):
        """要素の設定（完全置き換え版）"""
        if isinstance(key, siarray):
            key = key.coef_num[0]

        value = self._to_siarray(value)

        # 分子・分母の次数を合わせる
        max_num_degree = builtins.max(len(self.coef_num), len(value.coef_num))
        max_den_degree = builtins.max(len(self.coef_den), len(value.coef_den))

        # 分子の次数を拡張
        while len(self.coef_num) < max_num_degree:
            self.coef_num.append(np.zeros_like(self.coef_num[0]))

        # 分母の次数を拡張
        while len(self.coef_den) < max_den_degree:
            self.coef_den.append(np.ones_like(self.coef_den[0]))

        # 分子を完全に置き換え
        for i in range(len(value.coef_num)):
            if i < len(self.coef_num):
                self.coef_num[i][key] = value.coef_num[i].copy()
            else:
                self.coef_num.append(np.zeros_like(self.coef_num[0]))
                self.coef_num[i][key] = value.coef_num[i].copy()

        # value よりも高次の分子項がある場合はゼロにする
        for i in range(len(value.coef_num), len(self.coef_num)):
            self.coef_num[i][key] = 0.0

        # 分母を完全に置き換え
        for i in range(len(value.coef_den)):
            if i < len(self.coef_den):
                self.coef_den[i][key] = value.coef_den[i].copy()
            else:
                self.coef_den.append(np.ones_like(self.coef_den[0]))
                self.coef_den[i][key] = value.coef_den[i].copy()

        # value よりも高次の分母項がある場合は1にする
        for i in range(len(value.coef_den), len(self.coef_den)):
            self.coef_den[i][key] = 1.0

    # ------------------------------------------------------------------------
    # 配列操作メソッド
    # ------------------------------------------------------------------------

    @property
    def T(self):
        new_coef_num = [c.T for c in self.coef_num]
        new_coef_den = [c.T for c in self.coef_den]
        return siarray.from_rational(new_coef_num, new_coef_den)

    def transpose(self, axes=None):
        new_coef_num = [np.transpose(c, axes) for c in self.coef_num]
        new_coef_den = [np.transpose(c, axes) for c in self.coef_den]
        return siarray.from_rational(new_coef_num, new_coef_den)

    def inv(self):
        new_coef_num = [np.linalg.inv(c) for c in self.coef_num]
        return siarray.from_rational(new_coef_num, [c.copy() for c in self.coef_den])

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        new_coef_num = [c.reshape(shape) for c in self.coef_num]
        new_coef_den = [c.reshape(shape) for c in self.coef_den]
        result = siarray.from_rational(new_coef_num, new_coef_den)
        result._shape = shape
        return result

    def argsort(self, axis=-1, kind=None, order=None):
        output = self.data
        sorted_idx = np.argsort(output, axis=axis, kind=kind, order=order)
        tracker = IntervalTracker.get_instance()
        for i in range(len(sorted_idx) - 1):
            idx = sorted_idx[i]
            next_idx = sorted_idx[i + 1]
            inequality = self[idx] - self[next_idx]
            _intervals = calc_polynominal_interval(
                np.array(inequality.coef_num),
                np.array(inequality.coef_den),
                z=tracker.get_z(),
            )
            tracker.update(_intervals)
        coef_num = [np.array(sorted_idx)]
        coef_den = [np.ones_like(sorted_idx)]
        return siarray.from_rational(coef_num, coef_den)

    def max(x, axis=None, keepdims=False):
        """最大値（siarray対応）"""
        if not isinstance(x, siarray):
            x = siarray(x)

        tracker = IntervalTracker.get_instance()
        z = tracker.get_z()

        if axis is None:
            # 全要素から最大値を探す
            flat_data = x.data.flatten()
            max_idx = np.argmax(flat_data)

            # 最大値が他の全要素より大きいことを制約に追加
            for i in range(len(flat_data)):
                if i != max_idx:
                    flat_idx_max = np.unravel_index(max_idx, x._shape)
                    flat_idx_i = np.unravel_index(i, x._shape)
                    diff = x[flat_idx_max] - x[flat_idx_i]
                    num = np.array(
                        [c.flatten()[0] if c.size > 0 else 0 for c in diff.coef_num]
                    )
                    den = np.array(
                        [c.flatten()[0] if c.size > 0 else 1 for c in diff.coef_den]
                    )
                    intervals = calc_polynominal_interval(num, den, z)
                    tracker.update(intervals)

            flat_idx = np.unravel_index(max_idx, x._shape)
            return x[flat_idx]
        else:
            # 特定の軸に沿った最大値
            if axis < 0:
                axis = len(x._shape) + axis

            max_indices = np.argmax(x.data, axis=axis)

            if keepdims:
                result_shape = list(x._shape)
                result_shape[axis] = 1
            else:
                result_shape = tuple(s for i, s in enumerate(x._shape) if i != axis)

            result_coef_num = [
                np.zeros(result_shape if result_shape else (1,)) for _ in x.coef_num
            ]
            result_coef_den = [
                np.zeros(result_shape if result_shape else (1,)) for _ in x.coef_den
            ]

            for idx in np.ndindex(max_indices.shape):
                max_pos = max_indices[idx]
                src_idx = list(idx)
                src_idx.insert(axis, max_pos)
                src_idx = tuple(src_idx)

                # 制約を追加：最大値が同じ軸の他の要素より大きい
                for i in range(x._shape[axis]):
                    if i != max_pos:
                        other_idx = list(idx)
                        other_idx.insert(axis, i)
                        other_idx = tuple(other_idx)
                        diff = x[src_idx] - x[other_idx]
                        num = np.array(
                            [c.flatten()[0] if c.size > 0 else 0 for c in diff.coef_num]
                        )
                        den = np.array(
                            [c.flatten()[0] if c.size > 0 else 1 for c in diff.coef_den]
                        )
                        intervals = calc_polynominal_interval(num, den, z)
                        tracker.update(intervals)

                dst_idx = (
                    idx
                    if not keepdims
                    else tuple(list(idx[:axis]) + [0] + list(idx[axis:]))
                )
                for deg in range(len(x.coef_num)):
                    result_coef_num[deg][dst_idx] = x.coef_num[deg][src_idx]
                for deg in range(len(x.coef_den)):
                    result_coef_den[deg][dst_idx] = x.coef_den[deg][src_idx]

            return siarray.from_rational(result_coef_num, result_coef_den)

    def min(x, axis=None, keepdims=False):
        """最小値（siarray対応）"""
        if not isinstance(x, siarray):
            x = siarray(x)

        tracker = IntervalTracker.get_instance()
        z = tracker.get_z()

        if axis is None:
            # 全要素から最小値を探す
            flat_data = x.data.flatten()
            min_idx = np.argmin(flat_data)

            # 最小値が他の全要素より小さいことを制約に追加
            for i in range(len(flat_data)):
                if i != min_idx:
                    flat_idx_min = np.unravel_index(min_idx, x._shape)
                    flat_idx_i = np.unravel_index(i, x._shape)
                    diff = x[flat_idx_i] - x[flat_idx_min]  # other - min >= 0
                    num = np.array(
                        [c.flatten()[0] if c.size > 0 else 0 for c in diff.coef_num]
                    )
                    den = np.array(
                        [c.flatten()[0] if c.size > 0 else 1 for c in diff.coef_den]
                    )
                    intervals = calc_polynominal_interval(num, den, z, threshold=0.0)
                    tracker.update(intervals)

            flat_idx = np.unravel_index(min_idx, x._shape)
            return x[flat_idx]
        else:
            # 特定の軸に沿った最小値
            if axis < 0:
                axis = len(x._shape) + axis

            min_indices = np.argmin(x.data, axis=axis)

            if keepdims:
                result_shape = list(x._shape)
                result_shape[axis] = 1
            else:
                result_shape = tuple(s for i, s in enumerate(x._shape) if i != axis)

            result_coef_num = [
                np.zeros(result_shape if result_shape else (1,)) for _ in x.coef_num
            ]
            result_coef_den = [
                np.zeros(result_shape if result_shape else (1,)) for _ in x.coef_den
            ]

            for idx in np.ndindex(min_indices.shape):
                min_pos = min_indices[idx]
                src_idx = list(idx)
                src_idx.insert(axis, min_pos)
                src_idx = tuple(src_idx)

                # 制約を追加：最小値が同じ軸の他の要素より小さい
                for i in range(x._shape[axis]):
                    if i != min_pos:
                        other_idx = list(idx)
                        other_idx.insert(axis, i)
                        other_idx = tuple(other_idx)
                        diff = x[other_idx] - x[src_idx]  # other - min >= 0
                        num = np.array(
                            [c.flatten()[0] if c.size > 0 else 0 for c in diff.coef_num]
                        )
                        den = np.array(
                            [c.flatten()[0] if c.size > 0 else 1 for c in diff.coef_den]
                        )
                        intervals = calc_polynominal_interval(
                            num, den, z, threshold=0.0
                        )
                        tracker.update(intervals)

                dst_idx = (
                    idx
                    if not keepdims
                    else tuple(list(idx[:axis]) + [0] + list(idx[axis:]))
                )
                for deg in range(len(x.coef_num)):
                    result_coef_num[deg][dst_idx] = x.coef_num[deg][src_idx]
                for deg in range(len(x.coef_den)):
                    result_coef_den[deg][dst_idx] = x.coef_den[deg][src_idx]

            return siarray.from_rational(result_coef_num, result_coef_den)

    def copy(self):
        return siarray.from_rational(
            [c.copy() for c in self.coef_num], [c.copy() for c in self.coef_den]
        )

    # ------------------------------------------------------------------------
    # 集約メソッド
    # ------------------------------------------------------------------------

    def sum(self, axis=None, keepdims=False):
        """配列要素の合計 (有理関数版)"""
        if axis is None:
            # 全要素の合計
            acc_num = [np.zeros(())]
            acc_den = [np.ones(())]

            # 配列を平坦化して全要素を加算
            flat_indices = np.ndindex(self._shape)

            for idx in flat_indices:
                # 各要素の係数を取得
                elem_num = [c[idx] if c.size > 1 else c.item() for c in self.coef_num]
                elem_den = [c[idx] if c.size > 1 else c.item() for c in self.coef_den]

                # 累積に加算: acc + elem = (acc_num * elem_den + elem_num * acc_den) / (acc_den * elem_den)
                new_num = self._poly_add(
                    self._poly_mul(acc_num, elem_den), self._poly_mul(elem_num, acc_den)
                )
                new_den = self._poly_mul(acc_den, elem_den)

                acc_num = new_num
                acc_den = new_den

            return siarray.from_rational(acc_num, acc_den)

        else:
            # 特定の軸に沿った合計
            if axis < 0:
                axis = len(self._shape) + axis

            if axis < 0 or axis >= len(self._shape):
                raise ValueError(
                    f"axis {axis} is out of bounds for array of dimension {len(self._shape)}"
                )

            # 結果の形状を計算
            if keepdims:
                result_shape = list(self._shape)
                result_shape[axis] = 1
                result_shape = tuple(result_shape)
            else:
                result_shape = tuple(s for i, s in enumerate(self._shape) if i != axis)

            if result_shape == ():
                result_shape_for_init = (1,)
            else:
                result_shape_for_init = result_shape

            # 結果の係数リストを初期化
            # 最悪ケースの次数を見積もる
            axis_size = self._shape[axis]
            max_num_degree = (len(self.coef_num) - 1) * axis_size
            max_den_degree = (len(self.coef_den) - 1) * axis_size

            result_coef_num = [
                np.zeros(result_shape_for_init) for _ in range(max_num_degree + 1)
            ]
            result_coef_den = [
                np.zeros(result_shape_for_init) for _ in range(max_den_degree + 1)
            ]

            # 結果の各位置について計算
            result_indices = np.ndindex(result_shape_for_init)

            for result_idx in result_indices:
                # 累積用の有理関数 (初期値: 0/1)
                acc_num = [np.zeros(())]
                acc_den = [np.ones(())]

                # 指定された軸に沿って加算
                for i in range(axis_size):
                    # 元の配列のインデックスを構築
                    if keepdims:
                        src_idx = list(result_idx)
                        src_idx[axis] = i
                        src_idx = tuple(src_idx)
                    else:
                        src_idx = list(result_idx)
                        src_idx.insert(axis, i)
                        src_idx = tuple(src_idx)

                    # 要素の係数を取得
                    elem_num = [
                        c[src_idx] if c.size > 1 else c.item() for c in self.coef_num
                    ]
                    elem_den = [
                        c[src_idx] if c.size > 1 else c.item() for c in self.coef_den
                    ]

                    # 累積に加算
                    new_num = self._poly_add(
                        self._poly_mul(acc_num, elem_den),
                        self._poly_mul(elem_num, acc_den),
                    )
                    new_den = self._poly_mul(acc_den, elem_den)

                    acc_num = new_num
                    acc_den = new_den

                # 結果に格納
                for deg, coef in enumerate(acc_num):
                    result_coef_num[deg][result_idx] = coef

                for deg, coef in enumerate(acc_den):
                    result_coef_den[deg][result_idx] = coef

            # 末尾のゼロ係数を削除
            while len(result_coef_num) > 1 and np.allclose(result_coef_num[-1], 0):
                result_coef_num.pop()
            while len(result_coef_den) > 1 and np.allclose(result_coef_den[-1], 0):
                result_coef_den.pop()

            # 形状を調整
            if result_shape == ():
                result_coef_num = [c.item() * np.ones(()) for c in result_coef_num]
                result_coef_den = [c.item() * np.ones(()) for c in result_coef_den]
            elif not keepdims:
                result_coef_num = [c.reshape(result_shape) for c in result_coef_num]
                result_coef_den = [c.reshape(result_shape) for c in result_coef_den]

            return siarray.from_rational(result_coef_num, result_coef_den)

    def mean(self, axis=None, keepdims=False):
        """配列要素の平均 (有理関数版)"""
        if axis is None:
            # 全要素の平均
            total = self.sum(axis=None, keepdims=False)
            count = np.prod(self._shape)
            return total / count

        else:
            # 特定の軸に沿った平均
            if axis < 0:
                axis = len(self._shape) + axis

            if axis < 0 or axis >= len(self._shape):
                raise ValueError(
                    f"axis {axis} is out of bounds for array of dimension {len(self._shape)}"
                )

            # 軸に沿った合計を計算
            total = self.sum(axis=axis, keepdims=keepdims)

            # その軸のサイズで割る
            count = self._shape[axis]

            return total / count

    # ------------------------------------------------------------------------
    # プロパティ
    # ------------------------------------------------------------------------

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def size(self):
        return np.prod(self._shape)

    @property
    def dtype(self):
        return self.coef_num[0].dtype

    # ------------------------------------------------------------------------
    # 推論関連メソッド
    # ------------------------------------------------------------------------

    def add_prob_vec_nodes(self, prob_vec: "siarray"):
        """確率変数ベクトルを保存"""
        self.prob_vec = prob_vec

    def add_observed_model(self, model: "siarray"):
        """観測されたモデルを保存"""
        self.observed_model = model

    def add_algorithm(self, algorithm: callable):
        """選択アルゴリズム関数を保存"""
        self.algorithm_func = algorithm

    def forward_si(self, a: np.ndarray, b: np.ndarray, z: float):
        """
        アルゴリズムをRationalFunctionで実行し、制約区間を計算
        """
        tracker = IntervalTracker.get_instance()
        tracker.reset()
        tracker.set_z(z)

        # 通常の値で実行してモデルを取得
        # prob_vec = a + b * z
        # model = self.algorithm_func(siarray(prob_vec))

        prob_vec_rf = siarray.from_rational(
            numerator=[a, b],
            denominator=[np.ones_like(a)],
        )
        model = self.algorithm_func(prob_vec_rf)

        # 最終的なintervalsを取得
        intervals = tracker.get()
        # モデルの出力値を取得
        if isinstance(model, siarray):
            output = model.data
        else:
            output = model

        return output, intervals

    def model_selector(self, searched_model: np.ndarray | "siarray") -> bool:
        """2つのモデルが同じかどうかを判定"""
        data1 = (
            self.observed_model.data
            if isinstance(self.observed_model, siarray)
            else np.asarray(self.observed_model)
        )
        data2 = (
            searched_model.data
            if isinstance(searched_model, siarray)
            else np.asarray(searched_model)
        )
        comparison = np.allclose(data1, data2)
        return comparison

    def to_numpy(self):
        """NumPy配列を返す"""
        return self.data.copy()

    def values(self):
        """データの値を返す"""
        return self.data

    def __repr__(self):
        """repr表示 - 詳細情報を含む"""
        data_repr = repr(self.data)
        return f"siarray({data_repr})"

    def __str__(self):
        """str表示 - print()で使用される"""
        data_str = str(self.data)
        return f"siarray({data_str})"


# ============================================================================
# ユーティリティ関数
# ============================================================================


def calc_polynominal_interval(
    numerator: np.ndarray, denominator: np.ndarray, z: float, threshold: float = 0.0
) -> RealSubset:
    """有理関数 > threshold となるzの範囲を計算"""

    # denominator > 0となる場合，numerator - denominator * threshold > 0を解く

    if len(numerator) < len(denominator):
        numerator = np.concatenate(
            [numerator, np.zeros(len(denominator) - len(numerator))]
        )
    elif len(denominator) < len(numerator):
        denominator = np.concatenate(
            [denominator, np.zeros(len(numerator) - len(denominator))]
        )

    # 多項式の次数を確認
    # print(f"Numerator degree: {len(numerator) - 1}, Denominator degree: {len(denominator) - 1}")

    positive_conditioned_invervals = RealSubset(
        polynomial_iso_sign_interval(denominator, z)
    )
    positive_intervals = RealSubset(
        polynomial_iso_sign_interval(numerator - denominator * threshold, z)
    )
    intervals = positive_conditioned_invervals & positive_intervals

    # denominator < 0 となる場合，numerator - denominator * threshold < 0を解く
    negative_conditioned_invervals = RealSubset(
        polynomial_iso_sign_interval(-denominator, z)
    )
    negative_intervals = RealSubset(
        polynomial_iso_sign_interval(denominator * threshold - numerator, z)
    )
    intervals = intervals | (negative_conditioned_invervals & negative_intervals)
    return RealSubset(intervals)


def mean(x, axis=None, keepdims=False):
    """平均（siarray対応）"""
    if isinstance(x, siarray):
        return x.mean(axis=axis, keepdims=keepdims)
    else:
        return siarray(x).mean(axis=axis, keepdims=keepdims)


def sum(x, axis=None, keepdims=False):
    """合計（siarray対応）"""
    if isinstance(x, siarray):
        return x.sum(axis=axis, keepdims=keepdims)
    else:
        return siarray(x).sum(axis=axis, keepdims=keepdims)


def abs(x):
    """絶対値（siarray対応）"""
    if isinstance(x, siarray):
        return x.__abs__()
    else:
        return siarray(x).__abs__()


def argsort(x, axis=-1, kind=None, order=None):
    """ソートのインデックス（siarray対応）"""
    if isinstance(x, siarray):
        return x.argsort(axis=axis, kind=kind, order=order)
    else:
        return siarray(x).argsort(axis=axis, kind=kind, order=order)


def max(x, axis=None, keepdims=False):
    """最大値（siarray対応）"""
    if isinstance(x, siarray):
        return siarray.max(x, axis=axis, keepdims=keepdims)
    else:
        return siarray(x).max(axis=axis, keepdims=keepdims)


def min(x, axis=None, keepdims=False):
    """最小値（siarray対応）"""
    if isinstance(x, siarray):
        return siarray.min(x, axis=axis, keepdims=keepdims)
    else:
        return siarray(x).min(axis=axis, keepdims=keepdims)


def stack(arrays: list["siarray"], axis=0) -> "siarray":
    """siarrayのスタック"""
    if not arrays:
        raise ValueError("arrays must not be empty")

    # 各次数ごとにスタック
    max_num_degree = builtins.max(len(arr.coef_num) for arr in arrays)
    max_den_degree = builtins.max(len(arr.coef_den) for arr in arrays)

    stacked_num = []
    for deg in range(max_num_degree):
        coef_list = []
        for arr in arrays:
            if deg < len(arr.coef_num):
                coef_list.append(arr.coef_num[deg])
            else:
                coef_list.append(np.zeros_like(arr.coef_num[0]))
        stacked_num.append(np.stack(coef_list, axis=axis))

    stacked_den = []
    for deg in range(max_den_degree):
        coef_list = []
        for arr in arrays:
            if deg < len(arr.coef_den):
                coef_list.append(arr.coef_den[deg])
            else:
                coef_list.append(np.ones_like(arr.coef_den[0]))
        stacked_den.append(np.stack(coef_list, axis=axis))

    return siarray.from_rational(stacked_num, stacked_den)


def array(X):
    return siarray(X)
