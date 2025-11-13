from __future__ import annotations

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

        # リスト内包表記で各項を計算してから合計
        coef_num_terms = [self.coef_num[i] * (z**i) for i in range(len(self.coef_num))]
        coef_den_terms = [self.coef_den[i] * (z**i) for i in range(len(self.coef_den))]

        # NumPy の sum を使用
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
        n = max(len(p1), len(p2))
        result = []
        for i in range(n):
            c1 = p1[i].copy() if i < len(p1) else np.zeros_like(p1[0])
            c2 = p2[i].copy() if i < len(p2) else np.zeros_like(p2[0])
            result.append(c1 + c2)
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
        new_coef_num = self._poly_add(
            self._poly_mul(self.coef_num, other.coef_den),
            [-c.copy() for c in self._poly_mul(other.coef_num, self.coef_den)],
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
        # オーバーフローを防ぐため，定数による割り算は直接計算
        if len(other.coef_num) == 1 and len(other.coef_den) == 1:
            new_coef_num = [c / other.coef_num[0] for c in self.coef_num]
            new_coef_den = [c / other.coef_den[0] for c in self.coef_den]
        else:
            new_coef_num = self._poly_mul(self.coef_num, other.coef_den)
            new_coef_den = self._poly_mul(self.coef_den, other.coef_num)
        return siarray.from_rational(new_coef_num, new_coef_den)

    def __rtruediv__(self, other):
        return self._to_siarray(other).__truediv__(self)

    def __matmul__(self, other):
        other = self._to_siarray(other)
        new_coef_num = []
        for i in range(len(self.coef_num)):
            for j in range(len(other.coef_num)):
                new_coef_num.append(self.coef_num[i] @ other.coef_num[j])

        new_coef_den = [np.ones_like(new_coef_num[i]) for i in range(len(new_coef_num))]
        return siarray.from_rational(new_coef_num, new_coef_den)

    # def __matmul__(self, other):
    #     """行列積 (有理関数版)"""
    #     other = self._to_siarray(other)

    #     # 形状チェック
    #     if len(self._shape) < 1 or len(other._shape) < 1:
    #         raise ValueError("matmul requires at least 1-dimensional arrays")

    #     # ベクトル・行列の次元を取得
    #     if len(self._shape) == 1:
    #         m, k = 1, self._shape[0]
    #         self_is_1d = True
    #     else:
    #         m, k = self._shape[-2:]
    #         self_is_1d = False

    #     if len(other._shape) == 1:
    #         k2, n = other._shape[0], 1
    #         other_is_1d = True
    #     else:
    #         k2, n = other._shape[-2:]
    #         other_is_1d = False

    #     if k != k2:
    #         raise ValueError(f"matmul: Input operand shapes {self._shape} and {other._shape} are not compatible")

    #     # 結果の形状を決定
    #     if self_is_1d and other_is_1d:
    #         result_shape = ()
    #         m, n = 1, 1
    #     elif self_is_1d:
    #         result_shape = (n,)
    #         m = 1
    #     elif other_is_1d:
    #         result_shape = (m,)
    #         n = 1
    #     else:
    #         result_shape = (m, n)

    #     # 行列積を計算: C[i,j] = Σ_l A[i,l] * B[l,j]
    #     results = []

    #     for i in range(m):
    #         for j in range(n):
    #             # 累積用の有理関数 (初期値: 0)
    #             acc = siarray.constant(0.0)

    #             for l in range(k):
    #                 # A[i,l]とB[l,j]を取得
    #                 if self_is_1d:
    #                     a_num = [np.array(c.flat[l]) for c in self.coef_num]
    #                     a_den = [np.array(c.flat[l]) for c in self.coef_den]
    #                 else:
    #                     if len(self._shape) == 2:
    #                         a_num = [c[i, l] for c in self.coef_num]
    #                         a_den = [c[i, l] for c in self.coef_den]
    #                     else:
    #                         idx_a = i * k + l
    #                         a_num = [np.array(c.flat[idx_a]) for c in self.coef_num]
    #                         a_den = [np.array(c.flat[idx_a]) for c in self.coef_den]

    #                 if other_is_1d:
    #                     b_num = [np.array(c.flat[l]) for c in other.coef_num]
    #                     b_den = [np.array(c.flat[l]) for c in other.coef_den]
    #                 else:
    #                     if len(other._shape) == 2:
    #                         b_num = [c[l, j] for c in other.coef_num]
    #                         b_den = [c[l, j] for c in other.coef_den]
    #                     else:
    #                         idx_b = l * n + j
    #                         b_num = [np.array(c.flat[idx_b]) for c in other.coef_num]
    #                         b_den = [np.array(c.flat[idx_b]) for c in other.coef_den]

    #                 # siarrayオブジェクトとして作成
    #                 a_elem = siarray.from_rational(a_num, a_den)
    #                 b_elem = siarray.from_rational(b_num, b_den)

    #                 # A[i,l] * B[l,j]を計算して累積
    #                 prod = a_elem * b_elem
    #                 acc = acc + prod

    #             results.append(acc)

    #     # 結果を結合
    #     if result_shape == ():
    #         return results[0]

    #     # 最大次数を取得
    #     max_num_deg = max(len(r.coef_num) for r in results)
    #     max_den_deg = max(len(r.coef_den) for r in results)

    #     # 係数を結合
    #     coef_num_combined = []
    #     coef_den_combined = []

    #     for deg in range(max_num_deg):
    #         coeffs = [r.coef_num[deg] if deg < len(r.coef_num) else np.array(0.0)
    #                 for r in results]
    #         coef_num_combined.append(np.array(coeffs).reshape(result_shape))

    #     for deg in range(max_den_deg):
    #         coeffs = [r.coef_den[deg] if deg < len(r.coef_den) else np.array(0.0)
    #                 for r in results]
    #         coef_den_combined.append(np.array(coeffs).reshape(result_shape))

    #     return siarray.from_rational(coef_num_combined, coef_den_combined)

    def __rmatmul__(self, other):
        """右側からの行列積"""
        return self._to_siarray(other).__matmul__(self)

    def __neg__(self):
        new_coef_num = [-c.copy() for c in self.coef_num]
        return siarray.from_rational(new_coef_num, [c.copy() for c in self.coef_den])

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

    def __pow__(self, power):
        """累乗演算"""
        if isinstance(power, siarray):
            power_val = power.data
        else:
            power_val = power
        result_data = self.data**power_val
        return siarray.constant(result_data)

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
        """要素の設定"""
        if isinstance(key, siarray):
            key = key.coef_num[0]

        value = self._to_siarray(value)

        for i in range(len(value.coef_num)):
            if i < len(self.coef_num):
                self.coef_num[i][key] = value.coef_num[i].copy()
        for i in range(len(value.coef_den)):
            if i < len(self.coef_den):
                self.coef_den[i][key] = value.coef_den[i].copy()

    # ------------------------------------------------------------------------
    # 配列操作メソッド
    # ------------------------------------------------------------------------

    @property
    def T(self):
        """転置"""
        new_coef_num = [c.T for c in self.coef_num]
        new_coef_den = [c.T for c in self.coef_den]
        return siarray.from_rational(new_coef_num, new_coef_den)

    def transpose(self, *axes):
        """転置（軸指定可能）"""
        if axes:
            new_coef_num = [np.transpose(c, axes) for c in self.coef_num]
            new_coef_den = [np.transpose(c, axes) for c in self.coef_den]
        else:
            new_coef_num = [c.T for c in self.coef_num]
            new_coef_den = [c.T for c in self.coef_den]
        return siarray.from_rational(new_coef_num, new_coef_den)

    def inv(self):
        """逆行列"""
        new_coef_num = [np.linalg.inv(c) for c in self.coef_num]
        return siarray.from_rational(new_coef_num, [c.copy() for c in self.coef_den])

    def reshape(self, *shape):
        """形状変更"""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]

        new_coef_num = [c.reshape(shape) for c in self.coef_num]
        new_coef_den = [c.reshape(shape) for c in self.coef_den]
        result = siarray.from_rational(new_coef_num, new_coef_den)
        result._shape = shape
        return result

    def argsort(self, axis=-1, kind=None, order=None):
        """ソートのインデックスを返す"""
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

    def copy(self):
        """siarrayオブジェクトのコピーを作成"""
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
        # print("debug: foward")
        # model = self.algorithm_func(siarray(prob_vec))

        prob_vec_rf = siarray.from_rational(
            numerator=[a, b],
            denominator=[np.ones_like(a)],
        )

        print("debug: forward_si")
        model = self.algorithm_func(prob_vec_rf)

        # 最終的なintervalsを取得
        intervals = tracker.get()
        # モデルの出力値を取得
        if isinstance(model, siarray):
            output = model.data
        else:
            output = model

        print("debug", intervals)
        return output, intervals

    # ------------------------------------------------------------------------
    # ユーティリティメソッド
    # ------------------------------------------------------------------------

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
        print("debug", comparison)
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
    # print("debug", numerator)
    """有理関数 > threshold となるzの範囲を計算"""
    if numerator.ndim < denominator.ndim:
        numerator = np.concatenate(
            [numerator, np.zeros(denominator.ndim - numerator.ndim)]
        )
    elif denominator.ndim < numerator.ndim:
        denominator = np.concatenate(
            [denominator, np.zeros(numerator.ndim - denominator.ndim)]
        )
    intervals = polynomial_iso_sign_interval(numerator - denominator * threshold, z)
    # print("debug:", intervals)
    return RealSubset(intervals)


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


def array(X):
    return siarray(X)
