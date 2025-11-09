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


class RationalFunction:
    def __init__(
        self,
        numerator: list[np.ndarray],
        denominator: list[np.ndarray] | None = None,
    ):
        self.coef_num = list(numerator)
        self.coef_den = (
            list(denominator)
            if denominator is not None
            else [np.ones_like(self.coef_num[0])]
        )
        self.shape = numerator[0].shape

    def _to_rational_function(self, other):
        from .utils import siarray

        if isinstance(other, RationalFunction):
            return other
        elif isinstance(other, siarray):
            return RationalFunction(
                numerator=[other.data],
                denominator=[np.ones_like(other.data)],
            )
        else:
            return RationalFunction(
                numerator=[np.array(other)],
                denominator=[np.ones_like(np.array(other))],
            )

    @staticmethod
    def _poly_add(p1: list[np.ndarray], p2: list[np.ndarray]) -> list[np.ndarray]:
        n = max(len(p1), len(p2))
        result = []
        for i in range(n):
            c1 = p1[i] if i < len(p1) else np.zeros_like(p1[0])
            c2 = p2[i] if i < len(p2) else np.zeros_like(p2[0])
            result.append(c1 + c2)
        return result

    @staticmethod
    def _poly_mul(p1: list[np.ndarray], p2: list[np.ndarray]) -> list[np.ndarray]:
        n = len(p1) + len(p2) - 1
        result = [np.zeros_like(p1[0])] * n
        for i, c1 in enumerate(p1):
            for j, c2 in enumerate(p2):
                result[i + j] = result[i + j] + c1 * c2
        return result

    def __add__(self, other: "RationalFunction") -> "RationalFunction":
        other = self._to_rational_function(other)
        num1 = self._poly_mul(self.coef_num, other.coef_den)
        num2 = self._poly_mul(other.coef_num, self.coef_den)
        new_coef_num = self._poly_add(num1, num2)
        new_coef_den = self._poly_mul(self.coef_den, other.coef_den)
        return RationalFunction(new_coef_num, new_coef_den)

    def __sub__(self, other: "RationalFunction") -> "RationalFunction":
        other = self._to_rational_function(other)
        num1 = self._poly_mul(self.coef_num, other.coef_den)
        num2 = self._poly_mul(other.coef_num, self.coef_den)
        # _poly_addを使って次元の違いに対応
        neg_num2 = [-c for c in num2]
        new_coef_num = self._poly_add(num1, neg_num2)
        new_coef_den = self._poly_mul(self.coef_den, other.coef_den)
        return RationalFunction(new_coef_num, new_coef_den)

    def __truediv__(self, other: "RationalFunction") -> "RationalFunction":
        other = self._to_rational_function(other)
        new_coef_num = self._poly_mul(self.coef_num, other.coef_den)
        new_coef_den = self._poly_mul(self.coef_den, other.coef_num)
        return RationalFunction(new_coef_num, new_coef_den)

    def __mul__(self, other: "RationalFunction") -> "RationalFunction":
        other = self._to_rational_function(other)
        new_coef_num = self._poly_mul(self.coef_num, other.coef_num)
        new_coef_den = self._poly_mul(self.coef_den, other.coef_den)
        return RationalFunction(new_coef_num, new_coef_den)

    def __matmul__(self, other: "RationalFunction") -> "RationalFunction":
        other = self._to_rational_function(other)

        # selfとotherの形状を正規化（1次元を2次元として扱う）
        if len(self.shape) == 1:
            self_rows = 1
            self_cols = self.shape[0]
            self_is_vector = True
        else:
            self_rows = self.shape[0]
            self_cols = self.shape[1]
            self_is_vector = False

        if len(other.shape) == 1:
            other_cols = 1
            other_is_vector = True
        else:
            other_cols = other.shape[1]
            other_is_vector = False

        # 結果の行列を初期化
        output = RationalFunction(
            numerator=[np.zeros((self_rows, other_cols))],
            denominator=[np.ones((self_rows, other_cols))],
        )

        for i in range(self_rows):
            for j in range(other_cols):
                sum_ = RationalFunction(
                    numerator=[np.zeros((1,))],
                    denominator=[np.ones((1,))],
                )
                for k in range(self_cols):
                    # selfの要素取得
                    if self_is_vector:
                        self_elem = self[k]
                    else:
                        self_elem = self[i, k]

                    # otherの要素取得
                    if other_is_vector:
                        other_elem = other[k]
                    else:
                        other_elem = other[k, j]

                    sum_ = sum_ + (self_elem * other_elem)
                output[i, j] = sum_

        # 結果の形状を調整
        if self_is_vector and other_is_vector:
            # ベクトル @ ベクトル → スカラー
            output.shape = (1,)
            output.coef_num = [c.flatten() for c in output.coef_num]
            output.coef_den = [c.flatten() for c in output.coef_den]
        elif other_is_vector:
            # 行列 @ ベクトル → ベクトル
            output.shape = (self_rows,)
            output.coef_num = [c.flatten() for c in output.coef_num]
            output.coef_den = [c.flatten() for c in output.coef_den]
        elif self_is_vector:
            # ベクトル @ 行列 → ベクトル
            output.shape = (other_cols,)
            output.coef_num = [c.flatten() for c in output.coef_num]
            output.coef_den = [c.flatten() for c in output.coef_den]

        return output

    def __getitem__(self, key) -> "RationalFunction":
        return RationalFunction(
            [c[key] for c in self.coef_num],
            [c[key] for c in self.coef_den],
        )

    def __setitem__(self, key, value):
        """要素の設定（破壊的操作）"""
        if isinstance(value, RationalFunction):
            value_rf = value
        else:
            from .utils import siarray

            value_siarray = (
                value if isinstance(value, siarray) else siarray.constant(value)
            )
            value_rf = RationalFunction(
                numerator=[value_siarray.data],
                denominator=[np.ones_like(value_siarray.data)],
                z=None,
            )
        for i in range(len(self.coef_num)):
            self.coef_num[i][key] = value_rf.coef_num[i]
        for i in range(len(self.coef_den)):
            self.coef_den[i][key] = value_rf.coef_den[i]

    def __lt__(self, other: "RationalFunction") -> "RationalFunction":
        """self < other を計算し、制約をグローバルintervalsに追加"""
        other = self._to_rational_function(other)
        diff = self - other
        # self - other < 0 の制約を追加
        neg_num = np.array(
            [-c.flatten()[0] if c.size > 0 else 0 for c in diff.coef_num]
        )
        den = np.array([c.flatten()[0] if c.size > 0 else 1 for c in diff.coef_den])
        z = IntervalTracker.get_instance().get_z()
        new_intervals = calc_polynominal_interval(neg_num, den, z, threshold=0.0)

        # print("debug lt(<) intervals:", new_intervals)

        IntervalTracker.get_instance().update(new_intervals)

        result_value = diff.data < 0
        return RationalFunction(
            [result_value.astype(float)],
            [np.ones_like(result_value, dtype=float)],
        )

    def __le__(self, other: "RationalFunction") -> "RationalFunction":
        """self <= other"""
        other = self._to_rational_function(other)
        diff = self - other
        neg_num = np.array(
            [-c.flatten()[0] if c.size > 0 else 0 for c in diff.coef_num]
        )
        den = np.array([c.flatten()[0] if c.size > 0 else 1 for c in diff.coef_den])
        z = IntervalTracker.get_instance().get_z()
        new_intervals = calc_polynominal_interval(neg_num, den, z, threshold=0.0)
        IntervalTracker.get_instance().update(new_intervals)

        result_value = diff.data <= 0
        return RationalFunction(
            [result_value.astype(float)],
            [np.ones_like(result_value, dtype=float)],
        )

    def __gt__(self, other: "RationalFunction") -> "RationalFunction":
        """self > other"""
        other = self._to_rational_function(other)
        diff = self - other
        num = np.array([c.flatten()[0] if c.size > 0 else 0 for c in diff.coef_num])
        den = np.array([c.flatten()[0] if c.size > 0 else 1 for c in diff.coef_den])
        z = IntervalTracker.get_instance().get_z()
        new_intervals = calc_polynominal_interval(num, den, z, threshold=0.0)

        # print("debug gt(>) intervals:", new_intervals)

        IntervalTracker.get_instance().update(new_intervals)

        result_value = diff.data > 0
        return RationalFunction(
            [result_value.astype(float)],
            [np.ones_like(result_value, dtype=float)],
        )

    def __ge__(self, other: "RationalFunction") -> "RationalFunction":
        """self >= other"""
        other = self._to_rational_function(other)
        diff = self - other
        num = np.array([c.flatten()[0] if c.size > 0 else 0 for c in diff.coef_num])
        den = np.array([c.flatten()[0] if c.size > 0 else 1 for c in diff.coef_den])
        z = IntervalTracker.get_instance().get_z()
        new_intervals = calc_polynominal_interval(num, den, threshold=0.0)

        IntervalTracker.get_instance().update(new_intervals)

        result_value = diff.data >= 0
        return RationalFunction(
            [result_value.astype(float)],
            [np.ones_like(result_value, dtype=float)],
        )

    def T(self) -> "RationalFunction":
        new_coef_num = [c.T for c in self.coef_num]
        new_coef_den = [c.T for c in self.coef_den]
        return RationalFunction(new_coef_num, new_coef_den)

    def inv(self) -> "RationalFunction":
        new_coef_num = [np.linalg.inv(c) for c in self.coef_num]
        return RationalFunction(new_coef_num, self.coef_den)

    def abs(self) -> "RationalFunction":
        z = IntervalTracker.get_instance().get_z()
        tracker = IntervalTracker.get_instance()
        positive_idx = (self.data >= 0).view(bool)
        negative_idx = (self.data < 0).view(bool)

        new_coef_num = [
            self.coef_num[i] * positive_idx + (-self.coef_num[i]) * negative_idx
            for i in range(len(self.coef_num))
        ]
        new_coef_den = self.coef_den

        if positive_idx.any():
            # positive部分：元の係数をそのまま使用
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

            for j in range(max(pos_coef_num.shape[1], pos_coef_den.shape[1])):
                _num = pos_coef_num[:, j]
                _den = pos_coef_den[:, j]
                # 0埋めして次元を合わせる
                if len(_num) < len(_den):
                    _num = np.concatenate([_num, np.zeros(len(_den) - len(_num))])
                elif len(_den) < len(_num):
                    _den = np.concatenate([_den, np.zeros(len(_num) - len(_den))])
                pos_intervals = calc_polynominal_interval(_num, _den, z)
                print("debug pos_intervals:", pos_intervals)
                tracker.update(pos_intervals)

        if negative_idx.any():
            # negative部分：係数を反転
            neg_coef_num = np.array(
                [-self.coef_num[i][negative_idx] for i in range(len(self.coef_num))]
            )
            neg_coef_den = np.array(
                [self.coef_den[i][negative_idx] for i in range(len(self.coef_den))]
            )
            for j in range(max(neg_coef_num.shape[1], neg_coef_den.shape[1])):
                _num = neg_coef_num[:, j]
                _den = neg_coef_den[:, j]
                if len(_num) < len(_den):
                    _num = np.concatenate([_num, np.zeros(len(_den) - len(_num))])
                elif len(_den) < len(_num):
                    _den = np.concatenate([_den, np.zeros(len(_num) - len(_den))])
                neg_intervals = calc_polynominal_interval(_num, _den, z)
                print("debug neg_intervals:", neg_intervals)
                tracker.update(neg_intervals)
        return RationalFunction(new_coef_num, new_coef_den)

    def argsort(self, axis=-1, kind=None, order=None) -> "RationalFunction":
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
            print("debug argsort intervals:", _intervals)
            tracker.update(_intervals)
        coef_num = [np.array(sorted_idx)]
        coef_den = [np.ones_like(sorted_idx)]
        return RationalFunction(coef_num, coef_den)

    def copy(self):
        """RationalFunctionのコピーを作成"""
        return RationalFunction(
            [c.copy() for c in self.coef_num],
            [c.copy() for c in self.coef_den],
        )

    @property
    def data(self) -> np.ndarray:
        z = IntervalTracker.get_instance().get_z()
        coef_num_val = sum(self.coef_num[i] * (z**i) for i in range(len(self.coef_num)))
        coef_den_val = sum(self.coef_den[i] * (z**i) for i in range(len(self.coef_den)))
        return (coef_num_val / coef_den_val).reshape(self.shape)


def calc_polynominal_interval(
    numerator: np.ndarray, denominator: np.ndarray, z: float, threshold: float = 0.0
) -> RealSubset:
    """有理関数 > threshold となるzの範囲を計算"""
    # 次元合わせ
    if numerator.ndim < denominator.ndim:
        numerator = np.concatenate(
            [numerator, np.zeros(denominator.ndim - numerator.ndim)]
        )
    elif denominator.ndim < numerator.ndim:
        denominator = np.concatenate(
            [denominator, np.zeros(numerator.ndim - denominator.ndim)]
        )
    intervals = polynomial_iso_sign_interval(numerator - denominator * threshold, z)
    # print("debug intervals:", intervals)
    return RealSubset(intervals)
