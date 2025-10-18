import numpy as np
from sicore import RealSubset, polynomial_iso_sign_interval


class RationalFunction:
    def __init__(
        self,
        numerator: list[np.ndarray],
        denominator: list[np.ndarray] | None,
        z: float | None = None,
        intervals: RealSubset | None = None,
    ):
        self.coef_num = list(numerator)
        self.coef_den = (
            list(denominator)
            if denominator is not None
            else [np.ones_like(self.coef_num[0])]
        )
        self.z = z
        if intervals is None:
            self.intervals = RealSubset([[-float("inf"), float("inf")]])
        else:
            self.intervals = intervals
        self.shape = numerator[0].shape

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
        intervals = self.intervals & other.intervals
        num1 = self._poly_mul(self.coef_num, other.coef_den)
        num2 = self._poly_mul(other.coef_num, self.coef_den)
        new_coef_num = self._poly_add(num1, num2)
        new_coef_den = self._poly_mul(self.coef_den, other.coef_den)
        return RationalFunction(new_coef_num, new_coef_den, self.z, intervals)

    def __sub__(self, other: "RationalFunction") -> "RationalFunction":
        intervals = self.intervals & other.intervals
        num1 = self._poly_mul(self.coef_num, other.coef_den)
        num2 = self._poly_mul(other.coef_num, self.coef_den)
        new_coef_num = [a - b for a, b in zip(num1, num2)]
        new_coef_den = self._poly_mul(self.coef_den, other.coef_den)
        return RationalFunction(new_coef_num, new_coef_den, self.z, intervals)

    def __mul__(self, other: "RationalFunction") -> "RationalFunction":
        intervals = self.intervals & other.intervals
        new_coef_num = self._poly_mul(self.coef_num, other.coef_num)
        new_coef_den = self._poly_mul(self.coef_den, other.coef_den)
        return RationalFunction(new_coef_num, new_coef_den, self.z, intervals)

    def __truediv__(self, other: "RationalFunction") -> "RationalFunction":
        intervals = self.intervals & other.intervals
        new_coef_num = self._poly_mul(self.coef_num, other.coef_den)
        new_coef_den = self._poly_mul(self.coef_den, other.coef_num)
        return RationalFunction(new_coef_num, new_coef_den, self.z, intervals)

    def __matmul__(self, other: "RationalFunction") -> "RationalFunction":
        intervals = self.intervals & other.intervals
        new_coef_num = [c1 @ c2 for c1 in self.coef_num for c2 in other.coef_num]
        new_coef_den = [c1 @ c2 for c1 in self.coef_den for c2 in other.coef_den]
        return RationalFunction(new_coef_num, new_coef_den, self.z, intervals)

    # スライスに対応
    def __getitem__(self, key) -> "RationalFunction":
        return RationalFunction(
            [c[key] for c in self.coef_num],
            [c[key] for c in self.coef_den],
            self.z,
            self.intervals,
        )

    def T(self) -> "RationalFunction":
        new_coef_num = [c.T for c in self.coef_num]
        new_coef_den = [c.T for c in self.coef_den]
        return RationalFunction(new_coef_num, new_coef_den, self.z, self.intervals)

    # 1次的に定数ベクトルのみに対応
    def inv(self) -> "RationalFunction":
        new_coef_num = [np.linalg.inv(c) for c in self.coef_num]
        # new_coef_den = [np.linalg.inv(c) for c in self.coef_den]
        return RationalFunction(new_coef_num, self.coef_den, self.z, self.intervals)

    def abs(self, z) -> "RationalFunction":
        positive_idx = (self.output() >= 0).view(bool)
        negative_idx = (self.output() < 0).view(bool)

        new_coef_num = [
            self.coef_num[i] * positive_idx + (-self.coef_num[i]) * negative_idx
            for i in range(len(self.coef_num))
        ]
        new_coef_den = self.coef_den

        # print('debug', self.coef_num)
        # print('debug', self.coef_den)

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
                self.intervals = self.intervals & pos_intervals

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
                self.intervals = self.intervals & neg_intervals
        return RationalFunction(new_coef_num, new_coef_den, self.z, self.intervals)

    def argsort(self, axis=-1, kind=None, order=None) -> "RationalFunction":
        output = self.output()
        sorted_idx = np.argsort(output, axis=axis, kind=kind, order=order)
        for i in range(len(sorted_idx) - 1):
            idx = sorted_idx[i]
            next_idx = sorted_idx[i + 1]
            inequality = self[idx] - self[next_idx]
            _intervals = calc_polynominal_interval(
                np.array(inequality.coef_num), np.array(inequality.coef_den), self.z
            )
            self.intervals = self.intervals & _intervals
        coef_num = [np.array(sorted_idx)]
        coef_den = [np.ones_like(sorted_idx)]
        return RationalFunction(coef_num, coef_den, self.z, self.intervals)

    def output(self) -> np.ndarray:
        coef_num_val = sum(
            self.coef_num[i] * (self.z**i) for i in range(len(self.coef_num))
        )
        coef_den_val = sum(
            self.coef_den[i] * (self.z**i) for i in range(len(self.coef_den))
        )
        return (coef_num_val / coef_den_val).reshape(self.shape)


def calc_polynominal_interval(
    numerator: np.ndarray, denominator: np.ndarray, z: float, threshold: float = 0.0
) -> RealSubset:
    """
    有理関数 > threshold となるzの範囲を計算
    """
    # print("debug", numerator)
    # print("debug", denominator)
    # print("debug", numerator - denominator * threshold)
    # denominatorとnumratorの次元合わせはなんとかしたい
    if numerator.ndim < denominator.ndim:
        numerator = np.concatenate(
            [numerator, np.zeros(denominator.ndim - numerator.ndim)]
        )
    elif denominator.ndim < numerator.ndim:
        denominator = np.concatenate(
            [denominator, np.zeros(numerator.ndim - denominator.ndim)]
        )
    intervals = polynomial_iso_sign_interval(numerator - denominator * threshold, z)

    return RealSubset(intervals)
