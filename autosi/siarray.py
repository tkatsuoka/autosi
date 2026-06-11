from __future__ import annotations

import numpy as np

from .interval_tracker import IntervalTracker
from ._mixin_arithmetic import ArithmeticMixin
from ._mixin_comparison import ComparisonMixin
from ._mixin_index import IndexMixin
from ._mixin_array import ArrayMixin
from ._mixin_reduction import ReductionMixin


class siarray(ArithmeticMixin, ComparisonMixin, IndexMixin, ArrayMixin, ReductionMixin):
    """有理関数として表現された配列。選択的推論の区間追跡に使用する。"""

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
            self.coef_num = [c.copy() for c in X.coef_num]
            self.coef_den = [c.copy() for c in X.coef_den]
            self._shape = X._shape
        elif numerator is not None:
            self.coef_num = list(numerator)
            self.coef_den = (
                list(denominator)
                if denominator is not None
                else [np.ones_like(self.coef_num[0])]
            )
            self._shape = numerator[0].shape
        else:
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
        cls,
        numerator: list[np.ndarray],
        denominator: list[np.ndarray] | None = None,
    ) -> "siarray":
        """有理関数として作成"""
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
        """現在の z での有理関数の値を返す"""
        z = IntervalTracker.get_instance().get_z()
        if z is None:
            z = 0.0
        num_val = np.sum(
            [self.coef_num[i] * (z**i) for i in range(len(self.coef_num))], axis=0
        )
        den_val = np.sum(
            [self.coef_den[i] * (z**i) for i in range(len(self.coef_den))], axis=0
        )
        return (num_val / den_val).reshape(self._shape)

    @property
    def deg(self) -> tuple[int, int]:
        """(分子の次数, 分母の次数) を返す"""
        return (len(self.coef_num) - 1, len(self.coef_den) - 1)

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
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return np.prod(self._shape)

    @property
    def dtype(self):
        return self.coef_num[0].dtype

    # ------------------------------------------------------------------------
    # 推論関連メソッド
    # ------------------------------------------------------------------------

    def add_prob_vec_nodes(self, prob_vec: "siarray") -> None:
        """確率変数ベクトルを保存"""
        self.prob_vec = prob_vec

    def add_observed_model(self, model: "siarray") -> None:
        """観測されたモデルを保存"""
        self.observed_model = model

    def add_algorithm(self, algorithm: callable) -> None:
        """選択アルゴリズム関数を保存"""
        self.algorithm_func = algorithm

    def forward_si(self, a: np.ndarray, b: np.ndarray, z: float):
        """アルゴリズムをRationalFunctionで実行し、制約区間を計算"""
        from . import poly_memo

        # レベルC: 新しい z が前回の最終(最小)累積区間に含まれれば、全制約が
        # z を含むため出力・区間が一致する。アルゴリズム再実行を丸ごとスキップ。
        if poly_memo.is_enabled() and z is not None:
            cached = poly_memo.run_lookup(z)
            if cached is not None:
                return cached

        tracker = IntervalTracker.get_instance()
        tracker.reset()
        tracker.set_z(z)

        prob_vec_rf = siarray.from_rational(
            numerator=[a, b],
            denominator=[np.ones_like(a)],
        )
        model = self.algorithm_func(prob_vec_rf)

        intervals = tracker.get()
        output = model.data if isinstance(model, siarray) else model

        if poly_memo.is_enabled() and z is not None:
            poly_memo.run_store(z, output, intervals)
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
        return np.allclose(data1, data2)

    def to_numpy(self) -> np.ndarray:
        """NumPy配列を返す"""
        return self.data.copy()

    def values(self) -> np.ndarray:
        """データの値を返す"""
        return self.data

    def __repr__(self) -> str:
        return f"siarray({repr(self.data)})"

    def __str__(self) -> str:
        return f"siarray({str(self.data)})"
