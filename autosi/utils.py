from __future__ import annotations

import numpy as np

from .rationalfunc import IntervalTracker, RationalFunction


class siarray:
    def __init__(
        self, X: np.ndarray | float | int | "siarray", name: str | None = None
    ):
        """
        Parameters
        ----------
        X : np.ndarray, float, int, or siarray
            ラップする配列またはスカラー値
        name : str, optional
            この変数の名前（デバッグ用）
        """
        if isinstance(X, siarray):
            self.data = X.data
            self.name = name or X.name
        else:
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            self.data = X
            self.name = name or f"input_{id(X)}"

    # ------------------------------------------------------------------------
    # ファクトリメソッド
    # ------------------------------------------------------------------------

    @classmethod
    def constant(cls, value: np.ndarray | float | int) -> "siarray":
        """定数ノードを作成"""
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        obj = cls.__new__(cls)
        obj.data = value
        obj.name = f"const_{id(value)}"
        return obj

    # ------------------------------------------------------------------------
    # ヘルパーメソッド
    # ------------------------------------------------------------------------

    def _to_siarray(self, other) -> "siarray":
        """他の値をsiarrayオブジェクトに変換"""
        if not isinstance(other, siarray):
            return siarray.constant(other)
        return other

    def _dispatch_operation(self, other, op_name):
        """演算をRationalFunctionまたはsiarrayで処理"""
        if isinstance(other, RationalFunction):
            # siarrayをRationalFunctionに変換してから演算
            self_rf = RationalFunction(
                numerator=[self.data],
                denominator=[np.ones_like(self.data)],
                z=None,
            )
            return getattr(self_rf, op_name)(other)
        return None

    # ------------------------------------------------------------------------
    # 算術演算子
    # ------------------------------------------------------------------------

    def __add__(self, other):
        result = self._dispatch_operation(other, "__add__")
        if result is not None:
            return result

        other = self._to_siarray(other)
        result_data = self.data + other.data
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"({self.name}+{other.name})"
        return result

    def __radd__(self, other):
        result = self._dispatch_operation(other, "__radd__")
        if result is not None:
            return result
        return self.__add__(other)

    def __sub__(self, other):
        result = self._dispatch_operation(other, "__sub__")
        if result is not None:
            return result

        other = self._to_siarray(other)
        result_data = self.data - other.data
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"({self.name}-{other.name})"
        return result

    def __rsub__(self, other):
        result = self._dispatch_operation(other, "__rsub__")
        if result is not None:
            return result
        return self._to_siarray(other).__sub__(self)

    def __mul__(self, other):
        result = self._dispatch_operation(other, "__mul__")
        if result is not None:
            return result
        other = self._to_siarray(other)
        result_data = self.data * other.data
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"({self.name}*{other.name})"
        return result

    def __rmul__(self, other):
        result = self._dispatch_operation(other, "__rmul__")
        if result is not None:
            return result
        return self.__mul__(other)

    def __truediv__(self, other):
        result = self._dispatch_operation(other, "__truediv__")
        if result is not None:
            return result

        other = self._to_siarray(other)
        result_data = self.data / other.data
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"({self.name}/{other.name})"
        return result

    def __rtruediv__(self, other):
        result = self._dispatch_operation(other, "__rtruediv__")
        if result is not None:
            return result
        return self._to_siarray(other).__truediv__(self)

    def __matmul__(self, other):
        result = self._dispatch_operation(other, "__matmul__")
        if result is not None:
            return result
        other = self._to_siarray(other)
        result_data = self.data @ other.data
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"({self.name}@{other.name})"
        return result

    def __rmatmul__(self, other):
        result = self._dispatch_operation(other, "__rmatmul__")
        if result is not None:
            return result
        return self._to_siarray(other).__matmul__(self)

    def __neg__(self):
        result_data = -self.data
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"(-{self.name})"
        return result

    def __abs__(self):
        result_data = np.abs(self.data)
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"abs({self.name})"
        return result

    def __pow__(self, power):
        """累乗演算"""
        if isinstance(power, siarray):
            power_val = power.data
        else:
            power_val = power
        result_data = self.data**power_val
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"({self.name}**{power_val})"
        return result

    # ------------------------------------------------------------------------
    # 比較演算子
    # ------------------------------------------------------------------------

    def __lt__(self, other):
        result = self._dispatch_operation(other, "__lt__")
        if result is not None:
            return result

        other = self._to_siarray(other)
        result_data = self.data < other.data
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"({self.name}<{other.name})"
        return result

    def __le__(self, other):
        result = self._dispatch_operation(other, "__le__")
        if result is not None:
            return result

        other = self._to_siarray(other)
        result_data = self.data <= other.data
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"({self.name}<={other.name})"
        return result

    def __gt__(self, other):
        result = self._dispatch_operation(other, "__gt__")
        if result is not None:
            return result

        other = self._to_siarray(other)
        result_data = self.data > other.data
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"({self.name}>{other.name})"
        return result

    def __ge__(self, other):
        result = self._dispatch_operation(other, "__ge__")
        if result is not None:
            return result

        other = self._to_siarray(other)
        result_data = self.data >= other.data
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"({self.name}>={other.name})"
        return result

    def __eq__(self, other):
        other = self._to_siarray(other)
        result_data = np.equal(self.data, other.data)
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"({self.name}=={other.name})"
        return result

    def __ne__(self, other):
        other = self._to_siarray(other)
        result_data = np.not_equal(self.data, other.data)
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"({self.name}!={other.name})"
        return result

    # ------------------------------------------------------------------------
    # インデックス演算
    # ------------------------------------------------------------------------

    def __getitem__(self, key):
        """要素の取得"""
        if isinstance(key, RationalFunction):
            key = key.coef_num[0]
        elif isinstance(key, siarray):
            key = key.data
        
        result_data = self.data[key]
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"{self.name}[{key}]"
        return result

    def __setitem__(self, key, value):
        """要素の設定（破壊的操作）"""
        value_siarray = value if isinstance(value, siarray) else siarray.constant(value)
        self.data[key] = value_siarray.data
        self.name = f"{self.name}[{key}]=updated"

    # ------------------------------------------------------------------------
    # 配列操作メソッド
    # ------------------------------------------------------------------------

    @property
    def T(self):
        """転置"""
        result_data = self.data.T
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"{self.name}.T"
        return result

    def transpose(self, *axes):
        """転置（軸指定可能）"""
        if axes:
            result_data = np.transpose(self.data, axes)
        else:
            result_data = self.data.T

        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"{self.name}.transpose({axes})"
        return result

    def inv(self):
        """逆行列"""
        result_data = np.linalg.inv(self.data)
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"{self.name}.inv()"
        return result

    def reshape(self, *shape):
        """形状変更"""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]

        result_data = self.data.reshape(shape)
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"{self.name}.reshape({shape})"
        return result

    def argsort(self, axis=-1, kind=None, order=None):
        """ソートのインデックスを返す"""
        result_data = np.argsort(self.data, axis=axis, kind=kind, order=order)
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"{self.name}.argsort(axis={axis})"
        return result

    def copy(self):
        """siarrayオブジェクトのコピーを作成"""
        copied = siarray.__new__(siarray)
        copied.data = self.data.copy()
        copied.name = self.name
        return copied

    # ------------------------------------------------------------------------
    # 集約メソッド
    # ------------------------------------------------------------------------

    def sum(
        self, axis=None, dtype=None, out=None, keepdims=False, initial=None, where=None
    ):
        """合計"""
        # NumPy互換のため、サポートしていない引数は無視
        if out is not None:
            raise NotImplementedError("out parameter is not supported for siarray")
        if initial is not None:
            raise NotImplementedError("initial parameter is not supported for siarray")
        if where is not None:
            raise NotImplementedError("where parameter is not supported for siarray")

        result_data = np.sum(self.data, axis=axis, dtype=dtype, keepdims=keepdims)
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"{self.name}.sum(axis={axis})"
        return result

    def mean(self, axis=None, keepdims=False):
        """平均"""
        result_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"{self.name}.mean(axis={axis})"
        return result

    # ------------------------------------------------------------------------
    # プロパティ
    # ------------------------------------------------------------------------

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

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

    def calc_intervals(self, a: np.ndarray, b: np.ndarray, z: float):
        """
        アルゴリズムをRationalFunctionで実行し、制約区間を計算

        Parameters
        ----------
        a : np.ndarray
            多項式の定数項係数
        b : np.ndarray
            多項式の1次項係数
        z : float
            評価点

        Returns
        -------
        output : np.ndarray
            評価結果
        intervals : RealSubset
            有効な区間
        """

        # intervalsをリセット
        tracker = IntervalTracker.get_instance()
        tracker.reset()

        # 通常の値で実行してモデルを取得
        prob_vec = a + b * z
        model = self.algorithm_func(siarray(prob_vec))

        # RationalFunctionで実行して区間を収集
        prob_vec_rf = RationalFunction(
            numerator=[a, b],
            denominator=[np.ones_like(a)],
            z=z,
        )

        # RationalFunctionでラップした環境でアルゴリズム実行
        print("debug RationalFunction execution")
        self.algorithm_func(prob_vec_rf)

        # 最終的なintervalsを取得
        intervals = tracker.get()
        # モデルの出力値を取得
        if isinstance(model, siarray):
            output = model.data
        else:
            output = model

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
        return f"siarray({self.name}, shape={self.data.shape})"


# ============================================================================
# ユーティリティ関数
# ============================================================================


def sum(x, axis=None, keepdims=False):
    """合計（siarray対応）"""
    if isinstance(x, siarray):
        return x.sum(axis=axis, keepdims=keepdims)
    elif isinstance(x, RationalFunction):
        # 実装してないエラー
        raise NotImplementedError("sum is not implemented for RationalFunction")
    else:
        return siarray(x).sum(axis=axis, keepdims=keepdims)


def abs(x):
    """絶対値（siarray対応）"""
    if isinstance(x, siarray):
        return x.__abs__()
    elif isinstance(x, RationalFunction):
        return x.abs()
    else:
        return siarray(x).__abs__()


def argsort(x, axis=-1, kind=None, order=None):
    """ソートのインデックス（siarray対応）"""
    if isinstance(x, siarray):
        return x.argsort(axis=axis, kind=kind, order=order)
    elif isinstance(x, RationalFunction):
        return x.argsort(axis=axis, kind=kind, order=order)
    else:
        return siarray(x).argsort(axis=axis, kind=kind, order=order)
