from collections import deque
from enum import Enum
from typing import Union

import numpy as np
from sicore import RealSubset

from .rationalfunc import RationalFunction


class OpType(Enum):
    """演算の種類を定義"""

    INPUT = "input"
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    TRUEDIV = "truediv"
    MATMUL = "matmul"
    NEG = "neg"
    TRANSPOSE = "transpose"
    INV = "inv"
    GETITEM = "getitem"
    POW = "pow"
    SUM = "sum"
    MEAN = "mean"
    RESHAPE = "reshape"
    CONSTANT = "constant"
    ABS = "abs"
    ARGSORT = "argsort"
    WHERE = "where"


class Node:
    """計算グラフのノード"""

    _id_counter = 0

    def __init__(
        self,
        op_type: OpType,
        input_node: list["Node"] | None = None,
        value: np.ndarray | None = None,
        output: np.ndarray | None = None,  # 計算結果を保存
        output_node: list["Node"] | None = None,  # 出力先ノードのリスト
        **kwargs,
    ):
        self.id = Node._id_counter
        Node._id_counter += 1
        self.op_type = op_type
        self.input_node = input_node or []
        self.output_node = output_node or []  # このノードの出力を使用するノードのリスト
        self.value = value  # 定数ノードの場合の値
        self.output = output  # このノードの計算結果
        self.kwargs = kwargs  # 演算に必要な追加パラメータ
        self.shape = None

        # outputが与えられた場合はそのshapeを使用
        if output is not None:
            self.shape = output.shape
        elif value is not None:
            self.shape = value.shape

        # 入力ノードに対して、このノードを出力先として登録
        for input_node in self.input_node:
            if self not in input_node.output_node:
                input_node.output_node.append(self)


class siarray:
    """numpyの演算をラップし、計算グラフを構築するクラス"""

    def __init__(self, X: np.ndarray | float | int, name: str | None = None):
        if isinstance(X, siarray):
            self.data = X.data
            self.node = X.node
            self.name = name or X.name
        else:
            # numpy配列に変換
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            self.data = X
            self.name = name or f"input_{Node._id_counter}"
            # 入力ノードを作成（outputも保存）
            self.node = Node(OpType.INPUT, value=X, output=X)

    @classmethod
    def constant(cls, value: np.ndarray | float | int):
        """定数ノードを作成"""
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        obj = cls.__new__(cls)
        obj.data = value
        obj.name = f"const_{Node._id_counter}"
        # 定数ノードもoutputを保存
        obj.node = Node(OpType.CONSTANT, value=value, output=value)
        return obj

    def __repr__(self):
        return f"siarray({self.name}, shape={self.data.shape}, op={self.node.op_type.value})"

    def __add__(self, other):
        other = self._to_siarray(other)
        result_data = self.data + other.data
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"({self.name}+{other.name})"
        # outputを保存し、入力ノードのoutput_nodeリストも更新される
        result.node = Node(
            OpType.ADD, input_node=[self.node, other.node], output=result_data
        )
        return result

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._to_siarray(other)
        result_data = self.data - other.data
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"({self.name}-{other.name})"
        result.node = Node(
            OpType.SUB, input_node=[self.node, other.node], output=result_data
        )
        return result

    def __rsub__(self, other):
        other = self._to_siarray(other)
        return other.__sub__(self)

    def __mul__(self, other):
        other = self._to_siarray(other)
        result_data = self.data * other.data
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"({self.name}*{other.name})"
        result.node = Node(
            OpType.MUL, input_node=[self.node, other.node], output=result_data
        )
        return result

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = self._to_siarray(other)
        result_data = self.data / other.data
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"({self.name}/{other.name})"
        result.node = Node(
            OpType.TRUEDIV, input_node=[self.node, other.node], output=result_data
        )
        return result

    def __rtruediv__(self, other):
        other = self._to_siarray(other)
        return other.__truediv__(self)

    def __matmul__(self, other):
        other = self._to_siarray(other)
        result_data = self.data @ other.data
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"({self.name}@{other.name})"
        result.node = Node(
            OpType.MATMUL, input_node=[self.node, other.node], output=result_data
        )
        return result

    def __rmatmul__(self, other):
        other = self._to_siarray(other)
        return other.__matmul__(self)

    def __neg__(self):
        result_data = -self.data
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"(-{self.name})"
        result.node = Node(OpType.NEG, input_node=[self.node], output=result_data)
        return result

    def __pow__(self, power):
        if isinstance(power, siarray):
            power_val = power.data
            power_node = power.node
        else:
            power_val = power
            power_node = Node(
                OpType.CONSTANT, value=np.array(power), output=np.array(power)
            )

        result_data = self.data**power_val
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"({self.name}**{power_val})"
        result.node = Node(
            OpType.POW, input_node=[self.node, power_node], output=result_data
        )
        return result

    def __abs__(self):
        """絶対値を計算"""
        result_data = np.abs(self.data)
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"abs({self.name})"
        result.node = Node(OpType.ABS, input_node=[self.node], output=result_data)
        return result

    # インデックス演算
    def __getitem__(self, key):
        # keyがsiarrayの場合、その実際のデータを使用
        if isinstance(key, siarray):
            actual_key = key.data
            # スカラーの場合はitemで取り出す
            if actual_key.ndim == 0 or (actual_key.ndim == 1 and actual_key.size == 1):
                actual_key = actual_key.item()
            result_data = self.data[actual_key]
            result = siarray.__new__(siarray)
            result.data = result_data
            result.name = f"{self.name}[{key.name}]"
            # keyのノードも入力として記録、outputも保存
            result.node = Node(
                OpType.GETITEM,
                input_node=[self.node, key.node],
                key=actual_key,
                output=result_data,
            )
        else:
            result_data = self.data[key]
            result = siarray.__new__(siarray)
            result.data = result_data
            result.name = f"{self.name}[{key}]"
            result.node = Node(
                OpType.GETITEM, input_node=[self.node], key=key, output=result_data
            )
        return result

    # 属性とメソッド
    @property
    def T(self):
        result_data = self.data.T
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"{self.name}.T"
        result.node = Node(OpType.TRANSPOSE, input_node=[self.node], output=result_data)
        return result

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

    def transpose(self, *axes):
        if axes:
            result_data = np.transpose(self.data, axes)
        else:
            result_data = self.data.T
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"{self.name}.transpose({axes})"
        result.node = Node(
            OpType.TRANSPOSE, input_node=[self.node], axes=axes, output=result_data
        )
        return result

    def inv(self):
        result_data = np.linalg.inv(self.data)
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"{self.name}.inv()"
        result.node = Node(OpType.INV, input_node=[self.node], output=result_data)
        return result

    def sum(self, axis=None, keepdims=False):
        result_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"{self.name}.sum(axis={axis})"
        result.node = Node(
            OpType.SUM,
            input_node=[self.node],
            axis=axis,
            keepdims=keepdims,
            output=result_data,
        )
        return result

    def mean(self, axis=None, keepdims=False):
        result_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"{self.name}.mean(axis={axis})"
        result.node = Node(
            OpType.MEAN,
            input_node=[self.node],
            axis=axis,
            keepdims=keepdims,
            output=result_data,
        )
        return result

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        result_data = self.data.reshape(shape)
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"{self.name}.reshape({shape})"
        result.node = Node(
            OpType.RESHAPE, input_node=[self.node], shape=shape, output=result_data
        )
        return result

    def argsort(self, axis=-1, kind=None, order=None):
        """ソートした要素のインデックスを返す"""
        result_data = np.argsort(self.data, axis=axis, kind=kind, order=order)
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"{self.name}.argsort(axis={axis})"
        result.node = Node(
            OpType.ARGSORT,
            input_node=[self.node],
            axis=axis,
            kind=kind,
            order=order,
            output=result_data,
        )
        return result

    def where(self, condition, x, y):
        """条件に基づいて要素を選択"""
        condition = self._to_siarray(condition)
        x = self._to_siarray(x)
        y = self._to_siarray(y)
        result_data = np.where(condition.data, x.data, y.data)
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"where({condition.name}, {x.name}, {y.name})"
        result.node = Node(
            OpType.CONSTANT,
            input_node=[condition.node, x.node, y.node],
            output=result_data,
        )
        return result

    def _topological_sort(self, node: Node = None) -> list[Node]:
        """
        計算グラフをトポロジカルソートする

        Parameters:
        -----------
        node : Node, optional
            ソートの起点となるノード（デフォルトはself.node）

        Returns:
        --------
        List[Node]
            トポロジカル順序でソートされたノードのリスト
        """
        if node is None:
            node = self.node

        # 訪問済みノードと結果リスト
        visited = set()
        result = []

        def dfs(n: Node):
            if n.id in visited:
                return
            visited.add(n.id)

            # 入力ノードを先に処理
            for input_node in n.input_node:
                dfs(input_node)

            # 現在のノードを結果に追加
            result.append(n)

        dfs(node)
        return result

    def _topological_sort_kahn(self, node: Node = None) -> list[Node]:
        """
        Kahnのアルゴリズムを使用したトポロジカルソート
        （循環検出機能付き）

        Parameters:
        -----------
        node : Node, optional
            ソートの起点となるノード（デフォルトはself.node）

        Returns:
        --------
        list[Node]
            トポロジカル順序でソートされたノードのリスト

        Raises:
        -------
        ValueError
            グラフに循環が検出された場合
        """
        if node is None:
            node = self.node

        # まず、関連する全ノードを収集
        all_nodes = self._collect_all_nodes(node)

        # 各ノードの入次数を計算
        in_degree = {n.id: 0 for n in all_nodes}

        for n in all_nodes:
            for output in n.output_node:
                if output.id in in_degree:
                    in_degree[output.id] += 1

        # 入次数が0のノードをキューに追加
        queue = deque([n for n in all_nodes if in_degree[n.id] == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)

            # 出力ノードの入次数を減らす
            for output in current.output_node:
                if output.id in in_degree:
                    in_degree[output.id] -= 1
                    if in_degree[output.id] == 0:
                        queue.append(output)

        # 全ノードが処理されたか確認（循環チェック）
        if len(result) != len(all_nodes):
            raise ValueError("計算グラフに循環が検出されました")

        return result

    def _collect_all_nodes(self, node: Node) -> set[Node]:
        """
        指定されたノードから到達可能な全ノードを収集

        Parameters:
        -----------
        node : Node
            起点となるノード

        Returns:
        --------
        Set[Node]
            到達可能な全ノードのセット
        """
        visited = set()
        to_visit = [node]

        while to_visit:
            current = to_visit.pop()
            if current.id not in visited:
                visited.add(current.id)
                to_visit.extend(current.input_node)
                # 逆方向（出力ノード）も含める場合
                # to_visit.extend(current.output_node)

        return {n for n in [node] + node.input_node if n.id in visited}

    def add_prob_vec_nodes(self, prob_vec: "siarray"):
        """
        確率変数ベクトルのノードを保存
        """
        self.prob_vec_nodes = prob_vec.node

    def add_observed_model(self, model: "siarray"):
        """
        観測されたモデルのノードを保存
        """
        self.observed_model = model

    def algorithm(self, a: np.ndarray, b: np.ndarray, z: float) -> np.ndarray:
        """
        計算グラフをRationalFunctionに変換し、a,bを多項式係数として使用して演算を実行
        （トポロジカルソートを使用した改良版）
        """
        self.intervals = RealSubset([[-float("inf"), float("inf")]])

        # トポロジカルソートでノードの計算順序を決定
        sorted_nodes = self._topological_sort()

        # 入力ノードを特定
        input_nodes = [n for n in sorted_nodes if n.op_type == OpType.INPUT]

        if len(input_nodes) == 0:
            raise ValueError("No input nodes found in the computation graph")

        # 入力ノードにRationalFunctionを割り当てる
        node_values = {}

        # 入力ノードは定数として設定
        for node in input_nodes:
            data = node.value if node.value is not None else node.output
            node_values[node.id] = RationalFunction(
                numerator=[data],
                denominator=[np.ones_like(data)],
                z=z,
            )

        # a, bベクトルをinput_shapeに合わせる
        a = a.reshape(self.prob_vec_nodes.shape)
        b = b.reshape(self.prob_vec_nodes.shape)

        # 確率変数の入力ノードには[a, b]を係数とする多項式を設定
        node_values[self.prob_vec_nodes.id] = RationalFunction(
            numerator=[a, b],
            denominator=[np.ones_like(a)],
            z=z,
        )

        # トポロジカル順序で計算を実行
        for node in sorted_nodes:
            if node.id in node_values:
                continue  # 既に値が設定されている（入力ノード）

            # このノードの計算を実行
            # print(f"Computing node #{node.id} of type {node.op_type}")
            # print(f" Inputs: {[n.id for n in node.input_node]}")
            result = self._forward_si(node, node_values, z)
            # print("foward_si")
            node_values[node.id] = result
            # print(result.intervals)
            self.intervals = self.intervals & result.intervals
            # print(self.intervals)


        # print(self.observed_model.values())
        return node_values[self.observed_model.node.id].output(), self.intervals

    def _forward_si(self, node: Node, node_values: dict, z: float) -> RationalFunction:
        """
        単一ノードの計算を実行
        （トポロジカルソート済みなので、入力ノードの値は既に計算済み）
        """
        if node.op_type == OpType.CONSTANT:
            data = node.value if node.value is not None else node.output
            return RationalFunction(
                numerator=[data],
                denominator=[np.ones_like(data)],
                z=z,
            )

        # 入力値を取得
        input_rfs = [node_values[n.id] for n in node.input_node]

        # 演算を実行
        if node.op_type == OpType.ADD:
            return input_rfs[0] + input_rfs[1]
        elif node.op_type == OpType.SUB:
            return input_rfs[0] - input_rfs[1]
        elif node.op_type == OpType.MUL:
            return input_rfs[0] * input_rfs[1]
        elif node.op_type == OpType.TRUEDIV:
            return input_rfs[0] / input_rfs[1]
        elif node.op_type == OpType.MATMUL:
            return input_rfs[0] @ input_rfs[1]
        elif node.op_type == OpType.NEG:
            minus_one = RationalFunction(
                numerator=[-np.ones_like(input_rfs[0].coef_num[0])],
                denominator=[np.ones_like(input_rfs[0].coef_num[0])],
                z=z,
            )
            return minus_one * input_rfs[0]
        elif node.op_type == OpType.TRANSPOSE:
            return input_rfs[0].T()
        elif node.op_type == OpType.INV:
            return input_rfs[0].inv()
        elif node.op_type == OpType.POW:
            # 累乗の処理
            base = input_rfs[0]
            if len(input_rfs) >= 2:
                power_rf = input_rfs[1]
                if len(power_rf.coef_num) == 1:
                    power_val = power_rf.coef_num[0]
                    if hasattr(power_val, "item"):
                        power_val = power_val.item()
                else:
                    raise NotImplementedError("Non-constant power not supported")
            else:
                power_node = node.input_node[1] if len(node.input_node) > 1 else None
                if power_node and (
                    power_node.value is not None or power_node.output is not None
                ):
                    power_val = (
                        power_node.value
                        if power_node.value is not None
                        else power_node.output
                    )
                    if hasattr(power_val, "item"):
                        power_val = power_val.item()
                else:
                    raise ValueError("Power value not found")

            if isinstance(power_val, (int, np.integer)) and power_val >= 0:
                result = base
                for _ in range(int(power_val) - 1):
                    result = result * base
                return result
            else:
                raise NotImplementedError(f"Power {power_val} not supported")
        elif node.op_type == OpType.GETITEM:
            key = node.kwargs.get("key")
            return input_rfs[0][key]
        elif node.op_type == OpType.ABS:
            return input_rfs[0].abs(z)
        elif node.op_type == OpType.ARGSORT:
            return input_rfs[0].argsort(
                axis=node.kwargs.get("axis", -1),
                kind=node.kwargs.get("kind", None),
                order=node.kwargs.get("order", None),
            )
        elif node.op_type == OpType.RESHAPE:
            return input_rfs[0]
        else:
            raise NotImplementedError(f"Operation {node.op_type} not implemented")

    def model_selector(
        self,
        searched_model: Union[np.ndarray, "siarray"],
    ) -> bool | np.ndarray:
        """
        2つの入力が同じかどうかを判定する

        Parameters:
        -----------
        observed_model : np.ndarray or siarray
            比較する最初の入力/
        searched_model : np.ndarray or siarray
            比較する2番目の入力
        Returns:
        --------
        bool or np.ndarray
            スカラーの場合はbool、配列の場合は要素ごとの比較結果のnp.ndarray
        """
        if isinstance(self.observed_model, siarray):
            data1 = self.observed_model.data
        else:
            data1 = np.asarray(self.observed_model)

        if isinstance(searched_model, siarray):
            data2 = searched_model.data
        else:
            data2 = np.asarray(searched_model)
        comparison = np.allclose(data1, data2)
        print('debug', data1, data2, comparison)

        return comparison

    def _to_siarray(self, other):
        """他の値をsiarrayオブジェクトに変換"""
        if not isinstance(other, siarray):
            return siarray.constant(other)
        return other

    def to_numpy(self):
        """numpy配列を返す"""
        return self.data.copy()

    def values(self):
        """データの値を表示（デバッグ用）"""
        return self.data

    def show(self, max_rows=10, max_cols=10):
        """データの要約を表示"""
        print(f"siarray object: {self.name}")
        print(f"Shape: {self.data.shape}")
        print(f"Operation: {self.node.op_type.value}")
        print(f"Data type: {self.data.dtype}")

        if self.data.ndim == 1:
            if self.data.size <= max_rows:
                print("Values:", self.data)
            else:
                print(f"First {max_rows // 2}:", self.data[: max_rows // 2])
                print(f"Last {max_rows // 2}:", self.data[-max_rows // 2 :])
        elif self.data.ndim == 2:
            rows, cols = self.data.shape
            if rows <= max_rows and cols <= max_cols:
                print("Values:\n", self.data)
            else:
                print(
                    f"Top-left ({min(max_rows // 2, rows)} x {min(max_cols // 2, cols)}):"
                )
                print(self.data[: min(max_rows // 2, rows), : min(max_cols // 2, cols)])
                if rows > max_rows or cols > max_cols:
                    print("...")
        else:
            print(f"Array with shape {self.data.shape}")
            print("First elements:", self.data.flat[: min(20, self.data.size)])

    def to_rational_function(
        self, z: float | None = None, intervals: RealSubset | None = None
    ) -> "RationalFunction":
        """計算グラフをRationalFunctionに変換"""
        if intervals is None:
            intervals = RealSubset([[-float("inf"), float("inf")]])
        from_graph = GraphToRational(self.node, z, intervals)
        return from_graph.convert()


# モジュールレベルの関数
def abs(x):
    if isinstance(x, siarray):
        return x.__abs__()
    else:
        x = siarray(x)
        return x.__abs__()


def argsort(x, axis=-1, kind=None, order=None):
    if isinstance(x, siarray):
        return x.argsort(axis=axis, kind=kind, order=order)
    else:
        x = siarray(x)
        return x.argsort(axis=axis, kind=kind, order=order)


def where(condition, x, y):
    """条件に基づいて要素を選択（グラフ構造を保存）"""
    # すべてをsiarrayに変換

    if not isinstance(x, siarray):
        x = siarray.constant(x)
    if not isinstance(y, siarray):
        y = siarray.constant(y)
    
    # numpy演算で結果を計算
    result_data = np.where(condition.data, x.data, y.data)
    
    # 新しいsiarrayオブジェクトを作成
    result = siarray.__new__(siarray)
    result.data = result_data
    result.name = f"where({condition.name}, {x.name}, {y.name})"
    
    # WHEREノードを作成（3つの入力ノードを持つ）
    result.node = Node(
        OpType.WHERE,
        input_node=[condition.node, x.node, y.node],
        output=result_data,
    )
    
    return result


def trace_graph(
    node: Node,
    indent: int = 0,
    visited: set = None,
    show_values: bool = False,
    names: dict = None,
) -> str:
    """
    計算グラフを上から下（根から葉）に向かって表示

    Parameters:
    -----------
    node : Node
        表示するノード
    indent : int
        インデントレベル
    visited : set
        既に訪問したノードのIDセット
    show_values : bool
        出力値の概要を表示するかどうか
    names : dict
        ノードIDから変数名へのマッピング
    """
    if visited is None:
        visited = set()
        names = {}

    # ノードに名前を割り当て
    if node.id not in names:
        if node.op_type == OpType.INPUT:
            names[node.id] = f"input{node.id}"
        elif node.op_type == OpType.CONSTANT:
            if node.output is not None and node.output.ndim == 0:
                # スカラー定数の場合は値を表示
                names[node.id] = str(node.output.item())
            else:
                names[node.id] = f"const{node.id}"

    indent_str = "  " * indent

    # ノード表示を構築（演算記号を使用）
    display = ""

    if node.op_type == OpType.INPUT:
        display = names[node.id]
    elif node.op_type == OpType.CONSTANT:
        display = names[node.id]
    elif node.op_type == OpType.ADD and len(node.input_node) == 2:
        left = names.get(node.input_node[0].id, f"#{node.input_node[0].id}")
        right = names.get(node.input_node[1].id, f"#{node.input_node[1].id}")
        display = f"({left} + {right})"
        names[node.id] = display
    elif node.op_type == OpType.SUB and len(node.input_node) == 2:
        left = names.get(node.input_node[0].id, f"#{node.input_node[0].id}")
        right = names.get(node.input_node[1].id, f"#{node.input_node[1].id}")
        display = f"({left} - {right})"
        names[node.id] = display
    elif node.op_type == OpType.MUL and len(node.input_node) == 2:
        left = names.get(node.input_node[0].id, f"#{node.input_node[0].id}")
        right = names.get(node.input_node[1].id, f"#{node.input_node[1].id}")
        display = f"({left} * {right})"
        names[node.id] = display
    elif node.op_type == OpType.TRUEDIV and len(node.input_node) == 2:
        left = names.get(node.input_node[0].id, f"#{node.input_node[0].id}")
        right = names.get(node.input_node[1].id, f"#{node.input_node[1].id}")
        display = f"({left} / {right})"
        names[node.id] = display
    elif node.op_type == OpType.MATMUL and len(node.input_node) == 2:
        left = names.get(node.input_node[0].id, f"#{node.input_node[0].id}")
        right = names.get(node.input_node[1].id, f"#{node.input_node[1].id}")
        display = f"({left} @ {right})"
        names[node.id] = display
    elif node.op_type == OpType.NEG:
        operand = names.get(node.input_node[0].id, f"#{node.input_node[0].id}")
        display = f"(-{operand})"
        names[node.id] = display
    elif node.op_type == OpType.TRANSPOSE:
        operand = names.get(node.input_node[0].id, f"#{node.input_node[0].id}")
        display = f"{operand}.T"
        names[node.id] = display
    elif node.op_type == OpType.INV:
        operand = names.get(node.input_node[0].id, f"#{node.input_node[0].id}")
        display = f"inv({operand})"
        names[node.id] = display
    elif node.op_type == OpType.POW and len(node.input_node) >= 1:
        base = names.get(node.input_node[0].id, f"#{node.input_node[0].id}")
        if len(node.input_node) > 1:
            power = names.get(node.input_node[1].id, "?")
        else:
            power = "?"
        display = f"({base}**{power})"
        names[node.id] = display
    elif node.op_type == OpType.ABS:
        operand = names.get(node.input_node[0].id, f"#{node.input_node[0].id}")
        display = f"abs({operand})"
        names[node.id] = display
    elif node.op_type == OpType.ARGSORT:
        operand = names.get(node.input_node[0].id, f"#{node.input_node[0].id}")
        axis = node.kwargs.get("axis", -1)
        display = f"{operand}.argsort(axis={axis})"
        names[node.id] = display
    elif node.op_type == OpType.SUM:
        operand = names.get(node.input_node[0].id, f"#{node.input_node[0].id}")
        axis = node.kwargs.get("axis", None)
        display = f"{operand}.sum(axis={axis})"
        names[node.id] = display
    elif node.op_type == OpType.MEAN:
        operand = names.get(node.input_node[0].id, f"#{node.input_node[0].id}")
        axis = node.kwargs.get("axis", None)
        display = f"{operand}.mean(axis={axis})"
        names[node.id] = display
    elif node.op_type == OpType.RESHAPE:
        operand = names.get(node.input_node[0].id, f"#{node.input_node[0].id}")
        shape = node.kwargs.get("shape", "?")
        display = f"{operand}.reshape{shape}"
        names[node.id] = display
    elif node.op_type == OpType.GETITEM:
        operand = names.get(node.input_node[0].id, f"#{node.input_node[0].id}")
        key = node.kwargs.get("key", "?")
        # keyを簡潔に表示
        if isinstance(key, tuple):
            key_str = str(key)
        elif isinstance(key, slice):
            key_str = f"{key.start}:{key.stop}" + (f":{key.step}" if key.step else "")
        else:
            key_str = str(key)
        display = f"{operand}[{key_str}]"
        names[node.id] = display
    else:
        display = node.op_type.value

    # 情報を構築
    info_parts = [display]

    # 出力の形状
    if node.output is not None:
        info_parts.append(f"-> shape{node.output.shape}")
        if show_values and node.output.size <= 5:
            if node.output.ndim == 0:
                info_parts.append(f"= {node.output.item():.4f}")
            else:
                vals = node.output.flatten()[:5]
                vals_str = ", ".join([f"{v:.4f}" for v in vals])
                info_parts.append(f"= [{vals_str}...]")
    elif node.shape is not None:
        info_parts.append(f"-> shape{node.shape}")

    # idの表示
    info_parts.append(f"(id=#{node.id})")

    # 既に訪問済みかチェック
    if node.id in visited:
        info_parts.append("[↑already shown]")
        result = f"{indent_str}{'│ ' if indent > 0 else ''}{' '.join(info_parts)}\n"
        return result

    visited.add(node.id)
    result = f"{indent_str}{'│ ' if indent > 0 else ''}{' '.join(info_parts)}\n"

    # 入力ノードを表示
    if node.input_node:
        for i, input_node in enumerate(node.input_node):
            is_last = i == len(node.input_node) - 1
            if is_last:
                result += f"{indent_str}└─ "
            else:
                result += f"{indent_str}├─ "
            # 再帰的に表示
            child_str = trace_graph(input_node, indent + 1, visited, show_values, names)
            # 最初の行を現在の行に追加
            lines = child_str.split("\n")
            if lines:
                first_line = lines[0].strip()
                result = result[:-1] + first_line + "\n"
                for line in lines[1:]:
                    if line:
                        result += line + "\n"

    return result


def show_graph(siarray_obj, show_values: bool = False):
    """
    siarrayオブジェクトの計算グラフを表示する便利関数

    Parameters:
    -----------
    siarray_obj : siarray
        表示するsiarrayオブジェクト
    show_values : bool
        ノードの出力値を表示するかどうか
    """
    print(f"=== Computation Graph for {siarray_obj.name} ===")
    print(trace_graph(siarray_obj.node, show_values=show_values))
    print(f"Final output shape: {siarray_obj.data.shape}")
    if siarray_obj.data.size <= 10:
        print(f"Final output value: {siarray_obj.data}")


class GraphToRational:
    """計算グラフをRationalFunctionに変換するクラス"""

    def __init__(
        self,
        root_node: Node,
        z: float | None = None,
        intervals: RealSubset | None = None,
    ):
        self.root_node = root_node
        self.z = z
        if intervals is None:
            self.intervals = RealSubset([[-float("inf"), float("inf")]])
        else:
            self.intervals = intervals
        self.visited = {}

    def convert(self) -> "RationalFunction":
        """ノードをRationalFunctionに変換"""
        return self._convert_node(self.root_node)

    def _convert_node(self, node: Node) -> "RationalFunction":
        """再帰的にノードを変換"""
        if id(node) in self.visited:
            return self.visited[id(node)]

        if node.op_type == OpType.INPUT or node.op_type == OpType.CONSTANT:
            # 入力または定数ノード
            # outputがあればそれを使用、なければvalueを使用
            data = node.output if node.output is not None else node.value
            result = RationalFunction(
                numerator=[data],
                denominator=[np.ones_like(data)],
                z=self.z,
                intervals=self.intervals,
            )

        elif node.op_type == OpType.ADD:
            left = self._convert_node(node.input_node[0])
            right = self._convert_node(node.input_node[1])
            result = left + right

        elif node.op_type == OpType.SUB:
            left = self._convert_node(node.input_node[0])
            right = self._convert_node(node.input_node[1])
            result = left - right

        elif node.op_type == OpType.MUL:
            left = self._convert_node(node.input_node[0])
            right = self._convert_node(node.input_node[1])
            result = left * right

        elif node.op_type == OpType.TRUEDIV:
            left = self._convert_node(node.input_node[0])
            right = self._convert_node(node.input_node[1])
            result = left / right

        elif node.op_type == OpType.MATMUL:
            left = self._convert_node(node.input_node[0])
            right = self._convert_node(node.input_node[1])
            result = left @ right

        elif node.op_type == OpType.NEG:
            operand = self._convert_node(node.input_node[0])
            # 負の値は-1を掛ける
            minus_one = RationalFunction(
                numerator=[-np.ones_like(operand.coef_num[0])],
                denominator=[np.ones_like(operand.coef_num[0])],
                z=self.z,
                intervals=self.intervals,
            )
            result = minus_one * operand

        elif node.op_type == OpType.TRANSPOSE:
            operand = self._convert_node(node.input_node[0])
            result = operand.T()

        elif node.op_type == OpType.INV:
            operand = self._convert_node(node.input_node[0])
            result = operand.inv()

        elif node.op_type == OpType.POW:
            base = self._convert_node(node.input_node[0])
            # outputがあればそれを使用、なければvalueを使用
            power_data = (
                node.input_node[1].output
                if node.input_node[1].output is not None
                else node.input_node[1].value
            )
            # 累乗の処理（整数乗のみサポート）
            if isinstance(power_data, (int, np.integer)) and power_data >= 0:
                result = base
                for _ in range(int(power_data) - 1):
                    result = result * base
            else:
                raise NotImplementedError(
                    f"Power {power_data} not supported for RationalFunction"
                )
        else:
            raise NotImplementedError(
                f"Operation {node.op_type} not yet supported for RationalFunction conversion"
            )

        self.visited[id(node)] = result
        return result
