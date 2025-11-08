"""
Refactored siarray module - 計算グラフを構築するNumPyラッパー

構成:
- OpType: 演算の種類を定義するEnum
- Node: 計算グラフのノード
- siarray: NumPy配列をラップし、計算グラフを構築するメインクラス
- GraphToRational: 計算グラフをRationalFunctionに変換
- ユーティリティ関数: trace_graph, show_graph, abs, argsort
"""

from __future__ import annotations

from collections import deque
from enum import Enum
import numpy as np
from sicore import RealSubset
from .rationalfunc import RationalFunction


# ============================================================================
# OpType: 演算の種類を定義
# ============================================================================

class OpType(Enum):
    """計算グラフでサポートされる演算の種類"""
    INPUT = "input"
    CONSTANT = "constant"
    # 二項演算
    ADD = "add"
    SUB = "sub"
    MUL = "mul"
    TRUEDIV = "truediv"
    MATMUL = "matmul"
    # 単項演算
    NEG = "neg"
    ABS = "abs"
    TRANSPOSE = "transpose"
    INV = "inv"
    # 集約演算
    SUM = "sum"
    MEAN = "mean"
    # 変形演算
    RESHAPE = "reshape"
    GETITEM = "getitem"
    SETITEM = "setitem"
    # その他
    POW = "pow"
    ARGSORT = "argsort"
    # 比較演算
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    EQ = "eq"
    NE = "ne"


# ============================================================================
# Node: 計算グラフのノード
# ============================================================================

class Node:
    """
    計算グラフのノード
    
    各ノードは一つの演算を表し、入力ノードと出力ノードへの参照を持つ。
    """
    
    _id_counter = 0

    def __init__(
        self,
        op_type: OpType,
        input_node: list["Node"] | None = None,
        value: np.ndarray | None = None,
        output: np.ndarray | None = None,
        output_node: list["Node"] | None = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        op_type : OpType
            このノードが表す演算の種類
        input_node : List[Node], optional
            入力となるノードのリスト
        value : np.ndarray, optional
            定数ノードの場合の値
        output : np.ndarray, optional
            このノードの計算結果
        output_node : List[Node], optional
            このノードの出力を使用するノードのリスト
        **kwargs
            演算に必要な追加パラメータ（axis, keepdims, key など）
        """
        self.id = Node._id_counter
        Node._id_counter += 1
        
        self.op_type = op_type
        self.input_node = input_node or []
        self.output_node = output_node or []
        self.value = value
        self.output = output
        self.kwargs = kwargs
        
        # shapeの設定
        self.shape = None
        if output is not None:
            self.shape = output.shape
        elif value is not None:
            self.shape = value.shape
        
        # 入力ノードに対して、このノードを出力先として登録
        for inp in self.input_node:
            if self not in inp.output_node:
                inp.output_node.append(self)

    def __repr__(self) -> str:
        return f"Node(id={self.id}, op={self.op_type.value}, shape={self.shape})"


# ============================================================================
# siarray: メインクラス
# ============================================================================

class siarray:
    """
    NumPy配列をラップし、演算の計算グラフを自動的に構築するクラス
    
    このクラスは通常のNumPy演算をサポートしつつ、内部で計算グラフを構築する。
    構築されたグラフは後でRationalFunctionに変換できる。
    """

    def __init__(
        self, 
        X: np.ndarray | float | int | "siarray", 
        name: str | None = None
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
            self.node = X.node
            self.name = name or X.name
        else:
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            self.data = X
            self.name = name or f"input_{Node._id_counter}"
            self.node = Node(OpType.INPUT, value=X, output=X)

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
        obj.name = f"const_{Node._id_counter}"
        obj.node = Node(OpType.CONSTANT, value=value, output=value)
        return obj

    # ------------------------------------------------------------------------
    # ヘルパーメソッド
    # ------------------------------------------------------------------------

    def _to_siarray(self, other) -> "siarray":
        """他の値をsiarrayオブジェクトに変換"""
        if not isinstance(other, siarray):
            return siarray.constant(other)
        return other

    def _create_binary_op(
        self, 
        other, 
        op_type: OpType, 
        numpy_op,
        symbol: str
    ) -> "siarray":
        """二項演算の共通処理"""
        other = self._to_siarray(other)
        result_data = numpy_op(self.data, other.data)
        
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"({self.name}{symbol}{other.name})"
        result.node = Node(
            op_type, 
            input_node=[self.node, other.node], 
            output=result_data
        )
        return result

    def _create_unary_op(
        self, 
        op_type: OpType, 
        numpy_op,
        name_template: str
    ) -> "siarray":
        """単項演算の共通処理"""
        result_data = numpy_op(self.data)
        
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = name_template.format(name=self.name)
        result.node = Node(op_type, input_node=[self.node], output=result_data)
        return result

    # ------------------------------------------------------------------------
    # 算術演算子
    # ------------------------------------------------------------------------

    def __add__(self, other):
        return self._create_binary_op(other, OpType.ADD, np.add, "+")

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._create_binary_op(other, OpType.SUB, np.subtract, "-")

    def __rsub__(self, other):
        return self._to_siarray(other).__sub__(self)

    def __mul__(self, other):
        return self._create_binary_op(other, OpType.MUL, np.multiply, "*")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._create_binary_op(other, OpType.TRUEDIV, np.true_divide, "/")

    def __rtruediv__(self, other):
        return self._to_siarray(other).__truediv__(self)

    def __matmul__(self, other):
        return self._create_binary_op(other, OpType.MATMUL, np.matmul, "@")

    def __rmatmul__(self, other):
        return self._to_siarray(other).__matmul__(self)

    def __neg__(self):
        return self._create_unary_op(OpType.NEG, np.negative, "(-{name})")

    def __abs__(self):
        return self._create_unary_op(OpType.ABS, np.abs, "abs({name})")

    def __pow__(self, power):
        """累乗演算"""
        if isinstance(power, siarray):
            power_val = power.data
            power_node = power.node
        else:
            power_val = power
            power_node = Node(OpType.CONSTANT, value=np.array(power), output=np.array(power))

        result_data = self.data ** power_val
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"({self.name}**{power_val})"
        result.node = Node(OpType.POW, input_node=[self.node, power_node], output=result_data)
        return result

    # ------------------------------------------------------------------------
    # 比較演算子
    # ------------------------------------------------------------------------

    def __lt__(self, other):
        return self._create_binary_op(other, OpType.LT, np.less, "<")

    def __le__(self, other):
        return self._create_binary_op(other, OpType.LE, np.less_equal, "<=")

    def __gt__(self, other):
        return self._create_binary_op(other, OpType.GT, np.greater, ">")

    def __ge__(self, other):
        return self._create_binary_op(other, OpType.GE, np.greater_equal, ">=")

    def __eq__(self, other):
        return self._create_binary_op(other, OpType.EQ, np.equal, "==")

    def __ne__(self, other):
        return self._create_binary_op(other, OpType.NE, np.not_equal, "!=")

    # ------------------------------------------------------------------------
    # インデックス演算
    # ------------------------------------------------------------------------

    def __getitem__(self, key):
        """要素の取得"""
        # keyがsiarrayの場合
        if isinstance(key, siarray):
            actual_key = key.data
            if actual_key.ndim == 0 or (actual_key.ndim == 1 and actual_key.size == 1):
                actual_key = actual_key.item()
            
            result_data = self.data[actual_key]
            result = siarray.__new__(siarray)
            result.data = result_data
            result.name = f"{self.name}[{key.name}]"
            result.node = Node(
                OpType.GETITEM,
                input_node=[self.node, key.node],
                key=actual_key,
                output=result_data
            )
        else:
            result_data = self.data[key]
            result = siarray.__new__(siarray)
            result.data = result_data
            result.name = f"{self.name}[{key}]"
            result.node = Node(
                OpType.GETITEM, 
                input_node=[self.node], 
                key=key, 
                output=result_data
            )
        return result

    def __setitem__(self, key, value):
        """要素の設定（破壊的操作）"""
        value_siarray = value if isinstance(value, siarray) else siarray.constant(value)
        
        new_data = self.data.copy()
        new_data[key] = value_siarray.data
        
        new_node = Node(
            OpType.SETITEM,
            input_node=[self.node, value_siarray.node],
            output=new_data,
            key=key
        )
        
        self.data = new_data
        self.node = new_node
        self.name = f"{self.name}[{key}]=updated"

    # ------------------------------------------------------------------------
    # 配列操作メソッド
    # ------------------------------------------------------------------------

    @property
    def T(self):
        """転置"""
        return self._create_unary_op(OpType.TRANSPOSE, lambda x: x.T, "{name}.T")

    def transpose(self, *axes):
        """転置（軸指定可能）"""
        if axes:
            result_data = np.transpose(self.data, axes)
        else:
            result_data = self.data.T
        
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"{self.name}.transpose({axes})"
        result.node = Node(
            OpType.TRANSPOSE, 
            input_node=[self.node], 
            axes=axes, 
            output=result_data
        )
        return result

    def inv(self):
        """逆行列"""
        result_data = np.linalg.inv(self.data)
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"{self.name}.inv()"
        result.node = Node(OpType.INV, input_node=[self.node], output=result_data)
        return result

    def reshape(self, *shape):
        """形状変更"""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        
        result_data = self.data.reshape(shape)
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"{self.name}.reshape({shape})"
        result.node = Node(
            OpType.RESHAPE, 
            input_node=[self.node], 
            shape=shape, 
            output=result_data
        )
        return result

    def argsort(self, axis=-1, kind=None, order=None):
        """ソートのインデックスを返す"""
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
            output=result_data
        )
        return result

    # ------------------------------------------------------------------------
    # 集約メソッド
    # ------------------------------------------------------------------------

    def sum(self, axis=None, keepdims=False):
        """合計"""
        result_data = np.sum(self.data, axis=axis, keepdims=keepdims)
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"{self.name}.sum(axis={axis})"
        result.node = Node(
            OpType.SUM,
            input_node=[self.node],
            axis=axis,
            keepdims=keepdims,
            output=result_data
        )
        return result

    def mean(self, axis=None, keepdims=False):
        """平均"""
        result_data = np.mean(self.data, axis=axis, keepdims=keepdims)
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"{self.name}.mean(axis={axis})"
        result.node = Node(
            OpType.MEAN,
            input_node=[self.node],
            axis=axis,
            keepdims=keepdims,
            output=result_data
        )
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
    # グラフ構築・解析メソッド
    # ------------------------------------------------------------------------

    def add_prob_vec_nodes(self, prob_vec: "siarray"):
        """確率変数ベクトルのノードを保存"""
        self.prob_vec_nodes = prob_vec.node

    def add_observed_model(self, model: "siarray"):
        """観測されたモデルのノードを保存"""
        self.observed_model = model

    def _topological_sort(self, node: Node | None = None) -> list[Node]:
        """DFSによるトポロジカルソート"""
        if node is None:
            node = self.node

        visited = set()
        result = []

        def dfs(n: Node):
            if n.id in visited:
                return
            visited.add(n.id)
            for input_node in n.input_node:
                dfs(input_node)
            result.append(n)

        dfs(node)
        return result

    def _topological_sort_kahn(self, node: Node | None = None) -> list[Node]:
        """Kahnのアルゴリズムによるトポロジカルソート（循環検出付き）"""
        if node is None:
            node = self.node

        all_nodes = self._collect_all_nodes(node)
        in_degree = {n.id: 0 for n in all_nodes}

        for n in all_nodes:
            for output in n.output_node:
                if output.id in in_degree:
                    in_degree[output.id] += 1

        queue = deque([n for n in all_nodes if in_degree[n.id] == 0])
        result = []

        while queue:
            current = queue.popleft()
            result.append(current)
            for output in current.output_node:
                if output.id in in_degree:
                    in_degree[output.id] -= 1
                    if in_degree[output.id] == 0:
                        queue.append(output)

        if len(result) != len(all_nodes):
            raise ValueError("計算グラフに循環が検出されました")

        return result

    def _collect_all_nodes(self, node: Node) -> set[Node]:
        """ノードから到達可能な全ノードを収集"""
        visited = set()
        to_visit = [node]

        while to_visit:
            current = to_visit.pop()
            if current.id not in visited:
                visited.add(current.id)
                to_visit.extend(current.input_node)

        return {n for n in [node] + node.input_node if n.id in visited}

    # ------------------------------------------------------------------------
    # RationalFunction変換・評価
    # ------------------------------------------------------------------------
    def  add_algorithm(self, algorithm: callable):
        """選択アルゴリズム関数を保存"""
        self.algorithm_func = algorithm

    def calc_intervals(self, a: np.ndarray, b: np.ndarray, z: float):
        """
        計算グラフをRationalFunctionに変換し、多項式係数a,bで評価
        
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
        # 確率変数ベクトルをsiarrayとして構築 (a + b*z の値)
        prob_vec_values = a + b * z
        prob_vec_siarray = siarray(prob_vec_values.reshape(self.prob_vec_nodes.shape))
        
        # アルゴリズムを実行して計算グラフを構築
        # algorithm_func(prob_vec) -> (eta, idx) などを返す
        output = self.algorithm_func(prob_vec_siarray)
        
        # 計算グラフからintervalsを抽出
        intervals = self._extract_intervals_from_graph(output, a, b, z)
        
        # observed_modelが設定されている場合はそれを使用、なければoutput_siarrayを使用
        if hasattr(self, 'observed_model') and self.observed_model is not None:
            result_value = self.observed_model.data
        else:
            result_value = output.data
        
        return result_value, intervals


    def _extract_intervals_from_graph(self, output_siarray, a, b, z):
        """
        siarrayの計算グラフからintervalsを抽出
        
        Parameters
        ----------
        output_siarray : siarray
            出力のsiarrayオブジェクト
        a : np.ndarray
            多項式の定数項係数
        b : np.ndarray
            多項式の1次項係数
        z : float
            評価点
        
        Returns
        -------
        intervals : RealSubset
            有効な区間
        """
        # 初期化: 全区間を有効とする
        intervals = RealSubset([[-float("inf"), float("inf")]])
        
        # 計算グラフをトポロジカル順序でソート
        sorted_nodes = output_siarray._topological_sort()
        
        # ノードごとの計算結果を保存する辞書
        node_values = {}
        
        # 確率変数ノードにRationalFunctionを設定 (a + b*z)
        a_reshaped = a.reshape(self.prob_vec_nodes.shape)
        b_reshaped = b.reshape(self.prob_vec_nodes.shape)
        node_values[self.prob_vec_nodes.id] = RationalFunction(
            numerator=[a_reshaped, b_reshaped],
            denominator=[np.ones_like(a_reshaped)],
            z=z,
        )
        
        # 入力ノードを定数として設定
        input_nodes = [n for n in sorted_nodes if n.op_type == OpType.INPUT]
        for node in input_nodes:
            if node.id == self.prob_vec_nodes.id:
                continue
            data = node.value if node.value is not None else node.output
            node_values[node.id] = RationalFunction(
                numerator=[data],
                denominator=[np.ones_like(data)],
                z=z,
            )
        
        # トポロジカル順序で各ノードを計算
        for node in sorted_nodes:
            if node.id in node_values:
                continue
            # ノードを計算
            result = self._forward_si(node, node_values, z)
            node_values[node.id] = result
            
            # 有効な区間を更新
            intervals = intervals & result.intervals
        
        return intervals

    def algorithm(self, a: np.ndarray, b: np.ndarray, z: float) -> tuple[np.ndarray, RealSubset]:
        """
        計算グラフをRationalFunctionに変換し、多項式係数a,bで評価
        
        Returns
        -------
        output : np.ndarray
            評価結果
        intervals : RealSubset
            有効な区間
        """

        

        self.intervals = RealSubset([[-float("inf"), float("inf")]])
        sorted_nodes = self._topological_sort()
        
        input_nodes = [n for n in sorted_nodes if n.op_type == OpType.INPUT]
        if len(input_nodes) == 0:
            raise ValueError("No input nodes found in the computation graph")

        node_values = {}
        
        # 入力ノードを定数として設定
        for node in input_nodes:
            data = node.value if node.value is not None else node.output
            node_values[node.id] = RationalFunction(
                numerator=[data],
                denominator=[np.ones_like(data)],
                z=z,
            )

        # 確率変数ノードに多項式を設定
        a = a.reshape(self.prob_vec_nodes.shape)
        b = b.reshape(self.prob_vec_nodes.shape)
        node_values[self.prob_vec_nodes.id] = RationalFunction(
            numerator=[a, b],
            denominator=[np.ones_like(a)],
            z=z,
        )

        # トポロジカル順序で計算
        for node in sorted_nodes:
            if node.id in node_values:
                continue
            
            result = self._forward_si(node, node_values, z)
            node_values[node.id] = result
            self.intervals = self.intervals & result.intervals

        return node_values[self.observed_model.node.id].output(), self.intervals

    def _forward_si(self, node: Node, node_values: dict, z: float) -> RationalFunction:
        """単一ノードの計算を実行（RationalFunction版）"""
        if node.op_type == OpType.CONSTANT:
            data = node.value if node.value is not None else node.output
            return RationalFunction(
                numerator=[data],
                denominator=[np.ones_like(data)],
                z=z,
            )

        input_rfs = [node_values[n.id] for n in node.input_node]

        # 演算の実行
        op_handlers = {
            OpType.ADD: lambda: input_rfs[0] + input_rfs[1],
            OpType.SUB: lambda: input_rfs[0] - input_rfs[1],
            OpType.MUL: lambda: input_rfs[0] * input_rfs[1],
            OpType.TRUEDIV: lambda: input_rfs[0] / input_rfs[1],
            OpType.MATMUL: lambda: input_rfs[0] @ input_rfs[1],
            OpType.NEG: lambda: RationalFunction(
                numerator=[-np.ones_like(input_rfs[0].coef_num[0])],
                denominator=[np.ones_like(input_rfs[0].coef_num[0])],
                z=z,
            ) * input_rfs[0],
            OpType.TRANSPOSE: lambda: input_rfs[0].T(),
            OpType.INV: lambda: input_rfs[0].inv(),
            OpType.GETITEM: lambda: input_rfs[0][node.kwargs.get("key")],
            OpType.ABS: lambda: input_rfs[0].abs(z),
            OpType.ARGSORT: lambda: input_rfs[0].argsort(
                axis=node.kwargs.get("axis", -1),
                kind=node.kwargs.get("kind", None),
                order=node.kwargs.get("order", None)
            ),
            OpType.RESHAPE: lambda: input_rfs[0],
            OpType.LT: lambda: input_rfs[0] < input_rfs[1],
            OpType.LE: lambda: input_rfs[0] <= input_rfs[1],
            OpType.GT: lambda: input_rfs[0] > input_rfs[1],
            OpType.GE: lambda: input_rfs[0] >= input_rfs[1],
        }

        if node.op_type in op_handlers:
            return op_handlers[node.op_type]()
        elif node.op_type == OpType.POW:
            return self._handle_pow(node, input_rfs, z)
        elif node.op_type == OpType.SETITEM:
            return self._handle_setitem(node, input_rfs, z)
        else:
            raise NotImplementedError(f"Operation {node.op_type} not implemented")

    def _handle_pow(self, node: Node, input_rfs: list, z: float) -> RationalFunction:
        """累乗演算の処理"""
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
            if power_node and (power_node.value is not None or power_node.output is not None):
                power_val = power_node.value if power_node.value is not None else power_node.output
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

    def _handle_setitem(self, node: Node, input_rfs: list, z: float) -> RationalFunction:
        """SETITEM演算の処理"""
        key = node.kwargs.get("key")
        base_rf = input_rfs[0]
        value_rf = input_rfs[1]
        
        new_coef_num = [c.copy() for c in base_rf.coef_num]
        new_coef_den = [c.copy() for c in base_rf.coef_den]
        
        for i in range(len(new_coef_num)):
            if i < len(value_rf.coef_num):
                new_coef_num[i][key] = value_rf.coef_num[i]
            else:
                new_coef_num[i][key] = 0
        
        for i in range(len(new_coef_den)):
            if i < len(value_rf.coef_den):
                new_coef_den[i][key] = value_rf.coef_den[i]
            else:
                new_coef_den[i][key] = 1
        
        combined_intervals = base_rf.intervals & value_rf.intervals
        return RationalFunction(new_coef_num, new_coef_den, z, combined_intervals)

    def to_rational_function(
        self, 
        z: float | None = None, 
        intervals: RealSubset | None = None
    ) -> RationalFunction:
        """計算グラフをRationalFunctionに変換"""
        if intervals is None:
            intervals = RealSubset([[-float("inf"), float("inf")]])
        converter = GraphToRational(self.node, z, intervals)
        return converter.convert()

    # ------------------------------------------------------------------------
    # ユーティリティメソッド
    # ------------------------------------------------------------------------

    def model_selector(self, searched_model: np.ndarray | "siarray") -> bool:
        """2つのモデルが同じかどうかを判定"""
        data1 = self.observed_model.data if isinstance(self.observed_model, siarray) else np.asarray(self.observed_model)
        data2 = searched_model.data if isinstance(searched_model, siarray) else np.asarray(searched_model)
        comparison = np.allclose(data1, data2)
        # print("debug", data1, data2, comparison)
        return comparison

    def to_numpy(self):
        """NumPy配列を返す"""
        return self.data.copy()
    
    def copy(self):
        """siarrayオブジェクトのコピーを作成"""
        copied = siarray.__new__(siarray)
        copied.data = self.data.copy()
        copied.name = self.name
        copied.node = self.node
        return copied

    def values(self):
        """データの値を返す"""
        return self.data

    def __repr__(self):
        return f"siarray({self.name}, shape={self.data.shape}, op={self.node.op_type.value})"

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
                print(f"First {max_rows // 2}:", self.data[:max_rows // 2])
                print(f"Last {max_rows // 2}:", self.data[-max_rows // 2:])
        elif self.data.ndim == 2:
            rows, cols = self.data.shape
            if rows <= max_rows and cols <= max_cols:
                print("Values:\n", self.data)
            else:
                print(f"Top-left ({min(max_rows // 2, rows)} x {min(max_cols // 2, cols)}):")
                print(self.data[:min(max_rows // 2, rows), :min(max_cols // 2, cols)])
                if rows > max_rows or cols > max_cols:
                    print("...")
        else:
            print(f"Array with shape {self.data.shape}")
            print("First elements:", self.data.flat[:min(20, self.data.size)])


# ============================================================================
# GraphToRational: 計算グラフをRationalFunctionに変換
# ============================================================================

class GraphToRational:
    """計算グラフをRationalFunctionに変換するクラス"""

    def __init__(
        self,
        root_node: Node,
        z: float | None = None,
        intervals: RealSubset | None = None,
    ):
        """
        Parameters
        ----------
        root_node : Node
            変換の起点となるノード
        z : float, optional
            RationalFunctionのパラメータ
        intervals : RealSubset, optional
            有効な区間
        """
        self.root_node = root_node
        self.z = z
        self.intervals = intervals if intervals is not None else RealSubset([[-float("inf"), float("inf")]])
        self.visited = {}

    def convert(self) -> RationalFunction:
        """ノードをRationalFunctionに変換"""
        return self._convert_node(self.root_node)

    def _convert_node(self, node: Node) -> RationalFunction:
        """再帰的にノードをRationalFunctionに変換"""
        if id(node) in self.visited:
            return self.visited[id(node)]

        # 入力・定数ノード
        if node.op_type in (OpType.INPUT, OpType.CONSTANT):
            result = self._create_constant_rf(node)
        # 二項演算
        elif node.op_type == OpType.ADD:
            result = self._convert_binary_op(node, lambda l, r: l + r)
        elif node.op_type == OpType.SUB:
            result = self._convert_binary_op(node, lambda l, r: l - r)
        elif node.op_type == OpType.MUL:
            result = self._convert_binary_op(node, lambda l, r: l * r)
        elif node.op_type == OpType.TRUEDIV:
            result = self._convert_binary_op(node, lambda l, r: l / r)
        elif node.op_type == OpType.MATMUL:
            result = self._convert_binary_op(node, lambda l, r: l @ r)
        # 単項演算
        elif node.op_type == OpType.NEG:
            result = self._convert_neg(node)
        elif node.op_type == OpType.TRANSPOSE:
            result = self._convert_node(node.input_node[0]).T()
        elif node.op_type == OpType.INV:
            result = self._convert_node(node.input_node[0]).inv()
        elif node.op_type == OpType.POW:
            result = self._convert_pow(node)
        else:
            raise NotImplementedError(
                f"Operation {node.op_type} not yet supported for RationalFunction conversion"
            )

        self.visited[id(node)] = result
        return result

    def _create_constant_rf(self, node: Node) -> RationalFunction:
        """定数ノードからRationalFunctionを作成"""
        data = node.output if node.output is not None else node.value
        return RationalFunction(
            numerator=[data],
            denominator=[np.ones_like(data)],
            z=self.z,
            intervals=self.intervals,
        )

    def _convert_binary_op(self, node: Node, op) -> RationalFunction:
        """二項演算を変換"""
        left = self._convert_node(node.input_node[0])
        right = self._convert_node(node.input_node[1])
        return op(left, right)

    def _convert_neg(self, node: Node) -> RationalFunction:
        """負の演算を変換"""
        operand = self._convert_node(node.input_node[0])
        minus_one = RationalFunction(
            numerator=[-np.ones_like(operand.coef_num[0])],
            denominator=[np.ones_like(operand.coef_num[0])],
            z=self.z,
            intervals=self.intervals,
        )
        return minus_one * operand

    def _convert_pow(self, node: Node) -> RationalFunction:
        """累乗演算を変換"""
        base = self._convert_node(node.input_node[0])
        power_data = (
            node.input_node[1].output
            if node.input_node[1].output is not None
            else node.input_node[1].value
        )
        
        if isinstance(power_data, (int, np.integer)) and power_data >= 0:
            result = base
            for _ in range(int(power_data) - 1):
                result = result * base
            return result
        else:
            raise NotImplementedError(
                f"Power {power_data} not supported for RationalFunction"
            )


# ============================================================================
# ユーティリティ関数
# ============================================================================

def abs(x):
    """絶対値（siarray対応）"""
    if isinstance(x, siarray):
        return x.__abs__()
    else:
        return siarray(x).__abs__()

def sum(self, axis=None, keepdims=False):
    """合計"""
    result_data = np.sum(self.data, axis=axis, keepdims=keepdims)
    if isinstance(self, siarray):
        result = siarray.__new__(siarray)
        result.data = result_data
        result.name = f"{self.name}.sum(axis={axis})"
        result.node = Node(
            OpType.SUM,
            input_node=[self.node],
            axis=axis,
            keepdims=keepdims,
            output=result_data
        )
        return result

def argsort(x, axis=-1, kind=None, order=None):
    """ソートのインデックス（siarray対応）"""
    if isinstance(x, siarray):
        return x.argsort(axis=axis, kind=kind, order=order)
    else:
        return siarray(x).argsort(axis=axis, kind=kind, order=order)


def trace_graph(
    node: Node,
    indent: int = 0,
    visited: set | None = None,
    show_values: bool = False,
    names: dict | None = None,
) -> str:
    """
    計算グラフを上から下（根から葉）に向かって表示
    
    Parameters
    ----------
    node : Node
        表示するノード
    indent : int
        インデントレベル
    visited : Set, optional
        既に訪問したノードのIDセット
    show_values : bool
        出力値の概要を表示するかどうか
    names : Dict, optional
        ノードIDから変数名へのマッピング
    
    Returns
    -------
    str
        グラフの文字列表現
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
                names[node.id] = str(node.output.item())
            else:
                names[node.id] = f"const{node.id}"

    indent_str = "  " * indent
    display = _format_node_display(node, names)

    # 情報を構築
    info_parts = [display]

    # 出力の形状と値
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
            
            child_str = trace_graph(input_node, indent + 1, visited, show_values, names)
            lines = child_str.split("\n")
            if lines:
                first_line = lines[0].strip()
                result = result[:-1] + first_line + "\n"
                for line in lines[1:]:
                    if line:
                        result += line + "\n"

    return result


def _format_node_display(node: Node, names: dict) -> str:
    """ノードの表示形式をフォーマット"""
    op_formatters = {
        OpType.INPUT: lambda: names[node.id],
        OpType.CONSTANT: lambda: names[node.id],
        OpType.ADD: lambda: _format_binary_op(node, names, "+"),
        OpType.SUB: lambda: _format_binary_op(node, names, "-"),
        OpType.MUL: lambda: _format_binary_op(node, names, "*"),
        OpType.TRUEDIV: lambda: _format_binary_op(node, names, "/"),
        OpType.MATMUL: lambda: _format_binary_op(node, names, "@"),
        OpType.NEG: lambda: f"(-{_get_node_name(node.input_node[0], names)})",
        OpType.TRANSPOSE: lambda: f"{_get_node_name(node.input_node[0], names)}.T",
        OpType.INV: lambda: f"inv({_get_node_name(node.input_node[0], names)})",
        OpType.ABS: lambda: f"abs({_get_node_name(node.input_node[0], names)})",
        OpType.POW: lambda: _format_pow(node, names),
        OpType.ARGSORT: lambda: f"{_get_node_name(node.input_node[0], names)}.argsort(axis={node.kwargs.get('axis', -1)})",
        OpType.SUM: lambda: f"{_get_node_name(node.input_node[0], names)}.sum(axis={node.kwargs.get('axis', None)})",
        OpType.MEAN: lambda: f"{_get_node_name(node.input_node[0], names)}.mean(axis={node.kwargs.get('axis', None)})",
        OpType.RESHAPE: lambda: f"{_get_node_name(node.input_node[0], names)}.reshape{node.kwargs.get('shape', '?')}",
        OpType.GETITEM: lambda: _format_getitem(node, names),
    }

    if node.op_type in op_formatters:
        display = op_formatters[node.op_type]()
        names[node.id] = display
        return display
    else:
        return node.op_type.value


def _get_node_name(node: Node, names: dict) -> str:
    """ノードの名前を取得"""
    return names.get(node.id, f"#{node.id}")


def _format_binary_op(node: Node, names: dict, symbol: str) -> str:
    """二項演算のフォーマット"""
    left = _get_node_name(node.input_node[0], names)
    right = _get_node_name(node.input_node[1], names)
    return f"({left} {symbol} {right})"


def _format_pow(node: Node, names: dict) -> str:
    """累乗演算のフォーマット"""
    base = _get_node_name(node.input_node[0], names)
    power = _get_node_name(node.input_node[1], names) if len(node.input_node) > 1 else "?"
    return f"({base}**{power})"


def _format_getitem(node: Node, names: dict) -> str:
    """インデックス演算のフォーマット"""
    operand = _get_node_name(node.input_node[0], names)
    key = node.kwargs.get("key", "?")
    
    if isinstance(key, tuple):
        key_str = str(key)
    elif isinstance(key, slice):
        key_str = f"{key.start}:{key.stop}" + (f":{key.step}" if key.step else "")
    else:
        key_str = str(key)
    
    return f"{operand}[{key_str}]"


def show_graph(siarray_obj: siarray, show_values: bool = False):
    """
    siarrayオブジェクトの計算グラフを表示
    
    Parameters
    ----------
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