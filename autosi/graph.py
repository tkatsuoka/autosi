from __future__ import annotations

from .utils import Node, OpType, siarray


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
    power = (
        _get_node_name(node.input_node[1], names) if len(node.input_node) > 1 else "?"
    )
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
