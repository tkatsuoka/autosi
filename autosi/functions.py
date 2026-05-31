from __future__ import annotations

import builtins

import numpy as np

from .siarray import siarray
from .interval_tracker import IntervalTracker
from .poly_utils import calc_polynominal_intervals_batch


def _batch_order_constraints(
    tracker: IntervalTracker, diffs: list[siarray], z: float
) -> None:
    """差分 siarray のリストをバッチ処理して tracker に制約を追加する。

    _add_order_constraint を n 回呼ぶ代わりに calc_polynominal_intervals_batch で一括処理。
    """
    if not diffs:
        return
    d_num = builtins.max(len(d.coef_num) for d in diffs)
    d_den = builtins.max(len(d.coef_den) for d in diffs)
    n = len(diffs)
    nums = np.zeros((n, d_num))
    dens = np.zeros((n, d_den))
    for i, diff in enumerate(diffs):
        for k, c in enumerate(diff.coef_num):
            nums[i, k] = float(c)
        for k, c in enumerate(diff.coef_den):
            dens[i, k] = float(c)
    for interval in calc_polynominal_intervals_batch(nums, dens, z):
        tracker.update(interval)


def max(x, axis=None, keepdims=False):
    """最大値（siarray対応）"""
    if not isinstance(x, siarray):
        x = siarray(x)

    tracker = IntervalTracker.get_instance()
    z = tracker.get_z()

    if axis is None:
        flat_data = x.data.flatten()
        max_idx = int(np.argmax(flat_data))
        max_key = np.unravel_index(max_idx, x._shape)
        diffs = [
            x[max_key] - x[np.unravel_index(i, x._shape)]
            for i in range(len(flat_data)) if i != max_idx
        ]
        _batch_order_constraints(tracker, diffs, z)
        return x[max_key]

    if axis < 0:
        axis = len(x._shape) + axis
    max_indices = np.argmax(x.data, axis=axis)

    if keepdims:
        result_shape = list(x._shape)
        result_shape[axis] = 1
    else:
        result_shape = tuple(s for i, s in enumerate(x._shape) if i != axis)

    shape_init = result_shape if result_shape else (1,)
    result_num = [np.zeros(shape_init) for _ in x.coef_num]
    result_den = [np.zeros(shape_init) for _ in x.coef_den]

    for idx in np.ndindex(max_indices.shape):
        max_pos = max_indices[idx]
        src_idx = tuple(list(idx[:axis]) + [max_pos] + list(idx[axis:]))
        diffs = [
            x[src_idx] - x[tuple(list(idx[:axis]) + [i] + list(idx[axis:]))]
            for i in range(x._shape[axis]) if i != max_pos
        ]
        _batch_order_constraints(tracker, diffs, z)
        dst_idx = (
            idx if not keepdims
            else tuple(list(idx[:axis]) + [0] + list(idx[axis:]))
        )
        for deg in range(len(x.coef_num)):
            result_num[deg][dst_idx] = x.coef_num[deg][src_idx]
        for deg in range(len(x.coef_den)):
            result_den[deg][dst_idx] = x.coef_den[deg][src_idx]

    return siarray.from_rational(result_num, result_den)


def min(x, axis=None, keepdims=False):
    """最小値（siarray対応）"""
    if not isinstance(x, siarray):
        x = siarray(x)

    tracker = IntervalTracker.get_instance()
    z = tracker.get_z()

    if axis is None:
        flat_data = x.data.flatten()
        min_idx = int(np.argmin(flat_data))
        min_key = np.unravel_index(min_idx, x._shape)
        diffs = [
            x[np.unravel_index(i, x._shape)] - x[min_key]
            for i in range(len(flat_data)) if i != min_idx
        ]
        _batch_order_constraints(tracker, diffs, z)
        return x[min_key]

    if axis < 0:
        axis = len(x._shape) + axis
    min_indices = np.argmin(x.data, axis=axis)

    if keepdims:
        result_shape = list(x._shape)
        result_shape[axis] = 1
    else:
        result_shape = tuple(s for i, s in enumerate(x._shape) if i != axis)

    shape_init = result_shape if result_shape else (1,)
    result_num = [np.zeros(shape_init) for _ in x.coef_num]
    result_den = [np.zeros(shape_init) for _ in x.coef_den]

    for idx in np.ndindex(min_indices.shape):
        min_pos = min_indices[idx]
        src_idx = tuple(list(idx[:axis]) + [min_pos] + list(idx[axis:]))
        diffs = [
            x[tuple(list(idx[:axis]) + [i] + list(idx[axis:]))] - x[src_idx]
            for i in range(x._shape[axis]) if i != min_pos
        ]
        _batch_order_constraints(tracker, diffs, z)
        dst_idx = (
            idx if not keepdims
            else tuple(list(idx[:axis]) + [0] + list(idx[axis:]))
        )
        for deg in range(len(x.coef_num)):
            result_num[deg][dst_idx] = x.coef_num[deg][src_idx]
        for deg in range(len(x.coef_den)):
            result_den[deg][dst_idx] = x.coef_den[deg][src_idx]

    return siarray.from_rational(result_num, result_den)


def argmax(x, axis=None):
    """argmax（siarray対応）"""
    if not isinstance(x, siarray):
        x = siarray(x)

    tracker = IntervalTracker.get_instance()
    z = tracker.get_z()

    if axis is None:
        flat_data = x.data.flatten()
        max_idx = int(np.argmax(flat_data))
        max_key = np.unravel_index(max_idx, x._shape)
        diffs = [
            x[max_key] - x[np.unravel_index(i, x._shape)]
            for i in range(len(flat_data)) if i != max_idx
        ]
        _batch_order_constraints(tracker, diffs, z)
        return siarray(np.array(max_idx))

    if axis < 0:
        axis = len(x._shape) + axis
    max_indices = np.argmax(x.data, axis=axis)

    for idx in np.ndindex(max_indices.shape if max_indices.shape else (1,)):
        max_pos = int(max_indices[idx]) if max_indices.shape else int(max_indices)
        src_idx = tuple(list(idx[:axis]) + [max_pos] + list(idx[axis:]))
        diffs = [
            x[src_idx] - x[tuple(list(idx[:axis]) + [i] + list(idx[axis:]))]
            for i in range(x._shape[axis]) if i != max_pos
        ]
        _batch_order_constraints(tracker, diffs, z)

    return siarray(max_indices)


def mean(x, axis=None, keepdims=False):
    """平均（siarray対応）"""
    if not isinstance(x, siarray):
        x = siarray(x)
    return x.mean(axis=axis, keepdims=keepdims)


def sum(x, axis=None, keepdims=False):
    """合計（siarray対応）"""
    if not isinstance(x, siarray):
        x = siarray(x)
    return x.sum(axis=axis, keepdims=keepdims)


def abs(x):
    """絶対値（siarray対応）"""
    if not isinstance(x, siarray):
        x = siarray(x)
    return x.__abs__()


def argsort(x, axis=-1, kind=None, order=None):
    """ソートのインデックス（siarray対応）"""
    if not isinstance(x, siarray):
        x = siarray(x)
    return x.argsort(axis=axis, kind=kind, order=order)


def stack(arrays: list[siarray], axis: int = 0) -> siarray:
    """siarrayのスタック"""
    if not arrays:
        raise ValueError("arrays must not be empty")

    max_num_deg = builtins.max(len(arr.coef_num) for arr in arrays)
    max_den_deg = builtins.max(len(arr.coef_den) for arr in arrays)

    stacked_num = [
        np.stack(
            [
                arr.coef_num[deg] if deg < len(arr.coef_num)
                else np.zeros_like(arr.coef_num[0])
                for arr in arrays
            ],
            axis=axis,
        )
        for deg in range(max_num_deg)
    ]
    stacked_den = [
        np.stack(
            [
                arr.coef_den[deg] if deg < len(arr.coef_den)
                else np.ones_like(arr.coef_den[0])
                for arr in arrays
            ],
            axis=axis,
        )
        for deg in range(max_den_deg)
    ]

    return siarray.from_rational(stacked_num, stacked_den)


def array(X) -> siarray:
    """siarray を作成する便利関数"""
    return siarray(X)
