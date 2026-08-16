from __future__ import annotations

import builtins

import numpy as np

from .siarray import siarray
from .constraints import add_order_constraints


def _selection_diffs(x, sel_key, other_keys, want_min):
    """Build the list of difference siarrays between the selected and other elements.

    ``max`` / ``argmax`` impose ``selected - other > 0``; ``min`` imposes
    ``other - selected > 0``.

    Parameters
    ----------
    x : siarray
        Source array.
    sel_key : tuple
        Index of the selected element.
    other_keys : list of tuple
        Indices of the other elements.
    want_min : bool
        Whether the selection is a minimum.

    Returns
    -------
    list of siarray
        Scalar-valued difference rational functions.
    """
    if want_min:
        return [x[k] - x[sel_key] for k in other_keys]
    return [x[sel_key] - x[k] for k in other_keys]


def _reduce_select(x, axis, keepdims, *, want_min, return_index):
    """Shared implementation of ``max`` / ``min`` / ``argmax``.

    Tracks the interval over which the selection (max or min) result is unchanged,
    then returns the selected index when ``return_index`` is True, or the selected
    value otherwise.

    Parameters
    ----------
    x : siarray or array-like
        Input array.
    axis : int or None
        Axis to reduce over. ``None`` reduces over all elements.
    keepdims : bool
        Whether to keep the reduced axis as size 1 (value path only).
    want_min : bool
        Select the minimum instead of the maximum.
    return_index : bool
        Return the selected index instead of the selected value.

    Returns
    -------
    siarray
        Selected value(s) or index/indices.
    """
    if not isinstance(x, siarray):
        x = siarray(x)

    argext = np.argmin if want_min else np.argmax

    # --- Reduce over all elements ---
    if axis is None:
        flat_data = x.data.flatten()
        sel_idx = int(argext(flat_data))
        sel_key = np.unravel_index(sel_idx, x.shape)
        other_keys = [
            np.unravel_index(i, x.shape)
            for i in range(len(flat_data)) if i != sel_idx
        ]
        add_order_constraints(_selection_diffs(x, sel_key, other_keys, want_min))
        return siarray(np.array(sel_idx)) if return_index else x[sel_key]

    # --- Reduce along a specific axis ---
    if axis < 0:
        axis = len(x.shape) + axis
    sel_indices = argext(x.data, axis=axis)

    if return_index:
        # Only track constraints and return indices (no value gathering needed)
        for idx in np.ndindex(sel_indices.shape if sel_indices.shape else (1,)):
            sel_pos = int(sel_indices[idx]) if sel_indices.shape else int(sel_indices)
            sel_key = tuple(list(idx[:axis]) + [sel_pos] + list(idx[axis:]))
            other_keys = [
                tuple(list(idx[:axis]) + [i] + list(idx[axis:]))
                for i in range(x.shape[axis]) if i != sel_pos
            ]
            add_order_constraints(_selection_diffs(x, sel_key, other_keys, want_min))
        return siarray(sel_indices)

    # Return values: track constraints and gather the selected coefficients
    if keepdims:
        result_shape = list(x.shape)
        result_shape[axis] = 1
    else:
        result_shape = tuple(s for i, s in enumerate(x.shape) if i != axis)

    shape_init = result_shape if result_shape else (1,)
    result_num = [np.zeros(shape_init) for _ in x.coef_num]
    result_den = [np.zeros(shape_init) for _ in x.coef_den]

    for idx in np.ndindex(sel_indices.shape):
        sel_pos = sel_indices[idx]
        sel_key = tuple(list(idx[:axis]) + [sel_pos] + list(idx[axis:]))
        other_keys = [
            tuple(list(idx[:axis]) + [i] + list(idx[axis:]))
            for i in range(x.shape[axis]) if i != sel_pos
        ]
        add_order_constraints(_selection_diffs(x, sel_key, other_keys, want_min))
        dst_idx = (
            idx if not keepdims
            else tuple(list(idx[:axis]) + [0] + list(idx[axis:]))
        )
        for deg in range(len(x.coef_num)):
            result_num[deg][dst_idx] = x.coef_num[deg][sel_key]
        for deg in range(len(x.coef_den)):
            result_den[deg][dst_idx] = x.coef_den[deg][sel_key]

    return siarray.from_rational(result_num, result_den)


def max(x, axis=None, keepdims=False):
    """Maximum (siarray-aware)."""
    return _reduce_select(x, axis, keepdims, want_min=False, return_index=False)


def min(x, axis=None, keepdims=False):
    """Minimum (siarray-aware)."""
    return _reduce_select(x, axis, keepdims, want_min=True, return_index=False)


def argmax(x, axis=None):
    """Index of the maximum (siarray-aware)."""
    return _reduce_select(x, axis, keepdims=False, want_min=False, return_index=True)


def argmin(x, axis=None):
    """Index of the minimum (siarray-aware)."""
    return _reduce_select(x, axis, keepdims=False, want_min=True, return_index=True)


def mean(x, axis=None, keepdims=False):
    """Mean (siarray-aware)."""
    if not isinstance(x, siarray):
        x = siarray(x)
    return x.mean(axis=axis, keepdims=keepdims)


def sum(x, axis=None, keepdims=False):
    """Sum (siarray-aware)."""
    if not isinstance(x, siarray):
        x = siarray(x)
    return x.sum(axis=axis, keepdims=keepdims)


def var(x, axis=None, keepdims=False, ddof=0):
    """Variance (siarray-aware)."""
    if not isinstance(x, siarray):
        x = siarray(x)
    return x.var(axis=axis, keepdims=keepdims, ddof=ddof)


def prod(x, axis=None, keepdims=False):
    """Product (siarray-aware)."""
    if not isinstance(x, siarray):
        x = siarray(x)
    return x.prod(axis=axis, keepdims=keepdims)


def sort(x, axis=-1, kind=None, order=None):
    """Sorted values (siarray-aware)."""
    if not isinstance(x, siarray):
        x = siarray(x)
    return x.sort(axis=axis, kind=kind, order=order)


def flatten(x):
    """1-D copy of the array (siarray-aware)."""
    if not isinstance(x, siarray):
        x = siarray(x)
    return x.flatten()


def abs(x):
    """Absolute value (siarray-aware)."""
    if not isinstance(x, siarray):
        x = siarray(x)
    return x.__abs__()


def argsort(x, axis=-1, kind=None, order=None):
    """Sort indices (siarray-aware)."""
    if not isinstance(x, siarray):
        x = siarray(x)
    return x.argsort(axis=axis, kind=kind, order=order)


def stack(arrays: list[siarray], axis: int = 0) -> siarray:
    """Stack siarrays along a new axis.

    Coefficient lists of differing length are zero-padded (numerators) or one-padded
    (denominators) before stacking.

    Parameters
    ----------
    arrays : list of siarray
        Arrays to stack (must be non-empty).
    axis : int, optional
        Axis along which to stack (default 0).

    Returns
    -------
    siarray
        The stacked array.
    """
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


def concatenate(arrays: list[siarray], axis: int = 0) -> siarray:
    """Join siarrays along an existing axis.

    Coefficient lists of differing length are zero-padded (numerators) or one-padded
    (denominators) before concatenation.

    Parameters
    ----------
    arrays : list of siarray
        Arrays to concatenate (must be non-empty).
    axis : int, optional
        Axis along which to concatenate (default 0).

    Returns
    -------
    siarray
        The concatenated array.
    """
    if not arrays:
        raise ValueError("arrays must not be empty")

    max_num_deg = builtins.max(len(arr.coef_num) for arr in arrays)
    max_den_deg = builtins.max(len(arr.coef_den) for arr in arrays)

    cat_num = [
        np.concatenate(
            [
                arr.coef_num[deg] if deg < len(arr.coef_num)
                else np.zeros_like(arr.coef_num[0])
                for arr in arrays
            ],
            axis=axis,
        )
        for deg in range(max_num_deg)
    ]
    cat_den = [
        np.concatenate(
            [
                arr.coef_den[deg] if deg < len(arr.coef_den)
                else np.ones_like(arr.coef_den[0])
                for arr in arrays
            ],
            axis=axis,
        )
        for deg in range(max_den_deg)
    ]

    return siarray.from_rational(cat_num, cat_den)


def array(X) -> siarray:
    """Convenience constructor for siarray."""
    return siarray(X)
