"""Polynomial arithmetic and interval computation on coefficient lists.

Polynomials are represented as ascending coefficient lists (``p[i]`` is the
coefficient of ``z**i``), with array-valued coefficients so that a whole
siarray shares one list. This module provides the ring operations
(:func:`poly_add`, :func:`poly_mul`, ...), degree reduction, root finding
via companion-matrix eigenvalues, and the conversion of sign conditions
``num/den > 0`` into intervals of ``z``.
"""

from __future__ import annotations

import builtins

import numpy as np
from numpy.polynomial.polynomial import polydiv, polytrim
from sicore import RealSubset, polynomial_iso_sign_interval


def poly_add(p1: list[np.ndarray], p2: list[np.ndarray]) -> list[np.ndarray]:
    """Add two polynomials.

    Adds the shorter operand into a copy of the longer one to minimize copies.

    Parameters
    ----------
    p1, p2 : list of np.ndarray
        Ascending coefficient lists (``p[i]`` is the coefficient of ``z**i``).

    Returns
    -------
    list of np.ndarray
        Coefficients of ``p1 + p2``.
    """
    d1, d2 = len(p1), len(p2)
    if d1 >= d2:
        result = [c.copy() for c in p1]
        for i in range(d2):
            result[i] = result[i] + p2[i]
    else:
        result = [c.copy() for c in p2]
        for i in range(d1):
            result[i] = result[i] + p1[i]
    return result


def poly_sub(p1: list[np.ndarray], p2: list[np.ndarray]) -> list[np.ndarray]:
    """Subtract two polynomials.

    Parameters
    ----------
    p1, p2 : list of np.ndarray
        Ascending coefficient lists.

    Returns
    -------
    list of np.ndarray
        Coefficients of ``p1 - p2``.
    """
    d1, d2 = len(p1), len(p2)
    n = builtins.max(d1, d2)
    result = []
    for i in range(n):
        c1 = p1[i] if i < d1 else np.zeros_like(p1[0])
        c2 = p2[i] if i < d2 else np.zeros_like(p2[0])
        result.append(c1 - c2)
    return result


def poly_mul(p1: list[np.ndarray], p2: list[np.ndarray]) -> list[np.ndarray]:
    """Multiply two polynomials (vectorized).

    Uses three paths: direct product for two constants, ``np.convolve`` for scalar
    coefficients, and slice-wise accumulation (an ``O(d1)`` Python loop) for array
    coefficients.

    Parameters
    ----------
    p1, p2 : list of np.ndarray
        Ascending coefficient lists.

    Returns
    -------
    list of np.ndarray
        Coefficients of ``p1 * p2``.
    """
    d1, d2 = len(p1), len(p2)

    # Fast path: constant times constant
    if d1 == 1 and d2 == 1:
        return [p1[0] * p2[0]]

    n = d1 + d2 - 1
    shape1 = np.asarray(p1[0]).shape
    shape2 = np.asarray(p2[0]).shape

    # Both scalar coefficients: use np.convolve (FFT-based)
    if shape1 == () and shape2 == ():
        a = np.array([float(c) for c in p1])
        b = np.array([float(c) for c in p2])
        conv = np.convolve(a, b)
        return [np.array(conv[k]) for k in range(n)]

    # Array coefficients: vectorize the inner loop with slice-wise addition,
    # i.e. result[i+j] += p1[i] * p2[j] -> result[i:i+d2] += p1[i] * stacked_p2
    shape = np.broadcast_shapes(shape1, shape2)
    a = np.asarray(p1)  # (d1, *shape1)
    b = np.asarray(p2)  # (d2, *shape2)
    dtype = np.result_type(a.dtype, b.dtype)
    result = np.zeros((n, *shape), dtype=dtype)

    # Align dimensions so a[i] * b broadcasts against the result slice,
    # e.g. shape1=(n,), shape2=() -> b is (d2,) -> reshape to (d2, 1)
    ndim_diff = len(shape1) - len(shape2)
    if ndim_diff > 0:
        b = b.reshape((d2,) + shape2 + (1,) * ndim_diff)

    for i in range(d1):
        result[i : i + d2] += a[i] * b  # inner loop eliminated
    return [result[k] for k in range(n)]


def _denoms_equal(d1: list[np.ndarray], d2: list[np.ndarray]) -> bool:
    """Return whether two denominator polynomials are equal."""
    if len(d1) != len(d2):
        return False
    return all(np.array_equal(a, b) for a, b in zip(d1, d2))


def _denom_is_one(coef_den: list[np.ndarray]) -> bool:
    """Return whether the denominator is the constant 1 (degree-0, all ones)."""
    return len(coef_den) == 1 and np.allclose(coef_den[0], 1.0)


def poly_roots(coefs: np.ndarray) -> tuple:
    """Compute the roots of several polynomials (independent of ``z``).

    The roots do not depend on ``z``, so they can be cached across the ``z``
    iterations of ``forward_si`` (memoization level A). Conversion to an interval is
    done per ``z`` by :func:`iso_from_roots`.

    Parameters
    ----------
    coefs : np.ndarray, shape (n, d+1)
        Each row holds the ascending coefficients of one polynomial.

    Returns
    -------
    tuple
        ``("const", n)``, ``("linear", roots(n,))`` or ``("poly", real_roots(n, d))``.
    """
    n, d1 = coefs.shape
    d = d1 - 1

    if d == 0:
        return ("const", n)  # constant polynomial: no roots -> (-inf, inf)

    if d == 1:
        # Linear: root = -a/b (closed form, fastest)
        a, b = coefs[:, 0], coefs[:, 1]
        valid = np.abs(b) > 1e-15
        roots = np.where(valid, -a / np.where(valid, b, 1.0), np.nan)
        return ("linear", roots)

    # Higher degree: batch eigenvalues of the companion matrices.
    # companion[k, 1:, :-1] = I(d-1), companion[k, :, -1] = -coef[k, :-1] / leading[k]
    leading = coefs[:, -1]
    C = np.zeros((n, d, d))
    C[:, 1:, :-1] = np.eye(d - 1)  # sub-diagonal ones (shared by all polynomials)
    safe_leading = np.where(np.abs(leading) > 1e-15, leading, 1.0)
    C[:, :, -1] = -coefs[:, :-1] / safe_leading[:, np.newaxis]

    eigenvalues = np.linalg.eigvals(C)  # (n, d) complex, batched
    # Use the real part of complex roots, matching polynomial_iso_sign_interval
    return ("poly", eigenvalues.real)


def iso_from_roots(roots: tuple, z: float) -> np.ndarray:
    """Build the iso-sign interval at ``z`` from precomputed roots (level A).

    Parameters
    ----------
    roots : tuple
        Output of :func:`poly_roots`.
    z : float
        Point around which the sign-preserving interval is taken.

    Returns
    -------
    np.ndarray, shape (n, 2)
        Rows ``[lower, upper]`` for each polynomial.
    """
    kind = roots[0]
    if kind == "const":
        n = roots[1]
        result = np.empty((n, 2))
        result[:, 0] = -np.inf
        result[:, 1] = np.inf
        return result

    if kind == "linear":
        r = roots[1]
        result = np.empty((len(r), 2))
        result[:, 0] = np.where(r < z, r, -np.inf)
        result[:, 1] = np.where(r > z, r, np.inf)
        return result

    real_roots = roots[1]  # (n, d)
    n = real_roots.shape[0]
    result = np.empty((n, 2))
    result[:, 0] = -np.inf
    result[:, 1] = np.inf

    # Largest root below z -> lower bound
    below = np.where(real_roots < z - 1e-10, real_roots, np.nan)
    has_below = ~np.all(np.isnan(below), axis=1)
    if has_below.any():
        result[has_below, 0] = np.nanmax(below[has_below], axis=1)

    # Smallest root above z -> upper bound
    above = np.where(real_roots > z + 1e-10, real_roots, np.nan)
    has_above = ~np.all(np.isnan(above), axis=1)
    if has_above.any():
        result[has_above, 1] = np.nanmin(above[has_above], axis=1)

    return result


def _batch_iso_sign_intervals(coefs: np.ndarray, z: float) -> np.ndarray:
    """Batch-compute iso-sign intervals for several polynomials (non-memoized)."""
    return iso_from_roots(poly_roots(coefs), z)


def calc_polynomial_intervals_batch(
    numerators: np.ndarray,    # (n, deg_num+1) ascending coefficients
    denominators: np.ndarray,  # (n, deg_den+1) ascending coefficients
    z: float,
    threshold: float = 0.0,
) -> list[RealSubset]:
    """Batch-compute the intervals where ``num_i / den_i > threshold``.

    Instead of calling ``polynomial_iso_sign_interval`` ``n`` times, this processes
    all rows with batched eigenvalue computation. It is the batched, mathematically
    equivalent counterpart of :func:`calc_polynomial_interval`::

        iso_sign(den, z) & iso_sign(num - den * threshold, z)

    Parameters
    ----------
    numerators : np.ndarray, shape (n, deg_num+1)
        Ascending numerator coefficients per row.
    denominators : np.ndarray, shape (n, deg_den+1)
        Ascending denominator coefficients per row.
    z : float
        Point around which the sign-preserving intervals are taken.
    threshold : float, optional
        Comparison threshold (default 0.0).

    Returns
    -------
    list of RealSubset
        One interval per row.
    """
    d_num, d_den = numerators.shape[1], denominators.shape[1]
    d_max = max(d_num, d_den)
    if d_num < d_max:
        numerators = np.pad(numerators, ((0, 0), (0, d_max - d_num)))
    if d_den < d_max:
        denominators = np.pad(denominators, ((0, 0), (0, d_max - d_den)))

    adj_num = numerators - denominators * threshold

    # When memoization is enabled, go through the root/interval caches to avoid
    # recomputation across z iterations.
    from . import poly_memo
    if poly_memo.is_enabled() and z is not None:
        return poly_memo.batch_intervals(denominators, adj_num, z)

    # Two batched calls handle all n rows at once.
    cond_intervals = _batch_iso_sign_intervals(denominators, z)  # (n, 2)
    adj_intervals  = _batch_iso_sign_intervals(adj_num, z)        # (n, 2)

    return combine_intervals(cond_intervals, adj_intervals)


def combine_intervals(cond_intervals: np.ndarray, adj_intervals: np.ndarray) -> list[RealSubset]:
    """Intersect denominator-condition and numerator intervals row by row.

    Parameters
    ----------
    cond_intervals, adj_intervals : np.ndarray, shape (n, 2)
        Rows ``[lower, upper]``.

    Returns
    -------
    list of RealSubset
        Per-row intersection.
    """
    return [
        RealSubset([cond_intervals[i].tolist()]) & RealSubset([adj_intervals[i].tolist()])
        for i in range(len(cond_intervals))
    ]


def _poly_gcd_1d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute the GCD of two 1-D polynomials (Euclid, ascending, monic)."""
    a = np.atleast_1d(polytrim(np.asarray(a, dtype=float)))
    b = np.atleast_1d(polytrim(np.asarray(b, dtype=float)))

    while len(b) > 0 and not np.allclose(b, 0, atol=1e-10):
        _, r = polydiv(a, b)
        r = np.atleast_1d(polytrim(r))
        a, b = b, r

    # Normalize to monic (leading coefficient 1)
    if len(a) > 0 and not np.isclose(a[-1], 0, atol=1e-10):
        a = a / a[-1]
    return a if len(a) > 0 else np.array([1.0])


def poly_reduce(
    num: list[np.ndarray], den: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Reduce a rational polynomial to lowest terms.

    Returns the inputs unchanged when either side is a constant (degree 0), because
    the GCD is then necessarily a constant. Handles both scalar and array
    coefficients.

    Parameters
    ----------
    num, den : list of np.ndarray
        Ascending numerator/denominator coefficient lists.

    Returns
    -------
    tuple of (list of np.ndarray, list of np.ndarray)
        Reduced ``(num, den)``.
    """
    # If either side is constant, the polynomial GCD is constant -> no reduction
    if len(num) <= 1 or len(den) <= 1:
        return num, den

    shape = np.asarray(num[0]).shape

    if shape == ():
        return _poly_reduce_scalar(num, den)
    else:
        return _poly_reduce_array(num, den, shape)


def _poly_reduce_scalar(
    num: list[np.ndarray], den: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Reduce a scalar-coefficient rational polynomial (fast path)."""
    a = np.array([float(c) for c in num])
    b = np.array([float(c) for c in den])
    gcd = _poly_gcd_1d(a, b)

    if len(gcd) <= 1:
        return num, den

    q_num, r_num = polydiv(a, gcd)
    q_den, r_den = polydiv(b, gcd)

    if not (np.allclose(r_num, 0, atol=1e-10) and np.allclose(r_den, 0, atol=1e-10)):
        return num, den

    q_num = np.atleast_1d(polytrim(q_num))
    q_den = np.atleast_1d(polytrim(q_den))
    if len(q_num) == 0:
        q_num = np.array([0.0])
    if len(q_den) == 0:
        q_den = np.array([1.0])

    return [np.array(c) for c in q_num], [np.array(c) for c in q_den]


def _poly_reduce_array(
    num: list[np.ndarray], den: list[np.ndarray], shape: tuple
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Reduce an array-coefficient rational polynomial element-wise by GCD.

    Each element is reduced independently; only elements that actually reduce are
    updated. If no element reduces, the original lists are returned unchanged.

    Parameters
    ----------
    num, den : list of np.ndarray
        Ascending coefficient lists with array-valued coefficients.
    shape : tuple
        Logical shape of the coefficient arrays.

    Returns
    -------
    tuple of (list of np.ndarray, list of np.ndarray)
        Reduced ``(num, den)``.
    """
    n = np.asarray(num[0]).size

    # Coefficient matrices: convert to (n, degree+1)
    nums_mat = np.array([c.flatten() for c in num], dtype=float).T  # (n, d_num)
    dens_mat = np.array([c.flatten() for c in den], dtype=float).T  # (n, d_den)

    new_num_rows: list[np.ndarray] = []
    new_den_rows: list[np.ndarray] = []
    any_reduced = False

    for i in range(n):
        a_i = np.atleast_1d(polytrim(nums_mat[i]))
        b_i = np.atleast_1d(polytrim(dens_mat[i]))
        if len(a_i) == 0:
            a_i = np.array([0.0])
        if len(b_i) == 0:
            b_i = np.array([1.0])

        gcd_i = _poly_gcd_1d(a_i, b_i)

        if len(gcd_i) <= 1:
            # Constant GCD -> this element does not reduce
            new_num_rows.append(a_i)
            new_den_rows.append(b_i)
            continue

        q_num_i, r_num_i = polydiv(a_i, gcd_i)
        q_den_i, r_den_i = polydiv(b_i, gcd_i)

        # Skip if it does not divide cleanly (floating-point error)
        if not (np.allclose(r_num_i, 0, atol=1e-10) and np.allclose(r_den_i, 0, atol=1e-10)):
            new_num_rows.append(a_i)
            new_den_rows.append(b_i)
            continue

        q_num_i = np.atleast_1d(polytrim(q_num_i))
        q_den_i = np.atleast_1d(polytrim(q_den_i))
        if len(q_num_i) == 0:
            q_num_i = np.array([0.0])
        if len(q_den_i) == 0:
            q_den_i = np.array([1.0])

        new_num_rows.append(q_num_i)
        new_den_rows.append(q_den_i)
        any_reduced = True

    if not any_reduced:
        return num, den

    # Pad to the post-reduction max degree and rebuild the coefficient lists
    max_d_num = max(len(r) for r in new_num_rows)
    max_d_den = max(len(r) for r in new_den_rows)

    new_nums_mat = np.zeros((n, max_d_num))
    new_dens_mat = np.zeros((n, max_d_den))
    for i, (r_n, r_d) in enumerate(zip(new_num_rows, new_den_rows)):
        new_nums_mat[i, :len(r_n)] = r_n
        new_dens_mat[i, :len(r_d)] = r_d

    new_num_list = [new_nums_mat[:, k].reshape(shape) for k in range(max_d_num)]
    new_den_list = [new_dens_mat[:, k].reshape(shape) for k in range(max_d_den)]

    return new_num_list, new_den_list


def calc_polynomial_interval(
    numerator: np.ndarray,
    denominator: np.ndarray,
    z: float,
    threshold: float = 0.0,
) -> RealSubset:
    """Compute the range of ``z`` where the rational function exceeds ``threshold``.

    Dispatches to the memoized path when memoization is enabled, otherwise to
    :func:`calc_polynomial_interval_direct`.

    Parameters
    ----------
    numerator, denominator : np.ndarray
        Ascending coefficient vectors.
    z : float
        Point around which the sign-preserving interval is taken.
    threshold : float, optional
        Comparison threshold (default 0.0).

    Returns
    -------
    RealSubset
        The interval where ``numerator / denominator > threshold``.
    """
    from . import poly_memo
    if poly_memo.is_enabled() and z is not None:
        return poly_memo.scalar_interval(numerator, denominator, z, threshold)
    return calc_polynomial_interval_direct(numerator, denominator, z, threshold)


def calc_polynomial_interval_direct(
    numerator: np.ndarray,
    denominator: np.ndarray,
    z: float,
    threshold: float = 0.0,
) -> RealSubset:
    """Non-memoized implementation behind :func:`calc_polynomial_interval`.

    Parameters
    ----------
    numerator, denominator : np.ndarray
        Ascending coefficient vectors.
    z : float
        Point around which the sign-preserving interval is taken.
    threshold : float, optional
        Comparison threshold (default 0.0).

    Returns
    -------
    RealSubset
        The interval where ``numerator / denominator > threshold``.
    """
    if len(numerator) < len(denominator):
        numerator = np.concatenate(
            [numerator, np.zeros(len(denominator) - len(numerator))]
        )
    elif len(denominator) < len(numerator):
        denominator = np.concatenate(
            [denominator, np.zeros(len(numerator) - len(denominator))]
        )

    # Case denominator > 0: solve numerator - denominator * threshold > 0
    pos_cond = RealSubset(polynomial_iso_sign_interval(denominator, z))
    pos_int = RealSubset(
        polynomial_iso_sign_interval(numerator - denominator * threshold, z)
    )
    intervals = pos_cond & pos_int

    # Case denominator < 0: solve numerator - denominator * threshold < 0
    neg_cond = RealSubset(polynomial_iso_sign_interval(-denominator, z))
    neg_int = RealSubset(
        polynomial_iso_sign_interval(denominator * threshold - numerator, z)
    )
    intervals = intervals | (neg_cond & neg_int)

    return RealSubset(intervals)
