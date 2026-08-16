"""Reduction methods of :class:`~autosi.siarray` (``sum``, ``mean``, ``var``,
``prod``).

Pure rational arithmetic with no tracking. Fast paths cover the common
case where every denominator is the constant 1.
"""

from __future__ import annotations

import numpy as np

from .poly_utils import poly_add, poly_mul, _denom_is_one


def _accumulate_rational_sum(terms):
    """Fold a sequence of rational functions ``(num, den)`` into their sum.

    Repeats ``a/b + c/d = (a*d + c*b) / (b*d)`` over the sequence. Shared by the
    general path of ``sum`` / ``mean`` (when the denominator is not the constant 1).

    Parameters
    ----------
    terms : iterable of (list of np.ndarray, list of np.ndarray)
        Sequence of ``(numerator, denominator)`` coefficient lists.

    Returns
    -------
    tuple of (list of np.ndarray, list of np.ndarray)
        Coefficients of the accumulated sum.
    """
    acc_num = [np.zeros(())]
    acc_den = [np.ones(())]
    for elem_num, elem_den in terms:
        acc_num = poly_add(poly_mul(acc_num, elem_den), poly_mul(elem_num, acc_den))
        acc_den = poly_mul(acc_den, elem_den)
    return acc_num, acc_den


def _accumulate_rational_prod(terms):
    """Fold a sequence of rational functions ``(num, den)`` into their product.

    Multiplies numerators and denominators term by term. Shared by the general path
    of ``prod`` (when the denominator is not the constant 1, or the numerators are
    not degree 0).

    Parameters
    ----------
    terms : iterable of (list of np.ndarray, list of np.ndarray)
        Sequence of ``(numerator, denominator)`` coefficient lists.

    Returns
    -------
    tuple of (list of np.ndarray, list of np.ndarray)
        Coefficients of the accumulated product.
    """
    acc_num = [np.ones(())]
    acc_den = [np.ones(())]
    for elem_num, elem_den in terms:
        acc_num = poly_mul(acc_num, elem_num)
        acc_den = poly_mul(acc_den, elem_den)
    return acc_num, acc_den


class ReductionMixin:
    """Reduction methods for siarray (``sum``, ``mean``, ``var``, ``prod``)."""

    def sum(self, axis=None, keepdims=False):
        """Sum of the array elements (rational-function version)."""
        if axis is None:
            return self._sum_all()
        return self._sum_axis(axis, keepdims)

    def _sum_all(self):
        """Sum over all elements."""
        # Fast path when every denominator is the constant 1: sum coefficient arrays
        if _denom_is_one(self.coef_den):
            result_num = [np.sum(c) for c in self.coef_num]
            return type(self).from_rational(result_num, [np.ones(())])

        terms = (
            (
                [c[idx] if c.size > 1 else c.item() for c in self.coef_num],
                [c[idx] if c.size > 1 else c.item() for c in self.coef_den],
            )
            for idx in np.ndindex(self._shape)
        )
        acc_num, acc_den = _accumulate_rational_sum(terms)
        return type(self).from_rational(acc_num, acc_den)

    def _sum_axis(self, axis: int, keepdims: bool):
        """Sum along a specific axis."""
        if axis < 0:
            axis = len(self._shape) + axis
        if axis < 0 or axis >= len(self._shape):
            raise ValueError(
                f"axis {axis} is out of bounds for array of dimension {len(self._shape)}"
            )

        if keepdims:
            result_shape = tuple(
                1 if i == axis else s for i, s in enumerate(self._shape)
            )
        else:
            result_shape = tuple(s for i, s in enumerate(self._shape) if i != axis)

        result_shape_for_init = result_shape if result_shape else (1,)
        axis_size = self._shape[axis]

        # Fast path when every denominator is the constant 1: just np.sum along axis
        if _denom_is_one(self.coef_den):
            result_num = [np.sum(c, axis=axis, keepdims=keepdims) for c in self.coef_num]
            if result_shape == ():
                result_num = [c.item() * np.ones(()) for c in result_num]
            return type(self).from_rational(result_num, [np.ones(result_shape_for_init)])

        # Initialize result buffers using a worst-case degree estimate
        max_num_deg = (len(self.coef_num) - 1) * axis_size
        max_den_deg = (len(self.coef_den) - 1) * axis_size
        result_num = [np.zeros(result_shape_for_init) for _ in range(max_num_deg + 1)]
        result_den = [np.zeros(result_shape_for_init) for _ in range(max_den_deg + 1)]

        for result_idx in np.ndindex(result_shape_for_init):
            # Enumerate the (num, den) of each element reduced into this output cell
            def _terms(result_idx=result_idx):
                for i in range(axis_size):
                    src_idx = list(result_idx)
                    if keepdims:
                        src_idx[axis] = i
                    else:
                        src_idx.insert(axis, i)
                    src_idx = tuple(src_idx)
                    yield (
                        [c[src_idx] if c.size > 1 else c.item() for c in self.coef_num],
                        [c[src_idx] if c.size > 1 else c.item() for c in self.coef_den],
                    )

            acc_num, acc_den = _accumulate_rational_sum(_terms())

            for deg, coef in enumerate(acc_num):
                result_num[deg][result_idx] = coef
            for deg, coef in enumerate(acc_den):
                result_den[deg][result_idx] = coef

        # Drop trailing zero coefficients
        while len(result_num) > 1 and np.allclose(result_num[-1], 0):
            result_num.pop()
        while len(result_den) > 1 and np.allclose(result_den[-1], 0):
            result_den.pop()

        # Adjust the shape
        if result_shape == ():
            result_num = [c.item() * np.ones(()) for c in result_num]
            result_den = [c.item() * np.ones(()) for c in result_den]
        elif not keepdims:
            result_num = [c.reshape(result_shape) for c in result_num]
            result_den = [c.reshape(result_shape) for c in result_den]

        return type(self).from_rational(result_num, result_den)

    def mean(self, axis=None, keepdims=False):
        """Mean of the array elements (rational-function version)."""
        if axis is None:
            return self.sum(axis=None, keepdims=False) / np.prod(self._shape)

        if axis < 0:
            axis = len(self._shape) + axis
        if axis < 0 or axis >= len(self._shape):
            raise ValueError(
                f"axis {axis} is out of bounds for array of dimension {len(self._shape)}"
            )
        return self.sum(axis=axis, keepdims=keepdims) / self._shape[axis]

    def var(self, axis=None, keepdims=False, ddof=0):
        """Variance of the array elements (rational-function version).

        Computed as ``mean((x - mean(x)) ** 2)`` with the denominator ``N - ddof``,
        which stays rational in ``z`` (unlike the standard deviation, which is not).

        Parameters
        ----------
        axis : int or None, optional
            Axis to reduce over. ``None`` reduces over all elements.
        keepdims : bool, optional
            Keep the reduced axis as size 1 (default False).
        ddof : int, optional
            Delta degrees of freedom; the divisor is ``N - ddof`` (default 0).

        Returns
        -------
        siarray
            The variance.
        """
        # mean with keepdims so it broadcasts against self for the subtraction
        m = self.mean(axis=axis, keepdims=True)
        dev = self - m
        ss = (dev * dev).sum(axis=axis, keepdims=keepdims)

        if axis is None:
            n = self.size
        else:
            ax = axis if axis >= 0 else len(self._shape) + axis
            n = self._shape[ax]
        return ss / float(n - ddof)

    def prod(self, axis=None, keepdims=False):
        """Product of the array elements (rational-function version)."""
        if axis is None:
            return self._prod_all()
        return self._prod_axis(axis, keepdims)

    def _prod_all(self):
        """Product over all elements."""
        # Fast path: degree-0 numerator with constant-1 denominator -> plain np.prod
        if len(self.coef_num) == 1 and _denom_is_one(self.coef_den):
            return type(self).from_rational([np.prod(self.coef_num[0])], [np.ones(())])

        terms = (
            (
                [c[idx] if c.size > 1 else c.item() for c in self.coef_num],
                [c[idx] if c.size > 1 else c.item() for c in self.coef_den],
            )
            for idx in np.ndindex(self._shape)
        )
        acc_num, acc_den = _accumulate_rational_prod(terms)
        return type(self).from_rational(acc_num, acc_den)

    def _prod_axis(self, axis: int, keepdims: bool):
        """Product along a specific axis."""
        if axis < 0:
            axis = len(self._shape) + axis
        if axis < 0 or axis >= len(self._shape):
            raise ValueError(
                f"axis {axis} is out of bounds for array of dimension {len(self._shape)}"
            )

        if keepdims:
            result_shape = tuple(
                1 if i == axis else s for i, s in enumerate(self._shape)
            )
        else:
            result_shape = tuple(s for i, s in enumerate(self._shape) if i != axis)

        result_shape_for_init = result_shape if result_shape else (1,)
        axis_size = self._shape[axis]

        # Fast path: degree-0 numerator with constant-1 denominator -> np.prod on axis
        if len(self.coef_num) == 1 and _denom_is_one(self.coef_den):
            result_num = [np.prod(self.coef_num[0], axis=axis, keepdims=keepdims)]
            if result_shape == ():
                result_num = [result_num[0].item() * np.ones(())]
            return type(self).from_rational(result_num, [np.ones(result_shape_for_init)])

        # The product of axis_size polynomials of degree d has degree d * axis_size
        max_num_deg = (len(self.coef_num) - 1) * axis_size
        max_den_deg = (len(self.coef_den) - 1) * axis_size
        result_num = [np.zeros(result_shape_for_init) for _ in range(max_num_deg + 1)]
        result_den = [np.zeros(result_shape_for_init) for _ in range(max_den_deg + 1)]

        for result_idx in np.ndindex(result_shape_for_init):
            # Enumerate the (num, den) of each element reduced into this output cell
            def _terms(result_idx=result_idx):
                for i in range(axis_size):
                    src_idx = list(result_idx)
                    if keepdims:
                        src_idx[axis] = i
                    else:
                        src_idx.insert(axis, i)
                    src_idx = tuple(src_idx)
                    yield (
                        [c[src_idx] if c.size > 1 else c.item() for c in self.coef_num],
                        [c[src_idx] if c.size > 1 else c.item() for c in self.coef_den],
                    )

            acc_num, acc_den = _accumulate_rational_prod(_terms())

            for deg, coef in enumerate(acc_num):
                result_num[deg][result_idx] = coef
            for deg, coef in enumerate(acc_den):
                result_den[deg][result_idx] = coef

        # Drop trailing zero coefficients
        while len(result_num) > 1 and np.allclose(result_num[-1], 0):
            result_num.pop()
        while len(result_den) > 1 and np.allclose(result_den[-1], 0):
            result_den.pop()

        # Adjust the shape
        if result_shape == ():
            result_num = [c.item() * np.ones(()) for c in result_num]
            result_den = [c.item() * np.ones(()) for c in result_den]
        elif not keepdims:
            result_num = [c.reshape(result_shape) for c in result_num]
            result_den = [c.reshape(result_shape) for c in result_den]

        return type(self).from_rational(result_num, result_den)
