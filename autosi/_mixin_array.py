"""Array-manipulation methods of :class:`~autosi.siarray`.

Shape and layout operations (``T``, ``transpose``, ``reshape``, ``sort``,
...) applied uniformly to every coefficient array. Sorting operations
record the ordering comparisons as selection events.
"""

from __future__ import annotations

import numpy as np

from .constraints import add_order_constraints


class ArrayMixin:
    """Array-manipulation methods for siarray."""

    @property
    def T(self):
        """Transpose (matrix transpose of every coefficient array)."""
        return type(self).from_rational(
            [c.T for c in self.coef_num],
            [c.T for c in self.coef_den],
        )

    def transpose(self, axes=None):
        """Permute axes of every coefficient array."""
        return type(self).from_rational(
            [np.transpose(c, axes) for c in self.coef_num],
            [np.transpose(c, axes) for c in self.coef_den],
        )

    def inv(self):
        """Matrix-invert the (constant) numerator coefficients."""
        return type(self).from_rational(
            [np.linalg.inv(c) for c in self.coef_num],
            [c.copy() for c in self.coef_den],
        )

    def reshape(self, *shape):
        """Reshape every coefficient array."""
        if len(shape) == 1 and isinstance(shape[0], (list, tuple)):
            shape = shape[0]
        new_num = [c.reshape(shape) for c in self.coef_num]
        new_den = [c.reshape(shape) for c in self.coef_den]
        result = type(self).from_rational(new_num, new_den)
        result._shape = shape
        return result

    def argsort(self, axis=-1, kind=None, order=None):
        """Return sort indices, tracking the adjacent ordering as a selection event."""
        sorted_idx = np.argsort(self.data, axis=axis, kind=kind, order=order)
        n = len(sorted_idx) - 1
        if n > 0:
            # Track the interval over which the adjacent ordering (sort order) holds
            diffs = [self[sorted_idx[i]] - self[sorted_idx[i + 1]] for i in range(n)]
            add_order_constraints(diffs)
        return type(self).from_rational(
            [np.array(sorted_idx)],
            [np.ones_like(sorted_idx)],
        )

    def sort(self, axis=-1, kind=None, order=None):
        """Return the sorted values, tracking the ordering as a selection event.

        Companion to :meth:`argsort`: it tracks the same adjacent-ordering
        constraints but returns the elements gathered in sorted order. Supports 1-D
        arrays (as :meth:`argsort` does).

        Returns
        -------
        siarray
            The elements in ascending order.
        """
        sorted_idx = np.argsort(self.data, axis=axis, kind=kind, order=order)
        n = len(sorted_idx) - 1
        if n > 0:
            diffs = [self[sorted_idx[i]] - self[sorted_idx[i + 1]] for i in range(n)]
            add_order_constraints(diffs)
        # Gather the coefficients in sorted order
        return type(self).from_rational(
            [c[sorted_idx].copy() for c in self.coef_num],
            [c[sorted_idx].copy() for c in self.coef_den],
        )

    def flatten(self):
        """Return a 1-D copy (flatten every coefficient array)."""
        return type(self).from_rational(
            [c.flatten() for c in self.coef_num],
            [c.flatten() for c in self.coef_den],
        )

    def copy(self):
        """Return a deep copy."""
        return type(self).from_rational(
            [c.copy() for c in self.coef_num],
            [c.copy() for c in self.coef_den],
        )
