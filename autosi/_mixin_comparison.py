"""Comparison operators of :class:`~autosi.siarray` (``< > <= >=``).

Each comparison reduces to a sign condition on a rational function of
``z`` and records the interval on which its outcome is unchanged.
Equality operators are untracked (see the paper's appendix): non-identical
rational functions agree only on a null set.
"""

from __future__ import annotations

import numpy as np

from .constraints import add_constraint, add_batch_constraints


class ComparisonMixin:
    """Comparison operators for siarray."""

    def _track_diff_constraint(self, other, negate: bool) -> np.ndarray:
        """Track the interval constraint of ``diff = self - other`` and return its value.

        Scalars take a lightweight single-element path; arrays are tracked in a
        single batch over all elements.

        Parameters
        ----------
        other : siarray or scalar
            Right-hand operand.
        negate : bool
            When True, track ``-diff / den > 0`` (used by ``__lt__`` / ``__le__``).

        Returns
        -------
        np.ndarray
            The value of ``diff`` at the current ``z``.
        """
        other = self._to_siarray(other)
        diff = self - other

        if diff.data.size == 1:
            # Scalar: single-element path (lightweight)
            num = np.array([c.flatten()[0] if c.size > 0 else 0 for c in diff.coef_num])
            den = np.array([c.flatten()[0] if c.size > 0 else 1 for c in diff.coef_den])
            if negate:
                num = -num
            add_constraint(num, den)
        else:
            # Array: track all elements in one batch
            nums = np.array([c.flatten() for c in diff.coef_num]).T  # (n_elems, d_num)
            dens = np.array([c.flatten() for c in diff.coef_den]).T  # (n_elems, d_den)
            if negate:
                nums = -nums
            add_batch_constraints(nums, dens)

        return diff.data

    def _scalar_or_array(self, result: np.ndarray):
        """Return a Python bool for a scalar, or a NumPy array otherwise."""
        if result.size == 1:
            return bool(result.item())
        return result

    def __lt__(self, other):
        # diff < 0  <=>  -diff / den > 0
        data = self._track_diff_constraint(other, negate=True)
        return self._scalar_or_array(data < 0)

    def __le__(self, other):
        data = self._track_diff_constraint(other, negate=True)
        return self._scalar_or_array(data <= 0)

    def __gt__(self, other):
        # diff > 0  <=>  diff / den > 0
        data = self._track_diff_constraint(other, negate=False)
        return self._scalar_or_array(data > 0)

    def __ge__(self, other):
        data = self._track_diff_constraint(other, negate=False)
        return self._scalar_or_array(data >= 0)

    def __eq__(self, other):
        other = self._to_siarray(other)
        return self._scalar_or_array(np.equal(self.data, other.data))

    def __ne__(self, other):
        other = self._to_siarray(other)
        return self._scalar_or_array(np.not_equal(self.data, other.data))
