"""Arithmetic operators of :class:`~autosi.siarray` (``+ - * / @`` and friends).

All operations are closed over rational functions and impose no constraint
on ``z``; they only manipulate the stored coefficient lists.
"""

from __future__ import annotations

import numpy as np

from .constraints import add_batch_constraints
from .poly_utils import (
    poly_mul,
    poly_add,
    poly_sub,
    poly_reduce,
    _denoms_equal,
)


class ArithmeticMixin:
    """Arithmetic operators for siarray."""

    def _common_denominator(self):
        """Put all elements over a common denominator.

        Uses the product of the *unique* denominators only.

        Returns
        -------
        siarray
            Equivalent array sharing a single denominator across elements. Returns
            ``self`` unchanged when the denominator is already uniform.
        """
        # Fast check: are the coefficient arrays of every degree element-wise equal?
        d0 = [c.flat[0] for c in self.coef_den]
        if all(np.allclose(c, d0[deg]) for deg, c in enumerate(self.coef_den)):
            return self

        flat_size = self.coef_num[0].size

        # Map each element to the unique denominator it belongs to
        unique_dens: list = []
        elem_to_unique: list[int] = []

        for i in range(flat_size):
            den = [c.flat[i] for c in self.coef_den]
            found_idx = -1
            for j, u in enumerate(unique_dens):
                if len(u) == len(den) and all(
                    np.isclose(u[k], den[k]) for k in range(len(den))
                ):
                    found_idx = j
                    break
            if found_idx == -1:
                found_idx = len(unique_dens)
                unique_dens.append(den)
            elem_to_unique.append(found_idx)

        # Precompute, for each unique denominator, the product of all the others
        n_unique = len(unique_dens)
        multipliers = []
        for i in range(n_unique):
            mult = [np.array(1.0)]
            for j in range(n_unique):
                if i != j:
                    mult = poly_mul(mult, unique_dens[j])
            multipliers.append(mult)

        common_den = poly_mul(multipliers[0], unique_dens[0])

        # Adjust each element's numerator accordingly
        max_deg = len(common_den)
        new_num = [np.zeros(self._shape) for _ in range(max_deg)]

        for i in range(flat_size):
            elem_num = [c.flat[i] for c in self.coef_num]
            adjusted_num = poly_mul(elem_num, multipliers[elem_to_unique[i]])
            idx = np.unravel_index(i, self._shape)
            for deg, coef in enumerate(adjusted_num):
                if deg < max_deg:
                    new_num[deg][idx] = coef

        new_den = [np.full(self._shape, c) for c in common_den]
        return type(self).from_rational(new_num, new_den)

    def __add__(self, other):
        other = self._to_siarray(other)
        # Same denominator: add numerators only (avoid multiplying denominators)
        if _denoms_equal(self.coef_den, other.coef_den):
            return type(self).from_rational(
                poly_add(self.coef_num, other.coef_num),
                list(self.coef_den),
            )
        new_num = poly_add(
            poly_mul(self.coef_num, other.coef_den),
            poly_mul(other.coef_num, self.coef_den),
        )
        new_den = poly_mul(self.coef_den, other.coef_den)
        new_num, new_den = poly_reduce(new_num, new_den)
        return type(self).from_rational(new_num, new_den)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        other = self._to_siarray(other)
        # Same denominator: subtract numerators only (avoid multiplying denominators)
        if _denoms_equal(self.coef_den, other.coef_den):
            return type(self).from_rational(
                poly_sub(self.coef_num, other.coef_num),
                list(self.coef_den),
            )
        new_num = poly_sub(
            poly_mul(self.coef_num, other.coef_den),
            poly_mul(other.coef_num, self.coef_den),
        )
        new_den = poly_mul(self.coef_den, other.coef_den)
        new_num, new_den = poly_reduce(new_num, new_den)
        return type(self).from_rational(new_num, new_den)

    def __rsub__(self, other):
        return self._to_siarray(other).__sub__(self)

    def __mul__(self, other):
        other = self._to_siarray(other)
        return type(self).from_rational(
            poly_mul(self.coef_num, other.coef_num),
            poly_mul(self.coef_den, other.coef_den),
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        other = self._to_siarray(other)
        if len(other.coef_num) == 1 and len(other.coef_den) == 1:
            # Division by a constant: degree is unchanged, so poly_reduce is not needed
            new_num = [c / other.coef_num[0] for c in self.coef_num]
            new_den = [c / other.coef_den[0] for c in self.coef_den]
            return type(self).from_rational(new_num, new_den)
        new_num = poly_mul(self.coef_num, other.coef_den)
        new_den = poly_mul(self.coef_den, other.coef_num)
        new_num, new_den = poly_reduce(new_num, new_den)
        return type(self).from_rational(new_num, new_den)

    def __rtruediv__(self, other):
        return self._to_siarray(other).__truediv__(self)

    def __matmul__(self, other):
        other = self._to_siarray(other)

        # Put both operands over a common denominator
        self_cd = self._common_denominator()
        other_cd = other._common_denominator()

        n = len(self_cd.coef_num) + len(other_cd.coef_num) - 1
        new_num = [
            np.zeros_like(self_cd.coef_num[0] @ other_cd.coef_num[0])
            for _ in range(n)
        ]
        for i, c1 in enumerate(self_cd.coef_num):
            for j, c2 in enumerate(other_cd.coef_num):
                new_num[i + j] = new_num[i + j] + c1 @ c2

        result_shape = new_num[0].shape
        den_val = self_cd.coef_den[0].flat[0] * other_cd.coef_den[0].flat[0]
        new_den = [np.full(result_shape, den_val)]
        return type(self).from_rational(new_num, new_den)

    def __rmatmul__(self, other):
        return self._to_siarray(other).__matmul__(self)

    def __neg__(self):
        return type(self).from_rational(
            [-c.copy() for c in self.coef_num], self.coef_den
        )

    def __pow__(self, power):
        """Integer power via repeated multiplication."""
        power_val = int(power.data.item()) if isinstance(power, type(self)) else power
        result = self
        for _ in range(power_val - 1):
            result = result * self
        return result

    def __abs__(self):
        positive_idx = (self.data >= 0).view(bool)
        negative_idx = (self.data < 0).view(bool)

        new_num = [
            np.where(positive_idx, self.coef_num[i].copy(), -self.coef_num[i].copy())
            for i in range(len(self.coef_num))
        ]
        new_den = [c.copy() for c in self.coef_den]

        # Track the interval over which each element keeps its sign (+ if positive,
        # - if negative): impose num > 0 on the positive side and -num > 0 on the
        # negative side. Coefficient lists are (n_terms, n_elems), so transpose to
        # (n_elems, n_terms).
        if positive_idx.any():
            nums = np.array([c[positive_idx].flatten() for c in self.coef_num]).T
            dens = np.array([c[positive_idx].flatten() for c in self.coef_den]).T
            add_batch_constraints(nums, dens)

        if negative_idx.any():
            nums = np.array([-c[negative_idx].flatten() for c in self.coef_num]).T
            dens = np.array([c[negative_idx].flatten() for c in self.coef_den]).T
            add_batch_constraints(nums, dens)

        return type(self).from_rational(new_num, new_den)
