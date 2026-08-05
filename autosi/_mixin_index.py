"""Indexing and scalar-conversion operators of :class:`~autosi.siarray`.

Getitem/setitem over the coefficient lists plus the scalar dunders
(``int()``, ``bool()``, indexing) that let tracked values drive ordinary
Python control flow.
"""

from __future__ import annotations

import builtins

import numpy as np


class IndexMixin:
    """Indexing and scalar-conversion operators for siarray."""

    def __int__(self):
        """Convert to int (used by ``range`` etc.)."""
        if self.size != 1:
            raise TypeError("only size-1 arrays can be converted to Python scalars")
        return int(self.data.item())

    def __bool__(self):
        """Convert to bool (used by ``if`` statements)."""
        if self.size != 1:
            raise ValueError(
                "The truth value of an array with more than one element is ambiguous"
            )
        return bool(self.data.item())

    def __index__(self):
        """Use as an index (used by ``range`` etc.)."""
        if self.size != 1:
            raise TypeError(
                "only integer scalar arrays can be converted to a scalar index"
            )
        return int(self.data.item())

    def __getitem__(self, key):
        """Get element(s)."""
        if isinstance(key, type(self)):
            key = key.coef_num[0]
        return type(self).from_rational(
            [c[key].copy() for c in self.coef_num],
            [c[key].copy() for c in self.coef_den],
        )

    def __setitem__(self, key, value):
        """Set element(s) (full replacement)."""
        if isinstance(key, type(self)):
            key = key.coef_num[0]

        value = self._to_siarray(value)

        max_num_deg = builtins.max(len(self.coef_num), len(value.coef_num))
        max_den_deg = builtins.max(len(self.coef_den), len(value.coef_den))

        while len(self.coef_num) < max_num_deg:
            self.coef_num.append(np.zeros_like(self.coef_num[0]))
        while len(self.coef_den) < max_den_deg:
            self.coef_den.append(np.ones_like(self.coef_den[0]))

        # Fully replace the numerator
        for i in range(len(value.coef_num)):
            if i < len(self.coef_num):
                self.coef_num[i][key] = value.coef_num[i].copy()
            else:
                self.coef_num.append(np.zeros_like(self.coef_num[0]))
                self.coef_num[i][key] = value.coef_num[i].copy()
        # Zero out numerator terms higher than value's degree
        for i in range(len(value.coef_num), len(self.coef_num)):
            self.coef_num[i][key] = 0.0

        # Fully replace the denominator
        for i in range(len(value.coef_den)):
            if i < len(self.coef_den):
                self.coef_den[i][key] = value.coef_den[i].copy()
            else:
                self.coef_den.append(np.ones_like(self.coef_den[0]))
                self.coef_den[i][key] = value.coef_den[i].copy()
        # Set denominator terms higher than value's degree to 1
        for i in range(len(value.coef_den), len(self.coef_den)):
            self.coef_den[i][key] = 1.0
