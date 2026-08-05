"""The tracked array type at the core of AutoSI.

:class:`siarray` carries every element as a rational function of the line
parameter ``z`` instead of a plain number. Arithmetic stays within the
representation (sums, products, and quotients of rational functions are
again rational), while comparisons and selections record the interval of
``z`` on which their outcome is unchanged. The heavy lifting is split into
operator mixins (``_mixin_*``); this module defines the data structure,
its constructors, and value access.
"""

from __future__ import annotations

import numpy as np

from .interval_tracker import IntervalTracker
from ._mixin_arithmetic import ArithmeticMixin
from ._mixin_comparison import ComparisonMixin
from ._mixin_index import IndexMixin
from ._mixin_array import ArrayMixin
from ._mixin_reduction import ReductionMixin


class siarray(ArithmeticMixin, ComparisonMixin, IndexMixin, ArrayMixin, ReductionMixin):
    """Array whose elements are rational functions of ``z``.

    Each element is represented by ascending coefficient lists ``coef_num`` and
    ``coef_den``; selection and comparison operations on it record the selection
    event as a truncation interval (see :class:`IntervalTracker`).
    """

    def __init__(
        self,
        X: np.ndarray | float | int | "siarray" | None = None,
        numerator: list[np.ndarray] | None = None,
        denominator: list[np.ndarray] | None = None,
    ):
        """Construct from a value or from rational coefficient lists.

        Parameters
        ----------
        X : np.ndarray, float, int, or siarray, optional
            Array or scalar value to wrap.
        numerator : list of np.ndarray, optional
            Numerator coefficients of the rational function.
        denominator : list of np.ndarray, optional
            Denominator coefficients of the rational function.
        """
        if isinstance(X, siarray):
            self.coef_num = [c.copy() for c in X.coef_num]
            self.coef_den = [c.copy() for c in X.coef_den]
            self._shape = X._shape
        elif numerator is not None:
            self.coef_num = list(numerator)
            self.coef_den = (
                list(denominator)
                if denominator is not None
                else [np.ones_like(self.coef_num[0])]
            )
            self._shape = numerator[0].shape
        else:
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            self.coef_num = [X]
            self.coef_den = [np.ones_like(X)]
            self._shape = X.shape

    # ------------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------------

    @classmethod
    def constant(cls, value: np.ndarray | float | int) -> "siarray":
        """Create a constant node.

        Parameters
        ----------
        value : np.ndarray, float, or int
            Constant value.

        Returns
        -------
        siarray
            Degree-0 rational function holding ``value``.
        """
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        obj = cls.__new__(cls)
        obj.coef_num = [value]
        obj.coef_den = [np.ones_like(value)]
        obj._shape = value.shape
        return obj

    @classmethod
    def from_rational(
        cls,
        numerator: list[np.ndarray],
        denominator: list[np.ndarray] | None = None,
    ) -> "siarray":
        """Create a rational function from coefficient lists.

        Parameters
        ----------
        numerator : list of np.ndarray
            Ascending numerator coefficients.
        denominator : list of np.ndarray, optional
            Ascending denominator coefficients (default: constant 1).

        Returns
        -------
        siarray
            The constructed rational function array.
        """
        obj = cls.__new__(cls)
        obj.coef_num = list(numerator)
        obj.coef_den = (
            list(denominator)
            if denominator is not None
            else [np.ones_like(numerator[0])]
        )
        obj._shape = numerator[0].shape
        return obj

    # ------------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------------

    def _to_siarray(self, other) -> "siarray":
        """Convert another value into an siarray."""
        if isinstance(other, siarray):
            return other
        return siarray.constant(other)

    @property
    def data(self) -> np.ndarray:
        """Value of the rational function at the current ``z``.

        Returns
        -------
        np.ndarray
            The evaluated array. Uses ``z = 0`` when no ``z`` is set.
        """
        z = IntervalTracker.get_instance().get_z()
        if z is None:
            z = 0.0
        num_val = np.sum(
            [self.coef_num[i] * (z**i) for i in range(len(self.coef_num))], axis=0
        )
        den_val = np.sum(
            [self.coef_den[i] * (z**i) for i in range(len(self.coef_den))], axis=0
        )
        return (num_val / den_val).reshape(self._shape)

    @property
    def deg(self) -> tuple[int, int]:
        """Return ``(numerator degree, denominator degree)``."""
        return (len(self.coef_num) - 1, len(self.coef_den) - 1)

    # ------------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------------

    @property
    def shape(self):
        """Logical shape of the array."""
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self._shape)

    @property
    def size(self) -> int:
        """Total number of elements."""
        return np.prod(self._shape)

    @property
    def dtype(self):
        """Coefficient dtype."""
        return self.coef_num[0].dtype

    # ------------------------------------------------------------------------
    # Conversion methods
    # ------------------------------------------------------------------------

    def to_numpy(self) -> np.ndarray:
        """Return the value at the current ``z`` as a NumPy array (copy)."""
        return self.data.copy()

    def values(self) -> np.ndarray:
        """Return the value at the current ``z``."""
        return self.data

    def __repr__(self) -> str:
        return f"siarray({repr(self.data)})"

    def __str__(self) -> str:
        return f"siarray({str(self.data)})"
