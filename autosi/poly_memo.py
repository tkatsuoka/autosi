from __future__ import annotations

import numpy as np
from sicore import RealSubset


class PolyMemo:
    """Memoization of interval computation across the ``z`` iterations of forward_si.

    Within a single ``inference()`` (one p-value) the selection line ``a, b`` is
    fixed and only ``z`` changes, so the polynomial coefficients produced at each
    computation site are invariant across calls. Two cache levels exploit this:

    - Level A (root cache): polynomial roots (eigenvalues) do not depend on ``z``, so
      they are cached by coefficient hash to avoid recomputing the expensive
      ``eigvals``.
    - Level B (interval cache): an iso-sign interval is the maximal interval around
      ``z`` over which the result is constant, so the computation can be skipped when
      the next ``z`` falls inside the cached ``[lo, hi]``. Because the search
      alternates positive and negative ``z``, the interval cache is keyed per sign
      (True/False).

    Since ``a, b`` change for each p-value, the coefficients (and therefore the key)
    change too, so stale reuse cannot occur. To bound memory, ``reset()`` is called
    at the start of ``inference()`` and only the most recent entry per sign is kept
    at each computation site.
    """

    def __init__(self) -> None:
        self._enabled = True
        self._roots: dict = {}   # key -> (roots_den, roots_adj)  level A
        self._region: dict = {}  # key -> {sign(bool): (lo, hi, result)}  level B

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def set_enabled(self, flag: bool) -> None:
        """Enable or disable memoization."""
        self._enabled = bool(flag)

    def is_enabled(self) -> bool:
        """Return whether memoization is enabled."""
        return self._enabled

    def reset(self) -> None:
        """Clear all caches (call when switching p-value computation)."""
        self._roots.clear()
        self._region.clear()

    # ------------------------------------------------------------------
    # Memoization core
    # ------------------------------------------------------------------

    @staticmethod
    def _key(*arrays: np.ndarray) -> tuple:
        """Build a deterministic hash key from coefficient arrays."""
        parts: list = []
        for a in arrays:
            a = np.ascontiguousarray(a, dtype=float)
            parts.append((a.shape, a.tobytes()))
        return tuple(parts)

    def batch_intervals(
        self, denominators: np.ndarray, adj_num: np.ndarray, z: float
    ) -> list[RealSubset]:
        """Memoized computation of batched rational-function intervals.

        Parameters
        ----------
        denominators : np.ndarray, shape (n, deg)
            Ascending denominator coefficients per row.
        adj_num : np.ndarray, shape (n, deg)
            Ascending coefficients of ``numerator - denominator * threshold`` per row.
        z : float
            Point around which the sign-preserving intervals are taken.

        Returns
        -------
        list of RealSubset
            One interval per row.
        """
        from .poly_utils import poly_roots, iso_from_roots, combine_intervals

        key = self._key(denominators, adj_num)
        sign = z >= 0

        # Level B: if z lies in the cached interval, return the cached result
        region = self._region.get(key)
        if region is not None and sign in region:
            lo, hi, result = region[sign]
            if lo <= z <= hi:
                return result

        # Level A: roots are z-invariant, so fetch from cache (compute if absent)
        roots = self._roots.get(key)
        if roots is None:
            roots = (poly_roots(denominators), poly_roots(adj_num))
            self._roots[key] = roots

        cond_intervals = iso_from_roots(roots[0], z)  # (n, 2)
        adj_intervals = iso_from_roots(roots[1], z)   # (n, 2)
        result = combine_intervals(cond_intervals, adj_intervals)

        # Interval over which the result is constant = intersection of all iso
        # intervals (contains z)
        lo = max(cond_intervals[:, 0].max(), adj_intervals[:, 0].max())
        hi = min(cond_intervals[:, 1].min(), adj_intervals[:, 1].min())
        self._region.setdefault(key, {})[sign] = (lo, hi, result)
        return result

    def scalar_interval(
        self,
        numerator: np.ndarray,
        denominator: np.ndarray,
        z: float,
        threshold: float,
    ) -> RealSubset:
        """Memoized computation of a scalar rational-function interval (level B).

        Parameters
        ----------
        numerator, denominator : np.ndarray
            Ascending coefficient vectors.
        z : float
            Point around which the sign-preserving interval is taken.
        threshold : float
            Comparison threshold.

        Returns
        -------
        RealSubset
            The interval where ``numerator / denominator > threshold``.
        """
        from .poly_utils import calc_polynomial_interval_direct

        key = self._key(numerator, denominator) + (float(threshold),)
        sign = z >= 0

        region = self._region.get(key)
        if region is not None and sign in region:
            lo, hi, result = region[sign]
            if lo <= z <= hi:
                return result

        result = calc_polynomial_interval_direct(numerator, denominator, z, threshold)

        # Find the interval over which the result is constant: if z is inside the
        # solution set use that interval, otherwise use the complementary interval
        # (both are constant regions bounded by adjacent roots).
        try:
            lo, hi = result.find_interval_containing(z)
        except Exception:
            try:
                lo, hi = (~result).find_interval_containing(z)
            except Exception:
                lo, hi = z, z  # safe fallback: do not reuse the cache
        self._region.setdefault(key, {})[sign] = (lo, hi, result)
        return result


# Module-level singleton
_MEMO = PolyMemo()


def is_enabled() -> bool:
    """Return whether memoization is enabled."""
    return _MEMO.is_enabled()


def set_enabled(flag: bool) -> None:
    """Enable or disable memoization globally."""
    _MEMO.set_enabled(flag)


def reset() -> None:
    """Clear all caches."""
    _MEMO.reset()


def batch_intervals(denominators, adj_num, z):
    """Module-level wrapper for :meth:`PolyMemo.batch_intervals`."""
    return _MEMO.batch_intervals(denominators, adj_num, z)


def scalar_interval(numerator, denominator, z, threshold):
    """Module-level wrapper for :meth:`PolyMemo.scalar_interval`."""
    return _MEMO.scalar_interval(numerator, denominator, z, threshold)
