"""Helpers that push selection events (polynomial inequalities) into the tracker.

Comparison and selection operations (``max`` / ``argmax`` / ``argsort`` / ``abs`` /
``<`` ``>`` etc.) all perform the same task: find the range of ``z`` over which a
rational function keeps its sign, and accumulate it into the ``IntervalTracker`` by
intersection. That boilerplate is centralized here so each operation can call it in
a single, self-explanatory line.

Each function reads the current ``z`` from the ``IntervalTracker`` internally, so
callers do not need to pass ``z`` around.
"""

from __future__ import annotations

import numpy as np

from .interval_tracker import IntervalTracker
from .poly_utils import (
    calc_polynomial_interval,
    calc_polynomial_intervals_batch,
)


def _tracker_and_z() -> tuple[IntervalTracker, float | None]:
    """Return the shared tracker and its current ``z``."""
    tracker = IntervalTracker.get_instance()
    return tracker, tracker.get_z()


def add_constraint(numerator: np.ndarray, denominator: np.ndarray) -> None:
    """Add a single rational constraint ``numerator / denominator > 0``.

    Parameters
    ----------
    numerator : np.ndarray
        Ascending coefficients of the numerator polynomial.
    denominator : np.ndarray
        Ascending coefficients of the denominator polynomial.
    """
    tracker, z = _tracker_and_z()
    tracker.update(calc_polynomial_interval(numerator, denominator, z))


def add_batch_constraints(numerators: np.ndarray, denominators: np.ndarray) -> None:
    """Add ``n`` constraints ``num_i / den_i > 0`` from coefficient matrices.

    Parameters
    ----------
    numerators : np.ndarray, shape (n, deg_num)
        Each row holds the ascending coefficients of one polynomial.
    denominators : np.ndarray, shape (n, deg_den)
        Same as above, for the denominators.
    """
    tracker, z = _tracker_and_z()
    for interval in calc_polynomial_intervals_batch(numerators, denominators, z):
        tracker.update(interval)


def add_order_constraints(diffs: list) -> None:
    """Add one ``diff > 0`` constraint for each difference in ``diffs``.

    Used for the ordering constraints produced by ``max`` / ``min`` / ``argmax`` /
    ``argsort`` (``selected - other > 0``).

    Parameters
    ----------
    diffs : list of siarray
        Scalar-valued rational functions (coefficients are 0-d arrays).
    """
    if not diffs:
        return
    d_num = max(len(d.coef_num) for d in diffs)
    d_den = max(len(d.coef_den) for d in diffs)
    n = len(diffs)
    nums = np.zeros((n, d_num))
    dens = np.zeros((n, d_den))
    for i, diff in enumerate(diffs):
        for k, c in enumerate(diff.coef_num):
            nums[i, k] = float(c)
        for k, c in enumerate(diff.coef_den):
            dens[i, k] = float(c)
    add_batch_constraints(nums, dens)
