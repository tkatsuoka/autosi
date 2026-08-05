"""Process-wide accumulator for the truncation region.

The :class:`IntervalTracker` singleton is the hand-off point between the
user's algorithm and the inference loop: tracked operations read the
current ``z`` from it and intersect their sign-condition intervals into it.
"""

from __future__ import annotations

from sicore import RealSubset


class IntervalTracker:
    """Singleton holding the accumulated truncation region and the current ``z``.

    A single ``forward_si`` call (one re-execution of the selection algorithm at a
    given ``z``) accumulates the constraints it produces into ``_intervals`` by
    intersection. ``_z`` is the value of ``z`` currently being evaluated; comparison
    and selection operations use it to decide which branch is taken. Exactly one
    instance is shared per process (``get_instance``).

    Notes
    -----
    The tracker is the sole hand-off point between the user algorithm and the
    inference loop: operations read ``z`` from it and write intervals back into it.
    It is not thread-safe; do not run multiple inferences concurrently in the same
    process.
    """

    _instance: "IntervalTracker | None" = None  # the singleton instance

    def __init__(self) -> None:
        self._intervals: RealSubset | None = None
        self._z: float | None = None

    @classmethod
    def get_instance(cls) -> "IntervalTracker":
        """Return the shared instance, creating it on first use."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self, z: float | None = None) -> None:
        """Reset the accumulated region to the whole real line and set ``z``.

        Parameters
        ----------
        z : float or None, optional
            Value to store as the current ``z`` (default ``None``).
        """
        self._intervals = RealSubset([[-float("inf"), float("inf")]])
        self._z = z

    def get(self) -> RealSubset:
        """Return the accumulated region, initializing it if unset.

        Returns
        -------
        RealSubset
            The current accumulated truncation region.
        """
        if self._intervals is None:
            self.reset()
        return self._intervals

    def get_z(self) -> float | None:
        """Return the current ``z``.

        Returns
        -------
        float or None
            The value of ``z`` currently being evaluated.
        """
        return self._z

    def set_z(self, z: float) -> None:
        """Set the current ``z``.

        Parameters
        ----------
        z : float
            Value to evaluate the rational functions at.
        """
        self._z = z

    def update(self, new_intervals: RealSubset) -> RealSubset:
        """Intersect the accumulated region with ``new_intervals``.

        Parameters
        ----------
        new_intervals : RealSubset
            Region produced by a single selection event.

        Returns
        -------
        RealSubset
            The updated accumulated region.
        """
        if self._intervals is None:
            self._intervals = new_intervals
        else:
            self._intervals = self._intervals & new_intervals
        return self._intervals

    def get_state(self) -> dict:
        """Return the current state (for debugging).

        Returns
        -------
        dict
            Mapping with keys ``"intervals"`` and ``"z"``.
        """
        return {"intervals": self._intervals, "z": self._z}
