"""Entry points computing selective p-values.

:func:`inference` tests a linear statistic ``eta^T y`` (truncated normal);
:func:`inference_chi` tests a subspace norm ``||P y||`` (truncated chi).
Both re-run the user's algorithm along the selection line ``y(z) = a + b*z``
via a parametric search driven by sicore, assembling the truncation region
from the intervals recorded by the tracked operations.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from sicore import RealSubset, SelectiveInferenceChi, SelectiveInferenceNorm

from .siarray import siarray
from .interval_tracker import IntervalTracker
from . import poly_memo


class NoHypothesisError(Exception):
    """Raised when no hypothesis to test could be derived from the selection event.

    Raised by :func:`inference` when no ``z`` yields a selection matching the
    observed model and the test statistic becomes NaN.
    """


class InferenceProblem:
    """State and sicore-facing callbacks for a single selective inference session.

    sicore searches over many ``z`` along the selection line ``data(z) = a + b*z``,
    calling :meth:`forward_si` at each ``z``. ``forward_si`` re-runs the selection
    algorithm over ``z`` and returns its output and truncation region.
    :meth:`model_selector` checks whether the model selected during the search
    matches the observed one.

    Notes
    -----
    This state was previously attached to the ``eta`` siarray as dynamic attributes;
    it was extracted into a dedicated class to separate the data structure from the
    inference session.
    """

    def __init__(
        self,
        prob_vec: siarray,
        observed_model: siarray,
        algorithm: Callable[[siarray], siarray],
    ) -> None:
        self.prob_vec = prob_vec
        self.observed_model = observed_model
        self.algorithm = algorithm

    def forward_si(
        self, a: np.ndarray, b: np.ndarray, z: float
    ) -> tuple[np.ndarray, RealSubset]:
        """Re-run the algorithm at ``a + b*z`` and return (output, truncation region).

        Parameters
        ----------
        a, b : np.ndarray
            Offset and direction of the selection line.
        z : float
            Position along the selection line.

        Returns
        -------
        tuple of (np.ndarray, RealSubset)
            The algorithm output at ``z`` and the accumulated truncation region.
        """
        tracker = IntervalTracker.get_instance()
        tracker.reset()
        tracker.set_z(z)

        # Start the algorithm from y(z) = a + b*z (a degree-1 rational function of z)
        prob_vec_rf = siarray.from_rational(
            numerator=[a, b],
            denominator=[np.ones_like(a)],
        )
        model = self.algorithm(prob_vec_rf)

        intervals = tracker.get()
        output = model.data if isinstance(model, siarray) else model
        return output, intervals

    def model_selector(self, searched_model: np.ndarray | siarray) -> bool:
        """Return whether the searched model matches the observed model.

        Parameters
        ----------
        searched_model : np.ndarray or siarray
            Model selected during the parametric search.

        Returns
        -------
        bool
            True if the models match (``np.allclose``).
        """
        observed = (
            self.observed_model.data
            if isinstance(self.observed_model, siarray)
            else np.asarray(self.observed_model)
        )
        searched = (
            searched_model.data
            if isinstance(searched_model, siarray)
            else np.asarray(searched_model)
        )
        return np.allclose(observed, searched)


def _run_inference(
    si_calculator,
    prob_vec: siarray,
    observed_model: siarray | np.ndarray,
    algorithm: Callable[[siarray], siarray],
    memoize: bool,
    **kwargs,
):
    """Drive the parametric search for a pre-built sicore calculator.

    This engine is shared by all statistic types (Norm, Chi, ...); only the
    construction of ``si_calculator`` differs between them. The selection-line
    search, p-value computation, and NaN handling live here.

    Parameters
    ----------
    si_calculator : sicore SelectiveInference instance
        A ``SelectiveInferenceNorm``/``SelectiveInferenceChi``/... object.
    prob_vec : siarray
        Probability vector (``y``).
    observed_model : siarray or np.ndarray
        Observed model that the parametric search must match.
    algorithm : Callable[[siarray], siarray]
        Selection algorithm mapping ``y`` to its output.
    memoize : bool
        Whether to enable interval-computation memoization.
    **kwargs
        Forwarded to sicore's ``inference``.

    Returns
    -------
    sicore inference result
        Object exposing ``p_value`` and related quantities.

    Raises
    ------
    NoHypothesisError
        If no hypothesis is obtained (the test statistic is NaN).
    """
    # Toggle interval-computation memoization. Because a, b change for each p-value,
    # always reset at the start so no stale cache carries over.
    poly_memo.set_enabled(memoize)
    poly_memo.reset()

    problem = InferenceProblem(prob_vec, observed_model, algorithm)

    result = si_calculator.inference(
        algorithm=problem.forward_si,  # (a, b, z) -> (output, truncation region)
        model_selector=problem.model_selector,
        **kwargs,
    )

    if np.isnan(si_calculator.stat):
        raise NoHypothesisError("No hypothesis is obtained.")

    return result


def inference(
    eta: siarray,
    prob_vec: siarray,
    var: float | np.ndarray,
    algorithm: Callable[[siarray], siarray],
    model: siarray | None = None,
    memoize: bool = True,
    **kwargs,
):
    """Compute a selective p-value for a linear statistic ``eta^T y`` (Normal).

    Parameters
    ----------
    eta : siarray
        Contrast vector.
    prob_vec : siarray
        Probability vector (``y``).
    var : float or np.ndarray
        Variance or covariance matrix of ``prob_vec``.
    algorithm : Callable[[siarray], siarray]
        Selection algorithm mapping ``y`` to ``eta``.
    model : siarray, optional
        Observed model. When ``None``, ``eta`` is treated as the observed model.
    memoize : bool, optional
        Whether to enable interval-computation memoization (default True).
    **kwargs
        Forwarded to sicore's ``inference``.

    Returns
    -------
    sicore inference result
        Object exposing ``p_value`` and related quantities.

    Raises
    ------
    NoHypothesisError
        If no hypothesis is obtained (the test statistic is NaN).
    """
    # If no model is given, treat eta itself as the observed model
    observed_model = eta if model is None else model

    si_calculator = SelectiveInferenceNorm(
        prob_vec.values().ravel(), var, eta.values().ravel()
    )

    return _run_inference(
        si_calculator, prob_vec, observed_model, algorithm, memoize, **kwargs
    )


def inference_chi(
    projection: siarray | np.ndarray,
    prob_vec: siarray,
    var: float,
    algorithm: Callable[[siarray], siarray],
    model: siarray | np.ndarray,
    memoize: bool = True,
    **kwargs,
):
    """Compute a selective p-value for a subspace norm ``||P y||`` (Chi).

    Unlike :func:`inference`, the test targets the norm of ``y`` projected onto a
    subspace (a chi statistic) rather than a single linear contrast. There is no
    ``eta`` to fall back on, so the observed ``model`` must be given explicitly.

    Parameters
    ----------
    projection : siarray or np.ndarray
        Projection matrix ``P`` defining the tested subspace.
    prob_vec : siarray
        Probability vector (``y``).
    var : float
        Variance of ``prob_vec``.
    algorithm : Callable[[siarray], siarray]
        Selection algorithm mapping ``y`` to its output.
    model : siarray or np.ndarray
        Observed model that the parametric search must match.
    memoize : bool, optional
        Whether to enable interval-computation memoization (default True).
    **kwargs
        Forwarded to sicore's ``inference``.

    Returns
    -------
    sicore inference result
        Object exposing ``p_value`` and related quantities.

    Raises
    ------
    NoHypothesisError
        If no hypothesis is obtained (the test statistic is NaN).
    """
    projection_mat = (
        projection.values() if isinstance(projection, siarray) else np.asarray(projection)
    )

    si_calculator = SelectiveInferenceChi(
        prob_vec.values().ravel(), var, projection_mat
    )

    return _run_inference(
        si_calculator, prob_vec, model, algorithm, memoize, **kwargs
    )
