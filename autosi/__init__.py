"""AutoSI: automatic selective inference for algorithms written as array code.

Write a selection algorithm with the NumPy-like operations of this package
(:class:`siarray` and the functions in :mod:`autosi.functions`), then call
:func:`inference` to obtain a selective p-value. The selection event is
derived automatically by tracking every comparison and selection the
algorithm performs; no manual derivation is required.

Typical usage::

    import autosi as asi

    y = asi.array(y)          # wrap the observed data
    X = asi.array(X)

    def algorithm(y):
        ...                   # ordinary NumPy-like code using asi operations
        return eta

    eta = algorithm(y)
    result = asi.inference(eta=eta, prob_vec=y, var=1.0, algorithm=algorithm)
    print(result.p_value)
"""

from .inf import inference, inference_chi, NoHypothesisError
from .poly_memo import set_enabled as set_memoization, reset as reset_memoization
from .interval_tracker import IntervalTracker
from .siarray import siarray
from .functions import (
    abs,
    argmax,
    argmin,
    argsort,
    array,
    concatenate,
    flatten,
    max,
    mean,
    min,
    prod,
    sort,
    stack,
    sum,
    var,
)

__all__ = [
    "inference",
    "inference_chi",
    "NoHypothesisError",
    "set_memoization",
    "reset_memoization",
    "siarray",
    "argmax",
    "argmin",
    "argsort",
    "sort",
    "stack",
    "concatenate",
    "flatten",
    "mean",
    "sum",
    "prod",
    "var",
    "abs",
    "array",
    "max",
    "min",
    "IntervalTracker",
]
