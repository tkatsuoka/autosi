"""Backward-compatibility re-export shim.

The public API is exposed directly from the submodules by ``autosi/__init__.py``.
This module exists only to keep the legacy ``from autosi.utils import ...`` import
path working. New code should use ``import autosi as si``.
"""

from .interval_tracker import IntervalTracker
from .siarray import siarray
from .functions import (  # noqa: F401
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
    "IntervalTracker",
    "siarray",
    "abs",
    "argmax",
    "argmin",
    "argsort",
    "array",
    "concatenate",
    "flatten",
    "max",
    "mean",
    "min",
    "prod",
    "sort",
    "stack",
    "sum",
    "var",
]
